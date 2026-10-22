"""Configuration transactions for the agent loop.

The prepared-record types and their validation/guard layer, the
launch-reconfiguration protocol (prepare/publish/rollback/abort/finalize), the
reload preparation mechanics, and the reload and policy replacement
transactions live here as module-level functions. ``AgentLoop`` keeps thin
same-name delegate methods, so every existing private-method caller is
unchanged. The module never imports ``_loop`` at runtime: moved code reaches
the loop through the instance passed in, which keeps live-state reads late-bound
and the module-level ``_loop`` patch seams effective.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chartreux.core.agent_loop.backend_lifetime import BackendPublish
from chartreux.core.agent_loop.errors import AgentLoopStateError
from chartreux.core.agents.launch import (
    LaunchCandidate,
    _child_personal_posture,
    _validate_tools,
)
from chartreux.core.agents.manager import AgentManager
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.dispatch.renderer import task_description_for_config
from chartreux.core.dispatch.route_models import ReplacementCapability
from chartreux.core.dispatch.session import pending_policy_changes
from chartreux.core.hooks.manager import HooksManager
from chartreux.core.hooks.models import HookConfigResult
from chartreux.core.llm.types import BackendLike
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.recovery_models import PackageTier
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.skills.compatibility import ConversationPolicyFacts
from chartreux.core.skills.manager import SkillManager
from chartreux.core.subagents import LaunchConfig
from chartreux.core.tools import secret_redaction
from chartreux.core.tools.manager import ToolManager
from chartreux.core.workspace import Workspace
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.core.agent_loop import AgentLoop
    from chartreux.core.policy.bundle import ActivatedPolicy
    from chartreux.core.tools.mcp.registry import MCPRegistry


# Prepared records.


class _SwappableConfigSource:
    """Config getter for reload-prepared managers.

    Points at the target agent's config while preparation runs off-loop, then is
    repointed to the live config inside the synchronous commit. The prepared
    managers are not shared with the running turn until commit, so the running
    turn only ever observes the live getter.
    """

    def __init__(self, getter: Callable[[], ChartreuxConfigSchema]) -> None:
        self._getter = getter
        self.live = False

    def get(self) -> ChartreuxConfigSchema:
        return self._getter()

    def point_to(self, getter: Callable[[], ChartreuxConfigSchema]) -> None:
        self._getter = getter
        self.live = True


@dataclass(frozen=True, slots=True)
class _PreparedPolicyReplacement:
    owner: AgentLoop
    session_generation: int
    expected_token: object
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]
    previous_manager: ToolManager
    tool_manager: ToolManager
    inherited_restrictions: tuple[SourceRestrictions, ...]
    inherited_workspace: Workspace | None


@dataclass(frozen=True, slots=True)
class _PreparedReload:
    config: ChartreuxConfigSchema
    backend: BackendLike
    agent_manager: AgentManager
    tool_manager: ToolManager
    skill_manager: SkillManager
    system_prompt: str
    config_source: _SwappableConfigSource
    hook_config_result: HookConfigResult | None
    skills_adopted: int


@dataclass(frozen=True, slots=True)
class _PreparedLaunchReconfiguration:
    candidate: LaunchCandidate
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]
    consumers: _PreparedReload
    previous_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]
    previous_agent_manager: AgentManager
    previous_tool_manager: ToolManager
    previous_skill_manager: SkillManager
    previous_system_message: LLMMessage | None
    previous_instruction_read_files: frozenset[Path]
    previous_prices: tuple[float, float, float | None]
    previous_launch_overrides: LaunchConfig | None
    previous_launch_purpose: str | None
    previous_launch_package_tier: PackageTier | None
    previous_launch_replacement_capability: ReplacementCapability | None
    previous_committed_model: CommittedModelIdentity | None
    session_generation: int
    parent_authority_revision: int | None
    owns_backend: bool


# Validation and guard layer.


def _job_authority_config(config: ChartreuxConfigSchema) -> dict[str, Any]:
    data = config.model_dump(mode="python")
    data["tools"] = {
        # Custom Bash configs may add restriction-bearing fields. Preserve
        # the whole canonical configuration rather than an incomplete allowlist.
        name: settings
        for name, settings in data.get("tools", {}).items()
        if name in {"bash", "bash_start"}
    }
    data["credential_names"] = secret_redaction.ScrubPolicy.from_config(config)
    return copy.deepcopy({
        key: data.get(key)
        for key in (
            "enabled_tools",
            "disabled_tools",
            "tools",
            "tool_paths",
            "authorized_roots_by_project",
            "credential_env_passthrough",
            "credential_names",
        )
    })


def _guard_job_authority_config(loop: AgentLoop, config: ChartreuxConfigSchema) -> None:
    if loop._job_authority_config(config) != loop._execution_authority_config:
        loop.require_background_jobs_idle()


def _require_policy_idle(loop: AgentLoop) -> None:
    if (
        loop._active_turn is not None
        or loop._tool_event_queue is not None
        or loop._holders
    ):
        raise AgentLoopStateError("Policy replacement requires an idle loop")


def _guard_job_policy(
    loop: AgentLoop,
    staged: ConfigOrchestrator[ChartreuxConfigSchema],
    inherited: tuple[SourceRestrictions, ...],
    workspace: Workspace | None,
) -> None:
    loop._guard_job_authority_config(staged.config)

    def shell_sources(sources: tuple[SourceRestrictions, ...]) -> list[object]:
        return [
            (
                source.identity,
                tuple(
                    tool
                    for tool in source.tools
                    if tool.tool_name in {"bash", "bash_start"}
                ),
                source.authorized_roots,
            )
            for source in sources
        ]

    if (
        shell_sources(
            loop.config_orchestrator.restrictions + loop._inherited_restrictions
        )
        != shell_sources(staged.restrictions + inherited)
        or workspace != loop._inherited_workspace
    ):
        loop.require_background_jobs_idle()


def _validate_policy_replacement(
    loop: AgentLoop, prepared: _PreparedPolicyReplacement
) -> None:
    """Validate every participant before the tree performs any publication."""
    if (
        prepared.owner is not loop
        or prepared.session_generation != loop._session_generation
        or prepared.previous_manager is not loop.tool_manager
    ):
        raise ValueError("Stale policy runtime preparation")
    loop._require_policy_idle()
    loop._guard_job_policy(
        prepared.orchestrator,
        prepared.inherited_restrictions,
        prepared.inherited_workspace,
    )
    loop.config_orchestrator._check_policy_commit(
        prepared.expected_token, prepared.orchestrator
    )


# Launch reconfiguration protocol (prepare/publish/rollback/abort/finalize).


async def prepare_launch_reconfiguration(
    loop: AgentLoop,
    candidate: LaunchCandidate,
    *,
    expected_session_generation: int,
    expected_parent_authority_revision: int,
) -> _PreparedLaunchReconfiguration:
    """Stage only changed retained-child consumers without publishing them."""
    await loop.wait_until_ready()
    loop._require_policy_idle()
    if expected_session_generation != loop._session_generation:
        raise AgentLoopStateError("Stale launch reconfiguration generation")
    parent_revision = (
        loop._parent_authority_revision_getter()
        if loop._parent_authority_revision_getter is not None
        else None
    )
    if parent_revision != expected_parent_authority_revision:
        raise AgentLoopStateError("Parent authority changed during reservation")
    orchestrator = candidate.orchestrator.copy()
    loop._guard_job_authority_config(orchestrator.config)
    reuse_backend = (
        loop._injected_backend is not None
        or orchestrator.config.get_active_provider()
        == loop.config.get_active_provider()
    )
    tool_fields = ("enabled_tools", "disabled_tools", "tools")
    previous_overrides = loop.launch_overrides or LaunchConfig()
    replace_tools = any(
        getattr(previous_overrides, field)
        != getattr(candidate.semantic_overrides, field)
        for field in tool_fields
    )
    preparation = asyncio.create_task(
        asyncio.to_thread(
            loop._prepare_launch_consumers,
            orchestrator.config,
            replace_tools=replace_tools,
            reuse_backend=reuse_backend,
        )
    )
    try:
        consumers = await asyncio.shield(preparation)
    except asyncio.CancelledError as cancellation:
        while not preparation.done():
            try:
                await asyncio.shield(preparation)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not preparation.cancelled():
            try:
                abandoned = preparation.result()
            except Exception:
                abandoned = None
            if abandoned is not None and not reuse_backend:
                with contextlib.suppress(Exception):
                    await abandoned.backend.__aexit__(None, None, None)
        raise cancellation
    system_message = (
        loop.messages[0].model_copy(deep=True)
        if loop.messages and loop.messages[0].role is Role.system
        else None
    )
    return _PreparedLaunchReconfiguration(
        candidate=candidate,
        orchestrator=orchestrator,
        consumers=consumers,
        previous_orchestrator=loop._config_orchestrator,
        previous_agent_manager=loop.agent_manager,
        previous_tool_manager=loop.tool_manager,
        previous_skill_manager=loop.skill_manager,
        previous_system_message=system_message,
        previous_instruction_read_files=loop.tool_manager._instruction_read_files,
        previous_prices=(
            loop.stats.input_price_per_million,
            loop.stats.output_price_per_million,
            loop.stats.cached_input_price_per_million,
        ),
        previous_launch_overrides=(
            loop.launch_overrides.model_copy(deep=True)
            if loop.launch_overrides is not None
            else None
        ),
        previous_launch_purpose=loop.launch_purpose,
        previous_launch_package_tier=loop.launch_package_tier,
        previous_launch_replacement_capability=loop.launch_replacement_capability,
        previous_committed_model=loop.committed_model,
        session_generation=expected_session_generation,
        parent_authority_revision=parent_revision,
        owns_backend=not reuse_backend,
    )


def _prepare_launch_consumers(
    loop: AgentLoop,
    target_config: ChartreuxConfigSchema,
    *,
    replace_tools: bool,
    reuse_backend: bool,
) -> _PreparedReload:
    """Build the narrow consumer set required by a retained re-task."""
    config_source = _SwappableConfigSource(lambda: target_config)
    tool_manager = loop.tool_manager
    if replace_tools:
        tool_manager = ToolManager(
            config_source.get,
            mcp_registry=loop.mcp_registry,
            defer_mcp=True,
            bound_identity=loop._bound_identity,
            discovery_source=loop.tool_manager,
            restriction_getter=lambda: loop.config_orchestrator.restrictions,
            accepted_token_getter=lambda: (
                loop.config_orchestrator.accepted_token if config_source.live else None
            ),
            inherited_restrictions=loop._inherited_restrictions,
            inherited_workspace=loop._inherited_workspace,
            inherited_plan_write_scopes=loop._inherited_plan_write_scopes,
            parent_authority_getter=loop._parent_authority_getter,
            parent_authority_revision_getter=loop._parent_authority_revision_getter,
            cwd=loop.cwd,
            harness_files=loop.harness_files,
            scratchpad_dir=loop.scratchpad_dir,
            # The rebuilt manager serves the conversation's FROZEN route
            # table — the same source launch admission resolves against —
            # never the retask target config's live routes (design §6, §10).
            task_description=task_description_for_config(
                target_config, routes=loop.frozen_purpose_routes
            ),
            session_root_grants=loop._session_root_grants,
            dangerous_command_authority_provider=loop._dangerous_command_grants,
        )
    system_prompt = loop._render_system_prompt(
        loop.skill_manager, target_config, tool_manager, loop.agent_manager
    )
    return _PreparedReload(
        config=target_config,
        backend=loop.backend if reuse_backend else loop.backend_factory(target_config),
        agent_manager=loop.agent_manager,
        tool_manager=tool_manager,
        skill_manager=loop.skill_manager,
        system_prompt=system_prompt,
        config_source=config_source,
        hook_config_result=loop._hook_config_result,
        skills_adopted=loop._skills_adopted,
    )


def _revalidate_launch_tool_ceilings(
    loop: AgentLoop, candidate: LaunchCandidate
) -> None:
    """Fail closed when the live parent authority no longer admits a retask.

    Resolve-time admission validated the candidate's tool overrides against
    the parent's delegation ceiling at that moment. This publish-time re-check
    re-derives BOTH ceilings from the LIVE authority chain — the parent's
    delegation set (never its personal inventory: personal posture and
    delegation are independent axes, design §8) and the child identity's
    personal posture — so a parent authority change that does not bump the
    loop's authority revision (an in-place config edit, a refresh) can never
    publish a beyond-ceiling child, and an unavailable ancestor aborts the
    retask instead of opening it. The launch module's own validators are
    reused verbatim so the publish-time rule cannot drift from admission.

    Only loops with a parent authority chain are re-checked: production
    retask/replacement targets are always child runtimes (depth limit one),
    and a parentless loop's staged candidate is admitted by its own resolve.
    """
    if loop._parent_authority_getter is None:
        return
    try:
        ceiling_manager = loop._parent_authority_getter()
    except Exception:
        ceiling_manager = None
    if ceiling_manager is None:
        raise AgentLoopStateError(
            "Parent authority unavailable during launch reconfiguration"
        )
    _validate_tools(
        candidate.semantic_overrides,
        candidate.authority_inputs.known_tool_names,
        frozenset(ceiling_manager.delegable_tools),
        _child_personal_posture(candidate.identity),
    )


def publish_launch_reconfiguration(
    loop: AgentLoop, prepared: _PreparedLaunchReconfiguration
) -> BackendPublish:
    """Synchronously publish reversible state immediately before turn start."""
    loop._require_policy_idle()
    parent_revision = (
        loop._parent_authority_revision_getter()
        if loop._parent_authority_revision_getter is not None
        else None
    )
    if (
        prepared.previous_orchestrator is not loop._config_orchestrator
        or prepared.session_generation != loop._session_generation
        or prepared.parent_authority_revision != parent_revision
    ):
        raise AgentLoopStateError("Stale launch reconfiguration")
    # Authority hardening across the lifecycle: re-derive the admission
    # ceilings from the live parent chain before any state becomes visible,
    # so a retask can never widen authority past the current delegation set.
    _revalidate_launch_tool_ceilings(loop, prepared.candidate)
    loop._guard_job_authority_config(prepared.orchestrator.config)
    backend_publish = loop._backend_lifetime.publish_reversible(
        prepared.consumers.backend
    )
    loop._config_orchestrator = prepared.orchestrator
    prepared.consumers.agent_manager._orchestrator = prepared.orchestrator
    loop.agent_manager = prepared.consumers.agent_manager
    loop.tool_manager = prepared.consumers.tool_manager
    loop.skill_manager = prepared.consumers.skill_manager
    prepared.consumers.config_source.point_to(lambda: loop.config)
    loop._publish_system_prompt(prepared.consumers.system_prompt)
    model = prepared.candidate.effective_model
    loop.stats.update_pricing(
        model.input_price, model.output_price, model.cached_input_price
    )
    loop.launch_overrides = prepared.candidate.semantic_overrides.model_copy(deep=True)
    # The retask's own launch metadata installs with the reconfiguration:
    # the purpose label is the retask's (never copied from the previous run),
    # and the committed tier/capability keep describing the package's true
    # tier for the recovery boundary.
    loop.launch_purpose = prepared.candidate.purpose
    loop.launch_package_tier = prepared.candidate.effective_tier
    loop.launch_replacement_capability = prepared.candidate.replacement_capability
    loop.committed_model = prepared.candidate.committed_model
    loop.config.attach_committed_model(loop.committed_model)
    return backend_publish


def rollback_launch_reconfiguration(
    loop: AgentLoop,
    prepared: _PreparedLaunchReconfiguration,
    backend_publish: BackendPublish,
) -> None:
    backend_publish.rollback()
    loop._config_orchestrator = prepared.previous_orchestrator
    prepared.previous_agent_manager._orchestrator = prepared.previous_orchestrator
    loop.agent_manager = prepared.previous_agent_manager
    loop.tool_manager = prepared.previous_tool_manager
    loop.skill_manager = prepared.previous_skill_manager
    loop.tool_manager.set_instruction_read_files(
        prepared.previous_instruction_read_files
    )
    loop._authority_revision += 1
    if prepared.previous_system_message is not None:
        loop.messages.update_system_prompt(
            str(prepared.previous_system_message.content or "")
        )
    loop.stats.update_pricing(*prepared.previous_prices)
    loop.launch_overrides = prepared.previous_launch_overrides
    loop.launch_purpose = prepared.previous_launch_purpose
    loop.launch_package_tier = prepared.previous_launch_package_tier
    loop.launch_replacement_capability = prepared.previous_launch_replacement_capability
    loop.committed_model = prepared.previous_committed_model
    loop.config.attach_committed_model(loop.committed_model)


async def abort_launch_reconfiguration(
    loop: AgentLoop, prepared: _PreparedLaunchReconfiguration
) -> None:
    if prepared.owns_backend:
        with contextlib.suppress(Exception):
            await prepared.consumers.backend.__aexit__(None, None, None)


def finalize_launch_reconfiguration(
    loop: AgentLoop,
    prepared: _PreparedLaunchReconfiguration,
    backend_publish: BackendPublish,
) -> None:
    loop._execution_authority_config = loop._job_authority_config(loop.config)
    backend_publish.finalize(whole_turn_active=True)
    if prepared.previous_tool_manager is not prepared.consumers.tool_manager:
        prepared.previous_tool_manager._retire_authority()


# Reload preparation mechanics.


def _prepare_reload(
    loop: AgentLoop,
    target_config: ChartreuxConfigSchema,
    reload_hooks: bool,
    *,
    mcp_registry: MCPRegistry | None = None,
) -> _PreparedReload:
    loop._guard_job_authority_config(target_config)
    # Preserve the retained deployment before constructing a replacement backend.
    # An unchanged selection expression is provenance, not a request to resolve a
    # different deployment for the same base model.
    if (
        loop.committed_model is not None
        and target_config.active_model == loop._committed_selection
    ):
        from chartreux.core.model_catalog.resolver import resolver_for

        resolved = resolver_for(target_config).resolve_committed(
            loop.committed_model, allowed_models=target_config.allowed_models
        )
        target_config.attach_committed_model(
            resolved.identity.model_copy(
                update={
                    "thinking": loop._thinking_for_committed_reload(
                        target_config, resolved.identity
                    )
                }
            )
        )
    # Load the configured utility prompt before any runtime mutation or
    # client IO (the full-skeleton prompt axis is retired with 16a).
    _ = target_config.compaction_prompt
    # The candidate trust policy is carried by the backend being prepared.
    return loop._prepare_reload_consumers(
        target_config, reload_hooks, mcp_registry=mcp_registry
    )


def _thinking_for_committed_reload(
    loop: AgentLoop, config: ChartreuxConfigSchema, identity: CommittedModelIdentity
) -> str:
    override = config.thinking_overrides.get(identity.base_model)
    if override is not None:
        return override
    if loop._is_subagent and identity.thinking is not None:
        return identity.thinking
    catalog = config.catalog_snapshot.catalog
    selection = config.active_model or "@main"
    if selection.startswith("@"):
        role = catalog.roles.get(selection[1:])
        if role is not None and role.model == identity.base_model:
            return role.thinking
    return catalog.models[identity.base_model].thinking


def _policy_facts_from_bundle(activated: ActivatedPolicy) -> ConversationPolicyFacts:
    """The frozen identity/workflow selections a bound bundle committed.

    The single frozen source every SkillManager rebuild derives its
    eligibility facts from: the bound bundle's manifest selections (design
    §3, §6) — never the pending saved selections a bound conversation
    reports as applying to new conversations only.
    """
    return ConversationPolicyFacts(
        identity=activated.manifest.identity.selected,
        workflow=activated.manifest.workflow.selected,
    )


def _frozen_policy_facts(loop: AgentLoop) -> ConversationPolicyFacts:
    """The conversation's frozen policy facts for a rebuilt SkillManager.

    Reload rebuilds the skill manager around the RELOADED config, but skill
    eligibility reads the conversation's FROZEN facts — the bound bundle's
    committed identity/workflow selections, the same source the loop
    reports pending changes against — never the pending saved selections
    the reload just loaded: a pending selection applies to new
    conversations only (design §3, §6). Before the first render binds the
    bundle, the loop's existing frozen facts are the source.
    """
    activated = loop._bound_policy
    if activated is not None:
        return _policy_facts_from_bundle(activated)
    return loop.skill_manager.policy_facts


def _prepare_reload_consumers(
    loop: AgentLoop,
    target_config: ChartreuxConfigSchema,
    reload_hooks: bool,
    *,
    mcp_registry: MCPRegistry | None = None,
    discovery_source: ToolManager | None = None,
    reuse_backend: bool = False,
    hooks_loader: Callable[[], HookConfigResult],
) -> _PreparedReload:
    skills_adopted = loop._skills_adopted
    config_source = _SwappableConfigSource(lambda: target_config)
    tool_manager = ToolManager(
        config_source.get,
        mcp_registry=mcp_registry if mcp_registry is not None else loop.mcp_registry,
        defer_mcp=discovery_source is not None,
        bound_identity=loop._bound_identity,
        discovery_source=discovery_source,
        restriction_getter=lambda: loop.config_orchestrator.restrictions,
        accepted_token_getter=lambda: (
            loop.config_orchestrator.accepted_token if config_source.live else None
        ),
        inherited_restrictions=loop._inherited_restrictions,
        inherited_workspace=loop._inherited_workspace,
        inherited_plan_write_scopes=loop._inherited_plan_write_scopes,
        parent_authority_getter=loop._parent_authority_getter,
        parent_authority_revision_getter=loop._parent_authority_revision_getter,
        cwd=loop.cwd,
        harness_files=loop.harness_files,
        scratchpad_dir=loop.scratchpad_dir,
        # The task-tool description is conversation-owned policy like the
        # prompt: it renders from the conversation's FROZEN route table —
        # the same source launch admission resolves against — never the
        # reloaded live config, so a saved route edit stays pending instead
        # of half-applying to the model-facing table (design §6, §10).
        task_description=task_description_for_config(
            target_config, routes=loop.frozen_purpose_routes
        ),
        session_root_grants=loop._session_root_grants,
        dangerous_command_authority_provider=loop._dangerous_command_grants,
    )
    skill_manager = SkillManager(
        config_source.get,
        harness_files=loop.harness_files,
        policy_facts=_frozen_policy_facts(loop),
    )
    candidate_orchestrator = loop._config_orchestrator.copy(config=target_config)
    agent_manager = AgentManager(
        candidate_orchestrator, harness_files=loop.harness_files
    )
    system_prompt = loop._render_system_prompt(
        skill_manager, target_config, tool_manager, agent_manager
    )
    hook_config_result = hooks_loader() if reload_hooks else loop._hook_config_result
    return _PreparedReload(
        config=target_config,
        backend=loop.backend if reuse_backend else loop.backend_factory(target_config),
        agent_manager=agent_manager,
        tool_manager=tool_manager,
        skill_manager=skill_manager,
        system_prompt=system_prompt,
        config_source=config_source,
        hook_config_result=hook_config_result,
        skills_adopted=skills_adopted,
    )


# Reload transaction.


async def refresh_config(loop: AgentLoop) -> None:
    async def preflight(config: ChartreuxConfigSchema) -> None:
        loop._guard_job_authority_config(config)

    await loop._config_orchestrator.reload(preflight=preflight)
    loop._execution_authority_config = loop._job_authority_config(loop.config)
    loop.scrub_policy = secret_redaction.ScrubPolicy.from_config(loop.config)
    secret_redaction.register_session_policy(loop, loop.scrub_policy)
    secret_redaction.reset_cache()
    loop._retire_mcp_pool()
    loop._ensure_remote_registries()
    if loop.mcp_registry is not None:
        loop.mcp_registry.sync_active_servers(loop.config.mcp_servers)


async def reload_with_initial_messages(
    loop: AgentLoop,
    max_turns: int | None = None,
    max_price: float | None = None,
    reset_middleware: bool = True,
    reload_hooks: bool = False,
    reload_config: bool = False,
    session_save_config: ChartreuxConfigSchema | None = None,
) -> None:
    loop._reload_generation += 1
    generation = loop._reload_generation

    # Session switching may already have installed the target configuration
    # while the messages and logger still belong to the departing session.
    await loop.session_logger.save_interaction(
        loop.messages,
        loop.stats,
        session_save_config if session_save_config is not None else loop.config,
        loop.tool_manager,
    )

    # A newer reload superseded us while we were saving; don't mutate state.
    if generation != loop._reload_generation:
        return

    if max_turns is not None:
        loop._max_turns = max_turns
    if max_price is not None:
        loop._max_price = max_price
    loop._ensure_remote_registries()

    if reload_config:
        # The orchestrator owns source reads and acceptance. Preparing runtime
        # consumers must finish before its paired snapshot becomes visible.
        prepared_reload: _PreparedReload | None = None

        async def prepare(candidate: ChartreuxConfigSchema) -> None:
            nonlocal prepared_reload
            prepared_reload = await asyncio.to_thread(
                loop._prepare_reload, candidate, reload_hooks
            )
            if generation != loop._reload_generation:
                raise AgentLoopStateError("Configuration reload was superseded")
            loop._commit_reload(prepared_reload, reset_middleware)

        await loop._config_orchestrator.reload(preflight=prepare)
        return

    target_config = loop.config

    # Off-loop: skill discovery and system prompt I/O. reload() is awaited within a
    # turn, so that turn is suspended here -- nothing mutates the shared state this
    # reads, and the new objects are built locally before the commit below.
    prepared = await asyncio.to_thread(
        loop._prepare_reload, target_config, reload_hooks
    )

    # A newer reload superseded us; let it own the commit.
    if generation != loop._reload_generation:
        return

    # Synchronous swap: no await, so an in-flight turn can't observe a partial
    # update. Keep it that way -- don't make it async or move it off-thread.
    loop._commit_reload(prepared, reset_middleware)


def _commit_reload(  # noqa: PLR0915
    loop: AgentLoop, prepared: _PreparedReload, reset_middleware: bool
) -> None:
    loop._guard_job_authority_config(prepared.config)
    # Resolve identity before publishing any candidate runtime state. A changed
    # selection is an explicit reconfiguration; an unchanged expression is only
    # provenance and must retain the already committed deployment.
    from chartreux.core.model_catalog.resolver import resolver_for

    resolver = resolver_for(prepared.config)
    identity: CommittedModelIdentity | None
    if (
        loop.committed_model is not None
        and prepared.config.active_model == loop._committed_selection
    ):
        identity = resolver.resolve_committed(
            loop.committed_model, allowed_models=prepared.config.allowed_models
        ).identity.model_copy(
            update={
                "thinking": loop._thinking_for_committed_reload(
                    prepared.config, loop.committed_model
                )
            }
        )
    elif (
        loop.committed_model is None
        and loop._model_choice_pending()
        and not prepared.config.active_model
    ):
        # Committed-model recovery: an unrelated config write must not
        # silently commit the default. Stay unpinned; only an explicit
        # active_model selection (the branch below) opens the gate.
        identity = None
    else:
        identity = resolver.resolve(
            prepared.config.active_model or "@main",
            allowed_models=prepared.config.allowed_models,
        ).identity.model_copy(
            update={"thinking": prepared.config.get_active_model().thinking}
        )
    prepared.config.attach_committed_model(identity)
    # Finish fallible rendering and hook construction before retiring authority
    # or replacing any runtime object. A failed candidate leaves the old loop live.
    adopt_skills = prepared.skills_adopted == loop._skills_adopted
    system_prompt = (
        prepared.system_prompt
        if adopt_skills
        else loop._render_system_prompt(
            loop.skill_manager,
            prepared.config,
            prepared.tool_manager,
            prepared.agent_manager,
        )
    )
    hooks_manager = (
        HooksManager(prepared.hook_config_result.hooks, cwd=loop.cwd)
        if prepared.hook_config_result is not None
        else None
    )
    middleware = (
        loop._build_middleware_pipeline(prepared.config)
        if reset_middleware
        else loop.middleware_pipeline
    )
    # The prepared backend owns its candidate trust policy; publishing it
    # requires no process-global TLS mutation.
    # Now that the identity is live, the prepared managers should track the
    # live config so later refreshes (refresh_config) propagate to them.
    prepared.config_source.point_to(lambda: loop.config)
    prepared.agent_manager._orchestrator = loop._config_orchestrator
    previous_identity = loop.committed_model
    previous_selection = loop._committed_selection
    publication = loop._backend_lifetime.publish_reversible(prepared.backend)
    try:
        loop.committed_model = identity
        loop._committed_selection = prepared.config.active_model
        loop.config.attach_committed_model(loop.committed_model)
        loop.install_launch_metadata()
    except BaseException:
        publication.rollback()
        loop.committed_model = previous_identity
        loop._committed_selection = previous_selection
        loop.config.attach_committed_model(previous_identity)
        with contextlib.suppress(Exception):
            loop.install_launch_metadata()
        raise
    publication.finalize(whole_turn_active=loop._active_turn is not None)
    loop._retire_mcp_pool()
    loop.agent_manager = prepared.agent_manager
    loop.tool_manager._retire_authority()
    loop.tool_manager = prepared.tool_manager
    loop._authority_revision += 1
    loop.scrub_policy = secret_redaction.ScrubPolicy.from_config(prepared.config)
    loop._execution_authority_config = loop._job_authority_config(prepared.config)
    secret_redaction.register_session_policy(loop, loop.scrub_policy)
    if prepared.config.mcp_servers:
        loop._mcp_pool = loop._create_mcp_pool()
    if adopt_skills:
        loop.skill_manager = prepared.skill_manager
    loop._publish_system_prompt(system_prompt)
    loop._hook_config_result = prepared.hook_config_result
    loop._hooks_manager = hooks_manager
    loop.hook_config_issues = (
        prepared.hook_config_result.issues
        if prepared.hook_config_result is not None
        else []
    )
    loop.hooks_count = (
        len(prepared.hook_config_result.hooks)
        if prepared.hook_config_result is not None
        else 0
    )

    if len(loop.messages) == 1:
        loop.stats.reset_context_state()

    try:
        active_model = prepared.config.get_active_model()
        loop.stats.update_pricing(
            active_model.input_price,
            active_model.output_price,
            active_model.cached_input_price,
        )
    except ValueError:
        pass

    loop.middleware_pipeline = middleware
    # The value cache remains process-wide; reload invalidates stored/keyring
    # values. Already inherited child environments require pool retirement.
    secret_redaction.reset_cache()
    # A reload never re-postures: the conversation's frozen policy bundle
    # stays bound (design §10). Report what the reloaded saved state WOULD
    # change for new conversations — [agent] selection diffs and
    # resource-content diffs against the frozen manifest — without applying
    # any of it (the live reread here is for REPORTING only, never for prompt
    # refresh).
    activated = loop._bound_policy
    if activated is not None:
        pending = pending_policy_changes(
            prepared.config, activated, harness_files=loop.harness_files
        )
        loop.pending_policy_changes = pending
        for note in pending:
            logger.warning("Pending policy change after reload: %s", note)


# Policy replacement transactions.


async def _prepare_policy_replacement(
    loop: AgentLoop, *, source: str, tools: dict[str, Any], expected_token: object
) -> _PreparedPolicyReplacement:
    """Prepare local authority under the server's tree reservation."""
    loop._require_policy_idle()
    previous_manager = loop.tool_manager
    session_generation = loop._session_generation
    staged = await loop.config_orchestrator._stage_policy_replacement(
        source=source, tools=tools, expected_token=expected_token
    )
    return loop._prepare_policy_manager(
        staged,
        expected_token,
        previous_manager,
        session_generation,
        loop._inherited_restrictions,
    )


async def _prepare_root_replacement(
    loop: AgentLoop, *, source: str, roots: dict[str, list[str]], expected_token: object
) -> _PreparedPolicyReplacement:
    loop._require_policy_idle()
    previous_manager = loop.tool_manager
    generation = loop._session_generation
    staged = await loop.config_orchestrator._prepare_root_replacement(
        source=source, roots=roots, expected_token=expected_token
    )
    return loop._prepare_policy_manager(
        staged,
        expected_token,
        previous_manager,
        generation,
        loop._inherited_restrictions,
    )


async def _prepare_policy_refresh(
    loop: AgentLoop,
    *,
    previous: SourceRestrictions,
    replacement: SourceRestrictions,
    expected_token: object,
    inherited_workspace: Workspace | None = None,
) -> _PreparedPolicyReplacement:
    loop._require_policy_idle()
    previous_manager = loop.tool_manager
    session_generation = loop._session_generation
    inherited = tuple(
        replacement if source.identity == previous.identity else source
        for source in loop._inherited_restrictions
    )
    if any(source.identity is None for source in inherited):
        raise ValueError("Policy source provenance unavailable")
    staged = await loop.config_orchestrator._stage_policy_refresh(
        previous=previous, replacement=replacement, expected_token=expected_token
    )
    return loop._prepare_policy_manager(
        staged,
        expected_token,
        previous_manager,
        session_generation,
        inherited,
        inherited_workspace=inherited_workspace,
    )


def _prepare_policy_manager(
    loop: AgentLoop,
    staged: ConfigOrchestrator[ChartreuxConfigSchema],
    expected_token: object,
    previous_manager: ToolManager,
    session_generation: int,
    inherited: tuple[SourceRestrictions, ...],
    *,
    inherited_workspace: Workspace | None = None,
) -> _PreparedPolicyReplacement:
    inherited_workspace = inherited_workspace or loop._inherited_workspace
    loop._guard_job_policy(staged, inherited, inherited_workspace)

    # During preparation use the candidate, after synchronous publication use
    # live authority. Stale preparations are rejected, never installed.
    def authority() -> ConfigOrchestrator[ChartreuxConfigSchema]:
        return (
            staged
            if loop.config_orchestrator.accepted_token is expected_token
            else loop.config_orchestrator
        )

    manager = ToolManager(
        lambda: authority().config,
        mcp_registry=loop.mcp_registry,
        defer_mcp=True,
        bound_identity=loop._bound_identity,
        discovery_source=previous_manager,
        restriction_getter=lambda: authority().restrictions,
        accepted_token_getter=lambda: authority().accepted_token,
        inherited_restrictions=inherited,
        inherited_workspace=inherited_workspace,
        inherited_plan_write_scopes=loop._inherited_plan_write_scopes,
        parent_authority_getter=loop._parent_authority_getter,
        parent_authority_revision_getter=loop._parent_authority_revision_getter,
        cwd=loop.cwd,
        harness_files=loop.harness_files,
        scratchpad_dir=loop.scratchpad_dir,
        # The replacement manager keeps serving the conversation's FROZEN
        # route table — the bound bundle's snapshot, the same source launch
        # admission resolves against — never the staged live config's
        # routes (design §6, §10).
        task_description=task_description_for_config(
            authority().config, routes=loop.frozen_purpose_routes
        ),
        session_root_grants=loop._session_root_grants,
        dangerous_command_authority_provider=loop._dangerous_command_grants,
    )
    manager.set_instruction_read_files(previous_manager._instruction_read_files)
    for name in manager.registered_tools:
        manager.get_tool_config(name)
    for name in previous_manager._instances.keys() & manager.available_tools.keys():
        manager.get(name)
    loop.config_orchestrator._check_policy_token(expected_token)
    return _PreparedPolicyReplacement(
        loop,
        session_generation,
        expected_token,
        staged,
        previous_manager,
        manager,
        inherited,
        inherited_workspace,
    )


def _commit_policy_replacement(
    loop: AgentLoop, prepared: _PreparedPolicyReplacement, *, retire: bool = True
) -> None:
    loop._validate_policy_replacement(prepared)
    loop.config_orchestrator._commit_policy_replacement(
        prepared.orchestrator, expected_token=prepared.expected_token
    )
    loop._inherited_restrictions = prepared.inherited_restrictions
    loop._execution_authority_config = loop._job_authority_config(loop.config)
    loop._inherited_workspace = prepared.inherited_workspace
    if retire:
        prepared.previous_manager._retire_authority()
    loop.tool_manager = prepared.tool_manager
    loop._authority_revision += 1
