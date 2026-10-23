from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
import contextlib
import copy
from dataclasses import dataclass, field, replace
from functools import wraps
import inspect
import os
from pathlib import Path
import shutil
from string import Template
import threading
import time
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from pydantic import BaseModel, JsonValue

from chartreux.core.agent_loop import _config_transactions
from chartreux.core.agent_loop._completion_coordinator import CompletionCoordinator
from chartreux.core.agent_loop._config_transactions import (
    _policy_facts_from_bundle,
    _PreparedLaunchReconfiguration,
    _PreparedPolicyReplacement,
    _PreparedReload,
    _SwappableConfigSource as _SwappableConfigSource,
)
from chartreux.core.agent_loop._context_injection import ContextInjector
from chartreux.core.agent_loop._dangerous_command_grants import (
    DangerousCommandGrantStore,
)
from chartreux.core.agent_loop._file_state_context import (
    FileStateContext,
    RevisionRegistryPort,
)
from chartreux.core.agent_loop._loop_types import (
    AgentRuntimePolicy,
    AgentTurnOptions,
    RootAccountingOwner,
    ToolDecision,
    ToolExecutionResponse,
)
from chartreux.core.agent_loop._outward_redaction import OutwardEventRedactor
from chartreux.core.agent_loop._request_broker import InteractionRequestBroker
from chartreux.core.agent_loop._root_grants import SessionRootGrants
from chartreux.core.agent_loop._session_transitions import (
    _PREPARE_SCRATCHPAD,
    SessionTransitions,
    _PrepareScratchpad,
)
from chartreux.core.agent_loop._title_cadence import TitleCadence
from chartreux.core.agent_loop._tool_batch_runner import (
    ToolBatchRunner,
    _ToolInvocation,
)
from chartreux.core.agent_loop._tool_decisions import (
    ToolDecisionCoordinator,
    _dangerous_command_denial as _dangerous_command_denial,
    _declined_dangerous_command_decision as _declined_dangerous_command_decision,
    _declined_root_decision as _declined_root_decision,
    _out_of_root_denial as _out_of_root_denial,
)
from chartreux.core.agent_loop._tool_executor import ToolExecutor
from chartreux.core.agent_loop._transcript_recorder import TranscriptRecorder
from chartreux.core.agent_loop._workspace_relocator import WorkspaceRelocator
from chartreux.core.agent_loop.backend_lifetime import BackendLifetime, BackendPublish
from chartreux.core.agent_loop.errors import (
    AgentLoopError,
    AgentLoopLLMResponseError as AgentLoopLLMResponseError,
    AgentLoopStateError,
    EmptyReplayBudgetStop,
    ImagesNotSupportedError,
)
from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    LLMGateway,
    TranscriptAppend,
    select_backend,
)
from chartreux.core.agent_loop.title_controller import (
    ScheduledTitle,
    TitleController,
    TitleEventSink,
    TitleGateInputs,
    TitleScheduleInputs,
)
from chartreux.core.agent_loop_hooks import AgentLoopHooksMixin
from chartreux.core.agents.identities import validate_identity_selection
from chartreux.core.agents.launch import LaunchCandidate
from chartreux.core.agents.manager import AgentManager
from chartreux.core.background_jobs import (
    BackgroundJobRegistry,
    BackgroundJobsPort,
    BashListArgs,
)
from chartreux.core.checkpoints import Checkpointer, CheckpointRecorder, FileStore
from chartreux.core.compaction import (
    CompactionFailedError as CompactionFailedError,
    CompactionManager,
)
from chartreux.core.compaction.context import select_model_context
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig
from chartreux.core.config._dangerous_command_persistence import (
    read_saved_dangerous_command_grants,
)
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config._root_persistence import read_saved_roots
from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    get_harness_files_manager,
)
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.dangerous_command_grants import DangerousCommandGrantPersistencePort
from chartreux.core.deferred_questions import DeferredQuestionStore
from chartreux.core.dispatch.renderer import task_description_for_config
from chartreux.core.dispatch.route_models import PurposeRoute, ReplacementCapability
from chartreux.core.dispatch.session import bound_identity_reference
from chartreux.core.errors import ContextTooLongError
from chartreux.core.events import (
    AssistantEvent,
    BackgroundWorkEvent,
    BaseEvent,
    CompactEndEvent,
    CompactStartEvent,
    ReasoningEvent,
    SessionTitleUpdatedEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    UserMessageEvent,
)
from chartreux.core.file_state import FileStateStore
from chartreux.core.hooks.config import load_hooks_from_fs
from chartreux.core.hooks.manager import HooksManager
from chartreux.core.hooks.models import HookConfigResult, HookEvent, HookSessionContext
from chartreux.core.llm.backend.factory import create_backend
from chartreux.core.llm.failures import RequestRetryBudget, classify
from chartreux.core.llm.format import APIToolFormatHandler, ResolvedToolCall
from chartreux.core.llm.types import BackendLike
from chartreux.core.llm.utility_completion import is_fast_utility_model
from chartreux.core.llm_models import (
    AvailableTool,
    ImageAttachment,
    LLMChunk,
    LLMMessage,
    ManualShellContext,
    PersistedToolResult,
    Role,
    StrToolChoice,
    ToolCall,
    posting_time,
)
from chartreux.core.message_list import MessageList
from chartreux.core.middleware import (
    AutoCompactMiddleware,
    ContextWarningMiddleware,
    ConversationContext,
    MiddlewareAction,
    MiddlewarePipeline,
    MiddlewareResult,
    PriceLimitMiddleware,
    ResetReason,
    RunBudgetMiddleware,
    TokenLimitMiddleware,
    TurnLimitMiddleware,
)
from chartreux.core.model_catalog.availability import EligibleDeployment
from chartreux.core.plan_session import PlanSession
from chartreux.core.policy.bundle import (
    ActivatedPolicy,
    BundleStore,
    LiveSections,
    PolicyBundle,
)
from chartreux.core.policy.composition import ExecutionContext
from chartreux.core.policy.manifest import routes_from_snapshot
from chartreux.core.prompts import UtilityPrompt
from chartreux.core.recovery_models import PackageTier
from chartreux.core.review import ReviewManager
from chartreux.core.rewind import RewindManager
from chartreux.core.root_grants import RootGrantPort
from chartreux.core.scratchpad import cleanup_scratchpad, init_scratchpad
from chartreux.core.session.session_id import generate_session_id
from chartreux.core.session.session_lease import SessionLease
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import (
    POLICY_STORAGE_VERSION,
    AgentStats,
    ChildSessionLink,
    CommittedModelIdentity,
    LaunchMetadataV2,
    LaunchPersonaV2,
    SessionMetadata,
)
from chartreux.core.skills.manager import SkillManager
from chartreux.core.subagents import (
    LaunchConfig,
    SubagentManagementPort,
    SubagentRunnerPort,
)
from chartreux.core.system_prompt import (
    ProjectContextProvider,
    SystemPrompt,
    _add_commit_signature,
    _get_available_skills_section,
    _get_model_catalog_section,
    _get_scratchpad_section,
    _get_tool_aware_os_system_prompt,
    _git_status_caches,
)
from chartreux.core.timing import CompletedTurnTiming, InvocationTiming
from chartreux.core.tools import secret_redaction
from chartreux.core.tools.base import BaseTool, InvokeContext, ToolPermission
from chartreux.core.tools.builtins.skill import build_skill_result
from chartreux.core.tools.io_port import ToolIOPort
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.utils import DEFAULT_SENSITIVE_PATTERNS
from chartreux.core.usage import UsagePurpose
from chartreux.core.utils import (
    VIBE_STOP_EVENT_TAG,
    RetryObserver,
    RetryReason,
    is_user_cancellation_event,
)
from chartreux.core.workspace import Workspace
from chartreux.observability.logging import logger
from chartreux.user_content import UserDisplayContent, UserResource
from chartreux.utils.cache_store import CacheStore, InMemoryCacheStore
from chartreux.utils.paths import is_dangerous_directory


def _is_git_executable_available() -> bool:
    executable = os.environ.get("GIT_PYTHON_GIT_EXECUTABLE")
    if not executable:
        return shutil.which("git") is not None

    path = Path(executable).expanduser()
    if path.is_absolute() or os.sep in executable:
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(executable) is not None


if TYPE_CHECKING:
    from chartreux.core.config.harness_files._harness_manager import InstructionDocument
    from chartreux.core.tools.mcp.pool import MCPConnectionPool
    from chartreux.core.tools.mcp.registry import MCPRegistry


class _QueueTitleEventSink(TitleEventSink):
    def __init__(self, queue: asyncio.Queue[BaseEvent]) -> None:
        self._queue = queue

    def publish(self, event: BaseEvent) -> None:
        self._queue.put_nowait(event)


@dataclass(frozen=True, slots=True)
class _ActiveTurn:
    """Collaborators lent to the loop for one turn; its presence means one is running."""

    turn_id: str = field(default_factory=lambda: str(uuid4()))
    subagent_runner: SubagentRunnerPort | None = None
    tool_io: ToolIOPort | None = None
    retry_sink: RetryObserver | None = None
    timing: InvocationTiming = field(default_factory=InvocationTiming)
    prose_ids: list[str] = field(default_factory=list)


_NO_TURN = _ActiveTurn()

# Test-only kill switch for harnesses that run the real CLI against a mock model.
_DISABLE_AUTO_TITLE_ENV_VAR = "CHARTREUX_TEST_DISABLE_AUTO_TITLE"


def requires_init(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that awaits deferred initialization before executing the method."""
    if inspect.isasyncgenfunction(fn):

        @wraps(fn)
        async def gen_wrapper(self: AgentLoop, *args: Any, **kwargs: Any) -> Any:
            with secret_redaction.bind_policy(self.scrub_policy):
                await self.wait_until_ready()
            agen = fn(self, *args, **kwargs)
            sent: Any = None
            try:
                while True:
                    with secret_redaction.bind_policy(self.scrub_policy):
                        try:
                            event = await agen.asend(sent)
                        except StopAsyncIteration:
                            return
                    sent = yield event
            finally:
                with secret_redaction.bind_policy(self.scrub_policy):
                    await agen.aclose()

        return gen_wrapper

    @wraps(fn)
    async def wrapper(self: AgentLoop, *args: Any, **kwargs: Any) -> Any:
        with secret_redaction.bind_policy(self.scrub_policy):
            await self.wait_until_ready()
            return await fn(self, *args, **kwargs)

    return wrapper


def _resolve_identity_revision(
    reference: str, harness_files: HarnessFilesManager
) -> tuple[str, str]:
    """Resolve one identity reference to its content revision (a pure read).

    Used for the launch envelope before the startup bind has run: the envelope
    records the resolved identity reference and its revision so resume never
    re-derives identity from the live profile (design §7).
    """
    from chartreux.core.policy.composition import (
        policy_resource_loader,
        user_resource_directories,
    )
    from chartreux.core.policy.manifest import resource_revision
    from chartreux.core.policy.models import ResourceKind

    loader = policy_resource_loader(user_resource_directories(harness_files))
    resource = loader.resolve(ResourceKind.IDENTITY, reference)
    if resource.content is None:
        raise ValueError(f"identity resource {reference!r} resolved without content")
    return reference, resource_revision(resource.content)


def _bundle_from_activated(activated: ActivatedPolicy) -> PolicyBundle:
    """Reconstruct the persisted bundle of an activated policy (exact)."""
    return PolicyBundle(manifest=activated.manifest, frozen=activated.frozen)


def _live_sections(  # noqa: PLR0914 - one faithful mirror of the composition
    activated: ActivatedPolicy,
    config: ChartreuxConfigSchema,
    skill_manager: SkillManager,
    *,
    scratchpad_dir: Path | None,
    cwd: Path,
    harness_files: HarnessFilesManager,
) -> LiveSections:
    """The refreshable live sections of the served artifact (design §10).

    Mirrors the live-section inputs of the package-6 composition
    (``chartreux.core.system_prompt._build_composition_inputs``): the model
    and skills catalogs, the environment block, the scratchpad, the project
    context, the commit signature, and the AGENTS.md layers. The frozen
    sections never enter here — they come from the bound bundle. The
    per-launch instruction overlay and the execution context are FROZEN, so
    the catalog suppression for bounded children reads them from the bundle.
    """
    bounded_child = (
        activated.manifest.execution_context == ExecutionContext.BOUNDED_CHILD.value
    )
    overlay = activated.frozen.instruction_overlay

    model_info = None
    if config.include_model_info and config.catalog_snapshot is not None:
        sections = [f"Your model name is: `{config.get_active_model().alias}`"]
        if not bounded_child and overlay is None:
            catalog_section = _get_model_catalog_section(config)
            if catalog_section:
                sections.append(catalog_section)
        model_info = "\n\n".join(sections)

    skills = None
    scratchpad = None
    environment = None
    if config.include_prompt_detail:
        environment = _get_tool_aware_os_system_prompt(None)
        skills_section = _get_available_skills_section(skill_manager)
        if skills_section:
            skills = skills_section
        if scratchpad_dir is not None:
            scratchpad = _get_scratchpad_section(scratchpad_dir)

    project_context = None
    agents_documents: tuple[InstructionDocument, ...] = ()
    if config.include_project_context:
        is_dangerous, reason = is_dangerous_directory(cwd)
        if is_dangerous:
            template = UtilityPrompt.DANGEROUS_DIRECTORY.read()
            context = Template(template).safe_substitute(
                reason=reason.lower(), abs_path=cwd.resolve()
            )
        else:
            context = ProjectContextProvider(
                config=config.project_context,
                root_path=cwd,
                git_status_cache=_git_status_caches.cache_for(harness_files),
            ).get_full_context()
        context_parts = [context]
        cwd_resolved = cwd.resolve()
        extra_roots = [
            root
            for root in harness_files.project_roots
            if root.resolve() != cwd_resolved
        ]
        if extra_roots:
            dirs_lines = "\n".join(f" - {d}" for d in extra_roots)
            context_parts.append(
                "Additional working directories (treated with the same "
                "file-access permissions as the primary working directory):\n"
                + dirs_lines
            )
        project_context = "\n\n".join(context_parts)
        agents_documents = tuple(harness_files.load_instruction_documents())

    commit_signature = (
        _add_commit_signature() if config.include_commit_signature else None
    )

    return LiveSections(
        model_info=model_info,
        skills=skills,
        scratchpad=scratchpad,
        environment=environment,
        project_context=project_context,
        commit_signature=commit_signature,
        agents_documents=agents_documents,
    )


class AgentLoop(AgentLoopHooksMixin):  # noqa: PLR0904
    def __init__(  # noqa: PLR0913, PLR0915
        self,
        config_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        *,
        max_turns: int | None = None,
        max_price: float | None = None,
        max_tokens: int | None = None,
        max_session_tokens: int | None = None,
        run_max_price: float | None = None,
        run_max_session_tokens: int | None = None,
        backend: BackendLike | None = None,
        enable_streaming: bool = False,
        launch_context: dict[str, object] | None = None,
        is_subagent: bool = False,
        defer_heavy_init: bool = False,
        headless: bool = False,
        hook_config_result: HookConfigResult | None = None,
        mcp_registry: MCPRegistry | None = None,
        cache_store: CacheStore | None = None,
        auto_title_enabled: bool = False,
        user_input_capability: bool = False,
        parent_session_id: str | None = None,
        cwd: Path | None = None,
        harness_files: HarnessFilesManager | None = None,
        session_id: str | None = None,
        session_dir: Path | None = None,
        session_lease: SessionLease | None = None,
        inherited_workspace: Workspace | None = None,
        inherited_restrictions: tuple[SourceRestrictions, ...] = (),
        inherited_mode_restrictions: tuple[SourceRestrictions, ...] = (),
        inherited_plan_write_scopes: tuple[tuple[Path, Path | None], ...] = (),
        inherited_root_grants: tuple[Path, ...] = (),
        parent_authority_getter: Callable[[], ToolManager] | None = None,
        parent_authority_revision_getter: Callable[[], int] | None = None,
        launch_identity: str | None = None,
        launch_overrides: LaunchConfig | None = None,
        launch_purpose: str | None = None,
        # The effective package tier and the model-level replacement-capability
        # marking the launch resolved (loop-owned launch metadata, like the
        # purpose label): the agent record reads the tier so the recovery
        # boundary decides on the package's true tier.
        launch_package_tier: PackageTier | None = None,
        launch_replacement_capability: ReplacementCapability | None = None,
        frozen_instructions: str | None = None,
        committed_model: CommittedModelIdentity | None = None,
        accounting_owner: RootAccountingOwner | None = None,
        background_jobs: BackgroundJobsPort | None = None,
        bound_policy: ActivatedPolicy | None = None,
        root_grant_port: RootGrantPort | None = None,
        dangerous_command_grant_port: DangerousCommandGrantPersistencePort
        | None = None,
        file_revision_registry: RevisionRegistryPort | None = None,
        parent_file_state: FileStateStore | None = None,
    ) -> None:
        self._background_job_registry = None if is_subagent else BackgroundJobRegistry()
        self.background_jobs = (
            self._background_job_registry.root_port()
            if self._background_job_registry is not None
            else background_jobs
        )
        self._job_lifetime_generation = 0
        self.accounting_owner = accounting_owner
        self._detached_accounting_producers: set[asyncio.Task[None]] = set()
        self._inherited_restrictions = tuple(inherited_restrictions)
        self._inherited_mode_restrictions = tuple(inherited_mode_restrictions)
        self._inherited_plan_write_scopes = tuple(inherited_plan_write_scopes)
        self._parent_authority_getter = parent_authority_getter
        self._parent_authority_revision_getter = parent_authority_revision_getter
        self._authority_revision = 0
        self.launch_identity = launch_identity
        self.launch_overrides = (
            launch_overrides.model_copy(deep=True) if launch_overrides else None
        )
        # The purpose label whose route this launch resolved through (None
        # for a direct launch): the loop-owned launch-purpose record the
        # projection reports, never re-derived from identity or model.
        self.launch_purpose = launch_purpose
        # The effective package tier and replacement capability the launch
        # resolved: loop-owned launch metadata the agent record reads, so
        # the failure-time recovery boundary decides on the package's true
        # tier/capability instead of a default.
        self.launch_package_tier = launch_package_tier
        self.launch_replacement_capability = launch_replacement_capability
        # The subagent run this loop is currently executing (set at run
        # admission, cleared at run end): the attribution join key to the
        # app-side run record. None for root and utility calls.
        self.run_identity: str | None = None
        self.frozen_instructions = frozen_instructions
        self.committed_model = committed_model
        self._committed_selection: str | None = None
        self.cwd = (cwd or Path.cwd()).resolve()
        # A child's own cwd/config cannot reconstruct a narrower parent ceiling.
        # Direct callers must carry explicit parent authority, just like the factory.
        if is_subagent and inherited_workspace is None:
            raise ValueError("Subagent construction requires inherited_workspace")
        self._inherited_workspace = inherited_workspace
        self._session_root_grants = SessionRootGrants()
        # Inherited grants seed this session's own store: they are launch-time
        # authority, frozen here, and never widened by later parent grants.
        for inherited_root in inherited_root_grants:
            self._session_root_grants.grant(inherited_root)
        self._root_grant_port = root_grant_port
        # Runtime-owned dangerous-command grants (session/once/refusal); the
        # store is the provider port every tool manager of this loop consumes.
        self._dangerous_command_grants = DangerousCommandGrantStore()
        self._dangerous_command_grant_port = dangerous_command_grant_port
        self.harness_files = replace(
            harness_files or get_harness_files_manager(), cwd=self.cwd
        )
        self._config_orchestrator = config_orchestrator
        self._auto_title_enabled = auto_title_enabled
        self._user_input_capability = user_input_capability
        self._headless = headless
        self._is_subagent = is_subagent
        # The conversation's identity binding (design §3, decision log item 3):
        # the selection is validated ONCE here — an identity that is neither a
        # builtin, nor builtin:-qualified-and-existing, nor a resolvable user
        # resource fails closed with an actionable error (S2 unknown-identity
        # fail-open) — and frozen for this conversation's lifetime. Every tool
        # manager this loop builds derives posture from the bound identity, so
        # a reload or in-place edit that changes [agent] identity is a pending
        # selection for new conversations and never re-postures an existing
        # manager or child.
        #
        # A loop constructed WITH a bound policy bundle (a resumed or
        # reconstructed conversation) takes its identity from the persisted
        # bundle instead: validation applies to NEW bindings only, and resume
        # consumes the persisted content even when the source resource files
        # are gone (S2 follow-up: loop-construction identity validation vs
        # persisted-bundle resume).
        self._bundle_store = BundleStore()
        self._policy_bind_lock = threading.Lock()
        self.pending_policy_changes: tuple[str, ...] = ()
        if bound_policy is not None:
            self._bound_identity = bound_identity_reference(bound_policy)
            self._bound_policy: ActivatedPolicy | None = bound_policy
        else:
            self._bound_identity = validate_identity_selection(
                config_orchestrator.config.agent.identity,
                user_prompts_dirs=self.harness_files.user_prompts_dirs,
            )
            # Bound once at startup: the first prompt render captures the
            # conversation's frozen policy bundle (design §6, §10).
            self._bound_policy = None
        self.cache_store = cache_store or InMemoryCacheStore()

        self._defer_heavy_init = defer_heavy_init
        self._deferred_init_thread: threading.Thread | None = None
        self._deferred_init_lock = threading.Lock()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._init_error: Exception | None = None
        self._init_start_time = time.monotonic()
        self._skills_adopted: int = 0
        self._plan_attrs_task: asyncio.Task[None] | None = None
        self._reload_generation: int = 0
        self._session_generation: int = 0
        # Bounded deferred-question records for unattended question expiry
        # (plan U/OD1): the app server records on expiry; compaction carries
        # the records across the structural boundary.
        self.deferred_questions = DeferredQuestionStore()
        self._init_duration_pending: bool = defer_heavy_init
        self._last_init_duration_ms: int | None = None
        # Title and managed-job state results land here. The app-server's one
        # out-of-band drain surfaces them immediately, including between turns.
        self._out_of_band_events: asyncio.Queue[BaseEvent] = asyncio.Queue()
        if self._background_job_registry is not None:
            self._background_job_registry.event_sink = (
                self._out_of_band_events.put_nowait
            )

        self.session_id = session_id or generate_session_id()
        self._session_lease = session_lease
        self.parent_session_id = parent_session_id
        self.scratchpad_dir = (
            init_scratchpad(self.session_id) if not is_subagent else None
        )

        self.agent_manager = AgentManager(
            self._config_orchestrator, harness_files=self.harness_files
        )
        # Configuration parsing is pure; runtime IO is explicit and must precede
        # construction of any clients, including the MCP pool. The configured
        # utility prompt is the surviving prompt pre-warm (the full-skeleton
        # prompt axis is retired with package 16a).
        _ = self.config.compaction_prompt
        self.mcp_registry: MCPRegistry | None = (
            mcp_registry
            if defer_heavy_init
            else mcp_registry or self._create_mcp_registry()
        )
        self.scrub_policy = secret_redaction.ScrubPolicy.from_config(self.config)
        secret_redaction.register_session_policy(self, self.scrub_policy)
        self._retired_mcp_pools: list[MCPConnectionPool] = []
        self._mcp_pool: MCPConnectionPool | None = (
            None if defer_heavy_init else self._create_mcp_pool()
        )
        self.config.require_active_provider_api_key()
        config = self.config
        self._committed_selection = config.active_model
        if self.committed_model is None and config.catalog_snapshot is not None:
            from chartreux.core.model_catalog.resolver import resolver_for

            selection = config.active_model or "@main"
            resolved = resolver_for(config).resolve(
                selection, allowed_models=config.allowed_models
            )
            self.committed_model = resolved.identity.model_copy(
                update={"thinking": config.get_active_model().thinking}
            )
        if self.committed_model is not None:
            config.attach_committed_model(self.committed_model)
        # The task-tool description is conversation-owned policy like the
        # prompt: a loop constructed WITH a bound bundle (a reconstructed
        # conversation) serves the bundle's frozen route table — the same
        # source launch admission resolves against — never the pending live
        # config; a fresh conversation IS the current selections, so its
        # manager reads the live config (the bundle binds from it at the
        # first render, design §6, §10).
        if self._bound_policy is not None:
            task_description = task_description_for_config(
                config,
                routes=routes_from_snapshot(self._bound_policy.manifest.route_snapshot),
            )
        else:
            task_description = task_description_for_config(config)
        self.tool_manager = ToolManager(
            lambda: self.config,
            mcp_registry=self.mcp_registry,
            defer_mcp=True,
            bound_identity=self._bound_identity,
            restriction_getter=lambda: self.config_orchestrator.restrictions,
            accepted_token_getter=lambda: self.config_orchestrator.accepted_token,
            inherited_restrictions=self._inherited_restrictions,
            inherited_workspace=self._inherited_workspace,
            inherited_plan_write_scopes=self._inherited_plan_write_scopes,
            parent_authority_getter=self._parent_authority_getter,
            parent_authority_revision_getter=self._parent_authority_revision_getter,
            cwd=self.cwd,
            harness_files=self.harness_files,
            scratchpad_dir=self.scratchpad_dir,
            task_description=task_description,
            session_root_grants=self._session_root_grants,
            dangerous_command_authority_provider=self._dangerous_command_grants,
        )
        # Skill eligibility reads the conversation's frozen selections: a
        # loop constructed WITH a bound policy bundle (a resumed or
        # reconstructed conversation) derives them from the bundle's
        # manifest — the same frozen source the reload path reads — never
        # the pending saved config; a fresh conversation IS the current
        # selections, so the manager reads them from the live config.
        self.skill_manager = SkillManager(
            lambda: self.config,
            harness_files=self.harness_files,
            policy_facts=(
                _policy_facts_from_bundle(self._bound_policy)
                if self._bound_policy is not None
                else None
            ),
        )
        self._execution_authority_config = self._job_authority_config(self.config)
        self._max_turns = max_turns
        self._max_price = max_price
        self._max_tokens = max_tokens
        self._max_session_tokens = max_session_tokens
        self._run_max_price = run_max_price
        self._run_max_session_tokens = run_max_session_tokens
        self._plan_session = PlanSession()
        self._user_plan: str | None = None

        self.format_handler = APIToolFormatHandler()
        self._llm_gateway = LLMGateway()

        self._injected_backend = backend
        self._backend_lifetime = BackendLifetime(self.backend_factory())
        self._completion_providers: list[tuple[str, ...]] = []
        self._failover_switches: list[dict[str, str]] = []

        self.enable_streaming = enable_streaming
        self.middleware_pipeline = MiddlewarePipeline()
        self._setup_middleware()

        self.messages = MessageList()
        # Working-set file state: the session-side view of the files this
        # worker has observed (path -> last-seen revision + outline). A child
        # is seeded with a deep copy of its parent's working set at spawn; the
        # registry is the dispatcher-level revision truth shared across the
        # session tree (None when this loop was constructed outside it).
        self.file_state_context = FileStateContext(
            store=(
                parent_file_state.model_copy(deep=True)
                if parent_file_state is not None
                else None
            ),
            registry=file_revision_registry,
        )
        # Only the active completion retains execution arguments; transcript
        # copies are sanitized before insertion and never used for execution.
        self._execution_message: LLMMessage | None = None

        self.stats = AgentStats()
        # Retains admission names and resolved values only until outward
        # emission; the redactor re-reads the live scrub policy per event.
        self._outward_redactor = OutwardEventRedactor(
            scrub_policy_getter=lambda: self.scrub_policy,
            record_tool_call_event=lambda event: self._recorded_tool_call_event(event),
        )
        # Tool-response ordering and recording: while a tool batch is active,
        # response messages are staged and appended in tool-call index order
        # even though their events keep streaming in completion order; the
        # recorder also owns sanitized call copies, response messages, and
        # committed completion appends, borrowing this loop's MessageList.
        self._transcript_recorder = TranscriptRecorder(
            messages=self.messages,
            scrub_policy_getter=lambda: self.scrub_policy,
            format_handler_getter=lambda: self.format_handler,
            set_execution_message=lambda message: self._set_execution_message(message),
            note_prose=lambda message: self._note_transcript_prose(message),
            invalidate_cursor=lambda: (
                self.session_logger.invalidate_transcript_cursor()
            ),
        )
        self._request_broker = InteractionRequestBroker()
        # Tool decisions: the approval workflows (dangerous-command and
        # out-of-root denial formatting, prompts, resume, persistence) and
        # the shared root-prompt asker lifecycle live on the coordinator.
        self._tool_decisions = ToolDecisionCoordinator(
            grants_store=self._dangerous_command_grants,
            request_broker=self._request_broker,
            user_input_capability_getter=lambda: self._user_input_capability,
            is_subagent_getter=lambda: self._is_subagent,
            grant_port_getter=lambda: self._dangerous_command_grant_port,
            tool_manager_getter=lambda: self.tool_manager,
            session_id_getter=lambda: self.session_id,
            config_orchestrator_getter=lambda: self.config_orchestrator,
            read_saved_grants=(
                lambda orchestrator, project: read_saved_dangerous_command_grants(
                    orchestrator, project=project
                )
            ),
            root_grants_store=self._session_root_grants,
            root_grant_port_getter=lambda: self._root_grant_port,
            cwd_getter=lambda: self.cwd,
            read_saved_roots=(
                lambda orchestrator, project: read_saved_roots(
                    orchestrator, project=project
                )
            ),
            apply_approved_root_grant=lambda root: self._apply_approved_root_grant(
                root
            ),
        )
        # Batch runner: the wait registry, the batch queue, the tool tasks
        # and their completion monitor, the batch teardown, and the
        # call-event preparation/ordering bracket live on the runner; the
        # loop keeps read-only views and compatibility delegates.
        self._tool_batch_runner = ToolBatchRunner(
            active_turn_id_getter=lambda: self._turn.turn_id,
            transcript_recorder=self._transcript_recorder,
            request_broker=self._request_broker,
            stats_getter=lambda: self.stats,
            tool_manager_getter=lambda: self.tool_manager,
            harness_files_getter=lambda: self.harness_files,
            scrub_policy_getter=lambda: self.scrub_policy,
            format_handler_getter=lambda: self.format_handler,
            record_tool_call_event=lambda event: self._recorded_tool_call_event(event),
            execute_tool=lambda tool_call: self._process_one_tool_call(tool_call),
            register_admissions=lambda tool_calls, admitted: (
                self._outward_redactor.register_batch_admissions(tool_calls, admitted)
            ),
            tool_failure_event=lambda tool_call, error_msg: self._tool_failure_event(
                tool_call, error_msg
            ),
            abandon_root_requests=lambda: self._tool_decisions.abandon_root_requests(),
        )
        # Context injection: the skill-loaded detection, the synthetic skill
        # expansion (its separate direct path), and the mentioned-file
        # expansion (through the single-tool execution callback) live on
        # the injector; the loop keeps one compatibility delegate for the
        # directly driven skill path and re-points its call sites.
        self._context_injector = ContextInjector(
            messages=self.messages,
            transcript_recorder=self._transcript_recorder,
            skill_manager_getter=lambda: self.skill_manager,
            tool_manager_getter=lambda: self.tool_manager,
            harness_files_getter=lambda: self.harness_files,
            cwd_getter=lambda: self.cwd,
            model_context_getter=lambda: self._current_model_context(),
            record_tool_call=lambda tool_call: self._recorded_tool_call(tool_call),
            record_tool_call_event=lambda event: self._recorded_tool_call_event(event),
            execute_tool=lambda tool_call: self._process_one_tool_call(tool_call),
            build_skill_result=lambda skill_info, already_loaded: build_skill_result(
                skill_info, already_loaded=already_loaded
            ),
            save_messages=lambda: self._save_messages(),
        )
        # Completion coordination: the input/header/resource assembly, the
        # candidate/attempt helpers, the synchronous publication/rollback
        # helpers, and the entire attempt generator live on the coordinator;
        # the loop keeps compatibility delegates for every rider-referenced
        # helper and for the generator itself (returning the coordinator's
        # iterator directly, no per-chunk forwarding layer).
        self._completion_coordinator = CompletionCoordinator(
            config_getter=lambda: self.config,
            config_orchestrator=self._config_orchestrator,
            backend_lifetime=self._backend_lifetime,
            llm_gateway=self._llm_gateway,
            committed_model_getter=lambda: self.committed_model,
            stats_getter=lambda: self.stats,
            format_handler_getter=lambda: self.format_handler,
            session_id_getter=lambda: self.session_id,
            parent_session_id_getter=lambda: self.parent_session_id,
            is_subagent_getter=lambda: self._is_subagent,
            launch_identity_getter=lambda: self.launch_identity,
            launch_purpose_getter=lambda: self.launch_purpose,
            run_identity_getter=lambda: self.run_identity,
            accounting_owner_getter=lambda: self.accounting_owner,
            accounting_sink_getter=lambda: getattr(self, "_accounting_sink", None),
            usage_attribution_getter=lambda: getattr(self, "_usage_attribution", None),
            injected_backend_getter=lambda: self._injected_backend,
            max_tokens_getter=lambda: self._max_tokens,
            notice_retry=self.notice_retry,
            admit_replay_getter=lambda: self._admit_empty_replay,
            create_backend=lambda **kwargs: create_backend(**kwargs),
            restore_committed_model=lambda identity: self._restore_committed_model(
                identity
            ),
            install_launch_metadata=lambda: self.install_launch_metadata(),
            backend_metadata=lambda call_type: self._build_backend_metadata(call_type),
            enumerate_candidates=lambda messages, attempted_model, call_type=None: (
                self._failover_candidates(
                    messages, attempted_model, call_type=call_type
                )
            ),
            prepare_inputs=lambda **kwargs: self._completion_inputs(**kwargs),
            bind_resources=lambda backend, budget=None: self._call_resources(
                backend, budget
            ),
            materialize_model=lambda resolved, call_type, thinking=None: (
                self._attempt_model(resolved, call_type=call_type, thinking=thinking)
            ),
            select_budget=lambda budget, has_alternative: self._attempt_retry_budget(
                budget, has_alternative=has_alternative
            ),
            backend_for_candidate=lambda model, budget: self._backend_for_attempt(
                model, budget
            ),
            rollback_publication=lambda publication, previous_identity, candidate: (
                self._rollback_attempt(publication, previous_identity, candidate)
            ),
            append_transcript=lambda outcome: self._append_transcript(outcome),
            record_providers=lambda providers: self._record_completion_providers(
                providers
            ),
            record_switch=lambda **kwargs: self._record_failover_switch(**kwargs),
            commit_identity=lambda identity: self._commit_attempt_identity(identity),
            attempt_stream=lambda **kwargs: self._attempt_completion(**kwargs),
        )
        # Session transitions: the todo/file-state replay participants, the
        # active-model pricing helper, the resume preparation (its synchronous
        # commit stays on the loop), the reset preparation with its lease
        # cleanup (its commit stays on the loop too), and the clear-history
        # orchestration live on the service; the loop retains identity,
        # generation, stats, the accounting owner, and lease publication.
        self._session_transitions = SessionTransitions(
            file_state_context=self.file_state_context,
            out_of_band_events=self._out_of_band_events,
            detached_accounting_producers=self._detached_accounting_producers,
            config_getter=lambda: self.config,
            agent_manager_getter=lambda: self.agent_manager,
            tool_manager_getter=lambda: self.tool_manager,
            skill_manager_getter=lambda: self.skill_manager,
            stats_getter=lambda: self.stats,
            is_subagent_getter=lambda: self._is_subagent,
            background_jobs_getter=lambda: self.background_jobs,
            job_lifetime_generation_getter=lambda: self._job_lifetime_generation,
            scratchpad_dir_getter=lambda: self.scratchpad_dir,
            accounting_owner_getter=lambda: self.accounting_owner,
            auto_title_task_getter=lambda: self._auto_title_task,
            require_background_jobs_idle=lambda: self.require_background_jobs_idle(),
            prepare_scratchpad=lambda session_id: self.prepare_scratchpad_for_session(
                session_id
            ),
            render_system_prompt=(
                lambda skill_manager, config, scratchpad_dir, activated_policy=None: (
                    self._render_system_prompt(
                        skill_manager,
                        config,
                        scratchpad_dir=scratchpad_dir,
                        activated_policy=activated_policy,
                    )
                )
            ),
            messages=self.messages,
            deferred_questions=self.deferred_questions,
            session_id_getter=lambda: self.session_id,
            parent_session_id_getter=lambda: self.parent_session_id,
            session_lease_getter=lambda: self._session_lease,
            bound_policy_getter=lambda: self._ensure_bound_policy(),
            bound_identity_getter=lambda: self._bound_identity,
            bundle_store_getter=lambda: self._bundle_store,
            cwd_getter=lambda: self.cwd,
            harness_files_getter=lambda: self.harness_files,
            execution_context_getter=lambda: self._execution_context(),
            session_logger_getter=lambda: self.session_logger,
            middleware_pipeline_getter=lambda: self.middleware_pipeline,
            reset_session=lambda keep_parent: self._reset_session(
                keep_parent=keep_parent
            ),
            publish_fresh_stats=lambda: self._publish_fresh_stats(),
            generate_session_id=lambda suffix: generate_session_id(suffix=suffix),
        )
        self._active_turn: _ActiveTurn | None = None
        self.completed_turn_timing: CompletedTurnTiming | None = None
        # Operations that are not turns but still hold the session: anything
        # reading the working directory or acting on its repository for longer
        # than an instant. They exclude each other through _take_session, so an
        # operation announces itself rather than every other one having to know
        # it exists. A turn will not start while one is held either, which is
        # what makes it safe for a holder to await.
        self._holders: list[str] = []
        # The workspace relocator owns the relocation transaction and the
        # trust a move acquired: the directory a move granted session trust
        # to, so a later move releases that and not a grant somebody else
        # made. The loop keeps the workspace roots themselves, the session
        # holder gate, and the consumer-rebuild facade.
        self._workspace_relocator = WorkspaceRelocator(
            cwd_getter=lambda: self.cwd,
            harness_files_getter=lambda: self.harness_files,
            agent_manager_getter=lambda: self.agent_manager,
            config_orchestrator_getter=lambda: self.config_orchestrator,
            session_logger_getter=lambda: self.session_logger,
            has_active_turn=lambda: self._active_turn is not None,
            require_background_jobs_idle=lambda: self.require_background_jobs_idle(),
            take_session=lambda operation: self._take_session(operation),
            release_session=lambda operation: self._release_session(operation),
            assign_workspace=lambda cwd, harness_files: self._assign_workspace(
                cwd, harness_files
            ),
            destination_checkout=lambda target: self._destination_checkout(target),
            bind_workspace=lambda cwd, harness_files: self._bind_workspace(
                cwd, harness_files
            ),
            reload_with_initial_messages=lambda **kwargs: (
                self.reload_with_initial_messages(**kwargs)
            ),
        )
        self.launch_context = launch_context
        config = self.config
        try:
            active_model = config.get_active_model()
            self.stats.input_price_per_million = active_model.input_price
            self.stats.output_price_per_million = active_model.output_price
            self.stats.cached_input_price_per_million = active_model.cached_input_price
        except ValueError:
            pass

        self._current_user_message_id: str | None = None
        self._is_user_prompt_call: bool = False
        self._reactive_recovery_used: bool = False
        # The pending injected-message buffer and the tracked steering tasks
        # are owned by the context injector; the loop keeps compatibility
        # views (see the properties below).
        self._pending_clear_context: bool = False

        self.session_logger = SessionLogger(
            config.session_logging,
            self.session_id,
            cwd=self.cwd,
            session_dir=session_dir,
        )
        self._title_controller = TitleController(
            read_at_gate=lambda: TitleGateInputs(
                config=self.config, previous_title=self.session_logger.title
            ),
            writer=self.session_logger,
            event_sink=_QueueTitleEventSink(self._out_of_band_events),
        )
        if self.session_logger.session_metadata is not None:
            self.session_logger.session_metadata.parent_session_id = parent_session_id
            self.install_launch_metadata()
        self._hook_config_result = hook_config_result
        self._hooks_manager = (
            HooksManager(hook_config_result.hooks, cwd=self.cwd)
            if hook_config_result
            else None
        )
        self.hook_config_issues = (
            hook_config_result.issues if hook_config_result else []
        )
        self.hooks_count = len(hook_config_result.hooks) if hook_config_result else 0
        # Explicit hook-mixin seams (stage 14.3c): the mixin reads its host
        # dependencies through these late-bound callbacks, never directly,
        # so hook reload, session reset, and relocation stay visible.
        self._hooks_manager_getter = lambda: self._hooks_manager
        self._hook_session_context = lambda: self._build_hook_session_context()
        self._stats_getter = lambda: self.stats
        self._messages_getter = lambda: self.messages
        checkpointer = Checkpointer()
        file_store = FileStore()
        self.checkpoint_recorder = CheckpointRecorder(
            checkpointer, self.messages, file_store
        )
        # Single-tool execution: the invocation helper with today's
        # snapshot/admission/invoke span lives on the executor; the loop
        # keeps the decision port, the wait registry, the transcript choke
        # points, and the synchronous invoke-context builder.
        self._tool_executor = ToolExecutor(
            stats_getter=lambda: self.stats,
            tool_manager_getter=lambda: self.tool_manager,
            dangerous_command_grants=self._dangerous_command_grants,
            checkpoint_recorder=self.checkpoint_recorder,
            harness_files_getter=lambda: self.harness_files,
            build_invoke_context=lambda tool_call: self._build_invoke_context(
                tool_call
            ),
            handle_tool_skip=lambda tool_call, decision: self._handle_tool_skip(
                tool_call, decision
            ),
            run_post_tool_and_finalize=(
                lambda tool_call, finalization: self._run_post_tool_and_finalize(
                    tool_call, finalization
                )
            ),
            observe_file_state=lambda tool_name, result_model: self._observe_file_state(
                tool_name, result_model
            ),
            serialize_tool_input=lambda tool_call: self._serialize_tool_input(
                tool_call
            ),
            run_pre_tool_pipeline=(
                lambda tool_call, tool_input: self._run_pre_tool_pipeline(
                    tool_call, tool_input
                )
            ),
            should_execute_tool=lambda tool, args, call_id: self._should_execute_tool(
                tool, args, call_id
            ),
            is_steering_cancellation=lambda call_id: self._is_steering_cancellation(
                call_id
            ),
            finalize_cancelled_tool=(
                lambda tool_call, tool_input, cancel_text, **kwargs: (
                    self._finalize_cancelled_tool(
                        tool_call, tool_input, cancel_text, **kwargs
                    )
                )
            ),
            tool_failure_event=lambda tool_call, error_msg: self._tool_failure_event(
                tool_call, error_msg
            ),
            handle_tool_response=lambda tool_call, text: self._handle_tool_response(
                tool_call, text
            ),
        )
        self.review_manager = ReviewManager(checkpointer, file_store)
        self.rewind_manager = RewindManager(
            checkpointer,
            messages=self.messages,
            save_messages=self._save_messages,
            reset_session=self._reset_session,
            files=file_store,
            fence_transcript=self._fence_rewind_transcript,
        )
        self.compaction_manager = CompactionManager(
            messages=self.messages,
            stats_getter=lambda: self.stats,
            config_getter=lambda: self.config,
            complete=self._complete,
            available_tools=lambda: self.format_handler.get_available_tools(
                self.tool_manager
            ),
            tool_choice=self.format_handler.get_tool_choice,
            save=self._save_messages,
            file_state_getter=lambda: self.file_state_context.store,
            deferred_questions_getter=lambda: self.deferred_questions,
            bound_policy_getter=lambda: self._ensure_bound_policy(),
        )

        if defer_heavy_init:
            self._start_deferred_init()
        else:
            self._complete_init()
            if err := self._init_error:
                raise err

    @property
    def _admitted_tool_policies(self) -> dict[str, secret_redaction.ScrubPolicy]:
        """Read-only compatibility view of the redactor's retained snapshots."""
        return self._outward_redactor.admitted_tool_policies

    @property
    def _tool_response_order(self) -> dict[str, int] | None:
        """Read-only compatibility view of the recorder's ordering window."""
        return self._transcript_recorder.response_order

    @property
    def _staged_tool_responses(self) -> dict[int, list[LLMMessage]]:
        """Read-only compatibility view of the recorder's staging buffers."""
        return self._transcript_recorder.staged_responses

    @property
    def _tool_invocations(self) -> dict[tuple[str, str], _ToolInvocation]:
        """Read-only compatibility view of the runner's wait registry."""
        return self._tool_batch_runner.invocations

    @property
    def _tool_event_queue(self) -> asyncio.Queue[BaseEvent | None] | None:
        """Read-only compatibility view of the runner's active batch queue."""
        return self._tool_batch_runner.tool_event_queue

    @property
    def _pending_injected_messages(self) -> list[LLMMessage]:
        """Read/write compatibility view of the injector's pending buffer."""
        return self._context_injector.pending_injected_messages

    @_pending_injected_messages.setter
    def _pending_injected_messages(self, messages: list[LLMMessage]) -> None:
        self._context_injector.pending_injected_messages = messages

    @property
    def _steering_injections(self) -> set[asyncio.Task[Any]]:
        """Read-only compatibility view of the injector's steering tasks."""
        return self._context_injector.steering_injections

    def _start_deferred_init(self) -> threading.Thread:
        """Spawn a daemon thread that finishes deferred heavy I/O once."""
        with self._deferred_init_lock:
            if self._deferred_init_thread is not None:
                return self._deferred_init_thread
            if self._closing:
                raise AgentLoopStateError("Cannot initialize a closing agent loop")

            thread = threading.Thread(
                target=self._complete_init, daemon=True, name="agent_loop_init"
            )
            self._deferred_init_thread = thread
            thread.start()
            return thread

    @property
    def is_initialized(self) -> bool:
        """Whether deferred initialization has completed (successfully or not)."""
        if not self._defer_heavy_init:
            return True
        thread = self._deferred_init_thread
        return thread is not None and not thread.is_alive()

    @property
    def active_background_job_count(self) -> int:
        """Live public count; retired runtimes do not advertise borrowed jobs."""
        if self._closing or self.background_jobs is None:
            return 0
        return self.background_jobs.committed_active_count

    def _complete_init(self) -> None:
        """Run deferred heavy I/O: MCP discovery.

        Intended to be called from a background thread when
        ``defer_heavy_init=True`` was passed to ``__init__``.
        """
        with self._deferred_init_lock:
            if self._closing:
                return
        try:
            with secret_redaction.bind_policy(self.scrub_policy):
                self._ensure_remote_registries()
                self.tool_manager.integrate_all(raise_on_mcp_failure=True)
                self._publish_system_prompt(self._build_system_prompt())
        except Exception as exc:
            self._init_error = exc

    async def wait_until_ready(self) -> None:
        """Await deferred initialization and record its duration."""
        await self._await_deferred_init()
        self._ensure_init_duration_recorded()

    async def _await_deferred_init(self) -> None:
        """Await the deferred init thread and plan-attribute task."""
        if self._defer_heavy_init:
            thread = self._start_deferred_init()
            await asyncio.to_thread(thread.join)
            if err := self._init_error:
                raise copy.copy(err).with_traceback(err.__traceback__)
        for task in (self._plan_attrs_task,):
            if task is None or task is asyncio.current_task():
                continue
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _ensure_init_duration_recorded(self) -> None:
        """Record initialization duration exactly once when initialization is armed."""
        if self._last_init_duration_ms is not None:
            return
        if not self._init_duration_pending and not self._defer_heavy_init:
            return
        self._last_init_duration_ms = int(
            (time.monotonic() - self._init_start_time) * 1000
        )
        self._init_duration_pending = False

    @property
    def config_orchestrator(self) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        return self._config_orchestrator

    @property
    def config(self) -> ChartreuxConfigSchema:
        return self.agent_manager.config

    @property
    def runtime_policy(self) -> AgentRuntimePolicy:
        return AgentRuntimePolicy(
            max_turns=self._max_turns,
            max_price=self._max_price,
            max_tokens=self._max_tokens,
            max_session_tokens=self._max_session_tokens,
            run_max_price=self._run_max_price,
            run_max_session_tokens=self._run_max_session_tokens,
            enable_streaming=self.enable_streaming,
            launch_context=self.launch_context,
            headless=self._headless,
            hook_config_result=self._hook_config_result,
            cache_store=self.cache_store,
            user_input_capability=self._user_input_capability,
            auto_title_enabled=self._auto_title_enabled,
            inherited_restrictions=self._inherited_restrictions,
            inherited_mode_restrictions=self._inherited_mode_restrictions,
            inherited_workspace=self._inherited_workspace,
            inherited_plan_write_scopes=self._inherited_plan_write_scopes,
            parent_authority_getter=self._parent_authority_getter,
            parent_authority_revision_getter=self._parent_authority_revision_getter,
        )

    @property
    def _dispatch_root_grants(self) -> tuple[Path, ...]:
        """Sorted bare canonical roots granted to this session at a dispatch."""
        store = getattr(self, "_session_root_grants", None)
        return tuple(sorted(store.roots)) if store is not None else ()

    @property
    def child_runtime_policy(self) -> AgentRuntimePolicy:
        """Capture accepted parent authority before child overrides or awaits.

        Only construction fixes this snapshot. Future explicit revocation must
        refresh descendants; ordinary child config/grants cannot replace it.
        Mode restrictions are execution scope, not persistent user policy.
        """
        sources = self.config_orchestrator.restrictions
        return replace(
            self.runtime_policy,
            inherited_workspace=self.tool_manager.workspace,
            parent_authority_getter=lambda: self.tool_manager,
            parent_authority_revision_getter=lambda: self._authority_revision,
            inherited_restrictions=tuple(
                dict.fromkeys(
                    self._inherited_restrictions
                    + tuple(source for source in sources if source.kind == "source")
                )
            ),
            inherited_root_grants=self._dispatch_root_grants,
        )

    async def record_child_session(
        self, child: AgentLoop, tool_call_id: str, agent_name: str
    ) -> ChildSessionLink:
        parent_dir = self.session_logger.session_dir
        child_dir = child.session_logger.session_dir
        relative_path = (
            str(child_dir.relative_to(parent_dir))
            if parent_dir is not None and child_dir is not None
            else None
        )
        link = ChildSessionLink(
            session_id=child.session_id,
            tool_call_id=tool_call_id,
            agent=agent_name,
            relative_path=relative_path,
        )
        metadata = self.session_logger.session_metadata
        if metadata is None:
            return link
        existing = next(
            (
                item
                for item in metadata.child_sessions
                if item.tool_call_id == tool_call_id
            ),
            None,
        )
        if existing is not None:
            if existing != link:
                raise RuntimeError(
                    f"Tool call {tool_call_id} is already linked to a child session"
                )
            return existing
        metadata.child_sessions.append(link)
        try:
            await self._save_messages()
            await self.session_logger.persist_child_sessions()
        except BaseException:
            metadata.child_sessions.remove(link)
            raise
        return link

    async def replace_child_session(
        self, old_session_id: str, child: AgentLoop, tool_call_id: str
    ) -> ChildSessionLink:
        parent_dir = self.session_logger.session_dir
        child_dir = child.session_logger.session_dir
        replacement = ChildSessionLink(
            session_id=child.session_id,
            tool_call_id=tool_call_id,
            agent="subagent",
            relative_path=(
                str(child_dir.relative_to(parent_dir))
                if parent_dir is not None and child_dir is not None
                else None
            ),
        )
        metadata = self.session_logger.session_metadata
        if metadata is None:
            return replacement
        index = next(
            (
                index
                for index, link in enumerate(metadata.child_sessions)
                if link.session_id == old_session_id
                and link.tool_call_id == tool_call_id
            ),
            None,
        )
        if index is None:
            raise RuntimeError(f"Child session link not found: {old_session_id}")
        previous = metadata.child_sessions[index]
        metadata.child_sessions[index] = replacement
        try:
            await self.session_logger.persist_child_sessions()
        except BaseException:
            metadata.child_sessions[index] = previous
            raise
        return replacement

    async def forget_child_session(
        self, child_session_id: str, tool_call_id: str
    ) -> None:
        metadata = self.session_logger.session_metadata
        if metadata is None:
            return
        index = next(
            (
                index
                for index, link in enumerate(metadata.child_sessions)
                if link.session_id == child_session_id
                and link.tool_call_id == tool_call_id
            ),
            None,
        )
        if index is None:
            return
        link = metadata.child_sessions.pop(index)
        try:
            await self.session_logger.persist_child_sessions()
        except BaseException:
            metadata.child_sessions.insert(index, link)
            raise

    async def persist_empty_session(self) -> None:
        await self._save_messages(allow_empty=True)

    async def refresh_config(self) -> None:
        await _config_transactions.refresh_config(self)

    @property
    def is_subagent(self) -> bool:
        """Whether this loop runs as a subagent (no own interactive turn)."""
        return self._is_subagent

    @property
    def user_input_capability(self) -> bool:
        """Whether this loop may ask the user questions interactively."""
        return self._user_input_capability

    @property
    def session_generation(self) -> int:
        """Monotonic session generation, bumped on transcript rewind/rebind."""
        return self._session_generation

    def resolve_user_input_request(self, request_id: str, result: BaseModel) -> None:
        self._request_broker.resolve_user_input(request_id, result)

    def reject_request(self, request_id: str, error: BaseException) -> None:
        self._request_broker.reject(request_id, error)

    def bind_root_grant_port(self, port: RootGrantPort) -> None:
        """Bind the runtime-owned port that applies approved session root grants."""
        self._root_grant_port = port

    def bind_dangerous_command_grant_port(
        self, port: DangerousCommandGrantPersistencePort
    ) -> None:
        """Bind the runtime-owned persistence port for approved project grants."""
        self._dangerous_command_grant_port = port

    def apply_root_grant(self, root: Path) -> bool:
        """Apply a user-approved root grant to this session's live authority."""
        granted = self._session_root_grants.grant(root)
        self.tool_manager._invalidate_session_root_grants()
        return granted

    async def _apply_approved_root_grant(self, root: Path) -> None:
        port = self._root_grant_port
        if port is not None:
            await port.grant_root(self.session_id, root)
        else:
            self.apply_root_grant(root)

    @property
    def init_duration_ms(self) -> int | None:
        return self._last_init_duration_ms

    @property
    def _auto_title_task(self) -> asyncio.Task[None] | None:
        return self._title_controller.task

    @property
    def _title_cadence(self) -> TitleCadence:
        return self._title_controller.cadence

    @_title_cadence.setter
    def _title_cadence(self, cadence: TitleCadence) -> None:
        self._title_controller.cadence = cadence

    async def aclose(self) -> None:
        task = self._close_task
        if task is None or (task.done() and task.exception() is not None):
            task = asyncio.create_task(
                self._aclose_owned(), name=f"vibe-agent-close:{self.session_id}"
            )
            self._close_task = task
            task.add_done_callback(self._observe_close_outcome)
        await asyncio.shield(task)
        secret_redaction.unregister_session_policy(self)

    def _observe_close_outcome(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            logger.error("Owned agent cleanup was cancelled: %s", self.session_id)
            return
        if exc := task.exception():
            logger.error(
                "Owned agent cleanup failed: %s", self.session_id, exc_info=exc
            )

    @staticmethod
    def _job_authority_config(config: ChartreuxConfigSchema) -> dict[str, Any]:
        return _config_transactions._job_authority_config(config)

    def _guard_job_authority_config(self, config: ChartreuxConfigSchema) -> None:
        _config_transactions._guard_job_authority_config(self, config)

    async def retire_background_jobs(self) -> None:
        """Settle the old root lifetime without stranding admission on resume failure."""
        registry = self._background_job_registry
        if registry is not None:
            cleanup = asyncio.create_task(registry.aclose())
            cancellation: asyncio.CancelledError | None = None
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError as exc:
                    cancellation = exc
            cleanup.result()
            replacement = BackgroundJobRegistry(
                generation=self._job_lifetime_generation + 1,
                event_sink=self._out_of_band_events.put_nowait,
            )
            self._background_job_registry = replacement
            self.background_jobs = replacement.root_port()
            self._job_lifetime_generation += 1
            if cancellation is not None:
                raise cancellation

    def require_background_jobs_idle(self) -> None:
        if self.background_jobs is not None and self.background_jobs.active_count:
            raise AgentLoopStateError(
                "Stop live background jobs before changing execution authority"
            )

    async def _aclose_owned(self) -> None:
        with self._deferred_init_lock:
            self._closing = True
            deferred_init_thread = self._deferred_init_thread
        if self._background_job_registry is not None:
            await self._background_job_registry.aclose()
        if deferred_init_thread is not None:
            await asyncio.to_thread(deferred_init_thread.join, 5.0)
            if deferred_init_thread.is_alive():
                raise TimeoutError(
                    "Deferred initialization did not stop during shutdown"
                )
        if self._mcp_pool is not None:
            self._mcp_pool.retire()
        await self._title_controller.aclose()
        for task in (self._plan_attrs_task,):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
        pools = list(self._retired_mcp_pools)
        if self._mcp_pool is not None:
            pools.append(self._mcp_pool)
        outcomes = await asyncio.gather(
            *(pool.aclose() for pool in pools), return_exceptions=True
        )
        self._retired_mcp_pools = [
            pool for pool in self._retired_mcp_pools if not pool.cleanup_complete
        ]
        errors = [result for result in outcomes if isinstance(result, BaseException)]
        if errors:
            # Keep ownership and session resources until cleanup actually finishes.
            # A later close can join a timed-out worker; it must not replay its call.
            raise BaseExceptionGroup("MCP shutdown cleanup incomplete", errors)
        await self._backend_lifetime.aclose()
        self._workspace_relocator.release_move_trust()
        self.harness_files.revoke_session_trusts()
        cleanup_scratchpad(self.scratchpad_dir)
        lease = self._session_lease
        self._session_lease = None
        if lease is not None:
            await asyncio.to_thread(lease.release)

    @staticmethod
    def _create_mcp_registry() -> MCPRegistry:
        from chartreux.core.tools.mcp.registry import MCPRegistry

        return MCPRegistry()

    def _create_mcp_pool(self) -> MCPConnectionPool:
        from chartreux.core.tools.mcp.pool import MCPConnectionPool

        return MCPConnectionPool(policy=self.scrub_policy)

    def _retire_mcp_pool(self) -> None:
        self._retired_mcp_pools = [
            pool for pool in self._retired_mcp_pools if not pool.cleanup_complete
        ]
        if self._mcp_pool is not None:
            self._mcp_pool.retire(drain_active=True)
            self._retired_mcp_pools.append(self._mcp_pool)
            self._mcp_pool = None

    def _ensure_remote_registries(self) -> None:
        if self.mcp_registry is None and self.config.mcp_servers:
            self.mcp_registry = self._create_mcp_registry()
            self.tool_manager.set_mcp_registry(self.mcp_registry)

        if self._mcp_pool is None and self.config.mcp_servers:
            self._mcp_pool = self._create_mcp_pool()

    def _execution_context(self) -> ExecutionContext:
        """The composed session's execution context (design §5 step 4)."""
        if self.launch_identity is not None:
            return ExecutionContext.BOUNDED_CHILD
        return (
            ExecutionContext.HEADLESS_MAIN
            if self._headless
            else ExecutionContext.INTERACTIVE_MAIN
        )

    def _ensure_bound_policy(self) -> ActivatedPolicy:
        """Bind the conversation's frozen policy bundle once (startup capture).

        The first render captures the current selections AND their resource
        content into the versioned bundle (design §3, §6; decision log items
        3 and 6). Every later render — refresh, reload, retask — serves the
        captured bundle and never rereads the frozen policy from disk (the
        invariant: no live reread of frozen policy during prompt refresh).
        """
        with self._policy_bind_lock:
            if self._bound_policy is None:
                self._bind_policy_bundle()
            assert self._bound_policy is not None
            return self._bound_policy

    def _bind_policy_bundle(self) -> None:
        """Capture the conversation's policy bundle and install it for persistence.

        The captured bundle becomes the conversation's persisted policy record
        (design §6, §10): it is installed into the session metadata so the
        existing save path persists it inline, and into the loop's bounded
        bundle store (never evicted while the conversation is resumable).
        """
        from chartreux.core.dispatch.session import bind_policy_bundle

        bundle, activated = bind_policy_bundle(
            self.config,
            execution_context=self._execution_context(),
            skill_manager=self.skill_manager,
            scratchpad_dir=self.scratchpad_dir,
            cwd=self.cwd,
            harness_files=self.harness_files,
            role_instructions=self.frozen_instructions,
        )
        self._bound_policy = activated
        self._bundle_store.put(self.session_id, bundle)
        self._bundle_store.set_resumable(self.session_id, True)
        metadata = self.session_logger.session_metadata
        if metadata is not None:
            metadata.policy_bundle = bundle.model_dump(mode="json")
            metadata.policy_storage_version = POLICY_STORAGE_VERSION

    def _identity_binding(self) -> tuple[str, str]:
        """The conversation's resolved identity reference and content revision."""
        activated = self._bound_policy
        if activated is not None:
            return (
                activated.manifest.identity.selected,
                activated.manifest.identity_revision,
            )
        return _resolve_identity_revision(self._bound_identity, self.harness_files)

    @property
    def frozen_purpose_routes(self) -> Mapping[str, PurposeRoute]:
        """The conversation's frozen purpose-route table (design §6, §10).

        The route snapshot the bound bundle's manifest captured at bind
        time, rebuilt as the executable ``[orchestration.routes]`` table.
        Launch admission resolves purpose routes against THIS table — the
        same source the rendered launch-target table carries — never the
        reloaded live config, so a mid-conversation route edit stays
        pending (reported by ``pending_policy_changes``) instead of
        producing a mixed-generation launch. Binding on demand matches the
        render path: the first reader captures the bundle every later
        reader serves, so admission and rendering can never diverge.
        """
        from chartreux.core.policy.manifest import routes_from_snapshot

        return routes_from_snapshot(self._ensure_bound_policy().manifest.route_snapshot)

    def _render_system_prompt(
        self,
        skill_manager: SkillManager,
        config: ChartreuxConfigSchema | None = None,
        tool_manager: ToolManager | None = None,
        agent_manager: AgentManager | None = None,
        *,
        scratchpad_dir: Path | None | _PrepareScratchpad = _PREPARE_SCRATCHPAD,
        activated_policy: ActivatedPolicy | None = None,
    ) -> str:
        """Render the served prompt from the conversation's bound policy bundle.

        The frozen sections — identity/workflow/strategy bodies, the universal
        fragments, the launch-target table, and the per-launch instruction
        overlay — come from the captured bundle, never reread from disk
        (design §10). Only the refreshable live sections (model and skills
        catalogs, environment, scratchpad, project context, commit signature,
        AGENTS.md layers) are recomputed from the current session state, so a
        refresh picks up tool/skill/catalog changes without re-posturing the
        conversation. ``activated_policy`` renders a DIFFERENT conversation's
        bundle (resume preparation); the default serves this loop's bound one.
        """
        _ = tool_manager, agent_manager  # live sections do not read managers
        activated = activated_policy or self._ensure_bound_policy()
        prompt_config = config or self.config
        live = _live_sections(
            activated,
            prompt_config,
            skill_manager,
            scratchpad_dir=(
                self.scratchpad_dir
                if isinstance(scratchpad_dir, _PrepareScratchpad)
                else scratchpad_dir
            ),
            cwd=self.cwd,
            harness_files=self.harness_files,
        )
        result = activated.render(live)
        return SystemPrompt(result.text, result.instruction_read_files)

    def _publish_system_prompt(self, prompt: str) -> None:
        """Install text and its loader provenance together, never during staging."""
        self.messages.update_system_prompt(prompt)
        self.tool_manager.set_instruction_read_files(
            getattr(prompt, "instruction_read_files", frozenset())
        )
        self._authority_revision += 1

    def _build_system_prompt(self) -> str:
        return self._render_system_prompt(self.skill_manager)

    async def prepare_launch_reconfiguration(
        self,
        candidate: LaunchCandidate,
        *,
        expected_session_generation: int,
        expected_parent_authority_revision: int,
    ) -> _PreparedLaunchReconfiguration:
        return await _config_transactions.prepare_launch_reconfiguration(
            self,
            candidate,
            expected_session_generation=expected_session_generation,
            expected_parent_authority_revision=expected_parent_authority_revision,
        )

    def _prepare_launch_consumers(
        self,
        target_config: ChartreuxConfigSchema,
        *,
        replace_tools: bool,
        reuse_backend: bool,
    ) -> _PreparedReload:
        return _config_transactions._prepare_launch_consumers(
            self,
            target_config,
            replace_tools=replace_tools,
            reuse_backend=reuse_backend,
        )

    def publish_launch_reconfiguration(
        self, prepared: _PreparedLaunchReconfiguration
    ) -> BackendPublish:
        return _config_transactions.publish_launch_reconfiguration(self, prepared)

    def rollback_launch_reconfiguration(
        self, prepared: _PreparedLaunchReconfiguration, backend_publish: BackendPublish
    ) -> None:
        _config_transactions.rollback_launch_reconfiguration(
            self, prepared, backend_publish
        )

    def _model_choice_pending(self) -> bool:
        """True while a recovered session awaits an explicit model selection.

        Mirrors ``committed_model_recovery_issue`` in the app-server
        projection: the stored launch envelope still commits a model while
        the loop no longer carries one, so resolving a default here would
        silently re-pin the session and dissolve the pending-choice issue.
        """
        if self.committed_model is not None:
            return False
        metadata = self.session_logger.session_metadata
        envelope = metadata.launch_config if metadata is not None else None
        return isinstance(envelope, LaunchMetadataV2)

    def _launch_metadata(self) -> LaunchMetadataV2 | None:
        if self.committed_model is None:
            return None
        identity, identity_revision = self._identity_binding()
        return LaunchMetadataV2(
            version=2,
            overrides=self.launch_overrides or LaunchConfig(),
            # The persona is re-keyed on the identity revision (design §7):
            # resume re-reads this binding instead of re-deriving identity
            # from the live configuration.
            persona=LaunchPersonaV2(
                identity=identity,
                identity_revision=identity_revision,
                instructions=self.frozen_instructions,
            ),
            committed_model=self.committed_model,
            # The committed purpose/tier/capability ride the envelope: the
            # loop's re-install keeps the persisted launch state describing
            # the package's true purpose/tier/capability.
            purpose=self.launch_purpose,
            package_tier=self.launch_package_tier,
            replacement_capability=self.launch_replacement_capability,
        )

    def install_launch_metadata(self) -> None:
        """Synchronously install accepted launch state before any lock wait."""
        metadata = self.session_logger.session_metadata
        if metadata is not None:
            # The bundle is the persisted policy record (design §6, §10):
            # keep the metadata's bundle in step with the bound policy.
            activated = self._bound_policy
            if activated is not None:
                metadata.policy_bundle = _bundle_from_activated(activated).model_dump(
                    mode="json"
                )
                metadata.policy_storage_version = POLICY_STORAGE_VERSION
        if (metadata := self._launch_metadata()) is not None:
            self.session_logger.install_launch_config(metadata)

    async def persist_launch_metadata(self) -> None:
        if (metadata := self._launch_metadata()) is not None:
            await self.session_logger.persist_launch_config(metadata)

    async def abort_launch_reconfiguration(
        self, prepared: _PreparedLaunchReconfiguration
    ) -> None:
        await _config_transactions.abort_launch_reconfiguration(self, prepared)

    def finalize_launch_reconfiguration(
        self, prepared: _PreparedLaunchReconfiguration, backend_publish: BackendPublish
    ) -> None:
        _config_transactions.finalize_launch_reconfiguration(
            self, prepared, backend_publish
        )

    @requires_init
    async def refresh_system_prompt(self) -> None:
        """Rebuild and replace the system prompt with current tool/skill state."""
        prompt = await asyncio.to_thread(self._build_system_prompt)
        self._publish_system_prompt(prompt)

    @property
    def backend(self) -> BackendLike:
        return self._backend_lifetime.active

    @property
    def _turn(self) -> _ActiveTurn:
        return self._active_turn or _NO_TURN

    def _take_session(self, operation: str) -> None:
        """Claim the session for *operation*, refusing if something else holds it.

        Both the claim and the refusal, because mutual exclusion needs each side
        to do both. An operation that only announced itself could still start
        inside another one's awaits.
        """
        if self._holders:
            raise AgentLoopStateError(
                f"Cannot start {operation} while {self._holders[0]} is running"
            )
        self._holders.append(operation)

    def _release_session(self, operation: str) -> None:
        self._holders.remove(operation)

    async def notice_retry(self, reason: RetryReason) -> None:
        if (sink := self._turn.retry_sink) is not None:
            await sink(reason)

    def backend_factory(
        self, config: ChartreuxConfigSchema | None = None
    ) -> BackendLike:
        return self._injected_backend or self._select_backend(config)

    def _select_backend(
        self, config: ChartreuxConfigSchema | None = None
    ) -> BackendLike:
        return select_backend(
            config or self.config, on_retry=self.notice_retry, factory=create_backend
        )

    def refresh_backend_for_resume(self) -> None:
        """Retire this loop's failed backend and install a fresh one (WP2).

        Resume-time backend refresh (handoff §1.4 item 6): ONLY this loop's
        active backend is retired, through its ownership-aware
        ``BackendLifetime`` — it closes once no turn or borrow protects it.
        Sibling agents own separate backend lifetimes and are never touched,
        and no global client reset occurs (F3 borrowed-backend semantics).
        The replacement is created through the normal backend factory, so
        provider configuration and the shared availability registry apply.
        Never callable while a turn is active: the failed run's tasks are
        joined before the resume admission reaches this point.
        """
        if self._active_turn is not None:
            raise AgentLoopStateError(
                "Cannot refresh the backend while a turn is active"
            )
        self._backend_lifetime.replace(self._select_backend(), whole_turn_active=False)

    async def _save_messages(self, *, allow_empty: bool = False) -> None:
        # The persisted policy record is the conversation's bound bundle
        # (design §6, §10): a save forces the startup capture when no render
        # has bound it yet (deferred init still warming up), so a record is
        # never persisted without its frozen bundle — a bundle-less record is
        # a legacy record and cannot be resumed.
        self._ensure_bound_policy()
        await self.session_logger.save_interaction(
            self.messages,
            self.stats,
            self.config,
            self.tool_manager,
            allow_empty=allow_empty,
        )

    @requires_init
    async def inject_user_context(  # noqa: PLR0913
        self,
        content: str,
        *,
        as_message: bool = False,
        inject_implicit: bool = False,
        images: list[ImageAttachment] | None = None,
        input_text: str | None = None,
        resources: list[UserResource] | None = None,
        client_message_id: str | None = None,
        manual_shell: ManualShellContext | None = None,
        before_commit: Callable[[], None] | None = None,
        on_commit: Callable[[UserMessageEvent], None] | None = None,
        on_injected_event: Callable[[BaseEvent], None] | None = None,
        on_context_ready: Callable[[], object] | None = None,
    ) -> list[BaseEvent]:
        # Compatibility delegate: the user-context construction, the callback
        # bracketing, and the pending/steering state live on the context
        # injector; the deferred-init gate stays on this entrypoint.
        return await self._context_injector.inject_user_context(
            content,
            as_message=as_message,
            inject_implicit=inject_implicit,
            images=images,
            input_text=input_text,
            resources=resources,
            client_message_id=client_message_id,
            manual_shell=manual_shell,
            before_commit=before_commit,
            on_commit=on_commit,
            on_injected_event=on_injected_event,
            on_context_ready=on_context_ready,
        )

    @requires_init
    async def act(
        self,
        msg: str | None,
        client_message_id: str | None = None,
        *,
        auto_title: str | None = None,
        images: list[ImageAttachment] | None = None,
        user_display_content: UserDisplayContent | None = None,
        input_text: str | None = None,
        resources: list[UserResource] | None = None,
        subagent_runner: SubagentRunnerPort | None = None,
        tool_io: ToolIOPort | None = None,
        turn_options: AgentTurnOptions | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        try:
            active_model = self.config.get_active_model()
        except ValueError:
            active_model = None
        if images and active_model is not None and not active_model.supports_images:
            raise ImagesNotSupportedError(
                active_model.display_name or active_model.alias
            )
        if self._active_turn is not None:
            raise AgentLoopStateError("A turn is already active")
        # A holder is not a turn, so the check above cannot see one. A move
        # refuses while a turn is active and then awaits twice; without this a
        # turn could begin in that window and bind its tools to the directory
        # being moved out from under them.
        if self._holders:
            raise AgentLoopStateError(
                f"Cannot start a turn while {self._holders[0]} is running"
            )
        self._backend_lifetime.drain(whole_turn_active=False)
        options = turn_options or AgentTurnOptions()
        if options.user_initiated_retry and self.committed_model is not None:
            self.config_orchestrator.availability_registry.reset_base(
                self.committed_model.base_model
            )
        self._outward_redactor.clear_admissions()
        self.completed_turn_timing = None
        self._active_turn = _ActiveTurn(
            turn_id=options.turn_id or str(uuid4()),
            subagent_runner=subagent_runner,
            tool_io=tool_io,
            retry_sink=options.retry_sink,
        )
        self._active_turn.timing.start()
        try:
            self._clean_message_history()
            self.checkpoint_recorder.create_checkpoint()
            try:
                async with contextlib.aclosing(
                    self._conversation_loop(
                        msg,
                        client_message_id=client_message_id,
                        auto_title=auto_title,
                        images=images,
                        user_display_content=user_display_content,
                        input_text=input_text,
                        resources=resources,
                        injected=options.injected,
                    )
                ) as conversation:
                    async for event in self._outward_redactor.sanitize(conversation):
                        yield event
            finally:
                try:
                    await self._finalize_turn_timing()
                finally:
                    self.checkpoint_recorder.seal_turn()
        finally:
            self._outward_redactor.clear_admissions()
            self._tool_batch_runner.clear_turn(self._turn.turn_id)
            self._active_turn = None
            self._backend_lifetime.drain(whole_turn_active=False)

    async def _finalize_turn_timing(self) -> None:
        turn = self._active_turn
        if turn is None:
            return
        duration = turn.timing.finish()
        if duration is None:
            return
        retained = {
            message.message_id: message
            for message in self.messages
            if message.role is Role.assistant and message.content
        }
        for message_id in reversed(turn.prose_ids):
            if (message := retained.get(message_id)) is not None:
                message.turn_duration = duration
                self.completed_turn_timing = CompletedTurnTiming(message_id, duration)
                # The owner can precede the logger's last saved boundary message.
                try:
                    self.session_logger.invalidate_transcript_cursor()
                    await self._save_messages()
                except Exception:
                    logger.warning("Could not persist turn timing", exc_info=True)
                break

    def _last_user_message(self) -> LLMMessage | None:
        return AgentLoop._last_user_message_from(select_model_context(self.messages))

    def _current_model_context(self) -> list[LLMMessage]:
        return select_model_context(self.messages)

    @staticmethod
    def _last_user_message_from(messages: Sequence[LLMMessage]) -> LLMMessage | None:
        return next(
            (m for m in reversed(messages) if m.role == Role.user and not m.injected),
            None,
        )

    def set_max_turns(self, max_turns: int) -> None:
        self._max_turns = max_turns
        self._setup_middleware()

    def set_max_tokens(self, max_tokens: int) -> None:
        self._max_tokens = max_tokens

    def _setup_middleware(self) -> None:
        self.middleware_pipeline = self._build_middleware_pipeline(self.config)

    def _build_middleware_pipeline(
        self, config: ChartreuxConfigSchema
    ) -> MiddlewarePipeline:
        """Construct the candidate pipeline without clearing the active one."""
        pipeline = MiddlewarePipeline()

        if self._max_turns is not None:
            pipeline.add(TurnLimitMiddleware(self._max_turns))

        if self._max_price is not None:
            pipeline.add(PriceLimitMiddleware(self._max_price))

        if self._max_session_tokens is not None:
            pipeline.add(TokenLimitMiddleware(self._max_session_tokens))

        # The run budget sits after the absolute session backstops: pipeline
        # order is evaluation order and the first STOP wins.
        if self._run_max_price is not None or self._run_max_session_tokens is not None:
            pipeline.add(
                RunBudgetMiddleware(
                    run_max_price=self._run_max_price,
                    run_max_session_tokens=self._run_max_session_tokens,
                )
            )

        pipeline.add(AutoCompactMiddleware())
        if config.context_warnings:
            pipeline.add(ContextWarningMiddleware(0.5))

        return pipeline

    async def _handle_middleware_result(
        self, result: MiddlewareResult
    ) -> AsyncGenerator[BaseEvent]:
        match result.action:
            case MiddlewareAction.STOP:
                yield AssistantEvent(
                    content=f"<{VIBE_STOP_EVENT_TAG}>{result.reason}</{VIBE_STOP_EVENT_TAG}>",
                    stopped_by_middleware=True,
                    middleware_metadata=result.metadata,
                )

            case MiddlewareAction.INJECT_MESSAGE:
                if result.message:
                    injected_message = LLMMessage(
                        role=Role.user, content=result.message, injected=True
                    )
                    self.messages.append(injected_message)

            case MiddlewareAction.COMPACT:
                async for event in self._run_compaction():
                    yield event

            case MiddlewareAction.CONTINUE:
                pass

    async def _run_compaction(self) -> AsyncGenerator[BaseEvent]:
        # Auto/reactive compaction: emit boundary events, compact, report status.
        old_tokens = self.stats.context_tokens
        threshold = self.config.get_active_model().auto_compact_threshold
        old_session_id = self.session_id
        tool_call_id = str(uuid4())

        yield CompactStartEvent(
            tool_call_id=tool_call_id,
            current_context_tokens=old_tokens,
            threshold=threshold,
        )

        try:
            summary = await self.compact()
        except asyncio.CancelledError:
            raise

        yield CompactEndEvent(
            tool_call_id=tool_call_id,
            summary_length=len(summary),
            old_session_id=old_session_id,
            new_session_id=self.session_id,
        )

    @property
    def user_plan(self) -> str | None:
        # The legacy ``_user_plan`` field is a fallback for paths that resolve a
        # plan without an explicit user selection.
        return self._user_plan

    @property
    def file_state(self) -> FileStateStore:
        """The working-set file state store (path -> last-seen revision + outline)."""
        return self.file_state_context.store

    @property
    def file_revision_registry(self) -> RevisionRegistryPort | None:
        """The dispatcher-level revision registry shared across the session tree."""
        return self.file_state_context.registry

    def file_state_handoff_message(self) -> LLMMessage | None:
        """The spawn handoff block for this loop's working set, if any.

        Redacted like every message entering the transcript; ``None`` when the
        working set is empty (nothing was handed down).
        """
        message = self.file_state_context.handoff_message()
        if message is None:
            return None
        return message.model_copy(
            update={"content": secret_redaction.redact(message.content or "")}
        )

    def _observe_file_state(self, tool_name: str, result_model: BaseModel) -> None:
        """Record one successful tool result into the working-set file state.

        Fail-safe: an observation bug must never break a turn — the store and
        registry are updated at observation time, so a failure here only loses
        metadata, never conversation state.
        """
        try:
            self.file_state_context.observe_tool_result(
                tool_name,
                result_model,
                sensitive_patterns=self._tool_sensitive_patterns(tool_name),
            )
        except Exception as exc:
            logger.warning(
                "File state observation failed tool=%s error=%s",
                tool_name,
                type(exc).__name__,
            )

    def _tool_sensitive_patterns(self, tool_name: str) -> list[str]:
        """The observing tool's sensitive patterns (defaults when unknown)."""
        try:
            config = self.tool_manager.get_tool_config(tool_name)
        except Exception:
            return list(DEFAULT_SENSITIVE_PATTERNS)
        return list(config.sensitive_patterns) or list(DEFAULT_SENSITIVE_PATTERNS)

    def _flush_file_state_blocks(self) -> None:
        """Append pending working-set state blocks as the last messages.

        Blocks are append-only and last-positioned: nothing already in the
        transcript is rewritten, so the cached request prefix stays valid.
        Fail-safe like observation: a flush bug must never break a turn.
        """
        try:
            blocks = self.file_state_context.flush_blocks()
        except Exception as exc:
            logger.warning("File state block flush failed error=%s", type(exc).__name__)
            return
        for message in blocks:
            self.messages.append(
                message.model_copy(
                    update={"content": secret_redaction.redact(message.content or "")}
                )
            )

    def set_user_plan(self, user_plan: str | None) -> None:
        self._user_plan = user_plan

    def _should_self_heal(self) -> bool:
        # Recover from an overflow at most once per turn; strict mode surfaces it.
        return (
            not self._reactive_recovery_used
            and not self.config.raise_on_compaction_failure
        )

    def _get_context(self) -> ConversationContext:
        return ConversationContext(
            messages=self.messages, stats=self.stats, config=self.config
        )

    def _build_backend_metadata(self, call_type: str | None = None) -> dict[str, str]:
        metadata = {
            "session_id": self.session_id,
            "call_type": call_type
            or ("main_call" if self._is_user_prompt_call else "secondary_call"),
        }
        if self.parent_session_id is not None:
            metadata["parent_session_id"] = self.parent_session_id
        if self._current_user_message_id is not None:
            metadata["message_id"] = self._current_user_message_id
        if self.user_plan is not None:
            metadata["user_plan"] = self.user_plan
        if isinstance(self.launch_context, dict):
            for key in (
                "agent_entrypoint",
                "agent_version",
                "client_name",
                "client_version",
                "terminal_emulator",
            ):
                value = self.launch_context.get(key)
                if value is not None:
                    metadata[key] = str(value)
        return metadata

    async def _open_user_turn(
        self,
        user_msg: str | None,
        *,
        client_message_id: str | None = None,
        auto_title: str | None = None,
        images: list[ImageAttachment] | None = None,
        user_display_content: UserDisplayContent | None = None,
        input_text: str | None = None,
        resources: list[UserResource] | None = None,
        injected: bool = False,
    ) -> AsyncGenerator[BaseEvent]:
        if user_msg is None:
            # Idle Plan preparation already appended the reviewed seed. Do not
            # fabricate a second user message for the implementation phase.
            last_user = self._last_user_message()
            self._current_user_message_id = (
                last_user.message_id if last_user is not None else None
            )
            return
        if injected:
            self.messages.append(
                LLMMessage(
                    role=Role.user,
                    content=user_msg,
                    injected=True,
                    images=images or None,
                    user_display_content=user_display_content,
                    input_text=input_text,
                    resources=resources or None,
                )
            )
            last_user = self._last_user_message()
            self._current_user_message_id = (
                last_user.message_id if last_user is not None else None
            )
            return

        user_message = LLMMessage(
            role=Role.user,
            content=user_msg,
            message_id=client_message_id,
            posted_at=posting_time(),
            images=images or None,
            user_display_content=user_display_content,
            input_text=input_text,
            resources=resources or None,
        )
        self.messages.append(user_message)
        self.stats.steps += 1
        self._current_user_message_id = user_message.message_id

        if user_message.message_id is None:
            raise AgentLoopError("User message must have a message_id")

        yield UserMessageEvent(
            content=input_text if input_text is not None else user_msg,
            message_id=user_message.message_id,
            posted_at=user_message.posted_at,
            images=list(user_message.images or []),
            user_display_content=user_message.user_display_content,
            resources=list(user_message.resources or []),
        )

        async for event in self._context_injector.inject_skill(user_msg):
            yield event

        async for event in self._context_injector.inject_mentions(user_msg):
            yield event

        if auto_title is not None and self.session_logger.set_initial_auto_title(
            auto_title
        ):
            yield SessionTitleUpdatedEvent(title=auto_title, session_id=self.session_id)

        if self._hooks_manager:
            self._hooks_manager.reset_retry_count()

    def _has_unanswered_user_message(self) -> bool:
        """Whether the live tail ends in a user message no assistant answers.

        PI1/T (S5): with anchored result placement the settled tool result
        lands after its owning assistant message, BEFORE the steered user
        message, so a steer can be the LAST message while its turn is still
        in flight. The conversation loop must not break on role alone there —
        it would strand the steer unprocessed. A trailing user message has no
        assistant reply after it by construction.
        """
        if not self.messages:
            return False
        return self.messages[-1].role == Role.user

    async def _conversation_loop(  # noqa: PLR0912, PLR0915 - conversation lifecycle
        self,
        user_msg: str | None,
        client_message_id: str | None = None,
        *,
        auto_title: str | None = None,
        images: list[ImageAttachment] | None = None,
        user_display_content: UserDisplayContent | None = None,
        input_text: str | None = None,
        resources: list[UserResource] | None = None,
        injected: bool = False,
    ) -> AsyncGenerator[BaseEvent]:
        async for event in self._open_user_turn(
            user_msg,
            client_message_id=client_message_id,
            auto_title=auto_title,
            images=images,
            user_display_content=user_display_content,
            input_text=input_text,
            resources=resources,
            injected=injected,
        ):
            yield event

        completed_normally = False
        try:
            should_break_loop = False
            first_llm_turn = True
            self._reactive_recovery_used = False
            while not should_break_loop:
                self._is_user_prompt_call = False
                result = await self.middleware_pipeline.run_before_turn(
                    self._get_context()
                )
                async for event in self._handle_middleware_result(result):
                    yield event

                if result.action == MiddlewareAction.STOP:
                    return

                user_cancelled = False
                self._is_user_prompt_call = first_llm_turn
                try:
                    async with contextlib.aclosing(self._perform_llm_turn()) as turn:
                        async for event in turn:
                            user_cancelled = (
                                user_cancelled or is_user_cancellation_event(event)
                            )
                            yield event
                except EmptyReplayBudgetStop as stop:
                    # The gateway denied the bounded empty replay at a budget
                    # checkpoint: re-evaluate the FULL pipeline for the
                    # authoritative stop reason and terminate through the
                    # middleware stop path — no replay, failover, compaction,
                    # retry, step increment, title, hooks, or recovery.
                    failure = classify(stop)
                    if failure.retry_eligible or failure.failover_eligible:
                        raise AgentLoopStateError(
                            "Empty replay budget stop is retry- or failover-eligible"
                        ) from stop
                    result = await self.middleware_pipeline.run_before_turn(
                        self._get_context()
                    )
                    if result.action is not MiddlewareAction.STOP:
                        raise AgentLoopStateError(
                            "Empty replay denied without a middleware STOP"
                        ) from None
                    async for event in self._handle_middleware_result(result):
                        yield event
                    return
                except ContextTooLongError:
                    if not self._should_self_heal():
                        raise
                    self._reactive_recovery_used = True
                    async for event in self._run_compaction():
                        yield event
                    continue  # retry the turn — still the user's first response
                # A turn ran to completion: count it against the turn budget (so
                # an overflow-and-retry never does) and mark later turns as
                # follow-ups.
                await self._settle_steering_injections()
                self.stats.steps += 1
                first_llm_turn = False
                # Per-turn save so the on-disk log stays fresh; after the
                # inner loop so pre_tool rewrites land in the snapshot.
                await self._save_messages()
                self._is_user_prompt_call = False

                # Schedule after each model step, not only at turn end: a single
                # tool-heavy turn can run for minutes, and the first title should
                # land as soon as there is usable context. The step ending with
                # an assistant answer (not a tool result) means the turn is
                # completing, which the cadence uses to time the initial title.
                async for event in self._schedule_title_generation_events(
                    turn_completing=self.messages[-1].role != Role.tool
                ):
                    yield event

                await self._settle_steering_injections()
                last_message = self.messages[-1]
                drained = self._context_injector.drain_pending()
                # PI1/T (S5): anchored result placement can leave a steered
                # user message as the last message of an in-flight turn; the
                # loop continues so the steer is processed, not stranded.
                should_break_loop = (
                    last_message.role != Role.tool
                    and not drained
                    and not self._has_unanswered_user_message()
                )

                if user_cancelled:
                    return

                if should_break_loop:
                    retry_msg, hook_events = await self._dispatch_post_turn_hooks()
                    for hook_event in hook_events:
                        yield hook_event
                    should_break_loop = self._queue_post_turn_retry(retry_msg)
                    # The final save can yield long enough for a background-agent
                    # completion to queue an injection. Save before the last drain so
                    # no await remains between observing an empty queue and exit.
                    if should_break_loop:
                        await self._save_messages()
                    had_steering = bool(self._steering_injections)
                    await self._settle_steering_injections()
                    if self._context_injector.drain_pending() or had_steering:
                        should_break_loop = False
            completed_normally = True
        finally:
            if not completed_normally:
                await self._settle_steering_injections(cancel=True)
                await self._save_messages()

    async def _settle_steering_injections(self, *, cancel: bool = False) -> None:
        # Compatibility delegate: settlement lives on the context injector;
        # the patch rider (tests/app_server/test_wait_steering.py:869/:875)
        # captures and patches this name on the loop, and the loop call sites
        # keep going through it.
        await self._context_injector.settle(cancel=cancel)

    def _queue_post_turn_retry(self, retry_msg: LLMMessage | None) -> bool:
        # Returns whether the loop should still break (no retry queued).
        if retry_msg is None:
            return True
        self.messages.append(retry_msg)
        return False

    async def _schedule_title_generation_events(
        self, *, turn_completing: bool
    ) -> AsyncGenerator[BackgroundWorkEvent, None]:
        scheduled = self._maybe_schedule_title_generation(
            turn_completing=turn_completing
        )
        if scheduled is None:
            return
        try:
            yield scheduled.started_event
        except (GeneratorExit, asyncio.CancelledError):
            scheduled.abort_delivery()
            raise
        else:
            scheduled.release()

    def _maybe_schedule_title_generation(
        self, *, turn_completing: bool
    ) -> ScheduledTitle | None:
        resources = self._call_resources(
            self.backend, RequestRetryBudget(self.config.api_retry_max_elapsed_time)
        )
        return self._title_controller.schedule(
            TitleScheduleInputs(
                messages=tuple(self.messages),
                session_id=self.session_id,
                turn_completing=turn_completing,
                enabled=self._auto_title_enabled,
                disabled_by_test_switch=(
                    os.environ.get(_DISABLE_AUTO_TITLE_ENV_VAR) == "1"
                ),
                logging_enabled=self.session_logger.enabled,
                title_is_manual=self.session_logger.title_source == "manual",
                periodic=is_fast_utility_model(self.config),
                accounting_sink=resources.accounting_sink,
                usage_attribution=resources.usage_attribution,
            )
        )

    async def out_of_band_events(self) -> AsyncGenerator[BaseEvent, None]:
        """Stream events produced outside a turn.

        The delivery layer drains this as the single consumer, so title,
        background-work and managed-job state events stay ordered and surface once.
        """
        while True:
            yield await self._out_of_band_events.get()

    def _cancel_auto_title_task(self) -> None:
        self._title_controller.cancel()

    def _reset_title_state(self) -> None:
        self._title_controller.reset()
        while not self._out_of_band_events.empty():
            self._out_of_band_events.get_nowait()

    async def _inject_invoked_skill(
        self, user_msg: str
    ) -> AsyncGenerator[BaseEvent, None]:
        # Compatibility delegate: the synthetic skill expansion lives on the
        # context injector; tests drive this name directly
        # (test_agent_tool_call.py:503).
        async for event in self._context_injector.inject_skill(user_msg):
            yield event

    async def _perform_llm_turn(self) -> AsyncGenerator[BaseEvent, None]:
        await self._settle_steering_injections()
        # Append point for working-set state blocks. This is the ONLY flush
        # point: the conversation loop continues while the last message is a
        # tool result, so a block must never be appended after a batch (it
        # would end the turn early). Flushing here appends the block as the
        # last message at append time — after the previous batch's tool
        # results, before this turn's assistant message — which keeps provider
        # tool_use/tool_result adjacency intact and the cached request prefix
        # stable. It also catches steering-injected reads (e.g. @mentioned
        # files) observed outside a tool batch.
        self._flush_file_state_blocks()
        self._execution_message = None
        try:
            if self.enable_streaming:
                async for event in self._stream_assistant_events():
                    yield event
            else:
                assistant_event = await self._get_assistant_event()
                if assistant_event.content:
                    yield assistant_event

            last_message = self._execution_message or self.messages[-1]
            parsed = self.format_handler.parse_message(last_message)
        finally:
            self._execution_message = None
        resolved = self.format_handler.resolve_tool_calls(parsed, self.tool_manager)

        if not resolved.tool_calls and not resolved.failed_calls:
            return

        async with contextlib.aclosing(
            self._tool_batch_runner.handle_tool_calls(resolved)
        ) as calls:
            async for event in calls:
                yield event

    async def _stream_assistant_events(
        self,
    ) -> AsyncGenerator[AssistantEvent | ReasoningEvent | ToolCallEvent]:
        message_id: str | None = None
        reasoning_message_id: str | None = None
        emitted_tool_call_ids = set[str]()

        async for chunk in self._chat_streaming():
            if message_id is None:
                message_id = chunk.message.message_id
            if reasoning_message_id is None:
                reasoning_message_id = chunk.message.reasoning_message_id

            if chunk.message.reasoning_content:
                yield ReasoningEvent(
                    content=chunk.message.reasoning_content,
                    message_id=reasoning_message_id,
                    posted_at=chunk.message.posted_at,
                )

            if chunk.message.content:
                yield AssistantEvent(
                    content=chunk.message.content,
                    message_id=message_id,
                    posted_at=chunk.message.posted_at,
                )

            for event in self._tool_batch_runner.build_tool_call_events(
                chunk.message.tool_calls, emitted_tool_call_ids
            ):
                emitted_tool_call_ids.add(event.tool_call_id)
                yield event

    async def _get_assistant_event(self) -> AssistantEvent:
        llm_result = await self._chat()
        return AssistantEvent(
            content=llm_result.message.content or "",
            message_id=llm_result.message.message_id,
            posted_at=llm_result.message.posted_at,
        )

    def _recorded_tool_call(self, tool_call: ToolCall) -> ToolCall:
        return self._transcript_recorder.record_call(tool_call)

    def _recorded_tool_call_event(self, event: ToolCallEvent) -> ToolCallEvent:
        return self._transcript_recorder.record_call_event(event)

    def outstanding_wait_call_ids(self, turn_id: str) -> tuple[str, ...]:
        # Compatibility delegate: the batch wait registry lives on the
        # runner; tests and the app server call this name on the loop.
        return self._tool_batch_runner.outstanding_wait_call_ids(turn_id)

    def is_waiting_only(self, turn_id: str) -> bool:
        # Compatibility delegate: tests patch this name on the loop
        # instance and the app server reads it, so the delegate stays.
        return self._tool_batch_runner.is_waiting_only(turn_id)

    def cancel_outstanding_waits_for_steering(self, turn_id: str) -> tuple[str, ...]:
        # Compatibility delegate: tests wrap this name on the loop
        # instance and the app server calls it, so the delegate stays.
        return self._tool_batch_runner.cancel_waits_for_steering(turn_id)

    def _is_steering_cancellation(self, call_id: str) -> bool:
        # Compatibility delegate: the executor's wait-registration port
        # resolves this name on the loop, so instance patches of it keep
        # intercepting (test_agent_wait_steering.py:454-461).
        return self._tool_batch_runner.is_steering_cancellation(call_id)

    async def _process_one_tool_call(
        self, tool_call: ResolvedToolCall
    ) -> AsyncGenerator[ToolResultEvent | ToolStreamEvent | HookEvent]:
        # Compatibility delegate: one complete tool lifecycle (lookup,
        # serialization, pre-hook pipeline, decision, invocation, and the
        # cancellation/failure envelope) lives on the executor. The loop
        # call sites keep going through this name so instance patches of it
        # keep intercepting (test_secret_redaction_exfil.py:437/:487) and
        # direct calls keep working (test_background_jobs.py:367/:435).
        async for event in self._tool_executor.process(tool_call):
            yield event

    def _build_invoke_context(self, tool_call: ResolvedToolCall) -> InvokeContext:
        """Build the invocation context for one tool call (executor seam).

        Synchronous on purpose: the executor enters the invocation right
        after the execution admission gate, so no await or yielding helper
        may sit between the gate and the invoke. Every live turn value
        stays on the loop's side of the seam.
        """
        return InvokeContext(
            tool_call_id=tool_call.call_id,
            agent_manager=self.agent_manager,
            session_dir=self.session_logger.session_dir,
            launch_context=self.launch_context,
            interaction_requests=self._request_broker,
            subagent_runner=self._turn.subagent_runner,
            subagent_manager=(
                None
                if self._is_subagent
                else cast(SubagentManagementPort, self._turn.subagent_runner)
            ),
            plan_file_path=self._plan_session.plan_file_path,
            request_clear_context_callback=self._request_clear_context,
            skill_manager=self.skill_manager,
            is_skill_loaded=self._context_injector.skill_already_loaded,
            scratchpad_dir=self.scratchpad_dir,
            hook_config_result=self._hook_config_result,
            session_id=self.session_id,
            mcp_pool=self._mcp_pool,
            tool_io=self._turn.tool_io,
            background_jobs=self.background_jobs,
            is_subagent=self._is_subagent,
            register_wait_task=lambda task, key=(self._turn.turn_id, tool_call.call_id): (
                self._tool_batch_runner.register_wait_task(key, task)
            ),
        )

    async def _should_execute_tool(
        self, tool: BaseTool, args: BaseModel, tool_call_id: str | None = None
    ) -> ToolDecision:
        tool_name = tool.get_name()
        # Always evaluate invocation guards, even with old grants/bypass flags.
        ctx = tool.resolve_permission(args)
        config_perm = self.tool_manager.get_tool_config(tool_name).permission
        if config_perm == ToolPermission.NEVER or (
            ctx is not None and ctx.permission == ToolPermission.NEVER
        ):
            if (
                ctx is not None
                and ctx.denial_kind == "out_of_root"
                and config_perm != ToolPermission.NEVER
            ):
                decision = await self._tool_decisions.resolve_out_of_root_grant(
                    tool, args, ctx, tool_name, tool_call_id
                )
                if decision is not None:
                    return decision
            if (
                ctx is not None
                and ctx.denial_kind == "dangerous_command"
                and ctx.dangerous_command_proposal is not None
                and config_perm != ToolPermission.NEVER
            ):
                decision = await self._tool_decisions.resolve_dangerous_command_grant(
                    tool, args, ctx, tool_name, tool_call_id
                )
                if decision is not None:
                    return decision
            return ToolDecision(
                verdict=ToolExecutionResponse.SKIP,
                approval_type=ToolPermission.NEVER,
                feedback=(ctx.reason if ctx is not None else None)
                or f"Tool '{tool_name}' is disabled by policy",
            )
        # Tools execute automatically unless denied. Positive accident guards
        # must return NEVER at their resolver; opaque extensions are
        # trusted code, not proven filesystem-contained implementations.
        return ToolDecision(
            verdict=ToolExecutionResponse.EXECUTE, approval_type=ToolPermission.ALWAYS
        )

    async def _persist_project_root_grant(self, root: Path) -> str:
        # Compatibility delegate: root persistence lives on the coordinator.
        return await self._tool_decisions.persist_project_root_grant(root)

    async def _prompt_session_root_grant(
        self, root: Path, tool_call_id: str | None
    ) -> str:
        # Compatibility delegate: the shared root-prompt implementation and
        # its asker lifecycle live on the coordinator.
        return await self._tool_decisions.prompt_root(root, tool_call_id)

    def _build_hook_session_context(self) -> HookSessionContext:
        """Build the hook session context (mixin seam, stage 14.3c).

        The mixin reaches this through the late-bound _hook_session_context
        callback, so the session identity, transcript path, cwd, and parent
        stay on the loop's side of the seam and relocation stays visible.
        """
        transcript = ""
        if self.session_logger.enabled and self.session_logger.session_dir is not None:
            transcript = str(self.session_logger.messages_filepath.resolve())
        return HookSessionContext(
            session_id=self.session_id,
            transcript_path=transcript,
            cwd=str(self.cwd),
            parent_session_id=self.parent_session_id,
        )

    def _handle_tool_response(
        self,
        tool_call: ResolvedToolCall,
        text: str,
        persisted_result: PersistedToolResult | None = None,
        images: list[ImageAttachment] | None = None,
    ) -> None:
        # Host seam for AgentLoopHooksMixin: the single choke point for every
        # tool response entering the message list lives on the recorder.
        self._transcript_recorder.record_response(
            tool_call, text, persisted_result, images
        )

    def _tool_failure_event(
        self, tool_call: ResolvedToolCall, error_msg: str, cancelled: bool = False
    ) -> ToolResultEvent:
        """Create a ToolResultEvent for a failed tool and record the failure."""
        self._handle_tool_response(tool_call, error_msg)
        return ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            error=error_msg,
            cancelled=cancelled,
            tool_call_id=tool_call.call_id,
        )

    def _completion_inputs(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        call_type: str | None = None,
        account_conversation: bool = True,
        purpose: UsagePurpose = UsagePurpose.CONVERSATION,
    ) -> CompletionInputs:
        return self._completion_coordinator.build_completion_inputs(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            call_type=call_type,
            account_conversation=account_conversation,
            purpose=purpose,
        )

    def _call_resources(
        self, backend: BackendLike, budget: RequestRetryBudget | None = None
    ) -> CallResources:
        return self._completion_coordinator.build_call_resources(backend, budget)

    def _admit_empty_replay(self) -> bool:
        if self._max_session_tokens is not None and (
            self.stats.session_total_llm_tokens > self._max_session_tokens
        ):
            return False
        # The run checkpoint guards replay admission too, so an empty replay
        # cannot bypass the per-run budget. The middleware is stateless, so a
        # throwaway instance reuses the exact checkpoint logic.
        if self._run_max_price is not None or self._run_max_session_tokens is not None:
            run_budget = RunBudgetMiddleware(
                run_max_price=self._run_max_price,
                run_max_session_tokens=self._run_max_session_tokens,
            )
            if run_budget.evaluate(self.stats).action is MiddlewareAction.STOP:
                return False
        return self._max_price is None or (
            self.stats.known_cost_total <= self._max_price
            and not self.stats.has_unknown_cost
        )

    def _append_transcript(self, outcome: TranscriptAppend) -> None:
        self._transcript_recorder.record_completion(outcome)

    def _set_execution_message(self, message: LLMMessage) -> None:
        """Retain the raw completion message for post-turn tool-call parsing."""
        self._execution_message = message

    def _note_transcript_prose(self, message: LLMMessage) -> None:
        if (
            self._active_turn is not None
            and message.role is Role.assistant
            and message.content
            and message.message_id is not None
        ):
            self._active_turn.prose_ids.append(message.message_id)

    def completion_metadata_mark(self) -> tuple[int, int]:
        """Return a cursor for completion visibility produced after this point."""
        return len(self._completion_providers), len(self._failover_switches)

    def completion_metadata_since(self, mark: tuple[int, int]) -> dict[str, JsonValue]:
        """Return additive result metadata for completions after ``mark``."""
        providers_mark, switches_mark = mark
        jobs = []
        if self.background_jobs is not None:
            with secret_redaction.bind_policy(self.scrub_policy):
                with contextlib.suppress(ValueError):
                    jobs = [
                        job.model_dump(mode="json")
                        for job in self.background_jobs.list(
                            BashListArgs(include_finished=True)
                        ).jobs
                    ]
        return cast(
            dict[str, JsonValue],
            {
                "providers_used": [
                    list(providers)
                    for providers in self._completion_providers[providers_mark:]
                ],
                "switch_notices": self._failover_switches[switches_mark:],
                **({"background_jobs": jobs} if jobs else {}),
            },
        )

    def _record_completion_providers(self, providers: Sequence[str]) -> None:
        self._completion_providers.append(tuple(providers))

    def _record_failover_switch(
        self, *, base_model: str, old_provider: str, new_provider: str, reason: str
    ) -> None:
        self._failover_switches.append({
            "base_model": base_model,
            "old_provider": old_provider,
            "new_provider": new_provider,
            "reason": reason,
        })

    def count_history_images_unsupported_by_active_model(self) -> int:
        try:
            active_model = self.config.get_active_model()
        except ValueError:
            return 0
        if active_model.supports_images:
            return 0
        return sum(1 for message in self._current_model_context() if message.images)

    def _failover_candidates(
        self,
        messages: Sequence[LLMMessage],
        attempted_model: ModelConfig,
        *,
        call_type: str | None = None,
    ) -> tuple[EligibleDeployment, ...]:
        # Compatibility delegate: candidate enumeration lives on the
        # completion coordinator; test_failover_coordinator.py drives and
        # patches this name (called :142/:146/:186/:270/:280/:332/:349/:762/
        # :822, patched :1023).
        return self._completion_coordinator.failover_candidates(
            messages, attempted_model, call_type=call_type
        )

    def _backend_for_attempt(
        self, model: ModelConfig, budget: RequestRetryBudget
    ) -> BackendLike:
        # Compatibility delegate: attempt-backend construction lives on the
        # completion coordinator; tests patch this name via MethodType
        # (test_failover_coordinator.py:368/:428/:684/:926/:1113,
        # test_usage_accounting.py:561/:665,
        # test_model_catalog_system_boundaries.py:671).
        return self._completion_coordinator.backend_for_attempt(model, budget)

    def _attempt_model(
        self, resolved: object, *, call_type: str | None, thinking: str | None = None
    ) -> ModelConfig:
        # Compatibility delegate: candidate-model materialization lives on
        # the completion coordinator; test_usage_accounting.py:544/:628 wraps
        # and patches this name (:560/:664), test_failover_coordinator.py:762
        # calls it.
        return self._completion_coordinator.attempt_model(
            resolved, call_type=call_type, thinking=thinking
        )

    def _attempt_retry_budget(
        self, budget: RequestRetryBudget, *, has_alternative: bool
    ) -> RequestRetryBudget:
        # Compatibility delegate: per-attempt deadline selection lives on the
        # completion coordinator; test_failover_coordinator.py:439/:778/:779
        # calls this name directly.
        return self._completion_coordinator.attempt_retry_budget(
            budget, has_alternative=has_alternative
        )

    def _restore_committed_model(self, identity: CommittedModelIdentity | None) -> None:
        """Restore the committed identity and reinstall launch metadata."""
        self.committed_model = identity
        self.config.attach_committed_model(identity)
        self.install_launch_metadata()

    def _commit_attempt_identity(self, identity: CommittedModelIdentity) -> None:
        """Publish the attempt identity: committed model plus config attach."""
        self.committed_model = identity
        self.config.attach_committed_model(identity)

    def _rollback_attempt(
        self,
        publication: object,
        previous_identity: CommittedModelIdentity | None,
        candidate: EligibleDeployment,
    ) -> None:
        # Compatibility delegate: publication rollback lives on the
        # completion coordinator; test_failover_coordinator.py:832 drives this
        # name directly (with install_launch_metadata patched at :830), and
        # the restore goes through the _restore_committed_model helper so the
        # patched metadata install keeps intercepting.
        self._completion_coordinator.rollback_attempt(
            publication, previous_identity, candidate
        )

    def _attempt_completion(
        self,
        *,
        messages: Sequence[LLMMessage],
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        call_type: str | None,
        transcript: bool,
        streaming: bool,
        fallback_model: ModelConfig,
        account_conversation: bool = True,
        purpose: UsagePurpose = UsagePurpose.CONVERSATION,
    ) -> AsyncGenerator[LLMChunk]:
        # Compatibility delegate: the entire attempt generator (cleanup paths
        # included) lives on the completion coordinator; the delegate returns
        # the coordinator's iterator directly, with no per-chunk forwarding
        # layer. test_failover_coordinator.py:1057 captures and patches this
        # name, so the nonstreaming/streaming facades keep resolving it on
        # the loop at call time.
        return self._completion_coordinator.attempt(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            call_type=call_type,
            transcript=transcript,
            streaming=streaming,
            fallback_model=fallback_model,
            account_conversation=account_conversation,
            purpose=purpose,
        )

    async def _attempt_nonstreaming(
        self,
        *,
        messages: Sequence[LLMMessage],
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        call_type: str | None,
        transcript: bool,
        fallback_model: ModelConfig,
        account_conversation: bool = True,
        purpose: UsagePurpose = UsagePurpose.CONVERSATION,
    ) -> LLMChunk:
        # Compatibility delegate: the nonstreaming facade lives on the
        # completion coordinator; test_failover_coordinator.py:548 patches
        # this name, so _complete/_chat keep resolving it on the loop.
        return await self._completion_coordinator.complete(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            call_type=call_type,
            transcript=transcript,
            fallback_model=fallback_model,
            account_conversation=account_conversation,
            purpose=purpose,
        )

    async def _complete(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        call_type: str | None,
        account_conversation: bool = True,
        purpose: UsagePurpose = UsagePurpose.CONVERSATION,
    ) -> LLMChunk:
        return await self._attempt_nonstreaming(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            call_type=call_type,
            transcript=False,
            fallback_model=model,
            account_conversation=account_conversation,
            purpose=purpose,
        )

    async def _chat(
        self, model_override: ModelConfig | None = None, *, call_type: str | None = None
    ) -> LLMChunk:
        active_model = model_override or self.config.get_active_model()
        return await self._attempt_nonstreaming(
            messages=self.messages,
            tools=self.format_handler.get_available_tools(self.tool_manager),
            tool_choice=self.format_handler.get_tool_choice(),
            call_type=call_type,
            transcript=True,
            fallback_model=active_model,
        )

    def _chat_streaming(self) -> AsyncGenerator[LLMChunk]:
        # The streaming facade returns the attempt generator directly (no
        # added per-chunk forwarding layer): closing the returned stream
        # closes the attempt generator itself, preserving the outer-stream
        # close contract (test_failover_coordinator.py:839), and the :1057
        # rider keeps intercepting through the loop-resolved delegate.
        active_model = self.config.get_active_model()
        return self._attempt_completion(
            messages=self.messages,
            tools=self.format_handler.get_available_tools(self.tool_manager),
            tool_choice=self.format_handler.get_tool_choice(),
            call_type=None,
            transcript=True,
            streaming=True,
            fallback_model=active_model,
        )

    def _clean_message_history(self) -> None:
        ACCEPTABLE_HISTORY_SIZE = 2
        if len(self.messages) < ACCEPTABLE_HISTORY_SIZE:
            return
        self._fill_missing_tool_responses()

    def _fill_missing_tool_responses(self) -> None:
        self._transcript_recorder.repair_missing_responses()

    def _fence_rewind_transcript(self) -> None:
        # The generation increment stays on the loop's transition fence; the
        # recorder owns the transcript cursor invalidation.
        self._session_generation += 1
        self._transcript_recorder.fence_transcript()

    async def _reset_session(self, keep_parent: bool = True) -> None:
        # Preparation (fresh identity, policy, prompt, accounting owner, and
        # lease, with the lease released on any failure) lives on the
        # session-transitions service; this synchronous commit stays on the
        # loop, as does the job retirement that precedes it.
        await self.retire_background_jobs()
        prepared = await self._session_transitions.prepare_reset(
            keep_parent=keep_parent
        )
        self.accounting_owner = prepared.owner
        self._session_generation += 1
        self.session_id = prepared.session_id
        self.parent_session_id = prepared.parent_session_id
        # A root reset binds FRESH resources (a new bundle from the current
        # selections, design §10); a subagent rewind keeps the child's own
        # bound bundle. Either way the commit installs the new generation
        # whole — a failed preparation never reaches here.
        self._bound_policy = prepared.new_policy_bundle
        self._bound_identity = prepared.new_identity
        # The same transaction rebinds the ToolManager's bound identity, so
        # effective tool authority always matches the loop's current identity
        # (a cleared conversation never keeps the previous identity's
        # posture).
        self.tool_manager._bind_identity(prepared.new_identity)
        # ...and rebinds the SkillManager to the NEW conversation's frozen
        # selections, so skill eligibility never keeps the previous
        # conversation's policy (design §3, §6).
        self.skill_manager = prepared.new_skill_manager
        self._bundle_store.put(
            self.session_id, _bundle_from_activated(prepared.new_policy_bundle)
        )
        self._bundle_store.set_resumable(self.session_id, True)
        self.install_launch_metadata()
        self.tool_manager._install_task_description(prepared.new_task_description)
        self._publish_system_prompt(prepared.new_prompt)
        self.replace_session_lease(prepared.lease)
        self._reset_title_state()
        # The reset starts a new session in place (a history clear or a
        # rewind fork): session grants, once permits, and refusal memory
        # expire with the old session and are never auto-restored across
        # the boundary, so drop them and the cached authority they fed.
        self._dangerous_command_grants.clear_session_state()
        self.tool_manager._invalidate_dangerous_command_authority()

    def replace_session_lease(self, lease: SessionLease | None) -> None:
        previous = self._session_lease
        self._session_lease = lease
        if previous is not None:
            previous.release()

    def prepare_scratchpad_for_session(self, session_id: str) -> Path | None:
        """Create the scratchpad needed to rebind without changing live state."""
        return None if self._is_subagent else init_scratchpad(session_id)

    def rebind_to_session(
        self,
        session_id: str,
        session_dir: Path,
        loaded_messages: list[LLMMessage],
        *,
        session_metadata: SessionMetadata,
        parent_session_id: str | None = None,
        stats: AgentStats | None = None,
        prepared_scratchpad: Path | None | _PrepareScratchpad = _PREPARE_SCRATCHPAD,
        accounting_owner: RootAccountingOwner | None = None,
        bound_policy: ActivatedPolicy | None = None,
    ) -> None:
        """Swap session identity in-place, reusing expensive runtime infrastructure.

        Kept (session-independent): MCP pools, the tool and skill
        registries, git context, config, and the backend client.

        Reimported from the resumed session: session ID, parent, message history,
        stats, and the session-logger binding.

        Reset so nothing leaks across the session boundary: tool-permission
        approvals, checkpoint/rewind state, the plan session, per-turn middleware
        and tool state, and the scratchpad directory.

        Atomicity: the only intentionally fail-able work (scratchpad creation)
        runs before any mutation. The commit section is designed to be infallible
        — pure assignments and resets — so a resume either fully applies or leaves
        the loop untouched. A bug in any commit step would leave the loop
        half-rebound; callers should treat an unexpected raise as fatal.

        ``bound_policy`` is the resumed conversation's activated policy bundle
        (the persisted policy record, design §10): the rebind re-installs the
        bundle's committed identity binding and serves its frozen content —
        identity validation applies to NEW bindings only.
        """
        prepared = self._session_transitions.prepare_resume(
            session_id,
            session_metadata=session_metadata,
            prepared_scratchpad=prepared_scratchpad,
            accounting_owner=accounting_owner,
            bound_policy=bound_policy,
        )
        # Commit — assignments and in-place resets only, from here on infallible.
        if self._background_job_registry is not None:
            self._background_job_registry.retire()
        self._background_job_registry = prepared.replacement_jobs
        self.background_jobs = prepared.replacement_port
        self._job_lifetime_generation += 1
        self.accounting_owner = prepared.owner
        self._session_generation += 1
        self.session_id = session_id
        self.parent_session_id = parent_session_id
        self.scratchpad_dir = prepared.scratchpad_dir
        self.tool_manager.set_scratchpad_dir(prepared.scratchpad_dir)
        # The persisted bundle is the conversation's policy record (design
        # §10): the rebind re-installs its committed identity binding and
        # keeps the metadata's bundle in step, so the resumed conversation
        # serves its frozen content even when the source files are gone.
        self._bound_policy = prepared.resumed_policy_bundle
        self._bound_identity = prepared.resumed_identity
        # The same transaction rebinds the ToolManager's bound identity, so
        # effective tool authority always matches the resumed conversation's
        # committed identity binding.
        self.tool_manager._bind_identity(prepared.resumed_identity)
        # ...and rebinds the SkillManager to the RESUMED conversation's
        # frozen selections (the persisted bundle's), so skill eligibility
        # follows the rebind like the identity binding (design §3, §6).
        self.skill_manager = prepared.resumed_skill_manager
        self._bundle_store.put(
            session_id, _bundle_from_activated(prepared.resumed_policy_bundle)
        )
        self._bundle_store.set_resumable(session_id, True)
        session_metadata.policy_bundle = _bundle_from_activated(
            prepared.resumed_policy_bundle
        ).model_dump(mode="json")
        session_metadata.policy_storage_version = POLICY_STORAGE_VERSION
        self.session_logger.apply_resumed_session(
            session_id, session_dir, session_metadata
        )
        # Atomically preserve any system prompt the deferred-init thread may
        # have inserted between snapshot and commit, instead of snapshotting
        # system messages outside the lock and racing update_system_prompt.
        self.messages.reset_preserving_system(loaded_messages)
        if stats is not None:
            self.stats = stats
        else:
            self.stats = AgentStats.create_fresh(self.stats)
            self._session_transitions.apply_active_model_pricing()
        self._reset_session_scoped_state()
        self._session_transitions.restore_todo_state(loaded_messages)
        self._session_transitions.restore_file_state(loaded_messages)
        self._session_transitions.restore_deferred_questions(loaded_messages)
        self.tool_manager._install_task_description(prepared.resumed_task_description)
        self._publish_system_prompt(prepared.resumed_prompt)
        cleanup_scratchpad(prepared.previous_scratchpad)

    def _reset_session_scoped_state(self) -> None:
        # Clear any duration the picker recorded so it doesn't leak into the
        # resumed session. ``_init_start_time`` is intentionally kept: the
        # metric measures ``__init__ -> ready``.
        self._init_duration_pending = False
        self._last_init_duration_ms = None
        self.checkpoint_recorder.reset()
        self.middleware_pipeline.reset()
        self.tool_manager.reset_all()
        self._plan_session = PlanSession()
        self._user_plan = None
        self._pending_injected_messages = []
        self._pending_clear_context = False
        self._current_user_message_id = None
        self._is_user_prompt_call = False
        self._reactive_recovery_used = False
        self._reset_title_state()
        # A resume rebinds the loop in place: session grants, once permits,
        # and refusal memory expire with the previous session and are never
        # auto-restored across the boundary, so drop them and the cached
        # authority they fed.
        self._dangerous_command_grants.clear_session_state()
        self.tool_manager._invalidate_dangerous_command_authority()

    def _publish_fresh_stats(self) -> None:
        self.stats = AgentStats.create_fresh(self.stats)
        self.stats.trigger_listeners()

    @requires_init
    async def clear_history(self) -> None:
        # The clear orchestration (save, reset, trim, fresh stats, pricing,
        # middleware/tool/file-state reset) lives on the session-transitions
        # service; the loop keeps this name because tests capture and patch
        # the instance attribute (tests/app_server/test_subagents.py:5854).
        await self._session_transitions.clear_history()

    @requires_init
    async def compact(self, extra_instructions: str = "") -> str:
        # Summary generation and the context boundary live in the manager; the loop
        # keeps the surrounding lifecycle (clean history, save, middleware).
        try:
            self._clean_message_history()
            await self._save_messages()
            summary = await self.compaction_manager.compact(extra_instructions)
            self.middleware_pipeline.reset(reset_reason=ResetReason.COMPACT)
            # A compacted conversation reads very differently from its first
            # turn; force a title refresh at the next scheduling point.
            self._title_controller.mark_compaction()
            return summary
        except Exception:
            await self._save_messages()
            raise

    async def _request_clear_context(self) -> None:
        """Signal that the context should be cleared at the next turn boundary.

        The actual clear is deferred so the in-flight tool turn can finish
        appending its tool-result before ``self.messages`` is wiped.
        """
        self._pending_clear_context = True

    @requires_init
    async def relocate(self, cwd: Path) -> None:
        """Move the session to *cwd*, or leave it exactly where it was.

        Refused while a turn is active. Tool calls in a batch run as concurrent
        tasks, so a move alongside a file or shell call would leave that call
        writing to the old directory, and swapping the objects it holds cannot
        stop one already running. A subagent runs inside its parent's turn, so
        this covers those too.

        Refused as well while an operation holds the session. A turn is not
        the only thing that reads the working directory or acts on its
        repository, and the rest are invisible to the turn check.

        Raises:
            AgentLoopStateError: Something is in flight, or *cwd* is not a
                directory.
        """
        # The relocation transaction (reservation, destination validation,
        # bind/restore, move-acquired trust tracking) lives on the workspace
        # relocator; the loop keeps this name because the app server calls it
        # and tests patch it (tests/app_server/test_relocate.py:192).
        await self._workspace_relocator.relocate(cwd)

    def _destination_checkout(self, target: Path) -> Path | None:
        # Kept for the instance patch seam (tests/agent_loop/
        # test_background_jobs.py:44); the relocator resolves it late-bound.
        return self._workspace_relocator.destination_checkout(target)

    def _assign_workspace(self, cwd: Path, harness_files: HarnessFilesManager) -> None:
        """Point the loop's workspace roots at *cwd* (the relocator's seam)."""
        self.cwd = cwd
        self.harness_files = harness_files

    async def _bind_workspace(
        self, cwd: Path, harness_files: HarnessFilesManager
    ) -> None:
        # Kept for the instance patch seam (tests/agent_loop/
        # test_background_jobs.py:141); the relocator resolves it late-bound.
        await self._workspace_relocator.bind_workspace(cwd, harness_files)

    @requires_init
    async def reload_with_initial_messages(
        self,
        max_turns: int | None = None,
        max_price: float | None = None,
        reset_middleware: bool = True,
        reload_hooks: bool = False,
        reload_config: bool = False,
        session_save_config: ChartreuxConfigSchema | None = None,
    ) -> None:
        await _config_transactions.reload_with_initial_messages(
            self,
            max_turns,
            max_price,
            reset_middleware,
            reload_hooks,
            reload_config,
            session_save_config,
        )

    async def _prepare_policy_replacement(
        self, *, source: str, tools: dict[str, Any], expected_token: object
    ) -> _PreparedPolicyReplacement:
        return await _config_transactions._prepare_policy_replacement(
            self, source=source, tools=tools, expected_token=expected_token
        )

    async def _prepare_root_replacement(
        self, *, source: str, roots: dict[str, list[str]], expected_token: object
    ) -> _PreparedPolicyReplacement:
        return await _config_transactions._prepare_root_replacement(
            self, source=source, roots=roots, expected_token=expected_token
        )

    def _require_policy_idle(self) -> None:
        _config_transactions._require_policy_idle(self)

    async def _prepare_policy_refresh(
        self,
        *,
        previous: SourceRestrictions,
        replacement: SourceRestrictions,
        expected_token: object,
        inherited_workspace: Workspace | None = None,
    ) -> _PreparedPolicyReplacement:
        return await _config_transactions._prepare_policy_refresh(
            self,
            previous=previous,
            replacement=replacement,
            expected_token=expected_token,
            inherited_workspace=inherited_workspace,
        )

    def _prepare_policy_manager(
        self,
        staged: ConfigOrchestrator[ChartreuxConfigSchema],
        expected_token: object,
        previous_manager: ToolManager,
        session_generation: int,
        inherited: tuple[SourceRestrictions, ...],
        *,
        inherited_workspace: Workspace | None = None,
    ) -> _PreparedPolicyReplacement:
        return _config_transactions._prepare_policy_manager(
            self,
            staged,
            expected_token,
            previous_manager,
            session_generation,
            inherited,
            inherited_workspace=inherited_workspace,
        )

    def _guard_job_policy(
        self,
        staged: ConfigOrchestrator[ChartreuxConfigSchema],
        inherited: tuple[SourceRestrictions, ...],
        workspace: Workspace | None,
    ) -> None:
        _config_transactions._guard_job_policy(self, staged, inherited, workspace)

    def _validate_policy_replacement(
        self, prepared: _PreparedPolicyReplacement
    ) -> None:
        _config_transactions._validate_policy_replacement(self, prepared)

    def _commit_policy_replacement(
        self, prepared: _PreparedPolicyReplacement, *, retire: bool = True
    ) -> None:
        _config_transactions._commit_policy_replacement(self, prepared, retire=retire)

    def _prepare_reload(
        self,
        target_config: ChartreuxConfigSchema,
        reload_hooks: bool,
        *,
        mcp_registry: MCPRegistry | None = None,
    ) -> _PreparedReload:
        return _config_transactions._prepare_reload(
            self, target_config, reload_hooks, mcp_registry=mcp_registry
        )

    def _thinking_for_committed_reload(
        self, config: ChartreuxConfigSchema, identity: CommittedModelIdentity
    ) -> str:
        return _config_transactions._thinking_for_committed_reload(
            self, config, identity
        )

    def _prepare_reload_consumers(
        self,
        target_config: ChartreuxConfigSchema,
        reload_hooks: bool,
        *,
        mcp_registry: MCPRegistry | None = None,
        discovery_source: ToolManager | None = None,
        reuse_backend: bool = False,
    ) -> _PreparedReload:
        return _config_transactions._prepare_reload_consumers(
            self,
            target_config,
            reload_hooks,
            mcp_registry=mcp_registry,
            discovery_source=discovery_source,
            reuse_backend=reuse_backend,
            hooks_loader=lambda: load_hooks_from_fs(harness_files=self.harness_files),
        )

    def _commit_reload(self, prepared: _PreparedReload, reset_middleware: bool) -> None:
        _config_transactions._commit_reload(self, prepared, reset_middleware)
