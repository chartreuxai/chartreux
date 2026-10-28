from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from chartreux.core.agents.identities import get_builtin_identity
from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    get_harness_files_manager,
)
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.core.agents.launch import LaunchCandidate
    from chartreux.core.config import ChartreuxConfigSchema
    from chartreux.core.dispatch.route_models import PurposeRoute
    from chartreux.core.recovery_override import CooldownOverrideLedger, OverridePermit


def identity_denylist_keys(identity: str) -> tuple[str, ...]:
    """Return equivalent bare and qualified spellings for denylist matching.

    The denylist is SYMMETRIC across spellings (review finding 1): a pattern
    written in any equivalent spelling blocks the identity in every
    spelling. A ``user:``-qualified custom identity also matches its bare
    name; a ``builtin:``-qualified identity also matches its bare name; a
    bare built-in also matches its ``builtin:``-qualified form; and a bare
    custom id also matches its ``user:``-qualified form — so denying
    ``coder`` blocks ``builtin:coder``, denying ``builtin:coder`` blocks
    ``coder``, denying ``user:foo`` blocks bare ``foo``, and denying ``foo``
    blocks ``user:foo``.
    """
    if identity.startswith("user:"):
        return (identity, identity.removeprefix("user:"))
    if identity.startswith("builtin:"):
        return (identity, identity.removeprefix("builtin:"))
    if get_builtin_identity(identity) is not None:
        return (identity, f"builtin:{identity}")
    return (identity, f"user:{identity}")


def requested_identity_reference(args: Any) -> str | None:
    """The launch identity a canonical task call names, if any.

    Precedence: the structured ``identity`` field, then a model shorthand's
    identity component (``@fast:researcher``). ``None`` when the call names no
    launch target (a plain retained continuation).
    """
    if getattr(args, "identity", None) is not None:
        return args.identity
    model = getattr(args, "model", None)
    if model is not None:
        from chartreux.core.agents.identity_launch import parse_model_shorthand

        shorthand = parse_model_shorthand(model)
        if shorthand is not None and shorthand.identity is not None:
            return shorthand.identity
    return None


def resolve_identity_candidate(
    args: Any,
    *,
    parent_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    tool_inventory: Mapping[str, object],
    authorized_tool_names: frozenset[str],
    history: Sequence[Any] = (),
    session_model: str | None = None,
    user_prompts_dirs: Sequence[Any] | None = None,
    routes: Mapping[str, PurposeRoute] | None = None,
    override_permit: OverridePermit | None = None,
    override_ledger: CooldownOverrideLedger | None = None,
) -> LaunchCandidate:
    """Resolve one fresh child launch through the canonical grammar (11c).

    Package 8's resource-free resolution is the resolution mechanism: the
    identity (and its revision), the model selection, the effective thinking,
    and the authority axes come from :func:`resolve_identity_launch`. The
    returned :class:`LaunchCandidate` is assembled by the existing deep
    validation (history compatibility, availability, thinking encoding, and
    the two-ceil tool-override admission): an explicit canonical model
    selection is injected as the explicit launch model, and the
    session-default selection hands its canonically committed deployment
    through the committed-model path.

    ``routes`` is the purpose-route table the resolution consumes. Launch
    admission passes the conversation's LIVE route table — the in-memory
    ``[orchestration.routes]`` table the reload swaps atomically, the same
    source the rendered launch-target table carries — so a validated route
    edit applies to every launch admitted after the reload commit, while
    already-admitted launches keep their pinned bindings (ADR 0022).
    ``None`` keeps the live config's table for direct, conversation-less
    callers.
    """
    from chartreux.core.agents.identity_launch import resolve_identity_launch
    from chartreux.core.agents.launch import resolve_launch
    from chartreux.core.launch_types import LaunchConfig

    canonical_args = args
    if (
        args.model is None
        and args.config is not None
        and "model" in args.config.model_fields_set
    ):
        # The schema's documented promotion: the legacy nested config.model is
        # the canonical top-level model selection (a dual supply is rejected by
        # the schema itself).
        canonical_args = args.model_copy(update={"model": args.config.model})
    resolved = resolve_identity_launch(
        canonical_args,
        config=parent_orchestrator.config,
        authorized_tool_names=authorized_tool_names,
        session_model=session_model,
        user_prompts_dirs=user_prompts_dirs,
        routes=routes,
    )
    semantic = args.config
    data = (
        semantic.model_dump(exclude_unset=True, mode="python")
        if semantic is not None
        else {}
    )
    if resolved.overlay_body is not None:
        # The matched route's overlay rides the persona instructions (design
        # §9: the execution seat is coder plus a reviewer-derived overlay):
        # it joins the frozen persona/bundle content — the identity binding
        # and its revision are untouched, and additive per-launch instructions
        # stay ahead of the overlay.
        base_instructions = data.get("instructions")
        data["instructions"] = (
            f"{base_instructions}\n\n{resolved.overlay_body}"
            if base_instructions
            else resolved.overlay_body
        )
    if canonical_args.model is not None or canonical_args.purpose is not None:
        # An explicit model selection (structured, shorthand, promoted
        # config.model, or a purpose route) is the explicit launch selection:
        # the deep validation re-resolves the expression with the
        # availability-aware candidate filter.
        data["model"] = resolved.model_expression
        committed = None
    else:
        # The session-default selection hands its canonically committed
        # deployment to the deep validation (the committed-model path, which
        # does not re-select against availability cooldowns — the recovery
        # controller measures those as launch failures, it does not predict
        # them).
        committed = resolved.committed_model
    return resolve_launch(
        identity=resolved.identity,
        config=LaunchConfig.model_validate(data),
        tool_inventory=tool_inventory,
        parent_orchestrator=parent_orchestrator,
        authorized_tool_names=authorized_tool_names,
        history=history,
        committed_model=committed,
        purpose=resolved.purpose,
        # The resolved effective tier and replacement capability ride the
        # candidate: the recovery boundary reads the tier through the child
        # loop's launch metadata and the agent record, so a package launched
        # through a mechanical-tier route is recovered as mechanical.
        effective_tier=resolved.effective_tier,
        replacement_capability=resolved.replacement_capability,
        # C4: an explicitly requested un-stick launch resolves its model
        # THROUGH the minted permit, so the cooled forced deployment stays
        # assignable; every other admission check still applies.
        override_permit=override_permit,
        override_ledger=override_ledger,
    )


class AgentManager:
    """The loop's identity-surface manager: configuration and launch facts.

    The profile registry is retired with the profile axis (package 16a):
    agents/*.toml discovery, profile lookup, and the enabled/disabled agent
    filters are gone, and custom identities resolve from the user prompts
    directory through the package-1 strict loader instead. What remains is
    the handle the loop and the tool context carry for configuration access,
    and loud detection of retired on-disk inputs (never consumed, never
    deleted or converted). The retired dispatch activation validation is gone
    with the dispatch axis (package 16b): routing resolves through
    ``[orchestration.routes.*]`` at launch, and a legacy ``[dispatch]`` table
    is reported loudly by the catalog loader.
    """

    def __init__(
        self,
        orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        harness_files: HarnessFilesManager | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._harness_files = harness_files or get_harness_files_manager()
        self._warn_retired_inputs()

    def _warn_retired_inputs(self) -> None:
        """Report retired on-disk inputs loudly; user files are never touched."""
        retired_agents = self._harness_files.retired_agent_dirs
        if retired_agents:
            logger.warning(
                "Agent profiles are retired and no longer load: found agents/*.toml "
                "in %s. Use identities instead — select [agent] identity in "
                "config.toml or pass the task tool's identity field; custom "
                "identities are prompt files in the user prompts directory. The "
                "profile files are left untouched.",
                ", ".join(str(path) for path in retired_agents),
            )
        retired_prompts = self._harness_files.retired_project_prompt_dirs
        if retired_prompts:
            logger.warning(
                "Project-level custom prompt directories are retired and no "
                "longer load: found %s. Move custom identity prompts to the user "
                "prompts directory; workflows and orchestration strategies are "
                "user-level too. The files are left untouched.",
                ", ".join(str(path) for path in retired_prompts),
            )

    @property
    def config(self) -> ChartreuxConfigSchema:
        return self._orchestrator.config
