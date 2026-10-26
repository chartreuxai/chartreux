from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterator
from contextlib import AbstractAsyncContextManager, contextmanager, suppress
from dataclasses import dataclass, replace
import fnmatch
from typing import TYPE_CHECKING, Protocol

from chartreux.app_server._subagent_records import AgentRecord
from chartreux.app_server._subagent_recovery import _effective_eligibility
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agents.identity_launch import (
    parse_model_shorthand,
    validate_launch_purpose_binding,
)
from chartreux.core.agents.launch import FrozenPersona, LaunchCandidate, resolve_launch
from chartreux.core.agents.manager import (
    identity_denylist_keys,
    requested_identity_reference,
    resolve_identity_candidate,
)
from chartreux.core.dispatch.renderer import session_model_alias
from chartreux.core.dispatch.route_models import ReplacementCapability
from chartreux.core.model_catalog.resolver import ModelResolutionError, resolver_for
from chartreux.core.recovery_models import (
    FailureEnvelope,
    PackageTier,
    RecoveryEligibility,
    canonical_endpoint_identity,
)
from chartreux.core.recovery_policy import (
    ReplacementCandidate,
    ReplacementDecision,
    ReplacementPolicyError,
    candidates_from_catalog,
    route_recovery_facts,
    validate_replacement_selection,
)
from chartreux.core.session.saved_sessions import delete_saved_session
from chartreux.core.subagents import (
    InvalidLaunchModelError,
    LaunchConfig,
    LaunchConfigError,
    RunStatus,
    TaskArgs,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.models import ToolPermission, ToolPermissionError
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server._runtime import AgentRuntimeFactory
    from chartreux.app_server._sessions import SessionRuntime

_AGENT_ID_HIGH_WATER_MARK_KEY = "_app_server_agent_id_high_water_mark"


class LaunchStateView(Protocol):
    """Narrow typed view over the registry-owned launch state.

    Dynamic access: every member is read through the view at call time, so
    replaceable scalars (root identity, generation, admission flags, the
    runtime factory) and the sequence counter are never snapshotted at
    construction — registry state mutated or rebound after the
    collaborator is built stays visible, and instance patches of the
    registry delegates still shadow.
    """

    _root: SessionRuntime | None
    _generation_identity: tuple[str, int] | None
    _admission_closed: bool
    _draining_children: bool
    _policy_reserved: bool
    _creating_children: int
    _children: dict[str, SessionRuntime]
    _child_links: dict[str, tuple[SessionRuntime, str]]
    _next_agent_seq: int
    _runtime_factory: AgentRuntimeFactory
    _agent_records: dict[str, AgentRecord]
    _package_replacements: dict[str, int]
    _registry_lock: asyncio.Lock

    def _require_policy_unreserved(self) -> None: ...

    def _resolve_launch_candidate(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        *,
        runtime: SessionRuntime | None = None,
        identity: str | None = None,
    ) -> LaunchCandidate: ...

    def _build_child_runtime(self, child: AgentLoop) -> SessionRuntime: ...

    def _discard_child(self, child: AgentLoop) -> Awaitable[None]: ...

    def _child_publication(
        self, runtime: SessionRuntime
    ) -> AbstractAsyncContextManager[None]: ...

    def _invalidate_recovery_locked(self, record: AgentRecord | None) -> None: ...


@dataclass
class ChildPublicationAttempt:
    """Per-invocation state for one child publication (ASC area 1).

    Holds the parent runtime and captured identity, the resolved launch
    candidate and tool-call id, the constructed child/runtime once they
    exist, and the two transaction flags the rollback region reads. The
    registry-owned child maps stay canonical; the attempt carries only
    the publication transaction's local state.
    """

    parent: SessionRuntime
    identity: tuple[str, int]
    candidate: LaunchCandidate
    tool_call_id: str
    child: AgentLoop | None = None
    runtime: SessionRuntime | None = None
    link_recorded: bool = False
    projection_started: bool = False

    async def rollback(self, view: LaunchStateView) -> None:
        """Remove every durable/runtime trace of an unaccepted publication.

        Replaces the local transaction flags of the former registry method
        without widening the cleanup region or changing exception
        suppression.
        """
        child = self.child
        runtime = self.runtime
        assert child is not None and runtime is not None
        session_id = child.session_id
        view._children.pop(session_id, None)
        view._child_links.pop(session_id, None)
        if self.projection_started:
            with suppress(Exception):
                await self.parent.turns.unlink_subagent(self.tool_call_id, session_id)
        if self.link_recorded:
            with suppress(Exception):
                await self.parent.agent_loop.forget_child_session(
                    session_id, self.tool_call_id
                )
        with suppress(Exception):
            await runtime.close()
        with suppress(Exception):
            await delete_saved_session(session_id, child.config.session_logging)


class ChildLaunchAdmission:
    """Launch admission, publication, and rollback for child sessions.

    Narrow collaborator of the session runtime registry (ASC area 1).
    Identity validation and candidate resolution (17.1a) are pure functions
    of the parent runtime and the launch arguments: they read no registry
    state, take no locks, and hold no state of their own. Sequence
    allocation and restoration, admission validation, the creation fence,
    registered-child publication, and created-child rollback (17.1b) run
    over the registry-owned typed view, which also supplies the runtime
    construction, child-publication context, and discard dependencies with
    call-time dynamic access. Canonical child maps, admission flags, and
    the sequence counter stay registry-owned (§6.7); only the existing
    registry locks are used; no private collaborator locks, tasks, queues,
    or relay layers. The registry keeps thin same-name delegates so
    existing instance patches and direct calls keep working.
    """

    def __init__(self, view: LaunchStateView) -> None:
        self._view = view

    def require_identity_allowed(self, parent: SessionRuntime, identity: str) -> None:
        """Apply effective Task admission to the RESOLVED launch identity.

        The denylist keys on the same identity surface the task tool enforces
        (design §12): a resolved identity reference — including route/shorthand
        selections and a retained instance's binding — checked after final
        resolution and before any child resource is constructed. A
        ``user:``-qualified custom identity also matches its bare name.
        """
        manager = parent.agent_loop.tool_manager
        if isinstance(manager, ToolManager) and "task" not in manager.available_tools:
            raise ToolPermissionError("Task tool is disabled")
        config = manager.get_tool_config("task")
        if (
            isinstance(manager, ToolManager)
            and config.permission == ToolPermission.NEVER
        ):
            raise ToolPermissionError("Task tool permission is never")
        if any(
            fnmatch.fnmatch(key, pattern)
            for key in identity_denylist_keys(identity)
            for pattern in config.denylist
        ):
            raise ToolPermissionError(f"Task denied for launch identity: {identity}")

    def resolve_candidate(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        *,
        runtime: SessionRuntime | None = None,
        identity: str | None = None,
    ) -> LaunchCandidate:
        """Resolve the launch candidate for a fresh or reused child.

        A FRESH launch (no retained runtime) resolves through the canonical
        grammar (package 8's resource-free resolution): the identity, the
        model expression, the effective thinking, and the authority axes
        come from ``resolve_identity_launch`` — the legacy profile path no
        longer selects them (11c cutover). A RETAINED launch keeps the
        committed-model/frozen-persona machinery: the record's identity
        revision and overlay stay immutable across the retask — the retained
        identity is the child loop's committed launch binding, falling back
        to the record's identity and the child's installed selection. The
        retained launch still goes through the SAME canonical resolution as
        a fresh launch for everything it selects: the top-level model field
        is the canonical model selection (the schema promotes a legacy
        nested config.model), and a supplied purpose/seat binding is
        validated through the route grammar — never copied from the previous
        run, never accepted unvalidated.
        """
        child = runtime.agent_loop if runtime is not None else None
        if child is None:
            candidate = resolve_identity_candidate(
                args,
                parent_orchestrator=parent.agent_loop.config_orchestrator,
                tool_inventory={
                    name: object()
                    for name in parent.agent_loop.tool_manager.registered_tools
                },
                # The admission ceiling is the parent's DELEGATION set (granted
                # configuration ∩ the parent identity's delegation posture ∩
                # every ancestor's delegation ceiling), not its personal
                # inventory: personal posture and delegation are independent
                # axes.
                authorized_tool_names=frozenset(
                    parent.agent_loop.tool_manager.delegable_tools
                ),
                session_model=parent.agent_loop.config.active_model,
                user_prompts_dirs=parent.agent_loop.harness_files.user_prompts_dirs,
                # Purpose routes resolve against the conversation's LIVE
                # route table — the in-memory table the reload swaps
                # atomically, the same source the rendered launch-target
                # table carries — so a validated route edit applies to
                # every launch admitted after the reload commit (ADR 0022).
            )
            claim = args.config
            if claim is not None and claim.predecessor_agent_id is not None:
                # A replacement claim carries the package's validated
                # effective tier end to end: admission judged the claim
                # against the claimed tier or the predecessor's established
                # tier, so the successor's record and loop carry that tier
                # instead of the direct-launch default.
                inherited_tier = claim.package_tier
                if inherited_tier is None:
                    predecessor = self._view._agent_records.get(
                        claim.predecessor_agent_id
                    )
                    if predecessor is not None:
                        inherited_tier = predecessor.package_tier
                if inherited_tier is not None:
                    candidate = replace(candidate, effective_tier=inherited_tier)
            return candidate
        retained_identity = (
            child.launch_identity or identity or child.config.agent.identity
        )
        retained_config = args.config
        if args.model is not None:
            # The top-level model field is the canonical selection on every
            # launch path: promote it over the retained semantic config (a
            # dual supply is rejected by the schema itself). The shorthand
            # grammar normalizes exactly like fresh resolution: the model
            # component selects the model, and the identity component is
            # checked against the retained record's committed identity by
            # the caller's retained-identity match.
            model_expression = args.model
            shorthand = parse_model_shorthand(args.model)
            if shorthand is not None:
                model_expression = shorthand.model
            retained_config = LaunchConfig.model_validate({
                **(
                    retained_config.model_dump(exclude_unset=True, mode="python")
                    if retained_config is not None
                    else {}
                ),
                "model": model_expression,
            })
        if args.purpose is not None:
            # The retained instance's committed identity/model are pinned by
            # its record, so the route's own defaults are validated for
            # availability only — never adopted — but an invalid purpose or
            # seat binding fails loudly instead of being accepted. The
            # binding validates against the conversation's LIVE route table,
            # so a purpose whose route was edited by a reload is validated
            # exactly as a fresh launch validates it (ADR 0022).
            validate_launch_purpose_binding(
                args.purpose,
                config=parent.agent_loop.config,
                seat=args.seat,
                session_model=parent.agent_loop.config.active_model,
            )
        return resolve_launch(
            identity=retained_identity,
            config=retained_config,
            parent_orchestrator=child.config_orchestrator,
            purpose=args.purpose,
            tool_inventory={
                name: object()
                for name in parent.agent_loop.tool_manager.registered_tools
            },
            # The admission ceiling is the parent's DELEGATION set (granted
            # configuration ∩ the parent identity's delegation posture ∩ every
            # ancestor's delegation ceiling), not its personal inventory:
            # personal posture and delegation are independent axes.
            authorized_tool_names=frozenset(
                parent.agent_loop.tool_manager.delegable_tools
            ),
            history=tuple(child.messages),
            accumulated_overrides=child.launch_overrides,
            committed_model=child.committed_model,
            frozen_persona=(
                FrozenPersona(instructions=child.frozen_instructions)
                if child.frozen_instructions is not None
                else None
            ),
            # The retained instance's committed tier/capability ride the
            # retask's candidate, so the launch metadata the recovery
            # boundary reads keeps describing the package's true tier.
            effective_tier=child.launch_package_tier or PackageTier.SUBSTANTIVE,
            replacement_capability=(
                child.launch_replacement_capability or ReplacementCapability.NONE
            ),
        )

    def validate_admission(
        self, parent: SessionRuntime, identity: tuple[str, int]
    ) -> None:
        """Reject a launch whose admission window closed or identity changed."""
        live_identity = (
            parent.agent_loop.session_id,
            parent.agent_loop._session_generation,
        )
        admission_blocked = (
            self._view._admission_closed or self._view._draining_children
        )
        identity_changed = (
            self._view._root is not parent
            or self._view._generation_identity != identity
            or live_identity != identity
        )
        if admission_blocked or identity_changed or parent._closed is True:
            message = (
                "Agent admission is closed"
                if self._view._admission_closed
                else "Agent admission changed during child creation"
            )
            raise RuntimeError(message)

    @staticmethod
    def _claim_shape_guards(args: TaskArgs, claim: LaunchConfig) -> None:
        """Claim-shape guards: a NEW agent_id, an explicit model, an explicit package."""
        if args.agent_id is not None:
            raise LaunchConfigError(
                "config.predecessor_agent_id",
                "a replacement claim launches a NEW agent: agent_id must be None",
            )
        if args.model is None and claim.model is None:
            # The canonical top-level model field and the legacy nested
            # config.model are the same selection (the schema promotes the
            # nested form); a dual supply is a schema validation error.
            raise LaunchConfigError(
                "model",
                "a replacement claim must name the replacement model explicitly "
                "(the top-level model field or config.model)",
            )
        if claim.work_package_id is None:
            raise LaunchConfigError(
                "config.work_package_id",
                "a replacement claim must name its work package: the per-package "
                "replacement budget is keyed on work_package_id",
            )

    def _predecessor_envelope_locked(
        self, parent: SessionRuntime, predecessor_id: str
    ) -> tuple[AgentRecord, FailureEnvelope]:
        """The owned predecessor record and its latest failed run's envelope."""
        record = self._view._agent_records.get(predecessor_id)
        if record is None:
            raise ReplacementPolicyError(
                ["predecessor_unknown"],
                f"Replacement claim rejected: predecessor agent "
                f"{predecessor_id} is not a live record (closed/evicted "
                "agents cannot be replaced); re-dispatch the package fresh",
            )
        if record.parent_session_id != parent.agent_loop.session_id:
            raise ReplacementPolicyError(
                ["predecessor_not_owned"],
                f"Replacement claim rejected: predecessor agent "
                f"{predecessor_id} is not owned by this parent session",
            )
        # The failed run's envelope, latest-first over the current run
        # and the run history.
        failed_run = next(
            (
                run
                for run in (record.current_run, *reversed(record.run_history))
                if run is not None
                and run.status is RunStatus.FAILED
                and run.failure is not None
            ),
            None,
        )
        if failed_run is None or failed_run.failure is None:
            raise ReplacementPolicyError(
                ["predecessor_has_no_failure_envelope"],
                f"Replacement claim rejected: predecessor agent "
                f"{predecessor_id} has no failed run with a failure envelope",
            )
        envelope = failed_run.failure
        # Review R6: the envelope must come from the predecessor's LATEST
        # run, and that run must be terminal — a claim never launches a
        # replacement alongside a resumed successor or after a newer run
        # already displaced the failed one.
        if (
            failed_run.run_id != record.latest_run_id
            or record.current_run is not None
            and record.current_run.run_id != failed_run.run_id
        ):
            raise ReplacementPolicyError(
                ["predecessor_run_not_terminal"],
                f"Replacement claim rejected: predecessor agent "
                f"{predecessor_id} has a newer run than its failed run "
                f"{failed_run.run_id}; the failure is not the latest "
                "terminal state, so the package has no failed run to "
                "replace",
            )
        return record, envelope

    @staticmethod
    def _check_lineage_and_barrier_locked(
        record: AgentRecord, claim: LaunchConfig, envelope: FailureEnvelope
    ) -> None:
        """Lineage match-or-establish plus the reconciliation barrier (locked).

        A predecessor that already carries package lineage (it was itself a
        replacement) pins the claim to it; otherwise the claim establishes
        the lineage. The write-ownership transfer happens only after
        reconciliation — a suspended barrier that still requires
        reconcile_result blocks the claim; verified unrepeatable damage
        prohibits any repeat of the package.
        """
        if (
            record.work_package_id is not None
            and record.work_package_id != claim.work_package_id
        ):
            raise ReplacementPolicyError(
                ["work_package_lineage_mismatch"],
                f"Replacement claim rejected: package "
                f"{claim.work_package_id} does not match the predecessor's "
                f"established package {record.work_package_id}",
            )
        if (
            record.package_tier is not None
            and claim.package_tier is not None
            and record.package_tier is not claim.package_tier
        ):
            raise ReplacementPolicyError(
                ["package_tier_lineage_mismatch"],
                f"Replacement claim rejected: tier {claim.package_tier.value} "
                f"does not match the predecessor's established tier "
                f"{record.package_tier.value}",
            )
        eligibility, _codes = _effective_eligibility(record, envelope)
        if eligibility is RecoveryEligibility.RECONCILE_REQUIRED:
            raise ReplacementPolicyError(
                ["reconcile_required_before_replacement"],
                f"Replacement claim rejected: predecessor agent "
                f"{record.agent_id} is suspended with a reconcile-required "
                "barrier; submit reconcile_result first — the transfer "
                "happens only after reconciliation",
            )
        if eligibility is RecoveryEligibility.PROHIBITED:
            raise ReplacementPolicyError(
                ["recovery_prohibited"],
                f"Replacement claim rejected: predecessor agent "
                f"{record.agent_id} has verified unrepeatable damage; the "
                "package is not repeatable on any model",
            )

    def _validate_claim_policy_locked(
        self,
        parent: SessionRuntime,
        record: AgentRecord,
        claim: LaunchConfig,
        envelope: FailureEnvelope,
        *,
        claim_model: str,
        model_field: str,
    ) -> ReplacementDecision:
        """Per-package budget + policy validation (locked); returns the decision.

        The budget spans the package lineage (keyed on work_package_id);
        exhaustion rejects with ``package_replacements_exhausted`` so the
        orchestrator parks the package. The policy validation resolves the
        claimed model (the canonical top-level model field, or the legacy
        nested config.model the schema promotes), builds the claimed
        candidate, and runs the full failure-time decision procedure —
        endpoint independence against the ATTEMPTED failed endpoint, tier
        capability, allowlist/price cap.
        """
        recovery = parent.agent_loop.config.recovery.subagents
        package_key = claim.work_package_id
        assert package_key is not None  # the claim-shape guard ran first
        used = self._view._package_replacements.get(package_key, 0)
        if used >= recovery.fallback_replacements_max_per_package:
            raise ReplacementPolicyError(
                ["package_replacements_exhausted"],
                f"Replacement claim rejected: package {package_key} has "
                f"used its budget of "
                f"{recovery.fallback_replacements_max_per_package} "
                "replacements; park the package and continue independent work",
            )
        config = parent.agent_loop.config
        try:
            resolver = resolver_for(config)
            resolved = resolver.resolve(claim_model)
        except ModelResolutionError as exc:
            raise InvalidLaunchModelError(model_field, str(exc)) from exc
        # The recovery boundary labels candidates by canonical base model,
        # so route preset references canonicalize through the catalog
        # mapping; a target's explicit replacement-capability marking rides
        # the claimed candidate and the candidate set — a separate axis from
        # the package tier, and absent markings keep the legacy tier-label
        # derivation (the shipped routes mark nothing). The facts derive
        # from the conversation's LIVE route table — the same source launch
        # admission resolves against (ADR 0022).
        mechanical_models, capability_markings = route_recovery_facts(
            config.purpose_routes.routes,
            session_model=session_model_alias(config),
            canonical_models=resolver.canonical_model_map(),
            catalog_prices=resolver.catalog_prices(),
        )
        claimed = ReplacementCandidate(
            base_model=resolved.base_model,
            provider=resolved.deployment.provider,
            wire_name=resolved.deployment.name,
            endpoint=canonical_endpoint_identity(resolved.provider.api_base),
            tier=(
                PackageTier.MECHANICAL
                if resolved.base_model in mechanical_models
                else PackageTier.SUBSTANTIVE
            ),
            capability=(
                capability_markings.get(resolved.base_model, ReplacementCapability.NONE)
                if capability_markings
                else None
            ),
            output_price_per_million=resolved.deployment.prices.output,
        )
        snapshot = config.catalog_snapshot
        assert snapshot is not None  # resolver_for raised otherwise
        candidates = candidates_from_catalog(
            snapshot.catalog,
            mechanical_base_models=mechanical_models,
            capabilities=(capability_markings or None),
        )
        effective_tier = (
            claim.package_tier
            if claim.package_tier is not None
            else (
                record.package_tier
                if record.package_tier is not None
                else PackageTier.SUBSTANTIVE
            )
        )
        decision = validate_replacement_selection(
            envelope=envelope,
            package_tier=effective_tier,
            claimed=claimed,
            candidates=candidates,
            config=recovery,
        )
        if decision.disposition != "selected":
            raise ReplacementPolicyError(
                decision.reason_codes,
                decision.notice
                or f"Replacement claim rejected for package {package_key}",
            )
        return decision

    async def validate_replacement_claim(
        self, parent: SessionRuntime, args: TaskArgs
    ) -> ReplacementDecision:
        """Validate a WP5 failure-time replacement claim at launch admission.

        The orchestrator claims a replacement by launching a NEW agent
        (``args.agent_id is None``): the replacement model is named by the
        canonical top-level ``model`` field (the legacy nested
        ``config.model`` is promoted to it by the schema), and ``config``
        carries the claim: ``predecessor_agent_id`` (the failed agent),
        ``work_package_id`` (the package identity the per-package budget is
        keyed on), and ``package_tier`` (mechanical/substantive; unknown
        defaults conservatively to substantive). There is NO dispatch-time
        substitution table and NO automatic cross-model fallback: the claim
        is validated HERE, against the predecessor's failed run's failure
        envelope, at failure time.

        Admission order (the write-ownership handoff): claim-shape guards →
        predecessor ownership + failed-run envelope → lineage
        match-or-establish → reconciliation barrier → per-package budget →
        policy validation (endpoint independence against the ATTEMPTED failed
        endpoint, tier capability, allowlist/price cap) → canonical identity
        resolution + Task admission. Only then is the old recovery
        invalidated and the package counter incremented — BEFORE the new
        run's write token is registered at accept_background, so the
        transfer happens only after reconciliation state is settled and the
        explicit handoff is admitted. Review finding 2: the identity
        resolution and admission COMPLETE before any consequential
        replacement-state change — an invalid or denied launch identity
        leaves zero side effects (recovery intact, budget unconsumed, no
        supersession). Budgets and recurrence counters span replacements:
        they are never reset.

        Policy rejections raise :class:`ReplacementPolicyError` (the
        orchestrator parks the package on those reason codes); claim-shape
        violations raise :class:`LaunchConfigError` /
        :class:`InvalidLaunchModelError`.
        """
        claim = args.config
        assert claim is not None and claim.predecessor_agent_id is not None
        self._claim_shape_guards(args, claim)
        # The claimed model is the canonical top-level model field, or the
        # legacy nested config.model the schema promotes — the same selection
        # the fresh launch resolves, so the policy validates what the new
        # agent will actually run.
        claim_model = args.model if args.model is not None else claim.model
        model_field = "model" if args.model is not None else "config.model"
        assert claim_model is not None  # the claim-shape guard ran first
        async with self._view._registry_lock:
            record, envelope = self._predecessor_envelope_locked(
                parent, claim.predecessor_agent_id
            )
            self._check_lineage_and_barrier_locked(record, claim, envelope)
            decision = self._validate_claim_policy_locked(
                parent,
                record,
                claim,
                envelope,
                claim_model=claim_model,
                model_field=model_field,
            )
            # Review R6: exclusive package ownership — a predecessor
            # already superseded (an earlier claim or busy replacement
            # admitted) can never ground a second claim: repeated claims
            # must not create multiple replacements. The check runs AFTER
            # the per-package budget so an exhausted package still reports
            # its exhaustion reason.
            if record.recovery_superseded or record.reserved_for_replacement:
                raise ReplacementPolicyError(
                    ["predecessor_already_superseded"],
                    f"Replacement claim rejected: predecessor agent "
                    f"{record.agent_id} was already superseded by an "
                    "admitted replacement; the package already has its "
                    "replacement",
                )
            # Review finding 2 (A2): canonical identity resolution and
            # admission COMPLETE before any consequential replacement-state
            # change. The invalidation and budget consumption below are the
            # consequential mutation; an invalid or denied launch identity
            # must fail BEFORE them — a rejected launch leaves zero side
            # effects (recovery intact, budget unconsumed, no supersession).
            # Resolution and admission are pure in the parent runtime and
            # the launch arguments (no registry state, no locks), so they
            # run safely under the registry lock, and they mirror the
            # fresh-launch admission in create_registered exactly.
            requested_identity = requested_identity_reference(args)
            if requested_identity is not None:
                self.require_identity_allowed(parent, requested_identity)
            candidate = self.resolve_candidate(parent, args)
            self.require_identity_allowed(
                parent, candidate.orchestrator.config.agent.identity
            )
            # Handoff ordering: the old recovery is disabled and the package
            # counter incremented BEFORE the new run's write token is
            # registered at accept_background — the write-ownership transfer
            # happens only after reconciliation state is settled and the
            # explicit claim is admitted. The predecessor is never
            # auto-restarted; its lineage, counters, and artifacts survive.
            package_key = claim.work_package_id
            assert package_key is not None  # the claim-shape guard ran first
            self._view._invalidate_recovery_locked(record)
            self._view._package_replacements[package_key] = (
                self._view._package_replacements.get(package_key, 0) + 1
            )
            logger.info(decision.notice)
            return decision

    @contextmanager
    def creation_fence(self) -> Iterator[None]:
        """Hold the child-creation fence (policy reservation + counter)."""
        self._view._require_policy_unreserved()
        self._view._creating_children += 1
        try:
            yield
        finally:
            self._view._creating_children -= 1

    def restore_sequence(self, root: SessionRuntime) -> None:
        """Restore the agent-id high-water mark from persisted metadata."""
        metadata = root.agent_loop.session_logger.session_metadata
        config = metadata.config if metadata is not None else None
        high_water_mark = (
            config.get(_AGENT_ID_HIGH_WATER_MARK_KEY) if config is not None else None
        )
        if isinstance(high_water_mark, int) and not isinstance(high_water_mark, bool):
            self._view._next_agent_seq = max(
                self._view._next_agent_seq, high_water_mark
            )

    async def issue_agent_id(self, parent: SessionRuntime) -> str:
        """Allocate the next agent id, persisting the high-water mark."""
        self._view._next_agent_seq += 1
        reserved_seq = self._view._next_agent_seq
        metadata = parent.agent_loop.session_logger.session_metadata
        if metadata is not None:
            config = dict(metadata.config or {})
            config[_AGENT_ID_HIGH_WATER_MARK_KEY] = reserved_seq
            metadata.config = config
            await parent.agent_loop.session_logger._persist_metadata_field(
                "config", config
            )
        return f"agent-{reserved_seq}"

    async def create_registered(
        self, parent: SessionRuntime, args: TaskArgs, ctx: InvokeContext
    ) -> tuple[SessionRuntime, str]:
        """Construct, register, and project one child, rolling back on failure.

        The second element is the child's committed launch identity reference.
        """
        candidate = self._view._resolve_launch_candidate(parent, args)
        # The identity denylist applies AFTER final resolution (11c): a
        # route/shorthand/structured identity launch is checked against its
        # resolved identity — the same surface the task tool enforces — before
        # any child resource is constructed.
        self.require_identity_allowed(
            parent, candidate.orchestrator.config.agent.identity
        )
        attempt = ChildPublicationAttempt(
            parent=parent,
            identity=(
                parent.agent_loop.session_id,
                parent.agent_loop._session_generation,
            ),
            candidate=candidate,
            tool_call_id=ctx.tool_call_id,
        )
        self.validate_admission(parent, attempt.identity)
        with self.creation_fence():
            child = await self._view._runtime_factory.create_child(
                parent.agent_loop, candidate
            )
            # Spawn handoff, inside the creation fence: the child's first
            # message lists the working-set files the dispatcher hands it
            # (paths + last-seen revisions + current outlines), ahead of the
            # task brief. None when the parent has no working set.
            handoff = child.file_state_handoff_message()
            if handoff is not None:
                child.messages.append(handoff)
            attempt.child = child
        try:
            self.validate_admission(parent, attempt.identity)
        except BaseException:
            await self._view._discard_child(child)
            raise
        runtime = self._view._build_child_runtime(child)
        attempt.runtime = runtime
        async with self._view._child_publication(runtime):
            try:
                self.validate_admission(parent, attempt.identity)
            except BaseException:
                await self._view._discard_child(child)
                raise
            self._view._children[child.session_id] = runtime
            self._view._child_links[child.session_id] = (parent, ctx.tool_call_id)
            try:
                await child.persist_empty_session()
                self.validate_admission(parent, attempt.identity)
                await parent.agent_loop.record_child_session(
                    child, ctx.tool_call_id, candidate.identity
                )
                attempt.link_recorded = True
                self.validate_admission(parent, attempt.identity)
                attempt.projection_started = True
                if parent.turns._projector is not None:
                    await parent.turns.link_subagent(ctx.tool_call_id, child.session_id)
                self.validate_admission(parent, attempt.identity)
            except BaseException:
                await attempt.rollback(self._view)
                raise
            return runtime, candidate.identity

    async def rollback_created(
        self, parent: SessionRuntime, runtime: SessionRuntime, tool_call_id: str
    ) -> None:
        """Remove every durable/runtime trace of a child rejected before acceptance."""
        session_id = runtime.agent_loop.session_id
        self._view._children.pop(session_id, None)
        self._view._child_links.pop(session_id, None)
        with suppress(Exception):
            await parent.turns.unlink_subagent(tool_call_id, session_id)
        with suppress(Exception):
            await parent.agent_loop.forget_child_session(session_id, tool_call_id)
        with suppress(Exception):
            await runtime.close()
        with suppress(Exception):
            await delete_saved_session(
                session_id, runtime.agent_loop.config.session_logging
            )
