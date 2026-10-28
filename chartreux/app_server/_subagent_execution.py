from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import ValidationError

from chartreux.app_server._subagent_finalization import (
    RunFinalization,
    _check_runner,
    finalize_background_run,
)
from chartreux.app_server._subagent_records import (
    AgentRecord,
    CompletionCapability,
    FrozenCompletionContract,
    RunRecord,
    StoredRunResult,
    _ActiveWorkSlot,
    _AgentState,
    _AgentTombstone,
    _ReplacementClaim,
    _ResumeAdmission,
)
from chartreux.app_server.models import PublicTurn, PublicTurnStatus, TextContentBlock
from chartreux.app_server.protocol import TurnInterruptParams, TurnStartParams
from chartreux.core import completion_checks
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop.resume_progress import (
    arm_resume_progress_deadline,
    clear_resume_progress_deadline,
)
from chartreux.core.agents.identities import AgentIdentity, get_builtin_identity
from chartreux.core.agents.manager import requested_identity_reference
from chartreux.core.completion_contract import (
    COMPLETION_ENVELOPE_MARKER,
    CompletionReceipt,
    CompletionRejectionReason,
)
from chartreux.core.events import BaseEvent, ToolStreamEvent
from chartreux.core.git.errors import GitError
from chartreux.core.git.repo import GitRepo
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.recovery_models import (
    FailureEnvelope,
    PackageTier,
    RecoveryReference,
    build_completion_rejection_envelope,
    build_failure_envelope,
    canonical_endpoint_identity,
)
from chartreux.core.recovery_observation import (
    FailureObservation,
    observe_exception,
    preserve_run_observation,
    record_observation,
    turn_observation,
)
from chartreux.core.recovery_policy import ReplacementDecision
from chartreux.core.subagents import (
    AgentBusyError,
    AgentClosedError,
    AgentEviction,
    LaunchConfig,
    LaunchOutcome,
    RetentionPolicy,
    RunStatus,
    RunStopReason,
    SubagentRunAccumulator,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
    normalize_task_summary,
)
from chartreux.core.tools.models import ToolPermission
from chartreux.core.utils import kill_async_subprocess
from chartreux.model_display import format_model_display_name
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime
    from chartreux.app_server._streaming import BoundedEventQueue
    from chartreux.app_server._subagent_launch import ChildLaunchAdmission, OverrideMint
    from chartreux.app_server._subagent_results import PublicationDecision
    from chartreux.app_server._turns import TurnStartAction
    from chartreux.core.agent_loop._config_transactions import (
        _PreparedLaunchReconfiguration,
    )
    from chartreux.core.agent_loop.backend_lifetime import BackendPublish
    from chartreux.core.agent_loop_hooks import CompletionEvidenceReceipt
    from chartreux.core.agents.launch import LaunchCandidate
    from chartreux.core.session_types import AgentStats
    from chartreux.core.tools.base import InvokeContext

_BACKGROUND_PROGRESS_LIMIT = 32
_MAX_RUN_HISTORY = 32


def cancellation_finalization(
    *,
    run_id: str,
    contract: FrozenCompletionContract | None,
    admission_head: str | None,
) -> RunFinalization:
    """The cancelled-run finalization stub (P6b).

    A run cancelled before its audit settled never credits the worker with
    verifier evidence: the stub is cached so no later finalization re-runs
    the bound checks, the admission baseline is preserved unchanged, and a
    contract that binds check templates stamps the harness-side
    ``unavailable`` word — the verifier evidence never became available.
    """
    binds_checks = (
        contract is not None
        and contract.envelope is not None
        and any(
            requirement.template is not None
            for requirement in contract.envelope.requirements
        )
    )
    return RunFinalization(
        run_id=run_id,
        admission_head=admission_head,
        runtime_failure="unavailable" if binds_checks else None,
    )


def apply_cancellation_disposition(
    accumulator: SubagentRunAccumulator,
    run_id: str,
    contract: FrozenCompletionContract | None,
    admission_head: str | None,
    stop_reason: RunStopReason | None,
) -> None:
    """The cancelled-run disposition (P6b): the reason recorded, stub cached.

    The recorded error forces every later ``build_result`` completed=False
    (no worker credit), and the stub finalization cached on the run-keyed
    accumulator cache makes the awaited finalization after this disposition
    a cache hit that runs no checks.
    """
    reason = stop_reason if stop_reason is not None else RunStopReason.USER_CANCELLED
    accumulator.record_error(f"run cancelled ({reason.value})")
    accumulator._finalization_cache[run_id] = cancellation_finalization(
        run_id=run_id, contract=contract, admission_head=admission_head
    )


#: The bound on the claim scan's JSON-object candidates (P12): the marker
#: pre-filter keeps a no-claim report off the scan entirely.
_CLAIM_SCAN_MAX_OBJECTS = 64


def extract_completion_claim(
    response: str,
) -> CompletionReceipt | Mapping[str, Any] | None:
    """The worker's completion-receipt claim parsed from its final report.

    The P13 worker contract emits a strict receipt — a JSON object carrying
    the ``completion_envelope_v1`` marker and the ``satisfied_requirement_ids``
    key — in the final report. The LAST such object is the operative claim:
    the final report's receipt wins over any earlier prose, and the nested
    envelope (which carries the same marker but no ``satisfied_requirement_ids``
    key) is not the claim. A well-formed payload validates into the P1
    receipt model; a grammar-violating payload rides RAW so the truth table
    classifies the violation as a disclosure fact instead of escaping as an
    exception. A report with no receipt-shaped object carries no claim
    (``None``) — an availability fact, never a fabricated one.
    """
    if COMPLETION_ENVELOPE_MARKER not in response:
        return None
    decoder = json.JSONDecoder()
    claim: Mapping[str, Any] | None = None
    candidates = 0
    for index, char in enumerate(response):
        if char != "{":
            continue
        candidates += 1
        if candidates > _CLAIM_SCAN_MAX_OBJECTS:
            break
        try:
            payload, _end = decoder.raw_decode(response, index)
        except ValueError:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("marker") == COMPLETION_ENVELOPE_MARKER
            and "satisfied_requirement_ids" in payload
        ):
            claim = payload
    if claim is None:
        return None
    try:
        return CompletionReceipt.model_validate(claim)
    except ValidationError:
        # A grammar violation is a grammar fact, never a worker fault: the
        # raw payload rides so the truth table classifies it.
        return claim


def operative_completion_claim(
    accumulator: SubagentRunAccumulator, *, turn_completed: bool
) -> CompletionReceipt | Mapping[str, Any] | None:
    """The worker's receipt claim, gated on a healthy completed turn (P12).

    The claim is operative only on a healthy completed turn's publication:
    a cancelled or failed run's primary disposition stays authoritative
    (the cancellation reason or the runtime failure), so no claim rides —
    the composition then classifies the run as an unaddressed mandate (a
    disclosure fact), never a worker-side attribution.
    """
    if not turn_completed:
        return None
    return extract_completion_claim("".join(accumulator._response))


def rejection_envelope(
    context: BackgroundExecutionContext, result: TaskResult
) -> FailureEnvelope | None:
    """The measured rejection envelope for one audit-rejected run (P12).

    When the composed audit rejects the run's claimed completion — the
    enforce-mode worker-side disposition (``claim_contradicted``) — the run
    finalizes as FAILED with the P11 rejection envelope, and the result's
    audit stamp, the run status, and the envelope agree. The decision reads
    ONLY the built result's typed stamp and the run's terminal state: the
    truth table attributes a rejection solely while the verifier stayed
    healthy, and only a healthy completed turn's publication can be
    audit-rejected, so primary runtime failures, cancellation reasons, and
    honest abort/blocker outcomes never carry the rejection envelope. The
    envelope is PROHIBITED for recovery — a typed audit rejection
    authorizes a cross-agent replacement, never a warm same-agent
    auto-resume (the suspension gates stay shut).

    Persisted-result compatibility: ``TaskResult.completion_audit`` is
    omitted from dumps when ``None`` (the pre-audit dump shape is
    unchanged); mixed-version consumption — an old reader consuming a new
    persisted result with a populated audit — is unsupported.
    """
    audit = result.completion_audit
    if (
        result.completed
        or audit is None
        or audit.rejection is not CompletionRejectionReason.CLAIM_CONTRADICTED
    ):
        return None
    rejection = audit.rejection
    if context.run_record.status is not RunStatus.COMPLETED:
        # A cancelled or failed turn keeps its primary disposition: the
        # rejection envelope never replaces a runtime failure or a
        # cancellation reason.
        return None
    contract = context.run_record.completion_contract
    if contract is None or contract.audit_mode != "enforce":
        # Escalation requires an enforce-mode contradiction disposition;
        # stamp mode never escalates.
        return None
    try:
        return build_completion_rejection_envelope(
            agent_id=context.agent_id,
            run_id=context.run_id,
            rejection=rejection,
            turn_id=context.turn_id,
            work_package_id=context.record.work_package_id,
            predecessor_agent_id=context.record.predecessor_agent_id,
            recovery_chain_id=context.record.recovery_chain_id,
            message=result.response,
        )
    except Exception as exc:
        # Total by contract: an envelope-construction failure is logged and
        # reported as None — it never masks the run's audit rejection.
        logger.warning(
            "Failed to build the completion rejection envelope", exc_info=exc
        )
        return None


async def finalize_run_with_drain(
    *,
    run_id: str,
    accumulator: SubagentRunAccumulator,
    contract: FrozenCompletionContract | None,
    admission_head: str | None,
    runtime: SessionRuntime,
) -> RunFinalization:
    """The drain-wrapped once-only finalization (P6b).

    Wraps ``finalize_background_run`` with check-spawn tracking scoped to
    THIS task: spawns recorded while this wrapper's window is open carry
    their owning task, so a sibling run's checks — and the worker's own
    managed shell jobs — are never collateral. If cancellation lands
    mid-audit (case b), the recorded check processes are force-terminated
    and reaped before the CancelledError propagates — a completed check
    reaps safely (its group is left intact), an in-flight check is killed
    with its group — and the partial findings are discarded: the cache
    write happens only once the finalization settles (P6a), so no partial
    evidence is ever published.

    The seam is the module-level ``spawn_shell_command`` the check runner
    resolves at call time (the same seam the check tests patch). The
    restore only fires when this wrapper is still the installed seam; a
    leftover live sibling wrapper delegates transparently.
    """
    original_spawn = completion_checks.spawn_shell_command
    recorded: list[tuple[asyncio.Task[Any] | None, asyncio.subprocess.Process]] = []

    async def tracking_spawn(command: str, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await original_spawn(command, **kwargs)
        recorded.append((asyncio.current_task(), process))
        return process

    completion_checks.spawn_shell_command = tracking_spawn
    try:
        return await finalize_background_run(
            run_id=run_id,
            accumulator=accumulator,
            contract=contract,
            admission_head=admission_head,
            runtime=runtime,
        )
    except asyncio.CancelledError:
        current = asyncio.current_task()
        for owner, process in recorded:
            if owner is not current:
                continue
            with suppress(Exception):
                await kill_async_subprocess(process, kill_exited_process_group=False)
        raise
    finally:
        if completion_checks.spawn_shell_command is tracking_spawn:
            completion_checks.spawn_shell_command = original_spawn


def _tool_call_effect_sets(messages: Sequence[LLMMessage]) -> tuple[set[str], set[str]]:
    """Transcript-visible tool-call id sets: (requested, settled).

    Assistant ``tool_calls`` request effects; ``Role.tool`` messages settle
    them by ``tool_call_id``.
    """
    requested: set[str] = set()
    settled: set[str] = set()
    for message in messages:
        if message.role is Role.assistant:
            for call in message.tool_calls or ():
                if call.id:
                    requested.add(call.id)
        elif message.role is Role.tool and message.tool_call_id:
            settled.add(message.tool_call_id)
    return requested, settled


def _tool_effect_inputs(messages: Sequence[LLMMessage]) -> tuple[bool, bool]:
    """Transcript-visible tool-effect evidence: (requested, pending).

    Assistant ``tool_calls`` request effects; ``Role.tool`` messages settle
    them by ``tool_call_id``. This is transcript evidence only — hooks,
    jobs, and child launches are not transcript-visible, so the envelope's
    tool_effects derivation treats absence of tool calls as unknown, never
    as settled (F4).
    """
    requested, settled = _tool_call_effect_sets(messages)
    return bool(requested), bool(requested - settled)


def _outstanding_tool_call_ids(messages: Sequence[LLMMessage]) -> set[str]:
    """The tool calls the transcript shows requested but never settled.

    Review R5: a reconcile submission must account for exactly these ids —
    an unrelated id, an incomplete subset, a repeated id, or an empty
    submission verifies nothing about the unknown effects and never clears
    the barrier.
    """
    requested, settled = _tool_call_effect_sets(messages)
    return requested - settled


#: The cumulative AgentStats fields a replacement inherits from its
#: predecessor (review R7): the middleware limits evaluate these against
#: the loop-local stats, so a fresh replacement loop must not start them
#: from zero. Per-turn measurements and pricing stay loop-local.
_CUMULATIVE_STAT_FIELDS = (
    "steps",
    "session_prompt_tokens",
    "session_completion_tokens",
    "session_cached_tokens",
    "tool_calls_agreed",
    "tool_calls_rejected",
    "tool_calls_hook_denied",
    "tool_calls_failed",
    "tool_calls_succeeded",
    "known_cost_total",
)


def _transfer_cumulative_stats(source: AgentStats, target: AgentStats) -> None:
    """Seed a replacement's fresh stats with the predecessor's consumption.

    Review R7: a replacement creates a fresh loop whose AgentStats start at
    zero while turn/price/token middleware limits are evaluated against
    those LOCAL stats — without the transfer, replacement renewed every
    budget the predecessor had already consumed. The cumulative counters
    add up; per-turn measurements and pricing stay local to the new loop.
    """
    for field_name in _CUMULATIVE_STAT_FIELDS:
        setattr(
            target,
            field_name,
            getattr(target, field_name) + getattr(source, field_name),
        )
    target.has_unknown_cost = target.has_unknown_cost or source.has_unknown_cost


def _verification_capability(parent: SessionRuntime) -> CompletionCapability:
    """The run-scoped harness verification capability (P4).

    Derived from the parent delegation authority: the harness
    acceptance-check runner may act only under authority the parent could
    delegate, so the capability is available exactly when the parent's
    delegation ceiling carries the shell tool. A parent whose delegation
    ceiling excludes the shell leaves the capability denied, and a parent
    that cannot report its authority leaves it unavailable; both stamp the
    check unavailable at finalization — never a worker fault.
    """
    try:
        delegable = set(parent.agent_loop.tool_manager.delegable_tools)
    except Exception:
        return CompletionCapability.UNAVAILABLE
    if "bash" in delegable:
        return CompletionCapability.AVAILABLE
    return CompletionCapability.DENIED


def _shell_less_launch(args: TaskArgs, identity_facts: AgentIdentity | None) -> bool:
    """Whether this launch is shell-less (round-2 amendment 10).

    A shell-less launch is a custom identity (no harness-known posture) or a
    launch explicitly disabling the shell tool per-launch; the shipped
    researcher/advisor/reviewer seats are NOT shell-less by identity.
    """
    if identity_facts is None:
        return True
    config = args.config
    if config is None:
        return False
    if config.disabled_tools is not None and "bash" in config.disabled_tools:
        return True
    override = (config.tools or {}).get("bash")
    return override is not None and override.permission is ToolPermission.NEVER


def _capture_admission_head(cwd: Path) -> str | None:
    """Capture the admission HEAD baseline for one run (P4, round-7).

    PRE-EXECUTION: called at admission, never at finalization, so a worker
    commit moves the publication HEAD away from the captured baseline. A
    probe failure (not a repository, unborn HEAD, git unavailable) is
    informational: the baseline is None, never an admission failure.
    """
    try:
        return GitRepo.head_commit_at(cwd)
    except GitError:
        return None


def _budget_fit_description(
    run_max_price: float | None, run_max_session_tokens: int | None
) -> str:
    """The advisory budget-fit description for one admission (P14)."""
    limits = []
    if run_max_price is not None:
        limits.append(f"price<={run_max_price:.2f}")
    if run_max_session_tokens is not None:
        limits.append(f"tokens<={run_max_session_tokens}")
    return ", ".join(limits) if limits else "unconfigured (no run-budget middleware)"


def freeze_run_admission(
    *,
    args: TaskArgs,
    identity: str,
    identity_revision: str | None,
    tier: PackageTier | None,
    runtime: SessionRuntime,
    parent: SessionRuntime,
) -> tuple[FrozenCompletionContract, str | None]:
    """Freeze the completion contract at admission, capture the HEAD baseline.

    The contract freezes identity/seat/tier, cwd/roots, the mandates, the
    envelope requirement/exemption, the audit mode, and the capability
    inputs. The admission HEAD baseline is captured alongside the frozen
    contract — only when the contract mandates a commit artifact; runs
    without a commit mandate perform no HEAD probes.
    """
    loop = runtime.agent_loop
    try:
        roots = tuple(
            str(root) for root in loop.tool_manager.workspace.authorized_roots
        )
    except Exception:
        # The workspace is the canonical authority; a loop that cannot
        # report it still admits — the contract falls back to the cwd.
        roots = (str(loop.cwd),)
    identity_facts = get_builtin_identity(identity.removeprefix("builtin:"))
    contract = FrozenCompletionContract.freeze(
        identity=identity,
        identity_revision=identity_revision,
        seat=args.seat,
        tier=tier,
        cwd=str(loop.cwd),
        roots=roots,
        completion_check=args.completion_check,
        required_artifacts=args.required_artifacts,
        shell_less=_shell_less_launch(args, identity_facts),
        audit_mode=loop.config.completion.mode,
        capability=_verification_capability(parent),
    )
    admission_head = (
        _capture_admission_head(loop.cwd) if contract.requires_commit_artifact else None
    )
    return contract, admission_head


class ExecutionStateView(Protocol):
    """Narrow typed view over the registry-owned execution state.

    Dynamic access: every member is read through the view at call time, so
    the active-work slot set, the monitor-task set, the record/child maps,
    the latest-run index, the shared registry lock, the result store, the
    pending notifications, the drain flag, the notification suppression,
    the retention policy, and the clock are never snapshotted at
    construction — registry state mutated after the collaborator is built
    stays visible,
    and instance patches of the registry delegates (including the background
    admission validation, the launch-candidate resolution, the identity
    admission, the child creation + rollback pair, the run-id allocation,
    the clock, the retention policy, the run-outcome resolution, the
    generation check, the result expiry, the write-token registration, the
    publication decision, the agents-update emission, the record naming,
    and the close-on-completion transition) still shadow.
    """

    _active_work_slots: set[_ActiveWorkSlot]
    _agent_records: dict[str, AgentRecord]
    _children: dict[str, SessionRuntime]
    _latest_run_ids: dict[str, str]
    _registry_lock: asyncio.Lock
    _result_store: dict[tuple[str, str], StoredRunResult]
    _pending_notifications: set[tuple[str, str]]
    _monitor_tasks: set[asyncio.Task[None]]
    _draining_children: bool
    _suppressed_notifications: set[str]
    _retention_policy: RetentionPolicy
    _clock: Callable[[], float]

    # C4: the launch-admission collaborator owns the override mint.
    _launch_admission: ChildLaunchAdmission

    def _validate_background_admission(
        self, parent: SessionRuntime, identity: tuple[str, int]
    ) -> None: ...

    def _live_tombstone_locked(
        self, agent_id: str, now: float
    ) -> _AgentTombstone | None: ...

    @staticmethod
    def _tombstone_reuse_error(
        agent_id: str, tombstone: _AgentTombstone
    ) -> AgentClosedError: ...

    def _validate_replacement_locked(self, claim: _ReplacementClaim) -> None: ...

    def _require_task_identity_allowed(
        self, parent: SessionRuntime, identity: str
    ) -> None: ...

    def _rearm_reaper_locked(self, exclude: frozenset[str] = frozenset()) -> None: ...

    def _resolve_launch_candidate(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        *,
        runtime: SessionRuntime | None = None,
        identity: str | None = None,
    ) -> LaunchCandidate: ...

    def _issue_agent_id(self, parent: SessionRuntime) -> Coroutine[Any, Any, str]: ...

    def _create_registered_child(
        self, parent: SessionRuntime, args: TaskArgs, ctx: InvokeContext
    ) -> Coroutine[Any, Any, tuple[SessionRuntime, str]]: ...

    def _rollback_created_child(
        self, parent: SessionRuntime, runtime: SessionRuntime, tool_call_id: str
    ) -> Coroutine[Any, Any, None]: ...

    def _rollback_reused_claim(
        self,
        record: AgentRecord,
        previous_idle_since: float | None,
        *,
        apply_policy: bool = False,
    ) -> Coroutine[Any, Any, None]: ...

    @staticmethod
    def _new_run_id(agent_id: str) -> str: ...

    def _resolve_run_outcome(
        self, run: RunRecord, turn: PublicTurn | None = None
    ) -> tuple[RunStatus, RunStopReason | None]: ...

    def _generation_is_current(self, generation: int) -> bool: ...

    def _expire_results_locked(self) -> None: ...

    def _register_writer_locked(
        self, agent_id: str, run_id: str, token: object
    ) -> None: ...

    def _publish_locked(
        self, context: BackgroundExecutionContext
    ) -> PublicationDecision: ...

    def _emit_agents_update(
        self, evictions: list[AgentEviction] | None = None
    ) -> Coroutine[Any, Any, None]: ...

    @staticmethod
    def _record_name(record: AgentRecord) -> str: ...

    def _close_unretained_locked(
        self, record: AgentRecord, run: RunRecord | None, *, name: str
    ) -> bool: ...

    def _suspend_after_failure_locked(
        self, record: AgentRecord, run: RunRecord
    ) -> bool: ...

    def _observe_run_completion_locked(
        self, record: AgentRecord, run: RunRecord
    ) -> None: ...


@dataclass
class ForegroundExecutionContext:
    """Per-invocation state for one foreground execution (ASC area 3).

    Carries the child runtime and loop, the bounded progress queue, the run
    accumulator, and the completion metadata mark captured at admission; the
    previous event sink, the turn id, the completion task, and the work slot;
    and the terminal result once the stream settles. The registry-owned slot
    set stays canonical; the context holds only the foreground run's
    per-invocation state. Contexts hold REFERENCES, not copies.
    """

    runtime: SessionRuntime
    child: AgentLoop
    progress: BoundedEventQueue[ToolStreamEvent]
    accumulator: SubagentRunAccumulator
    completion_metadata_mark: tuple[int, int]
    tool_call_id: str
    previous_event_sink: Callable[[BaseEvent], Awaitable[None]] | None
    slot: _ActiveWorkSlot
    turn_id: str | None = None
    completion: asyncio.Task[PublicTurn] | None = None
    result: TaskResult | None = None
    # P4: the completion contract frozen at foreground admission, and the
    # admission HEAD baseline captured alongside it (only when the contract
    # mandates a commit artifact).
    completion_contract: FrozenCompletionContract | None = None
    admission_head: str | None = None
    # P2/P3: the receipt sink installed on the child loop at admission (the
    # run accumulator's side-channel callback); cleared at run teardown.
    completion_evidence_sink: Callable[[CompletionEvidenceReceipt], None] | None = None

    async def consume_event(self, event: BaseEvent) -> None:
        """Feed one child event into the bounded progress queue.

        The former foreground consume_event closure of the registry's
        admitted run (the 17.6a de-captured consumer), converted to a
        context method: the accumulator, the progress queue, and the
        tool-call id are context fields. The queue-feeding body is
        unchanged.
        """
        update = self.accumulator.observe(event, tool_call_id=self.tool_call_id)
        if update is None:
            return
        try:
            self.progress.put_nowait(update)
        except asyncio.QueueFull:
            with suppress(asyncio.QueueEmpty):
                self.progress.get_nowait()
            self.progress.put_nowait(update)

    def discard_work_slot(
        self, view: ExecutionStateView, _task: asyncio.Task[PublicTurn]
    ) -> None:
        """Release the capacity slot when the completion task settles.

        The former slot discard callback of the registry's admitted run,
        converted to a named context method (the context-method-takes-the
        view convention): the registry-owned active-work slot set stays
        canonical and is read through the view at call time.
        """
        view._active_work_slots.discard(self.slot)


@dataclass
class BackgroundExecutionContext:
    """Per-invocation state for one background execution (ASC area 3).

    Carries the exact child runtime, record, and run record with their IDs;
    the created/reused status and launch identity; the previous idle
    timestamp and the generation/authority reservations; the retained
    candidate and prepared reconfiguration; the prior record/retention/
    latest-run/index snapshot the rollback restores; the previous event
    sink; the accumulator and completion metadata mark; the prepared
    completion future and the installed monitor; the publication token,
    turn id, start action, and backend publication; and the mutable
    committed flag. Contexts hold REFERENCES, not copies: the monitor reads
    the same committed flag the acceptance writes, and the registry-owned
    state stays canonical.
    """

    runtime: SessionRuntime
    record: AgentRecord
    run_record: RunRecord
    agent_id: str
    run_id: str
    tool_call_id: str
    created_record: bool
    launch_identity: str | None
    previous_idle_since: float | None
    reserved_child_generation: int | None
    reserved_parent_authority_revision: int | None
    retained_candidate: LaunchCandidate | None
    prepared_reconfiguration: _PreparedLaunchReconfiguration | None
    prior_current_run: RunRecord | None
    prior_retained: bool
    prior_latest_run_id: str | None
    prior_registry_latest_present: bool
    prior_registry_latest: str | None
    prior_idle_since: float | None
    previous_event_sink: Callable[[BaseEvent], Awaitable[None]] | None
    accumulator: SubagentRunAccumulator
    completion_metadata_mark: tuple[int, int]
    completion_placeholder: asyncio.Future[None]
    publication_token: object
    # P2/P3: the receipt sink installed on the child loop at admission (the
    # run accumulator's side-channel callback); cleared at run teardown so a
    # retained loop never hands a later run's receipts to a spent accumulator.
    completion_evidence_sink: Callable[[CompletionEvidenceReceipt], None] | None = None
    monitor: asyncio.Task[None] | None = None
    turn_id: str | None = None
    action: TurnStartAction | None = None
    backend_publication: BackendPublish | None = None
    committed: bool = False
    # WP5: the validated failure-time replacement decision for a claimed
    # replacement launch (carries the milestone notice with its honest cost
    # implication); None for unclaimed launches.
    replacement_decision: ReplacementDecision | None = None
    # C4: the bounded un-stick diagnostics snapshot for an explicit
    # override_cooldown launch, taken pre-start (the child's first model
    # call consumes the coordinator binding); None for ordinary launches.
    override_diagnostics: dict[str, object] | None = None

    async def consume_event(self, event: BaseEvent) -> None:
        """Feed one child event into the run record's progress summaries.

        The former background consume_event closure of the registry's
        admitted run (the 17.6a de-captured consumer), converted to a
        context method: the accumulator, the run record, and the tool-call
        id are context fields. The observation body is unchanged.
        """
        update = self.accumulator.observe(event, tool_call_id=self.tool_call_id)
        if update is None:
            return
        if len(self.run_record.progress_summaries) < _BACKGROUND_PROGRESS_LIMIT:
            self.run_record.progress_summaries.append(update.message)
        else:
            self.run_record.progress_overflow = True

    def settle_prepared_waiters(self, _task: asyncio.Task[None]) -> None:
        """Settle the admission-time completion future exactly once.

        The former settle_prepared_waiters done-callback of the registry's
        admitted run (the 17.6a de-captured callback), converted to a
        context method: the prepared completion future is a context field.
        A waiter can snapshot the prepared run before its monitor is
        installed; keep that admission-time future linked to the same
        completion, including a prepared-start stop.
        """
        if not self.completion_placeholder.done():
            self.completion_placeholder.set_result(None)


class SubagentExecution:
    """Invocation orchestration for subagent execution (ASC area 3).

    Narrow collaborator of the session runtime registry: the execution
    per-invocation contexts and their converted closures — the event
    consumers and the prepared-waiter callback as context methods, the
    foreground teardown as the executor's teardown_foreground, the
    background preparation + rollback as the executor's prepare_background
    + rollback_background, the background monitor + finalization as the
    executor's monitor, and the background acceptance + acknowledgment as
    the executor's accept_background + acknowledge_background — live here,
    while the run envelope and the foreground streaming stay on the
    registry and operate on the contexts.
    Canonical state stays registry-owned (§6.7); no private collaborator
    locks; no new tasks/queues/relay layers. The registry wires the
    collaborator in its constructor; teardown_foreground reads no registry
    state, so it stays a @staticmethod and the admitted-run path calls it
    through the class, keeping the registry's patch seams. The background
    path is not driven through spec'd mock registries, so its moved regions
    read registry state through the view — instance patches and the manual
    clock still shadow.
    """

    def __init__(self, view: ExecutionStateView) -> None:
        self._view = view

    @staticmethod
    async def teardown_foreground(context: ForegroundExecutionContext) -> None:
        """Tear down one foreground execution after its stream settles.

        Replaces the former teardown_foreground closure of the registry's
        admitted run: the completion task and the runtime are context
        fields (narrowed with an assert — the ReleaseContext.cleanup alias
        convention). The cleanup region and the exception suppression are
        unchanged.
        """
        errors: list[BaseException] = []
        completion = context.completion
        runtime = context.runtime
        assert completion is not None
        try:
            if not completion.done():
                active_turn = runtime.turns.active_turn
                if active_turn is not None:
                    runtime.turns.interrupt(
                        TurnInterruptParams(
                            session_id=active_turn.session_id,
                            expected_turn_id=active_turn.id,
                        )
                    )
                else:
                    await runtime.turns.close()
            await asyncio.gather(completion, return_exceptions=True)
        except BaseException as exc:
            errors.append(exc)
        try:
            await runtime.close()
        except BaseException as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Failed to tear down foreground agent", errors)

    async def prepare_background(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        parent: SessionRuntime,
        parent_identity: tuple[str, int],
        args: TaskArgs,
        ctx: InvokeContext,
        slot: _ActiveWorkSlot,
        *,
        replacement: _ReplacementClaim | None,
        summary_task: str | None,
        resume: _ResumeAdmission | None = None,
        replacement_decision: ReplacementDecision | None = None,
    ) -> BackgroundExecutionContext:
        """Prepare one background execution and return its context.

        The former background preparation region of the registry's admitted
        run (ASC area 3, step 3): the reuse reservation, the candidate
        resolution and reconfiguration preparation, the created-child path,
        and the run-record setup, moved intact. Registry state is read
        through the view at call time; the returned context carries the
        preparation + acceptance-tail state as references.

        WP2: ``resume`` carries a checked resume admission — a RESTORING
        record is admitted (an ordinary reuse still requires IDLE), the
        child session keeps its original transcript link (no re-link to a
        synthesized tool call), and the run id is the successor id
        published at admission.

        WP5: ``replacement_decision`` is the validated failure-time
        replacement decision for a claimed replacement launch (already
        admitted by the launch admission); the fresh record carries the
        claim's predecessor/package lineage from ``args.config``.
        """
        self._view._validate_background_admission(parent, parent_identity)
        retained_candidate: LaunchCandidate | None = None
        previous_idle_since: float | None = None
        reserved_child_generation: int | None = None
        reserved_parent_authority_revision: int | None = None
        if args.agent_id is not None:
            async with self._view._registry_lock:
                self._view._validate_background_admission(parent, parent_identity)
                record = self._view._agent_records.get(args.agent_id)
                if record is None:
                    tombstone = self._view._live_tombstone_locked(
                        args.agent_id, self._view._clock()
                    )
                    if tombstone is not None:
                        raise self._view._tombstone_reuse_error(
                            args.agent_id, tombstone
                        )
                    raise UnknownAgentError(
                        f"Unknown agent_id instance handle: {args.agent_id}. Check the agent_id with check_agents"
                    )
                if replacement is not None:
                    self._view._validate_replacement_locked(replacement)
                    if record is not replacement.record:
                        raise RuntimeError("Agent replacement residency changed")
                elif record.reserved_for_replacement or (
                    record.state is not _AgentState.IDLE
                    and not (
                        resume is not None and record.state is _AgentState.RESTORING
                    )
                ):
                    raise AgentBusyError("Agent is already running a task")
                if (
                    record.runtime._closed is True
                    or self._view._children.get(record.session_id) is not record.runtime
                ):
                    raise UnknownAgentError(
                        f"Unknown agent_id instance handle: {args.agent_id}. Check the agent_id with check_agents"
                    )
                self._view._require_task_identity_allowed(parent, record.identity)
                previous_idle_since = record.idle_since
                reserved_child_generation = (
                    record.runtime.agent_loop._session_generation
                )
                reserved_parent_authority_revision = (
                    parent.agent_loop._authority_revision
                )
                record.reserved_for_replacement = False
                # Keep ownership through the launch acknowledgment: this
                # generator's finalizer may run after another retask reserves.
                record.state = _AgentState.RUNNING
                record.idle_since = None
                self._view._rearm_reaper_locked()
        else:
            record = None
        created_record = False
        launch_identity: str | None = None
        effective_task = summary_task if summary_task is not None else args.task
        prepared_reconfiguration = None
        override_diagnostics: dict[str, object] | None = None
        if record is not None:
            # C4: an explicit override_cooldown request mints its permit here,
            # after the record/admission checks and before the candidate
            # resolution (the deep validation must resolve the cooled forced
            # deployment THROUGH the permit). A ledger denial raises before
            # any budget is consumed. The mint runs INSIDE the
            # rollback-protected region: the reuse reservation above already
            # flipped the record to RUNNING, so a denial (or a model-resolution
            # failure) must restore the retained record to IDLE — stranding it
            # RUNNING leaves it permanently busy with no run.
            from chartreux.app_server._subagent_launch import (
                bind_override_to_loop,
                override_launch_diagnostics,
            )

            mint: OverrideMint | None = None
            try:
                mint = self._view._launch_admission.mint_override_permit(
                    parent, args, committed=record.runtime.agent_loop.committed_model
                )
            except BaseException:
                if mint is not None:
                    mint.ledger.release(mint.permit)
                await self._view._rollback_reused_claim(
                    record, previous_idle_since, apply_policy=replacement is not None
                )
                raise
            if args.config is not None or isinstance(
                record.runtime.agent_loop.launch_overrides, LaunchConfig
            ):
                try:
                    if mint is not None:
                        # C4: the minted permit must reach the model-assignment
                        # availability predicate, and the registry's thin
                        # delegate has a fixed signature — the override path
                        # resolves through the admission collaborator directly.
                        retained_candidate = (
                            self._view._launch_admission.resolve_candidate(
                                parent,
                                args,
                                runtime=record.runtime,
                                identity=record.identity,
                                override_permit=mint.permit,
                                override_ledger=mint.ledger,
                            )
                        )
                    else:
                        retained_candidate = self._view._resolve_launch_candidate(
                            parent,
                            args,
                            runtime=record.runtime,
                            identity=record.identity,
                        )
                except BaseException:
                    if mint is not None:
                        mint.ledger.release(mint.permit)
                    await self._view._rollback_reused_claim(
                        record,
                        previous_idle_since,
                        apply_policy=replacement is not None,
                    )
                    raise
            if retained_candidate is not None:
                try:
                    assert reserved_child_generation is not None
                    assert reserved_parent_authority_revision is not None
                    prepared_reconfiguration = await record.runtime.agent_loop.prepare_launch_reconfiguration(
                        retained_candidate,
                        expected_session_generation=reserved_child_generation,
                        expected_parent_authority_revision=reserved_parent_authority_revision,
                    )
                except BaseException:
                    if mint is not None:
                        mint.ledger.release(mint.permit)
                    await self._view._rollback_reused_claim(
                        record,
                        previous_idle_since,
                        apply_policy=replacement is not None,
                    )
                    raise
            if parent.turns._projector is not None:
                try:
                    if resume is None:
                        # WP2: a resumed child session keeps its original
                        # transcript link; re-linking to a synthesized tool
                        # call would corrupt the parent's projection.
                        await parent.turns.link_subagent(
                            ctx.tool_call_id, record.session_id
                        )
                    async with self._view._registry_lock:
                        self._view._validate_background_admission(
                            parent, parent_identity
                        )
                        if (
                            self._view._agent_records.get(record.agent_id) is not record
                            or self._view._children.get(record.session_id)
                            is not record.runtime
                            or record.runtime._closed is True
                        ):
                            raise RuntimeError(
                                "Agent admission changed during child reuse"
                            )
                except BaseException:
                    if prepared_reconfiguration is not None:
                        await record.runtime.agent_loop.abort_launch_reconfiguration(
                            prepared_reconfiguration
                        )
                    if mint is not None:
                        mint.ledger.release(mint.permit)
                    await self._view._rollback_reused_claim(
                        record,
                        previous_idle_since,
                        apply_policy=replacement is not None,
                    )
                    raise
            if mint is not None:
                # The permit binds to the retained child's NEXT logical model
                # call (C3's slot channel); the diagnostics snapshot is taken
                # pre-start, before the child consumes the binding.
                bind_override_to_loop(record.runtime.agent_loop, mint)
                override_diagnostics = override_launch_diagnostics(
                    record.runtime.agent_loop
                )
            runtime = record.runtime
        else:
            requested_identity = requested_identity_reference(args)
            if requested_identity is not None:
                self._view._require_task_identity_allowed(parent, requested_identity)
            runtime, launch_identity = await self._view._create_registered_child(
                parent, args, ctx
            )
            # C4: the fresh launch minted its permit inside create_registered
            # and bound it to the child's first model call; snapshot the
            # bounded diagnostics pre-start, before the child consumes the
            # binding.
            from chartreux.app_server._subagent_launch import (
                override_launch_diagnostics,
            )

            override_diagnostics = override_launch_diagnostics(runtime.agent_loop)
            agent_id = await self._view._issue_agent_id(parent)
            # WP5: a claimed replacement launch (args.agent_id is None with
            # a config.predecessor_agent_id claim) gets a NEW agent_id whose
            # record carries the predecessor/package lineage from the claim.
            claim = args.config
            # The record carries the launch-resolved identity binding (11c):
            # the same (identity, revision) pair the launch envelope persists,
            # read back from the constructed child's committed binding.
            record_identity, record_identity_revision = (
                runtime.agent_loop._identity_binding()
            )
            record = AgentRecord(
                agent_id=agent_id,
                identity=record_identity,
                identity_revision=record_identity_revision,
                session_id=runtime.agent_loop.session_id,
                runtime=runtime,
                root_generation=parent.agent_loop._session_generation,
                parent_session_id=parent.agent_loop.session_id,
                initial_task_summary=normalize_task_summary(
                    args.task_summary, fallback=effective_task
                ),
                state=_AgentState.RUNNING,
                work_package_id=claim.work_package_id if claim is not None else None,
                predecessor_agent_id=(
                    claim.predecessor_agent_id if claim is not None else None
                ),
                # The claimed tier wins; otherwise the record carries the
                # launch-resolved effective tier (the route target's explicit
                # tier), so the recovery boundary decides on the package's
                # true tier instead of defaulting conservatively.
                package_tier=(
                    claim.package_tier
                    if claim is not None and claim.package_tier is not None
                    else runtime.agent_loop.launch_package_tier
                ),
            )
            try:
                async with self._view._registry_lock:
                    self._view._validate_background_admission(parent, parent_identity)
                    if self._view._children.get(record.session_id) is not runtime:
                        raise RuntimeError(
                            "Agent admission changed during child creation"
                        )
                    self._view._agent_records[agent_id] = record
            except BaseException:
                await self._view._rollback_created_child(
                    parent, runtime, ctx.tool_call_id
                )
                raise
            created_record = True
            # Review R7: a claimed replacement inherits the predecessor's
            # cumulative execution budgets — a fresh loop starts every
            # middleware limit from zero, so without the transfer the
            # replacement renews the budgets the predecessor consumed.
            if claim is not None and claim.predecessor_agent_id is not None:
                predecessor = self._view._agent_records.get(claim.predecessor_agent_id)
                if predecessor is not None:
                    _transfer_cumulative_stats(
                        predecessor.runtime.agent_loop.stats, runtime.agent_loop.stats
                    )

        record.work_slot = slot
        agent_id = record.agent_id
        # WP2: a resume admission publishes the successor run id allocated
        # at admission; every other path allocates a fresh one.
        run_id = (
            resume.successor_run_id
            if resume is not None
            else self._view._new_run_id(agent_id)
        )
        # B2a: every run admission — fresh, reused, replacement successor,
        # resume successor — anchors the per-run baseline here, downstream
        # of the cumulative-stat transfer and the resume stats rebind, so a
        # replacement's anchor reflects the inherited baseline. The anchor
        # is per-run state, deliberately outside _CUMULATIVE_STAT_FIELDS.
        runtime.agent_loop.stats.anchor_run_baseline()
        prior_current_run = record.current_run
        prior_retained = record.retained
        prior_latest_run_id = record.latest_run_id
        prior_registry_latest_present = agent_id in self._view._latest_run_ids
        prior_registry_latest = self._view._latest_run_ids.get(agent_id)
        prior_idle_since = record.idle_since
        previous_event_sink = runtime.turns._event_sink
        publication_token = object()
        accumulator = SubagentRunAccumulator()
        # P2/P3 wiring: install the run's receipt sink on the child loop —
        # the post-tool finalization seam emits each invocation's receipt
        # into THIS run's accumulator chronology. Cleared at run teardown:
        # the loop persists across a retained agent's runs, and a stale
        # sink must never receive a later run's receipts (P3 requires a
        # fresh accumulator per run).
        completion_evidence_sink = accumulator.completion_evidence_sink()
        runtime.agent_loop._completion_evidence_sink = completion_evidence_sink
        completion_placeholder = asyncio.get_running_loop().create_future()
        # P4: freeze the completion contract at admission. A resume carries
        # the failed run's run-scoped snapshot (contract and admission HEAD
        # baseline); every other admission — fresh, reuse, replacement —
        # freezes a NEW contract from this launch's args.
        if resume is not None:
            completion_contract = resume.failed_run.completion_contract
            admission_head = resume.failed_run.admission_head
        else:
            completion_contract, admission_head = freeze_run_admission(
                args=args,
                identity=record.identity,
                identity_revision=record.identity_revision,
                tier=(
                    retained_candidate.effective_tier
                    if retained_candidate is not None
                    else runtime.agent_loop.launch_package_tier
                ),
                runtime=runtime,
                parent=parent,
            )
        # P14: the ONE advisory budget-fit log line at admission. Advisory
        # only: it never gates admission and never rejects a launch.
        run_budget = runtime.agent_loop.runtime_policy
        logger.info(
            "Budget-fit advisory for %s/%s (%s): %s",
            agent_id,
            run_id,
            record.identity,
            _budget_fit_description(
                run_budget.run_max_price, run_budget.run_max_session_tokens
            ),
        )
        run_record = RunRecord(
            run_id=run_id,
            agent_id=agent_id,
            identity=record.identity,
            identity_revision=record.identity_revision,
            status=RunStatus.RUNNING,
            completion_task=completion_placeholder,
            task_summary=normalize_task_summary(
                args.task_summary, fallback=effective_task
            ),
            # P1-c2: the run's claimed difficulty category — non-None only on
            # a demanding-route launch, surfaced on the acknowledgment.
            difficulty=(
                args.config.difficulty.value
                if args.config is not None and args.config.difficulty is not None
                else None
            ),
            completion_contract=completion_contract,
            admission_head=admission_head,
        )
        record.current_run = run_record
        record.latest_run_id = run_id
        self._view._latest_run_ids[agent_id] = run_id
        # The run's attribution join key: the child loop stamps every ledger
        # record made during this run with the run id (cleared at run end).
        runtime.agent_loop.run_identity = run_id
        # C5b: the first-semantic-progress deadline arms ONLY for an
        # explicitly marked resumed context (the WP2 resume admission);
        # every other admission clears any stale watch, so ordinary runs
        # are structurally unaffected.
        if resume is not None:
            arm_resume_progress_deadline(runtime.agent_loop)
        else:
            clear_resume_progress_deadline(runtime.agent_loop)
        # Commit the retention policy for this admission. A fresh launch
        # resolves explicit task arg -> generation snapshot default; a reuse
        # inherits the committed policy unless the caller explicitly overrides
        # it (retain=False is a final continuation that closes after this run).
        if created_record:
            assert launch_identity is not None
            # Stable first-admission timestamp for deterministic Active
            # ordering; preserved across reuse and run-history pruning.
            record.first_admitted_at = self._view._clock()
            record.retained = (
                args.retain
                if args.retain is not None
                else self._view._retention_policy.retain_agents
            )
            # Capture the identity's display label for later naming work
            # (design §5): a built-in identity's shipped display name. A
            # custom identity prompt carries no explicit display name, so
            # the derivation helper falls back to the identity reference.
            builtin_identity = get_builtin_identity(
                record.identity.removeprefix("builtin:")
            )
            record.identity_display_label = (
                builtin_identity.display_name if builtin_identity is not None else None
            )
        elif args.retain is not None:
            record.retained = args.retain
        run_record.start_count = sum(
            message.role is Role.assistant for message in runtime.agent_loop.messages
        )
        completion_metadata_mark = runtime.agent_loop.completion_metadata_mark()
        return BackgroundExecutionContext(
            runtime=runtime,
            record=record,
            run_record=run_record,
            agent_id=agent_id,
            run_id=run_id,
            tool_call_id=ctx.tool_call_id,
            created_record=created_record,
            launch_identity=launch_identity,
            previous_idle_since=previous_idle_since,
            reserved_child_generation=reserved_child_generation,
            reserved_parent_authority_revision=reserved_parent_authority_revision,
            retained_candidate=retained_candidate,
            prepared_reconfiguration=prepared_reconfiguration,
            prior_current_run=prior_current_run,
            prior_retained=prior_retained,
            prior_latest_run_id=prior_latest_run_id,
            prior_registry_latest_present=prior_registry_latest_present,
            prior_registry_latest=prior_registry_latest,
            prior_idle_since=prior_idle_since,
            previous_event_sink=previous_event_sink,
            accumulator=accumulator,
            completion_evidence_sink=completion_evidence_sink,
            completion_metadata_mark=completion_metadata_mark,
            completion_placeholder=completion_placeholder,
            publication_token=publication_token,
            replacement_decision=replacement_decision,
            override_diagnostics=override_diagnostics,
        )

    async def rollback_background(
        self,
        context: BackgroundExecutionContext,
        parent: SessionRuntime,
        *,
        replacement: _ReplacementClaim | None,
    ) -> None:
        """Roll back a failed background admission.

        The former rollback region of the registry's admitted run (ASC area
        3, step 3): the prepared-start abort, the reconfiguration rollback,
        the event-sink restoration, the completion cancellation, the write
        token consumption (the results publication decision — the decision
        itself is unused on rollback), the prior-state restore, and the
        created/reused rollback, moved intact. The caller's except block
        keeps the re-raise tail.

        C4: an unaccepted run releases its bound override permit — a typed
        no-op when the transport already consumed it or another net ran.
        """
        from chartreux.app_server._subagent_launch import release_bound_override

        release_bound_override(context.runtime.agent_loop)
        if context.action is not None and not context.committed:
            context.action.abort()
        if context.prepared_reconfiguration is not None and not context.committed:
            if context.backend_publication is not None:
                context.runtime.agent_loop.rollback_launch_reconfiguration(
                    context.prepared_reconfiguration, context.backend_publication
                )
            else:
                await context.runtime.agent_loop.abort_launch_reconfiguration(
                    context.prepared_reconfiguration
                )
        context.runtime.turns._event_sink = context.previous_event_sink
        # P2/P3 rollback: the unaccepted run's receipt sink is spent —
        # clear it so a retained loop never leaks into a later run.
        if (
            context.runtime.agent_loop._completion_evidence_sink
            is context.completion_evidence_sink
        ):
            context.runtime.agent_loop._completion_evidence_sink = None
        attempted = context.run_record.completion_task
        if not attempted.done():
            attempted.cancel()
        with suppress(asyncio.CancelledError):
            await attempted
        async with self._view._registry_lock:
            self._view._publish_locked(context)
            owns_attempt = (
                self._view._agent_records.get(context.agent_id) is context.record
                and context.record.current_run is context.run_record
            )
            if owns_attempt:
                context.record.current_run = context.prior_current_run
                context.record.retained = context.prior_retained
                context.record.latest_run_id = context.prior_latest_run_id
                context.record.idle_since = context.prior_idle_since
                if self._view._latest_run_ids.get(context.agent_id) == context.run_id:
                    if context.prior_registry_latest_present:
                        assert context.prior_registry_latest is not None
                        self._view._latest_run_ids[context.agent_id] = (
                            context.prior_registry_latest
                        )
                    else:
                        self._view._latest_run_ids.pop(context.agent_id, None)
        if context.created_record:
            async with self._view._registry_lock:
                if self._view._agent_records.get(context.agent_id) is context.record:
                    self._view._agent_records.pop(context.agent_id)
            await self._view._rollback_created_child(
                parent, context.runtime, context.tool_call_id
            )
        else:
            await self._view._rollback_reused_claim(
                context.record,
                context.previous_idle_since,
                apply_policy=replacement is not None,
            )

    def _build_failure_envelope(
        self,
        context: BackgroundExecutionContext,
        *,
        observation: FailureObservation | None,
        turn: PublicTurn | None,
        turns_used: int,
    ) -> FailureEnvelope | None:
        """Build the public typed failure envelope for one failed run (WP1).

        Total by contract: the envelope is additive diagnostics — any
        construction failure is logged and reported as ``None`` (no
        envelope), never masking the primary run failure. The envelope is
        bounded and redacted; full diagnostics stay in the private WP0
        observation carrier.
        """
        try:
            agent_loop = context.runtime.agent_loop
            committed = agent_loop.committed_model
            # Review R11: the attempted deployment frozen at the completion
            # boundary — the identity that actually served the failed attempt
            # (the same attempt the observed endpoint describes) — preferred
            # over the committed identity the failed-attempt rollback
            # restored.
            attempted_deployment = (
                observation.attempted_deployment
                if observation is not None
                and observation.attempted_deployment is not None
                else (
                    (committed.base_model, committed.provider, committed.wire_name)
                    if committed is not None
                    else None
                )
            )
            exclusion_reasons: list[str] | None = None
            if turn is not None and turn.error is not None:
                details = turn.error.details
                if isinstance(details, dict):
                    exclusions = details.get("exclusions")
                    if isinstance(exclusions, list):
                        exclusion_reasons = [
                            reason
                            for item in exclusions
                            if isinstance(item, dict)
                            and isinstance(reason := item.get("reason"), str)
                        ]
            tool_calls_present, tool_calls_pending = _tool_effect_inputs(
                agent_loop.messages
            )
            endpoint_correlated = False
            endpoint_identity = canonical_endpoint_identity(
                observation.endpoint if observation is not None else None
            )
            if endpoint_identity is not None:
                endpoint_correlated = agent_loop.config_orchestrator.availability_registry.endpoint_correlation_active(
                    endpoint_identity
                )
            return build_failure_envelope(
                observation=observation,
                agent_id=context.agent_id,
                run_id=context.run_id,
                turn_id=context.turn_id,
                turns_used=turns_used,
                attempted_deployment=attempted_deployment,
                configured_model=agent_loop.config.active_model or None,
                stop_reason=(
                    context.run_record.stop_reason.value
                    if context.run_record.stop_reason is not None
                    else None
                ),
                exclusion_reasons=exclusion_reasons,
                tool_calls_present=tool_calls_present,
                tool_calls_pending=tool_calls_pending,
                endpoint_correlated=endpoint_correlated,
                # Review R10: a resumed run's envelope keeps the record's
                # recovery identity — the episode chain and the actual
                # same-agent resume count — never a fresh chain derived
                # per run or a reset counter.
                same_agent_resumes_used=context.record.same_agent_resumes_used,
                recovery_chain_id=context.record.recovery_chain_id,
                # WP5: the envelope carries the run's predecessor/package
                # lineage (the WP4 recovery controller keys endpoint
                # correlation on the envelope's work_package_id).
                work_package_id=context.record.work_package_id,
                predecessor_agent_id=context.record.predecessor_agent_id,
            )
        except Exception as exc:
            logger.warning("Failed to build failure envelope", exc_info=exc)
            return None

    @staticmethod
    def _result_update(
        context: BackgroundExecutionContext,
        failure_envelope: FailureEnvelope | None,
        run_status: RunStatus | None,
    ) -> dict[str, Any]:
        """Result update carrying identity/metadata plus typed failure evidence."""
        update: dict[str, Any] = {
            "agent_id": context.agent_id,
            "run_id": context.run_id,
            "metadata": context.runtime.agent_loop.completion_metadata_since(
                context.completion_metadata_mark
            ),
        }
        if failure_envelope is not None:
            update["failure"] = failure_envelope
            update["run_status"] = run_status
            update["recovery"] = RecoveryReference.from_envelope(failure_envelope)
        return update

    async def monitor(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        context: BackgroundExecutionContext,
        parent: SessionRuntime,
        slot: _ActiveWorkSlot,
    ) -> None:
        """Monitor one background execution and finalize its result.

        The former monitor closure of the registry's admitted run (ASC area
        3, step 4): the run observation, the result build, the residency and
        publication transaction, the completion notification, and the
        close-on-completion transition, moved intact with the scheduling
        unchanged (the monitor task creation stays with the acceptance
        flow). The turn id is read from the context field and narrowed with
        an assert (the teardown_foreground precedent); the parent runtime
        and the capacity slot are parameters. The token-based publication
        authority (the residency check, the notification suppression, and
        the exact write-token check/pop) is decided by the results
        collaborator's publish_locked through the view; the result store
        write and the pending-notification add stay in this locked block,
        gated by the returned PublicationDecision. Registry state is read
        through the view at call time; the event-sink restore and the
        committed gate are unchanged.
        """
        turn_id = context.turn_id
        assert turn_id is not None
        try:
            turn = await context.runtime.turns.wait_for_operation(turn_id)
            observation: FailureObservation | None = None
            if turn.error is not None:
                context.accumulator.record_error(turn.error.message)
                # The turn's PublicError is message-only; link the typed
                # observation captured before that projection (WP0 F5) to
                # the failed run so recovery can classify it later. The
                # carrier stays private — the public projection below is
                # unchanged.
                observation = turn_observation(turn_id)
                if observation is not None:
                    observation = preserve_run_observation(
                        context.agent_id, context.run_id, observation
                    )
            end_count = sum(
                message.role is Role.assistant
                for message in context.runtime.agent_loop.messages
            )
            turns_used = end_count - context.run_record.start_count
            context.run_record.status, context.run_record.stop_reason = (
                self._view._resolve_run_outcome(context.run_record, turn)
            )
            # P6b case (a): a run cancelled before the audit starts takes
            # the cancellation disposition — the reason is recorded, the
            # stub finalization is cached, and no check spawns.
            if context.run_record.status is RunStatus.CANCELLED:
                apply_cancellation_disposition(
                    context.accumulator,
                    context.run_id,
                    context.run_record.completion_contract,
                    context.run_record.admission_head,
                    context.run_record.stop_reason,
                )
            # WP1: the public typed failure envelope rides the failed run's
            # result and run record; bounded and redacted, full diagnostics
            # stay in the private observation carrier.
            failure_envelope = (
                self._build_failure_envelope(
                    context, observation=observation, turn=turn, turns_used=turns_used
                )
                if turn.error is not None
                else None
            )
            if failure_envelope is not None:
                context.run_record.failure = failure_envelope
            # P6a: the once-only finalization, awaited immediately BEFORE
            # the result build — build_result reads the accumulator's
            # finalization cache and runs no checks itself. No registry
            # lock is held across the check subprocesses. P6b wraps the
            # finalization with the drain so a cancellation mid-audit kills
            # this run's in-flight check processes and discards the partial
            # findings.
            await finalize_run_with_drain(
                run_id=context.run_id,
                accumulator=context.accumulator,
                contract=context.run_record.completion_contract,
                admission_head=context.run_record.admission_head,
                runtime=context.runtime,
            )
            # P12: the call site wires the composition surface — the run
            # key (the once-only finalization cache read), the frozen
            # contract (the mandate surface and the audit mode), the
            # worker's receipt claim (operative only on a healthy
            # completed turn), and the acceptance-check runner.
            result = context.accumulator.build_result(
                turns_used=turns_used,
                completed=turn.status is PublicTurnStatus.COMPLETED,
                run_id=context.run_id,
                contract=context.run_record.completion_contract,
                claim=operative_completion_claim(
                    context.accumulator,
                    turn_completed=context.run_record.status is RunStatus.COMPLETED,
                ),
                runner=_check_runner(context.runtime),
            )
            # P12: the publication-rejection lifecycle — an audit rejection
            # finalizes the run FAILED with the measured rejection envelope;
            # the result's audit stamp, the run status, and the envelope
            # agree.
            envelope = rejection_envelope(context, result)
            if envelope is not None:
                context.run_record.status = RunStatus.FAILED
                context.run_record.stop_reason = RunStopReason.ERROR
                context.run_record.failure = envelope
                failure_envelope = envelope
            context.run_record.result = result.model_copy(
                update=self._result_update(
                    context, failure_envelope, context.run_record.status
                )
            )
        except asyncio.CancelledError:
            context.run_record.status, context.run_record.stop_reason = (
                self._view._resolve_run_outcome(
                    context.run_record,
                    context.runtime.turns.operation_terminal_turn(turn_id),
                )
            )
            # P6b: the cancelled run takes the cancellation disposition even
            # when the monitor never reached the normal path — the reason is
            # recorded (no worker credit), the stub finalization is cached,
            # and exactly one terminal result builds with completed=False.
            apply_cancellation_disposition(
                context.accumulator,
                context.run_id,
                context.run_record.completion_contract,
                context.run_record.admission_head,
                context.run_record.stop_reason,
            )
            # P12: the cancelled run wires the mandate surface and the
            # finalization cache read; the worker's claim is NOT operative
            # on a cancelled run, so no claim rides (the cancellation
            # reason stays the run's primary disposition).
            context.run_record.result = context.accumulator.build_result(
                turns_used=sum(
                    message.role is Role.assistant
                    for message in context.runtime.agent_loop.messages
                )
                - context.run_record.start_count,
                completed=context.run_record.status is RunStatus.COMPLETED,
                run_id=context.run_id,
                contract=context.run_record.completion_contract,
                runner=_check_runner(context.runtime),
            ).model_copy(
                update={
                    "agent_id": context.agent_id,
                    "run_id": context.run_id,
                    "metadata": context.runtime.agent_loop.completion_metadata_since(
                        context.completion_metadata_mark
                    ),
                }
            )
            raise
        except Exception as exc:
            context.accumulator.record_error(str(exc))
            # Preserve the typed monitor failure (WP0 F5): the monitor
            # itself failed to observe the terminal turn, so capture the
            # exception here — the turn-level observation (if any) stays
            # turn-keyed. The carrier is private; the public projection
            # below is unchanged.
            monitor_observation = observe_exception(
                exc,
                phase="monitor",
                turn_id=turn_id,
                agent_id=context.agent_id,
                run_id=context.run_id,
            )
            record_observation(monitor_observation)
            context.run_record.status = RunStatus.FAILED
            context.run_record.stop_reason = RunStopReason.ERROR
            # Measure turns from observed work (assistant messages), never
            # from a synthetic zero: a monitor failure after real work must
            # not erase the run's progress (WP0 F5 zero-turn fix).
            end_count = sum(
                message.role is Role.assistant
                for message in context.runtime.agent_loop.messages
            )
            turns_used = end_count - context.run_record.start_count
            # WP1: a harness-side monitor failure is represented honestly as
            # an INTERNAL-class envelope (phase=monitor), never relabeled.
            monitor_envelope = self._build_failure_envelope(
                context,
                observation=monitor_observation,
                turn=None,
                turns_used=turns_used,
            )
            # Review R12: the PRIMARY failure stays authoritative evidence —
            # a secondary result/finalization failure never replaces the
            # run's published typed envelope (nor the result's recovery
            # reference) with an INTERNAL-class monitor one.
            if context.run_record.failure is None:
                context.run_record.failure = monitor_envelope
            failure_envelope = context.run_record.failure
            # P6a: the once-only finalization is awaited before this
            # fallback build too — a cache hit when the normal path already
            # finalized (a result-build failure), a fresh finalization when
            # the monitor never reached the normal build. The cancellation
            # path caches the stub finalization (P6b) and runs no checks.
            await finalize_run_with_drain(
                run_id=context.run_id,
                accumulator=context.accumulator,
                contract=context.run_record.completion_contract,
                admission_head=context.run_record.admission_head,
                runtime=context.runtime,
            )
            # P12: the fallback build wires the mandate surface and the
            # finalization cache read; the worker's claim is NOT operative
            # here — the run's primary failure stays authoritative (no
            # rejection envelope ever replaces it).
            context.run_record.result = context.accumulator.build_result(
                turns_used=turns_used,
                completed=False,
                run_id=context.run_id,
                contract=context.run_record.completion_contract,
                runner=_check_runner(context.runtime),
            ).model_copy(
                update=self._result_update(context, failure_envelope, RunStatus.FAILED)
            )
        finally:
            if context.run_record.result is not None:
                context.run_record.result = context.run_record.result.model_copy(
                    update={"stop_reason": context.run_record.stop_reason}
                )
            active_task = context.runtime.turns._active_task
            if context.record.reserved_for_replacement:
                # The old execution is joined, but its allocation now
                # belongs to the handoff, including notification awaits.
                pass
            elif active_task is not None and not active_task.done():
                # Release/drain may cancel the monitor before provider
                # cancellation finishes; keep counting the actual work.
                active_task.add_done_callback(
                    lambda _: self._view._active_work_slots.discard(slot)
                )
            else:
                self._view._active_work_slots.discard(slot)
            context.runtime.turns._event_sink = context.previous_event_sink
            if context.runtime.agent_loop.run_identity == context.run_id:
                context.runtime.agent_loop.run_identity = None
            # P2/P3 run-end hygiene: the run's receipt sink is spent — a
            # retained loop must never hand a later run's receipts to this
            # run's accumulator. Guarded: a successor run may already have
            # installed its own sink on the same loop.
            if (
                context.runtime.agent_loop._completion_evidence_sink
                is context.completion_evidence_sink
            ):
                context.runtime.agent_loop._completion_evidence_sink = None
            # C5b run-end hygiene: the run's resume-progress watch is spent —
            # no later run of this loop inherits it.
            clear_resume_progress_deadline(context.runtime.agent_loop)
            if not context.committed:
                return  # noqa: B012
            context.run_record.completed_at = self._view._clock()
            committed_model = context.runtime.agent_loop.committed_model
            if committed_model is not None:
                context.run_record.terminal_identity = (
                    committed_model.base_model,
                    committed_model.provider,
                    committed_model.wire_name,
                )
            key = (context.agent_id, context.run_id)
            async with self._view._registry_lock:
                decision = self._view._publish_locked(context)
                if decision.resident:
                    context.record.compacting = False
                    context.record.compaction_entry_id = None
                    context.record.stop_reason = context.run_record.stop_reason
                    committed_model = context.runtime.agent_loop.committed_model
                    context.record.base_model = (
                        committed_model.base_model
                        if committed_model is not None
                        else None
                    )
                    if committed_model is not None:
                        context.record.effective_model = format_model_display_name(
                            committed_model.provider, committed_model.wire_name
                        )
                    context.record.active_provider = (
                        committed_model.provider
                        if committed_model is not None
                        else None
                    )
                    context.record.state = (
                        _AgentState.RUNNING
                        if context.record.reserved_for_replacement
                        else _AgentState.FINALIZING
                    )
                    context.record.idle_since = context.run_record.completed_at
                    context.record.last_task_summary = context.run_record.task_summary
                    context.record.last_run_status = context.run_record.status
                    context.record.current_run = None
                    if context.run_record not in context.record.run_history:
                        context.record.run_history.append(context.run_record)
                        if len(context.record.run_history) > _MAX_RUN_HISTORY:
                            context.record.run_history = context.record.run_history[
                                -_MAX_RUN_HISTORY:
                            ]
                if decision.write_permitted and context.run_record.result is not None:
                    self._view._result_store[key] = StoredRunResult(
                        agent_id=context.agent_id,
                        run_id=context.run_id,
                        result=context.run_record.result,
                        completed_at=context.run_record.completed_at,
                        root_generation=context.record.root_generation,
                        terminal_identity=context.run_record.terminal_identity,
                    )
                if decision.notify_permitted:
                    self._view._pending_notifications.add(key)
                self._view._expire_results_locked()
            if not decision.notify_permitted:
                return  # noqa: B012
            try:
                await self._view._emit_agents_update()
            except Exception as exc:
                logger.warning(
                    "Failed to publish background agent update", exc_info=exc
                )
            try:
                async with self._view._registry_lock:
                    notification_permitted = (
                        self._view._generation_is_current(
                            context.record.root_generation
                        )
                        and self._view._agent_records.get(context.agent_id)
                        is context.record
                        and self._view._children.get(context.record.session_id)
                        is context.runtime
                        and not self._view._draining_children
                        and context.agent_id not in self._view._suppressed_notifications
                        and key in self._view._result_store
                        and key in self._view._pending_notifications
                    )
                if not notification_permitted:
                    return  # noqa: B012
                # Notify the parent that a background run completed.
                status_word = (
                    "budget-stopped (budget exceeded)"
                    if context.run_record.stop_reason is RunStopReason.BUDGET_EXCEEDED
                    else "budget-stopped (budget unverifiable)"
                    if context.run_record.stop_reason
                    is RunStopReason.BUDGET_UNVERIFIABLE
                    else "completed"
                    if context.run_record.status is RunStatus.COMPLETED
                    else "failed"
                    if context.run_record.status is RunStatus.FAILED
                    else "cancelled (by user)"
                    if context.run_record.stop_reason is RunStopReason.USER_CANCELLED
                    else "retasked"
                    if context.run_record.stop_reason is RunStopReason.RETASKED
                    else "cancelled"
                )
                # The notification derives from this run's own
                # record state at completion (§5): name alongside
                # handle, run, and outcome.
                notification = (
                    f"Background agent {self._view._record_name(context.record)} "
                    f"({context.agent_id}, run {context.run_id})"
                    f" {status_word}."
                    " Use get_agent_result to retrieve the result."
                )
                if (
                    context.run_record.stop_reason is not RunStopReason.USER_CANCELLED
                    and parent.turns.active_turn is None
                    and parent.turns._active_task is None
                    and not parent.turns.queue_state.items
                ):
                    try:
                        _response, _start = parent.turns.start(
                            TurnStartParams(
                                session_id=parent.agent_loop.session_id,
                                message=[TextContentBlock(text=notification)],
                            )
                        )
                        _start()
                    except Exception:
                        parent.agent_loop._pending_injected_messages.append(
                            LLMMessage(
                                role=Role.user, content=notification, injected=True
                            )
                        )
                else:
                    parent.agent_loop._pending_injected_messages.append(
                        LLMMessage(role=Role.user, content=notification, injected=True)
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to finalize background agent notification", exc_info=exc
                )
            finally:
                async with self._view._registry_lock:
                    self._view._pending_notifications.discard(key)
                    if (
                        self._view._agent_records.get(context.agent_id)
                        is context.record
                        and self._view._children.get(context.record.session_id)
                        is context.runtime
                        and not context.record.reserved_for_replacement
                        and context.record.state is _AgentState.FINALIZING
                        and self._view._generation_is_current(
                            context.record.root_generation
                        )
                    ):
                        # Completion transition (WP2): a recoverably failed
                        # run SUSPENDS the record — warm, resident, explicitly
                        # resumable — instead of closing it; a retained
                        # instance otherwise returns to idle; an unretained
                        # instance closes (detach + `closed` tombstone +
                        # tracked runtime close) after the result was
                        # published above.
                        if not self._view._suspend_after_failure_locked(
                            context.record, context.run_record
                        ) and not self._view._close_unretained_locked(
                            context.record,
                            context.run_record,
                            name=(
                                f"vibe-subagent-close-on-completion:{context.agent_id}"
                            ),
                        ):
                            context.record.state = _AgentState.IDLE
                    # WP4: observe the settled run for the automatic recovery
                    # controller AFTER the completion transition — a
                    # recoverably failed run is SUSPENDED by then. INERT
                    # unless [recovery.subagents] enabled.
                    self._view._observe_run_completion_locked(
                        context.record, context.run_record
                    )
                    self._view._expire_results_locked()
                    if not self._view._draining_children:
                        self._view._rearm_reaper_locked()
                if not self._view._draining_children:
                    try:
                        await self._view._emit_agents_update()
                    except Exception as exc:
                        logger.warning(
                            "Failed to publish agent completion transition",
                            exc_info=exc,
                        )

    async def accept_background(
        self,
        context: BackgroundExecutionContext,
        parent: SessionRuntime,
        parent_identity: tuple[str, int],
        slot: _ActiveWorkSlot,
    ) -> None:
        """Accept one prepared background execution and commit the run.

        The former acceptance region of the registry's admitted run (ASC
        area 3, step 5): the admission revalidation under the registry
        lock, the monitor task creation (the scheduling is unchanged — the
        task is created here exactly as the acceptance flow created it), the
        prepared-waiter settle install, the monitor-task tracking, and the
        acceptance writes (the write-token registration routed to the
        results collaborator, the committed flag, the slot transfer, the
        launch metadata install + finalize), moved intact. The start action
        is read from the context field (the existing assert narrows it); the
        parent runtime, the parent identity, and the capacity slot are
        parameters. Registry state is read through the view at call time.
        """
        action = context.action
        async with self._view._registry_lock:
            self._view._validate_background_admission(parent, parent_identity)
            if (
                self._view._agent_records.get(context.agent_id) is not context.record
                or self._view._children.get(context.record.session_id)
                is not context.runtime
                or context.record.current_run is not context.run_record
                or context.runtime._closed is True
            ):
                raise RuntimeError("Agent admission changed before launch")
            if context.reserved_child_generation is not None and (
                context.runtime.agent_loop._session_generation
                != context.reserved_child_generation
                or parent.agent_loop._authority_revision
                != context.reserved_parent_authority_revision
            ):
                raise RuntimeError("Agent authority changed before launch")
            self._view._require_task_identity_allowed(parent, context.record.identity)
            self._view._register_writer_locked(
                context.agent_id, context.run_id, context.publication_token
            )
        monitor_coro = self.monitor(context, parent, slot)
        try:
            context.monitor = asyncio.create_task(
                monitor_coro, name=f"vibe-subagent-background:{context.run_id}"
            )
        except BaseException:
            monitor_coro.close()
            raise
        context.run_record.completion_task = context.monitor

        context.monitor.add_done_callback(context.settle_prepared_waiters)
        self._view._monitor_tasks.add(context.monitor)
        context.monitor.add_done_callback(self._view._monitor_tasks.discard)
        assert action is not None
        if context.prepared_reconfiguration is not None:
            context.backend_publication = (
                context.runtime.agent_loop.publish_launch_reconfiguration(
                    context.prepared_reconfiguration
                )
            )
        action()
        context.run_record.started_at = self._view._clock()
        context.record.stop_reason = None
        context.record.compacting = False
        context.record.compaction_entry_id = None
        context.record.context_window = (
            context.runtime.agent_loop.config.get_active_model().auto_compact_threshold
        )
        if context.prepared_reconfiguration is not None:
            # Acceptance makes the new envelope authoritative in memory
            # before persistence can block on the logger save lock.
            context.runtime.agent_loop.install_launch_metadata()
        if context.retained_candidate is not None:
            context.record.effective_model = (
                context.retained_candidate.effective_model.alias
            )
            context.record.effective_thinking = (
                context.retained_candidate.effective_thinking
            )
        else:
            context.record.effective_model = (
                context.runtime.agent_loop.config.active_model
            )
            context.record.effective_thinking = (
                context.runtime.agent_loop.config.get_active_model().thinking
            )
        committed_model = context.runtime.agent_loop.committed_model
        if committed_model is not None:
            context.record.effective_model = format_model_display_name(
                committed_model.provider, committed_model.wire_name
            )
        context.record.base_model = (
            committed_model.base_model if committed_model is not None else None
        )
        context.record.active_provider = (
            committed_model.provider if committed_model is not None else None
        )
        context.committed = True
        slot.transferred = True
        if context.prepared_reconfiguration is not None:
            assert context.backend_publication is not None
            context.runtime.agent_loop.finalize_launch_reconfiguration(
                context.prepared_reconfiguration, context.backend_publication
            )

    async def acknowledge_background(
        self,
        context: BackgroundExecutionContext,
        *,
        replacement: _ReplacementClaim | None,
        defer_launch_agents_update: bool,
    ) -> TaskResult:
        """Acknowledge one accepted background execution.

        The former acknowledgment region of the registry's admitted run
        (ASC area 3, step 5): the launch-metadata persistence, the initial
        agents update, and the TaskResult construction, moved intact; the
        caller yields the returned acknowledgment. The yield -> return
        restructure keeps the admitted-run generator single-layered: no
        unowned forwarding generator is introduced, and a cancellation
        during the metadata persistence (after acceptance) does not roll
        the accepted run back — accepted execution is NOT acknowledgment
        delivery.
        """
        if context.prepared_reconfiguration is not None:
            # Acceptance has happened: persistence failure is reported without
            # rolling the live configuration or accepted run back.
            await context.runtime.agent_loop.persist_launch_metadata()
        if not defer_launch_agents_update:
            try:
                await self._view._emit_agents_update()
            except Exception as exc:
                logger.warning(
                    "Failed to publish initial background agent update", exc_info=exc
                )
        # An explicit status distinguishes this successful launch acknowledgment
        # from the terminal result that get_agent_result later returns. The
        # guidance is policy-aware: retained agents stay resident and can be
        # continued; unretained agents close on completion with the result
        # still retrievable.
        if context.record.retained:
            policy_guidance = (
                " This agent is retained and stays resident after the run "
                "completes; to continue this instance, use "
                f'task(agent_id="{context.agent_id}", background=true, task="...").'
            )
        else:
            policy_guidance = (
                " This agent is not retained and closes when the run "
                "completes; the result stays retrievable via "
                "get_agent_result. Launch with retain=true to keep a "
                "resident agent."
            )
        # The name is derived from the record's committed admission state;
        # refusals and failed admissions never relabel (§5).
        agent_name = self._view._record_name(context.record)
        response_text = (
            f"Background agent {agent_name} ({context.agent_id}, run {context.run_id}) "
            "launched and running; use check_agents to monitor, "
            "get_agent_result or wait_for_agent to retrieve the "
            "result." + policy_guidance
        )
        # WP5: every substitution emits its milestone notice (with the honest
        # cost implication) on the acknowledgment, and the ack carries the
        # validated decision plus the predecessor/package lineage.
        decision = context.replacement_decision
        if decision is not None and decision.notice:
            response_text += f"\n{decision.notice}"
        return TaskResult(
            response=response_text,
            turns_used=0,
            completed=True,
            status="launched",
            launch_outcome=LaunchOutcome.LAUNCHED,
            agent_id=context.agent_id,
            run_id=context.run_id,
            name=agent_name,
            work_package_id=context.record.work_package_id,
            predecessor_agent_id=context.record.predecessor_agent_id,
            replacement=decision,
            metadata={
                "base_model": context.record.base_model,
                "active_provider": context.record.active_provider,
                "effective_model": context.record.effective_model,
                # P1-c2: the demanding route's structured difficulty category
                # rides the acknowledgment, so escalations are auditable
                # without re-reading transcripts. Admission validated it.
                **(
                    {"difficulty": context.run_record.difficulty}
                    if context.run_record.difficulty is not None
                    else {}
                ),
                **(
                    {
                        "replaced_run_id": replacement.run.run_id,
                        "replacement_run_id": context.run_id,
                    }
                    if replacement is not None
                    else {}
                ),
                # C4: the bounded un-stick diagnostics for an explicit
                # override_cooldown launch (selected deployment, exclusions
                # bypassed, remaining allowance, next permitted time).
                **(
                    {"cooldown_override": context.override_diagnostics}
                    if context.override_diagnostics is not None
                    else {}
                ),
            },
        )
