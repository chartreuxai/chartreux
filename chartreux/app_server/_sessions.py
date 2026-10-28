from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
    Sequence,
)
from contextlib import ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from chartreux.app_server._child_runtime_lifecycle import (
    ChildDrain,
    ChildRuntimeAssembler,
)
from chartreux.app_server._execution import (
    SessionExecution,
    SessionExecutionConflict,
    SessionExecutionKind,
)
from chartreux.app_server._model import ProtocolModel
from chartreux.app_server._projection import (
    project_history,
    project_launch_purposes,
    project_session_log,
)
from chartreux.app_server._recovery_controller import RecoveryController
from chartreux.app_server._root_session import SessionHandoff, rebind_history
from chartreux.app_server._runtime import (
    AgentRuntimeFactory,
    SessionDangerousCommandGrantPort,
    SessionRootGrantPort,
    close_agent_loop,
)
from chartreux.app_server._session_history import SessionHistory
from chartreux.app_server._state import build_public_state
from chartreux.app_server._streaming import BoundedEventQueue, stream_until_complete
from chartreux.app_server._subagent_authority import DangerousCommandAuthority
from chartreux.app_server._subagent_execution import (
    BackgroundExecutionContext as BackgroundExecutionContext,
    ForegroundExecutionContext,
    SubagentExecution,
    apply_cancellation_disposition,
    finalize_run_with_drain,
    freeze_run_admission,
    operative_completion_claim,
)
from chartreux.app_server._subagent_finalization import _check_runner as _check_runner
from chartreux.app_server._subagent_launch import (
    ChildLaunchAdmission,
    override_launch_diagnostics,
)
from chartreux.app_server._subagent_records import (
    _DANGEROUS_COMMAND_DENYLIST_PROBES as _DANGEROUS_COMMAND_DENYLIST_PROBES,
    AgentRecord as AgentRecord,
    DangerousCommandDenylistBlock as DangerousCommandDenylistBlock,
    DangerousCommandGrantEntry as DangerousCommandGrantEntry,
    FrozenCompletionContract as FrozenCompletionContract,
    ResidentTranscriptReadSnapshot as ResidentTranscriptReadSnapshot,
    RunRecord as RunRecord,
    StoredChildSession as StoredChildSession,
    StoredRunResult as StoredRunResult,
    TranscriptReadSnapshot as TranscriptReadSnapshot,
    _accepted_saved_grants as _accepted_saved_grants,
    _ActiveWorkSlot as _ActiveWorkSlot,
    _AgentState as _AgentState,
    _AgentTombstone as _AgentTombstone,
    _blocking_denylist as _blocking_denylist,
    _denylist_blocked_operations as _denylist_blocked_operations,
    _explicit_shell_denylist as _explicit_shell_denylist,
    _grant_entries as _grant_entries,
    _grant_entry_sort_key as _grant_entry_sort_key,
    _grant_wire as _grant_wire,
    _ReplacementClaim as _ReplacementClaim,
)
from chartreux.app_server._subagent_recovery import (
    SubagentRecovery,
    _recovery_config as _recovery_config,
)
from chartreux.app_server._subagent_release import (
    AgentRelease,
    ReleaseContext as ReleaseContext,
)
from chartreux.app_server._subagent_replacement import ReplacementCoordinator
from chartreux.app_server._subagent_results import (
    _MAX_STORED_RESULTS as _MAX_STORED_RESULTS,
    PublicationDecision,
    RunResults,
)
from chartreux.app_server._subagent_retention import AgentRetention, EvictionNotice
from chartreux.app_server._subagent_transcript import TranscriptReads
from chartreux.app_server._suspension_lifecycle import SuspensionLifecycle
from chartreux.app_server._turns import DeliverCallback, TurnController, TurnStartAction
from chartreux.app_server.models import (
    CallbackOutput,
    OpenCallbackState,
    PublicCallbackEntry,
    PublicFailureEnvelope,
    PublicHistoryEntry,
    PublicRecoveryEligibility,
    PublicSessionState,
    PublicTurn,
    PublicTurnQueue,
    PublicTurnStatus,
    PublicTurnStopReason,
    TextContentBlock,
)
from chartreux.app_server.protocol import (
    AgentSummaryModel,
    CallbackResultError,
    RunStopReason as PublicRunStopReason,
    TurnStartParams,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop import _PreparedPolicyReplacement
from chartreux.core.agents.launch import LaunchCandidate
from chartreux.core.agents.manager import requested_identity_reference
from chartreux.core.config._restrictions import (
    SourceRestrictions,
    partition_policy_sources,
)
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import RecoveryConfig
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.events import BaseEvent, ToolStreamEvent
from chartreux.core.llm_models import Role
from chartreux.core.recovery_models import FailureEnvelope, RecoveryEligibility
from chartreux.core.session_types import ChildSessionLink
from chartreux.core.subagents import (
    AgentAvailability,
    AgentBusyError,
    AgentClosedError,
    AgentEviction,
    AgentIdentityMismatchError,
    AgentSummary,
    CancelOutcome,
    CancelResult,
    EffectVerification,
    LaunchConfigError,
    LaunchOutcome,
    ReconcileResult,
    ReleaseAgentOutcome,
    ResumeAgentResult,
    RetentionPolicy,
    RunStatus,
    RunStopReason,
    SubagentRunAccumulator,
    SubagentRunnerPort,
    TaskArgs,
    TaskResult,
    WaitForAgentResult,
    derive_agent_name,
    prepare_subagent_prompt,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.io_port import ToolIOPort
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server.protocol import (
        DangerousCommandAuthorityReadResponse,
        DangerousCommandRevokeResponse,
    )

type Notify = Callable[[str, ProtocolModel], Awaitable[None]]
type NotifyAgents = Callable[
    [list[AgentSummaryModel], list[AgentEviction]], Awaitable[None]
]
type Wakeup = Callable[[float], Awaitable[None]]
type EventWatermark = Callable[[str], int]

# Per root-session registry; retained idle agents do not consume active-work slots.
_DEFAULT_MAX_RUNNING_SUBAGENTS = 16


def _public_failure_envelope(
    envelope: FailureEnvelope | None,
) -> PublicFailureEnvelope | None:
    """Project a run's core failure envelope onto the public wire model (WP1).

    The public app-server layer cannot import core models (app-server
    boundary), so the agents/update projection converts the canonical
    envelope at this server boundary; the wire model mirrors the §1.7
    schema field-for-field (pinned by a mirror-contract test).
    """
    if envelope is None:
        return None
    return PublicFailureEnvelope.model_validate(
        envelope.model_dump(mode="json", by_alias=True)
    )


@dataclass(slots=True)
class SessionRuntime:
    agent_loop: AgentLoop
    turns: TurnController
    execution: SessionExecution
    history: SessionHistory
    retire_accounting: Callable[[], Awaitable[None]] | None = None
    _closed: bool = field(default=False, init=False, repr=False)
    _close_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    async def close(self) -> None:
        task = self._close_task
        if task is None or (task.done() and task.exception() is not None):
            task = asyncio.create_task(
                self._close_owned(),
                name=f"vibe-session-runtime-close:{self.agent_loop.session_id}",
            )
            self._close_task = task
            task.add_done_callback(self._observe_close_outcome)
        await asyncio.shield(task)

    def _observe_close_outcome(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            logger.error(
                "Owned session-runtime cleanup was cancelled: %s",
                self.agent_loop.session_id,
            )
            return
        if exc := task.exception():
            logger.error(
                "Owned session-runtime cleanup failed: %s",
                self.agent_loop.session_id,
                exc_info=exc,
            )

    async def _close_owned(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        for cleanup in (self.turns.close, lambda: close_agent_loop(self.agent_loop)):
            try:
                await cleanup()
            except BaseException as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Failed to close session runtime", errors)
        # Producers must settle before root writer retirement or child untracking;
        # failed cleanup retains the accounting binding for retry.
        if self.retire_accounting is not None:
            await self.retire_accounting()
        self._closed = True


class SessionRuntimeRegistry(SubagentRunnerPort):  # noqa: PLR0904
    def __init__(  # noqa: PLR0915
        self,
        notify_child: Notify,
        deliver_callback: DeliverCallback,
        event_watermark: EventWatermark,
        tool_io: ToolIOPort | None = None,
        runtime_factory: AgentRuntimeFactory | None = None,
        notify_agents: NotifyAgents | None = None,
        clock: Callable[[], float] = time.monotonic,
        wakeup: Wakeup = asyncio.sleep,
        max_running_subagents: int | None = None,
    ) -> None:
        if max_running_subagents is not None and (
            isinstance(max_running_subagents, bool)
            or not isinstance(max_running_subagents, int)
            or max_running_subagents < 1
        ):
            raise ValueError("max_running_subagents must be a positive integer")
        self._max_running_subagents = max_running_subagents
        self._active_work_slots: set[_ActiveWorkSlot] = set()
        self._notify_child = notify_child
        self._notify_agents = notify_agents
        self._deliver_callback = deliver_callback
        self._event_watermark = event_watermark
        self._tool_io = tool_io
        self._runtime_factory = runtime_factory or AgentRuntimeFactory()
        self._root: SessionRuntime | None = None
        self._children: dict[str, SessionRuntime] = {}
        self._child_links: dict[str, tuple[SessionRuntime, str]] = {}
        self._stored_children: dict[str, StoredChildSession] = {}
        self._readable_children: dict[str, SessionRuntime] = {}
        self._ensure_child_lock = asyncio.Lock()
        self._policy_reserved = False
        self._creating_children = 0
        self._agent_records: dict[str, AgentRecord] = {}
        self._evicted_agents: dict[str, _AgentTombstone] = {}
        self._result_store: dict[tuple[str, str], StoredRunResult] = {}
        self._expired_results: set[tuple[str, str]] = set()
        self._latest_run_ids: dict[str, str] = {}
        self._wait_leases: dict[tuple[str, str], int] = {}
        self._pending_notifications: set[tuple[str, str]] = set()
        self._result_write_tokens: dict[tuple[str, str], object] = {}
        # WP5: per-work-package replacement counter (keyed by work_package_id)
        # against recovery.fallback_replacements_max_per_package; the budget
        # spans replacements — a package's replacements are counted, never
        # reset, across the package's lineage.
        self._package_replacements: dict[str, int] = {}
        self._teardown_tasks: set[asyncio.Task[None]] = set()
        self._pending_child_closes: dict[
            str, tuple[SessionRuntime, asyncio.Task[None]]
        ] = {}
        self._detached_child_ids: set[str] = set()
        self._monitor_tasks: set[asyncio.Task[None]] = set()
        self._eviction_tasks: set[asyncio.Task[bool]] = set()
        self._registry_lock = asyncio.Lock()
        self._clock = clock
        self._wakeup = wakeup
        self._reaper_task: asyncio.Task[None] | None = None
        self._reaper_deadline: float | None = None
        # Reaper tasks that handed their schedule to a fresh timer before
        # awaiting eviction cleanup (displaced reapers). Drain owns them
        # until they settle, so drain_children cannot return — or publish
        # its final empty inventory — while a displaced reaper can still
        # deliver a payload.
        self._detached_reaper_tasks: set[asyncio.Task[None]] = set()
        self._pending_evictions: dict[str, object] = {}
        self._retention_policy = RetentionPolicy(
            idle_ttl_seconds=0, max_idle_agents=0, retain_agents=False
        )
        self._generation_identity: tuple[str, int] | None = None
        self._draining_children = False
        # Set while draining when an outstanding inventory publication is
        # suppressed or cancelled (a reaper pruning across the drain
        # boundary, or an in-flight publication aborted by cancelling its
        # reaper): drain must still emit its final empty inventory so the
        # disappearance is never lost.
        self._drain_suppressed_inventory = False
        self._suppressed_notifications: set[str] = set()
        self._suppression_owners: dict[str, int] = {}
        self._admission_closed = False
        self._next_agent_seq = 0
        self._dangerous_command_authority = DangerousCommandAuthority(
            resolve_live_session=self._require_live_session,
            resolve_session_loop=self._resolve_session_loop,
            resolve_root=lambda: self._root,
        )
        self._transcript_reads = TranscriptReads(self)
        self._launch_admission = ChildLaunchAdmission(self)
        self._run_results = RunResults(self)
        self._agent_retention = AgentRetention(self)
        self._replacement = ReplacementCoordinator(self)
        self._agent_release = AgentRelease(self)
        self._execution = SubagentExecution(self)
        self._child_runtime = ChildRuntimeAssembler(self)
        self._child_drain = ChildDrain(self)
        self._recovery = SubagentRecovery(self)
        self._suspension = SuspensionLifecycle(self)
        # WP4: the bounded automatic recovery controller. INERT unless
        # [recovery.subagents] enabled — suspension and the explicit
        # resume/reconcile ports above are untouched either way.
        self._recovery_controller = RecoveryController(self)

    def __getattr__(self, name: str) -> Any:
        # Registries built without __init__ (tests construct bare instances
        # with object.__new__ and set only the state they exercise) lazily
        # wire the collaborators on first use; the reserve/close facade is
        # the only port those bare registries call. The WP4 recovery
        # controller's constructor never touches the view, so lazy wiring is
        # inert for a bare registry (its cancel-all iterates no episodes).
        if name == "_child_drain":
            drain = ChildDrain(self)
            self._child_drain = drain
            return drain
        if name == "_recovery_controller":
            controller = RecoveryController(self)
            self._recovery_controller = controller
            return controller
        raise AttributeError(f"SessionRuntimeRegistry has no attribute {name!r}")

    def _policy_runtimes(self, root: SessionRuntime) -> tuple[SessionRuntime, ...]:
        """Include detached runtimes until their outstanding turns have settled."""
        candidates = [
            root,
            *self._children.values(),
            *(runtime for runtime, _ in self._pending_child_closes.values()),
        ]
        unique: list[SessionRuntime] = []
        seen: set[int] = set()
        for runtime in candidates:
            if id(runtime) not in seen:
                unique.append(runtime)
                seen.add(id(runtime))
        return tuple(unique)

    def _require_policy_unreserved(self) -> None:
        if self._policy_reserved:
            raise SessionExecutionConflict("Policy tree replacement is in progress")

    @contextmanager
    def reserve_config(self) -> Iterator[None]:
        """Hold idle admission for the retained tree through preparation and IO."""
        self._require_policy_unreserved()
        if self._root is None:
            raise SessionExecutionConflict("No registered root session")
        if self._ensure_child_lock.locked() or self._creating_children:
            raise SessionExecutionConflict("Child construction is in progress")
        runtimes = self._policy_runtimes(self._root)
        for runtime in runtimes:
            runtime.execution.require_idle()
            runtime.turns.require_policy_idle()
            runtime.agent_loop._require_policy_idle()
            if runtime._closed:
                raise SessionExecutionConflict("Session runtime is closed")
        with ExitStack() as stack:
            for runtime in runtimes:
                stack.enter_context(
                    runtime.execution.reserve(
                        SessionExecutionKind.LIFECYCLE, "configuration"
                    )
                )
            self._policy_reserved = True
            try:
                yield
            finally:
                self._policy_reserved = False

    @property
    def policy_revision(self) -> str:
        """Public revision binds accepted config to the current session lifetime."""
        if self._root is None:
            raise ValueError("Policy requires the registered root session")
        loop = self._root.agent_loop
        return (
            f"{loop.session_id}:{loop._session_generation}:"
            f"{loop.config_orchestrator.accepted_token}"
        )

    def root_source(self, session_id: str) -> SourceRestrictions:
        root = self._root
        if root is None or root.agent_loop.session_id != session_id:
            raise ValueError("Roots require the registered root session")
        orchestrator = root.agent_loop.config_orchestrator
        sources = [
            source
            for source in orchestrator.restrictions
            if source.kind == "source"
            and source.identity is not None
            and source.identity.owner == orchestrator.policy_owner
            and any(
                type(layer) is UserConfigLayer
                and layer.name == source.layer_name
                and layer.source_locator == source.locator
                for layer in orchestrator.layers
            )
        ]
        if len(sources) != 1:
            raise ValueError("Roots require one actual owned user source")
        return sources[0]

    def roots_revision(self, session_id: str) -> str:
        source = self.root_source(session_id)
        assert source.identity is not None
        return (
            f"{self.policy_revision}:{source.identity.owner}:{source.identity.source}"
        )

    async def replace_roots(
        self, *, session_id: str, roots: list[str], expected_revision: str
    ) -> str:
        source = self.root_source(session_id)
        if expected_revision != self.roots_revision(session_id):
            raise ValueError("Stale accepted roots revision")
        assert self._root is not None
        project = str(self._root.agent_loop.cwd)
        mapping = {
            str(entry.project): [str(path) for path in entry.roots]
            for entry in source.authorized_roots
        }
        mapping[project] = roots
        await self.replace_policy(
            session_id=session_id,
            source=source.layer_name,
            tools={},
            expected_revision=self.policy_revision,
            roots=mapping,
        )
        return self.roots_revision(session_id)

    async def save_root_grant(
        self, *, session_id: str, root: Path, expected_revision: str
    ) -> ConfigSaveResult:
        """Persist one approved root grant through the root orchestrator only.

        The calling session (root or child) is resolved through the registry,
        and the save is keyed by that registered session's cwd; a caller-supplied
        project path is never trusted. Child orchestrators never write: the save
        always targets the root orchestrator. Mid-turn safe: the orchestrator
        mutation lock is the only reservation, and nothing is published.
        """
        caller = self._resolve_session_loop(session_id)
        root_runtime = self._root
        if caller is None or root_runtime is None:
            raise ValueError("Root grant save requires a registered session and root")
        return (
            await root_runtime.agent_loop.config_orchestrator.save_project_root_grant(
                project=caller.cwd, root=root, expected_revision=expected_revision
            )
        )

    def _root_grant_save(
        self, session_id: str, root: Path, expected_revision: str
    ) -> Awaitable[ConfigSaveResult]:
        return self.save_root_grant(
            session_id=session_id, root=root, expected_revision=expected_revision
        )

    async def save_dangerous_command_grant(
        self, *, session_id: str, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        """Delegate the grant save to the dangerous-command authority collaborator."""
        return await self._dangerous_command_authority.save_dangerous_command_grant(
            session_id=session_id,
            operation_id=operation_id,
            expected_revision=expected_revision,
        )

    def _dangerous_command_grant_save(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> Awaitable[ConfigSaveResult]:
        # The loop's proposal project is not a trusted key: the save is keyed
        # by the registered caller session's cwd, exactly like the roots save.
        return self._dangerous_command_authority._dangerous_command_grant_save(
            session_id, project, operation_id, expected_revision
        )

    def resolve_live_session(self, session_id: str) -> SessionRuntime | None:
        """The registered root or an active child runtime, never a stale one."""
        root = self._root
        if (
            root is not None
            and not root._closed
            and root.agent_loop.session_id == session_id
        ):
            return root
        child = self._children.get(session_id)
        if child is not None and not child._closed:
            return child
        return None

    def _require_live_session(self, session_id: str) -> SessionRuntime:
        runtime = self.resolve_live_session(session_id)
        if runtime is None:
            raise ValueError(
                "Dangerous-command authority requires a live registered session"
            )
        return runtime

    async def read_dangerous_command_authority(
        self, session_id: str
    ) -> DangerousCommandAuthorityReadResponse:
        """Delegate the authority read to the authority collaborator."""
        return await self._dangerous_command_authority.read_dangerous_command_authority(
            session_id
        )

    async def revoke_dangerous_command_grant(
        self,
        *,
        session_id: str,
        operation_id: str,
        expected_live_generation: int,
        expected_disk_revision: str | None,
    ) -> DangerousCommandRevokeResponse:
        """Delegate the grant revocation to the authority collaborator."""
        return await self._dangerous_command_authority.revoke_dangerous_command_grant(
            session_id=session_id,
            operation_id=operation_id,
            expected_live_generation=expected_live_generation,
            expected_disk_revision=expected_disk_revision,
        )

    async def replace_policy(
        self,
        *,
        session_id: str,
        source: str,
        tools: dict[str, Any],
        expected_revision: str,
        roots: dict[str, list[str]] | None = None,
    ) -> str:
        """Reserve the actual retained tree before the first suspension.

        No source is written. Copies follow source identity; an equal independent
        child assertion is not a copy and remains in force.
        """
        self._require_policy_unreserved()
        root = self._root
        if root is None or root.agent_loop.session_id != session_id:
            raise ValueError("Policy replacement requires the registered root session")
        orchestrator = root.agent_loop.config_orchestrator
        token = orchestrator.accepted_token
        if self.policy_revision != expected_revision:
            raise ValueError("Stale accepted policy revision")
        previous = next(
            (r for r in orchestrator.restrictions if r.layer_name == source), None
        )
        if (
            previous is None
            or previous.kind != "source"
            or previous.identity is None
            or previous.identity.owner != orchestrator.policy_owner
        ):
            raise ValueError("Policy replacement requires an owned non-mode source")
        if self._ensure_child_lock.locked() or self._creating_children:
            raise SessionExecutionConflict("Child construction is in progress")
        runtimes = self._policy_runtimes(root)
        if len({id(r.agent_loop.config_orchestrator) for r in runtimes}) != len(
            runtimes
        ):
            raise ValueError("Retained sessions must own independent orchestrators")
        links = dict(self._child_links)
        tokens = {
            id(r): r.agent_loop.config_orchestrator.accepted_token for r in runtimes
        }
        for runtime in runtimes:
            runtime.execution.require_idle()
            runtime.turns.require_policy_idle()
            runtime.agent_loop._require_policy_idle()
            partition_policy_sources(
                runtime.agent_loop.config_orchestrator.restrictions
                + runtime.agent_loop.runtime_policy.inherited_restrictions
                + runtime.agent_loop.runtime_policy.inherited_mode_restrictions,
                owner=runtime.agent_loop.config_orchestrator.policy_owner,
            )
            if runtime._closed:
                raise SessionExecutionConflict("Session runtime is closed")
        with ExitStack() as stack:
            for runtime in runtimes:
                stack.enter_context(
                    runtime.execution.reserve(
                        SessionExecutionKind.LIFECYCLE, "policy-replacement"
                    )
                )
            self._policy_reserved = True
            try:
                prepared_root = (
                    await root.agent_loop._prepare_policy_replacement(
                        source=source, tools=tools, expected_token=token
                    )
                    if roots is None
                    else await root.agent_loop._prepare_root_replacement(
                        source=source, roots=roots, expected_token=token
                    )
                )
                replacement = next(
                    r
                    for r in prepared_root.orchestrator.restrictions
                    if r.layer_name == source
                )
                if roots is not None and tuple(
                    entry
                    for entry in replacement.authorized_roots
                    if entry.project != root.agent_loop.cwd
                ) != tuple(
                    entry
                    for entry in previous.authorized_roots
                    if entry.project != root.agent_loop.cwd
                ):
                    raise ValueError("Unrelated project root interpretation changed")
                by_runtime = await self._prepare_policy_descendants(
                    runtimes,
                    prepared_root,
                    previous,
                    replacement,
                    tokens,
                    replace_roots=roots is not None,
                )
                prepared = list(by_runtime.values())
                # Validate the whole closure before the first synchronous commit.
                if (
                    self._root is not root
                    or self._policy_runtimes(root) != runtimes
                    or (roots is not None and self._child_links != links)
                ):
                    raise SessionExecutionConflict(
                        "Policy tree changed during preparation"
                    )
                for runtime in runtimes:
                    candidate = by_runtime[id(runtime)]
                    runtime.turns.require_policy_idle()
                    if runtime._closed:
                        raise SessionExecutionConflict(
                            "Session runtime closed during preparation"
                        )
                    candidate.owner._validate_policy_replacement(candidate)
                for candidate in prepared:
                    candidate.owner._commit_policy_replacement(candidate, retire=False)
                for candidate in prepared:
                    candidate.previous_manager._retire_authority()
                return self.policy_revision
            finally:
                self._policy_reserved = False

    async def _prepare_policy_descendants(
        self,
        runtimes: tuple[SessionRuntime, ...],
        prepared_root: _PreparedPolicyReplacement,
        previous: SourceRestrictions,
        replacement: SourceRestrictions,
        tokens: dict[int, object],
        *,
        replace_roots: bool,
    ) -> dict[int, _PreparedPolicyReplacement]:
        """Recompute ancestor ceilings top down, regardless of insertion order."""
        by_runtime = {id(runtimes[0]): prepared_root}
        pending = list(runtimes[1:])
        while pending:
            ready = [
                runtime
                for runtime in pending
                if not replace_roots
                or (
                    runtime.agent_loop.session_id in self._child_links
                    and id(self._child_links[runtime.agent_loop.session_id][0])
                    in by_runtime
                )
            ]
            if not ready:
                raise ValueError("Retained policy tree parent unavailable")
            for runtime in ready:
                parent = (
                    self._child_links[runtime.agent_loop.session_id][0]
                    if replace_roots
                    else runtimes[0]
                )
                by_runtime[
                    id(runtime)
                ] = await runtime.agent_loop._prepare_policy_refresh(
                    previous=previous,
                    replacement=replacement,
                    expected_token=tokens[id(runtime)],
                    inherited_workspace=(
                        by_runtime[id(parent)].tool_manager.workspace
                        if replace_roots
                        else None
                    ),
                )
                pending.remove(runtime)
        return by_runtime

    def bind_root(self, runtime: SessionRuntime) -> None:
        self._require_policy_unreserved()
        if self._root is not None:
            raise RuntimeError("A root session runtime is already registered")
        self._root = runtime
        self.begin_root_generation()
        runtime.agent_loop.bind_root_grant_port(
            SessionRootGrantPort(self._resolve_session_loop, self._root_grant_save)
        )
        runtime.agent_loop.bind_dangerous_command_grant_port(
            SessionDangerousCommandGrantPort(
                self._resolve_session_loop, self._dangerous_command_grant_save
            )
        )

    def _resolve_session_loop(self, session_id: str) -> AgentLoop | None:
        root = self._root
        if root is not None and root.agent_loop.session_id == session_id:
            return root.agent_loop
        child = self._children.get(session_id) or self._readable_children.get(
            session_id
        )
        return child.agent_loop if child is not None else None

    def begin_root_generation(self) -> None:
        root = self._root
        if root is None:
            self._retention_policy = RetentionPolicy(
                idle_ttl_seconds=0, max_idle_agents=0, retain_agents=False
            )
            self._generation_identity = None
            return
        identity = (root.agent_loop.session_id, root.agent_loop._session_generation)
        if (
            self._generation_identity is not None
            and self._generation_identity != identity
        ):
            self._retire_generation_agents()
        config = root.agent_loop.config.subagents
        self._restore_agent_id_high_water_mark(root)
        self._retention_policy = RetentionPolicy(
            idle_ttl_seconds=config.idle_ttl_seconds,
            max_idle_agents=config.max_idle_agents,
            retain_agents=config.retain_agents,
            tombstone_ttl_seconds=config.tombstone_ttl_seconds,
            max_tombstones=config.max_tombstones,
        )
        self._generation_identity = identity

    def _acquire_notification_suppression(self, agent_id: str) -> None:
        self._suppression_owners[agent_id] = (
            self._suppression_owners.get(agent_id, 0) + 1
        )
        self._suppressed_notifications.add(agent_id)

    def _release_notification_suppression(self, agent_id: str) -> None:
        owners = self._suppression_owners.get(agent_id, 0)
        if owners > 1:
            self._suppression_owners[agent_id] = owners - 1
        else:
            self._suppression_owners.pop(agent_id, None)
            self._suppressed_notifications.discard(agent_id)

    def _retire_generation_agents(self) -> None:
        # WP4: a retired generation forfeits pending automatic recovery along
        # with its records — the controller's timers are cancelled here and
        # their re-validation (generation fence) is the backstop.
        self._recovery_controller.cancel_all()
        records = list(self._agent_records.values())
        # Drop terminal metadata belonging to the retiring generation. The
        # matching is on the full (parent session, generation) identity captured
        # at admission, which runs before `begin_root_generation` updates the
        # identity. Results, expiry markers, latest-run indexes, and wait leases
        # are deliberately preserved.
        old_identity = self._generation_identity
        removed_tombstones = [
            agent_id
            for agent_id, tombstone in self._evicted_agents.items()
            if old_identity is not None and tombstone.parent_identity == old_identity
        ]
        for agent_id in removed_tombstones:
            self._evicted_agents.pop(agent_id, None)
            self._pending_evictions.pop(agent_id, None)
        for record in records:
            self._acquire_notification_suppression(record.agent_id)
            # WP3: a retired generation forfeits its suspended records'
            # storage-seam state along with the records themselves.
            self._suspension.discard_locked(record.agent_id)
            self._agent_records.pop(record.agent_id, None)
            self._children.pop(record.session_id, None)
            self._child_links.pop(record.session_id, None)
            monitors = {
                run.completion_task
                for run in [*record.run_history, record.current_run]
                if run is not None
                and isinstance(run.completion_task, asyncio.Task)
                and not run.completion_task.done()
            }
            for monitor in monitors:
                monitor.cancel()

            async def teardown_retired(
                retired: AgentRecord = record,
                pending: set[asyncio.Task[Any]] = monitors,
            ) -> None:
                try:
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    await retired.runtime.close()
                finally:
                    self._release_notification_suppression(retired.agent_id)

            self._track_child_close(
                record.runtime,
                teardown_retired(),
                name=f"vibe-subagent-generation-drain:{record.agent_id}",
            )
        self._rearm_reaper_locked()
        if removed_tombstones or records:
            # Publish the new inventory whenever anything was removed —
            # terminal metadata or resident records. Retiring a generation
            # that contains only resident agents (e.g. a retained idle
            # advisor) removes records with no agents/update, and the browser
            # would otherwise keep ghost rows until the next unrelated
            # update.
            publish = asyncio.create_task(
                self._emit_agents_update(),
                name="vibe-subagent-generation-agents-update",
            )
            self._teardown_tasks.add(publish)
            publish.add_done_callback(self._teardown_tasks.discard)

    def _generation_is_current(self, generation: int) -> bool:
        root = self._root
        if self._generation_identity is None:
            return True
        return (
            root is not None
            and self._generation_identity
            == (root.agent_loop.session_id, root.agent_loop._session_generation)
            and generation == root.agent_loop._session_generation
        )

    def release_root(self, runtime: SessionRuntime) -> None:
        self._require_policy_unreserved()
        if self._root is runtime:
            self._root = None
            self._generation_identity = None

    def resolve_child(
        self, session_id: str
    ) -> SessionRuntime | StoredChildSession | None:
        """Delegate to the transcript reads collaborator."""
        return self._transcript_reads.resolve_child(session_id)

    def stored_child_is_current(self, stored: StoredChildSession) -> bool:
        """Delegate to the transcript reads collaborator."""
        return self._transcript_reads.stored_child_is_current(stored)

    @staticmethod
    def _transcript_link_identity(
        link: ChildSessionLink | None,
    ) -> tuple[str, str, str, str | None] | None:
        return TranscriptReads._transcript_link_identity(link)

    @staticmethod
    def _transcript_link_for(
        root: SessionRuntime, child_session_id: str
    ) -> ChildSessionLink | None:
        return TranscriptReads._transcript_link_for(root, child_session_id)

    async def resolve_resident_transcript_read(
        self, agent_id: str
    ) -> ResidentTranscriptReadSnapshot | None:
        """Delegate to the transcript reads collaborator."""
        return await self._transcript_reads.resolve_resident_transcript_read(agent_id)

    async def resident_transcript_read_is_current(
        self, snapshot: ResidentTranscriptReadSnapshot
    ) -> bool:
        """Delegate to the transcript reads collaborator."""
        return await self._transcript_reads.resident_transcript_read_is_current(
            snapshot
        )

    async def resolve_transcript_read(self, agent_id: str) -> TranscriptReadSnapshot:
        """Delegate to the transcript reads collaborator."""
        return await self._transcript_reads.resolve_transcript_read(agent_id)

    async def transcript_read_is_current(
        self, snapshot: TranscriptReadSnapshot
    ) -> bool:
        """Delegate to the transcript reads collaborator."""
        return await self._transcript_reads.transcript_read_is_current(snapshot)

    def references_child(self, session_id: str) -> bool:
        """Delegate to the transcript reads collaborator."""
        return self._transcript_reads.references_child(session_id)

    def _track_child_close(
        self,
        runtime: SessionRuntime,
        cleanup: Coroutine[Any, Any, None] | None = None,
        *,
        name: str,
    ) -> asyncio.Task[None]:
        session_id = runtime.agent_loop.session_id
        task = asyncio.create_task(cleanup or runtime.close(), name=name)
        self._teardown_tasks.add(task)
        self._pending_child_closes[session_id] = (runtime, task)

        def completed(done: asyncio.Task[None]) -> None:
            if done.cancelled() or done.exception() is not None:
                return
            self._teardown_tasks.discard(done)
            pending = self._pending_child_closes.get(session_id)
            if pending == (runtime, done):
                self._pending_child_closes.pop(session_id, None)

        task.add_done_callback(completed)
        return task

    @asynccontextmanager
    async def _child_publication(self, runtime: SessionRuntime) -> AsyncIterator[None]:
        acquired = False
        try:
            async with self._ensure_child_lock:
                acquired = True
                yield
        except BaseException:
            if not acquired:
                self._track_child_close(
                    runtime,
                    name=(
                        "vibe-subagent-unpublished-teardown:"
                        f"{runtime.agent_loop.session_id}"
                    ),
                )
            raise

    @staticmethod
    def _retire_teardown_task(task: asyncio.Task[None]) -> None:
        if task.done() and not task.cancelled():
            failure = task.exception()
            if failure is not None:
                failure.__traceback__ = None

    async def _join_pending_child_close(self, session_id: str) -> None:
        pending = self._pending_child_closes.get(session_id)
        if pending is None:
            return
        runtime, task = pending
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except BaseException:
            # Failed cleanup retains ownership; retry it before rematerializing.
            await runtime.close()
        if runtime._closed:
            self._teardown_tasks.discard(task)
            if self._pending_child_closes.get(session_id) == (runtime, task):
                self._pending_child_closes.pop(session_id, None)

    async def ensure_child(self, session_id: str) -> bool:
        self._require_policy_unreserved()
        if session_id in self._children:
            return True
        async with self._ensure_child_lock:
            self._require_policy_unreserved()
            if session_id in self._children:
                return True
            await self._join_pending_child_close(session_id)
            stored = self._stored_children.get(session_id)
            if stored is not None and not self.stored_child_is_current(stored):
                self._stored_children.pop(session_id, None)
                self._readable_children.pop(session_id, None)
                self._detached_child_ids.discard(session_id)
                stored = None
            resolved = (
                (
                    stored.parent_runtime,
                    stored.parent_runtime.agent_loop,
                    stored.child_dir,
                    stored.link,
                )
                if stored is not None
                else self._resolve_child_link(session_id)
            )
            if resolved is None:
                return False
            parent_runtime, root_agent_loop, child_dir, link = resolved
            child: AgentLoop | None = None
            try:
                child = await self._runtime_factory.resume_child(
                    root_agent_loop, link.agent, link.session_id, child_dir
                )
                runtime = self._build_child_runtime(
                    child, base_history=project_history(child)
                )
            except Exception as exc:
                logger.warning(
                    "Failed to ensure child session session_id=%s path=%s",
                    session_id,
                    child_dir,
                    exc_info=exc,
                )
                if child is not None:
                    await self._discard_child(child)
                if isinstance(exc, LaunchConfigError):
                    raise
                return False
            self._children[child.session_id] = runtime
            self._stored_children.pop(child.session_id, None)
            self._detached_child_ids.discard(child.session_id)
            self._child_links[child.session_id] = (parent_runtime, link.tool_call_id)
        return True

    def _resolve_child_link(
        self, session_id: str
    ) -> tuple[SessionRuntime, AgentLoop, Path, ChildSessionLink] | None:
        """Delegate to the transcript reads collaborator."""
        return self._transcript_reads._resolve_child_link(session_id)

    def child_belongs_to(self, session_id: str, root_session_id: str) -> bool:
        child = self._children.get(session_id)
        return (
            child is not None and child.agent_loop.parent_session_id == root_session_id
        )

    def handoff_root(self, old_session_id: str, new_session_id: str) -> None:
        self._require_policy_unreserved()
        for runtime in self._children.values():
            if runtime.agent_loop.parent_session_id == old_session_id:
                runtime.agent_loop.parent_session_id = new_session_id
        self.begin_root_generation()

    async def handoff_active_turn(
        self,
        old_session_id: str,
        *,
        current_history: list[PublicHistoryEntry],
        callbacks: list[PublicCallbackEntry],
        active_turn: PublicTurn | None,
        completed_turns: list[PublicTurn],
        turn_queue: PublicTurnQueue,
        history_limit: int = 200,
    ) -> SessionHandoff:
        self._require_policy_unreserved()
        runtime = self._require_child(old_session_id)
        new_session_id = runtime.agent_loop.session_id
        if old_session_id == new_session_id:
            raise RuntimeError("Session handoff did not change the session ID")
        if new_session_id in self._children:
            raise RuntimeError(f"Child session is already registered: {new_session_id}")
        parent, tool_call_id = self._child_links[old_session_id]
        if active_turn is None:
            # An idle clear creates a fresh core identity without its parent.
            # The registered runtime remains this parent's child.
            runtime.agent_loop.parent_session_id = parent.agent_loop.session_id
            metadata = runtime.agent_loop.session_logger.session_metadata
            if metadata is not None:
                metadata.parent_session_id = parent.agent_loop.session_id

        await parent.agent_loop.replace_child_session(
            old_session_id, runtime.agent_loop, tool_call_id
        )
        self._children.pop(old_session_id)
        self._children[new_session_id] = runtime
        self._child_links.pop(old_session_id)
        self._child_links[new_session_id] = (parent, tool_call_id)
        for record in self._agent_records.values():
            if record.session_id == old_session_id:
                record.session_id = new_session_id
        runtime.history.replace(rebind_history(runtime.history.base, new_session_id))
        for child in self._children.values():
            if child.agent_loop.parent_session_id == old_session_id:
                child.agent_loop.parent_session_id = new_session_id
        if parent.turns._projector is not None:
            try:
                await parent.turns.replace_subagent(
                    tool_call_id, old_session_id, new_session_id
                )
            except Exception:
                # Historical effects can belong to an inactive parent turn; their
                # stale session ID is display-only because metadata was updated above.
                logger.debug(
                    "Skipping subagent projection update during handoff: "
                    "tool_call_id %s not in active turn",
                    tool_call_id,
                )
        state = build_public_state(
            runtime.agent_loop,
            history=runtime.history.base,
            current_history=current_history,
            callbacks=callbacks,
            turns=[
                *completed_turns,
                *([active_turn] if active_turn is not None else []),
            ],
            retrying=None,
            history_limit=history_limit,
        )
        state = state.model_copy(
            update={
                "event_id": self._event_watermark(new_session_id),
                "turn_queue": turn_queue,
            }
        )
        return SessionHandoff(
            old_session_id=old_session_id,
            new_session_id=new_session_id,
            state=state,
            session_log=project_session_log(runtime.agent_loop),
        )

    def public_state(
        self,
        session_id: str,
        history_limit: int,
        *,
        turns_limit: int | None = None,
        include_history: bool = True,
        include_turns: bool = True,
    ) -> PublicSessionState:
        return self._public_state(
            self._require_child(session_id),
            history_limit,
            turns_limit=turns_limit,
            include_history=include_history,
            include_turns=include_turns,
        )

    def _public_state(
        self,
        runtime: SessionRuntime,
        history_limit: int,
        *,
        turns_limit: int | None = None,
        include_history: bool = True,
        include_turns: bool = True,
    ) -> PublicSessionState:
        callbacks = [
            entry
            for entry in runtime.history.all(runtime.turns.history)
            if isinstance(entry, PublicCallbackEntry)
        ]
        state = build_public_state(
            runtime.agent_loop,
            history=runtime.history.base,
            current_history=runtime.turns.history,
            callbacks=callbacks,
            turns=runtime.turns.turns,
            retrying=runtime.turns.retrying,
            history_limit=history_limit,
            turns_limit=turns_limit,
            include_history=include_history,
            include_turns=include_turns,
        )
        return state.model_copy(
            update={
                "event_id": self._event_watermark(runtime.agent_loop.session_id),
                "turn_queue": runtime.turns.queue_state,
            }
        )

    def history(self, session_id: str) -> list[PublicHistoryEntry]:
        runtime = self._require_child(session_id)
        return runtime.history.all(runtime.turns.history)

    def turns(self, session_id: str) -> list[PublicTurn]:
        return self._require_child(session_id).turns.turns

    def active_callbacks(self) -> list[PublicCallbackEntry]:
        return [
            callback
            for runtime in self._children.values()
            for callback in runtime.turns.callbacks
            if isinstance(callback.state, OpenCallbackState)
        ]

    async def answer_callback(
        self, session_id: str, callback_id: str, output: CallbackOutput
    ) -> str:
        return await self._require_child(session_id).turns.answer_callback(
            callback_id, output
        )

    async def reject_callback(
        self, session_id: str, callback_id: str, error: CallbackResultError
    ) -> str:
        return await self._require_child(session_id).turns.reject_callback(
            callback_id, error
        )

    async def close(self) -> None:
        await self.close_children()

    @contextmanager
    def _child_creation(self) -> Iterator[None]:
        """Delegate to the child launch admission collaborator."""
        with self._launch_admission.creation_fence():
            yield

    def _restore_agent_id_high_water_mark(self, root: SessionRuntime) -> None:
        """Delegate to the child launch admission collaborator."""
        self._launch_admission.restore_sequence(root)

    async def _issue_agent_id(self, parent: SessionRuntime) -> str:
        """Delegate to the child launch admission collaborator."""
        return await self._launch_admission.issue_agent_id(parent)

    @staticmethod
    def _new_run_id(agent_id: str) -> str:
        return f"{agent_id}-run-{uuid4().hex[:12]}"

    def _validate_background_admission(
        self, parent: SessionRuntime, identity: tuple[str, int]
    ) -> None:
        """Delegate to the child launch admission collaborator."""
        self._launch_admission.validate_admission(parent, identity)

    def _resolve_launch_candidate(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        *,
        runtime: SessionRuntime | None = None,
        identity: str | None = None,
    ) -> LaunchCandidate:
        return self._launch_admission.resolve_candidate(
            parent, args, runtime=runtime, identity=identity
        )

    async def _create_registered_child(
        self, parent: SessionRuntime, args: TaskArgs, ctx: InvokeContext
    ) -> tuple[SessionRuntime, str]:
        """Delegate to the child launch admission collaborator."""
        return await self._launch_admission.create_registered(parent, args, ctx)

    @staticmethod
    def _start_child_turn(
        runtime: SessionRuntime,
        args: TaskArgs,
        ctx: InvokeContext,
        session_id: str,
        *,
        completion_contract: FrozenCompletionContract | None = None,
    ) -> tuple[str, TurnStartAction]:
        text = prepare_subagent_prompt(args.task, ctx)
        if completion_contract is not None:
            # P4: the frozen contract renders into the child prompt — a
            # worker cannot reference requirements it never received.
            text = f"{text}\n\n{completion_contract.render_prompt_section()}"
        response, start = runtime.turns.start(
            TurnStartParams(
                session_id=session_id, message=[TextContentBlock(text=text)]
            )
        )
        return response.turn.id, start

    async def _rollback_created_child(
        self, parent: SessionRuntime, runtime: SessionRuntime, tool_call_id: str
    ) -> None:
        """Delegate to the child launch admission collaborator."""
        await self._launch_admission.rollback_created(parent, runtime, tool_call_id)

    async def _rollback_reused_claim(
        self,
        record: AgentRecord,
        previous_idle_since: float | None,
        *,
        apply_policy: bool = False,
    ) -> None:
        """Delegate to the replacement coordinator collaborator."""
        await self._replacement.rollback_reuse(
            record, previous_idle_since, apply_policy=apply_policy
        )

    def _require_task_identity_allowed(
        self, parent: SessionRuntime, identity: str
    ) -> None:
        """Apply effective Task admission before constructing child resources."""
        self._launch_admission.require_identity_allowed(parent, identity)

    def _require_retained_identity_match(
        self, record: AgentRecord, args: TaskArgs
    ) -> None:
        """Compare the requested identity against the record's on BOTH sides of
        a retained continuation (11c): a canonical identity selection —
        structured or shorthand — that names a different identity than the
        retained instance raises the identity-mismatch error; a plain
        continuation naming no target reuses the instance as-is and keeps
        the record's identity revision and overlay immutable.
        """
        requested_identity = requested_identity_reference(args)
        retained_identity = record.identity
        if requested_identity is not None and requested_identity != retained_identity:
            raise AgentIdentityMismatchError(
                f"Requested identity {requested_identity} does not "
                f"match retained agent identity {retained_identity}"
            )

    @staticmethod
    def _replacement_refusal(record: AgentRecord, outcome: LaunchOutcome) -> TaskResult:
        """Delegate to the replacement coordinator collaborator."""
        return ReplacementCoordinator.replacement_refusal(record, outcome)

    @staticmethod
    def _tombstone_reuse_error(
        agent_id: str, tombstone: _AgentTombstone
    ) -> AgentClosedError:
        """Delegate to the replacement coordinator collaborator."""
        return ReplacementCoordinator.tombstone_reuse_error(agent_id, tombstone)

    async def _reserve_replacement(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> _ReplacementClaim | TaskResult | None:
        """Delegate to the replacement coordinator collaborator."""
        return await self._replacement.reserve(args, ctx)

    def _validate_replacement_locked(self, claim: _ReplacementClaim) -> None:
        """Delegate to the replacement coordinator collaborator."""
        self._replacement.validate_locked(claim)

    def _clear_replacement_locked(self, record: AgentRecord) -> None:
        """Delegate to the replacement coordinator collaborator."""
        self._replacement.clear_locked(record)

    async def _finish_replacement_unwind(
        self, record: AgentRecord, run: RunRecord
    ) -> None:
        """Delegate to the replacement coordinator collaborator."""
        await self._replacement.finish_unwind(record, run)

    @asynccontextmanager
    async def _replacement_handoff(
        self, claim: _ReplacementClaim
    ) -> AsyncIterator[None]:
        """Delegate to the replacement coordinator collaborator."""
        async with self._replacement.handoff(claim):
            yield

    async def run(
        self,
        args: TaskArgs,
        ctx: InvokeContext,
        *,
        defer_launch_agents_update: bool = False,
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        if args.replace_run:
            claim = await self._reserve_replacement(args, ctx)
            if isinstance(claim, TaskResult):
                yield claim
                return
            if claim is not None:
                async with self._replacement_handoff(claim):
                    replacement_args = args.model_copy(
                        update={
                            "task": "This task supersedes the interrupted task.\n\n"
                            + args.task
                        }
                    )
                    launch = self._run_admitted(
                        replacement_args,
                        ctx,
                        claim.slot,
                        defer_launch_agents_update=defer_launch_agents_update,
                        replacement=claim,
                        summary_task=args.task,
                    )
                    try:
                        async for event in launch:
                            yield event
                    finally:
                        await launch.aclose()
                return
        if args.agent_id is not None and args.background:
            record = getattr(self, "_agent_records", {}).get(args.agent_id)
            if record is not None and (
                record.reserved_for_replacement or record.state is not _AgentState.IDLE
            ):
                raise AgentBusyError("Agent is already running a task")
            if record is not None:
                self._require_retained_identity_match(record, args)
        # Check and reserve without awaiting: concurrent creation/reuse attempts
        # consume capacity before they can allocate runtimes or call providers.
        if not hasattr(self, "_active_work_slots"):
            self._active_work_slots = set()
        running = len(self._active_work_slots)
        cap = getattr(self, "_max_running_subagents", None)
        if cap is None:
            root = getattr(self, "_root", None)
            cap = (
                root.agent_loop.config.subagents.max_running_subagents
                if root is not None
                else _DEFAULT_MAX_RUNNING_SUBAGENTS
            )
        if running >= cap:
            raise RuntimeError(
                f"Subagent launch rejected: {running} agents already running "
                f"(cap {cap}); "
                "wait for completion or release agents before launching more"
            )
        slot = _ActiveWorkSlot()
        self._active_work_slots.add(slot)
        launch = SessionRuntimeRegistry._run_admitted(
            self, args, ctx, slot, defer_launch_agents_update=defer_launch_agents_update
        )
        try:
            async for event in launch:
                yield event
        finally:
            try:
                await launch.aclose()
            finally:
                if not slot.transferred:
                    self._active_work_slots.discard(slot)

    async def _run_admitted(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        args: TaskArgs,
        ctx: InvokeContext,
        slot: _ActiveWorkSlot,
        *,
        defer_launch_agents_update: bool = False,
        replacement: _ReplacementClaim | None = None,
        summary_task: str | None = None,
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        if ctx.is_subagent:
            raise RuntimeError("Agent depth limit of 1 reached")
        parent = self._runtime(ctx.session_id)
        parent_identity = (
            parent.agent_loop.session_id,
            parent.agent_loop._session_generation,
        )
        if args.background:
            # WP5: a replacement claim (config.predecessor_agent_id on a
            # fresh-agent launch) is validated at failure time by the launch
            # admission BEFORE preparation — the old recovery is disabled and
            # the per-package budget consumed before the new run's write
            # token is registered at acceptance.
            replacement_decision = None
            if (
                args.agent_id is None
                and args.config is not None
                and args.config.predecessor_agent_id is not None
            ):
                replacement_decision = (
                    await self._launch_admission.validate_replacement_claim(
                        parent, args
                    )
                )
            # Background execution context (ASC area 3, step 3): the
            # preparation region moved to the executor's prepare_background;
            # the returned context carries the preparation + acceptance-tail
            # state (references, not copies).
            context = await self._execution.prepare_background(
                parent,
                parent_identity,
                args,
                ctx,
                slot,
                replacement=replacement,
                summary_task=summary_task,
                replacement_decision=replacement_decision,
            )

            try:
                context.runtime.turns._event_sink = context.consume_event
                turn_id, action = SessionRuntimeRegistry._start_child_turn(
                    context.runtime,
                    args,
                    ctx,
                    context.runtime.agent_loop.session_id,
                    completion_contract=context.run_record.completion_contract,
                )
                context.turn_id = turn_id
                context.action = action
                context.run_record.initial_turn_id = turn_id

                await self._execution.accept_background(
                    context, parent, parent_identity, slot
                )
            except BaseException as exc:
                # The rollback region moved to the executor's
                # rollback_background; the re-raise tail stays here.
                await self._execution.rollback_background(
                    context, parent, replacement=replacement
                )
                if isinstance(exc, asyncio.CancelledError):
                    raise exc
                raise
            yield await self._execution.acknowledge_background(
                context,
                replacement=replacement,
                defer_launch_agents_update=defer_launch_agents_update,
            )
            return

        if args.agent_id is not None:
            raise ValueError("agent_id is only supported for background tasks")
        if args.retain is not None:
            raise ValueError("retain is only supported for background tasks")
        requested_identity = requested_identity_reference(args)
        if requested_identity is not None:
            self._require_task_identity_allowed(parent, requested_identity)
        runtime, _launch_identity = await self._create_registered_child(
            parent, args, ctx
        )
        child = runtime.agent_loop
        # C6b: snapshot the bounded un-stick diagnostics for an explicit
        # override_cooldown launch pre-start (create_registered minted and
        # bound the permit; the child's first model call consumes the
        # binding), so the foreground result carries them exactly like the
        # background acknowledgment (C4). Gated on the explicit request:
        # ordinary and automatic launches never carry a binding (F6), and
        # the foreground surface is also driven through spec'd mock
        # registries whose loops carry no coordinator slot.
        override_diagnostics = (
            override_launch_diagnostics(child) if args.override_cooldown else None
        )
        # P4: freeze the completion contract at foreground admission and
        # capture the admission HEAD baseline alongside it (PRE-EXECUTION,
        # only when the contract mandates a commit artifact). The identity
        # binding is the canonical source; the launch-resolved reference is
        # the fallback for a child constructed without one.
        binding = child._identity_binding()
        if isinstance(binding, tuple) and binding:
            bound_identity, bound_identity_revision = binding[0], binding[1]
        else:
            bound_identity = (
                _launch_identity or requested_identity_reference(args) or ""
            )
            bound_identity_revision = None
        completion_contract, admission_head = freeze_run_admission(
            args=args,
            identity=bound_identity,
            identity_revision=bound_identity_revision,
            tier=child.launch_package_tier,
            runtime=runtime,
            parent=parent,
        )
        # Foreground execution context (ASC area 3, step 2): the
        # per-invocation state of one foreground run. The event consumer and
        # the slot discard callback are context methods, the teardown is the
        # executor's teardown_foreground(context); the orchestration below
        # operates on the context.
        context = ForegroundExecutionContext(
            runtime=runtime,
            child=child,
            progress=BoundedEventQueue[ToolStreamEvent](),
            accumulator=SubagentRunAccumulator(),
            completion_metadata_mark=child.completion_metadata_mark(),
            tool_call_id=ctx.tool_call_id,
            previous_event_sink=runtime.turns._event_sink,
            slot=slot,
            completion_contract=completion_contract,
            admission_head=admission_head,
        )
        # P2/P3 wiring: install the run's receipt sink on the child loop —
        # the post-tool finalization seam emits each invocation's receipt
        # into THIS run's accumulator chronology. Cleared at run teardown.
        completion_evidence_sink = context.accumulator.completion_evidence_sink()
        child._completion_evidence_sink = completion_evidence_sink
        context.completion_evidence_sink = completion_evidence_sink
        runtime.turns._event_sink = context.consume_event
        try:
            turn_id, start = SessionRuntimeRegistry._start_child_turn(
                runtime,
                args,
                ctx,
                child.session_id,
                completion_contract=completion_contract,
            )
            start()
        except BaseException:
            runtime.turns._event_sink = context.previous_event_sink
            child._completion_evidence_sink = None
            await self._rollback_created_child(parent, runtime, ctx.tool_call_id)
            raise
        context.turn_id = turn_id
        completion = asyncio.create_task(
            runtime.turns.wait_for_operation(turn_id),
            name=f"vibe-subagent-turn:{child.session_id}",
        )
        context.completion = completion
        slot.transferred = True
        try:
            async for item in stream_until_complete(
                context.progress, completion, event_task_name="vibe-subagent-progress"
            ):
                yield item
            turn = await completion
            if turn.error is not None:
                context.accumulator.record_error(turn.error.message)
            turns_used = sum(
                message.role is Role.assistant for message in child.messages
            )
            # P6b: the once-only finalization is awaited immediately BEFORE
            # the result build, drain-wrapped like the monitor's sites; a run
            # cancelled before the audit starts (the turn settled INTERRUPTED)
            # takes the cancellation disposition instead — the stub
            # finalization is cached and no check spawns.
            stop_reason: RunStopReason | None = None
            if turn.status is PublicTurnStatus.INTERRUPTED:
                stop_reason = (
                    self._run_stop_reason(turn) or RunStopReason.USER_CANCELLED
                )
                apply_cancellation_disposition(
                    context.accumulator,
                    turn_id,
                    completion_contract,
                    admission_head,
                    stop_reason,
                )
            await finalize_run_with_drain(
                run_id=turn_id,
                accumulator=context.accumulator,
                contract=completion_contract,
                admission_head=admission_head,
                runtime=runtime,
            )
            result_update: dict[str, Any] = {
                "metadata": {
                    **child.completion_metadata_since(context.completion_metadata_mark),
                    # C6b: the bounded un-stick diagnostics ride the
                    # foreground result's metadata exactly like the background
                    # acknowledgment (C4).
                    **(
                        {"cooldown_override": override_diagnostics}
                        if override_diagnostics is not None
                        else {}
                    ),
                }
            }
            if stop_reason is not None:
                result_update["stop_reason"] = stop_reason
            # P12: the call site wires the composition surface — the run
            # key (the once-only finalization cache read), the frozen
            # contract (the mandate surface and the audit mode), the
            # worker's receipt claim (operative only on a healthy
            # completed turn), and the acceptance-check runner.
            context.result = context.accumulator.build_result(
                turns_used=turns_used,
                completed=turn.status is PublicTurnStatus.COMPLETED,
                run_id=turn_id,
                contract=completion_contract,
                claim=operative_completion_claim(
                    context.accumulator,
                    turn_completed=turn.status is PublicTurnStatus.COMPLETED,
                ),
                runner=_check_runner(runtime),
            ).model_copy(update=result_update)
            if turn.error is not None:
                raise RuntimeError(turn.error.message)
        except asyncio.CancelledError:
            if context.result is None:
                # P6b case (b): cancellation mid-run — the disposition runs
                # without awaits (no further scheduling before the stub is
                # cached), the terminal result builds exactly once, and the
                # cancellation propagates; the drain wrapper already killed
                # this run's in-flight check processes.
                terminal = runtime.turns.operation_terminal_turn(turn_id)
                terminal_reason = (
                    self._run_stop_reason(terminal) if terminal is not None else None
                )
                apply_cancellation_disposition(
                    context.accumulator,
                    turn_id,
                    completion_contract,
                    admission_head,
                    terminal_reason,
                )
                # P12: the cancelled run wires the mandate surface and
                # the finalization cache read; the worker's claim is NOT
                # operative on a cancelled run, so no claim rides — the
                # terminal turn's reason stays preserved on the result.
                context.result = context.accumulator.build_result(
                    turns_used=sum(
                        message.role is Role.assistant for message in child.messages
                    ),
                    completed=False,
                    run_id=turn_id,
                    contract=completion_contract,
                    runner=_check_runner(runtime),
                ).model_copy(
                    update={
                        "metadata": {
                            **child.completion_metadata_since(
                                context.completion_metadata_mark
                            ),
                            # C6b: the cancelled foreground result carries the
                            # bounded un-stick diagnostics like the settled one.
                            **(
                                {"cooldown_override": override_diagnostics}
                                if override_diagnostics is not None
                                else {}
                            ),
                        },
                        "stop_reason": terminal_reason
                        if terminal_reason is not None
                        else RunStopReason.USER_CANCELLED,
                    }
                )
            raise
        finally:
            # P6b: the capacity slot stays accounted while the finalization
            # verification runs and is released only here — the former
            # completion done-callback released it before the checks ran.
            context.discard_work_slot(self, completion)
            runtime.turns._event_sink = context.previous_event_sink
            # P2/P3 run-end hygiene: the run's receipt sink is spent.
            if child._completion_evidence_sink is context.completion_evidence_sink:
                child._completion_evidence_sink = None
            if self._children.get(child.session_id) is runtime:
                parent_runtime, tool_call_id = self._child_links[child.session_id]
                parent_loop = parent_runtime.agent_loop
                parent_dir = parent_loop.session_logger.session_dir
                child_dir = child.session_logger.session_dir
                if parent_dir is not None and child_dir is not None:
                    link = ChildSessionLink(
                        session_id=child.session_id,
                        tool_call_id=tool_call_id,
                        agent=(
                            child.launch_identity
                            or requested_identity_reference(args)
                            or ""
                        ),
                        relative_path=str(child_dir.relative_to(parent_dir)),
                    )
                    self._stored_children[child.session_id] = StoredChildSession(
                        parent_runtime=parent_runtime,
                        root_session_id=parent_loop.session_id,
                        root_generation=parent_loop._session_generation,
                        child_dir=child_dir,
                        link=link,
                    )
                else:
                    # Logging-disabled children have no on-disk transcript. Retain the
                    # closed runtime as a read-only snapshot after removing it from
                    # reuse and policy ownership.
                    self._readable_children[child.session_id] = runtime
                self._children.pop(child.session_id, None)
                self._child_links.pop(child.session_id, None)
                self._detached_child_ids.add(child.session_id)
            self._track_child_close(
                runtime,
                SubagentExecution.teardown_foreground(context),
                name=f"vibe-subagent-foreground-teardown:{child.session_id}",
            )

        assert context.result is not None
        yield context.result

    def _expire_results_locked(self) -> None:
        """Delegate to the run results collaborator."""
        self._run_results.expire_locked()

    def _register_writer_locked(
        self, agent_id: str, run_id: str, token: object
    ) -> None:
        """Delegate to the run results collaborator."""
        self._run_results.register_writer_locked(agent_id, run_id, token)

    def _publish_locked(
        self, context: BackgroundExecutionContext
    ) -> PublicationDecision:
        """Delegate to the run results collaborator."""
        return self._run_results.publish_locked(context)

    def _purge_agent_locked(self, agent_id: str) -> set[tuple[str, str]]:
        """Delegate to the run results collaborator."""
        return self._run_results.purge_agent_locked(agent_id)

    def _agent_turns_used(self, record: AgentRecord) -> int | None:
        run = record.current_run
        if run is not None:
            return max(
                0,
                sum(
                    message.role is Role.assistant
                    for message in record.runtime.agent_loop.messages
                )
                - run.start_count,
            )
        if record.run_history and record.run_history[-1].result is not None:
            return record.run_history[-1].result.turns_used
        return None

    def _run_elapsed_seconds(self, run: RunRecord | None, now: float) -> float | None:
        if run is None or run.started_at is None:
            return None
        end = run.completed_at if run.completed_at is not None else now
        return max(0.0, end - run.started_at)

    @staticmethod
    def _record_launch_key(record: AgentRecord) -> float:
        """First-admission timestamp (active sort).

        The timestamp is set at first admission and survives reuse and
        run-history pruning; the earliest run start is only a fallback for
        records admitted before the timestamp was tracked.
        """
        if record.first_admitted_at is not None:
            return record.first_admitted_at
        starts = [
            run.started_at
            for run in (record.current_run, *record.run_history)
            if run is not None and run.started_at is not None
        ]
        return min(starts) if starts else float("inf")

    @staticmethod
    def _record_name(record: AgentRecord) -> str:
        """Derive the human-facing ``{identity}-{model}-{number}`` name (§5).

        The name is derived from the record's committed admission state
        (identity display label captured at admission, canonical base model
        with the effective-model fallback) and the handle's sequence digits.
        """
        return derive_agent_name(
            record.agent_id,
            record.identity,
            record.identity_display_label,
            record.base_model,
            record.effective_model,
        )

    def _agent_summaries(self) -> list[AgentSummary]:
        now = self._clock()
        global_ttl = (
            self._retention_policy.idle_ttl_seconds
            if self._generation_identity is not None
            else (
                self._root.agent_loop.config.subagents.idle_ttl_seconds
                if self._root
                else 0
            )
        )
        entries: list[tuple[AgentRecord, AgentSummary]] = []
        for record in self._agent_records.values():
            ttl = record.effective_idle_ttl(global_ttl)
            run = record.current_run
            latest_run_id = record.latest_run_id or (
                record.run_history[-1].run_id if record.run_history else None
            )
            idle_seconds = (
                max(0.0, now - record.idle_since)
                if record.state in {_AgentState.IDLE, _AgentState.FINALIZING}
                and record.idle_since is not None
                else None
            )
            # WP1: the latest failed run's public typed failure envelope,
            # searched latest-first (current run, then run history); the
            # run record is the canonical store, so it survives result expiry.
            last_failure = None
            if run is not None and run.failure is not None:
                last_failure = run.failure
            else:
                for history_run in reversed(record.run_history):
                    if history_run.failure is not None:
                        last_failure = history_run.failure
                        break
            # WP2: the suspended barrier's recovery linkage. The published
            # envelope stays immutable; recovery_eligibility is the EFFECTIVE
            # view (a reconcile_result submission flips it). None when no
            # barrier is active.
            recovery_eligibility = None
            if record.state is _AgentState.SUSPENDED and last_failure is not None:
                if record.reconcile_outcome == "cleared":
                    recovery_eligibility = RecoveryEligibility.ELIGIBLE
                elif record.reconcile_outcome == "prohibited":
                    recovery_eligibility = RecoveryEligibility.PROHIBITED
                else:
                    recovery_eligibility = last_failure.recovery.eligibility
            latest_run = run or (record.run_history[-1] if record.run_history else None)
            summary = AgentSummary(
                agent_id=record.agent_id,
                identity=record.identity,
                identity_revision=record.identity_revision,
                availability=record.availability,
                current_run_id=run.run_id if run is not None else None,
                current_run_status=run.status if run is not None else None,
                turns_used=self._agent_turns_used(record),
                last_run_status=record.last_run_status,
                initial_task_summary=record.initial_task_summary,
                current_task_summary=(
                    run.task_summary if run is not None else record.last_task_summary
                ),
                idle_seconds=idle_seconds,
                latest_run_id=latest_run_id,
                run_elapsed_seconds=self._run_elapsed_seconds(
                    run or (record.run_history[-1] if record.run_history else None), now
                ),
                ttl_remaining_seconds=(
                    max(0.0, ttl - idle_seconds)
                    if ttl and idle_seconds is not None
                    else None
                ),
                effective_model=record.effective_model,
                base_model=record.base_model,
                active_provider=record.active_provider,
                effective_thinking=record.effective_thinking,
                context_tokens=record.context_tokens,
                context_window=record.context_window,
                compacting=record.compacting,
                stop_reason=record.stop_reason,
                last_failure=last_failure,
                recovery_chain_id=record.recovery_chain_id,
                same_agent_resumes_used=(
                    record.same_agent_resumes_used
                    if record.recovery_chain_id is not None
                    else None
                ),
                resumed_from_run_id=(
                    latest_run.resumed_from_run_id if latest_run is not None else None
                ),
                recovery_eligibility=recovery_eligibility,
                # WP5: the predecessor/package lineage pair of the latest
                # claimed run (a replacement carries its predecessor's package
                # identity; None for a fresh agent).
                work_package_id=record.work_package_id,
                predecessor_agent_id=record.predecessor_agent_id,
                result_expired=(
                    (record.agent_id, latest_run_id) in self._expired_results
                    if latest_run_id is not None
                    else False
                ),
                name=self._record_name(record),
            )
            entries.append((record, summary))
        # Deterministic server ordering: Active (oldest launch first), then
        # Retained Idle (most recently completed first), then Recent terminal
        # metadata (newest closure first); numeric agent id breaks ties.
        # WP2: suspended/restoring records hold resident runtimes awaiting
        # recovery, so they sort with Active, never with Retained Idle.
        active = [
            entry
            for entry in entries
            if entry[0].state
            in {
                _AgentState.RUNNING,
                _AgentState.FINALIZING,
                _AgentState.SUSPENDED,
                _AgentState.RESTORING,
            }
        ]
        idle = [
            entry
            for entry in entries
            if entry[0].state
            not in {
                _AgentState.RUNNING,
                _AgentState.FINALIZING,
                _AgentState.SUSPENDED,
                _AgentState.RESTORING,
            }
        ]
        active.sort(
            key=lambda entry: (
                self._record_launch_key(entry[0]),
                self._agent_id_sort_key(entry[0].agent_id),
            )
        )
        idle.sort(
            key=lambda entry: (
                -(
                    entry[0].idle_since
                    if entry[0].idle_since is not None
                    else float("-inf")
                ),
                self._agent_id_sort_key(entry[0].agent_id),
            )
        )
        tombstones = sorted(
            self._evicted_agents.values(),
            key=lambda tombstone: (
                -tombstone.closed_at,
                self._agent_id_sort_key(tombstone.summary.agent_id),
            ),
        )
        return [
            *(summary for _record, summary in active),
            *(summary for _record, summary in idle),
            *(tombstone.summary for tombstone in tombstones),
        ]

    def _visible_summaries(self) -> list[AgentSummary]:
        """Public inventory: summaries minus expired terminal metadata.

        An expired tombstone behaves as absent in every published listing.
        The internal `_agent_summaries` helper stays unfiltered (bounded by
        the reaper and the insertion cap) so frozen snapshots remain
        readable until the reaper prunes them.
        """
        now = self._clock()
        expired = {
            agent_id
            for agent_id, tombstone in self._evicted_agents.items()
            if self._tombstone_expired(tombstone, now)
        }
        if not expired:
            return self._agent_summaries()
        return [
            summary
            for summary in self._agent_summaries()
            if summary.agent_id not in expired
        ]

    async def _emit_agents_update(
        self, evictions: list[AgentEviction] | None = None
    ) -> None:
        if self._notify_agents is None:
            return
        await self._notify_agents(
            [
                AgentSummaryModel(
                    agent_id=summary.agent_id,
                    identity=summary.identity,
                    identity_revision=summary.identity_revision,
                    name=summary.name,
                    launch_purposes=(
                        project_launch_purposes(record.runtime.agent_loop)
                        if (record := self._agent_records.get(summary.agent_id))
                        is not None
                        else []
                    ),
                    availability=summary.availability.value,
                    current_run_id=summary.current_run_id,
                    current_run_status=(
                        summary.current_run_status.value
                        if summary.current_run_status is not None
                        else None
                    ),
                    turns_used=summary.turns_used,
                    last_run_status=(
                        summary.last_run_status.value
                        if summary.last_run_status is not None
                        else None
                    ),
                    initial_task_summary=summary.initial_task_summary,
                    current_task_summary=summary.current_task_summary,
                    idle_seconds=summary.idle_seconds,
                    latest_run_id=summary.latest_run_id,
                    run_elapsed_seconds=summary.run_elapsed_seconds,
                    ttl_remaining_seconds=summary.ttl_remaining_seconds,
                    effective_model=summary.effective_model,
                    base_model=summary.base_model,
                    active_provider=summary.active_provider,
                    effective_thinking=summary.effective_thinking,
                    context_tokens=summary.context_tokens,
                    context_window=summary.context_window,
                    compacting=summary.compacting,
                    stop_reason=(
                        PublicRunStopReason(summary.stop_reason.value)
                        if summary.stop_reason is not None
                        else None
                    ),
                    last_failure=_public_failure_envelope(summary.last_failure),
                    recovery_chain_id=summary.recovery_chain_id,
                    same_agent_resumes_used=summary.same_agent_resumes_used,
                    resumed_from_run_id=summary.resumed_from_run_id,
                    recovery_eligibility=(
                        PublicRecoveryEligibility(summary.recovery_eligibility.value)
                        if summary.recovery_eligibility is not None
                        else None
                    ),
                    # WP5: the predecessor/package lineage pair.
                    work_package_id=summary.work_package_id,
                    predecessor_agent_id=summary.predecessor_agent_id,
                    result_expired=summary.result_expired,
                )
                for summary in self._visible_summaries()
            ],
            evictions or [],
        )

    def _schedule_agents_update_locked(self) -> None:
        """Publish the current inventory on a later turn (caller holds the lock)."""
        if self._notify_agents is None or self._draining_children:
            return

        async def publish() -> None:
            try:
                await self._emit_agents_update()
            except Exception as exc:
                logger.warning(
                    "Failed to publish background agent update", exc_info=exc
                )

        task = asyncio.create_task(publish(), name="vibe-subagent-agents-update")
        self._teardown_tasks.add(task)
        task.add_done_callback(self._teardown_tasks.discard)

    async def check_agents(self) -> list[AgentSummary]:
        async with self._registry_lock:
            self._expire_results_locked()
            # Lazy terminal-metadata expiry: an expired tombstone behaves as
            # absent in the public listing. The internal `_agent_summaries`
            # helper stays unfiltered (bounded by the reaper and the insertion
            # cap) so frozen snapshots remain readable until reaped.
            return self._visible_summaries()

    @staticmethod
    def _interrupt_run(record: AgentRecord, run: RunRecord) -> bool:
        """Delegate to the run results collaborator."""
        return RunResults.interrupt_operation(record, run)

    async def cancel_run(
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        reason: RunStopReason,
        requester_session_id: str,
    ) -> CancelResult:
        """Delegate to the run results collaborator (WP3 cancellation gate).

        An authoritative cancellation of a SUSPENDED agent invalidates its
        pending recovery FIRST (handoff §1.8: cancellation invalidates
        scheduled recovery): the barrier converts to a closed terminal
        tombstone — the failed run's stored result survives — and the
        cancel reports STOP_REQUESTED. A foreign/stale request, a
        non-suspended record, or a run pinned away from the barrier falls
        through to the run results collaborator's own authority checks.
        """
        intercepted: CancelResult | None = None
        async with self._registry_lock:
            record = self._agent_records.get(agent_id)
            if record is not None and record.state is _AgentState.SUSPENDED:
                if (
                    not self._admission_closed
                    and not self._draining_children
                    and (
                        reason is RunStopReason.USER_CANCELLED
                        or requester_session_id == record.parent_session_id
                    )
                ):
                    barrier_run_id = record.latest_run_id or (
                        record.current_run.run_id
                        if record.current_run is not None
                        else None
                    )
                    if run_id is None or run_id == barrier_run_id:
                        intercepted = self._cancel_suspension_locked(record, reason)
        if intercepted is not None:
            try:
                await self._emit_agents_update()
            except Exception as exc:
                logger.warning(
                    "Failed to publish suspension cancellation update", exc_info=exc
                )
            return intercepted
        return await self._run_results.cancel(
            agent_id, run_id, reason=reason, requester_session_id=requester_session_id
        )

    def _cancel_suspension_locked(
        self, record: AgentRecord, reason: RunStopReason
    ) -> CancelResult:
        """Invalidate a suspended barrier on an authoritative cancellation.

        The caller holds the registry lock. The conversion keeps the failed
        run's stored result and closes the warm runtime through the normal
        ownership paths; the tombstone records the cancellation stop reason.
        WP4: the cancellation also cancels the record's pending automatic
        recovery (rollback: cancel timers, retain counters/results).
        """
        barrier_run_id = record.latest_run_id or (
            record.current_run.run_id if record.current_run is not None else None
        )
        record.stop_reason = reason
        self._recovery_controller.cancel_pending(record.agent_id)
        self._suspension_expire_record_locked(record)
        return CancelResult(
            outcome=CancelOutcome.STOP_REQUESTED,
            run_id=barrier_run_id,
            stop_reason=reason,
        )

    def _resolve_run(
        self, agent_id: str, run_id: str | None
    ) -> tuple[tuple[str, str], RunRecord | StoredRunResult] | None:
        """Delegate to the run results collaborator."""
        return self._run_results.resolve_locked(agent_id, run_id)

    async def get_agent_result(
        self, agent_id: str, run_id: str | None = None
    ) -> TaskResult | None:
        """Delegate to the run results collaborator."""
        return await self._run_results.get_result(agent_id, run_id)

    async def wait_for_agent(
        self, agent_id: str, run_id: str | None = None, *, timeout: float | None = None
    ) -> WaitForAgentResult:
        """Delegate to the run results collaborator."""
        return await self._run_results.wait(agent_id, run_id, timeout=timeout)

    async def _still_running_disposition(
        self,
        agent_id: str,
        key: tuple[str, str],
        target: RunRecord,
        *,
        run_id: str | None,
        wait_started_at: float,
    ) -> WaitForAgentResult:
        """Delegate to the run results collaborator."""
        return await self._run_results.still_running_disposition(
            agent_id, key, target, run_id=run_id, wait_started_at=wait_started_at
        )

    def _eligible_idle_locked(self) -> list[AgentRecord]:
        return [
            record
            for record in self._agent_records.values()
            if record.state is _AgentState.IDLE
            and not record.reserved_for_replacement
            and record.idle_since is not None
            and self._generation_is_current(record.root_generation)
        ]

    @staticmethod
    def _agent_id_sort_key(agent_id: str) -> int:
        """Numeric suffix of an ``agent-N`` handle for deterministic tie-breaks."""
        prefix = "agent-"
        if agent_id.startswith(prefix):
            tail = agent_id[len(prefix) :]
            if tail.isdigit():
                return int(tail)
        return -1

    def _build_terminal_tombstone(
        self,
        record: AgentRecord,
        run: RunRecord | None,
        availability: AgentAvailability,
        idle_seconds: float,
    ) -> _AgentTombstone | None:
        """Delegate to the agent retention collaborator."""
        return self._agent_retention.build_terminal(
            record, run, availability, idle_seconds
        )

    def _insert_tombstone_locked(
        self, agent_id: str, tombstone: _AgentTombstone
    ) -> None:
        """Delegate to the agent retention collaborator."""
        self._agent_retention.insert_locked(agent_id, tombstone)

    @staticmethod
    def _tombstone_expired(tombstone: _AgentTombstone, now: float) -> bool:
        """Delegate to the agent retention collaborator."""
        return AgentRetention.tombstone_expired(tombstone, now)

    def _live_tombstone_locked(
        self, agent_id: str, now: float
    ) -> _AgentTombstone | None:
        """Delegate to the agent retention collaborator."""
        return self._agent_retention.live_tombstone_locked(agent_id, now)

    def _prune_expired_tombstones_locked(self, now: float) -> list[str]:
        """Delegate to the agent retention collaborator."""
        return self._agent_retention.prune_locked(now)

    def _close_unretained_locked(
        self, record: AgentRecord, run: RunRecord | None, *, name: str
    ) -> bool:
        """Delegate to the agent retention collaborator."""
        return self._agent_retention.close_unretained_locked(record, run, name=name)

    def _rearm_reaper_locked(self, exclude: frozenset[str] = frozenset()) -> None:
        """Delegate to the agent retention collaborator."""
        self._agent_retention.rearm_locked(exclude)

    async def _publish_inventory_update(self) -> None:
        """Delegate to the agent retention collaborator."""
        await self._agent_retention.publish_inventory_change()

    def _detach_reaper_locked(self) -> asyncio.Task[None] | None:
        """Delegate to the agent retention collaborator."""
        return self._agent_retention.detach_reaper_locked()

    async def _run_reaper(self, delay: float) -> None:
        """Delegate to the agent retention collaborator."""
        await self._agent_retention.run_reaper(delay)

    async def _evict_agent(
        self,
        agent_id: str,
        reason: str,
        notifications: list[EvictionNotice]
        | list[tuple[AgentRecord, AgentEviction, _AgentTombstone | None, object]]
        | None = None,
    ) -> bool:
        """Delegate to the agent retention collaborator.

        The ``notifications`` parameter keeps the historical four-element
        tuple annotation alongside the ``EvictionNotice`` list: the
        test_subagent_lifecycle.py:1316 rider patches this delegate through
        a tuple-annotated wrapper, while the reaper passes the notice list.
        """
        return await self._agent_retention.evict(
            agent_id, reason, cast("list[EvictionNotice] | None", notifications)
        )

    @staticmethod
    def _eviction_run_segments(
        record: AgentRecord, eviction: AgentEviction
    ) -> tuple[str, str]:
        """Delegate to the agent retention collaborator."""
        return AgentRetention.eviction_run_segments(record, eviction)

    async def _notify_evictions(self, notifications: list[EvictionNotice]) -> bool:
        """Delegate to the agent retention collaborator."""
        return await self._agent_retention.notify_evictions(notifications)

    def _prepare_release_locked(
        self, agent_id: str
    ) -> tuple[asyncio.Task[None], ReleaseAgentOutcome]:
        """Delegate to the agent release collaborator."""
        return self._agent_release.prepare_locked(agent_id)

    async def _join_release_teardown(
        self, teardown: asyncio.Task[None], agent_id: str, outcome: ReleaseAgentOutcome
    ) -> ReleaseAgentOutcome:
        """Delegate to the agent release collaborator."""
        return await AgentRelease.join_teardown(teardown, agent_id, outcome)

    async def release_agent(self, agent_id: str) -> ReleaseAgentOutcome:
        """Delegate to the agent release collaborator."""
        # WP4: releasing an agent invalidates its pending automatic
        # recovery FIRST (rollback: cancel timers, retain counters/results).
        self._recovery_controller.cancel_pending(agent_id)
        return await self._agent_release.release(agent_id)

    async def release_agent_guarded(
        self, agent_id: str, expected_run_id: str | None = None
    ) -> ReleaseAgentOutcome:
        """Delegate to the agent release collaborator."""
        # WP4: see release_agent.
        self._recovery_controller.cancel_pending(agent_id)
        return await self._agent_release.release_guarded(agent_id, expected_run_id)

    def _suspend_after_failure_locked(
        self, record: AgentRecord, run: RunRecord
    ) -> bool:
        """Delegate to the recovery collaborator (WP2 suspension hook)."""
        return self._recovery.suspend_after_failure_locked(record, run)

    def _observe_run_completion_locked(
        self, record: AgentRecord, run: RunRecord
    ) -> None:
        """Delegate to the automatic recovery controller (WP4 observation).

        INERT unless ``[recovery.subagents] enabled`` — suspension (WP2/WP3)
        and the explicit resume/reconcile ports are untouched either way.
        """
        self._recovery_controller.observe_run_locked(record, run)

    def _invalidate_recovery_locked(self, record: AgentRecord | None) -> None:
        """Delegate to the recovery collaborator (WP2 invalidation).

        WP3: the invalidation also drops the record's suspension storage
        seam state — release/replacement forfeits any suspended capture.
        WP4: the invalidation also stops the record's automatic-recovery
        episode and cancels its pending schedule.
        """
        self._recovery_controller.invalidate(record)
        self._recovery.invalidate_locked(record)
        if record is not None:
            self._suspension.discard_locked(record.agent_id)

    def _suspension_admit_locked(
        self, record: AgentRecord, config: RecoveryConfig
    ) -> bool:
        """Delegate to the suspension lifecycle collaborator (WP3 byte cap)."""
        return self._suspension.admit_locked(record, config)

    def _suspension_capture_locked(self, record: AgentRecord) -> None:
        """Delegate to the suspension lifecycle collaborator (WP3 seam)."""
        self._suspension.capture_locked(record)

    def _suspension_next_expiry_locked(self) -> float | None:
        """Delegate to the suspension lifecycle collaborator (WP3 TTL)."""
        return self._suspension.next_expiry_locked(_recovery_config(self))

    def _suspension_expire_elapsed_locked(self, now: float) -> list[AgentRecord]:
        """Delegate to the suspension lifecycle collaborator (WP3 expiry)."""
        return self._suspension.expire_elapsed_locked(now, _recovery_config(self))

    def _suspension_expire_record_locked(self, record: AgentRecord) -> None:
        """Delegate to the suspension lifecycle collaborator (WP3 conversion)."""
        self._suspension.expire_locked(record)

    def _suspension_discard_locked(self, agent_id: str) -> None:
        """Delegate to the suspension lifecycle collaborator (WP3 seam)."""
        self._suspension.discard_locked(agent_id)

    async def resume_agent(
        self,
        agent_id: str,
        expected_run_id: str,
        idempotency_key: str,
        *,
        requester_session_id: str,
        override_cooldown: bool = False,
    ) -> ResumeAgentResult:
        """Delegate to the recovery collaborator (WP2 explicit resume)."""
        return await self._recovery.resume_agent(
            agent_id,
            expected_run_id,
            idempotency_key,
            requester_session_id=requester_session_id,
            override_cooldown=override_cooldown,
        )

    async def reconcile_result(
        self,
        agent_id: str,
        expected_run_id: str,
        transcript_boundary: str,
        verified_effects: Sequence[EffectVerification],
        *,
        requester_session_id: str,
    ) -> ReconcileResult:
        """Delegate to the recovery collaborator (WP2 reconciliation)."""
        return await self._recovery.reconcile_result(
            agent_id,
            expected_run_id,
            transcript_boundary,
            verified_effects,
            requester_session_id=requester_session_id,
        )

    @classmethod
    def _resolve_run_outcome(
        cls, run: RunRecord, turn: PublicTurn | None = None
    ) -> tuple[RunStatus, RunStopReason | None]:
        """Requested reasons attribute interruption, never completion or failure."""
        if turn is None:
            status = (
                run.status
                if run.status is not RunStatus.RUNNING
                else RunStatus.CANCELLED
            )
            inferred = run.stop_reason
            if inferred is None and status is RunStatus.CANCELLED:
                inferred = RunStopReason.ORCHESTRATOR_CANCELLED
            elif inferred is None and status is RunStatus.FAILED:
                inferred = RunStopReason.ERROR
        else:
            status = (
                RunStatus.FAILED
                if turn.error is not None or turn.status is PublicTurnStatus.FAILED
                else RunStatus.COMPLETED
                if turn.status is PublicTurnStatus.COMPLETED
                else RunStatus.CANCELLED
            )
            inferred = cls._run_stop_reason(turn)
        reason = (
            run.requested_stop_reason or inferred
            if status is RunStatus.CANCELLED
            else inferred
        )
        return status, reason

    @staticmethod
    def _run_stop_reason(turn: PublicTurn) -> RunStopReason | None:
        if turn.stop_reason is PublicTurnStopReason.LIMIT:
            return RunStopReason.BUDGET_EXCEEDED
        if turn.stop_reason is not None:
            return RunStopReason(turn.stop_reason.value)
        if turn.error is not None or turn.status is PublicTurnStatus.FAILED:
            return RunStopReason.ERROR
        if turn.status is PublicTurnStatus.INTERRUPTED:
            return RunStopReason.USER_CANCELLED
        return None

    async def _capture_child_notification(
        self,
        runtime: SessionRuntime,
        record: AgentRecord | None,
        run: RunRecord | None,
        params: ProtocolModel,
    ) -> None:
        """Delegate to the child runtime assembler collaborator."""
        await self._child_runtime.capture_child_notification(
            runtime, record, run, params
        )

    def _build_child_runtime(
        self,
        child: AgentLoop,
        *,
        base_history: list[PublicHistoryEntry] | None = None,
        event_sink: Callable[[BaseEvent], Awaitable[None]] | None = None,
    ) -> SessionRuntime:
        """Delegate to the child runtime assembler collaborator."""
        return self._child_runtime.build(
            child, SessionRuntime, base_history=base_history, event_sink=event_sink
        )

    def _runtime(self, session_id: str | None) -> SessionRuntime:
        """Delegate to the child runtime assembler collaborator."""
        return self._child_runtime.runtime(session_id)

    def _require_child(self, session_id: str) -> SessionRuntime:
        """Delegate to the child runtime assembler collaborator."""
        return self._child_runtime.require_child(session_id)

    def reopen_child_admission(self) -> None:
        """Reopen child admission after a root transition completes."""
        self._admission_closed = False

    async def reserve_resume_admission(self, root: SessionRuntime) -> None:
        """Delegate to the child drain collaborator (WP4 prelude).

        A session switch forfeits pending automatic recovery along with the
        retiring generation: cancel every controller schedule before resume
        admission is reserved (rollback: cancel timers, retain counters).
        """
        self._recovery_controller.cancel_all()
        await self._child_drain.reserve_resume(root)

    async def close_children(self) -> None:
        """Delegate to the child drain collaborator."""
        await self._child_drain.close_children()

    async def drain_children(self) -> None:
        """Delegate to the child drain collaborator (WP4 prelude).

        Shutdown cancels every pending automatic-recovery schedule (rollback:
        cancel timers, retain counters/results).
        """
        self._recovery_controller.cancel_all()
        await self._child_drain.drain_children()

    @staticmethod
    async def _discard_child(child: AgentLoop) -> None:
        """Delegate to the child drain collaborator."""
        await ChildDrain.discard_child(child, close_agent_loop)
