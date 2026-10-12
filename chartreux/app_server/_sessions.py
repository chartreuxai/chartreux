from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
)
from contextlib import ExitStack, asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, field
import enum
import fnmatch
import itertools
from pathlib import Path
import time
from typing import Any, cast
from uuid import uuid4

from chartreux.app_server._execution import (
    SessionExecution,
    SessionExecutionConflict,
    SessionExecutionKind,
)
from chartreux.app_server._model import ProtocolModel
from chartreux.app_server._projection import (
    project_history,
    project_launch_slot_purposes,
    project_session_log,
)
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
from chartreux.app_server._turns import DeliverCallback, TurnController, TurnStartAction
from chartreux.app_server.models import (
    CallbackOutput,
    OpenCallbackState,
    PublicCallbackEntry,
    PublicCheckpointEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
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
    DangerousCommandAuthorityReadResponse,
    DangerousCommandDenylistBlockWire,
    DangerousCommandGrantScopeWire,
    DangerousCommandGrantWire,
    DangerousCommandRevocationOutcome,
    DangerousCommandRevokeResponse,
    HistoryEntryAddedParams,
    HistoryEntryUpdatedParams,
    RunStopReason as PublicRunStopReason,
    SessionCompactedParams,
    StatsUpdatedParams,
    TurnCompletedParams,
    TurnInterruptParams,
    TurnStartParams,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop import _PreparedPolicyReplacement
from chartreux.core.agents.launch import FrozenPersona, LaunchCandidate, resolve_launch
from chartreux.core.agents.models import AgentProfile
from chartreux.core.config._dangerous_command_persistence import (
    read_saved_dangerous_command_grants,
)
from chartreux.core.config._restrictions import (
    SourceRestrictions,
    partition_policy_sources,
)
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.dangerous_commands import (
    OPERATION_REGISTRY,
    DangerousCommandAuthority,
    RepositoryScope,
)
from chartreux.core.events import BaseEvent, ToolStreamEvent
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.saved_sessions import delete_saved_session
from chartreux.core.session_types import ChildSessionLink
from chartreux.core.subagents import (
    AgentAvailability,
    AgentBusyError,
    AgentClosedError,
    AgentEvictedError,
    AgentEviction,
    AgentProfileMismatchError,
    AgentResultExpiredError,
    AgentSummary,
    CancelOutcome,
    CancelResult,
    LaunchConfig,
    LaunchConfigError,
    LaunchOutcome,
    ReleaseAgentOutcome,
    RetentionPolicy,
    RunStatus,
    RunStopReason,
    SubagentRunAccumulator,
    SubagentRunnerPort,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
    WaitForAgentResult,
    WaitStatus,
    derive_agent_name,
    normalize_task_summary,
    prepare_subagent_prompt,
    still_running_guidance,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    canonical_git_command,
    classify_dangerous_git_command,
    matches_command_prefix,
    resolve_repository_scope,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    _split_command_tokens,
)
from chartreux.core.tools.io_port import ToolIOPort
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.models import ToolPermission, ToolPermissionError
from chartreux.model_display import format_model_display_name
from chartreux.observability.logging import logger

type Notify = Callable[[str, ProtocolModel], Awaitable[None]]
type NotifyAgents = Callable[
    [list[AgentSummaryModel], list[AgentEviction]], Awaitable[None]
]
type Wakeup = Callable[[float], Awaitable[None]]
type EventWatermark = Callable[[str], int]

_BACKGROUND_PROGRESS_LIMIT = 32
_MAX_RUN_HISTORY = 32
_MAX_STORED_RESULTS = 32
# Per root-session registry; retained idle agents do not consume active-work slots.
_DEFAULT_MAX_RUNNING_SUBAGENTS = 16
_AGENT_ID_HIGH_WATER_MARK_KEY = "_app_server_agent_id_high_water_mark"


@dataclass(eq=False, slots=True)
class _ActiveWorkSlot:
    transferred: bool = False


class _AgentState(enum.StrEnum):
    RUNNING = "running"
    FINALIZING = "finalizing"
    IDLE = "idle"
    EVICTING = "evicting"


@dataclass(frozen=True, slots=True)
class StoredRunResult:
    agent_id: str
    run_id: str
    result: TaskResult
    completed_at: float
    root_generation: int
    terminal_identity: tuple[str, str, str] | None = None


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


@dataclass(slots=True)
class RunRecord:
    run_id: str
    agent_id: str
    profile: str
    status: RunStatus
    completion_task: asyncio.Future[Any]
    task_summary: str | None = None
    start_count: int = 0
    result: TaskResult | None = None
    started_at: float | None = None
    completed_at: float | None = None
    terminal_identity: tuple[str, str, str] | None = None
    progress_summaries: list[str] = field(default_factory=list)
    progress_overflow: bool = False
    stop_reason: RunStopReason | None = None
    requested_stop_reason: RunStopReason | None = None
    initial_turn_id: str | None = None


@dataclass(slots=True)
class AgentRecord:
    agent_id: str
    profile: str
    session_id: str
    runtime: SessionRuntime
    root_generation: int
    idle_ttl_seconds: int | None = None
    retained: bool = False
    profile_display_label: str | None = None
    parent_session_id: str = ""
    initial_task_summary: str | None = None
    state: _AgentState = _AgentState.IDLE
    current_run: RunRecord | None = None
    run_history: list[RunRecord] = field(default_factory=list)
    idle_since: float | None = None
    last_task_summary: str | None = None
    last_run_status: RunStatus | None = None
    latest_run_id: str | None = None
    effective_model: str | None = None
    base_model: str | None = None
    active_provider: str | None = None
    effective_thinking: str | None = None
    context_tokens: int | None = None
    context_window: int | None = None
    compacting: bool = False
    compaction_entry_id: str | None = None
    stop_reason: RunStopReason | None = None
    reserved_for_replacement: bool = False
    replacement_owner: _ReplacementClaim | None = None
    work_slot: _ActiveWorkSlot | None = None
    first_admitted_at: float | None = None

    def effective_idle_ttl(self, global_ttl: int | float) -> int | float:
        return (
            self.idle_ttl_seconds if self.idle_ttl_seconds is not None else global_ttl
        )

    @property
    def availability(self) -> AgentAvailability:
        if self.state is _AgentState.RUNNING:
            return AgentAvailability.RUNNING
        if self.state is _AgentState.FINALIZING:
            return AgentAvailability.FINALIZING
        return AgentAvailability.IDLE

    @availability.setter
    def availability(self, value: AgentAvailability) -> None:
        self.state = (
            _AgentState.RUNNING
            if value is AgentAvailability.RUNNING
            else _AgentState.IDLE
        )


@dataclass(frozen=True, slots=True)
class _ReplacementClaim:
    record: AgentRecord
    run: RunRecord
    parent: SessionRuntime
    parent_identity: tuple[str, int]
    child_generation: int
    authority_revision: int
    slot: _ActiveWorkSlot


@dataclass(frozen=True, slots=True)
class _AgentTombstone:
    """Private terminal metadata kept alongside the public agent summary.

    ``reason`` is ``"closed"`` for a completed-then-closed instance (reuse
    raises :class:`AgentClosedError`) and ``"evicted"`` for a TTL/cap eviction
    (reuse raises :class:`AgentEvictedError`). ``closed_at``/``expires_at`` bound
    the metadata lifetime; ``expires_at`` is always set for a stored tombstone
    (a zero tombstone TTL stores no terminal metadata at all).
    ``profile_display_label`` is captured at admission for later naming work.
    """

    summary: AgentSummary
    child_session_id: str
    parent_identity: tuple[str, int]
    reason: str = "evicted"
    closed_at: float = 0.0
    expires_at: float | None = None
    profile_display_label: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptReadSnapshot:
    """Authorization captured for a disk-only child transcript read."""

    agent_id: str
    child_session_id: str
    child_dir: Path | None
    parent_dir: Path | None
    parent_identity: tuple[str, int]
    parent_runtime_token: int
    link_identity: tuple[str, str, str, str | None] | None

    @property
    def has_saved_transcript(self) -> bool:
        return self.child_dir is not None


@dataclass(frozen=True, slots=True)
class ResidentTranscriptReadSnapshot:
    """An authorized immutable message snapshot from a resident child runtime."""

    agent_id: str
    child_session_id: str
    parent_identity: tuple[str, int]
    parent_runtime_token: int
    record_token: int
    runtime_token: int
    messages: list[LLMMessage]


@dataclass(frozen=True, slots=True)
class StoredChildSession:
    """A child record whose runtime has not been materialized."""

    parent_runtime: SessionRuntime
    root_session_id: str
    root_generation: int
    child_dir: Path
    link: ChildSessionLink


@dataclass(frozen=True, slots=True)
class DangerousCommandGrantEntry:
    """One (operation, repository scope) authority entry of an inspection."""

    operation_id: str
    scope: RepositoryScope


@dataclass(frozen=True, slots=True)
class DangerousCommandDenylistBlock:
    """An explicit denylist pattern and the registry operations it vetoes."""

    pattern: str
    operations: tuple[str, ...]


def _grant_entry_sort_key(entry: DangerousCommandGrantEntry) -> tuple[str, str, str]:
    scope = entry.scope
    return (entry.operation_id, str(scope.git_dir), str(scope.work_tree))


def _grant_entries(
    authorities: tuple[DangerousCommandAuthority, ...],
) -> tuple[DangerousCommandGrantEntry, ...]:
    """Flatten authority snapshots into sorted (operation, scope) entries."""
    entries = {
        DangerousCommandGrantEntry(operation_id, authority.scope)
        for authority in authorities
        for operation_id in authority.operations
    }
    return tuple(sorted(entries, key=_grant_entry_sort_key))


def _grant_wire(entry: DangerousCommandGrantEntry) -> DangerousCommandGrantWire:
    scope = entry.scope
    return DangerousCommandGrantWire(
        operation_id=entry.operation_id,
        scope=DangerousCommandGrantScopeWire(
            git_dir=str(scope.git_dir),
            work_tree=str(scope.work_tree) if scope.work_tree is not None else None,
        ),
    )


def _accepted_saved_grants(
    loop: AgentLoop, project: Path
) -> tuple[DangerousCommandGrantEntry, ...]:
    """Saved user-config authority the session's accepted sources contribute.

    The union covers the session's inherited restrictions (a child accepts its
    parent's sources) and its own accepted restrictions; the same source can
    appear in both, so entries dedupe by project and operations. Only the
    registered project's grants are relevant to its inspection.
    """
    sources = (
        *loop.runtime_policy.inherited_restrictions,
        *loop.config_orchestrator.restrictions,
    )
    seen: set[tuple[str, tuple[str, ...]]] = set()
    entries: set[DangerousCommandGrantEntry] = set()
    for source in sources:
        for grant in source.dangerous_command_grants:
            key = (str(grant.project), grant.operations)
            if key in seen:
                continue
            seen.add(key)
            if str(grant.project) != str(project):
                continue
            scope = resolve_repository_scope(grant.project, GitRepositoryRelocation())
            if scope is None:
                continue
            entries.update(
                DangerousCommandGrantEntry(operation_id, scope)
                for operation_id in grant.operations
            )
    return tuple(sorted(entries, key=_grant_entry_sort_key))


# One canonical probe form per registry operation. The denylist is matched
# with the resolver's own prefix/canonical-git matching, so a probe doubles as
# the operation's representative command; tests pin each probe to exactly its
# operation through the frozen classifier.
_DANGEROUS_COMMAND_DENYLIST_PROBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("git.push", ("git", "push")),
    ("git.push.force", ("git", "push", "--force")),
    ("git.push.delete", ("git", "push", "--delete")),
    ("git.push.mirror", ("git", "push", "--mirror")),
    ("git.checkout", ("git", "checkout")),
    ("git.restore", ("git", "restore")),
    ("git.switch.discard", ("git", "switch", "--discard-changes")),
    ("git.stash.drop", ("git", "stash", "drop")),
    ("git.stash.clear", ("git", "stash", "clear")),
    ("git.reflog.expire", ("git", "reflog", "expire")),
    ("git.reflog.delete", ("git", "reflog", "delete")),
)


def _explicit_shell_denylist(loop: AgentLoop) -> tuple[str, ...]:
    """Configured bash denylist entries, without the shipped defaults.

    Mirrors the tool manager's restriction assembly (inherited plus accepted
    sources); the merged tool config also carries built-in defaults, which are
    not explicit contributions and never mention dangerous git operations.
    """
    sources = (
        *loop.runtime_policy.inherited_restrictions,
        *loop.config_orchestrator.restrictions,
    )
    patterns: list[str] = []
    for source in sources:
        for restriction in source.tools:
            if restriction.tool_name != "bash":
                continue
            for pattern in restriction.denylist:
                if pattern not in patterns:
                    patterns.append(pattern)
    return tuple(patterns)


def _denylist_blocked_operations(pattern: str) -> tuple[str, ...]:
    """Registry operations one explicit denylist pattern absolutely vetoes.

    A pattern vetoes an operation when it prefix-matches one of the
    operation's representative probe forms (the resolver's own prefix and
    canonical-git matching), or when the pattern is itself a complete
    grantable form of the operation (``git push origin --delete``). Patterns
    that only block through an unclassifiable extension (an end-of-options
    marker, say) are not reported; the surface is informational and never
    suggests a grant overrides a veto.
    """
    pattern_tokens = _split_command_tokens(pattern)
    if not pattern_tokens:
        return ()
    direct = classify_dangerous_git_command(pattern_tokens)
    direct_operations = (
        direct.grantable.operation_ids if direct.grantable is not None else ()
    )
    blocked: list[str] = []
    for operation_id, probe in _DANGEROUS_COMMAND_DENYLIST_PROBES:
        if operation_id in direct_operations:
            blocked.append(operation_id)
            continue
        tokens = list(probe)
        canonical = canonical_git_command(tokens)
        if matches_command_prefix(tokens, pattern_tokens) or (
            canonical is not None and matches_command_prefix(canonical, pattern_tokens)
        ):
            blocked.append(operation_id)
    return tuple(blocked)


def _blocking_denylist(loop: AgentLoop) -> tuple[DangerousCommandDenylistBlock, ...]:
    """Explicit denylist entries blocking dangerous git operations.

    The denylist is an absolute veto checked before grantability, so these
    entries are reported apart from grant authority: no live or saved grant
    overrides them.
    """
    blocks: list[DangerousCommandDenylistBlock] = []
    for pattern in _explicit_shell_denylist(loop):
        operations = _denylist_blocked_operations(pattern)
        if operations:
            blocks.append(DangerousCommandDenylistBlock(pattern, operations))
    return tuple(blocks)


class SessionRuntimeRegistry(SubagentRunnerPort):  # noqa: PLR0904
    def __init__(
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
        """Persist one approved dangerous-command grant through the root orchestrator only.

        The calling session (root or child) is resolved through the registry,
        and the save is keyed by that registered session's cwd; a caller-supplied
        project path is never trusted. Child orchestrators never write: the save
        always targets the root orchestrator. Mid-turn safe: the orchestrator
        mutation lock is the only reservation, and nothing is published.
        """
        caller = self._resolve_session_loop(session_id)
        root_runtime = self._root
        if caller is None or root_runtime is None:
            raise ValueError(
                "Dangerous-command grant save requires a registered session and root"
            )
        return await root_runtime.agent_loop.config_orchestrator.save_dangerous_command_grant(
            project=caller.cwd,
            operation_id=operation_id,
            expected_revision=expected_revision,
        )

    def _dangerous_command_grant_save(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> Awaitable[ConfigSaveResult]:
        # The loop's proposal project is not a trusted key: the save is keyed
        # by the registered caller session's cwd, exactly like the roots save.
        del project
        return self.save_dangerous_command_grant(
            session_id=session_id,
            operation_id=operation_id,
            expected_revision=expected_revision,
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
        """Inspect one registered session's dangerous-command authority.

        The project is always the registered session's own cwd; no
        caller-supplied project key is accepted. Live grants come from the
        session's own store, accepted saved grants from the restrictions the
        session enforces, and the disk read is the fresh P2 projection with
        its revision. Blocking denylist entries are reported apart from grant
        authority because no grant overrides an explicit denylist veto.
        """
        runtime = self._require_live_session(session_id)
        loop = runtime.agent_loop
        store = loop._dangerous_command_grants
        project = loop.cwd
        root = self._root
        if root is None:
            raise ValueError("Dangerous-command authority requires a registered root")
        saved = await read_saved_dangerous_command_grants(
            root.agent_loop.config_orchestrator, project=project
        )
        return DangerousCommandAuthorityReadResponse(
            project=str(project),
            live_authority_generation=store.authority_generation,
            live_grants=[
                _grant_wire(entry)
                for entry in _grant_entries(store.dangerous_command_authorities())
            ],
            accepted_saved_grants=[
                _grant_wire(entry) for entry in _accepted_saved_grants(loop, project)
            ],
            saved_operations=(
                list(saved.operations) if saved.unavailable is None else None
            ),
            disk_revision=saved.user_revision,
            blocking_denylist=[
                DangerousCommandDenylistBlockWire(
                    pattern=block.pattern, operations=list(block.operations)
                )
                for block in _blocking_denylist(loop)
            ],
        )

    async def revoke_dangerous_command_grant(
        self,
        *,
        session_id: str,
        operation_id: str,
        expected_live_generation: int,
        expected_disk_revision: str | None,
    ) -> DangerousCommandRevokeResponse:
        """Revoke one operation's live and saved authority for a session.

        Live removal is guarded by the session store's monotonic generation:
        a stale expected generation aborts the whole revocation (the saved
        side is untouched) and reports the current generation to re-read. A
        matching generation removes the operation's session grants at every
        scope and invalidates installed pending once permits immediately;
        cached authority is dropped. A question already awaiting an answer is
        not cancelled: answering it after the revocation still grants fresh
        authority for the decided scope. Saved removal is the dedicated
        revision-checked user-layer remove routed through the root
        orchestrator and keyed by the registered session's cwd. The two
        outcomes are reported separately, and a failed saved removal warns
        that the authority can return on the next reload or start. When the
        session's accepted saved restrictions still authorize the revoked
        operation — whether or not the runtime store ever held it — the
        removal warns that the session keeps that authority until the next
        reload or start. The reported generation is read after the saved-side
        attempt, so a concurrent approval during the disk write is
        reflected.
        """
        runtime = self._require_live_session(session_id)
        loop = runtime.agent_loop
        store = loop._dangerous_command_grants
        if store.authority_generation != expected_live_generation:
            return DangerousCommandRevokeResponse(
                live=DangerousCommandRevocationOutcome(
                    status="conflict", generation=store.authority_generation
                ),
                saved=DangerousCommandRevocationOutcome(status="skipped"),
                live_authority_generation=store.authority_generation,
            )
        requirements = {
            (operation_id, authority.scope)
            for authority in store.dangerous_command_authorities()
            if operation_id in authority.operations
        }
        store.revoke_session(requirements)
        store.invalidate_pending(requirements)
        loop.tool_manager._invalidate_dangerous_command_authority()
        operation = OPERATION_REGISTRY[operation_id]
        warnings: list[str] = []
        if any(
            entry.operation_id == operation_id
            for entry in _accepted_saved_grants(loop, loop.cwd)
        ):
            # The runtime store no longer carries the operation — it may
            # never have — but the session's accepted restrictions still
            # authorize it, so the session's live authority persists until a
            # reload or restart.
            warnings.append(
                f"The live {operation_id} grant is removed, but this session "
                f"keeps {operation_id} authority from accepted config until "
                "the next config reload or session start."
            )
        if not operation.persistable:
            # Consequential operations are never saved, so no saved authority
            # can exist to remove.
            saved_outcome = DangerousCommandRevocationOutcome(status="not_applicable")
        elif expected_disk_revision is None:
            saved_outcome = DangerousCommandRevocationOutcome(status="skipped")
        else:
            root = self._root
            if root is None:
                raise ValueError(
                    "Dangerous-command revocation requires a registered root"
                )
            result = await root.agent_loop.config_orchestrator.remove_dangerous_command_grant(
                project=loop.cwd,
                operation_id=operation_id,
                expected_revision=expected_disk_revision,
            )
            if result.error == "conflict":
                saved_outcome = DangerousCommandRevocationOutcome(status="conflict")
            elif result.persistence == "saved" and result.error is None:
                saved_outcome = DangerousCommandRevocationOutcome(
                    status="removed", revision=result.revision
                )
            else:
                saved_outcome = DangerousCommandRevocationOutcome(
                    status="failed", revision=result.revision
                )
            if saved_outcome.status != "removed":
                warnings.append(
                    f"Removing the saved {operation_id} grant for {loop.cwd} did "
                    "not succeed, so the authority can return on the next config "
                    "reload or session start."
                )
        # Read the generation after the saved-side attempt: a concurrent
        # approval during the disk write bumps it, and the response reports
        # the post-operation value.
        generation = store.authority_generation
        return DangerousCommandRevokeResponse(
            live=DangerousCommandRevocationOutcome(
                status="removed", generation=generation
            ),
            saved=saved_outcome,
            live_authority_generation=generation,
            warning=" ".join(warnings) if warnings else None,
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
        """Find a live child or a stable snapshot of its stored link."""
        if child := self._children.get(session_id):
            return child
        if child := self._readable_children.get(session_id):
            return child
        if stored := self._stored_children.get(session_id):
            return stored
        resolved = self._resolve_child_link(session_id)
        if resolved is None:
            return None
        parent_runtime, root_agent_loop, child_dir, link = resolved
        return StoredChildSession(
            parent_runtime=parent_runtime,
            root_session_id=root_agent_loop.session_id,
            root_generation=root_agent_loop._session_generation,
            child_dir=child_dir,
            link=link,
        )

    def stored_child_is_current(self, stored: StoredChildSession) -> bool:
        """Check that a stored read did not race root or child-link changes."""
        root = self._root
        if self._stored_children.get(stored.link.session_id) is stored:
            return (
                root is stored.parent_runtime
                and root is not None
                and root.agent_loop.session_id == stored.root_session_id
                and root.agent_loop._session_generation == stored.root_generation
                and stored.link.session_id not in self._children
            )
        if (
            root is not stored.parent_runtime
            or root is None
            or root.agent_loop.session_id != stored.root_session_id
            or root.agent_loop._session_generation != stored.root_generation
            or stored.link.session_id in self._children
        ):
            return False
        resolved = self._resolve_child_link(stored.link.session_id)
        return (
            resolved is not None
            and resolved[0] is stored.parent_runtime
            and resolved[2] == stored.child_dir
            and resolved[3] == stored.link
        )

    @staticmethod
    def _transcript_link_identity(
        link: ChildSessionLink | None,
    ) -> tuple[str, str, str, str | None] | None:
        if link is None:
            return None
        return (link.session_id, link.tool_call_id, link.agent, link.relative_path)

    @staticmethod
    def _transcript_link_for(
        root: SessionRuntime, child_session_id: str
    ) -> ChildSessionLink | None:
        metadata = root.agent_loop.session_logger.session_metadata
        if metadata is None:
            return None
        return next(
            (
                link
                for link in metadata.child_sessions
                if link.session_id == child_session_id
            ),
            None,
        )

    async def resolve_resident_transcript_read(
        self, agent_id: str
    ) -> ResidentTranscriptReadSnapshot | None:
        """Snapshot a running resident child's messages under its MessageList lock."""
        async with self._registry_lock:
            root = self._root
            if root is None:
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            identity = (root.agent_loop.session_id, root.agent_loop._session_generation)
            if self._generation_identity != identity:
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            record = self._agent_records.get(agent_id)
            if record is None:
                # Expired terminal metadata behaves as absent for transcript reads.
                if self._live_tombstone_locked(agent_id, self._clock()) is None:
                    raise UnknownAgentError(f"Unknown agent: {agent_id}")
                return None
            if (
                record.root_generation != identity[1]
                or record.parent_session_id != identity[0]
                or record.state not in {_AgentState.RUNNING, _AgentState.FINALIZING}
                or self._children.get(record.session_id) is not record.runtime
            ):
                return None
            return ResidentTranscriptReadSnapshot(
                agent_id=agent_id,
                child_session_id=record.session_id,
                parent_identity=identity,
                parent_runtime_token=id(root),
                record_token=id(record),
                runtime_token=id(record.runtime),
                messages=list(record.runtime.agent_loop.messages),
            )

    async def resident_transcript_read_is_current(
        self, snapshot: ResidentTranscriptReadSnapshot
    ) -> bool:
        """Check that a resident transcript snapshot remains authorized."""
        async with self._registry_lock:
            root = self._root
            if (
                root is None
                or id(root) != snapshot.parent_runtime_token
                or (root.agent_loop.session_id, root.agent_loop._session_generation)
                != snapshot.parent_identity
                or self._generation_identity != snapshot.parent_identity
            ):
                return False
            record = self._agent_records.get(snapshot.agent_id)
            return (
                record is not None
                and id(record) == snapshot.record_token
                and record.session_id == snapshot.child_session_id
                and record.parent_session_id == snapshot.parent_identity[0]
                and record.root_generation == snapshot.parent_identity[1]
                and record.state in {_AgentState.RUNNING, _AgentState.FINALIZING}
                and id(record.runtime) == snapshot.runtime_token
                and self._children.get(record.session_id) is record.runtime
            )

    async def resolve_transcript_read(self, agent_id: str) -> TranscriptReadSnapshot:
        """Capture parent-authorized disk-read inputs without touching child runtime state."""
        async with self._registry_lock:
            root = self._root
            if root is None:
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            identity = (root.agent_loop.session_id, root.agent_loop._session_generation)
            if self._generation_identity != identity:
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            record = self._agent_records.get(agent_id)
            tombstone = self._live_tombstone_locked(agent_id, self._clock())
            if record is not None:
                if (
                    record.root_generation != identity[1]
                    or record.parent_session_id != identity[0]
                ):
                    raise UnknownAgentError(f"Unknown agent: {agent_id}")
                child_session_id = record.session_id
            elif tombstone is not None:
                if tombstone.parent_identity != identity:
                    raise UnknownAgentError(f"Unknown agent: {agent_id}")
                child_session_id = tombstone.child_session_id
            else:
                raise UnknownAgentError(f"Unknown agent: {agent_id}")

            link = self._transcript_link_for(root, child_session_id)
            link_identity = self._transcript_link_identity(link)
            parent_dir = root.agent_loop.session_logger.session_dir
            if link is None or link.relative_path is None or parent_dir is None:
                return TranscriptReadSnapshot(
                    agent_id=agent_id,
                    child_session_id=child_session_id,
                    child_dir=None,
                    parent_dir=None,
                    parent_identity=identity,
                    parent_runtime_token=id(root),
                    link_identity=link_identity,
                )
            relative_path = Path(link.relative_path)
            if relative_path.is_absolute() or any(
                part in {"", ".", ".."} for part in relative_path.parts
            ):
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            child_dir = parent_dir / relative_path
            if not child_dir.is_relative_to(parent_dir):
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            return TranscriptReadSnapshot(
                agent_id=agent_id,
                child_session_id=child_session_id,
                child_dir=child_dir,
                parent_dir=parent_dir,
                parent_identity=identity,
                parent_runtime_token=id(root),
                link_identity=link_identity,
            )

    async def transcript_read_is_current(
        self, snapshot: TranscriptReadSnapshot
    ) -> bool:
        """Check whether an authorized transcript read remains bound to its parent link."""
        async with self._registry_lock:
            root = self._root
            if (
                root is None
                or id(root) != snapshot.parent_runtime_token
                or (root.agent_loop.session_id, root.agent_loop._session_generation)
                != snapshot.parent_identity
                or self._generation_identity != snapshot.parent_identity
            ):
                return False
            record = self._agent_records.get(snapshot.agent_id)
            tombstone = self._live_tombstone_locked(snapshot.agent_id, self._clock())
            authorized = (
                record is not None
                and record.session_id == snapshot.child_session_id
                and record.parent_session_id == snapshot.parent_identity[0]
                and record.root_generation == snapshot.parent_identity[1]
            ) or (
                tombstone is not None
                and tombstone.child_session_id == snapshot.child_session_id
                and tombstone.parent_identity == snapshot.parent_identity
            )
            return (
                authorized
                and self._transcript_link_identity(
                    self._transcript_link_for(root, snapshot.child_session_id)
                )
                == snapshot.link_identity
            )

    def references_child(self, session_id: str) -> bool:
        return (
            session_id in self._children
            or session_id in self._pending_child_closes
            or session_id in self._detached_child_ids
            or self._resolve_child_link(session_id) is not None
        )

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
        parent_runtime = self._root
        if parent_runtime is None:
            return None
        root_agent_loop = parent_runtime.agent_loop
        metadata = root_agent_loop.session_logger.session_metadata
        parent_dir = root_agent_loop.session_logger.session_dir
        if metadata is None or parent_dir is None:
            return None
        parent_root = parent_dir.resolve()
        link = next(
            (
                link
                for link in metadata.child_sessions
                if link.session_id == session_id and link.relative_path is not None
            ),
            None,
        )
        if link is None or link.relative_path is None:
            return None
        child_dir = (parent_dir / link.relative_path).resolve()
        if not child_dir.is_relative_to(parent_root) or not child_dir.is_dir():
            return None
        return parent_runtime, root_agent_loop, child_dir, link

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
        self._require_policy_unreserved()
        self._creating_children += 1
        try:
            yield
        finally:
            self._creating_children -= 1

    def _restore_agent_id_high_water_mark(self, root: SessionRuntime) -> None:
        metadata = root.agent_loop.session_logger.session_metadata
        config = metadata.config if metadata is not None else None
        high_water_mark = (
            config.get(_AGENT_ID_HIGH_WATER_MARK_KEY) if config is not None else None
        )
        if isinstance(high_water_mark, int) and not isinstance(high_water_mark, bool):
            self._next_agent_seq = max(self._next_agent_seq, high_water_mark)

    async def _issue_agent_id(self, parent: SessionRuntime) -> str:
        self._next_agent_seq += 1
        reserved_seq = self._next_agent_seq
        metadata = parent.agent_loop.session_logger.session_metadata
        if metadata is not None:
            config = dict(metadata.config or {})
            config[_AGENT_ID_HIGH_WATER_MARK_KEY] = reserved_seq
            metadata.config = config
            await parent.agent_loop.session_logger._persist_metadata_field(
                "config", config
            )
        return f"agent-{reserved_seq}"

    @staticmethod
    def _new_run_id(agent_id: str) -> str:
        return f"{agent_id}-run-{uuid4().hex[:12]}"

    def _validate_background_admission(
        self, parent: SessionRuntime, identity: tuple[str, int]
    ) -> None:
        live_identity = (
            parent.agent_loop.session_id,
            parent.agent_loop._session_generation,
        )
        admission_blocked = self._admission_closed or self._draining_children
        identity_changed = (
            self._root is not parent
            or self._generation_identity != identity
            or live_identity != identity
        )
        if admission_blocked or identity_changed or parent._closed is True:
            message = (
                "Agent admission is closed"
                if self._admission_closed
                else "Agent admission changed during child creation"
            )
            raise RuntimeError(message)

    def _resolve_launch_candidate(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        *,
        runtime: SessionRuntime | None = None,
        profile: str | None = None,
    ) -> LaunchCandidate:
        child = runtime.agent_loop if runtime is not None else None
        retained_profile = (
            parent.agent_loop.agent_manager.get_agent(profile)
            if profile is not None
            else None
        )
        return resolve_launch(
            profile_name=(
                args.agent_type if "agent_type" in args.model_fields_set else profile
            ),
            config=args.config,
            parent_orchestrator=(
                child.config_orchestrator
                if child is not None
                else parent.agent_loop.config_orchestrator
            ),
            tool_inventory={
                name: object()
                for name in parent.agent_loop.tool_manager.registered_tools
            },
            authorized_tool_names=frozenset(
                parent.agent_loop.tool_manager.available_tools
            ),
            history=tuple(child.messages) if child is not None else (),
            agent_manager=parent.agent_loop.agent_manager,
            retained_profile=retained_profile,
            accumulated_overrides=(
                child.launch_overrides if child is not None else None
            ),
            committed_model=(
                child.committed_model
                if child is not None
                else parent.agent_loop.committed_model
            ),
            frozen_persona=(
                FrozenPersona(
                    system_prompt_id=child.frozen_system_prompt_id,
                    instructions=child.frozen_instructions,
                )
                if child is not None and child.frozen_system_prompt_id is not None
                else None
            ),
        )

    async def _create_registered_child(
        self, parent: SessionRuntime, args: TaskArgs, ctx: InvokeContext
    ) -> tuple[SessionRuntime, AgentProfile]:
        candidate = self._resolve_launch_candidate(parent, args)
        generation_identity = (
            parent.agent_loop.session_id,
            parent.agent_loop._session_generation,
        )
        self._validate_background_admission(parent, generation_identity)
        with self._child_creation():
            child = await self._runtime_factory.create_child(
                parent.agent_loop, candidate
            )
        try:
            self._validate_background_admission(parent, generation_identity)
        except BaseException:
            await self._discard_child(child)
            raise
        runtime = self._build_child_runtime(child)
        async with self._child_publication(runtime):
            try:
                self._validate_background_admission(parent, generation_identity)
            except BaseException:
                await self._discard_child(child)
                raise
            self._children[child.session_id] = runtime
            self._child_links[child.session_id] = (parent, ctx.tool_call_id)
            link_recorded = projection_started = False
            try:
                await child.persist_empty_session()
                self._validate_background_admission(parent, generation_identity)
                await parent.agent_loop.record_child_session(
                    child, ctx.tool_call_id, candidate.profile.name
                )
                link_recorded = True
                self._validate_background_admission(parent, generation_identity)
                projection_started = True
                if parent.turns._projector is not None:
                    await parent.turns.link_subagent(ctx.tool_call_id, child.session_id)
                self._validate_background_admission(parent, generation_identity)
            except BaseException:
                self._children.pop(child.session_id, None)
                self._child_links.pop(child.session_id, None)
                if projection_started:
                    with suppress(Exception):
                        await parent.turns.unlink_subagent(
                            ctx.tool_call_id, child.session_id
                        )
                if link_recorded:
                    with suppress(Exception):
                        await parent.agent_loop.forget_child_session(
                            child.session_id, ctx.tool_call_id
                        )
                with suppress(Exception):
                    await runtime.close()
                with suppress(Exception):
                    await delete_saved_session(
                        child.session_id, child.config.session_logging
                    )
                raise
            return runtime, candidate.profile

    @staticmethod
    def _start_child_turn(
        runtime: SessionRuntime, args: TaskArgs, ctx: InvokeContext, session_id: str
    ) -> tuple[str, TurnStartAction]:
        response, start = runtime.turns.start(
            TurnStartParams(
                session_id=session_id,
                message=[
                    TextContentBlock(text=prepare_subagent_prompt(args.task, ctx))
                ],
            )
        )
        return response.turn.id, start

    async def _rollback_created_child(
        self, parent: SessionRuntime, runtime: SessionRuntime, tool_call_id: str
    ) -> None:
        """Remove every durable/runtime trace of a child rejected before acceptance."""
        session_id = runtime.agent_loop.session_id
        self._children.pop(session_id, None)
        self._child_links.pop(session_id, None)
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

    async def _rollback_reused_claim(
        self,
        record: AgentRecord,
        previous_idle_since: float | None,
        *,
        apply_policy: bool = False,
    ) -> None:
        async with self._registry_lock:
            if (
                self._agent_records.get(record.agent_id) is not record
                or record.state is not _AgentState.RUNNING
                or record.reserved_for_replacement
                or record.runtime._closed is True
                or self._children.get(record.session_id) is not record.runtime
            ):
                return
            record.state = _AgentState.IDLE
            record.idle_since = (
                previous_idle_since
                if previous_idle_since is not None
                else self._clock()
            )
            # Re-apply the committed policy only for a busy-replacement unwind:
            # a failed admission that followed a stopped run closes an unretained
            # instance instead of leaving it resident. A plain reuse failure
            # restores the prior idle residency. No-op for retained records.
            if apply_policy:
                closed = self._close_unretained_locked(
                    record,
                    record.run_history[-1] if record.run_history else None,
                    name=f"vibe-subagent-replacement-unwind:{record.agent_id}",
                )
                if closed:
                    # Publish the terminal transition: the client inventory
                    # must reflect the closure.
                    self._schedule_agents_update_locked()
            self._rearm_reaper_locked()

    def _require_task_profile_allowed(
        self, parent: SessionRuntime, profile: str
    ) -> None:
        """Apply effective Task admission before constructing child resources."""
        manager = parent.agent_loop.tool_manager
        if isinstance(manager, ToolManager) and "task" not in manager.available_tools:
            raise ToolPermissionError("Task tool is disabled")
        config = manager.get_tool_config("task")
        if (
            isinstance(manager, ToolManager)
            and config.permission == ToolPermission.NEVER
        ):
            raise ToolPermissionError("Task tool permission is never")
        if any(fnmatch.fnmatch(profile, pattern) for pattern in config.denylist):
            raise ToolPermissionError(f"Task denied for agent profile: {profile}")

    @staticmethod
    def _replacement_refusal(record: AgentRecord, outcome: LaunchOutcome) -> TaskResult:
        run = record.current_run
        return TaskResult(
            response=f"Replacement not launched: {outcome.value}",
            turns_used=0,
            completed=False,
            launch_outcome=outcome,
            agent_id=record.agent_id,
            run_id=run.run_id if run is not None else record.latest_run_id,
            stop_reason=run.requested_stop_reason
            if run is not None
            else record.stop_reason,
        )

    @staticmethod
    def _tombstone_reuse_error(
        agent_id: str, tombstone: _AgentTombstone
    ) -> AgentClosedError:
        """Select the typed reuse error by tombstone reason (§3.1 matrix).

        A ``closed`` tombstone rejects reuse with :class:`AgentClosedError` and
        continuation guidance; an ``evicted`` tombstone rejects it with the
        :class:`AgentEvictedError` subtype. Both admission sites look the
        tombstone up through :meth:`_live_tombstone_locked`, so a tombstone
        whose metadata deadline passed (before the reaper prunes it) behaves
        as unknown for reuse, per the §3.1 matrix.
        """
        if tombstone.reason == "closed":
            return AgentClosedError(
                "This instance is closed; launch a new task without "
                "agent_id. Use retain=true when a resident continuation "
                "is required."
            )
        return AgentEvictedError(f"Agent evicted: {agent_id}; start a new agent")

    async def _reserve_replacement(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> _ReplacementClaim | TaskResult | None:
        if args.agent_id is None or not args.background:
            raise ValueError("replace_run requires agent_id and background mode")
        if ctx.is_subagent:
            raise RuntimeError("Agent depth limit of 1 reached")
        parent = self._runtime(ctx.session_id)
        identity = (parent.agent_loop.session_id, parent.agent_loop._session_generation)
        async with self._registry_lock:
            self._validate_background_admission(parent, identity)
            record = self._agent_records.get(args.agent_id)
            if record is None:
                tombstone = self._live_tombstone_locked(args.agent_id, self._clock())
                if tombstone is not None:
                    raise self._tombstone_reuse_error(args.agent_id, tombstone)
                raise UnknownAgentError(
                    f"Unknown agent_id instance handle: {args.agent_id}. Check the agent_id with check_agents"
                )
            if ctx.session_id != record.parent_session_id:
                raise ToolPermissionError(
                    "Only the owning parent may replace this agent's run"
                )
            if (
                "agent_type" in args.model_fields_set
                and record.profile != args.agent_type
            ):
                raise AgentProfileMismatchError(
                    f"Agent profile mismatch: expected {record.profile}, got {args.agent_type}"
                )
            self._require_task_profile_allowed(parent, record.profile)
            if record.reserved_for_replacement:
                return self._replacement_refusal(
                    record, LaunchOutcome.REJECTED_RESERVATION
                )
            if record.state is _AgentState.IDLE:
                return None
            run = record.current_run
            if (
                record.state is _AgentState.FINALIZING
                or run is None
                or run.status is not RunStatus.RUNNING
                or run.initial_turn_id is None
                or record.runtime.turns.operation_pending_turn_id(run.initial_turn_id)
                is None
            ):
                return self._replacement_refusal(
                    record, LaunchOutcome.ALREADY_FINISHING
                )
            if run.requested_stop_reason is not None:
                return self._replacement_refusal(record, LaunchOutcome.ALREADY_STOPPING)
            if args.config is not None:
                raise LaunchConfigError(
                    "Busy replacement cannot change launch configuration"
                )
            if args.retain is not None and args.retain is not record.retained:
                raise LaunchConfigError(
                    "Busy replacement cannot change retention policy"
                )
            if (
                record.runtime._closed is True
                or self._children.get(record.session_id) is not record.runtime
                or not self._generation_is_current(record.root_generation)
                or record.work_slot not in self._active_work_slots
            ):
                raise RuntimeError("Agent is no longer eligible for replacement")
            assert record.work_slot is not None
            claim = _ReplacementClaim(
                record,
                run,
                parent,
                identity,
                record.runtime.agent_loop._session_generation,
                parent.agent_loop._authority_revision,
                record.work_slot,
            )
            run.requested_stop_reason = RunStopReason.RETASKED
            record.reserved_for_replacement = True
            record.replacement_owner = claim
            return claim

    def _validate_replacement_locked(self, claim: _ReplacementClaim) -> None:
        record = claim.record
        self._validate_background_admission(claim.parent, claim.parent_identity)
        resident = (
            self._agent_records.get(record.agent_id) is record
            and self._children.get(record.session_id) is record.runtime
            and record.runtime._closed is not True
        )
        authorized = (
            record.runtime.agent_loop._session_generation == claim.child_generation
            and claim.parent.agent_loop._authority_revision == claim.authority_revision
            and record.parent_session_id == claim.parent_identity[0]
            and self._generation_is_current(record.root_generation)
        )
        if (
            not resident
            or not authorized
            or not record.reserved_for_replacement
            or record.replacement_owner is not claim
            or claim.slot not in self._active_work_slots
        ):
            raise RuntimeError("Agent replacement admission changed during handoff")
        self._require_task_profile_allowed(claim.parent, record.profile)

    def _clear_replacement_locked(self, record: AgentRecord) -> None:
        record.reserved_for_replacement = False
        record.replacement_owner = None
        if record.current_run is None:
            if record.work_slot is not None:
                self._active_work_slots.discard(record.work_slot)
            if self._agent_records.get(record.agent_id) is record:
                latest = record.run_history[-1] if record.run_history else None
                if latest is not None and not latest.completion_task.done():
                    record.state = _AgentState.FINALIZING
                    record.idle_since = self._clock()
                    task = asyncio.create_task(
                        self._finish_replacement_unwind(record, latest),
                        name=f"vibe-subagent-replacement-unwind:{record.agent_id}",
                    )
                    self._teardown_tasks.add(task)
                    task.add_done_callback(self._teardown_tasks.discard)
                # Re-apply the committed policy: an unretained instance whose run
                # settled closes instead of returning to idle.
                elif self._close_unretained_locked(
                    record,
                    latest,
                    name=f"vibe-subagent-replacement-unwind:{record.agent_id}",
                ):
                    self._schedule_agents_update_locked()
                else:
                    record.state = _AgentState.IDLE
                    record.idle_since = self._clock()
        self._rearm_reaper_locked()

    async def _finish_replacement_unwind(
        self, record: AgentRecord, run: RunRecord
    ) -> None:
        # Cancellation may arrive after the monitor's idle/eviction decision,
        # while its final agents update is still awaiting delivery.
        with suppress(asyncio.CancelledError):
            await asyncio.shield(run.completion_task)
        async with self._registry_lock:
            resident = (
                self._agent_records.get(record.agent_id) is record
                and self._children.get(record.session_id) is record.runtime
                and self._generation_is_current(record.root_generation)
            )
            if (
                resident
                and record.latest_run_id == run.run_id
                and record.current_run is None
                and record.state is _AgentState.FINALIZING
                and not record.reserved_for_replacement
            ):
                # Re-apply the committed policy after the awaited run settles.
                if self._close_unretained_locked(
                    record,
                    run,
                    name=f"vibe-subagent-replacement-unwind:{record.agent_id}",
                ):
                    self._schedule_agents_update_locked()
                else:
                    record.state = _AgentState.IDLE
                    record.idle_since = self._clock()
                self._rearm_reaper_locked()

    @asynccontextmanager
    async def _replacement_handoff(
        self, claim: _ReplacementClaim
    ) -> AsyncIterator[None]:
        try:
            self._interrupt_run(claim.record, claim.run)
            await asyncio.shield(claim.run.completion_task)
            claim.slot.transferred = False
            yield
        finally:
            # Retry lock acquisition on every cancellation: unwinding must finish
            # before caller cancellation propagates, even under repeated interrupts.
            cancelled = False
            while True:
                try:
                    async with self._registry_lock:
                        owner = claim.record.replacement_owner
                        if owner is claim and claim.record.reserved_for_replacement:
                            self._clear_replacement_locked(claim.record)
                        elif owner is claim:
                            claim.record.replacement_owner = None
                        if (
                            owner is None or owner is claim
                        ) and not claim.slot.transferred:
                            self._active_work_slots.discard(claim.slot)
                    break
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError

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
        if args.background:  # noqa: PLR1702
            self._validate_background_admission(parent, parent_identity)
            retained_candidate: LaunchCandidate | None = None
            previous_idle_since: float | None = None
            reserved_child_generation: int | None = None
            reserved_parent_authority_revision: int | None = None
            if args.agent_id is not None:
                async with self._registry_lock:
                    self._validate_background_admission(parent, parent_identity)
                    record = self._agent_records.get(args.agent_id)
                    if record is None:
                        tombstone = self._live_tombstone_locked(
                            args.agent_id, self._clock()
                        )
                        if tombstone is not None:
                            raise self._tombstone_reuse_error(args.agent_id, tombstone)
                        raise UnknownAgentError(
                            f"Unknown agent_id instance handle: {args.agent_id}. Check the agent_id with check_agents"
                        )
                    if (
                        "agent_type" in args.model_fields_set
                        and record.profile != args.agent_type
                    ):
                        raise AgentProfileMismatchError(
                            f"Agent profile mismatch: expected {record.profile}, got {args.agent_type}"
                        )
                    if replacement is not None:
                        self._validate_replacement_locked(replacement)
                        if record is not replacement.record:
                            raise RuntimeError("Agent replacement residency changed")
                    elif (
                        record.reserved_for_replacement
                        or record.state is not _AgentState.IDLE
                    ):
                        raise AgentBusyError("Agent is already running a task")
                    if (
                        record.runtime._closed is True
                        or self._children.get(record.session_id) is not record.runtime
                    ):
                        raise UnknownAgentError(
                            f"Unknown agent_id instance handle: {args.agent_id}. Check the agent_id with check_agents"
                        )
                    self._require_task_profile_allowed(parent, record.profile)
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
                    self._rearm_reaper_locked()
            else:
                record = None
            created_record = False
            launch_profile: AgentProfile | None = None
            effective_task = summary_task if summary_task is not None else args.task
            prepared_reconfiguration = None
            if record is not None:
                if args.config is not None or isinstance(
                    record.runtime.agent_loop.launch_overrides, LaunchConfig
                ):
                    try:
                        retained_candidate = self._resolve_launch_candidate(
                            parent, args, runtime=record.runtime, profile=record.profile
                        )
                    except BaseException:
                        await self._rollback_reused_claim(
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
                        await self._rollback_reused_claim(
                            record,
                            previous_idle_since,
                            apply_policy=replacement is not None,
                        )
                        raise
                if parent.turns._projector is not None:
                    try:
                        await parent.turns.link_subagent(
                            ctx.tool_call_id, record.session_id
                        )
                        async with self._registry_lock:
                            self._validate_background_admission(parent, parent_identity)
                            if (
                                self._agent_records.get(record.agent_id) is not record
                                or self._children.get(record.session_id)
                                is not record.runtime
                                or record.runtime._closed is True
                            ):
                                raise RuntimeError(
                                    "Agent admission changed during child reuse"
                                )
                    except BaseException:
                        if prepared_reconfiguration is not None:
                            await (
                                record.runtime.agent_loop.abort_launch_reconfiguration(
                                    prepared_reconfiguration
                                )
                            )
                        await self._rollback_reused_claim(
                            record,
                            previous_idle_since,
                            apply_policy=replacement is not None,
                        )
                        raise
                runtime = record.runtime
            else:
                self._require_task_profile_allowed(parent, args.agent_type)
                runtime, profile = await self._create_registered_child(
                    parent, args, ctx
                )
                launch_profile = profile
                agent_id = await self._issue_agent_id(parent)
                record = AgentRecord(
                    agent_id=agent_id,
                    profile=runtime.agent_loop.launch_profile or args.agent_type,
                    session_id=runtime.agent_loop.session_id,
                    runtime=runtime,
                    root_generation=parent.agent_loop._session_generation,
                    idle_ttl_seconds=profile.idle_ttl_seconds,
                    parent_session_id=parent.agent_loop.session_id,
                    initial_task_summary=normalize_task_summary(
                        args.task_summary, fallback=effective_task
                    ),
                    state=_AgentState.RUNNING,
                )
                try:
                    async with self._registry_lock:
                        self._validate_background_admission(parent, parent_identity)
                        if self._children.get(record.session_id) is not runtime:
                            raise RuntimeError(
                                "Agent admission changed during child creation"
                            )
                        self._agent_records[agent_id] = record
                except BaseException:
                    await self._rollback_created_child(
                        parent, runtime, ctx.tool_call_id
                    )
                    raise
                created_record = True

            record.work_slot = slot
            agent_id = record.agent_id
            run_id = self._new_run_id(agent_id)
            prior_current_run = record.current_run
            prior_retained = record.retained
            prior_latest_run_id = record.latest_run_id
            prior_registry_latest_present = agent_id in self._latest_run_ids
            prior_registry_latest = self._latest_run_ids.get(agent_id)
            prior_idle_since = record.idle_since
            previous_event_sink = runtime.turns._event_sink
            committed = False
            backend_publication = None
            action: TurnStartAction | None = None
            publication_token = object()
            accumulator = SubagentRunAccumulator()
            completion_placeholder = asyncio.get_running_loop().create_future()
            run_record = RunRecord(
                run_id=run_id,
                agent_id=agent_id,
                profile=record.profile,
                status=RunStatus.RUNNING,
                completion_task=completion_placeholder,
                task_summary=normalize_task_summary(
                    args.task_summary, fallback=effective_task
                ),
            )
            record.current_run = run_record
            record.latest_run_id = run_id
            self._latest_run_ids[agent_id] = run_id
            # Commit the retention policy for this admission. A fresh launch
            # resolves explicit task arg -> profile setting -> generation
            # snapshot default; a reuse inherits the committed policy unless
            # the caller explicitly overrides it (retain=False is a final
            # continuation that closes after this run).
            if created_record:
                assert launch_profile is not None
                # Stable first-admission timestamp for deterministic Active
                # ordering; preserved across reuse and run-history pruning.
                record.first_admitted_at = self._clock()
                if args.retain is not None:
                    record.retained = args.retain
                elif launch_profile.retain is not None:
                    record.retained = launch_profile.retain
                else:
                    record.retained = self._retention_policy.retain_agents
                # Capture the profile display label only when the profile's
                # display_name was explicitly provided (design §5): a TOML
                # profile without display_name synthesizes a title-case
                # picker label, which must not become the name segment —
                # the derivation helper falls back to the profile name.
                record.profile_display_label = (
                    launch_profile.display_name
                    if launch_profile.display_name_explicit
                    else None
                )
            elif args.retain is not None:
                record.retained = args.retain
            run_record.start_count = sum(
                message.role is Role.assistant
                for message in runtime.agent_loop.messages
            )
            completion_metadata_mark = runtime.agent_loop.completion_metadata_mark()

            async def consume_event(event: BaseEvent) -> None:
                update = accumulator.observe(event, tool_call_id=ctx.tool_call_id)
                if update is None:
                    return
                if len(run_record.progress_summaries) < _BACKGROUND_PROGRESS_LIMIT:
                    run_record.progress_summaries.append(update.message)
                else:
                    run_record.progress_overflow = True

            try:
                runtime.turns._event_sink = consume_event
                turn_id, action = SessionRuntimeRegistry._start_child_turn(
                    runtime, args, ctx, runtime.agent_loop.session_id
                )
                run_record.initial_turn_id = turn_id

                async def monitor() -> None:  # noqa: PLR0912, PLR0915
                    try:
                        turn = await runtime.turns.wait_for_operation(turn_id)
                        if turn.error is not None:
                            accumulator.record_error(turn.error.message)
                        end_count = sum(
                            message.role is Role.assistant
                            for message in runtime.agent_loop.messages
                        )
                        run_record.result = accumulator.build_result(
                            turns_used=end_count - run_record.start_count,
                            completed=turn.status is PublicTurnStatus.COMPLETED,
                        ).model_copy(
                            update={
                                "agent_id": agent_id,
                                "run_id": run_id,
                                "metadata": runtime.agent_loop.completion_metadata_since(
                                    completion_metadata_mark
                                ),
                            }
                        )
                        run_record.status, run_record.stop_reason = (
                            self._resolve_run_outcome(run_record, turn)
                        )
                    except asyncio.CancelledError:
                        run_record.status, run_record.stop_reason = (
                            self._resolve_run_outcome(
                                run_record,
                                runtime.turns.operation_terminal_turn(turn_id),
                            )
                        )
                        run_record.result = accumulator.build_result(
                            turns_used=sum(
                                message.role is Role.assistant
                                for message in runtime.agent_loop.messages
                            )
                            - run_record.start_count,
                            completed=run_record.status is RunStatus.COMPLETED,
                        ).model_copy(
                            update={
                                "agent_id": agent_id,
                                "run_id": run_id,
                                "metadata": runtime.agent_loop.completion_metadata_since(
                                    completion_metadata_mark
                                ),
                            }
                        )
                        raise
                    except Exception as exc:
                        accumulator.record_error(str(exc))
                        run_record.status = RunStatus.FAILED
                        run_record.stop_reason = RunStopReason.ERROR
                        run_record.result = accumulator.build_result(
                            turns_used=0, completed=False
                        ).model_copy(
                            update={
                                "agent_id": agent_id,
                                "run_id": run_id,
                                "metadata": runtime.agent_loop.completion_metadata_since(
                                    completion_metadata_mark
                                ),
                            }
                        )
                    finally:
                        if run_record.result is not None:
                            run_record.result = run_record.result.model_copy(
                                update={"stop_reason": run_record.stop_reason}
                            )
                        active_task = runtime.turns._active_task
                        if record.reserved_for_replacement:
                            # The old execution is joined, but its allocation now
                            # belongs to the handoff, including notification awaits.
                            pass
                        elif active_task is not None and not active_task.done():
                            # Release/drain may cancel the monitor before provider
                            # cancellation finishes; keep counting the actual work.
                            active_task.add_done_callback(
                                lambda _: self._active_work_slots.discard(slot)
                            )
                        else:
                            self._active_work_slots.discard(slot)
                        runtime.turns._event_sink = previous_event_sink
                        if not committed:
                            return  # noqa: B012
                        run_record.completed_at = self._clock()
                        committed_model = runtime.agent_loop.committed_model
                        if committed_model is not None:
                            run_record.terminal_identity = (
                                committed_model.base_model,
                                committed_model.provider,
                                committed_model.wire_name,
                            )
                        key = (agent_id, run_id)
                        async with self._registry_lock:
                            resident = (
                                self._agent_records.get(agent_id) is record
                                and self._children.get(record.session_id) is runtime
                                and record.current_run is run_record
                                and self._generation_is_current(record.root_generation)
                            )
                            suppress_notification = (
                                not resident
                                or self._draining_children
                                or agent_id in self._suppressed_notifications
                            )
                            if resident:
                                record.compacting = False
                                record.compaction_entry_id = None
                                record.stop_reason = run_record.stop_reason
                                committed_model = runtime.agent_loop.committed_model
                                record.base_model = (
                                    committed_model.base_model
                                    if committed_model is not None
                                    else None
                                )
                                if committed_model is not None:
                                    record.effective_model = format_model_display_name(
                                        committed_model.provider,
                                        committed_model.wire_name,
                                    )
                                record.active_provider = (
                                    committed_model.provider
                                    if committed_model is not None
                                    else None
                                )
                                record.state = (
                                    _AgentState.RUNNING
                                    if record.reserved_for_replacement
                                    else _AgentState.FINALIZING
                                )
                                record.idle_since = run_record.completed_at
                                record.last_task_summary = run_record.task_summary
                                record.last_run_status = run_record.status
                                record.current_run = None
                                if run_record not in record.run_history:
                                    record.run_history.append(run_record)
                                    if len(record.run_history) > _MAX_RUN_HISTORY:
                                        record.run_history = record.run_history[
                                            -_MAX_RUN_HISTORY:
                                        ]
                            publication_permitted = (
                                self._result_write_tokens.get(key) is publication_token
                            )
                            if publication_permitted and run_record.result is not None:
                                self._result_store[key] = StoredRunResult(
                                    agent_id=agent_id,
                                    run_id=run_id,
                                    result=run_record.result,
                                    completed_at=run_record.completed_at,
                                    root_generation=record.root_generation,
                                    terminal_identity=run_record.terminal_identity,
                                )
                            if publication_permitted and not suppress_notification:
                                self._pending_notifications.add(key)
                            if publication_permitted:
                                self._result_write_tokens.pop(key, None)
                            self._expire_results_locked()
                        if suppress_notification or not publication_permitted:
                            return  # noqa: B012
                        try:
                            await self._emit_agents_update()
                        except Exception as exc:
                            logger.warning(
                                "Failed to publish background agent update",
                                exc_info=exc,
                            )
                        try:
                            async with self._registry_lock:
                                notification_permitted = (
                                    self._generation_is_current(record.root_generation)
                                    and self._agent_records.get(agent_id) is record
                                    and self._children.get(record.session_id) is runtime
                                    and not self._draining_children
                                    and agent_id not in self._suppressed_notifications
                                    and key in self._result_store
                                    and key in self._pending_notifications
                                )
                            if not notification_permitted:
                                return  # noqa: B012
                            # Notify the parent that a background run completed.
                            status_word = (
                                "budget-stopped (budget exceeded)"
                                if run_record.stop_reason
                                is RunStopReason.BUDGET_EXCEEDED
                                else "budget-stopped (budget unverifiable)"
                                if run_record.stop_reason
                                is RunStopReason.BUDGET_UNVERIFIABLE
                                else "completed"
                                if run_record.status is RunStatus.COMPLETED
                                else "failed"
                                if run_record.status is RunStatus.FAILED
                                else "cancelled (by user)"
                                if run_record.stop_reason
                                is RunStopReason.USER_CANCELLED
                                else "retasked"
                                if run_record.stop_reason is RunStopReason.RETASKED
                                else "cancelled"
                            )
                            # The notification derives from this run's own
                            # record state at completion (§5): name alongside
                            # handle, run, and outcome.
                            notification = (
                                f"Background agent {self._record_name(record)} "
                                f"({agent_id}, run {run_id})"
                                f" {status_word}."
                                " Use get_agent_result to retrieve the result."
                            )
                            if (
                                run_record.stop_reason
                                is not RunStopReason.USER_CANCELLED
                                and parent.turns.active_turn is None
                                and parent.turns._active_task is None
                                and not parent.turns.queue_state.items
                            ):
                                try:
                                    _response, _start = parent.turns.start(
                                        TurnStartParams(
                                            session_id=parent.agent_loop.session_id,
                                            message=[
                                                TextContentBlock(text=notification)
                                            ],
                                        )
                                    )
                                    _start()
                                except Exception:
                                    parent.agent_loop._pending_injected_messages.append(
                                        LLMMessage(
                                            role=Role.user,
                                            content=notification,
                                            injected=True,
                                        )
                                    )
                            else:
                                parent.agent_loop._pending_injected_messages.append(
                                    LLMMessage(
                                        role=Role.user,
                                        content=notification,
                                        injected=True,
                                    )
                                )
                        except Exception as exc:
                            logger.warning(
                                "Failed to finalize background agent notification",
                                exc_info=exc,
                            )
                        finally:
                            async with self._registry_lock:
                                self._pending_notifications.discard(key)
                                if (
                                    self._agent_records.get(agent_id) is record
                                    and self._children.get(record.session_id) is runtime
                                    and not record.reserved_for_replacement
                                    and record.state is _AgentState.FINALIZING
                                    and self._generation_is_current(
                                        record.root_generation
                                    )
                                ):
                                    # Completion transition: a retained instance
                                    # returns to idle; an unretained instance closes
                                    # (detach + `closed` tombstone + tracked runtime
                                    # close) after the result was published above.
                                    if not self._close_unretained_locked(
                                        record,
                                        run_record,
                                        name=(
                                            "vibe-subagent-close-on-completion:"
                                            f"{agent_id}"
                                        ),
                                    ):
                                        record.state = _AgentState.IDLE
                                self._expire_results_locked()
                                if not self._draining_children:
                                    self._rearm_reaper_locked()
                            if not self._draining_children:
                                try:
                                    await self._emit_agents_update()
                                except Exception as exc:
                                    logger.warning(
                                        "Failed to publish agent completion transition",
                                        exc_info=exc,
                                    )

                async with self._registry_lock:
                    self._validate_background_admission(parent, parent_identity)
                    if (
                        self._agent_records.get(agent_id) is not record
                        or self._children.get(record.session_id) is not runtime
                        or record.current_run is not run_record
                        or runtime._closed is True
                    ):
                        raise RuntimeError("Agent admission changed before launch")
                    if reserved_child_generation is not None and (
                        runtime.agent_loop._session_generation
                        != reserved_child_generation
                        or parent.agent_loop._authority_revision
                        != reserved_parent_authority_revision
                    ):
                        raise RuntimeError("Agent authority changed before launch")
                    self._require_task_profile_allowed(parent, record.profile)
                    self._result_write_tokens[(agent_id, run_id)] = publication_token
                monitor_coro = monitor()
                try:
                    completion = asyncio.create_task(
                        monitor_coro, name=f"vibe-subagent-background:{run_id}"
                    )
                except BaseException:
                    monitor_coro.close()
                    raise
                run_record.completion_task = completion

                def settle_prepared_waiters(_task: asyncio.Task[None]) -> None:
                    # A waiter can snapshot the prepared run before its monitor
                    # is installed. Keep that admission-time future linked to
                    # the same completion, including a prepared-start stop.
                    if not completion_placeholder.done():
                        completion_placeholder.set_result(None)

                completion.add_done_callback(settle_prepared_waiters)
                self._monitor_tasks.add(completion)
                completion.add_done_callback(self._monitor_tasks.discard)
                assert action is not None
                if prepared_reconfiguration is not None:
                    backend_publication = (
                        runtime.agent_loop.publish_launch_reconfiguration(
                            prepared_reconfiguration
                        )
                    )
                action()
                run_record.started_at = self._clock()
                record.stop_reason = None
                record.compacting = False
                record.compaction_entry_id = None
                record.context_window = (
                    runtime.agent_loop.config.get_active_model().auto_compact_threshold
                )
                if prepared_reconfiguration is not None:
                    # Acceptance makes the new envelope authoritative in memory
                    # before persistence can block on the logger save lock.
                    runtime.agent_loop.install_launch_metadata()
                if retained_candidate is not None:
                    record.effective_model = retained_candidate.effective_model.alias
                    record.effective_thinking = retained_candidate.effective_thinking
                else:
                    record.effective_model = runtime.agent_loop.config.active_model
                    record.effective_thinking = (
                        runtime.agent_loop.config.get_active_model().thinking
                    )
                committed_model = runtime.agent_loop.committed_model
                if committed_model is not None:
                    record.effective_model = format_model_display_name(
                        committed_model.provider, committed_model.wire_name
                    )
                record.base_model = (
                    committed_model.base_model if committed_model is not None else None
                )
                record.active_provider = (
                    committed_model.provider if committed_model is not None else None
                )
                committed = True
                slot.transferred = True
                if prepared_reconfiguration is not None:
                    assert backend_publication is not None
                    runtime.agent_loop.finalize_launch_reconfiguration(
                        prepared_reconfiguration, backend_publication
                    )
            except BaseException as exc:
                if action is not None and not committed:
                    action.abort()
                if prepared_reconfiguration is not None and not committed:
                    if backend_publication is not None:
                        runtime.agent_loop.rollback_launch_reconfiguration(
                            prepared_reconfiguration, backend_publication
                        )
                    else:
                        await runtime.agent_loop.abort_launch_reconfiguration(
                            prepared_reconfiguration
                        )
                runtime.turns._event_sink = previous_event_sink
                attempted = run_record.completion_task
                if not attempted.done():
                    attempted.cancel()
                with suppress(asyncio.CancelledError):
                    await attempted
                async with self._registry_lock:
                    if (
                        self._result_write_tokens.get((agent_id, run_id))
                        is publication_token
                    ):
                        self._result_write_tokens.pop((agent_id, run_id), None)
                    owns_attempt = (
                        self._agent_records.get(agent_id) is record
                        and record.current_run is run_record
                    )
                    if owns_attempt:
                        record.current_run = prior_current_run
                        record.retained = prior_retained
                        record.latest_run_id = prior_latest_run_id
                        record.idle_since = prior_idle_since
                        if self._latest_run_ids.get(agent_id) == run_id:
                            if prior_registry_latest_present:
                                assert prior_registry_latest is not None
                                self._latest_run_ids[agent_id] = prior_registry_latest
                            else:
                                self._latest_run_ids.pop(agent_id, None)
                if created_record:
                    async with self._registry_lock:
                        if self._agent_records.get(agent_id) is record:
                            self._agent_records.pop(agent_id)
                    await self._rollback_created_child(
                        parent, runtime, ctx.tool_call_id
                    )
                else:
                    await self._rollback_reused_claim(
                        record,
                        previous_idle_since,
                        apply_policy=replacement is not None,
                    )
                if isinstance(exc, asyncio.CancelledError):
                    raise exc
                raise
            if prepared_reconfiguration is not None:
                # Acceptance has happened: persistence failure is reported without
                # rolling the live configuration or accepted run back.
                await runtime.agent_loop.persist_launch_metadata()
            if not defer_launch_agents_update:
                try:
                    await self._emit_agents_update()
                except Exception as exc:
                    logger.warning(
                        "Failed to publish initial background agent update",
                        exc_info=exc,
                    )
            # An explicit status distinguishes this successful launch acknowledgment
            # from the terminal result that get_agent_result later returns. The
            # guidance is policy-aware: retained agents stay resident and can be
            # continued; unretained agents close on completion with the result
            # still retrievable.
            if record.retained:
                policy_guidance = (
                    " This agent is retained and stays resident after the run "
                    "completes; to continue this instance, use "
                    f'task(agent_id="{agent_id}", background=true, task="...").'
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
            agent_name = self._record_name(record)
            yield TaskResult(
                response=(
                    f"Background agent {agent_name} ({agent_id}, run {run_id}) "
                    "launched and running; use check_agents to monitor, "
                    "get_agent_result or wait_for_agent to retrieve the "
                    "result." + policy_guidance
                ),
                turns_used=0,
                completed=True,
                status="launched",
                launch_outcome=LaunchOutcome.LAUNCHED,
                agent_id=agent_id,
                run_id=run_id,
                name=agent_name,
                metadata={
                    "base_model": record.base_model,
                    "active_provider": record.active_provider,
                    "effective_model": record.effective_model,
                    **(
                        {
                            "replaced_run_id": replacement.run.run_id,
                            "replacement_run_id": run_id,
                        }
                        if replacement is not None
                        else {}
                    ),
                },
            )
            return

        if args.agent_id is not None:
            raise ValueError("agent_id is only supported for background tasks")
        if args.retain is not None:
            raise ValueError("retain is only supported for background tasks")
        self._require_task_profile_allowed(parent, args.agent_type)
        runtime, _idle_ttl_seconds = await self._create_registered_child(
            parent, args, ctx
        )
        child = runtime.agent_loop
        progress = BoundedEventQueue[ToolStreamEvent]()
        accumulator = SubagentRunAccumulator()
        completion_metadata_mark = child.completion_metadata_mark()

        async def consume_event(event: BaseEvent) -> None:
            update = accumulator.observe(event, tool_call_id=ctx.tool_call_id)
            if update is None:
                return
            try:
                progress.put_nowait(update)
            except asyncio.QueueFull:
                with suppress(asyncio.QueueEmpty):
                    progress.get_nowait()
                progress.put_nowait(update)

        previous_event_sink = runtime.turns._event_sink
        runtime.turns._event_sink = consume_event
        try:
            turn_id, start = SessionRuntimeRegistry._start_child_turn(
                runtime, args, ctx, child.session_id
            )
            start()
        except BaseException:
            runtime.turns._event_sink = previous_event_sink
            await self._rollback_created_child(parent, runtime, ctx.tool_call_id)
            raise
        completion = asyncio.create_task(
            runtime.turns.wait_for_operation(turn_id),
            name=f"vibe-subagent-turn:{child.session_id}",
        )
        slot.transferred = True
        completion.add_done_callback(lambda _: self._active_work_slots.discard(slot))
        result: TaskResult | None = None

        async def teardown_foreground() -> None:
            errors: list[BaseException] = []
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

        try:
            async for item in stream_until_complete(
                progress, completion, event_task_name="vibe-subagent-progress"
            ):
                yield item
            turn = await completion
            if turn.error is not None:
                accumulator.record_error(turn.error.message)
            turns_used = sum(
                message.role is Role.assistant for message in child.messages
            )
            result = accumulator.build_result(
                turns_used=turns_used,
                completed=turn.status is PublicTurnStatus.COMPLETED,
            ).model_copy(
                update={
                    "metadata": child.completion_metadata_since(
                        completion_metadata_mark
                    )
                }
            )
            if turn.error is not None:
                raise RuntimeError(turn.error.message)
        finally:
            runtime.turns._event_sink = previous_event_sink
            if self._children.get(child.session_id) is runtime:
                parent_runtime, tool_call_id = self._child_links[child.session_id]
                parent_loop = parent_runtime.agent_loop
                parent_dir = parent_loop.session_logger.session_dir
                child_dir = child.session_logger.session_dir
                if parent_dir is not None and child_dir is not None:
                    link = ChildSessionLink(
                        session_id=child.session_id,
                        tool_call_id=tool_call_id,
                        agent=child.launch_profile or args.agent_type,
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
                teardown_foreground(),
                name=f"vibe-subagent-foreground-teardown:{child.session_id}",
            )

        assert result is not None
        yield result

    def _expire_results_locked(self) -> None:
        unreferenced_by_generation: dict[int, list[StoredRunResult]] = {}
        for key, stored in self._result_store.items():
            if key not in self._pending_notifications and not self._wait_leases.get(
                key
            ):
                unreferenced_by_generation.setdefault(
                    stored.root_generation, []
                ).append(stored)
        for unreferenced in unreferenced_by_generation.values():
            excess = len(unreferenced) - _MAX_STORED_RESULTS
            if excess <= 0:
                continue
            for stored in sorted(
                unreferenced, key=lambda item: (item.completed_at, item.agent_id)
            )[:excess]:
                key = (stored.agent_id, stored.run_id)
                self._result_store.pop(key, None)
                self._expired_results.add(key)
                tombstone = self._evicted_agents.get(stored.agent_id)
                if (
                    tombstone is not None
                    and tombstone.summary.current_run_id == stored.run_id
                ):
                    tombstone.summary.result_expired = True
                record = self._agent_records.get(stored.agent_id)
                if record is not None:
                    for run in record.run_history:
                        if run.run_id == stored.run_id:
                            run.result = None

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
        """Derive the human-facing ``{profile}-{model}-{number}`` name (§5).

        The name is derived from the record's committed admission state
        (profile display label captured at admission, canonical base model
        with the effective-model fallback) and the handle's sequence digits.
        """
        return derive_agent_name(
            record.agent_id,
            record.profile,
            record.profile_display_label,
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
            summary = AgentSummary(
                agent_id=record.agent_id,
                profile=record.profile,
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
        active = [
            entry
            for entry in entries
            if entry[0].state in {_AgentState.RUNNING, _AgentState.FINALIZING}
        ]
        idle = [
            entry
            for entry in entries
            if entry[0].state not in {_AgentState.RUNNING, _AgentState.FINALIZING}
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
                    profile=summary.profile,
                    name=summary.name,
                    slot_purposes=(
                        project_launch_slot_purposes(
                            record.runtime.agent_loop, summary.profile
                        )
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
        """Non-destructive and operation-pinned; release owns fallback teardown."""
        if run.initial_turn_id is None:
            return False
        return record.runtime.turns.interrupt_operation(run.initial_turn_id)

    async def cancel_run(  # noqa: PLR0911
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        reason: RunStopReason,
        requester_session_id: str,
    ) -> CancelResult:
        async with self._registry_lock:
            # Only the server's user route supplies USER_CANCELLED. Tool stops
            # retain parent ownership; users may stop any run in this registry.
            user_cancel = reason is RunStopReason.USER_CANCELLED
            record = self._agent_records.get(agent_id)
            tombstone = self._evicted_agents.get(agent_id)
            if record is None and tombstone is None:
                if (
                    not user_cancel
                    and agent_id in self._children
                    and not any(
                        item.session_id == agent_id
                        for item in self._agent_records.values()
                    )
                ):
                    raise ValueError(
                        "Foreground agents cannot be stopped with cancel_agent; "
                        "only background runs are supported"
                    )
                return CancelResult(outcome=CancelOutcome.UNKNOWN_RUN, run_id=run_id)
            if self._admission_closed or self._draining_children:
                return CancelResult(outcome=CancelOutcome.FORBIDDEN, run_id=run_id)
            if record is None:
                assert tombstone is not None
                summary = tombstone.summary
                if (
                    not user_cancel
                    and requester_session_id != tombstone.parent_identity[0]
                ):
                    return CancelResult(outcome=CancelOutcome.FORBIDDEN, run_id=run_id)
                return CancelResult(
                    outcome=CancelOutcome.NOT_RUNNING,
                    run_id=summary.latest_run_id or summary.current_run_id,
                    stop_reason=summary.stop_reason,
                )
            if not user_cancel and requester_session_id != record.parent_session_id:
                return CancelResult(outcome=CancelOutcome.FORBIDDEN, run_id=run_id)
            if run_id is None:
                run_id = (
                    record.current_run.run_id
                    if record.current_run is not None
                    else record.latest_run_id
                )
            run = next(
                (
                    item
                    for item in [record.current_run, *record.run_history]
                    if item is not None and item.run_id == run_id
                ),
                None,
            )
            key = (agent_id, run_id)
            if run is None and run_id != record.latest_run_id:
                if key not in self._result_store and key not in self._expired_results:
                    return CancelResult(
                        outcome=CancelOutcome.UNKNOWN_RUN, run_id=run_id
                    )
            if (
                record.state is _AgentState.FINALIZING
                and run_id == record.latest_run_id
            ):
                return CancelResult(
                    outcome=CancelOutcome.ALREADY_FINISHING,
                    run_id=run_id,
                    stop_reason=record.stop_reason,
                )
            if (
                run is not None
                and record.current_run is run
                and record.state is _AgentState.RUNNING
                and (
                    run.initial_turn_id is None
                    or record.runtime.turns.operation_pending_turn_id(
                        run.initial_turn_id
                    )
                    is None
                )
            ):
                return CancelResult(
                    outcome=CancelOutcome.ALREADY_FINISHING,
                    run_id=run_id,
                    stop_reason=run.stop_reason,
                )
            if (
                run is None
                or record.current_run is not run
                or record.state is not _AgentState.RUNNING
                or run.status is not RunStatus.RUNNING
            ):
                return CancelResult(
                    outcome=CancelOutcome.NOT_RUNNING,
                    run_id=run_id,
                    stop_reason=run.stop_reason if run is not None else None,
                )
            if run.requested_stop_reason is not None:
                return CancelResult(
                    outcome=CancelOutcome.ALREADY_STOPPING,
                    run_id=run_id,
                    stop_reason=run.requested_stop_reason,
                )
            run.requested_stop_reason = reason
        # This helper is synchronous: no scheduling gap between arbitration and
        # interruption, no lock across cleanup awaits, and no monitor cancellation.
        if not self._interrupt_run(record, run):
            # No await separates the latch from delivery; a failed delivery did
            # not accept a stop and must not leave a misleading stopping latch.
            run.requested_stop_reason = None
            return CancelResult(
                outcome=CancelOutcome.ALREADY_FINISHING,
                run_id=run.run_id,
                stop_reason=run.stop_reason,
            )
        return CancelResult(
            outcome=CancelOutcome.STOP_REQUESTED,
            run_id=run.run_id,
            stop_reason=run.requested_stop_reason,
        )

    def _resolve_run(
        self, agent_id: str, run_id: str | None
    ) -> tuple[tuple[str, str], RunRecord | StoredRunResult] | None:
        record = self._agent_records.get(agent_id)
        if run_id is None:
            run_id = self._latest_run_ids.get(agent_id)
            if (
                run_id is None
                and record is not None
                and (record.latest_run_id is not None or record.current_run is not None)
            ):
                run_id = (
                    record.latest_run_id
                    if record.latest_run_id is not None
                    else cast(RunRecord, record.current_run).run_id
                )
            if run_id is None:
                candidates = [
                    stored
                    for stored in self._result_store.values()
                    if stored.agent_id == agent_id
                ]
                if candidates:
                    latest = max(candidates, key=lambda item: item.completed_at)
                    return (agent_id, latest.run_id), latest
                return None
        key = (agent_id, run_id)
        if key in self._expired_results:
            raise AgentResultExpiredError(
                f"Result expired for run {run_id} on agent {agent_id}"
            )
        if record is not None:
            if record.current_run is not None and record.current_run.run_id == run_id:
                return key, record.current_run
            past = next(
                (item for item in record.run_history if item.run_id == run_id), None
            )
            if past is not None and past.result is not None:
                return key, past
        stored = self._result_store.get(key)
        if stored is not None:
            return key, stored
        return None

    async def get_agent_result(
        self, agent_id: str, run_id: str | None = None
    ) -> TaskResult | None:
        async with self._registry_lock:
            self._expire_results_locked()
            resolved = self._resolve_run(agent_id, run_id)
        if resolved is None:
            raise UnknownAgentError(f"Unknown run {run_id} for agent {agent_id}")
        target = resolved[1]
        if isinstance(target, RunRecord):
            if target.status is RunStatus.RUNNING:
                return None
            if target.result is None:
                raise AgentResultExpiredError(
                    f"Result expired for run {target.run_id} on agent {agent_id}"
                )
            return target.result
        return target.result

    async def wait_for_agent(
        self, agent_id: str, run_id: str | None = None, *, timeout: float | None = None
    ) -> WaitForAgentResult:
        """Wait for a run and return the typed disposition (design §6).

        ``completed`` carries the finished run's TaskResult; ``still_running``
        is a normal outcome when the timeout expires first. The timeout never
        cancels the run (the completion task is shielded) and the wait lease
        keeps the result protected from count expiry. Race rule: every
        disposition reports the run the caller pinned at wait start, never a
        newer run's progress.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive or None")
        wait_started_at = self._clock()
        async with self._registry_lock:
            resolved = self._resolve_run(agent_id, run_id)
            if resolved is None:
                raise UnknownAgentError(f"Unknown run {run_id} for agent {agent_id}")
            key, target = resolved
            self._wait_leases[key] = self._wait_leases.get(key, 0) + 1
        try:
            if isinstance(target, RunRecord) and target.status is RunStatus.RUNNING:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(target.completion_task), timeout=timeout
                    )
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling():
                        raise
                    async with self._registry_lock:
                        released = key not in self._wait_leases
                    if not released:
                        raise
                except TimeoutError:
                    return await self._still_running_disposition(
                        agent_id,
                        key,
                        target,
                        run_id=run_id,
                        wait_started_at=wait_started_at,
                    )
            async with self._registry_lock:
                if key not in self._wait_leases:
                    resolved = self._resolve_run(agent_id, run_id)
                    if resolved is None:
                        raise UnknownAgentError(
                            f"Unknown run {run_id} for agent {agent_id}"
                        )
                    raise AgentResultExpiredError(
                        f"Result expired for run {key[1]} on agent {agent_id}"
                    )
                stored = self._result_store.get(key)
                result = stored.result if stored is not None else target.result
                if result is None:
                    raise RuntimeError(f"Run {key[1]} completed without a result")
                return WaitForAgentResult(status=WaitStatus.COMPLETED, result=result)
        finally:
            async with self._registry_lock:
                leases = self._wait_leases.get(key, 0)
                if leases > 1:
                    self._wait_leases[key] = leases - 1
                else:
                    self._wait_leases.pop(key, None)
                self._expire_results_locked()

    async def _still_running_disposition(
        self,
        agent_id: str,
        key: tuple[str, str],
        target: RunRecord,
        *,
        run_id: str | None,
        wait_started_at: float,
    ) -> WaitForAgentResult:
        """Build the ``still_running`` disposition for an expired wait (§6).

        The pinned run identity is preserved across timeout/release/replacement
        races: the disposition reports the run this wait targeted. When the
        pinned run actually settled at the timeout boundary, the completed
        disposition is returned instead. A wait fenced by release keeps the
        existing typed errors.
        """
        async with self._registry_lock:
            if key not in self._wait_leases:
                # Released while waiting: keep the release fencing semantics.
                resolved = self._resolve_run(agent_id, run_id)
                if resolved is None:
                    raise UnknownAgentError(
                        f"Unknown run {run_id} for agent {agent_id}"
                    )
                raise AgentResultExpiredError(
                    f"Result expired for run {key[1]} on agent {agent_id}"
                )
            stored = self._result_store.get(key)
            result = stored.result if stored is not None else target.result
            if result is not None:
                return WaitForAgentResult(status=WaitStatus.COMPLETED, result=result)
            if target.status is not RunStatus.RUNNING:
                raise RuntimeError(f"Run {key[1]} completed without a result")
            elapsed = max(0.0, self._clock() - wait_started_at)
            record = self._agent_records.get(agent_id)
            # Pinned-run progress: only the record's live current run has a
            # meaningful in-flight turn count; a displaced run reports zero.
            if record is not None and record.current_run is target:
                turns_used = self._agent_turns_used(record) or 0
            else:
                turns_used = 0
            return WaitForAgentResult(
                status=WaitStatus.STILL_RUNNING,
                agent_id=agent_id,
                run_id=key[1],
                elapsed_seconds=elapsed,
                run_status=target.status,
                turns_used=turns_used,
                guidance=still_running_guidance(
                    agent_id=agent_id,
                    run_id=key[1],
                    elapsed_seconds=elapsed,
                    run_status=target.status,
                    turns_used=turns_used,
                ),
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
        """Construct terminal metadata shared by close-on-completion and eviction.

        The summary freezes the record/run fields at insertion time.
        ``expires_at`` is ``closed_at + tombstone_ttl_seconds``. Returns
        ``None`` when the tombstone TTL is zero: no terminal metadata is
        retained at all, and the handle behaves as metadata-expired/unknown
        once the record detaches (the result stays published and
        retrievable).
        """
        agent_id = record.agent_id
        latest_run_id = record.latest_run_id or (
            run.run_id if run is not None else None
        )
        now = self._clock()
        ttl = self._retention_policy.tombstone_ttl_seconds
        if ttl <= 0:
            return None
        summary = AgentSummary(
            agent_id=agent_id,
            profile=record.profile,
            availability=availability,
            current_run_id=run.run_id if run is not None else None,
            current_run_status=run.status if run is not None else None,
            turns_used=self._agent_turns_used(record),
            last_run_status=record.last_run_status,
            initial_task_summary=record.initial_task_summary,
            current_task_summary=run.task_summary if run is not None else None,
            idle_seconds=idle_seconds,
            latest_run_id=latest_run_id,
            run_elapsed_seconds=self._run_elapsed_seconds(run, now),
            effective_model=record.effective_model,
            base_model=record.base_model,
            active_provider=record.active_provider,
            effective_thinking=record.effective_thinking,
            context_tokens=record.context_tokens,
            context_window=record.context_window,
            compacting=record.compacting,
            stop_reason=record.stop_reason,
            result_expired=(
                (agent_id, latest_run_id) in self._expired_results
                if latest_run_id is not None
                else False
            ),
            name=self._record_name(record),
        )
        return _AgentTombstone(
            summary=summary,
            child_session_id=record.session_id,
            parent_identity=(record.parent_session_id, record.root_generation),
            reason="closed" if availability is AgentAvailability.CLOSED else "evicted",
            closed_at=now,
            expires_at=now + ttl,
            profile_display_label=record.profile_display_label,
        )

    def _insert_tombstone_locked(
        self, agent_id: str, tombstone: _AgentTombstone
    ) -> None:
        """Store terminal metadata and enforce the oldest-first count cap."""
        self._evicted_agents[agent_id] = tombstone
        cap = self._retention_policy.max_tombstones
        if cap <= 0:
            return
        while len(self._evicted_agents) > cap:
            oldest_id = min(
                self._evicted_agents,
                key=lambda aid: (
                    self._evicted_agents[aid].closed_at,
                    self._agent_id_sort_key(aid),
                ),
            )
            self._evicted_agents.pop(oldest_id, None)
            # The capped tombstone's pending-eviction authority settles with
            # it: the eviction it authorized can no longer notify once its
            # metadata is gone, so the entry must not outlive the tombstone.
            self._pending_evictions.pop(oldest_id, None)

    @staticmethod
    def _tombstone_expired(tombstone: _AgentTombstone, now: float) -> bool:
        return tombstone.expires_at is not None and now >= tombstone.expires_at

    def _live_tombstone_locked(
        self, agent_id: str, now: float
    ) -> _AgentTombstone | None:
        """Return the tombstone for ``agent_id`` unless its metadata has expired."""
        tombstone = self._evicted_agents.get(agent_id)
        if tombstone is not None and not self._tombstone_expired(tombstone, now):
            return tombstone
        return None

    def _prune_expired_tombstones_locked(self, now: float) -> list[str]:
        """Drop expired terminal metadata only (results, markers, leases survive).

        Returns the pruned agent ids so callers can publish the reduced
        inventory.
        """
        expired = [
            agent_id
            for agent_id, tombstone in self._evicted_agents.items()
            if self._tombstone_expired(tombstone, now)
        ]
        for agent_id in expired:
            self._evicted_agents.pop(agent_id, None)
            self._pending_evictions.pop(agent_id, None)
        return expired

    def _close_unretained_locked(
        self, record: AgentRecord, run: RunRecord | None, *, name: str
    ) -> bool:
        """Detach and close a settled resident record whose policy is unretained.

        Stores a ``closed`` tombstone (none at all when the tombstone TTL is
        zero) and joins the runtime close through the tracked teardown set.
        Returns ``False`` without acting when the record is not a settled
        resident (draining, detached, reserved, retained, or still running).
        """
        agent_id = record.agent_id
        if self._draining_children:
            return False
        if self._agent_records.get(agent_id) is not record:
            return False
        if record.reserved_for_replacement or record.retained:
            return False
        active = record.current_run
        if active is not None and active.status is RunStatus.RUNNING:
            return False
        record.state = _AgentState.EVICTING
        self._agent_records.pop(agent_id, None)
        self._children.pop(record.session_id, None)
        self._child_links.pop(record.session_id, None)
        tombstone = self._build_terminal_tombstone(
            record, run, AgentAvailability.CLOSED, idle_seconds=0.0
        )
        if tombstone is not None:
            self._insert_tombstone_locked(agent_id, tombstone)
        self._track_child_close(record.runtime, name=name)
        return True

    def _rearm_reaper_locked(self, exclude: frozenset[str] = frozenset()) -> None:
        if self._draining_children:
            return
        current = asyncio.current_task()
        if self._reaper_task is current:
            return
        # Replacing a live reaper displaces it: register the displaced task
        # as drain-owned (its finalizer discards it when it settles) so
        # drain_children cannot return while it can still publish.
        displaced = self._detach_reaper_locked()
        if displaced is not None:
            displaced.cancel()
        self._reaper_deadline = None
        policy = self._retention_policy
        now = self._clock()
        delays: list[float] = []
        eligible = [
            record
            for record in self._eligible_idle_locked()
            if record.agent_id not in exclude
        ]
        ttl_eligible = [
            record
            for record in eligible
            if record.effective_idle_ttl(policy.idle_ttl_seconds) > 0
        ]
        if policy.max_idle_agents and len(eligible) > policy.max_idle_agents:
            delays.append(0.0)
        elif ttl_eligible:
            delays.append(
                max(
                    0.0,
                    min(
                        cast(float, record.idle_since)
                        + record.effective_idle_ttl(policy.idle_ttl_seconds)
                        for record in ttl_eligible
                    )
                    - now,
                )
            )
        if policy.tombstone_ttl_seconds > 0:
            expiries = [
                tombstone.expires_at
                for tombstone in self._evicted_agents.values()
                if tombstone.expires_at is not None
            ]
            if expiries:
                delays.append(max(0.0, min(expiries) - now))
        if not delays:
            return
        delay = min(delays)
        self._reaper_task = asyncio.create_task(
            self._run_reaper(delay), name="vibe-subagent-idle-reaper"
        )
        self._reaper_deadline = now + delay

    async def _publish_inventory_update(self) -> None:
        """Publish the current (possibly reduced) inventory after reaping."""
        if self._draining_children:
            # Drain owns the final inventory publication: a reaper settling
            # mid-drain must not publish a transient pre-drain snapshot on
            # top of the empty payload drain is about to emit. The
            # suppression is recorded so drain still emits the final empty
            # inventory — the disappearance must not be lost at the boundary.
            self._drain_suppressed_inventory = True
            return
        try:
            await self._emit_agents_update()
        except Exception as exc:
            logger.warning("Failed to publish tombstone expiry update", exc_info=exc)

    def _detach_reaper_locked(self) -> asyncio.Task[None] | None:
        """Unregister the live reaper, keeping drain ownership of its task.

        The caller holds the registry lock. The displaced task is
        registered in ``_detached_reaper_tasks`` so drain_children
        joins/cancels it like any other tracked task instead of returning
        while it can still publish; the displaced task's finalizer discards
        it when it settles (cancel, normal completion, or exception).
        Returns the displaced task (``None`` when no reaper is registered).
        """
        displaced = self._reaper_task
        self._reaper_task = None
        if displaced is not None:
            self._detached_reaper_tasks.add(displaced)
        return displaced

    async def _run_reaper(self, delay: float) -> None:
        pruned_tombstones: list[str] = []
        try:
            await self._wakeup(delay)
            async with self._registry_lock:
                if asyncio.current_task() is not self._reaper_task:
                    return
                now = self._clock()
                policy = self._retention_policy
                eligible = sorted(
                    self._eligible_idle_locked(),
                    key=lambda record: (record.idle_since, record.agent_id),
                )
                victims: list[tuple[str, str]] = []
                victims.extend(
                    (record.agent_id, "ttl")
                    for record in eligible
                    if (
                        now - cast(float, record.idle_since)
                        >= record.effective_idle_ttl(policy.idle_ttl_seconds)
                        > 0
                    )
                )
                selected = {agent_id for agent_id, _ in victims}
                remaining = [r for r in eligible if r.agent_id not in selected]
                if policy.max_idle_agents and len(remaining) > policy.max_idle_agents:
                    victims.extend(
                        (record.agent_id, "idle_cap")
                        for record in remaining[
                            : len(remaining) - policy.max_idle_agents
                        ]
                    )
                # Expired terminal metadata is pruned here; results, expiry
                # markers, latest-run indexes, and wait leases are untouched.
                pruned_tombstones = self._prune_expired_tombstones_locked(now)
                if victims:
                    # Hand the schedule to a fresh timer BEFORE awaiting
                    # eviction cleanup: stalled runtime cleanup must not hold
                    # idle/metadata deadline scheduling hostage. This task
                    # unregisters itself (no self-cancellation), so its
                    # finalizer will not rearm; the fresh timer owns the
                    # schedule from here. The victims are excluded because
                    # their idle deadlines are moot once evicted. The
                    # displaced task is registered as drain-owned so
                    # drain_children joins/cancels it like any other tracked
                    # task instead of returning while it can still publish.
                    self._detach_reaper_locked()
                    self._rearm_reaper_locked(
                        exclude=frozenset(agent_id for agent_id, _ in victims)
                    )
            eviction_notifications: list[
                tuple[AgentRecord, AgentEviction, _AgentTombstone | None, object]
            ] = []
            eviction_tasks = [
                asyncio.create_task(
                    self._evict_agent(agent_id, reason, eviction_notifications),
                    name=f"vibe-subagent-eviction:{agent_id}",
                )
                for agent_id, reason in victims
            ]
            self._eviction_tasks.update(eviction_tasks)
            for task in eviction_tasks:
                task.add_done_callback(self._eviction_tasks.discard)
            published = False
            if eviction_tasks:
                eviction_results = asyncio.gather(
                    *eviction_tasks, return_exceptions=True
                )
                try:
                    await asyncio.shield(eviction_results)
                except asyncio.CancelledError:
                    await asyncio.shield(eviction_results)
                    published = await self._notify_evictions(eviction_notifications)
                    if not published and (pruned_tombstones or eviction_notifications):
                        await self._publish_inventory_update()
                    raise
                published = await self._notify_evictions(eviction_notifications)
            # Publish whenever the inventory actually changed — pruned
            # terminal metadata and/or actual evictions — regardless of
            # whether eviction tasks existed or what they did. Permitted
            # eviction notifications already published this same inventory.
            if not published and (pruned_tombstones or eviction_notifications):
                await self._publish_inventory_update()
        finally:
            # A pending reaper can be destroyed while no event loop is
            # running (teardown or interpreter shutdown); the rearm
            # bookkeeping is meaningless there and must stay silent. With a
            # running loop, rearm failures propagate normally.
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No running loop: nothing to rearm and nowhere to report.
                pass
            else:
                # This reaper is settling — handoff-displaced or
                # rearm-displaced: drain ownership ends with the task, on
                # every settle path (cancel, normal completion, exception).
                current = asyncio.current_task()
                if current is not None:
                    self._detached_reaper_tasks.discard(current)
                async with self._registry_lock:
                    if asyncio.current_task() is self._reaper_task:
                        self._reaper_task = None
                        self._rearm_reaper_locked()

    async def _evict_agent(
        self,
        agent_id: str,
        reason: str,
        notifications: list[
            tuple[AgentRecord, AgentEviction, _AgentTombstone | None, object]
        ]
        | None = None,
    ) -> bool:
        if reason not in {"ttl", "idle_cap"}:
            raise ValueError(f"Unknown eviction reason: {reason}")
        async with self._registry_lock:
            record = self._agent_records.get(agent_id)
            if (
                record is None
                or record.state is not _AgentState.IDLE
                or record.reserved_for_replacement
                or not self._generation_is_current(record.root_generation)
            ):
                return False
            cap = self._retention_policy.max_idle_agents
            if reason == "idle_cap" and (
                not cap or len(self._eligible_idle_locked()) <= cap
            ):
                return False
            record.state = _AgentState.EVICTING
            self._agent_records.pop(agent_id)
            self._children.pop(record.session_id, None)
            self._child_links.pop(record.session_id, None)
            # Release-invalidatable authority for this pending eviction. A
            # TTL>0 eviction is authorized by the stored tombstone; a zero-TTL
            # eviction stores no metadata, so this entry is what authorizes
            # (and release revokes) its notification.
            authority = object()
            self._pending_evictions[agent_id] = authority
            run = record.current_run or (
                record.run_history[-1] if record.run_history else None
            )
            latest_run_id = record.latest_run_id or (
                run.run_id if run is not None else None
            )
            if latest_run_id is not None:
                self._latest_run_ids[agent_id] = latest_run_id
            idle_duration = max(
                0.0,
                self._clock()
                - (
                    record.idle_since
                    if record.idle_since is not None
                    else self._clock()
                ),
            )
            tombstone = self._build_terminal_tombstone(
                record, run, AgentAvailability.EVICTED, idle_duration
            )
            if tombstone is not None:
                self._insert_tombstone_locked(agent_id, tombstone)
                # Rearm promptly so tombstone expiry is scheduled even when
                # the awaited runtime cleanup below stalls. A reaper-driven
                # eviction skips this while its coordinating reaper is still
                # registered (that reaper rearms when it finishes; cancelling
                # it here would abort the in-flight run) — but after a
                # schedule handoff (no reaper registered), or when this
                # tombstone expires before the armed timer, the rearm is
                # safe and required.
                if (
                    asyncio.current_task() not in self._eviction_tasks
                    or self._reaper_task is None
                    or (
                        tombstone.expires_at is not None
                        and (
                            self._reaper_deadline is None
                            or tombstone.expires_at < self._reaper_deadline
                        )
                    )
                ):
                    self._rearm_reaper_locked()
        teardown = self._track_child_close(
            record.runtime, name=f"vibe-subagent-teardown:{agent_id}"
        )
        teardown_result = await asyncio.shield(
            asyncio.gather(teardown, return_exceptions=True)
        )
        if teardown_result and isinstance(teardown_result[0], BaseException):
            logger.warning(
                "Failed to close evicted background agent agent_id=%s",
                agent_id,
                exc_info=teardown_result[0],
            )
        eviction = AgentEviction(
            agent_id=agent_id,
            run_id=run.run_id if run is not None else "",
            reason=reason,  # type: ignore[arg-type]
            idle_duration_seconds=idle_duration,
            root_generation=record.root_generation,
        )
        if notifications is not None:
            notifications.append((record, eviction, tombstone, authority))
        else:
            await self._notify_evictions([(record, eviction, tombstone, authority)])
        return True

    @staticmethod
    def _eviction_run_segments(
        record: AgentRecord, eviction: AgentEviction
    ) -> tuple[str, str]:
        """The latest-run identity and outcome segments for an eviction notice.

        Design §5: eviction notifications carry the name alongside the handle,
        run, and outcome. Both segments stay empty when the evicted record has
        no run matching the eviction's run id, so a run-less eviction keeps its
        handle-only wording.
        """
        run = next(
            (
                candidate
                for candidate in (record.current_run, *record.run_history)
                if candidate is not None and candidate.run_id == eviction.run_id
            ),
            None,
        )
        if run is None:
            return "", ""
        outcome = (
            "completed"
            if run.status is RunStatus.COMPLETED
            else "failed"
            if run.status is RunStatus.FAILED
            else "cancelled"
            if run.status is RunStatus.CANCELLED
            else "unknown"
        )
        return f", run {run.run_id}", f", last run {outcome}"

    async def _notify_evictions(
        self,
        notifications: list[
            tuple[AgentRecord, AgentEviction, _AgentTombstone | None, object]
        ],
    ) -> bool:
        """Authorize and deliver eviction notifications.

        Returns ``True`` when an inventory update was published. A TTL>0
        eviction is authorized only while the tombstone it built is still
        stored; a zero-TTL eviction stores no metadata, so it is authorized
        only while its release-invalidatable authority entry survives. The
        authority entry is consumed by this check on every settlement path
        (both TTL variants): a settled eviction notification never leaves
        the entry behind.
        """
        permitted: list[tuple[AgentRecord, AgentEviction]] = []
        async with self._registry_lock:
            for record, eviction, tombstone, authority in sorted(
                notifications, key=lambda notification: notification[1].agent_id
            ):
                # Consume the release-invalidatable authority on every
                # settlement: this notification attempt settles the pending
                # entry for both TTL variants, so a delivered (or fenced)
                # eviction never leaves its authority behind.
                stored_authority = self._pending_evictions.pop(record.agent_id, None)
                if tombstone is not None:
                    authorized = self._evicted_agents.get(record.agent_id) is tombstone
                else:
                    authorized = stored_authority is authority
                if (
                    self._generation_is_current(record.root_generation)
                    and not self._draining_children
                    and record.agent_id not in self._suppressed_notifications
                    and authorized
                ):
                    permitted.append((record, eviction))

            for parent_session_id, parent_notifications in itertools.groupby(
                sorted(
                    permitted,
                    key=lambda notification: notification[0].parent_session_id,
                ),
                key=lambda notification: notification[0].parent_session_id,
            ):
                root = self._root
                parent = (
                    root
                    if root is not None
                    and root.agent_loop.session_id == parent_session_id
                    else self._children.get(parent_session_id)
                )
                if parent is None:
                    continue
                parent_notifications = list(parent_notifications)
                detailed = [
                    (record, eviction, *self._eviction_run_segments(record, eviction))
                    for record, eviction in parent_notifications
                ]
                details = "; ".join(
                    f"{self._record_name(record)} "
                    f"({eviction.agent_id}{run_segment}, "
                    f"{eviction.reason}, idle {eviction.idle_duration_seconds:g}s"
                    f"{outcome_segment})"
                    for record, eviction, run_segment, outcome_segment in detailed
                )
                if len(detailed) == 1:
                    record, eviction, run_segment, outcome_segment = detailed[0]
                    notification = (
                        f"Background agent {self._record_name(record)} "
                        f"({eviction.agent_id}{run_segment}) evicted "
                        f"({eviction.reason}, "
                        f"idle {eviction.idle_duration_seconds:g}s"
                        f"{outcome_segment}); "
                        "launch a new agent for further work."
                    )
                else:
                    notification = (
                        "Background agents evicted: "
                        f"{details}; launch a new agent for further work."
                    )
                try:
                    if (
                        parent.turns.active_turn is None
                        and parent.turns._active_task is None
                        and not parent.turns.queue_state.items
                    ):
                        try:
                            _response, start = parent.turns.start(
                                TurnStartParams(
                                    session_id=parent.agent_loop.session_id,
                                    message=[TextContentBlock(text=notification)],
                                )
                            )
                            start()
                        except Exception:
                            parent.agent_loop._pending_injected_messages.append(
                                LLMMessage(
                                    role=Role.user, content=notification, injected=True
                                )
                            )
                    else:
                        parent.agent_loop._pending_injected_messages.append(
                            LLMMessage(
                                role=Role.user, content=notification, injected=True
                            )
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to notify parent of background agent eviction",
                        exc_info=exc,
                    )
        if permitted:
            try:
                await self._emit_agents_update([
                    eviction for _record, eviction in permitted
                ])
            except Exception as exc:
                logger.warning(
                    "Failed to publish background agent eviction", exc_info=exc
                )
                return False
            return True
        return False

    def _prepare_release_locked(  # noqa: PLR0915
        self, agent_id: str
    ) -> tuple[asyncio.Task[None], ReleaseAgentOutcome]:
        """Destructive release critical section; the caller must hold the lock.

        Purges results, expiry markers, pending notifications, wait leases,
        and write tokens; detaches the record; clears a replacement
        reservation; removes terminal metadata and its pending-eviction
        authority; rearms the reaper; and returns the tracked teardown task
        together with the release outcome. Raises UnknownAgentError when
        nothing about the handle is known.
        """
        self._acquire_notification_suppression(agent_id)
        record = self._agent_records.pop(agent_id, None)
        if record is not None and record.reserved_for_replacement:
            self._clear_replacement_locked(record)
        known_evicted = self._evicted_agents.pop(agent_id, None) is not None
        # Purge the pending-eviction authority alongside everything else:
        # a zero-TTL eviction's notification is authorized only while
        # this entry survives.
        pending_eviction = self._pending_evictions.pop(agent_id, None) is not None
        release_generation = (
            record.root_generation
            if record is not None
            else self._root.agent_loop._session_generation
            if self._root is not None
            else None
        )
        notify_release = (
            record is not None and self._generation_is_current(record.root_generation)
        ) or known_evicted
        keys = {key for key in self._result_store if key[0] == agent_id}
        keys.update(key for key in self._expired_results if key[0] == agent_id)
        keys.update(key for key in self._pending_notifications if key[0] == agent_id)
        keys.update(key for key in self._wait_leases if key[0] == agent_id)
        keys.update(key for key in self._result_write_tokens if key[0] == agent_id)
        for key in keys:
            self._result_store.pop(key, None)
            self._expired_results.discard(key)
            self._pending_notifications.discard(key)
            self._wait_leases.pop(key, None)
            self._result_write_tokens.pop(key, None)
        self._latest_run_ids.pop(agent_id, None)
        if record is None and not known_evicted and not pending_eviction and not keys:
            self._release_notification_suppression(agent_id)
            raise UnknownAgentError(f"Unknown agent: {agent_id}")
        self._rearm_reaper_locked()

        async def cleanup() -> None:
            errors: list[BaseException] = []
            try:
                if record is not None:
                    monitor_tasks = {
                        run.completion_task
                        for run in [*record.run_history, record.current_run]
                        if run is not None
                        and isinstance(run.completion_task, asyncio.Task)
                        and run.completion_task in self._monitor_tasks
                        and not run.completion_task.done()
                    }
                    current_run = record.current_run
                    for task in monitor_tasks:
                        task.cancel()
                    try:
                        if current_run is not None:
                            interrupted = self._interrupt_run(record, current_run)
                            if (
                                not interrupted
                                and current_run.status is RunStatus.RUNNING
                            ):
                                await record.runtime.turns.close()
                        if monitor_tasks:
                            await asyncio.gather(*monitor_tasks, return_exceptions=True)
                    except BaseException as exc:
                        errors.append(exc)
                    finally:
                        self._children.pop(record.session_id, None)
                        self._child_links.pop(record.session_id, None)
                        try:
                            await record.runtime.close()
                        except BaseException as exc:
                            errors.append(exc)
                if notify_release:
                    async with self._registry_lock:
                        notification_permitted = (
                            (
                                release_generation is None
                                and self._generation_identity is None
                            )
                            or (
                                release_generation is not None
                                and self._generation_is_current(release_generation)
                            )
                        ) and (
                            not self._draining_children
                            and self._suppression_owners.get(agent_id) == 1
                        )
                    if notification_permitted:
                        try:
                            await self._emit_agents_update()
                        except BaseException as exc:
                            errors.append(exc)
                if len(errors) == 1:
                    raise errors[0]
                if errors:
                    raise BaseExceptionGroup(
                        "Failed to release background agent", errors
                    )
            finally:
                self._release_notification_suppression(agent_id)

        if record is not None:
            teardown = self._track_child_close(
                record.runtime, cleanup(), name=f"vibe-subagent-release:{agent_id}"
            )
        else:
            teardown = asyncio.create_task(
                cleanup(), name=f"vibe-subagent-release:{agent_id}"
            )
            self._teardown_tasks.add(teardown)
            teardown.add_done_callback(self._teardown_tasks.discard)
        return (
            teardown,
            ReleaseAgentOutcome.EVICTED
            if known_evicted
            else ReleaseAgentOutcome.RELEASED,
        )

    async def _join_release_teardown(
        self, teardown: asyncio.Task[None], agent_id: str, outcome: ReleaseAgentOutcome
    ) -> ReleaseAgentOutcome:
        """Join a release teardown task, preserving cancellation semantics."""
        cancellation: asyncio.CancelledError | None = None
        while not teardown.done():
            try:
                await asyncio.shield(teardown)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
            except BaseException:
                break
        try:
            await teardown
        except BaseException as exc:
            if cancellation is None:
                raise
            logger.warning(
                "Failed to release background agent agent_id=%s", agent_id, exc_info=exc
            )
        if cancellation is not None:
            raise cancellation
        return outcome

    async def release_agent(self, agent_id: str) -> ReleaseAgentOutcome:
        """Close and remove a retained agent or evicted tombstone.

        Unguarded by design: the model tool may release active instances
        (design §3.1). Returns RELEASED for a live agent and EVICTED when
        removing its tombstone. Raises ValueError for unknown agent_id.
        """
        async with self._registry_lock:
            teardown, outcome = self._prepare_release_locked(agent_id)
        return await self._join_release_teardown(teardown, agent_id, outcome)

    async def release_agent_guarded(  # noqa: PLR0911
        self, agent_id: str, expected_run_id: str | None = None
    ) -> ReleaseAgentOutcome:
        """Guarded release for UI callers (design §3.1, §8).

        ATOMIC under the registry lock: attached-root/current-generation
        authority, the expected latest-run identity, a non-active state, and
        no replacement reservation are all verified BEFORE any destructive
        work — never a check-then-release sequence across lock releases. A
        running/finalizing or replacement-reserved record, or a stale
        ``expected_run_id``, returns CONFLICT without stopping or purging
        anything (a stale idle→running race never stops the new run).
        Released, metadata-expired, or unknown handles return UNKNOWN.
        Attached-root authority is verified exactly like the transcript
        authorization sites: the handle must belong to the currently attached
        root session and current generation, so a foreign-parent or
        stale-generation record or tombstone behaves as unknown — never a
        purge. On success the existing release cleanup semantics run:
        results, expiry markers, leases, tokens, and pending-eviction
        authorities are purged, the runtime close is tracked, the reaper is
        rearmed, and the reduced inventory is published.
        """
        async with self._registry_lock:
            record = self._agent_records.get(agent_id)
            tombstone = self._live_tombstone_locked(agent_id, self._clock())
            if record is None and tombstone is None:
                # Released, metadata-expired, or never seen: nothing to guard.
                return ReleaseAgentOutcome.UNKNOWN
            root = self._root
            identity = (
                (root.agent_loop.session_id, root.agent_loop._session_generation)
                if root is not None
                else None
            )
            if identity is None or self._generation_identity != identity:
                # No attached root (or a stale generation snapshot): there is
                # no authority to release under.
                return ReleaseAgentOutcome.UNKNOWN
            if record is not None:
                if (
                    record.parent_session_id != identity[0]
                    or record.root_generation != identity[1]
                ):
                    # Foreign-parent or stale-generation resident record.
                    return ReleaseAgentOutcome.UNKNOWN
                if record.reserved_for_replacement:
                    return ReleaseAgentOutcome.CONFLICT
                if record.state in {_AgentState.RUNNING, _AgentState.FINALIZING}:
                    return ReleaseAgentOutcome.CONFLICT
                latest_run_id = record.latest_run_id or (
                    record.run_history[-1].run_id if record.run_history else None
                )
            else:
                assert tombstone is not None
                if tombstone.parent_identity != identity:
                    # Foreign-parent or stale-generation terminal metadata.
                    return ReleaseAgentOutcome.UNKNOWN
                latest_run_id = tombstone.summary.latest_run_id
            if expected_run_id is not None and latest_run_id != expected_run_id:
                return ReleaseAgentOutcome.CONFLICT
            teardown, outcome = self._prepare_release_locked(agent_id)
        return await self._join_release_teardown(teardown, agent_id, outcome)

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

    async def _capture_child_notification(  # noqa: PLR0912
        self,
        runtime: SessionRuntime,
        record: AgentRecord | None,
        run: RunRecord | None,
        params: ProtocolModel,
    ) -> None:
        # The sink may await arbitrary transport work. Recheck the exact claim,
        # not just the session id, so reuse and teardown cannot revive old updates.
        async with self._registry_lock:
            if record is None or run is None:
                return
            if (
                self._agent_records.get(record.agent_id) is not record
                or record.runtime is not runtime
                or self._children.get(record.session_id) is not runtime
                or record.current_run is not run
            ):
                return
            if (
                record.state is not _AgentState.RUNNING
                or runtime._closed
                or self._draining_children
                or record.agent_id in self._suppressed_notifications
                or not self._generation_is_current(record.root_generation)
            ):
                return
            if getattr(params, "session_id", None) != record.session_id:
                return
            turn_id = getattr(params, "turn_id", None)
            active_turn = runtime.turns.active_turn
            if turn_id is not None and (
                active_turn is None or turn_id != active_turn.id
            ):
                return
            before = (
                record.context_tokens,
                record.context_window,
                record.compacting,
                record.stop_reason,
            )
            if isinstance(params, StatsUpdatedParams):
                record.context_tokens = (
                    params.stats.context_tokens
                    if params.stats.context_tokens >= 0
                    else None
                )
                record.context_window = params.context_window
            elif isinstance(params, HistoryEntryAddedParams):
                entry = params.entry
                if (
                    isinstance(entry, PublicCheckpointEntry)
                    and entry.kind == "compaction"
                    and entry.generation_status
                    is PublicEntryGenerationStatus.IN_PROGRESS
                ):
                    record.compacting = True
                    record.compaction_entry_id = entry.id
            elif isinstance(params, SessionCompactedParams):
                # This precedes stats and the completed checkpoint patch. Never
                # expose the old numerator with activity already cleared.
                record.context_tokens = None
                record.compacting = False
                record.compaction_entry_id = None
            elif isinstance(params, HistoryEntryUpdatedParams):
                if params.entry_id == record.compaction_entry_id and any(
                    patch.path == "/generationStatus" and patch.value == "completed"
                    for patch in params.patch
                ):
                    # Finalization also completes checkpoints on error. Only
                    # CompactEndEvent supplies structured summary details; use
                    # that proof when logging-off compaction has no ID handoff.
                    succeeded = any(
                        patch.path == "/details"
                        and isinstance(patch.value, dict)
                        and "summaryLength" in patch.value
                        and "oldSessionId" in patch.value
                        and "newSessionId" in patch.value
                        for patch in params.patch
                    )
                    if succeeded:
                        record.context_tokens = None
                        record.compacting = False
                    record.compaction_entry_id = None
            elif isinstance(params, TurnCompletedParams):
                if active_turn is not None and params.turn.id != active_turn.id:
                    return
                record.compacting = False
                record.compaction_entry_id = None
                _, run.stop_reason = self._resolve_run_outcome(run, params.turn)
                record.stop_reason = run.stop_reason
            after = (
                record.context_tokens,
                record.context_window,
                record.compacting,
                record.stop_reason,
            )
        if before != after:
            try:
                await self._emit_agents_update()
            except Exception as exc:
                logger.warning("Failed to publish child context update", exc_info=exc)

    def _build_child_runtime(
        self,
        child: AgentLoop,
        *,
        base_history: list[PublicHistoryEntry] | None = None,
        event_sink: Callable[[BaseEvent], Awaitable[None]] | None = None,
    ) -> SessionRuntime:
        # The runtime-owned port resolves the calling session, so a child's
        # approved grant lands in that child's store only, never the parent's.
        # Its save routes through the root orchestrator keyed by the registered
        # caller's cwd; child orchestrators never write.
        child.bind_root_grant_port(
            SessionRootGrantPort(self._resolve_session_loop, self._root_grant_save)
        )
        child.bind_dangerous_command_grant_port(
            SessionDangerousCommandGrantPort(
                self._resolve_session_loop, self._dangerous_command_grant_save
            )
        )
        execution = SessionExecution()
        history = SessionHistory(base_history or [])
        runtime: SessionRuntime | None = None

        def snapshot_state() -> PublicSessionState:
            if runtime is None:
                raise RuntimeError("Child session runtime is not bound")
            return self._public_state(runtime, 200)

        async def notify_child(method: str, params: ProtocolModel) -> None:
            bound = runtime
            record = next(
                (
                    record
                    for record in self._agent_records.values()
                    if record.runtime is bound
                ),
                None,
            )
            run = record.current_run if record is not None else None
            generation = child._session_generation
            await self._notify_child(method, params)
            if bound is not None and child._session_generation == generation:
                await self._capture_child_notification(bound, record, run, params)

        turns = TurnController(
            child,
            notify_child,
            self._deliver_callback,
            execution,
            self,
            snapshot_state=snapshot_state,
            tool_io=self._tool_io,
            event_sink=event_sink,
            session_coordinator=self,
        )
        runtime = SessionRuntime(
            child,
            turns,
            execution,
            history,
            retire_accounting=lambda: self._runtime_factory.retire_child_accounting(
                child
            ),
        )
        return runtime

    def _runtime(self, session_id: str | None) -> SessionRuntime:
        if session_id is None:
            raise RuntimeError("Subagent parent session is missing")
        root = self._root
        if root is not None and root.agent_loop.session_id == session_id:
            return root
        child = self._children.get(session_id)
        if child is None:
            raise RuntimeError(f"Subagent parent session not found: {session_id}")
        return child

    def _require_child(self, session_id: str) -> SessionRuntime:
        child = self._children.get(session_id) or self._readable_children.get(
            session_id
        )
        if child is None:
            raise KeyError(session_id)
        return child

    async def reserve_resume_admission(self, root: SessionRuntime) -> None:
        """Close child admission atomically with the live-work check."""
        async with self._ensure_child_lock:
            reasons = []
            if (
                root.execution.active is not None
                and root.execution.active.kind is not SessionExecutionKind.LIFECYCLE
            ):
                reasons.append("a turn or operation is running")
            if root.turns.has_queued_turns:
                reasons.append("input is queued")
            if any(
                isinstance(callback.state, OpenCallbackState)
                for callback in root.turns.callbacks
            ):
                reasons.append("a decision is pending")
            if self._creating_children or any(
                (
                    record.current_run is not None
                    and record.current_run.status is RunStatus.RUNNING
                )
                or record.reserved_for_replacement
                for record in self._agent_records.values()
            ):
                reasons.append("background agents are running")
            if reasons:
                raise SessionExecutionConflict(
                    "Cannot switch sessions while "
                    + ", ".join(reasons)
                    + ". Wait for work to finish, answer pending decisions, remove queued input, "
                    "or stop work first; then retry. Nothing was switched."
                )
            self._admission_closed = True

    async def close_children(self) -> None:
        self._require_policy_unreserved()
        self._admission_closed = True
        await self.drain_children()

    async def drain_children(self) -> None:  # noqa: PLR0912, PLR0915
        self._require_policy_unreserved()
        self._draining_children = True
        async with self._registry_lock:
            for record in self._agent_records.values():
                if record.reserved_for_replacement:
                    self._clear_replacement_locked(record)
        errors: list[BaseException] = []
        try:
            reaper = self._reaper_task
            self._reaper_task = None
            self._reaper_deadline = None
            if reaper is not None and reaper is not asyncio.current_task():
                # Cancelling the reaper may abort an in-flight publication:
                # record the lost intent so the final empty inventory below
                # still surfaces the disappearance.
                self._drain_suppressed_inventory = True
                reaper.cancel()
                await asyncio.gather(reaper, return_exceptions=True)
            # A reaper that handed its schedule to a fresh timer (a
            # displaced reaper) is still an inventory producer until it
            # settles: drain owns it too, so drain cannot return — or
            # publish its final empty inventory — while a displaced reaper
            # can still deliver a payload. The registered reaper is never
            # in the detached set (the handoff unregisters before arming
            # the fresh timer); the guard keeps the join single.
            detached_reapers = [
                task
                for task in self._detached_reaper_tasks
                if task is not asyncio.current_task() and task is not reaper
            ]
            for task in detached_reapers:
                task.cancel()
            if detached_reapers:
                # Same for displaced reapers: a cancelled reaper may have
                # been about to publish its reduced inventory.
                self._drain_suppressed_inventory = True
                await asyncio.gather(*detached_reapers, return_exceptions=True)
            monitors = [
                task
                for task in self._monitor_tasks
                if task is not asyncio.current_task()
            ]
            for task in monitors:
                task.cancel()
            if monitors:
                await asyncio.gather(*monitors, return_exceptions=True)
            evictions = [
                task
                for task in self._eviction_tasks
                if task is not asyncio.current_task()
            ]
            if evictions:
                await asyncio.gather(*evictions, return_exceptions=True)
            async with self._ensure_child_lock:
                self._require_policy_unreserved()
                background_tasks = [
                    record.current_run.completion_task
                    for record in self._agent_records.values()
                    if record.current_run is not None
                    and record.current_run.status is RunStatus.RUNNING
                ]
                for task in background_tasks:
                    task.cancel()
                if background_tasks:
                    await asyncio.gather(*background_tasks, return_exceptions=True)
                agents_changed = bool(self._agent_records or self._evicted_agents)
                self._agent_records.clear()
                self._evicted_agents.clear()
                self._pending_evictions.clear()
                self._result_store.clear()
                self._expired_results.clear()
                self._latest_run_ids.clear()
                self._wait_leases.clear()
                self._pending_notifications.clear()
                self._result_write_tokens.clear()
                children = list(self._children.values())
                pending_close_runtimes = list(
                    {
                        id(runtime): runtime
                        for runtime, _ in self._pending_child_closes.values()
                    }.values()
                )
                self._children.clear()
                self._child_links.clear()
                self._stored_children.clear()
                self._readable_children.clear()
                self._detached_child_ids.clear()
            teardown = [
                task
                for task in self._teardown_tasks
                if task is not asyncio.current_task()
            ]
            if teardown:
                await asyncio.gather(*teardown, return_exceptions=True)
            for runtime in pending_close_runtimes:
                if not runtime._closed:
                    try:
                        await runtime.close()
                    except BaseException as exc:
                        errors.append(exc)
                        continue
                for session_id, pending in list(self._pending_child_closes.items()):
                    if pending[0] is runtime:
                        self._pending_child_closes.pop(session_id, None)
                        failed_wrapper = pending[1]
                        self._teardown_tasks.discard(failed_wrapper)
                        self._retire_teardown_task(failed_wrapper)
                close_task = runtime._close_task
                if close_task is not None:
                    self._teardown_tasks.discard(close_task)
            self._suppression_owners.clear()
            self._suppressed_notifications.clear()
            for runtime in children:
                try:
                    await runtime.close()
                except BaseException as exc:
                    errors.append(exc)
            # A disappearance publication suppressed or cancelled at the
            # drain boundary (a reaper pruning the final tombstone across
            # the boundary, or an in-flight publication aborted by
            # cancelling its reaper) must still surface: the final empty
            # inventory is emitted so the delivered client inventory never
            # keeps a ghost row for the drained registry.
            if agents_changed or self._drain_suppressed_inventory:
                try:
                    await self._emit_agents_update()
                except BaseException as exc:
                    errors.append(exc)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise BaseExceptionGroup(
                    "Failed to drain child session runtimes", errors
                )
        finally:
            self._draining_children = False
            self._drain_suppressed_inventory = False

    @staticmethod
    async def _discard_child(child: AgentLoop) -> None:
        with suppress(Exception):
            await close_agent_loop(child)
