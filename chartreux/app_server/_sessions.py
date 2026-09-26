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
from copy import deepcopy
from dataclasses import dataclass, field, replace
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
from chartreux.app_server._projection import project_history, project_session_log
from chartreux.app_server._root_session import SessionHandoff, rebind_history
from chartreux.app_server._runtime import AgentRuntimeFactory, close_agent_loop
from chartreux.app_server._session_history import SessionHistory
from chartreux.app_server._state import build_public_state
from chartreux.app_server._streaming import BoundedEventQueue, stream_until_complete
from chartreux.app_server._turns import DeliverCallback, TurnController, TurnStartAction
from chartreux.app_server.models import (
    CallbackOutput,
    CancelledEffectState,
    CompletedEffectState,
    EffectResultDisplay,
    FailedEffectState,
    OpenCallbackState,
    PublicCallbackEntry,
    PublicError,
    PublicHistoryEntry,
    PublicSessionState,
    PublicTurn,
    PublicTurnQueue,
    PublicTurnStatus,
    TextContentBlock,
)
from chartreux.app_server.protocol import (
    AgentSummaryModel,
    CallbackResultError,
    TurnInterruptParams,
    TurnStartParams,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop import _PreparedPolicyReplacement
from chartreux.core.agents.launch import FrozenPersona, LaunchCandidate, resolve_launch
from chartreux.core.agents.models import AgentProfile
from chartreux.core.config._restrictions import (
    SourceRestrictions,
    partition_policy_sources,
)
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.events import BaseEvent, ToolStreamEvent
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.saved_sessions import delete_saved_session
from chartreux.core.session_types import ChildSessionLink
from chartreux.core.subagents import (
    AgentAvailability,
    AgentBusyError,
    AgentEvictedError,
    AgentEviction,
    AgentProfileMismatchError,
    AgentResultExpiredError,
    AgentSummary,
    InvalidLaunchModelError,
    LaunchConfig,
    LaunchConfigError,
    ReleaseAgentOutcome,
    RunStatus,
    SubagentRunAccumulator,
    SubagentRunnerPort,
    TaskArgs,
    TaskMemberResult,
    TaskResult,
    UnknownAgentError,
    normalize_task_summary,
    prepare_subagent_prompt,
)
from chartreux.core.tools.base import InvokeContext
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
_AGENT_ID_HIGH_WATER_MARK_KEY = "_app_server_agent_id_high_water_mark"


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
        self._closed = True


@dataclass(frozen=True, slots=True)
class _FanOutPreflight:
    candidate: LaunchCandidate
    parent: SessionRuntime
    parent_identity: tuple[str, int]
    parent_orchestrator: object
    accepted_config_token: object | None
    parent_config_snapshot: object
    catalog_snapshot: object | None
    authority_revision: int
    profile_name: str
    profile_snapshot: AgentProfile
    model_expression: str
    launch_config: LaunchConfig


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
    completed_at: float | None = None
    terminal_identity: tuple[str, str, str] | None = None
    progress_summaries: list[str] = field(default_factory=list)
    progress_overflow: bool = False


@dataclass(slots=True)
class AgentRecord:
    agent_id: str
    profile: str
    session_id: str
    runtime: SessionRuntime
    root_generation: int
    idle_ttl_seconds: int | None = None
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
class _AgentTombstone:
    """Private eviction data kept alongside the public agent summary."""

    summary: AgentSummary
    child_session_id: str
    parent_identity: tuple[str, int]


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
    ) -> None:
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
        self._fan_out_effect_projectors: dict[str, Any] = {}
        self._pending_fan_out_result_leases: set[str] = set()
        self._fan_out_result_leases: dict[str, tuple[str, str]] = {}
        self._eviction_tasks: set[asyncio.Task[bool]] = set()
        self._registry_lock = asyncio.Lock()
        self._clock = clock
        self._wakeup = wakeup
        self._reaper_task: asyncio.Task[None] | None = None
        self._retention_policy = (0, 0)
        self._generation_identity: tuple[str, int] | None = None
        self._draining_children = False
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

    def begin_root_generation(self) -> None:
        root = self._root
        if root is None:
            self._retention_policy = (0, 0)
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
        self._retention_policy = (config.idle_ttl_seconds, config.max_idle_agents)
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
        if not records:
            return
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
                if agent_id not in self._evicted_agents:
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
            tombstone = self._evicted_agents.get(agent_id)
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
            tombstone = self._evicted_agents.get(snapshot.agent_id)
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
            profile_name=(args.agent if "agent" in args.model_fields_set else profile),
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

    def _fan_out_preflight_is_current(
        self, parent: SessionRuntime, args: TaskArgs, preflight: _FanOutPreflight
    ) -> bool:
        config = args.config
        if config is None or not isinstance(config.model, str):
            return False
        profile_name = args.agent if "agent" in args.model_fields_set else "worker"
        try:
            current_profile = parent.agent_loop.agent_manager.get_agent(profile_name)
        except ValueError:
            return False

        orchestrator = parent.agent_loop.config_orchestrator
        identity = (parent.agent_loop.session_id, parent.agent_loop._session_generation)
        # tool_inventory/authorized_tool_names are intentionally not snapshotted:
        # policy replacement bumps the checked authority revision, launch
        # reconfiguration swaps the checked orchestrator, and require_idle()-gated
        # MCP mutations cannot overlap this mid-turn fan-out window.
        return (
            preflight.parent is parent
            and preflight.parent_identity == identity
            and self._generation_identity == identity
            and preflight.parent_orchestrator is orchestrator
            and preflight.accepted_config_token is orchestrator.accepted_token
            and preflight.parent_config_snapshot is parent.agent_loop.config
            and preflight.catalog_snapshot is parent.agent_loop.config.catalog_snapshot
            and preflight.authority_revision == parent.agent_loop._authority_revision
            and preflight.profile_name == profile_name
            and preflight.profile_snapshot == current_profile
            and preflight.profile_snapshot == preflight.candidate.profile
            and preflight.candidate.profile.name == profile_name
            and preflight.model_expression == config.model
            and preflight.candidate.semantic_overrides.model == config.model
            and preflight.launch_config.model_fields_set == config.model_fields_set
            and preflight.launch_config.model_dump(exclude_unset=True, mode="python")
            == config.model_dump(exclude_unset=True, mode="python")
            and preflight.candidate.orchestrator.availability_registry.is_available(
                preflight.candidate.committed_model.base_model,
                preflight.candidate.committed_model.provider,
            )
        )

    async def _create_registered_child(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        ctx: InvokeContext,
        *,
        preflight: _FanOutPreflight | None = None,
    ) -> tuple[SessionRuntime, int | None]:
        candidate = (
            preflight.candidate
            if preflight is not None
            and self._fan_out_preflight_is_current(parent, args, preflight)
            else self._resolve_launch_candidate(parent, args)
        )
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
                should_link_projection = parent.turns._projector is not None and (
                    ctx.tool_call_id not in self._pending_fan_out_result_leases
                    or ctx.tool_call_id in self._fan_out_effect_projectors
                )
                if should_link_projection:
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
            return runtime, candidate.profile.idle_ttl_seconds

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
        self, record: AgentRecord, previous_idle_since: float | None
    ) -> None:
        async with self._registry_lock:
            if (
                self._agent_records.get(record.agent_id) is not record
                or record.state is not _AgentState.RUNNING
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

    async def _start_fan_out_member_effect(
        self, parent: SessionRuntime, parent_tool_call_id: str, member_tool_call_id: str
    ) -> None:
        """Create the synthetic projected effect required for a fan-out child link."""
        projector = parent.turns._projector
        if projector is not None:
            try:
                update = projector.start_fan_out_member_effect(
                    parent_tool_call_id, member_tool_call_id
                )
            except ValueError:
                return
            self._fan_out_effect_projectors[member_tool_call_id] = projector
            await parent.turns._emit_projected(update)

    async def _complete_fan_out_member_effect(
        self,
        parent: SessionRuntime,
        member_tool_call_id: str,
        status: str,
        message: str = "",
    ) -> None:
        """Complete a synthetic fan-out effect with its member's terminal status."""
        projector = self._fan_out_effect_projectors.pop(member_tool_call_id, None)
        if projector is None:
            return
        display = EffectResultDisplay(success=status == "completed", message=message)
        if status == "completed":
            state = CompletedEffectState(output=None, display=display)
        elif status == "cancelled":
            state = CancelledEffectState(reason=message, display=display)
        else:
            state = FailedEffectState(
                error=PublicError(message=message), display=display
            )
        try:
            update = projector.complete_effect(member_tool_call_id, state)
        except ValueError:
            return
        await parent.turns._emit_projected(update)

    def _hold_fan_out_result_lease(
        self, member_tool_call_id: str, agent_id: str, run_id: str
    ) -> None:
        """Protect a committed member result before its launch acknowledgement yields."""
        key = (agent_id, run_id)
        self._wait_leases[key] = self._wait_leases.get(key, 0) + 1
        self._fan_out_result_leases[member_tool_call_id] = key

    async def _release_fan_out_result_lease(self, member_tool_call_id: str) -> None:
        key = self._fan_out_result_leases.pop(member_tool_call_id, None)
        if key is None:
            return
        async with self._registry_lock:
            leases = self._wait_leases.get(key, 0)
            if leases > 1:
                self._wait_leases[key] = leases - 1
            else:
                self._wait_leases.pop(key, None)
            self._expire_results_locked()

    async def _release_fan_out_members(self, member_tool_call_ids: list[str]) -> None:
        """Release every member committed before a fan-out launch was cancelled."""
        self._pending_fan_out_result_leases.difference_update(member_tool_call_ids)
        agent_ids = {
            record.agent_id
            for record in self._agent_records.values()
            if self._child_links.get(record.session_id, (None, None))[1]
            in member_tool_call_ids
        }
        await asyncio.gather(
            *(
                self._release_fan_out_result_lease(effect_id)
                for effect_id in member_tool_call_ids
            ),
            *(self.release_agent(agent_id) for agent_id in agent_ids),
            return_exceptions=True,
        )

    def _watch_fan_out_member(
        self,
        parent: SessionRuntime,
        member_tool_call_id: str,
        member: TaskMemberResult,
        candidate: LaunchCandidate,
    ) -> None:
        """Update a background fan-out member and its effect when its run finishes."""
        assert member.agent_id is not None and member.run_id is not None
        agent_id = member.agent_id
        run_id = member.run_id

        async def watch() -> None:
            try:
                result = await self.wait_for_agent(agent_id, run_id)
                identity = self._fan_out_member_identity(agent_id, run_id, candidate)
                member.base_model = identity["base_model"]
                member.provider = identity["provider"]
                member.display_name = identity["display_name"]
                if result.completed:
                    await self._complete_fan_out_member_effect(
                        parent, member_tool_call_id, "completed", "task completed"
                    )
                elif (
                    self._fan_out_member_terminal_status(agent_id, run_id)
                    is RunStatus.CANCELLED
                ):
                    await self._complete_fan_out_member_effect(
                        parent, member_tool_call_id, "cancelled", "task cancelled"
                    )
                else:
                    await self._complete_fan_out_member_effect(
                        parent, member_tool_call_id, "failed", result.response
                    )
            except asyncio.CancelledError:
                await self._complete_fan_out_member_effect(
                    parent, member_tool_call_id, "cancelled", "task cancelled"
                )
                raise
            except BaseException as exc:
                member.status = "failed"
                member.error = {"code": "wait_failed", "message": str(exc)}
                await self._complete_fan_out_member_effect(
                    parent, member_tool_call_id, "failed", str(exc)
                )
            finally:
                await self._release_fan_out_result_lease(member_tool_call_id)

        task = asyncio.create_task(
            watch(), name=f"vibe-fan-out-member:{agent_id}:{run_id}"
        )
        self._monitor_tasks.add(task)
        task.add_done_callback(self._monitor_tasks.discard)

    def _fan_out_member_terminal_status(self, agent_id: str, run_id: str) -> RunStatus:
        """Return the retained terminal status for an exact fan-out run."""
        record = self._agent_records.get(agent_id)
        if record is not None:
            for run in (record.current_run, *record.run_history):
                if run is not None and run.run_id == run_id:
                    if run.status is not RunStatus.RUNNING:
                        return run.status
        raise RuntimeError(
            f"Fan-out run {run_id} on agent {agent_id} has no terminal status"
        )

    def _fan_out_member_identity(
        self, agent_id: str | None, run_id: str | None, candidate: LaunchCandidate
    ) -> dict[str, str]:
        """Return the concrete identity captured for this exact run."""
        terminal_identity: tuple[str, str, str] | None = None
        if agent_id is not None and run_id is not None:
            record = self._agent_records.get(agent_id)
            if record is not None:
                runs = (record.current_run, *record.run_history)
                for run in runs:
                    if run is not None and run.run_id == run_id:
                        terminal_identity = run.terminal_identity
                        break
            if terminal_identity is None:
                stored = self._result_store.get((agent_id, run_id))
                if stored is not None:
                    terminal_identity = stored.terminal_identity
        if terminal_identity is not None:
            base_model, provider, wire_name = terminal_identity
            return {
                "base_model": base_model,
                "provider": provider,
                "display_name": format_model_display_name(provider, wire_name),
            }
        committed = candidate.committed_model
        return {
            "base_model": committed.base_model,
            "provider": committed.provider,
            "display_name": format_model_display_name(
                committed.provider, committed.wire_name
            ),
        }

    async def _run_fan_out(  # noqa: PLR0912, PLR0914, PLR0915
        self, parent: SessionRuntime, args: TaskArgs, ctx: InvokeContext
    ) -> TaskResult:
        """Launch every explicit role member through the ordinary retained path."""
        if args.agent_id is not None:
            raise InvalidLaunchModelError("agent_id", "fan_out cannot reuse an agent")
        if args.config is None or "model" not in args.config.model_fields_set:
            raise InvalidLaunchModelError(
                "config.model", "fan_out requires an explicitly supplied @role model"
            )
        expression = args.config.model
        if not isinstance(expression, str):
            raise InvalidLaunchModelError(
                "config.model", "fan_out requires exactly one string @role model"
            )
        if not expression.startswith("@") or expression == "@":
            raise InvalidLaunchModelError(
                "config.model", "fan_out requires an explicitly supplied @role model"
            )
        snapshot = parent.agent_loop.config.catalog_snapshot
        if snapshot is None:
            raise InvalidLaunchModelError(
                "config.model", "Model catalog is unavailable"
            )
        role = snapshot.catalog.roles.get(expression[1:])
        if role is None:
            raise InvalidLaunchModelError(
                "config.model", f"Unknown model role {expression!r}"
            )
        members = role.models

        # Resolve every member before creating any child. Passing its canonical base
        # rather than the role also makes each child assignment-time concrete. A
        # member-local model rejection does not prevent runnable siblings from being
        # launched, but the whole call still requires at least one runnable member.
        member_args: list[
            tuple[int, str, TaskArgs, LaunchCandidate, _FanOutPreflight]
        ] = []
        skipped: dict[int, TaskMemberResult] = {}
        first_rejection: tuple[str, LaunchConfigError] | None = None
        for index, base in enumerate(members):
            member_config = args.config.model_copy(update={"model": base})
            member = args.model_copy(
                update={"config": member_config, "fan_out": False, "background": True}
            )
            try:
                candidate = self._resolve_launch_candidate(parent, member)
            except InvalidLaunchModelError as exc:
                if first_rejection is None:
                    first_rejection = (base, exc)
                definition = snapshot.catalog.models.get(base)
                deployment = (
                    definition.deployments[0]
                    if definition is not None and definition.deployments
                    else None
                )
                provider = deployment.provider if deployment is not None else ""
                wire_name = deployment.name if deployment is not None else base
                skipped[index] = TaskMemberResult(
                    index=index,
                    base_model=base,
                    provider=provider,
                    display_name=format_model_display_name(provider, wire_name),
                    status="skipped",
                    error={"code": "preflight_rejected", "message": str(exc)},
                )
                continue
            except LaunchConfigError as exc:
                raise InvalidLaunchModelError(
                    "config.model", f"fan_out member {base!r} rejected: {exc}"
                ) from exc
            assert member.config is not None and isinstance(member.config.model, str)
            preflight = _FanOutPreflight(
                candidate=candidate,
                parent=parent,
                parent_identity=(
                    parent.agent_loop.session_id,
                    parent.agent_loop._session_generation,
                ),
                parent_orchestrator=parent.agent_loop.config_orchestrator,
                accepted_config_token=(
                    parent.agent_loop.config_orchestrator.accepted_token
                ),
                parent_config_snapshot=parent.agent_loop.config,
                catalog_snapshot=parent.agent_loop.config.catalog_snapshot,
                authority_revision=parent.agent_loop._authority_revision,
                profile_name=candidate.profile.name,
                profile_snapshot=deepcopy(candidate.profile),
                model_expression=member.config.model,
                launch_config=member.config.model_copy(deep=True),
            )
            member_args.append((index, base, member, candidate, preflight))

        if not member_args:
            assert first_rejection is not None
            base, exc = first_rejection
            raise InvalidLaunchModelError(
                "config.model", f"fan_out member {base!r} rejected: {exc}"
            ) from exc

        launched: list[
            tuple[int, str, LaunchCandidate, TaskResult | BaseException]
        ] = []
        started_member_effect_ids: list[str] = []
        try:
            for index, base, member, candidate, preflight in member_args:
                member_ctx = replace(
                    ctx, tool_call_id=f"{ctx.tool_call_id}:fan-out:{index}"
                )
                launch_succeeded = False
                try:
                    ack: TaskResult | None = None
                    started_member_effect_ids.append(member_ctx.tool_call_id)
                    await self._start_fan_out_member_effect(
                        parent, ctx.tool_call_id, member_ctx.tool_call_id
                    )
                    self._pending_fan_out_result_leases.add(member_ctx.tool_call_id)
                    async for event in self.run(
                        member,
                        member_ctx,
                        preflight=preflight,
                        defer_launch_agents_update=True,
                    ):
                        if isinstance(event, TaskResult):
                            ack = event
                    assert ack is not None
                    launched.append((index, base, candidate, ack))
                    launch_succeeded = True
                except asyncio.CancelledError:
                    await asyncio.gather(
                        *(
                            self._complete_fan_out_member_effect(
                                parent,
                                effect_id,
                                "cancelled",
                                "fan-out launch cancelled",
                            )
                            for effect_id in started_member_effect_ids
                        ),
                        self._release_fan_out_members(started_member_effect_ids),
                        return_exceptions=True,
                    )
                    raise
                except BaseException as exc:
                    await self._complete_fan_out_member_effect(
                        parent, member_ctx.tool_call_id, "failed", str(exc)
                    )
                    launched.append((index, base, candidate, exc))
                finally:
                    if not launch_succeeded:
                        self._pending_fan_out_result_leases.discard(
                            member_ctx.tool_call_id
                        )
        finally:
            if not self._draining_children:
                try:
                    await self._emit_agents_update()
                except Exception as exc:
                    logger.warning(
                        "Failed to publish coalesced fan-out agent update", exc_info=exc
                    )

        collected_results: dict[tuple[str, str], TaskResult] = {}

        async def terminal(
            item: tuple[int, str, LaunchCandidate, TaskResult | BaseException],
        ) -> TaskMemberResult:
            index, _base, candidate, value = item
            member_tool_call_id = f"{ctx.tool_call_id}:fan-out:{index}"
            common = {
                "index": index,
                **self._fan_out_member_identity(None, None, candidate),
            }
            if isinstance(value, BaseException):
                return TaskMemberResult(
                    **common,
                    status="failed",
                    error={"code": "launch_failed", "message": str(value)},
                )
            if args.background:
                common = {
                    "index": index,
                    **self._fan_out_member_identity(
                        value.agent_id, value.run_id, candidate
                    ),
                }
                member = TaskMemberResult(
                    **common,
                    status="running",
                    agent_id=value.agent_id,
                    run_id=value.run_id,
                )
                self._watch_fan_out_member(
                    parent, member_tool_call_id, member, candidate
                )
                return member
            assert value.agent_id is not None and value.run_id is not None
            try:
                result = await self.wait_for_agent(value.agent_id, value.run_id)
                collected_results[(value.agent_id, value.run_id)] = result
            except asyncio.CancelledError:
                await self._release_fan_out_result_lease(member_tool_call_id)
                await self._complete_fan_out_member_effect(
                    parent, member_tool_call_id, "cancelled", "fan-out wait cancelled"
                )
                raise
            except BaseException as exc:
                await self._release_fan_out_result_lease(member_tool_call_id)
                await self._complete_fan_out_member_effect(
                    parent, member_tool_call_id, "failed", str(exc)
                )
                return TaskMemberResult(
                    **common,
                    status="failed",
                    agent_id=value.agent_id,
                    run_id=value.run_id,
                    error={"code": "wait_failed", "message": str(exc)},
                )
            common = {
                "index": index,
                **self._fan_out_member_identity(
                    value.agent_id, value.run_id, candidate
                ),
            }
            terminal_status = (
                self._fan_out_member_terminal_status(value.agent_id, value.run_id)
                if not result.completed
                else None
            )
            await self._release_fan_out_result_lease(member_tool_call_id)
            if result.completed:
                await self._complete_fan_out_member_effect(
                    parent, member_tool_call_id, "completed", "task completed"
                )
                return TaskMemberResult(
                    **common,
                    status="completed",
                    agent_id=value.agent_id,
                    run_id=value.run_id,
                    result=result.response,
                    metadata=result.metadata,
                )
            if terminal_status is RunStatus.CANCELLED:
                await self._complete_fan_out_member_effect(
                    parent, member_tool_call_id, "cancelled", "task cancelled"
                )
                return TaskMemberResult(
                    **common,
                    status="cancelled",
                    agent_id=value.agent_id,
                    run_id=value.run_id,
                    metadata=result.metadata,
                )
            await self._complete_fan_out_member_effect(
                parent, member_tool_call_id, "failed", result.response
            )
            return TaskMemberResult(
                **common,
                status="failed",
                agent_id=value.agent_id,
                run_id=value.run_id,
                error={"code": "runtime_failed", "message": result.response},
                metadata=result.metadata,
            )

        # gather deliberately does not cancel siblings when one run fails.
        try:
            terminal_results = await asyncio.gather(
                *(terminal(item) for item in launched)
            )
        except asyncio.CancelledError:
            await self._release_fan_out_members(started_member_effect_ids)
            raise
        results_by_index = {member.index: member for member in terminal_results}
        results_by_index.update(skipped)
        results = [results_by_index[index] for index in range(len(members))]
        turns_used = sum(result.turns_used for result in collected_results.values())
        return TaskResult(
            response="",
            turns_used=turns_used,
            completed=all(
                member.status in {"running", "completed", "skipped"}
                for member in results
            ),
            members=results,
        )

    async def run(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        args: TaskArgs,
        ctx: InvokeContext,
        *,
        preflight: _FanOutPreflight | None = None,
        defer_launch_agents_update: bool = False,
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        if ctx.is_subagent:
            raise RuntimeError("Agent depth limit of 1 reached")
        parent = self._runtime(ctx.session_id)
        if args.fan_out:
            yield await self._run_fan_out(parent, args, ctx)
            return
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
                        if args.agent_id in self._evicted_agents:
                            raise AgentEvictedError(
                                f"Agent evicted: {args.agent_id}; start a new agent"
                            )
                        raise UnknownAgentError(f"Unknown agent: {args.agent_id}")
                    if (
                        "agent" in args.model_fields_set
                        and record.profile != args.agent
                    ):
                        raise AgentProfileMismatchError(
                            f"Agent profile mismatch: expected {record.profile}, got {args.agent}"
                        )
                    if record.state is not _AgentState.IDLE:
                        raise AgentBusyError("Agent is already running a task")
                    if (
                        record.runtime._closed is True
                        or self._children.get(record.session_id) is not record.runtime
                    ):
                        raise UnknownAgentError(f"Unknown agent: {args.agent_id}")
                    self._require_task_profile_allowed(parent, record.profile)
                    previous_idle_since = record.idle_since
                    reserved_child_generation = (
                        record.runtime.agent_loop._session_generation
                    )
                    reserved_parent_authority_revision = (
                        parent.agent_loop._authority_revision
                    )
                    record.state = _AgentState.RUNNING
                    record.idle_since = None
                    self._rearm_reaper_locked()
            else:
                record = None
            created_record = False
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
                        await self._rollback_reused_claim(record, previous_idle_since)
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
                        await self._rollback_reused_claim(record, previous_idle_since)
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
                        await self._rollback_reused_claim(record, previous_idle_since)
                        raise
                runtime = record.runtime
            else:
                self._require_task_profile_allowed(parent, args.agent)
                runtime, idle_ttl_seconds = await self._create_registered_child(
                    parent, args, ctx, preflight=preflight
                )
                agent_id = await self._issue_agent_id(parent)
                record = AgentRecord(
                    agent_id=agent_id,
                    profile=runtime.agent_loop.launch_profile or args.agent,
                    session_id=runtime.agent_loop.session_id,
                    runtime=runtime,
                    root_generation=parent.agent_loop._session_generation,
                    idle_ttl_seconds=idle_ttl_seconds,
                    parent_session_id=parent.agent_loop.session_id,
                    initial_task_summary=normalize_task_summary(
                        args.task_summary, fallback=args.task
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

            agent_id = record.agent_id
            run_id = self._new_run_id(agent_id)
            prior_current_run = record.current_run
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
                    args.task_summary, fallback=args.task
                ),
            )
            record.current_run = run_record
            record.latest_run_id = run_id
            self._latest_run_ids[agent_id] = run_id
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
                        if turn.error is not None:
                            run_record.status = RunStatus.FAILED
                        elif turn.status is PublicTurnStatus.COMPLETED:
                            run_record.status = RunStatus.COMPLETED
                        else:
                            run_record.status = RunStatus.CANCELLED
                    except asyncio.CancelledError:
                        run_record.status = RunStatus.CANCELLED
                        run_record.result = TaskResult(
                            response="",
                            turns_used=0,
                            completed=False,
                            agent_id=agent_id,
                            run_id=run_id,
                            metadata=runtime.agent_loop.completion_metadata_since(
                                completion_metadata_mark
                            ),
                        )
                        raise
                    except Exception as exc:
                        accumulator.record_error(str(exc))
                        run_record.status = RunStatus.FAILED
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
                                record.state = _AgentState.FINALIZING
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
                                "completed"
                                if run_record.status is RunStatus.COMPLETED
                                else "failed"
                                if run_record.status is RunStatus.FAILED
                                else "cancelled"
                            )
                            notification = (
                                f"Background agent {agent_id} (run {run_id})"
                                f" {status_word}."
                                " Use get_agent_result to retrieve the result."
                            )
                            if (
                                parent.turns.active_turn is None
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
                                    and record.state is _AgentState.FINALIZING
                                    and self._generation_is_current(
                                        record.root_generation
                                    )
                                ):
                                    if (
                                        record.effective_idle_ttl(
                                            self._retention_policy[0]
                                        )
                                        == 0
                                        and record.idle_ttl_seconds is None
                                        and self._retention_policy[1] == 0
                                    ):
                                        record.state = _AgentState.EVICTING
                                        self._agent_records.pop(agent_id, None)
                                        self._children.pop(record.session_id, None)
                                        self._child_links.pop(record.session_id, None)
                                        self._evicted_agents[agent_id] = (
                                            _AgentTombstone(
                                                summary=AgentSummary(
                                                    agent_id=agent_id,
                                                    profile=record.profile,
                                                    availability=AgentAvailability.EVICTED,
                                                    current_run_id=run_id,
                                                    current_run_status=run_record.status,
                                                    last_run_status=record.last_run_status,
                                                    initial_task_summary=record.initial_task_summary,
                                                    current_task_summary=run_record.task_summary,
                                                    idle_seconds=0.0,
                                                    effective_model=record.effective_model,
                                                    base_model=record.base_model,
                                                    active_provider=record.active_provider,
                                                    effective_thinking=record.effective_thinking,
                                                    result_expired=(agent_id, run_id)
                                                    in self._expired_results,
                                                ),
                                                child_session_id=record.session_id,
                                                parent_identity=(
                                                    record.parent_session_id,
                                                    record.root_generation,
                                                ),
                                            )
                                        )
                                        self._track_child_close(
                                            runtime,
                                            name=(
                                                "vibe-subagent-zero-retention-teardown:"
                                                f"{agent_id}"
                                            ),
                                        )
                                    else:
                                        record.state = _AgentState.IDLE
                                self._expire_results_locked()
                                if not self._draining_children:
                                    self._rearm_reaper_locked()
                            if not self._draining_children:
                                try:
                                    await self._emit_agents_update()
                                except Exception as exc:
                                    logger.warning(
                                        "Failed to publish zero-retention eviction",
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
                if ctx.tool_call_id in self._pending_fan_out_result_leases:
                    self._pending_fan_out_result_leases.remove(ctx.tool_call_id)
                    self._hold_fan_out_result_lease(ctx.tool_call_id, agent_id, run_id)
                if prepared_reconfiguration is not None:
                    assert backend_publication is not None
                    runtime.agent_loop.finalize_launch_reconfiguration(
                        prepared_reconfiguration, backend_publication
                    )
            except BaseException as exc:
                if committed:
                    await self._release_fan_out_result_lease(ctx.tool_call_id)
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
                    await self._rollback_reused_claim(record, previous_idle_since)
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
            # from the terminal result that get_agent_result later returns.
            yield TaskResult(
                response=(
                    "Background agent launched and running; use check_agents to monitor, "
                    "get_agent_result or wait_for_agent to retrieve the result."
                ),
                turns_used=0,
                completed=True,
                status="launched",
                agent_id=agent_id,
                run_id=run_id,
                metadata={
                    "base_model": record.base_model,
                    "active_provider": record.active_provider,
                    "effective_model": record.effective_model,
                },
            )
            return

        if args.agent_id is not None:
            raise ValueError("agent_id is only supported for background tasks")
        self._require_task_profile_allowed(parent, args.agent)
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
                        agent=child.launch_profile or args.agent,
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

    def _agent_summaries(self) -> list[AgentSummary]:
        now = self._clock()
        global_ttl = (
            self._retention_policy[0]
            if self._generation_identity is not None
            else (
                self._root.agent_loop.config.subagents.idle_ttl_seconds
                if self._root
                else 0
            )
        )
        resident = []
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
            resident.append(
                AgentSummary(
                    agent_id=record.agent_id,
                    profile=record.profile,
                    availability=record.availability,
                    current_run_id=run.run_id if run is not None else None,
                    current_run_status=run.status if run is not None else None,
                    turns_used=self._agent_turns_used(record),
                    last_run_status=record.last_run_status,
                    initial_task_summary=record.initial_task_summary,
                    current_task_summary=(
                        run.task_summary
                        if run is not None
                        else record.last_task_summary
                    ),
                    idle_seconds=idle_seconds,
                    ttl_remaining_seconds=(
                        max(0.0, ttl - idle_seconds)
                        if ttl and idle_seconds is not None
                        else None
                    ),
                    effective_model=record.effective_model,
                    base_model=record.base_model,
                    active_provider=record.active_provider,
                    effective_thinking=record.effective_thinking,
                    result_expired=(
                        (record.agent_id, latest_run_id) in self._expired_results
                        if latest_run_id is not None
                        else False
                    ),
                )
            )
        return [
            *resident,
            *(tombstone.summary for tombstone in self._evicted_agents.values()),
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
                    ttl_remaining_seconds=summary.ttl_remaining_seconds,
                    effective_model=summary.effective_model,
                    base_model=summary.base_model,
                    active_provider=summary.active_provider,
                    effective_thinking=summary.effective_thinking,
                    result_expired=summary.result_expired,
                )
                for summary in self._agent_summaries()
            ],
            evictions or [],
        )

    async def check_agents(self) -> list[AgentSummary]:
        async with self._registry_lock:
            self._expire_results_locked()
            return self._agent_summaries()

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
    ) -> TaskResult:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive or None")
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
                return result
        finally:
            async with self._registry_lock:
                leases = self._wait_leases.get(key, 0)
                if leases > 1:
                    self._wait_leases[key] = leases - 1
                else:
                    self._wait_leases.pop(key, None)
                self._expire_results_locked()

    def _eligible_idle_locked(self) -> list[AgentRecord]:
        return [
            record
            for record in self._agent_records.values()
            if record.state is _AgentState.IDLE
            and record.idle_since is not None
            and self._generation_is_current(record.root_generation)
        ]

    def _rearm_reaper_locked(self) -> None:
        if self._draining_children:
            return
        current = asyncio.current_task()
        if self._reaper_task is current:
            return
        if self._reaper_task is not None:
            self._reaper_task.cancel()
        self._reaper_task = None
        ttl, cap = self._retention_policy
        eligible = self._eligible_idle_locked()
        ttl_eligible = [
            record for record in eligible if record.effective_idle_ttl(ttl) > 0
        ]
        if not eligible or (not ttl_eligible and cap == 0):
            return
        if cap and len(eligible) > cap:
            delay = 0.0
        elif ttl_eligible:
            delay = max(
                0.0,
                min(
                    cast(float, record.idle_since) + record.effective_idle_ttl(ttl)
                    for record in ttl_eligible
                )
                - self._clock(),
            )
        else:
            return
        self._reaper_task = asyncio.create_task(
            self._run_reaper(delay), name="vibe-subagent-idle-reaper"
        )

    async def _run_reaper(self, delay: float) -> None:
        try:
            await self._wakeup(delay)
            async with self._registry_lock:
                if asyncio.current_task() is not self._reaper_task:
                    return
                now = self._clock()
                ttl, cap = self._retention_policy
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
                        >= record.effective_idle_ttl(ttl)
                        > 0
                    )
                )
                selected = {agent_id for agent_id, _ in victims}
                remaining = [r for r in eligible if r.agent_id not in selected]
                if cap and len(remaining) > cap:
                    victims.extend(
                        (record.agent_id, "idle_cap")
                        for record in remaining[: len(remaining) - cap]
                    )
            eviction_notifications: list[
                tuple[AgentRecord, AgentEviction, _AgentTombstone]
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
            if eviction_tasks:
                eviction_results = asyncio.gather(
                    *eviction_tasks, return_exceptions=True
                )
                try:
                    await asyncio.shield(eviction_results)
                except asyncio.CancelledError:
                    await asyncio.shield(eviction_results)
                    await self._notify_evictions(eviction_notifications)
                    raise
                await self._notify_evictions(eviction_notifications)
        finally:
            async with self._registry_lock:
                if asyncio.current_task() is self._reaper_task:
                    self._reaper_task = None
                    self._rearm_reaper_locked()

    async def _evict_agent(
        self,
        agent_id: str,
        reason: str,
        notifications: list[tuple[AgentRecord, AgentEviction, _AgentTombstone]]
        | None = None,
    ) -> bool:
        if reason not in {"ttl", "idle_cap"}:
            raise ValueError(f"Unknown eviction reason: {reason}")
        async with self._registry_lock:
            record = self._agent_records.get(agent_id)
            if (
                record is None
                or record.state is not _AgentState.IDLE
                or not self._generation_is_current(record.root_generation)
            ):
                return False
            _, cap = self._retention_policy
            if reason == "idle_cap" and (
                not cap or len(self._eligible_idle_locked()) <= cap
            ):
                return False
            record.state = _AgentState.EVICTING
            self._agent_records.pop(agent_id)
            self._children.pop(record.session_id, None)
            self._child_links.pop(record.session_id, None)
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
            tombstone = _AgentTombstone(
                summary=AgentSummary(
                    agent_id=agent_id,
                    profile=record.profile,
                    availability=AgentAvailability.EVICTED,
                    current_run_id=run.run_id if run is not None else None,
                    current_run_status=run.status if run is not None else None,
                    turns_used=self._agent_turns_used(record),
                    last_run_status=record.last_run_status,
                    initial_task_summary=record.initial_task_summary,
                    current_task_summary=run.task_summary if run is not None else None,
                    idle_seconds=idle_duration,
                    effective_model=record.effective_model,
                    base_model=record.base_model,
                    active_provider=record.active_provider,
                    effective_thinking=record.effective_thinking,
                    result_expired=(agent_id, latest_run_id) in self._expired_results
                    if latest_run_id is not None
                    else False,
                ),
                child_session_id=record.session_id,
                parent_identity=(record.parent_session_id, record.root_generation),
            )
            self._evicted_agents[agent_id] = tombstone
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
            notifications.append((record, eviction, tombstone))
        else:
            await self._notify_evictions([(record, eviction, tombstone)])
        return True

    async def _notify_evictions(
        self, notifications: list[tuple[AgentRecord, AgentEviction, _AgentTombstone]]
    ) -> None:
        permitted: list[tuple[AgentRecord, AgentEviction]] = []
        async with self._registry_lock:
            for record, eviction, tombstone in sorted(
                notifications, key=lambda notification: notification[1].agent_id
            ):
                if (
                    self._generation_is_current(record.root_generation)
                    and not self._draining_children
                    and record.agent_id not in self._suppressed_notifications
                    and self._evicted_agents.get(record.agent_id) is tombstone
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
                details = "; ".join(
                    f"{eviction.agent_id} ({eviction.reason}, idle "
                    f"{eviction.idle_duration_seconds:g}s)"
                    for _record, eviction in parent_notifications
                )
                if len(parent_notifications) == 1:
                    _record, eviction = parent_notifications[0]
                    notification = (
                        f"Background agent {eviction.agent_id} evicted "
                        f"({eviction.reason}, idle {eviction.idle_duration_seconds:g}s); "
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

    async def release_agent(self, agent_id: str) -> ReleaseAgentOutcome:  # noqa: PLR0915
        async with self._registry_lock:  # noqa: PLR1702
            self._acquire_notification_suppression(agent_id)
            record = self._agent_records.pop(agent_id, None)
            known_evicted = self._evicted_agents.pop(agent_id, None) is not None
            release_generation = (
                record.root_generation
                if record is not None
                else self._root.agent_loop._session_generation
                if self._root is not None
                else None
            )
            notify_release = (
                record is not None
                and self._generation_is_current(record.root_generation)
            ) or known_evicted
            keys = {key for key in self._result_store if key[0] == agent_id}
            keys.update(key for key in self._expired_results if key[0] == agent_id)
            keys.update(
                key for key in self._pending_notifications if key[0] == agent_id
            )
            keys.update(key for key in self._wait_leases if key[0] == agent_id)
            keys.update(key for key in self._result_write_tokens if key[0] == agent_id)
            for key in keys:
                self._result_store.pop(key, None)
                self._expired_results.discard(key)
                self._pending_notifications.discard(key)
                self._wait_leases.pop(key, None)
                self._result_write_tokens.pop(key, None)
            self._latest_run_ids.pop(agent_id, None)
            if record is None and not known_evicted and not keys:
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
                                active_turn = record.runtime.turns.active_turn
                                if active_turn is not None:
                                    record.runtime.turns.interrupt(
                                        TurnInterruptParams(
                                            session_id=active_turn.session_id,
                                            expected_turn_id=active_turn.id,
                                        )
                                    )
                                elif current_run.status is RunStatus.RUNNING:
                                    await record.runtime.turns.close()
                            if monitor_tasks:
                                await asyncio.gather(
                                    *monitor_tasks, return_exceptions=True
                                )
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
        return (
            ReleaseAgentOutcome.EVICTED
            if known_evicted
            else ReleaseAgentOutcome.RELEASED
        )

    def _build_child_runtime(
        self,
        child: AgentLoop,
        *,
        base_history: list[PublicHistoryEntry] | None = None,
        event_sink: Callable[[BaseEvent], Awaitable[None]] | None = None,
    ) -> SessionRuntime:
        execution = SessionExecution()
        history = SessionHistory(base_history or [])
        runtime: SessionRuntime | None = None

        def snapshot_state() -> PublicSessionState:
            if runtime is None:
                raise RuntimeError("Child session runtime is not bound")
            return self._public_state(runtime, 200)

        turns = TurnController(
            child,
            self._notify_child,
            self._deliver_callback,
            execution,
            self,
            snapshot_state=snapshot_state,
            tool_io=self._tool_io,
            event_sink=event_sink,
            session_coordinator=self,
        )
        runtime = SessionRuntime(child, turns, execution, history)
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

    async def close_children(self) -> None:
        self._require_policy_unreserved()
        self._admission_closed = True
        await self.drain_children()

    async def drain_children(self) -> None:  # noqa: PLR0912, PLR0915
        self._require_policy_unreserved()
        self._draining_children = True
        errors: list[BaseException] = []
        try:
            reaper = self._reaper_task
            self._reaper_task = None
            if reaper is not None and reaper is not asyncio.current_task():
                reaper.cancel()
                await asyncio.gather(reaper, return_exceptions=True)
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
                self._result_store.clear()
                self._expired_results.clear()
                self._latest_run_ids.clear()
                self._wait_leases.clear()
                self._pending_fan_out_result_leases.clear()
                self._fan_out_result_leases.clear()
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
            if agents_changed:
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

    @staticmethod
    async def _discard_child(child: AgentLoop) -> None:
        with suppress(Exception):
            await close_agent_loop(child)
