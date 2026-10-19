from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import aclosing, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
from typing import Literal, cast
from uuid import uuid4

from pydantic import JsonValue

from chartreux.app_server._execution import (
    ActiveSessionExecution,
    SessionExecution,
    SessionExecutionKind,
    cancel_tasks,
)
from chartreux.app_server._model import ProtocolModel
from chartreux.app_server._projection import (
    committed_model_recovery_issue,
    project_stats,
)
from chartreux.app_server._projector import EventProjector, ProjectedUpdate
from chartreux.app_server._root_recovery import (
    RootRecoveryGate,
    RootRecoverySessionView,
)
from chartreux.app_server._root_session import SessionCoordinator, rebind_history
from chartreux.app_server._state import session_preview
from chartreux.app_server._turn_input import vibe_content_blocks
from chartreux.app_server._turn_queue import TurnQueue, TurnQueueEnqueueResult
from chartreux.app_server._utils import (
    DecodedInput,
    decode_content_blocks,
    decode_input,
    now_ms,
    public_error,
)
from chartreux.app_server.models import (
    BlockedEffectState,
    CallbackOutput,
    EffectDetail,
    EffectState,
    PendingEffectState,
    PublicCallbackEntry,
    PublicEffectEntry,
    PublicError,
    PublicHistoryEntry,
    PublicMessageSource,
    PublicRetryCategory,
    PublicRetryState,
    PublicSessionState,
    PublicTurn,
    PublicTurnQueue,
    PublicTurnStatus,
    PublicTurnStopReason,
    RootRecoveryNoticeDetail,
    RunningEffectState,
    ScheduledLoopFiredNoticeDetail,
    UserInputCallbackDetail,
    UserInputCallbackExpiry,
    UserInputCallbackOutput,
    UserQuestionRequest,
)
from chartreux.app_server.protocol import (
    CallbackResultError,
    ContextInjectParams,
    ContextInjectResponse,
    JsonPatchOperation,
    SessionCompactedParams,
    SessionContentBlock,
    SessionSnapshotParams,
    SessionUpdatedParams,
    StatsUpdatedParams,
    TurnCompletedParams,
    TurnContextInputEntry,
    TurnEnqueueParams,
    TurnInterruptParams,
    TurnInterruptResponse,
    TurnQueueUpdatedParams,
    TurnRetryingParams,
    TurnStartedParams,
    TurnStartParams,
    TurnStartResponse,
    TurnSteerParams,
    TurnSteerResponse,
    TurnUpdatedParams,
)
from chartreux.core.agent_loop import AgentLoop, AgentTurnOptions
from chartreux.core.agent_loop._transcript_integrity import (
    TranscriptIntegrityError,
    validate_transcript,
)
from chartreux.core.compaction import reorder_for_tool_adjacency
from chartreux.core.compaction.context import select_model_context
from chartreux.core.events import (
    AssistantEvent,
    BackgroundJobsChangedEvent,
    BackgroundWorkEvent,
    BaseEvent,
    CompactEndEvent,
    ToolWaitStateChangedEvent,
    UserInputOrigin,
    UserInputRequestEvent,
    UserMessageEvent,
)
from chartreux.core.llm_models import ImageAttachment, ManualShellContext
from chartreux.core.model_catalog.availability import (
    AvailabilitySnapshot,
    availability_snapshot,
)
from chartreux.core.recovery_models import build_root_turn_failure
from chartreux.core.recovery_observation import (
    observe_exception,
    record_observation,
    turn_observation,
)
from chartreux.core.subagents import SubagentRunnerPort
from chartreux.core.tools.io_port import ToolIOPort
from chartreux.core.utils.retry import RetryCategory, RetryReason
from chartreux.observability.logging import logger
from chartreux.questions import QuestionKind, UnansweredQuestion, UnansweredQuestions
from chartreux.user_content import UserResource

type Notify = Callable[[str, ProtocolModel], Awaitable[None]]
type DeliverCallback = Callable[[PublicCallbackEntry], Awaitable[None]]
type CoreEventSink = Callable[[BaseEvent], Awaitable[None]]
type BackgroundWorkSink = Callable[
    [BackgroundWorkEvent | BackgroundJobsChangedEvent], Awaitable[None]
]
type SnapshotState = Callable[[], PublicSessionState]

# Retry hooks may cross a worker thread. Context keeps each notice tied to the
# turn whose model request produced it, even if delivery happens later.
_retry_turn_id: ContextVar[str | None] = ContextVar(
    "app_server_retry_turn_id", default=None
)

#: R2: the single persistent harness-effects projector carrying the root
#: recovery milestone notices (deterministic ids; readable via the normal
#: history across a reconnect; never injected as fresh instructions).
_ROOT_RECOVERY_EFFECT_ID = "root-recovery"


class TurnConflictError(RuntimeError):
    pass


class ModelChoicePendingError(RuntimeError):
    """A turn was requested before a recovered session chose a model."""


class StaleTurnError(RuntimeError):
    def __init__(self, active_turn_id: str) -> None:
        self.active_turn_id = active_turn_id
        super().__init__("Active turn does not match expectedTurnId")


class CallbackNotFoundError(RuntimeError):
    pass


class CallbackConflictError(RuntimeError):
    pass


class CallbackClosedError(RuntimeError):
    pass


class CallbackRejectedError(RuntimeError):
    pass


_STEER_RECEIPT_CAPACITY = 128

# The unattended-expiry floor mirrors the [user_questions] config validation
# (both configured timeout bounds are >= 60s); an override outside the
# configured range is rejected, never clamped (plan U/OD1).
_MIN_UNATTENDED_TIMEOUT_SECONDS = 60.0

# The terminal history reason for an expired-unanswered callback (plan U/OD1).
EXPIRED_CALLBACK_REASON = "unanswered_timeout"

# Model-facing continuation guidance carried by the server-owned unanswered
# disposition (plan U/OD1).
UNANSWERED_CONTINUATION_GUIDANCE = (
    "The user did not answer in time, so no choice was made and no permission "
    "was granted. The declared unattended default describes how to proceed "
    "without the answer: defer the affected work or keep the existing "
    "choice. Do not re-ask the same question in a loop; continue with the "
    "deferred outcome, or ask a new, different question if one is still "
    "needed. If the user later provides the answer explicitly, treat it as "
    "a new user message."
)


@dataclass(slots=True)
class SteerReceipt:
    params: TurnSteerParams
    task: asyncio.Task[TurnSteerResponse] | None = None
    event: UserMessageEvent | None = None
    events: list[BaseEvent] = field(default_factory=list)
    pending_updates: list[ProjectedUpdate] = field(default_factory=list)
    recovery_error: BaseException | None = None


@dataclass(slots=True)
class CallbackExpiry:
    """Authoritative unattended-expiry timer state (plan U/OD1).

    Owned by the callback record — not the tool and not the TUI. The policy
    is snapshotted at callback open; the monotonic deadline arbitrates
    against answer/keep-waiting/cancel/reconnect under the record's
    resolution lock, so exactly one terminal disposition wins. A stale timer
    (session or turn generation changed) never fires.
    """

    deadline: float
    opened_at: float
    timeout_seconds: float
    questions: list[UnansweredQuestion]
    session_id: str
    session_generation: int
    turn_id: str
    timer: asyncio.Task[None] | None = None
    kept_waiting: bool = False


@dataclass(slots=True)
class CallbackRecord:
    event: UserInputRequestEvent
    future: asyncio.Future[CallbackOutput | UnansweredQuestions]
    resolution: CallbackOutput | CallbackResultError | None = None
    core_resolved: bool = False
    resolution_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    expiry: CallbackExpiry | None = None
    expired: bool = False


def _public_retry_category(category: RetryCategory) -> PublicRetryCategory:
    match category:
        case RetryCategory.RATE_LIMITED:
            return PublicRetryCategory.RATE_LIMITED
        case RetryCategory.SERVER_ERROR:
            return PublicRetryCategory.SERVER_ERROR
        case RetryCategory.TIMED_OUT:
            return PublicRetryCategory.TIMED_OUT
        case RetryCategory.CONNECTION:
            return PublicRetryCategory.CONNECTION
        case RetryCategory.UNKNOWN:
            return PublicRetryCategory.UNKNOWN


class TurnStartAction:
    def __init__(
        self,
        controller: TurnController,
        turn: PublicTurn,
        session_execution: ActiveSessionExecution,
        run: Callable[[], None],
    ) -> None:
        self._controller = controller
        self._turn = turn
        self._session_execution = session_execution
        self._run = run
        self._resolved = False

    def __call__(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        if self._controller._pending_start is self:
            self._controller._pending_start = None
        try:
            self._run()
        except BaseException:
            self._controller._abort_unstarted_turn(self._turn, self._session_execution)
            raise

    def abort(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        if self._controller._pending_start is self:
            self._controller._pending_start = None
        self._controller._abort_unstarted_turn(self._turn, self._session_execution)

    def cancel(self) -> None:
        """Settle an accepted but unstarted operation; rollback still uses abort."""
        if self._resolved:
            return
        self.abort()
        self._controller._settle_unstarted_cancellation(self._turn)


class TurnController:  # noqa: PLR0904
    def __init__(
        self,
        agent_loop: AgentLoop,
        notify: Notify,
        deliver_callback: DeliverCallback,
        execution: SessionExecution,
        subagent_runner: SubagentRunnerPort,
        *,
        snapshot_state: SnapshotState,
        tool_io: ToolIOPort | None = None,
        event_sink: CoreEventSink | None = None,
        background_work_sink: BackgroundWorkSink | None = None,
        session_coordinator: SessionCoordinator | None = None,
    ) -> None:
        self._agent_loop = agent_loop
        self._notify = notify
        self._deliver_callback = deliver_callback
        self._execution = execution
        self._subagent_runner = subagent_runner
        self._snapshot_state = snapshot_state
        self._tool_io = tool_io
        self._event_sink = event_sink
        self._background_work_sink = background_work_sink
        self._session_coordinator = session_coordinator
        self._session_execution: ActiveSessionExecution | None = None
        self._active_turn: PublicTurn | None = None
        self._retrying: PublicRetryState | None = None
        self._completed_turns: list[PublicTurn] = []
        self._published_completions: set[str] = set()
        self._active_task: asyncio.Task[None] | None = None
        self._pending_start: TurnStartAction | None = None
        self._unstarted_turn_id: str | None = None
        self._projector: EventProjector | None = None
        self._harness_effects: dict[str, EventProjector] = {}
        self._history: list[PublicHistoryEntry] = []
        self._callbacks: dict[str, CallbackRecord] = {}
        # Monotonic clock / wakeup seams for the unattended-expiry timer
        # (plan U/OD1): tests override these per instance.
        self._clock: Callable[[], float] = time.monotonic
        self._wakeup: Callable[[float], Awaitable[None]] = asyncio.sleep
        self._scheduled_loop_id: str | None = None
        self._turn_queue = TurnQueue()
        self._queue_lock = asyncio.Lock()
        self._steer_lock = asyncio.Lock()
        self._steer_receipts: dict[str, SteerReceipt] = {}
        self._steer_replay_floor = 0
        self._steer_evidence_lost = False
        self._steer_tasks: set[asyncio.Task[TurnSteerResponse]] = set()
        self._closing = False
        self._queue_tasks: set[asyncio.Task[None]] = set()
        self._handoff_pending = False
        # R2/OD3: the app-server root recovery gate (authoritative
        # eligibility, episode accounting, single-flight admission). Its
        # config/session/availability providers read LIVE state at every
        # decision; its notices stage into the harness-effects projector.
        self._root_recovery_tasks: set[asyncio.Task[None]] = set()
        self._root_recovery_gate = RootRecoveryGate(
            config=lambda: self._agent_loop.config.recovery.root,
            session_view=self._root_recovery_view,
            availability=self._root_recovery_availability,
            stage_notice=self._stage_root_recovery_notice,
            emit_projected=self._emit_projected,
            # PI1/T (S6): the gate's schedule/admission pre-check validates
            # the live transcript (fail closed on a poisoned transcript).
            transcript_integrity=lambda: self._agent_loop.messages,
        )

    @property
    def root_recovery(self) -> RootRecoveryGate:
        """The R2 root automatic-recovery gate (the wire surface's target)."""
        return self._root_recovery_gate

    @property
    def active_turn(self) -> PublicTurn | None:
        return self._active_turn

    @property
    def retrying(self) -> PublicRetryState | None:
        return self._retrying

    @property
    def completed_turns(self) -> list[PublicTurn]:
        return self._completed_turns.copy()

    @property
    def queue_state(self) -> PublicTurnQueue:
        return self._turn_queue.state

    def require_policy_idle(self) -> None:
        active_task = self._active_task is not None and not self._active_task.done()
        queue_work = self._queue_lock.locked() or any(
            not task.done() for task in self._queue_tasks
        )
        if (
            self._active_turn is not None
            or active_task
            or self._turn_queue
            or self._handoff_pending
            or queue_work
        ):
            raise TurnConflictError("Policy replacement requires idle turns and queue")

    def _require_policy_unreserved(self) -> None:
        active = self._execution.active
        if active is not None and active.id in {"policy-replacement", "configuration"}:
            raise TurnConflictError("Session tree configuration change is in progress")

    def require_model_choice(self) -> None:
        # A session recovered from a missing committed model must not silently
        # fall back to the configured default: an explicit selection (a pinned
        # active_model) is required before any turn can start. Turn-adjacent
        # operations that would run an LLM completion (compaction, queueing a
        # turn) gate on the same pending choice.
        if (
            issue := committed_model_recovery_issue(self._agent_loop)
        ) is not None and not self._agent_loop.config.active_model:
            raise ModelChoicePendingError(issue.message)

    @property
    def has_queued_turns(self) -> bool:
        return bool(self._turn_queue)

    @property
    def turns(self) -> list[PublicTurn]:
        return [
            *self._completed_turns,
            *([self._active_turn] if self._active_turn is not None else []),
        ]

    @property
    def history(self) -> list[PublicHistoryEntry]:
        current = self._projector.history if self._projector is not None else []
        harness_effects = [
            entry
            for projector in self._harness_effects.values()
            for entry in projector.history
        ]
        return [*self._history, *current, *harness_effects]

    @property
    def callbacks(self) -> list[PublicCallbackEntry]:
        if self._projector is None:
            return []
        return [
            entry
            for entry in self._projector.history
            if isinstance(entry, PublicCallbackEntry)
        ]

    def start(
        self,
        params: TurnStartParams,
        *,
        scheduled_loop_id: str | None = None,
        queue_item_id: str | None = None,
        queued_contexts: tuple[DecodedInput, ...] = (),
    ) -> tuple[TurnStartResponse, TurnStartAction]:
        self._require_policy_unreserved()
        self.require_model_choice()
        # PI1/T (S6 turn admission): fail closed BEFORE scheduling or
        # appending — a poisoned transcript would fail the same way at the
        # provider. Typed local diagnostic: never transient, never retried,
        # never scheduled; projected through public_error as an internal
        # error carrying category=transcript_integrity.
        integrity_violations = validate_transcript(
            self._agent_loop.messages, live_tail_exempt=True
        )
        if integrity_violations:
            raise TranscriptIntegrityError(
                integrity_violations, context="turn start admission"
            )
        active_task = self._active_task
        if (
            active_task is not None
            and not active_task.done()
            and active_task is not asyncio.current_task()
        ):
            raise TurnConflictError("A turn is already running")
        if queue_item_id is None and self._turn_queue:
            raise TurnConflictError("Resume queued turns before starting a new turn")
        # Admission is synchronous: reject exclusive lifecycle/shell ownership
        # before assigning live turn state.
        self._execution.require_idle()
        self._retrying = None
        decoded = decode_input(
            params, session_dir=self._agent_loop.session_logger.session_dir
        )
        turn = PublicTurn(
            id=str(uuid4()),
            session_id=params.session_id,
            status=PublicTurnStatus.IN_PROGRESS,
            started_at=now_ms(),
            queue_item_id=queue_item_id,
        )
        session_execution = self._execution.begin(SessionExecutionKind.TURN, turn.id)
        self._session_execution = session_execution
        self._active_turn = turn
        self._scheduled_loop_id = scheduled_loop_id
        self._projector = EventProjector(
            params.session_id,
            turn.id,
            session_preview=session_preview(self._agent_loop),
        )

        def start_turn() -> None:
            retry_turn_token = _retry_turn_id.set(turn.id)
            run = self._run_turn(
                turn,
                decoded.prompt,
                session_execution=session_execution,
                params=params,
                images=decoded.images,
                input_text=(decoded.input_text if decoded.resources else None),
                resources=decoded.resources,
                queued_contexts=queued_contexts,
            )
            try:
                self._unstarted_turn_id = turn.id
                self._active_task = asyncio.create_task(run)
            except BaseException:
                run.close()
                raise
            finally:
                _retry_turn_id.reset(retry_turn_token)
            # R2: feed the start to the recovery gate — an admitted
            # continuation's idempotency key links the successor (and stages
            # the `started` notice); any other start while a schedule is
            # pending is a takeover that cancels it.
            self._observe_root_turn_started(turn.id, params.idempotency_key)

        action = TurnStartAction(self, turn, session_execution, start_turn)
        self._pending_start = action
        return TurnStartResponse(turn=turn), action

    def _abort_unstarted_turn(
        self, turn: PublicTurn, session_execution: ActiveSessionExecution
    ) -> None:
        if self._active_turn is not turn or self._active_task is not None:
            return
        self._active_turn = None
        self._projector = None
        self._retrying = None
        self._scheduled_loop_id = None
        self._session_execution = None
        if self._execution.active is session_execution:
            self._execution.finish(session_execution)

    def enqueue(
        self, params: TurnEnqueueParams
    ) -> tuple[TurnQueueEnqueueResult, Callable[[], None] | None]:
        self._require_policy_unreserved()
        # Fail fast with the same actionable error as a turn start: a queued
        # turn could not promote while the choice is pending, and promotion
        # failures inside queue tasks are silent, so a permissive enqueue
        # would strand the item until an unrelated queue mutation.
        self.require_model_choice()
        result = self._turn_queue.enqueue(params, validate=self._validate_queued_input)
        if result.duplicate:
            return result, None
        return result, self._after_queue_response(promote=True)

    def replace_queued_turn(
        self, queue_item_id: str, params: TurnEnqueueParams
    ) -> tuple[TurnQueueEnqueueResult, Callable[[], None] | None]:
        self._require_policy_unreserved()
        result = self._turn_queue.replace(
            queue_item_id, params, validate=self._validate_queued_input
        )
        if result.duplicate:
            return result, None
        return result, self._after_queue_response(promote=True)

    def _validate_queued_input(self, params: TurnEnqueueParams) -> None:
        for entry in params.entries:
            self._decode_queued_content(entry.content)

    def remove_queued_turn(
        self, queue_item_id: str
    ) -> tuple[bool, Callable[[], None] | None]:
        self._require_policy_unreserved()
        removed = self._turn_queue.remove(queue_item_id)
        if not removed:
            return False, None
        return True, self._after_queue_response(promote=True)

    def resume_queue(self) -> Callable[[], None] | None:
        self._require_policy_unreserved()
        resumed = self._turn_queue.resume()
        # Explicit resume also kicks promotion for pending work in an
        # already-unpaused queue.
        if not resumed and not self._turn_queue:
            return None
        return self._after_queue_response(promote=True)

    async def wait_for_turn(self, turn_id: str) -> PublicTurn:
        task = self._active_task
        if task is None or self._active_turn is None:
            raise StaleTurnError(turn_id)
        if self._active_turn.id != turn_id:
            raise StaleTurnError(self._active_turn.id)
        await task
        completed = next(
            (turn for turn in self._completed_turns if turn.id == turn_id), None
        )
        if completed is None:
            raise RuntimeError(f"Turn did not complete: {turn_id}")
        return completed

    def operation_pending_turn_id(self, initial_turn_id: str) -> str | None:
        """Return the executing/promised turn, or None past execution terminal.

        Completion is recorded before publication and controller teardown. A
        linked successor remains cancellable even before it has an active turn.
        This synchronous query is safe inside the registry's admission lock.
        """
        turn_id = initial_turn_id
        visited: set[str] = set()
        while turn_id not in visited:
            visited.add(turn_id)
            completed = next(
                (turn for turn in self._completed_turns if turn.id == turn_id), None
            )
            if completed is None:
                return turn_id
            if completed.next_turn_id is None:
                return None
            turn_id = completed.next_turn_id
        raise RuntimeError(f"Cyclic turn continuation: {turn_id}")

    def operation_terminal_turn(self, initial_turn_id: str) -> PublicTurn | None:
        """Return the immutable execution outcome, even while teardown is pending."""
        turn_id = initial_turn_id
        visited: set[str] = set()
        while turn_id not in visited:
            visited.add(turn_id)
            completed = next(
                (turn for turn in self._completed_turns if turn.id == turn_id), None
            )
            if completed is None:
                return None
            if completed.next_turn_id is None:
                return completed
            turn_id = completed.next_turn_id
        raise RuntimeError(f"Cyclic turn continuation: {turn_id}")

    def interrupt_operation(self, initial_turn_id: str) -> bool:
        """Interrupt only this operation, including a promised continuation."""
        pending_id = self.operation_pending_turn_id(initial_turn_id)
        if pending_id is None:
            return False
        active = self._active_turn
        if active is not None and active.id == pending_id:
            self.interrupt(
                TurnInterruptParams(
                    session_id=active.session_id, expected_turn_id=active.id
                )
            )
            return True
        predecessor = next(
            (turn for turn in self._completed_turns if turn.next_turn_id == pending_id),
            None,
        )
        execution = self._execution.active
        if predecessor is None or active is not None:
            return False
        if execution is not None and execution.id != pending_id:
            return False
        if self._active_task is not None and not self._active_task.done():
            # The handoff driver owns recovery of the promised successor. It
            # must join its publication/cleanup before wait_for_operation settles.
            self._active_task.cancel()
        else:
            self._settle_unstarted_cancellation(
                PublicTurn(
                    id=pending_id,
                    session_id=predecessor.session_id,
                    status=PublicTurnStatus.IN_PROGRESS,
                    started_at=now_ms(),
                )
            )
        return True

    def _settle_unstarted_cancellation(
        self, turn: PublicTurn, cancelled_task: asyncio.Task[None] | None = None
    ) -> None:
        completed = turn.model_copy(
            update={"status": PublicTurnStatus.INTERRUPTED, "completed_at": now_ms()}
        )
        self._completed_turns.append(completed)

        async def settle() -> None:
            try:
                if cancelled_task is not None:
                    await asyncio.gather(cancelled_task, return_exceptions=True)
                await self._publish_turn_completed(completed)
                await self._emit_status("idle")
            finally:
                execution = self._execution.active
                if execution is not None and execution.id == turn.id:
                    self._execution.finish(execution)
                self._session_execution = None
                self._scheduled_loop_id = None
                self._unstarted_turn_id = None
                try:
                    await self._after_turn_terminal(PublicTurnStatus.INTERRUPTED)
                finally:
                    if self._active_task is asyncio.current_task():
                        self._active_task = None

        self._active_task = asyncio.create_task(settle())

    async def wait_for_operation(self, initial_turn_id: str) -> PublicTurn:
        """Wait for linked turns and their controller teardown, not queued work.

        A handoff driver must establish the successor before its task finishes,
        or record a terminal successor if starting it fails.
        """
        turn_id = initial_turn_id
        visited: set[str] = set()
        while turn_id not in visited:
            visited.add(turn_id)
            task = self._active_task
            active = self._active_turn
            # A prepared start may be observed by an eagerly-started waiter before
            # its action installs the operation task. Yield until that synchronous
            # commit step either starts or aborts the turn.
            while (
                task is None
                and active is not None
                and active.id == turn_id
                and self._pending_start is not None
            ):
                await asyncio.sleep(0)
                task = self._active_task
                active = self._active_turn
            if task is not None and (active is None or active.id == turn_id):
                # Cancellation belongs to the caller's explicit interrupt path.
                await asyncio.shield(task)
            completed = next(
                (turn for turn in self._completed_turns if turn.id == turn_id), None
            )
            if completed is None:
                raise RuntimeError(f"Turn did not complete: {turn_id}")
            if completed.next_turn_id is None:
                return completed
            turn_id = completed.next_turn_id
            # The queue promotion task is scheduled but may not have run yet.
            # Yield until the next turn's active task appears or the turn is
            # found completed.
            while True:
                await asyncio.sleep(0)
                if self._active_task is not None:
                    break
                if any(turn.id == turn_id for turn in self._completed_turns):
                    break
        raise RuntimeError(f"Cyclic turn continuation: {turn_id}")

    async def link_subagent(self, tool_call_id: str, child_session_id: str) -> None:
        projector = self._projector
        if projector is None:
            raise RuntimeError("Cannot link a child session without an active turn")
        for update in projector.link_subagent(tool_call_id, child_session_id):
            await self._emit_projected(update)

    async def replace_subagent(
        self, tool_call_id: str, old_session_id: str, new_session_id: str
    ) -> None:
        projector = self._projector
        if projector is None:
            raise RuntimeError("Cannot replace a child session without an active turn")
        for update in projector.replace_subagent(
            tool_call_id, old_session_id, new_session_id
        ):
            await self._emit_projected(update)

    async def unlink_subagent(self, tool_call_id: str, child_session_id: str) -> None:
        projector = self._projector
        if projector is None:
            return
        for update in projector.unlink_subagent(tool_call_id, child_session_id):
            await self._emit_projected(update)

    async def start_effect(
        self, *, session_id: str, entry_id: str, title: str, detail: EffectDetail
    ) -> None:
        if entry_id in self._harness_effects or any(
            entry.id == entry_id for entry in self._history
        ):
            raise ValueError(f"Duplicate harness effect: {entry_id}")
        projector = EventProjector(session_id, None)
        self._harness_effects[entry_id] = projector
        try:
            await self._emit_projected(
                projector.start_effect(entry_id, title=title, detail=detail)
            )
        except BaseException:
            self._harness_effects.pop(entry_id, None)
            raise

    async def append_effect_output(self, entry_id: str, text: str) -> None:
        projector = self._require_harness_effect(entry_id)
        await self._emit_projected(projector.append_effect_output(entry_id, text))

    async def complete_effect(self, entry_id: str, state: EffectState) -> None:
        projector = self._require_harness_effect(entry_id)
        events = projector.complete_effect(entry_id, state)
        # Retired before the emit, which can fail. The effect is already finished
        # in the projector's own history, so a registration left behind would
        # make `history` report it as still running for the rest of the session
        # - a worse account of a finished effect than a dropped notification.
        self._history.extend(projector.history)
        self._harness_effects.pop(entry_id, None)
        await self._emit_projected(events)

    async def _drain_steer_updates(self, receipt: SteerReceipt) -> None:
        while receipt.pending_updates:
            # Keep the projection until notification succeeds. Replays never
            # project again, but can heal a failed or ambiguous notification.
            await self._emit_projected(receipt.pending_updates[0])
            receipt.pending_updates.pop(0)

    def _prune_steer_receipts(self) -> None:
        # Keep 64 settled, retired deliveries, plus recoverable active/pending
        # work up to a total of 128 receipts. Never evict an in-flight operation.
        active_id = self._active_turn.id if self._active_turn is not None else None
        retired = [
            key
            for key, receipt in self._steer_receipts.items()
            if receipt.params.expected_turn_id != active_id
            and receipt.task is not None
            and receipt.task.done()
            and not receipt.pending_updates
        ]
        for key in retired[:-64]:
            receipt = self._steer_receipts.pop(key)
            for index, turn in enumerate(self._completed_turns):
                if turn.id == receipt.params.expected_turn_id:
                    self._steer_replay_floor = max(self._steer_replay_floor, index + 1)
                    break
        for key, receipt in list(self._steer_receipts.items()):
            if len(self._steer_receipts) < _STEER_RECEIPT_CAPACITY:
                break
            if (
                receipt.params.expected_turn_id != active_id
                and receipt.task is not None
                and receipt.task.done()
            ):
                self._steer_receipts.pop(key)
                # Pending projections may already have reached a client. Losing
                # their content must not become evidence of non-commitment.
                self._steer_evidence_lost = True

    def known_steer_receipt(self, params: TurnSteerParams) -> bool:
        self._prune_steer_receipts()
        receipt = self._steer_receipts.get(params.idempotency_key or "")
        return receipt is not None and receipt.params == params

    def steer_committed(self, params: TurnSteerParams) -> bool | None:
        receipt = self._steer_receipts.get(params.idempotency_key or "")
        if receipt is not None and receipt.params.session_id != params.session_id:
            # Compaction changes the client's session identity, not the original
            # delivery's commitment. A mismatched replay is not proof of rejection.
            return None
        if receipt is None and (
            self._steer_evidence_lost
            or any(
                turn.id == params.expected_turn_id
                for turn in self._completed_turns[: self._steer_replay_floor]
            )
        ):
            # An expired replay window is not proof of rejection. Do not tell
            # an ambiguous client to enqueue context that may already be seen.
            return None
        return (
            receipt is not None
            and receipt.params == params
            and receipt.event is not None
        )

    def _validate_steer(self, params: TurnSteerParams) -> None:
        turn = self._require_active_turn(params.expected_turn_id)
        if not params.require_waiting_only:
            return
        execution = self._execution.active
        if params.session_id != turn.session_id:
            raise TurnConflictError("Steering session does not match the active turn")
        if (
            execution is None
            or execution.kind is not SessionExecutionKind.TURN
            or execution.id != turn.id
        ):
            raise TurnConflictError("Active turn does not own session execution")
        if (
            self._closing
            or self._handoff_pending
            or (self._active_task is not None and self._active_task.cancelling())
            or any(not record.core_resolved for record in self._callbacks.values())
        ):
            raise TurnConflictError("Active turn cannot accept steering")
        if params.require_waiting_only and not self._agent_loop.is_waiting_only(
            turn.id
        ):
            raise TurnConflictError("Active turn is not waiting-only")

    async def steer(self, params: TurnSteerParams) -> TurnSteerResponse:
        async with self._steer_lock:
            self._prune_steer_receipts()
            receipt = self._steer_receipts.get(params.idempotency_key or "")
            if receipt is not None:
                if receipt.params != params:
                    raise TurnConflictError(
                        "Steering idempotency key has a different payload"
                    )
                if (
                    receipt.task is not None
                    and receipt.task.done()
                    and receipt.event is not None
                ):
                    await self._drain_steer_updates(receipt)
                    return TurnSteerResponse()
            else:
                if params.require_waiting_only and not params.idempotency_key:
                    raise TurnConflictError(
                        "Conditional steering requires an idempotency key"
                    )
                self._validate_steer(params)
                if (
                    params.idempotency_key
                    and len(self._steer_receipts) >= _STEER_RECEIPT_CAPACITY
                ):
                    raise TurnConflictError("Steering receipt capacity reached")
                receipt = SteerReceipt(params=params.model_copy(deep=True))
                receipt.task = asyncio.create_task(self._run_steer(receipt))
                self._steer_tasks.add(receipt.task)
                receipt.task.add_done_callback(self._steer_tasks.discard)
                receipt.task.add_done_callback(lambda _: self._prune_steer_receipts())
                # Observe failures even if the transport waiter disconnects.
                receipt.task.add_done_callback(
                    lambda task: task.exception() if not task.cancelled() else None
                )
                if params.idempotency_key:
                    self._steer_receipts[params.idempotency_key] = receipt
            assert receipt.task is not None
        return await asyncio.shield(receipt.task)

    async def _run_steer(self, receipt: SteerReceipt) -> TurnSteerResponse:
        params = receipt.params
        projector = self._projector

        def on_injected_event(event: BaseEvent) -> None:
            # Retain completed expansion work before persistence can fail.
            receipt.events.append(event)
            if projector is not None:
                receipt.pending_updates.extend(
                    projector.project(event, user_message_source="turn_steer")
                )

        def on_commit(event: UserMessageEvent) -> None:
            receipt.event = event
            # Mutate the original turn projection once, before any async recovery
            # or notification failure can retire it or cause a retry.
            on_injected_event(event)

        try:
            decoded = decode_input(
                params, session_dir=self._agent_loop.session_logger.session_dir
            )
            await self._agent_loop.inject_user_context(
                decoded.prompt,
                as_message=True,
                inject_implicit=params.inject_invoked_skill,
                images=decoded.images or None,
                input_text=decoded.input_text if decoded.resources else None,
                resources=decoded.resources or None,
                client_message_id=params.client_user_message_id,
                before_commit=lambda: self._validate_steer(params),
                on_commit=on_commit,
                on_injected_event=on_injected_event,
                on_context_ready=lambda: (
                    self._agent_loop.cancel_outstanding_waits_for_steering(
                        params.expected_turn_id
                    )
                ),
            )
            await self._drain_steer_updates(receipt)
            return TurnSteerResponse()
        except BaseException as exc:
            receipt.recovery_error = exc
            if receipt.event is not None:
                raise RuntimeError(
                    "Steering committed; accepted-operation recovery failed"
                ) from exc
            raise

    def interrupt(self, params: TurnInterruptParams) -> TurnInterruptResponse:
        if self.operation_terminal_turn(params.expected_turn_id) is not None:
            raise StaleTurnError(params.expected_turn_id)
        execution = self._execution.active
        if not (
            self._handoff_pending
            and self._active_turn is None
            and execution is not None
            and execution.kind is SessionExecutionKind.LIFECYCLE
            and (
                execution.id == params.expected_turn_id
                or any(
                    turn.id == params.expected_turn_id
                    and turn.next_turn_id == execution.id
                    for turn in self._completed_turns
                )
            )
        ):
            self._require_active_turn(params.expected_turn_id)
        if self._active_task is not None:
            self._active_task.cancel()
            turn = self._active_turn
            if turn is not None and turn.id == self._unstarted_turn_id:
                self._active_turn = None
                self._projector = None
                self._settle_unstarted_cancellation(turn, self._active_task)
        elif self._pending_start is not None:
            self._pending_start.cancel()
        return TurnInterruptResponse()

    async def inject(
        self,
        params: ContextInjectParams,
        *,
        manual_shell: ManualShellContext | None = None,
    ) -> ContextInjectResponse:
        self._execution.require_idle()
        if self._active_turn is not None:
            raise TurnConflictError("Use turn/steer while a turn is active")
        decoded = decode_input(
            params, session_dir=self._agent_loop.session_logger.session_dir
        )
        projector = EventProjector(
            params.session_id,
            f"injection:{uuid4()}",
            session_preview=session_preview(self._agent_loop),
        )
        events = await self._agent_loop.inject_user_context(
            decoded.prompt,
            as_message=params.as_message,
            inject_implicit=params.inject_invoked_skill,
            images=decoded.images or None,
            input_text=decoded.input_text if decoded.resources else None,
            resources=decoded.resources or None,
            client_message_id=params.client_user_message_id,
            manual_shell=manual_shell,
        )
        for event in events:
            for update in projector.project(event, user_message_source="harness"):
                await self._emit_projected(update)
        self._history.extend(projector.history)
        return ContextInjectResponse(entries=projector.history)

    async def answer_callback(self, callback_id: str, output: CallbackOutput) -> str:
        record = self._callbacks.get(callback_id)
        if record is None:
            raise CallbackNotFoundError(f"Callback not found: {callback_id}")
        async with record.resolution_lock:
            if record.expired:
                # A definite expired response: the unattended timer already
                # resolved this callback (plan U/OD1).
                raise CallbackClosedError(f"Callback expired unanswered: {callback_id}")
            if record.core_resolved and record.resolution is None:
                raise CallbackClosedError(f"Callback is closed: {callback_id}")
            if record.resolution is not None:
                if record.resolution.model_dump(mode="json") == output.model_dump(
                    mode="json"
                ):
                    return "duplicate"
                raise CallbackConflictError("Callback already has a different answer")
            match record.event, output:
                case UserInputRequestEvent(), UserInputCallbackOutput():
                    pass
                case _:
                    raise CallbackConflictError("Callback answer has the wrong type")
            if self._projector is not None:
                for update in self._projector.resolve_callback(callback_id, output):
                    await self._emit_projected(update)
            await self._emit_status("running")
            record.resolution = output
            self._cancel_expiry_timer(record)
            if not record.future.done():
                record.future.set_result(output)
            return "accepted"

    async def reject_callback(
        self, callback_id: str, error: CallbackResultError
    ) -> str:
        record = self._callbacks.get(callback_id)
        if record is None:
            raise CallbackNotFoundError(f"Callback not found: {callback_id}")
        async with record.resolution_lock:
            if record.expired:
                raise CallbackClosedError(f"Callback expired unanswered: {callback_id}")
            if record.core_resolved and record.resolution is None:
                raise CallbackClosedError(f"Callback is closed: {callback_id}")
            if record.resolution is not None:
                if record.resolution.model_dump(mode="json") == error.model_dump(
                    mode="json"
                ):
                    return "duplicate"
                raise CallbackConflictError("Callback already has a different answer")
            record.resolution = error
            self._reject_callback_record(record, CallbackRejectedError(error.message))
            return "accepted"

    async def keep_waiting_callback(
        self, callback_id: str
    ) -> Literal["kept", "already_terminal", "no_deadline"]:
        """Lift the unattended-expiry deadline for a callback (plan U/OD1).

        The question stays open without a deadline (OPEN_WITHOUT_DEADLINE);
        the expiry timer is cancelled under the record's resolution lock so
        the answer/expiry/keep-waiting race still has exactly one winner.
        """
        record = self._callbacks.get(callback_id)
        if record is None:
            raise CallbackNotFoundError(f"Callback not found: {callback_id}")
        async with record.resolution_lock:
            expiry = record.expiry
            if expiry is None:
                return "no_deadline"
            if (
                record.resolution is not None
                or record.core_resolved
                or record.future.done()
            ):
                return "already_terminal"
            expiry.kept_waiting = True
            if expiry.timer is not None:
                expiry.timer.cancel()
                expiry.timer = None
            return "kept"

    async def close(self) -> None:
        self._closing = True
        # R2: invalidate the recovery gate's episodes/projections/admissions
        # (session change / shutdown cancels timers and invalidates
        # references; the failed turns, notices, and history stay retained).
        self._root_recovery_gate.invalidate_all()
        errors: list[BaseException] = []
        if self._root_recovery_tasks:
            errors.extend(
                await cancel_tasks(
                    self._root_recovery_tasks.copy(), label="root recovery notices"
                )
            )
        if self._steer_tasks:
            errors.extend(
                await cancel_tasks(self._steer_tasks.copy(), label="steering")
            )
        if self._pending_start is not None:
            self._pending_start.abort()
        tasks = [*self._queue_tasks]
        if self._active_task is not None:
            tasks.append(self._active_task)
        if tasks:
            errors.extend(await cancel_tasks(tasks, label="turn controller"))
        try:
            await self._clear_retrying()
        except BaseException as exc:
            errors.append(exc)
        if self._session_execution is not None:
            self._execution.finish(self._session_execution)
            self._session_execution = None
        self._scheduled_loop_id = None
        self._cancel_callbacks("App server closed")
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Failed to close turn controller", errors)

    async def reset(self) -> Callable[[], None] | None:
        queue_changed = bool(self._turn_queue) or self._turn_queue.paused
        await self.close()
        self._active_turn = None
        self._completed_turns.clear()
        self._published_completions.clear()
        self._active_task = None
        self._session_execution = None
        self._projector = None
        self._harness_effects.clear()
        self._history.clear()
        self._callbacks.clear()
        self._steer_receipts.clear()
        self._steer_replay_floor = 0
        # Reset discards receipt and turn identities (including expired replay
        # evidence). Unknown retries can no longer prove non-commitment.
        self._steer_evidence_lost = True
        self._closing = False
        self._turn_queue.reset()
        if not queue_changed:
            return None
        return self._after_queue_reset_response(self._agent_loop.session_id)

    async def _run_turn(
        self,
        turn: PublicTurn,
        prompt: str | None,
        *,
        session_execution: ActiveSessionExecution,
        params: TurnStartParams,
        images: list[ImageAttachment],
        input_text: str | None,
        resources: list[UserResource],
        queued_contexts: tuple[DecodedInput, ...],
    ) -> None:
        """Drive both public phases in one operation task, ahead of queued work."""
        self._unstarted_turn_id = None
        successor: PublicTurn | None = None
        status = PublicTurnStatus.FAILED
        try:
            status, error, stop_reason = await self._run_turn_phase(
                turn,
                prompt,
                params=params,
                images=images,
                input_text=input_text,
                resources=resources,
                queued_contexts=queued_contexts,
            )
            await self._finalize_turn(
                turn, status, error, stop_reason, session_execution
            )
        except (asyncio.CancelledError, Exception) as exc:
            status = (
                PublicTurnStatus.INTERRUPTED
                if isinstance(exc, asyncio.CancelledError)
                else PublicTurnStatus.FAILED
            )
            # A continuation can be promised before it has an active PublicTurn.
            # Recover that identity when cancellation hits the handoff gap.
            predecessor = next(
                (item for item in self._completed_turns if item.id == turn.id), None
            )
            if (
                successor is None
                and predecessor is not None
                and predecessor.next_turn_id
            ):
                successor = PublicTurn(
                    id=predecessor.next_turn_id,
                    session_id=predecessor.session_id,
                    status=PublicTurnStatus.IN_PROGRESS,
                    started_at=now_ms(),
                )
            # Recovery is joined even if another interrupt arrives while publishing
            # the predecessor or its promised terminal successor.
            recovery = asyncio.create_task(
                self._recover_failed_operation(turn, successor, status, exc)
            )
            while not recovery.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(recovery)
            recovery.result()
        finally:
            execution = self._execution.active
            if execution is not None and (
                execution.id == turn.id
                or (successor is not None and execution.id == successor.id)
            ):
                self._execution.finish(execution)
            self._session_execution = None
            self._scheduled_loop_id = None
            self._handoff_pending = False
            try:
                await self._after_turn_terminal(status)
            finally:
                if self._active_task is asyncio.current_task():
                    self._active_task = None

    async def _recover_failed_operation(
        self,
        turn: PublicTurn,
        successor: PublicTurn | None,
        status: PublicTurnStatus,
        exc: BaseException,
    ) -> None:
        predecessor = next(
            (item for item in self._completed_turns if item.id == turn.id), None
        )
        if (
            successor is not None
            and predecessor is not None
            and predecessor.next_turn_id == successor.id
        ):
            # Recording the link is not publication. Reconcile it first so an
            # attached act() consumer can route the successor's terminal event.
            try:
                await self._publish_turn_completed(predecessor)
            finally:
                await self._terminalize_failed_turn(successor, status, exc)
        else:
            await self._terminalize_failed_turn(turn, status, exc)

    async def _publish_turn_completed(self, completed: PublicTurn) -> None:
        if completed.id in self._published_completions:
            return

        async def publish() -> None:
            await self._notify(
                "turn/completed",
                TurnCompletedParams(
                    event_id=0,
                    session_id=completed.session_id,
                    turn=completed,
                    emitted_at=now_ms(),
                ),
            )
            self._published_completions.add(completed.id)

        if completed.next_turn_id is None:
            await publish()
            return

        # Join link publication rather than cancelling an ambiguous partial send.
        # The cancelled driver still takes recovery, without publishing it twice.
        publication = asyncio.create_task(publish())
        cancelled = False
        while not publication.done():
            try:
                await asyncio.shield(publication)
            except asyncio.CancelledError:
                cancelled = True
        publication.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _terminalize_failed_turn(
        self, turn: PublicTurn, status: PublicTurnStatus, exc: BaseException
    ) -> None:
        # R2: capture the typed terminal projection and feed the finish
        # transition BEFORE _cancel_callbacks — the interaction/effect
        # barriers are read while the callbacks are still open (their
        # absence afterwards proves nothing).
        self._capture_root_turn_failure(
            turn,
            stop_reason=(
                "user_cancelled" if status is PublicTurnStatus.INTERRUPTED else None
            ),
        )
        await self._observe_root_turn_finished(turn.id, status)
        # Cleanup may itself be the failing operation. Never let a second
        # projector/callback failure prevent the promised terminal notification.
        with suppress(Exception):
            self._cancel_callbacks("Turn ended")
        projector = self._projector
        if projector is not None:
            with suppress(Exception):
                updates = projector.finalize(
                    cancelled=status is PublicTurnStatus.INTERRUPTED,
                    timing=self._agent_loop.completed_turn_timing,
                    messages=self._agent_loop.messages,
                    replay_timing=True,
                )
                for update in updates:
                    with suppress(Exception):
                        await self._emit_projected(update)
            replacements = {entry.id: entry for entry in projector.history}
            existing_ids = {entry.id for entry in self._history}
            self._history = [
                replacements.get(entry.id, entry) for entry in self._history
            ]
            self._history.extend(
                entry for entry in projector.history if entry.id not in existing_ids
            )
        self._active_turn = None
        self._projector = None
        self._retrying = None
        if isinstance(exc, Exception):
            # Preserve the typed failure evidence before the message-only
            # PublicError projection discards it (WP0 F5; the public
            # envelope is WP1's scope — this carrier stays private).
            record_observation(
                observe_exception(exc, phase="turn_terminal", turn_id=turn.id)
            )
        completed = turn.model_copy(
            update={
                "session_id": self._agent_loop.session_id,
                "status": status,
                "completed_at": now_ms(),
                "error": public_error(exc) if isinstance(exc, Exception) else None,
                "next_turn_id": None,
            }
        )
        for index, existing in enumerate(self._completed_turns):
            if existing.id == turn.id:
                self._completed_turns[index] = completed
                break
        else:
            self._completed_turns.append(completed)
        # Record before notification: backpressure or a broken transport must not
        # leave operation waiters following a dangling link.
        await self._publish_turn_completed(completed)
        await self._emit_status("idle")

    async def _run_turn_phase(
        self,
        turn: PublicTurn,
        prompt: str | None,
        *,
        params: TurnStartParams,
        images: list[ImageAttachment],
        input_text: str | None,
        resources: list[UserResource],
        queued_contexts: tuple[DecodedInput, ...],
    ) -> tuple[PublicTurnStatus, PublicError | None, PublicTurnStopReason | None]:
        status = PublicTurnStatus.COMPLETED
        error: PublicError | None = None
        stop_reason: PublicTurnStopReason | None = None
        try:
            last_context_tokens = await self._announce_turn_started(turn)
            await self._inject_queued_contexts(queued_contexts)
            async with aclosing(
                self._agent_loop.act(
                    prompt,
                    client_message_id=params.client_user_message_id,
                    auto_title=params.auto_title,
                    images=images or None,
                    input_text=input_text,
                    resources=resources or None,
                    user_display_content=params.user_display_content,
                    subagent_runner=self._subagent_runner,
                    tool_io=self._tool_io,
                    turn_options=AgentTurnOptions(
                        turn_id=turn.id,
                        retry_sink=self._emit_retrying,
                        injected=params.injected,
                        user_initiated_retry=params.user_initiated_retry,
                    ),
                )
            ) as events:
                async for event in events:
                    if isinstance(event, ToolWaitStateChangedEvent):
                        await self._update_waiting_only(event.turn_id)
                        continue
                    await self._clear_retrying()
                    if self._event_sink is not None:
                        await self._event_sink(event)
                    if isinstance(
                        event, (BackgroundWorkEvent, BackgroundJobsChangedEvent)
                    ):
                        if self._background_work_sink is not None:
                            await self._background_work_sink(event)
                        continue
                    if isinstance(event, UserInputRequestEvent):
                        await self._handle_core_request(event)
                        continue
                    if (
                        isinstance(event, AssistantEvent)
                        and event.stopped_by_middleware
                    ):
                        stop_reason = (
                            PublicTurnStopReason.BUDGET_UNVERIFIABLE
                            if event.middleware_metadata.get("code")
                            == "budget-unverifiable"
                            else PublicTurnStopReason.LIMIT
                        )
                    await self._project_events([event])
                    context_tokens = self._agent_loop.stats.context_tokens
                    if context_tokens != last_context_tokens:
                        last_context_tokens = context_tokens
                        await self._emit_stats()
                    if isinstance(event, UserMessageEvent) and (
                        loop_id := self._scheduled_loop_id
                    ):
                        self._scheduled_loop_id = None
                        await self._emit_scheduled_loop_notice(loop_id)
        except asyncio.CancelledError:
            status = PublicTurnStatus.INTERRUPTED
        except Exception as exc:
            status = PublicTurnStatus.FAILED
            # Preserve the typed failure evidence before the message-only
            # PublicError projection discards it (WP0 F5; the public
            # envelope is WP1's scope — this carrier stays private).
            record_observation(observe_exception(exc, phase="turn", turn_id=turn.id))
            # R2: capture the typed terminal projection for the recovery
            # gate BEFORE public_error() discards the exception context.
            self._capture_root_turn_failure(turn, stop_reason=None)
            error = public_error(exc)
        return status, error, stop_reason

    async def _update_waiting_only(self, turn_id: str) -> None:
        turn = self._active_turn
        if turn is None or turn.id != turn_id:
            return
        try:
            self._validate_steer(
                TurnSteerParams(
                    session_id=turn.session_id,
                    expected_turn_id=turn_id,
                    message=[],
                    require_waiting_only=True,
                )
            )
            waiting_only = True
        except (TurnConflictError, StaleTurnError):
            waiting_only = False
        if turn.waiting_only != waiting_only:
            turn.waiting_only = waiting_only
            await self._notify(
                "turn/updated",
                TurnUpdatedParams(
                    event_id=0,
                    session_id=turn.session_id,
                    turn=turn.model_copy(deep=True),
                    emitted_at=now_ms(),
                ),
            )

    async def _announce_turn_started(self, turn: PublicTurn) -> int:
        await self._notify(
            "turn/started",
            TurnStartedParams(
                event_id=0, session_id=turn.session_id, turn=turn, emitted_at=now_ms()
            ),
        )
        await self._emit_status("running")
        await self._emit_stats()
        return self._agent_loop.stats.context_tokens

    async def _emit_scheduled_loop_notice(self, loop_id: str) -> None:
        projector = self._projector
        turn = self._active_turn
        if projector is None or turn is None:
            raise RuntimeError("Cannot emit a loop notice without an active turn")
        await self._emit_projected(
            projector.add_notice(
                f"scheduled-loop:{turn.id}",
                message=f"Loop `{loop_id}` fired",
                detail=ScheduledLoopFiredNoticeDetail(loop_id=loop_id),
            )
        )

    # ------------------------------------------------------------------
    # R2/OD3 — the root automatic-recovery gate's observation hooks
    # ------------------------------------------------------------------

    def _root_recovery_view(self) -> RootRecoverySessionView:
        """The live interaction state the gate decides on (never snapshotted
        at construction — read at every schedule/admission).
        """
        active = self._execution.active
        return RootRecoverySessionView(
            session_id=self._agent_loop.session_id,
            active_turn_id=(
                self._active_turn.id if self._active_turn is not None else None
            ),
            queued_work=bool(self._turn_queue),
            open_callbacks=any(
                not record.core_resolved for record in self._callbacks.values()
            ),
            stop_requested=bool(
                self._active_task is not None and self._active_task.cancelling()
            ),
            handoff_pending=self._handoff_pending,
            lifecycle_reserved=(
                active is not None and active.kind is SessionExecutionKind.LIFECYCLE
            ),
            closing=self._closing,
        )

    def _root_recovery_availability(self) -> AvailabilitySnapshot:
        """The NON-MUTATING same-base eligibility snapshot (R2): never claims
        or releases a single-flight probe, never records or clears anything.
        """
        loop = self._agent_loop
        config = loop.config
        snapshot = config.catalog_snapshot
        committed = loop.committed_model
        if snapshot is None or committed is None:
            # Missing evidence never blocks scheduling on its own — the
            # admission itself is the authority.
            return AvailabilitySnapshot(
                available=True,
                permanent=False,
                endpoint_blocked=False,
                probe_in_flight=False,
                next_check_delay=0.0,
            )
        return availability_snapshot(
            snapshot=snapshot,
            committed=committed,
            registry=loop.config_orchestrator.availability_registry,
            config=config,
            history=reorder_for_tool_adjacency(select_model_context(loop.messages)),
            thinking=config.get_active_model().thinking,
            compaction_base=config.compaction_model or None,
            allowed_models=config.allowed_models,
        )

    def _stage_root_recovery_notice(
        self, detail: RootRecoveryNoticeDetail, message: str
    ) -> ProjectedUpdate | None:
        """Stage one root-recovery milestone notice into the session history.

        The deterministic entry id (``root-recovery:{milestone}:
        {episode_id}:{attempt}``) makes the emission idempotent at the
        projection layer; the entry rides the normal history (readable after
        an overnight interval and across a reconnect) and is never injected
        as a fresh instruction that would trigger another turn. The projector
        stays registered under one harness-effects key for the session's
        lifetime, so rebind/reset handle it like every other harness effect.
        """
        projector = self._harness_effects.get(_ROOT_RECOVERY_EFFECT_ID)
        if projector is None:
            projector = EventProjector(self._agent_loop.session_id, None)
            self._harness_effects[_ROOT_RECOVERY_EFFECT_ID] = projector
        entry_id = (
            f"root-recovery:{detail.milestone}:{detail.episode_id}:{detail.attempt}"
        )
        if any(entry.id == entry_id for entry in projector.history):
            return None
        return projector.add_notice(entry_id, message=message, detail=detail)

    def _deliver_root_recovery_notices(self, updates: list[ProjectedUpdate]) -> None:
        """Deliver staged notice updates (sync context: the entries are
        already durable in the history; delivery is best-effort).
        """
        if not updates:
            return

        async def deliver() -> None:
            for update in updates:
                with suppress(Exception):
                    await self._emit_projected(update)

        task = asyncio.create_task(deliver())
        self._root_recovery_tasks.add(task)
        task.add_done_callback(self._root_recovery_tasks.discard)

    def _observe_root_turn_started(
        self, turn_id: str, idempotency_key: str | None
    ) -> None:
        """Feed one turn start to the gate (sync context). Total by
        contract: an observation failure must never mask the turn start.
        """
        try:
            updates = self._root_recovery_gate.observe_turn_started(
                turn_id, idempotency_key
            )
        except Exception:
            logger.exception("root recovery turn-start observation failed")
            return
        self._deliver_root_recovery_notices(updates)

    async def _observe_root_turn_finished(
        self, turn_id: str, status: PublicTurnStatus
    ) -> None:
        """Feed one turn's terminal transition to the gate (async context).

        Total by contract: an observation failure must never mask the
        terminalization it observes.
        """
        try:
            await self._root_recovery_gate.observe_turn_finished(turn_id, status)
        except Exception:
            logger.exception("root recovery turn-finish observation failed")

    def _root_interaction_barrier(self) -> tuple[bool, str]:
        """The interaction barrier read BEFORE callback cancellation (R2):
        an open approval/question, queued user work, or unsettled
        predecessor cleanup. Absence of callbacks AFTERWARD proves nothing.
        """
        if any(not record.core_resolved for record in self._callbacks.values()):
            return True, "an approval or user question callback is still open"
        if self._turn_queue:
            return True, "queued user work is pending"
        if self._handoff_pending:
            return True, "predecessor cleanup is still settling"
        return False, ""

    def _root_effects_settled(self) -> bool:
        """Whether every projected tool effect of the turn reached a settled
        state (pending/running/blocked effects are unsettled).
        """
        projector = self._projector
        if projector is None:
            return True
        return not any(
            isinstance(entry, PublicEffectEntry)
            and isinstance(
                entry.state,
                (PendingEffectState, RunningEffectState, BlockedEffectState),
            )
            for entry in projector.history
        )

    def _capture_root_turn_failure(
        self, turn: PublicTurn, *, stop_reason: str | None
    ) -> None:
        """Stash the typed terminal projection for the recovery gate (R2).

        Captured at ``_run_turn_phase``'s except — BEFORE ``public_error()``
        destroys the exception context — and at the top of
        ``_terminalize_failed_turn`` — BEFORE ``_cancel_callbacks`` — so the
        eligibility decision never depends on post-cancellation state. Total
        by contract: a capture failure must never mask the primary failure.
        """
        try:
            barrier, barrier_detail = self._root_interaction_barrier()
            failure = build_root_turn_failure(
                observation=turn_observation(turn.id),
                turn_id=turn.id,
                committed=self._agent_loop.committed_model,
                stop_reason=stop_reason,
                effects_settled=self._root_effects_settled(),
                interaction_barrier=barrier,
                barrier_detail=barrier_detail,
            )
            self._root_recovery_gate.capture_terminal(failure)
        except Exception:
            logger.exception("root recovery projection capture failed")

    async def _finalize_turn(
        self,
        turn: PublicTurn,
        status: PublicTurnStatus,
        error: PublicError | None,
        stop_reason: PublicTurnStopReason | None,
        session_execution: ActiveSessionExecution,
        *,
        next_turn_id: str | None = None,
    ) -> None:
        await self._clear_retrying()
        projector = self._projector
        if projector is not None:
            for update in projector.finalize(
                cancelled=status is PublicTurnStatus.INTERRUPTED,
                timing=self._agent_loop.completed_turn_timing,
                messages=self._agent_loop.messages,
            ):
                await self._emit_projected(update)
            self._history.extend(projector.history)
        # R2: feed the terminal transition to the recovery gate (a COMPLETED
        # successor recovers the episode; an INTERRUPTED one is a user
        # takeover; a FAILED one keeps the episode for the next schedule).
        await self._observe_root_turn_finished(turn.id, status)
        self._cancel_callbacks("Turn ended")
        completed = turn.model_copy(
            update={
                "status": status,
                "completed_at": now_ms(),
                "error": error,
                "stop_reason": stop_reason,
                "next_turn_id": next_turn_id,
                "waiting_only": False,
            }
        )
        self._completed_turns.append(completed)
        if next_turn_id is not None:
            self._active_turn = None
            self._projector = None
            self._session_execution = None
            self._execution.finish(session_execution)
            self._execution.begin(SessionExecutionKind.LIFECYCLE, next_turn_id)
        await self._emit_stats()
        self._active_turn = None
        self._projector = None
        self._session_execution = None
        await self._emit_status("idle")
        self._execution.finish(session_execution)
        self._prune_steer_receipts()
        await self._publish_turn_completed(completed)

    def _after_queue_response(self, *, promote: bool) -> Callable[[], None]:
        def after_response() -> None:
            self._spawn_queue_task(self._after_queue_mutation(promote=promote))

        return after_response

    def _after_queue_reset_response(self, session_id: str) -> Callable[[], None]:
        queue = self.queue_state

        def after_response() -> None:
            self._spawn_queue_task(self._publish_queue_reset(session_id, queue))

        return after_response

    def _spawn_queue_task(self, action: Coroutine[object, object, None]) -> None:
        task = asyncio.create_task(action, name="vibe-turn-queue")
        self._queue_tasks.add(task)
        task.add_done_callback(self._queue_task_finished)

    def _queue_task_finished(self, task: asyncio.Task[None]) -> None:
        self._queue_tasks.discard(task)
        if task.cancelled():
            return
        if error := task.exception():
            asyncio.get_running_loop().call_exception_handler({
                "message": "Turn queue task failed",
                "exception": error,
                "task": task,
            })

    async def _after_queue_mutation(self, *, promote: bool) -> None:
        async with self._queue_lock:
            await self._emit_queue_updated()
            if promote:
                await self._promote_next()

    async def _publish_queue_reset(
        self, session_id: str, queue: PublicTurnQueue
    ) -> None:
        async with self._queue_lock:
            await self._emit_queue_updated(session_id=session_id, queue=queue)

    async def _after_turn_terminal(self, status: PublicTurnStatus) -> None:
        async with self._queue_lock:
            if status is PublicTurnStatus.INTERRUPTED:
                if self._turn_queue.pause():
                    await self._emit_queue_updated()
                return
            await self._promote_next()

    async def _promote_next(self) -> None:
        if (
            self._handoff_pending
            or self._active_turn is not None
            or self._execution.active is not None
        ):
            return
        active_task = self._active_task
        if (
            active_task is not None
            and not active_task.done()
            and active_task is not asyncio.current_task()
        ):
            return
        record = self._turn_queue.peek_next()
        if record is None:
            return
        queued = record.params
        queued_contexts = tuple(
            self._decode_queued_content(entry.content)
            for entry in queued.entries
            if isinstance(entry, TurnContextInputEntry)
        )
        user_entry = queued.user_entry
        if user_entry is None:
            promoted = self._turn_queue.pop_next()
            if promoted is not record:
                raise RuntimeError("Turn queue changed while promoting its next item")
            await self._emit_queue_updated()
            await self._inject_queued_contexts(queued_contexts)
            await self._promote_next()
            return
        _, start_turn = self.start(
            TurnStartParams(
                session_id=self._agent_loop.session_id,
                message=vibe_content_blocks(user_entry.content),
                client_user_message_id=user_entry.entry_id or record.queued_turn.id,
                user_display_content=(
                    user_entry.annotations.chartreux_user_display_content
                ),
            ),
            queue_item_id=record.queued_turn.id,
            queued_contexts=queued_contexts,
        )
        try:
            promoted = self._turn_queue.pop_next()
            if promoted is not record:
                raise RuntimeError("Turn queue changed while promoting its next item")
            await self._emit_queue_updated()
        except BaseException:
            start_turn.abort()
            raise
        start_turn()

    def _decode_queued_content(
        self, content: list[SessionContentBlock]
    ) -> DecodedInput:
        return decode_content_blocks(
            vibe_content_blocks(content),
            session_dir=self._agent_loop.session_logger.session_dir,
        )

    async def _inject_queued_contexts(self, contexts: tuple[DecodedInput, ...]) -> None:
        for decoded in contexts:
            await self._agent_loop.inject_user_context(
                decoded.prompt,
                as_message=False,
                images=decoded.images or None,
                input_text=decoded.input_text if decoded.resources else None,
                resources=decoded.resources or None,
            )

    async def _emit_queue_updated(
        self, *, session_id: str | None = None, queue: PublicTurnQueue | None = None
    ) -> None:
        target_session_id = (
            self._agent_loop.session_id if session_id is None else session_id
        )
        await self._notify(
            "turn_queue_updated",
            TurnQueueUpdatedParams(
                event_id=0,
                session_id=target_session_id,
                queue=queue if queue is not None else self.queue_state,
                emitted_at=now_ms(),
            ),
        )

    async def _handle_core_request(self, event: UserInputRequestEvent) -> None:
        try:
            await self._request_user_input(event)
        except CallbackRejectedError:
            raise
        except Exception as exc:
            record = self._callbacks.get(event.request_id)
            if record is None:
                self._agent_loop.reject_request(event.request_id, exc)
            else:
                self._reject_callback_record(record, exc)
            raise

    async def _request_user_input(self, event: UserInputRequestEvent) -> None:
        if not isinstance(event.args, UserQuestionRequest):
            raise TypeError(
                f"Unsupported user-input request: {type(event.args).__name__}"
            )
        # Admission for unattended expiry (plan U/OD1): the public snapshot
        # rides on the callback detail for clients. An out-of-range timeout
        # override is rejected here, before any callback is opened.
        expiry = self._unattended_expiry(event)
        output = await self._open_callback(
            event,
            UserInputCallbackDetail(
                request=event.args,
                related_entry_id=event.tool_call_id,
                unattended_expiry=expiry,
            ),
            title="User input required",
        )
        record = self._callbacks[event.request_id]
        if isinstance(output, UnansweredQuestions):
            # Server-owned expiry disposition: resolve the core broker request
            # with it and record exactly one deferred-question record. Expiry
            # never approves anything; the tool result IS the unanswered
            # disposition.
            self._agent_loop.resolve_user_input_request(event.request_id, output)
            record.core_resolved = True
            self._agent_loop.deferred_questions.record_unanswered(
                session_id=self._agent_loop.session_id,
                request_id=event.request_id,
                unanswered=output,
            )
            return
        if not isinstance(output, UserInputCallbackOutput):
            raise RuntimeError("Client returned the wrong user-input result")
        self._agent_loop.resolve_user_input_request(event.request_id, output.result)
        record.core_resolved = True

    @staticmethod
    def _echoed_questions(
        request_id: str, args: UserQuestionRequest
    ) -> list[UnansweredQuestion] | None:
        """Echoed questions with stable IDs, or None when not defaultable.

        Every question must be explicitly defaultable (plan U/OD1): a
        requires_user question (or a missing default) blocks the whole batch,
        so a mixed or protected batch never expires.
        """
        echoed: list[UnansweredQuestion] = []
        for index, question in enumerate(args.questions):
            default = question.unattended_default
            if question.kind is QuestionKind.REQUIRES_USER or default is None:
                return None
            echoed.append(
                UnansweredQuestion(
                    question_id=f"{request_id}:{index}",
                    question=question.question,
                    header=question.header,
                    kind=question.kind,
                    unattended_default=default,
                )
            )
        return echoed or None

    def _unattended_expiry(
        self, event: UserInputRequestEvent
    ) -> UserInputCallbackExpiry | None:
        """Admission for unattended expiry (plan U/OD1).

        Returns the public expiry snapshot to attach to the callback detail,
        or None when the request is permanently ineligible. Raises
        ValueError when a model-supplied timeout override does not fit the
        configured range (rejected, never clamped).

        Every gate is provenance- or policy-based, never model-controlled:
        runtime approvals (dangerous-command and root-access grants) share
        the broker and callback plumbing but are permanently ineligible, and
        a requires_user question (or a missing default) blocks the batch.
        """
        args = event.args
        if not isinstance(args, UserQuestionRequest):
            return None
        loop = self._agent_loop
        eligible = (
            # Runtime approvals are permanently ineligible (see docstring).
            event.origin is UserInputOrigin.MODEL_QUESTION
            # Root agent in a supported interactive session only.
            and not loop.is_subagent
            and loop.user_input_capability
            and loop.config.user_questions.unattended_timeout_enabled
        )
        if not eligible:
            return None
        questions = self._echoed_questions(event.request_id, args)
        if questions is None:
            return None
        config = loop.config.user_questions
        # The only actions a default may declare are defer_affected_work and
        # preserve_existing_choice: neither authorizes anything new, and no
        # option label can become an answer through expiry.
        timeout = config.default_timeout_seconds
        # The timeout override is tool-owned (AskUserQuestionArgs); the
        # generic runtime request stays free of it.
        override = getattr(args, "timeout_seconds", None)
        if override is not None:
            if (
                override < _MIN_UNATTENDED_TIMEOUT_SECONDS
                or override > config.max_timeout_seconds
            ):
                raise ValueError(
                    "ask_user_question timeout_seconds must fit the configured "
                    "[user_questions] range "
                    f"[{_MIN_UNATTENDED_TIMEOUT_SECONDS:g}, "
                    f"{config.max_timeout_seconds:g}] seconds; got "
                    f"{override:g} (rejected, not clamped)"
                )
            timeout = override
        return UserInputCallbackExpiry(timeout_seconds=timeout, questions=questions)

    async def _open_callback(
        self,
        event: UserInputRequestEvent,
        detail: UserInputCallbackDetail,
        *,
        title: str,
    ) -> CallbackOutput | UnansweredQuestions:
        projector = self._projector
        turn = self._active_turn
        if projector is None or turn is None:
            raise RuntimeError("Cannot open callback without an active turn")
        callback_id = event.request_id
        record = CallbackRecord(
            event=event, future=asyncio.get_running_loop().create_future()
        )
        if callback_id in self._callbacks:
            raise RuntimeError(f"Duplicate callback request: {callback_id}")
        self._callbacks[callback_id] = record
        for update in projector.open_callback(callback_id, detail, title):
            await self._emit_projected(update)
        entry = next(
            cast(PublicCallbackEntry, entry)
            for entry in projector.history
            if isinstance(entry, PublicCallbackEntry)
            and entry.callback_id == callback_id
        )
        await self._emit_status("blocked", callback=entry)
        await self._deliver_callback(entry)
        # The authoritative unattended-expiry timer lives beside the callback
        # record (plan U/OD1) and is armed only AFTER delivery succeeds: a
        # delivery failure is not user silence. A detached client is a
        # silent no-op delivery, so server-side expiry works detached.
        if detail.unattended_expiry is not None:
            record.expiry = self._arm_expiry(
                callback_id, detail.unattended_expiry, turn.id
            )
        return await record.future

    def _arm_expiry(
        self, callback_id: str, policy: UserInputCallbackExpiry, turn_id: str
    ) -> CallbackExpiry:
        """Arm the authoritative unattended-expiry timer (plan U/OD1)."""
        opened_at = self._clock()
        expiry = CallbackExpiry(
            deadline=opened_at + policy.timeout_seconds,
            opened_at=opened_at,
            timeout_seconds=policy.timeout_seconds,
            questions=policy.questions,
            session_id=self._agent_loop.session_id,
            session_generation=self._agent_loop.session_generation,
            turn_id=turn_id,
        )
        expiry.timer = asyncio.create_task(
            self._expire_callback(callback_id, expiry),
            name=f"app-server-question-expiry:{callback_id}",
        )
        return expiry

    async def _expire_callback(self, callback_id: str, expiry: CallbackExpiry) -> None:
        """Fire the unattended-expiry timer (plan U/OD1).

        Waits out the monotonic deadline, then arbitrates under the callback
        record's resolution lock: answer, expiry, keep-waiting, cancellation,
        and reconnect races have exactly one winner. A stale timer (session
        or turn generation changed) never fires, and there is no timer
        restoration after a restart.
        """
        try:
            await self._wakeup(max(0.0, expiry.deadline - self._clock()))
        except asyncio.CancelledError:
            # Keep-waiting, cancellation, or controller close won the race.
            return
        record = self._callbacks.get(callback_id)
        if record is None:
            return
        async with record.resolution_lock:
            if (
                record.resolution is not None
                or record.core_resolved
                or record.future.done()
                or record.expiry is not expiry
                or expiry.kept_waiting
            ):
                return
            loop = self._agent_loop
            if (
                loop.session_id != expiry.session_id
                or loop.session_generation != expiry.session_generation
            ):
                return
            turn = self._active_turn
            if turn is None or turn.id != expiry.turn_id:
                return
            record.expired = True
            record.core_resolved = True
            unanswered = UnansweredQuestions(
                questions=expiry.questions,
                elapsed_seconds=self._clock() - expiry.opened_at,
                timeout_seconds=expiry.timeout_seconds,
                continuation_guidance=UNANSWERED_CONTINUATION_GUIDANCE,
            )
            # Closure is published BEFORE the future resolves (mirrors
            # answer_callback): the effect-unblock patch must land while the
            # related effect is still running — a completed effect is frozen,
            # and a rejected patch drops the notification, gaps the client's
            # event sequence, and triggers a resync.
            if self._projector is not None:
                for update in self._projector.expire_callback(
                    callback_id, EXPIRED_CALLBACK_REASON
                ):
                    await self._emit_projected(update)
            record.future.set_result(unanswered)
        await self._emit_status("running")

    def _cancel_expiry_timer(self, record: CallbackRecord) -> None:
        expiry = record.expiry
        if expiry is not None and expiry.timer is not None:
            expiry.timer.cancel()
            expiry.timer = None

    async def _project_events(
        self,
        events: list[BaseEvent],
        *,
        user_message_source: PublicMessageSource = "turn_start",
    ) -> None:
        projector = self._projector
        if projector is None:
            return
        for event in events:
            await self._handoff_session_if_needed(projector, event)
            updates = projector.project(event, user_message_source=user_message_source)
            for update in updates:
                await self._emit_projected(update)

    async def _handoff_session_if_needed(
        self, projector: EventProjector, event: BaseEvent
    ) -> None:
        old_session_id = projector.session_id
        new_session_id = self._agent_loop.session_id
        if old_session_id == new_session_id:
            return
        coordinator = self._session_coordinator
        turn = self._active_turn
        if coordinator is None or turn is None:
            raise RuntimeError("Core changed session identity without a coordinator")
        if not isinstance(event, CompactEndEvent):
            raise RuntimeError(
                f"Core changed session identity before {type(event).__name__}"
            )
        await self._adopt_session_identity(old_session_id, event)

    async def _adopt_session_identity(
        self, old_session_id: str, event: CompactEndEvent
    ) -> None:
        coordinator = self._session_coordinator
        if coordinator is None:
            raise RuntimeError("Core changed session identity without a coordinator")
        new_session_id = self._agent_loop.session_id
        turn = self._active_turn
        self._history = rebind_history(self._history, new_session_id)
        if self._projector is not None:
            self._projector.rebind_session(new_session_id)
        if turn is not None:
            turn.session_id = new_session_id
        else:
            for projector in self._harness_effects.values():
                projector.rebind_session(new_session_id)
        self._turn_queue.rebind_session(new_session_id)
        handoff = await coordinator.handoff_active_turn(
            old_session_id,
            current_history=self.history,
            callbacks=self.callbacks,
            active_turn=turn,
            completed_turns=self.completed_turns,
            turn_queue=self.queue_state,
        )
        common = {
            "event_id": 0,
            "session_id": handoff.new_session_id,
            "old_session_id": handoff.old_session_id,
            "state": handoff.state,
            "session_log": handoff.session_log,
            "emitted_at": now_ms(),
        }
        match event:
            case CompactEndEvent():
                await self._notify(
                    "session/compacted",
                    SessionCompactedParams(
                        **common, summary_length=event.summary_length
                    ),
                )
        await self._emit_stats()

    async def _emit_projected(self, update: ProjectedUpdate) -> None:
        await self._notify(update.method, update.params)

    async def _emit_retrying(self, reason: RetryReason) -> None:
        turn = self._active_turn
        if turn is None or _retry_turn_id.get() != turn.id:
            return
        retrying = PublicRetryState(
            turn_id=turn.id,
            category=_public_retry_category(reason.category),
            detail=reason.detail,
        )
        self._retrying = retrying
        await self._emit_retry_snapshot()
        await self._notify(
            "turn/retrying",
            TurnRetryingParams(
                session_id=self._agent_loop.session_id,
                category=retrying.category,
                detail=retrying.detail,
            ),
        )

    async def _clear_retrying(self) -> None:
        if self._retrying is None:
            return
        self._retrying = None
        await self._emit_retry_snapshot()

    async def _emit_retry_snapshot(self) -> None:
        await self.emit_snapshot()

    async def emit_snapshot(
        self, *, include_history: bool = True, include_turns: bool = True
    ) -> None:
        state = self._snapshot_state()
        if not include_history or not include_turns:
            state = state.model_copy(
                update={
                    "history": state.history if include_history else None,
                    "history_before_cursor": (
                        state.history_before_cursor if include_history else None
                    ),
                    "turns": state.turns if include_turns else None,
                }
            )
        await self._notify(
            "session/snapshot",
            SessionSnapshotParams(
                event_id=0,
                session_id=state.session.id,
                state=state,
                emitted_at=now_ms(),
            ),
        )

    async def _emit_stats(self) -> None:
        try:
            context_window = (
                self._agent_loop.config.get_active_model().auto_compact_threshold
            )
        except ValueError:
            context_window = None
        await self._notify(
            "session/statsUpdated",
            StatsUpdatedParams(
                event_id=0,
                session_id=self._agent_loop.session_id,
                stats=project_stats(self._agent_loop),
                context_window=context_window,
                emitted_at=now_ms(),
            ),
        )

    async def _emit_status(
        self, status: str, *, callback: PublicCallbackEntry | None = None
    ) -> None:
        if self._active_turn is None and status != "idle":
            return
        if status == "running":
            value: JsonValue = {
                "type": "running",
                "activeTurnId": cast(PublicTurn, self._active_turn).id,
            }
        elif status == "blocked" and callback is not None:
            value = {
                "type": "blocked",
                "activeTurnId": cast(PublicTurn, self._active_turn).id,
                "callbackId": callback.callback_id,
                "reason": callback.detail.kind,
            }
        else:
            value = {"type": "idle"}
        await self._notify(
            "session/updated",
            SessionUpdatedParams(
                event_id=0,
                session_id=self._agent_loop.session_id,
                patch=[
                    JsonPatchOperation(op="replace", path="/status", value=value),
                    JsonPatchOperation(op="replace", path="/updatedAt", value=now_ms()),
                ],
                emitted_at=now_ms(),
            ),
        )

    def _require_active_turn(self, turn_id: str) -> PublicTurn:
        if self._active_turn is None:
            raise TurnConflictError("No active turn")
        if self._active_turn.id != turn_id:
            raise StaleTurnError(self._active_turn.id)
        return self._active_turn

    def _require_harness_effect(self, entry_id: str) -> EventProjector:
        projector = self._harness_effects.get(entry_id)
        if projector is None:
            raise ValueError(f"Harness effect not found: {entry_id}")
        return projector

    def _cancel_callbacks(self, reason: str) -> None:
        for record in self._callbacks.values():
            self._reject_callback_record(record, CallbackRejectedError(reason))

    def _reject_callback_record(
        self, record: CallbackRecord, error: BaseException
    ) -> None:
        if record.core_resolved:
            return
        record.core_resolved = True
        self._cancel_expiry_timer(record)
        self._agent_loop.reject_request(record.event.request_id, error)
        if not record.future.done():
            record.future.set_exception(error)
