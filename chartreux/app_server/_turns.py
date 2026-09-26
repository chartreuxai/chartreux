from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import aclosing, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import cast
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
    CallbackOutput,
    EffectDetail,
    EffectState,
    PublicCallbackEntry,
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
    ScheduledLoopFiredNoticeDetail,
    UserInputCallbackDetail,
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
)
from chartreux.core.agent_loop import AgentLoop, AgentTurnOptions
from chartreux.core.events import (
    AssistantEvent,
    BackgroundWorkEvent,
    BaseEvent,
    CompactEndEvent,
    UserInputRequestEvent,
    UserMessageEvent,
)
from chartreux.core.llm_models import ImageAttachment, ManualShellContext
from chartreux.core.subagents import SubagentRunnerPort
from chartreux.core.tools.io_port import ToolIOPort
from chartreux.core.utils.retry import RetryCategory, RetryReason
from chartreux.user_content import UserResource

type Notify = Callable[[str, ProtocolModel], Awaitable[None]]
type DeliverCallback = Callable[[PublicCallbackEntry], Awaitable[None]]
type CoreEventSink = Callable[[BaseEvent], Awaitable[None]]
type BackgroundWorkSink = Callable[[BackgroundWorkEvent], Awaitable[None]]
type SnapshotState = Callable[[], PublicSessionState]

# Retry hooks may cross a worker thread. Context keeps each notice tied to the
# turn whose model request produced it, even if delivery happens later.
_retry_turn_id: ContextVar[str | None] = ContextVar(
    "app_server_retry_turn_id", default=None
)


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


@dataclass(slots=True)
class CallbackRecord:
    event: UserInputRequestEvent
    future: asyncio.Future[CallbackOutput]
    resolution: CallbackOutput | CallbackResultError | None = None
    core_resolved: bool = False
    resolution_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


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
        self._projector: EventProjector | None = None
        self._harness_effects: dict[str, EventProjector] = {}
        self._history: list[PublicHistoryEntry] = []
        self._callbacks: dict[str, CallbackRecord] = {}
        self._scheduled_loop_id: str | None = None
        self._turn_queue = TurnQueue()
        self._queue_lock = asyncio.Lock()
        self._queue_tasks: set[asyncio.Task[None]] = set()
        self._handoff_pending = False

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
        active_task = self._active_task
        if (
            active_task is not None
            and not active_task.done()
            and active_task is not asyncio.current_task()
        ):
            raise TurnConflictError("A turn is already running")
        if queue_item_id is None and self._turn_queue:
            raise TurnConflictError("Resume queued turns before starting a new turn")
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
                self._active_task = asyncio.create_task(run)
            except BaseException:
                run.close()
                raise
            finally:
                _retry_turn_id.reset(retry_turn_token)

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
        if not self._turn_queue.resume():
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

    async def steer(self, params: TurnSteerParams) -> TurnSteerResponse:
        self._require_active_turn(params.expected_turn_id)
        decoded = decode_input(
            params, session_dir=self._agent_loop.session_logger.session_dir
        )
        events = await self._agent_loop.inject_user_context(
            decoded.prompt,
            as_message=True,
            inject_implicit=params.inject_invoked_skill,
            images=decoded.images or None,
            input_text=decoded.input_text if decoded.resources else None,
            resources=decoded.resources or None,
            client_message_id=params.client_user_message_id,
        )
        await self._project_events(events, user_message_source="turn_steer")
        return TurnSteerResponse()

    def interrupt(self, params: TurnInterruptParams) -> TurnInterruptResponse:
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
        elif self._pending_start is not None:
            self._pending_start.abort()
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

    async def close(self) -> None:
        errors: list[BaseException] = []
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
        # Cleanup may itself be the failing operation. Never let a second
        # projector/callback failure prevent the promised terminal notification.
        with suppress(Exception):
            self._cancel_callbacks("Turn ended")
        projector = self._projector
        if projector is not None:
            with suppress(Exception):
                projector.finalize(cancelled=status is PublicTurnStatus.INTERRUPTED)
            existing_ids = {entry.id for entry in self._history}
            self._history.extend(
                entry for entry in projector.history if entry.id not in existing_ids
            )
        self._active_turn = None
        self._projector = None
        self._retrying = None
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
                        retry_sink=self._emit_retrying,
                        injected=params.injected,
                        user_initiated_retry=(
                            params.user_initiated_retry or params.injected
                        ),
                    ),
                )
            ) as events:
                async for event in events:
                    await self._clear_retrying()
                    if self._event_sink is not None:
                        await self._event_sink(event)
                    if isinstance(event, BackgroundWorkEvent):
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
                        stop_reason = PublicTurnStopReason.LIMIT
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
            error = public_error(exc)
        return status, error, stop_reason

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
                cancelled=status is PublicTurnStatus.INTERRUPTED
            ):
                await self._emit_projected(update)
            self._history.extend(projector.history)
        self._cancel_callbacks("Turn ended")
        completed = turn.model_copy(
            update={
                "status": status,
                "completed_at": now_ms(),
                "error": error,
                "stop_reason": stop_reason,
                "next_turn_id": next_turn_id,
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
                client_user_message_id=user_entry.entry_id,
                user_display_content=(
                    user_entry.annotations.chartreux_user_display_content
                ),
            ),
            queue_item_id=record.queued_turn.id,
            queued_contexts=queued_contexts,
        )
        promoted = self._turn_queue.pop_next()
        if promoted is not record:
            raise RuntimeError("Turn queue changed while promoting its next item")
        await self._emit_queue_updated()
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
        output = await self._open_callback(
            event,
            UserInputCallbackDetail(
                request=event.args, related_entry_id=event.tool_call_id
            ),
            title="User input required",
        )
        if not isinstance(output, UserInputCallbackOutput):
            raise RuntimeError("Client returned the wrong user-input result")
        self._agent_loop.resolve_user_input_request(event.request_id, output.result)
        self._callbacks[event.request_id].core_resolved = True

    async def _open_callback(
        self,
        event: UserInputRequestEvent,
        detail: UserInputCallbackDetail,
        *,
        title: str,
    ) -> CallbackOutput:
        projector = self._projector
        if projector is None or self._active_turn is None:
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
        return await record.future

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
            context_window = 0
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
        self._agent_loop.reject_request(record.event.request_id, error)
        if not record.future.done():
            record.future.set_exception(error)
