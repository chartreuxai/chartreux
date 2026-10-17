"""Batch tool-runner collaborator for AgentLoop (ALL cluster: batching).

Owns the batch wait registry (the invocation registry keyed
``(turn_id, call_id)``, the waiting-only state, and every wait-registry
operation), the active tool-event queue, the tool tasks and their
completion monitor, and the batch teardown. Also owns the call-event
preparation (``build_tool_call_events``) and the ordering bracket around
a batch (``handle_tool_calls`` -> ``run``). The runner borrows the
transcript recorder and the interaction request broker; the
admission-redaction snapshots stay with the redactor, the root
futures/askers with the decision coordinator, and the single-tool
lifecycle with the executor, all reached through narrow late-bound
callbacks. Live loop values (turn identity, stats, tool manager, harness
files, scrub policy, format handler) are read through getters, so
session reset, manager replacement, and relocation stay visible.

Ordering contracts preserved exactly: one queue per batch with the
existing task count, sentinel placement, and broker binding; the runner
joins the tool tasks before ``act()`` releases its active-turn guard,
including repeated cancellation and stream closure; the teardown order
(join, registry pops, wait-state publication, batch close, queue slot
reset, monitor cancel) is unchanged; the registered-wait vs outer-task
cancellation distinction is preserved in ``is_steering_cancellation``;
no TaskGroup rewrite.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Generator
import contextlib
from dataclasses import dataclass

from chartreux.core.agent_loop._request_broker import InteractionRequestBroker
from chartreux.core.agent_loop._transcript_recorder import TranscriptRecorder
from chartreux.core.agent_loop.errors import AgentLoopStateError
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.events import (
    BaseEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    ToolWaitStateChangedEvent,
)
from chartreux.core.hooks.models import HookEvent
from chartreux.core.llm.format import (
    APIToolFormatHandler,
    FailedToolCall,
    ResolvedMessage,
    ResolvedToolCall,
)
from chartreux.core.llm_models import ToolCall
from chartreux.core.session_types import AgentStats
from chartreux.core.subagents import TaskResult
from chartreux.core.tools import secret_redaction
from chartreux.core.tools.builtins.wait_for_agent import WaitForAgent
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.ui import ToolUIDataAdapter
from chartreux.core.utils import TOOL_ERROR_TAG
from chartreux.observability.logging import logger


@dataclass(slots=True)
class _ToolInvocation:
    task: asyncio.Task[None]
    is_wait: bool
    wait_task: asyncio.Task[TaskResult] | None = None
    steering_cancel_requested: bool = False


class ToolBatchRunner:
    """Own the batch wait registry, the batch queue, and the batch lifecycle.

    The registry maps ``(turn_id, call_id)`` to the invocation's task,
    whether the call is a wait, its registered wait task, and whether
    steering cancellation was requested for it. ``is_steering_cancellation``
    preserves the distinction between cancelling the registered wait task
    and cancelling the outer invocation task: it reports steering only
    when the registry entry was marked and the CURRENT task is not itself
    being cancelled. ``run`` executes one batch on a single queue: the
    per-call events stream first, then the concurrent execution yields
    events in completion order until the monitor's sentinel.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        active_turn_id_getter: Callable[[], str],
        transcript_recorder: TranscriptRecorder,
        request_broker: InteractionRequestBroker,
        stats_getter: Callable[[], AgentStats],
        tool_manager_getter: Callable[[], ToolManager],
        harness_files_getter: Callable[[], HarnessFilesManager],
        scrub_policy_getter: Callable[[], secret_redaction.ScrubPolicy],
        format_handler_getter: Callable[[], APIToolFormatHandler],
        record_tool_call_event: Callable[[ToolCallEvent], ToolCallEvent],
        execute_tool: Callable[
            [ResolvedToolCall],
            AsyncGenerator[ToolResultEvent | ToolStreamEvent | HookEvent],
        ],
        register_admissions: Callable[
            [list[ResolvedToolCall], secret_redaction.ScrubPolicy], None
        ],
        tool_failure_event: Callable[[ResolvedToolCall, str], ToolResultEvent],
        abandon_root_requests: Callable[[], None],
    ) -> None:
        # Borrowed shared ownership: the transcript recorder and the
        # interaction request broker are the loop's own objects, never
        # cloned or replaced.
        self._transcript_recorder = transcript_recorder
        self._request_broker = request_broker
        # Live getters: the turn rotates per act() call; stats is replaced
        # on session reset; the tool manager is replaced by configuration
        # transactions; harness files are replaced on relocation; the scrub
        # policy is re-read at batch admission and per redaction boundary;
        # the format handler is read at each failed-call recording.
        self._active_turn_id_getter = active_turn_id_getter
        self._stats_getter = stats_getter
        self._tool_manager_getter = tool_manager_getter
        self._harness_files_getter = harness_files_getter
        self._scrub_policy_getter = scrub_policy_getter
        self._format_handler_getter = format_handler_getter
        # Narrow late-bound callbacks: call-event recording, single-tool
        # execution, admission registration, monitor failure recording, and
        # root-request abandonment all resolve through the loop at call
        # time, so instance patches of the loop's methods keep
        # intercepting (patch riders: _process_one_tool_call at
        # test_secret_redaction_exfil.py:437/:487 and
        # test_background_jobs.py:367/:435; _recorded_tool_call_event through
        # the redactor seam; _tool_failure_event shared with the executor).
        self._record_tool_call_event = record_tool_call_event
        self._execute_tool = execute_tool
        self._register_admissions = register_admissions
        self._tool_failure_event = tool_failure_event
        self._abandon_root_requests = abandon_root_requests
        self._tool_invocations: dict[tuple[str, str], _ToolInvocation] = {}
        self._waiting_only = False
        self._tool_event_queue: asyncio.Queue[BaseEvent | None] | None = None

    @property
    def invocations(self) -> dict[tuple[str, str], _ToolInvocation]:
        """Read-only view of the invocation registry (loop compatibility)."""
        return self._tool_invocations

    @property
    def tool_event_queue(self) -> asyncio.Queue[BaseEvent | None] | None:
        """Read-only view of the active batch queue (loop compatibility)."""
        return self._tool_event_queue

    @property
    def active(self) -> bool:
        """Whether a tool batch is currently admitted (one queue at a time)."""
        return self._tool_event_queue is not None

    def clear_turn(self, turn_id: str) -> None:
        """Drop the turn's registry entries and reset the waiting-only flag."""
        for key in tuple(self._tool_invocations):
            if key[0] == turn_id:
                del self._tool_invocations[key]
        self._waiting_only = False

    def outstanding_wait_call_ids(self, turn_id: str) -> tuple[str, ...]:
        return tuple(
            call_id
            for (identity, call_id), invocation in self._tool_invocations.items()
            if identity == turn_id and invocation.is_wait and not invocation.task.done()
        )

    def is_waiting_only(self, turn_id: str) -> bool:
        live = [
            invocation
            for (identity, _), invocation in self._tool_invocations.items()
            if identity == turn_id and not invocation.task.done()
        ]
        return bool(live) and all(
            invocation.is_wait
            and not invocation.task.cancelling()
            and not invocation.steering_cancel_requested
            and invocation.wait_task is not None
            and not invocation.wait_task.done()
            and not invocation.wait_task.cancelling()
            for invocation in live
        )

    def cancel_waits_for_steering(self, turn_id: str) -> tuple[str, ...]:
        cancelled = []
        for (identity, call_id), invocation in self._tool_invocations.items():
            wait = invocation.wait_task
            if identity != turn_id or not invocation.is_wait:
                continue
            if invocation.task.done() or invocation.task.cancelling():
                continue
            if (
                invocation.steering_cancel_requested
                or wait is None
                or wait.done()
                or wait.cancelling()
            ):
                continue
            invocation.steering_cancel_requested = True
            wait.cancel()
            cancelled.append(call_id)
        self.publish_wait_state(turn_id)
        return tuple(cancelled)

    def publish_wait_state(self, turn_id: str) -> None:
        if turn_id != self._active_turn_id_getter():
            return
        waiting_only = self.is_waiting_only(turn_id)
        if waiting_only != self._waiting_only:
            self._waiting_only = waiting_only
            if self._tool_event_queue is not None:
                self._tool_event_queue.put_nowait(
                    ToolWaitStateChangedEvent(
                        turn_id=turn_id, waiting_only=waiting_only
                    )
                )

    def register_wait_task(
        self, key: tuple[str, str], task: asyncio.Task[TaskResult] | None
    ) -> None:
        if (invocation := self._tool_invocations.get(key)) is not None:
            invocation.wait_task = task
            self.publish_wait_state(key[0])

    def is_steering_cancellation(self, call_id: str) -> bool:
        invocation = self._tool_invocations.get((
            self._active_turn_id_getter(),
            call_id,
        ))
        task = asyncio.current_task()
        return (
            invocation is not None
            and invocation.steering_cancel_requested
            and task is not None
            and not task.cancelling()
        )

    def build_tool_call_events(
        self, tool_calls: list[ToolCall] | None, emitted_ids: set[str]
    ) -> Generator[ToolCallEvent, None, None]:
        for tc in tool_calls or []:
            if tc.id is None or not tc.function.name:
                continue
            if tc.id in emitted_ids:
                continue

            tool_class = self._tool_manager_getter().available_tools.get(
                tc.function.name
            )
            if tool_class is None:
                continue

            event = ToolCallEvent(
                tool_call_id=tc.id,
                tool_call_index=tc.index,
                tool_name=tc.function.name,
                tool_class=tool_class,
            )
            yield self._record_tool_call_event(
                event.model_copy(
                    update={
                        "presentation": ToolUIDataAdapter(
                            tool_class, harness_files=self._harness_files_getter()
                        ).get_call_presentation(event)
                    }
                )
            )

    async def handle_tool_calls(
        self, resolved: ResolvedMessage
    ) -> AsyncGenerator[BaseEvent]:
        self._transcript_recorder.begin_batch()
        try:
            async for event in self._emit_failed_tool_events(resolved.failed_calls):
                yield event
            if not resolved.tool_calls:
                return

            for batch in [resolved.tool_calls]:
                async with contextlib.aclosing(self.run(batch)) as events:
                    async for event in events:
                        yield event
        finally:
            self._transcript_recorder.end_batch()

    async def run(
        self, tool_calls: list[ResolvedToolCall]
    ) -> AsyncGenerator[BaseEvent]:
        for tool_call in tool_calls:
            event = ToolCallEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                args=tool_call.validated_args,
                tool_call_id=tool_call.call_id,
            )
            event = event.model_copy(
                update={
                    "presentation": ToolUIDataAdapter(
                        tool_call.tool_class, harness_files=self._harness_files_getter()
                    ).get_call_presentation(event)
                }
            )
            event = self._record_tool_call_event(event)
            self._transcript_recorder.record_presentation(event)
            yield event

        async with contextlib.aclosing(self._run_concurrently(tool_calls)) as events:
            async for event in events:
                yield event

    async def _emit_failed_tool_events(
        self, failed_calls: list[FailedToolCall]
    ) -> AsyncGenerator[ToolResultEvent]:
        for failed in failed_calls:
            error_msg = f"<{TOOL_ERROR_TAG}>{failed.tool_name}: {failed.error}</{TOOL_ERROR_TAG}>"
            yield ToolResultEvent(
                tool_name=failed.tool_name,
                tool_class=None,
                error=error_msg,
                tool_call_id=failed.call_id,
            )
            self._stats_getter().tool_calls_failed += 1
            # Same choke point as successful responses: pydantic validation
            # errors echo model-supplied argument values, so the failure text
            # can carry a secret and must be redacted before it joins the
            # message list.
            self._transcript_recorder.append_response(
                failed.call_id,
                self._format_handler_getter().create_failed_tool_response_message(
                    failed, secret_redaction.redact(error_msg)
                ),
            )

    async def _run_concurrently(
        self, tool_calls: list[ResolvedToolCall]
    ) -> AsyncGenerator[BaseEvent]:
        """Execute multiple tool calls concurrently, yielding events as they arrive."""
        queue: asyncio.Queue[BaseEvent | None] = asyncio.Queue()
        if self._tool_event_queue is not None:
            raise AgentLoopStateError("A tool batch is already active")
        self._tool_event_queue = queue
        self._request_broker.bind(queue)

        # Execution retains only names/passthrough; the value snapshot is never
        # bound in a tool task or placed in a queue/event.
        admitted = self._scrub_policy_getter()
        self._register_admissions(tool_calls, admitted)
        turn_id = self._active_turn_id_getter()
        keys = [(turn_id, tc.call_id) for tc in tool_calls]
        tasks = []
        for tc, key in zip(tool_calls, keys, strict=True):
            task = asyncio.create_task(
                self._execute_tool_to_queue(tc, queue, admitted, key)
            )
            tasks.append(task)
            self._tool_invocations[key] = _ToolInvocation(
                task=task, is_wait=issubclass(tc.tool_class, WaitForAgent)
            )

        async def _signal_when_all_done() -> None:
            try:
                results = await self._join_tool_tasks(tasks)
                for tool_call, result in zip(tool_calls, results, strict=True):
                    if not isinstance(result, BaseException):
                        continue
                    if isinstance(result, asyncio.CancelledError):
                        continue
                    error_msg = (
                        f"<{TOOL_ERROR_TAG}>{tool_call.tool_name} failed during "
                        f"pre-tool processing: {result}</{TOOL_ERROR_TAG}>"
                    )
                    logger.warning(
                        "Pre-tool processing failed tool=%s tool_call_id=%s error=%s",
                        tool_call.tool_name,
                        tool_call.call_id,
                        type(result).__name__,
                    )
                    self._stats_getter().tool_calls_failed += 1
                    with secret_redaction.bind_policy(admitted):
                        await queue.put(self._tool_failure_event(tool_call, error_msg))
            finally:
                await queue.put(None)

        monitor = asyncio.create_task(_signal_when_all_done())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            # Closing the event stream must join tools before act() releases its
            # active-turn guard, just as cancellation and normal completion do.
            interrupted = await self._cancel_and_join_tool_tasks(tasks)
            for key in keys:
                self._tool_invocations.pop(key, None)
            self.publish_wait_state(turn_id)
            self._close_tool_batch(queue)
            self._tool_event_queue = None
            if not monitor.done():
                monitor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor
            if interrupted:
                raise asyncio.CancelledError

    async def _execute_tool_to_queue(
        self,
        tc: ResolvedToolCall,
        queue: asyncio.Queue[BaseEvent | None],
        admitted: secret_redaction.ScrubPolicy,
        key: tuple[str, str],
    ) -> None:
        """Run a single tool call under its admission policy, sending events to the queue."""
        try:
            with secret_redaction.bind_policy(admitted):
                async with contextlib.aclosing(self._execute_tool(tc)) as events:
                    async for event in events:
                        await queue.put(event)
        finally:
            self._tool_invocations.pop(key, None)
            self.publish_wait_state(key[0])

    def _close_tool_batch(self, queue: asyncio.Queue[BaseEvent | None]) -> None:
        self._request_broker.unbind(queue)
        # Every waiter is settled when a tool batch ends, so no prompt can
        # still have one; a lingering asker only holds an abandoned request.
        self._abandon_root_requests()

    @staticmethod
    async def _join_tool_tasks(
        tasks: list[asyncio.Task[None]],
    ) -> list[None | BaseException]:
        return await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _cancel_and_join_tool_tasks(tasks: list[asyncio.Task[None]]) -> bool:
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        joined = asyncio.gather(*tasks, return_exceptions=True)
        interrupted = False
        while not joined.done():
            try:
                await asyncio.shield(joined)
            except asyncio.CancelledError:
                interrupted = True
        return interrupted
