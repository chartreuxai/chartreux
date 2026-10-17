from __future__ import annotations

from collections.abc import Callable
from typing import Any

from chartreux.core.agent_loop.llm_gateway import TranscriptAppend
from chartreux.core.events import ToolCallEvent
from chartreux.core.llm.format import APIToolFormatHandler, ResolvedToolCall
from chartreux.core.llm_models import (
    ImageAttachment,
    LLMMessage,
    PersistedToolResult,
    Role,
    ToolCall,
)
from chartreux.core.message_list import MessageList
from chartreux.core.tools import secret_redaction
from chartreux.core.utils import CancellationReason, get_user_cancellation_message


class TranscriptRecorder:
    """Own tool-response ordering state for the transcript message list.

    Borrows the authoritative MessageList (never cloned or replaced); owns
    the response-order map, staging buffers, and next index. While a batch is
    active, response messages are staged and appended in tool-call index
    order even though their events keep streaming in completion order.

    Also owns the recording half of transcript construction: sanitized
    tool-call copies, tool-response messages, and committed completion
    appends. The loop supplies the raw execution message and the prose note
    through synchronous callbacks, and transcript-cursor invalidation through
    a late-bound callback, so the recorder never reaches back into the loop.
    """

    def __init__(
        self,
        *,
        messages: MessageList,
        scrub_policy_getter: Callable[[], secret_redaction.ScrubPolicy],
        format_handler_getter: Callable[[], APIToolFormatHandler],
        set_execution_message: Callable[[LLMMessage], None],
        note_prose: Callable[[LLMMessage], None],
        invalidate_cursor: Callable[[], None],
    ) -> None:
        self._messages = messages
        # Live getters and late-bound callbacks: the scrub policy is
        # reassigned on reload and policy replacement (even mid-turn), and
        # the format handler, session logger, and raw execution-message slot
        # are all resolved at call time, never captured at construction.
        self._scrub_policy_getter = scrub_policy_getter
        self._format_handler_getter = format_handler_getter
        self._set_execution_message = set_execution_message
        self._note_prose = note_prose
        self._invalidate_cursor = invalidate_cursor
        self._response_order: dict[str, int] | None = None
        self._staged_responses: dict[int, list[LLMMessage]] = {}
        self._next_index = 0

    @property
    def response_order(self) -> dict[str, int] | None:
        """Read-only compatibility view of the active ordering window."""
        return self._response_order

    @property
    def staged_responses(self) -> dict[int, list[LLMMessage]]:
        """Read-only compatibility view of the staged response buffers."""
        return self._staged_responses

    def begin_batch(self) -> None:
        """Stage tool-response messages so they append in tool-call order.

        Providers require ``tool_response`` messages to follow the order of
        the ``tool_calls`` that produced them, but batched tools complete out
        of order and their events must keep streaming in completion order.
        While ordering is active, ``append_response`` buffers each message and
        appends index ``i`` only once index ``i - 1`` has appended, so
        streaming stays responsive and the message list stays deterministic.
        """
        ordered_calls: list[ToolCall] = []
        for message in reversed(self._messages):
            if message.role == Role.assistant and message.tool_calls:
                ordered_calls = list(message.tool_calls)
                break
        order: dict[str, int] = {}
        for index, tool_call in enumerate(ordered_calls):
            if tool_call.id and tool_call.id not in order:
                order[tool_call.id] = index
        self._response_order = order
        self._staged_responses = {}
        self._next_index = 0

    def append_response(self, call_id: str, message: LLMMessage) -> None:
        """Append a tool-response message, in call order during a batch."""
        order = self._response_order
        index = order.get(call_id) if order is not None else None
        if index is None:
            # No active batch (or a call outside it): append directly.
            self._messages.append(message)
            return
        self._staged_responses.setdefault(index, []).append(message)
        while (
            staged := self._staged_responses.pop(self._next_index, None)
        ) is not None:
            self._messages.extend(staged)
            self._next_index += 1

    def end_batch(self) -> None:
        """Flush staged responses and close the ordering window.

        A call aborted before producing any response leaves a gap here;
        ``repair_missing_responses`` backfills it before the next completion,
        exactly as it did before ordering existed.
        """
        order = self._response_order
        staged = self._staged_responses
        self._response_order = None
        self._staged_responses = {}
        self._next_index = 0
        if order is None:
            return
        for index in sorted(staged):
            self._messages.extend(staged[index])

    def record_call(self, tool_call: ToolCall) -> ToolCall:
        """Return the sanitized transcript copy of a tool call."""
        with secret_redaction.bind_policy(self._scrub_policy_getter()):
            return secret_redaction.sanitize_recorded_tool_call(tool_call)

    def record_call_event(self, event: ToolCallEvent) -> ToolCallEvent:
        """Return the sanitized transcript copy of a tool-call event."""
        with secret_redaction.bind_policy(self._scrub_policy_getter()):
            return secret_redaction.sanitize_recorded_tool_call(event)

    def record_presentation(self, event: ToolCallEvent) -> None:
        """Copy the event's presentation onto the recorded tool call."""
        if event.presentation is None:
            return
        for message in reversed(self._messages):
            for tool_call in message.tool_calls or []:
                if tool_call.id == event.tool_call_id:
                    tool_call.presentation = event.presentation
                    return

    def record_response(
        self,
        tool_call: ResolvedToolCall,
        text: str,
        persisted_result: PersistedToolResult | None = None,
        images: list[ImageAttachment] | None = None,
    ) -> None:
        # Single choke point for every tool response entering the message
        # list: redacting here covers both model input and the persisted
        # session log, including the structured tool_result payload carried
        # alongside.
        text = secret_redaction.redact(text)
        if persisted_result is not None:
            persisted_result = secret_redaction.redact_persisted_result(
                persisted_result
            )
        message = LLMMessage.model_validate(
            self._format_handler_getter().create_tool_response_message(tool_call, text)
        )
        updates: dict[str, Any] = {}
        if persisted_result is not None:
            updates["tool_result"] = persisted_result
        if images is not None:
            updates["images"] = images
        self.append_response(
            tool_call.call_id,
            message.model_copy(update=updates) if updates else message,
        )

    def record_completion(self, outcome: TranscriptAppend) -> None:
        """Append a committed completion, retaining the raw execution copy.

        Only the active completion retains execution arguments: the raw
        message is supplied to the loop through the synchronous
        ``set_execution_message`` callback, then the sanitized copy is
        appended and transcript copies are never used for execution.
        """
        message = outcome.message
        if outcome.committed_model is not None:
            message = message.model_copy(
                update={"deployment_identity": outcome.committed_model.model_dump()}
            )
        if message.tool_calls:
            self._set_execution_message(message)
            message = message.model_copy(
                update={
                    "tool_calls": [self.record_call(tc) for tc in message.tool_calls]
                }
            )
        self._messages.append(message)
        self._note_prose(message)

    def repair_missing_responses(self) -> None:
        """Backfill responses for tool calls that never produced one."""
        i = 1
        while i < len(self._messages):  # noqa: PLR1702
            msg = self._messages[i]

            if msg.role == "assistant" and msg.tool_calls:
                expected_responses = len(msg.tool_calls)

                if expected_responses > 0:
                    responded_ids: set[str] = set()
                    j = i + 1
                    while j < len(self._messages) and self._messages[j].role == "tool":
                        tool_call_id = self._messages[j].tool_call_id
                        if tool_call_id is not None:
                            responded_ids.add(tool_call_id)
                        j += 1

                    if len(responded_ids) < expected_responses:
                        insertion_point = j

                        for tool_call_data in msg.tool_calls:
                            if (tool_call_data.id or "") in responded_ids:
                                continue

                            empty_response = LLMMessage(
                                role=Role.tool,
                                tool_call_id=tool_call_data.id or "",
                                name=(
                                    (tool_call_data.function.name or "")
                                    if tool_call_data.function
                                    else ""
                                ),
                                content=str(
                                    get_user_cancellation_message(
                                        CancellationReason.TOOL_NO_RESPONSE
                                    )
                                ),
                            )

                            if insertion_point < len(self._messages):
                                # An interior insertion invalidates the cursor's
                                # boundary proof, even if repeated records make
                                # the shifted boundary slot look unchanged.
                                self._invalidate_cursor()
                            self._messages.insert(insertion_point, empty_response)
                            insertion_point += 1

                    i = i + 1 + expected_responses
                    continue

            i += 1

    def fence_transcript(self) -> None:
        """Invalidate the transcript cursor (the rewind fence's save half)."""
        self._invalidate_cursor()
