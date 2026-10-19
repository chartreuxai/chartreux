from __future__ import annotations

from collections.abc import Callable
from typing import Any

from chartreux.core.agent_loop._transcript_integrity import (
    check_tool_message_admission,
    is_synthetic_tool_placeholder,
)
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
            # No active batch (or a call outside it): anchor the result at
            # its owning assistant call (PI1/T S5) instead of appending at
            # the list end behind a steered user message.
            self._append_anchored_response(call_id, message)
            return
        self._staged_responses.setdefault(index, []).append(message)
        while (
            staged := self._staged_responses.pop(self._next_index, None)
        ) is not None:
            for staged_message in staged:
                self._append_anchored_response(
                    staged_message.tool_call_id or call_id, staged_message
                )
            self._next_index += 1

    def end_batch(self) -> None:
        """Flush staged responses and close the ordering window.

        A call aborted before producing any response leaves a gap here;
        ``repair_missing_responses`` backfills it before the next completion,
        exactly as it did before ordering existed. Staged results are placed
        at their owning assistant call (PI1/T S5), so a message steered into
        the batch's gap no longer separates a call from its result.
        """
        order = self._response_order
        staged = self._staged_responses
        self._response_order = None
        self._staged_responses = {}
        self._next_index = 0
        if order is None:
            return
        for index in sorted(staged):
            for staged_message in staged[index]:
                self._append_anchored_response(
                    staged_message.tool_call_id or "", staged_message
                )

    def _append_anchored_response(self, call_id: str, message: LLMMessage) -> None:
        """Record a tool result at its owning assistant call (PI1/T S5).

        Branch 2 (pinned): the settled result is recorded at its CORRECT
        position — after the owning assistant message, before any steered
        user message — replacing a synthetic placeholder if present. The raw
        transcript then matches what ``reorder_for_tool_adjacency`` already
        produces, so mid-turn steering no longer leaves the unrepairable raw
        order behind. Appending a genuine result where a synthetic placeholder
        exists is the designed replacement, not a violation.

        Call identity is scoped per originating assistant call: the owner is
        the LAST assistant message carrying the call id (tool-call ids are
        reused across turns), and the designed replacement only swaps a
        placeholder inside that owner's own result run.
        """
        if message.role != Role.tool or not call_id:
            self._messages.append(message)
            return
        owner_index = self._find_owning_assistant(call_id)
        if owner_index is None:
            # No anchor exists for this result (the direct-execution seam
            # records results for calls with no recorded assistant
            # message): the append boundary admits the ownerless append
            # unless it would create an unrepairable [user, tool]
            # adjacency; the request-build/save/load/admission seams police
            # the resulting transcript and fail closed on it.
            check_tool_message_admission(self._messages, len(self._messages), message)
            self._messages.append(message)
            return
        owner_calls = {
            tool_call.id
            for tool_call in self._messages[owner_index].tool_calls or []
            if tool_call.id
        }
        insertion = owner_index + 1
        while (
            insertion < len(self._messages)
            and self._messages[insertion].role == Role.tool
            and self._messages[insertion].tool_call_id in owner_calls
        ):
            existing = self._messages[insertion]
            if (
                not is_synthetic_tool_placeholder(message)
                and existing.tool_call_id == call_id
                and is_synthetic_tool_placeholder(existing)
            ):
                # Designed replacement: a genuine result replaces a synthetic
                # TOOL_NO_RESPONSE placeholder for the same call in place, in
                # the owner's own result run (a reused call id under a LATER
                # assistant call never touches an earlier turn's placeholder).
                check_tool_message_admission(self._messages, insertion, message)
                self._messages.replace(insertion, message)
                return
            insertion += 1
        check_tool_message_admission(self._messages, insertion, message)
        if insertion < len(self._messages):
            # An interior insertion invalidates the cursor's boundary proof,
            # even if repeated records make the shifted boundary slot look
            # unchanged.
            self._invalidate_cursor()
        self._messages.insert(insertion, message)

    def _find_owning_assistant(self, call_id: str) -> int | None:
        """The index of the assistant message carrying this tool call."""
        for index in range(len(self._messages) - 1, -1, -1):
            message = self._messages[index]
            if message.role != Role.assistant:
                continue
            for tool_call in message.tool_calls or []:
                if tool_call.id == call_id:
                    return index
        return None

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
        """Backfill responses for tool calls that never produced one.

        PI1/T (the poison amplifier, fixed at the root): the lookahead is
        bounded to the LOGICAL BLOCK — it collects responded tool_call_ids
        past non-tool messages (a steered user message can land between the
        assistant call and its genuine result) and stops at the next
        assistant message carrying tool_calls. Call identity is scoped to
        the originating assistant call. A placeholder is inserted only for a
        call with NO result anywhere in its block (and never when any result
        for the call exists anywhere in the list), so this repair can no
        longer create a duplicate of a genuine result — the PI1 four-message
        shape can no longer be CREATED. The designed dangling-call repair
        (hard crash mid-tool, no genuine result anywhere) still inserts a
        placeholder.
        """
        i = 1
        while i < len(self._messages):  # noqa: PLR1702
            msg = self._messages[i]

            if msg.role == "assistant" and msg.tool_calls:
                expected_responses = len(msg.tool_calls)

                if expected_responses > 0:
                    call_ids = {
                        tool_call.id for tool_call in msg.tool_calls if tool_call.id
                    }
                    responded_ids, insertion_point = self._scan_logical_block(
                        i, call_ids
                    )

                    if len(responded_ids) < expected_responses:
                        for tool_call_data in msg.tool_calls:
                            call_id = tool_call_data.id or ""
                            if call_id in responded_ids:
                                continue
                            if self._has_owned_result(i, call_id):
                                # Defense in depth: a result resolving to THIS
                                # assistant call exists outside the block
                                # (legacy blind-append shape) — never create a
                                # duplicate of it. A result recorded for a
                                # reused call id under a different assistant
                                # call does not count.
                                continue

                            empty_response = LLMMessage(
                                role=Role.tool,
                                tool_call_id=call_id,
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

                    i += 1
                    continue

            i += 1

    def _scan_logical_block(
        self, assistant_index: int, call_ids: set[str]
    ) -> tuple[set[str], int]:
        """Collect responded call ids and the insertion point for one block.

        The block runs from the assistant message to the next assistant
        message carrying tool_calls. Responded ids are collected past
        non-tool messages (a steered user message can sit between the call
        and its genuine result); the insertion point stays at the end of the
        contiguous tool run so backfilled placeholders keep call order.
        """
        responded: set[str] = set()
        insertion_point = assistant_index + 1
        j = assistant_index + 1
        while j < len(self._messages):
            following = self._messages[j]
            if following.role == "assistant" and following.tool_calls:
                break
            if following.role == "tool":
                if following.tool_call_id in call_ids:
                    responded.add(following.tool_call_id)
                if insertion_point == j:
                    insertion_point = j + 1
            j += 1
        return responded, insertion_point

    def _has_owned_result(self, assistant_index: int, call_id: str) -> bool:
        """Whether any tool result for this call belongs to this assistant call.

        Scoped per originating assistant call: the owner of a result is the
        last assistant message carrying the call id before it. A result
        recorded for a reused call id under a LATER assistant call belongs to
        that later call and does not block this one's placeholder; any result
        resolving to THIS call — inside or outside its logical block (the
        legacy blind-append shape appends at the list end) — does.
        """
        last_owner: dict[str, int] = {}
        for position, message in enumerate(self._messages):
            if message.role == Role.assistant:
                for tool_call in message.tool_calls or []:
                    if tool_call.id:
                        last_owner[tool_call.id] = position
            elif message.role == Role.tool and message.tool_call_id == call_id:
                if last_owner.get(call_id) == assistant_index:
                    return True
        return False

    def fence_transcript(self) -> None:
        """Invalidate the transcript cursor (the rewind fence's save half)."""
        self._invalidate_cursor()
