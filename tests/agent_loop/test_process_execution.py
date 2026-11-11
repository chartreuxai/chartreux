"""The agent-process wire schema round-trips what the port contract carries."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel
import pytest

from chartreux.app_server.agent_wire import (
    UnsupportedAgentEventError,
    UnsupportedAgentRequestError,
    UnsupportedUserInputResultError,
    decode_event,
    decode_retry_reason,
    decode_transcript_page,
    decode_turn_request,
    decode_user_input_result,
    encode_event,
    encode_retry_reason,
    encode_transcript_page,
    encode_turn_request,
    encode_user_input_result,
)
from chartreux.app_server.protocol import (
    AgentTranscriptGetResponse,
    AgentTranscriptState,
)
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.events import (
    AssistantEvent,
    BackgroundWorkEvent,
    BaseEvent,
    CompactEndEvent,
    CompactStartEvent,
    ReasoningEvent,
    SessionTitleUpdatedEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    ToolWaitStateChangedEvent,
    UserInputRequestEvent,
    UserMessageEvent,
    WaitingForInputEvent,
)
from chartreux.core.hooks.models import HookStartEvent
from chartreux.core.llm_models import ImageAttachment, InlineImageSource
from chartreux.core.utils.retry import RetryCategory, RetryReason
from chartreux.questions import (
    QuestionChoice,
    QuestionKind,
    UnansweredQuestion,
    UnansweredQuestions,
    UnattendedDefault,
    UnattendedDefaultAction,
    UserAnswer,
    UserQuestion,
    UserQuestionRequest,
    UserQuestionResult,
)
from chartreux.utils.tool_presentation import (
    EffectCallDisplay,
    ToolCallPresentation,
    ToolEffectKind,
)


class _ToolArgs(BaseModel):
    path: str


class _ToolClass:
    pass


def _presentation() -> ToolCallPresentation:
    return ToolCallPresentation(
        kind=ToolEffectKind.FILE_READ,
        display=EffectCallDisplay(summary="read a.txt", status_text="reading"),
    )


@pytest.mark.parametrize(
    "event",
    [
        AssistantEvent(content="done", message_id="m1"),
        ReasoningEvent(content="thinking"),
        UserMessageEvent(content="hello", message_id="u1"),
        ToolStreamEvent(tool_name="bash", message="out", tool_call_id="call-1"),
        ToolWaitStateChangedEvent(turn_id="turn-1", waiting_only=False),
        WaitingForInputEvent(task_id="task-1", label="ask"),
        CompactStartEvent(current_context_tokens=10, threshold=20, tool_call_id="c1"),
        CompactEndEvent(summary_length=3, tool_call_id="c1"),
        SessionTitleUpdatedEvent(title="Title", session_id="session-1"),
        BackgroundWorkEvent(
            work_id="w1", kind="session_title", phase="started", session_id="session-1"
        ),
        HookStartEvent(hook_name="hook", tool_call_id="call-1"),
    ],
)
def test_json_safe_events_round_trip(event: BaseEvent) -> None:
    assert decode_event(encode_event(event)) == event


def test_tool_call_event_round_trip_keeps_projection_fields() -> None:
    event = ToolCallEvent(
        tool_call_id="call-1",
        tool_name="read_file",
        tool_class=cast(Any, _ToolClass),
        tool_call_index=2,
        args=_ToolArgs(path="a.txt"),
        presentation=_presentation(),
    )

    decoded = decode_event(encode_event(event))

    assert isinstance(decoded, ToolCallEvent)
    assert decoded.tool_call_id == "call-1"
    assert decoded.tool_name == "read_file"
    assert decoded.tool_call_index == 2
    assert decoded.args == {"path": "a.txt"}
    assert decoded.presentation == _presentation()


def test_tool_result_event_round_trip_keeps_the_json_result() -> None:
    event = ToolResultEvent(
        tool_name="read_file",
        tool_class=None,
        tool_call_id="call-1",
        result=_ToolArgs(path="a.txt"),
        duration=0.5,
    )

    decoded = decode_event(encode_event(event))

    assert isinstance(decoded, ToolResultEvent)
    assert decoded.tool_call_id == "call-1"
    assert decoded.result == {"path": "a.txt"}
    assert decoded.duration == 0.5


def test_user_input_request_event_round_trip_rebuilds_the_args_model() -> None:
    request = UserQuestionRequest(
        questions=[
            UserQuestion(
                question="Which one?",
                options=[QuestionChoice(label="A"), QuestionChoice(label="B")],
            )
        ]
    )
    event = UserInputRequestEvent(
        request_id="req-1", args=request, tool_call_id="call-1"
    )

    decoded = decode_event(encode_event(event))

    assert isinstance(decoded, UserInputRequestEvent)
    assert decoded.args == request


def test_an_undeclared_event_fails_typed() -> None:
    class _UndeclaredEvent(BaseEvent):
        pass

    with pytest.raises(UnsupportedAgentEventError, match="_UndeclaredEvent"):
        encode_event(_UndeclaredEvent())


def test_turn_request_round_trip_carries_user_content() -> None:
    image = ImageAttachment(
        source=InlineImageSource(kind="inline", data="aGk="),
        alias="a.png",
        mime_type="image/png",
    )
    request = ExecutionTurnRequest(
        message="hello",
        client_message_id="client-1",
        auto_title="Title",
        images=[image],
        input_text="typed",
        turn_options=AgentTurnOptions(
            turn_id="turn-1", injected=True, user_initiated_retry=True
        ),
    )

    decoded = decode_turn_request(encode_turn_request(request))

    assert decoded.message == "hello"
    assert decoded.client_message_id == "client-1"
    assert decoded.auto_title == "Title"
    assert decoded.images == [image]
    assert decoded.input_text == "typed"
    assert decoded.turn_options == AgentTurnOptions(
        turn_id="turn-1", injected=True, user_initiated_retry=True
    )


def test_turn_request_refuses_host_side_ports() -> None:
    params = encode_turn_request(ExecutionTurnRequest(message="hello"))
    params["tool_io"] = {"kind": "host-only"}

    with pytest.raises(UnsupportedAgentRequestError, match="cannot cross"):
        decode_turn_request(params)


def test_retry_reason_round_trip() -> None:
    reason = RetryReason(category=RetryCategory.RATE_LIMITED, detail="HTTP 429")

    assert decode_retry_reason(encode_retry_reason(reason)) == reason


def test_user_input_result_round_trip_and_refusal() -> None:
    result = UserQuestionResult(answers=[UserAnswer(question="q", answer="a")])
    default = UnattendedDefault(
        action=UnattendedDefaultAction.DEFER_AFFECTED_WORK,
        description="defer the affected work",
    )
    unanswered = UnansweredQuestions(
        questions=[
            UnansweredQuestion(
                question_id="req-1:0",
                question="Which one?",
                kind=QuestionKind.INFORMATIONAL,
                unattended_default=default,
            )
        ],
        elapsed_seconds=1.0,
        timeout_seconds=60.0,
        continuation_guidance="continue without the answer",
    )

    assert decode_user_input_result(encode_user_input_result(result)) == result
    assert decode_user_input_result(encode_user_input_result(unanswered)) == unanswered

    class _ForeignResult(BaseModel):
        value: int

    with pytest.raises(UnsupportedUserInputResultError, match="_ForeignResult"):
        encode_user_input_result(_ForeignResult(value=1))


def test_transcript_page_round_trip() -> None:
    page = AgentTranscriptGetResponse(state=AgentTranscriptState.NO_SAVED_TRANSCRIPT)

    assert decode_transcript_page(encode_transcript_page(page)) == page
