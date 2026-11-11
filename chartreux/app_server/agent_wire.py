"""Wire codec for the agent-process turn protocol.

The root agent runs in its own process behind the agent-side execution port.
This module is the closed schema between that process and its host: turn
requests travel one way, turn events, retry notices, terminal outcomes, and
transcript pages travel the other. Only declared frame payloads are accepted;
an event or request the schema does not name fails typed instead of being
guessed at.

Two core event models carry process-local values that cannot cross a pipe:
``ToolCallEvent`` and ``ToolResultEvent`` hold the tool's Python class, and
tool arguments and results are typed models whose classes live with the tool.
The wire carries the tool name and the JSON form of those values, which is
exactly what the coordinator-side projection reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, JsonValue, TypeAdapter

from chartreux.app_server.protocol import AgentTranscriptGetResponse
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.events import (
    AssistantEvent,
    BackgroundJobsChangedEvent,
    BackgroundWorkEvent,
    BaseEvent,
    CompactEndEvent,
    CompactStartEvent,
    ReasoningEvent,
    RouteTableAppliedEvent,
    SessionTitleUpdatedEvent,
    TokenUsageUpdatedEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    ToolWaitStateChangedEvent,
    UserInputRequestEvent,
    UserMessageEvent,
    WaitingForInputEvent,
)
from chartreux.core.hooks.models import (
    HookEndEvent,
    HookRunEndEvent,
    HookRunStartEvent,
    HookStartEvent,
)
from chartreux.core.llm_models import ImageAttachment
from chartreux.core.utils.retry import RetryCategory, RetryReason
from chartreux.questions import (
    UnansweredQuestions,
    UnattendedExpiryPolicy,
    UserQuestionRequest,
    UserQuestionResult,
)
from chartreux.user_content import UserDisplayContent, UserResource
from chartreux.utils.tool_presentation import (
    ToolCallPresentation,
    ToolResultPresentation,
)

AGENT_OPEN_METHOD = "agent/open"
SUBAGENT_OPEN_METHOD = "subagent/open"
AGENT_TURN_START_METHOD = "agent/turn/start"
AGENT_TURN_EVENT_METHOD = "agent/turn/event"
AGENT_TURN_RETRYING_METHOD = "agent/turn/retrying"
AGENT_TURN_ENDED_METHOD = "agent/turn/ended"
AGENT_TURN_RESOLVE_METHOD = "agent/turn/resolve"
AGENT_TURN_REJECT_METHOD = "agent/turn/reject"
AGENT_TURN_RELAY_METHOD = "agent/turn/relay"
AGENT_TURN_ANSWER_METHOD = "agent/turn/answer"
AGENT_TURN_REFUSE_METHOD = "agent/turn/refuse"
AGENT_TURN_KEEP_WAITING_METHOD = "agent/turn/keepWaiting"
AGENT_TURN_CANCEL_METHOD = "agent/turn/cancel"
AGENT_TRANSCRIPT_PAGE_METHOD = "agent/transcript/page"

# The cross-process policy-tree phases (WP5 [5D]): one node serves these
# frames, and the coordinator-side transaction drives them in order.
POLICY_REVISION_METHOD = "policy/revision"
POLICY_RESERVE_METHOD = "policy/reserve"
POLICY_PREPARE_METHOD = "policy/prepare"
POLICY_COMMIT_METHOD = "policy/commit"
POLICY_RETIRE_METHOD = "policy/retire"
POLICY_ABORT_METHOD = "policy/abort"
POLICY_METHODS = frozenset({
    POLICY_REVISION_METHOD,
    POLICY_RESERVE_METHOD,
    POLICY_PREPARE_METHOD,
    POLICY_COMMIT_METHOD,
    POLICY_RETIRE_METHOD,
    POLICY_ABORT_METHOD,
})

TURN_OUTCOME_COMPLETED = "completed"
TURN_OUTCOME_FAILED = "failed"
TURN_OUTCOME_INTERRUPTED = "interrupted"


class AgentWireError(RuntimeError):
    """Base class for the agent-process wire schema failures."""


class UnsupportedAgentEventError(AgentWireError):
    """The agent produced an event the process wire schema does not name."""

    def __init__(self, event_name: str) -> None:
        super().__init__(f"event {event_name!r} is not part of the agent wire schema")
        self.event_name = event_name


class UnsupportedAgentRequestError(AgentWireError):
    """A turn request carries a value that has no process representation."""


class UnsupportedUserInputResultError(AgentWireError):
    """A user-input result class is not part of the agent wire schema."""


class RemoteUserInputRejectedError(AgentWireError):
    """The host rejected an open user-input request across the process wire."""


_JSON_EVENT_TYPES: tuple[type[BaseEvent], ...] = (
    AssistantEvent,
    BackgroundJobsChangedEvent,
    BackgroundWorkEvent,
    CompactEndEvent,
    CompactStartEvent,
    HookEndEvent,
    HookRunEndEvent,
    HookRunStartEvent,
    HookStartEvent,
    ReasoningEvent,
    RouteTableAppliedEvent,
    SessionTitleUpdatedEvent,
    TokenUsageUpdatedEvent,
    ToolStreamEvent,
    ToolWaitStateChangedEvent,
    UserMessageEvent,
    WaitingForInputEvent,
)
_EVENT_TYPES: dict[str, type[BaseEvent]] = {
    event_type.__name__: event_type for event_type in _JSON_EVENT_TYPES
}

_USER_INPUT_ARGS_TYPES: dict[str, type[BaseModel]] = {
    UserQuestionRequest.__name__: UserQuestionRequest
}
_USER_INPUT_RESULT_TYPES: dict[str, type[BaseModel]] = {
    result_type.__name__: result_type
    for result_type in (UserQuestionResult, UnansweredQuestions)
}

_IMAGES_ADAPTER: TypeAdapter[list[ImageAttachment]] = TypeAdapter(list[ImageAttachment])
_RESOURCES_ADAPTER: TypeAdapter[list[UserResource]] = TypeAdapter(list[UserResource])


def encode_event(event: BaseEvent) -> dict[str, Any]:
    """Project one core event into its JSON-safe wire payload."""
    match event:
        case ToolCallEvent():
            return {
                "type": "ToolCallEvent",
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "tool_call_index": event.tool_call_index,
                "args": _encode_value(event.args),
                "presentation": _encode_value(event.presentation),
            }
        case ToolResultEvent():
            payload = event.model_dump(mode="json", exclude={"tool_class", "result"})
            payload["type"] = "ToolResultEvent"
            payload["result"] = _encode_value(event.result)
            return payload
        case UserInputRequestEvent():
            return {
                "type": "UserInputRequestEvent",
                "request_id": event.request_id,
                "tool_call_id": event.tool_call_id,
                "origin": event.origin.value,
                "args_type": type(event.args).__name__,
                "args": _encode_value(event.args),
                "unattended_expiry": (
                    event.unattended_expiry.model_dump(mode="json")
                    if event.unattended_expiry is not None
                    else None
                ),
            }
    event_type = _EVENT_TYPES.get(type(event).__name__)
    if event_type is None:
        raise UnsupportedAgentEventError(type(event).__name__)
    return {"type": event_type.__name__, "data": event.model_dump(mode="json")}


def decode_event(payload: Mapping[str, Any]) -> BaseEvent:
    """Reconstruct one core event from its wire payload."""
    kind = payload.get("type")
    if not isinstance(kind, str):
        raise UnsupportedAgentEventError(repr(kind))
    if kind == "ToolCallEvent":
        return ToolCallEvent.model_construct(
            tool_call_id=payload["tool_call_id"],
            tool_name=payload["tool_name"],
            tool_class=None,
            tool_call_index=payload.get("tool_call_index"),
            args=payload.get("args"),
            presentation=_decode_presentation(
                ToolCallPresentation, payload.get("presentation")
            ),
        )
    if kind == "ToolResultEvent":
        fields = dict(payload)
        fields.pop("type", None)
        fields["tool_class"] = None
        fields["presentation"] = _decode_presentation(
            ToolResultPresentation, fields.get("presentation")
        )
        return ToolResultEvent.model_construct(**fields)
    if kind == "UserInputRequestEvent":
        args_type = payload.get("args_type")
        args_model = _USER_INPUT_ARGS_TYPES.get(str(args_type))
        if args_model is None:
            raise UnsupportedAgentRequestError(
                f"user-input args {args_type!r} are not part of the wire schema"
            )
        expiry_payload = payload.get("unattended_expiry")
        return UserInputRequestEvent(
            request_id=payload["request_id"],
            tool_call_id=payload["tool_call_id"],
            origin=payload["origin"],
            args=args_model.model_validate(payload["args"]),
            unattended_expiry=(
                UnattendedExpiryPolicy.model_validate(expiry_payload)
                if expiry_payload is not None
                else None
            ),
        )
    event_type = _EVENT_TYPES.get(kind)
    if event_type is None:
        raise UnsupportedAgentEventError(kind)
    return event_type.model_validate(payload.get("data"))


def encode_turn_request(request: ExecutionTurnRequest) -> dict[str, Any]:
    """Encode the serializable part of a turn request."""
    options = request.turn_options or AgentTurnOptions()
    return {
        "message": request.message,
        "client_message_id": request.client_message_id,
        "auto_title": request.auto_title,
        "images": [image.model_dump(mode="json") for image in request.images or ()],
        "user_display_content": _encode_value(request.user_display_content),
        "input_text": request.input_text,
        "resources": [_encode_value(resource) for resource in request.resources or ()],
        "turn_options": {
            "turn_id": options.turn_id,
            "injected": options.injected,
            "user_initiated_retry": options.user_initiated_retry,
        },
    }


def decode_turn_request(params: Mapping[str, Any]) -> ExecutionTurnRequest:
    """Rebuild a turn request, refusing values the wire cannot carry."""
    if params.get("subagent_runner") is not None or params.get("tool_io") is not None:
        raise UnsupportedAgentRequestError(
            "host-side subagent and tool I/O ports cannot cross the process wire"
        )
    options = params.get("turn_options") or {}
    display_content = params.get("user_display_content")
    return ExecutionTurnRequest(
        message=params.get("message"),
        client_message_id=params.get("client_message_id"),
        auto_title=params.get("auto_title"),
        images=_IMAGES_ADAPTER.validate_python(params.get("images") or []),
        user_display_content=(
            UserDisplayContent.model_validate(display_content)
            if display_content is not None
            else None
        ),
        input_text=params.get("input_text"),
        resources=_RESOURCES_ADAPTER.validate_python(params.get("resources") or []),
        turn_options=AgentTurnOptions(
            turn_id=options.get("turn_id"),
            injected=bool(options.get("injected")),
            user_initiated_retry=bool(options.get("user_initiated_retry")),
        ),
    )


def encode_retry_reason(reason: RetryReason) -> dict[str, str]:
    return {"category": reason.category.value, "detail": reason.detail}


def decode_retry_reason(payload: Mapping[str, Any]) -> RetryReason:
    return RetryReason(
        category=RetryCategory(payload["category"]), detail=str(payload["detail"])
    )


def encode_user_input_result(result: BaseModel) -> dict[str, Any]:
    type_name = type(result).__name__
    if type_name not in _USER_INPUT_RESULT_TYPES:
        raise UnsupportedUserInputResultError(
            f"user-input result {type_name!r} is not part of the wire schema"
        )
    return {"type": type_name, "data": result.model_dump(mode="json")}


def decode_user_input_result(payload: Mapping[str, Any]) -> BaseModel:
    result_type = _USER_INPUT_RESULT_TYPES.get(str(payload.get("type")))
    if result_type is None:
        raise UnsupportedUserInputResultError(
            f"user-input result {payload.get('type')!r} is not part of the wire schema"
        )
    return result_type.model_validate(payload.get("data"))


def encode_transcript_page(response: AgentTranscriptGetResponse) -> dict[str, Any]:
    return response.model_dump(mode="json")


def decode_transcript_page(payload: Mapping[str, Any]) -> AgentTranscriptGetResponse:
    return AgentTranscriptGetResponse.model_validate(payload)


def _encode_value(value: BaseModel | JsonValue) -> JsonValue:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    return value


def _decode_presentation[PresentationT: BaseModel](
    presentation_type: type[PresentationT], payload: object
) -> PresentationT | None:
    if payload is None:
        return None
    return presentation_type.model_validate(payload)
