from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from chartreux.core.tools.base import BaseTool
else:
    BaseTool = Any

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.llm_models import ImageAttachment
from chartreux.core.session_types import AgentStats
from chartreux.user_content import UserDisplayContent, UserResource
from chartreux.utils.tool_presentation import (
    ToolCallPresentation,
    ToolResultPresentation,
)


class BaseEvent(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class UserMessageEvent(BaseEvent):
    content: str
    message_id: str
    images: list[ImageAttachment] = Field(default_factory=list)
    user_display_content: UserDisplayContent | None = None
    resources: list[UserResource] = Field(default_factory=list)


class AssistantEvent(BaseEvent):
    content: str
    stopped_by_middleware: bool = False
    message_id: str | None = None

    def __add__(self, other: AssistantEvent) -> AssistantEvent:
        return AssistantEvent(
            content=self.content + other.content,
            stopped_by_middleware=self.stopped_by_middleware
            or other.stopped_by_middleware,
            message_id=self.message_id or other.message_id,
        )


class ReasoningEvent(BaseEvent):
    content: str
    message_id: str | None = None


class ToolCallEvent(BaseEvent):
    tool_call_id: str
    tool_name: str
    tool_class: type[BaseTool]
    tool_call_index: int | None = None
    args: BaseModel | None = None
    presentation: ToolCallPresentation | None = None


class ToolResultEvent(BaseEvent):
    tool_name: str
    tool_class: type[BaseTool] | None
    result: BaseModel | None = None
    error: str | None = None
    error_display: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    cancelled: bool = False
    duration: float | None = None
    tool_call_id: str
    presentation: ToolResultPresentation | None = None


class ToolStreamEvent(BaseEvent):
    tool_name: str
    message: str
    tool_call_id: str


class WaitingForInputEvent(BaseEvent):
    task_id: str
    label: str | None = None
    predefined_answers: list[str] | None = None


class RequestEvent(BaseEvent):
    request_id: str


class UserInputRequestEvent(RequestEvent):
    args: BaseModel
    tool_call_id: str


class TokenUsageUpdatedEvent(BaseEvent):
    stats: AgentStats
    context_window: int


class CompactStartEvent(BaseEvent):
    current_context_tokens: int
    threshold: int
    # WORKAROUND: Using tool_call to communicate compact events to the client.
    # This should be revisited when the ACP protocol defines how compact events
    # should be represented.
    # [RFD](https://agentclientprotocol.com/rfds/session-usage)
    tool_call_id: str


class CompactEndEvent(BaseEvent):
    summary_length: int
    old_session_id: str | None = None
    new_session_id: str | None = None
    # WORKAROUND: Using tool_call to communicate compact events to the client.
    # This should be revisited when the ACP protocol defines how compact events
    # should be represented.
    # [RFD](https://agentclientprotocol.com/rfds/session-usage)
    tool_call_id: str


class SessionTitleUpdatedEvent(BaseEvent):
    title: str
    session_id: str


class BackgroundWorkEvent(BaseEvent):
    work_id: str
    kind: Literal["session_title"]
    phase: Literal["started", "finished"]
    session_id: str
