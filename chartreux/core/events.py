from __future__ import annotations

from abc import ABC
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from chartreux.core.tools.base import BaseTool
else:
    BaseTool = Any

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.dispatch.route_models import AppliedRouteChange
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
    posted_at: datetime | None = None
    images: list[ImageAttachment] = Field(default_factory=list)
    user_display_content: UserDisplayContent | None = None
    resources: list[UserResource] = Field(default_factory=list)


class AssistantEvent(BaseEvent):
    content: str
    stopped_by_middleware: bool = False
    middleware_metadata: dict[str, Any] = Field(default_factory=dict)
    message_id: str | None = None
    posted_at: datetime | None = None

    def __add__(self, other: AssistantEvent) -> AssistantEvent:
        return AssistantEvent(
            content=self.content + other.content,
            stopped_by_middleware=self.stopped_by_middleware
            or other.stopped_by_middleware,
            middleware_metadata=self.middleware_metadata or other.middleware_metadata,
            message_id=self.message_id or other.message_id,
            posted_at=self.posted_at or other.posted_at,
        )


class ReasoningEvent(BaseEvent):
    content: str
    message_id: str | None = None
    posted_at: datetime | None = None


class ToolCallEvent(BaseEvent):
    tool_call_id: str
    tool_name: str
    tool_class: type[BaseTool]
    tool_call_index: int | None = None
    args: BaseModel | None = None
    presentation: ToolCallPresentation | None = None


class ToolCancellationOrigin(StrEnum):
    STEERING = "steering"


class ToolWaitStateChangedEvent(BaseEvent):
    turn_id: str
    waiting_only: bool


class ToolResultEvent(BaseEvent):
    tool_name: str
    tool_class: type[BaseTool] | None
    result: BaseModel | None = None
    error: str | None = None
    error_display: str | None = None
    # R7: a failed or timed-out shell run may still have written files
    # (partial execution), so its write-classified paths ride the error
    # event: the bash tool attaches them to the raised ToolError and the
    # tool executor copies them here, and the submission scan collects
    # them exactly as a successful run's would be. Empty for other tools
    # and for errors raised before the command ran.
    error_write_paths: list[str] = Field(default_factory=list)
    error_write_paths_unknown: bool = False
    # A1: the failed or cancelled run's mutation-roots footprint (the resolved
    # work-tree roots of its checkout/switch invocations), copied off the
    # raised error the same way — a mutation-scope footprint the submission
    # scan never walks; the completion audit consumes it.
    error_mutation_roots: list[str] = Field(default_factory=list)
    error_mutation_roots_unknown: bool = False
    skipped: bool = False
    skip_reason: str | None = None
    cancelled: bool = False
    cancellation_origin: ToolCancellationOrigin | None = None
    duration: float | None = None
    # P2 (round-9): the harness-minted, run-scoped completion-evidence
    # receipt occurrence ID. Minted by the tool executor at the tool-call
    # dispatch site BEFORE the invocation starts, it rides every result
    # event the invocation produces so the accumulator's collection (P3)
    # can register it unchanged; tool_call_id alone never keys pairing
    # (tool-call ids are reused across turns).
    occurrence_id: str | None = None
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


class UserInputOrigin(StrEnum):
    """Trusted provenance for a user-input request (plan U/OD1).

    ``MODEL_QUESTION`` marks a model-originated ``ask_user_question`` call —
    the only origin eligible for unattended expiry. ``RUNTIME_APPROVAL`` marks
    a runtime approval/grant prompt (dangerous-command and root-access
    approvals share the request broker but never call the ask tool); those
    requests are permanently ineligible for expiry. The origin is stamped by
    trusted code through the interaction port, never by model arguments, and
    defaults to the ineligible origin so a forgotten call site fails closed.
    """

    MODEL_QUESTION = "model_question"
    RUNTIME_APPROVAL = "runtime_approval"


class UserInputRequestEvent(RequestEvent):
    args: BaseModel
    tool_call_id: str
    origin: UserInputOrigin = UserInputOrigin.RUNTIME_APPROVAL


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


class RouteTableAppliedEvent(BaseEvent):
    """A validated route edit applied at a reload commit (ADR 0022).

    Routes are bindings, not frozen content: the edit applies to subsequent
    launch admissions in the same conversation, while every already-admitted
    launch keeps its pinned binding. The changes are diffed against the
    manifest's bind-time route snapshot — the frozen integrity and diff
    anchor — and ride the out-of-band channel so the session surfaces them
    as a conversation-visible notice.
    """

    session_id: str
    changes: tuple[AppliedRouteChange, ...]


class BackgroundJobsChangedEvent(BaseEvent):
    """State-only notification fenced by managed-job lifetime, not transcript."""

    root_lifetime_id: str
    generation: int = Field(ge=0)
    revision: int = Field(ge=0)
    active_count: int = Field(ge=0)


class BackgroundWorkEvent(BaseEvent):
    work_id: str
    kind: Literal["session_title"]
    phase: Literal["started", "finished"]
    session_id: str
