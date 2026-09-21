from __future__ import annotations

from datetime import datetime
from enum import StrEnum, auto
from functools import cache
from typing import Annotated, Literal, Self

from pydantic import AliasChoices, Field, JsonValue, TypeAdapter, model_validator

from chartreux.agents import AgentSafety, AgentType
from chartreux.app_server._effect_models import (
    EffectDetail as EffectDetail,
    FileEditEffectBatchInput as FileEditEffectBatchInput,
    FileEditEffectChange as FileEditEffectChange,
    FileEditEffectDetail as FileEditEffectDetail,
    FileEditEffectInput as FileEditEffectInput,
    FileEditEffectOccurrence as FileEditEffectOccurrence,
    FileEditEffectOutput as FileEditEffectOutput,
    FileReadEffectDetail as FileReadEffectDetail,
    FileReadEffectInput as FileReadEffectInput,
    FileReadEffectOutput as FileReadEffectOutput,
    FileSearchEffectDetail as FileSearchEffectDetail,
    FileSearchEffectInput as FileSearchEffectInput,
    FileSearchEffectMatch as FileSearchEffectMatch,
    FileSearchEffectOutput as FileSearchEffectOutput,
    FileWriteEffectDetail as FileWriteEffectDetail,
    FileWriteEffectInput as FileWriteEffectInput,
    FileWriteEffectOutput as FileWriteEffectOutput,
    GenericEffectDetail as GenericEffectDetail,
    ShellEffectDetail as ShellEffectDetail,
    ShellEffectInput as ShellEffectInput,
    ShellEffectOutput as ShellEffectOutput,
    SkillEffectDetail as SkillEffectDetail,
    SkillEffectInput as SkillEffectInput,
    SkillEffectOutput as SkillEffectOutput,
    SubagentEffectDetail as SubagentEffectDetail,
    SubagentEffectInput as SubagentEffectInput,
    SubagentEffectOutput as SubagentEffectOutput,
    TodoEffectDetail as TodoEffectDetail,
    TodoEffectInput as TodoEffectInput,
    TodoEffectItem as TodoEffectItem,
    TodoEffectOutput as TodoEffectOutput,
    TodoEffectPriority as TodoEffectPriority,
    TodoEffectStatus as TodoEffectStatus,
    UserQuestionEffectDetail as UserQuestionEffectDetail,
    WebFetchEffectDetail as WebFetchEffectDetail,
    WebFetchEffectInput as WebFetchEffectInput,
    WebFetchEffectOutput as WebFetchEffectOutput,
    WebSearchEffectDetail as WebSearchEffectDetail,
    WebSearchEffectInput as WebSearchEffectInput,
    WebSearchEffectOutput as WebSearchEffectOutput,
    WebSearchEffectSource as WebSearchEffectSource,
    WorktreeEffectDetail as WorktreeEffectDetail,
    WorktreeEffectInput as WorktreeEffectInput,
    effect_input_json as effect_input_json,
)
from chartreux.app_server._model import ProtocolModel
from chartreux.questions import (
    QuestionChoice as QuestionChoice,
    UserAnswer as UserAnswer,
    UserQuestion as UserQuestion,
    UserQuestionRequest as UserQuestionRequest,
    UserQuestionResult as UserQuestionResult,
)
from chartreux.user_content import (
    UserDisplayContent as UserDisplayContent,
    UserResource,
)
from chartreux.utils.tool_presentation import (
    EffectCallDisplay as EffectCallDisplay,
    EffectResultDisplay as EffectResultDisplay,
)


class IdentityEntityView(ProtocolModel):
    id: str
    name: str


class IdentityView(ProtocolModel):
    id: str
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    workspace: IdentityEntityView | None = None
    organization: IdentityEntityView | None = None

    @property
    def name(self) -> str | None:
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        if self.first_name:
            return self.first_name
        return self.email


class FileImageSource(ProtocolModel):
    kind: Literal["file"] = "file"
    path: str


class InlineImageSource(ProtocolModel):
    kind: Literal["inline"] = "inline"
    data: str


ImageSource = Annotated[
    FileImageSource | InlineImageSource, Field(discriminator="kind")
]


class ImageAttachment(ProtocolModel):
    source: ImageSource
    alias: str
    mime_type: str


class MentionStats(ProtocolModel):
    count: int = 0
    context_types: dict[str, int] = Field(default_factory=dict)
    file_extensions: dict[str, int] = Field(default_factory=dict)


class PreparedPrompt(ProtocolModel):
    display_text: str
    prompt_text: str
    images: list[ImageAttachment] = Field(default_factory=list)
    auto_title: str | None = None
    mentions: MentionStats = Field(default_factory=MentionStats)


type WorkspaceTrustDecision = Literal["trust_repo", "trust_cwd", "decline"]
type WorkspaceTrustStatus = Literal["trusted", "session", "untrusted"]


class WorkspaceTrustDetails(ProtocolModel):
    cwd: str
    repo_root: str | None = None
    detected_files: list[str] = Field(default_factory=list)
    repo_detected_files: list[str] = Field(default_factory=list)
    repo_explicitly_untrusted: bool = False
    settings_path: str
    available_decisions: list[WorkspaceTrustDecision] = Field(default_factory=list)


class TextContentBlock(ProtocolModel):
    type: Literal["text"] = "text"
    text: str


class ImageContentBlock(ProtocolModel):
    type: Literal["image"] = "image"
    attachment: ImageAttachment


class ResourceContentBlock(ProtocolModel):
    type: Literal["resource"] = "resource"
    resource: UserResource


ContentBlock = Annotated[
    TextContentBlock | ImageContentBlock | ResourceContentBlock,
    Field(discriminator="type"),
]


class MessageAnnotations(ProtocolModel):
    chartreux_user_display_content: UserDisplayContent | None = Field(
        default=None,
        # This is a published protocol key; retain the legacy wire name for
        # compatibility with existing app-server clients.
        alias="vibe.userDisplayContent",
        exclude_if=lambda value: value is None,
    )


class SessionTextContentBlock(ProtocolModel):
    type: Literal["text"] = "text"
    text: str = ""


class SessionImageContentBlock(ProtocolModel):
    type: Literal["image"] = "image"
    uri: str
    media_type: str | None = None
    alt_text: str | None = None


class SessionResourceLinkContentBlock(ProtocolModel):
    type: Literal["resource_link"] = "resource_link"
    uri: str
    name: str | None = None
    title: str | None = None
    description: str | None = None
    media_type: str | None = None
    size: int | None = Field(default=None, ge=0)


class SessionEmbeddedResourceContentBlock(ProtocolModel):
    type: Literal["embedded_resource"] = "embedded_resource"
    uri: str
    media_type: str | None = None
    text: str | None = None
    blob: str | None = None

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if (self.text is None) == (self.blob is None):
            raise ValueError("Embedded resources require exactly one of text or blob")
        return self


SessionContentBlock = Annotated[
    SessionTextContentBlock
    | SessionImageContentBlock
    | SessionResourceLinkContentBlock
    | SessionEmbeddedResourceContentBlock,
    Field(discriminator="type"),
]


class TurnContextInputEntry(ProtocolModel):
    role: Literal["context"] = "context"
    entry_id: str | None = None
    content: list[SessionContentBlock] = Field(min_length=1)
    annotations: MessageAnnotations = Field(default_factory=MessageAnnotations)

    @property
    def input(self) -> list[SessionContentBlock]:
        return self.content


class TurnUserInputEntry(ProtocolModel):
    role: Literal["user"] = "user"
    entry_id: str | None = None
    content: list[SessionContentBlock] = Field(min_length=1)
    annotations: MessageAnnotations = Field(default_factory=MessageAnnotations)

    @property
    def input(self) -> list[SessionContentBlock]:
        return self.content


TurnInputEntry = Annotated[
    TurnContextInputEntry | TurnUserInputEntry, Field(discriminator="role")
]


def validate_turn_input_entries(entries: list[TurnInputEntry]) -> None:
    user_positions = [
        index for index, entry in enumerate(entries) if entry.role == "user"
    ]
    if len(user_positions) > 1:
        raise ValueError("Turn input accepts at most one user entry")
    if user_positions and user_positions[0] != len(entries) - 1:
        raise ValueError("The user entry must be the final turn input entry")


class UserInputCallbackDetail(ProtocolModel):
    kind: Literal["user_input"] = "user_input"
    request: UserQuestionRequest
    related_entry_id: str | None = None


CallbackDetail = UserInputCallbackDetail


class UserInputCallbackOutput(ProtocolModel):
    type: Literal["user_input"] = "user_input"
    result: UserQuestionResult


CallbackOutput = UserInputCallbackOutput


class PublicEntryGenerationStatus(StrEnum):
    IN_PROGRESS = auto()
    COMPLETED = auto()


class PublicTurnStatus(StrEnum):
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    INTERRUPTED = auto()


class PublicTurnStopReason(StrEnum):
    LIMIT = auto()


class PublicRetryCategory(StrEnum):
    RATE_LIMITED = auto()
    SERVER_ERROR = auto()
    TIMED_OUT = auto()
    CONNECTION = auto()
    UNKNOWN = auto()


class PublicRetryState(ProtocolModel):
    turn_id: str
    category: PublicRetryCategory
    detail: str


class TurnErrorCode(StrEnum):
    RATE_LIMIT = auto()
    CONTEXT_TOO_LONG = auto()
    RESPONSE_TOO_LONG = auto()
    REFUSAL = auto()
    INVALID_IMAGE_ATTACHMENT = auto()
    IMAGES_NOT_SUPPORTED = auto()
    COMPACTION_FAILED = auto()
    INCOMPLETE_STREAM = auto()
    BACKEND_ERROR = auto()
    INVALID_MODEL = auto()
    INVALID_API_KEY = auto()
    INTERNAL_ERROR = auto()


class PublicError(ProtocolModel):
    message: str
    code: str | None = None
    details: JsonValue = None


class TokenUsage(ProtocolModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AgentStatsSnapshot(ProtocolModel):
    steps: int = 0
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_cached_tokens: int = 0
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0
    cached_input_price_per_million: float | None = None
    known_cost_total: float = Field(
        default=0.0, validation_alias=AliasChoices("knownCostTotal", "known_cost_total")
    )
    has_unknown_cost: bool = Field(
        default=False,
        validation_alias=AliasChoices("hasUnknownCost", "has_unknown_cost"),
    )
    tool_calls_agreed: int = 0
    tool_calls_rejected: int = 0
    tool_calls_failed: int = 0
    tool_calls_succeeded: int = 0
    context_tokens: int = 0
    last_turn_prompt_tokens: int = 0
    last_turn_completion_tokens: int = 0
    last_turn_cached_tokens: int = 0
    last_turn_duration: float = 0.0
    tokens_per_second: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _mark_token_only_legacy_stats_incomplete(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        for wire_name, field_name in (
            ("knownCostTotal", "known_cost_total"),
            ("hasUnknownCost", "has_unknown_cost"),
        ):
            if wire_name in value:
                value.pop(field_name, None)
            elif field_name in value:
                value[wire_name] = value.pop(field_name)
        if value and "knownCostTotal" not in value:
            value.setdefault("knownCostTotal", 0.0)
            value["hasUnknownCost"] = True
        return value

    @property
    def session_total_llm_tokens(self) -> int:
        return self.session_prompt_tokens + self.session_completion_tokens

    @property
    def last_turn_total_tokens(self) -> int:
        return self.last_turn_prompt_tokens + self.last_turn_completion_tokens

    @property
    def session_cost(self) -> float | None:
        if self.has_unknown_cost:
            return None
        return self.known_cost_total

    @property
    def token_usage(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.session_prompt_tokens,
            output_tokens=self.session_completion_tokens,
            total_tokens=self.session_total_llm_tokens,
        )


class ConfigIssue(ProtocolModel):
    file: str
    message: str


class DebugLogEntry(ProtocolModel):
    id: str
    timestamp: datetime
    ppid: int
    pid: int
    level: str
    message: str
    raw_line: str


class DebugLogPage(ProtocolModel):
    entries: list[DebugLogEntry]
    has_more: bool
    cursor: int | None = None


class AgentSummary(ProtocolModel):
    name: str
    display_name: str
    description: str
    safety: AgentSafety
    agent_type: AgentType


class SkillSummary(ProtocolModel):
    name: str
    description: str
    prompt: str
    user_invocable: bool = True
    source: Literal["builtin", "local"] = "local"
    scope: Literal["builtin", "global", "project"] = "global"


class ToolSummary(ProtocolModel):
    name: str
    is_custom: bool = False


class MCPSourceStatus(StrEnum):
    DISABLED = auto()
    CONNECTED = auto()
    ENABLED = auto()
    NEEDS_AUTH = auto()
    UNAVAILABLE = auto()


class MCPToolSummary(ProtocolModel):
    name: str
    description: str = ""
    enabled: bool = True


class MCPSourceSummary(ProtocolModel):
    name: str
    transport: str
    status: MCPSourceStatus
    tools: list[MCPToolSummary] = Field(default_factory=list)
    error: str | None = None


class MCPState(ProtocolModel):
    sources: list[MCPSourceSummary] = Field(default_factory=list)
    discovery_errors: dict[str, str] = Field(default_factory=dict)

    @property
    def needs_auth(self) -> list[str]:
        return sorted(
            source.name
            for source in self.sources
            if source.status is MCPSourceStatus.NEEDS_AUTH
        )

    @property
    def statuses(self) -> dict[str, str]:
        return {source.name: source.status.value for source in self.sources}


class SessionLogSummary(ProtocolModel):
    enabled: bool
    session_id: str | None = None
    persisted: bool = False
    path: str | None = None
    title: str | None = None
    needs_initial_auto_title: bool = False


class SavedSessionSummary(ProtocolModel):
    session_id: str
    cwd: str
    parent_session_id: str | None = None
    title: str | None = None
    end_time: str | None = None
    preview: str = ""

    @property
    def option_id(self) -> str:
        return self.session_id

    @property
    def short_id(self) -> str:
        return self.session_id[:8]


class PendingEffectState(ProtocolModel):
    status: Literal["pending"] = "pending"


class RunningEffectState(ProtocolModel):
    status: Literal["running"] = "running"
    output_text: str = ""


class BlockedEffectState(ProtocolModel):
    status: Literal["blocked"] = "blocked"
    callback_id: str
    output_text: str = ""


class CompletedEffectState(ProtocolModel):
    status: Literal["completed"] = "completed"
    output: JsonValue = None
    output_text: str = ""
    duration_ms: float = 0.0
    display: EffectResultDisplay


class FailedEffectState(ProtocolModel):
    status: Literal["failed"] = "failed"
    error: PublicError
    output: JsonValue = None
    output_text: str = ""
    duration_ms: float = 0.0
    display: EffectResultDisplay


class CancelledEffectState(ProtocolModel):
    status: Literal["cancelled"] = "cancelled"
    reason: str
    output_text: str = ""
    duration_ms: float = 0.0
    display: EffectResultDisplay | None = None


class SkippedEffectState(ProtocolModel):
    status: Literal["skipped"] = "skipped"
    reason: str
    display: EffectResultDisplay


EffectState = Annotated[
    PendingEffectState
    | RunningEffectState
    | BlockedEffectState
    | CompletedEffectState
    | FailedEffectState
    | CancelledEffectState
    | SkippedEffectState,
    Field(discriminator="status"),
]


class OpenCallbackState(ProtocolModel):
    status: Literal["open"] = "open"


class AnsweredCallbackState(ProtocolModel):
    status: Literal["answered"] = "answered"
    output: CallbackOutput


class CancelledCallbackState(ProtocolModel):
    status: Literal["cancelled"] = "cancelled"
    reason: str


class ExpiredCallbackState(ProtocolModel):
    status: Literal["expired"] = "expired"
    reason: str


CallbackState = Annotated[
    OpenCallbackState
    | AnsweredCallbackState
    | CancelledCallbackState
    | ExpiredCallbackState,
    Field(discriminator="status"),
]


class _PublicHistoryEntryBase(ProtocolModel):
    id: str
    session_id: str
    turn_id: str | None = None
    created_at: int
    updated_at: int
    generation_status: PublicEntryGenerationStatus
    related_entry_id: str | None = None


type PublicMessageSource = Literal["turn_start", "turn_steer", "harness"]


class PublicMessageEntry(_PublicHistoryEntryBase):
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: list[ContentBlock]
    source: PublicMessageSource | None = None
    user_display_content: UserDisplayContent | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(
            block.text for block in self.content if isinstance(block, TextContentBlock)
        )

    @property
    def images(self) -> list[ImageAttachment]:
        return [
            block.attachment
            for block in self.content
            if isinstance(block, ImageContentBlock)
        ]


class PublicReasoningEntry(_PublicHistoryEntryBase):
    type: Literal["reasoning"] = "reasoning"
    text: str
    summary: list[str] = Field(default_factory=list)


class PublicEffectEntry(_PublicHistoryEntryBase):
    type: Literal["effect"] = "effect"
    title: str
    detail: EffectDetail
    state: EffectState


class PublicCallbackEntry(_PublicHistoryEntryBase):
    type: Literal["callback"] = "callback"
    callback_id: str
    title: str
    detail: CallbackDetail
    state: CallbackState


class HookScope(StrEnum):
    POST_AGENT = auto()
    PRE_TOOL = auto()
    POST_TOOL = auto()


class HookSeverity(StrEnum):
    OK = auto()
    WARNING = auto()
    ERROR = auto()


class HookNoticeDetail(ProtocolModel):
    kind: Literal[
        "hook_run_started", "hook_run_completed", "hook_started", "hook_completed"
    ]
    scope: HookScope = HookScope.POST_AGENT
    tool_name: str | None = None
    tool_call_id: str | None = None
    hook_name: str | None = None
    status: HookSeverity | None = None
    content: str | None = None


class ContextClearedNoticeDetail(ProtocolModel):
    kind: Literal["context_cleared"] = "context_cleared"
    plan_file_path: str | None = None


class SessionTitleUpdatedNoticeDetail(ProtocolModel):
    kind: Literal["session_title_updated"] = "session_title_updated"
    title: str


class PlanReviewStartedNoticeDetail(ProtocolModel):
    kind: Literal["plan_review_started"] = "plan_review_started"
    file_path: str


class PlanReviewEndedNoticeDetail(ProtocolModel):
    kind: Literal["plan_review_ended"] = "plan_review_ended"


class WaitingForInputNoticeDetail(ProtocolModel):
    kind: Literal["waiting_for_input"] = "waiting_for_input"
    task_id: str
    label: str | None = None
    predefined_answers: list[str] | None = None


class ScheduledLoopFiredNoticeDetail(ProtocolModel):
    kind: Literal["scheduled_loop_fired"] = "scheduled_loop_fired"
    loop_id: str


NoticeDetail = Annotated[
    HookNoticeDetail
    | ContextClearedNoticeDetail
    | SessionTitleUpdatedNoticeDetail
    | PlanReviewStartedNoticeDetail
    | PlanReviewEndedNoticeDetail
    | WaitingForInputNoticeDetail
    | ScheduledLoopFiredNoticeDetail,
    Field(discriminator="kind"),
]


class PublicCheckpointEntry(_PublicHistoryEntryBase):
    type: Literal["checkpoint"] = "checkpoint"
    kind: str
    message: str | None = None
    details: JsonValue = None


class PublicNoticeEntry(_PublicHistoryEntryBase):
    type: Literal["notice"] = "notice"
    level: Literal["info", "warning", "error"]
    message: str
    detail: NoticeDetail


PublicHistoryEntry = Annotated[
    PublicMessageEntry
    | PublicReasoningEntry
    | PublicEffectEntry
    | PublicCallbackEntry
    | PublicCheckpointEntry
    | PublicNoticeEntry,
    Field(discriminator="type"),
]


@cache
def _public_history_entry_adapter() -> TypeAdapter[PublicHistoryEntry]:
    return TypeAdapter(PublicHistoryEntry)


def validate_history_entry(value: object) -> PublicHistoryEntry:
    return _public_history_entry_adapter().validate_python(
        value, by_alias=True, by_name=False
    )


class HistoryCursor(ProtocolModel):
    before: str | None = None
    after: str | None = None


class PublicHistoryPage(ProtocolModel):
    entries: list[PublicHistoryEntry] = Field(default_factory=list)
    cursor: HistoryCursor = Field(default_factory=HistoryCursor)
    range: Literal["latest", "page"] = "latest"


class IdleSessionStatus(ProtocolModel):
    type: Literal["idle"] = "idle"


class RunningSessionStatus(ProtocolModel):
    type: Literal["running"] = "running"
    active_turn_id: str


class BlockedSessionStatus(ProtocolModel):
    type: Literal["blocked"] = "blocked"
    active_turn_id: str
    callback_id: str
    reason: str


class FailedSessionStatus(ProtocolModel):
    type: Literal["failed"] = "failed"
    message: str


PublicSessionStatus = Annotated[
    IdleSessionStatus
    | RunningSessionStatus
    | BlockedSessionStatus
    | FailedSessionStatus,
    Field(discriminator="type"),
]


class PublicSession(ProtocolModel):
    id: str
    root_session_id: str | None = None
    parent_session_id: str | None = None
    title: str | None = None
    preview: str = ""
    status: PublicSessionStatus
    created_at: int
    updated_at: int
    cwd: str | None = None
    workspace_roots: list[str] = Field(default_factory=list)
    model: str | None = None
    token_usage: TokenUsage | None = None
    context_usage: TokenUsage | None = None


class PublicQueuedTurn(ProtocolModel):
    id: str
    created_at: int
    entries: list[TurnInputEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        validate_turn_input_entries(self.entries)
        return self


TURN_QUEUE_MAX_ITEMS = 32


class PublicTurnQueue(ProtocolModel):
    items: list[PublicQueuedTurn] = Field(default_factory=list)
    paused: bool = False
    max_items: int = Field(default=TURN_QUEUE_MAX_ITEMS, ge=1, strict=True)


class PublicTurn(ProtocolModel):
    id: str
    session_id: str
    status: PublicTurnStatus
    started_at: int
    completed_at: int | None = None
    error: PublicError | None = None
    stop_reason: PublicTurnStopReason | None = None
    queue_item_id: str | None = None
    next_turn_id: str | None = None


class PublicSessionState(ProtocolModel):
    # This is a persisted app-server protocol discriminator; retain the legacy
    # wire value so saved state and existing clients remain compatible.
    format: Literal["vibe.public-session-state/v1"] = "vibe.public-session-state/v1"
    event_id: int = Field(ge=0, strict=True)
    session: PublicSession
    is_quiescent: bool | None = None
    history: list[PublicHistoryEntry] | None = None
    history_before_cursor: str | None = None
    turns: list[PublicTurn] | None = None
    active_callbacks: list[PublicCallbackEntry] = Field(default_factory=list)
    turn_queue: PublicTurnQueue = Field(default_factory=PublicTurnQueue)
    retrying: PublicRetryState | None = None

    @property
    def latest_turn(self) -> PublicTurn | None:
        if not self.turns:
            return None
        return self.turns[-1]


class JsonPatchOperation(ProtocolModel):
    op: Literal["add", "append", "replace", "remove", "test"]
    path: str
    value: JsonValue = None


class ScheduledLoop(ProtocolModel):
    id: str
    prompt: str
    interval_seconds: int
    next_fire_at: float


class CompactionDetails(ProtocolModel):
    current_context_tokens: int | None = None
    threshold: int | None = None
    summary_length: int | None = None
    old_session_id: str | None = None
    new_session_id: str | None = None
