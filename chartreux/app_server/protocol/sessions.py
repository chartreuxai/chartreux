"""Session lifecycle, review, identity, loops, narration, and session log wire models."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal

from pydantic import Field, model_validator

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.models import (
    IdentityView,
    PublicHistoryEntry,
    PublicSession,
    PublicSessionState,
    PublicTurn,
    ScheduledLoop,
    SessionLogSummary,
)
from chartreux.app_server.protocol.common import (
    AgentConfig,
    EventWatermarkResponse,
    PageRequest,
)
from chartreux.app_server.review import (
    ReviewFile,
    ReviewFileStatus,
    ReviewHunk,
    ReviewOwner,
    ReviewScope,
    ReviewTarget,
)


class SessionOpenParams(ProtocolModel):
    agent_config: AgentConfig = Field(default_factory=AgentConfig)
    history_limit: int = Field(default=200, ge=1, le=500)

    @property
    def cwd(self) -> str | None:
        return self.agent_config.cwd or self.agent_config.workdir


class SessionKind(StrEnum):
    """Lifecycle role of a session as seen by the server.

    ``NORMAL`` — a genuine user-initiated session; emits new-session telemetry
    and is persisted to disk as soon as a turn runs.

    ``EPHEMERAL`` — a throwaway session used to warm up the runtime while the
    in-app picker is shown; it is discarded on resume and must not emit
    new-session telemetry or be counted as a new session.
    """

    NORMAL = auto()
    EPHEMERAL = auto()


class SessionStartParams(SessionOpenParams):
    idempotency_key: str | None = None
    kind: SessionKind = SessionKind.NORMAL


class SessionStartResponse(EventWatermarkResponse):
    state: PublicSessionState


class SessionReadParams(ProtocolModel):
    session_id: str
    history: PageRequest | None = Field(default_factory=PageRequest)
    turns: PageRequest | None = Field(default_factory=PageRequest)

    @property
    def include_history(self) -> bool:
        return self.history is not None

    @property
    def include_turns(self) -> bool:
        return self.turns is not None

    @property
    def history_limit(self) -> int:
        return self.history.limit if self.history is not None else 1

    @property
    def turns_limit(self) -> int:
        return self.turns.limit if self.turns is not None else 1


class SessionReadResponse(EventWatermarkResponse):
    state: PublicSessionState


class SessionResumeParams(SessionOpenParams):
    session_id: str


class SessionResumeResponse(EventWatermarkResponse):
    state: PublicSessionState


class SessionContinueParams(SessionOpenParams):
    """Chartreux extension that resumes the latest eligible session."""


class SessionContinueResponse(EventWatermarkResponse):
    state: PublicSessionState


class SessionForkParams(ProtocolModel):
    idempotency_key: str | None = None
    source_session_id: str
    entry_id: str | None = None
    agent_config: AgentConfig | None = None
    after_turn_id: str | None = None
    history_limit: int = Field(default=200, ge=1, le=500)
    attach: bool = True


class SessionForkResponse(EventWatermarkResponse):
    source_session_id: str
    state: PublicSessionState


class SessionStopParams(ProtocolModel):
    session_id: str
    reason: str | None = None


class SessionStopResponse(ProtocolModel):
    closed: bool = True


class SessionCloseParams(ProtocolModel):
    session_id: str


class SessionCloseResponse(ProtocolModel):
    closed: bool = True


class SessionListParams(ProtocolModel):
    cursor: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    root_session_id: str | None = None
    parent_session_id: str | None = None
    cwd: str | None = None


class SessionListResponse(ProtocolModel):
    items: list[PublicSession] = Field(default_factory=list)
    next_cursor: str | None = None
    previous_cursor: str | None = None
    # The session `--continue` would resume: the tty-scoped last-session
    # pointer when it still exists, else the most recently updated session.
    # Resolved server-side so the pointer stays behind the app-server boundary.
    continue_session_id: str | None = None

    @property
    def data(self) -> list[PublicSession]:
        return self.items


class SessionDeleteParams(ProtocolModel):
    session_id: str


class SessionTitleUpdateParams(ProtocolModel):
    session_id: str
    title: str


class SessionTitleUpdateResponse(ProtocolModel):
    title: str
    updated_at: str | None = None
    last_event_id: int | None = None


class SessionHistoryListParams(ProtocolModel):
    session_id: str
    turn_id: str | None = None
    page: PageRequest = Field(default_factory=PageRequest)

    @property
    def cursor(self) -> str | None:
        return self.page.cursor

    @property
    def limit(self) -> int:
        return self.page.limit

    @property
    def sort_direction(self) -> Literal["forward", "backward"]:
        return self.page.direction


class SessionHistoryListResponse(ProtocolModel):
    items: list[PublicHistoryEntry] = Field(default_factory=list)
    next_cursor: str | None = None
    previous_cursor: str | None = None

    @property
    def data(self) -> list[PublicHistoryEntry]:
        return self.items

    @property
    def backwards_cursor(self) -> str | None:
        return self.previous_cursor


class SessionHistoryGetParams(ProtocolModel):
    session_id: str
    history_limit: int = Field(default=200, ge=1, le=500)


class SessionHistoryGetResponse(ProtocolModel):
    history: list[PublicHistoryEntry]


class SessionTurnsListParams(ProtocolModel):
    session_id: str
    page: PageRequest = Field(default_factory=PageRequest)

    @property
    def cursor(self) -> str | None:
        return self.page.cursor

    @property
    def limit(self) -> int:
        return self.page.limit

    @property
    def sort_direction(self) -> Literal["forward", "backward"]:
        return self.page.direction


class SessionTurnsListResponse(ProtocolModel):
    items: list[PublicTurn] = Field(default_factory=list)
    next_cursor: str | None = None
    previous_cursor: str | None = None

    @property
    def data(self) -> list[PublicTurn]:
        return self.items

    @property
    def backwards_cursor(self) -> str | None:
        return self.previous_cursor


class SessionShellCommandParams(ProtocolModel):
    session_id: str
    command: str | None = None
    cwd: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0, le=600)
    operation_id: str | None = None
    action: Literal["run", "interrupt"] = "run"

    @model_validator(mode="after")
    def validate_action(self) -> SessionShellCommandParams:
        if self.action == "run" and (self.command is None or not self.command.strip()):
            raise ValueError("command is required for action='run'")
        if self.action == "interrupt" and self.operation_id is None:
            raise ValueError("operation_id is required for action='interrupt'")
        return self


class SessionShellCommandResponse(ProtocolModel):
    accepted: Literal[True] = True
    last_event_id: int


class SessionReadyWaitParams(ProtocolModel):
    session_id: str


class SessionReadyReadParams(ProtocolModel):
    session_id: str


class SessionReadyReadResponse(ProtocolModel):
    ready: bool


class SessionReadyWaitResponse(ProtocolModel):
    ready: bool = True
    init_duration_ms: int | None = None


class IdentityReadParams(ProtocolModel):
    session_id: str


class IdentityReadResponse(ProtocolModel):
    identity: IdentityView | None = None


class SessionRewindReadParams(ProtocolModel):
    """Inspect a session-local, memory-only checkpoint lost on restart or resume."""

    session_id: str
    entry_id: str


class SessionRewindReadResponse(ProtocolModel):
    has_file_changes: bool
    paths: list[str] = Field(default_factory=list)


class SessionRewindParams(ProtocolModel):
    """Rewind via a session-local, memory-only checkpoint lost on restart or resume."""

    session_id: str
    entry_id: str
    restore_files: bool = False
    inplace: bool = False


class SessionRewindResponse(ProtocolModel):
    message: str
    restore_errors: list[str]
    restored_paths: list[str]
    state: PublicSessionState
    session_log: SessionLogSummary


class SessionRelocateParams(ProtocolModel):
    session_id: str
    cwd: str


class SessionRelocateResponse(ProtocolModel):
    state: PublicSessionState


class ReviewStateParams(ProtocolModel):
    """Read review state backed by session-local, memory-only checkpoints.

    Checkpoints are lost when the process restarts or the session is resumed.
    """

    session_id: str


class ReviewStateResponse(ProtocolModel):
    files: list[ReviewFile]
    scopes: list[ReviewScope]


class ReviewBaselineParams(ProtocolModel):
    session_id: str
    path: str


class ReviewBaselineResponse(ProtocolModel):
    content: str


class ReviewTurnDiffParams(ProtocolModel):
    session_id: str
    path: str
    owner: ReviewOwner


class ReviewTurnDiffResponse(ProtocolModel):
    status: ReviewFileStatus
    baseline: str
    current: str


class ReviewHunksParams(ProtocolModel):
    session_id: str
    path: str
    owner: ReviewOwner | None = None


class ReviewHunksResponse(ProtocolModel):
    hunks: list[ReviewHunk]


class ReviewMutationParams(ProtocolModel):
    session_id: str
    target: ReviewTarget


class SessionLogReadParams(ProtocolModel):
    session_id: str


class SessionLogReadResponse(ProtocolModel):
    log: SessionLogSummary


class LoopsListParams(ProtocolModel):
    session_id: str


class LoopsListResponse(ProtocolModel):
    loops: list[ScheduledLoop]


class LoopsCreateParams(ProtocolModel):
    session_id: str
    interval: str
    prompt: str


class LoopsCreateResponse(ProtocolModel):
    loop: ScheduledLoop


class LoopsDeleteParams(ProtocolModel):
    session_id: str
    loop_id: str


class LoopsDeleteResponse(ProtocolModel):
    loop: ScheduledLoop


class LoopsClearParams(ProtocolModel):
    session_id: str


class LoopsClearResponse(ProtocolModel):
    count: int


class NarrationSummarizeParams(ProtocolModel):
    session_id: str
    user_message: str
    assistant_text: str
    error: str | None = None
    message_id: str | None = None


class NarrationSummarizeResponse(ProtocolModel):
    summary: str | None = None


class SessionHistoryClearParams(ProtocolModel):
    session_id: str


class SessionHistoryClearResponse(ProtocolModel):
    state: PublicSessionState
    session_log: SessionLogSummary


class SessionCompactParams(ProtocolModel):
    session_id: str
    extra_instructions: str = ""


class SessionCompactResponse(ProtocolModel):
    summary: str
    state: PublicSessionState
    session_log: SessionLogSummary
