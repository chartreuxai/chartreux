"""Agent transcript wire models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol, Self

from pydantic import Field, JsonValue, model_validator

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.models import (
    EffectDetail,
    EffectState,
    PublicEntryGenerationStatus,
)
from chartreux.utils.tool_presentation import (
    ToolCallPresentation,
    ToolResultPresentation,
)

MAX_AGENT_TRANSCRIPT_ID_LENGTH = 512
MAX_AGENT_TRANSCRIPT_CURSOR_LENGTH = 4096
MAX_AGENT_TRANSCRIPT_DISPLAY_TEXT_LENGTH = 8 * 1024


class AgentTranscriptEntryKind(StrEnum):
    USER_TEXT = "user_text"
    ASSISTANT_TEXT = "assistant_text"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class AgentTranscriptToolStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentTranscriptTruncation(StrEnum):
    DISPLAY_TEXT_LIMIT = "display_text_limit"


class AgentTranscriptEntry(ProtocolModel):
    """One chronological, widget-ready projection entry.

    ``created_at`` and ``updated_at`` are synthetic transcript-order ordinals,
    not wall-clock timestamps. ``posted_at`` is the nullable canonical posting
    time; old messages remain unstamped.
    ``generation_status`` is synthesized as completed because LLM messages have
    no lifecycle marker. ``title`` is a synthesized role/tool label. ``digest``
    is the content-revision hash used by paging; tool-call digests include their
    matched result because it contributes to the projected effect state.
    Attachment metadata includes names and counts only, never image sources.
    """

    entry_id: str = Field(min_length=1, max_length=MAX_AGENT_TRANSCRIPT_ID_LENGTH)
    kind: AgentTranscriptEntryKind
    display_text: str = Field(max_length=MAX_AGENT_TRANSCRIPT_DISPLAY_TEXT_LENGTH)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: int = Field(ge=0)
    updated_at: int = Field(ge=0)
    posted_at: datetime | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    turn_duration_ms: float | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    generation_status: PublicEntryGenerationStatus
    title: str = Field(max_length=MAX_AGENT_TRANSCRIPT_ID_LENGTH)
    tool_name: str | None = Field(
        default=None,
        max_length=MAX_AGENT_TRANSCRIPT_ID_LENGTH,
        exclude_if=lambda value: value is None,
    )
    tool_call_id: str | None = Field(
        default=None,
        max_length=MAX_AGENT_TRANSCRIPT_ID_LENGTH,
        exclude_if=lambda value: value is None,
    )
    arguments: JsonValue = Field(default=None, exclude_if=lambda value: value is None)
    result: dict[str, JsonValue] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    output_text: str | None = Field(
        default=None,
        max_length=MAX_AGENT_TRANSCRIPT_DISPLAY_TEXT_LENGTH,
        exclude_if=lambda value: value is None,
    )
    status: AgentTranscriptToolStatus | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    call_presentation: ToolCallPresentation | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    result_presentation: ToolResultPresentation | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    detail: EffectDetail | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    state: EffectState | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    attachment_names: list[str] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    attachment_count: int = Field(default=0, ge=0, exclude_if=lambda value: value == 0)
    truncated: bool = False
    truncation: AgentTranscriptTruncation | None = None

    @model_validator(mode="after")
    def validate_entry_payload(self) -> Self:
        if self.kind in {
            AgentTranscriptEntryKind.USER_TEXT,
            AgentTranscriptEntryKind.ASSISTANT_TEXT,
            AgentTranscriptEntryKind.REASONING,
        } and (self.tool_name is not None or self.tool_call_id is not None):
            raise ValueError("text entries must not carry tool identity")
        if self.kind is AgentTranscriptEntryKind.TOOL_CALL:
            if self.tool_name is None:
                raise ValueError("tool call entries require tool_name")
            if self.detail is None or self.state is None or self.status is None:
                raise ValueError("tool call entries require effect payload and status")
        if self.kind is AgentTranscriptEntryKind.TOOL_RESULT:
            if self.tool_call_id is None:
                raise ValueError("tool result entries require tool_call_id")
            if self.tool_name is None:
                raise ValueError("tool result entries require tool_name")
            if self.detail is None or self.state is None or self.status is None:
                raise ValueError(
                    "tool result entries require effect payload and status"
                )
        if self.attachment_count < len(self.attachment_names):
            raise ValueError("attachment_count cannot be less than attachment_names")
        if self.truncated != (self.truncation is not None):
            raise ValueError("truncated and truncation must agree")
        return self


class AgentTranscriptGetParams(ProtocolModel):
    """Read a bounded, backward page from an attached parent's agent transcript."""

    agent_id: str = Field(min_length=1, max_length=MAX_AGENT_TRANSCRIPT_ID_LENGTH)
    before: str | None = Field(
        default=None, max_length=MAX_AGENT_TRANSCRIPT_CURSOR_LENGTH
    )
    limit: int = Field(default=50, ge=1, le=200)


class AgentTranscriptState(StrEnum):
    AVAILABLE = "available"
    NO_SAVED_TRANSCRIPT = "no_saved_transcript"
    CHANGED = "changed"
    EXCEEDS_VIEWER_LIMIT = "exceeds_viewer_limit"


class AgentTranscriptGetResponse(ProtocolModel):
    """Typed result of ``agent/transcript/get``; never contains paths or session IDs.

    ``available`` is the only state that carries a page.  It always carries
    ``entries`` (which may be empty for a valid saved-empty transcript),
    ``oldestCursor``, and ``hasMore``.  The other states carry neither page
    fields nor a cursor: ``no_saved_transcript`` means no saved transcript is
    available, ``changed`` means ``Transcript changed; refresh``, and
    ``exceeds_viewer_limit`` means ``Transcript exceeds viewer limit``.

    A ``before`` cursor encodes the stable digest of its boundary entry: its
    persisted message ID when present plus a content-digest hash of that
    entry's canonical serialization.  On page-up, the reader must locate that
    boundary by ID and digest; if it is absent or the digest differs, it must
    return ``changed`` rather than an empty page.  The cursor for a response is
    derived from the oldest entry actually returned.
    """

    state: AgentTranscriptState
    entries: list[AgentTranscriptEntry] | None = None
    oldest_cursor: str | None = Field(
        default=None, max_length=MAX_AGENT_TRANSCRIPT_CURSOR_LENGTH
    )
    has_more: bool | None = None

    @model_validator(mode="after")
    def validate_state_payload(self) -> Self:
        if self.state is AgentTranscriptState.AVAILABLE:
            if self.entries is None or self.has_more is None:
                raise ValueError("available transcript responses require page fields")
        elif (
            self.entries is not None
            or self.oldest_cursor is not None
            or self.has_more is not None
        ):
            raise ValueError("non-available transcript responses must not carry a page")
        return self


class AgentTranscriptSource(Protocol):
    """Transcript reader contract implemented by the session-resources facade."""

    async def read_agent_transcript(
        self, agent_id: str, *, before: str | None = None, limit: int = 50
    ) -> AgentTranscriptGetResponse: ...
