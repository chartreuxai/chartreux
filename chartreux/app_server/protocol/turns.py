"""Turn queue, turn start/steer/interrupt, context inject, and callback wire models."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.models import (
    CallbackOutput,
    ContentBlock,
    MentionStats,
    PublicCallbackEntry,
    PublicHistoryEntry,
    PublicTurn,
    PublicTurnQueue,
    SessionContentBlock,
    TurnInputEntry,
    TurnUserInputEntry,
    UserDisplayContent,
    validate_turn_input_entries,
)
from chartreux.app_server.protocol.common import EventWatermarkResponse


class _TurnQueueInputParams(ProtocolModel):
    idempotency_key: str | None = None
    session_id: str
    entries: list[TurnInputEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        validate_turn_input_entries(self.entries)
        return self

    @property
    def input(self) -> list[SessionContentBlock]:
        return [block for entry in self.entries for block in entry.content]

    @property
    def user_entry(self) -> TurnUserInputEntry | None:
        return next(
            (entry for entry in reversed(self.entries) if entry.role == "user"), None
        )

    @property
    def message_entry_id(self) -> str | None:
        user_entry = self.user_entry
        return user_entry.entry_id if user_entry is not None else None


class TurnEnqueueParams(_TurnQueueInputParams):
    pass


class TurnEnqueueResponse(ProtocolModel):
    queue_item_id: str


class TurnQueueReplaceParams(_TurnQueueInputParams):
    queue_item_id: str

    def as_enqueue_params(self) -> TurnEnqueueParams:
        return TurnEnqueueParams(
            idempotency_key=self.idempotency_key,
            session_id=self.session_id,
            entries=self.entries,
        )


class TurnQueueReplaceResponse(ProtocolModel):
    queue_item_id: str


class TurnQueueReadParams(ProtocolModel):
    session_id: str


class TurnQueueReadResponse(ProtocolModel):
    queue: PublicTurnQueue


class TurnQueueRemoveParams(ProtocolModel):
    session_id: str
    queue_item_id: str


class TurnQueueRemoveResponse(ProtocolModel):
    pass


class TurnQueueResumeParams(ProtocolModel):
    session_id: str


class TurnQueueResumeResponse(ProtocolModel):
    pass


class TurnStartParams(ProtocolModel):
    idempotency_key: str | None = None
    session_id: str
    message: list[ContentBlock]
    injected: bool = False
    user_initiated_retry: bool = False
    client_user_message_id: str | None = None
    auto_title: str | None = None
    user_display_content: UserDisplayContent | None = None
    mention_stats: MentionStats | None = None

    @property
    def input(self) -> list[ContentBlock]:
        return self.message


class TurnStartResponse(EventWatermarkResponse):
    turn: PublicTurn


class TurnSteerParams(ProtocolModel):
    require_waiting_only: bool = False
    idempotency_key: str | None = None
    session_id: str
    expected_turn_id: str
    message: list[ContentBlock]
    client_user_message_id: str | None = None
    inject_invoked_skill: bool = True
    mention_stats: MentionStats | None = None

    @property
    def input(self) -> list[ContentBlock]:
        return self.message


class TurnSteerResponse(EventWatermarkResponse):
    accepted: Literal[True] = True


class TurnInterruptParams(ProtocolModel):
    session_id: str
    expected_turn_id: str


class TurnInterruptResponse(EventWatermarkResponse):
    accepted: Literal[True] = True


class ContextInjectParams(ProtocolModel):
    session_id: str
    input: list[ContentBlock]
    as_message: bool = False
    inject_invoked_skill: bool = False
    client_user_message_id: str | None = None
    mention_stats: MentionStats | None = None


class ContextInjectResponse(ProtocolModel):
    entries: list[PublicHistoryEntry]


class CallbackCallParams(ProtocolModel):
    callback: PublicCallbackEntry


class CallbackCallResponse(ProtocolModel):
    callback_id: str
    accepted: bool = True


class CallbackRespondParams(ProtocolModel):
    session_id: str
    callback_id: str
    output: CallbackOutput


class CallbackRespondResponse(ProtocolModel):
    status: Literal["accepted", "duplicate"]


class CallbackResultError(ProtocolModel):
    message: str
    code: str | None = None
    details: JsonValue = None


class CallbackResult(ProtocolModel):
    callback_id: str
    output: JsonValue = None
    error: CallbackResultError | None = None


class CallbackResultParams(ProtocolModel):
    session_id: str
    result: CallbackResult

    @property
    def callback_id(self) -> str:
        return self.result.callback_id


class CallbackResultResponse(EventWatermarkResponse):
    accepted: Literal[True] = True


def validate_callback_acknowledgement(
    callback_id: str, response: CallbackCallResponse
) -> CallbackCallResponse:
    if callback_id != response.callback_id:
        raise ValueError("Callback acknowledgement does not match the request")
    return response
