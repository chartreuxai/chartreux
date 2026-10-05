"""Tests for the live app-server wire models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from pydantic import ValidationError
import pytest

from chartreux.app_server._model import validate_wire
from chartreux.app_server.models import (
    IdleSessionStatus,
    MCPSourceSummary,
    PublicEntryGenerationStatus,
    PublicQueuedTurn,
    PublicRetryCategory,
    PublicRetryState,
    PublicSession,
    PublicSessionState,
    PublicTurn,
    PublicTurnQueue,
    PublicTurnStatus,
    UsageComponentBreakdown,
    UsageComponentSummary,
    UsageCoverageWarning,
    UsageModelSummary,
    UsageTotals,
    UsageWindowSummaries,
    UsageWindowSummary,
)
from chartreux.app_server.protocol import (
    SERVER_METHODS,
    AgentConfig,
    AgentEvictionModel,
    AgentsCancelParams,
    AgentsCancelResponse,
    AgentSummaryModel,
    AgentsUpdateParams,
    AgentTranscriptEntry,
    AgentTranscriptEntryKind,
    AgentTranscriptGetParams,
    AgentTranscriptGetResponse,
    AgentTranscriptState,
    AgentTranscriptTruncation,
    CallbackResult,
    CallbackResultError,
    CallbackResultResponse,
    CancelOutcome,
    EventWatermarkResponse,
    InitializeParams,
    MCPReadParams,
    MCPToggleParams,
    MessageAnnotations,
    PageRequest,
    SessionContinueResponse,
    SessionEmbeddedResourceContentBlock,
    SessionForkResponse,
    SessionHistoryListParams,
    SessionImageContentBlock,
    SessionReadResponse,
    SessionResourceLinkContentBlock,
    SessionResumeResponse,
    SessionShellCommandResponse,
    SessionStartParams,
    SessionStartResponse,
    SessionTextContentBlock,
    TurnContextInputEntry,
    TurnEnqueueParams,
    TurnEnqueueResponse,
    TurnInterruptResponse,
    TurnQueueReadResponse,
    TurnQueueRemoveParams,
    TurnQueueRemoveResponse,
    TurnQueueReplaceParams,
    TurnQueueReplaceResponse,
    TurnQueueResumeResponse,
    TurnQueueUpdatedParams,
    TurnStartResponse,
    TurnSteerResponse,
    TurnUserInputEntry,
    UsageReadParams,
    UsageReadResponse,
    UsageUpdatedParams,
    server_notification_registry,
    validate_notification_method,
)
from chartreux.user_content import UserDisplayContent


def _usage_summaries() -> UsageWindowSummaries:
    def summary(end: str) -> UsageWindowSummary:
        return UsageWindowSummary(
            start_local=datetime.fromisoformat("2026-06-01T00:00:00+02:00"),
            end_local=datetime.fromisoformat(end + "T00:00:00+02:00"),
            start_utc=datetime.fromisoformat("2026-05-31T22:00:00+00:00"),
            end_utc=datetime.fromisoformat(end + "T00:00:00+02:00").astimezone(UTC),
            timezone="Europe/Paris",
            requests=2,
            input_tokens=100,
            cached_input_tokens=20,
            output_tokens=10,
            known_cost_usd=0.25,
            has_known_cost=True,
            has_unknown_cost=True,
            has_unknown_tokens=True,
        )

    return UsageWindowSummaries(
        day=summary("2026-06-02"),
        week=summary("2026-06-08"),
        month=summary("2026-07-01"),
    )


def test_usage_read_defaults_and_host_level_params() -> None:
    assert "usage/read" in SERVER_METHODS
    params = validate_wire(UsageReadParams, {})
    assert params.window == "day"
    assert params.project_key is None
    assert "session_id" not in UsageReadParams.model_fields
    assert params.model_dump(mode="json", exclude_none=True) == {"window": "day"}
    params = UsageReadParams(window="month", project_key="git:/repo/.git")
    wire = {"window": "month", "projectKey": "git:/repo/.git"}
    assert params.model_dump(mode="json", exclude_none=True) == wire
    assert validate_wire(UsageReadParams, wire) == params
    assert validate_wire(UsageReadParams, {"projectKey": None}).project_key is None


@pytest.mark.parametrize(
    "wire",
    [
        {"window": "year"},
        {"window": None},
        {"project_key": "private"},
        {"projectKey": 1},
        {"sessionId": "session"},
        {"extra": True},
    ],
)
def test_usage_read_rejects_invalid_wire_params(wire: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        validate_wire(UsageReadParams, wire)


def test_usage_read_response_wire_round_trip() -> None:
    response = UsageReadResponse(
        as_of=datetime.fromisoformat("2026-06-01T12:00:00+00:00"),
        revision=7,
        summaries=_usage_summaries(),
        window="week",
        project_key="git:/repo/.git",
        models=[
            UsageModelSummary(
                model="alias",
                provider="provider",
                wire_name="deployment",
                requests=2,
                input_tokens=100,
                cached_input_tokens=20,
                output_tokens=10,
                known_cost_usd=0.25,
                has_known_cost=True,
                has_unknown_cost=True,
                has_unknown_tokens=True,
            )
        ],
        components=UsageComponentBreakdown(
            uncached_input=UsageComponentSummary(
                tokens=80, known_cost_usd=0.1, has_known_cost=True
            ),
            cached_input=UsageComponentSummary(
                tokens=20, known_cost_usd=0.05, has_known_cost=True
            ),
            output=UsageComponentSummary(
                tokens=10,
                known_cost_usd=0.1,
                has_known_cost=True,
                has_unknown_cost=True,
                has_unknown_tokens=True,
            ),
        ),
        warnings=[UsageCoverageWarning(code="malformed-record", record_id="record")],
    )
    wire = response.model_dump(mode="json", exclude_none=True)
    assert wire["asOf"] == "2026-06-01T12:00:00Z"
    assert wire["projectKey"] == "git:/repo/.git"
    assert wire["summaries"]["day"]["startLocal"] == "2026-06-01T00:00:00+02:00"
    assert wire["summaries"]["day"]["startUtc"] == "2026-05-31T22:00:00Z"
    assert wire["summaries"]["day"]["currency"] == "USD"
    assert wire["models"][0]["wireName"] == "deployment"
    assert wire["models"][0]["cachedInputTokens"] == 20
    assert wire["components"]["uncachedInput"]["tokens"] == 80
    assert wire["warnings"] == [{"code": "malformed-record", "recordId": "record"}]
    assert validate_wire(UsageReadResponse, wire) == response
    assert UsageReadResponse.model_validate_json(response.model_dump_json()) == response


def test_usage_read_response_pre_session_defaults() -> None:
    response = UsageReadResponse(
        as_of=datetime.fromisoformat("2026-06-01T12:00:00+00:00"),
        revision=0,
        summaries=_usage_summaries(),
    )
    wire = response.model_dump(mode="json", exclude_none=True)
    assert "projectKey" not in wire
    assert response.window == "day"
    assert response.models == []
    assert response.warnings == []
    assert response.components.uncached_input.tokens == 0
    assert validate_wire(UsageReadResponse, wire) == response


def test_usage_updated_global_notification_registration_and_round_trip() -> None:
    summaries = _usage_summaries()
    summaries.day.degraded = True
    params = UsageUpdatedParams(
        as_of=datetime.fromisoformat("2026-06-01T12:00:00+00:00"),
        revision=8,
        summaries=summaries,
        degraded=True,
    )
    assert server_notification_registry()["usage/updated"] is UsageUpdatedParams
    validate_notification_method("usage/updated", params)
    wire = params.model_dump(mode="json", exclude_none=True)
    assert set(wire) == {"asOf", "revision", "summaries", "degraded"}
    assert wire["degraded"] is True
    assert wire["summaries"]["day"]["degraded"] is True
    assert wire["summaries"]["week"]["degraded"] is False
    assert set(wire["summaries"]) == {"day", "week", "month"}
    assert validate_wire(UsageUpdatedParams, wire) == params
    with pytest.raises(ValueError, match="does not match"):
        validate_notification_method("session/statsUpdated", params)
    with pytest.raises(ValidationError):
        validate_wire(UsageUpdatedParams, wire | {"sessionId": "session"})


@pytest.mark.parametrize("model", [UsageTotals, UsageComponentSummary])
@pytest.mark.parametrize("cost", [-1, float("inf"), float("nan")])
def test_usage_rejects_invalid_costs(model: type[Any], cost: float) -> None:
    with pytest.raises(ValidationError):
        model(known_cost_usd=cost)


@pytest.mark.parametrize("revision", [-1, True, 1.5])
@pytest.mark.parametrize("model", [UsageReadResponse, UsageUpdatedParams])
def test_usage_rejects_invalid_revisions(model: type[Any], revision: Any) -> None:
    with pytest.raises(ValidationError):
        model(
            as_of="2026-06-01T12:00:00Z",
            revision=revision,
            summaries=_usage_summaries(),
        )


@pytest.mark.parametrize("model", [UsageReadResponse, UsageUpdatedParams])
def test_usage_rejects_naive_as_of_and_private_wire_spelling(model: type[Any]) -> None:
    wire: dict[str, Any] = {
        "asOf": "2026-06-01T12:00:00Z",
        "revision": 0,
        "summaries": _usage_summaries().model_dump(mode="json"),
    }
    with pytest.raises(ValidationError):
        validate_wire(model, wire | {"asOf": "2026-06-01T12:00:00"})
    with pytest.raises(ValidationError):
        validate_wire(model, wire | {"as_of": wire["asOf"]})
    wire["summaries"]["day"]["input_tokens"] = 100
    with pytest.raises(ValidationError):
        validate_wire(model, wire)


@pytest.mark.parametrize(
    "field", ["requests", "inputTokens", "cachedInputTokens", "outputTokens"]
)
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_usage_rejects_invalid_totals(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        validate_wire(UsageTotals, {field: value})


@pytest.mark.parametrize("state", ["loading", "ready", "unavailable"])
def test_usage_snapshot_states_and_completeness(state: str) -> None:
    totals = UsageTotals.model_validate({"state": state})
    assert totals.requests == 0
    assert totals.has_known_cost is False
    free = UsageTotals(requests=1, has_known_cost=True)
    unknown = UsageTotals(requests=1, has_unknown_cost=True, has_unknown_tokens=True)
    assert free.known_cost_usd == unknown.known_cost_usd == 0
    assert free.has_known_cost is not unknown.has_known_cost


@pytest.mark.parametrize(
    "code",
    [
        "malformed-record",
        "unreadable",
        "write-failed",
        "torn-tail",
        "unsupported-schema",
    ],
)
def test_usage_coverage_warning_wire_contract(code: str) -> None:
    warning = UsageCoverageWarning.model_validate({"code": code})
    assert warning.model_dump(mode="json", exclude_none=True) == {"code": code}


@pytest.mark.parametrize("outcome", list(CancelOutcome))
def test_agents_cancel_public_wire_contract(outcome: CancelOutcome) -> None:
    assert "agents/cancel" in SERVER_METHODS
    assert validate_wire(AgentsCancelParams, {"agentId": "agent"}).run_id is None
    params = validate_wire(AgentsCancelParams, {"agentId": "agent", "runId": "run"})
    assert params.model_dump(mode="json") == {"agentId": "agent", "runId": "run"}
    wire = {"outcome": outcome.value, "runId": "run", "stopReason": "user_cancelled"}
    response = validate_wire(AgentsCancelResponse, wire)
    assert response.outcome is outcome
    assert response.model_dump(mode="json") == wire


@pytest.mark.parametrize(
    "wire",
    [
        {},
        {"agent_id": "a"},
        {"agentId": ""},
        {"agentId": 1},
        {"agentId": "a", "run_id": "r"},
        {"agentId": "a", "runId": ""},
        {"agentId": "a", "extra": True},
    ],
)
def test_agents_cancel_rejects_invalid_wire_params(wire: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        validate_wire(AgentsCancelParams, wire)


def test_agents_cancel_rejects_private_response_spelling() -> None:
    with pytest.raises(ValidationError):
        validate_wire(AgentsCancelResponse, {"outcome": "not_running", "run_id": "r"})
    with pytest.raises(ValidationError):
        validate_wire(AgentsCancelResponse, {"outcome": "cancelled"})


def _turn_queue() -> PublicTurnQueue:
    return PublicTurnQueue(
        items=[
            PublicQueuedTurn(
                id="queue-1",
                created_at=10,
                entries=[
                    TurnUserInputEntry(
                        entry_id="user-2",
                        content=[SessionTextContentBlock(text="next prompt")],
                    )
                ],
            )
        ],
        paused=True,
    )


def test_public_protocol_has_no_session_mcp_methods() -> None:
    assert not [
        method for method in SERVER_METHODS if method.startswith("session/mcp/")
    ]


def test_agent_transcript_get_wire_contract() -> None:
    params = validate_wire(
        AgentTranscriptGetParams, {"agentId": "agent-1", "before": "cursor-1"}
    )
    response = AgentTranscriptGetResponse(
        state=AgentTranscriptState.AVAILABLE,
        entries=[
            AgentTranscriptEntry(
                entry_id="entry-1",
                kind=AgentTranscriptEntryKind.REASONING,
                display_text="thinking",
                digest="a" * 64,
                created_at=0,
                updated_at=0,
                generation_status=PublicEntryGenerationStatus.COMPLETED,
                title="Reasoning",
                truncated=True,
                truncation=AgentTranscriptTruncation.DISPLAY_TEXT_LIMIT,
            )
        ],
        oldest_cursor="cursor-1",
        has_more=True,
    )

    assert response.entries is not None
    assert "agent/transcript/get" in SERVER_METHODS
    assert params.model_dump(mode="json") == {
        "agentId": "agent-1",
        "before": "cursor-1",
        "limit": 50,
    }
    assert response.model_dump(mode="json") == {
        "state": "available",
        "entries": [
            {
                "entryId": "entry-1",
                "kind": "reasoning",
                "displayText": "thinking",
                "digest": "a" * 64,
                "createdAt": 0,
                "updatedAt": 0,
                "generationStatus": "completed",
                "title": "Reasoning",
                "truncated": True,
                "truncation": "display_text_limit",
            }
        ],
        "oldestCursor": "cursor-1",
        "hasMore": True,
    }
    with pytest.raises(ValidationError):
        AgentTranscriptEntry.model_validate(
            response.entries[0].model_dump(mode="json") | {"toolName": "tool"}
        )
    with pytest.raises(ValidationError):
        validate_wire(AgentTranscriptGetParams, {"agent_id": "agent-1"})


@pytest.mark.parametrize("limit", [1, 200])
def test_agent_transcript_get_accepts_limit_bounds(limit: int) -> None:
    assert AgentTranscriptGetParams(agent_id="agent-1", limit=limit).limit == limit


@pytest.mark.parametrize("limit", [0, 201])
def test_agent_transcript_get_rejects_invalid_limit(limit: int) -> None:
    with pytest.raises(ValidationError):
        AgentTranscriptGetParams(agent_id="agent-1", limit=limit)


@pytest.mark.parametrize(
    ("field", "value"), [("agent_id", "a" * 513), ("before", "c" * 4097)]
)
def test_agent_transcript_get_rejects_oversized_identifiers(
    field: str, value: str
) -> None:
    with pytest.raises(ValidationError):
        AgentTranscriptGetParams(**cast(Any, {"agent_id": "agent-1", field: value}))


def test_agent_transcript_states_cannot_masquerade_as_empty_pages() -> None:
    assert (
        AgentTranscriptGetResponse(
            state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
        ).model_dump(mode="json")["state"]
        == "no_saved_transcript"
    )
    with pytest.raises(ValidationError):
        AgentTranscriptGetResponse(state=AgentTranscriptState.AVAILABLE, entries=[])
    with pytest.raises(ValidationError):
        AgentTranscriptGetResponse(
            state=AgentTranscriptState.NO_SAVED_TRANSCRIPT,
            entries=[],
            oldest_cursor=None,
            has_more=False,
        )


def test_agent_transcript_response_does_not_expose_storage_identity() -> None:
    response = AgentTranscriptGetResponse(
        state=AgentTranscriptState.AVAILABLE,
        entries=[],
        oldest_cursor=None,
        has_more=False,
    )

    serialized = response.model_dump(mode="json")
    assert not {
        "path",
        "filePath",
        "childSessionId",
        "sessionId",
        "child_session_id",
    }.intersection(serialized)


def test_mcp_protocol_rejects_removed_legacy_fields() -> None:
    for field in ("source", "kind", "status"):
        with pytest.raises(ValidationError):
            MCPReadParams.model_validate({"sessionId": "session-1", field: "legacy"})
        with pytest.raises(ValidationError):
            MCPToggleParams.model_validate({
                "name": "server",
                "disabled": False,
                field: "legacy",
            })


def test_wire_models_serialize_camel_case_and_reject_snake_case_wire_keys() -> None:
    params = SessionHistoryListParams.model_validate({
        "sessionId": "session-1",
        "page": {"cursor": "entry-1", "limit": 10, "direction": "backward"},
    })

    assert params.model_dump(mode="json") == {
        "sessionId": "session-1",
        "turnId": None,
        "page": {"cursor": "entry-1", "limit": 10, "direction": "backward"},
    }

    with pytest.raises(ValidationError):
        validate_wire(SessionHistoryListParams, {"session_id": "session-1"})


def test_public_session_state_carries_optional_retry_state() -> None:
    session = PublicSession(
        id="session-1", status=IdleSessionStatus(), created_at=1, updated_at=1
    )

    idle = PublicSessionState(event_id=0, session=session)
    retrying = PublicSessionState(
        event_id=1,
        session=session,
        retrying=PublicRetryState(
            turn_id="turn-1",
            category=PublicRetryCategory.RATE_LIMITED,
            detail="HTTP 429",
        ),
    )

    assert idle.model_dump(mode="json")["retrying"] is None
    assert retrying.model_dump(mode="json")["retrying"] == {
        "turnId": "turn-1",
        "category": "rate_limited",
        "detail": "HTTP 429",
    }


def test_public_session_state_carries_optional_runtime_quiescence() -> None:
    session = PublicSession(
        id="session-1", status=IdleSessionStatus(), created_at=1, updated_at=1
    )

    unknown = PublicSessionState(event_id=0, session=session)
    pending = PublicSessionState(event_id=1, session=session, is_quiescent=False)

    assert unknown.model_dump(mode="json")["isQuiescent"] is None
    assert pending.model_dump(mode="json")["isQuiescent"] is False


def test_agent_config_carries_app_server_and_vibe_launch_fields() -> None:
    config = AgentConfig.model_validate({
        "completion": {"type": "mistral", "model": "mistral-large-latest"},
        "sandbox": {"type": "managed", "networkAccess": False},
        "instructions": "Use the project conventions.",
        "workdir": "/workspace",
        "workspaceRoots": ["/workspace", "/shared"],
        "tools": [
            {
                "name": "select_customer",
                "description": "Ask the client to select a customer.",
                "inputSchema": {"type": "object"},
                "outputSchema": {"type": "object"},
            }
        ],
        "hooks": [{"type": "before_tool", "name": "guard"}],
    })

    assert config.completion is not None
    assert config.completion.model == "mistral-large-latest"
    assert config.workdir == "/workspace"
    assert config.workspace_roots == ["/workspace", "/shared"]
    assert config.tools[0].input_schema == {"type": "object"}


def test_session_start_wraps_vibe_configuration_in_agent_config() -> None:
    params = SessionStartParams(
        agent_config=AgentConfig(cwd="/workspace", headless=True), history_limit=100
    )

    assert params.model_dump(mode="json") == {
        "agentConfig": {
            "completion": None,
            "sandbox": None,
            "instructions": "",
            "workdir": None,
            "tools": [],
            "hooks": [],
            "cwd": "/workspace",
            "workspaceRoots": [],
            "worktree": None,
            "enabledTools": None,
            "disabledTools": [],
            "maxTurns": None,
            "maxPrice": None,
            "maxSessionTokens": None,
            "headless": True,
            "trustWorkspace": False,
            "mcpServers": [],
        },
        "historyLimit": 100,
        "idempotencyKey": None,
        "kind": "normal",
    }


def test_initialize_accepts_the_desktop_entrypoint_off_the_wire() -> None:
    """Chartreux Desktop identifies itself here; the value drives analytics attribution."""
    params = validate_wire(
        InitializeParams,
        {
            "clientInfo": {
                "name": "vibe_desktop",
                "title": "Vibe Desktop",
                "version": "1.2.3",
                "entrypoint": "desktop",
            }
        },
    )

    assert params.client_info.entrypoint == "desktop"
    assert params.client_info.name == "vibe_desktop"


def test_page_request_uses_canonical_pagination_shape() -> None:
    assert PageRequest(limit=10, direction="forward").model_dump(mode="json") == {
        "cursor": None,
        "limit": 10,
        "direction": "forward",
    }


def test_public_turn_queue_serializes_order_and_turn_link() -> None:
    queue = _turn_queue()
    turn = PublicTurn(
        id="turn-2",
        session_id="session-1",
        status=PublicTurnStatus.IN_PROGRESS,
        started_at=20,
        queue_item_id="queue-1",
    )

    assert queue.model_dump(mode="json") == {
        "items": [
            {
                "id": "queue-1",
                "createdAt": 10,
                "entries": [
                    {
                        "role": "user",
                        "entryId": "user-2",
                        "content": [{"type": "text", "text": "next prompt"}],
                        "annotations": {},
                    }
                ],
            }
        ],
        "paused": True,
        "maxItems": 32,
    }
    assert turn.model_dump(mode="json")["queueItemId"] == "queue-1"


def test_turn_queue_protocol_models_use_camel_case() -> None:
    queue = _turn_queue()
    params = TurnEnqueueParams(
        idempotency_key="enqueue-1",
        session_id="session-1",
        entries=[
            TurnUserInputEntry(
                entry_id="user-2", content=[SessionTextContentBlock(text="next prompt")]
            )
        ],
    )
    replace = TurnQueueReplaceParams(
        idempotency_key="replace-1",
        session_id="session-1",
        queue_item_id="queue-1",
        entries=params.entries,
    )
    remove = TurnQueueRemoveParams(session_id="session-1", queue_item_id="queue-1")
    notification = TurnQueueUpdatedParams(
        event_id=3, session_id="session-1", emitted_at=30, queue=queue
    )

    assert params.model_dump(mode="json") == {
        "idempotencyKey": "enqueue-1",
        "sessionId": "session-1",
        "entries": [
            {
                "role": "user",
                "entryId": "user-2",
                "content": [{"type": "text", "text": "next prompt"}],
                "annotations": {},
            }
        ],
    }
    assert params.input == params.entries[0].content
    assert replace.model_dump(mode="json") == {
        "idempotencyKey": "replace-1",
        "sessionId": "session-1",
        "entries": params.model_dump(mode="json")["entries"],
        "queueItemId": "queue-1",
    }
    assert remove.model_dump(mode="json") == {
        "sessionId": "session-1",
        "queueItemId": "queue-1",
    }
    assert notification.model_dump(mode="json") == {
        "eventId": 3,
        "sessionId": "session-1",
        "emittedAt": 30,
        "queue": queue.model_dump(mode="json"),
    }

    assert (
        TurnEnqueueParams(
            session_id="session-1",
            entries=[
                TurnUserInputEntry(
                    content=[SessionTextContentBlock(text="next prompt")]
                )
            ],
        ).idempotency_key
        is None
    )


def test_canonical_enqueue_rejects_the_old_vibe_payload() -> None:
    with pytest.raises(ValidationError):
        validate_wire(
            TurnEnqueueParams,
            {
                "sessionId": "session-1",
                "messageEntryId": "user-2",
                "message": [{"type": "text", "text": "next prompt"}],
                "replaceQueueItemId": "queue-1",
            },
        )


def test_canonical_enqueue_accepts_context_user_content_and_annotations() -> None:
    display = UserDisplayContent(
        version="1", host="vibe", content=[{"type": "text", "text": "display text"}]
    )
    params = TurnEnqueueParams(
        session_id="session-1",
        entries=[
            TurnContextInputEntry(
                entry_id="context-1",
                content=[SessionTextContentBlock(text="hidden context")],
            ),
            TurnUserInputEntry(
                entry_id="user-1",
                content=[
                    SessionTextContentBlock(text="visible prompt"),
                    SessionImageContentBlock(
                        uri="data:image/png;base64,aGVsbG8=",
                        media_type="image/png",
                        alt_text="image.png",
                    ),
                    SessionResourceLinkContentBlock(
                        uri="file:///workspace/notes.md", name="notes.md"
                    ),
                    SessionEmbeddedResourceContentBlock(
                        uri="file:///workspace/context.txt",
                        media_type="text/plain",
                        text="attached context",
                    ),
                ],
                annotations=MessageAnnotations.model_validate({
                    "vibe.userDisplayContent": display
                }),
            ),
        ],
    )

    dumped = params.model_dump(mode="json")

    assert [entry["role"] for entry in dumped["entries"]] == ["context", "user"]
    assert [block["type"] for block in dumped["entries"][1]["content"]] == [
        "text",
        "image",
        "resource_link",
        "embedded_resource",
    ]
    assert dumped["entries"][1]["annotations"] == {
        "vibe.userDisplayContent": display.model_dump(mode="json")
    }


def test_legacy_user_display_annotation_wire_key_is_preserved() -> None:
    annotations = MessageAnnotations.model_validate({
        "vibe.userDisplayContent": {
            "version": "1",
            "host": "chartreux",
            "content": [{"type": "text", "text": "display text"}],
        }
    })

    assert annotations.chartreux_user_display_content is not None
    assert annotations.model_dump(mode="json") == {
        "vibe.userDisplayContent": {
            "version": "1",
            "host": "chartreux",
            "content": [{"type": "text", "text": "display text"}],
        }
    }


def test_legacy_public_session_state_format_is_preserved() -> None:
    session = PublicSession(
        id="session-1", status=IdleSessionStatus(), created_at=0, updated_at=0
    )
    state = PublicSessionState(event_id=0, session=session)

    assert state.format == "vibe.public-session-state/v1"
    assert state.model_dump(mode="json")["format"] == "vibe.public-session-state/v1"


def test_canonical_enqueue_requires_the_user_entry_to_be_last() -> None:
    with pytest.raises(ValidationError, match="final turn input entry"):
        TurnEnqueueParams(
            session_id="session-1",
            entries=[
                TurnUserInputEntry(
                    content=[SessionTextContentBlock(text="visible prompt")]
                ),
                TurnContextInputEntry(
                    content=[SessionTextContentBlock(text="hidden context")]
                ),
            ],
        )


def test_turn_queue_methods_are_advertised() -> None:
    assert {
        "app_server/session/turn/enqueue",
        "app_server/session/turn/queue/read",
        "app_server/session/turn/queue/remove",
        "app_server/session/turn/queue/replace",
        "app_server/session/turn/queue/resume",
    }.issubset(SERVER_METHODS)
    assert {
        "turn/enqueue",
        "turn/queue/read",
        "turn/queue/remove",
        "turn/queue/resume",
        "vibe/turn/queue/replace",
    }.isdisjoint(SERVER_METHODS)


def test_feedback_methods_are_not_advertised() -> None:
    assert {"feedback/record", "feedback/shouldShow"}.isdisjoint(SERVER_METHODS)


def test_turn_queue_command_results_are_minimal() -> None:
    assert TurnEnqueueResponse(queue_item_id="queue-1").model_dump(mode="json") == {
        "queueItemId": "queue-1"
    }
    assert TurnQueueReplaceResponse(queue_item_id="queue-1").model_dump(
        mode="json"
    ) == {"queueItemId": "queue-1"}
    assert TurnQueueReadResponse(queue=_turn_queue()).model_dump(mode="json") == {
        "queue": _turn_queue().model_dump(mode="json")
    }
    assert TurnQueueRemoveResponse().model_dump(mode="json") == {}
    assert TurnQueueResumeResponse().model_dump(mode="json") == {}


@pytest.mark.parametrize(
    "response_type",
    [
        SessionStartResponse,
        SessionReadResponse,
        SessionResumeResponse,
        SessionContinueResponse,
        SessionForkResponse,
        TurnStartResponse,
        TurnSteerResponse,
        TurnInterruptResponse,
        CallbackResultResponse,
    ],
)
def test_event_watermark_responses_share_a_base(
    response_type: type[EventWatermarkResponse],
) -> None:
    assert issubclass(response_type, EventWatermarkResponse)


def test_event_watermark_defaults_and_shell_requires_an_event_id() -> None:
    assert TurnSteerResponse().model_dump(mode="json") == {
        "lastEventId": 0,
        "accepted": True,
    }

    with pytest.raises(ValidationError):
        validate_wire(SessionShellCommandResponse, {"accepted": True})


def test_callback_result_accepts_rfc_output_or_error_shapes() -> None:
    assert CallbackResult.model_validate({
        "callbackId": "callback-1",
        "output": {"approved": True},
    }).model_dump(mode="json") == {
        "callbackId": "callback-1",
        "output": {"approved": True},
        "error": None,
    }
    assert CallbackResult(
        callback_id="callback-1",
        error=CallbackResultError(
            message="Client tool is unavailable",
            code="client_unavailable",
            details={"retryable": False},
        ),
    ).model_dump(mode="json") == {
        "callbackId": "callback-1",
        "output": None,
        "error": {
            "message": "Client tool is unavailable",
            "code": "client_unavailable",
            "details": {"retryable": False},
        },
    }

    assert CallbackResult(callback_id="callback-1").model_dump(mode="json") == {
        "callbackId": "callback-1",
        "output": None,
        "error": None,
    }


def test_direct_mcp_source_has_no_plugin_owner_field() -> None:
    configured = validate_wire(
        MCPSourceSummary,
        {"name": "linear", "transport": "streamable-http", "status": "connected"},
    )
    assert "plugin_name" not in MCPSourceSummary.model_fields
    assert "pluginName" not in configured.model_dump(mode="json")
    for key in ("pluginName", "plugin_name"):
        with pytest.raises(ValidationError):
            validate_wire(
                MCPSourceSummary,
                {
                    "name": "linear",
                    "kind": "server",
                    "transport": "streamable-http",
                    "status": "connected",
                    key: "removed",
                },
            )


def test_agents_update_carries_retention_metadata() -> None:
    update = AgentsUpdateParams(
        event_id=1,
        session_id="session-1",
        emitted_at=2,
        agents=[
            AgentSummaryModel(
                agent_id="agent-1",
                profile="worker",
                availability="evicted",
                initial_task_summary="Initial task",
                current_task_summary=None,
                idle_seconds=3.5,
                ttl_remaining_seconds=42.0,
                effective_model="strong",
                base_model="base",
                active_provider="test-provider",
                effective_thinking="high",
                result_expired=True,
            )
        ],
        evictions=[
            AgentEvictionModel(
                agent_id="agent-1",
                run_id="run-1",
                reason="ttl",
                idle_duration_seconds=3.5,
                root_generation=4,
            )
        ],
    )

    assert update.model_dump(mode="json")["agents"][0] == {
        "agentId": "agent-1",
        "profile": "worker",
        "availability": "evicted",
        "currentRunId": None,
        "currentRunStatus": None,
        "turnsUsed": None,
        "lastRunStatus": None,
        "stopReason": None,
        "contextTokens": None,
        "contextWindow": None,
        "compacting": False,
        "initialTaskSummary": "Initial task",
        "currentTaskSummary": None,
        "idleSeconds": 3.5,
        "runElapsedSeconds": None,
        "latestRunId": None,
        "ttlRemainingSeconds": 42.0,
        "effectiveModel": "strong",
        "baseModel": "base",
        "activeProvider": "test-provider",
        "effectiveThinking": "high",
        "resultExpired": True,
    }
    assert update.model_dump(mode="json")["evictions"][0]["reason"] == "ttl"


def test_agent_context_contract_accepts_older_wire_payloads() -> None:
    old = {"agentId": "child", "profile": "worker", "availability": "idle"}
    summary = validate_wire(AgentSummaryModel, old)
    assert summary.context_tokens is None and summary.context_window is None
    assert summary.compacting is False and summary.stop_reason is None
    assert summary.run_elapsed_seconds is None and summary.latest_run_id is None
    current = old | {
        "runElapsedSeconds": 252.0,
        "latestRunId": "run-1",
        "contextTokens": 135000,
        "contextWindow": 400000,
        "compacting": True,
    }
    dump = validate_wire(AgentSummaryModel, current).model_dump(mode="json")
    assert {key: dump[key] for key in current} == current


@pytest.mark.parametrize(
    "reason",
    [
        "user_cancelled",
        "orchestrator_cancelled",
        "retasked",
        "budget_exceeded",
        "budget_unverifiable",
        "error",
    ],
)
def test_run_stop_reason_contract(reason: str) -> None:
    from chartreux.app_server.models import PublicTurnStopReason, RunStopReason
    from chartreux.core.subagents import (
        AgentAvailability,
        AgentSummary,
        RunStatus,
        RunStopReason as CoreRunStopReason,
    )

    assert {item.value for item in RunStopReason} == {
        item.value for item in CoreRunStopReason
    }
    assert PublicTurnStopReason(reason).value == reason
    core = AgentSummary(
        "child",
        "worker",
        AgentAvailability.IDLE,
        "run",
        RunStatus.COMPLETED,
        stop_reason=CoreRunStopReason(reason),
    )
    wire = AgentSummaryModel(
        agent_id=core.agent_id,
        profile=core.profile,
        availability=core.availability.value,
        stop_reason=RunStopReason(core.stop_reason),
    )
    assert (
        validate_wire(AgentSummaryModel, wire.model_dump(mode="json")).stop_reason
        == reason
    )
    with pytest.raises(ValidationError):
        validate_wire(
            AgentSummaryModel,
            {
                "agentId": "child",
                "profile": "worker",
                "availability": "idle",
                "stopReason": "guess",
            },
        )


def test_posting_times_are_optional_and_independent_of_lifecycle_ordinals() -> None:
    from chartreux.app_server.models import PublicMessageEntry

    message = {
        "id": "m",
        "sessionId": "s",
        "createdAt": 10,
        "updatedAt": 11,
        "generationStatus": "completed",
        "role": "user",
        "content": [],
    }
    transcript = {
        "entryId": "m",
        "kind": "user_text",
        "displayText": "hello",
        "digest": "a" * 64,
        "createdAt": 0,
        "updatedAt": 1,
        "generationStatus": "completed",
        "title": "User",
    }
    for model, old in (
        (PublicMessageEntry, message),
        (AgentTranscriptEntry, transcript),
    ):
        assert validate_wire(model, old).posted_at is None
        stamped = validate_wire(model, old | {"postedAt": "2026-06-01T12:34:56Z"})
        dump = stamped.model_dump(mode="json")
        assert dump["postedAt"] == "2026-06-01T12:34:56Z"
        assert (
            dump["createdAt"] == old["createdAt"]
            and dump["updatedAt"] == old["updatedAt"]
        )
