from __future__ import annotations

import asyncio

import pytest

from chartreux.app_server._utils import public_error
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.models import PublicTurnStatus, TurnErrorCode
from chartreux.app_server.session import AppServerTurnError
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop.errors import (
    AgentLoopLLMResponseError,
    EmptyLLMResponseError,
)
from chartreux.core.compaction import CompactionFailedError
from chartreux.core.errors import RefusalError
from chartreux.core.llm.exceptions import (
    BackendError,
    IncompleteStreamError,
    PayloadSummary,
)
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import attach_test_app_server_session, build_test_app_server


def test_public_empty_llm_response_error_has_safe_context() -> None:
    exc = EmptyLLMResponseError("provider", "model")

    assert isinstance(exc, AgentLoopLLMResponseError)
    assert vars(exc) == {"provider": "provider", "model": "model"}
    error = public_error(exc)

    assert error.code == TurnErrorCode.EMPTY_LLM_RESPONSE
    assert error.model_dump(mode="json")["code"] == "empty_llm_response"
    assert error.details == {"provider": "provider", "model": "model"}
    assert error.message == "The model returned an empty assistant response."


def test_public_refusal_error_mapping_is_unchanged() -> None:
    error = public_error(RefusalError("provider", "model", "safety", "declined"))

    assert error.code == TurnErrorCode.REFUSAL
    assert error.model_dump(mode="json")["code"] == "refusal"
    assert error.details == {
        "provider": "provider",
        "model": "model",
        "category": "safety",
        "explanation": "declined",
    }


@pytest.mark.asyncio
async def test_failed_root_turn_publishes_empty_llm_response() -> None:
    agent_loop = build_test_agent_loop()

    async def empty_act(*_args, **_kwargs):
        raise EmptyLLMResponseError("provider", "model")
        yield

    agent_loop.act = empty_act
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    session = await attach_test_app_server_session(
        AppServerClient(client_transport, run_peer=server.serve)
    )

    async def consume_turn() -> None:
        async for _event in session.act("hello"):
            pass

    try:
        with pytest.raises(AppServerTurnError) as exc_info:
            await asyncio.wait_for(consume_turn(), timeout=5)

        assert exc_info.value.error.code == "empty_llm_response"
        assert exc_info.value.error.details == {
            "provider": "provider",
            "model": "model",
        }
        assert session.state.turns
        turn = session.state.turns[-1]
        assert turn.status == PublicTurnStatus.FAILED
        assert turn.error is not None
        assert turn.error.model_dump(mode="json")["code"] == "empty_llm_response"
    finally:
        await session.close()
        await agent_loop.aclose()


def _make_invalid_model_backend_error() -> BackendError:
    return BackendError(
        provider="test-provider",
        endpoint="/v1/chat/completions",
        status=400,
        reason="Bad Request",
        headers={},
        body_text='{"error":{"type":"invalid_model"}}',
        parsed_error=None,
        model="bad-model",
        payload_summary=PayloadSummary(
            model="bad-model",
            message_count=1,
            approx_chars=10,
            temperature=0.0,
            has_tools=False,
            tool_choice=None,
        ),
    )


def test_public_compaction_error_preserves_reason() -> None:
    error = public_error(CompactionFailedError("tool_call"))

    assert error.code == TurnErrorCode.COMPACTION_FAILED
    assert error.model_dump(mode="json")["code"] == "compaction_failed"
    assert error.details == {"reason": "tool_call"}


def test_public_incomplete_stream_error_has_distinct_code() -> None:
    error = public_error(IncompleteStreamError("provider", "model"))

    assert error.code == TurnErrorCode.INCOMPLETE_STREAM
    assert error.details == {"provider": "provider", "model": "model"}


def test_public_error_invalid_model_direct_backend_error() -> None:
    be = _make_invalid_model_backend_error()
    error = public_error(be)

    assert error.code == TurnErrorCode.INVALID_MODEL


def test_public_error_invalid_model_wrapped_runtime_error() -> None:
    be = _make_invalid_model_backend_error()
    wrapped = RuntimeError("API error: ...")
    wrapped.__cause__ = be
    error = public_error(wrapped)

    assert error.code == TurnErrorCode.INVALID_MODEL
