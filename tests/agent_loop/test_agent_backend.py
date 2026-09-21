from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from chartreux.core.config import ChartreuxConfigSchema, ModelConfig, ProviderConfig
from chartreux.core.errors import RefusalError
from chartreux.core.llm.backend.generic import GenericBackend
from chartreux.core.llm.exceptions import IncompleteStreamError
from chartreux.core.llm_models import (
    Backend,
    FunctionCall,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    StopInfo,
    ToolCall,
)
from chartreux.core.tools.base import BaseToolConfig, ToolPermission
from tests.conftest import ConfigBuilder, build_test_agent_loop, make_test_models
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _two_model_vibe_config(
    active_model: str, build_config: ConfigBuilder
) -> ChartreuxConfigSchema:
    """ChartreuxConfigSchema with two models so we can switch active_model."""
    models = [
        ModelConfig(
            name="mistral-vibe-cli-latest", provider="mistral", alias="devstral-latest"
        ),
        ModelConfig(
            name="devstral-small-latest", provider="mistral", alias="devstral-small"
        ),
    ]
    providers = [
        ProviderConfig(
            name="mistral",
            api_base="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            backend=Backend.MISTRAL,
        )
    ]
    return build_config(active_model=active_model, models=models, providers=providers)


@pytest.mark.asyncio
async def test_passes_x_affinity_header_when_asking_an_answer(
    vibe_config: ChartreuxConfigSchema,
):
    backend = FakeBackend([mock_llm_chunk(content="Response")])
    agent = build_test_agent_loop(config=vibe_config, backend=backend)

    [_ async for _ in agent.act("Hello")]

    assert len(backend.requests_extra_headers) > 0
    headers = backend.requests_extra_headers[0]
    assert headers is not None
    assert "x-affinity" in headers
    assert headers["x-affinity"] == agent.session_id


@pytest.mark.asyncio
async def test_passes_x_affinity_header_when_asking_an_answer_streaming(
    vibe_config: ChartreuxConfigSchema,
):
    backend = FakeBackend([mock_llm_chunk(content="Response")])
    agent = build_test_agent_loop(
        config=vibe_config, backend=backend, enable_streaming=True
    )

    [_ async for _ in agent.act("Hello")]

    assert len(backend.requests_extra_headers) > 0
    headers = backend.requests_extra_headers[0]
    assert headers is not None
    assert "x-affinity" in headers
    assert headers["x-affinity"] == agent.session_id


@pytest.mark.asyncio
async def test_max_tokens_is_passed_to_backend(vibe_config: ChartreuxConfigSchema):
    backend = FakeBackend([mock_llm_chunk(content="Response")])
    agent = build_test_agent_loop(config=vibe_config, backend=backend)

    agent.set_max_tokens(8192)
    [_ async for _ in agent.act("Hello")]

    assert backend.requests_max_tokens == [8192]


@pytest.mark.asyncio
async def test_max_tokens_is_passed_to_streaming_backend(
    vibe_config: ChartreuxConfigSchema,
):
    backend = FakeBackend([mock_llm_chunk(content="Response")])
    agent = build_test_agent_loop(
        config=vibe_config, backend=backend, enable_streaming=True
    )

    agent.set_max_tokens(8192)
    [_ async for _ in agent.act("Hello")]

    assert backend.requests_max_tokens == [8192]


@pytest.mark.asyncio
async def test_updates_tokens_stats_based_on_backend_response(
    vibe_config: ChartreuxConfigSchema,
):
    chunk = mock_llm_chunk(content="Response", prompt_tokens=100, completion_tokens=50)
    backend = FakeBackend([chunk])
    agent = build_test_agent_loop(config=vibe_config, backend=backend)

    [_ async for _ in agent.act("Hello")]

    assert agent.stats.context_tokens == 150


@pytest.mark.asyncio
async def test_updates_tokens_stats_based_on_backend_response_streaming(
    vibe_config: ChartreuxConfigSchema,
):
    final_chunk = mock_llm_chunk(
        content="Complete", prompt_tokens=200, completion_tokens=75
    )
    backend = FakeBackend([final_chunk])
    agent = build_test_agent_loop(
        config=vibe_config, backend=backend, enable_streaming=True
    )

    [_ async for _ in agent.act("Hello")]

    assert agent.stats.context_tokens == 275


@pytest.mark.asyncio
async def test_streaming_without_finish_reason_records_partial_response(
    vibe_config: ChartreuxConfigSchema,
):
    backend = FakeBackend([mock_llm_chunk(content="partial", stop_reason=None)])
    agent = build_test_agent_loop(
        config=vibe_config, backend=backend, enable_streaming=True
    )
    agent.stats.context_tokens = 99

    with pytest.raises(IncompleteStreamError):
        [_ async for _ in agent.act("Hello")]

    assert agent.messages[-1].role is Role.assistant
    assert agent.messages[-1].content == "partial"
    assert agent.stats.context_tokens == 15


@pytest.mark.asyncio
async def test_streaming_without_finish_reason_allowed_when_provider_opts_out(
    build_config: ConfigBuilder,
):
    config = build_config(
        providers=[
            ProviderConfig(
                name="mistral",
                api_base="https://api.mistral.ai/v1",
                api_key_env_var="MISTRAL_API_KEY",
                backend=Backend.GENERIC,
                emits_finish_reason=False,
            )
        ]
    )
    backend = FakeBackend([mock_llm_chunk(content="complete", stop_reason=None)])
    agent = build_test_agent_loop(config=config, backend=backend, enable_streaming=True)

    events = [event async for event in agent.act("Hello")]

    assert events
    assert agent.messages[-1].role is Role.assistant
    assert agent.messages[-1].content == "complete"


@pytest.mark.asyncio
async def test_passes_session_id_to_backend(vibe_config: ChartreuxConfigSchema):
    backend = FakeBackend([mock_llm_chunk(content="Response")])
    agent = build_test_agent_loop(config=vibe_config, backend=backend)

    [_ async for _ in agent.act("Hello")]

    assert len(backend.requests_metadata) > 0
    meta = backend.requests_metadata[0]
    assert meta is not None
    assert meta["session_id"] == agent.session_id
    assert "parent_session_id" not in meta
    assert "message_id" in meta
    assert meta["call_type"] == "main_call"


@pytest.mark.asyncio
async def test_passes_parent_session_id_to_backend_after_reset(
    vibe_config: ChartreuxConfigSchema,
):
    backend = FakeBackend([
        [mock_llm_chunk(content="Response")],
        [mock_llm_chunk(content="Response after reset")],
    ])
    agent = build_test_agent_loop(config=vibe_config, backend=backend)

    [_ async for _ in agent.act("Hello")]
    first_session_id = agent.session_id

    await agent._reset_session()
    [_ async for _ in agent.act("Hello again")]

    assert len(backend.requests_metadata) >= 2
    reset_meta = backend.requests_metadata[1]
    assert reset_meta is not None
    assert reset_meta["session_id"] == agent.session_id
    assert reset_meta["parent_session_id"] == first_session_id


@pytest.mark.asyncio
async def test_passes_launch_context_to_backend(vibe_config: ChartreuxConfigSchema):
    launch_context = dict[str, object](
        agent_entrypoint="acp",
        agent_version="2.0.0",
        client_name="vibe_ide",
        client_version="0.5.0",
        terminal_emulator="ghostty",
    )
    backend = FakeBackend([mock_llm_chunk(content="Response")])
    agent = build_test_agent_loop(
        config=vibe_config,
        backend=backend,
        enable_streaming=True,
        launch_context=launch_context,
    )

    [_ async for _ in agent.act("Hello")]

    assert len(backend.requests_metadata) > 0
    meta = backend.requests_metadata[0]
    assert meta is not None
    assert meta["agent_entrypoint"] == "acp"
    assert meta["agent_version"] == "2.0.0"
    assert meta["client_name"] == "vibe_ide"
    assert meta["client_version"] == "0.5.0"
    assert meta["terminal_emulator"] == "ghostty"
    assert meta["session_id"] == agent.session_id
    assert "message_id" in meta
    assert meta["call_type"] == "main_call"


def _generic_provider_vibe_config(build_config: ConfigBuilder) -> ChartreuxConfigSchema:
    """ChartreuxConfigSchema with generic backend so no metadata header is sent."""
    providers = [
        ProviderConfig(
            name="mistral",
            api_base="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            backend=Backend.GENERIC,
        )
    ]
    return build_config(providers=providers)


@pytest.mark.asyncio
async def test_mistral_metadata_includes_user_plan(build_config: ConfigBuilder) -> None:
    backend = FakeBackend([mock_llm_chunk(content="Response")])
    agent = build_test_agent_loop(
        config=_two_model_vibe_config("devstral-latest", build_config), backend=backend
    )
    agent.set_user_plan("Team")

    [_ async for _ in agent.act("Hello")]

    assert len(backend.requests_metadata) == 1
    metadata = backend.requests_metadata[0]
    assert metadata is not None
    assert metadata["user_plan"] == "Team"


@pytest.mark.asyncio
async def test_mistral_metadata_header_call_type_per_turn(
    build_config: ConfigBuilder,
) -> None:
    """First LLM call in a turn is main_call; second call (after tools) is secondary_call."""
    tool_call = ToolCall(
        id="call_1",
        index=0,
        function=FunctionCall(name="todo", arguments='{"action": "read"}'),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="Checking todos.", tool_calls=[tool_call])],
        [mock_llm_chunk(content="Here are your todos.")],
    ])
    config = build_config(
        providers=[
            ProviderConfig(
                name="mistral",
                api_base="https://api.mistral.ai/v1",
                api_key_env_var="MISTRAL_API_KEY",
                backend=Backend.MISTRAL,
            )
        ],
        enabled_tools=["todo"],
        tools={"todo": BaseToolConfig(permission=ToolPermission.ALWAYS)},
    )
    agent = build_test_agent_loop(config=config, backend=backend)

    [_ async for _ in agent.act("What's on my todo list?")]

    assert len(backend.requests_metadata) == 2
    first_metadata = backend.requests_metadata[0]
    second_metadata = backend.requests_metadata[1]
    assert first_metadata is not None
    assert second_metadata is not None
    assert first_metadata["call_type"] == "main_call"
    assert second_metadata["call_type"] == "secondary_call"


@pytest.mark.asyncio
async def test_auto_compact_emits_summary_and_next_turn_metadata(
    build_config: ConfigBuilder,
) -> None:
    """Compact emits summary then user-turn backend metadata in order."""
    backend = FakeBackend([
        [mock_llm_chunk(content="<summary>done</summary>")],
        [mock_llm_chunk(content="<final>")],
    ])
    config = build_config(
        models=make_test_models(auto_compact_threshold=1),
        providers=[
            ProviderConfig(
                name="mistral",
                api_base="https://api.mistral.ai/v1",
                api_key_env_var="MISTRAL_API_KEY",
                backend=Backend.MISTRAL,
            )
        ],
    )
    agent = build_test_agent_loop(config=config, backend=backend)
    agent.stats.context_tokens = 2
    original_session_id = agent.session_id

    [_ async for _ in agent.act("Hello")]

    assert len(backend.requests_metadata) == 2
    assert len(backend.requests_extra_headers) == 2
    compact_metadata = backend.requests_metadata[0]
    user_turn_metadata = backend.requests_metadata[1]
    assert compact_metadata is not None
    assert user_turn_metadata is not None
    assert compact_metadata["call_type"] == "secondary_call"
    assert compact_metadata["session_id"] == original_session_id
    assert "parent_session_id" not in compact_metadata
    assert user_turn_metadata["call_type"] == "main_call"
    assert user_turn_metadata["session_id"] == agent.session_id
    assert "parent_session_id" not in user_turn_metadata

    compact_headers = backend.requests_extra_headers[0]
    user_turn_headers = backend.requests_extra_headers[1]
    assert compact_headers is not None
    assert user_turn_headers is not None
    assert compact_headers["x-affinity"] == original_session_id
    assert user_turn_headers["x-affinity"] == agent.session_id


@pytest.mark.asyncio
async def test_generic_provider_has_no_metadata_header(
    build_config: ConfigBuilder,
) -> None:
    """Non-Mistral provider does not send the metadata header."""
    backend = FakeBackend([mock_llm_chunk(content="Response")])
    config = _generic_provider_vibe_config(build_config)
    agent = build_test_agent_loop(config=config, backend=backend)

    [_ async for _ in agent.act("Hello")]

    assert len(backend.requests_extra_headers) == 1
    headers = backend.requests_extra_headers[0]
    assert headers is not None
    assert "metadata" not in headers


@pytest.mark.asyncio
async def test_provider_extra_headers_are_forwarded_at_backend_boundary() -> None:
    provider = ProviderConfig(
        name="custom",
        api_base="https://custom.example.com/v1",
        extra_headers={"X-Custom-Auth": "token123", "X-Org-Id": "org-456"},
    )
    backend = GenericBackend(provider=provider)
    make_request = AsyncMock(
        return_value={
            "choices": [{"message": {"role": "assistant", "content": "Response"}}]
        }
    )
    backend._make_request = make_request

    await backend.complete(
        model=ModelConfig(name="test-model", provider="custom", alias="test"),
        messages=[LLMMessage(role=Role.user, content="Hello")],
    )

    call = make_request.await_args
    assert call is not None
    headers = call.args[2]
    assert headers["X-Custom-Auth"] == "token123"
    assert headers["X-Org-Id"] == "org-456"


def _refusal_chunk() -> LLMChunk:
    return LLMChunk(
        message=LLMMessage(role=Role.assistant, content=""),
        usage=LLMUsage(prompt_tokens=10, completion_tokens=2),
        stop=StopInfo(reason="refusal"),
    )


@pytest.mark.asyncio
async def test_refusal_stop_reason_raises_refusal_error(
    vibe_config: ChartreuxConfigSchema,
):
    backend = FakeBackend([_refusal_chunk()])
    agent = build_test_agent_loop(config=vibe_config, backend=backend)

    with pytest.raises(RefusalError):
        [_ async for _ in agent.act("Hello")]


@pytest.mark.asyncio
async def test_refusal_stop_reason_raises_refusal_error_streaming(
    vibe_config: ChartreuxConfigSchema,
):
    backend = FakeBackend([_refusal_chunk()])
    agent = build_test_agent_loop(
        config=vibe_config, backend=backend, enable_streaming=True
    )

    with pytest.raises(RefusalError):
        [_ async for _ in agent.act("Hello")]


def _refusal_chunk_with_details() -> LLMChunk:
    return LLMChunk(
        message=LLMMessage(role=Role.assistant, content=""),
        usage=LLMUsage(prompt_tokens=10, completion_tokens=2),
        stop=StopInfo(
            reason="refusal",
            category="cyber",
            explanation="This request was declined for safety reasons.",
        ),
    )


@pytest.mark.asyncio
async def test_refusal_error_carries_category_and_explanation(
    vibe_config: ChartreuxConfigSchema,
):
    backend = FakeBackend([_refusal_chunk_with_details()])
    agent = build_test_agent_loop(config=vibe_config, backend=backend)

    with pytest.raises(RefusalError) as exc_info:
        [_ async for _ in agent.act("Hello")]

    err = exc_info.value
    assert err.category == "cyber"
    assert err.explanation == "This request was declined for safety reasons."
    assert "This request was declined for safety reasons." in str(err)
    assert "cyber" in str(err)


@pytest.mark.asyncio
async def test_refusal_error_carries_category_and_explanation_streaming(
    vibe_config: ChartreuxConfigSchema,
):
    backend = FakeBackend([_refusal_chunk_with_details()])
    agent = build_test_agent_loop(
        config=vibe_config, backend=backend, enable_streaming=True
    )

    with pytest.raises(RefusalError) as exc_info:
        [_ async for _ in agent.act("Hello")]

    err = exc_info.value
    assert err.category == "cyber"
    assert err.explanation == "This request was declined for safety reasons."
    assert "This request was declined for safety reasons." in str(err)
    assert "cyber" in str(err)
