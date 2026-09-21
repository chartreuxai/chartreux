from __future__ import annotations

from importlib.metadata import version
import json
from unittest.mock import Mock

import httpx
from mistralai.client.utils.retries import BackoffStrategy, RetryConfig
from mistralai.extra.observability import telemetry
from opentelemetry import trace
import pytest
import respx

from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.events import ToolResultEvent
from chartreux.core.llm.backend.factory import create_backend
from chartreux.core.llm.backend.generic import GenericBackend
from chartreux.core.llm.backend.mistral import MistralBackend
from chartreux.core.llm_models import Backend, FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.tools.base import BaseToolConfig, ToolPermission
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


@pytest.fixture(
    params=[
        (setting, global_provider)
        for setting in ("dedicated", "global", "true")
        for global_provider in (False, True)
    ]
)
def hostile_telemetry(request, monkeypatch):
    """Trap instrumentation with and without an installed global provider."""
    setting, global_provider = request.param
    monkeypatch.setenv("MISTRAL_SDK_TELEMETRY", setting)
    monkeypatch.setenv("MISTRAL_API_KEY", "synthetic-test-key")

    def forbidden(*args, **kwargs):
        pytest.fail("Telemetry exporter/provider/span setup must not run")

    tracer = Mock(spec=trace.Tracer)
    tracer.start_span.side_effect = forbidden
    tracer.start_as_current_span.side_effect = forbidden
    provider = (
        Mock(spec=trace.TracerProvider)
        if global_provider
        else trace.ProxyTracerProvider()
    )
    monkeypatch.setattr(provider, "get_tracer", lambda *args, **kwargs: tracer)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(telemetry, "_create_telemetry_tracer_provider", forbidden)
    monkeypatch.setattr(telemetry, "_use_global_tracer_provider", forbidden)
    monkeypatch.setattr(telemetry, "_load_otel_sdk", forbidden)
    yield
    tracer.start_span.assert_not_called()
    tracer.start_as_current_span.assert_not_called()


def _assert_untraced(request: httpx.Request) -> None:
    assert not {"traceparent", "tracestate", "baggage"}.intersection(request.headers)
    assert request.extensions.get("_tracing_span") is None
    assert request.extensions.get("_tracing_span_context") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("enter_context", [False, True])
async def test_real_sdk_inference_opt_out(hostile_telemetry, streaming, enter_context):
    # This is an intentionally pinned internal seam, not a constructor API.
    assert version("mistralai") == "2.6.0"
    retries = []

    async def on_retry(reason):
        retries.append(reason)

    backend = create_backend(
        provider=ProviderConfig(
            name="mistral",
            api_base="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            backend=Backend.MISTRAL,
        ),
        on_retry=on_retry,
    )
    assert isinstance(backend, MistralBackend)
    assert backend._client is None
    backend._retry_config = RetryConfig(
        strategy="backoff",
        backoff=BackoffStrategy(
            initial_interval=1, max_interval=1, exponent=1, max_elapsed_time=1000
        ),
        retry_connection_errors=True,
    )
    clients = []
    responses = []
    requests = []

    def respond(request):
        _assert_untraced(request)
        requests.append(request)
        if len(requests) % 2:
            response = httpx.Response(503, json={"message": "retry"})
        else:
            payload = {
                "id": "cmpl-test",
                "object": "chat.completion.chunk" if streaming else "chat.completion",
                "created": 1234567890,
                "model": "mistral-test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "delta" if streaming else "message": {
                            "role": "assistant",
                            "content": "hi",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 23,
                    "completion_tokens": 11,
                    "total_tokens": 34,
                },
            }
            if streaming:
                response = httpx.Response(
                    200,
                    content=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode(),
                    headers={"Content-Type": "text/event-stream"},
                )
            else:
                response = httpx.Response(200, json=payload)
        responses.append(response)
        return response

    with respx.mock(base_url="https://api.mistral.ai") as api:
        api.post("/v1/chat/completions").mock(side_effect=respond)
        # Reopen the same backend to prove every newly constructed SDK opts out.
        for _ in range(2):
            if enter_context:
                await backend.__aenter__()
            try:
                model = ModelConfig(
                    name="mistral-test", provider="mistral", alias="test"
                )
                messages = [LLMMessage(role=Role.user, content="hi")]
                if streaming:
                    chunks = [
                        chunk
                        async for chunk in backend.complete_streaming(
                            model=model,
                            messages=messages,
                            temperature=0.2,
                            tools=None,
                            max_tokens=None,
                            tool_choice=None,
                            extra_headers={"x-test-caller": "retained"},
                            metadata={"call_type": "main_call"},
                        )
                    ]
                else:
                    chunks = [
                        await backend.complete(
                            model=model,
                            messages=messages,
                            temperature=0.2,
                            tools=None,
                            max_tokens=None,
                            tool_choice=None,
                            extra_headers={"x-test-caller": "retained"},
                            metadata={"call_type": "main_call"},
                        )
                    ]
                assert [chunk.message.content for chunk in chunks] == ["hi"]
                assert chunks[0].usage is not None
                assert chunks[0].usage.prompt_tokens == 23
                assert chunks[0].usage.completion_tokens == 11
                client = backend._client
                assert client is not None
                assert client.sdk_configuration.__dict__["telemetry"] is False
                clients.append(client)
            finally:
                await backend.aclose()
            assert backend._client is None
            assert backend._http_client is None
    assert clients[0] is not clients[1]
    assert len(retries) == 2
    assert len(requests) == 4
    assert all(response.is_closed for response in responses)
    assert all(request.headers["x-test-caller"] == "retained" for request in requests)
    assert all(
        json.loads(request.content)["metadata"] == {"call_type": "main_call"}
        for request in requests
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_generic_inference_without_spans(hostile_telemetry, streaming):
    backend = create_backend(
        provider=ProviderConfig(
            name="generic",
            api_base="https://provider.example/v1",
            api_key_env_var="",
            backend=Backend.GENERIC,
        )
    )
    assert isinstance(backend, GenericBackend)
    payload = {
        "choices": [
            {
                "message" if not streaming else "delta": {
                    "role": "assistant",
                    "content": "hi",
                }
            }
        ],
        "usage": {"prompt_tokens": 23, "completion_tokens": 11},
    }
    response = (
        httpx.Response(
            200,
            content=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode(),
            headers={"Content-Type": "text/event-stream"},
        )
        if streaming
        else httpx.Response(200, json=payload)
    )
    model = ModelConfig(name="test", provider="generic", alias="test")
    messages = [LLMMessage(role=Role.user, content="hi")]
    with respx.mock(base_url="https://provider.example") as api:
        route = api.post("/v1/chat/completions").mock(return_value=response)
        async with backend:
            if streaming:
                chunks = [
                    chunk
                    async for chunk in backend.complete_streaming(
                        model=model, messages=messages
                    )
                ]
            else:
                chunks = [await backend.complete(model=model, messages=messages)]
        assert route.call_count == 1
        _assert_untraced(route.calls[0].request)
    assert [chunk.message.content for chunk in chunks] == ["hi"]
    assert chunks[0].usage is not None
    assert chunks[0].usage.prompt_tokens == 23
    assert chunks[0].usage.completion_tokens == 11
    assert response.is_closed


@pytest.mark.asyncio
async def test_agent_tool_turn_without_spans(hostile_telemetry):
    tool_call = ToolCall(
        id="call_1",
        index=0,
        function=FunctionCall(name="todo", arguments='{"action": "read"}'),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="Let me check.", tool_calls=[tool_call])],
        [mock_llm_chunk(content="Done.")],
    ])
    config = build_test_vibe_config(
        enabled_tools=["todo"],
        tools={"todo": BaseToolConfig(permission=ToolPermission.ALWAYS)},
    )
    agent = build_test_agent_loop(config=config, backend=backend)
    events = [event async for event in agent.act("What are my todos?")]
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].error is None
    assert results[0].result is not None
    assert agent.messages[-1].content == "Done."
