from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.config.models import MissingAPIKeyError
from chartreux.core.llm.backend.factory import create_backend
from chartreux.core.llm.provider_smoke import probe_provider_smoke
from chartreux.core.llm_models import Backend, LLMMessage, Role
from chartreux.utils.api_keys import ApiKeyOrigin, ApiKeySource

CREDENTIAL = ("test-token", ApiKeyOrigin(ApiKeySource.ENVIRONMENT, "TEST_KEY"))


def _response(style, index):
    text = "red" if index == 2 else "323"
    if style == "anthropic":
        content = [{"type": "text", "text": text}]
        if index == 0:
            content = [
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "chartreux_smoke_echo",
                    "input": {"value": "smoke-ok"},
                }
            ]
        return {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": content,
            "stop_reason": "tool_use" if index == 0 else "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 2},
        }
    if style == "openai-responses":
        output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ]
        if index == 0:
            output = [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "chartreux_smoke_echo",
                    "arguments": '{"value":"smoke-ok"}',
                }
            ]
        return {
            "id": "resp-1",
            "status": "completed",
            "output": output,
            "usage": {"input_tokens": 2, "output_tokens": 2},
        }
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if index == 0:
        message["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "chartreux_smoke_echo",
                    "arguments": '{"value":"smoke-ok"}',
                },
            }
        ]
    return {
        "id": "chat-1",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if index == 0 else "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "style", ["openai", "openai-responses", "anthropic", "mistral"]
)
async def test_probe_inputs_reach_actual_adapter_payloads(style):
    provider = ProviderConfig(
        name="target",
        api_base="https://example.test"
        if style == "anthropic"
        else "https://example.test/v1",
        api_key_env_var="TEST_KEY",
        backend=Backend.MISTRAL if style == "mistral" else Backend.GENERIC,
        api_style=style if style != "mistral" else "openai",
    )
    model = ModelConfig(
        name="test-model", alias="test-alias", provider="target", supports_images=True
    )
    endpoint = {"openai-responses": "responses", "anthropic": "messages"}.get(
        style, "chat/completions"
    )
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_response(style, len(payloads) - 1))

    with (
        respx.mock as router,
        patch(
            "chartreux.core.llm.provider_smoke.resolve_api_key_with_origin",
            return_value=CREDENTIAL,
        ) as resolve,
        patch(
            f"chartreux.core.llm.backend.{'mistral' if style == 'mistral' else 'generic'}.resolve_api_key_with_origin",
            side_effect=AssertionError("must not resolve again"),
        ),
    ):
        router.post(f"https://example.test/v1/{endpoint}").mock(side_effect=handler)
        result = await probe_provider_smoke(
            model=model, provider=provider, max_tokens=2048
        )
    assert result.tool.status == result.image.status == "pass"
    assert result.thinking.status == "unverified"
    resolve.assert_called_once()
    assert len(payloads) == 3
    assert all(not payload.get("stream", False) for payload in payloads)
    assert all(
        payload.get("max_tokens", payload.get("max_output_tokens")) == 2048
        for payload in payloads
    )
    tool, thinking, image = payloads
    if style == "anthropic":
        assert tool["tools"][0]["name"] == "chartreux_smoke_echo"
        assert tool["tools"][0]["input_schema"]["properties"]["value"]["enum"] == [
            "smoke-ok"
        ]
        assert tool["tool_choice"] == {"type": "tool", "name": "chartreux_smoke_echo"}
        assert thinking["thinking"]["type"] == "adaptive"
        assert thinking["output_config"]["effort"] == "low"
        part = next(
            part for part in image["messages"][0]["content"] if part["type"] == "image"
        )
        encoded = part["source"]["data"]
        assert part["source"]["media_type"] == "image/png"
    elif style == "openai-responses":
        assert tool["tools"][0]["name"] == "chartreux_smoke_echo"
        assert tool["tool_choice"]["name"] == "chartreux_smoke_echo"
        assert thinking["reasoning"]["effort"] == "low"
        part = next(
            part
            for part in image["input"][0]["content"]
            if part["type"] == "input_image"
        )
        encoded = part["image_url"].split(",", 1)[1]
    else:
        assert tool["tools"][0]["function"]["name"] == "chartreux_smoke_echo"
        assert tool["tools"][0]["function"]["parameters"]["properties"]["value"][
            "enum"
        ] == ["smoke-ok"]
        assert tool["tool_choice"]["function"]["name"] == "chartreux_smoke_echo"
        assert thinking["reasoning_effort"] == ("high" if style == "mistral" else "low")
        part = next(
            part
            for part in image["messages"][0]["content"]
            if part["type"] == "image_url"
        )
        encoded = part["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.asyncio
async def test_responses_truncation_fails_probe_despite_valid_tool_call():
    provider = ProviderConfig(
        name="target",
        api_base="https://example.test/v1",
        api_key_env_var="TEST_KEY",
        api_style="openai-responses",
    )
    model = ModelConfig(
        name="test-model",
        alias="test-alias",
        provider="target",
        supported_thinking_levels=["off"],
    )
    data = _response("openai-responses", 0)
    data.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    with (
        respx.mock as router,
        patch(
            "chartreux.core.llm.provider_smoke.resolve_api_key_with_origin",
            return_value=CREDENTIAL,
        ),
    ):
        route = router.post("https://example.test/v1/responses").mock(
            return_value=httpx.Response(200, json=data)
        )
        result = await probe_provider_smoke(model=model, provider=provider)
    assert route.call_count == 1
    assert result.tool.status == "fail"
    assert result.tool.reason == "output_truncated"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", [Backend.GENERIC, Backend.MISTRAL])
async def test_explicit_absent_credential_fails_fast_without_resolution(backend):
    provider = ProviderConfig(
        name="test",
        api_base="https://example.test/v1",
        api_key_env_var="TEST_KEY",
        backend=backend,
    )
    with patch(
        f"chartreux.core.llm.backend.{backend}.resolve_api_key_with_origin",
        side_effect=AssertionError("must not resolve"),
    ):
        with pytest.raises(MissingAPIKeyError):
            create_backend(provider=provider, resolved_credential=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "style", ["openai", "openai-responses", "anthropic", "mistral"]
)
async def test_probe_never_retries_or_exposes_raw_errors(style, monkeypatch, caplog):
    monkeypatch.setenv("MISTRAL_DEBUG", "1")
    caplog.set_level("DEBUG")
    provider = ProviderConfig(
        name="test",
        api_base="https://example.test"
        if style == "anthropic"
        else "https://example.test/v1",
        api_key_env_var="TEST_KEY",
        backend=Backend.MISTRAL if style == "mistral" else Backend.GENERIC,
        api_style=style if style != "mistral" else "openai",
    )
    model = ModelConfig(
        name="test-model",
        alias="test-alias",
        provider="test",
        supported_thinking_levels=["off"],
    )
    endpoint = {"openai-responses": "responses", "anthropic": "messages"}.get(
        style, "chat/completions"
    )
    with (
        respx.mock as router,
        patch(
            "chartreux.core.llm.provider_smoke.resolve_api_key_with_origin",
            return_value=CREDENTIAL,
        ),
    ):
        route = router.post(f"https://example.test/v1/{endpoint}").mock(
            return_value=httpx.Response(503, json={"error": "test-token raw-body"})
        )
        result = await probe_provider_smoke(model=model, provider=provider)
    assert route.call_count == 1
    assert result.tool.status == "fail"
    assert result.tool.reason == "request_failed"
    assert "test-token" not in str(result) and "raw-body" not in str(result)
    assert "test-token" not in caplog.text and "raw-body" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", [Backend.GENERIC, Backend.MISTRAL])
async def test_explicit_absent_anonymous_does_not_resolve_or_use_sdk_env(
    backend_name, monkeypatch
):
    monkeypatch.setenv("MISTRAL_API_KEY", "unrelated-process-secret")
    provider = ProviderConfig(
        name="anonymous", api_base="https://example.test/v1", backend=backend_name
    )
    backend = create_backend(provider=provider, resolved_credential=None)
    with (
        respx.mock as router,
        patch(
            f"chartreux.core.llm.backend.{backend_name}.resolve_api_key_with_origin",
            side_effect=AssertionError,
        ),
    ):
        route = router.post("https://example.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_response("openai", 1))
        )
        try:
            await backend.complete(
                model=ModelConfig(
                    name="test-model", alias="test-alias", provider="anonymous"
                ),
                messages=[LLMMessage(role=Role.user, content="hi")],
                temperature=0,
                tools=None,
                max_tokens=32,
                tool_choice=None,
                extra_headers=None,
            )
            assert route.call_count == 1
            assert "authorization" not in route.calls[0].request.headers
            assert "unrelated-process-secret" not in str(route.calls[0].request.headers)
        finally:
            await backend.__aexit__(None, None, None)
