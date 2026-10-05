from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import reduce
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from mistralai.client.models import UsageInfo
import pytest

from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.llm.backend.anthropic import AnthropicAdapter
from chartreux.core.llm.backend.generic import OpenAIAdapter
from chartreux.core.llm.backend.mistral import MistralBackend, _parse_usage
from chartreux.core.llm.backend.openai_responses import OpenAIResponsesAdapter
from chartreux.core.llm_models import LLMChunk, LLMMessage, LLMUsage, Role

PROVIDER = ProviderConfig(name="test", api_base="https://example.test/v1")


def presence(usage: LLMUsage | None) -> tuple[bool, bool, bool]:
    if usage is None:
        return False, False, False
    return (
        usage.prompt_tokens_reported,
        usage.completion_tokens_reported,
        usage.cached_tokens_reported,
    )


def combine(chunks: list[LLMChunk]) -> LLMChunk:
    return reduce(lambda left, right: left + right, chunks)


def wire_usage(family: str, value: int) -> dict[str, Any]:
    if family == "chat":
        return {
            "prompt_tokens": value,
            "completion_tokens": value,
            "prompt_tokens_details": {"cached_tokens": value},
        }
    if family == "responses":
        return {
            "input_tokens": value,
            "output_tokens": value,
            "input_tokens_details": {"cached_tokens": value},
        }
    return {
        "input_tokens": value,
        "output_tokens": value,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": value,
    }


def adapter_for(family: str) -> Any:
    return {
        "chat": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "responses": OpenAIResponsesAdapter,
    }[family]()


@pytest.mark.parametrize("family", ["chat", "anthropic", "responses"])
@pytest.mark.parametrize("usage_kind", ["absent", "null", "empty", "zero", "partial"])
def test_non_streaming_presence(family: str, usage_kind: str) -> None:
    data: dict[str, Any] = {"output": []} if family == "responses" else {}
    expected = (False, False, False)
    if usage_kind == "null":
        data["usage"] = None
    elif usage_kind == "empty":
        data["usage"] = {}
    elif usage_kind == "zero":
        data["usage"] = wire_usage(family, 0)
        expected = (True, True, True)
    elif usage_kind == "partial":
        key = "prompt_tokens" if family == "chat" else "input_tokens"
        data["usage"] = {key: 0}
        expected = (True, False, False)
    chunk = adapter_for(family).parse_response(data, PROVIDER)
    assert presence(chunk.usage) == expected
    assert chunk.usage is not None
    assert (
        chunk.usage.prompt_tokens,
        chunk.usage.completion_tokens,
        chunk.usage.cached_tokens,
    ) == (0, 0, 0)


@pytest.mark.parametrize("family", ["chat", "anthropic", "responses", "mistral"])
@pytest.mark.parametrize("component", ["prompt", "completion", "cached"])
def test_components_are_independently_reported(family: str, component: str) -> None:
    keys = {
        "chat": {
            "prompt": "prompt_tokens",
            "completion": "completion_tokens",
            "cached": "prompt_tokens_details",
        },
        "responses": {
            "prompt": "input_tokens",
            "completion": "output_tokens",
            "cached": "input_tokens_details",
        },
        "anthropic": {
            "prompt": "input_tokens",
            "completion": "output_tokens",
            "cached": "cache_read_input_tokens",
        },
    }
    usage_data = {
        keys["chat" if family == "mistral" else family][component]: {"cached_tokens": 0}
        if component == "cached" and family != "anthropic"
        else 0
    }
    if family == "mistral":
        usage = _parse_usage(UsageInfo.model_validate(usage_data))
    else:
        data = {"usage": usage_data, "output": []}
        usage = adapter_for(family).parse_response(data, PROVIDER).usage
    assert presence(usage) == (
        component == "prompt",
        component == "completion",
        component == "cached",
    )
    assert presence((usage or LLMUsage()) + LLMUsage()) == presence(usage)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "response.reasoning_summary_text.delta", "delta": "thinking"},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "name": "tool"},
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "{}",
        },
        {
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "arguments": "{}",
        },
        {"type": "response.unknown"},
    ],
)
def test_responses_internal_chunks_do_not_report_usage(event: dict[str, Any]) -> None:
    chunk = OpenAIResponsesAdapter().parse_response(event, PROVIDER)
    assert presence(chunk.usage) == (False, False, False)


def test_anthropic_start_does_not_invent_completion_usage() -> None:
    chunk = AnthropicAdapter().parse_response({
        "type": "message_start",
        "message": {"usage": {"input_tokens": 0}},
    })
    assert presence(chunk.usage) == (True, False, False)


def streaming_events(family: str, value: int | None) -> list[dict[str, Any]]:
    if family == "chat":
        events = [
            {"choices": [{"delta": {"content": "hello"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": []},
        ]
        if value is not None:
            events[-1]["usage"] = wire_usage(family, value)
        return events
    if family == "responses":
        events = [
            {"type": "response.created"},
            {"type": "response.output_text.delta", "delta": "hello"},
            {"type": "response.output_item.done", "item": {}},
            {"type": "response.completed", "response": {"output": []}},
        ]
        if value is not None:
            events[-1]["response"]["usage"] = wire_usage(family, value)
        return events
    # Anthropic reports input at message_start and output at message_delta.
    # Here only the last usage-bearing event arrives, so input/cache stay absent.
    events = [
        {"type": "message_start", "message": {}},
        {"type": "ping"},
        {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hello"},
        },
        {"type": "content_block_stop"},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    if value is not None:
        events[-2]["usage"] = {"output_tokens": value}
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["chat", "anthropic", "responses"])
@pytest.mark.parametrize("value", [None, 0, 7])
async def test_streaming_presence(family: str, value: int | None) -> None:
    async def raw() -> AsyncGenerator[dict[str, Any]]:
        for event in streaming_events(family, value):
            yield event

    chunks = [
        parsed.chunk
        async for parsed in adapter_for(family).parse_stream(raw(), PROVIDER)
    ]
    for chunk in chunks[: -2 if family == "anthropic" else -1]:
        assert presence(chunk.usage) == (False, False, False)
    total = combine(chunks)
    expected = (
        (False, False, False)
        if value is None
        else ((False, True, False) if family == "anthropic" else (True, True, True))
    )
    assert presence(total.usage) == expected
    assert total.usage is None or total.usage.completion_tokens == (value or 0)
    if family == "anthropic":
        assert presence(chunks[-1].usage) == (False, False, False)


@pytest.mark.parametrize("value", [0, 9])
def test_anthropic_split_usage(value: int) -> None:
    adapter = AnthropicAdapter()
    start = adapter.parse_response({
        "type": "message_start",
        "message": {"usage": wire_usage("anthropic", value)},
    })
    end = adapter.parse_response({
        "type": "message_delta",
        "usage": {"output_tokens": value},
    })
    assert presence(start.usage) == (True, True, True)
    assert presence(end.usage) == (False, True, False)
    total = start + end
    assert presence(total.usage) == (True, True, True)
    assert total.usage is not None
    assert (
        total.usage.prompt_tokens,
        total.usage.completion_tokens,
        total.usage.cached_tokens,
    ) == (2 * value, value, value)


def test_usage_addition_and_round_trip() -> None:
    missing = LLMUsage()
    reported_zero = LLMUsage(prompt_tokens=0)
    assert presence(missing + missing) == (False, False, False)
    assert presence(missing + reported_zero) == (True, False, False)
    assert presence(reported_zero + missing) == (True, False, False)
    assert presence(LLMUsage.model_validate(missing.model_dump())) == (
        False,
        False,
        False,
    )
    assert presence(LLMUsage.model_validate(reported_zero.model_dump())) == (
        True,
        False,
        False,
    )
    chunk = LLMChunk(message=LLMMessage(role=Role.assistant), usage=reported_zero)
    bookkeeping = LLMChunk(message=LLMMessage(role=Role.assistant))
    assert presence((chunk + bookkeeping).usage) == (True, False, False)
    assert presence((bookkeeping + chunk).usage) == (True, False, False)


class SDKStream:
    def __init__(self, usages: list[UsageInfo | None]) -> None:
        self.usages = usages
        self.response = SimpleNamespace(headers={})

    async def __aenter__(self) -> SDKStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def __aiter__(self) -> AsyncGenerator[Any]:
        for usage in self.usages:
            yield SimpleNamespace(data=SimpleNamespace(choices=[], usage=usage))


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "usage_kind", ["absent", "empty", "zero", "partial", "nonzero"]
)
async def test_mistral_usage_paths(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, usage_kind: str
) -> None:
    usage = None
    expected = (False, False, False)
    value = 0
    if usage_kind == "empty":
        usage = UsageInfo()
    elif usage_kind in {"zero", "nonzero"}:
        value = 7 if usage_kind == "nonzero" else 0
        usage = UsageInfo.model_validate(wire_usage("chat", value))
        expected = (True, True, True)
    elif usage_kind == "partial":
        usage = UsageInfo(prompt_tokens=0)
        expected = (True, False, False)
    backend = MistralBackend(provider=PROVIDER, resolved_credential=None)
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hello", tool_calls=None),
                finish_reason=None,
            )
        ],
        usage=usage,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            complete_async=AsyncMock(return_value=response),
            stream_async=AsyncMock(
                return_value=SDKStream([None, UsageInfo(), usage, None])
            ),
        )
    )
    monkeypatch.setattr(backend, "_get_client", lambda: client)
    kwargs: dict[str, Any] = dict(
        model=ModelConfig(name="mistral-small-latest", provider="test", alias="test"),
        messages=[],
        temperature=0.2,
        tools=None,
        max_tokens=None,
        tool_choice=None,
        extra_headers=None,
    )
    if streaming:
        chunks = [chunk async for chunk in backend.complete_streaming(**kwargs)]
        assert presence(chunks[0].usage) == (False, False, False)
        assert presence(chunks[1].usage) == (False, False, False)
        assert presence(chunks[-1].usage) == (False, False, False)
        chunk = combine(chunks)
    else:
        chunk = await backend.complete(**kwargs)
    assert presence(chunk.usage) == expected
    assert chunk.usage is not None
    assert (
        chunk.usage.prompt_tokens,
        chunk.usage.completion_tokens,
        chunk.usage.cached_tokens,
    ) == (value, value, value)


@pytest.mark.parametrize("value", [None, "invalid"])
def test_mistral_invalid_cached_count_is_not_reported(value: Any) -> None:
    usage = UsageInfo.model_validate({
        "prompt_tokens_details": {"cached_tokens": value}
    })
    assert presence(_parse_usage(usage)) == (False, False, False)
