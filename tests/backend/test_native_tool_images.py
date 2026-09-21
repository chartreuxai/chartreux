from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from chartreux.core.config import ProviderConfig
from chartreux.core.llm.backend.anthropic import AnthropicAdapter
from chartreux.core.llm.backend.openai_responses import OpenAIResponsesAdapter
from chartreux.core.llm_models import (
    FileImageSource,
    FunctionCall,
    ImageAttachment,
    InlineImageSource,
    LLMMessage,
    Role,
    ToolCall,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")
INLINE_B64 = "aW5saW5l"


def _provider(*, supports_tool_result_images: bool = True) -> ProviderConfig:
    return ProviderConfig(
        name="responses",
        api_base="https://example.com/v1",
        supports_tool_result_images=supports_tool_result_images,
    )


def _file_image(tmp_path: Path) -> ImageAttachment:
    path = tmp_path / "file.png"
    path.write_bytes(PNG_BYTES)
    return ImageAttachment(
        source=FileImageSource(path=path), alias="file.png", mime_type="image/png"
    )


def _inline_image() -> ImageAttachment:
    return ImageAttachment(
        source=InlineImageSource(data=INLINE_B64),
        alias="inline.png",
        mime_type="image/png",
    )


def _call(call_id: str) -> ToolCall:
    return ToolCall(id=call_id, function=FunctionCall(name="tool", arguments="{}"))


def _payload(
    adapter: AnthropicAdapter | OpenAIResponsesAdapter,
    messages: list[LLMMessage],
    *,
    provider: ProviderConfig | None = None,
) -> dict[str, Any]:
    prepared = adapter.prepare_request(
        model_name="claude-test" if isinstance(adapter, AnthropicAdapter) else "gpt-4o",
        messages=messages,
        temperature=0.0,
        tools=None,
        max_tokens=None,
        tool_choice=None,
        enable_streaming=False,
        provider=provider or _provider(),
    )
    return json.loads(prepared.body)


def test_anthropic_tool_result_images_are_native_and_grouped(tmp_path: Path) -> None:
    payload = _payload(
        AnthropicAdapter(),
        [
            LLMMessage(
                role=Role.assistant, tool_calls=[_call("first"), _call("second")]
            ),
            LLMMessage(
                role=Role.tool,
                content="file result",
                tool_call_id="first",
                images=[_file_image(tmp_path), _inline_image()],
            ),
            LLMMessage(
                role=Role.tool,
                content="inline result",
                tool_call_id="second",
                images=[_inline_image()],
            ),
        ],
    )

    results = payload["messages"][1]["content"]
    assert results == [
        {
            "type": "tool_result",
            "tool_use_id": "first",
            "content": [
                {"type": "text", "text": "file result"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": PNG_B64,
                    },
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": INLINE_B64,
                    },
                },
            ],
        },
        {
            "type": "tool_result",
            "tool_use_id": "second",
            "content": [
                {"type": "text", "text": "inline result"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": INLINE_B64,
                    },
                },
            ],
            "cache_control": {"type": "ephemeral"},
        },
    ]


def test_anthropic_text_only_tool_result_is_unchanged() -> None:
    payload = _payload(
        AnthropicAdapter(),
        [LLMMessage(role=Role.tool, content="result", tool_call_id="call")],
    )

    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call",
                    "content": "result",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]


def test_responses_tool_result_images_are_native_without_synthetic_turn(
    tmp_path: Path,
) -> None:
    messages = [
        LLMMessage(
            role=Role.assistant,
            tool_calls=[_call("first"), _call("second")],
            reasoning_payloads=[{"type": "reasoning", "encrypted_content": "secret"}],
        ),
        LLMMessage(
            role=Role.tool,
            content="file result",
            tool_call_id="first",
            images=[_file_image(tmp_path), _inline_image()],
        ),
        LLMMessage(role=Role.tool, content="plain result", tool_call_id="second"),
    ]

    payload = _payload(OpenAIResponsesAdapter(), messages)

    assert payload["input"] == [
        {"type": "reasoning", "encrypted_content": "secret"},
        {
            "type": "function_call",
            "call_id": "first",
            "name": "tool",
            "arguments": "{}",
        },
        {
            "type": "function_call",
            "call_id": "second",
            "name": "tool",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "first",
            "output": [
                {"type": "input_text", "text": "file result"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{PNG_B64}",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{INLINE_B64}",
                },
            ],
        },
        {"type": "function_call_output", "call_id": "second", "output": "plain result"},
    ]
    assert [message.model_dump() for message in messages if message.role == Role.tool][
        0
    ]["images"]


def test_responses_text_only_tool_result_is_unchanged() -> None:
    payload = _payload(
        OpenAIResponsesAdapter(),
        [LLMMessage(role=Role.tool, content="result", tool_call_id="call")],
    )

    assert payload["input"] == [
        {"type": "function_call_output", "call_id": "call", "output": "result"}
    ]


def test_responses_tool_image_capability_override_projects_fallback() -> None:
    messages = [
        LLMMessage(role=Role.assistant, tool_calls=[_call("a"), _call("b")]),
        LLMMessage(
            role=Role.tool,
            content="image result",
            tool_call_id="b",
            images=[_inline_image()],
        ),
        LLMMessage(role=Role.tool, content="plain result", tool_call_id="a"),
    ]
    provider = _provider(supports_tool_result_images=False)

    payload = _payload(OpenAIResponsesAdapter(), messages, provider=provider)

    assert payload["input"][-3:] == [
        {"type": "function_call_output", "call_id": "b", "output": "image result"},
        {"type": "function_call_output", "call_id": "a", "output": "plain result"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Image from tool call b"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{INLINE_B64}",
                },
            ],
        },
    ]
    assert provider.model_dump()["supports_tool_result_images"] is False
    assert (
        ProviderConfig.model_validate(provider.model_dump()).supports_tool_result_images
        is False
    )
    assert messages[1].images is not None
