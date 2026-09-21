from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from chartreux.core.agent_loop_hooks import HookTextReplacement
from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.llm.backend._tool_images import project_tool_images
from chartreux.core.llm.backend.anthropic import AnthropicAdapter
from chartreux.core.llm.backend.generic import OpenAIAdapter
from chartreux.core.llm.backend.mistral import MistralMapper
from chartreux.core.llm.backend.openai_responses import OpenAIResponsesAdapter
from chartreux.core.llm_models import (
    Backend,
    FunctionCall,
    InlineImageSource,
    Role,
    ToolCall,
)
from chartreux.core.model_catalog.availability import (
    AllDeploymentsUnavailableError,
    ExclusionReason,
)
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.tools.base import ToolPermission
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    set_agent_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQAAAAA3bvkkAAAACklEQVQI12NoAAAAggCB3UNq9AAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


def _call(call_id: str, name: str, arguments: dict[str, str], index: int) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(name=name, arguments=json.dumps(arguments)),
    )


def _config(*, two_models: bool = False):
    models = [
        ModelConfig(
            name="vision", provider="mistral", alias="vision", supports_images=True
        )
    ]
    if two_models:
        models.append(
            ModelConfig(
                name="text", provider="mistral", alias="text", supports_images=False
            )
        )
    return build_test_vibe_config(
        active_model="vision",
        models=models,
        providers=[
            ProviderConfig(
                name="mistral",
                api_base="https://api.mistral.ai/v1",
                api_key_env_var="MISTRAL_API_KEY",
                backend=Backend.MISTRAL,
            )
        ],
        enabled_tools=["read_image", "read_file"],
        tools={
            "read_image": {"permission": ToolPermission.ALWAYS.value},
            "read_file": {"permission": ToolPermission.ALWAYS.value},
        },
    )


async def _run_mixed_batch(tmp_path: Path):
    image = tmp_path / "source.png"
    image.write_bytes(PNG_BYTES)
    text = tmp_path / "note.txt"
    text.write_text("plain result\n", encoding="utf-8")
    calls = [
        _call("image", "read_image", {"file_path": str(image)}, 1),
        _call("text", "read_file", {"file_path": str(text)}, 2),
        _call("failed", "read_image", {"file_path": str(tmp_path / "missing.png")}, 0),
    ]
    backend = FakeBackend([
        [mock_llm_chunk(content="tools", tool_calls=calls)],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(config=_config(), backend=backend, cwd=tmp_path)
    async for _ in agent.act("inspect these"):
        pass
    return agent


def _payload(adapter, messages, *, model: str = "test-model"):
    request = adapter.prepare_request(
        model_name=model,
        messages=messages,
        temperature=0.0,
        tools=None,
        max_tokens=None,
        tool_choice=None,
        enable_streaming=False,
        provider=ProviderConfig(name="test", api_base="https://example.test/v1"),
        api_key="test",
    )
    return json.loads(request.body)


@pytest.mark.asyncio
async def test_real_read_image_survives_mixed_parallel_results_through_all_backends(
    tmp_path: Path,
) -> None:
    agent = await _run_mixed_batch(tmp_path)
    tool_messages = [message for message in agent.messages if message.role is Role.tool]
    by_id = {message.tool_call_id: message for message in tool_messages}
    assert set(by_id) == {"image", "text", "failed"}
    assert [image.alias for image in by_id["image"].images or []] == ["source.png"]
    assert by_id["text"].images is None
    assert by_id["failed"].images is None

    generic = _payload(OpenAIAdapter(), agent.messages)
    synthetic = next(
        message
        for message in generic["messages"]
        if message["role"] == "user"
        and isinstance(message["content"], list)
        and message["content"][0]["text"] == "Image from tool call image"
    )
    assert (
        synthetic["content"][1]["image_url"]["url"]
        == f"data:image/png;base64,{PNG_B64}"
    )

    mistral_messages = project_tool_images(agent.messages)
    mistral = [
        MistralMapper().prepare_message(message).model_dump()
        for message in mistral_messages
    ]
    mistral_synthetic = next(
        message
        for message in mistral
        if message["role"] == "user"
        and isinstance(message["content"], list)
        and message["content"][0]["text"] == "Image from tool call image"
    )
    assert (
        mistral_synthetic["content"][1]["image_url"]["url"]
        == f"data:image/png;base64,{PNG_B64}"
    )

    anthropic = _payload(AnthropicAdapter(), agent.messages, model="claude-test")
    result = next(
        block
        for message in anthropic["messages"]
        for block in message["content"]
        if block["type"] == "tool_result" and block["tool_use_id"] == "image"
    )
    assert result["content"][1]["source"]["data"] == PNG_B64
    assert not any(
        message["role"] == "user" and len(message["content"]) == 2
        for message in anthropic["messages"]
    )

    responses = _payload(OpenAIResponsesAdapter(), agent.messages)
    output = next(
        item
        for item in responses["input"]
        if item.get("call_id") == "image" and item["type"] == "function_call_output"
    )
    assert output["output"][1] == {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{PNG_B64}",
    }
    assert not any(
        item.get("role") == "user"
        and isinstance(item.get("content"), list)
        and item["content"][0].get("text") == "Image from tool call image"
        for item in responses["input"]
    )


class _ReplacementHooks:
    def reset_retry_count(self) -> None:
        pass

    async def run(self, _invocation):
        yield HookTextReplacement(text="replaced")


@pytest.mark.asyncio
async def test_read_image_hook_replacement_and_committed_nonvision_switch_fail_closed(
    tmp_path: Path,
) -> None:
    image = tmp_path / "source.png"
    image.write_bytes(PNG_BYTES)
    call = _call("image", "read_image", {"file_path": str(image)}, 0)
    backend = FakeBackend([
        [mock_llm_chunk(content="tools", tool_calls=[call])],
        [mock_llm_chunk(content="done")],
    ])
    config = _config(two_models=True)
    agent = build_test_agent_loop(config=config, backend=backend, cwd=tmp_path)
    agent._hooks_manager = _ReplacementHooks()  # type: ignore[assignment]
    async for _ in agent.act("inspect"):
        pass
    tool_message = next(
        message for message in agent.messages if message.role is Role.tool
    )
    assert tool_message.content == "replaced"
    assert tool_message.images is not None

    committed = CommittedModelIdentity(
        base_model="text",
        provider="mistral/default",
        wire_name="text",
        catalog_revision=config.catalog_snapshot.revision,
    )
    switched = config.model_copy(deep=True)
    switched.attach_committed_model(committed)
    set_agent_config(agent, switched)
    agent.committed_model = committed

    assert "read_image" not in agent.tool_manager.available_tools
    with pytest.raises(AllDeploymentsUnavailableError) as error:
        async for _ in agent.act("continue"):
            pass
    assert {exclusion.reason for exclusion in error.value.exclusions} == {
        ExclusionReason.IMAGES_UNSUPPORTED
    }
    assert any(message.images for message in agent.messages)
    assert isinstance(tool_message.images[0].source, InlineImageSource)
    assert base64.b64decode(tool_message.images[0].source.data) == PNG_BYTES
