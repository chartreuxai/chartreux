from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from chartreux.core.config import ModelConfig, ProviderConfig, SessionLoggingConfig
from chartreux.core.llm_models import Backend, FunctionCall, Role, ToolCall
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.tools.base import ToolPermission
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQAAAAA3bvkkAAAACklEQVQI12NoAAAAggCB3UNq9AAAAABJRU5ErkJggg=="
)


@pytest.mark.asyncio
async def test_read_image_session_round_trip_uses_snapshot_and_resumes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(PNG_BYTES)
    call = ToolCall(
        id="image",
        index=0,
        function=FunctionCall(
            name="read_image", arguments=json.dumps({"file_path": str(source)})
        ),
    )
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    config = build_test_vibe_config(
        active_model="vision",
        models=[
            ModelConfig(
                name="vision", provider="mistral", alias="vision", supports_images=True
            )
        ],
        providers=[
            ProviderConfig(
                name="mistral",
                api_base="https://api.mistral.ai/v1",
                api_key_env_var="MISTRAL_API_KEY",
                backend=Backend.MISTRAL,
            )
        ],
        session_logging=logging,
        enabled_tools=["read_image"],
        tools={"read_image": {"permission": ToolPermission.ALWAYS.value}},
    )
    agent = build_test_agent_loop(
        config=config,
        backend=FakeBackend([
            [mock_llm_chunk(content="reading", tool_calls=[call])],
            [mock_llm_chunk(content="done")],
        ]),
        cwd=tmp_path,
    )
    async for _ in agent.act("inspect"):
        pass
    session_dir = agent.session_logger.session_dir
    assert session_dir is not None
    image_message = next(
        message for message in agent.messages if message.role is Role.tool
    )
    assert image_message.images is not None
    snapshot = image_message.images[0].source.path  # type: ignore[union-attr]
    assert snapshot.read_bytes() == PNG_BYTES

    source.write_bytes(b"changed")
    source.unlink()
    await agent.aclose()

    loaded, _ = SessionLoader.load_session(session_dir)
    loaded_image = next(message for message in loaded if message.role is Role.tool)
    assert loaded_image.images is not None
    assert loaded_image.images[0].source.path.read_bytes() == PNG_BYTES  # type: ignore[union-attr]
    # Fallback payload turns are request-local projections, never durable history.
    assert not any(
        message.role is Role.user
        and isinstance(message.content, str)
        and message.content.startswith("Image from tool call")
        for message in loaded
    )

    resumed = build_test_agent_loop(config=config, backend=FakeBackend(), cwd=tmp_path)
    resumed.session_id = agent.session_id
    resumed.session_logger.resume_existing_session(agent.session_id, session_dir)
    system_messages = [
        message for message in resumed.messages if message.role is Role.system
    ]
    resumed.messages.reset([*system_messages, *loaded])
    try:
        restored = next(
            message for message in resumed.messages if message.role is Role.tool
        )
        assert restored.images is not None
        assert restored.images[0].source.path.read_bytes() == PNG_BYTES  # type: ignore[union-attr]
    finally:
        await resumed.aclose()
