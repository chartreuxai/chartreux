from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_loader import SessionLoader
from tests.agent_loop.test_agent_tool_call import make_todo_tool_call
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_legacy_resume_retains_synthetic_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    original = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["todo"],
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
        ),
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="call", tool_calls=[make_todo_tool_call("reused")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ]),
    )
    _ = [event async for event in original.act("old session")]
    directory = original.session_logger.session_dir
    assert directory is not None
    messages, _ = SessionLoader.load_session(directory)
    messages.append(
        LLMMessage(role=Role.assistant, tool_calls=[make_todo_tool_call("missing")])
    )
    fresh = build_test_agent_loop(
        config=original.config,
        session_id=original.session_id,
        session_dir=directory,
        backend=FakeBackend([[mock_llm_chunk(content="continued")]]),
    )
    fresh.messages.reset_preserving_system(messages)
    fresh._clean_message_history()
    assert any(
        m.role == Role.tool and m.tool_call_id == "missing" for m in fresh.messages
    )
    _ = [event async for event in fresh.act("continue legacy")]
    assert not (directory / "journal.jsonl").exists()
    await fresh.aclose()
    await original.aclose()
