from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend


@pytest.fixture
def transcript_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentLoop:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
        ),
        backend=FakeBackend(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["reset", "rebind", "rewind"])
async def test_real_transition_discards_queued_save(
    transcript_loop: AgentLoop, transition: str
) -> None:
    loop = transcript_loop
    loop.messages.append(LLMMessage(role=Role.user, content="retained"))
    loop.messages.append(LLMMessage(role=Role.user, content="removed"))
    await loop._save_messages()
    logger = loop.session_logger
    old_path = logger.messages_filepath
    before = old_path.read_bytes()
    epoch = logger._transcript_cursor_generation
    await logger._save_lock.acquire()
    stale = asyncio.create_task(loop._save_messages())
    await asyncio.sleep(0)
    rewind = None
    if transition == "reset":
        await loop._reset_session()
    elif transition == "rebind":
        assert logger.session_dir is not None
        assert logger.session_metadata is not None
        loop.rebind_to_session(
            loop.session_id,
            logger.session_dir,
            [loop.messages[-2]],
            session_metadata=logger.session_metadata,
            prepared_scratchpad=None,
        )
    else:
        rewind = asyncio.create_task(
            loop.rewind_manager.rewind_to_message(
                len(loop.messages) - 1, restore_files=False, inplace=True
            )
        )
        await asyncio.sleep(0)
        assert logger._transcript_cursor_generation > epoch
    logger._save_lock.release()
    await stale
    if transition == "rewind":
        assert rewind is not None
        await rewind
        assert "removed" not in old_path.read_text()
    else:
        assert old_path.read_bytes() == before
    assert logger._transcript_cursor_generation > epoch
    await loop.aclose()
