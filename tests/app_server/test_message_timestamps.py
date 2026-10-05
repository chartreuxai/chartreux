from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chartreux.app_server._agent_transcript import read_live_agent_transcript
from chartreux.app_server._projection import project_history
from chartreux.app_server._projector import EventProjector
from chartreux.app_server.models import PublicMessageEntry
from chartreux.core.events import AssistantEvent, ReasoningEvent, UserMessageEvent
from chartreux.core.llm_models import LLMMessage, Role, use_posting_clock
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

POSTED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.mark.asyncio
async def test_live_completed_resumed_and_remounted_messages_agree(
    tmp_path: Path,
) -> None:
    times = iter([POSTED, POSTED + timedelta(seconds=10)])
    agent = build_test_agent_loop(
        backend=FakeBackend([
            mock_llm_chunk(content="", reasoning_content="thinking"),
            mock_llm_chunk(content="answer"),
            mock_llm_chunk(content="!"),
        ]),
        enable_streaming=True,
        cwd=tmp_path,
    )
    projector = EventProjector(agent.session_id, "turn-1")
    with use_posting_clock(lambda: next(times)):
        async for event in agent.act("question"):
            if isinstance(event, (UserMessageEvent, AssistantEvent, ReasoningEvent)):
                projector.project(event)
    projector.finalize()
    live = [
        entry for entry in projector.history if isinstance(entry, PublicMessageEntry)
    ]
    expected = [(entry.id, entry.text, entry.posted_at) for entry in live]
    assert [entry.posted_at for entry in live] == [
        POSTED,
        POSTED + timedelta(seconds=10),
    ]

    def no_clock() -> datetime:
        raise AssertionError("projection must never consult the posting clock")

    with use_posting_clock(no_clock):
        for _ in range(2):
            restored = [
                entry
                for entry in project_history(agent)
                if isinstance(entry, PublicMessageEntry)
            ]
            assert [
                (entry.id, entry.text, entry.posted_at) for entry in restored
            ] == expected
            child = read_live_agent_transcript(list(agent.messages))
            assert child.entries is not None
            text_entries = [
                entry
                for entry in child.entries
                if entry.kind in {"user_text", "assistant_text"}
            ]
            assert [entry.posted_at for entry in text_entries] == [
                entry.posted_at for entry in live
            ]
    assert (
        live[0].model_dump(mode="json", by_alias=True)["postedAt"]
        == "2026-01-02T03:04:05Z"
    )
    await agent.aclose()


def test_later_fragments_and_completion_never_replace_first_timestamp() -> None:
    projector = EventProjector("session", "turn")
    projector.project(AssistantEvent(content="first", message_id="a", posted_at=POSTED))
    projector.project(
        AssistantEvent(
            content=" later", message_id="a", posted_at=POSTED + timedelta(minutes=1)
        )
    )
    projector.finalize()
    entry = projector.history[0]
    assert isinstance(entry, PublicMessageEntry)
    assert entry.posted_at == POSTED
    assert entry.text == "first later"


@pytest.mark.asyncio
async def test_legacy_history_and_events_remain_unstamped() -> None:
    agent = build_test_agent_loop()
    messages = [
        LLMMessage(role=Role.user, content="old", message_id="u"),
        LLMMessage(role=Role.assistant, content="answer", message_id="a"),
    ]
    agent.messages.reset(messages)
    projector = EventProjector("session", "turn")
    projector.project(UserMessageEvent(content="old", message_id="u"))
    projector.project(AssistantEvent(content="answer", message_id="a"))
    for entry in [*project_history(agent), *projector.history]:
        assert isinstance(entry, PublicMessageEntry)
        assert entry.posted_at is None
        assert "postedAt" not in entry.model_dump(mode="json", by_alias=True)
    child = read_live_agent_transcript(messages)
    assert child.entries is not None
    assert all(entry.posted_at is None for entry in child.entries)
    await agent.aclose()
