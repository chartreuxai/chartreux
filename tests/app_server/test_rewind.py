from __future__ import annotations

import asyncio
from contextlib import aclosing, suppress
from pathlib import Path
from typing import Any

import pytest

from chartreux.app_server.events import HistoryEntryAdded
from chartreux.app_server.models import PublicCheckpointEntry, PublicMessageEntry
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_loader import SessionLoader
from tests.archive_history_support import descriptor_for_prefix, trim_prefix_payloads
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_rewind_inplace_truncates_session(tmp_path: Path) -> None:
    # The ACP/host surface passes inplace=True: it truncates the current session
    # (same id) and persists the truncated history to disk, so reopening the
    # session shows the rewound state rather than the pre-rewind conversation.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="First response")],
        [mock_llm_chunk(content="Second response")],
    ])
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    session = await create_test_app_server_session(agent_loop)

    try:
        _ = [
            event
            async for event in session.act("first message", client_message_id="user-1")
        ]
        _ = [
            event
            async for event in session.act("second message", client_message_id="user-2")
        ]
        original_session_id = session.session_id

        result = await session.resources.sessions.rewind(
            "user-2", restore_files=False, inplace=True
        )

        original_path = SessionLoader.find_session_by_id(
            original_session_id, config.session_logging
        )
        assert original_path is not None
        original_messages, _ = SessionLoader.load_session(original_path)
    finally:
        await session.close()
        await agent_loop.aclose()

    assert result.state.session.id == original_session_id
    assert session.history == result.state.history
    assert [
        entry.text for entry in session.history if isinstance(entry, PublicMessageEntry)
    ] == ["first message", "First response"]
    assert isinstance(session.history[-1], PublicCheckpointEntry)
    assert session.history[-1].kind == "rewind"
    assert all(entry.session_id == result.state.session.id for entry in session.history)
    # The persisted log is truncated in place: the rewound turns are gone.
    assert [
        message.content
        for message in original_messages
        if message.role is not Role.system
    ] == ["first message", "First response"]


@pytest.mark.asyncio
async def test_rewind_defaults_to_forking_and_preserves_original_session(
    tmp_path: Path,
) -> None:
    # The CLI /rewind surface omits inplace, so it must keep forking: a new
    # session is created and the original conversation stays recoverable on
    # disk. Guards against a caller-agnostic handler truncating the CLI too.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="First response")],
        [mock_llm_chunk(content="Second response")],
    ])
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    session = await create_test_app_server_session(agent_loop)

    try:
        _ = [
            event
            async for event in session.act("first message", client_message_id="user-1")
        ]
        _ = [
            event
            async for event in session.act("second message", client_message_id="user-2")
        ]
        original_session_id = session.session_id

        result = await session.resources.sessions.rewind("user-2", restore_files=False)

        original_path = SessionLoader.find_session_by_id(
            original_session_id, config.session_logging
        )
        assert original_path is not None
        original_messages, _ = SessionLoader.load_session(original_path)
    finally:
        await session.close()
        await agent_loop.aclose()

    assert result.state.session.id != original_session_id
    assert [
        message.content
        for message in original_messages
        if message.role is not Role.system
    ] == ["first message", "First response", "second message", "Second response"]


@pytest.mark.asyncio
async def test_fork_rewind_ignores_a_read_answered_after_the_session_changed(
    tmp_path: Path,
) -> None:
    # A resync can still be reading when the fork answer lands, and that read
    # names the session being left behind. Applying it would put the projection
    # back on the source and, because none of its entry ids are in the child,
    # replay the whole abandoned transcript as freshly added entries.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="First response")],
        [mock_llm_chunk(content="Second response")],
    ])
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    session = await create_test_app_server_session(agent_loop)
    forked = asyncio.Event()
    added: list[str] = []

    async def collect_added() -> None:
        async with aclosing(session.events()) as events:
            async for event in events:
                if isinstance(event, HistoryEntryAdded):
                    added.append(event.entry.id)

    try:
        _ = [
            event
            async for event in session.act("first message", client_message_id="user-1")
        ]
        _ = [
            event
            async for event in session.act("second message", client_message_id="user-2")
        ]
        original_session_id = session.session_id
        client = session._connection.current
        assert client is not None
        request = client.request
        reading = asyncio.Event()

        async def delayed_read(
            method: str, params: Any = None, *, wait_for_incoming: bool = False
        ) -> dict[str, Any]:
            if method != "session/read":
                return await request(
                    method, params, wait_for_incoming=wait_for_incoming
                )
            reading.set()
            response = await request(
                method, params, wait_for_incoming=wait_for_incoming
            )
            await forked.wait()
            return response

        client.request = delayed_read  # type: ignore[method-assign]
        listener = asyncio.create_task(collect_added())
        resync = asyncio.create_task(session._resync(client))
        await reading.wait()

        result = await session.resources.sessions.rewind("user-2", restore_files=False)
        forked.set()
        await resync
        listener.cancel()
        with suppress(asyncio.CancelledError):
            await listener
    finally:
        await session.close()
        await agent_loop.aclose()

    assert result.state.session.id != original_session_id
    assert session.session_id == result.state.session.id
    assert added == []
    assert [
        entry.text for entry in session.history if isinstance(entry, PublicMessageEntry)
    ] == ["first message", "First response"]


@pytest.mark.asyncio
async def test_inplace_rewind_ignores_a_read_answered_before_the_truncation(
    tmp_path: Path,
) -> None:
    # An in-place rewind keeps the session id, so a resync already reading when
    # the truncation lands cannot tell from the id that its answer describes a
    # conversation that no longer exists. Neither can it tell from the event
    # watermark: a request moves no watermark, so the read and the rewind
    # answer for the same one. Applying it would restore the rewound turns and
    # replay them as freshly added entries, and rewinding again from there
    # would carry on from a turn the session has already left behind.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="First response")],
        [mock_llm_chunk(content="Second response")],
    ])
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    session = await create_test_app_server_session(agent_loop)
    rewound = asyncio.Event()
    added: list[str] = []

    async def collect_added() -> None:
        async with aclosing(session.events()) as events:
            async for event in events:
                if isinstance(event, HistoryEntryAdded):
                    added.append(event.entry.id)

    try:
        _ = [
            event
            async for event in session.act("first message", client_message_id="user-1")
        ]
        _ = [
            event
            async for event in session.act("second message", client_message_id="user-2")
        ]
        original_session_id = session.session_id
        client = session._connection.current
        assert client is not None
        request = client.request
        reading = asyncio.Event()

        async def delayed_read(
            method: str, params: Any = None, *, wait_for_incoming: bool = False
        ) -> dict[str, Any]:
            if method != "session/read":
                return await request(
                    method, params, wait_for_incoming=wait_for_incoming
                )
            reading.set()
            response = await request(
                method, params, wait_for_incoming=wait_for_incoming
            )
            await rewound.wait()
            return response

        client.request = delayed_read  # type: ignore[method-assign]
        listener = asyncio.create_task(collect_added())
        resync = asyncio.create_task(session._resync(client))
        await reading.wait()

        result = await session.resources.sessions.rewind(
            "user-2", restore_files=False, inplace=True
        )
        rewound.set()
        await resync
        listener.cancel()
        with suppress(asyncio.CancelledError):
            await listener
    finally:
        await session.close()
        await agent_loop.aclose()

    assert result.state.session.id == original_session_id
    assert session.session_id == original_session_id
    assert added == []
    assert [
        entry.text for entry in session.history if isinstance(entry, PublicMessageEntry)
    ] == ["first message", "First response"]


def _sealed_loop(tmp_path: Path) -> Any:
    """An agent loop whose first settled round is sealed (M1 §3).

    [system?, first, first reply, boundary, second, second reply]: the first
    round is the archived prefix and is payload-trimmed in RAM, the
    compaction boundary that authorized the seal and the second round stay
    live.
    """
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    agent_loop = build_test_agent_loop(config=config)
    agent_loop.messages.extend([
        LLMMessage(role=Role.user, content="First", message_id="user-1"),
        LLMMessage(
            role=Role.assistant,
            content="First answer",
            message_id="assistant-1",
            reasoning_payloads=[{"opaque": "payload-1"}],
        ),
        LLMMessage(
            role=Role.user,
            content="Compacted context",
            message_id="boundary-1",
            injected=True,
            context_boundary="compaction",
        ),
        LLMMessage(role=Role.user, content="Second", message_id="user-2"),
        LLMMessage(
            role=Role.assistant, content="Second answer", message_id="assistant-2"
        ),
    ])
    return agent_loop, config


async def _sealed_logged_loop(tmp_path: Path) -> Any:
    agent_loop, config = _sealed_loop(tmp_path)
    logger = agent_loop.session_logger
    await logger.save_interaction(
        agent_loop.messages,
        agent_loop.stats,
        agent_loop.config,
        agent_loop.tool_manager,
    )
    descriptor = descriptor_for_prefix(
        logger.messages_filepath,
        2,
        session_id=agent_loop.session_id,
        message_epoch=agent_loop.messages.message_epoch,
    )
    agent_loop.messages.set_archive_descriptor(descriptor)
    logger.retain_archive_descriptor(descriptor)
    trim_prefix_payloads(agent_loop.messages, 2)
    return agent_loop, config


@pytest.mark.asyncio
async def test_inplace_rewind_to_an_archived_turn_persists_complete_history(
    tmp_path: Path,
) -> None:
    agent_loop, config = await _sealed_logged_loop(tmp_path)
    original_session_id = agent_loop.session_id
    session = await create_test_app_server_session(agent_loop)
    try:
        result = await session.resources.sessions.rewind(
            "user-2", restore_files=False, inplace=True
        )
        hydrated = next(
            message
            for message in agent_loop.messages
            if message.message_id == "assistant-1"
        )
        assert hydrated.reasoning_payloads == [{"opaque": "payload-1"}]
    finally:
        await session.close()
        await agent_loop.aclose()

    assert result.state.session.id == original_session_id
    original_path = SessionLoader.find_session_by_id(
        original_session_id, config.session_logging
    )
    assert original_path is not None
    persisted, _ = SessionLoader.load_session(original_path)
    archived = next(
        message for message in persisted if message.message_id == "assistant-1"
    )
    assert archived.content == "First answer"
    assert archived.reasoning_payloads == [{"opaque": "payload-1"}]


@pytest.mark.asyncio
async def test_branch_fork_materializes_the_archived_prefix_completely(
    tmp_path: Path,
) -> None:
    agent_loop, config = await _sealed_logged_loop(tmp_path)
    session = await create_test_app_server_session(agent_loop)
    try:
        response = await session.resources.sessions.fork("user-2", attach=False)
    finally:
        await session.close()
        await agent_loop.aclose()

    forked_path = SessionLoader.find_session_by_id(
        response.state.session.id, config.session_logging
    )
    assert forked_path is not None
    forked_messages, _ = SessionLoader.load_session(forked_path)
    archived = next(
        message for message in forked_messages if message.message_id == "assistant-1"
    )
    assert archived.content == "First answer"
    assert archived.reasoning_payloads == [{"opaque": "payload-1"}]


def test_app_server_archive_paths_use_the_rewind_package() -> None:
    """One hydration implementation, one archive read interface (P4a1)."""
    import chartreux.app_server._handler as handler
    import chartreux.app_server._runtime as runtime
    from chartreux.core.rewind import manager
    from chartreux.core.session import transcript_archive

    assert handler.archived_history_for is manager.archived_history_for
    assert runtime.archived_history_for is manager.archived_history_for
    assert runtime.complete_history is manager.complete_history
    assert (
        manager.hydrate_archived_messages
        is transcript_archive.hydrate_archived_messages
    )
