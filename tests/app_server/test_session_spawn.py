"""Session spawn, turn streaming, close, initialize handshake, transport wire, clear-history, and root-replacement tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from chartreux.app_server._model import validate_wire
from chartreux.app_server._projection import project_history
from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server.client import AppServerClient, AppServerConnectionClosed
from chartreux.app_server.events import (
    HistoryEntryAdded,
    HistoryEntryUpdated,
    SessionSnapshot,
    StatsUpdated,
    TurnCompleted,
    TurnStarted,
)
from chartreux.app_server.models import (
    CompletedEffectState,
    FileReadEffectDetail,
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicMessageEntry,
    PublicNoticeEntry,
    ResourceContentBlock,
    ScheduledLoopFiredNoticeDetail,
    TurnErrorCode,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ClientInfo,
    ConfigWriteOpWire,
    ConfigWriteParams,
    Notification,
    ProtocolErrorCode,
    RuntimeSnapshot,
    RuntimeUpdatedParams,
    ServerRequest,
    SessionOptions,
    SessionReadParams,
    SessionReadResponse,
    SessionReadyWaitParams,
    SessionReadyWaitResponse,
    SessionStartParams,
    SessionUpdatedParams,
)
from chartreux.app_server.session import AppServerTurnError, _PublishedEvent
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.compaction import CompactionFailedError
from chartreux.core.config import ModelConfig, SessionLoggingConfig
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.events import (
    AssistantEvent,
    CompactEndEvent,
    CompactStartEvent,
    UserMessageEvent,
)
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.session_types import ScheduledLoop
from chartreux.core.tools.models import ToolPermission
from chartreux.user_content import UserResourceLink
from tests.app_server._session_runtime_support import (
    _consume,
    _create_reconnectable_session,
    _isolate_recovery_registry,  # noqa: F401 - autouse fixture registered via module namespace
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    create_test_app_server_session,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend, FakeInterruptedStreamingBackend


def _wire_read_request(session_id: str) -> dict:
    return SessionReadParams(session_id=session_id).model_dump(
        mode="json", by_alias=True
    )


@pytest.mark.asyncio
async def test_injected_turn_hides_user_after_stream_failure() -> None:
    backend = FakeInterruptedStreamingBackend([
        [mock_llm_chunk(content="partial")],
        [mock_llm_chunk(content="recovered")],
    ])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    session = await create_test_app_server_session(agent_loop)

    try:
        with pytest.raises(AppServerTurnError) as exc_info:
            await _consume(session.act("hi", client_message_id="u1"))

        assert agent_loop.messages[-1].role is Role.assistant
        assert agent_loop.messages[-1].content == "partial"
        await _consume(session.act("hidden continuation", injected=True))
        restored_history = project_history(agent_loop)
    finally:
        await session.close()
        await agent_loop.aclose()

    assert exc_info.value.error.code == TurnErrorCode.BACKEND_ERROR
    assert backend.streaming_attempts == 2
    assistant_messages = [
        entry
        for entry in session.history
        if isinstance(entry, PublicMessageEntry) and entry.role == "assistant"
    ]
    assert [entry.text for entry in assistant_messages] == ["partial", "recovered"]
    user_entries = [
        entry
        for entry in session.history
        if isinstance(entry, PublicMessageEntry) and entry.role == "user"
    ]
    assert len(user_entries) == 1
    restored_user_entries = [
        entry
        for entry in restored_history
        if isinstance(entry, PublicMessageEntry) and entry.role == "user"
    ]
    restored_assistant_entries = [
        entry
        for entry in restored_history
        if isinstance(entry, PublicMessageEntry) and entry.role == "assistant"
    ]
    assert len(restored_user_entries) == 1
    assert [entry.text for entry in restored_assistant_entries] == [
        "partial",
        "recovered",
    ]
    injected_message = backend.requests_messages[-1][-1]
    assert injected_message.role is Role.user
    assert injected_message.injected is True
    assert injected_message.content == "hidden continuation"


@pytest.mark.asyncio
async def test_explicit_compaction_failure_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)

    async def fail_compact(_instructions: str = "") -> str:
        raise CompactionFailedError("tool_call")

    monkeypatch.setattr(agent_loop, "compact", fail_compact)
    try:
        with pytest.raises(AppServerResponseError) as exc_info:
            await session.compact()
    finally:
        await session.close()
        await agent_loop.aclose()

    assert exc_info.value.error.code is ProtocolErrorCode.COMPACTION_FAILED
    assert exc_info.value.error.data == {"reason": "tool_call"}


@pytest.mark.asyncio
async def test_slow_unsolicited_event_consumer_applies_bounded_backpressure() -> None:
    session = await create_test_app_server_session(build_test_agent_loop())
    queue = session._unsolicited_events
    event = SessionSnapshot(session.state)
    try:
        for _ in range(queue.maxsize):
            queue.put_nowait(_PublishedEvent(session._event_generation, event))

        blocked = asyncio.create_task(session._publish_event(event))
        await asyncio.sleep(0)
        assert not blocked.done()

        published = queue.get_nowait()
        assert isinstance(published, _PublishedEvent)
        assert published.event is event
        await asyncio.wait_for(blocked, timeout=1)
        assert queue.qsize() == queue.maxsize
    finally:
        while not queue.empty():
            queue.get_nowait()
        await session.close()


@pytest.mark.asyncio
async def test_session_close_does_not_wait_for_full_event_queues() -> None:
    session = await create_test_app_server_session(build_test_agent_loop())
    event = SessionSnapshot(session.state)
    for queue in (session._events, session._unsolicited_events):
        for _ in range(queue.maxsize):
            queue.put_nowait(_PublishedEvent(session._event_generation, event))

    await asyncio.wait_for(session.close(), timeout=1)

    assert session._events.qsize() == 1
    assert session._unsolicited_events.qsize() == 1


@pytest.mark.asyncio
async def test_session_close_does_not_reconnect_after_stop() -> None:
    """*Prepare*: A reconnectable in-process app-server session.
    *Do*: Close the session through its public client facade.
    *Assert*: Intentional shutdown does not open a replacement connection.
    """
    # Prepare
    session = await _create_reconnectable_session(build_test_agent_loop())
    reconnect = Mock(wraps=session._connection._client_factory)
    session._connection._client_factory = reconnect

    # Do
    await asyncio.wait_for(session.close(), timeout=1)

    # Assert
    reconnect.assert_not_called()


@pytest.mark.asyncio
async def test_turn_pushes_fresh_context_while_tool_runs() -> None:
    tool_call = ToolCall(
        id="todo-1",
        index=0,
        function=FunctionCall(name="todo", arguments='{"action":"read"}'),
    )
    backend = FakeBackend([
        [
            mock_llm_chunk(content="", reasoning_content="thinking"),
            mock_llm_chunk(content="Let me check", tool_calls=[tool_call]),
        ],
        [mock_llm_chunk(content="Done")],
    ])
    config = build_test_vibe_config(
        enabled_tools=["todo"],
        tools={"todo": {"permission": ToolPermission.ALWAYS.value}},
    )
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    session = await create_test_app_server_session(agent_loop)

    try:
        events = [event async for event in session.act("use a tool")]
    finally:
        await session.close()
        await agent_loop.aclose()

    tool_done_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, HistoryEntryUpdated)
        and isinstance(event.entry, PublicEffectEntry)
        and isinstance(event.entry.state, CompletedEffectState)
    )
    assert any(
        isinstance(event, StatsUpdated) and event.params.stats.context_tokens > 0
        for event in events[:tool_done_index]
    )
    zero_updates = sum(
        isinstance(event, StatsUpdated) and event.params.stats.context_tokens == 0
        for event in events
    )
    assert zero_updates == 1


@pytest.mark.asyncio
async def test_turn_preserves_resource_blocks_and_normalizes_model_input() -> None:
    backend = FakeBackend([mock_llm_chunk(content="done")])
    agent_loop = build_test_agent_loop(backend=backend)
    session = await create_test_app_server_session(agent_loop)
    resource = UserResourceLink(
        uri="file:///workspace/spec.md",
        media_type="text/markdown",
        title="Specification",
    )

    try:
        await _consume(session.act("Review this", resources=[resource]))
    finally:
        await session.close()
        await agent_loop.aclose()

    request_user = next(
        message for message in backend.requests_messages[0] if message.role is Role.user
    )
    assert request_user.content is not None
    assert "Review this" in request_user.content
    assert "uri: file:///workspace/spec.md" in request_user.content
    public_user = next(
        entry
        for entry in session.history
        if isinstance(entry, PublicMessageEntry) and entry.role == "user"
    )
    assert public_user.text == "Review this"
    assert any(
        isinstance(block, ResourceContentBlock)
        and block.resource.uri == "file:///workspace/spec.md"
        for block in public_user.content
    )


@pytest.mark.asyncio
async def test_prepared_image_crosses_turn_start_boundary() -> None:
    config = build_test_vibe_config(
        active_model="image-model",
        models=[
            ModelConfig(
                name="image-model",
                provider="mistral",
                alias="image-model",
                supports_images=True,
            )
        ],
    )
    backend = FakeBackend([mock_llm_chunk(content="done")])
    agent_loop = build_test_agent_loop(config=config, backend=backend)
    session = await create_test_app_server_session(agent_loop)
    Path("shot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

    try:
        prepared = await session.resources.workspace.prepare_prompt("look @shot.png")
        await _consume(
            session.act(prepared.prompt_text, images=prepared.images or None)
        )
    finally:
        await session.close()
        await agent_loop.aclose()

    assert prepared.images[0].alias == "shot.png"
    request_user = next(
        message for message in backend.requests_messages[0] if message.role is Role.user
    )
    assert request_user.images is not None
    assert request_user.images[0].alias == "shot.png"


@pytest.mark.asyncio
async def test_at_file_mention_projects_without_presentation_snapshot() -> None:
    Path("notes.md").write_text("hello world", encoding="utf-8")
    backend = FakeBackend([mock_llm_chunk(content="done")])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    session = await create_test_app_server_session(agent_loop)

    try:
        await _consume(session.act("read @notes.md what does this do"))
    finally:
        await session.close()
        await agent_loop.aclose()

    read_effect = next(
        entry
        for entry in session.history
        if isinstance(entry, PublicEffectEntry)
        and isinstance(entry.detail, FileReadEffectDetail)
    )
    assert isinstance(read_effect.state, CompletedEffectState)

    injected_call = next(
        tool_call
        for message in agent_loop.messages
        for tool_call in message.tool_calls or []
        if tool_call.function.name == "read_file"
    )
    assert injected_call.presentation is not None


@pytest.mark.asyncio
async def test_completed_turn_refreshes_runtime_projection() -> None:
    backend = FakeBackend([mock_llm_chunk(content="done")])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    session = await create_test_app_server_session(agent_loop)
    await agent_loop.config_orchestrator.set_field(
        "/theme", "server-updated-theme", target_layer=OverridesLayer.NAME
    )

    try:
        _ = [event async for event in session.act("refresh runtime")]
    finally:
        await session.close()
        await agent_loop.aclose()

    assert session.resources.config.current.theme == "server-updated-theme"


@pytest.mark.asyncio
async def test_stale_runtime_update_after_resume_does_not_close_connection() -> None:
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)
    runtime = session.resources.runtime
    current_context_window = runtime.context_window
    assert current_context_window is not None
    stale_update = RuntimeUpdatedParams(
        session_id=f"stale-{session.session_id}",
        runtime=RuntimeSnapshot(
            config=session.resources.config.current,
            skills=runtime.skills,
            tools=runtime.tools,
            stats=runtime.stats,
            context_window=current_context_window + 1,
            issues=runtime.issues,
            hooks_count=runtime.hooks_count,
            mcp=runtime.mcp,
        ),
    )

    try:
        await session.connect()
        client = session._connection.current
        assert client is not None
        await session._handle_notification(
            client,
            Notification(
                method="runtime/updated",
                params=stale_update.model_dump(mode="json", by_alias=True),
            ),
        )
        assert runtime.context_window == current_context_window
        await runtime.refresh()
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_persisted_session_resume_appends_checkpoint(tmp_path: Path) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    session_id = saved.session_id
    await saved.aclose()

    session = await attach_test_app_server_session(
        start_test_app_server(build_test_agent_loop(config=config)),
        resume_session_id=session_id,
    )
    try:
        checkpoints = [
            entry
            for entry in session.history
            if isinstance(entry, PublicCheckpointEntry) and entry.kind == "resume"
        ]
    finally:
        await session.close()

    assert len(checkpoints) == 1
    assert checkpoints[0].message == (
        "Session resumed. Live and retained agent handles were not restored; "
        "old agent IDs now raise UnknownAgentError."
    )


@pytest.mark.asyncio
async def test_due_loop_runs_as_an_unsolicited_server_turn(tmp_path: Path) -> None:
    backend = FakeBackend([mock_llm_chunk(content="scheduled response")])
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    metadata = agent_loop.session_logger.session_metadata
    assert metadata is not None
    before_fire = time.time()
    metadata.loops = [
        ScheduledLoop(
            id="scheduled-1",
            interval_seconds=30,
            prompt="scheduled prompt",
            next_fire_at=before_fire - 1,
            created_at=before_fire - 31,
        )
    ]
    session = await create_test_app_server_session(agent_loop)

    try:
        events = []
        async with asyncio.timeout(2):
            async for event in session.events():
                events.append(event)
                if isinstance(event, TurnCompleted):
                    break
    finally:
        await session.close()

    assert any(isinstance(event, TurnStarted) for event in events)
    assert any(
        isinstance(event, HistoryEntryAdded)
        and isinstance(event.entry, PublicMessageEntry)
        and event.entry.role == "user"
        and event.entry.text == "scheduled prompt"
        for event in events
    )
    entries = [event.entry for event in events if isinstance(event, HistoryEntryAdded)]
    notice = next(entry for entry in entries if isinstance(entry, PublicNoticeEntry))
    assert isinstance(notice.detail, ScheduledLoopFiredNoticeDetail)
    assert notice.detail.loop_id == "scheduled-1"
    assert notice.message == "Loop `scheduled-1` fired"
    user_index = next(
        index
        for index, entry in enumerate(entries)
        if isinstance(entry, PublicMessageEntry) and entry.role == "user"
    )
    assert entries.index(notice) == user_index + 1
    assert session.state.latest_turn is not None
    assert session.state.latest_turn.status == "completed"
    assert metadata.loops[0].next_fire_at >= before_fire + 30


@pytest.mark.asyncio
async def test_session_history_get_is_reachable_while_root_active() -> None:
    # The history resource reads through the selected backend Host, so a
    # missing session must surface NOT_FOUND while another root is attached.
    session = await create_test_app_server_session(build_test_agent_loop())
    try:
        with pytest.raises(AppServerResponseError) as excinfo:
            await session.resources.sessions.get_session_history("does-not-exist")
    finally:
        await session.close()

    assert excinfo.value.error.code is ProtocolErrorCode.NOT_FOUND


@pytest.mark.asyncio
async def test_session_history_get_returns_projected_entries_and_respects_limit() -> (
    None
):
    # No custom save_dir: the backend Host reads from the default
    # test-isolated session directory.
    config = build_test_vibe_config(session_logging=SessionLoggingConfig(enabled=True))
    saved = build_test_agent_loop(config=config)
    saved.messages.reset([
        LLMMessage(role=Role.system, content="system"),
        LLMMessage(role=Role.user, content="first question"),
        LLMMessage(role=Role.assistant, content="first answer"),
        LLMMessage(role=Role.user, content="second question"),
        LLMMessage(role=Role.assistant, content="second answer"),
    ])
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    session = await create_test_app_server_session(build_test_agent_loop(config=config))
    try:
        full = await session.resources.sessions.get_session_history(saved_session_id)
        limited = await session.resources.sessions.get_session_history(
            saved_session_id, history_limit=1
        )
    finally:
        await session.close()

    roles = [e.role for e in full if isinstance(e, PublicMessageEntry)]
    assert Role.user in roles
    assert Role.assistant in roles
    dumped = " ".join(e.model_dump_json() for e in full)
    assert "first question" in dumped
    assert "second question" in dumped
    assert "second answer" in dumped
    assert 0 < len(limited) < len(full)


@pytest.mark.asyncio
async def test_headless_session_does_not_run_persisted_loops(tmp_path: Path) -> None:
    backend = FakeBackend([mock_llm_chunk(content="direct response")])
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    metadata = agent_loop.session_logger.session_metadata
    assert metadata is not None
    now = time.time()
    metadata.loops = [
        ScheduledLoop(
            id="scheduled-1",
            interval_seconds=30,
            prompt="scheduled prompt",
            next_fire_at=now - 1,
            created_at=now - 31,
        )
    ]
    session = await attach_test_app_server_session(
        start_test_app_server(agent_loop), session_options=SessionOptions(headless=True)
    )

    try:
        await asyncio.sleep(0.1)
        assert session.state.latest_turn is None
        await _consume(session.act("direct prompt"))
    finally:
        await session.close()

    assert all(
        not isinstance(entry, PublicMessageEntry) or entry.text != "scheduled prompt"
        for entry in session.history
    )


@pytest.mark.asyncio
async def test_attach_sends_snapshot_response_before_buffered_live_events() -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    try:
        server._begin_attachment()
        await server._notify(
            "session/updated",
            SessionUpdatedParams(
                event_id=0, session_id="session-1", patch=[], emitted_at=1
            ),
        )
        await server._send({"jsonrpc": "2.0", "id": "attach", "result": {}})
        await server._finish_attachment(True)

        messages = client_transport.messages()
        response = await anext(messages)
        notification = await anext(messages)

        assert response == {"jsonrpc": "2.0", "id": "attach", "result": {}}
        assert notification["method"] == "session/updated"
        assert notification["params"]["eventId"] == 1
    finally:
        await client_transport.close()
        await server_transport.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_runtime_mutation_result_publishes_the_coherent_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    apply_patch = AsyncMock(wraps=agent_loop.config_orchestrator.apply_session_patch)
    monkeypatch.setattr(
        agent_loop.config_orchestrator, "apply_session_patch", apply_patch
    )

    try:
        await client.initialize(ClientInfo(name="runtime-update-test", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())
        await client.request(
            "config/write",
            ConfigWriteParams(
                session_id=agent_loop.session_id,
                ops=[
                    ConfigWriteOpWire(
                        op="set", path="/disable_welcome_banner_animation", value=True
                    ),
                    ConfigWriteOpWire(
                        op="set", path="/autocopy_to_clipboard", value=False
                    ),
                ],
            ),
        )

        apply_patch.assert_awaited_once()
        call = apply_patch.await_args
        assert call is not None
        operations = call.args[0]
        assert [operation.path for operation in operations] == [
            "/disable_welcome_banner_animation",
            "/autocopy_to_clipboard",
        ]

        incoming = client.incoming()
        async with asyncio.timeout(1):
            while True:
                message = await anext(incoming)
                if (
                    isinstance(message, Notification)
                    and message.method == "runtime/updated"
                ):
                    break
        await incoming.aclose()

        update = validate_wire(RuntimeUpdatedParams, message.params)
        assert update.session_id == agent_loop.session_id
    finally:
        await client.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_initialize_must_be_first_request() -> None:
    agent_loop = build_test_agent_loop()
    client = start_test_app_server(agent_loop)

    try:
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request("session/read", {"sessionId": agent_loop.session_id})
    finally:
        await client.close()
        await agent_loop.aclose()

    assert exc_info.value.error.code is ProtocolErrorCode.NOT_INITIALIZED


@pytest.mark.asyncio
async def test_initialize_requires_initialized_notification() -> None:
    agent_loop = build_test_agent_loop()
    client = start_test_app_server(agent_loop)

    try:
        response = await client.initialize(
            ClientInfo(name="test", version="1"),
            ClientCapabilities(callback_kinds=["user_input"]),
        )
        assert response.model_dump(mode="json") == {
            "serverInfo": {
                "name": "chartreux-app-server",
                "version": response.server_info.version,
            }
        }

        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "session/read", _wire_read_request(agent_loop.session_id)
            )
        assert exc_info.value.error.code is ProtocolErrorCode.NOT_INITIALIZED

        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())
        public = validate_wire(
            SessionReadResponse,
            await client.request(
                "session/read", _wire_read_request(agent_loop.session_id)
            ),
        )
        assert public.state.session.id == agent_loop.session_id
    finally:
        await client.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_initialize_rejects_unknown_capability_fields() -> None:
    agent_loop = build_test_agent_loop()
    client = start_test_app_server(agent_loop)

    try:
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "initialize",
                {
                    "clientInfo": {"name": "test", "version": "1"},
                    "capabilities": {"futureCapability": True},
                },
            )
    finally:
        await client.close()
        await agent_loop.aclose()

    assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS


@pytest.mark.asyncio
async def test_cancelled_client_request_is_removed_from_pending_requests() -> None:
    client_transport, _ = memory_transport_pair()
    client = AppServerClient(client_transport)
    task = asyncio.create_task(client.request("test/wait"))

    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert client._pending == {}
    await client.close()


@pytest.mark.asyncio
async def test_transport_eof_fails_pending_requests() -> None:
    client_transport, server_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    task = asyncio.create_task(client.request("test/wait"))

    await asyncio.sleep(0)
    await server_transport.close()

    with pytest.raises(AppServerConnectionClosed):
        await task
    assert client._pending == {}
    await client.close()


@pytest.mark.asyncio
async def test_server_messages_preserve_wire_order() -> None:
    client_transport, server_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    await client.start()

    await server_transport.send({
        "jsonrpc": "2.0",
        "method": "history/entryAdded",
        "params": {"entry": {}},
    })
    await server_transport.send({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "callback/call",
        "params": {},
    })

    incoming = client.incoming()
    first = await anext(incoming)
    second = await anext(incoming)

    assert isinstance(first, Notification)
    assert first.method == "history/entryAdded"
    assert isinstance(second, ServerRequest)
    assert second.method == "callback/call"

    await incoming.aclose()
    await client.close()
    await server_transport.close()


@pytest.mark.asyncio
async def test_active_turn_fails_when_server_connection_closes() -> None:
    agent_loop = build_test_agent_loop()
    client = start_test_app_server(agent_loop)
    session = await attach_test_app_server_session(client)
    started = asyncio.Event()

    async def blocking_act(msg: str, **_kwargs):
        yield UserMessageEvent(content=msg, message_id="blocked-user")
        started.set()
        await asyncio.Event().wait()

    agent_loop.act = blocking_act
    turn = asyncio.create_task(_consume(session.act("wait")))
    try:
        await started.wait()
        peer = client._run_peer
        assert peer is not None
        await cast(Any, peer).__self__._transport.close()
        with pytest.raises(AppServerConnectionClosed):
            await asyncio.wait_for(turn, timeout=1)
    finally:
        turn.cancel()
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_connection_shutdown_cancels_blocked_request_without_drain_delay() -> (
    None
):
    agent_loop = build_test_agent_loop()
    client = start_test_app_server(agent_loop)
    session = await attach_test_app_server_session(client)
    entered = asyncio.Event()

    async def wait_until_ready() -> None:
        entered.set()
        await asyncio.Event().wait()

    agent_loop.wait_until_ready = wait_until_ready
    request = asyncio.create_task(
        client.request(
            "session/ready/wait", SessionReadyWaitParams(session_id=session.session_id)
        )
    )
    await entered.wait()

    await asyncio.wait_for(client.close(), timeout=1)
    with pytest.raises(AppServerConnectionClosed):
        await request
    await session.close()


@pytest.mark.asyncio
async def test_ready_wait_response_carries_init_duration_ms() -> None:
    agent_loop = build_test_agent_loop(defer_heavy_init=True)
    client = start_test_app_server(agent_loop)
    session = await attach_test_app_server_session(client)
    try:
        raw = await client.request(
            "session/ready/wait", SessionReadyWaitParams(session_id=session.session_id)
        )
        response = validate_wire(SessionReadyWaitResponse, raw)

        assert response.init_duration_ms is not None
        assert isinstance(response.init_duration_ms, int)
        assert response.init_duration_ms >= 0
        assert response.init_duration_ms == agent_loop.init_duration_ms
    finally:
        await session.close()
        await client.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_clear_history_adopts_replacement_before_next_turn() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="Before")],
        [mock_llm_chunk(content="After")],
    ])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    session = await create_test_app_server_session(agent_loop)

    try:
        await _consume(session.act("first"))
        original_session_id = session.session_id
        await session.clear_history()
        replacement_session_id = session.session_id
        events = await _consume(session.act("second"))
    finally:
        await session.close()

    assert replacement_session_id != original_session_id
    assert any(isinstance(event, HistoryEntryAdded) for event in events)
    assert all(entry.session_id == replacement_session_id for entry in session.history)
    assert any(
        isinstance(entry, PublicCheckpointEntry) and entry.kind == "clear"
        for entry in session.history
    )


@pytest.mark.asyncio
async def test_clear_history_refreshes_the_cached_context_gauge() -> None:
    """*Prepare*: A session that spent context, and a server reporting it cleared.
    *Do*: Clear the history.
    *Assert*: The client's cached stats followed the server, with no turn between.

    Clients render the gauge from this cache, so a clear that left it holding the
    discarded conversation would keep showing a full context until the next turn.
    """
    # Prepare
    backend = FakeBackend([[mock_llm_chunk(content="Before")]])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    session = await create_test_app_server_session(agent_loop)

    # Do
    try:
        agent_loop.stats.context_tokens = 50_000
        await session.resources.refresh()
        filled = session.resources.runtime.stats.context_tokens
        agent_loop.stats.context_tokens = 0
        await session.clear_history()
        cleared = session.resources.runtime.stats.context_tokens
    finally:
        await session.close()

    # Assert
    assert filled == 50_000
    assert cleared == 0


@pytest.mark.asyncio
async def test_compact_refreshes_the_cached_context_gauge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: A session whose compaction empties the context server-side.
    *Do*: Compact.
    *Assert*: The client's cached stats followed, with no turn in between.
    """
    # Prepare
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)

    async def compact(_instructions: str = "") -> str:
        agent_loop.stats.context_tokens = 0
        return "summary"

    monkeypatch.setattr(agent_loop, "compact", compact)

    # Do
    try:
        agent_loop.stats.context_tokens = 50_000
        await session.resources.refresh()
        filled = session.resources.runtime.stats.context_tokens
        await session.compact()
        cleared = session.resources.runtime.stats.context_tokens
    finally:
        await session.close()

    # Assert
    assert filled == 50_000
    assert cleared == 0


@pytest.mark.asyncio
async def test_auto_compaction_keeps_active_turn_in_same_session() -> None:
    agent_loop = build_test_agent_loop()
    agent_loop.stats.session_prompt_tokens = 11
    agent_loop.stats.session_completion_tokens = 7
    original_session_id = agent_loop.session_id

    async def compacting_act(msg: str, **_kwargs):
        yield UserMessageEvent(content=msg, message_id="user-1")
        yield CompactStartEvent(
            current_context_tokens=100, threshold=90, tool_call_id="compact-1"
        )
        agent_loop.stats.session_prompt_tokens += 2
        agent_loop.stats.session_completion_tokens += 1
        yield CompactEndEvent(
            tool_call_id="compact-1",
            summary_length=12,
            old_session_id=original_session_id,
            new_session_id=original_session_id,
        )
        yield AssistantEvent(content="continued", message_id="assistant-1")

    agent_loop.act = compacting_act
    session = await create_test_app_server_session(agent_loop)
    try:
        await _consume(session.act("continue"))
    finally:
        await session.close()

    assert session.session_id == original_session_id
    assert any(
        isinstance(entry, PublicCheckpointEntry) and entry.kind == "compaction"
        for entry in session.history
    )
    assert all(entry.session_id == original_session_id for entry in session.history)
    assert session.state.latest_turn is not None
    assert session.state.latest_turn.session_id == original_session_id
    assert session.state.latest_turn.status == "completed"
    assert session.exit_summary().usage.input_tokens == 2
    assert session.exit_summary().usage.output_tokens == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_call", [1, 2], ids=["legacy-runtime", "app-server"])
async def test_root_replacement_keeps_serving_when_previous_shutdown_fails(
    monkeypatch: pytest.MonkeyPatch, failure_call: int
) -> None:
    """*Prepare*: Previous-backend shutdown fails in either replacement cleanup layer.
    *Do*: Attach a fork that replaces the active root.
    *Assert*: The replacement remains active and continues serving requests.
    """
    # Prepare
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)
    original_shutdown = SessionBackendImpl.shutdown
    call_count = 0

    async def failing_shutdown(self: SessionBackendImpl) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == failure_call:
            raise RuntimeError("disk full — shutdown failed")
        await original_shutdown(self)

    monkeypatch.setattr(SessionBackendImpl, "shutdown", failing_shutdown)

    try:
        # Do
        await session.resources.sessions.fork(attach=True)
        await session.resources.refresh()

        # Assert
        assert session.session_id != agent_loop.session_id
        await session.clear_history()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_install_root_restores_previous_backend_when_replacement_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)

    async def fail_replacement_setup(self) -> None:
        raise RuntimeError("replacement setup failed")

    monkeypatch.setattr(
        SessionRuntimeRegistry, "drain_children", fail_replacement_setup
    )

    try:
        with pytest.raises(AppServerResponseError):
            await session.resources.sessions.fork(attach=True)

        await session.resources.refresh()
        assert session.session_id == agent_loop.session_id
        await session.clear_history()
    finally:
        await session.close()
