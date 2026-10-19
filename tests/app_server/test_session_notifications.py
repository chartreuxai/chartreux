"""Notification resync, reconnect, callback delivery, and event-gap session tests."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

from pydantic import ValidationError
import pytest

from chartreux.app_server._model import validate_wire
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import (
    CallbackRequested,
    HistoryEntryUpdated,
    SessionSnapshot,
    SessionUpdated,
    TurnRetrying,
)
from chartreux.app_server.models import (
    CompletedEffectState,
    PublicCallbackEntry,
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicMessageEntry,
    PublicNoticeEntry,
    PublicRetryCategory,
    RootRecoveryNoticeDetail,
    UserAnswer,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    CallbackCallResponse,
    CallbackResultError,
    ClientCapabilities,
    ClientInfo,
    Notification,
    ProtocolErrorCode,
    SessionOptions,
    SessionReadParams,
    SessionReadResponse,
    SessionResumeParams,
    validate_callback_acknowledgement,
)
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerSession, AppServerTurnError
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.events import AssistantEvent, UserMessageEvent
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.utils import RetryReason
from tests.app_server._session_runtime_support import (
    _consume,
    _create_reconnectable_session,
    _isolate_recovery_registry,  # noqa: F401 - autouse fixture registered via module namespace
    _next_event,
    _question_tool_call,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    create_test_app_server_session,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_callback_round_trips_user_input() -> None:
    tool_call = ToolCall(
        id="question-1",
        index=0,
        function=FunctionCall(
            name="ask_user_question",
            arguments=json.dumps({
                "questions": [
                    {
                        "question": "Ship it?",
                        "options": [{"label": "Yes"}, {"label": "No"}],
                    }
                ]
            }),
        ),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[tool_call])],
        [mock_llm_chunk(content="Done")],
    ])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    session = await create_test_app_server_session(agent_loop)
    callback: PublicCallbackEntry | None = None

    try:
        events = []
        async for event in session.act("ask me"):
            if isinstance(event, CallbackRequested):
                callback = event.callback
                await session.respond_to_callback(
                    event.callback.callback_id,
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[UserAnswer(question="Ship it?", answer="Yes")]
                        )
                    ),
                )
            events.append(event)
    finally:
        await session.close()
        await agent_loop.aclose()

    assert any(isinstance(event, HistoryEntryUpdated) for event in events)
    assert callback is not None
    assert callback.related_entry_id == "question-1"
    effect = next(
        entry for entry in session.history if isinstance(entry, PublicEffectEntry)
    )
    assert isinstance(effect.state, CompletedEffectState)
    assert isinstance(effect.state.output, dict)
    assert effect.state.output["cancelled"] is False
    answers = effect.state.output["answers"]
    assert isinstance(answers, list)
    first_answer = answers[0]
    assert isinstance(first_answer, dict)
    assert first_answer["answer"] == "Yes"
    assert session.state.active_callbacks == []


@pytest.mark.asyncio
async def test_tool_execution_has_no_approval_callback() -> None:
    tool_call = ToolCall(
        id="todo-1",
        index=0,
        function=FunctionCall(name="todo", arguments='{"action":"read"}'),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[tool_call])],
        [mock_llm_chunk(content="Done")],
    ])
    config = build_test_vibe_config(enabled_tools=["todo"])
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    session = await create_test_app_server_session(agent_loop)

    try:
        async for event in session.act("read todos"):
            assert not isinstance(event, CallbackRequested)
    finally:
        await session.close()
        await agent_loop.aclose()

    effect = next(
        entry for entry in session.history if isinstance(entry, PublicEffectEntry)
    )
    assert isinstance(effect.state, CompletedEffectState)


async def _open_wire_notification_session(agent_loop):
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    client_info = ClientInfo(name="notification-validation-test", version="1")
    await client.initialize(client_info)
    await client.notify("initialized")
    session = await AppServerSession.open(
        client, client_info=client_info, capabilities=ClientCapabilities()
    )
    return session, client, server_transport, server


@pytest.mark.asyncio
async def test_malformed_projection_notification_resyncs_without_interrupting_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    session, client, server_transport, server = await _open_wire_notification_session(
        agent_loop
    )
    request_entered = asyncio.Event()
    release_request = asyncio.Event()
    resynced = asyncio.Event()
    original_dispatch = server._dispatch_request
    original_resync = session._resync
    read_count = 0

    async def delay_first_read(method: str, raw_params: dict[str, Any]):
        nonlocal read_count
        if method == "session/read":
            read_count += 1
            if read_count == 1:
                request_entered.set()
                await release_request.wait()
        return await original_dispatch(method, raw_params)

    async def tracked_resync(resync_client) -> None:
        await original_resync(resync_client)
        resynced.set()

    monkeypatch.setattr(server, "_dispatch_request", delay_first_read)
    monkeypatch.setattr(session, "_resync", tracked_resync)
    server._event_watermarks[session.session_id] = session.state.event_id + 3
    expected_event_id = server._event_watermark(session.session_id)
    in_flight = asyncio.create_task(
        client.request("session/read", SessionReadParams(session_id=session.session_id))
    )
    try:
        await asyncio.wait_for(request_entered.wait(), timeout=1)
        await server_transport.send({
            "jsonrpc": "2.0",
            "method": "session/updated",
            "params": {
                "eventId": expected_event_id,
                "sessionId": session.session_id,
                "patch": "not-a-patch-list",
                "emittedAt": 1,
            },
        })
        await asyncio.wait_for(resynced.wait(), timeout=1)

        assert session._connection.current is client
        assert not in_flight.done()
        assert session._state.projection.last_event_id == expected_event_id

        release_request.set()
        response = await asyncio.wait_for(in_flight, timeout=1)
        assert response["state"]["session"]["id"] == session.session_id
    finally:
        release_request.set()
        if not in_flight.done():
            in_flight.cancel()
        await asyncio.gather(in_flight, return_exceptions=True)
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_failed_notification_resync_escalates_to_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    session, client, server_transport, _server = await _open_wire_notification_session(
        agent_loop
    )
    resync_started = asyncio.Event()
    reconnect = AsyncMock(return_value=False)

    async def fail_resync(_resync_client) -> None:
        resync_started.set()
        raise RuntimeError("session/read failed")

    monkeypatch.setattr(session, "_resync", fail_resync)
    monkeypatch.setattr(session._connection, "reconnect", reconnect)
    try:
        await server_transport.send({
            "jsonrpc": "2.0",
            "method": "session/updated",
            "params": {"eventId": 1, "sessionId": session.session_id},
        })
        await asyncio.wait_for(resync_started.wait(), timeout=1)
        async with asyncio.timeout(1):
            while reconnect.await_count == 0:
                await asyncio.sleep(0)
        reconnect.assert_awaited_once_with(client)
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_failed_resync_does_not_start_coalesce_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    session, client, _server_transport, _server = await _open_wire_notification_session(
        agent_loop
    )
    read_count = 0

    async def fail_once(_resync_client) -> None:
        nonlocal read_count
        read_count += 1
        if read_count == 1:
            raise RuntimeError("session/read failed")

    monkeypatch.setattr(session, "_read_resync_snapshot", fail_once)
    try:
        with pytest.raises(RuntimeError, match="session/read failed"):
            await session._resync(client)
        assert session._resync_in_flight is False
        assert session._last_resync_completed_at is None

        await session._resync(client)

        assert read_count == 2
        assert session._last_resync_completed_at is not None
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_malformed_ephemeral_notifications_are_dropped_without_resync(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_loop = build_test_agent_loop()
    session, client, server_transport, _server = await _open_wire_notification_session(
        agent_loop
    )
    resync = AsyncMock()
    reconnect = AsyncMock(return_value=False)
    methods = (
        "warning",
        "error",
        "turn/retrying",
        "mcp_catalog/authRequired",
        "mcp_catalog/authUrl",
    )
    monkeypatch.setattr(session, "_resync", resync)
    monkeypatch.setattr(session._connection, "reconnect", reconnect)
    try:
        with caplog.at_level(logging.WARNING, logger="vibe"):
            for index, method in enumerate(methods, start=1):
                await server_transport.send({
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": {},
                })
                async with asyncio.timeout(1):
                    while (
                        sum(
                            f"Dropping malformed app-server notification method {method}"
                            in record.message
                            for record in caplog.records
                        )
                        < 1
                    ):
                        await asyncio.sleep(0)
                assert session._connection.current is client
                assert session._message_task is not None
                assert not session._message_task.done()
                assert (
                    len([
                        record
                        for record in caplog.records
                        if "Dropping malformed app-server notification method"
                        in record.message
                    ])
                    == index
                )

        resync.assert_not_awaited()
        reconnect.assert_not_awaited()
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_malformed_runtime_update_resynchronizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    session, client, server_transport, server = await _open_wire_notification_session(
        agent_loop
    )
    original_resync = session._resync
    resynced = asyncio.Event()
    reconnect = AsyncMock(return_value=False)
    await agent_loop.config_orchestrator.set_field(
        "/theme",
        "server-runtime-after-malformed-update",
        target_layer=OverridesLayer.NAME,
    )
    cached_theme = session.resources.config.current.theme

    async def tracked_resync(resync_client) -> None:
        await original_resync(resync_client)
        resynced.set()

    monkeypatch.setattr(session, "_resync", tracked_resync)
    monkeypatch.setattr(session._connection, "reconnect", reconnect)
    server._event_watermarks[session.session_id] = session.state.event_id + 2
    expected_event_id = server._event_watermark(session.session_id)
    try:
        await server_transport.send({
            "jsonrpc": "2.0",
            "method": "runtime/updated",
            "params": {},
        })
        await asyncio.wait_for(resynced.wait(), timeout=1)
        async with asyncio.timeout(1):
            while (
                session.resources.config.current.theme
                != "server-runtime-after-malformed-update"
            ):
                await asyncio.sleep(0)

        assert cached_theme != session.resources.config.current.theme
        assert session._state.projection.last_event_id == expected_event_id
        assert session._connection.current is client
        reconnect.assert_not_awaited()
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_malformed_notification_coalesced_resync_runs_after_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    session, _client, server_transport, server = await _open_wire_notification_session(
        agent_loop
    )
    now = [10.0]
    read_count = 0
    original_dispatch = server._dispatch_request
    sleep_delays: list[float] = []

    async def count_reads(method: str, raw_params: dict[str, Any]):
        nonlocal read_count
        if method == "session/read":
            read_count += 1
        return await original_dispatch(method, raw_params)

    async def advance_clock(delay: float) -> None:
        sleep_delays.append(delay)
        now[0] += delay
        await asyncio.sleep(0)

    monkeypatch.setattr(server, "_dispatch_request", count_reads)
    session._resync_clock = lambda: now[0]
    session._resync_sleep = advance_clock

    async def send_malformed() -> None:
        await server_transport.send({
            "jsonrpc": "2.0",
            "method": "session/updated",
            "params": {
                "eventId": 1,
                "sessionId": session.session_id,
                "patch": "not-a-patch-list",
                "emittedAt": 1,
            },
        })

    try:
        await send_malformed()
        async with asyncio.timeout(1):
            while read_count < 1 or session._resync_in_flight:
                await asyncio.sleep(0)

        now[0] += 0.1
        await send_malformed()
        async with asyncio.timeout(1):
            while read_count < 2 or session._resync_in_flight:
                await asyncio.sleep(0)

        assert sleep_delays == pytest.approx([0.4])
        assert session._resync_pending is False
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_malformed_notification_log_does_not_include_payload_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent_loop = build_test_agent_loop()
    session, _client, server_transport, _server = await _open_wire_notification_session(
        agent_loop
    )
    sentinel = "private-notification-sentinel"
    try:
        with caplog.at_level(logging.WARNING, logger="vibe"):
            await server_transport.send({
                "jsonrpc": "2.0",
                "method": "warning",
                "params": {"warning": sentinel},
            })
            async with asyncio.timeout(1):
                while not any(
                    "Dropping malformed app-server notification method warning"
                    in record.message
                    for record in caplog.records
                ):
                    await asyncio.sleep(0)

        assert all(sentinel not in record.message for record in caplog.records)
        assert all(sentinel not in str(record.args) for record in caplog.records)
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_notification_resync_storm_uses_one_session_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    session, _client, server_transport, server = await _open_wire_notification_session(
        agent_loop
    )
    first_read_started = asyncio.Event()
    release_read = asyncio.Event()
    read_count = 0
    original_dispatch = server._dispatch_request

    async def delay_first_read(method: str, raw_params: dict[str, Any]):
        nonlocal read_count
        if method == "session/read":
            read_count += 1
            if read_count == 1:
                first_read_started.set()
                await release_read.wait()
        return await original_dispatch(method, raw_params)

    monkeypatch.setattr(server, "_dispatch_request", delay_first_read)
    try:
        for _ in range(32):
            await server_transport.send({
                "jsonrpc": "2.0",
                "method": "session/updated",
                "params": {
                    "eventId": 1,
                    "sessionId": session.session_id,
                    "patch": "not-a-patch-list",
                    "emittedAt": 1,
                },
            })
        await server_transport.send({
            "jsonrpc": "2.0",
            "method": "storm/complete",
            "params": {},
        })
        await asyncio.wait_for(first_read_started.wait(), timeout=1)
        release_read.set()

        async with asyncio.timeout(1):
            while "storm/complete" not in session._dropped_notification_methods:
                await asyncio.sleep(0)

        assert read_count == 1
    finally:
        release_read.set()
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_unknown_notification_methods_are_bounded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent_loop = build_test_agent_loop()
    session, client, _server_transport, _server = await _open_wire_notification_session(
        agent_loop
    )
    try:
        with caplog.at_level(logging.WARNING, logger="vibe"):
            for index in range(100):
                await session._handle_notification(
                    client, Notification(method=f"future/method-{index}", params={})
                )

        assert len(session._dropped_notification_methods) == 64
        assert (
            len([
                record
                for record in caplog.records
                if "Dropping unknown app-server notification method" in record.message
            ])
            == 64
        )
        assert (
            sum(
                "suppressing further unknown-method warnings" in record.message
                for record in caplog.records
            )
            == 1
        )
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_unknown_notification_does_not_interrupt_pump_or_in_flight_rpc(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_loop = build_test_agent_loop()
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    client_info = ClientInfo(name="unknown-notification-test", version="1")
    await client.initialize(client_info)
    await client.notify("initialized")
    session = await AppServerSession.open(
        client, client_info=client_info, capabilities=ClientCapabilities()
    )
    request_entered = asyncio.Event()
    release_request = asyncio.Event()
    original_dispatch = server._dispatch_request

    async def delay_read(method: str, raw_params: dict[str, Any]):
        if method == "session/read":
            request_entered.set()
            await release_request.wait()
        return await original_dispatch(method, raw_params)

    monkeypatch.setattr(server, "_dispatch_request", delay_read)
    stream = session.events()
    projected_event = asyncio.create_task(_next_event(stream, SessionUpdated))
    in_flight = asyncio.create_task(
        client.request("session/read", SessionReadParams(session_id=session.session_id))
    )
    try:
        await asyncio.wait_for(request_entered.wait(), timeout=1)
        unknown = {
            "jsonrpc": "2.0",
            "method": "future/resourceUpdated",
            "params": {"revision": 1},
        }
        with caplog.at_level(logging.WARNING, logger="vibe"):
            await server_transport.send(unknown)
            await server_transport.send(unknown)
            async with asyncio.timeout(1):
                while (
                    "future/resourceUpdated"
                    not in session._dropped_notification_methods
                ):
                    await asyncio.sleep(0)

            assert not in_flight.done()
            current_client = session._connection.current
            assert current_client is client
            event_id = session._state.projection.last_event_id + 1
            await server_transport.send({
                "jsonrpc": "2.0",
                "method": "session/updated",
                "params": {
                    "eventId": event_id,
                    "sessionId": session.session_id,
                    "patch": [
                        {
                            "op": "replace",
                            "path": "/title",
                            "value": "known event survived",
                        }
                    ],
                    "emittedAt": 1,
                },
            })
            event = await asyncio.wait_for(projected_event, timeout=1)

        assert event.session.title == "known event survived"
        assert session._connection.current is client
        assert (
            sum(
                "Dropping unknown app-server notification method future/resourceUpdated"
                in record.message
                for record in caplog.records
            )
            == 1
        )

        release_request.set()
        response = await asyncio.wait_for(in_flight, timeout=1)
        assert response["state"]["session"]["id"] == session.session_id
    finally:
        release_request.set()
        if not projected_event.done():
            projected_event.cancel()
        if not in_flight.done():
            in_flight.cancel()
        await asyncio.gather(projected_event, in_flight, return_exceptions=True)
        await stream.aclose()
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_pump_read_failures_use_increasing_reconnect_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    session, _server = await _create_reconnectable_session_with_server(agent_loop)
    existing_pump = session._message_task
    if existing_pump is not None:
        existing_pump.cancel()
        await asyncio.gather(existing_pump, return_exceptions=True)
    session._message_task = None
    client = session._connection.current
    assert client is not None
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async def read_failure(_client) -> Exception:
        return RuntimeError("read failed")

    reconnect = AsyncMock(side_effect=[True, True, False])
    monkeypatch.setattr(session, "_reconnect_sleep", record_sleep)
    monkeypatch.setattr(session, "_receive_messages", read_failure)
    monkeypatch.setattr(session._connection, "reconnect", reconnect)
    monkeypatch.setattr(session, "_attach_with_backoff", AsyncMock(return_value=client))

    try:
        session._message_task = asyncio.create_task(session._pump_messages())
        await asyncio.wait_for(session._message_task, timeout=1)

        assert delays == [0.25, 0.5, 1.0]
        assert reconnect.await_count == 3
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_reconnect_backoff_is_bounded_and_close_interrupts_wait() -> None:
    agent_loop = build_test_agent_loop()
    session, server = await _create_reconnectable_session_with_server(agent_loop)
    sleeps: asyncio.Queue[tuple[float, asyncio.Event]] = asyncio.Queue()

    async def controlled_sleep(delay: float) -> None:
        gate = asyncio.Event()
        sleeps.put_nowait((delay, gate))
        await gate.wait()

    async def fail_resume(_params: SessionResumeParams):
        raise RuntimeError("backend unavailable")

    server._session_backend_host.resume = fail_resume  # type: ignore[method-assign]
    session._reconnect_sleep = controlled_sleep
    client = session._connection.current
    assert client is not None

    try:
        await client.close()
        expected_delays = (0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 5.0)
        for index, expected in enumerate(expected_delays):
            delay, gate = await asyncio.wait_for(sleeps.get(), timeout=1)
            assert delay == expected
            if index < len(expected_delays) - 1:
                gate.set()

        await asyncio.wait_for(session.close(), timeout=1)
        assert session._message_task is None
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_event_stream_recovers_after_attach_failure() -> None:
    agent_loop = build_test_agent_loop()
    session, server = await _create_reconnectable_session_with_server(agent_loop)
    sleeps: asyncio.Queue[tuple[float, asyncio.Event]] = asyncio.Queue()
    resume_calls = 0
    original_resume = server._session_backend_host.resume

    async def fail_once(params: SessionResumeParams):
        nonlocal resume_calls
        resume_calls += 1
        if resume_calls == 1:
            raise RuntimeError("backend temporarily unavailable")
        return await original_resume(params)

    async def controlled_sleep(delay: float) -> None:
        gate = asyncio.Event()
        sleeps.put_nowait((delay, gate))
        await gate.wait()

    server._session_backend_host.resume = fail_once  # type: ignore[method-assign]
    session._reconnect_sleep = controlled_sleep
    stream = session.events()
    next_event = asyncio.create_task(anext(stream))
    client = session._connection.current
    assert client is not None

    try:
        await asyncio.sleep(0)
        await client.close()
        delay, gate = await asyncio.wait_for(sleeps.get(), timeout=1)
        assert delay == 0.25
        gate.set()
        delay, gate = await asyncio.wait_for(sleeps.get(), timeout=1)
        assert delay == 0.5
        gate.set()

        event = await asyncio.wait_for(next_event, timeout=1)
        assert isinstance(event, SessionSnapshot)
        assert resume_calls >= 2
        assert session._message_task is not None
        assert not session._message_task.done()
    finally:
        if not next_event.done():
            next_event.cancel()
        await asyncio.gather(next_event, return_exceptions=True)
        await stream.aclose()
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_reconnect_backoff_resets_after_successful_message_flow() -> None:
    agent_loop = build_test_agent_loop()
    session, server = await _create_reconnectable_session_with_server(agent_loop)
    sleeps: asyncio.Queue[tuple[float, asyncio.Event]] = asyncio.Queue()
    resume_calls = 0
    original_resume = server._session_backend_host.resume

    async def fail_twice_then_again(params: SessionResumeParams):
        nonlocal resume_calls
        resume_calls += 1
        if resume_calls in {1, 2, 4}:
            raise RuntimeError("backend temporarily unavailable")
        return await original_resume(params)

    async def controlled_sleep(delay: float) -> None:
        gate = asyncio.Event()
        sleeps.put_nowait((delay, gate))
        await gate.wait()

    server._session_backend_host.resume = fail_twice_then_again  # type: ignore[method-assign]
    session._reconnect_sleep = controlled_sleep
    stream = session.events()
    next_event = asyncio.create_task(anext(stream))
    client = session._connection.current
    assert client is not None

    try:
        await asyncio.sleep(0)
        await client.close()
        delay, gate = await asyncio.wait_for(sleeps.get(), timeout=1)
        assert delay == 0.25
        gate.set()
        delay, gate = await asyncio.wait_for(sleeps.get(), timeout=1)
        assert delay == 0.5
        gate.set()
        delay, gate = await asyncio.wait_for(sleeps.get(), timeout=1)
        assert delay == 1.0
        gate.set()

        event = await asyncio.wait_for(next_event, timeout=1)
        assert isinstance(event, SessionSnapshot)
        recovered_client = session._connection.current
        assert recovered_client is not None
        root = server._root
        assert root is not None
        await server._notify("runtime/updated", root.runtime_updated_params())
        async with asyncio.timeout(1):
            while (
                session._reconnect_backoff_seconds != 0.25
                or session._message_task is None
            ):
                await asyncio.sleep(0)
        await recovered_client.close()

        delay, gate = await asyncio.wait_for(sleeps.get(), timeout=1)
        assert delay == 0.25
        gate.set()
        async with asyncio.timeout(1):
            while resume_calls < 4:
                await asyncio.sleep(0)
        assert resume_calls == 4
    finally:
        if not next_event.done():
            next_event.cancel()
        await asyncio.gather(next_event, return_exceptions=True)
        await stream.aclose()
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_reconnect_resumes_live_turn_and_redelivers_open_callback() -> None:
    tool_call = _question_tool_call()
    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[tool_call])],
        [mock_llm_chunk(content="Done")],
    ])
    config = build_test_vibe_config(enabled_tools=["ask_user_question"])
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    session = await _create_reconnectable_session(agent_loop)
    stream = session.act("read todos")

    try:
        first_callback = await _next_event(stream, CallbackRequested)
        disconnected_client = session._connection.current
        assert disconnected_client is not None
        await disconnected_client.close()

        snapshot = await _next_event(stream, SessionSnapshot)
        redelivered = await _next_event(stream, CallbackRequested)
        assert snapshot.state.session.id == session.session_id
        assert redelivered.callback.callback_id == first_callback.callback.callback_id
        assert session._connection.current is not disconnected_client

        await session.respond_to_callback(
            redelivered.callback.callback_id,
            UserInputCallbackOutput(
                result=UserQuestionResult(
                    answers=[UserAnswer(question="Ship it?", answer="Yes")]
                )
            ),
        )
        _ = [event async for event in stream]
    finally:
        await stream.aclose()
        await session.close()

    effect = next(
        entry for entry in session.history if isinstance(entry, PublicEffectEntry)
    )
    assert isinstance(effect.state, CompletedEffectState)
    assert not any(
        isinstance(entry, PublicCheckpointEntry) and entry.kind == "resume"
        for entry in session.history
    )


@pytest.mark.asyncio
async def test_resume_and_reconnect_preserve_session_options(tmp_path: Path) -> None:
    # Prepare
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()
    resume_params: list[SessionResumeParams] = []
    options = SessionOptions(cwd=str(tmp_path), disabled_tools=["bash"])
    session = await _create_reconnectable_session(
        build_test_agent_loop(config=config),
        session_options=options,
        resume_params=resume_params,
    )

    try:
        # Do
        await session.resume(saved_session_id)
        snapshot = asyncio.create_task(_next_event(session.events(), SessionSnapshot))
        client = session._connection.current
        assert client is not None
        await client.close()
        await asyncio.wait_for(snapshot, timeout=1)

        # Assert
        assert [params.agent_config for params in resume_params] == [options, options]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_live_retry_is_in_session_read_and_reconnect_snapshot() -> None:
    retry_started = asyncio.Event()
    agent_loop = build_test_agent_loop()

    async def retrying_act(*_args, turn_options, **_kwargs):
        assert turn_options.retry_sink is not None
        await turn_options.retry_sink(RetryReason.from_http_status(429))
        retry_started.set()
        await asyncio.Event().wait()
        yield AssistantEvent(content="unreachable", message_id="assistant-1")

    agent_loop.act = retrying_act
    session = await _create_reconnectable_session(agent_loop)
    stream = session.act("wait for provider")
    retry_snapshot_task = asyncio.create_task(_next_event(stream, SessionSnapshot))

    try:
        await asyncio.wait_for(retry_started.wait(), timeout=1)
        retry_snapshot = await asyncio.wait_for(retry_snapshot_task, timeout=1)
        await _next_event(stream, TurnRetrying)
        client = session._connection.current
        assert client is not None
        response = validate_wire(
            SessionReadResponse,
            await client.request(
                "session/read", SessionReadParams(session_id=session.session_id)
            ),
        )
        retrying = response.state.retrying
        assert retrying is not None
        assert retrying.category is PublicRetryCategory.RATE_LIMITED
        assert retrying.detail == "HTTP 429"
        assert response.state.latest_turn is not None
        assert retrying.turn_id == response.state.latest_turn.id
        assert retry_snapshot.state.retrying == retrying

        await client.close()
        snapshot = await _next_event(stream, SessionSnapshot)
        assert snapshot.state.retrying == retrying

        await session.interrupt()
        await _consume(stream)
    finally:
        if not retry_snapshot_task.done():
            retry_snapshot_task.cancel()
            with suppress(asyncio.CancelledError):
                await retry_snapshot_task
        await stream.aclose()
        await session.close()


@pytest.mark.asyncio
async def test_callback_opened_while_detached_is_delivered_after_reconnect() -> None:
    backend_started = asyncio.Event()
    release_backend = asyncio.Event()
    allow_reconnect = asyncio.Event()
    tool_call = _question_tool_call()

    class GatedBackend(FakeBackend):
        async def complete_streaming(self, **kwargs):
            backend_started.set()
            await release_backend.wait()
            async for chunk in super().complete_streaming(**kwargs):
                yield chunk

    backend = GatedBackend([
        [mock_llm_chunk(content="", tool_calls=[tool_call])],
        [mock_llm_chunk(content="Done")],
    ])
    config = build_test_vibe_config(enabled_tools=["ask_user_question"])
    session = await _create_reconnectable_session(
        build_test_agent_loop(config=config, backend=backend, enable_streaming=True),
        reconnect_gate=allow_reconnect,
    )
    stream = session.act("open approval while detached")
    callback_task = asyncio.create_task(_next_event(stream, CallbackRequested))

    try:
        await asyncio.wait_for(backend_started.wait(), timeout=1)
        disconnected_client = session._connection.current
        assert disconnected_client is not None
        await disconnected_client.close()
        release_backend.set()
        await asyncio.sleep(0.05)
        assert not callback_task.done()

        allow_reconnect.set()
        callback = await asyncio.wait_for(callback_task, timeout=1)
        await session.respond_to_callback(
            callback.callback.callback_id,
            UserInputCallbackOutput(
                result=UserQuestionResult(
                    answers=[UserAnswer(question="Ship it?", answer="Yes")]
                )
            ),
        )
        _ = [event async for event in stream]
    finally:
        allow_reconnect.set()
        release_backend.set()
        if not callback_task.done():
            callback_task.cancel()
            with suppress(asyncio.CancelledError):
                await callback_task
        await stream.aclose()
        await session.close()

    effect = next(
        entry for entry in session.history if isinstance(entry, PublicEffectEntry)
    )
    assert isinstance(effect.state, CompletedEffectState)


@pytest.mark.asyncio
async def test_callback_result_requires_output_or_error() -> None:
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)

    try:
        client = await session._ensure_attached()
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "callback/result",
                {
                    "sessionId": session.session_id,
                    "result": {"callbackId": "callback-1"},
                },
            )
    finally:
        await session.close()
        await agent_loop.aclose()

    assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
    assert (
        exc_info.value.error.message == "Callback result must include output or error"
    )


@pytest.mark.asyncio
async def test_callback_error_is_retry_safe_and_rejects_conflicts() -> None:
    tool_call = _question_tool_call()
    backend = FakeBackend([[mock_llm_chunk(content="", tool_calls=[tool_call])]])
    config = build_test_vibe_config(enabled_tools=["ask_user_question"])
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    session = await create_test_app_server_session(agent_loop)
    error = CallbackResultError(
        message="Client cannot answer this callback",
        code="client_unavailable",
        details={"retryable": False},
    )

    try:
        with pytest.raises(AppServerTurnError, match=error.message):
            async for event in session.act("read todos"):
                if not isinstance(event, CallbackRequested):
                    continue
                await session.reject_callback(event.callback.callback_id, error)
                await session.reject_callback(event.callback.callback_id, error)
                with pytest.raises(AppServerResponseError) as exc_info:
                    await session.reject_callback(
                        event.callback.callback_id,
                        CallbackResultError(message="A different callback error"),
                    )
                assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT
    finally:
        await session.close()
        await agent_loop.aclose()


def test_callback_delivery_acknowledgement_has_no_semantic_output() -> None:
    response = CallbackCallResponse(callback_id="approval-1")

    assert validate_callback_acknowledgement("approval-1", response) is response
    assert response.model_dump(mode="json", by_alias=True) == {
        "callbackId": "approval-1",
        "accepted": True,
    }

    with pytest.raises(ValueError, match="does not match"):
        validate_callback_acknowledgement("approval-2", response)


@pytest.mark.asyncio
async def test_completed_turn_does_not_retain_callback_for_redelivery() -> None:
    tool_call = _question_tool_call()
    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[tool_call])],
        [mock_llm_chunk(content="Done")],
    ])
    config = build_test_vibe_config(enabled_tools=["ask_user_question"])
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    session = await attach_test_app_server_session(
        AppServerClient(client_transport, run_peer=server.serve)
    )

    try:
        async for event in session.act("read todos"):
            if isinstance(event, CallbackRequested):
                await session.respond_to_callback(
                    event.callback.callback_id,
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[UserAnswer(question="Ship it?", answer="Yes")]
                        )
                    ),
                )

        assert legacy_backend(server).session.turns.callbacks == []
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_event_gap_resynchronizes_through_session_read(monkeypatch) -> None:
    agent_loop = build_test_agent_loop()
    turn_started = asyncio.Event()
    release_turn = asyncio.Event()

    async def blocking_act(msg: str, **_kwargs):
        turn_started.set()
        await release_turn.wait()
        yield UserMessageEvent(content=msg, message_id="resynced-user")
        yield AssistantEvent(content="Recovered", message_id="resynced-assistant")

    agent_loop.act = blocking_act
    client = start_test_app_server(agent_loop)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    request = client.request
    methods = []
    resynced = asyncio.Event()

    async def tracked_request(method, params=None, *, wait_for_incoming=False):
        methods.append(method)
        return await request(method, params, wait_for_incoming=wait_for_incoming)

    monkeypatch.setattr(client, "request", tracked_request)
    resync = session._resync

    async def tracked_resync(resync_client) -> None:
        await resync(resync_client)
        resynced.set()

    monkeypatch.setattr(session, "_resync", tracked_resync)
    server._event_watermarks[session.session_id] = session.state.event_id + 1

    turn = asyncio.create_task(_consume(session.act("recover")))
    try:
        await turn_started.wait()
        await resynced.wait()
        release_turn.set()
        events = await turn
    finally:
        release_turn.set()
        turn.cancel()
        await session.close()

    assert "session/read" in methods
    assert any(isinstance(event, SessionSnapshot) for event in events)
    assert session._state.projection._last_event_id == server._event_watermark(
        session.session_id
    )
    assert any(
        isinstance(entry, PublicMessageEntry)
        and entry.role == "assistant"
        and entry.text == "Recovered"
        for entry in session.history
    )


@pytest.mark.asyncio
async def test_terminal_event_gap_completes_waiting_turn_from_snapshot(
    monkeypatch,
) -> None:
    agent_loop = build_test_agent_loop()

    async def completed_act(msg: str, **_kwargs):
        yield UserMessageEvent(content=msg, message_id="user-1")
        yield AssistantEvent(content="done", message_id="assistant-1")

    agent_loop.act = completed_act
    session = await create_test_app_server_session(agent_loop)
    client = session._connection.current
    assert client is not None
    request = client.request
    methods: list[str] = []

    async def tracked_request(method, params=None, *, wait_for_incoming=False):
        methods.append(method)
        return await request(method, params, wait_for_incoming=wait_for_incoming)

    monkeypatch.setattr(client, "request", tracked_request)
    consume = session._state.projection.consume
    gap_injected = False

    def consume_with_terminal_gap(notification):
        nonlocal gap_injected
        if notification.method == "turn/completed" and not gap_injected:
            gap_injected = True
            notification.params["eventId"] += 1
        return consume(notification)

    monkeypatch.setattr(session._state.projection, "consume", consume_with_terminal_gap)
    try:
        events = await asyncio.wait_for(_consume(session.act("finish")), timeout=1)
    finally:
        await session.close()

    assert gap_injected
    assert "session/read" in methods
    assert any(isinstance(event, SessionSnapshot) for event in events)
    assert session.state.latest_turn is not None
    assert session.state.latest_turn.status == "completed"


async def _create_reconnectable_session_with_server(
    agent_loop,
) -> tuple[AppServerSession, AppServer]:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)

    async def serve_initial_connection() -> None:
        await server.serve_connection(server_transport, close_on_disconnect=False)

    def reconnect() -> AppServerClient:
        next_client_transport, next_server_transport = memory_transport_pair()

        async def serve_connection() -> None:
            await server.serve_connection(
                next_server_transport, close_on_disconnect=False
            )

        return AppServerClient(next_client_transport, run_peer=serve_connection)

    client = AppServerClient(client_transport, run_peer=serve_initial_connection)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="retry-test-client", version="1"),
        capabilities=ClientCapabilities(),
        client_factory=reconnect,
    )
    return session, server


def test_root_recovery_notice_detail_discriminates_on_kind() -> None:
    # R2: the milestone notice is a member of the NoticeDetail union,
    # discriminated by kind="root_recovery" — a PublicNoticeEntry carrying
    # it round-trips through the wire into RootRecoveryNoticeDetail.
    entry = validate_wire(
        PublicNoticeEntry,
        {
            "id": "root-recovery:scheduled:root-auto-t:1",
            "sessionId": "s",
            "createdAt": 1,
            "updatedAt": 1,
            "generationStatus": "completed",
            "type": "notice",
            "level": "info",
            "message": "Automatic retry 1 of 2 scheduled in 30.0s",
            "detail": {
                "kind": "root_recovery",
                "milestone": "scheduled",
                "episodeId": "root-auto-t",
                "attempt": 1,
                "limit": 2,
                "delaySeconds": 30.0,
                "eligibleClass": "timeout",
                "affectedDeployment": "test-first/first",
            },
        },
    )
    assert isinstance(entry.detail, RootRecoveryNoticeDetail)
    assert entry.detail.kind == "root_recovery"
    assert entry.detail.milestone == "scheduled"
    assert entry.detail.episode_id == "root-auto-t"
    assert entry.detail.attempt == 1
    assert entry.detail.limit == 2
    assert entry.detail.delay_seconds == 30.0
    assert entry.detail.eligible_class == "timeout"
    assert entry.detail.affected_deployment == "test-first/first"
    # The wire spelling is camelCase; snake_case is rejected.
    assert "episodeId" in entry.model_dump(mode="json")["detail"]
    with pytest.raises(ValidationError):
        validate_wire(
            PublicNoticeEntry,
            {
                "id": "n",
                "sessionId": "s",
                "createdAt": 1,
                "updatedAt": 1,
                "generationStatus": "completed",
                "type": "notice",
                "level": "info",
                "message": "m",
                "detail": {"kind": "root_recovery", "episode_id": "root-auto-t"},
            },
        )


@pytest.mark.parametrize(
    "milestone", ["scheduled", "started", "recovered", "exhausted", "cancelled"]
)
def test_root_recovery_notice_milestone_vocabulary(milestone: str) -> None:
    detail = validate_wire(
        RootRecoveryNoticeDetail, {"milestone": milestone, "episodeId": "e"}
    )
    assert detail.milestone == milestone
    with pytest.raises(ValidationError):
        validate_wire(RootRecoveryNoticeDetail, {"milestone": "paused"})
    with pytest.raises(ValidationError):
        validate_wire(RootRecoveryNoticeDetail, {"kind": "subagent_recovery"})
