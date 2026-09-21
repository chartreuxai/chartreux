from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from chartreux.app_server._runtime import (
    AgentRuntimeFactory,
    HarnessProcess,
    RootOpenRequest,
    _resolve_resume_session_id,
)
from chartreux.app_server._session_backend_impl import (
    SessionBackendHostImpl,
    SessionBackendImpl,
)
from chartreux.app_server._session_backend_port import (
    SessionBackend,
    SessionBackendError,
    SessionBackendHost,
)
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import HistoryEntryAdded, ServerWarning
from chartreux.app_server.models import PublicError, TextContentBlock
from chartreux.app_server.protocol import (
    ClientCapabilities,
    ClientInfo,
    ContextInjectParams,
    ProtocolErrorCode,
    ServerWarningParams,
    SessionOptions,
    SessionReadParams,
    SessionSettingsUpdateParams,
)
from chartreux.app_server.server import _SESSION_BACKEND_METHODS, AppServer
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.session.session_lease import SessionBusyError, SessionLease
from chartreux.core.session.session_loader import SessionLoader
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import build_test_app_server


def test_session_backend_contract_covers_the_complete_session_lifecycle() -> None:
    assert _protocol_members(SessionBackend) == {
        "authorization_changed",
        "compact",
        "dispatch_extension",
        "enqueue_turn",
        "flush_events",
        "guard_request",
        "inject_context",
        "interrupt_turn",
        "open_callbacks",
        "publish_callback",
        "publish_notification",
        "read",
        "read_mcp",
        "read_policy",
        "read_roots",
        "read_turn_queue",
        "reconfigure_mcp",
        "references_child",
        "reject_callback_delivery",
        "reload_config",
        "remove_queued_turn",
        "replace_policy",
        "replace_queued_turn",
        "replace_roots",
        "respond_to_callback",
        "resume_turn_queue",
        "runtime_updated_params",
        "session_id",
        "shutdown",
        "start_turn",
        "steer_turn",
        "subscribe",
        "suspend_mcp",
        "update_settings",
        "write_config",
    }
    assert _SESSION_BACKEND_METHODS == {
        "callback/result",
        "config/policy/read",
        "config/policy/replace",
        "config/reload",
        "config/write",
        "policy/roots/read",
        "policy/roots/replace",
        "session/compact",
        "session/context/inject",
        "session/settings/update",
        "app_server/session/turn/enqueue",
        "app_server/session/turn/queue/read",
        "app_server/session/turn/queue/remove",
        "app_server/session/turn/queue/replace",
        "app_server/session/turn/queue/resume",
        "turn/interrupt",
        "turn/start",
        "turn/steer",
    }


def test_session_backend_host_contract_owns_session_selection() -> None:
    assert _protocol_members(SessionBackendHost) == {
        "continue_latest",
        "fork",
        "list",
        "read",
        "rename",
        "resume",
        "shutdown",
        "start",
        "stop_background_tasks",
    }


def test_session_backend_errors_keep_semantic_code_and_data() -> None:
    error = SessionBackendError(
        ProtocolErrorCode.STALE_TURN,
        "The active turn changed",
        {"activeTurnId": "turn-2"},
    )

    assert error.code is ProtocolErrorCode.STALE_TURN
    assert error.data == {"activeTurnId": "turn-2"}
    assert str(error) == "The active turn changed"
    assert ProtocolErrorCode.CALLBACK_CLOSED.value == "callback_closed"


def test_app_server_passes_services_to_session_backend_host_factory() -> None:
    captured_services: object | None = None
    expected_host = cast(SessionBackendHost, object())

    def factory(services: object) -> SessionBackendHost:
        nonlocal captured_services
        captured_services = services
        return expected_host

    _, server_transport = memory_transport_pair()
    server = AppServer(server_transport, session_backend_host_factory=factory)

    assert captured_services is server
    assert server._session_backend_host is expected_host


def test_app_server_rejects_empty_session_backend_host_factory_result() -> None:
    def empty_factory(_: object) -> SessionBackendHost:
        return cast(SessionBackendHost, None)

    _, server_transport = memory_transport_pair()

    with pytest.raises(TypeError, match="must return a SessionBackendHost"):
        AppServer(server_transport, session_backend_host_factory=empty_factory)


@pytest.mark.asyncio
async def test_app_server_shutdown_waits_for_host_cleanup() -> None:
    """*Prepare*: A Session Host whose shutdown remains active.
    *Do*: Close the App Server before allowing Host cleanup to finish.
    *Assert*: App Server shutdown waits until Host cleanup is complete.
    """
    # Prepare
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    class SlowShutdownHost:
        async def shutdown(self) -> None:
            entered.set()
            await release.wait()
            finished.set()

    _, server_transport = memory_transport_pair()
    host = SlowShutdownHost()
    server = AppServer(
        server_transport,
        session_backend_host_factory=lambda _services: cast(SessionBackendHost, host),
    )

    # Do
    closing = asyncio.create_task(server._close_root())
    await entered.wait()
    await asyncio.sleep(0)

    # Assert
    assert not closing.done()
    assert not finished.is_set()
    release.set()
    await asyncio.wait_for(closing, timeout=1)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_app_server_root_is_the_legacy_session_backend() -> None:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(build_test_agent_loop(), server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="test", version="0"),
        capabilities=ClientCapabilities(),
    )
    try:
        host = server._session_backend_host
        backend = server._require_root()
        _accept_session_backend_host(host)
        _accept_session_backend(backend)
        assert isinstance(backend, SessionBackendImpl)

        event_task = server._backend_event_task
        assert event_task is not None
        event_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await event_task

        response = await backend.read(SessionReadParams(session_id=session.session_id))
        subscription = await backend.subscribe(
            SessionReadParams(session_id=session.session_id)
        )
        with pytest.raises(SessionBackendError) as exc_info:
            await backend.subscribe(SessionReadParams(session_id=session.session_id))
        assert exc_info.value.code is ProtocolErrorCode.CONFLICT
        injected = await backend.inject_context(
            ContextInjectParams(
                session_id=session.session_id,
                input=[TextContentBlock(text="remember this")],
                as_message=True,
            )
        )
        envelope = await anext(subscription.events)
        await backend.inject_context(
            ContextInjectParams(
                session_id=session.session_id,
                input=[TextContentBlock(text="drop this event")],
                as_message=True,
            )
        )
        backend._events.get_nowait()
        await backend.inject_context(
            ContextInjectParams(
                session_id=session.session_id,
                input=[TextContentBlock(text="detect the gap")],
                as_message=True,
            )
        )
        settings_response = await backend.update_settings(
            SessionSettingsUpdateParams(session_id=session.session_id, max_turns=3)
        )

        assert isinstance(host, SessionBackendHostImpl)
        assert response.state.session.id == session.session_id
        assert subscription.snapshot.state.session.id == session.session_id
        assert (
            subscription.snapshot.last_event_id == subscription.snapshot.state.event_id
        )
        assert isinstance(envelope.event, HistoryEntryAdded)
        assert envelope.event.entry == injected.response.entries[0]
        assert settings_response.response.model_dump() == {}
        with pytest.raises(SessionBackendError) as gap_exc_info:
            await anext(subscription.events)
        assert gap_exc_info.value.code is ProtocolErrorCode.STALE_CURSOR
        replacement = await backend.subscribe(
            SessionReadParams(session_id=session.session_id)
        )
        assert replacement.snapshot.state.session.id == session.session_id
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_legacy_backend_subscription_forwards_direct_events_and_closes() -> None:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(build_test_agent_loop(), server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="test", version="0"),
        capabilities=ClientCapabilities(),
    )
    backend = server._require_root()
    assert isinstance(backend, SessionBackendImpl)
    event_task = server._backend_event_task
    assert event_task is not None
    event_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await event_task
    subscription = await backend.subscribe(
        SessionReadParams(session_id=session.session_id)
    )

    handled = await backend.publish_notification(
        "warning", ServerWarningParams(warning=PublicError(message="careful"))
    )
    envelope = await anext(subscription.events)
    ignored = await backend.publish_notification(
        "unknown/event", ServerWarningParams(warning=PublicError(message="ignored"))
    )

    assert handled is True
    assert ignored is False
    assert isinstance(envelope.event, ServerWarning)
    assert envelope.event_id is None
    assert envelope.method == "warning"

    next_event = asyncio.ensure_future(anext(subscription.events))
    await backend.shutdown()
    with pytest.raises(StopAsyncIteration):
        await next_event
    await client.close()


@pytest.mark.parametrize(
    ("stored_ids", "reference", "expected"),
    [
        (["12345678-first"], "12345678-first", "12345678-first"),
        (["12345678-first"], "12345678", "12345678-first"),
        (["12345678-first", "12345678"], "12345678", "12345678"),
        (["12345678-first", "12345678-second"], "12345678-first", "12345678-first"),
        (["12345678-first", "12345678-first"], "12345678", "12345678-first"),
        (["12345678-first"], "1234567", "1234567"),
        ([], "missing-session", "missing-session"),
    ],
)
def test_resume_reference_resolution(
    stored_ids: list[str], reference: str, expected: str
) -> None:
    with patch.object(
        SessionLoader,
        "list_sessions",
        return_value=[{"session_id": value} for value in stored_ids],
    ):
        assert (
            _resolve_resume_session_id(ChartreuxConfigSchema(), reference) == expected
        )


def test_resume_reference_rejects_ambiguous_short_id() -> None:
    with (
        patch.object(
            SessionLoader,
            "list_sessions",
            return_value=[
                {"session_id": "12345678-first"},
                {"session_id": "12345678-second"},
            ],
        ),
        pytest.raises(ValueError, match="Session ID is ambiguous"),
    ):
        _resolve_resume_session_id(ChartreuxConfigSchema(), "12345678")


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["12345678-first", "12345678"])
@pytest.mark.parametrize("already_attached", [False, True])
async def test_resume_canonicalizes_before_acquiring_existing_lease(
    tmp_path: Path, reference: str, already_attached: bool
) -> None:
    canonical = "12345678-first"
    config = ChartreuxConfigSchema(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    process = HarnessProcess()
    try:
        with (
            SessionLease(tmp_path, canonical),
            patch.object(
                SessionLoader, "list_sessions", return_value=[{"session_id": canonical}]
            ),
            patch.object(
                process,
                "build_root_blueprint",
                new=AsyncMock(return_value=SimpleNamespace(config=config)),
            ),
            patch("chartreux.app_server._runtime._load_session") as load_session,
            pytest.raises(SessionBusyError) as error,
        ):
            if already_attached:
                await AgentRuntimeFactory().resume_root(
                    cast(AgentLoop, SimpleNamespace(config=config)), reference
                )
            else:
                await process.open_root(
                    RootOpenRequest(
                        options=SessionOptions(),
                        client_info=ClientInfo(name="test", version="0"),
                        session_id=reference,
                    )
                )
        assert error.value.session_id == canonical
        load_session.assert_not_called()
        assert not (tmp_path / "active" / "12345678.lock").exists()
    finally:
        await process.close()


def _accept_session_backend(backend: SessionBackend) -> None:
    pass


def _accept_session_backend_host(backend: SessionBackendHost) -> None:
    pass


def _protocol_members(protocol: type[object]) -> set[str]:
    return {name for name in protocol.__dict__ if not name.startswith("_")}
