from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server.client import AppServerClient, AppServerConnectionClosed
from chartreux.app_server.connection import AppServerConnection
from chartreux.app_server.session import AppServerSession, _StreamClosed
from chartreux.app_server.transport import memory_transport_pair


class _AbruptTransport:
    async def send(self, message: dict[str, Any]) -> None:
        del message

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        raise OSError("connection reset")
        yield {}

    async def close(self) -> None:
        pass


class _EofClient:
    async def incoming(self) -> AsyncIterator[object]:
        if False:
            yield None


class _Connection:
    def __init__(self, client: _EofClient) -> None:
        self.current = client
        self.attached = True
        self.reconnect_calls = 0

    async def reconnect(self, failed_client: _EofClient) -> bool:
        assert failed_client is self.current
        self.reconnect_calls += 1
        return False


def _session_for_close(client: _EofClient) -> AppServerSession:
    session = object.__new__(AppServerSession)
    session._connection = _Connection(client)  # type: ignore[assignment]
    session._closing = False
    session._message_task = None
    session._reconnect_sleep = asyncio.sleep
    session._reconnect_backoff_seconds = 0.25
    session._pump_stop = asyncio.Event()
    session._events = asyncio.Queue()
    session._unsolicited_events = asyncio.Queue()
    session._event_generation = 0
    return session


class _BlockingCloseClient:
    def __init__(self, *, block_close: bool = False) -> None:
        self.block_close = block_close
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.closed = False

    async def close(self) -> None:
        if self.block_close:
            self.close_started.set()
            await self.release_close.wait()
        self.closed = True


@pytest.mark.asyncio
async def test_close_during_reconnect_does_not_install_replacement() -> None:
    failed_client = _BlockingCloseClient(block_close=True)
    replacement = _BlockingCloseClient()
    factory_calls = 0

    def client_factory() -> AppServerClient:
        nonlocal factory_calls
        factory_calls += 1
        return cast(AppServerClient, replacement)

    connection = cast(AppServerConnection, object.__new__(AppServerConnection))
    connection._client = cast(AppServerClient, failed_client)
    connection._client_factory = client_factory
    connection._lock = asyncio.Lock()
    connection._closed = False

    reconnect = asyncio.create_task(
        connection.reconnect(cast(AppServerClient, failed_client))
    )
    await asyncio.wait_for(failed_client.close_started.wait(), timeout=1)
    close = asyncio.create_task(connection.close())
    await asyncio.sleep(0)
    assert connection.current is None

    failed_client.release_close.set()
    assert await asyncio.wait_for(reconnect, timeout=1) is False
    await asyncio.wait_for(close, timeout=1)

    assert factory_calls == 0
    assert connection.current is None
    assert not replacement.closed


class _BlockingIncomingClient(_EofClient):
    def __init__(self) -> None:
        self.incoming_started = asyncio.Event()
        self.release_incoming = asyncio.Event()

    async def incoming(self) -> AsyncIterator[object]:
        self.incoming_started.set()
        await self.release_incoming.wait()
        if False:
            yield None


@pytest.mark.asyncio
async def test_close_during_attach_closes_streams_without_receiving_on_new_client() -> (
    None
):
    initial_client = _EofClient()

    class ReconnectingConnection(_Connection):
        async def reconnect(self, failed_client: _EofClient) -> bool:
            assert failed_client is self.current
            self.reconnect_calls += 1
            return True

    session = _session_for_close(initial_client)
    session._connection = ReconnectingConnection(initial_client)  # type: ignore[assignment]
    attach_started = asyncio.Event()
    release_attach = asyncio.Event()
    attached_client = _BlockingIncomingClient()

    async def attach(*, attach_session: bool = True):
        assert attach_session
        attach_started.set()
        await release_attach.wait()
        return cast(AppServerClient, attached_client)

    session._attach_with_backoff = attach  # type: ignore[method-assign]
    pump = asyncio.create_task(session._pump_messages())
    try:
        await asyncio.wait_for(attach_started.wait(), timeout=1)
        session.begin_close()
        release_attach.set()
        await asyncio.wait_for(pump, timeout=1)

        assert not attached_client.incoming_started.is_set()
        assert isinstance(session._events.get_nowait(), _StreamClosed)
        assert isinstance(session._unsolicited_events.get_nowait(), _StreamClosed)
    finally:
        release_attach.set()
        attached_client.release_incoming.set()
        if not pump.done():
            pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


@pytest.mark.asyncio
async def test_normal_eof_ends_session_events_without_traceback() -> None:
    session = _session_for_close(_EofClient())
    session.begin_close()
    session._ensure_attached = AsyncMock()  # type: ignore[method-assign]

    await session._pump_messages()

    assert [event async for event in session.events()] == []


@pytest.mark.asyncio
async def test_abrupt_disconnect_is_reported_as_connection_closed() -> None:
    client = AppServerClient(_AbruptTransport())
    await client.start()

    with pytest.raises(AppServerConnectionClosed, match="connection failed"):
        _ = [message async for message in client.incoming()]

    await client.close()


@pytest.mark.asyncio
async def test_pending_requests_complete_or_cancel_cleanly() -> None:
    client_transport, server_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    pending = asyncio.create_task(client.request("pending"))
    await asyncio.sleep(0)
    await server_transport.close()

    with pytest.raises(AppServerConnectionClosed):
        await pending
    await client.close()

    cancellable_transport, cancellable_server = memory_transport_pair()
    cancellable_client = AppServerClient(cancellable_transport)
    cancelled = asyncio.create_task(cancellable_client.request("cancelled"))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    await cancellable_client.close()
    await cancellable_server.close()
    assert cancellable_client._reader_task is not None
    assert cancellable_client._reader_task.done()


@pytest.mark.asyncio
async def test_intentional_close_ends_event_stream_without_reconnecting() -> None:
    session = _session_for_close(_EofClient())
    session.begin_close()
    session._ensure_attached = AsyncMock()  # type: ignore[method-assign]

    await session._pump_messages()

    assert cast("_Connection", session._connection).reconnect_calls == 0
    assert [event async for event in session.events()] == []
    assert isinstance(session._events.get_nowait(), _StreamClosed)


@pytest.mark.asyncio
async def test_unexpected_errors_remain_observable() -> None:
    session = _session_for_close(_EofClient())
    session._ensure_attached = AsyncMock()  # type: ignore[method-assign]
    session._unsolicited_events.put_nowait(
        _StreamClosed(RuntimeError("programming bug"))
    )

    with pytest.raises(RuntimeError, match="programming bug"):
        _ = [event async for event in session.events()]


def _connection_for_reconnect(
    failed_client: AppServerClient, client_factory: Callable[[], AppServerClient]
) -> AppServerConnection:
    connection = cast(AppServerConnection, object.__new__(AppServerConnection))
    connection._client = failed_client
    connection._client_factory = client_factory
    connection._presentation_handler = None
    connection._attached_session_id = "attached"
    connection._lock = asyncio.Lock()
    connection._closed = False
    return connection


@pytest.mark.asyncio
async def test_refusing_client_factory_ends_reconnect_as_false() -> None:
    """The stdio route's factory raises by design (terminal disconnect); the
    reconnect contract must observe that as False, never as an escape.
    """
    failed_client = _BlockingCloseClient()
    factory_calls = 0

    def client_factory() -> AppServerClient:
        nonlocal factory_calls
        factory_calls += 1
        raise RuntimeError("This app-server transport cannot reconnect")

    connection = _connection_for_reconnect(
        cast(AppServerClient, failed_client), client_factory
    )

    assert await connection.reconnect(cast(AppServerClient, failed_client)) is False

    assert factory_calls == 1
    assert connection.current is None
    assert failed_client.closed


@pytest.mark.asyncio
async def test_refusing_reconnect_takes_the_typed_close_path_via_pump() -> None:
    """With the refusal observed as False, the pump takes the designed typed
    close: both event streams settle with AppServerConnectionClosed instead of
    the pump task faulting unretrieved.
    """
    failed_client = _EofClient()

    def client_factory() -> AppServerClient:
        raise RuntimeError("This app-server transport cannot reconnect")

    connection = _connection_for_reconnect(
        cast(AppServerClient, failed_client), client_factory
    )
    session = _session_for_close(failed_client)
    session._connection = connection  # type: ignore[assignment]
    session._connection_lost = False
    session._connection_listeners = []
    session._reconnect_backoff_seconds = 0.01

    pump = asyncio.create_task(session._pump_messages())
    try:
        await asyncio.wait_for(pump, timeout=2)

        for queue in (session._events, session._unsolicited_events):
            item = queue.get_nowait()
            assert isinstance(item, _StreamClosed)
            assert isinstance(item.error, AppServerConnectionClosed)
            assert "App-server connection closed" in str(item.error)
        assert connection.current is None
    finally:
        if not pump.done():
            pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


@pytest.mark.asyncio
async def test_replacement_client_factory_reconnects_successfully() -> None:
    """A factory returning a fresh client (the in-process route) still
    reconnects: True installs the replacement and invalidates the attachment.
    """
    failed_client = _BlockingCloseClient()
    replacement = _BlockingCloseClient()

    def client_factory() -> AppServerClient:
        return cast(AppServerClient, replacement)

    connection = _connection_for_reconnect(
        cast(AppServerClient, failed_client), client_factory
    )

    assert await connection.reconnect(cast(AppServerClient, failed_client)) is True

    assert connection.current is replacement
    assert connection.attached is False
    assert failed_client.closed
    assert not replacement.closed
