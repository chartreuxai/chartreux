from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server.client import AppServerClient, AppServerConnectionClosed
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
        self.reconnect_calls = 0

    async def reconnect(self, failed_client: _EofClient) -> bool:
        assert failed_client is self.current
        self.reconnect_calls += 1
        return False


def _session_for_close(client: _EofClient) -> AppServerSession:
    session = object.__new__(AppServerSession)
    session._connection = _Connection(client)  # type: ignore[assignment]
    session._closing = False
    session._events = asyncio.Queue()
    session._unsolicited_events = asyncio.Queue()
    session._event_generation = 0
    return session


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
