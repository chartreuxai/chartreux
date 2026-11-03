"""Client-side manual-compaction timeout and failure-state adoption.

The timeout tests drive ``AppServerClient.request`` over an in-memory transport
pair, so no frame leaves the process. The compaction tests attach a real
app-server session and answer ``session/compact`` locally: the summarization is
the one step that would otherwise reach a provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import math
from typing import Any

import pytest

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server._session_resources import _COMPACT_REQUEST_TIMEOUT_SECONDS
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ProtocolError,
    ProtocolErrorCode,
    SessionCompactResponse,
)
from chartreux.app_server.transport import memory_transport_pair
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import attach_test_app_server_session, start_test_app_server

_UNUSABLE_TIMEOUTS = (0.0, -1.0, math.nan, math.inf, -math.inf)


class _RecordingTransport:
    """A transport that records frames instead of delivering them."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        await asyncio.Event().wait()
        yield {}

    async def close(self) -> None:
        pass


def _stub_compact(
    monkeypatch: pytest.MonkeyPatch,
    client: AppServerClient,
    outcome: dict[str, Any] | AppServerResponseError,
) -> list[tuple[str, dict[str, Any]]]:
    """Answer ``session/compact`` locally and record every request call."""
    calls: list[tuple[str, dict[str, Any]]] = []
    original = client.request

    async def request(
        method: str, params: ProtocolModel | dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        calls.append((method, kwargs))
        if method != "session/compact":
            return await original(method, params, **kwargs)
        if isinstance(outcome, AppServerResponseError):
            raise outcome
        return outcome

    monkeypatch.setattr(client, "request", request)
    return calls


def _kwargs_for(calls: list[tuple[str, dict[str, Any]]], method: str) -> dict[str, Any]:
    return next(kwargs for called, kwargs in calls if called == method)


def _compaction_failure(code: ProtocolErrorCode, data: Any) -> AppServerResponseError:
    return AppServerResponseError(
        ProtocolError(
            code=code,
            message="The conversation is compacted; reconcile the session state",
            data=data,
        )
    )


@pytest.mark.parametrize("value", _UNUSABLE_TIMEOUTS)
def test_constructor_rejects_unusable_default_timeouts(value: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        AppServerClient(_RecordingTransport(), request_timeout=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", _UNUSABLE_TIMEOUTS)
async def test_unusable_override_is_rejected_before_the_frame_is_sent(
    value: float,
) -> None:
    transport = _RecordingTransport()
    client = AppServerClient(transport)
    try:
        with pytest.raises(ValueError, match="finite positive"):
            await client.request("session/compact", timeout=value)
        assert transport.sent == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_request_without_override_uses_the_instance_default() -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport, request_timeout=0.02)
    try:
        with pytest.raises(TimeoutError, match="test/wait"):
            await client.request("test/wait")
    finally:
        await client.close()
        await peer_transport.close()


@pytest.mark.asyncio
async def test_override_extends_the_wait_beyond_the_instance_default() -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport, request_timeout=0.02)
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    try:
        pending = asyncio.create_task(client.request("test/slow", timeout=5.0))
        tasks.append(pending)
        frame = await anext(peer_transport.messages())
        # Longer than the instance default, far shorter than the override.
        await asyncio.sleep(0.1)
        assert not pending.done()
        await peer_transport.send({
            "jsonrpc": "2.0",
            "id": frame["id"],
            "result": {"ok": True},
        })
        assert await pending == {"ok": True}
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()
        await peer_transport.close()


@pytest.mark.asyncio
async def test_override_shortens_the_wait_below_the_instance_default() -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport, request_timeout=30.0)
    try:
        with pytest.raises(TimeoutError, match="test/quick"):
            await asyncio.wait_for(
                client.request("test/quick", timeout=0.02), timeout=5
            )
    finally:
        await client.close()
        await peer_transport.close()


@pytest.mark.asyncio
async def test_late_response_to_an_abandoned_request_is_dropped() -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport, request_timeout=30.0)
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    try:
        abandoned = asyncio.create_task(client.request("session/compact", timeout=0.02))
        tasks.append(abandoned)
        frame = await anext(peer_transport.messages())
        with pytest.raises(TimeoutError, match="session/compact"):
            await abandoned
        assert client._abandoned_request_ids == {frame["id"]}

        # The answer arrives after the client gave up: it must not surface as a
        # protocol error, and the connection must stay usable for the next call.
        follow_up = asyncio.create_task(client.request("session/compact", timeout=5.0))
        tasks.append(follow_up)
        await peer_transport.send({
            "jsonrpc": "2.0",
            "id": frame["id"],
            "result": {"summary": "late"},
        })
        follow_up_frame = await anext(peer_transport.messages())
        await peer_transport.send({
            "jsonrpc": "2.0",
            "id": follow_up_frame["id"],
            "result": {"summary": "ok"},
        })
        assert await follow_up == {"summary": "ok"}
        assert client._abandoned_request_ids == set()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()
        await peer_transport.close()


@pytest.mark.asyncio
async def test_compact_alone_carries_the_fixed_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = start_test_app_server(build_test_agent_loop())
    session = await attach_test_app_server_session(client)
    try:
        resource = session.resources.sessions
        log = await resource.read_log()
        response = SessionCompactResponse(
            summary="compacted", state=session.state, session_log=log
        )
        calls = _stub_compact(
            monkeypatch, client, response.model_dump(mode="json", by_alias=True)
        )

        assert await resource.compact() == "compacted"
        assert _COMPACT_REQUEST_TIMEOUT_SECONDS == 750.0
        assert _kwargs_for(calls, "session/compact")["timeout"] == 750.0

        await resource.read_log()
        assert "timeout" not in _kwargs_for(calls, "session/log/read")
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_persistence_failure_state_is_adopted_before_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = start_test_app_server(build_test_agent_loop())
    session = await attach_test_app_server_session(client)
    try:
        before = session.state
        settled = before.model_copy(
            update={
                "event_id": before.event_id + 9,
                "session": before.session.model_copy(update={"title": "Compacted"}),
            }
        )
        error = _compaction_failure(
            ProtocolErrorCode.COMPACTION_FAILED,
            {
                "reason": "persistence",
                "state": settled.model_dump(mode="json", by_alias=True),
            },
        )
        _stub_compact(monkeypatch, client, error)

        with pytest.raises(AppServerResponseError) as excinfo:
            await session.resources.sessions.compact()

        assert excinfo.value is error
        data = excinfo.value.error.data
        assert isinstance(data, dict)
        assert data["reason"] == "persistence"
        assert session.state.event_id == before.event_id + 9
        assert session.state.session.title == "Compacted"
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["malformed-state", "state-not-an-object", "missing-state", "other-code"]
)
async def test_unusable_failure_state_never_masks_the_error(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    client = start_test_app_server(build_test_agent_loop())
    session = await attach_test_app_server_session(client)
    try:
        before = session.state
        stated = before.model_copy(update={"event_id": before.event_id + 5}).model_dump(
            mode="json", by_alias=True
        )
        code = ProtocolErrorCode.COMPACTION_FAILED
        data: Any = {"reason": "persistence", "state": {"event_id": -1}}
        match case:
            case "state-not-an-object":
                data = {"reason": "persistence", "state": "not-a-state"}
            case "missing-state":
                data = {"reason": "persistence"}
            case "other-code":
                code = ProtocolErrorCode.CONFLICT
                data = {"reason": "persistence", "state": stated}
        error = _compaction_failure(code, data)
        _stub_compact(monkeypatch, client, error)

        with pytest.raises(AppServerResponseError) as excinfo:
            await session.resources.sessions.compact()

        assert excinfo.value is error
        assert session.state.event_id == before.event_id
    finally:
        await session.close()
