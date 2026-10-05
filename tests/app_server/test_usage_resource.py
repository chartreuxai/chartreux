from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from chartreux.app_server.client import AppServerClient
from chartreux.app_server.client_state import ClientBootstrap, ClientSessionState
from chartreux.app_server.models import (
    UsageCoverageWarning,
    UsageWindowSummaries,
    UsageWindowSummary,
)
from chartreux.app_server.protocol import (
    ClientCapabilities,
    ClientInfo,
    Notification,
    SessionOptions,
    UsageReadParams,
    UsageReadResponse,
    UsageUpdatedParams,
)
from chartreux.app_server.session import AppServerSession


def _response(revision: int, project_key: str | None = None) -> UsageReadResponse:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    summary = UsageWindowSummary(
        start_local=now,
        end_local=now + timedelta(days=1),
        start_utc=now,
        end_utc=now + timedelta(days=1),
        timezone="UTC",
        requests=revision,
    )
    return UsageReadResponse(
        as_of=now,
        revision=revision,
        summaries=UsageWindowSummaries(day=summary, week=summary, month=summary),
        project_key=project_key,
    )


def _notification(revision: int) -> Notification:
    response = _response(revision)
    params = UsageUpdatedParams(
        as_of=response.as_of, revision=revision, summaries=response.summaries
    )
    return Notification(
        method="usage/updated", params=params.model_dump(mode="json", by_alias=True)
    )


def _session(monkeypatch: pytest.MonkeyPatch, client: Mock) -> AppServerSession:
    state = Mock(spec=ClientSessionState)
    state.session_id = "unattached"
    monkeypatch.setattr(
        "chartreux.app_server.session.ClientSessionState", lambda _: state
    )
    return AppServerSession(
        cast(AppServerClient, client),
        cast(ClientBootstrap, Mock()),
        ClientInfo(name="usage-test", version="1"),
        ClientCapabilities(),
        SessionOptions(),
    )


def _client(revision: int) -> Mock:
    client = Mock(spec=AppServerClient)
    client.request = AsyncMock(
        return_value=_response(revision).model_dump(mode="json", by_alias=True)
    )
    return client


@pytest.mark.asyncio
async def test_usage_resource_forwards_degraded_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(1)
    session = _session(monkeypatch, client)
    session.resources.usage._connection._connect_host = AsyncMock(return_value=client)
    response = _response(1)
    response.warnings = [UsageCoverageWarning(code="write-failed")]
    response.summaries.day.degraded = True
    client.request.return_value = response.model_dump(mode="json", by_alias=True)
    await session.resources.usage.read()
    current = session.resources.usage.current
    assert current is not None and current.degraded
    assert current.summaries.day.degraded
    params = current.model_copy(update={"revision": 2})
    await session._handle_notification(
        cast(AppServerClient, client),
        Notification(method="usage/updated", params=params.model_dump(mode="json")),
    )
    assert session.resources.usage.current == params


@pytest.mark.asyncio
async def test_usage_resource_notifications_before_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(0)
    session = _session(monkeypatch, client)
    callback = Mock()
    unsubscribe = session.resources.usage.subscribe(callback)
    for revision in (2, 2, 1, 3):
        await session._handle_notification(
            cast(AppServerClient, client), _notification(revision)
        )
    assert session.resources.usage.revision == 3
    assert session.resources.usage.current == UsageUpdatedParams.model_validate(
        _notification(3).params
    )
    assert callback.call_count == 2
    assert not session._connection.attached
    client.request.assert_not_awaited()
    unsubscribe()
    unsubscribe()
    await session._handle_notification(cast(AppServerClient, client), _notification(4))
    assert callback.call_count == 2
    await session._handle_notification(
        cast(AppServerClient, client), Notification(method="usage/updated", params={})
    )
    assert session.resources.usage.revision == 4


@pytest.mark.asyncio
async def test_usage_resource_read_before_attachment_and_during_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(2)
    session = _session(monkeypatch, client)

    # Keep the pump alive without requiring a fake transport.
    async def incoming():
        await session._pump_stop.wait()
        if False:
            yield

    client.incoming = incoming
    try:
        response = await session.resources.usage.read()
        assert response.revision == 2
        client.request.assert_awaited_once_with("usage/read", UsageReadParams())
        assert not session._connection.attached
        session._starting_turn = True
        filtered = _response(3, "git:root")
        client.request.return_value = filtered.model_dump(mode="json", by_alias=True)
        assert await session.resources.usage.read("month", "git:root") == filtered
        assert session.resources.usage.project_key == "git:root"
        assert session.resources.usage.revision == 2
        assert session._starting_turn
        client.request.assert_awaited_with(
            "usage/read", UsageReadParams(window="month", project_key="git:root")
        )
    finally:
        session.begin_close()
        if session._message_task is not None:
            await session._message_task


@pytest.mark.asyncio
async def test_usage_resource_reconnect_reads_fresh_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _client(10)
    second = _client(1)
    session = _session(monkeypatch, first)
    session._connection._client_factory = lambda: cast(AppServerClient, second)
    session._reconnect_sleep = AsyncMock()
    refreshed = asyncio.Event()
    session.resources.usage.subscribe(lambda _: refreshed.set())
    await session.resources.consume_notification(_notification(10))
    refreshed.clear()

    async def disconnected():
        if False:
            yield

    async def incoming():
        await session._pump_stop.wait()
        if False:
            yield

    first.incoming = disconnected
    second.incoming = incoming
    try:
        await session._ensure_host_connected()
        await asyncio.wait_for(refreshed.wait(), timeout=2)
        second.request.assert_awaited_once_with("usage/read", UsageReadParams())
        assert session.resources.usage.revision == 1
        assert not session._connection.attached
    finally:
        session.begin_close()
        if session._message_task is not None:
            await session._message_task


@pytest.mark.asyncio
async def test_usage_resource_read_does_not_overwrite_newer_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(1)
    session = _session(monkeypatch, client)
    session._ensure_host_connected = AsyncMock(return_value=client)
    session.resources.usage._connection._connect_host = session._ensure_host_connected
    await session.resources.consume_notification(_notification(5))
    response = await session.resources.usage.read()
    assert response.revision == 1
    assert session.resources.usage.revision == 5


@pytest.mark.asyncio
async def test_usage_resource_discards_read_from_previous_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(1)
    session = _session(monkeypatch, client)
    session.resources.usage._connection._connect_host = AsyncMock(return_value=client)
    started = asyncio.Event()
    release = asyncio.Event()

    async def request(method, params):
        assert method == "usage/read"
        if not started.is_set():
            started.set()
            await release.wait()
            response = _response(99, "old-project")
        else:
            response = _response(1, "new-project")
        return response.model_dump(mode="json", by_alias=True)

    client.request.side_effect = request
    old_read = asyncio.create_task(session.resources.usage.read())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        await session.resources.usage.refresh_after_reconnect()
        release.set()
        assert (await old_read).revision == 99
        assert session.resources.usage.revision == 1
        assert session.resources.usage.project_key == "new-project"
    finally:
        release.set()
        await old_read
