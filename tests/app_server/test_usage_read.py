from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._host import project_usage
from chartreux.app_server._runtime import HarnessProcess
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ClientInfo,
    Notification,
    ProtocolErrorCode,
    SessionStartParams,
    StatsReadParams,
    StatsReadResponse,
    UsageReadParams,
    UsageReadResponse,
    UsageUpdatedParams,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core import _usage_io
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.usage import UsageOutcome, UsagePrices, UsageRecord, UsageState
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import create_legacy_app_server


def record(record_id: str, project_key: str = "project:test") -> UsageRecord:
    return UsageRecord(
        record_id=record_id,
        occurred_at=datetime.now(UTC),
        root_session_id="ledger-root",
        session_id="ledger-root",
        agent_role="root",
        model="test-model",
        provider="test-provider",
        wire_name="test-wire",
        project_key=project_key,
        outcome=UsageOutcome.COMPLETED,
        usage_state=UsageState.COMPLETE,
        input_tokens=10,
        output_tokens=5,
        cached_input_tokens=2,
        prices_usd_per_million=UsagePrices(input=1, output=2, cached_input=0.5),
        known_cost_usd=0.000019,
        has_unknown_cost=False,
    )


@asynccontextmanager
async def connection(
    process: HarnessProcess, open_root: AsyncMock, *, disabled: bool = False
):
    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(
        server_transport,
        open_root=open_root,
        host_handler=process.host_handler,
        runtime_factory=process.runtime_factory,
    )
    client = AppServerClient(client_transport, run_peer=server.serve)
    try:
        await client.initialize(
            ClientInfo(name="usage-test", version="1"),
            ClientCapabilities(
                disabled_notifications=["usage/updated"] if disabled else []
            ),
        )
        await client.notify("initialized")
        yield client, server
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["write-failed", "unreadable"])
async def test_host_notification_distinguishes_degraded_from_empty(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    process = HarnessProcess()
    received: list[UsageUpdatedParams] = []
    unsubscribe = process.host_handler.subscribe_usage(received.append)
    try:
        await process.usage_service.wait_ready()
        empty = received[-1]
        assert not empty.degraded
        assert empty.summaries.day.requests == 0

        def fail(*args, **kwargs):
            raise PermissionError("private failure detail")

        if failure == "write-failed":
            monkeypatch.setattr(_usage_io, "durable_append", fail)
            await process.root_usage_writer("ledger-root").append(record("failed"))
        else:
            path = config_dir / "usage" / "ledger-root" / "usage.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            monkeypatch.setattr(_usage_io, "read_usage_file", fail)
        snapshot = await process.usage_service.reconcile()
        degraded = received[-1]
        assert degraded.revision > empty.revision
        assert degraded.degraded
        assert degraded.summaries.day.requests == empty.summaries.day.requests == 0
        assert degraded.summaries != empty.summaries
        for summary in (
            degraded.summaries.day,
            degraded.summaries.week,
            degraded.summaries.month,
        ):
            assert summary.degraded
            assert not summary.has_unknown_cost
        response = project_usage(snapshot)
        assert response.warnings[0].code == failure
        assert "private failure detail" not in degraded.model_dump_json()
    finally:
        unsubscribe()
        await process.close()


@pytest.mark.asyncio
async def test_pre_session_read_and_response_shape(config_dir: Path) -> None:
    process = HarnessProcess()
    open_root = AsyncMock(side_effect=AssertionError("usage must not open a session"))
    writer = process.root_usage_writer("ledger-root")
    await writer.append(record("one"))
    # A malformed ledger entry must remain visible as a coverage warning.
    path = config_dir / "usage" / "ledger-root" / "usage.jsonl"
    with path.open("a") as stream:
        stream.write("not-json\n")
    try:
        async with connection(process, open_root) as (client, server):
            response = UsageReadResponse.model_validate(
                await client.request("usage/read", UsageReadParams(window="month"))
            )
            assert server._root is None
            assert not server._connection_attached
            open_root.assert_not_awaited()
            assert response.project_key is None
            assert response.window == "month"
            assert response.revision > 0
            assert response.warnings[0].code == "malformed-record"
            for summary in (
                response.summaries.day,
                response.summaries.week,
                response.summaries.month,
            ):
                assert summary.requests == 1
                assert summary.start_utc <= response.as_of < summary.end_utc
                assert summary.start_local == summary.start_utc
                assert summary.timezone
                assert summary.currency == "USD"
            assert response.models[0].wire_name == "test-wire"
            assert response.components.uncached_input.tokens == 8
            filtered = UsageReadResponse.model_validate(
                await client.request(
                    "usage/read", UsageReadParams(project_key="different-project")
                )
            )
            assert filtered.summaries.day.requests == 0
            assert filtered.project_key is None
            assert not server._event_watermarks
    finally:
        await process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params", [{"window": "year"}, {"projectKey": []}, {"filter": "bad"}]
)
async def test_invalid_usage_params(config_dir: Path, params: dict) -> None:
    process = HarnessProcess()
    try:
        async with connection(process, AsyncMock()) as (client, server):
            with pytest.raises(AppServerResponseError) as error:
                await client.request("usage/read", params)
            assert error.value.error.code == ProtocolErrorCode.INVALID_PARAMS
            assert server._root is None
            await client.request("usage/read")
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_attached_and_active_turn_read_preserve_stats(
    config_dir: Path, tmp_path: Path
) -> None:
    from chartreux.core.llm_models import LLMChunk
    from tests.mock.utils import mock_llm_chunk
    from tests.stubs.fake_backend import FakeBackend

    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingBackend(FakeBackend):
        async def complete(self, **kwargs: Any) -> LLMChunk:
            entered.set()
            await release.wait()
            return mock_llm_chunk(content="answer")

        async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
            entered.set()
            await release.wait()
            yield mock_llm_chunk(content="answer")

    process = HarnessProcess()
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=False, generate_titles=False)
    )
    root = build_test_agent_loop(config=config, backend=BlockingBackend())
    root.accounting_owner = await process.create_accounting_owner(
        root.session_id, tmp_path
    )
    process._track_accounting_loop(root)
    try:
        async with connection(process, AsyncMock(return_value=root)) as (
            client,
            server,
        ):
            await client.request("session/start", SessionStartParams())
            before = await client.request(
                "stats/read", StatsReadParams(session_id=root.session_id)
            )
            StatsReadResponse.model_validate(before)
            response = UsageReadResponse.model_validate(
                await client.request("usage/read")
            )
            assert response.project_key == root.accounting_owner.project_key
            assert (
                await client.request(
                    "stats/read", StatsReadParams(session_id=root.session_id)
                )
                == before
            )
            await client.request(
                "turn/start",
                {
                    "sessionId": root.session_id,
                    "message": [{"type": "text", "text": "hello"}],
                },
            )
            await asyncio.wait_for(entered.wait(), timeout=2)
            watermarks = dict(server._event_watermarks)
            active = UsageReadResponse.model_validate(
                await client.request("usage/read")
            )
            assert active.project_key == response.project_key
            assert not release.is_set()
            assert server._event_watermarks == watermarks
            release.set()
    finally:
        await process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", [False, True])
async def test_host_notifications_unattached_coalesced_and_cleaned_up(
    config_dir: Path, disabled: bool
) -> None:
    process = HarnessProcess()
    await process.usage_service.wait_ready()
    published: list[int] = []
    unsubscribe = process.usage_service.subscribe(
        lambda snapshot: published.append(snapshot.revision)
    )
    try:
        async with connection(process, AsyncMock(), disabled=disabled) as (
            client,
            server,
        ):
            await client.request("usage/read")  # initialization barrier, no attachment
            received: list[UsageUpdatedParams] = []

            async def collect() -> None:
                async for message in client.incoming():
                    if (
                        isinstance(message, Notification)
                        and message.method == "usage/updated"
                    ):
                        received.append(
                            UsageUpdatedParams.model_validate(message.params)
                        )

            collector = asyncio.create_task(collect())
            writer = process.root_usage_writer("ledger-root")
            await writer.append(record("one"))
            await writer.append(record("two"))
            await asyncio.sleep(0.4)
            await client.request("usage/read")  # drain pending transport messages
            assert len(published) == 1
            assert len(received) == (0 if disabled else 1)
            if received:
                assert received[0].revision == published[0]
                assert received[0].summaries.day.requests == 2
                assert "eventId" not in received[0].model_dump(by_alias=True)
            assert not server._connection_attached
            assert server._root is None
            assert not server._event_watermarks
            collector.cancel()
            await asyncio.gather(collector, return_exceptions=True)
        # Only this test's observer remains after connection teardown.
        assert len(process.usage_service._callbacks) == 1
    finally:
        unsubscribe()
        await process.close()
