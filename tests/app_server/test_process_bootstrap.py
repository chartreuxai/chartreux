from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from io import BytesIO
import json
from pathlib import Path
import sys
from threading import Event
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server import stdio
from chartreux.app_server._runtime import HarnessServer
from chartreux.app_server.process_bootstrap import (
    BOOTSTRAP_ENV_VAR,
    PROCESS_PROTOCOL_VERSION,
    ProtocolVersionMismatchError,
    ReadyFrameMissingError,
    ReadyTimeoutError,
    bootstrap_requested,
    encode_ready_frame,
    parse_ready_frame,
    spawn_process,
    write_ready_frame,
)
from chartreux.app_server.transport import (
    ProcessPipeJsonRpcTransport,
    memory_transport_pair,
)

_EXCHANGE_FIXTURE = (
    Path(__file__).parents[1] / "fixtures" / "process_bootstrap" / "exchange.json"
)


class _BlockingWriter:
    def __init__(self) -> None:
        self.output = BytesIO()
        self.started = Event()
        self.release = Event()

    def write(self, data: bytes) -> int:
        self.started.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test writer was not released")
        return self.output.write(data)

    def flush(self) -> None:
        self.output.flush()


def _child_script(
    body: str, *, protocol_version: int = PROCESS_PROTOCOL_VERSION
) -> list[str]:
    ready_literal = encode_ready_frame(protocol_version).decode().strip()
    announcement = f"sys.stdout.write({ready_literal!r} + '\\n')\nsys.stdout.flush()\n"
    return [sys.executable, "-c", "import sys\n" + announcement + body]


def _echo_child(protocol_version: int = PROCESS_PROTOCOL_VERSION) -> list[str]:
    return _child_script(
        "for line in sys.stdin:\n    sys.stdout.write(line)\n    sys.stdout.flush()\n",
        protocol_version=protocol_version,
    )


def _sink_child(sink: Path) -> list[str]:
    return _child_script(
        f"with open({str(sink)!r}, 'w', encoding='utf-8') as handle:\n"
        "    for line in sys.stdin:\n"
        "        handle.write(line)\n"
        "        handle.flush()\n"
    )


async def _read_messages(
    stream: AsyncIterator[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    received: list[dict[str, Any]] = []
    async for message in stream:
        received.append(message)
        if len(received) == count:
            break
    return received


@pytest.mark.asyncio
async def test_spawn_process_receives_ready_frame_and_carries_json_rpc() -> None:
    spawned = await spawn_process(_echo_child(), ready_timeout=5)
    try:
        assert spawned.ready.protocol_version == PROCESS_PROTOCOL_VERSION
        await spawned.transport.send({"jsonrpc": "2.0", "id": "one", "method": "ping"})
        received = await _read_messages(spawned.transport.messages(), 1)
        assert received == [{"jsonrpc": "2.0", "id": "one", "method": "ping"}]
    finally:
        await spawned.transport.close()
    assert spawned.process.wait(timeout=5) == 0


@pytest.mark.asyncio
async def test_spawn_process_rejects_protocol_version_mismatch() -> None:
    child_version = PROCESS_PROTOCOL_VERSION + 1

    with pytest.raises(ProtocolVersionMismatchError) as caught:
        await spawn_process(_echo_child(child_version), ready_timeout=5)

    assert caught.value.expected == PROCESS_PROTOCOL_VERSION
    assert caught.value.received == child_version
    assert "version mismatch" in str(caught.value)


@pytest.mark.asyncio
async def test_spawn_process_reports_eof_before_ready() -> None:
    with pytest.raises(ReadyFrameMissingError, match="closed its pipe"):
        await spawn_process([sys.executable, "-c", "pass"], ready_timeout=5)


@pytest.mark.asyncio
async def test_spawn_process_times_out_when_ready_never_arrives() -> None:
    started = time.monotonic()

    with pytest.raises(ReadyTimeoutError, match="announced no readiness"):
        await spawn_process(
            [sys.executable, "-c", "import time; time.sleep(1)"], ready_timeout=0.1
        )

    assert time.monotonic() - started < 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_frame", "error"),
    [
        (json.dumps({"jsonrpc": "2.0", "method": "boot"}), "did not announce"),
        ("not-json", "malformed frame"),
    ],
)
async def test_spawn_process_rejects_a_first_frame_that_is_not_ready(
    first_frame: str, error: str
) -> None:
    script = (
        f"import sys\nsys.stdout.write({first_frame!r} + '\\n')\nsys.stdout.flush()\n"
    )

    with pytest.raises(ReadyFrameMissingError, match=error):
        await spawn_process([sys.executable, "-c", script], ready_timeout=5)


@pytest.mark.asyncio
async def test_process_pipe_close_drains_pending_messages_before_eof(
    tmp_path: Path,
) -> None:
    sink = tmp_path / "received.ndjson"
    spawned = await spawn_process(_sink_child(sink), ready_timeout=5)
    sent = [
        {"jsonrpc": "2.0", "method": "presentation/delta", "params": {"seq": 1}},
        {"jsonrpc": "2.0", "method": "presentation/delta", "params": {"seq": 2}},
        {"jsonrpc": "2.0", "method": "presentation/delta", "params": {"seq": 3}},
    ]
    try:
        for message in sent:
            await spawned.transport.send(message)
    finally:
        await spawned.transport.close()

    assert spawned.process.wait(timeout=5) == 0
    received = [json.loads(line) for line in sink.read_bytes().splitlines()]
    assert received == sent


@pytest.mark.asyncio
async def test_process_pipe_transport_preserves_draining_close_without_process() -> (
    None
):
    writer = _BlockingWriter()
    transport = ProcessPipeJsonRpcTransport(BytesIO(), writer, outbox_size=1)

    await transport.send({"id": 1})
    assert await asyncio.to_thread(writer.started.wait, 2)
    await transport.send({"id": 2})
    pending = asyncio.create_task(transport.send({"id": 3}))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not pending.done()

    writer.release.set()
    await asyncio.wait_for(pending, timeout=2)
    await asyncio.wait_for(transport.close(), timeout=2)

    messages = [json.loads(line) for line in writer.output.getvalue().splitlines()]
    assert messages == [{"id": 1}, {"id": 2}, {"id": 3}]


def test_process_pipe_transport_rejects_invalid_terminate_grace() -> None:
    with pytest.raises(ValueError, match="terminate_grace"):
        ProcessPipeJsonRpcTransport(BytesIO(), BytesIO(), terminate_grace=0)


def test_write_ready_frame_emits_exactly_one_json_line() -> None:
    writer = BytesIO()

    write_ready_frame(writer)

    lines = writer.getvalue().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "chartreuxBootstrap": "ready",
        "protocolVersion": PROCESS_PROTOCOL_VERSION,
    }


@pytest.mark.parametrize(
    "raw",
    [
        b'{"jsonrpc":"2.0","method":"boot"}',
        b'{"chartreuxBootstrap":"ready","protocolVersion":"1"}',
        b'{"chartreuxBootstrap":"ready"}',
    ],
)
def test_parse_ready_frame_rejects_frames_that_are_not_ready(raw: bytes) -> None:
    with pytest.raises(ReadyFrameMissingError):
        parse_ready_frame(raw)


def test_bootstrap_requested_follows_the_env_marker() -> None:
    assert bootstrap_requested({BOOTSTRAP_ENV_VAR: "1"})
    assert not bootstrap_requested({})
    assert not bootstrap_requested({BOOTSTRAP_ENV_VAR: "0"})


@pytest.mark.asyncio
@pytest.mark.parametrize("announce_ready", [True, False])
async def test_serve_stdio_announces_readiness_only_when_spawned(
    monkeypatch: pytest.MonkeyPatch, announce_ready: bool
) -> None:
    harness = AsyncMock(spec=HarnessServer)
    monkeypatch.setattr(stdio, "create_harness_server", AsyncMock(return_value=harness))
    output = BytesIO()

    await stdio.serve_stdio(
        reader=BytesIO(), writer=output, announce_ready=announce_ready
    )

    assert output.getvalue() == (encode_ready_frame() if announce_ready else b"")
    harness.serve.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_transport_exchange_matches_the_differential_fixture() -> None:
    exchange = json.loads(_EXCHANGE_FIXTURE.read_text())
    messages = exchange["messages"]
    assert exchange["protocolVersion"] == PROCESS_PROTOCOL_VERSION

    in_process, server_side = memory_transport_pair()
    try:
        for message in messages:
            await in_process.send(message)
        in_process_received = await _read_messages(
            server_side.messages(), len(messages)
        )
    finally:
        await in_process.close()
        await server_side.close()

    spawned = await spawn_process(_echo_child(), ready_timeout=5)
    try:
        for message in messages:
            await spawned.transport.send(message)
        process_received = await _read_messages(
            spawned.transport.messages(), len(messages)
        )
    finally:
        await spawned.transport.close()

    assert in_process_received == messages
    assert process_received == messages
