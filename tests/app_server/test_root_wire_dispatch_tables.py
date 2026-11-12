"""The two-sided wire foundation: closed tables, origins, version, settlement.

The per-direction dispatch tables are closed and directional (item 1 wire
policy): every landed emitter's method appears in its peer's receiving table, an
id-bearing frame naming an unlisted method is answered with a typed compat
rejection, and an id-less one is recorded as a visible warning while the session
keeps serving. Request ids carry an origin namespace so the two peers can never
collide, the ready frame's protocol version carries the extension, and a closed
pipe settles every request pending on either end with a typed transport error.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, cast

import pytest

from chartreux.app_server import agent_wire
from chartreux.app_server.agent_admission import OperationMarker
from chartreux.app_server.agent_wire import (
    AGENT_NOTICE_METHOD,
    AGENT_OP_ADMISSION_METHOD,
    AGENT_OP_CANCEL_METHOD,
    AGENT_ORIGIN,
    AGENT_RECEIVING_NOTIFICATIONS,
    AGENT_RECEIVING_REQUESTS,
    AGENT_TURN_ENDED_METHOD,
    AGENT_TURN_INTERRUPT_METHOD,
    AGENT_TURN_RELAY_METHOD,
    AGENT_TURN_START_METHOD,
    COORDINATOR_ORIGIN,
    COORDINATOR_RECEIVING_NOTIFICATIONS,
    COORDINATOR_RECEIVING_REQUESTS,
    HOST_SUBAGENT_ENDED_METHOD,
    HOST_SUBAGENT_EVENT_METHOD,
    WireTransportClosedError,
    agent_request_id,
    coordinator_request_id,
    request_id_origin,
)
from chartreux.app_server.process_bootstrap import (
    PROCESS_PROTOCOL_VERSION,
    ProtocolVersionMismatchError,
    ReadyFrame,
    SpawnedProcess,
    encode_ready_frame,
    parse_ready_frame,
    spawn_process,
)
from chartreux.app_server.root_agent import RootAgentServer
from chartreux.app_server.root_agent_process import (
    RootAgentExitedError,
    RootAgentProcess,
)
from chartreux.app_server.transport import JsonRpcTransport
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.events import SessionTitleUpdatedEvent

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_ROOT = _REPO_ROOT / "chartreux" / "app_server"

_AGENT_END_EMITTERS = frozenset({"_notify"})
_COORDINATOR_END_EMITTERS = frozenset({"_request", "_schedule_notification"})

_WIRE_METHOD_OF_CODE = "WireMethodNotSupportedError"

# The op-9 host-run frames answer a run the agent end opened, so their generic
# coverage payload names a run id rather than a turn or user-input request.
_HOST_RUN_NOTIFICATIONS = frozenset({
    HOST_SUBAGENT_ENDED_METHOD,
    HOST_SUBAGENT_EVENT_METHOD,
})


class _FramePipe:
    """A transport double: the test feeds frames and reads what the end sent."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._incoming: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            frame = await self._incoming.get()
            if frame is None:
                return
            yield frame

    def feed(self, frame: dict[str, Any]) -> None:
        self._incoming.put_nowait(frame)

    def close_pipe(self) -> None:
        self._incoming.put_nowait(None)

    async def close(self) -> None:
        self.closed = True
        self.close_pipe()


class _ProcessDouble:
    """A process handle whose liveness the test decides."""

    def __init__(self) -> None:
        self.exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


class _RefusingFactory:
    """A loop factory with no loop: every served request fails typed."""

    async def open(self, params: Mapping[str, Any]) -> Any:
        raise RuntimeError("this test owns no agent loop")

    async def close(self) -> None:
        pass


async def _until(predicate: Callable[[], object], *, timeout: float = 5.0) -> None:
    """Spin until a predicate's value is truthy; collection predicates are fine."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("the endpoint never reached the expected state")
        await asyncio.sleep(0.01)


@asynccontextmanager
async def _agent_endpoint() -> AsyncIterator[tuple[_FramePipe, RootAgentServer]]:
    pipe = _FramePipe()
    server = RootAgentServer(cast("JsonRpcTransport", pipe), factory=_RefusingFactory())
    serving = asyncio.create_task(server.serve())
    try:
        yield pipe, server
    finally:
        pipe.close_pipe()
        await asyncio.wait_for(serving, timeout=5)


def _coordinator_endpoint(pipe: _FramePipe) -> RootAgentProcess:
    spawned = SpawnedProcess(
        process=cast("subprocess.Popen[bytes]", _ProcessDouble()),
        transport=cast("Any", pipe),
        ready=ReadyFrame(protocol_version=PROCESS_PROTOCOL_VERSION),
    )
    return RootAgentProcess(spawned, heartbeat_interval=30.0)


def _replies(pipe: _FramePipe) -> list[dict[str, Any]]:
    return [frame for frame in pipe.sent if "id" in frame]


def _notifications(pipe: _FramePipe, method: str) -> list[dict[str, Any]]:
    return [frame for frame in pipe.sent if frame.get("method") == method]


def _named_methods(path: Path, emitters: frozenset[str]) -> set[str]:
    """The method strings one endpoint's source emits through its emitters."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    declared = vars(agent_wire)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in emitters and node.args:
                resolved = _resolved_method(node.args[0], declared)
                if resolved is not None:
                    found.add(resolved)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "method":
                    resolved = _resolved_method(value, declared)
                    if resolved is not None:
                        found.add(resolved)
    return found


def _resolved_method(node: ast.expr, declared: Mapping[str, Any]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        value = declared.get(node.id)
        return value if isinstance(value, str) else None
    return None


def test_the_two_directions_carry_disjoint_closed_tables() -> None:
    assert AGENT_RECEIVING_REQUESTS
    assert AGENT_RECEIVING_NOTIFICATIONS
    assert COORDINATOR_RECEIVING_NOTIFICATIONS
    # A method is one shape or the other: never both, in either direction.
    assert not AGENT_RECEIVING_REQUESTS & AGENT_RECEIVING_NOTIFICATIONS
    assert not COORDINATOR_RECEIVING_REQUESTS & COORDINATOR_RECEIVING_NOTIFICATIONS
    for table in (
        AGENT_RECEIVING_REQUESTS,
        AGENT_RECEIVING_NOTIFICATIONS,
        COORDINATOR_RECEIVING_REQUESTS,
        COORDINATOR_RECEIVING_NOTIFICATIONS,
    ):
        assert all(isinstance(method, str) for method in table)


def test_every_landed_emitter_names_a_method_its_peer_table_lists() -> None:
    agent_receives = AGENT_RECEIVING_REQUESTS | AGENT_RECEIVING_NOTIFICATIONS
    coordinator_receives = COORDINATOR_RECEIVING_REQUESTS | (
        COORDINATOR_RECEIVING_NOTIFICATIONS
    )
    coordinator_emits = _named_methods(
        _MODULE_ROOT / "root_agent_process.py", _COORDINATOR_END_EMITTERS
    ) | _named_methods(
        _MODULE_ROOT / "child_agent_process.py", _COORDINATOR_END_EMITTERS
    )
    agent_emits = _named_methods(
        _MODULE_ROOT / "root_agent.py", _AGENT_END_EMITTERS
    ) | _named_methods(_MODULE_ROOT / "child_agent.py", _AGENT_END_EMITTERS)

    assert agent_emits, "the agent-end scan found no emitter"
    assert coordinator_emits, "the coordinator-end scan found no emitter"
    assert agent_emits <= coordinator_receives
    assert coordinator_emits <= agent_receives


def test_request_ids_carry_their_origin_namespace() -> None:
    assert request_id_origin(coordinator_request_id(1)) == COORDINATOR_ORIGIN
    assert request_id_origin(agent_request_id(1)) == AGENT_ORIGIN
    assert coordinator_request_id(1) != agent_request_id(1)
    assert request_id_origin("1") is None
    assert request_id_origin(None) is None
    assert request_id_origin(7) is None


@pytest.mark.asyncio
async def test_an_unknown_request_is_rejected_typed_and_serving_continues() -> None:
    async with _agent_endpoint() as (pipe, server):
        pipe.feed({
            "jsonrpc": "2.0",
            "id": coordinator_request_id(99),
            "method": "agent/not-a-method",
        })
        await _until(lambda: _replies(pipe))

        reply = _replies(pipe)[0]
        assert reply["id"] == coordinator_request_id(99)
        assert reply["error"]["code"] == _WIRE_METHOD_OF_CODE
        assert "agent/not-a-method" in reply["error"]["message"]
        assert [record.method for record in server.diagnostics] == [
            "agent/not-a-method"
        ]

        # The session keeps serving: a later valid frame still reaches its handler.
        pipe.feed({
            "jsonrpc": "2.0",
            "method": AGENT_TURN_START_METHOD,
            "params": {"turn_id": "turn-1"},
        })
        await _until(lambda: _notifications(pipe, AGENT_TURN_ENDED_METHOD))
        ended = _notifications(pipe, AGENT_TURN_ENDED_METHOD)[0]
        assert ended["params"]["outcome"] == "failed"


@pytest.mark.asyncio
async def test_an_unknown_notification_is_a_visible_warning_not_a_failure() -> None:
    async with _agent_endpoint() as (pipe, server):
        pipe.feed({"jsonrpc": "2.0", "method": "agent/turn/not-a-notice"})
        await _until(lambda: server.diagnostics)

        diagnostic = server.diagnostics[0]
        assert diagnostic.method == "agent/turn/not-a-notice"
        assert diagnostic.direction == COORDINATOR_ORIGIN
        assert "the session continues" in diagnostic.message()
        assert not _replies(pipe)

        # A notification carries no reply correlation, so the loop just goes on.
        pipe.feed({
            "jsonrpc": "2.0",
            "method": AGENT_TURN_START_METHOD,
            "params": {"turn_id": "turn-1"},
        })
        await _until(lambda: _notifications(pipe, AGENT_TURN_ENDED_METHOD))


@pytest.mark.asyncio
async def test_every_agent_request_table_entry_reaches_its_handler() -> None:
    for method in sorted(AGENT_RECEIVING_REQUESTS):
        async with _agent_endpoint() as (pipe, server):
            pipe.feed({
                "jsonrpc": "2.0",
                "id": coordinator_request_id(1),
                "method": method,
            })
            await _until(lambda: _replies(pipe))

            reply = _replies(pipe)[0]
            assert reply["id"] == coordinator_request_id(1), method
            # The listed method reached its handler: only an unlisted one is
            # answered with the compat rejection.
            assert reply["error"]["code"] != _WIRE_METHOD_OF_CODE, method
            assert not server.diagnostics, method


@pytest.mark.asyncio
async def test_every_agent_notification_table_entry_reaches_its_handler() -> None:
    for method in sorted(AGENT_RECEIVING_NOTIFICATIONS):
        async with _agent_endpoint() as (pipe, server):
            if method in _HOST_RUN_NOTIFICATIONS:
                server.open_host_run("run-1")
                params: dict[str, Any] = {"run_id": "run-1"}
            elif method == AGENT_OP_CANCEL_METHOD:
                # An id-less cancel naming no live operation is an idempotent
                # no-op: the operation's own settlement is the only outcome.
                params = {"op_id": "incident-1:7"}
            else:
                params = (
                    {"turn_id": "turn-1"}
                    if method in {AGENT_TURN_START_METHOD, AGENT_TURN_INTERRUPT_METHOD}
                    else {"request_id": "request-1"}
                )
            pipe.feed({"jsonrpc": "2.0", "method": method, "params": params})
            await asyncio.sleep(0.05)

            assert not server.diagnostics, method
            assert not _replies(pipe), method


@pytest.mark.asyncio
async def test_an_agent_end_reply_settles_only_in_its_origin_namespace() -> None:
    async with _agent_endpoint() as (pipe, server):
        request_id, future = server.register_agent_request()
        assert request_id_origin(request_id) == AGENT_ORIGIN

        # A coordinator-origin id answers no agent-origin request.
        pipe.feed({"jsonrpc": "2.0", "id": coordinator_request_id(1), "result": {}})
        await _until(lambda: server.diagnostics)
        assert not future.done()

        pipe.feed({"jsonrpc": "2.0", "id": request_id, "result": {"ok": True}})
        await _until(future.done)
        assert future.result() == {"ok": True}


@pytest.mark.asyncio
async def test_an_unknown_notification_never_kills_the_coordinator_end() -> None:
    pipe = _FramePipe()
    process = _coordinator_endpoint(pipe)
    try:
        pipe.feed({"jsonrpc": "2.0", "method": "agent/turn/not-a-notice"})
        await _until(lambda: process.diagnostics)

        diagnostic = process.diagnostics[0]
        assert diagnostic.method == "agent/turn/not-a-notice"
        assert diagnostic.direction == AGENT_ORIGIN
        assert "the session continues" in diagnostic.message()
        assert process.alive
        assert process.death is None

        # The pump still settles requests after the unknown frame.
        claim = asyncio.create_task(process.keep_waiting_user_input("request-1"))
        await _until(lambda: pipe.sent)
        pipe.feed({
            "jsonrpc": "2.0",
            "id": pipe.sent[0]["id"],
            "result": {"outcome": "kept"},
        })
        assert await asyncio.wait_for(claim, timeout=5) == "kept"
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_the_coordinator_rejects_an_agent_origin_request_typed() -> None:
    pipe = _FramePipe()
    process = _coordinator_endpoint(pipe)
    try:
        pipe.feed({
            "jsonrpc": "2.0",
            "id": agent_request_id(1),
            "method": "host/tool-io",
        })
        await _until(lambda: pipe.sent)

        reply = pipe.sent[0]
        assert reply["id"] == agent_request_id(1)
        assert reply["error"]["code"] == _WIRE_METHOD_OF_CODE
        assert "host/tool-io" in reply["error"]["message"]
        assert process.alive
        assert process.death is None
        assert [record.method for record in process.diagnostics] == ["host/tool-io"]
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_a_late_reply_is_recorded_instead_of_dropped_silently() -> None:
    pipe = _FramePipe()
    process = _coordinator_endpoint(pipe)
    try:
        pipe.feed({"jsonrpc": "2.0", "id": coordinator_request_id(404), "result": {}})
        await _until(lambda: process.diagnostics)

        assert process.diagnostics[0].request_id == coordinator_request_id(404)
        assert process.alive
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_every_coordinator_notification_table_entry_reaches_its_handler() -> None:
    pipe = _FramePipe()
    process = _coordinator_endpoint(pipe)
    try:
        for method in sorted(COORDINATOR_RECEIVING_NOTIFICATIONS):
            params: dict[str, Any] = (
                # A turn-less notice carries an out-of-band event, not a turn.
                {
                    "event": agent_wire.encode_event(
                        SessionTitleUpdatedEvent(title="Titled", session_id="s")
                    )
                }
                if method == AGENT_NOTICE_METHOD
                # An admission frame carries the long-operation marker record.
                else agent_wire.encode_operation_marker(
                    OperationMarker(
                        request_id=coordinator_request_id(1),
                        op_id="inc-1:1",
                        method="agent/open",
                        state="active",
                        deadline=1.0,
                    )
                )
                if method == AGENT_OP_ADMISSION_METHOD
                else {"turn_id": "turn-unknown"}
            )
            pipe.feed({"jsonrpc": "2.0", "method": method, "params": params})
        await asyncio.sleep(0.05)

        assert not process.diagnostics
        assert process.alive
    finally:
        await process.close()


def test_the_protocol_version_carries_the_bidirectional_extension() -> None:
    # The origin namespaces and the closed tables are a wire change, so the
    # version moved past the one-frame turn wire.
    assert PROCESS_PROTOCOL_VERSION >= 2
    assert parse_ready_frame(encode_ready_frame()).protocol_version == (
        PROCESS_PROTOCOL_VERSION
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_previous_protocol_version_is_rejected_at_the_ready_frame() -> None:
    previous = PROCESS_PROTOCOL_VERSION - 1
    ready_literal = encode_ready_frame(previous).decode().strip()
    child = [
        sys.executable,
        "-c",
        "import sys\n"
        f"sys.stdout.write({ready_literal!r} + '\\n')\n"
        "sys.stdout.flush()\n",
    ]

    with pytest.raises(ProtocolVersionMismatchError) as caught:
        await spawn_process(child, ready_timeout=15)

    assert caught.value.expected == PROCESS_PROTOCOL_VERSION
    assert caught.value.received == previous


@pytest.mark.asyncio
async def test_a_closed_pipe_settles_the_agent_ends_pending_requests() -> None:
    pipe = _FramePipe()
    server = RootAgentServer(cast("JsonRpcTransport", pipe), factory=_RefusingFactory())
    first_id, first = server.register_agent_request()
    second_id, second = server.register_agent_request()

    serving = asyncio.create_task(server.serve())
    pipe.feed({"jsonrpc": "2.0", "method": AGENT_TURN_RELAY_METHOD})
    await asyncio.sleep(0)
    pipe.close_pipe()
    await asyncio.wait_for(serving, timeout=5)

    for request_id, future in ((first_id, first), (second_id, second)):
        assert future.done(), request_id
        with pytest.raises(WireTransportClosedError):
            future.result()


@pytest.mark.asyncio
async def test_a_closed_pipe_settles_the_coordinators_pending_work() -> None:
    pipe = _FramePipe()
    process = _coordinator_endpoint(pipe)
    request = asyncio.create_task(process.keep_waiting_user_input("request-1"))
    turn = process.run_turn(
        ExecutionTurnRequest(
            message="hello", turn_options=AgentTurnOptions(turn_id="turn-1")
        )
    )
    first_frame = asyncio.create_task(turn.__anext__())
    try:
        # Both the request and the turn are in flight before the pipe closes.
        await _until(lambda: len(pipe.sent) >= 2)
        pipe.close_pipe()

        with pytest.raises(RootAgentExitedError):
            await asyncio.wait_for(request, timeout=5)
        with pytest.raises(RootAgentExitedError):
            await asyncio.wait_for(first_frame, timeout=5)
        assert not process.alive
    finally:
        await turn.aclose()
        await process.close()
