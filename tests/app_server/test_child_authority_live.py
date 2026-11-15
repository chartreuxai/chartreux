"""G5b pins: the child side of the live child-authority contract, end to end.

G5a landed the owner half: a typed ``agent/authority/check`` question, the
strict codecs, the mandatory two-sided capability and an endpoint answered from
the owner's CURRENT policy. G5b lands the half that CONSUMES it:

* the relay (user-approved ownership amendment): a spawned child's only wire
  peer is the client that spawned it, so that client serves the child's
  question from the endpoint the owner bound into it, and the owner pushes one
  ``agent/authority/changed`` invalidation down every live child's pipe;
* the child-side client: one fenced question per invocation with the child's
  generation, a fresh per-run epoch, its last-known revision vector and an
  agent-origin request id, answered allow/deny only;
* the enforcement: an advisory ask before local resolution, the real gate at
  the execution handoff AFTER the file-snapshot await, and the catalog axis in
  the tool manager, so a process child's advertised catalog and its execution
  gate can never disagree;
* the parent incarnation that fences every question, carried on the spawn
  payload exactly as F6/F11 carry their keys.

Every pin is fail-closed: an unbound client, a dead pipe, a timeout, an
unreadable answer, a foreign/late/reordered reply and a rolled-back vector are
all refusals. The last pin is the acceptance discrimination: a REAL process
child held between its advisory precheck and its final admission, while the
owner tightens that same tool's policy, must DENY with no side effect. On the
pre-G5b tree the child holds no client at all, so it IGNORES the tightening and
the write lands -- which is the failure this package removes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.app_server._connection_protocol import ClientCapabilities, ClientInfo
from chartreux.app_server.agent_wire import (
    AGENT_AUTHORITY_ALLOW,
    AGENT_AUTHORITY_CHECK_METHOD,
    AGENT_AUTHORITY_DECISION_KIND_TOOL_AVAILABILITY,
    AGENT_AUTHORITY_DECISION_KIND_TOOL_INVOCATION,
    AGENT_AUTHORITY_DENY,
    AUTHORITY_REVISION_COMPONENTS,
    AgentAuthorityChanged,
    AgentAuthorityCheck,
    AgentAuthorityDecision,
    agent_request_id,
    decode_authority_check,
    encode_authority_changed,
    encode_authority_check,
    encode_authority_decision,
    implementation_identity,
)
from chartreux.app_server.child_agent_process import ProcessSubagentSpawn
from chartreux.app_server.root_agent import AgentChildHost, RootAgentServer
from chartreux.app_server.root_agent_process import RootAgentProcess
from chartreux.app_server.transport import (
    STDIO_TRANSPORT,
    TRANSPORT_ENV_VAR,
    memory_transport_pair,
)
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.models import ToolPermission
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.differential import egress
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CLIENT_INFO = ClientInfo(name="chartreux-tests", version="1.0")
_CLIENT_CAPABILITIES = ClientCapabilities(callback_kinds=["user_input"])


def _wire() -> Any:
    """The G5b additions to the wire contract, imported lazily.

    This module must still COLLECT on the pre-G5b tree: the acceptance
    discrimination pin runs there and has to behave differently, not fail to
    import. Everything G5b added is therefore reached through this accessor.
    """
    from chartreux.app_server import agent_wire

    return agent_wire


def _child_agent_module() -> Any:
    from chartreux.app_server import child_agent

    return child_agent


def _zero_revision() -> dict[str, int]:
    return {component: 0 for component in sorted(AUTHORITY_REVISION_COMPONENTS)}


def _check(**overrides: Any) -> AgentAuthorityCheck:
    values: dict[str, Any] = {
        "decision_kind": AGENT_AUTHORITY_DECISION_KIND_TOOL_INVOCATION,
        "tool_name": "write_file",
        "implementation": "chartreux.core.tools.builtins.write_file:WriteFile",
        "arguments": {"file_path": "/tmp/probe.txt", "content": "probe"},
        "child_agent_id": "coder-attempt-1",
        "child_generation": 1,
        "epoch": "epoch-a",
        "parent_session_id": "session-1",
        "parent_incarnation": "owner-1",
        "request_id": agent_request_id(1),
        "revision": _zero_revision(),
    }
    values.update(overrides)
    return AgentAuthorityCheck(**values)


# ---------------------------------------------------------------------------
# The relay: a child's question reaches the owner, and invalidation reaches
# the child.
# ---------------------------------------------------------------------------


class _RecordingWire:
    """The transport stand-in: every frame the relay sends lands here."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)


def _relay(responder: Any = None) -> tuple[Any, _RecordingWire]:
    """A real ChildAgentProcess relay over a recording transport (G5b shape)."""
    process = object.__new__(RootAgentProcess)
    process._child_authority_responder = responder
    process._host_tasks = set()
    process._notifications = set()
    process._diagnostics = []
    wire = _RecordingWire()
    process._transport = cast(Any, wire)
    return process, wire


async def _settle(process: Any) -> None:
    """Let every frame the relay scheduled finish, surfacing any failure."""
    for _ in range(4):
        await asyncio.sleep(0)
        pending = [*process._host_tasks, *process._notifications]
        if not pending:
            return
        await asyncio.gather(*pending)


def test_the_child_authority_frame_is_served_by_the_bound_endpoint() -> None:
    """The literal child frame routes to the owner's endpoint, not a refusal.

    This is the relay's reason to exist: before G5b the frame was answered
    ``WireMethodNotSupportedError`` because the spawned child's only peer
    dispatched on the coordinator-direction table. The endpoint is bound by the
    owner that published the child (``AgentChildHost._bind_child_authority``),
    and the child's answer is exactly what the endpoint returned.
    """
    seen: list[Mapping[str, Any]] = []

    def responder(params: Mapping[str, Any]) -> dict[str, Any]:
        seen.append(params)
        check = decode_authority_check(params)
        return encode_authority_decision(
            AgentAuthorityDecision(
                decision=AGENT_AUTHORITY_ALLOW,
                reason=None,
                epoch=check.epoch,
                parent_session_id=check.parent_session_id,
                parent_incarnation=check.parent_incarnation,
                child_generation=check.child_generation,
                request_id=check.request_id,
                revision=dict(check.revision),
            )
        )

    async def run() -> None:
        process, wire = _relay(responder)
        check = _check()
        process._dispatch({
            "jsonrpc": "2.0",
            "id": agent_request_id(1),
            "method": AGENT_AUTHORITY_CHECK_METHOD,
            "params": encode_authority_check(check),
        })
        await _settle(process)
        assert process._diagnostics == []
        assert seen == [encode_authority_check(check)]
        assert len(wire.sent) == 1
        frame = wire.sent[0]
        assert frame["id"] == agent_request_id(1)
        assert decode_authority_check(seen[0]) == check
        assert frame["result"]["decision"] == AGENT_AUTHORITY_ALLOW
        assert frame["result"]["request_id"] == check.request_id

    asyncio.run(run())


def test_the_relay_answers_typed_when_no_endpoint_is_bound() -> None:
    """A missing hop denies: no endpoint answers a typed error, never an allow."""

    async def run() -> None:
        process, wire = _relay(None)
        process._dispatch({
            "jsonrpc": "2.0",
            "id": agent_request_id(1),
            "method": AGENT_AUTHORITY_CHECK_METHOD,
            "params": encode_authority_check(_check()),
        })
        await _settle(process)
        assert len(wire.sent) == 1
        frame = wire.sent[0]
        assert "error" in frame and "result" not in frame
        assert "no bound authority endpoint" in frame["error"]["message"]

    asyncio.run(run())


def test_the_relay_refuses_a_request_id_outside_the_agent_namespace() -> None:
    """A foreign id can never settle a future this end never minted."""
    calls: list[Mapping[str, Any]] = []

    def responder(params: Mapping[str, Any]) -> dict[str, Any]:
        calls.append(params)
        return encode_authority_decision(
            AgentAuthorityDecision(
                decision=AGENT_AUTHORITY_ALLOW,
                reason=None,
                epoch="epoch-a",
                parent_session_id="session-1",
                parent_incarnation="owner-1",
                child_generation=1,
                request_id=agent_request_id(1),
                revision=_zero_revision(),
            )
        )

    async def run() -> None:
        process, wire = _relay(responder)
        process._dispatch({
            "jsonrpc": "2.0",
            "id": "coordinator-7",
            "method": AGENT_AUTHORITY_CHECK_METHOD,
            "params": encode_authority_check(_check()),
        })
        await _settle(process)
        assert calls == []
        assert len(wire.sent) == 1
        assert "error" in wire.sent[0]

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_the_owner_pushes_one_invalidation_to_every_live_child() -> None:
    """A moved vector reaches every live child; a dead child never stalls it."""
    delivered: list[dict[str, Any]] = []

    class _Child:
        def __init__(self, *, broken: bool = False) -> None:
            self.broken = broken

        async def notify_authority_changed(self, params: Mapping[str, Any]) -> None:
            if self.broken:
                raise RuntimeError("pipe gone")
            delivered.append(dict(params))

    children = [
        SimpleNamespace(spawned=SimpleNamespace(wire_client=_Child())),
        SimpleNamespace(spawned=SimpleNamespace(wire_client=_Child())),
        SimpleNamespace(spawned=SimpleNamespace(wire_client=_Child(broken=True))),
    ]
    server = object.__new__(RootAgentServer)
    server._child_host = cast(Any, SimpleNamespace(children=lambda: list(children)))

    params = encode_authority_changed(
        AgentAuthorityChanged(
            session_id="session-1",
            incarnation="owner-1",
            revision={"policy": 4, "registry": 1, "grants": 0, "ancestors": 0},
        )
    )
    await server._broadcast_authority_changed(params)
    assert delivered == [params, params]


# ---------------------------------------------------------------------------
# The child-side client: fencing, fail-closed defaults, cache invalidation.
# ---------------------------------------------------------------------------


class _ScriptedOwner:
    """A scripted answerer standing where the owner's pipe would be."""

    def __init__(
        self,
        *,
        verdicts: list[str] | None = None,
        deny_names: set[str] | None = None,
        revision: dict[str, int] | None = None,
        raises: BaseException | None = None,
        tamper: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.verdicts = list(verdicts or [])
        self.deny_names = set(deny_names or ())
        self.revision = revision
        self.raises = raises
        self.tamper = tamper
        self.checks: list[AgentAuthorityCheck] = []

    async def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == AGENT_AUTHORITY_CHECK_METHOD
        check = decode_authority_check(params)
        self.checks.append(check)
        if self.raises is not None:
            raise self.raises
        if check.tool_name in self.deny_names:
            verdict = AGENT_AUTHORITY_DENY
        elif self.verdicts:
            verdict = self.verdicts.pop(0)
        else:
            verdict = AGENT_AUTHORITY_ALLOW
        payload = encode_authority_decision(
            AgentAuthorityDecision(
                decision=verdict,
                reason=None if verdict == AGENT_AUTHORITY_ALLOW else "not on the axis",
                epoch=check.epoch,
                parent_session_id=check.parent_session_id,
                parent_incarnation=check.parent_incarnation,
                child_generation=check.child_generation,
                request_id=check.request_id,
                revision=dict(self.revision or check.revision),
            )
        )
        if self.tamper is not None:
            payload = self.tamper(payload)
        return payload


def _client(
    owner: Any, *, incarnation: str | None = "owner-1", timeout: float = 10.0
) -> Any:
    module = _child_agent_module()
    return module.ChildAuthorityClient(
        owner,
        agent_id="coder-attempt-1",
        session_id="session-1",
        parent_incarnation=incarnation,
        timeout=timeout,
    )


class _GatedOwner:
    """A scripted owner that parks the first catalog question until released.

    The park is the mid-pass window: while the pass is gated awaiting this
    one question, the test delivers the owner's invalidation frame.
    """

    def __init__(self) -> None:
        self.checks: list[AgentAuthorityCheck] = []
        self.deny_names: set[str] = set()
        self.arrived = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == AGENT_AUTHORITY_CHECK_METHOD
        check = decode_authority_check(params)
        self.checks.append(check)
        if check.decision_kind == AGENT_AUTHORITY_DECISION_KIND_TOOL_AVAILABILITY:
            self.arrived.set()
            await self.release.wait()
        verdict = (
            AGENT_AUTHORITY_DENY
            if check.tool_name in self.deny_names
            else AGENT_AUTHORITY_ALLOW
        )
        return encode_authority_decision(
            AgentAuthorityDecision(
                decision=verdict,
                reason=None if verdict == AGENT_AUTHORITY_ALLOW else "revoked",
                epoch=check.epoch,
                parent_session_id=check.parent_session_id,
                parent_incarnation=check.parent_incarnation,
                child_generation=check.child_generation,
                request_id=check.request_id,
                revision=dict(check.revision),
            )
        )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_an_allowed_answer_admits_and_a_denied_answer_refuses() -> None:
    """Only the owner's own allow admits; a deny keeps the owner's reason."""
    from chartreux.core.tools.builtins.write_file import WriteFile

    owner = _ScriptedOwner(verdicts=[AGENT_AUTHORITY_ALLOW, AGENT_AUTHORITY_DENY])
    client = _client(owner)
    arguments = {"file_path": "/tmp/probe.txt", "content": "probe"}
    assert (
        await client.denial(
            tool_name="write_file", tool_class=WriteFile, arguments=arguments
        )
        is None
    )
    refusal = await client.denial(
        tool_name="write_file", tool_class=WriteFile, arguments=arguments
    )
    assert refusal is not None and "not on the axis" in refusal

    # Every question names the child's registry-provenance implementation, not
    # a name alone, and carries the canonical arguments the call will run with.
    assert [check.implementation for check in owner.checks] == [
        implementation_identity(WriteFile)
    ] * 2
    assert owner.checks[0].arguments == arguments
    assert [check.request_id for check in owner.checks] == [
        agent_request_id(1),
        agent_request_id(2),
    ]
    assert client.refusal_count == 1


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_an_unbound_client_denies_every_question_and_withholds_the_catalog() -> (
    None
):
    """A child published without an incarnation asks nothing and runs nothing."""
    from chartreux.core.tools.builtins.write_file import WriteFile

    owner = _ScriptedOwner()
    client = _client(owner, incarnation=None)
    assert client.bound is False
    refusal = await client.denial(
        tool_name="write_file", tool_class=WriteFile, arguments={"file_path": "x"}
    )
    assert refusal is not None and "no decision can be asked for" in refusal
    await client.prime_catalog({"write_file": WriteFile, "read_file": WriteFile})
    assert client.catalog_withholds("write_file") is True
    assert client.catalog_withholds("read_file") is True
    assert owner.checks == []


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("tamper", "fragment"),
    [
        (lambda p: dict(p, request_id=agent_request_id(9)), "different request"),
        (lambda p: dict(p, epoch="epoch-other"), "superseded child epoch"),
        (lambda p: dict(p, child_generation=99), "superseded child generation"),
        (lambda p: dict(p, parent_session_id="someone-else"), "another session"),
        (lambda p: dict(p, parent_incarnation="owner-0"), "another owner incarnation"),
    ],
)
async def test_every_reply_mismatch_is_a_refusal(
    tamper: Callable[[dict[str, Any]], dict[str, Any]], fragment: str
) -> None:
    """A reply minted for another question refuses the call it answers."""
    from chartreux.core.tools.builtins.write_file import WriteFile

    client = _client(
        _ScriptedOwner(
            revision={"policy": 3, "registry": 0, "grants": 0, "ancestors": 0},
            tamper=tamper,
        )
    )
    refusal = await client.denial(
        tool_name="write_file", tool_class=WriteFile, arguments={"file_path": "x"}
    )
    assert refusal is not None and fragment in refusal


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_a_rolled_back_revision_vector_is_a_refusal() -> None:
    """A vector that goes backwards is refused, and never lowers what was seen."""
    from chartreux.core.tools.builtins.write_file import WriteFile

    owner = _ScriptedOwner(
        revision={"policy": 3, "registry": 0, "grants": 0, "ancestors": 0}
    )
    client = _client(owner)
    assert (
        await client.denial(
            tool_name="write_file", tool_class=WriteFile, arguments={"file_path": "x"}
        )
        is None
    )
    assert client.revision["policy"] == 3

    owner.revision = _zero_revision()
    refusal = await client.denial(
        tool_name="write_file", tool_class=WriteFile, arguments={"file_path": "x"}
    )
    assert refusal is not None and "rolled-back" in refusal
    assert client.revision["policy"] == 3


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_an_unreachable_owner_denies_the_gate_and_aborts_the_catalog() -> None:
    """A dead pipe or a timeout is a denial, never a manufactured allow."""
    from chartreux.core.tools.builtins.write_file import WriteFile

    dead = _ScriptedOwner(raises=RuntimeError("pipe gone"))
    client = _client(dead)
    with pytest.raises(_child_agent_module().ChildAuthorityUnavailable):
        await client.denial(
            tool_name="write_file", tool_class=WriteFile, arguments={"file_path": "x"}
        )

    # The tool gate turns that into a skip with a reason the model can read.
    manager = _attached_manager(client)
    reason = await manager.live_authority_denial(
        "write_file", WriteFile, {"file_path": "x"}
    )
    assert reason is not None and "live authority is unavailable" in reason

    # A catalog pass cannot be completed either: it aborts typed, and every
    # name it never got an answer for stays withheld.
    with pytest.raises(_child_agent_module().ChildAuthorityUnavailable):
        await client.prime_catalog({"write_file": WriteFile})
    assert client.catalog_withholds("write_file") is True

    slow = _client(_SlowOwner(), timeout=0.05)
    with pytest.raises(_child_agent_module().ChildAuthorityUnavailable):
        await slow.denial(
            tool_name="write_file", tool_class=WriteFile, arguments={"file_path": "x"}
        )


class _SlowOwner:
    async def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(5)
        raise AssertionError("the bounded wait must expire first")  # pragma: no cover


def _attached_manager(client: Any) -> Any:
    loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["read_file", "write_file"])
    )
    loop.attach_live_authority(client)
    return loop.tool_manager


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_an_unreadable_answer_is_a_refusal() -> None:
    """An undecodable reply refuses the call it answers, never the pass."""
    from chartreux.core.tools.builtins.write_file import WriteFile

    client = _client(_ScriptedOwner(tamper=lambda payload: {"decision": "maybe"}))
    refusal = await client.denial(
        tool_name="write_file", tool_class=WriteFile, arguments={"file_path": "x"}
    )
    assert refusal is not None and "unreadable decision" in refusal


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_an_invalidation_drops_the_catalog_and_never_caches_an_allow() -> None:
    """The cache is dropped on invalidation; the gate asks again regardless."""
    from chartreux.core.tools.builtins.write_file import WriteFile

    owner = _ScriptedOwner(verdicts=[AGENT_AUTHORITY_ALLOW, AGENT_AUTHORITY_DENY])
    client = _client(owner)
    await client.prime_catalog({"write_file": WriteFile})
    assert client.catalog_withholds("write_file") is False
    assert client.catalog_stale is False

    # A moved vector DROPS the cache: the frame carries no policy to apply.
    client.receive_changed(
        AgentAuthorityChanged(
            session_id="session-1",
            incarnation="owner-1",
            revision={"policy": 1, "registry": 0, "grants": 0, "ancestors": 0},
        )
    )
    assert client.change_count == 1
    assert client.catalog_stale is True
    assert client.catalog_withholds("write_file") is True

    # The execution gate never depended on the cache: it asks again, and the
    # owner's fresh answer (a deny here) is what the call runs on. A DROPPED
    # invalidation therefore costs a stale catalog, never a stale allow.
    refusal = await client.denial(
        tool_name="write_file", tool_class=WriteFile, arguments={"file_path": "x"}
    )
    assert refusal is not None

    # A foreign session or a rolled-back vector is ignored outright.
    client.receive_changed(
        AgentAuthorityChanged(
            session_id="other",
            incarnation="owner-1",
            revision={"policy": 9, "registry": 0, "grants": 0, "ancestors": 0},
        )
    )
    assert client.change_count == 2
    assert client.revision["policy"] == 1


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_an_invalidation_mid_pass_never_lets_the_pass_republish() -> None:
    """A changed frame landing mid-pass must not be undone by the pass's end.

    The pass asks one question per name, each answer awaited on the wire, so
    an owner invalidation can land between two questions. On the pre-W7 tree
    the pass's ``finally`` republished its gathered verdicts with
    ``stale=False`` anyway, advertising a revoked name while the scheduled
    refresh was still running. The interrupted pass must instead leave the
    dropped fail-closed cache for that refresh to republish.
    """
    from chartreux.core.tools.builtins.write_file import WriteFile

    tools = {"read_file": WriteFile, "write_file": WriteFile}
    owner = _GatedOwner()
    client = _client(owner)

    # The pass starts and parks on its first catalog question.
    pass_task = asyncio.create_task(client.prime_catalog(tools))
    await asyncio.wait_for(owner.arrived.wait(), 5)

    # The owner's policy moves while the pass is in flight: the cache drops.
    client.receive_changed(
        AgentAuthorityChanged(
            session_id="session-1",
            incarnation="owner-1",
            revision={"policy": 1, "registry": 0, "grants": 0, "ancestors": 0},
        )
    )
    assert client.catalog_stale is True
    assert client.catalog_withholds("write_file") is True

    # The pass finishes its questions (all answered allow); its end must NOT
    # republish over the invalidation it was interrupted by.
    owner.release.set()
    await asyncio.wait_for(pass_task, 5)
    assert [check.tool_name for check in client_checks(client)] == [
        "read_file",
        "write_file",
    ]
    assert client.catalog_stale is True
    assert client.catalog_withholds("write_file") is True
    assert client.catalog_withholds("read_file") is True

    # The already-scheduled refresh re-asks and publishes fresh: the revoked
    # name stays withheld, the rest is advertised, and the cache is not stale.
    owner.deny_names = {"write_file"}
    await client.prime_catalog(tools)
    assert client.catalog_stale is False
    assert client.catalog_withholds("write_file") is True
    assert client.catalog_withholds("read_file") is False


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_a_new_run_rejects_an_answer_fenced_to_the_previous_run() -> None:
    """A retained child's next run is a fresh incarnation for its questions."""
    from chartreux.core.tools.builtins.write_file import WriteFile

    owner = _ScriptedOwner()
    client = _client(owner)
    await client.begin_run()
    first_generation = client.generation
    first_epoch = owner.checks
    assert first_epoch == []  # no question was asked yet

    answer = await client.denial(
        tool_name="write_file", tool_class=WriteFile, arguments={"file_path": "x"}
    )
    assert answer is None
    first = owner.checks[0]

    await client.begin_run()
    assert client.generation == first_generation + 1

    # A reply minted for the earlier run is refused, whatever it says.
    stale = _ScriptedOwner(tamper=lambda payload: dict(payload, epoch=first.epoch))
    replaying = _client(stale)
    await replaying.begin_run()
    await replaying.begin_run()
    refusal = await replaying.denial(
        tool_name="write_file", tool_class=WriteFile, arguments={"file_path": "x"}
    )
    assert refusal is not None and "superseded child epoch" in refusal

    # The sequence only ever advances: the fresh question is a new request id.
    refusal = await client.denial(
        tool_name="write_file", tool_class=WriteFile, arguments={"file_path": "x"}
    )
    assert refusal is None
    assert [check.request_id for check in owner.checks] == [
        agent_request_id(1),
        agent_request_id(2),
    ]
    assert owner.checks[1].child_generation == client.generation


# ---------------------------------------------------------------------------
# The manager and the loop: catalog parity, ask ordering, the final gate.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_a_process_child_withholds_and_denies_on_the_same_name() -> None:
    """The catalog axis and the execution gate agree, both from one decision."""
    loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["read_file", "write_file"])
    )
    manager = loop.tool_manager
    # A root manager withholds nothing, whatever it is asked.
    assert manager.child_catalog_withholdings() == frozenset()
    assert manager._child_catalog_withholds("write_file") is False

    client = _client(_ScriptedOwner(deny_names={"write_file"}))
    loop.attach_live_authority(client)
    # G5b/docstring: a process child has NO in-process ancestor, so the old
    # "None parent authority means root" reading is gone -- the live client is
    # what makes this manager a bounded child's.
    assert manager._parent_authority_getter is None
    assert manager._is_bounded_child is True

    await client.prime_catalog({
        name: cls for name, cls in manager.available_tools.items()
    })
    assert client.catalog_withholds("read_file") is False
    assert client.catalog_withholds("write_file") is True
    # Parity: the name the catalog axis withholds is the name the execution
    # gate refuses, and the name it advertises is the name the gate admits --
    # both read the same client, so there is no "advertised but denied" (or
    # its inverse) state between the two surfaces.
    withholdings = manager.child_catalog_withholdings()
    assert "write_file" in withholdings
    assert "read_file" not in withholdings
    from chartreux.core.tools.builtins.write_file import WriteFile

    assert (
        await manager.live_authority_denial("write_file", WriteFile, {"file_path": "x"})
        is not None
    )
    assert (
        await manager.live_authority_denial("read_file", WriteFile, {"file_path": "x"})
        is None
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_loop_asks_the_owner_before_its_own_resolution(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """An owner-refused call never reaches a local grant or approval path."""
    from chartreux.core.tools.builtins.write_file import WriteFileArgs

    loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["read_file", "write_file"]),
        cwd=tmp_path,
    )
    client = _client(_ScriptedOwner(verdicts=[AGENT_AUTHORITY_DENY]))
    loop.attach_live_authority(client)
    # The local policy would DENY this tool outright. The owner is asked
    # anyway, and its answer is the one the decision carries: the ask comes
    # first, so a refusal is never turned into a grant prompt.
    _disable(monkeypatch, loop.tool_manager, "write_file")
    tool = loop.tool_manager.get("write_file")
    decision = await loop._should_execute_tool(
        tool, WriteFileArgs(file_path=str(tmp_path / "x.txt"), content="x")
    )
    assert decision.verdict.value == "skip"
    assert decision.feedback is not None and "not on the axis" in decision.feedback
    assert len(client_checks(client)) == 1
    assert not (tmp_path / "x.txt").exists()


def client_checks(client: Any) -> list[Any]:
    """The questions one scripted client asked (its owner's log)."""
    return list(client._call_host.checks)


def _disable(monkeypatch: Any, manager: Any, tool_name: str) -> None:
    """Disable one tool on the loop's live config, as a policy commit would."""
    tools = dict(manager._config.tools)
    tools[tool_name] = {"permission": ToolPermission.NEVER.value}
    denied = manager._config.model_copy(update={"tools": tools})
    monkeypatch.setattr(manager, "_config_getter", lambda: denied)
    manager._invalidate_session_root_grants()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_execution_gate_asks_again_inside_the_snapshot_window(
    tmp_path: Path,
) -> None:
    """The advisory allow is not the admission: the gate asks again and wins."""
    from chartreux.core.events import ToolResultEvent
    from chartreux.core.llm_models import FunctionCall, ToolCall

    target = tmp_path / "leaked.txt"
    call = ToolCall(
        id="held",
        index=0,
        function=FunctionCall(
            name="write_file",
            arguments=json.dumps({"file_path": str(target), "content": "leaked"}),
        ),
    )
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[call])],
        [mock_llm_chunk(content="done")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["read_file", "write_file"]),
        backend=backend,
        cwd=tmp_path,
    )
    # The advisory pass allows; the answer taken AFTER the file-snapshot await
    # refuses. That is the whole point of the second ask: the window between
    # them is where an owner's revocation lands.
    client = _client(
        _ScriptedOwner(verdicts=[AGENT_AUTHORITY_ALLOW, AGENT_AUTHORITY_DENY])
    )
    loop.attach_live_authority(client)

    results: list[ToolResultEvent] = []
    async for event in loop.act("write the file"):
        if isinstance(event, ToolResultEvent):
            results.append(event)

    assert len(results) == 1
    assert results[0].skip_reason is not None
    assert "not on the axis" in results[0].skip_reason
    assert not target.exists()
    asked = [check.tool_name for check in client_checks(client)]
    assert asked == ["write_file", "write_file"]
    assert [check.decision_kind for check in client_checks(client)] == [
        AGENT_AUTHORITY_DECISION_KIND_TOOL_INVOCATION
    ] * 2


# ---------------------------------------------------------------------------
# Acceptance discrimination: a REAL process child, held inside the window.
# ---------------------------------------------------------------------------

_CHILD_AGENT_SCRIPT = """\
import asyncio
import json
import os
import time
from pathlib import Path

from chartreux.app_server.child_agent import (
    HarnessChildLoopFactory,
    serve_child_agent,
)
from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.tools.builtins.write_file import WriteFile
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

init_harness_files_manager("user", "project")

GATE = Path(os.environ["G5B_GATE_DIR"])
TARGET = Path(os.environ["G5B_TARGET"])
WORKSPACE = Path(os.environ["G5B_WORKSPACE"])


def _held_snapshot(self, args):
    # The invocation's world-change window: the executor awaits this before
    # the final admission gate. It parks until the test releases it, so the
    # owner's tightening is ordered strictly inside that window.
    (GATE / "precheck").write_text("1")
    deadline = time.time() + 120
    while time.time() < deadline:
        if (GATE / "release").exists():
            return None
        time.sleep(0.01)
    return None


class _Process:
    # Only the HarnessProcess seam is replaced; the REAL child factory code
    # (spec decode, AgentConfig/ClientInfo validation, the inherited-workspace
    # and policy decodes, the transcript logging) all runs.
    async def open_root(self, request):
        call = ToolCall(
            id="held",
            index=0,
            function=FunctionCall(
                name="write_file",
                arguments=json.dumps({
                    "file_path": str(TARGET),
                    "content": "leaked",
                }),
            ),
        )
        loop = build_test_agent_loop(
            config=build_test_vibe_config(
                session_logging=request.session_logging,
                enabled_tools=["read_file", "write_file"],
            ),
            backend=FakeBackend([
                [mock_llm_chunk(tool_calls=[call])],
                [mock_llm_chunk(content="done")],
            ]),
            cwd=WORKSPACE,
        )
        WriteFile.get_file_snapshot = _held_snapshot
        return loop

    async def close(self):
        pass


factory = HarnessChildLoopFactory()
factory._process = _Process()
asyncio.run(serve_child_agent(factory=factory, announce_ready=bootstrap_requested()))
"""


class _RecordingSpawn:
    """Delegates to the real spawn port and records every spec it carried."""

    agent_id = "recorded"

    def __init__(self, real: ProcessSubagentSpawn) -> None:
        self._real = real
        self.specs: list[Any] = []

    async def spawn(self, spec: Any) -> Any:
        self.specs.append(spec)
        return await self._real.spawn(spec)


def _child_env(
    guard_dir: Path, gate: Path, target: Path, workspace: Path
) -> dict[str, str]:
    env = dict(os.environ)
    env[TRANSPORT_ENV_VAR] = STDIO_TRANSPORT
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["G5B_GATE_DIR"] = str(gate)
    env["G5B_TARGET"] = str(target)
    env["G5B_WORKSPACE"] = str(workspace)
    paths = [str(_REPO_ROOT), str(guard_dir)]
    inherited = env.get("PYTHONPATH")
    if inherited:
        paths.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.update(egress.child_environment(egress.MODE_ALL))
    return env


async def _await_path(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"the child never reached {path.name}")


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_a_real_child_denies_when_the_owner_tightens_inside_the_window(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Held between precheck and admission, a tightened owner must deny.

    The child is a REAL OS process spawned through ``ProcessSubagentSpawn`` and
    served by the real ``ChildAgentServer``; its questions travel its own pipe
    to the client that spawned it and are answered by this test's owner. The
    first question is answered ALLOW -- the child could not reach its
    file-snapshot window otherwise, which is itself the proof that the relay
    works end to end. Then the owner disables the tool, the window closes, and
    the call must be refused with no side effect.

    On the pre-G5b tree the child carries no authority client at all, so there
    is no question to answer and no second gate to run: the write lands and
    this assertion fails. That is the defect G5b removes.
    """
    gate = tmp_path / "gate"
    gate.mkdir()
    target = tmp_path / "leaked.txt"
    sessions = tmp_path / "sessions"
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(
                enabled=True, save_dir=str(sessions), session_prefix="session"
            ),
            enabled_tools=["read_file", "write_file"],
        ),
        cwd=tmp_path,
    )
    with egress.bootstrap_directory() as guard_dir:
        spawn = _RecordingSpawn(
            ProcessSubagentSpawn(
                env=_child_env(guard_dir, gate, target, tmp_path),
                cwd=str(_REPO_ROOT),
                ready_timeout=120,
                request_timeout=120,
                # The interpreter running this suite: the child needs no PATH
                # lookup, so the pin also runs inside the bwrap gate (whose PATH is
                # minimal) with the same interpreter and PYTHONPATH.
                command=[sys.executable, "-c", _CHILD_AGENT_SCRIPT],
            )
        )
        host = AgentChildHost(
            loop,
            spawn=spawn,
            client_info=_CLIENT_INFO,
            client_capabilities=_CLIENT_CAPABILITIES,
        )
        transport, peer = memory_transport_pair()
        server = RootAgentServer(transport, factory=cast(Any, SimpleNamespace()))
        server.install_child_host(host)
        server._loop = loop
        server._published_authority_revision = server._authority_revision_vector()
        serving = asyncio.create_task(server.serve())
        handle = None
        try:
            args = TaskArgs(task="write the file", identity="coder", retain=False)
            handle = await host.publish(
                args,
                InvokeContext(tool_call_id="task-1", session_id=loop.session_id),
                attempt_id="attempt-live-1",
                retained=False,
            )
            # G5b: the spawn payload named this owner's incarnation, so the
            # child can fence (and this owner can recognise) its questions.
            # The carrier key only exists on a G5b tree; on an earlier tree the
            # decisive fact below is behavioural instead.
            wire = _wire()
            key = getattr(wire, "PARENT_INCARNATION_PAYLOAD_KEY", None)
            if key is not None:
                assert _published_payload(host)[key] == (
                    wire.encode_parent_incarnation(server.authority_incarnation)
                )

            run = asyncio.create_task(
                host.run_child(
                    handle,
                    args,
                    InvokeContext(tool_call_id="task-2", session_id=loop.session_id),
                )
            )
            await _await_path(gate / "precheck", 120)
            # The child is parked INSIDE the invocation window: its advisory
            # question was answered allow (that is how it got here), and the
            # final admission has not been taken yet.
            _disable(monkeypatch, loop.tool_manager, "write_file")
            (gate / "release").write_text("1")
            result = await asyncio.wait_for(run, 120)
            assert result is not None
            assert not target.exists(), (
                "the child executed a tool the owner had revoked inside the "
                "invocation window"
            )
        finally:
            if handle is not None:
                await handle.spawned.close()
            for close in (transport.close, peer.close):
                try:
                    await close()
                except BaseException:
                    pass
            await asyncio.gather(serving, return_exceptions=True)
            await host.close()
            await loop.aclose()


def _published_payload(host: AgentChildHost) -> Mapping[str, Any]:
    """The payload the host's last publication built."""
    spawn = cast(Any, host._spawn)
    assert spawn.specs, "the host published nothing"
    return spawn.specs[-1].payload
