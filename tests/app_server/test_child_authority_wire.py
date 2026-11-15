"""G5a pins: the live child-authority wire contract and its owner endpoint.

Round 3 of the panel (agent-85 design, user-ruled option B) replaced the F11
static inherited-policy snapshot with a per-decision authority request: a child
asks its IMMEDIATE parent, over the agent wire, one typed question per tool
invocation, and the parent answers from its CURRENT policy at request time.
The answer is allow/deny only -- a child never receives policy it could execute
on its own -- and invalidation travels as a separate, id-less notification.

This module holds the owner/parent half of that contract (G5a): the typed wire
codecs and their strict refusal matrix, the mandatory two-sided capability, the
owner-side decision endpoint (allow/deny, current-policy evaluation, revision
fencing, tightening ordering), the coalesced authority-change emission, and the
observability relay/bind the coordinator uses. The child-side producer and the
enforcement that consumes these decisions are G5b and are not exercised here.

Fail-closed is the theme of every pin: a missing capability, a foreign or
unreadable input, a stale epoch, a replayed or reordered request, a rolled-back
revision vector and an unknown implementation all deny -- never a default allow
and never a decision on stale authority. No wall-clock value takes part in any
ordering.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.app_server._subagent_launch import ChildPublicationError
from chartreux.app_server.agent_wire import (
    AGENT_AUTHORITY_ALLOW,
    AGENT_AUTHORITY_CHANGED_METHOD,
    AGENT_AUTHORITY_CHECK_FIELDS,
    AGENT_AUTHORITY_CHECK_METHOD,
    AGENT_AUTHORITY_DECISION_KIND_TOOL_INVOCATION,
    AGENT_AUTHORITY_DENY,
    AGENT_RECEIVING_REQUESTS,
    AUTHORITY_REVISION_COMPONENTS,
    COORDINATOR_RECEIVING_NOTIFICATIONS,
    AgentAuthorityChanged,
    AgentAuthorityCheck,
    AgentAuthorityDecision,
    AgentWireError,
    agent_request_id,
    decode_authority_changed,
    decode_authority_check,
    decode_authority_decision,
    encode_authority_changed,
    encode_authority_check,
    encode_authority_decision,
    implementation_identity,
)
from chartreux.app_server.process_bootstrap import (
    ADVERTISED_CAPABILITIES,
    AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY,
    encode_ready_frame,
    parse_ready_frame,
)
from chartreux.app_server.process_session_backend import ProcessSessionBackend
import chartreux.app_server.root_agent as root_agent
from chartreux.app_server.root_agent import (
    AgentChildHost,
    AgentChildHostError,
    RootAgentServer,
    _ancestor_authority_revision,
)
from chartreux.app_server.root_agent_process import RootAgentProcess
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.models import ToolPermission
from tests.conftest import build_test_agent_loop, build_test_vibe_config

_TIMEOUT = 20.0
"""Bounded patience for one wire frame, generous under parallel load."""


def _zero_revision() -> dict[str, int]:
    return {component: 0 for component in sorted(AUTHORITY_REVISION_COMPONENTS)}


def _check(**overrides: Any) -> AgentAuthorityCheck:
    values: dict[str, Any] = {
        "decision_kind": AGENT_AUTHORITY_DECISION_KIND_TOOL_INVOCATION,
        "tool_name": "web_fetch",
        "implementation": "chartreux.core.tools.builtins.web_fetch:WebFetch",
        "arguments": {"url": "https://example.com"},
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


def _decision(**overrides: Any) -> AgentAuthorityDecision:
    values: dict[str, Any] = {
        "decision": AGENT_AUTHORITY_ALLOW,
        "reason": None,
        "epoch": "epoch-a",
        "parent_session_id": "session-1",
        "parent_incarnation": "owner-1",
        "child_generation": 1,
        "request_id": agent_request_id(1),
        "revision": _zero_revision(),
    }
    values.update(overrides)
    return AgentAuthorityDecision(**values)


def _changed(**overrides: Any) -> AgentAuthorityChanged:
    values: dict[str, Any] = {
        "session_id": "session-1",
        "incarnation": "owner-1",
        "revision": _zero_revision(),
    }
    values.update(overrides)
    return AgentAuthorityChanged(**values)


# ---------------------------------------------------------------------------
# Wire codecs: exact round trips and the strict refusal matrix.
# ---------------------------------------------------------------------------


def test_the_authority_check_codec_round_trips_exactly() -> None:
    """A question survives an encode/decode with every field intact."""
    check = _check(
        arguments={"url": "https://example.com/a", "depth": 2},
        revision={"policy": 3, "registry": 7, "grants": 0, "ancestors": 1},
    )
    assert decode_authority_check(encode_authority_check(check)) == check


def test_the_authority_decision_codec_round_trips_and_refuses_an_unknown_verdict() -> (
    None
):
    """An answer survives a round trip; an invented verdict never encodes."""
    for decision in (
        _decision(),
        _decision(decision=AGENT_AUTHORITY_DENY, reason="no"),
    ):
        assert (
            decode_authority_decision(encode_authority_decision(decision)) == decision
        )
    with pytest.raises(AgentWireError):
        encode_authority_decision(_decision(decision="maybe"))
    payload = encode_authority_decision(_decision())
    payload["decision"] = "maybe"
    with pytest.raises(AgentWireError):
        decode_authority_decision(payload)


def test_the_authority_changed_codec_round_trips_exactly() -> None:
    """The invalidation snapshot round trips and carries only its vector."""
    changed = _changed(
        revision={"policy": 9, "registry": 1, "grants": 2, "ancestors": 3}
    )
    assert decode_authority_changed(encode_authority_changed(changed)) == changed
    assert set(encode_authority_changed(changed)) == {
        "session_id",
        "incarnation",
        "revision",
    }


def test_the_check_codec_refuses_every_inexact_frame() -> None:
    """The check codec never repairs: exact keys, types and vector only."""
    baseline = encode_authority_check(_check())
    assert frozenset(baseline) == AGENT_AUTHORITY_CHECK_FIELDS

    def without(key: str) -> dict[str, Any]:
        payload = dict(baseline)
        del payload[key]
        return payload

    def replaced(key: str, value: Any) -> dict[str, Any]:
        payload = dict(baseline)
        payload[key] = value
        return payload

    mutations: list[dict[str, Any]] = [
        without("tool_name"),
        without("revision"),
        {**baseline, "extra": 1},
        replaced("child_generation", "1"),
        replaced("child_generation", True),
        replaced("child_generation", -1),
        replaced("decision_kind", "unknown-kind"),
        replaced("tool_name", ""),
        replaced("epoch", 5),
        replaced("arguments", ["not", "a", "mapping"]),
        replaced("revision", {"policy": 0, "registry": 0, "grants": 0}),
        replaced("revision", {**_zero_revision(), "extra": 0}),
        replaced("revision", {**_zero_revision(), "policy": "0"}),
        replaced("revision", {**_zero_revision(), "grants": -1}),
    ]
    for payload in mutations:
        with pytest.raises(AgentWireError):
            decode_authority_check(payload)


def test_the_decision_and_changed_codecs_refuse_every_inexact_frame() -> None:
    """Both reply frames are exactly as strict as the question they answer."""
    decision = encode_authority_decision(_decision())
    changed = encode_authority_changed(_changed())

    def missing(payload: dict[str, Any], key: str) -> dict[str, Any]:
        copy = dict(payload)
        del copy[key]
        return copy

    with pytest.raises(AgentWireError):
        decode_authority_decision(missing(decision, "reason"))
    with pytest.raises(AgentWireError):
        decode_authority_decision({**decision, "extra": True})
    with pytest.raises(AgentWireError):
        decode_authority_decision({**decision, "child_generation": True})
    with pytest.raises(AgentWireError):
        decode_authority_decision({**decision, "revision": {"policy": 0}})
    with pytest.raises(AgentWireError):
        decode_authority_decision({**decision, "request_id": ""})

    with pytest.raises(AgentWireError):
        decode_authority_changed(missing(changed, "incarnation"))
    with pytest.raises(AgentWireError):
        decode_authority_changed({**changed, "extra": 0})
    with pytest.raises(AgentWireError):
        decode_authority_changed({**changed, "revision": {"policy": 0}})
    with pytest.raises(AgentWireError):
        decode_authority_changed({**changed, "revision": {**_zero_revision(), "x": 0}})


def test_the_implementation_identity_is_the_module_and_qualified_name() -> None:
    """Both peers compute one comparable identity from the defining class."""
    from chartreux.core.tools.builtins.web_fetch import WebFetch

    assert implementation_identity(WebFetch) == (
        "chartreux.core.tools.builtins.web_fetch:WebFetch"
    )


# ---------------------------------------------------------------------------
# Mandatory two-sided capability and the closed direction tables.
# ---------------------------------------------------------------------------


def test_the_live_child_authority_capability_is_advertised_and_round_trips() -> None:
    """This build announces the mandatory capability both ends must see."""
    assert AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY == "live-child-authority"
    assert AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY in ADVERTISED_CAPABILITIES
    assert (
        AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY
        in parse_ready_frame(encode_ready_frame()).capabilities
    )


def test_the_direction_tables_admit_the_authority_frames() -> None:
    """The check is an owner-served request; the change is a coordinator relay."""
    assert AGENT_AUTHORITY_CHECK_METHOD == "agent/authority/check"
    assert AGENT_AUTHORITY_CHANGED_METHOD == "agent/authority/changed"
    assert AGENT_AUTHORITY_CHECK_METHOD in AGENT_RECEIVING_REQUESTS
    assert AGENT_AUTHORITY_CHANGED_METHOD in COORDINATOR_RECEIVING_NOTIFICATIONS


def test_a_process_child_without_the_capability_is_refused(monkeypatch: Any) -> None:
    """Neither side may silently downgrade to the F11 static snapshot.

    The child's ready frame is what carries its half, so a spawned process
    whose advertised set lacks the capability fails the publication typed; an
    in-process child (no wire peer) is unaffected; the owner's own half is the
    advertised set of this process, checked before any child is reserved.
    """
    # Child side: a process child that did not announce the capability.
    lacking = SimpleNamespace(
        wire_client=SimpleNamespace(advertised_capabilities=frozenset())
    )
    with pytest.raises(AgentChildHostError, match="live-child-authority"):
        AgentChildHost._require_live_authority_capability(lacking)
    announcing = SimpleNamespace(
        wire_client=SimpleNamespace(
            advertised_capabilities=frozenset({
                AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY
            })
        )
    )
    AgentChildHost._require_live_authority_capability(announcing)
    # In-process children carry no wire peer and are not gated by it.
    AgentChildHost._require_live_authority_capability(SimpleNamespace(wire_client=None))


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_owner_refuses_publication_without_its_own_capability(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The owner's own half: no advertised capability means no child at all."""
    loop = _owner_loop(tmp_path)
    host = AgentChildHost(loop, spawn=_NullSpawn())
    try:
        monkeypatch.setattr(root_agent, "ADVERTISED_CAPABILITIES", frozenset())
        with pytest.raises(ChildPublicationError, match="live-child-authority"):
            await host.publish(
                _task_args(),
                _ctx(loop, "task-1"),
                attempt_id="attempt-1",
                retained=True,
            )
    finally:
        await host.close()
        await loop.aclose()


# ---------------------------------------------------------------------------
# The owner-side decision endpoint.
# ---------------------------------------------------------------------------


class _NullSpawn:
    """A spawn adapter that answers a handle without a wire peer."""

    agent_id = "recorded"

    def __init__(self) -> None:
        self.specs: list[Any] = []

    async def spawn(self, spec: Any) -> Any:
        self.specs.append(spec)
        return SimpleNamespace(
            agent_id=spec.agent_id,
            session_id="child-session-1",
            child_session_dir=None,
            completion_source=None,
            wire_client=None,
            close=_noop,
        )


async def _noop() -> None:
    return None


def _owner_loop(tmp_path: Path) -> Any:
    return build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(
                enabled=True,
                save_dir=str(tmp_path / "sessions"),
                session_prefix="session",
            )
        )
    )


def _task_args() -> TaskArgs:
    return TaskArgs(task="authority probe", identity="coder", retain=True)


def _ctx(loop: Any, call_id: str) -> InvokeContext:
    return InvokeContext(tool_call_id=call_id, session_id=loop.session_id)


class _Frames:
    """One shared reader over the peer's inbound wire, with deferred frames."""

    def __init__(self, iterator: AsyncIterator[dict[str, Any]]) -> None:
        self._iterator = iterator
        self._deferred: list[dict[str, Any]] = []

    async def _pull(self, predicate: Any, timeout: float) -> dict[str, Any]:
        for index, message in enumerate(self._deferred):
            if predicate(message):
                return self._deferred.pop(index)
        while True:
            message = await asyncio.wait_for(
                self._iterator.__anext__(), timeout=timeout
            )
            if predicate(message):
                return message
            self._deferred.append(message)

    async def reply(self, rpc_id: str, *, timeout: float = _TIMEOUT) -> dict[str, Any]:
        return await self._pull(lambda m: m.get("id") == rpc_id, timeout)

    async def notification(
        self, method: str, *, timeout: float = _TIMEOUT
    ) -> dict[str, Any]:
        return await self._pull(
            lambda m: m.get("method") == method and "id" not in m, timeout
        )


class _Owner:
    """A served root that owns one published child, plus its wire peer."""

    def __init__(
        self,
        loop: Any,
        host: AgentChildHost,
        server: RootAgentServer,
        transport: Any,
        peer: Any,
        serving: asyncio.Task[None],
        frames: _Frames,
        handle: Any,
    ) -> None:
        self.loop = loop
        self.host = host
        self.server = server
        self.transport = transport
        self.peer = peer
        self.serving = serving
        self.frames = frames
        self.handle = handle

    async def ask(
        self,
        *,
        tool_name: str = "web_fetch",
        implementation: str | None = None,
        arguments: Mapping[str, Any] | None = None,
        child_agent_id: str | None = None,
        generation: int = 1,
        epoch: str = "epoch-a",
        request_id: str | None = None,
        session_id: str | None = None,
        incarnation: str | None = None,
        revision: Mapping[str, int] | None = None,
        rpc_id: str | None = None,
    ) -> AgentAuthorityDecision:
        manager = self.loop.tool_manager
        if implementation is None:
            selected = manager.delegable_tools.get(tool_name)
            implementation = (
                implementation_identity(selected) if selected is not None else "x:y"
            )
        if arguments is None:
            arguments = {"url": "https://example.com"}
        if request_id is None:
            request_id = agent_request_id(1)
        if rpc_id is None:
            rpc_id = f"rpc-{request_id}-{generation}"
        check = AgentAuthorityCheck(
            decision_kind=AGENT_AUTHORITY_DECISION_KIND_TOOL_INVOCATION,
            tool_name=tool_name,
            implementation=implementation,
            arguments=dict(arguments),
            child_agent_id=child_agent_id or self.handle.agent_id,
            child_generation=generation,
            epoch=epoch,
            parent_session_id=session_id or self.loop.session_id,
            parent_incarnation=incarnation or self.server.authority_incarnation,
            request_id=request_id,
            revision=dict(revision or _zero_revision()),
        )
        await self.peer.send({
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": AGENT_AUTHORITY_CHECK_METHOD,
            "params": encode_authority_check(check),
        })
        message = await self.frames.reply(rpc_id)
        assert "result" in message, message
        return decode_authority_decision(message["result"])

    async def close(self) -> None:
        for close in (self.transport.close, self.peer.close):
            try:
                await close()
            except BaseException:
                pass
        await asyncio.gather(self.serving, return_exceptions=True)
        await self.host.close()
        await self.loop.aclose()


async def _open_owner(tmp_path: Path) -> _Owner:
    loop = _owner_loop(tmp_path)
    host = AgentChildHost(loop, spawn=_NullSpawn())
    transport, peer = memory_transport_pair()
    server = RootAgentServer(transport, factory=cast(Any, SimpleNamespace()))
    server.install_child_host(host)
    server._loop = loop
    server._published_authority_revision = server._authority_revision_vector()
    serving = asyncio.create_task(server.serve())
    frames = _Frames(peer.messages())
    handle = await host.publish(
        _task_args(), _ctx(loop, "task-1"), attempt_id="attempt-1", retained=True
    )
    return _Owner(loop, host, server, transport, peer, serving, frames, handle)


def _tighten(monkeypatch: Any, manager: Any, tool_name: str) -> None:
    """Disable one tool on the owner's live policy, advancing its generation."""
    tools = dict(manager._config.tools)
    tools[tool_name] = {"permission": ToolPermission.NEVER.value}
    denied = manager._config.model_copy(update={"tools": tools})
    monkeypatch.setattr(manager, "_config_getter", lambda: denied)
    manager._invalidate_session_root_grants()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_owner_allows_a_delegable_invocation_from_an_owned_child(
    tmp_path: Path,
) -> None:
    """A shipped, delegable tool with canonical args is allowed from a child.

    The answer is fenced to the exact question and carries the owner's
    revision vector, never policy the child could execute on its own.
    """
    owner = await _open_owner(tmp_path)
    try:
        decision = await owner.ask(request_id=agent_request_id(1))
        assert decision.decision == AGENT_AUTHORITY_ALLOW
        assert decision.reason is None
        assert decision.request_id == agent_request_id(1)
        assert decision.epoch == "epoch-a"
        assert decision.child_generation == 1
        assert decision.parent_session_id == owner.loop.session_id
        assert decision.parent_incarnation == owner.server.authority_incarnation
        assert set(decision.revision) == set(AUTHORITY_REVISION_COMPONENTS)
        assert all(isinstance(value, int) for value in decision.revision.values())
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_owner_denies_every_unprovable_invocation(tmp_path: Path) -> None:
    """Not delegable, unknown implementation, bad args and unowned all deny."""
    owner = await _open_owner(tmp_path)
    try:
        undelegable = await owner.ask(
            tool_name="definitely_not_a_tool",
            implementation="x:y",
            arguments={},
            request_id=agent_request_id(1),
        )
        assert undelegable.decision == AGENT_AUTHORITY_DENY
        assert "not delegable" in (undelegable.reason or "")

        foreign = await owner.ask(
            implementation="chartreux.core.tools.builtins.write_file:WriteFile",
            request_id=agent_request_id(2),
        )
        assert foreign.decision == AGENT_AUTHORITY_DENY
        assert "different implementation" in (foreign.reason or "")

        bad_args = await owner.ask(
            tool_name="bash_start",
            arguments={"command": ["not", "a", "string"]},
            request_id=agent_request_id(3),
        )
        assert bad_args.decision == AGENT_AUTHORITY_DENY
        assert "could not be resolved" in (bad_args.reason or "")

        unowned = await owner.ask(
            child_agent_id="ghost-1", request_id=agent_request_id(4)
        )
        assert unowned.decision == AGENT_AUTHORITY_DENY
        assert "not a child this agent owns" in (unowned.reason or "")
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_tightening_ordered_after_a_decision_is_seen_by_the_next(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The endpoint reads CURRENT policy: a later tightening reverses an allow.

    The child presents the vector its previous answer carried; the owner's
    policy generation has advanced past it, and the re-decision denies -- an
    accepted-policy commit ordered before the decision is never bypassed.
    """
    owner = await _open_owner(tmp_path)
    try:
        allowed = await owner.ask(request_id=agent_request_id(1))
        assert allowed.decision == AGENT_AUTHORITY_ALLOW
        before = dict(allowed.revision)

        _tighten(monkeypatch, owner.loop.tool_manager, "web_fetch")

        denied = await owner.ask(request_id=agent_request_id(2), revision=before)
        assert denied.decision == AGENT_AUTHORITY_DENY
        assert "disabled by the owner's policy" in (denied.reason or "")
        assert denied.revision["policy"] > before["policy"]
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_fence_refuses_a_replayed_reordered_or_superseded_request(
    tmp_path: Path,
) -> None:
    """Epoch, generation and request sequence fence before any policy is read."""
    owner = await _open_owner(tmp_path)
    try:
        first = await owner.ask(request_id=agent_request_id(5))
        assert first.decision == AGENT_AUTHORITY_ALLOW

        replay = await owner.ask(request_id=agent_request_id(5))
        assert replay.decision == AGENT_AUTHORITY_DENY
        assert "late or replayed" in (replay.reason or "")

        reordered = await owner.ask(request_id=agent_request_id(4))
        assert reordered.decision == AGENT_AUTHORITY_DENY
        assert "late or replayed" in (reordered.reason or "")

        foreign_epoch = await owner.ask(request_id=agent_request_id(6), epoch="epoch-b")
        assert foreign_epoch.decision == AGENT_AUTHORITY_DENY
        assert "epoch this generation never minted" in (foreign_epoch.reason or "")

        stale_generation = await owner.ask(request_id=agent_request_id(7), generation=0)
        assert stale_generation.decision == AGENT_AUTHORITY_DENY
        assert "superseded child incarnation" in (stale_generation.reason or "")

        reused_epoch = await owner.ask(
            request_id=agent_request_id(8), generation=2, epoch="epoch-a"
        )
        assert reused_epoch.decision == AGENT_AUTHORITY_DENY
        assert "reuses the superseded epoch" in (reused_epoch.reason or "")
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_fence_refuses_a_rolled_back_revision_vector(tmp_path: Path) -> None:
    """A componentwise decrease is a rollback and never decides anything."""
    owner = await _open_owner(tmp_path)
    try:
        high = {"policy": 5, "registry": 5, "grants": 0, "ancestors": 0}
        accepted = await owner.ask(request_id=agent_request_id(1), revision=high)
        assert accepted.decision == AGENT_AUTHORITY_ALLOW

        rolled_back = await owner.ask(
            request_id=agent_request_id(2),
            revision={"policy": 4, "registry": 5, "grants": 0, "ancestors": 0},
        )
        assert rolled_back.decision == AGENT_AUTHORITY_DENY
        assert "rolled-back authority revision" in (rolled_back.reason or "")
        assert "policy" in (rolled_back.reason or "")
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_fence_refuses_a_foreign_session_or_incarnation(
    tmp_path: Path,
) -> None:
    """An answer is minted only for this owner's own open session."""
    owner = await _open_owner(tmp_path)
    try:
        foreign_session = await owner.ask(
            request_id=agent_request_id(1), session_id="another-session"
        )
        assert foreign_session.decision == AGENT_AUTHORITY_DENY
        assert "does not own" in (foreign_session.reason or "")

        stale_owner = await owner.ask(
            request_id=agent_request_id(2), incarnation="superseded-owner"
        )
        assert stale_owner.decision == AGENT_AUTHORITY_DENY
        assert "superseded owner incarnation" in (stale_owner.reason or "")

        foreign_origin = await owner.ask(request_id="request-3")
        assert foreign_origin.decision == AGENT_AUTHORITY_DENY
        assert "not an agent-origin request id" in (foreign_origin.reason or "")
    finally:
        await owner.close()


# ---------------------------------------------------------------------------
# Coalesced invalidation emission and the observability relay.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_policy_commit_emits_one_fresh_authority_change_snapshot(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A served commit that moved the vector publishes one fresh snapshot."""
    loop = _owner_loop(tmp_path)
    host = AgentChildHost(loop, spawn=_NullSpawn())
    transport, peer = memory_transport_pair()
    server = RootAgentServer(transport, factory=cast(Any, SimpleNamespace()))
    server.install_child_host(host)
    server._loop = loop
    server._published_authority_revision = server._authority_revision_vector()
    serving = asyncio.create_task(server.serve())
    frames = _Frames(peer.messages())
    try:
        handle = await host.publish(
            _task_args(), _ctx(loop, "task-1"), attempt_id="attempt-1", retained=True
        )
        owner = _Owner(loop, host, server, transport, peer, serving, frames, handle)
        _tighten(monkeypatch, loop.tool_manager, "web_fetch")
        await owner.ask(request_id=agent_request_id(1))
        frame = await frames.notification(AGENT_AUTHORITY_CHANGED_METHOD)
        assert "id" not in frame
        changed = decode_authority_changed(frame["params"])
        assert changed.session_id == loop.session_id
        assert changed.incarnation == server.authority_incarnation
        assert changed.revision == server.authority_revision
        assert changed.revision["policy"] > 1
    finally:
        for close in (transport.close, peer.close):
            try:
                await close()
            except BaseException:
                pass
        await asyncio.gather(serving, return_exceptions=True)
        await host.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_authority_change_relay_hands_the_frame_to_the_bound_sink() -> None:
    """The coordinator relay decodes typed and drops what it cannot decode."""
    process = object.__new__(RootAgentProcess)
    process._authority_changed_sink = None
    process._notifications = set()

    seen: list[AgentAuthorityChanged] = []

    async def sink(changed: AgentAuthorityChanged) -> None:
        seen.append(changed)

    process._authority_changed_sink = sink
    payload = encode_authority_changed(
        _changed(revision={"policy": 4, "registry": 2, "grants": 0, "ancestors": 1})
    )
    process._route_sink_frame(AGENT_AUTHORITY_CHANGED_METHOD, payload)
    await asyncio.gather(*process._notifications)
    assert seen == [
        AgentAuthorityChanged(
            session_id="session-1",
            incarnation="owner-1",
            revision={"policy": 4, "registry": 2, "grants": 0, "ancestors": 1},
        )
    ]

    # An undecodable frame is dropped with a diagnostic, never raised.
    process._route_sink_frame(AGENT_AUTHORITY_CHANGED_METHOD, {"session_id": "x"})
    await asyncio.gather(*process._notifications)
    assert len(seen) == 1

    # An unbound sink (no live backend attached) drops the frame.
    process._authority_changed_sink = None
    process._route_sink_frame(AGENT_AUTHORITY_CHANGED_METHOD, payload)
    assert process._notifications == set()
    assert len(seen) == 1


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_backend_observes_the_owner_vector_and_drops_foreign_sessions() -> (
    None
):
    """The backend records the owner's vector for observability only."""
    backend = object.__new__(ProcessSessionBackend)
    backend._process = cast(Any, SimpleNamespace(session_id="session-1"))
    backend.authority_incarnation = None
    backend.authority_revision = None
    backend.authority_change_count = 0

    await backend._observe_authority_changed(
        _changed(
            session_id="session-1",
            incarnation="owner-1",
            revision={"policy": 3, "registry": 1, "grants": 0, "ancestors": 0},
        )
    )
    assert backend.authority_incarnation == "owner-1"
    assert backend.authority_revision == {
        "policy": 3,
        "registry": 1,
        "grants": 0,
        "ancestors": 0,
    }
    assert backend.authority_change_count == 1

    await backend._observe_authority_changed(
        _changed(session_id="someone-else", incarnation="other")
    )
    assert backend.authority_incarnation == "owner-1"
    assert backend.authority_change_count == 1


def test_an_unreadable_ancestor_chain_is_never_treated_as_a_root() -> None:
    """A None getter is a root; a chain that cannot be read is not."""
    assert (
        _ancestor_authority_revision(SimpleNamespace(_parent_authority_getter=None))
        == 0
    )
    broken = SimpleNamespace(
        _parent_authority_getter=lambda: None, _parent_authority=lambda: None
    )
    assert _ancestor_authority_revision(broken) is None
    assert _ancestor_authority_revision(SimpleNamespace()) is None
