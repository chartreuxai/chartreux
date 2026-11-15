"""Backend-contract wire tests for the typed live-agent-inventory carrier.

``agents/inventory/read`` (v0.7.9 point 5) answers one control-class snapshot:
the revision and the owner's wire projection sampled together, plus the ROOT
incarnation pair the client's per-incarnation revision floor compares within.
On the stdio route the owner is the root agent's child host, so the nodes here
prove the answer crosses from the handles that process spawned -- never from a
coordinator copy of the inventory -- and that every retained-child mutation
the host guards (the guarded release) advances the carried revision.
"""

from __future__ import annotations

from typing import Any

import pytest

from chartreux.app_server.protocol import ClientCapabilities, SessionOptions
from chartreux.app_server.root_agent import AgentChildHandle, AgentChildHost
from chartreux.core.subagents import TaskResult
from tests.app_server.backend_contract.conftest import (
    connect_backend_contract_stdio_host,
    install_memory_root_agent,
)
from tests.conftest import build_test_agent_loop


class _FakeWireClient:
    """A live child's own page reader; the inventory read never calls it."""

    async def read_page(
        self, agent_id: str, *, before: str | None = None, limit: int = 50
    ) -> Any:
        raise AssertionError("the inventory read never pages a transcript")


class _FakeSpawned:
    """A spawned child handle stand-in: live, with a page reader and no dir."""

    alive = True
    child_session_dir: str | None = None
    root_session_dir: str | None = None

    def __init__(self) -> None:
        self.wire_client = _FakeWireClient()

    async def close(self) -> None:
        self.alive = False


class _FakeSpawnPort:
    """A spawn adapter the injected handles never draw from."""

    budget: Any = None

    async def spawn(self, spec: Any) -> Any:
        raise AssertionError("the injected handles are published directly")


def _retained_child(host: AgentChildHost) -> AgentChildHandle:
    handle = AgentChildHandle(
        agent_id="child",
        identity="coder",
        attempt_id="attempt-1",
        spawned=_FakeSpawned(),
        session_id="child-session",
        child_dir=None,
        retained=True,
        slot=object(),
    )
    handle.run_id = "run-1"
    handle.result = TaskResult(
        response="done", turns_used=1, completed=True, agent_id="child", run_id="run-1"
    )
    host._children["child"] = handle
    host._advance_inventory_revision()
    return handle


@pytest.mark.asyncio
async def test_inventory_read_carries_the_incarnation_and_revision_on_the_selected_transport(
    backend_contract_transport_session: Any,
) -> None:
    """The read answers one typed snapshot on both transports.

    The attached session is the incarnation the snapshot names, the
    revision is a non-negative integer the client may floor on, and an
    owner with no children answers the honest empty inventory rather than
    skipping the projection.
    """
    session = backend_contract_transport_session
    snapshot = await session.read_agent_inventory()
    assert snapshot.session_id == session.session_id
    assert isinstance(snapshot.generation, int)
    assert snapshot.generation >= 0
    assert snapshot.revision >= 0
    assert snapshot.agents == []


@pytest.mark.asyncio
async def test_the_stdio_owner_answers_from_the_child_handles_it_spawned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stdio owner projects its own handles; a release advances the revision.

    Only the root agent's child host holds the retained completed child, so
    the response carrying it proves the read crossed the wire to the owner
    instead of being assembled from a coordinator copy. The guarded release
    removes the handle and bumps the owner's monotone revision, so the next
    read answers a strictly higher revision with the smaller inventory.
    """
    loop = build_test_agent_loop()
    child_host = AgentChildHost(loop, spawn=_FakeSpawnPort())
    install_memory_root_agent(monkeypatch, loop, child_host=child_host)
    connection = await connect_backend_contract_stdio_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    session = await connection.host.open_session()
    try:
        _retained_child(child_host)

        first = await session.read_agent_inventory()
        assert first.session_id == loop.session_id
        assert first.generation == loop.session_generation
        agents = {agent.agent_id: agent for agent in first.agents}
        assert set(agents) == {"child"}
        child = agents["child"]
        assert child.availability == "idle"
        assert child.current_run_id is None
        assert child.last_run_status == "completed"
        assert child.turns_used == 1
        assert child.latest_run_id == "run-1"

        released = await session.release_agent("child", "run-1")
        assert released.outcome.value == "released"

        second = await session.read_agent_inventory()
        assert second.session_id == first.session_id
        assert second.generation == first.generation
        assert second.revision > first.revision
        assert second.agents == []
    finally:
        await session.close()
        await connection.host.close()
        await loop.aclose()
