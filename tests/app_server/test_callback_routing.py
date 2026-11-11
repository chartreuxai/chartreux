"""Manifest-routed callback delivery in the coordinator.

Producer emission is WP3/WP5 work, so these tests drive the coordinator side
against a test-local producer stand-in: it registers the routing rows a
producer will register and stands in for the connection that serves the agent.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._session_backend_port import (
    SessionBackend,
    SessionBackendHost,
)
from chartreux.app_server.models import (
    OpenCallbackState,
    PublicCallbackEntry,
    PublicEntryGenerationStatus,
    UserInputCallbackDetail,
)
from chartreux.app_server.protocol import ClientCapabilities
from chartreux.app_server.registry_manifest import (
    AgentRoute,
    CallbackTarget,
    RegistryManifest,
)
from chartreux.app_server.server import AppServer
from chartreux.app_server.transport import MemoryJsonRpcTransport, memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.questions import QuestionChoice, UserQuestion, UserQuestionRequest


class _AgentLoop:
    """Hashable stand-in for the root loop keying the registry manifests."""

    session_id = "root-session"


class _ProducerStandIn:
    """Test-local producer publishing agent routes and their callbacks."""

    def __init__(self, manifest: RegistryManifest) -> None:
        self.manifest = manifest

    def publish(self, agent_id: str, incarnation_id: str) -> AgentRoute:
        route = AgentRoute("connection-1", incarnation_id)
        self.manifest.register_agent(agent_id, route)
        return route

    def route(self, callback_id: str, agent_id: str) -> None:
        self.manifest.route_callback(callback_id, agent_id)


class _RootBackend:
    """Root backend stand-in carrying the handler hop the server walks."""

    def __init__(
        self, loop: _AgentLoop | None = None, factory: AgentRuntimeFactory | None = None
    ) -> None:
        self.handler = SimpleNamespace(_agent_loop=loop, _runtime_factory=factory)


def _manifest(tmp_path: Path) -> RegistryManifest:
    session_dir = tmp_path / "sessions" / "root"
    session_dir.mkdir(parents=True)
    manifest = RegistryManifest(session_dir)
    manifest.initialize()
    return manifest


def _routed_root(manifest: RegistryManifest) -> _RootBackend:
    loop = _AgentLoop()
    factory = AgentRuntimeFactory()
    factory.registry_manifests[cast(AgentLoop, loop)] = manifest
    return _RootBackend(loop, factory)


def _question(text: str) -> UserQuestionRequest:
    return UserQuestionRequest(
        questions=[
            UserQuestion(
                question=text,
                options=[QuestionChoice(label="Allow"), QuestionChoice(label="Deny")],
            )
        ]
    )


def _callback(
    callback_id: str, *, text: str = "Approve the workspace reset?"
) -> PublicCallbackEntry:
    return PublicCallbackEntry(
        id=f"callback:{callback_id}",
        session_id="child-session",
        turn_id="turn-1",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        callback_id=callback_id,
        title="Input required",
        detail=UserInputCallbackDetail(request=_question(text)),
        state=OpenCallbackState(),
    )


def _server(root: object) -> tuple[AppServer, MemoryJsonRpcTransport]:
    client_transport, server_transport = memory_transport_pair()
    server = AppServer(
        server_transport,
        session_backend_host_factory=lambda _services: cast(
            SessionBackendHost, SimpleNamespace()
        ),
    )
    server._root = cast(SessionBackend, root)
    server._connection_attached = True
    server._client_capabilities = ClientCapabilities(callback_kinds=["user_input"])
    return server, client_transport


async def _frames(transport: MemoryJsonRpcTransport) -> list[dict[str, Any]]:
    """Every frame the server has already sent on this connection."""
    stream = cast(AsyncGenerator[dict[str, Any], None], transport.messages())
    frames: list[dict[str, Any]] = []
    try:
        while True:
            try:
                frames.append(await asyncio.wait_for(anext(stream), 0.05))
            except TimeoutError:
                return frames
    finally:
        await stream.aclose()


def _only_delivery(server: AppServer, frame: dict[str, Any]) -> Any:
    return server._callback_requests[cast(int, frame["id"])]


@pytest.mark.asyncio
async def test_routed_callback_is_delivered_through_its_manifest_row(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    route = producer.publish("agent-1", "incarnation-1")
    producer.route("callback-1", "agent-1")
    server, client = _server(_routed_root(manifest))

    await server._deliver_callback(_callback("callback-1"))

    frames = await _frames(client)
    assert [frame["method"] for frame in frames] == ["callback/call"]
    delivery = _only_delivery(server, frames[0])
    assert delivery.target == CallbackTarget(agent_id="agent-1", route=route)


@pytest.mark.asyncio
async def test_stale_incarnation_row_is_retired_and_reregistered(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    producer.publish("agent-1", "incarnation-1")
    producer.route("callback-1", "agent-1")
    server, client = _server(_routed_root(manifest))

    await server._deliver_callback(_callback("callback-1"))
    assert len(await _frames(client)) == 1

    producer.publish("agent-1", "incarnation-2")
    assert json.loads(manifest.path.read_text(encoding="utf-8"))["callbacks"] == {}

    await server._deliver_callback(_callback("callback-1"))
    assert await _frames(client) == []
    assert len(server._callback_requests) == 1

    producer.route("callback-1", "agent-1")
    await server._deliver_callback(_callback("callback-1"))

    frames = await _frames(client)
    assert [frame["method"] for frame in frames] == ["callback/call"]
    assert _only_delivery(server, frames[0]).target == CallbackTarget(
        agent_id="agent-1", route=AgentRoute("connection-1", "incarnation-2")
    )


@pytest.mark.asyncio
async def test_removed_agent_stops_prompt_delivery(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    producer.publish("agent-1", "incarnation-1")
    producer.route("callback-1", "agent-1")
    server, client = _server(_routed_root(manifest))

    producer.publish("agent-1", "incarnation-1")
    manifest.remove_agent("agent-1")

    await server._deliver_callback(_callback("callback-1"))

    assert await _frames(client) == []
    assert server._callback_requests == {}
    # Re-registering the agent does not resurrect the retired prompt.
    producer.publish("agent-1", "incarnation-3")
    await server._deliver_callback(_callback("callback-1"))
    assert await _frames(client) == []


@pytest.mark.asyncio
async def test_in_process_callback_without_a_manifest_keeps_memory_delivery() -> None:
    server, client = _server(_RootBackend())

    await server._deliver_callback(_callback("callback-1"))

    frames = await _frames(client)
    assert [frame["method"] for frame in frames] == ["callback/call"]
    assert _only_delivery(server, frames[0]).target is None


@pytest.mark.asyncio
async def test_manifest_without_a_route_still_delivers_in_process_callbacks(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    server, client = _server(_routed_root(manifest))

    await server._deliver_callback(_callback("callback-1"))

    frames = await _frames(client)
    assert [frame["method"] for frame in frames] == ["callback/call"]
    assert _only_delivery(server, frames[0]).target is None


@pytest.mark.asyncio
async def test_persisted_manifest_never_carries_approval_payloads(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    producer.publish("agent-1", "incarnation-1")
    producer.route("callback-1", "agent-1")
    server, _client = _server(_routed_root(manifest))

    await server._deliver_callback(
        _callback("callback-1", text="Approve the workspace reset?")
    )

    persisted = manifest.path.read_text(encoding="utf-8")
    assert json.loads(persisted) == {
        "agents": {
            "agent-1": {
                "connection_id": "connection-1",
                "incarnation_id": "incarnation-1",
            }
        },
        "callbacks": {"callback-1": "agent-1"},
    }
    assert "Approve the workspace reset?" not in persisted
