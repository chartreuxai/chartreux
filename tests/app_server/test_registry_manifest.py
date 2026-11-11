from __future__ import annotations

import json
from pathlib import Path

import pytest

from chartreux.app_server.protocol import (
    AgentTranscriptGetResponse,
    AgentTranscriptState,
)
from chartreux.app_server.registry_manifest import (
    AgentRoute,
    CallbackTarget,
    LiveAgentUnavailableError,
    RegistryManifest,
)


def test_artifact_placement_and_callback_rows(tmp_path: Path) -> None:
    session = tmp_path / "session"
    agents = session / "agents"
    agents.mkdir(parents=True)
    transcript = agents / "existing"
    transcript.write_text("existing", encoding="utf-8")
    registry = RegistryManifest(session)
    registry.initialize()
    fence_inode = registry.fence_path.stat().st_ino
    route = AgentRoute("connection", "incarnation")
    registry.register_agent("agent", route)
    registry.route_callback("callback", "agent")
    assert registry.callback_route("callback") == route
    data = json.loads(registry.path.read_text(encoding="utf-8"))
    assert data == {
        "agents": {
            "agent": {"connection_id": "connection", "incarnation_id": "incarnation"}
        },
        "callbacks": {"callback": "agent"},
    }
    assert {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()} == {
        Path("session/agents/existing"),
        Path("session/.writer-fence.lock"),
        Path("session.registry.json"),
    }
    registry.remove_agent("agent")
    registry.initialize()
    assert registry.fence_path.stat().st_ino == fence_inode
    assert transcript.read_text(encoding="utf-8") == "existing"
    with pytest.raises(LiveAgentUnavailableError):
        registry.callback_route("callback")


@pytest.mark.asyncio
async def test_live_page_requires_reader(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = RegistryManifest(session)
    registry.initialize()
    route = AgentRoute("connection", "incarnation")
    registry.register_agent("agent", route)
    with pytest.raises(LiveAgentUnavailableError):
        await registry.read_live_page("agent")

    calls: list[tuple[str, str | None, int]] = []
    page = AgentTranscriptGetResponse(
        state=AgentTranscriptState.AVAILABLE, entries=[], has_more=False
    )

    async def reader(
        agent: str, before: str | None, limit: int
    ) -> AgentTranscriptGetResponse:
        calls.append((agent, before, limit))
        return page

    registry.register_agent("agent", route, reader)
    assert await registry.read_live_page("agent", before="cursor", limit=3) is page
    assert calls == [("agent", "cursor", 3)]
    registry.remove_agent("agent")
    with pytest.raises(LiveAgentUnavailableError):
        await registry.read_live_page("agent")


def test_new_incarnation_retires_callback(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = RegistryManifest(session)
    registry.initialize()
    registry.register_agent("agent", AgentRoute("connection", "first"))
    registry.route_callback("callback", "agent")
    registry.register_agent("agent", AgentRoute("connection", "second"))
    with pytest.raises(LiveAgentUnavailableError):
        registry.callback_route("callback")
    assert json.loads(registry.path.read_text(encoding="utf-8"))["callbacks"] == {}


def test_delivery_target_distinguishes_routed_from_unrouted(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = RegistryManifest(session)
    registry.initialize()
    assert registry.delivery_target("callback") is None
    route = AgentRoute("connection", "incarnation")
    registry.register_agent("agent", route)
    registry.route_callback("callback", "agent")
    assert registry.delivery_target("callback") == CallbackTarget("agent", route)


def test_retired_route_keeps_the_prompt_undeliverable(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = RegistryManifest(session)
    registry.initialize()
    registry.register_agent("agent", AgentRoute("connection", "first"))
    registry.route_callback("callback", "agent")

    registry.remove_agent("agent")

    assert json.loads(registry.path.read_text(encoding="utf-8")) == {
        "agents": {},
        "callbacks": {},
    }
    with pytest.raises(LiveAgentUnavailableError):
        registry.delivery_target("callback")
    # A later registration of the agent does not resurrect the retired prompt.
    registry.register_agent("agent", AgentRoute("connection", "second"))
    with pytest.raises(LiveAgentUnavailableError):
        registry.delivery_target("callback")
    assert registry.delivery_target("other-callback") is None


def test_failed_registration_restores_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = RegistryManifest(session)
    registry.initialize()
    original = AgentRoute("connection", "first")
    registry.register_agent("agent", original)
    registry.route_callback("callback", "agent")

    def fail_write() -> None:
        raise OSError("write failed")

    monkeypatch.setattr(registry, "_write", fail_write)
    with pytest.raises(OSError, match="write failed"):
        registry.register_agent("agent", AgentRoute("connection", "second"))
    assert registry.callback_route("callback") == original
    assert registry.delivery_target("callback") == CallbackTarget("agent", original)


def test_initialize_does_not_modify_existing_fence(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = RegistryManifest(session)
    registry.initialize()
    registry.fence_path.write_text("sentinel", encoding="utf-8")
    before = registry.fence_path.stat()
    registry.initialize()
    after = registry.fence_path.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert registry.fence_path.read_text(encoding="utf-8") == "sentinel"
