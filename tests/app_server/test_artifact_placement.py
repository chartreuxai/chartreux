from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory, RootOpenRequest
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import ClientCapabilities, ClientInfo, SessionOptions
from chartreux.app_server.registry_manifest import (
    AgentRoute,
    RegistryManifestInitializationError,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import create_legacy_app_server


@pytest.mark.asyncio
async def test_session_start_and_resume_artifact_placement(tmp_path: Path) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        )
    )
    loop = build_test_agent_loop(cwd=tmp_path, config=config)
    directory = loop.session_logger.session_dir
    assert directory is not None
    agents = directory / "agents"
    agents.mkdir(parents=True)
    transcript = agents / "existing"
    transcript.write_text("existing", encoding="utf-8")
    before = {path for path in tmp_path.rglob("*") if path.is_file()}

    async def open_root(_request: RootOpenRequest) -> AgentLoop:
        return loop

    client_transport, server_transport = memory_transport_pair()
    factory = AgentRuntimeFactory()
    server = create_legacy_app_server(
        server_transport, open_root=open_root, runtime_factory=factory
    )
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="placement-test", version="1"),
        capabilities=ClientCapabilities(),
        session_options=SessionOptions(cwd=str(tmp_path)),
    )
    try:
        registry = factory.registry_manifests[loop]
        assert registry.path.parent == directory.parent
        assert registry.fence_path.parent == directory
        assert {path for path in tmp_path.rglob("*") if path.is_file()} - before == {
            registry.path,
            registry.fence_path,
        }
        fence = registry.fence_path.stat()
        await loop.persist_empty_session()
        await factory.resume_root(loop, loop.session_id)
        assert registry.fence_path.stat().st_ino == fence.st_ino
        assert registry.fence_path.stat().st_mtime_ns == fence.st_mtime_ns
        assert transcript.read_text(encoding="utf-8") == "existing"
    finally:
        await session.close()
        await client.close()


@pytest.mark.asyncio
async def test_warm_resume_manifest_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        )
    )
    loop = build_test_agent_loop(cwd=tmp_path, config=config)
    try:
        await loop.persist_empty_session()
        directory = loop.session_logger.session_dir
        assert directory is not None
        factory = AgentRuntimeFactory()
        factory.initialize_registry_manifest(loop, directory)
        registry = factory.registry_manifests[loop]
        route = AgentRoute("connection", "incarnation")
        registry.register_agent("agent", route)
        registry.route_callback("callback", "agent")
        original = registry.path.read_bytes()
        session_id = loop.session_id

        def fail_write() -> None:
            raise OSError("write failed")

        monkeypatch.setattr(registry, "_write", fail_write)
        with pytest.raises(RegistryManifestInitializationError):
            await factory.resume_root(loop, session_id)
        assert loop.session_id == session_id
        assert factory.registry_manifests[loop] is registry
        assert registry.callback_route("callback") == route
        assert registry.path.read_bytes() == original
    finally:
        await loop.aclose()
