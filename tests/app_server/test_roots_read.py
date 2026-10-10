from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    ClientInfo,
    RootsReadParams,
    RootsReadResponse,
    SessionStartParams,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from tests.stubs.app_server import build_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


@pytest.mark.asyncio
async def test_roots_read_projects_saved_effective_and_dynamic_roots(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    source = config_dir / "config.toml"
    user = UserConfigLayer(path=source, name="user")
    overlay = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, overlay],
        default_layer_resolver=lambda: overlay,
    )
    loop = AgentLoop(
        config_orchestrator=orchestrator,
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )
    a, b = memory_transport_pair()
    server = build_test_app_server(loop, b)
    client = AppServerClient(a, run_peer=server.serve)
    try:
        await client.initialize(ClientInfo(name="roots", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())
        granted = tmp_path / "session-grant"
        granted.mkdir()
        loop._session_root_grants.grant(granted)
        raw = await client.request(
            "policy/roots/read", RootsReadParams(session_id=loop.session_id)
        )
        response = RootsReadResponse.model_validate(raw)
        assert response.project == str(project)
        assert response.effective_roots == [str(granted.resolve())]
        assert response.saved_roots == []
        assert response.user_revision
        assert response.roots == []
        revision = response.revision

        source.write_text(
            f'[authorized_roots_by_project]\n"{project}" = ["{granted}"]\n'
        )
        refreshed = RootsReadResponse.model_validate(
            await client.request(
                "policy/roots/read", RootsReadParams(session_id=loop.session_id)
            )
        )
        assert refreshed.saved_roots == [str(granted.resolve())]
        assert refreshed.user_revision != response.user_revision
        assert refreshed.revision == revision

        replaced = await client.request(
            "policy/roots/replace",
            {
                "sessionId": loop.session_id,
                "expectedRevision": revision,
                "scope": "session",
                "userInitiated": True,
                "roots": [],
            },
        )
        assert replaced["revision"]
    finally:
        await client.close()
        await loop.aclose()
