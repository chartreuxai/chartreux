from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.session.session_lease import SessionBusyError, SessionLease
from chartreux.core.tools.mcp import MCPRegistry
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


async def _real_orchestrator() -> ConfigOrchestrator[ChartreuxConfigSchema]:
    default = DefaultConfigLayer(schema=ChartreuxConfigSchema)
    layer = OverridesLayer(data={})

    def default_layer_resolver() -> ConfigLayer[RawConfig]:
        return layer

    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[default, layer],
        default_layer_resolver=default_layer_resolver,
    )


@pytest.mark.asyncio
async def test_fork_supports_implicit_target_set_field_on_forked_loop() -> None:
    orchestrator = await _real_orchestrator()
    assert orchestrator.config.auto_compact_threshold != 12_345

    agent = AgentLoop(
        orchestrator, backend=FakeBackend(), mcp_registry=FakeMCPRegistry()
    )
    forked = await AgentRuntimeFactory().fork(agent, None)
    try:
        failures = await forked.config_orchestrator.set_field(
            "/auto_compact_threshold", 12_345
        )

        assert failures == []
        assert forked.config_orchestrator.config.auto_compact_threshold == 12_345
        assert agent.config_orchestrator.config.auto_compact_threshold != 12_345
    finally:
        await forked.aclose()
        await agent.aclose()


@pytest.mark.parametrize("derived_kind", ["fork", "child"])
@pytest.mark.asyncio
async def test_derived_runtime_holds_the_shared_session_lease(
    derived_kind: str, tmp_path: Path
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path), session_prefix="session"
        )
    )
    agent = AgentLoop(
        FakeConfigOrchestrator(config),
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )
    factory = AgentRuntimeFactory()
    derived = (
        await factory.fork(agent, None)
        if derived_kind == "fork"
        else await factory.create_child(agent, "worker")
    )

    try:
        with pytest.raises(SessionBusyError):
            SessionLease(tmp_path, derived.session_id).acquire()
    finally:
        await derived.aclose()
        await agent.aclose()

    SessionLease(tmp_path, derived.session_id).acquire().release()


@pytest.mark.parametrize("derived_kind", ["fork", "child"])
@pytest.mark.asyncio
async def test_derived_runtime_clones_preconfigured_mcp_registry_before_init(
    derived_kind: str, tmp_path: Path
) -> None:
    registry = MCPRegistry(descriptor_cache_root=tmp_path / "descriptors")
    agent = AgentLoop(
        FakeConfigOrchestrator(build_test_vibe_config()),
        backend=FakeBackend(),
        mcp_registry=registry,
    )
    factory = AgentRuntimeFactory()

    derived = (
        await factory.fork(agent, None)
        if derived_kind == "fork"
        else await factory.create_child(agent, "worker")
    )

    try:
        assert derived.mcp_registry is not None
        assert derived.mcp_registry is not registry
        assert derived.mcp_registry._descriptor_cache_root == tmp_path / "descriptors"
    finally:
        await derived.aclose()
        await agent.aclose()
