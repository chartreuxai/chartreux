from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tomli_w

from chartreux.app_server._config_write import config_write_targets
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    ClientInfo,
    ConfigWriteOpWire,
    ConfigWriteParams,
    SessionStartParams,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import build_default_orchestrator
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.trusted_folders import trusted_folders_manager
from tests.conftest import build_test_vibe_config
from tests.stubs.app_server import build_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


async def _write_model_ops(
    *, config: ChartreuxConfigSchema, ops: list[ConfigWriteOpWire]
) -> dict[str, Any]:
    client_transport, server_transport = memory_transport_pair()
    Path(UserConfigLayer().source_locator).write_text(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    )
    orchestrator = await build_default_orchestrator()
    agent_loop = AgentLoop(
        config_orchestrator=orchestrator,
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)

    try:
        await client.initialize(ClientInfo(name="config-write-test", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())
        response = await client.request(
            "config/write", ConfigWriteParams(session_id=agent_loop.session_id, ops=ops)
        )
    finally:
        await client_transport.close()
        await server_transport.close()
        await agent_loop.aclose()

    assert response["rejected"] is True
    assert response["persistence"] == "not_saved"
    assert agent_loop.config.available_models() == config.available_models()
    return response


async def _write_model_field(
    *, config: ChartreuxConfigSchema, alias: str, field: str, value: Any
) -> dict[str, Any]:
    models = await _write_model_ops(
        config=config,
        ops=[ConfigWriteOpWire(op="set", path=f"/models/{alias}/{field}", value=value)],
    )
    return models


def test_config_write_translation_preserves_order_paths_and_targets() -> None:
    from chartreux.app_server._config_write import config_write_ops_to_patches

    ops = [
        ConfigWriteOpWire(
            op="set",
            path="/models/a~1b~0c/thinking",
            value="low",
            target_layer="user-toml",
        ),
        ConfigWriteOpWire(
            op="remove", path="/models/a~1b~0c/thinking", target_layer="user-toml"
        ),
        ConfigWriteOpWire(
            op="set",
            path="/auto_compact_threshold",
            value=123,
            target_layer="overrides",
        ),
        ConfigWriteOpWire(op="set", path="", value={"theme": "dark"}),
    ]
    patches = config_write_ops_to_patches(ops)
    assert [p.path for p in patches] == [o.path for o in ops]
    assert [p.target_layer_name for p in patches] == [o.target_layer for o in ops]
    assert [p.to_json_patch() for p in patches] == [
        {"op": "add", "path": ops[0].path, "value": "low"},
        {"op": "remove", "path": ops[1].path},
        {"op": "add", "path": ops[2].path, "value": 123},
        {"op": "add", "path": "", "value": {"theme": "dark"}},
    ]


@pytest.mark.asyncio
async def test_session_catalog_batch_requires_explicit_user_save() -> None:
    models = await _write_model_ops(
        config=build_test_vibe_config(active_model="glm-5-3"),
        ops=[
            ConfigWriteOpWire(op="set", path="/models/glm-5-3/thinking", value="low"),
            ConfigWriteOpWire(op="set", path="/models/glm-5-3/temperature", value=0.7),
        ],
    )
    assert models["rejected"] is True


@pytest.mark.asyncio
async def test_session_catalog_field_requires_explicit_user_save() -> None:
    # Definitions belong to the user source, including changes to existing models.
    config = build_test_vibe_config(active_model="glm-5-3")
    persisted = await _write_model_field(
        config=config, alias="glm-5-3", field="thinking", value="low"
    )

    assert persisted["persistence"] == "not_saved"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_user", [False, True])
async def test_real_wire_targets_and_catalog_write_enforcement(
    tmp_path: Path, with_user: bool
) -> None:
    from chartreux.app_server.protocol import (
        ConfigFieldsReadParams,
        ConfigFieldsReadResponse,
    )
    from chartreux.core.agent_loop import AgentLoop
    from chartreux.core.config._catalog import CATALOG_DEFINITION_FIELDS
    from chartreux.core.config.layers.default import DefaultConfigLayer
    from chartreux.core.config.layers.overrides import OverridesLayer
    from chartreux.core.config.layers.user import UserConfigLayer
    from chartreux.core.config.orchestrator import ConfigOrchestrator
    from tests.stubs.fake_backend import FakeBackend
    from tests.stubs.fake_mcp_registry import FakeMCPRegistry

    user = UserConfigLayer(path=tmp_path / "config.toml", name="renamed-user")
    session = OverridesLayer(
        data={"session_logging": {"enabled": False, "generate_titles": False}}
    )
    orch = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[
            DefaultConfigLayer(schema=ChartreuxConfigSchema),
            *([user] if with_user else []),
            session,
        ],
        default_layer_resolver=lambda: session,
    )
    loop = AgentLoop(
        config_orchestrator=orch, backend=FakeBackend(), mcp_registry=FakeMCPRegistry()
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    try:
        await client.initialize(ClientInfo(name="catalog-wire", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())
        raw = await client.request(
            "config/fields/read", ConfigFieldsReadParams(session_id=loop.session_id)
        )
        response = ConfigFieldsReadResponse.model_validate(raw)
        for field in response.fields:
            expected = (
                (["renamed-user"] if with_user else [])
                if field.name in CATALOG_DEFINITION_FIELDS
                else ["overrides", *(["renamed-user"] if with_user else [])]
            )
            assert field.writable_targets == expected
        before = orch.config
        raw = await client.request(
            "config/write",
            ConfigWriteParams(
                session_id=loop.session_id,
                ops=[
                    ConfigWriteOpWire(
                        op="set", path="/models", value={}, target_layer="overrides"
                    )
                ],
            ),
        )
        assert raw["rejected"] is True
        assert orch.config is before
        assert not (tmp_path / "config.toml").exists()
    finally:
        await client_transport.close()
        await server_transport.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_config_write_targets_offer_user_project_and_session(
    config_dir: Path, tmp_working_directory: Path
) -> None:
    project_vibe_dir = tmp_working_directory / ".chartreux"
    project_vibe_dir.mkdir(parents=True, exist_ok=True)
    (project_vibe_dir / "config.toml").write_text('theme = "project-theme"\n')
    trusted_folders_manager.add_trusted(project_vibe_dir)

    orchestrator = await build_default_orchestrator()

    assert config_write_targets(orchestrator) == [
        "overrides",
        "user-toml",
        "project-toml",
    ]


@pytest.mark.asyncio
async def test_config_write_targets_skip_untrusted_project(
    config_dir: Path, tmp_working_directory: Path
) -> None:
    project_vibe_dir = tmp_working_directory / ".chartreux"
    project_vibe_dir.mkdir(parents=True, exist_ok=True)
    (project_vibe_dir / "config.toml").write_text('theme = "project-theme"\n')

    orchestrator = await build_default_orchestrator()

    assert config_write_targets(orchestrator) == ["overrides", "user-toml"]


@pytest.mark.asyncio
async def test_config_write_targets_skip_undiscovered_project(
    config_dir: Path, tmp_working_directory: Path
) -> None:
    orchestrator = await build_default_orchestrator()

    assert config_write_targets(orchestrator) == ["overrides", "user-toml"]
