from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from chartreux.app_server._resources import _inventory_item_states
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    ClientInfo,
    ConfigWriteOpWire,
    ConfigWriteParams,
    SessionStartParams,
    SettingsReadParams,
    SettingsReadResponse,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layer import ConfigLayerError
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.settings_catalog import (
    DEFERRED_SETTINGS,
    EDITABLE_BY_PATH,
    EDITABLE_SETTINGS,
    EXCLUDED_SETTINGS,
    LINK_SETTINGS,
    VISIBLE_SETTINGS,
)
from tests.stubs.app_server import build_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


@pytest.mark.parametrize(
    ("enabled", "disabled", "pattern_driven"),
    [
        (["bash"], ["BASH"], False),
        (["bash"], ["bas*"], True),
        (["bas*"], ["bash"], True),
        (["bas*"], ["b*"], True),
    ],
)
def test_tools_inventory_intersects_filters_without_changing_other_categories(
    enabled: list[str], disabled: list[str], pattern_driven: bool
) -> None:
    states = _inventory_item_states(
        {"tools": ["bash"], "skills": ["bash"], "agents": ["bash"]},
        {
            f"{side}_{category}": values
            for category in ("tools", "skills", "agents")
            for side, values in (("enabled", enabled), ("disabled", disabled))
        },
    )
    assert not states["tools"]["bash"].effective
    assert not states["tools"]["bash"].default_effective
    assert states["tools"]["bash"].pattern_driven is pattern_driven
    for category in ("skills", "agents"):
        assert states[category]["bash"].effective
        assert states[category]["bash"].pattern_driven is (enabled != ["bash"])


def test_curated_registry_is_complete_and_validated() -> None:
    assert len(EDITABLE_SETTINGS) == len(EDITABLE_BY_PATH) == 41
    assert "ascii_chrome" in EDITABLE_BY_PATH
    assert all(
        item.label and item.description and item.empty for item in EDITABLE_SETTINGS
    )
    assert all(
        item.timing and isinstance(item.timing_verified, bool)
        for item in EDITABLE_SETTINGS
    )
    assert EDITABLE_BY_PATH["subagents.max_idle_agents"].minimum == 0
    assert EDITABLE_BY_PATH["subagents.idle_ttl_seconds"].minimum == 0
    assert "subagents.max_running_subagents" not in EDITABLE_BY_PATH
    assert "subagents.max_running_subagents" not in {
        item.path for item in VISIBLE_SETTINGS
    }
    assert {item.path for item in EDITABLE_SETTINGS}.isdisjoint(EXCLUDED_SETTINGS)
    assert DEFERRED_SETTINGS == ()
    assert {item.command for item in LINK_SETTINGS} == {
        "/theme",
        "/log-level",
        "/mcp",
        "/providers",
        "/web-search",
        "/proxy-setup",
    }
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError):
            EDITABLE_BY_PATH["api_timeout"].validate(bad)
    EDITABLE_BY_PATH["api_timeout"].validate(0.0)
    EDITABLE_BY_PATH["project_context.timeout_seconds"].validate(0.0)
    with pytest.raises(ValueError):
        EDITABLE_BY_PATH["subagents.max_idle_agents"].validate(True)


@pytest.mark.asyncio
async def test_missing_user_layer_is_view_only(monkeypatch: pytest.MonkeyPatch) -> None:
    user = UserConfigLayer(path=Path("missing-settings.toml"), name="user")
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
        await client.initialize(ClientInfo(name="settings-no-user", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())

        async def fail_user_load(self: UserConfigLayer, *, force: bool = False):
            raise ConfigLayerError(self.name, "user file unavailable")

        monkeypatch.setattr(UserConfigLayer, "load", fail_user_load)
        response = SettingsReadResponse.model_validate(
            await client.request(
                "config/settings/read", SettingsReadParams(session_id=loop.session_id)
            )
        )
        assert response.view_only
        assert response.user_layer == "user"
        assert all(not field.saved_explicit for field in response.fields)
        assert {field.origin for field in response.fields} == {"live config"}
    finally:
        await a.close()
        await b.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_non_user_layer_build_error_falls_back_to_live_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), overlay],
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
        await client.initialize(ClientInfo(name="settings-layer-error", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())

        async def fail_build(self: DefaultConfigLayer, *, force: bool = False):
            raise ConfigLayerError(self.name, "malformed config")

        monkeypatch.setattr(DefaultConfigLayer, "load", fail_build)
        response = SettingsReadResponse.model_validate(
            await client.request(
                "config/settings/read", SettingsReadParams(session_id=loop.session_id)
            )
        )
        fields = {field.path: field for field in response.fields}
        assert (
            fields["show_greeting"].effective_value
            is loop.config_orchestrator.config.show_greeting
        )
        assert {field.origin for field in response.fields} == {"live config"}
    finally:
        await a.close()
        await b.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_snapshot_pairs_force_loaded_revision_with_sparse_leaf_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import chartreux.core.prompts as prompts

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "my-prompt.md").write_text("Custom")
    monkeypatch.setattr(
        prompts,
        "get_harness_files_manager",
        lambda: SimpleNamespace(
            project_prompts_dirs=[prompts_dir], user_prompts_dirs=[]
        ),
    )
    source = tmp_path / "config.toml"
    source.write_text("[session_logging]\nenabled = false\n")
    user = UserConfigLayer(path=source, name="renamed-user")
    overlay = OverridesLayer(data={"session_logging": {"generate_titles": True}})
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
        await client.initialize(ClientInfo(name="settings", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())
        source.write_text("[session_logging]\nenabled = true\n")
        raw = await client.request(
            "config/settings/read", SettingsReadParams(session_id=loop.session_id)
        )
        response = SettingsReadResponse.model_validate(raw)
        assert [item.path for item in response.catalog] == [
            item.path
            for item in (*VISIBLE_SETTINGS, *DEFERRED_SETTINGS, *LINK_SETTINGS)
        ]
        fields = {item.path: item for item in response.fields}
        catalog = {item.path: item for item in response.catalog}
        assert catalog["system_prompt_id"].choices == tuple(
            sorted((
                "cli",
                "explore",
                "tests",
                "minimal",
                "worker",
                "advisor",
                "reviewer",
                "my-prompt",
            ))
        )
        assert catalog["compaction_prompt_id"].choices == ("compact", "my-prompt")
        assert set(response.inventories) == {"tools", "skills", "agents"}
        assert response.inventories["tools"] == loop.tool_manager.settings_inventory
        assert response.inventories["skills"] == loop.skill_manager.settings_inventory
        assert response.inventories["agents"] == loop.agent_manager.settings_inventory
        assert "bash" in response.inventories["tools"]
        assert response.inventories["skills"]
        assert response.inventories["agents"]
        for category in ("tools", "skills", "agents"):
            descriptor = catalog[f"inventory_{category}"]
            assert descriptor.control == "toggle_inventory"
            assert descriptor.inventory == category
            assert f"enabled_{category}" in fields
            assert f"disabled_{category}" in fields
            assert f"enabled_{category}" not in catalog
            assert f"disabled_{category}" not in catalog
        for path in ("agent_paths", "skill_paths", "tool_paths"):
            assert catalog[path].control is None
        assert fields["enabled_tools"].effective_value == []
        assert response.user_revision != user.fingerprint
        assert fields["session_logging.enabled"].effective_value is True
        assert fields["session_logging.enabled"].saved_value is True
        assert fields["session_logging.enabled"].origin == "renamed-user"
        assert fields["session_logging.generate_titles"].origin == "overrides"
        assert fields["session_logging.generate_titles"].saved_explicit is False
        assert tomllib.loads(source.read_text())["session_logging"]["enabled"] is True
        saved = await client.request(
            "config/write",
            ConfigWriteParams(
                session_id=loop.session_id,
                target="user",
                expected_revision=response.user_revision,
                ops=[
                    ConfigWriteOpWire(
                        op="set",
                        path="/session_logging/generate_titles",
                        value=True,
                        target_layer="renamed-user",
                    )
                ],
            ),
        )
        assert saved["persistence"] == "saved"
        persisted = tomllib.loads(source.read_text())
        assert persisted["session_logging"] == {
            "enabled": True,
            "generate_titles": True,
        }
        updated = SettingsReadResponse.model_validate(
            await client.request(
                "config/settings/read", SettingsReadParams(session_id=loop.session_id)
            )
        )
        assert updated.user_revision == saved["revision"]
        assert next(
            field
            for field in updated.fields
            if field.path == "session_logging.generate_titles"
        ).saved_explicit
        removed = await client.request(
            "config/write",
            ConfigWriteParams(
                session_id=loop.session_id,
                target="user",
                expected_revision=updated.user_revision,
                ops=[
                    ConfigWriteOpWire(
                        op="remove",
                        path="/session_logging/generate_titles",
                        target_layer="renamed-user",
                    )
                ],
            ),
        )
        assert removed["persistence"] == "saved"
        assert tomllib.loads(source.read_text())["session_logging"] == {"enabled": True}
        stale = await client.request(
            "config/write",
            ConfigWriteParams(
                session_id=loop.session_id,
                target="user",
                expected_revision=updated.user_revision,
                ops=[ConfigWriteOpWire(op="set", path="/show_greeting", value=False)],
            ),
        )
        assert stale["persistence"] == "not_saved"
        assert stale["failures"] == ["conflict"]
        assert "show_greeting" not in tomllib.loads(source.read_text())
        list_save = await client.request(
            "config/write",
            ConfigWriteParams(
                session_id=loop.session_id,
                target="user",
                expected_revision=removed["revision"],
                ops=[
                    ConfigWriteOpWire(
                        op="set",
                        path="/enabled_tools",
                        value=["tool-*", "re:^custom_"],
                        target_layer="renamed-user",
                    )
                ],
            ),
        )
        assert list_save["persistence"] == "saved"
        assert tomllib.loads(source.read_text())["enabled_tools"] == [
            "tool-*",
            "re:^custom_",
        ]
        list_read = SettingsReadResponse.model_validate(
            await client.request(
                "config/settings/read", SettingsReadParams(session_id=loop.session_id)
            )
        )
        assert next(
            field for field in list_read.fields if field.path == "enabled_tools"
        ).saved_value == ["tool-*", "re:^custom_"]
        assert "bash" in list_read.inventories["tools"]
    finally:
        await a.close()
        await b.close()
        await loop.aclose()
