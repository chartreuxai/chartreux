from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chartreux.core.agents.manager import AgentManager
from chartreux.core.agents.models import (
    ADVISOR,
    BUILTIN_SUBAGENTS,
    REVIEWER,
    WORKER,
    AgentProfile,
    AgentSafety,
    AgentType,
)
from chartreux.core.agents.registry import AgentRegistry
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.dispatch.presets import STANDALONE_PRESET
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot, load_catalog
from tests.conftest import ConfigBuilder, OrchestratorLoader


class TestAgentProfile:
    @pytest.mark.parametrize("name", ["agent-0", "agent-1", "agent-123"])
    def test_instance_handle_names_are_reserved(
        self, tmp_path: Path, name: str
    ) -> None:
        path = tmp_path / f"{name}.toml"
        path.write_text('agent_type = "subagent"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="reserved for agent instance handles"):
            AgentProfile.from_toml(path)

    @pytest.mark.parametrize("name", ["agent-helper", "agent-1-extra"])
    def test_non_handle_profile_names_keep_internal_enum(
        self, tmp_path: Path, name: str
    ) -> None:
        path = tmp_path / f"{name}.toml"
        path.write_text('agent_type = "subagent"\n', encoding="utf-8")
        profile = AgentProfile.from_toml(path)
        assert profile.name == name
        assert profile.agent_type is AgentType.SUBAGENT

    @pytest.mark.parametrize(
        (
            "profile",
            "role",
            "prompt_id",
            "enabled_tools",
            "idle_ttl_seconds",
            "thinking",
        ),
        [
            (WORKER, "medium", "worker", None, None, None),
            (
                ADVISOR,
                "large",
                "advisor",
                ["read_file", "grep", "web_search", "web_fetch", "skill"],
                0,
                None,
            ),
            (REVIEWER, "medium", "reviewer", None, None, None),
        ],
    )
    def test_builtin_profiles_have_role_prompt_and_tools(
        self,
        profile: AgentProfile,
        role: str,
        prompt_id: str,
        enabled_tools: list[str] | None,
        idle_ttl_seconds: int | None,
        thinking: dict[str, str] | None,
    ) -> None:
        assert profile.agent_type == AgentType.SUBAGENT
        assert profile.safety == AgentSafety.NEUTRAL
        assert profile.role == role
        assert profile.idle_ttl_seconds == idle_ttl_seconds
        assert profile.overrides["system_prompt_id"] == prompt_id
        assert profile.overrides.get("enabled_tools") == enabled_tools
        assert profile.overrides.get("thinking_overrides") == thinking

    def test_profile_idle_ttl_is_metadata_not_an_override(self, tmp_path: Path) -> None:
        profile_path = tmp_path / "persistent.toml"
        profile_path.write_text("idle_ttl_seconds = 0\n", encoding="utf-8")

        profile = AgentProfile.from_toml(profile_path)

        assert profile.idle_ttl_seconds == 0
        assert profile.overrides == {}

    def test_profile_role_is_metadata_not_an_override(self, tmp_path: Path) -> None:
        profile_path = tmp_path / "worker.toml"
        profile_path.write_text('role = "custom-worker"\n', encoding="utf-8")

        profile = AgentProfile.from_toml(profile_path)

        assert profile.role == "custom-worker"
        assert profile.overrides == {}

    def test_profile_role_with_at_sign_is_rejected(self, tmp_path: Path) -> None:
        profile_path = tmp_path / "worker.toml"
        profile_path.write_text('role = "worker@fast"\n', encoding="utf-8")

        with pytest.raises(ValueError, match="without '@'"):
            AgentProfile.from_toml(profile_path)

    def test_profile_active_model_is_rejected_with_role_guidance(
        self, tmp_path: Path
    ) -> None:
        profile_path = tmp_path / "worker.toml"
        profile_path.write_text('active_model = "small"\n', encoding="utf-8")

        with pytest.raises(ValueError, match="use role instead"):
            AgentProfile.from_toml(profile_path)

    def test_builtin_subagents_contains_role_profiles(self) -> None:
        assert BUILTIN_SUBAGENTS == {
            "worker": WORKER,
            "advisor": ADVISOR,
            "reviewer": REVIEWER,
        }


class TestAgentManager:
    @pytest.fixture
    def manager(
        self,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> AgentManager:
        return AgentManager(load_orchestrator(build_config()))

    def test_registry_loads_and_rediscovers_without_migrating_profile_files(
        self,
        tmp_path: Path,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> None:
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        old_profile = agents_dir / "old-profile.toml"
        original = b'display_name = "Old Profile"\n'
        old_profile.write_bytes(original)
        registry = AgentRegistry(
            load_orchestrator(build_config(agent_paths=[agents_dir])),
            HarnessFilesManager(sources=()),
        )
        rediscovered = agents_dir / "rediscovered.toml"
        rediscovered.write_text(
            'display_name = "Rediscovered"\ndescription = "A test profile"\n',
            encoding="utf-8",
        )
        registry.rediscover(HarnessFilesManager(sources=()))
        assert "rediscovered" in registry.discovered
        assert old_profile.read_bytes() == original

    def test_builtin_explore_is_absent_but_custom_explore_is_discoverable(
        self,
        tmp_path: Path,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> None:
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "explore.toml").write_text(
            'agent_type = "subagent"\ndescription = "Custom explore profile"\n',
            encoding="utf-8",
        )
        manager = AgentManager(
            load_orchestrator(build_config(agent_paths=[agents_dir]))
        )

        assert "explore" not in BUILTIN_SUBAGENTS
        assert manager.get_agent("explore").description == "Custom explore profile"

    def test_get_subagents_includes_role_profiles(self, manager: AgentManager) -> None:
        assert {agent.name for agent in manager.get_subagents()} >= {
            "worker",
            "advisor",
            "reviewer",
        }

    @pytest.mark.parametrize("idle_ttl_seconds", ["-1", '"invalid"', "true"])
    def test_invalid_profile_idle_ttl_is_rejected_at_discovery(
        self,
        tmp_path: Path,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
        idle_ttl_seconds: str,
    ) -> None:
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "invalid.toml").write_text(
            f"idle_ttl_seconds = {idle_ttl_seconds}\n", encoding="utf-8"
        )

        manager = AgentManager(
            load_orchestrator(build_config(agent_paths=[agents_dir]))
        )

        with pytest.raises(ValueError, match="not found"):
            manager.get_agent("invalid")

    @pytest.mark.parametrize("name", ["worker", "advisor", "reviewer"])
    def test_user_profile_cannot_shadow_builtin_at_activation(
        self,
        name: str,
        tmp_path: Path,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> None:
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / f"{name}.toml").write_text(
            'description = "Custom role profile"\n', encoding="utf-8"
        )
        orchestrator = load_orchestrator(build_config(agent_paths=[agents_dir]))
        manager = AgentManager(orchestrator)
        assert manager.get_agent(name) is BUILTIN_SUBAGENTS[name]
        assert any("S2" in note for note in orchestrator.config.validation_warnings)
        snapshot = orchestrator.config.catalog_snapshot
        assert snapshot.dispatch is STANDALONE_PRESET
        assert any(
            "S2" in item and name in item for item in snapshot.dispatch_diagnostics
        )

    def test_dispatch_activation_warning_is_projected_and_recomputed(
        self,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> None:
        from chartreux.app_server._projection import project_config_view

        orchestrator = load_orchestrator(build_config())
        snapshot = orchestrator.config.catalog_snapshot
        policy = snapshot.dispatch.model_copy(
            update={
                "contrasts": "Use slot `mechanical` for purpose `implementation`; use slot `implementor` for purpose `implementation`."
            }
        )
        orchestrator.config.attach_catalog_snapshot(replace(snapshot, dispatch=policy))
        registry = AgentRegistry(orchestrator, HarnessFilesManager(sources=()))
        warnings = project_config_view(orchestrator.config).validation_warnings
        assert any(note.startswith("R3 ") for note in warnings)
        assert orchestrator.config.catalog_snapshot.dispatch == policy
        orchestrator.config.attach_catalog_snapshot(snapshot)
        registry.validate_dispatch()
        assert not any(
            note.startswith("R3 ") for note in orchestrator.config.validation_warnings
        )

    def test_get_builtin_subagent(self, manager: AgentManager) -> None:
        assert manager.get_agent("worker") is WORKER

    @pytest.mark.parametrize("available", [False, True])
    def test_dispatch_profile_binding_uses_completed_discovery(
        self,
        tmp_path: Path,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
        available: bool,
    ) -> None:
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        if available:
            (agents_dir / "custom.toml").write_text('role = "small"\n')
        orchestrator = load_orchestrator(build_config(agent_paths=[agents_dir]))
        snapshot = orchestrator.config.catalog_snapshot
        slots = dict(snapshot.dispatch.slots)
        slots["mechanical"] = slots["mechanical"].model_copy(
            update={"profile": "custom"}
        )
        policy = snapshot.dispatch.model_copy(
            update={
                "slots": slots,
                "contrasts": "Use slot `mechanical` for searches; use slot `implementor` for implementation.",
            }
        )
        orchestrator.config.attach_catalog_snapshot(replace(snapshot, dispatch=policy))
        if available:
            registry = AgentRegistry(orchestrator, HarnessFilesManager(sources=()))
            assert (
                registry.discovered["custom"].source_path == agents_dir / "custom.toml"
            )
        else:
            AgentRegistry(orchestrator, HarnessFilesManager(sources=()))
            recovered = orchestrator.config.catalog_snapshot
            assert recovered.dispatch is STANDALONE_PRESET
            assert any(
                "R1 slots.mechanical.profile" in item and "custom" in item
                for item in recovered.dispatch_diagnostics
            )
            assert policy.slots["mechanical"].profile == "custom"

    @pytest.mark.parametrize("user_overlay", [False, True])
    def test_activation_falls_back_visibly_without_repair(
        self,
        tmp_path: Path,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
        monkeypatch: pytest.MonkeyPatch,
        user_overlay: bool,
    ) -> None:
        from unittest.mock import Mock

        warning = Mock()
        monkeypatch.setattr("chartreux.core.agents.registry.logger.warning", warning)
        orchestrator = load_orchestrator(build_config())
        path = tmp_path / "models.toml"
        original = b""
        if user_overlay:
            path.write_text(
                '[dispatch.slots.mechanical]\nprofile = "missing-profile"\n'
                '[providers.mistral]\napi_base = "https://preserved.example/v1"\n'
            )
            original = path.read_bytes()
            snapshot = load_catalog(path)
        else:
            catalog = SHIPPED_CATALOG.model_copy(update={"roles": {}})
            snapshot = CatalogSnapshot(catalog, "no-tier-roles")
        orchestrator.config.attach_catalog_snapshot(snapshot)

        AgentManager(orchestrator)

        recovered = orchestrator.config.catalog_snapshot
        assert recovered.dispatch is STANDALONE_PRESET
        assert recovered.catalog is snapshot.catalog
        assert recovered.overlaid_providers == snapshot.overlaid_providers
        assert recovered.revision == snapshot.revision
        diagnostic = recovered.dispatch_diagnostics[-1]
        assert "using the shipped standalone preset" in diagnostic
        warning.assert_called_with("%s", diagnostic)
        if user_overlay:
            assert "R1 slots.mechanical.profile" in diagnostic
            assert "dispatch config was bypassed" in diagnostic
            assert (
                recovered.catalog.providers["mistral"].api_base
                == "https://preserved.example/v1"
            )
            assert snapshot.dispatch.slots["mechanical"].profile == "missing-profile"
            assert path.read_bytes() == original
        else:
            assert "S1 slots.mechanical.role" in diagnostic
            assert recovered.catalog.roles == {}
            assert recovered.dispatch.slots["mechanical"].role == "@small"

    def test_get_nonexistent_agent_raises(self, manager: AgentManager) -> None:
        with pytest.raises(ValueError, match="not found"):
            manager.get_agent("nonexistent-agent")
