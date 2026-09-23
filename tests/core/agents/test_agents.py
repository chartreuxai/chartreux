from __future__ import annotations

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
from tests.conftest import ConfigBuilder, OrchestratorLoader


class TestAgentProfile:
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
            (WORKER, "small-worker", "worker", None, None, None),
            (
                ADVISOR,
                "advisor",
                "advisor",
                ["read_file", "grep", "web_search", "web_fetch"],
                0,
                None,
            ),
            (REVIEWER, "small-reviewer", "reviewer", None, None, None),
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
        profile_path.write_text('role = "small-worker"\n', encoding="utf-8")

        profile = AgentProfile.from_toml(profile_path)

        assert profile.role == "small-worker"
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
    def test_user_profile_overrides_builtin_role_profile(
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
        manager = AgentManager(
            load_orchestrator(build_config(agent_paths=[agents_dir]))
        )

        assert manager.get_agent(name).description == "Custom role profile"

    def test_get_builtin_subagent(self, manager: AgentManager) -> None:
        assert manager.get_agent("worker") is WORKER

    def test_get_nonexistent_agent_raises(self, manager: AgentManager) -> None:
        with pytest.raises(ValueError, match="not found"):
            manager.get_agent("nonexistent-agent")
