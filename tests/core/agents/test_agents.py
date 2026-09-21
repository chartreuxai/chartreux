from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.agents.manager import AgentManager
from chartreux.core.agents.models import (
    BUILTIN_SUBAGENTS,
    WORKER,
    AgentSafety,
    AgentType,
)
from chartreux.core.agents.registry import AgentRegistry
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.harness_files import HarnessFilesManager
from tests.conftest import ConfigBuilder, OrchestratorLoader


class TestAgentProfile:
    def test_worker_agent_is_subagent(self) -> None:
        assert WORKER.agent_type == AgentType.SUBAGENT

    def test_worker_agent_has_neutral_safety(self) -> None:
        assert WORKER.safety == AgentSafety.NEUTRAL

    def test_worker_agent_has_no_overrides(self) -> None:
        assert WORKER.overrides == {}
        assert WORKER.instructions is None

    def test_builtin_subagents_contains_worker(self) -> None:
        assert BUILTIN_SUBAGENTS["worker"] is WORKER


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

    def test_get_subagents_includes_worker(self, manager: AgentManager) -> None:
        assert "worker" in [agent.name for agent in manager.get_subagents()]

    def test_get_builtin_subagent(self, manager: AgentManager) -> None:
        assert manager.get_agent("worker") is WORKER

    def test_get_nonexistent_agent_raises(self, manager: AgentManager) -> None:
        with pytest.raises(ValueError, match="not found"):
            manager.get_agent("nonexistent-agent")
