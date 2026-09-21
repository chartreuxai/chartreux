from __future__ import annotations

from typing import TYPE_CHECKING

from chartreux.core.agents.models import BUILTIN_SUBAGENTS, AgentProfile, AgentType
from chartreux.core.agents.registry import AgentRegistry
from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    get_harness_files_manager,
)
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.utils import name_matches
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema


class AgentManager:
    """Discover subagents; primary-agent selection is intentionally unsupported."""

    def __init__(
        self,
        orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        harness_files: HarnessFilesManager | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._registry = AgentRegistry(
            orchestrator, harness_files or get_harness_files_manager()
        )
        if custom_names := [n for n in self._discovered if n not in BUILTIN_SUBAGENTS]:
            logger.info(
                "Discovered custom agents %s in %s",
                " ".join(custom_names),
                " ".join(str(p) for p in self._registry.search_paths),
            )

    def rebind(self, harness_files: HarnessFilesManager) -> None:
        self._registry.rediscover(harness_files)

    @property
    def _discovered(self) -> dict[str, AgentProfile]:
        return self._registry.discovered

    @property
    def config(self) -> ChartreuxConfigSchema:
        return self._orchestrator.config

    @property
    def available_agents(self) -> dict[str, AgentProfile]:
        return {
            name: profile
            for name, profile in self._discovered.items()
            if self._is_agent_available(name)
        }

    def _is_agent_available(self, name: str) -> bool:
        if enabled := self.config.enabled_agents:
            return name_matches(name, enabled)
        return not name_matches(name, self.config.disabled_agents)

    def get_agent(self, name: str) -> AgentProfile:
        if agent := self.available_agents.get(name):
            return agent
        raise ValueError(f"Agent '{name}' not found or is disabled")

    def resolve_launch_profile(self, name: str) -> AgentProfile:
        """Return an explicitly selected profile with a typed launch error."""
        from chartreux.core.subagents import MissingAgentProfileError

        try:
            return self.get_agent(name)
        except ValueError as exc:
            raise MissingAgentProfileError(
                "agent", f"Selected agent profile '{name}' is unavailable"
            ) from exc

    def get_subagents(self) -> list[AgentProfile]:
        return [
            agent
            for agent in self.available_agents.values()
            if agent.agent_type == AgentType.SUBAGENT
        ]
