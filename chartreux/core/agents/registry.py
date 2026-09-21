from __future__ import annotations

from collections.abc import Mapping
import copy
from pathlib import Path
from typing import TYPE_CHECKING, cast

from chartreux.core.agents.models import BUILTIN_SUBAGENTS, AgentProfile
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.config.layers.agent_profile import AgentProfileLayer
from chartreux.core.config.layers.launch_overrides import LaunchOverridesLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.paths import dedup_paths
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema


def apply_profile_overrides(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    overrides: dict[str, object],
) -> None:
    """Install profile overrides before the first runtime override layer."""
    layers = orchestrator.layers
    profile = next(
        (
            (index, layer)
            for index, layer in enumerate(layers)
            if isinstance(layer, AgentProfileLayer)
        ),
        None,
    )
    if profile is None:
        profile_layer = AgentProfileLayer(data=overrides)
        profile_index = None
    else:
        profile_index, current_layer = profile
        profile_layer = AgentProfileLayer(data=overrides, name=current_layer.name)
        orchestrator.remove_layer(profile_index)

    insertion_index = next(
        (
            index
            for index, layer in enumerate(orchestrator.layers)
            if isinstance(layer, OverridesLayer)
        ),
        profile_index if profile_index is not None else len(orchestrator.layers),
    )
    orchestrator.insert_layer(profile_layer, insertion_index)
    orchestrator.rebuild()


def _mutable_copy(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _mutable_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_copy(item) for item in value]
    return copy.deepcopy(value)


def apply_launch_overrides(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], data: dict[str, object]
) -> None:
    """Install one child-only launch layer after runtime overrides."""
    layers = orchestrator.layers
    current = next(
        (
            (index, layer)
            for index, layer in enumerate(layers)
            if isinstance(layer, LaunchOverridesLayer)
        ),
        None,
    )
    launch_layer = LaunchOverridesLayer(
        data=cast(dict[str, object], _mutable_copy(data)),
        name=current[1].name if current is not None else LaunchOverridesLayer.NAME,
    )
    if current is not None:
        orchestrator.remove_layer(current[0])
    insertion_index = max(
        (
            index + 1
            for index, layer in enumerate(orchestrator.layers)
            if isinstance(layer, OverridesLayer)
        ),
        default=len(orchestrator.layers),
    )
    orchestrator.insert_layer(launch_layer, insertion_index)
    orchestrator.rebuild()


def build_child_orchestrator(
    source: ConfigOrchestrator[ChartreuxConfigSchema],
    profile_overrides: dict[str, object] | None,
    launch_overrides: Mapping[str, object],
) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    """Build the authoritative resource-free child configuration assembly.

    ``None`` preserves the source's captured profile layer for retained agents.
    """
    candidate = source._copy_for_child()
    if profile_overrides is not None:
        apply_profile_overrides(candidate, profile_overrides)
    apply_launch_overrides(candidate, dict(launch_overrides))
    return candidate


class AgentRegistry:
    """Discovers and parses agent-profile definitions from the filesystem.

    Owns search-path resolution and TOML loading. Each custom
    profile is validated by folding it onto a throwaway orchestrator copy, so a
    broken definition is dropped at discovery rather than at selection.
    """

    def __init__(
        self,
        orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        harness_files: HarnessFilesManager,
    ) -> None:
        self._orchestrator = orchestrator
        self._harness_files = harness_files
        self.search_paths = self._compute_search_paths()
        self.discovered = self._discover()

    def rediscover(self, harness_files: HarnessFilesManager) -> None:
        """Point discovery at a different set of project directories.

        Search paths are resolved once at construction, so a session that has
        moved keeps offering the agents of the directory it left until this runs.
        """
        self._harness_files = harness_files
        self.search_paths = self._compute_search_paths()
        self.discovered = self._discover()

    def _compute_search_paths(self) -> list[Path]:
        mgr = self._harness_files
        return dedup_paths([
            *(p for p in self._orchestrator.config.agent_paths if p.is_dir()),
            *mgr.project_agents_dirs,
            *mgr.user_agents_dirs,
        ])

    def _discover(self) -> dict[str, AgentProfile]:
        agents: dict[str, AgentProfile] = dict(BUILTIN_SUBAGENTS)
        custom_names: set[str] = set()

        for base in self.search_paths:
            if not base.is_dir():
                continue
            for agent_file in base.glob("*.toml"):
                if not agent_file.is_file():
                    continue
                if (
                    agent := self._try_load(agent_file, self._orchestrator.copy())
                ) is not None:
                    if agent.name in custom_names:
                        logger.debug(
                            "Skipping duplicate agent '%s' at %s",
                            agent.name,
                            agent_file,
                        )
                        continue
                    custom_names.add(agent.name)
                    if agent.name in BUILTIN_SUBAGENTS:
                        logger.info(
                            "Custom agent '%s' overrides builtin agent", agent.name
                        )
                    agents[agent.name] = agent

        return agents

    def _try_load(
        self, agent_file: Path, candidate: ConfigOrchestrator[ChartreuxConfigSchema]
    ) -> AgentProfile | None:
        try:
            agent = AgentProfile.from_toml(agent_file)
            apply_profile_overrides(candidate, agent.overrides)
            return agent
        except Exception as e:
            logger.warning("Failed to load agent at %s: %s", agent_file, e)
            return None
