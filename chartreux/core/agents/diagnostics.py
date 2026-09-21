from __future__ import annotations

from typing import TYPE_CHECKING

from chartreux.core.utils import name_matches

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema


def excluded_agent_message(name: str, config: ChartreuxConfigSchema) -> str:
    """Generate a message explaining why an agent is not available based on the config."""
    label = "Agent"
    fix = "select an enabled agent"
    if enabled := config.enabled_agents:
        if not name_matches(name, enabled):
            return (
                f"{label} '{name}' is not in 'enabled_agents' {enabled}. "
                f"Add '{name}' to 'enabled_agents', or {fix}."
            )
    elif name_matches(name, config.disabled_agents):
        return (
            f"{label} '{name}' is in 'disabled_agents' "
            f"{config.disabled_agents}. Remove '{name}' from "
            f"'disabled_agents', or {fix}."
        )
    return f"Agent '{name}' is not available. It may be disabled or excluded by your config."
