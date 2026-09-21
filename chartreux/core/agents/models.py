from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any

from chartreux.agents import AgentSafety, AgentType


@dataclass(frozen=True)
class AgentProfile:
    name: str
    display_name: str
    description: str
    safety: AgentSafety
    agent_type: AgentType = AgentType.AGENT
    overrides: dict[str, Any] = field(default_factory=dict)
    instructions: str | None = None
    idle_ttl_seconds: int | None = None

    @classmethod
    def from_toml(cls, path: Path) -> AgentProfile:
        with path.open("rb") as f:
            data = tomllib.load(f)
        idle_ttl_seconds = data.pop("idle_ttl_seconds", None)
        if idle_ttl_seconds is not None and (
            isinstance(idle_ttl_seconds, bool)
            or not isinstance(idle_ttl_seconds, int)
            or idle_ttl_seconds < 0
        ):
            raise ValueError(
                "idle_ttl_seconds must be an integer greater than or equal to 0"
            )
        return cls(
            name=path.stem,
            display_name=data.pop("display_name", path.stem.replace("-", " ").title()),
            description=data.pop("description", ""),
            safety=AgentSafety(data.pop("safety", AgentSafety.NEUTRAL)),
            agent_type=AgentType(data.pop("agent_type", AgentType.AGENT)),
            instructions=data.pop("instructions", None),
            idle_ttl_seconds=idle_ttl_seconds,
            overrides=data,
        )


WORKER = AgentProfile(
    name="worker",
    display_name="Worker",
    description="General-purpose subagent for delegated tasks",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
    overrides={},
    instructions=None,
)

BUILTIN_SUBAGENTS: dict[str, AgentProfile] = {"worker": WORKER}
