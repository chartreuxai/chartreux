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
    role: str | None = None

    @classmethod
    def from_toml(cls, path: Path) -> AgentProfile:
        with path.open("rb") as f:
            data = tomllib.load(f)
        idle_ttl_seconds = data.pop("idle_ttl_seconds", None)
        role = data.pop("role", None)
        if "active_model" in data:
            raise ValueError(
                "active_model is not supported in agent profiles; use role instead"
            )
        if role is not None and (not isinstance(role, str) or not role or "@" in role):
            raise ValueError("role must be a non-empty role name without '@'")
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
            role=role,
            overrides=data,
        )


WORKER = AgentProfile(
    name="worker",
    display_name="Worker",
    description="General-purpose subagent for delegated tasks",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
    # Applied when the orchestrator explicitly escalates this profile to glm-5-3.
    overrides={"system_prompt_id": "worker", "thinking_overrides": {"glm-5-3": "high"}},
    instructions=None,
    role="small-worker",
)

ADVISOR = AgentProfile(
    name="advisor",
    display_name="Advisor",
    description="Independent perspective on architectural guidance and risks",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
    overrides={
        "system_prompt_id": "advisor",
        "enabled_tools": ["read_file", "grep", "web_search", "web_fetch"],
    },
    instructions=None,
    idle_ttl_seconds=0,
    role="advisor",
)

REVIEWER = AgentProfile(
    name="reviewer",
    display_name="Reviewer",
    description="Independent read-only review of code and plans",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
    overrides={
        "system_prompt_id": "reviewer",
        "thinking_overrides": {"glm-5-3": "high"},
    },
    instructions=None,
    role="medium-reviewer",
)

BUILTIN_SUBAGENTS: dict[str, AgentProfile] = {
    "worker": WORKER,
    "advisor": ADVISOR,
    "reviewer": REVIEWER,
}
