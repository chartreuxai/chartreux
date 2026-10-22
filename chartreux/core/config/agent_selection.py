from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator


class AgentSelectionConfig(BaseModel):
    """The ``[agent]`` selections: identity, workflow, orchestration strategy.

    This is the ONE definition of the agent-selection schema (the previously
    duplicated weak definition in ``config/models.py`` was removed; the wired
    schema path imports it from here). The validator is the STRONGER one: it
    rejects malformed ids, abusive ``builtin:`` qualification (empty or
    ``none`` names), and — for the identity field — ``builtin:``-qualified
    names that do not exist, so an invalid selection fails at config load
    with an actionable error (design criterion 2; S2 unknown-identity
    fail-open).
    """

    model_config = ConfigDict(extra="forbid")

    identity: str = "coder"
    workflow: str = "minimal"
    orchestration_strategy: str = "none"

    @field_validator("identity", "workflow", "orchestration_strategy")
    @classmethod
    def valid_id(cls, value: str, info: ValidationInfo) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_:-]*", value) or ".." in value:
            raise ValueError("invalid agent resource id")
        if ":" in value:
            source, name = value.split(":", 1)
            if source not in {"builtin", "user"} or not name or ":" in name:
                raise ValueError(f"invalid agent resource qualification: {value!r}")
            if source == "builtin" and name == "none":
                raise ValueError(f"invalid built-in resource qualification: {value!r}")
            if info.field_name == "identity" and source == "builtin":
                # Imported lazily: the identities module must not depend on the
                # config package (import cycle).
                from chartreux.core.agents.identities import get_builtin_identity

                if get_builtin_identity(name) is None:
                    raise ValueError(f"unknown built-in identity: {value!r}")
        return value
