from __future__ import annotations

from typing import Any

from pydantic import ValidationError
import pytest

from chartreux.core.config import agent_selection
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import AgentSelectionConfig
from chartreux.core.policy.resources import ResourceLoader, ResourceResolutionError


def test_agent_selection_defaults_and_schema_defaults() -> None:
    assert AgentSelectionConfig().model_dump() == {
        "identity": "coder",
        "workflow": "minimal",
        "orchestration_strategy": "none",
    }
    assert (
        ChartreuxConfigSchema().agent.model_dump()
        == AgentSelectionConfig().model_dump()
    )


def test_schema_accepts_explicit_agent_selections() -> None:
    schema = ChartreuxConfigSchema(
        agent=AgentSelectionConfig(
            identity="researcher", workflow="minimal", orchestration_strategy="none"
        )
    )
    assert schema.agent.identity == "researcher"


def test_one_definition_serves_the_wired_schema_path() -> None:
    """The stronger validator is THE definition on the wired schema path.

    The previously duplicated weak definition in ``config/models.py`` was
    removed: the models module re-exports the single definition, and the
    schema's ``agent`` field is annotated with that same class (S2 W2).
    """
    assert agent_selection.AgentSelectionConfig is AgentSelectionConfig
    assert (
        ChartreuxConfigSchema.model_fields["agent"].annotation is AgentSelectionConfig
    )


@pytest.mark.parametrize(
    "identity",
    [
        "builtin:reviever",  # typo'd built-in qualification
        "builtin:",  # empty built-in name
        "builtin:none",  # abusive built-in qualification
        "builtin:coder:extra",  # second qualification separator
        "user:",  # empty user name
        "bad..id",  # malformed id
    ],
)
def test_invalid_identity_selections_fail_at_config_load(identity: str) -> None:
    """Invalid selections fail with an actionable error (design criterion 2)."""
    with pytest.raises(ValidationError):
        AgentSelectionConfig(identity=identity)
    with pytest.raises(ValidationError):
        ChartreuxConfigSchema.model_validate({"agent": {"identity": identity}})


@pytest.mark.parametrize(
    "selection",
    [
        {"identity": "builtin:researcher"},
        {"identity": "user:my-custom"},
        {"identity": "my-custom"},  # bare custom ids pass the schema
        {"workflow": "builtin:minimal"},
        {"workflow": "user:my-workflow"},
        {"orchestration_strategy": "user:explicit"},
    ],
)
def test_valid_selections_keep_their_strength(selection: dict[str, Any]) -> None:
    defaults = AgentSelectionConfig().model_dump()
    schema = ChartreuxConfigSchema.model_validate({"agent": selection})
    assert schema.agent.model_dump() == {**defaults, **selection}


def test_workflow_and_strategy_reject_none_and_malformed_qualifications() -> None:
    with pytest.raises(ValidationError, match="built-in resource"):
        AgentSelectionConfig(workflow="builtin:none")
    with pytest.raises(ValidationError, match="qualification"):
        AgentSelectionConfig(orchestration_strategy="user:")
    with pytest.raises(ValidationError, match="invalid agent resource id"):
        AgentSelectionConfig(workflow="bad..id")


def test_none_strategy_resolves_without_reading_strategy_body(tmp_path) -> None:
    loader = ResourceLoader({}, {})
    resolved = loader.resolve(
        __import__(
            "chartreux.core.policy.models", fromlist=["ResourceKind"]
        ).ResourceKind.STRATEGY,
        "none",
    )
    assert (resolved.id, resolved.source, resolved.content, resolved.path) == (
        "none",
        "builtin",
        None,
        None,
    )


def test_strategy_conflicts_with_none_delegation() -> None:
    loader = ResourceLoader({}, {})
    with pytest.raises(ResourceResolutionError, match="delegation=none"):
        loader.validate_strategy("none", "explicit")
