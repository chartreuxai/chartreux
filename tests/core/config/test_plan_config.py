from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.core.config.chartreux_schema import (
    ChartreuxConfigSchema,
    create_default_config,
)
from chartreux.core.config.models import (
    DEFAULT_PLAN_EXECUTOR_IDENTITY,
    DEFAULT_PLAN_EXECUTOR_MODEL,
    DEFAULT_PLAN_EXECUTOR_THINKING,
    DEFAULT_PLAN_PLANNER_MODEL,
    DEFAULT_PLAN_PLANNER_THINKING,
    PlanExecutorConfig,
    PlanPlannerConfig,
    PlanSectionConfig,
)


def test_plan_section_shipped_defaults_are_glm_high_and_ml4_medium() -> None:
    plan = PlanSectionConfig()

    assert plan.planner.model == "glm-5-3"
    assert plan.planner.thinking == "high"
    assert plan.executor.model == "mistral-large-4"
    assert plan.executor.thinking == "medium"
    assert plan.executor.identity == "orchestrator"


def test_plan_defaults_constants_match_the_shipped_selections() -> None:
    assert DEFAULT_PLAN_PLANNER_MODEL == "glm-5-3"
    assert DEFAULT_PLAN_PLANNER_THINKING == "high"
    assert DEFAULT_PLAN_EXECUTOR_MODEL == "mistral-large-4"
    assert DEFAULT_PLAN_EXECUTOR_THINKING == "medium"
    assert DEFAULT_PLAN_EXECUTOR_IDENTITY == "orchestrator"


def test_plan_section_is_registered_on_the_root_schema() -> None:
    config = ChartreuxConfigSchema()

    assert isinstance(config.plan.planner, PlanPlannerConfig)
    assert isinstance(config.plan.executor, PlanExecutorConfig)
    assert config.plan.executor.identity == "orchestrator"


def test_plan_section_defaults_ship_as_configuration() -> None:
    assert create_default_config()["plan"] == {
        "planner": {"model": "glm-5-3", "thinking": "high"},
        "executor": {
            "model": "mistral-large-4",
            "thinking": "medium",
            "identity": "orchestrator",
        },
    }


def test_plan_section_overrides_change_the_selections() -> None:
    config = ChartreuxConfigSchema.model_validate({
        "plan": {
            "planner": {"model": "glm-5-3", "thinking": "max"},
            "executor": {
                "model": "mistral-large-4",
                "thinking": "low",
                "identity": "coder",
            },
        }
    })

    assert config.plan.planner.model == "glm-5-3"
    assert config.plan.planner.thinking == "max"
    assert config.plan.executor.model == "mistral-large-4"
    assert config.plan.executor.thinking == "low"
    assert config.plan.executor.identity == "coder"


def test_plan_section_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        ChartreuxConfigSchema.model_validate({
            "plan": {"planner": {"temperature": 0.2}}
        })
    with pytest.raises(ValidationError):
        ChartreuxConfigSchema.model_validate({
            "plan": {"executor": {"identity": "orchestrator", "x": 1}}
        })


@pytest.mark.parametrize("thinking", ["ultra", "", 1])
def test_plan_section_rejects_invalid_thinking_levels(thinking: object) -> None:
    with pytest.raises(ValidationError):
        ChartreuxConfigSchema.model_validate({
            "plan": {"planner": {"thinking": thinking}}
        })


@pytest.mark.parametrize("model", ["", 3])
def test_plan_section_rejects_invalid_model_aliases(model: object) -> None:
    with pytest.raises(ValidationError):
        ChartreuxConfigSchema.model_validate({"plan": {"executor": {"model": model}}})


def test_plan_section_rejects_unknown_builtin_identity() -> None:
    with pytest.raises(ValidationError):
        ChartreuxConfigSchema.model_validate({
            "plan": {"executor": {"identity": "builtin:missing"}}
        })


def test_plan_section_rejects_malformed_identity() -> None:
    with pytest.raises(ValidationError):
        ChartreuxConfigSchema.model_validate({
            "plan": {"executor": {"identity": "not an id"}}
        })


@pytest.mark.asyncio
async def test_plan_section_sparse_toml_layer_preserves_other_defaults(
    tmp_path: Path,
) -> None:
    from chartreux.core.config.layers.user import UserConfigLayer
    from chartreux.core.config.orchestrator import ConfigOrchestrator

    path = tmp_path / "config.toml"
    path.write_text('[plan.executor]\nidentity = "coder"\n')
    user = UserConfigLayer(path=path)
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema, layers=[user], default_layer_resolver=lambda: user
    )

    assert orchestrator.config.plan.executor.identity == "coder"
    assert orchestrator.config.plan.executor.model == "mistral-large-4"
    assert orchestrator.config.plan.executor.thinking == "medium"
    assert orchestrator.config.plan.planner.model == "glm-5-3"
    assert orchestrator.config.plan.planner.thinking == "high"
