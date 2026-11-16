"""Retirement pins for the ``[plan]`` config section (plan-mode removal).

The plan-mode feature is removed: the ``plan`` section, its model classes,
and its default constants no longer exist, and both configuration surfaces
fail loudly with successor guidance instead of silently ignoring the keys:

- a ``[plan]`` block in any TOML layer is rejected by a ``mode="before"``
  schema validator;
- a ``CHARTREUX_PLAN__*`` environment variable is rejected by the retired
  field guidance in source validation;
- the shipped default configuration carries no ``plan`` key.
"""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from chartreux.core.config import models
from chartreux.core.config._source_validation import validate_environment_names
from chartreux.core.config.chartreux_schema import (
    ChartreuxConfigSchema,
    create_default_config,
)


def test_plan_section_symbols_are_gone() -> None:
    assert not hasattr(models, "PlanSectionConfig")
    assert not hasattr(models, "PlanPlannerConfig")
    assert not hasattr(models, "PlanExecutorConfig")
    for name in (
        "DEFAULT_PLAN_PLANNER_MODEL",
        "DEFAULT_PLAN_PLANNER_THINKING",
        "DEFAULT_PLAN_EXECUTOR_MODEL",
        "DEFAULT_PLAN_EXECUTOR_THINKING",
        "DEFAULT_PLAN_EXECUTOR_IDENTITY",
    ):
        assert not hasattr(models, name), name


def test_plan_section_is_not_a_schema_field() -> None:
    assert "plan" not in ChartreuxConfigSchema.model_fields


def test_plan_block_is_rejected_with_retired_guidance() -> None:
    with pytest.raises(ValidationError, match="'plan' is retired"):
        ChartreuxConfigSchema.model_validate({"plan": {"planner": {}}})


def test_default_configuration_carries_no_plan_key() -> None:
    assert "plan" not in create_default_config()


def test_plan_environment_names_are_rejected_with_retired_guidance() -> None:
    with pytest.raises(ValidationError, match="retired"):
        validate_environment_names(
            ChartreuxConfigSchema,
            {"CHARTREUX_PLAN__PLANNER__MODEL": "glm-5-3"},
            source="environment",
        )
