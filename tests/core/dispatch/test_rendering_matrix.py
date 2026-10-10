"""Cross-product merge gate for policy, roster, CLI and task rendering."""

from __future__ import annotations

import re

import pytest

from chartreux.core.dispatch.lint import lint_rendered
from chartreux.core.dispatch.presets import SHIPPED_PRESETS
from chartreux.core.dispatch.renderer import (
    render_cli_prompt,
    render_task_description,
    roster_for,
    task_skeleton,
)
from chartreux.core.prompts import SystemPrompt
from chartreux.core.system_prompt import _interpolate_prompt
from tests.conftest import multi_model_catalog, single_model_catalog

FIXTURES = SystemPrompt.CLI
TIERS = ("@scout", "@worker", "@heavy")


@pytest.mark.parametrize("mode,policy", SHIPPED_PRESETS.items())
@pytest.mark.parametrize(
    "catalog", [single_model_catalog(), multi_model_catalog()], ids=["single", "multi"]
)
def test_policy_roster_rendering_matrix(mode, policy, catalog):
    shape = roster_for(catalog, policy)
    cli = render_cli_prompt(policy, shape, _interpolate_prompt(FIXTURES.read()))
    task = render_task_description(policy, shape, task_skeleton())
    assert not re.search(r"\$dispatch(?:_[a-z_]+)?", cli + task)
    assert lint_rendered(cli, policy.mode) == ()
    assert "Route by" in task
    if shape.single_model:
        assert all(tier not in cli and tier not in task for tier in TIERS)
    elif policy.mode.value == "orchestrated":
        assert "@scout" in cli and "@worker" in cli
        assert "@scout" in task
