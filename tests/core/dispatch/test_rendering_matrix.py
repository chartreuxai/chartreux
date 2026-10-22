"""Cross-product merge gates: the composed artifacts across selection x catalog."""

from __future__ import annotations

import re

import pytest

from chartreux.core.agents import AgentManager
from chartreux.core.agents.identities import get_builtin_identity
from chartreux.core.config.agent_selection import AgentSelectionConfig
from chartreux.core.dispatch.lint import lint_composed
from chartreux.core.dispatch.renderer import (
    render_task_description_for_composition,
    task_skeleton,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.skills.manager import SkillManager
from chartreux.core.system_prompt import get_universal_system_prompt
from tests.conftest import (
    build_test_vibe_config,
    multi_model_catalog,
    single_model_catalog,
)
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

ORCHESTRATED_SELECTION = AgentSelectionConfig(
    identity="orchestrator",
    workflow="ceremony",
    orchestration_strategy="builtin:default",
)
SCENARIOS: tuple[tuple[str, AgentSelectionConfig], ...] = (
    ("standalone", AgentSelectionConfig()),
    ("orchestrated", ORCHESTRATED_SELECTION),
    ("orchestrated-singlemodel", ORCHESTRATED_SELECTION),
)
CATALOGS: tuple[tuple[str, ModelCatalog], ...] = (
    ("single", single_model_catalog()),
    ("multi", multi_model_catalog()),
)


@pytest.mark.parametrize(
    ("scenario", "selection"), SCENARIOS, ids=[name for name, _ in SCENARIOS]
)
@pytest.mark.parametrize(
    "catalog", [catalog for _, catalog in CATALOGS], ids=[name for name, _ in CATALOGS]
)
def test_composition_rendering_matrix(scenario, selection, catalog):
    # The composed artifacts across the selection × catalog cross-product:
    # no placeholders, S8-clean, and the launch-target table binds the
    # session model of the scenario's catalog in both documents.
    config = build_test_vibe_config(
        agent=selection,
        include_model_info=True,
        include_prompt_detail=False,
        include_project_context=False,
        include_commit_signature=False,
    ).attach_catalog_snapshot(CatalogSnapshot(catalog, "matrix"))
    cli = str(
        get_universal_system_prompt(
            config,
            SkillManager(lambda: config),
            AgentManager(FakeConfigOrchestrator(config)),
        )
    )
    task = render_task_description_for_composition(
        routes=dict(config.purpose_routes.routes),
        session_model=config.get_active_model().alias,
        skeleton=task_skeleton(),
    )
    assert not re.search(r"\$dispatch(?:_[a-z_]+)?", cli + task)
    facts = get_builtin_identity(selection.identity)
    assert lint_composed(cli, identity_facts=facts, workflow=selection.workflow) == ()
    # The worker disclosure requirement rides in the task description.
    assert "worker disclosure requirement" in task
    # The launch-target table binds the session model in both documents.
    alias = config.get_active_model().alias
    row = f"| `implementation` | `coder` | substantive | — | `{alias}` |"
    assert row in cli
    assert row in task
    # The composition opens at character zero and re-anchors last.
    assert "Today's date is" in cli.splitlines()[1]
    assert cli.rstrip().endswith("this closing note adds no new rules.")
