"""Capture the composed dispatch goldens for the golden scenarios.

Run from the repository root with ``uv run python tests/fixtures/dispatch/capture_rendered.py``.
Each golden is the composed artifact for one scenario (design §5): the CLI
prompt — the nine-step composition served by ``get_universal_system_prompt``,
date pinned to 2000-01-01, catalog pinned — and the task tool description
rendered from the package-4 route facts and the scenario's session model.
The orchestrated scenarios compose the orchestrator + ceremony + default
selection; the single-model case uses an explicit one-model fixture, not
SHIPPED_CATALOG, so the single-model golden stays stable when the shipped
roster changes. The WP0 legacy baseline bytes the previous goldens carried
are preserved untouched under ``legacy-orchestrated/`` (ADR 0018-G.1
superseded, see the ADR amendment).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

from chartreux.core.agents import AgentManager
from chartreux.core.config.agent_selection import AgentSelectionConfig
from chartreux.core.config.harness_files import init_harness_files_manager
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

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
PINNED_DATE = date(2000, 1, 1)

ORCHESTRATED_SELECTION = AgentSelectionConfig(
    identity="orchestrator",
    workflow="ceremony",
    orchestration_strategy="builtin:default",
)
SCENARIOS: tuple[tuple[str, AgentSelectionConfig, ModelCatalog], ...] = (
    ("standalone", AgentSelectionConfig(), multi_model_catalog()),
    ("orchestrated", ORCHESTRATED_SELECTION, multi_model_catalog()),
    ("orchestrated-singlemodel", ORCHESTRATED_SELECTION, single_model_catalog()),
)


def _composed_config(selection: AgentSelectionConfig, catalog: ModelCatalog):
    return build_test_vibe_config(
        agent=selection,
        include_model_info=True,
        include_prompt_detail=False,
        include_project_context=False,
        include_commit_signature=False,
    ).attach_catalog_snapshot(CatalogSnapshot(catalog, "golden"))


def _render_cli(selection: AgentSelectionConfig, catalog: ModelCatalog) -> str:
    config = _composed_config(selection, catalog)
    with patch("chartreux.core.system_prompt.date") as date_mock:
        date_mock.today.return_value = PINNED_DATE
        return str(
            get_universal_system_prompt(
                config,
                SkillManager(lambda: config),
                AgentManager(FakeConfigOrchestrator(config)),
            )
        )


def _render_task(selection: AgentSelectionConfig, catalog: ModelCatalog) -> str:
    config = _composed_config(selection, catalog)
    return render_task_description_for_composition(
        routes=dict(config.purpose_routes.routes),
        session_model=config.get_active_model().alias,
        skeleton=task_skeleton(),
    )


def main() -> None:
    init_harness_files_manager("user", "project")
    for name, selection, catalog in SCENARIOS:
        directory = OUTPUT / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "cli.md").write_bytes(
            (_render_cli(selection, catalog).rstrip("\n") + "\n").encode("utf-8")
        )
        (directory / "task.md").write_bytes(
            (_render_task(selection, catalog).rstrip("\n") + "\n").encode("utf-8")
        )
        print(f"captured {name}/cli.md and {name}/task.md")


if __name__ == "__main__":
    main()
