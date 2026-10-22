from __future__ import annotations

from datetime import date
from unittest.mock import patch

from chartreux.core.agents import AgentManager
from chartreux.core.config.agent_selection import AgentSelectionConfig
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.skills.manager import SkillManager
from chartreux.core.system_prompt import get_universal_system_prompt
from tests.conftest import build_test_vibe_config, multi_model_catalog
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


def test_prompt_size_report_fixed_fixture(capsys) -> None:
    """Report reproducible character counts; allocations are descriptive only."""
    catalog = multi_model_catalog()
    counts: dict[str, int] = {}
    for label, selection in (
        ("coder+minimal", AgentSelectionConfig(identity="coder", workflow="minimal")),
        (
            "orchestrator+ceremony+builtin:default",
            AgentSelectionConfig(
                identity="orchestrator",
                workflow="ceremony",
                orchestration_strategy="builtin:default",
            ),
        ),
    ):
        config = build_test_vibe_config(
            agent=selection,
            include_model_info=True,
            include_prompt_detail=False,
            include_project_context=False,
            include_commit_signature=False,
        ).attach_catalog_snapshot(CatalogSnapshot(catalog, "size-report"))
        with patch("chartreux.core.system_prompt.date") as date_mock:
            date_mock.today.return_value = date(2000, 1, 1)
            prompt = str(
                get_universal_system_prompt(
                    config,
                    SkillManager(lambda config=config: config),
                    AgentManager(FakeConfigOrchestrator(config)),
                )
            )
        counts[label] = len(prompt)
    print(
        f"FIXTURE catalog=multi_model_catalog skills=empty AGENTS=empty date=2000-01-01 strategy=builtin:default counts={counts}"
    )
    assert set(counts) == {"coder+minimal", "orchestrator+ceremony+builtin:default"}
    assert all(value > 0 for value in counts.values())
