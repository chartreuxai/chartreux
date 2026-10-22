"""Rendered dispatch prompts: composed goldens, launch targets, and S8.

The goldens pin the COMPOSED artifacts (design §5): the nine-step
composition's CLI prompt (date pinned to 2000-01-01, catalog pinned) and the
route-facts-rendered task tool description. Package 16b retired the legacy
policy-keyed rendering (``render_cli_prompt``, ``render_task_description``,
``render_routing_region``, ``roster_for`` and the slot tables behind them);
these tests pin the composition surface and the config-keyed task tool
description that replaced it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re
import tempfile
from unittest.mock import patch

import pytest

from chartreux.core.agents import AgentManager
from chartreux.core.agents.identities import get_builtin_identity
from chartreux.core.config.agent_selection import AgentSelectionConfig
from chartreux.core.dispatch.lint import lint_composed
from chartreux.core.dispatch.renderer import (
    DispatchRenderError,
    render_launch_target_region,
    render_task_description_for_composition,
    session_model_alias,
    task_description_for_config,
    task_skeleton,
)
from chartreux.core.dispatch.routes import shipped_routes
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.resolver import ModelResolver
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.skills.manager import SkillManager
from chartreux.core.system_prompt import get_universal_system_prompt
from chartreux.core.tools.manager import ToolManager
from tests.conftest import (
    build_test_vibe_config,
    multi_model_catalog,
    single_model_catalog,
)
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

FIXTURES = Path(__file__).parents[2] / "fixtures" / "dispatch"
PINNED_DATE = date(2000, 1, 1)
ORCHESTRATED_SELECTION = AgentSelectionConfig(
    identity="orchestrator",
    workflow="ceremony",
    orchestration_strategy="builtin:default",
)
# The golden scenarios: (directory, [agent] selection, catalog). The
# single-model case uses an explicit one-model fixture, not SHIPPED_CATALOG:
# the shipped roster is two-model since the ML4 entry landed, and the
# single-model rendering must stay stable regardless of how the shipped
# roster evolves.
SCENARIOS: tuple[tuple[str, AgentSelectionConfig, ModelCatalog], ...] = (
    ("standalone", AgentSelectionConfig(), multi_model_catalog()),
    ("orchestrated", ORCHESTRATED_SELECTION, multi_model_catalog()),
    ("orchestrated-singlemodel", ORCHESTRATED_SELECTION, single_model_catalog()),
)

# The direct-execution identity contract, pinned verbatim in the composed
# standalone prompt (the coder body is the served contract).
CODER_CONTRACT_SENTENCES = (
    "You are the coder: a direct, lean executor.",
    "Delegation is optional — when you delegate a bounded piece of work, "
    "the child is a bounded coder",
    "Finish the task, prove it works, report briefly.",
)

# --- Composed rendering (design §5) ----------------------------------------


def _composed_config(selection: AgentSelectionConfig, catalog: ModelCatalog):
    return build_test_vibe_config(
        agent=selection,
        include_model_info=True,
        include_prompt_detail=False,
        include_project_context=False,
        include_commit_signature=False,
    ).attach_catalog_snapshot(CatalogSnapshot(catalog, "golden"))


def _render_composed_cli(selection: AgentSelectionConfig, catalog: ModelCatalog) -> str:
    # The composed CLI prompt: the nine-step composition served by
    # get_universal_system_prompt, date pinned to 2000-01-01.
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


def _render_composed_task(
    selection: AgentSelectionConfig, catalog: ModelCatalog
) -> str:
    # The composed task tool description, rendered from the package-4 route
    # facts and the session model of the scenario's catalog.
    config = _composed_config(selection, catalog)
    return render_task_description_for_composition(
        routes=dict(config.purpose_routes.routes),
        session_model=config.get_active_model().alias,
        skeleton=task_skeleton(),
    )


# --- Goldens (composed artifacts) ------------------------------------------


def _without_final_newline(text: str) -> str:
    return text.removesuffix("\n")


@pytest.mark.parametrize(
    ("directory", "selection", "catalog"),
    SCENARIOS,
    ids=[name for name, _, _ in SCENARIOS],
)
def test_rendered_cli_and_task_match_goldens(
    directory: str, selection: AgentSelectionConfig, catalog: ModelCatalog
) -> None:
    # The goldens pin the COMPOSED artifacts (design §5), not raw skeletons:
    # the nine-step composition's CLI prompt and the route-facts-rendered
    # task tool description.
    assert _without_final_newline(
        _render_composed_cli(selection, catalog)
    ) == _without_final_newline((FIXTURES / directory / "cli.md").read_text())
    assert _without_final_newline(
        _render_composed_task(selection, catalog)
    ) == _without_final_newline((FIXTURES / directory / "task.md").read_text())


def test_goldens_are_deterministic() -> None:
    # The composition is deterministic: re-rendering each scenario yields
    # byte-identical artifacts.
    for _directory, selection, catalog in SCENARIOS:
        assert _render_composed_cli(selection, catalog) == _render_composed_cli(
            selection, catalog
        )
        assert _render_composed_task(selection, catalog) == _render_composed_task(
            selection, catalog
        )


@pytest.mark.parametrize(
    ("directory", "selection", "catalog"),
    SCENARIOS,
    ids=[name for name, _, _ in SCENARIOS],
)
def test_rendered_prompts_carry_worker_report_disclosure(
    directory: str, selection: AgentSelectionConfig, catalog: ModelCatalog
) -> None:
    # The task tool description always carries the brief-composition rule
    # (the task skeleton carries $dispatch_disclosure). The CLI-side rule
    # rides in the identity bodies that carry the placeholder: the
    # orchestrator composes every task brief, so its composed prompt carries
    # the rule; the coder body carries no $dispatch_disclosure token.
    cli = _render_composed_cli(selection, catalog)
    task = _render_composed_task(selection, catalog)
    assert "Every task brief must carry the worker disclosure requirement" in (task)
    assert "including edits later reverted" in task
    if selection.identity == "orchestrator":
        assert "Every task brief must carry the worker disclosure requirement" in (cli)
        assert "including edits later reverted" in cli
    else:
        assert (
            "Every task brief must carry the worker disclosure requirement" not in cli
        )


# --- Composed identity contract (contracts B) -------------------------------


def test_composed_standalone_prompt_carries_the_coder_contract() -> None:
    prompt = _render_composed_cli(AgentSelectionConfig(), multi_model_catalog())
    # The body wraps at column ~76; compare whitespace-normalized text.
    normalized = " ".join(prompt.split())
    for sentence in CODER_CONTRACT_SENTENCES:
        assert " ".join(sentence.split()) in normalized


def test_composed_standalone_prompt_has_no_orchestrator_edit_prohibition() -> None:
    prompt = _render_composed_cli(AgentSelectionConfig(), multi_model_catalog())
    assert "never edit repo files yourself" not in prompt


def test_composed_orchestrated_prompt_keeps_edit_prohibition() -> None:
    prompt = _render_composed_cli(ORCHESTRATED_SELECTION, multi_model_catalog())
    assert "never edit repo files yourself" in prompt


def test_direct_context_valve_is_operational_and_prominent(config_dir: Path) -> None:
    # The context-hygiene valve survives in the direct-execution identity
    # region. A custom direct identity's body carries the placeholder, so
    # the valve reaches the served prompt operational and prominent: the
    # context-growth warning precedes the delegation valve.
    user_prompts = config_dir / "prompts"
    user_prompts.mkdir()
    (user_prompts / "valve.md").write_text(
        "You are a custom direct executor.\n\n## Behavior\n\n$dispatch_identity\n",
        encoding="utf-8",
    )
    config = build_test_vibe_config(
        agent=AgentSelectionConfig(identity="user:valve"),
        include_model_info=False,
        include_prompt_detail=False,
        include_project_context=False,
        include_commit_signature=False,
    )
    prompt = str(
        get_universal_system_prompt(
            config,
            SkillManager(lambda: config),
            AgentManager(FakeConfigOrchestrator(config)),
        )
    )
    valve = (
        "When context is already long (including when the user says so), "
        "or a step would read or produce more than a few hundred lines, delegate "
        "that bounded piece to a child before searching or reading source "
        "yourself; keep only its concise result in the main context."
    )
    assert valve in prompt
    assert prompt.index("grows your context") < prompt.index(valve)


# --- WP0 baseline compatibility (contract G.1, superseded) ------------------
#
# ADR 0018-G.1 pinned the orchestrated multi-model rendering to the WP0 legacy
# routing bytes. The roster rename (large/medium/small -> worker/scout/heavy)
# rewrote the curated prose blocks, so that byte baseline is superseded by the
# regenerated active goldens; the legacy captures under legacy-orchestrated/
# are preserved untouched as historical artifacts. The test below pins that
# the regenerated composed goldens supersede the legacy bytes.


def test_orchestrated_multi_model_golden_supersedes_legacy_bytes() -> None:
    cli = (FIXTURES / "orchestrated" / "cli.md").read_text()
    task = (FIXTURES / "orchestrated" / "task.md").read_text()
    legacy_cli = (FIXTURES / "legacy-orchestrated" / "cli-routing.md").read_text()
    legacy_task = (FIXTURES / "legacy-orchestrated" / "task-routing.md").read_text()
    # The regenerated goldens pin the COMPOSED artifacts; the WP0 legacy bytes
    # no longer appear (G.1 superseded by the regenerated goldens), and the
    # composed routing is the launch-target table.
    assert legacy_cli.rstrip("\n") not in cli
    assert legacy_task.rstrip("\n") not in task
    assert "### Launch targets" in cli
    assert "### Launch targets" in task
    # The composed orchestrator prompt carries the delegation-posture regions.
    assert "never edit repo files yourself" in cli
    assert "Route by purpose: the launch-target table" in cli


# --- Launch-target rendering (roster-aware routing successor) ---------------
#
# The retired roster-aware slot tables rendered tiers per roster size. The
# successor binds every shipped default target to the session model, so one
# configured model and ten produce identical, tier-free routing.


@pytest.mark.parametrize(
    "catalog", [single_model_catalog(), multi_model_catalog()], ids=["single", "multi"]
)
def test_launch_targets_bind_the_session_model_regardless_of_roster_size(
    catalog: ModelCatalog,
) -> None:
    config = _composed_config(ORCHESTRATED_SELECTION, catalog)
    task = _render_composed_task(ORCHESTRATED_SELECTION, catalog)
    alias = config.get_active_model().alias
    rows = [line for line in task.splitlines() if line.startswith("| `")]
    assert rows
    for row in rows:
        assert row.endswith(f"| `{alias}` |")


def test_launch_targets_render_deep_review_seats_honestly() -> None:
    # Deep review renders its three seats — analytical and peer on the
    # reviewer identity, execution on the coder identity with the
    # review-execution overlay — and reports the distinct-model requirement
    # instead of faking diversity.
    region = render_launch_target_region(shipped_routes(), "glm-5-3")
    assert (
        "| `review.deep` | `reviewer` | substantive | analytical | `glm-5-3` |"
        in region
    )
    assert "| `review.deep` | `reviewer` | substantive | peer | `glm-5-3` |" in region
    assert "| `review.deep` | `coder` | substantive | execution | `glm-5-3` |" in region
    assert (
        "- `review.deep`: 3 distinct canonical models (counted by canonical "
        "model, not deployment)." in region
    )


def test_every_role_reference_in_composed_prompts_resolves() -> None:
    for _directory, selection, catalog in SCENARIOS:
        resolver = ModelResolver(CatalogSnapshot(catalog, "rendered-roles"))
        texts = (
            _render_composed_cli(selection, catalog),
            _render_composed_task(selection, catalog),
        )
        for text in texts:
            for role in dict.fromkeys(re.findall(r"@([a-z][a-z0-9-]*)", text)):
                if role == "role":
                    continue
                assert resolver.resolve(f"@{role}").base_model


# --- Lifecycle prose synchronization (WP5A) --------------------------------
#
# The $dispatch_reuse block carries the subagent lifecycle semantics in the
# task tool description; pin its sentences so a prose reversion fails the
# rendering suite, not just the golden diff.

REUSE_LIFECYCLE_SENTENCES = (
    "omit `retain` to inherit the instance's committed retention policy",
    "a `still_running` timeout is a normal outcome",
    "A `closed` or `evicted` agent cannot be reused",
)


@pytest.mark.parametrize(
    ("directory", "selection", "catalog"),
    SCENARIOS,
    ids=[name for name, _, _ in SCENARIOS],
)
def test_rendered_task_description_documents_lifecycle_semantics(
    directory: str, selection: AgentSelectionConfig, catalog: ModelCatalog
) -> None:
    rendered = _render_composed_task(selection, catalog)
    for sentence in REUSE_LIFECYCLE_SENTENCES:
        assert sentence in rendered
    # Background-first launch guidance and opt-in retention come from the
    # task skeleton itself.
    assert "Launch in the background even when the result is a dependency" in rendered
    assert "Retention is opt-in." in rendered
    assert "for a final continuation that closes afterward" in rendered


def test_composed_orchestrated_cli_documents_agent_lifecycle() -> None:
    prompt = _render_composed_cli(ORCHESTRATED_SELECTION, multi_model_catalog())
    # The orchestrator identity body carries the agent-lifecycle paragraph.
    assert "Manage delegated work through the agent-lifecycle tools" in prompt
    assert "Launch in the background even when the result is a dependency" in prompt
    assert "`still_running` timeout result is a normal outcome" in prompt


# --- S8 reversion tripwire on the composed artifacts -------------------------
#
# S8 lints the composed output, keyed on the typed identity facts and the
# workflow — not on DispatchMode.


@pytest.mark.parametrize(
    ("directory", "selection", "catalog"),
    SCENARIOS,
    ids=[name for name, _, _ in SCENARIOS],
)
def test_s8_anchor_tripwire_on_composed_output(
    directory: str, selection: AgentSelectionConfig, catalog: ModelCatalog
) -> None:
    cli = _render_composed_cli(selection, catalog)
    facts = get_builtin_identity(selection.identity)
    assert lint_composed(cli, identity_facts=facts, workflow=selection.workflow) == ()
    assert "$dispatch" not in cli
    assert "$dispatch" not in _render_composed_task(selection, catalog)


def test_s8_rejects_standalone_edit_prohibition() -> None:
    # Residual gate language in a non-ceremony composition is a leak.
    cli = _render_composed_cli(AgentSelectionConfig(), multi_model_catalog())
    diagnostics = lint_composed(
        cli + " Never edit repo files yourself.",
        identity_facts=get_builtin_identity("coder"),
        workflow="minimal",
    )
    assert any(item.id == "S8" for item in diagnostics)


def test_s8_rejects_orchestrator_without_delegation_obligation() -> None:
    # The orchestrator's delegation obligation is the S8 anchor: the
    # edit prohibition itself rides in the ceremony workflow's implement
    # region and is not a mode-keyed lint rule.
    cli = _render_composed_cli(ORCHESTRATED_SELECTION, multi_model_catalog())
    stripped = re.sub(r"delegation is required", "redacted", cli, flags=re.IGNORECASE)
    assert "delegation is required" not in stripped.lower()
    diagnostics = lint_composed(
        stripped,
        identity_facts=get_builtin_identity("orchestrator"),
        workflow="ceremony",
    )
    assert any(item.id == "S8" for item in diagnostics)


# --- Placeholder hygiene and custom prompts ---------------------------------


def test_rendered_outputs_have_no_placeholders() -> None:
    # The composed artifacts.
    for _directory, selection, catalog in SCENARIOS:
        assert "$dispatch" not in _render_composed_cli(selection, catalog)
        assert "$dispatch" not in _render_composed_task(selection, catalog)


def test_unresolved_placeholder_is_rejected() -> None:
    skeleton = task_skeleton() + "\n$dispatch_unknown\n"
    with pytest.raises(DispatchRenderError, match=r"\$dispatch_unknown"):
        render_task_description_for_composition(
            routes=shipped_routes(), session_model="glm-5-3", skeleton=skeleton
        )


def test_custom_prompt_without_placeholders_passes_through() -> None:
    custom = "# Custom prompt\n\nNothing to render here.\n"
    assert (
        render_task_description_for_composition(
            routes=shipped_routes(), session_model="glm-5-3", skeleton=custom
        )
        == custom
    )


# --- Task tool description injection (contracts L) --------------------------


def _task_spec_description(manager: ToolManager) -> str:
    specs = {spec.name: spec for spec in manager.available_tool_specs()}
    return specs["task"].description


def test_tool_manager_serves_route_bound_task_description() -> None:
    config = build_test_vibe_config()
    manager = ToolManager(lambda: config)
    description = _task_spec_description(manager)
    assert "$dispatch" not in description
    # The served description is the composition render: the launch-target
    # table and the brief-composition rule ride in it.
    assert "### Launch targets" in description
    assert "worker disclosure requirement" in description
    assert description == task_description_for_config(config)


def test_task_description_is_keyed_by_the_session_model() -> None:
    single = build_test_vibe_config()
    multi = build_test_vibe_config().attach_catalog_snapshot(
        CatalogSnapshot(multi_model_catalog(), "rendering-test")
    )
    single_description = _task_spec_description(ToolManager(lambda: single))
    multi_description = _task_spec_description(ToolManager(lambda: multi))
    # The launch-target table binds each configuration's session model.
    assert single_description != multi_description
    assert f"| `{session_model_alias(single)}` |" in single_description
    assert f"| `{session_model_alias(multi)}` |" in multi_description
    assert "### Launch targets" in single_description
    assert "### Launch targets" in multi_description
    # Each manager's spec cache serves its own configuration's description.
    assert _task_spec_description(ToolManager(lambda: single)) == single_description


def test_tool_manager_serves_shipped_catalog_task_description() -> None:
    config = build_test_vibe_config().attach_catalog_snapshot(
        CatalogSnapshot(SHIPPED_CATALOG, "rendering-test")
    )
    description = _task_spec_description(ToolManager(lambda: config))
    assert "### Launch targets" in description
    assert description == task_description_for_config(config)


@pytest.mark.parametrize("runtime_description", [None, "Runtime-rendered description."])
def test_custom_task_prompt_override_wins_over_the_rendered_default(
    runtime_description,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tools = Path(tmp) / "tools"
        (tools / "prompts").mkdir(parents=True)
        (tools / "prompts" / "task.md").write_text(
            "Custom task description.", encoding="utf-8"
        )
        config = build_test_vibe_config(tool_paths=[str(tools)])
        manager = ToolManager(lambda: config, task_description=runtime_description)
        assert _task_spec_description(manager) == "Custom task description."
        manager._install_task_description("Next runtime render.")
        assert _task_spec_description(manager) == "Custom task description."
