"""Rendered dispatch prompts: goldens, compatibility, and the S8 tripwire."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from chartreux.core.dispatch.lint import RosterShape, lint_rendered
from chartreux.core.dispatch.presets import (
    ORCHESTRATED_PRESET,
    SHIPPED_PRESETS,
    STANDALONE_PRESET,
)
from chartreux.core.dispatch.renderer import (
    DispatchRenderError,
    render_cli_prompt,
    render_routing_region,
    render_task_description,
    render_task_regions,
    roster_for,
    task_skeleton,
)
from chartreux.core.dispatch.schema import DispatchPolicy
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.prompts import SystemPrompt
from chartreux.core.system_prompt import _interpolate_prompt
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_vibe_config, multi_model_catalog

FIXTURES = Path(__file__).parents[2] / "fixtures" / "dispatch"
PINNED_DATE = date(2000, 1, 1)
TIER_NAMES = ("@small", "@medium", "@large")

STANDALONE_SENTENCES = (
    "You may implement directly: make the approved repo edits yourself. "
    "Stay within the approved plan; change minimally.",
    "Verification stays delegated. Dispatch checks to a subagent, scaled to "
    "the change, then dispatch an independent review against the approved "
    "plan — a fresh reviewer, never the author. Never run tests, builds, or "
    "other verification yourself; never review your own work; never claim a "
    "check you did not run.",
    "Direct execution grows your context: everything you read or run is "
    "re-sent on every call. Keep reads targeted, bound shell output, move "
    "large artifacts to the scratchpad, and say when your context is getting "
    "long.",
    "write_file/edit — repo files within the approved scope; scratchpad for "
    "temporary artifacts. read_file — files the task names or cites. bash — "
    "orchestration metadata only, always with timeouts; delegate searches, "
    "exploration, tests, and builds.",
    "Delegation is optional. When you delegate, route by purpose through "
    "the configured slots and pass the tier explicitly; the slot table is "
    "authoritative. When you implement directly, the tier rules govern only "
    "work you delegate.",
)


def _shape(policy: DispatchPolicy, catalog: ModelCatalog) -> RosterShape:
    return roster_for(catalog, policy)


def _render_cli(policy: DispatchPolicy, catalog: ModelCatalog) -> str:
    with patch("chartreux.core.system_prompt.date") as date_mock:
        date_mock.today.return_value = PINNED_DATE
        return render_cli_prompt(
            policy,
            _shape(policy, catalog),
            _interpolate_prompt(SystemPrompt.CLI.read()),
        )


def _render_task(policy: DispatchPolicy, catalog: ModelCatalog) -> str:
    return render_task_description(policy, _shape(policy, catalog), task_skeleton())


# --- Goldens ---------------------------------------------------------------


def _without_final_newline(text: str) -> str:
    return text.removesuffix("\n")


@pytest.mark.parametrize(
    ("directory", "policy", "catalog"),
    [
        ("standalone", STANDALONE_PRESET, multi_model_catalog()),
        ("orchestrated", ORCHESTRATED_PRESET, multi_model_catalog()),
        ("orchestrated-singlemodel", ORCHESTRATED_PRESET, SHIPPED_CATALOG),
    ],
    ids=["standalone", "orchestrated", "orchestrated-singlemodel"],
)
def test_rendered_cli_and_task_match_goldens(
    directory: str, policy: DispatchPolicy, catalog: ModelCatalog
) -> None:
    assert _without_final_newline(
        _render_cli(policy, catalog)
    ) == _without_final_newline((FIXTURES / directory / "cli.md").read_text())
    assert _without_final_newline(
        _render_task(policy, catalog)
    ) == _without_final_newline((FIXTURES / directory / "task.md").read_text())


def test_goldens_are_deterministic() -> None:
    for policy in SHIPPED_PRESETS.values():
        assert _render_cli(policy, SHIPPED_CATALOG) == _render_cli(
            policy, SHIPPED_CATALOG
        )


@pytest.mark.parametrize("catalog", [SHIPPED_CATALOG, multi_model_catalog()])
def test_standalone_context_valve_is_operational_and_prominent(
    catalog: ModelCatalog,
) -> None:
    prompt = _render_cli(STANDALONE_PRESET, catalog)
    valve = (
        "When context is already long (including when the user says so), "
        "or a step would read or produce more than a few hundred lines, delegate "
        "that bounded piece to a worker before searching or reading source "
        "yourself; keep only its concise result in the main context. This "
        "context-hygiene requirement overrides optional delegation; keep the "
        "session in standalone mode."
    )
    assert valve in STANDALONE_PRESET.instructions
    assert valve in prompt
    assert prompt.index("Direct execution grows your context") < prompt.index(valve)
    assert prompt.index(valve) < prompt.index("Delegation is optional")


# --- WP0 baseline compatibility (contracts G.1) -----------------------------


def test_orchestrated_multi_model_routing_preserves_legacy_cli_bytes() -> None:
    legacy = (FIXTURES / "legacy-orchestrated" / "cli-routing.md").read_text()
    region = render_routing_region(
        ORCHESTRATED_PRESET, _shape(ORCHESTRATED_PRESET, multi_model_catalog())
    )
    # The baseline captured the physical source lines, including the final
    # newline of the last paragraph.
    assert region.encode().startswith(legacy.rstrip("\n").encode() + b"\n\n")


def test_orchestrated_multi_model_routing_preserves_legacy_task_bytes() -> None:
    legacy = (FIXTURES / "legacy-orchestrated" / "task-routing.md").read_text()
    region = render_task_regions(
        ORCHESTRATED_PRESET, _shape(ORCHESTRATED_PRESET, multi_model_catalog())
    )["routing"]
    assert region.encode().startswith(legacy.rstrip("\n").encode() + b"\n\n")


def test_orchestrated_multi_model_golden_contains_legacy_bytes() -> None:
    cli = (FIXTURES / "orchestrated" / "cli.md").read_text()
    task = (FIXTURES / "orchestrated" / "task.md").read_text()
    legacy_cli = (FIXTURES / "legacy-orchestrated" / "cli-routing.md").read_text()
    legacy_task = (FIXTURES / "legacy-orchestrated" / "task-routing.md").read_text()
    assert legacy_cli.rstrip("\n") in cli
    assert legacy_task.rstrip("\n") in task


# --- Roster-aware rendering (contracts D/G) --------------------------------


def test_single_model_routing_is_tier_free() -> None:
    shape = _shape(ORCHESTRATED_PRESET, SHIPPED_CATALOG)
    assert shape.single_model
    region = render_routing_region(ORCHESTRATED_PRESET, shape)
    task_region = render_task_regions(ORCHESTRATED_PRESET, shape)["routing"]
    for tier in TIER_NAMES:
        assert tier not in region
        assert tier not in task_region
    assert "Route by task kind through the configured slots" in region
    assert "`mechanical` (profile `worker`) takes search" in region


def test_multi_model_routing_uses_tiers() -> None:
    shape = _shape(ORCHESTRATED_PRESET, multi_model_catalog())
    assert not shape.single_model
    region = render_routing_region(ORCHESTRATED_PRESET, shape)
    assert "`@small` for search, grep, exploration, verification" in region
    assert "`@medium` for all substantive implementation" in region


def test_roster_shape_ignores_thinking_and_provider_diversity() -> None:
    # One canonical model across thinking levels stays single-model.
    assert _shape(ORCHESTRATED_PRESET, SHIPPED_CATALOG).single_model
    # The same canonical model reached through two providers is still a
    # single-model roster: provider deployments never manufacture diversity.
    shared = ModelCatalog.model_validate({
        "providers": {
            "alpha": {"api_base": "https://alpha.test"},
            "beta": {"api_base": "https://beta.test"},
        },
        "models": {
            "shared-model": {
                "thinking": "medium",
                "deployments": [
                    {"provider": "alpha", "name": "alpha-wire"},
                    {"provider": "beta", "name": "beta-wire"},
                ],
            }
        },
        "roles": {
            "orchestrator": {
                "model": "shared-model",
                "thinking": "high",
                "description": "main assistant preset",
            },
            "small": {
                "model": "shared-model",
                "thinking": "low",
                "description": "small preset",
            },
            "medium": {
                "model": "shared-model",
                "thinking": "medium",
                "description": "medium preset",
            },
            "large": {
                "model": "shared-model",
                "thinking": "high",
                "description": "large preset",
            },
        },
    })
    assert _shape(ORCHESTRATED_PRESET, shared).single_model
    assert not _shape(ORCHESTRATED_PRESET, multi_model_catalog()).single_model


def test_standalone_slot_table_is_roster_aware() -> None:
    multi = render_routing_region(
        STANDALONE_PRESET, _shape(STANDALONE_PRESET, multi_model_catalog())
    )
    single = render_routing_region(
        STANDALONE_PRESET, _shape(STANDALONE_PRESET, SHIPPED_CATALOG)
    )
    assert "| Slot | Profile | Role | Model | Thinking | Purposes |" in multi
    assert "| `mechanical` | `worker` | `@small` | `alpha-model` | low |" in multi
    assert "| `mechanical` | `worker` | — | `glm-5-3` | low |" in single
    for tier in TIER_NAMES:
        assert tier not in single


# --- Standalone preset (contracts B) ---------------------------------------


def test_standalone_rendering_contains_contract_sentences_verbatim() -> None:
    rendered = _render_cli(STANDALONE_PRESET, SHIPPED_CATALOG)
    for sentence in STANDALONE_SENTENCES:
        assert sentence in rendered
    assert "The opening gate and acceptance lattice remain unchanged." in rendered
    assert "Saved policy changes apply next session." in rendered


def test_standalone_rendering_has_no_orchestrator_edit_prohibition() -> None:
    rendered = _render_cli(STANDALONE_PRESET, SHIPPED_CATALOG)
    assert "never edit repo files yourself" not in rendered
    assert "5. **Implement** the approved plan." in rendered


def test_orchestrated_rendering_keeps_edit_prohibition() -> None:
    rendered = _render_cli(ORCHESTRATED_PRESET, SHIPPED_CATALOG)
    assert "never edit repo files yourself" in rendered


# --- S8 reversion tripwire ---------------------------------------------------


@pytest.mark.parametrize("policy", list(SHIPPED_PRESETS.values()))
@pytest.mark.parametrize("catalog", [SHIPPED_CATALOG, multi_model_catalog()])
def test_s8_anchor_tripwire_on_rendered_output(
    policy: DispatchPolicy, catalog: ModelCatalog
) -> None:
    rendered = _render_cli(policy, catalog)
    assert lint_rendered(rendered, policy.mode) == ()
    assert "$dispatch" not in rendered


def test_s8_rejects_standalone_edit_prohibition() -> None:
    rendered = _render_cli(STANDALONE_PRESET, SHIPPED_CATALOG)
    diagnostics = lint_rendered(
        rendered + " Never edit repo files yourself.", STANDALONE_PRESET.mode
    )
    assert any(item.id == "S8" for item in diagnostics)


def test_s8_rejects_orchestrated_without_edit_prohibition() -> None:
    rendered = _render_cli(ORCHESTRATED_PRESET, SHIPPED_CATALOG)
    stripped = rendered.replace("never edit repo files yourself.", "")
    diagnostics = lint_rendered(stripped, ORCHESTRATED_PRESET.mode)
    assert any(item.id == "S8" for item in diagnostics)


# --- Placeholder hygiene and custom prompts ---------------------------------


def test_rendered_outputs_have_no_placeholders() -> None:
    for policy in SHIPPED_PRESETS.values():
        for catalog in (SHIPPED_CATALOG, multi_model_catalog()):
            assert "$dispatch" not in _render_cli(policy, catalog)
            assert "$dispatch" not in _render_task(policy, catalog)


def test_unresolved_placeholder_is_rejected() -> None:
    skeleton = SystemPrompt.CLI.read() + "\n$dispatch_unknown\n"
    with pytest.raises(DispatchRenderError, match=r"\$dispatch_unknown"):
        render_cli_prompt(
            ORCHESTRATED_PRESET, _shape(ORCHESTRATED_PRESET, SHIPPED_CATALOG), skeleton
        )


def test_custom_prompt_without_placeholders_passes_through() -> None:
    custom = "# Custom prompt\n\nNothing to render here.\n"
    assert (
        render_cli_prompt(
            ORCHESTRATED_PRESET, _shape(ORCHESTRATED_PRESET, SHIPPED_CATALOG), custom
        )
        == custom
    )


# --- Task tool description injection (contracts L) --------------------------


def _task_spec_description(manager: ToolManager) -> str:
    specs = {spec.name: spec for spec in manager.available_tool_specs()}
    return specs["task"].description


def test_tool_manager_serves_policy_bound_task_description() -> None:
    config = build_test_vibe_config()
    manager = ToolManager(lambda: config)
    description = _task_spec_description(manager)
    assert "$dispatch" not in description
    # The shipped catalog is single-model: the served routing is task-kind.
    assert "Route by the work, not the profile default" in description
    assert "Delegation is optional" in description
    assert description == _render_task(STANDALONE_PRESET, SHIPPED_CATALOG)


def test_task_description_is_keyed_by_policy() -> None:
    single = build_test_vibe_config()
    multi = build_test_vibe_config().attach_catalog_snapshot(
        CatalogSnapshot(
            multi_model_catalog(), "rendering-test", dispatch=ORCHESTRATED_PRESET
        )
    )
    single_description = _task_spec_description(ToolManager(lambda: single))
    multi_description = _task_spec_description(ToolManager(lambda: multi))
    assert single_description != multi_description
    assert "Delegation is optional" in single_description
    assert 'config={"model": "@small"}' in multi_description
    # Each manager's spec cache serves its own policy's description.
    assert _task_spec_description(ToolManager(lambda: single)) == single_description


def test_tool_manager_serves_standalone_task_description() -> None:
    config = build_test_vibe_config().attach_catalog_snapshot(
        CatalogSnapshot(SHIPPED_CATALOG, "rendering-test", dispatch=STANDALONE_PRESET)
    )
    description = _task_spec_description(ToolManager(lambda: config))
    assert "Delegation is optional" in description
    assert description == _render_task(STANDALONE_PRESET, SHIPPED_CATALOG)


@pytest.mark.parametrize("runtime_description", [None, "Runtime-rendered description."])
def test_custom_task_prompt_override_wins_without_policy_binding(
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
        manager._install_task_description("Next runtime policy.")
        assert _task_spec_description(manager) == "Custom task description."
