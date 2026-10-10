"""Rendered dispatch prompts: goldens, compatibility, and the S8 tripwire."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from chartreux.core.dispatch.lint import RosterShape, lint_rendered, parse_references
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
from chartreux.core.model_catalog.resolver import ModelResolver
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.prompts import SystemPrompt
from chartreux.core.system_prompt import _interpolate_prompt
from chartreux.core.tools.manager import ToolManager
from tests.conftest import (
    build_test_vibe_config,
    multi_model_catalog,
    single_model_catalog,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "dispatch"
PINNED_DATE = date(2000, 1, 1)
ROSTER_ROLES = ("@scout", "@worker", "@heavy", "@orchestrator")

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
        # The single-model case uses an explicit one-model fixture, not
        # SHIPPED_CATALOG: the shipped roster is two-model since the ML4 entry
        # landed, and the single-model rendering must stay stable regardless of
        # how the shipped roster evolves (ADR 0018-G.1's byte baseline is
        # superseded by these regenerated goldens; see the compatibility tests
        # below).
        ("orchestrated-singlemodel", ORCHESTRATED_PRESET, single_model_catalog()),
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


# --- WP0 baseline compatibility (contract G.1, superseded) ------------------
#
# ADR 0018-G.1 pinned the orchestrated multi-model rendering to the WP0 legacy
# routing bytes. The roster rename (large/medium/small -> worker/scout/heavy)
# rewrote the curated prose blocks, so that byte baseline is superseded by the
# regenerated active goldens; the legacy captures under legacy-orchestrated/
# are preserved untouched as historical artifacts.


def test_orchestrated_multi_model_routing_uses_the_current_roster_roles() -> None:
    shape = _shape(ORCHESTRATED_PRESET, multi_model_catalog())
    region = render_routing_region(ORCHESTRATED_PRESET, shape)
    assert "`@scout` for search, grep, exploration, verification" in region
    assert "`@worker` for all substantive implementation" in region
    assert "`@heavy` for architecture, cross-subsystem design" in region
    task_region = render_task_regions(ORCHESTRATED_PRESET, shape)["routing"]
    assert 'config={"model": "@scout"}' in task_region
    assert "`@heavy` launches are for architecture, design, planning" in task_region


def test_orchestrated_multi_model_golden_supersedes_legacy_bytes() -> None:
    cli = (FIXTURES / "orchestrated" / "cli.md").read_text()
    task = (FIXTURES / "orchestrated" / "task.md").read_text()
    legacy_cli = (FIXTURES / "legacy-orchestrated" / "cli-routing.md").read_text()
    legacy_task = (FIXTURES / "legacy-orchestrated" / "task-routing.md").read_text()
    # The regenerated goldens render the renamed roster; the WP0 legacy bytes
    # no longer appear (G.1 superseded by the regenerated goldens).
    assert legacy_cli.rstrip("\n") not in cli
    assert legacy_task.rstrip("\n") not in task
    assert "`@scout` for search" in cli
    assert 'config={"model": "@scout"}' in task


# --- Roster-aware rendering (contracts D/G) --------------------------------


def test_single_model_routing_is_tier_free() -> None:
    shape = _shape(ORCHESTRATED_PRESET, single_model_catalog())
    assert shape.single_model
    region = render_routing_region(ORCHESTRATED_PRESET, shape)
    task_region = render_task_regions(ORCHESTRATED_PRESET, shape)["routing"]
    for role in ROSTER_ROLES:
        assert role not in region
        assert role not in task_region
    assert "Route by task kind through the configured slots" in region
    assert "`mechanical` (profile `worker`) takes search" in region


def test_multi_model_routing_uses_tiers() -> None:
    shape = _shape(ORCHESTRATED_PRESET, multi_model_catalog())
    assert not shape.single_model
    region = render_routing_region(ORCHESTRATED_PRESET, shape)
    assert "`@scout` for search, grep, exploration, verification" in region
    assert "`@worker` for all substantive implementation" in region


def test_shipped_roster_renders_deep_review_without_three_model_diversity_claim() -> (
    None
):
    # The shipped two-model roster renders deep review honestly: ML4 analytical
    # and peer seats plus the GLM execution seat, with no diversity claim.
    shape = _shape(ORCHESTRATED_PRESET, SHIPPED_CATALOG)
    assert not shape.single_model
    region = render_routing_region(ORCHESTRATED_PRESET, shape)
    assert (
        "| `analytical-reviewer` | `reviewer` | `@heavy` | `mistral-large-4` | high |"
        in region
    )
    assert (
        "| `peer-reviewer` | `reviewer` | `@heavy` | `mistral-large-4` | high |"
        in region
    )
    assert (
        "| `execution-reviewer` | `reviewer` | `@worker` | `glm-5-3` | medium |"
        in region
    )


def test_every_role_reference_in_rendered_shipped_prompts_resolves() -> None:
    resolver = ModelResolver(CatalogSnapshot(SHIPPED_CATALOG, "rendered-roles"))
    for policy in SHIPPED_PRESETS.values():
        shape = _shape(policy, SHIPPED_CATALOG)
        texts = (
            _render_cli(policy, SHIPPED_CATALOG),
            _render_task(policy, SHIPPED_CATALOG),
        )
        assert shape.bindings
        for text in texts:
            for role in dict.fromkeys(parse_references(text).roles):
                if role == "role":
                    continue
                assert resolver.resolve(f"@{role}").base_model


def test_roster_shape_ignores_thinking_and_provider_diversity() -> None:
    # One canonical model across thinking levels stays single-model.
    assert _shape(ORCHESTRATED_PRESET, single_model_catalog()).single_model
    # The shipped roster is two-model since the ML4 entry landed.
    assert not _shape(ORCHESTRATED_PRESET, SHIPPED_CATALOG).single_model
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
            "scout": {
                "model": "shared-model",
                "thinking": "low",
                "description": "scout preset",
            },
            "worker": {
                "model": "shared-model",
                "thinking": "medium",
                "description": "worker preset",
            },
            "heavy": {
                "model": "shared-model",
                "thinking": "high",
                "description": "heavy preset",
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
        STANDALONE_PRESET, _shape(STANDALONE_PRESET, single_model_catalog())
    )
    assert "| Slot | Profile | Role | Model | Thinking | Purposes |" in multi
    assert "| `mechanical` | `worker` | `@scout` | `alpha-model` | low |" in multi
    assert "| `mechanical` | `worker` | — | `solo-model` | low |" in single
    for role in ROSTER_ROLES:
        assert role not in single


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
    # Standalone task routing is slot-based regardless of roster size.
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
    assert 'config={"model": "@scout"}' in multi_description
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
