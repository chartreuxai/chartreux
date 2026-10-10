"""Batch A review regressions at policy write, render, and resume boundaries."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from chartreux.core.agents.models import BUILTIN_SUBAGENTS
from chartreux.core.dispatch.lint import lint_catalog
from chartreux.core.dispatch.presets import SHIPPED_PRESETS
from chartreux.core.dispatch.renderer import (
    render_routing_region,
    render_task_regions,
    roster_for,
)
from chartreux.core.dispatch.schema import DispatchMode, DispatchPolicy
from chartreux.core.dispatch.session import (
    SessionPolicyError,
    bind_policy,
    resume_policy,
)
from chartreux.core.model_catalog.contracts import (
    CatalogChanges,
    CatalogValidationError,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import (
    CatalogSnapshot,
    CatalogStore,
    merge_dispatch_overlay,
)
from tests.conftest import build_test_vibe_config, multi_model_catalog


@pytest.mark.parametrize("mode", list(SHIPPED_PRESETS))
@pytest.mark.parametrize("catalog", [SHIPPED_CATALOG, multi_model_catalog()])
def test_overlay_authority_and_curated_blocks_in_all_renderings(mode, catalog):
    policy = merge_dispatch_overlay({
        "mode": mode,
        "vocabulary": {"audit": {"description": "Inspect widget invariants."}},
        "slots": {"mechanical": {"role": "@heavy", "purposes": ["audit"]}},
        "failure_routing": "widget: use slot `mechanical`.",
        "contrasts": "Use slot `mechanical` for purpose `audit`; use slot `implementor` for implementation.",
    })
    shape = roster_for(catalog, policy)
    for text in (
        render_routing_region(policy, shape),
        render_task_regions(policy, shape)["routing"],
    ):
        assert "Inspect widget invariants." in text
        assert "| `mechanical` | `worker` |" in text
        assert "widget: use slot `mechanical`." in text
        assert policy.contrasts in text
        assert policy.compositions in text
        assert str(shape.slot_bindings["mechanical"][1]) in text


def test_empty_roster_renders_every_unavailable_slot_honestly():
    policy = SHIPPED_PRESETS[DispatchMode.ORCHESTRATED]
    catalog = SHIPPED_CATALOG.model_copy(update={"models": {}})
    shape = roster_for(catalog, policy)
    for text in (
        render_routing_region(policy, shape),
        render_task_regions(policy, shape)["routing"],
    ):
        assert "no slot binding is available" in text.lower()
        for name in policy.slots:
            assert f"Slot `{name}` unavailable:" in text
        assert "`@scout`" not in text
        assert "`@worker`" not in text
        assert "`@heavy`" not in text


@pytest.mark.parametrize(
    "dispatch",
    [
        {"compositions": "review.deep: use slot `missing`."},
        {"compositions": "review.deep: use slots `reviewer`, `reviewer`."},
        {"failure_routing": "widget: use slot `missing`."},
        {"instructions": "Use purpose `missing`."},
        {"instructions": "Use `@missing`."},
    ],
)
def test_write_boundary_rejects_dangling_references(tmp_path, dispatch):
    path = tmp_path / "models.toml"
    path.write_text("# preserve me\n")
    result = CatalogStore(path).apply_changes(CatalogChanges("", {}, dispatch=dispatch))
    assert isinstance(result, CatalogValidationError)
    assert path.read_text() == "# preserve me\n"


def test_prose_emails_and_handles_are_not_role_references():
    policy = merge_dispatch_overlay({
        "instructions": "Contact owner@example.com or @maintainer."
    })
    assert lint_catalog(policy, SHIPPED_CATALOG) == ()


@pytest.mark.parametrize("field", ["slot", "profile"])
def test_dispatch_names_reject_table_delimiters(field):
    raw = SHIPPED_PRESETS[DispatchMode.ORCHESTRATED].model_dump(mode="json")
    if field == "slot":
        raw["slots"]["bad|name"] = raw["slots"].pop("mechanical")
    else:
        raw["slots"]["mechanical"]["profile"] = "bad|name"
    with pytest.raises(ValidationError):
        DispatchPolicy.model_validate(raw)


def test_resume_without_catalog_snapshot_validates_saved_roles():
    config = build_test_vibe_config()
    bound = bind_policy(config)
    slots = dict(bound.policy.slots)
    slots["mechanical"] = slots["mechanical"].model_copy(update={"role": "@removed"})
    bound = bound.model_copy(
        update={"policy": bound.policy.model_copy(update={"slots": slots})}
    )
    config.attach_catalog_snapshot(None)
    with pytest.raises(SessionPolicyError, match="role removed"):
        resume_policy(config, bound.model_dump(mode="json"), BUILTIN_SUBAGENTS)


def test_resume_without_catalog_snapshot_uses_effective_shipped_catalog():
    config = build_test_vibe_config()
    bound = bind_policy(config)
    config.attach_catalog_snapshot(None)
    resumed = resume_policy(config, bound.model_dump(mode="json"), BUILTIN_SUBAGENTS)
    assert resumed.bindings == bound.bindings
    assert resumed.render_policy.mode == bound.render_policy.mode


def test_resume_removed_secondary_model_retains_orchestration_and_bindings():
    catalog = multi_model_catalog()
    config = build_test_vibe_config().attach_catalog_snapshot(
        CatalogSnapshot(
            catalog, "before", dispatch=SHIPPED_PRESETS[DispatchMode.ORCHESTRATED]
        )
    )
    bound = bind_policy(config)
    models = dict(catalog.models)
    models.pop("alpha-model")
    config.attach_catalog_snapshot(
        CatalogSnapshot(catalog.model_copy(update={"models": models}), "after")
    )
    resumed = resume_policy(config, bound.model_dump(mode="json"), BUILTIN_SUBAGENTS)
    config.attach_dispatch_policy(resumed)
    assert resumed.render_policy.mode == "orchestrated"
    assert resumed.bindings == bound.bindings
    assert any("slot 'mechanical':" in note for note in config.validation_warnings)
    assert not any("slot 'advisor':" in note for note in config.validation_warnings)
    from chartreux.core.agents.launch import _resolved_model
    from chartreux.core.subagents import InvalidLaunchModelError

    with pytest.raises(InvalidLaunchModelError):
        _resolved_model(config, "@scout")


def test_fallback_diagnostic_is_projected_to_clients():
    from chartreux.app_server._projection import project_config_view
    from chartreux.core.model_catalog.loader import resolve_dispatch_overlay

    policy, diagnostics = resolve_dispatch_overlay({"dispatch": {"mode": "invalid"}})
    config = build_test_vibe_config().attach_catalog_snapshot(
        CatalogSnapshot(
            SHIPPED_CATALOG,
            "fallback",
            dispatch=policy,
            dispatch_diagnostics=diagnostics,
        )
    )
    assert diagnostics[0] in project_config_view(config).validation_warnings


def test_startup_broken_model_does_not_select_standalone():
    from chartreux.core.model_catalog.loader import resolve_dispatch_overlay

    catalog = SHIPPED_CATALOG.model_copy(update={"models": {}})
    policy, _ = resolve_dispatch_overlay(
        {"dispatch": {"mode": "orchestrated"}}, catalog=catalog
    )
    assert policy.mode == "orchestrated"
    config = build_test_vibe_config().attach_catalog_snapshot(
        CatalogSnapshot(catalog, "broken", dispatch=policy)
    )
    bound = bind_policy(config)
    assert bound.render_policy.mode == "orchestrated"
    assert len(bound.diagnostics) == len(policy.slots)
