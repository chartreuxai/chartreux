"""User-overlay loading, persistence, and diagnostics for the dispatch table."""

from __future__ import annotations

import logging
from pathlib import Path
import tomllib

from pydantic import ValidationError
import pytest
import tomli_w

from chartreux.core.config.default_orchestrator import build_default_orchestrator
from chartreux.core.dispatch import (
    DEFAULT_DISPATCH_MODE,
    SHIPPED_PRESETS,
    SHIPPED_PURPOSES,
    STANDALONE_PRESET,
    DispatchPolicy,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import (
    CatalogLoadError,
    CatalogSnapshot,
    load_catalog,
    merge_catalog_overlay,
    merge_dispatch_overlay,
    resolve_dispatch_overlay,
)
from chartreux.core.model_catalog.resolver import ModelResolver
from chartreux.utils.paths import get_chartreux_home


def test_roles_only_overlay_loads_exactly_as_before(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text('[roles.orchestrator]\nthinking = "medium"\n')

    snapshot = load_catalog(path)

    assert snapshot.catalog.roles["orchestrator"].thinking == "medium"
    assert snapshot.catalog.roles["orchestrator"].model == "glm-5-3"
    # No automatic mode inference from legacy tier entries.
    assert snapshot.dispatch == SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE]
    assert snapshot.dispatch.mode == DEFAULT_DISPATCH_MODE
    assert snapshot.dispatch_diagnostics == ()


def test_sparse_patch_of_a_removed_role_fails_validation() -> None:
    # A sparse [roles.small] patch no longer inherits the shipped model and
    # thinking fields (the tier roles were removed with the roster rename), so
    # it fails schema validation with a clear error instead of retargeting.
    with pytest.raises(ValidationError, match="model"):
        merge_catalog_overlay(
            SHIPPED_CATALOG, {"roles": {"small": {"thinking": "low"}}}
        )


def test_complete_old_role_definition_remains_a_user_defined_role() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG,
        {
            "roles": {
                "small": {
                    "description": "custom preset",
                    "model": "glm-5-3",
                    "thinking": "low",
                }
            }
        },
    )
    resolver = ModelResolver(CatalogSnapshot(catalog, "user-role"))
    assert catalog.roles["small"].model == "glm-5-3"
    assert resolver.resolve("@small").base_model == "glm-5-3"


def test_dispatch_overlay_binding_a_removed_role_fails_with_actionable_lint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.toml"
    path.write_text('[dispatch.slots.implementor]\nrole = "@small"\n')

    snapshot = load_catalog(path)

    assert snapshot.dispatch == STANDALONE_PRESET
    assert snapshot.dispatch_diagnostics
    diagnostic = snapshot.dispatch_diagnostics[0]
    assert "S1" in diagnostic
    assert "@small" in diagnostic
    assert "implementor" in diagnostic


def test_missing_models_toml_selects_the_shipped_default_dispatch(
    tmp_path: Path,
) -> None:
    snapshot = load_catalog(tmp_path / "models.toml")
    assert snapshot.catalog == SHIPPED_CATALOG
    assert snapshot.dispatch == SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE]
    assert snapshot.dispatch_diagnostics == ()


def test_dispatch_mode_selects_preset_and_absent_fields_inherit(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text('[dispatch]\nmode = "standalone"\n')

    snapshot = load_catalog(path)

    assert snapshot.dispatch == STANDALONE_PRESET
    assert snapshot.dispatch_diagnostics == ()
    assert snapshot.catalog == SHIPPED_CATALOG


def test_dispatch_overlay_sparse_semantics_and_list_replacement() -> None:
    raw = {
        "slots": {
            "implementor": {"purposes": ["implementation", "search"]},
            "custom": {
                "profile": "worker",
                "role": "@scout",
                "purposes": ["mechanical-edit"],
                "implements": "routine",
                "review_eligible": False,
            },
        },
        "vocabulary": {"custom-work": {"description": "Bounded custom work."}},
    }

    policy = merge_dispatch_overlay(raw)

    # List-valued fields replace; absent slot fields inherit the preset.
    assert policy.slots["implementor"].purposes == ("implementation", "search")
    assert policy.slots["implementor"].role == "@worker"
    assert policy.slots["implementor"].profile == "worker"
    # A new slot carries its own complete definition.
    assert policy.slots["custom"].purposes == ("mechanical-edit",)
    # Vocabulary additions extend the shipped entries.
    assert policy.vocabulary["custom-work"].description == "Bounded custom work."
    assert set(SHIPPED_PURPOSES) < set(policy.vocabulary)
    assert policy.mode == DEFAULT_DISPATCH_MODE
    # The resolved policy round-trips through TOML serialization.
    assert merge_dispatch_overlay(tomllib.loads(tomli_w.dumps(raw))) == policy
    assert (
        DispatchPolicy.model_validate(tomllib.loads(tomli_w.dumps(policy.model_dump())))
        == policy
    )


def test_dispatch_vocabulary_merges_per_entry_over_shipped() -> None:
    policy = merge_dispatch_overlay({
        "vocabulary": {"search": {"description": "Custom."}}
    })
    assert policy.vocabulary["search"].description == "Custom."
    assert policy.vocabulary["implementation"] == SHIPPED_PURPOSES["implementation"]


@pytest.mark.parametrize(
    "patch",
    [
        {"mode": "implement directly and delegate as needed"},
        {"mode": "custom"},
        {"mode": 3},
        {"unknown": "value"},
        {"slots": {"implementor": {"purposes": ["typo"]}}},
        {"slots": {"implementor": {"implements": "sometimes"}}},
        {"slots": {"implementor": {"role": "medium"}}},
        {"slots": {"implementor": "worker"}},
        {"slots": "worker"},
        {"vocabulary": "not-a-table"},
        {"vocabulary": {"custom": {}}},
    ],
)
def test_dispatch_overlay_schema_errors_fail_at_load(patch: object) -> None:
    with pytest.raises((ValidationError, ValueError)):
        merge_dispatch_overlay(patch)  # type: ignore[arg-type]


def test_dispatch_content_participates_in_revision(tmp_path: Path) -> None:
    plain = tmp_path / "plain.toml"
    plain.write_text('[roles.scout]\nthinking = "low"\n')
    selected = tmp_path / "selected.toml"
    selected.write_text(
        '[roles.scout]\nthinking = "low"\n\n[dispatch]\nmode = "orchestrated"\n'
    )
    edited = tmp_path / "edited.toml"
    edited.write_text(
        '[roles.scout]\nthinking = "low"\n\n[dispatch]\nmode = "standalone"\n\n'
        '[dispatch.slots.implementor]\nrole = "@scout"\n'
    )

    base = load_catalog(plain)
    chosen = load_catalog(selected)
    changed = load_catalog(edited)

    assert base.catalog == chosen.catalog == changed.catalog
    assert len({base.revision, chosen.revision, changed.revision}) == 3
    # Revision derivation is deterministic for identical content.
    assert load_catalog(selected).revision == chosen.revision


def test_invalid_dispatch_falls_back_to_standalone_with_visible_diagnostic(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "models.toml"
    path.write_text("""
[providers.custom]
api_base = "https://custom.test"

[models.custom-model]
deployments = [{ provider = "custom", name = "custom-wire" }]

[dispatch]
mode = "standalone"

[dispatch.slots.implementor]
purposes = ["not-a-purpose"]
""")

    with caplog.at_level(logging.WARNING, logger="vibe"):
        snapshot = load_catalog(path)

    assert snapshot.dispatch == STANDALONE_PRESET
    assert len(snapshot.dispatch_diagnostics) == 1
    diagnostic = snapshot.dispatch_diagnostics[0]
    assert "standalone" in diagnostic
    assert str(path) in diagnostic
    # Valid provider/model entries in the same file are not discarded.
    assert "custom" in snapshot.catalog.providers
    assert "custom-model" in snapshot.catalog.models
    # The invalid file is never overwritten.
    assert "not-a-purpose" in path.read_text()
    # The fallback diagnostic is visible at startup.
    assert any(
        "standalone" in record.getMessage() and str(path) in record.getMessage()
        for record in caplog.records
    )


def test_non_table_dispatch_is_atomically_rejected_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text('dispatch = "route everything to the biggest model"\n')

    snapshot = load_catalog(path)

    assert snapshot.dispatch == STANDALONE_PRESET
    assert snapshot.dispatch_diagnostics
    assert snapshot.catalog == SHIPPED_CATALOG


def test_invalid_dispatch_does_not_mask_fatal_catalog_errors(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text('[providers.bad]\nunknown = true\n\n[dispatch]\nmode = "bogus"\n')

    with pytest.raises(CatalogLoadError, match="Invalid model catalog"):
        load_catalog(path)


def test_invalid_dispatch_revision_supports_repair_and_rejects_stale_save(
    tmp_path: Path,
) -> None:
    from chartreux.core.model_catalog.contracts import (
        CatalogChanges,
        CatalogValidationError,
        CatalogWriteResult,
    )
    from chartreux.core.model_catalog.loader import CatalogStore

    path = tmp_path / "models.toml"
    original = '[dispatch]\nmode = "bogus"\n'
    path.write_text(original)
    snapshot = load_catalog(path)
    assert path.read_text() == original
    path.write_text('[dispatch]\nmode = "other-invalid"\n')
    changed = load_catalog(path)
    assert changed.dispatch == snapshot.dispatch
    assert changed.revision != snapshot.revision
    store = CatalogStore(path)
    stale = store.apply_changes(
        CatalogChanges(
            provider_id="",
            provider={},
            expected_revision=snapshot.revision,
            dispatch={"mode": "standalone"},
        )
    )
    assert isinstance(stale, CatalogValidationError)
    assert '"other-invalid"' in path.read_text()
    repaired = store.apply_changes(
        CatalogChanges(
            provider_id="",
            provider={},
            expected_revision=changed.revision,
            dispatch={"mode": "standalone"},
        )
    )
    assert isinstance(repaired, CatalogWriteResult)
    assert not load_catalog(path).dispatch_diagnostics
    assert repaired.snapshot.revision == load_catalog(path).revision


def test_resolve_dispatch_overlay_reports_source_in_diagnostic() -> None:
    policy, diagnostics = resolve_dispatch_overlay(
        {"dispatch": {"mode": "bogus"}}, source="example.toml"
    )
    assert policy == STANDALONE_PRESET
    assert len(diagnostics) == 1
    assert "example.toml" in diagnostics[0]
    assert "standalone" in diagnostics[0]


def test_catalog_overlay_accepts_dispatch_without_catalog_authority() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG, {"dispatch": {"mode": "standalone"}}
    )
    assert catalog == SHIPPED_CATALOG
    with pytest.raises(ValueError, match="unknown fields"):
        merge_catalog_overlay(SHIPPED_CATALOG, {"dispatching": {}})


@pytest.mark.asyncio
async def test_startup_orchestrator_carries_dispatch_fallback() -> None:
    home = get_chartreux_home()
    (home / "models.toml").write_text('[dispatch]\nmode = "bogus"\n')

    orchestrator = await build_default_orchestrator()

    snapshot = orchestrator.config.catalog_snapshot
    assert snapshot is not None
    assert snapshot.dispatch == STANDALONE_PRESET
    assert snapshot.dispatch_diagnostics
    assert snapshot.catalog == SHIPPED_CATALOG
