"""User-overlay loading, persistence, and diagnostics for the model catalog.

The legacy ``[dispatch]`` table is retired (package 16b): the loader reports
it loudly with successor guidance (never merged, never linted, never
rewritten), catalog writes preserve it byte-for-byte, and the revision covers
the served catalog alone.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.core.config.default_orchestrator import build_default_orchestrator
from chartreux.core.model_catalog.contracts import (
    CatalogChanges,
    CatalogValidationError,
    CatalogWriteResult,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import (
    CatalogLoadError,
    CatalogSnapshot,
    CatalogStore,
    load_catalog,
    merge_catalog_overlay,
)
from chartreux.core.model_catalog.resolver import ModelResolver
from chartreux.utils.paths import get_chartreux_home


def test_roles_only_overlay_loads_exactly_as_before(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text('[roles.main]\nthinking = "medium"\n')

    snapshot = load_catalog(path)

    assert snapshot.catalog.roles["main"].thinking == "medium"
    assert snapshot.catalog.roles["main"].model == "glm-5-3"
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


def test_missing_models_toml_loads_shipped_catalog(tmp_path: Path) -> None:
    snapshot = load_catalog(tmp_path / "models.toml")
    assert snapshot.catalog == SHIPPED_CATALOG
    assert snapshot.dispatch_diagnostics == ()


def test_legacy_dispatch_table_is_reported_loudly_with_successor_guidance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.toml"
    path.write_text(
        '[roles.scout]\nthinking = "medium"\n\n[dispatch]\nmode = "standalone"\n'
    )

    snapshot = load_catalog(path)

    # The retired table is reported, never merged: the snapshot carries no
    # dispatch state at all, and the report names the source and the successor.
    assert not hasattr(snapshot, "dispatch")
    assert len(snapshot.dispatch_diagnostics) == 1
    report = snapshot.dispatch_diagnostics[0]
    assert "[dispatch]" in report
    assert "retired" in report
    assert "[orchestration.routes.<purpose>]" in report
    assert str(path) in report
    # Valid catalog entries in the same file are unaffected.
    assert snapshot.catalog.roles["scout"].thinking == "medium"
    assert snapshot.catalog == merge_catalog_overlay(
        SHIPPED_CATALOG, {"roles": {"scout": {"thinking": "medium"}}}
    )


def test_legacy_dispatch_report_is_logged_at_startup(
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
""")

    with caplog.at_level(logging.WARNING, logger="vibe"):
        snapshot = load_catalog(path)

    assert len(snapshot.dispatch_diagnostics) == 1
    report = snapshot.dispatch_diagnostics[0]
    # Valid provider/model entries in the same file are not discarded.
    assert "custom" in snapshot.catalog.providers
    assert "custom-model" in snapshot.catalog.models
    # The retired file is never overwritten.
    assert '[dispatch]\nmode = "standalone"\n' in path.read_text()
    # The loud report is visible at startup.
    assert any(
        "[dispatch]" in record.getMessage() and str(path) in record.getMessage()
        for record in caplog.records
    )
    assert report in {record.getMessage() for record in caplog.records}


def test_non_table_dispatch_is_reported_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text('dispatch = "route everything to the biggest model"\n')

    snapshot = load_catalog(path)

    assert snapshot.dispatch_diagnostics
    assert snapshot.catalog == SHIPPED_CATALOG


def test_legacy_dispatch_does_not_mask_fatal_catalog_errors(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text('[providers.bad]\nunknown = true\n\n[dispatch]\nmode = "bogus"\n')

    with pytest.raises(CatalogLoadError, match="Invalid model catalog"):
        load_catalog(path)


def test_dispatch_content_does_not_participate_in_revision(tmp_path: Path) -> None:
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

    # The revision covers the served catalog alone: a retired legacy table
    # never moves it, so all three overlays share one revision.
    assert base.catalog == chosen.catalog == changed.catalog
    assert base.revision == chosen.revision == changed.revision
    assert chosen.dispatch_diagnostics
    assert changed.dispatch_diagnostics
    # Revision derivation is deterministic for identical content.
    assert load_catalog(selected).revision == chosen.revision


def test_stale_catalog_save_rejected_and_dispatch_table_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.toml"
    original = '[dispatch]\nmode = "standalone"\n\n[roles.scout]\nthinking = "low"\n'
    path.write_text(original)
    snapshot = load_catalog(path)
    assert path.read_text() == original
    # A [dispatch]-only edit does not move the revision.
    path.write_text(
        '[dispatch]\nmode = "orchestrated"\n\n[roles.scout]\nthinking = "low"\n'
    )
    dispatch_only = load_catalog(path)
    assert dispatch_only.revision == snapshot.revision
    # A catalog edit does: the stale-save rejection keys on catalog content.
    path.write_text(
        '[dispatch]\nmode = "orchestrated"\n\n[roles.scout]\nthinking = "medium"\n'
    )
    changed = load_catalog(path)
    assert changed.revision != snapshot.revision
    store = CatalogStore(path)
    roles = {"main": {"model": "glm-5-3", "thinking": "high"}}
    stale = store.apply_changes(
        CatalogChanges(
            provider_id="",
            provider={},
            roles=roles,
            expected_revision=snapshot.revision,
        )
    )
    assert isinstance(stale, CatalogValidationError)
    saved = store.apply_changes(
        CatalogChanges(
            provider_id="", provider={}, roles=roles, expected_revision=changed.revision
        )
    )
    assert isinstance(saved, CatalogWriteResult)
    # The catalog writer never repairs or rewrites the retired dispatch table.
    assert '[dispatch]\nmode = "orchestrated"\n' in path.read_text()
    assert load_catalog(path).dispatch_diagnostics
    assert saved.snapshot.revision == load_catalog(path).revision


def test_catalog_overlay_accepts_dispatch_without_catalog_authority() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG, {"dispatch": {"mode": "standalone"}}
    )
    assert catalog == SHIPPED_CATALOG
    with pytest.raises(ValueError, match="unknown fields"):
        merge_catalog_overlay(SHIPPED_CATALOG, {"dispatching": {}})


@pytest.mark.asyncio
async def test_startup_orchestrator_reports_legacy_dispatch() -> None:
    home = get_chartreux_home()
    (home / "models.toml").write_text('[dispatch]\nmode = "bogus"\n')

    orchestrator = await build_default_orchestrator()

    snapshot = orchestrator.config.catalog_snapshot
    assert snapshot is not None
    assert len(snapshot.dispatch_diagnostics) == 1
    report = snapshot.dispatch_diagnostics[0]
    assert "retired" in report
    assert "[orchestration.routes.<purpose>]" in report
    assert snapshot.catalog == SHIPPED_CATALOG
    # The loud report reaches the served configuration's validation warnings.
    assert report in orchestrator.config.validation_warnings
