from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tomllib
from typing import Any

import pytest

from chartreux.core.dispatch import DEFAULT_DISPATCH_MODE, SHIPPED_PRESETS
from chartreux.core.model_catalog import materialization
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import (
    _snapshot,
    _snapshot_for_overlay,
    load_catalog,
)
from chartreux.core.model_catalog.materialization import (
    MaterializationError,
    materialize_default_catalog,
    render_default_template,
    template_fingerprint,
)


def _parsed_template() -> dict[str, Any]:
    return tomllib.loads(render_default_template())


def test_template_carries_the_shipped_default_setup_without_providers() -> None:
    template = _parsed_template()

    # No provider tables: providers stay inherited from the shipped catalog,
    # which keeps overlaid_providers empty (no provenance flip).
    assert "providers" not in template
    assert set(template["models"]) == set(SHIPPED_CATALOG.models)
    assert set(template["roles"]) == set(SHIPPED_CATALOG.roles)
    for name, definition in SHIPPED_CATALOG.models.items():
        rendered = template["models"][name]
        assert rendered["thinking"] == definition.thinking
        assert [
            (deployment["provider"], deployment["name"])
            for deployment in rendered["deployments"]
        ] == [
            (deployment.provider, deployment.name)
            for deployment in definition.deployments
        ]
    for name, definition in SHIPPED_CATALOG.roles.items():
        rendered = template["roles"][name]
        assert rendered["model"] == definition.model
        assert rendered["thinking"] == definition.thinking

    # Sparse dispatch: the mode plus slot role bindings only; prose and slot
    # metadata inherit from the shipped preset.
    policy = SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE]
    dispatch = template["dispatch"]
    assert dispatch["mode"] == DEFAULT_DISPATCH_MODE.value
    assert set(dispatch) == {"mode", "slots"}
    assert set(dispatch["slots"]) == set(policy.slots)
    for name, slot in policy.slots.items():
        assert dispatch["slots"][name] == {"role": slot.role}


def test_template_round_trips_to_the_shipped_snapshot() -> None:
    snapshot = _snapshot_for_overlay(_parsed_template(), source="<default template>")
    shipped = _snapshot(SHIPPED_CATALOG)

    assert snapshot.catalog == shipped.catalog
    assert snapshot.dispatch == shipped.dispatch
    assert snapshot.revision == shipped.revision
    assert snapshot.overlaid_providers == frozenset()
    assert snapshot.dispatch_diagnostics == ()


def test_template_fingerprint_is_derived_from_content() -> None:
    template = render_default_template()

    assert (
        template_fingerprint(template) == sha256(template.encode("utf-8")).hexdigest()
    )
    # Rendering is deterministic, so the fingerprint identifies the revision.
    assert template_fingerprint(render_default_template()) == template_fingerprint(
        template
    )
    # Exact bytes, not semantic TOML: stripping the informational header
    # changes the reserved hash even though effective content is unchanged.
    rewritten = template.removeprefix(
        "# Chartreux model catalog — the shipped default setup, ready to edit.\n"
    )
    assert tomllib.loads(rewritten) == tomllib.loads(template)
    assert template_fingerprint(rewritten) != template_fingerprint(template)


def test_materialize_publishes_a_revision_neutral_template(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"

    snapshot = materialize_default_catalog(path)

    assert snapshot is not None
    assert path.read_text(encoding="utf-8") == render_default_template()
    assert sha256(path.read_bytes()).hexdigest() == template_fingerprint(
        render_default_template()
    )
    # Materialize -> load is equivalent to loading the in-code defaults.
    loaded = load_catalog(path)
    assert loaded.catalog == SHIPPED_CATALOG
    assert loaded.dispatch == SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE]
    assert loaded.overlaid_providers == frozenset()
    assert loaded.revision == load_catalog(tmp_path / "absent.toml").revision
    assert loaded.revision == snapshot.revision
    # Publication leaves no temporary siblings behind.
    assert list(tmp_path.iterdir()) == [path]


def test_materialize_reports_directory_fsync_failure_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "models.toml"
    link_calls = 0
    original_link = materialization.os.link

    def counted_link(source: Path, target: Path) -> None:
        nonlocal link_calls
        link_calls += 1
        original_link(source, target)

    def fail_directory_fsync(directory: Path) -> None:
        raise OSError("directory fsync failed")

    monkeypatch.setattr(materialization.os, "link", counted_link)
    monkeypatch.setattr(materialization, "_fsync_directory", fail_directory_fsync)

    snapshot = materialize_default_catalog(path)

    assert snapshot is not None
    assert path.read_text(encoding="utf-8") == render_default_template()
    assert link_calls == 1
    assert list(tmp_path.iterdir()) == [path]
    assert "published, durability uncertain" in caplog.text

    path = tmp_path / "models.toml"
    existing = b'[providers.custom]\napi_base = "https://custom.test"\n'
    path.write_bytes(existing)

    assert materialize_default_catalog(path) is None

    assert path.read_bytes() == existing
    assert list(tmp_path.iterdir()) == [path]


def test_materialize_loses_an_external_create_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "models.toml"
    external = b'[providers.custom]\napi_base = "https://custom.test"\n'
    original = materialization._validate_round_trip

    def create_external(template: str) -> None:
        # Another process creates the catalog between the absence check and
        # the exclusive publication.
        path.write_bytes(external)
        original(template)

    monkeypatch.setattr(materialization, "_validate_round_trip", create_external)

    assert materialize_default_catalog(path) is None

    assert path.read_bytes() == external
    assert list(tmp_path.iterdir()) == [path]


def test_materialize_refuses_a_template_that_diverges_from_shipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "models.toml"
    divergent = render_default_template().replace("input = 1.4", "input = 9.9")
    assert divergent != render_default_template()
    monkeypatch.setattr(materialization, "render_default_template", lambda: divergent)

    with pytest.raises(MaterializationError, match="round-trip"):
        materialize_default_catalog(path)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_materialize_refuses_a_template_that_does_not_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "models.toml"
    monkeypatch.setattr(
        materialization,
        "render_default_template",
        lambda: "[models.broken]\ndeployments = []\n",
    )

    with pytest.raises(MaterializationError, match="validate"):
        materialize_default_catalog(path)

    assert not path.exists()


def test_publish_exclusive_creates_once_and_never_replaces(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"

    assert materialization._publish_exclusive(path, b"first\n")
    assert path.read_bytes() == b"first\n"
    assert not materialization._publish_exclusive(path, b"second\n")
    assert path.read_bytes() == b"first\n"
    assert list(tmp_path.iterdir()) == [path]


def test_publish_exclusive_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "home" / "models.toml"

    assert materialization._publish_exclusive(path, b"template\n")

    assert path.read_bytes() == b"template\n"


def test_materialize_uses_the_chartreux_home_by_default(config_dir: Path) -> None:
    catalog_path = config_dir / "models.toml"

    snapshot = materialize_default_catalog()

    assert snapshot is not None
    assert catalog_path.read_text(encoding="utf-8") == render_default_template()
    # A second attempt is a no-op: the file now exists.
    assert materialize_default_catalog() is None
    assert catalog_path.read_text(encoding="utf-8") == render_default_template()
