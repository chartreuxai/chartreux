"""Onboarding-only materialization of the shipped default catalog template.

A fresh install runs without ``models.toml``: the shipped catalog serves the
defaults from code. When onboarding completes without the user having saved a
catalog, this module publishes the shipped default setup as an editable
template, so the file exists and is user-owned from the first completed run.

Materialization is strictly additive. It is invoked only from onboarding,
which is reachable only after legacy reconciliation (legacy catalog tables in
``config.toml`` fail the config load with the migration hint), and publication
is an exclusive create: any existing ``models.toml`` — written by ``chartreux
models migrate``, by an earlier save, or created by another process between
the absence check and the write — always wins.

The template omits ``[providers]`` entirely, so provider tables stay inherited
from the shipped catalog and ``overlaid_providers`` stays empty (no provenance
flip for doctor output, workbench badges, and seeding). Its identity is a
content fingerprint — the sha256 of the rendered template — never a comment
marker, so future template changes are detectable. Materializing is
revision-neutral: the loaded snapshot is identical to the absent-file one.
"""

from __future__ import annotations

from hashlib import sha256
import logging
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any

from pydantic import ValidationError
import tomli_w

from chartreux.core.dispatch import DEFAULT_DISPATCH_MODE, SHIPPED_PRESETS
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import (
    CatalogSnapshot,
    _snapshot,
    _snapshot_for_overlay,
    load_catalog,
)
from chartreux.utils.paths import get_chartreux_home

logger = logging.getLogger(__name__)

# Informational only: this is not a durable identity marker. Any byte rewrite,
# including a TOML rewrite that only strips comments, changes the fingerprint.
_TEMPLATE_HEADER = """\
# Chartreux model catalog — the shipped default setup, ready to edit.
# This file overlays the built-in defaults: deleting an entry restores the
# shipped value. Provider definitions stay inherited from the shipped catalog
# unless you add a [providers] table here.
"""


class MaterializationError(RuntimeError):
    """The rendered template does not round-trip to the shipped defaults."""


def render_default_template() -> str:
    """Render the editable default ``models.toml`` template.

    The template carries the shipped default setup — both model tables with
    their deployments, the four role presets, and sparse ``[dispatch]`` slot
    role bindings — and no provider tables.
    """
    policy = SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE]
    overlay: dict[str, Any] = {
        "models": {
            name: definition.model_dump(
                mode="json", exclude_defaults=True, exclude_none=True
            )
            for name, definition in SHIPPED_CATALOG.models.items()
        },
        "roles": {
            name: definition.model_dump(
                mode="json", exclude_defaults=True, exclude_none=True
            )
            for name, definition in SHIPPED_CATALOG.roles.items()
        },
        # Sparse dispatch: the mode plus slot role bindings only; prose and
        # slot metadata inherit from the shipped preset.
        "dispatch": {
            "mode": DEFAULT_DISPATCH_MODE.value,
            "slots": {name: {"role": slot.role} for name, slot in policy.slots.items()},
        },
    }
    return _TEMPLATE_HEADER + tomli_w.dumps(overlay)


def template_fingerprint(content: str) -> str:
    """Return the reserved exact-rendered-content hash (sha256 hex).

    The fingerprint identifies only these exact bytes. Any rewrite, including
    a TOML rewrite that strips comments without changing effective content,
    changes the hash; it is not rewrite-stable or a drift advisory.
    """
    return sha256(content.encode("utf-8")).hexdigest()


def materialize_default_catalog(path: Path | None = None) -> CatalogSnapshot | None:
    """Publish the default template at ``path`` when it does not exist yet.

    This is the onboarding-only entry point. It writes nothing when the file
    already exists — an existing catalog, whether migrated, saved, or created
    by another process, always wins — and it refuses to write a template that
    fails round-trip validation. Returns the snapshot loaded from the
    materialized file, or ``None`` when nothing was written. Publication can
    succeed before a directory-fsync error; in that case it logs that
    durability is uncertain and still returns the loaded snapshot.
    """
    catalog_path = path or get_chartreux_home() / "models.toml"
    if catalog_path.exists():
        return None
    template = render_default_template()
    _validate_round_trip(template)
    if not _publish_exclusive(catalog_path, template.encode("utf-8")):
        # An external create won the race between the absence check and the
        # publication; materialization loses and never clobbers.
        return None
    return load_catalog(catalog_path)


def _validate_round_trip(template: str) -> None:
    """Refuse to publish a template that diverges from the shipped defaults.

    The rendered template must load through the ordinary overlay path to
    exactly what the runtime serves without a file: the same catalog, the
    same dispatch policy, and the same revision (materialization is
    revision-neutral), with no dispatch diagnostics and no overlaid providers.
    """
    try:
        overlay = tomllib.loads(template)
    except tomllib.TOMLDecodeError as exc:
        raise MaterializationError(f"template is not valid TOML: {exc}") from exc
    try:
        snapshot = _snapshot_for_overlay(overlay, source="<default template>")
    except (ValidationError, ValueError) as exc:
        raise MaterializationError(f"template does not validate: {exc}") from exc
    shipped = _snapshot(SHIPPED_CATALOG)
    divergent = [
        field
        for field, actual, expected in (
            ("catalog", snapshot.catalog, shipped.catalog),
            ("dispatch", snapshot.dispatch, shipped.dispatch),
            ("revision", snapshot.revision, shipped.revision),
        )
        if actual != expected
    ]
    if snapshot.overlaid_providers:
        divergent.append("overlaid_providers")
    if snapshot.dispatch_diagnostics:
        divergent.append("dispatch_diagnostics")
    if divergent:
        raise MaterializationError(
            "template does not round-trip to the shipped defaults "
            f"(divergent: {', '.join(divergent)})"
        )


def _publish_exclusive(path: Path, data: bytes) -> bool:
    """Create ``path`` with ``data`` only when it does not exist yet.

    Returns ``True`` when this call created the file. The payload is staged in
    a unique temporary sibling and hard-linked into place: the link is the
    atomic exclusive create, so a file that appears between the caller's
    absence check and this write (an external create) makes it fail and the
    existing file is left untouched. A crash never leaves a partial template
    at ``path`` — only a dot-prefixed temporary sibling.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        try:
            _fsync_directory(path.parent)
        except OSError as error:
            # The link already published the complete file. Only durability of
            # the directory entry is uncertain; do not retry or undo publication.
            logger.warning(
                "Default catalog materialization could not be fully completed "
                "(published, durability uncertain): %s",
                error,
            )
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
