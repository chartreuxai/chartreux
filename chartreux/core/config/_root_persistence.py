"""Purpose-built user-layer root persistence: the only sanctioned roots writer and saved-roots reader."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from chartreux.core.config._root_authority import ROOTS_FIELD, validate_root_source
from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import normalize_authorized_roots
from chartreux.core.config.types import (
    ConcurrencyConflictError,
    ConfigDurabilityError,
    ConfigSaveResult,
)

if TYPE_CHECKING:
    from chartreux.core.config.orchestrator import ConfigOrchestrator
    from chartreux.core.config.schema import ConfigSchema


@dataclass(frozen=True, slots=True)
class SavedRootsRead:
    """Read-only saved-roots projection for one project (paths and revision only).

    ``roots`` is empty when the project has no saved grants. ``unavailable``
    marks a read that could not be sourced from disk: no actual user layer is
    installed (``"no_user_source"``), or the backing store could not be read or
    its roots interpreted (``"read"``). A missing user file is not a failure:
    it reads as no saved grants with the missing-store revision.
    """

    roots: tuple[str, ...] = ()
    user_revision: str | None = None
    unavailable: Literal["no_user_source", "read"] | None = None


def select_actual_user_layer(
    layers: Iterable[ConfigLayer[RawConfig]],
) -> UserConfigLayer:
    """Return the one installed actual user layer, or fail closed.

    Exact concrete types only: a subclass claiming user origin is not an
    actual user source, and roots authority never derives from a layer name.
    """
    installed = [layer for layer in layers if type(layer) is UserConfigLayer]
    if len(installed) != 1:
        raise ValueError(
            "Root persistence requires exactly one installed actual user source"
        )
    return installed[0]


def merge_project_root_grant(
    raw: Mapping[str, Any], *, project: Path, root: Path
) -> dict[str, Any]:
    """Merge one approved root into one project's entry of a disk document.

    Only the target project's saved entry changes; every other entry and all
    unrelated settings are preserved from *raw*. Canonicalization follows
    ``normalize_authorized_roots``, so a duplicate canonical root merges as a
    no-op and relative or malformed definitions raise ``ValueError``.
    """
    merged = dict(raw)
    saved = merged.get(ROOTS_FIELD)
    if saved is None:
        entries: dict[str, list[str]] = {}
    elif isinstance(saved, dict) and all(
        isinstance(key, str)
        and isinstance(roots, list)
        and all(isinstance(item, str) for item in roots)
        for key, roots in saved.items()
    ):
        entries = dict(saved)
    else:
        raise ValueError(f"Invalid {ROOTS_FIELD} definition (source: user)")
    canonical = normalize_authorized_roots(entries)
    project_path = Path(project).expanduser()
    if not project_path.is_absolute():
        raise ValueError(f"Invalid {ROOTS_FIELD} project key (source: user)")
    key = str(project_path.resolve())
    canonical[key] = [*canonical.get(key, []), str(root)]
    merged[ROOTS_FIELD] = normalize_authorized_roots(canonical)
    return merged


async def persist_project_root_grant[S: ConfigSchema](
    orchestrator: ConfigOrchestrator[S],
    *,
    project: Path,
    root: Path,
    expected_revision: str,
) -> ConfigSaveResult:
    """Save one approved project root grant without publishing runtime state.

    The caller must hold the orchestrator mutation lock and have admitted a
    persisting (root) orchestrator. The user layer is force-loaded through a
    private copy, so the merge reads disk rather than the in-memory view and
    the live layer cache is never touched. The write is an optimistic
    ``save_checked`` replacement; cancellation and durability uncertainty are
    reported in the result. Nothing is published: live caches, the accepted
    token, and the merged config stay unchanged (application is "unchanged"),
    and the caller applies the session grant itself.
    """
    try:
        layer = select_actual_user_layer(orchestrator.layers)
        staged = copy.deepcopy(layer)
        loaded = await staged.load(force=True)
        disk_revision = staged.fingerprint
        if disk_revision is None or disk_revision != expected_revision:
            return ConfigSaveResult("user", "not_saved", "unchanged", error="conflict")
        merged = merge_project_root_grant(
            loaded.model_dump(), project=project, root=root
        )
        validate_root_source(merged, layer=staged)
        patched = staged.validate_output(merged)
    except Exception:
        return ConfigSaveResult("user", "not_saved", "unchanged", error="validation")

    persistence: Literal["saved", "durability_uncertain"] = "saved"
    # Once admitted, the writer must finish before the mutation lock is
    # released. Cancellation prevents publication, not an already-running
    # filesystem replacement.
    writer = asyncio.create_task(
        layer.save_checked(patched, expected_revision=disk_revision)
    )
    cancelled = False
    try:
        while True:
            try:
                revision = await asyncio.shield(writer)
                break
            except asyncio.CancelledError:
                cancelled = True
                if writer.cancelled():
                    raise
    except ConcurrencyConflictError:
        return ConfigSaveResult("user", "not_saved", "unchanged", error="conflict")
    except ConfigDurabilityError as exc:
        revision = exc.revision
        persistence = "durability_uncertain"
    except Exception:
        return ConfigSaveResult("user", "not_saved", "unchanged", error="write")

    if cancelled:
        return ConfigSaveResult("user", persistence, "unchanged", revision, "cancelled")
    return ConfigSaveResult("user", persistence, "unchanged", revision)


async def read_saved_roots[S: ConfigSchema](
    orchestrator: ConfigOrchestrator[S], *, project: Path
) -> SavedRootsRead:
    """Read one project's saved roots and the user-layer revision from disk.

    Read-only: no orchestrator or layer lock is taken or held, nothing is
    published, and the live layer cache is never touched. The user layer is
    force-loaded through a private copy, so the result reflects disk rather
    than the in-memory view, and the reported revision is the freshly loaded
    layer fingerprint. A relative *project* is a caller error and raises
    ``ValueError``, mirroring the persistence merge.
    """
    project_path = Path(project).expanduser()
    if not project_path.is_absolute():
        raise ValueError(f"Invalid {ROOTS_FIELD} project key (source: user)")
    key = project_path.resolve()
    try:
        layer = select_actual_user_layer(orchestrator.layers)
    except ValueError:
        return SavedRootsRead(unavailable="no_user_source")
    try:
        staged = copy.deepcopy(layer)
        loaded = await staged.load(force=True)
        revision = staged.fingerprint
        if revision is None:
            return SavedRootsRead(unavailable="read")
        authorities = validate_root_source(loaded.model_dump(), layer=staged)
    except Exception:
        return SavedRootsRead(unavailable="read")
    roots = next((item.roots for item in authorities if item.project == key), ())
    return SavedRootsRead(
        roots=tuple(str(root) for root in roots), user_revision=revision
    )
