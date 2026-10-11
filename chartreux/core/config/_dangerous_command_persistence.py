"""Purpose-built user-layer grant persistence: the only sanctioned grants writer and saved-grants reader."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from chartreux.core.config._dangerous_command_authority import (
    GRANTS_FIELD,
    validate_grant_source,
)
from chartreux.core.config._root_persistence import select_actual_user_layer
from chartreux.core.config.models import normalize_dangerous_command_grants
from chartreux.core.config.types import (
    ConcurrencyConflictError,
    ConfigDurabilityError,
    ConfigSaveResult,
)
from chartreux.core.dangerous_commands import OPERATION_REGISTRY

if TYPE_CHECKING:
    from chartreux.core.config.orchestrator import ConfigOrchestrator
    from chartreux.core.config.schema import ConfigSchema


@dataclass(frozen=True, slots=True)
class SavedGrantsRead:
    """Read-only saved-grants projection for one project (operations and revision only).

    ``operations`` is empty when the project has no saved grants. ``unavailable``
    marks a read that could not be sourced from disk: no actual user layer is
    installed (``"no_user_source"``), or the backing store could not be read or
    its grants interpreted (``"read"``). A missing user file is not a failure:
    it reads as no saved grants with the missing-store revision.
    """

    operations: tuple[str, ...] = ()
    user_revision: str | None = None
    unavailable: Literal["no_user_source", "read"] | None = None


class _GrantEdit(Protocol):
    def __call__(
        self, raw: Mapping[str, Any], *, project: Path, operation_id: str
    ) -> dict[str, Any]: ...


def _require_persistable_operation(operation_id: str) -> None:
    operation = OPERATION_REGISTRY.get(operation_id)
    if operation is None or not operation.persistable:
        raise ValueError(f"Invalid {GRANTS_FIELD} operation (source: user)")


def _canonical_saved_entries(saved: Any) -> dict[str, list[str]]:
    if not isinstance(saved, dict) or not all(
        isinstance(key, str)
        and isinstance(operations, list)
        and all(isinstance(item, str) for item in operations)
        for key, operations in saved.items()
    ):
        raise ValueError(f"Invalid {GRANTS_FIELD} definition (source: user)")
    return dict(saved)


def _canonical_project_key(project: Path) -> str:
    project_path = Path(project).expanduser()
    if not project_path.is_absolute():
        raise ValueError(f"Invalid {GRANTS_FIELD} project key (source: user)")
    return str(project_path.resolve())


def merge_project_dangerous_command_grant(
    raw: Mapping[str, Any], *, project: Path, operation_id: str
) -> dict[str, Any]:
    """Merge one approved grant into one project's entry of a disk document.

    Only the target project's saved entry changes; every other entry and all
    unrelated settings are preserved from *raw*. Canonicalization follows
    ``normalize_dangerous_command_grants``, so a duplicate canonical grant merges
    as a no-op and relative or malformed definitions raise ``ValueError``.
    """
    _require_persistable_operation(operation_id)
    merged = dict(raw)
    saved = merged.get(GRANTS_FIELD)
    entries = {} if saved is None else _canonical_saved_entries(saved)
    canonical = normalize_dangerous_command_grants(entries)
    key = _canonical_project_key(project)
    canonical[key] = [*canonical.get(key, []), operation_id]
    merged[GRANTS_FIELD] = normalize_dangerous_command_grants(canonical)
    return merged


def remove_project_dangerous_command_grant(
    raw: Mapping[str, Any], *, project: Path, operation_id: str
) -> dict[str, Any]:
    """Remove one saved grant from one project's entry of a disk document.

    The project's entry is dropped once its last operation is removed, and the
    field itself is dropped when no project entries remain; every other entry
    and all unrelated settings are preserved from *raw*. Removing an absent
    grant leaves the document's grant state unchanged.
    """
    _require_persistable_operation(operation_id)
    merged = dict(raw)
    saved = merged.get(GRANTS_FIELD)
    if saved is None:
        return merged
    canonical = normalize_dangerous_command_grants(_canonical_saved_entries(saved))
    key = _canonical_project_key(project)
    operations = [item for item in canonical.get(key, []) if item != operation_id]
    if operations:
        canonical[key] = operations
    else:
        canonical.pop(key, None)
    if canonical:
        merged[GRANTS_FIELD] = canonical
    else:
        merged.pop(GRANTS_FIELD, None)
    return merged


async def _persist_grant_edit[S: ConfigSchema](
    orchestrator: ConfigOrchestrator[S],
    *,
    project: Path,
    operation_id: str,
    expected_revision: str,
    edit: _GrantEdit,
) -> ConfigSaveResult:
    """Apply one revision-checked grant edit without publishing runtime state.

    The caller must hold the orchestrator mutation lock and have admitted a
    persisting (root) orchestrator. The user layer is force-loaded through a
    private copy, so the edit reads disk rather than the in-memory view and
    the live layer cache is never touched. The write is an optimistic
    ``save_checked`` replacement; cancellation and durability uncertainty are
    reported in the result. Nothing is published: live caches, the accepted
    token, and the merged config stay unchanged (application is "unchanged"),
    and the caller applies any session grant itself.
    """
    try:
        layer = select_actual_user_layer(orchestrator.layers)
        staged = copy.deepcopy(layer)
        loaded = await staged.load(force=True)
        disk_revision = staged.fingerprint
        if disk_revision is None or disk_revision != expected_revision:
            return ConfigSaveResult("user", "not_saved", "unchanged", error="conflict")
        merged = edit(loaded.model_dump(), project=project, operation_id=operation_id)
        validate_grant_source(merged, layer=staged)
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


async def persist_dangerous_command_grant[S: ConfigSchema](
    orchestrator: ConfigOrchestrator[S],
    *,
    project: Path,
    operation_id: str,
    expected_revision: str,
) -> ConfigSaveResult:
    """Save one approved grant without publishing runtime state."""
    return await _persist_grant_edit(
        orchestrator,
        project=project,
        operation_id=operation_id,
        expected_revision=expected_revision,
        edit=merge_project_dangerous_command_grant,
    )


async def remove_dangerous_command_grant[S: ConfigSchema](
    orchestrator: ConfigOrchestrator[S],
    *,
    project: Path,
    operation_id: str,
    expected_revision: str,
) -> ConfigSaveResult:
    """Remove one saved grant without publishing runtime state."""
    return await _persist_grant_edit(
        orchestrator,
        project=project,
        operation_id=operation_id,
        expected_revision=expected_revision,
        edit=remove_project_dangerous_command_grant,
    )


async def read_saved_dangerous_command_grants[S: ConfigSchema](
    orchestrator: ConfigOrchestrator[S], *, project: Path
) -> SavedGrantsRead:
    """Read one project's saved grants and the user-layer revision from disk.

    Read-only: no orchestrator or layer lock is taken or held, nothing is
    published, and the live layer cache is never touched. The user layer is
    force-loaded through a private copy, so the result reflects disk rather
    than the in-memory view, and the reported revision is the freshly loaded
    layer fingerprint. A relative *project* is a caller error and raises
    ``ValueError``, mirroring the persistence edits.
    """
    key = _canonical_project_key(project)
    try:
        layer = select_actual_user_layer(orchestrator.layers)
    except ValueError:
        return SavedGrantsRead(unavailable="no_user_source")
    try:
        staged = copy.deepcopy(layer)
        loaded = await staged.load(force=True)
        revision = staged.fingerprint
        if revision is None:
            return SavedGrantsRead(unavailable="read")
        authorities = validate_grant_source(loaded.model_dump(), layer=staged)
    except Exception:
        return SavedGrantsRead(unavailable="read")
    operations = next(
        (item.operations for item in authorities if str(item.project) == key), ()
    )
    return SavedGrantsRead(operations=operations, user_revision=revision)
