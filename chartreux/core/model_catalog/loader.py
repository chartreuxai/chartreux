"""Load the model catalog from shipped defaults and one user overlay only."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import threading
import tomllib
from typing import Any

from pydantic import ValidationError

from chartreux.core.dispatch import (
    DEFAULT_DISPATCH_MODE,
    SHIPPED_PRESETS,
    STANDALONE_PRESET,
    DispatchMode,
    DispatchPolicy,
)
from chartreux.core.model_catalog.contracts import (
    CatalogChanges,
    CatalogValidationError,
    CatalogWriteResult,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.schema import (
    BaseModelDefinition,
    DeploymentDefinition,
    ModelCatalog,
    ProviderDefinition,
    RoleDefinition,
    valid_provider_name,
)
from chartreux.observability.logging import logger
from chartreux.utils.paths import get_chartreux_home


class CatalogLoadError(ValueError):
    """An actionable error while parsing or validating ``models.toml``."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(f"Invalid model catalog {path}: {message}")
        self.path = path


@dataclass(frozen=True)
class CatalogSnapshot:
    """An immutable validated catalog and a content-derived revision identifier."""

    catalog: ModelCatalog
    revision: str
    overlaid_providers: frozenset[str] = frozenset()
    dispatch: DispatchPolicy = SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE]
    dispatch_diagnostics: tuple[str, ...] = ()


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a TOML table")
    return value


def _check_keys(raw: Mapping[str, Any], fields: set[str], location: str) -> None:
    if location.startswith("roles.") and "models" in raw:
        raise ValueError(
            f"{location}.models is obsolete; set one {location}.model and "
            f"{location}.thinking for this preset"
        )
    unknown = set(raw) - fields
    if unknown:
        migration_hints = {
            "tags": "replace [tags] with [roles]",
            "aliases": "remove aliases; use canonical model names",
        }
        hints = [
            migration_hints[field] for field in unknown if field in migration_hints
        ]
        message = f"{location} has unknown fields: {sorted(unknown)!r}"
        if hints:
            message = f"{message}; migration required: {'; '.join(hints)}"
        raise ValueError(message)


def _merge_deployments(
    base: Sequence[DeploymentDefinition], patch: Any, base_name: str
) -> list[dict[str, Any]]:
    if not isinstance(patch, list) or not patch:
        raise ValueError(f"models.{base_name}.deployments must be a non-empty array")
    known = {
        deployment.provider: deployment.model_dump(mode="python") for deployment in base
    }
    fields = set(DeploymentDefinition.model_fields)
    merged: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(patch):
        entry = _require_mapping(raw_entry, f"models.{base_name}.deployments[{index}]")
        _check_keys(entry, fields, f"models.{base_name}.deployments[{index}]")
        provider = entry.get("provider")
        if not isinstance(provider, str):
            raise ValueError(
                f"models.{base_name}.deployments[{index}].provider is required"
            )
        merged.append({**known.get(provider, {}), **entry})
    return merged


def merge_catalog_overlay(
    shipped: ModelCatalog, overlay: Mapping[str, Any]
) -> ModelCatalog:
    """Apply a sparse overlay without giving ordinary config layers authority.

    The optional ``dispatch`` table is accepted here but not merged: dispatch
    resolution is owned by :func:`resolve_dispatch_overlay`, which rejects an
    invalid table atomically instead of failing the whole catalog.
    """
    _check_keys(overlay, {"providers", "models", "roles", "dispatch"}, "catalog")
    result = shipped.model_dump(mode="python")

    if "providers" in overlay:
        providers = _require_mapping(overlay["providers"], "providers")
        for provider_id, raw_patch in providers.items():
            patch = _require_mapping(raw_patch, f"providers.{provider_id}")
            _check_keys(
                patch, set(ProviderDefinition.model_fields), f"providers.{provider_id}"
            )
            current = result["providers"].get(provider_id, {})
            result["providers"][provider_id] = {**current, **patch}

    if "models" in overlay:
        models = _require_mapping(overlay["models"], "models")
        for base_name, raw_patch in models.items():
            patch = _require_mapping(raw_patch, f"models.{base_name}")
            _check_keys(
                patch, set(BaseModelDefinition.model_fields), f"models.{base_name}"
            )
            current = result["models"].get(base_name, {})
            merged = {**current, **patch}
            if "deployments" in patch:
                prior = shipped.models.get(base_name)
                merged["deployments"] = _merge_deployments(
                    prior.deployments if prior is not None else [],
                    patch["deployments"],
                    base_name,
                )
            result["models"][base_name] = merged

    if "roles" in overlay:
        roles = _require_mapping(overlay["roles"], "roles")
        for role_name, raw_patch in roles.items():
            patch = _require_mapping(raw_patch, f"roles.{role_name}")
            _check_keys(patch, set(RoleDefinition.model_fields), f"roles.{role_name}")
            current = result["roles"].get(role_name, {})
            result["roles"][role_name] = {**current, **patch}

    return ModelCatalog.model_validate(result)


_DISPATCH_TABLE_FIELDS = frozenset(DispatchPolicy.model_fields)
_DISPATCH_ENTRY_MERGE_FIELDS = frozenset({"vocabulary", "slots"})


def _dispatch_error_summary(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for error in exc.errors(
            include_input=False, include_context=False, include_url=False
        ):
            location = ".".join(str(item) for item in error["loc"])
            parts.append(f"{location}: {error['msg']}" if location else error["msg"])
        return "; ".join(parts) or "invalid dispatch table"
    return str(exc)


def _selected_dispatch_preset(mode: Any) -> DispatchPolicy:
    if isinstance(mode, DispatchMode):
        selected = mode
    else:
        try:
            selected = DispatchMode(str(mode))
        except ValueError:
            raise ValueError(
                "S9 dispatch.mode must be a shipped mode name: "
                f"{sorted(item.value for item in DispatchMode)}"
            ) from None
    return SHIPPED_PRESETS[selected]


def _merge_dispatch_table(
    base: dict[str, Any], patch: Mapping[str, Any]
) -> dict[str, Any]:
    """Sparse-merge one ``[dispatch]`` table over a shipped preset dump.

    Absent fields inherit the preset. Entry tables (``vocabulary``, ``slots``)
    merge per entry; list-valued fields such as slot ``purposes`` replace
    wholesale rather than merging within the list.
    """
    merged = dict(base)
    for key, value in patch.items():
        if key in _DISPATCH_ENTRY_MERGE_FIELDS:
            if not isinstance(value, Mapping):
                raise ValueError(f"dispatch.{key} must be a TOML table")
            table = dict(merged[key])
            for name, entry in value.items():
                if not isinstance(entry, Mapping):
                    raise ValueError(f"dispatch.{key}.{name} must be a TOML table")
                current = table.get(name)
                if not isinstance(current, Mapping):
                    current = {}
                table[name] = {**current, **entry}
            merged[key] = table
        else:
            merged[key] = value
    return merged


def merge_dispatch_overlay(patch: Mapping[str, Any]) -> DispatchPolicy:
    """Resolve a sparse ``[dispatch]`` overlay table against its shipped preset.

    ``mode`` selects the shipped preset to patch (absent means the shipped
    default); every other absent field inherits that preset.
    """
    if not isinstance(patch, Mapping):
        raise ValueError("dispatch must be a TOML table")
    unknown = set(patch) - _DISPATCH_TABLE_FIELDS
    if unknown:
        raise ValueError(f"dispatch has unknown fields: {sorted(unknown)!r}")
    base = _selected_dispatch_preset(patch.get("mode", DEFAULT_DISPATCH_MODE))
    merged = _merge_dispatch_table(base.model_dump(mode="json"), patch)
    return DispatchPolicy.model_validate(merged)


def resolve_dispatch_overlay(
    overlay: Mapping[str, Any],
    *,
    source: str = "user catalog overlay",
    catalog: ModelCatalog | None = None,
) -> tuple[DispatchPolicy, tuple[str, ...]]:
    """Resolve the effective dispatch policy carried by a raw catalog overlay.

    An invalid ``[dispatch]`` table is rejected atomically: the shipped
    standalone preset is selected with a visible diagnostic, and the invalid
    table is never rewritten. Valid provider, model, and role entries in the
    same file are unaffected.
    """
    from chartreux.core.dispatch.lint import lint_catalog, reject_errors

    # Provider/model validation stays outside dispatch recovery.
    merged_catalog = (
        catalog
        if catalog is not None
        else merge_catalog_overlay(SHIPPED_CATALOG, overlay)
    )
    raw = overlay.get("dispatch")
    try:
        policy = (
            SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE]
            if raw is None
            else merge_dispatch_overlay(raw)
        )
        reject_errors(lint_catalog(policy, merged_catalog))
        return policy, ()
    except (ValidationError, ValueError, TypeError) as exc:
        return dispatch_fallback(_dispatch_error_summary(exc), source=source)


def dispatch_fallback(
    reason: str, *, source: str
) -> tuple[DispatchPolicy, tuple[str, ...]]:
    """Reject a dispatch candidate intact, without repairing catalog bindings."""
    repair = (
        "repair the catalog role binding named by S1 (not only [dispatch])"
        if "S1 " in reason
        else "repair [dispatch] or the rejected profile named below"
    )
    diagnostic = (
        f"S10 Invalid dispatch policy in {source}: using the shipped standalone "
        "preset. The candidate dispatch config was bypassed and is left unchanged; "
        f"{repair} in the user catalog overlay. {reason}"
    )
    return STANDALONE_PRESET, (diagnostic,)


def fallback_dispatch_snapshot(
    snapshot: CatalogSnapshot, reason: str
) -> CatalogSnapshot:
    """Apply activation recovery while retaining all validated catalog entries."""
    policy, diagnostics = dispatch_fallback(reason, source="registry activation")
    # Activation does not reload or edit the source catalog. Preserve its revision
    # so committed model/deployment identities and child isolation stay stable.
    return replace(
        snapshot,
        dispatch=policy,
        dispatch_diagnostics=(*snapshot.dispatch_diagnostics, *diagnostics),
    )


def _snapshot(
    catalog: ModelCatalog,
    overlaid_providers: frozenset[str] = frozenset(),
    dispatch: DispatchPolicy | None = None,
    dispatch_diagnostics: tuple[str, ...] = (),
) -> CatalogSnapshot:
    policy = (
        dispatch if dispatch is not None else SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE]
    )
    payload = {
        "catalog": catalog.model_dump(mode="json"),
        "dispatch": policy.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return CatalogSnapshot(
        catalog,
        sha256(encoded).hexdigest(),
        overlaid_providers,
        policy,
        dispatch_diagnostics,
    )


def _snapshot_for_overlay(
    overlay: Mapping[str, Any], *, source: str
) -> CatalogSnapshot:
    """Validate one raw overlay end to end: catalog fatally, dispatch with fallback."""
    catalog = merge_catalog_overlay(SHIPPED_CATALOG, overlay)
    dispatch, diagnostics = resolve_dispatch_overlay(
        overlay, source=source, catalog=catalog
    )
    snapshot = _snapshot(
        catalog,
        frozenset(valid_provider_name(name) for name in overlay.get("providers", {})),
        dispatch,
        diagnostics,
    )
    if diagnostics:
        # Recovery changes the served policy, not the source identity. Include the
        # rejected candidate so distinct invalid tables cannot evade stale saves.
        candidate = overlay.get(
            "dispatch", SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE].model_dump(mode="json")
        )
        encoded = json.dumps(
            {"catalog": catalog.model_dump(mode="json"), "dispatch": candidate},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        snapshot = replace(snapshot, revision=sha256(encoded).hexdigest())
    return snapshot


def load_catalog(path: Path | None = None) -> CatalogSnapshot:
    """Load shipped definitions plus ``CHARTREUX_HOME/models.toml`` if it exists."""
    catalog_path = path or get_chartreux_home() / "models.toml"
    try:
        with catalog_path.open("rb") as stream:
            overlay = tomllib.load(stream)
    except FileNotFoundError:
        return _snapshot(SHIPPED_CATALOG)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CatalogLoadError(catalog_path, str(exc)) from exc

    try:
        snapshot = _snapshot_for_overlay(overlay, source=str(catalog_path))
    except (ValidationError, ValueError) as exc:
        raise CatalogLoadError(catalog_path, str(exc)) from exc
    for diagnostic in snapshot.dispatch_diagnostics:
        logger.warning("%s", diagnostic)
    return snapshot


class CatalogStore:
    """The only mutable writer for the user catalog overlay."""

    _write_lock = threading.Lock()

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_chartreux_home() / "models.toml"

    def _lock_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.lock")

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        """Serialize overlay read-modify-write cycles across application processes."""
        lock_path = self._lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _read_overlay(self) -> dict[str, Any]:
        try:
            with self.path.open("rb") as stream:
                overlay = tomllib.load(stream)
        except FileNotFoundError:
            return {}
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise CatalogLoadError(self.path, str(exc)) from exc
        if not isinstance(overlay, dict):
            raise CatalogLoadError(self.path, "catalog root must be a TOML table")
        return overlay

    def apply_changes(
        self, changes: CatalogChanges
    ) -> CatalogWriteResult | CatalogValidationError:
        """Apply one catalog-wide batch without widening the overlay's authority.

        Model deployment patches are additions or edits keyed by provider.  Because
        the overlay loader treats deployment arrays as replacements, a changed array
        is rebuilt from the effective order while retaining every raw overlay entry.
        A supplied dispatch table atomically replaces only the overlay's ``[dispatch]``
        section and is validated before anything is written; an invalid batch leaves
        the file unchanged, including an invalid pre-existing dispatch table.
        """
        with self._write_lock, self._file_lock():
            overlay = self._read_overlay()
            try:
                current = _snapshot_for_overlay(overlay, source=str(self.path))
                if (
                    changes.expected_revision is not None
                    and current.revision != changes.expected_revision
                ):
                    return CatalogValidationError(
                        "Catalog changed since this draft was opened; reload before saving."
                    )
                if changes.dispatch is not None:
                    # Validate the replacement dispatch table before touching disk.
                    merge_dispatch_overlay(changes.dispatch)
                candidate = _apply_catalog_changes(overlay, current.catalog, changes)
                snapshot = _snapshot_for_overlay(candidate, source=str(self.path))
                if changes.dispatch is not None:
                    from chartreux.core.dispatch.lint import lint_catalog, reject_errors

                    reject_errors(
                        lint_catalog(
                            merge_dispatch_overlay(changes.dispatch), snapshot.catalog
                        )
                    )
            except (ValidationError, ValueError, TypeError) as exc:
                return CatalogValidationError(str(exc))

            if candidate == overlay:
                return CatalogWriteResult(snapshot, changed=False)

            try:
                self._atomic_write(candidate)
            except OSError as exc:
                raise CatalogLoadError(self.path, str(exc)) from exc
            return CatalogWriteResult(snapshot, changed=True)

    def upsert_provider(self, provider: dict[str, Any], provider_id: str) -> None:
        """Compatibility wrapper for callers that only persist a provider."""
        result = self.apply_changes(
            CatalogChanges(provider_id, {}, providers={provider_id: provider})
        )
        if isinstance(result, CatalogValidationError):
            raise CatalogLoadError(self.path, result.message)

    def _atomic_write(self, overlay: Mapping[str, Any]) -> None:
        """Durably replace the overlay using a unique temporary sibling."""
        import tomli_w

        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                tomli_w.dump(dict(overlay), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)


def _table(overlay: dict[str, Any], name: str) -> dict[str, Any]:
    value = overlay.setdefault(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _patch_table(
    table: dict[str, Any], key: str, patch: Mapping[str, object], location: str
) -> None:
    existing = table.get(key, {})
    if not isinstance(existing, dict):
        raise ValueError(f"{location}.{key} must be a TOML table")
    for field, value in patch.items():
        # An explicitly supplied default is an intentional future-proof override.
        if field not in existing or existing[field] != value:
            existing[field] = value
    if existing:
        table[key] = existing


def _apply_catalog_changes(
    overlay: dict[str, Any], current: ModelCatalog, changes: CatalogChanges
) -> dict[str, Any]:
    """Return a candidate raw overlay after applying a provider-management batch."""
    candidate = _copy_catalog_value(overlay)
    providers = _table(candidate, "providers")
    for provider_id, patch in changes.provider_patches.items():
        _patch_table(providers, provider_id, patch, "providers")

    if changes.models:
        models = _table(candidate, "models")
        for base_name, raw_patch in changes.models.items():
            if not isinstance(raw_patch, Mapping):
                raise ValueError(f"models.{base_name} must be a TOML table")
            patch = dict(raw_patch)
            deployments = patch.pop("deployments", None)
            _patch_table(models, base_name, patch, "models")
            if deployments is not None:
                _patch_deployments(
                    models, base_name, deployments, current.models.get(base_name)
                )

    if changes.roles is not None:
        roles = _table(candidate, "roles")
        for role_name, raw_patch in changes.roles.items():
            if not isinstance(raw_patch, Mapping):
                raise ValueError(f"roles.{role_name} must be a TOML table")
            _patch_table(roles, role_name, raw_patch, "roles")

    if changes.dispatch is not None:
        # The supplied table atomically replaces only the dispatch section;
        # every other section of the overlay is preserved verbatim.
        if not isinstance(changes.dispatch, Mapping):
            raise ValueError("dispatch must be a TOML table")
        candidate["dispatch"] = _copy_catalog_value(dict(changes.dispatch))

    for name in ("providers", "models"):
        if not candidate.get(name) and name not in overlay:
            candidate.pop(name, None)
    return candidate


def _patch_deployments(  # noqa: PLR0912
    models: dict[str, Any],
    base_name: str,
    requested: object,
    current: BaseModelDefinition | None,
) -> None:
    if not isinstance(requested, Sequence) or isinstance(requested, (str, bytes)):
        raise ValueError(f"models.{base_name}.deployments must be an array")
    existing = models.get(base_name, {})
    if not isinstance(existing, dict):
        raise ValueError(f"models.{base_name} must be a TOML table")
    raw_deployments = existing.get("deployments", [])
    if not isinstance(raw_deployments, list):
        raise ValueError(f"models.{base_name}.deployments must be an array")
    raw_by_provider: dict[str, dict[str, Any]] = {}
    for entry in raw_deployments:
        if not isinstance(entry, dict) or not isinstance(entry.get("provider"), str):
            raise ValueError(
                f"models.{base_name}.deployments contains an invalid entry"
            )
        raw_by_provider[entry["provider"]] = entry

    requested_by_provider: dict[str, dict[str, Any]] = {}
    for entry in requested:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("provider"), str):
            raise ValueError(
                f"models.{base_name}.deployments entries require a provider"
            )
        provider = entry["provider"]
        if provider in requested_by_provider:
            raise ValueError(f"models.{base_name} has duplicate provider {provider!r}")
        requested_by_provider[provider] = dict(entry)

    effective = current.deployments if current is not None else ()
    effective_by_provider = {
        deployment.provider: deployment for deployment in effective
    }
    changed = False
    for provider, patch in requested_by_provider.items():
        raw = raw_by_provider.get(provider)
        deployment = effective_by_provider.get(provider)
        if deployment is None:
            if "name" not in patch:
                raise ValueError(
                    f"models.{base_name}.deployments for {provider!r} requires a name"
                )
            changed = True
        elif raw is not None:
            if any(
                field not in raw or raw[field] != value
                for field, value in patch.items()
            ):
                changed = True
        # provider is identity-only; other supplied fields are explicit overrides.
        elif patch.get("name") != deployment.name or set(patch) - {"provider", "name"}:
            changed = True

    if not changed:
        return

    rebuilt: list[dict[str, Any]] = []
    for deployment in effective:
        provider = deployment.provider
        raw = raw_by_provider.get(provider)
        patch = requested_by_provider.get(provider)
        if raw is not None:
            entry = dict(raw)
        else:
            # A stub inherits only when the shipped deployment has no user patch.
            entry = {"provider": provider}
        if patch is not None:
            entry.update(patch)
        rebuilt.append(entry)
    for provider, patch in requested_by_provider.items():
        if provider not in effective_by_provider:
            rebuilt.append(patch)
    existing["deployments"] = rebuilt
    models[base_name] = existing


def _copy_catalog_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_catalog_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_catalog_value(item) for item in value]
    return value
