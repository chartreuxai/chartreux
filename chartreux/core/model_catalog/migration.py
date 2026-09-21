"""One-shot import of legacy config.toml model definitions into models.toml."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import threading
import tomllib
from typing import Any

import tomli_w

from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import merge_catalog_overlay
from chartreux.core.model_catalog.matching import match_discovered_model
from chartreux.utils.paths import get_chartreux_home

_MIGRATION_HINT = (
    "Run `chartreux models migrate` to move catalog definitions to models.toml."
)
_MARKER_NAME = ".models-migration-recovery.toml"
_LOCK_NAME = ".models-migration.lock"
_BACKUP_SUFFIX = ".pre-model-catalog-migration.bak"
_LEGACY_PROVIDER_DEFAULTS = {"project_id": "", "region": ""}
_MIGRATION_LOCKS: dict[Path, Any] = {}
_MIGRATION_LOCKS_GUARD = threading.Lock()
_MIGRATION_LOCK_STATE = threading.local()


class MigrationError(RuntimeError):
    pass


class MigrationConflictError(MigrationError):
    pass


@dataclass(frozen=True)
class MigrationPlan:
    config_path: Path
    catalog_path: Path
    backup_path: Path
    catalog: dict[str, Any]
    cleaned_config: bytes
    original_config: bytes


def _provider_id(name: str) -> str:
    return name if "/" in name else f"{name}/default"


def _legacy_models(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # The former internal alias-map representation is accepted by the
        # dedicated parser without asking the new catalog schema to validate it.
        return [
            dict(value, alias=key) if isinstance(value, dict) else value
            for key, value in raw.items()
        ]
    raise MigrationError("Legacy [models] must be an array of tables.")


def _legacy_provider_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in raw.items() if key != "name"}
    for field, default in _LEGACY_PROVIDER_DEFAULTS.items():
        value = payload.pop(field, default)
        if value != default:
            raise MigrationConflictError(
                f"Legacy provider {raw['name']!r} sets unsupported {field!r} "
                f"to {value!r}; move it to a supported provider configuration manually."
            )
    return payload


def _migration_base(provider_id: str, wire_name: str) -> str | None:
    """Find one shipped identity for migration without widening discovery matching."""
    matches = match_discovered_model(SHIPPED_CATALOG, provider_id, wire_name)
    if matches.kind in {"existing", "base_exists_other_provider"}:
        return matches.existing_base

    candidates = {
        base
        for base, definition in SHIPPED_CATALOG.models.items()
        if wire_name in definition.aliases
        or any(deployment.name == wire_name for deployment in definition.deployments)
    }
    if len(candidates) == 1:
        return candidates.pop()
    if len(candidates) > 1:
        raise MigrationConflictError(
            f"Ambiguous shipped alias {wire_name!r}: {sorted(candidates)!r}; resolve manually."
        )
    return None


def _legacy_deployment(
    entry: dict[str, Any], *, base: str, reconciled: bool
) -> dict[str, Any]:
    provider_id = _provider_id(entry["provider"])
    definition = SHIPPED_CATALOG.models.get(base)
    shipped = (
        next(
            (
                deployment
                for deployment in definition.deployments
                if deployment.provider == provider_id
            ),
            None,
        )
        if definition is not None
        else None
    )
    full_definition = not reconciled or shipped is None
    deployment: dict[str, Any] = {"provider": provider_id}
    if full_definition or shipped is None or entry["name"] != shipped.name:
        deployment["name"] = entry["name"]

    for field, default in (
        ("supports_images", False),
        ("supported_thinking_levels", None),
        ("auto_compact_threshold", None),
    ):
        value = entry.get(field, default)
        if full_definition or value != getattr(shipped, field):
            if value is not None:
                deployment[field] = value

    prices = {
        "input": entry.get("input_price"),
        "output": entry.get("output_price"),
        "cached_input": entry.get("cached_input_price"),
    }
    if full_definition:
        prices = {key: value for key, value in prices.items() if value is not None}
    else:
        prices = {
            key: value
            for key, value in prices.items()
            if value is not None
            and (shipped is None or value != getattr(shipped.prices, key))
        }
    if prices:
        deployment["prices"] = prices
    return deployment


def _shipped_aliases(base: str, entries: list[dict[str, Any]]) -> tuple[str, ...]:
    aliases = set(SHIPPED_CATALOG.models[base].aliases)
    aliases.update(entry.get("alias", base) for entry in entries)
    aliases.discard(base)
    return tuple(sorted(aliases))


def _build_catalog(  # noqa: PLR0912, PLR0914, PLR0915
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    raw_providers = data.get("providers", [])
    if not isinstance(raw_providers, list):
        raise MigrationError("Legacy [providers] must be an array of tables.")
    providers: dict[str, dict[str, Any]] = {}
    for raw in raw_providers:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise MigrationError("Each legacy provider must have a name.")
        name = raw["name"]
        provider_id = _provider_id(name)
        payload = _legacy_provider_payload(raw)
        if provider_id in providers and providers[provider_id] != payload:
            raise MigrationConflictError(
                f"Provider {name!r} has conflicting definitions."
            )
        providers[provider_id] = payload

    grouped: dict[str, list[dict[str, Any]]] = {}
    aliases: dict[str, str] = {}
    reconciled: set[str] = set()
    for raw in _legacy_models(data.get("models", [])):
        if not isinstance(raw, dict):
            raise MigrationError("Each legacy model must be a table.")
        name = raw.get("name")
        provider = raw.get("provider")
        alias = raw.get("alias", name)
        if not all(
            isinstance(value, str) and value for value in (name, provider, alias)
        ):
            raise MigrationError(
                "Each legacy model must have name, provider, and alias."
            )
        assert isinstance(name, str)
        assert isinstance(provider, str)
        assert isinstance(alias, str)
        provider_id = _provider_id(provider)
        base = _migration_base(provider_id, name) or name
        if base in SHIPPED_CATALOG.models:
            reconciled.add(base)
        grouped.setdefault(base, []).append(raw)
        for legacy_name in (name, alias):
            previous = aliases.get(legacy_name)
            if previous is not None and previous != base:
                raise MigrationConflictError(
                    f"Alias {legacy_name!r} identifies both {previous!r} and {base!r}."
                )
            aliases[legacy_name] = base

    models: dict[str, Any] = {}
    for base, entries in grouped.items():
        semantic = {
            (entry.get("temperature", 0.2), entry.get("thinking", "off"))
            for entry in entries
        }
        if len(semantic) != 1:
            names = sorted(str(entry.get("alias", base)) for entry in entries)
            raise MigrationConflictError(
                f"Ambiguous legacy identity {base!r}: aliases {names!r} have different inference settings; resolve manually."
            )
        first = entries[0]
        deployment_by_provider: dict[str, dict[str, Any]] = {}
        for entry in entries:
            provider_id = _provider_id(entry["provider"])
            deployment = _legacy_deployment(
                entry, base=base, reconciled=base in reconciled
            )
            prior = deployment_by_provider.get(provider_id)
            if prior is not None and prior != deployment:
                raise MigrationConflictError(
                    f"Ambiguous legacy identity {base!r}: multiple deployments for {provider_id!r}; resolve manually."
                )
            deployment_by_provider[provider_id] = deployment
        if base in reconciled:
            models[base] = {
                "aliases": _shipped_aliases(base, entries),
                "thinking": first.get("thinking", "off"),
                "temperature": first.get("temperature", 0.2),
                "deployments": list(deployment_by_provider.values()),
            }
        else:
            model_aliases = tuple(
                sorted({entry.get("alias", base) for entry in entries} - {base})
            )
            models[base] = {
                "aliases": model_aliases,
                "thinking": first.get("thinking", "off"),
                "temperature": first.get("temperature", 0.2),
                "deployments": list(deployment_by_provider.values()),
            }
    catalog = {"providers": providers, "models": models}
    try:
        merge_catalog_overlay(SHIPPED_CATALOG, catalog)
    except ValueError as exc:
        raise MigrationError(f"Migrated catalog is invalid: {exc}") from exc
    return catalog, aliases


def _canonicalize_selections(config: dict[str, Any], aliases: dict[str, str]) -> bytes:
    """Remove catalog tables and canonicalize selections in parsed legacy TOML."""
    config.pop("providers", None)
    config.pop("models", None)

    for field in ("active_model", "compaction_model"):
        value = config.get(field)
        if isinstance(value, str):
            config[field] = aliases.get(value, value)

    overrides = config.get("thinking_overrides")
    if isinstance(overrides, dict):
        canonical_overrides: dict[str, Any] = {}
        for key, value in overrides.items():
            if not isinstance(key, str):
                raise MigrationError("Legacy thinking override keys must be strings.")
            canonical_key = aliases.get(key, key)
            if canonical_key in canonical_overrides:
                raise MigrationConflictError(
                    f"Ambiguous legacy identity {canonical_key!r}: multiple thinking overrides resolve to it; resolve manually."
                )
            canonical_overrides[canonical_key] = value
        config["thinking_overrides"] = canonical_overrides

    return tomli_w.dumps(config).encode()


def plan_migration(config_path: Path, catalog_path: Path) -> MigrationPlan:
    try:
        original = config_path.read_bytes()
    except FileNotFoundError as exc:
        raise MigrationError(f"No config.toml exists at {config_path}.") from exc
    try:
        parsed = tomllib.loads(original.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise MigrationError(f"Cannot parse legacy config.toml: {exc}") from exc
    if not {"providers", "models"}.intersection(parsed):
        raise MigrationError(f"No legacy catalog tables found. {_MIGRATION_HINT}")
    catalog, aliases = _build_catalog(parsed)
    cleaned = _canonicalize_selections(parsed, aliases)
    return MigrationPlan(
        config_path,
        catalog_path,
        config_path.with_name(config_path.name + _BACKUP_SUFFIX),
        catalog,
        cleaned,
        original,
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _marker_path(config_path: Path) -> Path:
    return config_path.parent / _MARKER_NAME


def _lock_path(config_path: Path) -> Path:
    return config_path.parent / _LOCK_NAME


@contextmanager
def _migration_lock(config_path: Path) -> Iterator[None]:
    """Acquire a reentrant, process-exclusive migration lock for this home."""
    lock_path = _lock_path(config_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _MIGRATION_LOCKS_GUARD:
        process_lock = _MIGRATION_LOCKS.setdefault(lock_path, threading.RLock())
    if not process_lock.acquire(blocking=False):
        raise MigrationConflictError("Migration already in progress.")

    depths = getattr(_MIGRATION_LOCK_STATE, "depths", None)
    if depths is None:
        depths = {}
        _MIGRATION_LOCK_STATE.depths = depths
    depth = depths.get(lock_path, 0)
    depths[lock_path] = depth + 1
    try:
        if depth:
            yield
            return
        with lock_path.open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise MigrationConflictError("Migration already in progress.") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
    finally:
        if depth:
            depths[lock_path] = depth
        else:
            del depths[lock_path]
        process_lock.release()


def _finish_recovery(config_path: Path, catalog_path: Path) -> bool:
    marker = _marker_path(config_path)
    if not marker.exists():
        return False
    try:
        recovery = tomllib.loads(marker.read_text("utf-8"))
        expected = Path(recovery["catalog_path"])
        expected_digest = recovery["catalog_digest"]
        expected_original_digest = recovery["original_config_digest"]
        expected_cleaned_digest = recovery["cleaned_config_digest"]
        cleaned = base64.b64decode(recovery["cleaned_config"], validate=True)
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise MigrationError(
            f"Invalid migration recovery marker {marker}: {exc}"
        ) from exc
    if expected != catalog_path or not catalog_path.exists():
        raise MigrationConflictError(
            "Migration recovery is incomplete; models.toml must be resolved manually."
        )
    try:
        actual_digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise MigrationConflictError(
            "Migration recovery is incomplete; models.toml must be resolved manually."
        ) from exc
    if not isinstance(expected_digest, str) or actual_digest != expected_digest:
        raise MigrationConflictError(
            "Migration recovery found a changed models.toml; resolve it manually."
        )
    try:
        tomllib.loads(cleaned.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        backup_path = config_path.with_name(config_path.name + _BACKUP_SUFFIX)
        raise MigrationError(
            f"Invalid recovered config in {marker}; inspect backup at {backup_path}: {exc}"
        ) from exc
    try:
        actual_config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise MigrationConflictError(
            "Migration recovery is incomplete; config.toml must be resolved manually."
        ) from exc
    if not all(
        isinstance(digest, str)
        for digest in (expected_original_digest, expected_cleaned_digest)
    ):
        raise MigrationError(
            f"Invalid migration recovery marker {marker}: invalid config digest"
        )
    if hashlib.sha256(cleaned).hexdigest() != expected_cleaned_digest:
        raise MigrationConflictError(
            "Migration recovery found a changed cleaned config payload; resolve it manually."
        )
    if actual_config_digest == expected_cleaned_digest:
        marker.unlink()
        _fsync_directory(config_path.parent)
        return True
    if actual_config_digest != expected_original_digest:
        raise MigrationConflictError(
            "Migration recovery found a changed config.toml; resolve it manually."
        )
    _atomic_write(config_path, cleaned)
    _fsync_directory(config_path.parent)
    marker.unlink()
    _fsync_directory(config_path.parent)
    return True


def _recovery_marker_data(plan: MigrationPlan) -> bytes:
    catalog_contents = tomli_w.dumps(plan.catalog).encode()
    return tomli_w.dumps({
        "catalog_path": str(plan.catalog_path),
        "catalog_digest": hashlib.sha256(catalog_contents).hexdigest(),
        "original_config_digest": hashlib.sha256(plan.original_config).hexdigest(),
        "cleaned_config_digest": hashlib.sha256(plan.cleaned_config).hexdigest(),
        "cleaned_config": base64.b64encode(plan.cleaned_config).decode(),
    }).encode()


def _catalog_matches_plan(plan: MigrationPlan) -> bool:
    try:
        existing = tomllib.loads(plan.catalog_path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    return existing == tomllib.loads(tomli_w.dumps(plan.catalog))


def _can_finish_catalog_write(plan: MigrationPlan) -> bool:
    try:
        backup = plan.backup_path.read_bytes()
    except OSError:
        return False
    return backup == plan.original_config and _catalog_matches_plan(plan)


def apply_migration(plan: MigrationPlan) -> str:
    with _migration_lock(plan.config_path):
        if _finish_recovery(plan.config_path, plan.catalog_path):
            return "Recovered interrupted migration; config.toml cleanup completed."

        # A preview plan can be stale by the time --apply runs; migrate the bytes
        # currently on disk so intervening user edits are never silently discarded.
        plan = plan_migration(plan.config_path, plan.catalog_path)
        marker = _marker_path(plan.config_path)
        if plan.catalog_path.exists():
            if not _can_finish_catalog_write(plan):
                raise MigrationConflictError(
                    f"{plan.catalog_path} already exists; resolve the catalog manually rather than overwriting it."
                )
            _atomic_write(marker, _recovery_marker_data(plan))
            _fsync_directory(plan.config_path.parent)
            _finish_recovery(plan.config_path, plan.catalog_path)
            return "Recovered interrupted migration; config.toml cleanup completed."

        _atomic_write(plan.backup_path, plan.original_config)
        _fsync_directory(plan.config_path.parent)
        _atomic_write(plan.catalog_path, tomli_w.dumps(plan.catalog).encode())
        _fsync_directory(plan.config_path.parent)
        _atomic_write(marker, _recovery_marker_data(plan))
        _fsync_directory(plan.config_path.parent)
        _atomic_write(plan.config_path, plan.cleaned_config)
        _fsync_directory(plan.config_path.parent)
        marker.unlink()
        _fsync_directory(plan.config_path.parent)
        return f"Migrated catalog to {plan.catalog_path}; backup: {plan.backup_path}"


def run_models_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="chartreux models")
    subcommands = parser.add_subparsers(dest="command", required=True)
    migrate = subcommands.add_parser("migrate")
    mode = migrate.add_mutually_exclusive_group()
    mode.add_argument(
        "--preview",
        action="store_true",
        help="Show changes without writing them (default).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Write models.toml and remove legacy catalog tables.",
    )
    args = parser.parse_args(argv)
    home = get_chartreux_home()
    config_path = home / "config.toml"
    catalog_path = home / "models.toml"
    try:
        if args.apply:
            with _migration_lock(config_path):
                if _finish_recovery(config_path, catalog_path):
                    print(
                        "Recovered interrupted migration; config.toml cleanup completed."
                    )
                    return
                plan = plan_migration(config_path, catalog_path)
                print(apply_migration(plan))
        else:
            plan = plan_migration(config_path, catalog_path)
            print(
                f"Preview: would write {catalog_path}, back up {config_path} to {plan.backup_path}, and remove legacy [providers]/[models] tables."
            )
    except MigrationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
