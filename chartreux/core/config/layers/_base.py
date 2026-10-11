from __future__ import annotations

from abc import abstractmethod
import asyncio
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import tempfile
import time
import tomllib

import tomli_w

from chartreux.core.config.fingerprint import (
    capture_stable_file,
    create_file_fingerprint,
)
from chartreux.core.config.layer import ConfigLayer, ConfigStorageError, RawConfig
from chartreux.core.config.models import (
    ModelConfig,
    normalize_model_configs,
    serialize_model_configs,
)
from chartreux.core.config.types import (
    EMPTY_CONFIG_SNAPSHOT,
    MISSING_BACKING_STORE_DATA_FINGERPRINT,
    ConcurrencyConflictError,
    ConfigDurabilityError,
    LayerConfigSnapshot,
)

# Bounded wait for the cross-process write lock: a wedged holder delays a
# save instead of blocking it forever.
_WRITE_LOCK_ACQUIRE_TIMEOUT = 5.0
# Interval between non-blocking lock acquisition attempts.
_WRITE_LOCK_RETRY_INTERVAL = 0.01


class BaseTomlConfigLayer(ConfigLayer[RawConfig]):
    """Shared read/write logic for TOML file-backed config layers.

    Subclasses only resolve ``_target_path``; this base reads the file into a
    snapshot and persists patches atomically.
    """

    @property
    @abstractmethod
    def _target_path(self) -> Path:
        """The TOML file this layer reads from and writes to."""
        ...

    @property
    def source_locator(self) -> str:
        return str(self._target_path.absolute())

    async def _build_config_snapshot(self) -> LayerConfigSnapshot:
        try:
            return await asyncio.to_thread(_read_toml_snapshot, self._target_path)
        except OSError as e:
            raise ConfigStorageError(self.name, self._target_path, "read") from e

    async def save_checked(
        self, next_config: RawConfig, *, expected_revision: str
    ) -> str:
        """Write a prepared source without publishing its cache.

        Callers own scope/effective validation and runtime admission. The
        revision check, temp write, and replacement all run under the backing
        store's cross-process write lock, and the revision is re-read under
        the lock immediately before the replace: cooperating Chartreux
        writers are fully serialized, and an external writer that raced the
        temp write is reported as a conflict instead of being overwritten.

        This remains optimistic conflict detection, not a filesystem
        compare-and-swap: a non-cooperating writer that replaces the store
        between the final under-lock revision read and the rename is still
        overwritten, and the stat-based fingerprint (``dev:ino:mtime_ns:size``)
        cannot observe a same-size in-place write within one mtime tick.
        """
        async with self._lock:
            if not await self._check_trust():
                raise ValueError("Cannot save an untrusted source")
            return await asyncio.to_thread(
                _write_toml_snapshot,
                self._target_path,
                next_config,
                expected_revision=expected_revision,
            )

    def _stage_saved_data(self, data: RawConfig, revision: str) -> None:
        """Adopt a replaced source into a staged layer, not the live cache."""
        self._stage_loaded_data(data, revision)

    async def _save_to_store(self, next_config: RawConfig) -> str:
        try:
            return await asyncio.to_thread(
                _write_toml_snapshot, self._target_path, next_config
            )
        except OSError as e:
            raise ConfigStorageError(self.name, self._target_path, "write") from e


def _read_toml_snapshot(path: Path) -> LayerConfigSnapshot:
    if not path.exists():
        return EMPTY_CONFIG_SNAPSHOT

    with capture_stable_file(path) as (file, fingerprint):
        data = tomllib.load(file)

    return LayerConfigSnapshot(
        data=_internal_toml_document(data), fingerprint=fingerprint
    )


def _write_lock_path(path: Path) -> Path:
    """Dot-prefixed sibling lock file, matching the catalog store convention."""
    return path.with_name(f".{path.name}.lock")


@contextmanager
def _exclusive_write_lock(path: Path) -> Iterator[None]:
    """Hold the exclusive cross-process write lock for one backing store.

    The lock is an ``flock`` on a dot-prefixed sibling that persists by
    design: unlinking a flock target would let a later writer lock a fresh
    inode while an earlier holder still owns the unlinked one. Acquisition
    retries non-blocking within a bounded budget, so a wedged holder cannot
    block a save permanently. A read-only existing lock file still locks: the
    append open falls back to a read-only descriptor, ``flock(LOCK_EX)`` works
    on read-only handles on Linux, and cross-user serialization relies on
    both writers locking the same inode.
    """
    lock_path = _write_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = lock_path.open("a")
    except PermissionError:
        # An existing lock file need not be writable (another user created it,
        # or a sync tool made it read-only); only the lock matters.
        stream = lock_path.open("rb")
    with stream:
        deadline = time.monotonic() + _WRITE_LOCK_ACQUIRE_TIMEOUT
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for the write lock of {path}"
                    ) from None
                time.sleep(_WRITE_LOCK_RETRY_INTERVAL)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _require_revision(path: Path, expected_revision: str) -> None:
    """Reject a backing store that no longer carries *expected_revision*."""
    try:
        with path.open("rb") as current:
            actual = create_file_fingerprint(current)
    except FileNotFoundError:
        actual = MISSING_BACKING_STORE_DATA_FINGERPRINT
    if actual != expected_revision:
        raise ConcurrencyConflictError(expected_revision, actual)


def _write_toml_snapshot(
    path: Path, next_config: RawConfig, *, expected_revision: str | None = None
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)

    with _exclusive_write_lock(path):
        if expected_revision is not None:
            _require_revision(path, expected_revision)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tomli_w.dump(_canonical_toml_document(next_config), tmp_file)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                fingerprint = create_file_fingerprint(tmp_file)

            if expected_revision is not None:
                # Re-read under the lock: an external writer that replaced the
                # store during the temp write must be reported, not overwritten.
                _require_revision(path, expected_revision)
            tmp_path.replace(path)
            tmp_path = None
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError as exc:
                raise ConfigDurabilityError(fingerprint) from exc
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    return fingerprint


def _canonical_toml_document(config: RawConfig) -> dict[str, object]:
    """Persist model maps as legacy [[models]] arrays for extension compatibility."""
    data = config.model_dump()
    models = data.get("models")
    if _is_model_config_mapping(models):
        data["models"] = serialize_model_configs(models)
    return data


def _internal_toml_document(data: dict[str, object]) -> dict[str, object]:
    """Load TOML model arrays as alias maps so patches can address models by key."""
    models = data.get("models")
    if models and (
        _is_model_config_sequence(models) or _is_model_config_mapping(models)
    ):
        try:
            normalized = normalize_model_configs(models)
        except (ValueError, TypeError):
            # Preserve malformed source data for schema-aware, redacted validation.
            # This generic storage adapter cannot grant catalog authority.
            return data
        data = dict(data)
        data["models"] = normalized
    return data


def _is_model_config_sequence(value: object) -> bool:
    if not isinstance(value, list):
        return False
    return all(
        isinstance(model, Mapping) or isinstance(model, ModelConfig) for model in value
    )


def _is_model_config_mapping(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(
        isinstance(model, Mapping) or isinstance(model, ModelConfig)
        for model in value.values()
    )
