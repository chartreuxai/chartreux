"""Synchronous checked-write primitive for small file-backed stores.

Synchronous sibling of the async TOML-layer mechanics in ``_base.py`` (the
trust callers are sync CLI/env/doctor paths): the lock, revision check,
temp write, final under-lock re-check, atomic replace, and file+directory
fsync all run as ONE synchronous primitive here, reusing the same exclusive
cross-process ``flock`` on the dot-prefixed sibling lock file.

Invariants:

- Atomic replace: the new payload lands via a single ``os.replace`` of a
  fully written, fsynced temp file; readers never observe partial content
  and the store's inode always changes (never a truncate-and-write).
- Durability: the temp file is fsynced before the replace and the parent
  directory is fsynced after it.
- Serialization: every mutation runs under the exclusive lock, and the
  revision is re-read under the lock immediately before the replace, so
  cooperating writers are fully serialized and a stale writer that waited
  on a held lock is refused (conflict) instead of resurrecting old content.
- Explicit outcomes: every call reports one ``CheckedWriteOutcome``;
  there is no silent yes. A post-replace fsync failure is reported as
  ``DURABILITY_UNCERTAIN`` (the replace happened; only its durability is
  unconfirmed), never as "not saved".
- Malformed refusal: a store that fails ``parse`` is never rewritten, and
  the refusal is reported distinctly from a conflict.

Revision scheme: a content hash (sha256 of the exact bytes), NOT the
``dev:ino:mtime_ns:size`` stat fingerprint the async layers use. The stat
fingerprint cannot observe a same-size in-place write within one mtime
tick; the content hash detects every byte change, so that blind spot is
closed here and pinned by a test.

Honest limitation (not closable without a filesystem CAS): a
NON-cooperating writer that replaces the store between the final
under-lock revision read and the ``os.replace`` is still overwritten.
Cooperating writers take the lock and are fully serialized; an external
writer whose write lands any time before the final under-lock check is
detected and reported as a conflict (also pinned by a test).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

from chartreux.core.config.layers._base import _exclusive_write_lock
from chartreux.core.config.types import MISSING_BACKING_STORE_DATA_FINGERPRINT

__all__ = [
    "CheckedWriteOutcome",
    "CheckedWriteResult",
    "checked_write_store",
    "store_revision",
]


class CheckedWriteOutcome(StrEnum):
    """The explicit outcome of one checked write; never a silent yes."""

    SAVED = auto()
    NOT_SAVED = auto()
    DURABILITY_UNCERTAIN = auto()
    CONFLICT = auto()
    MALFORMED = auto()
    LOCKED = auto()


@dataclass(frozen=True, slots=True)
class CheckedWriteResult:
    """Outcome of one checked write.

    ``revision`` is the store's content revision as observed by the call:
    the new revision after a replace (``SAVED``/``DURABILITY_UNCERTAIN``),
    the current revision when the write was refused, and ``None`` when the
    lock could not be acquired.
    """

    outcome: CheckedWriteOutcome
    revision: str | None = None
    error: str | None = None


def store_revision(path: Path) -> str:
    """Content-hash revision of a backing store; the missing-store sentinel when absent."""
    try:
        return _content_revision(path.read_bytes())
    except FileNotFoundError:
        return MISSING_BACKING_STORE_DATA_FINGERPRINT


def checked_write_store(
    path: Path,
    edit: Callable[[Any], bytes],
    *,
    parse: Callable[[bytes], Any],
    missing_value: Any = None,
    expected_revision: str | None = None,
) -> CheckedWriteResult:
    """Atomically read, check, edit, and durably replace one backing store.

    The whole sequence runs under the store's exclusive cross-process lock
    as one primitive: read the current bytes, refuse a malformed store,
    check ``expected_revision``, produce the new payload with ``edit``,
    write and fsync a temp file, re-check the revision under the lock,
    replace atomically, and fsync the directory. ``parse`` interprets the
    current bytes and raises on malformed content; ``missing_value`` is
    handed to ``edit`` when the store does not exist yet. ``edit`` receives
    the parsed current content (or ``missing_value``) and returns the new
    payload bytes; an exception from ``edit`` propagates (caller error).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _exclusive_write_lock(path):
            return _checked_write_locked(
                path,
                edit,
                parse=parse,
                missing_value=missing_value,
                expected_revision=expected_revision,
            )
    except TimeoutError:
        return CheckedWriteResult(
            CheckedWriteOutcome.LOCKED,
            error=f"timed out waiting for the write lock of {path}",
        )


def _content_revision(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _read_store(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _checked_write_locked(
    path: Path,
    edit: Callable[[Any], bytes],
    *,
    parse: Callable[[bytes], Any],
    missing_value: Any,
    expected_revision: str | None,
) -> CheckedWriteResult:
    current_bytes = _read_store(path)
    if current_bytes is None:
        revision = MISSING_BACKING_STORE_DATA_FINGERPRINT
        current = missing_value
    else:
        revision = _content_revision(current_bytes)
        try:
            current = parse(current_bytes)
        except Exception as exc:
            return CheckedWriteResult(
                CheckedWriteOutcome.MALFORMED,
                revision=revision,
                error=f"backing store is malformed and was not rewritten: {exc}",
            )
    if expected_revision is not None and revision != expected_revision:
        return CheckedWriteResult(
            CheckedWriteOutcome.CONFLICT,
            revision=revision,
            error=f"expected revision {expected_revision}, store carries {revision}",
        )
    return _replace_store(
        path, edit(current), revision=revision, expected_revision=expected_revision
    )


def _replace_store(
    path: Path, payload: bytes, *, revision: str, expected_revision: str | None
) -> CheckedWriteResult:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            # Assigned before the fsync so a pre-replace failure still cleans
            # the temp file up in the finally below.
            tmp_path = Path(tmp_file.name)
            tmp_file.write(payload)
            tmp_file.flush()
            _fsync(tmp_file.fileno())
        if expected_revision is not None:
            # Re-read under the lock: an external writer that replaced the
            # store during the temp write must be reported, not overwritten.
            actual = store_revision(path)
            if actual != expected_revision:
                return CheckedWriteResult(
                    CheckedWriteOutcome.CONFLICT,
                    revision=actual,
                    error=(
                        f"expected revision {expected_revision}, store carries {actual}"
                    ),
                )
        os.replace(tmp_path, path)
        tmp_path = None
        new_revision = _content_revision(payload)
        try:
            _fsync_directory(path.parent)
        except OSError as exc:
            return CheckedWriteResult(
                CheckedWriteOutcome.DURABILITY_UNCERTAIN,
                revision=new_revision,
                error=f"store replaced but directory fsync failed: {exc}",
            )
        return CheckedWriteResult(CheckedWriteOutcome.SAVED, revision=new_revision)
    except OSError as exc:
        return CheckedWriteResult(
            CheckedWriteOutcome.NOT_SAVED, revision=revision, error=str(exc)
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _fsync(fd: int) -> None:
    os.fsync(fd)


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
