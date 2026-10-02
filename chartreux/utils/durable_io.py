"""Opt-in persistence barriers; errors propagate and writes are not transactional.

Callers must serialize writers. A failed barrier may leave published bytes on disk;
retrying must reconcile that state rather than assume the write did not happen.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import tempfile
from threading import RLock

# Only completed ancestor-publication chains enter this process-local cache.
_SYNCED_DIRECTORIES: set[tuple[int, int]] = set()
_DIRECTORY_LOCK = RLock()


def fsync_directory(path: Path) -> None:
    """Sync a directory, raising if the platform or filesystem cannot do so."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_mkdir(path: Path) -> None:
    """Create owner-only directories and durably publish their ancestor links.

    Existing links are synced on first observation, including after a failed
    attempt. A cached parent still needs a barrier to publish a new child, but
    its own already-durable ancestor links do not need another barrier.
    """
    path = path.absolute()
    with _DIRECTORY_LOCK:
        missing: list[Path] = []
        for directory in (path, *path.parents):
            if directory.exists():
                break
            missing.append(directory)
        for directory in reversed(missing):
            directory.mkdir(mode=0o700, exist_ok=True)
        completed: set[tuple[int, int]] = set()
        for directory in (path, *path.parents):
            info = directory.stat()
            identity = (info.st_dev, info.st_ino)
            if identity in _SYNCED_DIRECTORIES:
                if completed:
                    # Publish the last uncached child's dirent in this parent.
                    fsync_directory(directory)
                break
            fsync_directory(directory)
            completed.add(identity)
        # Do not remember even successful interior barriers if the chain failed.
        _SYNCED_DIRECTORIES.update(completed)


def durable_append(
    path: Path,
    content: bytes,
    *,
    ensure_newline: bool = False,
    sync_parent: bool = True,
) -> None:
    """Append bytes and sync the file, optionally syncing its parent directory.

    The parent must exist. New files are owner-only; existing modes are retained.
    The default directory barrier also covers failed new-file publication retries.
    Serialized callers may omit it for an already durably published file only.
    """
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "a+b") as stream:
        if ensure_newline and stream.seek(0, os.SEEK_END):
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    if sync_parent:
        fsync_directory(path.parent)


def durable_replace(path: Path, content: bytes) -> None:
    """Publish an owner-only same-directory replacement with file and dir barriers.

    The parent must exist. Failure after replace is not a rollback.
    """
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()
