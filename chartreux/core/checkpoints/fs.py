from __future__ import annotations

import os
from pathlib import Path
import secrets
import shutil
from typing import Protocol


class Filesystem(Protocol):
    """Disk operations the checkpoint file store needs. ``read_bytes`` returns
    None only when the file is absent, and raises when a file exists but cannot
    be read, so a transient read failure is never mistaken for a deletion. The
    others raise on failure.
    """

    def read_bytes(self, path: str) -> bytes | None: ...

    def write_bytes(self, path: str, data: bytes) -> None: ...

    def remove(self, path: str) -> None: ...

    def exists(self, path: str) -> bool: ...


class DiskFilesystem:
    def read_bytes(self, path: str) -> bytes | None:
        try:
            return Path(path).read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            return None

    def write_bytes(self, path: str, data: bytes) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Stage bytes beside the target; a failed write must not truncate it.
        # This is a single-file replacement, not a durable or multi-file commit.
        # Create existing-file staging privately, and restore its mode before
        # writing any bytes. Missing files retain normal creation permissions.
        tmp = target.parent / f".tmp.{secrets.token_hex(16)}"
        mode = 0o600 if target.exists() else 0o666
        staged = open(tmp, "xb", opener=lambda path, flags: os.open(path, flags, mode))
        try:
            with staged:
                try:
                    shutil.copymode(target, tmp)
                except FileNotFoundError:
                    pass
                staged.write(data)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)

    def remove(self, path: str) -> None:
        os.remove(path)

    def exists(self, path: str) -> bool:
        return Path(path).exists()
