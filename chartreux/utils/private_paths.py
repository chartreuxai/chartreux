"""Owner-only permission helpers for application-owned paths."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import stat

logger = logging.getLogger(__name__)
_GROUP_OTHER = 0o077


def open_nofollow(path: Path, *, directory: bool | None = None) -> int:
    """Open *path* through descriptor-bound, no-follow component traversal."""
    path = path.expanduser()
    parts = path.parts
    if path.is_absolute():
        fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        parts = parts[1:]
    else:
        fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if not last or directory is True:
                flags |= os.O_DIRECTORY
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        if directory is False and stat.S_ISDIR(info.st_mode):
            raise IsADirectoryError(path)
        return fd
    except BaseException:
        os.close(fd)
        raise


def restrict_fd(fd: int) -> int:
    """Strip group/other permission bits from an open descriptor."""
    mode = stat.S_IMODE(os.fstat(fd).st_mode)
    if not mode & _GROUP_OTHER:
        return 0
    os.fchmod(fd, mode & ~_GROUP_OTHER)
    return 1


def restrict_private_path(path: Path) -> int:
    """Strip group/other bits using a descriptor reached without following links."""
    if os.name != "posix":
        return 0
    try:
        fd = open_nofollow(path)
        try:
            return restrict_fd(fd)
        finally:
            os.close(fd)
    except OSError as error:
        logger.debug("Permission repair skipped path=%s err=%s", path, error)
        return 0


def restrict_private_file(path: Path) -> int:
    """Strip group/other bits from one regular file reached without following links."""
    if os.name != "posix":
        return 0
    try:
        # Only regular files are ever opened. Opening other file types can be
        # harmful by itself: opening a FIFO's read side unblocks a writer
        # blocked on the reader/writer handshake, so the S_ISREG check after
        # the open would skip the chmod but the damage is already done.
        if not stat.S_ISREG(os.lstat(path.expanduser()).st_mode):
            return 0
    except OSError as error:
        logger.debug("Permission repair skipped path=%s err=%s", path, error)
        return 0
    try:
        fd = open_nofollow(path, directory=False)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return 0
            return restrict_fd(fd)
        finally:
            os.close(fd)
    except OSError as error:
        logger.debug("Permission repair skipped path=%s err=%s", path, error)
        return 0


def ensure_private_directory(path: Path) -> None:
    """Create a private leaf directory if needed and safely repair an existing leaf."""
    try:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
    except OSError as error:
        logger.debug("Private directory setup skipped path=%s err=%s", path, error)
        return
    restrict_private_path(path)
