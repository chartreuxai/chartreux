"""Owner-only permission repair for application-owned storage."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import stat
from typing import TYPE_CHECKING

from chartreux.utils.private_paths import (
    ensure_private_directory as ensure_private_directory,
    open_nofollow as _open_nofollow,
    restrict_fd as _restrict_fd,
    restrict_private_file as restrict_private_file,
    restrict_private_path as restrict_private_path,
)

if TYPE_CHECKING:
    from chartreux.core.config import SessionLoggingConfig

logger = logging.getLogger(__name__)
METADATA_FILENAME = "meta.json"
MESSAGES_FILENAME = "messages.jsonl"
_SESSION_FILES = frozenset({METADATA_FILENAME, MESSAGES_FILENAME})
_QUARANTINE_PREFIX = f"{MESSAGES_FILENAME}.corrupt-"
_SESSION_DIRECTORIES = frozenset({"attachments"})


def _recognized_session_fd(root_fd: int, name: str, prefix: str) -> int | None:
    if not name.startswith(f"{prefix}_"):
        return None
    session_fd: int | None = None
    metadata_fd: int | None = None
    messages_fd: int | None = None
    try:
        session_fd = os.open(
            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
        )
        metadata_fd = os.open(
            METADATA_FILENAME,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=session_fd,
        )
        if not stat.S_ISREG(os.fstat(metadata_fd).st_mode):
            raise OSError("session metadata marker must be a regular file")
        try:
            messages_fd = os.open(
                MESSAGES_FILENAME,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=session_fd,
            )
        except FileNotFoundError:
            messages_fd = None
        if messages_fd is not None:
            if not stat.S_ISREG(os.fstat(messages_fd).st_mode):
                raise OSError("session transcript marker must be a regular file")
            os.close(messages_fd)
            messages_fd = None
        with os.fdopen(metadata_fd, encoding="utf-8") as metadata_file:
            metadata_fd = None
            metadata = json.load(metadata_file)
        if not isinstance(metadata, dict) or not (
            isinstance(metadata.get("session_id"), str) and metadata["session_id"]
        ):
            os.close(session_fd)
            return None
        return session_fd
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        if messages_fd is not None:
            os.close(messages_fd)
        if metadata_fd is not None:
            os.close(metadata_fd)
        if session_fd is not None:
            os.close(session_fd)
        return None


def _restrict_named_file(parent_fd: int, name: str) -> int:
    try:
        fd = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
        )
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return 0
            return _restrict_fd(fd)
        finally:
            os.close(fd)
    except OSError:
        return 0


def _restrict_known_session_artifacts(session_fd: int) -> int:
    tightened = _restrict_fd(session_fd)
    for filename in os.listdir(session_fd):
        if filename in _SESSION_FILES or filename.startswith(_QUARANTINE_PREFIX):
            tightened += _restrict_named_file(session_fd, filename)
    for dirname in _SESSION_DIRECTORIES:
        try:
            directory_fd = os.open(
                dirname, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=session_fd
            )
        except OSError:
            continue
        try:
            tightened += _restrict_fd(directory_fd)
            for filename in os.listdir(directory_fd):
                tightened += _restrict_named_file(directory_fd, filename)
        finally:
            os.close(directory_fd)
    return tightened


def restrict_session_permissions(config: SessionLoggingConfig) -> int:
    """Repair store-recognized sessions and only their known artifact vocabulary."""
    if os.name != "posix":
        return 0
    root = Path(config.permission_repair_dir or config.save_dir)
    try:
        root_fd = _open_nofollow(root, directory=True)
    except OSError as error:
        logger.debug("Session permission repair skipped root=%s err=%s", root, error)
        return 0
    try:
        tightened = _restrict_fd(root_fd)
        for name in os.listdir(root_fd):
            session_fd = _recognized_session_fd(root_fd, name, config.session_prefix)
            if session_fd is None:
                continue
            try:
                tightened += _restrict_known_session_artifacts(session_fd)
            except OSError as error:
                logger.debug(
                    "Session permission repair skipped path=%s err=%s", name, error
                )
            finally:
                os.close(session_fd)
        return tightened
    except OSError as error:
        logger.debug("Session permission repair skipped root=%s err=%s", root, error)
        return 0
    finally:
        os.close(root_fd)
