"""Blocking, POSIX-only ledger operations; callers provide a thread lock.

The stable sidecar lock serializes independent writer instances/processes. Never
replace or unlink it. Every reconciliation repeats file and directory barriers:
visible bytes alone are not proof of a durably settled append.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
from typing import Any

from chartreux.utils.durable_io import durable_append, durable_mkdir
from chartreux.utils.private_paths import open_nofollow, restrict_fd


@dataclass(frozen=True)
class ParsedUsageLines:
    records: tuple[dict[str, Any], ...]
    malformed_lines: int = 0
    torn_tail: bool = False


def parse_usage_jsonl(content: bytes) -> ParsedUsageLines:
    """Recover JSON objects without losing valid history around damaged lines.

    A valid final object without a newline is retained (important for dedup after
    a short write of the delimiter). An invalid unterminated tail is ignored and
    flagged separately from malformed, newline-terminated records. Schema/model
    validation belongs to the reader, not this framing helper.
    """
    records: list[dict[str, Any]] = []
    malformed = 0
    lines = content.split(b"\n")
    torn_tail = bool(lines[-1])
    for index, line in enumerate(lines):
        if not line:
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Ledger lines must be objects")
        except (ValueError, UnicodeDecodeError, RecursionError):
            if index != len(lines) - 1:
                malformed += 1
            continue
        records.append(value)
    return ParsedUsageLines(tuple(records), malformed, torn_tail)


def read_usage_file(path: Path) -> bytes:
    """Read a ledger without following a final-component symlink."""
    descriptor = open_nofollow(path, directory=False)
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def reserve_usage_root(usage_dir: Path, root_session_id: str) -> bool:
    """Exclusively claim an empty ledger directory, never a session/transcript.

    Return False only for a root collision; other IO failures degrade coverage.
    A failed publication barrier may leave the claimed directory behind; never
    recycle it.
    """
    if (
        not root_session_id
        or root_session_id in {".", ".."}
        or any(separator in root_session_id for separator in ("/", "\\", "\x00"))
    ):
        raise ValueError("Root session ID must be a single path component")
    durable_mkdir(usage_dir)
    directory = usage_dir / root_session_id
    try:
        directory.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError:
        return False
    durable_mkdir(directory)
    return True


def append_usage_record(
    usage_dir: Path, root_session_id: str, record_id: str, content: bytes
) -> bool:
    """Append/reconcile one record; return whether new bytes were appended.

    Errors propagate to the non-raising writer facade. Retrying with the same ID
    is safe even after a write, file sync, or directory publication failure.
    """
    if (
        not root_session_id
        or root_session_id in {".", ".."}
        or any(separator in root_session_id for separator in ("/", "\\", "\x00"))
    ):
        raise ValueError("Root session ID must be a single path component")
    directory = usage_dir / root_session_id
    durable_mkdir(directory)
    for private_dir in (usage_dir, directory):
        descriptor = open_nofollow(private_dir, directory=True)
        try:
            restrict_fd(descriptor)
        finally:
            os.close(descriptor)
    lock_fd = os.open(
        directory / ".usage.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        restrict_fd(lock_fd)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        path = directory / "usage.jsonl"
        try:
            descriptor = open_nofollow(path, directory=False)
        except FileNotFoundError:
            existing = b""
        else:
            with os.fdopen(descriptor, "rb") as stream:
                restrict_fd(stream.fileno())
                existing = stream.read()
        present = any(
            record.get("record_id") == record_id
            for record in parse_usage_jsonl(existing).records
        )
        # ensure_newline never truncates: even an invalid tail remains as a
        # separate damaged line. Empty content still flushes/fsyncs and publishes
        # the file's directory entry on record-present reconciliation.
        durable_append(path, b"" if present else content, ensure_newline=True)
        return not present
    finally:
        # Closing also releases flock, including when acquisition failed.
        os.close(lock_fd)
