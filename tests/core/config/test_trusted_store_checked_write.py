"""Checked-write primitive arms for small file-backed stores (plan row O4a1).

Pattern carrier: tests/core/config/test_config_write_conflict.py. The
primitive under test is the synchronous sibling of the async layer
mechanics, chartreux/core/config/layers/_checked_store.py. It stays inert
for trust writers until O4a2; nothing here touches trusted_folders.py.
"""

from __future__ import annotations

from collections.abc import Callable
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import tomllib
from typing import Any

import pytest
import tomli_w

from chartreux.core.config.fingerprint import create_file_fingerprint
from chartreux.core.config.layers import _base, _checked_store
from chartreux.core.config.layers._checked_store import (
    CheckedWriteOutcome,
    checked_write_store,
    store_revision,
)
from chartreux.core.config.types import MISSING_BACKING_STORE_DATA_FINGERPRINT

ORIGINAL = b'trusted = ["/a"]\n'
EXTERNAL = b'trusted = ["/external"]\n'
MALFORMED = b"trusted = [ not valid toml \n"

_CHILD_HOLD_LOCK = """
import fcntl
import sys
import time
with open(sys.argv[1], "a") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    print("locked", flush=True)
    time.sleep(1.5)
"""

_CHILD_CHECKED_WRITE = """
import sys
import tomllib
from pathlib import Path
import tomli_w
from chartreux.core.config.layers._checked_store import (
    checked_write_store,
    store_revision,
)
path = Path(sys.argv[1])
def edit(current):
    data = dict(current)
    trusted = list(data.get("trusted", []))
    if "/child" not in trusted:
        trusted.append("/child")
    data["trusted"] = trusted
    return tomli_w.dumps(data).encode()
def parse(data):
    return tomllib.loads(data.decode())
result = checked_write_store(
    path, edit, parse=parse, expected_revision=store_revision(path)
)
print(result.outcome.value, result.revision, flush=True)
"""


def _parse_toml(data: bytes) -> Any:
    return tomllib.loads(data.decode())


def _toml_edit(extra: str) -> Callable[[Any], bytes]:
    def edit(current: Any) -> bytes:
        data = dict(current) if isinstance(current, dict) else {}
        trusted = list(data.get("trusted", []))
        if extra not in trusted:
            trusted.append(extra)
        data["trusted"] = trusted
        return tomli_w.dumps(data).encode()

    return edit


def _tmp_files(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*.tmp"))


def test_missing_store_is_created_under_the_missing_revision(tmp_path: Path) -> None:
    path = tmp_path / "store.toml"

    result = checked_write_store(
        path,
        _toml_edit("/a"),
        parse=_parse_toml,
        missing_value={},
        expected_revision=MISSING_BACKING_STORE_DATA_FINGERPRINT,
    )

    assert result.outcome == CheckedWriteOutcome.SAVED
    assert _parse_toml(path.read_bytes()) == {"trusted": ["/a"]}
    assert result.revision == store_revision(path)
    assert result.revision != MISSING_BACKING_STORE_DATA_FINGERPRINT
    assert _base._write_lock_path(path).is_file()
    assert _tmp_files(path) == []


def test_atomic_replace_uses_a_new_inode_and_leaves_no_temp_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store.toml"
    path.write_bytes(ORIGINAL)
    old_inode = path.stat().st_ino

    result = checked_write_store(
        path,
        _toml_edit("/b"),
        parse=_parse_toml,
        expected_revision=store_revision(path),
    )

    assert result.outcome == CheckedWriteOutcome.SAVED
    # Rename-based replace: exact bytes, a fresh inode, no truncate-and-write.
    assert path.read_bytes() == tomli_w.dumps({"trusted": ["/a", "/b"]}).encode()
    assert path.stat().st_ino != old_inode
    assert result.revision == store_revision(path)
    assert _tmp_files(path) == []


def test_file_fsync_precedes_replace_and_directory_fsync_follows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "store.toml"
    path.write_bytes(ORIGINAL)
    events: list[str] = []
    real_fsync = _checked_store._fsync
    real_fsync_dir = _checked_store._fsync_directory
    real_replace = os.replace

    def rec_fsync(fd: int) -> None:
        events.append("file-fsync")
        real_fsync(fd)

    def rec_fsync_dir(directory: Path) -> None:
        events.append("dir-fsync")
        real_fsync_dir(directory)

    def rec_replace(src: Any, dst: Any) -> None:
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(_checked_store, "_fsync", rec_fsync)
    monkeypatch.setattr(_checked_store, "_fsync_directory", rec_fsync_dir)
    monkeypatch.setattr(os, "replace", rec_replace)

    result = checked_write_store(
        path,
        _toml_edit("/b"),
        parse=_parse_toml,
        expected_revision=store_revision(path),
    )

    assert result.outcome == CheckedWriteOutcome.SAVED
    assert events == ["file-fsync", "replace", "dir-fsync"]


def test_pre_replace_failure_is_not_saved_and_keeps_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "store.toml"
    path.write_bytes(ORIGINAL)
    old_revision = store_revision(path)

    def boom(fd: int) -> None:
        raise OSError("simulated disk full during temp fsync")

    monkeypatch.setattr(_checked_store, "_fsync", boom)

    result = checked_write_store(
        path, _toml_edit("/b"), parse=_parse_toml, expected_revision=old_revision
    )

    assert result.outcome == CheckedWriteOutcome.NOT_SAVED
    assert result.revision == old_revision
    assert path.read_bytes() == ORIGINAL
    assert _tmp_files(path) == []


def test_post_replace_fsync_failure_is_durability_uncertain_not_not_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "store.toml"
    path.write_bytes(ORIGINAL)
    payload = tomli_w.dumps({"trusted": ["/a", "/b"]}).encode()

    def boom_dir(directory: Path) -> None:
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(_checked_store, "_fsync_directory", boom_dir)

    result = checked_write_store(
        path,
        lambda current: payload,
        parse=_parse_toml,
        expected_revision=store_revision(path),
    )

    # Explicit distinct outcome: the replace happened, only durability is
    # unconfirmed — never a silent yes and never "definitely not saved".
    assert result.outcome == CheckedWriteOutcome.DURABILITY_UNCERTAIN
    assert result.outcome != CheckedWriteOutcome.SAVED
    assert result.outcome != CheckedWriteOutcome.NOT_SAVED
    assert result.revision == _checked_store._content_revision(payload)
    assert path.read_bytes() == payload
    assert _tmp_files(path) == []


def test_malformed_store_is_refused_byte_for_byte(tmp_path: Path) -> None:
    path = tmp_path / "store.toml"
    path.write_bytes(MALFORMED)

    result = checked_write_store(
        path,
        _toml_edit("/b"),
        parse=_parse_toml,
        expected_revision=store_revision(path),
    )

    assert result.outcome == CheckedWriteOutcome.MALFORMED
    assert result.revision == store_revision(path)
    assert path.read_bytes() == MALFORMED
    assert _tmp_files(path) == []


def test_held_lock_stale_writer_is_refused(tmp_path: Path) -> None:
    """Mutation target (plan O4a1): removing the revision check fails here."""
    path = tmp_path / "store.toml"
    path.write_bytes(ORIGINAL)
    stale_revision = store_revision(path)
    box: dict[str, Any] = {}

    def stale_writer() -> None:
        box["result"] = checked_write_store(
            path,
            _toml_edit("/stale"),
            parse=_parse_toml,
            expected_revision=stale_revision,
        )

    with _base._write_lock_path(path).open("a") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        writer = threading.Thread(target=stale_writer)
        writer.start()
        time.sleep(0.1)
        # The stale writer waits for the cooperating lock holder.
        assert writer.is_alive()
        # An external writer moves the store while the stale writer waits.
        path.write_bytes(EXTERNAL)
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        writer.join(timeout=5)

    result = box["result"]
    assert result.outcome == CheckedWriteOutcome.CONFLICT
    assert result.revision == store_revision(path)
    # The external writer's content survives; no stale resurrection.
    assert path.read_bytes() == EXTERNAL
    assert _tmp_files(path) == []


def test_external_write_during_temp_write_conflicts_under_the_held_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "store.toml"
    path.write_bytes(ORIGINAL)
    real_fsync = _checked_store._fsync

    def intervening_fsync(fd: int) -> None:
        real_fsync(fd)
        # A non-cooperating writer replaces the store after the first
        # under-lock revision read and before the replacement.
        path.write_bytes(EXTERNAL)

    monkeypatch.setattr(_checked_store, "_fsync", intervening_fsync)

    result = checked_write_store(
        path,
        _toml_edit("/b"),
        parse=_parse_toml,
        expected_revision=store_revision(path),
    )

    assert result.outcome == CheckedWriteOutcome.CONFLICT
    assert result.revision == _checked_store._content_revision(EXTERNAL)
    assert path.read_bytes() == EXTERNAL
    assert _tmp_files(path) == []


def test_cross_process_lock_holder_blocks_until_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "store.toml"
    path.write_bytes(ORIGINAL)
    monkeypatch.setattr(_base, "_WRITE_LOCK_ACQUIRE_TIMEOUT", 0.2)

    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_HOLD_LOCK, str(_base._write_lock_path(path))],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"

        result = checked_write_store(
            path,
            _toml_edit("/b"),
            parse=_parse_toml,
            expected_revision=store_revision(path),
        )

        assert result.outcome == CheckedWriteOutcome.LOCKED
        assert result.revision is None
        assert path.read_bytes() == ORIGINAL
    finally:
        child.wait(timeout=10)
    assert child.returncode == 0

    # After the holder exits, the bounded wait no longer applies.
    ok = checked_write_store(
        path,
        _toml_edit("/b"),
        parse=_parse_toml,
        expected_revision=store_revision(path),
    )
    assert ok.outcome == CheckedWriteOutcome.SAVED
    assert _parse_toml(path.read_bytes()) == {"trusted": ["/a", "/b"]}


def test_cross_process_write_is_refused_for_a_stale_reader_until_reread(
    tmp_path: Path,
) -> None:
    """Restart semantics at the primitive: a reader holding a stale revision
    is refused; only a fresh read (restart) observes the other process's write.
    """
    path = tmp_path / "store.toml"
    path.write_bytes(ORIGINAL)
    stale_revision = store_revision(path)

    child = subprocess.run(
        [sys.executable, "-c", _CHILD_CHECKED_WRITE, str(path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    child_outcome, child_revision = child.stdout.split()
    assert child_outcome == CheckedWriteOutcome.SAVED.value

    # The stale in-memory revision is refused, not silently applied.
    refused = checked_write_store(
        path, _toml_edit("/stale"), parse=_parse_toml, expected_revision=stale_revision
    )
    assert refused.outcome == CheckedWriteOutcome.CONFLICT
    # A fresh read observes the other process's write.
    assert store_revision(path) == child_revision
    assert _parse_toml(path.read_bytes()) == {"trusted": ["/a", "/child"]}


def test_same_size_inplace_write_within_one_mtime_tick_is_detected(
    tmp_path: Path,
) -> None:
    """The stat fingerprint (dev:ino:mtime_ns:size) is blind to a same-size
    in-place write within one mtime tick; the primitive's content-hash
    revision is not, so the stale write still conflicts.
    """
    path = tmp_path / "store.toml"
    path.write_bytes(b'trusted = ["/a"]\n')
    stale_revision = store_revision(path)
    with path.open("rb") as file:
        stat_before = create_file_fingerprint(file)

    replacement = b'trusted = ["/b"]\n'
    assert len(replacement) == path.stat().st_size
    stat = path.stat()
    path.write_bytes(replacement)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with path.open("rb") as file:
        stat_after = create_file_fingerprint(file)

    # The stat-based scheme cannot see this change...
    assert stat_before == stat_after
    # ...but the content-hash revision does.
    assert store_revision(path) != stale_revision
    result = checked_write_store(
        path, _toml_edit("/c"), parse=_parse_toml, expected_revision=stale_revision
    )
    assert result.outcome == CheckedWriteOutcome.CONFLICT
    assert path.read_bytes() == replacement
