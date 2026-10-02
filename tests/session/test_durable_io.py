from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
import stat
from typing import Any
from unittest.mock import MagicMock

import pytest

from chartreux.utils import durable_io


@pytest.mark.parametrize("operation", ["append", "replace"])
@pytest.mark.parametrize(
    "stage", ["open", "write", "flush", "file_fsync", "directory_fsync"]
)
def test_strict_write_failures_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, stage: str
) -> None:
    path = tmp_path / "durable"
    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.name = str(tmp_path / "temporary")
    stream.fileno.return_value = 123

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected " + stage)

    # Keep real descriptors managed even when replacing their Python stream.
    with ExitStack() as cleanup:
        if stage in {"write", "flush"}:
            getattr(stream, stage).side_effect = fail
            if operation == "append":
                original_open = os.open

                def tracked_open(*args: Any, **kwargs: Any) -> int:
                    fd = original_open(*args, **kwargs)
                    cleanup.callback(os.close, fd)
                    return fd

                monkeypatch.setattr(os, "open", tracked_open)
                monkeypatch.setattr(os, "fdopen", lambda *a, **kw: stream)
            else:
                monkeypatch.setattr(
                    durable_io.tempfile, "NamedTemporaryFile", lambda **kw: stream
                )
        elif stage == "open":
            monkeypatch.setattr(os, "open", fail)
        else:
            original_fsync = os.fsync

            def fsync(fd: int) -> None:
                is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
                if is_dir == (stage == "directory_fsync"):
                    fail()
                original_fsync(fd)

            monkeypatch.setattr(os, "fsync", fsync)
        operation_fn = (
            durable_io.durable_append
            if operation == "append"
            else durable_io.durable_replace
        )
        with pytest.raises(OSError, match="injected " + stage):
            operation_fn(path, b"content")


def test_strict_replace_failure_raises_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "durable"
    path.write_bytes(b"old")

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected replace")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="injected replace"):
        durable_io.durable_replace(path, b"new")
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("operation", ["append", "replace"])
def test_strict_new_file_fsync_order_and_private_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    calls: list[str] = []
    original_fsync = os.fsync

    def fsync(fd: int) -> None:
        calls.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    path = tmp_path / "durable"
    operation_fn = (
        durable_io.durable_append
        if operation == "append"
        else durable_io.durable_replace
    )
    operation_fn(path, b"content")
    assert calls == ["file", "directory"]
    assert path.read_bytes() == b"content"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_durable_mkdir_publishes_new_ancestor_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(durable_io, "_SYNCED_DIRECTORIES", set())
    monkeypatch.setattr(durable_io, "fsync_directory", calls.append)
    path = tmp_path / "new" / "session"
    durable_io.durable_mkdir(path)
    assert calls == [path, *path.parents]
    assert path.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o700


def test_durable_mkdir_cache_tracks_identity_and_publishes_new_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(durable_io, "_SYNCED_DIRECTORIES", set())
    calls: list[Path] = []
    sync = durable_io.fsync_directory

    def observed(path: Path) -> None:
        calls.append(path)
        sync(path)

    monkeypatch.setattr(durable_io, "fsync_directory", observed)
    path = tmp_path / "session"
    durable_io.durable_mkdir(path)
    calls.clear()
    durable_io.durable_mkdir(path)
    assert calls == []
    child = path / "child"
    durable_io.durable_mkdir(child)
    assert calls == [child, path]
    # Keep the old inode allocated so recreation demonstrably has a new identity.
    path.rename(tmp_path / "old-session")
    calls.clear()
    durable_io.durable_mkdir(path)
    assert calls == [path, tmp_path]


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_durable_mkdir_failed_chain_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: int
) -> None:
    monkeypatch.setattr(durable_io, "_SYNCED_DIRECTORIES", set())
    calls: list[Path] = []
    sync = durable_io.fsync_directory
    path = tmp_path / "new" / "session"

    def interrupted(directory: Path) -> None:
        calls.append(directory)
        if len(calls) == stage + 1:
            raise OSError("injected ancestor barrier")
        sync(directory)

    monkeypatch.setattr(durable_io, "fsync_directory", interrupted)
    with pytest.raises(OSError, match="injected ancestor barrier"):
        durable_io.durable_mkdir(path)
    assert len(calls) == stage + 1
    assert durable_io._SYNCED_DIRECTORIES == set()
    calls.clear()

    def observed(directory: Path) -> None:
        calls.append(directory)
        sync(directory)

    monkeypatch.setattr(durable_io, "fsync_directory", observed)
    durable_io.durable_mkdir(path)
    assert calls == [path, *path.parents]


def test_durable_append_repairs_missing_newline(tmp_path: Path) -> None:
    path = tmp_path / "messages"
    path.write_bytes(b"first")
    durable_io.durable_append(path, b"second\n", ensure_newline=True)
    assert path.read_bytes() == b"first\nsecond\n"
