from __future__ import annotations

import logging
import multiprocessing
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from chartreux.core.config import SessionLoggingConfig
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session.session_permissions import (
    ensure_private_directory,
    restrict_private_path,
    restrict_session_permissions,
)
from chartreux.observability.logging import _ChartreuxFileHandler, init_file_logging

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX modes only")


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _config(root: Path, *, enabled: bool = True) -> SessionLoggingConfig:
    return SessionLoggingConfig(
        save_dir=str(root), session_prefix="test", enabled=enabled
    )


def _repair_in_child(root: str) -> None:
    restrict_session_permissions(_config(Path(root)))


def _legacy_session(root: Path, name: str = "test_old") -> tuple[Path, Path]:
    session = root / name
    session.mkdir(parents=True)
    metadata = session / "meta.json"
    metadata.write_text('{"session_id": "legacy"}')
    messages = session / "messages.jsonl"
    messages.write_text("")
    nested = session / "attachments"
    nested.mkdir()
    artifact = nested / "secret.txt"
    artifact.write_text("secret")
    for path in (root, session, metadata, messages, nested, artifact):
        path.chmod(0o775 if path.is_dir() else 0o674)
    return session, artifact


def test_existing_session_tree_loses_group_other_and_preserves_owner_bits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sessions"
    session, artifact = _legacy_session(root)
    artifact.chmod(0o754)

    assert restrict_session_permissions(_config(root)) == 6
    assert _mode(root) == 0o700
    assert _mode(session) == 0o700
    assert _mode(artifact) == 0o700
    assert restrict_session_permissions(_config(root)) == 0


def test_repair_skips_symlinks_and_unrelated_root_files(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    session, _ = _legacy_session(root)
    target = tmp_path / "target"
    target.write_text("outside")
    target.chmod(0o666)
    (session / "linked").symlink_to(target)
    sentinel = root / "sentinel.txt"
    sentinel.write_text("unrelated")
    sentinel.chmod(0o666)

    restrict_session_permissions(_config(root))

    assert _mode(target) == 0o666
    assert _mode(sentinel) == 0o666


def test_decoy_messages_directory_and_unrelated_files_are_untouched(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sessions"
    decoy = root / "test_decoy"
    decoy.mkdir(parents=True)
    (decoy / "messages.jsonl").write_text("decoy")
    fake_env = decoy / ".env"
    fake_env.write_text("not-a-secret")
    nested = decoy / "unrelated"
    nested.mkdir()
    for path in (decoy, decoy / "messages.jsonl", fake_env, nested):
        path.chmod(0o775 if path.is_dir() else 0o666)

    restrict_session_permissions(_config(root))

    assert _mode(decoy) == 0o775
    assert _mode(decoy / "messages.jsonl") == 0o666
    assert _mode(fake_env) == 0o666
    assert _mode(nested) == 0o775


def test_fifo_transcript_marker_does_not_block_repair(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    session = root / "test_fifo"
    session.mkdir(parents=True, mode=0o755)
    session.chmod(0o755)
    (session / "meta.json").write_text('{"session_id": "fifo"}')
    os.mkfifo(session / "messages.jsonl", mode=0o666)
    initial_session_mode = _mode(session)

    process = multiprocessing.Process(target=_repair_in_child, args=(str(root),))
    process.start()
    process.join(timeout=2)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("permission repair blocked while opening a FIFO")

    assert process.exitcode == 0
    assert _mode(session) == initial_session_mode


def test_metadata_only_session_and_quarantine_are_repaired(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    session = root / "test_metadata_only"
    session.mkdir(parents=True)
    metadata = session / "meta.json"
    metadata.write_text('{"session_id": "metadata-only"}')
    quarantine = session / "messages.jsonl.corrupt-20250101T000000Z-deadbeef"
    quarantine.write_text("corrupt")
    for path in (root, session, metadata, quarantine):
        path.chmod(0o775 if path.is_dir() else 0o674)

    assert restrict_session_permissions(_config(root)) == 4
    assert _mode(session) == 0o700
    assert _mode(metadata) == 0o600
    assert _mode(quarantine) == 0o600


def test_each_configured_root_repaired_even_when_logging_disabled(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "one", tmp_path / "two"]
    sessions = [_legacy_session(root)[0] for root in roots]

    for root in roots:
        SessionLogger(_config(root, enabled=False), "ignored")

    assert all(_mode(root) == 0o700 for root in roots)
    assert all(_mode(session) == 0o700 for session in sessions)


def test_private_directory_leaves_parent_unchanged(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = parent / "home"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    initial_parent_mode = _mode(parent)
    child.mkdir(mode=0o755)

    ensure_private_directory(child)

    assert _mode(parent) == initial_parent_mode
    assert _mode(child) == 0o700


def test_symlinked_root_or_ancestor_is_not_repaired(tmp_path: Path) -> None:
    real = tmp_path / "real"
    _legacy_session(real)
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)

    assert restrict_session_permissions(_config(link)) == 0
    assert _mode(real) == 0o775


def test_logger_does_not_chmod_target_of_explicit_symlink_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o775)
    real.chmod(0o775)
    initial_mode = _mode(real)
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)

    SessionLogger(_config(link), "new")

    assert _mode(real) == initial_mode


def test_permission_errors_are_nonfatal(tmp_path: Path) -> None:
    path = tmp_path / "private"
    path.write_text("x")
    with patch(
        "chartreux.core.session.session_permissions.os.fchmod", side_effect=OSError
    ):
        assert restrict_private_path(path) == 0


def test_existing_log_directory_and_file_become_private(tmp_path: Path) -> None:
    directory = tmp_path / "logs"
    directory.mkdir(mode=0o755)
    log_file = directory / "chartreux.log"
    log_file.write_text("old")
    log_file.chmod(0o644)
    target = logging.getLogger(f"permissions-{id(tmp_path)}")

    init_file_logging(log_file, target_logger=target)

    assert _mode(directory) == 0o700
    assert _mode(log_file) == 0o600
    for handler in list(target.handlers):
        handler.close()
        target.removeHandler(handler)


def test_rotated_logs_are_private(tmp_path: Path) -> None:
    log_file = tmp_path / "messages.jsonl"
    handler = _ChartreuxFileHandler(log_file, maxBytes=1, backupCount=2)
    handler.emit(logging.makeLogRecord({"msg": "first"}))
    handler.emit(logging.makeLogRecord({"msg": "second"}))
    handler.close()

    assert all(_mode(path) & 0o077 == 0 for path in tmp_path.glob("messages.jsonl*"))


def test_session_metadata_uses_trusted_git_executable(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    marker = tmp_path / "git-marker"
    executable = tmp_path / "trusted-git"
    executable.write_text(
        f"#!/bin/sh\nprintf invoked > {marker}\nprintf 'abc123\\nmain\\n'\n"
    )
    executable.chmod(0o700)

    with patch(
        "chartreux.core.session.session_logger.resolve_git_executable",
        return_value=str(executable),
    ):
        logger = SessionLogger(_config(root), "trusted", cwd=tmp_path)

    assert marker.read_text() == "invoked"
    assert logger.session_metadata is not None
    assert logger.session_metadata.git_commit == "abc123"
    assert logger.session_metadata.git_branch == "main"
