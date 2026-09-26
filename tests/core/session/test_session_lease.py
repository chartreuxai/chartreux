from __future__ import annotations

import os
from pathlib import Path

import pytest

from chartreux.core.session import session_lease as lease_module
from chartreux.core.session.session_lease import SessionBusyError, SessionLease

SESSION_ID = "019ffb1e-741d-7f90-84df-ef66011876ca"


def test_session_lease_is_exclusive_and_recoverable(tmp_path: Path) -> None:
    first = SessionLease(tmp_path, SESSION_ID).acquire()
    try:
        with pytest.raises(SessionBusyError):
            SessionLease(tmp_path, SESSION_ID).acquire()
    finally:
        first.release()

    assert not first.path.exists()
    SessionLease(tmp_path, SESSION_ID).acquire().release()


def test_session_lease_rejects_a_path_shaped_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid session ID"):
        SessionLease(tmp_path, "../escape")


def test_session_lease_accepts_a_safe_legacy_identity(tmp_path: Path) -> None:
    lease = SessionLease(tmp_path, "resumable-with-stats").acquire()

    assert lease.path == tmp_path / "active" / "resumable-with-stats.lock"
    lease.release()


def test_session_lease_rejects_a_symlinked_active_namespace(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "active").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        SessionLease(tmp_path, SESSION_ID).acquire()


def test_session_lease_rolls_back_when_the_diagnostic_cannot_be_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_fsync(_fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    with pytest.raises(OSError, match="disk full"):
        SessionLease(tmp_path, SESSION_ID).acquire()

    monkeypatch.undo()
    # The failed acquire must not leave the lock held or the file behind.
    assert not (tmp_path / "active" / f"{SESSION_ID}.lock").exists()
    SessionLease(tmp_path, SESSION_ID).acquire().release()


def test_session_lease_rolls_back_on_a_base_exception_during_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Cancelled(BaseException):
        """Stands in for Ctrl-C: outside the Exception hierarchy."""

    def cancelled() -> str:
        raise Cancelled

    monkeypatch.setattr(lease_module, "_timestamp", cancelled)

    with pytest.raises(Cancelled):
        SessionLease(tmp_path, SESSION_ID).acquire()

    monkeypatch.undo()
    assert not (tmp_path / "active" / f"{SESSION_ID}.lock").exists()
    SessionLease(tmp_path, SESSION_ID).acquire().release()


def test_session_lease_release_tolerates_an_unlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = SessionLease(tmp_path, SESSION_ID).acquire()

    def failing_unlink(self: Path, *, missing_ok: bool = False) -> None:
        raise OSError("unlink failed")

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    # The release must succeed even when the cleanup unlink fails.
    lease.release()
