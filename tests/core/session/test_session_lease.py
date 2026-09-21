from __future__ import annotations

from pathlib import Path

import pytest

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
