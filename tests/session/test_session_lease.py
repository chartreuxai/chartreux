from __future__ import annotations

from collections.abc import Callable, Iterator
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

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

    # Post-state: the failed unlink really left the lock file behind (the
    # premise is exercised), but the session itself was released — a fresh
    # lease for the same session acquires cleanly.
    assert (tmp_path / "active" / f"{SESSION_ID}.lock").exists()
    SessionLease(tmp_path, SESSION_ID).acquire().release()


_READY_TIMEOUT_S = 5.0
_CHILD_WAIT_TIMEOUT_S = 5.0

# The holder publishes an OS-level lease (flock) and then waits for the parent
# to create its stop file. Killing it mid-wait is the owner-death scenario: no
# release, no diagnostic rewrite, the file left behind.
_LEASE_HOLDER_SCRIPT = """\
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from chartreux.core.session.session_lease import SessionLease

root = Path(sys.argv[1])
session_id = sys.argv[2]
ready_path = Path(sys.argv[3])
stop_path = Path(sys.argv[4])

lease = SessionLease(root, session_id).acquire()
try:
    ready_path.write_text(json.dumps({"process_id": os.getpid()}))
    deadline = time.monotonic() + 30.0
    while not stop_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
finally:
    lease.release()
"""


class _LeaseHolder:
    def __init__(
        self,
        process: subprocess.Popen[str],
        root: Path,
        ready_path: Path,
        stop_path: Path,
    ) -> None:
        self.process = process
        self.root = root
        self.ready_path = ready_path
        self.stop_path = stop_path

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def lock_path(self) -> Path:
        return self.root / "active" / f"{SESSION_ID}.lock"

    def ready(self) -> bool:
        return self.ready_path.exists()

    def alive(self) -> bool:
        return self.process.poll() is None

    def kill(self) -> int:
        if self.alive():
            self.process.kill()
        return self.process.wait(timeout=_CHILD_WAIT_TIMEOUT_S)

    def stop(self) -> int:
        self.stop_path.write_text("stop")
        return self.process.wait(timeout=_CHILD_WAIT_TIMEOUT_S)

    def stderr_text(self) -> str:
        if self.process.stderr is None:
            return ""
        return self.process.stderr.read()


@pytest.fixture
def start_lease_holder(tmp_path: Path) -> Iterator[Callable[[], _LeaseHolder]]:
    script = tmp_path / "lease_holder.py"
    script.write_text(_LEASE_HOLDER_SCRIPT)
    control = tmp_path / "control"
    control.mkdir()
    root = tmp_path / "root"
    started: list[_LeaseHolder] = []

    def start() -> _LeaseHolder:
        index = len(started)
        ready_path = control / f"{index}.ready"
        stop_path = control / f"{index}.stop"
        process = subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(root),
                SESSION_ID,
                str(ready_path),
                str(stop_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        holder = _LeaseHolder(process, root, ready_path, stop_path)
        started.append(holder)
        _await_ready(holder)
        return holder

    try:
        yield start
    finally:
        for holder in started:
            if holder.alive():
                holder.kill()


def _await_ready(holder: _LeaseHolder) -> None:
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if holder.ready():
            return
        if not holder.alive():
            break
        time.sleep(0.01)
    holder.kill()
    raise AssertionError(
        f"lease holder never published readiness: {holder.stderr_text()!r}"
    )


def _diagnostic(holder: _LeaseHolder) -> dict[str, object]:
    return json.loads(holder.lock_path.read_text())


def test_a_killed_owner_leaves_an_inert_lock_a_fresh_process_reacquires(
    start_lease_holder: Callable[[], _LeaseHolder],
) -> None:
    dead = start_lease_holder()
    initial = _diagnostic(dead)
    assert initial["lease_version"] == 1
    assert initial["session_id"] == SESSION_ID
    assert initial["process_id"] == dead.pid

    # Owner death without release: the child never runs its release path.
    assert dead.kill() == -signal.SIGKILL

    # The file survives and still names the dead owner, but its lock died with
    # the process — the leftover is inert, which is what makes recovery possible.
    assert dead.lock_path.exists()
    assert _diagnostic(dead)["process_id"] == dead.pid

    fresh = start_lease_holder()
    try:
        assert fresh.pid != dead.pid
        diagnostic = _diagnostic(fresh)
        assert diagnostic["process_id"] == fresh.pid
        assert diagnostic["session_id"] == SESSION_ID
    finally:
        assert fresh.stop() == 0

    # An actual release (not a stale file) unlinks the lock.
    assert not fresh.lock_path.exists()


def test_a_live_subprocess_owner_blocks_a_second_owner(
    start_lease_holder: Callable[[], _LeaseHolder],
) -> None:
    live = start_lease_holder()
    try:
        with pytest.raises(SessionBusyError):
            SessionLease(live.root, SESSION_ID).acquire()

        # The refused acquirer neither disturbed nor inherited the lease.
        assert live.alive()
        assert _diagnostic(live)["process_id"] == live.pid
    finally:
        assert live.stop() == 0

    # Once the live owner releases, the in-process owner acquires freely.
    SessionLease(live.root, SESSION_ID).acquire().release()
