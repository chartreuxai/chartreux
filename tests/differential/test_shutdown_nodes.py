"""WP6 [6D]: shutdown-drain and lease-release nodes at the WP7 boundary.

Every node runs the same production shutdown surface twice — with the process
boundary removed and inside a spawned, egress-blocked child — and the two
normalized streams must be identical. What the nodes pin (design WP7 [7B],
tied to the [5E] drain):

* a lease is exclusive: a second holder is refused typed, not silently shared;
* release is idempotent and total: the on-disk lease disappears;
* a SIGKILLed holder releases its lock through the kernel while leaving the
  stale lease file, so the next acquire succeeds — the release-on-death case a
  shutdown path relies on;
* a bounded drain reports DRAINED, an expired bound reports TIMED_OUT and never
  reads as a successful drain, and a second drain is idempotent;
* after the drain closes admission, a launch is refused typed (quiescence);
* the ordered shutdown node: the lease outlives the drain and is released by
  the shutdown, not by the drain;
* the [7B] ordered tree-shutdown node: SIGTERM (grace then kill) → close IPC →
  reap, with the tree admission shut for the whole stop; an expired bound is
  typed and leaves the tree admission closed, so no new tree starts before a
  reap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.differential import ledger
from tests.differential.lifecycle import (
    JUMPING_CLOCK_STEP,
    LEASE_SESSION,
    LifecycleDriver,
)
from tests.differential.replay import (
    COMPONENT_LIFECYCLE,
    digest,
    in_process_component,
    normalize_frames,
    subprocess_component,
)

IN_PROCESS = "in_process"
SPAWNED = "spawned"
TOPOLOGIES = (IN_PROCESS, SPAWNED)
_NODE_TIMEOUT = 120.0


def _host(topology: str):
    if topology == IN_PROCESS:
        return in_process_component(
            component=COMPONENT_LIFECYCLE, timeout=_NODE_TIMEOUT
        )
    return subprocess_component(component=COMPONENT_LIFECYCLE, timeout=_NODE_TIMEOUT)


def _both(base: Path, ops: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Run one shutdown trace in both topologies and return the common stream."""
    runs: dict[str, tuple[dict[str, Any], ...]] = {}
    for topology in TOPOLOGIES:
        run_base = base / topology
        run_base.mkdir(parents=True, exist_ok=True)
        with _host(topology) as channel:
            driver = LifecycleDriver(channel)
            ready = driver.initialize({"base": str(run_base)})
            assert ready["op"] == "init"
            runs[topology] = driver.run(ops)
    normalized = {topology: normalize_frames(runs[topology]) for topology in TOPOLOGIES}
    # Each topology used its own base directory; the component scrubbed it.
    assert normalized[IN_PROCESS] == normalized[SPAWNED]
    assert digest(normalized[IN_PROCESS]) == digest(normalized[SPAWNED])
    ledger.record(ledger.FAMILY_SHUTDOWN, {"ops": ops}, normalized[IN_PROCESS])
    return normalized[IN_PROCESS]


@pytest.mark.timeout(300)
def test_lease_exclusivity_release_and_reacquire(tmp_path: Path) -> None:
    frames = _both(
        tmp_path,
        [
            {"kind": "lease_acquire"},
            {"kind": "lease_conflict"},
            {"kind": "lease_release"},
            {"kind": "lease_reacquire"},
            {"kind": "lease_release"},
        ],
    )

    acquired = frames[0]
    assert acquired["op"] == "lease_acquire"
    diagnostic = acquired["diagnostic"]
    assert diagnostic["exists"] is True
    assert diagnostic["lease_version"] == 1
    assert diagnostic["session_id"] == LEASE_SESSION
    assert diagnostic["process_id_is_int"] is True
    assert diagnostic["acquired_at_present"] is True
    assert diagnostic["path"] == f"<base>/lease-root/active/{LEASE_SESSION}.lock"

    conflict = frames[1]
    assert conflict["error"] == "SessionBusyError"
    assert conflict["busy_session"] == LEASE_SESSION

    released = frames[2]
    assert released["after_release"] == {"exists": False}
    assert released["idempotent_second_release"] is True

    reacquired = frames[3]
    assert reacquired["diagnostic"]["exists"] is True
    assert reacquired["diagnostic"]["session_id"] == LEASE_SESSION
    assert frames[4]["after_release"] == {"exists": False}


@pytest.mark.timeout(300)
def test_killed_holder_releases_its_lock_and_leaves_a_reusable_file(
    tmp_path: Path,
) -> None:
    frames = _both(tmp_path, [{"kind": "lease_holder_crash"}])
    crash = frames[0]

    assert crash["op"] == "lease_holder_crash"
    assert crash["held"]["exists"] is True
    assert crash["held"]["session_id"] == LEASE_SESSION
    # The injected death is real: the holder did not exit on its own.
    assert crash["holder_exit"] != 0
    # The on-disk lease survives the holder; its lock does not.
    assert crash["after_kill"]["exists"] is True
    assert crash["after_kill"]["session_id"] == LEASE_SESSION
    assert crash["reacquired"] is True
    assert crash["reacquire"]["exists"] is True
    assert crash["reacquire"]["session_id"] == LEASE_SESSION
    assert crash["after_release"] == {"exists": False}


@pytest.mark.timeout(300)
def test_drain_is_bounded_typed_and_idempotent(tmp_path: Path) -> None:
    frames = _both(
        tmp_path,
        [{"kind": "drain"}, {"kind": "drain_timeout"}, {"kind": "drain_second"}],
    )

    drained = frames[0]
    assert drained["status"] == "drained"
    assert drained["complete"] is True
    assert drained["timed_out"] is False
    assert drained["remaining"] == 0
    assert drained["admission_closed"] is True

    # The bound expired: TIMED_OUT never reads as a successful drain.
    timed_out = frames[1]
    assert timed_out["status"] == "timed_out"
    assert timed_out["timed_out"] is True
    assert timed_out["complete"] is False
    assert timed_out["elapsed"] == JUMPING_CLOCK_STEP * 2

    second = frames[2]
    assert second["first"] == "drained"
    assert second["second"] == "drained"
    assert second["second_remaining"] == 0
    assert second["admission_closed"] is True


@pytest.mark.timeout(300)
def test_quiescence_denial_and_the_ordered_shutdown_node(tmp_path: Path) -> None:
    frames = _both(tmp_path, [{"kind": "admission"}, {"kind": "shutdown_sequence"}])

    admission = frames[0]
    assert admission["before_drain"] == "reserved:1"
    assert admission["after_drain"] == "SubagentAdmissionClosedError"

    shutdown = frames[1]
    assert shutdown["kind"] == "shutdown_sequence"
    assert shutdown["acquired"]["exists"] is True
    assert shutdown["drain_status"] == "drained"
    # The drain does not release the session lease; the shutdown does.
    assert shutdown["lease_during_drain"]["exists"] is True
    assert shutdown["lease_during_drain"]["session_id"] == LEASE_SESSION
    assert shutdown["after_release"] == {"exists": False}


@pytest.mark.timeout(300)
def test_ordered_tree_shutdown_node(tmp_path: Path) -> None:
    frames = _both(tmp_path, [{"kind": "shutdown_order"}])
    shutdown = frames[0]

    assert shutdown["kind"] == "shutdown_order"
    assert shutdown["status"] == "complete"
    assert shutdown["complete"] is True
    assert shutdown["phases"] == ["terminate", "close_ipc", "reap"]
    # Both children are signalled; the one that ignores SIGTERM is killed.
    assert shutdown["terminated"] == 2
    assert shutdown["killed"] == 1
    assert shutdown["reaped"] == 2
    assert shutdown["remaining"] == 0
    assert shutdown["signals"] == [["terminate"], ["terminate", "kill"]]
    # Admission was already shut while the grace window was being spent.
    assert shutdown["admission_during"] == "denied"
    # The old tree was reaped inside the one bound, so a new tree may start.
    assert shutdown["admission_after"] == "open"


@pytest.mark.timeout(300)
def test_an_expired_tree_shutdown_never_reads_as_success(tmp_path: Path) -> None:
    frames = _both(tmp_path, [{"kind": "shutdown_order", "expired": True}])
    shutdown = frames[0]

    assert shutdown["status"] == "timed_out"
    assert shutdown["complete"] is False
    assert shutdown["reaped"] == 1
    assert shutdown["remaining"] == 1
    # The unreaped process keeps the tree admission closed: no new tree starts
    # before the old one is actually reaped.
    assert shutdown["admission_after"] == "closed"
