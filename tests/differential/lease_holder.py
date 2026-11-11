"""The lease-holding child the [6D] shutdown nodes SIGKILL (WP6 [6D]).

The child acquires the real :class:`~chartreux.core.session.session_lease.SessionLease`,
announces it on stdout, and then blocks so the parent can kill it while it
still holds the lock. The kernel releases the lock when the process dies; the
on-disk lease file survives. That is exactly the lease-release-on-death case a
shutdown path has to keep working (WP7 [7B]/[7C]).

The egress guard is installed by the generated ``sitecustomize`` from the
spawn-env marker at interpreter startup; the child refuses to run when the
marker did not reach the transport, exactly like the replay child.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from chartreux.core.session.session_lease import SessionLease
from tests.differential import egress

_USAGE = "lease_holder LEASE_ROOT SESSION_ID"


def main() -> int:
    expected = os.environ.get(egress.EGRESS_BLOCK_ENV, "")
    installed = egress.installed_mode()
    if not expected or installed != expected:
        raise SystemExit(
            "lease holder requires the egress marker to be applied before import "
            f"({egress.EGRESS_BLOCK_ENV}={expected!r}, installed={installed!r})"
        )
    if len(sys.argv) != 3:
        raise SystemExit(_USAGE)
    lease = SessionLease(Path(sys.argv[1]), sys.argv[2]).acquire()
    print("held", flush=True)
    sys.stdin.readline()
    lease.release()
    print("released", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
