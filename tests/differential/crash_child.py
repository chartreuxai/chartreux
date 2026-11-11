"""The marker-writing child the [6C] crash oracle kills (WP6 [6C]).

The child announces each durable step on stdout and then blocks, so the parent
can SIGKILL it at a known point: before the start marker, after it, or after
the terminal result record. The markers are the production [5C] records — the
crash oracle never fabricates them.

The egress guard is installed by the generated ``sitecustomize`` from the
spawn-env marker at interpreter startup; the child refuses to run when the
marker did not reach the transport, exactly like the replay child.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from chartreux.app_server._subagent_transcript import (
    ChildResultRecord,
    ChildStartMarker,
    ChildTranscriptStore,
    child_result_digest,
)
from tests.differential import egress

#: A fixed record timestamp keeps both topologies' digests identical.
FIXED_TIMESTAMP = "2026-11-10T00:00:00+00:00"
MODES = ("before_start", "after_start", "after_result")
_USAGE = "crash_child MODE AGENT SESSION RUN CHILD_DIR ROOT_SESSION_DIR"


def main() -> int:
    expected = os.environ.get(egress.EGRESS_BLOCK_ENV, "")
    installed = egress.installed_mode()
    if not expected or installed != expected:
        raise SystemExit(
            "crash child requires the egress marker to be applied before import "
            f"({egress.EGRESS_BLOCK_ENV}={expected!r}, installed={installed!r})"
        )
    if len(sys.argv) != 7:
        raise SystemExit(_USAGE)
    mode, agent_id, session_id, run_id, child_dir, root_session_dir = sys.argv[1:]
    store = ChildTranscriptStore(
        Path(child_dir), root_session_dir=Path(root_session_dir)
    )
    if mode == "before_start":
        print("ready", flush=True)
        sys.stdin.readline()
        return 0
    store.record_start(
        ChildStartMarker(
            agent_id=agent_id,
            session_id=session_id,
            run_id=run_id,
            started_at=FIXED_TIMESTAMP,
        )
    )
    print("started", flush=True)
    if mode == "after_start":
        sys.stdin.readline()
        return 0
    store.record_result(
        ChildResultRecord(
            agent_id=agent_id,
            session_id=session_id,
            run_id=run_id,
            result_id=f"result-{run_id}",
            digest=child_result_digest(
                agent_id=agent_id,
                session_id=session_id,
                run_id=run_id,
                outcome="completed",
                error=None,
            ),
            outcome="completed",
            completed_at=FIXED_TIMESTAMP,
        )
    )
    print("result", flush=True)
    sys.stdin.readline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
