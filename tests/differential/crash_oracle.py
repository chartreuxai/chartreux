"""WP6 [6C]: child crash containment and ``agents/*`` reconciliation oracle.

The oracle hosts the production [5C] recovery surface behind the [6A] channel
contract, so the same function runs with the process boundary removed and
inside the spawned, egress-blocked replay child:

* the child-owned durable markers are written by real processes
  (``tests.differential.crash_child``) and the process is SIGKILLed at the
  announced point — before the start marker, after it, or after the terminal
  result record;
* adjudication and reconciliation come from the production
  :class:`~chartreux.app_server._subagent_transcript.ChildTranscriptStore` and
  :class:`~chartreux.app_server._subagent_recovery.ChildTranscriptReconciler`;
* the receipt model is the reconciler's ``consumed_result_ids``: a completed
  result is ``unconsumed`` until the root records its receipt, and a second
  scan is idempotent — never a replay instruction.

The recorded frames carry only scrubbed paths and the fixed marker timestamps,
so both topologies normalize to the same digest.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from chartreux.app_server._subagent_recovery import (
    ChildTranscriptReconciler,
    ChildTranscriptReconciliation,
)
from chartreux.app_server._subagent_transcript import (
    ChildTranscriptStore,
    child_transcript_dir_name,
)
from tests.differential import egress
from tests.differential.replay import REPLAY_ROOT, Channel, ReplayError

FIXED_TIMESTAMP = "20261110_120000"
ROOT_SESSION_DIR = "root-session"
CRASH_CHILD_MODULE = "tests.differential.crash_child"
SCENARIOS = ("before_start", "after_start", "after_result")
_SYNC_TOKENS = {
    "before_start": "ready",
    "after_start": "started",
    "after_result": "result",
}
_SYNC_TIMEOUT_LINES = 64
_KILL_TIMEOUT = 10.0


class ChildCrashHost:
    """Spawn, kill, and reconcile real child processes on both topologies."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel
        self._run = f"run-{os.getpid()}"
        self._sequence = 0
        self._base: Path | None = None
        self._root_dir: Path | None = None
        self._session_id = "differential-root-session"
        self._adopted: dict[str, str] = {}

    # -- instruction stream ----------------------------------------------
    def serve(self) -> None:
        while (message := self._channel.receive()) is not None:
            kind = message.get("type")
            if kind == "init":
                self._initialize(message)
                self._observe("init", -1, {})
            elif kind == "op":
                index = message.get("index")
                op = message.get("op")
                if type(index) is not int or not isinstance(op, dict):
                    raise ReplayError(f"malformed crash op: {message!r}")
                self._observe(str(op.get("kind", "")), index, self._run_op(op))
            else:
                raise ReplayError(f"unknown crash oracle message: {message!r}")

    def _initialize(self, message: Mapping[str, Any]) -> None:
        base = message.get("base")
        if not isinstance(base, str) or not base:
            raise ReplayError(f"crash oracle init carries no base: {message!r}")
        self._base = Path(base)
        self._root_dir = self._base / ROOT_SESSION_DIR
        self._agents_dir().mkdir(parents=True, exist_ok=True)

    def _agents_dir(self) -> Path:
        if self._root_dir is None:
            raise ReplayError("crash oracle has no root session directory")
        return self._root_dir / "agents"

    def _base_dir(self) -> Path:
        if self._base is None:
            raise ReplayError("crash oracle has no trace base")
        return self._base

    def _scrub(self, text: str | None) -> str | None:
        if text is None:
            return None
        return text.replace(str(self._base_dir()), "<base>")

    def _run_op(self, op: Mapping[str, Any]) -> dict[str, object]:
        kind = str(op.get("kind", ""))
        if kind == "crash":
            scenario = str(op.get("scenario", ""))
            if scenario not in SCENARIOS:
                raise ReplayError(f"unknown crash scenario: {scenario!r}")
            return {"crash": self._crash(scenario)}
        if kind == "rescan":
            return {"rescan": self._rescan()}
        if kind == "restart":
            return {"restart": self._restart()}
        raise ReplayError(f"unknown crash oracle operation: {kind!r}")

    # -- scenarios --------------------------------------------------------
    def _crash(self, scenario: str) -> dict[str, object]:
        agent_id = f"worker-{scenario}"
        child_dir = self._child_dir(agent_id)
        child_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"run-{scenario}"
        process = self._spawn(child_dir, scenario, run_id)
        self._await_step(process, _SYNC_TOKENS[scenario])
        returncode = self._kill(process)
        reconciliation = self._reconcile(child_dir, agent_id)
        if reconciliation.result_id is not None:
            self._adopted[scenario] = reconciliation.result_id
        row = self._row(scenario, reconciliation)
        row["returncode"] = returncode
        return row

    def _rescan(self) -> dict[str, object]:
        reconciler = ChildTranscriptReconciler(self._root_dir_path())
        rows = reconciler.scan(consumed_result_ids=set(self._adopted.values()))
        return {
            "scans": [self._scan_row(reconciliation) for reconciliation in rows],
            "consumed_result_ids": sorted(self._adopted.values()),
        }

    def _restart(self) -> dict[str, object]:
        # A fresh run gets a fresh transcript directory and a fresh run id; the
        # crashed transcripts are never resumed or re-executed.
        agent_id = "worker-restart"
        child_dir = self._child_dir(agent_id)
        child_dir.mkdir(parents=True, exist_ok=True)
        process = self._spawn(child_dir, "after_result", "run-restart")
        self._await_step(process, _SYNC_TOKENS["after_result"])
        returncode = self._kill(process)
        reconciliation = self._reconcile(child_dir, agent_id)
        restart = self._row("restart", reconciliation)
        restart["returncode"] = returncode
        unchanged = [
            self._row(
                scenario,
                self._reconcile(
                    self._child_dir(f"worker-{scenario}"), f"worker-{scenario}"
                ),
            )
            for scenario in SCENARIOS
        ]
        return {"run": restart, "crashed_transcripts_unchanged": unchanged}

    def _child_dir(self, agent_id: str) -> Path:
        return self._agents_dir() / child_transcript_dir_name(
            agent_id, self._session_id, timestamp=FIXED_TIMESTAMP
        )

    def _root_dir_path(self) -> Path:
        if self._root_dir is None:
            raise ReplayError("crash oracle has no root session directory")
        return self._root_dir

    def _spawn(self, child_dir: Path, mode: str, run_id: str) -> subprocess.Popen[str]:
        environment = os.environ.copy()
        environment.update(egress.child_environment(egress.MODE_ALL))
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                CRASH_CHILD_MODULE,
                mode,
                child_dir.name.split("_", 1)[0],
                self._session_id,
                run_id,
                str(child_dir),
                str(self._root_dir_path()),
            ],
            cwd=str(REPLAY_ROOT),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @staticmethod
    def _await_step(process: subprocess.Popen[str], token: str) -> None:
        stdout = process.stdout
        if stdout is None:
            raise ReplayError("crash child did not expose its stdout pipe")
        for _ in range(_SYNC_TIMEOUT_LINES):
            line = stdout.readline()
            if not line:
                stderr = process.stderr.read() if process.stderr is not None else ""
                raise ReplayError(
                    f"crash child exited before {token!r}: {stderr.strip()}"
                )
            if line.strip() == token:
                return
        raise ReplayError(f"crash child never announced {token!r}")

    @staticmethod
    def _kill(process: subprocess.Popen[str]) -> int:
        # The child is blocked on stdin by design, so the kill is the crash.
        process.kill()
        try:
            process.wait(timeout=_KILL_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_KILL_TIMEOUT)
        return int(process.returncode or 0)

    def _reconcile(
        self, child_dir: Path, agent_id: str
    ) -> ChildTranscriptReconciliation:
        return ChildTranscriptReconciler(self._root_dir_path()).reconcile(
            child_dir, agent_id=agent_id
        )

    def _row(
        self, scenario: str, reconciliation: ChildTranscriptReconciliation
    ) -> dict[str, object]:
        child_dir = Path(reconciliation.child_dir)
        store = ChildTranscriptStore(child_dir, root_session_dir=self._root_dir_path())
        message = reconciliation.coordinator_message()
        params = message.get("params")
        coordinator = params if isinstance(params, Mapping) else {}
        return {
            "scenario": scenario,
            "outcome": reconciliation.outcome.value,
            "run_id": reconciliation.run_id,
            "result_id": reconciliation.result_id,
            "digest": reconciliation.digest,
            "unconsumed": reconciliation.unconsumed,
            "has_unconsumed_result": reconciliation.has_unconsumed_result,
            "denial": self._scrub(reconciliation.denial),
            "replay_instruction": False,
            "markers": {
                "start": store.start_marker_path.exists(),
                "result": store.result_path.exists(),
            },
            "transcript_dir": self._scrub(str(child_dir)),
            "coordinator": {
                "method": message.get("method"),
                "outcome": coordinator.get("outcome"),
                "unconsumed": coordinator.get("unconsumed"),
                "digest": coordinator.get("digest"),
            },
        }

    def _scan_row(
        self, reconciliation: ChildTranscriptReconciliation
    ) -> dict[str, object]:
        return {
            "outcome": reconciliation.outcome.value,
            "result_id": reconciliation.result_id,
            "unconsumed": reconciliation.unconsumed,
            "denial": self._scrub(reconciliation.denial),
            "transcript_dir": self._scrub(str(reconciliation.child_dir)),
        }

    # -- observation -------------------------------------------------------
    def _observe(self, op: str, index: int, payload: dict[str, object]) -> None:
        frame: dict[str, object] = {
            "type": "obs",
            "run": self._run,
            "sequence": self._sequence,
            "op": op,
            "op_index": index,
        }
        frame.update(payload)
        self._channel.send(frame)
        self._sequence += 1


def run_child_crash_loop(channel: Channel) -> None:
    """Component entry point shared by both topologies."""
    ChildCrashHost(channel).serve()


class ChildCrashDriver:
    """Drive one crash/reconciliation trace over any channel."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel

    def initialize(self, init: Mapping[str, Any]) -> dict[str, Any]:
        self._channel.send({"type": "init", **init})
        return self._next_obs(expect=-1)

    def run(self, ops: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        frames: list[dict[str, Any]] = []
        for index, op in enumerate(ops):
            self._channel.send({"type": "op", "index": index, "op": op})
            frames.append(self._next_obs(expect=index))
        return tuple(frames)

    def _next_obs(self, *, expect: int) -> dict[str, Any]:
        while True:
            frame = self._channel.receive()
            if frame is None:
                raise ReplayError("crash oracle closed mid-trace")
            if frame.get("type") == "obs" and frame.get("op_index") == expect:
                return frame
