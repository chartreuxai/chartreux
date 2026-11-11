"""WP6 [6D]: shutdown-drain and lease-release nodes for the differential gate.

The component hosts the production surfaces a shutdown path crosses at the WP7
boundary, so the same function runs with the process boundary removed and
inside the spawned, egress-blocked replay child:

``lease_*``
    :class:`~chartreux.core.session.session_lease.SessionLease` — acquire, the
    busy conflict of a second holder, release, re-acquire, and release on
    death: a SIGKILLed holder process leaves the on-disk lease file behind
    while the kernel drops its lock, so the next acquire must succeed.
``drain_*``
    :class:`~chartreux.app_server._sessions.SessionRuntimeRegistry` drain via
    the production :class:`~chartreux.app_server._child_runtime_lifecycle.ChildDrain`
    — a bounded drain reports DRAINED, an expired bound reports TIMED_OUT and
    never reads as a successful drain, and a second drain is idempotent.
``admission``
    The [5E] quiescence denial: a launch reserves while the registry is open
    and is refused typed once admission closed.
``shutdown_sequence``
    The ordered node the tracer takes at the shutdown boundary — acquire the
    lease, drain the children, confirm the lease outlived the drain, then
    release: the lease is released by the shutdown, not by the drain.

Clocks are injected, so no node waits on wall time and both topologies
normalize to the same digest. Paths are scrubbed to ``<base>``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import IO, Any

from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server.process_bootstrap import (
    ProcessTreeBudget,
    ProcessTreeClosingError,
    ProcessTreeShutdown,
)
from chartreux.app_server.subagent_ports import SubagentAdmissionClosedError
from chartreux.core.session.session_lease import SessionBusyError, SessionLease
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.differential import egress
from tests.differential.replay import REPLAY_ROOT, Channel, ReplayError

LEASE_ROOT_DIR = "lease-root"
LEASE_SESSION = "differential-lease"
LEASE_HOLDER_MODULE = "tests.differential.lease_holder"
DRAIN_TIMEOUT_SECONDS = 10.0
JUMPING_CLOCK_STEP = 100.0
_SYNC_TIMEOUT_LINES = 64
_HOLDER_TIMEOUT = 10.0


async def _noop_notify(*_args: object, **_kwargs: object) -> None:
    return None


async def _noop_callback(*_args: object, **_kwargs: object) -> None:
    return None


class _ManualClock:
    """The injected monotonic clock; a node advances it only by command."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _JumpClock:
    """Every read after the first advances by ``step`` (an expired drain bound)."""

    def __init__(self, step: float) -> None:
        self.step = step
        self.value = 0.0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls > 1:
            self.value += self.step
        return self.value


class _ControlledProcess:
    """A deterministic process double for the ordered-shutdown node."""

    stdin: IO[bytes] | None = None
    stdout: IO[bytes] | None = None

    def __init__(self, *, exits_on_terminate: bool, exits_on_kill: bool = True) -> None:
        self._alive = True
        self._exits_on_terminate = exits_on_terminate
        self._exits_on_kill = exits_on_kill
        self.signals: list[str] = []

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self.signals.append("terminate")
        if self._exits_on_terminate:
            self._alive = False

    def kill(self) -> None:
        self.signals.append("kill")
        if self._exits_on_kill:
            self._alive = False


class LifecycleHost:
    """The shutdown/lease component: one trace's ops over any channel."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel
        self._run = f"run-{os.getpid()}"
        self._sequence = 0
        self._base: Path | None = None
        self._lease: SessionLease | None = None
        self._guards = ExitStack()

    # -- instruction stream ----------------------------------------------
    async def serve(self) -> None:
        try:
            while (message := self._channel.receive()) is not None:
                kind = message.get("type")
                if kind == "init":
                    self._initialize(message)
                    self._observe("init", -1, {})
                elif kind == "op":
                    index = message.get("index")
                    op = message.get("op")
                    if type(index) is not int or not isinstance(op, dict):
                        raise ReplayError(f"malformed lifecycle op: {message!r}")
                    self._observe(
                        str(op.get("kind", "")), index, await self._run_op(op)
                    )
                else:
                    raise ReplayError(f"unknown lifecycle message: {message!r}")
        finally:
            self._guards.close()

    def _initialize(self, message: Mapping[str, Any]) -> None:
        base = message.get("base")
        if not isinstance(base, str) or not base:
            raise ReplayError(f"lifecycle init carries no base: {message!r}")
        self._base = Path(base)
        self._lease_root().mkdir(parents=True, exist_ok=True)

    def _base_dir(self) -> Path:
        if self._base is None:
            raise ReplayError("lifecycle host has no trace base")
        return self._base

    def _lease_root(self) -> Path:
        return self._base_dir() / LEASE_ROOT_DIR

    def _scrub(self, text: str) -> str:
        return text.replace(str(self._base_dir()), "<base>")

    # -- ops --------------------------------------------------------------
    async def _run_op(self, op: Mapping[str, Any]) -> dict[str, object]:
        kind = str(op.get("kind", ""))
        if kind == "lease_acquire":
            return self._lease_acquire()
        if kind == "lease_conflict":
            return self._lease_conflict()
        if kind == "lease_release":
            return self._lease_release()
        if kind == "lease_reacquire":
            return self._lease_reacquire()
        if kind == "lease_holder_crash":
            return self._lease_holder_crash()
        if kind == "drain":
            return await self._drain(kind, _ManualClock())
        if kind == "drain_timeout":
            return await self._drain(kind, _JumpClock(JUMPING_CLOCK_STEP))
        if kind == "drain_second":
            return await self._drain_second()
        if kind == "admission":
            return await self._admission()
        if kind == "shutdown_sequence":
            return await self._shutdown_sequence()
        if kind == "shutdown_order":
            return await self._shutdown_order(expired=bool(op.get("expired")))
        raise ReplayError(f"unknown lifecycle op: {kind!r}")

    # -- lease -------------------------------------------------------------
    def _lease_path(self) -> Path:
        return self._lease_root() / "active" / f"{LEASE_SESSION}.lock"

    def _lease_diagnostic(self) -> dict[str, object]:
        path = self._lease_path()
        if not path.exists():
            return {"exists": False}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            "exists": True,
            "path": self._scrub(str(path)),
            "lease_version": raw.get("lease_version"),
            "session_id": raw.get("session_id"),
            "process_id_is_int": isinstance(raw.get("process_id"), int),
            "acquired_at_present": isinstance(raw.get("acquired_at"), str),
        }

    def _lease_acquire(self) -> dict[str, object]:
        self._lease = SessionLease(self._lease_root(), LEASE_SESSION).acquire()
        return {"kind": "lease_acquire", "diagnostic": self._lease_diagnostic()}

    def _lease_conflict(self) -> dict[str, object]:
        try:
            second = SessionLease(self._lease_root(), LEASE_SESSION).acquire()
        except SessionBusyError as exc:
            return {
                "kind": "lease_conflict",
                "error": type(exc).__name__,
                "busy_session": exc.session_id,
            }
        second.release()
        return {"kind": "lease_conflict", "error": None, "busy_session": None}

    def _lease_release(self) -> dict[str, object]:
        lease = self._lease
        if lease is None:
            raise ReplayError("lease release without an acquired lease")
        lease.release()
        first = self._lease_diagnostic()
        # Release is idempotent: a second release on the same handle is a no-op
        # rather than an error, so a shutdown path can release twice.
        lease.release()
        return {
            "kind": "lease_release",
            "after_release": first,
            "idempotent_second_release": self._lease_diagnostic() == first,
        }

    def _lease_reacquire(self) -> dict[str, object]:
        self._lease = SessionLease(self._lease_root(), LEASE_SESSION).acquire()
        return {"kind": "lease_reacquire", "diagnostic": self._lease_diagnostic()}

    def _lease_holder_crash(self) -> dict[str, object]:
        holder = self._spawn_holder()
        try:
            self._await_step(holder, "held")
            held = self._lease_diagnostic()
            returncode = self._kill(holder)
            after_kill = self._lease_diagnostic()
            # The kernel released the dead holder's lock; the next acquire
            # must succeed and rewrite the stale diagnostic.
            reacquired = SessionLease(self._lease_root(), LEASE_SESSION).acquire()
            try:
                after_reacquire = self._lease_diagnostic()
                reacquired_ok = True
            finally:
                reacquired.release()
            return {
                "kind": "lease_holder_crash",
                "held": held,
                "holder_exit": returncode,
                "after_kill": after_kill,
                "reacquire": after_reacquire,
                "reacquired": reacquired_ok,
                "after_release": self._lease_diagnostic(),
            }
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait(timeout=_HOLDER_TIMEOUT)

    def _spawn_holder(self) -> subprocess.Popen[str]:
        environment = os.environ.copy()
        environment.update(egress.child_environment(egress.MODE_ALL))
        bootstrap = self._guards.enter_context(egress.bootstrap_directory())
        environment["PYTHONPATH"] = os.pathsep.join([
            str(bootstrap),
            str(REPLAY_ROOT),
            *([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else []),
        ])
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                LEASE_HOLDER_MODULE,
                str(self._lease_root()),
                LEASE_SESSION,
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
            raise ReplayError("lease holder did not expose its stdout pipe")
        for _ in range(_SYNC_TIMEOUT_LINES):
            line = stdout.readline()
            if not line:
                stderr = process.stderr.read() if process.stderr is not None else ""
                raise ReplayError(
                    f"lease holder exited before {token!r}: {stderr.strip()}"
                )
            if line.strip() == token:
                return
        raise ReplayError(f"lease holder never announced {token!r}")

    @staticmethod
    def _kill(process: subprocess.Popen[str]) -> int:
        process.kill()
        try:
            process.wait(timeout=_HOLDER_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_HOLDER_TIMEOUT)
        return int(process.returncode or 0)

    # -- drain -------------------------------------------------------------
    @staticmethod
    def _registry(clock: Any) -> SessionRuntimeRegistry:
        return SessionRuntimeRegistry(
            _noop_notify,
            _noop_callback,
            lambda _session_id: 0,
            clock=clock,
            shutdown_timeout=DRAIN_TIMEOUT_SECONDS,
        )

    async def _drain(self, kind: str, clock: Any) -> dict[str, object]:
        registry = self._registry(clock)
        report = await registry.close_children()
        return {
            "kind": kind,
            "status": report.status.value,
            "drained": report.drained_children,
            "terminated": report.terminated_children,
            "remaining": report.remaining_children,
            "elapsed": round(report.elapsed, 6),
            "complete": report.complete,
            "timed_out": report.timed_out,
            "admission_closed": bool(registry._admission_closed),
        }

    async def _drain_second(self) -> dict[str, object]:
        registry = self._registry(_ManualClock())
        first = await registry.close_children()
        second = await registry.close_children()
        return {
            "kind": "drain_second",
            "first": first.status.value,
            "second": second.status.value,
            "second_remaining": second.remaining_children,
            "admission_closed": bool(registry._admission_closed),
        }

    def _reserve(self, registry: SessionRuntimeRegistry) -> str:
        args = TaskArgs(task="shutdown node", identity="coder", background=True)
        context = InvokeContext(tool_call_id="shutdown-node", session_id=LEASE_SESSION)
        try:
            slot = registry._reserve_admission(args, context)
        except SubagentAdmissionClosedError:
            return "SubagentAdmissionClosedError"
        try:
            return f"reserved:{len(registry._active_work_slots)}"
        finally:
            registry._active_work_slots.discard(slot)

    async def _admission(self) -> dict[str, object]:
        registry = self._registry(_ManualClock())
        before = self._reserve(registry)
        await registry.close_children()
        after = self._reserve(registry)
        return {"kind": "admission", "before_drain": before, "after_drain": after}

    async def _shutdown_sequence(self) -> dict[str, object]:
        acquired = self._lease_acquire()
        drained = await self._drain("drain", _ManualClock())
        during_drain = self._lease_diagnostic()
        released = self._lease_release()
        return {
            "kind": "shutdown_sequence",
            "acquired": acquired["diagnostic"],
            "drain_status": drained["status"],
            "lease_during_drain": during_drain,
            "after_release": released["after_release"],
        }

    async def _shutdown_order(self, *, expired: bool) -> dict[str, object]:
        """The [7B] ordered node: admission → (drain) → terminate/kill → IPC → reap.

        The process doubles and the clock are deterministic, so both
        topologies normalize to the same frame: SIGTERM, then SIGKILL inside
        the grace window, then close-IPC, then reap. ``expired`` pins the
        typed-timeout half: a process that survives the expired bound leaves
        the tree admission closed, so no new tree starts before a reap.
        """
        clock = _ManualClock(0.0)
        budget = ProcessTreeBudget(max_processes=4, reserved_members=1)
        cooperative = _ControlledProcess(exits_on_terminate=True)
        stubborn = _ControlledProcess(
            exits_on_terminate=False, exits_on_kill=not expired
        )
        admission: list[str] = []

        async def spend(delay: float) -> None:
            clock.value += delay
            if admission:
                return
            try:
                slot = budget.reserve()
            except ProcessTreeClosingError:
                admission.append("denied")
            else:
                admission.append("admitted")
                budget.release(slot)

        tree = ProcessTreeShutdown(
            budget,
            processes=lambda: (cooperative, stubborn),
            timeout=0.001 if expired else DRAIN_TIMEOUT_SECONDS,
            grace=2.0,
            clock=clock,
            sleep=spend,
        )
        report = await tree.shutdown()
        return {
            "kind": "shutdown_order",
            "status": report.status.value,
            "phases": [phase.value for phase in report.phases],
            "terminated": report.terminated,
            "killed": report.killed,
            "reaped": report.reaped,
            "remaining": report.remaining,
            "complete": report.complete,
            "elapsed": round(report.elapsed, 6),
            "signals": [cooperative.signals, stubborn.signals],
            "admission_during": admission[0] if admission else "no-wait",
            "admission_after": "closed" if budget.shutting_down else "open",
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


def run_lifecycle_loop(channel: Channel) -> None:
    """Component entry point shared by both topologies."""
    asyncio.run(LifecycleHost(channel).serve())


class LifecycleDriver:
    """Drive one shutdown/lease trace over any channel."""

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
                raise ReplayError("lifecycle host closed mid-trace")
            if frame.get("type") == "obs" and frame.get("op_index") == expect:
                return frame
