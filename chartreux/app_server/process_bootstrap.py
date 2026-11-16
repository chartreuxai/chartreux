"""Framed readiness handshake for spawned app-server processes.

The parent spawns a child and reads exactly one readiness frame from the
child's stdout before any JSON-RPC traffic flows. The frame is a single
newline-delimited JSON object, so it shares the wire format of the transport it
precedes but carries a key that no JSON-RPC envelope can: JSON-RPC requires a
``jsonrpc`` member, while the readiness frame carries ``chartreuxBootstrap``.

That single frame is also the whole handshake: the installing protocol version
must match, EOF or a foreign frame before readiness is a typed failure with no
retry negotiation, and the wait is ONE startup deadline on a monotonic clock
(:class:`ReadyDeadline`). Physical spawns pass through the tree budget
(:class:`ProcessTreeBudget`), so a coordinator cannot grow its process tree
past its hard cap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
import json
import math
import os
import subprocess
import time
from typing import Any

from chartreux.app_server.transport import (
    DEFAULT_TERMINATE_GRACE,
    PROCESS_POLL_INTERVAL,
    BinaryLineReader,
    BinaryLineWriter,
    ProcessPipeJsonRpcTransport,
    StoppableProcess,
    close_process_ipc,
    signal_kill,
    signal_terminate,
    terminate_process,
)
from chartreux.core.tmp_ownership import (
    SpawnLedgerBinding,
    ambient_binding,
    process_identity_tuple,
)

PROCESS_PROTOCOL_VERSION = 2
"""The process wire version both ends must share.

Version 2 is the bidirectional agent-wire extension: origin-prefixed request
ids and the closed per-direction dispatch tables. A pair that does not share a
version rejects at the ready frame instead of misreading frames.
"""

AGENT_WIRE_CAPABILITY_RECOVERY_EVIDENCE = "recovery-evidence"
"""Additive agent-wire capability announced in the ready frame.

A child that announces it serves the ``agent/recovery/evidence`` request and
attaches the typed terminal projection to its ``agent/turn/ended`` frames, so
the coordinator-local root-recovery gate can decide on owner-side evidence.
A peer that does not announce it is a version skew, answered typed with a
restart instruction instead of a silent fallback.
"""

AGENT_WIRE_CAPABILITY_LIVE_AGENT_INVENTORY = "live-agent-inventory"
"""Additive agent-wire capability announced in the ready frame.

A child that announces it serves the ``agent/agents/inventory/read`` request,
so the coordinator reads the live agent inventory as one typed, revised
snapshot answered by the process that owns the children -- never from a
coordinator cache. A peer that does not announce it is a version skew,
answered typed with a restart instruction instead of a stale fallback.
"""

AGENT_WIRE_CAPABILITY_ROOT_AGENTS_UPDATE = "root-agents-update"
"""Additive agent-wire capability announced in the ready frame.

A root agent that announces it emits the ``agent/agents/update`` id-less
notification at its child-inventory mutation moments, so the coordinator
relays them onto the client-facing ``agents/update`` and root-owned children
populate the agent browser. A peer that does not announce it never has the
sink bound, so the browser falls back to the live inventory read alone.
"""

AGENT_WIRE_CAPABILITY_ROOT_RUN_COMPLETIONS = "root-run-completions"
"""Additive agent-wire capability announced in the ready frame.

A root agent that announces it emits the ``agent/run/completed`` id-less
notification at each background child run's settle moment and serves the
``agent/runs/completed/read`` recovery ledger, so the coordinator can tell the
parent model a run finished and wake it. A peer that does not announce it never
has the completion sink bound and never wakes a parent from a snapshot it
cannot vouch for -- the ordinary tool response stays the only completion path.
"""

AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY = "live-child-authority"
"""Additive agent-wire capability announced in the ready frame.

A process that announces it takes part in live child authority: the owner
serves the typed ``agent/authority/check`` decision per invocation from its
CURRENT policy and emits ``agent/authority/changed`` when that policy moves,
and a child asks its immediate parent instead of deciding from a spawn-time
snapshot. The capability is MANDATORY on both ends of a process child: a peer
that does not announce it is a version skew, and child publication is refused
rather than silently downgraded to the static inherited policy (F11).
"""

ADVERTISED_CAPABILITIES = frozenset({
    AGENT_WIRE_CAPABILITY_RECOVERY_EVIDENCE,
    AGENT_WIRE_CAPABILITY_LIVE_AGENT_INVENTORY,
    AGENT_WIRE_CAPABILITY_ROOT_AGENTS_UPDATE,
    AGENT_WIRE_CAPABILITY_ROOT_RUN_COMPLETIONS,
    AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY,
})
"""The capability set this installation announces in its ready frame."""

BOOTSTRAP_ENV_VAR = "CHARTREUX_PROCESS_BOOTSTRAP"

_READY_KEY = "chartreuxBootstrap"
_READY_VALUE = "ready"

DEFAULT_SPAWN_READY_TIMEOUT = 15.0
"""The one startup deadline of a spawned process (plan.md numeric bounds)."""

DEFAULT_SHUTDOWN_TIMEOUT = 10.0
"""The ONE shutdown bound of a process tree (plan.md numeric bounds).

It covers the whole ordered shutdown: admission stop, in-flight interrupt,
save drain, terminate/kill escalation, IPC close, and reap. A shutdown that
runs past it reports a typed timeout and never a successful drain.
"""

PROCESS_TREE_FIXED_MEMBERS = 2
"""The coordinator and the root agent: the members every tree carries."""

DEFAULT_MAX_CHILD_PROCESSES = 16
"""Mirrors ``SubagentsConfig.max_running_subagents``'s default (design.md:448)."""

DEFAULT_MAX_PROCESS_TREE_SIZE = PROCESS_TREE_FIXED_MEMBERS + DEFAULT_MAX_CHILD_PROCESSES
"""The hard cap on live processes in one coordinator-owned tree."""

type MonotonicClock = Callable[[], float]

type DeadlineSleep = Callable[[float], Awaitable[None]]


async def _default_sleep(delay: float) -> None:
    await asyncio.sleep(delay)


class ProcessBootstrapError(RuntimeError):
    """Spawning failed: no child, no usable bootstrap state, or a denied spawn."""


class ReadyFrameMissingError(ProcessBootstrapError):
    """The child closed its pipe or sent something other than readiness first."""


class ProtocolVersionMismatchError(ProcessBootstrapError):
    """The child speaks a different process protocol than this installation."""

    def __init__(self, *, expected: int, received: int) -> None:
        super().__init__(
            "process protocol version mismatch: "
            f"expected {expected}, child announced {received}; "
            "restart both ends on the same chartreux version — no traffic was sent"
        )
        self.expected = expected
        self.received = received


class ReadyTimeoutError(ProcessBootstrapError):
    """The child announced no readiness within the startup deadline."""


class SpawnCapExceededError(ProcessBootstrapError):
    """The process tree is at its hard cap; the spawn is denied, not started."""

    def __init__(self, *, live: int, cap: int) -> None:
        self.live = live
        self.cap = cap
        super().__init__(
            f"Process spawn denied: {live} spawned processes are live "
            f"(cap {cap}); close a process before spawning another"
        )


class ProcessTreeClosingError(ProcessBootstrapError):
    """The tree is shutting down; no new process starts before its reap.

    A spawn attempted while the old tree is still being stopped is denied
    typed instead of racing the shutdown — the old tree is fully reaped
    before a new one may start.
    """


@dataclass(frozen=True, slots=True)
class ReadyFrame:
    protocol_version: int
    capabilities: frozenset[str] = frozenset()
    """The additive capabilities the child announced; empty for an older peer."""


@dataclass(frozen=True, slots=True)
class ReadyDeadline:
    """The ONE startup deadline of a spawned process, on a monotonic clock.

    ``clock`` reads time monotonically (``time.monotonic`` by default) and is
    read exactly once at construction, so a wall-clock step never extends or
    shortens a bound. Tests inject a fake clock plus a fake ``sleep`` to spend
    the deadline deterministically instead of waiting it out.
    """

    timeout: float
    clock: MonotonicClock = time.monotonic
    started_at: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeout", _validated_timeout(self.timeout))
        object.__setattr__(self, "started_at", self.clock())

    @property
    def expires_at(self) -> float:
        return self.started_at + self.timeout

    def remaining(self) -> float:
        """Seconds left before the deadline; zero or less once it expired."""
        return self.expires_at - self.clock()

    def expired(self) -> bool:
        return self.remaining() <= 0

    async def expire(self, sleep: DeadlineSleep) -> None:
        """Return once this deadline is spent on its clock."""
        remaining = self.remaining()
        if remaining <= 0:
            return
        await sleep(remaining)


@dataclass(frozen=True, slots=True, eq=False)
class ProcessSlot:
    """One reserved place in a process tree's spawn budget."""


class ProcessTreeBudget:
    """The hard cap on live processes in one coordinator-owned tree (WP7 [7A]).

    The tree's fixed members (the coordinator, the root agent) are always
    present, so ``reserved_members`` of the cap never counts as spawnable and
    the remainder is the concurrent-child budget (design.md:448). Reservation
    is synchronous: between the occupancy check and the insertion there is no
    ``await``, so concurrent admissions observe each other's slots. An
    exhausted cap raises the typed :class:`SpawnCapExceededError` before any
    process exists — never a silent wait. A crashed process whose handle was
    never retired is pruned at the next reservation instead of holding the
    budget forever.
    """

    def __init__(
        self,
        max_processes: int = DEFAULT_MAX_PROCESS_TREE_SIZE,
        *,
        reserved_members: int = PROCESS_TREE_FIXED_MEMBERS,
    ) -> None:
        if isinstance(max_processes, bool) or not isinstance(max_processes, int):
            raise ValueError("max_processes must be a positive integer")
        if isinstance(reserved_members, bool) or not isinstance(reserved_members, int):
            raise ValueError("reserved_members must be a non-negative integer")
        if max_processes < 1:
            raise ValueError("max_processes must be a positive integer")
        if reserved_members < 0 or reserved_members >= max_processes:
            raise ValueError("reserved_members must leave a spawnable slot")
        self._max_processes = max_processes
        self._reserved_members = reserved_members
        self._live: dict[ProcessSlot, subprocess.Popen[bytes] | None] = {}
        # WP7 [7B]: while the tree is shutting down no new process starts,
        # and the gate stays shut until the old tree is reaped.
        self._shutting_down = False

    @property
    def max_processes(self) -> int:
        return self._max_processes

    @property
    def capacity(self) -> int:
        """How many spawned processes fit beside the tree's fixed members."""
        return self._max_processes - self._reserved_members

    @property
    def live_processes(self) -> int:
        return len(self._live)

    @property
    def shutting_down(self) -> bool:
        """Whether the tree is stopping; spawns are denied until it is reaped."""
        return self._shutting_down

    def begin_shutdown(self) -> None:
        """Stop admission for this tree: every later reserve is denied typed."""
        self._shutting_down = True

    def end_shutdown(self) -> None:
        """Reopen admission; only the owner calls this after the tree was reaped."""
        self._shutting_down = False

    def live_handles(self) -> tuple[subprocess.Popen[bytes], ...]:
        """The live process handles of this tree, pruned of already-exited ones."""
        self.prune()
        return tuple(process for process in self._live.values() if process is not None)

    def reserve(self) -> ProcessSlot:
        """Take one spawn slot, or raise the typed cap/closing denial."""
        if self._shutting_down:
            raise ProcessTreeClosingError(
                "Process spawn denied: the process tree is shutting down; "
                "no new process starts before the old tree is reaped"
            )
        self.prune()
        if len(self._live) >= self.capacity:
            raise SpawnCapExceededError(live=len(self._live), cap=self.capacity)
        slot = ProcessSlot()
        self._live[slot] = None
        return slot

    def attach(self, slot: ProcessSlot, process: subprocess.Popen[bytes]) -> None:
        """Bind a reserved slot to the process it admitted."""
        if slot in self._live:
            self._live[slot] = process

    def release(self, slot: ProcessSlot) -> None:
        """Return a slot to the budget; releasing twice is a no-op."""
        self._live.pop(slot, None)

    def prune(self) -> None:
        """Forget slots whose process already exited without retiring."""
        for slot, process in list(self._live.items()):
            if process is not None and process.poll() is not None:
                del self._live[slot]


@dataclass(frozen=True, slots=True)
class SpawnedProcess:
    process: subprocess.Popen[bytes]
    transport: ProcessPipeJsonRpcTransport
    ready: ReadyFrame
    budget: ProcessTreeBudget | None = None
    slot: ProcessSlot | None = None

    def retire(self) -> None:
        """Release this process's tree-budget slot; retiring twice is a no-op."""
        if self.budget is not None and self.slot is not None:
            self.budget.release(self.slot)


class ShutdownStatus(StrEnum):
    """The typed disposition of one ordered tree shutdown (WP7 [7B])."""

    COMPLETE = "complete"
    TIMED_OUT = "timed_out"


class ShutdownPhase(StrEnum):
    """The ordered phases of a tree shutdown (design.md R2-W1)."""

    STOP_ADMISSION = "stop_admission"
    INTERRUPT = "interrupt"
    DRAIN_SAVES = "drain_saves"
    TERMINATE = "terminate"
    CLOSE_IPC = "close_ipc"
    REAP = "reap"


@dataclass(frozen=True, slots=True)
class ProcessTreeShutdownReport:
    """What the process-level half of a tree shutdown actually achieved.

    ``remaining`` counts processes the shutdown could not reap inside its
    bound. A report with any remaining process (or one that expired) is
    ``TIMED_OUT``: it never reads as a successful stop, and the tree budget
    stays closed so no new tree starts before the old one is reaped.
    """

    status: ShutdownStatus
    phases: tuple[ShutdownPhase, ...] = ()
    terminated: int = 0
    killed: int = 0
    ipc_closed: int = 0
    reaped: int = 0
    remaining: int = 0
    elapsed: float = 0.0

    @property
    def timed_out(self) -> bool:
        """Whether the shutdown bound expired before every process was reaped."""
        return self.status is ShutdownStatus.TIMED_OUT

    @property
    def complete(self) -> bool:
        """Whether every process was reaped inside the bound."""
        return (
            self.status is ShutdownStatus.COMPLETE
            and self.remaining == 0
            and ShutdownPhase.REAP in self.phases
        )


class ProcessTreeShutdown:
    """The ordered terminate/kill → close-IPC → reap half of a tree shutdown.

    One bound (``DEFAULT_SHUTDOWN_TIMEOUT``) covers the whole stop: admission
    is closed on the tree budget first so no new process starts, every live
    handle is signalled (SIGTERM inside ``grace``, then SIGKILL), the pipes
    close, and only then is the tree reaped. The clock and sleeper are
    injectable, so the bound is spent deterministically in tests instead of
    waited out, and an expired bound reports ``TIMED_OUT`` with the processes
    it could not reap rather than a clean stop.
    """

    def __init__(
        self,
        budget: ProcessTreeBudget | None = None,
        *,
        processes: Callable[[], Sequence[StoppableProcess]] | None = None,
        timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
        grace: float = DEFAULT_TERMINATE_GRACE,
        clock: MonotonicClock = time.monotonic,
        sleep: DeadlineSleep = _default_sleep,
    ) -> None:
        self._budget = budget
        self._processes = processes
        self._timeout = _validated_timeout(timeout)
        self._grace = _validated_timeout(grace)
        self._clock = clock
        self._sleep = sleep

    async def shutdown(
        self, *, timeout: float | None = None
    ) -> ProcessTreeShutdownReport:
        """Run the ordered stop under one bound; never unbounded."""
        started = self._clock()
        bound = self._timeout if timeout is None else max(0.0, timeout)
        deadline = started + bound
        if self._budget is not None:
            # Stop admission before the first signal: a spawn arriving now is
            # denied typed instead of starting beside the dying tree.
            self._budget.begin_shutdown()
        handles = self._handles()
        terminated = 0
        for handle in handles:
            if signal_terminate(handle):
                terminated += 1
        await self._await_exit(handles, min(self._clock() + self._grace, deadline))
        killed = 0
        for handle in handles:
            if handle.poll() is None and signal_kill(handle):
                killed += 1
        await self._await_exit(handles, deadline)
        ipc_closed = sum(1 for handle in handles if close_process_ipc(handle))
        reaped = sum(1 for handle in handles if handle.poll() is not None)
        remaining = len(handles) - reaped
        if self._budget is not None:
            self._budget.prune()
            if remaining == 0:
                # The old tree is fully reaped: a new one may start.
                self._budget.end_shutdown()
        status = ShutdownStatus.COMPLETE if remaining == 0 else ShutdownStatus.TIMED_OUT
        return ProcessTreeShutdownReport(
            status=status,
            phases=(
                ShutdownPhase.TERMINATE,
                ShutdownPhase.CLOSE_IPC,
                ShutdownPhase.REAP,
            ),
            terminated=terminated,
            killed=killed,
            ipc_closed=ipc_closed,
            reaped=reaped,
            remaining=remaining,
            elapsed=self._clock() - started,
        )

    def _handles(self) -> tuple[StoppableProcess, ...]:
        if self._processes is not None:
            return tuple(self._processes())
        if self._budget is not None:
            return self._budget.live_handles()
        return ()

    async def _await_exit(
        self, handles: Sequence[StoppableProcess], until: float
    ) -> None:
        pending = list(handles)
        while True:
            pending = [handle for handle in pending if handle.poll() is None]
            if not pending:
                return
            remaining = until - self._clock()
            if remaining <= 0:
                return
            await self._sleep(min(PROCESS_POLL_INTERVAL, remaining))


def encode_ready_frame(
    protocol_version: int = PROCESS_PROTOCOL_VERSION,
    *,
    capabilities: object = ADVERTISED_CAPABILITIES,
) -> bytes:
    """Encode the single framed announcement the child writes before traffic.

    The frame also advertises the additive capabilities this installation
    carries, so the parent can tell a same-version peer that lacks one from a
    peer that has it.
    """
    if isinstance(protocol_version, bool) or not isinstance(protocol_version, int):
        raise ValueError("protocol_version must be a positive integer")
    if protocol_version < 1:
        raise ValueError("protocol_version must be a positive integer")
    frame = {
        _READY_KEY: _READY_VALUE,
        "protocolVersion": protocol_version,
        "capabilities": sorted(_validated_capabilities(capabilities)),
    }
    return json.dumps(frame, separators=(",", ":")).encode() + b"\n"


def write_ready_frame(
    writer: BinaryLineWriter,
    *,
    protocol_version: int = PROCESS_PROTOCOL_VERSION,
    capabilities: object = ADVERTISED_CAPABILITIES,
) -> None:
    """Announce readiness on the child's pipe before any JSON-RPC traffic."""
    writer.write(encode_ready_frame(protocol_version, capabilities=capabilities))
    writer.flush()


def parse_ready_frame(
    raw: bytes | str, *, expected_protocol_version: int = PROCESS_PROTOCOL_VERSION
) -> ReadyFrame:
    """Validate a child's first frame against this installation's version.

    A frame that carries no ``capabilities`` member is an older same-version
    peer and parses with an empty set; a malformed member fails typed.
    """
    try:
        value: Any = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReadyFrameMissingError(
            "child sent a malformed frame before readiness"
        ) from exc
    if not isinstance(value, dict) or value.get(_READY_KEY) != _READY_VALUE:
        raise ReadyFrameMissingError("child did not announce readiness before traffic")
    version = value.get("protocolVersion")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ReadyFrameMissingError("child sent a malformed readiness frame")
    if version != expected_protocol_version:
        raise ProtocolVersionMismatchError(
            expected=expected_protocol_version, received=version
        )
    announced = value.get("capabilities")
    if announced is None:
        capabilities: frozenset[str] = frozenset()
    else:
        try:
            capabilities = _validated_capabilities(announced)
        except ValueError as exc:
            raise ReadyFrameMissingError(
                "child sent a malformed readiness frame"
            ) from exc
    return ReadyFrame(protocol_version=version, capabilities=capabilities)


def bootstrap_requested(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this process was spawned to announce readiness on its pipe."""
    source = os.environ if environ is None else environ
    return source.get(BOOTSTRAP_ENV_VAR) == "1"


async def spawn_process(
    command: Sequence[str],
    *,
    expected_protocol_version: int = PROCESS_PROTOCOL_VERSION,
    ready_timeout: float = DEFAULT_SPAWN_READY_TIMEOUT,
    clock: MonotonicClock = time.monotonic,
    sleep: DeadlineSleep = _default_sleep,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    budget: ProcessTreeBudget | None = None,
    ledger: SpawnLedgerBinding | None = None,
) -> SpawnedProcess:
    """Spawn ``command`` and wait for its single readiness frame.

    The tree budget admits the spawn before any process exists, so an
    exhausted cap is the typed denial rather than a started-then-killed
    child. The slot returns to the budget when the handshake fails and again
    when the returned handle retires.

    With a ``ledger`` binding (else the ambient one) the worker process is
    registered RECORD-BEFORE-SPAWN in the invoking context's allocation
    ledger: the worker is NOT setsid'd — it inherits the session's process
    group and is registered by its spawner like any other spawn. The intent
    is appended before the fork (a typed refusal propagates pre-fork when the
    ledger cannot record it), a failed Popen retracts it, and a successful
    spawn completes it with the child's full identity tuple captured
    synchronously in the spawn frame.
    """
    if not command:
        raise ValueError("command must not be empty")
    deadline = ReadyDeadline(ready_timeout, clock=clock)
    slot = budget.reserve() if budget is not None else None
    child_env = dict(os.environ if env is None else env)
    child_env[BOOTSTRAP_ENV_VAR] = "1"
    binding = ledger if ledger is not None else ambient_binding()
    handle = None
    if binding is not None:
        handle = binding.begin_spawn(is_leader_group=False)
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=child_env,
            cwd=cwd,
        )
    except BaseException:
        if binding is not None and handle is not None:
            binding.retract_spawn(handle)
        _release_slot(budget, slot)
        raise
    if binding is not None and handle is not None:
        # In-frame capture: no yield between the Popen and this append, so a
        # fast-exiting child is never left uncompleted.
        binding.complete_spawn(handle, process.pid, process_identity_tuple(process.pid))
    if budget is not None and slot is not None:
        budget.attach(slot, process)
    reader = process.stdout
    writer = process.stdin
    if reader is None or writer is None:
        await asyncio.to_thread(terminate_process, process)
        _release_slot(budget, slot)
        raise ProcessBootstrapError("spawned child has no standard pipe")
    try:
        ready = await _await_ready(
            reader, expected_protocol_version, deadline, sleep=sleep
        )
    except BaseException:
        await asyncio.to_thread(
            terminate_process, process, grace=DEFAULT_TERMINATE_GRACE
        )
        _release_slot(budget, slot)
        raise
    return SpawnedProcess(
        process=process,
        transport=ProcessPipeJsonRpcTransport(reader, writer, process=process),
        ready=ready,
        budget=budget,
        slot=slot,
    )


async def _await_ready(
    reader: BinaryLineReader,
    expected_protocol_version: int,
    deadline: ReadyDeadline,
    *,
    sleep: DeadlineSleep = _default_sleep,
) -> ReadyFrame:
    """Read the child's one ready frame, bounded by the startup deadline."""
    read = asyncio.create_task(asyncio.to_thread(reader.readline))
    expiry = asyncio.create_task(deadline.expire(sleep))
    try:
        done, _ = await asyncio.wait(
            {read, expiry}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        expiry.cancel()
        await asyncio.gather(expiry, return_exceptions=True)
    if read not in done:
        read.cancel()
        read.add_done_callback(_discard_outcome)
        raise ReadyTimeoutError(
            f"child announced no readiness within {deadline.timeout}s"
        )
    raw = read.result()
    if not raw:
        raise ReadyFrameMissingError(
            "child closed its pipe before announcing readiness"
        )
    return parse_ready_frame(raw, expected_protocol_version=expected_protocol_version)


def _discard_outcome(task: asyncio.Task[bytes]) -> None:
    with suppress(BaseException):
        task.result()


def _release_slot(budget: ProcessTreeBudget | None, slot: ProcessSlot | None) -> None:
    if budget is not None and slot is not None:
        budget.release(slot)


def _validated_timeout(value: float) -> float:
    try:
        valid = math.isfinite(value) and value > 0
    except TypeError as exc:
        raise ValueError("ready_timeout must be a finite positive number") from exc
    if not valid:
        raise ValueError("ready_timeout must be a finite positive number")
    return value


def _validated_capabilities(capabilities: object) -> frozenset[str]:
    """Normalize one announced capability set, refusing anything malformed."""
    if isinstance(capabilities, str | Mapping) or not isinstance(
        capabilities, Collection
    ):
        raise ValueError("capabilities must be a collection of names")
    names = frozenset(capabilities)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("capabilities must be non-empty strings")
    return names
