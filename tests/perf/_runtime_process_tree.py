"""X8c: observation-only process-tree CPU/RSS observation read from ``/proc``.

Plan §4 lists process-tree CPU/RSS, per-thread CPU and task counts among the
observation-only quantities: they are fully reported together with their
reasons and never gated — no threshold, no pass/fail, no invented value where
``/proc`` does not answer. Everything here observes; nothing instruments a
production path, and nothing here runs on the measured workload's control flow.

Two honesty rules shape the API. A platform without ``/proc`` reports
``unsupported`` with its reason instead of raising or fabricating zeros, and a
process that is gone (or unreadable, or whose ``stat`` line does not parse) is
recorded as such: an aggregate only ever sums values that were actually read,
and every unread member carries the reason it is missing.

Sampling twice over the same tree yields a :class:`CpuDelta` set. A pid whose
``starttime`` changed between the samples is reported as a reused pid, never
subtracted across the boundary; a pid that disappeared is reported as missing;
a pid that appeared is reported as added. Deltas are reported raw — a negative
CPU delta is not clamped here, because nothing in this module is a gated slope.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from time import perf_counter_ns

PROC_ROOT = Path("/proc")

#: Schema id of the observation record this module emits.
PROCESS_TREE_SCHEMA = "chartreux.runtime_process_tree_observation.v1"

#: Plan §4: these quantities are reported, never gated.
OBSERVATION_ONLY_REASON = (
    "process-tree CPU/RSS, thread CPU and task counts are observation-only "
    "(plan §4): reported with their reasons, never gated"
)

OUTCOME_OBSERVED = "observed"
OUTCOME_PARTIAL = "partial"
OUTCOME_MISSING = "missing"
OUTCOME_UNSUPPORTED = "unsupported"

REASON_UNSUPPORTED_PLATFORM = "unsupported_platform"
REASON_PROC_UNAVAILABLE = "proc_unavailable"
REASON_PROC_UNREADABLE = "proc_unreadable"
REASON_PROCESS_MISSING = "process_missing"
REASON_STAT_UNREADABLE = "stat_unreadable"
REASON_STAT_MALFORMED = "stat_malformed"
REASON_RSS_UNAVAILABLE = "rss_unavailable"
REASON_THREADS_UNREADABLE = "threads_unreadable"
REASON_PID_REUSED = "pid_reused"

_BYTES_PER_KIB = 1024


def _clock_ticks_per_second() -> int:
    try:
        ticks = int(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, OSError, ValueError):
        return 100
    return ticks if ticks > 0 else 100


_TICKS_PER_SECOND = _clock_ticks_per_second()


def _ticks_to_seconds(ticks: int) -> float:
    return ticks / _TICKS_PER_SECOND


@dataclass(frozen=True, slots=True)
class SupportStatus:
    """Whether this host exposes the ``/proc`` surface the observer reads."""

    supported: bool
    reason: str | None
    platform: str
    proc_root: Path

    def record(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "reason": self.reason,
            "platform": self.platform,
            "proc_root": str(self.proc_root),
        }


def process_tree_support(
    *, platform: str | None = None, proc_root: Path | None = None
) -> SupportStatus:
    """Report whether process-tree observation is available on this host.

    Off Linux the answer is ``unsupported`` with ``unsupported_platform``; a
    platform that names Linux but carries no readable ``/proc`` surface answers
    ``unsupported`` with ``proc_unavailable``. Both are recorded reasons, never
    exceptions: the plan requires the unsupported branch to be honest, not loud.
    """
    current = sys.platform if platform is None else platform
    root = PROC_ROOT if proc_root is None else Path(proc_root)
    if current != "linux":
        return SupportStatus(False, REASON_UNSUPPORTED_PLATFORM, current, root)
    if not (root / "self" / "stat").is_file():
        return SupportStatus(False, REASON_PROC_UNAVAILABLE, current, root)
    return SupportStatus(True, None, current, root)


@dataclass(frozen=True, slots=True)
class StatFields:
    """The ``/proc/<pid>/stat`` fields this observer decodes."""

    pid: int
    ppid: int
    state: str
    utime_ticks: int
    stime_ticks: int
    cutime_ticks: int
    cstime_ticks: int
    num_threads: int
    starttime_ticks: int


def parse_stat(text: str) -> StatFields | None:
    """Parse one ``/proc/<pid>/stat`` line, or return None when it is malformed.

    ``comm`` is the only field that may contain spaces and parentheses, so the
    head is cut at its opening ``(`` and the tail at the last ``)``; the tail's
    whitespace-separated fields then start at state (field 3 of the kernel's
    documented layout).
    """
    head, separator, remainder = text.partition(" (")
    if not separator:
        return None
    _, closing, tail = remainder.rpartition(")")
    if not closing:
        return None
    fields = tail.split()
    if len(fields) < 20:
        return None
    try:
        return StatFields(
            pid=int(head.strip()),
            ppid=int(fields[1]),
            state=fields[0],
            utime_ticks=int(fields[11]),
            stime_ticks=int(fields[12]),
            cutime_ticks=int(fields[13]),
            cstime_ticks=int(fields[14]),
            num_threads=int(fields[17]),
            starttime_ticks=int(fields[19]),
        )
    except ValueError:
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def read_stat(pid: int, proc_root: Path) -> tuple[StatFields | None, str | None]:
    """Read one process's stat line, with the reason a read failed."""
    path = proc_root / str(pid) / "stat"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None, REASON_PROCESS_MISSING
    except OSError:
        return None, REASON_STAT_UNREADABLE
    fields = parse_stat(text)
    if fields is None:
        return None, REASON_STAT_MALFORMED
    return fields, None


def read_rss_bytes(pid: int, proc_root: Path) -> tuple[int | None, str | None]:
    """Read one process's ``VmRSS`` in bytes, or the reason it is unavailable.

    A zombie process answers ``/proc`` without a ``VmRSS`` line; that is an
    unavailable value with a reason, never a fabricated zero.
    """
    path = proc_root / str(pid) / "status"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None, REASON_PROCESS_MISSING
    except OSError:
        return None, REASON_RSS_UNAVAILABLE
    for line in text.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None, REASON_RSS_UNAVAILABLE
        try:
            return int(parts[1]) * _BYTES_PER_KIB, None
        except ValueError:
            return None, REASON_RSS_UNAVAILABLE
    return None, REASON_RSS_UNAVAILABLE


@dataclass(frozen=True, slots=True)
class ThreadCpu:
    """One thread's cumulative CPU time."""

    tid: int
    name: str
    cpu_s: float


@dataclass(frozen=True, slots=True)
class ProcessSample:
    """One process's CPU/RSS/thread observation at one instant."""

    pid: int
    ppid: int | None
    state: str | None
    starttime_ticks: int | None
    cpu_user_s: float | None
    cpu_system_s: float | None
    cpu_reaped_children_s: float | None
    rss_bytes: int | None
    rss_reason: str | None
    num_threads: int | None
    task_count: int | None
    threads: tuple[ThreadCpu, ...]
    threads_reason: str | None

    @property
    def cpu_self_s(self) -> float | None:
        """This process's own CPU (user + system), excluding reaped children."""
        if self.cpu_user_s is None or self.cpu_system_s is None:
            return None
        return self.cpu_user_s + self.cpu_system_s

    @property
    def reason(self) -> str | None:
        """The first reason any part of this member could not be read."""
        return self.threads_reason or self.rss_reason

    def record(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "ppid": self.ppid,
            "state": self.state,
            "starttime_ticks": self.starttime_ticks,
            "cpu_user_s": _rounded(self.cpu_user_s),
            "cpu_system_s": _rounded(self.cpu_system_s),
            "cpu_self_s": _rounded(self.cpu_self_s),
            "cpu_reaped_children_s": _rounded(self.cpu_reaped_children_s),
            "rss_bytes": self.rss_bytes,
            "rss_reason": self.rss_reason,
            "num_threads": self.num_threads,
            "task_count": self.task_count,
            "threads_reason": self.threads_reason,
            "threads": [
                {
                    "tid": thread.tid,
                    "name": thread.name,
                    "cpu_s": _rounded(thread.cpu_s),
                }
                for thread in self.threads
            ],
        }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def read_threads(
    pid: int, proc_root: Path
) -> tuple[tuple[ThreadCpu, ...], int | None, str | None]:
    """Read a process's task directory: per-thread CPU and the task count."""
    task_dir = proc_root / str(pid) / "task"
    try:
        with os.scandir(task_dir) as entries:
            tids = sorted(int(entry.name) for entry in entries if entry.name.isdigit())
    except FileNotFoundError:
        return (), None, REASON_PROCESS_MISSING
    except OSError:
        return (), None, REASON_THREADS_UNREADABLE

    threads: list[ThreadCpu] = []
    reason: str | None = None
    for tid in tids:
        fields, failure = read_stat(tid, task_dir)
        if fields is None:
            if failure != REASON_PROCESS_MISSING and reason is None:
                reason = failure
            continue
        name_text = _read_text(task_dir / str(tid) / "comm")
        name = name_text.strip() if name_text and name_text.strip() else f"tid-{tid}"
        threads.append(
            ThreadCpu(
                tid=tid,
                name=name,
                cpu_s=_ticks_to_seconds(fields.utime_ticks + fields.stime_ticks),
            )
        )
    if reason is None and len(threads) != len(tids):
        reason = REASON_THREADS_UNREADABLE
    return tuple(threads), len(tids), reason


@dataclass(frozen=True, slots=True)
class MissingMember:
    """A process that could not contribute a sample, and why."""

    pid: int
    reason: str

    def record(self) -> dict[str, object]:
        return {"pid": self.pid, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class TreeSample:
    """One observation of a whole process tree at one instant."""

    requested_pid: int
    captured_ns: int
    outcome: str
    reason: str | None
    proc_root: Path
    processes: tuple[ProcessSample, ...]
    missing: tuple[MissingMember, ...]

    @property
    def pids(self) -> tuple[int, ...]:
        return tuple(process.pid for process in self.processes)

    @property
    def process_count(self) -> int:
        return len(self.processes)

    @property
    def task_count(self) -> int | None:
        return _sum_int(process.task_count for process in self.processes)

    @property
    def thread_count(self) -> int:
        return sum(len(process.threads) for process in self.processes)

    @property
    def rss_bytes(self) -> int | None:
        return _sum_int(process.rss_bytes for process in self.processes)

    @property
    def cpu_self_s(self) -> float | None:
        return _sum_float(process.cpu_self_s for process in self.processes)

    @property
    def cpu_reaped_children_s(self) -> float | None:
        return _sum_float(process.cpu_reaped_children_s for process in self.processes)

    @property
    def cpu_total_s(self) -> float | None:
        """Self CPU plus reaped-descendant CPU; never double-counted."""
        own = self.cpu_self_s
        reaped = self.cpu_reaped_children_s
        if own is None and reaped is None:
            return None
        return (own or 0.0) + (reaped or 0.0)

    @property
    def reasons(self) -> tuple[str, ...]:
        collected = {member.reason for member in self.missing}
        collected.update(
            process.reason for process in self.processes if process.reason is not None
        )
        if self.reason is not None:
            collected.add(self.reason)
        return tuple(sorted(collected))

    def record(self) -> dict[str, object]:
        return {
            "requested_pid": self.requested_pid,
            "captured_ns": self.captured_ns,
            "outcome": self.outcome,
            "reason": self.reason,
            "proc_root": str(self.proc_root),
            "process_count": self.process_count,
            "task_count": self.task_count,
            "thread_count": self.thread_count,
            "rss_bytes": self.rss_bytes,
            "cpu_self_s": _rounded(self.cpu_self_s),
            "cpu_reaped_children_s": _rounded(self.cpu_reaped_children_s),
            "cpu_total_s": _rounded(self.cpu_total_s),
            "reasons": list(self.reasons),
            "processes": [process.record() for process in self.processes],
            "missing": [member.record() for member in self.missing],
            "observation_only": True,
            "gated": False,
        }


def _sum_int(values: Iterable[int | None]) -> int | None:
    total = 0
    seen = False
    for value in values:
        if value is None:
            continue
        total += value
        seen = True
    return total if seen else None


def _sum_float(values: Iterable[float | None]) -> float | None:
    total = 0.0
    seen = False
    for value in values:
        if value is None:
            continue
        total += value
        seen = True
    return total if seen else None


def _tree_pids(root_pid: int, proc_root: Path) -> tuple[int, ...]:
    """Return the root pid followed by its descendants, breadth-first.

    Membership is derived from the kernel's parent links (``ppid`` in each
    ``stat`` line), not from ``task/*/children``, so it does not depend on an
    optional kernel feature. A pid listed but already gone is skipped.
    """
    with os.scandir(proc_root) as entries:
        named = [entry.name for entry in entries]
    present: set[int] = set()
    children: dict[int, list[int]] = {}
    for name in named:
        if not name.isdigit():
            continue
        member_pid = int(name)
        present.add(member_pid)
        fields, _ = read_stat(member_pid, proc_root)
        if fields is None:
            continue
        children.setdefault(fields.ppid, []).append(member_pid)
    if root_pid not in present:
        return ()
    ordered = [root_pid]
    queue: deque[int] = deque((root_pid,))
    while queue:
        parent = queue.popleft()
        for child in sorted(children.get(parent, ())):
            if child in ordered:
                continue
            ordered.append(child)
            queue.append(child)
    return tuple(ordered)


def _observe_process(
    pid: int, proc_root: Path
) -> tuple[ProcessSample | None, str | None]:
    fields, failure = read_stat(pid, proc_root)
    if fields is None:
        return None, failure or REASON_STAT_UNREADABLE
    rss_bytes, rss_reason = read_rss_bytes(pid, proc_root)
    threads, task_count, threads_reason = read_threads(pid, proc_root)
    return (
        ProcessSample(
            pid=fields.pid,
            ppid=fields.ppid,
            state=fields.state,
            starttime_ticks=fields.starttime_ticks,
            cpu_user_s=_ticks_to_seconds(fields.utime_ticks),
            cpu_system_s=_ticks_to_seconds(fields.stime_ticks),
            cpu_reaped_children_s=_ticks_to_seconds(
                fields.cutime_ticks + fields.cstime_ticks
            ),
            rss_bytes=rss_bytes,
            rss_reason=rss_reason,
            num_threads=fields.num_threads,
            task_count=task_count,
            threads=threads,
            threads_reason=threads_reason,
        ),
        None,
    )


def observe_process_tree(
    pid: int | None = None,
    *,
    proc_root: Path | None = None,
    platform: str | None = None,
) -> TreeSample:
    """Take one observation of ``pid``'s process tree (default: this process).

    The result never raises for an unavailable platform, a vanished process or
    an unparseable line: each is recorded with its outcome and reason.
    """
    target = os.getpid() if pid is None else pid
    root = PROC_ROOT if proc_root is None else Path(proc_root)
    support = process_tree_support(platform=platform, proc_root=root)
    if not support.supported:
        return TreeSample(
            requested_pid=target,
            captured_ns=perf_counter_ns(),
            outcome=OUTCOME_UNSUPPORTED,
            reason=support.reason,
            proc_root=root,
            processes=(),
            missing=(),
        )

    try:
        pids = _tree_pids(target, root)
    except OSError:
        return TreeSample(
            requested_pid=target,
            captured_ns=perf_counter_ns(),
            outcome=OUTCOME_UNSUPPORTED,
            reason=REASON_PROC_UNREADABLE,
            proc_root=root,
            processes=(),
            missing=(),
        )
    if not pids:
        return TreeSample(
            requested_pid=target,
            captured_ns=perf_counter_ns(),
            outcome=OUTCOME_MISSING,
            reason=REASON_PROCESS_MISSING,
            proc_root=root,
            processes=(),
            missing=(MissingMember(target, REASON_PROCESS_MISSING),),
        )

    processes: list[ProcessSample] = []
    missing: list[MissingMember] = []
    for member in pids:
        sample, failure = _observe_process(member, root)
        if sample is None:
            missing.append(MissingMember(member, failure or REASON_STAT_UNREADABLE))
            continue
        processes.append(sample)

    if any(member.pid == target for member in missing):
        outcome, reason = OUTCOME_MISSING, REASON_PROCESS_MISSING
    elif missing or any(process.reason is not None for process in processes):
        outcome, reason = OUTCOME_PARTIAL, None
    else:
        outcome, reason = OUTCOME_OBSERVED, None
    return TreeSample(
        requested_pid=target,
        captured_ns=perf_counter_ns(),
        outcome=outcome,
        reason=reason,
        proc_root=root,
        processes=tuple(processes),
        missing=tuple(missing),
    )


@dataclass(frozen=True, slots=True)
class CpuDelta:
    """One process's change between two tree samples (observation-only)."""

    pid: int
    starttime_ticks: int
    elapsed_s: float
    cpu_s: float | None
    cpu_fraction: float | None
    rss_bytes_delta: int | None

    def record(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "starttime_ticks": self.starttime_ticks,
            "elapsed_s": _rounded(self.elapsed_s),
            "cpu_s": _rounded(self.cpu_s),
            "cpu_fraction": _rounded(self.cpu_fraction),
            "rss_bytes_delta": self.rss_bytes_delta,
        }


@dataclass(frozen=True, slots=True)
class TreeDelta:
    """The change of a whole process tree between two samples."""

    root_pid: int
    elapsed_s: float
    processes: tuple[CpuDelta, ...]
    missing: tuple[MissingMember, ...]
    reused: tuple[MissingMember, ...]
    added: tuple[int, ...]

    @property
    def cpu_s(self) -> float | None:
        return _sum_float(delta.cpu_s for delta in self.processes)

    @property
    def rss_bytes_delta(self) -> int | None:
        return _sum_int(delta.rss_bytes_delta for delta in self.processes)

    def record(self) -> dict[str, object]:
        return {
            "root_pid": self.root_pid,
            "elapsed_s": _rounded(self.elapsed_s),
            "cpu_s": _rounded(self.cpu_s),
            "rss_bytes_delta": self.rss_bytes_delta,
            "processes": [delta.record() for delta in self.processes],
            "missing": [member.record() for member in self.missing],
            "reused": [member.record() for member in self.reused],
            "added": list(self.added),
            "observation_only": True,
            "gated": False,
        }


def _difference(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def _difference_int(after: int | None, before: int | None) -> int | None:
    if after is None or before is None:
        return None
    return after - before


def sample_delta(previous: TreeSample, current: TreeSample) -> TreeDelta:
    """Difference two observations of one tree, keeping pid identity honest.

    A pid present in both samples with a different ``starttime`` is a reused
    pid: it is reported as such and never subtracted, so a recycled pid cannot
    fabricate CPU or RSS change. A pid absent from the later sample is reported
    missing; a pid only present in the later sample is reported added. CPU
    fractions may exceed 1.0 for a multithreaded process — this is a raw
    observation, not a gate.
    """
    elapsed_s = max(0.0, (current.captured_ns - previous.captured_ns) / 1_000_000_000)
    remaining = {process.pid: process for process in current.processes}
    deltas: list[CpuDelta] = []
    missing: list[MissingMember] = []
    reused: list[MissingMember] = []
    for before in previous.processes:
        after = remaining.pop(before.pid, None)
        if after is None:
            missing.append(MissingMember(before.pid, REASON_PROCESS_MISSING))
            continue
        if before.starttime_ticks != after.starttime_ticks:
            reused.append(MissingMember(before.pid, REASON_PID_REUSED))
            continue
        cpu_s = _difference(after.cpu_self_s, before.cpu_self_s)
        deltas.append(
            CpuDelta(
                pid=before.pid,
                starttime_ticks=before.starttime_ticks or 0,
                elapsed_s=elapsed_s,
                cpu_s=cpu_s,
                cpu_fraction=(
                    None if cpu_s is None or elapsed_s <= 0 else cpu_s / elapsed_s
                ),
                rss_bytes_delta=_difference_int(after.rss_bytes, before.rss_bytes),
            )
        )
    return TreeDelta(
        root_pid=previous.requested_pid,
        elapsed_s=elapsed_s,
        processes=tuple(deltas),
        missing=tuple(missing),
        reused=tuple(reused),
        added=tuple(sorted(remaining)),
    )


@dataclass(slots=True)
class ProcessTreeObserver:
    """Collects repeated observations of one process tree (observation-only)."""

    pid: int | None = None
    proc_root: Path = PROC_ROOT
    samples: list[TreeSample] = field(default_factory=list)

    def sample(self) -> TreeSample:
        """Take one observation and keep it in the series."""
        observed = observe_process_tree(self.pid, proc_root=self.proc_root)
        self.samples.append(observed)
        return observed

    def deltas(self) -> tuple[TreeDelta, ...]:
        """The deltas between consecutive samples of the series."""
        return tuple(
            sample_delta(earlier, later)
            for earlier, later in zip(self.samples, self.samples[1:], strict=False)
        )

    def record(self) -> dict[str, object]:
        """Emit the series as one observation record (never a threshold row)."""
        support = process_tree_support(proc_root=self.proc_root)
        first = self.samples[0] if self.samples else None
        last = self.samples[-1] if self.samples else None
        outcomes: dict[str, int] = {}
        reasons: set[str] = set()
        for observed in self.samples:
            outcomes[observed.outcome] = outcomes.get(observed.outcome, 0) + 1
            reasons.update(observed.reasons)
        rss_series = [
            observed.rss_bytes
            for observed in self.samples
            if observed.rss_bytes is not None
        ]
        cpu_series = [
            observed.cpu_self_s
            for observed in self.samples
            if observed.cpu_self_s is not None
        ]
        return {
            "schema": PROCESS_TREE_SCHEMA,
            "observation_only": True,
            "gated": False,
            "observation_only_reason": OBSERVATION_ONLY_REASON,
            "support": support.record(),
            "requested_pid": first.requested_pid if first else self.pid,
            "sample_count": len(self.samples),
            "outcomes": dict(sorted(outcomes.items())),
            "reasons": sorted(reasons),
            "rss_bytes": {
                "first": first.rss_bytes if first else None,
                "last": last.rss_bytes if last else None,
                "max": max(rss_series) if rss_series else None,
            },
            "cpu_self_s": {
                "first": _rounded(first.cpu_self_s) if first else None,
                "last": _rounded(last.cpu_self_s) if last else None,
                "max": _rounded(max(cpu_series)) if cpu_series else None,
            },
            "task_count": {
                "first": first.task_count if first else None,
                "last": last.task_count if last else None,
            },
            "samples": [observed.record() for observed in self.samples],
            "deltas": [delta.record() for delta in self.deltas()],
        }


__all__ = [
    "OBSERVATION_ONLY_REASON",
    "OUTCOME_MISSING",
    "OUTCOME_OBSERVED",
    "OUTCOME_PARTIAL",
    "OUTCOME_UNSUPPORTED",
    "PROCESS_TREE_SCHEMA",
    "PROC_ROOT",
    "REASON_PID_REUSED",
    "REASON_PROCESS_MISSING",
    "REASON_PROC_UNAVAILABLE",
    "REASON_PROC_UNREADABLE",
    "REASON_RSS_UNAVAILABLE",
    "REASON_STAT_MALFORMED",
    "REASON_STAT_UNREADABLE",
    "REASON_THREADS_UNREADABLE",
    "REASON_UNSUPPORTED_PLATFORM",
    "CpuDelta",
    "MissingMember",
    "ProcessSample",
    "ProcessTreeObserver",
    "StatFields",
    "SupportStatus",
    "ThreadCpu",
    "TreeDelta",
    "TreeSample",
    "observe_process_tree",
    "parse_stat",
    "process_tree_support",
    "read_rss_bytes",
    "read_stat",
    "read_threads",
    "sample_delta",
]
