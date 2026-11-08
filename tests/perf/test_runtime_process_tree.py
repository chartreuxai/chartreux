"""X8c contract node: observation-only process-tree CPU/RSS observation.

The node holds the plan's §4 boundary for the process-tree quantity: it is
OBSERVED, fully reported with its reasons, and never gated. What it asserts is
the decoding contract on surfaces whose numbers are known — a synthetic
``/proc`` tree with pinned ticks, RSS and thread values — and the honesty
contract on surfaces that are not: a platform without ``/proc`` reports
``unsupported`` with its reason instead of fabricating zeros, a reaped child is
reported missing with its reason, and a recycled pid is never subtracted across
a sample boundary.

The synthetic-child phase spawns one tiny child under the scenario egress
guard (plan §1: fail-closed egress wherever anything is spawned). The child
installs the same non-loopback socket block for itself, probes it, and leaves
an evidence record proving the block is live and that it carries no credential,
the redirected config/scratch homes, and disabled bytecode writing. It holds
itself alive until the parent releases it, so the observation of a live tree and
the observation of a missing one are both deterministic rather than raced.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import perf_counter

import pytest

from tests.perf._runtime_metrics import repository_root
from tests.perf._runtime_process_tree import (
    OBSERVATION_ONLY_REASON,
    OUTCOME_MISSING,
    OUTCOME_OBSERVED,
    OUTCOME_PARTIAL,
    OUTCOME_UNSUPPORTED,
    PROC_ROOT,
    PROCESS_TREE_SCHEMA,
    REASON_PROC_UNAVAILABLE,
    REASON_PROCESS_MISSING,
    REASON_RSS_UNAVAILABLE,
    REASON_UNSUPPORTED_PLATFORM,
    MissingMember,
    ProcessTreeObserver,
    observe_process_tree,
    parse_stat,
    process_tree_support,
    read_stat,
    sample_delta,
)
from tests.perf._runtime_scenario import (
    CONFIG_HOME_ENV,
    PROVIDER_CREDENTIAL_NAMES,
    SCRATCHPAD_ENV,
    EgressGuard,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="process-tree observation reads /proc, which is Linux-only",
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_CONTRACT_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

#: TEST-NET-1: a non-loopback destination no probe may ever reach.
_EGRESS_PROBE = "192.0.2.1"

#: The marker the spawned child installs for itself; it mirrors the scenario
#: guard's wording so both blocks report a non-loopback refusal identically.
_CHILD_EGRESS_MARKER = "provider network blocked by performance scenario egress guard"
_CHILD_EVIDENCE_FILENAME = "process-tree-child-egress.json"
_CHILD_RELEASE_FILENAME = "process-tree-child-release"
_CHILD_WAIT_S = 2.0
_CHILD_TICKS_PER_SECOND = int(os.sysconf("SC_CLK_TCK"))


def _stat_line(
    *,
    pid: int,
    ppid: int,
    utime: int,
    stime: int,
    starttime: int,
    cutime: int = 0,
    cstime: int = 0,
    threads: int = 1,
    state: str = "S",
    comm: str = "synthetic",
) -> str:
    """Render one ``/proc/<pid>/stat`` line in the kernel's documented layout."""
    tail = [
        state,
        str(ppid),
        str(pid),
        str(pid),
        "0",
        "-1",
        "4194304",
        "0",
        "0",
        "0",
        "0",
        str(utime),
        str(stime),
        str(cutime),
        str(cstime),
        "20",
        "0",
        str(threads),
        "0",
        str(starttime),
    ]
    return f"{pid} ({comm}) {' '.join(tail)}\n"


def _write_thread(
    directory: Path, *, tid: int, name: str, utime: int, stime: int
) -> None:
    task = directory / "task" / str(tid)
    task.mkdir(parents=True)
    (task / "stat").write_text(
        _stat_line(
            pid=tid, ppid=int(directory.name), utime=utime, stime=stime, starttime=tid
        ),
        encoding="utf-8",
    )
    (task / "comm").write_text(f"{name}\n", encoding="utf-8")


def _write_fake_proc(root: Path) -> Path:
    """A synthetic ``/proc`` surface carrying pinned, known numbers.

    pid 4242 (parent) holds 100/50 user/system ticks, 2048 KiB RSS and two
    threads; pid 4243 (its child) holds 7/3 ticks plus 2/1 ticks of reaped
    children, 512 KiB RSS and one thread. pid 9999 is unrelated and must not
    appear in 4242's tree.
    """
    (root / "self").mkdir(parents=True)
    (root / "self" / "stat").write_text(
        _stat_line(pid=4242, ppid=1, utime=0, stime=0, starttime=1), encoding="utf-8"
    )

    parent = root / "4242"
    parent.mkdir()
    (parent / "stat").write_text(
        _stat_line(pid=4242, ppid=1, utime=100, stime=50, starttime=5000, threads=2),
        encoding="utf-8",
    )
    (parent / "status").write_text(
        "Name:\tsynthetic-parent\nVmRSS:\t    2048 kB\n", encoding="utf-8"
    )
    _write_thread(parent, tid=4242, name="synthetic-main", utime=60, stime=30)
    _write_thread(parent, tid=4342, name="synthetic-worker", utime=40, stime=20)

    child = root / "4243"
    child.mkdir()
    (child / "stat").write_text(
        _stat_line(
            pid=4243, ppid=4242, utime=7, stime=3, cutime=2, cstime=1, starttime=5100
        ),
        encoding="utf-8",
    )
    (child / "status").write_text(
        "Name:\tsynthetic-child\nVmRSS:\t     512 kB\n", encoding="utf-8"
    )
    _write_thread(child, tid=4243, name="synthetic-child", utime=7, stime=3)

    unrelated = root / "9999"
    unrelated.mkdir()
    (unrelated / "stat").write_text(
        _stat_line(pid=9999, ppid=1, utime=5, stime=5, starttime=7000), encoding="utf-8"
    )
    return root


def _assert_support_reporting(tmp_path: Path) -> None:
    """Off-Linux and /proc-less hosts report unsupported + reason, not zeros."""
    support = process_tree_support()
    assert (support.supported, support.reason) == (True, None)

    off_platform = process_tree_support(platform="darwin")
    assert (off_platform.supported, off_platform.reason) == (
        False,
        REASON_UNSUPPORTED_PLATFORM,
    )
    absent_proc = process_tree_support(proc_root=tmp_path / "no-such-proc")
    assert (absent_proc.supported, absent_proc.reason) == (
        False,
        REASON_PROC_UNAVAILABLE,
    )

    for sample, reason in (
        (observe_process_tree(platform="darwin"), REASON_UNSUPPORTED_PLATFORM),
        (
            observe_process_tree(proc_root=tmp_path / "no-such-proc"),
            REASON_PROC_UNAVAILABLE,
        ),
    ):
        assert sample.outcome == OUTCOME_UNSUPPORTED
        assert sample.reason == reason
        record = sample.record()
        assert record["processes"] == []
        assert record["missing"] == []
        assert record["rss_bytes"] is None
        assert record["cpu_total_s"] is None
        assert record["task_count"] is None
        assert record["thread_count"] == 0
        assert record["reasons"] == [reason]
        assert record["gated"] is False
        assert record["observation_only"] is True


def _assert_stat_decoding() -> None:
    """The stat parser reads the documented layout and rejects garbage."""
    real = (PROC_ROOT / "self" / "stat").read_text(encoding="utf-8", errors="replace")
    fields = parse_stat(real)
    assert fields is not None
    assert fields.pid == os.getpid()

    assert parse_stat("") is None
    assert parse_stat("4242 (unterminated 1 2 3") is None
    assert parse_stat("4242 (short) S 1 2") is None

    # comm is the only field that may carry spaces and parentheses.
    weird = parse_stat(
        _stat_line(
            pid=4242, ppid=77, utime=3, stime=4, starttime=99, comm="a (weird) name"
        )
    )
    assert weird is not None
    assert (weird.pid, weird.ppid) == (4242, 77)
    assert (weird.utime_ticks, weird.stime_ticks) == (3, 4)
    assert weird.starttime_ticks == 99


def _assert_fake_proc_tree(tmp_path: Path) -> None:
    """Known /proc numbers decode to the expected tree quantities."""
    root = _write_fake_proc(tmp_path / "fake-proc")
    ticks = _CHILD_TICKS_PER_SECOND

    sample = observe_process_tree(4242, proc_root=root)
    assert sample.outcome == OUTCOME_OBSERVED, sample.record()
    assert sample.pids == (4242, 4243)
    assert sample.missing == ()
    assert sample.reasons == ()
    assert 9999 not in sample.pids
    assert sample.process_count == 2
    assert sample.task_count == 3
    assert sample.thread_count == 3
    assert sample.rss_bytes == (2048 + 512) * 1024
    assert sample.cpu_self_s == pytest.approx((100 + 50 + 7 + 3) / ticks)
    assert sample.cpu_reaped_children_s == pytest.approx(3 / ticks)
    assert sample.cpu_total_s == pytest.approx(163 / ticks)

    parent, child = sample.processes
    assert (parent.pid, parent.ppid, parent.state) == (4242, 1, "S")
    assert parent.cpu_user_s == pytest.approx(100 / ticks)
    assert parent.cpu_system_s == pytest.approx(50 / ticks)
    assert parent.cpu_reaped_children_s == pytest.approx(0.0)
    assert parent.rss_bytes == 2048 * 1024
    assert parent.num_threads == parent.task_count == 2
    assert [(thread.name, thread.cpu_s) for thread in parent.threads] == [
        ("synthetic-main", pytest.approx(90 / ticks)),
        ("synthetic-worker", pytest.approx(60 / ticks)),
    ]
    assert (child.pid, child.ppid) == (4243, 4242)
    assert child.cpu_self_s == pytest.approx(10 / ticks)
    assert child.cpu_reaped_children_s == pytest.approx(3 / ticks)
    assert child.rss_bytes == 512 * 1024
    assert child.task_count == 1

    record = sample.record()
    assert record["gated"] is False
    assert record["observation_only"] is True
    assert record["outcome"] == OUTCOME_OBSERVED
    assert json.loads(json.dumps(record)) == record

    # A member that answers without VmRSS (a zombie does) is partial, not zero.
    (root / "4243" / "status").write_text(
        "Name:\tsynthetic-child\nState:\tZ (zombie)\n", encoding="utf-8"
    )
    partial = observe_process_tree(4242, proc_root=root)
    assert partial.outcome == OUTCOME_PARTIAL
    assert partial.reasons == (REASON_RSS_UNAVAILABLE,)
    assert partial.processes[1].rss_reason == REASON_RSS_UNAVAILABLE
    assert partial.rss_bytes == 2048 * 1024
    assert partial.cpu_total_s == pytest.approx(163 / ticks)


def _assert_own_tree() -> None:
    """This process's own tree is observed with real values, never gated."""
    own = observe_process_tree()
    assert own.requested_pid == os.getpid()
    assert own.pids[0] == os.getpid()
    assert own.outcome in {OUTCOME_OBSERVED, OUTCOME_PARTIAL}
    assert own.rss_bytes is not None
    assert own.rss_bytes > 0
    assert own.cpu_total_s is not None
    assert own.cpu_total_s >= 0
    assert own.task_count is not None
    assert own.task_count >= 1
    assert own.thread_count >= 1

    root_process = own.processes[0]
    assert root_process.pid == os.getpid()
    assert os.getpid() in {thread.tid for thread in root_process.threads}
    assert root_process.num_threads == root_process.task_count
    assert root_process.reason is None


def _assert_delta_semantics() -> None:
    """Series records stay observation-only; pid identity is never crossed."""
    observer = ProcessTreeObserver()
    observer.sample()
    sum(range(200_000))
    observer.sample()

    deltas = observer.deltas()
    assert len(deltas) == 1
    first_delta = deltas[0]
    assert first_delta.root_pid == os.getpid()
    assert first_delta.elapsed_s >= 0
    assert first_delta.cpu_s is None or first_delta.cpu_s >= 0

    record = observer.record()
    assert record["schema"] == PROCESS_TREE_SCHEMA
    assert record["sample_count"] == 2
    assert record["requested_pid"] == os.getpid()
    assert record["gated"] is False
    assert record["observation_only"] is True
    assert record["observation_only_reason"] == OBSERVATION_ONLY_REASON
    assert record["deltas"] == [delta.record() for delta in deltas]
    assert record["reasons"] == []
    assert "threshold" not in json.dumps(record)

    # A pid is identified by (pid, starttime): a recycled pid is never crossed.
    earlier = observe_process_tree()
    retimed = replace(
        earlier,
        processes=tuple(
            replace(process, starttime_ticks=(process.starttime_ticks or 0) + 1)
            for process in earlier.processes
        ),
    )
    reused = sample_delta(earlier, retimed)
    assert reused.processes == ()
    assert reused.missing == ()
    assert {member.pid for member in reused.reused} == set(earlier.pids)

    # A vanished member is reported missing; a new one is reported added.
    vanished = sample_delta(earlier, replace(earlier, processes=()))
    assert vanished.processes == ()
    assert {member.pid for member in vanished.missing} == set(earlier.pids)
    assert vanished.added == ()

    grown = sample_delta(
        earlier,
        replace(
            earlier,
            processes=(
                *earlier.processes,
                replace(earlier.processes[0], pid=os.getpid() + 1),
            ),
        ),
    )
    assert grown.added == (os.getpid() + 1,)


def _probe_non_loopback() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.connect((_EGRESS_PROBE, 80))
    finally:
        probe.close()


def _install_guard(tmp_path: Path) -> EgressGuard:
    """Fail-closed egress on the spawning path too (plan §1)."""
    guard = EgressGuard(tmp_path / "egress")
    guard.install()
    guard.assert_closed()
    assert Path(os.environ[CONFIG_HOME_ENV]) == guard.config_home
    assert not [name for name in PROVIDER_CREDENTIAL_NAMES if name in os.environ]
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    with pytest.raises(OSError, match="blocked"):
        _probe_non_loopback()
    return guard


def _child_program() -> str:
    """The synthetic child: install its own block, attest, then hold still."""
    lines = (
        "import json, os, socket, time",
        "",
        f"MARKER = {_CHILD_EGRESS_MARKER!r}",
        "SCRATCH = os.environ['CHARTREUX_PERF_SCRATCHPAD']",
        f"EVIDENCE = os.path.join(SCRATCH, {_CHILD_EVIDENCE_FILENAME!r})",
        f"RELEASE = os.path.join(SCRATCH, {_CHILD_RELEASE_FILENAME!r})",
        "",
        "_connect = socket.socket.connect",
        "",
        "def _blocked(sock, address, *args, **kwargs):",
        "    host = address[0] if isinstance(address, tuple) and address else address",
        "    if host in ('127.0.0.1', '::1', 'localhost'):",
        "        return _connect(sock, address, *args, **kwargs)",
        "    raise OSError(MARKER)",
        "",
        "socket.socket.connect = _blocked",
        "probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
        "try:",
        f"    probe.connect(({_EGRESS_PROBE!r}, 80))",
        "except OSError as error:",
        "    blocked = MARKER in str(error)",
        "else:",
        "    blocked = False",
        "finally:",
        "    probe.close()",
        "",
        "evidence = {",
        "    'schema': 'chartreux.process_tree_child_egress.v1',",
        "    'pid': os.getpid(),",
        "    'blocked': blocked,",
        "    'marker': MARKER,",
        "    'environment': sorted(os.environ),",
        "    'bytecode_disabled': os.environ.get('PYTHONDONTWRITEBYTECODE') == '1',",
        "    'config_home': os.environ.get('CHARTREUX_HOME'),",
        "    'scratch_home': SCRATCH,",
        "    'cwd': os.getcwd(),",
        "}",
        "with open(EVIDENCE, 'w', encoding='utf-8') as handle:",
        "    json.dump(evidence, handle)",
        "",
        "deadline = time.monotonic() + 5.0",
        "while not os.path.exists(RELEASE) and time.monotonic() < deadline:",
        "    time.sleep(0.002)",
        "sum(range(100000))",
    )
    return "\n".join(lines) + "\n"


def _child_environment(guard: EgressGuard) -> dict[str, str]:
    """An allowlisted child environment: no credential, redirected homes."""
    environment = {
        name: os.environ[name]
        for name in ("PATH", "TMPDIR", "LANG", "LC_ALL")
        if name in os.environ
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment[CONFIG_HOME_ENV] = str(guard.config_home)
    environment[SCRATCHPAD_ENV] = str(guard.scratch_home)
    assert not [name for name in PROVIDER_CREDENTIAL_NAMES if name in environment]
    return environment


def _spawn_synthetic_child(guard: EgressGuard) -> subprocess.Popen[str]:
    repository = repository_root()
    assert guard.root != repository
    assert repository not in guard.root.parents
    return subprocess.Popen(
        [sys.executable, "-c", _child_program()],
        cwd=guard.root,
        env=_child_environment(guard),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _assert_synthetic_child(tmp_path: Path) -> None:
    """A live real tree is observed; the same tree after exit is missing."""
    guard = _install_guard(tmp_path)
    process: subprocess.Popen[str] | None = None
    try:
        process = _spawn_synthetic_child(guard)
        alive = observe_process_tree(process.pid)
        assert alive.outcome == OUTCOME_OBSERVED, alive.record()
        assert alive.pids == (process.pid,)
        assert alive.rss_bytes is not None
        assert alive.rss_bytes > 0
        assert alive.task_count == 1
        assert alive.thread_count == 1
        assert alive.processes[0].num_threads == 1
        cpu_self_s = alive.processes[0].cpu_self_s
        assert cpu_self_s is not None
        assert cpu_self_s >= 0

        (guard.scratch_home / _CHILD_RELEASE_FILENAME).write_text(
            "release\n", encoding="utf-8"
        )
        assert process.wait(timeout=_CHILD_WAIT_S) == 0

        evidence_path = guard.scratch_home / _CHILD_EVIDENCE_FILENAME
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert evidence["blocked"] is True
        assert evidence["marker"] == _CHILD_EGRESS_MARKER
        assert evidence["pid"] == process.pid
        assert evidence["bytecode_disabled"] is True
        assert evidence["config_home"] == str(guard.config_home)
        assert evidence["scratch_home"] == str(guard.scratch_home)
        assert Path(evidence["cwd"]) == guard.root
        assert not [
            name
            for name in PROVIDER_CREDENTIAL_NAMES
            if name in evidence["environment"]
        ]
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=_CHILD_WAIT_S)
        guard.restore()

    assert process is not None
    fields, failure = read_stat(process.pid, PROC_ROOT)
    assert (fields, failure) == (None, REASON_PROCESS_MISSING)

    gone = observe_process_tree(process.pid)
    assert gone.outcome == OUTCOME_MISSING
    assert gone.reason == REASON_PROCESS_MISSING
    assert gone.missing == (MissingMember(process.pid, REASON_PROCESS_MISSING),)
    assert gone.processes == ()
    record = gone.record()
    assert record["rss_bytes"] is None
    assert record["cpu_total_s"] is None

    delta = sample_delta(alive, gone)
    assert delta.processes == ()
    assert [(member.pid, member.reason) for member in delta.missing] == [
        (process.pid, REASON_PROCESS_MISSING)
    ]


@pytest.mark.timeout(_CONTRACT_TIMEOUT_S)
def test_runtime_process_tree_contract(tmp_path: Path) -> None:
    """Process-tree CPU/RSS is observed, reported and never gated."""
    started = perf_counter()
    _assert_support_reporting(tmp_path)
    _assert_stat_decoding()
    _assert_fake_proc_tree(tmp_path)
    _assert_own_tree()
    _assert_delta_semantics()
    _assert_synthetic_child(tmp_path)
    elapsed = perf_counter() - started
    assert elapsed <= _CONTRACT_WINDOW_S, (
        f"process-tree contract took {elapsed:.3f}s, window {_CONTRACT_WINDOW_S}s"
    )
