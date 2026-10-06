"""Finite Linux process tests; every fixture reclaims its own groups/escaped PID."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
from pathlib import Path
import shlex
import signal
import sys

import pytest

from chartreux.core.background_jobs import (
    BackgroundJobRegistry,
    BashReadArgs,
    BashStartArgs,
    BashStopArgs,
)
from chartreux.core.tools import secret_redaction as sr
from chartreux.core.utils.async_subprocess import (
    ProcessGroupOwnership,
    _process_identity,
    registered_process_group,
)


async def test_identity_read_handles_process_disappearing_during_read(monkeypatch):
    class Vanished:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            raise ProcessLookupError("process disappeared during stat read")

    monkeypatch.setattr("builtins.open", lambda *args: Vanished())
    assert _process_identity(123) is None


@pytest.mark.parametrize("leader_disappears_during_scan", [False, True])
async def test_reused_leaderless_group_never_signals(
    monkeypatch, leader_disappears_during_scan
):
    from chartreux.core.utils import async_subprocess as asp

    owner = ProcessGroupOwnership(123, (100, 123, 123, "S"))
    monkeypatch.setattr(
        asp,
        "_process_identity",
        lambda pid: (
            ((100, 123, 123, "S") if leader_disappears_during_scan else None)
            if pid == 123
            else (200, 123, 123, "S")
        ),
    )
    monkeypatch.setattr(os, "listdir", lambda path: ["456"])
    monkeypatch.setattr(
        os, "killpg", lambda *args: pytest.fail("reused group signaled")
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        owner.signal(signal.SIGTERM)
    assert not owner.valid
    owner.signal(signal.SIGKILL)


async def test_identity_changed_settles_failed_and_close_succeeds(monkeypatch):
    from chartreux.core.utils.async_subprocess import ProcessGroupIdentityChanged

    async with managed() as registry:
        port = registry.root_port()
        started = await port.start(BashStartArgs(command="sleep 15"))
        job = registry._jobs[started.job.job_id]
        assert job.owner is not None and job.process is not None
        proc = job.process
        # Reclaim the actual fixture process before injecting historical reuse.
        os.killpg(proc.pid, signal.SIGKILL)

        def changed():
            raise ProcessGroupIdentityChanged("Process group identity changed")

        monkeypatch.setattr(job.owner, "members", changed)
        assert job.supervisor is not None
        await asyncio.wait_for(job.supervisor, 3)
        assert job.state == "failed" and not job.owner.valid
        assert registry.active_count == 0
        fresh = port.reserve(BashStartArgs(command="true"))
        registry.rollback(fresh)
        await registry.aclose()


async def test_stop_cancels_pending_spawn_and_reclaims_handle(monkeypatch):
    from chartreux.core import background_jobs as bg

    entered, release = asyncio.Event(), asyncio.Event()
    original = bg.spawn_shell_command

    async def blocked(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(bg, "spawn_shell_command", blocked)
    async with managed() as registry:
        port = registry.root_port()
        starting = asyncio.create_task(port.start(BashStartArgs(command="sleep 15")))
        await entered.wait()
        job_id = next(iter(registry._jobs))
        stopping = asyncio.create_task(port.stop(BashStopArgs(job_id=job_id)))
        await asyncio.sleep(0)
        release.set()
        result = await asyncio.wait_for(stopping, 3)
        assert result.job.state == "stopped" and registry.active_count == 0
        with pytest.raises(asyncio.CancelledError):
            await starting


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(sys.platform != "linux", reason="Linux process identity"),
]


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sr, "_loaded_credentials", lambda: [])
    monkeypatch.setattr(sr, "_oauth_secret_values", lambda: frozenset())
    monkeypatch.setattr(sr, "credential_env_var_names", lambda *args: frozenset())


def python_command(source: str) -> str:
    return f"exec {shlex.quote(sys.executable)} -c {shlex.quote(source)}"


async def file_ready(path: Path) -> str:
    async with asyncio.timeout(5):
        while not path.exists() or not path.read_text():
            await asyncio.sleep(0.001)
        return path.read_text()


@asynccontextmanager
async def managed() -> AsyncIterator[BackgroundJobRegistry]:
    registry = BackgroundJobRegistry()
    try:
        yield registry
    finally:
        jobs = tuple(registry._jobs.values())
        try:
            await asyncio.wait_for(registry.aclose(), 6)
        finally:
            for job in jobs:
                if job.owner is not None and job.owner.valid:
                    job.owner.signal(signal.SIGKILL)
                if job.process is not None:
                    job.process._transport.close()  # type: ignore[attr-defined]
                    await asyncio.wait_for(job.process.wait(), 2)


@pytest.mark.parametrize("ignore_term", [False, True])
async def test_term_shutdown_and_kill_escalation(
    tmp_path: Path, ignore_term: bool
) -> None:
    ready = tmp_path / "ready"
    source = (
        "import os, signal\n"
        "signal.alarm(15)\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + "print('stdout!', flush=True)\n"
        "os.write(2, b'stderr!\\n')\n"
        f"open({str(ready)!r}, 'w').write('ready')\n"
        "while True: signal.pause()\n"
    )
    async with managed() as registry:
        port = registry.root_port()
        start = await port.start(BashStartArgs(command=python_command(source)))
        await file_ready(ready)
        job = registry._jobs[start.job.job_id]
        assert job.process is not None and job.process.stderr is None
        result = await asyncio.wait_for(port.stop(BashStopArgs(job_id=job.job_id)), 5)
        assert result.job.state == "stopped"
        assert result.job.exit_code == (
            -signal.SIGKILL if ignore_term else -signal.SIGTERM
        )
        page = await port.read(BashReadArgs(job_id=job.job_id))
        assert "stdout!" in "".join(record.text for record in page.records)
        assert "stderr!" in "".join(record.text for record in page.records)
        assert registry.active_count == 0
        assert job.capture_task is not None and job.capture_task.done()
        assert job.supervisor is not None and job.supervisor.done()
        assert job.process._transport.is_closing()  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="not owned"):
            registered_process_group(job.process)


@pytest.mark.parametrize("pipes", ["held", "closed", "escaped"])
@pytest.mark.parametrize("late_delivery", [False, True])
async def test_leader_exit_settles_residual_group_and_pipes(
    tmp_path: Path, pipes: str, late_delivery: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = tmp_path / "child"
    if late_delivery:
        factory = asyncio.create_subprocess_shell
        listdir = os.listdir
        raced = False

        async def after_exit(*args, **kwargs):
            proc = await factory(*args, **kwargs)
            async with asyncio.timeout(3):
                while proc.returncode is None:
                    await asyncio.sleep(0.001)
            return proc

        def raced_snapshot(path):
            nonlocal raced
            if path == "/proc" and not raced:
                raced = True
                return []
            return listdir(path)

        monkeypatch.setattr(asyncio, "create_subprocess_shell", after_exit)
        monkeypatch.setattr(os, "listdir", raced_snapshot)
    source = (
        "import os, signal\n"
        "r,w = os.pipe()\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        " os.close(r)\n"
        " signal.alarm(15)\n"
        + (" os.setsid()\n" if pipes == "escaped" else "")
        + (" os.close(1); os.close(2)\n" if pipes == "closed" else "")
        + f" open({str(ready)!r}, 'w').write(str(os.getpid()))\n"
        " os.write(w,b'!'); os.close(w)\n"
        " while True: signal.pause()\n"
        "os.close(w); os.read(r,1); os.close(r)\n"
        # No witness gate: the leader exits as soon as its child is ready.
        "print('leader!', flush=True)\n"
    )
    child: int | None = None
    try:
        async with managed() as registry:
            start = await registry.root_port().start(
                BashStartArgs(command=python_command(source))
            )
            child = int(await file_ready(ready))
            job = registry._jobs[start.job.job_id]
            assert job.owner is not None
            if late_delivery:
                assert job.owner.leader is None  # Gone before registration/witness.
            assert job.supervisor is not None
            await asyncio.wait_for(asyncio.shield(job.supervisor), 5)
            await registry.aclose()
            assert registry.active_count == 0
            assert job.output_complete
            assert job.output_incomplete == (pipes == "escaped")
            identity = _process_identity(child)
            if pipes == "escaped":
                assert identity is not None and identity[3] != "Z"
                assert job.state == "failed"
            else:
                assert identity is None or identity[3] == "Z"
                assert job.state == "exited"
    finally:
        if child is None and ready.exists():
            child = int(ready.read_text())
        if child is not None:
            identity = _process_identity(child)
            if identity is not None and identity[3] != "Z":
                os.kill(child, signal.SIGKILL)
            async with asyncio.timeout(2):
                while (identity := _process_identity(child)) is not None and identity[
                    3
                ] != "Z":
                    await asyncio.sleep(0.001)


async def test_immediate_exit_and_late_stop_never_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with managed() as registry:
        port = registry.root_port()
        start = await port.start(BashStartArgs(command="exit 23"))
        job = registry._jobs[start.job.job_id]
        assert job.supervisor is not None
        await asyncio.wait_for(asyncio.shield(job.supervisor), 3)
        revision = registry.revision

        def forbidden(*args: object) -> None:
            pytest.fail("finished job attempted historical group signaling")

        monkeypatch.setattr(os, "killpg", forbidden)
        result = await port.stop(BashStopArgs(job_id=job.job_id))
        assert result.already_finished and result.job.state == "exited"
        assert result.job.exit_code == 23
        assert registry.revision == revision


@pytest.mark.parametrize("root_stops", [False, True])
async def test_root_creator_read_stop_and_denied_sibling_races(
    tmp_path: Path, root_stops: bool
) -> None:
    ready = tmp_path / "ready"
    async with managed() as registry:
        root = registry.root_port()
        creator = registry.creator_port(object())
        sibling = registry.creator_port(object())
        start = await creator.start(
            BashStartArgs(
                command=python_command(
                    "import signal\nsignal.alarm(15)\n"
                    f"open({str(ready)!r}, 'w').write('ready')\n"
                    "while True: signal.pause()\n"
                )
            )
        )
        await file_ready(ready)
        reader, stopper = (creator, root) if root_stops else (root, creator)
        job = registry._jobs[start.job.job_id]
        reading = asyncio.create_task(
            reader.read(BashReadArgs(job_id=job.job_id, wait_seconds=30))
        )
        stops = [
            asyncio.create_task(stopper.stop(BashStopArgs(job_id=job.job_id)))
            for _ in range(2)
        ]
        with pytest.raises(ValueError, match="unavailable"):
            await sibling.stop(BashStopArgs(job_id=job.job_id))
        with pytest.raises(ValueError, match="unavailable"):
            await sibling.read(BashReadArgs(job_id=job.job_id))
        await asyncio.wait_for(asyncio.gather(*stops), 5)
        page = await asyncio.wait_for(reading, 2)
        assert page.job.output_complete
        assert registry.active_count == 0
        assert job.stop_task is not None and job.stop_task.done()
        assert job.supervisor is not None and job.supervisor.done()


async def test_close_releases_local_fds_and_buffers(tmp_path: Path) -> None:
    baseline = len(os.listdir("/proc/self/fd"))
    async with managed() as registry:
        port = registry.root_port()
        starts = await asyncio.gather(
            *(port.start(BashStartArgs(command="sleep 15")) for _ in range(3))
        )
        readers = [
            asyncio.create_task(
                port.read(BashReadArgs(job_id=start.job.job_id, wait_seconds=30))
            )
            for start in starts
        ]
        await asyncio.wait_for(registry.aclose(), 5)
        results = await asyncio.gather(*readers, return_exceptions=True)
        assert all(isinstance(result, ValueError) for result in results)
        assert registry.active_count == 0 and not registry._jobs
        assert not registry._admissions
        assert len(os.listdir("/proc/self/fd")) <= baseline
        with pytest.raises(ValueError, match="closed"):
            await port.start(BashStartArgs(command="true"))


@pytest.mark.parametrize(
    "current", [(101, 123, 123, "S"), (100, 321, 123, "S"), (100, 123, 321, "S")]
)
async def test_stale_group_identity_refused_before_signal(
    monkeypatch: pytest.MonkeyPatch, current: tuple[int, int, int, str]
) -> None:
    from chartreux.core.utils import async_subprocess as asp

    owner = ProcessGroupOwnership(123, (100, 123, 123, "S"))
    monkeypatch.setattr(asp, "_process_identity", lambda pid: current)

    def forbidden(*args: object) -> None:
        pytest.fail("stale ownership signaled a process group")

    monkeypatch.setattr(os, "killpg", forbidden)
    monkeypatch.setattr(os, "getpgid", forbidden)
    with pytest.raises(RuntimeError, match="identity changed"):
        owner.signal(signal.SIGTERM)


async def test_empty_group_invalidates_identity_before_pid_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.core.utils import async_subprocess as asp

    owner = ProcessGroupOwnership(123, (100, 123, 123, "S"))
    monkeypatch.setattr(asp, "_process_identity", lambda pid: None)
    monkeypatch.setattr(os, "listdir", lambda path: [])
    assert not owner.members() and not owner.valid
    monkeypatch.setattr(asp, "_process_identity", lambda pid: (200, 123, 123, "S"))

    def forbidden(*args: object) -> None:
        pytest.fail("invalidated identity signaled a reused process group")

    monkeypatch.setattr(os, "killpg", forbidden)
    owner.signal(signal.SIGKILL)


@pytest.mark.parametrize("leader", [None, (100, 123, 123, "S")])
async def test_first_leaderless_witness_retries_empty_spawn_snapshot(
    monkeypatch, leader
):
    from chartreux.core.utils import async_subprocess as asp

    owner = ProcessGroupOwnership(123, leader, spawn_started=99)
    snapshots = iter([[], [], ["456"]])
    monkeypatch.setattr(os, "listdir", lambda path: next(snapshots))
    monkeypatch.setattr(
        asp,
        "_process_identity",
        lambda pid: None if pid == 123 else (101, 123, 123, "S"),
    )
    signals = []
    monkeypatch.setattr(os, "killpg", lambda *args: signals.append(args))
    owner.signal(signal.SIGTERM)
    assert signals == [(123, signal.SIGTERM)]
    assert owner.valid and owner.witnessed[456] == (101, 123, 123)


@pytest.mark.parametrize("leader", [None, (100, 123, 123, "S")])
async def test_registered_empty_group_cannot_adopt_late_repopulation(
    monkeypatch, leader
):
    from chartreux.core.utils import async_subprocess as asp

    owner = ProcessGroupOwnership(123, leader, spawn_started=99)
    identities = {456: (101, 123, 123, "S")}
    monkeypatch.setattr(asp, "_process_identity", identities.get)
    scans = []

    def snapshot(path):
        scans.append(path)
        return list(map(str, identities))

    monkeypatch.setattr(os, "listdir", snapshot)
    assert owner.members()
    identities.clear()
    scans.clear()
    assert not owner.members() and not owner.valid
    assert len(scans) == 3
    identities[789] = (10000, 123, 123, "S")
    monkeypatch.setattr(
        os, "killpg", lambda *args: pytest.fail("recycled group signaled")
    )
    await asp.terminate_process_group(owner, grace=0, kill_timeout=0)
    assert not owner.members() and not owner.valid


async def test_continuous_registered_group_adopts_late_fork_at_close(monkeypatch):
    from chartreux.core.utils import async_subprocess as asp

    owner = ProcessGroupOwnership(123, None, spawn_started=99)
    identities = {456: (101, 123, 123, "S")}
    monkeypatch.setattr(asp, "_process_identity", identities.get)
    monkeypatch.setattr(os, "listdir", lambda path: list(map(str, identities)))
    assert owner.members()
    identities[789] = (10000, 123, 123, "S")
    assert owner.members()
    assert owner.witnessed[789] == (10000, 123, 123)
    del identities[456]
    assert owner.members()
    signals = []

    def stop(pgid, sig):
        signals.append((pgid, sig))
        identities.clear()

    monkeypatch.setattr(os, "killpg", stop)
    await asp.terminate_process_group(owner, grace=0, kill_timeout=0)
    assert signals == [(123, signal.SIGTERM)]
    assert not owner.valid


@pytest.mark.parametrize("birth", [98, 102])
async def test_registered_leaderless_group_rejects_predating_or_changed_member(
    monkeypatch, birth
):
    from chartreux.core.utils import async_subprocess as asp

    owner = ProcessGroupOwnership(123, None, spawn_started=99)
    owner.witnessed[456] = (101, 123, 123)
    monkeypatch.setattr(os, "listdir", lambda path: ["456"])
    monkeypatch.setattr(
        asp,
        "_process_identity",
        lambda pid: None if pid == 123 else (birth, 123, 123, "S"),
    )
    monkeypatch.setattr(
        os, "killpg", lambda *args: pytest.fail("reused group signaled")
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        owner.signal(signal.SIGTERM)
    assert not owner.valid


async def test_four_idle_jobs_bound_event_loop_overhead(monkeypatch):
    import time

    original = os.listdir
    scans = 0

    def costly_snapshot(path):
        nonlocal scans
        if path == "/proc":
            scans += 1
            time.sleep(0.012)  # Reproduce the measured scan cost, deterministically.
        return original(path)

    loop = asyncio.get_running_loop()

    async def heartbeat():
        delays = []
        deadline = loop.time() + 0.6
        while loop.time() < deadline:
            due = loop.time() + 0.005
            future = loop.create_future()
            loop.call_at(due, future.set_result, None)
            await future
            delays.append(max(0, loop.time() - due))
        return sum(delays) / len(delays)

    async with asyncio.timeout(6), managed() as registry:
        baseline = await heartbeat()
        await asyncio.gather(
            *(
                registry.root_port().start(BashStartArgs(command="sleep 15"))
                for _ in range(4)
            )
        )
        monkeypatch.setattr(os, "listdir", costly_snapshot)
        overhead = await heartbeat() - baseline
        assert overhead < 0.01, (baseline, overhead)
        assert scans == 0  # Live leader identity checks must not scan /proc.
        print(
            f"idle jobs mean loop overhead: {overhead:.6f}s; baseline: {baseline:.6f}s"
        )
