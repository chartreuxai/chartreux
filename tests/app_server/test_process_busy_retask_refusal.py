"""H4b: a busy process-route re-task is refused with the registry's typed error.

The in-process route has always refused a re-task of a record that is not idle
with ``AgentBusyError``. The process route admitted the second run anyway: the
launch acknowledgment overwrote the handle's run id and task while the first
run was still streaming, so the row transiently reported the second run's
identity for a run that was never launched, and only the child's own refusal
stopped it downstream.

These pins hold the process route to the in-process semantics: the refusal is
the SAME typed class (``AgentBusyError``, the one the retention registry
raises), it is raised before any admission state moves, the running run keeps
its run id and its task, and the fresh-spawn and re-task-after-terminal paths
are unchanged.

Behavioral discrimination: on the base tree the refusal pins fail with
``Failed: DID NOT RAISE``, because the second admission overwrites the live
handle. The re-task-after-terminal and fresh-spawn pins hold on both trees and
are disclosed as robustness guards, not discrimination.

The H6 section below extends the same surface to the rest of the admission
lifecycle: a FOREGROUND run (which owns no task on the handle, so the
task-based busy state read it idle), a retained child that died while idle
(whose re-task takes the fresh-spawn path and left its admission claim
behind), and a release arriving while a re-task holds that claim or after a
handle is already closed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
import time
from typing import Any

import pytest

from chartreux.app_server.root_agent import AgentChildHost
from chartreux.app_server.subagent_ports import SubagentCompletion
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.models import SubagentsConfig
from chartreux.core.events import AssistantEvent
from chartreux.core.subagents import (
    AgentBusyError,
    AgentClosedError,
    ReleaseAgentOutcome,
    SubagentManagementError,
    TaskArgs,
    TaskResult,
)
from chartreux.core.tools.base import InvokeContext
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_COMPLETED = "completed"
_BUSY_MESSAGE = "Agent is already running a task"


class _GatedSource:
    """A child completion source that relays nothing until a run is released.

    A retained child's process serves every re-task through the same source,
    so each ``stream`` call takes the next scripted run and its own gate. A
    run that is still waiting on its gate is exactly the RUNNING state a busy
    refusal has to protect.
    """

    def __init__(self, *runs: Sequence[Any]) -> None:
        self.runs = [list(frames) for frames in runs]
        self.releases = [asyncio.Event() for _ in self.runs]
        self.streams = 0

    def release(self, index: int = 0) -> None:
        self.releases[index].set()

    def stream(self, request: Any) -> Any:
        index = self.streams
        self.streams += 1
        frames = self.runs[index] if index < len(self.runs) else []
        gate = self.releases[index] if index < len(self.releases) else None
        return self._iter(frames, gate)

    async def _iter(
        self, frames: list[Any], gate: asyncio.Event | None
    ) -> AsyncGenerator[Any, None]:
        if gate is not None:
            await gate.wait()
        for frame in frames:
            yield frame


class _SpawnedChild:
    """One spawned child double: a completion source plus a close counter."""

    def __init__(self, agent_id: str, session_id: str, source: Any) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.child_session_dir = None
        self.completion_source = source
        self.wire_client = None
        self.closes = 0
        self.alive = True

    async def close(self) -> None:
        self.closes += 1
        self.alive = False


class _RecordingSpawn:
    """The spawn adapter these pins launch children through."""

    def __init__(self, *sources: Any) -> None:
        self._sources = list(sources)
        self.spawned: list[_SpawnedChild] = []

    async def spawn(self, spec: Any) -> _SpawnedChild:
        source = self._sources.pop(0) if self._sources else _GatedSource()
        child = _SpawnedChild(
            agent_id=spec.agent_id,
            session_id=f"child-session-{len(self.spawned) + 1}",
            source=source,
        )
        self.spawned.append(child)
        return child


def _root(tmp_path: Path) -> Any:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
        )
    )
    config = config.model_copy(update={"subagents": SubagentsConfig()})
    return build_test_agent_loop(
        config=config, backend=FakeBackend([[mock_llm_chunk(content="ack")]])
    )


def _ctx(loop: Any, call_id: str) -> InvokeContext:
    return InvokeContext(tool_call_id=call_id, session_id=loop.session_id)


def _completed_frames(response: str) -> list[Any]:
    return [
        AssistantEvent(content=response),
        SubagentCompletion(outcome=_COMPLETED, turns_used=1),
    ]


async def _consume(
    host: AgentChildHost, args: TaskArgs, ctx: InvokeContext
) -> list[Any]:
    """Drive one launch through the runner, exactly as the task tool does."""
    return [item async for item in host.runner.run(args, ctx)]


async def _start(host: AgentChildHost, loop: Any, *, call_id: str) -> tuple[Any, Any]:
    """Launch one retained background child through the runner; answer it."""
    args = TaskArgs(
        task="Inspect the project", identity="coder", background=True, retain=True
    )
    events = await _consume(host, args, _ctx(loop, call_id))
    ack = events[-1]
    assert isinstance(ack, TaskResult)
    assert ack.agent_id is not None
    handle = host._manageable(ack.agent_id)
    assert handle is not None
    return handle, ack


def _retask(
    handle: Any, *, task: str = "Continue the work", retain: bool = True
) -> TaskArgs:
    return TaskArgs(
        task=task,
        identity="coder",
        background=True,
        agent_id=handle.agent_id,
        retain=retain,
    )


def _rows(host: AgentChildHost) -> dict[str, Any]:
    return {row.agent_id: row for row in host.inventory_snapshot().agents}


# ---------------------------------------------------------------------------
# The refusal itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_busy_re_task_is_refused_with_the_in_process_typed_error(
    tmp_path: Path,
) -> None:
    """A re-task of a running retained child raises the registry's busy error."""
    loop = _root(tmp_path)
    source = _GatedSource(_completed_frames("run one"), _completed_frames("run two"))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    handle: Any = None
    try:
        handle, first_ack = await _start(host, loop, call_id="task-1")
        assert handle.running
        running_row = _rows(host)[handle.agent_id]
        scheduled_before = len(host._tasks)

        with pytest.raises(AgentBusyError) as refused:
            await _consume(
                host, _retask(handle, task="Continue while busy"), _ctx(loop, "task-2")
            )

        # The refusal is the in-process class and its message, not a lookalike.
        assert type(refused.value) is AgentBusyError
        assert isinstance(refused.value, SubagentManagementError)
        assert str(refused.value) == _BUSY_MESSAGE

        # The refusal moved no admission state: the run keeps its identity,
        # its task object, its row state, and nothing new was scheduled.
        assert handle.run_id == first_ack.run_id
        assert handle.task is not None and not handle.task.done()
        assert len(host._tasks) == scheduled_before
        after = _rows(host)[handle.agent_id]
        assert after.availability == running_row.availability == "running"
        assert after.current_run_id == first_ack.run_id
        assert after.current_task_summary == running_row.current_task_summary
        # Only the refused call named a different task, and it never landed.
        assert "Continue while busy" not in (after.current_task_summary or "")

        # The first run finishes untouched, and no second run was consumed.
        source.release(0)
        await asyncio.gather(handle.task, return_exceptions=True)
        result = await host.get_agent_result(handle.agent_id, first_ack.run_id)
        streams = source.streams
    finally:
        for index in range(len(source.releases)):
            source.release(index)
        await host.close()
        await loop.aclose()

    assert result is not None and result.response == "run one"
    assert streams == 1


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_busy_refusal_precedes_publication_and_keeps_retention(
    tmp_path: Path,
) -> None:
    """A busy re-task cannot change the live handle's retention policy.

    Publication applies the re-task's retention argument before admission, so
    a refusal made only at admission would let a refused ``retain=false``
    quietly change a running child's settle. The runner refuses first, which
    is what keeps this pin observable.
    """
    loop = _root(tmp_path)
    source = _GatedSource(_completed_frames("run one"))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    handle: Any = None
    try:
        handle, ack = await _start(host, loop, call_id="task-1")
        with pytest.raises(AgentBusyError):
            await _consume(host, _retask(handle, retain=False), _ctx(loop, "task-2"))
        retained_after_refusal = handle.retained
        linked_after_refusal = handle.linked
    finally:
        for index in range(len(source.releases)):
            source.release(index)
        if handle is not None and handle.task is not None:
            await asyncio.gather(handle.task, return_exceptions=True)
        await host.close()
        await loop.aclose()

    assert retained_after_refusal is True
    assert linked_after_refusal is True
    assert ack.run_id is not None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_admission_refuses_a_busy_handle_at_the_synchronous_point(
    tmp_path: Path,
) -> None:
    """The host's own admission refuses too, without overwriting the run id.

    This is the synchronous check that a concurrent re-task cannot get past:
    the runner's earlier refusal keeps the ordinary path free of side effects,
    but only the admission point sees both competing re-tasks.
    """
    loop = _root(tmp_path)
    source = _GatedSource(_completed_frames("run one"))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    handle: Any = None
    try:
        handle, ack = await _start(host, loop, call_id="task-1")
        task_before = handle.task
        scheduled_before = len(host._tasks)
        with pytest.raises(AgentBusyError):
            host.admit_background(handle, _retask(handle), _ctx(loop, "task-2"))
        run_id_after_refusal = handle.run_id
        task_after_refusal = handle.task
        scheduled_after_refusal = len(host._tasks)
    finally:
        for index in range(len(source.releases)):
            source.release(index)
        if handle is not None and handle.task is not None:
            await asyncio.gather(handle.task, return_exceptions=True)
        await host.close()
        await loop.aclose()

    assert run_id_after_refusal == ack.run_id
    assert task_after_refusal is task_before
    assert scheduled_after_refusal == scheduled_before


# ---------------------------------------------------------------------------
# The paths that must stay unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_re_task_after_the_run_settles_is_still_admitted(tmp_path: Path) -> None:
    """A retained idle child still accepts a re-task with its own new run id."""
    loop = _root(tmp_path)
    source = _GatedSource(_completed_frames("run one"), _completed_frames("run two"))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    handle: Any = None
    second_ack: Any = None
    try:
        handle, first_ack = await _start(host, loop, call_id="task-1")
        source.release(0)
        await asyncio.gather(handle.task, return_exceptions=True)
        idle_row = _rows(host)[handle.agent_id]

        events = await _consume(host, _retask(handle), _ctx(loop, "task-2"))
        second_ack = events[-1]
        source.release(1)
        await asyncio.gather(handle.task, return_exceptions=True)
        first = await host.get_agent_result(handle.agent_id, first_ack.run_id)
        second = await host.get_agent_result(handle.agent_id, second_ack.run_id)
    finally:
        for index in range(len(source.releases)):
            source.release(index)
        await host.close()
        await loop.aclose()

    assert idle_row.availability == "idle"
    assert isinstance(second_ack, TaskResult)
    assert second_ack.status == "launched"
    assert second_ack.run_id is not None and second_ack.run_id != first_ack.run_id
    assert first is not None and first.response == "run one"
    assert second is not None and second.response == "run two"
    assert source.streams == 2


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_fresh_background_spawn_is_unaffected(tmp_path: Path) -> None:
    """A launch with no agent id is admitted exactly as before."""
    loop = _root(tmp_path)
    source = _GatedSource(_completed_frames("fresh run"))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    handle: Any = None
    try:
        handle, ack = await _start(host, loop, call_id="task-1")
        assert handle.running
        source.release(0)
        await asyncio.gather(handle.task, return_exceptions=True)
        result = await host.get_agent_result(handle.agent_id, ack.run_id)
    finally:
        for index in range(len(source.releases)):
            source.release(index)
        await host.close()
        await loop.aclose()

    assert ack.status == "launched"
    assert ack.run_id is not None
    assert result is not None and result.response == "fresh run"


# ---------------------------------------------------------------------------
# Two re-tasks of one idle handle: exactly one is admitted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_concurrent_re_tasks_admit_one_and_the_loser_cannot_change_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A racing re-task is refused before it can rewrite the winner's handle.

    Both re-tasks observe the retained child idle, so both pass the running
    check; the difference is the publication await between that check and the
    admission. The pin parks the explicit ``retain=false`` re-task inside its
    link await (the instrumented publication await point) and starts the
    inheriting re-task there. The second re-task must be refused with the
    registry's typed busy error BEFORE it reaches publication, and the
    handle's retention, run id and task must be the first re-task's -- on the
    base tree the second one publishes over the first and is admitted itself,
    so the explicit continuation runs under the inherited policy it never
    asked for.
    """
    from chartreux.app_server import _subagent_launch as launch_module

    loop = _root(tmp_path)
    source = _GatedSource(_completed_frames("run one"), _completed_frames("run two"))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    handle: Any = None
    entered = asyncio.Event()
    gate = asyncio.Event()
    link_calls = 0
    original_link = launch_module.record_process_child_link

    async def gated_link(*args: Any, **kwargs: Any) -> Any:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 1:
            entered.set()
            await gate.wait()
        return await original_link(*args, **kwargs)

    explicit = "Continue as this instance's last run"
    inherited = "Continue as a resident instance"
    try:
        handle, first_ack = await _start(host, loop, call_id="task-1")
        source.release(0)
        await asyncio.gather(handle.task, return_exceptions=True)
        assert _rows(host)[handle.agent_id].availability == "idle"
        monkeypatch.setattr(launch_module, "record_process_child_link", gated_link)

        winner = asyncio.create_task(
            _consume(
                host, _retask(handle, task=explicit, retain=False), _ctx(loop, "task-a")
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=30)
        # The first re-task is now suspended AFTER committing its retention,
        # inside its publication -- the window the race needs.
        loser = asyncio.create_task(
            _consume(
                host,
                TaskArgs(
                    task=inherited,
                    identity="coder",
                    background=True,
                    agent_id=handle.agent_id,
                ),
                _ctx(loop, "task-b"),
            )
        )
        loser_outcome = (
            await asyncio.wait_for(
                asyncio.gather(loser, return_exceptions=True), timeout=30
            )
        )[0]
        gate.set()
        winner_outcome = (
            await asyncio.wait_for(
                asyncio.gather(winner, return_exceptions=True), timeout=30
            )
        )[0]

        # The explicit re-task is the one that wins; the inheriting one is
        # refused with the registry's typed class and message.
        assert isinstance(winner_outcome, list), (
            f"the explicit-retain re-task must be admitted, not {winner_outcome!r}"
        )
        assert isinstance(loser_outcome, AgentBusyError), loser_outcome
        assert type(loser_outcome) is AgentBusyError
        assert str(loser_outcome) == _BUSY_MESSAGE
        ack = winner_outcome[-1]
        assert isinstance(ack, TaskResult)
        assert ack.status == "launched"

        # The refused re-task never reached publication: one link was
        # recorded (the winner's), and the winner's own policy is the
        # handle's committed one -- never the loser's inherit-read or write.
        assert link_calls == 1
        assert handle.retained is False
        assert handle.run_id == ack.run_id
        assert handle.reuse_claim is None
        running_row = _rows(host)[handle.agent_id]
        assert running_row.availability == "running"
        assert running_row.current_run_id == ack.run_id

        # The winner's run is the only one the child serves, and it runs to
        # the winner's own settle: one background run, one result.
        source.release(1)
        await asyncio.gather(handle.task, return_exceptions=True)
        result = await host.get_agent_result(handle.agent_id, ack.run_id)
        assert handle.current_task_summary == explicit
        assert source.streams == 2
        assert handle.closed
    finally:
        for index in range(len(source.releases)):
            source.release(index)
        if handle is not None and handle.task is not None:
            await asyncio.gather(handle.task, return_exceptions=True)
        await host.close()
        await loop.aclose()

    assert first_ack.run_id is not None
    assert result is not None and result.response == "run two"


# ---------------------------------------------------------------------------
# H6: the busy state covers a foreground run, and an admission claim is finite
# ---------------------------------------------------------------------------

_CLOSED_MESSAGE = (
    "This instance is closed; launch a new task without agent_id. "
    "Use retain=true when a resident continuation is required."
)


async def _wait_for_stream(
    source: _GatedSource, count: int, *, timeout: float = 30.0
) -> None:
    """Wait until the child has been asked to serve its ``count``-th run."""
    deadline = time.monotonic() + timeout
    while source.streams < count:
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"the child was asked for {source.streams} runs, not {count}"
            )
        await asyncio.sleep(0.01)


async def _drain(host: AgentChildHost, *sources: _GatedSource) -> None:
    """Release every scripted gate and join whatever is still running here."""
    for source in sources:
        for index in range(len(source.releases)):
            source.release(index)
    await asyncio.gather(*list(host._tasks), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_mid_flight_foreground_run_refuses_a_concurrent_re_task(
    tmp_path: Path,
) -> None:
    """A foreground run makes the child busy, although no task owns it.

    A foreground ``task`` call is awaited by the runner on the caller's own
    stream, so the handle holds no ``asyncio.Task`` for it. On the base tree the
    busy state reads the task alone and therefore reports the child idle for the
    whole foreground run, so a concurrent background re-task is ADMITTED: the
    launch acknowledgment rewrites the handle's run id and a second run starts on
    the one child while the first is still streaming.
    """
    loop = _root(tmp_path)
    source = _GatedSource(_completed_frames("run one"), _completed_frames("run two"))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    handle: Any = None
    foreground: asyncio.Task[Any] | None = None
    try:
        handle, first_ack = await _start(host, loop, call_id="task-1")
        source.release(0)
        await asyncio.gather(handle.task, return_exceptions=True)
        assert _rows(host)[handle.agent_id].availability == "idle"
        scheduled_before = len(host._tasks)

        foreground = asyncio.create_task(
            _consume(
                host,
                TaskArgs(
                    task="Continue in the foreground",
                    identity="coder",
                    background=False,
                    agent_id=handle.agent_id,
                    retain=True,
                ),
                _ctx(loop, "task-2"),
            )
        )
        await _wait_for_stream(source, 2)
        # The run is in flight and nothing on the handle owns it.
        assert handle.task is not None and handle.task.done()
        run_before = handle.run_id
        assert run_before is not None and run_before != first_ack.run_id

        with pytest.raises(AgentBusyError) as refused:
            await _consume(
                host, _retask(handle, task="Continue while busy"), _ctx(loop, "task-3")
            )

        # The same typed class and message the in-process admission raises.
        assert type(refused.value) is AgentBusyError
        assert str(refused.value) == _BUSY_MESSAGE

        # The refusal moved no admission state: the foreground run keeps its
        # identity and its task, the child was asked for exactly its one run,
        # and nothing new was scheduled.
        assert handle.run_id == run_before
        assert handle.current_task_summary == "Continue in the foreground"
        assert source.streams == 2
        assert len(host._tasks) == scheduled_before

        source.release(1)
        events = await asyncio.wait_for(foreground, timeout=30)
        result = events[-1]
    finally:
        if foreground is not None:
            await _drain(host, source)
            await asyncio.gather(foreground, return_exceptions=True)
        await host.close()
        await loop.aclose()

    assert isinstance(result, TaskResult)
    assert result.response == "run two"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_dead_retained_child_re_task_spawns_fresh_and_drops_the_claim(
    tmp_path: Path,
) -> None:
    """A re-task of a crashed retained child takes the fresh-spawn path.

    The claim that admits a re-task is taken before publication, and the
    fresh-spawn path a dead retained child takes never runs the reuse branch
    that released it. On the base tree the dead handle therefore keeps the claim
    forever: the spawn succeeds, but every later re-task of that agent_id answers
    a misleading busy refusal and the row stays stranded. Every publication exit
    releases the claim, so the fresh spawn leaves nothing behind and the next
    re-task of the same id behaves like the first.
    """
    loop = _root(tmp_path)
    # Each spawned child is a process of its own, so each one serves its own
    # scripted run: the crashed child first, then one fresh child per re-task.
    first_source = _GatedSource(_completed_frames("run one"))
    second_source = _GatedSource(_completed_frames("fresh two"))
    third_source = _GatedSource(_completed_frames("fresh three"))
    adapter = _RecordingSpawn(first_source, second_source, third_source)
    host = AgentChildHost(loop, spawn=adapter)
    handle: Any = None
    try:
        handle, first_ack = await _start(host, loop, call_id="task-1")
        first_source.release(0)
        await asyncio.gather(handle.task, return_exceptions=True)
        old_agent_id = handle.agent_id
        # The child crashed while idle: its process reports no life, exactly as
        # a reaped child does, so the re-task must spawn a fresh one.
        handle.spawned.alive = False

        events = await _consume(host, _retask(handle), _ctx(loop, "task-2"))
        fresh_ack = events[-1]
        claim_after_fresh_spawn = handle.reuse_claim
        spawned_after_fresh = len(adapter.spawned)

        # The same agent id is re-taskable again: the refusal the base tree
        # answers here is the stranded claim, not a real busy child.
        again = await _consume(host, _retask(handle), _ctx(loop, "task-3"))
        second_ack = again[-1]
        claim_after_second = handle.reuse_claim
        spawned_total = len(adapter.spawned)
        # Each fresh child serves exactly its own run, to completion.
        await _drain(host, first_source, second_source, third_source)
        fresh_result = await host.get_agent_result(fresh_ack.agent_id, fresh_ack.run_id)
        second_result = await host.get_agent_result(
            second_ack.agent_id, second_ack.run_id
        )
    finally:
        await _drain(host, first_source, second_source, third_source)
        await host.close()
        await loop.aclose()

    assert claim_after_fresh_spawn is None
    assert isinstance(fresh_ack, TaskResult)
    assert fresh_ack.status == "launched"
    assert fresh_ack.agent_id != old_agent_id
    assert spawned_after_fresh == 2
    assert claim_after_second is None
    assert isinstance(second_ack, TaskResult)
    assert second_ack.status == "launched"
    assert second_ack.agent_id not in (old_agent_id, fresh_ack.agent_id)
    assert spawned_total == 3
    assert fresh_result is not None and fresh_result.response == "fresh two"
    assert second_result is not None and second_result.response == "fresh three"
    assert first_ack.run_id is not None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_release_during_a_claimed_publication_cannot_purge_the_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither release path may retire a handle an admission has claimed.

    The pin parks a re-task inside its publication await with its admission
    claim held -- the window in which the handle is neither idle (the run is
    about to start) nor running (no run exists yet). On the base tree both
    releases answer success anyway: the handle is closed and purged, and the
    parked winner is then admitted onto the purged handle -- the release having
    promised a retained resident that the very next admission discarded.
    """
    from chartreux.app_server import _subagent_launch as launch_module

    loop = _root(tmp_path)
    source = _GatedSource(_completed_frames("run one"), _completed_frames("run two"))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    handle: Any = None
    winner: asyncio.Task[Any] | None = None
    entered = asyncio.Event()
    gate = asyncio.Event()
    link_calls = 0
    original_link = launch_module.record_process_child_link

    async def gated_link(*args: Any, **kwargs: Any) -> Any:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 1:
            entered.set()
            await gate.wait()
        return await original_link(*args, **kwargs)

    try:
        handle, first_ack = await _start(host, loop, call_id="task-1")
        source.release(0)
        await asyncio.gather(handle.task, return_exceptions=True)
        monkeypatch.setattr(launch_module, "record_process_child_link", gated_link)
        winner = asyncio.create_task(
            _consume(host, _retask(handle), _ctx(loop, "task-2"))
        )
        await asyncio.wait_for(entered.wait(), timeout=30)
        assert handle.reuse_claim is not None

        guarded = await host.release_guarded(handle.agent_id)
        with pytest.raises(AgentBusyError) as refused:
            await host.release_agent(handle.agent_id)
        assert type(refused.value) is AgentBusyError
        assert str(refused.value) == _BUSY_MESSAGE
        closed_after_releases = handle.closed
        still_held = handle.agent_id in {child.agent_id for child in host.children()}

        gate.set()
        outcome = (
            await asyncio.wait_for(asyncio.gather(winner, return_exceptions=True), 30)
        )[0]
        ack = outcome[-1] if isinstance(outcome, list) else None
        running_row = _rows(host).get(handle.agent_id)
        admitted = (
            isinstance(ack, TaskResult)
            and ack.status == "launched"
            and running_row is not None
            and running_row.availability == "running"
            and running_row.current_run_id == ack.run_id
        )
        handle_run_id = handle.run_id
        handle_closed = handle.closed
        source.release(1)
        await asyncio.gather(handle.task, return_exceptions=True)
        result = await host.get_agent_result(handle.agent_id, handle_run_id)
    finally:
        gate.set()
        await _drain(host, source)
        if winner is not None:
            await asyncio.gather(winner, return_exceptions=True)
        await host.close()
        await loop.aclose()

    assert guarded is ReleaseAgentOutcome.CONFLICT
    assert closed_after_releases is False
    assert still_held is True
    assert admitted, outcome
    assert handle_closed is False
    assert result is not None and result.response == "run two"
    assert first_ack.run_id is not None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_admission_refuses_a_closed_handle_with_the_registry_closed_error(
    tmp_path: Path,
) -> None:
    """A closed handle refuses a run with the registry's typed closed error.

    The admission guard reads the running state only, so on the base tree a
    closed handle is admitted: a run id is minted for a child whose process is
    gone, and the launch acknowledgment describes a run nobody can observe. The
    refusal is the same typed closed error the reuse admission raises (G1a), and
    it moves no admission state.
    """
    loop = _root(tmp_path)
    source = _GatedSource(_completed_frames("run one"))
    host = AgentChildHost(loop, spawn=_RecordingSpawn(source))
    handle: Any = None
    try:
        events = await _consume(
            host,
            TaskArgs(
                task="One-shot work", identity="coder", background=True, retain=False
            ),
            _ctx(loop, "task-1"),
        )
        ack = events[-1]
        handle = host._manageable(ack.agent_id)
        assert handle is not None
        source.release(0)
        await asyncio.gather(handle.task, return_exceptions=True)
        assert handle.closed
        run_before = handle.run_id
        scheduled_before = len(host._tasks)

        with pytest.raises(AgentClosedError) as refused:
            host.admit_background(
                handle, _retask(handle, retain=False), _ctx(loop, "task-2")
            )

        assert type(refused.value) is AgentClosedError
        assert str(refused.value) == _CLOSED_MESSAGE
        run_after = handle.run_id
        scheduled_after = len(host._tasks)
    finally:
        await _drain(host, source)
        await host.close()
        await loop.aclose()

    assert run_after == run_before
    assert scheduled_after == scheduled_before
    assert ack.run_id is not None
