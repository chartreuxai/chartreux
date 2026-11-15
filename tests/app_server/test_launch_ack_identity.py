"""The process-path launch acknowledgment: display handle and honest metadata.

R2 of the residuals RCA (agent-98) found two process-path gaps against the
in-process admission:

* the acknowledgment prose printed the raw 40-character wire handle
  (``coder-attempt-<32 hex>``) where a human-facing sentence belongs, while
  the same sentence's ``agent_id="..."`` continuation instruction is what a
  re-task actually passes back;
* the acknowledgment's ``TaskResult.metadata`` was ``None``, so the model and
  provider rendered as unknown at launch where the in-process acknowledgment
  carries them (``_subagent_execution.py`` populates ``base_model``,
  ``active_provider`` and ``effective_model`` from the child record it
  admitted).

``admit_background`` returns synchronously right after scheduling the run, so
no observation can populate that same acknowledgment. The accepted plan's
ruling is therefore honest unknown: the model/provider/thinking keys are
present with ``None`` values, never guessed from the parent, and the
in-process route keeps carrying its own observed values.

Behavioral discrimination: on the pre-H2 base the prose carries the full
handle in the display parenthetical and ``ack.metadata`` is ``None``, so the
display, metadata and parity pins here fail; the full-handle pins (the
structured ``agent_id``, the ``agent_id=`` instruction, update text and
``check_agents`` rows) hold on both trees and are recorded as guards.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from chartreux.app_server._subagent_launch import new_publication_attempt_id
from chartreux.app_server.root_agent import AgentChildHost
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.events import ToolResultEvent
from chartreux.core.subagents import TaskArgs, TaskResult
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.builtins.task import Task
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

#: The model/provider keys the in-process acknowledgment populates from the
#: child record it admitted. The process path answers the same keys as
#: explicit ``None`` rather than omitting them.
_IN_PROCESS_MODEL_KEYS = frozenset({"base_model", "active_provider", "effective_model"})


class _ScriptedSource:
    """A child completion source serving one (empty) frame list per run."""

    def __init__(self, *runs: Sequence[Any]) -> None:
        self._runs = [list(run) for run in runs]
        self._index = 0

    def stream(self, request: Any) -> Any:
        index = self._index
        self._index += 1
        frames = self._runs[index] if index < len(self._runs) else []
        return self._iter(frames)

    async def _iter(self, frames: list[Any]) -> AsyncGenerator[Any, None]:
        for frame in frames:
            yield frame


class _SpawnedChild:
    """One spawned child double: a completion source plus a close counter.

    No wire client is attached, so the publication needs no capability
    negotiation and the host observes no child-side telemetry -- exactly the
    first instant of a real launch, before anything has been read.
    """

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
        source = self._sources.pop(0) if self._sources else _ScriptedSource(())
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
    return build_test_agent_loop(
        config=config, backend=FakeBackend([[mock_llm_chunk(content="ack")]])
    )


def _ctx(loop: Any, call_id: str) -> InvokeContext:
    return InvokeContext(tool_call_id=call_id, session_id=loop.session_id)


async def _launch(
    host: AgentChildHost,
    loop: Any,
    *,
    retained: bool,
    call_id: str = "task-1",
    reuse: Any = None,
) -> tuple[Any, TaskResult]:
    """Publish and acknowledge one background run, settling it before return."""
    args = TaskArgs(task="Inspect the project", identity="coder")
    ctx = _ctx(loop, call_id)
    handle = await host.publish(
        args,
        ctx,
        attempt_id=new_publication_attempt_id(),
        retained=retained,
        reuse=reuse,
    )
    ack = host.admit_background(handle, args, ctx)
    task = handle.task
    assert task is not None
    await asyncio.gather(task, return_exceptions=True)
    return handle, ack


def _short(handle: Any) -> str:
    """The display form this ack must render: ``attempt-<first 8 of uuid>``."""
    attempt = handle.agent_id.rsplit("attempt-", 1)[1]
    return f"attempt-{attempt[:8]}"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_background_ack_shortens_only_the_sentence_handle(tmp_path: Path) -> None:
    """The ack's parenthetical is short; its actionable handle stays full."""
    loop = _root(tmp_path)
    host = AgentChildHost(loop, spawn=_RecordingSpawn())
    try:
        handle, ack = await _launch(host, loop, retained=True)
    finally:
        await host.close()
        await loop.aclose()

    short = _short(handle)
    assert short.startswith("attempt-") and len(short) == len("attempt-") + 8
    assert short == f"attempt-{handle.attempt_id.removeprefix('attempt-')[:8]}"
    # The human-facing parenthetical is short ...
    assert f"({short}) launched and running in its own process" in ack.response
    assert f"({handle.agent_id})" not in ack.response
    # ... while the re-task instruction names the handle a caller must pass.
    assert f'agent_id="{handle.agent_id}"' in ack.response
    assert ack.response.count(handle.agent_id) == 1
    # The structured fields are the programmatic surface: never shortened.
    assert ack.agent_id == handle.agent_id
    assert ack.run_id is not None and ack.status == "launched"
    assert ack.name == f"coder-{handle.seq}"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_background_ack_metadata_is_explicitly_unknown(tmp_path: Path) -> None:
    """The launch ack answers honest unknown, never a guess from the parent."""
    loop = _root(tmp_path)
    host = AgentChildHost(loop, spawn=_RecordingSpawn())
    try:
        _, ack = await _launch(host, loop, retained=True)
    finally:
        await host.close()
        await loop.aclose()

    assert ack.metadata is not None
    assert ack.metadata == {
        "base_model": None,
        "active_provider": None,
        "effective_model": None,
        "effective_thinking": None,
    }
    # Honest unknown renders as unknown: no invented model segment and no
    # projected "models" value for a caller to read as an observation.
    event = ToolResultEvent(
        tool_name="task", tool_class=Task, tool_call_id="task-1", result=ack
    )
    display = Task.get_result_display(event)
    assert display.message == f"agent {ack.agent_id} (run {ack.run_id})"
    assert Task.project_result(ack) is None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_retained_re_task_ack_carries_the_same_honest_metadata(
    tmp_path: Path,
) -> None:
    """A re-task of a retained handle is admitted the same honest-unknown way."""
    loop = _root(tmp_path)
    host = AgentChildHost(loop, spawn=_RecordingSpawn())
    try:
        handle, first = await _launch(host, loop, retained=True)
        args = TaskArgs(
            task="Continue the work", agent_id=handle.agent_id, identity="coder"
        )
        ctx = _ctx(loop, "task-2")
        reused = await host.publish(
            args,
            ctx,
            attempt_id=new_publication_attempt_id(),
            retained=True,
            reuse=handle,
        )
        second = host.admit_background(reused, args, ctx)
        task = handle.task
        assert task is not None
        await asyncio.gather(task, return_exceptions=True)
    finally:
        await host.close()
        await loop.aclose()

    assert reused is handle
    assert second.metadata == first.metadata
    assert (
        second.metadata is not None and set(second.metadata) >= _IN_PROCESS_MODEL_KEYS
    )
    assert all(value is None for value in second.metadata.values())
    # The re-task's structured handle is still the full one.
    assert second.agent_id == handle.agent_id
    assert f'agent_id="{handle.agent_id}"' in second.response


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_in_process_ack_carries_the_same_keys_with_observed_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Parity: the in-process ack populates exactly the keys the process ack
    names as unknown.

    The process route answers the same metadata CONTRACT; only the values
    differ, because no admission-time observation exists on that path.
    """
    from tests.app_server._subagent_test_support import (
        _background_result,
        _zero_retention_registry,
    )

    backend = FakeBackend([[mock_llm_chunk(content="child answer")]])
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    ctx = _ctx(parent, "task-in-process")
    in_process = await _background_result(
        registry, TaskArgs(task="Inspect the project", identity="coder"), ctx
    )
    await parent.aclose()

    assert in_process.metadata is not None
    # The in-process admission observed the child record, so its model keys
    # carry values ...
    assert set(in_process.metadata) >= _IN_PROCESS_MODEL_KEYS
    assert in_process.metadata["base_model"] is not None
    assert in_process.metadata["active_provider"] is not None
    # ... while the process ack's key set is the same contract. Both are
    # pinned together so neither route can drift into a different shape.
    loop_root = _root(tmp_path)
    host = AgentChildHost(loop_root, spawn=_RecordingSpawn())
    try:
        _, process_ack = await _launch(host, loop_root, retained=False)
    finally:
        await host.close()
        await loop_root.aclose()

    assert process_ack.metadata is not None
    assert set(process_ack.metadata) >= _IN_PROCESS_MODEL_KEYS
    assert all(process_ack.metadata[key] is None for key in _IN_PROCESS_MODEL_KEYS)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_inventory_and_results_keep_the_full_handle(tmp_path: Path) -> None:
    """Every programmatic surface still carries the full 40-character handle."""
    loop = _root(tmp_path)
    host = AgentChildHost(loop, spawn=_RecordingSpawn())
    try:
        handle, ack = await _launch(host, loop, retained=True)
        rows = {row.agent_id: row for row in host.inventory_snapshot().agents}
        checked = await host.check_agents()
        ledger = host._run_completion_ledger()
        stored = (handle.agent_id, ack.run_id) in host._results
    finally:
        await host.close()
        await loop.aclose()

    # check_agents, the inventory projection and the completion ledger (the
    # wake text's source) all name the full handle, never the short form.
    assert handle.agent_id in rows
    assert _short(handle) not in rows
    assert [summary.agent_id for summary in checked] == [handle.agent_id]
    assert [entry.agent_id for entry in ledger] == [handle.agent_id]
    # A run-pinned read resolves by the full handle.
    assert stored


def test_short_form_never_collides_with_an_unexpected_handle_shape() -> None:
    """Only the exact ``attempt-<32 hex>`` shape is shortened."""
    # Imported here so the surface pins above stay importable (and therefore
    # behaviorally discriminating) on a tree that predates the helper.
    from chartreux.app_server.root_agent import _short_display_handle

    attempt = uuid4().hex
    assert _short_display_handle(f"coder-attempt-{attempt}") == f"attempt-{attempt[:8]}"
    assert _short_display_handle(f"attempt-{attempt}") == f"attempt-{attempt[:8]}"
    # Unexpected shapes are returned untouched, never mangled.
    for raw in (
        "coder-3",
        f"attempt-{attempt[:8]}",
        f"coder-attempt-{attempt.upper()}",
        "coder-attempt-",
        f"coder-attempt-{attempt[:-1]}",
        "coder",
        "",
    ):
        assert _short_display_handle(raw) == raw
