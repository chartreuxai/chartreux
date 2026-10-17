from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.core.agent_loop.errors import AgentLoopStateError
from chartreux.core.background_jobs import (
    BashListArgs,
    BashReadArgs,
    BashStartArgs,
    BashStopArgs,
    StartResult,
)
from chartreux.core.tools.base import InvokeContext
from tests.conftest import build_test_agent_loop


@pytest.mark.asyncio
async def test_new_launch_after_relocation_uses_new_cwd(tmp_path, monkeypatch):
    loop = build_test_agent_loop(cwd=tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    port = loop.background_jobs
    registry = loop._background_job_registry
    assert port is not None and registry is not None
    ctx = InvokeContext(tool_call_id="cwd", background_jobs=port)

    async def launch():
        tool = loop.tool_manager.get("bash_start")
        result = [
            result async for result in tool.run(BashStartArgs(command="pwd"), ctx)
        ][0]
        assert isinstance(result, StartResult)
        return result

    try:
        old = await launch()
        job = registry._jobs[old.job.job_id]
        assert job.supervisor is not None
        await job.supervisor
        monkeypatch.setattr(loop, "_destination_checkout", lambda _: target)
        await loop.relocate(target)
        assert loop.cwd == target
        new = await launch()
        job = registry._jobs[new.job.job_id]
        assert job.supervisor is not None
        await job.supervisor
        for result, cwd in ((old, tmp_path), (new, target)):
            page = await port.read(BashReadArgs(job_id=result.job.job_id))
            assert "".join(r.text for r in page.records).strip() == str(cwd)
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_child_incarnations_borrow_capacity_and_survive_teardown() -> None:
    root = build_test_agent_loop()
    assert root.background_jobs is not None
    child = build_test_agent_loop(
        is_subagent=True,
        inherited_workspace=root.tool_manager.workspace,
        background_jobs=root.background_jobs.borrow(),
        session_id="stored-child-id",
    )
    resumed = build_test_agent_loop(
        is_subagent=True,
        inherited_workspace=root.tool_manager.workspace,
        background_jobs=root.background_jobs.borrow(),
        session_id="stored-child-id",
    )
    try:
        assert child.background_jobs is not None
        assert resumed.background_jobs is not None
        started = await child.background_jobs.start(BashStartArgs(command="sleep 60"))
        job_id = started.job.job_id
        assert [job.job_id for job in root.background_jobs.list().jobs] == [job_id]
        assert not resumed.background_jobs.list().jobs
        with pytest.raises(ValueError, match="unavailable"):
            resumed.background_jobs.authorize(job_id)
        metadata = child.completion_metadata_since(child.completion_metadata_mark())
        assert metadata["background_jobs"] == [started.job.model_dump(mode="json")]
        await child.aclose()
        assert root.background_jobs.active_count == 1
        await root.background_jobs.stop(BashStopArgs(job_id=job_id))
        assert root.background_jobs.active_count == 0
    finally:
        await resumed.aclose()
        await child.aclose()
        await root.aclose()


@pytest.mark.asyncio
async def test_authority_guard_counts_pending_but_not_finished_records() -> None:
    loop = build_test_agent_loop()
    assert loop.background_jobs is not None
    registry = loop._background_job_registry
    assert registry is not None
    try:
        pending = loop.background_jobs.reserve(BashStartArgs(command="echo pending"))
        cosmetic = loop.config.model_copy(
            update={"ask_confirmation_on_exit": False}, deep=True
        )
        loop._guard_job_authority_config(cosmetic)
        reduction = loop.config.model_copy(
            update={"disabled_tools": [*loop.config.disabled_tools, "bash"]}, deep=True
        )
        with pytest.raises(AgentLoopStateError, match="background jobs"):
            loop._guard_job_authority_config(reduction)
        for field in ("denylist_standalone", "custom_restriction"):
            restricted = loop.config.model_copy(deep=True)
            restricted.tools.setdefault("bash", {})[field] = ["python"]
            with pytest.raises(AgentLoopStateError, match="background jobs"):
                loop._guard_job_authority_config(restricted)
        registry.rollback(pending)
        finished = loop.background_jobs.reserve(BashStartArgs(command="echo finished"))
        registry.commit(finished)
        registry.observe_leader(finished, 0)
        registry.capture(finished, b"done", final=True)
        registry.settle(finished)
        assert loop.background_jobs.list(BashListArgs(include_finished=True)).jobs
        loop._guard_job_authority_config(reduction)
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_relocation_rejects_live_jobs_before_workspace_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = build_test_agent_loop(cwd=tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    assert loop.background_jobs is not None
    try:
        started = await loop.background_jobs.start(BashStartArgs(command="sleep 60"))
        bind = AsyncMock()
        monkeypatch.setattr(loop, "_destination_checkout", lambda _: target)
        monkeypatch.setattr(loop, "_bind_workspace", bind)
        await loop.relocate(tmp_path)
        with pytest.raises(AgentLoopStateError, match="background jobs"):
            await loop.relocate(target)
        bind.assert_not_called()
        assert loop.cwd == tmp_path
        await loop.background_jobs.stop(BashStopArgs(job_id=started.job.job_id))
        await loop.relocate(target)
        bind.assert_awaited_once()
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_reset_invalidates_old_handles_and_ports() -> None:
    loop = build_test_agent_loop()
    old_port = loop.background_jobs
    assert old_port is not None
    try:
        started = await old_port.start(BashStartArgs(command="sleep 60"))
        old_id = loop.session_id
        await loop._reset_session()
        assert loop.session_id != old_id
        assert loop.background_jobs is not None
        assert not loop.background_jobs.list().jobs
        with pytest.raises(ValueError):
            old_port.list()
        with pytest.raises(ValueError):
            loop.background_jobs.authorize(started.job.job_id)
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_cleanup_failure_keeps_registry_scratchpad_and_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    registry = loop._background_job_registry
    assert registry is not None
    scratchpad = loop.scratchpad_dir
    assert scratchpad is not None
    lease = MagicMock()
    loop._session_lease = lease
    close = AsyncMock(side_effect=[RuntimeError("cleanup unsettled"), None])
    monkeypatch.setattr(registry, "aclose", close)
    with pytest.raises(RuntimeError, match="cleanup unsettled"):
        await loop.aclose()
    assert scratchpad.exists()
    assert loop._session_lease is lease
    assert loop._background_job_registry is registry
    lease.release.assert_not_called()
    await loop.aclose()
    lease.release.assert_called_once()
    assert not scratchpad.exists()


@pytest.mark.asyncio
async def test_cancelled_retirement_reopens_admission_after_owned_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    registry = loop._background_job_registry
    assert registry is not None
    entered = asyncio.Event()
    release = asyncio.Event()
    original = registry.aclose

    async def blocked_close() -> None:
        entered.set()
        await release.wait()
        await original()

    monkeypatch.setattr(registry, "aclose", blocked_close)
    waiter = asyncio.create_task(loop.retire_background_jobs())
    await entered.wait()
    waiter.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    try:
        assert loop.background_jobs is not None
        pending = loop.background_jobs.reserve(BashStartArgs(command="echo fresh"))
        assert loop._background_job_registry is not None
        loop._background_job_registry.rollback(pending)
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_transcript_fence_does_not_change_job_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    port = loop.background_jobs
    assert port is not None
    try:
        started = await port.start(BashStartArgs(command="sleep 60"))
        generation = loop._job_lifetime_generation
        loop._fence_rewind_transcript()
        assert loop._job_lifetime_generation == generation
        assert loop.background_jobs is port
        port.authorize(started.job.job_id)
        monkeypatch.setattr(
            loop.compaction_manager, "compact", AsyncMock(return_value="summary")
        )
        await loop.compact()
        assert loop.background_jobs is port
        assert port.active_count == 1
        # A cancelled ordinary waiter does not own registry supervision.
        waiter = asyncio.create_task(
            port.read(
                BashReadArgs(job_id=started.job.job_id, cursor=0, wait_seconds=30)
            )
        )
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert port.active_count == 1
    finally:
        await loop.aclose()
