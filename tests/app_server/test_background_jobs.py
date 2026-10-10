from __future__ import annotations

import asyncio
from pathlib import Path
import shlex
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._root_session import RootSessionCoordinator
from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._session_history import SessionHistory
from chartreux.app_server._state import build_public_state, build_stored_public_state
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import SessionSnapshot
from chartreux.app_server.protocol import (
    ClientInfo,
    ConfigWriteOpWire,
    ConfigWriteParams,
    ConfigWriteResponse,
    SessionReadParams,
    SessionStartParams,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.background_jobs import (
    BackgroundJobRegistry,
    BashReadArgs,
    BashStartArgs,
    BashStopArgs,
)
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.events import AssistantEvent, BackgroundJobsChangedEvent
from tests.app_server.test_wait_steering import steer_params, until, waiting_runtime
from tests.conftest import build_test_agent_loop
from tests.core.agent_loop.test_accepted_source_snapshot import make_orchestrator
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import build_test_app_server, create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend


async def test_backend_cleanup_failure_preserves_worktree_protection_for_retry(
    monkeypatch,
):
    from chartreux.app_server._session_backend_impl import SessionBackendImpl

    order = []
    session = MagicMock()

    async def close():
        order.append("close")
        if len(order) == 1:
            raise RuntimeError("job cleanup failed")

    session.close = AsyncMock(side_effect=close)
    coordinator = MagicMock(attached_session_id=None)
    handler = MagicMock(close=AsyncMock())
    children = MagicMock(close=AsyncMock())
    backend = SessionBackendImpl(
        session, MagicMock(), coordinator, handler, children, MagicMock()
    )
    token = MagicMock()
    backend.worktree_token = token

    def release(self):
        order.append("release")
        self.worktree_token = None

    async def rollback(self):
        order.append("rollback")

    monkeypatch.setattr(SessionBackendImpl, "_release_worktree_holder", release)
    monkeypatch.setattr(SessionBackendImpl, "_roll_back_unstarted_worktree", rollback)
    with pytest.raises(RuntimeError, match="job cleanup failed"):
        await backend.shutdown()
    assert order == ["close"] and backend.worktree_token is token
    assert not backend._closed
    await backend.shutdown()
    assert order == ["close", "close", "release", "rollback"]
    assert backend._closed


pytestmark = pytest.mark.asyncio


async def test_steering_wait_preserves_child_job_supervision() -> None:
    loop, manager, runtime, turn_id = await waiting_runtime()
    assert loop.background_jobs is not None
    child = build_test_agent_loop(
        is_subagent=True,
        inherited_workspace=loop.tool_manager.workspace,
        background_jobs=loop.background_jobs.borrow(),
    )
    try:
        assert child.background_jobs is not None
        started = await child.background_jobs.start(BashStartArgs(command="sleep 60"))
        assert loop.is_waiting_only(turn_id)
        receipt = await runtime.turns.steer(steer_params(loop, turn_id))
        assert receipt.accepted
        await runtime.turns.wait_for_operation(turn_id)
        assert manager.released == ["child"]
        await child.aclose()
        assert loop.background_jobs.active_count == 1
        loop.background_jobs.authorize(started.job.job_id)
        await loop.background_jobs.stop(BashStopArgs(job_id=started.job.job_id))
    finally:
        await child.aclose()
        await runtime.close()
        await loop.aclose()


async def test_config_publication_blocks_authority_not_cosmetic_changes(
    tmp_path: Path,
) -> None:
    root = await make_root(tmp_path)
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(root, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    try:
        await client.initialize(ClientInfo(name="jobs-test", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())
        assert root.background_jobs is not None
        started = await root.background_jobs.start(BashStartArgs(command="sleep 60"))
        token = root.config_orchestrator.accepted_token
        response = await client.request(
            "config/write",
            ConfigWriteParams(
                session_id=root.session_id,
                ops=[
                    ConfigWriteOpWire(op="set", path="/disabled_tools", value=["bash"])
                ],
            ),
        )
        response = ConfigWriteResponse.model_validate(response)
        assert response.failures
        assert root.config_orchestrator.accepted_token is token
        assert "bash" not in root.config.disabled_tools
        cosmetic = await client.request(
            "config/write",
            ConfigWriteParams(
                session_id=root.session_id,
                ops=[
                    ConfigWriteOpWire(
                        op="set", path="/ask_confirmation_on_exit", value=False
                    )
                ],
            ),
        )
        cosmetic = ConfigWriteResponse.model_validate(cosmetic)
        assert not cosmetic.failures
        assert not root.config.ask_confirmation_on_exit
        assert root.background_jobs.active_count == 1
        await root.background_jobs.stop(BashStopArgs(job_id=started.job.job_id))
        response = await client.request(
            "config/write",
            ConfigWriteParams(
                session_id=root.session_id,
                ops=[
                    ConfigWriteOpWire(op="set", path="/disabled_tools", value=["bash"])
                ],
            ),
        )
        response = ConfigWriteResponse.model_validate(response)
        assert not response.failures
        assert "bash" in root.config.disabled_tools
    finally:
        await client.close()
        await server.close()
        await root.aclose()


async def make_root(tmp_path: Path) -> AgentLoop:
    path = tmp_path / "settings.toml"
    path.write_text("")
    orchestrator = await make_orchestrator(path)
    assert not await orchestrator.set_field(
        "/session_logging",
        {"enabled": True, "save_dir": str(tmp_path / "sessions")},
        target_layer=OverridesLayer.NAME,
    )
    root = AgentLoop(
        config_orchestrator=orchestrator, cwd=tmp_path, backend=FakeBackend()
    )
    await root.persist_empty_session()
    return root


async def test_factory_resumed_child_cannot_adopt_previous_incarnation_jobs(
    tmp_path: Path,
) -> None:
    root = await make_root(tmp_path)
    factory = AgentRuntimeFactory()
    child = await factory.create_child(root, "worker")
    resumed = None
    try:
        await child.wait_until_ready()
        await child.persist_empty_session()
        directory = child.session_logger.session_dir
        assert directory is not None
        assert child.background_jobs is not None
        started = await child.background_jobs.start(BashStartArgs(command="sleep 60"))
        await child.aclose()
        resumed = await factory.resume_child(
            root, "worker", child.session_id, directory
        )
        await resumed.wait_until_ready()
        assert resumed.background_jobs is not None
        assert not resumed.background_jobs.list().jobs
        with pytest.raises(ValueError):
            resumed.background_jobs.authorize(started.job.job_id)
        assert root.background_jobs is not None
        assert root.background_jobs.active_count == 1
        await root.background_jobs.stop(BashStopArgs(job_id=started.job.job_id))
    finally:
        if resumed is not None:
            await resumed.aclose()
        await child.aclose()
        await root.aclose()


async def test_fork_isolated_and_resume_retires_old_jobs(tmp_path: Path) -> None:
    root = await make_root(tmp_path)
    factory = AgentRuntimeFactory()
    fork = None
    old_port = root.background_jobs
    assert old_port is not None
    try:
        started = await old_port.start(BashStartArgs(command="sleep 60"))
        fork = await factory.fork(root, None)
        assert fork.background_jobs is not None
        assert not fork.background_jobs.list().jobs
        with pytest.raises(ValueError):
            fork.background_jobs.authorize(started.job.job_id)
        target_id = fork.session_id
        await fork.persist_empty_session()
        await fork.aclose()
        await factory.resume_root(root, target_id)
        assert root.session_id == target_id
        assert root.background_jobs is not None
        assert not root.background_jobs.list().jobs
        with pytest.raises(ValueError):
            old_port.list()
    finally:
        if fork is not None:
            await fork.aclose()
        await root.aclose()


async def test_resume_preparation_failure_retains_identity_and_reopens_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = await make_root(tmp_path)
    factory = AgentRuntimeFactory()
    target = await factory.fork(root, None)
    await target.persist_empty_session()
    await target.aclose()
    old_id = root.session_id
    old_port = root.background_jobs
    assert old_port is not None
    try:
        await old_port.start(BashStartArgs(command="sleep 60"))

        def fail_scratchpad(_: str) -> Path:
            raise OSError("scratchpad unavailable")

        monkeypatch.setattr(root, "prepare_scratchpad_for_session", fail_scratchpad)
        with pytest.raises(OSError, match="scratchpad unavailable"):
            await factory.resume_root(root, target.session_id)
        assert root.session_id == old_id
        assert root.background_jobs is not None
        assert not root.background_jobs.list().jobs
        with pytest.raises(ValueError):
            old_port.list()
        started = await root.background_jobs.start(BashStartArgs(command="sleep 60"))
        await root.background_jobs.stop(BashStopArgs(job_id=started.job.job_id))
    finally:
        await root.aclose()


async def test_idle_job_publication_child_close_and_turn_admission() -> None:
    root = build_test_agent_loop(backend=FakeBackend(mock_llm_chunk(content="ok")))
    session = await create_test_app_server_session(root)
    child = build_test_agent_loop(
        is_subagent=True,
        inherited_workspace=root.tool_manager.workspace,
        background_jobs=root.background_jobs.borrow() if root.background_jobs else None,
    )
    snapshots = []

    async def collect() -> None:
        async for event in session.events():
            if isinstance(event, SessionSnapshot):
                snapshots.append(event.state)

    collector = asyncio.create_task(collect())
    try:
        assert session.state.active_background_job_count == 0
        assert root.background_jobs is not None
        first = await root.background_jobs.start(BashStartArgs(command="sleep 60"))
        await until(lambda: session.state.active_background_job_count == 1)
        assert child.background_jobs is not None
        second = await child.background_jobs.start(
            BashStartArgs(command="printf 'hello!\\n'; sleep 60")
        )
        await child.aclose()
        await until(lambda: session.state.active_background_job_count == 2)
        assert session.state.is_quiescent is False
        page = await root.background_jobs.read(
            BashReadArgs(job_id=second.job.job_id, wait_seconds=1)
        )
        assert page.records
        # A request/response barrier flushes prior notifications. Output has
        # arrived but only the two admissions have published state snapshots.
        client = session._connection.current
        assert client is not None
        await client.request(
            "session/read", SessionReadParams(session_id=root.session_id)
        )
        assert [state.active_background_job_count for state in snapshots] == [1, 2]
        collector.cancel()
        await asyncio.gather(collector, return_exceptions=True)
        async for event in session.act("a live server must not block this turn"):
            if isinstance(event, SessionSnapshot):
                snapshots.append(event.state)
        collector = asyncio.create_task(collect())
        assert root.background_jobs.active_count == 2
        await root.background_jobs.stop(BashStopArgs(job_id=first.job.job_id))
        await until(lambda: session.state.active_background_job_count == 1)
        await root.background_jobs.stop(BashStopArgs(job_id=second.job.job_id))
        await until(lambda: session.state.active_background_job_count == 0)
        await until(lambda: snapshots[-1].active_background_job_count == 0)
        assert session.state.is_quiescent is True
        counts = [state.active_background_job_count for state in snapshots]
        transitions = [
            count
            for index, count in enumerate(counts)
            if index == 0 or count != counts[index - 1]
        ]
        assert transitions == [1, 2, 1, 0]
    finally:
        collector.cancel()
        await asyncio.gather(collector, return_exceptions=True)
        await child.aclose()
        await session.close()
        await root.aclose()


async def test_child_only_natural_finalization_after_child_close(
    tmp_path: Path,
) -> None:
    root = build_test_agent_loop(cwd=tmp_path)
    session = await create_test_app_server_session(root)
    assert root.background_jobs is not None
    child = build_test_agent_loop(
        is_subagent=True,
        inherited_workspace=root.tool_manager.workspace,
        background_jobs=root.background_jobs.borrow(),
    )
    try:
        assert child.background_jobs is not None
        started = await child.background_jobs.start(
            BashStartArgs(
                command=(
                    "printf 'ready!'; while ! test -f "
                    f"{shlex.quote(str(tmp_path / 'release'))}; do sleep 0.01; done"
                )
            )
        )
        await until(lambda: session.state.active_background_job_count == 1)
        page = await root.background_jobs.read(
            BashReadArgs(job_id=started.job.job_id, wait_seconds=1)
        )
        assert page.records
        await child.aclose()
        assert session.state.active_background_job_count == 1
        inactive = build_public_state(
            child,
            history=[],
            current_history=[],
            callbacks=[],
            turns=[],
            retrying=None,
            history_limit=200,
        )
        assert inactive.active_background_job_count == 0
        (tmp_path / "release").write_text("")
        await until(lambda: session.state.active_background_job_count == 0)
        assert session.state.is_quiescent is True
        assert root.background_jobs.list().jobs == ()
    finally:
        await child.aclose()
        await session.close()
        await root.aclose()


async def test_turn_delivered_job_state_is_deduplicated_not_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = build_test_agent_loop()
    assert root.background_jobs is not None
    changed = BackgroundJobsChangedEvent(
        root_lifetime_id=root.background_jobs.lifetime_id,
        generation=root.background_jobs.generation,
        revision=1,
        active_count=1,
    )

    async def act(*args, **kwargs):
        yield changed
        yield changed
        yield changed.model_copy(update={"root_lifetime_id": "stale", "revision": 100})
        yield changed.model_copy(update={"revision": 2, "active_count": 0})
        yield AssistantEvent(content="done", message_id="assistant")

    monkeypatch.setattr(root, "act", act)
    session = await create_test_app_server_session(root)
    try:
        counts = []
        async for event in session.act("test"):
            if isinstance(event, SessionSnapshot):
                counts.append(event.state.active_background_job_count)
        assert counts.count(1) == 1
        assert counts[-1] == 0
        assert len(session.history) == 1
    finally:
        await session.close()
        await root.aclose()


async def test_job_state_fences_deduplication_output_silence_and_stored_zero(
    tmp_path: Path,
) -> None:
    root = await make_root(tmp_path)
    coordinator = RootSessionCoordinator(
        root, AsyncMock(), AsyncMock(), lambda session_id: 0, SessionHistory([])
    )
    events: list[BackgroundJobsChangedEvent] = []
    registry = BackgroundJobRegistry(event_sink=events.append)
    port = registry.root_port()
    try:
        pending = port.reserve(BashStartArgs(command="synthetic"))
        assert events == []
        registry.rollback(pending)
        assert events == []
        job_id = port.reserve(BashStartArgs(command="synthetic"))
        registry.commit(job_id)
        registry.capture(job_id, b"safe output\n", final=True)
        registry.observe_leader(job_id, 0)
        assert len(events) == 1
        registry.settle(job_id)
        registry.settle(job_id)
        assert [event.active_count for event in events] == [1, 0]
        assert root.background_jobs is not None
        current = BackgroundJobsChangedEvent(
            root_lifetime_id=root.background_jobs.lifetime_id,
            generation=root.background_jobs.generation,
            revision=2,
            active_count=2,
        )
        assert coordinator.update_background_jobs(current)
        assert not coordinator.update_background_jobs(current)
        for update in (
            {"revision": 1},
            {"root_lifetime_id": "retired"},
            {"generation": current.generation + 1},
        ):
            assert not coordinator.update_background_jobs(
                current.model_copy(update=update)
            )
        assert coordinator.active_background_job_count == 2
        # Transcript fencing is deliberately independent of managed-job lifetime.
        root._session_generation += 1
        assert coordinator.update_background_jobs(
            current.model_copy(update={"revision": 3, "active_count": 0})
        )
        assert coordinator.is_quiescent
        metadata = root.session_logger.session_metadata
        assert metadata is not None
        stored = build_stored_public_state(
            root.session_id, [], metadata, history_limit=200
        )
        assert stored.active_background_job_count == 0
        await root.retire_background_jobs()
        assert not coordinator.update_background_jobs(
            current.model_copy(update={"revision": 100})
        )
        assert coordinator.active_background_job_count == 0
    finally:
        await registry.aclose()
        await root.aclose()
