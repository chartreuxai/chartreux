from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from chartreux.app_server import _runtime as runtime
from chartreux.app_server._session_runtime_impl import SessionRuntimeControllerImpl
from chartreux.app_server._sessions import AgentRecord, SessionRuntimeRegistry
from chartreux.app_server._worktree_session import WorktreeResolution
from chartreux.app_server.protocol import (
    AgentConfig,
    ClientCapabilities,
    ClientInfo,
    SessionOptions,
    SessionStartParams,
)
from chartreux.core._usage_startup import (
    StartupAccountingContext,
    create_startup_accounting_context,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm import utility_completion
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm_models import LLMChunk
from chartreux.core.session.saved_sessions import delete_saved_session
from chartreux.core.session_types import SessionMetadata
from chartreux.core.usage import (
    CoverageWarning,
    CoverageWarningCode,
    SnapshotState,
    UsageOutcome,
    UsagePrices,
    UsagePurpose,
    UsageRecord,
    UsageState,
    UsageWriteDisposition,
    UsageWriteResult,
)
from chartreux.core.usage_project import resolve_project_key_async
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


class AttemptBackend(FakeBackend):
    async def complete(self, **kwargs: Any) -> LLMChunk:
        notify_request_started()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        notify_request_started()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


async def open_test_root(
    process: runtime.HarnessProcess,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: AttemptBackend,
    *,
    logging_enabled: bool,
    startup_accounting: StartupAccountingContext | None = None,
    startup_accounting_claimed: bool = False,
) -> AgentLoop:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=logging_enabled,
            save_dir=str(tmp_path / "transcripts"),
            generate_titles=False,
        )
    )
    monkeypatch.setattr(
        runtime,
        "build_default_orchestrator",
        AsyncMock(return_value=FakeConfigOrchestrator(config)),
    )
    monkeypatch.setattr(
        process, "_build_mcp_registry_impl", AsyncMock(return_value=FakeMCPRegistry())
    )
    real_agent_loop = runtime.AgentLoop

    def build_loop(**kwargs: Any) -> AgentLoop:
        return real_agent_loop(backend=backend, **kwargs)

    monkeypatch.setattr(runtime, "AgentLoop", build_loop)
    return await process.open_root(
        runtime.RootOpenRequest(
            options=SessionOptions(cwd=str(tmp_path), headless=True),
            client_info=ClientInfo(name="usage-test", version="1"),
            startup_accounting=startup_accounting,
            startup_accounting_claimed=startup_accounting_claimed,
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("logging_enabled", [True, False])
async def test_root_turn_records_attribution_independent_of_transcripts(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    logging_enabled: bool,
) -> None:
    process = runtime.HarnessProcess()
    backend = AttemptBackend([mock_llm_chunk(content="answer")])
    root = await open_test_root(
        process, monkeypatch, tmp_path, backend, logging_enabled=logging_enabled
    )
    session = await create_test_app_server_session(root)
    try:
        async for _event in session.act("hello"):
            pass
        owner = root.accounting_owner
        assert owner is not None
        path = config_dir / "usage" / root.session_id / "usage.jsonl"
        records = [
            UsageRecord.model_validate_json(line)
            for line in path.read_text().splitlines()
        ]
        assert len(records) == len(backend.requests_messages) == 1
        record = records[0]
        model = root.config.get_active_model()
        assert record.root_session_id == record.session_id == root.session_id
        assert record.parent_session_id is None
        assert record.agent_role == "root"
        assert record.agent_profile is None
        assert (
            record.project_key
            == owner.project_key
            == await resolve_project_key_async(tmp_path)
        )
        assert record.purpose == UsagePurpose.CONVERSATION
        assert (record.model, record.provider, record.wire_name) == (
            model.alias,
            model.provider,
            model.name,
        )
        assert owner.writer is process.root_usage_writer(root.session_id)
        with pytest.raises(FrozenInstanceError):
            setattr(owner, "project_key", "changed")  # noqa: B010
        # aread consumes local invalidation, without an external reconciliation.
        snapshot = await process.usage_service.aread()
        assert snapshot.selected.request_count == 1
    finally:
        await session.close()
        await process.retire_root_usage_writer(root.session_id)
        await process.close()


@pytest.mark.asyncio
async def test_standalone_agent_loop_keeps_accounting_disabled(
    config_dir: Path,
) -> None:
    backend = AttemptBackend([mock_llm_chunk(content="answer")])
    root = build_test_agent_loop(backend=backend)
    session = await create_test_app_server_session(root)
    try:
        assert root.accounting_owner is None
        assert root._call_resources(backend).accounting_sink is None
        async for _event in session.act("hello"):
            pass
        assert len(backend.requests_messages) == 1
        assert not (config_dir / "usage").exists()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_process_starts_shared_service_and_allocates_writers_lazily() -> None:
    process = runtime.HarnessProcess()
    try:
        assert process._root_usage_writers == {}
        assert process.usage_service._initial is not None
        # start is idempotent: construction already scheduled this readiness task.
        assert process.usage_service.start() is process.usage_service.start()
        await process.usage_service.wait_ready()
        assert process.usage_service.state == SnapshotState.READY
        first = process.root_usage_writer("root-a")
        assert process.root_usage_writer("root-a") is first
        assert process.root_usage_writer("root-b") is not first
        assert len(process._root_usage_writers) == 2
        await process.retire_root_usage_writer("root-a")
        assert "root-a" not in process._root_usage_writers
        assert first not in process.usage_service._writers
        assert len(process.usage_service._writers) == 1
        await process.retire_root_usage_writer("root-b")
        assert not process.usage_service._writers
        for _ in range(3):
            reopened = process.root_usage_writer("root-a")
            assert reopened is not first
            assert len(process.usage_service._writers) == 1
            await process.retire_root_usage_writer("root-a")
            assert not process.usage_service._writers
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_failed_writer_close_keeps_service_subscription_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = runtime.HarnessProcess()
    writer = process.root_usage_writer("root-a")
    close_writer = writer.aclose
    monkeypatch.setattr(
        writer, "aclose", AsyncMock(side_effect=OSError("close failed"))
    )
    try:
        with pytest.raises(OSError, match="close failed"):
            await process.retire_root_usage_writer("root-a")
        assert process._root_usage_writers["root-a"] is writer
        assert writer in process.usage_service._writers
        monkeypatch.setattr(writer, "aclose", close_writer)
        await process.retire_root_usage_writer("root-a")
        assert writer not in process.usage_service._writers
        assert "root-a" not in process._root_usage_writers
    finally:
        monkeypatch.setattr(writer, "aclose", close_writer)
        await process.close()


@pytest.mark.asyncio
async def test_sink_failure_warns_without_failing_or_retrying_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    process = runtime.HarnessProcess()
    backend = AttemptBackend([mock_llm_chunk(content="answer")])
    root = await open_test_root(
        process, monkeypatch, tmp_path, backend, logging_enabled=False
    )
    owner = root.accounting_owner
    assert owner is not None
    monkeypatch.setattr(
        owner.writer, "append", AsyncMock(side_effect=OSError("private"))
    )
    session = await create_test_app_server_session(root)
    try:
        async for _event in session.act("hello"):
            pass
        assert len(backend.requests_messages) == 1
        assert "recorded usage coverage is degraded" in caplog.text
        assert "private" not in caplog.text
    finally:
        await session.close()
        await process.retire_root_usage_writer(root.session_id)
        await process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("logging_enabled", [True, False])
async def test_children_share_root_ledger_with_nested_attribution_and_independent_close(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    logging_enabled: bool,
) -> None:
    process = runtime.HarnessProcess()
    backend = AttemptBackend([
        [mock_llm_chunk(content="child answer")],
        [mock_llm_chunk(content="grandchild answer")],
        [mock_llm_chunk(content="sibling answer")],
    ])
    root = await open_test_root(
        process, monkeypatch, tmp_path, backend, logging_enabled=logging_enabled
    )
    sessions = []
    try:
        await root.persist_empty_session()
        owner = root.accounting_owner
        assert owner is not None
        child = await process.runtime_factory.create_child(root, "worker")
        child_session = await create_test_app_server_session(child)
        sessions.append(child_session)
        async for _event in child_session.act("hello child"):
            pass
        grandchild = await process.runtime_factory.create_child(child, "worker")
        grandchild_session = await create_test_app_server_session(grandchild)
        sessions.append(grandchild_session)
        async for _event in grandchild_session.act("hello grandchild"):
            pass
        sibling = await process.runtime_factory.create_child(root, "worker")
        sibling_session = await create_test_app_server_session(sibling)
        sessions.append(sibling_session)
        assert child.accounting_owner is grandchild.accounting_owner is owner
        assert sibling.accounting_owner is owner
        if not logging_enabled:
            assert root.session_logger.session_dir is None
            assert child.session_logger.session_dir is None
            assert grandchild.session_logger.session_dir is None
        await child_session.close()
        # A sibling still uses exactly the writer held by the closed child.
        async for _event in sibling_session.act("hello sibling"):
            pass
        path = config_dir / "usage" / root.session_id / "usage.jsonl"
        records = [
            UsageRecord.model_validate_json(line)
            for line in path.read_text().splitlines()
        ]
        assert len(records) == len(backend.requests_messages) == 3
        assert [
            (record.session_id, record.parent_session_id) for record in records
        ] == [
            (child.session_id, root.session_id),
            (grandchild.session_id, child.session_id),
            (sibling.session_id, root.session_id),
        ]
        for record in records:
            assert record.root_session_id == root.session_id
            assert record.project_key == owner.project_key
            assert record.agent_role == "subagent"
            assert record.agent_profile == "worker"
        assert list((config_dir / "usage").glob("*/usage.jsonl")) == [path]
    finally:
        for session in reversed(sessions):
            await session.close()
        await runtime.close_agent_loop(root)
        await process.retire_root_usage_writer(root.session_id)
        await process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_task_scratchpad_access_and_empty_failure_settle_usage(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty: bool
) -> None:
    import json

    from chartreux.core.llm_models import FunctionCall, ToolCall
    from chartreux.core.subagents import TaskArgs, TaskResult
    from chartreux.core.tools.base import InvokeContext
    from chartreux.core.tools.models import ToolPermission

    process = runtime.HarnessProcess()
    backend = AttemptBackend()
    root = await open_test_root(
        process, monkeypatch, tmp_path, backend, logging_enabled=False
    )
    registry = SessionRuntimeRegistry(
        AsyncMock(), AsyncMock(), lambda _: 0, runtime_factory=process.runtime_factory
    )
    root_runtime = registry._build_child_runtime(root)
    root_runtime.turns._projector = MagicMock()
    root_runtime.turns.link_subagent = AsyncMock()
    registry.bind_root(root_runtime)
    assert root.scratchpad_dir is not None
    script = root.scratchpad_dir / "ledger-script.py"

    def call(name: str, args: dict[str, str], index: int) -> LLMChunk:
        return mock_llm_chunk(
            tool_calls=[
                ToolCall(
                    id=f"scratch-{index}",
                    index=0,
                    function=FunctionCall(name=name, arguments=json.dumps(args)),
                )
            ]
        )

    if not empty:
        backend._streams = [
            [
                call(
                    "write_file",
                    {"file_path": str(script), "content": "print('scratch-ok')\n"},
                    1,
                )
            ],
            [call("read_file", {"file_path": str(script)}, 2)],
            [call("bash", {"command": f"python {script}"}, 3)],
            [mock_llm_chunk(content="scratch complete")],
        ]
    try:
        # These paths are outside cwd: access must come from the live parent grant.
        for name, args in [
            ("write_file", {"file_path": str(script), "content": "x"}),
            ("read_file", {"file_path": str(script)}),
            ("bash", {"command": f"python {script}"}),
        ]:
            tool = root.tool_manager.get(name)
            permission = tool.resolve_permission(tool.validate_arguments(args))
            assert permission is not None
            assert permission.permission == ToolPermission.ALWAYS
        stream = registry.run(
            TaskArgs(
                task="Use the session scratchpad", agent_type="worker", background=False
            ),
            InvokeContext(tool_call_id="merged-task", session_id=root.session_id),
        )
        if empty:
            with pytest.raises(RuntimeError, match="empty assistant response"):
                async for _item in stream:
                    pass
        else:
            results = [item async for item in stream if isinstance(item, TaskResult)]
            assert results[-1].completed
            assert script.read_text() == "print('scratch-ok')\n"
            # Tool output is retained in the backend's next request after execution.
            messages = backend.requests_messages[-1]
            for call_id in ("scratch-2", "scratch-3"):
                assert any(
                    message.content and "scratch-ok" in message.content
                    for message in messages
                    if message.tool_call_id == call_id
                )
        await registry.drain_children()
        records = read_ledger(config_dir, root.session_id)
        assert len(records) == len(backend.requests_messages)
        assert records
        assert all(record.root_session_id == root.session_id for record in records)
        assert all(record.parent_session_id == root.session_id for record in records)
        assert all(
            record.agent_role == "subagent" and record.agent_profile == "worker"
            for record in records
        )
        assert all(record.session_id != root.session_id for record in records)
        assert len({record.session_id for record in records}) == 1
        if empty:
            assert all(record.outcome == UsageOutcome.FAILED for record in records)
        else:
            assert all(record.outcome == UsageOutcome.COMPLETED for record in records)
        snapshot = await process.usage_service.aread()
        assert snapshot.selected.request_count == len(records)
    finally:
        await registry.drain_children()
        await root_runtime.close()
        await process.close()


@pytest.mark.asyncio
async def test_resumed_child_inherits_explicit_root_owner(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    backend = AttemptBackend([
        [mock_llm_chunk(content="original answer")],
        [mock_llm_chunk(content="resumed answer")],
    ])
    root = await open_test_root(
        process, monkeypatch, tmp_path, backend, logging_enabled=True
    )
    sessions = []
    try:
        await root.persist_empty_session()
        child = await process.runtime_factory.create_child(root, "worker")
        child_session = await create_test_app_server_session(child)
        sessions.append(child_session)
        async for _event in child_session.act("original task"):
            pass
        child_id = child.session_id
        child_dir = child.session_logger.session_dir
        assert child_dir is not None
        await child_session.close()
        resumed = await process.runtime_factory.resume_child(
            root, "worker", child_id, child_dir
        )
        assert resumed.accounting_owner is root.accounting_owner
        resumed_session = await create_test_app_server_session(resumed)
        sessions.append(resumed_session)
        async for _event in resumed_session.act("resume task"):
            pass
        path = config_dir / "usage" / root.session_id / "usage.jsonl"
        records = [
            UsageRecord.model_validate_json(line)
            for line in path.read_text().splitlines()
        ]
        assert len(records) == len(backend.requests_messages) == 2
        for record in records:
            assert record.root_session_id == root.session_id
            assert record.session_id == child_id
            assert record.parent_session_id == root.session_id
            assert record.agent_role == "subagent"
            assert record.agent_profile == "worker"
            assert root.accounting_owner is not None
            assert record.project_key == root.accounting_owner.project_key
    finally:
        for session in reversed(sessions):
            await session.close()
        await runtime.close_agent_loop(root)
        await process.retire_root_usage_writer(root.session_id)
        await process.close()


@pytest.mark.asyncio
async def test_late_child_finalization_keeps_invocation_local_attribution(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class GatedBackend(AttemptBackend):
        def __init__(self) -> None:
            super().__init__([mock_llm_chunk(content="late answer")])
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def complete(self, **kwargs: Any) -> LLMChunk:
            notify_request_started()
            self.started.set()
            await self.release.wait()
            return await super().complete(**kwargs)

    process = runtime.HarnessProcess()
    backend = GatedBackend()
    root = await open_test_root(
        process, monkeypatch, tmp_path, backend, logging_enabled=False
    )
    child = await process.runtime_factory.create_child(root, "worker")
    original_id = child.session_id
    original_parent = child.parent_session_id
    task = None
    try:
        await child.wait_until_ready()
        resources = child._call_resources(backend)
        inputs = child._completion_inputs(
            model=child.config.get_active_model(),
            messages=child.messages,
            tools=None,
            tool_choice=None,
        )
        task = asyncio.create_task(child._llm_gateway.complete(inputs, resources))
        await asyncio.wait_for(backend.started.wait(), 2)
        # Simulate identity moving on without implementing WP6c's rebind lifecycle.
        child.session_id = "later-session"
        child.parent_session_id = "later-parent"
        child.launch_profile = "later-profile"
        backend.release.set()
        await task
        path = config_dir / "usage" / root.session_id / "usage.jsonl"
        records = [
            UsageRecord.model_validate_json(line)
            for line in path.read_text().splitlines()
        ]
        assert len(records) == 1
        record = records[0]
        assert record.session_id == original_id
        assert record.parent_session_id == original_parent == root.session_id
        assert record.agent_profile == "worker"
        assert record.agent_role == "subagent"
        assert record.root_session_id == root.session_id
        assert root.accounting_owner is not None
        assert record.project_key == root.accounting_owner.project_key
        current = child._call_resources(backend).usage_attribution
        assert current is not None
        assert current.session_id == "later-session"
        assert current.parent_session_id == "later-parent"
        assert current.agent_profile == "later-profile"
    finally:
        backend.release.set()
        if task is not None:
            await task
        child.session_id = original_id
        child.parent_session_id = original_parent
        child.launch_profile = "worker"
        await runtime.close_agent_loop(child)
        await runtime.close_agent_loop(root)
        await process.retire_root_usage_writer(root.session_id)
        await process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("blueprint_resume", [True, False])
@pytest.mark.parametrize("retire_before_resume", [True, False])
async def test_root_resume_appends_to_existing_ledger(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blueprint_resume: bool,
    retire_before_resume: bool,
) -> None:
    process = runtime.HarnessProcess()
    backend = AttemptBackend([
        [mock_llm_chunk(content="original answer")],
        [mock_llm_chunk(content="resumed answer")],
    ])
    original = await open_test_root(
        process, monkeypatch, tmp_path, backend, logging_enabled=True
    )
    session = await create_test_app_server_session(original)
    resumed = None
    try:
        async for _event in session.act("original turn"):
            pass
        root_id = original.session_id
        owner = original.accounting_owner
        assert owner is not None
        await session.close()
        if retire_before_resume:
            await process.retire_root_usage_writer(root_id)
            assert owner.writer not in process.usage_service._writers
        request = runtime.RootOpenRequest(
            options=SessionOptions(cwd=str(tmp_path), headless=True),
            client_info=ClientInfo(name="usage-test", version="1"),
            session_id=root_id if blueprint_resume else None,
        )
        resumed = await process.open_root(request)
        if not blueprint_resume:
            await process.runtime_factory.resume_root(resumed, root_id)
        resumed_owner = resumed.accounting_owner
        assert resumed_owner is not None
        assert (resumed_owner is owner) is (not retire_before_resume)
        assert (resumed_owner.writer is owner.writer) is (not retire_before_resume)
        assert resumed_owner.writer is process.root_usage_writer(root_id)
        assert resumed_owner.writer in process.usage_service._writers
        session = await create_test_app_server_session(resumed)
        async for _event in session.act("resumed turn"):
            pass
        path = config_dir / "usage" / root_id / "usage.jsonl"
        records = [
            UsageRecord.model_validate_json(line)
            for line in path.read_text().splitlines()
        ]
        assert len(records) == len(backend.requests_messages) == 2
        assert all(record.root_session_id == root_id for record in records)
        assert all(record.session_id == root_id for record in records)
        assert all(record.project_key == owner.project_key for record in records)
        assert list((config_dir / "usage").glob("*/usage.jsonl")) == [path]
    finally:
        await session.close()
        if resumed is not None:
            await runtime.close_agent_loop(resumed)
        for root_id in list(process._root_usage_writers):
            await process.retire_root_usage_writer(root_id)
        await process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("new_identity", [True, False])
async def test_rebind_freezes_late_invocation_and_cancelled_title_sink(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    new_identity: bool,
) -> None:
    class GatedBackend(AttemptBackend):
        def __init__(self) -> None:
            super().__init__([
                [mock_llm_chunk(content="late answer")],
                [mock_llm_chunk(content="new answer")],
            ])
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def complete(self, **kwargs: Any) -> LLMChunk:
            notify_request_started()
            self.started.set()
            await self.release.wait()
            return await super().complete(**kwargs)

    process = runtime.HarnessProcess()
    backend = GatedBackend()
    root = await open_test_root(
        process, monkeypatch, tmp_path, backend, logging_enabled=False
    )
    old_id = root.session_id
    old_owner = root.accounting_owner
    assert old_owner is not None
    target_id = "rebound-root" if new_identity else old_id
    task = None
    title_task = None
    title_started = asyncio.Event()
    title_cancelled = asyncio.Event()
    title_release = asyncio.Event()
    try:
        await root.wait_until_ready()
        resources = root._call_resources(backend)
        attribution = resources.usage_attribution
        sink = resources.accounting_sink
        assert attribution is not None and sink is not None
        title_record = UsageRecord(
            **attribution.model_dump(exclude={"purpose"}),
            purpose=UsagePurpose.TITLE,
            record_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            outcome=UsageOutcome.COMPLETED,
            usage_state=UsageState.MISSING,
            prices_usd_per_million=UsagePrices(),
            known_cost_usd=0,
            has_unknown_cost=True,
        )

        async def stale_title() -> None:
            title_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                title_cancelled.set()
                await title_release.wait()
                await sink(title_record)

        title_task = asyncio.create_task(stale_title())
        root._title_controller._task = title_task
        await title_started.wait()
        inputs = root._completion_inputs(
            model=root.config.get_active_model(),
            messages=root.messages,
            tools=None,
            tool_choice=None,
        )
        task = asyncio.create_task(root._llm_gateway.complete(inputs, resources))
        await asyncio.wait_for(backend.started.wait(), 2)
        replacement = await process.runtime_factory.prepare_rebind_accounting_owner(
            root, target_id
        )
        metadata = SessionMetadata(
            session_id=target_id,
            start_time="2026-01-01T00:00:00Z",
            end_time=None,
            git_commit=None,
            git_branch=None,
            environment={},
            username="test",
        )
        if new_identity:
            with pytest.raises(ValueError, match="Prepare accounting"):
                root.rebind_to_session(
                    target_id, tmp_path, [], session_metadata=metadata
                )
            assert root.session_id == old_id
            assert root.accounting_owner is old_owner
        root.rebind_to_session(
            target_id,
            tmp_path,
            [],
            session_metadata=metadata,
            prepared_scratchpad=None,
            accounting_owner=replacement,
        )
        assert root.accounting_owner is replacement
        assert (replacement is old_owner) is (not new_identity)
        assert replacement is not None
        assert (replacement.writer is old_owner.writer) is (not new_identity)
        await asyncio.wait_for(title_cancelled.wait(), 2)
        assert root._auto_title_task is None
        assert title_task in root._detached_accounting_producers
        # A drain here has no queued writes yet and must not retire the old sink.
        await old_owner.writer.drain()
        backend.release.set()
        await task
        title_release.set()
        await title_task
        await root._llm_gateway.complete(inputs, root._call_resources(backend))
        old_path = config_dir / "usage" / old_id / "usage.jsonl"
        new_path = config_dir / "usage" / target_id / "usage.jsonl"
        old_records = [
            UsageRecord.model_validate_json(line)
            for line in old_path.read_text().splitlines()
        ]
        assert len(old_records) == (2 if new_identity else 3)
        assert all(record.root_session_id == old_id for record in old_records)
        assert all(record.session_id == old_id for record in old_records)
        assert all(
            record.project_key == old_owner.project_key for record in old_records
        )
        assert old_records[1].purpose == UsagePurpose.TITLE
        if new_identity:
            new_records = [
                UsageRecord.model_validate_json(line)
                for line in new_path.read_text().splitlines()
            ]
            assert len(new_records) == 1
            assert new_records[0].root_session_id == target_id
            assert new_records[0].session_id == target_id
        assert title_task not in root._detached_accounting_producers
    finally:
        backend.release.set()
        title_release.set()
        if task is not None:
            await task
        if title_task is not None:
            await title_task
        await runtime.close_agent_loop(root)
        for root_id in list(process._root_usage_writers):
            await process.retire_root_usage_writer(root_id)
        await process.close()


class LifecycleGatedBackend(AttemptBackend):
    def __init__(self) -> None:
        super().__init__([
            [mock_llm_chunk(content="late answer")],
            [mock_llm_chunk(content="new answer")],
        ])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, **kwargs: Any) -> LLMChunk:
        notify_request_started()
        self.started.set()
        await self.release.wait()
        return await super().complete(**kwargs)


async def start_gated_completion(
    root: AgentLoop, backend: LifecycleGatedBackend
) -> asyncio.Task[LLMChunk]:
    await root.wait_until_ready()
    inputs = root._completion_inputs(
        model=root.config.get_active_model(),
        messages=root.messages,
        tools=None,
        tool_choice=None,
    )
    task = asyncio.create_task(
        root._llm_gateway.complete(inputs, root._call_resources(backend))
    )
    await asyncio.wait_for(backend.started.wait(), 2)
    return task


def read_ledger(config_dir: Path, root_id: str) -> list[UsageRecord]:
    return [
        UsageRecord.model_validate_json(line)
        for line in (config_dir / "usage" / root_id / "usage.jsonl")
        .read_text()
        .splitlines()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("logging_enabled", [True, False])
async def test_fork_owns_ledger_and_preserves_source_inflight_attribution(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    logging_enabled: bool,
) -> None:
    process = runtime.HarnessProcess()
    backend = LifecycleGatedBackend()
    source = await open_test_root(
        process, monkeypatch, tmp_path, backend, logging_enabled=logging_enabled
    )
    owner = source.accounting_owner
    assert owner is not None
    task = None
    forked = None
    session = None
    try:
        task = await start_gated_completion(source, backend)
        forked = await process.runtime_factory.fork(source, None)
        fork_owner = forked.accounting_owner
        assert fork_owner is not None and fork_owner is not owner
        assert fork_owner.root_session_id == forked.session_id != source.session_id
        assert fork_owner.project_key == owner.project_key
        assert fork_owner.writer is process.root_usage_writer(forked.session_id)
        assert fork_owner.writer is not owner.writer
        assert forked.parent_session_id == source.session_id
        assert source.accounting_owner is owner
        backend.release.set()
        await task
        session = await create_test_app_server_session(forked)
        async for _event in session.act("fork turn"):
            pass
        source_records = read_ledger(config_dir, source.session_id)
        fork_records = read_ledger(config_dir, forked.session_id)
        assert len(source_records) == len(fork_records) == 1
        assert source_records[0].root_session_id == source.session_id
        assert source_records[0].session_id == source.session_id
        assert source_records[0].parent_session_id is None
        record = fork_records[0]
        assert record.root_session_id == record.session_id == forked.session_id
        assert record.parent_session_id == source.session_id
        assert record.agent_role == "root"
        assert record.project_key == owner.project_key
        assert len(list((config_dir / "usage").glob("*/usage.jsonl"))) == 2
    finally:
        backend.release.set()
        if task is not None:
            await task
        if session is not None:
            await session.close()
        elif forked is not None:
            await runtime.close_agent_loop(forked)
        await runtime.close_agent_loop(source)
        for root_id in list(process._root_usage_writers):
            await process.retire_root_usage_writer(root_id)
        await process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("preserve_identity", [True, False])
@pytest.mark.parametrize("logging_enabled", [True, False])
async def test_clear_history_prepares_owner_and_freezes_inflight_attribution(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preserve_identity: bool,
    logging_enabled: bool,
) -> None:
    process = runtime.HarnessProcess()
    backend = LifecycleGatedBackend()
    root = await open_test_root(
        process, monkeypatch, tmp_path, backend, logging_enabled=logging_enabled
    )
    old_id = root.session_id
    old_owner = root.accounting_owner
    assert old_owner is not None
    task = None
    session = None
    try:
        task = await start_gated_completion(root, backend)
        if preserve_identity:
            # Production clear mints an ID; exercise the identity-preserving seam
            # too, so an unchanged root binding never rotates its ledger.
            # Release the old lease only for this synthetic same-ID reset: the
            # real minting path necessarily acquires a different session lease.
            root.replace_session_lease(None)
            monkeypatch.setattr(
                "chartreux.core.agent_loop._loop.generate_session_id",
                lambda **kwargs: old_id,
            )
        real_reset = root.session_logger.reset_session

        def check_prepared(session_id: str, **kwargs: Any) -> None:
            assert session_id in process._root_accounting_owners
            assert root.accounting_owner is old_owner
            real_reset(session_id, **kwargs)

        monkeypatch.setattr(root.session_logger, "reset_session", check_prepared)
        await root.clear_history()
        new_id = root.session_id
        owner = root.accounting_owner
        assert owner is not None
        assert owner.root_session_id == new_id
        assert owner.project_key == old_owner.project_key
        assert (new_id == old_id) is preserve_identity
        assert (owner is old_owner) is preserve_identity
        assert (owner.writer is old_owner.writer) is preserve_identity
        assert root.parent_session_id is None
        backend.release.set()
        await task
        session = await create_test_app_server_session(root)
        async for _event in session.act("new history"):
            pass
        old_records = read_ledger(config_dir, old_id)
        assert len(old_records) == (2 if preserve_identity else 1)
        assert all(record.root_session_id == old_id for record in old_records)
        assert all(record.session_id == old_id for record in old_records)
        assert all(
            record.project_key == old_owner.project_key for record in old_records
        )
        new_records = read_ledger(config_dir, new_id)
        assert new_records[-1].session_id == new_id
        assert new_records[-1].root_session_id == new_id
        assert new_records[-1].parent_session_id is None
        assert len(list((config_dir / "usage").glob("*/usage.jsonl"))) == (
            1 if preserve_identity else 2
        )
    finally:
        backend.release.set()
        if task is not None:
            await task
        if session is not None:
            await session.close()
        else:
            await runtime.close_agent_loop(root)
        for root_id in list(process._root_usage_writers):
            await process.retire_root_usage_writer(root_id)
        await process.close()


@pytest.mark.asyncio
async def test_clear_history_owner_preparation_failure_keeps_live_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    root = await open_test_root(
        process,
        monkeypatch,
        tmp_path,
        AttemptBackend([mock_llm_chunk(content="answer")]),
        logging_enabled=True,
    )
    session = await create_test_app_server_session(root)
    try:
        async for _event in session.act("original history"):
            pass
        old_id = root.session_id
        old_owner = root.accounting_owner
        old_messages = list(root.messages)
        old_stats = root.stats
        old_lease = root._session_lease

        def fail_preparation(session_id: str) -> Any:
            raise OSError("owner unavailable")

        monkeypatch.setattr(process, "root_usage_writer", fail_preparation)
        # The factory is held by the immutable owner; inject failure at its
        # writer-allocation seam, not by replacing the owner's frozen binding.
        with pytest.raises(OSError, match="owner unavailable"):
            await root.clear_history()
        assert root.session_id == old_id
        assert root.accounting_owner is old_owner
        assert list(root.messages) == old_messages
        assert root.stats is old_stats
        assert root._session_lease is old_lease
    finally:
        await session.close()
        for root_id in list(process._root_usage_writers):
            await process.retire_root_usage_writer(root_id)
        await process.close()


def finalization_record(agent_loop: AgentLoop, purpose: UsagePurpose) -> UsageRecord:
    attribution = agent_loop._call_resources(agent_loop.backend).usage_attribution
    assert attribution is not None
    return UsageRecord(
        **attribution.model_dump(exclude={"purpose"}),
        purpose=purpose,
        record_id=str(uuid4()),
        occurred_at=datetime.now(UTC),
        outcome=UsageOutcome.INTERRUPTED,
        usage_state=UsageState.MISSING,
        prices_usd_per_million=UsagePrices(),
        known_cost_usd=0,
        has_unknown_cost=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("purpose", [UsagePurpose.CONVERSATION, UsagePurpose.TITLE])
@pytest.mark.parametrize("drain_timeout", [True, False])
async def test_process_shutdown_settles_slow_child_write(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    purpose: UsagePurpose,
    drain_timeout: bool,
) -> None:
    process = runtime.HarnessProcess()
    root = await open_test_root(
        process, monkeypatch, tmp_path, AttemptBackend([]), logging_enabled=False
    )
    child = await process.runtime_factory.create_child(root, "worker")
    owner = root.accounting_owner
    assert owner is not None
    record = finalization_record(child, purpose)
    started = threading.Event()
    release = threading.Event()
    append = owner.writer._writer.append

    def slow_append(record: UsageRecord) -> Any:
        started.set()
        assert release.wait(3)
        return append(record)

    monkeypatch.setattr(owner.writer._writer, "append", slow_append)
    close_writer = owner.writer.aclose

    async def checked_close_writer() -> None:
        assert process.usage_service._closed
        assert owner.writer in process.usage_service._writers
        assert not owner.writer._pending
        await close_writer()

    monkeypatch.setattr(owner.writer, "aclose", checked_close_writer)
    write = asyncio.create_task(owner.writer.append(record))
    closing = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        if drain_timeout:
            monkeypatch.setattr(runtime, "_USAGE_SETTLEMENT_TIMEOUT", 0.01)
        closing = asyncio.create_task(process.close())
        if drain_timeout:
            with pytest.raises(TimeoutError, match="Usage writer did not drain"):
                await asyncio.wait_for(closing, 2)
            assert root.session_id in process._root_usage_drains
            assert not owner.writer._closed
            assert owner.writer in process.usage_service._writers
            monkeypatch.setattr(runtime, "_USAGE_SETTLEMENT_TIMEOUT", 5.0)
            closing = asyncio.create_task(process.close())
        await asyncio.sleep(0.05)
        assert not closing.done()
        assert process.usage_service._closed
        assert not process.usage_service._callbacks
        # The event loop remains responsive while the blocking append is in flight.
        release.set()
        await asyncio.wait_for(closing, 3)
        await write
        assert read_ledger(config_dir, root.session_id) == [record]
        assert owner.writer._closed
        assert process._root_usage_writers == {}
        assert process.usage_service._writers == {}
    finally:
        release.set()
        await write
        if closing is not None:
            await closing
        await process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("process_shutdown", [True, False])
async def test_retirement_waits_for_stale_cancelled_title_finalization(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_shutdown: bool,
) -> None:
    process = runtime.HarnessProcess()
    root = await open_test_root(
        process, monkeypatch, tmp_path, AttemptBackend([]), logging_enabled=False
    )
    old_id = root.session_id
    owner = root.accounting_owner
    assert owner is not None
    record = finalization_record(root, UsagePurpose.TITLE)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def stale_title() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
            await owner.writer(record)

    title = asyncio.create_task(stale_title())
    root._title_controller._task = title
    retiring = None
    try:
        await started.wait()
        await root.clear_history()
        await cancelled.wait()
        assert title in root._detached_accounting_producers
        await owner.writer.drain()  # No write has been submitted yet.
        retiring = asyncio.create_task(
            process.close()
            if process_shutdown
            else process.retire_root_usage_writer(old_id)
        )
        await asyncio.sleep(0.05)
        assert not retiring.done()
        assert not owner.writer._closed
        assert title in root._detached_accounting_producers
        assert root in process._accounting_loops
        release.set()
        await asyncio.wait_for(retiring, 3)
        assert owner.writer._closed
        assert read_ledger(config_dir, old_id) == [record]
    finally:
        release.set()
        await title
        if retiring is not None:
            await retiring
        await process.close()


@pytest.mark.asyncio
async def test_root_retirement_preserves_sibling_writer_and_deleted_session_ledger(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    root = await open_test_root(
        process, monkeypatch, tmp_path, AttemptBackend([]), logging_enabled=True
    )
    sibling = await process.runtime_factory.fork(root, None)
    child = await process.runtime_factory.create_child(root, "worker")
    owner = root.accounting_owner
    sibling_owner = sibling.accounting_owner
    assert owner is not None and sibling_owner is not None
    try:
        await root.persist_empty_session()
        root_id = root.session_id
        record = finalization_record(root, UsagePurpose.CONVERSATION)
        await owner.writer(record)
        await runtime.close_agent_loop(child)
        assert not owner.writer._closed
        await process.retire_root_usage_writer(root_id)
        assert owner.writer._closed
        assert not sibling_owner.writer._closed
        assert root not in process._accounting_loops
        assert child not in process._accounting_loops
        assert sibling in process._accounting_loops
        assert root_id not in process._accounting_loop_roots
        sibling_record = finalization_record(sibling, UsagePurpose.CONVERSATION)
        await sibling_owner.writer(sibling_record)
        assert read_ledger(config_dir, sibling.session_id) == [sibling_record]
        await delete_saved_session(root_id, root.config.session_logging)
        assert read_ledger(config_dir, root_id) == [record]
        assert (config_dir / "usage" / root_id / "usage.jsonl").is_file()
    finally:
        await process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("evict", [True, False])
async def test_child_teardown_untracks_all_bindings_without_retiring_writer(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evict: bool
) -> None:
    process = runtime.HarnessProcess()
    root = await open_test_root(
        process, monkeypatch, tmp_path, AttemptBackend([]), logging_enabled=False
    )
    registry = SessionRuntimeRegistry(
        AsyncMock(), AsyncMock(), lambda _: 0, runtime_factory=process.runtime_factory
    )
    child = await process.runtime_factory.create_child(root, "worker")
    child_runtime = registry._build_child_runtime(child)
    record = AgentRecord(
        agent_id="agent-1",
        profile="worker",
        session_id=child.session_id,
        runtime=child_runtime,
        root_generation=root._session_generation,
    )
    registry._agent_records[record.agent_id] = record
    registry._children[child.session_id] = child_runtime
    owner = root.accounting_owner
    assert owner is not None
    # Historical bindings must be removed too, not only the current owner bucket.
    process._accounting_loop_roots["historical-root"] = {child}
    try:
        if evict:
            assert await registry._evict_agent(record.agent_id, "ttl")
        else:
            await registry.release_agent(record.agent_id)
        assert child_runtime._closed
        assert child not in process._accounting_loops
        assert all(
            child not in loops for loops in process._accounting_loop_roots.values()
        )
        assert root in process._accounting_loops
        assert process.root_usage_writer(root.session_id) is owner.writer
        assert owner.writer in process.usage_service._writers
        assert not owner.writer._closed
        root_record = finalization_record(root, UsagePurpose.CONVERSATION)
        await owner.writer(root_record)
        assert read_ledger(config_dir, root.session_id) == [root_record]
    finally:
        await child_runtime.close()
        await process.close()


@pytest.mark.asyncio
async def test_unsettled_child_cleanup_keeps_tracking_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    root = await open_test_root(
        process, monkeypatch, tmp_path, AttemptBackend([]), logging_enabled=False
    )
    registry = SessionRuntimeRegistry(
        AsyncMock(), AsyncMock(), lambda _: 0, runtime_factory=process.runtime_factory
    )
    child = await process.runtime_factory.create_child(root, "worker")
    child_runtime = registry._build_child_runtime(child)
    release = asyncio.Event()

    async def finalize() -> None:
        await release.wait()

    producer = asyncio.create_task(finalize())
    child._detached_accounting_producers.add(producer)
    monkeypatch.setattr(runtime, "_USAGE_SETTLEMENT_TIMEOUT", 0.01)
    try:
        with pytest.raises(TimeoutError, match="Accounting producers"):
            await child_runtime.close()
        assert not child_runtime._closed
        assert child in process._accounting_loops
        assert child in process._accounting_loop_roots[root.session_id]
        release.set()
        await producer
        await child_runtime.close()
        assert child not in process._accounting_loops
        assert all(
            child not in loops for loops in process._accounting_loop_roots.values()
        )
    finally:
        release.set()
        await producer
        await child_runtime.close()
        await process.close()


def startup_finalization_record(context: StartupAccountingContext) -> UsageRecord:
    return UsageRecord(
        **context.attribution(build_test_vibe_config().get_active_model()).model_dump(),
        record_id=str(uuid4()),
        occurred_at=datetime.now(UTC),
        outcome=UsageOutcome.COMPLETED,
        usage_state=UsageState.MISSING,
        prices_usd_per_million=UsagePrices(),
        known_cost_usd=0,
        has_unknown_cost=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [True, False])
async def test_startup_request_never_selects_resume_or_continue(
    tmp_path: Path, resume: bool
) -> None:
    context = await create_startup_accounting_context(tmp_path)
    with pytest.raises(ValueError, match="new root"):
        runtime.RootOpenRequest(
            options=SessionOptions(),
            client_info=ClientInfo(name="test", version="1"),
            session_id=context.identity.root_session_id if resume else None,
            continue_latest=not resume,
            startup_accounting=context,
        )
    assert context.state == "available"


@pytest.mark.asyncio
async def test_startup_adoption_reuses_id_writer_and_replays_failed_settlement(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = await create_startup_accounting_context(tmp_path)
    record = startup_finalization_record(context)
    early = context.early_writer()
    await early(record)
    await early.aclose()
    failed = record.model_copy(update={"record_id": str(uuid4())})
    warning = CoverageWarning(
        code=CoverageWarningCode.WRITE_FAILED,
        root_session_id=context.identity.root_session_id,
        record_id=failed.record_id,
    )
    context.on_early_settlement(
        failed, UsageWriteResult(UsageWriteDisposition.FAILED, warning)
    )
    process = runtime.HarnessProcess()
    root = None
    try:
        # Entering a different workspace must not recompute project identity.
        root = await open_test_root(
            process,
            monkeypatch,
            tmp_path / "worktree",
            AttemptBackend([]),
            logging_enabled=False,
            startup_accounting=context,
        )
        assert context.state == "adopted"
        assert root.session_id == context.identity.root_session_id
        owner = root.accounting_owner
        assert owner is not None
        assert owner.project_key == context.identity.project_key
        assert owner.writer._writer is context.writer
        assert read_ledger(config_dir, root.session_id) == [record]
        snapshot = await process.usage_service.aread()
        assert snapshot.selected.request_count == 1
        assert warning in snapshot.warnings
        with pytest.raises(RuntimeError, match="single-use"):
            await process.open_root(
                runtime.RootOpenRequest(
                    options=SessionOptions(),
                    client_info=ClientInfo(name="test", version="1"),
                    startup_accounting=context,
                )
            )
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_failed_startup_keeps_paid_ledger_without_phantom_session(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = await create_startup_accounting_context(tmp_path)
    record = startup_finalization_record(context)
    early = context.early_writer()
    await early(record)
    await early.aclose()
    process = runtime.HarnessProcess()

    def fail(*args: Any, **kwargs: Any) -> AgentLoop:
        raise RuntimeError("construction failed")

    monkeypatch.setattr(runtime._RootRuntimeBlueprint, "build", fail)
    try:
        with pytest.raises(RuntimeError, match="construction failed"):
            await open_test_root(
                process,
                monkeypatch,
                tmp_path,
                AttemptBackend([]),
                logging_enabled=True,
                startup_accounting=context,
            )
        assert context.state == "abandoned"
        assert read_ledger(config_dir, context.identity.root_session_id) == [record]
        assert not list((tmp_path / "transcripts").glob("**/meta.json"))
        assert process._staged_roots == {}
        assert process._accounting_loops == set()
        assert (await process.usage_service.reconcile()).selected.request_count == 1
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_shutdown_settles_naming_before_any_agent_loop(
    config_dir: Path, tmp_path: Path
) -> None:
    context = await create_startup_accounting_context(tmp_path)
    process = runtime.HarnessProcess()
    owner = process.startup_accounting_owner(context)
    record = startup_finalization_record(context)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def naming() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
            await owner.writer(record)

    producer = asyncio.create_task(naming())
    process.track_startup_producer(context, producer)
    closing = None
    try:
        await started.wait()
        closing = asyncio.create_task(process.close())
        await asyncio.wait_for(cancelled.wait(), 2)
        assert not closing.done()
        assert not owner.writer._closed
        assert not process._accounting_loops
        release.set()
        await asyncio.wait_for(closing, 3)
        assert producer.done()
        assert owner.writer._closed
        assert context.state == "abandoned"
        assert read_ledger(config_dir, context.identity.root_session_id) == [record]
    finally:
        release.set()
        await producer
        if closing is not None:
            await closing
        await process.close()


def startup_controller(
    process: runtime.HarnessProcess, open_root: Any
) -> SessionRuntimeControllerImpl:
    services = MagicMock()
    services.client_info.return_value = ClientInfo(name="usage-test", version="1")
    services.client_capabilities.return_value = ClientCapabilities()
    return SessionRuntimeControllerImpl(
        open_root=open_root,
        runtime_factory=process.runtime_factory,
        host_handler=process.host_handler,
        stage_root=None,
        services=services,
        allocate_startup_accounting=process.allocate_startup_accounting,
        abandon_startup_accounting=process.abandon_startup_accounting,
        track_startup_producer=process.track_startup_producer,
    )


async def fake_startup_naming(
    options: SessionOptions, **bindings: Any
) -> WorktreeResolution:
    await utility_completion.run_utility_completion(
        config=build_test_vibe_config(),
        system_prompt="Name the worktree",
        user_content="Fix the bug",
        max_tokens=24,
        request_timeout_seconds=1.5,
        retry_budget_seconds=0,
        purpose=UsagePurpose.WORKTREE_NAMING,
        **bindings,
    )
    return WorktreeResolution(options=options)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "before", "after", "root"])
async def test_controller_startup_accounting_sequence(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    process = runtime.HarnessProcess()
    backend = AttemptBackend([mock_llm_chunk(content="fix-bug")])
    monkeypatch.setattr(utility_completion, "create_backend", lambda **_: backend)
    contexts = []
    allocate = process.allocate_startup_accounting

    async def reserve(workspace: Path) -> Any:
        assert workspace == tmp_path.resolve()
        context, owner = await allocate(workspace)
        contexts.append(context)
        return context, owner

    async def resolve(options: SessionOptions, **bindings: Any) -> WorktreeResolution:
        if failure == "before":
            raise RuntimeError("startup failed")
        result = await fake_startup_naming(options, **bindings)
        if failure == "after":
            raise RuntimeError("startup failed")
        return result

    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        if failure == "root":
            raise RuntimeError("startup failed")
        return await open_test_root(
            process,
            monkeypatch,
            tmp_path,
            AttemptBackend([mock_llm_chunk(content="answer")]),
            logging_enabled=False,
            startup_accounting=request.startup_accounting,
            startup_accounting_claimed=request.startup_accounting_claimed,
        )

    controller = startup_controller(process, open_root)
    controller._allocate_startup_accounting = reserve
    monkeypatch.setattr(controller._worktrees, "resolve_for_start", resolve)
    try:
        params = SessionStartParams(agent_config=AgentConfig(cwd=str(tmp_path)))
        if failure is not None:
            with pytest.raises(RuntimeError, match="startup failed"):
                await controller._open_runtime(params, None)
            context = contexts[0]
            assert context.state == "abandoned"
            path = (
                config_dir / "usage" / context.identity.root_session_id / "usage.jsonl"
            )
            if failure == "before":
                assert not path.exists()
                assert backend.requests_messages == []
            else:
                assert (
                    len(read_ledger(config_dir, context.identity.root_session_id)) == 1
                )
            assert process._root_usage_writers == {}
            assert process._root_accounting_owners == {}
            assert process._startup_contexts == {}
            assert process._startup_producers == {}
            assert controller.current_session() is None
            assert process._staged_roots == {}
        else:
            opened = await controller._open_runtime(params, None)
            root = opened.agent_loop
            session = await create_test_app_server_session(root)
            try:
                async for _ in session.act("hello"):
                    pass
            finally:
                await session.close()
            records = read_ledger(config_dir, root.session_id)
            assert [record.purpose for record in records] == [
                UsagePurpose.WORKTREE_NAMING,
                UsagePurpose.CONVERSATION,
            ]
            assert all(
                record.root_session_id == record.session_id == root.session_id
                for record in records
            )
            assert contexts[0].identity.root_session_id == root.session_id
            assert contexts[0].state == "adopted"
            assert (
                root.accounting_owner
                is process._root_accounting_owners[root.session_id]
            )
    finally:
        await process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [True, False])
async def test_controller_existing_session_does_not_reserve_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    process = runtime.HarnessProcess()
    opening = AsyncMock(side_effect=RuntimeError("stop"))
    controller = startup_controller(process, opening)
    allocation = AsyncMock()
    controller._allocate_startup_accounting = allocation
    monkeypatch.setattr(
        controller._worktrees,
        "resolve_for_start",
        AsyncMock(
            return_value=WorktreeResolution(options=SessionOptions(cwd=str(tmp_path)))
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="stop"):
            await controller._open_runtime(
                SessionStartParams(agent_config=AgentConfig(cwd=str(tmp_path))),
                "existing" if resume else None,
                continue_latest=not resume,
            )
        allocation.assert_not_called()
        assert opening.call_args.args[0].startup_accounting is None
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_concurrent_connection_startups_keep_separate_contexts(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    monkeypatch.setattr(
        utility_completion,
        "create_backend",
        lambda **_: AttemptBackend([mock_llm_chunk(content="fix")]),
    )
    contexts = []
    arrived = asyncio.Event()

    async def resolve(options: SessionOptions, **bindings: Any) -> WorktreeResolution:
        result = await fake_startup_naming(options, **bindings)
        contexts.append(bindings["usage_attribution"])
        if len(contexts) == 2:
            arrived.set()
        await asyncio.wait_for(arrived.wait(), 3)
        return result

    controllers = [
        startup_controller(process, AsyncMock(side_effect=RuntimeError("stop")))
        for _ in range(2)
    ]
    for controller in controllers:
        monkeypatch.setattr(controller._worktrees, "resolve_for_start", resolve)
    try:
        results = await asyncio.gather(
            *[
                controller._open_runtime(
                    SessionStartParams(agent_config=AgentConfig(cwd=str(tmp_path))),
                    None,
                )
                for controller in controllers
            ],
            return_exceptions=True,
        )
        assert all(isinstance(result, RuntimeError) for result in results)
        ids = {context.root_session_id for context in contexts}
        assert len(ids) == 2
        for root_id in ids:
            assert len(read_ledger(config_dir, root_id)) == 1
        assert process._root_accounting_owners == {}
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_controller_shutdown_settles_active_naming(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    contexts = []

    async def resolve(options: SessionOptions, **bindings: Any) -> WorktreeResolution:
        attribution = bindings["usage_attribution"]
        contexts.append(attribution)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
            record = startup_finalization_record(
                process._startup_contexts[attribution.root_session_id]
            )
            await bindings["accounting_sink"](record)
            raise
        raise AssertionError("Naming must be cancelled")

    controller = startup_controller(process, AsyncMock())
    monkeypatch.setattr(controller._worktrees, "resolve_for_start", resolve)
    startup = asyncio.create_task(
        controller._open_runtime(
            SessionStartParams(agent_config=AgentConfig(cwd=str(tmp_path))), None
        )
    )
    closing = None
    try:
        await asyncio.wait_for(started.wait(), 3)
        owner = process._root_accounting_owners[contexts[0].root_session_id]
        closing = asyncio.create_task(process.close())
        await asyncio.wait_for(cancelled.wait(), 3)
        assert not closing.done()
        assert not owner.writer._closed
        release.set()
        await asyncio.wait_for(closing, 3)
        with pytest.raises(asyncio.CancelledError):
            await startup
        assert len(read_ledger(config_dir, contexts[0].root_session_id)) == 1
        assert process._root_usage_writers == {}
    finally:
        release.set()
        await asyncio.gather(startup, return_exceptions=True)
        if closing is not None:
            await closing
        await process.close()


@pytest.mark.asyncio
async def test_concurrent_startup_claims_are_single_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    context, _owner = await process.allocate_startup_accounting(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blueprint(*args: Any) -> Any:
        entered.set()
        await release.wait()
        raise RuntimeError("construction failed")

    build = AsyncMock(side_effect=blueprint)
    monkeypatch.setattr(process, "build_root_blueprint", build)
    request = runtime.RootOpenRequest(
        options=SessionOptions(cwd=str(tmp_path)),
        client_info=ClientInfo(name="test", version="1"),
        startup_accounting=context,
    )
    first = asyncio.create_task(process.open_root(request))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        with pytest.raises(RuntimeError, match="single-use"):
            await process.open_root(request)
        release.set()
        with pytest.raises(RuntimeError, match="construction failed"):
            await first
        build.assert_awaited_once()
        assert context.state == "abandoned"
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        await process.abandon_startup_accounting(context)
        await process.close()


@pytest.mark.asyncio
async def test_controller_losing_claim_does_not_retire_winning_writer(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    context, owner = await process.allocate_startup_accounting(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    record = startup_finalization_record(context)

    async def naming(options: SessionOptions, **bindings: Any) -> WorktreeResolution:
        entered.set()
        await release.wait()
        await bindings["accounting_sink"](record)
        return WorktreeResolution(options=options)

    async def opening(request: runtime.RootOpenRequest) -> AgentLoop:
        return await open_test_root(
            process,
            monkeypatch,
            tmp_path,
            AttemptBackend([]),
            logging_enabled=False,
            startup_accounting=request.startup_accounting,
            startup_accounting_claimed=request.startup_accounting_claimed,
        )

    controllers = [startup_controller(process, opening) for _ in range(2)]
    for controller in controllers:
        controller._initial_startup_accounting = context
        controller._startup_accounting_owner = process.startup_accounting_owner
        monkeypatch.setattr(controller._worktrees, "resolve_for_start", naming)
    params = SessionStartParams(agent_config=AgentConfig(cwd=str(tmp_path)))
    winner = asyncio.create_task(controllers[0]._open_runtime(params, None))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        with pytest.raises(RuntimeError, match="single-use"):
            await controllers[1]._open_runtime(params, None)
        assert context.state == "claimed"
        assert (
            process._root_usage_writers[context.identity.root_session_id]
            is owner.writer
        )
        assert not owner.writer._closed
        release.set()
        opened = await asyncio.wait_for(winner, 3)
        assert context.state == "adopted"
        assert opened.agent_loop.accounting_owner is owner
        assert read_ledger(config_dir, context.identity.root_session_id) == [record]
    finally:
        release.set()
        await asyncio.gather(winner, return_exceptions=True)
        await process.close()


@pytest.mark.asyncio
async def test_retirement_releases_loop_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    try:
        root = await open_test_root(
            process, monkeypatch, tmp_path, AttemptBackend([]), logging_enabled=False
        )
        assert root in process._accounting_loops
        await process.retire_root_usage_writer(root.session_id)
        assert root not in process._accounting_loops
        assert not process._accounting_loop_roots
    finally:
        await process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_owned", [False, True])
async def test_stdio_eof_closes_only_automatically_owned_process(
    caller_owned: bool,
) -> None:
    from chartreux.app_server.transport import memory_transport_pair

    process = runtime.HarnessProcess() if caller_owned else None
    client, transport = memory_transport_pair()
    harness = await runtime.create_harness_server(
        transport, transport_kind="stdio", process=process
    )
    owned = process or harness._owned_process
    assert owned is not None
    writer = owned.root_usage_writer("eof-root")
    try:
        await client.close()
        await harness.serve()
        assert owned._closed is (not caller_owned)
        assert owned.usage_service._closed is (not caller_owned)
        assert writer._closed is (not caller_owned)
        if not caller_owned:
            assert not owned._root_usage_writers
    finally:
        await owned.close()


@pytest.mark.asyncio
async def test_local_harness_close_releases_owned_usage_resources() -> None:
    from chartreux.app_server.local import LocalHarness

    harness = LocalHarness(runtime.LocalHarnessOptions())
    process = harness._host._process_for()
    writer = process.root_usage_writer("local-root")
    await harness.close()
    await harness.close()
    assert process._closed
    assert process.usage_service._closed
    assert writer._closed
