"""B1c: the ledger's run-id join key (accepted-work association).

A subagent run's ledger records carry that run's id, so each record joins the
app-side ``RunRecord`` that holds the acceptance outcome. Root and utility
records carry None: a root loop never enters a run, and the title and
worktree-naming paths ride the run-free utility seam.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server import _runtime as runtime
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server.protocol import ClientInfo, SessionOptions
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm import utility_completion
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm_models import LLMChunk, LLMMessage, Role
from chartreux.core.session.title_model import generate_session_title
from chartreux.core.subagents import RunStatus, TaskArgs, TaskResult, WaitStatus
from chartreux.core.tools.base import InvokeContext
from chartreux.core.usage import UsagePurpose, UsageRecord
from tests.conftest import build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


class AttemptBackend(FakeBackend):
    async def complete(self, **kwargs: Any) -> LLMChunk:
        notify_request_started()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator:
        notify_request_started()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


async def open_test_root(
    process: runtime.HarnessProcess,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: FakeBackend,
) -> AgentLoop:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=False, save_dir=str(tmp_path / "transcripts"), generate_titles=False
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
        )
    )


def read_ledger(config_dir: Path, root_id: str) -> list[UsageRecord]:
    return [
        UsageRecord.model_validate_json(line)
        for line in (config_dir / "usage" / root_id / "usage.jsonl")
        .read_text()
        .splitlines()
    ]


def bind_registry(
    process: runtime.HarnessProcess, root: AgentLoop
) -> tuple[SessionRuntimeRegistry, Any]:
    registry = SessionRuntimeRegistry(
        AsyncMock(), AsyncMock(), lambda _: 0, runtime_factory=process.runtime_factory
    )
    root_runtime = registry._build_child_runtime(root)
    root_runtime.turns._projector = MagicMock()
    root_runtime.turns.link_subagent = AsyncMock()
    registry.bind_root(root_runtime)
    return registry, root_runtime


async def run_retained_subagent(
    registry: SessionRuntimeRegistry, root: AgentLoop, tool_call_id: str
) -> TaskResult:
    results = [
        item
        async for item in registry.run(
            TaskArgs(
                task="Inspect the project",
                identity="coder",
                background=True,
                retain=True,
            ),
            InvokeContext(tool_call_id=tool_call_id, session_id=root.session_id),
        )
        if isinstance(item, TaskResult)
    ]
    assert results and results[-1].run_id is not None
    return results[-1]


@pytest.mark.asyncio
async def test_subagent_records_carry_run_id(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    backend = AttemptBackend([
        [mock_llm_chunk(content="child answer")],
        [mock_llm_chunk(content="notification received")],
    ])
    root = await open_test_root(process, monkeypatch, tmp_path, backend)
    registry, root_runtime = bind_registry(process, root)
    try:
        launch = await run_retained_subagent(registry, root, "b1c-task")
        run_id = launch.run_id
        assert run_id is not None and launch.agent_id is not None
        await registry.wait_for_agent(launch.agent_id, run_id)
        # The parent notification turn (if already drained) is the only record
        # outside the run; child records all carry the run's id.
        records = [
            record
            for record in read_ledger(config_dir, root.session_id)
            if record.session_id != root.session_id
        ]
        assert records
        assert all(record.run_id == run_id for record in records)
    finally:
        await registry.drain_children()
        await root_runtime.close()
        await process.close()


@pytest.mark.asyncio
async def test_root_records_run_id_none(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    backend = AttemptBackend([mock_llm_chunk(content="answer")])
    root = await open_test_root(process, monkeypatch, tmp_path, backend)
    session = await create_test_app_server_session(root)
    try:
        async for _event in session.act("hello"):
            pass
        records = read_ledger(config_dir, root.session_id)
        assert records
        assert all(record.run_id is None for record in records)
    finally:
        await session.close()
        await runtime.close_agent_loop(root)
        await process.retire_root_usage_writer(root.session_id)
        await process.close()


@pytest.mark.asyncio
async def test_utility_records_run_id_none(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    backend = AttemptBackend([mock_llm_chunk(content="Session Title")])
    root = await open_test_root(process, monkeypatch, tmp_path, backend)
    try:
        await root.wait_until_ready()
        # The worktree-naming path rides the same run_utility_completion seam;
        # the title path here goes through the loop's real resource assembly.
        monkeypatch.setattr(utility_completion, "create_backend", lambda **_: backend)
        resources = root._call_resources(backend)
        attribution = resources.usage_attribution
        sink = resources.accounting_sink
        assert attribution is not None and sink is not None
        title = await generate_session_title(
            [LLMMessage(role=Role.user, content="please title this session")],
            config=root.config,
            accounting_sink=sink,
            usage_attribution=attribution,
        )
        assert title == "Session Title"
        owner = root.accounting_owner
        assert owner is not None
        await owner.writer.drain()
        records = read_ledger(config_dir, root.session_id)
        assert any(record.purpose is UsagePurpose.TITLE for record in records)
        assert all(record.run_id is None for record in records)
    finally:
        await runtime.close_agent_loop(root)
        await process.retire_root_usage_writer(root.session_id)
        await process.close()


@pytest.mark.asyncio
async def test_ledger_record_joins_run_record(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    backend = AttemptBackend([
        [mock_llm_chunk(content="child answer")],
        [mock_llm_chunk(content="notification received")],
    ])
    root = await open_test_root(process, monkeypatch, tmp_path, backend)
    registry, root_runtime = bind_registry(process, root)
    try:
        launch = await run_retained_subagent(registry, root, "b1c-join")
        agent_id = launch.agent_id
        assert agent_id is not None
        run_id = launch.run_id
        assert run_id is not None
        disposition = await registry.wait_for_agent(agent_id, run_id)
        assert disposition.status is WaitStatus.COMPLETED
        assert disposition.result is not None
        run_record = registry._agent_records[agent_id].run_history[-1]
        assert run_record.run_id == run_id
        records = read_ledger(config_dir, root.session_id)
        joined = [record for record in records if record.run_id == run_record.run_id]
        assert joined
        assert run_record.status is RunStatus.COMPLETED
    finally:
        await registry.drain_children()
        await root_runtime.close()
        await process.close()
