"""Cross-cutting acceptance: real producers, durable storage, and public projections."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncGenerator
from datetime import UTC
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from git import Repo
import httpx
import pytest

from chartreux.app_server import _runtime as runtime
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.config import StatusLineConfigView
from chartreux.app_server.protocol import (
    ClientCapabilities,
    ClientInfo,
    SessionOptions,
    UsageReadResponse,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.cli.textual_ui.widgets.session_status_line import (
    SessionStatusState,
    format_segment,
)
from chartreux.core._usage_startup import (
    StartupAccountingContext,
    create_startup_accounting_context,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.git.worktree import naming_model
from chartreux.core.llm import utility_completion
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm_models import LLMChunk, LLMMessage, LLMUsage, Role, StopInfo
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.session.saved_sessions import delete_saved_session
from chartreux.core.session.title_model import generate_session_title
from chartreux.core.usage import (
    CoverageWarningCode,
    UsageOutcome,
    UsagePurpose,
    UsageReader,
    UsageService,
    UsageState,
)
from tests.stubs.app_server import (
    create_legacy_app_server,
    create_test_app_server_session,
)
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


class LedgerBackend(FakeBackend):
    """Transport-attempt fake, with deterministic usage and optional mid-turn loss."""

    def __init__(self, *, fail_at: int | None = None, content: str = "answer") -> None:
        super().__init__()
        self.attempts: list[tuple[str, str, str]] = []
        self.fail_at = fail_at
        self.content = content

    async def complete(self, **kwargs: Any) -> LLMChunk:
        notify_request_started()
        model = kwargs["model"]
        self.attempts.append((model.alias, model.provider, model.name))
        if len(self.attempts) == self.fail_at:
            raise httpx.ConnectError("synthetic transport loss")
        return LLMChunk(
            message=LLMMessage(
                role=Role.assistant,
                content="<summary>retained facts</summary>"
                if model.alias == "compact"
                else self.content,
            ),
            usage=LLMUsage(prompt_tokens=100, completion_tokens=10, cached_tokens=20),
            stop=StopInfo(reason="stop"),
        )

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        yield await self.complete(**kwargs)


def ledger_config(workspace: Path, *, logging_enabled: bool) -> ChartreuxConfigSchema:
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                name: {"api_base": f"https://{name}.invalid"}
                for name in ("first", "second")
            },
            "models": {
                alias: {
                    "deployments": [
                        {
                            "provider": provider,
                            "name": f"{alias}-{provider}",
                            "prices": {"input": rate, "output": 2, "cached_input": 0.5},
                        }
                        for provider, rate in (("first", 1), ("second", 3))
                    ]
                }
                for alias in ("base", "compact")
            },
            "roles": {"worker": {"model": "base", "thinking": "off"}},
        }),
        "integration-ledger",
    )
    return ChartreuxConfigSchema.model_validate(
        {
            "active_model": "base",
            "compaction_model": "compact",
            "session_logging": SessionLoggingConfig(
                enabled=logging_enabled,
                save_dir=str(workspace / "transcripts"),
                generate_titles=False,
            ),
        },
        context={"catalog_snapshot": snapshot},
    ).attach_catalog_snapshot(snapshot)


async def open_root(
    process: runtime.HarnessProcess,
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    backend: LedgerBackend,
    *,
    logging_enabled: bool = False,
    startup: StartupAccountingContext | None = None,
) -> AgentLoop:
    config = ledger_config(workspace, logging_enabled=logging_enabled)
    monkeypatch.setattr(
        runtime,
        "build_default_orchestrator",
        AsyncMock(return_value=FakeConfigOrchestrator(config)),
    )
    monkeypatch.setattr(
        process, "_build_mcp_registry_impl", AsyncMock(return_value=FakeMCPRegistry())
    )

    def build_loop(**kwargs: Any) -> AgentLoop:
        return AgentLoop(backend=backend, **kwargs)

    monkeypatch.setattr(runtime, "AgentLoop", build_loop)
    return await process.open_root(
        runtime.RootOpenRequest(
            options=SessionOptions(cwd=str(workspace), headless=True),
            client_info=ClientInfo(name="ledger-integration", version="1"),
            startup_accounting=startup,
        )
    )


async def turn(session: AppServerSession, message: str = "hello") -> None:
    async for _event in session.act(message):
        pass


async def fresh_service(home: Path) -> tuple[UsageReader, UsageService]:
    # Never inspect the producer's cache: acceptance must survive process restart.
    reader = UsageReader(home / "usage")
    service = UsageService(reader, timezone_resolver=lambda: UTC)
    await service.wait_ready()
    return reader, service


@pytest.mark.asyncio
@pytest.mark.parametrize("logging_enabled", [True, False])
async def test_all_producers_survive_restart_and_transcript_deletion(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    logging_enabled: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context = await create_startup_accounting_context(workspace)
    config = ledger_config(workspace, logging_enabled=logging_enabled)
    utility = LedgerBackend(content="ledger-work")
    monkeypatch.setattr(utility_completion, "create_backend", lambda **_: utility)
    monkeypatch.setattr(
        naming_model,
        "build_default_orchestrator",
        AsyncMock(return_value=FakeConfigOrchestrator(config)),
    )
    early = context.early_writer()
    try:
        assert (
            await naming_model.suggest_worktree_name(
                "build a ledger",
                cwd=workspace,
                accounting_sink=early,
                usage_attribution_factory=context.attribution,
            )
            == "ledger-work"
        )
    finally:
        await early.aclose()

    backend = LedgerBackend(fail_at=2)
    process = runtime.HarnessProcess()
    root = await open_root(
        process,
        monkeypatch,
        workspace,
        backend,
        logging_enabled=logging_enabled,
        startup=context,
    )
    session = await create_test_app_server_session(root)
    child_session = None
    try:
        assert root.session_id == context.identity.root_session_id
        assert context.state == "adopted"
        await turn(session, "first question")
        await turn(session, "second question")  # first fails, second deployment wins
        child = await process.runtime_factory.create_child(root, "worker")
        child_session = await create_test_app_server_session(child)
        await turn(child_session, "child task")
        assert await root.compaction_manager.compact() == "retained facts"
        resources = root._call_resources(root.backend)
        assert (
            await generate_session_title(
                root.messages,
                config=root.config,
                accounting_sink=resources.accounting_sink,
                usage_attribution=resources.usage_attribution,
            )
            == "ledger-work"
        )
        root_id, child_id = root.session_id, child.session_id
        owner = root.accounting_owner
        assert owner is not None
        project_key = owner.project_key
        logging_config = root.config.session_logging
        transcript_dir = root.session_logger.session_dir
        if not logging_enabled:
            assert transcript_dir is None
            assert child.session_logger.session_dir is None
    finally:
        if child_session is not None:
            await child_session.close()
        await session.close()
        await process.close()

    if logging_enabled:
        assert transcript_dir is not None and transcript_dir.exists()
        await delete_saved_session(root_id, logging_config)
        assert not transcript_dir.exists()

    reader, service = await fresh_service(config_dir)
    try:
        records = reader.snapshot.records
        assert len(records) == len({record.record_id for record in records}) == 7
        assert len(backend.attempts) == 5
        assert len(utility.attempts) == 2
        assert Counter(record.purpose for record in records) == {
            UsagePurpose.CONVERSATION: 4,
            UsagePurpose.COMPACTION: 1,
            UsagePurpose.TITLE: 1,
            UsagePurpose.WORKTREE_NAMING: 1,
        }
        assert all(record.root_session_id == root_id for record in records)
        assert all(record.project_key == project_key for record in records)
        assert list((config_dir / "usage").glob("*/usage.jsonl")) == [
            config_dir / "usage" / root_id / "usage.jsonl"
        ]
        conversations = [r for r in records if r.purpose == UsagePurpose.CONVERSATION]
        assert [
            (r.model, r.provider, r.wire_name) for r in conversations
        ] == backend.attempts[:4]
        assert [r.outcome for r in conversations] == [
            UsageOutcome.COMPLETED,
            UsageOutcome.FAILED,
            UsageOutcome.COMPLETED,
            UsageOutcome.COMPLETED,
        ]
        assert conversations[1].usage_state == UsageState.MISSING
        assert conversations[1].input_tokens is None
        child_record = conversations[-1]
        assert child_record.session_id == child_id
        assert child_record.parent_session_id == root_id
        assert (child_record.agent_role, child_record.agent_profile) == (
            "subagent",
            "worker",
        )
        for record in records:
            if record is not child_record:
                assert record.session_id == root_id
                assert record.parent_session_id is None
            if record.outcome == UsageOutcome.COMPLETED:
                assert (
                    record.input_tokens,
                    record.output_tokens,
                    record.cached_input_tokens,
                ) == (100, 10, 20)
                assert record.known_cost_usd == pytest.approx(
                    0.00011 if record.provider == "first" else 0.00027
                )
        assert records[0].agent_role == "startup"
        assert [
            (r.model, r.provider, r.wire_name)
            for r in records
            if r.purpose == UsagePurpose.COMPACTION
        ] == backend.attempts[4:]
        assert [
            (r.model, r.provider, r.wire_name)
            for r in records
            if r.purpose in {UsagePurpose.WORKTREE_NAMING, UsagePurpose.TITLE}
        ] == utility.attempts
        snapshot = await service.aread("month")
        assert snapshot.selected.request_count == 7
        assert snapshot.selected.input_tokens == 600
        assert snapshot.selected.output_tokens == 60
        assert snapshot.selected.cached_input_tokens == 120
        assert snapshot.selected.has_unknown_cost
        assert snapshot.selected.known_cost_usd == pytest.approx(
            sum(r.known_cost_usd for r in records)
        )
        assert Counter({
            (r.model, r.provider, r.wire_name): r.request_count for r in snapshot.models
        }) == Counter((r.model, r.provider, r.wire_name) for r in records)
        assert not snapshot.warnings
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_global_project_and_linked_worktree_totals_after_restart(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_path = tmp_path / "repo"
    linked = tmp_path / "linked"
    other = tmp_path / "other"
    other.mkdir()
    with Repo.init(repo_path, initial_branch="main") as repo:
        with repo.config_writer() as config:
            config.set_value("user", "name", "Tester")
            config.set_value("user", "email", "tester@example.invalid")
        repo.index.commit("initial")
        repo.git.worktree("add", "--detach", str(linked))
    process = runtime.HarnessProcess()
    keys = []
    backend = LedgerBackend()
    try:
        for workspace in (repo_path, linked, other):
            root = await open_root(process, monkeypatch, workspace, backend)
            assert root.accounting_owner is not None
            keys.append(root.accounting_owner.project_key)
            session = await create_test_app_server_session(root)
            try:
                await turn(session)
            finally:
                await session.close()
    finally:
        await process.close()
    assert keys[0] == keys[1] != keys[2]
    reader, service = await fresh_service(config_dir)
    try:
        assert len(reader.snapshot.records) == len(backend.attempts) == 3
        global_snapshot = await service.aread("month")
        repo_snapshot = await service.aread("month", keys[0])
        other_snapshot = await service.aread("month", keys[2])
        assert global_snapshot.selected.request_count == 3
        assert repo_snapshot.selected.request_count == 2
        assert other_snapshot.selected.request_count == 1
        assert repo_snapshot.as_of == other_snapshot.as_of == global_snapshot.as_of
        assert (
            repo_snapshot.revision
            == other_snapshot.revision
            == global_snapshot.revision
        )
        assert global_snapshot.selected.known_cost_usd == pytest.approx(
            repo_snapshot.selected.known_cost_usd
            + other_snapshot.selected.known_cost_usd
        )
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_disk_failure_does_not_retry_or_fail_over_a_real_turn(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    backend = LedgerBackend()
    root = await open_root(process, monkeypatch, tmp_path, backend)
    session = await create_test_app_server_session(root)
    writes = []

    def disk_full(*args: Any) -> bool:
        writes.append(args)
        raise OSError("synthetic disk full")

    try:
        with monkeypatch.context() as fault:
            fault.setattr("chartreux.core._usage_io.append_usage_record", disk_full)
            await turn(session)
        assert len(writes) == 2  # bounded storage retry, never an inference retry
        assert backend.attempts == [("base", "first", "base-first")]
        assert root.committed_model is not None
        assert root.committed_model.provider == "first"
        degraded = await process.usage_service.aread()
        assert degraded.selected.request_count == 0
        assert [warning.code for warning in degraded.warnings] == [
            CoverageWarningCode.WRITE_FAILED
        ]
        assert degraded.warnings[0].root_session_id == root.session_id
        await turn(session, "storage recovered")
        assert len(backend.attempts) == 2
    finally:
        await session.close()
        await process.close()
    reader, service = await fresh_service(config_dir)
    try:
        assert len(reader.snapshot.records) == 1
        assert (await service.aread()).selected.request_count == 1
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_torn_tail_preserves_completed_turns_after_restart(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime.HarnessProcess()
    backend = LedgerBackend()
    root = await open_root(process, monkeypatch, tmp_path, backend)
    session = await create_test_app_server_session(root)
    try:
        await turn(session)
        await turn(session)
    finally:
        await session.close()
        await process.close()
    path = config_dir / "usage" / root.session_id / "usage.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"schema_version":1,"record_id":"interrupted')
    reader, service = await fresh_service(config_dir)
    try:
        assert len(reader.snapshot.records) == len(backend.attempts) == 2
        snapshot = await service.aread("month")
        assert snapshot.selected.request_count == 2
        assert snapshot.selected.input_tokens == 200
        assert snapshot.selected.known_cost_usd == pytest.approx(0.00022)
        assert [w.code for w in snapshot.warnings] == [CoverageWarningCode.TORN_TAIL]
        assert snapshot.selected.warnings == snapshot.warnings
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_rpc_client_cache_and_statusline_share_service_revision(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer = runtime.HarnessProcess()
    root = await open_root(producer, monkeypatch, tmp_path, LedgerBackend())
    old_session = await create_test_app_server_session(root)
    try:
        await turn(old_session)
    finally:
        await old_session.close()
        await producer.close()

    # A new host scans the prior host's durable records; the client uses real RPC.
    process = runtime.HarnessProcess()
    backend = LedgerBackend()
    root = await open_root(process, monkeypatch, tmp_path, backend)
    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(
        server_transport,
        open_root=AsyncMock(return_value=root),
        host_handler=process.host_handler,
        runtime_factory=process.runtime_factory,
    )
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="ledger-status", version="1"),
        capabilities=ClientCapabilities(),
        session_options=SessionOptions(headless=True),
    )
    try:
        first = await session.resources.usage.read()
        assert first.summaries.day.requests == 1
        updated = asyncio.Event()
        unsubscribe = session.resources.usage.subscribe(lambda _: updated.set())
        await turn(session)
        await asyncio.wait_for(updated.wait(), timeout=5)
        response = await session.resources.usage.read()
        raw = UsageReadResponse.model_validate(await client.request("usage/read"))
        cached = session.resources.usage.current
        assert cached is not None
        snapshot = await process.usage_service.aread()
        assert first.revision < response.revision
        assert raw.revision == response.revision == cached.revision == snapshot.revision
        assert raw.as_of == response.as_of == cached.as_of == snapshot.as_of
        assert raw.summaries == response.summaries == cached.summaries
        assert cached.summaries.day.requests == snapshot.selected.request_count == 2
        state = SessionStatusState(
            cwd=tmp_path,
            usage_day=cached.summaries.day,
            usage_week=cached.summaries.week,
            usage_month=cached.summaries.month,
        )
        for segment, label, summary in (
            ("spend-today", "Today", raw.summaries.day),
            ("spend-week", "Week", raw.summaries.week),
            ("spend-month", "Month", raw.summaries.month),
        ):
            assert (
                format_segment(segment, state, StatusLineConfigView())
                == f"{label} ${summary.known_cost_usd:.2f}"
            )
        unsubscribe()
    finally:
        await session.close()
        await server.close()
        await process.close()
