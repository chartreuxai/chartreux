"""Session listing for continue, continue/resume/rebind, and survives-a-resume tests for the local harness."""

from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.app_server import _runtime as runtime
import chartreux.app_server._host as host_module
from chartreux.app_server._projection import project_message_history
from chartreux.app_server._projector import ProjectedUpdate
from chartreux.app_server._turns import TurnController
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.local import LocalHarnessHost
from chartreux.app_server.models import PublicHistoryEntry
from chartreux.app_server.protocol import (
    AutoWorktreeInput,
    ClientCapabilities,
    ClientInfo,
    HistoryEntryUpdatedParams,
    SessionListParams,
    SessionOptions,
)
from chartreux.app_server.registry_manifest import SessionWriterFence
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import (
    FunctionCall,
    LLMMessage,
    PersistedToolResult,
    Role,
    ToolCall,
)
from chartreux.core.message_list import MessageList
from chartreux.core.session.resume_sessions import ResumeSessionInfo
from chartreux.core.session.session_lease import SessionBusyError
from chartreux.core.session.session_loader import MESSAGES_FILENAME, SessionLoader
import chartreux.core.session.session_logger as session_logger_module
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session.transcript_archive import (
    ArchivedPrefixV1,
    ArchiveIntegrityError,
)
from chartreux.core.utils.tags import CancellationReason, get_user_cancellation_message
from tests.app_server._local_runtime_support import (
    _configure_local_accounting,
    _init_repo,
)
from tests.app_server.test_usage_attribution import read_ledger
from tests.archive_history_support import descriptor_for_prefix, write_transcript
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import create_legacy_app_server
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_local_root_close_retires_accounting_and_resume_reopens_ledger(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_local_accounting(monkeypatch, tmp_path, logging_enabled=True)
    harness = LocalHarnessHost()
    options = runtime.LocalHarnessOptions(
        session_options=SessionOptions(cwd=str(tmp_path), headless=True)
    )
    host = await harness.connect(options)
    resumed_host = None
    try:
        session = await host.open_session()
        root_id = session.session_id
        async for _ in session.act("first"):
            pass
        process = harness._process
        assert process is not None
        writer = process._root_usage_writers[root_id]
        paths = set((config_dir / "usage").rglob("*.jsonl"))
        await session.close()
        assert writer._closed
        assert not process._closed
        assert not process.usage_service._closed
        assert root_id not in process._root_usage_writers
        assert root_id not in process._root_accounting_owners
        assert root_id not in process._accounting_loop_roots
        assert not process._accounting_loops
        assert set((config_dir / "usage").rglob("*.jsonl")) == paths
        assert len(read_ledger(config_dir, root_id)) == 1

        resumed_host = await harness.connect(
            runtime.LocalHarnessOptions(
                session_options=options.session_options,
                session=runtime.ResumeSessionIntent(root_id),
            )
        )
        resumed = await resumed_host.open_session()
        assert resumed.session_id == root_id
        assert process._root_usage_writers[root_id] is not writer
        async for _ in resumed.act("second"):
            pass
        await resumed.close()
        records = read_ledger(config_dir, root_id)
        assert len(records) == 2
        assert all(record.root_session_id == root_id for record in records)
        assert set((config_dir / "usage").rglob("*.jsonl")) == paths
        assert not process._root_usage_writers
        assert not process._accounting_loops
    finally:
        await host.close()
        if resumed_host is not None:
            await resumed_host.close()
        await harness.close()


@pytest.mark.asyncio
async def test_durable_resume_round_trip_holds_writer_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WP3 [3D]: a live root holds the session-dir write fence; the durable
    resume round-trip persists both turns and re-takes the fence after close.
    """
    _configure_local_accounting(monkeypatch, tmp_path, logging_enabled=True)
    logging_config = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions")
    )
    harness = LocalHarnessHost()
    options = runtime.LocalHarnessOptions(
        session_options=SessionOptions(cwd=str(tmp_path), headless=True)
    )
    host = await harness.connect(options)
    resumed_host = None
    try:
        session = await host.open_session()
        root_id = session.session_id
        async for _ in session.act("first turn"):
            pass
        session_dir = SessionLoader.find_session_by_id(root_id, logging_config)
        assert session_dir is not None
        fence_path = session_dir / ".writer-fence.lock"
        assert fence_path.is_file()
        # The live root writer holds the flock: a second writer is refused.
        with pytest.raises(SessionBusyError):
            SessionWriterFence(session_dir, root_id).acquire()
        messages, _ = SessionLoader.load_session(session_dir)
        contents = [m.content for m in messages if isinstance(m.content, str)]
        assert "first turn" in contents
        assert "answer" in contents
        await session.close()

        # Root close released the fence; the lockfile is kept (stable inode).
        inode = fence_path.stat().st_ino
        SessionWriterFence(session_dir, root_id).acquire().release()
        assert fence_path.stat().st_ino == inode

        resumed_host = await harness.connect(
            runtime.LocalHarnessOptions(
                session_options=options.session_options,
                session=runtime.ResumeSessionIntent(root_id),
            )
        )
        resumed = await resumed_host.open_session()
        assert resumed.session_id == root_id
        with pytest.raises(SessionBusyError):
            SessionWriterFence(session_dir, root_id).acquire()
        async for _ in resumed.act("second turn"):
            pass
        messages, _ = SessionLoader.load_session(session_dir)
        contents = [m.content for m in messages if isinstance(m.content, str)]
        assert "first turn" in contents
        assert "second turn" in contents
        await resumed.close()
        SessionWriterFence(session_dir, root_id).acquire().release()
    finally:
        await host.close()
        if resumed_host is not None:
            await resumed_host.close()
        await harness.close()


class ClosingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.closed = True


def test_session_list_falls_back_for_malformed_saved_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config()
    monkeypatch.setattr(
        host_module,
        "list_local_resume_sessions",
        lambda _config, _cwd: [
            ResumeSessionInfo(
                session_id="session-1",
                cwd="/workspace",
                start_time="not-a-timestamp",
                updated_at="",
            )
        ],
    )
    monkeypatch.setattr(host_module, "now_ms", lambda: 123_456)
    monkeypatch.setattr(
        host_module.SessionLoader, "get_first_user_message", lambda *_args: ""
    )

    response = host_module.project_session_list(config, SessionListParams())

    assert response.items[0].created_at == 123_456
    assert response.items[0].updated_at == 123_456


def _stub_session_list_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        host_module,
        "list_local_resume_sessions",
        lambda _config, _cwd: [
            ResumeSessionInfo(
                session_id="newer", cwd="/workspace", updated_at="2026-01-02"
            ),
            ResumeSessionInfo(
                session_id="older", cwd="/workspace", updated_at="2026-01-01"
            ),
        ],
    )
    monkeypatch.setattr(
        host_module.SessionLoader, "get_first_user_message", lambda *_args: ""
    )


def test_session_list_continue_id_prefers_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config()
    _stub_session_list_sources(monkeypatch)
    monkeypatch.setattr(
        host_module.last_session_pointer, "load", lambda _config: "older"
    )

    response = host_module.project_session_list(config, SessionListParams())

    assert response.continue_session_id == "older"


def test_session_list_continue_id_falls_back_to_newest_when_pointer_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config()
    _stub_session_list_sources(monkeypatch)
    monkeypatch.setattr(host_module.last_session_pointer, "load", lambda _config: None)

    response = host_module.project_session_list(config, SessionListParams())

    assert response.continue_session_id == "newer"


def test_session_list_continue_id_ignores_stale_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config()
    _stub_session_list_sources(monkeypatch)
    monkeypatch.setattr(
        host_module.last_session_pointer, "load", lambda _config: "does-not-exist"
    )

    response = host_module.project_session_list(config, SessionListParams())

    assert response.continue_session_id == "newer"


def test_session_list_continue_id_accepts_forked_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config()
    monkeypatch.setattr(
        host_module,
        "list_local_resume_sessions",
        lambda _config, _cwd: [
            ResumeSessionInfo(
                session_id="root", cwd="/workspace", updated_at="2026-01-02"
            ),
            ResumeSessionInfo(
                session_id="fork",
                cwd="/workspace",
                updated_at="2026-01-01",
                parent_session_id="root",
            ),
        ],
    )
    monkeypatch.setattr(
        host_module.SessionLoader, "get_first_user_message", lambda *_args: ""
    )
    monkeypatch.setattr(
        host_module.last_session_pointer, "load", lambda _config: "fork"
    )

    response = host_module.project_session_list(config, SessionListParams())

    assert response.continue_session_id == "fork"


@pytest.mark.asyncio
async def test_a_dropped_notification_still_survives_a_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    emit = TurnController._emit_projected
    loops: list[AgentLoop] = []
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        )
    )

    async def drop_the_completion(
        self: TurnController, update: ProjectedUpdate
    ) -> None:
        if isinstance(update.params, HistoryEntryUpdatedParams):
            raise RuntimeError("the client went away")
        await emit(self, update)

    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        assert request.options.cwd is not None
        agent_loop = build_test_agent_loop(cwd=Path(request.options.cwd), config=config)
        loops.append(agent_loop)
        return agent_loop

    monkeypatch.setattr(TurnController, "_emit_projected", drop_the_completion)
    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="dropped-resume-client", version="1"),
        capabilities=ClientCapabilities(),
        session_options=SessionOptions(
            cwd=str(tmp_path), worktree=AutoWorktreeInput(prompt="fix the login bug")
        ),
    )
    try:
        # The live entry is cosmetic; the persisted metadata is the only thing a
        # resume rebuilds from. Losing the completion notification must not cost
        # the durable record too, or the worktree disappears from the transcript
        # in exactly the failure this notification path is allowed to have.
        agent_loop = loops[0]
        await agent_loop.persist_empty_session()
        session_dir = agent_loop.session_logger.session_dir
        assert session_dir is not None
    finally:
        await session.close()
        await client.close()

    rebuilt = _rebuild_from_disk(agent_loop.session_id, session_dir)
    effects = [entry for entry in rebuilt if entry.type == "effect"]
    assert len(effects) == 1
    assert effects[0].detail.kind == "worktree"


def _rebuild_from_disk(session_id: str, session_dir: Path) -> list[PublicHistoryEntry]:
    """Rebuild a transcript the way a resume does: from the files, not memory."""
    messages, _ = SessionLoader.load_session(session_dir)
    return project_message_history(
        session_id, messages, SessionLoader.load_metadata(session_dir)
    )


@pytest.mark.asyncio
async def test_the_worktree_entry_survives_a_resume(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    loops: list[AgentLoop] = []
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        )
    )

    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        assert request.options.cwd is not None
        agent_loop = build_test_agent_loop(cwd=Path(request.options.cwd), config=config)
        loops.append(agent_loop)
        return agent_loop

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="resume-client", version="1"),
        capabilities=ClientCapabilities(),
        session_options=SessionOptions(
            cwd=str(tmp_path), worktree=AutoWorktreeInput(prompt="fix the login bug")
        ),
    )
    try:
        # The metadata file does not exist yet when the worktree is recorded, so
        # the field patch has nothing to write into: this full save is what
        # first carries the worktree to disk.
        agent_loop = loops[0]
        await agent_loop.persist_empty_session()
        session_dir = agent_loop.session_logger.session_dir
        assert session_dir is not None
    finally:
        await session.close()
        await client.close()

    # A harness effect living only in the TurnController is gone by here, and
    # the runner reclaims the process a second after every turn - so the round
    # trip through the file is the whole guarantee.
    assert SessionLoader.load_metadata(session_dir).created_worktree is not None

    rebuilt = _rebuild_from_disk(agent_loop.session_id, session_dir)
    effects = [entry for entry in rebuilt if entry.type == "effect"]
    assert len(effects) == 1
    assert effects[0].detail.kind == "worktree"
    assert effects[0].detail.input is not None
    assert effects[0].detail.input.name == "fix-the-login-bug"
    # The worktree exists before anything else the transcript can show.
    assert rebuilt[0] is effects[0]


@pytest.mark.asyncio
async def test_session_continue_opens_the_resumed_root_once_over_json_rpc() -> None:
    resumed = build_test_agent_loop()
    requests: list[runtime.RootOpenRequest] = []

    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        requests.append(request)
        return resumed

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    session = await AppServerSession.start(
        AppServerClient(client_transport, run_peer=server.serve),
        client_info=ClientInfo(name="continue-client", version="1"),
        capabilities=ClientCapabilities(),
        session_options=SessionOptions(cwd=str(Path.cwd())),
        continue_session=True,
    )
    try:
        assert session.session_id == resumed.session_id
        assert len(requests) == 1
        assert requests[0].continue_latest is True
        assert requests[0].session_id is None
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_resume_rebinds_session_in_place_preserving_backend(
    tmp_path: Path,
) -> None:
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    source_backend = ClosingBackend()
    source = build_test_agent_loop(
        config=build_test_vibe_config(session_logging=logging), backend=source_backend
    )
    source.stats.session_prompt_tokens = 11
    source.stats.session_completion_tokens = 7
    source.stats.context_tokens = 18
    await source.persist_empty_session()

    original_session_id = source.session_id
    await runtime.AgentRuntimeFactory().resume_root(source, original_session_id)

    try:
        # resume_root mutates in-place: same object, same backend, stats preserved.
        assert source.session_id == original_session_id
        assert source.backend is source_backend
        assert source.stats.session_prompt_tokens == 11
        assert source.stats.session_completion_tokens == 7
        assert not source_backend.closed
    finally:
        await source.aclose()


def _placeholder(call_id: str) -> LLMMessage:
    """The synthetic TOOL_NO_RESPONSE backfill placeholder (PI1 poison)."""
    return LLMMessage(
        role=Role.tool,
        tool_call_id=call_id,
        name="read_file",
        content=str(get_user_cancellation_message(CancellationReason.TOOL_NO_RESPONSE)),
    )


def _poisoned_transcript() -> list[LLMMessage]:
    """[user, assistant(call), placeholder, steered user, genuine result].

    The repairable PI1 shape: ``SessionLoader.load_session`` repairs it in
    memory (the shadowed placeholder drops) and the repair report rides the
    metadata to the write-owning resume install.
    """
    return [
        LLMMessage(role=Role.user, content="question", message_id="u1"),
        LLMMessage(
            role=Role.assistant,
            content="working",
            tool_calls=[
                ToolCall(
                    id="t1",
                    index=0,
                    function=FunctionCall(name="read_file", arguments="{}"),
                )
            ],
        ),
        _placeholder("t1"),
        LLMMessage(role=Role.user, content="steered mid-turn"),
        LLMMessage(
            role=Role.tool,
            tool_call_id="t1",
            name="read_file",
            content="file contents",
            tool_result=PersistedToolResult(output={"text": "file contents"}),
        ),
    ]


async def _build_warm_session(
    tmp_path: Path, transcript_messages: list[LLMMessage], *, archived_count: int = 1
) -> tuple[AgentLoop, Path, ArchivedPrefixV1]:
    """A persisted session whose on-disk transcript carries a retained
    archive descriptor (the warm-resume precondition).
    """
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    source = build_test_agent_loop(
        config=build_test_vibe_config(session_logging=logging)
    )
    await source.persist_empty_session()
    session_dir = source.session_logger.session_dir
    assert session_dir is not None
    transcript = session_dir / MESSAGES_FILENAME
    write_transcript(transcript, transcript_messages)
    descriptor = descriptor_for_prefix(
        transcript,
        archived_count,
        session_id=source.session_id,
        message_epoch=source.messages.message_epoch,
    )
    source.session_logger.retain_archive_descriptor(descriptor)
    return source, transcript, descriptor


def _txn_outcome_spy(monkeypatch: pytest.MonkeyPatch, outcomes: list[str]) -> None:
    """Record the outcome of every clear-and-reseal transaction run."""
    real_txn = SessionLogger.clear_and_reseal_warm_resume

    async def _txn_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        result = await real_txn(self, *args, **kwargs)
        outcomes.append(result.outcome)
        return result

    monkeypatch.setattr(SessionLogger, "clear_and_reseal_warm_resume", _txn_spy)


def _migration_snapshot_spy(
    monkeypatch: pytest.MonkeyPatch, migrated_bytes: list[bytes]
) -> None:
    """Snapshot the live transcript right after the durable repair migration."""
    real_install = runtime._install_transcript_repair

    def _install_spy(
        session_path: Path, messages: list[LLMMessage], metadata: dict[str, object]
    ) -> tuple[list[LLMMessage], bool]:
        result = real_install(session_path, messages, metadata)
        migrated_bytes.append((session_path / MESSAGES_FILENAME).read_bytes())
        return result

    monkeypatch.setattr(runtime, "_install_transcript_repair", _install_spy)


@pytest.mark.asyncio
async def test_warm_child_continuation_clears_the_archive_descriptor_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2b clear-and-reseal INVERTS the M1 warm-continuation retention rule.

    The warm-resume rebind (``rebind_to_session`` for a resident root,
    ``resume_child`` for a child) commits through
    ``SessionLogger.apply_resumed_session`` and then runs the warm-resume
    clear-and-reseal transaction: the retained archive descriptor is proven
    fully hydrated from the target disk and then CLEARED — never re-pinned —
    so the next compaction seal mints a fresh receipt at the new epoch (trim
    re-arms through that seal, never from the resume path itself).
    """
    source, _transcript, _descriptor = await _build_warm_session(
        tmp_path, [LLMMessage(role=Role.user, content="question", message_id="u1")]
    )
    outcomes: list[str] = []
    _txn_outcome_spy(monkeypatch, outcomes)
    try:
        await runtime.AgentRuntimeFactory().resume_root(source, source.session_id)

        # INVERTED (the old node asserted `archive_descriptor is descriptor`
        # after resume): clear-and-reseal cleared the retained descriptor —
        # the authority transitioned under the resume transaction.
        assert outcomes == ["proved_and_transitioned"]
        assert source.session_logger.archive_descriptor is None
        assert source.messages.archive_descriptor is None
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_warm_resume_transaction_failure_after_replacement_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transaction arm (Amendment W): a failure AFTER the staged repair
    replaced the live transcript but BEFORE the rebind/authority transition
    rolls back to the recovery artifact's verified bytes and RETAINS the
    archive authority (fail-closed, never a silent partial state).
    """
    source, transcript, descriptor = await _build_warm_session(
        tmp_path, _poisoned_transcript()
    )
    outcomes: list[str] = []
    _txn_outcome_spy(monkeypatch, outcomes)
    migrated_bytes: list[bytes] = []
    _migration_snapshot_spy(monkeypatch, migrated_bytes)

    prove_calls: list[int] = []
    real_prove = session_logger_module.prove_rebound_hydration

    def _failing_reprove(*args: Any, **kwargs: Any) -> Any:
        prove_calls.append(len(prove_calls) + 1)
        if len(prove_calls) == 2:
            # The staged re-prove — after the replacement, before the
            # rebind/authority transition: fail closed.
            raise ArchiveIntegrityError("injected post-replacement failure")
        return real_prove(*args, **kwargs)

    monkeypatch.setattr(
        session_logger_module, "prove_rebound_hydration", _failing_reprove
    )

    try:
        await runtime.AgentRuntimeFactory().resume_root(source, source.session_id)

        # The transaction reported the rollback; the retained authority and
        # the on-disk bytes (restored from the verified artifact) survive.
        assert outcomes == ["failed_repair_rolled_back"]
        assert source.session_logger.archive_descriptor is descriptor
        assert migrated_bytes and transcript.read_bytes() == migrated_bytes[0]
        session_dir = source.session_logger.session_dir
        assert session_dir is not None
        assert list(session_dir.glob("messages.jsonl.recovered-*"))
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_warm_resume_transaction_cancellation_after_replacement_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transaction arm (Amendment W): cancellation in the window AFTER the
    staged repair replaced the live path but BEFORE the authority transition
    drains the staging worker, restores the artifact's verified bytes over
    the live path, and re-raises — the retained authority stays.
    """
    source, transcript, descriptor = await _build_warm_session(
        tmp_path, _poisoned_transcript()
    )
    migrated_bytes: list[bytes] = []
    _migration_snapshot_spy(monkeypatch, migrated_bytes)

    entered = threading.Event()
    block = threading.Event()
    prove_calls = 0
    real_prove = session_logger_module.prove_rebound_hydration

    def _blocking_reprove(*args: Any, **kwargs: Any) -> Any:
        nonlocal prove_calls
        prove_calls += 1
        if prove_calls == 2:
            # The staged re-prove: hold the window open so the cancellation
            # lands after the replacement, before the transition.
            entered.set()
            assert block.wait(5)
        return real_prove(*args, **kwargs)

    monkeypatch.setattr(
        session_logger_module, "prove_rebound_hydration", _blocking_reprove
    )

    factory = runtime.AgentRuntimeFactory()
    task = asyncio.create_task(factory.resume_root(source, source.session_id))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0.1)
    block.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    try:
        # The cancellation rolled the replacement back and kept the authority.
        assert source.session_logger.archive_descriptor is descriptor
        assert migrated_bytes and transcript.read_bytes() == migrated_bytes[0]
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_resume_child_runs_clear_and_reseal_after_the_rebind_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The genuinely child-path node (plan round 3): ``resume_child`` builds
    a NEW child around the stored session and runs the clear-and-reseal
    transaction after the message reset and the logger rebind (the :988-996
    ordering holds under the new semantics). The retained root authority is
    untouched by the child resume; the new child's archive authority is
    CLEARED — the expected re-arm outcome (a fresh receipt at the next
    seal's new epoch).
    """
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    config = build_test_vibe_config(session_logging=logging)
    parent = build_test_agent_loop(config=config)
    factory = runtime.AgentRuntimeFactory()
    try:
        await parent.persist_empty_session()
        parent_dir = parent.session_logger.session_dir
        assert parent_dir is not None
        parent_transcript = parent_dir / MESSAGES_FILENAME
        write_transcript(
            parent_transcript,
            [LLMMessage(role=Role.user, content="root question", message_id="r1")],
        )
        parent_descriptor = descriptor_for_prefix(
            parent_transcript,
            1,
            session_id=parent.session_id,
            message_epoch=parent.messages.message_epoch,
        )
        parent.session_logger.retain_archive_descriptor(parent_descriptor)

        child = await factory.create_child(parent, "coder")
        await child.wait_until_ready()
        child.messages.append(
            LLMMessage(role=Role.user, content="child question", message_id="c1")
        )
        await child.persist_empty_session()
        child_dir = child.session_logger.session_dir
        assert child_dir is not None
        child_id = child.session_id
        await child.aclose()

        order: list[str] = []
        outcomes: list[str] = []
        real_reset = MessageList.reset_preserving_system
        real_apply = SessionLogger.apply_resumed_session
        real_txn = SessionLogger.clear_and_reseal_warm_resume

        def _reset_spy(self: MessageList, *args: Any, **kwargs: Any) -> Any:
            order.append("reset_preserving_system")
            return real_reset(self, *args, **kwargs)

        def _apply_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
            order.append("apply_resumed_session")
            return real_apply(self, *args, **kwargs)

        async def _txn_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
            order.append("clear_and_reseal_warm_resume")
            result = await real_txn(self, *args, **kwargs)
            outcomes.append(result.outcome)
            return result

        monkeypatch.setattr(MessageList, "reset_preserving_system", _reset_spy)
        monkeypatch.setattr(SessionLogger, "apply_resumed_session", _apply_spy)
        monkeypatch.setattr(SessionLogger, "clear_and_reseal_warm_resume", _txn_spy)

        resumed = await factory.resume_child(parent, "coder", child_id, child_dir)
        try:
            # Child ordering: the transaction rides the rebind — the last
            # three steps are reset -> logger rebind -> clear-and-reseal, and
            # the transaction ran exactly once on the child path.
            assert order[-3:] == [
                "reset_preserving_system",
                "apply_resumed_session",
                "clear_and_reseal_warm_resume",
            ]
            assert order.count("clear_and_reseal_warm_resume") == 1
            assert outcomes == ["proved_and_transitioned"]
            # New child (not the retained incarnation), bound to the stored
            # session, with its archive authority CLEARED (clear-and-reseal).
            assert resumed is not child
            assert resumed.session_id == child_id
            assert resumed.session_logger.archive_descriptor is None
            assert resumed.messages.archive_descriptor is None
            # The retained root authority is untouched by the child resume.
            assert parent.session_logger.archive_descriptor is parent_descriptor
        finally:
            await resumed.aclose()
    finally:
        await parent.aclose()


def test_continue_prefers_valid_terminal_session_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    session_path = tmp_path / "session"
    monkeypatch.setattr(runtime.last_session_pointer, "load", lambda _config: "saved")
    monkeypatch.setattr(
        runtime.SessionLoader,
        "find_session_by_id",
        lambda *_args, **_kwargs: session_path,
    )

    source = cast(AgentLoop, SimpleNamespace(config=config))
    assert runtime.AgentRuntimeFactory().resolve_latest(source, Path.cwd()) == "saved"


def test_continue_falls_back_to_latest_session_for_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    session_path = tmp_path / "latest"
    monkeypatch.setattr(runtime.last_session_pointer, "load", lambda _config: "stale")
    monkeypatch.setattr(
        runtime.SessionLoader, "find_session_by_id", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runtime.SessionLoader,
        "find_latest_session",
        lambda *_args, **_kwargs: session_path,
    )
    monkeypatch.setattr(
        runtime.SessionLoader,
        "load_session",
        lambda _path: ([], {"session_id": "latest-id"}),
    )

    source = cast(AgentLoop, SimpleNamespace(config=config))
    assert (
        runtime.AgentRuntimeFactory().resolve_latest(source, Path.cwd()) == "latest-id"
    )


def test_continue_requires_session_logging() -> None:
    config = build_test_vibe_config(session_logging=SessionLoggingConfig(enabled=False))
    source = cast(AgentLoop, SimpleNamespace(config=config))

    with pytest.raises(
        runtime.RuntimeSessionNotFoundError, match="Session logging is disabled"
    ):
        runtime.AgentRuntimeFactory().resolve_latest(source, Path.cwd())
