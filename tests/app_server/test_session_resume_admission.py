"""In-place resume, resume timing, compaction resume, fork, and resume admission tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._runtime import (
    AgentRuntimeFactory,
    HarnessProcess,
    RootOpenRequest,
    RuntimeSessionNotFoundError,
    _apply_stored_stats,
    _is_legacy_root_metadata,
    _mark_legacy_root_cost_incomplete,
    _same_concrete_identity,
)
from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import SessionSnapshot
from chartreux.app_server.models import PublicCheckpointEntry, PublicMessageEntry
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ClientInfo,
    PageRequest,
    ProtocolErrorCode,
    SessionResumeParams,
    SessionStartParams,
    SessionTurnsListParams,
    SessionTurnsListResponse,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.compaction import select_model_context
from chartreux.core.config import ModelConfig, SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_lease import SessionLease
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session_types import (
    AgentStats,
    CommittedModelIdentity,
    SessionMetadata,
)
from tests.app_server._session_runtime_support import (
    _consume,
    _isolate_recovery_registry,  # noqa: F401 - autouse fixture registered via module namespace
    _wire_resume_request,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    create_legacy_app_server,
    create_test_app_server_session,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_in_place_resume_clears_previous_turn_state(tmp_path: Path) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    backend = FakeBackend([[mock_llm_chunk(content="first answer")]])
    client = start_test_app_server(
        build_test_agent_loop(config=config, backend=backend, enable_streaming=True)
    )
    session = await attach_test_app_server_session(client)
    try:
        await _consume(session.act("first question"))
        # Resuming a different session rebinds the loop in place; the reused turn
        # controller must be reset so the previous session's turns do not leak.
        await session.resume(saved_session_id[:8])
        await session.read_turn_queue()
        result = await client.request(
            "session/turns/list",
            SessionTurnsListParams(
                session_id=saved_session_id, page=PageRequest(limit=10)
            ),
        )
    finally:
        await session.close()

    assert SessionTurnsListResponse.model_validate(result).items == []


@pytest.mark.asyncio
async def test_in_place_resume_discards_events_from_previous_session(
    tmp_path: Path,
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    session = await create_test_app_server_session(build_test_agent_loop(config=config))
    old_event = SessionSnapshot(session.state)
    await session._publish_event(old_event)
    try:
        await session.resume(saved_session_id)
        new_event = SessionSnapshot(session.state)
        await session._publish_event(new_event)

        events = session.events()
        observed = await asyncio.wait_for(anext(events), timeout=1)
        await events.aclose()
    finally:
        await session.close()

    assert observed is new_event


@pytest.mark.asyncio
async def test_in_place_resume_resets_scratchpad(tmp_path: Path) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    source = build_test_agent_loop(config=config)
    scratchpad_before = source.scratchpad_dir
    try:
        await AgentRuntimeFactory().resume_root(source, saved_session_id)
        # The scratchpad must be re-derived for the resumed session.
        assert source.scratchpad_dir != scratchpad_before
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_in_place_resume_leaves_source_untouched_when_session_missing(
    tmp_path: Path,
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    source = build_test_agent_loop(config=config)
    original_session_id = source.session_id
    try:
        with pytest.raises(RuntimeSessionNotFoundError):
            await AgentRuntimeFactory().resume_root(source, "missing-session")
        # The fallible load happens before any mutation, so a failed resume must
        # leave the live loop bound to its original session.
        assert source.session_id == original_session_id
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_in_place_resume_releases_lease_when_metadata_preparation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    def fail_stats(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("invalid persisted stats")

    monkeypatch.setattr("chartreux.app_server._runtime._build_stats", fail_stats)
    source = build_test_agent_loop(config=config)
    try:
        with pytest.raises(RuntimeError, match="invalid persisted stats"):
            await AgentRuntimeFactory().resume_root(source, saved_session_id)
        with SessionLease(tmp_path, saved_session_id):
            pass
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_in_place_resume_rebinds_before_waiting_for_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    source = build_test_agent_loop(config=config)
    session_id_when_awaited: list[str] = []

    async def fake_await_deferred_init() -> None:
        session_id_when_awaited.append(source.session_id)

    monkeypatch.setattr(source, "_await_deferred_init", fake_await_deferred_init)
    try:
        factory = AgentRuntimeFactory()
        await factory.resume_root(source, saved_session_id)
        # The rebind happens before wait_until_ready so the transcript is
        # visible immediately; the session is already the resumed one.
        assert source.session_id == saved_session_id
        await factory.finish_resume_root(source, saved_session_id)
    finally:
        await source.aclose()

    # Deferred init is awaited after the rebind, so it observes the resumed
    # session id — the init thread's update_system_prompt inserts at
    # position 0, landing correctly on top of the resumed messages.
    assert session_id_when_awaited == [saved_session_id]
    assert source.session_id == saved_session_id


@pytest.mark.asyncio
async def test_resume_records_init_duration(tmp_path: Path) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config, defer_heavy_init=True)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    source = build_test_agent_loop(config=config, defer_heavy_init=True)
    try:
        factory = AgentRuntimeFactory()
        await factory.resume_root(source, saved_session_id)
        await factory.finish_resume_root(source, saved_session_id)
        await source.wait_until_ready()
    finally:
        await source.aclose()

    assert source.init_duration_ms is not None
    assert isinstance(source.init_duration_ms, int)
    assert source.init_duration_ms >= 0


@pytest.mark.asyncio
async def test_wait_until_ready_records_init_duration_on_resume_without_finish_resume(
    tmp_path: Path,
) -> None:
    # session/ready/wait reads via wait_until_ready, which must record the
    # duration itself even if the background finish_resume_root hasn't run.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config, defer_heavy_init=True)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    source = build_test_agent_loop(config=config, defer_heavy_init=True)
    try:
        await AgentRuntimeFactory().resume_root(source, saved_session_id)
        await source.wait_until_ready()
    finally:
        await source.aclose()

    assert source.init_duration_ms is not None
    assert isinstance(source.init_duration_ms, int)
    assert source.init_duration_ms >= 0


@pytest.mark.asyncio
async def test_resume_does_not_leak_picker_init_duration(tmp_path: Path) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config, defer_heavy_init=True)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    source = build_test_agent_loop(config=config, defer_heavy_init=True)
    try:
        await source.wait_until_ready()
        assert source.init_duration_ms is not None
        picker_duration = source.init_duration_ms

        await AgentRuntimeFactory().resume_root(source, saved_session_id)
        # Rebind clears the picker's duration so it doesn't leak; _init_start_time
        # is kept, so the resumed session re-measures from __init__.
        assert source.init_duration_ms is None

        await source.wait_until_ready()
        assert source.init_duration_ms is not None
        assert source.init_duration_ms >= picker_duration
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_non_deferred_resume_stays_none(tmp_path: Path) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config, defer_heavy_init=False)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    source = build_test_agent_loop(config=config, defer_heavy_init=False)
    try:
        factory = AgentRuntimeFactory()
        await factory.resume_root(source, saved_session_id)
        await factory.finish_resume_root(source, saved_session_id)
        await source.wait_until_ready()
    finally:
        await source.aclose()

    assert source.init_duration_ms is None


@pytest.mark.asyncio
async def test_compaction_keeps_session_and_restores_full_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    config = build_test_vibe_config(session_logging=logging)
    backend = FakeBackend([
        [mock_llm_chunk(content="Before compaction")],
        [mock_llm_chunk(content="<summary>First turn completed</summary>")],
        [mock_llm_chunk(content="After compaction")],
    ])
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    session = await create_test_app_server_session(agent_loop)

    try:
        await _consume(session.act("first question", client_message_id="user-1"))
        session_id = session.session_id
        await session.compact()
        assert session.session_id == session_id
        assert [
            entry.text
            for entry in session.history
            if isinstance(entry, PublicMessageEntry)
        ] == ["first question", "Before compaction"]
        assert (
            sum(
                isinstance(entry, PublicCheckpointEntry) and entry.kind == "compaction"
                for entry in session.history
            )
            == 1
        )
        await _consume(session.act("second question", client_message_id="user-2"))
        session_dir = agent_loop.session_logger.session_dir
        assert session_dir is not None
    finally:
        await session.close()

    saved_messages, metadata = SessionLoader.load_session(session_dir)
    assert metadata["session_id"] == session_id
    assert [
        message.content
        for message in saved_messages
        if message.role in {Role.user, Role.assistant} and not message.context_boundary
    ] == ["first question", "Before compaction", "second question", "After compaction"]
    boundaries = [
        message
        for message in saved_messages
        if message.context_boundary == "compaction"
    ]
    assert len(boundaries) == 1
    assert len(SessionLoader.list_sessions(logging)) == 1

    resumed_backend = FakeBackend([[mock_llm_chunk(content="Resumed answer")]])
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: resumed_backend
    )
    resumed = await attach_test_app_server_session(
        start_test_app_server(
            build_test_agent_loop(
                config=config, backend=resumed_backend, enable_streaming=True
            )
        ),
        resume_session_id=session_id,
    )
    try:
        assert any(
            isinstance(entry, PublicCheckpointEntry) and entry.kind == "compaction"
            for entry in resumed.history
        )
        assert [
            entry.text
            for entry in resumed.history
            if isinstance(entry, PublicMessageEntry)
        ] == [
            "first question",
            "Before compaction",
            "second question",
            "After compaction",
        ]
        await _consume(resumed.act("third question", client_message_id="user-3"))
    finally:
        await resumed.close()

    model_context = resumed_backend.requests_messages[0]
    assert len(model_context) == 5
    assert model_context[1].context_boundary == "compaction"
    assert [message.content for message in model_context[2:]] == [
        "second question",
        "After compaction",
        "third question",
    ]


@pytest.mark.asyncio
async def test_in_process_resume_rebases_exit_usage(tmp_path: Path) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config)
    saved.stats.session_prompt_tokens = 11
    saved.stats.session_completion_tokens = 7
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    session = await create_test_app_server_session(build_test_agent_loop(config=config))
    try:
        await session.resume(saved_session_id)
        summary = session.exit_summary()
    finally:
        await session.close()

    assert summary.usage.input_tokens == 0
    assert summary.usage.output_tokens == 0


@pytest.mark.asyncio
async def test_in_place_resume_resets_stats_when_target_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    source = build_test_agent_loop(config=config)
    source.stats.session_prompt_tokens = 123
    source.stats.session_completion_tokens = 45
    monkeypatch.setattr(
        "chartreux.app_server._runtime._build_stats", lambda *a, **k: None
    )
    try:
        await AgentRuntimeFactory().resume_root(source, saved_session_id)
        assert source.stats.session_prompt_tokens == 0
        assert source.stats.session_completion_tokens == 0
        active_model = source.config.get_active_model()
        assert source.stats.input_price_per_million == active_model.input_price
        assert source.stats.output_price_per_million == active_model.output_price
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_v1_resume_explicitly_reselects_current_model_and_marks_cost_incomplete(
    tmp_path: Path,
) -> None:
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    saved_config = build_test_vibe_config(
        active_model="glm-5-3", session_logging=logging
    )
    saved = build_test_agent_loop(config=saved_config)
    saved.stats.known_cost_total = 9.0
    await saved.persist_empty_session()
    session_id = saved.session_id
    session_dir = saved.session_logger.session_dir
    assert session_dir is not None
    await saved.aclose()

    metadata_path = session_dir / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["launch_config"] = {
        "version": 1,
        "profile": "legacy",
        "overrides": {},
        "persona": {"system_prompt_id": "tests", "instructions": None},
    }
    metadata_path.write_text(json.dumps(metadata))

    current_config = build_test_vibe_config(
        active_model="glm-5-3", session_logging=logging
    )
    source = build_test_agent_loop(config=current_config)
    try:
        await AgentRuntimeFactory().resume_root(source, session_id)
        assert source.committed_model is not None
        assert source.committed_model.base_model == "glm-5-3"
        assert source.config.get_active_model().alias == "glm-5-3"
        assert source.stats.known_cost_total == 0.0
        assert source.stats.has_unknown_cost is True
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_resume_backfills_cached_price_for_legacy_stats() -> None:
    config = build_test_vibe_config(
        active_model="cached-model",
        models=[
            ModelConfig(
                name="cached-model",
                provider="mistral",
                alias="cached-model",
                input_price=1.0,
                output_price=2.0,
                cached_input_price=0.1,
            )
        ],
    )
    replacement = build_test_agent_loop(config=config)
    try:
        legacy_metadata: dict[str, object] = {
            "stats": {
                "session_prompt_tokens": 100,
                "session_completion_tokens": 50,
                "input_price_per_million": 1.0,
                "output_price_per_million": 2.0,
            }
        }
        _apply_stored_stats(replacement, legacy_metadata)
        assert replacement.stats.session_prompt_tokens == 100
        assert replacement.stats.cached_input_price_per_million == 0.1
    finally:
        await replacement.aclose()


@pytest.mark.asyncio
async def test_attached_fork_rebases_exit_usage(tmp_path: Path) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    agent_loop = build_test_agent_loop(config=config)
    agent_loop.stats.session_prompt_tokens = 11
    agent_loop.stats.session_completion_tokens = 7
    session = await create_test_app_server_session(agent_loop)

    try:
        await session.resources.sessions.fork(attach=True)
        await session.resources.refresh()
        summary = session.exit_summary()
    finally:
        await session.close()

    assert summary.usage.input_tokens == 0
    assert summary.usage.output_tokens == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_id", "expected_context"),
    [
        pytest.param("u1", ["first", "first reply"], id="before-compaction"),
        pytest.param(
            "u2",
            ["first compacted context", "second", "second reply"],
            id="between-compactions",
        ),
        pytest.param(
            "u3",
            ["second compacted context", "third", "third reply"],
            id="after-compactions",
        ),
    ],
)
async def test_fork_uses_latest_compaction_boundary_in_copied_prefix(
    message_id: str, expected_context: list[str]
) -> None:
    source = build_test_agent_loop()
    source.messages.reset([
        LLMMessage(role=Role.system, content="system"),
        LLMMessage(role=Role.user, content="first", message_id="u1"),
        LLMMessage(role=Role.assistant, content="first reply"),
        LLMMessage(
            role=Role.user,
            content="first compacted context",
            injected=True,
            context_boundary="compaction",
        ),
        LLMMessage(role=Role.user, content="second", message_id="u2"),
        LLMMessage(role=Role.assistant, content="second reply"),
        LLMMessage(
            role=Role.user,
            content="second compacted context",
            injected=True,
            context_boundary="compaction",
        ),
        LLMMessage(role=Role.user, content="third", message_id="u3"),
        LLMMessage(role=Role.assistant, content="third reply"),
    ])

    forked = await AgentRuntimeFactory().fork(source, message_id)
    try:
        assert forked.parent_session_id == source.session_id
        assert [
            message.content
            for message in select_model_context(forked.messages)
            if message.role is not Role.system
        ] == expected_context
    finally:
        await forked.aclose()
        await source.aclose()


@pytest.mark.asyncio
async def test_detached_fork_transfers_live_runtime_without_session_logging() -> None:
    process = HarnessProcess()
    source_loop = build_test_agent_loop()

    async def open_source(_request: RootOpenRequest) -> AgentLoop:
        return source_loop

    source_client_transport, source_server_transport = memory_transport_pair()
    source_server = create_legacy_app_server(
        source_server_transport,
        open_root=open_source,
        runtime_factory=process.runtime_factory,
        stage_root=process.stage_root,
    )
    source = await AppServerSession.start(
        AppServerClient(source_client_transport, run_peer=source_server.serve),
        client_info=ClientInfo(name="fork-test", version="1"),
        capabilities=ClientCapabilities(),
    )
    child: AppServerSession | None = None
    try:
        await source.resources.sessions.update_settings(max_turns=7, max_tokens=4096)
        fork = await source.resources.sessions.fork(attach=False)

        child_client_transport, child_server_transport = memory_transport_pair()
        child_server = create_legacy_app_server(
            child_server_transport,
            open_root=process.open_root,
            runtime_factory=process.runtime_factory,
            stage_root=process.stage_root,
        )
        child = await AppServerSession.start(
            AppServerClient(child_client_transport, run_peer=child_server.serve),
            client_info=ClientInfo(name="fork-test", version="1"),
            capabilities=ClientCapabilities(),
            resume_session_id=fork.state.session.id,
        )

        assert child.session_id == fork.state.session.id
        runtime_policy = legacy_backend(child_server).session.agent_loop.runtime_policy
        assert runtime_policy.max_turns == 7
        assert runtime_policy.max_tokens == 4096
    finally:
        if child is not None:
            await child.close()
        await source.close()
        await process.close()


@pytest.mark.asyncio
async def test_harness_process_closes_unclaimed_fork_runtime() -> None:
    process = HarnessProcess()
    staged = build_test_agent_loop()
    close = AsyncMock(wraps=staged.aclose)
    staged.aclose = close

    await process.stage_root(staged)
    await process.close()

    close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_detached_fork_reserves_source_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = HarnessProcess()
    source_loop = build_test_agent_loop()
    entered = asyncio.Event()
    release = asyncio.Event()
    fork_runtime = process.runtime_factory.fork

    async def blocking_fork(source: AgentLoop, message_id: str | None) -> AgentLoop:
        entered.set()
        await release.wait()
        return await fork_runtime(source, message_id)

    monkeypatch.setattr(process.runtime_factory, "fork", blocking_fork)

    async def open_source(_request: RootOpenRequest) -> AgentLoop:
        return source_loop

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(
        server_transport,
        open_root=open_source,
        runtime_factory=process.runtime_factory,
        stage_root=process.stage_root,
    )
    session = await AppServerSession.start(
        AppServerClient(client_transport, run_peer=server.serve),
        client_info=ClientInfo(name="fork-test", version="1"),
        capabilities=ClientCapabilities(),
    )
    fork = asyncio.create_task(session.resources.sessions.fork(attach=False))
    await entered.wait()
    try:
        with pytest.raises(AppServerResponseError) as exc_info:
            await _consume(session.act("racing prompt"))
        assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT
    finally:
        release.set()
        await fork
        await session.close()
        await process.close()


@pytest.mark.asyncio
async def test_concurrent_resumes_replace_roots_serially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = build_test_agent_loop(session_id="initial")
    factory = AgentRuntimeFactory()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    sources = []

    async def resume_root(source, session_id):
        sources.append(source)
        if session_id == "first":
            first_entered.set()
            await release_first.wait()
        source.session_id = session_id  # simulate in-place rebind

    monkeypatch.setattr(factory, "resume_root", resume_root)

    async def open_root(_request: RootOpenRequest):
        return initial

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(
        server_transport, open_root=open_root, runtime_factory=factory
    )
    client = AppServerClient(client_transport, run_peer=server.serve)
    await client.initialize(ClientInfo(name="resume-test", version="1"))
    await client.notify("initialized")
    await client.request("session/start", SessionStartParams())

    first_resume = asyncio.create_task(
        client.request("session/resume", _wire_resume_request("first"))
    )
    await first_entered.wait()
    second_resume = asyncio.create_task(
        client.request("session/resume", _wire_resume_request("second"))
    )
    await asyncio.sleep(0)
    # The second resume waits behind the lifecycle lock; only the first has
    # started, and both always share the same loop object (in-place mutation).
    assert sources == [initial]
    release_first.set()
    try:
        await asyncio.gather(first_resume, second_resume)
        # Second resume runs against the already-rebounded loop (still initial).
        assert len(sources) == 2
        assert sources[0] is initial
        assert sources[1] is initial
        assert isinstance(server._root, SessionBackendImpl)
        assert server._root.session.agent_loop.session_id == "second"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_resume_reserves_session_before_loading_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = build_test_agent_loop(session_id="initial")
    replacement = build_test_agent_loop(session_id="replacement")
    entered = asyncio.Event()
    release = asyncio.Event()
    factory = AgentRuntimeFactory()

    async def resume_root(_source, _session_id):
        entered.set()
        await release.wait()
        return replacement

    monkeypatch.setattr(factory, "resume_root", resume_root)

    async def open_root(_request: RootOpenRequest):
        return initial

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(
        server_transport, open_root=open_root, runtime_factory=factory
    )
    client = AppServerClient(client_transport, run_peer=server.serve)
    await client.initialize(ClientInfo(name="resume-test", version="1"))
    await client.notify("initialized")
    await client.request("session/start", SessionStartParams())

    resume = asyncio.create_task(
        client.request("session/resume", _wire_resume_request(replacement.session_id))
    )
    await entered.wait()
    try:
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "turn/start",
                {
                    "sessionId": initial.session_id,
                    "input": [{"type": "text", "text": "racing prompt"}],
                },
            )
        assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT
    finally:
        release.set()
        await resume
        await client.close()


@pytest.mark.asyncio
async def test_compaction_reserves_session_before_awaiting_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    session = await create_test_app_server_session(agent_loop)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def compact(_instructions: str = "") -> str:
        entered.set()
        await release.wait()
        return "summary"

    monkeypatch.setattr(agent_loop, "compact", compact)
    compaction = asyncio.create_task(session.compact())
    await entered.wait()
    try:
        with pytest.raises(AppServerResponseError) as exc_info:
            await _consume(session.act("racing prompt"))
        assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT
    finally:
        release.set()
        await compaction
        await session.close()


@pytest.mark.asyncio
async def test_resume_returns_before_init_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = build_test_agent_loop(session_id="initial")
    waiting = asyncio.Event()
    release = asyncio.Event()

    async def blocking_wait_until_ready() -> None:
        waiting.set()
        await release.wait()

    factory = AgentRuntimeFactory()

    async def fake_resume_root(source: object, session_id: str) -> None:
        source.session_id = session_id  # type: ignore[attr-defined]

    async def fake_finish_resume_root(
        source: object, session_id: str, **kwargs: object
    ) -> None:
        await source.wait_until_ready()  # type: ignore[attr-defined]

    monkeypatch.setattr(factory, "resume_root", fake_resume_root)
    monkeypatch.setattr(factory, "finish_resume_root", fake_finish_resume_root)

    async def open_root(_request: RootOpenRequest):
        return initial

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(
        server_transport, open_root=open_root, runtime_factory=factory
    )
    client = AppServerClient(client_transport, run_peer=server.serve)
    await client.initialize(ClientInfo(name="resume-test", version="1"))
    await client.notify("initialized")
    await client.request("session/start", SessionStartParams())

    monkeypatch.setattr(initial, "wait_until_ready", blocking_wait_until_ready)
    resume = asyncio.create_task(
        client.request("session/resume", SessionResumeParams(session_id="replacement"))
    )
    await waiting.wait()
    # The resume RPC must return before deferred init finishes — that is the
    # whole point of fast resume. finish_resume_root is still blocked on
    # wait_until_ready, so the resume task should already be done.
    assert resume.done()
    try:
        await resume
    finally:
        release.set()
        await client.close()


@pytest.mark.asyncio
async def test_resume_rebinds_session_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    initial = build_test_agent_loop(session_id="initial")
    initial_close = AsyncMock(wraps=initial.aclose)
    initial.aclose = initial_close
    factory = AgentRuntimeFactory()

    async def fake_resume_root(source: object, session_id: str) -> None:
        # Simulate in-place rebind: only session_id changes, same object.
        source.session_id = session_id  # type: ignore[attr-defined]

    monkeypatch.setattr(factory, "resume_root", fake_resume_root)

    async def open_root(_request: RootOpenRequest):
        return initial

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(
        server_transport, open_root=open_root, runtime_factory=factory
    )
    client = AppServerClient(client_transport, run_peer=server.serve)
    await client.initialize(ClientInfo(name="resume-test", version="1"))
    await client.notify("initialized")
    await client.request("session/start", SessionStartParams())

    try:
        await client.request(
            "session/resume", SessionResumeParams(session_id="replacement")
        )
        # The same loop object stays active — no new loop was built or swapped in.
        assert isinstance(server._root, SessionBackendImpl)
        assert server._root.session.agent_loop is initial
        assert server._root.session.agent_loop.session_id == "replacement"
        # The runtime is never closed — it's still the active root.
        initial_close.assert_not_awaited()
    finally:
        await client.close()


def test_resume_identity_compares_provider_and_wire_name() -> None:
    original = CommittedModelIdentity(
        base_model="base", provider="one", wire_name="wire", catalog_revision="a"
    )
    changed_provider = original.model_copy(update={"provider": "two"})
    changed_revision = original.model_copy(update={"catalog_revision": "b"})

    assert _same_concrete_identity(original, changed_provider) is False
    assert _same_concrete_identity(original, changed_revision) is True


def test_envelope_less_root_marks_historical_cost_incomplete() -> None:
    metadata = SessionMetadata(
        session_id="legacy",
        start_time="2026-01-01T00:00:00+00:00",
        end_time=None,
        git_commit=None,
        git_branch=None,
        environment={},
        username="test",
    )
    stats = AgentStats(known_cost_total=4.0)

    assert _is_legacy_root_metadata(metadata) is True
    _mark_legacy_root_cost_incomplete(stats, metadata)

    assert stats.has_unknown_cost is True
    assert stats.known_cost_total == 0.0
