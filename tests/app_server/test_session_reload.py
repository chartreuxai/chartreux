"""Tests for session reload and model-change resume boundaries."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._model import validate_wire
from chartreux.app_server._runtime import AgentRuntimeFactory, RootOpenRequest
from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.models import (
    PublicCheckpointEntry,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicNoticeEntry,
    RouteTableAppliedNoticeDetail,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientInfo,
    ConfigMutationResponse,
    ConfigReloadParams,
    HistoryEntryAddedParams,
    Notification,
    PageRequest,
    ProtocolErrorCode,
    RuntimeUpdatedParams,
    SessionHistoryClearParams,
    SessionResumeParams,
    SessionStartParams,
    SessionTurnsListParams,
    SessionTurnsListResponse,
)
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ModelConfig, SessionLoggingConfig
from chartreux.core.dispatch.route_models import PackageTier, PurposeRoute, RouteTarget
from chartreux.core.dispatch.routes import SESSION_MODEL, shipped_routes
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_lease import SessionBusyError, SessionLease
from tests.app_server._session_runtime_support import (
    _consume,
    _isolate_recovery_registry,  # noqa: F401 - autouse fixture registered via module namespace
    _wire_resume_request,
)
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    set_agent_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    create_legacy_app_server,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend


async def _model_change_resume_context(
    tmp_path: Path,
) -> tuple[AppServerSession, AppServer, AgentLoop, str, SessionLease]:
    models = [
        ModelConfig(name="model-a", provider="mistral", alias="model-a"),
        ModelConfig(name="model-b", provider="mistral", alias="model-b"),
    ]
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    saved = build_test_agent_loop(
        config=build_test_vibe_config(
            active_model="model-b", models=models, session_logging=logging
        )
    )
    saved.messages.append(LLMMessage(role=Role.user, content="target transcript"))
    saved.stats.session_prompt_tokens = 17
    await saved.persist_empty_session()
    target_session_id = saved.session_id
    await saved.aclose()

    source = build_test_agent_loop(
        config=build_test_vibe_config(
            active_model="model-a", models=models, session_logging=logging
        )
    )
    source.messages.append(LLMMessage(role=Role.user, content="old transcript"))
    source.stats.session_prompt_tokens = 41
    old_lease = SessionLease(tmp_path, source.session_id).acquire()
    source._session_lease = old_lease

    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(source, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    return session, server, source, target_session_id, old_lease


def _assert_old_resume_state(
    server: AppServer, source: AgentLoop, old_session_id: str, old_lease: SessionLease
) -> None:
    backend = legacy_backend(server)
    assert source.session_id == old_session_id
    assert source._session_lease is old_lease
    assert source.config.get_active_model().alias == "model-a"
    assert source.stats.session_prompt_tokens == 41
    assert [
        message.content for message in source.messages if message.role is Role.user
    ] == ["old transcript"]
    assert backend.coordinator.attached_session_id == old_session_id


@pytest.mark.asyncio
async def test_reload_failure_after_model_change_preserves_old_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        session,
        server,
        source,
        target_session_id,
        old_lease,
    ) = await _model_change_resume_context(tmp_path)
    old_session_id = source.session_id
    monkeypatch.setattr(
        source,
        "reload_with_initial_messages",
        AsyncMock(side_effect=RuntimeError("reload failed")),
    )
    try:
        with pytest.raises(Exception, match="reload failed"):
            await legacy_backend(server).handler.dispatch(
                "session/resume", _wire_resume_request(target_session_id)
            )
        _assert_old_resume_state(server, source, old_session_id, old_lease)
        with SessionLease(tmp_path, target_session_id):
            pass
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_cancellation_during_reload_preserves_old_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        session,
        server,
        source,
        target_session_id,
        old_lease,
    ) = await _model_change_resume_context(tmp_path)
    old_session_id = source.session_id
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_reload(**_kwargs: object) -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(source, "reload_with_initial_messages", blocked_reload)
    try:
        resume = asyncio.create_task(
            legacy_backend(server).handler.dispatch(
                "session/resume", _wire_resume_request(target_session_id)
            )
        )
        await entered.wait()
        resume.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resume
        _assert_old_resume_state(server, source, old_session_id, old_lease)
        with SessionLease(tmp_path, target_session_id):
            pass
    finally:
        release.set()
        await session.close()


@pytest.mark.asyncio
async def test_cancellation_during_lease_acquisition_no_state_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        session,
        server,
        source,
        target_session_id,
        old_lease,
    ) = await _model_change_resume_context(tmp_path)
    old_session_id = source.session_id
    entered = asyncio.Event()
    release = asyncio.Event()
    original_to_thread = asyncio.to_thread

    async def blocked_to_thread(
        function: object, /, *args: object, **kwargs: object
    ) -> object:
        if getattr(function, "__name__", None) == "_acquire_session_lease":
            entered.set()
            await release.wait()
        return await original_to_thread(function, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "chartreux.app_server._runtime.asyncio.to_thread", blocked_to_thread
    )
    try:
        resume = asyncio.create_task(
            legacy_backend(server).handler.dispatch(
                "session/resume", _wire_resume_request(target_session_id)
            )
        )
        await entered.wait()
        resume.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resume
        _assert_old_resume_state(server, source, old_session_id, old_lease)
        with SessionLease(tmp_path, target_session_id):
            pass
    finally:
        release.set()
        await session.close()


@pytest.mark.asyncio
async def test_scratchpad_preparation_failure_preserves_old_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        session,
        server,
        source,
        target_session_id,
        old_lease,
    ) = await _model_change_resume_context(tmp_path)
    old_session_id = source.session_id

    def fail_scratchpad(_session_id: str) -> Path:
        raise OSError("scratchpad preparation failed")

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.init_scratchpad", fail_scratchpad
    )
    try:
        with pytest.raises(Exception, match="scratchpad preparation failed"):
            await legacy_backend(server).handler.dispatch(
                "session/resume", _wire_resume_request(target_session_id)
            )
        _assert_old_resume_state(server, source, old_session_id, old_lease)
        with SessionLease(tmp_path, target_session_id):
            pass
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_cancellation_at_controller_turn_reset_boundary(tmp_path: Path) -> None:
    (
        session,
        server,
        source,
        target_session_id,
        _old_lease,
    ) = await _model_change_resume_context(tmp_path)
    backend = legacy_backend(server)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_reset() -> None:
        entered.set()
        await release.wait()

    backend.session.turns.reset = blocked_reset
    try:
        resume = asyncio.create_task(
            backend.handler.dispatch(
                "session/resume", _wire_resume_request(target_session_id)
            )
        )
        await entered.wait()
        resume.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resume
        assert (
            backend.coordinator.current_session_id
            == backend.coordinator.attached_session_id
        )
    finally:
        release.set()
        await session.close()


@pytest.mark.asyncio
async def test_successful_model_change_resume_persists_correct_state(
    tmp_path: Path,
) -> None:
    (
        session,
        server,
        source,
        target_session_id,
        _old_lease,
    ) = await _model_change_resume_context(tmp_path)
    old_session_id = source.session_id
    backend = legacy_backend(server)
    try:
        await backend.handler.dispatch(
            "session/resume", _wire_resume_request(target_session_id)
        )
        assert source.session_id == target_session_id
        assert backend.coordinator.attached_session_id == target_session_id
        assert source._session_lease is not None
        assert source.config.get_active_model().alias == "model-b"
        assert source.stats.session_prompt_tokens == 17
        assert [
            message.content for message in source.messages if message.role is Role.user
        ] == ["target transcript"]
        with SessionLease(tmp_path, old_session_id):
            pass
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_failed_resume_keeps_previous_root_and_reports_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = build_test_agent_loop(session_id="initial")
    factory = AgentRuntimeFactory()
    monkeypatch.setattr(
        factory,
        "resume_root",
        AsyncMock(side_effect=RuntimeError("boom while rebuilding")),
    )

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
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "session/resume", SessionResumeParams(session_id="replacement")
            )
        # The failure surfaces the underlying reason rather than a generic
        # "Invalid request parameters", and the previous root stays active.
        assert "Failed to resume session replacement" in exc_info.value.error.message
        assert "boom while rebuilding" in exc_info.value.error.message
        assert isinstance(server._root, SessionBackendImpl)
        assert server._root.session.agent_loop.session_id == "initial"
        # The still-attached root keeps serving requests instead of bricking.
        await client.request(
            "session/history/clear", SessionHistoryClearParams(session_id="initial")
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_busy_resume_reports_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    initial = build_test_agent_loop(session_id="initial")
    factory = AgentRuntimeFactory()
    monkeypatch.setattr(
        factory, "resume_root", AsyncMock(side_effect=SessionBusyError("replacement"))
    )

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
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "session/resume", SessionResumeParams(session_id="replacement")
            )
        assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT
        assert exc_info.value.error.message == "Session is already open: replacement"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_resume_reports_success_when_post_rebind_refresh_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

        # The rebind commits before root state is refreshed. A malformed stored
        # transcript can make project_history raise something other than the
        # ValidationError the projection already tolerates (e.g. KeyError). That
        # must not fail the resume nor leave the previous session's history behind.
        def boom(_loop: object) -> list[PublicHistoryEntry]:
            raise KeyError("unexpected stored shape")

        monkeypatch.setattr("chartreux.app_server._root_session.project_history", boom)
        await session.resume(saved_session_id)

        checkpoints = [
            entry
            for entry in session.history
            if isinstance(entry, PublicCheckpointEntry) and entry.kind == "resume"
        ]
        # Resume still succeeds: the checkpoint is appended and the view is rebound
        # to the resumed session with a degraded (empty) projected history rather
        # than the previous session's stale transcript.
        assert len(checkpoints) == 1
        assert not any(
            isinstance(entry, PublicMessageEntry) for entry in session.history
        )
        # The root is attached to the resumed session, so it keeps serving RPCs.
        result = await client.request(
            "session/turns/list",
            SessionTurnsListParams(
                session_id=saved_session_id, page=PageRequest(limit=10)
            ),
        )
        assert SessionTurnsListResponse.model_validate(result).items == []
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_reload_reports_pending_identity_change_without_reposturing(
    tmp_path: Path,
) -> None:
    # Design §10: a config reload mid-conversation retains the frozen policy
    # bytes and reports the pending [agent] selection change — the saved
    # selection applies to new conversations only.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    agent_loop = build_test_agent_loop(config=config)
    try:
        bound = agent_loop._bound_policy
        assert bound is not None
        frozen_prompt = agent_loop.messages[0].content

        set_agent_config(
            agent_loop, build_test_vibe_config(agent={"identity": "researcher"})
        )
        await agent_loop.reload_with_initial_messages()

        assert agent_loop._bound_policy is bound
        assert agent_loop.messages[0].content == frozen_prompt
        assert any(
            "identity selection change" in note
            for note in agent_loop.pending_policy_changes
        )
    finally:
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_reload_applies_route_edit_and_publishes_the_applied_notice() -> None:
    # ADR 0022 checks 3 and 4 end to end over the wire: a validated route
    # edit applied at a config reload is adopted by the admission source,
    # served by the task tool description (the served launch-target table
    # equals the admission table), and the applied notice publishes through
    # history/entryAdded — the "applies to new conversations only" route
    # pending text is gone.
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["task"], tools={"task": {"permission": "always"}}
        )
    )
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    try:
        await client.initialize(ClientInfo(name="route-edit-test", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())

        routes = dict(shipped_routes())
        routes["research"] = PurposeRoute(
            targets=(
                RouteTarget(
                    model=SESSION_MODEL, identity="coder", tier=PackageTier.SUBSTANTIVE
                ),
            )
        )
        set_agent_config(
            agent_loop,
            build_test_vibe_config(
                enabled_tools=["task"],
                tools={"task": {"permission": "always"}},
                orchestration={"routes": routes},
            ),
        )
        await client.request(
            "config/reload", ConfigReloadParams(session_id=agent_loop.session_id)
        )

        # The admission source is the edited live table...
        assert agent_loop.live_purpose_routes["research"].targets[0].identity == (
            "coder"
        )
        # ...and the served task tool description shows the same table.
        served = agent_loop.tool_manager._tool_descriptions["task"]
        research_row = next(
            line for line in served.splitlines() if line.startswith("| `research` |")
        )
        assert "`coder`" in research_row

        # Check 4: the applied notice publishes through history/entryAdded.
        incoming = client.incoming()
        async with asyncio.timeout(5):
            while True:
                message = await anext(incoming)
                if not (
                    isinstance(message, Notification)
                    and message.method == "history/entryAdded"
                ):
                    continue
                params = validate_wire(HistoryEntryAddedParams, message.params)
                entry = params.entry
                if isinstance(entry, PublicNoticeEntry) and isinstance(
                    entry.detail, RouteTableAppliedNoticeDetail
                ):
                    break
        await incoming.aclose()

        assert entry.message.startswith("Route table updated")
        assert "`research`" in entry.message
        assert "apply to subsequent launches" in entry.message
        assert "in-flight children keep their bindings" in entry.message
        assert [change.purpose for change in entry.detail.changes] == ["research"]
        change = entry.detail.changes[0]
        assert "researcher" in change.before
        assert "coder" in change.after
        # The route edit is applied, not pending: no route pending note, and
        # the identity/workflow/strategy pending branches stay silent too.
        assert agent_loop.pending_policy_changes == ()
    finally:
        await client.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_reload_persistence_failure_is_explicit_and_notifies_runtime_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    persist_launch_metadata = AsyncMock(side_effect=RuntimeError("disk unavailable"))
    monkeypatch.setattr(agent_loop, "persist_launch_metadata", persist_launch_metadata)

    try:
        await client.initialize(ClientInfo(name="reload-persistence-test", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())
        response = ConfigMutationResponse.model_validate(
            await client.request(
                "config/reload", ConfigReloadParams(session_id=agent_loop.session_id)
            )
        )

        assert response.launch_metadata_persisted is False
        persist_launch_metadata.assert_awaited_once()
        incoming = client.incoming()
        async with asyncio.timeout(1):
            while True:
                message = await anext(incoming)
                if (
                    isinstance(message, Notification)
                    and message.method == "runtime/updated"
                ):
                    break
        await incoming.aclose()

        update = validate_wire(RuntimeUpdatedParams, message.params)
        assert update.session_id == agent_loop.session_id
    finally:
        await client.close()
        await agent_loop.aclose()
