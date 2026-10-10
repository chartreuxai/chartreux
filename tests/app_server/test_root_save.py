"""Project root-grant persistence: the policy/roots/save RPC and its prompt wiring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tomllib
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.models import (
    UserAnswer,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ClientInfo,
)
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._root_authority import ROOTS_FIELD
from chartreux.core.config._root_persistence import SavedRootsRead
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.events import BaseEvent, ToolResultEvent, UserInputRequestEvent
from chartreux.core.llm_models import FunctionCall, LLMChunk, Role, ToolCall
from chartreux.questions import (
    UserAnswer as QuestionAnswer,
    UserQuestionResult as QuestionResult,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import build_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry

pytestmark = pytest.mark.asyncio


def _read_call(target: Path, call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="read_file", arguments=json.dumps({"file_path": str(target)})
        ),
    )


def _read_turn(target: Path, call_id: str) -> list[list[LLMChunk]]:
    """One scripted turn: a read_file call stream, then a closing stream."""
    return [
        [mock_llm_chunk(tool_calls=[_read_call(target, call_id)])],
        [mock_llm_chunk(content="done")],
    ]


async def _make_loop(
    project: Path, config_path: Path, backend: FakeBackend
) -> AgentLoop:
    """Build a root loop over a real user layer so saves reach the config file."""
    user = UserConfigLayer(path=config_path)
    overlay = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, overlay],
        default_layer_resolver=lambda: overlay,
    )
    return AgentLoop(
        config_orchestrator=orchestrator,
        cwd=project,
        backend=backend,
        mcp_registry=FakeMCPRegistry(),
        user_input_capability=True,
    )


async def _open_session(
    loop: AgentLoop,
) -> tuple[AppServer, AppServerClient, AppServerSession]:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="root-save-test", version="0"),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
    )
    return server, client, session


async def _drive(session: AppServerSession, answer: str | None = None) -> list[str]:
    """Run one turn, answering every root-grant callback with *answer*."""
    callbacks: list[str] = []
    async for event in session.act("Exercise root grants"):
        if isinstance(event, CallbackRequested):
            callbacks.append(event.callback.callback_id)
            if answer is not None:
                await session.respond_to_callback(
                    event.callback.callback_id,
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[UserAnswer(question="grant", answer=answer)]
                        )
                    ),
                )
    return callbacks


async def _drive_loop(
    agent: AgentLoop, answer: str | None = None
) -> tuple[list[BaseEvent], list[UserInputRequestEvent]]:
    """Run one loop turn directly, answering every root-grant prompt."""
    events: list[BaseEvent] = []
    requests: list[UserInputRequestEvent] = []
    async with asyncio.timeout(10):
        async for event in agent.act("Exercise root grants"):
            if isinstance(event, UserInputRequestEvent):
                requests.append(event)
                agent.resolve_user_input_request(
                    event.request_id,
                    QuestionResult(
                        answers=[QuestionAnswer(question="grant", answer=answer or "")]
                    ),
                )
            events.append(event)
    return events, requests


def _results(events: list[BaseEvent]) -> list[ToolResultEvent]:
    return [event for event in events if isinstance(event, ToolResultEvent)]


def _tool_texts(loop: AgentLoop) -> list[str]:
    return [
        message.content or "" for message in loop.messages if message.role == Role.tool
    ]


def _saved_roots(config_path: Path) -> dict[str, list[str]]:
    if not config_path.exists():
        return {}
    data: dict[str, Any] = tomllib.loads(config_path.read_text())
    roots = data.get(ROOTS_FIELD, {})
    return {str(key): [str(root) for root in value] for key, value in roots.items()}


async def test_always_prompt_saves_the_grant_and_resumes_the_call(
    tmp_path: Path, config_dir: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    config_path = config_dir / "config.toml"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_read_turn(target, "read-1")])
    )
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, "Always for this project")
        assert len(callbacks) == 1
        # The session grant applies exactly like Allow-session does.
        assert loop._session_root_grants.roots == {outside.resolve()}
        assert loop.tool_manager.workspace.allows(target)
        texts = _tool_texts(loop)
        assert any("granted content" in text for text in texts)
        note = next(text for text in texts if "Root grant for" in text)
        assert "applies to this session" in note
        assert "saved in your user config" in note
        assert str(project.resolve()) in note
        # The user config gains the project entry.
        assert _saved_roots(config_path) == {
            str(project.resolve()): [str(outside.resolve())]
        }
    finally:
        await session.close()
        await loop.aclose()


async def test_save_succeeds_during_an_active_turn_without_an_idle_reservation(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    config_path = config_dir / "config.toml"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_read_turn(target, "read-1")])
    )
    server, _, session = await _open_session(loop)

    def _no_idle_reservation(registry: SessionRuntimeRegistry) -> None:
        raise AssertionError("policy/roots/save must not reserve tree idle admission")

    monkeypatch.setattr(SessionRuntimeRegistry, "reserve_config", _no_idle_reservation)
    try:
        callbacks: list[str] = []
        async for event in session.act("read the outside file"):
            if isinstance(event, CallbackRequested):
                callbacks.append(event.callback.callback_id)
                backend = server._require_root()
                assert isinstance(backend, SessionBackendImpl)
                # The turn is active with the root-grant prompt pending.
                assert backend.session.turns.active_turn is not None
                read = await session.resources.config.read_roots()
                assert read.user_revision
                response = await session.resources.config.save_root(
                    root=str(outside.resolve()),
                    expected_revision=read.user_revision,
                    user_initiated=True,
                )
                assert response.persistence == "saved"
                assert response.error is None
                assert response.application == "unchanged"
                assert _saved_roots(config_path) == {
                    str(project.resolve()): [str(outside.resolve())]
                }
                await session.respond_to_callback(
                    callbacks[0],
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[UserAnswer(question="grant", answer="Deny")]
                        )
                    ),
                )
        assert len(callbacks) == 1
        # The save persisted nothing to the session: the denial still stands.
        assert loop._session_root_grants.roots == set()
        assert loop._session_root_grants.denied_roots == {outside.resolve()}
    finally:
        await session.close()
        await loop.aclose()


async def test_child_always_saves_through_the_root_orchestrator_keyed_by_child_cwd(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    nested = project / "nested"
    nested.mkdir()
    target = project / "note.txt"
    target.write_text("parent project content")
    config_path = config_dir / "config.toml"
    parent = await _make_loop(project, config_path, FakeBackend())
    server, _, session = await _open_session(parent)
    try:
        root_backend = server._require_root()
        assert isinstance(root_backend, SessionBackendImpl)
        registry = root_backend.children
        policy = parent.child_runtime_policy

        def build_child(call_id: str) -> AgentLoop:
            return AgentLoop(
                config_orchestrator=parent.config_orchestrator.copy(),
                cwd=nested,
                backend=FakeBackend([*_read_turn(target, call_id)]),
                is_subagent=True,
                inherited_workspace=policy.inherited_workspace,
                inherited_restrictions=policy.inherited_restrictions,
                inherited_mode_restrictions=policy.inherited_mode_restrictions,
                inherited_plan_write_scopes=policy.inherited_plan_write_scopes,
                inherited_root_grants=policy.inherited_root_grants,
                user_input_capability=True,
                parent_authority_getter=policy.parent_authority_getter,
                parent_authority_revision_getter=policy.parent_authority_revision_getter,
            )

        child = build_child("child-read")
        sibling = build_child("sibling-read")
        registry._children[child.session_id] = registry._build_child_runtime(child)
        registry._children[sibling.session_id] = registry._build_child_runtime(sibling)
        # The root orchestrator writes; a child orchestrator never does.
        root_save = AsyncMock(wraps=parent.config_orchestrator.save_project_root_grant)
        monkeypatch.setattr(
            parent.config_orchestrator, "save_project_root_grant", root_save
        )
        child_save = AsyncMock(
            side_effect=AssertionError("child orchestrators never write")
        )
        monkeypatch.setattr(
            child.config_orchestrator, "save_project_root_grant", child_save
        )
        try:
            events, requests = await _drive_loop(child, "Always for this project")
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert not results[0].skipped
            assert results[0].result is not None
            assert "parent project content" in results[0].result.model_dump_json()
            # The calling child gets the grant; parent and sibling stay unchanged.
            assert child._session_root_grants.roots == {project.resolve()}
            assert parent._session_root_grants.roots == set()
            assert sibling._session_root_grants.roots == set()
            assert not sibling.tool_manager.workspace.allows(target)
            # The save routed through the root orchestrator, keyed by the
            # child's registered cwd, never by a caller-supplied project path.
            assert root_save.call_count == 1
            assert root_save.call_args.kwargs["project"] == nested.resolve()
            assert root_save.call_args.kwargs["root"] == project.resolve()
            child_save.assert_not_called()
            assert _saved_roots(config_path) == {
                str(nested.resolve()): [str(project.resolve())]
            }
        finally:
            await child.aclose()
            await sibling.aclose()
    finally:
        await session.close()
        await parent.aclose()


async def test_a_fresh_session_reads_the_saved_root_from_disk(
    tmp_path: Path, config_dir: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    config_path = config_dir / "config.toml"
    first = await _make_loop(project, config_path, FakeBackend())
    _, _, first_session = await _open_session(first)
    try:
        read = await first_session.resources.config.read_roots()
        revision = read.user_revision
        assert revision is not None
        saved = await first_session.resources.config.save_root(
            root=str(outside.resolve()), expected_revision=revision, user_initiated=True
        )
        assert saved.persistence == "saved"
        assert saved.error is None
    finally:
        await first_session.close()
        await first.aclose()
    # A fresh session with the same cwd sees the saved root on disk.
    second = await _make_loop(project, config_path, FakeBackend())
    _, _, second_session = await _open_session(second)
    try:
        reread = await second_session.resources.config.read_roots()
        assert reread.saved_roots == [str(outside.resolve())]
        assert reread.user_revision == saved.revision
        assert _saved_roots(config_path) == {
            str(project.resolve()): [str(outside.resolve())]
        }
    finally:
        await second_session.close()
        await second.aclose()


async def test_always_prompt_reports_a_stale_revision_conflict_without_retry(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    config_path = config_dir / "config.toml"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_read_turn(target, "read-1")])
    )
    _, _, session = await _open_session(loop)
    # The loop reads a stale user revision, so the real save path conflicts.
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.read_saved_roots",
        AsyncMock(return_value=SavedRootsRead(roots=(), user_revision="vibe:stale")),
    )
    try:
        callbacks = await _drive(session, "Always for this project")
        assert len(callbacks) == 1
        # The session grant is still applied and the tool call resumed.
        assert loop._session_root_grants.roots == {outside.resolve()}
        assert loop.tool_manager.workspace.allows(target)
        texts = _tool_texts(loop)
        assert any("granted content" in text for text in texts)
        note = next(text for text in texts if "Root grant for" in text)
        assert "applies to this session only" in note
        assert "retry" in note
        # No automatic retry: nothing reached the config.
        assert _saved_roots(config_path) == {}
    finally:
        await session.close()
        await loop.aclose()


async def test_save_reports_a_stale_revision_as_a_conflict_result(
    tmp_path: Path, config_dir: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    config_path = config_dir / "config.toml"
    loop = await _make_loop(project, config_path, FakeBackend())
    _, _, session = await _open_session(loop)
    try:
        stale = await session.resources.config.save_root(
            root=str(outside.resolve()),
            expected_revision="vibe:stale-revision",
            user_initiated=True,
        )
        assert stale.persistence == "not_saved"
        assert stale.error == "conflict"
        assert stale.application == "unchanged"
        # Nothing was written and the session authority is untouched.
        assert _saved_roots(config_path) == {}
        assert loop._session_root_grants.roots == set()
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.parametrize(
    ("result", "persistence", "error"),
    [
        (
            ConfigSaveResult("user", "not_saved", "unchanged", error="write"),
            "not_saved",
            "write",
        ),
        (
            ConfigSaveResult("user", "durability_uncertain", "unchanged", "rev-2"),
            "durability_uncertain",
            None,
        ),
        (
            ConfigSaveResult("user", "saved", "unchanged", "rev-3", "cancelled"),
            "saved",
            "cancelled",
        ),
    ],
)
async def test_save_reports_persistence_outcomes_honestly(
    tmp_path: Path,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: ConfigSaveResult,
    persistence: str,
    error: str | None,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    config_path = config_dir / "config.toml"
    loop = await _make_loop(project, config_path, FakeBackend())
    _, _, session = await _open_session(loop)
    try:
        monkeypatch.setattr(
            loop.config_orchestrator,
            "save_project_root_grant",
            AsyncMock(return_value=result),
        )
        response = await session.resources.config.save_root(
            root=str(outside.resolve()), expected_revision="rev-1", user_initiated=True
        )
        assert response.persistence == persistence
        assert response.error == error
        assert response.application == "unchanged"
        assert response.target == "user"
    finally:
        await session.close()
        await loop.aclose()


async def test_legacy_roots_read_and_replace_round_trip_survives_a_save(
    tmp_path: Path, config_dir: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    session_root = tmp_path / "session-root"
    session_root.mkdir()
    config_path = config_dir / "config.toml"
    loop = await _make_loop(project, config_path, FakeBackend())
    _, _, session = await _open_session(loop)
    try:
        resource = session.resources.config
        first = await resource.read_roots()
        assert first.roots == []
        assert first.saved_roots == []
        replaced = await resource.replace_roots(
            roots=[str(session_root)],
            expected_revision=first.revision,
            user_initiated=True,
        )
        assert replaced.revision
        second = await resource.read_roots()
        assert second.roots == [str(session_root)]
        user_revision = second.user_revision
        assert user_revision is not None
        saved = await resource.save_root(
            root=str(outside.resolve()),
            expected_revision=user_revision,
            user_initiated=True,
        )
        assert saved.persistence == "saved"
        third = await resource.read_roots()
        # The save published nothing to the session roots.
        assert third.roots == [str(session_root)]
        assert third.saved_roots == [str(outside.resolve())]
        restored = await resource.replace_roots(
            roots=[], expected_revision=third.revision, user_initiated=True
        )
        assert restored.revision
        final = await resource.read_roots()
        assert final.roots == []
        assert final.saved_roots == [str(outside.resolve())]
        assert _saved_roots(config_path) == {
            str(project.resolve()): [str(outside.resolve())]
        }
    finally:
        await session.close()
        await loop.aclose()


async def test_save_requires_user_initiation_and_a_registered_session(
    tmp_path: Path, config_dir: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    config_path = config_dir / "config.toml"
    loop = await _make_loop(project, config_path, FakeBackend())
    _, client, session = await _open_session(loop)
    try:
        base: dict[str, Any] = {
            "sessionId": loop.session_id,
            "root": str(outside.resolve()),
            "expectedRevision": "rev-1",
            "userInitiated": True,
        }
        with pytest.raises(AppServerResponseError):
            await client.request("policy/roots/save", base | {"userInitiated": False})
        with pytest.raises(AppServerResponseError):
            await client.request(
                "policy/roots/save", base | {"sessionId": "unregistered-session"}
            )
        # Neither rejected attempt wrote anything.
        assert _saved_roots(config_path) == {}
    finally:
        await session.close()
        await loop.aclose()
