"""Dangerous-command prompts reach the client through the existing callback path.

The app-server binds the loop's DangerousCommandGrantPersistencePort when the
root or child session registers, so a mid-turn dangerous-command question flows
through the same user-input callback machinery as every other user question:
the client sees the typed request, its answer resolves exactly the outstanding
request, and the persistence outcome is reported apart from the live grant.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shlex
import subprocess
import tomllib
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._runtime import SessionDangerousCommandGrantPort
from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.models import (
    UserAnswer,
    UserInputCallbackDetail,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import ClientCapabilities, ClientInfo
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._dangerous_command_authority import GRANTS_FIELD
from chartreux.core.config._dangerous_command_persistence import SavedGrantsRead
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.events import BaseEvent, ToolResultEvent, UserInputRequestEvent
from chartreux.core.llm_models import FunctionCall, LLMChunk, Role, ToolCall
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    resolve_repository_scope,
)
from chartreux.questions import (
    UserAnswer as QuestionAnswer,
    UserQuestionResult as QuestionResult,
)
from chartreux.utils import AgentEntrypoint
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import build_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry

pytestmark = pytest.mark.asyncio


def make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def _bash_call(command: str, call_id: str, index: int = 0) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(name="bash", arguments=json.dumps({"command": command})),
    )


def _turn(command: str, call_id: str) -> list[list[LLMChunk]]:
    """One scripted turn: a bash call stream, then a closing stream."""
    return [
        [mock_llm_chunk(tool_calls=[_bash_call(command, call_id)])],
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
        client_info=ClientInfo(name="dangerous-prompt-test", version="0"),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
    )
    return server, client, session


async def _drive(session: AppServerSession, answer: str | None = None) -> list[str]:
    """Run one turn, answering every dangerous-command callback with *answer*."""
    callbacks: list[str] = []
    async for event in session.act("Exercise dangerous-command grants"):
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
    """Run one loop turn directly, answering every dangerous-command prompt."""
    events: list[BaseEvent] = []
    requests: list[UserInputRequestEvent] = []
    async with asyncio.timeout(10):
        async for event in agent.act("Exercise dangerous-command grants"):
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


def _saved_grants(config_path: Path) -> dict[str, list[str]]:
    if not config_path.exists():
        return {}
    data: dict[str, Any] = tomllib.loads(config_path.read_text())
    grants = data.get(GRANTS_FIELD, {})
    return {str(key): [str(item) for item in value] for key, value in grants.items()}


def _authorities(loop: AgentLoop) -> list[tuple[frozenset[str], object]]:
    return [
        (authority.operations, authority.scope)
        for authority in loop._dangerous_command_grants.dangerous_command_authorities()
    ]


def _scope_of(path: Path):
    scope = resolve_repository_scope(path, GitRepositoryRelocation())
    assert scope is not None
    return scope


async def test_prompt_reaches_the_client_and_the_answer_resolves_the_request(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    command = f"git push; touch {marker.name}"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "push-1")])
    )
    _, _, session = await _open_session(loop)
    try:
        callbacks: list[str] = []
        async for event in session.act("Exercise dangerous-command grants"):
            if isinstance(event, CallbackRequested):
                callbacks.append(event.callback.callback_id)
                detail = event.callback.detail
                assert isinstance(detail, UserInputCallbackDetail)
                question = detail.request.questions[0]
                assert question.header == "Dangerous command"
                # The literal whole call is displayed, shlex-joined.
                assert (
                    shlex.join(("git", "push;", "touch", marker.name))
                    in question.question
                )
                assert str(project) in question.question
                assert "git.push" in question.question
                assert [option.label for option in question.options] == [
                    "Deny",
                    "Allow once",
                    "Allow this session",
                    "Always for this project",
                ]
                await session.respond_to_callback(
                    event.callback.callback_id,
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[
                                UserAnswer(
                                    question="grant", answer="Allow this session"
                                )
                            ]
                        )
                    ),
                )
        # Exactly one prompt, and the answer resolved the outstanding request:
        # the original call resumed and executed.
        assert len(callbacks) == 1
        assert marker.exists()
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(project))]
        assert config_path.read_bytes() == disk
    finally:
        await session.close()
        await loop.aclose()


async def test_answering_a_question_pending_across_a_revoke_still_grants(
    tmp_path: Path, config_dir: Path
) -> None:
    """A question pending in the broker survives a revocation.

    Revocation invalidates installed once permits, not a question already
    awaiting an answer: answering it after the revoke still grants fresh
    authority for the decided scope, and the original call resumes.
    """
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    command = f"git push; touch {marker.name}"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "pending-1")])
    )
    _, _, session = await _open_session(loop)
    try:
        callbacks: list[str] = []
        async for event in session.act("Exercise dangerous-command grants"):
            if isinstance(event, CallbackRequested):
                callbacks.append(event.callback.callback_id)
                # The question is now pending in the broker: revoke first.
                # No live authority exists yet, so the live removal is
                # reported as removed without removing anything.
                read = await session.resources.config.read_dangerous_command_authority()
                revocation = (
                    await session.resources.config.revoke_dangerous_command_grant(
                        operation_id="git.push",
                        expected_live_generation=read.live_authority_generation,
                        expected_disk_revision=read.disk_revision,
                        user_initiated=True,
                    )
                )
                assert revocation.live.status == "removed"
                assert revocation.saved.status == "removed"
                assert revocation.warning is None
                assert _authorities(loop) == []
                # The pending question was not cancelled: answering it after
                # the revocation installs fresh authority for the decided
                # scope, and the original call resumed and executed.
                await session.respond_to_callback(
                    event.callback.callback_id,
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[
                                UserAnswer(
                                    question="grant", answer="Allow this session"
                                )
                            ]
                        )
                    ),
                )
        assert len(callbacks) == 1
        assert marker.exists()
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(project))]
    finally:
        await session.close()
        await loop.aclose()


async def test_allow_once_authorizes_exactly_one_call_over_the_wire(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    command = f"git push; touch {marker.name}"
    backend = FakeBackend([*[*_turn(command, "once-1")], *[*_turn(command, "once-2")]])
    loop = await _make_loop(project, config_path, backend)
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, "Allow once")
        assert len(callbacks) == 1
        assert marker.exists()
        # The once permit authorized exactly this invocation: no authority
        # remains, so the identical second call prompts again instead of
        # resuming on the revoked approval.
        assert _authorities(loop) == []
        callbacks = await _drive(session, "Deny")
        assert len(callbacks) == 1
        assert _authorities(loop) == []
        assert any(
            "user declined; do not retry this command" in text
            for text in _tool_texts(loop)
        )
    finally:
        await session.close()
        await loop.aclose()


async def test_always_for_this_project_saves_the_grant_and_resumes_the_call(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    command = f"git push; touch {marker.name}"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "always-1")])
    )
    _, _, session = await _open_session(loop)
    try:
        # The runtime construction replaced the loop's default port: the
        # registry routes saves through the root orchestrator.
        assert isinstance(
            loop._dangerous_command_grant_port, SessionDangerousCommandGrantPort
        )
        callbacks = await _drive(session, "Always for this project")
        assert len(callbacks) == 1
        assert marker.exists()
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(project))]
        note = next(
            text for text in _tool_texts(loop) if "Dangerous-command grant" in text
        )
        assert "applies to this session" in note
        assert "saved in your user config" in note
        assert str(project.resolve()) in note
        # The user config gains the project entry.
        assert _saved_grants(config_path) == {str(project.resolve()): ["git.push"]}
    finally:
        await session.close()
        await loop.aclose()


async def test_forged_consequential_always_is_rejected_over_the_wire(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    command = f"git stash drop; touch {marker.name}"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "forged-1")])
    )
    _, _, session = await _open_session(loop)
    try:
        callbacks: list[str] = []
        async for event in session.act("Exercise dangerous-command grants"):
            if isinstance(event, CallbackRequested):
                callbacks.append(event.callback.callback_id)
                detail = event.callback.detail
                assert isinstance(detail, UserInputCallbackDetail)
                question = detail.request.questions[0]
                # A consequential call never offers the persistent choice.
                assert [option.label for option in question.options] == [
                    "Deny",
                    "Allow once",
                    "Allow this session",
                ]
                # The forged answer is rejected at parse time: it authorizes
                # nothing, exactly as an in-process completion would.
                await session.respond_to_callback(
                    event.callback.callback_id,
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[
                                UserAnswer(
                                    question="grant", answer="Always for this project"
                                )
                            ]
                        )
                    ),
                )
        assert len(callbacks) == 1
        assert not marker.exists()
        assert _authorities(loop) == []
        assert any(
            "user declined; do not retry this command" in text
            for text in _tool_texts(loop)
        )
        assert config_path.read_bytes() == disk
    finally:
        await session.close()
        await loop.aclose()


async def test_a_non_choice_answer_cannot_masquerade_as_an_approval(
    tmp_path: Path, config_dir: Path
) -> None:
    """An ordinary free-text completion fails closed over the wire."""
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    command = f"git push; touch {marker.name}"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "free-1")])
    )
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, "yes, please")
        assert len(callbacks) == 1
        assert not marker.exists()
        assert _authorities(loop) == []
        assert any(
            "user declined; do not retry this command" in text
            for text in _tool_texts(loop)
        )
    finally:
        await session.close()
        await loop.aclose()


async def test_save_conflict_leaves_the_live_approval_intact_and_reports_conflict(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    command = f"git push; touch {marker.name}"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "conflict-1")])
    )
    _, _, session = await _open_session(loop)
    # The loop reads a stale user revision, so the real save path conflicts.
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.read_saved_dangerous_command_grants",
        AsyncMock(
            return_value=SavedGrantsRead(operations=(), user_revision="vibe:stale")
        ),
    )
    try:
        callbacks = await _drive(session, "Always for this project")
        assert len(callbacks) == 1
        # The live approval survives the conflict and the call resumed.
        assert marker.exists()
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(project))]
        note = next(
            text for text in _tool_texts(loop) if "Dangerous-command grant" in text
        )
        assert "applies to this session only" in note
        assert "retry" in note
        # No automatic retry: nothing reached the config.
        assert _saved_grants(config_path) == {}
    finally:
        await session.close()
        await loop.aclose()


async def test_denied_answer_grants_nothing(tmp_path: Path, config_dir: Path) -> None:
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    command = f"git push; touch {marker.name}"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "deny-1")])
    )
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, "Deny")
        assert len(callbacks) == 1
        assert not marker.exists()
        assert _authorities(loop) == []
        assert any(
            "user declined; do not retry this command" in text
            for text in _tool_texts(loop)
        )
        assert config_path.read_bytes() == disk
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.parametrize("entrypoint", ["programmatic", "acp"])
async def test_session_without_user_input_capability_denies_without_prompting(
    tmp_path: Path, config_dir: Path, entrypoint: AgentEntrypoint
) -> None:
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    command = f"git push; touch {marker.name}"
    user = UserConfigLayer(path=config_path)
    overlay = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, overlay],
        default_layer_resolver=lambda: overlay,
    )
    loop = AgentLoop(
        config_orchestrator=orchestrator,
        cwd=project,
        backend=FakeBackend([*_turn(command, "no-input-1")]),
        mcp_registry=FakeMCPRegistry(),
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(
            name="dangerous-prompt-test", version="0", entrypoint=entrypoint
        ),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
    )
    try:
        callbacks = await _drive(session, "Allow this session")
        assert callbacks == []
        assert not marker.exists()
        assert _authorities(loop) == []
        texts = _tool_texts(loop)
        assert any(GRANTS_FIELD in text and "config.toml" in text for text in texts)
        assert config_path.read_bytes() == disk
    finally:
        await session.close()
        await loop.aclose()


async def test_a_project_grant_does_not_follow_another_worktree(
    tmp_path: Path, config_dir: Path
) -> None:
    """Worktree isolation: authority in one repository never covers another."""
    project = make_repo(tmp_path / "project")
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "init",
        ],
        check=True,
    )
    worktree = project / "wt"
    subprocess.run(
        ["git", "-C", str(project), "worktree", "add", "-q", str(worktree), "-b", "wt"],
        check=True,
    )
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    backend = FakeBackend([
        *[*_turn(f"git -C wt push; touch {marker.name}", "wt-1")],
        *[*_turn("git push", "proj-1")],
    ])
    loop = await _make_loop(project, config_path, backend)
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, "Allow this session")
        assert len(callbacks) == 1
        assert marker.exists()
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(worktree))]
        # The main worktree is a different repository identity: the project
        # grant does not follow it, so the second call prompts again.
        callbacks = await _drive(session, "Deny")
        assert len(callbacks) == 1
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(worktree))]
    finally:
        await session.close()
        await loop.aclose()


async def test_child_with_user_input_capability_denies_ask_class_and_reports_a_blocker(
    tmp_path: Path, config_dir: Path
) -> None:
    """Blanket child denial: a capable child never surfaces the approval prompt.

    The child's own gap is locally grantable (the parent holds session
    authority, so this is not a parent ceiling), yet the child denies with
    remediation instead of prompting: the blocker flows back to the parent
    and dangerous_command_grants_by_project stays the only child
    pre-authorization path.
    """
    project = make_repo(tmp_path / "project")
    nested = project / "nested"
    nested.mkdir()
    marker = nested / "executed.txt"
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    parent = await _make_loop(project, config_path, FakeBackend())
    server, _, session = await _open_session(parent)
    try:
        root_backend = server._require_root()
        assert isinstance(root_backend, SessionBackendImpl)
        registry = root_backend.children
        assert isinstance(registry, SessionRuntimeRegistry)
        policy = parent.child_runtime_policy
        # The parent already holds session authority, so the child's own gap
        # is locally grantable instead of a parent ceiling.
        parent._dangerous_command_grants.grant_session({
            ("git.push", _scope_of(project))
        })
        parent.tool_manager._invalidate_dangerous_command_authority()

        child = AgentLoop(
            config_orchestrator=parent.config_orchestrator.copy(),
            cwd=nested,
            backend=FakeBackend([*_turn(f"git push; touch {marker.name}", "child-1")]),
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
        registry._children[child.session_id] = registry._build_child_runtime(child)
        # The runtime construction bound the child's persistence port too.
        assert isinstance(
            child._dangerous_command_grant_port, SessionDangerousCommandGrantPort
        )
        try:
            events, requests = await _drive_loop(child, "Always for this project")
            # No prompt surfaces even with user-input capability and a bound
            # persistence port: the ask-class operation denies outright.
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert not marker.exists()
            assert any(
                GRANTS_FIELD in text and "config.toml" in text
                for text in _tool_texts(child)
            )
            # No authority was installed anywhere and nothing was persisted.
            assert _authorities(child) == []
            assert _authorities(parent) == [
                (frozenset({"git.push"}), _scope_of(project))
            ]
            assert _saved_grants(config_path) == {}
            assert config_path.read_bytes() == disk
        finally:
            await child.aclose()
    finally:
        await session.close()
        await parent.aclose()


async def test_paired_main_and_child_for_the_same_ungranted_ask_class_call(
    tmp_path: Path, config_dir: Path
) -> None:
    """The same ungranted call prompts in the main session, denies in a child."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    marker_name = "executed.txt"
    command = f"git push; touch {marker_name}"
    main = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "main-1")])
    )
    _, _, session = await _open_session(main)
    try:
        callbacks = await _drive(session, "Allow once")
        # The main session surfaces the approval dialog and the granted call
        # executes.
        assert len(callbacks) == 1
        assert (project / marker_name).exists()
    finally:
        await session.close()
        await main.aclose()
    # Clear the main half's marker so the child half starts unexecuted.
    (project / marker_name).unlink()

    parent = await _make_loop(project, config_path, FakeBackend())
    server, _, session = await _open_session(parent)
    try:
        root_backend = server._require_root()
        assert isinstance(root_backend, SessionBackendImpl)
        registry = root_backend.children
        assert isinstance(registry, SessionRuntimeRegistry)
        policy = parent.child_runtime_policy
        # The parent holds session authority so the child's gap is locally
        # grantable instead of a parent ceiling: the paired child denial is
        # the blanket child gate, not a ceiling denial.
        parent._dangerous_command_grants.grant_session({
            ("git.push", _scope_of(project))
        })
        parent.tool_manager._invalidate_dangerous_command_authority()
        child = AgentLoop(
            config_orchestrator=parent.config_orchestrator.copy(),
            cwd=project,
            backend=FakeBackend([*_turn(command, "child-1")]),
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
        registry._children[child.session_id] = registry._build_child_runtime(child)
        try:
            events, requests = await _drive_loop(child, "Allow once")
            # The child denies the same operation with remediation.
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert not (project / marker_name).exists()
            assert any(
                GRANTS_FIELD in text and "config.toml" in text
                for text in _tool_texts(child)
            )
            assert _authorities(child) == []
            assert config_path.read_bytes() == disk
        finally:
            await child.aclose()
    finally:
        await session.close()
        await parent.aclose()


async def test_child_cannot_lift_a_parent_dangerous_ceiling(
    tmp_path: Path, config_dir: Path
) -> None:
    """A parent's dangerous gap is a ceiling: no child grant can lift it."""
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    parent = await _make_loop(project, config_path, FakeBackend())
    server, _, session = await _open_session(parent)
    try:
        root_backend = server._require_root()
        assert isinstance(root_backend, SessionBackendImpl)
        registry = root_backend.children
        policy = parent.child_runtime_policy
        child = AgentLoop(
            config_orchestrator=parent.config_orchestrator.copy(),
            cwd=project,
            backend=FakeBackend([*_turn(f"git push; touch {marker.name}", "child-1")]),
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
        registry._children[child.session_id] = registry._build_child_runtime(child)
        try:
            events, requests = await _drive_loop(child, "Allow this session")
            # The parent's gap is a ceiling: the child never prompts and no
            # local grant is installed.
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert not marker.exists()
            assert _authorities(child) == []
            assert _authorities(parent) == []
        finally:
            await child.aclose()
    finally:
        await session.close()
        await parent.aclose()


async def test_child_inherits_approved_project_authority(
    tmp_path: Path, config_dir: Path
) -> None:
    """A saved project grant travels down the tree; the child needs no prompt."""
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    # The saved grant is accepted by the parent's orchestrator before the
    # session starts, so both parent and child enforce it.
    config_path.write_text(
        config_path.read_text()
        + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n',
        encoding="utf-8",
    )
    parent = await _make_loop(project, config_path, FakeBackend())
    server, _, session = await _open_session(parent)
    try:
        root_backend = server._require_root()
        assert isinstance(root_backend, SessionBackendImpl)
        registry = root_backend.children
        policy = parent.child_runtime_policy
        child = AgentLoop(
            config_orchestrator=parent.config_orchestrator.copy(),
            cwd=project,
            backend=FakeBackend([*_turn(f"git push; touch {marker.name}", "child-1")]),
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
        registry._children[child.session_id] = registry._build_child_runtime(child)
        try:
            events, requests = await _drive_loop(child)
            # The inherited project authority covers the child's identical
            # call: no prompt, no local grant.
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert not results[0].skipped
            assert marker.exists()
            assert _authorities(child) == []
            assert _authorities(parent) == []
        finally:
            await child.aclose()
    finally:
        await session.close()
        await parent.aclose()


# v0.5.9 P2-a: the destination-aware egress rule resolves non-loopback
# network-client egress to the same ask-class as git push, so the P1-g
# gates compose unchanged — the dialog in an interactive main session,
# denial with remediation in headless and child sessions. The command
# targets TEST-NET-1 (192.0.2.1, RFC 5737) with a one-second cap: the
# curl fails fast and the trailing touch still proves execution.
EGRESS_COMMAND = "curl -s --max-time 1 https://192.0.2.1/; touch {marker}"


async def test_paired_main_and_child_for_the_same_ungranted_egress_call(
    tmp_path: Path, config_dir: Path
) -> None:
    """The same ungranted egress call prompts in the main session, denies in a child."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    marker_name = "executed.txt"
    command = EGRESS_COMMAND.format(marker=marker_name)
    main = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "main-1")])
    )
    _, _, session = await _open_session(main)
    try:
        callbacks: list[str] = []
        async for event in session.act("Exercise dangerous-command grants"):
            if isinstance(event, CallbackRequested):
                callbacks.append(event.callback.callback_id)
                detail = event.callback.detail
                assert isinstance(detail, UserInputCallbackDetail)
                question = detail.request.questions[0]
                # The egress approval is not described as a git command.
                assert question.question.startswith("Allow this dangerous command?")
                assert "network.egress" in question.question
                assert "192.0.2.1" in question.question
                # network.egress is publication-class: the persistent
                # choice is offered, exactly like git push.
                assert [option.label for option in question.options] == [
                    "Deny",
                    "Allow once",
                    "Allow this session",
                    "Always for this project",
                ]
                await session.respond_to_callback(
                    event.callback.callback_id,
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[UserAnswer(question="grant", answer="Allow once")]
                        )
                    ),
                )
        # The main session surfaces the approval dialog, and the once
        # permit covers the egress requirement: the call executes.
        assert len(callbacks) == 1
        assert (project / marker_name).exists()
    finally:
        await session.close()
        await main.aclose()
    # Clear the main half's marker so the child half starts unexecuted.
    (project / marker_name).unlink()

    parent = await _make_loop(project, config_path, FakeBackend())
    server, _, session = await _open_session(parent)
    try:
        root_backend = server._require_root()
        assert isinstance(root_backend, SessionBackendImpl)
        registry = root_backend.children
        assert isinstance(registry, SessionRuntimeRegistry)
        policy = parent.child_runtime_policy
        # The parent holds session authority so the child's gap is locally
        # grantable instead of a parent ceiling: the paired child denial
        # is the blanket child gate (P1-g), not a ceiling denial.
        parent._dangerous_command_grants.grant_session({
            ("network.egress", _scope_of(project))
        })
        parent.tool_manager._invalidate_dangerous_command_authority()
        child = AgentLoop(
            config_orchestrator=parent.config_orchestrator.copy(),
            cwd=project,
            backend=FakeBackend([*_turn(command, "child-1")]),
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
        registry._children[child.session_id] = registry._build_child_runtime(child)
        try:
            events, requests = await _drive_loop(child, "Allow once")
            # The child denies the same egress call with remediation: no
            # prompt even with user-input capability, no execution, and the
            # config knob is named.
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert not (project / marker_name).exists()
            assert any(
                GRANTS_FIELD in text and "config.toml" in text
                for text in _tool_texts(child)
            )
            assert _authorities(child) == []
            assert config_path.read_bytes() == disk
        finally:
            await child.aclose()
    finally:
        await session.close()
        await parent.aclose()


@pytest.mark.parametrize("entrypoint", ["programmatic", "acp"])
async def test_headless_egress_denies_with_remediation(
    tmp_path: Path, config_dir: Path, entrypoint: AgentEntrypoint
) -> None:
    """A headless session denies egress with remediation instead of prompting."""
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    command = EGRESS_COMMAND.format(marker=marker.name)
    user = UserConfigLayer(path=config_path)
    overlay = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, overlay],
        default_layer_resolver=lambda: overlay,
    )
    loop = AgentLoop(
        config_orchestrator=orchestrator,
        cwd=project,
        backend=FakeBackend([*_turn(command, "egress-headless-1")]),
        mcp_registry=FakeMCPRegistry(),
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(
            name="dangerous-prompt-test", version="0", entrypoint=entrypoint
        ),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
    )
    try:
        callbacks = await _drive(session, "Allow this session")
        assert callbacks == []
        assert not marker.exists()
        assert _authorities(loop) == []
        texts = _tool_texts(loop)
        assert any(GRANTS_FIELD in text and "config.toml" in text for text in texts)
        assert config_path.read_bytes() == disk
    finally:
        await session.close()
        await loop.aclose()
