"""Cross-boundary acceptance tests for dangerous-command permissions.

These tests compose the whole feature through real registered sessions: a
real app server over an in-process transport, a real agent loop over a real
user config layer (so persistence reaches the actual config file), a real
client session whose dangerous-command question travels the user-input
callback wire, and real git repositories whose only remote is a local bare
repository. No model call is real: the fake backend scripts every turn.

The scenarios pin the composed system rather than units: each approval choice
resolving through the registered session (deny with refusal memory, allow
once with re-prompt, allow session, always with the config-file write), the
explicit denylist veto, hard guards, other denials winning over the question,
invocation-specific once approvals for concurrent identical calls, stale
callback completions, save conflicts, revocation through the real endpoint
with restart survival, session-only grants not surviving new or resumed
sessions, project authority not following a sibling worktree or another
repository, the parent ceiling, and unchanged routine policy.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import time
import tomllib
from typing import Any

import pytest
import tomli_w

from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.models import (
    UserAnswer,
    UserInputCallbackDetail,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ClientInfo,
    ProtocolErrorCode,
)
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._dangerous_command_authority import GRANTS_FIELD
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.dangerous_commands import RepositoryScope
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
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import build_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry

pytestmark = pytest.mark.asyncio

DECLINED_FEEDBACK = "user declined; do not retry this command"
DANGEROUS_DENIAL_FRAGMENT = "requires approval for a dangerous git operation"
DENY_PROJECTION_OPERATIONS = [
    "git.push",
    "git.push.force",
    "git.push.delete",
    "git.push.mirror",
]
TURN_PROMPT = "Exercise dangerous-command grants"


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    return path


def _make_remote(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(path)], check=True)
    return path


def _commit(path: Path) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
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


def _link_remote(project: Path, remote: Path) -> None:
    subprocess.run(
        ["git", "-C", str(project), "remote", "add", "origin", str(remote)], check=True
    )


def _remote_has_main(remote: Path) -> bool:
    return (
        subprocess.run(
            ["git", "--git-dir", str(remote), "rev-parse", "--verify", "-q", "main"],
            check=False,
        ).returncode
        == 0
    )


def _commit_count(project: Path) -> int:
    output = subprocess.run(
        ["git", "-C", str(project), "log", "--oneline"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return len(output.strip().splitlines())


def _call(tool: str, command: str, call_id: str, index: int = 0) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(name=tool, arguments=json.dumps({"command": command})),
    )


def _turn(command: str, call_id: str, tool: str = "bash") -> list[list[LLMChunk]]:
    """One scripted turn: a shell call stream, then a closing stream."""
    return [
        [mock_llm_chunk(tool_calls=[_call(tool, command, call_id)])],
        [mock_llm_chunk(content="done")],
    ]


def _multi_call_turn(calls: list[tuple[str, str, str]]) -> list[list[LLMChunk]]:
    """One turn with several shell calls, then a closing stream."""
    return [
        [
            mock_llm_chunk(
                tool_calls=[
                    _call(tool, command, call_id, index)
                    for index, (call_id, tool, command) in enumerate(calls)
                ]
            )
        ],
        [mock_llm_chunk(content="done")],
    ]


async def _make_loop(
    project: Path, config_path: Path, backend: FakeBackend
) -> AgentLoop:
    """Build a root loop over a real user layer so disk IO reaches the file."""
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
    loop: AgentLoop, *, resume_session_id: str | None = None
) -> tuple[AppServer, AppServerClient, AppServerSession]:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="dangerous-acceptance-test", version="0"),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
        resume_session_id=resume_session_id,
    )
    return server, client, session


def _grant_answer(label: str) -> UserInputCallbackOutput:
    return UserInputCallbackOutput(
        result=UserQuestionResult(answers=[UserAnswer(question="grant", answer=label)])
    )


def _assert_dangerous_question(detail: object) -> None:
    assert isinstance(detail, UserInputCallbackDetail)
    question = detail.request.questions[0]
    assert question.header == "Dangerous command"
    assert "git.push" in question.question
    assert [option.label for option in question.options] == [
        "Deny",
        "Allow once",
        "Allow this session",
        "Always for this project",
    ]


def _forbid_callbacks(_index: int) -> str:
    raise AssertionError("No dangerous-command question was expected")


async def _drive(
    session: AppServerSession, answer: str | Callable[[int], str]
) -> list[str]:
    """Run one turn, answering the i-th dangerous-command callback."""
    callbacks: list[str] = []
    async for event in session.act(TURN_PROMPT):
        if isinstance(event, CallbackRequested):
            callbacks.append(event.callback.callback_id)
            _assert_dangerous_question(event.callback.detail)
            label = answer(len(callbacks) - 1) if callable(answer) else answer
            await session.respond_to_callback(
                event.callback.callback_id, _grant_answer(label)
            )
    return callbacks


async def _drive_loop(
    agent: AgentLoop,
) -> tuple[list[BaseEvent], list[UserInputRequestEvent]]:
    """Run one loop turn directly, answering every dangerous-command prompt."""
    events: list[BaseEvent] = []
    requests: list[UserInputRequestEvent] = []
    async with asyncio.timeout(10):
        async for event in agent.act(TURN_PROMPT):
            if isinstance(event, UserInputRequestEvent):
                requests.append(event)
                agent.resolve_user_input_request(
                    event.request_id,
                    QuestionResult(
                        answers=[QuestionAnswer(question="grant", answer="Deny")]
                    ),
                )
            events.append(event)
    return events, requests


def _tool_pairs(loop: AgentLoop) -> list[tuple[str, str]]:
    return [
        (message.tool_call_id or "", message.content or "")
        for message in loop.messages
        if message.role == Role.tool and message.content
    ]


def _tool_texts(loop: AgentLoop) -> list[str]:
    return [text for _call_id, text in _tool_pairs(loop)]


def _results(events: list[BaseEvent]) -> list[ToolResultEvent]:
    return [event for event in events if isinstance(event, ToolResultEvent)]


def _saved_grants(config_path: Path) -> dict[str, list[str]]:
    if not config_path.exists():
        return {}
    data: dict[str, Any] = tomllib.loads(config_path.read_text(encoding="utf-8"))
    grants = data.get(GRANTS_FIELD, {})
    return {str(key): [str(item) for item in value] for key, value in grants.items()}


def _authorities(loop: AgentLoop) -> list[tuple[frozenset[str], RepositoryScope]]:
    return [
        (authority.operations, authority.scope)
        for authority in loop._dangerous_command_grants.dangerous_command_authorities()
    ]


def _scope_of(path: Path) -> RepositoryScope:
    scope = resolve_repository_scope(path, GitRepositoryRelocation())
    assert scope is not None
    return scope


async def _wait_for_marker(marker: Path, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists():
            return True
        await asyncio.sleep(0.05)
    return marker.exists()


async def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for the pending callback")


@pytest.mark.parametrize("tool", ["bash", "bash_start"])
async def test_deny_refuses_execution_and_remembers_the_refusal(
    tmp_path: Path, config_dir: Path, tool: str
) -> None:
    """Deny: no execution, refusal feedback, and no re-prompt for that call."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "deny-marker"
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    command = f"git push origin main; touch {marker.name}"
    backend = FakeBackend([
        *_turn(command, "deny-1", tool),
        *_turn(command, "deny-2", tool),
    ])
    loop = await _make_loop(project, config_path, backend)
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, "Deny")
        assert len(callbacks) == 1
        if tool == "bash":
            assert not marker.exists()
        else:
            assert not await _wait_for_marker(marker, timeout=1.0)
        assert not _remote_has_main(remote)
        assert any(DECLINED_FEEDBACK in text for text in _tool_texts(loop))
        assert _authorities(loop) == []
        # Refusal memory: the same call never re-prompts and never executes.
        callbacks = await _drive(session, _forbid_callbacks)
        assert callbacks == []
        assert not marker.exists()
        assert not _remote_has_main(remote)
        declined = [text for text in _tool_texts(loop) if DECLINED_FEEDBACK in text]
        assert len(declined) == 2
        assert _authorities(loop) == []
        assert config_path.read_bytes() == disk
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.parametrize("tool", ["bash", "bash_start"])
async def test_allow_once_executes_exactly_one_call_and_reprompts(
    tmp_path: Path, config_dir: Path, tool: str
) -> None:
    """Allow once: exactly that call executes; the same command re-prompts."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "once-marker"
    config_path = config_dir / "config.toml"
    command = f"git push origin main; touch {marker.name}"
    backend = FakeBackend([
        *_turn(command, "once-1", tool),
        *_turn(command, "once-2", tool),
    ])
    loop = await _make_loop(project, config_path, backend)
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, "Allow once")
        assert len(callbacks) == 1
        if tool == "bash":
            assert marker.exists()
        else:
            assert await _wait_for_marker(marker)
        assert _remote_has_main(remote)
        # The once permit authorized exactly this invocation: no authority
        # remains, so the identical call prompts again instead of resuming.
        assert _authorities(loop) == []
        marker.unlink()
        callbacks = await _drive(session, "Deny")
        assert len(callbacks) == 1
        assert not marker.exists()
        assert any(DECLINED_FEEDBACK in text for text in _tool_texts(loop))
        assert _authorities(loop) == []
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.parametrize("tool", ["bash", "bash_start"])
async def test_allow_this_session_executes_subsequent_calls_without_prompting(
    tmp_path: Path, config_dir: Path, tool: str
) -> None:
    """Allow this session: later calls of the operation+scope never prompt."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "session-marker"
    config_path = config_dir / "config.toml"
    command = f"git push origin main; touch {marker.name}"
    backend = FakeBackend([
        *_turn(command, "session-1", tool),
        *_turn(command, "session-2", tool),
    ])
    loop = await _make_loop(project, config_path, backend)
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, "Allow this session")
        assert len(callbacks) == 1
        if tool == "bash":
            assert marker.exists()
        else:
            assert await _wait_for_marker(marker)
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(project))]
        # The second identical call resumes without prompting.
        marker.unlink()
        callbacks = await _drive(session, _forbid_callbacks)
        assert callbacks == []
        if tool == "bash":
            assert marker.exists()
        else:
            assert await _wait_for_marker(marker)
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(project))]
        # A session grant is never persisted.
        assert _saved_grants(config_path) == {}
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.parametrize("tool", ["bash", "bash_start"])
async def test_always_for_project_executes_and_persists_to_the_user_config(
    tmp_path: Path, config_dir: Path, tool: str
) -> None:
    """Always for this project: the call executes and the config file gains it."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "always-marker"
    config_path = config_dir / "config.toml"
    command = f"git push origin main; touch {marker.name}"
    backend = FakeBackend([
        *_turn(command, "always-1", tool),
        *_turn(command, "always-2", tool),
    ])
    loop = await _make_loop(project, config_path, backend)
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, "Always for this project")
        assert len(callbacks) == 1
        if tool == "bash":
            assert marker.exists()
        else:
            assert await _wait_for_marker(marker)
        assert _remote_has_main(remote)
        note = next(
            text for text in _tool_texts(loop) if "Dangerous-command grant" in text
        )
        assert "saved in your user config" in note
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(project))]
        # The user-layer config file gains the project entry.
        assert _saved_grants(config_path) == {str(project.resolve()): ["git.push"]}
        # The saved grant covers later calls in the same session: no prompt.
        marker.unlink()
        callbacks = await _drive(session, _forbid_callbacks)
        assert callbacks == []
        if tool == "bash":
            assert marker.exists()
        else:
            assert await _wait_for_marker(marker)
    finally:
        await session.close()
        await loop.aclose()


async def test_explicit_denylist_vetoes_even_with_a_saved_grant(
    tmp_path: Path, config_dir: Path
) -> None:
    """An explicit denylist entry vetoes before any question or grant."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "veto-marker"
    config_path = config_dir / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n'
        + '\n[tools.bash]\ndenylist = ["git push"]\n',
        encoding="utf-8",
    )
    command = f"git push origin main; touch {marker.name}"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "veto-1")])
    )
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, _forbid_callbacks)
        assert callbacks == []
        assert not marker.exists()
        assert not _remote_has_main(remote)
        texts = _tool_texts(loop)
        assert any("matches denylist pattern 'git push'" in text for text in texts)
        assert not any(DANGEROUS_DENIAL_FRAGMENT in text for text in texts)
        # The saved grant exists and is accepted, but the denylist vetoes it.
        read = await session.resources.config.read_dangerous_command_authority()
        assert [grant.operation_id for grant in read.accepted_saved_grants] == [
            "git.push"
        ]
        assert [
            (block.pattern, block.operations) for block in read.blocking_denylist
        ] == [("git push", DENY_PROJECTION_OPERATIONS)]
        # The resolver denial carries no dangerous-command proposal at all.
        tool = loop.tool_manager.get("bash")
        args = tool.validate_arguments({"command": command})
        resolved = tool.resolve_permission(args)
        assert resolved is not None
        assert resolved.denial_kind is None
        assert resolved.dangerous_command_proposal is None
        assert _authorities(loop) == []
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.parametrize("guard", ["git reset --hard", "git clean -fd"])
async def test_hard_guards_stay_denied_without_a_question(
    tmp_path: Path, config_dir: Path, guard: str
) -> None:
    """Hard guards: no question, no grant path, even under a session grant."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "guard-marker"
    config_path = config_dir / "config.toml"
    command = f"{guard}; touch {marker.name}"
    backend = FakeBackend([
        *_turn("git push origin main", "guard-push"),
        *_turn(command, "guard-call"),
    ])
    loop = await _make_loop(project, config_path, backend)
    _, _, session = await _open_session(loop)
    try:
        # A session grant for publication exists before the hard guard runs.
        callbacks = await _drive(session, "Allow this session")
        assert len(callbacks) == 1
        assert _remote_has_main(remote)
        callbacks = await _drive(session, _forbid_callbacks)
        assert callbacks == []
        assert not marker.exists()
        texts = _tool_texts(loop)
        guard_texts = [text for text in texts if "Command denied" in text]
        assert guard_texts
        assert any(guard in text for text in guard_texts)
        assert not any(DANGEROUS_DENIAL_FRAGMENT in text for text in guard_texts)
        # The session grant cannot lift the hard guard.
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(project))]
        assert _commit_count(project) == 1
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.parametrize(
    ("command", "expected_denial"),
    [
        (
            "git push origin main; sudo true",
            "Command denied by a sensitive command rule",
        ),
        (
            "git push origin main; cat /etc/hostname",
            "Shell path is outside the authorized workspace",
        ),
    ],
)
async def test_other_denial_wins_over_the_dangerous_question(
    tmp_path: Path, config_dir: Path, command: str, expected_denial: str
) -> None:
    """A separately-denied segment returns the other denial, never a question."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    config_path = config_dir / "config.toml"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "compound-1")])
    )
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, _forbid_callbacks)
        assert callbacks == []
        assert not _remote_has_main(remote)
        texts = _tool_texts(loop)
        assert any(expected_denial in text for text in texts)
        assert not any(DANGEROUS_DENIAL_FRAGMENT in text for text in texts)
        assert _authorities(loop) == []
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.parametrize("tool", ["bash", "bash_start"])
async def test_two_concurrent_identical_once_calls_need_two_approvals(
    tmp_path: Path, config_dir: Path, tool: str
) -> None:
    """Two concurrent identical calls: one approval cannot satisfy both."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "twin-marker"
    config_path = config_dir / "config.toml"
    command = f"git push origin main; touch {marker.name}"
    backend = FakeBackend(
        _multi_call_turn([("twin-a", tool, command), ("twin-b", tool, command)])
    )
    loop = await _make_loop(project, config_path, backend)
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(
            session, lambda index: "Allow once" if index == 0 else "Deny"
        )
        assert len(callbacks) == 2
        assert callbacks[0] != callbacks[1]
        if tool == "bash":
            assert marker.exists()
        else:
            assert await _wait_for_marker(marker)
        assert _remote_has_main(remote)
        # Exactly one invocation executed: the other was refused, so a single
        # approval satisfied exactly one call.
        pairs = dict(_tool_pairs(loop))
        assert set(pairs) == {"twin-a", "twin-b"}
        declined = [
            call_id for call_id, text in pairs.items() if DECLINED_FEEDBACK in text
        ]
        assert len(declined) == 1
        other = next(call_id for call_id in pairs if call_id not in declined)
        assert "Command denied" not in pairs[other]
        # No authority outlives either invocation.
        assert _authorities(loop) == []
    finally:
        await session.close()
        await loop.aclose()


async def test_interrupted_turn_stale_callback_completion_grants_nothing(
    tmp_path: Path, config_dir: Path
) -> None:
    """A completion arriving after turn termination is rejected and grants nothing."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "stale-marker"
    config_path = config_dir / "config.toml"
    command = f"git push origin main; touch {marker.name}"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "stale-1")])
    )
    _, _, session = await _open_session(loop)
    try:
        callback_id: str | None = None
        async for event in session.act(TURN_PROMPT):
            if isinstance(event, CallbackRequested):
                callback_id = event.callback.callback_id
                await session.interrupt()
        assert callback_id is not None
        with pytest.raises(AppServerResponseError) as excinfo:
            await session.respond_to_callback(
                callback_id, _grant_answer("Allow this session")
            )
        assert excinfo.value.error.code is ProtocolErrorCode.CALLBACK_CLOSED
        assert _authorities(loop) == []
        assert not marker.exists()
        assert not _remote_has_main(remote)
    finally:
        await session.close()
        await loop.aclose()


async def test_closed_session_stale_callback_completion_grants_nothing(
    tmp_path: Path, config_dir: Path
) -> None:
    """A completion arriving after session termination cannot be delivered."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "closed-marker"
    config_path = config_dir / "config.toml"
    command = f"git push origin main; touch {marker.name}"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "closed-1")])
    )
    _, _, session = await _open_session(loop)
    collected: list[str] = []

    async def consume() -> None:
        async for event in session.act(TURN_PROMPT):
            if isinstance(event, CallbackRequested):
                collected.append(event.callback.callback_id)

    task = asyncio.create_task(consume())
    closed = False
    try:
        await _wait_until(lambda: len(collected) == 1)
        callback_id = collected[0]
        await session.close()
        closed = True
        with pytest.raises(RuntimeError, match="connection is closed"):
            await session.respond_to_callback(
                callback_id, _grant_answer("Allow this session")
            )
        assert _authorities(loop) == []
        assert not marker.exists()
        assert not _remote_has_main(remote)
    finally:
        if not closed:
            await session.close()
        await asyncio.gather(task, return_exceptions=True)
        await loop.aclose()


async def test_a_consumed_callback_cannot_grant_a_second_call(
    tmp_path: Path, config_dir: Path
) -> None:
    """A settled callback rejects a different late answer and grants nothing."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "consumed-marker"
    config_path = config_dir / "config.toml"
    command = f"git push origin main; touch {marker.name}"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "consumed-1")])
    )
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, "Allow once")
        assert len(callbacks) == 1
        assert marker.exists()
        assert _authorities(loop) == []
        # An identical duplicate completion is a no-op, never a second grant.
        await session.respond_to_callback(callbacks[0], _grant_answer("Allow once"))
        assert _authorities(loop) == []
        # A different late answer conflicts and installs nothing.
        with pytest.raises(AppServerResponseError) as excinfo:
            await session.respond_to_callback(
                callbacks[0], _grant_answer("Allow this session")
            )
        assert excinfo.value.error.code is ProtocolErrorCode.CONFLICT
        assert _authorities(loop) == []
        assert _remote_has_main(remote)
    finally:
        await session.close()
        await loop.aclose()


async def test_save_conflict_keeps_the_live_grant_and_the_intervening_writer(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conflicting disk write: the call runs, persistence reports failure."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    other = _make_repo(tmp_path / "other")
    marker = project / "conflict-marker"
    config_path = config_dir / "config.toml"
    command = f"git push origin main; touch {marker.name}"
    loop = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "conflict-1")])
    )
    _, _, session = await _open_session(loop)
    original_save = loop.config_orchestrator.save_dangerous_command_grant

    async def intervening_then_save(
        *, project: Path, operation_id: str, expected_revision: str
    ) -> Any:
        # A concurrent writer lands a grant for another project on disk
        # between the loop's revision read and the checked save.
        config_path.write_text(
            config_path.read_text(encoding="utf-8")
            + f'\n[{GRANTS_FIELD}]\n"{other.resolve()}" = ["git.push"]\n',
            encoding="utf-8",
        )
        return await original_save(
            project=project,
            operation_id=operation_id,
            expected_revision=expected_revision,
        )

    monkeypatch.setattr(
        loop.config_orchestrator, "save_dangerous_command_grant", intervening_then_save
    )
    try:
        callbacks = await _drive(session, "Always for this project")
        assert len(callbacks) == 1
        # The call executed under the live approval.
        assert marker.exists()
        assert _remote_has_main(remote)
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(project))]
        # Persistence reports the conflict apart from the live grant.
        note = next(
            text for text in _tool_texts(loop) if "Dangerous-command grant" in text
        )
        assert "applies to this session only" in note
        assert "rejected because the config changed since it was read" in note
        # The config file keeps the intervening writer's content; the
        # approved grant never landed on disk.
        assert _saved_grants(config_path) == {str(other.resolve()): ["git.push"]}
    finally:
        await session.close()
        await loop.aclose()


async def test_revocation_removes_live_authority_immediately_and_survives_restart(
    tmp_path: Path, config_dir: Path
) -> None:
    """Revocation through the real endpoint: live authority ends at once."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "revoke-marker"
    config_path = config_dir / "config.toml"
    command = f"git push origin main; touch {marker.name}"
    backend = FakeBackend([*_turn(command, "revoke-1"), *_turn(command, "revoke-2")])
    loop = await _make_loop(project, config_path, backend)
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, "Always for this project")
        assert len(callbacks) == 1
        assert marker.exists()
        assert _saved_grants(config_path) == {str(project.resolve()): ["git.push"]}
        read = await session.resources.config.read_dangerous_command_authority()
        assert [grant.operation_id for grant in read.live_grants] == ["git.push"]
        assert read.saved_operations == ["git.push"]
        # Revocation through the real endpoint removes live and saved
        # authority with separate outcomes.
        revoked = await session.resources.config.revoke_dangerous_command_grant(
            operation_id="git.push",
            expected_live_generation=read.live_authority_generation,
            expected_disk_revision=read.disk_revision,
            user_initiated=True,
        )
        assert revoked.live.status == "removed"
        assert revoked.saved.status == "removed"
        assert revoked.warning is None
        assert _authorities(loop) == []
        assert _saved_grants(config_path) == {}
        # The next call of the operation re-prompts: live authority is gone.
        marker.unlink()
        callbacks = await _drive(session, "Deny")
        assert len(callbacks) == 1
        assert not marker.exists()
        assert any(DECLINED_FEEDBACK in text for text in _tool_texts(loop))
    finally:
        await session.close()
        await loop.aclose()

    # A restart (a fresh session over the same config) reads no grant from
    # disk: the disk removal survived.
    restarted = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "restart-1")])
    )
    _, _, restarted_session = await _open_session(restarted)
    try:
        read = (
            await restarted_session.resources.config.read_dangerous_command_authority()
        )
        assert read.live_grants == []
        assert read.saved_operations == []
        assert read.accepted_saved_grants == []
        callbacks = await _drive(restarted_session, "Deny")
        assert len(callbacks) == 1
        assert not marker.exists()
    finally:
        await restarted_session.close()
        await restarted.aclose()


async def test_session_only_grants_do_not_survive_new_or_resumed_sessions(
    tmp_path: Path, config_dir: Path
) -> None:
    """Session-only grants stay runtime state: fresh sessions start empty."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "fresh-marker"
    config_path = config_dir / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config["session_logging"] = {
        "enabled": True,
        "save_dir": str(tmp_path / "sessions"),
    }
    config_path.write_bytes(tomli_w.dumps(config).encode())
    command = f"git push origin main; touch {marker.name}"
    backend = FakeBackend([*_turn(command, "fresh-1"), *_turn(command, "fresh-2")])
    loop = await _make_loop(project, config_path, backend)
    _, _, session = await _open_session(loop)
    saved_session_id = loop.session_id
    try:
        callbacks = await _drive(session, "Allow this session")
        assert len(callbacks) == 1
        assert marker.exists()
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(project))]
        # The session grant covers the second call without prompting.
        marker.unlink()
        callbacks = await _drive(session, _forbid_callbacks)
        assert callbacks == []
        assert marker.exists()
        # A session-only grant never reaches the config file.
        assert _saved_grants(config_path) == {}
        await loop.persist_empty_session()
    finally:
        await session.close()
        await loop.aclose()

    # A brand-new session over the same config starts with no runtime
    # authority: the same call prompts again.
    fresh = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "fresh-3")])
    )
    _, _, fresh_session = await _open_session(fresh)
    try:
        read = await fresh_session.resources.config.read_dangerous_command_authority()
        assert read.live_grants == []
        marker.unlink()
        callbacks = await _drive(fresh_session, "Deny")
        assert len(callbacks) == 1
        assert not marker.exists()
    finally:
        await fresh_session.close()
        await fresh.aclose()

    # A resumed session (the persisted session id, a new runtime) also
    # starts with no runtime authority restored.
    resumed = await _make_loop(
        project, config_path, FakeBackend([*_turn(command, "resumed-1")])
    )
    _, _, resumed_session = await _open_session(
        resumed, resume_session_id=saved_session_id
    )
    try:
        read = await resumed_session.resources.config.read_dangerous_command_authority()
        assert read.live_grants == []
        callbacks = await _drive(resumed_session, "Deny")
        assert len(callbacks) == 1
        assert not marker.exists()
        assert _authorities(resumed) == []
    finally:
        await resumed_session.close()
        await resumed.aclose()


async def test_project_authority_does_not_cover_a_sibling_worktree_or_other_repo(
    tmp_path: Path, config_dir: Path
) -> None:
    """A saved project grant never covers another repository identity."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    worktree = project / "wt"
    subprocess.run(
        ["git", "-C", str(project), "worktree", "add", "-q", str(worktree), "-b", "wt"],
        check=True,
    )
    nested = _make_repo(project / "nested")
    _link_remote(nested, remote)
    main_marker = project / "main-marker"
    wt_marker = project / "wt-marker"
    nested_marker = project / "nested-marker"
    config_path = config_dir / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n',
        encoding="utf-8",
    )
    backend = FakeBackend([
        *_turn(f"git push origin main; touch {main_marker.name}", "main-1"),
        *_turn(f"git -C wt push origin main; touch {wt_marker.name}", "wt-1"),
        *_turn(
            f"git -C nested push origin main; touch {nested_marker.name}", "nested-1"
        ),
    ])
    loop = await _make_loop(project, config_path, backend)
    _, _, session = await _open_session(loop)
    try:
        # The saved project grant covers the session's own repository.
        callbacks = await _drive(session, _forbid_callbacks)
        assert callbacks == []
        assert main_marker.exists()
        assert _remote_has_main(remote)
        # A sibling worktree of the same repository is a different identity.
        callbacks = await _drive(session, "Deny")
        assert len(callbacks) == 1
        assert not wt_marker.exists()
        # Another repository inside the workspace is a different identity.
        callbacks = await _drive(session, "Deny")
        assert len(callbacks) == 1
        assert not nested_marker.exists()
        assert _authorities(loop) == []
        # The denials persisted nothing.
        assert _saved_grants(config_path) == {str(project.resolve()): ["git.push"]}
        assert _scope_of(worktree) != _scope_of(project)
        assert _scope_of(nested) != _scope_of(project)
    finally:
        await session.close()
        await loop.aclose()


async def test_child_approval_cannot_lift_a_parent_dangerous_ceiling(
    tmp_path: Path, config_dir: Path
) -> None:
    """A parent's dangerous gap is a ceiling: no child grant or prompt lifts it."""
    project, remote = (
        _make_repo(tmp_path / "project"),
        _make_remote(tmp_path / "remote.git"),
    )
    _link_remote(project, remote)
    _commit(project)
    marker = project / "ceiling-marker"
    config_path = config_dir / "config.toml"
    command = f"git push origin main; touch {marker.name}"
    parent = await _make_loop(project, config_path, FakeBackend())
    server, _, session = await _open_session(parent)
    try:
        root = server._require_root()
        assert isinstance(root, SessionBackendImpl)
        registry = root.children
        policy = parent.child_runtime_policy
        child = AgentLoop(
            config_orchestrator=parent.config_orchestrator.copy(),
            cwd=project,
            backend=FakeBackend([
                *_turn(command, "child-1"),
                *_turn(command, "child-2"),
            ]),
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
            # No question reaches any client: the parent's gap is a ceiling.
            events, requests = await _drive_loop(child)
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert DANGEROUS_DENIAL_FRAGMENT in (results[0].skip_reason or "")
            assert not marker.exists()
            assert not _remote_has_main(remote)
            assert _authorities(child) == []
            assert _authorities(parent) == []
            # The ceiling classification is exact: parent_ceiling, no proposal.
            tool = child.tool_manager.get("bash")
            args = tool.validate_arguments({"command": command})
            resolved = tool.resolve_permission(args)
            assert resolved is not None
            assert resolved.denial_kind == "parent_ceiling"
            assert resolved.dangerous_command_proposal is None
            # Even a child-side session grant cannot lift the ceiling.
            child._dangerous_command_grants.grant_session({
                ("git.push", _scope_of(project))
            })
            child.tool_manager._invalidate_dangerous_command_authority()
            events, requests = await _drive_loop(child)
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert DANGEROUS_DENIAL_FRAGMENT in (results[0].skip_reason or "")
            assert not marker.exists()
            assert _authorities(child) == [
                (frozenset({"git.push"}), _scope_of(project))
            ]
            assert _authorities(parent) == []
        finally:
            await child.aclose()
    finally:
        await session.close()
        await parent.aclose()


async def test_routine_git_commands_execute_without_prompts(
    tmp_path: Path, config_dir: Path
) -> None:
    """Ordinary git commands keep the existing auto-execution policy."""
    project = _make_repo(tmp_path / "project")
    seed = project / "seed.txt"
    seed.write_text("seed\n", encoding="utf-8")
    _commit(project)
    seed.write_text("seed v2\n", encoding="utf-8")
    config_path = config_dir / "config.toml"
    backend = FakeBackend(
        _multi_call_turn([
            ("routine-status", "bash", "git status"),
            ("routine-log", "bash", "git log --oneline"),
            ("routine-add", "bash", "git add seed.txt"),
            ("routine-commit", "bash", "git commit -m routine"),
        ])
    )
    loop = await _make_loop(project, config_path, backend)
    _, _, session = await _open_session(loop)
    try:
        callbacks = await _drive(session, _forbid_callbacks)
        assert callbacks == []
        pairs = dict(_tool_pairs(loop))
        assert set(pairs) == {
            "routine-status",
            "routine-log",
            "routine-add",
            "routine-commit",
        }
        assert "On branch main" in pairs["routine-status"]
        assert _commit_count(project) == 2
        assert _authorities(loop) == []
    finally:
        await session.close()
        await loop.aclose()
