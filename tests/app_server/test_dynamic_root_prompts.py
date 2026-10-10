"""End-to-end mid-turn root prompts through the app-server callback path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from chartreux.app_server._runtime import SessionRootGrantPort, create_harness_server
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.host import AppServerHost
from chartreux.app_server.models import (
    UserAnswer,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import ClientCapabilities, ClientInfo, SessionOptions
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.llm_models import Role
from chartreux.utils import AgentEntrypoint
from tests.backend.data.mistral import mistral_completion
from tests.constants import (
    CHAT_COMPLETIONS_PATH,
    CONNECTORS_BOOTSTRAP_PATH,
    MISTRAL_BASE_URL,
)

pytestmark = pytest.mark.asyncio


def _sse(
    content: str, tool_calls: list[dict[str, Any]] | None = None
) -> httpx.Response:
    response = mistral_completion(content, tool_calls=tool_calls)
    response["model"] = "mistral-vibe-cli-latest"
    response["object"] = "chat.completion.chunk"
    choice = response["choices"][0]
    message = choice.pop("message")
    if tool_calls is None:
        message.pop("tool_calls")
    choice["delta"] = message
    payload = b"data: " + json.dumps(response).encode() + b"\n\ndata: [DONE]"
    return httpx.Response(
        200,
        stream=httpx.ByteStream(stream=payload),
        headers={"Content-Type": "text/event-stream"},
    )


def _tool_call(
    call_id: str, name: str, arguments: dict[str, str], index: int = 0
) -> dict[str, Any]:
    return {
        "id": call_id,
        "index": index,
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


@pytest.fixture
def mistral_api(respx_mock: respx.MockRouter) -> respx.Route:
    respx_mock.get(f"{MISTRAL_BASE_URL}{CONNECTORS_BOOTSTRAP_PATH}").mock(
        return_value=httpx.Response(200, json={"connectors": []})
    )
    return respx_mock.post(f"{MISTRAL_BASE_URL}{CHAT_COMPLETIONS_PATH}")


async def _connect(entrypoint: AgentEntrypoint, project: Path) -> AppServerHost:
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(server_transport, transport_kind="in_process")
    client = AppServerClient(client_transport, run_peer=harness.serve)
    return await AppServerHost.connect(
        client,
        client_info=ClientInfo(
            name="root-grant-test", version="0", entrypoint=entrypoint
        ),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
        session_options=SessionOptions(cwd=str(project)),
        client_factory=harness.connect_client,
    )


async def _drive(session: AppServerSession, answer: str | None = None) -> list[str]:
    """Run one turn, answering every root-grant callback with *answer*."""
    callbacks: list[str] = []
    async for event in session.act("read the outside file"):
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


def _root_loop(registry: SessionRuntimeRegistry) -> AgentLoop:
    root = registry._root
    assert root is not None
    return root.agent_loop


def _tool_messages(loop: AgentLoop) -> list[str]:
    return [
        message.content or "" for message in loop.messages if message.role == Role.tool
    ]


async def test_approved_grant_applies_to_the_calling_session(
    tmp_path: Path,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    mistral_api: respx.Route,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    registries: list[SessionRuntimeRegistry] = []
    bind = SessionRuntimeRegistry.bind_root

    def capture(registry: SessionRuntimeRegistry, runtime: Any) -> None:
        bind(registry, runtime)
        registries.append(registry)

    monkeypatch.setattr(SessionRuntimeRegistry, "bind_root", capture)
    mistral_api.mock(
        side_effect=[
            _sse(
                "",
                tool_calls=[
                    _tool_call("read-1", "read_file", {"file_path": str(target)})
                ],
            ),
            _sse("done"),
        ]
    )
    host = await _connect("cli", project)
    session = await host.open_session()
    try:
        callbacks = await _drive(session, "Allow this session")
        assert len(callbacks) == 1
        assert len(registries) == 1
        loop = _root_loop(registries[0])
        assert isinstance(loop._root_grant_port, SessionRootGrantPort)
        assert loop._session_root_grants.roots == {outside.resolve()}
        assert loop.tool_manager.workspace.allows(target)
        assert any("granted content" in text for text in _tool_messages(loop))
        assert config_path.read_bytes() == disk
    finally:
        await session.close()
        await host.close()


async def test_denied_grant_skips_the_tool_call(
    tmp_path: Path,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    mistral_api: respx.Route,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    registries: list[SessionRuntimeRegistry] = []
    bind = SessionRuntimeRegistry.bind_root

    def capture(registry: SessionRuntimeRegistry, runtime: Any) -> None:
        bind(registry, runtime)
        registries.append(registry)

    monkeypatch.setattr(SessionRuntimeRegistry, "bind_root", capture)
    mistral_api.mock(
        side_effect=[
            _sse(
                "",
                tool_calls=[
                    _tool_call("read-1", "read_file", {"file_path": str(target)})
                ],
            ),
            _sse("done"),
        ]
    )
    host = await _connect("cli", project)
    session = await host.open_session()
    try:
        callbacks = await _drive(session, "Deny")
        assert len(callbacks) == 1
        loop = _root_loop(registries[0])
        assert loop._session_root_grants.roots == set()
        assert loop._session_root_grants.denied_roots == {outside.resolve()}
        assert any(
            "user declined; do not retry this path" in text
            for text in _tool_messages(loop)
        )
        assert config_path.read_bytes() == disk
    finally:
        await session.close()
        await host.close()


async def test_concurrent_out_of_root_calls_produce_one_callback(
    tmp_path: Path,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    mistral_api: respx.Route,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    first = outside / "first.txt"
    first.write_text("first granted content")
    second = outside / "second.txt"
    second.write_text("second granted content")
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    registries: list[SessionRuntimeRegistry] = []
    bind = SessionRuntimeRegistry.bind_root

    def capture(registry: SessionRuntimeRegistry, runtime: Any) -> None:
        bind(registry, runtime)
        registries.append(registry)

    monkeypatch.setattr(SessionRuntimeRegistry, "bind_root", capture)
    mistral_api.mock(
        side_effect=[
            _sse(
                "",
                tool_calls=[
                    _tool_call("read-1", "read_file", {"file_path": str(first)}, 0),
                    _tool_call("read-2", "read_file", {"file_path": str(second)}, 1),
                ],
            ),
            _sse("done"),
        ]
    )
    host = await _connect("cli", project)
    session = await host.open_session()
    try:
        callbacks = await _drive(session, "Allow this session")
        assert len(callbacks) == 1
        loop = _root_loop(registries[0])
        assert loop._session_root_grants.roots == {outside.resolve()}
        texts = _tool_messages(loop)
        assert any("first granted content" in text for text in texts)
        assert any("second granted content" in text for text in texts)
        assert config_path.read_bytes() == disk
    finally:
        await session.close()
        await host.close()


@pytest.mark.parametrize("entrypoint", ["programmatic", "acp"])
async def test_session_without_user_input_capability_denies_without_prompting(
    tmp_path: Path,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    mistral_api: respx.Route,
    entrypoint: AgentEntrypoint,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    registries: list[SessionRuntimeRegistry] = []
    bind = SessionRuntimeRegistry.bind_root

    def capture(registry: SessionRuntimeRegistry, runtime: Any) -> None:
        bind(registry, runtime)
        registries.append(registry)

    monkeypatch.setattr(SessionRuntimeRegistry, "bind_root", capture)
    mistral_api.mock(
        side_effect=[
            _sse(
                "",
                tool_calls=[
                    _tool_call("read-1", "read_file", {"file_path": str(target)})
                ],
            ),
            _sse("done"),
        ]
    )
    host = await _connect(entrypoint, project)
    session = await host.open_session()
    try:
        callbacks = await _drive(session, "Allow this session")
        assert callbacks == []
        loop = _root_loop(registries[0])
        assert loop._session_root_grants.roots == set()
        texts = _tool_messages(loop)
        assert any(
            "authorized_roots_by_project" in text and "config.toml" in text
            for text in texts
        )
        assert config_path.read_bytes() == disk
    finally:
        await session.close()
        await host.close()
