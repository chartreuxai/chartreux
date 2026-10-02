from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._tool_io import ClientToolIO
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    ClientCapabilities,
    ClientInfo,
    ClientToolReadTextFileParams,
    ClientToolReadTextFileResponse,
    ClientToolTerminalCreateParams,
    ClientToolTerminalCreateResponse,
    ClientToolTerminalOutputResponse,
    ClientToolTerminalWaitResponse,
    EmptyResponse,
    JsonRpcSuccessResponse,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop._loop import ToolExecutionResponse
from chartreux.core.events import ToolResultEvent
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins.bash import BashArgs, BashToolConfig
from chartreux.core.tools.builtins.read_file import ReadFileArgs
from chartreux.core.tools.io_port import ShellCommandRequest
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import build_test_app_server, legacy_backend
from tests.stubs.fake_backend import FakeBackend


class FakeClientRequester:
    def __init__(
        self, terminal_exit: ClientToolTerminalWaitResponse | None = None
    ) -> None:
        self.calls: list[tuple[str, ProtocolModel]] = []
        self.terminal_exit = terminal_exit or ClientToolTerminalWaitResponse(
            exit_code=0
        )

    async def __call__[ResultT: ProtocolModel](
        self, method: str, params: ProtocolModel, response_type: type[ResultT]
    ) -> ResultT:
        self.calls.append((method, params))
        match method:
            case "clientTool/readTextFile":
                request = cast(ClientToolReadTextFileParams, params)
                content = "first\nsecond\nthird\n" if request.limit else "a\r\nb\r\n"
                response: ProtocolModel = ClientToolReadTextFileResponse(
                    content=content
                )
            case "clientTool/terminal/create":
                response = ClientToolTerminalCreateResponse(terminal_id="terminal-1")
            case "clientTool/terminal/wait":
                response = self.terminal_exit
            case "clientTool/terminal/output":
                response = ClientToolTerminalOutputResponse(
                    output="host output", truncated=False
                )
            case _:
                response = EmptyResponse()
        return response_type.model_validate(response.model_dump(mode="json"))


class FakeClientToolBridge:
    def __init__(
        self,
        requester: FakeClientRequester,
        capabilities: ClientCapabilities,
        session_id: str = "root",
    ) -> None:
        self._requester = requester
        self._capabilities = capabilities
        self._session_id = session_id

    def client_capabilities(self) -> ClientCapabilities:
        return self._capabilities

    def current_session_id(self) -> str:
        return self._session_id

    async def request_client_result[ResultT: ProtocolModel](
        self, method: str, params: ProtocolModel, response_type: type[ResultT]
    ) -> ResultT:
        return await self._requester(method, params, response_type)


@pytest.mark.asyncio
async def test_client_tool_io_projects_typed_filesystem_and_terminal_requests() -> None:
    requester = FakeClientRequester()
    capabilities = ClientCapabilities(
        client_tools=["filesystem/read", "filesystem/write", "terminal"]
    )
    tool_io = ClientToolIO(FakeClientToolBridge(requester, capabilities))

    bounded = await tool_io.read_lines(
        Path("/workspace/file.txt"), start_line=1, limit=2, max_bytes=100
    )
    text = await tool_io.read_text(Path("/workspace/file.txt"))
    result = await tool_io.run_shell(
        ShellCommandRequest(
            session_id="root",
            tool_call_id="call-1",
            command="/bin/bash",
            args=["-c", "echo hi"],
            env={"CI": "true"},
            cwd=Path("/workspace"),
            timeout=1,
            max_output_bytes=100,
        )
    )

    assert bounded.lines == ["first", "second"]
    assert bounded.was_truncated
    assert text.text == "a\nb\n"
    assert text.newline == "\r\n"
    assert result.stdout == "host output"
    terminal_create = cast(
        ClientToolTerminalCreateParams,
        next(
            params
            for method, params in requester.calls
            if method == "clientTool/terminal/create"
        ),
    )
    assert terminal_create.command == "/bin/bash"
    assert terminal_create.args == ["-c", "echo hi"]
    assert terminal_create.env == {"CI": "true"}
    methods = [method for method, _ in requester.calls]
    assert methods == [
        "clientTool/readTextFile",
        "clientTool/readTextFile",
        "clientTool/terminal/create",
        "clientTool/terminal/wait",
        "clientTool/terminal/output",
        "clientTool/terminal/release",
    ]


@pytest.mark.asyncio
async def test_client_terminal_signal_is_not_reported_as_success() -> None:
    requester = FakeClientRequester(ClientToolTerminalWaitResponse(signal="SIGTERM"))
    tool_io = ClientToolIO(
        FakeClientToolBridge(requester, ClientCapabilities(client_tools=["terminal"]))
    )

    result = await tool_io.run_shell(
        ShellCommandRequest(
            session_id="root",
            tool_call_id="call-1",
            command="sleep 10",
            cwd=Path("/workspace"),
            timeout=1,
            max_output_bytes=100,
        )
    )

    assert result.returncode == -1
    assert result.stderr == "Process terminated by SIGTERM"


@pytest.mark.asyncio
async def test_server_ignores_late_client_tool_response_after_cancellation() -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    server._connection_attached = True
    request = asyncio.create_task(
        server._request_client_result("clientTool/test", EmptyResponse(), EmptyResponse)
    )
    outgoing = await anext(client_transport.messages())
    request_id = cast(int, outgoing["id"])

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    await server._handle_response(JsonRpcSuccessResponse(id=request_id, result={}))

    assert server._abandoned_client_request_ids == set()
    await server.close()
    await client_transport.close()
    await agent_loop.aclose()


@pytest.mark.asyncio
async def test_child_sessions_share_the_server_owned_tool_io_port() -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="test", version="1"),
        capabilities=ClientCapabilities(),
    )
    try:
        children = legacy_backend(server).children
        child = await AgentRuntimeFactory().create_child(agent_loop, "worker")
        child_runtime = children._build_child_runtime(child)

        assert child_runtime.turns._tool_io is children._tool_io

        await child_runtime.close()
    finally:
        await session.close()
    await client_transport.close()
    await agent_loop.aclose()


@pytest.mark.asyncio
async def test_wp2_npm_outside_path_never_reaches_client_without_any_grant(
    tmp_path: Path,
) -> None:
    """Outside npm operands are denied even without stale configuration."""
    project = tmp_path / "project"
    project.mkdir()
    command = f"npm install {tmp_path / 'outside-package'}"
    call = ToolCall(
        id="npm",
        index=0,
        function=FunctionCall(name="bash", arguments=json.dumps({"command": command})),
    )
    agent = build_test_agent_loop(
        cwd=project,
        backend=FakeBackend([
            [mock_llm_chunk(tool_calls=[call])],
            [mock_llm_chunk(content="done")],
        ]),
    )
    requester = FakeClientRequester()
    tool_io = ClientToolIO(
        FakeClientToolBridge(requester, ClientCapabilities(client_tools=["terminal"]))
    )
    try:
        events = [event async for event in agent.act("exercise npm", tool_io=tool_io)]
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        assert result.skipped and result.result is None
        assert requester.calls == []
        assert agent.stats.tool_calls_rejected == 1
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,shell_command",
    [
        ("bash", "cat {target}"),
        ("read_file", ""),
        ("write_file", ""),
        ("bash", "npm install {target}"),
    ],
)
@pytest.mark.parametrize("pattern", ["*", "npm *"])
async def test_outside_operands_never_reach_client_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    pattern: str,
    shell_command: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("unchanged")
    args = (
        {"command": shell_command.format(target=target)}
        if name == "bash"
        else {
            "file_path": str(target),
            **({"content": "changed"} if name == "write_file" else {}),
        }
    )
    call = ToolCall(
        id="denied",
        index=0,
        function=FunctionCall(name=name, arguments=json.dumps(args)),
    )
    agent = build_test_agent_loop(
        cwd=project,
        config=build_test_vibe_config(
            tools={
                name: {
                    "permission": "always",
                    **(
                        {"allowlist": [pattern, str(tmp_path / "*")]}
                        if name != "bash"
                        else {}
                    ),
                }
            }
        ),
        backend=FakeBackend([
            [mock_llm_chunk(tool_calls=[call])],
            [mock_llm_chunk(content="Denied safely")],
        ]),
    )
    if name == "bash":
        stale = BashToolConfig(permission=ToolPermission.ALWAYS).model_copy(
            update={"allowlist": [pattern]}
        )
        monkeypatch.setattr(agent.tool_manager, "get_tool_config", lambda _: stale)
    requester = FakeClientRequester()
    tool_io = ClientToolIO(
        FakeClientToolBridge(
            requester,
            ClientCapabilities(
                client_tools=["terminal", "filesystem/read", "filesystem/write"]
            ),
        )
    )
    spawn = AsyncMock(
        side_effect=AssertionError("Denied command reached local executor")
    )
    monkeypatch.setattr("chartreux.core.tools.builtins.bash.spawn_shell_command", spawn)
    try:
        if name in {"bash", "read_file"}:
            decision = await agent._should_execute_tool(
                agent.tool_manager.get(name),
                BashArgs(command=args["command"])
                if name == "bash"
                else ReadFileArgs(file_path=str(target)),
            )
            assert decision.approval_type is ToolPermission.NEVER
            assert decision.verdict is ToolExecutionResponse.SKIP
        events = [
            event async for event in agent.act("exercise client tools", tool_io=tool_io)
        ]
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        assert result.skipped and result.result is None
        assert requester.calls == []
        spawn.assert_not_called()
        assert target.read_text() == "unchanged"
    finally:
        await agent.aclose()
