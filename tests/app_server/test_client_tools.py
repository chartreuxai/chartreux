from __future__ import annotations

import asyncio
import json
from pathlib import Path
import socket
import subprocess
from typing import cast
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._connection_protocol import ClientToolCapability
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
from chartreux.core.dangerous_commands import (
    REMOTE_EXEC_NETWORK_LOCALITY_CAPABILITY,
    REMOTE_EXEC_OPERATION_ID,
    REMOTE_EXEC_SSH_TRUST_CAPABILITY,
    DangerousCommandAuthority,
    canonical_remote_destination,
)
from chartreux.core.events import ToolResultEvent
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.tools.base import (
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    resolve_repository_scope,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.builtins.read_file import ReadFileArgs
from chartreux.core.tools.io_port import ShellCommandRequest
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import collect_result, mock_llm_chunk
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
async def test_non_attesting_client_terminal_executor_refuses_glob_bearing_shell_text() -> (
    None
):
    """A ``terminal`` capability is not an executor attestation.

    The server never learns the client's executor, and the protocol carries no
    execution-bound attestation entry, so expansion-bearing shell text refuses
    before the client receives any terminal request. A literal argv — the safe
    fallback that never re-enters a shell — still runs.
    """
    requester = FakeClientRequester()
    capabilities = ClientCapabilities(client_tools=["terminal"])
    tool_io = ClientToolIO(FakeClientToolBridge(requester, capabilities))
    assert capabilities.client_tools == ["terminal"]
    assert tool_io.supports_terminal

    for command, args in (("cat *.txt", None), ("/bin/bash", ["-c", "ls *.py"])):
        with pytest.raises(RuntimeError, match="needs shell pathname expansion"):
            await tool_io.run_shell(
                ShellCommandRequest(
                    session_id="root",
                    tool_call_id="call-1",
                    command=command,
                    args=args,
                    cwd=Path("/workspace"),
                    timeout=1,
                    max_output_bytes=100,
                )
            )
    assert requester.calls == []

    result = await tool_io.run_shell(
        ShellCommandRequest(
            session_id="root",
            tool_call_id="call-2",
            command="cat",
            args=["a.txt"],
            cwd=Path("/workspace"),
            timeout=1,
            max_output_bytes=100,
        )
    )
    assert result.stdout == "host output"
    terminal_create = cast(
        ClientToolTerminalCreateParams,
        next(
            params
            for method, params in requester.calls
            if method == "clientTool/terminal/create"
        ),
    )
    assert terminal_create.command == "cat"
    assert terminal_create.args == ["a.txt"]


# --- S4b1b: execution-bound remote-exec attestation on the client_tools channel
#
# A ``terminal`` entry alone is not an executor attestation. The two
# attestation entries (network-namespace locality AND the filesystem/SSH
# trust context) together form the execution-bound statement that lets a
# pinned IP-literal ssh command ride into clientTool/terminal/create; any
# client that does not attest both — every ACP client and every older client
# — hard-denies before any terminal request is made.

_S4B1B_COMMAND = "ssh deploy@8.8.8.8 uptime"
_S4B1B_PINNED_COMMAND = (
    "ssh -o UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts"
    " -o HostKeyAlias=deploy@8.8.8.8:22 -o StrictHostKeyChecking=yes"
    " -o BatchMode=yes deploy@8.8.8.8 uptime"
)
_ATTESTING_CLIENT_TOOLS: list[ClientToolCapability] = [
    "terminal",
    REMOTE_EXEC_NETWORK_LOCALITY_CAPABILITY,
    REMOTE_EXEC_SSH_TRUST_CAPABILITY,
]


@pytest.fixture
def s4b1b_interfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic locality: a fixed local-interface enumeration (S4b1a seam)."""

    def fake_getaddrinfo(host, port):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.10.20", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def _s4b1b_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


def _s4b1b_bash_tool(repo: Path, *, granted: bool) -> Bash:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo)
    if granted:
        scope = resolve_repository_scope(repo, GitRepositoryRelocation())
        assert scope is not None
        tool.dangerous_command_authorities_getter = lambda: (
            DangerousCommandAuthority(
                operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
                scope=scope,
                source="runtime_grant",
                destination=canonical_remote_destination("deploy@8.8.8.8:22"),
            ),
        )
    else:
        tool.dangerous_command_authorities_getter = lambda: ()
    return tool


@pytest.mark.asyncio
async def test_execution_bound_attesting_client_receives_the_pinned_ssh_command(
    tmp_path: Path, s4b1b_interfaces
) -> None:
    """Both attestation entries + a covering grant: the pinned command —
    approved address plus fixed host-key identity — is the command the client
    terminal is asked to create.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _s4b1b_bash_tool(repo, granted=True)
    requester = FakeClientRequester()
    tool_io = ClientToolIO(
        FakeClientToolBridge(
            requester, ClientCapabilities(client_tools=_ATTESTING_CLIENT_TOOLS)
        )
    )
    assert tool_io.remote_exec_attestation.execution_bound
    ctx = InvokeContext(tool_call_id="call-1", session_id="root", tool_io=tool_io)

    result = await collect_result(tool.run(BashArgs(command=_S4B1B_COMMAND), ctx))

    assert result.exit_code == 0
    terminal_create = cast(
        ClientToolTerminalCreateParams,
        next(
            params
            for method, params in requester.calls
            if method == "clientTool/terminal/create"
        ),
    )
    assert terminal_create.command == _S4B1B_PINNED_COMMAND


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_tools",
    [
        ["terminal"],  # no attestation entries (every ACP client, older clients)
        # locality only: not an execution-bound statement; hard-denies
        # transfers (characterized at this row, discriminating arm at S4b2c).
        ["terminal", REMOTE_EXEC_NETWORK_LOCALITY_CAPABILITY],
    ],
    ids=["non_attesting", "locality_only"],
)
async def test_non_execution_bound_client_never_receives_the_ssh_command(
    tmp_path: Path, s4b1b_interfaces, client_tools: list[ClientToolCapability]
) -> None:
    """A terminal client whose executor is not attested as execution-bound
    (differing namespace or trust context) hard-denies the pinned ssh call
    before any client request is made — even with a covering grant.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _s4b1b_bash_tool(repo, granted=True)
    requester = FakeClientRequester()
    tool_io = ClientToolIO(
        FakeClientToolBridge(requester, ClientCapabilities(client_tools=client_tools))
    )
    assert not tool_io.remote_exec_attestation.execution_bound
    ctx = InvokeContext(tool_call_id="call-1", session_id="root", tool_io=tool_io)

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashArgs(command=_S4B1B_COMMAND), ctx))

    assert "does not attest an execution-bound remote-exec context" in str(err.value)
    assert requester.calls == []


@pytest.mark.asyncio
async def test_attesting_client_without_a_grant_never_receives_the_ssh_command(
    tmp_path: Path, s4b1b_interfaces
) -> None:
    """Authority must coincide with the pin: an execution-bound client with
    no network.remote-exec grant covering the destination sends no request.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _s4b1b_bash_tool(repo, granted=False)
    requester = FakeClientRequester()
    tool_io = ClientToolIO(
        FakeClientToolBridge(
            requester, ClientCapabilities(client_tools=_ATTESTING_CLIENT_TOOLS)
        )
    )
    ctx = InvokeContext(tool_call_id="call-1", session_id="root", tool_io=tool_io)

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashArgs(command=_S4B1B_COMMAND), ctx))

    assert "no network.remote-exec grant covers the destination" in str(err.value)
    assert requester.calls == []


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
        child = await AgentRuntimeFactory().create_child(agent_loop, "coder")
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
