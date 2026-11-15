from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tomllib
from typing import Any, cast

import httpx
import pytest
import pytest_asyncio
import respx
import tomli_w

from chartreux.app_server._runtime import create_harness_server
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.host import AppServerHost
from chartreux.app_server.process_bootstrap import ADVERTISED_CAPABILITIES
from chartreux.app_server.protocol import ClientCapabilities, ClientInfo, SessionOptions
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import (
    IN_PROCESS_TRANSPORT,
    STDIO_TRANSPORT,
    TRANSPORT_ENV_VAR,
    memory_transport_pair,
    resolve_transport_selection,
)
from chartreux.core.agent_loop import AgentLoop
from tests.app_server import executed_receipts
from tests.backend.data.mistral import mistral_completion
from tests.conftest import build_test_agent_loop
from tests.constants import (
    CHAT_COMPLETIONS_PATH,
    CONNECTORS_BOOTSTRAP_PATH,
    MISTRAL_BASE_URL,
)


@pytest.fixture(autouse=True)
def _isolate_recovery_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


@dataclass(frozen=True, slots=True)
class BackendContractConnection:
    client: AppServerClient
    host: AppServerHost


class _GatedSSEStream(httpx.AsyncByteStream):
    def __init__(
        self, payload: bytes, started: asyncio.Event, release: asyncio.Event
    ) -> None:
        self._payload = payload
        self._started = started
        self._release = release

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self._started.set()
        await self._release.wait()
        yield self._payload


async def connect_backend_contract_host(
    *, session_options: SessionOptions, capabilities: ClientCapabilities
) -> BackendContractConnection:
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(server_transport, transport_kind="in_process")
    client = AppServerClient(client_transport, run_peer=harness.serve)
    host = await AppServerHost.connect(
        client,
        client_info=ClientInfo(name="backend-contract-test", version="0"),
        capabilities=capabilities,
        session_options=session_options,
        client_factory=harness.connect_client,
    )
    return BackendContractConnection(client=client, host=host)


async def connect_backend_contract_client(
    *, session_options: SessionOptions, capabilities: ClientCapabilities
) -> AppServerClient:
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(server_transport, transport_kind="in_process")
    client = AppServerClient(client_transport, run_peer=harness.serve)
    try:
        await client.start()
        await client.initialize(
            ClientInfo(name="backend-contract-test", version="0"), capabilities
        )
        await client.notify("initialized")
    except BaseException:
        await client.close()
        raise
    return client


@pytest.fixture
def backend_contract_session_options() -> SessionOptions:
    return SessionOptions()


@pytest.fixture
def backend_contract_capabilities() -> ClientCapabilities:
    return ClientCapabilities()


#: The transports a contract node can be parameterized over. The ids carry
#: the transport name, so ``-k stdio`` selects exactly the stdio legs (the
#: B1 gate's selector for this directory).
BACKEND_CONTRACT_TRANSPORTS = (IN_PROCESS_TRANSPORT, STDIO_TRANSPORT)


@pytest.fixture(params=BACKEND_CONTRACT_TRANSPORTS, ids=BACKEND_CONTRACT_TRANSPORTS)
def backend_contract_transport(request: pytest.FixtureRequest) -> str:
    """The transport one contract node runs on (``in_process`` or ``stdio``)."""
    transport = cast(str, request.param)
    executed_receipts.stamp_transport(transport)
    return transport


@pytest.fixture
def stdio_after_autouse(
    monkeypatch: pytest.MonkeyPatch, backend_contract_transport: str
) -> str:
    """Select stdio AFTER the suite-wide autouse, when the node asks for it.

    ``tests/conftest.py``'s autouse pins ``CHARTREUX_TRANSPORT=in_process``
    for the whole suite; a node running on stdio overrides it here, after
    that autouse ran, and the effective selection is asserted rather than
    trusted from the ambient environment.
    """
    if backend_contract_transport == STDIO_TRANSPORT:
        monkeypatch.setenv(TRANSPORT_ENV_VAR, STDIO_TRANSPORT)
        assert resolve_transport_selection(environ=os.environ).kind == STDIO_TRANSPORT
    return backend_contract_transport


@pytest.fixture
def backend_contract_mistral_api(respx_mock: respx.MockRouter) -> respx.Route:
    respx_mock.get(f"{MISTRAL_BASE_URL}{CONNECTORS_BOOTSTRAP_PATH}").mock(
        return_value=httpx.Response(200, json={"connectors": []})
    )
    return respx_mock.post(f"{MISTRAL_BASE_URL}{CHAT_COMPLETIONS_PATH}")


@pytest.fixture
def backend_contract_mistral_response() -> Callable[..., httpx.Response]:
    def build(
        content: str, *, tool_calls: list[dict[str, Any]] | None = None
    ) -> httpx.Response:
        return httpx.Response(
            200,
            stream=httpx.ByteStream(stream=_mistral_sse_payload(content, tool_calls)),
            headers={"Content-Type": "text/event-stream"},
        )

    return build


@pytest.fixture
def backend_contract_gated_mistral_response() -> Callable[..., httpx.Response]:
    def build(
        content: str, *, started: asyncio.Event, release: asyncio.Event
    ) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_GatedSSEStream(
                _mistral_sse_payload(content, tool_calls=None), started, release
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    return build


@pytest.fixture
def backend_contract_mistral_completion() -> Callable[[str], httpx.Response]:
    def build(content: str) -> httpx.Response:
        response = mistral_completion(content)
        response["model"] = "mistral-vibe-cli-latest"
        return httpx.Response(200, json=response)

    return build


def _mistral_sse_payload(
    content: str, tool_calls: list[dict[str, Any]] | None
) -> bytes:
    response = mistral_completion(content, tool_calls=tool_calls)
    response["model"] = "mistral-vibe-cli-latest"
    response["object"] = "chat.completion.chunk"
    choice = response["choices"][0]
    message = choice.pop("message")
    if tool_calls is None:
        message.pop("tool_calls")
    choice["delta"] = message
    return b"data: " + json.dumps(response).encode() + b"\n\ndata: [DONE]"


@pytest_asyncio.fixture
async def backend_contract_host(
    backend_contract_session_options: SessionOptions,
    backend_contract_capabilities: ClientCapabilities,
) -> AsyncIterator[AppServerHost]:
    connection = await connect_backend_contract_host(
        session_options=backend_contract_session_options,
        capabilities=backend_contract_capabilities,
    )
    try:
        yield connection.host
    finally:
        await connection.host.close()


@pytest_asyncio.fixture
async def backend_contract_connection(
    backend_contract_session_options: SessionOptions,
    backend_contract_capabilities: ClientCapabilities,
) -> AsyncIterator[BackendContractConnection]:
    connection = await connect_backend_contract_host(
        session_options=backend_contract_session_options,
        capabilities=backend_contract_capabilities,
    )
    try:
        yield connection
    finally:
        await connection.host.close()


@pytest_asyncio.fixture
async def backend_contract_session(
    backend_contract_connection: BackendContractConnection,
    backend_contract_mistral_api: respx.Route,
) -> AsyncIterator[AppServerSession]:
    session = await backend_contract_connection.host.open_session()
    try:
        yield session
    finally:
        await session.close()


@pytest_asyncio.fixture
async def backend_contract_persistent_connection(
    config_dir: Path, tmp_path: Path
) -> AsyncIterator[BackendContractConnection]:
    config_file = config_dir / "config.toml"
    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    config["session_logging"] = {
        "enabled": True,
        "save_dir": str(tmp_path / "sessions"),
    }
    config_file.write_text(tomli_w.dumps(config), encoding="utf-8")
    connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    try:
        yield connection
    finally:
        await connection.host.close()


@pytest_asyncio.fixture
async def backend_contract_persistent_session(
    backend_contract_mistral_api: respx.Route,
    backend_contract_persistent_connection: BackendContractConnection,
) -> AsyncIterator[AppServerSession]:
    session = await backend_contract_persistent_connection.host.open_session()
    try:
        yield session
    finally:
        await session.close()


class _MemoryLoopFactory:
    """Serve one already-built loop through a RootAgentServer, in-process."""

    def __init__(self, loop: AgentLoop, child_host: Any = None) -> None:
        self._loop = loop
        # A process-owned root installs the child host its factory exposes;
        # the management-surface nodes inject one to exercise the owner.
        self.child_host = child_host

    async def open(self, params: Mapping[str, Any]) -> AgentLoop:
        return self._loop

    async def close(self) -> None:
        return None


class _LiveProcessHandle:
    """A process-handle stand-in whose pipe is a memory transport pair."""

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def install_memory_root_agent(
    monkeypatch: pytest.MonkeyPatch, loop: AgentLoop, child_host: Any = None
) -> None:
    """Replace the stdio spawn seam with a root agent served over a memory wire.

    The "spawned" root agent then runs inside the test process -- no child
    process and no provider spawn -- so the process-session backend (the
    stdio topology's carrier) stays exercisable without the B1 spawned
    gate. Only the spawn is replaced: the backend, the server routing and
    the agent wire are the production ones.

    The stdio selection is made here AFTER the suite autouse pinned
    ``in_process`` (this runs from a fixture or test body, both after that
    autouse), the child's actual transport is asserted at the spawn seam --
    the env materialized for the spawn is what a real spawned root would
    inherit -- and the memory root's ready frame advertises this
    installation's additive agent-wire capabilities, exactly as a spawned
    child of this build would.
    """
    from chartreux.app_server import process_session_backend as psb_module
    from chartreux.app_server.process_bootstrap import (
        PROCESS_PROTOCOL_VERSION,
        ReadyFrame,
        SpawnedProcess,
    )
    from chartreux.app_server.root_agent import RootAgentServer
    from chartreux.app_server.root_agent_process import RootAgentProcess

    monkeypatch.setenv(TRANSPORT_ENV_VAR, STDIO_TRANSPORT)
    assert resolve_transport_selection(environ=os.environ).kind == STDIO_TRANSPORT
    executed_receipts.stamp_transport(STDIO_TRANSPORT)

    async def fake_spawn(**kwargs: Any) -> RootAgentProcess:
        spawn_env = kwargs.get("env")
        # The child's actual transport: the spawn env materialized from the
        # stdio selection is what a spawned root inherits.
        assert isinstance(spawn_env, Mapping)
        assert spawn_env.get(TRANSPORT_ENV_VAR) == STDIO_TRANSPORT
        client_transport, server_transport = memory_transport_pair()
        serving = asyncio.create_task(
            RootAgentServer(
                server_transport,
                factory=_MemoryLoopFactory(loop, child_host=child_host),
            ).serve()
        )
        spawned = SpawnedProcess(
            process=cast("Any", _LiveProcessHandle()),
            transport=cast("Any", client_transport),
            ready=ReadyFrame(
                protocol_version=PROCESS_PROTOCOL_VERSION,
                capabilities=ADVERTISED_CAPABILITIES,
            ),
        )
        request_timeout = kwargs.get("request_timeout")
        process = RootAgentProcess(
            spawned,
            heartbeat_interval=30.0,
            request_timeout=(
                request_timeout if isinstance(request_timeout, float) else 30.0
            ),
        )
        try:
            await process.open(
                session_options=kwargs["session_options"],
                client_info=kwargs["client_info"],
            )
        except BaseException:
            with suppress(BaseException):
                await asyncio.wait_for(serving, timeout=10)
            raise
        return process

    monkeypatch.setattr(psb_module, "spawn_root_agent_process", fake_spawn)


async def connect_backend_contract_stdio_host(
    *, session_options: SessionOptions, capabilities: ClientCapabilities
) -> BackendContractConnection:
    """The in_process connector's stdio twin.

    The session backend is the process-session backend (transport kind
    stdio); pair it with :func:`install_memory_root_agent` so the root agent
    runs over a memory wire instead of a spawned child.
    """
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(server_transport, transport_kind="stdio")
    client = AppServerClient(client_transport, run_peer=harness.serve)
    host = await AppServerHost.connect(
        client,
        client_info=ClientInfo(name="backend-contract-test", version="0"),
        capabilities=capabilities,
        session_options=session_options,
        client_factory=harness.connect_client,
    )
    return BackendContractConnection(client=client, host=host)


@pytest_asyncio.fixture
async def backend_contract_transport_connection(
    monkeypatch: pytest.MonkeyPatch,
    stdio_after_autouse: str,
    backend_contract_session_options: SessionOptions,
    backend_contract_capabilities: ClientCapabilities,
) -> AsyncIterator[BackendContractConnection]:
    """One connection on the node's selected transport, post-autouse.

    The stdio leg selects stdio after the suite autouse and runs the root
    agent over a memory wire (see :func:`install_memory_root_agent`, whose
    spawn seam asserts the child's actual transport); the in_process leg is
    the ordinary connector. Node ids carry the transport, so ``-k stdio``
    selects the stdio legs.
    """
    loop: AgentLoop | None = None
    if stdio_after_autouse == STDIO_TRANSPORT:
        loop = build_test_agent_loop()
        install_memory_root_agent(monkeypatch, loop)
        connection = await connect_backend_contract_stdio_host(
            session_options=backend_contract_session_options,
            capabilities=backend_contract_capabilities,
        )
    else:
        connection = await connect_backend_contract_host(
            session_options=backend_contract_session_options,
            capabilities=backend_contract_capabilities,
        )
    try:
        yield connection
    finally:
        await connection.host.close()
        if loop is not None:
            await loop.aclose()


@pytest_asyncio.fixture
async def backend_contract_transport_session(
    backend_contract_transport_connection: BackendContractConnection,
    backend_contract_mistral_api: respx.Route,
) -> AsyncIterator[AppServerSession]:
    """One session on the node's selected transport."""
    session = await backend_contract_transport_connection.host.open_session()
    try:
        yield session
    finally:
        await session.close()
