from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Collection
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from chartreux.acp.agent import ChartreuxAcpAgent, SessionStarter
from chartreux.app_server._runtime import HarnessServer, create_harness_server
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.local import LocalHarnessOptions, ResumeSessionIntent
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ProtocolError,
    ProtocolErrorCode,
    SessionOptions,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import (
    IN_PROCESS_TRANSPORT,
    STDIO_TRANSPORT,
    memory_transport_pair,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import (
    ChartreuxConfigSchema,
    ModelConfig,
    SessionLoggingConfig,
)
from chartreux.core.config.harness_files import get_harness_files_manager
from chartreux.core.dispatch.session import bind_policy_bundle
from chartreux.core.llm_models import LLMChunk, LLMMessage, LLMUsage, Role, StopInfo
from chartreux.core.policy.composition import ExecutionContext
from chartreux.core.session_types import POLICY_STORAGE_VERSION
from chartreux.core.skills.manager import SkillManager
from tests.app_server import executed_receipts
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import start_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_client import FakeClient

#: The transports an ACP session node can be pinned on. The ids carry the
#: transport name, so ``-k stdio`` selects exactly the stdio legs.
ACP_TRANSPORTS = (IN_PROCESS_TRANSPORT, STDIO_TRANSPORT)


@dataclass(slots=True)
class AcpTransportHarness:
    """One ACP agent whose session sits on a chosen transport."""

    transport: str
    agent: ChartreuxAcpAgent
    client: FakeClient
    server: AppServerClient
    loop: AgentLoop
    _harness: HarnessServer | None = None

    async def close(self) -> None:
        await self.agent.close()
        await self.server.close()
        if self._harness is not None:
            await self._harness.close()


async def start_acp_agent_on(
    transport: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_options: SessionOptions,
    config: ChartreuxConfigSchema | None = None,
    backend: FakeBackend | None = None,
) -> AcpTransportHarness:
    """Start one ACP agent whose session runs on ``transport``.

    The stdio leg is the production topology driven over a memory wire: the
    spawn seam is replaced by a root agent served in this process, so the
    process-session backend, the server routing and the agent wire are the
    real ones. Spawned-child legs land with the PTY harness (C5c).
    """
    loop = build_test_agent_loop(config=config, backend=backend, enable_streaming=True)
    executed_receipts.stamp_transport(transport)
    harness: HarnessServer | None = None
    if transport == STDIO_TRANSPORT:
        from tests.app_server.backend_contract.conftest import install_memory_root_agent

        install_memory_root_agent(monkeypatch, loop)
        client_transport, server_transport = memory_transport_pair()
        harness = await create_harness_server(
            server_transport, transport_kind=STDIO_TRANSPORT
        )
        server = AppServerClient(client_transport, run_peer=harness.serve)
    elif transport == IN_PROCESS_TRANSPORT:
        server = start_test_app_server(loop)
    else:  # pragma: no cover - a typo in the fixture parameter, not a code path
        raise AssertionError(f"unsupported ACP test transport: {transport}")

    async def start_session(options: LocalHarnessOptions) -> AppServerSession:
        return await AppServerSession.start(
            server,
            client_info=options.client.info,
            capabilities=options.client.capabilities,
            session_options=session_options,
        )

    starter: SessionStarter = start_session
    agent = ChartreuxAcpAgent(session_starter=starter)
    client = FakeClient()
    agent.on_connect(client)
    client.on_connect(agent)
    return AcpTransportHarness(
        transport=transport,
        agent=agent,
        client=client,
        server=server,
        loop=loop,
        _harness=harness,
    )


@pytest_asyncio.fixture
async def acp_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[..., Awaitable[AcpTransportHarness]]]:
    """Start ACP agents on a chosen transport, closed again at teardown."""
    started: list[AcpTransportHarness] = []

    async def start(
        transport: str,
        *,
        session_options: SessionOptions | None = None,
        config: ChartreuxConfigSchema | None = None,
        backend: FakeBackend | None = None,
    ) -> AcpTransportHarness:
        harness = await start_acp_agent_on(
            transport,
            monkeypatch,
            session_options=session_options or SessionOptions(),
            config=config,
            backend=backend,
        )
        started.append(harness)
        return harness

    try:
        yield start
    finally:
        for harness in reversed(started):
            await harness.close()


@dataclass(slots=True)
class RecordedRequests:
    """The app-server methods a session's client actually issued."""

    methods: list[str]

    def clear(self) -> None:
        self.methods.clear()

    def issued(self, method: str) -> int:
        return self.methods.count(method)


def record_requests(
    monkeypatch: pytest.MonkeyPatch,
    server: AppServerClient,
    *,
    blocked: Collection[str] = frozenset(),
) -> RecordedRequests:
    """Record every app-server request a session's client sends.

    The ACP extension surface is a mapping onto app-server methods, so the
    mapping is what the pins assert: the wire method that actually crossed is
    the one this records, not the one a test or a docstring claims. A blocked
    method is recorded and answered with a typed protocol error instead of
    reaching the server, so a pin can assert the request a call issues without
    letting it persist anything.
    """
    recorded = RecordedRequests([])
    original = server.request

    async def recording(method: str, params: Any = None) -> Any:
        recorded.methods.append(method)
        if method in blocked:
            raise AppServerResponseError(
                ProtocolError(
                    code=ProtocolErrorCode.INTERNAL_ERROR,
                    message=f"{method} is blocked by this pin",
                )
            )
        return await original(method, params)

    monkeypatch.setattr(server, "request", recording)
    return recorded


@pytest.fixture
def acp_request_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., RecordedRequests]:
    """The recorder a transport harness exposes to a node."""

    def record(
        server: AppServerClient, *, blocked: Collection[str] = frozenset()
    ) -> RecordedRequests:
        return record_requests(monkeypatch, server, blocked=blocked)

    return record


@pytest.fixture
def backend() -> FakeBackend:
    backend = FakeBackend(
        LLMChunk(
            message=LLMMessage(role=Role.assistant, content="Hi"),
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1),
            stop=StopInfo(reason="stop"),
        )
    )
    return backend


def _create_acp_agent(
    session_starter: SessionStarter | None = None,
) -> ChartreuxAcpAgent:
    vibe_acp_agent = (
        ChartreuxAcpAgent(session_starter=session_starter)
        if session_starter is not None
        else ChartreuxAcpAgent()
    )
    client = FakeClient()

    vibe_acp_agent.on_connect(client)
    client.on_connect(vibe_acp_agent)

    return vibe_acp_agent  # pyright: ignore[reportReturnType]


@pytest.fixture
def acp_agent_loop(backend: FakeBackend) -> ChartreuxAcpAgent:
    async def start_session(options: LocalHarnessOptions) -> AppServerSession:
        loop = build_test_agent_loop(backend=backend, enable_streaming=True)
        return await AppServerSession.start(
            start_test_app_server(loop),
            client_info=options.client.info,
            capabilities=options.client.capabilities,
            session_options=options.session_options,
        )

    return _create_acp_agent(start_session)


@pytest.fixture
def acp_agent_with_session_config(
    backend: FakeBackend, temp_session_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ChartreuxAcpAgent, FakeClient]:
    session_config = SessionLoggingConfig(
        save_dir=str(temp_session_dir), session_prefix="session", enabled=True
    )
    config = build_test_vibe_config(
        active_model="glm-5-2",
        models=[ModelConfig(name="glm-5-2", provider="mistral", alias="glm-5-2")],
        session_logging=session_config,
    )

    async def start_session(options: LocalHarnessOptions) -> AppServerSession:
        loop = build_test_agent_loop(
            config=config, backend=backend, enable_streaming=True
        )
        return await AppServerSession.start(
            start_test_app_server(loop),
            client_info=options.client.info,
            capabilities=options.client.capabilities,
            session_options=options.session_options,
            resume_session_id=(
                options.session.session_id
                if isinstance(options.session, ResumeSessionIntent)
                else None
            ),
        )

    vibe_acp_agent = ChartreuxAcpAgent(session_starter=start_session)
    client = FakeClient()
    vibe_acp_agent.on_connect(client)
    client.on_connect(vibe_acp_agent)

    return vibe_acp_agent, client


@pytest.fixture
def temp_session_dir(tmp_path: Path) -> Path:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    return session_dir


@pytest.fixture
def create_test_session():
    """Create a test session with configurable messages and metadata.

    Supports both messages parameter (for load_session tests) and
    end_time parameter (for list_sessions tests).
    """

    def _create_session(
        session_dir: Path,
        session_id: str,
        cwd: str,
        messages: list[dict] | None = None,
        title: str | None = None,
        end_time: str | None = None,
        parent_session_id: str | None = None,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = session_dir / f"session_{timestamp}_{session_id[:8]}"
        session_folder.mkdir(exist_ok=True)

        if messages is None:
            messages = [{"role": "user", "content": "Hello"}]

        messages_file = session_folder / "messages.jsonl"
        with messages_file.open("w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        metadata = {
            "session_id": session_id,
            "start_time": "2024-01-01T12:00:00Z",
            "end_time": end_time or "2024-01-01T12:05:00Z",
            "git_commit": None,
            "git_branch": None,
            "username": "test-user",
            "environment": {"working_directory": cwd},
            "title": title,
        }
        if parent_session_id is not None:
            metadata["parent_session_id"] = parent_session_id
        # Package 10: the persisted policy record of a conversation is its
        # frozen policy bundle (design §10) — resume rejects bundle-less
        # legacy records actionably, so the fixture composes the same
        # current-format record production writes.
        config = build_test_vibe_config()
        bundle, _activated = bind_policy_bundle(
            config,
            execution_context=ExecutionContext.INTERACTIVE_MAIN,
            skill_manager=SkillManager(lambda: config),
            scratchpad_dir=None,
            cwd=Path(cwd),
            harness_files=get_harness_files_manager(),
            role_instructions=None,
        )
        metadata["policy_bundle"] = bundle.model_dump(mode="json")
        metadata["policy_storage_version"] = POLICY_STORAGE_VERSION

        metadata_file = session_folder / "meta.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f)

        return session_folder

    return _create_session
