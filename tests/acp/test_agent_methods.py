from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

from acp.schema import (
    AgentMessageChunk,
    SessionInfoUpdate,
    TextContentBlock,
    UsageUpdate,
    UserMessageChunk,
)
import pytest

from chartreux.acp.agent import ChartreuxAcpAgent, SessionStarter
from chartreux.acp.exceptions import InvalidRequestError, NotImplementedMethodError
from chartreux.app_server.local import LocalHarnessOptions
from chartreux.app_server.models import (
    IdleSessionStatus,
    PublicMessageEntry,
    PublicSession,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    PageRequest,
    ProtocolError,
    ProtocolErrorCode,
    RuntimeReadParams,
    RuntimeReadResponse,
    SessionOptions,
    SessionReadParams,
    SessionTurnsListParams,
    SessionTurnsListResponse,
    SkillsInstalledParams,
    SkillsInstalledResponse,
    SkillsListParams,
    SkillsListResponse,
    StatsReadParams,
    StatsReadResponse,
    ToolsListParams,
    ToolsListResponse,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import (
    IN_PROCESS_TRANSPORT,
    STDIO_TRANSPORT,
    TRANSPORT_ENV_VAR,
)
from chartreux.core.config import ModelConfig, SessionLoggingConfig
from chartreux.core.llm_models import LLMChunk, LLMMessage, LLMUsage, Role, StopInfo
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from tests import conftest as tests_conftest
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import start_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_client import FakeClient


@pytest.fixture
def documented_context_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare a provider context limit on the catalogs the fixture builds.

    The session-config fixture defines a throwaway deployment, which documents
    no limit; the ACP usage update is skipped when the size is unknown, so the
    update path would never be exercised. Declare the fact on the materialized
    test catalog instead of changing what the fixture asks for.
    """
    original = tests_conftest._test_catalog_snapshot

    def with_documented_limit(kwargs: dict[str, Any]) -> CatalogSnapshot:
        snapshot = original(kwargs)
        raw = snapshot.catalog.model_dump(mode="json")
        for model in raw["models"].values():
            for deployment in model["deployments"]:
                if deployment["context_limit"] is None:
                    deployment["context_limit"] = 1_000_000
        return CatalogSnapshot(ModelCatalog.model_validate(raw), snapshot.revision)

    monkeypatch.setattr(tests_conftest, "_test_catalog_snapshot", with_documented_limit)


async def _new_session(agent: ChartreuxAcpAgent) -> str:
    response = await agent.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    return response.session_id


def _client(agent: ChartreuxAcpAgent) -> FakeClient:
    assert isinstance(agent.client, FakeClient)
    return agent.client


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_session_responses_expose_validation_warnings(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    notes = [
        "R3 contrasts: purpose mismatch",
        "Dispatch slot 'mechanical': unavailable",
    ]
    session = Mock(id="diagnostic-session")
    session.app_server.resources.config.current.validation_warnings = notes
    session.app_server.state = Mock()
    monkeypatch.setattr(
        acp_agent_loop, "_create_session", AsyncMock(return_value=session)
    )
    monkeypatch.setattr(acp_agent_loop, "_config_options", Mock(return_value=[]))
    monkeypatch.setattr(
        acp_agent_loop,
        "_trust_meta",
        AsyncMock(return_value={"workspace_trust": {"status": "trusted"}}),
    )
    monkeypatch.setattr(acp_agent_loop, "_send_usage_update", Mock())
    monkeypatch.setattr(acp_agent_loop, "_load_complete_history", AsyncMock())
    monkeypatch.setattr("chartreux.acp.agent.replay_session_updates", lambda _: [])

    if resume:
        response = await acp_agent_loop.load_session(cwd=".", session_id=session.id)
    else:
        response = await acp_agent_loop.new_session(cwd=".")
    assert response is not None
    assert response.field_meta == {
        "workspace_trust": {"status": "trusted"},
        "validation_warnings": notes,
    }
    assert response.model_dump(by_alias=True)["_meta"]["validation_warnings"] == notes


@pytest.mark.asyncio
async def test_close_session_remains_retryable_after_cleanup_failure(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = await _new_session(acp_agent_loop)
    session = acp_agent_loop.sessions[session_id]
    original_close = session.close
    attempts = 0

    async def fail_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cleanup failed")
        await original_close()

    monkeypatch.setattr(session, "close", fail_once)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await acp_agent_loop.close_session(session_id)

    assert acp_agent_loop.sessions[session_id] is session

    await acp_agent_loop.close_session(session_id)

    assert session_id not in acp_agent_loop.sessions


@pytest.mark.asyncio
async def test_delete_session_remains_retryable_after_cleanup_failure(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = await _new_session(acp_agent_loop)
    session = acp_agent_loop.sessions[session_id]
    original_close = session.close
    attempts = 0

    async def fail_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cleanup failed")
        await original_close()

    monkeypatch.setattr(session, "close", fail_once)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await acp_agent_loop.ext_method("session/delete", {"sessionId": session_id})

    assert acp_agent_loop.sessions[session_id] is session

    await acp_agent_loop.ext_method("session/delete", {"sessionId": session_id})

    assert attempts == 2
    assert session_id not in acp_agent_loop.sessions


@pytest.mark.asyncio
async def test_delete_session_uses_same_session_id_after_compaction(
    acp_agent_with_session_config: tuple[ChartreuxAcpAgent, FakeClient],
    backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, _ = acp_agent_with_session_config
    created = await agent.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    await agent.prompt(
        session_id=created.session_id,
        prompt=[TextContentBlock(type="text", text="Hello")],
    )
    session = agent.sessions[created.session_id]
    backend._streams.append([mock_llm_chunk(content="<summary>done</summary>")])

    await session.app_server.compact()

    session_id = session.app_server.session_id
    assert session_id == created.session_id
    assert session.app_server.resources.runtime.session_log.persisted
    host = Mock(delete_session=AsyncMock())
    monkeypatch.setattr(agent, "_host_resources", AsyncMock(return_value=host))

    await agent.ext_method("session/delete", {"sessionId": created.session_id})

    assert created.session_id not in agent.sessions
    host.delete_session.assert_awaited_once_with(session_id)


@pytest.mark.asyncio
async def test_config_options_delegate_to_typed_app_server_resources(
    acp_agent_loop: ChartreuxAcpAgent,
) -> None:
    session_id = await _new_session(acp_agent_loop)
    session = acp_agent_loop.sessions[session_id]
    active_model = session.app_server.resources.config.current.active_model.alias

    model = await acp_agent_loop.set_config_option("model", session_id, active_model)
    thinking = await acp_agent_loop.set_config_option("thinking", session_id, "low")
    max_turns = await acp_agent_loop.set_config_option("max_turns", session_id, "3")
    max_tokens = await acp_agent_loop.set_config_option(
        "max_tokens", session_id, "2000"
    )

    assert all(
        response is not None for response in (model, thinking, max_turns, max_tokens)
    )
    with pytest.raises(InvalidRequestError):
        await acp_agent_loop.set_config_option("unknown", session_id, "x")


def _acp_agent_with_allowed_models() -> ChartreuxAcpAgent:
    config = build_test_vibe_config(
        active_model="allowed",
        allowed_models=["glm-5-2", "allowed"],
        models=[
            ModelConfig(name="allowed", provider="mistral", alias="allowed"),
            ModelConfig(name="blocked", provider="mistral", alias="blocked"),
        ],
    )

    async def start_session(options: LocalHarnessOptions) -> AppServerSession:
        loop = build_test_agent_loop(
            config=config, backend=FakeBackend(), enable_streaming=True
        )
        return await AppServerSession.start(
            start_test_app_server(loop),
            client_info=options.client.info,
            capabilities=options.client.capabilities,
            session_options=options.session_options,
        )

    starter: SessionStarter = start_session
    agent = ChartreuxAcpAgent(session_starter=starter)
    client = FakeClient()
    agent.on_connect(client)
    client.on_connect(agent)
    return agent


@pytest.mark.asyncio
async def test_set_config_option_rejects_model_excluded_by_allowed_models() -> None:
    agent = _acp_agent_with_allowed_models()
    session_id = await _new_session(agent)

    with pytest.raises(InvalidRequestError):
        await agent.set_config_option("model", session_id, "blocked")

    allowed = await agent.set_config_option("model", session_id, "allowed")
    assert allowed is not None


@pytest.mark.asyncio
async def test_set_config_option_raises_instead_of_reporting_an_empty_config(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = await _new_session(acp_agent_loop)
    session = acp_agent_loop.sessions[session_id]
    active_model = session.app_server.resources.config.current.active_model.alias

    async def busy(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AppServerResponseError(
            ProtocolError(
                code=ProtocolErrorCode.INTERNAL_ERROR,
                message="Session is busy running prompt 1",
            )
        )

    monkeypatch.setattr(session.app_server.resources.config, "update", busy)

    with pytest.raises(InvalidRequestError) as excinfo:
        await acp_agent_loop.set_config_option("model", session_id, active_model)

    assert "busy" in str(excinfo.value)


@pytest.mark.asyncio
async def test_prompt_reports_public_usage_and_stable_message_ids(
    acp_agent_loop: ChartreuxAcpAgent,
) -> None:
    session_id = await _new_session(acp_agent_loop)
    client = _client(acp_agent_loop)
    client._session_updates.clear()

    response = await acp_agent_loop.prompt(
        session_id=session_id, prompt=[TextContentBlock(type="text", text="Hello")]
    )

    def usage_updates() -> list[UsageUpdate]:
        return [
            notification.update
            for notification in client._session_updates
            if isinstance(notification.update, UsageUpdate)
        ]

    for _ in range(50):
        await asyncio.sleep(0)
        if len(usage_updates()) >= 2:
            break

    assert response.usage is not None
    assert response.usage.total_tokens == 2
    messages = [
        notification.update
        for notification in client._session_updates
        if isinstance(notification.update, UserMessageChunk | AgentMessageChunk)
    ]
    assert [message.content.text for message in messages] == ["Hello", "Hi"]
    assert all(message.message_id for message in messages)
    usage = usage_updates()
    assert len(usage) >= 2
    assert usage[-1].used == 2


@pytest.mark.asyncio
async def test_initialize_advertises_http_mcp_and_set_mode_is_unsupported(
    acp_agent_loop: ChartreuxAcpAgent,
) -> None:
    response = await acp_agent_loop.initialize(protocol_version=1)

    capabilities = response.agent_capabilities
    assert capabilities is not None
    assert capabilities.mcp_capabilities is not None
    assert capabilities.mcp_capabilities.http is True
    with pytest.raises(NotImplementedMethodError, match="session/set_mode"):
        await acp_agent_loop.set_session_mode("session", "mode")


@pytest.mark.asyncio
async def test_fork_returns_session_state_without_replaying_history(
    acp_agent_with_session_config: tuple[ChartreuxAcpAgent, FakeClient],
) -> None:
    agent, client = acp_agent_with_session_config
    try:
        source_id = await _new_session(agent)
        await agent.prompt(
            session_id=source_id, prompt=[TextContentBlock(type="text", text="Hello")]
        )
        client._session_updates.clear()

        response = await agent.fork_session(
            source_id, cwd=str(Path.cwd()), mcp_servers=[]
        )
        await asyncio.sleep(0)

        child = agent.sessions[response.session_id]
        assert any(
            isinstance(entry, PublicMessageEntry)
            for entry in child.app_server.state.history or []
        )
        assert response.config_options is not None
        assert not any(
            isinstance(notification.update, UserMessageChunk | AgentMessageChunk)
            for notification in client._session_updates
        )
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_fork_session_sends_usage_update(
    documented_context_limit: None,
    acp_agent_with_session_config: tuple[ChartreuxAcpAgent, FakeClient],
) -> None:
    agent, client = acp_agent_with_session_config
    try:
        source_id = await _new_session(agent)
        await agent.prompt(
            session_id=source_id, prompt=[TextContentBlock(type="text", text="Hello")]
        )
        client._session_updates.clear()

        await agent.fork_session(source_id, cwd=str(Path.cwd()), mcp_servers=[])
        await asyncio.sleep(0)

        usage_updates = [
            notification.update
            for notification in client._session_updates
            if isinstance(notification.update, UsageUpdate)
        ]
        assert len(usage_updates) >= 1
        assert usage_updates[-1].size > 0
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_resume_session_sends_usage_update(
    documented_context_limit: None,
    acp_agent_with_session_config: tuple[ChartreuxAcpAgent, FakeClient],
    temp_session_dir: Path,
    create_test_session,
) -> None:
    agent, client = acp_agent_with_session_config
    session_id = "resumable-with-stats"
    create_test_session(
        temp_session_dir,
        session_id,
        str(Path.cwd()),
        messages=[{"role": "user", "content": "Hello"}],
    )
    client._session_updates.clear()

    await agent.resume_session(
        session_id=session_id, cwd=str(Path.cwd()), mcp_servers=[]
    )
    await asyncio.sleep(0)

    usage_updates = [
        notification.update
        for notification in client._session_updates
        if isinstance(notification.update, UsageUpdate)
    ]
    assert len(usage_updates) >= 1
    await agent.close()


@pytest.mark.asyncio
async def test_load_session_replays_history_beyond_snapshot_page(
    acp_agent_with_session_config: tuple[ChartreuxAcpAgent, FakeClient],
    temp_session_dir: Path,
    create_test_session,
) -> None:
    agent, client = acp_agent_with_session_config
    session_id = "long-history-session"
    messages = [{"role": "user", "content": f"message {index}"} for index in range(205)]
    create_test_session(
        temp_session_dir, session_id, str(Path.cwd()), messages=messages
    )
    client._session_updates.clear()

    await agent.load_session(cwd=str(Path.cwd()), session_id=session_id, mcp_servers=[])

    replayed = [
        notification.update.content.text
        for notification in client._session_updates
        if isinstance(notification.update, UserMessageChunk)
    ]
    assert replayed == [f"message {index}" for index in range(205)]


@pytest.mark.asyncio
async def test_title_and_session_listing_cross_the_app_server_boundary(
    acp_agent_loop: ChartreuxAcpAgent,
) -> None:
    session_id = await _new_session(acp_agent_loop)

    title = await acp_agent_loop.ext_method(
        "session/set_title", {"sessionId": session_id, "title": "Reviewed"}
    )
    listed = await acp_agent_loop.list_sessions(cwd=str(Path.cwd()))

    assert title == {}
    assert (
        acp_agent_loop.sessions[session_id].app_server.state.session.title == "Reviewed"
    )
    assert listed.sessions == []
    updates = [
        notification.update
        for notification in _client(acp_agent_loop)._session_updates
        if isinstance(notification.update, SessionInfoUpdate)
    ]
    assert updates[-1].title == "Reviewed"


@pytest.mark.asyncio
async def test_list_sessions_falls_back_to_preview_when_no_title(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An LLM title is only generated after a few turns; until then the ACP list
    # must surface the first-message preview so entries stay identifiable.
    items = [
        PublicSession(
            id="untitled-session",
            title=None,
            preview="MARKER-ONE first message",
            status=IdleSessionStatus(),
            created_at=0,
            updated_at=0,
            cwd=str(Path.cwd()),
        ),
        PublicSession(
            id="titled-session",
            title="Generated title",
            preview="MARKER-TWO first message",
            status=IdleSessionStatus(),
            created_at=0,
            updated_at=0,
            cwd=str(Path.cwd()),
        ),
    ]
    resources = Mock()
    resources.list_sessions = AsyncMock(return_value=items)
    monkeypatch.setattr(
        acp_agent_loop, "_host_resources", AsyncMock(return_value=resources)
    )

    listed = await acp_agent_loop.list_sessions(cwd=str(Path.cwd()))

    titles = {session.session_id: session.title for session in listed.sessions}
    assert titles["untitled-session"] == "MARKER-ONE first message"
    assert titles["titled-session"] == "Generated title"


# -- both-transport pins for the latent ACP-reachable reads -------------------
#
# These methods are reachable from an ACP-hosted session but were in-process
# only: the stdio topology refused them. Each node runs on both transports and
# asserts the same typed answer, so a carrier that stops answering fails here
# instead of surfacing as a client-side refusal at runtime. The stdio leg is
# the production topology over a memory wire; the spawned-child legs land with
# the PTY harness (C5c).

ACP_TRANSPORTS = (IN_PROCESS_TRANSPORT, STDIO_TRANSPORT)

_CATALOG_READ_PARAMS = {
    "skills/list": SkillsListParams,
    "skills/installed": SkillsInstalledParams,
    "tools/list": ToolsListParams,
    "stats/read": StatsReadParams,
}

#: The app-server method each ACP rewind call must reach. The manifest (C5a)
#: cites the node ids in ``REWIND_MAPPING_TEST_IDS`` for this mapping; each node
#: asserts it still runs under a listed id, so a rename or a move fails the pin
#: instead of silently orphaning the manifest entry.
REWIND_APP_SERVER_METHODS = {
    "rewind/preview": "session/rewind/read",
    "rewind/to": "session/rewind",
}
REWIND_MAPPING_TEST_IDS = {
    method: (
        f"tests/acp/test_agent_methods.py::test_rewind_mapping_pins"
        f"[{method}-{IN_PROCESS_TRANSPORT}]",
        f"tests/acp/test_agent_methods.py::test_rewind_mapping_pins"
        f"[{method}-{STDIO_TRANSPORT}]",
    )
    for method in REWIND_APP_SERVER_METHODS
}


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
async def test_latent_catalog_reads_answer_on_both_transports(
    transport: str, acp_sessions
) -> None:
    """in_process and stdio: the same owner answers the same projection."""
    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(Path.cwd()))
    )
    session_id = await _new_session(harness.agent)

    listed_skills = SkillsListResponse.model_validate(
        await harness.server.request(
            "skills/list", SkillsListParams(session_id=session_id)
        )
    )
    installed_skills = SkillsInstalledResponse.model_validate(
        await harness.server.request(
            "skills/installed", SkillsInstalledParams(session_id=session_id)
        )
    )
    tools = ToolsListResponse.model_validate(
        await harness.server.request(
            "tools/list", ToolsListParams(session_id=session_id)
        )
    )
    stats = StatsReadResponse.model_validate(
        await harness.server.request(
            "stats/read", StatsReadParams(session_id=session_id)
        )
    )

    assert isinstance(listed_skills.skills, list)
    assert installed_skills.skills == []
    assert "write_file" in {tool.name for tool in tools.tools}
    assert stats.stats.steps == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
@pytest.mark.parametrize("method", sorted(_CATALOG_READ_PARAMS))
async def test_latent_catalog_reads_refuse_another_session(
    transport: str, method: str, acp_sessions
) -> None:
    """The owner refuses an id it does not serve, on both transports."""
    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(Path.cwd()))
    )
    await _new_session(harness.agent)

    with pytest.raises(AppServerResponseError) as excinfo:
        await harness.server.request(
            method, _CATALOG_READ_PARAMS[method](session_id="not-this-session")
        )

    assert excinfo.value.error.code is ProtocolErrorCode.NOT_FOUND


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
async def test_session_turns_list_pages_the_live_projection_on_both_transports(
    transport: str, acp_sessions, backend: FakeBackend
) -> None:
    """The turn page is the live projection, whichever end holds it."""
    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(Path.cwd())), backend=backend
    )
    session_id = await _new_session(harness.agent)
    await harness.agent.prompt(
        session_id=session_id, prompt=[TextContentBlock(type="text", text="Hello")]
    )

    page = SessionTurnsListResponse.model_validate(
        await harness.server.request(
            "session/turns/list", SessionTurnsListParams(session_id=session_id)
        )
    )

    assert [turn.status for turn in page.items] == ["completed"]
    assert page.next_cursor is None
    assert page.previous_cursor is None

    # An id neither live nor saved reaches the host projection, which does not
    # know it: the narrowed fallback answers the unknown method, identically on
    # both transports.
    with pytest.raises(AppServerResponseError) as excinfo:
        await harness.server.request(
            "session/turns/list", SessionTurnsListParams(session_id="not-this-session")
        )
    assert excinfo.value.error.code is ProtocolErrorCode.METHOD_NOT_FOUND


async def _turns_page(
    harness_server: Any, session_id: str, **page: Any
) -> SessionTurnsListResponse:
    """One page of a session's turns, over whichever transport the node holds."""
    return SessionTurnsListResponse.model_validate(
        await harness_server.request(
            "session/turns/list",
            SessionTurnsListParams(session_id=session_id, page=PageRequest(**page)),
        )
    )


def _prompt_chunk() -> LLMChunk:
    return LLMChunk(
        message=LLMMessage(role=Role.assistant, content="Hi"),
        usage=LLMUsage(prompt_tokens=1, completion_tokens=1),
        stop=StopInfo(reason="stop"),
    )


class _PausableBackend(FakeBackend):
    """A backend whose completion waits until the test releases it."""

    def __init__(self) -> None:
        super().__init__(_prompt_chunk())
        self.answering = asyncio.Event()
        self.release = asyncio.Event()

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        self.answering.set()
        await self.release.wait()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
async def test_session_turns_list_survives_the_post_turn_adoptions(
    transport: str, acp_sessions, backend: FakeBackend
) -> None:
    """The settled turn survives the reads its own turn provokes.

    A runtime/read and a session/read each adopt a fresh agent-built state,
    and the agent builds no turn projection of its own: the coordinator's own
    records must keep the completed turn on the page afterwards instead of a
    wholesale adoption wiping it.
    """
    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(Path.cwd())), backend=backend
    )
    session_id = await _new_session(harness.agent)
    await harness.agent.prompt(
        session_id=session_id, prompt=[TextContentBlock(type="text", text="Hello")]
    )

    settled = await _turns_page(harness.server, session_id)
    assert [turn.status for turn in settled.items] == ["completed"]

    read = RuntimeReadResponse.model_validate(
        await harness.server.request(
            "runtime/read", RuntimeReadParams(session_id=session_id)
        )
    )
    assert read.ready is True
    after_runtime_read = await _turns_page(harness.server, session_id)
    assert [turn.id for turn in after_runtime_read.items] == [
        turn.id for turn in settled.items
    ]

    # The read adopts the agent's own state as the projection: the page must
    # name the same settled turn afterwards, not an empty list.
    await harness.server.request(
        "session/read", SessionReadParams(session_id=session_id)
    )
    after_session_read = await _turns_page(harness.server, session_id)
    assert [turn.status for turn in after_session_read.items] == ["completed"]
    assert [turn.id for turn in after_session_read.items] == [
        turn.id for turn in settled.items
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
async def test_session_turns_list_pages_three_settled_turns_in_start_order(
    transport: str, acp_sessions
) -> None:
    """Both ends page the same order and the same cursors over three turns."""
    backend = FakeBackend([[_prompt_chunk()] for _ in range(3)])
    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(Path.cwd())), backend=backend
    )
    session_id = await _new_session(harness.agent)
    for text in ("first", "second", "third"):
        await harness.agent.prompt(
            session_id=session_id, prompt=[TextContentBlock(type="text", text=text)]
        )

    every = await _turns_page(harness.server, session_id)
    ids = [turn.id for turn in every.items]
    assert [turn.status for turn in every.items] == ["completed"] * 3
    assert every.next_cursor is None
    assert every.previous_cursor is None

    first = await _turns_page(harness.server, session_id, limit=2, direction="forward")
    assert [turn.id for turn in first.items] == ids[:2]
    assert first.next_cursor == ids[1]
    assert first.previous_cursor is None

    second = await _turns_page(
        harness.server, session_id, limit=2, direction="forward", cursor=ids[1]
    )
    assert [turn.id for turn in second.items] == ids[2:]
    assert second.next_cursor is None
    assert second.previous_cursor == ids[2]

    # Backward from the tail id answers the page that precedes it, in order.
    back = await _turns_page(
        harness.server, session_id, limit=2, direction="backward", cursor=ids[2]
    )
    assert [turn.id for turn in back.items] == ids[:2]
    assert back.next_cursor is None
    assert back.previous_cursor == ids[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
async def test_session_turns_list_of_a_fresh_session_is_honestly_empty(
    transport: str, acp_sessions
) -> None:
    """A session that ran no turn answers an empty page, not a fabricated one."""
    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(Path.cwd()))
    )
    session_id = await _new_session(harness.agent)

    fresh = await _turns_page(harness.server, session_id)
    assert fresh.items == []
    assert fresh.next_cursor is None
    assert fresh.previous_cursor is None

    # The empty answer is the record store's own state, so an adoption in
    # between (a session read) cannot turn it into anything else.
    await harness.server.request(
        "session/read", SessionReadParams(session_id=session_id)
    )
    after_read = await _turns_page(harness.server, session_id)
    assert after_read.items == []
    assert after_read.next_cursor is None
    assert after_read.previous_cursor is None


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
async def test_session_turns_list_replaces_the_turn_in_place_when_it_settles(
    transport: str, acp_sessions
) -> None:
    """A start pages in_progress; its settlement replaces it, never appends."""
    backend = _PausableBackend()
    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(Path.cwd())), backend=backend
    )
    session_id = await _new_session(harness.agent)
    prompt = asyncio.create_task(
        harness.agent.prompt(
            session_id=session_id, prompt=[TextContentBlock(type="text", text="Hello")]
        )
    )
    try:
        await asyncio.wait_for(backend.answering.wait(), timeout=10)
        # Poll for the start's own projection: a backend call may precede the
        # frame's publication, and the armed completion holds the turn open.
        running = await _turns_page(harness.server, session_id)
        for _ in range(100):
            if running.items:
                break
            await asyncio.sleep(0.02)
            running = await _turns_page(harness.server, session_id)
        assert [turn.status for turn in running.items] == ["in_progress"]
        running_id = running.items[0].id
        assert running.items[0].completed_at is None

        backend.release.set()
        await asyncio.wait_for(prompt, timeout=10)

        settled = await _turns_page(harness.server, session_id)
        assert [turn.id for turn in settled.items] == [running_id]
        assert [turn.status for turn in settled.items] == ["completed"]
    finally:
        if not prompt.done():
            prompt.cancel()
            with suppress(BaseException):
                await prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
@pytest.mark.parametrize("method", sorted(REWIND_APP_SERVER_METHODS))
async def test_rewind_mapping_pins(
    transport: str,
    method: str,
    request: pytest.FixtureRequest,
    acp_sessions,
    acp_request_recorder,
) -> None:
    """Each ACP rewind call crosses as its own app-server method.

    The ACP surface is a mapping, so the pin is the request that actually
    crossed -- not the ACP name and not a docstring. A checkpoint this session
    never recorded is a typed refusal; the mapping stands either way.
    """
    assert request.node.nodeid in REWIND_MAPPING_TEST_IDS[method]

    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(Path.cwd()))
    )
    session_id = await _new_session(harness.agent)
    recorded = acp_request_recorder(harness.server)

    with pytest.raises(InvalidRequestError):
        await harness.agent.ext_method(
            method, {"sessionId": session_id, "messageId": "no-such-entry"}
        )

    assert recorded.issued(REWIND_APP_SERVER_METHODS[method]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
async def test_rewind_refusal_is_worded_for_an_rpc_caller(
    transport: str, acp_sessions
) -> None:
    """C2d: an uncarried rewind reaches an RPC client as an unavailable operation.

    The editor cannot set the agent server's transport selector, so the stdio
    refusal names the operation and the restart its operator can perform rather
    than echoing the selector. The in-process route carries rewind, so a bad
    checkpoint there keeps its own diagnostic instead of reading as
    unsupported.
    """
    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(Path.cwd()))
    )
    session_id = await _new_session(harness.agent)

    with pytest.raises(InvalidRequestError) as caught:
        await harness.agent.ext_method(
            "rewind/to", {"sessionId": session_id, "messageId": "no-such-entry"}
        )

    message = str(caught.value)
    if transport == STDIO_TRANSPORT:
        assert "Session rewind is not available from this agent server" in message
        assert "in-process topology" in message
        assert TRANSPORT_ENV_VAR not in message
    else:
        assert "not available from this agent server" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
async def test_fork_refusal_is_worded_for_an_rpc_caller(
    transport: str, acp_sessions, tmp_path: Path
) -> None:
    """C2f: an uncarried fork reaches an RPC client as an unavailable operation.

    Fork stays deferred over the stdio topology. The ACP surface used to
    re-raise the raw wire text, which names the operator's transport selector;
    it now goes through the same rewording rewind uses. The in-process route
    carries fork, so its own behaviour is unchanged.
    """
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            save_dir=str(tmp_path), session_prefix="session", enabled=True
        )
    )
    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(Path.cwd())), config=config
    )
    session_id = await _new_session(harness.agent)

    if transport == STDIO_TRANSPORT:
        with pytest.raises(InvalidRequestError) as caught:
            await harness.agent.fork_session(
                session_id, cwd=str(Path.cwd()), mcp_servers=[]
            )
        message = str(caught.value)
        assert "Session fork is not available from this agent server" in message
        assert "in-process topology" in message
        assert TRANSPORT_ENV_VAR not in message
    else:
        # The in-process route carries fork, so a bad checkpoint keeps its own
        # diagnostic instead of reading as an unsupported operation.
        with pytest.raises(InvalidRequestError) as caught:
            await harness.agent.fork_session(
                session_id,
                cwd=str(Path.cwd()),
                mcp_servers=[],
                messageId="no-such-entry",
            )
        message = str(caught.value)
        assert "not available from this agent server" not in message
        assert "not carried over the stdio topology" not in message
