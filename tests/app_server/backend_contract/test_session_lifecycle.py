from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
import tomllib
from typing import Any, cast

import httpx
import pytest
import pytest_asyncio
import respx
import tomli_w

from chartreux.app_server._runtime import create_harness_server
from chartreux.app_server._session_backend_port import SessionBackendError
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.models import (
    PublicMessageEntry,
    SessionTextContentBlock,
    TurnUserInputEntry,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    PageRequest,
    ProtocolErrorCode,
    RuntimeReadParams,
    RuntimeReadResponse,
    SessionHistoryClearParams,
    SessionHistoryClearResponse,
    SessionHistoryListParams,
    SessionHistoryListResponse,
    SessionOptions,
    SessionReadParams,
    SessionReadResponse,
    SessionResumeParams,
    SessionResumeResponse,
    SessionStopParams,
    SessionStopResponse,
    TurnEnqueueParams,
)
from chartreux.app_server.root_agent_process import (
    RootAgentSessionNotFound,
    RootDeath,
    RootFailureSignature,
)
from chartreux.app_server.session import AppServerSession, SessionStopUnknownOutcome
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.models import SessionLoggingConfig
from tests.app_server.backend_contract.conftest import (
    BackendContractConnection,
    connect_backend_contract_host,
    connect_backend_contract_stdio_host,
    install_memory_root_agent,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import attach_test_app_server_session, build_test_app_server
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_session_start_returns_the_attached_public_session(
    backend_contract_session: AppServerSession,
) -> None:
    assert (
        backend_contract_session.state.session.id == backend_contract_session.session_id
    )


@pytest.mark.asyncio
async def test_session_read_returns_the_public_snapshot_and_event_watermark(
    backend_contract_connection: BackendContractConnection,
    backend_contract_session: AppServerSession,
) -> None:
    result = await backend_contract_connection.client.request(
        "session/read",
        SessionReadParams(session_id=backend_contract_session.session_id),
    )

    response = SessionReadResponse.model_validate(result)
    assert response.state.session.id == backend_contract_session.session_id
    assert "vibe" not in result
    assert "state" in result
    assert result["lastEventId"] == response.state.event_id
    assert "latestTurn" not in result["state"]
    assert isinstance(result["state"]["history"], list)
    assert isinstance(result["state"]["turns"], list)


@pytest.mark.asyncio
async def test_fork_attaches_the_public_child_session(
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[[str], httpx.Response],
    backend_contract_persistent_session: AppServerSession,
) -> None:
    backend_contract_mistral_api.mock(
        return_value=backend_contract_mistral_response("First answer")
    )
    _ = [
        event
        async for event in backend_contract_persistent_session.act(
            "First", client_message_id="user-1"
        )
    ]
    source_session_id = backend_contract_persistent_session.session_id

    fork = await backend_contract_persistent_session.resources.sessions.fork("user-1")

    assert fork.source_session_id == source_session_id
    assert backend_contract_persistent_session.session_id != source_session_id
    assert (
        backend_contract_persistent_session.state.session.parent_session_id
        == source_session_id
    )
    assert [
        entry.text
        for entry in backend_contract_persistent_session.history
        if isinstance(entry, PublicMessageEntry)
    ] == ["First", "First answer"]


@pytest.mark.asyncio
async def test_detached_fork_returns_a_public_child_without_rebinding_the_source(
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[[str], httpx.Response],
    backend_contract_persistent_session: AppServerSession,
) -> None:
    backend_contract_mistral_api.mock(
        side_effect=[
            backend_contract_mistral_response("First answer"),
            backend_contract_mistral_response("Second answer"),
        ]
    )
    _ = [
        event
        async for event in backend_contract_persistent_session.act(
            "First", client_message_id="user-1"
        )
    ]
    _ = [
        event
        async for event in backend_contract_persistent_session.act(
            "Second", client_message_id="user-2"
        )
    ]
    source_session_id = backend_contract_persistent_session.session_id

    fork = await backend_contract_persistent_session.resources.sessions.fork(
        "user-1", attach=False
    )

    assert backend_contract_persistent_session.session_id == source_session_id
    assert fork.source_session_id == source_session_id
    assert fork.state.session.id != source_session_id
    assert fork.state.session.parent_session_id == source_session_id
    assert [
        entry.text
        for entry in fork.state.history or []
        if isinstance(entry, PublicMessageEntry)
    ] == ["First", "First answer"]


@pytest.mark.asyncio
async def test_detached_fork_is_readable_and_resumable_from_a_new_host(
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[[str], httpx.Response],
    backend_contract_persistent_session: AppServerSession,
) -> None:
    backend_contract_mistral_api.mock(
        return_value=backend_contract_mistral_response("First answer")
    )
    _ = [
        event
        async for event in backend_contract_persistent_session.act(
            "First", client_message_id="user-1"
        )
    ]
    fork = await backend_contract_persistent_session.resources.sessions.fork(
        "user-1", attach=False
    )

    connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    resumed: AppServerSession | None = None
    try:
        stored = await connection.host.read_session(fork.state.session.id)
        resumed = await connection.host.resume_session(fork.state.session.id)
    finally:
        if resumed is None:
            await connection.host.close()
        else:
            await resumed.close()

    assert stored.session.id == fork.state.session.id
    assert (
        stored.session.parent_session_id
        == backend_contract_persistent_session.session_id
    )
    assert resumed is not None
    assert resumed.session_id == fork.state.session.id
    assert [
        entry.text for entry in resumed.history if isinstance(entry, PublicMessageEntry)
    ] == ["First", "First answer"]


@pytest.mark.asyncio
async def test_in_place_resume_replaces_the_public_snapshot_and_keeps_the_session_usable(
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[[str], httpx.Response],
    backend_contract_persistent_session: AppServerSession,
) -> None:
    backend_contract_mistral_api.mock(
        side_effect=[
            backend_contract_mistral_response("Before resume"),
            backend_contract_mistral_response("After resume"),
        ]
    )
    _ = [event async for event in backend_contract_persistent_session.act("first")]
    session_id = backend_contract_persistent_session.session_id

    await backend_contract_persistent_session.resume(session_id)
    _ = [event async for event in backend_contract_persistent_session.act("second")]

    assert backend_contract_persistent_session.state.session.id == session_id
    assert [
        entry.text
        for entry in backend_contract_persistent_session.history
        if isinstance(entry, PublicMessageEntry)
    ] == ["first", "Before resume", "second", "After resume"]


@pytest.mark.asyncio
async def test_continue_attaches_the_latest_persisted_session_and_keeps_it_usable(
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[[str], httpx.Response],
    backend_contract_persistent_connection: BackendContractConnection,
) -> None:
    backend_contract_mistral_api.mock(
        side_effect=[
            backend_contract_mistral_response("saved"),
            backend_contract_mistral_response("continued"),
        ]
    )
    session = await backend_contract_persistent_connection.host.open_session()
    try:
        _ = [event async for event in session.act("save this")]
        session_id = session.session_id
    finally:
        await session.close()

    continued_connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    continued = await continued_connection.host.continue_session()
    try:
        _ = [event async for event in continued.act("continue this")]
        assert continued.session_id == session_id
        assert continued.state.session.id == session_id
        assert [
            entry.text
            for entry in continued.history
            if isinstance(entry, PublicMessageEntry)
        ] == ["save this", "saved", "continue this", "continued"]
    finally:
        await continued.close()


@pytest.mark.asyncio
async def test_resume_attaches_the_requested_persisted_session_and_keeps_it_usable(
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[[str], httpx.Response],
    backend_contract_persistent_connection: BackendContractConnection,
) -> None:
    backend_contract_mistral_api.mock(
        side_effect=[
            backend_contract_mistral_response("saved"),
            backend_contract_mistral_response("resumed"),
        ]
    )
    session = await backend_contract_persistent_connection.host.open_session()
    try:
        _ = [event async for event in session.act("save this")]
        session_id = session.session_id
    finally:
        await session.close()

    resumed_connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    resumed = await resumed_connection.host.resume_session(session_id)
    try:
        _ = [event async for event in resumed.act("resume this")]
        assert resumed.session_id == session_id
        assert [
            entry.text
            for entry in resumed.history
            if isinstance(entry, PublicMessageEntry)
        ] == ["save this", "saved", "resume this", "resumed"]
    finally:
        await resumed.close()


# ---------------------------------------------------------------------------
# Live-history carrier contract (package C1e): session/history/list is
# answered by the process that owns the loop on both transports, the host
# route is retained for saved/non-attached ids, the not-found-to-host
# fallback is narrowed for attached ids, a dead-but-attached root answers
# the death receipt, and session/history/clear keeps its public behavior.
#
# As with the handoff contract above: in_process runs the legacy backend in
# this process; stdio runs the process-session backend whose root agent is
# served over a memory wire (install_memory_root_agent) -- no spawned child,
# because the B1 spawned-provider gate's tests/pty legs land with C5c. An
# in-process-only pass is not spawned-stdio parity proof. The loops here are
# FakeBackend-driven, so turns complete without a provider call.
# ---------------------------------------------------------------------------


def _history_list_params(
    session_id: str, *, cursor: str | None = None, limit: int = 200
) -> dict[str, Any]:
    return SessionHistoryListParams(
        session_id=session_id,
        page=PageRequest(cursor=cursor, limit=limit, direction="backward"),
    ).model_dump(mode="json", by_alias=True)


def _history_loop(**kwargs: Any) -> AgentLoop:
    """A FakeBackend loop whose two completions answer "one" and "two"."""
    backend = FakeBackend([
        [mock_llm_chunk(content="one")],
        [mock_llm_chunk(content="two")],
    ])
    return build_test_agent_loop(backend=backend, enable_streaming=True, **kwargs)


@pytest_asyncio.fixture(params=["in_process", "stdio"])
async def history_contract(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """One app-server per transport, both serving the same FakeBackend loop."""
    loop = _history_loop()
    client_transport, server_transport = memory_transport_pair()
    harness = None
    server = None
    if request.param == "stdio":
        install_memory_root_agent(monkeypatch, loop)
        harness = await create_harness_server(server_transport, transport_kind="stdio")
        client = AppServerClient(client_transport, run_peer=harness.serve)
    else:
        server = build_test_app_server(loop, server_transport)
        client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    try:
        yield client, session, request.param
    finally:
        await session.close()
        await client.close()
        if harness is not None:
            await harness.close()
        elif server is not None:
            await server.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_history_list_serves_the_live_projection_on_both_transports(
    history_contract: Any,
) -> None:
    """The attached session's live page is the agent's own projection.

    The contract session is not persisted, so on stdio a host route could
    only answer NOT_FOUND: a successful page here proves the root agent
    carrier answered, and it must agree entry-for-entry with the session's
    own projection.
    """
    client, session, _transport = history_contract
    _ = [event async for event in session.act("first", client_message_id="u1")]
    _ = [event async for event in session.act("second", client_message_id="u2")]
    session_id = session.session_id

    response = SessionHistoryListResponse.model_validate(
        await client.request("session/history/list", _history_list_params(session_id))
    )
    assert [
        entry.text for entry in response.items if isinstance(entry, PublicMessageEntry)
    ] == ["first", "one", "second", "two"]
    assert [entry.id for entry in response.items] == [
        entry.id for entry in session.history
    ]

    # Backward paging through the carrier keeps the same entry ids.
    latest = SessionHistoryListResponse.model_validate(
        await client.request(
            "session/history/list", _history_list_params(session_id, limit=2)
        )
    )
    assert [
        entry.text for entry in latest.items if isinstance(entry, PublicMessageEntry)
    ] == ["second", "two"]
    assert latest.next_cursor is not None
    earlier = SessionHistoryListResponse.model_validate(
        await client.request(
            "session/history/list",
            _history_list_params(session_id, cursor=latest.next_cursor, limit=2),
        )
    )
    assert [
        entry.text for entry in earlier.items if isinstance(entry, PublicMessageEntry)
    ] == ["first", "one"]


@pytest.mark.asyncio
async def test_history_list_keeps_the_plain_host_not_found_for_other_ids(
    backend_contract_transport_connection: BackendContractConnection,
    backend_contract_transport_session: AppServerSession,
) -> None:
    """The narrowed fallback applies to the attached id only.

    A saved/non-attached id still travels the host route, so an unknown one
    answers the host's own plain NOT_FOUND -- not the attached-id mismatch.
    """
    client = backend_contract_transport_connection.client
    with pytest.raises(AppServerResponseError) as exc_info:
        await client.request(
            "session/history/list", _history_list_params("no-such-session")
        )
    assert exc_info.value.error.code is ProtocolErrorCode.NOT_FOUND


@pytest.mark.asyncio
async def test_history_list_attached_id_mismatch_is_typed_not_host_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A not-found from the attached id's carrier is a typed mismatch.

    The carrier is made to refuse the attached id with NOT_FOUND: the server
    must answer the typed conflict naming the agent-owned projection, never
    silently serve the persisted host page for an attached id.
    """
    loop = _history_loop()
    install_memory_root_agent(monkeypatch, loop)
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(server_transport, transport_kind="stdio")
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(client)
    try:
        process = cast(Any, harness._server._root)._process

        async def refusing_history_list(*args: Any, **kwargs: Any) -> Any:
            raise RootAgentSessionNotFound("Session not found")

        monkeypatch.setattr(process, "history_list", refusing_history_list)
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "session/history/list", _history_list_params(session.session_id)
            )
        error = exc_info.value.error
        assert error.code is ProtocolErrorCode.CONFLICT
        assert "attached" in error.message
    finally:
        await session.close()
        await client.close()
        await harness.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_history_list_dead_but_attached_answers_the_death_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead-but-attached root routes to the carrier and gets the receipt.

    Death is not a not-found, so the host projection is never served for the
    attached id: the request fails with the typed death receipt.
    """
    loop = _history_loop()
    install_memory_root_agent(monkeypatch, loop)
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(server_transport, transport_kind="stdio")
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(client)
    process = None
    try:
        process = cast(Any, harness._server._root)._process
        process._death = RootDeath("the root agent process exited", exit_code=1)
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "session/history/list", _history_list_params(session.session_id)
            )
        error = exc_info.value.error
        assert error.code is ProtocolErrorCode.INTERNAL_ERROR
        assert "death-receipt" in error.message
    finally:
        if process is not None:
            process._death = None
        await session.close()
        await client.close()
        await harness.close()
        await loop.aclose()


@pytest_asyncio.fixture(params=["in_process", "stdio"])
async def history_persistent_contract(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    config_dir: Path,
    tmp_path: Path,
) -> Any:
    """One persistent app-server per transport, both serving the same loop."""
    save_dir = tmp_path / "sessions"
    config_file = config_dir / "config.toml"
    base = tomllib.loads(config_file.read_text(encoding="utf-8"))
    base["session_logging"] = {"enabled": True, "save_dir": str(save_dir)}
    config_file.write_text(tomli_w.dumps(base), encoding="utf-8")
    loop = _history_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(save_dir))
        )
    )
    client_transport, server_transport = memory_transport_pair()
    harness = None
    server = None
    if request.param == "stdio":
        install_memory_root_agent(monkeypatch, loop)
        harness = await create_harness_server(server_transport, transport_kind="stdio")
        client = AppServerClient(client_transport, run_peer=harness.serve)
    else:
        server = build_test_app_server(loop, server_transport)
        client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    try:
        yield client, session, request.param
    finally:
        await session.close()
        await client.close()
        if harness is not None:
            await harness.close()
        elif server is not None:
            await server.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_saved_session_keeps_the_host_route_and_reopen_cursor_compat(
    history_persistent_contract: Any,
) -> None:
    """Live and persisted pages agree on entry ids, so a cursor reopens.

    The live page is read through the agent carrier while the session is
    attached; after the session is saved, the same id is read through the
    host route on a fresh connection, and the cursors minted live page the
    persisted entries identically -- the live-to-saved reopen keeps cursor
    compatibility.
    """
    client, session, transport = history_persistent_contract
    _ = [event async for event in session.act("first", client_message_id="u1")]
    _ = [event async for event in session.act("second", client_message_id="u2")]
    session_id = session.session_id

    live_latest = SessionHistoryListResponse.model_validate(
        await client.request(
            "session/history/list", _history_list_params(session_id, limit=2)
        )
    )
    assert live_latest.next_cursor is not None
    live_earlier = SessionHistoryListResponse.model_validate(
        await client.request(
            "session/history/list",
            _history_list_params(session_id, cursor=live_latest.next_cursor, limit=2),
        )
    )
    await session.close()

    if transport == "stdio":
        second = await connect_backend_contract_stdio_host(
            session_options=SessionOptions(), capabilities=ClientCapabilities()
        )
    else:
        second = await connect_backend_contract_host(
            session_options=SessionOptions(), capabilities=ClientCapabilities()
        )
    try:
        # No session attached on the second connection: the saved id is
        # answered by the retained host route.
        saved_latest = SessionHistoryListResponse.model_validate(
            await second.client.request(
                "session/history/list", _history_list_params(session_id, limit=2)
            )
        )
        assert [entry.id for entry in saved_latest.items] == [
            entry.id for entry in live_latest.items
        ]
        saved_earlier = SessionHistoryListResponse.model_validate(
            await second.client.request(
                "session/history/list",
                _history_list_params(
                    session_id, cursor=live_latest.next_cursor, limit=2
                ),
            )
        )
        assert [entry.id for entry in saved_earlier.items] == [
            entry.id for entry in live_earlier.items
        ]
    finally:
        await second.host.close()


@pytest.mark.asyncio
async def test_history_clear_public_behavior_on_both_transports(
    history_contract: Any,
) -> None:
    """session/history/clear is carried and its receipt agrees with the list.

    After one completed turn the clear answers on both transports with the
    state it left behind; a subsequent live history page for that state's
    session id is served (on stdio through the agent carrier, never a
    not-found) and carries exactly the receipt's history entries.
    """
    client, session, _transport = history_contract
    _ = [event async for event in session.act("hello", client_message_id="u1")]
    session_id = session.session_id

    cleared = SessionHistoryClearResponse.model_validate(
        await client.request(
            "session/history/clear",
            SessionHistoryClearParams(session_id=session_id).model_dump(
                mode="json", by_alias=True
            ),
        )
    )
    new_id = cleared.state.session.id
    listed = SessionHistoryListResponse.model_validate(
        await client.request("session/history/list", _history_list_params(new_id))
    )
    assert [entry.id for entry in listed.items] == [
        entry.id for entry in cleared.state.history or []
    ]


# ---------------------------------------------------------------------------
# C2b contract (package C2b): the death receipt and the explicit
# session/resume respawn+restore with memorial state in the resume receipt,
# on both transports. As above: stdio runs the process-session backend whose
# root agent is served over a memory wire (install_memory_root_agent) — no
# spawned child, because the B1 spawned-provider gate's tests/pty legs land
# with C5c. An in-process-only pass is not spawned-stdio parity proof.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(params=["in_process", "stdio"])
async def c2b_resume_contract(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """One app-server per transport whose stdio root can be declared dead."""
    loop = _history_loop()
    client_transport, server_transport = memory_transport_pair()
    harness = None
    server = None
    if request.param == "stdio":
        install_memory_root_agent(monkeypatch, loop)
        harness = await create_harness_server(server_transport, transport_kind="stdio")
        client = AppServerClient(client_transport, run_peer=harness.serve)
    else:
        server = build_test_app_server(loop, server_transport)
        client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    try:
        yield client, session, request.param, harness
    finally:
        await session.close()
        await client.close()
        if harness is not None:
            await harness.close()
        elif server is not None:
            await server.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_resume_after_death_carries_receipt_and_memorial_on_both_transports(
    c2b_resume_contract: Any,
) -> None:
    """C2b: the resume receipt's death/memorial fields agree across transports.

    On stdio, after a confirmed root death a dead-but-attached request answers
    the typed death receipt, and the explicit session/resume respawns the
    root (memory wire), restores the durable session, and carries the retired
    generation's death receipt plus its memorial queue/shell state — never
    re-admitted for execution on the new generation. On in_process the same
    resume is the re-attach arm: no death receipt, empty memorial, and the
    session stays usable on both transports.
    """
    client, session, transport, harness = c2b_resume_contract
    _ = [event async for event in session.act("first", client_message_id="u1")]
    session_id = session.session_id
    backend = None
    if transport == "stdio":
        assert harness is not None
        backend = cast(Any, harness._server._root)
        # The doomed generation's coordinator-owned durable state.
        backend._turn_queue.enqueue(
            TurnEnqueueParams(
                session_id=session_id,
                entries=[
                    TurnUserInputEntry(
                        content=[SessionTextContentBlock(text="queued prompt")]
                    )
                ],
            )
        )
        backend.process._declare_death(
            "the root agent process exited",
            signature=RootFailureSignature.DEATH_RECEIPT,
        )
        # A dead-but-attached root answers the death receipt, never the host
        # projection, on the carried read.
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "session/history/list", _history_list_params(session_id)
            )
        assert "death-receipt" in str(exc_info.value)

    response = SessionResumeResponse.model_validate(
        await client.request(
            "session/resume",
            SessionResumeParams(session_id=session_id).model_dump(
                mode="json", by_alias=True
            ),
        )
    )
    # The durable session is restored under the same id on both transports.
    assert response.state.session.id == session_id
    if transport == "stdio":
        assert harness is not None
        assert response.death_receipt is not None
        assert "death-receipt" in response.death_receipt
        assert [(item.kind, item.text) for item in response.memorial] == [
            ("queued_turn", "queued prompt")
        ]
        new_backend = cast(Any, harness._server._root)
        assert new_backend is not backend
        # Memorial state is never re-admitted on the new generation.
        assert new_backend._turn_queue.state.items == []
        # The old generation is fenced: it answers the death receipt.
        assert backend is not None
        with pytest.raises(SessionBackendError) as fence_exc:
            backend.guard_request()
        assert "death-receipt" in str(fence_exc.value)
    else:
        assert response.death_receipt is None
        assert response.memorial == []

    # The resumed session stays usable. On the in_process leg a further turn
    # completes; on the stdio memory-wire leg the respawned root serves the
    # same loop object whose lifetime the retired generation's shutdown closed
    # (a production respawn builds a fresh loop in the new process), so
    # usability is asserted through the reads: the restored state and a live
    # ready runtime on the new generation.
    if transport == "stdio":
        resumed_read = SessionReadResponse.model_validate(
            await client.request(
                "session/read", SessionReadParams(session_id=session_id)
            )
        )
        assert resumed_read.state.session.id == session_id
        assert (
            resumed_read.state.turn_queue is None
            or not resumed_read.state.turn_queue.items
        )
        runtime_read = RuntimeReadResponse.model_validate(
            await client.request(
                "runtime/read", RuntimeReadParams(session_id=session_id)
            )
        )
        assert runtime_read.ready is True
    else:
        _ = [
            event async for event in session.act("after resume", client_message_id="u2")
        ]


# ---------------------------------------------------------------------------
# C2c contract (package C2c): the coordinator session/stop lifecycle on both
# transports -- the attached stop settles callbacks and queued work before
# the ack and then terminates the root, a stop against a settled session via
# a fresh connection acks, and a dead-but-undetected root answers the typed
# death receipt instead of hanging to the request timeout. Same harness shape
# as the contracts above: stdio runs the process-session backend whose root
# agent is served over a memory wire (install_memory_root_agent) -- no spawned
# child, because the B1 spawned-provider gate's tests/pty legs land with C5c.
# An in-process-only pass is not spawned-stdio parity proof.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(params=["in_process", "stdio"])
async def c2c_stop_contract(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    config_dir: Path,
    tmp_path: Path,
) -> Any:
    """One persistent app-server per transport whose stdio root can be declared dead."""
    save_dir = tmp_path / "sessions"
    config_file = config_dir / "config.toml"
    base = tomllib.loads(config_file.read_text(encoding="utf-8"))
    base["session_logging"] = {"enabled": True, "save_dir": str(save_dir)}
    config_file.write_text(tomli_w.dumps(base), encoding="utf-8")
    loop = _history_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(save_dir))
        )
    )
    client_transport, server_transport = memory_transport_pair()
    harness = None
    server = None
    if request.param == "stdio":
        install_memory_root_agent(monkeypatch, loop)
        harness = await create_harness_server(server_transport, transport_kind="stdio")
        client = AppServerClient(client_transport, run_peer=harness.serve)
    else:
        server = build_test_app_server(loop, server_transport)
        client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    try:
        yield client, session, request.param, harness, server
    finally:
        # The contract's stop was already answered and the server closes the
        # connection at that answer, so the teardown stop lands on the closed
        # channel and can only report its outcome as unknown.
        with suppress(SessionStopUnknownOutcome):
            await session.close()
        await client.close()
        if harness is not None:
            await harness.close()
        elif server is not None:
            await server.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_session_stop_settles_queued_work_then_acks_on_both_transports(
    c2c_stop_contract: Any,
) -> None:
    """C2c: the attached stop is answered by the coordinator on both
    transports. Queued work settles (the coordinator-owned queue pauses)
    BEFORE the ack, the root then terminates and persists ahead of the ack
    frame, and the server closes the attached runtime at the ack.
    """
    client, session, transport, harness, server = c2c_stop_contract
    session_id = session.session_id
    target = harness._server if harness is not None else server
    assert target is not None
    backend = cast(Any, target._root)
    # Queue two turns: the first promotes into the active turn, the second
    # stays queued, so the stop has queued work to settle before the ack.
    for text in ("first prompt", "queued prompt"):
        await client.request(
            "app_server/session/turn/enqueue",
            TurnEnqueueParams(
                session_id=session_id,
                entries=[
                    TurnUserInputEntry(content=[SessionTextContentBlock(text=text)])
                ],
            ).model_dump(mode="json", by_alias=True),
        )
    ack = SessionStopResponse.model_validate(
        await client.request("session/stop", SessionStopParams(session_id=session_id))
    )
    assert ack.closed is True
    assert target._root is None
    assert target._shutdown_complete is True
    if transport == "stdio":
        # The stop settled the queue and terminated the root before the ack.
        assert backend._turn_queue.paused is True
        assert backend._closed is True
        assert backend.process._closed is True
    else:
        assert backend.handler._closed is True


@pytest.mark.asyncio
async def test_settled_session_stop_acks_on_a_fresh_connection_both_transports(
    c2c_stop_contract: Any,
) -> None:
    """C2c: repeat-safe outside the closed connection -- a stop against the
    settled (saved) session via a fresh connection is an ack on both
    transports, and an unknown id stays NOT_FOUND.
    """
    client, session, transport, _harness, _server = c2c_stop_contract
    _ = [event async for event in session.act("first", client_message_id="u1")]
    session_id = session.session_id
    await session.close()

    def connect_second() -> Any:
        if transport == "stdio":
            return connect_backend_contract_stdio_host(
                session_options=SessionOptions(), capabilities=ClientCapabilities()
            )
        return connect_backend_contract_host(
            session_options=SessionOptions(), capabilities=ClientCapabilities()
        )

    # Any stop closes its connection at the ack, so each leg gets a fresh one.
    unknown = await connect_second()
    try:
        with pytest.raises(AppServerResponseError) as exc_info:
            await unknown.client.request(
                "session/stop", SessionStopParams(session_id="no-such-session")
            )
        assert exc_info.value.error.code is ProtocolErrorCode.NOT_FOUND
    finally:
        await unknown.host.close()

    second = await connect_second()
    try:
        ack = SessionStopResponse.model_validate(
            await second.client.request(
                "session/stop", SessionStopParams(session_id=session_id)
            )
        )
        assert ack.closed is True
    finally:
        await second.host.close()


@pytest.mark.asyncio
async def test_stop_against_a_dead_but_undetected_root_answers_the_death_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2c: the stop carrier re-checks liveness before anything crosses the
    wire, so a dead-but-undetected root is declared now and answers the typed
    death receipt rather than hanging to the request timeout.
    """
    loop = _history_loop()
    install_memory_root_agent(monkeypatch, loop)
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(server_transport, transport_kind="stdio")
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(client)
    try:
        backend = cast(Any, harness._server._root)
        process = backend.process
        assert process.death is None
        # The process is gone but the watcher has not fired: death is
        # still undetected when the stop arrives.
        monkeypatch.setattr(type(process._spawned.process), "poll", lambda self: 1)
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "session/stop", SessionStopParams(session_id=session.session_id)
            )
        assert "death-receipt" in str(exc_info.value)
        assert process.death is not None
    finally:
        # The death receipt settled the stop and the server closes the
        # connection at it, so the teardown stop lands on the closed channel
        # and can only report its outcome as unknown.
        with suppress(SessionStopUnknownOutcome):
            await session.close()
        await client.close()
        await harness.close()
        await loop.aclose()
