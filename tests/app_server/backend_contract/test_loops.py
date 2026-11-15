from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from pathlib import Path
import time
from typing import Any, cast

import httpx
import pytest
import respx

from chartreux.app_server._runtime import create_harness_server
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.models import (
    PublicMessageEntry,
    PublicNoticeEntry,
    ScheduledLoopFiredNoticeDetail,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ProtocolErrorCode,
    SessionOptions,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from tests.app_server.backend_contract.conftest import (
    BackendContractConnection,
    connect_backend_contract_host,
    install_memory_root_agent,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import attach_test_app_server_session
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_loop_resource_supports_crud(
    backend_contract_transport_session: AppServerSession,
) -> None:
    """*Prepare*: An attached session with no scheduled loops, on either transport.
    *Do*: Create, list, delete, and clear loops through the public resource.
    *Assert*: Every mutation returns the typed loop state shared by both backends.
    """
    # Prepare
    assert await backend_contract_transport_session.resources.loops.list() == []

    # Do
    first = await backend_contract_transport_session.resources.loops.create(
        "30s", "first"
    )
    second = await backend_contract_transport_session.resources.loops.create(
        "1m", "second"
    )
    listed = await backend_contract_transport_session.resources.loops.list()
    deleted = await backend_contract_transport_session.resources.loops.delete(first.id)
    cleared = await backend_contract_transport_session.resources.loops.clear()

    # Assert
    assert listed == [first, second]
    assert deleted == first
    assert cleared == 1
    assert await backend_contract_transport_session.resources.loops.list() == []


@pytest.mark.asyncio
async def test_loop_resource_returns_typed_validation_errors(
    backend_contract_transport_session: AppServerSession,
) -> None:
    """*Prepare*: An attached session accepting loop resource requests.
    *Do*: Create a loop with an invalid interval, then delete an unknown id.
    *Assert*: Both backends return the same typed refusals and change nothing.
    """
    # Prepare / Do
    with pytest.raises(AppServerResponseError) as exc_info:
        await backend_contract_transport_session.resources.loops.create(
            "invalid", "prompt"
        )

    # Assert
    assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
    assert "Invalid interval" in exc_info.value.error.message

    with pytest.raises(AppServerResponseError) as delete_info:
        await backend_contract_transport_session.resources.loops.delete("no-such-loop")

    assert delete_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
    assert "No scheduled loop with id" in delete_info.value.error.message
    assert await backend_contract_transport_session.resources.loops.list() == []


@pytest.mark.asyncio
async def test_loop_carrier_answers_from_the_process_that_holds_the_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stdio route crosses to the agent's own schedule, never a coordinator copy.

    The process-session backend holds no loop manager, so any of these four
    requests answered here would have to invent one -- a second schedule with
    its own fire times. Each public call is observed crossing to the owner, and
    the owner's reply is the state the public resource returns.
    """
    owned_loop = build_test_agent_loop(
        backend=FakeBackend([[mock_llm_chunk(content="one")]])
    )
    install_memory_root_agent(monkeypatch, owned_loop)
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(server_transport, transport_kind="stdio")
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(client)
    try:
        process = cast(Any, harness._server._root)._process
        crossings: list[str] = []

        def recorded(name: str, owner: Callable[..., Any]) -> Callable[..., Any]:
            async def crossed(*args: Any, **kwargs: Any) -> Any:
                crossings.append(name)
                return await owner(*args, **kwargs)

            return crossed

        for name in ("list_loops", "create_loop", "delete_loop", "clear_loops"):
            owner = getattr(process, name)
            monkeypatch.setattr(process, name, recorded(name, owner))

        created = await session.resources.loops.create("30s", "carried")
        listed = await session.resources.loops.list()
        deleted = await session.resources.loops.delete(created.id)
        cleared = await session.resources.loops.clear()
        remaining = await session.resources.loops.list()

        assert crossings == [
            "create_loop",
            "list_loops",
            "delete_loop",
            "clear_loops",
            "list_loops",
        ]
        assert listed == [created]
        assert created.prompt == "carried"
        assert deleted == created
        assert cleared == 0
        assert remaining == []
    finally:
        await session.close()
        await client.close()
        await harness.close()
        await owned_loop.aclose()


@pytest.mark.asyncio
async def test_loop_schedule_survives_resume_from_a_new_host(
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[[str], httpx.Response],
    backend_contract_persistent_connection: BackendContractConnection,
) -> None:
    """*Prepare*: A persisted session with one scheduled loop, then a fresh backend Host.
    *Do*: Resume the session and list its loops.
    *Assert*: The same schedule is restored for both session implementations.
    """
    # Prepare
    backend_contract_mistral_api.mock(
        return_value=backend_contract_mistral_response("stored")
    )
    session = await backend_contract_persistent_connection.host.open_session()
    loop = await session.resources.loops.create("30s", "persist me")
    _ = [event async for event in session.act("persist the session")]
    session_id = session.session_id
    await session.close()
    resumed_connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )

    # Do
    resumed = await resumed_connection.host.resume_session(session_id)
    try:
        restored = await resumed.resources.loops.list()
    finally:
        await resumed.close()

    # Assert
    assert restored == [loop]


@pytest.mark.asyncio
async def test_clear_history_preserves_loop_schedules(
    backend_contract_persistent_session: AppServerSession,
) -> None:
    """*Prepare*: A session with one scheduled loop.
    *Do*: Clear its history.
    *Assert*: The replacement session preserves the loop schedule.
    """
    # Prepare
    loop = await backend_contract_persistent_session.resources.loops.create(
        "30s", "keep me"
    )

    # Do
    await backend_contract_persistent_session.clear_history()
    after_clear = await backend_contract_persistent_session.resources.loops.list()

    # Assert
    assert after_clear == [loop]


@pytest.mark.asyncio
async def test_fork_preserves_loop_schedules(
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[[str], httpx.Response],
    backend_contract_persistent_session: AppServerSession,
) -> None:
    """*Prepare*: A persisted session with one scheduled loop.
    *Do*: Fork and attach to a copy of that session.
    *Assert*: The fork has the same loop schedule.
    """
    # Prepare
    backend_contract_mistral_api.mock(
        return_value=backend_contract_mistral_response("stored")
    )
    loop = await backend_contract_persistent_session.resources.loops.create(
        "30s", "keep me"
    )
    _ = [event async for event in backend_contract_persistent_session.act("persist")]

    # Do
    fork = await backend_contract_persistent_session.resources.sessions.fork()
    after_fork = await backend_contract_persistent_session.resources.loops.list()

    # Assert
    assert fork.source_session_id != fork.state.session.id
    assert after_fork == [loop]


@pytest.mark.asyncio
async def test_unified_due_loop_runs_as_an_unsolicited_turn(
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[[str], httpx.Response],
    backend_contract_persistent_connection: BackendContractConnection,
    tmp_path: Path,
) -> None:
    """*Prepare*: A persisted Unified session whose loop is already due.
    *Do*: Resume it and wait for the background scheduler.
    *Assert*: The turn runs, the notice follows its user message, and the loop advances.
    """
    # Prepare
    session = await backend_contract_persistent_connection.host.open_session()
    loop = await session.resources.loops.create("30s", "scheduled prompt")
    await session.inject_user_context("persist the session", as_message=True)
    session_id = session.session_id
    await session.close()
    schedule_path = _mark_legacy_loop_due(tmp_path, session_id)
    backend_contract_mistral_api.mock(
        return_value=backend_contract_mistral_response("scheduled response")
    )
    resumed_connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )

    # Do
    resumed = await resumed_connection.host.resume_session(session_id)
    try:
        async with asyncio.timeout(3):
            while (
                resumed.state.latest_turn is None
                or resumed.state.latest_turn.status != "completed"
            ):
                await asyncio.sleep(0.01)
        loops = await resumed.resources.loops.list()
    finally:
        await resumed.close()

    # Assert
    entries = resumed.history
    user_index = next(
        index
        for index, entry in enumerate(entries)
        if isinstance(entry, PublicMessageEntry) and entry.text == "scheduled prompt"
    )
    notice = entries[user_index + 1]
    assert isinstance(notice, PublicNoticeEntry)
    assert notice.message == f"Loop `{loop.id}` fired"
    assert isinstance(notice.detail, ScheduledLoopFiredNoticeDetail)
    assert notice.detail.loop_id == loop.id
    assert loops[0].next_fire_at > time.time()
    assert (
        json.loads(schedule_path.read_text(encoding="utf-8"))["loops"][0][
            "next_fire_at"
        ]
        > time.time()
    )


@pytest.mark.asyncio
async def test_headless_unified_resume_does_not_fire_due_loops(
    backend_contract_mistral_api: respx.Route,
    backend_contract_persistent_connection: BackendContractConnection,
    tmp_path: Path,
) -> None:
    """*Prepare*: A persisted Unified session with an overdue loop.
    *Do*: Resume the session in headless mode and allow scheduler time to elapse.
    *Assert*: No model request or scheduled turn is started.
    """
    # Prepare
    session = await backend_contract_persistent_connection.host.open_session()
    await session.resources.loops.create("30s", "must not run")
    await session.inject_user_context("persist the session", as_message=True)
    session_id = session.session_id
    await session.close()
    _mark_legacy_loop_due(tmp_path, session_id)
    resumed_connection = await connect_backend_contract_host(
        session_options=SessionOptions(headless=True), capabilities=ClientCapabilities()
    )

    # Do
    resumed = await resumed_connection.host.resume_session(session_id)
    try:
        await asyncio.sleep(0.15)
    finally:
        await resumed.close()

    # Assert
    assert backend_contract_mistral_api.call_count == 0
    assert resumed.state.latest_turn is None
    assert all(
        not isinstance(entry, PublicMessageEntry) or entry.text != "must not run"
        for entry in resumed.history
    )


def _mark_legacy_loop_due(tmp_path: Path, session_id: str) -> Path:
    for path in (tmp_path / "sessions").glob("session_*/meta.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("session_id") == session_id:
            payload["loops"][0]["next_fire_at"] = time.time() - 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            return path
    raise AssertionError(f"Could not find persisted session {session_id}")
