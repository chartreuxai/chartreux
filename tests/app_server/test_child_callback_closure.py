"""End-to-end child-callback closure over the parent app-server client.

Q1 (`ask_user_question` execution-denied for bounded children) retired the
only end-to-end child-callback delivery assertion
(``test_subagents_assembly_drain.py::test_child_callbacks_round_trip_using_child_session_id``).
This module replaces that delivery coverage and adds the Q2b closure wire.
The child callback is injected through the callback seam — the child
TurnController's broker request handler — because after Q1 no shipped
producer can mint a child question. The channel is defense-in-depth for
pre-Q1 callbacks still open in a running session and for future
child-askable callback kinds.

Child callbacks are delivered to the parent client (``callback/call``) but
their history never enters the root projection. Closing one must therefore
reach the parent through ``callback/childClosed``; the client matches the
closure against its recorded owner, so a duplicate, a late, or a foreign
closure can never dismiss a newer callback.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

import pytest

from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import (
    AppServerEvent,
    CallbackRequested,
    ChildCallbackClosed,
    SessionSnapshot,
)
from chartreux.app_server.models import (
    UserAnswer,
    UserInputCallbackOutput,
    UserQuestion,
    UserQuestionRequest,
    UserQuestionResult,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ChildCallbackClosedParams,
    ClientCapabilities,
    ClientInfo,
    ProtocolErrorCode,
)
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.events import UserInputOrigin, UserInputRequestEvent
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from chartreux.questions import QuestionChoice
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    _background_result,
    _config,
)
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import build_test_app_server, legacy_backend


async def _start_session(agent_loop) -> tuple[AppServerSession, AppServer]:
    """Start one app-server session whose client can reconnect in place."""
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)

    def make_client(next_client_transport, next_server_transport) -> AppServerClient:
        async def serve_connection() -> None:
            await server.serve_connection(
                next_server_transport, close_on_disconnect=False
            )

        return AppServerClient(next_client_transport, run_peer=serve_connection)

    def reconnect() -> AppServerClient:
        return make_client(*memory_transport_pair())

    session = await AppServerSession.start(
        make_client(client_transport, server_transport),
        client_info=ClientInfo(name="child-closure-client", version="1"),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
        client_factory=reconnect,
    )
    return session, server


async def _next_event[EventT](
    events: AsyncIterator[AppServerEvent], event_type: type[EventT]
) -> EventT:
    async for event in events:
        if isinstance(event, event_type):
            return event
    raise AssertionError(f"Event stream ended before {event_type.__name__}")


async def _events_until_idle(
    events: AsyncIterator[AppServerEvent], *, idle_timeout: float = 0.25
) -> list[AppServerEvent]:
    """Collect queued events until the stream has been quiet for a moment."""
    collected: list[AppServerEvent] = []
    iterator = events.__aiter__()

    async def next_event() -> AppServerEvent:
        return await anext(iterator)

    while True:
        pending = asyncio.create_task(next_event())
        try:
            collected.append(
                await asyncio.wait_for(asyncio.shield(pending), idle_timeout)
            )
        except TimeoutError:
            pending.cancel()
            with suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending
            return collected
        except StopAsyncIteration:
            return collected


def _child_question(request_id: str) -> UserInputRequestEvent:
    return UserInputRequestEvent(
        request_id=request_id,
        args=UserQuestionRequest(
            questions=[
                UserQuestion(
                    question="Ship it?",
                    options=[QuestionChoice(label="Yes"), QuestionChoice(label="No")],
                )
            ]
        ),
        tool_call_id=f"{request_id}-effect",
        origin=UserInputOrigin.MODEL_QUESTION,
    )


async def _launch_child(
    monkeypatch: pytest.MonkeyPatch, session: AppServerSession, server: AppServer
) -> tuple[BlockingBackend, Any]:
    child_backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    registry = legacy_backend(server).children
    launch = await _background_result(
        registry,
        TaskArgs(task="child work", retain=True),
        InvokeContext(tool_call_id="child-call", session_id=session.session_id),
    )
    assert launch.agent_id is not None
    assert launch.run_id is not None
    record = registry._agent_records[launch.agent_id]
    await asyncio.wait_for(child_backend.started.wait(), 2)
    return child_backend, record


async def _inject_child_callback(
    session: AppServerSession, record: Any, request_id: str
) -> tuple[CallbackRequested, asyncio.Task[None]]:
    """Open one child callback through the broker/callback seam."""
    pending = asyncio.create_task(
        record.runtime.turns._handle_core_request(_child_question(request_id))
    )
    callback = await asyncio.wait_for(
        _next_event(session.events(), CallbackRequested), 2
    )
    assert callback.callback.session_id == record.session_id
    assert callback.callback.callback_id == request_id
    return callback, pending


async def _close_child(
    session: AppServerSession, record: Any, pending: asyncio.Task[None]
) -> None:
    response = await session.cancel_agent(record.agent_id, record.latest_run_id)
    assert response.outcome.value == "stop_requested"
    with suppress(Exception):
        async with asyncio.timeout(2):
            await pending


@pytest.mark.asyncio
async def test_child_callback_closure_reaches_the_parent_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = build_test_agent_loop(config=_config())
    session, server = await _start_session(parent)
    try:
        _, record = await _launch_child(monkeypatch, session, server)
        callback, pending = await _inject_child_callback(session, record, "child-q-1")
        # The callback was delivered end-to-end and owned by the child session
        # (the coverage Q1 retired from the assembly-drain suite).
        assert session._callback_sessions == {
            callback.callback.callback_id: record.session_id
        }

        await _close_child(session, record, pending)
        closure = await asyncio.wait_for(
            _next_event(session.events(), ChildCallbackClosed), 2
        )
        assert closure.params.callback_id == callback.callback.callback_id
        assert closure.params.session_id == record.session_id

        # The client forgot the dead callback, so a later answer fails locally
        # on the typed path instead of being routed at a dead child.
        assert session._callback_sessions == {}
        with pytest.raises(AppServerResponseError) as exc_info:
            await session.respond_to_callback(
                callback.callback.callback_id, _child_answer()
            )
        assert exc_info.value.error.code is ProtocolErrorCode.NOT_FOUND

        # A duplicate closure is never re-emitted by the producer: the child
        # projector no longer holds an open entry to close.
        await record.runtime.turns._cancel_callbacks("duplicate")
        trailing = await _events_until_idle(session.events())
        assert not [
            event
            for event in trailing
            if isinstance(event, ChildCallbackClosed)
            and event.params.callback_id == callback.callback.callback_id
        ]
    finally:
        await session.close()
        await parent.aclose()


def _child_answer() -> UserInputCallbackOutput:
    return UserInputCallbackOutput(
        result=UserQuestionResult(
            answers=[UserAnswer(question="Ship it?", answer="Yes")]
        )
    )


@pytest.mark.asyncio
async def test_duplicate_child_closure_is_harmless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = build_test_agent_loop(config=_config())
    session, server = await _start_session(parent)
    try:
        _, record = await _launch_child(monkeypatch, session, server)
        callback, pending = await _inject_child_callback(session, record, "child-q-2")
        await _close_child(session, record, pending)
        first = await asyncio.wait_for(
            _next_event(session.events(), ChildCallbackClosed), 2
        )

        duplicate = ChildCallbackClosedParams(
            session_id=first.params.session_id,
            callback_id=first.params.callback_id,
            reason=first.params.reason,
        )
        await server.notify(ChildCallbackClosedParams.NOTIFICATION_METHOD, duplicate)
        second = await asyncio.wait_for(
            _next_event(session.events(), ChildCallbackClosed), 2
        )
        assert second.params == duplicate
        assert session._callback_sessions == {}
        assert callback.callback.callback_id == duplicate.callback_id
    finally:
        await session.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_late_child_closure_never_touches_a_newer_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = build_test_agent_loop(config=_config())
    session, server = await _start_session(parent)
    try:
        _, first_record = await _launch_child(monkeypatch, session, server)
        first, first_pending = await _inject_child_callback(
            session, first_record, "child-q-3a"
        )
        await _close_child(session, first_record, first_pending)
        first_closure = await asyncio.wait_for(
            _next_event(session.events(), ChildCallbackClosed), 2
        )

        # A newer child callback is now active.
        _, second_record = await _launch_child(monkeypatch, session, server)
        second, second_pending = await _inject_child_callback(
            session, second_record, "child-q-3b"
        )
        assert session._callback_sessions == {
            second.callback.callback_id: second_record.session_id
        }

        # A late duplicate of the FIRST closure must carry its own identity and
        # leave the newer callback's ownership untouched.
        await server.notify(
            ChildCallbackClosedParams.NOTIFICATION_METHOD,
            ChildCallbackClosedParams(
                session_id=first_closure.params.session_id,
                callback_id=first_closure.params.callback_id,
                reason=first_closure.params.reason,
            ),
        )
        late = await asyncio.wait_for(
            _next_event(session.events(), ChildCallbackClosed), 2
        )
        assert late.params.callback_id == first.callback.callback_id
        assert late.params.callback_id != second.callback.callback_id
        assert session._callback_sessions == {
            second.callback.callback_id: second_record.session_id
        }

        await _close_child(session, second_record, second_pending)
        second_closure = await asyncio.wait_for(
            _next_event(session.events(), ChildCallbackClosed), 2
        )
        assert second_closure.params.callback_id == second.callback.callback_id
        assert session._callback_sessions == {}
    finally:
        await session.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_unrelated_owner_closure_is_ignored_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = build_test_agent_loop(config=_config())
    session, server = await _start_session(parent)
    try:
        _, record = await _launch_child(monkeypatch, session, server)
        callback, pending = await _inject_child_callback(session, record, "child-q-4")

        # Same callback id, WRONG owner: identity matching must keep the known
        # mapping, so a foreign session can never close this callback.
        await server.notify(
            ChildCallbackClosedParams.NOTIFICATION_METHOD,
            ChildCallbackClosedParams(
                session_id="child-session-unknown",
                callback_id=callback.callback.callback_id,
                reason="foreign",
            ),
        )
        foreign = await asyncio.wait_for(
            _next_event(session.events(), ChildCallbackClosed), 2
        )
        assert foreign.params.session_id == "child-session-unknown"
        assert session._callback_sessions == {
            callback.callback.callback_id: record.session_id
        }

        # A wholly unknown identity is ignored just as safely, and the live
        # stream keeps working afterwards.
        await server.notify(
            ChildCallbackClosedParams.NOTIFICATION_METHOD,
            ChildCallbackClosedParams(
                session_id="child-session-unknown",
                callback_id="callback-unknown",
                reason="unknown",
            ),
        )
        unknown = await asyncio.wait_for(
            _next_event(session.events(), ChildCallbackClosed), 2
        )
        assert unknown.params.callback_id == "callback-unknown"
        assert session._callback_sessions == {
            callback.callback.callback_id: record.session_id
        }

        await _close_child(session, record, pending)
        closure = await asyncio.wait_for(
            _next_event(session.events(), ChildCallbackClosed), 2
        )
        assert closure.params.callback_id == callback.callback.callback_id
        assert session._callback_sessions == {}
    finally:
        await session.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_reconnect_does_not_re_request_a_closed_child_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = build_test_agent_loop(config=_config())
    session, server = await _start_session(parent)
    try:
        _, record = await _launch_child(monkeypatch, session, server)
        callback, pending = await _inject_child_callback(session, record, "child-q-5")
        closed_id = callback.callback.callback_id
        await _close_child(session, record, pending)
        closure = await asyncio.wait_for(
            _next_event(session.events(), ChildCallbackClosed), 2
        )
        assert closure.params.callback_id == closed_id
        # Redelivery iterates exactly this set, so a closed callback cannot be
        # re-requested: the child runtime no longer holds it.
        assert not [
            open_callback
            for open_callback in legacy_backend(server).open_callbacks()
            if open_callback.callback_id == closed_id
        ]

        disconnected_client = session._connection.current
        assert disconnected_client is not None
        await disconnected_client.close()

        snapshot = await asyncio.wait_for(
            _next_event(session.events(), SessionSnapshot), 2
        )
        assert snapshot.state.session.id == session.session_id
        assert session._connection.current is not disconnected_client

        # Root-only redelivery stays intact: the closed child callback is not
        # requested again, and no closure is replayed.
        trailing = await _events_until_idle(session.events())
        assert not [
            event
            for event in trailing
            if isinstance(event, CallbackRequested)
            and event.callback.callback_id == closed_id
        ]
        assert not [
            event
            for event in trailing
            if isinstance(event, ChildCallbackClosed)
            and event.params.callback_id == closed_id
        ]
    finally:
        await session.close()
        await parent.aclose()
