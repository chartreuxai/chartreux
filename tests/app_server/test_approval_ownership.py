"""Approval/callback decision state lives agent-side; the coordinator relays (WP3 [3C-2]).

The root loop's request broker owns the callback records (futures, resolution
locks, expiry timers, terminal dispositions); ``TurnController._callbacks`` is
a live relay view of that broker, and answers/expiry/cancellations cross the
execution port into agent-side arbitration. These tests pin that split:
the coordinator holds no decision state of its own, the unattended-expiry
timer runs agent-side under the controller's clock/wakeup seams, and a
manifest-routed callback is still settled by the agent-side broker.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from pathlib import Path
import time
from typing import Any

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory, RootOpenRequest
from chartreux.app_server._turns import TurnController
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.models import (
    AnsweredCallbackState,
    ExpiredCallbackState,
    PublicCallbackEntry,
    UserAnswer,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import AppServerResponseError, ProtocolErrorCode
from chartreux.app_server.registry_manifest import (
    AgentRoute,
    CallbackTarget,
    RegistryManifest,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.models import UserQuestionsConfig
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.questions import UnansweredQuestions
from tests.app_server._session_runtime_support import (  # noqa: F401 - autouse fixture registered via module namespace
    _consume,
    _isolate_recovery_registry,
    _next_event,
    _question_tool_call,
)
from tests.app_server._subagent_test_support import _GatedWakeup, _ManualClock
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    create_legacy_app_server,
    legacy_backend,
)
from tests.stubs.fake_backend import FakeBackend


def _question_answer() -> UserInputCallbackOutput:
    return UserInputCallbackOutput(
        result=UserQuestionResult(
            answers=[UserAnswer(question="Ship it?", answer="Yes")]
        )
    )


def _defaultable_question_call() -> ToolCall:
    return ToolCall(
        id="question-1",
        index=0,
        function=FunctionCall(
            name="ask_user_question",
            arguments=json.dumps({
                "questions": [
                    {
                        "question": "Ship it?",
                        "options": [{"label": "Yes"}, {"label": "No"}],
                        "kind": "preference",
                        "unattended_default": {
                            "action": "defer_affected_work",
                            "description": "Defer the release until the user answers.",
                        },
                    }
                ]
            }),
        ),
    )


async def _open_question_session(
    *, tool_call: ToolCall | None = None, config: Any = None
) -> tuple[Any, TurnController, AgentLoop, Any]:
    """Attach a session whose first turn opens one ``ask_user_question``."""
    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[tool_call or _question_tool_call()])],
        [mock_llm_chunk(content="Done")],
    ])
    agent_loop = build_test_agent_loop(
        config=config or build_test_vibe_config(enabled_tools=["ask_user_question"]),
        backend=backend,
        enable_streaming=True,
        user_input_capability=True,
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    session = await attach_test_app_server_session(
        AppServerClient(client_transport, run_peer=server.serve)
    )
    return session, legacy_backend(server).session.turns, agent_loop, server


async def _wait_until(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for the agent-side state")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_approval_is_decided_agent_side_and_the_coordinator_only_relays() -> None:
    session, turns, agent_loop, _server = await _open_question_session()
    events = session.act("read todos")
    try:
        # The coordinator's callback view IS the agent-side broker's record
        # dict: a property, not instance state.
        assert isinstance(type(turns)._callbacks, property)
        assert "_callbacks" not in vars(turns)
        assert turns._callbacks is agent_loop.request_broker.records()

        requested = await _next_event(events, CallbackRequested)
        callback_id = requested.callback.callback_id
        record = turns._callbacks[callback_id]
        assert record is agent_loop.request_broker.records()[callback_id]
        assert not record.core_resolved
        assert record.resolution is None

        # Answering relays through the execution port; the broker record
        # carries the terminal disposition and settles the waiter agent-side.
        answer = _question_answer()
        await session.respond_to_callback(callback_id, answer)

        assert record.core_resolved is True
        assert record.resolution == answer.result
        assert record.future.result() == answer.result
        entry = next(
            item
            for item in session.history
            if isinstance(item, PublicCallbackEntry) and item.callback_id == callback_id
        )
        assert isinstance(entry.state, AnsweredCallbackState)
        # A late, different answer is arbitrated agent-side as a conflict.
        other = UserInputCallbackOutput(
            result=UserQuestionResult(
                answers=[UserAnswer(question="Ship it?", answer="No")]
            )
        )
        with pytest.raises(AppServerResponseError) as conflict:
            await session.respond_to_callback(callback_id, other)
        assert conflict.value.error.code is ProtocolErrorCode.CONFLICT
    finally:
        with suppress(Exception):
            await asyncio.wait_for(_consume(events), timeout=2)
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_unattended_expiry_is_armed_and_fired_agent_side() -> None:
    config = build_test_vibe_config(enabled_tools=["ask_user_question"]).model_copy(
        update={
            "user_questions": UserQuestionsConfig(
                unattended_timeout_enabled=True,
                default_timeout_seconds=60.0,
                max_timeout_seconds=120.0,
            )
        }
    )
    session, turns, agent_loop, _server = await _open_question_session(
        tool_call=_defaultable_question_call(), config=config
    )
    # The production timer seams: the controller's clock/wakeup are read by
    # the agent-side broker through the bound getters.
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    turns._clock = clock
    turns._wakeup = wakeup
    events = session.act("ask me")
    try:
        requested = await _next_event(events, CallbackRequested)
        callback_id = requested.callback.callback_id
        record = turns._callbacks[callback_id]
        assert record.expiry is not None

        await asyncio.wait_for(wakeup.called.wait(), timeout=5)
        assert wakeup.calls == [60.0]

        # Advance past the deadline and let the agent-side timer fire.
        clock.advance(120.0)
        for gate in wakeup.gates:
            gate.set()
        await _wait_until(lambda: record.expired)

        # Expiry never approves anything: the waiter resolves with the
        # unanswered disposition and exactly one deferred record is kept.
        result = record.future.result()
        assert isinstance(result, UnansweredQuestions)
        deferred = agent_loop.deferred_questions.records()
        assert len(deferred) == 1
        assert deferred[0].request_id == callback_id
        entry = next(
            item
            for item in session.history
            if isinstance(item, PublicCallbackEntry) and item.callback_id == callback_id
        )
        assert isinstance(entry.state, ExpiredCallbackState)

        # A late answer is refused with the expired disposition, decided
        # agent-side under the record's resolution lock.
        with pytest.raises(AppServerResponseError) as late:
            await session.respond_to_callback(callback_id, _question_answer())
        assert late.value.error.code is ProtocolErrorCode.CALLBACK_CLOSED
        assert "expired" in late.value.error.message
    finally:
        with suppress(Exception):
            await asyncio.wait_for(_consume(events), timeout=2)
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_manifest_routed_callback_is_settled_by_the_agent_side_broker(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_question_tool_call()])],
        [mock_llm_chunk(content="Done")],
    ])
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["ask_user_question"]),
        backend=backend,
        enable_streaming=True,
    )
    session_dir = tmp_path / "sessions" / "root"
    session_dir.mkdir(parents=True)
    manifest = RegistryManifest(session_dir)
    manifest.initialize()
    runtime_factory = AgentRuntimeFactory()
    runtime_factory.registry_manifests[agent_loop] = manifest

    async def open_root(request: RootOpenRequest) -> AgentLoop:
        return agent_loop

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(
        server_transport, open_root=open_root, runtime_factory=runtime_factory
    )
    session = await attach_test_app_server_session(
        AppServerClient(client_transport, run_peer=server.serve)
    )
    turns = legacy_backend(server).session.turns
    events = session.act("read todos")
    try:
        requested = await _next_event(events, CallbackRequested)
        callback_id = requested.callback.callback_id

        # Route the open callback through the registry manifest and deliver:
        # the coordinator records the manifest-row delivery.
        route = AgentRoute(connection_id="connection-1", incarnation_id="inc-1")
        manifest.register_agent(agent_loop.session_id, route)
        manifest.route_callback(callback_id, agent_loop.session_id)
        entry = next(
            item
            for item in session.history
            if isinstance(item, PublicCallbackEntry) and item.callback_id == callback_id
        )
        await server._deliver_callback(entry)
        delivery = next(
            item
            for item in server._callback_requests.values()
            if item.callback_id == callback_id and item.target is not None
        )
        assert delivery.target == CallbackTarget(
            agent_id=agent_loop.session_id, route=route
        )

        # The answer still settles the agent-side broker record.
        answer = _question_answer()
        await session.respond_to_callback(callback_id, answer)
        record = agent_loop.request_broker.records()[callback_id]
        assert record is turns._callbacks[callback_id]
        assert record.core_resolved is True
        assert record.resolution == answer.result
        assert record.future.result() == answer.result
        settled = next(
            item
            for item in session.history
            if isinstance(item, PublicCallbackEntry) and item.callback_id == callback_id
        )
        assert isinstance(settled.state, AnsweredCallbackState)
    finally:
        with suppress(Exception):
            await asyncio.wait_for(_consume(events), timeout=2)
        await session.close()
        await agent_loop.aclose()
