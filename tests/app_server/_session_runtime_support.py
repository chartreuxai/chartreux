"""Shared test-support surface for the app-server session suite (W10, stage 19c).

The definitions here are the shared module-level helper surface of
``tests/app_server/test_session.py``, relocated AST-verbatim in the 19c
phase-(i) helper-relocation commit: the autouse XDG-state isolation
fixture ``_isolate_recovery_registry`` (it covers that file's whole test
population; the file re-registers it by import, and every 19c phase-(ii)
destination module must re-register it the same way so its moved
population keeps the isolation, per the plan §1.4 fixture/mark
preservation rule), the wire-request builder ``_wire_resume_request``,
the question tool-call builder ``_question_tool_call``, the event-stream
helpers (``_consume``, ``_next_event``), and the reconnectable-session
factory ``_create_reconnectable_session`` (borderline: four reconnect
users plus the close test, over-moved per the asymmetry rule). No test
bodies live here; the test population stays in ``test_session.py`` until
the phase-(ii) concern moves. Importer: ``test_session.py`` itself (the
seam enumeration found zero external importers of that file).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path

import pytest

from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    ClientCapabilities,
    ClientInfo,
    SessionOptions,
    SessionResumeParams,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.llm_models import FunctionCall, ToolCall
from tests.stubs.app_server import build_test_app_server


@pytest.fixture(autouse=True)
def _isolate_recovery_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def _wire_resume_request(session_id: str) -> dict:
    return SessionResumeParams(session_id=session_id).model_dump(
        mode="json", by_alias=True
    )


def _question_tool_call() -> ToolCall:
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
                    }
                ]
            }),
        ),
    )


async def _consume(events):
    return [event async for event in events]


async def _next_event[EventT](
    events: AsyncIterator[object], event_type: type[EventT]
) -> EventT:
    async for event in events:
        if isinstance(event, event_type):
            return event
    raise AssertionError(f"Event stream ended before {event_type.__name__}")


async def _create_reconnectable_session(
    agent_loop,
    *,
    reconnect_gate: asyncio.Event | None = None,
    session_options: SessionOptions | None = None,
    resume_params: list[SessionResumeParams] | None = None,
) -> AppServerSession:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    if resume_params is not None:
        original_resume = server._session_backend_host.resume

        async def record_resume(params: SessionResumeParams):
            resume_params.append(params)
            return await original_resume(params)

        server._session_backend_host.resume = record_resume  # type: ignore[method-assign]

    def make_client(
        next_client_transport, next_server_transport, *, wait_for_gate: bool
    ) -> AppServerClient:
        async def serve_connection() -> None:
            if wait_for_gate and reconnect_gate is not None:
                await reconnect_gate.wait()
            await server.serve_connection(
                next_server_transport, close_on_disconnect=False
            )

        return AppServerClient(next_client_transport, run_peer=serve_connection)

    def reconnect() -> AppServerClient:
        return make_client(*memory_transport_pair(), wait_for_gate=True)

    return await AppServerSession.start(
        make_client(client_transport, server_transport, wait_for_gate=False),
        client_info=ClientInfo(name="reconnect-client", version="1"),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
        session_options=session_options,
        client_factory=reconnect,
    )
