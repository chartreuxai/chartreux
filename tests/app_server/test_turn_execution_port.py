"""The turn controller drives turn execution through the agent-side port.

Turn ordering and callbacks stay where they are; this only proves the app server
reaches the loop through the port, so a process-backed adapter can replace the
in-process one without touching the controller.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from chartreux.app_server.client import AppServerClient
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
from chartreux.core.agent_loop._in_process_execution import InProcessTurnExecution
from chartreux.core.events import BaseEvent
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import attach_test_app_server_session, build_test_app_server
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_turn_controller_runs_turns_through_the_execution_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[ExecutionTurnRequest] = []
    original = InProcessTurnExecution.run_turn

    async def recording_run_turn(
        self: InProcessTurnExecution, request: ExecutionTurnRequest
    ) -> AsyncIterator[BaseEvent]:
        requests.append(request)
        async for event in original(self, request):
            yield event

    monkeypatch.setattr(InProcessTurnExecution, "run_turn", recording_run_turn)

    agent_loop = build_test_agent_loop(
        backend=FakeBackend([[mock_llm_chunk(content="done")]])
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    session = await attach_test_app_server_session(
        AppServerClient(client_transport, run_peer=server.serve)
    )
    try:
        async for _ in session.act("hello"):
            pass
    finally:
        await session.close()
        await agent_loop.aclose()

    assert [request.message for request in requests] == ["hello"]
    assert requests[0].turn_options is not None
