"""Parent auto-compaction firing while a background subagent run is active.

The parent's compaction boundary must not disturb the running child or its
result: the run completes, the result stays deliverable, and the completion
notification lands after the boundary by construction.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from chartreux.app_server.models import PublicCheckpointEntry
from chartreux.app_server.server import AppServer
from chartreux.core.llm_models import Role
from tests.app_server.test_subagents import (
    GatedSequenceBackend,
    _config,
    _consume,
    _task_call,
    _wait_for_turn_completion,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_parent_auto_compact_during_background_run_keeps_result_deliverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = GatedSequenceBackend([[mock_llm_chunk(content="child finished")]])
    child_started, child_release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent_backend = FakeBackend([
        # Turn 1: launch the background subagent, then acknowledge it.
        [mock_llm_chunk(content="", tool_calls=[_task_call(background=True)])],
        [mock_llm_chunk(content="parent acknowledged the launch")],
        # Turn 2: auto-compact fires first (summary), then the turn itself.
        [mock_llm_chunk(content="<summary>parent compacted mid-run</summary>")],
        [mock_llm_chunk(content="parent continued after compaction")],
        # Notification turn after the child completes.
        [mock_llm_chunk(content="notification acknowledged")],
    ])
    parent = build_test_agent_loop(
        config=_config(), backend=parent_backend, enable_streaming=True
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)

    try:
        # Turn 1: launch the background subagent and let the parent go idle.
        await _consume(session.act("launch background work"))
        await asyncio.wait_for(child_started.wait(), timeout=5)
        registry = legacy_backend(server).children
        summary = (await registry.check_agents())[0]
        assert summary.availability.value == "running"
        assert summary.current_run_id is not None
        agent_id = summary.agent_id
        run_id = summary.current_run_id

        # Force the parent's auto-compact to fire mid-run: drive the parent
        # loop past its auto-compact threshold and start another turn while
        # the child is still running.
        parent.stats.context_tokens = 500_000
        await _consume(session.act("continue while the child works"))
        assert any(
            isinstance(entry, PublicCheckpointEntry) and entry.kind == "compaction"
            for entry in session.history
        )
        # The compaction boundary is real: the post-compaction model request
        # starts from the compacted context.
        assert parent_backend.requests_messages[3][1].context_boundary == "compaction"
        # The child run survived the parent's compaction untouched.
        assert (await registry.check_agents())[0].availability.value == "running"

        # The subagent completes after the parent's compaction boundary and its
        # accumulated result is intact and deliverable.
        child_release.set()
        result = await asyncio.wait_for(
            registry.wait_for_agent(agent_id, run_id), timeout=5
        )
        assert result.completed
        assert result.response == "child finished"
        assert await registry.get_agent_result(agent_id, run_id) == result
        await _wait_for_turn_completion(session)

        # The notification lands after the boundary by construction.
        messages = list(parent.messages)
        boundary_index = next(
            index
            for index, message in enumerate(messages)
            if message.context_boundary == "compaction"
        )
        notification_index = next(
            index
            for index, message in enumerate(messages)
            if message.role is Role.user
            and f"Background agent {agent_id}" in str(message.content)
        )
        assert notification_index > boundary_index
        assert messages[-1].role is Role.assistant
        assert messages[-1].content == "notification acknowledged"

        # The notification turn's model request carries the compacted context
        # with the notification after the boundary — nothing pre-boundary leaked
        # back in front of it.
        notification_request = parent_backend.requests_messages[-1]
        boundary_positions = [
            index
            for index, message in enumerate(notification_request)
            if message.context_boundary == "compaction"
        ]
        notification_positions = [
            index
            for index, message in enumerate(notification_request)
            if f"Background agent {agent_id}" in str(message.content)
        ]
        assert boundary_positions and notification_positions
        assert max(boundary_positions) < min(notification_positions)
    finally:
        child_release.set()
        await session.close()
