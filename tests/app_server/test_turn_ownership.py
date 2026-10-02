from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from chartreux.app_server._execution import (
    SessionExecution,
    SessionExecutionConflict,
    SessionExecutionKind,
)
from chartreux.app_server._turns import TurnController
from chartreux.app_server.models import TextContentBlock
from chartreux.app_server.protocol import TurnStartParams
from tests.conftest import build_test_agent_loop
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["rewind", "compaction", "reload"])
async def test_lifecycle_conflict_preserves_execution_ownership(operation: str) -> None:
    loop = build_test_agent_loop(backend=FakeBackend())
    execution = SessionExecution()
    controller = TurnController(
        loop, AsyncMock(), AsyncMock(), execution, Mock(), snapshot_state=Mock()
    )
    owner = execution.begin(SessionExecutionKind.LIFECYCLE, operation)
    before = list(loop.messages)
    with pytest.raises(SessionExecutionConflict):
        controller.start(
            TurnStartParams(
                session_id=loop.session_id, message=[TextContentBlock(text="rejected")]
            )
        )
    assert list(loop.messages) == before
    assert controller.active_turn is None
    assert execution.active is owner
    await loop.aclose()
