from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server.models import AgentStatsSnapshot
from chartreux.cli.textual_ui.widgets.compact import CompactMessage
from chartreux.cli.textual_ui.widgets.messages import UserMessage
from chartreux.cli.textual_ui.widgets.session_status_line import SessionStatusLine
from tests.conftest import build_test_chartreux_app

_SPENT_TOKENS = 120_000
_CONTEXT_WINDOW = 200_000


class TestCompactMessage:
    def test_get_content_after_compaction(self) -> None:
        message = CompactMessage()

        message.set_complete()

        assert message.get_content() == "Compaction completed."


@pytest.mark.asyncio
async def test_completed_compaction_keeps_visible_history() -> None:
    app = build_test_chartreux_app()

    async with app.run_test() as pilot:
        earlier_message = UserMessage("Before compaction")
        compact_message = CompactMessage()
        await app._mount_and_scroll(earlier_message)
        await app._mount_and_scroll(compact_message)

        compact_message.set_complete()
        await pilot.pause()

        assert earlier_message.parent is app._messages_area


@pytest.mark.asyncio
async def test_compaction_empties_the_context_gauge() -> None:
    """*Prepare*: Fill the context, as a conversation long enough to compact would.
    *Do*: Compact.
    *Assert*: The gauge drops to what survived, without waiting for the next turn.
    """
    app = build_test_chartreux_app()
    surviving_summary_tokens = 4_000

    async with app.run_test():
        # Prepare
        runtime_state = app.app_server.resources.runtime._state
        runtime_state.stats = AgentStatsSnapshot(
            context_tokens=_SPENT_TOKENS, session_prompt_tokens=_SPENT_TOKENS
        )
        runtime_state.context_window = _CONTEXT_WINDOW
        app._refresh_context_progress()
        assert app.query_one(SessionStatusLine).state.context_tokens == _SPENT_TOKENS

        def _shrink_to_summary(**_kwargs: object) -> str:
            # Stands in for the runtime read `AppServerSession.compact` performs,
            # which is pinned by `test_compact_refreshes_the_cached_context_gauge`.
            # This test owns the other half: the widget re-reads that cache rather
            # than waiting for the next `session/statsUpdated`.
            runtime_state.stats = AgentStatsSnapshot(
                context_tokens=surviving_summary_tokens,
                session_prompt_tokens=_SPENT_TOKENS,
            )
            return "summary"

        app.app_server.compact = AsyncMock(side_effect=_shrink_to_summary)

        # Do
        await app._run_compact(CompactMessage())

        # Assert
        assert (
            app.query_one(SessionStatusLine).state.context_tokens
            == surviving_summary_tokens
        )


@pytest.mark.asyncio
async def test_a_successful_compaction_clears_the_queued_prompts() -> None:
    """The server reset the conversation, so the folded queue no longer applies."""
    app = build_test_chartreux_app()
    cleared: list[int] = []

    async def clear_server_queue() -> None:
        cleared.append(1)

    async with app.run_test():
        assert app.event_handler is not None
        app._queue.clear_server_queue = clear_server_queue
        app.app_server.compact = AsyncMock(return_value="summary")

        await app._run_compact(CompactMessage())

    assert cleared == [1]


@pytest.mark.asyncio
async def test_a_failed_compaction_reports_the_error_and_clears_its_slot() -> None:
    app = build_test_chartreux_app()

    async with app.run_test():
        assert app.event_handler is not None
        message = CompactMessage()
        app.event_handler.current_compact = message
        app._agent_task = asyncio.current_task()
        app.app_server.compact = AsyncMock(side_effect=RuntimeError("disk unavailable"))

        await app._run_compact(message)

        # The typed persistence error reaches the widget instead of vanishing.
        assert message.error_message == "disk unavailable"
        assert message.get_content() == "Error: disk unavailable"
        # The slot is released, so the next /compact is not mistaken for a
        # running one.
        assert app._agent_task is None
        assert app.event_handler.current_compact is None


@pytest.mark.asyncio
async def test_an_interrupted_compaction_is_reported_and_reraises() -> None:
    app = build_test_chartreux_app()

    async with app.run_test():
        assert app.event_handler is not None
        message = CompactMessage()
        app.event_handler.current_compact = message
        app._agent_task = asyncio.current_task()
        app.app_server.compact = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await app._run_compact(message)

        assert message.get_content() == "Error: Compaction interrupted"
        # Cleanup runs on the interrupted path too, and the cancellation itself
        # is never swallowed.
        assert app._agent_task is None
        assert app.event_handler.current_compact is None
