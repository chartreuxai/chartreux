"""WP4 [4B] presentation-loss and connection-loss UX tests.

The TUI observes the session's ``PresentationConsumer`` states through a
listener: a retire while a view is installed surfaces the transient
stale/refreshing notice, a replacement incarnation's atomic snapshot install
reinstalls the transcript, and a terminal failure surfaces a persistent typed
banner — never silent loss. Connection loss/resume drives the typed
connection-failure notice through the session's connection listeners.
"""

from __future__ import annotations

from typing import Any

import pytest

from chartreux.app_server.events import (
    DEFAULT_PRESENTATION_STREAM_ID,
    RETIRE_METHOD,
    SNAPSHOT_PAGE_METHOD,
    TERMINAL_METHOD,
    StreamPhase,
)
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from tests.conftest import build_test_chartreux_app


def _page(generation: int, payload: str, boundary: int | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": SNAPSHOT_PAGE_METHOD,
        "params": {
            "pageIndex": 0,
            "pageCount": 1,
            "payload": payload,
            "sequenceBoundary": boundary,
        },
    }


def _retire(generation: int) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": RETIRE_METHOD}


def _terminal(generation: int, reason: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": TERMINAL_METHOD,
        "params": {"disposition": "stream_aborted", "reason": reason},
    }


async def _settle(pilot, rounds: int = 6) -> None:
    for _ in range(rounds):
        await pilot.pause(0.05)


@pytest.mark.asyncio
async def test_retire_then_replacement_install_shows_refreshing_then_snapshot() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        session = app.app_server
        flow = app._session_flow
        assert flow._presentation_bound_session is session
        consumer = session.presentation
        stream_id = DEFAULT_PRESENTATION_STREAM_ID

        # Initial incarnation installs (generation 0): ordinary load, no notice.
        consumer.receive_frame(stream_id, 0, _page(0, "{}", 0))
        await _settle(pilot)
        assert flow._presentation_refresh_notice is None
        assert flow._presentation_stream_states[stream_id].installed_generation == 0

        # The current incarnation retires: transient stale/refreshing notice.
        consumer.receive_frame(stream_id, 0, _retire(0))
        await _settle(pilot)
        notice = flow._presentation_refresh_notice
        assert notice is not None
        assert notice.is_mounted
        assert isinstance(notice, UserCommandMessage)
        assert "stale" in str(notice._content)
        assert flow._presentation_stream_states[stream_id].refreshing is True

        # The replacement announces and its snapshot installs atomically:
        # the notice clears and the new generation is recorded.
        consumer.receive_frame(stream_id, 1, _page(1, "{}", 5))
        await _settle(pilot)
        assert flow._presentation_refresh_notice is None
        state = flow._presentation_stream_states[stream_id]
        assert state.installed_generation == 1
        assert state.refreshing is False
        assert state.failure is None
        # The stream itself is installed for the replacement incarnation.
        assert consumer.stream(stream_id).phase is StreamPhase.INSTALLED


@pytest.mark.asyncio
async def test_terminal_frame_surfaces_persistent_typed_failure_banner() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        session = app.app_server
        flow = app._session_flow
        consumer = session.presentation
        stream_id = DEFAULT_PRESENTATION_STREAM_ID

        consumer.receive_frame(stream_id, 0, _page(0, "{}", 0))
        await _settle(pilot)
        consumer.receive_frame(stream_id, 0, _retire(0))
        await _settle(pilot)
        assert flow._presentation_refresh_notice is not None

        consumer.receive_frame(stream_id, 1, _terminal(1, "producer EOF"))
        await _settle(pilot)

        # The transient notice is replaced by the persistent typed banner.
        assert flow._presentation_refresh_notice is None
        banner = flow._presentation_failure_banner
        assert banner is not None
        assert isinstance(banner, ErrorMessage)
        assert banner.is_mounted
        assert "producer EOF" in str(banner._error)
        state = flow._presentation_stream_states[stream_id]
        assert state.failure is not None
        assert "producer EOF" in state.failure
        # The banner is persistent: later frames do not remove it.
        consumer.fire_due(stream_id)
        await _settle(pilot)
        assert flow._presentation_failure_banner is banner


@pytest.mark.asyncio
async def test_connection_loss_notice_appears_and_clears_on_resume() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        session = app.app_server
        flow = app._session_flow

        session._notify_connection(True, "connection reset by peer")
        await _settle(pilot)
        notice = flow._connection_notice
        assert notice is not None
        assert notice.is_mounted
        assert isinstance(notice, UserCommandMessage)
        assert "connection reset by peer" in str(notice._content)
        assert session.connection_lost is True

        # Resume: the notice clears and the session re-announces (retire) via
        # the presentation consumer without arming a second timer driver.
        consumer = session.presentation
        stream_id = DEFAULT_PRESENTATION_STREAM_ID
        consumer.receive_frame(stream_id, 0, _page(0, "{}", 0))
        await _settle(pilot)
        driver = session._presentation_timer_task
        session._notify_connection(False, "")
        session._reannounce_presentation()
        await _settle(pilot)
        assert flow._connection_notice is None
        assert session.connection_lost is False
        assert session._presentation_timer_task is driver
        stream = consumer.stream(stream_id)
        assert stream.phase is StreamPhase.AWAITING
        (timer,) = stream.timers
        assert timer.generation == 0


@pytest.mark.asyncio
async def test_stopping_the_event_listener_unbinds_the_observers() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        session = app.app_server
        flow = app._session_flow
        assert flow._presentation_bound_session is session

        await app._stop_app_server_event_listener()

        assert flow._presentation_bound_session is None
        # Frames after unbind no longer drive the TUI surface.
        consumer = session.presentation
        stream_id = DEFAULT_PRESENTATION_STREAM_ID
        consumer.receive_frame(stream_id, 0, _page(0, "{}", 0))
        await _settle(pilot)
        assert stream_id not in flow._presentation_stream_states


@pytest.mark.asyncio
async def test_install_presentation_snapshot_adopts_bounded_backfill() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        flow = app._session_flow
        windowing = app._windowing
        # Seed a stale backfill from a previous generation, then install.
        windowing.set_backfill(["stale-1", "stale-2"])  # type: ignore[list-item]
        assert windowing.has_backfill

        await flow.install_presentation_snapshot()
        await _settle(pilot)

        # The old backfill was discarded wholesale: the snapshot's history is
        # empty here, so no pre-snapshot content can be served.
        assert not windowing.has_backfill
