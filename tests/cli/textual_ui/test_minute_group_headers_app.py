"""Minute-group headers across the live/restored seam, in the real app.

The window derives its stamps from the visible entry set; the live delivery
path stamps arrivals that own no unit (notice records, fold interiors). These
tests run both paths against the same app, so the shared walk is exercised
where it matters: after a restore, across a rebuild, and under a live
timestamp-gate change.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from chartreux.app_server.events import HistoryEntryAdded
from chartreux.app_server.models import (
    CompletedEffectState,
    EffectCallDisplay,
    EffectResultDisplay,
    GenericEffectDetail,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicNoticeEntry,
    PublicReasoningEntry,
    ScheduledLoopFiredNoticeDetail,
)
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.handlers.event_handler import EventHandler
from chartreux.cli.textual_ui.widgets.message_header import (
    format_group_header_timestamp,
)
from chartreux.cli.textual_ui.widgets.minute_group_header import MinuteGroupHeader
from tests.conftest import build_test_chartreux_app

TURN_ID = "turn-live"


def _local(moment: datetime) -> datetime:
    return moment.astimezone()


def _at(hour: int, minute: int, *, day: int = 6) -> tuple[int, int, int, int, int]:
    moment = _local(datetime(2026, 11, day, hour, minute))
    return (moment.year, moment.month, moment.day, moment.hour, moment.minute)


def _epoch_ms(hour: int, minute: int, *, day: int = 6) -> int:
    return int(_local(datetime(2026, 11, day, hour, minute)).timestamp()) * 1000


def _label(hour: int, minute: int) -> str:
    return format_group_header_timestamp(_at(hour, minute))


def _base(entry_id: str) -> dict[str, object]:
    return {
        "id": entry_id,
        "session_id": "session-1",
        "updated_at": 1,
        "generation_status": PublicEntryGenerationStatus.COMPLETED,
    }


def _restored_reasoning(entry_id: str, hour: int, minute: int) -> PublicReasoningEntry:
    """A record restored from a v0.7.4 log: the persisted per-event clock."""
    return PublicReasoningEntry(
        **_base(entry_id),  # type: ignore[arg-type]
        created_at=20,
        text="thinking",
        event_at=_epoch_ms(hour, minute),
    )


def _live_reasoning(entry_id: str, hour: int, minute: int) -> PublicReasoningEntry:
    return PublicReasoningEntry(
        **_base(entry_id),  # type: ignore[arg-type]
        created_at=_epoch_ms(hour, minute),
        turn_id=TURN_ID,
        text="thinking",
    )


def _live_effect(entry_id: str, hour: int, minute: int) -> PublicEffectEntry:
    detail = GenericEffectDetail(
        tool_name="tool", display=EffectCallDisplay(summary="tool", status_text="run")
    )
    state = CompletedEffectState(
        display=EffectResultDisplay(success=True, message="done"), duration_ms=1000
    )
    return PublicEffectEntry(
        **_base(entry_id),  # type: ignore[arg-type]
        created_at=_epoch_ms(hour, minute),
        turn_id=TURN_ID,
        title="tool",
        detail=detail,
        state=state,
    )


def _scheduled_loop_notice(entry_id: str, hour: int, minute: int) -> PublicNoticeEntry:
    return PublicNoticeEntry(
        **_base(entry_id),  # type: ignore[arg-type]
        created_at=_epoch_ms(hour, minute),
        turn_id=TURN_ID,
        level="info",
        message="Loop fired",
        detail=ScheduledLoopFiredNoticeDetail(loop_id="2158d161"),
    )


def _headers(app: ChartreuxApp) -> list[str]:
    return [header.label for header in app._messages_area.query(MinuteGroupHeader)]


def _history(app: ChartreuxApp) -> list[PublicHistoryEntry]:
    history = app.app_server._state.projection.state.history
    assert history is not None
    return history


def _handler(app: ChartreuxApp) -> EventHandler:
    assert app.event_handler is not None
    return app.event_handler


async def _settle(pilot, rounds: int = 12) -> None:
    for _ in range(rounds):
        await pilot.pause()


@pytest.mark.asyncio
async def test_restored_stamp_then_same_minute_live_notice_mounts_one_header() -> None:
    # A2: restore stamps 14:20; a live notice arriving in that same minute must
    # not open a second header.
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 32)) as pilot:
        await app._session_ready.wait()
        history = _history(app)
        history.append(_restored_reasoning("r1", 14, 20))
        await app._resume_history_from_messages()
        await _settle(pilot)
        assert _headers(app) == [_label(14, 20)]

        await _handler(app).handle_event(
            HistoryEntryAdded(_scheduled_loop_notice("n1", 14, 20))
        )
        await _settle(pilot)

        assert _headers(app) == [_label(14, 20)]


@pytest.mark.asyncio
async def test_restored_stamp_then_live_notice_in_a_new_minute_mounts_one_header() -> (
    None
):
    # A2 positive control: the live notice crossing the minute does open one.
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 32)) as pilot:
        await app._session_ready.wait()
        history = _history(app)
        history.append(_restored_reasoning("r1", 14, 20))
        await app._resume_history_from_messages()
        await _settle(pilot)
        assert _headers(app) == [_label(14, 20)]

        await _handler(app).handle_event(
            HistoryEntryAdded(_scheduled_loop_notice("n1", 14, 27))
        )
        await _settle(pilot)

        assert _headers(app) == [_label(14, 20), _label(14, 27)]


@pytest.mark.asyncio
async def test_rebuild_after_a_live_notice_keeps_the_same_minute_unit_header() -> None:
    # N2: the rebuild removes the notice's record (it owns no rebuilt widget),
    # so its stamp must stop suppressing the unit that continues its minute.
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 32)) as pilot:
        await app._session_ready.wait()
        history = _history(app)
        history.append(_scheduled_loop_notice("n1", 14, 20))
        history.append(_live_reasoning("r1", 14, 20))

        await _handler(app).handle_event(
            HistoryEntryAdded(_scheduled_loop_notice("n1", 14, 20))
        )
        await _handler(app).handle_event(
            HistoryEntryAdded(_live_reasoning("r1", 14, 20))
        )
        await _settle(pilot)
        assert _headers(app) == [_label(14, 20)]

        await app._rebuild_transcript_from_current_session()
        await _settle(pilot)

        # The notice's record no longer renders, so the reasoning unit is the
        # minute's only visible record and must carry its header exactly once.
        headers = app._messages_area.query(MinuteGroupHeader)
        assert [header.label for header in headers] == [_label(14, 20)]


@pytest.mark.asyncio
async def test_the_timestamp_gate_withdraws_and_restores_every_minute_header() -> None:
    # W1: the window owns the unit header; the live path owns the fold-interior
    # and notice-record stamps. One preference change must reconcile all three.
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 32)) as pilot:
        await app._session_ready.wait()
        history = _history(app)
        history.append(_restored_reasoning("r1", 14, 20))
        await app._resume_history_from_messages()
        await _settle(pilot)

        # A live fold continues past 14:20, stamping its interior at 14:21; a
        # notice record opens 14:22 with a stamp of its own.
        await _handler(app).handle_event(
            HistoryEntryAdded(_live_reasoning("r2", 14, 20))
        )
        await _handler(app).handle_event(HistoryEntryAdded(_live_effect("e1", 14, 21)))
        await _handler(app).handle_event(
            HistoryEntryAdded(_scheduled_loop_notice("n1", 14, 22))
        )
        await _settle(pilot)
        assert _headers(app) == [_label(14, 20), _label(14, 21), _label(14, 22)]

        await app.app_server.resources.config.update({"show_message_timestamps": False})
        await _settle(pilot)
        assert _headers(app) == []

        await app.app_server.resources.config.update({"show_message_timestamps": True})
        await _settle(pilot)
        assert _headers(app) == [_label(14, 20), _label(14, 21), _label(14, 22)]
