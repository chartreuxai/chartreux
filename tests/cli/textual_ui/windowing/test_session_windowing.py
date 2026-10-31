from __future__ import annotations

import gc
import weakref

from chartreux.app_server.models import (
    PublicCheckpointEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
    TextContentBlock,
)
from chartreux.cli.textual_ui.windowing.state import (
    LOAD_MORE_BATCH_SIZE,
    SessionWindowing,
)


def _message(index: int, content: str = "x") -> PublicMessageEntry:
    return PublicMessageEntry(
        id=f"message-{index}",
        session_id="session-1",
        created_at=index,
        updated_at=index,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        role="user",
        content=[TextContentBlock(text=content)],
        source="harness",
    )


def _checkpoint(index: int) -> PublicCheckpointEntry:
    return PublicCheckpointEntry(
        id=f"checkpoint-{index}",
        session_id="session-1",
        created_at=index,
        updated_at=index,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        kind="internal",
    )


def test_recompute_backfill_keeps_cursor_when_oldest_widgets_skip_entry_prefix() -> (
    None
):
    w = SessionWindowing(LOAD_MORE_BATCH_SIZE)
    w.set_backfill([_message(index) for index in range(80)])
    assert w.remaining == 80

    history: list[PublicHistoryEntry] = [
        *[_message(index) for index in range(80)],
        *[_checkpoint(index) for index in range(80, 85)],
        _message(85, content="visible"),
    ]
    has_backfill = w.recompute_backfill(history, admitted_start_index=80)
    assert has_backfill
    assert w.remaining == 80


def test_recompute_backfill_does_not_resurrect_evicted_entries() -> None:
    history: list[PublicHistoryEntry] = [_message(index) for index in range(100)]
    w = SessionWindowing(LOAD_MORE_BATCH_SIZE)
    w.set_backfill(history[:70])
    assert w.remaining == 70

    has_backfill = w.recompute_backfill(history, admitted_start_index=70)
    assert has_backfill
    assert w.remaining == 70
    assert w.next_load_more_batch() is not None
    assert w.remaining == 60
    w.recompute_backfill(history, admitted_start_index=60)
    assert w.remaining == 60


def test_backfill_list_drops_after_full_admission() -> None:
    w = SessionWindowing(LOAD_MORE_BATCH_SIZE)
    w.set_backfill([_message(index) for index in range(2 * LOAD_MORE_BATCH_SIZE)])

    first = w.next_load_more_batch()
    assert first is not None
    assert first.start_index == LOAD_MORE_BATCH_SIZE
    assert w.remaining == LOAD_MORE_BATCH_SIZE
    assert len(w._backfill_entries) == 2 * LOAD_MORE_BATCH_SIZE

    second = w.next_load_more_batch()
    assert second is not None
    assert second.start_index == 0
    assert w.remaining == 0
    assert w._backfill_entries == []
    assert w.next_load_more_batch() is None
    assert not w.has_backfill


def test_backfill_release_spares_aliased_entries_collects_superseded() -> None:
    live_history: list[PublicHistoryEntry] = [_message(index) for index in range(10)]
    superseded: list[PublicHistoryEntry] = [_message(index) for index in range(10)]
    w = SessionWindowing(LOAD_MORE_BATCH_SIZE)
    w.set_backfill([*superseded, *live_history])
    superseded_refs = [weakref.ref(entry) for entry in superseded]
    aliased_refs = [weakref.ref(entry) for entry in live_history]
    del superseded

    while w.next_load_more_batch() is not None:
        pass
    gc.collect()

    # Entries aliased by the live projection history stay pinned; versions
    # held only by the backfill list are collectable after full admission.
    assert all(ref() is not None for ref in aliased_refs)
    assert all(ref() is None for ref in superseded_refs)
