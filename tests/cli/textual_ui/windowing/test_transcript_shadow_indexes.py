"""Reconcile indexes and their maintenance (P3b2a/P3b2b).

The windowing suites drive the indexes through real admission, eviction,
restore and protection transitions. P3b2a proved the indexes against the
scans with a loud consistency oracle, which the P3b2b cutover removed once it
made the indexes the live reconcile path; these tests pin what remains under
test: every transition carries each facet (residency, placeholders,
checkpoint membership, neighbour adjacency, content-order links, and the
live cache) to the right answer.

Fixtures use stubbed placement geometry -- the seam ``register_mounted``
already accepts -- so the index maintenance is the only thing under test.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence, ValuesView
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
from textual.geometry import Region, Size
from textual.widget import Widget

from chartreux.cli.textual_ui.transcript_presenter import TranscriptPresenter
from chartreux.cli.textual_ui.windowing.transcript import (
    TranscriptUnit,
    TranscriptWindow,
    _CompactingSet,
)
from tests.cli.textual_ui.test_history_grouping import _message
from tests.cli.textual_ui.windowing.test_session_windowing import _checkpoint


class _Stream(Widget):
    """Placement parent with stubbed geometry; no Textual app is mounted."""

    def __init__(self, width: int = 60, height: int = 10) -> None:
        super().__init__()
        self._stub_children: list[Widget] = []
        self._stub_size = Size(width, height)

    @property
    def children(self) -> Sequence[Widget]:
        return self._stub_children

    @property
    def size(self) -> Size:
        return self._stub_size

    @property
    def virtual_size(self) -> Size:
        return self._stub_size

    @property
    def region(self) -> Region:
        return Region(0, 0, self._stub_size.width, self._stub_size.height)

    @property
    def is_at_bottom(self) -> bool:
        return True

    def attach(self, row: Widget) -> None:
        self._stub_children.append(row)


class _Row(Widget):
    """One mounted placement root with a fixed screen region."""

    def __init__(self, stream: _Stream, top: int, height: int) -> None:
        super().__init__()
        self._stream = stream
        self._stub_region = Region(0, top, stream.size.width, height)

    @property
    def parent(self) -> _Stream:
        return self._stream

    @property
    def region(self) -> Region:
        return self._stub_region

    @property
    def text_selection(self) -> None:
        # The gate reads selection off mounted rows; an unmounted stub has none.
        return None


class _PlaceholderStub:
    """Stand-in for an evicted unit's PlacementPlaceholder with fixed geometry."""

    __slots__ = ("parent", "region")

    def __init__(self, top: int, height: int = 8) -> None:
        self.parent = None
        self.region = Region(0, top, 60, height)

    def invalidate_geometry(self) -> None:
        return None


class _BatchUpdate:
    """Synchronous no-op stand-in for ``App.batch_update``."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


class _HostStub:
    """Exactly the host attributes the reconcile gate reads."""

    def __init__(self, stream: _Stream) -> None:
        self._shutdown_started = False
        self._chat_widget = stream
        self._messages_area = stream
        self.event_handler = None
        self._rewind_highlighted_widget = None
        self._queue_selected_widget = None
        self._agent_transcript_focus_target = None
        self._active_turn_start = None
        self._picker = SimpleNamespace(previewing=False)

    def batch_update(self) -> _BatchUpdate:
        return _BatchUpdate()


def _presenter(stream: _Stream) -> tuple[TranscriptPresenter, TranscriptWindow]:
    presenter = TranscriptPresenter(get_show_timestamps=lambda: False)
    presenter._bind_app(cast("Any", _HostStub(stream)))
    return presenter, presenter._transcript


def _mount_rows(window: TranscriptWindow, stream: _Stream, height: int) -> None:
    for offset, unit_id in enumerate(window.unit_ids):
        row = _Row(stream, offset * height, height)
        stream.attach(row)
        window.register_mounted(unit_id, [row])


def _send_reconcile_to_a_recorder(
    monkeypatch: pytest.MonkeyPatch, window: TranscriptWindow
) -> list[dict[str, object]]:
    """Keep the gate from scheduling a reconcile the stub geometry cannot run."""
    scheduled: list[dict[str, object]] = []
    monkeypatch.setattr(
        window, "request_reconcile", lambda *_, **kwargs: scheduled.append(kwargs)
    )
    return scheduled


def _linked_order(window: TranscriptWindow) -> list[str]:
    order: list[str] = []
    current = window._shadow.head
    while current is not None:
        order.append(current)
        current = window._shadow.after[current]
    return order


class _CountingIds(list[str]):
    """Registry id list that counts full traversals (slices are not counted)."""

    def __init__(self, unit_ids: list[str]) -> None:
        super().__init__(unit_ids)
        self.visits = 0

    def __iter__(self) -> Iterator[str]:
        for unit_id in super().__iter__():
            self.visits += 1
            yield unit_id


class _CountingUnits:
    """Unit-map stand-in that counts ``.values()`` scans and item reads."""

    def __init__(self, units: dict[str, TranscriptUnit]) -> None:
        self._units = units
        self.values_calls = 0
        self.getitem_calls = 0

    def values(self) -> ValuesView[TranscriptUnit]:
        self.values_calls += 1
        return self._units.values()

    def get(self, key: str, default: TranscriptUnit | None = None) -> Any:
        self.getitem_calls += 1
        return self._units.get(key, default)

    def __getitem__(self, key: str) -> TranscriptUnit:
        self.getitem_calls += 1
        return self._units[key]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._units, name)


def test_indexes_follow_identity_adoption() -> None:
    stream = _Stream()
    _, window = _presenter(stream)
    pending = _message(0)
    pending.id = ""
    window.admit([pending], start_index=0)
    transient = window.unit_ids[0]
    assert transient.startswith("transient-")
    row = _Row(stream, 0, 8)
    stream.attach(row)
    window.register_mounted(transient, [row])
    assert window._shadow.mounted == {transient}

    pending.id = "message-adopted"
    window.admit([pending], start_index=0)

    assert window.unit_ids == ["message-adopted"]
    assert window._shadow.mounted == {"message-adopted"}
    assert _linked_order(window) == ["message-adopted"]
    # The live scan counts the placement under its adopted identity too.
    assert window.eviction_candidates(stream.region.y, stream.region.height) == []

    # A placeholder-owning unit carries its index entries across a rename:
    # build the cached content order under the adopted id first, then adopt
    # again and require the cached order to follow the rename.
    unit = window.units["message-adopted"]
    unit.mounted_roots = []
    unit.placeholders = [cast("Any", _PlaceholderStub(0))]
    window._shadow_unit(unit)
    assert window.placeholder_unit_ids() == ["message-adopted"]

    pending.id = "message-readopted"
    window.admit([pending], start_index=0)

    assert window._shadow.placeholders == {"message-readopted"}
    assert window.placeholder_unit_ids() == ["message-readopted"]


def test_indexes_follow_admission_mount_pin_and_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _Stream()
    presenter, window = _presenter(stream)
    window.admit([_message(index) for index in range(3, 6)], start_index=3)
    window.admit([_message(index) for index in range(3)], start_index=0)
    _mount_rows(window, stream, height=8)

    assert window.unit_ids == [f"message-{index}" for index in range(6)]
    assert _linked_order(window) == window.unit_ids
    assert window._shadow.mounted == set(window.unit_ids)
    assert window._shadow.placeholders == set()
    # No checkpoint yet, so nothing claims a traversable neighbour.
    assert window._shadow.checkpoints == set()
    assert window._shadow.neighbors == set()

    # A pinned placement leaves the mounted-row budget on both sides.
    window.units["message-1"].pin_reasons.add("active-turn")
    assert window.eviction_candidates(stream.region.y, stream.region.height) == []
    window.units["message-1"].pin_reasons.clear()

    # Checkpoint membership and adjacency are re-derived on replacement.
    checkpoint = _checkpoint(1).model_copy(
        update={"id": "message-1", "kind": "compaction"}
    )
    window.update_entry(checkpoint)
    assert window._shadow.checkpoints == {"message-1"}
    assert window._shadow.neighbors == {"message-1"}

    _send_reconcile_to_a_recorder(monkeypatch, window)
    presenter.request_transcript_reconcile()  # every facet agrees

    window.remove_unit("message-1")
    assert window._shadow.checkpoints == set()
    assert window._shadow.neighbors == set()
    assert window._shadow.mounted == set(window.unit_ids)
    assert _linked_order(window) == window.unit_ids
    assert window.eviction_candidates(stream.region.y, stream.region.height) == []

    window.reset()
    assert window._shadow.mounted == set()
    assert window._shadow.placeholders == set()
    assert window._shadow.checkpoints == set()
    assert window._shadow.neighbors == set()
    assert window._shadow.live == set()
    assert _linked_order(window) == []


def test_active_turn_pin_loop_is_bounded_to_the_affected_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _Stream()
    presenter, window = _presenter(stream)
    window.admit([_message(index) for index in range(50)], start_index=0)
    _mount_rows(window, stream, height=8)
    _send_reconcile_to_a_recorder(monkeypatch, window)
    counted = _CountingIds(window.unit_ids)
    window.unit_ids = counted

    presenter._app._active_turn_start = 40
    counted.visits = 0
    presenter.request_transcript_reconcile()
    assert counted.visits == 0, "the active-turn pin loop walked the whole registry"
    pinned = {
        key for key, unit in window.units.items() if "active-turn" in unit.pin_reasons
    }
    assert pinned == {f"message-{index}" for index in range(40, 50)}

    presenter._app._active_turn_start = None
    counted.visits = 0
    presenter.request_transcript_reconcile()
    assert counted.visits == 0


def test_checkpoint_replaced_by_message_drops_its_neighbor_entry() -> None:
    stream = _Stream(height=10)
    _, window = _presenter(stream)
    window.admit([_message(index) for index in range(200)], start_index=0)
    _mount_rows(window, stream, height=8)
    checkpoint = _checkpoint(150).model_copy(
        update={"id": "message-150", "kind": "compaction"}
    )

    # While the entry is a checkpoint with mounted neighbours, the unit is
    # protected from eviction as a compaction neighbour.
    window.update_entry(checkpoint)
    assert window._shadow.checkpoints == {"message-150"}
    assert window._shadow.neighbors == {"message-150"}
    candidates = window.eviction_candidates(
        stream.region.y,
        stream.region.height,
        compaction_neighbors=frozenset(window._shadow.neighbors),
    )
    assert "message-150" not in candidates

    # Replacing the checkpoint with a message must drop the neighbours entry,
    # or the unit stays protected (and out of the eviction candidates).
    window.update_entry(_message(150))
    assert window._shadow.checkpoints == set()
    assert window._shadow.neighbors == set()
    candidates = window.eviction_candidates(
        stream.region.y,
        stream.region.height,
        compaction_neighbors=frozenset(window._shadow.neighbors),
    )
    assert "message-150" in candidates

    # The admit re-admission path funnels through the same re-derivation.
    window.update_entry(checkpoint)
    assert window._shadow.neighbors == {"message-150"}
    window.admit([_message(150)], start_index=150)
    assert window._shadow.checkpoints == set()
    assert window._shadow.neighbors == set()


def test_placeholder_index_serves_content_order_without_a_per_query_resort() -> None:
    stream = _Stream()
    _, window = _presenter(stream)
    window.admit([_message(index) for index in range(6)], start_index=0)
    # Two evicted units, registered in reverse content order.
    for index in (4, 1):
        unit = window.units[f"message-{index}"]
        unit.placeholders = [cast("Any", object())]
        window._shadow_unit(unit)
    assert window.placeholder_unit_ids() == ["message-1", "message-4"]

    counting = _CountingUnits(window.units)
    window.units = cast("dict[str, TranscriptUnit]", counting)
    assert window.placeholder_unit_ids() == ["message-1", "message-4"]
    assert counting.getitem_calls == 0, "the cached order was re-derived per query"

    # A membership change invalidates the cached order.
    unit = window.units["message-4"]
    unit.placeholders = []
    window._shadow_unit(unit)
    assert window.placeholder_unit_ids() == ["message-1"]


def test_admit_live_history_reads_residency_from_the_shadow_index() -> None:
    stream = _Stream()
    presenter, window = _presenter(stream)
    window.admit([_message(0), _message(1)], start_index=0)
    _mount_rows(window, stream, height=8)
    assert window._shadow.mounted == set(window.unit_ids)
    cast("Any", presenter._app).app_server = SimpleNamespace(
        history=[_message(index) for index in range(4)]
    )
    counting = _CountingUnits(window.units)
    window.units = cast("dict[str, TranscriptUnit]", counting)

    presenter.admit_live_history()

    assert counting.values_calls == 0, "admit_live_history scanned the registry"


def test_restore_candidates_window_matches_the_full_placeholder_scan() -> None:
    """The windowed restore selection answers exactly what the full scan does."""
    stream = _Stream(height=10)
    _, window = _presenter(stream)
    window.admit([_message(index) for index in range(40)], start_index=0)
    _mount_rows(window, stream, height=8)
    # Evict every unit: each keeps its slot as a placeholder carrying the row's
    # geometry, so the placement regions stay monotonic in content order.
    for position, unit_id in enumerate(window.unit_ids):
        unit = window.units[unit_id]
        unit.mounted_roots = []
        unit.placeholders = [cast("Any", _PlaceholderStub(position * 8))]
        window._shadow_unit(unit)
    assert len(window._shadow.placeholders) == 40

    def in_window(unit_id: str, top: int, height: int) -> bool:
        # The per-unit viewport check the reconcile loop applies to candidates.
        unit = window.units[unit_id]
        first, last = unit.placeholders[0], unit.placeholders[-1]
        return first.region.y < top + 2 * height and last.region.bottom > top - height

    def full_scan(top: int, height: int) -> list[str]:
        return [
            unit_id
            for unit_id in window.placeholder_unit_ids()
            if window.units[unit_id].placeholders and in_window(unit_id, top, height)
        ]

    def windowed(top: int, height: int) -> list[str]:
        return [
            unit_id
            for unit_id in window._restore_candidates(top, height)
            if window.units[unit_id].placeholders and in_window(unit_id, top, height)
        ]

    for top in range(0, 40 * 8, 7):
        for height in (5, 10, 40):
            assert windowed(top, height) == full_scan(top, height), (
                f"windowed restore candidates diverged at top={top} height={height}"
            )

    # The selection is bounded by the window, not by the placeholder count.
    expected = full_scan(300, 10)
    assert expected
    counting = _CountingUnits(window.units)
    window.units = cast("dict[str, TranscriptUnit]", counting)
    candidates = window._restore_candidates(300, 10)
    assert [
        unit_id for unit_id in candidates if in_window(unit_id, 300, 10)
    ] == expected
    assert counting.getitem_calls <= 24, (
        f"the windowed scan read {counting.getitem_calls} of 40 placeholder units"
    )


@pytest.mark.asyncio
async def test_prefix_readmission_rebases_the_active_turn_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _Stream()
    presenter, window = _presenter(stream)
    window.admit([_message(index) for index in range(5, 10)], start_index=5)
    # TurnStarted records the admitted end as the turn's first timeline index.
    presenter._app._active_turn_start = window.admitted_end_index
    monkeypatch.setattr(window, "build_unit", lambda unit_id, indices: [])
    area = SimpleNamespace(size=SimpleNamespace(width=60))

    # A whole-history prefix re-admitted at the current start displaces the
    # recorded timeline; the marker must follow it instead of staying stale-low
    # and over-pinning the older suffix.
    await presenter.mount_history_batch(
        [_message(index) for index in range(100, 105)],
        cast("Widget", area),
        start_index=5,
    )

    assert window.admitted_start_index == 5
    assert presenter._app._active_turn_start == window.admitted_end_index == 15
    window.admit([_message(index) for index in range(10, 12)], start_index=15)
    ids = window.unit_ids
    first = bisect_left(
        ids,
        presenter._app._active_turn_start,
        key=lambda key: window.units[key].start_index,
    )
    assert ids[first:] == ["message-10", "message-11"]


def test_live_dirty_set_compacts_after_a_placement_spike() -> None:
    """A dirty-set spike must not leave a peak-sized hash table behind.

    The presenter scans ``sorted(live_dirty)`` every tick, and set iteration
    walks the whole table, so a table left at the spike's size would make the
    per-tick cost track the historical peak instead of live membership.
    """
    stream = _Stream()
    _, window = _presenter(stream)
    window.admit([_message(index) for index in range(2000)], start_index=0)
    for unit in window.units.values():
        window._shadow_unit(unit)  # a placement-transition spike
    assert len(window._shadow.live_dirty) == 2000
    spike_bytes = sys.getsizeof(window._shadow.live_dirty)

    survivors = set(window.unit_ids[-3:])
    for unit_id in list(window._shadow.live_dirty):
        window._observe_live(unit_id, unit_id in survivors)
    window._commit_live_scan()

    assert window._shadow.live == survivors
    # The re-dirtied survivors keep the set nonempty, so only a compaction
    # (not the empty-clear) can release the spike-sized table.
    assert set(window._shadow.live_dirty) == survivors
    assert sys.getsizeof(window._shadow.live_dirty) < spike_bytes // 4


def test_mounted_set_compacts_after_a_mounted_spike() -> None:
    """A residency spike must not leave a peak-sized hash table behind.

    The presenter's selection scan and the reconcile's eviction scan iterate
    ``shadow.mounted`` every tick, and set iteration walks the whole table,
    so a table left at the spike's size would make the per-tick cost track
    the historical peak instead of live residency.
    """
    stream = _Stream()
    _, window = _presenter(stream)
    window.admit([_message(index) for index in range(2000)], start_index=0)
    _mount_rows(window, stream, height=8)  # the mounted-membership spike
    assert len(window._shadow.mounted) == 2000
    spike_bytes = sys.getsizeof(window._shadow.mounted)

    survivors = set(window.unit_ids[-3:])
    for unit_id in window.unit_ids:
        if unit_id in survivors:
            continue
        unit = window.units[unit_id]
        unit.mounted_roots = []
        unit.placeholders = [cast("Any", _PlaceholderStub(0))]
        window._shadow_unit(unit)  # the eviction that settles the spike

    assert set(window._shadow.mounted) == survivors
    assert sys.getsizeof(window._shadow.mounted) < spike_bytes // 4


def test_live_set_compacts_after_a_live_membership_spike() -> None:
    """A live-membership spike must not leave a peak-sized hash table behind.

    The presenter reads ``frozenset(shadow.live)`` on every protection tick,
    and set iteration walks the whole table, so a table left at the spike's
    size would make the per-tick cost track the historical spike instead of
    live membership.
    """
    stream = _Stream()
    _, window = _presenter(stream)
    window.admit([_message(index) for index in range(2000)], start_index=0)
    window._mark_live_dirty(window.unit_ids)
    for unit_id in window.unit_ids:
        window._observe_live(unit_id, True)  # the live-membership spike
    assert len(window._shadow.live) == 2000
    spike_bytes = sys.getsizeof(window._shadow.live)

    survivors = set(window.unit_ids[-3:])
    window._mark_live_dirty(window.unit_ids)
    for unit_id in window.unit_ids:
        window._observe_live(unit_id, unit_id in survivors)

    assert set(window._shadow.live) == survivors
    assert sys.getsizeof(window._shadow.live) < spike_bytes // 4


def test_neighbors_set_compacts_after_a_checkpoint_spike() -> None:
    """A checkpoint-density spike must not leave a peak-sized table behind.

    The presenter reads ``frozenset(shadow.neighbors)`` on every protection
    tick, and set iteration walks the whole table, so a table left at the
    spike's size would make the per-tick cost track the historical checkpoint
    density instead of the surviving adjacency.
    """
    stream = _Stream()
    _, window = _presenter(stream)
    window.admit([_message(index) for index in range(2000)], start_index=0)
    _mount_rows(window, stream, height=8)  # a mounted neighbour for every unit
    checkpoint = _checkpoint(0)
    for unit in window.units.values():
        unit.entries = [checkpoint]
        window._shadow_entries(unit)  # the checkpoint-dense spike
    for unit_id in window.unit_ids:
        # The triple re-derivation can discard a unit whose neighbour has not
        # been re-derived yet, so settle the order with one full pass.
        window._shadow_neighbors(unit_id)
    assert len(window._shadow.neighbors) == 2000
    spike_bytes = sys.getsizeof(window._shadow.neighbors)

    survivors = set(window.unit_ids[-3:])
    for index, unit_id in enumerate(window.unit_ids):
        unit = window.units[unit_id]
        if unit_id in survivors:
            continue
        unit.mounted_roots = []
        unit.placeholders = [cast("Any", _PlaceholderStub(0))]
        unit.entries = [_message(index)]
        window._shadow_unit(unit)
        window._shadow_entries(unit)  # the eviction that settles the spike

    assert set(window._shadow.neighbors) == survivors
    assert sys.getsizeof(window._shadow.neighbors) < spike_bytes // 4


def test_placeholders_set_compacts_after_a_placeholder_spike() -> None:
    """A placeholder spike must not leave a peak-sized table behind.

    The restore scan's ``placeholder_unit_ids()`` fallback sorts the set, and
    set iteration walks the whole table, so a table left at the spike's size
    would make a rebuild cost track the historical spike instead of the
    surviving placeholders.
    """
    stream = _Stream()
    _, window = _presenter(stream)
    window.admit([_message(index) for index in range(2000)], start_index=0)
    for unit in window.units.values():
        unit.placeholders = [cast("Any", _PlaceholderStub(0))]
        window._shadow_unit(unit)  # the placeholder spike
    assert len(window._shadow.placeholders) == 2000
    spike_bytes = sys.getsizeof(window._shadow.placeholders)

    survivors = set(window.unit_ids[-3:])
    for unit_id, unit in window.units.items():
        if unit_id in survivors:
            continue
        unit.placeholders = []
        window._shadow_unit(unit)  # the restore that settles the spike

    assert set(window._shadow.placeholders) == survivors
    assert sys.getsizeof(window._shadow.placeholders) < spike_bytes // 4


@pytest.mark.parametrize(
    "mutator", ["ixor", "symmetric_difference_update", "iand"], ids=str
)
def test_compacting_set_compacts_on_every_shrinking_mutator(mutator: str) -> None:
    """Every shrinking mutator must run the compaction check, not only ``-=``.

    ``^=`` and ``symmetric_difference_update`` drop most of a peak-sized set
    while keeping its table, exactly like ``discard``. ``&=`` already lets
    CPython rebuild a right-sized table, and this override keeps that path on
    ``compact()`` so the high-water mark tracks the shrink too.
    """
    values = _CompactingSet(f"unit-{index}" for index in range(2000))
    survivor = "unit-1999"
    peak_bytes = sys.getsizeof(values)
    dropped = set(values) - {survivor}

    if mutator == "ixor":
        values ^= dropped
    elif mutator == "symmetric_difference_update":
        values.symmetric_difference_update(dropped)
    else:
        values &= {survivor}

    assert values == {survivor}
    assert sys.getsizeof(values) < peak_bytes // 4


def test_reslot_relinks_shadow_adjacency_after_a_start_index_move() -> None:
    """A moved unit must re-link the shadow adjacency, not keep stale links.

    ``_replace_member`` re-sorts ``unit_ids`` when a re-admitted member moves
    its unit's start index; the shadow's before/after links mirror that
    order, so they must follow the move or neighbour protection keeps the
    stale adjacency.
    """
    stream = _Stream(height=40)
    _, window = _presenter(stream)
    window.admit([_message(index) for index in range(3)], start_index=0)
    window.update_entry(
        _checkpoint(1).model_copy(update={"id": "message-1", "kind": "compaction"})
    )
    row = _Row(stream, 0, 8)
    stream.attach(row)
    window.register_mounted("message-0", [row])
    assert window.unit_ids == ["message-0", "message-1", "message-2"]
    assert _linked_order(window) == window.unit_ids
    assert window._shadow.checkpoints == {"message-1"}
    assert window._shadow.neighbors == {"message-1"}

    unit = window.units["message-0"]
    window._replace_member(unit, "message-0", _message(0), index=10)

    assert window.unit_ids == ["message-1", "message-2", "message-0"]
    assert _linked_order(window) == window.unit_ids
    shadow = window._shadow
    assert shadow.head == "message-1"
    assert shadow.tail == "message-0"
    assert shadow.before["message-1"] is None
    assert shadow.after["message-1"] == "message-2"
    assert shadow.before["message-0"] == "message-2"
    assert shadow.after["message-0"] is None
    # The checkpoint's only neighbour is now the unmounted message-2: the
    # stale pre-move adjacency (mounted message-0 before it) must not keep
    # it protected.
    assert shadow.neighbors == set()
