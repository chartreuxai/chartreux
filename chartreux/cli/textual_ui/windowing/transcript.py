from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence, Set as AbstractSet
from dataclasses import dataclass, field
from typing import cast
from weakref import WeakKeyDictionary, ref

from textual.containers import VerticalScroll
from textual.dom import NoScreen
from textual.widget import Widget

from chartreux.app_server.models import (
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicReasoningEntry,
)
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.messages import StreamingMessageBase
from chartreux.cli.textual_ui.widgets.tool_grouping import (
    ToolGroupExpansionState,
    ToolGroupKey,
    entry_keeps_tool_group,
)
from chartreux.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolGroup,
    ToolResultMessage,
)
from chartreux.cli.textual_ui.windowing.fingerprints import (
    Fingerprint,
    fingerprint_entry,
)
from chartreux.cli.textual_ui.windowing.history import (
    _build_history_widgets_raw,
    history_entry_renders_widget,
)
from chartreux.cli.textual_ui.windowing.placeholder import PlacementPlaceholder


@dataclass(frozen=True, slots=True)
class TranscriptAnchor:
    """Logical reading position, independent of a placement's DOM lifetime."""

    entry_id: str
    history_index: int
    row_offset: int
    screen_row: int
    following: bool
    scroll_revision: int


@dataclass(slots=True)
class TranscriptUnit:
    id: str
    member_entry_ids: list[str]
    entries: list[PublicHistoryEntry]
    start_index: int
    group_key: ToolGroupKey | None = None
    # Compact per-entry fingerprints: the registry's revision basis, compared
    # in place of retained deep copies of every entry.
    entry_fingerprints: list[Fingerprint] = field(default_factory=list)
    mounted_roots: list[Widget] = field(default_factory=list)
    placeholders: list[PlacementPlaceholder] = field(default_factory=list)
    content_height: int | None = None
    # Parent Stream's outer size.width, not the placement's content width.
    geometry_width: int | None = None
    geometry_version: int = 0
    pin_reasons: set[str] = field(default_factory=set)
    restoring: bool = False
    restore_generation: int = 0


class _CompactingSet(set[str]):
    """A set with a high-water mark, so a membership spike can shrink.

    ``set.discard()`` never shrinks a set's hash table, so after a spike (a
    placement-transition dirty spike, a mounted-, live- or neighbour-
    membership spike, a placeholder spike) the table stays peak-sized behind
    the surviving members -- and iterating the set walks every slot of it:
    the presenter's ``sorted(live_dirty)`` scan, its ``frozenset(live)`` and
    ``frozenset(neighbors)`` reads on every protection tick, the reconcile's
    eviction scan, and the restore scan's ``sorted(placeholders)`` rebuild.

    Every mutator keeps ``peak`` at the high-water mark, and every mutator
    that can shrink the set runs ``compact()``'s cheap check, which rehashes
    once the size falls below half the peak, so scans track live membership,
    not the historical peak. Compacting mutators: ``discard``, ``remove``,
    ``pop``, ``difference_update``, ``__isub__``, ``intersection_update``,
    ``__iand__`` (which delegates to ``intersection_update``), and
    ``__ixor__``/``symmetric_difference_update`` (which note the peak too,
    since they can also grow the set). Growing-only mutators -- ``add``,
    ``update`` and ``__ior__`` -- raise the peak; ``clear`` drops it to zero.
    Membership semantics are exactly ``set``'s.
    """

    def __init__(self, values: Iterable[str] = ()) -> None:
        super().__init__(values)
        self.peak = len(self)

    def _note(self) -> None:
        self.peak = max(self.peak, len(self))

    def add(self, element: str) -> None:
        super().add(element)
        self._note()

    def update(self, *others: Iterable[str]) -> None:
        super().update(*others)
        self._note()

    def __ior__(self, other: AbstractSet[str]) -> _CompactingSet:
        super().__ior__(other)
        self._note()
        return self

    def __ixor__(self, other: AbstractSet[str]) -> _CompactingSet:
        super().__ixor__(other)
        self._note()
        self.compact()
        return self

    def symmetric_difference_update(self, other: Iterable[str]) -> None:
        super().symmetric_difference_update(other)
        self._note()
        self.compact()

    def discard(self, element: object) -> None:
        super().discard(cast("str", element))
        self.compact()

    def remove(self, element: str) -> None:
        super().remove(element)
        self.compact()

    def pop(self) -> str:
        element = super().pop()
        self.compact()
        return element

    def clear(self) -> None:
        super().clear()
        self.peak = 0

    def difference_update(self, *others: Iterable[object]) -> None:
        super().difference_update(*others)
        self.compact()

    def __isub__(self, other: AbstractSet[object]) -> _CompactingSet:
        super().__isub__(other)
        self.compact()
        return self

    def intersection_update(self, *others: Iterable[object]) -> None:
        super().intersection_update(*others)
        self.compact()

    def __iand__(self, other: AbstractSet[object]) -> _CompactingSet:
        self.intersection_update(other)
        return self

    def compact(self) -> None:
        """Release a spike-sized table once the size falls below half the peak."""
        size = len(self)
        if size == 0:
            self.clear()
        elif size * 2 < self.peak:
            members = set(self)
            self.clear()
            self.update(members)


class _ShadowIndexes:
    """The reconcile indexes, maintained incrementally at every transition.

    P3b2a installed these as shadow mirrors behind a loud scan-comparing
    oracle; the P3b2b cutover made them the live path and removed the oracle,
    so the reconcile scans read these directly.

    Placeholder presence, residency, checkpoint membership and content-order
    adjacency are re-derived per unit at the transition that moves them
    (admission, eviction, restore, protection, removal). Row heights and pin
    reasons are read live by the budget facet instead: layout moves heights,
    and the presenter and the app write ``pin_reasons`` outside the registry,
    so caching either would need a notification the registry does not receive.

    The retained/live facet cannot come from registry transitions alone --
    retained widgets, spinners and streaming buffers mutate between
    reconciles without the registry hearing about it. It is a dirty-marked
    cache: units carry a mark (placement transitions, a changed retained set,
    the active turn, and every unit that read live last time), and the
    presenter re-reads and commits only the marked units.
    """

    def __init__(self) -> None:
        # Placeholder presence: units owning at least one Stream placeholder.
        # A compacting set like live_dirty: the restore scan's fallback sorts
        # this set, so a placeholder spike must release its table once the
        # restore settles most of them.
        self.placeholders: _CompactingSet = _CompactingSet()
        # The placeholder set in content order, or None when stale: rebuilt by
        # the next placeholder_unit_ids() query, never by a per-tick re-sort.
        self.placeholders_order: list[str] | None = None
        # Residency: units whose placement roots are mounted. A compacting
        # set like live_dirty: a mounted-membership spike must release its
        # table once eviction settles residency back down.
        self.mounted: _CompactingSet = _CompactingSet()
        # Units whose retained entries include a checkpoint, and those among
        # them with a mounted content neighbour.
        self.checkpoints: set[str] = set()
        # Compacted like live: the presenter reads frozenset(neighbors) every
        # protection tick, so a checkpoint-dense spike must release its table
        # once the settle leaves only the resident adjacency.
        self.neighbors: _CompactingSet = _CompactingSet()
        # Validated retained/live cache: committed observations, the units
        # whose value must be re-read, and the retained-widget signature the
        # observations were taken under. live is compacted because the
        # presenter reads frozenset(live) every protection tick: a live spike
        # that settles back to the resident tail must release its table.
        self.live: _CompactingSet = _CompactingSet()
        self.live_dirty: _CompactingSet = _CompactingSet()
        self.retained: frozenset[int] = frozenset()
        # Content order, the adjacency the neighbour answer is derived from.
        self.before: dict[str, str | None] = {}
        self.after: dict[str, str | None] = {}
        self.head: str | None = None
        self.tail: str | None = None

    def link_append(self, unit_id: str) -> None:
        self.before[unit_id] = self.tail
        self.after[unit_id] = None
        if self.tail is None:
            self.head = unit_id
        else:
            self.after[self.tail] = unit_id
        self.tail = unit_id

    def link_prepend(self, unit_ids: Sequence[str]) -> None:
        """Move an appended block to the head of the content order."""
        if not unit_ids:
            return
        first, last = unit_ids[0], unit_ids[-1]
        before = self.before[first]
        if before is None:
            self.head = None
            self.tail = None
        else:
            self.after[before] = None
            self.tail = before
        head = self.head
        self.before[first] = None
        self.after[last] = head
        if head is None:
            self.tail = last
        else:
            self.before[head] = last
        self.head = first

    def link_rename(self, previous_id: str, public_id: str) -> None:
        before = self.before.pop(previous_id, None)
        after = self.after.pop(previous_id, None)
        self.before[public_id] = before
        self.after[public_id] = after
        if before is None:
            self.head = public_id
        else:
            self.after[before] = public_id
        if after is None:
            self.tail = public_id
        else:
            self.before[after] = public_id

    def link_unlink(self, unit_id: str) -> None:
        if unit_id not in self.before:
            return
        before = self.before.pop(unit_id)
        after = self.after.pop(unit_id)
        if before is None:
            self.head = after
        else:
            self.after[before] = after
        if after is None:
            self.tail = before
        else:
            self.before[after] = before

    def link_reslot(self, unit_id: str, before: str | None, after: str | None) -> None:
        """Re-insert a unit between two neighbours after a content-order move."""
        self.link_unlink(unit_id)
        self.before[unit_id] = before
        self.after[unit_id] = after
        if before is None:
            self.head = unit_id
        else:
            self.after[before] = unit_id
        if after is None:
            self.tail = unit_id
        else:
            self.before[after] = unit_id

    def clear(self) -> None:
        self.placeholders.clear()
        self.placeholders_order = None
        self.mounted.clear()
        self.checkpoints.clear()
        self.neighbors.clear()
        self.live.clear()
        self.live_dirty.clear()
        self.retained = frozenset()
        self.before.clear()
        self.after.clear()
        self.head = None
        self.tail = None


class TranscriptWindow:
    def __init__(
        self,
        *,
        tools_collapsed: bool = True,
        expansion_state: ToolGroupExpansionState | None = None,
        entry_expansion_state: EntryExpansionState | None = None,
    ) -> None:
        self.tools_collapsed = tools_collapsed
        self.expansion_state = expansion_state
        self.entry_expansion_state = entry_expansion_state
        self.unit_ids: list[str] = []
        self.units: dict[str, TranscriptUnit] = {}
        self._entry_to_unit: dict[str, str] = {}
        self._object_ids: dict[int, tuple[ref[PublicHistoryEntry], str]] = {}
        self._admitted_indices: dict[str, int] = {}
        self._occupied_indices: set[int] = set()
        self._index_start = 0
        self._index_end = -1
        self._geometry_width: int | None = None
        # Content-ordered unit ids that currently hold a Stream placement. The
        # index stores order only: placement regions carry the scroll offset,
        # so rows are read live at query time and scrolling never invalidates it.
        self._placements: list[str] = []
        # Units whose placement was mounted in this frame and has no measured
        # geometry until the next layout pass: its region reads as row 0, which
        # breaks the placement index's screen ordering.
        self._unmeasured_placements: set[str] = set()
        self._evicting: set[str] = set()
        # Reconcile indexes (P3b2a installed them behind an oracle, P3b2b cut
        # the scans over to them): maintained at every transition and read by
        # the live reconcile path.
        self._shadow = _ShadowIndexes()
        self._reconcile_lock = asyncio.Lock()
        self._reconcile_task: asyncio.Task[None] | None = None
        self._reconcile_active = False
        self._reconcile_pending = False
        self._reconcile_latest: Callable[[], Awaitable[None]] | None = None
        self._scroll_revision = 0
        self._session_generation = 0
        self._next_fallback = 0
        self.geometry_version = 0
        self._group_revision = expansion_state.revision if expansion_state else 0
        self._group_reset_generation = (
            expansion_state.reset_generation if expansion_state else 0
        )
        self._entry_revision = (
            entry_expansion_state.revision if entry_expansion_state else 0
        )
        self._entry_reset_generation = (
            entry_expansion_state.reset_generation if entry_expansion_state else 0
        )
        self._group_default = (
            expansion_state.default_collapsed if expansion_state else True
        )
        self._entry_default = (
            entry_expansion_state.default_collapsed if entry_expansion_state else True
        )

    @property
    def admitted_start_index(self) -> int:
        return self._index_start

    @property
    def admitted_end_index(self) -> int:
        return self._index_end + 1

    # -- Reconcile indexes --------------------------------------------------
    #
    # P3b2a proved these against the scans with a loud oracle; the P3b2b
    # cutover removed the oracle and made the indexes the answers the
    # reconcile path reads.

    def _shadow_unit(self, unit: TranscriptUnit) -> None:
        """Re-derive one unit's residency and placeholder entries."""
        shadow = self._shadow
        if unit.mounted_roots:
            shadow.mounted.add(unit.id)
        else:
            shadow.mounted.discard(unit.id)
        if unit.placeholders:
            if unit.id not in shadow.placeholders:
                shadow.placeholders.add(unit.id)
                shadow.placeholders_order = None
        elif unit.id in shadow.placeholders:
            shadow.placeholders.discard(unit.id)
            shadow.placeholders_order = None
        # A placement transition makes any cached liveness stale.
        shadow.live_dirty.add(unit.id)
        self._shadow_neighbors(unit.id)

    def _shadow_entries(self, unit: TranscriptUnit) -> None:
        """Re-derive one unit's checkpoint membership from its entries."""
        shadow = self._shadow
        if any(isinstance(entry, PublicCheckpointEntry) for entry in unit.entries):
            shadow.checkpoints.add(unit.id)
        else:
            shadow.checkpoints.discard(unit.id)
        self._shadow_neighbors(unit.id)

    def _shadow_admitted_entry(
        self, unit: TranscriptUnit, entry: PublicHistoryEntry
    ) -> None:
        """Record an appended member's effect on checkpoint membership."""
        if isinstance(entry, PublicCheckpointEntry):
            self._shadow.checkpoints.add(unit.id)
            self._shadow_neighbors(unit.id)

    def _shadow_replaced_entry(
        self,
        unit: TranscriptUnit,
        previous: PublicHistoryEntry,
        entry: PublicHistoryEntry,
    ) -> None:
        """Re-derive checkpoint membership when a replacement flips the type."""
        if isinstance(previous, PublicCheckpointEntry) != isinstance(
            entry, PublicCheckpointEntry
        ):
            self._shadow_entries(unit)

    def _shadow_neighbors(self, unit_id: str) -> None:
        """Re-derive neighbour membership for a unit and its two neighbours.

        Only this triple can move when ``unit_id`` mounts, unmounts, joins the
        checkpoint set, or shifts in the content order.
        """
        shadow = self._shadow
        for candidate in (
            shadow.before.get(unit_id),
            unit_id,
            shadow.after.get(unit_id),
        ):
            if candidate is None:
                continue
            if candidate not in shadow.checkpoints:
                # Leaving the checkpoint set ends neighbour protection too.
                shadow.neighbors.discard(candidate)
                continue
            adjacent = (shadow.before.get(candidate), shadow.after.get(candidate))
            if any(item is not None and item in shadow.mounted for item in adjacent):
                shadow.neighbors.add(candidate)
            else:
                shadow.neighbors.discard(candidate)

    def _shadow_forget(self, unit_id: str) -> None:
        """Drop a removed unit from every facet and relink its neighbours."""
        shadow = self._shadow
        before, after = shadow.before.get(unit_id), shadow.after.get(unit_id)
        if unit_id in shadow.placeholders:
            shadow.placeholders.discard(unit_id)
            shadow.placeholders_order = None
        shadow.mounted.discard(unit_id)
        shadow.checkpoints.discard(unit_id)
        shadow.neighbors.discard(unit_id)
        shadow.live.discard(unit_id)
        shadow.live_dirty.discard(unit_id)
        self._unmeasured_placements.discard(unit_id)
        shadow.link_unlink(unit_id)
        for candidate in (before, after):
            if candidate is not None:
                self._shadow_neighbors(candidate)

    def _shadow_rename(self, previous_id: str, public_id: str) -> None:
        """Carry every facet across an identity adoption."""
        shadow = self._shadow
        # Membership must be read before the loop renames the id away, or the
        # cached content order keeps serving the pre-rename id.
        had_placeholders = previous_id in shadow.placeholders
        for members in (
            shadow.placeholders,
            shadow.mounted,
            shadow.checkpoints,
            shadow.neighbors,
            shadow.live,
            shadow.live_dirty,
        ):
            if previous_id in members:
                members.discard(previous_id)
                members.add(public_id)
        if had_placeholders:
            shadow.placeholders_order = None
        if previous_id in self._unmeasured_placements:
            # The mark names the unit's placeholder, so it follows the id; left
            # behind it names no unit and the pending check drops the guard.
            self._unmeasured_placements.discard(previous_id)
            self._unmeasured_placements.add(public_id)
        shadow.link_rename(previous_id, public_id)
        self._shadow_neighbors(public_id)

    def _begin_live_scan(self, retained: Iterable[object]) -> None:
        """Open a live-scan tick, dirtying placements retention could move.

        The presenter's existing reads feed the marking: the retained set comes
        from widget state, and a changed set invalidates every mounted unit,
        because any of them could now hold a retained descendant.
        """
        shadow = self._shadow
        signature = frozenset(id(widget) for widget in retained)
        if signature != shadow.retained:
            shadow.retained = signature
            shadow.live_dirty |= shadow.mounted

    def _mark_live_dirty(self, unit_ids: Iterable[str]) -> None:
        """Request a re-read of these placements' liveness."""
        self._shadow.live_dirty.update(unit_ids)

    def _invalidate_live_index(self) -> None:
        """Force a re-read of every mounted unit.

        A tick that skipped the widget walk -- or that evaluated no liveness at
        all -- cannot claim the cached observations still hold.
        """
        shadow = self._shadow
        shadow.live_dirty |= shadow.mounted

    def _observe_live(self, unit_id: str, live: bool) -> None:
        """Answer the index's re-read request with one observed liveness value.

        A unit the index did not mark dirty keeps its cached value: that is
        what makes the cache meaningful, and what the verification compares
        against the walk.
        """
        shadow = self._shadow
        if unit_id not in shadow.live_dirty:
            return
        if live:
            shadow.live.add(unit_id)
        else:
            shadow.live.discard(unit_id)
        shadow.live_dirty.discard(unit_id)

    def _commit_live_scan(self) -> None:
        """Re-dirty live units: their flags keep mutating after the walk."""
        shadow = self._shadow
        shadow.live_dirty |= shadow.live
        # The re-dirtied live members keep the set nonempty, so an empty check
        # alone cannot release a spike-sized table; compact() rehashes when
        # the marks settle below half the high-water mark, keeping the
        # presenter's sorted(live_dirty) scan proportional to live membership
        # instead of the historical peak.
        shadow.live_dirty.compact()

    # -- Index answers read by the reconcile path (P3b2b cutover) ---------

    @property
    def has_placeholders(self) -> bool:
        """Whether any unit currently owns a Stream placeholder."""
        return bool(self._shadow.placeholders)

    def placeholder_unit_ids(self) -> list[str]:
        """Placeholder-owning units in content order, for the restore scan.

        The order is cached on the shadow index and rebuilt only when a
        transition moves membership or a member's start index, so an
        unchanged reconcile tick does not re-sort the placeholder set.
        """
        shadow = self._shadow
        order = shadow.placeholders_order
        if order is None:
            order = sorted(
                shadow.placeholders, key=lambda unit_id: self.units[unit_id].start_index
            )
            shadow.placeholders_order = order
        return list(order)

    def budgeted_rows(
        self, high_mark: int, *, exclude_placeholders: bool = False
    ) -> int:
        """Sum the mounted-row budget over the resident index.

        Row heights and pin reasons are read live: layout moves heights and
        the presenter and the app write ``pin_reasons`` outside the registry.
        Only resident units can count, so the sum is bounded by residency,
        not registry size.
        """
        shadow = self._shadow
        rows = 0
        for unit_id in shadow.mounted:
            unit = self.units.get(unit_id)
            if unit is None or not unit.mounted_roots:
                continue
            if exclude_placeholders and unit_id in shadow.placeholders:
                continue
            if unit.pin_reasons:
                continue
            roots = unit.mounted_roots
            height = roots[-1].region.bottom - roots[0].region.y
            if height > high_mark:
                continue
            rows += height
        return rows

    def _entry_id(self, entry: PublicHistoryEntry) -> str:
        public_id = entry.id
        object_id = id(entry)
        recorded = self._object_ids.get(object_id)
        previous = (
            recorded[1] if recorded is not None and recorded[0]() is entry else None
        )

        records = self._object_ids

        def forget(dead: ref[PublicHistoryEntry]) -> None:
            if (record := records.get(object_id)) and record[0] is dead:
                del records[object_id]

        if public_id:
            if previous and previous != public_id and previous in self._entry_to_unit:
                self.adopt_identity(previous, public_id)
            self._object_ids[object_id] = (ref(entry, forget), public_id)
            return public_id
        if previous is None:
            self._next_fallback += 1
            previous = f"transient-{self._next_fallback}"
            self._object_ids[object_id] = (ref(entry, forget), previous)
        return previous

    def adopt_identity(self, previous_id: str, public_id: str) -> None:
        if previous_id == public_id:
            return
        if public_id in self._entry_to_unit:
            raise ValueError(f"entry already admitted: {public_id}")
        unit_id = self._entry_to_unit.pop(previous_id)
        unit = self.units[unit_id]
        offset = unit.member_entry_ids.index(previous_id)
        unit.member_entry_ids[offset] = public_id
        self._entry_to_unit[public_id] = unit_id
        if previous_id in self._admitted_indices:
            self._admitted_indices[public_id] = self._admitted_indices.pop(previous_id)
        if unit_id == previous_id:
            self.units[public_id] = self.units.pop(previous_id)
            unit.id = public_id
            self.unit_ids[self.unit_ids.index(previous_id)] = public_id
            self._entry_to_unit[public_id] = public_id
            self._rename_placement(previous_id, public_id)
            self._shadow_rename(previous_id, public_id)
            for member_id in unit.member_entry_ids:
                self._entry_to_unit[member_id] = public_id
        if (
            unit.group_key in {ToolGroupKey(previous_id), ToolGroupKey("")}
            and offset == 0
        ):
            unit.group_key = ToolGroupKey(public_id)
        self.geometry_version += 1
        unit.geometry_version = self.geometry_version
        unit.content_height = None

    def _invalidate(self, unit: TranscriptUnit) -> None:
        if unit.restoring:
            unit.restore_generation += 1
        self.geometry_version += 1
        unit.geometry_version = self.geometry_version
        unit.content_height = None
        for placeholder in unit.placeholders:
            placeholder.invalidate_geometry()

    def _check_batch(self, ids: list[str], start_index: int) -> None:
        occupied = self._occupied_indices
        prefix = (
            bool(occupied)
            and start_index == self._index_start
            and all(key not in self._admitted_indices for key in ids)
        )
        if len(ids) != len(set(ids)) or (
            not prefix
            and any(
                key not in self._admitted_indices and start_index + offset in occupied
                for offset, key in enumerate(ids)
            )
        ):
            raise ValueError("interleaved transcript admission")
        known = [
            (offset, self._admitted_indices[key])
            for offset, key in enumerate(ids)
            if key in self._admitted_indices
        ]
        if known:
            if any(start_index + offset != old for offset, old in known):
                raise ValueError("interleaved transcript admission")
            first, last = known[0][0], known[-1][0]
            if any(
                key not in self._admitted_indices for key in ids[first : last + 1]
            ) or any(
                old - previous_old != offset - previous_offset
                for (previous_offset, previous_old), (offset, old) in zip(
                    known, known[1:], strict=False
                )
            ):
                raise ValueError("interleaved transcript admission")
        elif self._admitted_indices and ids:
            first = self._index_start
            last = self._index_end
            if not (
                start_index + len(ids) <= first
                or start_index == first  # _shift_for_prefix rebases the old timeline.
                or start_index == last + 1
            ):
                raise ValueError("interleaved transcript admission")

    def _shift_for_prefix(self, ids: list[str], start_index: int) -> int:
        """Re-base the recorded timeline onto a new prefix; return the shift."""
        if not (
            ids
            and self._admitted_indices
            and start_index == self._index_start
            and all(key not in self._admitted_indices for key in ids)
        ):
            return 0
        # New prefix displaces the previously recorded timeline positions.
        shift = len(ids)
        for key in self._admitted_indices:
            self._admitted_indices[key] += shift
        self._occupied_indices = {index + shift for index in self._occupied_indices}
        self._index_start += shift
        self._index_end += shift
        for unit in self.units.values():
            unit.start_index += shift
            # Index re-base changes ordering/status, not measured placement height.
            self.geometry_version += 1
            unit.geometry_version = self.geometry_version
        return shift

    def _sync_width(self, geometry_width: int | None) -> None:
        if geometry_width is not None and geometry_width != self._geometry_width:
            self._geometry_width = geometry_width
            for unit in self.units.values():
                if unit.geometry_width != geometry_width:
                    unit.geometry_width = geometry_width
                    self._invalidate(unit)

    def _record_index(self, entry_id: str, index: int) -> None:
        self._admitted_indices[entry_id] = index
        self._occupied_indices.add(index)
        if len(self._occupied_indices) == 1:
            self._index_start = self._index_end = index
        else:
            self._index_start = min(self._index_start, index)
            self._index_end = max(self._index_end, index)

    def _place_new_units(self, prior_units: int, prior_first: int | None) -> None:
        if prior_first is None or not self.unit_ids[prior_units:]:
            return
        new_ids = self.unit_ids[prior_units:]
        del self.unit_ids[prior_units:]
        if self.units[new_ids[0]].start_index < prior_first:
            self.unit_ids[:0] = new_ids
            self._shadow.link_prepend(new_ids)
        else:
            self.unit_ids.extend(new_ids)
        # Only the block's edges neighbour units outside it; interior
        # adjacencies were linked as the units were admitted.
        self._shadow_neighbors(new_ids[0])
        self._shadow_neighbors(new_ids[-1])

    def _initial_group(
        self, batch: Sequence[PublicHistoryEntry], start_index: int
    ) -> TranscriptUnit | None:
        if (
            self.unit_ids
            and start_index == self.admitted_end_index
            and self.units[self.unit_ids[-1]].group_key is not None
            and batch
            and entry_keeps_tool_group(batch[0])
        ):
            return self.units[self.unit_ids[-1]]
        return None

    def _reslot_unit_ids(self, unit_id: str, start_index: int) -> None:
        """Re-insert a moved unit in ``unit_ids`` at its start_index position.

        ``unit_ids`` stays ordered by start_index -- the presenter's
        active-turn bisect resolves against that order -- so a member whose
        start index moves re-slots here just as it does in the placement index.
        The shadow adjacency mirrors ``unit_ids``, so the moved unit is
        re-linked at its new slot too; leaving the old links in place would
        keep neighbour protection on the stale adjacency.
        """
        shadow = self._shadow
        old_before, old_after = shadow.before.get(unit_id), shadow.after.get(unit_id)
        ids = self.unit_ids
        ids.remove(unit_id)
        low, high = 0, len(ids)
        while low < high:
            middle = (low + high) // 2
            if self.units[ids[middle]].start_index <= start_index:
                low = middle + 1
            else:
                high = middle
        ids.insert(low, unit_id)
        new_before = ids[low - 1] if low > 0 else None
        new_after = ids[low + 1] if low + 1 < len(ids) else None
        shadow.link_reslot(unit_id, new_before, new_after)
        for candidate in (old_before, old_after, new_before, new_after, unit_id):
            if candidate is not None:
                self._shadow_neighbors(candidate)

    def _replace_member(
        self, unit: TranscriptUnit, entry_id: str, entry: PublicHistoryEntry, index: int
    ) -> None:
        """Re-admit an admitted member: retained data, fingerprints, start index."""
        offset = unit.member_entry_ids.index(entry_id)
        previous = unit.entries[offset]
        unit.entries[offset] = entry
        self._shadow_replaced_entry(unit, previous, entry)
        fingerprint = fingerprint_entry(entry)
        if unit.entry_fingerprints[offset] != fingerprint:
            unit.entry_fingerprints[offset] = fingerprint
            self._invalidate(unit)
        new_start = index - offset
        if unit.start_index != new_start:
            # A moved start index re-orders content order: re-slot the unit in
            # the placement index so the anchor bisect keeps resolving by it,
            # and in unit_ids so the presenter's pin bisect stays valid.
            was_placed = unit.id in self._placements
            if was_placed:
                self._discard_placement(unit.id)
            unit.start_index = new_start
            self._invalidate(unit)
            if unit.placeholders:
                self._shadow.placeholders_order = None
            self._reslot_unit_ids(unit.id, new_start)
            if was_placed:
                self._index_placement(unit.id)

    def admit(
        self,
        batch: Sequence[PublicHistoryEntry],
        *,
        start_index: int,
        geometry_width: int | None = None,
        extend_group: bool = False,
    ) -> int:
        """Admit a contiguous timeline batch; new entries must extend either edge.

        Pass the parent Stream's outer ``size.width`` as ``geometry_width``;
        this is not the placeholder's Stream content-height width baseline.

        Returns the prefix shift: how many timeline positions a re-admitted
        whole-history prefix displaced the previously recorded ones by, so the
        caller can re-base index-keyed state (the app's active-turn start).
        """
        ids = [self._entry_id(entry) for entry in batch]
        self._check_batch(ids, start_index)
        self._sync_width(geometry_width)
        prefix_shift = self._shift_for_prefix(ids, start_index)
        prior_units = len(self.unit_ids)
        prior_first = self.units[self.unit_ids[0]].start_index if prior_units else None
        current = self._initial_group(batch, start_index) if extend_group else None
        for index, (entry, entry_id) in enumerate(
            zip(batch, ids, strict=True), start_index
        ):
            self._record_index(entry_id, index)
            grouped = entry_keeps_tool_group(entry)
            if not grouped:
                current = None
            if entry_id in self._entry_to_unit:
                existing = self.units[self._entry_to_unit[entry_id]]
                self._replace_member(existing, entry_id, entry, index)
                if grouped:
                    current = existing
                continue
            if grouped:
                if current is None:
                    if not isinstance(entry, PublicEffectEntry | PublicReasoningEntry):
                        continue
                    current = TranscriptUnit(
                        id=entry_id,
                        member_entry_ids=[],
                        entries=[],
                        start_index=index,
                        group_key=ToolGroupKey(entry.id),
                        geometry_width=geometry_width,
                    )
                    self.units[entry_id] = current
                    self.unit_ids.append(entry_id)
                    self._shadow.link_append(entry_id)
                unit = current
            else:
                if not history_entry_renders_widget(entry):
                    continue
                unit = TranscriptUnit(
                    entry_id, [], [], index, geometry_width=geometry_width
                )
                self.units[entry_id] = unit
                self.unit_ids.append(entry_id)
                self._shadow.link_append(entry_id)
            unit.member_entry_ids.append(entry_id)
            unit.entries.append(entry)
            unit.entry_fingerprints.append(fingerprint_entry(entry))
            self._entry_to_unit[entry_id] = unit.id
            unit.start_index = min(unit.start_index, index)
            if geometry_width is not None:
                unit.geometry_width = geometry_width
            self._shadow_admitted_entry(unit, entry)
            self._invalidate(unit)
        self._place_new_units(prior_units, prior_first)
        return prefix_shift

    def sync_expansion(self) -> set[str]:
        affected: set[str] = set()
        if self.expansion_state is not None:
            state = self.expansion_state
            if state.revision != self._group_revision:
                changed = state.changed_keys_since(self._group_revision)
                affected.update(
                    unit.id for unit in self.units.values() if unit.group_key in changed
                )
                if (
                    state.reset_generation != self._group_reset_generation
                    or state.default_collapsed != self._group_default
                ):
                    affected.update(
                        unit.id for unit in self.units.values() if unit.group_key
                    )
                self._group_reset_generation = state.reset_generation
                self._group_default = state.default_collapsed
                self._group_revision = state.revision
        if self.entry_expansion_state is not None:
            state = self.entry_expansion_state
            if state.revision != self._entry_revision:
                changed = state.changed_ids_since(self._entry_revision)
                affected.update(
                    self._entry_to_unit[key]
                    for key in changed
                    if key in self._entry_to_unit
                )
                if (
                    state.reset_generation != self._entry_reset_generation
                    or state.default_collapsed != self._entry_default
                ):
                    affected.update(self.unit_ids)
                self._entry_reset_generation = state.reset_generation
                self._entry_default = state.default_collapsed
                self._entry_revision = state.revision
        if affected:
            self.geometry_version += 1
        for unit_id in affected:
            unit = self.units[unit_id]
            if unit.restoring:
                unit.restore_generation += 1
            unit.content_height = None
            unit.geometry_version = self.geometry_version
            for placeholder in unit.placeholders:
                placeholder.invalidate_geometry()
        return affected

    def reset(self) -> None:
        self._session_generation += 1
        self._reconcile_active = False
        self._reconcile_pending = False
        self._reconcile_latest = None
        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            self._reconcile_task = None
        for unit in self.units.values():
            unit.restore_generation += 1
            self._discard_placeholders(unit)
        self.unit_ids.clear()
        self.units.clear()
        self._entry_to_unit.clear()
        self._object_ids.clear()
        self._admitted_indices.clear()
        self._occupied_indices.clear()
        self._placements.clear()
        self._index_start = 0
        self._index_end = -1
        self._geometry_width = None
        self._evicting.clear()
        self._unmeasured_placements.clear()
        self._shadow.clear()
        self.geometry_version += 1
        self._group_revision = (
            self.expansion_state.revision if self.expansion_state else 0
        )
        self._group_reset_generation = (
            self.expansion_state.reset_generation if self.expansion_state else 0
        )
        self._entry_revision = (
            self.entry_expansion_state.revision if self.entry_expansion_state else 0
        )
        self._entry_reset_generation = (
            self.entry_expansion_state.reset_generation
            if self.entry_expansion_state
            else 0
        )
        self._group_default = (
            self.expansion_state.default_collapsed if self.expansion_state else True
        )
        self._entry_default = (
            self.entry_expansion_state.default_collapsed
            if self.entry_expansion_state
            else True
        )

    def update_entry(self, entry: PublicHistoryEntry) -> None:
        """Refresh retained data even when its placement is currently evicted."""
        unit_id = self._entry_to_unit.get(entry.id)
        if unit_id is None:
            return
        unit = self.units[unit_id]
        offset = unit.member_entry_ids.index(entry.id)
        fingerprint = fingerprint_entry(entry)
        if unit.entry_fingerprints[offset] != fingerprint:
            previous = unit.entries[offset]
            unit.entries[offset] = entry
            self._shadow_replaced_entry(unit, previous, entry)
            unit.entry_fingerprints[offset] = fingerprint
            self._invalidate(unit)

    def admit_reused_root(
        self, entry: PublicMessageEntry, root: Widget, *, index: int
    ) -> bool:
        """Extend the last placement when a retry continues its assistant root."""
        if not self.unit_ids or index != self.admitted_end_index:
            return False
        unit = self.units[self.unit_ids[-1]]
        if unit.group_key is not None or root not in unit.mounted_roots:
            return False
        self._check_batch([entry.id], index)
        self._record_index(entry.id, index)
        unit.member_entry_ids.append(entry.id)
        unit.entries.append(entry)
        unit.entry_fingerprints.append(fingerprint_entry(entry))
        self._entry_to_unit[entry.id] = unit.id
        self._shadow_admitted_entry(unit, entry)
        self._invalidate(unit)
        return True

    def has_admitted_entry(self, entry_id: str) -> bool:
        """Report registry membership, including units whose placement is evicted."""
        return entry_id in self._entry_to_unit

    def mounted_entry_widgets(self, entry_id: str) -> list[Widget]:
        """Resolve retained DOM via unit membership, never revive evicted/removed rows."""
        unit_id = self._entry_to_unit.get(entry_id)
        if unit_id is None:
            return []
        unit = self.units[unit_id]
        if unit.placeholders:
            return []
        roots = [
            root
            for root in unit.mounted_roots
            if root.is_mounted and root.parent is not None
        ]
        if unit.group_key is None:
            return roots
        return [
            child
            for root in roots
            if isinstance(root, ToolGroup)
            for child in root.content_container.children
            if (isinstance(child, ToolCallMessage) and child.tool_call_id == entry_id)
            or (isinstance(child, ToolResultMessage) and child._entry.id == entry_id)
        ]

    def register_mounted(self, unit_id: str, roots: Sequence[Widget]) -> None:
        """Register mounted roots; rejects units still owning placeholders.

        ``content_height`` sums root.region.height, excluding collapsed margins
        between placements; it is not the unit's DOM footprint.
        """
        unit = self.units[unit_id]
        if (
            unit_id in self._evicting
            or unit.placeholders
            or not roots
            or any(root.parent is not roots[0].parent for root in roots)
        ):
            raise ValueError("unit roots must share a Stream parent")
        parent = roots[0].parent
        if not isinstance(parent, Widget):
            raise ValueError("unit roots must be mounted in a Widget")
        positions = [list(parent.children).index(root) for root in roots]
        if positions != list(range(positions[0], positions[0] + len(roots))):
            raise ValueError("unit roots must be contiguous")
        # Roots already registered here are restorations: only a unit entering
        # the DOM for the first time joins the placement index.
        placed = bool(unit.mounted_roots)
        unit.mounted_roots = list(roots)
        unit.content_height = sum(root.region.height for root in roots)
        unit.geometry_width = parent.size.width
        if not placed:
            self._index_placement(unit_id)
        self._shadow_unit(unit)

    def register_restored(self, unit_id: str, roots: Sequence[Widget]) -> None:
        """Register restored roots after their placeholders leave the Stream DOM."""
        unit = self.units[unit_id]
        if any(placeholder.parent is not None for placeholder in unit.placeholders):
            raise ValueError("unit placeholders must be removed before restore")
        placeholders = unit.placeholders
        unit.placeholders = []
        self._shadow_unit(unit)
        try:
            self.register_mounted(unit_id, roots)
        except BaseException:
            unit.placeholders = placeholders
            self._shadow_unit(unit)
            raise
        self.geometry_version += 1

    def eviction_candidates(
        self,
        viewport_top: int,
        viewport_height: int,
        *,
        low_mark: int = 1000,
        high_mark: int = 1500,
        selected: frozenset[str] = frozenset(),
        live: frozenset[str] = frozenset(),
        compaction_neighbors: frozenset[str] = frozenset(),
    ) -> list[str]:
        """Choose distant mounted units using screen-space viewport coordinates.

        ``viewport_top`` is the screen-space top of the Stream's visible area
        (not its scroll offset); ``viewport_height`` is that area's height.
        Root ``region.y`` coordinates already include the scroll offset.

        The resident index bounds the walk: only mounted units can be
        budgeted or evicted, so the scan is over residency, not registry
        size. Row heights and pin reasons are still read live per resident
        unit, because layout and the external pin writers move them without
        a registry transition.
        """
        if not 0 <= low_mark < high_mark:
            raise ValueError("expected 0 <= low_mark < high_mark")
        if self._evicting:
            # An awaited DOM replacement is not yet reflected in the registry.
            return []
        overscan_top = viewport_top - viewport_height
        overscan_bottom = viewport_top + 2 * viewport_height
        candidates: list[tuple[int, str]] = []
        budgeted_rows = 0
        for unit_id in self._shadow.mounted:
            unit = self.units.get(unit_id)
            if unit is None or not unit.mounted_roots:
                continue
            if unit.placeholders:
                continue
            roots = unit.mounted_roots
            top = roots[0].region.y
            bottom = roots[-1].region.bottom
            height = bottom - top
            if height > high_mark or unit.pin_reasons:
                # Frozen and oversized placements are outside the mounted-row budget.
                continue
            budgeted_rows += height
            if (
                unit_id in self._evicting
                or unit_id in selected
                or unit_id in live
                or unit_id in compaction_neighbors
            ):
                # These mounted rows count, but cannot be evicted.
                continue
            if bottom <= overscan_top or top >= overscan_bottom:
                distance = max(overscan_top - bottom, top - overscan_bottom)
                candidates.append((distance, unit_id))
        if budgeted_rows <= high_mark:
            return []
        evict: list[str] = []
        for _, unit_id in sorted(candidates, reverse=True):
            evict.append(unit_id)
            roots = self.units[unit_id].mounted_roots
            budgeted_rows -= roots[-1].region.bottom - roots[0].region.y
            if budgeted_rows <= low_mark:
                break
        return evict

    async def evict_unit(self, unit_id: str) -> list[PlacementPlaceholder]:
        unit = self.units[unit_id]
        roots = unit.mounted_roots
        if (
            unit_id in self._evicting
            or not roots
            or unit.placeholders
            or unit.pin_reasons
        ):
            raise ValueError("unit is not evictable")
        parent = roots[0].parent
        if not isinstance(parent, Widget) or any(
            root.parent is not parent for root in roots
        ):
            raise ValueError("unit roots must share a parent")
        placeholders = [PlacementPlaceholder.from_widget(root) for root in roots]
        self._evicting.add(unit_id)
        try:
            with parent.app.batch_update():
                try:
                    await parent.mount_all(placeholders, before=roots[0])
                    for root in roots:
                        await root.remove()
                except BaseException as original:
                    # Rollback owns its own task: repeated cancellation of the caller
                    # must not interrupt the re-mount/remove sequence halfway through.
                    async def restore() -> None:
                        missing = [root for root in roots if root.parent is not parent]
                        if missing:
                            anchor = next(
                                (root for root in roots if root.parent is parent),
                                next(
                                    (p for p in placeholders if p.parent is parent),
                                    None,
                                ),
                            )
                            if anchor is not None:
                                await parent.mount_all(missing, before=anchor)
                            else:
                                await parent.mount_all(missing)
                        for placeholder in placeholders:
                            if placeholder.parent is parent:
                                await placeholder.remove()

                    rollback = asyncio.create_task(restore())
                    while not rollback.done():
                        try:
                            await asyncio.shield(rollback)
                        except asyncio.CancelledError:
                            # Preserve the original failure after restoration finishes.
                            pass
                        except BaseException as failure:
                            raise failure from original
                    failure = rollback.exception()
                    if failure is not None:
                        raise failure from original
                    raise
            unit.mounted_roots = []
            unit.placeholders = placeholders
            self._shadow_unit(unit)
            self.geometry_version += 1
            return placeholders
        finally:
            self._evicting.discard(unit_id)
            # Either outcome leaves freshly mounted widgets -- the placeholders,
            # or the roots a rollback re-mounted -- without rows until layout.
            self._mark_unmeasured(unit)

    @staticmethod
    def _discard_placeholders(unit: TranscriptUnit) -> None:
        for placeholder in unit.placeholders:
            if placeholder._reserved:
                placeholder.release()
            if placeholder.parent is not None:

                async def discard(widget: PlacementPlaceholder = placeholder) -> None:
                    try:
                        if widget.parent is not None:
                            await widget.remove()
                    except Exception:
                        # Session teardown may remove the Stream concurrently.
                        pass

                asyncio.create_task(discard())
        unit.placeholders.clear()

    def remove_unit(self, unit_id: str) -> None:
        unit = self.units.pop(unit_id)
        unit.restore_generation += 1
        self._discard_placeholders(unit)
        self.unit_ids.remove(unit_id)
        self._shadow_forget(unit_id)
        self._discard_placement(unit_id)
        for member_id in unit.member_entry_ids:
            self._entry_to_unit.pop(member_id, None)
            index = self._admitted_indices.pop(member_id, None)
            if index is not None:
                self._occupied_indices.discard(index)
        if self._occupied_indices:
            self._index_start = min(self._occupied_indices)
            self._index_end = max(self._occupied_indices)
        else:
            self._index_start, self._index_end = 0, -1

    @staticmethod
    async def _layout_pass(parent: Widget) -> None:
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[None] = loop.create_future()
        # Layout passes may run inside an App message handler (for example when
        # prepending history). Waiting on a callback queued to the App's own
        # message pump would block that callback; the active Screen has an
        # independent message pump and can deliver the refresh callback.
        if not parent.is_mounted or not parent.app.is_running:
            return
        try:
            screen = parent.screen
        except NoScreen:
            return
        if not screen.call_after_refresh(
            lambda: not ready.done() and ready.set_result(None)
        ):
            return
        while not ready.done():
            if not parent.app.is_running or not parent.is_mounted:
                return
            try:
                await asyncio.wait_for(asyncio.shield(ready), timeout=0.5)
            except TimeoutError:
                continue

    @staticmethod
    def _placement(unit: TranscriptUnit) -> Widget | None:
        if unit.restoring and unit.mounted_roots:
            return unit.mounted_roots[0]
        if unit.placeholders:
            return unit.placeholders[0]
        return unit.mounted_roots[0] if unit.mounted_roots else None

    def _placement_rows(self, unit_id: str) -> tuple[int, int] | None:
        """The screen rows a unit's placement representative occupies.

        None when the placement index cannot order this unit: the unit or its
        representative is gone, or the representative has no measured rows yet.
        A widget mounted in this frame reads as region (0,0,0,0) until the
        layout that measures it, so its rows would claim row 0 -- above every
        scrolled-up neighbour -- instead of its real slot.
        """
        unit = self.units.get(unit_id)
        placement = self._placement(unit) if unit is not None else None
        if placement is None:
            return None
        region = placement.region
        if region.width <= 0:
            return None
        return region.y, region.bottom

    def _placement_order_holds(self, index: list[str], low: int, edge: int) -> bool:
        """Verify a boundary bisected out of the placement index.

        The bisect assumes the index is ordered by screen row: every placement
        before ``low`` ends at or above ``edge``, and every one from ``low`` on
        ends below it. A placement without measured rows breaks that order: it
        reports row 0, so it claims to end above a non-negative edge and below
        a negative one. Those claims are consistent with what the search
        already trusts, and there they drag the answer away from the boundary
        the search never visits -- a run of above-edge rows *after* the
        boundary for a non-negative edge, a run of below-edge rows *before* it
        for a negative one. The boundary's own neighbours are checked as well,
        so a placement without measured rows next to the answer is caught
        wherever it sits. The checks read one or two rows, so a query costs
        O(1) beyond the bisect; when one fails the caller answers from the
        content-ordered walk, which reads the same rows in content order and
        cannot be fooled by the ordering break.
        """
        if low < len(index):
            rows = self._placement_rows(index[low])
            if rows is None or rows[1] <= edge:
                return False
        if low > 0:
            # Covers ``low == len(index)`` too: the last placement is then the
            # one whose above-edge claim carried the search off the end.
            previous = self._placement_rows(index[low - 1])
            if previous is None or previous[1] > edge:
                return False
        if edge < 0 and index and self._placement_rows(index[0]) is None:
            return False
        return True

    def _index_placement(self, unit_id: str) -> None:
        """Insert a newly placed unit in content order.

        ``start_index`` is the registry's content order -- ``_place_new_units``
        orders admissions by it -- so a mounting unit slots in by bisection.
        """
        start_index = self.units[unit_id].start_index
        index = self._placements
        low, high = 0, len(index)
        while low < high:
            middle = (low + high) // 2
            if self.units[index[middle]].start_index <= start_index:
                low = middle + 1
            else:
                high = middle
        index.insert(low, unit_id)

    def _discard_placement(self, unit_id: str) -> None:
        try:
            self._placements.remove(unit_id)
        except ValueError:
            pass

    def _rename_placement(self, previous_id: str, public_id: str) -> None:
        index = self._placements
        for position, placed_id in enumerate(index):
            if placed_id == previous_id:
                index[position] = public_id
                return

    @staticmethod
    def _set_scroll_flag(view: Widget, name: str, enabled: bool) -> None:
        setattr(view, name, enabled)

    @staticmethod
    def _scroll_view(stream: Widget) -> Widget:
        return stream.parent if isinstance(stream.parent, VerticalScroll) else stream

    def _reading_anchor(self, stream: Widget) -> tuple[str, int, int] | None:
        """Locate the placement holding the viewport top.

        The placement index orders units by content only. Rows carry the scroll
        offset, so they are read from the live placements on every query:
        scrolling moves the answer, never the index.
        """
        top = self._scroll_view(stream).region.y
        if self._unmeasured_placement_pending():
            # A placeholder mounted this frame reads as row 0 above every
            # scrolled-up neighbour until its layout lands, which breaks the
            # index ordering: bisecting it anchors far below the viewport.
            return self._walk_anchor(top)
        index = self._placements
        low, high = 0, len(index)
        while low < high:
            middle = (low + high) // 2
            rows = self._placement_rows(index[middle])
            if rows is None:
                # A placement left the DOM, or has no rows to order by yet --
                # the ordered walk defines the answer until it is registered.
                return self._walk_anchor(top)
            if rows[1] <= top:
                low = middle + 1
            else:
                high = middle
        if not self._placement_order_holds(index, low, top):
            return self._walk_anchor(top)
        if low == len(index):
            return None
        rows = self._placement_rows(index[low])
        if rows is None:
            return self._walk_anchor(top)
        offset = max(0, top - rows[0])
        return index[low], offset, rows[0] + offset

    def _walk_anchor(self, top: int) -> tuple[str, int, int] | None:
        """Content-ordered placement walk, the fallback for a stale index entry."""
        for unit_id in self.unit_ids:
            placement = self._placement(self.units[unit_id])
            if placement is not None and placement.region.bottom > top:
                offset = max(0, top - placement.region.y)
                return unit_id, offset, placement.region.y + offset
        return None

    def capture_anchor(
        self, stream: Widget, *, following: bool
    ) -> TranscriptAnchor | None:
        """Capture the top inspected placement before a reflow or teardown."""
        reading = self._reading_anchor(stream)
        if reading is None:
            return None
        unit_id, offset, screen_row = reading
        unit = self.units[unit_id]
        return TranscriptAnchor(
            entry_id=unit.member_entry_ids[0],
            history_index=unit.start_index,
            row_offset=offset,
            screen_row=screen_row,
            following=following,
            scroll_revision=self._scroll_revision,
        )

    async def restore_anchor(
        self, stream: Widget, anchor: TranscriptAnchor, *, check_revision: bool = True
    ) -> None:
        """Restore the nearest available row, unless a later user scroll owns the view."""
        view = self._scroll_view(stream)
        if check_revision and self._scroll_revision != anchor.scroll_revision:
            return
        if anchor.following:
            self._set_scroll_flag(view, "_transcript_engine_scroll", True)
            try:
                view.scroll_end(animate=False, immediate=True)
                view.anchor()
            finally:
                self._set_scroll_flag(view, "_transcript_engine_scroll", False)
            return
        unit_id = self._entry_to_unit.get(anchor.entry_id)
        if unit_id is None and self.unit_ids:
            unit_id = min(
                self.unit_ids,
                key=lambda key: abs(self.units[key].start_index - anchor.history_index),
            )
        unit = self.units.get(unit_id) if unit_id is not None else None
        placement = self._placement(unit) if unit is not None else None
        if placement is None:
            return
        view.release_anchor()
        for _ in range(3):
            await self._layout_pass(stream)
            if check_revision and self._scroll_revision != anchor.scroll_revision:
                return
            offset = min(anchor.row_offset, max(0, placement.region.height - 1))
            delta = placement.region.y + offset - anchor.screen_row
            if not delta:
                break
            self._set_scroll_flag(view, "_transcript_engine_scroll", True)
            try:
                view.scroll_to(
                    y=view.scroll_offset.y + delta, animate=False, immediate=True
                )
            finally:
                self._set_scroll_flag(view, "_transcript_engine_scroll", False)

    @staticmethod
    async def _initialize_roots(roots: Sequence[Widget]) -> None:
        for root in roots:
            for widget in (root, *root.walk_children()):
                if isinstance(widget, StreamingMessageBase):
                    await widget.write_initial_content()

    def _scroll_superseded(
        self, stream: Widget, revision: int, scroll_start: float
    ) -> bool:
        return self._scroll_revision != revision or (
            not getattr(stream, "_transcript_tracks_user_scroll", False)
            and stream.scroll_offset.y != scroll_start
        )

    async def _finish_restore(
        self,
        unit: TranscriptUnit,
        stream: Widget,
        roots: list[Widget],
        placeholders: list[PlacementPlaceholder],
        anchor: tuple[str, int, int] | None,
        valid: Callable[[], bool],
        scroll_revision: int,
        scroll_start: float,
        scroll_superseded: bool,
    ) -> bool:
        # A stable height across a refresh is the settle signal. The floor
        # allows delayed content to get a layout opportunity; the cap handles
        # continuously changing widgets without wedging the reservation.
        previous_height: int | None = None
        view = self._scroll_view(stream)
        superseded = scroll_superseded
        for _ in range(12):
            await self._layout_pass(stream)
            if not valid():
                return False
            superseded |= self._scroll_superseded(view, scroll_revision, scroll_start)
            height = sum(root.region.height for root in roots)
            if previous_height is not None and height == previous_height:
                break
            previous_height = height
        for placeholder in placeholders:
            await placeholder.remove()
            placeholder.release()
            if not valid():
                return False
        await self._layout_pass(stream)
        if not valid():
            return False
        superseded |= self._scroll_superseded(view, scroll_revision, scroll_start)
        if not superseded and anchor is None:
            self._set_scroll_flag(view, "_transcript_engine_scroll", True)
            try:
                view.scroll_end(animate=False, immediate=True)
            finally:
                self._set_scroll_flag(view, "_transcript_engine_scroll", False)
        if not superseded and anchor is not None:
            anchor_id, row_offset, screen_row = anchor
            target = self.units.get(anchor_id)
            placement = self._placement(target) if target is not None else None
            if anchor_id == unit.id:
                placement = roots[0]
            if placement is not None:
                delta = placement.region.y + row_offset - screen_row
                self._set_scroll_flag(view, "_transcript_engine_scroll", True)
                try:
                    view.scroll_to(
                        y=view.scroll_offset.y + delta, animate=False, immediate=True
                    )
                finally:
                    self._set_scroll_flag(view, "_transcript_engine_scroll", False)
                await self._layout_pass(stream)
                if not valid():
                    return False
        # The live id: identity adoption may have renamed the unit while the
        # placeholders were leaving the DOM.
        self.register_restored(unit.id, roots)
        return True

    @staticmethod
    async def _rollback_restore(
        stream: Widget,
        roots: list[Widget],
        placeholders: list[PlacementPlaceholder],
        *,
        registered: bool,
    ) -> None:
        if registered:
            missing = [p for p in placeholders if p.parent is not stream]
            if missing:
                before = next((root for root in roots if root.parent is stream), None)
                if before is not None:
                    await stream.mount_all(missing, before=before)
                else:
                    await stream.mount_all(missing)
        for root in roots:
            if root.parent is stream:
                await root.remove()
        for placeholder in placeholders:
            if placeholder.parent is stream:
                if registered:
                    placeholder.release()
                else:
                    await placeholder.remove()

    def _restore_roots(self, unit: TranscriptUnit, roots: Sequence[Widget]) -> None:
        """Adopt restored roots while the unit's placeholders still hold its slot."""
        unit.mounted_roots = list(roots)
        self._shadow_unit(unit)
        # The roots are mounted in place of an interior slot and stay without
        # rows until the layout pass that measures them.
        self._mark_unmeasured(unit)

    def _release_roots(self, unit: TranscriptUnit) -> None:
        """Drop the roots a failed or superseded restore had mounted."""
        unit.mounted_roots = []
        self._shadow_unit(unit)

    async def restore_unit(
        self,
        unit_id: str,
        stream: Widget,
        history_widget_indices: WeakKeyDictionary[Widget, int],
        *,
        follow_bottom: bool,
        initialize: Callable[[Sequence[Widget]], Awaitable[None]] | None = None,
    ) -> bool:
        async with self._reconcile_lock:
            unit = self.units.get(unit_id)
            if unit is None or not unit.placeholders or unit.restoring:
                return False
            placeholders = list(unit.placeholders)
            if any(placeholder.parent is not stream for placeholder in placeholders):
                return False
            unit.restoring = True
            unit.restore_generation += 1
            generation = unit.restore_generation
            session = self._session_generation
            # The unit, not its entry-time id: identity adoption renames the
            # registry entry while an awaited DOM operation below is in flight
            # (admission does not take the reconcile lock), and the restore
            # still owns the same unit -- both its roots and its placeholders.
            # Addressed by the old id, the rename read as "the unit left the
            # registry": the rollback then detached both representatives.
            valid = lambda: (
                session == self._session_generation
                and self.units.get(unit.id) is unit
                and unit.restore_generation == generation
                and unit.placeholders == placeholders
            )
            anchor = None if follow_bottom else self._reading_anchor(stream)
            try:
                roots = self.build_unit(unit_id, history_widget_indices)
                if not roots:
                    raise ValueError("restored unit has no roots")
                scroll_revision = self._scroll_revision
                for placeholder in placeholders:
                    placeholder.reserve()
            except BaseException:
                unit.restoring = False
                for placeholder in placeholders:
                    if placeholder._reserved:
                        placeholder.release()
                raise
        completed = False
        view = self._scroll_view(stream)
        self._set_scroll_flag(view, "_transcript_layout_change", True)
        try:
            await stream.mount_all(roots, before=placeholders[0])
            self._restore_roots(unit, roots)
            await self._layout_pass(stream)
            if not valid():
                return False
            scroll_start = view.scroll_offset.y
            await (initialize or self._initialize_roots)(roots)
            if not valid():
                return False
            scroll_superseded = self._scroll_superseded(
                view, scroll_revision, scroll_start
            )
            async with self._reconcile_lock:
                if not valid():
                    return False
                completed = await self._finish_restore(
                    unit,
                    stream,
                    roots,
                    placeholders,
                    anchor,
                    valid,
                    scroll_revision,
                    scroll_start,
                    scroll_superseded,
                )
            return completed
        finally:
            try:
                if not completed:
                    async with self._reconcile_lock:
                        await self._rollback_restore(
                            stream,
                            roots,
                            placeholders,
                            registered=self.units.get(unit.id) is unit,
                        )
            finally:
                self._set_scroll_flag(view, "_transcript_layout_change", False)
                unit.restoring = False
                if not completed and self.units.get(unit.id) is unit:
                    self._release_roots(unit)
                    # The rollback re-mounts the placeholders it removed, so
                    # the unit's slot has no rows until the next layout either.
                    self._mark_unmeasured(unit)

    def _window_low(self, index: list[str], edge: int) -> int | None:
        """Bisect the first placement whose bottom passes ``edge``.

        Returns None when a placement left the DOM (or the registry) without
        the index hearing about it, so the caller can fall back to the full
        placeholder scan.
        """
        low, high = 0, len(index)
        while low < high:
            middle = (low + high) // 2
            rows = self._placement_rows(index[middle])
            if rows is None:
                return None
            if rows[1] <= edge:
                low = middle + 1
            else:
                high = middle
        if self._placement_order_holds(index, low, edge):
            return low
        # The index is not ordered by screen row around this answer, so the
        # bisect cannot be trusted: answer from the content-ordered scan of the
        # same index, reading rows exactly as the walk reads them. This path
        # runs only while a placement without measured rows is in the index.
        for position, unit_id in enumerate(index):
            unit = self.units.get(unit_id)
            placement = self._placement(unit) if unit is not None else None
            if placement is not None and placement.region.bottom > edge:
                return position
        return len(index)

    def _mark_unmeasured(self, unit: TranscriptUnit) -> None:
        """Mark a unit whose placement has no measured rows yet.

        The id is read off the live unit, never captured when the transition
        started: the awaited DOM operations inside an eviction let identity
        adoption rename the unit mid-flight, and ``admit`` does not take the
        reconcile lock. A mark left on the entry-time id names no unit, so the
        pending check discards it and drops the guard exactly when the
        unmeasured placement is mounted.
        """
        if self.units.get(unit.id) is unit:
            self._unmeasured_placements.add(unit.id)

    def _unmeasured_placement_pending(self) -> bool:
        """True while a marked unit's placement has no measured geometry yet.

        A placement mounted in this frame reports a zero-width region until
        the layout pass that measures it, so it reads as row 0 -- above every
        scrolled-up neighbour -- and the placement index stops being ordered
        by screen row. The representative checked is the row the index reads
        (``_placement``: the restoring root while one is mounted, else the
        placeholder), and a unit that is gone, whose representative left the
        DOM, or whose representative is fully measured leaves the set: a
        detached row can never be measured, so a mark that waited for it would
        hold the full-scan fallback for the rest of the session. A settled
        registry therefore costs the empty-set check, and the scan is bounded
        by the transitions of the last frame.
        """
        pending = self._unmeasured_placements
        if not pending:
            return False
        for unit_id in list(pending):
            unit = self.units.get(unit_id)
            placement = self._placement(unit) if unit is not None else None
            if (
                placement is None
                or placement.parent is None
                or placement.region.width > 0
            ):
                pending.discard(unit_id)
        return bool(pending)

    def _restore_candidates(self, top: int, height: int) -> list[str]:
        """Placeholder-owning units whose slot can intersect the viewport window.

        The placement index is ordered by start_index, and placement order IS
        screen order: each placement's region sits below every earlier
        placement's (region.y includes the scroll offset, so the order is
        scroll-invariant), which makes the index's bisect plus forward scan
        visit exactly the placements whose regions can intersect the window.
        The index carries live screen regions, so the candidate set is the
        window's slice of it: per-tick work is bounded by the placements near
        the viewport, never by the registry's placeholder count. A unit owning
        placeholders is always placed, so an index that does not cover every
        placeholder unit -- or a placement that left the DOM without the index
        hearing about it -- falls back to the full placeholder scan.
        """
        shadow = self._shadow
        if not shadow.placeholders:
            return []
        # A placement mounted this frame but not yet laid out has no row to
        # window by: the scroll that follows an eviction reaches this scan
        # before the measuring layout, and one unmeasured row 0 breaks the
        # index ordering the bisect relies on. Window only once every
        # placement in the index carries measured rows.
        if self._unmeasured_placement_pending():
            return self.placeholder_unit_ids()
        index = self._placements
        # Exact coverage check at O(1): mounting is what indexes a unit and
        # eviction keeps the slot, so the index is exactly the mounted plus
        # placeholder-owning units (the two facets are disjoint), and its
        # length equals |mounted| + |placeholders| iff every placeholder unit
        # is in it. The old len(index) < len(placeholders) guard was vacuous
        # whenever the index held enough non-placeholder units.
        if len(index) != len(shadow.mounted) + len(shadow.placeholders):
            return self.placeholder_unit_ids()
        low = self._window_low(index, top - height)
        if low is None:
            return self.placeholder_unit_ids()
        candidates: list[str] = []
        # The unit just before the window can straddle its top edge when it
        # spans several placeholders; earlier units end above it.
        position = max(0, low - 1)
        while position < len(index):
            unit_id = index[position]
            unit = self.units.get(unit_id)
            placement = self._placement(unit) if unit is not None else None
            if placement is None:
                return self.placeholder_unit_ids()
            if placement.region.y >= top + 2 * height:
                break
            if unit is not None and unit.placeholders:
                candidates.append(unit_id)
            position += 1
        candidates.sort(key=lambda unit_id: self.units[unit_id].start_index)
        return candidates

    async def _reconcile_once(
        self,
        stream: Widget,
        history_widget_indices: WeakKeyDictionary[Widget, int],
        initialize: Callable[[Sequence[Widget]], Awaitable[None]] | None,
        follow_bottom: bool,
        low_mark: int,
        high_mark: int,
        selected: frozenset[str],
        live: frozenset[str],
        compaction_neighbors: frozenset[str],
    ) -> None:
        self._sync_width(stream.size.width)
        self.sync_expansion()
        view = self._scroll_view(stream)
        top, height = view.region.y, view.region.height
        # The placement index windows the restore candidates to the viewport:
        # per-tick work is bounded by residency near the window, not by the
        # number of placeholders in the registry. A completed restore clears
        # the placeholders the window just listed, so the loop re-checks each
        # unit before restoring it.
        for unit_id in self._restore_candidates(top, height):
            unit = self.units.get(unit_id)
            if unit is None or not unit.placeholders:
                continue
            first, last = unit.placeholders[0], unit.placeholders[-1]
            if first.region.y < top + 2 * height and last.region.bottom > top - height:
                await self.restore_unit(
                    unit_id,
                    stream,
                    history_widget_indices,
                    initialize=initialize,
                    follow_bottom=follow_bottom,
                )
        for unit_id in self.eviction_candidates(
            view.region.y,
            view.region.height,
            low_mark=low_mark,
            high_mark=high_mark,
            selected=selected,
            live=live,
            compaction_neighbors=compaction_neighbors,
        ):
            async with self._reconcile_lock:
                if unit_id in self.units and self.units[unit_id].mounted_roots:
                    await self.evict_unit(unit_id)

    def request_reconcile(
        self,
        stream: Widget,
        history_widget_indices: WeakKeyDictionary[Widget, int],
        *,
        follow_bottom: bool,
        initialize: Callable[[Sequence[Widget]], Awaitable[None]] | None = None,
        low_mark: int = 1000,
        high_mark: int = 1500,
        selected: frozenset[str] = frozenset(),
        live: frozenset[str] = frozenset(),
        compaction_neighbors: frozenset[str] = frozenset(),
    ) -> asyncio.Task[None]:
        async def latest() -> None:
            await self._reconcile_once(
                stream,
                history_widget_indices,
                initialize,
                follow_bottom,
                low_mark,
                high_mark,
                selected,
                live,
                compaction_neighbors,
            )

        self._reconcile_latest = latest
        if self._reconcile_task is not None and not self._reconcile_task.done():
            if self._reconcile_active:
                self._reconcile_pending = True
            return asyncio.create_task(self._await_reconcile(self._reconcile_task))

        async def run() -> None:
            session = self._session_generation
            await asyncio.sleep(0)
            if session != self._session_generation:
                return
            self._reconcile_active = True
            try:
                while True:
                    self._reconcile_pending = False
                    current = self._reconcile_latest
                    if current is not None:
                        await current()
                    if (
                        not self._reconcile_pending
                        or session != self._session_generation
                    ):
                        break
            finally:
                if session == self._session_generation:
                    self._reconcile_active = False

        self._reconcile_task = asyncio.create_task(run())
        return asyncio.create_task(self._await_reconcile(self._reconcile_task))

    @staticmethod
    async def _await_reconcile(task: asyncio.Task[None]) -> None:
        await asyncio.shield(task)

    def build_unit(
        self, unit_id: str, history_widget_indices: WeakKeyDictionary[Widget, int]
    ) -> list[Widget]:
        unit = self.units[unit_id]
        entries = unit.entries
        if (
            unit.group_key is None
            and len(entries) > 1
            and all(
                isinstance(entry, PublicMessageEntry) and entry.role == "assistant"
                for entry in entries
            )
        ):
            messages = [
                entry for entry in entries if isinstance(entry, PublicMessageEntry)
            ]
            entries = [
                messages[-1].model_copy(
                    update={
                        "content": [
                            part for entry in messages for part in entry.content
                        ]
                    }
                )
            ]
        return _build_history_widgets_raw(
            entries,
            start_index=unit.start_index,
            history_widget_indices=history_widget_indices,
            tools_collapsed=self.tools_collapsed,
            expansion_state=self.expansion_state,
            entry_expansion_state=self.entry_expansion_state,
        )

    def flat_widgets(
        self, history_widget_indices: WeakKeyDictionary[Widget, int]
    ) -> list[Widget]:
        widgets: list[Widget] = []
        for unit_id in self.unit_ids:
            widgets.extend(self.build_unit(unit_id, history_widget_indices))
        return widgets
