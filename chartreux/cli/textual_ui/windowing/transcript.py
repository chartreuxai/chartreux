from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary, ref

from textual.containers import VerticalScroll
from textual.dom import NoScreen
from textual.widget import Widget

from chartreux.app_server.models import (
    PublicEffectEntry,
    PublicHistoryEntry,
    PublicReasoningEntry,
)
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.messages import StreamingMessageBase
from chartreux.cli.textual_ui.widgets.tool_grouping import (
    ToolGroupExpansionState,
    ToolGroupKey,
    entry_keeps_tool_group,
)
from chartreux.cli.textual_ui.windowing.history import (
    _build_history_widgets_raw,
    history_entry_renders_widget,
)
from chartreux.cli.textual_ui.windowing.placeholder import PlacementPlaceholder


@dataclass(slots=True)
class TranscriptUnit:
    id: str
    member_entry_ids: list[str]
    entries: list[PublicHistoryEntry]
    start_index: int
    group_key: ToolGroupKey | None = None
    entry_snapshots: list[PublicHistoryEntry] = field(default_factory=list)
    mounted_roots: list[Widget] = field(default_factory=list)
    placeholders: list[PlacementPlaceholder] = field(default_factory=list)
    content_height: int | None = None
    # Parent Stream's outer size.width, not the placement's content width.
    geometry_width: int | None = None
    geometry_version: int = 0
    pin_reasons: set[str] = field(default_factory=set)
    restoring: bool = False
    restore_generation: int = 0


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
        self._evicting: set[str] = set()
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

    def _shift_for_prefix(self, ids: list[str], start_index: int) -> None:
        if not (
            ids
            and self._admitted_indices
            and start_index == self._index_start
            and all(key not in self._admitted_indices for key in ids)
        ):
            return
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
        else:
            self.unit_ids.extend(new_ids)

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

    def admit(
        self,
        batch: Sequence[PublicHistoryEntry],
        *,
        start_index: int,
        geometry_width: int | None = None,
        extend_group: bool = False,
    ) -> None:
        """Admit a contiguous timeline batch; new entries must extend either edge.

        Pass the parent Stream's outer ``size.width`` as ``geometry_width``;
        this is not the placeholder's Stream content-height width baseline.
        """
        ids = [self._entry_id(entry) for entry in batch]
        self._check_batch(ids, start_index)
        self._sync_width(geometry_width)
        self._shift_for_prefix(ids, start_index)
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
                member_index = existing.member_entry_ids.index(entry_id)
                if existing.entry_snapshots[member_index] != entry:
                    existing.entry_snapshots[member_index] = entry.model_copy(deep=True)
                    self._invalidate(existing)
                existing.entries[member_index] = entry
                new_start = index - member_index
                if existing.start_index != new_start:
                    existing.start_index = new_start
                    self._invalidate(existing)
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
                unit = current
            else:
                if not history_entry_renders_widget(entry):
                    continue
                unit = TranscriptUnit(
                    entry_id, [], [], index, geometry_width=geometry_width
                )
                self.units[entry_id] = unit
                self.unit_ids.append(entry_id)
            unit.member_entry_ids.append(entry_id)
            unit.entries.append(entry)
            unit.entry_snapshots.append(entry.model_copy(deep=True))
            self._entry_to_unit[entry_id] = unit.id
            unit.start_index = min(unit.start_index, index)
            if geometry_width is not None:
                unit.geometry_width = geometry_width
            self._invalidate(unit)
        self._place_new_units(prior_units, prior_first)

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
        self._index_start = 0
        self._index_end = -1
        self._geometry_width = None
        self._evicting.clear()
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
        unit.mounted_roots = list(roots)
        unit.content_height = sum(root.region.height for root in roots)
        unit.geometry_width = parent.size.width

    def register_restored(self, unit_id: str, roots: Sequence[Widget]) -> None:
        """Register restored roots after their placeholders leave the Stream DOM."""
        unit = self.units[unit_id]
        if any(placeholder.parent is not None for placeholder in unit.placeholders):
            raise ValueError("unit placeholders must be removed before restore")
        placeholders = unit.placeholders
        unit.placeholders = []
        try:
            self.register_mounted(unit_id, roots)
        except BaseException:
            unit.placeholders = placeholders
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
        for unit_id in self.unit_ids:
            unit = self.units[unit_id]
            if not unit.mounted_roots or unit.placeholders:
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
            self.geometry_version += 1
            return placeholders
        finally:
            self._evicting.discard(unit_id)

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

    @staticmethod
    def _set_scroll_flag(view: Widget, name: str, enabled: bool) -> None:
        setattr(view, name, enabled)

    @staticmethod
    def _scroll_view(stream: Widget) -> Widget:
        return stream.parent if isinstance(stream.parent, VerticalScroll) else stream

    def _reading_anchor(self, stream: Widget) -> tuple[str, int, int] | None:
        top = self._scroll_view(stream).region.y
        for unit_id in self.unit_ids:
            placement = self._placement(self.units[unit_id])
            if placement is not None and placement.region.bottom > top:
                offset = max(0, top - placement.region.y)
                return unit_id, offset, placement.region.y + offset
        return None

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
        unit_id: str,
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
            if anchor_id == unit_id:
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
        self.register_restored(unit_id, roots)
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
            valid = lambda: (
                session == self._session_generation
                and self.units.get(unit_id) is unit
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
            unit.mounted_roots = list(roots)
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
                    unit_id,
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
                            registered=self.units.get(unit_id) is unit,
                        )
            finally:
                self._set_scroll_flag(view, "_transcript_layout_change", False)
                unit.restoring = False
                if not completed and self.units.get(unit_id) is unit:
                    unit.mounted_roots = []

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
        for unit_id in list(self.unit_ids):
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
        return _build_history_widgets_raw(
            unit.entries,
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
