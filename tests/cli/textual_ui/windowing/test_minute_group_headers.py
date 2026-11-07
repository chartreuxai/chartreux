from __future__ import annotations

from datetime import datetime
from weakref import WeakKeyDictionary

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.geometry import Size
from textual.widget import Widget

from chartreux.app_server._projection import project_message_history
from chartreux.app_server.events import HistoryEntryAdded
from chartreux.app_server.models import (
    CompletedEffectState,
    EffectCallDisplay,
    EffectResultDisplay,
    GenericEffectDetail,
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicMessageEntry,
    PublicNoticeEntry,
    PublicReasoningEntry,
    RootRecoveryNoticeDetail,
    RouteTableAppliedNoticeDetail,
    ScheduledLoopFiredNoticeDetail,
    TextContentBlock,
)
from chartreux.cli.textual_ui.handlers.event_handler import EventHandler
from chartreux.cli.textual_ui.widgets.message_header import (
    format_group_header_timestamp,
)
from chartreux.cli.textual_ui.widgets.messages import (
    ReasoningMessage,
    UserCommandMessage,
)
from chartreux.cli.textual_ui.widgets.minute_group_header import MinuteGroupHeader
from chartreux.cli.textual_ui.widgets.tool_grouping import (
    ToolGroupExpansionState,
    ToolGroupKey,
)
from chartreux.cli.textual_ui.widgets.tools import ToolGroup
from chartreux.cli.textual_ui.windowing.history import minute_marker_minutes
from chartreux.cli.textual_ui.windowing.transcript import TranscriptWindow
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall

TURN_ID = "turn-live"


class _App(App[None]):
    CSS = "#stream { width: 60; height: 20; layout: stream; overflow-y: auto; }"

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="stream")


class _Root(Widget):
    def __init__(self, height: int = 2) -> None:
        super().__init__()
        self.height = height

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return self.height


class _GatedWindow(TranscriptWindow):
    """The session presenter always supplies this gate; tests flip it live."""

    def __init__(self, *, show: bool = True, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._show = show
        self._get_show_timestamps = lambda: self._show


def _local(moment: datetime) -> datetime:
    """Interpret a wall-clock time as local, so minute keys are host-stable."""
    return moment.astimezone()


def _at(hour: int, minute: int, *, day: int = 6) -> tuple[int, int, int, int, int]:
    moment = _local(datetime(2026, 11, day, hour, minute))
    return (moment.year, moment.month, moment.day, moment.hour, moment.minute)


def _epoch_ms(hour: int, minute: int, *, day: int = 6) -> int:
    return int(_local(datetime(2026, 11, day, hour, minute)).timestamp()) * 1000


def _base(entry_id: str) -> dict[str, object]:
    return {
        "id": entry_id,
        "session_id": "session-1",
        "updated_at": 1,
        "generation_status": PublicEntryGenerationStatus.COMPLETED,
    }


def _user(entry_id: str, hour: int, minute: int) -> PublicMessageEntry:
    return PublicMessageEntry(
        **_base(entry_id),  # type: ignore[arg-type]
        created_at=1,
        role="user",
        content=[TextContentBlock(text="question")],
        posted_at=_local(datetime(2026, 11, 6, hour, minute)),
    )


def _restored_reasoning(entry_id: str, hour: int, minute: int) -> PublicReasoningEntry:
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


def _legacy_reasoning(entry_id: str, hour: int, minute: int) -> PublicReasoningEntry:
    """A pre-v0.7.4 record: the synthetic restore ordinal, no real clock."""
    return PublicReasoningEntry(
        **_base(entry_id),  # type: ignore[arg-type]
        created_at=_epoch_ms(hour, minute),
        text="thinking",
    )


def _effect(
    entry_id: str, hour: int, minute: int, *, live: bool = False
) -> PublicEffectEntry:
    detail = GenericEffectDetail(
        tool_name="tool", display=EffectCallDisplay(summary="tool", status_text="run")
    )
    state = CompletedEffectState(
        display=EffectResultDisplay(success=True, message="done"), duration_ms=300_000
    )
    if live:
        return PublicEffectEntry(
            **_base(entry_id),  # type: ignore[arg-type]
            created_at=_epoch_ms(hour, minute),
            turn_id=TURN_ID,
            title="tool",
            detail=detail,
            state=state,
        )
    return PublicEffectEntry(
        **_base(entry_id),  # type: ignore[arg-type]
        created_at=21,
        title="tool",
        detail=detail,
        state=state,
        event_at=_epoch_ms(hour, minute),
    )


def _compaction(entry_id: str, hour: int, minute: int) -> PublicCheckpointEntry:
    return PublicCheckpointEntry(
        **_base(entry_id),  # type: ignore[arg-type]
        created_at=_epoch_ms(hour, minute),
        turn_id=TURN_ID,
        kind="compaction",
    )


async def _reconcile(
    window: TranscriptWindow, stream: VerticalScroll, pilot: object
) -> None:
    await window._reconcile_once(
        stream,
        WeakKeyDictionary(),
        None,
        True,
        1000,
        1500,
        frozenset(),
        frozenset(),
        frozenset(),
    )
    await pilot.pause()  # type: ignore[attr-defined]


def _label(hour: int, minute: int) -> str:
    """The label a header for that bucket renders, with the host's own clock."""
    return format_group_header_timestamp(_at(hour, minute))


def _headers(stream: VerticalScroll) -> list[MinuteGroupHeader]:
    return [child for child in stream.children if isinstance(child, MinuteGroupHeader)]


# -- Pure marker walk --------------------------------------------------------


def test_message_head_advances_the_marker_without_a_group_header() -> None:
    markers = minute_marker_minutes([
        _user("m1", 14, 20),
        _restored_reasoning("r1", 14, 20),
    ])

    assert markers == [None, None]


def test_minute_bucket_insertion_by_kind() -> None:
    markers = minute_marker_minutes([
        _user("m1", 14, 20),
        _restored_reasoning("r1", 14, 21),
        _effect("e1", 14, 21),
        _compaction("c1", 14, 22),
    ])

    assert markers == [None, _at(14, 21), None, _at(14, 22)]


def test_adjacent_units_in_one_minute_share_one_marker() -> None:
    markers = minute_marker_minutes([
        _restored_reasoning("r1", 14, 20),
        _compaction("c1", 14, 20),
    ])

    assert markers == [_at(14, 20), None]


def test_effect_stamp_is_its_start_minute_not_its_result() -> None:
    # The effect started at 14:20 and ran for five minutes; the following
    # record is still inside 14:20, so the elapsed run never re-stamps.
    markers = minute_marker_minutes([
        _effect("e1", 14, 20),
        _restored_reasoning("r1", 14, 20),
    ])

    assert markers == [_at(14, 20), None]


def test_records_without_a_real_clock_never_bucket() -> None:
    # A legacy record restores with the synthetic timestamp+index ordinal
    # (which looks like a real epoch) and no turn id: it is never a clock.
    markers = minute_marker_minutes([_legacy_reasoning("r1", 14, 20)])

    assert markers == [None]


def test_live_entries_bucket_from_the_projector_clock() -> None:
    markers = minute_marker_minutes([_live_reasoning("r1", 14, 20)])

    assert markers == [_at(14, 20)]


def test_group_header_formats_today_as_hh_mm_and_other_days_with_the_date() -> None:
    minute = _at(14, 20)

    assert (
        format_group_header_timestamp(minute, now=_local(datetime(2026, 11, 6, 18, 5)))
        == "14:20"
    )
    assert (
        format_group_header_timestamp(minute, now=_local(datetime(2026, 11, 7, 9, 5)))
        == "2026-11-06 14:20"
    )


# -- Window mounts -----------------------------------------------------------


@pytest.mark.asyncio
async def test_new_minute_mounts_one_group_header_before_the_records() -> None:
    entries = [_user("m1", 14, 20), _restored_reasoning("r1", 14, 21)]
    window = _GatedWindow()
    window.admit(entries, start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = {unit_id: [_Root()] for unit_id in window.unit_ids}
        for unit_roots in roots.values():
            await stream.mount_all(unit_roots)
        await pilot.pause()
        for unit_id, unit_roots in roots.items():
            window.register_mounted(unit_id, unit_roots)
        await _reconcile(window, stream, pilot)

        headers = _headers(stream)
        assert len(headers) == 1
        assert headers[0].label == _label(14, 21)
        children = list(stream.children)
        assert (
            children.index(headers[0])
            == children.index(roots[window.unit_ids[1]][0]) - 1
        )
        assert window.units[window.unit_ids[0]].head_minute is None
        assert window.units[window.unit_ids[1]].head_minute == _at(14, 21)


@pytest.mark.asyncio
async def test_the_header_never_becomes_the_units_entry_root() -> None:
    # The presenter resolves a unit by mounted_roots[0] (group extension, timing
    # widgets), so the stamp must sit beside that root, not replace it.
    window = _GatedWindow()
    window.admit([_live_reasoning("r1", 14, 20)], start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        root = _Root()
        await stream.mount_all([root])
        await pilot.pause()
        unit = window.units[window.unit_ids[0]]
        window.register_mounted(unit.id, [root])
        await _reconcile(window, stream, pilot)

        header = _headers(stream)[0]
        assert unit.minute_header is header
        assert unit.mounted_roots == [root]
        children = list(stream.children)
        assert children.index(header) == children.index(root) - 1


@pytest.mark.asyncio
async def test_adjacent_units_in_one_minute_mount_one_shared_header() -> None:
    entries = [_restored_reasoning("r1", 14, 20), _compaction("c1", 14, 20)]
    window = _GatedWindow()
    window.admit(entries, start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = {unit_id: [_Root()] for unit_id in window.unit_ids}
        for unit_roots in roots.values():
            await stream.mount_all(unit_roots)
        await pilot.pause()
        for unit_id, unit_roots in roots.items():
            window.register_mounted(unit_id, unit_roots)
        await _reconcile(window, stream, pilot)

        assert len(window.unit_ids) == 2
        headers = _headers(stream)
        assert len(headers) == 1
        assert headers[0].label == _label(14, 20)
        assert headers[0].parent is stream


@pytest.mark.asyncio
async def test_live_delivery_into_a_new_minute_mounts_exactly_one_header() -> None:
    window = _GatedWindow()
    window.admit([_user("m1", 14, 20)], start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all([_Root()])
        await pilot.pause()
        window.register_mounted(window.unit_ids[0], list(stream.children))
        await _reconcile(window, stream, pilot)
        assert _headers(stream) == []

        # A live reasoning entry lands in the same minute: no stamp of its own.
        window.admit([_live_reasoning("r1", 14, 20)], start_index=1)
        late = window.units[window.unit_ids[1]]
        late_roots = [_Root()]
        await stream.mount_all(late_roots)
        await pilot.pause()
        window.register_mounted(late.id, late_roots)
        await _reconcile(window, stream, pilot)
        assert _headers(stream) == []

        # A live entry crossing the minute renders exactly one header.
        window.admit([_live_reasoning("r2", 14, 27)], start_index=2)
        fresh = window.units[window.unit_ids[2]]
        fresh_roots = [_Root()]
        await stream.mount_all(fresh_roots)
        await pilot.pause()
        window.register_mounted(fresh.id, fresh_roots)
        await _reconcile(window, stream, pilot)

        headers = _headers(stream)
        assert len(headers) == 1
        assert headers[0].label == _label(14, 27)


@pytest.mark.asyncio
async def test_evicting_the_marker_unit_keeps_the_minute_and_restores_its_header() -> (
    None
):
    window = _GatedWindow()
    window.admit([_restored_reasoning("r1", 14, 20)], start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        first_roots = [_Root()]
        await stream.mount_all(first_roots)
        await pilot.pause()
        marker_unit = window.unit_ids[0]
        window.register_mounted(marker_unit, first_roots)
        await _reconcile(window, stream, pilot)
        assert [header.label for header in _headers(stream)] == [_label(14, 20)]

        await window.evict_unit(marker_unit)
        await pilot.pause()
        # Eviction keeps the unit's entries, so the marker still derives from
        # them (the reconcile restores this visible unit and its header).
        assert window.units[marker_unit].head_minute == _at(14, 20)

        # A same-minute record admitted after that eviction opens no second
        # header: the marker survived the unit leaving the DOM.
        window.admit([_restored_reasoning("r2", 14, 20)], start_index=1)
        late = window.units[window.unit_ids[1]]
        late_roots = [_Root()]
        await stream.mount_all(late_roots)
        await pilot.pause()
        window.register_mounted(late.id, late_roots)
        await _reconcile(window, stream, pilot)
        assert late.head_minute is None
        assert [header.label for header in _headers(stream)] == [_label(14, 20)]


@pytest.mark.asyncio
async def test_rebuild_after_reset_regenerates_headers() -> None:
    entries = [_user("m1", 14, 20), _restored_reasoning("r1", 14, 21)]
    window = _GatedWindow()
    window.admit(entries, start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = {unit_id: [_Root()] for unit_id in window.unit_ids}
        for unit_roots in roots.values():
            await stream.mount_all(unit_roots)
        await pilot.pause()
        for unit_id, unit_roots in roots.items():
            window.register_mounted(unit_id, unit_roots)
        await _reconcile(window, stream, pilot)
        assert [header.label for header in _headers(stream)] == [_label(14, 21)]

        window.reset()
        await stream.remove_children()
        window.admit(entries, start_index=0)
        rebuilt = {unit_id: [_Root()] for unit_id in window.unit_ids}
        for unit_roots in rebuilt.values():
            await stream.mount_all(unit_roots)
        await pilot.pause()
        for unit_id, unit_roots in rebuilt.items():
            window.register_mounted(unit_id, unit_roots)
        await _reconcile(window, stream, pilot)

        assert [header.label for header in _headers(stream)] == [_label(14, 21)]


@pytest.mark.asyncio
async def test_gating_off_mounts_no_header_and_turning_it_on_mounts_one() -> None:
    window = _GatedWindow(show=False)
    window.admit([_restored_reasoning("r1", 14, 20)], start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all([_Root()])
        await pilot.pause()
        window.register_mounted(window.unit_ids[0], list(stream.children))
        await _reconcile(window, stream, pilot)
        assert _headers(stream) == []

        window._show = True
        await _reconcile(window, stream, pilot)
        assert [header.label for header in _headers(stream)] == [_label(14, 20)]

        window._show = False
        await _reconcile(window, stream, pilot)
        assert _headers(stream) == []


@pytest.mark.asyncio
async def test_mounted_entry_widgets_resolve_the_entry_roots_only() -> None:
    # A compaction record is its own unit, so the unit's roots are its entries.
    window = _GatedWindow()
    window.admit([_compaction("c1", 14, 20)], start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        root = _Root()
        await stream.mount_all([root])
        await pilot.pause()
        window.register_mounted(window.unit_ids[0], [root])
        await _reconcile(window, stream, pilot)
        assert len(_headers(stream)) == 1

        assert window.mounted_entry_widgets("c1") == [root]


def _stamped_messages() -> list[LLMMessage]:
    return [
        LLMMessage(
            role=Role.user,
            content="question",
            message_id="u",
            posted_at=_local(datetime(2026, 11, 6, 14, 20)),
        ),
        LLMMessage(
            role=Role.assistant,
            content="answer",
            message_id="a",
            reasoning_content="thinking",
            reasoning_at=_epoch_ms(14, 20),
            tool_calls=[
                ToolCall(
                    id="c",
                    index=0,
                    function=FunctionCall(name="todo", arguments="{}"),
                    started_at=_epoch_ms(14, 20),
                )
            ],
        ),
        LLMMessage(role=Role.tool, content="done", tool_call_id="c", name="todo"),
    ]


def _unstamped_messages() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.user, content="question", message_id="u"),
        LLMMessage(
            role=Role.assistant,
            content="answer",
            message_id="a",
            reasoning_content="thinking",
            tool_calls=[
                ToolCall(
                    id="c", index=0, function=FunctionCall(name="todo", arguments="{}")
                )
            ],
        ),
        LLMMessage(role=Role.tool, content="done", tool_call_id="c", name="todo"),
    ]


@pytest.mark.asyncio
async def test_restored_legacy_log_mounts_no_header_while_a_stamped_log_does() -> None:
    legacy = project_message_history("session", _unstamped_messages(), None)
    stamped = project_message_history("session", _stamped_messages(), None)
    assert minute_marker_minutes([
        entry for entry in legacy if isinstance(entry, PublicReasoningEntry)
    ]) == [None]

    window = _GatedWindow()
    window.admit(stamped, start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = {unit_id: [_Root()] for unit_id in window.unit_ids}
        for unit_roots in roots.values():
            await stream.mount_all(unit_roots)
        await pilot.pause()
        for unit_id, unit_roots in roots.items():
            window.register_mounted(unit_id, unit_roots)
        await _reconcile(window, stream, pilot)

        # The restored user message opens the minute; the reasoning and the
        # tool call inside it take no redundant stamp of their own.
        assert _headers(stream) == []
        assert window.units[window.unit_ids[0]].head_minute is None


@pytest.mark.asyncio
async def test_restored_legacy_window_mounts_no_headers() -> None:
    legacy = project_message_history("session", _unstamped_messages(), None)
    window = _GatedWindow()
    window.admit(legacy, start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = {unit_id: [_Root()] for unit_id in window.unit_ids}
        for unit_roots in roots.values():
            await stream.mount_all(unit_roots)
        await pilot.pause()
        for unit_id, unit_roots in roots.items():
            window.register_mounted(unit_id, unit_roots)
        await _reconcile(window, stream, pilot)

        assert _headers(stream) == []
        assert all(unit.head_minute is None for unit in window.units.values())


# -- A1: minute transitions inside a tool-group fold ------------------------


def _fold_markers(group: ToolGroup) -> list[MinuteGroupHeader]:
    return [
        child
        for child in group.content_container.children
        if isinstance(child, MinuteGroupHeader)
    ]


def test_a_multi_minute_fold_stamps_each_minute_inside_its_fold() -> None:
    expansion = ToolGroupExpansionState()
    window = _GatedWindow(expansion_state=expansion)
    window.admit(
        [
            _restored_reasoning("r1", 14, 20),
            _effect("e1", 14, 21),
            _restored_reasoning("r2", 14, 21),
            _restored_reasoning("r3", 14, 22),
        ],
        start_index=0,
    )
    assert len(window.unit_ids) == 1
    unit = window.units[window.unit_ids[0]]
    assert unit.group_key == ToolGroupKey("r1")

    (group,) = window.build_unit(unit.id, WeakKeyDictionary())
    assert isinstance(group, ToolGroup)
    assert group.is_collapsed is True
    assert expansion.revision == 0

    children = list(group.content_container.children)
    assert [
        (marker.label, children.index(marker)) for marker in _fold_markers(group)
    ] == [(_label(14, 21), 1), (_label(14, 22), 5)]
    # The fold keeps its own shape: one group, its entries in order, no
    # expansion-store write from the stamps inside it.
    assert isinstance(children[0], ReasoningMessage)
    assert [type(child).__name__ for child in children] == [
        "ReasoningMessage",
        "MinuteGroupHeader",
        "ToolCallMessage",
        "ToolResultMessage",
        "ReasoningMessage",
        "MinuteGroupHeader",
        "ReasoningMessage",
    ]
    assert expansion.revision == 0
    assert expansion.is_collapsed(ToolGroupKey("r1")) is True


@pytest.mark.asyncio
async def test_the_unit_after_a_multi_minute_fold_takes_no_duplicate_marker() -> None:
    window = _GatedWindow()
    window.admit(
        [
            _live_reasoning("r1", 14, 20),
            _effect("e1", 14, 22, live=True),
            _compaction("c1", 14, 22),
            _compaction("c2", 14, 24),
        ],
        start_index=0,
    )
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = {unit_id: [_Root()] for unit_id in window.unit_ids}
        for unit_roots in roots.values():
            await stream.mount_all(unit_roots)
        await pilot.pause()
        for unit_id, unit_roots in roots.items():
            window.register_mounted(unit_id, unit_roots)
        await _reconcile(window, stream, pilot)

        assert len(window.unit_ids) == 3
        fold, later, next_minute = (
            window.units[unit_id] for unit_id in window.unit_ids
        )
        assert [len(fold.entries), len(later.entries), len(next_minute.entries)] == [
            2,
            1,
            1,
        ]
        assert fold.head_minute == _at(14, 20)
        # The fold's last entry already stamped 14:22 inside the fold.
        assert later.head_minute is None
        assert next_minute.head_minute == _at(14, 24)
        assert [header.label for header in _headers(stream)] == [
            _label(14, 20),
            _label(14, 24),
        ]


# -- A3: residency invalidation ---------------------------------------------


@pytest.mark.asyncio
async def test_an_evicted_marker_hands_the_minute_to_the_resident_unit() -> None:
    window = _GatedWindow()
    window.admit(
        [
            _compaction("c1", 14, 20),
            _compaction("c2", 14, 20),
            _compaction("c3", 14, 20),
        ],
        start_index=0,
    )
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        marker_unit, filler, later = window.unit_ids
        # The filler pushes the marker more than a viewport above the reader,
        # so the reconcile restores only the tail it can see.
        roots = {marker_unit: _Root(), filler: _Root(height=80), later: _Root()}
        await stream.mount_all([roots[unit_id] for unit_id in window.unit_ids])
        await pilot.pause()
        for unit_id in window.unit_ids:
            window.register_mounted(unit_id, [roots[unit_id]])
        stream.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        await _reconcile(window, stream, pilot)
        headers = _headers(stream)
        assert [header.label for header in headers] == [_label(14, 20)]
        assert window.units[marker_unit].minute_header is headers[0]

        await window.evict_unit(marker_unit)
        await pilot.pause()
        stream.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        await _reconcile(window, stream, pilot)

        # The marker's unit stays a placeholder, so the minute must be rendered
        # by the resident unit that continues it -- not lost with the header.
        assert window.units[marker_unit].placeholders
        headers = _headers(stream)
        assert [header.label for header in headers] == [_label(14, 20)]
        assert window.units[filler].minute_header is headers[0]
        assert window.units[later].head_minute is None

        # A later same-minute unit mounted while the marker is still evicted
        # takes no second stamp either.
        window.admit([_compaction("c4", 14, 20)], start_index=3)
        tail = window.units[window.unit_ids[-1]]
        tail_roots = [_Root()]
        await stream.mount_all(tail_roots)
        await pilot.pause()
        window.register_mounted(tail.id, tail_roots)
        await _reconcile(window, stream, pilot)
        assert tail.head_minute is None
        assert [header.label for header in _headers(stream)] == [_label(14, 20)]

        # Re-admitting the marker's unit re-derives: the stamp returns to it,
        # still exactly once.
        assert window.units[marker_unit].placeholders
        assert await window.restore_unit(
            marker_unit, stream, WeakKeyDictionary(), follow_bottom=False
        )
        await _reconcile(window, stream, pilot)
        headers = _headers(stream)
        assert [header.label for header in headers] == [_label(14, 20)]
        assert window.units[marker_unit].minute_header is headers[0]


# -- A2: notices rendered as records ----------------------------------------


def _scheduled_loop_notice(entry_id: str, hour: int, minute: int) -> PublicNoticeEntry:
    return PublicNoticeEntry(
        **_base(entry_id),  # type: ignore[arg-type]
        created_at=_epoch_ms(hour, minute),
        turn_id=TURN_ID,
        level="info",
        message="Loop fired",
        detail=ScheduledLoopFiredNoticeDetail(loop_id="2158d161"),
    )


def _root_recovery_notice(entry_id: str, hour: int, minute: int) -> PublicNoticeEntry:
    return PublicNoticeEntry(
        **_base(entry_id),  # type: ignore[arg-type]
        created_at=_epoch_ms(hour, minute),
        turn_id=TURN_ID,
        level="info",
        message="Recovery scheduled",
        detail=RootRecoveryNoticeDetail(milestone="scheduled", episode_id="ep-1"),
    )


def _route_table_notice(entry_id: str, hour: int, minute: int) -> PublicNoticeEntry:
    return PublicNoticeEntry(
        **_base(entry_id),  # type: ignore[arg-type]
        created_at=_epoch_ms(hour, minute),
        turn_id=TURN_ID,
        level="info",
        message="Routes applied",
        detail=RouteTableAppliedNoticeDetail(),
    )


NOTICE_RECORDS = (_scheduled_loop_notice, _root_recovery_notice, _route_table_notice)


class _MountRecorder:
    def __init__(self) -> None:
        self.widgets: list[Widget] = []

    async def __call__(self, widget: Widget, **_kwargs: object) -> None:
        self.widgets.append(widget)


def _live_notice_handler(
    recorder: _MountRecorder, show: dict[str, bool]
) -> EventHandler:
    return EventHandler(
        mount_callback=recorder,
        get_tools_collapsed=lambda: False,
        get_show_message_timestamps=lambda: show["value"],
    )


@pytest.mark.asyncio
async def test_each_stamped_notice_record_stamps_exactly_one_new_minute() -> None:
    recorder = _MountRecorder()
    show = {"value": True}
    handler = _live_notice_handler(recorder, show)
    # A window-admitted record opens 14:20; its stamp is the message/group
    # header below, so the notices landing in that minute take none.
    await handler.handle_event(HistoryEntryAdded(_live_reasoning("r1", 14, 20)))

    for offset, notice in enumerate(NOTICE_RECORDS):
        before = len(recorder.widgets)
        await handler.handle_event(HistoryEntryAdded(notice(f"n{offset}", 14, 20)))
        added = recorder.widgets[before:]
        assert not any(isinstance(widget, MinuteGroupHeader) for widget in added)
        assert isinstance(added[-1], UserCommandMessage)

    for offset, notice in enumerate(NOTICE_RECORDS):
        before = len(recorder.widgets)
        await handler.handle_event(
            HistoryEntryAdded(notice(f"m{offset}", 14, 21 + offset))
        )
        added = recorder.widgets[before:]
        stamps = [widget for widget in added if isinstance(widget, MinuteGroupHeader)]
        assert [stamp.label for stamp in stamps] == [_label(14, 21 + offset)]
        assert isinstance(added[-1], UserCommandMessage)


@pytest.mark.asyncio
async def test_gated_off_notice_records_carry_no_stamp() -> None:
    recorder = _MountRecorder()
    show = {"value": False}
    handler = _live_notice_handler(recorder, show)

    for offset, notice in enumerate(NOTICE_RECORDS):
        before = len(recorder.widgets)
        await handler.handle_event(
            HistoryEntryAdded(notice(f"n{offset}", 14, 21 + offset))
        )
        added = recorder.widgets[before:]
        assert not any(isinstance(widget, MinuteGroupHeader) for widget in added)
        assert isinstance(added[-1], UserCommandMessage)

    # The gate is a rendering choice, not a reason to lose the marker: turning
    # it back on resumes the rule from the records already seen.
    show["value"] = True
    before = len(recorder.widgets)
    await handler.handle_event(HistoryEntryAdded(_scheduled_loop_notice("n9", 14, 24)))
    assert [
        widget.label
        for widget in recorder.widgets[before:]
        if isinstance(widget, MinuteGroupHeader)
    ] == [_label(14, 24)]


@pytest.mark.asyncio
async def test_a_resident_notice_record_suppresses_the_next_unit_marker() -> None:
    # The notice record is part of the visible transcript even though it owns no
    # window unit: while its widget is mounted, its clock advances the
    # transcript-spanning marker.
    window = _GatedWindow()
    window.admit([_live_reasoning("r1", 14, 19)], start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        first_roots = [_Root()]
        await stream.mount_all(first_roots)
        await pilot.pause()
        window.register_mounted(window.unit_ids[0], first_roots)
        await _reconcile(window, stream, pilot)
        assert [header.label for header in _headers(stream)] == [_label(14, 19)]

        window.admit([_scheduled_loop_notice("n1", 14, 20)], start_index=1)
        notice_roots = [_Root()]
        await stream.mount_all(notice_roots)
        await pilot.pause()
        window._bind_stamped_record("n1", notice_roots[0])
        window.admit([_live_reasoning("r2", 14, 20)], start_index=2)
        late = window.units[window.unit_ids[-1]]
        late_roots = [_Root()]
        await stream.mount_all(late_roots)
        await pilot.pause()
        window.register_mounted(late.id, late_roots)
        await _reconcile(window, stream, pilot)

        assert late.head_minute is None
        assert [header.label for header in _headers(stream)] == [_label(14, 19)]


@pytest.mark.asyncio
async def test_an_unmounted_notice_record_opens_the_unit_continuing_its_minute() -> (
    None
):
    # Admissions make a notice part of the visible entry set, but a notice whose
    # record widget never rendered (or was dropped by a rebuild) shows the
    # reader nothing: it must not swallow the unit's only header.
    window = _GatedWindow()
    window.admit([_scheduled_loop_notice("n1", 14, 20)], start_index=0)
    window.admit([_live_reasoning("r1", 14, 20)], start_index=1)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        unit = window.units[window.unit_ids[-1]]
        await stream.mount_all([_Root()])
        await pilot.pause()
        window.register_mounted(unit.id, list(stream.children))
        await _reconcile(window, stream, pilot)

        assert unit.head_minute == _at(14, 20)
        assert [header.label for header in _headers(stream)] == [_label(14, 20)]


@pytest.mark.asyncio
async def test_a_rebuild_that_drops_the_notice_record_restores_the_unit_header() -> (
    None
):
    window = _GatedWindow()
    entries = [_scheduled_loop_notice("n1", 14, 20), _live_reasoning("r1", 14, 20)]
    window.admit(entries, start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        notice_root = _Root()
        await stream.mount_all([notice_root])
        await pilot.pause()
        window._bind_stamped_record("n1", notice_root)
        unit = window.units[window.unit_ids[-1]]
        await stream.mount_all([_Root()])
        await pilot.pause()
        window.register_mounted(unit.id, [stream.children[-1]])
        await _reconcile(window, stream, pilot)
        assert unit.head_minute is None

        # The rebuild drops every child: the notice renders nothing now, so the
        # unit is the minute's only visible record and opens it again.
        window.reset()
        await stream.remove_children()
        window.admit(entries, start_index=0)
        rebuilt = window.units[window.unit_ids[-1]]
        await stream.mount_all([_Root()])
        await pilot.pause()
        window.register_mounted(rebuilt.id, [stream.children[-1]])
        await _reconcile(window, stream, pilot)

        assert rebuilt.head_minute == _at(14, 20)
        assert [header.label for header in _headers(stream)] == [_label(14, 20)]
