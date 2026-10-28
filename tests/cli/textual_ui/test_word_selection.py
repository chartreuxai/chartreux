from __future__ import annotations

from textual import events
from textual.geometry import Offset
from textual.widget import Widget

from chartreux.cli.textual_ui.word_selection import WordSelectScreen


def test_get_widget_and_offset_at_drops_detached_widget(monkeypatch):
    screen = WordSelectScreen(id="_default")
    detached = Widget()
    assert detached.parent is None

    monkeypatch.setattr(
        screen._compositor,
        "get_widget_and_offset_at",
        lambda x, y: (detached, Offset(1, 1)),
    )

    assert screen.get_widget_and_offset_at(0, 0) == (None, None)


def test_get_widget_and_offset_at_keeps_mounted_widget(monkeypatch):
    screen = WordSelectScreen(id="_default")
    parent = Widget()
    child = Widget()
    child._parent = parent
    offset = Offset(2, 3)

    monkeypatch.setattr(
        screen._compositor, "get_widget_and_offset_at", lambda x, y: (child, offset)
    )

    assert screen.get_widget_and_offset_at(0, 0) == (child, offset)


def test_forward_event_delivery_uses_filtered_hit_test(monkeypatch):
    """Double-click delivery must consult the filtered hit-test — the same
    one selection uses — so a DOM-detached widget under the cursor is dropped
    for delivery too (Textualize/textual#6643).
    """
    screen = WordSelectScreen(id="_default")
    leaf = Widget()
    filtered_calls: list[tuple[int, int]] = []
    unfiltered_calls: list[tuple[int, int]] = []

    def filtered(self, x: int, y: int) -> tuple[Widget | None, Offset | None]:
        filtered_calls.append((x, y))
        return leaf, Offset(1, 2)

    def unfiltered(self, x: int, y: int) -> tuple[Widget | None, Offset | None]:
        unfiltered_calls.append((x, y))
        return None, None

    monkeypatch.setattr(WordSelectScreen, "get_widget_and_offset_at", filtered)
    monkeypatch.setattr(WordSelectScreen, "get_widget_at", unfiltered)

    screen._forward_event(
        events.Click(
            widget=leaf,
            x=3,
            y=4,
            delta_x=0,
            delta_y=0,
            button=0,
            shift=False,
            meta=False,
            ctrl=False,
            chain=2,
        )
    )

    assert filtered_calls == [(3, 4)]
    assert unfiltered_calls == []
