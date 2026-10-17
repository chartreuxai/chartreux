"""Browser and catalog screen tests for the provider workbench suite (visibility projection, bookmarks, splits, root actions, provider rows, model list, primary regions, resize, catalog filters)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from textual.widgets import Input, OptionList, SelectionList, Static
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection

from chartreux.core.model_catalog.contracts import (
    DiscoveryError,
    DiscoveryItem,
    DiscoveryResult,
    ModelEdits,
    OptionalEdit,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.presets import FULLY_CUSTOM, MISTRAL
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.ui.providers.management_state import ManagementState, PendingModel
from chartreux.ui.providers.workbench import WorkbenchView
from tests.ui.providers._workbench_support import (
    Host,
    add_preset,
    expand,
    open_split_view,
    press_option,
    setup,
    wait_until,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("view", "editing", "confirm", "stage", "expected"),
    [
        (WorkbenchView.ACTIONS, "base", None, None, {"editor"}),
        (WorkbenchView.ACTIONS, "base", "shared-key", None, {"editor", "confirm"}),
        (
            WorkbenchView.ACTIONS,
            None,
            "discard",
            None,
            {"actions", "provider-operations", "confirm"},
        ),
        (WorkbenchView.PICKER, None, None, None, {"picker"}),
        (WorkbenchView.PROTOCOL, None, None, "connection", {"protocol"}),
        (WorkbenchView.CHOOSE, None, None, "choose", {"choose"}),
        (
            WorkbenchView.CONNECTION,
            None,
            None,
            "connection",
            {"actions", "connection-actions"},
        ),
        (WorkbenchView.MODELS, None, None, "models", {"models", "models-actions"}),
        (WorkbenchView.MODELS, None, None, None, {"models"}),
        (WorkbenchView.CATALOG, None, None, None, {"catalog", "catalog-filter"}),
        (
            WorkbenchView.PRESETS,
            None,
            "discard",
            None,
            {"presets", "presets-actions", "confirm"},
        ),
        (
            WorkbenchView.PRESET_EDITOR,
            None,
            "discard",
            None,
            {"preset-editor", "preset-editor-actions", "confirm"},
        ),
    ],
)
async def test_visibility_projection_is_idempotent(
    view: WorkbenchView,
    editing: str | None,
    confirm: str | None,
    stage: str | None,
    expected: set[str],
) -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._editing = editing
        screen._confirm = confirm
        screen._stage = stage
        screen._activate_view(view)

        def visible_controls() -> set[str]:
            return {
                name
                for name in screen.VIEW_CONTROLS
                if screen.query_one(f"#wb-{name}").display
            }

        assert visible_controls() == expected
        assert screen.focused
        initial_focus = screen.focused.id
        assert initial_focus == (
            "wb-confirm-actions"
            if confirm
            else "wb-input"
            if editing
            else f"wb-{screen.VIEW_GROUPS[view][0]}"
        )
        for size in ((80, 24), (120, 36), (48, 24), (47, 23), (80, 24)):
            await pilot.resize_terminal(*size)
            focus = screen.focused
            for _ in range(3):
                screen._update_help()
                screen._sync_view()
                assert visible_controls() == expected
                assert screen.focused is focus
            await pilot.pause()
            assert visible_controls() == expected
            # Projection preserves focus, including while the minimum-size
            # warning hides the workbench; it never redirects focus implicitly.
            assert screen.focused is focus


@pytest.mark.asyncio
@pytest.mark.parametrize("host_display", [False, True])
async def test_visibility_projection_preserves_host_pending_banner(
    host_display: bool,
) -> None:
    screen, _services = setup()
    async with Host(screen).run_test() as pilot:
        await expand(pilot, screen)
        banner = screen.query_one("#wb-pending-action")
        banner.display = host_display
        for confirm in (None, "discard", None):
            screen._confirm = confirm
            screen._activate_view(WorkbenchView.ACTIONS)
            for _ in range(3):
                screen._update_help()
                screen._sync_view()
                assert banner.display is host_display
                assert banner.has_class("wb-confirm-underlay") is bool(
                    confirm and host_display
                )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("replacement", "disabled", "expected"),
    [
        (("new", "a", "c", "d"), (), "c"),
        (("new", "a", "b", "c"), (), "b"),
        (("a", "b", "c", "d"), ("b", "c"), "a"),
        (("new", "other", "last"), ("new",), "last"),
        ((), (), None),
    ],
)
async def test_group_bookmark_repairs_selectable_identity(
    replacement: tuple[str, ...], disabled: tuple[str, ...], expected: str | None
) -> None:
    screen, _services = setup()
    async with Host(screen).run_test() as pilot:
        rows = screen.query_one("#wb-providers", OptionList)
        rows.clear_options()
        rows.add_options([Option(key, id=key) for key in ("a", "b", "c", "d")])
        rows.highlighted = 1
        position = screen._position_before_rebuild(
            WorkbenchView.PROVIDERS, "wb-providers"
        )
        assert position and position.identities == ("a", "b", "c", "d")
        rows.clear_options()
        rows.add_options([
            Option(key, id=key, disabled=key in disabled) for key in replacement
        ])
        screen._restore_group_position(position)
        await pilot.pause()
        assert (
            rows.highlighted_option.id if rows.highlighted_option else None
        ) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
@pytest.mark.parametrize(
    "view", [WorkbenchView.PROVIDERS, WorkbenchView.ACTIONS, WorkbenchView.CONNECTION]
)
async def test_split_groups_bound_arrows_remember_positions_and_fit(size, view) -> None:  # type: ignore[no-untyped-def]
    screen, services = setup()
    discovery_calls = 0
    original_discover = screen.discovery_service

    async def counted_discover(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal discovery_calls
        discovery_calls += 1
        return await original_discover(*args, **kwargs)

    screen.discovery_service = counted_discover
    async with Host(screen).run_test(size=size) as pilot:
        if view == WorkbenchView.ACTIONS:
            await expand(pilot, screen)
        elif view == WorkbenchView.CONNECTION:
            await add_preset(pilot, screen, FULLY_CUSTOM)
        groups = screen._focus_groups()
        assert groups == [f"wb-{name}" for name in screen.VIEW_GROUPS[view]]
        payload = screen.state.catalog.model_dump() if screen.state else None
        connection = screen.state.connection if screen.state else screen._add
        positions = []
        for group in groups:
            widget = screen.query_one(f"#{group}", OptionList)
            assert widget.display and widget.region.height > 0
            assert widget.region.bottom <= screen.query_one("#wb-help").region.y
            assert widget.has_focus
            await pilot.press("home")
            first = widget.highlighted
            await pilot.press("up", "k", "up")
            assert widget.highlighted == first and widget.has_focus
            await pilot.press("end")
            last = widget.highlighted
            await pilot.press("down", "j", "down", "space")
            assert widget.highlighted == last and widget.has_focus
            assert widget.highlighted_option
            positions.append(widget.highlighted_option.id)
            await pilot.press("tab")
        for group, identity in zip(groups, positions, strict=True):
            widget = screen.query_one(f"#{group}", OptionList)
            assert widget.highlighted_option
            assert widget.has_focus and widget.highlighted_option.id == identity
            await pilot.press("tab")
        await pilot.press("shift+tab")
        assert screen.focused
        assert screen.focused.id == groups[-1]
        assert screen.query_one("#wb-hint").region.bottom <= size[1]
        assert (
            screen.state.catalog.model_dump() == payload
            if screen.state
            else payload is None
        )
        assert (screen.state.connection if screen.state else screen._add) == connection
        assert not services.writes and not services.keys and discovery_calls == 0
        if view == WorkbenchView.CONNECTION:
            assert (
                sum(
                    option.id == "continue"
                    for group in groups
                    for option in screen.query_one(f"#{group}", OptionList).options
                )
                == 1
            )
            assert not screen.query_one("#wb-connection-form").display


@pytest.mark.asyncio
async def test_root_actions_survive_no_match_filter_and_dirty_rebuild() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state
        screen.state.connection = replace(
            screen.state.connection, api_base="https://dirty.test"
        )
        await pilot.press("escape")
        browser = screen.query_one("#wb-providers", OptionList)
        browser.highlighted = browser.get_option_index("two")
        screen._refresh_browser()
        assert browser.highlighted_option and browser.highlighted_option.id == "two"
        await pilot.press("z", "z", "z")
        assert (
            browser.highlighted_option and browser.highlighted_option.id == "\x00empty"
        )
        await pilot.press("tab")
        actions = screen.query_one("#wb-root-actions", OptionList)
        assert actions.has_focus
        await press_option(pilot, actions, "\x00models")
        assert screen.view == WorkbenchView.CATALOG
        await pilot.press("escape")
        assert actions.highlighted_option
        assert actions.has_focus and actions.highlighted_option.id == "\x00models"
        assert screen.state
        assert screen.filter_text == "zzz" and screen.state.dirty
        assert not services.writes and not services.keys


@pytest.mark.asyncio
@pytest.mark.parametrize("pointer", [False, True])
@pytest.mark.parametrize(
    "group,action,expected",
    [
        ("wb-root-actions", "\x00models", WorkbenchView.CATALOG),
        ("wb-root-actions", "\x00presets", WorkbenchView.PRESETS),
        ("wb-root-actions", "\x00add", WorkbenchView.CONNECTION),
        ("wb-root-actions", "\x00apply", WorkbenchView.PROVIDERS),
        ("wb-root-actions", "\x00discard", WorkbenchView.PROVIDERS),
        ("wb-provider-operations", "discover", WorkbenchView.ACTIONS),
        ("wb-provider-operations", "apply", WorkbenchView.ACTIONS),
        ("wb-provider-operations", "collision:b", WorkbenchView.ACTIONS),
        ("wb-provider-operations", "retry-reload", WorkbenchView.ACTIONS),
        ("wb-provider-operations", "models", WorkbenchView.MODELS),
        ("wb-provider-operations", "presets", WorkbenchView.PRESETS),
        ("wb-provider-operations", "discard", WorkbenchView.ACTIONS),
        ("wb-connection-actions", "continue", WorkbenchView.MODELS),
    ],
)
async def test_moved_action_pointer_keyboard_parity(
    pointer, group, action, expected
) -> None:  # type: ignore[no-untyped-def]
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        if group == "wb-provider-operations" or action in {"\x00apply", "\x00discard"}:
            await expand(pilot, screen)
            assert screen.state
            if action.removeprefix("\x00") in {"apply", "discard"}:
                screen.state.connection = replace(
                    screen.state.connection, api_base="https://dirty.test"
                )
            elif action == "collision:b":
                screen.state.select("b")
            elif action == "retry-reload":
                screen._reload_failed = True
            screen._refresh_actions()
            if group == "wb-root-actions":
                screen._refresh_browser()
                await pilot.press("escape")
        elif group == "wb-connection-actions":
            await add_preset(pilot, screen, FULLY_CUSTOM)
            assert screen._add
            screen._add = replace(
                screen._add, name="new", api_base="https://new.test/v1"
            )
            screen._refresh_add_connection()
        widget = screen.query_one(f"#{group}", OptionList)
        if pointer:
            index = widget.get_option_index(action)
            await pilot.click(widget, offset=(8, index))
        else:
            await press_option(pilot, widget, action)
        await wait_until(pilot, lambda: not screen._busy)
        assert screen.view == expected
        assert bool(screen._confirm) is (
            action in {"discard", "\x00discard", "collision:b"}
        )
        assert len(services.writes) == (
            1 if action in {"continue", "apply", "\x00apply"} else 0
        )
        if action in {"apply", "\x00apply"}:
            assert services.writes[0].provider["api_base"] == "https://dirty.test"
        assert not services.keys


@pytest.mark.asyncio
async def test_group_bookmarks_isolate_providers_and_roles() -> None:
    screen, _services = setup()
    async with Host(screen).run_test() as pilot:
        await expand(pilot, screen)
        assert screen.state
        actions = screen.query_one("#wb-actions", OptionList)
        actions.highlighted = actions.get_option_index("style")
        operations = screen.query_one("#wb-provider-operations", OptionList)
        operations.highlighted = operations.get_option_index("models")
        screen.state.for_provider("two")
        screen._refresh_actions()
        assert (
            operations.highlighted_option
            and operations.highlighted_option.id == "discover"
        )
        operations.highlighted = operations.get_option_index("presets")
        assert actions.highlighted_option and actions.highlighted_option.id == "base"
        actions.highlighted = actions.get_option_index("env")
        screen.state.for_provider("one")
        screen._refresh_actions()
        assert actions.highlighted_option and actions.highlighted_option.id == "style"
        assert (
            operations.highlighted_option
            and operations.highlighted_option.id == "models"
        )
        screen.state.for_provider("two")
        screen._refresh_actions()
        assert (
            operations.highlighted_option
            and operations.highlighted_option.id == "presets"
        )
        assert actions.highlighted_option and actions.highlighted_option.id == "env"

        screen._preset_role = "orchestrator"
        screen._preset_draft = screen.state.preset("orchestrator")
        screen._refresh_preset_editor()
        editor = screen.query_one("#wb-preset-editor", OptionList)
        editor.highlighted = editor.get_option_index("thinking")
        screen._preset_role = "other"
        screen._preset_draft = screen.state.preset("other")
        screen._refresh_preset_editor()
        assert editor.highlighted_option and editor.highlighted_option.id == "model"
        editor_actions = screen.query_one("#wb-preset-editor-actions", OptionList)
        editor_actions.highlighted = editor_actions.get_option_index("cancel")
        screen._preset_role = "orchestrator"
        screen._refresh_preset_editor()
        assert editor.highlighted_option and editor.highlighted_option.id == "thinking"
        assert (
            editor_actions.highlighted_option
            and editor_actions.highlighted_option.id == "apply"
        )
        screen._preset_role = "other"
        screen._refresh_preset_editor()
        assert editor.highlighted_option and editor.highlighted_option.id == "model"
        assert (
            editor_actions.highlighted_option
            and editor_actions.highlighted_option.id == "cancel"
        )


@pytest.mark.asyncio
async def test_pending_rename_bookmark_uses_wire_not_selection_value() -> None:
    screen, _services = setup()
    async with Host(screen).run_test() as pilot:
        await expand(pilot, screen)
        assert screen.state
        screen.state.pending["wire-pending"] = PendingModel(
            "wire-pending", "old-name", True
        )
        screen._refresh_models()
        models = screen.query_one("#wb-models", SelectionList)
        models.highlighted = next(
            i
            for i, option in enumerate(models.options)
            if cast(Selection[str], option).value == "old-name"
        )
        screen.state.pending["wire-pending"] = replace(
            screen.state.pending["wire-pending"], canonical_name="z-renamed"
        )
        screen.state.pending["other-wire"] = PendingModel(
            "other-wire", "old-name", True
        )
        screen._refresh_models()
        assert models.highlighted is not None
        assert (
            cast(Selection[str], models.get_option_at_index(models.highlighted)).value
            == "z-renamed"
        )
        assert "z-renamed" in models.selected and "old-name" in models.selected


@pytest.mark.asyncio
async def test_nonfocused_rebuild_keeps_focus_and_contents_owner() -> None:
    screen, _services = setup()
    async with Host(screen).run_test() as pilot:
        await expand(pilot, screen)
        assert screen.state
        screen._refresh_models()
        models = screen.query_one("#wb-models", SelectionList)
        actions = screen.query_one("#wb-actions", OptionList)
        actions.focus()
        screen.state.for_provider("two")
        screen._refresh_models()
        await pilot.pause()
        assert actions.has_focus and not models.display
        assert (
            WorkbenchView.MODELS,
            "wb-models",
            ("one", "management"),
        ) in screen._positions
        assert not any(
            view == WorkbenchView.ACTIONS and group == "wb-models"
            for view, group, _context in screen._positions
        )
        screen._choose_preset(FULLY_CUSTOM)
        screen._connection_action("name")
        await pilot.pause()
        focused = screen.focused
        assert focused is screen.query_one("#wb-input", Input)
        screen._refresh_add_connection()
        await pilot.pause()
        assert screen.focused is focused


@pytest.mark.asyncio
async def test_catalog_bookmarks_isolate_applied_filter_and_frame_snapshot() -> None:
    screen, _services = setup()
    async with Host(screen).run_test() as pilot:
        screen._open_catalog()
        await pilot.pause()
        rows = screen.query_one("#wb-catalog", OptionList)
        rows.focus()
        rows.highlighted = rows.get_option_index("model:b")
        opener = screen._frame()
        assert opener.position and opener.context == ("all",)
        screen._model_filter = "one"
        screen._refresh_catalog()
        assert rows.highlighted_option and rows.highlighted_option.id == "model:a"
        screen._model_filter = None
        screen._refresh_catalog()
        assert rows.highlighted_option and rows.highlighted_option.id == "model:b"
        rows.highlighted = rows.get_option_index("model:a")
        screen._capture_group_position(rows)
        screen._restore_frame(opener)
        await pilot.pause()
        assert (
            rows.has_focus
            and rows.highlighted_option
            and rows.highlighted_option.id == "model:b"
        )
        assert (
            WorkbenchView.GLOBAL_CATALOG,
            "wb-catalog",
            ("all",),
        ) in screen._positions


@pytest.mark.asyncio
async def test_group_bookmark_scroll_and_credential_caret_without_text() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state
        for index in range(60):
            wire = f"long-{index:02}"
            screen.state.pending[wire] = PendingModel(wire, wire, True)
        screen._refresh_models()
        screen._activate_view(WorkbenchView.MODELS)
        models = screen.query_one("#wb-models", SelectionList)
        models.highlighted = 40
        await pilot.pause()
        models.scroll_to(y=25, animate=False, immediate=True, force=True)
        before = models.scroll_y
        assert before > 0
        screen._refresh_models()
        await pilot.pause()
        assert models.scroll_y == before and models.highlighted == 40
        screen._choose_preset(MISTRAL)
        screen._connection_action("key")
        await pilot.pause()
        editor = screen.query_one("#wb-input", Input)
        editor.value = "bookmark-must-not-retain-this"
        editor.cursor_position = 4
        frame = screen._frame()
        assert frame.position and frame.position.cursor_position == 4
        assert "bookmark-must-not-retain-this" not in repr(frame)
        assert "bookmark-must-not-retain-this" not in repr(screen._positions)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_composite_model_groups_are_reversible_and_bounded(size) -> None:  # type: ignore[no-untyped-def]
    screen, services = setup()
    async with Host(screen).run_test(size=size) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        screen._connection_action("continue")
        await wait_until(pilot, lambda: not screen._busy)
        models = screen.query_one("#wb-models", SelectionList)
        actions = screen.query_one("#wb-models-actions", OptionList)
        models.highlighted = models.option_count - 1
        models.focus()
        selected = list(models.selected)
        await pilot.press("down", "right", "left")
        assert models.has_focus and models.highlighted == models.option_count - 1
        await pilot.press("tab")
        assert actions.has_focus
        actions.highlighted = actions.option_count - 1
        await pilot.press("down", "right", "left")
        assert actions.has_focus and actions.highlighted == actions.option_count - 1
        await pilot.press("tab")
        assert screen.query_one("#wb-help").has_focus
        assert "Tab Models" in str(screen.query_one("#wb-hint").render())
        assert "Shift+Tab Actions" in str(screen.query_one("#wb-hint").render())
        await pilot.press("tab")
        assert models.has_focus and models.highlighted == models.option_count - 1
        await pilot.press("shift+tab", "shift+tab")
        assert actions.has_focus and actions.highlighted == actions.option_count - 1
        await pilot.press("shift+tab")
        assert models.has_focus
        models.highlighted = 0
        await pilot.press("up", "k")
        assert models.has_focus and models.highlighted == 0
        await pilot.press("tab")
        actions.highlighted = 0
        await pilot.press("up", "k")
        assert actions.has_focus and actions.highlighted == 0
        assert list(models.selected) == selected
        assert len(services.writes) == 1  # connection only
        await pilot.resize_terminal(120, 36)
        await pilot.press("tab", "tab")
        assert models.has_focus and models.highlighted == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_catalog_groups_bound_arrows_and_repair_filtered_identity(size) -> None:  # type: ignore[no-untyped-def]
    screen, services = setup()
    async with Host(screen).run_test(size=size) as pilot:
        screen._open_catalog()
        await pilot.pause()
        models = screen.query_one("#wb-catalog", OptionList)
        filters = screen.query_one("#wb-catalog-filter", OptionList)
        models.highlighted = models.option_count - 1
        await pilot.press("down", "j", "right", "left")
        assert models.has_focus and models.highlighted == models.option_count - 1
        await pilot.press("tab")
        assert screen.query_one("#wb-help").has_focus
        await pilot.press("tab")
        assert filters.has_focus
        filters.highlighted = filters.option_count - 1
        await pilot.press("down", "j")
        assert filters.has_focus and filters.highlighted == filters.option_count - 1
        await pilot.press("shift+tab", "shift+tab")
        assert models.has_focus and models.highlighted == models.option_count - 1
        models.highlighted = 0
        await pilot.press("up", "k")
        assert models.highlighted == 0
        await pilot.press("shift+tab")
        assert filters.has_focus and filters.highlighted == filters.option_count - 1
        filters.highlighted = 1
        await pilot.press("up", "k")
        assert filters.highlighted == 1  # heading is not selectable
        await press_option(pilot, filters, "filter:one")
        assert models.highlighted_option and models.highlighted_option.id == "model:a"
        screen._model_filter = "missing"
        screen._refresh_catalog()
        await pilot.press("tab")
        assert models.has_focus and models.highlighted_option.id == "\x00empty"
        await pilot.press("down", "enter")
        assert screen.view is WorkbenchView.CATALOG
        await pilot.press("tab", "tab")
        assert filters.has_focus
        assert "Help" in str(screen.query_one("#wb-help").render())
        assert "Shift+Tab" in str(screen.query_one("#wb-hint").render())
        assert not services.writes


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_model_details_pointer_is_separate_and_restores_identity_scroll(
    size,
) -> None:  # type: ignore[no-untyped-def]
    screen, services = setup()
    services.discovery_result = DiscoveryResult(
        tuple(DiscoveryItem(f"model-{i:03d}") for i in range(60))
    )
    async with Host(screen).run_test(size=size) as pilot:
        await expand(pilot, screen)
        screen._select_action("discover")
        await wait_until(pilot, lambda: not screen._busy)
        screen._select_action("models")
        await pilot.pause()
        models = screen.query_one("#wb-models", SelectionList)
        await pilot.click(models, offset=(10, 0))
        await pilot.pause()
        assert models.get_option_at_index(0).value not in models.selected
        assert screen.state is not None
        screen.state.select("model-039")
        screen._refresh_models()
        await pilot.pause()
        models.highlighted = 40
        models.scroll_to_highlight()
        await pilot.pause()
        identity = models.get_option_at_index(40).value
        before = list(models.selected)
        scroll = models.scroll_offset.y
        row = 40 - scroll
        assert "Details" in models.render_line(row).text
        await pilot.click(models, offset=(models.size.width - 5, row))
        await pilot.pause()
        assert screen._detail == identity
        assert list(models.selected) == before
        await pilot.press("escape")
        await pilot.pause()
        assert (
            models.has_focus
            and models.get_option_at_index(models.highlighted).value == identity
        )
        assert models.scroll_offset.y == scroll
        await pilot.press("enter")
        assert screen._detail == identity
        await pilot.press("escape")
        assert not services.writes


@pytest.mark.asyncio
async def test_models_empty_groups_busy_and_confirmation_remain_usable() -> None:
    screen, services = setup()
    services.discovery_result = DiscoveryResult(())
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        screen._connection_action("continue")
        await wait_until(pilot, lambda: not screen._busy)
        models = screen.query_one("#wb-models", SelectionList)
        actions = screen.query_one("#wb-models-actions", OptionList)
        assert cast(Selection[str], models.highlighted_option).value == "\x00empty"
        await pilot.press("up", "down", "space", "enter", "right")
        assert models.has_focus and screen._detail is None
        assert not models.selected
        await pilot.press("tab")
        assert actions.has_focus
        await pilot.press("shift+tab")
        actions.disabled = True
        await pilot.press("tab")
        assert screen.query_one("#wb-help").has_focus
        assert "Tab Models" in str(screen.query_one("#wb-hint").render())
        actions.disabled = False
        await pilot.press("tab")
        screen._busy = True
        await pilot.press("tab", "shift+tab", "enter")
        assert models.has_focus and screen._detail is None
        screen._busy = False
        screen._confirm = "discard"
        screen._update_help()
        await pilot.pause()
        await pilot.press("tab", "shift+tab")
        assert screen.query_one("#wb-confirm-actions").has_focus
        await pilot.press("escape")
        assert screen._confirm is None and models.has_focus
        assert len(services.writes) == 1


@pytest.mark.asyncio
async def test_models_refresh_repairs_removed_identity_to_nearest_row() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(120, 36)) as pilot:
        await expand(pilot, screen)
        screen._select_action("discover")
        await pilot.pause()
        await wait_until(pilot, lambda: not screen._busy)
        screen._select_action("models")
        await pilot.pause()
        models = screen.query_one("#wb-models", SelectionList)
        models.highlighted = 1
        identity = cast(Selection[str], models.highlighted_option).value
        cast(ManagementState, screen.state).discovery = DiscoveryResult((
            DiscoveryItem("new"),
        ))
        screen._refresh_models()
        await pilot.pause()
        assert models.has_focus and models.highlighted == 1
        assert cast(Selection[str], models.highlighted_option).value != identity
        await pilot.press("tab", "shift+tab")
        assert models.has_focus and models.highlighted == 1
        assert not services.writes


@pytest.mark.asyncio
async def test_provider_double_click_cannot_activate_new_view_at_same_position() -> (
    None
):
    screen, _services = setup()
    app = Host(screen)
    async with app.run_test(size=(80, 24)) as pilot:
        browser = screen.query_one("#wb-providers", OptionList)
        await pilot.click(browser, offset=(15, 2), times=2)
        await pilot.pause()
        assert screen.view is WorkbenchView.ACTIONS
        assert screen.query_one("#wb-actions", OptionList).display
        assert not screen.query_one("#wb-protocol", OptionList).display
        await pilot.pause(0.6)
        await pilot.click(screen.query_one("#wb-actions", OptionList), offset=(15, 1))
        await pilot.pause()
        assert screen.view is WorkbenchView.PROTOCOL


@pytest.mark.asyncio
async def test_provider_action_labels_distinguish_draft_from_disk() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        actions = screen.query_one("#wb-provider-operations", OptionList)
        ids = {option.id for option in actions.options}
        assert "name" not in ids
        assert "Provider: one" in str(screen.query_one("#wb-filter").render())
        assert "to disk" in str(
            next(o.prompt for o in actions.options if o.id == "apply")
        )
        assert "all pending" in str(
            next(o.prompt for o in actions.options if o.id == "discard")
        )
        await press_option(pilot, actions, "models")
        assert "click/Space toggles inclusion" in str(
            screen.query_one("#wb-filter").render()
        )
        screen._open_detail("a")
        await pilot.pause()
        assert "Apply all catalog edits to disk later." in str(
            screen.query_one("#wb-help").render()
        )
        detail = screen.query_one("#wb-detail-actions", OptionList)
        assert "catalog draft" in str(
            next(o.prompt for o in detail.options if o.id == "save-detail")
        )
        await press_option(pilot, detail, "save-detail")
        assert screen.view is WorkbenchView.MODELS
        assert not services.writes


@pytest.mark.asyncio
async def test_provider_ascii_footer_and_neutral_confirmation_fit() -> None:
    screen, _services = setup()
    app = Host(screen)
    setattr(app, "config", SimpleNamespace(ascii_chrome=True))  # noqa: B010
    async with app.run_test(size=(80, 24)) as pilot:
        screen._open_catalog()
        await pilot.pause()
        hint = str(screen.query_one("#wb-hint").render())
        assert "Tab Help" in hint and "Shift+Tab Filter" in hint and "Esc Back" in hint
        assert not any(glyph in hint for glyph in "↑↓←→")
        await pilot.press("escape")
        await expand(pilot, screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://changed.test"
        await pilot.press("enter")
        screen._select_action("discard")
        await pilot.pause()
        confirm = screen.query_one("#wb-confirm")
        assert confirm.styles.border.top[0] == "solid"
        assert confirm.border_title == "Confirm change"
        confirm_hint = str(screen.query_one("#wb-confirm-help").render())
        assert "Up/Down" in confirm_hint and "↑↓" not in confirm_hint


@pytest.mark.asyncio
async def test_root_filter_escape_and_draft_runnable_count() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        browser = screen.query_one("#wb-providers", OptionList)
        one = next(option for option in browser.options if option.id == "one")
        assert "1 runnable model" in str(one.prompt)
        assert "Filter: type to filter" in str(screen.query_one("#wb-filter").render())
        screen.filter_text = "one"
        screen._refresh_browser()
        assert "1 of 2 providers" in str(screen.query_one("#wb-filter").render())
        await pilot.press("escape")
        assert screen.filter_text == ""

    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        browser = screen.query_one("#wb-providers", OptionList)
        state = screen.state
        assert state
        state.enabled_by_deployment["a", "one"] = False
        screen._refresh_browser()
        one = next(option for option in browser.options if option.id == "one")
        assert "0 runnable models" in str(one.prompt)


@pytest.mark.asyncio
async def test_root_draft_can_reenable_disabled_deployment() -> None:
    screen, _services = setup()
    data = screen.snapshot.catalog.model_dump()
    data["models"]["a"]["deployments"][0]["disabled"] = True
    screen.snapshot = CatalogSnapshot(
        ModelCatalog.model_validate(data), "disabled-deployment"
    )
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        browser = screen.query_one("#wb-providers", OptionList)
        one = next(option for option in browser.options if option.id == "one")
        assert "0 runnable models" in str(one.prompt)
        await expand(pilot, screen)
        state = screen.state
        assert state
        state.enabled_by_deployment["a", "one"] = True
        screen._refresh_browser()
        one = next(option for option in browser.options if option.id == "one")
        assert "1 runnable model" in str(one.prompt)


@pytest.mark.asyncio
async def test_provider_rows_keep_disabled_and_pending_status_after_switch() -> None:
    screen, _services = setup()
    data = screen.snapshot.catalog.model_dump()
    data["providers"]["one"]["disabled"] = True
    data["models"]["b"]["disabled"] = True
    screen.snapshot = CatalogSnapshot(ModelCatalog.model_validate(data), "disabled-b")
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        one = next(
            option
            for option in screen.query_one("#wb-providers", OptionList).options
            if option.id == "one"
        )
        assert "Disabled" in str(one.prompt)
        await expand(pilot, screen)
        assert screen.state
        screen._refresh_browser()
        two = next(
            option
            for option in screen.query_one("#wb-providers", OptionList).options
            if option.id == "two"
        )
        assert "No enabled models" in str(two.prompt)
        state = screen.state
        assert state
        state.pending_by_provider["two"]["pending"] = PendingModel(
            "pending", "pending", True
        )
        state.discoveries["two"] = DiscoveryError("connection", "failed")
        screen._view = WorkbenchView.PROVIDERS
        screen.query_one("#wb-providers").display = True
        screen._refresh_browser()
        two = next(
            option
            for option in screen.query_one("#wb-providers", OptionList).options
            if option.id == "two"
        )
        assert "No enabled models" not in str(two.prompt)
        assert "1 runnable model" in str(two.prompt)
        assert "Discovery Failed" in str(two.prompt)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_global_catalog_escape_restores_opening_provider_and_draft(size) -> None:  # type: ignore[no-untyped-def]
    screen, _services = setup()
    async with Host(screen).run_test(size=size) as pilot:
        await pilot.press("enter")
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "base")
        screen.query_one("#wb-input", Input).value = "https://changed.test"
        await pilot.press("enter", "escape")

        await press_option(
            pilot, screen.query_one("#wb-root-actions", OptionList), "\x00models"
        )
        await press_option(
            pilot, screen.query_one("#wb-catalog", OptionList), "model:b"
        )
        assert screen._detail == "b"
        await pilot.press("escape", "escape")

        assert screen.state and screen.state.provider_id == "one"
        assert screen.state.connection.api_base == "https://changed.test"
        assert screen.query_one("#wb-actions").display is False
        browser = screen.query_one("#wb-providers", OptionList)
        assert browser.display and screen.query_one("#wb-root-actions").has_focus


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_discard_confirmation_restores_browser_focus(size) -> None:  # type: ignore[no-untyped-def]
    screen, _services = setup()
    async with Host(screen).run_test(size=size) as pilot:
        await pilot.press("enter")
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "base")
        screen.query_one("#wb-input", Input).value = "https://changed.test"
        await pilot.press("enter", "escape")
        await press_option(
            pilot, screen.query_one("#wb-root-actions", OptionList), "\x00discard"
        )
        await pilot.press("down", "enter")

        browser = screen.query_one("#wb-providers", OptionList)
        assert browser.display and screen.query_one("#wb-root-actions").has_focus
        await pilot.press("tab")
        assert browser.has_focus
        await press_option(pilot, browser, "two")
        assert screen.state and screen.state.provider_id == "two"
        assert screen.query_one("#wb-actions").has_focus


@pytest.mark.asyncio
async def test_model_actions_are_pinned_outside_checklist() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        screen._connection_action("continue")
        await pilot.pause()
        models = screen.query_one("#wb-models", SelectionList)
        values = [str(cast(Selection[str], option).value) for option in models.options]
        assert not any(
            value.startswith("\x00") and value != "\x00empty" for value in values
        )
        actions = screen.query_one("#wb-models-actions", OptionList)
        assert [option.id for option in actions.options] == [
            "retry-discovery",
            "edit-connection",
            "manual",
            "add-another",
            "continue-presets",
        ]
        assert actions.display and actions.region.bottom <= screen.app.size.height
        models.highlighted = 0
        models.focus()
        screen._update_help()
        assert "Space" in str(screen.query_one("#wb-help").render())
        assert "Enter" in str(screen.query_one("#wb-help").render())
        await pilot.press("tab")
        assert actions.has_focus
        await pilot.press("shift+tab")
        assert models.has_focus and models.highlighted == 0


@pytest.mark.asyncio
async def test_below_minimum_shows_size_requirement() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(47, 23)) as pilot:
        await pilot.pause()
        assert screen.query_one("#wb-small").display
        assert not screen.query_one("#workbench").display


@pytest.mark.asyncio
async def test_busy_resize_toggles_size_fallback_without_clearing_busy() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._set_busy(True)

        await pilot.resize_terminal(47, 23)
        await pilot.pause()
        assert screen.query_one("#wb-small").display
        assert not screen.query_one("#workbench").display
        assert screen._busy
        assert "Resize to continue" in str(
            screen.query_one("#wb-small", Static).content
        )

        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert not screen.query_one("#wb-small").display
        assert screen.query_one("#workbench").display
        assert screen._busy


@pytest.mark.asyncio
async def test_filter_precedes_collapse_and_count_tracks_draft() -> None:
    screen, _services = setup(active="b")
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("models")
        screen.query_one("#wb-models", SelectionList).deselect("a")
        await pilot.pause()
        browser = screen.query_one("#wb-providers", OptionList)
        assert "0 runnable models" in str(browser.get_option_at_index(1).prompt)
        screen.filter_text = "one"
        screen._refresh_browser()
        await pilot.press("escape")
        assert screen.filter_text == "one"
        assert not screen.query_one("#wb-models").display
        await pilot.press("escape")
        assert screen.query_one("#wb-providers").display
        await pilot.press("escape")
        assert not screen.filter_text


@pytest.mark.asyncio
async def test_workbench_one_primary_region_and_confirmation_scope() -> None:
    screen, _ = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.query_one("#wb-actions").display
        assert not screen.query_one("#wb-providers").display
        assert "Provider: one" in str(screen.query_one("#wb-filter").render())
        screen._select_action("models")
        assert screen.query_one("#wb-models").display
        assert not screen.query_one("#wb-actions").display
        screen._open_detail("a")
        await pilot.pause()
        assert screen.query_one("#wb-detail-fields").display
        assert not screen.query_one("#wb-models").display
        assert "Model details: a" in str(screen.query_one("#wb-filter").render())
        screen.query_one("#price-input", Input).value = "-3"
        screen._save_detail()
        assert screen.query_one("#price-input", Input).has_class("-invalid")
        assert "Error:" in str(screen.query_one("#error-input").render())
        screen.query_one("#price-input", Input).value = "2"
        screen._save_detail()
        screen._select_action("discard")
        confirm_actions = screen.query_one("#wb-confirm-actions", OptionList)
        assert confirm_actions.highlighted_option is not None
        assert confirm_actions.highlighted_option.id == "cancel"
        assert "all catalog edits across providers and global roles" in str(
            screen.query_one("#wb-confirm-text").render()
        )
        assert "Saved credentials remain saved" in str(
            screen.query_one("#wb-confirm-text").render()
        )
        await pilot.press("escape")
        assert screen.state and screen.state.dirty


@pytest.mark.asyncio
async def test_empty_model_row_is_not_a_toggle() -> None:
    screen, _ = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state
        screen.state.for_provider("two")
        screen._refresh_models()
        screen.state.for_provider("one")
        screen.state.snapshot = CatalogSnapshot(
            screen.state.snapshot.catalog.model_copy(update={"models": {}}), "empty"
        )
        screen._select_action("models")
        screen._refresh_models()
        models = screen.query_one("#wb-models", SelectionList)
        assert models.highlighted == 0
        assert "No models configured" in str(models.get_option_at_index(0).prompt)
        await pilot.press("enter", "space")
        assert not models.selected


@pytest.mark.asyncio
async def test_catalog_refuses_details_for_unconfigured_discovered_model() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        state = screen.state
        assert state is not None
        state.discovery = services.discovery_result
        screen._open_catalog()
        await pilot.pause()
        assert screen._stage != "models"
        screen._open_detail("new")
        await pilot.pause()
        assert screen.view is WorkbenchView.CATALOG
        assert screen._detail is None
        assert screen._detail_preview_wire is None
        assert screen._message == "Enable this discovered model before editing details."
        assert "new" not in state.pending
        assert not services.writes


@pytest.mark.asyncio
async def test_primary_region_visibility_and_focus_across_workbench_views() -> None:
    screen, _services = setup()
    primary_ids = (
        "wb-providers",
        "wb-actions",
        "wb-choose",
        "wb-picker",
        "wb-catalog",
        "wb-models",
        "wb-detail-fields",
        "wb-editor",
        "wb-protocol",
        "wb-presets",
        "wb-confirm",
    )

    def assert_primary() -> None:
        visible = [screen.query_one(f"#{name}") for name in primary_ids]
        shown = [widget for widget in visible if widget.display]
        assert len(shown) == 1
        _, legal = screen._visibility_projection()
        assert {
            name
            for name in screen.VIEW_CONTROLS
            if screen.query_one(f"#wb-{name}").display
        } == legal - {"help", "hint"}
        focused = screen.focused
        assert focused is not None
        assert any(
            screen.query_one(f"#wb-{name}") in focused.ancestors_with_self
            for name in legal
        )

    async with Host(screen).run_test(size=(80, 24)) as pilot:
        assert_primary()
        await expand(pilot, screen)
        assert_primary()
        screen._select_action("models")
        assert_primary()
        screen._open_detail("a")
        await pilot.pause()
        assert_primary()
        await pilot.press("escape")
        assert_primary()
        await pilot.press("escape")
        assert_primary()
        screen._open_presets()
        assert_primary()


@pytest.mark.asyncio
@pytest.mark.parametrize("action_id", ["retry-discovery", "edit-connection", "manual"])
async def test_resize_preserves_highlighted_pinned_model_action(action_id: str) -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(120, 36)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        screen._connection_action("continue")
        await pilot.pause()
        actions = screen.query_one("#wb-models-actions", OptionList)
        actions.highlighted = next(
            i for i, option in enumerate(actions.options) if option.id == action_id
        )
        actions.focus()
        assert actions.highlighted_option and actions.highlighted_option.id == action_id

        await pilot.resize_terminal(100, 30)

        assert actions.highlighted_option and actions.highlighted_option.id == action_id
        assert actions.has_focus


@pytest.mark.asyncio
async def test_large_provider_filter_scroll_keeps_catalog_model_visible() -> None:
    screen, _services = setup()
    catalog_data = screen.snapshot.catalog.model_dump()
    catalog_data["providers"].update({
        f"provider-{index:02d}": {"api_base": f"https://p{index:02d}.test"}
        for index in range(24)
    })
    screen.snapshot = CatalogSnapshot(
        ModelCatalog.model_validate(catalog_data), "many-providers"
    )
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_catalog()
        filters = screen.query_one("#wb-catalog-filter", OptionList)
        assert filters.option_count >= 26
        filters.focus()
        filters.highlighted = filters.option_count - 1
        filters.scroll_to_highlight()
        await pilot.pause()
        assert filters.scroll_offset.y > 0
        assert screen.query_one("#wb-catalog").display
        assert any(
            option.id == "model:a"
            for option in screen.query_one("#wb-catalog", OptionList).options
        )


@pytest.mark.asyncio
async def test_catalog_filter_cursor_survives_resize_before_filter_is_applied() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(120, 36)) as pilot:
        screen._open_catalog()
        filters = screen.query_one("#wb-catalog-filter", OptionList)
        filters.focus()
        await pilot.press("down")
        assert filters.highlighted_option is not None
        assert filters.highlighted_option.id == "filter:one"
        assert screen._model_filter is None

        await pilot.resize_terminal(100, 30)

        assert filters.highlighted_option is not None
        assert filters.highlighted_option.id == "filter:one"
        assert screen._model_filter is None
        await pilot.press("enter")
        assert screen._model_filter == "one"


@pytest.mark.asyncio
async def test_shared_pending_canonical_catalog_owner_picker_opens_live_drafts() -> (
    None
):
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        screen.state.select("wire-new-one", canonical_name="new")
        screen.state.for_provider("two")
        screen.state.select("wire-new-two", canonical_name="new")
        screen.state.for_provider("one")
        screen._open_catalog()
        screen._select_catalog("model:new")
        picker = screen.query_one("#wb-picker", OptionList)
        assert picker.display
        assert {str(option.id) for option in picker.options} == {
            "deployment:one:new",
            "deployment:two:new",
        }
        picker.highlighted = 0
        await pilot.press("enter")
        await pilot.pause()
        assert screen._detail == "new"
        assert screen.state and screen.state.provider_id == "one"
        await pilot.press("escape")
        assert screen.query_one("#wb-catalog").display
        screen._select_catalog("filter:two")
        assert screen._model_filter == "two"
        screen._select_catalog("model:new")
        await pilot.pause()
        assert screen._detail == "new"
        assert screen.state and screen.state.provider_id == "two"
        await pilot.press("escape")
        assert screen.query_one("#wb-catalog").display


@pytest.mark.asyncio
async def test_global_catalog_filter_includes_pending_draft_provider() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        screen._connection_action("continue")
        await pilot.pause()
        models = screen.query_one("#wb-models", SelectionList)
        models.select("new")
        await pilot.pause()

        screen._open_catalog()
        filters = screen.query_one("#wb-catalog-filter", OptionList)
        assert "filter:mistral" in {str(option.id) for option in filters.options}
        screen._select_catalog("filter:mistral")
        assert screen._model_filter == "mistral"
        assert "model:new" in {
            str(option.id)
            for option in screen.query_one("#wb-catalog", OptionList).options
        }


@pytest.mark.asyncio
async def test_catalog_filter_focus_help_and_escape_levels() -> None:
    screen, _services = setup()
    catalog_data = screen.snapshot.catalog.model_dump()
    catalog_data["providers"].update({
        f"provider-{index:02d}": {"api_base": f"https://p{index:02d}.test"}
        for index in range(24)
    })
    screen.snapshot = CatalogSnapshot(
        ModelCatalog.model_validate(catalog_data), "many-providers"
    )
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_catalog()
        filters = screen.query_one("#wb-catalog-filter", OptionList)
        filters.focus()
        help_text = str(screen.query_one("#wb-help").render()).lower()
        footer = str(screen.query_one("#wb-hint").render())
        assert "filter" in help_text
        assert "esc goes back" in help_text
        assert "clear filter" not in help_text
        assert "tab" in footer.lower() and "shift" in footer.lower()
        assert "back" in footer.lower()
        assert "clear filter" not in footer.lower()
        screen._select_catalog("filter:two")
        filters.focus()
        help_text = str(screen.query_one("#wb-help").render()).lower()
        footer = str(screen.query_one("#wb-hint").render()).lower()
        assert "esc clears this filter first" in help_text
        assert "clear filter" in footer
        await pilot.press("escape")
        assert screen.query_one("#wb-catalog").display
        assert screen._model_filter is None
        help_text = str(screen.query_one("#wb-help").render()).lower()
        footer = str(screen.query_one("#wb-hint").render()).lower()
        assert "esc goes back" in help_text
        assert "clear filter" not in help_text
        assert "back" in footer
        assert "clear filter" not in footer
        await pilot.press("escape")
        assert screen.query_one("#wb-providers").display


@pytest.mark.asyncio
async def test_global_catalog_pending_draft_opens_owning_provider_detail() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        screen._connection_action("continue")
        await pilot.pause()
        models = screen.query_one("#wb-models", SelectionList)
        models.select("new")
        await pilot.pause()
        screen._open_catalog()
        assert any(
            option.id == "model:new"
            for option in screen.query_one("#wb-catalog", OptionList).options
        )
        screen._select_catalog("model:new")
        await pilot.pause()
        assert screen._detail == "new"
        await pilot.press("escape")
        assert screen.query_one("#wb-catalog").display
        assert screen._detail is None


@pytest.mark.asyncio
async def test_forward_actions_visible_and_detail_save_reachable() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        assert screen.query_one("#wb-actions").display
        screen._save_key("test-key")
        screen._connection_action("continue")
        await pilot.pause()
        values = {
            option.id
            for option in screen.query_one("#wb-models-actions", OptionList).options
        }
        assert {"add-another", "continue-presets"} <= values
        screen._select_action("discover")
        await pilot.pause()
        screen._select_action("models")
        models = screen.query_one("#wb-models", SelectionList)
        models.select("new")
        models.highlighted = next(
            i
            for i, option in enumerate(models.options)
            if cast(Selection[str], option).value == "new"
        )
        await pilot.press("enter")
        await pilot.pause()
        fields = screen.query_one("#wb-detail-fields", OptionList)
        assert fields.display
        assert "Save and continue to presets persists the draft." in str(
            screen.query_one("#wb-help").render()
        )
        assert not any(option.id == "save-detail" for option in fields.options)
        screen.query_one("#price-input", Input).value = "2"
        await press_option(
            pilot, screen.query_one("#wb-detail-actions", OptionList), "save-detail"
        )
        assert screen._detail is None
        assert screen.query_one("#wb-models").display


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (80, 48)])
async def test_models_forward_action_is_reachable_below_deep_model_list(size) -> None:  # type: ignore[no-untyped-def]
    screen, services = setup()
    services.discovery_result = DiscoveryResult(
        tuple(DiscoveryItem(f"model-{index:03d}") for index in range(120))
    )
    async with Host(screen).run_test(size=size) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        connection_actions = screen.query_one("#wb-connection-actions", OptionList)
        connection_actions.highlighted = next(
            i
            for i, option in enumerate(connection_actions.options)
            if option.id == "continue"
        )
        connection_actions.focus()
        await pilot.press("enter")
        await wait_until(pilot, lambda: not screen._busy)
        models = screen.query_one("#wb-models", SelectionList)
        models.focus()
        selected = models.highlighted
        actions = screen.query_one("#wb-models-actions", OptionList)
        assert actions.display and actions.region.bottom <= size[1]
        await pilot.press("tab")
        assert actions.has_focus and models.highlighted == selected
        await pilot.press("shift+tab")
        assert models.has_focus and models.highlighted == selected
        await pilot.press("tab", "down", "down", "down", "down")
        assert (
            actions.highlighted_option
            and actions.highlighted_option.id == "continue-presets"
        )
        await pilot.press("enter")
        await wait_until(pilot, lambda: screen.view == WorkbenchView.PRESETS)
        assert screen.query_one("#wb-presets").has_focus


@pytest.mark.asyncio
async def test_confirmations_keep_dimmed_opener_visible_and_focus_actions() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("models")
        screen._open_detail("a")
        await pilot.pause()
        screen.query_one("#price-input", Input).value = "2"
        await pilot.press("escape")

        confirm = screen.query_one("#wb-confirm")
        confirm_actions = screen.query_one("#wb-confirm-actions", OptionList)
        assert screen._confirm == "detail-discard"
        assert confirm.display and screen.query_one("#wb-detail-fields").display
        assert confirm_actions.has_focus

        await pilot.press("escape")
        screen._save_detail()
        await pilot.press("escape")
        screen._select_action("discard")
        await pilot.pause()
        actions = screen.query_one("#wb-actions")
        assert screen._confirm == "discard"
        assert confirm.display and actions.display
        assert confirm_actions.has_focus


@pytest.mark.asyncio
async def test_discard_confirmation_context_names_the_available_choices() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("models")
        screen._open_detail("a")
        screen.query_one("#price-input", Input).value = "1"
        await pilot.press("escape")
        assert screen._confirm == "detail-discard"
        context = str(screen.query_one("#wb-filter").render())
        message = str(screen.query_one("#wb-confirm-text").render())
        help_text = str(screen.query_one("#wb-help").render())
        assert context == "Unsaved model edits"
        assert "Unsaved model edits" in message
        assert "Keep editing" in message
        assert "save" in message.lower()
        assert "discard" in message.lower()
        assert "Confirm:" not in context + message + help_text


@pytest.mark.asyncio
@pytest.mark.parametrize("pointer", [False, True])
@pytest.mark.parametrize(
    ("view", "group", "action", "writes"),
    [
        ("catalog", "wb-catalog-actions", "\x00apply", 1),
        ("catalog", "wb-catalog-actions", "\x00discard", 0),
        ("detail", "wb-detail-actions", "save-detail", 0),
        ("presets", "wb-presets-actions", "finish", 1),
        ("presets", "wb-presets-actions", "add-another", 1),
        ("preset-editor", "wb-preset-editor-actions", "apply", 0),
        ("preset-editor", "wb-preset-editor-actions", "cancel", 0),
    ],
)
async def test_split_action_has_unique_target_and_pointer_parity(
    view, group, action, writes, pointer
) -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await open_split_view(pilot, screen, view)
        targets = [
            widget.id
            for widget in screen.query(OptionList)
            if widget.display and all(parent.display for parent in widget.ancestors)
            for option in widget.options
            if option.id == action
        ]
        assert targets == [group]
        actions = screen.query_one(f"#{group}", OptionList)
        assert actions.region.bottom <= screen.size.height
        if pointer:
            index = actions.get_option_index(action)
            await pilot.click(actions, offset=(5, index))
        else:
            await press_option(pilot, actions, action)
        await wait_until(pilot, lambda: not screen._busy)
        assert len(services.writes) == writes
        assert not services.keys
        if view == "detail":
            assert screen.view == WorkbenchView.MODELS
            assert screen.state and screen.state.edits[
                "a"
            ].input_price == OptionalEdit.set(2)
            assert screen.query_one("#wb-models").has_focus
        elif view == "preset-editor":
            assert screen.view == WorkbenchView.PRESETS
            assert screen.state and screen.state.preset("orchestrator") == (
                "a",
                "off" if action == "apply" else "medium",
            )
            opener = screen.query_one("#wb-presets", OptionList).highlighted_option
            assert opener and opener.id == "preset:orchestrator"
        elif action == "\x00apply":
            assert screen.view == WorkbenchView.CATALOG
            assert not actions.display
            assert screen.query_one("#wb-catalog").has_focus
        elif action == "\x00discard":
            assert screen._confirm == "discard"
        elif action == "add-another":
            assert screen.view == WorkbenchView.CHOOSE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("view", "group", "body"),
    [
        ("catalog", "wb-catalog-actions", "wb-catalog"),
        ("detail", "wb-detail-actions", "wb-detail-fields"),
        ("presets", "wb-presets-actions", "wb-presets"),
        ("preset-editor", "wb-preset-editor-actions", "wb-preset-editor"),
    ],
)
async def test_split_actions_boundaries_and_reversible_tab(view, group, body) -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await open_split_view(pilot, screen, view)
        rows = screen.query_one(f"#{body}", OptionList)
        actions = screen.query_one(f"#{group}", OptionList)
        await pilot.press("end")
        assert rows.highlighted_option
        selected = rows.highlighted_option.id
        await pilot.press("tab")
        assert actions.has_focus
        await pilot.press("home", "up", "k")
        assert actions.highlighted == 0
        await pilot.press("end", "down", "j")
        assert actions.highlighted == actions.option_count - 1
        assert actions.highlighted_option
        last = actions.highlighted_option.id
        await pilot.press("space", "shift+tab")
        assert (
            rows.has_focus
            and rows.highlighted_option
            and rows.highlighted_option.id == selected
        )
        await pilot.press("tab")
        assert (
            actions.has_focus
            and actions.highlighted_option
            and actions.highlighted_option.id == last
        )
        groups = screen._focus_groups()
        for direction, key in ((1, "tab"), (-1, "shift+tab")):
            index = groups.index(group)
            for step in range(1, len(groups) + 1):
                await pilot.press(key)
                assert (
                    screen.focused
                    and screen.focused.id
                    == groups[(index + direction * step) % len(groups)]
                )
        screen._confirm = "discard"
        screen._sync_view()
        assert actions.has_class("wb-confirm-underlay")
        screen.action_confirm_no()
        screen._set_busy(True)
        assert actions.disabled
        screen._set_busy(False)
        assert not actions.disabled
        assert not services.writes and not services.keys


@pytest.mark.asyncio
@pytest.mark.parametrize("view", ["catalog", "presets"])
async def test_long_split_bodies_leave_actions_reachable(view) -> None:
    screen, services = setup()
    data = services.catalog.catalog.model_dump()
    if view == "catalog":
        data["models"].update({f"long-{i:03}": data["models"]["a"] for i in range(80)})
    else:
        data["roles"].update({
            f"role-{i:03}": {"model": "a", "thinking": "off"} for i in range(80)
        })
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(data), "long")
    screen.snapshot = services.catalog
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await open_split_view(pilot, screen, view)
        body = screen.query_one(f"#wb-{view}", OptionList)
        assert body.max_scroll_y > 0
        await pilot.press("end", "tab")
        actions = screen.query_one(f"#wb-{view}-actions", OptionList)
        assert actions.has_focus and actions.region.bottom <= 24
        assert actions.size.height >= 2
        await pilot.press("shift+tab")
        assert body.has_focus and body.highlighted == body.option_count - 1
        assert body.scroll_y > 0
        assert not services.writes


@pytest.mark.asyncio
async def test_catalog_split_filter_model_and_action_bookmarks_are_independent() -> (
    None
):
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._open_catalog()
        await pilot.pause()
        assert not screen.query_one("#wb-catalog-actions").display
        assert screen._focus_groups() == ["wb-catalog-filter", "wb-catalog", "wb-help"]
        models = screen.query_one("#wb-catalog", OptionList)
        filters = screen.query_one("#wb-catalog-filter", OptionList)
        await pilot.press("end", "shift+tab")
        assert filters.has_focus
        await press_option(pilot, filters, "filter:one")
        await pilot.press("tab")
        assert models.highlighted_option and models.highlighted_option.id == "model:a"
        assert screen.state
        screen.state.set_edits("a", ModelEdits(input_price=OptionalEdit.set(2)))
        screen._refresh_catalog()
        await pilot.press("tab", "end", "shift+tab", "shift+tab")
        assert filters.has_focus
        await press_option(pilot, filters, "filter:")
        await pilot.press("tab")
        assert models.highlighted_option and models.highlighted_option.id == "model:b"
        await pilot.press("tab")
        actions = screen.query_one("#wb-catalog-actions", OptionList)
        assert (
            actions.highlighted_option
            and actions.highlighted_option.id == "\x00discard"
        )
        await pilot.press("shift+tab", "shift+tab")
        assert filters.highlighted_option and filters.highlighted_option.id == "filter:"
        assert not services.writes
