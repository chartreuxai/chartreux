"""Bounded keyboard navigation and transient ownership acceptance cases."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Input, OptionList, SelectionList
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection

from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.ui.providers.workbench import WorkbenchView
from tests.ui.providers.test_workbench import (
    MISTRAL,
    Host,
    add_preset,
    expand,
    press_option,
    setup,
    wait_until,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("view", "group"),
    [
        (WorkbenchView.PROVIDERS, "wb-providers"),
        (WorkbenchView.PROVIDERS, "wb-root-actions"),
        (WorkbenchView.ACTIONS, "wb-actions"),
        (WorkbenchView.ACTIONS, "wb-provider-operations"),
        (WorkbenchView.CONNECTION, "wb-actions"),
        (WorkbenchView.CONNECTION, "wb-connection-actions"),
        (WorkbenchView.MODELS, "wb-models"),
        (WorkbenchView.MODELS, "wb-models-actions"),
        (WorkbenchView.CATALOG, "wb-catalog-filter"),
        (WorkbenchView.CATALOG, "wb-catalog"),
        (WorkbenchView.CATALOG, "wb-catalog-actions"),
        (WorkbenchView.DETAIL, "wb-detail-fields"),
        (WorkbenchView.DETAIL, "wb-detail-actions"),
        (WorkbenchView.PRESETS, "wb-presets"),
        (WorkbenchView.PRESETS, "wb-presets-actions"),
        (WorkbenchView.PRESET_EDITOR, "wb-preset-editor"),
        (WorkbenchView.PRESET_EDITOR, "wb-preset-editor-actions"),
        (WorkbenchView.CHOOSE, "wb-choose"),
        (WorkbenchView.DEPLOYMENTS, "wb-picker"),
        (WorkbenchView.PICKER, "wb-picker"),
        (WorkbenchView.PROTOCOL, "wb-protocol"),
    ],
)
async def test_every_list_bounds_keyboard_and_skips_disabled(view, group) -> None:
    screen, services = setup()
    discovery = AsyncMock(wraps=screen.discovery_service)
    screen.discovery_service = discovery
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        # Fixture rows exercise the same local widget bindings in every group;
        # only navigation keys are sent, never command handlers.
        await expand(pilot, screen)
        assert screen.state
        payload = deepcopy(screen.state.changes())
        widget = screen.query_one(f"#{group}", OptionList)
        widget.clear_options()
        for index in range(5):
            if isinstance(widget, SelectionList):
                widget.add_option(
                    Selection(str(index), str(index), disabled=index % 2 == 0)
                )
            else:
                widget.add_option(
                    Option(str(index), id=str(index), disabled=index % 2 == 0)
                )
        screen._view = view
        widget.display = True
        widget.focus()
        await pilot.pause()
        for key in ("home", "up", "k", "pageup"):
            await pilot.press(key)
            assert widget.has_focus and widget.highlighted == 1
        await pilot.press("down")
        assert widget.highlighted == 3
        for key in ("end", "down", "j", "pagedown"):
            await pilot.press(key)
            assert widget.has_focus and widget.highlighted == 3
        await pilot.press("up")
        assert widget.highlighted == 1
        widget.disable_option_at_index(3)
        await pilot.press("end", "down", "j", "pagedown", "home", "up", "k", "pageup")
        assert widget.highlighted == 1 and widget.has_focus
        widget.disable_option_at_index(1)
        await pilot.press("home", "end", "up", "down", "pageup", "pagedown")
        assert widget.highlighted is None and widget.has_focus
        assert screen.state.changes() == payload
        assert screen.state.provider_id == "one"
        discovery.assert_not_called()
        assert not services.writes and not services.keys and not services.active_models


@pytest.mark.asyncio
async def test_escape_priority_busy_confirmation_help_field_editor_and_caret() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "base")
        editor = screen.query_one("#wb-input", Input)
        await pilot.press("home", "right", "right")
        caret = editor.cursor_position
        text = editor.value
        await pilot.press("f1", "tab")
        screen._confirm = "shared-key"
        screen._update_help()
        screen._busy = True
        await pilot.press("escape", "tab", "shift+tab")
        assert screen._confirm == "shared-key" and screen._help_open
        assert screen._editing == "base"
        screen._busy = False
        await pilot.press("escape")
        assert screen._confirm is None and screen._help_open
        assert screen._editing == "base"
        await pilot.press("escape")
        assert not screen._help_open and screen._editing == "base"
        assert (
            editor.has_focus
            and editor.cursor_position == caret
            and editor.value == text
        )
        await pilot.press("escape")
        assert screen._editing is None
        fields = screen.query_one("#wb-actions", OptionList)
        assert fields.highlighted_option
        assert fields.has_focus and fields.highlighted_option.id == "base"
        assert not services.writes and not services.keys


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", [False, True])
async def test_manual_editor_restores_action_opener(accept) -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        await press_option(
            pilot, screen.query_one("#wb-connection-actions", OptionList), "continue"
        )
        await wait_until(pilot, lambda: not screen._busy)
        actions = screen.query_one("#wb-models-actions", OptionList)
        await press_option(pilot, actions, "manual")
        before = len(services.writes)
        await pilot.press(
            "m", "y", "m", "o", "d", "e", "l", "enter" if accept else "escape"
        )
        assert screen._editing is None and actions.has_focus
        assert actions.highlighted_option
        assert actions.highlighted_option.id == "manual"
        assert len(services.writes) == before


@pytest.mark.asyncio
async def test_root_presets_returns_to_action_then_tab_to_browser_without_writes() -> (
    None
):
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        actions = screen.query_one("#wb-root-actions", OptionList)
        await press_option(pilot, actions, "\x00presets")
        await pilot.press("escape")
        assert actions.highlighted_option
        assert actions.has_focus and actions.highlighted_option.id == "\x00presets"
        await pilot.press("shift+tab", "z", "j", "k")
        assert screen.filter_text == "zjk"
        await pilot.press("escape")
        assert not screen.filter_text and screen.view == WorkbenchView.PROVIDERS
        assert not services.writes and not services.keys and not services.active_models


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", [False, True])
async def test_collision_keep_editing_opens_separate_editor_and_restores_operation(
    accept,
) -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state
        screen.state.select("b")
        screen._refresh_actions()
        actions = screen.query_one("#wb-provider-operations", OptionList)
        await press_option(pilot, actions, "collision:b")
        assert screen._confirm == "existing:b"
        await pilot.press("escape")
        assert screen._confirm is None and screen._editing == "canonical:b"
        assert screen.state.pending["b"].decision == "separate"
        await pilot.press("enter" if accept else "escape")
        assert screen._editing is None and actions.has_focus
        assert (
            actions.highlighted_option
            and actions.highlighted_option.id == "collision:b"
        )
        assert not services.writes and not services.keys


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "view",
    [
        "providers",
        "actions",
        "connection",
        "models",
        "catalog",
        "detail",
        "presets",
        "preset-editor",
    ],
)
async def test_composite_tab_remembers_identity_scroll_and_skips_disabled_groups(
    view,
) -> None:
    from tests.ui.providers.test_workbench import FULLY_CUSTOM, open_split_view

    screen, services = setup()
    discovery = AsyncMock(wraps=screen.discovery_service)
    screen.discovery_service = discovery
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        if view == "actions":
            await expand(pilot, screen)
        elif view in {"connection", "models"}:
            await add_preset(pilot, screen, FULLY_CUSTOM)
            if view == "models":
                screen._stage = "models"
                screen._models_list_open = True
                screen._activate_view(WorkbenchView.MODELS)
        elif view != "providers":
            await open_split_view(pilot, screen, view)
        groups = screen._focus_groups()
        positions = {}
        for group in groups:
            widget = screen.query_one(f"#{group}")
            if isinstance(widget, OptionList):
                screen._position_before_rebuild(screen.view, group)
                widget.clear_options()
                for index in range(40):
                    if isinstance(widget, SelectionList):
                        widget.add_option(Selection(str(index), str(index)))
                    else:
                        widget.add_option(Option(str(index), id=str(index)))
        await pilot.pause()
        while screen.focused and screen.focused.id != groups[0]:
            await pilot.press("shift+tab")
        payload = deepcopy(screen.state.changes()) if screen.state else None
        for group in groups:
            assert screen.focused and screen.focused.id == group
            if group != "wb-help":
                await pilot.press("end", "up")
            widget = screen.query_one(f"#{group}")
            positions[group] = (getattr(widget, "highlighted", None), widget.scroll_y)
            await pilot.press("tab")
        for direction in ("tab", "shift+tab"):
            for _ in groups:
                widget = screen.focused
                assert widget and widget.id in positions
                assert (
                    getattr(widget, "highlighted", None),
                    widget.scroll_y,
                ) == positions[widget.id]
                await pilot.press(direction)
        if len(groups) > 1:
            second = screen.query_one(f"#{groups[1]}")
            second.disabled = True
            await pilot.press("tab")
            assert screen.focused and screen.focused.id != groups[1]
            second.disabled = False
        if screen.state:
            assert screen.state.changes() == payload
        discovery.assert_not_called()
        assert not services.writes and not services.keys


@pytest.mark.asyncio
async def test_help_restores_nonzero_scroll_and_dirty_root_has_one_escape_owner() -> (
    None
):
    screen, services = setup()
    host = Host(screen)
    async with host.run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state
        screen.state.select("b")
        await pilot.press("escape")
        browser = screen.query_one("#wb-providers", OptionList)
        browser.clear_options()
        browser.add_options([Option(str(i), id=str(i)) for i in range(60)])
        await pilot.press("end", "up")
        identity, scroll = browser.highlighted, browser.scroll_y
        assert scroll > 0
        await pilot.press("f1", "shift+tab")
        assert screen.query_one("#wb-help").has_focus
        await pilot.press("down", "escape")
        assert (
            browser.has_focus
            and browser.highlighted == identity
            and browser.scroll_y == scroll
        )
        assert screen._confirm is None and not host.results
        await pilot.press("escape")
        assert screen._confirm == "close" and not host.results
        await pilot.press("escape")
        assert screen._confirm is None and not host.results
        assert (
            browser.has_focus
            and browser.highlighted == identity
            and browser.scroll_y == scroll
        )
        assert not services.writes and not services.keys


@pytest.mark.asyncio
async def test_unchanged_stage_back_restores_root_add_opener() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        actions = screen.query_one("#wb-root-actions", OptionList)
        await press_option(pilot, actions, "\x00add")
        await pilot.press("escape")
        assert actions.has_focus
        assert actions.highlighted_option and actions.highlighted_option.id == "\x00add"
        await pilot.press("enter")
        assert screen.view == WorkbenchView.CONNECTION
        await pilot.press("escape")
        assert screen._stage is None and screen._confirm is None
        assert actions.has_focus
        assert actions.highlighted_option and actions.highlighted_option.id == "\x00add"
        assert not services.writes and not services.keys


@pytest.mark.asyncio
async def test_deployment_picker_and_detail_restore_scrolled_catalog_opener() -> None:
    screen, services = setup()
    data = screen.snapshot.catalog.model_dump()
    for index in range(60):
        model = deepcopy(data["models"]["a"])
        model["deployments"] = [
            *model["deployments"],
            {"provider": "two", "name": f"other-{index}"},
        ]
        data["models"][f"z{index:02}"] = model
    screen.snapshot = CatalogSnapshot(ModelCatalog.model_validate(data), "long")
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await press_option(
            pilot, screen.query_one("#wb-root-actions", OptionList), "\x00models"
        )
        rows = screen.query_one("#wb-catalog", OptionList)
        await pilot.press("end", "up")
        assert rows.highlighted_option
        identity, scroll = rows.highlighted_option.id, rows.scroll_y
        assert identity and scroll > 0
        await pilot.press("enter")
        assert screen.view == WorkbenchView.DEPLOYMENTS
        await pilot.press("end", "up", "escape")
        assert (
            rows.has_focus
            and rows.highlighted_option
            and rows.highlighted_option.id == identity
            and rows.scroll_y == scroll
        )
        await pilot.press("enter", "enter")
        assert screen.view == WorkbenchView.DETAIL
        await pilot.press("escape")
        assert (
            rows.has_focus
            and rows.highlighted_option
            and rows.highlighted_option.id == identity
            and rows.scroll_y == scroll
        )
        assert screen._confirm is None
        assert not services.writes and not services.keys


@pytest.mark.asyncio
async def test_protocol_and_preset_pickers_unwind_one_owner_to_exact_field() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        fields = screen.query_one("#wb-actions", OptionList)
        await press_option(pilot, fields, "style")
        assert screen.view == WorkbenchView.PROTOCOL
        await pilot.press("down", "escape")
        assert (
            fields.has_focus
            and fields.highlighted_option
            and fields.highlighted_option.id == "style"
        )
        await press_option(
            pilot, screen.query_one("#wb-provider-operations", OptionList), "presets"
        )
        roles = screen.query_one("#wb-presets", OptionList)
        await press_option(pilot, roles, "preset:orchestrator")
        pairs = screen.query_one("#wb-preset-editor", OptionList)
        await press_option(pilot, pairs, "model")
        assert screen.view == WorkbenchView.PICKER
        await pilot.press("end", "escape")
        assert screen.view == WorkbenchView.PRESET_EDITOR
        assert (
            pairs.has_focus
            and pairs.highlighted_option
            and pairs.highlighted_option.id == "model"
        )
        await pilot.press("escape")
        assert screen.view == WorkbenchView.PRESETS
        assert (
            roles.has_focus
            and roles.highlighted_option
            and roles.highlighted_option.id == "preset:orchestrator"
        )
        assert not services.writes and not services.keys
