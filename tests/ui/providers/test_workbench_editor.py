"""Model detail editor tests for the provider workbench suite (detail editor, role editor, manual model editor, deployment pickers into detail, detail save/escape)."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
from textual.widgets import Input, OptionList, SelectionList
from textual.widgets.selection_list import Selection

from chartreux.core.model_catalog.contracts import ConfigPersistResult
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.presets import FULLY_CUSTOM, MISTRAL
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.ui.providers.workbench import WorkbenchView
from tests.ui.providers._workbench_support import (
    Host,
    add_preset,
    expand,
    press_model_option,
    press_option,
    setup,
)


@pytest.mark.asyncio
async def test_minimum_48x24_detail_is_keyboard_reachable() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(48, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("models")
        screen._open_detail("a")
        await pilot.pause()
        assert screen.query_one("#wb-detail-fields").display
        assert screen.query_one("#wb-help").size.height <= 2
        await pilot.press("escape")
        assert screen._detail is None


@pytest.mark.asyncio
async def test_invalid_scalar_has_adjacent_error_and_narrow_help_is_reachable() -> None:
    screen, _ = setup()
    async with Host(screen).run_test(size=(48, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("key")
        await pilot.press("enter")
        assert screen.query_one("#wb-input", Input).has_class("-invalid")
        assert "Error: Enter a key" in str(screen.query_one("#wb-field-error").render())
        screen.action_back()
        screen._select_action("models")
        screen._update_help()
        assert len(str(screen.query_one("#wb-hint").render())) <= 46
        await pilot.press("f1")
        assert "Shortcuts:" in str(screen.query_one("#wb-help").render())


@pytest.mark.asyncio
async def test_global_catalog_detail_escape_restores_catalog_cursor_and_filter() -> (
    None
):
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_catalog()
        catalog = screen.query_one("#wb-catalog", OptionList)
        assert [str(option.id) for option in catalog.options] == ["model:a", "model:b"]
        catalog.highlighted = 0
        await pilot.press("enter")
        await pilot.pause()
        assert screen._detail == "a"
        await pilot.press("escape")
        assert screen._detail is None
        assert screen.query_one("#wb-catalog").display
        restored = screen.query_one("#wb-catalog", OptionList)
        assert restored.highlighted_option is not None
        assert restored.highlighted_option.id == "model:a"
        assert screen._model_filter is None
        await pilot.press("down")
        assert restored.highlighted_option is not None
        assert restored.highlighted_option.id == "model:b"


@pytest.mark.asyncio
async def test_catalog_detail_save_and_escape_restore_filter_selection_and_focus() -> (
    None
):
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_catalog()
        screen._select_catalog("filter:two")
        catalog = screen.query_one("#wb-catalog", OptionList)
        catalog.highlighted = next(
            i for i, option in enumerate(catalog.options) if option.id == "model:b"
        )
        catalog.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert screen._detail == "b"
        screen.query_one("#price-input", Input).value = "3"
        screen._save_detail()
        await pilot.pause()
        assert screen.query_one("#wb-catalog").display
        assert screen._model_filter == "two"
        assert catalog.has_focus
        assert catalog.highlighted_option is not None
        assert catalog.highlighted_option.id == "model:b"
        await pilot.press("enter")
        await pilot.press("escape")
        assert screen._detail is None
        assert screen.query_one("#wb-catalog").display
        assert screen._model_filter == "two"
        assert catalog.has_focus
        assert catalog.highlighted_option is not None
        assert catalog.highlighted_option.id == "model:b"
        await pilot.press("escape")
        assert screen.query_one("#wb-catalog").display
        assert screen._model_filter is None
        await pilot.press("escape")
        assert screen.query_one("#wb-providers").display
        assert screen.query_one("#wb-providers").has_focus


@pytest.mark.asyncio
async def test_provider_draft_detail_save_escape_restores_model_cursor() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("discover")
        await pilot.pause()
        screen._select_action("models")
        models = screen.query_one("#wb-models", SelectionList)
        new_index = next(
            i
            for i, option in enumerate(models.options)
            if cast(Selection[str], option).value == "new"
        )
        models.select("new")
        models.highlighted = new_index
        await pilot.press("enter")
        await pilot.pause()
        assert screen._detail == "new"
        screen.query_one("#price-input", Input).value = "2"
        screen._save_detail()
        assert screen._detail is None
        restored = screen.query_one("#wb-models", SelectionList)
        assert restored.display
        assert restored.highlighted_option is not None
        assert cast(Selection[str], restored.highlighted_option).value == "new"
        await pilot.press("down")
        assert restored.highlighted == new_index
        await pilot.press("up")
        assert restored.highlighted == new_index - 1


@pytest.mark.asyncio
async def test_new_mistral_provider_can_edit_discovered_model_and_continue_to_presets() -> (
    None
):
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        screen._connection_action("continue")
        await pilot.pause()
        screen._select_action("discover")
        await pilot.pause()
        screen._select_action("models")
        models = screen.query_one("#wb-models", SelectionList)
        models.highlighted = next(
            i
            for i, option in enumerate(models.options)
            if cast(Selection[str], option).value == "new"
        )
        models.select("new")
        await pilot.press("enter")
        await pilot.pause()
        screen.query_one("#price-output", Input).value = "0"
        screen._save_detail()
        await press_model_option(pilot, screen, "\x00continue-presets")
        await pilot.pause()
        assert services.writes
        assert "new" in services.writes[-1].models
        assert screen.view == WorkbenchView.PRESETS


@pytest.mark.asyncio
async def test_discovered_model_detail_preview_does_not_create_pending_draft() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, FULLY_CUSTOM)
        assert screen._add is not None
        screen._add = replace(
            screen._add, name="preview", api_base="https://preview.test/v1"
        )
        screen._connection_action("continue")
        await pilot.pause()
        screen._select_action("discover")
        await pilot.pause()
        screen._select_action("models")
        state = screen.state
        assert state is not None and "new" not in state.pending
        original_dirty = state.dirty
        models = screen.query_one("#wb-models", SelectionList)
        models.highlighted = next(
            i
            for i, option in enumerate(models.options)
            if cast(Selection[str], option).value == "new"
        )
        models.focus()
        await pilot.press("enter")
        assert screen._detail == "new"
        assert "new" not in state.pending
        await pilot.press("escape")
        assert "new" not in state.pending
        assert state.dirty == original_dirty


@pytest.mark.asyncio
async def test_manual_model_editor_escape_restores_models_focus() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        screen._connection_action("continue")
        await pilot.pause()
        screen._select_action("manual")
        assert screen.query_one("#wb-editor").display
        await pilot.press("escape")
        assert screen.query_one("#wb-models").display
        assert screen.query_one("#wb-models").has_focus


@pytest.mark.asyncio
async def test_multi_deployment_picker_detail_escape_returns_catalog() -> None:
    screen, _services = setup()
    catalog_data = screen.snapshot.catalog.model_dump()
    catalog_data["models"]["a"]["deployments"] = [
        *catalog_data["models"]["a"]["deployments"],
        {"provider": "two", "name": "other-a"},
    ]
    screen.snapshot = CatalogSnapshot(
        ModelCatalog.model_validate(catalog_data), "multi-deployment"
    )
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_catalog()
        screen._select_catalog("model:a")
        picker = screen.query_one("#wb-picker", OptionList)
        assert picker.display and picker.option_count == 2
        await pilot.press("enter")
        await pilot.pause()
        assert screen._detail == "a"
        await pilot.press("escape")
        assert screen.query_one("#wb-catalog").display
        assert screen._detail is None
        assert not screen._picker
        screen._select_catalog("filter:two")
        assert screen._model_filter == "two"
        await pilot.press("escape")
        assert screen.query_one("#wb-catalog").display
        assert screen._model_filter is None
        await pilot.press("escape")
        assert screen.query_one("#wb-providers").display


@pytest.mark.asyncio
async def test_multi_deployment_picker_survives_resize_without_persisting_active() -> (
    None
):
    screen, services = setup()
    catalog_data = screen.snapshot.catalog.model_dump()
    catalog_data["models"]["a"]["deployments"] = [
        *catalog_data["models"]["a"]["deployments"],
        {"provider": "two", "name": "other-a"},
    ]
    screen.snapshot = CatalogSnapshot(
        ModelCatalog.model_validate(catalog_data), "multi-deployment"
    )
    persisted: list[str] = []

    async def persist_active_model(expression: str) -> ConfigPersistResult:
        persisted.append(expression)
        return ConfigPersistResult(True)

    services.persist_active_model = persist_active_model  # type: ignore[method-assign]
    async with Host(screen).run_test(size=(120, 36)) as pilot:
        screen._open_catalog()
        catalog = screen.query_one("#wb-catalog", OptionList)
        catalog.highlighted = next(
            i for i, option in enumerate(catalog.options) if option.id == "model:a"
        )
        await pilot.press("enter")
        picker = screen.query_one("#wb-picker", OptionList)
        assert screen._view == WorkbenchView.DEPLOYMENTS
        assert [str(option.id) for option in picker.options] == [
            "deployment:one:a",
            "deployment:two:a",
        ]
        assert picker.highlighted == 0
        assert picker.has_focus
        assert "Choose deployment to edit" in str(
            screen.query_one("#wb-filter").render()
        )
        help_text = str(screen.query_one("#wb-help").render())
        assert "Enter opens the selected deployment's model details." in help_text
        assert "saves the active model" not in help_text

        await pilot.resize_terminal(100, 30)
        picker = screen.query_one("#wb-picker", OptionList)
        assert screen._view == WorkbenchView.DEPLOYMENTS
        assert [str(option.id) for option in picker.options] == [
            "deployment:one:a",
            "deployment:two:a",
        ]
        assert picker.highlighted == 0
        assert picker.has_focus
        assert "Choose deployment to edit" in str(
            screen.query_one("#wb-filter").render()
        )
        assert "Enter opens the selected deployment's model details." in str(
            screen.query_one("#wb-help").render()
        )
        assert not persisted

        await pilot.press("enter")
        assert screen._view == WorkbenchView.DETAIL
        assert screen._detail == "a"
        assert not persisted


@pytest.mark.asyncio
async def test_detail_save_consumes_catalog_frame_and_escape_returns_provider_browser() -> (
    None
):
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_catalog()
        screen._select_catalog("model:a")
        await pilot.pause()
        assert screen._detail == "a"
        screen._save_detail()
        assert screen.query_one("#wb-catalog").display
        assert screen._detail is None
        await pilot.press("escape")
        assert screen.query_one("#wb-providers").display
        assert screen.query_one("#wb-providers").has_focus


@pytest.mark.asyncio
async def test_dirty_detail_escape_offers_keep_save_discard() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("models")
        screen._open_detail("a")
        screen.query_one("#price-input", Input).value = "2"
        await pilot.press("escape")
        assert screen._confirm == "detail-discard"
        confirm_actions = screen.query_one("#wb-confirm-actions", OptionList)
        ids = [str(option.id) for option in confirm_actions.options]
        assert ids == ["cancel", "save-detail", "discard-detail"]
        await pilot.press("down", "down", "enter")
        assert screen._detail is None


@pytest.mark.asyncio
async def test_catalog_detail_apply_and_discard_keep_root_rows_current() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("models")
        screen._open_detail("a")
        screen.query_one("#price-input", Input).value = "2"
        screen._save_detail()
        await pilot.press("escape", "escape")
        browser = screen.query_one("#wb-root-actions", OptionList)
        browser_ids = {str(option.id) for option in browser.options}
        assert {"\x00apply", "\x00discard"} <= browser_ids
        await press_option(pilot, browser, "\x00apply")
        await pilot.pause()
        assert services.writes
        browser = screen.query_one("#wb-root-actions", OptionList)
        assert not {"\x00apply", "\x00discard"} & {
            str(option.id) for option in browser.options
        }
        assert not screen.state or not screen.state.dirty

        await expand(pilot, screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://discard.test"
        await pilot.press("enter")
        await pilot.press("escape")
        browser = screen.query_one("#wb-root-actions", OptionList)
        assert {"\x00apply", "\x00discard"} <= {
            str(option.id) for option in browser.options
        }
        await press_option(pilot, browser, "\x00discard")
        await pilot.press("down", "enter")
        browser = screen.query_one("#wb-root-actions", OptionList)
        assert not {"\x00apply", "\x00discard"} & {
            str(option.id) for option in browser.options
        }
        assert not screen.state or not screen.state.dirty


@pytest.mark.asyncio
async def test_dirty_detail_escape_can_save_and_invalid_continue_stays_in_form() -> (
    None
):
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("models")
        screen._open_detail("a")
        screen.query_one("#price-input", Input).value = "2"
        await pilot.press("escape")
        confirm = screen.query_one("#wb-confirm-actions", OptionList)
        confirm.highlighted = next(
            i for i, option in enumerate(confirm.options) if option.id == "save-detail"
        )
        await pilot.press("enter")
        assert screen._detail is None


@pytest.mark.asyncio
async def test_detail_invalid_price_keeps_focus_and_clears_error_when_fixed() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("models")
        screen._open_detail("a")
        price = screen.query_one("#price-input", Input)
        price.value = "-1"
        detail_fields = screen.query_one("#wb-detail-fields", OptionList)
        await press_option(
            pilot, screen.query_one("#wb-detail-actions", OptionList), "save-detail"
        )
        await pilot.pause()
        assert detail_fields.has_focus
        highlighted = detail_fields.highlighted_option
        assert highlighted is not None and highlighted.id == "input"
        assert "Error:" in str(highlighted.prompt)
        assert price.has_class("-invalid")
        assert "Error:" in str(screen.query_one("#error-input").render())

        price.value = "1"
        detail_fields.focus()
        await press_option(
            pilot, screen.query_one("#wb-detail-actions", OptionList), "save-detail"
        )
        await pilot.pause()
        assert not price.has_class("-invalid")
        assert not str(screen.query_one("#error-input").render()).strip()


@pytest.mark.asyncio
async def test_detail_fields_edit_with_arrows_enter_space_and_escape() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("models")
        screen._open_detail("a")
        await pilot.pause()
        rows = screen.query_one("#wb-detail-fields", OptionList)
        assert rows.display and rows.has_focus and rows.size.height >= 8
        for key, value in (("thinking", "low"), ("temperature", "0.5"), ("input", "2")):
            await press_option(pilot, rows, key)
            editor = screen.query_one("#wb-input", Input)
            assert editor.has_focus
            editor.value = value
            await pilot.press("enter")
            highlighted = rows.highlighted_option
            assert rows.has_focus and highlighted is not None and highlighted.id == key
        await press_option(pilot, rows, "output")
        editor = screen.query_one("#wb-input", Input)
        editor.value = "9"
        await pilot.press("escape")
        highlighted = rows.highlighted_option
        assert rows.has_focus and highlighted is not None and highlighted.id == "output"
        assert screen.query_one("#price-output", Input).value != "9"
        rows.highlighted = next(
            i for i, option in enumerate(rows.options) if option.id == "images"
        )
        await pilot.press("space")
        assert (
            "supports-images"
            in screen.query_one("#wb-image-support", SelectionList).selected
        )
        await press_option(
            pilot, screen.query_one("#wb-detail-actions", OptionList), "save-detail"
        )
        assert screen._detail is None
        assert screen.query_one("#wb-models").display


@pytest.mark.asyncio
async def test_role_editor_picker_restore_exact_bookmarks_without_writes() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_presets()
        roles = screen.query_one("#wb-presets", OptionList)
        await press_option(pilot, roles, "preset:orchestrator")
        fields = screen.query_one("#wb-preset-editor", OptionList)
        await pilot.press("down", "enter")
        assert screen.view == WorkbenchView.PICKER
        await pilot.press("escape")
        assert (
            fields.has_focus
            and fields.highlighted_option
            and fields.highlighted_option.id == "thinking"
        )
        await pilot.press("tab", "end", "enter")
        assert (
            roles.has_focus
            and roles.highlighted_option
            and roles.highlighted_option.id == "preset:orchestrator"
        )
        await press_option(pilot, roles, "preset:other")
        assert fields.highlighted_option and fields.highlighted_option.id == "model"
        await pilot.press("tab")
        actions = screen.query_one("#wb-preset-editor-actions", OptionList)
        assert actions.highlighted_option and actions.highlighted_option.id == "apply"
        await pilot.press("escape")
        await press_option(pilot, roles, "preset:orchestrator")
        assert fields.highlighted_option and fields.highlighted_option.id == "thinking"
        await pilot.press("tab")
        assert actions.highlighted_option and actions.highlighted_option.id == "cancel"
        assert not services.writes and not services.keys
