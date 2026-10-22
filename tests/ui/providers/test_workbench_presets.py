"""Presets view and preset editor tests for the provider workbench suite (keyless preset creation, compact layout, thinking picker, preset-pair management save, paired preset editor)."""

from __future__ import annotations

import pytest
from textual.widgets import Input, OptionList

from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.presets import ProviderPreset
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.ui.providers.workbench import WorkbenchView
from tests.ui.providers._workbench_support import Host, add_preset, press_option, setup


@pytest.mark.asyncio
async def test_keyless_preset_skips_credential_and_creates() -> None:
    screen, services = setup()
    preset = ProviderPreset(
        "local-keyless", "Local Keyless", "http://localhost:11434/v1", "openai", ""
    )
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, preset)
        assert "key" in {
            option.id for option in screen.query_one("#wb-actions", OptionList).options
        }
        screen._connection_action("continue")
        await pilot.pause()
        screen._select_action("manual")
        screen.query_one("#wb-input", Input).value = "local-model"
        await pilot.press("enter")
        screen._select_action("create")
        await pilot.pause()
        assert services.writes[0].provider["api_key_env_var"] == ""
        assert not services.keys


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (80, 48)])
async def test_compact_presets_and_forward_actions_fit_without_scrolling(size) -> None:
    screen, services = setup()
    catalog = services.catalog.catalog.model_dump()
    catalog["roles"] = {
        role: {"model": "a", "thinking": "off"}
        for role in ("orchestrator", "explore", "plan", "implement", "review", "verify")
    }
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(catalog), "test")
    screen.snapshot = services.catalog
    async with Host(screen).run_test(size=size) as pilot:
        screen._open_presets()
        await pilot.pause()
        rows = screen.query_one("#wb-presets", OptionList)
        ids = [str(option.id) for option in rows.options]
        assert len(ids) == 7
        assert all(f"preset:{role}" in ids for role in catalog["roles"])
        actions = screen.query_one("#wb-presets-actions", OptionList)
        assert [option.id for option in actions.options] == ["finish", "add-another"]
        assert actions.display and actions.size.height >= 2
        assert rows.size.height >= 7
        assert rows.virtual_size.height <= rows.size.height
        assert rows.scroll_y == 0
        assert rows.has_focus


@pytest.mark.asyncio
async def test_preset_thinking_picker_lists_only_supported_levels() -> None:
    screen, services = setup()
    catalog = services.catalog.catalog.model_dump()
    catalog["models"]["a"]["deployments"][0]["supported_thinking_levels"] = [
        "off",
        "low",
    ]
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(catalog), "test")
    screen.snapshot = services.catalog
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_presets()
        presets = screen.query_one("#wb-presets", OptionList)
        await press_option(pilot, presets, "preset:orchestrator")
        editor = screen.query_one("#wb-preset-editor", OptionList)
        await press_option(pilot, editor, "thinking")
        picker = screen.query_one("#wb-picker", OptionList)
        assert picker.has_focus
        assert [str(option.id) for option in picker.options] == [
            "\x00unsupported",
            "off",
            "low",
        ]
        await press_option(pilot, picker, "low")
        assert editor.has_focus
        await press_option(
            pilot, screen.query_one("#wb-preset-editor-actions", OptionList), "apply"
        )
        assert presets.has_focus
        assert screen.state and screen.state.role_presets["orchestrator"] == (
            "a",
            "low",
        )


@pytest.mark.asyncio
async def test_management_save_keeps_invalid_preset_pair_in_editor() -> None:
    screen, services = setup()
    catalog = services.catalog.catalog.model_dump()
    catalog["models"]["a"]["deployments"][0]["supported_thinking_levels"] = [
        "off",
        "low",
    ]
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(catalog), "test")
    screen.snapshot = services.catalog
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_presets()
        assert screen.state
        screen.state.set_role_preset("orchestrator", "a", "high")
        await press_option(
            pilot, screen.query_one("#wb-presets-actions", OptionList), "finish"
        )
        assert screen.view == WorkbenchView.PRESETS
        assert screen._feedback_kind == "error"
        assert not services.writes
        assert screen.query_one("#wb-presets").has_focus


@pytest.mark.asyncio
async def test_paired_preset_editor_cancel_and_apply_preserve_thinking() -> None:
    screen, services = setup()
    catalog = services.catalog.catalog.model_dump()
    catalog["models"]["a"]["deployments"][0]["supported_thinking_levels"] = [
        "off",
        "low",
    ]
    catalog["models"]["b"]["deployments"][0]["supported_thinking_levels"] = [
        "low",
        "high",
    ]
    catalog["roles"]["orchestrator"] = {"model": "a", "thinking": "low"}
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(catalog), "test")
    screen.snapshot = services.catalog
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_presets()
        presets = screen.query_one("#wb-presets", OptionList)
        await press_option(pilot, presets, "preset:orchestrator")
        editor = screen.query_one("#wb-preset-editor", OptionList)
        await press_option(pilot, editor, "model")
        await press_option(pilot, screen.query_one("#wb-picker", OptionList), "b")
        assert [option.id for option in editor.options] == ["model", "thinking"]
        assert [
            option.id
            for option in screen.query_one(
                "#wb-preset-editor-actions", OptionList
            ).options
        ] == ["apply", "cancel"]
        assert "Thinking  low" in str(editor.get_option_at_index(1).prompt)
        await press_option(
            pilot, screen.query_one("#wb-preset-editor-actions", OptionList), "cancel"
        )
        assert screen.state and screen.state.preset("orchestrator") == ("a", "low")
        assert (
            presets.has_focus
            and presets.highlighted_option
            and presets.highlighted_option.id == "preset:orchestrator"
        )
        await pilot.press("enter")
        await press_option(pilot, editor, "model")
        await press_option(pilot, screen.query_one("#wb-picker", OptionList), "b")
        await press_option(
            pilot, screen.query_one("#wb-preset-editor-actions", OptionList), "apply"
        )
        assert screen.state.preset("orchestrator") == ("b", "low")
        await press_option(pilot, presets, "preset:other")
        assert editor.has_focus
        assert editor.highlighted_option and editor.highlighted_option.id == "model"


@pytest.mark.asyncio
async def test_paired_preset_editor_requires_supported_replacement_before_apply() -> (
    None
):
    screen, services = setup()
    catalog = services.catalog.catalog.model_dump()
    catalog["models"]["a"]["deployments"][0]["supported_thinking_levels"] = ["off"]
    catalog["models"]["b"]["deployments"][0]["supported_thinking_levels"] = ["high"]
    catalog["roles"]["orchestrator"] = {"model": "a", "thinking": "off"}
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(catalog), "test")
    screen.snapshot = services.catalog
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_presets()
        await press_option(
            pilot, screen.query_one("#wb-presets", OptionList), "preset:orchestrator"
        )
        editor = screen.query_one("#wb-preset-editor", OptionList)
        await press_option(pilot, editor, "model")
        await press_option(pilot, screen.query_one("#wb-picker", OptionList), "b")
        assert "unsupported" in str(editor.get_option_at_index(1).prompt)
        await press_option(
            pilot, screen.query_one("#wb-preset-editor-actions", OptionList), "apply"
        )
        assert screen.state and screen.state.preset("orchestrator") == ("a", "off")
        assert (
            editor.has_focus
            and editor.highlighted_option
            and editor.highlighted_option.id == "thinking"
        )
        await pilot.press("enter")
        await press_option(pilot, screen.query_one("#wb-picker", OptionList), "high")
        await press_option(
            pilot, screen.query_one("#wb-preset-editor-actions", OptionList), "apply"
        )
        assert screen.state.preset("orchestrator") == ("b", "high")
