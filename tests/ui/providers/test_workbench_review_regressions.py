"""Runtime regressions for the second Providers navigation review."""

from __future__ import annotations

import asyncio

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Input, OptionList

from chartreux.core.model_catalog.contracts import DiscoveryItem, DiscoveryResult
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.presets import MISTRAL
from chartreux.core.model_catalog.schema import ModelCatalog
from tests.ui.providers.test_workbench import Host, add_preset, expand, setup


@pytest.mark.asyncio
async def test_shared_key_cancel_after_minimum_size_restores_exact_caret() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("key")
        await pilot.pause()
        editor = screen.query_one("#wb-input", Input)
        editor.value = "replacement"
        editor.cursor_position = 4
        origin = screen._help_focus_id
        await pilot.press("enter")
        assert screen._confirm == "shared-key"
        await pilot.resize_terminal(47, 24)
        await pilot.resize_terminal(80, 24)
        assert screen._help_focus_id == origin
        await pilot.press("escape")
        assert editor.has_focus and editor.cursor_position == 4
        assert editor.value == "replacement" and not services.keys


@pytest.mark.asyncio
async def test_long_shared_key_confirmation_keyboard_scroll_is_overlay_local() -> None:
    screen, services = setup()
    data = screen.snapshot.catalog.model_dump()
    data["providers"].update({
        f"shared-provider-{i:02}": {
            "api_base": "https://example.test",
            "api_key_env_var": "SHARED_KEY",
        }
        for i in range(30)
    })
    screen.snapshot = CatalogSnapshot(ModelCatalog.model_validate(data), "long")
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("key")
        await pilot.pause()
        screen.query_one("#wb-input", Input).value = "replacement"
        await pilot.press("enter")
        overlay = screen.query_one("#wb-confirm", VerticalScroll)
        actions = screen.query_one("#wb-confirm-actions", OptionList)
        assert actions.region.y >= overlay.region.bottom
        for _ in range(20):
            await pilot.press("pagedown")
        assert overlay.scroll_y > 0
        assert overlay.region.y <= actions.region.y < overlay.region.bottom
        assert actions.has_focus and screen._confirm == "shared-key"
        await pilot.press("tab", "shift+tab", "up", "down")
        assert actions.has_focus and not services.keys and not services.writes
        await pilot.press("escape")
        assert screen.query_one("#wb-input").has_focus


@pytest.mark.asyncio
@pytest.mark.parametrize("cross_view", [False, True])
async def test_bookmarked_help_remains_tabbable_when_expanded_or_reentered(
    cross_view,
) -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("models")
        await pilot.pause()
        await pilot.press("tab", "tab")
        assert screen.query_one("#wb-models").has_focus
        if cross_view:
            await pilot.press("escape", "escape")
            screen._open_catalog()
            await pilot.pause()
            await pilot.press("tab")
            assert screen.query_one("#wb-help").has_focus
            await pilot.press("escape")
            await expand(pilot, screen)
            screen._select_action("models")
            await pilot.pause()
        else:
            await pilot.press("f1")
            assert screen._help_open
        await pilot.press("tab")
        assert screen.query_one("#wb-help").has_focus
        if not cross_view:
            assert screen._help_open
        await pilot.press("shift+tab")
        assert screen.query_one("#wb-models").has_focus and not screen._help_open


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["insertion", "rename-delete"])
async def test_repaired_catalog_identity_is_revealed_after_reentry(change) -> None:
    screen, _services = setup()
    data = screen.snapshot.catalog.model_dump()
    data["models"] = {
        f"m{i:02}": {"deployments": [{"provider": "one", "name": f"wire-{i}"}]}
        for i in range(70)
    }
    data["roles"] = {}
    screen.snapshot = CatalogSnapshot(ModelCatalog.model_validate(data), "many")
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_catalog()
        await pilot.pause()
        catalog = screen.query_one("#wb-catalog", OptionList)
        catalog.highlighted = 0
        await pilot.pause()
        await pilot.press("tab")
        assert screen.query_one("#wb-help").has_focus
        models = data["models"]
        if change == "insertion":
            for i in range(60):
                models[f"early-{i:02}"] = models["m00"]
            expected = "model:m00"
        else:
            # Delete the bookmarked row and rename the next neighbour beyond
            # the viewport. Identity repair chooses the surviving old m02.
            models["z-renamed"] = models.pop("m01")
            models.pop("m00")
            for i in range(60):
                models[f"early-{i:02}"] = models["m02"]
            expected = "model:m02"
        screen.snapshot = CatalogSnapshot(ModelCatalog.model_validate(data), "changed")
        screen._refresh_catalog()
        await pilot.pause()
        await pilot.press("shift+tab")
        assert catalog.has_focus and catalog.highlighted_option
        assert catalog.highlighted_option.id == expected
        assert catalog.highlighted is not None
        assert (
            catalog.scroll_y
            <= catalog.highlighted
            < catalog.scroll_y + catalog.scrollable_content_region.height
        )
        assert catalog.scroll_y > 0


@pytest.mark.asyncio
async def test_pointer_focus_closes_help_without_activating_destination() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        await pilot.press("f1", "shift+tab")
        assert screen.query_one("#wb-help").has_focus and screen._help_open
        await pilot.click("#wb-actions", offset=(4, 0))
        assert screen.query_one("#wb-actions").has_focus
        assert not screen._help_open and screen._editing is None
        assert not services.writes and not services.keys
        await pilot.click("#wb-actions", offset=(4, 0))
        assert screen._editing == "base"


@pytest.mark.asyncio
async def test_superseded_discovery_does_not_clear_newer_same_provider_feedback() -> (
    None
):
    screen, _services = setup()
    releases = [asyncio.Event(), asyncio.Event()]
    started = [asyncio.Event(), asyncio.Event()]
    calls = 0

    async def discover(*args, **kwargs):
        nonlocal calls
        index = calls
        calls += 1
        started[index].set()
        await releases[index].wait()
        return DiscoveryResult((DiscoveryItem(f"new-{index}"),))

    screen.discovery_service = discover
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        state = screen.state
        first = asyncio.create_task(screen._discover(state))
        await started[0].wait()
        second = asyncio.create_task(screen._discover(state))
        await started[1].wait()
        releases[0].set()
        await first
        assert screen._feedback_kind == "running"
        assert "Discovering models for one" in str(screen._feedback())
        releases[1].set()
        await second
        assert screen._feedback_kind == "info"
        assert "Discovered 1 models for one" in str(screen._feedback())
        assert state.discoveries["one"] == DiscoveryResult((DiscoveryItem("new-1"),))


@pytest.mark.asyncio
async def test_connection_has_one_forward_action_and_masked_key_editor() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        actions = screen.query_one("#wb-connection-actions", OptionList)
        assert [row.id for row in actions.options] == ["continue"]
        assert not screen.query_one("#wb-connection-form").display
        assert screen.query_one("#wb-connection-key", Input).value == ""
        screen._connection_action("key")
        await pilot.pause()
        editor = screen.query_one("#wb-input", Input)
        assert editor.has_focus and editor.password
        editor.value = "test-key"
        await pilot.press("enter")
        assert services.keys == [("MISTRAL_API_KEY", "test-key")]
        assert screen._credential_input_id == "wb-input"


@pytest.mark.asyncio
async def test_filter_switch_with_removal_repairs_nearest_surviving_catalog_row() -> (
    None
):
    screen, _services = setup()
    data = screen.snapshot.catalog.model_dump()
    data["models"] = {
        f"m{i}": {"deployments": [{"provider": "one", "name": f"wire-{i}"}]}
        for i in range(6)
    }
    data["roles"] = {}
    screen.snapshot = CatalogSnapshot(ModelCatalog.model_validate(data), "filter")
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._open_catalog()
        await pilot.pause()
        catalog = screen.query_one("#wb-catalog", OptionList)
        catalog.highlighted = 3
        await pilot.pause()
        data["models"].pop("m3")
        data["models"].pop("m4")
        screen.snapshot = CatalogSnapshot(ModelCatalog.model_validate(data), "removed")
        screen._model_filter = "one"
        screen._refresh_catalog()
        await pilot.pause()
        assert (
            catalog.highlighted_option and catalog.highlighted_option.id == "model:m2"
        )
        assert catalog.has_focus
