"""Keyboard journeys for Provider Settings."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, OptionList, SelectionList, Static
from textual.widgets.selection_list import Selection

from chartreux.core.model_catalog.contracts import (
    CatalogChanges,
    CatalogValidationError,
    CatalogWriteResult,
    ConfigPersistResult,
    ConfigReloadResult,
    CredentialSaveResult,
    DiscoveryError,
    DiscoveryItem,
    DiscoveryResult,
    ModelEdits,
    OptionalEdit,
    ProviderDraft,
    ProviderWorkbenchResult,
    TLSConfig,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import (
    CatalogSnapshot,
    CatalogStore,
    load_catalog,
    merge_catalog_overlay,
)
from chartreux.core.model_catalog.presets import FULLY_CUSTOM, MISTRAL, ProviderPreset
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.ui.providers.management_state import ManagementState, PendingModel
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen, WorkbenchView
from tests.snapshots.snapshot_event_loop import install_snapshot_wake


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
        actions = screen.query_one("#wb-actions", OptionList)
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
        detail = screen.query_one("#wb-detail-fields", OptionList)
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


def snapshot() -> CatalogSnapshot:
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "one": {
                    "api_base": "https://one.test",
                    "api_key_env_var": "SHARED_KEY",
                },
                "two": {
                    "api_base": "https://two.test",
                    "api_key_env_var": "SHARED_KEY",
                },
            },
            "models": {
                "a": {
                    "deployments": [
                        {"provider": "one", "name": "wire-a", "prices": {"input": 1.0}}
                    ]
                },
                "b": {"deployments": [{"provider": "two", "name": "wire-b"}]},
            },
            "roles": {
                "orchestrator": {"model": "a", "thinking": "medium"},
                "other": {"model": "b", "thinking": "high"},
            },
        }),
        "test",
    )


@dataclass
class Services:
    catalog: CatalogSnapshot = field(default_factory=snapshot)
    writes: list[CatalogChanges] = field(default_factory=list)
    keys: list[tuple[str, str]] = field(default_factory=list)
    active_models: list[str] = field(default_factory=list)
    discovery_result: DiscoveryResult = field(
        default_factory=lambda: DiscoveryResult((
            DiscoveryItem("wire-a"),
            DiscoveryItem("b"),
            DiscoveryItem("new"),
        ))
    )
    fail: bool = False

    def resolve_key(self, env: str) -> str | None:
        return "configured" if env in {"SHARED_KEY", "NEW_KEY"} else None

    def save_key(self, env_var: str, key: str) -> CredentialSaveResult:
        self.keys.append((env_var, key))
        return CredentialSaveResult("saved")

    async def discover(
        self,
        provider: ProviderDraft,
        credential: str | None,
        tls: TLSConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> DiscoveryResult:
        return self.discovery_result

    async def persist_theme(self, theme: str) -> ConfigPersistResult:
        return ConfigPersistResult(True)

    async def persist_active_model(self, expression: str) -> ConfigPersistResult:
        self.active_models.append(expression)
        return ConfigPersistResult(True)

    def apply_changes(
        self, changes: CatalogChanges
    ) -> CatalogWriteResult | CatalogValidationError:
        if self.fail:
            return CatalogValidationError("Test rejection")
        self.writes.append(changes)
        if changes.provider_patches or changes.models or changes.roles:
            overlay: dict[str, object] = {
                "providers": {
                    provider_id: dict(patch)
                    for provider_id, patch in changes.provider_patches.items()
                },
                "models": dict(changes.models),
            }
            if changes.roles:
                overlay["roles"] = dict(changes.roles)
            self.catalog = CatalogSnapshot(
                merge_catalog_overlay(self.catalog.catalog, overlay), "created"
            )
        return CatalogWriteResult(self.catalog, True)

    async def reload_catalog_and_config(self) -> ConfigReloadResult:
        return ConfigReloadResult(self.catalog)


class Host(App[None]):
    def __init__(self, screen: ProviderWorkbenchScreen) -> None:
        super().__init__()
        self.provider_screen = screen
        self.results: list[ProviderWorkbenchResult] = []

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        install_snapshot_wake()
        self.push_screen(self.provider_screen, self._record)

    def _record(self, result: ProviderWorkbenchResult | None) -> None:
        if result:
            self.results.append(result)


def setup(*, active: str | None = "a") -> tuple[ProviderWorkbenchScreen, Services]:
    services = Services()
    return ProviderWorkbenchScreen(
        snapshot=services.catalog,
        discovery=services.discover,
        catalog_writer=services,
        credentials=services,
        config=services,
        credential_resolver=services.resolve_key,
    ), services


@pytest.mark.asyncio
async def test_onboarding_presets_use_configured_custom_model_without_mistral() -> None:
    screen, services = setup()
    catalog = SHIPPED_CATALOG.model_dump(mode="python")
    catalog["providers"]["custom"] = {
        "api_base": "https://custom.test/v1",
        "api_key_env_var": "SHARED_KEY",
    }
    catalog["models"]["my-model"] = {
        "deployments": [{"provider": "custom", "name": "my-model"}]
    }
    services.catalog = CatalogSnapshot(
        ModelCatalog.model_validate(catalog), "custom", frozenset({"custom"})
    )
    screen.snapshot = services.catalog
    screen.mode = "onboarding"
    screen.initial_view = "presets"

    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert screen.view == WorkbenchView.PRESETS
        assert screen.state is not None
        assert all(
            screen.state.preset(role)[0] == "my-model"
            for role in ("orchestrator", "large", "medium", "small")
        )
        screen.state.set_role_preset("large", "my-model", "low")
        screen._open_presets()
        assert screen.state.preset("large") == ("my-model", "low")
        assert "Save presets and continue" in str(
            next(
                option.prompt
                for option in screen.query_one("#wb-presets", OptionList).options
                if option.id == "finish"
            )
        )
        await press_option(pilot, screen.query_one("#wb-presets", OptionList), "finish")
        await wait_until(pilot, lambda: bool(services.writes))
        assert services.writes[-1].roles is not None
        assert all(
            pair["model"] == "my-model" for pair in services.writes[-1].roles.values()
        )


@pytest.mark.asyncio
async def test_onboarding_no_ready_model_keeps_presets_unavailable() -> None:
    screen, services = setup()
    catalog = SHIPPED_CATALOG.model_dump(mode="python")
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(catalog), "shipped")
    screen.snapshot = services.catalog
    screen.mode = "onboarding"
    screen.initial_view = "presets"
    screen.credential_resolver = lambda _env: None
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert screen.state is not None
        assert not screen.state.role_presets
        await press_option(pilot, screen.query_one("#wb-presets", OptionList), "finish")
        assert screen.view == WorkbenchView.PRESETS
        assert screen._feedback_kind == "error"
        assert not services.writes


@pytest.mark.asyncio
async def test_add_provider_retains_existing_catalog_draft() -> None:
    screen, services = setup(active="b")
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://changed.test"
        await pilot.press("enter")
        await pilot.press("escape")
        assert screen.state and screen.state.dirty
        browser = screen.query_one("#wb-providers", OptionList)
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "\x00add"
        )
        browser.focus()
        await pilot.press("enter")
        assert screen._stage == "connection"
        for key, value in (
            ("name", "other-gateway"),
            ("base", "https://other.test/v1"),
        ):
            screen._connection_action(key)
            screen.query_one("#wb-input", Input).value = value
            await pilot.press("enter")
        screen._connection_action("continue")
        assert screen.state and screen.state.dirty
        screen._select_action("create")
        await wait_until(pilot, lambda: not screen._busy)
        assert len(services.writes) == 1
        write = services.writes[0]
        assert write.provider_id == "other-gateway"
        assert write.providers["one"] == {"api_base": "https://changed.test"}
        assert write.provider["api_base"] == "https://other.test/v1"


@pytest.mark.asyncio
async def test_onboarding_starts_at_provider_and_shipped_mistral_can_be_configured() -> (
    None
):
    shipped = load_catalog(Path("/nonexistent/chartreux-models.toml"))
    services = Services(catalog=shipped)
    screen = ProviderWorkbenchScreen(
        snapshot=shipped,
        discovery=services.discover,
        catalog_writer=services,
        credentials=services,
        config=services,
        credential_resolver=services.resolve_key,
        mode="onboarding",
        initial_view="providers",
    )
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert screen.query_one("#wb-providers").has_focus
        browser = screen.query_one("#wb-providers", OptionList)
        assert any(option.id == "\x00presets" for option in browser.options)
        assert any("Key Required" in str(option.prompt) for option in browser.options)
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "mistral"
        )
        await pilot.press("enter")
        screen._save_key("test-key")
        screen._connection_action("continue")
        await pilot.pause()
        assert screen.state and screen.state.provider_id == "mistral"
        assert screen._stage == "models"
        assert "mistral" in screen.snapshot.catalog.providers


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_unready_shipped_mistral_root_is_honest(size) -> None:  # type: ignore[no-untyped-def]
    services = Services(
        catalog=load_catalog(Path("/nonexistent/chartreux-models.toml"))
    )
    screen = ProviderWorkbenchScreen(
        snapshot=services.catalog,
        discovery=services.discover,
        catalog_writer=services,
        credentials=services,
        config=services,
        credential_resolver=services.resolve_key,
        mode="onboarding",
    )
    async with Host(screen).run_test(size=size) as pilot:
        browser = screen.query_one("#wb-providers", OptionList)
        mistral = next(option for option in browser.options if option.id == "mistral")
        assert "Key Required" in str(mistral.prompt)
        assert "0 runnable models" in str(mistral.prompt)
        assert any(option.id == "\x00presets" for option in browser.options)
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "mistral"
        )
        assert "sets up" in str(screen.query_one("#wb-help").render())
        await pilot.press("enter")
        assert screen._stage == "connection"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_ready_shipped_mistral_root_manages_and_finishes(size) -> None:  # type: ignore[no-untyped-def]
    services = Services(
        catalog=load_catalog(Path("/nonexistent/chartreux-models.toml"))
    )
    services.resolve_key = lambda env: (
        "configured" if env == "MISTRAL_API_KEY" else None
    )  # type: ignore[method-assign]
    screen = ProviderWorkbenchScreen(
        snapshot=services.catalog,
        discovery=services.discover,
        catalog_writer=services,
        credentials=services,
        config=services,
        credential_resolver=services.resolve_key,
        mode="onboarding",
    )
    async with Host(screen).run_test(size=size) as pilot:
        browser = screen.query_one("#wb-providers", OptionList)
        mistral = next(option for option in browser.options if option.id == "mistral")
        assert "Key Set" in str(mistral.prompt)
        assert "1 runnable model" in str(mistral.prompt)
        assert any(option.id == "\x00presets" for option in browser.options)
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "mistral"
        )
        screen._update_help()
        assert "manages" in str(screen.query_one("#wb-help").render())
        await pilot.press("enter")
        assert screen.state and screen.state.provider_id == "mistral"
        assert screen._provider_open


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
async def test_overlaid_mistral_missing_key_enters_management() -> None:
    services = Services(
        catalog=load_catalog(Path("/nonexistent/chartreux-models.toml"))
    )
    services.catalog = CatalogSnapshot(
        services.catalog.catalog, services.catalog.revision, frozenset({"mistral"})
    )
    screen = ProviderWorkbenchScreen(
        snapshot=services.catalog,
        discovery=services.discover,
        catalog_writer=services,
        credentials=services,
        config=services,
        credential_resolver=services.resolve_key,
    )
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        browser = screen.query_one("#wb-providers", OptionList)
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "mistral"
        )
        await pilot.press("enter")
        assert screen.state and screen.state.provider_id == "mistral"
        assert screen._provider_open


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
async def test_dirty_ready_onboarding_saves_before_preset_finish() -> None:
    services = Services(
        catalog=load_catalog(Path("/nonexistent/chartreux-models.toml"))
    )
    services.resolve_key = lambda env: (
        "configured" if env == "MISTRAL_API_KEY" else None
    )  # type: ignore[method-assign]
    screen = ProviderWorkbenchScreen(
        snapshot=services.catalog,
        discovery=services.discover,
        catalog_writer=services,
        credentials=services,
        config=services,
        credential_resolver=services.resolve_key,
        mode="onboarding",
    )
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        browser = screen.query_one("#wb-providers", OptionList)
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "mistral"
        )
        await pilot.press("enter")
        assert screen.state
        screen.state.connection = replace(
            screen.state.connection, api_base="https://changed-mistral.test/v1"
        )
        screen._view = WorkbenchView.PROVIDERS
        screen.query_one("#wb-providers").display = True
        screen._refresh_browser()
        assert any(option.id == "\x00apply" for option in browser.options)
        screen._request_commit()
        await wait_until(pilot, lambda: not screen._busy)
        screen._refresh_browser()
        assert any(option.id == "\x00presets" for option in browser.options)
        screen._open_presets()
        assert screen.view == WorkbenchView.PRESETS


async def expand(pilot, screen: ProviderWorkbenchScreen) -> None:  # type: ignore[no-untyped-def]
    browser = screen.query_one("#wb-providers", OptionList)
    browser.highlighted = next(
        i for i, option in enumerate(browser.options) if option.id == "one"
    )
    await pilot.press("enter")
    assert screen.state is not None and screen.state.provider_id == "one"


async def wait_until(pilot, predicate) -> None:  # type: ignore[no-untyped-def]
    for _ in range(100):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("condition was not reached within 1 second")


async def press_option(pilot, widget: OptionList, option_id: str) -> None:
    """Select an option using the same keys a user would press."""
    for _ in range(widget.option_count + 1):
        if widget.highlighted_option and widget.highlighted_option.id == option_id:
            await pilot.press("enter")
            return
        await pilot.press("down")
    raise AssertionError(f"option {option_id!r} was not reachable")


async def press_model_option(
    pilot, screen: ProviderWorkbenchScreen, value: str
) -> None:  # type: ignore[no-untyped-def]
    if value.startswith("\x00") and value != "\x00empty":
        actions = screen.query_one("#wb-models-actions", OptionList)
        action = value.removeprefix("\x00")
        actions.highlighted = next(
            index for index, option in enumerate(actions.options) if option.id == action
        )
        actions.focus()
        await pilot.press("enter")
        return
    models = screen.query_one("#wb-models", SelectionList)
    models.highlighted = next(
        index
        for index, option in enumerate(models.options)
        if cast(Selection[str], option).value == value
    )
    models.focus()
    await pilot.press("enter")


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
            pilot, screen.query_one("#wb-providers", OptionList), "\x00models"
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
        assert browser.display and browser.has_focus


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
            pilot, screen.query_one("#wb-providers", OptionList), "\x00discard"
        )
        await pilot.press("down", "enter")

        browser = screen.query_one("#wb-providers", OptionList)
        assert browser.display and browser.has_focus
        await pilot.press("down", "enter")
        assert screen.state and screen.state.provider_id == "two"
        assert screen.query_one("#wb-actions").has_focus


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_existing_collision_confirmation_restores_actions_focus(size) -> None:  # type: ignore[no-untyped-def]
    screen, _services = setup()
    async with Host(screen).run_test(size=size) as pilot:
        await pilot.press("enter")
        assert screen.state
        screen.state.select("b")
        screen._refresh_actions()
        await press_option(
            pilot, screen.query_one("#wb-actions", OptionList), "collision:b"
        )
        assert screen._confirm == "existing:b"
        await pilot.press("down", "enter")

        actions = screen.query_one("#wb-actions", OptionList)
        assert screen.state.pending["b"].decision == "add_existing"
        assert actions.display and actions.has_focus
        previous = actions.highlighted
        await pilot.press("down")
        assert actions.highlighted != previous


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_failed_connection_save_keeps_connection_focus(size) -> None:  # type: ignore[no-untyped-def]
    screen, services = setup()
    async with Host(screen).run_test(size=size) as pilot:
        await press_option(
            pilot, screen.query_one("#wb-providers", OptionList), "\x00add"
        )
        screen.query_one("#wb-connection-name", Input).value = "new-gateway"
        screen.query_one("#wb-connection-base", Input).value = "https://new.test/v1"
        await pilot.pause()
        services.fail = True
        screen._connection_action("continue")
        await wait_until(pilot, lambda: not screen._busy)
        assert screen._stage == "connection"
        assert screen.query_one("#wb-actions", OptionList).has_focus
        assert "Failed to create" in screen._message


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_explicit_edit_connection_keeps_saved_new_provider(size) -> None:  # type: ignore[no-untyped-def]
    screen, _services = setup()
    async with Host(screen).run_test(size=size) as pilot:
        await pilot.press("enter")
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "base")
        screen.query_one("#wb-input", Input).value = "https://changed.test"
        await pilot.press("enter", "escape")
        await press_option(
            pilot, screen.query_one("#wb-providers", OptionList), "\x00add"
        )
        screen.query_one("#wb-connection-name", Input).value = "new-gateway"
        screen.query_one("#wb-connection-base", Input).value = "https://new.test/v1"
        await pilot.pause()
        screen._connection_action("continue")
        await pilot.pause()
        assert screen.state and screen.state.provider_id == "new-gateway"

        await press_model_option(pilot, screen, "\x00edit-connection")
        assert screen._stage == "connection"
        screen._connection_action("continue")
        await pilot.pause()
        assert screen.state and screen.state.provider_id == "new-gateway"
        assert screen.state.connection.api_base == "https://new.test/v1"


@pytest.mark.asyncio
async def test_browser_adopt_key_and_shared_confirmation() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        assert "Key Set" in str(
            screen.query_one("#wb-providers", OptionList).get_option_at_index(1).prompt
        )
        await expand(pilot, screen)
        screen._select_action("key")
        assert screen.query_one("#wb-input", Input).password
        screen.query_one("#wb-input", Input).value = "new-secret"
        await pilot.press("enter")
        assert screen._confirm == "shared-key"
        assert "two" in str(screen.query_one("#wb-confirm-text").render())
        assert "new-secret" not in str(screen.query_one("#wb-confirm-text").render())
        await pilot.press("escape")
        assert not services.keys
        await pilot.press("enter")
        await pilot.press("down", "enter")
        assert services.keys == [("SHARED_KEY", "new-secret")]
        assert screen.query_one("#wb-input", Input).value == ""


@pytest.mark.asyncio
async def test_connection_apply_failure_preserves_draft_and_discard() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://changed.test"
        await pilot.press("enter")
        assert screen.state and screen.state.dirty
        services.fail = True
        screen._select_action("apply")
        await pilot.pause()
        assert screen.state and screen.state.dirty
        assert "Failed to apply" in screen._message
        services.fail = False
        screen._select_action("apply")
        await pilot.pause()
        assert services.writes[0].provider == {"api_base": "https://changed.test"}
        await wait_until(pilot, lambda: not screen._busy)
        screen._select_action("env")
        screen.query_one("#wb-input", Input).value = "NEW_KEY"
        await pilot.press("enter")
        screen._select_action("discard")
        assert screen._confirm == "discard"
        await pilot.press("escape")
        assert screen.state and screen.state.dirty
        screen._select_action("discard")
        await pilot.press("down", "enter")
        assert screen.state and not screen.state.dirty


@pytest.mark.asyncio
async def test_discover_toggle_details_and_collision() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("discover")
        await pilot.pause()
        assert screen.state and len(screen.state.model_rows()) == 3
        screen._select_action("models")
        models = screen.query_one("#wb-models", SelectionList)
        assert "new" not in models.selected
        models.select("new")
        await pilot.pause()
        assert screen.state and screen.state.pending["new"].enabled
        screen._open_detail("new")
        await pilot.pause()
        assert screen.query_one("#wb-detail-fields").size.height >= 8
        screen.query_one("#price-input", Input).value = "0"
        screen._save_detail()
        assert screen.state.pending["new"].edits.input_price.value == 0
        screen._select_action("apply")
        await pilot.pause()
        assert services.writes and "new" in services.writes[0].models


@pytest.mark.asyncio
async def test_stale_discovery_invalidated_on_connection_change() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state
        state = screen.state
        provider_id, generation = state.begin_discovery()
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://other.test"
        await pilot.press("enter")
        assert not state.accept_discovery(
            provider_id, generation, DiscoveryResult((DiscoveryItem("stale"),))
        )
        assert "stale" not in str(state.model_rows())


@pytest.mark.asyncio
async def test_delayed_discovery_is_scoped_to_current_provider() -> None:
    screen, services = setup()
    started = asyncio.Event()
    release = asyncio.Event()
    saved_two = DiscoveryResult((DiscoveryItem("current-two"),))

    async def discover(
        provider: ProviderDraft,
        credential: str | None,
        tls: TLSConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> DiscoveryResult:
        if provider.name == "one":
            started.set()
            await release.wait()
            return DiscoveryResult((DiscoveryItem("late-one"),))
        return saved_two

    screen = ProviderWorkbenchScreen(
        snapshot=services.catalog,
        discovery=discover,
        catalog_writer=services,
        credentials=services,
        config=services,
        credential_resolver=services.resolve_key,
    )
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state
        state = screen.state
        task = asyncio.create_task(screen._discover(state))
        await started.wait()
        state.discoveries["two"] = saved_two
        state.for_provider("two")
        release.set()
        await task
        assert state.provider_id == "two"
        assert state.discovery is saved_two
        assert "late-one" not in str(state.discoveries.get("one"))
        assert all(
            item.wire_name != "late-one"
            for pending in state.pending_by_provider.values()
            for item in pending.values()
        )


@pytest.mark.asyncio
async def test_separate_collision_and_reversible_disable_at_80x24() -> None:
    screen, services = setup(active="b")
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("discover")
        await pilot.pause()
        screen._select_action("models")
        models = screen.query_one("#wb-models", SelectionList)
        models.deselect("a")
        await pilot.pause()
        assert screen.state and screen.state.enabled["a"] is False
        models.select("a")
        await pilot.pause()
        assert screen.state and not screen.state.dirty
        models.select("b")
        await pilot.pause()
        assert screen.state
        screen.state.select("b")
        screen.state.pending["b"] = replace(
            screen.state.pending["b"], canonical_name="separate-b", decision="separate"
        )
        screen._refresh_models()
        assert screen.state.pending["b"].canonical_name == "separate-b"
        screen._select_action("apply")
        await pilot.pause()
        assert services.writes and "separate-b" in services.writes[0].models


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


async def add_preset(
    pilot, screen: ProviderWorkbenchScreen, preset: ProviderPreset
) -> None:  # type: ignore[no-untyped-def]
    browser = screen.query_one("#wb-providers", OptionList)
    if preset is MISTRAL:
        screen._choose_preset(MISTRAL)
    elif preset is FULLY_CUSTOM:
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "\x00add"
        )
        await pilot.press("enter")
    else:
        screen._choose_preset(preset)
    assert screen._stage == "connection"
    if preset is MISTRAL:
        screen.set_focus(screen.query_one("#wb-connection-name", Input))
        await pilot.pause()


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
async def test_add_preset_saves_connection_then_models() -> None:
    screen, services = setup(active="b")
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        screen._connection_action("continue")
        await pilot.pause()
        assert screen._stage == "models"
        assert screen.state and screen.state.connection.backend == "mistral"
        models = screen.query_one("#wb-models", SelectionList)
        assert "new" not in models.selected
        models.select("new")
        await pilot.pause()
        screen._open_detail("new")
        screen._save_detail()
        await press_model_option(pilot, screen, "\x00continue-presets")
        await wait_until(pilot, lambda: not screen._busy)
        assert len(services.writes) == 2
        assert services.writes[0].provider_id == "mistral"
        assert services.writes[0].provider["backend"] == "mistral"
        batch = services.writes[-1]
        assert batch.provider_id == "mistral"
        assert "new" in batch.models
        assert batch.roles is None
        assert screen._stage is None
        await pilot.pause()
        assert screen.query_one("#wb-presets").has_focus
        await press_option(
            pilot, screen.query_one("#wb-presets", OptionList), "add-another"
        )
        assert screen._stage == "choose"


@pytest.mark.asyncio
async def test_reconfiguring_overlaid_mistral_preserves_advanced_fields_and_confirms(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.toml"
    store = CatalogStore(path)
    store.upsert_provider(
        {
            "api_base": "https://custom.test/v1",
            "api_key_env_var": "CUSTOM_KEY",
            "api_style": "openai",
            "backend": "mistral",
            "extra_headers": {"X-Custom": "present"},
            "emits_finish_reason": False,
        },
        "mistral",
    )
    services = Services(catalog=load_catalog(path))
    screen = ProviderWorkbenchScreen(
        snapshot=services.catalog,
        discovery=services.discover,
        catalog_writer=services,
        credentials=services,
        config=services,
        credential_resolver=lambda _env: "configured",
    )
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        assert screen._add and screen._add.api_base == "https://custom.test/v1"
        assert screen._add.api_key_env_var == "CUSTOM_KEY"
        assert screen._add.extra_headers == {"X-Custom": "present"}
        screen._connection_action("base")
        screen.query_one("#wb-input", Input).value = "https://revised.test/v1"
        await pilot.press("enter")
        screen._connection_action("continue")
        await wait_until(pilot, lambda: not screen._busy)
        screen._select_action("create")
        assert screen._confirm == "overwrite-mistral", screen._message
        assert not services.writes
        screen.action_confirm_yes()
        await pilot.pause()
        assert len(services.writes) == 1
        assert services.writes[0].provider["extra_headers"] == {"X-Custom": "present"}
        assert services.writes[0].provider["emits_finish_reason"] is False
        assert services.writes[0].provider["api_base"] == "https://revised.test/v1"


@pytest.mark.asyncio
async def test_custom_provider_create_with_manual_model() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, FULLY_CUSTOM)
        for key, value in (("name", "Own Gateway"), ("base", "https://own.test/v1")):
            screen._connection_action(key)
            screen.query_one("#wb-input", Input).value = value
            await pilot.press("enter")
        screen._connection_action("style")
        picker = screen.query_one("#wb-protocol", OptionList)
        picker.highlighted = next(
            i for i, option in enumerate(picker.options) if option.id == "anthropic"
        )
        await pilot.press("space")
        assert screen._protocol_value == "anthropic"
        picker.highlighted = 0  # Cursor movement does not change the radio value.
        await pilot.press("enter")
        assert screen._add and screen._add.api_style == "anthropic"
        screen._connection_action("continue")
        await pilot.pause()
        screen._select_action("manual")
        screen.query_one("#wb-input", Input).value = "own-chat"
        await pilot.press("enter")
        screen._select_action("create")
        await wait_until(pilot, lambda: not screen._busy)
        assert len(services.writes) == 2
        assert services.writes[0].provider_id == "Own Gateway"
        assert services.writes[0].provider["api_style"] == "anthropic"
        assert "own-chat" in services.writes[-1].models
        assert screen._stage is None
        await pilot.pause()
        assert screen.query_one("#wb-presets").has_focus


@pytest.mark.asyncio
async def test_new_provider_name_preserves_spelling_and_reports_collisions() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, FULLY_CUSTOM)
        screen._connection_action("base")
        screen.query_one("#wb-input", Input).value = "https://gateway.test/v1"
        await pilot.press("enter")
        screen._connection_action("name")
        screen.query_one("#wb-input", Input).value = " one "
        await pilot.press("enter")
        assert screen._stage == "connection"
        assert "already exists" in screen._message
        for name in ("bad/name", "bad@name", "bad\x7fname", " "):
            screen.query_one("#wb-input", Input).value = name
            await pilot.press("enter")
            assert screen._stage == "connection"
            assert "Provider name" in screen._message
        screen.query_one("#wb-input", Input).value = " My Gateway "
        await pilot.press("enter")
        screen._connection_action("continue")
        assert screen.state and screen.state.provider_id == "My Gateway"
        assert screen._add and screen._add.name == "My Gateway"


@pytest.mark.asyncio
async def test_add_probe_failure_manual_retry_shared_key_and_stale() -> None:
    from chartreux.core.model_catalog.contracts import DiscoveryError

    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._connection_action("env")
        screen.query_one("#wb-input", Input).value = "SHARED_KEY"
        await pilot.press("enter")
        screen._connection_action("key")
        screen.query_one("#wb-input", Input).value = "private-credential"
        await pilot.press("enter")
        assert screen._confirm == "shared-key"
        assert "two" in str(screen.query_one("#wb-confirm-text").render())
        assert "private-credential" not in str(
            screen.query_one("#wb-confirm-text").render()
        )
        await pilot.press("escape")
        assert not services.keys
        screen._accept_editor()
        await pilot.press("down", "enter")
        assert services.keys == [("SHARED_KEY", "private-credential")]
        services.discovery_result = DiscoveryError("connection", "Unable to list")  # type: ignore[assignment]
        screen._connection_action("continue")
        await pilot.pause()
        assert screen._stage == "models" and "Discovery Failed" in str(
            screen.query_one("#wb-actions", OptionList).options
        )
        assert {"discover", "edit-connection", "edit-key", "manual"}.issubset({
            option.id for option in screen.query_one("#wb-actions", OptionList).options
        })
        state = screen.state
        assert state
        old = state.discovery_generation
        provider_id = state.provider_id
        services.discovery_result = DiscoveryResult((DiscoveryItem("recovered"),))
        screen._select_action("discover")
        await pilot.pause()
        assert not state.accept_discovery(
            provider_id, old, DiscoveryResult((DiscoveryItem("stale"),))
        )
        screen._select_action("manual")
        screen.query_one("#wb-input", Input).value = "manual-chat"
        await pilot.press("enter")
        screen._select_action("create")
        await pilot.pause()
        assert services.writes and "manual-chat" in services.writes[-1].models


@pytest.mark.asyncio
async def test_add_probe_invalidated_on_stage_change_and_close() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        screen._connection_action("continue")
        await pilot.pause()
        state = screen.state
        assert state is not None
        generation = state.discovery_generation
        provider_id = state.provider_id
        await press_model_option(pilot, screen, "\x00edit-connection")
        assert screen._stage == "connection"
        assert not state.accept_discovery(
            provider_id, generation, DiscoveryResult((DiscoveryItem("stale"),))
        )
        screen._connection_action("continue")
        await pilot.pause()
        generation = state.discovery_generation
        provider_id = state.provider_id
        screen._leave_add()
        assert not state.accept_discovery(
            provider_id, generation, DiscoveryResult((DiscoveryItem("closed"),))
        )


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
async def test_host_dismissal_returns_result() -> None:
    screen, _services = setup()
    host = Host(screen)
    async with host.run_test(size=(80, 24)) as pilot:
        await pilot.press("escape")
        await pilot.pause()
        assert host.results == [ProviderWorkbenchResult("cancelled", changed=False)]
        assert host.screen is not screen


@pytest.mark.asyncio
async def test_onboarding_key_only_change_cancels_but_keeps_saved_changes() -> None:
    screen, services = setup()
    screen.mode = "onboarding"
    host = Host(screen)
    async with host.run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._save_key("replacement-key")
        screen.action_back()
        screen.action_back()
        await pilot.pause()
    assert services.keys == [("SHARED_KEY", "replacement-key")]
    assert host.results == [ProviderWorkbenchResult("cancelled", changed=True)]


@pytest.mark.asyncio
async def test_discovery_filters_non_chat_and_deduplicates_canonical_aliases() -> None:
    screen, services = setup()
    services.discovery_result = DiscoveryResult((
        DiscoveryItem("wire-a"),
        DiscoveryItem("a"),
        DiscoveryItem("vendor-embed-1"),
        DiscoveryItem("chat-new"),
    ))
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("discover")
        await pilot.pause()
        assert screen.state is not None
        assert isinstance(screen.state.discovery, DiscoveryResult)
        assert {item.wire_id for item in screen.state.discovery.models} == {
            "wire-a",
            "chat-new",
        }


@pytest.mark.asyncio
async def test_separate_collision_toggle_preserves_draft() -> None:
    screen, services = setup(active="b")
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("discover")
        await pilot.pause()
        screen._select_action("models")
        models = screen.query_one("#wb-models", SelectionList)
        models.select("b")
        await pilot.pause()
        assert screen.state
        screen.state.select("b")
        item = replace(
            screen.state.pending["b"], canonical_name="separate-b", decision="separate"
        )
        screen.state.pending["b"] = item
        screen._refresh_models()
        screen.state.pending["b"] = replace(
            item, edits=ModelEdits(input_price=OptionalEdit.set(2.0))
        )
        models.deselect("separate-b")
        await pilot.pause()
        assert not screen.state.pending["b"].enabled
        models.select("separate-b")
        await pilot.pause()
        assert screen.state.pending["b"].enabled
        assert screen.state.pending["b"].decision == "separate"
        assert screen.state.pending["b"].edits.input_price.value == 2.0
        screen._select_action("apply")
        await pilot.pause()
        assert "separate-b" in services.writes[0].models


@pytest.mark.asyncio
async def test_existing_edit_apply_survives_real_store_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chartreux.core.model_catalog.loader as loader

    initial = snapshot()
    monkeypatch.setattr(loader, "SHIPPED_CATALOG", initial.catalog)
    path = tmp_path / "models.toml"
    initial = load_catalog(path)
    store = CatalogStore(path)

    class ReloadingServices(Services):
        async def reload_catalog_and_config(self) -> ConfigReloadResult:
            return ConfigReloadResult(load_catalog(path))

    services = ReloadingServices(catalog=initial)
    screen = ProviderWorkbenchScreen(
        snapshot=initial,
        discovery=services.discover,
        catalog_writer=store,
        credentials=services,
        config=services,
        credential_resolver=services.resolve_key,
    )
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://updated.test"
        await pilot.press("enter")
        screen._select_action("models")
        screen._open_detail("a")
        screen.query_one("#price-input", Input).value = "2.5"
        screen._save_detail()
        screen._select_action("apply")
        await pilot.pause()
        await wait_until(pilot, lambda: not screen._busy)
        assert screen.state is not None and not screen.state.dirty
        assert (
            screen.snapshot.catalog.providers["one"].api_base == "https://updated.test"
        )
        assert screen.snapshot.catalog.models["a"].deployments[0].prices.input == 2.5
        assert load_catalog(path).catalog == screen.snapshot.catalog


@pytest.mark.asyncio
async def test_commit_blocks_transitions_and_reconciles_on_shutdown() -> None:
    screen, services = setup()
    entered = threading.Event()
    release = threading.Event()
    original = services.apply_changes

    def delayed(changes: CatalogChanges) -> CatalogWriteResult | CatalogValidationError:
        entered.set()
        assert release.wait(timeout=5)
        return original(changes)

    services.apply_changes = delayed  # type: ignore[method-assign]
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://changed.test"
        await pilot.press("enter")
        screen._select_action("apply")
        assert screen._busy and "Saving…" in screen._message
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            screen._select_action("apply")
            screen._expand("two")
            screen._select_action("discard")
            screen._request_commit()
            await pilot.press("enter", "enter", "escape")
            await pilot.resize_terminal(47, 23)
            assert screen._busy and screen.state and screen.state.provider_id == "one"
            assert screen._message == "Saving…"
            assert not screen._dismissed and not services.writes
            shutdown = asyncio.create_task(screen.on_unmount())
            await asyncio.sleep(0)
            assert not shutdown.done()
        finally:
            release.set()
        await asyncio.wait_for(shutdown, 5)
        assert len(services.writes) == 1
        assert screen.state and not screen.state.dirty and not screen._busy
        assert screen.snapshot is services.catalog


@pytest.mark.asyncio
async def test_cancelled_commit_waiter_still_adopts_finished_write() -> None:
    screen, services = setup()
    entered = threading.Event()
    release = threading.Event()
    original = services.apply_changes

    def delayed(changes: CatalogChanges) -> CatalogWriteResult | CatalogValidationError:
        entered.set()
        assert release.wait(timeout=5)
        return original(changes)

    services.apply_changes = delayed  # type: ignore[method-assign]
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://changed.test"
        await pilot.press("enter")
        screen._select_action("apply")
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert screen._commit_task
            screen._commit_task.cancel()
            await asyncio.sleep(0)
            assert screen._busy
        finally:
            release.set()
        await asyncio.wait_for(screen._commit_task, 5)
        assert len(services.writes) == 1
        assert screen.snapshot is services.catalog
        assert not screen._busy and screen.state and not screen.state.dirty


@pytest.mark.asyncio
async def test_reload_failure_adopts_saved_snapshot_and_retries_without_write() -> None:
    screen, services = setup()
    attempts = 0

    async def reload() -> ConfigReloadResult:
        nonlocal attempts
        attempts += 1
        return (
            ConfigReloadResult(None, "runtime rejected")
            if attempts == 1
            else ConfigReloadResult(services.catalog)
        )

    services.reload_catalog_and_config = reload  # type: ignore[method-assign]
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://changed.test"
        await pilot.press("enter")
        screen._select_action("apply")
        await pilot.pause()
        assert screen._message == "Saved; Reload Failed."
        assert screen._warning == "runtime rejected"
        assert screen.state and not screen.state.dirty
        assert len(services.writes) == 1
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://second.test"
        await pilot.press("enter")
        screen._select_action("apply")
        assert len(services.writes) == 1
        assert "Retry Runtime Reload" in screen._message
        assert "Retry Runtime Reload" in str(
            screen.query_one("#wb-actions", OptionList).options
        )
        screen._select_action("retry-reload")
        await pilot.pause()
        assert attempts == 2 and len(services.writes) == 1


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
async def test_long_connection_summary_rebuilds_on_resize_and_preserves_detail() -> (
    None
):
    screen, _ = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("base")
        url = "https://" + "界" * 42 + ".example/v1"
        screen.query_one("#wb-input", Input).value = url
        await pilot.press("enter")
        actions = screen.query_one("#wb-actions", OptionList)
        base = next(option for option in actions.options if option.id == "base")
        assert "…" in str(base.prompt)
        assert url in str(screen.query_one("#wb-help").render()) or (
            screen.state is not None and screen.state.connection.api_base == url
        )
        await pilot.resize_terminal(48, 24)
        base = next(option for option in actions.options if option.id == "base")
        assert "…" in str(base.prompt)
        assert screen.state and screen.state.connection.api_base == url


@pytest.mark.asyncio
async def test_saved_credential_requires_confirmation_even_without_shared_provider() -> (
    None
):
    screen, services = setup()
    catalog = screen.snapshot.catalog.model_copy(
        update={"providers": {"one": screen.snapshot.catalog.providers["one"]}}
    )
    screen.snapshot = CatalogSnapshot(catalog, "one-only")
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("key")
        editor = screen.query_one("#wb-input", Input)
        editor.value = "replacement"
        await pilot.press("enter")
        assert screen._confirm == "shared-key" and not services.keys
        assert "one" in str(screen.query_one("#wb-confirm-text").render())
        await pilot.press("escape")
        assert editor.has_focus and editor.value == "replacement"
        assert not services.keys


@pytest.mark.asyncio
async def test_customized_connection_apply_requires_confirmation_and_restores_focus() -> (
    None
):
    screen, services = setup()
    screen.snapshot = CatalogSnapshot(
        screen.snapshot.catalog, "customized", frozenset({"one"})
    )
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://revised.test"
        await pilot.press("enter")
        screen._select_action("apply")
        assert screen._confirm == "replace-connection" and not services.writes
        assert "one" in str(screen.query_one("#wb-confirm-text").render())
        await pilot.press("escape")
        assert screen.query_one("#wb-actions").has_focus
        assert screen.state and screen.state.dirty
        screen._select_action("apply")
        screen.action_confirm_yes()
        await pilot.pause()
        assert len(services.writes) == 1


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
        focused = screen.focused
        assert focused is not None
        assert shown[0] in focused.ancestors_with_self

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
async def test_connection_key_enter_cancel_restores_field_and_accept_saves() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, FULLY_CUSTOM)
        env = screen.query_one("#wb-connection-env", Input)
        env.value = "SHARED_KEY"
        env.focus()
        await pilot.press("enter")
        key = screen.query_one("#wb-connection-key", Input)
        key.value = "private-secret"
        key.focus()
        await pilot.press("enter")
        assert screen._confirm is None
        assert screen.query_one("#wb-connection-actions").has_focus
        await pilot.press("enter")
        assert screen._confirm == "shared-key"
        await pilot.press("escape")
        assert screen._confirm is None
        assert key.value == "private-secret"
        assert key.has_focus
        await pilot.press("enter")
        await pilot.press("enter")
        await pilot.press("down", "enter")
        assert services.keys == [("SHARED_KEY", "private-secret")]


@pytest.mark.asyncio
async def test_connection_key_enter_without_secret_reports_error_without_hidden_focus() -> (
    None
):
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, FULLY_CUSTOM)
        env = screen.query_one("#wb-connection-env", Input)
        env.value = "NEW_KEY"
        env.focus()
        await pilot.press("enter")
        key = screen.query_one("#wb-connection-key", Input)
        key.focus()
        await pilot.press("enter")
        assert not services.keys
        actions = screen.query_one("#wb-connection-actions", OptionList)
        assert actions.has_focus
        assert next(
            option for option in actions.options if option.id == "save-key"
        ).disabled


@pytest.mark.asyncio
async def test_legacy_key_confirmation_uses_legacy_editor_after_connection_form() -> (
    None
):
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, FULLY_CUSTOM)
        env = screen.query_one("#wb-connection-env", Input)
        env.value = "SHARED_KEY"
        env.focus()
        await pilot.press("enter")
        connection_key = screen.query_one("#wb-connection-key", Input)
        connection_key.value = "connection-secret"
        connection_key.focus()
        await pilot.press("enter")
        await pilot.press("enter")
        assert screen._confirm == "shared-key"
        await pilot.press("escape")
        await pilot.press("escape")
        assert screen._confirm == "add-discard-simple"
        await pilot.press("down", "enter")
        await pilot.pause()
        browser = screen.query_one("#wb-providers", OptionList)
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "one"
        )
        await expand(pilot, screen)
        screen._select_action("key")
        editor = screen.query_one("#wb-input", Input)
        editor.value = "legacy-secret"
        await pilot.press("enter")
        assert screen._confirm == "shared-key"
        await pilot.press("escape")
        assert editor.has_focus and editor.value == "legacy-secret"
        await pilot.press("enter")
        await pilot.press("down", "enter")
        assert services.keys[-1] == ("SHARED_KEY", "legacy-secret")


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
async def test_new_provider_discard_returns_visible_provider_browser() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, FULLY_CUSTOM)
        screen.query_one("#wb-connection-name", Input).value = "draft-provider"
        screen.query_one("#wb-connection-name", Input).focus()
        await pilot.press("enter")
        await pilot.press("escape")
        assert screen._confirm == "add-discard-simple"
        await pilot.press("down", "enter")
        assert screen.query_one("#wb-providers").display
        assert screen.query_one("#wb-providers").has_focus
        assert screen._stage is None


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
        assert any(option.id == "save-detail" for option in fields.options)
        screen.query_one("#price-input", Input).value = "2"
        await press_option(pilot, fields, "save-detail")
        assert screen._detail is None
        assert screen.query_one("#wb-models").display


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
async def test_keyboard_mistral_save_models_moves_to_next_provider() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        actions = screen.query_one("#wb-connection-actions", OptionList)
        actions.highlighted = next(
            i for i, option in enumerate(actions.options) if option.id == "continue"
        )
        actions.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert screen._stage == "models"

        models = screen.query_one("#wb-models", SelectionList)
        models.highlighted = next(
            i
            for i, option in enumerate(models.options)
            if cast(Selection[str], option).value == "new"
        )
        models.select("new")
        await pilot.press("enter")
        await pilot.pause()
        screen.query_one("#price-input", Input).value = "1"
        detail_actions = screen.query_one("#wb-detail-actions", OptionList)
        detail_actions.focus()
        await pilot.press("enter")
        await press_model_option(pilot, screen, "\x00add-another")
        await pilot.pause()
        assert screen._stage == "choose"
        assert services.writes and "new" in services.writes[-1].models
        assert screen.query_one("#wb-choose", OptionList).has_focus
        await press_option(
            pilot, screen.query_one("#wb-choose", OptionList), "fully-custom"
        )
        assert screen._stage == "connection"


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
async def test_discarding_new_provider_restores_global_draft() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://draft.test"
        await pilot.press("enter")
        assert screen.state and screen.state.dirty
        await pilot.press("escape")

        browser = screen.query_one("#wb-providers", OptionList)
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "\x00add"
        )
        browser.focus()
        await pilot.press("enter")
        assert screen._stage == "connection"
        name = screen.query_one("#wb-connection-name", Input)
        name.value = "draft-provider"
        name.focus()
        await pilot.press("enter")
        await pilot.press("escape")
        assert screen._confirm == "add-discard-simple"
        await pilot.press("down", "enter")
        assert screen.query_one("#wb-providers").display
        assert screen.query_one("#wb-providers").has_focus
        await pilot.press("enter")
        assert screen.state and screen.state.provider_id == "one"
        assert screen.state.connection.api_base == "https://draft.test"


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
        browser = screen.query_one("#wb-providers", OptionList)
        browser_ids = {str(option.id) for option in browser.options}
        assert {"\x00apply", "\x00discard"} <= browser_ids
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "\x00apply"
        )
        browser.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert services.writes
        browser = screen.query_one("#wb-providers", OptionList)
        assert not {"\x00apply", "\x00discard"} & {
            str(option.id) for option in browser.options
        }
        assert not screen.state or not screen.state.dirty

        await expand(pilot, screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://discard.test"
        await pilot.press("enter")
        await pilot.press("escape")
        browser = screen.query_one("#wb-providers", OptionList)
        assert {"\x00apply", "\x00discard"} <= {
            str(option.id) for option in browser.options
        }
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "\x00discard"
        )
        browser.focus()
        await pilot.press("enter")
        await pilot.press("down", "enter")
        browser = screen.query_one("#wb-providers", OptionList)
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
async def test_invalid_connection_continue_stays_in_form_with_feedback() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        base = screen.query_one("#wb-connection-base", Input)
        base.value = "https://invalid-mistral.test"
        base.focus()
        await pilot.press("enter")
        actions = screen.query_one("#wb-connection-actions", OptionList)
        actions.highlighted = next(
            i for i, option in enumerate(actions.options) if option.id == "continue"
        )
        actions.focus()
        await pilot.press("enter")
        assert screen._stage == "connection"
        assert screen.query_one("#wb-actions").display
        assert screen._feedback_kind == "error"
        assert "v1" in str(screen.query_one("#wb-help").render()).lower()


@pytest.mark.asyncio
async def test_explicit_connection_edit_preserves_model_draft() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, MISTRAL)
        screen._save_key("test-key")
        screen._connection_action("continue")
        await pilot.pause()
        models = screen.query_one("#wb-models", SelectionList)
        models.select("new")
        await press_model_option(pilot, screen, "\x00edit-connection")
        assert screen._stage == "connection"
        screen._connection_action("base")
        screen.query_one(
            "#wb-input", Input
        ).value = "https://changed-after-models.test/v1"
        await pilot.press("enter")
        screen._connection_action("continue")
        await wait_until(pilot, lambda: not screen._busy)
        assert screen._stage == "models", (screen._message, screen._confirm)
        assert screen.state is not None
        assert "new" in screen.state.pending
        assert (
            screen.state.connection.api_base == "https://changed-after-models.test/v1"
        )
        assert services.writes[-1].models == {}


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
        await press_option(pilot, detail_fields, "save-detail")
        await pilot.pause()
        assert detail_fields.has_focus
        highlighted = detail_fields.highlighted_option
        assert highlighted is not None and highlighted.id == "input"
        assert "Error:" in str(highlighted.prompt)
        assert price.has_class("-invalid")
        assert "Error:" in str(screen.query_one("#error-input").render())

        price.value = "1"
        detail_fields.focus()
        await press_option(pilot, detail_fields, "save-detail")
        await pilot.pause()
        assert not price.has_class("-invalid")
        assert not str(screen.query_one("#error-input").render()).strip()


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
async def test_two_provider_onboarding_uses_arrows_and_forward_saves() -> None:
    screen, services = setup()
    screen.mode = "onboarding"

    async def model_action(pilot, value: str) -> None:  # type: ignore[no-untyped-def]
        models = screen.query_one("#wb-models", SelectionList)
        models.focus()
        await pilot.press("tab")
        actions = screen.query_one("#wb-models-actions", OptionList)
        await pilot.press("home")
        for _ in range(actions.option_count + 1):
            if (
                actions.highlighted_option
                and actions.highlighted_option.id == value.removeprefix("\x00")
            ):
                await pilot.press("enter")
                return
            await pilot.press("down")
        raise AssertionError(f"Model action {value!r} was not arrow reachable")

    async def connect(pilot, name: str) -> None:  # type: ignore[no-untyped-def]
        if screen.view == WorkbenchView.PROVIDERS:
            await press_option(
                pilot, screen.query_one("#wb-providers", OptionList), "\x00add"
            )
        if screen.view == WorkbenchView.CHOOSE:
            await press_option(
                pilot, screen.query_one("#wb-choose", OptionList), "fully-custom"
            )
        actions = screen.query_one("#wb-actions", OptionList)
        for key, value in (("name", name), ("base", f"https://{name}.test/v1")):
            await press_option(pilot, actions, key)
            assert screen.focused is screen.query_one("#wb-input", Input)
            screen.query_one("#wb-input", Input).value = value
            await pilot.press("enter")
            assert screen.focused is actions
        await press_option(pilot, actions, "continue")
        await wait_until(
            pilot, lambda: screen.view == WorkbenchView.MODELS and not screen._busy
        )
        assert screen.focused is screen.query_one("#wb-models", SelectionList)

    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await connect(pilot, "alpha")
        await model_action(pilot, "\x00manual")
        screen.query_one("#wb-input", Input).value = "alpha-model"
        await pilot.press("enter")
        await model_action(pilot, "\x00add-another")
        await wait_until(
            pilot, lambda: screen.view == WorkbenchView.CHOOSE and not screen._busy
        )
        assert "alpha" in screen.snapshot.catalog.providers

        await connect(pilot, "beta")
        await model_action(pilot, "\x00manual")
        screen.query_one("#wb-input", Input).value = "beta-model"
        await pilot.press("enter")
        await model_action(pilot, "\x00continue-presets")
        await wait_until(
            pilot, lambda: screen.view == WorkbenchView.PRESETS and not screen._busy
        )
        assert screen.focused is screen.query_one("#wb-presets", OptionList)
        assert {"alpha", "beta"} <= screen.snapshot.catalog.providers.keys()

        presets = screen.query_one("#wb-presets", OptionList)
        await press_option(pilot, presets, "preset:orchestrator")
        editor = screen.query_one("#wb-preset-editor", OptionList)
        await press_option(pilot, editor, "model")
        await press_option(
            pilot, screen.query_one("#wb-picker", OptionList), "alpha-model"
        )
        assert screen.focused is editor
        await press_option(pilot, editor, "thinking")
        await press_option(pilot, screen.query_one("#wb-picker", OptionList), "low")
        await press_option(pilot, editor, "apply")
        assert screen.focused is presets
        await press_option(pilot, presets, "finish")
        await wait_until(pilot, lambda: bool(services.writes[-1].roles))
        assert services.writes[-1].roles == {
            "orchestrator": {"model": "alpha-model", "thinking": "low"}
        }


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
        assert len(ids) == 9
        assert all(f"preset:{role}" in ids for role in catalog["roles"])
        assert ids[-2:] == ["finish", "add-another"]
        assert rows.size.height >= 9
        assert rows.virtual_size.height <= rows.size.height
        assert rows.scroll_y == 0
        assert rows.has_focus


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
        await press_option(pilot, rows, "save-detail")
        assert screen._detail is None
        assert screen.query_one("#wb-models").display


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
        await press_option(pilot, editor, "apply")
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
        await press_option(pilot, screen.query_one("#wb-presets", OptionList), "finish")
        assert screen.view == WorkbenchView.PRESETS
        assert screen._feedback_kind == "error"
        assert not services.writes
        assert screen.query_one("#wb-presets").has_focus


@pytest.mark.asyncio
async def test_blank_env_api_key_guides_to_masked_key_and_saves() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await press_option(
            pilot, screen.query_one("#wb-providers", OptionList), "\x00add"
        )
        assert screen._add is not None
        screen._add = replace(screen._add, name="Acme Gateway")
        screen._refresh_add_connection()
        actions = screen.query_one("#wb-actions", OptionList)
        assert "API key — Not set" in str(
            next(option.prompt for option in actions.options if option.id == "key")
        )
        await press_option(pilot, actions, "key")
        editor = screen.query_one("#wb-input", Input)
        assert editor.value == "ACME_GATEWAY_API_KEY" and not editor.password
        await pilot.press("enter")
        assert screen._add.api_key_env_var == "ACME_GATEWAY_API_KEY"
        assert editor.password and editor.has_focus
        editor.value = "secret-for-test"
        await pilot.press("enter")
        assert services.keys == [("ACME_GATEWAY_API_KEY", "secret-for-test")]
        assert editor.value == ""
        assert "API key — Saved" in str(
            next(option.prompt for option in actions.options if option.id == "key")
        )


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
        assert [option.id for option in editor.options] == [
            "model",
            "thinking",
            "apply",
            "cancel",
        ]
        assert "Thinking  low" in str(editor.get_option_at_index(1).prompt)
        await press_option(pilot, editor, "cancel")
        assert screen.state and screen.state.preset("orchestrator") == ("a", "low")
        assert (
            presets.has_focus
            and presets.highlighted_option
            and presets.highlighted_option.id == "preset:orchestrator"
        )
        await pilot.press("enter")
        await press_option(pilot, editor, "model")
        await press_option(pilot, screen.query_one("#wb-picker", OptionList), "b")
        await press_option(pilot, editor, "apply")
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
        await press_option(pilot, editor, "apply")
        assert screen.state and screen.state.preset("orchestrator") == ("a", "off")
        assert (
            editor.has_focus
            and editor.highlighted_option
            and editor.highlighted_option.id == "thinking"
        )
        await pilot.press("enter")
        await press_option(pilot, screen.query_one("#wb-picker", OptionList), "high")
        await press_option(pilot, editor, "apply")
        assert screen.state.preset("orchestrator") == ("b", "high")
