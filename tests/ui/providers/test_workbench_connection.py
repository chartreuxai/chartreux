"""Connection form tests for the provider workbench suite (keys and credentials, discovery and probes, add/edit provider flows, apply/save, collisions)."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import cast

import httpx
import pytest
from textual.widgets import Input, OptionList, SelectionList
from textual.widgets.selection_list import Selection

from chartreux.core.model_catalog.contracts import (
    ConfigReloadResult,
    DiscoveryItem,
    DiscoveryResult,
    ModelEdits,
    OptionalEdit,
    ProviderDraft,
    TLSConfig,
)
from chartreux.core.model_catalog.loader import (
    CatalogSnapshot,
    CatalogStore,
    load_catalog,
)
from chartreux.core.model_catalog.presets import FULLY_CUSTOM, MISTRAL
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
from tests.ui.providers._workbench_support import (
    Host,
    Services,
    add_preset,
    expand,
    press_model_option,
    press_option,
    setup,
    snapshot,
    wait_until,
)


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
        await press_option(
            pilot, screen.query_one("#wb-root-actions", OptionList), "\x00add"
        )
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
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_existing_collision_confirmation_restores_actions_focus(size) -> None:  # type: ignore[no-untyped-def]
    screen, _services = setup()
    async with Host(screen).run_test(size=size) as pilot:
        await pilot.press("enter")
        assert screen.state
        screen.state.select("b")
        screen._refresh_actions()
        await press_option(
            pilot,
            screen.query_one("#wb-provider-operations", OptionList),
            "collision:b",
        )
        assert screen._confirm == "existing:b"
        await pilot.press("down", "enter")

        actions = screen.query_one("#wb-provider-operations", OptionList)
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
            pilot, screen.query_one("#wb-root-actions", OptionList), "\x00add"
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
            pilot, screen.query_one("#wb-root-actions", OptionList), "\x00add"
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
            pilot, screen.query_one("#wb-presets-actions", OptionList), "add-another"
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
        assert screen.state
        assert screen._stage == "models" and isinstance(
            screen.state.discovery, DiscoveryError
        )
        assert "Discovery Failed" in str(screen.query_one("#wb-help").render())
        assert {"retry-discovery", "edit-connection", "manual"}.issubset({
            option.id
            for option in screen.query_one("#wb-models-actions", OptionList).options
        })
        assert screen.query_one("#wb-models-actions").display
        assert not screen.query_one("#wb-actions").display
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
            screen.query_one("#wb-provider-operations", OptionList).options
        )
        screen._select_action("retry-reload")
        await pilot.pause()
        assert attempts == 2 and len(services.writes) == 1


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
        await pilot.press("home", "right", "right", "right", "right")
        assert editor.cursor_position == 4
        help_focus_id = screen._help_focus_id
        await pilot.press("enter")
        assert screen._confirm == "shared-key" and not services.keys
        assert "one" in str(screen.query_one("#wb-confirm-text").render())
        await pilot.press("tab", "shift+tab")
        assert screen.query_one("#wb-confirm-actions").has_focus
        assert screen._help_focus_id == help_focus_id
        await pilot.press("escape")
        assert editor.has_focus and editor.value == "replacement"
        assert editor.cursor_position == 4
        assert screen._help_focus_id == help_focus_id
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
async def test_connection_key_enter_cancel_restores_field_and_accept_saves() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, FULLY_CUSTOM)
        screen._connection_action("env")
        editor = screen.query_one("#wb-input", Input)
        editor.value = "SHARED_KEY"
        await pilot.press("enter")
        screen._connection_action("key")
        editor.value = "private-secret"
        await pilot.press("enter")
        assert screen._confirm == "shared-key"
        await pilot.press("escape")
        assert screen._confirm is None
        assert editor.value == "private-secret"
        assert editor.has_focus
        assert not screen.query_one("#wb-connection-form").display
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
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "env")
        screen.query_one("#wb-input", Input).value = "NEW_KEY"
        await pilot.press("enter")
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "key")
        key = screen.query_one("#wb-input", Input)
        key.value = ""
        await pilot.press("enter")
        assert not services.keys
        assert key.has_focus and screen._editing == "key"
        assert "Error" in str(screen.query_one("#wb-field-error").render())


@pytest.mark.asyncio
async def test_legacy_key_confirmation_uses_legacy_editor_after_connection_form() -> (
    None
):
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await add_preset(pilot, screen, FULLY_CUSTOM)
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "env")
        screen.query_one("#wb-input", Input).value = "SHARED_KEY"
        await pilot.press("enter")
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "key")
        screen.query_one("#wb-input", Input).value = "connection-secret"
        await pilot.press("enter")
        assert screen._confirm == "shared-key"
        await pilot.press("escape")
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
        actions = screen.query_one("#wb-root-actions", OptionList)
        assert actions.has_focus
        assert actions.highlighted_option and actions.highlighted_option.id == "\x00add"
        assert screen._stage is None


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
async def test_discarding_new_provider_restores_global_draft() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://draft.test"
        await pilot.press("enter")
        assert screen.state and screen.state.dirty
        await pilot.press("escape")

        await press_option(
            pilot, screen.query_one("#wb-root-actions", OptionList), "\x00add"
        )
        assert screen._stage == "connection"
        name = screen.query_one("#wb-connection-name", Input)
        name.value = "draft-provider"
        name.focus()
        await pilot.press("enter")
        await pilot.press("escape")
        assert screen._confirm == "add-discard-simple"
        await pilot.press("down", "enter")
        assert screen.query_one("#wb-providers").display
        actions = screen.query_one("#wb-root-actions", OptionList)
        assert actions.has_focus
        assert actions.highlighted_option and actions.highlighted_option.id == "\x00add"
        await pilot.press("shift+tab", "enter")
        assert screen.state and screen.state.provider_id == "one"
        assert screen.state.connection.api_base == "https://draft.test"


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
async def test_blank_env_api_key_guides_to_masked_key_and_saves() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await press_option(
            pilot, screen.query_one("#wb-root-actions", OptionList), "\x00add"
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
