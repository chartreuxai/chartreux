"""First-run onboarding flow tests for the provider workbench suite (shipped mistral root, seeding readiness, key-only changes, host dismissal, commit/shutdown reconciliation, two-provider onboarding)."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import threading

import pytest
from textual.widgets import Input, OptionList, SelectionList

from chartreux.core.model_catalog.contracts import (
    CatalogChanges,
    CatalogValidationError,
    CatalogWriteResult,
    ProviderWorkbenchResult,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot, load_catalog
from chartreux.core.model_catalog.resolver import ModelResolver
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.ui.providers.management_state import ManagementState
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen, WorkbenchView
from tests.ui.providers._workbench_support import (
    Host,
    Services,
    expand,
    press_option,
    setup,
    wait_until,
)


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
    catalog["models"]["other-model"] = {
        "deployments": [{"provider": "custom", "name": "other-model"}]
    }
    catalog["roles"]["custom-review"] = {"model": "other-model", "thinking": "low"}
    services.catalog = CatalogSnapshot(
        ModelCatalog.model_validate(catalog), "custom", frozenset({"custom"})
    )
    screen.snapshot = services.catalog
    screen.mode = "onboarding"
    screen.initial_view = "presets"
    screen.selected_model = "other-model"

    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert screen.view == WorkbenchView.PRESETS
        assert screen.state is not None
        assert all(
            screen.state.preset(role)[0] == "other-model"
            for role in ("main", "worker", "scout", "heavy")
        )
        assert screen.state.preset("custom-review") == ("other-model", "low")
        screen.state.set_role_preset("heavy", "my-model", "low")
        screen._open_presets()
        assert screen.state.preset("heavy") == ("my-model", "low")
        assert screen.state.preset("custom-review") == ("other-model", "low")
        assert not any(
            str(option.id).startswith("preset:")
            for option in screen.query_one("#wb-presets", OptionList).options
        )
        assert "Finish setup" in str(
            next(
                option.prompt
                for option in screen.query_one(
                    "#wb-presets-actions", OptionList
                ).options
                if option.id == "finish"
            )
        )
        await press_option(
            pilot, screen.query_one("#wb-presets-actions", OptionList), "finish"
        )
        await wait_until(pilot, lambda: bool(services.writes))
        assert services.writes[-1].roles is not None
        assert all(
            services.writes[-1].roles[role]["model"] == "other-model"
            for role in ("main", "worker", "scout")
        )
        assert services.writes[-1].roles["heavy"]["model"] == "my-model"


@pytest.mark.asyncio
async def test_onboarding_no_ready_model_keeps_presets_unavailable() -> None:
    screen, services = setup()
    catalog = SHIPPED_CATALOG.model_dump(mode="python")
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(catalog), "shipped")
    screen.snapshot = services.catalog
    screen.mode = "onboarding"
    screen.initial_view = "presets"
    screen.selected_model = "glm-5-3"
    screen.credential_resolver = lambda _env: None
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert screen.state is not None
        assert not screen.state.role_presets
        await press_option(
            pilot, screen.query_one("#wb-presets-actions", OptionList), "finish"
        )
        assert screen.view == WorkbenchView.PRESETS
        assert screen._feedback_kind == "error"
        assert "credential MISTRAL_API_KEY is unavailable" in str(screen._message)
        assert not services.writes


@pytest.mark.asyncio
async def test_onboarding_seeding_readiness_selects_the_thinking_compatible_deployment() -> (
    None
):
    """Seeding must validate the deployment the runtime selects for the role."""
    screen, services = setup()
    catalog = ModelCatalog.model_validate({
        "providers": {
            "one": {"api_base": "https://one.test", "api_key_env_var": "ONE_KEY"},
            "two": {"api_base": "https://two.test", "api_key_env_var": "TWO_KEY"},
        },
        "models": {
            "a": {
                "deployments": [
                    {
                        "provider": "one",
                        "name": "wire-a",
                        "supported_thinking_levels": ["off", "low", "medium"],
                    },
                    {
                        "provider": "two",
                        "name": "wire-a-2",
                        "supported_thinking_levels": [
                            "off",
                            "low",
                            "medium",
                            "high",
                            "max",
                        ],
                    },
                ]
            },
            "b": {"deployments": [{"provider": "one", "name": "wire-b"}]},
        },
        "roles": {
            "orchestrator": {"model": "a", "thinking": "medium"},
            "heavy": {"model": "a", "thinking": "high"},
        },
    })
    services.catalog = CatalogSnapshot(catalog, "tiered", frozenset({"one", "two"}))
    screen.snapshot = services.catalog
    screen.mode = "onboarding"
    screen.initial_view = "presets"
    credentials = lambda env: "key" if env == "ONE_KEY" else None
    screen.credential_resolver = credentials

    # The runtime resolves @heavy past the low/medium deployment to the second
    # deployment, whose credential is unavailable.
    selected = ModelResolver(services.catalog).resolve("@heavy")
    assert (selected.deployment.provider, selected.deployment.name) == (
        "two",
        "wire-a-2",
    )
    assert credentials(selected.provider.api_key_env_var) is None

    # Seeding readiness reports the credential problem, not ready, for both the
    # bare pair check and the role check.
    draft = ManagementState.from_snapshot(services.catalog, "one")
    pair_reason = draft._preset_readiness("a", "high", credentials)
    assert pair_reason is not None and "TWO_KEY" in pair_reason
    role_reason = draft._preset_readiness("a", "high", credentials, role="heavy")
    assert role_reason is not None and "TWO_KEY" in role_reason

    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert screen.state is not None
        assert screen._onboarding_presets_seeded
        # The ready orchestrator preset is preserved; the unusable @heavy binding
        # is replaced with a ready candidate instead of preserved.
        assert screen.state.preset("orchestrator") == ("a", "medium")
        assert screen.state.preset("heavy") == ("b", "high")
        assert screen.state.validate_preset("orchestrator", credentials) is None
        assert screen.state.validate_preset("heavy", credentials) is None


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
        assert any(
            option.id == "\x00presets"
            for option in screen.query_one("#wb-root-actions", OptionList).options
        )
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
        assert any(
            option.id == "\x00presets"
            for option in screen.query_one("#wb-root-actions", OptionList).options
        )
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
        assert "2 runnable models" in str(mistral.prompt)
        assert any(
            option.id == "\x00presets"
            for option in screen.query_one("#wb-root-actions", OptionList).options
        )
        browser.highlighted = next(
            i for i, option in enumerate(browser.options) if option.id == "mistral"
        )
        screen._update_help()
        assert "manages" in str(screen.query_one("#wb-help").render())
        await pilot.press("enter")
        assert screen.state and screen.state.provider_id == "mistral"
        assert screen._provider_open


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
        assert any(
            option.id == "\x00apply"
            for option in screen.query_one("#wb-root-actions", OptionList).options
        )
        screen._request_commit()
        await wait_until(pilot, lambda: not screen._busy)
        screen._refresh_browser()
        assert any(
            option.id == "\x00presets"
            for option in screen.query_one("#wb-root-actions", OptionList).options
        )
        screen._open_presets()
        assert screen.view == WorkbenchView.PRESETS


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
async def test_two_provider_onboarding_uses_arrows_and_forward_saves() -> None:
    screen, services = setup()
    catalog = services.catalog.catalog.model_dump()
    catalog["roles"].update({
        role: {"model": "a", "thinking": "off"} for role in SHIPPED_CATALOG.roles
    })
    services.catalog = CatalogSnapshot(ModelCatalog.model_validate(catalog), "test")
    screen.snapshot = services.catalog
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
                pilot, screen.query_one("#wb-root-actions", OptionList), "\x00add"
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
        await press_option(
            pilot, screen.query_one("#wb-connection-actions", OptionList), "continue"
        )
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
        assert not any(
            str(option.id).startswith("preset:") for option in presets.options
        )
        await press_option(pilot, presets, "customize")
        await press_option(pilot, presets, "preset:orchestrator")
        editor = screen.query_one("#wb-preset-editor", OptionList)
        await press_option(pilot, editor, "model")
        await press_option(
            pilot, screen.query_one("#wb-picker", OptionList), "alpha-model"
        )
        assert screen.focused is editor
        await press_option(pilot, editor, "thinking")
        await press_option(pilot, screen.query_one("#wb-picker", OptionList), "low")
        await press_option(
            pilot, screen.query_one("#wb-preset-editor-actions", OptionList), "apply"
        )
        assert screen.focused is presets
        await press_option(
            pilot, screen.query_one("#wb-presets-actions", OptionList), "finish"
        )
        await wait_until(pilot, lambda: bool(services.writes[-1].roles))
        assert services.writes[-1].roles == {
            "orchestrator": {"model": "alpha-model", "thinking": "low"}
        }
