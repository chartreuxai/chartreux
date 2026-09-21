from __future__ import annotations

import asyncio
from pathlib import Path
from typing import NoReturn, cast

import pytest
from textual.widgets import Button, Input, Select, SelectionList, Static

from chartreux.core.model_catalog.loader import CatalogLoadError
from chartreux.ui.providers.contracts import (
    CatalogChanges,
    CredentialSaveResult,
    DiscoveryError,
    DiscoveryItem,
    DiscoveryResult,
    ModelSelectionDraft,
    ProviderDraft,
    ProviderFlowResult,
)
from tests.ui.providers.test_flow import FlowHost, make_flow, snapshot, wait_for


async def navigate_onboarding_to_review(pilot, flow, *, model_count: int = 1) -> None:  # type: ignore[no-untyped-def]
    """Exercise the 80×24 onboarding screens solely through keyboard input."""
    await pilot.press("enter")
    await wait_for(pilot, lambda: flow.step == "form")
    await pilot.press(*(["tab"] * 7), "enter")
    await wait_for(
        pilot, lambda: flow.step == "credential" and bool(flow.query("#key"))
    )
    await pilot.press("tab", *"mistral-key", "tab", "enter")
    await pilot.pause()
    await pilot.press("tab", "tab", "enter")
    await wait_for(pilot, lambda: flow.step == "models" and bool(flow.query("#models")))
    await pilot.press("tab", "tab", "space")
    if model_count == 2:
        await pilot.press("down", "space")
    await pilot.press("tab", "enter")
    await wait_for(pilot, lambda: flow.step == "review")


@pytest.mark.asyncio
async def test_editing_existing_provider_retains_its_id() -> None:
    flow, _ = make_flow()
    flow.management = True
    flow._overview_provider_id = "example/default"
    flow._select_existing_provider()
    async with FlowHost(flow).run_test() as pilot:
        flow._show("form")
        await pilot.pause()
        flow.query_one("#preset", Select).value = "generic-openai"
        flow.query_one("#name", Input).value = "Renamed Example"
        flow.query_one("#api-base", Input).value = "https://example.test/v1"
        flow.query_one("#env-var", Input).value = ""
        flow._save_form()
        await pilot.pause()
    assert flow.provider is not None
    assert flow.provider.provider_id == "example/default"


@pytest.mark.asyncio
async def test_stale_probe_result_after_back_and_form_edit_does_not_switch_screen() -> (
    None
):
    flow, _ = make_flow()
    completed = asyncio.Event()

    async def delayed_discovery(*_args: object, **_kwargs: object) -> DiscoveryResult:
        await completed.wait()
        return DiscoveryResult((DiscoveryItem("late-wire"),))

    flow.discovery = delayed_discovery
    flow._apply_preset("mistral")
    async with FlowHost(flow).run_test() as pilot:
        flow._show("probe")
        flow._run_probe()
        await pilot.pause()
        flow.action_back()
        await pilot.pause()
        flow.action_back()
        await pilot.pause()
        flow.query_one("#api-base", Input).value = "https://edited.example/v1"
        flow._save_form()
        await pilot.pause()
        completed.set()
        await pilot.pause()
        await pilot.pause()
        assert flow.step == "credential"


@pytest.mark.asyncio
async def test_clear_aliases_and_tags_actions_commit_explicitly_cleared_values() -> (
    None
):
    catalog = snapshot(
        models={
            "configured": {
                "aliases": ["old-alias"],
                "deployments": [{"provider": "example/default", "name": "wire"}],
            },
            "other": {
                "deployments": [{"provider": "example/default", "name": "other"}]
            },
        },
        tags={"preferred": ["configured", "other"]},
    )
    flow, services = make_flow(catalog=catalog)
    flow._select_existing_provider()
    flow._selected_models = {"wire": ModelSelectionDraft("wire", "configured")}
    flow._detail_wire = "wire"
    async with FlowHost(flow).run_test() as pilot:
        flow._show("review")
        await pilot.pause()
        flow._clear_detail_metadata("aliases")
        flow._clear_detail_metadata("tags")
        flow._commit()
        await pilot.pause()
    changes = services.changes[0]
    assert changes.models["configured"]["aliases"] == []  # type: ignore[index]
    assert changes.tags == {"preferred": ("other",)}


@pytest.mark.asyncio
async def test_partial_input_price_edit_preserves_inherited_other_prices() -> None:
    catalog = snapshot(
        models={
            "configured": {
                "deployments": [
                    {
                        "provider": "example/default",
                        "name": "wire",
                        "prices": {"input": 1, "output": 2, "cached_input": 0.5},
                    }
                ]
            }
        }
    )
    flow, services = make_flow(catalog=catalog)
    flow.provider = ProviderDraft(
        None,
        "example/default",
        "Example",
        "https://example.test/v1",
        "openai",
        "",
        None,
    )
    flow._selected_models = {"wire": ModelSelectionDraft("wire", "configured")}
    flow._detail_wire = "wire"
    async with FlowHost(flow).run_test() as pilot:
        flow._show("review")
        await pilot.pause()
        flow.query_one("#input-price", Input).value = "3"
        flow._commit()
        await pilot.pause()
    deployment = services.changes[0].models["configured"]["deployments"][0]  # type: ignore[index]
    assert deployment["prices"] == {"input": 3.0, "output": 2.0, "cached_input": 0.5}


@pytest.mark.asyncio
async def test_unchanged_rediscovery_skips_overlay_write_and_returns_unchanged() -> (
    None
):
    catalog = snapshot(
        models={
            "configured": {
                "deployments": [{"provider": "example/default", "name": "wire"}]
            }
        }
    )
    flow, services = make_flow(catalog=catalog)
    flow._select_existing_provider()
    flow._selected_models = {"wire": ModelSelectionDraft("wire", "configured")}
    flow._detail_wire = "wire"
    host = FlowHost(flow)
    async with host.run_test() as pilot:
        flow._show("review")
        await pilot.pause()
        flow._commit()
        await pilot.pause()
        assert flow.step == "again"
        assert services.changes == []
        flow._show("picker")
        await pilot.pause()
        flow.query_one("#active-model", Select).value = "configured"
        flow._finish()
        await pilot.pause()
        await pilot.pause()
    assert host.results[-1].changed is False


@pytest.mark.asyncio
async def test_edit_connection_details_commits_without_discovery() -> None:
    flow, services = make_flow((
        DiscoveryError("unsupported_listing", "Model listing is unsupported"),
    ))
    flow.management = True
    flow.step = "overview"
    flow._overview_provider_id = "example/default"
    host = FlowHost(flow)
    async with host.run_test() as pilot:
        await pilot.click("#edit")
        await pilot.pause()
        flow.query_one("#api-base", Input).value = "https://updated.example/v1"
        flow._save_form()
        await pilot.pause()
    assert services.discovery_calls == 0
    assert services.changes[0].provider == {"api_base": "https://updated.example/v1"}
    assert host.results == [ProviderFlowResult("completed", changed=True)]


@pytest.mark.asyncio
async def test_replace_credential_completes_without_discovery_or_catalog_write() -> (
    None
):
    flow, services = make_flow((
        DiscoveryError("unsupported_listing", "Model listing is unsupported"),
    ))
    flow.management = True
    flow.step = "overview"
    flow._overview_provider_id = "mistral/default"
    host = FlowHost(flow)
    async with host.run_test() as pilot:
        await pilot.click("#credential")
        await pilot.pause()
        flow.query_one("#key", Input).value = "replacement-secret"
        flow._save_credential()
        await pilot.pause()
    assert services.saved_keys == [("MISTRAL_API_KEY", "replacement-secret")]
    assert services.changes == []
    assert services.discovery_calls == 0
    assert host.results == [ProviderFlowResult("completed", changed=True)]


@pytest.mark.asyncio
async def test_session_only_credential_warning_is_returned_to_the_host() -> None:
    flow, services = make_flow()
    flow.management = True
    flow.step = "overview"
    flow._overview_provider_id = "mistral/default"
    services.save_key = lambda env_var, key: CredentialSaveResult(
        "session_only", "Key is available for this session only."
    )
    host = FlowHost(flow)
    async with host.run_test() as pilot:
        await pilot.click("#credential")
        await pilot.pause()
        flow.query_one("#key", Input).value = "replacement-secret"
        flow._save_credential()
        await pilot.pause()
    assert host.results == [
        ProviderFlowResult(
            "completed", changed=True, warning="Key is available for this session only."
        )
    ]


@pytest.mark.asyncio
async def test_invalid_connection_edit_surfaces_error_without_catalog_write() -> None:
    flow, services = make_flow()
    flow.management = True
    flow.step = "overview"
    flow._overview_provider_id = "example/default"
    async with FlowHost(flow).run_test() as pilot:
        await pilot.click("#edit")
        await pilot.pause()
        flow.query_one("#api-base", Input).value = "not-a-url"
        flow._save_form()
        await pilot.pause()
        assert "API base must be an HTTP(S) URL." in [
            str(item.render()) for item in flow.query(".error")
        ]
    assert services.changes == []
    assert services.discovery_calls == 0


@pytest.mark.asyncio
async def test_discover_models_management_path_still_reaches_review() -> None:
    flow, services = make_flow((DiscoveryResult((DiscoveryItem("mistral-wire"),)),))
    flow.management = True
    flow.step = "overview"
    flow._overview_provider_id = "mistral/default"
    async with FlowHost(flow).run_test() as pilot:
        await pilot.click("#discover")
        await pilot.pause()
        await pilot.pause()
        assert flow.step == "models"
        flow.query_one("#models", SelectionList).select("mistral-wire")
        flow._save_model_selection()
        await pilot.pause()
        assert flow.step == "review"
        flow._commit()
        await pilot.pause()
        assert flow.step == "again"
    assert services.discovery_calls == 1
    assert services.changes


def test_add_after_edit_back_resets_provider_identity_and_model_metadata() -> None:
    flow, _ = make_flow()
    flow.management = True
    flow._overview_provider_id = "example/default"
    flow._select_existing_provider()
    flow._selected_models = {"wire": ModelSelectionDraft("wire", "wire")}
    flow._selected_wires = {"wire"}
    flow._management_action = "edit"
    flow.step = "form"
    flow.action_back()
    event = type("Event", (), {"button": type("Button", (), {"id": "add"})})()
    flow.on_button_pressed(cast(Button.Pressed, event))
    assert flow.provider is None
    assert flow._selected_models == {}
    assert flow._selected_wires == set()


def test_rediscovery_uses_configured_provider_credential() -> None:
    flow, _ = make_flow()
    flow._select_existing_provider()
    assert flow.provider is not None
    assert flow.provider.key is None
    flow._overview_provider_id = "mistral/default"
    flow._select_existing_provider()
    assert flow.provider is not None
    assert flow.provider.key == "configured-key"


def test_final_picker_excludes_aliases_and_tags_without_usable_deployments() -> None:
    catalog = snapshot(
        models={
            "usable": {
                "aliases": ["u"],
                "deployments": [{"provider": "example/default", "name": "u"}],
            },
            "unusable": {
                "aliases": ["no"],
                "deployments": [
                    {"provider": "example/default", "name": "no", "disabled": True}
                ],
            },
        },
        tags={"usable-tag": ["usable"], "empty-tag": ["unusable"]},
    )
    flow, _ = make_flow(catalog=catalog)
    assert [value for _, value in flow._active_model_options()] == [
        "usable",
        "u",
        "@usable-tag",
    ]


@pytest.mark.asyncio
async def test_clear_prices_commits_unknown_prices() -> None:
    catalog = snapshot(
        models={
            "configured": {
                "deployments": [
                    {
                        "provider": "example/default",
                        "name": "wire",
                        "prices": {"input": 1, "output": 2},
                    }
                ]
            }
        }
    )
    flow, services = make_flow(catalog=catalog)
    flow._select_existing_provider()
    flow._selected_models = {"wire": ModelSelectionDraft("wire", "configured")}
    flow._detail_wire = "wire"
    async with FlowHost(flow).run_test() as pilot:
        flow._show("review")
        await pilot.pause()
        flow._clear_prices()
        flow._commit()
        await pilot.pause()
    deployment = services.changes[0].models["configured"]["deployments"][0]  # type: ignore[index]
    assert deployment["prices"] == {}


@pytest.mark.asyncio
async def test_shared_credential_variable_warns_before_replacement() -> None:
    flow, services = make_flow()
    flow._overview_provider_id = "example/default"
    flow._select_existing_provider()
    assert flow.provider is not None
    flow.provider = ProviderDraft(
        flow.provider.preset,
        flow.provider.provider_id,
        flow.provider.name,
        flow.provider.api_base,
        flow.provider.api_style,
        "MISTRAL_API_KEY",
        None,
    )
    async with FlowHost(flow).run_test() as pilot:
        flow._show("credential")
        await pilot.pause()
        flow.query_one("#key", Input).value = "replacement"
        flow._save_credential()
        await pilot.pause()
        assert "also used by" in (flow.error or "")
    assert services.saved_keys == []


@pytest.mark.asyncio
async def test_discovered_openai_model_commits_matcher_thinking_default() -> None:
    flow, services = make_flow()
    flow.provider = ProviderDraft(
        "generic-openai",
        "new/default",
        "New",
        "https://new.example/v1",
        "openai",
        "",
        None,
    )
    flow.discovered = (DiscoveryItem("unknown-wire"),)
    async with FlowHost(flow).run_test() as pilot:
        flow._show("models")
        await pilot.pause()
        flow.query_one("#models", SelectionList).select("unknown-wire")
        flow._save_model_selection()
        await wait_for(pilot, lambda: bool(flow.query("#input-price")))
        flow._commit()
        await pilot.pause()
    assert services.changes[0].models["unknown-wire"]["thinking"] == "off"  # type: ignore[index]


class FailingCatalogWriter:
    def apply_changes(self, changes: CatalogChanges) -> NoReturn:
        _ = changes
        raise CatalogLoadError(Path("models.toml"), "malformed catalog")


@pytest.mark.asyncio
async def test_malformed_catalog_during_save_renders_recoverable_flow_error() -> None:
    flow, _ = make_flow()
    flow._select_existing_provider()
    flow._selected_models = {"new-wire": ModelSelectionDraft("new-wire", "new-wire")}
    flow._detail_wire = "new-wire"
    flow.catalog_writer = FailingCatalogWriter()
    async with FlowHost(flow).run_test() as pilot:
        flow._show("review")
        await pilot.pause()
        flow._commit()
        await wait_for(
            pilot, lambda: flow.step == "review" and bool(flow.query("#continue"))
        )
        assert flow.step == "review"
        assert "Invalid model catalog" in (flow.error or "")
        assert flow.query_one("#continue", Button)


@pytest.mark.asyncio
async def test_review_initial_select_event_does_not_recompose_away_keyboard_input() -> (
    None
):
    flow, _ = make_flow()
    async with FlowHost(flow).run_test(size=(80, 24)) as pilot:
        await navigate_onboarding_to_review(pilot, flow)
        await pilot.press("tab", "tab", "end", "-", "e", "d", "i", "t", "e", "d")
        await pilot.pause()
        await pilot.pause()
        assert flow.query_one("#base-name", Input).value == "wire-edited"


@pytest.mark.asyncio
async def test_review_continue_is_visible_and_keyboard_activatable_at_80_by_24() -> (
    None
):
    flow, services = make_flow()
    async with FlowHost(flow).run_test(size=(80, 24)) as pilot:
        await navigate_onboarding_to_review(pilot, flow)
        button = flow.query_one("#continue", Button)
        assert button.region.y + button.region.height <= 24
        await pilot.press(*(["tab"] * 16), "enter")
        await wait_for(pilot, lambda: flow.step == "again")
    assert services.changes


@pytest.mark.asyncio
async def test_model_filter_preserves_and_allows_keyboard_deselection() -> None:
    items = (DiscoveryItem("alpha"), DiscoveryItem("beta"))
    flow, _ = make_flow((DiscoveryResult(items),))
    async with FlowHost(flow).run_test(size=(80, 24)) as pilot:
        await pilot.press("enter")
        await wait_for(pilot, lambda: flow.step == "form")
        await pilot.press(*(["tab"] * 7), "enter")
        await wait_for(
            pilot, lambda: flow.step == "credential" and bool(flow.query("#key"))
        )
        await pilot.press("tab", *"mistral-key", "tab", "enter")
        await pilot.pause()
        await pilot.press("tab", "tab", "enter")
        await wait_for(
            pilot, lambda: flow.step == "models" and bool(flow.query("#models"))
        )
        await pilot.press("tab", "tab", "space")
        await pilot.press("shift+tab", *"beta")
        await pilot.pause()
        await pilot.press("backspace", "backspace", "backspace", "backspace")
        await pilot.pause()
        assert "alpha" in flow.query_one("#models", SelectionList).selected
        await pilot.press("tab", "home", "space", "tab", "enter")
        await wait_for(pilot, lambda: flow.error is not None)
        assert flow.step == "models"
        assert flow._selected_wires == set()


@pytest.mark.asyncio
async def test_continue_without_model_renders_error_at_80_by_24() -> None:
    flow, _ = make_flow()
    async with FlowHost(flow).run_test(size=(80, 24)) as pilot:
        await pilot.press("enter")
        await wait_for(pilot, lambda: flow.step == "form")
        await pilot.press(*(["tab"] * 7), "enter")
        await wait_for(
            pilot, lambda: flow.step == "credential" and bool(flow.query("#key"))
        )
        await pilot.press("tab", *"mistral-key", "tab", "enter")
        await pilot.pause()
        await pilot.press("tab", "tab", "enter")
        await wait_for(pilot, lambda: flow.step == "models")
        await pilot.press("tab", "tab", "tab", "enter")
        await wait_for(pilot, lambda: bool(flow.query(".error")))
        assert "Select at least one model" in str(
            flow.query_one(".error", Static).render()
        )


@pytest.mark.asyncio
async def test_finish_without_active_model_returns_saved_success() -> None:
    flow, _ = make_flow()
    flow._changed = True
    flow.step = "again"
    host = FlowHost(flow)
    async with host.run_test(size=(80, 24)) as pilot:
        await wait_for(pilot, lambda: bool(flow.query("#finish-unchanged")))
        flow.query_one("#finish-unchanged", Button).focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: bool(host.results))
    assert host.results == [ProviderFlowResult("completed", changed=True)]


@pytest.mark.asyncio
async def test_slash_canonical_model_can_be_chosen_with_keyboard() -> None:
    catalog = snapshot(
        models={
            "vendor/model": {
                "deployments": [{"provider": "example/default", "name": "wire"}]
            }
        }
    )
    flow, services = make_flow(catalog=catalog)
    flow.step = "picker"
    host = FlowHost(flow)
    async with host.run_test(size=(80, 24)) as pilot:
        await wait_for(pilot, lambda: bool(flow.query("#active-model")))
        picker = flow.query_one("#active-model", Select)
        picker.focus()
        await pilot.press("enter", "down", "enter")
        flow.query_one("#finish", Button).focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: bool(host.results))
    assert services.active_models == ["vendor/model"]
    assert host.results == [
        ProviderFlowResult("completed", "vendor/model", changed=False)
    ]


@pytest.mark.asyncio
async def test_keyboard_clearing_tags_from_multiple_models_composes_pending_edits() -> (
    None
):
    catalog = snapshot(
        models={
            "first": {"deployments": [{"provider": "example/default", "name": "one"}]},
            "second": {"deployments": [{"provider": "example/default", "name": "two"}]},
            "third": {
                "deployments": [{"provider": "example/default", "name": "three"}]
            },
        },
        tags={"preferred": ["first", "second", "third"]},
    )
    flow, services = make_flow(
        (DiscoveryResult((DiscoveryItem("one"), DiscoveryItem("two"))),),
        catalog=catalog,
        management=True,
    )
    async with FlowHost(flow).run_test(size=(80, 24)) as pilot:
        await wait_for(pilot, lambda: flow.step == "overview")
        await pilot.press("enter", "down", "enter", "tab", "tab", "tab", "enter")
        await wait_for(
            pilot, lambda: flow.step == "models" and bool(flow.query("#models"))
        )
        await pilot.press("tab", "tab", "space", "down", "space", "tab", "enter")
        await wait_for(pilot, lambda: flow.step == "review")
        await pilot.press(*(["tab"] * 12), "enter")
        await pilot.pause()
        first_wire = flow.query_one("#detail-model", Select).value
        await pilot.press("tab", "enter", "down", "enter")
        await wait_for(
            pilot, lambda: flow.query_one("#detail-model", Select).value != first_wire
        )
        await pilot.press(*(["tab"] * 12), "enter")
        await pilot.press(*(["tab"] * 16), "enter")
        await wait_for(pilot, lambda: flow.step == "again")
    assert services.changes[0].tags == {"preferred": ("third",)}
