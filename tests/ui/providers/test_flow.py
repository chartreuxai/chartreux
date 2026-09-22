from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, SelectionList, Static

from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.ui.providers.contracts import (
    CatalogChanges,
    CatalogWriteResult,
    ConfigPersistResult,
    ConfigReloadResult,
    CredentialSaveResult,
    DiscoveryError,
    DiscoveryItem,
    DiscoveryResult,
    ModelSelectionDraft,
    ProviderDraft,
    ProviderFlowResult,
)
from chartreux.ui.providers.flow import ProviderManagementScreen


@dataclass
class FakeServices:
    discovery_results: list[DiscoveryResult | DiscoveryError]
    snapshot: CatalogSnapshot
    changes: list[CatalogChanges] = field(default_factory=list)
    discovery_calls: int = 0
    saved_keys: list[tuple[str, str]] = field(default_factory=list)
    active_models: list[str] = field(default_factory=list)

    async def discover(
        self, *_args: object, **_kwargs: object
    ) -> DiscoveryResult | DiscoveryError:
        self.discovery_calls += 1
        return self.discovery_results.pop(0)

    def apply_changes(self, changes: CatalogChanges) -> CatalogWriteResult:
        self.changes.append(changes)
        return CatalogWriteResult(self.snapshot, changed=True)

    def resolve_key(self, env_var: str) -> str | None:
        return "configured-key" if env_var == "MISTRAL_API_KEY" else None

    def save_key(self, env_var: str, key: str) -> CredentialSaveResult:
        self.saved_keys.append((env_var, key))
        return CredentialSaveResult("saved")

    async def persist_theme(self, theme: str) -> ConfigPersistResult:
        return ConfigPersistResult(True)

    async def reload_catalog_and_config(self) -> ConfigReloadResult:
        return ConfigReloadResult(self.snapshot)

    async def persist_active_model(self, expression: str) -> ConfigPersistResult:
        self.active_models.append(expression)
        return ConfigPersistResult(True)


class FlowHost(App[ProviderFlowResult | None]):
    def __init__(self, screen: ProviderManagementScreen) -> None:
        super().__init__()
        self.provider_screen = screen
        self.results: list[ProviderFlowResult] = []

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        self.push_screen(self.provider_screen, self._record_result)

    def _record_result(self, result: ProviderFlowResult | None) -> None:
        if result is not None:
            self.results.append(result)


def snapshot(
    *,
    models: dict[str, object] | None = None,
    roles: dict[str, object] | None = None,
    providers: dict[str, object] | None = None,
) -> CatalogSnapshot:
    catalog = ModelCatalog.model_validate({
        "providers": providers
        or {
            "example/default": {"api_base": "https://example.test/v1"},
            "mistral/default": {
                "api_base": "https://api.mistral.ai/v1",
                "api_key_env_var": "MISTRAL_API_KEY",
                "backend": "mistral",
            },
        },
        "models": models or {},
        "roles": roles or {},
    })
    return CatalogSnapshot(catalog, "test")


def make_flow(
    results: Sequence[DiscoveryResult | DiscoveryError] = (
        DiscoveryResult((DiscoveryItem("wire"),)),
    ),
    *,
    catalog: CatalogSnapshot | None = None,
    management: bool = False,
    validate_selection: Callable[[str | None], str | None] | None = None,
) -> tuple[ProviderManagementScreen, FakeServices]:
    effective_snapshot = catalog or snapshot(
        models={
            "wire": {"deployments": [{"provider": "example/default", "name": "wire"}]}
        }
    )
    services = FakeServices(list(results), effective_snapshot)
    return ProviderManagementScreen(
        discovery=services.discover,
        catalog_writer=services,
        credentials=services,
        config=services,
        snapshot=effective_snapshot,
        management=management,
        validate_selection=validate_selection,
    ), services


async def wait_for(pilot, predicate, *, attempts: int = 50) -> None:  # type: ignore[no-untyped-def]
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError("Timed out waiting for the expected UI state")


async def fill_generic_form(pilot, flow: ProviderManagementScreen) -> None:  # type: ignore[no-untyped-def]
    flow._show("form")
    await wait_for(pilot, lambda: flow.step == "form" and bool(flow.query("#preset")))
    flow.query_one("#preset", Select).value = "generic-openai"
    flow.query_one("#name", Input).value = "Example"
    flow.query_one("#api-base", Input).value = "https://example.test/v1"
    flow.query_one("#env-var", Input).value = "EXAMPLE_API_KEY"
    flow._save_form()
    await wait_for(
        pilot, lambda: flow.step == "credential" and bool(flow.query("#key"))
    )
    flow.query_one("#key", Input).value = "secret-value"
    flow._save_credential()
    await wait_for(pilot, lambda: flow.step == "models" and bool(flow.query("#models")))


@pytest.mark.asyncio
async def test_complete_flow_end_to_end_returns_typed_dismiss_result() -> None:
    flow, services = make_flow()
    host = FlowHost(flow)
    async with host.run_test() as pilot:
        await fill_generic_form(pilot, flow)
        assert flow.step == "models"
        models = flow.query_one("#models", SelectionList)
        models.select("wire")
        flow._save_model_selection()
        await wait_for(
            pilot, lambda: flow.step == "review" and bool(flow.query("#detail-model"))
        )
        flow._commit()
        await wait_for(pilot, lambda: flow.step == "again")
        flow._show("picker")
        await wait_for(
            pilot, lambda: flow.step == "picker" and bool(flow.query("#active-model"))
        )
        flow._finish("wire")
        await wait_for(
            pilot,
            lambda: (
                host.results == [ProviderFlowResult("completed", "wire", changed=True)]
            ),
        )
    assert services.changes and services.changes[0].provider_id == "example-2/default"
    assert host.results == [ProviderFlowResult("completed", "wire", changed=True)]


@pytest.mark.asyncio
async def test_onboarding_mistral_uses_existing_backend_and_persists_its_key() -> None:
    catalog = snapshot(
        models={
            "mistral-wire": {
                "deployments": [{"provider": "mistral/default", "name": "mistral-wire"}]
            }
        }
    )
    flow, services = make_flow(
        (DiscoveryResult((DiscoveryItem("mistral-wire"),)),), catalog=catalog
    )
    host = FlowHost(flow)
    async with host.run_test() as pilot:
        flow.query_one("#mistral", Button).press()
        await wait_for(
            pilot, lambda: flow.step == "credential" and bool(flow.query("#key"))
        )
        assert flow.provider is not None
        assert flow.provider.provider_id == "mistral/default"
        assert flow.provider.api_key_env_var == "MISTRAL_API_KEY"
        flow.query_one("#key", Input).value = "mistral-secret"
        flow._save_credential()
        await wait_for(
            pilot, lambda: flow.step == "models" and bool(flow.query("#models"))
        )
        flow.query_one("#models", SelectionList).select("mistral-wire")
        flow._save_model_selection()
        await wait_for(
            pilot, lambda: flow.step == "review" and bool(flow.query("#detail-model"))
        )
        flow._commit()
        await wait_for(pilot, lambda: flow.step == "again")
        flow._show("picker")
        await wait_for(pilot, lambda: bool(flow.query("#active-model")))
        flow._finish("mistral-wire")
        await wait_for(pilot, lambda: bool(host.results))
    assert services.saved_keys == [("MISTRAL_API_KEY", "mistral-secret")]
    assert services.changes == []
    assert services.active_models == ["mistral-wire"]
    assert host.results == [
        ProviderFlowResult("completed", "mistral-wire", changed=True)
    ]


@pytest.mark.asyncio
async def test_onboarding_mistral_falls_back_to_add_preset_without_existing_backend() -> (
    None
):
    flow, _services = make_flow(
        catalog=snapshot(
            providers={"example/default": {"api_base": "https://example.test/v1"}}
        )
    )
    async with FlowHost(flow).run_test() as pilot:
        flow.query_one("#mistral", Button).press()
        await wait_for(
            pilot, lambda: flow.step == "form" and bool(flow.query("#preset"))
        )
        assert flow.provider is not None
        assert flow.provider.provider_id == "mistral/default"


def test_manual_mistral_preset_still_deduplicates_existing_provider() -> None:
    flow, _services = make_flow(management=True)
    flow._apply_preset("mistral")
    assert flow.provider is not None
    assert flow.provider.provider_id == "mistral-2/default"


@pytest.mark.asyncio
async def test_back_cancel_from_every_step_discards_draft_and_returns_cancelled() -> (
    None
):
    for step in (
        "choice",
        "form",
        "credential",
        "probe",
        "models",
        "review",
        "again",
        "picker",
    ):
        flow, services = make_flow()
        flow._apply_preset("mistral")
        host = FlowHost(flow)
        async with host.run_test() as pilot:
            flow.step = step  # type: ignore[assignment]
            flow.refresh(recompose=True)
            await pilot.pause()
            if step not in {"choice", "form"}:
                flow.action_back()
                await pilot.pause()
            flow.action_cancel()
            await pilot.pause()
        assert host.results == [ProviderFlowResult("cancelled")]
        assert services.changes == []


@pytest.mark.asyncio
async def test_probe_failures_render_choices_and_manual_entry_reaches_review() -> None:
    errors = [
        DiscoveryError("auth_rejected", "Credentials were rejected"),
        DiscoveryError("rate_limited", "Slow down"),
        DiscoveryError("malformed", "Malformed response"),
    ]
    for error in errors:
        flow, _services = make_flow((error,))
        flow._apply_preset("mistral")
        host = FlowHost(flow)
        async with host.run_test() as pilot:
            flow._show("probe")
            flow._run_probe()
            await pilot.pause()
            await pilot.pause()
            assert error.message in [str(item.render()) for item in flow.query(Static)]
            assert {"retry", "edit-key", "manual"} <= {
                button.id for button in flow.query("Button")
            }
            flow.discovered = (DiscoveryItem(""),)
            flow._show("models")
            await pilot.pause()
            flow.query_one("#manual-wire-name", Input).value = "edited-wire"
            flow._save_model_selection()
            await pilot.pause()
            assert flow.step == "review"


@pytest.mark.asyncio
async def test_searchable_selection_filters_500_models_and_commits_filtered_wire_ids() -> (
    None
):
    items = tuple(DiscoveryItem(f"model-{index:03}") for index in range(500))
    flow, _services = make_flow((DiscoveryResult(items),))
    flow._apply_preset("mistral")
    flow.discovered = items
    async with FlowHost(flow).run_test() as pilot:
        flow._show("models")
        await pilot.pause()
        flow.query_one("#search", Input).value = "model-123"
        await pilot.pause()
        assert (
            str(flow.query_one("#model-count", Static).render())
            == "1 of 500 chat models"
        )
        models = flow.query_one("#models", SelectionList)
        models.select("model-123")
        flow._save_model_selection()
        await pilot.pause()
        assert tuple(flow._selected_models) == ("model-123",)


@pytest.mark.asyncio
async def test_optional_details_and_role_membership_are_committed() -> None:
    catalog = snapshot(
        models={
            "first": {
                "deployments": [{"provider": "example/default", "name": "first"}]
            },
            "second": {
                "deployments": [{"provider": "example/default", "name": "second"}]
            },
        },
        roles={
            "preferred": {
                "description": "preferred models",
                "models": ["first", "second"],
            }
        },
    )
    flow, services = make_flow(catalog=catalog)
    flow._apply_preset("mistral")
    flow._selected_models = {"third": ModelSelectionDraft("third", "third")}
    flow._detail_wire = "third"
    async with FlowHost(flow).run_test() as pilot:
        flow._show("review")
        await pilot.pause()
        flow.query_one("#input-price", Input).value = "0"
        flow.query_one("#roles", SelectionList).select("preferred")
        flow._save_details()
        await pilot.pause()
        flow._commit()
        await pilot.pause()
    changes = services.changes[0]
    deployment = changes.models["third"]["deployments"][0]  # type: ignore[index]
    assert deployment["prices"] == {"input": 0.0}
    assert changes.roles == {
        "preferred": {
            "description": "preferred models",
            "models": ["first", "second", "third"],
        }
    }


@pytest.mark.asyncio
async def test_masked_credentials_never_render_key_value() -> None:
    flow, _services = make_flow()
    flow._apply_preset("mistral")
    async with FlowHost(flow).run_test() as pilot:
        flow._show("credential")
        await pilot.pause()
        key = flow.query_one("#key", Input)
        key.value = "never-visible-secret"
        await pilot.pause()
        assert "never-visible-secret" not in str(flow.render())


def test_match_state_rendering_covers_existing_occupied_and_multiple() -> None:
    catalog = snapshot(
        models={
            "configured": {
                "deployments": [{"provider": "example/default", "name": "existing"}]
            },
            "occupied": {
                "deployments": [{"provider": "example/default", "name": "old-wire"}]
            },
            "first": {"deployments": [{"provider": "example/default", "name": "same"}]},
            "second": {
                "deployments": [{"provider": "example/default", "name": "same"}]
            },
        }
    )
    flow, _ = make_flow(catalog=catalog)
    flow.provider = ProviderDraft(
        None,
        "example/default",
        "Example",
        "https://example.test/v1",
        "openai",
        "",
        None,
    )
    assert flow._model_label(DiscoveryItem("existing")) == "existing\tconfigured"
    assert flow._configured_base(DiscoveryItem("zai-glm-5-3")) is None
    assert flow._configured_base(DiscoveryItem("occupied")) is None
    assert flow._configured_base(DiscoveryItem("same")) is None
    flow.discovered = (
        DiscoveryItem("existing"),
        DiscoveryItem("text-embedding-3-small"),
        DiscoveryItem("voxtral-mini"),
    )
    assert [item.wire_id for item in flow._picker_items()] == ["existing"]
    assert flow._is_configured_model(DiscoveryItem("existing"))


@pytest.mark.asyncio
async def test_final_picker_uses_expressions_and_shows_disabled_entries() -> None:
    catalog = snapshot(
        models={
            "usable": {
                "deployments": [{"provider": "example/default", "name": "wire"}]
            },
            "disabled": {
                "disabled": True,
                "deployments": [{"provider": "example/default", "name": "off"}],
            },
        },
        roles={"preferred": {"description": "preferred models", "models": ["usable"]}},
    )
    flow, _ = make_flow(catalog=catalog)
    async with FlowHost(flow).run_test() as pilot:
        flow._show("picker")
        await pilot.pause()
        values = [value for _label, value in flow._active_model_options()]
        assert values == ["usable", "@preferred"]
        assert any(
            "disabled (model disabled)" in str(item.render())
            for item in flow.query(Static)
        )
        assert "example/default/wire" not in values


@pytest.mark.asyncio
async def test_mistral_shared_key_confirmation_preserves_entered_key() -> None:
    catalog = snapshot(
        models={
            "mistral-wire": {
                "deployments": [{"provider": "mistral/default", "name": "mistral-wire"}]
            }
        }
    )
    flow, services = make_flow(
        (DiscoveryResult((DiscoveryItem("mistral-wire"),)),), catalog=catalog
    )
    host = FlowHost(flow)
    async with host.run_test() as pilot:
        flow._apply_preset("mistral")
        flow._show("credential")
        await pilot.pause()
        flow.query_one("#key", Input).value = "mistral-secret"
        flow._save_credential()
        await pilot.pause()
        assert flow.query_one("#key", Input).value == "mistral-secret"
        flow._save_credential()
        await pilot.pause()
        flow.query_one("#models", SelectionList).select("mistral-wire")
        flow._save_model_selection()
        await pilot.pause()
        flow._commit()
        await pilot.pause()
        flow._show("picker")
        await pilot.pause()
        flow._finish("mistral-wire")
        await pilot.pause()
        await pilot.pause()
    assert services.saved_keys == [("MISTRAL_API_KEY", "mistral-secret")]
    assert host.results == [
        ProviderFlowResult("completed", "mistral-wire", changed=True)
    ]


@pytest.mark.asyncio
async def test_multiple_match_collision_preserves_selected_base_role_memberships() -> (
    None
):
    catalog = snapshot(
        models={
            "first": {
                "deployments": [{"provider": "example/default", "name": "shared-wire"}]
            },
            "second": {
                "deployments": [{"provider": "example/default", "name": "shared-wire"}]
            },
        },
        roles={
            "first-role": {"description": "first", "models": ["first"]},
            "second-role": {"description": "second", "models": ["second"]},
        },
    )
    flow, services = make_flow(catalog=catalog)
    flow._overview_provider_id = "example/default"
    flow._select_existing_provider()
    flow.discovered = (DiscoveryItem("shared-wire"),)
    async with FlowHost(flow).run_test() as pilot:
        flow._show("models")
        await wait_for(pilot, lambda: bool(flow.query("#models")))
        flow.query_one("#models", SelectionList).select("shared-wire")
        flow._save_model_selection()
        await wait_for(pilot, lambda: bool(flow.query("#collision-choice")))
        flow.query_one("#collision-choice", Select).value = "second"
        await wait_for(
            pilot,
            lambda: flow.query_one("#roles", SelectionList).selected == ["second-role"],
        )
        assert flow.query_one("#roles", SelectionList).selected == ["second-role"]
        flow._commit()
        await wait_for(pilot, lambda: flow.step == "again")

    assert services.changes[0].roles is None
