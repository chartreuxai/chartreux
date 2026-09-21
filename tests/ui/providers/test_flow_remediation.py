from __future__ import annotations

from dataclasses import dataclass, field

from httpx import AsyncClient
import pytest
from textual.widgets import Button, Input, Label, Select, Static

from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.ui.providers.contracts import (
    CatalogChanges,
    CatalogWriteResult,
    DiscoveryItem,
    DiscoveryResult,
    ModelSelectionDraft,
    ProviderDraft,
    TLSConfig,
)
from chartreux.ui.providers.flow import ProviderManagementScreen
from tests.ui.providers.test_flow import (
    FakeServices,
    FlowHost,
    make_flow,
    snapshot,
    wait_for,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "expected_credential"),
    (("disable-auth", None), ("endpoint", None), ("env-var", None), ("preset", None)),
)
async def test_discovery_drops_stale_credential_after_connection_identity_change(
    change: str, expected_credential: str | None
) -> None:
    """Every connection-identity change requires a fresh credential approval."""
    flow, _ = make_flow()
    calls: list[tuple[ProviderDraft, str | None]] = []

    async def discovery(
        provider: ProviderDraft,
        credential: str | None,
        tls: TLSConfig,
        http_client: AsyncClient | None = None,
    ) -> DiscoveryResult:
        _ = tls, http_client
        calls.append((provider, credential))
        return DiscoveryResult((DiscoveryItem("wire"),))

    flow.discovery = discovery
    flow.provider = ProviderDraft(
        "mistral",
        "new/default",
        "New",
        "https://api.mistral.ai/v1",
        "openai",
        "MISTRAL_API_KEY",
        "stale-secret",
        backend="mistral",
    )
    flow._credential_value = "stale-secret"

    async with FlowHost(flow).run_test() as pilot:
        flow._show("form")
        await wait_for(pilot, lambda: bool(flow.query("#api-base")))
        if change == "disable-auth":
            flow.query_one("#env-var", Input).value = ""
        elif change == "endpoint":
            flow.query_one("#api-base", Input).value = "https://other.example/v1"
        elif change == "env-var":
            flow.query_one("#env-var", Input).value = "OTHER_API_KEY"
        else:
            flow.query_one("#preset", Select).value = "generic-openai"
            await pilot.pause()
            flow.query_one("#api-base", Input).value = "https://generic.example/v1"
        flow._save_form()
        await pilot.pause()

        assert flow.provider is not None
        assert flow.provider.key is None
        assert flow._credential_value is None
        if change == "disable-auth":
            flow._save_credential()
        else:
            # Discovery is not normally reachable until a new key is entered. Invoke
            # its boundary directly to prove it cannot reuse the discarded cache.
            flow._run_probe()
        await wait_for(pilot, lambda: len(calls) == 1)

    assert calls[0][1] == expected_credential
    if change == "disable-auth":
        assert calls[0][0].api_key_env_var == ""


@dataclass
class EvolvingServices(FakeServices):
    """Fake writer whose snapshots include every prior committed change."""

    changes: list[CatalogChanges] = field(default_factory=list)

    def apply_changes(self, changes: CatalogChanges) -> CatalogWriteResult:
        self.changes.append(changes)
        document = self.snapshot.catalog.model_dump(mode="python")
        providers = document["providers"]
        assert isinstance(providers, dict)
        providers.setdefault(changes.provider_id, {}).update(changes.provider)
        models = document["models"]
        assert isinstance(models, dict)
        for base, patch in changes.models.items():
            models.setdefault(base, {}).update(patch)
        if changes.tags is not None:
            document["tags"] = {
                tag: list(members) for tag, members in changes.tags.items()
            }
        self.snapshot = CatalogSnapshot(
            ModelCatalog.model_validate(document), "evolving"
        )
        return CatalogWriteResult(self.snapshot, changed=True)


def _provider(provider_id: str) -> ProviderDraft:
    return ProviderDraft(
        "generic-openai",
        provider_id,
        provider_id.rsplit("/", 1)[0],
        f"https://{provider_id.split('/', 1)[0]}.example/v1",
        "openai",
        "",
        None,
    )


@pytest.mark.asyncio
async def test_second_catalog_save_preserves_first_provider_tag_membership() -> None:
    initial = snapshot(providers={"existing/default": {"api_base": "https://old/v1"}})
    services = EvolvingServices([], initial)
    flow, _ = make_flow(catalog=initial)
    flow.catalog_writer = services
    flow.provider = _provider("provider-a/default")
    flow._selected_models = {"a-wire": ModelSelectionDraft("a-wire", "model-a")}
    flow._detail_wire = "a-wire"

    async with FlowHost(flow).run_test() as pilot:
        flow._show("review")
        await wait_for(pilot, lambda: bool(flow.query("#tags")))
        flow.query_one("#tags", Input).value = "preferred"
        flow._commit()
        await wait_for(pilot, lambda: flow.step == "again")
        assert services.changes[0].tags == {"preferred": ("model-a",)}

        flow.provider = _provider("provider-b/default")
        flow._selected_models = {"b-wire": ModelSelectionDraft("b-wire", "model-b")}
        flow._detail_wire = "b-wire"
        flow._show("review")
        await wait_for(pilot, lambda: bool(flow.query("#tags")))
        await pilot.pause()
        assert flow._detail_wire == "b-wire"
        flow.query_one("#tags", Input).value = "secondary"
        flow._commit()
        await pilot.pause()
        assert flow.step == "again", flow.error
        assert services.changes[1].tags == {
            "preferred": ("model-a",),
            "secondary": ("model-b",),
        }

    assert flow.catalog.tags["preferred"] == ("model-a",)
    assert flow.catalog.tags["secondary"] == ("model-b",)


def _review_flow() -> tuple[ProviderManagementScreen, FakeServices]:
    flow, services = make_flow()
    flow.provider = _provider("new/default")
    flow._selected_models = {"model-a": ModelSelectionDraft("model-a", "model-a")}
    flow._detail_wire = "model-a"
    return flow, services


@pytest.mark.asyncio
async def test_invalid_review_price_preserves_exact_typed_input() -> None:
    flow, _ = _review_flow()
    async with FlowHost(flow).run_test() as pilot:
        flow._show("review")
        await wait_for(pilot, lambda: bool(flow.query("#input-price")))
        flow.query_one("#input-price", Input).value = "not-a-price"
        flow._save_details()
        await wait_for(pilot, lambda: flow.error is not None)
        assert flow.query_one("#input-price", Input).value == "not-a-price"


@pytest.mark.asyncio
async def test_invalid_endpoint_preserves_exact_typed_input() -> None:
    flow, _ = _review_flow()
    async with FlowHost(flow).run_test() as pilot:
        flow._show("form")
        await wait_for(pilot, lambda: bool(flow.query("#api-base")))
        endpoint = flow.query_one("#api-base", Input)
        flow.query_one("#name", Input).value = "Edited provider name"
        endpoint.value = "not an endpoint"
        flow._save_form()
        await wait_for(
            pilot, lambda: flow.query_one("#api-base", Input) is not endpoint
        )
        assert flow.query_one("#api-base", Input).value == "not an endpoint"
        assert flow.query_one("#name", Input).value == "Edited provider name"


@pytest.mark.asyncio
async def test_review_model_switch_keeps_invalid_input_on_outgoing_model() -> None:
    flow, _ = _review_flow()
    flow._selected_models["model-b"] = ModelSelectionDraft("model-b", "model-b")
    async with FlowHost(flow).run_test() as pilot:
        flow._show("review")
        await wait_for(pilot, lambda: bool(flow.query("#detail-model")))
        flow.query_one("#input-price", Input).value = "invalid-a-price"
        flow.query_one("#detail-model", Select).value = "model-b"
        await wait_for(pilot, lambda: flow.error is not None)
        assert flow._detail_wire == "model-a"
        assert flow.query_one("#input-price", Input).value == "invalid-a-price"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ("clear", "bulk"))
async def test_review_tag_actions_preserve_unrelated_typed_inputs(action: str) -> None:
    flow, _ = _review_flow()
    flow._tag_orders = {"preferred": ("model-a", "other")}
    async with FlowHost(flow).run_test() as pilot:
        flow._show("review")
        await wait_for(pilot, lambda: bool(flow.query("#tags")))
        flow.query_one("#input-price", Input).value = "12.345"
        flow.query_one("#aliases", Input).value = "exact-alias"
        flow.query_one("#tags", Input).value = "preferred"
        if action == "clear":
            flow._clear_detail_metadata("tags")
        else:
            flow._assign_bulk_tags()
        await pilot.pause()
        assert flow.query_one("#input-price", Input).value == "12.345"
        assert flow.query_one("#aliases", Input).value == "exact-alias"
        assert flow.query_one("#tags", Input).value == (
            "" if action == "clear" else "preferred"
        )


@pytest.mark.asyncio
async def test_review_back_navigation_preserves_exact_typed_input() -> None:
    flow, _ = _review_flow()
    flow.discovered = (DiscoveryItem("model-a"),)
    flow._selected_wires = {"model-a"}
    async with FlowHost(flow).run_test() as pilot:
        flow._show("review")
        await wait_for(pilot, lambda: bool(flow.query("#aliases")))
        flow.query_one("#aliases", Input).value = "typed-before-back"
        flow.action_back()
        await wait_for(
            pilot, lambda: flow.step == "models" and bool(flow.query("#models"))
        )
        flow._save_model_selection()
        await wait_for(
            pilot, lambda: flow.step == "review" and bool(flow.query("#aliases"))
        )
        assert flow.query_one("#aliases", Input).value == "typed-before-back"


@pytest.mark.asyncio
async def test_provider_flow_renders_hint_labels_and_friendly_overview() -> None:
    catalog = snapshot(
        providers={
            "mistral/default": {
                "api_base": "https://api.mistral.ai/v1",
                "api_key_env_var": "MISTRAL_API_KEY",
                "backend": "mistral",
            }
        },
        models={
            "small": {"deployments": [{"provider": "mistral/default", "name": "small"}]}
        },
    )
    flow, _ = make_flow(catalog=catalog, management=True)
    async with FlowHost(flow).run_test() as pilot:
        await wait_for(pilot, lambda: bool(flow.query("#provider-shortcut-hint")))
        assert "Esc" in str(flow.query_one("#provider-shortcut-hint", Static).render())
        assert "Mistral" in flow._overview_label("mistral/default")
        assert "https://api.mistral.ai/v1" in flow._overview_label("mistral/default")
        assert "1 models" in flow._overview_label("mistral/default")
        flow._show("form")
        await wait_for(pilot, lambda: bool(flow.query("#name")))
        labels = [str(label.render()) for label in flow.query(Label)]
        assert "Provider name *" in labels
        assert "API base *" in labels
        assert "API style" in labels


@pytest.mark.asyncio
async def test_overview_requires_selection_and_disables_empty_actions() -> None:
    flow, _ = make_flow(management=True)
    async with FlowHost(flow).run_test() as pilot:
        await wait_for(pilot, lambda: bool(flow.query("#edit")))
        await pilot.click("#edit")
        await wait_for(pilot, lambda: flow.error is not None)
        assert flow.step == "overview"

    empty = CatalogSnapshot(
        ModelCatalog.model_validate({"providers": {}, "models": {}, "tags": {}}),
        "empty",
    )
    flow, _ = make_flow(catalog=empty, management=True)
    async with FlowHost(flow).run_test() as pilot:
        await wait_for(pilot, lambda: bool(flow.query("#edit")))
        for button_id in ("edit", "discover", "credential"):
            assert flow.query_one(f"#{button_id}", Button).disabled


@pytest.mark.asyncio
async def test_discovery_busy_state_disables_probe_controls() -> None:
    flow, _ = make_flow()
    flow._apply_preset("mistral")
    async with FlowHost(flow).run_test() as pilot:
        flow.error = "Discovering models…"
        flow._show("probe")
        await wait_for(pilot, lambda: bool(flow.query("#retry")))
        for button_id in ("retry", "manual", "edit-key"):
            assert flow.query_one(f"#{button_id}", Button).disabled


@pytest.mark.asyncio
async def test_onboarding_mistral_shortcut_drops_prior_provider_credential_cache() -> (
    None
):
    """A cached custom key must never be presented or discovered as Mistral's key."""
    flow, _ = make_flow()
    flow.provider = ProviderDraft(
        "generic-openai",
        "custom/default",
        "Custom",
        "https://custom.example/v1",
        "openai",
        "CUSTOM_API_KEY",
        "custom-secret",
    )
    flow._credential_value = "custom-secret"

    async with FlowHost(flow).run_test() as pilot:
        flow._show("choice")
        await wait_for(pilot, lambda: bool(flow.query("#mistral")))
        await pilot.click("#mistral")
        await wait_for(pilot, lambda: flow.step == "credential")
        assert flow.provider is not None
        assert flow.provider.provider_id == "mistral/default"
        assert flow._credential_value is None
        assert flow.query_one("#key", Input).value == ""


@pytest.mark.asyncio
async def test_repair_picker_back_returns_to_choice_without_reviewing_empty_provider() -> (
    None
):
    flow, _ = make_flow()
    flow.step = "picker"
    host = FlowHost(flow)
    async with host.run_test() as pilot:
        await wait_for(pilot, lambda: bool(flow.query("#active-model")))
        flow.action_back()
        await wait_for(
            pilot, lambda: flow.step == "choice" and bool(flow.query("#mistral"))
        )
        flow.action_back()
        await wait_for(pilot, lambda: bool(host.results))

    assert host.results[0].status == "cancelled"


@pytest.mark.asyncio
async def test_invalid_form_submission_drops_credential_after_endpoint_change() -> None:
    """Fixing a validation error must still require a key for a new endpoint."""
    flow, _ = make_flow()
    flow.provider = ProviderDraft(
        "generic-openai",
        "new/default",
        "Original name",
        "https://old.example/v1",
        "openai",
        "EXAMPLE_API_KEY",
        "stale-secret",
    )
    flow._credential_value = "stale-secret"

    async with FlowHost(flow).run_test() as pilot:
        flow._show("form")
        await wait_for(pilot, lambda: bool(flow.query("#api-base")))
        flow.query_one("#api-base", Input).value = "https://new.example/v1"
        flow.query_one("#name", Input).value = ""
        flow._save_form()
        await wait_for(pilot, lambda: flow.error is not None)

        flow.query_one("#name", Input).value = "Corrected name"
        flow._save_form()
        await wait_for(
            pilot, lambda: flow.step == "credential" and bool(flow.query("#key"))
        )

        assert flow.provider is not None
        assert flow.provider.key is None
        assert flow._credential_value is None
        assert flow.query_one("#key", Input).value == ""


@pytest.mark.asyncio
async def test_overview_and_active_model_use_keyboard_option_lists_with_catalog_status() -> (
    None
):
    catalog = snapshot(
        providers={
            "codex/local": {
                "api_base": "http://127.0.0.1:18080/v1",
                "api_style": "openai-responses",
            },
            "custom/default": {"api_base": "https://custom.example/v1"},
        },
        models={
            "canonical": {
                "aliases": ["old-name", "latest"],
                "deployments": [{"provider": "codex/local", "name": "wire"}],
            }
        },
    )
    flow, _ = make_flow(catalog=catalog, management=True)
    async with FlowHost(flow).run_test(size=(80, 24)) as pilot:
        await wait_for(pilot, lambda: bool(flow.query("#overview-provider")))
        overview = flow.query_one("#overview-provider")
        assert overview.has_focus
        assert "Available" in flow._overview_label("codex/local")
        assert "Configured" in flow._overview_label("custom/default")
        await pilot.press("j", "enter")
        assert flow._overview_provider_id == "custom/default"

        flow._show("picker")
        await wait_for(pilot, lambda: bool(flow.query("#active-model")))
        active = flow.query_one("#active-model")
        assert active.has_focus
        labels = [label for label, _value in flow._active_model_options()]
        assert len(labels) == 1
        assert "aliases: old-name, latest" in labels[0]
        assert "Available" in labels[0]


@pytest.mark.asyncio
async def test_enter_submits_credential_input() -> None:
    flow, services = make_flow()
    flow._overview_provider_id = "mistral/default"
    flow._select_existing_provider()
    async with FlowHost(flow).run_test(size=(80, 24)) as pilot:
        flow._show("credential")
        await wait_for(pilot, lambda: bool(flow.query("#key")))
        key = flow.query_one("#key", Input)
        key.value = "test-key"
        key.focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: flow.step == "models")
    assert services.saved_keys == [("MISTRAL_API_KEY", "test-key")]


@pytest.mark.asyncio
async def test_invalid_api_base_escape_returns_to_previous_step_without_crashing() -> (
    None
):
    flow, _ = make_flow()
    flow._apply_preset("generic-openai")
    async with FlowHost(flow).run_test(size=(80, 24)) as pilot:
        flow._show("form")
        await wait_for(pilot, lambda: bool(flow.query("#api-base")))
        flow.query_one("#api-base", Input).value = "not-a-url"
        flow._save_form()
        await wait_for(pilot, lambda: flow.error is not None)
        await pilot.press("escape")
        await wait_for(pilot, lambda: flow.step == "choice")


@pytest.mark.asyncio
async def test_models_escape_clears_filter_before_navigating_back() -> None:
    flow, _ = make_flow()
    flow.provider = _provider("new/default")
    flow.discovered = (DiscoveryItem("alpha"), DiscoveryItem("beta"))
    async with FlowHost(flow).run_test(size=(80, 24)) as pilot:
        flow._show("models")
        await wait_for(pilot, lambda: bool(flow.query("#search")))
        search = flow.query_one("#search", Input)
        search.value = "beta"
        search.focus()
        await pilot.press("escape")
        await wait_for(pilot, lambda: search.value == "")
        assert flow.step == "models"
        await pilot.press("escape")
        await wait_for(pilot, lambda: flow.step == "probe")


@pytest.mark.asyncio
async def test_occupied_slot_commits_to_its_existing_base() -> None:
    catalog = snapshot(
        models={
            "occupied": {
                "deployments": [{"provider": "example/default", "name": "old-wire"}]
            }
        }
    )
    flow, services = make_flow(catalog=catalog)
    flow._overview_provider_id = "example/default"
    flow._select_existing_provider()
    flow.discovered = (DiscoveryItem("occupied"),)
    async with FlowHost(flow).run_test(size=(80, 24)) as pilot:
        flow._show("models")
        await wait_for(pilot, lambda: bool(flow.query("#models")))
        await pilot.press("space", "tab", "enter")
        await wait_for(pilot, lambda: flow.step == "review")
        collision = flow.query_one("#collision-choice", Select)
        assert collision.value == "existing"
        flow.query_one("#continue", Button).focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: flow.step == "again")
    assert set(services.changes[0].models) == {"occupied"}
