from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Select, SelectionList, Static

from chartreux.cli.textual_ui.widgets.messages import UserMessage
from chartreux.core.agent_loop._loop import AgentLoop
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.model_catalog.loader import CatalogStore, load_catalog
from chartreux.ui.providers.contracts import (
    CatalogChanges,
    CatalogValidationError,
    ConfigPersistResult,
    ConfigReloadResult,
    CredentialSaveResult,
    DiscoveryItem,
    DiscoveryResult,
    ProviderDraft,
    ProviderFlowResult,
    TLSConfig,
)
from chartreux.ui.providers.flow import ProviderManagementScreen
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from tests.conftest import build_test_agent_loop, build_test_chartreux_app
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


@dataclass
class _Discovery:
    results: list[DiscoveryResult]
    calls: list[str] = field(default_factory=list)

    async def __call__(
        self,
        provider: ProviderDraft,
        credential: str | None,
        tls: TLSConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> DiscoveryResult:
        del credential, tls, http_client
        self.calls.append(provider.provider_id)
        return self.results.pop(0)


class _Credentials:
    def save_key(self, env_var: str, key: str) -> CredentialSaveResult:
        del env_var, key
        return CredentialSaveResult("saved")


class _FlowHost(App[ProviderFlowResult | None]):
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


class _RealConfigService:
    def __init__(self, orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]) -> None:
        self.orchestrator = orchestrator

    async def persist_theme(self, theme: str) -> ConfigPersistResult:
        del theme
        return ConfigPersistResult(True)

    async def reload_catalog_and_config(self) -> ConfigReloadResult:
        await self.orchestrator.reload()
        return ConfigReloadResult(self.orchestrator.config.catalog_snapshot)

    async def persist_active_model(self, expression: str) -> ConfigPersistResult:
        errors = await self.orchestrator.set_field(
            "/active_model", expression, reason="provider onboarding"
        )
        return ConfigPersistResult(not errors, str(errors[0]) if errors else None)


class _FailOnceStore:
    def __init__(self, store: CatalogStore) -> None:
        self.store = store
        self.fail = True

    def apply_changes(self, changes: CatalogChanges):  # type: ignore[no-untyped-def]
        if self.fail:
            self.fail = False
            return self.store.apply_changes(
                CatalogChanges(
                    changes.provider_id, changes.provider, tags={"invalid": ()}
                )
            )
        return self.store.apply_changes(changes)


class _ReloadFailure:
    async def persist_theme(self, theme: str) -> ConfigPersistResult:
        del theme
        return ConfigPersistResult(True)

    async def reload_catalog_and_config(self) -> ConfigReloadResult:
        return ConfigReloadResult(None, "synthetic reload failure")

    async def persist_active_model(self, expression: str) -> ConfigPersistResult:
        del expression
        raise AssertionError("active model must not persist after reload failure")


async def _wait_for(pilot, predicate, *, attempts: int = 50) -> None:  # type: ignore[no-untyped-def]
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError("Timed out waiting for the expected UI state")


async def _choose_active_model(  # type: ignore[no-untyped-def]
    pilot, flow: ProviderManagementScreen, expression: str
) -> None:
    picker = flow.query_one("#active-model", NavigableOptionList)
    picker.highlighted = picker.get_option_index(expression)
    picker.focus()
    await pilot.press("enter")


async def _add_generic_provider(
    pilot,
    flow: ProviderManagementScreen,
    *,
    name: str,
    wire: str,
    expected_step: str = "again",
) -> None:  # type: ignore[no-untyped-def]
    flow._show("form")
    await pilot.pause()
    flow.query_one("#preset", Select).value = "generic-openai"
    flow.query_one("#name", Input).value = name
    flow.query_one("#api-base", Input).value = f"https://{name.casefold()}.test/v1"
    flow.query_one("#env-var", Input).value = ""
    flow._save_form()
    await pilot.pause()
    flow._save_credential()
    await _wait_for(
        pilot, lambda: flow.step == "models" and bool(flow.query("#models"))
    )
    assert flow.step == "models"
    flow.query_one("#models", SelectionList).select(wire)
    flow._save_model_selection()
    await _wait_for(
        pilot,
        lambda: (
            flow.step == "review"
            and bool(flow.query("#detail-model"))
            and bool(flow.query("#continue"))
        ),
    )
    flow._commit()
    await pilot.pause()
    assert flow.step == expected_step, flow.error


async def _real_orchestrator(home: Path) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    config_path = home / "config.toml"
    config_path.write_text('active_model = "glm-5-2"\n')
    user = UserConfigLayer(path=config_path)
    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[user],
        default_layer_resolver=lambda: user,
        catalog_snapshot=load_catalog(home / "models.toml"),
        catalog_loader=lambda: load_catalog(home / "models.toml"),
    )


@pytest.mark.asyncio
async def test_provider_flow_adopts_unknown_model_through_real_disk_orchestrator(
    config_dir: Path,
) -> None:
    """The running loop observes an atomically-written catalog without losing history."""
    orchestrator = await _real_orchestrator(config_dir)
    loop = AgentLoop(
        config_orchestrator=orchestrator,
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )
    history = LLMMessage(role=Role.user, content="preserve this conversation")
    loop.messages.append(history)
    discovery = _Discovery([DiscoveryResult((DiscoveryItem("unknown-wire"),))])
    flow = ProviderManagementScreen(
        discovery=discovery,
        catalog_writer=CatalogStore(config_dir / "models.toml"),
        credentials=_Credentials(),
        config=_RealConfigService(orchestrator),
        snapshot=orchestrator.config.catalog_snapshot,
    )
    host = _FlowHost(flow)

    try:
        async with host.run_test() as pilot:
            await _add_generic_provider(
                pilot, flow, name="Unknown", wire="unknown-wire"
            )
            flow._show("picker")
            await pilot.pause()
            await _choose_active_model(pilot, flow, "unknown-wire")
            await pilot.pause()
            await pilot.pause()

        assert host.results == [
            ProviderFlowResult("completed", "unknown-wire", changed=True)
        ]
        assert discovery.calls == ["unknown/default"]
        assert "unknown-wire" in load_catalog(config_dir / "models.toml").catalog.models
        resolved = loop.config.get_active_model()
        assert (resolved.alias, resolved.name, resolved.provider) == (
            "unknown-wire",
            "unknown-wire",
            "unknown/default",
        )
        assert history in loop.messages
        assert sum(message == history for message in loop.messages) == 1
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_provider_flow_loop_commits_two_providers_and_combines_picker(
    config_dir: Path,
) -> None:
    discovery = _Discovery([
        DiscoveryResult((DiscoveryItem("first-wire"),)),
        DiscoveryResult((DiscoveryItem("second-wire"),)),
    ])
    store = CatalogStore(config_dir / "models.toml")
    initial = load_catalog(config_dir / "models.toml")
    flow = ProviderManagementScreen(
        discovery=discovery,
        catalog_writer=store,
        credentials=_Credentials(),
        config=_ReloadFailure(),
        snapshot=initial,
    )

    async with _FlowHost(flow).run_test() as pilot:
        await _add_generic_provider(pilot, flow, name="First", wire="first-wire")
        flow._show("form")  # The Add-another-provider loop returns to this same flow.
        await _add_generic_provider(pilot, flow, name="Second", wire="second-wire")
        flow._show("picker")
        await pilot.pause()
        values = [value for _label, value in flow._active_model_options()]

    catalog = load_catalog(config_dir / "models.toml").catalog
    assert discovery.calls == ["first/default", "second/default"]
    assert {"first/default", "second/default"} <= set(catalog.providers)
    assert {"first-wire", "second-wire"} <= set(catalog.models)
    assert {"first-wire", "second-wire"} <= set(values)


@pytest.mark.asyncio
async def test_provider_flow_recovers_catalog_validation_and_surfaces_reload_failure(
    config_dir: Path,
) -> None:
    store = CatalogStore(config_dir / "models.toml")
    initial = store.apply_changes(
        # Seed bytes so a failed mid-flow write has an observable nonempty baseline.
        CatalogChanges(
            "seed/default",
            {"api_base": "https://seed.test/v1"},
            {"seed": {"deployments": [{"provider": "seed/default", "name": "seed"}]}},
        )
    )
    assert not isinstance(initial, CatalogValidationError)
    before = (config_dir / "models.toml").read_bytes()
    discovery = _Discovery([DiscoveryResult((DiscoveryItem("retry-wire"),))])
    failed_writer = _FailOnceStore(store)
    flow = ProviderManagementScreen(
        discovery=discovery,
        catalog_writer=failed_writer,
        credentials=_Credentials(),
        config=_ReloadFailure(),
        snapshot=load_catalog(config_dir / "models.toml"),
    )

    async with _FlowHost(flow).run_test() as pilot:
        await _add_generic_provider(
            pilot, flow, name="Retry", wire="retry-wire", expected_step="review"
        )
        assert flow.step == "review"
        assert "cannot be empty" in (flow.error or "")
        assert (config_dir / "models.toml").read_bytes() == before
        flow._selected_models = {
            "good-wire": flow._selected_models["retry-wire"].__class__(
                "good-wire", "good-wire"
            )
        }
        flow._detail_wire = "good-wire"
        flow._show("review")
        await pilot.pause()
        flow._commit()
        await pilot.pause()
        assert flow.step == "again"
        flow._show("picker")
        await pilot.pause()
        await _choose_active_model(pilot, flow, "good-wire")
        await pilot.pause()
        await pilot.pause()
        assert flow.step == "picker"
        assert flow.error == "synthetic reload failure"
        assert "good-wire" in flow.snapshot.catalog.models

    assert "good-wire" in load_catalog(config_dir / "models.toml").catalog.models


@pytest.mark.asyncio
async def test_providers_host_completion_preserves_transcript_and_agent_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chartreux.core.model_catalog.discovery.discover_models",
        _Discovery([DiscoveryResult((DiscoveryItem("hosted-wire"),))]),
    )
    loop = build_test_agent_loop()
    state = LLMMessage(role=Role.user, content="existing agent state")
    loop.messages.append(state)
    app = build_test_chartreux_app(agent_loop=loop)

    async with app.run_test() as pilot:
        await app._messages_area.mount(UserMessage("existing transcript"))
        transcript = list(app._messages_area.children)
        assert await app._handle_command("/providers")
        for _ in range(50):
            await pilot.pause()
            if isinstance(app.screen, ProviderManagementScreen):
                break
        screen = app.screen
        assert isinstance(screen, ProviderManagementScreen)
        screen._show("form")
        await _add_generic_provider(pilot, screen, name="Hosted", wire="hosted-wire")
        screen._show("picker")
        await pilot.pause()
        await _choose_active_model(pilot, screen, "glm-5-2")
        for _ in range(50):
            await pilot.pause()
            if app.screen is not screen:
                break

        assert list(app._messages_area.children)[: len(transcript)] == transcript
        assert state in loop.messages
        assert sum(message == state for message in loop.messages) == 1
