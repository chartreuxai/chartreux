"""End-to-end catalog and host boundaries for Provider Settings."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from textual.app import App, ComposeResult
from textual.widgets import Input, OptionList, Static

from chartreux.cli.textual_ui.widgets.messages import UserMessage
from chartreux.core.agent_loop._loop import AgentLoop
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.model_catalog.contracts import (
    CatalogChanges,
    CatalogValidationError,
    ConfigPersistResult,
    ConfigReloadResult,
    CredentialSaveResult,
    DiscoveryItem,
    DiscoveryResult,
    ProviderDraft,
    ProviderWorkbenchResult,
    TLSConfig,
)
from chartreux.core.model_catalog.loader import CatalogStore, load_catalog
from chartreux.core.model_catalog.presets import FULLY_CUSTOM
from chartreux.ui.providers.workbench import (
    ProviderWorkbenchScreen,
    WorkbenchList,
    WorkbenchView,
)
from tests.conftest import build_test_chartreux_app
from tests.snapshots.snapshot_event_loop import install_snapshot_wake
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


@pytest_asyncio.fixture(autouse=True)
async def _snapshot_event_loop_wake() -> None:
    install_snapshot_wake()


class Services:
    def __init__(
        self,
        home: Path,
        orchestrator: ConfigOrchestrator[ChartreuxConfigSchema] | None = None,
    ):
        self.store = CatalogStore(home / "models.toml")
        self.home = home
        self.orchestrator = orchestrator
        self.fail_once = False
        self.reload_fails = False

    def resolve_key(self, env: str) -> str | None:
        return "test-key" if env == "MISTRAL_API_KEY" else None

    def save_key(self, env_var: str, key: str) -> CredentialSaveResult:
        return CredentialSaveResult("saved")

    async def discover(
        self,
        provider: ProviderDraft,
        credential: str | None,
        tls: TLSConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> DiscoveryResult:
        return DiscoveryResult((DiscoveryItem(f"{provider.name}-wire"),))

    async def persist_theme(self, theme: str) -> ConfigPersistResult:
        return ConfigPersistResult(True)

    async def reload_catalog_and_config(self) -> ConfigReloadResult:
        if self.reload_fails:
            return ConfigReloadResult(None, "synthetic reload failure")
        if self.orchestrator:
            await self.orchestrator.reload()
        return ConfigReloadResult(load_catalog(self.home / "models.toml"))

    def apply_changes(self, changes: CatalogChanges):  # type: ignore[no-untyped-def]
        if self.fail_once:
            self.fail_once = False
            return CatalogValidationError("synthetic catalog validation failure")
        return self.store.apply_changes(changes)

    def screen(self) -> ProviderWorkbenchScreen:
        return ProviderWorkbenchScreen(
            discovery=self.discover,
            catalog_writer=self,
            credentials=self,
            credential_resolver=self.resolve_key,
            config=self,
            snapshot=load_catalog(self.home / "models.toml"),
        )


class Host(App[None]):
    config: SimpleNamespace

    def __init__(self, screen: ProviderWorkbenchScreen):
        super().__init__()
        self.provider_screen = screen
        self.results: list[ProviderWorkbenchResult] = []

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        self.push_screen(self.provider_screen, self._record)

    def _record(self, result: ProviderWorkbenchResult | None) -> None:
        if result:
            self.results.append(result)


@pytest.mark.asyncio
async def test_workbench_ascii_chrome_cursor_and_feedback(config_dir: Path) -> None:
    screen = Services(config_dir).screen()
    app = Host(screen)
    app.config = SimpleNamespace(ascii_chrome=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        rows = screen.query_one(WorkbenchList)
        assert any(
            "> " in "".join(segment.text for segment in rows.render_line(y))
            for y in range(rows.size.height)
        )
        screen._message = "Key saved."
        screen._feedback_kind = "success"
        assert screen._feedback().plain.startswith("+ Saved:")
        screen._message = "Failed to save key"
        screen._feedback_kind = "error"
        assert screen._feedback().plain.startswith("x Failed:")


async def wait_for(pilot, predicate, *, tries: int = 100) -> None:  # type: ignore[no-untyped-def]
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError(
        f"Timed out waiting for provider transition: {pilot.app.screen._message}"
    )


async def add_provider(pilot, screen: ProviderWorkbenchScreen, name: str) -> None:  # type: ignore[no-untyped-def]
    if screen.view == WorkbenchView.PRESETS:
        await select_option(
            pilot, screen.query_one("#wb-presets", WorkbenchList), "add-another"
        )
    if screen.view == WorkbenchView.PROVIDERS:
        await select_option(
            pilot, screen.query_one("#wb-providers", OptionList), "\x00add"
        )
    if screen.view == WorkbenchView.CHOOSE:
        await select_option(
            pilot, screen.query_one("#wb-choose", WorkbenchList), FULLY_CUSTOM.id
        )
    for key, value in (("name", name), ("base", f"https://{name}.test/v1")):
        screen._connection_action(key)
        screen.query_one("#wb-input", Input).value = value
        await pilot.press("enter")
    screen._connection_action("continue")
    await wait_for(
        pilot, lambda: not screen._busy and screen.view == WorkbenchView.MODELS
    )
    assert screen.state is not None
    screen._select_action("manual")
    screen.query_one("#wb-input", Input).value = f"{name}-wire"
    await pilot.press("enter")
    screen._select_action("continue-presets")
    await wait_for(
        pilot,
        lambda: (
            not screen._busy
            and name in screen.snapshot.catalog.providers
            and screen.view == WorkbenchView.PRESETS
        ),
    )


async def select_option(
    pilot, widget: WorkbenchList | OptionList, option_id: str
) -> None:  # type: ignore[no-untyped-def]
    for _ in range(widget.option_count + 1):
        if widget.highlighted_option and widget.highlighted_option.id == option_id:
            await pilot.press("enter")
            return
        await pilot.press("down")
    raise AssertionError(f"Option {option_id!r} was not reachable.")


async def set_orchestrator_preset(
    pilot, screen: ProviderWorkbenchScreen, model: str, thinking: str = "off"
) -> None:  # type: ignore[no-untyped-def]
    presets = screen.query_one("#wb-presets", WorkbenchList)
    await select_option(pilot, presets, "preset:orchestrator")
    editor = screen.query_one("#wb-preset-editor", WorkbenchList)
    await select_option(pilot, editor, "model")
    await select_option(pilot, screen.query_one("#wb-picker", WorkbenchList), model)
    await select_option(pilot, editor, "thinking")
    await select_option(pilot, screen.query_one("#wb-picker", WorkbenchList), thinking)
    await select_option(pilot, editor, "apply")
    await select_option(pilot, presets, "finish")
    await wait_for(pilot, lambda: not screen._busy)


async def real_orchestrator(home: Path) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    path = home / "config.toml"
    path.write_text("")
    user = UserConfigLayer(path=path)
    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[user],
        default_layer_resolver=lambda: user,
        catalog_snapshot=load_catalog(home / "models.toml"),
        catalog_loader=lambda: load_catalog(home / "models.toml"),
    )


@pytest.mark.asyncio
async def test_unknown_model_adopted_on_disk_through_real_orchestrator(
    config_dir: Path,
) -> None:
    orchestrator = await real_orchestrator(config_dir)
    loop = AgentLoop(
        config_orchestrator=orchestrator,
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )
    history = LLMMessage(role=Role.user, content="preserve conversation")
    loop.messages.append(history)
    service = Services(config_dir, orchestrator)
    screen = service.screen()
    try:
        async with Host(screen).run_test() as pilot:
            await add_provider(pilot, screen, "unknown")
            await set_orchestrator_preset(pilot, screen, "unknown-wire")
        assert "unknown-wire" in load_catalog(config_dir / "models.toml").catalog.models
        assert (
            load_catalog(config_dir / "models.toml").catalog.roles["orchestrator"].model
            == "unknown-wire"
        )
        assert loop.config.get_active_model().provider == "unknown"
        assert loop.messages.count(history) == 1
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_two_provider_models_are_available_to_default_preset(
    config_dir: Path,
) -> None:
    screen = Services(config_dir).screen()
    async with Host(screen).run_test() as pilot:
        await add_provider(pilot, screen, "first")
        assert "first-wire" in load_catalog(config_dir / "models.toml").catalog.models
        await add_provider(pilot, screen, "second")
        await set_orchestrator_preset(pilot, screen, "second-wire")
    catalog = load_catalog(config_dir / "models.toml").catalog
    assert {"first", "second"} <= catalog.providers.keys()
    assert catalog.roles["orchestrator"].model == "second-wire"


@pytest.mark.asyncio
async def test_validation_recovery_and_saved_reload_failure(config_dir: Path) -> None:
    service = Services(config_dir)
    service.fail_once = True
    screen = service.screen()
    async with Host(screen).run_test() as pilot:
        screen._choose_preset(FULLY_CUSTOM)
        for key, value in (("name", "retry"), ("base", "https://retry.test/v1")):
            screen._connection_action(key)
            screen.query_one("#wb-input", Input).value = value
            await pilot.press("enter")
        screen._connection_action("continue")
        await wait_for(
            pilot, lambda: not screen._busy and "validation failure" in screen._message
        )
        assert "retry" not in load_catalog(config_dir / "models.toml").catalog.providers
        screen._connection_action("continue")
        await wait_for(
            pilot,
            lambda: (
                not screen._busy
                and screen.view == WorkbenchView.MODELS
                and screen.state is not None
                and screen.state.discovery is not None
            ),
        )
        screen._select_action("manual")
        screen.query_one("#wb-input", Input).value = "retry-wire"
        await pilot.press("enter")
        before = (
            (config_dir / "models.toml").read_bytes()
            if (config_dir / "models.toml").exists()
            else None
        )
        service.fail_once = True
        screen._select_action("continue-presets")
        await wait_for(
            pilot, lambda: not screen._busy and "validation failure" in screen._message
        )
        assert screen.state is not None and screen.state.dirty
        assert "✗ Failed: apply" in str(screen.query_one("#wb-help").render())
        if before is None:
            assert not (config_dir / "models.toml").exists()
        else:
            assert (config_dir / "models.toml").read_bytes() == before
        service.reload_fails = True
        screen._select_action("continue-presets")
        await wait_for(pilot, lambda: not screen._busy and screen._reload_failed)
        assert "retry" in screen.snapshot.catalog.providers
        assert "Saved; Reload Failed" in screen._message
        assert "retry" in load_catalog(config_dir / "models.toml").catalog.providers


@pytest.mark.asyncio
async def test_providers_host_completion_preserves_transcript_and_agent_state(
    config_dir: Path,
) -> None:
    orchestrator = await real_orchestrator(config_dir)
    loop = AgentLoop(
        config_orchestrator=orchestrator,
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )
    state = LLMMessage(role=Role.user, content="existing agent state")
    loop.messages.append(state)
    app = build_test_chartreux_app(agent_loop=loop)
    async with app.run_test() as pilot:
        await app._messages_area.mount(UserMessage("existing transcript"))
        transcript = list(app._messages_area.children)
        assert await app._handle_command("/providers")
        await wait_for(pilot, lambda: isinstance(app.screen, ProviderWorkbenchScreen))
        screen = app.screen
        assert isinstance(screen, ProviderWorkbenchScreen)
        screen.config = Services(config_dir, orchestrator)
        await add_provider(pilot, screen, "host-provider")
        await set_orchestrator_preset(pilot, screen, "host-provider-wire")
        screen._finish()
        await wait_for(pilot, lambda: app.screen is not screen)
        assert (
            "host-provider"
            in load_catalog(config_dir / "models.toml").catalog.providers
        )
        assert list(app._messages_area.children)[: len(transcript)] == transcript
        assert loop.messages.count(state) == 1


@pytest.mark.asyncio
async def test_customized_connection_replacement_is_confirmed_before_disk_write(
    config_dir: Path,
) -> None:
    services = Services(config_dir)
    services.store.upsert_provider(
        {"api_base": "https://original.test/v1", "api_style": "openai"}, "custom"
    )
    screen = services.screen()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        screen._expand("custom")
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://replaced.test/v1"
        await pilot.press("enter")
        screen._select_action("apply")
        assert screen._confirm == "replace-connection"
        assert (
            load_catalog(config_dir / "models.toml")
            .catalog.providers["custom"]
            .api_base
            == "https://original.test/v1"
        )
        screen.action_confirm_yes()
        await wait_for(pilot, lambda: not screen._busy and screen._confirm is None)
        assert (
            load_catalog(config_dir / "models.toml")
            .catalog.providers["custom"]
            .api_base
            == "https://replaced.test/v1"
        )
