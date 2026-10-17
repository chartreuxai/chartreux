"""Shared test-support surface for the provider workbench suite (W10, stage 19f).

The definitions here are the shared module-level helper surface of
``tests/ui/providers/test_workbench.py``, relocated AST-verbatim in the
19f phase-(i) helper-relocation commit: the catalog snapshot builder
(``snapshot``), the services double (``Services``), the app host
(``Host``), the screen factory (``setup``), and the shared pilot
helpers ``expand``, ``wait_until``, ``press_option``,
``press_model_option``, ``add_preset`` and ``open_split_view``. The
preset constants ``MISTRAL`` and ``FULLY_CUSTOM`` are re-exported from
``chartreux.core.model_catalog.presets`` for the seam importers. No
test bodies, fixtures, or marks live here; the test population stays
in ``test_workbench.py`` until the phase-(ii) concern moves.
Importers: ``test_workbench.py`` itself plus the seam files
``tests/setup/onboarding/test_screens.py``,
``tests/ui/providers/test_workbench_onboarding_summary.py``,
``tests/ui/providers/test_workbench_pointer_async.py``,
``tests/ui/providers/test_graduation.py``,
``tests/ui/providers/test_workbench_navigation.py``,
``tests/ui/providers/test_workbench_review_regressions.py`` and
``tests/snapshots/test_ui_snapshot_provider_workbench.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import httpx
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
    DiscoveryItem,
    DiscoveryResult,
    ModelEdits,
    OptionalEdit,
    ProviderDraft,
    ProviderWorkbenchResult,
    TLSConfig,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot, merge_catalog_overlay
from chartreux.core.model_catalog.presets import FULLY_CUSTOM, MISTRAL, ProviderPreset
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
from tests.snapshots.snapshot_event_loop import install_snapshot_wake


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


async def expand(pilot, screen: ProviderWorkbenchScreen) -> None:  # type: ignore[no-untyped-def]
    browser = screen.query_one("#wb-providers", OptionList)
    await press_option(pilot, browser, "one")
    assert screen.state is not None and screen.state.provider_id == "one"


async def wait_until(pilot, predicate) -> None:  # type: ignore[no-untyped-def]
    for _ in range(100):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("condition was not reached within 1 second")


async def press_option(pilot, widget: OptionList, option_id: str) -> None:
    """Enter the requested visible group, then select from its bounded list."""
    for _ in range(5):
        if widget.has_focus:
            break
        await pilot.press("tab")
    if not widget.has_focus:
        raise AssertionError(f"group {widget.id!r} was not reachable")
    await pilot.press("home")
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


async def add_preset(
    pilot, screen: ProviderWorkbenchScreen, preset: ProviderPreset
) -> None:  # type: ignore[no-untyped-def]
    if preset is MISTRAL:
        screen._choose_preset(MISTRAL)
    elif preset is FULLY_CUSTOM:
        await press_option(
            pilot, screen.query_one("#wb-root-actions", OptionList), "\x00add"
        )
    else:
        screen._choose_preset(preset)
    assert screen._stage == "connection"
    await pilot.pause()


async def open_split_view(pilot, screen, view: str) -> None:
    await expand(pilot, screen)
    assert screen.state
    if view == "catalog":
        screen.state.set_edits("a", ModelEdits(input_price=OptionalEdit.set(2)))
        screen._open_catalog()
    elif view == "detail":
        screen._select_action("models")
        screen._open_detail("a")
        screen.query_one("#price-input", Input).value = "2"
    else:
        screen._open_presets()
        if view == "preset-editor":
            await press_option(
                pilot,
                screen.query_one("#wb-presets", OptionList),
                "preset:orchestrator",
            )
            screen._preset_draft = ("a", "off")
        else:
            screen.state.set_role_preset("orchestrator", "a", "off")
    await pilot.pause()
