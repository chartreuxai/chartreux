"""Offline Provider Settings states for the TUI affordance review.

The hosts use the same in-memory catalog and services as the workbench tests.
Each preparation opens a real workbench view; no user configuration is read or
written and discovery never reaches the network.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from textual.app import App
from textual.pilot import Pilot
from textual.widgets import Input, OptionList

from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.presets import FULLY_CUSTOM
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
from tests.ui.providers.test_workbench import Host, Services, snapshot

Prepare = Callable[[Pilot], Awaitable[None]]
ProviderFixture = tuple[str, str, Callable[[], App], Prepare | None, tuple[str, ...]]


def _factory(*, onboarding: bool = False, two_deployments: bool = False) -> App:
    services = Services()
    data = snapshot().catalog.model_dump()
    data["roles"] = {
        role: {"model": "a", "thinking": "medium"}
        for role in ("orchestrator", "large", "medium", "small")
    }
    if two_deployments:
        data["models"]["a"]["deployments"] = [
            *data["models"]["a"]["deployments"],
            {"provider": "two", "name": "other-a"},
        ]
    services.catalog = CatalogSnapshot(
        ModelCatalog.model_validate(data), "review-fixture"
    )
    screen = ProviderWorkbenchScreen(
        snapshot=services.catalog,
        discovery=services.discover,
        catalog_writer=services,
        credentials=services,
        config=services,
        credential_resolver=services.resolve_key,
        mode="onboarding" if onboarding else "management",
    )
    return Host(screen)


def _screen(pilot: Pilot) -> ProviderWorkbenchScreen:
    screen = pilot.app.screen
    assert isinstance(screen, ProviderWorkbenchScreen)
    return screen


async def _choose(pilot: Pilot, widget_id: str, option_id: str) -> None:
    widget = _screen(pilot).query_one(widget_id, OptionList)
    widget.highlighted = next(
        index for index, option in enumerate(widget.options) if option.id == option_id
    )
    widget.focus()
    await pilot.press("enter")
    await pilot.pause()


async def _existing(pilot: Pilot) -> None:
    await _choose(pilot, "#wb-providers", "one")


async def _choose_view(pilot: Pilot) -> None:
    _screen(pilot)._start_add()
    await pilot.pause()


async def _connection(pilot: Pilot) -> None:
    await _choose_view(pilot)
    await _choose(pilot, "#wb-choose", FULLY_CUSTOM.id)


async def _catalog(pilot: Pilot) -> None:
    _screen(pilot)._open_catalog()
    await pilot.pause()


async def _deployment_picker(pilot: Pilot) -> None:
    await _catalog(pilot)
    await _choose(pilot, "#wb-catalog", "model:a")


async def _presets(pilot: Pilot) -> None:
    _screen(pilot)._open_presets()
    await pilot.pause()


async def _preset_editor(pilot: Pilot) -> None:
    await _presets(pilot)
    await _choose(pilot, "#wb-presets", "preset:orchestrator")


async def _preset_model(pilot: Pilot) -> None:
    await _preset_editor(pilot)
    await _choose(pilot, "#wb-preset-editor", "model")


async def _preset_thinking(pilot: Pilot) -> None:
    await _preset_editor(pilot)
    await _choose(pilot, "#wb-preset-editor", "thinking")


async def _protocol(pilot: Pilot) -> None:
    await _existing(pilot)
    await _choose(pilot, "#wb-actions", "style")


async def _inline_editor(pilot: Pilot) -> None:
    await _existing(pilot)
    await _choose(pilot, "#wb-actions", "base")
    _screen(pilot).query_one(
        "#wb-input", Input
    ).value = "https://draft.example.invalid/v1"
    await pilot.pause()


async def _discard_confirmation(pilot: Pilot) -> None:
    await _inline_editor(pilot)
    await pilot.press("enter")
    await pilot.pause()
    await _choose(pilot, "#wb-provider-operations", "discard")


async def _onboarding_model_detail(pilot: Pilot) -> None:
    await _existing(pilot)
    screen = _screen(pilot)
    # Isolate the onboarding models-stage wording with the fake provider catalog.
    screen._stage = "models"
    screen._open_detail("a")
    await pilot.pause()
    assert screen._detail == "a"
    assert "Save and continue to presets persists the draft" in str(
        screen.query_one("#wb-help").render()
    )


def provider_fixtures() -> list[ProviderFixture]:
    standard = _factory
    return [
        (
            "provider-choose",
            "ProviderWorkbenchScreen choose",
            standard,
            _choose_view,
            ("open Add provider",),
        ),
        (
            "provider-connection",
            "ProviderWorkbenchScreen connection",
            standard,
            _connection,
            ("open Add provider", "choose Fully custom"),
        ),
        (
            "provider-catalog-filter",
            "ProviderWorkbenchScreen catalog",
            standard,
            _catalog,
            ("open global catalog",),
        ),
        (
            "provider-deployments-picker",
            "ProviderWorkbenchScreen deployments",
            lambda: _factory(two_deployments=True),
            _deployment_picker,
            ("open global catalog", "open model a with two deployments"),
        ),
        (
            "provider-presets",
            "ProviderWorkbenchScreen presets",
            standard,
            _presets,
            ("open role presets",),
        ),
        (
            "provider-preset-editor",
            "ProviderWorkbenchScreen preset editor",
            standard,
            _preset_editor,
            ("open role presets", "edit Main assistant"),
        ),
        (
            "provider-preset-model-picker",
            "ProviderWorkbenchScreen preset model picker",
            standard,
            _preset_model,
            ("open role presets", "edit Main assistant", "choose Model"),
        ),
        (
            "provider-preset-thinking-picker",
            "ProviderWorkbenchScreen preset thinking picker",
            standard,
            _preset_thinking,
            ("open role presets", "edit Main assistant", "choose Thinking"),
        ),
        (
            "provider-protocol",
            "ProviderWorkbenchScreen protocol picker",
            standard,
            _protocol,
            ("open provider one", "choose API style"),
        ),
        (
            "provider-inline-editor",
            "ProviderWorkbenchScreen inline editor",
            standard,
            _inline_editor,
            ("open provider one", "edit API base without accepting"),
        ),
        (
            "provider-discard-confirm",
            "ProviderWorkbenchScreen discard confirmation",
            standard,
            _discard_confirmation,
            ("open provider one", "edit API base", "request Discard edits"),
        ),
        (
            "provider-onboarding-choose",
            "ProviderWorkbenchScreen onboarding choose",
            lambda: _factory(onboarding=True),
            _choose_view,
            ("open Add provider in onboarding",),
        ),
        (
            "provider-onboarding-connection",
            "ProviderWorkbenchScreen onboarding connection",
            lambda: _factory(onboarding=True),
            _connection,
            ("open Add provider in onboarding", "choose Fully custom"),
        ),
        (
            "provider-onboarding-model-detail",
            "ProviderWorkbenchScreen onboarding model detail",
            lambda: _factory(onboarding=True),
            _onboarding_model_detail,
            (
                "open fake provider one",
                "set onboarding models stage",
                "open model a detail",
            ),
        ),
    ]
