from __future__ import annotations

from collections.abc import Callable

import pytest

from chartreux.cli.commands import CommandRegistry
from chartreux.cli.textual_ui.app import _ProviderConfigService, _ProviderCredentials
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from chartreux.core.model_catalog.loader import CatalogStore
from chartreux.ui.providers.flow import ProviderManagementScreen
from tests.conftest import build_test_chartreux_app


async def _wait_until(pilot, predicate: Callable[[], bool], *, tries: int = 50) -> bool:
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return True
    return predicate()


def test_providers_command_is_registered_and_listed() -> None:
    registry = CommandRegistry()

    assert registry.parse_command("/providers") == (
        "providers",
        registry.commands["providers"],
        "",
    )
    assert registry.commands["providers"].handler == "_show_providers"
    assert "/providers" in registry.get_help_text()


@pytest.mark.asyncio
async def test_providers_rejects_an_active_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    monkeypatch.setattr(app, "_is_busy", lambda: True)

    async with app.run_test():
        assert await app._handle_command("/providers")
        assert any(
            "only available while no turn is active" in str(message._error)
            for message in app.query(ErrorMessage)
        )
        assert not app.query(ProviderManagementScreen)


@pytest.mark.asyncio
async def test_providers_mounts_a_fresh_screen_with_real_services() -> None:
    app = build_test_chartreux_app()

    async with app.run_test() as pilot:
        assert await app._handle_command("/providers")
        assert await _wait_until(
            pilot, lambda: isinstance(app.screen, ProviderManagementScreen)
        )
        screen = app.screen
        assert isinstance(screen, ProviderManagementScreen)
        assert isinstance(screen.catalog_writer, CatalogStore)
        assert isinstance(screen.credentials, _ProviderCredentials)
        assert isinstance(screen.config, _ProviderConfigService)
        screen.action_cancel()
        assert await _wait_until(
            pilot,
            lambda: any(
                "Provider management closed." in message._content
                for message in app.query(UserCommandMessage)
            ),
        )
