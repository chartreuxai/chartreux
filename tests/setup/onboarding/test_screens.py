from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.widgets import Static

from chartreux.setup.onboarding import OnboardingApp
from chartreux.setup.onboarding.base import OnboardingHost
from chartreux.setup.onboarding.screens.welcome import WelcomeScreen
from tests.conftest import build_test_vibe_config
from tests.ui.providers.test_workbench import Host, setup


@pytest.mark.asyncio
async def test_enter_completes_welcome_animation_then_opens_providers() -> None:
    app = OnboardingApp(config=build_test_vibe_config())
    show_providers = Mock()
    app._host = OnboardingHost(show_providers, Mock())

    async with app.run_test() as pilot:
        welcome = app.get_screen("welcome")
        assert isinstance(welcome, WelcomeScreen)

        await pilot.press("enter")

        assert welcome._typing_done
        assert welcome._prompt_visible
        assert not welcome.query_one("#enter-hint", Static).has_class("hidden")

        await pilot.press("enter")
        show_providers.assert_called_once_with()
        assert app.screen is welcome


@pytest.mark.asyncio
async def test_provider_workbench_is_keyboard_reachable_at_80_by_24() -> None:
    screen, _ = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        assert screen.query_one("#workbench").display
        await pilot.press("enter")
        assert screen.state is not None
        assert screen.query_one("#wb-actions").region.height > 0
        assert screen.query_one("#wb-actions").region.bottom <= screen.app.size.height
