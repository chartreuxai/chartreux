from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from chartreux.setup.onboarding import OnboardingApp
from chartreux.setup.onboarding.screens.theme_selection import (
    THEME_EXPLANATIONS,
    ThemeSelectionScreen,
)
from chartreux.setup.onboarding.screens.welcome import WelcomeScreen
from chartreux.ui.providers.contracts import ModelSelectionDraft
from tests.conftest import build_test_vibe_config
from tests.ui.providers.test_flow import make_flow, wait_for


@pytest.mark.asyncio
async def test_enter_completes_welcome_animation_before_advancing() -> None:
    app = OnboardingApp(config=build_test_vibe_config())

    async with app.run_test() as pilot:
        welcome = app.get_screen("welcome")
        assert isinstance(welcome, WelcomeScreen)

        await pilot.press("enter")

        assert welcome._typing_done
        assert welcome._prompt_visible
        assert not welcome.query_one("#enter-hint", Static).has_class("hidden")

        await pilot.press("enter")
        assert app.screen is app.get_screen("theme_selection")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (60, 24), (40, 24)])
async def test_theme_screen_explains_choices_and_adapts_to_narrow_terminals(
    size: tuple[int, int],
) -> None:
    app = OnboardingApp(config=build_test_vibe_config())

    async with app.run_test(size=size) as pilot:
        app.switch_screen("theme_selection")
        await pilot.pause()

        screen = app.get_screen("theme_selection")
        assert isinstance(screen, ThemeSelectionScreen)
        explanations = screen.query_one("#theme-explanations", Static).content
        assert str(explanations) == "\n".join(THEME_EXPLANATIONS)

        row = screen.query_one("#theme-row")
        preview = screen.query_one("#preview")
        assert row.region.width <= size[0]
        assert preview.region.width <= size[0]
        assert row.has_class("narrow") is (size[0] < 62)

        await pilot.press("down")
        assert screen.selected_theme == "light"


@pytest.mark.asyncio
async def test_provider_review_actions_fit_onboarding_at_80_by_24() -> None:
    app = OnboardingApp(config=build_test_vibe_config())
    flow, _ = make_flow()
    flow._selected_models = {"wire": ModelSelectionDraft("wire", "wire")}
    flow._detail_wire = "wire"

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(flow)
        await wait_for(pilot, lambda: bool(flow.query("#flow-content")))
        flow._show("review")
        await pilot.pause()

        content = flow.query_one("#flow-content")
        actions = flow.query_one("#provider-actions")
        continue_button = flow.query_one("#continue", Button)
        assert content.region.height > 0
        assert actions.region.y + actions.region.height <= app.size.height
        assert (
            continue_button.region.y + continue_button.region.height <= app.size.height
        )
