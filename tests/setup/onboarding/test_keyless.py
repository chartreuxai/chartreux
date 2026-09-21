from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.setup.auth.api_key_persistence import resolve_api_key_provider
from chartreux.setup.onboarding import OnboardingApp
from chartreux.setup.onboarding.context import OnboardingContext
from chartreux.setup.onboarding.screens.theme_selection import ThemeSelectionScreen
from chartreux.ui.providers.contracts import ConfigPersistResult
from tests.conftest import build_test_vibe_config


def _keyless_provider() -> ProviderConfig:
    return ProviderConfig(
        name="local", api_base="http://localhost:11434/v1", api_key_env_var=""
    )


def _keyless_context() -> OnboardingContext:
    provider = _keyless_provider()
    config = build_test_vibe_config(
        active_model="local",
        providers=[provider],
        models={"local": ModelConfig(name="local", provider="local", alias="local")},
    )
    return OnboardingContext.from_config(config)


def test_resolve_api_key_provider_preserves_keyless_selection() -> None:
    provider = _keyless_provider()
    assert resolve_api_key_provider(provider) is provider


def test_keyless_app_installs_shared_flow_not_legacy_api_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = OnboardingApp(config=_keyless_context())
    installed: dict[str, object] = {}
    pushed: list[str] = []
    monkeypatch.setattr(
        app, "install_screen", lambda screen, name: installed.__setitem__(name, screen)
    )
    monkeypatch.setattr(app, "push_screen", pushed.append)

    app.on_mount()

    assert set(installed) == {"welcome", "theme_selection"}
    assert pushed == ["welcome"]


@pytest.mark.asyncio
async def test_keyless_theme_step_starts_shared_provider_flow() -> None:
    config = AsyncMock()
    config.persist_theme.return_value = ConfigPersistResult(True)
    app = OnboardingApp(config=_keyless_context(), config_service=config)

    async with app.run_test() as pilot:
        screen = pilot.app.get_screen("theme_selection")
        assert isinstance(screen, ThemeSelectionScreen)
        screen.action_next()
        await pilot.pause()
        config.persist_theme.assert_awaited_once_with("auto")
