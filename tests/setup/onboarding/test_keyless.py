from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.setup.auth.api_key_persistence import resolve_api_key_provider
from chartreux.setup.onboarding import OnboardingApp, OnboardingCredentialService
from chartreux.setup.onboarding.context import OnboardingContext
from chartreux.ui.providers.contracts import ProviderWorkbenchResult
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
from tests.conftest import build_test_vibe_config
from tests.snapshots.snapshot_event_loop import install_snapshot_wake


@pytest_asyncio.fixture(autouse=True)
async def _snapshot_event_loop_wake() -> None:
    install_snapshot_wake()


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

    assert set(installed) == {"welcome"}
    assert pushed == ["welcome"]


@pytest.mark.asyncio
async def test_keyless_welcome_starts_shared_provider_flow() -> None:
    config = AsyncMock()
    app = OnboardingApp(config=_keyless_context(), config_service=config)
    assert isinstance(app._credentials, OnboardingCredentialService)
    assert app._credentials.provider.name == "local"
    captured: list[ProviderWorkbenchScreen] = []
    app.push_screen_wait = AsyncMock(
        side_effect=lambda screen: (
            captured.append(screen) or ProviderWorkbenchResult("cancelled")
        )
    )
    app.exit = lambda result: None  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.press("enter")
        await pilot.pause(0.1)

    assert len(captured) == 1
    screen = captured[0]
    assert screen.mode == "onboarding"
    assert screen.initial_view == "providers"
    assert screen.credentials is app._credentials
    assert app._credentials.provider.api_key_env_var == ""
    config.persist_theme.assert_not_awaited()
