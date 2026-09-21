from __future__ import annotations

import tomllib
from unittest.mock import AsyncMock

import pytest

from chartreux.core.config import build_default_orchestrator
from chartreux.core.utils.concurrency import run_sync
from chartreux.setup.onboarding import (
    OnboardingApp,
    OnboardingConfigService,
    OnboardingFailure,
    run_onboarding,
)
from chartreux.setup.onboarding.context import OnboardingContext
from chartreux.ui.providers.contracts import (
    ConfigPersistResult,
    ConfigReloadResult,
    ProviderFlowResult,
)
from chartreux.ui.providers.flow import ProviderManagementScreen
from tests.conftest import build_test_vibe_config


def test_from_config_keeps_only_initial_provider_input() -> None:
    context = OnboardingContext.from_config(build_test_vibe_config())
    assert context.provider.name == "mistral/default"
    assert not hasattr(context, "models")


def test_context_preserves_initial_theme() -> None:
    context = OnboardingContext.from_config(build_test_vibe_config())
    assert context.theme == "auto"


@pytest.mark.asyncio
async def test_theme_confirmation_persists_before_starting_shared_flow() -> None:
    config = AsyncMock()
    config.persist_theme.return_value = ConfigPersistResult(True)
    app = OnboardingApp(config=build_test_vibe_config(), config_service=config)
    pushed: list[ProviderManagementScreen] = []
    app.push_screen_wait = AsyncMock(
        side_effect=lambda screen: (
            pushed.append(screen) or ProviderFlowResult("cancelled")
        )
    )
    exits: list[ProviderFlowResult | None] = []
    app.exit = exits.append  # type: ignore[method-assign]

    await app._persist_theme_then_run_flow("auto")

    config.persist_theme.assert_awaited_once_with("auto")
    assert len(pushed) == 1
    assert exits == [ProviderFlowResult("cancelled")]


def test_onboarding_selected_model_writes_selections_only_config(config_dir) -> None:
    config_file = config_dir / "config.toml"
    config_file.write_text("", encoding="utf-8")
    orchestrator = run_sync(build_default_orchestrator())

    class SelectedModelApp:
        def run(self) -> str:
            return "glm-5-2"

    run_onboarding(app=SelectedModelApp(), orchestrator=orchestrator)  # type: ignore[arg-type]

    with config_file.open("rb") as stream:
        assert tomllib.load(stream) == {"active_model": "glm-5-2"}
    assert not (config_dir / "models.toml").exists()


@pytest.mark.asyncio
async def test_reload_failure_returns_retry_guidance_instead_of_success() -> None:
    config = AsyncMock()
    config.persist_theme.return_value = ConfigPersistResult(True)
    config.reload_catalog_and_config.return_value = ConfigReloadResult(
        None, "synthetic reload failure"
    )
    app = OnboardingApp(config=build_test_vibe_config(), config_service=config)
    app.push_screen_wait = AsyncMock(
        return_value=ProviderFlowResult("completed", "new-model", changed=True)
    )
    exits: list[ProviderFlowResult | OnboardingFailure | None] = []
    app.exit = exits.append  # type: ignore[method-assign]

    await app._persist_theme_then_run_flow("auto")

    config.reload_catalog_and_config.assert_awaited_once_with()
    assert exits == [
        OnboardingFailure(
            "Could not apply the provider changes: synthetic reload failure. "
            "Retry setup to adopt the saved changes."
        )
    ]


@pytest.mark.asyncio
async def test_config_service_persists_literal_auto_theme() -> None:
    orchestrator = AsyncMock()
    service = OnboardingConfigService(orchestrator)

    assert (await service.persist_theme("auto")).persisted
    orchestrator.set_field.assert_awaited_once_with(
        "/theme", "auto", reason="onboarding theme selection", target_layer="user-toml"
    )


@pytest.mark.asyncio
async def test_config_service_reports_set_field_failures() -> None:
    orchestrator = AsyncMock()
    orchestrator.set_field.return_value = ["disk unavailable"]
    service = OnboardingConfigService(orchestrator)

    result = await service.persist_active_model("example")

    assert not result.persisted
    assert result.message == "disk unavailable"
