from __future__ import annotations

import tomllib
from unittest.mock import AsyncMock

import pytest

from chartreux.core.config import ChartreuxConfigSchema, build_default_orchestrator
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.utils.concurrency import run_sync
from chartreux.setup.onboarding import (
    OnboardingApp,
    OnboardingConfigService,
    OnboardingCredentialService,
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
    assert context.provider is not None
    assert context.provider.name == "mistral/default"
    assert not hasattr(context, "models")


def test_context_preserves_initial_theme() -> None:
    context = OnboardingContext.from_config(build_test_vibe_config())
    assert context.theme == "auto"


def _disabled_active_model_config() -> ChartreuxConfigSchema:
    config = build_test_vibe_config(active_model="glm-5-3")
    catalog = config.catalog_snapshot.catalog
    disabled = catalog.models["glm-5-3"].model_copy(update={"disabled": True})
    config.attach_catalog_snapshot(
        CatalogSnapshot(
            catalog.model_copy(
                update={"models": {**catalog.models, "glm-5-3": disabled}}
            ),
            "disabled-active-model",
        )
    )
    return config


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(build_test_vibe_config(active_model="missing"), id="missing"),
        pytest.param(_disabled_active_model_config(), id="disabled"),
        pytest.param(
            build_test_vibe_config(active_model="glm-5-3", allowed_models=["not-glm"]),
            id="excluded",
        ),
    ],
)
def test_unresolvable_active_model_opens_onboarding_repair_entry(
    config: ChartreuxConfigSchema, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = OnboardingContext.from_config(config)
    app = OnboardingApp(config=context)
    installed: dict[str, object] = {}
    pushed: list[str] = []
    monkeypatch.setattr(
        app, "install_screen", lambda screen, name: installed.__setitem__(name, screen)
    )
    monkeypatch.setattr(app, "push_screen", pushed.append)

    app.on_mount()

    assert context.provider is not None
    assert set(installed) == {"welcome", "theme_selection"}
    assert pushed == ["welcome"]


def test_context_uses_active_provider_when_it_resolves() -> None:
    config = build_test_vibe_config()
    assert (
        OnboardingContext.from_config(config).provider == config.get_active_provider()
    )


def test_credential_adapter_uses_safe_repair_provider() -> None:
    config = build_test_vibe_config(active_model="glm-5-3")
    catalog = config.catalog_snapshot.catalog
    config.attach_catalog_snapshot(
        CatalogSnapshot(
            catalog.model_copy(
                update={
                    "providers": {
                        provider_id: provider.model_copy(update={"disabled": True})
                        for provider_id, provider in catalog.providers.items()
                    }
                }
            ),
            "no-repair-provider",
        )
    )

    app = OnboardingApp(config=config)

    assert isinstance(app._credentials, OnboardingCredentialService)
    assert app._credentials.provider.name == "onboarding/default"


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
    assert exits == [ProviderFlowResult("cancelled", changed=True)]


def test_onboarding_selected_model_writes_selections_only_config(config_dir) -> None:
    config_file = config_dir / "config.toml"
    config_file.write_text("", encoding="utf-8")
    orchestrator = run_sync(build_default_orchestrator())

    class SelectedModelApp:
        def run(self) -> str:
            return "glm-5-3"

    run_onboarding(app=SelectedModelApp(), orchestrator=orchestrator)  # type: ignore[arg-type]

    with config_file.open("rb") as stream:
        assert tomllib.load(stream) == {"active_model": "glm-5-3"}
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
async def test_theme_save_failure_returns_its_error() -> None:
    config = AsyncMock()
    config.persist_theme.return_value = ConfigPersistResult(False, "disk unavailable")
    app = OnboardingApp(config=build_test_vibe_config(), config_service=config)
    exits: list[OnboardingFailure] = []
    app.exit = exits.append  # type: ignore[method-assign]

    await app._persist_theme_then_run_flow("auto")

    assert exits == [
        OnboardingFailure("Could not save the selected theme: disk unavailable")
    ]


@pytest.mark.parametrize(
    ("result", "expected", "status"),
    [
        (OnboardingFailure("disk unavailable"), "disk unavailable", 1),
        (
            ProviderFlowResult("cancelled", changed=True),
            "Setup closed. Saved changes were kept.",
            0,
        ),
    ],
)
def test_onboarding_result_messages_are_honest(
    result: OnboardingFailure | ProviderFlowResult,
    expected: str,
    status: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class ResultApp:
        def run(self) -> OnboardingFailure | ProviderFlowResult:
            return result

    with pytest.raises(SystemExit) as error:
        run_onboarding(app=ResultApp(), orchestrator=AsyncMock())  # type: ignore[arg-type]

    assert error.value.code == status
    assert expected in capsys.readouterr().out


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


def test_unpinned_onboarding_context_resolves_orchestrator_role() -> None:
    config = build_test_vibe_config(active_model="")

    assert (
        OnboardingContext.from_config(config).provider == config.get_active_provider()
    )
    assert config.get_active_model().name == "zai-glm-5-3"
