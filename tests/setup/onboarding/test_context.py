from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._web_search_settings import project_web_search_settings
from chartreux.app_server.protocol import SettingsReadResponse
from chartreux.cli.textual_ui.screens.web_search import WebSearchScreen
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.setup.onboarding import (
    OnboardingApp,
    OnboardingCredentialService,
    OnboardingFailure,
    run_onboarding,
)
from chartreux.setup.onboarding.context import OnboardingContext
from chartreux.ui.providers.contracts import ConfigReloadResult, ProviderWorkbenchResult
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
from tests.cli.textual_ui.web_search_fixture import make_snapshot
from tests.conftest import build_test_vibe_config


def test_from_config_keeps_only_initial_provider_input() -> None:
    context = OnboardingContext.from_config(build_test_vibe_config())
    assert context.provider is not None
    assert context.provider.name == "mistral"
    assert not hasattr(context, "models")


@pytest.mark.parametrize("theme", ["auto", "light", "dark"])
def test_context_preserves_configured_theme(theme: str) -> None:
    context = OnboardingContext.from_config(build_test_vibe_config(theme=theme))
    assert context.theme == theme


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
    assert set(installed) == {"welcome"}
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
    assert app._credentials.provider.name == "onboarding"


@pytest.mark.asyncio
async def test_reload_failure_returns_retry_guidance_instead_of_success() -> None:
    config = AsyncMock()
    config.reload_catalog_and_config.return_value = ConfigReloadResult(
        None, "synthetic reload failure"
    )
    app = OnboardingApp(config=build_test_vibe_config(), config_service=config)
    app.push_screen_wait = AsyncMock(
        return_value=ProviderWorkbenchResult("completed", changed=True)
    )
    exits: list[ProviderWorkbenchResult | OnboardingFailure | None] = []
    app.exit = exits.append  # type: ignore[method-assign]

    await app._run_workbench()

    config.reload_catalog_and_config.assert_awaited_once_with()
    assert exits == [
        OnboardingFailure(
            "Could not apply the provider changes: synthetic reload failure. "
            "Retry setup to adopt the saved changes."
        )
    ]


@pytest.mark.asyncio
async def test_host_opens_search_after_presets_and_back_reopens_saved_presets() -> None:
    config = AsyncMock()
    search = AsyncMock()
    search.read.return_value = make_snapshot(
        {"provider": "exa"}, readiness="missing_key"
    )
    app = OnboardingApp(config=build_test_vibe_config(), config_service=config)
    app._search_service = AsyncMock(return_value=search)  # type: ignore[method-assign]
    app.push_screen_wait = AsyncMock(
        side_effect=[
            ProviderWorkbenchResult("completed", changed=True),
            "back",
            ProviderWorkbenchResult("completed"),
            "finish",
        ]
    )
    exits: list[ProviderWorkbenchResult | OnboardingFailure | None] = []
    app.exit = exits.append  # type: ignore[method-assign]

    await app._run_workbench()

    screens = [call.args[0] for call in app.push_screen_wait.await_args_list]
    assert isinstance(screens[0], ProviderWorkbenchScreen)
    assert screens[0].initial_view == "providers"
    assert isinstance(screens[1], WebSearchScreen)
    assert screens[1].mode == "onboarding"
    assert isinstance(screens[2], ProviderWorkbenchScreen)
    assert screens[2].initial_view == "presets"
    assert isinstance(screens[3], WebSearchScreen)
    assert exits == [ProviderWorkbenchResult("completed", changed=True)]
    config.reload_catalog_and_config.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_host_uses_configured_custom_mistral_search_without_prompt_or_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOM_MISTRAL_KEY", "available")
    mistral_config = build_test_vibe_config(
        active_model="mock",
        providers=[
            {
                "name": "mistral-search",
                "api_base": "https://custom-mistral.example/v1",
                "api_key_env_var": "CUSTOM_MISTRAL_KEY",
                "backend": "mistral",
            }
        ],
        models=[ModelConfig(name="mock", provider="mistral-search", alias="mock")],
    )
    projected = project_web_search_settings(
        mistral_config, [], user_layer="user", user_unavailable=False, fallback=False
    )
    assert projected.readiness == "ready"
    snapshot = SettingsReadResponse(
        fields=[], web_search=projected, user_layer="user", user_revision="revision"
    )
    service = AsyncMock()
    service.read.return_value = snapshot
    config_service = AsyncMock()
    app = OnboardingApp(config=mistral_config, config_service=config_service)
    app._search_service = AsyncMock(return_value=service)  # type: ignore[method-assign]
    app.push_screen_wait = AsyncMock(return_value=ProviderWorkbenchResult("completed"))
    exits: list[ProviderWorkbenchResult | OnboardingFailure | None] = []
    app.exit = exits.append  # type: ignore[method-assign]

    await app._run_workbench()

    assert app.push_screen_wait.await_count == 1
    assert exits == [ProviderWorkbenchResult("completed")]
    service.save.assert_not_awaited()
    config_service.reload_catalog_and_config.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["exa", "brave", "duckduckgo"])
async def test_host_preserves_ready_explicit_search_provider(provider: str) -> None:
    search = AsyncMock()
    search.read.return_value = make_snapshot({"provider": provider}, readiness="ready")
    app = OnboardingApp(config=build_test_vibe_config(), config_service=AsyncMock())
    app._search_service = AsyncMock(return_value=search)  # type: ignore[method-assign]
    app.push_screen_wait = AsyncMock(return_value=ProviderWorkbenchResult("completed"))
    exits: list[ProviderWorkbenchResult | OnboardingFailure | None] = []
    app.exit = exits.append  # type: ignore[method-assign]

    await app._run_workbench()

    assert app.push_screen_wait.await_count == 1
    assert exits == [ProviderWorkbenchResult("completed")]
    search.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_unready_explicit_search_is_not_bypassed_by_mistral_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "available")
    search = AsyncMock()
    search.read.return_value = make_snapshot(
        {"provider": "exa"}, readiness="missing_key"
    )
    app = OnboardingApp(config=build_test_vibe_config(), config_service=AsyncMock())
    app._search_service = AsyncMock(return_value=search)  # type: ignore[method-assign]
    app.push_screen_wait = AsyncMock(
        side_effect=[ProviderWorkbenchResult("completed"), "skip"]
    )
    exits: list[ProviderWorkbenchResult | OnboardingFailure | None] = []
    app.exit = exits.append  # type: ignore[method-assign]

    await app._run_workbench()

    screens = [call.args[0] for call in app.push_screen_wait.await_args_list]
    assert len(screens) == 2
    assert isinstance(screens[1], WebSearchScreen)
    assert exits == [ProviderWorkbenchResult("completed")]


@pytest.mark.asyncio
@pytest.mark.parametrize("view_only", [True, False])
async def test_host_does_not_skip_search_on_unverifiable_ready_projection(
    view_only: bool,
) -> None:
    snapshot = make_snapshot(readiness="ready", revision=None if view_only else "rev")
    if not view_only:
        assert snapshot.web_search is not None
        snapshot = snapshot.model_copy(
            update={
                "web_search": snapshot.web_search.model_copy(
                    update={
                        "fields": [
                            field.model_copy(update={"origin": "live config"})
                            for field in snapshot.web_search.fields
                        ]
                    }
                )
            }
        )
    search = AsyncMock()
    search.read.return_value = snapshot
    app = OnboardingApp(config=build_test_vibe_config(), config_service=AsyncMock())
    app._search_service = AsyncMock(return_value=search)  # type: ignore[method-assign]
    app.push_screen_wait = AsyncMock(
        side_effect=[ProviderWorkbenchResult("completed"), "skip"]
    )
    exits: list[ProviderWorkbenchResult | OnboardingFailure | None] = []
    app.exit = exits.append  # type: ignore[method-assign]

    await app._run_workbench()

    assert app.push_screen_wait.await_count == 2
    assert isinstance(app.push_screen_wait.await_args_list[1].args[0], WebSearchScreen)
    assert exits == [ProviderWorkbenchResult("completed")]


@pytest.mark.asyncio
async def test_host_read_failure_cannot_auto_complete_search() -> None:
    search = AsyncMock()
    search.read.side_effect = OSError("synthetic read failure")
    app = OnboardingApp(config=build_test_vibe_config(), config_service=AsyncMock())
    app._search_service = AsyncMock(return_value=search)  # type: ignore[method-assign]
    app.push_screen_wait = AsyncMock(return_value=ProviderWorkbenchResult("completed"))
    exits: list[ProviderWorkbenchResult | OnboardingFailure | None] = []
    app.exit = exits.append  # type: ignore[method-assign]

    await app._run_workbench()

    assert app.push_screen_wait.await_count == 1
    assert exits == [
        OnboardingFailure("Could not open Web search setup: synthetic read failure")
    ]


@pytest.mark.asyncio
async def test_existing_theme_is_applied_without_onboarding_persistence() -> None:
    config = AsyncMock()
    app = OnboardingApp(
        config=build_test_vibe_config(theme="light"), config_service=config
    )
    app.push_screen_wait = AsyncMock(return_value=ProviderWorkbenchResult("cancelled"))
    app.exit = lambda result: None  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        assert app.theme == "ansi-light"
        await pilot.press("enter")
        await pilot.press("enter")
        await pilot.pause(0.1)

    workbench_call = app.push_screen_wait.await_args
    assert workbench_call is not None
    workbench = workbench_call.args[0]
    assert isinstance(workbench, ProviderWorkbenchScreen)
    assert workbench.mode == "onboarding"
    assert workbench.initial_view == "providers"
    config.persist_theme.assert_not_awaited()


@pytest.mark.parametrize(
    ("result", "expected", "status"),
    [
        (OnboardingFailure("disk unavailable"), "disk unavailable", 1),
        (
            ProviderWorkbenchResult("cancelled", changed=True),
            "Setup closed. Saved changes were kept.",
            0,
        ),
    ],
)
def test_onboarding_result_messages_are_honest(
    result: OnboardingFailure | ProviderWorkbenchResult,
    expected: str,
    status: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class ResultApp:
        def run(self) -> OnboardingFailure | ProviderWorkbenchResult:
            return result

    with pytest.raises(SystemExit) as error:
        run_onboarding(app=ResultApp(), orchestrator=AsyncMock())  # type: ignore[arg-type]

    assert error.value.code == status
    assert expected in capsys.readouterr().out


def test_unpinned_onboarding_context_resolves_orchestrator_role() -> None:
    config = build_test_vibe_config(active_model="")

    assert (
        OnboardingContext.from_config(config).provider == config.get_active_provider()
    )
    assert config.get_active_model().name == "zai-glm-5-3"
