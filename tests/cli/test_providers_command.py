from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest

from chartreux.app_server._projection import project_config_view
from chartreux.cli.commands import CommandRegistry
from chartreux.cli.textual_ui.app import (
    ChartreuxApp,
    _ProviderConfigService,
    _ProviderCredentials,
)
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.model_catalog.loader import CatalogSnapshot, CatalogStore
from chartreux.core.model_catalog.schema import ModelCatalog
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


def test_provider_validation_uses_projected_expression_and_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "a/default": {"api_base": "https://a.test/v1"},
                "b/default": {
                    "api_base": "https://b.test/v1",
                    "api_key_env_var": "B_KEY",
                },
            },
            "models": {
                "a": {"deployments": [{"provider": "a/default", "name": "a"}]},
                "b": {"deployments": [{"provider": "b/default", "name": "b"}]},
            },
            "roles": {"preferred": {"models": ["a", "b"]}},
        }),
        "initial",
    )
    allowed_schema = ChartreuxConfigSchema(
        active_model="a", allowed_models=["a"]
    ).attach_catalog_snapshot(initial)
    expression_schema = ChartreuxConfigSchema(
        active_model="@preferred"
    ).attach_catalog_snapshot(initial)
    reordered = CatalogSnapshot(
        ModelCatalog.model_validate({
            **initial.catalog.model_dump(),
            "roles": {"preferred": {"models": ("b", "a")}},
        }),
        "reordered",
    )
    monkeypatch.delenv("B_KEY", raising=False)
    monkeypatch.setattr(
        "chartreux.core.model_catalog.loader.load_catalog", lambda: reordered
    )
    allowed_app = cast(
        ChartreuxApp, SimpleNamespace(config=project_config_view(allowed_schema))
    )
    expression_app = cast(
        ChartreuxApp, SimpleNamespace(config=project_config_view(expression_schema))
    )
    allowed_service = _ProviderConfigService(allowed_app)
    expression_service = _ProviderConfigService(expression_app)

    assert "allowed_models" in (allowed_service.validate_active_selection("b") or "")
    assert allowed_service.validate_active_selection("a") is None
    assert expression_service.validate_active_selection(None) is not None


def test_provider_validation_accepts_unpinned_default_from_config_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"glm/default": {"api_base": "https://glm.test/v1"}},
            "models": {
                "glm-5-2": {
                    "deployments": [{"provider": "glm/default", "name": "glm-5-2"}]
                }
            },
            "roles": {"orchestrator": {"models": ["glm-5-2"]}},
        }),
        "default",
    )
    schema = ChartreuxConfigSchema(active_model="").attach_catalog_snapshot(catalog)
    monkeypatch.setattr(
        "chartreux.core.model_catalog.loader.load_catalog", lambda: catalog
    )
    app = cast(ChartreuxApp, SimpleNamespace(config=project_config_view(schema)))

    assert _ProviderConfigService(app).validate_active_selection(None) is None
