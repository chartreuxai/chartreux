from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.cli.textual_ui.app import ChartreuxApp


class _ProviderCredentials:
    """Adapt provider credential persistence to the host-neutral flow contract."""

    def resolve_key(self, env_var: str) -> str | None:
        resolve_api_key = import_module("chartreux.utils.api_keys").resolve_api_key
        return resolve_api_key(env_var)

    def save_key(self, env_var: str, key: str) -> Any:
        outcome = persist_api_key_for_provider(env_var, key)
        credential_save_result = import_module(
            "chartreux.core.model_catalog.contracts"
        ).CredentialSaveResult
        if outcome == "completed":
            return credential_save_result("saved")
        if outcome.startswith("env_var_error"):
            return credential_save_result(
                "invalid_env_var", "Credential variable is invalid."
            )
        return credential_save_result(
            "session_only",
            "Could not save the key to disk; it remains available for this session.",
        )


class _ProviderConfigService:
    """Reload and persist provider settings without resetting the conversation UI."""

    def __init__(self, app: ChartreuxApp) -> None:
        self._app = app

    async def persist_theme(self, theme: str) -> Any:
        config_persist_result = import_module(
            "chartreux.core.model_catalog.contracts"
        ).ConfigPersistResult
        try:
            await self._app.app_server.resources.config.update({"theme": theme})
            await self._app._apply_config_to_ui()
        except Exception:
            logger.warning("Provider theme persistence failed", exc_info=True)
            return config_persist_result(False, "Could not save the theme.")
        return config_persist_result(True)

    async def reload_catalog_and_config(self) -> Any:
        contracts = import_module("chartreux.core.model_catalog.contracts")
        config_reload_result = contracts.ConfigReloadResult
        load_catalog = import_module("chartreux.core.model_catalog.loader").load_catalog
        try:
            await self._app.app_server.resources.config.reload(reload_runtime=True)
            snapshot = load_catalog()
            await self._app._apply_config_to_ui()
        except Exception:
            logger.warning("Provider catalog reload failed", exc_info=True)
            return config_reload_result(
                None, "Could not reload the provider catalog. Retry."
            )
        return config_reload_result(snapshot)

    async def persist_active_model(self, expression: str) -> Any:
        config_persist_result = import_module(
            "chartreux.core.model_catalog.contracts"
        ).ConfigPersistResult
        try:
            await self._app.app_server.resources.config.update({
                "active_model": expression
            })
            await self._app._apply_config_to_ui()
        except Exception:
            logger.warning("Provider active-model persistence failed", exc_info=True)
            return config_persist_result(False, "Could not save the active model.")
        return config_persist_result(True)

    def validate_active_selection(self, expression: str | None) -> str | None:
        """Return an actionable error unless the selection is usable at runtime."""
        try:
            load_catalog = import_module(
                "chartreux.core.model_catalog.loader"
            ).load_catalog
            model_resolver = import_module(
                "chartreux.core.model_catalog.resolver"
            ).ModelResolver
            resolve_api_key = import_module("chartreux.utils.api_keys").resolve_api_key
            missing_api_key_error = import_module(
                "chartreux.core.config.models"
            ).MissingAPIKeyError
            selected = (
                expression
                if expression is not None
                else self._app.config.active_model_expression
            ) or self._app.config.default_model_alias
            resolved = model_resolver(load_catalog()).resolve(
                selected, allowed_models=self._app.config.allowed_models
            )
            if resolved.provider.api_key_env_var and not resolve_api_key(
                resolved.provider.api_key_env_var
            ):
                raise missing_api_key_error(
                    resolved.provider.api_key_env_var, resolved.deployment.provider
                )
        except (RuntimeError, ValueError) as error:
            return str(error)
        return None


def persist_api_key_for_provider(env_var: str, key: str) -> str:
    """Persist an explicit provider key while retaining session-only fallback."""
    provider_config = import_module("chartreux.core.config").ProviderConfig
    persist_api_key = import_module(
        "chartreux.setup.auth.api_key_persistence"
    ).persist_api_key

    return persist_api_key(
        provider_config(
            name="provider", api_base="https://localhost", api_key_env_var=env_var
        ),
        key,
    )
