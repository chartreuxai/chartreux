from __future__ import annotations

import asyncio
from dataclasses import dataclass
import sys
from typing import Any

from rich import print as rprint
from textual.app import App

from chartreux.core.config import (
    ChartreuxConfigSchema,
    MissingAPIKeyError,
    ProviderConfig,
)
from chartreux.core.config.default_orchestrator import build_default_orchestrator
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.model_catalog.contracts import (
    ConfigPersistResult,
    ConfigReloadResult,
    CredentialSaveResult,
    ProviderWorkbenchResult,
    TLSConfig,
)
from chartreux.core.model_catalog.discovery import discover_models
from chartreux.core.model_catalog.loader import CatalogStore, load_catalog
from chartreux.core.model_catalog.materialization import (
    MaterializationError,
    materialize_default_catalog,
)
from chartreux.observability.logging import logger
from chartreux.setup.auth.api_key_persistence import persist_api_key
from chartreux.setup.onboarding.base import OnboardingHost
from chartreux.setup.onboarding.context import OnboardingContext
from chartreux.setup.onboarding.screens import WelcomeScreen
from chartreux.ui.providers.graduation import GraduationStore
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
from chartreux.ui.theme import resolve_auto_theme, resolve_theme, resolve_theme_name
from chartreux.utils.api_keys import resolve_api_key

_TEXTUAL_THEME_MAP = {"auto": None, "light": "ansi-light", "dark": "ansi-dark"}


@dataclass(frozen=True)
class OnboardingFailure:
    """A setup failure that must not be reported as successful completion."""

    message: str


class OnboardingCredentialService:
    """Adapt legacy .env persistence to the shared credential boundary."""

    def __init__(self, provider: ProviderConfig) -> None:
        self.provider = provider

    def resolve_key(self, env_var: str) -> str | None:
        return resolve_api_key(env_var)

    def save_key(self, env_var: str, key: str) -> CredentialSaveResult:
        result = persist_api_key(
            self.provider.model_copy(update={"api_key_env_var": env_var}), key
        )
        if result.startswith("env_var_error:"):
            return CredentialSaveResult("invalid_env_var", result)
        if result.startswith("save_error:"):
            return CredentialSaveResult("session_only", result)
        return CredentialSaveResult("saved")


class OnboardingConfigService:
    """User-config adapter used by the onboarding host and shared flow."""

    def __init__(self, orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]) -> None:
        self.orchestrator = orchestrator

    async def persist_theme(self, theme: str) -> ConfigPersistResult:
        try:
            failures = await self.orchestrator.set_field(
                "/theme",
                theme,
                reason="onboarding theme selection",
                target_layer="user-toml",
            )
            if isinstance(failures, list) and failures:
                return ConfigPersistResult(False, str(failures[0]))
        except (OSError, ValueError) as error:
            return ConfigPersistResult(False, str(error))
        return ConfigPersistResult(True)

    async def reload_catalog_and_config(self) -> ConfigReloadResult:
        try:
            await self.orchestrator.reload()
            return ConfigReloadResult(load_catalog())
        except (OSError, ValueError) as error:
            return ConfigReloadResult(None, str(error))


class OnboardingApp(App[ProviderWorkbenchResult | OnboardingFailure | None]):
    CSS_PATH = "onboarding.tcss"

    def __init__(
        self,
        config: OnboardingContext | ChartreuxConfigSchema | None = None,
        *,
        orchestrator: ConfigOrchestrator[ChartreuxConfigSchema] | None = None,
        discovery: Any = discover_models,
        catalog_writer: CatalogStore | None = None,
        credentials: Any | None = None,
        config_service: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            config = OnboardingContext.load()
        elif isinstance(config, ChartreuxConfigSchema):
            config = OnboardingContext.from_config(config)
        self._config = config
        self._orchestrator = orchestrator
        self._config_service = config_service
        self._discovery = discovery
        self._catalog_writer = catalog_writer or CatalogStore()
        self._credentials = credentials or OnboardingCredentialService(config.provider)
        self._graduation = GraduationStore()
        self._initial_theme = resolve_theme_name(config.theme)
        resolve_auto_theme()
        self._resolved_theme = resolve_theme(self._initial_theme)
        self._host = OnboardingHost(self._show_providers, self._cancel)

    def on_mount(self) -> None:
        textual_theme = _TEXTUAL_THEME_MAP[self._resolved_theme]
        if textual_theme is not None:
            self.theme = textual_theme
        self.install_screen(WelcomeScreen(self._host), "welcome")
        self.push_screen("welcome")

    def _show_providers(self) -> None:
        self.run_worker(
            self._run_workbench(), exclusive=True, name="onboarding-provider-workbench"
        )

    def _cancel(self) -> None:
        self.exit(None)

    async def _services(self) -> Any:
        if self._config_service is not None:
            return self._config_service
        if self._orchestrator is None:
            self._orchestrator = await build_default_orchestrator()
        self._config_service = OnboardingConfigService(self._orchestrator)
        return self._config_service

    def _graduation_models_saved(
        self,
        roster_before: frozenset[str],
        roster_after: frozenset[str],
        saved_models: frozenset[str],
    ) -> None:
        self._graduation.state.model_saved(roster_before, roster_after, saved_models)
        self._graduation.save()

    async def _run_workbench(self) -> None:
        config_service = await self._services()
        initial_view = "presets" if self._config.repair_default_preset else "providers"
        result = await self.push_screen_wait(
            ProviderWorkbenchScreen(
                discovery=self._discovery,
                catalog_writer=self._catalog_writer,
                credentials=self._credentials,
                credential_resolver=self._credentials.resolve_key,
                config=config_service,
                snapshot=load_catalog(),
                mode="onboarding",
                initial_view=initial_view,
                allowed_models=self._config.allowed_models,
                thinking_overrides=self._config.thinking_overrides,
                selected_model=self._config.selected_model,
                on_models_saved=self._graduation_models_saved,
                tls=TLSConfig(
                    enable_system_trust_store=self._config.enable_system_trust_store
                ),
            )
        )
        if result.changed:
            reloaded = await config_service.reload_catalog_and_config()
            if reloaded.snapshot is None:
                self.exit(
                    OnboardingFailure(
                        "Could not apply the provider changes: "
                        f"{reloaded.message or 'catalog reload failed'}. "
                        "Retry setup to adopt the saved changes."
                    )
                )
                return
        if result.status != "completed":
            self.exit(
                ProviderWorkbenchResult("cancelled", result.changed, result.warning)
            )
            return
        # Search is optional and remains configurable from Settings/commands.
        # Do not read or rewrite saved search choices on the setup critical path.
        self.exit(ProviderWorkbenchResult("completed", result.changed, result.warning))


def run_onboarding(
    app: App | None = None,
    *,
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema] | None = None,
) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    resolved_orchestrator = orchestrator or asyncio.run(build_default_orchestrator())
    onboarding_app = app or OnboardingApp(
        config=resolved_orchestrator.config, orchestrator=resolved_orchestrator
    )
    result = onboarding_app.run()
    if isinstance(result, OnboardingFailure):
        rprint(f"\nFailed: {result.message}\n")
        sys.exit(1)
    if result is None:
        rprint("\nSetup cancelled. See you next time!")
        sys.exit(0)
    if isinstance(result, ProviderWorkbenchResult):
        if result.warning:
            rprint(f"\n{result.warning}")
        if result.status == "cancelled":
            if result.changed:
                rprint("\nSetup closed. Saved changes were kept.")
            else:
                rprint("\nSetup cancelled. See you next time!")
            sys.exit(0)
    asyncio.run(resolved_orchestrator.reload())
    try:
        resolved_orchestrator.config.require_active_provider_api_key()
    except (MissingAPIKeyError, ValueError) as error:
        rprint(
            f"\nFailed: Setup could not activate the selected provider: {error}. "
            "Run `chartreux --setup` to choose a usable model or configure its credential.\n"
        )
        sys.exit(1)
    # A completed setup that saved no catalog still runs on the shipped
    # defaults in code; publish the editable template so the user owns the
    # file. Onboarding is the only caller — it is reachable only after legacy
    # reconciliation, and publication never overwrites an existing file, so a
    # migrated or saved roster always wins.
    try:
        materialize_default_catalog()
    except (MaterializationError, OSError) as error:
        logger.warning(
            "Default catalog materialization could not be fully completed "
            "(not published): %s",
            error,
        )
    if isinstance(result, ProviderWorkbenchResult):
        rprint('\nSetup complete. Run "chartreux" to start using the Chartreux CLI.\n')
    return resolved_orchestrator
