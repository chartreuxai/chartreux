from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import set_key, unset_key

from chartreux.core.config import ChartreuxConfigSchema, ProviderConfig
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.model_catalog.loader import CatalogLoadError, CatalogStore
from chartreux.core.paths import GLOBAL_ENV_FILE
from chartreux.core.utils.concurrency import run_sync
from chartreux.observability.logging import logger


def _save_api_key_to_env_file(env_key: str, api_key: str) -> None:
    GLOBAL_ENV_FILE.path.parent.mkdir(parents=True, exist_ok=True)
    set_key(GLOBAL_ENV_FILE.path, env_key, api_key)


def _remove_api_key_from_env_file(env_key: str) -> None:
    if not GLOBAL_ENV_FILE.path.exists():
        return
    unset_key(GLOBAL_ENV_FILE.path, env_key)


def _load_onboarding_provider() -> ProviderConfig:
    from chartreux.setup.onboarding.context import OnboardingContext

    return OnboardingContext.load().provider


def resolve_api_key_provider(provider: ProviderConfig | None = None) -> ProviderConfig:
    return provider or _load_onboarding_provider()


async def apply_provider_to_config(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema] | None,
    provider: ProviderConfig,
    *,
    reason: str = "onboarding",
) -> bool:
    """Persist provider metadata in models.toml, never in config.toml.

    ``orchestrator`` remains accepted for onboarding callers during the transition,
    but it deliberately has no catalog write authority.
    """
    payload = provider.model_dump(
        mode="json", exclude={"name"}, exclude_none=True, exclude_defaults=True
    )
    provider_id = provider.name if "/" in provider.name else f"{provider.name}/default"
    try:
        CatalogStore().upsert_provider(payload, provider_id)
    except CatalogLoadError as failure:
        logger.error(
            "Failed to persist provider to models.toml name=%s",
            provider.name,
            exc_info=failure,
        )
        return False
    return True


@dataclass(frozen=True, slots=True)
class ProviderCredentialsPersistRequest:
    """Provider configuration to persist in the user catalog."""

    provider: ProviderConfig


@dataclass(frozen=True, slots=True)
class ProviderCredentialsPersistResult:
    """Checked outcome of provider configuration persistence."""

    provider: bool

    @property
    def all_requested_succeeded(self) -> bool:
        return self.provider

    def first_failure(self) -> str | None:
        return None if self.provider else "provider"


def persist_provider_to_config(provider: ProviderConfig) -> bool:
    """Sync compatibility wrapper for the catalog-backed provider writer."""
    return run_sync(apply_provider_to_config(None, provider))


async def persist_provider_credentials(
    request: ProviderCredentialsPersistRequest, *, reason: str = "onboarding"
) -> ProviderCredentialsPersistResult:
    """Persist provider configuration and report whether the save succeeded."""
    provider_ok = await apply_provider_to_config(None, request.provider, reason=reason)
    return ProviderCredentialsPersistResult(provider=provider_ok)


def persist_api_key(provider: ProviderConfig, api_key: str) -> str:
    env_key = provider.api_key_env_var
    if not env_key:
        return "env_var_error:<empty>"
    try:
        os.environ[env_key] = api_key
    except ValueError:
        return f"env_var_error:{env_key}"
    try:
        _save_api_key_to_env_file(env_key, api_key)
    except (OSError, ValueError) as err:
        return f"save_error:{err}"
    return "completed"
