from __future__ import annotations

from dataclasses import dataclass
import os

from chartreux.core.config import (
    DEFAULT_THEME,
    ChartreuxConfigSchema,
    ModelConfig,
    ProviderConfig,
)
from chartreux.core.config.default_orchestrator import build_default_orchestrator
from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    get_harness_files_manager,
)
from chartreux.core.utils.concurrency import run_sync


@dataclass(frozen=True)
class OnboardingContext:
    """Initial onboarding inputs; catalog state is reloaded by the shared flow."""

    provider: ProviderConfig
    theme: str = DEFAULT_THEME
    has_provider_credentials: bool = False
    enable_system_trust_store: bool = False
    repair_active_model: bool = False

    @staticmethod
    def _fallback_provider(config: ChartreuxConfigSchema) -> ProviderConfig:
        snapshot = config.catalog_snapshot
        if snapshot is not None:
            for provider_id, definition in snapshot.catalog.providers.items():
                if not definition.disabled:
                    return config.get_provider_for_model(
                        ModelConfig(name="", provider=provider_id, alias="")
                    )
        return ProviderConfig(name="onboarding/default", api_base="https://localhost")

    @staticmethod
    def _has_usable_model(config: ChartreuxConfigSchema) -> bool:
        snapshot = config.catalog_snapshot
        if snapshot is None:
            return False
        catalog = snapshot.catalog
        return any(
            not model.disabled
            and any(
                not deployment.disabled
                and not catalog.providers[deployment.provider].disabled
                for deployment in model.deployments
            )
            for model in catalog.models.values()
        )

    @classmethod
    def from_config(cls, config: ChartreuxConfigSchema) -> OnboardingContext:
        repair_active_model = False
        try:
            provider = config.get_active_provider()
        except ValueError:
            provider = cls._fallback_provider(config)
            repair_active_model = cls._has_usable_model(config)
        return cls(
            provider=provider,
            theme=config.theme,
            has_provider_credentials=not provider.api_key_env_var
            or bool(os.environ.get(provider.api_key_env_var)),
            enable_system_trust_store=config.enable_system_trust_store,
            repair_active_model=repair_active_model,
        )

    @classmethod
    def load(
        cls, *, harness_files: HarnessFilesManager | None = None
    ) -> OnboardingContext:
        manager = harness_files or get_harness_files_manager()
        orchestrator = run_sync(build_default_orchestrator(harness_files=manager))
        return cls.from_config(orchestrator.config)
