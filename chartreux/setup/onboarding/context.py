from __future__ import annotations

from dataclasses import dataclass
import os

from chartreux.core.config import DEFAULT_THEME, ChartreuxConfigSchema, ProviderConfig
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

    @classmethod
    def from_config(cls, config: ChartreuxConfigSchema) -> OnboardingContext:
        provider = config.get_active_provider()
        return cls(
            provider=provider,
            theme=config.theme,
            has_provider_credentials=not provider.api_key_env_var
            or bool(os.environ.get(provider.api_key_env_var)),
            enable_system_trust_store=config.enable_system_trust_store,
        )

    @classmethod
    def load(
        cls, *, harness_files: HarnessFilesManager | None = None
    ) -> OnboardingContext:
        manager = harness_files or get_harness_files_manager()
        orchestrator = run_sync(build_default_orchestrator(harness_files=manager))
        return cls.from_config(orchestrator.config)
