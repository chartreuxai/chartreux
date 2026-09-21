from __future__ import annotations

import os
from pathlib import Path

import pytest

from chartreux.acp.agent import ChartreuxAcpAgent as ChartreuxAcpAgentLoop
from chartreux.core.config import (
    DEFAULT_MISTRAL_API_ENV_KEY,
    ProviderConfig,
    load_dotenv_values,
)
from chartreux.core.llm_models import Backend
from chartreux.setup.auth import AuthStateKind
from chartreux.setup.onboarding.context import OnboardingContext


def build_mistral_provider(
    *, api_key_env_var: str = DEFAULT_MISTRAL_API_ENV_KEY
) -> ProviderConfig:
    return ProviderConfig(
        name="mistral",
        api_base="https://api.mistral.ai/v1",
        api_key_env_var=api_key_env_var,
        backend=Backend.MISTRAL,
    )


def build_generic_provider(
    *, name: str = "custom", api_key_env_var: str = "CUSTOM_API_KEY"
) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        api_base="https://custom.example/v1",
        api_key_env_var=api_key_env_var,
        backend=Backend.GENERIC,
    )


def build_acp_agent_loop(
    provider: ProviderConfig,
    *,
    environ_before_dotenv_load: dict[str, str] | None = None,
) -> ChartreuxAcpAgentLoop:
    return ChartreuxAcpAgentLoop(
        onboarding_context_loader=lambda: OnboardingContext(provider=provider),
        environ_before_dotenv_load=environ_before_dotenv_load,
    )


def write_env_file(config_dir: Path, content: str) -> None:
    (config_dir / ".env").write_text(content, encoding="utf-8")


class TestACPAuthStatus:
    @pytest.mark.asyncio
    async def test_returns_signed_out_when_no_key_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(DEFAULT_MISTRAL_API_ENV_KEY, raising=False)
        response = await build_acp_agent_loop(build_mistral_provider()).ext_method(
            "auth/status", {}
        )
        assert response == {
            "authenticated": False,
            "authState": AuthStateKind.SIGNED_OUT.value,
            "customDomain": None,
        }

    @pytest.mark.asyncio
    async def test_returns_chartreux_home_env_file_for_dotenv_key(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(DEFAULT_MISTRAL_API_ENV_KEY, raising=False)
        write_env_file(config_dir, f"{DEFAULT_MISTRAL_API_ENV_KEY}=file-key\n")
        response = await build_acp_agent_loop(build_mistral_provider()).ext_method(
            "auth/status", {}
        )
        assert response == {
            "authenticated": True,
            "authState": AuthStateKind.CHARTREUX_HOME_ENV_FILE.value,
            "customDomain": None,
        }

    @pytest.mark.asyncio
    async def test_uses_startup_env_snapshot_for_dotenv_key(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(DEFAULT_MISTRAL_API_ENV_KEY, raising=False)
        write_env_file(config_dir, f"{DEFAULT_MISTRAL_API_ENV_KEY}=file-key\n")
        environ_before_dotenv_load = os.environ.copy()
        load_dotenv_values()
        response = await build_acp_agent_loop(
            build_mistral_provider(),
            environ_before_dotenv_load=environ_before_dotenv_load,
        ).ext_method("auth/status", {})
        assert response["authState"] == AuthStateKind.CHARTREUX_HOME_ENV_FILE.value

    @pytest.mark.asyncio
    async def test_returns_process_env_when_key_existed_before_dotenv(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(DEFAULT_MISTRAL_API_ENV_KEY, "process-key")
        write_env_file(config_dir, f"{DEFAULT_MISTRAL_API_ENV_KEY}=file-key\n")
        response = await build_acp_agent_loop(build_mistral_provider()).ext_method(
            "auth/status", {}
        )
        assert response == {
            "authenticated": True,
            "authState": AuthStateKind.PROCESS_ENV.value,
            "customDomain": None,
        }

    @pytest.mark.asyncio
    async def test_returns_auth_not_required_for_provider_without_env_key(self) -> None:
        response = await build_acp_agent_loop(
            build_generic_provider(name="llamacpp", api_key_env_var="")
        ).ext_method("auth/status", {})
        assert response == {
            "authenticated": True,
            "authState": AuthStateKind.AUTH_NOT_REQUIRED.value,
            "customDomain": None,
        }

    @pytest.mark.asyncio
    async def test_returns_chartreux_home_env_file_for_custom_key(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
        write_env_file(config_dir, "CUSTOM_API_KEY=file-key\n")
        response = await build_acp_agent_loop(build_generic_provider()).ext_method(
            "auth/status", {}
        )
        assert response == {
            "authenticated": True,
            "authState": AuthStateKind.CHARTREUX_HOME_ENV_FILE.value,
            "customDomain": None,
        }
