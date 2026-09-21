from __future__ import annotations

from pathlib import Path

from chartreux.core.config import DEFAULT_MISTRAL_API_ENV_KEY, ProviderConfig
from chartreux.core.llm_models import Backend
from chartreux.setup.auth import AuthState, AuthStateKind, assess_auth_state


def _mistral_provider() -> ProviderConfig:
    return ProviderConfig(
        name="mistral",
        api_base="https://api.mistral.ai/v1",
        api_key_env_var=DEFAULT_MISTRAL_API_ENV_KEY,
        backend=Backend.MISTRAL,
    )


def _generic_provider(*, api_key_env_var: str = "CUSTOM_API_KEY") -> ProviderConfig:
    return ProviderConfig(
        name="custom",
        api_base="https://custom.example/v1",
        api_key_env_var=api_key_env_var,
        backend=Backend.GENERIC,
    )


def _write_env_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_assess_signed_out_when_provider_requires_key_without_any_source(
    tmp_path: Path,
) -> None:
    assert assess_auth_state(
        _mistral_provider(), env_path=tmp_path / ".env", environ={}
    ) == AuthState(
        kind=AuthStateKind.SIGNED_OUT,
        can_use_active_provider=False,
        env_key=DEFAULT_MISTRAL_API_ENV_KEY,
    )


def test_assess_auth_not_required_when_provider_has_no_api_key_env_var(
    tmp_path: Path,
) -> None:
    assert assess_auth_state(
        _generic_provider(api_key_env_var=""), env_path=tmp_path / ".env", environ={}
    ) == AuthState(
        kind=AuthStateKind.AUTH_NOT_REQUIRED, can_use_active_provider=True, env_key=None
    )


def test_assess_chartreux_home_env_file_when_key_is_in_dotenv(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    _write_env_file(env_path, f"{DEFAULT_MISTRAL_API_ENV_KEY}=file-key\n")

    assert assess_auth_state(
        _mistral_provider(), env_path=env_path, environ={}
    ) == AuthState(
        kind=AuthStateKind.CHARTREUX_HOME_ENV_FILE,
        can_use_active_provider=True,
        env_key=DEFAULT_MISTRAL_API_ENV_KEY,
    )


def test_assess_process_env_when_key_is_only_in_process_env(tmp_path: Path) -> None:
    assert assess_auth_state(
        _mistral_provider(),
        env_path=tmp_path / ".env",
        environ={DEFAULT_MISTRAL_API_ENV_KEY: "process-key"},
    ) == AuthState(
        kind=AuthStateKind.PROCESS_ENV,
        can_use_active_provider=True,
        env_key=DEFAULT_MISTRAL_API_ENV_KEY,
    )


def test_assess_process_env_takes_precedence_when_it_predates_dotenv_load(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    _write_env_file(env_path, f"{DEFAULT_MISTRAL_API_ENV_KEY}=file-key\n")

    assert assess_auth_state(
        _mistral_provider(),
        env_path=env_path,
        environ={DEFAULT_MISTRAL_API_ENV_KEY: "process-key"},
        process_env_had_value_before_dotenv_load=True,
    ) == AuthState(
        kind=AuthStateKind.PROCESS_ENV,
        can_use_active_provider=True,
        env_key=DEFAULT_MISTRAL_API_ENV_KEY,
    )


def test_assess_chartreux_home_env_file_for_custom_provider(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    _write_env_file(env_path, "CUSTOM_API_KEY=file-key\n")

    assert assess_auth_state(
        _generic_provider(), env_path=env_path, environ={}
    ) == AuthState(
        kind=AuthStateKind.CHARTREUX_HOME_ENV_FILE,
        can_use_active_provider=True,
        env_key="CUSTOM_API_KEY",
    )
