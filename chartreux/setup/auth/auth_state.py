from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, auto
import os
from pathlib import Path

from dotenv import dotenv_values

from chartreux.core.config import ProviderConfig
from chartreux.core.paths import GLOBAL_ENV_FILE


class AuthStateKind(StrEnum):
    SIGNED_OUT = auto()
    AUTH_NOT_REQUIRED = auto()
    CHARTREUX_HOME_ENV_FILE = auto()
    PROCESS_ENV = auto()


@dataclass(frozen=True, slots=True)
class _AuthEnvSnapshot:
    env_key: str
    current_process_has_value: bool
    dotenv_has_value: bool
    process_env_had_value_before_dotenv_load: bool


@dataclass(frozen=True, slots=True)
class AuthState:
    kind: AuthStateKind
    can_use_active_provider: bool
    env_key: str | None


def _has_value(value: str | None) -> bool:
    return bool(value)


def _dotenv_has_value(env_path: Path, env_key: str) -> bool:
    if not env_path.is_file() and not env_path.is_fifo():
        return False

    value = dotenv_values(env_path).get(env_key)
    if not isinstance(value, str):
        return False
    return _has_value(value)


def _auth_state(
    kind: AuthStateKind, *, can_use_active_provider: bool, env_key: str | None = None
) -> AuthState:
    return AuthState(
        kind=kind, can_use_active_provider=can_use_active_provider, env_key=env_key
    )


def _capture_auth_env_snapshot(
    env_key: str,
    *,
    env_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    process_env_had_value_before_dotenv_load: bool = False,
) -> _AuthEnvSnapshot:
    resolved_env_path = env_path if env_path is not None else GLOBAL_ENV_FILE.path
    resolved_environ = environ if environ is not None else os.environ

    return _AuthEnvSnapshot(
        env_key=env_key,
        current_process_has_value=_has_value(resolved_environ.get(env_key)),
        dotenv_has_value=_dotenv_has_value(resolved_env_path, env_key),
        process_env_had_value_before_dotenv_load=process_env_had_value_before_dotenv_load,
    )


def assess_auth_state(
    provider: ProviderConfig,
    *,
    env_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    process_env_had_value_before_dotenv_load: bool = False,
) -> AuthState:
    env_key = provider.api_key_env_var
    if not env_key:
        return _auth_state(
            AuthStateKind.AUTH_NOT_REQUIRED, can_use_active_provider=True
        )

    auth_snapshot = _capture_auth_env_snapshot(
        env_key,
        env_path=env_path,
        environ=environ,
        process_env_had_value_before_dotenv_load=process_env_had_value_before_dotenv_load,
    )
    if (
        not auth_snapshot.current_process_has_value
        and not auth_snapshot.dotenv_has_value
    ):
        return _auth_state(
            AuthStateKind.SIGNED_OUT, can_use_active_provider=False, env_key=env_key
        )

    if auth_snapshot.process_env_had_value_before_dotenv_load:
        kind = AuthStateKind.PROCESS_ENV
    elif auth_snapshot.dotenv_has_value:
        kind = AuthStateKind.CHARTREUX_HOME_ENV_FILE
    elif auth_snapshot.current_process_has_value:
        kind = AuthStateKind.PROCESS_ENV
    else:
        raise AssertionError("assess_auth_state reached unreachable state")

    return _auth_state(kind, can_use_active_provider=True, env_key=env_key)


__all__ = ["AuthState", "AuthStateKind", "assess_auth_state"]
