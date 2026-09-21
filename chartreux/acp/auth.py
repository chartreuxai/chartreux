from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from typing import Any

from acp.schema import AuthenticateResponse, AuthMethodAgent

from chartreux.acp.exceptions import InvalidRequestError
from chartreux.core.config import load_dotenv_values
from chartreux.core.paths import GLOBAL_ENV_FILE
from chartreux.setup.auth import AuthState, assess_auth_state
from chartreux.setup.onboarding.context import OnboardingContext

type OnboardingContextLoader = Callable[[], OnboardingContext]


class AcpAuthController:
    def __init__(
        self,
        *,
        context_loader: OnboardingContextLoader | None = None,
        environ_before_dotenv_load: Mapping[str, str] | None = None,
    ) -> None:
        self._load_context = context_loader or OnboardingContext.load
        self._initial_environment = dict(
            environ_before_dotenv_load
            if environ_before_dotenv_load is not None
            else os.environ
        )

    def browser_methods(self, *, delegated: bool) -> list[AuthMethodAgent]:
        return []

    async def authenticate(
        self, method_id: str, arguments: dict[str, Any]
    ) -> AuthenticateResponse:
        raise InvalidRequestError(f"Unsupported auth method: {method_id}")

    def status(self) -> AuthState:
        load_dotenv_values(env_path=GLOBAL_ENV_FILE.path)
        provider = self._load_context().provider
        return assess_auth_state(
            provider,
            process_env_had_value_before_dotenv_load=bool(
                provider.api_key_env_var
                and self._initial_environment.get(provider.api_key_env_var)
            ),
        )

    def custom_domain(self) -> str | None:
        return None
