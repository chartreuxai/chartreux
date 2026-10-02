from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from chartreux.core.config import ProviderConfig
from chartreux.core.config._defaults import (
    DEFAULT_API_CONNECT_TIMEOUT,
    DEFAULT_API_POOL_TIMEOUT,
    DEFAULT_API_RETRY_MAX_ELAPSED_TIME,
    DEFAULT_API_TIMEOUT,
    DEFAULT_API_WRITE_TIMEOUT,
)
from chartreux.core.llm_models import Backend

if TYPE_CHECKING:
    from collections.abc import Callable

    from chartreux.core.llm.types import BackendLike
    from chartreux.core.utils import RequestRetryBudget, RetryObserver
    from chartreux.utils.api_keys import ApiKeyOrigin


class CredentialResolution(Enum):
    """Sentinel preserving the backend's default credential resolution behavior."""

    UNRESOLVED = "unresolved"


type ResolvedCredential = tuple[str, ApiKeyOrigin] | None | CredentialResolution


def _create_mistral_backend(**kwargs: Any) -> BackendLike:
    from chartreux.core.llm.backend.mistral import MistralBackend

    return MistralBackend(**kwargs)


def _create_generic_backend(**kwargs: Any) -> BackendLike:
    from chartreux.core.llm.backend.generic import GenericBackend, validate_api_style

    provider = kwargs["provider"]
    validate_api_style(provider.api_style)
    return GenericBackend(**kwargs)


# The factories import the backend modules on first use rather than at module
# level: the backends pull in heavy dependencies that would otherwise slow CLI
# startup.
BACKEND_FACTORY: dict[Backend, Callable[..., BackendLike]] = {
    Backend.MISTRAL: _create_mistral_backend,
    Backend.GENERIC: _create_generic_backend,
}


def create_backend(
    *,
    provider: ProviderConfig,
    timeout: float = DEFAULT_API_TIMEOUT,
    retry_max_elapsed_time: float = DEFAULT_API_RETRY_MAX_ELAPSED_TIME,
    retry_budget: RequestRetryBudget | None = None,
    connect_timeout: float = DEFAULT_API_CONNECT_TIMEOUT,
    write_timeout: float = DEFAULT_API_WRITE_TIMEOUT,
    pool_timeout: float = DEFAULT_API_POOL_TIMEOUT,
    enable_system_trust_store: bool = False,
    on_retry: RetryObserver | None = None,
    resolved_credential: ResolvedCredential = CredentialResolution.UNRESOLVED,
) -> BackendLike:
    backend = Backend(provider.backend)
    factory = BACKEND_FACTORY[backend]
    transport_timeouts = {
        "connect_timeout": connect_timeout,
        "write_timeout": write_timeout,
        "pool_timeout": pool_timeout,
    }
    backend_kwargs: dict[str, Any] = {
        "provider": provider,
        "timeout": timeout,
        "retry_max_elapsed_time": retry_max_elapsed_time,
        "enable_system_trust_store": enable_system_trust_store,
        "on_retry": on_retry,
        **transport_timeouts,
    }
    if resolved_credential is not CredentialResolution.UNRESOLVED:
        backend_kwargs["resolved_credential"] = resolved_credential
    if retry_budget is not None:
        backend_kwargs["retry_budget"] = retry_budget
    return factory(**backend_kwargs)
