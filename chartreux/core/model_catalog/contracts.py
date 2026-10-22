"""Frozen types and service boundaries for provider management.

Implementations in the discovery, catalog-writing, and UI work packages must use
these contracts rather than coupling screens to a concrete host or persistence
layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import httpx

if TYPE_CHECKING:
    from chartreux.core.model_catalog.loader import CatalogSnapshot

type ApiStyle = Literal["openai", "openai-responses", "anthropic"]
type EditState = Literal["untouched", "set", "cleared"]
type DiscoveryErrorCode = Literal[
    "auth_rejected",
    "unsupported_listing",
    "rate_limited",
    "connection",
    "tls",
    "malformed",
    "empty",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class OptionalEdit[T]:
    """An optional edit whose blank/cleared state is never inferred from its value."""

    state: EditState = "untouched"
    value: T | None = None

    @classmethod
    def set(cls, value: T) -> OptionalEdit[T]:
        """Represent an explicit replacement with ``value``."""
        return cls("set", value)

    @classmethod
    def cleared(cls) -> OptionalEdit[T]:
        """Represent an explicit removal of an inherited or prior value."""
        return cls("cleared")


@dataclass(frozen=True, slots=True)
class ProviderDraft:
    """Editable provider form state, including an explicit credential never re-read."""

    preset: str | None
    provider_id: str
    name: str
    api_base: str
    api_style: ApiStyle
    api_key_env_var: str
    key: str | None
    backend: str = "generic"
    reasoning_field_name: str = "reasoning_content"
    extra_headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelEdits:
    """Optional per-model metadata edits, preserving untouched versus cleared."""

    input_price: OptionalEdit[float] = field(default_factory=OptionalEdit)
    output_price: OptionalEdit[float] = field(default_factory=OptionalEdit)
    cached_input_price: OptionalEdit[float] = field(default_factory=OptionalEdit)
    thinking: OptionalEdit[str] = field(default_factory=OptionalEdit)
    temperature: OptionalEdit[float] = field(default_factory=OptionalEdit)
    supports_images: OptionalEdit[bool] = field(default_factory=OptionalEdit)
    auto_compact_threshold: OptionalEdit[float] = field(default_factory=OptionalEdit)


@dataclass(frozen=True, slots=True)
class DiscoveryItem:
    """A raw wire ID and the optional provider-supplied display label."""

    wire_id: str
    display_label: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """A successful, unfiltered provider listing with safe bounded diagnostics."""

    models: tuple[DiscoveryItem, ...]
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscoveryError:
    """A safe, typed discovery failure suitable for retry or manual fallback."""

    code: DiscoveryErrorCode
    message: str
    diagnostics: tuple[str, ...] = ()
    listing_unsupported: bool = False


@dataclass(frozen=True, slots=True)
class TLSConfig:
    """TLS policy supplied to discovery independently of a concrete HTTP client."""

    enable_system_trust_store: bool = False


@dataclass(frozen=True, slots=True)
class CatalogChanges:
    """One atomic catalog commit with positional legacy provider support."""

    provider_id: str
    provider: Mapping[str, object]
    models: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    roles: Mapping[str, Mapping[str, object]] | None = None
    providers: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    expected_revision: str | None = None

    @property
    def provider_patches(self) -> dict[str, Mapping[str, object]]:
        """Combine the positional legacy provider patch with catalog-wide patches."""
        patches = dict(self.providers)
        if self.provider:
            patches[self.provider_id] = {
                **patches.get(self.provider_id, {}),
                **self.provider,
            }
        return patches


@dataclass(frozen=True, slots=True)
class CatalogWriteResult:
    """The newly effective catalog snapshot after a successful batch write."""

    snapshot: CatalogSnapshot
    changed: bool


@dataclass(frozen=True, slots=True)
class CatalogValidationError:
    """A validation failure that leaves the catalog overlay unchanged."""

    message: str


@dataclass(frozen=True, slots=True)
class CredentialSaveResult:
    """Credential persistence outcome; session-only use is intentionally explicit."""

    status: Literal["saved", "session_only", "invalid_env_var"]
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ConfigPersistResult:
    """Outcome of persisting one user configuration setting."""

    persisted: bool
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ConfigReloadResult:
    """Outcome of reloading catalog and configuration at the runtime boundary."""

    snapshot: CatalogSnapshot | None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderWorkbenchResult:
    """Typed completion or cancellation returned by Provider Settings."""

    status: Literal["completed", "cancelled"]
    changed: bool = False
    warning: str | None = None


class DiscoveryService(Protocol):
    """Async provider-listing boundary; callers pass credentials without env lookup."""

    async def __call__(
        self,
        provider: ProviderDraft,
        credential: str | None,
        tls: TLSConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> DiscoveryResult | DiscoveryError:
        """Discover raw model IDs using the supplied client or a policy-aware client."""
        ...


class CatalogWriter(Protocol):
    """Synchronous catalog persistence boundary aligned with ``CatalogStore``."""

    def apply_changes(
        self, changes: CatalogChanges
    ) -> CatalogWriteResult | CatalogValidationError:
        """Apply one validated atomic overlay batch without mutating on failure."""
        ...


class CredentialService(Protocol):
    """Synchronous credential boundary that distinguishes disk and session outcomes."""

    def save_key(self, env_var: str, key: str) -> CredentialSaveResult:
        """Install ``key`` in the session and attempt durable credential storage."""
        ...


class ConfigService(Protocol):
    """Async user-config and runtime reload boundary for provider-management hosts."""

    async def persist_theme(self, theme: str) -> ConfigPersistResult:
        """Persist the literal theme selection, including ``auto``."""
        ...

    async def reload_catalog_and_config(self) -> ConfigReloadResult:
        """Reload the combined catalog and configuration after a catalog save."""
        ...
