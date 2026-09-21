"""Frozen types and service boundaries for provider management.

Implementations in the discovery, catalog-writing, and UI work packages must use
these contracts rather than coupling screens to a concrete host or persistence
layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    aliases: OptionalEdit[tuple[str, ...]] = field(default_factory=OptionalEdit)
    tags: OptionalEdit[tuple[str, ...]] = field(default_factory=OptionalEdit)


@dataclass(frozen=True, slots=True)
class ModelSelectionDraft:
    """A discovered wire ID selected for addition with optional metadata edits."""

    wire_name: str
    base_name: str
    edits: ModelEdits = field(default_factory=ModelEdits)


@dataclass(frozen=True, slots=True)
class ProviderManagementDraft:
    """All unsaved data for one provider-management flow instance."""

    provider: ProviderDraft
    selections: tuple[ModelSelectionDraft, ...] = ()
    theme: str | None = None
    active_model: str | None = None


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


@dataclass(frozen=True, slots=True)
class TLSConfig:
    """TLS policy supplied to discovery independently of a concrete HTTP client."""

    enable_system_trust_store: bool = False


@dataclass(frozen=True, slots=True)
class CatalogChanges:
    """One atomic provider catalog commit expressed as validated plain-data patches."""

    provider_id: str
    provider: Mapping[str, object]
    models: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    tags: Mapping[str, tuple[str, ...]] | None = None


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
class ProviderFlowResult:
    """Typed value passed to ``ModalScreen.dismiss`` by either flow host."""

    status: Literal["completed", "cancelled"]
    active_model: str | None = None
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

    async def persist_active_model(self, expression: str) -> ConfigPersistResult:
        """Persist only a v0.1 canonical name, alias, or ``@tag`` expression."""
        ...

    async def reload_catalog_and_config(self) -> ConfigReloadResult:
        """Reload the combined catalog and configuration before active-model adoption."""
        ...


def serialize_draft(draft: ProviderManagementDraft) -> dict[str, object]:
    """Return a plain-data representation for tests and host state handoff."""
    return {
        "provider": {
            "preset": draft.provider.preset,
            "provider_id": draft.provider.provider_id,
            "name": draft.provider.name,
            "api_base": draft.provider.api_base,
            "api_style": draft.provider.api_style,
            "api_key_env_var": draft.provider.api_key_env_var,
            "backend": draft.provider.backend,
            "reasoning_field_name": draft.provider.reasoning_field_name,
            "extra_headers": dict(draft.provider.extra_headers),
        },
        "selections": [
            {
                "wire_name": selection.wire_name,
                "base_name": selection.base_name,
                "edits": _serialize_edits(selection.edits),
            }
            for selection in draft.selections
        ],
        "theme": draft.theme,
        "active_model": draft.active_model,
    }


def _serialize_edits(edits: ModelEdits) -> dict[str, dict[str, object]]:
    return {
        name: {"state": edit.state, "value": edit.value}
        for name, edit in (
            ("input_price", edits.input_price),
            ("output_price", edits.output_price),
            ("cached_input_price", edits.cached_input_price),
            ("aliases", edits.aliases),
            ("tags", edits.tags),
        )
    }


def active_model_expression_is_valid(expression: str) -> bool:
    """Recognize v0.1 active-model expressions: name, alias, or a nonempty tag."""
    if not expression or "/" in expression:
        return False
    if expression.startswith("@"):
        return len(expression) > 1 and "@" not in expression[1:]
    return "@" not in expression


def selected_wire_ids(result: DiscoveryResult) -> Sequence[str]:
    """Expose discovered IDs without changing their provider-provided spelling."""
    return tuple(item.wire_id for item in result.models)
