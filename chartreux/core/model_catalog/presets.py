"""Editable provider-form presets; these are never catalog providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from chartreux.ui.providers.contracts import ApiStyle


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    """Defaults for one provider form, with ``None`` denoting user input."""

    id: str
    name: str
    api_base: str | None
    api_style: ApiStyle | None
    api_key_env_var: str | None
    backend: str = "generic"
    reasoning_field_name: str = "reasoning_content"


MISTRAL = ProviderPreset(
    "mistral",
    "Mistral",
    "https://api.mistral.ai/v1",
    "openai",
    "MISTRAL_API_KEY",
    backend="mistral",
)
OLLAMA_CLOUD = ProviderPreset(
    "ollama-cloud", "Ollama Cloud", "https://ollama.com/v1", "openai", "OLLAMA_API_KEY"
)
OPENCODE_GO = ProviderPreset(
    "opencode-go",
    "OpenCode Go",
    "https://opencode.ai/zen/go/v1",
    "openai",
    "OPENCODE_API_KEY",
)
GENERIC_OPENAI = ProviderPreset(
    "generic-openai", "Generic OpenAI-style", None, "openai", None
)
GENERIC_ANTHROPIC = ProviderPreset(
    "generic-anthropic", "Generic Anthropic-style", None, "anthropic", None
)
FULLY_CUSTOM = ProviderPreset("fully-custom", "Fully custom", None, None, None)

PRESETS: tuple[ProviderPreset, ...] = (
    MISTRAL,
    OLLAMA_CLOUD,
    OPENCODE_GO,
    GENERIC_OPENAI,
    GENERIC_ANTHROPIC,
    FULLY_CUSTOM,
)

PresetId = Literal[
    "mistral",
    "ollama-cloud",
    "opencode-go",
    "generic-openai",
    "generic-anthropic",
    "fully-custom",
]

__all__ = [
    "FULLY_CUSTOM",
    "GENERIC_ANTHROPIC",
    "GENERIC_OPENAI",
    "MISTRAL",
    "OLLAMA_CLOUD",
    "OPENCODE_GO",
    "PRESETS",
    "PresetId",
    "ProviderPreset",
]
