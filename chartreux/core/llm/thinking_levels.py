from __future__ import annotations

from collections.abc import Mapping

type ThinkingLevels = Mapping[str, str | None]
type ProviderKey = tuple[str, str | None]
type ModelKey = tuple[str, str]

# Provider tables describe both public-level encodability and wire values. They
# are intentionally dependency-light so launch resolution constructs no backend.
MISTRAL_THINKING_LEVELS: dict[str, str | None] = {
    "off": None,
    "low": "none",
    "medium": "high",
    "high": "high",
    "max": "high",
}
OPENAI_THINKING_LEVELS: dict[str, str | None] = {
    "off": None,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
}
OPENAI_RESPONSES_THINKING_LEVELS: dict[str, str | None] = {
    "off": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "xhigh",
}
ANTHROPIC_THINKING_LEVELS: dict[str, str | None] = {
    "off": None,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
}

PROVIDER_THINKING_LEVELS: dict[ProviderKey, ThinkingLevels] = {
    ("mistral", None): MISTRAL_THINKING_LEVELS,
    ("generic", "openai"): OPENAI_THINKING_LEVELS,
    ("generic", "openai-responses"): OPENAI_RESPONSES_THINKING_LEVELS,
    ("generic", "anthropic"): ANTHROPIC_THINKING_LEVELS,
}

GLM_5_3_THINKING_LEVELS: dict[str, str | None] = {
    "low": "low",
    "medium": "high",
    "high": "high",
    "max": "max",
}

# zai-glm-latest and zai-glm-5 are rolling aliases recorded as compatibility
# snapshots on 2026-09-16; unknown future IDs use the provider table.
MODEL_THINKING_LEVELS: dict[ModelKey, ThinkingLevels] = {
    ("mistral", "zai-glm-5-3"): GLM_5_3_THINKING_LEVELS,
    ("mistral", "zai-glm-5"): GLM_5_3_THINKING_LEVELS,
    ("mistral", "zai-glm-latest"): GLM_5_3_THINKING_LEVELS,
}


def get_thinking_levels(
    backend: str, api_style: str | None, model_name: str
) -> ThinkingLevels | None:
    """Return the exact model entry, or the provider table when one exists."""
    return MODEL_THINKING_LEVELS.get((
        backend,
        model_name,
    )) or PROVIDER_THINKING_LEVELS.get((
        backend,
        None if backend == "mistral" else api_style,
    ))


def get_thinking_wire_value(levels: ThinkingLevels, thinking: str) -> str | None:
    """Encode a requested level or raise the shared typed configuration error."""
    try:
        return levels[thinking]
    except KeyError as exc:
        # Delayed to keep the table registry dependency-light on normal paths.
        from chartreux.core.subagents import InvalidLaunchThinkingError

        message = (
            "Selected model cannot disable thinking"
            if thinking == "off"
            else "Configured thinking level is unsupported by the selected model"
        )
        raise InvalidLaunchThinkingError("config.thinking", message) from exc
