from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

type ThinkingLevels = Mapping[str, str | None]
type ProviderKey = tuple[str, str | None]
type ModelKey = tuple[str, str]

# The public vocabulary in semantic order. Every ordering rule — declared-level
# sets, nearest-level fallback, picker order — derives from this one tuple.
THINKING_LEVEL_ORDER: Final[tuple[str, ...]] = (
    "off",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

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

# Mistral Large 4 always reasons and accepts only low/medium/high, with no
# level that disables thinking (Mistral documentation, checked 2026-11).
MISTRAL_LARGE_4_THINKING_LEVELS: dict[str, str | None] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}

# GLM-5.3 declares low/high/max. `medium` is deliberately absent: it resolves
# to the nearest declared level and the substitution is reported.
GLM_5_3_THINKING_LEVELS: dict[str, str | None] = {
    "low": "low",
    "high": "high",
    "max": "max",
}

# zai-glm-latest and zai-glm-5 are rolling aliases recorded as compatibility
# snapshots on 2026-09-16; unknown future IDs use the provider table.
MODEL_THINKING_LEVELS: dict[ModelKey, ThinkingLevels] = {
    ("mistral", "mistral-large-4"): MISTRAL_LARGE_4_THINKING_LEVELS,
    ("mistral", "zai-glm-5-3"): GLM_5_3_THINKING_LEVELS,
    ("mistral", "zai-glm-5-3-flash"): GLM_5_3_THINKING_LEVELS,
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


def declared_thinking_levels(
    levels: ThinkingLevels, declared: Sequence[str] | None = None
) -> tuple[str, ...]:
    """The encodable levels in vocabulary order, narrowed by a declaration."""
    names = [level for level in THINKING_LEVEL_ORDER if level in levels]
    if declared is not None:
        names = [level for level in names if level in declared]
    return tuple(names)


def narrow_thinking_levels(
    levels: ThinkingLevels, declared: Sequence[str] | None
) -> ThinkingLevels:
    if declared is None:
        return levels
    return {level: levels[level] for level in declared if level in levels}


def thinking_level_is_known(thinking: str) -> bool:
    return thinking in THINKING_LEVEL_ORDER


@dataclass(frozen=True, slots=True)
class ThinkingResolution:
    """A requested level and the level the deployment can actually encode."""

    requested: str
    effective: str
    wire_value: str | None

    @property
    def fell_back(self) -> bool:
        return self.effective != self.requested


def _typed_error(message: str) -> Exception:
    # Delayed to keep the table registry dependency-light on normal paths.
    from chartreux.core.subagents import InvalidLaunchThinkingError

    return InvalidLaunchThinkingError("config.thinking", message)


def resolve_thinking_level(
    levels: ThinkingLevels, requested: str
) -> ThinkingResolution:
    """Resolve a requested level to the nearest declared level by position.

    An undeclared level falls back to the declared level nearest the request in
    ``THINKING_LEVEL_ORDER``; a tie resolves upward, toward more thinking. Only
    an unknown level name — never a level the model merely lacks — raises the
    typed configuration error.
    """
    if not thinking_level_is_known(requested):
        raise _typed_error("Unknown thinking level")
    declared = declared_thinking_levels(levels)
    if not declared:
        raise _typed_error("Selected model declares no thinking levels")
    if requested in levels:
        return ThinkingResolution(requested, requested, levels[requested])
    target = THINKING_LEVEL_ORDER.index(requested)
    effective = min(
        declared,
        key=lambda level: (
            abs(THINKING_LEVEL_ORDER.index(level) - target),
            -THINKING_LEVEL_ORDER.index(level),
        ),
    )
    return ThinkingResolution(requested, effective, levels[effective])


def get_thinking_wire_value(levels: ThinkingLevels, thinking: str) -> str | None:
    """Encode a requested level, falling back to the nearest declared level."""
    return resolve_thinking_level(levels, thinking).wire_value


def resolve_enabled_thinking_level(
    levels: ThinkingLevels, requested: str = "medium"
) -> ThinkingResolution:
    """Resolve a level that must keep thinking enabled.

    A request carrying retained reasoning cannot be encoded at a level that
    disables thinking, so those levels are excluded before the nearest one is
    chosen. A deployment with no enabled level raises the typed configuration
    error rather than letting an undeclared effort reach the provider.
    """
    enabled = {level: wire for level, wire in levels.items() if level != "off" and wire}
    if not enabled:
        raise _typed_error(
            "Selected model cannot re-enable thinking for retained reasoning"
        )
    return resolve_thinking_level(enabled, requested)


_reported_fallbacks: set[tuple[str, str, str]] = set()


def note_thinking_fallback(deployment: str, resolution: ThinkingResolution) -> bool:
    """Report one notice per new (deployment, requested, effective) triple.

    Returns whether this call produced the notice, so a selection boundary can
    tell a first fallback from a repeat without keeping its own state.
    """
    if not resolution.fell_back:
        return False
    key = (deployment, resolution.requested, resolution.effective)
    if key in _reported_fallbacks:
        return False
    _reported_fallbacks.add(key)
    from chartreux.observability.logging import logger

    logger.warning(
        "Thinking level %r is not declared by %s; using the nearest level %r",
        resolution.requested,
        deployment,
        resolution.effective,
    )
    return True


def reset_thinking_fallback_notices() -> None:
    """Forget reported fallbacks; tests use this to isolate notice assertions."""
    _reported_fallbacks.clear()
