"""Shared user-facing model identity formatting."""

from __future__ import annotations


def format_model_display_name(provider: str | None, wire_name: str | None) -> str:
    """Return a stable ``<provider>/<model>`` label when both parts are known.

    Legacy records may lack either part; retain the known value rather than
    producing a malformed separator or hiding the model entirely.
    """
    provider = provider.strip() if provider else ""
    wire_name = wire_name.strip() if wire_name else ""
    if provider and wire_name:
        return (
            wire_name
            if wire_name.startswith(f"{provider}/")
            else f"{provider}/{wire_name}"
        )
    return wire_name or provider or "unknown"
