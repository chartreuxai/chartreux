from __future__ import annotations

from chartreux.app_server.config import EffectiveThinkingView

UNKNOWN_EFFORT = "unknown"


def api_effort_label(effective: EffectiveThinkingView | None) -> str:
    """The provider wire effort as displayed; "none" when thinking is off."""
    if effective is None or not effective.known:
        return UNKNOWN_EFFORT
    return effective.api_effort if effective.api_effort is not None else "none"


def banner_annotation(effective: EffectiveThinkingView | None) -> str:
    """Compact suffix for a next request that will not match the selection.

    Empty when the request encodes exactly as selected; otherwise it names the
    API effort (or "unknown") instead of implying the selected level.
    """
    if effective is None or effective.matches_requested is True:
        return ""
    return f"; API: {api_effort_label(effective)}"


def picker_explanation(effective: EffectiveThinkingView | None) -> str | None:
    """One line explaining why the next request differs from the selection."""
    if effective is None or effective.matches_requested is True:
        return None
    if not effective.known:
        return (
            f"Selected {effective.requested}: the API effort for the next "
            "request is unknown."
        )
    if effective.semantic is not None and effective.semantic != effective.requested:
        return (
            f"Selected {effective.requested}; retained thinking history keeps "
            f"this model at {effective.semantic} (API effort "
            f"{api_effort_label(effective)}) for the next request."
        )
    if effective.effective is not None and effective.effective != effective.requested:
        return (
            f"Selected {effective.requested}; this model has no "
            f"{effective.requested} level, so the next request uses "
            f"{effective.effective} (API effort {api_effort_label(effective)})."
        )
    return (
        f"Selected {effective.requested}; this model's API effort is "
        f"{api_effort_label(effective)} for the next request."
    )


def status_rows(
    effective: EffectiveThinkingView | None, *, requested: str
) -> list[tuple[str, str]]:
    """The three /status values for the next request's thinking estimate."""
    if effective is None:
        return [
            ("Requested", requested),
            ("Semantic", UNKNOWN_EFFORT),
            ("API effort", UNKNOWN_EFFORT),
        ]
    semantic = (
        effective.semantic
        if effective.known and effective.semantic is not None
        else UNKNOWN_EFFORT
    )
    return [
        ("Requested", effective.requested),
        ("Semantic", semantic),
        ("API effort", api_effort_label(effective)),
    ]
