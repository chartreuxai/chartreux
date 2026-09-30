from __future__ import annotations

GRADIENT_COLORS = ["#B87333"]


def gradient_markup(text: str, offset: int, *, truecolor: bool = True) -> str:
    del offset
    if not truecolor:
        return f"[bold]{text}[/]"
    return f"[bold {GRADIENT_COLORS[0]}]{text}[/]"
