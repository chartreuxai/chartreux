from __future__ import annotations

GRADIENT_COLORS = [
    "#3A506B",
    "#4A6178",
    "#5A7287",
    "#6A7D8E",
    "#8A7D6E",
    "#A8794A",
    "#B87333",
    "#A8794A",
    "#8A7D6E",
    "#6A7D8E",
]


def gradient_markup(text: str, offset: int) -> str:
    result = []
    for index, char in enumerate(text):
        color = GRADIENT_COLORS[(index + offset) % len(GRADIENT_COLORS)]
        result.append(f"[bold {color}]{char}[/]")
    return "".join(result)
