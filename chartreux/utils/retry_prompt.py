from __future__ import annotations

from chartreux.utils import VIBE_WARNING_TAG


def build_retry_prompt(additional_instructions: str) -> str:
    message = (
        "The previous model stream ended before reaching its end. This retry starts "
        "a new turn. Completed tool results remain in the conversation; check them "
        "before taking action, because recent actions may repeat. Continue the "
        "unfinished response without repeating text already produced. If no "
        "response text was produced, answer the pending user request normally."
    )
    if instructions := additional_instructions.strip():
        message += (
            "\n\nFollow these additional instructions from the user while "
            f"continuing:\n{instructions}"
        )
    return f"<{VIBE_WARNING_TAG}>{message}</{VIBE_WARNING_TAG}>"
