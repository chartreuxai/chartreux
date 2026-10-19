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


def build_resume_prompt(resumed_from_run_id: str | None = None) -> str:
    """The resume marker + injected continuation for a resumed subagent run.

    WP2 (handoff §1.4 item 7): exactly ONE resume marker per admitted
    resume, mirroring the /retry text. The marker names the failed run it
    resumes; the continuation instructs the agent to reconcile completed
    actions rather than repeat them blindly. The turn carrying this text is
    injected and NEVER user-initiated (F6: an injected continuation never
    clears shared availability cooldowns — only a genuine user-initiated
    /retry does).
    """
    marker = "This is a resumed run"
    if resumed_from_run_id is not None:
        marker += f" (resume of run {resumed_from_run_id})"
    message = (
        f"{marker}. Continue the unfinished assignment from the preserved "
        "conversation. Inspect completed tool results and current workspace "
        "state before acting. Recent actions may have completed despite "
        "interruption; reconcile them rather than repeat them blindly."
    )
    return f"<{VIBE_WARNING_TAG}>{message}</{VIBE_WARNING_TAG}>"
