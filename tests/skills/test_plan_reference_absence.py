"""Absence pins for the removed plan-mode surface (v0.8.0).

The positive pins (former test_plan_reference_pins.py) were deleted in the
same commit as the skill text they pinned, because the skill header requires
text and pins to move together. These reduced ABSENCE PINS keep the removal
honest in the other direction: the PINNED FILE SET — the shipped skill text
(chartreux.py) plus docs/reference/commands.md and
docs/guides/sessions-workspaces.md, and no other file — must contain NONE of
the pinned plan-mode TOKENS, so reintroducing the surface in prose cannot
pass silently.

The bare word "plan" is deliberately NOT pinned: legitimate survivors outside
the plan-mode sense (for example commands.md "plan/input", or the kept
tools-safety.md / adr-0018 mentions that are out of the pinned file set)
use it and must keep compiling. The shipped prompt wraps hard, so all texts
are compared whitespace-normalized.
"""

from __future__ import annotations

from pathlib import Path
import re

from chartreux.core.skills.builtins.chartreux import SKILL

_REPO_ROOT = Path(__file__).parents[2]

# The pinned file set: shipped skill text + the two docs pages, ONLY.
_PINNED_DOC_FILES = (
    _REPO_ROOT / "docs" / "reference" / "commands.md",
    _REPO_ROOT / "docs" / "guides" / "sessions-workspaces.md",
)

_WHITESPACE = re.compile(r"\s+")

_PINNED_TOKENS = (
    "/plan",
    "plan mode",
    "plan-mode",
    "[plan",
    "plan_mode",
    "--plan",
    "CHARTREUX_PLAN__",
    "Plan/scratchpad",
    "plans/",
)


def _normalized(text: str) -> str:
    return _WHITESPACE.sub(" ", text)


def _pinned_texts() -> dict[str, str]:
    texts = {
        "chartreux skill (chartreux/core/skills/builtins/chartreux.py)": _normalized(
            SKILL.prompt
        )
    }
    for path in _PINNED_DOC_FILES:
        texts[str(path.relative_to(_REPO_ROOT))] = _normalized(
            path.read_text(encoding="utf-8")
        )
    return texts


def test_the_pinned_file_set_is_exactly_the_skill_and_the_two_docs() -> None:
    assert len(_PINNED_DOC_FILES) == 2
    for path in _PINNED_DOC_FILES:
        assert path.is_file(), path


def test_no_pinned_token_survives_in_the_pinned_file_set() -> None:
    for name, text in _pinned_texts().items():
        for token in _PINNED_TOKENS:
            assert token not in text, f"{token!r} still present in {name}"
