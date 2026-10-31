"""Documentation assertions for the OSC 7501 program-status registration.

The shipped skill's embedded configuration reference and the configuration
key table must both name the new ``emit_program_status`` key, and the terminal
guide must document the behavior it controls. These assertions keep that
docs-sync obligation honest when any of these surfaces drifts.
"""

from __future__ import annotations

from pathlib import Path

from chartreux.core.skills.builtins.chartreux import SKILL

_ROOT = Path(__file__).parents[1]
_DOCS = _ROOT / "docs"
CONFIGURATION = (_DOCS / "reference" / "configuration.md").read_text()
TERMINAL = (_DOCS / "guides" / "terminal.md").read_text()
CHANGELOG = (_ROOT / "CHANGELOG.md").read_text()


def test_skill_block_registers_emit_program_status() -> None:
    assert "emit_program_status = true" in SKILL.prompt
    assert "id=chartreux" not in SKILL.prompt
    assert "scoped to the program id" not in SKILL.prompt


def test_configuration_key_table_registers_emit_program_status() -> None:
    assert "| `emit_program_status` | `true` |" in CONFIGURATION
    assert "id=chartreux" not in CONFIGURATION
    assert "scoped to the program id" not in CONFIGURATION


def test_terminal_guide_documents_program_status() -> None:
    assert "## Program status (OSC 7501)" in TERMINAL
    assert "emit_program_status = false" in TERMINAL


def test_terminal_guide_documents_the_exit_clear() -> None:
    assert (
        "a bare `state=clear` that removes every OSC 7501 record on the terminal"
        in TERMINAL
    )
    assert "On exit Chartreux clears its record" in TERMINAL
    assert "id=chartreux" not in TERMINAL
    assert "scoped to the program id" not in TERMINAL
    assert "id=chartreux" not in CHANGELOG
    assert "scoped to the program id" not in CHANGELOG
