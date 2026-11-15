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
APP_SERVER = (_DOCS / "integrations" / "app-server.md").read_text()
CHANGELOG = (_ROOT / "CHANGELOG.md").read_text()


def _flat(text: str) -> str:
    """Flatten prose wrapping so sentence pins survive reflowed lines."""
    return " ".join(text.split())


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


def test_terminal_guide_documents_the_composed_status_semantics() -> None:
    """v0.7.9 point 5 (C7c): the guide's status sentence is the design §3.4
    replacement, exact sentence by sentence — main-scoped `idle`, composed
    `working`, `blocked` precedence, deferred `done`, interrupted `idle`,
    persistent `error`, and the live-session lag disclaimer.
    """
    flat = _flat(TERMINAL)
    assert (
        "`idle` means the main turn is idle and no active background subagent is known to be working."
        in flat
    )
    assert (
        "`working` means the main turn or an active background subagent is working."
        in flat
    )
    assert (
        "`blocked` takes precedence while a question or permission prompt needs attention."
        in flat
    )
    assert (
        "After a completed turn, `done` is deferred until its background work "
        "is no longer active; after an interrupted turn, the corresponding "
        "report is `idle`." in flat
    )
    assert "An error remains `error` until a later owner replaces it." in flat
    assert (
        "Reports depend on a live, responsive session and may lag a subagent transition."
        in flat
    )
    # The replaced pre-0.7.9 enumeration must not survive alongside it.
    assert "`idle` at the prompt" not in flat


def test_app_server_doc_documents_wire_status_scope_and_deferred_done() -> None:
    """v0.7.9 point 5 (C7c): the app-server integration doc states the wire
    side of the scope distinction — `{"type":"idle"}` stays main-turn-scoped
    with no composed variant, while the interactive terminal's `done` is
    deferred behind background work.
    """
    assert "## Session status scope" in APP_SERVER
    flat = _flat(APP_SERVER)
    assert (
        "When a completed main turn leaves background agents running, "
        '`session/updated` keeps patching `/status` to `{"type":"idle"}`' in flat
    )
    assert (
        "wire `idle` means the main turn is idle, not that no background work is active."
        in flat
    )
    assert "`running` carries the real `activeTurnId` of the new main turn" in flat
    assert (
        "No background-work status variant exists and the server composes "
        "no status on the client's behalf" in flat
    )
    assert (
        "after a completed turn the terminal's `done` is deferred until that "
        "background work is no longer active" in flat
    )
    assert (
        "Wire `idle` and terminal `working` can therefore coexist — a "
        "difference in scope, not wire parity." in flat
    )
