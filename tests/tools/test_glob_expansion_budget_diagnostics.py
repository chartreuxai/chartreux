"""Glob-expansion failure provenance in shell path-glob denials.

The pathname-expansion inspection has always failed closed, but its denial
collapsed every failure into one generic message. The expansion now reports
WHICH limit it hit — the directory-entry budget, the match-state budget, or
unsupported glob syntax — as provenance detail on the same path-glob denial.
Only the message changed: the verdicts, the caps, and the matcher are
untouched, and denials decided OUTSIDE the expansion inspection (a
caller-side option-like word or out-of-scope match, a token the inspector
never expanded) keep the generic message so a work budget is never
attributed to them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_diagnostics import MAX_DIAGNOSTIC_LENGTH
from chartreux.core.tools.builtins._shell_permission_resolver import (
    _MAX_EXPANSION_ENTRIES,
    _MAX_EXPANSION_MATCH_STATES,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.secret_redaction import ScrubPolicy, bind_policy

RESOLVER_MODULE = "chartreux.core.tools.builtins._shell_permission_resolver"

# The rendered path-glob denial marker, shared by every denial this file
# exercises: "Shell path glob candidate '<token>' in command segment
# '<part>' cannot be safely inspected".
PATH_GLOB_MARKER = "path glob candidate"
ENTRY_BUDGET_MARKER = "directory-entry budget"
MATCH_STATE_BUDGET_MARKER = "match-state budget"
UNSUPPORTED_SYNTAX_MARKER = "glob syntax"

# Substrings only a provenance detail would add; a generic denial
# (caller-side or never-inspected) must contain neither.
_NO_PROVENANCE = ("budget", UNSUPPORTED_SYNTAX_MARKER)


def _denied_reason(cwd: Path, command: str) -> str:
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=cwd)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    assert result.permission == ToolPermission.NEVER, (command, result)
    reason = result.reason or ""
    assert PATH_GLOB_MARKER in reason, reason
    return reason


def _assert_allowed(cwd: Path, command: str) -> None:
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=cwd)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    assert result.permission == ToolPermission.ALWAYS, (command, result)


@pytest.fixture(scope="module")
def entry_budget_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory whose enumeration alone exceeds the entry budget."""
    directory = tmp_path_factory.mktemp("entry-budget")
    (directory / "target.py").write_text("x = 1\n", encoding="utf-8")
    for index in range(_MAX_EXPANSION_ENTRIES + 1):
        (directory / f"entry-{index:05d}.txt").write_text("", encoding="utf-8")
    return directory


def test_entry_budget_denial_names_entry_budget(entry_budget_dir: Path) -> None:
    """An entry-budget exhaustion says so; it is not a generic denial."""
    reason = _denied_reason(entry_budget_dir, "grep *needle target.py")
    assert ENTRY_BUDGET_MARKER in reason
    assert MATCH_STATE_BUDGET_MARKER not in reason
    assert UNSUPPORTED_SYNTAX_MARKER not in reason


def test_match_state_budget_denial_names_match_state_budget(tmp_path: Path) -> None:
    """A match-state-budget exhaustion names the match-state table."""
    directory = tmp_path / "work"
    directory.mkdir()
    (directory / "target.py").write_text("x = 1\n", encoding="utf-8")
    # One 255-character entry: its (atoms + 1) x (name + 1) match-state
    # table alone exceeds the budget.
    (directory / ("a" * 255)).write_text("x = 2\n", encoding="utf-8")
    reason = _denied_reason(directory, f"grep {'?a' * 1100} target.py")
    assert MATCH_STATE_BUDGET_MARKER in reason
    assert ENTRY_BUDGET_MARKER not in reason


def test_budget_boundaries_preserve_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Just inside either budget stays allowed; one past it denies.

    The shipped caps are frozen — provenance work must not retune them —
    and the boundary verdicts are exactly the pre-provenance ones: only
    the denial message gained the attribution.
    """
    assert _MAX_EXPANSION_ENTRIES == 4096
    assert _MAX_EXPANSION_MATCH_STATES == 1 << 19

    directory = tmp_path / "work"
    directory.mkdir()
    (directory / "abcd").write_text("x = 1\n", encoding="utf-8")
    (directory / "target.py").write_text("x = 2\n", encoding="utf-8")
    # `a?c?` enumerates exactly two entries; its match-state table is
    # 5 * 5 (abcd) + 5 * 10 (target.py) = 75 states.
    command = "grep a?c? target.py"

    monkeypatch.setattr(f"{RESOLVER_MODULE}._MAX_EXPANSION_ENTRIES", 2)
    _assert_allowed(directory, command)
    monkeypatch.setattr(f"{RESOLVER_MODULE}._MAX_EXPANSION_ENTRIES", 1)
    reason = _denied_reason(directory, command)
    assert ENTRY_BUDGET_MARKER in reason

    monkeypatch.setattr(f"{RESOLVER_MODULE}._MAX_EXPANSION_ENTRIES", 2)
    monkeypatch.setattr(f"{RESOLVER_MODULE}._MAX_EXPANSION_MATCH_STATES", 75)
    _assert_allowed(directory, command)
    monkeypatch.setattr(f"{RESOLVER_MODULE}._MAX_EXPANSION_MATCH_STATES", 74)
    reason = _denied_reason(directory, command)
    assert MATCH_STATE_BUDGET_MARKER in reason


def test_unsupported_glob_is_not_budget_exhaustion(tmp_path: Path) -> None:
    """Unsupported glob syntax is its own category, never a work budget."""
    directory = tmp_path / "work"
    directory.mkdir()
    (directory / "target.py").write_text("x = 1\n", encoding="utf-8")
    (directory / "src").mkdir()
    (directory / "src" / "x.py").write_text("x = 2\n", encoding="utf-8")
    reason = _denied_reason(directory, "grep src/**/x.py target.py")
    assert UNSUPPORTED_SYNTAX_MARKER in reason
    assert ENTRY_BUDGET_MARKER not in reason
    assert MATCH_STATE_BUDGET_MARKER not in reason


def test_caller_denials_are_not_budget_exhaustion(tmp_path: Path) -> None:
    """Caller-side denials keep the generic message.

    A denial of the expansion's RESULTS — an option-like word the shell
    would substitute, or a match outside the authorized workspace — is
    decided after the inspection succeeded, so it must not carry a budget
    (or syntax) attribution.
    """
    directory = tmp_path / "work"
    directory.mkdir()
    (directory / "target.py").write_text("x = 1\n", encoding="utf-8")
    (directory / "-x").write_text("x = 2\n", encoding="utf-8")
    reason = _denied_reason(directory, "grep * target.py")
    assert not any(marker in reason for marker in _NO_PROVENANCE)

    (directory / "escape").symlink_to("/etc/passwd")
    reason = _denied_reason(directory, "grep escap* target.py")
    assert not any(marker in reason for marker in _NO_PROVENANCE)


def test_never_inspected_tokens_are_not_budget_exhaustion(tmp_path: Path) -> None:
    """Tokens the inspector never expands keep the generic message.

    A home-relative word, a word under a statically unknown cwd, and a
    partially quoted metacharacter are denied before any expansion runs;
    none of them may read as a work-budget exhaustion.
    """
    directory = tmp_path / "work"
    directory.mkdir()
    (directory / "target.py").write_text("x = 1\n", encoding="utf-8")
    (directory / "sub").mkdir()

    tilde = _denied_reason(directory, "grep ~/.x* target.py")
    assert not any(marker in tilde for marker in _NO_PROVENANCE)

    # `pushd` with extra operands leaves the cwd statically unknown; the
    # grep pattern is denied without any expansion attempt.
    unknown = _denied_reason(directory, "pushd sub extra; grep a*b target.py")
    assert not any(marker in unknown for marker in _NO_PROVENANCE)

    # A partially quoted metacharacter arrives unmarked and is denied
    # without an expansion attempt.
    partial = _denied_reason(directory, 'grep "*"x* target.py')
    assert not any(marker in partial for marker in _NO_PROVENANCE)


def test_lone_bracket_candidate_denies_fail_closed(tmp_path: Path) -> None:
    """A lone-bracket pattern candidate denies with the generic message.

    A pattern-position word whose only unquoted metacharacter is an
    UNCLOSED ``[`` (the shell treats it as a literal, so the inspector
    never expands it) denies fail closed at the path-glob rule itself —
    it must not fall through to the other operand diagnostics, and since
    no expansion inspection ran, no work-budget or syntax attribution may
    ride the denial.
    """
    directory = tmp_path / "work"
    directory.mkdir()
    (directory / "target.py").write_text("x = 1\n", encoding="utf-8")
    reason = _denied_reason(directory, "grep ~/x[y target.py")
    assert not any(marker in reason for marker in _NO_PROVENANCE)


def test_listing_glob_preserves_inspection_failure_reason(
    entry_budget_dir: Path, tmp_path: Path
) -> None:
    """The ls listing-glob site carries the same provenance.

    The listing-glob check routes through the same expansion inspection:
    its exhaustion keeps the entry-budget attribution, its caller-side
    (out-of-scope match) denial stays generic, and an expandable glob with
    no matches stays allowed.
    """
    reason = _denied_reason(entry_budget_dir, "ls *needle")
    assert ENTRY_BUDGET_MARKER in reason
    assert MATCH_STATE_BUDGET_MARKER not in reason

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "target.py").write_text("x = 1\n", encoding="utf-8")
    (plain / "escape").symlink_to("/etc/passwd")
    reason = _denied_reason(plain, "ls escap*")
    assert not any(marker in reason for marker in _NO_PROVENANCE)

    _assert_allowed(plain, "ls zzz*")


def test_expansion_diagnostics_are_bounded_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance rides the Diagnostic, so the renderer bounds and redacts it.

    The provenance detail passes through the same redaction, escaping, and
    bounding boundary as every other diagnostic field: a secret spelled in
    the denied token never reaches the rendered message, and the whole
    message stays within the diagnostic length cap.
    """
    secret = "syntheticsecret0123456789" * 8
    directory = tmp_path / "work"
    directory.mkdir()
    (directory / "target.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(f"{RESOLVER_MODULE}._MAX_EXPANSION_MATCH_STATES", 0)
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=directory)
    with bind_policy(ScrubPolicy(redaction_credentials=(("TEST_TOKEN", secret),))):
        context = tool.resolve_permission(
            BashArgs(command=f"grep {secret}?.py target.py")
        )
    assert context is not None and context.permission == ToolPermission.NEVER
    reason = context.reason or ""
    assert len(reason) <= MAX_DIAGNOSTIC_LENGTH
    assert secret not in reason
    assert "REDACTED" in reason
    assert MATCH_STATE_BUDGET_MARKER in reason
    assert not any(ord(character) < 32 for character in reason)
