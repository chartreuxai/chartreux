"""C2a: quote-aware grep-family pattern-operand hardening.

grep-family pattern-position tokens carrying glob metacharacters are treated
by QUOTING. A fully quoted metacharacter (``grep 'a*b' f``) is regex data —
the pre-C2a behavior, pinned by tests/recovery/test_grep_glob_candidates.py —
but an UNQUOTED one (``grep a*b f``) makes the shell pathname-expand the word
before grep runs, so the expansion's matches join the read-inspected surface:
in-root matches are allowed, out-of-root matches (or a cwd the resolver cannot
locate) deny through the path-glob rule. A PARTIALLY quoted metacharacter
(``grep '*'a* f``) fails closed via the same path-glob rule. The
``--include``/``--exclude`` glob values stay allowed only in their ATTACHED
spellings (C2a decision: they only filter recursive-search file names and
never become reads), pinned by ``include_glob_value-decided``; since the
14d7123 hardening, a SEPARATE-TOKEN unquoted value is expansion-inspected
like any other value-taking option's value.

The expansion boundary itself is conservative: the cwd is anchored literally
(never pattern-interpreted), bracket syntax the model cannot faithfully
expand (POSIX classes, collating symbols, brace expansion) fails closed
instead of authorizing an empty expansion, and an expanded filename whose
command-line spelling begins with ``-`` denies (option injection). Provenance
survives argv-wrapper reassembly (``command grep a*b f``): the synthetic
quoting never upgrades an unquoted metacharacter to quoted data.

The quote provenance itself is carried by ``split_shell_tokens``
(``_shell_argv.py``): token values are byte-identical to ``shlex.split`` and
are fuzz-checked against it below.
"""

from __future__ import annotations

from pathlib import Path
import random
import shlex
from typing import cast

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_argv import (
    PatternGlobCandidate,
    ShellToken,
    split_shell_tokens,
)
from chartreux.core.tools.builtins._shell_command_policy import (
    analyze_shell_command_policy,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.permissions import PermissionContext


def _permission(cwd: Path, command: str) -> PermissionContext:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    return result


def _write_fixtures(root: Path) -> None:
    (root / "target.py").write_text("x = 1\n", encoding="utf-8")
    # The crafted file is what an unquoted a*b pattern pathname-expands to.
    (root / "a*b").write_text("crafted\n", encoding="utf-8")


# (node id, command, expected permission, whether the fixture files exist)
_DISPOSITION_CASES = [
    pytest.param(
        "grep 'a*b' target.py",
        ToolPermission.ALWAYS,
        True,
        id="quoted_glob_pattern-allowed",
    ),
    pytest.param(
        "grep foo target.py",
        ToolPermission.ALWAYS,
        True,
        id="unquoted_plain_pattern-allowed",
    ),
    pytest.param(
        "grep [0-9] target.py",
        ToolPermission.ALWAYS,
        True,
        id="unquoted_bracket_no_match-allowed",
    ),
    pytest.param(
        "grep a*b target.py",
        ToolPermission.ALWAYS,
        True,
        id="unquoted_glob_pattern-inspected",
    ),
    pytest.param(
        "grep /etc/p* target.py",
        ToolPermission.NEVER,
        True,
        id="unquoted_glob_pattern-out_of_root",
    ),
    pytest.param(
        "grep '*'a* target.py", ToolPermission.NEVER, True, id="partial_quote-denied"
    ),
    pytest.param(
        "grep --include=*.py foo target.py",
        ToolPermission.ALWAYS,
        True,
        id="include_glob_value-decided",
    ),
]


@pytest.mark.parametrize(("command", "expected", "with_fixtures"), _DISPOSITION_CASES)
def test_pattern_disposition(
    tmp_path: Path, command: str, expected: ToolPermission, with_fixtures: bool
) -> None:
    if with_fixtures:
        _write_fixtures(tmp_path)
    context = _permission(tmp_path, command)
    assert context.permission is expected, (command, context)


@pytest.mark.parametrize(
    "gate",
    [
        pytest.param(
            [
                # B7: getopt-attached short; the bracket word is a FILE operand.
                "grep -eE '[0-9]' target.py",
                # C4/C5/C7: path-option values (pattern files) with globs.
                "grep -f '*.txt' target.py",
                "grep --file 'pat*.txt' target.py",
                "grep -f 'pat[0-9].txt' target.py",
                # E1/E2/E3: file-operand globs, quoted and unquoted.
                "grep foo '*.py'",
                "grep foo src/*.py",
                "grep foo '*.py' target.py",
                # G22-G25: outside-workspace operands and a bracket file operand.
                "grep -E 'a|b' /dev/null",
                "grep -e 'a|b' /dev/null",
                "grep -f /etc/hosts target.py",
                "grep -e 'foo' '[0-9].py'",
            ],
            id="g1_gate_protections-unchanged",
        )
    ],
)
def test_g1_gate_protections_unchanged(tmp_path: Path, gate: list[str]) -> None:
    """The pre-C2a denials the quote-aware inspection must not relax."""
    _write_fixtures(tmp_path)
    (tmp_path / "patterns.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("b\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("y = 2\n", encoding="utf-8")
    for command in gate:
        context = _permission(tmp_path, command)
        assert context.permission is ToolPermission.NEVER, (command, context)


# --- Expansion-boundary hardening --------------------------------------------


@pytest.mark.parametrize(
    "command", [pytest.param("grep a*b target.py", id="expansion_boundary-literal_cwd")]
)
def test_expansion_boundary_literal_cwd(tmp_path: Path, command: str) -> None:
    """A metacharacter-bearing cwd component is anchored literally.

    Pattern-interpreting the cwd (``root[1]`` read as the class ``[1]``)
    yields no matches and would authorize an empty expansion, while the real
    shell expansion reaches the out-of-root symlink: fail closed instead.
    """
    root = tmp_path / "root[1]"
    root.mkdir()
    (root / "target.py").write_text("x = 1\n", encoding="utf-8")
    (root / "a*b").symlink_to("/etc/hostname")
    context = _permission(root, command)
    assert context.permission is ToolPermission.NEVER, (command, context)


@pytest.mark.parametrize(
    "command",
    [pytest.param("grep [[:alpha:]] target.py", id="expansion_boundary-bracket_class")],
)
def test_expansion_boundary_bracket_class(tmp_path: Path, command: str) -> None:
    """POSIX bracket classes fail closed, never authorized-empty.

    The conservative expansion model cannot faithfully evaluate
    ``[[:alpha:]]`` (whose real matches here are out-of-root symlinks), so
    the token must not pass as an empty expansion.
    """
    (tmp_path / "target.py").write_text("x = 1\n", encoding="utf-8")
    for name in "ab":
        (tmp_path / name).symlink_to("/etc/hostname")
    context = _permission(tmp_path, command)
    assert context.permission is ToolPermission.NEVER, (command, context)


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(
            "grep [!a-z]* target.py", id="expansion_boundary-negated_letter_range"
        ),
        pytest.param(
            "grep [!c-a]* target.py", id="expansion_boundary-negated_reversed_range"
        ),
    ],
)
def test_expansion_boundary_negated_letter_range(tmp_path: Path, command: str) -> None:
    """A negated range must not shrink below bash's real match set.

    Case-broadening a POSITIVE range only over-denies, but broadening a
    NEGATED one excludes the opposite case every UTF-8 locale's ``[!a-z]``
    really matches: with ``Aardvark`` an out-of-root symlink, real bash
    pathname-expands ``[!a-z]*`` to it and grep reads it. The model keeps
    negated ranges un-broadened (their complement is the superset) and fails
    closed on reversed negated ranges: bash matches nothing for ``[c-a]``, so
    ``[!c-a]`` matches everything and no finite member set over-approximates
    it.
    """
    (tmp_path / "target.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "Aardvark.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "Aardvark").symlink_to("/etc/hostname")
    context = _permission(tmp_path, command)
    assert context.permission is ToolPermission.NEVER, (command, context)


@pytest.mark.parametrize(
    "command", [pytest.param("rg * target.py", id="expansion_option_injection-denied")]
)
def test_expansion_option_injection_denied(tmp_path: Path, command: str) -> None:
    """An expanded filename that parses as an option denies (fail closed).

    ``rg * target.py`` expands to ``rg --pre=id needle target.py target.py``:
    every expanded pathname is authorized, but ``--pre=id`` is an
    execution-bearing option injected by the expansion, so the expansion is
    denied.
    """
    (tmp_path / "--pre=id").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "needle").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "target.py").write_text("x = 1\n", encoding="utf-8")
    context = _permission(tmp_path, command)
    assert context.permission is ToolPermission.NEVER, (command, context)


@pytest.mark.parametrize(
    "command",
    [pytest.param("command grep a*b target.py", id="wrapper_provenance-preserved")],
)
def test_wrapper_provenance_preserved(tmp_path: Path, command: str) -> None:
    """The argv-wrapper reassembly keeps the pattern's active-glob provenance.

    Re-serializing the child argv with shlex.join must not upgrade the
    unquoted ``a*b`` to quoted regex data: the child is inspected exactly
    like the unwrapped unquoted case.
    """
    in_root = tmp_path / "in-root"
    in_root.mkdir()
    _write_fixtures(in_root)
    context = _permission(in_root, command)
    assert context.permission is ToolPermission.ALWAYS, (command, context)
    out_root = tmp_path / "out-root"
    out_root.mkdir()
    (out_root / "target.py").write_text("x = 1\n", encoding="utf-8")
    (out_root / "a*b").symlink_to("/etc/hostname")
    context = _permission(out_root, command)
    assert context.permission is ToolPermission.NEVER, (command, context)


@pytest.mark.parametrize(
    "command", [pytest.param("grep -e a*b target.py", id="option_value_glob-inspected")]
)
def test_option_value_glob_inspected(tmp_path: Path, command: str) -> None:
    """An unquoted glob-bearing -e value is inspected like the pattern position.

    The shell pathname-expands the -e value word into several words: the
    first becomes the option's regex and the rest become file operands
    (``grep -e axb ayb target.py``). With ``ayb`` an out-of-root symlink,
    real bash reads it, so the expansion must deny. The fully quoted value is
    pure regex data and stays allowed.
    """
    (tmp_path / "target.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "axb").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "ayb").symlink_to("/etc/hostname")
    context = _permission(tmp_path, command)
    assert context.permission is ToolPermission.NEVER, (command, context)
    quoted = _permission(tmp_path, "grep -e 'a*b' target.py")
    assert quoted.permission is ToolPermission.ALWAYS, quoted


# --- Provenance representation ----------------------------------------------


def test_meta_provenance_flags() -> None:
    _grep, quoted, unquoted, mixed, escaped, plain = cast(
        "list[ShellToken]", split_shell_tokens("grep 'a*b' a*b '*'a* a\\*b foo")
    )
    assert quoted.meta_quoted and not quoted.meta_unquoted and quoted == "a*b"
    assert not unquoted.meta_quoted and unquoted.meta_unquoted and unquoted == "a*b"
    assert mixed.meta_quoted and mixed.meta_unquoted and mixed == "*a*"
    assert escaped.meta_quoted and not escaped.meta_unquoted and escaped == "a*b"
    assert not plain.meta_quoted and not plain.meta_unquoted and plain == "foo"


def test_split_shell_tokens_matches_shlex_split() -> None:
    rng = random.Random(20261023)
    alphabet = list("abx*?[]") + list("'\"\\") + [" ", "\t", "-", ".", "/"]
    for _ in range(2000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
        try:
            expected = shlex.split(text)
        except ValueError:
            with pytest.raises(ValueError):
                split_shell_tokens(text)
            continue
        assert [str(token) for token in split_shell_tokens(text)] == expected, repr(
            text
        )


# --- Policy layer ------------------------------------------------------------


def test_policy_marks_unquoted_pattern_for_inspection() -> None:
    tokens = split_shell_tokens("grep a*b target.py")
    values = analyze_shell_command_policy(tokens).positional_values
    assert values is not None
    assert isinstance(values[0], PatternGlobCandidate)
    assert values[1] == "target.py"


def test_policy_keeps_quoted_pattern_excluded() -> None:
    tokens = split_shell_tokens("grep 'a*b' target.py")
    values = analyze_shell_command_policy(tokens).positional_values
    assert values == ("target.py",)


def test_policy_mixed_pattern_stays_plain_candidate() -> None:
    tokens = split_shell_tokens("grep '*'a* target.py")
    values = analyze_shell_command_policy(tokens).positional_values
    assert values is not None
    assert values == ("*a*", "target.py")
    assert not isinstance(values[0], PatternGlobCandidate)


def test_policy_plain_strings_keep_legacy_exclusion() -> None:
    values = analyze_shell_command_policy(["grep", "a*b", "target.py"])
    assert values.positional_values == ("target.py",)


def test_policy_rg_pattern_dispositions() -> None:
    expandable = analyze_shell_command_policy(
        split_shell_tokens("rg a*b target.py")
    ).positional_values
    assert expandable is not None
    assert isinstance(expandable[0], PatternGlobCandidate)
    assert expandable[1] == "target.py"

    quoted = analyze_shell_command_policy(
        split_shell_tokens("rg 'a*' target.py")
    ).positional_values
    assert quoted == ("target.py",)

    mixed = analyze_shell_command_policy(
        split_shell_tokens("rg '*'a* target.py")
    ).positional_values
    assert mixed is not None
    assert mixed == ("*a*", "target.py")
    assert not isinstance(mixed[0], PatternGlobCandidate)


def test_policy_include_glob_values_unchanged() -> None:
    for option in ("--include", "--exclude"):
        tokens = split_shell_tokens(f"grep {option}=*.py foo target.py")
        policy = analyze_shell_command_policy(tokens)
        assert policy.option_path_values == ()
        assert policy.positional_values is not None
        assert policy.positional_values == ("target.py",)
