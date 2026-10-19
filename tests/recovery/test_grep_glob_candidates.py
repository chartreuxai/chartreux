r"""G0 characterization + G1 regression: quoted grep regex operands vs the path-glob heuristic.

Track G0/G1 of the approved v0.5.6 plan (state.md, "G0/G1 — Reproduce first;
grammar fix only if demonstrated"; design source handoff §6.4). G0
characterized the false positive and landed this file; G1 fixed it at the
policy layer, and this file now carries the regression matrix.

G0 VERDICT — the false positive REPRODUCED, but not for plain alternation. The
handoff §6.4 first example, ``grep -E 'a|b'``, was ALLOWED in pre-G1 source:
the path-glob heuristic triggers on the glob metacharacters ``*``, ``?`` and
complete bracket classes, not on alternation itself. The false positive fired
when a quoted grep PATTERN operand (the first positional, when no ``-e`` /
``--regexp`` / ``-f`` supplies the pattern) contained any of those
metacharacters — including alternations that merely contain ``.*``. That is
exactly the reviewers' live denial shape: session 20261017_185339_075619cc,
reviewer_20261017_201803_0c84d9cc ran
``grep -n -i -m 15 -E 'release|v0\.5\.5.*tag|CHANGELOG|version bump|publish|push' ...``
and reviewer_20261017_192455_618dd7e1 ran
``grep -rn "class .*TurnParams\|injected: bool" ...``; both were denied with
"Shell path glob candidate ... cannot be safely inspected". The trigger was
the resolver's path-glob candidate heuristic, not a different denial family
and not argv normalization (quoting is normalized away before the heuristic;
the heuristic sees the bare pattern value).

G0 classification path (worktree feature/v056-track-g0 @ 545e271):

1. ``Bash.resolve_permission`` (chartreux/core/tools/builtins/bash.py:180-187)
   constructs ``ShellPermissionResolver`` and calls ``resolve_permission``
   (chartreux/core/tools/builtins/_shell_permission_resolver.py:1551).
2. grep is a path command: ``_READ_ONLY_COMMANDS_POSIX`` contains ``"grep"``
   (_shell_permission_resolver.py:420) and ``_PATH_COMMANDS``
   (_shell_permission_resolver.py:605-611) gates positional inspection, so
   ``path_candidates(..., inspect_positional_paths=True)`` runs
   (_shell_permission_resolver.py:1182-1186).
3. Pre-G1, the grep policy returned ``positional_values=parsed.positionals``
   (_shell_command_policy.py:1582 -> ``_standard_policy``, field set at the
   return): the PATTERN was positional #1, so it became a path candidate
   (``path_candidates``). ``-e`` / ``--regexp`` consume the pattern as an
   option value instead (_shell_argv.py:``_scan_options``), which is why
   those forms never trigger. ``-f`` / ``--file`` / ``--exclude-from``
   values are option path values (policy ``path_options``), which is why
   genuine globs there stay denied.
4. ``_GuardrailPart.access`` is ``None`` for a regular command part
   (_shell_permission_resolver.py:175-181), so the glob heuristic fired
   (_shell_permission_resolver.py:1206-1227) on any candidate containing
   ``*`` or ``?`` or a complete bracket class
   (``_has_active_bracket_glob``, _shell_permission_analysis.py:897-929),
   unless it is an ``ls`` scoped listing glob or a ``find -name`` /
   ``-iname`` / ``-path`` / ``-ipath`` operand.
5. Denial: ``Diagnostic("path_glob")``, rendered at _shell_diagnostics.py:143-147
   as "Shell path glob candidate '<token>' in command segment '<part>' cannot
   be safely inspected".

G1 RESULT — the false positive is FIXED at the policy layer
(_shell_command_policy.py, ``_standard_policy`` grep branch): when no
``-e`` / ``--regexp`` / ``-f`` / ``--file`` option supplies the pattern, the
first positional is the regex PATTERN, not a path, so it is excluded from
``positional_values`` (``_grep_positional_values``); only the remaining
positionals are path candidates. The resolver's glob heuristic itself is
unchanged. The 22 pattern cases G0 marked as known bugs are flipped to
``allowed`` below and pinned by ``_G1_PATTERN_OPERAND_IDS``; the 19
protection cases in ``_G1_PROTECTION_IDS`` stay denied.

Recorded diagnostic categories (one per matrix row):

- ``allowed`` — the resolver returned ``ToolPermission.ALWAYS``.
- ``denied-path-glob-candidate`` — ``NEVER`` with the ``path_glob``
  diagnostic above.
- ``denied-other`` — ``NEVER`` with a different diagnostic (observed:
  outside-workspace path authority).

No case raised, so the ``error`` category of the G0 vocabulary is empty.

Out-of-scope observations recorded for FUTURE scope decisions (NOT G1):
egrep/fgrep positionals are entirely uninspected (escape both protection
and false positive), and grep's ``--include`` / ``--exclude`` glob values are
not path candidates at all (a false NEGATIVE, separate from this gate).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    analyze_shell_command_policy,
    path_candidates,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig

ALLOWED = "allowed"
DENIED_PATH_GLOB = "denied-path-glob-candidate"
DENIED_OTHER = "denied-other"

# (case id, command, expected category, note)
_CASE = tuple[str, str, str, str]

# --- Quoted regex PATTERN operands (positional pattern position) -----------
_REGEX_PATTERN_CASES: list[_CASE] = [
    ("A1", "grep -E 'a|b' target.py", ALLOWED, "handoff §6.4 ex. 1: NOT a trigger"),
    ("A2", "grep -E 'a|b|c' target.py", ALLOWED, "plain alternation, 3 branches"),
    (
        "A3",
        "grep -E 'release|v0\\.5\\.5.*tag|CHANGELOG' target.py",
        ALLOWED,
        "G1 fixed: alternation containing '.*' — reviewer live shape",
    ),
    (
        "A4",
        'grep -rn "class .*TurnParams\\|injected: bool" target.py',
        ALLOWED,
        "G1 fixed: double-quoted '.*' — reviewer live shape",
    ),
    (
        "A5",
        "grep '[0-9]' target.py",
        ALLOWED,
        "G1 fixed: handoff §6.4 ex. 2 — quoted bracket class",
    ),
    ("A6", 'grep "[0-9]" target.py', ALLOWED, "G1 fixed: double-quoted bracket class"),
    ("A7", "grep 'a[bc]d' target.py", ALLOWED, "G1 fixed: bracket class in pattern"),
    (
        "A8",
        "grep 'colou?r' target.py",
        ALLOWED,
        "G1 fixed: '?' metacharacter in pattern",
    ),
    ("A9", "grep 'x{2,3}' target.py", ALLOWED, "braces are not glob metacharacters"),
    ("A10", "grep -E 'a|b'", ALLOWED, "pattern only, reads stdin"),
    ("A11", "grep 'a\\|b' target.py", ALLOWED, "basic-regex alternation"),
    ("A12", "grep 'a*' target.py", ALLOWED, "G1 fixed: '*' metacharacter in pattern"),
    ("A13", "grep -E 'a|b' target.py | wc -l", ALLOWED, "pipeline, benign pattern"),
    (
        "A14",
        "grep -E 'a.*b' target.py | wc -l",
        ALLOWED,
        "G1 fixed: pipeline, '.*' pattern",
    ),
    ("A15", 'grep -E "a|b" target.py', ALLOWED, "double-quoted plain alternation"),
]

# --- -e / --regexp forms (the pattern is an option VALUE, not positional) --
_REGEXP_OPTION_CASES: list[_CASE] = [
    ("B1", "grep -e 'a|b' target.py", ALLOWED, "pattern consumed by -e"),
    ("B2", "grep -e '[0-9]' target.py", ALLOWED, "pattern consumed by -e"),
    ("B3", "grep -e 'a.*b' target.py", ALLOWED, "pattern consumed by -e"),
    ("B4", "grep --regexp '[0-9]' target.py", ALLOWED, "pattern consumed by --regexp"),
    ("B5", "grep --regexp='[0-9]' target.py", ALLOWED, "attached --regexp= form"),
    ("B6", "grep -e'[0-9]' target.py", ALLOWED, "attached short -e form"),
    (
        "B7",
        "grep -eE '[0-9]' target.py",
        DENIED_PATH_GLOB,
        "protection: getopt attaches 'E' to -e, so '[0-9]' is the FILE operand",
    ),
]

# --- -f / --file / --exclude-from path options ------------------------------
_PATH_OPTION_CASES: list[_CASE] = [
    ("C1", "grep -f patterns.txt target.py", ALLOWED, "existing pattern file"),
    ("C2", "grep --file patterns.txt target.py", ALLOWED, "existing pattern file"),
    ("C3", "grep --exclude-from skip.txt target.py", ALLOWED, "existing exclude file"),
    (
        "C4",
        "grep -f '*.txt' target.py",
        DENIED_PATH_GLOB,
        "protection: genuine glob in a path option",
    ),
    (
        "C5",
        "grep --file 'pat*.txt' target.py",
        DENIED_PATH_GLOB,
        "protection: genuine glob in a path option",
    ),
    ("C6", "grep -f missing.txt target.py", ALLOWED, "missing file resolves in cwd"),
    (
        "C7",
        "grep -f 'pat[0-9].txt' target.py",
        DENIED_PATH_GLOB,
        "protection: bracket glob in a path option",
    ),
]

# --- "--" separator forms ----------------------------------------------------
_SEPARATOR_CASES: list[_CASE] = [
    ("D1", "grep -- 'a|b' target.py", ALLOWED, "'--' then benign alternation"),
    (
        "D2",
        "grep -- '[0-9]' target.py",
        ALLOWED,
        "G1 fixed: '--' then bracket-class pattern",
    ),
    (
        "D3",
        "grep -E -- 'a.*b' target.py",
        ALLOWED,
        "G1 fixed: '-E --' then '.*' pattern",
    ),
    ("D4", "grep -E 'a|b' -- target.py", ALLOWED, "benign pattern before '--'"),
    ("D5", "grep --regexp 'a|b' -- target.py", ALLOWED, "-e pattern before '--'"),
    ("D6", "grep -f patterns.txt -- target.py", ALLOWED, "-f before '--'"),
]

# --- Genuine path-glob operands (the protection that must survive G1) -------
_GENUINE_GLOB_CASES: list[_CASE] = [
    ("E1", "grep foo '*.py'", DENIED_PATH_GLOB, "protection: quoted file-operand glob"),
    (
        "E2",
        "grep foo src/*.py",
        DENIED_PATH_GLOB,
        "protection: unquoted file-operand glob",
    ),
    (
        "E3",
        "grep foo '*.py' target.py",
        DENIED_PATH_GLOB,
        "protection: file-operand glob with pattern",
    ),
    ("E4", "ls *.py", ALLOWED, "scoped listing-glob exemption (resolver :1215)"),
    ("E5", "ls src/*.py", ALLOWED, "scoped listing-glob exemption"),
    ("E6", "ls ../*.py", DENIED_PATH_GLOB, "protection: unscoped parent directory"),
    ("E7", "ls /tmp/*.py", DENIED_PATH_GLOB, "protection: outside-workspace glob"),
    ("E8", "cat *.py", DENIED_PATH_GLOB, "protection: non-ls reader glob"),
    (
        "E9",
        "ls src/[a-c]*.py",
        DENIED_PATH_GLOB,
        "protection: bracket globs are never supported_glob for ls",
    ),
    (
        "E10",
        "grep -f patterns.txt '*.py'",
        DENIED_PATH_GLOB,
        "protection: genuine glob file operand with -f",
    ),
    (
        "E11",
        "grep -E 'a|b' '*.py'",
        DENIED_PATH_GLOB,
        "protection: benign pattern + genuine file glob",
    ),
    (
        "E12",
        "grep 'a*' '*.py'",
        DENIED_PATH_GLOB,
        "protection (mixed): pattern 'a*' is regex data after G1, but the "
        "genuine file glob keeps this denied (offending token becomes '*.py')",
    ),
]

# --- find -name/-iname/-path/-ipath contrast --------------------------------
_FIND_CASES: list[_CASE] = [
    ("F1", "find . -name '*.py'", ALLOWED, "find -name operands are skipped (:1191)"),
    ("F2", "find . -iname '*.py'", ALLOWED, "find -iname operands are skipped"),
    ("F3", "find . -path '*/tests/*'", ALLOWED, "find -path operands are skipped"),
    ("F4", "find . -name '[0-9].py'", ALLOWED, "find -name bracket operand skipped"),
    ("F5", "find . -ipath '*.Py'", ALLOWED, "find -ipath operands are skipped"),
    (
        "F6",
        "find '*.py' -name x",
        DENIED_PATH_GLOB,
        "protection: positional starting-point glob (not after -name)",
    ),
    ("F7", "find src -name '*.py'", ALLOWED, "find -name with directory argument"),
]

# --- Misc contrast: flags, aliases, uninspected options, scope --------------
_CONTRAST_CASES: list[_CASE] = [
    ("G1", "grep 'foo.bar' target.py", ALLOWED, "dot is not a glob metacharacter"),
    ("G2", "grep -w 'a|b' target.py", ALLOWED, "flag + benign alternation"),
    (
        "G3",
        "grep -rn '[0-9]' target.py",
        ALLOWED,
        "G1 fixed: flags + bracket-class pattern",
    ),
    (
        "G4",
        "grep -n -i -m 15 -E 'release|v0\\.5\\.5.*tag|CHANGELOG' target.py",
        ALLOWED,
        "G1 fixed: exact reviewer command shape",
    ),
    (
        "G5",
        "grep -E a|b target.py",
        ALLOWED,
        "unquoted '|' splits the pipeline (grammar family, not a denial)",
    ),
    ("G6", "grep -E '?' target.py", ALLOWED, "G1 fixed: lone '?' pattern"),
    ("G7", "grep -E '[abc]' target.py", ALLOWED, "G1 fixed: simple bracket class"),
    ("G8", "grep -E '[!abc]' target.py", ALLOWED, "G1 fixed: negated bracket class"),
    ("G9", "grep -E '[]a]' target.py", ALLOWED, "G1 fixed: leading-] bracket class"),
    ("G10", "grep -E '[abc' target.py", ALLOWED, "unclosed bracket is not active"),
    ("G11", "grep -C 2 -E 'a|b' target.py", ALLOWED, "value-taking flag + pattern"),
    (
        "G12",
        "grep -r '[0-9]' src",
        ALLOWED,
        "G1 fixed: recursive + bracket pattern, dir operand",
    ),
    ("G13", "grep -rn '[0-9]'", ALLOWED, "G1 fixed: bracket pattern, no file operand"),
    ("G14", "grep -F '[0-9]' target.py", ALLOWED, "G1 fixed: -F fixed-string pattern"),
    ("G15", "grep -G '[0-9]' target.py", ALLOWED, "G1 fixed: -G basic-regex pattern"),
    ("G16", "grep -P '[0-9]' target.py", ALLOWED, "G1 fixed: -P perl-regex pattern"),
    (
        "G17",
        "grep '[0-9]' src",
        ALLOWED,
        "G1 fixed: bracket pattern + directory operand",
    ),
    (
        "G18",
        "egrep '[0-9]' target.py",
        ALLOWED,
        "observation: egrep is not in _PATH_COMMANDS — positionals uninspected",
    ),
    (
        "G19",
        "fgrep 'a*' target.py",
        ALLOWED,
        "observation: fgrep is not in _PATH_COMMANDS — positionals uninspected",
    ),
    (
        "G20",
        "grep --include='*.py' foo .",
        ALLOWED,
        "observation (false negative, NOT a G1 flip target): --include glob "
        "values are not path candidates",
    ),
    (
        "G21",
        "grep --exclude='*.py' foo target.py",
        ALLOWED,
        "observation (false negative, NOT a G1 flip target): --exclude likewise",
    ),
    (
        "G22",
        "grep -E 'a|b' /dev/null",
        DENIED_OTHER,
        "protection: outside-workspace file operand",
    ),
    (
        "G23",
        "grep -e 'a|b' /dev/null",
        DENIED_OTHER,
        "protection: outside-workspace file operand",
    ),
    (
        "G24",
        "grep -f /etc/hosts target.py",
        DENIED_OTHER,
        "protection: outside-workspace pattern file",
    ),
    (
        "G25",
        "grep -e 'foo' '[0-9].py'",
        DENIED_PATH_GLOB,
        "protection: bracket-glob FILE operand (pattern supplied via -e)",
    ),
]

_CASES: list[_CASE] = [
    *_REGEX_PATTERN_CASES,
    *_REGEXP_OPTION_CASES,
    *_PATH_OPTION_CASES,
    *_SEPARATOR_CASES,
    *_GENUINE_GLOB_CASES,
    *_FIND_CASES,
    *_CONTRAST_CASES,
]
_BY_ID: dict[str, _CASE] = {row[0]: row for row in _CASES}

# G1 REGRESSION — the quoted grep REGEX PATTERN operands that G0 demonstrated
# being denied as "path glob candidates" (the false positive). G1 fixed the
# policy layer: grep's first positional is the PATTERN, not a path, so it is
# no longer a path candidate. These rows must stay ALLOWED; flipping any of
# them back to a denial re-opens the demonstrated false positive.
_G1_PATTERN_OPERAND_IDS = frozenset({
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A12",
    "A14",
    "D2",
    "D3",
    "G3",
    "G4",
    "G6",
    "G7",
    "G8",
    "G9",
    "G12",
    "G13",
    "G14",
    "G15",
    "G16",
    "G17",
})

# G1 GATE — protections that must survive the grammar fix: genuine path
# globs (file operands, path-option values, unscoped or outside-workspace
# globs, positional find starting points, the -eE getopt-attached form)
# stay denied, and outside-workspace operands stay denied with their own
# diagnostic. G1 must not flip any of these to ALLOWED.
_G1_PROTECTION_IDS = frozenset({
    "B7",
    "C4",
    "C5",
    "C7",
    "E1",
    "E2",
    "E3",
    "E6",
    "E7",
    "E8",
    "E9",
    "E10",
    "E11",
    "E12",
    "F6",
    "G22",
    "G23",
    "G24",
    "G25",
})


@pytest.fixture
def bash_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Bash:
    """Self-contained copy of the tests/tools bash_tool fixture, plus files.

    Inlined so this file has no conftest dependency: a parallel package seeds
    tests/recovery/conftest.py, and G0 tests must not inherit its fixtures.
    The file set matches the G0 probe environment (target.py, patterns.txt,
    skip.txt, src/mod.py) so listing-glob outcomes are deterministic.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "target.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "patterns.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("b\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("y = 2\n", encoding="utf-8")
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    return Bash(config_getter=lambda: config, state=BaseToolState())


def _outcome(bash_tool: Bash, command: str) -> str:
    """Run one command through the real resolver and classify the verdict."""
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, f"resolver returned no context for: {command}"
    if context.permission is ToolPermission.ALWAYS:
        return ALLOWED
    reason = context.reason or ""
    if "path glob candidate" in reason:
        return DENIED_PATH_GLOB
    return DENIED_OTHER


@pytest.mark.parametrize(
    ("case_id", "command", "expected", "note"), _CASES, ids=[row[0] for row in _CASES]
)
def test_characterization_matrix(
    bash_tool: Bash, case_id: str, command: str, expected: str, note: str
) -> None:
    """The G0 record: each operand class -> actual resolver outcome."""
    assert _outcome(bash_tool, command) == expected, f"{case_id}: {note}"


def test_g1_pattern_operands_are_allowed(bash_tool: Bash) -> None:
    """G1 REGRESSION — the fixed false positive stays fixed.

    Every case here is a quoted grep REGEX PATTERN operand that pre-G1 source
    denied as a "path glob candidate" (the G0-demonstrated false positive:
    grep's first positional is a pattern, not a path). The G1 policy-layer fix
    excludes the pattern position from ``positional_values``, so each of
    these must resolve to ALLOWED. A denial here re-opens the false positive.
    """
    for case_id in sorted(_G1_PATTERN_OPERAND_IDS):
        command = _BY_ID[case_id][1]
        assert _outcome(bash_tool, command) == ALLOWED, case_id


def test_g1_protections_stay_denied(bash_tool: Bash) -> None:
    """G1 GATE — the protections that must survive the grammar fix.

    Genuine path globs (file operands, path-option values, unscoped or
    outside-workspace globs, positional find starting points, and the -eE
    getopt-attached form) stay denied; outside-workspace operands stay denied
    with their own diagnostic. The G1 fix must not flip any of these to
    ALLOWED.
    """
    for case_id in sorted(_G1_PROTECTION_IDS):
        command = _BY_ID[case_id][1]
        assert _outcome(bash_tool, command) != ALLOWED, case_id


def test_matrix_partition_is_consistent() -> None:
    """The record is structurally complete: ids unique, sets disjoint.

    Every allowed row is either a benign control or one of the G1-fixed
    pattern operands; every non-allowed row is a G1 protection. The fixed
    pattern-operand set is non-empty — the G0-demonstrated false positive —
    and the protection set is non-empty — the protections that survived G1.
    """
    ids = [row[0] for row in _CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    assert _G1_PATTERN_OPERAND_IDS, "the G1 regression set must be non-empty"
    assert _G1_PROTECTION_IDS, "the G1 protection set must be non-empty"
    assert _G1_PATTERN_OPERAND_IDS <= set(ids)
    assert _G1_PROTECTION_IDS <= set(ids)
    assert not (_G1_PATTERN_OPERAND_IDS & _G1_PROTECTION_IDS)
    for case_id in _G1_PATTERN_OPERAND_IDS:
        assert _BY_ID[case_id][2] == ALLOWED, case_id
    for case_id in _G1_PROTECTION_IDS:
        assert _BY_ID[case_id][2] != ALLOWED, case_id
    denied = {row[0] for row in _CASES if row[2] != ALLOWED}
    assert denied == _G1_PROTECTION_IDS


def test_policy_layer_treats_grep_pattern_positional_as_pattern() -> None:
    """Pin the G1 mechanism: the pattern position is not a path candidate.

    ``_standard_policy("grep", ...)`` now returns the positionals AFTER the
    regex pattern as ``positional_values``
    (_shell_command_policy.py:``_grep_positional_values``): the first
    positional — the regex pattern — is excluded whenever no ``-e`` /
    ``--regexp`` / ``-f`` / ``--file`` option supplies the pattern, so it
    never reaches ``path_candidates``. File operands and path-option values
    stay inspected, and a pattern-supplying option keeps every positional a
    file operand.
    """
    policy = analyze_shell_command_policy(["grep", "-E", "[0-9]", "target.py"])
    assert policy.positional_values == ("target.py",)
    assert path_candidates(
        ["grep", "-E", "[0-9]", "target.py"], inspect_positional_paths=True
    ) == ("target.py",)
    # A pattern option supplies the regex, so every positional is a file
    # operand and stays a path candidate.
    for option in ("-e", "--regexp"):
        policy = analyze_shell_command_policy(["grep", option, "[0-9]", "target.py"])
        assert policy.positional_values == ("target.py",)
    policy = analyze_shell_command_policy(["grep", "-f", "patterns.txt", "target.py"])
    assert policy.option_path_values == ("patterns.txt",)
    assert policy.positional_values == ("target.py",)
    # getopt attaches 'E' to -e, so the pattern is supplied and '[0-9]' is
    # the FILE operand: it stays a path candidate (protection B7).
    attached = analyze_shell_command_policy(["grep", "-eE", "[0-9]", "target.py"])
    assert attached.positional_values == ("[0-9]", "target.py")
    # A genuine file-operand glob stays a path candidate (protection E1/E12).
    assert path_candidates(["grep", "foo", "*.py"], inspect_positional_paths=True) == (
        "*.py",
    )
    # Separate observation (false NEGATIVE, not the G0 false positive): grep's
    # --include/--exclude glob values are not path candidates at all, so they
    # are never inspected. Recorded for future scoping; not a G1 flip target.
    # ('foo' is the pattern position and is excluded after G1; '.' is the
    # only file operand.)
    candidates = path_candidates(
        ["grep", "--include=*.py", "foo", "."], inspect_positional_paths=True
    )
    assert candidates == (".",)


def test_handoff_plain_alternation_example_is_not_a_trigger(bash_tool: Bash) -> None:
    """Correct handoff §6.4's first example against current source.

    ``grep -E 'a|b'`` (plain quoted alternation, no glob metacharacters) is
    ALLOWED today: the path-glob heuristic triggers on ``*``, ``?`` and
    complete bracket classes, not on alternation. The reviewers' live
    denials had patterns containing ``.*`` (see A3/A4/G4). E4 amendment
    request: reword handoff §6.4 and the plan's G0 rationale accordingly.
    """
    assert _outcome(bash_tool, "grep -E 'a|b' target.py") == ALLOWED
    assert _outcome(bash_tool, 'grep -E "a|b" target.py') == ALLOWED


def test_path_glob_denial_renders_the_path_glob_diagnostic(bash_tool: Bash) -> None:
    """The denial category is ``Diagnostic("path_glob")``.

    Rendered at _shell_diagnostics.py:143-147 as "Shell path glob candidate
    '<token>' in command segment '<part>' cannot be safely inspected". The
    triggering operand here is a genuine file-operand glob (the pattern
    position is regex data after G1 and no longer triggers).
    """
    context = bash_tool.resolve_permission(BashArgs(command="grep foo '*.py'"))
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.reason is not None
    assert "Shell path glob candidate" in context.reason
    assert "cannot be safely inspected" in context.reason
