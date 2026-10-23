"""Deep-review shell-policy hardening: review-verified blockers.

A deep review panel demonstrated working exploits against the shell
permission resolver; this file pins them closed with adversarial regression
tests. Every fix is fail-closed only: an unevaluable case denies, never
allows.

1. Uninspected expansion of grep-family option values. Only -e/--regexp
   values were expansion-inspected; any other value-taking option consumed
   its source token, so an unquoted glob value's expansion products beyond
   the first became grep file operands that were never path-inspected:
   ``grep --label a* -e root target.py`` resolved ALLOWED while real bash
   read an out-of-root symlinked file through the second product. Every
   value-taking option's separate-token value is now inspected exactly like
   -e/--regexp values are.
2. Exponential glob matcher. ``_match_glob_atoms`` backtracked recursively
   over every star's split points with no memoization, so a crafted pattern
   (``*a`` repeated plus an impossible suffix, against a long name) stalled
   permission resolution — measured ~3.4x growth per star, over 20 seconds
   at twelve stars. The matcher is now memoized dynamic programming, and
   ``_pattern_expansion`` carries an explicit work budget that fails closed.
3. ``**`` recursive glob under-modeled. A consecutive-star component parsed
   to two single-level star atoms, so the model matched one directory level
   while zsh (the default shell) expands ``src/**/x.py`` recursively,
   violating the model's superset invariant: the resolver resolved ALLOWED
   while real zsh passed an out-of-root symlinked file to grep as a file
   operand. A component with two or more consecutive stars is now
   unevaluable and denies.
4. Glob-bearing option WORD expanding to a bare value-taking option (round-2
   residual in the Blocker 1 family). An option-position word whose NAME
   portion carries unquoted glob metacharacters (``--file*``, ``--*``,
   ``--e*``, ``-*``, ``-v*``) is not a recognized option word — glob
   characters break ``_match_long``'s prefix matching, and a short cluster
   whose leading characters are not value shorts records no value — so the
   walk produced no candidate for the token after it. But the shell expands
   the word itself: with a decoy file named exactly like a bare value-taking
   option in the cwd, the product is that bare option, which consumes the
   next token as its value. That token was misclassified as the regex
   pattern position and excluded as data, so ``grep --file* /etc/passwd
   target.py`` resolved ALLOWED while real bash and zsh read /etc/passwd.
   The grep-family policy now fails closed on any such word: it joins the
   path candidates as a plain word the resolver's path-glob rule denies.

The file carries NO skip conditions by rule: if a shell binary is absent,
the affected tests FAIL, never skip.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import time

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_argv import (
    PatternGlobCandidate,
    split_shell_tokens,
)
from chartreux.core.tools.builtins._shell_command_policy import (
    analyze_shell_command_policy,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    _match_glob_atoms,
    _parse_glob_component,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.permissions import PermissionContext

# A readable file outside every authorized workspace root, used by the
# resolver-denial cases (mirrors the G24 grep pattern-file control).
OUT_OF_ROOT = "/etc/passwd"
# The out-of-root file's content contains this line's token, so a real-shell
# probe matches and its zero exit code proves the file was actually read.
OUT_OF_ROOT_MATCH = "root"

# The rendered path-glob denial marker: "Shell path glob candidate '<token>'
# in command segment '<part>' cannot be safely inspected".
PATH_GLOB_MARKER = "path glob candidate"


def _resolve(cwd: Path, command: str) -> PermissionContext:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    return result


def _run_shell(command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command], cwd=cwd, capture_output=True, text=True, timeout=60
    )


def test_option_value_expansion_inspected(tmp_path: Path) -> None:
    """Every value-taking grep-family option's expanded value is inspected.

    The Blocker 1 exploit shape: the shell pathname-expands the ``--label``
    value ``a*`` into ``aa ab``; the first product becomes the option's
    value and ``ab`` — an out-of-root symlink — becomes a grep file operand
    the resolver never inspected, so the command resolved ALLOWED while
    real bash read the out-of-root file (the zero exit code with a matching
    line proves the read). It must resolve NEVER. The equivalent
    fully-in-root shape with the same option stays allowed, a short
    value-taking option is inspected the same way, and the direct-operand
    and positional-glob controls keep denying.
    """
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    (hostile / "target.py").write_text("needle\n", encoding="utf-8")
    (hostile / "aa").write_text("needle\n", encoding="utf-8")
    (hostile / "ab").symlink_to(OUT_OF_ROOT)
    command = f"grep --label a* -e {OUT_OF_ROOT_MATCH} target.py"
    proc = _run_shell(command, hostile)
    assert proc.returncode == 0, (command, proc.stderr)
    context = _resolve(hostile, command)
    assert context.permission is ToolPermission.NEVER, (command, context)
    assert PATH_GLOB_MARKER in (context.reason or ""), context.reason

    # Policy layer: the option value joins the inspected path candidates as
    # a pattern glob, exactly like an unquoted -e value.
    policy = analyze_shell_command_policy(split_shell_tokens(command))
    assert policy.option_path_values == (PatternGlobCandidate("a*"),)

    # The equivalent fully-in-root shape with the same option stays allowed.
    benign = tmp_path / "benign"
    benign.mkdir()
    (benign / "target.py").write_text("needle\n", encoding="utf-8")
    (benign / "aa").write_text("needle\n", encoding="utf-8")
    (benign / "ab").write_text("needle\n", encoding="utf-8")
    context = _resolve(benign, command)
    assert context.permission is ToolPermission.ALWAYS, (command, context)

    # A short value-taking option's separate-token value is inspected too.
    short_form = f"grep -A a* -e {OUT_OF_ROOT_MATCH} target.py"
    context = _resolve(hostile, short_form)
    assert context.permission is ToolPermission.NEVER, (short_form, context)

    # Controls: the direct-operand and positional-glob shapes still deny.
    direct = f"grep {OUT_OF_ROOT_MATCH} ab"
    context = _resolve(hostile, direct)
    assert context.permission is ToolPermission.NEVER, (direct, context)
    positional = "grep a* target.py"
    context = _resolve(hostile, positional)
    assert context.permission is ToolPermission.NEVER, (positional, context)


def test_glob_matcher_bounded_polynomial(tmp_path: Path) -> None:
    """The glob matcher is polynomial and the expansion stays budgeted.

    The Blocker 2 exploit shape: the pattern ``*a`` x 12 + ``!x`` against
    the 32-char name ``a`` x 30 + ``!y`` has an impossible suffix, so the
    old backtracking matcher exhausted its whole (exponential) search space
    — measured ~3.4x growth per star, over 20 seconds at twelve stars,
    synchronously inside permission resolution. The memoized matcher
    resolves the same shape in milliseconds and denies: the pattern matches
    the out-of-root symlink ``a`` x 12 + ``!x``, so the expansion's match
    set joins the read-inspected surface. The legitimate one-level in-root
    shape stays allowed, and the matcher's atom semantics are pinned
    directly.
    """
    work = tmp_path / "work"
    work.mkdir()
    (work / "target.py").write_text("x = 1\n", encoding="utf-8")
    # The impossible suffix: the name ends "!y", so the pattern's "!x"
    # suffix can never match and a backtracking matcher must exhaust its
    # whole search space on it.
    (work / ("a" * 30 + "!y")).write_text("x = 2\n", encoding="utf-8")
    # The matching out-of-root symlink: the pattern matches this name, so
    # the expansion must deny once the matcher finishes.
    (work / ("a" * 12 + "!x")).symlink_to("/etc/hostname")
    pattern = "*a" * 12 + "!x"
    command = f"grep {pattern} target.py"
    deadline = time.monotonic() + 5.0
    context = _resolve(work, command)
    assert time.monotonic() < deadline, "the pathological pattern must resolve bounded"
    assert context.permission is ToolPermission.NEVER, (command, context)
    assert PATH_GLOB_MARKER in (context.reason or ""), context.reason

    # The matcher still accepts the legitimate one-level in-root shape.
    legit = tmp_path / "legit"
    legit.mkdir()
    (legit / "target.py").write_text("x = 1\n", encoding="utf-8")
    (legit / "axb").write_text("x = 2\n", encoding="utf-8")
    control = "grep a*b target.py"
    context = _resolve(legit, control)
    assert context.permission is ToolPermission.ALWAYS, (control, context)

    # Direct matcher semantics: stars, questions, and sets.
    def matches(component: str, name: str) -> bool:
        atoms = _parse_glob_component(component)
        assert atoms is not None, component
        return _match_glob_atoms(name, atoms)

    assert matches("a*b", "axb")
    assert matches("a*b", "ab")
    assert not matches("a*xb", "ayb")
    assert matches("a?b", "axb")
    assert not matches("a?b", "ab")
    assert matches("[ab]c", "bc")
    assert not matches("[ab]c", "cc")
    assert matches("[!ab]c", "cc")
    assert not matches("[!ab]c", "ac")
    assert matches("*", "anything")
    assert matches("a*", "a")


def test_recursive_globstar_fails_closed(tmp_path: Path) -> None:
    """A consecutive-star component fails closed instead of matching one level.

    The Blocker 3 exploit shape: ``**`` parsed to two single-level star
    atoms, so the model inspected only ``src/0/x.py`` while zsh (the
    default shell) expands ``src/**/x.py`` recursively to BOTH
    ``src/0/x.py`` and ``src/a/b/x.py`` — the second an out-of-root
    symlinked file that becomes a grep file operand. The resolver resolved
    ALLOWED; it must resolve NEVER. The plain single-star directory
    pattern stays allowed, and the parse layer fails closed on any
    consecutive-star run.
    """
    work = tmp_path / "work"
    work.mkdir()
    (work / "src" / "0").mkdir(parents=True)
    (work / "src" / "0" / "x.py").write_text("x = 1\n", encoding="utf-8")
    deep = work / "src" / "a" / "b"
    deep.mkdir(parents=True)
    (deep / "x.py").symlink_to("/etc/hostname")
    exploit = "grep src/**/x.py"
    # Real-shell probe: zsh's recursive expansion passes both paths to the
    # command, the second one the out-of-root symlinked file.
    proc = subprocess.run(
        ["zsh", "-c", "print -l src/**/x.py"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == ["src/0/x.py", "src/a/b/x.py"], proc.stdout
    context = _resolve(work, exploit)
    assert context.permission is ToolPermission.NEVER, (exploit, context)
    assert PATH_GLOB_MARKER in (context.reason or ""), context.reason

    # The plain single-star directory pattern still resolves allowed.
    control = "grep src/*/x.py"
    context = _resolve(work, control)
    assert context.permission is ToolPermission.ALWAYS, (control, context)

    # The parse layer fails closed on any consecutive-star run (the
    # accepted conservative trade-off: intra-segment ``a**b`` over-denies).
    assert _parse_glob_component("**") is None
    assert _parse_glob_component("***") is None
    assert _parse_glob_component("a**b") is None
    assert _parse_glob_component("*a*") is not None


def test_glob_bearing_option_word_fails_closed(tmp_path: Path) -> None:
    """A glob-bearing unrecognized option word fails closed.

    The round-2 residual in the Blocker 1 family. An option-position word
    whose NAME portion carries unquoted glob metacharacters (``--file*``,
    ``--*``, ``--e*``, ``-*``, ``-v*``) is not a recognized option word: glob
    characters break ``_match_long``'s prefix matching, and a short cluster
    whose leading characters are not value shorts records no value, so the
    walk produced no candidate for the token after it. But the shell expands
    the option word itself: with a decoy file named exactly like a bare
    value-taking option in the cwd, the product is that bare option, which
    consumes the next token as its value. That token was misclassified by
    ``_grep_positional_values`` as the regex pattern position and excluded as
    data — never path-inspected — so ``grep --file* /etc/passwd target.py``
    resolved ALLOWED while real bash and zsh read /etc/passwd. The
    grep-family policy now fails closed on any such word: it joins the path
    candidates as a plain word the resolver's path-glob rule denies. The
    attached-value globs stay allowed (the glob is confined to the attached
    VALUE, so every expansion product keeps the option prefix), and the
    fully quoted name-portion control keeps its verified verdict: the shell
    passes the literal word through, grep rejects the unknown option, and no
    file is read.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    # The out-of-workspace pattern file's content is what the -f probes
    # match, so their zero exit code proves the read.
    (outside / "patterns.txt").write_text("needle\n", encoding="utf-8")
    # The exclude-from file's content matches nothing; a failed open would
    # exit 2 with an error, so the probes' zero exit code proves the read.
    (outside / "excludes.txt").write_text("zzz\n", encoding="utf-8")
    patterns = "../outside/patterns.txt"
    excludes = "../outside/excludes.txt"

    def make_work(name: str, decoys: tuple[str, ...]) -> Path:
        work = tmp_path / name
        work.mkdir()
        (work / "target.py").write_text("needle\nplain\n", encoding="utf-8")
        (work / "needle").write_text("needle\n", encoding="utf-8")
        for decoy in decoys:
            (work / decoy).write_text("decoy\n", encoding="utf-8")
        return work

    # (directory, decoys, command, expected probe stdout): the decoy file is
    # named exactly like the bare value-taking option the glob-bearing word
    # expands to, so the product consumes the out-of-workspace value.
    shapes = [
        ("file-word", ("--file",), f"grep --file* {patterns} target.py", "needle\n"),
        ("dash-word", ("-f",), f"grep -* {patterns} target.py", "needle\n"),
        (
            "e-word",
            ("--exclude", "--exclude-dir", "--exclude-from"),
            f"grep --e* {excludes} needle target.py",
            "needle\n",
        ),
        ("v-word", ("-vf",), f"grep -v* {patterns} target.py", "plain\n"),
        (
            "star-word",
            ("--exclude", "--exclude-dir", "--exclude-from"),
            f"grep --* {excludes} needle target.py",
            "needle\n",
        ),
        (
            "quoted-prefix-word",
            ("--file",),
            f"grep '--file'* {patterns} target.py",
            "needle\n",
        ),
    ]
    for name, decoys, command, expected_stdout in shapes:
        work = make_work(name, decoys)
        # Real-shell probes: both shells expand the option word to the bare
        # decoy option, which reads the out-of-workspace value — the zero
        # exit code with the expected output proves the read.
        for shell in ("bash", "zsh"):
            proc = subprocess.run(
                [shell, "-c", command],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert proc.returncode == 0, (shell, command, proc.stderr)
            assert proc.stdout == expected_stdout, (shell, command, proc.stdout)
            assert proc.stderr == "", (shell, command, proc.stderr)
        context = _resolve(work, command)
        assert context.permission is ToolPermission.NEVER, (command, context)
        assert PATH_GLOB_MARKER in (context.reason or ""), context.reason

    # Policy layer: the glob-bearing option word joins the inspected path
    # candidates as a PLAIN word (never a PatternGlobCandidate), so the
    # resolver's path-glob rule denies it — the same fail-closed mechanism
    # the mixed-quote pattern rule uses.
    command = f"grep --file* {patterns} target.py"
    policy = analyze_shell_command_policy(split_shell_tokens(command))
    assert policy.option_path_values == ("--file*",)
    assert not isinstance(policy.option_path_values[0], PatternGlobCandidate)

    # The attached-value globs stay allowed: the glob is confined to the
    # attached VALUE, so every expansion product keeps the option prefix and
    # never becomes a bare option or a file operand.
    control = make_work("attached-controls", ())
    for filler in ("aa", "ab", "5a", "5b"):
        (control / filler).write_text("x = 1\n", encoding="utf-8")
    for command in (
        "grep --label=a* needle target.py",
        "grep --lab=a* needle target.py",
        "grep -Aa* needle target.py",
        "grep -m5* needle target.py",
        "grep --include=*.py needle target.py",
    ):
        context = _resolve(control, command)
        assert context.permission is ToolPermission.ALWAYS, (command, context)
        policy = analyze_shell_command_policy(split_shell_tokens(command))
        assert policy.option_path_values == (), command

    # The fully quoted name-portion control keeps its verified verdict: the
    # shell passes the literal word through (a quoted glob never expands),
    # so grep rejects the unrecognized option and reads nothing — the same
    # ALLOWED any unrecognized option word gets. The fail-closed rule is
    # provenance-aware and does not fire on a quoted glob.
    quoted = f"grep '--file*' {patterns} needle target.py"
    proc = _run_shell(quoted, control)
    assert proc.returncode == 2, (quoted, proc.stderr)
    assert "unrecognized option" in proc.stderr, (quoted, proc.stderr)
    context = _resolve(control, quoted)
    assert context.permission is ToolPermission.ALWAYS, (quoted, context)
    policy = analyze_shell_command_policy(split_shell_tokens(quoted))
    assert policy.option_path_values == (), quoted
