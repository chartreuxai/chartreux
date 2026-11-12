"""S2: glob execution binding for the ``cat``/``wc`` whitelist.

Before S2 every glob-bearing command operand was denied: the resolver modeled
the shell's pathname expansion, but the shell still executed the ORIGINAL
command string, so the inspected match set was only a preflight superset. S2
binds a whitelisted glob call to an immutable, fully literal argv plan: the
same classifier produces the plan at permission time and the execution
transport consumes it (never the original command string), so the words that
run are exactly the words that were authorized.

The whitelist is exactly: a standalone ``cat`` or ``wc``, no options (an
optional ``--`` terminator only), and path operands that are each literal or
one single-component ``*``/``?`` glob with a literal, authorized parent. Every
other glob-bearing operand keeps the existing path-glob denial, and the
``ls``-listing and grep-pattern glob admissions shipping today are untouched
(they are pinned by their own tests; the classifier-level pin below asserts
they never enter this transport).

The execution side re-derives the plan immediately before spawning (the
checkout precedent), so a mutation between the permission check and execution
is seen and denied; the transport then re-opens every planned operand through
a component-wise ``O_NOFOLLOW`` walk and refuses a swapped inode. It never
falls back to the shell: an unplannable whitelist call raises instead.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
from typing import Any

import pytest

from chartreux.core.background_jobs import BashStartArgs
from chartreux.core.tools.base import (
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.builtins._shell_argument_positions import (
    is_read_only_argument_position,
    read_only_positions,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    _GLOB_WHITELIST_COMMANDS,
    GlobExecutionPlan,
    GlobPlanDenial,
    QuotedWordPlanDenial,
    glob_execution_plan_for_command,
    quoted_word_execution_plan_for_command,
)
import chartreux.core.tools.builtins.bash as bash_module
from chartreux.core.tools.builtins.bash import (
    Bash,
    BashArgs,
    BashToolConfig,
    _require_glob_plan_containment,
)
from chartreux.core.tools.builtins.bash_start import BashStart
from chartreux.core.tools.permissions import PermissionContext
from tests.mock.utils import collect_result


def _tool(cwd: Path) -> Bash:
    return Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)


def _permission(cwd: Path, command: str) -> PermissionContext:
    context = _tool(cwd).resolve_permission(BashArgs(command=command))
    assert context is not None, command
    return context


def _plan(cwd: Path, command: str) -> GlobExecutionPlan | None:
    tool = _tool(cwd)
    return glob_execution_plan_for_command(
        command, cwd=tool.cwd, workspace=tool.workspace, authority=tool.path_authority
    )


@pytest.fixture
def work(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    (root / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    (root / "beta.txt").write_text("beta\n", encoding="utf-8")
    (root / "file.txt").write_text("file\n", encoding="utf-8")
    (root / "with space.txt").write_text("spaced\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "gamma.txt").write_text("gamma\n", encoding="utf-8")
    return root


@pytest.fixture
def matrix(tmp_path: Path) -> Path:
    """Every denied operand class, in one directory tree."""
    root = tmp_path / "matrix"
    root.mkdir()
    (root / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    (root / "beta.txt").write_text("beta\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "gamma.txt").write_text("gamma\n", encoding="utf-8")
    (root / "dir.txt").mkdir()
    (root / ".hidden.txt").write_text("hidden\n", encoding="utf-8")
    (root / "-x.txt").write_text("dash\n", encoding="utf-8")
    (root / "link.txt").symlink_to(root / "alpha.txt")
    (root / "linkdir").symlink_to(root / "sub")
    os.mkfifo(root / "pipe.fifo")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "escape.txt").write_text("escape\n", encoding="utf-8")
    return root


# --- positive admission boundary ---------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        pytest.param(
            "cat *.txt",
            ("cat", "alpha.txt", "beta.txt", "file.txt", "with space.txt"),
            id="bare_single_component-star",
        ),
        pytest.param(
            "cat alpha.txt *.txt",
            ("cat", "alpha.txt", "alpha.txt", "beta.txt", "file.txt", "with space.txt"),
            id="literal-plus-glob",
        ),
        pytest.param("cat sub/*.txt", ("cat", "sub/gamma.txt"), id="literal_parent"),
        pytest.param(
            "cat ./sub/*.txt", ("cat", "./sub/gamma.txt"), id="dot_relative_parent"
        ),
        pytest.param("cat fi?e.txt", ("cat", "file.txt"), id="question-mark"),
        pytest.param(
            "cat -- *.txt",
            ("cat", "alpha.txt", "beta.txt", "file.txt", "with space.txt"),
            id="option-terminator",
        ),
        pytest.param(
            "cat 'alpha.txt' *.txt",
            ("cat", "alpha.txt", "alpha.txt", "beta.txt", "file.txt", "with space.txt"),
            id="quoted_literal-plus-glob",
        ),
        pytest.param(
            "wc *.txt",
            ("wc", "alpha.txt", "beta.txt", "file.txt", "with space.txt"),
            id="wc-whitelist",
        ),
    ],
)
@pytest.mark.asyncio
async def test_whitelist_glob_call_binds_a_literal_plan_and_executes(
    work: Path, command: str, expected: tuple[str, ...]
) -> None:
    plan = _plan(work, command)
    assert plan is not None, command
    assert plan.argv == expected, (command, plan.argv)
    assert _permission(work, command).permission is ToolPermission.ALWAYS, command

    result = await collect_result(_tool(work).run(BashArgs(command=command)))

    assert result.exit_code == 0, command
    assert result.stdout, command


@pytest.mark.asyncio
async def test_wc_executes_each_literal_word(work: Path) -> None:
    result = await collect_result(_tool(work).run(BashArgs(command="wc *.txt")))

    assert result.exit_code == 0
    for name in ("alpha.txt", "beta.txt", "file.txt", "with space.txt"):
        assert name in result.stdout


def test_managed_background_launcher_keeps_the_glob_denial(work: Path) -> None:
    """Only a transport that can consume the plan may admit the whitelist.

    ``bash_start`` hands the command string to a shell, so it does not
    advertise the literal transport and keeps the existing path-glob denial
    even though the foreground tool admits the same call.
    """
    tool = BashStart(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=work
    )
    context = tool.resolve_permission(BashStartArgs(command="cat *.txt"))

    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert _plan(work, "cat *.txt") is not None


@pytest.mark.asyncio
async def test_background_execution_hard_denies_the_bound_plan(work: Path) -> None:
    """F1pkg1(c): the execution-time check, not only the permission layer.

    The background transport cannot consume a literal argv plan, so at
    execution it re-derives the bound plan and hard-denies — it never falls
    back to running the original shell text in the background shell.
    """
    tool = BashStart(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=work
    )
    started: list[str] = []

    class _RecordingPort:
        async def start(
            self,
            args: object,
            *,
            cwd: object = None,
            execute_command: str | None = None,
        ) -> object:
            started.append(str(execute_command))
            raise AssertionError("the background transport must not run a bound plan")

    port: Any = _RecordingPort()
    ctx = InvokeContext(tool_call_id="call-1", background_jobs=port)

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashStartArgs(command="cat *.txt"), ctx))

    assert "literal argv plan" in str(err.value)
    assert started == []


def test_plan_is_immutable(work: Path) -> None:
    plan = _plan(work, "cat *.txt")
    assert plan is not None
    assert isinstance(plan.argv, tuple)
    with pytest.raises(FrozenInstanceError):
        plan.argv = ()  # type: ignore[misc]


def test_all_literal_call_is_not_the_glob_transport(work: Path) -> None:
    """``cat alpha.txt`` keeps its shell execution: only glob calls bind."""
    assert _plan(work, "cat alpha.txt") is None
    assert _plan(work, "wc alpha.txt beta.txt") is None
    assert _permission(work, "cat alpha.txt").permission is ToolPermission.ALWAYS


def test_non_glob_option_call_reaches_the_ordinary_path(work: Path) -> None:
    """P2: an option word in a call with no glob is not a path-glob candidate.

    The option-token refusal used to run before the is-this-a-glob test, so
    ``wc -l file.txt`` denied as "path glob candidate '-l'". It is an
    ordinary literal call now: no plan binds and the path layer decides.
    """
    for command in (
        "wc -l file.txt",
        "cat -n file.txt",
        "head -n 2 file.txt",
        "sort file.txt",
        "uniq file.txt",
    ):
        assert _plan(work, command) is None, command
        assert _permission(work, command).permission is ToolPermission.ALWAYS, command


def test_glob_whitelist_is_consistent_with_the_read_position_table() -> None:
    """The transport set carries only table-classified read commands (P1c).

    Every command in the set has at least one read-only positional, and every
    expanded word the plan produces must land in a read-only position — which
    is what refuses `uniq`'s second positional in ``_glob_execution_plan``.
    """
    for command in _GLOB_WHITELIST_COMMANDS:
        assert read_only_positions(command), command
    assert is_read_only_argument_position("cat", "positional:3")
    assert is_read_only_argument_position("wc", "positional:2")
    assert is_read_only_argument_position("sort", "positional:2")
    assert is_read_only_argument_position("uniq", "positional:1")
    assert not is_read_only_argument_position("uniq", "positional:2")


@pytest.mark.asyncio
async def test_sort_glob_call_binds_a_literal_plan(work: Path) -> None:
    plan = _plan(work, "sort *.txt")

    assert plan is not None
    assert plan.argv == ("sort", "alpha.txt", "beta.txt", "file.txt", "with space.txt")
    assert _permission(work, "sort *.txt").permission is ToolPermission.ALWAYS

    result = await collect_result(_tool(work).run(BashArgs(command="sort *.txt")))

    assert result.exit_code == 0, result
    assert result.stdout.split() == ["alpha", "beta", "file", "spaced"]


def test_uniq_glob_call_refuses_the_write_position(work: Path) -> None:
    """A second expanded word would be uniq's OUTPUT file: the plan refuses."""
    with pytest.raises(GlobPlanDenial):
        _plan(work, "uniq *.txt")
    context = _permission(work, "uniq *.txt")

    assert context.permission is ToolPermission.NEVER
    assert "path glob candidate" in (context.reason or "")


@pytest.mark.asyncio
async def test_uniq_single_match_glob_call_binds_a_literal_plan(
    work: Path, tmp_path: Path
) -> None:
    """With one match the second positional stays empty, so the plan binds."""
    only = tmp_path / "single"
    only.mkdir()
    (only / "only.txt").write_text("one\none\ntwo\n", encoding="utf-8")

    plan = _plan(only, "uniq *.txt")

    assert plan is not None
    assert plan.argv == ("uniq", "only.txt")
    assert _permission(only, "uniq *.txt").permission is ToolPermission.ALWAYS

    result = await collect_result(_tool(only).run(BashArgs(command="uniq *.txt")))

    assert result.exit_code == 0, result
    assert result.stdout.splitlines() == ["one", "two"]


def test_sort_output_option_glob_call_stays_denied(work: Path) -> None:
    """The write option keeps its own denial: no plan may carry it."""
    with pytest.raises(GlobPlanDenial):
        _plan(work, "sort -o sorted.txt *.txt")
    assert _permission(work, "sort -o sorted.txt *.txt").permission is (
        ToolPermission.NEVER
    )


def test_non_whitelist_glob_admissions_never_enter_the_transport(work: Path) -> None:
    """The ls listing glob and the grep pattern glob stay outside S2.

    Their admissions keep their existing inspection (pinned by
    ``test_shell_policy_classification.py::test_benign_shell_forms`` and
    ``test_c2a_quote_aware_operands.py``); this transport must not absorb
    them.
    """
    assert _plan(work, "ls *.txt") is None
    assert _plan(work, "grep alpha *.txt") is None
    assert _permission(work, "ls *.txt").permission is ToolPermission.ALWAYS


# --- the design's deny matrix ------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("cat -n *.txt", id="options-denied"),
        pytest.param("wc -l *.txt", id="options-wc-denied"),
        pytest.param("cat [ab]*.txt", id="brackets-denied"),
        pytest.param("cat **/*.txt", id="globstar-denied"),
        pytest.param("cat {alpha,beta}.txt", id="braces-denied"),
        pytest.param("cat ~/*.txt", id="tilde-denied"),
        pytest.param("cat '*'a*.txt", id="mixed_quoting-denied"),
        pytest.param("cat *.nomatch", id="no_match-denied"),
        pytest.param("cat *.txt", id="option_like_result-denied"),
        pytest.param("cat -- *.txt", id="dash_after_terminator-denied"),
        pytest.param("cat *ir.txt", id="directory_match-denied"),
        pytest.param("cat *.fifo", id="non_regular_match-denied"),
        pytest.param("cat link.txt *.txt", id="symlink_literal_operand-denied"),
        pytest.param("cat *link*", id="symlink_match-denied"),
        pytest.param("cat linkdir/*.txt", id="symlink_component-denied"),
        pytest.param("cat ../outside/*.txt", id="read_authority-denied"),
        pytest.param("cat su*/*.txt", id="middle_component_wildcard-denied"),
        pytest.param("cat .*", id="dot_leading_pattern-denied"),
        pytest.param("cat *.txt > out.txt", id="redirect_context-denied"),
        pytest.param("cat *.txt | wc -l", id="pipeline_context-denied"),
        pytest.param("cat *.txt; echo $?", id="command_list_context-denied"),
        pytest.param("cat $(id) *.txt", id="command_substitution-denied"),
    ],
)
def test_deny_matrix(matrix: Path, command: str) -> None:
    assert _permission(matrix, command).permission is ToolPermission.NEVER, command


# --- execution transport: never the shell, never a stale plan ----------------


@pytest.mark.asyncio
async def test_literal_transport_never_invokes_the_shell(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell_calls: list[tuple[object, ...]] = []

    async def _record(*args: object, **kwargs: object) -> None:
        shell_calls.append(args)
        raise AssertionError("the glob transport must not use a shell")

    monkeypatch.setattr(bash_module, "spawn_shell_command", _record)

    result = await collect_result(_tool(work).run(BashArgs(command="cat *.txt")))

    assert result.exit_code == 0
    assert "alpha" in result.stdout
    assert shell_calls == []


@pytest.mark.asyncio
async def test_unbindable_glob_never_falls_back_to_the_shell(
    work: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mutation between check and execution denies; the shell never runs."""
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET\n", encoding="utf-8")
    assert _permission(work, "cat *.txt").permission is ToolPermission.ALWAYS

    (work / "alpha.txt").unlink()
    (work / "alpha.txt").symlink_to(secret)

    shell_calls: list[tuple[object, ...]] = []

    async def _record(*args: object, **kwargs: object) -> None:
        shell_calls.append(args)
        raise AssertionError("an unplannable glob must not use a shell")

    monkeypatch.setattr(bash_module, "spawn_shell_command", _record)

    with pytest.raises(ToolError) as err:
        await collect_result(_tool(work).run(BashArgs(command="cat *.txt")))

    assert "cannot be safely inspected" in str(err.value)
    assert shell_calls == []


def test_file_open_containment_rejects_a_symlink_swap(
    work: Path, tmp_path: Path
) -> None:
    """The OS file-open boundary refuses an operand swapped after planning."""
    plan = _plan(work, "cat *.txt")
    assert plan is not None

    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET\n", encoding="utf-8")
    (work / "alpha.txt").unlink()
    (work / "alpha.txt").symlink_to(secret)

    with pytest.raises(ToolError) as err:
        _require_glob_plan_containment(plan)

    assert "symlink-free" in str(err.value)


def test_file_open_containment_rejects_a_component_symlink_swap(
    work: Path, tmp_path: Path
) -> None:
    """A whole parent directory swapped for a symlink is refused too."""
    plan = _plan(work, "cat sub/*.txt")
    assert plan is not None

    outside = tmp_path / "outside-dir"
    outside.mkdir()
    (outside / "gamma.txt").write_text("outside\n", encoding="utf-8")
    (work / "sub").rename(work / "real-sub")
    (work / "sub").symlink_to(outside)

    with pytest.raises(ToolError) as err:
        _require_glob_plan_containment(plan)

    assert "symlink-free" in str(err.value)


@pytest.mark.asyncio
async def test_platform_without_containment_primitives_fails_closed(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bash_module, "_CONTAINMENT_OPEN_SUPPORTED", False)

    with pytest.raises(ToolError) as err:
        await collect_result(_tool(work).run(BashArgs(command="cat *.txt")))

    assert "containment" in str(err.value)


# --- S1 seam: every S1 admission keeps shell execution -----------------------


@pytest.mark.asyncio
async def test_same_invocation_status_probe_still_prints_real_status(
    work: Path,
) -> None:
    """``cmd; echo $?`` is not the glob whitelist and keeps its shell."""
    assert _plan(work, "cat alpha.txt; echo $?") is None

    ok = await collect_result(
        _tool(work).run(BashArgs(command="cat alpha.txt; echo $?"))
    )
    assert ok.exit_code == 0
    assert ok.stdout == "alpha\n0\n"

    failed = await collect_result(
        _tool(work).run(BashArgs(command="cat missing.txt; echo $?"))
    )
    assert failed.exit_code == 0
    assert failed.stdout == "1\n"


def test_quoted_and_escaped_equals_marker_admissions_are_unaffected(work: Path) -> None:
    assert _permission(work, "echo '=ls'").permission is ToolPermission.ALWAYS
    assert _permission(work, "echo \\=ls").permission is ToolPermission.ALWAYS
    assert _plan(work, "echo '=ls'") is None


# --- F1pkg3: an admitted quoted word never runs as shell text -----------------


def _quoted_plan(command: str):
    return quoted_word_execution_plan_for_command(command)


@pytest.mark.asyncio
async def test_quoted_word_transport_never_invokes_the_shell(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The admitted quoted word executes through the literal argv plan, and
    the shell-string transport is never invoked for the call.
    """
    shell_calls: list[tuple[object, ...]] = []

    async def _record(*args: object, **kwargs: object) -> None:
        shell_calls.append(args)
        raise AssertionError("the quoted-word transport must not use a shell")

    monkeypatch.setattr(bash_module, "spawn_shell_command", _record)

    result = await collect_result(
        _tool(work).run(BashArgs(command='X=alpha.txt; cat "$X"'))
    )

    assert result.exit_code == 0
    assert "alpha" in result.stdout
    assert shell_calls == []


@pytest.mark.asyncio
async def test_unbindable_quoted_word_admission_hard_denies(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An admission the execution boundary cannot re-derive refuses.

    The transport never falls back to running the shell text the admission
    replaced — the words that run must be the words that were inspected.
    """

    def _deny(*args: object, **kwargs: object):
        raise QuotedWordPlanDenial("the invocation changed shape")

    monkeypatch.setattr(bash_module, "quoted_word_execution_plan_for_command", _deny)
    shell_calls: list[tuple[object, ...]] = []

    async def _record(*args: object, **kwargs: object) -> None:
        shell_calls.append(args)
        raise AssertionError("a refused quoted word must not use a shell")

    monkeypatch.setattr(bash_module, "spawn_shell_command", _record)

    with pytest.raises(ToolError) as err:
        await collect_result(_tool(work).run(BashArgs(command='X=alpha.txt; cat "$X"')))

    assert "the invocation changed shape" in str(err.value)
    assert shell_calls == []


def test_all_literal_call_is_not_the_quoted_word_transport(work: Path) -> None:
    """A call without an admitted quoted word keeps its shell execution."""
    assert _quoted_plan("cat alpha.txt") is None
    assert _permission(work, "cat alpha.txt").permission is ToolPermission.ALWAYS


def test_managed_background_launcher_keeps_the_quoted_word_denial(work: Path) -> None:
    """Only a transport that consumes the literal plan may admit the word.

    ``bash_start`` hands the command string to a shell, so at permission time
    it keeps the expansion denial even though the foreground tool admits the
    same call.
    """
    tool = BashStart(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=work
    )
    context = tool.resolve_permission(BashStartArgs(command='X=alpha.txt; cat "$X"'))

    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "literal argv transport" in str(context.reason)
    assert _quoted_plan('X=alpha.txt; cat "$X"') is not None
