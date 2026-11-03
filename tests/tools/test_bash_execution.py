"""Bash tool execution tests for the bash suite (echo/cat execution, detached background jobs, timeouts, output truncation and drain caps, off-loop decoding, utf-32/latin-1 char caps, workdir resolution, benign unlisted execution, pager inheritance for local shell children, the CapturedShellResult returncode-alias contract, and the end-to-end expansion executions)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
from time import monotonic
from types import SimpleNamespace
from typing import Any

import pytest

from chartreux.core.background_jobs import BackgroundJobsPort, BashStartArgs
from chartreux.core.dangerous_commands import (
    REMOTE_EXEC_OPERATION_ID,
    DangerousCommandAuthority,
    RemoteExecAttestation,
    canonical_remote_destination,
)
from chartreux.core.tools.base import (
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    resolve_repository_scope,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    GlobPlanDenial,
    _pattern_expansion,
    glob_execution_plan_for_command,
    quoted_word_execution_plan_for_command,
)
import chartreux.core.tools.builtins.bash as bash_module
from chartreux.core.tools.builtins.bash import (
    Bash,
    BashArgs,
    BashToolConfig,
    CancelledShellCommand,
    CapturedShellResult,
    FailedShellCommand,
)
from chartreux.core.tools.builtins.bash_start import BashStart
import chartreux.core.utils.shell as shell_utils
from chartreux.core.utils.shell import spawn_shell_command
from tests.mock.utils import collect_result
from tests.tools._bash_test_support import bash as bash, locate_file as locate_file


def test_shell_results_keep_a_returncode_alias_for_post_tool_hooks():
    # The verbatim result dump is the `tool_output` payload handed to hooks.
    captured = CapturedShellResult(command="x", exit_code=2).model_dump(mode="json")
    assert captured["exit_code"] == 2
    assert captured["returncode"] == 2


@pytest.mark.asyncio
async def test_runs_echo_successfully(bash):
    result = await collect_result(bash.run(BashArgs(command="echo hello")))

    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_detached_background_job_survives_completed_command(bash, tmp_path):
    marker = tmp_path / "survived"
    command = (
        "nohup sh -c 'sleep 0.2; echo survived > survived; sleep 30' "
        ">background.log 2>&1 </dev/null & echo $!"
    )
    result = await collect_result(bash.run(BashArgs(command=command)))
    pid = int(result.stdout.strip())
    try:
        assert result.exit_code == 0
        async with asyncio.timeout(3):
            while not marker.exists():
                await asyncio.sleep(0.01)
        assert marker.read_text().strip() == "survived"
    finally:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.asyncio
async def test_fails_cat_command_with_missing_file(bash):
    with pytest.raises(ToolError) as err:
        await collect_result(bash.run(BashArgs(command="cat missing_file.txt")))

    message = str(err.value)
    assert "Command failed" in message
    assert "Return code: 1" in message
    assert "No such file or directory" in message


@pytest.mark.asyncio
async def test_uses_effective_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = BashToolConfig()
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    result = await collect_result(bash_tool.run(BashArgs(command="pwd")))

    assert result.stdout.strip() == str(tmp_path)


@pytest.mark.asyncio
async def test_handles_timeout(bash):
    with pytest.raises(ToolError) as err:
        await collect_result(bash.run(BashArgs(command="sleep 2", timeout=1)))

    assert "Command timed out after 1s" in str(err.value)


@pytest.mark.asyncio
async def test_failed_command_retains_its_write_paths(bash, tmp_path):
    # R7: a failed run may still have written files (partial execution);
    # the raised error carries the run's write-classified paths exactly as
    # a successful result would surface them, so the submission scan keeps
    # covering the footprint.
    with pytest.raises(FailedShellCommand) as err:
        await collect_result(
            bash.run(BashArgs(command="echo x > failed_out.txt; exit 3"))
        )

    assert err.value.write_paths == [str(tmp_path / "failed_out.txt")]
    assert err.value.write_paths_unknown is False


@pytest.mark.asyncio
async def test_timed_out_command_retains_its_write_paths(bash, tmp_path):
    # R7: same footprint retention on the timeout path — the command wrote
    # the file before the kill.
    with pytest.raises(FailedShellCommand) as err:
        await collect_result(
            bash.run(BashArgs(command="echo x > timed_out.txt; sleep 30", timeout=1))
        )

    assert err.value.write_paths == [str(tmp_path / "timed_out.txt")]
    assert err.value.write_paths_unknown is False


@pytest.mark.asyncio
async def test_cancelled_command_retains_its_write_paths(bash, tmp_path):
    # Same footprint retention on the cancellation path — the command wrote
    # the file before the interrupt; the cancellation carries the run's
    # write-classified paths for the executor's cancelled result event.
    # Observed inside the task, as the executor sees it: a task-cancelled
    # run's exception is replaced by a fresh instance at the task boundary.
    captured = {}

    async def run_and_capture():
        try:
            await collect_result(
                bash.run(BashArgs(command="echo x > cancelled_out.txt; sleep 30"))
            )
        except asyncio.CancelledError as exc:
            captured["cancellation"] = exc
            raise

    task = asyncio.ensure_future(run_and_capture())
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    cancellation = captured["cancellation"]
    assert isinstance(cancellation, CancelledShellCommand)
    assert cancellation.write_paths == [str(tmp_path / "cancelled_out.txt")]
    assert cancellation.write_paths_unknown is False


@pytest.mark.asyncio
async def test_truncates_output_to_max_bytes(bash):
    config = BashToolConfig(max_output_bytes=5)
    bash_tool = Bash(config_getter=lambda: config, state=BaseToolState())

    result = await collect_result(
        bash_tool.run(BashArgs(command="printf 'abcdefghij'"))
    )

    assert result.stdout == "abcde"
    assert result.stderr == ""
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_drains_both_streams_after_capture_cap(bash, monkeypatch):
    config = BashToolConfig(max_output_bytes=7)
    tool = Bash(config_getter=lambda: config, state=BaseToolState())
    original = bash_module._drain_shell_stream
    retained: list[int] = []

    async def measured_drain(stream, max_chars):
        captured = await original(stream, max_chars)
        retained.append(len(captured))
        return captured

    monkeypatch.setattr(bash_module, "_drain_shell_stream", measured_drain)
    command = "(yes O | head -c 1048576) & (yes E | head -c 1048576 >&2) & wait"
    start = monotonic()
    result = await asyncio.wait_for(
        collect_result(tool.run(BashArgs(command=command, timeout=20))), timeout=5
    )

    assert monotonic() - start < 5
    assert result.exit_code == 0
    assert result.stdout == "O\nO\nO\nO"
    assert result.stderr == "E\nE\nE\nE"
    assert sorted(retained) == [32, 32]


@pytest.mark.asyncio
async def test_shell_output_decoding_runs_off_loop(bash, monkeypatch):
    loop_thread = threading.get_ident()
    original = bash_module.decode_console_safe
    decode_threads: list[int] = []

    def checked_decode(raw, *, raise_on_error=False):
        decode_threads.append(threading.get_ident())
        return original(raw, raise_on_error=raise_on_error)

    monkeypatch.setattr(bash_module, "decode_console_safe", checked_decode)
    result = await collect_result(
        bash.run(BashArgs(command="printf hello; printf error >&2"))
    )

    assert result.stdout == "hello"
    assert result.stderr == "error"
    assert len(decode_threads) == 2
    assert all(thread != loop_thread for thread in decode_threads)


@pytest.mark.asyncio
async def test_bounded_shell_output_preserves_utf32_char_cap(tmp_path):
    source = tmp_path / "utf32.txt"
    source.write_bytes("ééééé".encode("utf-32"))
    config = BashToolConfig(max_output_bytes=3)
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=tmp_path)

    result = await collect_result(tool.run(BashArgs(command="cat utf32.txt")))

    assert result.stdout == "ééé"
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_cat_preserves_accents_from_latin1_encoded_file(bash, tmp_path):
    file = tmp_path / "menu.txt"
    file.write_bytes("café au lait\nthé glacé\n".encode("latin-1"))

    result = await collect_result(bash.run(BashArgs(command=f"cat {file.name}")))

    assert result.exit_code == 0
    assert "\ufffd" not in result.stdout
    assert result.stdout == "café au lait\nthé glacé\n"


@pytest.mark.asyncio
async def test_benign_unlisted_command_still_executes(bash):
    result = await collect_result(bash.run(BashArgs(command="echo still-runs")))

    assert result.exit_code == 0
    assert result.stdout.strip() == "still-runs"


@pytest.mark.asyncio
async def test_local_shell_children_inherit_suppressed_pager(tmp_path: Path) -> None:
    from chartreux.core.utils.shell import spawn_shell_command

    # Client terminals do not use this local shell environment, so repository
    # pager config is still denied by the resolver for that execution path.
    proc = await spawn_shell_command(
        'printf "%s %s" "$GIT_PAGER" "$PAGER"', cwd=tmp_path
    )
    stdout, _stderr = await proc.communicate()
    assert proc.returncode == 0
    assert stdout == b"cat cat"


# --- tier-2 static expansion: whitelisted idioms end to end --------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "contains", "absent"),
    [
        # The observed idiom.
        (
            "sed -n \"$(grep -n 'lit' lines.txt | head -1 | cut -d: -f1),+45p\" lines.txt",
            ("lit", "line46"),
            ("alpha", "line48"),
        ),
        # head -N with a single located line: the selector is a no-op that
        # still narrows to one address.
        (
            "sed -n \"$(grep -n 'lit' lines.txt | head -2 | cut -d: -f1),+3p\" lines.txt",
            ("lit", "line5"),
            ("alpha", "line7"),
        ),
        # No head: a single match needs no selector.
        (
            "sed -n \"$(grep -n 'lit' lines.txt | cut -d: -f1),+45p\" lines.txt",
            ("lit", "line46"),
            ("alpha", "line48"),
        ),
        # tail -1 selects the last match.
        (
            "sed -n \"$(grep -n 'line' lines.txt | tail -1 | cut -d: -f1),+2p\" lines.txt",
            ("line59",),
            ("line56", "alpha"),
        ),
        # The `,Np` absolute-range suffix.
        (
            "sed -n \"$(grep -n 'lit' lines.txt | head -1 | cut -d: -f1),3p\" lines.txt",
            ("lit", "beta"),
            ("line4",),
        ),
    ],
)
async def test_sed_locate_print_idiom_executes_successfully(
    bash,
    locate_file: str,
    command: str,
    contains: tuple[str, ...],
    absent: tuple[str, ...],
) -> None:
    context = bash.resolve_permission(BashArgs(command=command))
    assert context is None or context.permission is ToolPermission.ALWAYS

    result = await collect_result(bash.run(BashArgs(command=command)))

    assert result.exit_code == 0
    for fragment in contains:
        assert fragment in result.stdout
    for fragment in absent:
        assert fragment not in result.stdout


@pytest.mark.asyncio
async def test_sed_locate_print_no_match_prints_the_whole_file(
    bash, locate_file: str
) -> None:
    command = (
        "sed -n \"$(grep -n 'absent' lines.txt | head -1 | cut -d: -f1)p\" lines.txt"
    )
    context = bash.resolve_permission(BashArgs(command=command))
    assert context is None or context.permission is ToolPermission.ALWAYS

    result = await collect_result(bash.run(BashArgs(command=command)))

    # No match degrades to the suffix alone: `p` prints every line (benign).
    assert result.exit_code == 0
    assert "alpha" in result.stdout
    assert "line59" in result.stdout


@pytest.mark.asyncio
async def test_read_only_position_expansion_executes_with_inlined_literal(
    bash, tmp_path
) -> None:
    (tmp_path / "matches.txt").write_text("lit\n" * 7)
    command = "X=5; grep -m $X lit matches.txt"

    context = bash.resolve_permission(BashArgs(command=command))
    assert context is None or context.permission is ToolPermission.ALWAYS

    result = await collect_result(bash.run(BashArgs(command=command)))

    # The inlined literal is the real max-count: exactly five matches print.
    assert result.exit_code == 0
    assert result.stdout.splitlines() == ["lit"] * 5


@pytest.mark.asyncio
async def test_arithmetic_in_read_only_position_executes(
    bash, locate_file: str
) -> None:
    command = "N=2; head -n $((N+1)) lines.txt"

    context = bash.resolve_permission(BashArgs(command=command))
    assert context is None or context.permission is ToolPermission.ALWAYS

    result = await collect_result(bash.run(BashArgs(command=command)))

    assert result.exit_code == 0
    assert result.stdout.splitlines() == ["alpha", "lit", "beta"]


# --- F1pkg1(b): execution-observing executor option contract -------------------

_PINNED_EXECUTORS = ("/usr/bin/zsh", "/usr/bin/bash", "/bin/sh")


def _executor_ids() -> list[str]:
    return [path for path in _PINNED_EXECUTORS if Path(path).exists()]


@pytest.fixture
def expansion_dir(tmp_path: Path) -> Path:
    root = tmp_path / "expansion"
    root.mkdir()
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    (root / "b.txt").write_text("b\n", encoding="utf-8")
    (root / ".hidden").write_text("h\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "c.txt").write_text("c\n", encoding="utf-8")
    return root


async def _spawn_words(executor: str, command: str, cwd: Path) -> tuple[list[str], int]:
    proc = await spawn_shell_command(command, cwd=cwd)
    stdout, _stderr = await proc.communicate()
    assert proc.returncode is not None
    return stdout.decode("utf-8", "replace").splitlines(), proc.returncode


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", _executor_ids())
@pytest.mark.parametrize(
    ("token", "expected"),
    [
        # Extended syntax WITHOUT any *?[ metacharacter stays literal under the
        # pinned contract: the model admits these and predicts the literal word.
        ("^foo", ["^foo"]),
        ("foo#", ["foo#"]),
        ("a~b", ["a~b"]),
        # Ordinary admitted globs expand exactly as the model predicts.
        ("*.txt", ["a.txt", "b.txt"]),
        ("?.txt", ["a.txt", "b.txt"]),
        ("[ab].txt", ["a.txt", "b.txt"]),
    ],
)
async def test_pinned_executor_executes_what_the_model_predicts(
    executor: str, token: str, expected: list[str], expansion_dir: Path, monkeypatch
) -> None:
    """A4 differential: the pinned executing shell's actual expansion of an
    admitted form must equal the model's predicted expansion.
    """
    monkeypatch.setenv("SHELL", executor)
    words, returncode = await _spawn_words(
        executor, f"printf '%s\\n' {token}", expansion_dir
    )
    assert returncode == 0, (executor, token, words)
    assert words == expected, (executor, token, words)

    model = _pattern_expansion(expansion_dir, token)
    assert not isinstance(model, list) or sorted(
        word for _path, word in model
    ) == sorted(expected), (executor, token, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", _executor_ids())
async def test_extended_syntax_paren_form_fails_in_the_real_executor_too(
    executor: str, expansion_dir: Path, monkeypatch, tmp_path: Path
) -> None:
    """Execution-observing node: ``@(foo)`` carries no *?[ metacharacter, so the
    model's expansion layer treats the word as literal — the tree-sitter layer
    denies it as a syntax error. The pinned real executor must agree: it fails
    to parse the form (nonzero exit), it never expands it.
    """
    monkeypatch.setenv("SHELL", executor)
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=expansion_dir
    )
    result = tool.resolve_permission(BashArgs(command="echo @(foo)"))
    assert result is not None and result.permission is ToolPermission.NEVER

    _words, returncode = await _spawn_words(executor, "echo @(foo)", expansion_dir)
    assert returncode != 0, (executor, returncode)


@pytest.mark.asyncio
async def test_zsh_startup_pin_defeats_a_hostile_zshenv(
    expansion_dir: Path, monkeypatch, tmp_path: Path
) -> None:
    """A hostile $ZDOTDIR/.zshenv turning extendedglob on must not reach the
    spawned child: the contract pins ZDOTDIR away, so ``^a.txt`` stays literal
    instead of expanding to every non-a.txt entry.
    """
    if not Path("/usr/bin/zsh").exists():
        pytest.skip("zsh is not installed")
    hostile = tmp_path / "hostile-zdotdir"
    hostile.mkdir()
    (hostile / ".zshenv").write_text("setopt extendedglob\n", encoding="utf-8")
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    monkeypatch.setenv("ZDOTDIR", str(hostile))

    words, returncode = await _spawn_words(
        "/usr/bin/zsh", "printf '%s\\n' ^a.txt", expansion_dir
    )
    assert returncode == 0
    assert words == ["^a.txt"]


@pytest.mark.asyncio
async def test_bash_startup_pin_defeats_a_hostile_bash_env(
    expansion_dir: Path, monkeypatch, tmp_path: Path
) -> None:
    """A hostile $BASH_ENV enabling extglob must not reach the spawned child:
    the contract scrubs BASH_ENV, so ``@(foo)`` is still a syntax error and
    ``^a.txt`` still literal.
    """
    if not Path("/usr/bin/bash").exists():
        pytest.skip("bash is not installed")
    hostile = tmp_path / "hostile-bash-env.sh"
    hostile.write_text("shopt -s extglob\n", encoding="utf-8")
    monkeypatch.setenv("SHELL", "/usr/bin/bash")
    monkeypatch.setenv("BASH_ENV", str(hostile))

    words, returncode = await _spawn_words(
        "/usr/bin/bash", "printf '%s\\n' ^a.txt", expansion_dir
    )
    assert returncode == 0
    assert words == ["^a.txt"]
    _words, error_returncode = await _spawn_words(
        "/usr/bin/bash", "echo @(foo)", expansion_dir
    )
    assert error_returncode != 0


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", _executor_ids())
async def test_whitelist_glob_plan_equals_the_real_expansion(
    executor: str, expansion_dir: Path, monkeypatch
) -> None:
    """A4 differential at the transport: the literal argv plan the whitelist
    executes is exactly what the pinned real executor expands ``*.txt`` to.
    """
    monkeypatch.setenv("SHELL", executor)
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=expansion_dir
    )
    plan = glob_execution_plan_for_command(
        "cat *.txt",
        cwd=tool.cwd,
        workspace=tool.workspace,
        authority=tool.path_authority,
    )
    assert plan is not None
    actual, returncode = await _spawn_words(
        executor, "printf '%s\\n' *.txt", expansion_dir
    )
    assert returncode == 0
    assert sorted(plan.argv[1:]) == sorted(actual)


# --- F1pkg2: bounded trailing-star admission, bound to the actual spawn -------
#
# A final component whose trailing star run is exactly two stars is admitted
# only for a pinned executor whose contract proves the run cannot recurse
# (bash pins globstar off; a POSIX shell has no globstar), and only through the
# literal argv plan the whitelist already executes. The binding is what makes
# the admission sound: the words the OS receives are the plan's words, never
# the original shell text. Everything else — the recursive ``**/`` form, a
# non-trailing run, a run of three stars, a wildcard in a middle component, a
# zsh or unknown executor — keeps the fail-closed denial.


def _trailing_star_work(root: Path) -> None:
    flat = root / "flat"
    flat.mkdir()
    (flat / "a.txt").write_text("a\n", encoding="utf-8")
    (flat / "b.txt").write_text("b\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_trailing_star_plan_binds_the_actual_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The admitted trailing ``**`` runs as the plan's literal argv, and the
    pinned executor's own expansion of the same word is that same word list.
    """
    if not Path("/usr/bin/bash").exists():
        pytest.skip("bash is not installed")
    _trailing_star_work(tmp_path)
    monkeypatch.setenv("SHELL", "/usr/bin/bash")
    tool = _cwd_tool(tmp_path)
    plan = glob_execution_plan_for_command(
        "cat flat/**",
        cwd=tool.cwd,
        workspace=tool.workspace,
        authority=tool.path_authority,
    )
    assert plan is not None
    assert plan.argv == ("cat", "flat/a.txt", "flat/b.txt")
    context = tool.resolve_permission(BashArgs(command="cat flat/**"))
    assert context is not None and context.permission is ToolPermission.ALWAYS

    # A4 differential: the pinned executing shell expands the admitted word to
    # exactly the plan's words, so the plan binds the real expansion.
    actual, returncode = await _spawn_words(
        "/usr/bin/bash", "printf '%s\\n' flat/**", tmp_path
    )
    assert returncode == 0, actual
    assert actual == list(plan.argv[1:])

    # Execution binding: the argv transport consumes the plan and the
    # shell-string transport is never invoked for the admitted call.
    shell_calls: list[object] = []
    argv_calls: list[tuple[str, ...]] = []
    real_argv = bash_module.spawn_argv_command

    async def _record_shell(*args: object, **kwargs: object) -> None:
        shell_calls.append(args)
        raise AssertionError("an admitted trailing star must never run as shell text")

    async def _record_argv(argv: Any, **kwargs: Any) -> Any:
        argv_calls.append(tuple(argv))
        return await real_argv(argv, **kwargs)

    monkeypatch.setattr(bash_module, "spawn_shell_command", _record_shell)
    monkeypatch.setattr(bash_module, "spawn_argv_command", _record_argv)

    result = await collect_result(tool.run(BashArgs(command="cat flat/**")))

    assert result.exit_code == 0
    assert "a" in result.stdout and "b" in result.stdout
    assert shell_calls == []
    assert argv_calls == [plan.argv]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "executor"),
    [
        pytest.param("cat flat/**/a.txt", "/usr/bin/bash", id="recursive-form"),
        pytest.param("cat a**b*.txt", "/usr/bin/bash", id="non-trailing-run"),
        pytest.param("cat flat/**", "/usr/bin/zsh", id="zsh-executor"),
        pytest.param("cat flat/**", "/opt/unknown/ksh93", id="unknown-executor"),
    ],
)
async def test_trailing_star_denials_never_reach_a_shell(
    command: str, executor: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every denied spelling fails closed at execution, before any spawn."""
    _trailing_star_work(tmp_path)
    monkeypatch.setenv("SHELL", executor)
    tool = _cwd_tool(tmp_path)
    context = tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission is ToolPermission.NEVER, command

    spawned: list[object] = []

    async def _record_spawn(*args: object, **kwargs: object) -> None:
        spawned.append(args)
        raise AssertionError("a denied trailing star must not spawn anything")

    monkeypatch.setattr(bash_module, "spawn_shell_command", _record_spawn)
    monkeypatch.setattr(bash_module, "spawn_argv_command", _record_spawn)

    with pytest.raises(ToolError):
        await collect_result(tool.run(BashArgs(command=command)))

    assert spawned == []


def test_three_star_run_is_never_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run of three stars (zsh's symlink-following glob) is not the admitted
    spelling: the whitelist never binds it to a plan, and the policy layer
    keeps its own syntax denial.
    """
    _trailing_star_work(tmp_path)
    monkeypatch.setenv("SHELL", "/usr/bin/bash")
    tool = _cwd_tool(tmp_path)
    context = tool.resolve_permission(BashArgs(command="cat flat/***"))
    assert context is not None and context.permission is ToolPermission.NEVER

    try:
        plan = glob_execution_plan_for_command(
            "cat flat/***",
            cwd=tool.cwd,
            workspace=tool.workspace,
            authority=tool.path_authority,
        )
    except GlobPlanDenial:
        return
    assert plan is None


# --- F1pkg3: quoted-word binding, bound to the actual spawn --------------------
#
# A whole double-quoted `"$NAME"` / `"${NAME}"` argument is admitted only in a
# provably read-only position and only when a literal assignment to NAME
# precedes the sole command unchanged. The admission is bound to actual
# execution: the literal argv plan carries the resolved word to the spawn and
# the shell never re-expands the source text. Every other spelling — ambient,
# empty, a prefixed assignment value, a modifier, a concatenated or nested
# word — keeps the fail-closed expansion denial.


def _quoted_word_work(root: Path) -> None:
    (root / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    (root / "beta.txt").write_text("beta\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_quoted_variable_plan_binds_the_actual_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The admitted quoted word runs as the plan's literal argv, and the
    pinned executor's own expansion of the same word is that same word.
    """
    if not Path("/usr/bin/bash").exists():
        pytest.skip("bash is not installed")
    _quoted_word_work(tmp_path)
    monkeypatch.setenv("SHELL", "/usr/bin/bash")
    tool = _cwd_tool(tmp_path)
    command = 'X=alpha.txt; cat "$X"'
    plan = quoted_word_execution_plan_for_command(command)
    assert plan is not None
    assert plan.argv == ("cat", "alpha.txt")
    context = tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission is ToolPermission.ALWAYS

    # Differential: the pinned executing shell expands the admitted word to
    # exactly the plan's literal word.
    words, returncode = await _spawn_words(
        "/usr/bin/bash", 'X=alpha.txt; printf "%s\\n" "$X"', tmp_path
    )
    assert returncode == 0, words
    assert words == list(plan.argv[1:])

    # Execution binding: the argv transport consumes the plan and the
    # shell-string transport is never invoked for the admitted call.
    shell_calls: list[object] = []
    argv_calls: list[tuple[str, ...]] = []
    real_argv = bash_module.spawn_argv_command

    async def _record_shell(*args: object, **kwargs: object) -> None:
        shell_calls.append(args)
        raise AssertionError("an admitted quoted word must never run as shell text")

    async def _record_argv(argv: Any, **kwargs: Any) -> Any:
        argv_calls.append(tuple(argv))
        return await real_argv(argv, **kwargs)

    monkeypatch.setattr(bash_module, "spawn_shell_command", _record_shell)
    monkeypatch.setattr(bash_module, "spawn_argv_command", _record_argv)

    result = await collect_result(tool.run(BashArgs(command=command)))

    assert result.exit_code == 0
    assert result.stdout == "alpha\n"
    assert shell_calls == []
    assert argv_calls == [plan.argv]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        pytest.param('X=alpha.txt; cat "${X}"', ("cat", "alpha.txt"), id="braced"),
        pytest.param(
            'N=3; head -n "$N" alpha.txt',
            ("head", "-n", "3", "alpha.txt"),
            id="read-only-option-value",
        ),
        pytest.param(
            'X=alpha.txt; Y=beta.txt; cat "$X" "$Y"',
            ("cat", "alpha.txt", "beta.txt"),
            id="two-admitted-words",
        ),
    ],
)
async def test_quoted_variable_admissions_bind_every_admitted_word(
    command: str, expected: tuple[str, ...], tmp_path: Path
) -> None:
    """Each admitted spelling reaches the spawn as one literal argv element."""
    _quoted_word_work(tmp_path)
    tool = _cwd_tool(tmp_path)
    plan = quoted_word_execution_plan_for_command(command)
    assert plan is not None
    assert plan.argv == expected

    context = tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission is ToolPermission.ALWAYS
    result = await collect_result(tool.run(BashArgs(command=command)))
    assert result.exit_code == 0
    assert result.stdout


@pytest.mark.parametrize(
    "command",
    [
        pytest.param('cat "$X"', id="ambient"),
        pytest.param('X=alpha.txt; cat "$Y"', id="unbound-second-name"),
        pytest.param('X=; cat "$X"', id="empty-value"),
        pytest.param('X=y$Z; cat "$X"', id="prefixed-assignment-value"),
        pytest.param('X=alpha.txt; cat "${X:-fallback}"', id="modifier"),
        pytest.param('X=alpha.txt; cat "pre$X"', id="concatenated"),
        pytest.param('X=alpha.txt; cat "$X$X"', id="nested"),
        pytest.param('X=alpha.txt; Y=beta.txt; cat "${X}${Y}"', id="two-braced"),
        pytest.param('X=alpha.txt; cat "$X" | wc -l', id="pipeline"),
        pytest.param('X=alpha.txt; cat "$X"; echo done', id="command-list"),
        pytest.param('X=alpha.txt; cat "$X" > out.txt', id="redirect"),
        pytest.param("bash -c 'X=alpha.txt; cat \"$X\"'", id="nested-shell-payload"),
    ],
)
def test_quoted_variable_denials_keep_the_expansion_denial(
    command: str, tmp_path: Path
) -> None:
    """Every denied spelling resolves never and binds no literal plan."""
    _quoted_word_work(tmp_path)
    tool = _cwd_tool(tmp_path)
    context = tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission is ToolPermission.NEVER, command
    assert quoted_word_execution_plan_for_command(command) is None


# --- F1pkg1(c): four-transport pinned-plan consumption --------------------------
#
# Approved semantics must equal actual spawn: each transport — terminal, argv,
# shell-string, background — consumes the bound (pinned) contract at execution
# or hard-denies, and a literal argv never falls back to original shell text.


def _cwd_tool(cwd: Path) -> Bash:
    return Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)


def _glob_work(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_argv_transport_consumes_the_bound_plan_and_never_falls_back_to_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The literal argv transport consumes the bound plan at execution: the
    words the OS runs are exactly the plan's words, and the shell-string
    transport is never invoked for a plan-bound call.
    """
    _glob_work(tmp_path)
    tool = _cwd_tool(tmp_path)
    plan = glob_execution_plan_for_command(
        "cat *.txt",
        cwd=tool.cwd,
        workspace=tool.workspace,
        authority=tool.path_authority,
    )
    assert plan is not None

    shell_calls: list[object] = []
    argv_calls: list[tuple[str, ...]] = []
    real_argv = bash_module.spawn_argv_command

    async def _record_shell(*args: object, **kwargs: object) -> None:
        shell_calls.append(args)
        raise AssertionError("a bound plan must never fall back to shell text")

    async def _record_argv(argv: Any, **kwargs: Any) -> Any:
        argv_calls.append(tuple(argv))
        return await real_argv(argv, **kwargs)

    monkeypatch.setattr(bash_module, "spawn_shell_command", _record_shell)
    monkeypatch.setattr(bash_module, "spawn_argv_command", _record_argv)

    result = await collect_result(tool.run(BashArgs(command="cat *.txt")))

    assert result.exit_code == 0
    assert "a" in result.stdout and "b" in result.stdout
    assert shell_calls == []
    assert argv_calls == [plan.argv]


@pytest.mark.asyncio
async def test_shell_string_transport_spawns_the_checked_executor_with_pinned_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shell-string transport consumes the pinned contract at execution in
    one read: the binary checked against the contract is the binary handed to
    the spawn, and the child env carries that contract's pins.
    """
    real_create = shell_utils.asyncio.create_subprocess_shell
    captured: dict[str, Any] = {}

    async def _record(command: str, **kwargs: Any) -> Any:
        captured["command"] = command
        captured.update(kwargs)
        return await real_create(command, **kwargs)

    monkeypatch.setattr(shell_utils.asyncio, "create_subprocess_shell", _record)
    monkeypatch.setenv("SHELL", "/bin/sh")
    monkeypatch.setenv("BASH_ENV", "/tmp/chartreux-hostile-bash-env.sh")

    proc = await spawn_shell_command("printf hi", cwd=tmp_path)
    stdout, _stderr = await proc.communicate()
    assert proc.returncode == 0
    assert stdout == b"hi"
    assert captured["executable"] == "/bin/sh"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["LC_COLLATE"] == "C"
    assert "BASH_ENV" not in env


@pytest.mark.asyncio
async def test_shell_string_transport_hard_denies_an_unknown_executor_at_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An executor with no pinned contract fails closed AT THE SPAWN SITE:
    nothing is spawned and the denial names the contract. A mutation that
    bypasses the consumption check lets the spawn proceed and fails this node.
    """
    monkeypatch.setenv("SHELL", "/nonexistent/chartreux-unknown-executor")

    async def _must_not_spawn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn site was reached without a pinned contract")

    monkeypatch.setattr(shell_utils.asyncio, "create_subprocess_shell", _must_not_spawn)
    tool = _cwd_tool(tmp_path)
    # Non-expanding text is admitted at permission (nothing to model), so the
    # execution-time contract check is the only guard on this path.
    context = tool.resolve_permission(BashArgs(command="printf hi"))
    assert context is None or context.permission is ToolPermission.ALWAYS

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashArgs(command="printf hi")))

    assert "no pinned executor option contract" in str(err.value)


class _RecordingTerminalToolIO:
    """A terminal-capable tool_io that records every run_shell request."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    @property
    def supports_terminal(self) -> bool:
        return True

    async def run_shell(self, request: Any) -> Any:
        self.requests.append(request.command)
        raise AssertionError("the terminal transport must not receive this call")


@pytest.mark.asyncio
async def test_terminal_transport_never_receives_a_bound_plan_or_expansion_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The terminal transport consumes the bound contract or hard-denies: a
    bound glob plan is never routed to run_shell (the literal argv transport
    consumes it), and expansion-bearing shell text refuses at the tool site
    before any terminal request is made.
    """
    _glob_work(tmp_path)
    tool = _cwd_tool(tmp_path)
    tool_io: Any = _RecordingTerminalToolIO()
    ctx = InvokeContext(tool_call_id="call-1", session_id="session-1", tool_io=tool_io)

    argv_calls: list[tuple[str, ...]] = []
    real_argv = bash_module.spawn_argv_command

    async def _record_argv(argv: Any, **kwargs: Any) -> Any:
        argv_calls.append(tuple(argv))
        return await real_argv(argv, **kwargs)

    monkeypatch.setattr(bash_module, "spawn_argv_command", _record_argv)

    result = await collect_result(tool.run(BashArgs(command="cat *.txt"), ctx))
    assert result.exit_code == 0
    assert "a" in result.stdout
    assert tool_io.requests == []
    assert argv_calls and argv_calls[0][0] == "cat"

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashArgs(command="ls *.txt"), ctx))
    assert "needs shell pathname expansion" in str(err.value)
    assert tool_io.requests == []


@pytest.mark.asyncio
async def test_background_transport_hard_denies_a_bound_plan_at_execution(
    tmp_path: Path,
) -> None:
    """The background transport cannot consume a literal argv plan: at
    execution it re-derives the bound plan and hard-denies, never falling back
    to running the original shell text in the background shell.
    """
    _glob_work(tmp_path)
    tool = BashStart(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )

    started: list[str] = []

    class _RecordingPort:
        async def start(
            self, args: Any, *, cwd: Any = None, execute_command: str | None = None
        ) -> Any:
            started.append(str(execute_command))
            raise AssertionError("the background transport must not run a bound plan")

    port: Any = _RecordingPort()
    ctx = InvokeContext(tool_call_id="call-2", background_jobs=port)

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashStartArgs(command="cat *.txt"), ctx))

    assert "literal argv plan" in str(err.value)
    assert started == []


# --- S4b1b: per-spawn pinning of the approved ssh destination ----------------
#
# Authority, locality/resolution, and the spawn pin coincide: every transport
# that can run a standalone ssh call with a positive destination (literal IP,
# or since S4b2a a uniquely-non-locally-resolved hostname) receives the
# pinned argv — the approved address plus the fixed known-hosts/host-alias
# identity — and every other form (ambiguous/local/failed resolution, local
# literal, unattested client executor, missing grant) fails closed before
# any spawn.

_S4B1B_COMMAND = "ssh deploy@8.8.8.8 uptime"
_S4B1B_PINNED_ARGV = (
    "ssh",
    "-o",
    "UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts",
    "-o",
    "HostKeyAlias=deploy@8.8.8.8:22",
    "-o",
    "StrictHostKeyChecking=yes",
    "-o",
    "BatchMode=yes",
    "deploy@8.8.8.8",
    "uptime",
)
_S4B1B_PINNED_COMMAND = (
    "ssh -o UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts"
    " -o HostKeyAlias=deploy@8.8.8.8:22 -o StrictHostKeyChecking=yes"
    " -o BatchMode=yes deploy@8.8.8.8 uptime"
)


def _s4b1b_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


@pytest.fixture
def s4b1b_interfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic locality: a fixed local-interface enumeration (S4b1a seam)."""

    def fake_getaddrinfo(host, port):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.10.20", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def _s4b1b_grant(repo: Path) -> DangerousCommandAuthority:
    scope = resolve_repository_scope(repo, GitRepositoryRelocation())
    assert scope is not None
    return DangerousCommandAuthority(
        operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
        scope=scope,
        source="runtime_grant",
        destination=canonical_remote_destination("deploy@8.8.8.8:22"),
    )


class _FakeStream:
    async def read(self, n: int = -1) -> bytes:
        return b""


class _FakeProc:
    def __init__(self) -> None:
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()
        self.returncode = 0

    async def wait(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_local_spawn_pins_address_and_host_key_identity_into_the_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s4b1b_interfaces
) -> None:
    """The local argv transport receives exactly the pinned argv: the approved
    address and the fixed known-hosts/host-alias identity are in the words the
    OS runs, never the original command string.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: (_s4b1b_grant(repo),)

    argv_calls: list[tuple[str, ...]] = []
    shell_calls: list[object] = []

    async def _record_argv(argv: Any, **kwargs: Any) -> Any:
        argv_calls.append(tuple(argv))
        return _FakeProc()

    async def _must_not_spawn_shell(*args: object, **kwargs: object) -> None:
        shell_calls.append(args)
        raise AssertionError("a pinned remote-exec plan never runs as shell text")

    monkeypatch.setattr(bash_module, "spawn_argv_command", _record_argv)
    monkeypatch.setattr(bash_module, "spawn_shell_command", _must_not_spawn_shell)

    result = await collect_result(tool.run(BashArgs(command=_S4B1B_COMMAND)))

    assert result.exit_code == 0
    assert argv_calls == [_S4B1B_PINNED_ARGV]
    assert shell_calls == []


@pytest.mark.asyncio
async def test_local_spawn_without_a_covering_grant_never_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s4b1b_interfaces
) -> None:
    """Authority must coincide with the pin at the spawn: no
    network.remote-exec grant covering the destination, no spawn.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: ()

    async def _must_not_spawn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn site was reached without a grant")

    monkeypatch.setattr(bash_module, "spawn_argv_command", _must_not_spawn)

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashArgs(command=_S4B1B_COMMAND)))
    assert "no network.remote-exec grant covers the destination" in str(err.value)
    assert "deploy@8.8.8.8:22" in str(err.value)


@pytest.mark.asyncio
async def test_local_spawn_of_a_denied_form_fails_closed_before_the_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s4b1b_interfaces
) -> None:
    """A hostname destination whose resolution is ambiguous (the hermetic
    fake resolver answers example.com with two addresses) fails closed at
    execution: the plan re-derivation raises and nothing is spawned, even
    with a grant. (A unique non-local answer is admitted-with-pinning —
    pinned by test_resolved_address_rides_into_argv_per_spawn_point below.)
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: (_s4b1b_grant(repo),)

    async def _must_not_spawn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn site was reached for a denied form")

    monkeypatch.setattr(bash_module, "spawn_argv_command", _must_not_spawn)

    with pytest.raises(ToolError) as err:
        await collect_result(
            tool.run(BashArgs(command="ssh deploy@example.com uptime"))
        )
    assert "resolves to multiple addresses" in str(err.value)


class _RecordingAttestedToolIO:
    """A terminal-capable tool_io carrying a chosen remote-exec attestation."""

    def __init__(self, attestation: RemoteExecAttestation | None) -> None:
        self.requests: list[Any] = []
        self._attestation = attestation

    @property
    def supports_terminal(self) -> bool:
        return True

    @property
    def remote_exec_attestation(self) -> RemoteExecAttestation | None:
        return self._attestation

    async def run_shell(self, request: Any) -> Any:
        self.requests.append(request)
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)


def _s4b1b_terminal_ctx(tool_io: Any) -> InvokeContext:
    return InvokeContext(
        tool_call_id="call-s4b1b", session_id="session-1", tool_io=tool_io
    )


@pytest.mark.asyncio
async def test_terminal_transport_sends_the_pinned_command_to_an_attesting_client(
    tmp_path: Path, s4b1b_interfaces
) -> None:
    """The terminal transport is a client-owned executor: an execution-bound
    attestation (network-namespace locality AND SSH trust context) plus a
    covering grant lets the pinned command text ride into
    ShellCommandRequest.command.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: (_s4b1b_grant(repo),)
    tool_io = _RecordingAttestedToolIO(
        RemoteExecAttestation(network_locality=True, ssh_trust_context=True)
    )

    result = await collect_result(
        tool.run(BashArgs(command=_S4B1B_COMMAND), _s4b1b_terminal_ctx(tool_io))
    )

    assert result.exit_code == 0
    assert len(tool_io.requests) == 1
    assert tool_io.requests[0].command == _S4B1B_PINNED_COMMAND


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attestation",
    [
        None,  # no attestation entries (every ACP client, every older client)
        # locality only: an attestation of namespace locality alone is not an
        # execution-bound statement and hard-denies transfers (characterized
        # here; the discriminating assertion lands at S4b2c).
        RemoteExecAttestation(network_locality=True, ssh_trust_context=False),
        # trust context only: a differing network namespace denies.
        RemoteExecAttestation(network_locality=False, ssh_trust_context=True),
    ],
    ids=["none", "locality_only", "trust_only"],
)
async def test_terminal_transport_denies_a_non_execution_bound_client(
    tmp_path: Path, s4b1b_interfaces, attestation
) -> None:
    """X's differing-client-namespace arm: only an execution-bound statement
    (both entries) admits; anything else hard-denies before any terminal
    request is made — even with a covering grant.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: (_s4b1b_grant(repo),)
    tool_io = _RecordingAttestedToolIO(attestation)

    with pytest.raises(ToolError) as err:
        await collect_result(
            tool.run(BashArgs(command=_S4B1B_COMMAND), _s4b1b_terminal_ctx(tool_io))
        )
    assert "does not attest an execution-bound remote-exec context" in str(err.value)
    assert tool_io.requests == []


@pytest.mark.asyncio
async def test_terminal_transport_without_a_grant_sends_no_request(
    tmp_path: Path, s4b1b_interfaces
) -> None:
    repo = _s4b1b_repo(tmp_path)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: ()
    tool_io = _RecordingAttestedToolIO(
        RemoteExecAttestation(network_locality=True, ssh_trust_context=True)
    )

    with pytest.raises(ToolError) as err:
        await collect_result(
            tool.run(BashArgs(command=_S4B1B_COMMAND), _s4b1b_terminal_ctx(tool_io))
        )
    assert "no network.remote-exec grant covers the destination" in str(err.value)
    assert tool_io.requests == []


@pytest.mark.asyncio
async def test_terminal_transport_refuses_a_hostname_destination_before_any_request(
    tmp_path: Path, s4b1b_interfaces
) -> None:
    # The hermetic fake resolver answers example.com ambiguously, so the
    # hostname form is a resolution denial here; a unique non-local answer
    # is admitted-with-pinning (test_resolved_address_rides_into_argv_per_
    # spawn_point below).
    repo = _s4b1b_repo(tmp_path)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: (_s4b1b_grant(repo),)
    tool_io = _RecordingAttestedToolIO(
        RemoteExecAttestation(network_locality=True, ssh_trust_context=True)
    )

    with pytest.raises(ToolError) as err:
        await collect_result(
            tool.run(
                BashArgs(command="ssh deploy@example.com uptime"),
                _s4b1b_terminal_ctx(tool_io),
            )
        )
    assert "resolves to multiple addresses" in str(err.value)
    assert tool_io.requests == []


class _FakeStartResult:
    def model_copy(self, update: Any = None) -> Any:
        return self


class _RecordingStartPort(BackgroundJobsPort):
    def __init__(self) -> None:
        self.started: list[str] = []

    async def start(
        self, args: Any, *, cwd: Any = None, execute_command: str | None = None
    ) -> Any:
        self.started.append(str(execute_command))
        return _FakeStartResult()


@pytest.mark.asyncio
async def test_background_transport_spawns_the_pinned_command_text(
    tmp_path: Path, s4b1b_interfaces
) -> None:
    """The background transport is this process's own executor (attested by
    construction): with a covering grant it starts the job with the pinned
    command text — the approved address + fixed host-key identity in the
    spawned argv.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = BashStart(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo
    )
    tool.dangerous_command_authorities_getter = lambda: (_s4b1b_grant(repo),)
    port = _RecordingStartPort()
    ctx = InvokeContext(tool_call_id="call-s4b1b-bg", background_jobs=port)

    await collect_result(tool.run(BashStartArgs(command=_S4B1B_COMMAND), ctx))

    assert port.started == [_S4B1B_PINNED_COMMAND]


@pytest.mark.asyncio
async def test_background_transport_without_a_grant_refuses_before_the_start(
    tmp_path: Path, s4b1b_interfaces
) -> None:
    repo = _s4b1b_repo(tmp_path)
    tool = BashStart(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo
    )
    tool.dangerous_command_authorities_getter = lambda: ()
    port = _RecordingStartPort()
    ctx = InvokeContext(tool_call_id="call-s4b1b-bg2", background_jobs=port)

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashStartArgs(command=_S4B1B_COMMAND), ctx))
    assert "no network.remote-exec grant covers the destination" in str(err.value)
    assert port.started == []


@pytest.mark.asyncio
async def test_background_transport_refuses_a_hostname_destination(
    tmp_path: Path, s4b1b_interfaces
) -> None:
    # Ambiguous hermetic resolution of example.com: denied before the start;
    # a unique non-local answer is admitted-with-pinning (the S4b2a argv arm).
    repo = _s4b1b_repo(tmp_path)
    tool = BashStart(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo
    )
    tool.dangerous_command_authorities_getter = lambda: (_s4b1b_grant(repo),)
    port = _RecordingStartPort()
    ctx = InvokeContext(tool_call_id="call-s4b1b-bg3", background_jobs=port)

    with pytest.raises(ToolError) as err:
        await collect_result(
            tool.run(BashStartArgs(command="ssh deploy@example.com uptime"), ctx)
        )
    assert "resolves to multiple addresses" in str(err.value)
    assert port.started == []


# --- S4b2a: hostname resolve-validate-pin — the resolved address rides into
# the ACTUAL argv at EVERY spawn point (BBB/JJJ F4) ---------------------------
#
# A hostname destination is admitted only when the system resolver (honoring
# /etc/hosts) answers uniquely and non-locally; the RESOLVED address becomes
# the pinned destination operand of the spawned argv while the host-key
# identity (HostKeyAlias) stays bound to the granted hostname. The spawn-
# observing arms below prove it per spawn point; S4b2's mutation (spawn runs
# the original hostname text instead of the resolved pinned address) fails
# this node.

_S4B2A_COMMAND = "ssh deploy@example.com uptime"
_S4B2A_RESOLVED = "93.184.216.34"
_S4B2A_PINNED_ARGV = (
    "ssh",
    "-o",
    "UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts",
    "-o",
    "HostKeyAlias=deploy@example.com:22",
    "-o",
    "StrictHostKeyChecking=yes",
    "-o",
    "BatchMode=yes",
    f"deploy@{_S4B2A_RESOLVED}",
    "uptime",
)
_S4B2A_PINNED_COMMAND = (
    "ssh -o UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts"
    " -o HostKeyAlias=deploy@example.com:22 -o StrictHostKeyChecking=yes"
    " -o BatchMode=yes"
    f" deploy@{_S4B2A_RESOLVED} uptime"
)


@pytest.fixture
def s4b2a_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic S4b2a seams: example.com resolves to one non-local address
    through the module resolver seam; any other host fails resolution; the
    local-interface enumeration is fixed. No real DNS, no host interfaces.
    """
    import ipaddress

    from chartreux.core.tools.builtins import _remote_exec

    def fake_resolver(host: str) -> frozenset:
        if host == "example.com":
            return frozenset({ipaddress.ip_address(_S4B2A_RESOLVED)})
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(_remote_exec, "_hostname_resolver", fake_resolver)
    monkeypatch.setattr(
        _remote_exec,
        "_interface_addresses",
        lambda: frozenset({
            ipaddress.ip_address("192.168.10.20"),
            ipaddress.ip_address("::1"),
        }),
    )


def _s4b2a_grant(repo: Path) -> DangerousCommandAuthority:
    scope = resolve_repository_scope(repo, GitRepositoryRelocation())
    assert scope is not None
    return DangerousCommandAuthority(
        operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
        scope=scope,
        source="runtime_grant",
        destination=canonical_remote_destination("deploy@example.com:22"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("spawn_point", ["local_argv", "terminal", "background"])
async def test_resolved_address_rides_into_argv_per_spawn_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s4b2a_resolution, spawn_point: str
) -> None:
    """Per spawn point, the hostname-RESOLVED address and the fixed host-key
    identity ride into the ACTUAL argv the transport receives: the destination
    operand is ``deploy@93.184.216.34`` (never the original hostname text) and
    ``HostKeyAlias`` stays bound to the granted hostname ``deploy@example.com``.
    """
    repo = _s4b1b_repo(tmp_path)
    grant = _s4b2a_grant(repo)
    if spawn_point == "local_argv":
        tool = _cwd_tool(repo)
        tool.dangerous_command_authorities_getter = lambda: (grant,)
        argv_calls: list[tuple[str, ...]] = []

        async def _record_argv(argv: Any, **kwargs: Any) -> Any:
            argv_calls.append(tuple(argv))
            return _FakeProc()

        async def _must_not_spawn_shell(*args: object, **kwargs: object) -> None:
            raise AssertionError("a pinned remote-exec plan never runs as shell text")

        monkeypatch.setattr(bash_module, "spawn_argv_command", _record_argv)
        monkeypatch.setattr(bash_module, "spawn_shell_command", _must_not_spawn_shell)

        result = await collect_result(tool.run(BashArgs(command=_S4B2A_COMMAND)))

        assert result.exit_code == 0
        assert argv_calls == [_S4B2A_PINNED_ARGV]
        # Discriminating shape: resolved address as operand, hostname only in
        # the host-key alias pin.
        assert f"deploy@{_S4B2A_RESOLVED}" in argv_calls[0]
        assert "HostKeyAlias=deploy@example.com:22" in argv_calls[0]
        assert (
            "UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts" in argv_calls[0]
        )
        assert "deploy@example.com" not in argv_calls[0][-2]
    elif spawn_point == "terminal":
        tool = _cwd_tool(repo)
        tool.dangerous_command_authorities_getter = lambda: (grant,)
        tool_io = _RecordingAttestedToolIO(
            RemoteExecAttestation(network_locality=True, ssh_trust_context=True)
        )

        result = await collect_result(
            tool.run(BashArgs(command=_S4B2A_COMMAND), _s4b1b_terminal_ctx(tool_io))
        )

        assert result.exit_code == 0
        assert len(tool_io.requests) == 1
        sent = tool_io.requests[0].command
        assert sent == _S4B2A_PINNED_COMMAND
        assert f"deploy@{_S4B2A_RESOLVED}" in sent
        assert "HostKeyAlias=deploy@example.com:22" in sent
    else:
        tool = BashStart(
            config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo
        )
        tool.dangerous_command_authorities_getter = lambda: (grant,)
        port = _RecordingStartPort()
        ctx = InvokeContext(tool_call_id="call-s4b2a-bg", background_jobs=port)

        await collect_result(tool.run(BashStartArgs(command=_S4B2A_COMMAND), ctx))

        assert port.started == [_S4B2A_PINNED_COMMAND]
        assert f"deploy@{_S4B2A_RESOLVED}" in port.started[0]
        assert "HostKeyAlias=deploy@example.com:22" in port.started[0]


@pytest.mark.asyncio
async def test_hostname_spawn_without_a_covering_grant_never_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s4b2a_resolution
) -> None:
    """Authority must coincide with the resolved pin at the spawn: the grant
    is keyed on the hostname, so a literal-address grant never admits the
    hostname call — nothing is spawned.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: (_s4b1b_grant(repo),)

    async def _must_not_spawn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn site was reached without a covering grant")

    monkeypatch.setattr(bash_module, "spawn_argv_command", _must_not_spawn)

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashArgs(command=_S4B2A_COMMAND)))
    assert "no network.remote-exec grant covers the destination" in str(err.value)
    assert "deploy@example.com:22" in str(err.value)


# --- S4b2b: pinned-plan rebinding safety — a CHANGED resolution is never
# contacted (HHH) -----------------------------------------------------------
#
# The hermetic stateful resolver answers example.com with the validated
# non-local address on the FIRST observation (the plan's validated pin) and
# with the REBOUND answer on every later observation (the spawn-time
# re-observation inside the same plan transaction). Per spawn point the arms
# below assert the HHH safety invariant: the rebound answer NEVER reaches the
# spawned argv/command, and the outcome is exactly one of the two conforming
# halves — the spawn is DENIED, or the ORIGINAL validated pinned address rides
# into the argv. The S4b2b mutation (the rebound answer substituted for the
# validated pin at the spawn site) fails these nodes.

_S4B2B_REBOUND_LOCAL = "127.0.0.1"
_S4B2B_REBOUND_UNSPECIFIED = "0.0.0.0"
_S4B2B_REBOUND_OTHER = "1.1.1.1"


@pytest.fixture
def s4b2b_rebind(monkeypatch: pytest.MonkeyPatch):
    """Stateful rebind seam: first resolution -> the validated pin, every
    later resolution -> the rebound (changed) answer.

    Returns an installer taking the rebound answer spelling. No real DNS, no
    host interfaces.
    """
    import ipaddress

    from chartreux.core.tools.builtins import _remote_exec

    def _install(rebound: str = _S4B2B_REBOUND_LOCAL) -> None:
        state = {"calls": 0}

        def fake_resolver(host: str) -> frozenset:
            if host != "example.com":
                raise socket.gaierror(-2, "Name or service not known")
            state["calls"] += 1
            answer = _S4B2A_RESOLVED if state["calls"] == 1 else rebound
            return frozenset({ipaddress.ip_address(answer)})

        monkeypatch.setattr(_remote_exec, "_hostname_resolver", fake_resolver)
        monkeypatch.setattr(
            _remote_exec,
            "_interface_addresses",
            lambda: frozenset({
                ipaddress.ip_address("192.168.10.20"),
                ipaddress.ip_address("::1"),
            }),
        )

    _install()
    return _install


@pytest.mark.asyncio
@pytest.mark.parametrize("spawn_point", ["local_argv", "terminal", "background"])
@pytest.mark.parametrize(
    "rebound", [_S4B2B_REBOUND_LOCAL, _S4B2B_REBOUND_UNSPECIFIED, _S4B2B_REBOUND_OTHER]
)
async def test_rebound_resolution_never_reaches_the_spawn_per_spawn_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    s4b2b_rebind,
    spawn_point: str,
    rebound: str,
) -> None:
    """HHH safety invariant, per spawn point: example.com resolves to the
    validated non-local pin at the validation observation and to the REBOUND
    answer at the spawn re-observation. The rebound answer never reaches any
    spawned argv/command; the outcome is one of the two conforming halves —
    the spawn is denied, or the ORIGINAL pinned address rides into the argv.
    """
    s4b2b_rebind(rebound)
    repo = _s4b1b_repo(tmp_path)
    grant = _s4b2a_grant(repo)
    spawned: list[str] = []
    denied = False

    if spawn_point == "local_argv":
        tool = _cwd_tool(repo)
        tool.dangerous_command_authorities_getter = lambda: (grant,)

        async def _record_argv(argv: Any, **kwargs: Any) -> Any:
            spawned.append(" ".join(str(word) for word in argv))
            return _FakeProc()

        async def _must_not_spawn_shell(*args: object, **kwargs: object) -> None:
            raise AssertionError("a pinned remote-exec plan never runs as shell text")

        monkeypatch.setattr(bash_module, "spawn_argv_command", _record_argv)
        monkeypatch.setattr(bash_module, "spawn_shell_command", _must_not_spawn_shell)
        try:
            await collect_result(tool.run(BashArgs(command=_S4B2A_COMMAND)))
        except ToolError:
            denied = True
    elif spawn_point == "terminal":
        tool = _cwd_tool(repo)
        tool.dangerous_command_authorities_getter = lambda: (grant,)
        tool_io = _RecordingAttestedToolIO(
            RemoteExecAttestation(network_locality=True, ssh_trust_context=True)
        )
        try:
            await collect_result(
                tool.run(BashArgs(command=_S4B2A_COMMAND), _s4b1b_terminal_ctx(tool_io))
            )
        except ToolError:
            denied = True
        spawned.extend(str(request.command) for request in tool_io.requests)
    else:
        tool = BashStart(
            config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo
        )
        tool.dangerous_command_authorities_getter = lambda: (grant,)
        port = _RecordingStartPort()
        ctx = InvokeContext(tool_call_id="call-s4b2b-bg", background_jobs=port)
        try:
            await collect_result(tool.run(BashStartArgs(command=_S4B2A_COMMAND), ctx))
        except ToolError:
            denied = True
        spawned.extend(port.started)

    # The safety invariant, whichever half holds: the rebound (changed)
    # answer never reaches any spawn.
    for text in spawned:
        assert rebound not in text
    if denied:
        # Deny half (HHH outcome a): nothing reached any spawn at all.
        assert spawned == []
    else:
        # Execute-original-pinned half (HHH outcome b): the spawn carries
        # the ORIGINAL validated pinned address and the fixed host-key
        # identity bound to the granted hostname — never the changed answer.
        assert spawned
        for text in spawned:
            assert f"deploy@{_S4B2A_RESOLVED}" in text
            assert "HostKeyAlias=deploy@example.com:22" in text
            # The destination operand word is the resolved address, never the
            # original hostname text.
            assert "deploy@example.com" not in text.split()[-2]


@pytest.mark.asyncio
async def test_rebound_to_local_is_never_contacted_at_the_local_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s4b2b_rebind
) -> None:
    """The re-resolved-to-local arm at the local argv spawn, both HHH halves:
    the spawn is DENIED with a discriminating rebind reason (the changed
    local answer and the validated pin both named), OR the ORIGINAL pinned
    address rides into the argv; the rebound address never reaches the spawn
    either way.
    """
    s4b2b_rebind(_S4B2B_REBOUND_LOCAL)
    repo = _s4b1b_repo(tmp_path)
    grant = _s4b2a_grant(repo)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: (grant,)

    spawned: list[tuple[str, ...]] = []

    async def _record_argv(argv: Any, **kwargs: Any) -> Any:
        spawned.append(tuple(argv))
        return _FakeProc()

    monkeypatch.setattr(bash_module, "spawn_argv_command", _record_argv)

    denied = False
    message = ""
    try:
        await collect_result(tool.run(BashArgs(command=_S4B2A_COMMAND)))
    except ToolError as exc:
        denied = True
        message = str(exc)

    # The changed answer never reached the spawn, whichever half holds.
    for argv in spawned:
        assert _S4B2B_REBOUND_LOCAL not in " ".join(argv)
    if denied:
        # Deny half (HHH outcome a): nothing spawned, and the denial names
        # the rebind — the changed local answer and the validated pin.
        assert spawned == []
        assert "never contacted" in message
        assert _S4B2B_REBOUND_LOCAL in message
        assert _S4B2A_RESOLVED in message
    else:
        # Execute-original-pinned half (HHH outcome b): the argv carries
        # exactly the validated pinned plan.
        assert spawned == [_S4B2A_PINNED_ARGV]


# --- S4b2c: bounded transfers (scp/rsync) — the transfer rebinding arm and
# X's attestation-content arms, spawn-observing -------------------------------
#
# The transfer grammar classifies at the same seam as ssh; the pinned plan
# carries the resolved address + fixed host-key identity (scp: leading -o
# pins; rsync: the injected -e pinned ssh transport). The rebinding arm below
# reuses the s4b2b_rebind stateful seam for scp/rsync: the rebound answer
# NEVER reaches any spawn, and the outcome is one of the two conforming HHH
# halves. X's second arm proves, per spawn point, that admitted same-network
# ssh execution does NOT extend to a transfer on a client whose executor
# attests network-namespace locality ONLY (no filesystem/SSH trust context):
# the transfer is denied before any request, and on the server-owned local
# spawn the ssh execution in the same scenario IS admitted.

_S4B2C_SCP_COMMAND = "scp file deploy@example.com:backup"
_S4B2C_RSYNC_COMMAND = "rsync -a src/ deploy@example.com:dst/"


def _s4b2c_pinned_scp_argv() -> tuple[str, ...]:
    return (
        "scp",
        "-o",
        "UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts",
        "-o",
        "HostKeyAlias=deploy@example.com:22",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "BatchMode=yes",
        "file",
        f"deploy@{_S4B2A_RESOLVED}:backup",
    )


def _s4b2c_pinned_rsync_command() -> str:
    return (
        "rsync -e 'ssh"
        " -o UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts"
        " -o HostKeyAlias=deploy@example.com:22 -o StrictHostKeyChecking=yes"
        " -o BatchMode=yes' -a src/"
        f" deploy@{_S4B2A_RESOLVED}:dst/"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("spawn_point", ["local_argv", "terminal", "background"])
@pytest.mark.parametrize(
    "rebound", [_S4B2B_REBOUND_LOCAL, _S4B2B_REBOUND_UNSPECIFIED, _S4B2B_REBOUND_OTHER]
)
@pytest.mark.parametrize(
    "command", [_S4B2C_SCP_COMMAND, _S4B2C_RSYNC_COMMAND], ids=["scp", "rsync"]
)
async def test_transfer_rebound_resolution_never_reaches_the_spawn_per_spawn_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    s4b2b_rebind,
    spawn_point: str,
    rebound: str,
    command: str,
) -> None:
    """HHH safety invariant for transfers, per spawn point: example.com
    resolves to the validated non-local pin at the validation observation and
    to the REBOUND answer at the spawn re-observation. The rebound answer
    never reaches any spawned argv/command; the outcome is one of the two
    conforming halves — the spawn is denied, or the ORIGINAL pinned address
    rides into the argv.
    """
    s4b2b_rebind(rebound)
    repo = _s4b1b_repo(tmp_path)
    grant = _s4b2a_grant(repo)
    spawned: list[str] = []
    denied = False

    if spawn_point == "local_argv":
        tool = _cwd_tool(repo)
        tool.dangerous_command_authorities_getter = lambda: (grant,)

        async def _record_argv(argv: Any, **kwargs: Any) -> Any:
            spawned.append(" ".join(str(word) for word in argv))
            return _FakeProc()

        async def _must_not_spawn_shell(*args: object, **kwargs: object) -> None:
            raise AssertionError("a pinned remote-exec plan never runs as shell text")

        monkeypatch.setattr(bash_module, "spawn_argv_command", _record_argv)
        monkeypatch.setattr(bash_module, "spawn_shell_command", _must_not_spawn_shell)
        try:
            await collect_result(tool.run(BashArgs(command=command)))
        except ToolError:
            denied = True
    elif spawn_point == "terminal":
        tool = _cwd_tool(repo)
        tool.dangerous_command_authorities_getter = lambda: (grant,)
        tool_io = _RecordingAttestedToolIO(
            RemoteExecAttestation(network_locality=True, ssh_trust_context=True)
        )
        try:
            await collect_result(
                tool.run(BashArgs(command=command), _s4b1b_terminal_ctx(tool_io))
            )
        except ToolError:
            denied = True
        spawned.extend(str(request.command) for request in tool_io.requests)
    else:
        tool = BashStart(
            config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo
        )
        tool.dangerous_command_authorities_getter = lambda: (grant,)
        port = _RecordingStartPort()
        ctx = InvokeContext(tool_call_id="call-s4b2c-bg", background_jobs=port)
        try:
            await collect_result(tool.run(BashStartArgs(command=command), ctx))
        except ToolError:
            denied = True
        spawned.extend(port.started)

    # The safety invariant, whichever half holds: the rebound (changed)
    # answer never reaches any spawn.
    for text in spawned:
        assert rebound not in text
    if denied:
        # Deny half (HHH outcome a): nothing reached any spawn at all.
        assert spawned == []
    else:
        # Execute-original-pinned half (HHH outcome b): the spawn carries
        # the ORIGINAL validated pinned address and the fixed host-key
        # identity bound to the granted hostname — never the changed answer.
        assert spawned
        for text in spawned:
            assert f"deploy@{_S4B2A_RESOLVED}:" in text
            assert "HostKeyAlias=deploy@example.com:22" in text
            assert rebound not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("spawn_point", ["local_argv", "terminal", "background"])
@pytest.mark.parametrize(
    "command", [_S4B2C_SCP_COMMAND, _S4B2C_RSYNC_COMMAND], ids=["scp", "rsync"]
)
async def test_transfer_pins_address_and_host_key_identity_per_spawn_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    s4b2a_resolution,
    spawn_point: str,
    command: str,
) -> None:
    """Per spawn point, a bounded hostname transfer runs with the pinned
    plan: the RESOLVED address is the remote operand's host (never the
    original hostname text) and the fixed host-key identity rides along —
    scp as leading -o pins, rsync as the injected -e pinned ssh transport.
    """
    repo = _s4b1b_repo(tmp_path)
    grant = _s4b2a_grant(repo)
    if spawn_point == "local_argv":
        tool = _cwd_tool(repo)
        tool.dangerous_command_authorities_getter = lambda: (grant,)
        argv_calls: list[tuple[str, ...]] = []

        async def _record_argv(argv: Any, **kwargs: Any) -> Any:
            argv_calls.append(tuple(argv))
            return _FakeProc()

        async def _must_not_spawn_shell(*args: object, **kwargs: object) -> None:
            raise AssertionError("a pinned remote-exec plan never runs as shell text")

        monkeypatch.setattr(bash_module, "spawn_argv_command", _record_argv)
        monkeypatch.setattr(bash_module, "spawn_shell_command", _must_not_spawn_shell)

        result = await collect_result(tool.run(BashArgs(command=command)))

        assert result.exit_code == 0
        assert len(argv_calls) == 1
        argv = argv_calls[0]
        assert argv[0] in ("scp", "rsync")
        assert f"deploy@{_S4B2A_RESOLVED}:" in argv[-1]
        assert "deploy@example.com" not in argv[-1]
        assert "HostKeyAlias=deploy@example.com:22" in " ".join(argv)
        assert "UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts" in " ".join(
            argv
        )
        if argv[0] == "scp":
            assert argv == _s4b2c_pinned_scp_argv()
    elif spawn_point == "terminal":
        tool = _cwd_tool(repo)
        tool.dangerous_command_authorities_getter = lambda: (grant,)
        tool_io = _RecordingAttestedToolIO(
            RemoteExecAttestation(network_locality=True, ssh_trust_context=True)
        )

        result = await collect_result(
            tool.run(BashArgs(command=command), _s4b1b_terminal_ctx(tool_io))
        )

        assert result.exit_code == 0
        assert len(tool_io.requests) == 1
        sent = tool_io.requests[0].command
        assert f"deploy@{_S4B2A_RESOLVED}:" in sent
        assert "HostKeyAlias=deploy@example.com:22" in sent
        if command == _S4B2C_RSYNC_COMMAND:
            assert sent == _s4b2c_pinned_rsync_command()
    else:
        tool = BashStart(
            config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo
        )
        tool.dangerous_command_authorities_getter = lambda: (grant,)
        port = _RecordingStartPort()
        ctx = InvokeContext(tool_call_id="call-s4b2c-bg2", background_jobs=port)

        await collect_result(tool.run(BashStartArgs(command=command), ctx))

        assert len(port.started) == 1
        assert f"deploy@{_S4B2A_RESOLVED}:" in port.started[0]
        assert "HostKeyAlias=deploy@example.com:22" in port.started[0]


@pytest.mark.asyncio
async def test_same_scenario_admitted_ssh_exec_and_denied_transfer_on_a_locality_only_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s4b2a_resolution
) -> None:
    """X's second arm, spawn-observing, one scenario: the ADMITTED
    same-network ssh execution (the server-owned local spawn, attested by
    construction) does NOT extend to a transfer on the client-owned terminal
    transport whose executor attests network-namespace locality ONLY — no
    filesystem/SSH trust context. The ssh call spawns with its pinned argv;
    the transfer is denied before any terminal request is made.
    """
    repo = _s4b1b_repo(tmp_path)
    grant = _s4b2a_grant(repo)

    # Half 1: the admitted ssh execution on the local argv spawn.
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: (grant,)
    argv_calls: list[tuple[str, ...]] = []

    async def _record_argv(argv: Any, **kwargs: Any) -> Any:
        argv_calls.append(tuple(argv))
        return _FakeProc()

    monkeypatch.setattr(bash_module, "spawn_argv_command", _record_argv)
    result = await collect_result(tool.run(BashArgs(command=_S4B2A_COMMAND)))
    assert result.exit_code == 0
    assert argv_calls == [_S4B2A_PINNED_ARGV]

    # Half 2: the SAME transfer on the SAME scenario's client-owned terminal
    # transport with a LOCALITY-ONLY attestation entry — denied, no request.
    tool_io = _RecordingAttestedToolIO(
        RemoteExecAttestation(network_locality=True, ssh_trust_context=False)
    )
    with pytest.raises(ToolError) as err:
        await collect_result(
            tool.run(BashArgs(command=_S4B2C_SCP_COMMAND), _s4b1b_terminal_ctx(tool_io))
        )
    assert "filesystem/SSH trust context" in str(err.value)
    assert "locality only hard-denies transfers" in str(err.value)
    assert tool_io.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attestation",
    [
        None,  # no attestation entries (every ACP client, every older client)
        # locality only: an attestation of namespace locality alone is not an
        # execution-bound statement and hard-denies transfers (X's
        # discriminating assertion, assigned to S4b2c).
        RemoteExecAttestation(network_locality=True, ssh_trust_context=False),
        # trust context only: a differing network namespace denies.
        RemoteExecAttestation(network_locality=False, ssh_trust_context=True),
    ],
    ids=["none", "locality_only", "trust_only"],
)
@pytest.mark.parametrize(
    "command", [_S4B2C_SCP_COMMAND, _S4B2C_RSYNC_COMMAND], ids=["scp", "rsync"]
)
async def test_terminal_transport_denies_a_transfer_to_a_non_execution_bound_client(
    tmp_path: Path, s4b2a_resolution, attestation, command: str
) -> None:
    """A locality-only attestation entry (network_locality without
    ssh_trust_context) hard-denies the transfer at the client-owned spawn:
    no terminal request is made, even with a covering destination grant —
    the transfer's footprint crosses into the filesystem/SSH trust context
    the locality claim does not cover.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: (_s4b2a_grant(repo),)
    tool_io = _RecordingAttestedToolIO(attestation)

    with pytest.raises(ToolError) as err:
        await collect_result(
            tool.run(BashArgs(command=command), _s4b1b_terminal_ctx(tool_io))
        )
    assert "filesystem/SSH trust context" in str(err.value)
    assert "hard-denies transfers" in str(err.value)
    assert tool_io.requests == []


@pytest.mark.asyncio
async def test_terminal_transport_sends_the_pinned_transfer_to_an_execution_bound_client(
    tmp_path: Path, s4b2a_resolution
) -> None:
    """Positive control (the gate is discriminating, not refuse-all): an
    execution-bound attestation (both entries) plus a covering destination
    grant lets the pinned transfer command ride into ShellCommandRequest.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: (_s4b2a_grant(repo),)
    tool_io = _RecordingAttestedToolIO(
        RemoteExecAttestation(network_locality=True, ssh_trust_context=True)
    )

    result = await collect_result(
        tool.run(BashArgs(command=_S4B2C_RSYNC_COMMAND), _s4b1b_terminal_ctx(tool_io))
    )

    assert result.exit_code == 0
    assert len(tool_io.requests) == 1
    assert tool_io.requests[0].command == _s4b2c_pinned_rsync_command()


@pytest.mark.asyncio
async def test_local_spawn_of_a_transfer_without_a_covering_grant_never_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s4b2a_resolution
) -> None:
    """Authority must coincide with the pin at the spawn: no
    network.remote-exec grant covering the destination, no spawn.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: ()

    async def _must_not_spawn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn site was reached without a grant")

    monkeypatch.setattr(bash_module, "spawn_argv_command", _must_not_spawn)

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashArgs(command=_S4B2C_SCP_COMMAND)))
    assert "no network.remote-exec grant covers the destination" in str(err.value)
    assert "deploy@example.com:22" in str(err.value)


@pytest.mark.asyncio
async def test_local_spawn_of_a_denied_transfer_form_fails_closed_before_the_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s4b2a_resolution
) -> None:
    """A denied transfer spelling (no user in the remote spec) fails closed
    at execution even with a grant: the plan re-derivation raises and nothing
    is spawned.
    """
    repo = _s4b1b_repo(tmp_path)
    tool = _cwd_tool(repo)
    tool.dangerous_command_authorities_getter = lambda: (_s4b2a_grant(repo),)

    async def _must_not_spawn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn site was reached for a denied form")

    monkeypatch.setattr(bash_module, "spawn_argv_command", _must_not_spawn)

    with pytest.raises(ToolError) as err:
        await collect_result(tool.run(BashArgs(command="scp file host:backup")))
    assert "must name a user" in str(err.value)
