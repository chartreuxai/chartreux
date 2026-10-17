"""Bash tool execution tests for the bash suite (echo/cat execution, detached background jobs, timeouts, output truncation and drain caps, off-loop decoding, utf-32/latin-1 char caps, workdir resolution, benign unlisted execution, pager inheritance for local shell children, the CapturedShellResult returncode-alias contract, and the end-to-end expansion executions)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import threading
from time import monotonic

import pytest

from chartreux.core.tools.base import BaseToolState, ToolError, ToolPermission
import chartreux.core.tools.builtins.bash as bash_module
from chartreux.core.tools.builtins.bash import (
    Bash,
    BashArgs,
    BashToolConfig,
    CapturedShellResult,
)
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
