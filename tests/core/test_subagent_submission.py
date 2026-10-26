"""Submission-boundary secret-scan gate tests (P2-i chunk 2).

The gate runs inside
:meth:`chartreux.core.subagents.SubagentRunAccumulator.build_result` — the
single assembly point every subagent submission result flows through, on both
the foreground and the background publication paths. The accumulator
collects the session's touched paths (repo paths created or edited through
the file tools) from the tool results it observes; build_result scans them
before the result is surfaced as final.

Failure policy under test: a seeded secret-bearing artifact blocks the
submission (blocked, redacted, agent informed), a clean session's submission
passes unaffected, and the agent-facing block message never carries the
secret, the rule id, or the matched line.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
import pytest

from chartreux.core.events import AssistantEvent, ToolResultEvent
from chartreux.core.subagent_submission import (
    BLOCK_MESSAGE_MAX_PATHS,
    UNKNOWN_WRITE_FOOTPRINT,
    bash_write_paths_from_result_event,
    blocked_submission_response,
    touched_path_from_result_event,
)
from chartreux.core.subagents import EmptySubagentResponseError, SubagentRunAccumulator
from chartreux.core.tools.base import BaseToolState
from chartreux.core.tools.builtins.bash import (
    Bash,
    BashArgs,
    BashToolConfig,
    CapturedShellResult,
)
from chartreux.core.tools.builtins.bash_start import BashStart
from chartreux.core.tools.builtins.edit import Edit, EditResult
from chartreux.core.tools.builtins.read_file import ReadFile, ReadFileResult
from chartreux.core.tools.builtins.write_file import WriteFile, WriteFileResult
from chartreux.core.tools.secret_scan import (
    DEFAULT_MAX_SCAN_BYTES,
    SecretFinding,
    scan_paths,
)
from tests.mock.utils import collect_result

GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
SECRET_LINE = f"GITHUB_TOKEN = {GITHUB_TOKEN}"
RULE_ID = "github-token"


def _edit_event(path: str, *, skipped: bool = False) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="edit",
        tool_class=Edit,
        result=EditResult(file=path, message="edited", old_string="a", new_string="b"),
        tool_call_id="edit-1",
        skipped=skipped,
    )


def _write_event(path: str) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="write_file",
        tool_class=WriteFile,
        result=WriteFileResult(file_path=path, bytes_written=1, content="x"),
        tool_call_id="write-1",
    )


def _read_event(path: str) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="read_file",
        tool_class=ReadFile,
        result=ReadFileResult(file_path=path, content="x", num_lines=1, start_line=1),
        tool_call_id="read-1",
    )


def test_touched_path_extracted_from_file_mutation_results() -> None:
    assert touched_path_from_result_event(_edit_event("/repo/a.py")) == "/repo/a.py"
    assert touched_path_from_result_event(_write_event("/repo/b.py")) == "/repo/b.py"


def test_touched_path_ignores_reads_skips_and_bare_results() -> None:
    # A read result carries the same `file_path` attribute as a write result;
    # only the mutation effect kinds are scan scope.
    assert touched_path_from_result_event(_read_event("/repo/a.py")) is None
    assert (
        touched_path_from_result_event(_edit_event("/repo/a.py", skipped=True)) is None
    )
    assert (
        touched_path_from_result_event(
            ToolResultEvent(tool_name="edit", tool_class=None, tool_call_id="c")
        )
        is None
    )
    assert (
        touched_path_from_result_event(
            ToolResultEvent(
                tool_name="edit", tool_class=Edit, result=None, tool_call_id="c"
            )
        )
        is None
    )


def test_accumulator_collects_each_touched_path_once(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    accumulator = SubagentRunAccumulator()
    accumulator.observe(_edit_event(str(first)), tool_call_id="task")
    accumulator.observe(_write_event(str(second)), tool_call_id="task")
    accumulator.observe(_edit_event(str(first)), tool_call_id="task")
    accumulator.observe(_read_event(str(first)), tool_call_id="task")
    assert list(accumulator._touched_paths) == [str(first), str(second)]


def test_build_result_blocks_secret_bearing_touched_path(tmp_path: Path) -> None:
    artifact = tmp_path / "deploy.py"
    artifact.write_text(f"# config\n{SECRET_LINE}\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="wrote the config"), tool_call_id="task")
    accumulator.observe(_write_event(str(artifact)), tool_call_id="task")

    result = accumulator.build_result(turns_used=2)

    # The submission is blocked: it never surfaces as final, and the agent's
    # response is replaced by the redacted block message.
    assert result.completed is False
    assert result.turns_used == 2
    assert "wrote the config" not in result.response
    assert "failed the secret scan" in result.response
    assert str(artifact) in result.response
    # No disclosure: no secret, no rule id, no matched line.
    assert GITHUB_TOKEN not in result.response
    assert RULE_ID not in result.response
    assert SECRET_LINE not in result.response


def test_build_result_blocks_even_when_the_run_did_not_complete(tmp_path: Path) -> None:
    # Fail-closed: an incomplete or failed run's partial result is a
    # submission too, and a secret-bearing artifact blocks it all the same.
    artifact = tmp_path / "deploy.py"
    artifact.write_text(f"{SECRET_LINE}\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="partial"), tool_call_id="task")
    accumulator.observe(_edit_event(str(artifact)), tool_call_id="task")

    result = accumulator.build_result(turns_used=1, completed=False)

    assert result.completed is False
    assert "failed the secret scan" in result.response
    assert GITHUB_TOKEN not in result.response


def test_build_result_passes_clean_touched_path(tmp_path: Path) -> None:
    artifact = tmp_path / "clean.py"
    artifact.write_text(
        "def add(left: int, right: int) -> int:\n    return left + right\n",
        encoding="utf-8",
    )
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="wrote the config"), tool_call_id="task")
    accumulator.observe(_write_event(str(artifact)), tool_call_id="task")

    result = accumulator.build_result(turns_used=2)

    assert result.completed is True
    assert result.response == "wrote the config"


def test_build_result_without_touched_paths_is_unchanged() -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="done"), tool_call_id="task")
    assert accumulator.build_result(turns_used=1) == accumulator.build_result(
        turns_used=1
    )
    assert accumulator.build_result(turns_used=1).response == "done"

    empty = SubagentRunAccumulator()
    with pytest.raises(EmptySubagentResponseError, match="empty response"):
        empty.build_result(turns_used=0)


def test_build_result_skips_deleted_artifacts(tmp_path: Path) -> None:
    # A touched path deleted before submission is skipped by the scanner, so
    # the submission passes.
    artifact = tmp_path / "gone.py"
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="done"), tool_call_id="task")
    accumulator.observe(_write_event(str(artifact)), tool_call_id="task")
    assert accumulator.build_result(turns_used=1).response == "done"


def test_build_result_blocks_an_artifact_the_scan_cannot_finish(tmp_path: Path) -> None:
    # Fail-closed: an artifact larger than the per-file scan cap is not a
    # clean artifact. The submission is blocked and the agent is told the
    # artifact could not be fully inspected — no rule id, no content.
    artifact = tmp_path / "big-artifact.txt"
    artifact.write_text("x" * DEFAULT_MAX_SCAN_BYTES + "\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(
        AssistantEvent(content="wrote the artifact"), tool_call_id="task"
    )
    accumulator.observe(_write_event(str(artifact)), tool_call_id="task")

    result = accumulator.build_result(turns_used=1)

    assert result.completed is False
    assert "could not be fully inspected" in result.response
    assert str(artifact) in result.response
    assert "scan-incomplete" not in result.response


def test_build_result_blocks_a_secret_behind_a_nul_byte(tmp_path: Path) -> None:
    # The submission gate scans the same content the hook scans: a
    # NUL-containing artifact is split on NUL boundaries and every text run
    # is scanned, so a secret appended behind a NUL still blocks.
    artifact = tmp_path / "blob.bin"
    artifact.write_bytes(b"\x00\x01" + GITHUB_TOKEN.encode("utf-8"))
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="wrote the blob"), tool_call_id="task")
    accumulator.observe(_write_event(str(artifact)), tool_call_id="task")

    result = accumulator.build_result(turns_used=1)

    assert result.completed is False
    assert "failed the secret scan" in result.response
    assert str(artifact) in result.response
    assert GITHUB_TOKEN not in result.response
    assert RULE_ID not in result.response


def test_blocked_submission_response_caps_named_paths() -> None:
    paths = [
        f"/repo/artifact-{index}.py" for index in range(BLOCK_MESSAGE_MAX_PATHS + 1)
    ]
    findings = [
        SecretFinding(
            path=path, rule_id=RULE_ID, description="GitHub token", line=1, column=1
        )
        for path in paths
    ]

    message = blocked_submission_response(findings)

    for path in paths[:BLOCK_MESSAGE_MAX_PATHS]:
        assert path in message
    assert paths[BLOCK_MESSAGE_MAX_PATHS] not in message
    assert "1 more" in message
    # No disclosure at any path count: no rule id, no line, no column.
    assert RULE_ID not in message
    assert "GitHub token" not in message


def test_blocked_submission_response_names_one_path_per_artifact() -> None:
    findings = [
        SecretFinding(
            path="/repo/a.py",
            rule_id=RULE_ID,
            description="GitHub token",
            line=3,
            column=5,
        ),
        SecretFinding(
            path="/repo/a.py",
            rule_id="aws-access-key-id",
            description="AWS access key ID",
            line=4,
            column=1,
        ),
        SecretFinding(
            path="/repo/b.py",
            rule_id=RULE_ID,
            description="GitHub token",
            line=1,
            column=1,
        ),
    ]

    message = blocked_submission_response(findings)

    assert "/repo/a.py" in message
    assert "/repo/b.py" in message
    assert RULE_ID not in message
    assert "aws-access-key-id" not in message
    assert "AWS access key ID" not in message


def test_blocked_submission_response_reports_incomplete_scan_redacted() -> None:
    # An incomplete scan blocks with its own redacted message: the agent is
    # told the artifact could not be fully inspected, never the rule id or
    # any scan detail.
    findings = [
        SecretFinding(
            path="/repo/big.bin",
            rule_id="scan-incomplete",
            description=(
                "File could not be fully inspected: content past the "
                "4194304-byte scan cap was not inspected"
            ),
            line=1,
            column=1,
        )
    ]

    message = blocked_submission_response(findings)

    assert "could not be fully inspected" in message
    assert "/repo/big.bin" in message
    assert "scan-incomplete" not in message
    assert "4194304" not in message


def test_scan_of_touched_paths_matches_the_scanner(tmp_path: Path) -> None:
    # The gate scans exactly the accumulated touched paths, with the real
    # scanner: a secret seeded in one touched path is found, a clean one is
    # not.
    secret = tmp_path / "secret.py"
    secret.write_text(f"{SECRET_LINE}\n", encoding="utf-8")
    clean = tmp_path / "clean.py"
    clean.write_text("value = 1\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(_edit_event(str(secret)), tool_call_id="task")
    accumulator.observe(_edit_event(str(clean)), tool_call_id="task")

    findings = scan_paths(accumulator._touched_paths)

    assert {finding.path for finding in findings} == {str(secret)}
    assert accumulator.build_result(turns_used=1).completed is False


# R6: bash-write collection at the submission boundary. The bash tool
# surfaces the write-classified paths its permission analysis extracted
# (redirect targets, write-option values, write-classified command operands)
# on the result; the accumulator collects them alongside file-tool paths,
# and an unestablished write surface records the unknown-footprint sentinel.
# Where the channel matters (the redirect pin, the read-only pin) the tests
# run real shell commands through the real Bash tool; where only the gate
# matters they drive the accumulator with hand-built shell results.


def _bash_event(
    *paths: str, unknown: bool = False, command: str = "bash", skipped: bool = False
) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="bash",
        tool_class=Bash,
        result=CapturedShellResult(
            command=command, write_paths=list(paths), write_paths_unknown=unknown
        ),
        tool_call_id="bash-1",
        skipped=skipped,
    )


def test_bash_write_paths_extracted_from_shell_results() -> None:
    paths, unknown = bash_write_paths_from_result_event(
        _bash_event("/repo/out.txt", "/repo/err.txt", command="echo x > out.txt")
    )
    assert paths == ["/repo/out.txt", "/repo/err.txt"]
    assert unknown is False
    paths, unknown = bash_write_paths_from_result_event(_bash_event(unknown=True))
    assert paths == []
    assert unknown is True


class _FootprintlessShellResult(BaseModel):
    # A shell-kind result model that carries no footprint attributes at all
    # (a malformed or out-of-contract result the typed models never emit,
    # but the boundary must still survive).
    command: str = "bash"


class _NonListFootprintResult(BaseModel):
    # A shell-kind result model whose write_paths attribute is not a list.
    write_paths: str = "not-a-list"
    write_paths_unknown: bool = False


def _malformed_shell_event(result: BaseModel) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="bash", tool_class=Bash, result=result, tool_call_id="bash-1"
    )


def test_bash_write_paths_fail_closed_on_malformed_shell_results() -> None:
    # A shell result whose footprint attributes are missing or malformed
    # cannot prove its write surface was established, so it reads as the
    # unknown-footprint sentinel — never as an established empty footprint.
    # Well-formed shell results are unaffected.
    assert bash_write_paths_from_result_event(
        _malformed_shell_event(_FootprintlessShellResult())
    ) == ([], True)
    assert bash_write_paths_from_result_event(
        _malformed_shell_event(_NonListFootprintResult())
    ) == ([], True)
    assert bash_write_paths_from_result_event(_bash_event("/repo/out.txt")) == (
        ["/repo/out.txt"],
        False,
    )
    # A non-shell result without the attributes still carries no footprint.
    assert bash_write_paths_from_result_event(
        _malformed_shell_event(_FootprintlessShellResult()).model_copy(
            update={"tool_class": ReadFile, "tool_name": "read_file"}
        )
    ) == ([], False)


def _bash_start_event(result: BaseModel) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="bash_start",
        tool_class=BashStart,
        result=result,
        tool_call_id="bash-start-1",
    )


class _NonListStartResult(BaseModel):
    # A bash_start (TOOL-kind) result model whose write_paths attribute is
    # not a list.
    write_paths: str = "not-a-list"
    write_paths_unknown: bool = False


class _NullStartResult(BaseModel):
    # A bash_start result model declaring write_paths with a None value.
    write_paths: list[str] | None = None
    write_paths_unknown: bool = False


class _FootprintlessStartResult(BaseModel):
    # A TOOL-kind result model carrying no footprint attributes at all (the
    # result shape of every other TOOL-kind tool).
    command: str = "bash_start"


def test_bash_start_results_fail_closed_on_malformed_footprints() -> None:
    # bash_start keeps the plain TOOL effect kind, so the shape check cannot
    # key on the effect kind: it keys on the result carrying the footprint
    # fields at all. A bash_start result carrying them without well-formed
    # values cannot prove its write surface was established, so it reads as
    # the unknown-footprint sentinel — exactly like a shell-kind result —
    # never as an established empty one.
    assert bash_write_paths_from_result_event(
        _bash_start_event(_NonListStartResult())
    ) == ([], True)
    assert bash_write_paths_from_result_event(
        _bash_start_event(_NullStartResult())
    ) == ([], True)
    # A well-formed bash_start result is collected as before.
    assert bash_write_paths_from_result_event(
        _bash_start_event(
            CapturedShellResult(
                command="echo x > out.txt", write_paths=["/repo/out.txt"]
            )
        )
    ) == (["/repo/out.txt"], False)
    # A TOOL-kind result with no footprint attributes at all (another TOOL
    # tool's result shape) still carries no footprint.
    assert bash_write_paths_from_result_event(
        _bash_start_event(_FootprintlessStartResult())
    ) == ([], False)


def test_accumulator_records_unknown_footprint_for_malformed_bash_start() -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(_bash_start_event(_NonListStartResult()), tool_call_id="task")
    accumulator.observe(_bash_start_event(_NullStartResult()), tool_call_id="task")
    assert list(accumulator._touched_paths) == [UNKNOWN_WRITE_FOOTPRINT]


def test_accumulator_records_unknown_footprint_for_malformed_shell_results() -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(
        _malformed_shell_event(_FootprintlessShellResult()), tool_call_id="task"
    )
    accumulator.observe(
        _malformed_shell_event(_NonListFootprintResult()), tool_call_id="task"
    )
    assert list(accumulator._touched_paths) == [UNKNOWN_WRITE_FOOTPRINT]
    result = accumulator.build_result(turns_used=1)
    assert result.completed is False
    assert "could not be fully inspected" in result.response
    assert UNKNOWN_WRITE_FOOTPRINT in result.response


def test_bash_write_paths_ignore_skips_and_foreign_results() -> None:
    # A skipped shell call never ran; file-tool results carry no write-path
    # surface; a result without a tool class or a result model carries none.
    assert bash_write_paths_from_result_event(
        _bash_event("/repo/out.txt", skipped=True)
    ) == ([], False)
    assert bash_write_paths_from_result_event(_read_event("/repo/a.py")) == ([], False)
    assert bash_write_paths_from_result_event(_edit_event("/repo/a.py")) == ([], False)
    assert bash_write_paths_from_result_event(
        ToolResultEvent(tool_name="bash", tool_class=None, tool_call_id="c")
    ) == ([], False)
    assert bash_write_paths_from_result_event(
        ToolResultEvent(
            tool_name="bash", tool_class=Bash, result=None, tool_call_id="c"
        )
    ) == ([], False)


def test_accumulator_collects_bash_write_paths_and_the_unknown_sentinel() -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(_bash_event("/repo/out.txt"), tool_call_id="task")
    accumulator.observe(
        _bash_event("/repo/out.txt", "/repo/second.txt"), tool_call_id="task"
    )
    accumulator.observe(_bash_event(unknown=True), tool_call_id="task")
    assert list(accumulator._touched_paths) == [
        "/repo/out.txt",
        "/repo/second.txt",
        UNKNOWN_WRITE_FOOTPRINT,
    ]


def _failed_bash_event(
    *paths: str, unknown: bool = False, command: str = "bash"
) -> ToolResultEvent:
    # The error event the tool executor yields for a failed or timed-out
    # shell run: no result model, the run's write footprint on the
    # error-event fields.
    return ToolResultEvent(
        tool_name="bash",
        tool_class=Bash,
        error=f"bash failed: Command failed: {command!r}",
        error_write_paths=list(paths),
        error_write_paths_unknown=unknown,
        tool_call_id="bash-1",
    )


def test_bash_write_paths_collected_from_failed_run_error_events() -> None:
    # R7: a failed or timed-out run may still have written files (partial
    # execution); its write footprint rides the error event and is
    # collected exactly as a successful run's would be. An error raised
    # before the command ran carries no footprint.
    paths, unknown = bash_write_paths_from_result_event(
        _failed_bash_event("/repo/out.txt")
    )
    assert paths == ["/repo/out.txt"]
    assert unknown is False
    paths, unknown = bash_write_paths_from_result_event(
        _failed_bash_event(unknown=True)
    )
    assert paths == []
    assert unknown is True
    assert bash_write_paths_from_result_event(
        ToolResultEvent(
            tool_name="bash", tool_class=Bash, error="spawn failed", tool_call_id="c"
        )
    ) == ([], False)
    # A failed non-shell call carries no footprint.
    assert bash_write_paths_from_result_event(
        ToolResultEvent(
            tool_name="read_file",
            tool_class=ReadFile,
            error="read failed",
            tool_call_id="c",
        )
    ) == ([], False)


def test_accumulator_collects_write_paths_from_failed_bash_runs() -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(_failed_bash_event("/repo/out.txt"), tool_call_id="task")
    accumulator.observe(_failed_bash_event(unknown=True), tool_call_id="task")
    assert list(accumulator._touched_paths) == [
        "/repo/out.txt",
        UNKNOWN_WRITE_FOOTPRINT,
    ]


def _cancelled_bash_event(
    *paths: str, unknown: bool = False, command: str = "bash"
) -> ToolResultEvent:
    # The error event the tool executor yields for a cancelled shell run:
    # no result model, the run's write footprint on the error-event fields.
    return ToolResultEvent(
        tool_name="bash",
        tool_class=Bash,
        error=f"bash was interrupted: {command!r}",
        cancelled=True,
        error_write_paths=list(paths),
        error_write_paths_unknown=unknown,
        tool_call_id="bash-1",
    )


def test_bash_write_paths_collected_from_cancelled_run_error_events() -> None:
    # A cancelled run may still have written files; its write footprint rides
    # the cancelled event and is collected exactly as a failed run's would be.
    paths, unknown = bash_write_paths_from_result_event(
        _cancelled_bash_event("/repo/out.txt")
    )
    assert paths == ["/repo/out.txt"]
    assert unknown is False
    paths, unknown = bash_write_paths_from_result_event(
        _cancelled_bash_event(unknown=True)
    )
    assert paths == []
    assert unknown is True


def test_accumulator_collects_write_paths_from_cancelled_bash_runs() -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(_cancelled_bash_event("/repo/out.txt"), tool_call_id="task")
    accumulator.observe(_cancelled_bash_event(unknown=True), tool_call_id="task")
    assert list(accumulator._touched_paths) == [
        "/repo/out.txt",
        UNKNOWN_WRITE_FOOTPRINT,
    ]


def _failed_bash_start_event(
    *paths: str, unknown: bool = False, cancelled: bool = False
) -> ToolResultEvent:
    # The error event the tool executor yields for a failed or cancelled
    # bash_start admission: no result model, the started command's write
    # footprint on the error-event fields. bash_start keeps the plain TOOL
    # effect kind, so the footprint rides the error-event attributes.
    return ToolResultEvent(
        tool_name="bash_start",
        tool_class=BashStart,
        error="bash_start failed: Unable to start background job",
        cancelled=cancelled,
        error_write_paths=list(paths),
        error_write_paths_unknown=unknown,
        tool_call_id="start-1",
    )


def test_bash_start_write_paths_collected_from_failed_start_error_events() -> None:
    # A failed or cancelled start may still have spawned the command (partial
    # execution); its write footprint rides the error event and is collected
    # exactly as a successful start's would be, under the same attribute
    # contract despite the plain TOOL effect kind.
    paths, unknown = bash_write_paths_from_result_event(
        _failed_bash_start_event("/repo/out.txt")
    )
    assert paths == ["/repo/out.txt"]
    assert unknown is False
    paths, unknown = bash_write_paths_from_result_event(
        _failed_bash_start_event(unknown=True)
    )
    assert paths == []
    assert unknown is True
    paths, unknown = bash_write_paths_from_result_event(
        _failed_bash_start_event("/repo/out.txt", cancelled=True)
    )
    assert paths == ["/repo/out.txt"]
    assert unknown is False


def test_accumulator_collects_write_paths_from_failed_bash_starts() -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(_failed_bash_start_event("/repo/out.txt"), tool_call_id="task")
    accumulator.observe(_failed_bash_start_event(unknown=True), tool_call_id="task")
    assert list(accumulator._touched_paths) == [
        "/repo/out.txt",
        UNKNOWN_WRITE_FOOTPRINT,
    ]


@pytest.mark.asyncio
async def test_build_result_blocks_secret_written_by_a_shell_redirect(
    tmp_path: Path,
) -> None:
    # End-to-end through the accumulator: the child runs a real
    # `echo KEY... > out.txt` (the key seeded by concatenation above), the
    # bash result carries the analyzed write path, and the gate blocks the
    # submission exactly like a file-tool write.
    artifact = tmp_path / "out.txt"
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = await collect_result(
        tool.run(BashArgs(command=f'echo "{SECRET_LINE}" > out.txt'))
    )
    assert result.write_paths == [str(artifact)]
    assert result.write_paths_unknown is False
    assert artifact.read_text(encoding="utf-8") == f"{SECRET_LINE}\n"

    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="wrote the config"), tool_call_id="task")
    accumulator.observe(
        ToolResultEvent(
            tool_name="bash", tool_class=Bash, result=result, tool_call_id="bash-1"
        ),
        tool_call_id="task",
    )

    blocked = accumulator.build_result(turns_used=2)
    assert blocked.completed is False
    assert "failed the secret scan" in blocked.response
    assert str(artifact) in blocked.response
    assert "wrote the config" not in blocked.response
    # No disclosure: no secret, no rule id, no matched line.
    assert GITHUB_TOKEN not in blocked.response
    assert RULE_ID not in blocked.response
    assert SECRET_LINE not in blocked.response


def test_build_result_blocks_secret_written_via_tee(tmp_path: Path) -> None:
    # The tee channel: the command's write-classified operands join the
    # same scan as file-tool paths.
    artifact = tmp_path / "tee_out.txt"
    artifact.write_text(f"{SECRET_LINE}\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="wrote the config"), tool_call_id="task")
    accumulator.observe(
        _bash_event(str(artifact), command="echo x | tee tee_out.txt"),
        tool_call_id="task",
    )

    blocked = accumulator.build_result(turns_used=2)
    assert blocked.completed is False
    assert "failed the secret scan" in blocked.response
    assert str(artifact) in blocked.response
    assert GITHUB_TOKEN not in blocked.response
    assert RULE_ID not in blocked.response


@pytest.mark.asyncio
async def test_build_result_passes_a_read_only_shell_command(tmp_path: Path) -> None:
    # A command whose analysis established no writes (a pure read) collects
    # nothing, so the submission passes unaffected.
    source = tmp_path / "clean.py"
    source.write_text("value = 1\n", encoding="utf-8")
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    result = await collect_result(tool.run(BashArgs(command="cat clean.py")))
    assert result.write_paths == []
    assert result.write_paths_unknown is False

    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="read the file"), tool_call_id="task")
    accumulator.observe(
        ToolResultEvent(
            tool_name="bash", tool_class=Bash, result=result, tool_call_id="bash-1"
        ),
        tool_call_id="task",
    )

    passed = accumulator.build_result(turns_used=2)
    assert passed.completed is True
    assert passed.response == "read the file"


def test_build_result_blocks_an_unknown_shell_write_footprint() -> None:
    # Fail-closed: a shell command whose write surface could not be
    # established leaves an unscannable footprint, blocked exactly like a
    # scan-incomplete artifact — and the message carries no scan detail.
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="done"), tool_call_id="task")
    accumulator.observe(_bash_event(unknown=True), tool_call_id="task")

    blocked = accumulator.build_result(turns_used=1)
    assert blocked.completed is False
    assert "could not be fully inspected" in blocked.response
    assert UNKNOWN_WRITE_FOOTPRINT in blocked.response
    assert "scan-incomplete" not in blocked.response


def test_bash_write_paths_join_the_file_tool_scan(tmp_path: Path) -> None:
    # The union: a clean file-tool artifact plus a secret-bearing
    # bash-written artifact — the gate scans both and names only the
    # bash-written one.
    clean = tmp_path / "clean.py"
    clean.write_text("value = 1\n", encoding="utf-8")
    artifact = tmp_path / "out.txt"
    artifact.write_text(f"{SECRET_LINE}\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="done"), tool_call_id="task")
    accumulator.observe(_write_event(str(clean)), tool_call_id="task")
    accumulator.observe(_bash_event(str(artifact)), tool_call_id="task")

    blocked = accumulator.build_result(turns_used=1)
    assert blocked.completed is False
    assert str(artifact) in blocked.response
    assert str(clean) not in blocked.response
