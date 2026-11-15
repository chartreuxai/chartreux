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

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel
import pytest

from chartreux.core.agent_loop_hooks import CompletionEvidenceReceipt
from chartreux.core.completion_contract import (
    COMPLETION_CONTRACT_SCHEMA_VERSION,
    COMPLETION_ENVELOPE_MARKER,
    CompletionCheckTemplate,
    CompletionEnvelope,
    CompletionReceipt,
    CompletionRejectionReason,
    CompletionRequirement,
)
from chartreux.core.events import AssistantEvent, ToolResultEvent
from chartreux.core.subagent_submission import (
    BLOCK_MESSAGE_MAX_PATHS,
    REMEDIATION_POLICY_MISMATCH,
    REMEDIATION_RULE_HIT,
    REMEDIATION_SCAN_INCOMPLETE,
    SCAN_BLOCK_MAX_FINDINGS,
    SCAN_BLOCK_SCHEMA_VERSION,
    UNKNOWN_WRITE_FOOTPRINT,
    bash_write_paths_from_result_event,
    blocked_submission_response,
    build_scan_block,
    effective_scan_policy,
    resolve_scan_project_root,
    touched_path_from_result_event,
    verify_admission_policy,
)
from chartreux.core.subagents import EmptySubagentResponseError, SubagentRunAccumulator
from chartreux.core.tools import secret_scan
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
    MANIFEST_FILENAME,
    ApprovalKey,
    FindingReason,
    ScanPolicy,
    SecretFinding,
    committed_scan_policy,
    incomplete_finding,
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


def _audit_envelope() -> CompletionEnvelope:
    return CompletionEnvelope(
        requirements=(
            CompletionRequirement(
                requirement_id="req-1", template=CompletionCheckTemplate.PYTEST
            ),
        ),
        schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
        marker=COMPLETION_ENVELOPE_MARKER,
    )


def _audit_claim(envelope: CompletionEnvelope) -> CompletionReceipt:
    return CompletionReceipt(
        marker=COMPLETION_ENVELOPE_MARKER,
        schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
        envelope=envelope,
        satisfied_requirement_ids=("req-1",),
    )


def _observe_secret_free_write(accumulator: SubagentRunAccumulator, path: Path) -> None:
    path.write_text("payload\n", encoding="utf-8")
    accumulator.observe(AssistantEvent(content="done"), tool_call_id="task")
    accumulator.observe(
        ToolResultEvent(
            tool_name="write_file",
            tool_class=WriteFile,
            result=WriteFileResult(
                file_path=str(path), bytes_written=8, content="payload\n"
            ),
            tool_call_id="write-1",
            occurrence_id="occ-1",
        ),
        tool_call_id="write-1",
    )


def _emit_failed_check_receipt(accumulator: SubagentRunAccumulator) -> None:
    accumulator.completion_evidence_sink()(
        CompletionEvidenceReceipt(
            occurrence_id="occ-1",
            tool_call_id="write-1",
            tool_name="write_file",
            status="failure",
            final_text="1 failed",
            executed_args={"command": "pytest tests/test_x.py"},
        )
    )


def _enforce_audit_kwargs() -> dict[str, Any]:
    envelope = _audit_envelope()
    return {
        "contract": SimpleNamespace(envelope=envelope),
        "claim": _audit_claim(envelope),
        "runner": SimpleNamespace(
            check_argv=lambda template: ["pytest", "tests/test_x.py"]
        ),
        "mode": "enforce",
    }


def test_build_result_stamps_the_composed_audit(tmp_path: Path) -> None:
    # P10: the composed classification rides on the result in stamp mode.
    # The accumulator's collection gives the composition a real surface:
    # the observed write is registered in the chronology with its
    # collection-time snapshot.
    artifact = tmp_path / "out.txt"
    accumulator = SubagentRunAccumulator()
    _observe_secret_free_write(accumulator, artifact)
    result = accumulator.build_result(turns_used=1)
    assert result.completed is True
    assert result.completion_audit is not None
    assert result.completion_audit.claim_fidelity.value == "claim_unverified"
    assert result.completion_audit.mandate_satisfaction == "mandate_disabled"
    # The collection-time capture rode along: the chronology entry carries
    # the artifact's snapshot.
    entry = accumulator._chronology.entry("occ-1")
    assert entry is not None and entry.snapshot is not None


def test_build_result_enforce_mode_cancellation_never_blocks() -> None:
    # P10 enforce mode + P6b: a recorded cancellation is an informational
    # disposition (honesty economics) — it never blocks the submission.
    # The result builds completed=False, the response is the agent's own,
    # and no rejection message is fabricated from it.
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="done"), tool_call_id="task")
    accumulator.record_error("run cancelled (user_cancelled)")
    result = accumulator.build_result(turns_used=1, mode="enforce")
    assert result.completed is False
    # The recorded cancellation joins the response (the accumulator's
    # record_error appends the marker line): the agent's prose plus the
    # recorded error, never erased.
    assert result.response.startswith("done")
    assert "[Subagent error: run cancelled (user_cancelled)]" in result.response
    # The inert composition stamps nothing: no audit surface was collected.
    assert result.completion_audit is None


def test_build_result_enforce_mode_blocks_with_the_bounded_message(
    tmp_path: Path,
) -> None:
    # P10: enforce mode blocks the submission on a worker-side
    # disposition — the paired same-command conflict — with the bounded,
    # actionable rejection message; the stamp rides on the result.
    artifact = tmp_path / "out.txt"
    accumulator = SubagentRunAccumulator()
    _observe_secret_free_write(accumulator, artifact)
    _emit_failed_check_receipt(accumulator)
    result = accumulator.build_result(turns_used=1, **_enforce_audit_kwargs())
    assert result.completed is False
    assert result.response.startswith("[Completion audit rejected:")
    assert "req-1: contradicted" in result.response
    assert result.completion_audit is not None
    assert (
        result.completion_audit.rejection
        is CompletionRejectionReason.CLAIM_CONTRADICTED
    )


def test_build_result_secret_scan_takes_precedence_over_the_audit_block(
    tmp_path: Path,
) -> None:
    # P10: secret-scan precedence — when the scan blocks AND the audit
    # would block, the scan's redacted block message is published (the
    # scan's findings never leak into another surface) and the stamp
    # still rides on the result.
    artifact = tmp_path / "deploy.py"
    artifact.write_text(f"# config\n{SECRET_LINE}\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="done"), tool_call_id="task")
    accumulator.observe(
        ToolResultEvent(
            tool_name="write_file",
            tool_class=WriteFile,
            result=WriteFileResult(
                file_path=str(artifact), bytes_written=8, content="x"
            ),
            tool_call_id="write-1",
            occurrence_id="occ-1",
        ),
        tool_call_id="write-1",
    )
    _emit_failed_check_receipt(accumulator)
    result = accumulator.build_result(turns_used=1, **_enforce_audit_kwargs())
    assert result.completed is False
    assert result.response.startswith("[Submission blocked:")
    assert "Completion audit rejected" not in result.response
    assert SECRET_LINE not in result.response
    assert SECRET_LINE not in result.model_dump_json()
    assert result.completion_audit is not None
    assert (
        result.completion_audit.rejection
        is CompletionRejectionReason.CLAIM_CONTRADICTED
    )


def test_build_result_reads_the_finalization_cache(tmp_path: Path) -> None:
    # P10: repeated builds run no checks — build_result resolves the
    # once-only finalization (P6a) from the run-keyed cache and the
    # composition consumes it unchanged; a repeated build is a cache hit
    # with a stable stamp.
    artifact = tmp_path / "out.txt"
    accumulator = SubagentRunAccumulator()
    _observe_secret_free_write(accumulator, artifact)
    accumulator._finalization_cache["run-1"] = SimpleNamespace(
        checks=(),
        admission_head=None,
        publication_head=None,
        runtime_failure="unavailable",
    )
    first = accumulator.build_result(turns_used=1, run_id="run-1")
    second = accumulator.build_result(turns_used=1, run_id="run-1")
    assert first.completion_audit is not None
    assert second.completion_audit is not None
    assert second.completion_audit == first.completion_audit
    # The cached finalization's availability fact reached the stamp.
    assert first.completion_audit.primary_runtime_failure.value == "runtime_unavailable"


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


# A1: the mutation-roots footprint publication regression. A checkout/switch
# invocation's mutation scope is the resolved work-tree root — a distinct
# footprint class the resolver collection produces alongside write_paths.
# SCAN-SCOPE SEPARATION: it is a mutation-scope footprint only and must NEVER
# enter the submission scan's path list — the scan walks its collected paths
# recursively with a bounded budget, and a work-tree root would scan the whole
# repository and its budget overrun would block publication. The regression
# runs a REAL shell result (the real Bash tool performing a real checkout in a
# real repository that also holds an artifact larger than the per-file scan
# cap, so a scan of the root WOULD block) through the accumulator's observe
# collection, and asserts the completed submission passes.


def _checkout_repo(tmp_path: Path) -> Path:
    """A real git repository holding an artifact the scan cannot finish."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "p@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "P"], check=True)
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "base"], check=True)
    (repo / "big-artifact.bin").write_bytes(b"x" * (DEFAULT_MAX_SCAN_BYTES + 1))
    return repo


@pytest.mark.asyncio
async def test_completed_run_with_an_eligible_checkout_is_not_blocked_by_it(
    tmp_path: Path,
) -> None:
    repo = _checkout_repo(tmp_path)
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo)
    result = await collect_result(tool.run(BashArgs(command="git checkout -b side")))
    assert result.mutation_roots == [str(repo)]
    assert result.mutation_roots_unknown is False

    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="checked out"), tool_call_id="task")
    accumulator.observe(
        ToolResultEvent(
            tool_name="bash", tool_class=Bash, result=result, tool_call_id="bash-1"
        ),
        tool_call_id="task",
    )

    # The mutation-roots footprint is present in the collected metadata...
    assert accumulator.mutation_roots == [str(repo)]
    # ...and the work-tree root never entered the scan's path list: the
    # submission is NOT blocked by an incomplete scan over the repository.
    assert str(repo) not in accumulator._touched_paths
    passed = accumulator.build_result(turns_used=1)
    assert passed.completed is True
    assert passed.response == "checked out"
    assert "could not be fully inspected" not in passed.response


# S1-S2 (v0.7.3): the per-run scan cutoff. A "."-rooted write surface — a
# plain extraction, an in-place patch, a bare `git apply`, or a copy whose
# destination is a directory — makes the whole working tree a touched path.
# The gate still enumerates that tree, but now inspects only what the RUN
# wrote: the cutoff is captured when the accumulator is constructed, and every
# run admission (foreground, background, replacement, resume successor)
# constructs its own, so the filter is per run and never per retained-agent
# lifetime. These tests drive hand-built shell results — the exact shape the
# resolver produces — because only the gate's behavior is under test and no
# shell command needs to run.


def _directory_root_event(directory: Path) -> ToolResultEvent:
    """The resolver's collected "." write root, as a bash result carries it."""
    return _bash_event(str(directory), command="cp -r payload/. .")


def _backdate(path: Path, mtime_ns: int) -> None:
    """Pin a file's mtime, leaving its access time alone."""
    current = path.stat()
    os.utime(path, ns=(current.st_atime_ns, mtime_ns))


def test_directory_write_root_skips_pre_session_content(tmp_path: Path) -> None:
    pre_session = tmp_path / "vendored.py"
    pre_session.write_text(f"{SECRET_LINE}\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    _backdate(pre_session, accumulator._scan_cutoff_ns - 1)
    accumulator.observe(
        AssistantEvent(content="extracted the payload"), tool_call_id="task"
    )
    accumulator.observe(_directory_root_event(tmp_path), tool_call_id="task")

    result = accumulator.build_result(turns_used=2)

    # The pre-session file is not the run's artifact: the submission passes.
    assert result.completed is True
    assert result.response == "extracted the payload"


def test_directory_write_root_blocks_the_runs_own_secret(tmp_path: Path) -> None:
    accumulator = SubagentRunAccumulator()
    in_run = tmp_path / "written.py"
    in_run.write_text(f"{SECRET_LINE}\n", encoding="utf-8")
    accumulator.observe(
        AssistantEvent(content="extracted the payload"), tool_call_id="task"
    )
    accumulator.observe(_directory_root_event(tmp_path), tool_call_id="task")

    result = accumulator.build_result(turns_used=2)

    assert result.completed is False
    assert "failed the secret scan" in result.response
    assert str(in_run) in result.response
    assert GITHUB_TOKEN not in result.response


def test_explicit_file_path_is_unconditional_inside_the_run(tmp_path: Path) -> None:
    # The cutoff qualifies DIRECTORY paths only: a file the session names is
    # its own artifact whatever its mtime.
    artifact = tmp_path / "adopted.py"
    artifact.write_text(f"{SECRET_LINE}\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    _backdate(artifact, accumulator._scan_cutoff_ns - 1)
    accumulator.observe(AssistantEvent(content="adopted the file"), tool_call_id="task")
    accumulator.observe(_write_event(str(artifact)), tool_call_id="task")

    result = accumulator.build_result(turns_used=1)

    assert result.completed is False
    assert "failed the secret scan" in result.response
    assert str(artifact) in result.response


def test_scan_cutoff_is_per_run_not_per_retained_identity(tmp_path: Path) -> None:
    # The same retained agent's next run does not re-inspect the previous
    # run's artifacts: each accumulator carries its own construction instant,
    # so a lifetime-scoped cutoff (which would block both runs) is ruled out.
    artifact = tmp_path / "deploy.py"
    artifact.write_text(f"{SECRET_LINE}\n", encoding="utf-8")
    first = SubagentRunAccumulator()
    _backdate(artifact, first._scan_cutoff_ns + 1)
    second = SubagentRunAccumulator()
    assert second._scan_cutoff_ns > artifact.stat().st_mtime_ns
    for accumulator in (first, second):
        accumulator.observe(AssistantEvent(content="wrote it"), tool_call_id="task")
        accumulator.observe(_directory_root_event(tmp_path), tool_call_id="task")

    assert first.build_result(turns_used=1).completed is False
    passed = second.build_result(turns_used=1)
    assert passed.completed is True
    assert passed.response == "wrote it"


# --- A2: the admission-frozen committed-HEAD policy at the submission gate ----
#
# The manifest is frozen on the contract at admission from the COMMITTED HEAD
# blob and verified against the current HEAD at scan time; a worker's
# working-tree or staged manifest edit — or a mid-run manifest COMMIT — never
# authorizes its own submission. Seeded secrets stay concatenation-built so
# this file itself scans clean.

OTHER_TOKEN = "ghp_" + "z9Y8x7W6v5U4t3S2r1Q0p9O8n7M6l5K4j3H2"
SWAPPED_TOKEN = "ghp_" + "b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9"


def _policy_repo(tmp_path: Path) -> Path:
    """A real git repository with one committed file, as the run's cwd."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _git(repo, "config", "user.email", "p@example.com")
    _git(repo, "config", "user.name", "P")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True)


def _digest_of(line: str, rule_id: str) -> str:
    keys: list[ApprovalKey] = []
    secret_scan.scan_text(line + "\n", label="fixture.txt", keys=keys)
    return next(key.digest for key in keys if key.rule_id == rule_id)


def _manifest_entries(path: str, line: str) -> list[dict[str, str]]:
    return [
        {"path": path, "rule_id": RULE_ID, "digest": _digest_of(line, RULE_ID)},
        {
            "path": path,
            "rule_id": "generic-keyword-assignment",
            "digest": _digest_of(line, "generic-keyword-assignment"),
        },
    ]


def _write_manifest(repo: Path, entries: list[dict[str, str]]) -> None:
    (repo / secret_scan.MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": secret_scan.SCAN_POLICY_SCHEMA_VERSION,
                "approved_matches": entries,
                "residual_exclusions": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _commit_manifest(repo: Path, entries: list[dict[str, str]]) -> None:
    _write_manifest(repo, entries)
    _git(repo, "add", "--", secret_scan.MANIFEST_FILENAME)
    _git(repo, "commit", "-q", "-m", "approve synthetic matches")


def _gate_contract(repo: Path, policy: ScanPolicy | None) -> SimpleNamespace:
    # The shape freeze_run_admission freezes: cwd/roots plus the
    # admission-frozen committed-HEAD scan policy.
    return SimpleNamespace(
        shared_roots=(), roots=(str(repo),), cwd=str(repo), scan_policy=policy
    )


def _gate_result(repo: Path, policy: ScanPolicy | None, name: str, content: str):
    artifact = repo / name
    artifact.write_text(content, encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="wrote the config"), tool_call_id="task")
    accumulator.observe(_write_event(str(artifact)), tool_call_id="task")
    return accumulator.build_result(turns_used=1, contract=_gate_contract(repo, policy))


def test_gate_passes_an_admission_frozen_approved_match(tmp_path: Path) -> None:
    repo = _policy_repo(tmp_path)
    _commit_manifest(repo, _manifest_entries("fixture.txt", SECRET_LINE))
    # what freeze_run_admission froze at admission: the COMMITTED HEAD policy
    frozen = committed_scan_policy(repo)
    result = _gate_result(repo, frozen, "fixture.txt", SECRET_LINE + "\n")
    assert result.completed is True
    assert result.response == "wrote the config"


def test_gate_blocks_a_different_credential_at_the_approved_path(
    tmp_path: Path,
) -> None:
    repo = _policy_repo(tmp_path)
    _commit_manifest(repo, _manifest_entries("fixture.txt", SECRET_LINE))
    frozen = committed_scan_policy(repo)
    other_line = f"GITHUB_TOKEN = {OTHER_TOKEN}"
    result = _gate_result(repo, frozen, "fixture.txt", other_line + "\n")
    assert result.completed is False
    assert "failed the secret scan" in result.response
    assert OTHER_TOKEN not in result.response
    assert RULE_ID not in result.response


def test_gate_working_tree_manifest_edit_does_not_authorize(tmp_path: Path) -> None:
    repo = _policy_repo(tmp_path)
    _commit_manifest(repo, [])  # committed manifest approves nothing
    frozen = committed_scan_policy(repo)
    # the worker edits the WORKING-TREE manifest mid-run to add the approval
    _write_manifest(repo, _manifest_entries("fixture.txt", SECRET_LINE))
    result = _gate_result(repo, frozen, "fixture.txt", SECRET_LINE + "\n")
    assert result.completed is False
    assert "failed the secret scan" in result.response
    assert GITHUB_TOKEN not in result.response


def test_gate_staged_manifest_does_not_authorize(tmp_path: Path) -> None:
    repo = _policy_repo(tmp_path)
    _commit_manifest(repo, [])
    frozen = committed_scan_policy(repo)
    # the worker STAGES the approval (index) without committing it
    _write_manifest(repo, _manifest_entries("fixture.txt", SECRET_LINE))
    _git(repo, "add", "--", secret_scan.MANIFEST_FILENAME)
    result = _gate_result(repo, frozen, "fixture.txt", SECRET_LINE + "\n")
    assert result.completed is False
    assert "failed the secret scan" in result.response


def test_gate_mid_run_manifest_commit_fails_closed(tmp_path: Path) -> None:
    repo = _policy_repo(tmp_path)
    _commit_manifest(repo, _manifest_entries("fixture.txt", SECRET_LINE))
    frozen = committed_scan_policy(repo)
    assert any(key.path == "fixture.txt" for key in frozen.approved), (
        "the frozen policy carries the approval"
    )
    # mid-run, the worker COMMITS a manifest change: the current HEAD no
    # longer matches the admission-frozen revision, so every approval drops
    _commit_manifest(repo, [])
    assert committed_scan_policy(repo).revision_digest != frozen.revision_digest
    result = _gate_result(repo, frozen, "fixture.txt", SECRET_LINE + "\n")
    assert result.completed is False
    assert GITHUB_TOKEN not in result.response
    # A3: the stale freeze surfaces as a policy_mismatch finding that
    # COEXISTS with the independent rule hits — it never masks them — and
    # the redacted message still names the artifact.
    block = result.scan_block
    assert block is not None
    assert block.version == SCAN_BLOCK_SCHEMA_VERSION
    assert block.status == "blocked"
    by_reason = {finding.reason: finding for finding in block.findings}
    assert by_reason["policy_mismatch"].category == "scan_incomplete"
    assert by_reason["policy_mismatch"].path == MANIFEST_FILENAME
    assert by_reason["policy_mismatch"].location is None
    assert by_reason["match"].category == "rule_hit"
    assert by_reason["match"].location is not None
    assert block.finding_counts == {"rule_hit": 2, "scan_incomplete": 1}
    # The policy section exposes only the policy-REVISION digests of the
    # manifest file — never entry digests or matched bytes.
    assert block.policy is not None
    assert block.policy.frozen_revision_digest == frozen.revision_digest
    assert (
        block.policy.current_revision_digest
        == committed_scan_policy(repo).revision_digest
    )
    assert block.remediation == REMEDIATION_POLICY_MISMATCH
    assert "never retries" in block.remediation
    assert "self-approval" in block.remediation
    assert "may never approve its own submission" in block.remediation
    dumped = result.model_dump_json()
    assert GITHUB_TOKEN not in dumped
    assert SECRET_LINE not in dumped
    for key in frozen.approved:
        assert key.digest not in dumped
    # No report passthrough: the worker's report never reaches the result.
    assert "wrote the config" not in dumped
    # No automatic retry: rebuilding the same run is blocked again, with the
    # same typed block — the harness never clears the mismatch itself.
    again = _gate_result(repo, frozen, "fixture.txt", SECRET_LINE + "\n")
    assert again.completed is False
    assert again.scan_block is not None
    assert again.scan_block.finding_counts == block.finding_counts


def test_gate_value_swap_at_the_submission_boundary(tmp_path: Path) -> None:
    repo = _policy_repo(tmp_path)
    new_line = f"GITHUB_TOKEN = {SWAPPED_TOKEN}"
    # the new value's approval is already committed (approval-first)
    _commit_manifest(repo, _manifest_entries("fixture.txt", new_line))
    frozen = committed_scan_policy(repo)
    # the fixture was swapped to the approved new value: passes
    result = _gate_result(repo, frozen, "fixture.txt", new_line + "\n")
    assert result.completed is True
    # the swapped-out old value is not approved anymore: blocks
    result = _gate_result(repo, frozen, "fixture.txt", SECRET_LINE + "\n")
    assert result.completed is False
    assert "failed the secret scan" in result.response


def test_gate_symlink_escaping_the_repo_is_uncanonicalizable_and_blocks(
    tmp_path: Path,
) -> None:
    repo = _policy_repo(tmp_path)
    _commit_manifest(repo, _manifest_entries("f.py", SECRET_LINE))
    frozen = committed_scan_policy(repo)
    outside = tmp_path / "outside.py"
    outside.write_text(SECRET_LINE + "\n", encoding="utf-8")
    link = repo / "link.py"
    link.symlink_to(outside)
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="linked it"), tool_call_id="task")
    accumulator.observe(_write_event(str(link)), tool_call_id="task")
    # the whole tmp dir is authorized, so the escape target stays IN the
    # scan's scope (repo class) — but it resolves outside the git project
    # root, so it has no canonical identity
    contract = SimpleNamespace(
        shared_roots=(),
        roots=(str(tmp_path), str(repo)),
        cwd=str(repo),
        scan_policy=frozen,
    )
    result = accumulator.build_result(turns_used=1, contract=contract)
    # no canonical identity, so the approving policy cannot suppress the hit
    assert result.completed is False
    assert "failed the secret scan" in result.response


def test_gate_scratchpad_link_into_the_repo_keeps_the_approved_identity(
    tmp_path: Path,
) -> None:
    repo = _policy_repo(tmp_path)
    _commit_manifest(repo, _manifest_entries("f.py", SECRET_LINE))
    frozen = committed_scan_policy(repo)
    artifact = repo / "f.py"
    artifact.write_text(SECRET_LINE + "\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    link = scratch / "link.py"
    link.symlink_to(artifact)
    accumulator = SubagentRunAccumulator()
    accumulator.observe(
        AssistantEvent(content="wrote via the link"), tool_call_id="task"
    )
    accumulator.observe(_write_event(str(link)), tool_call_id="task")
    result = accumulator.build_result(
        turns_used=1, contract=_gate_contract(repo, frozen)
    )
    # the link spells the file through the scratchpad, but the gate derives
    # the SAME canonical identity ("f.py"), so the approval applies
    assert result.completed is True
    assert result.response == "wrote via the link"


def test_gate_without_an_admission_frozen_policy_blocks(tmp_path: Path) -> None:
    # A contract carrying no frozen policy (the pre-A2 shape) gets no
    # approvals at all: fail closed.
    repo = _policy_repo(tmp_path)
    _commit_manifest(repo, _manifest_entries("fixture.txt", SECRET_LINE))
    result = _gate_result(repo, None, "fixture.txt", SECRET_LINE + "\n")
    assert result.completed is False
    assert "failed the secret scan" in result.response


def test_resolve_scan_project_root_prefers_the_git_toplevel(tmp_path: Path) -> None:
    repo = _policy_repo(tmp_path)
    assert resolve_scan_project_root(str(repo), ()) == str(repo)
    subdir = repo / "sub"
    subdir.mkdir()
    assert resolve_scan_project_root(str(subdir), ()) == str(repo)
    plain = tmp_path / "plain"
    plain.mkdir()
    assert resolve_scan_project_root(str(plain), ()) == str(plain)
    assert resolve_scan_project_root(None, ()) is None
    assert resolve_scan_project_root(None, (str(plain),)) == str(plain)


def test_effective_scan_policy_fails_closed_on_mismatch_and_absence(
    tmp_path: Path,
) -> None:
    repo = _policy_repo(tmp_path)
    _commit_manifest(repo, _manifest_entries("fixture.txt", SECRET_LINE))
    frozen = committed_scan_policy(repo)
    # matching HEAD: the frozen policy is used as-is
    assert effective_scan_policy(frozen, str(repo)) is frozen
    # no project root: cannot verify the freeze, fail closed
    dropped = effective_scan_policy(frozen, None)
    assert dropped is not None and dropped.approved == frozenset()
    # no frozen policy: no approvals active
    assert effective_scan_policy(None, str(repo)) is None
    # mid-run change: mismatch drops every approval
    _commit_manifest(repo, [])
    dropped = effective_scan_policy(frozen, str(repo))
    assert dropped is not None
    assert dropped.approved == frozenset()
    assert dropped.revision_digest == committed_scan_policy(repo).revision_digest


# --- A3: bounded, versioned, typed scan-block diagnostics --------------------


def test_verify_admission_policy_reports_mismatch_and_revision_digests(
    tmp_path: Path,
) -> None:
    repo = _policy_repo(tmp_path)
    _commit_manifest(repo, _manifest_entries("fixture.txt", SECRET_LINE))
    frozen = committed_scan_policy(repo)
    # No frozen policy: no approvals, no mismatch detection.
    assert verify_admission_policy(None, str(repo)) == (None, False, None, None)
    # Matching HEAD: the frozen policy is used as-is, no mismatch.
    policy, mismatch, frozen_digest, current_digest = verify_admission_policy(
        frozen, str(repo)
    )
    assert policy is frozen
    assert mismatch is False
    assert frozen_digest == current_digest == frozen.revision_digest
    # Mid-run change: mismatch, fail closed, both revision digests reported.
    _commit_manifest(repo, [])
    policy, mismatch, frozen_digest, current_digest = verify_admission_policy(
        frozen, str(repo)
    )
    assert mismatch is True
    assert policy is not None and policy.approved == frozenset()
    assert frozen_digest == frozen.revision_digest
    assert current_digest == committed_scan_policy(repo).revision_digest
    # No project root: cannot verify the freeze — mismatch, fail closed, and
    # the current revision digest is not computable.
    policy, mismatch, frozen_digest, current_digest = verify_admission_policy(
        frozen, None
    )
    assert mismatch is True
    assert policy is not None and policy.approved == frozenset()
    assert frozen_digest == frozen.revision_digest
    assert current_digest is None


def test_build_result_attaches_the_typed_scan_block(tmp_path: Path) -> None:
    artifact = tmp_path / "deploy.py"
    artifact.write_text(f"# config\n{SECRET_LINE}\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="wrote the config"), tool_call_id="task")
    accumulator.observe(_write_event(str(artifact)), tool_call_id="task")

    result = accumulator.build_result(turns_used=2)

    assert result.completed is False
    # The redacted message is unchanged: rule hits only, so the rule-hit
    # branch, no rule id, no location, no secret.
    assert "failed the secret scan" in result.response
    assert RULE_ID not in result.response
    assert GITHUB_TOKEN not in result.response
    # The typed block carries the actionable detail: version, status, both
    # category counts, the rule hit with its safe path, location, reason
    # code, and the fixed remediation string.
    block = result.scan_block
    assert block is not None
    assert block.version == SCAN_BLOCK_SCHEMA_VERSION
    assert block.status == "blocked"
    assert block.finding_counts == {"rule_hit": 2, "scan_incomplete": 0}
    assert block.truncated_counts == {"rule_hit": 0, "scan_incomplete": 0}
    assert block.policy is None
    assert block.remediation == REMEDIATION_RULE_HIT
    assert "may never approve its own submission" in block.remediation
    hit = next(finding for finding in block.findings if finding.rule_id == RULE_ID)
    assert hit.category == "rule_hit"
    assert hit.reason == "match"
    assert hit.path == str(artifact)
    assert hit.location is not None and hit.location.line == 2
    assert hit.manifest_entries == []
    # The rule id lives in the typed block, never in the redacted response;
    # and no matched value, entry digest, or worker report is anywhere in
    # the serialized result.
    dumped = result.model_dump_json()
    assert RULE_ID in dumped
    assert GITHUB_TOKEN not in dumped
    assert SECRET_LINE not in dumped
    assert "wrote the config" not in dumped


def test_unblocked_result_carries_no_scan_block(tmp_path: Path) -> None:
    artifact = tmp_path / "clean.py"
    artifact.write_text("value = 1\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="done"), tool_call_id="task")
    accumulator.observe(_write_event(str(artifact)), tool_call_id="task")
    result = accumulator.build_result(turns_used=1)
    assert result.completed is True
    assert result.scan_block is None
    # Omitted from dumps entirely when None (the pre-A3 dump shape).
    assert "scan_block" not in result.model_dump(mode="json")


def test_build_result_scan_block_for_an_incomplete_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "big-artifact.txt"
    artifact.write_text("x" * DEFAULT_MAX_SCAN_BYTES + "\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(
        AssistantEvent(content="wrote the artifact"), tool_call_id="task"
    )
    accumulator.observe(_write_event(str(artifact)), tool_call_id="task")

    result = accumulator.build_result(turns_used=1)

    assert result.completed is False
    block = result.scan_block
    assert block is not None
    assert block.finding_counts == {"rule_hit": 0, "scan_incomplete": 1}
    finding = block.findings[0]
    assert finding.category == "scan_incomplete"
    assert finding.reason == "file_size_limit"
    assert finding.path == str(artifact)
    # An incomplete finding renders NO location (its (1,1) is internal).
    assert finding.location is None
    assert block.remediation == REMEDIATION_SCAN_INCOMPLETE
    assert "may never approve its own submission" in block.remediation


def test_build_result_unknown_footprint_scan_block() -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="done"), tool_call_id="task")
    accumulator.observe(_bash_event(unknown=True), tool_call_id="task")

    result = accumulator.build_result(turns_used=1)

    assert result.completed is False
    # The redacted message still names the sentinel path.
    assert UNKNOWN_WRITE_FOOTPRINT in result.response
    block = result.scan_block
    assert block is not None
    finding = block.findings[0]
    assert finding.category == "scan_incomplete"
    assert finding.reason == "unknown_write_footprint"
    # Nullability covers the sentinel PATH, not the location fields.
    assert finding.path is None
    assert finding.location is None
    assert block.remediation == REMEDIATION_SCAN_INCOMPLETE


def test_build_scan_block_truncation_is_category_preserving() -> None:
    # More rule hits than the cap plus incomplete findings: the truncation
    # never drops a whole category, and both category counts are present.
    findings = [
        SecretFinding(
            path=f"/repo/a-{index:03d}.py",
            rule_id=RULE_ID,
            description="GitHub token",
            line=1,
            column=1,
        )
        for index in range(SCAN_BLOCK_MAX_FINDINGS + 5)
    ]
    findings += [
        incomplete_finding(
            f"/repo/big-{index}.bin", "why", reason_code=FindingReason.FILE_SIZE_LIMIT
        )
        for index in range(2)
    ]

    block = build_scan_block(findings)

    assert len(block.findings) == SCAN_BLOCK_MAX_FINDINGS
    assert block.finding_counts == {
        "rule_hit": SCAN_BLOCK_MAX_FINDINGS + 5,
        "scan_incomplete": 2,
    }
    # Both categories survive the cap: one incomplete finding was swapped in
    # for a rule hit, so 6 rule hits and 1 incomplete finding were dropped.
    assert {finding.category for finding in block.findings} == {
        "rule_hit",
        "scan_incomplete",
    }
    assert block.truncated_counts == {"rule_hit": 6, "scan_incomplete": 1}
    assert block.remediation == REMEDIATION_RULE_HIT


def test_build_scan_block_orders_findings_deterministically() -> None:
    findings = [
        SecretFinding(
            path="/repo/b.py", rule_id=RULE_ID, description="d", line=9, column=2
        ),
        SecretFinding(
            path="/repo/a.py",
            rule_id="aws-access-key-id",
            description="d",
            line=1,
            column=1,
        ),
        SecretFinding(
            path="/repo/a.py", rule_id=RULE_ID, description="d", line=3, column=4
        ),
    ]
    block = build_scan_block(list(reversed(findings)))
    assert [
        (f.path, f.rule_id, f.location.line if f.location else None)
        for f in block.findings
    ] == [
        ("/repo/a.py", "aws-access-key-id", 1),
        ("/repo/a.py", RULE_ID, 3),
        ("/repo/b.py", RULE_ID, 9),
    ]
    again = build_scan_block(findings)
    assert block == again


def test_build_scan_block_remediation_selection_and_policy_section() -> None:
    rule_hit = SecretFinding(
        path="/repo/a.py", rule_id=RULE_ID, description="d", line=1, column=1
    )
    incomplete = incomplete_finding(
        "/repo/big.bin", "why", reason_code=FindingReason.FILE_SIZE_LIMIT
    )
    mismatch = incomplete_finding(
        MANIFEST_FILENAME, "why", reason_code=FindingReason.POLICY_MISMATCH
    )
    # Rule hits only: the rule-hit remediation, no policy section.
    block = build_scan_block([rule_hit])
    assert block.remediation == REMEDIATION_RULE_HIT
    assert block.policy is None
    # Incomplete only: the incomplete remediation.
    block = build_scan_block([incomplete])
    assert block.remediation == REMEDIATION_SCAN_INCOMPLETE
    assert block.policy is None
    # A policy_mismatch selects its own remediation and exposes ONLY the
    # policy-revision digests in the policy section.
    block = build_scan_block(
        [rule_hit, incomplete, mismatch],
        frozen_revision_digest="f" * 64,
        current_revision_digest="c" * 64,
    )
    assert block.remediation == REMEDIATION_POLICY_MISMATCH
    assert block.policy is not None
    assert block.policy.frozen_revision_digest == "f" * 64
    assert block.policy.current_revision_digest == "c" * 64
    assert {finding.reason for finding in block.findings} == {
        "match",
        "file_size_limit",
        "policy_mismatch",
    }
    # The mismatch coexists with the rule hit — it does not mask it.
    assert block.finding_counts == {"rule_hit": 1, "scan_incomplete": 2}


def test_gate_surfaces_per_entry_manifest_identifiers_only(tmp_path: Path) -> None:
    # Two approvals share one (path, rule); the artifact carries a THIRD
    # value: the block names WHICH entries share the hit's (path, rule) —
    # identifiers only, never digests.
    repo = _policy_repo(tmp_path)
    line_a = f"GITHUB_TOKEN = {OTHER_TOKEN}"
    line_b = f"GITHUB_TOKEN = {SWAPPED_TOKEN}"
    _commit_manifest(
        repo,
        [
            {
                "path": "fixture.txt",
                "rule_id": RULE_ID,
                "digest": _digest_of(line_a, RULE_ID),
            },
            {
                "path": "fixture.txt",
                "rule_id": RULE_ID,
                "digest": _digest_of(line_b, RULE_ID),
            },
        ],
    )
    frozen = committed_scan_policy(repo)
    assert len(frozen.entry_ids[("fixture.txt", RULE_ID)]) == 2

    result = _gate_result(repo, frozen, "fixture.txt", SECRET_LINE + "\n")

    assert result.completed is False
    block = result.scan_block
    assert block is not None
    hit = next(finding for finding in block.findings if finding.rule_id == RULE_ID)
    assert hit.manifest_entries == ["approved_matches[0]", "approved_matches[1]"]
    dumped = result.model_dump_json()
    assert GITHUB_TOKEN not in dumped
    for key in frozen.approved:
        assert key.digest not in dumped


def test_gate_suspect_path_artifact_block_is_opaque(tmp_path: Path) -> None:
    # A matched credential in the FILENAME: the block's path degrades to the
    # deterministic opaque identifier, in the typed block AND the redacted
    # message — the token appears nowhere in the result.
    artifact = tmp_path / ("leak-" + GITHUB_TOKEN + ".py")
    artifact.write_text(SECRET_LINE + "\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="wrote it"), tool_call_id="task")
    accumulator.observe(_write_event(str(artifact)), tool_call_id="task")

    result = accumulator.build_result(turns_used=1)

    assert result.completed is False
    block = result.scan_block
    assert block is not None
    assert block.findings
    for finding in block.findings:
        assert finding.path is not None
        assert finding.path.startswith("<opaque-path:")
    dumped = result.model_dump_json()
    assert GITHUB_TOKEN not in dumped
    assert GITHUB_TOKEN not in result.response
    assert str(artifact) not in dumped


def test_gate_control_character_path_is_stripped_in_the_block(tmp_path: Path) -> None:
    # A path carrying non-printable control characters is always stripped,
    # in the typed block and the redacted message alike.
    artifact = tmp_path / "we\x1bird.py"
    artifact.write_text(SECRET_LINE + "\n", encoding="utf-8")
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="wrote it"), tool_call_id="task")
    accumulator.observe(_write_event(str(artifact)), tool_call_id="task")

    result = accumulator.build_result(turns_used=1)

    assert result.completed is False
    block = result.scan_block
    assert block is not None
    for finding in block.findings:
        assert finding.path == str(tmp_path / "weird.py")
        assert finding.path is not None
        assert all(character.isprintable() for character in finding.path)
    assert "\x1b" not in result.response
