"""The one record a manual ``!`` command produces, whatever its outcome.

A manual command is user input: it runs under the user's own authority, and
its result reaches the agent as ordinary user-role text. Success, refusal,
spawn failure, nonzero exit, timeout, and interrupt all render through this
builder, so the scrubbing stage covers error representations exactly as it
covers command output, and persistence and model delivery share one redacted
value.
"""

from __future__ import annotations

from dataclasses import dataclass

from chartreux.core.llm_models import ManualShellContext, ManualShellStatus
from chartreux.core.tools import secret_redaction


@dataclass(frozen=True)
class ManualShellOutcome:
    """One completed manual shell attempt, before record redaction."""

    operation_id: str
    command: str
    cwd: str
    status: ManualShellStatus
    exit_code: int | None = None
    timed_out: bool = False
    interrupted: bool = False
    stdout: str = ""
    stderr: str = ""
    output_text: str = ""
    duration_ms: float = 0.0
    created_at: int = 0
    error: str | None = None


@dataclass(frozen=True)
class ManualShellRecord:
    """The model-facing record text and its timeline metadata."""

    text: str
    context: ManualShellContext


def build_manual_shell_record(
    outcome: ManualShellOutcome, *, max_output_bytes: int
) -> ManualShellRecord:
    """Redact one outcome, then render the record text from the redacted value."""
    context = secret_redaction.redact_model(
        ManualShellContext(
            operation_id=outcome.operation_id,
            command=outcome.command,
            cwd=outcome.cwd,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            output_text=outcome.output_text,
            status=outcome.status,
            exit_code=outcome.exit_code,
            timed_out=outcome.timed_out,
            interrupted=outcome.interrupted,
            duration_ms=outcome.duration_ms,
            created_at=outcome.created_at,
        )
    )
    error = outcome.error
    if error is not None:
        error = secret_redaction.redact(error)
    return ManualShellRecord(
        text=_record_text(context, error=error, max_output_bytes=max_output_bytes),
        context=context,
    )


def _record_text(
    context: ManualShellContext, *, error: str | None, max_output_bytes: int
) -> str:
    stdout = _cap_output(context.stdout, max_output_bytes)
    stderr = _cap_output(context.stderr, max_output_bytes)
    sections = [
        "Manual `!` command result from the user. Use this as context only.",
        f"Command: `{context.command}`",
        f"Working directory: `{context.cwd}`",
        _status_line(context),
    ]
    if error:
        sections.append(f"Error:\n```text\n{error.rstrip()}\n```")
    if stdout:
        sections.append(f"Stdout:\n```text\n{stdout.rstrip()}\n```")
    if stderr:
        sections.append(f"Stderr:\n```text\n{stderr.rstrip()}\n```")
    if not stdout and not stderr:
        sections.append(_unstreamed_output_line(context, max_output_bytes))
    return "\n\n".join(sections)


def _unstreamed_output_line(context: ManualShellContext, max_output_bytes: int) -> str:
    """A failed run keeps whatever output arrived before the error."""
    if not context.output_text:
        return "Output:\n```text\n(no output)\n```"
    output = _cap_output(context.output_text, max_output_bytes)
    return f"Output:\n```text\n{output.rstrip()}\n```"


def _status_line(context: ManualShellContext) -> str:
    if context.status == "timed_out":
        return "Status: timed out"
    if context.status == "interrupted":
        return "Status: interrupted by user"
    if context.status == "refused":
        return "Status: refused"
    if context.status == "failed" and context.exit_code is None:
        return "Status: failed"
    return f"Exit code: {context.exit_code if context.exit_code is not None else 0}"


def _cap_output(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated]"
