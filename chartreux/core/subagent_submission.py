"""Submission-boundary secret scan for subagent results (P2-i).

The gate runs when an agent session reports its work complete: every
subagent submission result is assembled through
:meth:`chartreux.core.subagents.SubagentRunAccumulator.build_result`, and
the accumulator collects the session's touched paths — repo paths created
or edited through the file tools, plus the write-classified paths shell
commands produced — as it observes the run's tool results. Before the
result is surfaced as final, the touched paths are scanned with the
gitleaks-style scanner (:mod:`chartreux.core.tools.secret_scan`).

Failure policy: a secret-bearing artifact blocks the submission, and so does
an artifact the scan cannot fully inspect (larger than the per-file scan cap
or unreadable — the scanner's ``scan-incomplete`` sentinel). The result
carries a redacted block message instead of the agent's response — the agent
is told the artifact failed the scan, never what matched (no rule id, no
matched line, no secret value; :class:`SecretFinding` carries no matched
value by construction). The invocation is harness-side and automatic; there
is deliberately no tool registration or other model-visible surface for this
module.
"""

from __future__ import annotations

from collections.abc import Sequence

from chartreux.core.events import ToolResultEvent
from chartreux.core.tools.secret_scan import SCAN_INCOMPLETE_RULE_ID, SecretFinding
from chartreux.utils.tool_presentation import ToolEffectKind

__all__ = [
    "BLOCK_MESSAGE_MAX_PATHS",
    "UNKNOWN_WRITE_FOOTPRINT",
    "bash_write_paths_from_result_event",
    "blocked_submission_response",
    "touched_path_from_result_event",
]

#: Effect kinds whose successful results created or edited a file.
_FILE_MUTATION_KINDS = (ToolEffectKind.FILE_EDIT, ToolEffectKind.FILE_WRITE)

#: Effect kind of the shell tool: its results carry the command's
#: write-classified paths alongside the captured output.
_SHELL_EFFECT_KIND = ToolEffectKind.SHELL

#: Attribute names a file-mutation result uses for the touched path
#: (``EditResult.file``, ``WriteFileResult.file_path``).
_RESULT_PATH_ATTRIBUTES = ("file", "file_path")

#: Cap on artifact paths named in the agent-facing block message.
BLOCK_MESSAGE_MAX_PATHS = 8

#: Touched-path sentinel recorded when a shell command's write surface could
#: not be established. The scan cannot cover an unknown footprint, so the
#: submission gate blocks on it exactly like a scan-incomplete artifact.
UNKNOWN_WRITE_FOOTPRINT = "<shell command with an unestablished write surface>"


def touched_path_from_result_event(event: ToolResultEvent) -> str | None:
    """The path a file-mutation tool result created or edited, else ``None``.

    Only created/edited paths are scan scope: a file the session merely read
    is not something the session produced. Skipped calls never ran, and
    results without a tool class or a result model carry no path.
    """
    if event.skipped or event.result is None or event.tool_class is None:
        return None
    if getattr(event.tool_class, "effect_kind", None) not in _FILE_MUTATION_KINDS:
        return None
    for attribute in _RESULT_PATH_ATTRIBUTES:
        path = getattr(event.result, attribute, None)
        if isinstance(path, str) and path:
            return path
    return None


def bash_write_paths_from_result_event(
    event: ToolResultEvent,
) -> tuple[list[str], bool]:
    """The write-classified paths a shell result reports, and its unknown flag.

    The bash tool surfaces the write paths its permission analysis extracted
    (redirect targets, write-option values, write-classified command
    operands) on the result, so bash-written artifacts join the same
    submission scan as file-tool paths; bash_start surfaces the started
    command's write surface on its result the same way. A failed, timed-out,
    or cancelled run may still have written files (partial execution): the
    bash tool attaches the run's write footprint to the raised error and
    bash_start attaches the started command's to its failed or cancelled
    start the same way, the tool executor copies either onto the error
    event, and it is collected from there. Skipped calls never ran. Returns
    ``(paths, unknown)``: ``unknown`` marks a command whose write surface
    could not be established, which the accumulator records as the
    :data:`UNKNOWN_WRITE_FOOTPRINT` sentinel. A result whose result model
    carries the footprint fields — the shell effect kind, or any result
    model declaring ``write_paths`` (bash_start's keeps the plain TOOL
    effect kind) — without well-formed values is fail-closed: it reads as
    an unknown footprint, never as an established empty one. A result of
    a tool whose model carries no footprint attributes at all keeps
    reading as an established empty one.
    """
    if event.skipped or event.tool_class is None:
        return [], False
    result = event.result
    shell = getattr(event.tool_class, "effect_kind", None) is _SHELL_EFFECT_KIND
    if result is not None:
        missing = object()
        paths = getattr(result, "write_paths", missing)
        unknown = bool(getattr(result, "write_paths_unknown", False))
        carries = shell or paths is not missing
        if not carries:
            # A tool whose result model has no footprint attributes at all
            # (a file tool, or any other non-shell tool) carries no shell
            # footprint.
            return [], False
        # A result carrying the footprint fields without well-formed values
        # (missing, a non-list, or an explicit None) cannot prove its write
        # surface was established, so it reads as unknown rather than empty —
        # for the TOOL-kind bash_start result exactly as for the shell kind.
        if not isinstance(paths, (list, tuple)):
            return [], True
    else:
        # A failed, timed-out, or cancelled run still wrote what it wrote
        # before failing: the executor surfaced the run's write footprint on
        # the error event. The attributes are read regardless of the effect
        # kind — bash_start's failed admission carries its footprint here
        # despite its plain TOOL kind — and an error without an attached
        # footprint carries none.
        paths = getattr(event, "error_write_paths", None)
        unknown = bool(getattr(event, "error_write_paths_unknown", False))
    if not isinstance(paths, (list, tuple)):
        return [], unknown
    return [path for path in paths if isinstance(path, str) and path], unknown


def blocked_submission_response(findings: Sequence[SecretFinding]) -> str:
    """Render the agent-facing message for a blocked submission.

    The message tells the agent that an artifact failed the scan and names
    the artifact path(s) so the agent can act. It never says what matched:
    no rule id, no matched line, no secret value — findings carry no matched
    value by construction, and none of their fields reach this message; only
    the path list does. When the block is an incomplete scan, the message says
    the artifact could not be fully inspected, still without any scan detail.
    """
    paths = sorted({finding.path for finding in findings})
    named = paths[:BLOCK_MESSAGE_MAX_PATHS]
    listing = "; ".join(named)
    hidden = len(paths) - len(named)
    if hidden > 0:
        listing += f"; ... {hidden} more"
    if any(finding.rule_id == SCAN_INCOMPLETE_RULE_ID for finding in findings):
        return (
            "[Submission blocked: an artifact created or edited in this "
            "session could not be fully inspected by the secret scan "
            f"(artifact: {listing}). The scan result is redacted. Inspect "
            "the artifact, remove any secret from it, and resubmit.]"
        )
    return (
        "[Submission blocked: an artifact created or edited in this session "
        f"failed the secret scan (artifact: {listing}). Remove the secret "
        "from the artifact and resubmit. The scan finding is redacted.]"
    )
