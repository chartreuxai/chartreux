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
value by construction).

Scope policy (S4): the touched paths are partitioned by path class before the
scan runs. Repository content — the case the block policy above governs — is
scanned unchanged and fail-closed, with vendored and cache directories pruned
from its walks. Runtime-owned transient content, classified by
:func:`classify_scan_path`, is skipped with a single bounded harness log
notice (:func:`transient_skip_notice`): it is never a finding, never a block,
and never model-visible. The pre-commit hook is unaffected and still scans
every staged blob, so no secret reaches a commit unexamined.

The invocation is harness-side and automatic; there is deliberately no tool
registration or other model-visible surface for this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import tempfile
from typing import Literal

from chartreux.core.events import ToolResultEvent
from chartreux.core.tools.secret_scan import SCAN_INCOMPLETE_RULE_ID, SecretFinding
from chartreux.core.tools.utils import resolve_tool_path
from chartreux.utils.tool_presentation import ToolEffectKind

__all__ = [
    "BLOCK_MESSAGE_MAX_PATHS",
    "PRUNED_NAMES",
    "TRANSIENT_NOTICE_MAX_PATHS",
    "UNKNOWN_WRITE_FOOTPRINT",
    "ScanPathClass",
    "bash_write_paths_from_result_event",
    "blocked_submission_response",
    "classify_scan_path",
    "touched_path_from_result_event",
    "transient_skip_notice",
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

#: Directory names the submission scan's walks never descend into: version
#: control metadata, virtualenvs, and tool caches hold no session-authored
#: repository content, and walking them costs the scan its bounded budget (a
#: single packfile can exceed the per-file cap, a virtualenv is mostly
#: binaries). Pruning applies during walks only — a top-level touched path is
#: never pruned by name, though it is excluded when it resolves under an
#: excluded root. The pre-commit hook scans staged blobs and is unaffected by
#: this list, so nothing reaches a commit unexamined.
PRUNED_NAMES: frozenset[str] = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    "node_modules",
    "dist",
    "site",
})

#: Cap on the transient paths named in the skip notice's single log line.
TRANSIENT_NOTICE_MAX_PATHS = 8

#: The submission scan's two path classes: repository content stays in the
#: scan (``"repo"``), runtime-owned transient content is skipped with a
#: notice (``"transient"``).
ScanPathClass = Literal["repo", "transient"]


def classify_scan_path(
    path: str | Path,
    scratchpad_roots: Iterable[str | Path] = (),
    authorized_roots: Iterable[str | Path] = (),
    *,
    cwd: str | Path | None = None,
) -> ScanPathClass:
    """Classify one touched path for the submission scan's scope (S4).

    ``"transient"`` marks a path the submission scan deliberately leaves
    alone: content under a runtime-owned scratchpad root (the roots the
    parent and the child both write), or under the operating system's
    temporary directory while outside every authorized workspace root.
    Everything else classifies ``"repo"`` and stays in the scan.

    The path is RESOLVED first, with the same resolver the file tools use, so
    a scratchpad symlink pointing into a repository resolves into the
    repository and classifies ``"repo"``: only content that is really
    transient is skipped. The tempdir branch is the fallback for paths no
    runtime owns, and the authorized-root precedence guard applies inside it:
    a session workspace rooted under the tempdir keeps its own subtree in the
    scan (that subtree is repository content for that session), while its
    siblings are transient.

    The tempdir branch is armed only under an ESTABLISHED workspace context.
    With no authorized root to compare against, a path under the tempdir
    cannot be proved outside the session's authorized roots, so it classifies
    ``"repo"`` (see :func:`_tempdir_classification`). Any classification
    failure — a non-path value, an unresolvable root, an OS error — also
    classifies ``"repo"``: the skip is fail-closed, never fail-open.
    """
    try:
        if isinstance(path, Path):
            raw = str(path)
        elif isinstance(path, str):
            raw = path
        else:
            return "repo"
        base = Path(cwd) if isinstance(cwd, (str, Path)) and str(cwd) else Path.cwd()
        resolved = resolve_tool_path(raw, base)
        for scratch_root in _resolved_roots(scratchpad_roots):
            if resolved.is_relative_to(scratch_root):
                return "transient"
        return _tempdir_classification(resolved, authorized_roots)
    except (OSError, RuntimeError, ValueError):
        return "repo"


def transient_skip_notice(paths: Sequence[str], total: int) -> str:
    """The bounded one-line notice naming the paths the scan skipped (S4).

    The gate logs this at INFO: the paths and their class are harness
    diagnostic detail, never a finding, never model-visible, and the list is
    capped so one submission cannot emit an unbounded line.
    """
    named = list(paths[:TRANSIENT_NOTICE_MAX_PATHS])
    listing = "; ".join(f"{path} [transient]" for path in named)
    hidden = len(paths) - len(named)
    if hidden > 0:
        listing += f"; ... {hidden} more"
    return (
        f"{len(paths)} of {total} touched path(s) skipped by the secret scan "
        f"as transient non-repo content: {listing}"
    )


def _resolved_roots(roots: Iterable[str | Path]) -> tuple[Path, ...]:
    """The resolvable roots in *roots*, in the order given.

    A single root value (not a collection of roots) reads as no roots, and a
    root that cannot be resolved is skipped: both readings leave the path
    unclassified, which the caller resolves toward the scan.
    """
    if isinstance(roots, (str, Path)):
        return ()
    resolved: list[Path] = []
    for root in roots:
        if not isinstance(root, (str, Path)) or not str(root):
            continue
        try:
            resolved.append(Path(root).expanduser().resolve())
        except (OSError, RuntimeError):
            continue
    return tuple(resolved)


def _system_temp_directory() -> Path | None:
    """The resolved OS tempdir, or ``None`` when it cannot be established."""
    try:
        return Path(tempfile.gettempdir()).resolve()
    except (OSError, RuntimeError):
        return None


def _tempdir_classification(
    resolved: Path, authorized_roots: Iterable[str | Path]
) -> ScanPathClass:
    """The tempdir branch of :func:`classify_scan_path` for a resolved path.

    A path outside the OS tempdir is repo. Inside it, the branch needs an
    established workspace context to compare against: an authorized root's
    own subtree is repo (the precedence guard), anything else under the
    tempdir is transient, and with no authorized root established the path
    stays in the scan.
    """
    temp_root = _system_temp_directory()
    if temp_root is None or not resolved.is_relative_to(temp_root):
        return "repo"
    authorized = _resolved_roots(authorized_roots)
    if not authorized:
        return "repo"
    if any(resolved.is_relative_to(root) for root in authorized):
        return "repo"
    return "transient"


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
