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

A3: a scan-blocked submission carries bounded, versioned, typed diagnostics
(:class:`ScanBlockDiagnostics`, assembled by :func:`build_scan_block`) alongside
the redacted message: every finding with its category, safe artifact path,
location (never rendered for incomplete findings), rule id, closed reason
code, and — for rule hits — the per-entry manifest identifiers of the policy
entries sharing its (path, rule). The block is category-preserving under
truncation (both category counts are always present), names a fixed remediation
string selected from the reason codes — which carries the self-approval
warning: a worker may never approve its own submission — and never auto-retries:
a ``policy_mismatch`` (the manifest changed after the run's admission froze)
is parent-retryable only, never masks rule hits, and exposes the frozen and
current policy-REVISION digests (of the manifest file, never entry digests or
matched bytes).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.events import ToolResultEvent
from chartreux.core.tools.secret_scan import (
    SCAN_INCOMPLETE_RULE_ID,
    FindingReason,
    ScanPolicy,
    SecretFinding,
    committed_scan_policy,
    empty_scan_policy,
    git_toplevel,
    strip_control_characters,
)
from chartreux.core.tools.utils import resolve_tool_path
from chartreux.utils.tool_presentation import ToolEffectKind

__all__ = [
    "BLOCK_MESSAGE_MAX_PATHS",
    "PRUNED_NAMES",
    "REMEDIATION_POLICY_MISMATCH",
    "REMEDIATION_RULE_HIT",
    "REMEDIATION_SCAN_INCOMPLETE",
    "SCAN_BLOCK_MAX_FINDINGS",
    "SCAN_BLOCK_SCHEMA_VERSION",
    "TRANSIENT_NOTICE_MAX_PATHS",
    "UNKNOWN_WRITE_FOOTPRINT",
    "ScanBlockDiagnostics",
    "ScanBlockFinding",
    "ScanBlockPolicy",
    "ScanPathClass",
    "bash_write_paths_from_result_event",
    "blocked_submission_response",
    "build_scan_block",
    "classify_scan_path",
    "effective_scan_policy",
    "resolve_scan_project_root",
    "touched_path_from_result_event",
    "transient_skip_notice",
    "verify_admission_policy",
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


def resolve_scan_project_root(
    cwd: str | Path | None, authorized_roots: Sequence[str | Path] = ()
) -> str | None:
    """The project root the submission scan derives canonical identities from.

    The repository root when one can be established (so a key matches the
    manifest's repo-relative paths from any checkout spelling), else the
    contract's cwd, else the first authorized root, else ``None``. ``None``
    leaves every identity uncanonicalizable, which fails closed: no hit can
    be suppressed.
    """
    base: str | Path | None = cwd
    if base is None or not str(base):
        base = authorized_roots[0] if authorized_roots else None
    if base is None or not str(base):
        return None
    toplevel = git_toplevel(base)
    return str(toplevel) if toplevel is not None else str(base)


def verify_admission_policy(
    frozen: ScanPolicy | None, project_root: str | Path | None
) -> tuple[ScanPolicy | None, bool, str | None, str | None]:
    """Verify the admission-frozen policy against the CURRENT committed HEAD.

    Returns ``(policy_for_scan, mismatch, frozen_revision_digest,
    current_revision_digest)``. *frozen* is the policy frozen on the contract
    at admission, computed from the COMMITTED HEAD blob (never the working
    tree). At scan time it is verified against the CURRENT committed HEAD: a
    mismatch — the manifest file changed after this run's admission froze,
    the stale-freeze case — fails closed by dropping every approval, so a
    worker that commits a manifest self-approval mid-run gains nothing, and
    *mismatch* reports it for the typed diagnostics. ``None`` (no
    admission-frozen policy on the contract) means no approvals active and no
    mismatch detection. A missing *project_root* cannot verify the freeze, so
    it also fails closed to the empty policy and reports a mismatch (the
    current revision digest is then not computable and reads ``None``).

    The two revision digests are policy-REVISION digests of the manifest file
    as a whole — never entry digests or matched bytes — and they are the only
    digests a scan-block diagnostic may expose.
    """
    if frozen is None:
        return None, False, None, None
    frozen_revision = frozen.revision_digest
    if project_root is None:
        return empty_scan_policy(frozen_revision), True, frozen_revision, None
    current = committed_scan_policy(project_root)
    current_revision = current.revision_digest
    if current_revision != frozen_revision:
        return (
            empty_scan_policy(current_revision),
            True,
            frozen_revision,
            current_revision,
        )
    return frozen, False, frozen_revision, current_revision


def effective_scan_policy(
    frozen: ScanPolicy | None, project_root: str | Path | None
) -> ScanPolicy | None:
    """The policy the submission scan runs with, given the admission-frozen one.

    *frozen* is the policy frozen on the contract at admission, computed from
    the COMMITTED HEAD blob (never the working tree). At scan time it is
    verified against the CURRENT committed HEAD: a mismatch — the manifest
    file changed after this run's admission froze — fails closed by dropping
    every approval, so a worker that commits a manifest self-approval mid-run
    gains nothing. ``None`` (no admission-frozen policy on the contract) means
    no approvals active. A missing *project_root* cannot verify the freeze,
    so it also fails closed to the empty policy.
    """
    return verify_admission_policy(frozen, project_root)[0]


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


# --- A3: bounded, versioned, typed scan-block diagnostics --------------------

#: Schema version of the typed scan-block diagnostics.
SCAN_BLOCK_SCHEMA_VERSION = 1

#: Cap on the findings a scan block carries. Truncation is
#: CATEGORY-PRESERVING: a truncation never drops a whole category, and both
#: category counts are always present in the block.
SCAN_BLOCK_MAX_FINDINGS = 32

#: The fixed remediation strings, selected from the closed reason codes. Each
#: carries the self-approval warning: a worker may never approve its own
#: submission.
REMEDIATION_RULE_HIT = (
    "Remove the secret from the named artifact and resubmit. A worker may "
    "never approve its own submission: do not add or edit approved-match "
    "manifest entries for your own run."
)
REMEDIATION_SCAN_INCOMPLETE = (
    "Make every touched artifact fully inspectable by the scan (within the "
    "per-file cap, readable, within the walk budgets) and resubmit. A worker "
    "may never approve its own submission: do not add or edit approved-match "
    "manifest entries for your own run."
)
REMEDIATION_POLICY_MISMATCH = (
    "The approved-match manifest changed after this run's admission froze "
    "(policy_mismatch). The harness never retries this block automatically: "
    "inspect the manifest change before re-dispatching — a mid-run manifest "
    "commit is a self-approval vector, and a worker may never approve its own "
    "submission. The frozen and current policy-revision digests are in the "
    "block's policy section."
)

#: The two finding categories of a scan block.
ScanBlockCategory = Literal["rule_hit", "scan_incomplete"]


class ScanBlockLocation(BaseModel):
    """A rule hit's match location; incomplete findings render none."""

    model_config = ConfigDict(extra="forbid")

    line: int
    column: int


class ScanBlockFinding(BaseModel):
    """One finding of a scan block, without any matched value.

    ``path`` is the finding's safe identifier (already opaque when the path
    string carried rule-matched content), and ``None`` for the
    unknown-write-footprint sentinel, which names no path. ``location`` is
    rendered only for rule hits — an incomplete finding's carried ``(1, 1)``
    is an internal sentinel. ``reason`` is the closed reason code.
    ``manifest_entries`` holds only per-entry manifest identifiers — never
    entry digests or byte values.
    """

    model_config = ConfigDict(extra="forbid")

    category: ScanBlockCategory
    path: str | None
    rule_id: str
    reason: str
    location: ScanBlockLocation | None
    manifest_entries: list[str] = Field(default_factory=list)


class ScanBlockPolicy(BaseModel):
    """The policy-revision digests of a ``policy_mismatch`` block.

    Both digests are SHA-256 of the manifest FILE as a whole (frozen at
    admission vs current committed HEAD) — never entry digests, never matched
    bytes — and they are the only digests a scan block exposes.
    """

    model_config = ConfigDict(extra="forbid")

    frozen_revision_digest: str | None
    current_revision_digest: str | None


class ScanBlockDiagnostics(BaseModel):
    """The bounded, versioned structure of a scan-blocked submission.

    ``status`` is always ``"blocked"`` and ``version`` pins the schema.
    ``findings`` is deterministically ordered and capped at
    :data:`SCAN_BLOCK_MAX_FINDINGS`; ``finding_counts`` and
    ``truncated_counts`` always carry BOTH categories, so a truncation can
    never hide a whole category. ``remediation`` is the fixed string selected
    from the reason codes. ``policy`` is set exactly when a ``policy_mismatch``
    finding is present. No field ever interpolates matched content or
    arbitrary exception text, and the block never carries the worker's report.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = SCAN_BLOCK_SCHEMA_VERSION
    status: Literal["blocked"] = "blocked"
    findings: list[ScanBlockFinding]
    finding_counts: dict[str, int]
    truncated_counts: dict[str, int]
    remediation: str
    policy: ScanBlockPolicy | None = None


def _finding_category(finding: SecretFinding) -> ScanBlockCategory:
    return (
        "scan_incomplete" if finding.rule_id == SCAN_INCOMPLETE_RULE_ID else "rule_hit"
    )


def _finding_sort_key(finding: SecretFinding) -> tuple[str, str, int, int]:
    return (finding.path, finding.rule_id, finding.line, finding.column)


def build_scan_block(
    findings: Sequence[SecretFinding],
    *,
    frozen_revision_digest: str | None = None,
    current_revision_digest: str | None = None,
) -> ScanBlockDiagnostics:
    """Assemble the bounded, versioned typed diagnostics of a scan block.

    The findings are ordered deterministically and capped at
    :data:`SCAN_BLOCK_MAX_FINDINGS`; the truncation is category-preserving —
    when a cap would drop a whole category, that category's first finding is
    kept in place of another category's last — and both category counts are
    always present (totals in ``finding_counts``, dropped per category in
    ``truncated_counts``). Incomplete findings render no location, and the
    unknown-write-footprint sentinel renders no path. The remediation string
    is selected from the closed reason codes (``policy_mismatch`` first, then
    ``match``, else the incomplete-scan remediation), and the policy section
    is set exactly when a ``policy_mismatch`` finding is present, carrying
    only the policy-REVISION digests. No field ever interpolates matched
    content or arbitrary exception text.
    """
    ordered = sorted(findings, key=_finding_sort_key)
    counts: dict[str, int] = {"rule_hit": 0, "scan_incomplete": 0}
    for finding in ordered:
        counts[_finding_category(finding)] += 1
    kept = list(ordered[:SCAN_BLOCK_MAX_FINDINGS])
    if len(ordered) > SCAN_BLOCK_MAX_FINDINGS:
        for category in ("rule_hit", "scan_incomplete"):
            if counts[category] and not any(
                _finding_category(finding) == category for finding in kept
            ):
                replacement = next(
                    finding
                    for finding in ordered
                    if _finding_category(finding) == category
                )
                for index in range(len(kept) - 1, -1, -1):
                    if _finding_category(kept[index]) != category:
                        kept.pop(index)
                        break
                kept.append(replacement)
        kept.sort(key=_finding_sort_key)
    truncated_counts = {
        category: counts[category]
        - sum(1 for finding in kept if _finding_category(finding) == category)
        for category in counts
    }
    block_findings = [
        ScanBlockFinding(
            category=_finding_category(finding),
            path=(
                None
                if finding.reason == FindingReason.UNKNOWN_WRITE_FOOTPRINT
                else strip_control_characters(finding.path)
            ),
            rule_id=finding.rule_id,
            reason=finding.reason.value,
            location=(
                None
                if _finding_category(finding) == "scan_incomplete"
                else ScanBlockLocation(line=finding.line, column=finding.column)
            ),
            manifest_entries=(
                list(finding.manifest_entries)
                if _finding_category(finding) == "rule_hit"
                else []
            ),
        )
        for finding in kept
    ]
    reasons = {finding.reason for finding in ordered}
    if FindingReason.POLICY_MISMATCH in reasons:
        remediation = REMEDIATION_POLICY_MISMATCH
    elif FindingReason.MATCH in reasons:
        remediation = REMEDIATION_RULE_HIT
    else:
        remediation = REMEDIATION_SCAN_INCOMPLETE
    policy = (
        ScanBlockPolicy(
            frozen_revision_digest=frozen_revision_digest,
            current_revision_digest=current_revision_digest,
        )
        if FindingReason.POLICY_MISMATCH in reasons
        else None
    )
    return ScanBlockDiagnostics(
        version=SCAN_BLOCK_SCHEMA_VERSION,
        status="blocked",
        findings=block_findings,
        finding_counts=counts,
        truncated_counts=truncated_counts,
        remediation=remediation,
        policy=policy,
    )
