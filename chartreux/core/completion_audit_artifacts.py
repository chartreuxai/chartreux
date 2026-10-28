"""P7: the C1/C3 artifact audit.

The artifact half of the completion audit: it resolves a worker's claimed
artifacts (the :class:`~chartreux.core.completion_contract.CompletionArtifactKind`
requirements the receipt reports satisfied) against the collected evidence
chronology, and classifies each claim on one of three dispositions — worker
credit, unverified (check-context), or unavailable (evidence unavailable).
The vocabulary has no contradicted member: final-state disagreement alone
never establishes worker fabrication, and the concurrent-edit attribution
rules constrain enforcement, not just coverage (round-2 amendment 2).

Commit mandates (C1). A claimed commit hash is matched against the
collected git evidence with a prefix match of at least seven characters in
BOTH directions, and matching happens ONLY in git-command results — the
captured stdout of a hash-bearing git invocation. A hash inside a command
string is a command echo (the bash display echoes the command,
``bash.py:241-247``) and is never evidence. Required-commit proof
additionally needs a CHANGED admission-vs-publication HEAD: a copied hash
observed while the HEAD is unchanged cannot satisfy the mandate, no matter
how well it matches.

Touched-path mandates (C3). Touched paths are extracted from result events
via the submission conventions (``subagent_submission.py``); the extraction
reuses those conventions and the COMPARISON logic is owned by this module.
Worker credit requires FILE-SPECIFIC evidence: the artifact's touched-path
match, or a collection-time content snapshot (P3's capture) keyed to the
path. The mutation-roots footprint is read from the shell-result metadata —
the separate mutation-scope list the accumulator exposes via its
``mutation_roots`` property, explicitly excluded from recursive scanning —
and consumed as CONSERVATIVE MUTATION SCOPE ONLY: it is never proof that any
particular descendant file changed, and a claim justified only by root
containment is classified unverified (check-context). Root-level worktree
footprints from checkout/switch join the scope list and compare like any
other root (round-5 read discipline).

Custom and MCP tool writes are out of scope (documented residual): only
file-tool touched paths and the shell's write-classified footprints are
collected, so a claim about an artifact produced through a custom or MCP
tool is never credited and never condemned — it resolves unverified with
nothing harness-observable either way.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
import posixpath
import re

from chartreux.core.completion_contract import (
    CompletionArtifactKind,
    CompletionRequirement,
)
from chartreux.core.completion_evidence import ContentSnapshot
from chartreux.core.events import ToolResultEvent
from chartreux.core.subagent_submission import (
    UNKNOWN_WRITE_FOOTPRINT,
    bash_write_paths_from_result_event,
    touched_path_from_result_event,
)
from chartreux.core.subagents import UNKNOWN_MUTATION_ROOT_FOOTPRINT

__all__ = [
    "MIN_COMMIT_HASH_CHARS",
    "ArtifactAuditDisposition",
    "ArtifactAuditFinding",
    "ArtifactEvidence",
    "audit_artifacts",
    "collect_touched_paths",
    "git_result_hashes",
    "hashes_match",
]

#: Minimum characters of a commit hash either side may carry for the prefix
#: match to bind: a shorter fragment is ambiguous, never evidence.
MIN_COMMIT_HASH_CHARS = 7

#: A plausible git object hash: lowercase hex, 7-40 characters.
_COMMIT_HASH_RE = re.compile(r"[0-9a-f]{7,40}")

#: Git subcommands whose captured stdout carries commit hashes. Conservative:
#: a hash inside the stdout of any other git command (or of a non-git
#: command) is never extracted.
_HASH_BEARING_GIT_SUBCOMMANDS = frozenset({
    "am",
    "branch",
    "cherry-pick",
    "commit",
    "commit-tree",
    "describe",
    "log",
    "ls-tree",
    "merge",
    "merge-base",
    "revert",
    "rev-list",
    "rev-parse",
    "show",
})


class ArtifactAuditDisposition(StrEnum):
    """The disposition vocabulary of one claimed artifact.

    ``worker_credited`` means FILE-SPECIFIC evidence matched the claim.
    ``unverified`` is the check-context disposition: no file-specific
    evidence matched, so the claim may not credit the worker — and it is
    never a contradiction. ``unavailable`` is an availability fact about the
    harness's evidence, never a worker disposition.
    """

    WORKER_CREDITED = "worker_credited"
    UNVERIFIED = "unverified"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ArtifactAuditFinding:
    """One claimed artifact's resolved disposition and evidence note."""

    requirement_id: str
    artifact: CompletionArtifactKind
    disposition: ArtifactAuditDisposition
    evidence: str


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    """The collected evidence surfaces one artifact audit consumes.

    All fields are the harness's own collection: touched paths and snapshots
    from the run's result events, the mutation-roots footprint from the
    shell-result metadata, the commit hashes observed in git-command
    results, and the admission (P4) and publication (P6a) HEAD snapshots.
    """

    touched_paths: tuple[str, ...] = ()
    snapshots: Mapping[str, ContentSnapshot] = field(default_factory=dict)
    mutation_roots: tuple[str, ...] = ()
    observed_commit_hashes: tuple[str, ...] = ()
    admission_head: str | None = None
    publication_head: str | None = None


def hashes_match(
    claimed: str, observed: str, *, min_chars: int = MIN_COMMIT_HASH_CHARS
) -> bool:
    """Whether two commit hashes denote the same object: a prefix match of
    at least ``min_chars`` characters in BOTH directions.

    Non-hex or over-long fragments never match; a sub-``min_chars`` fragment
    is ambiguous and never matches.
    """
    normalized_claimed = claimed.strip().lower()
    normalized_observed = observed.strip().lower()
    if _COMMIT_HASH_RE.fullmatch(normalized_claimed) is None:
        return False
    if _COMMIT_HASH_RE.fullmatch(normalized_observed) is None:
        return False
    if len(normalized_claimed) < min_chars or len(normalized_observed) < min_chars:
        return False
    return normalized_observed.startswith(normalized_claimed) or (
        normalized_claimed.startswith(normalized_observed)
    )


def _git_subcommand(command: str) -> str | None:
    """The git subcommand one command string invokes, or ``None``.

    Environment-prefix assignments (``NAME=value git ...``) are skipped, as
    are ``--no-pager`` and the argument-taking global options ``-C``, ``-c``
    and ``--git-dir``. Anything else non-git reads as ``None``.
    """
    tokens = command.strip().split()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name, separator, _ = token.partition("=")
        if separator and name.replace("-", "_").isidentifier():
            index += 1
            continue
        break
    if index >= len(tokens) or tokens[index] != "git":
        return None
    index += 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--no-pager":
            index += 1
            continue
        if token == "-C":
            index += 2
            continue
        if token == "-c":
            index += 3
            continue
        if token in {"--git-dir", "--work-tree"}:
            index += 2
            continue
        return token
    return None


def git_result_hashes(event: ToolResultEvent) -> tuple[str, ...]:
    """The commit hashes one git-command result's captured stdout carries.

    Matching evidence lives ONLY in git-command results: the captured stdout
    of a hash-bearing git invocation. A hash inside the command string is a
    command echo — the bash display echoes the command — and is never
    evidence. Skipped calls never ran, and an error event carries no
    captured stdout.
    """
    if event.skipped or event.result is None:
        return ()
    command = getattr(event.result, "command", None)
    stdout = getattr(event.result, "stdout", None)
    if not isinstance(command, str) or not isinstance(stdout, str):
        return ()
    if _git_subcommand(command) not in _HASH_BEARING_GIT_SUBCOMMANDS:
        return ()
    return tuple(dict.fromkeys(_COMMIT_HASH_RE.findall(stdout)))


def collect_touched_paths(
    events: Iterable[ToolResultEvent],
) -> tuple[tuple[str, ...], bool]:
    """The touched paths a run's result events report, via the submission
    conventions (:func:`subagent_submission.touched_path_from_result_event`
    and :func:`subagent_submission.bash_write_paths_from_result_event`) —
    file-tool touched paths plus the write-classified paths shell commands
    produced, collected from completed results AND from failed runs' error
    footprints (partial execution still wrote files).

    Returns ``(paths, unknown)``: ``unknown`` marks a run whose write
    surface could not be established, which the audit classifies as
    evidence unavailable — never contradictory.
    """
    paths: list[str] = []
    unknown = False
    for event in events:
        touched = touched_path_from_result_event(event)
        if touched is not None:
            paths.append(touched)
        write_paths, write_unknown = bash_write_paths_from_result_event(event)
        paths.extend(write_paths)
        unknown = unknown or write_unknown
    return tuple(dict.fromkeys(paths)), unknown


def _clean_path(path: str) -> str:
    stripped = path.strip()
    if not stripped:
        return stripped
    normalized = posixpath.normpath(stripped)
    return stripped if normalized in {".", ""} else normalized


def _is_reported_path(path: str) -> bool:
    return (
        isinstance(path, str)
        and bool(path.strip())
        and path not in {UNKNOWN_WRITE_FOOTPRINT, UNKNOWN_MUTATION_ROOT_FOOTPRINT}
    )


def _paths_equal(claimed: str, observed: str) -> bool:
    return _clean_path(claimed) == _clean_path(observed)


def _contained_in(path: str, root: str) -> bool:
    normalized_path = _clean_path(path)
    normalized_root = _clean_path(root)
    if not normalized_root:
        return False
    if normalized_root == "/":
        return normalized_path.startswith("/") and normalized_path != "/"
    prefix = normalized_root.rstrip("/") + "/"
    return normalized_path.startswith(prefix) and normalized_path != normalized_root


def _audit_commit(
    requirement: CompletionRequirement,
    artifact: CompletionArtifactKind,
    claimed_values: Mapping[str, str],
    evidence: ArtifactEvidence,
) -> ArtifactAuditFinding:
    claimed = claimed_values.get(requirement.requirement_id, "").strip().lower()
    if _COMMIT_HASH_RE.fullmatch(claimed) is None:
        return ArtifactAuditFinding(
            requirement_id=requirement.requirement_id,
            artifact=artifact,
            disposition=ArtifactAuditDisposition.UNVERIFIED,
            evidence="the claim names no commit hash to check (check-context)",
        )
    observed = evidence.observed_commit_hashes
    if (
        not observed
        and evidence.admission_head is None
        and evidence.publication_head is None
    ):
        return ArtifactAuditFinding(
            requirement_id=requirement.requirement_id,
            artifact=artifact,
            disposition=ArtifactAuditDisposition.UNAVAILABLE,
            evidence="no commit-hash evidence was collected (evidence unavailable)",
        )
    head_changed = (
        evidence.admission_head is not None
        and evidence.publication_head is not None
        and evidence.publication_head != evidence.admission_head
    )
    if head_changed:
        head_matched = evidence.publication_head is not None and hashes_match(
            claimed, evidence.publication_head
        )
        if head_matched or any(hashes_match(claimed, value) for value in observed):
            return ArtifactAuditFinding(
                requirement_id=requirement.requirement_id,
                artifact=artifact,
                disposition=ArtifactAuditDisposition.WORKER_CREDITED,
                evidence="changed-HEAD commit proof",
            )
    if any(hashes_match(claimed, value) for value in observed):
        return ArtifactAuditFinding(
            requirement_id=requirement.requirement_id,
            artifact=artifact,
            disposition=ArtifactAuditDisposition.UNVERIFIED,
            evidence=(
                "a copied hash with an unchanged HEAD cannot satisfy the "
                "commit mandate (check-context)"
            ),
        )
    if evidence.admission_head is None or evidence.publication_head is None:
        return ArtifactAuditFinding(
            requirement_id=requirement.requirement_id,
            artifact=artifact,
            disposition=ArtifactAuditDisposition.UNVERIFIED,
            evidence=(
                "the changed admission-vs-publication HEAD is unavailable "
                "(check-context)"
            ),
        )
    return ArtifactAuditFinding(
        requirement_id=requirement.requirement_id,
        artifact=artifact,
        disposition=ArtifactAuditDisposition.UNVERIFIED,
        evidence="no collected commit-hash evidence matches the claim (check-context)",
    )


def _audit_touched_path(
    requirement: CompletionRequirement,
    artifact: CompletionArtifactKind,
    claimed_values: Mapping[str, str],
    evidence: ArtifactEvidence,
) -> ArtifactAuditFinding:
    claimed = claimed_values.get(requirement.requirement_id, "")
    if not isinstance(claimed, str) or not claimed.strip():
        return ArtifactAuditFinding(
            requirement_id=requirement.requirement_id,
            artifact=artifact,
            disposition=ArtifactAuditDisposition.UNVERIFIED,
            evidence="the claim names no artifact path (check-context)",
        )
    touched = tuple(path for path in evidence.touched_paths if _is_reported_path(path))
    roots = tuple(root for root in evidence.mutation_roots if _is_reported_path(root))
    if any(_paths_equal(claimed, path) for path in touched):
        return ArtifactAuditFinding(
            requirement_id=requirement.requirement_id,
            artifact=artifact,
            disposition=ArtifactAuditDisposition.WORKER_CREDITED,
            evidence="touched-path match",
        )
    if any(_paths_equal(claimed, path) for path in evidence.snapshots):
        return ArtifactAuditFinding(
            requirement_id=requirement.requirement_id,
            artifact=artifact,
            disposition=ArtifactAuditDisposition.WORKER_CREDITED,
            evidence="collection-time snapshot",
        )
    if not touched and not evidence.snapshots and not roots:
        if (
            UNKNOWN_WRITE_FOOTPRINT in evidence.touched_paths
            or UNKNOWN_MUTATION_ROOT_FOOTPRINT in evidence.mutation_roots
        ):
            note = "the write surface could not be established (evidence unavailable)"
        else:
            note = "no write-surface evidence was collected (evidence unavailable)"
        return ArtifactAuditFinding(
            requirement_id=requirement.requirement_id,
            artifact=artifact,
            disposition=ArtifactAuditDisposition.UNAVAILABLE,
            evidence=note,
        )
    if any(_contained_in(claimed, root) for root in roots):
        return ArtifactAuditFinding(
            requirement_id=requirement.requirement_id,
            artifact=artifact,
            disposition=ArtifactAuditDisposition.UNVERIFIED,
            evidence=(
                "root containment only — conservative mutation scope, never "
                "proof a specific descendant changed (check-context)"
            ),
        )
    return ArtifactAuditFinding(
        requirement_id=requirement.requirement_id,
        artifact=artifact,
        disposition=ArtifactAuditDisposition.UNVERIFIED,
        evidence="no file-specific evidence for the claimed path (check-context)",
    )


def audit_artifacts(
    requirements: Iterable[CompletionRequirement],
    satisfied_ids: Iterable[str],
    claimed_values: Mapping[str, str] | None,
    evidence: ArtifactEvidence,
) -> tuple[ArtifactAuditFinding, ...]:
    """Resolve the claimed artifact requirements against the evidence.

    Only artifact requirements the receipt reports satisfied are resolved
    here: unclaimed mandates belong to the truth table's mandate axis, and
    check/envelope bindings belong to the check and disclosure audits. Each
    resolved claim carries one typed disposition; none of them contradicts
    a worker.
    """
    satisfied = set(satisfied_ids)
    values: Mapping[str, str] = {} if claimed_values is None else claimed_values
    findings: list[ArtifactAuditFinding] = []
    for requirement in requirements:
        if requirement.requirement_id not in satisfied:
            continue
        artifact = requirement.artifact
        if artifact is CompletionArtifactKind.COMMIT:
            findings.append(_audit_commit(requirement, artifact, values, evidence))
        elif artifact is CompletionArtifactKind.TOUCHED_PATH:
            findings.append(
                _audit_touched_path(requirement, artifact, values, evidence)
            )
    return tuple(findings)
