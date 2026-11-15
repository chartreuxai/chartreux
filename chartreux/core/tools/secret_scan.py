"""Gitleaks-style secret detection over a session's touched paths.

The scanner is pure static analysis: rule-table pattern matches plus Shannon
entropy gating for generic credential-like assignments. It performs no network
access and imports no live-provider machinery.

Findings describe *where* a rule matched — path, rule id, and location — and
never carry the matched value, so surfacing findings to the model cannot
disclose the secret itself. Content the scan cannot fully inspect — a file
truncated at the per-file cap or unreadable, a directory the walk cannot
enumerate, or a walk past its entry budget — yields a ``scan-incomplete``
sentinel finding, so an incomplete scan is never mistaken for a clean one.
Binary content is never skipped wholesale: NUL-containing content is split
on NUL boundaries and every text run is scanned, so a secret appended behind
a NUL byte is still found. The harness invokes :func:`scan_paths` at the
submission boundary with a per-run cutoff: a touched directory tree is
enumerated in full but only the files modified at or after the run began are
inspected, while an explicitly named file is always inspected. A caller may
additionally declare path classes it owns — transient roots to exclude and
directory names the walk must not descend into — which narrows what is
inspected without weakening what is: repository content stays in scope and
stays fail-closed. There is deliberately no tool registration or other
model-visible surface for this module.

When a caller supplies a *keys* collector, the scan additionally computes an
internal value-scoped approval key per rule hit — canonical path, rule id,
and SHA-256 over the rule's value-span bytes — at the match point, before any
matched bytes are discarded. Keys are internal bookkeeping: they never appear
in findings, results, or logs, and a hit whose rule is not approvable or whose
path carries no canonical identity yields no key, so it can never be
suppressed.

When a caller supplies a *policy* (a :class:`ScanPolicy` parsed from the
repo-committed approved-match manifest, :data:`MANIFEST_FILENAME`), an
approved rule hit is suppressed INSIDE scanning — including hits in files
discovered by directory walks — before the matched bytes are discarded. A
``scan-incomplete`` sentinel and a non-approvable rule's hit never suppress.
Both scan gates (the submission boundary and the pre-commit hook) read the
manifest from the COMMITTED HEAD blob, never the working tree or the git
index, so neither a worker nor a single commit can authorize its own secret.

Every finding carries a closed :class:`FindingReason` code (never a free-form
string), so consumers classify blocks without parsing descriptions. And every
finding's path is a SAFE identifier: non-printable control characters are
always stripped, and a path whose own string contains rule-matched content
(a matched credential can occur in a FILENAME) degrades to a deterministic
opaque per-finding identifier, so no finding — and no consumer printing one,
the pre-commit hook included — can ever disclose matched bytes through its
path. Matched bytes and manifest-entry digests never appear in findings,
results, or logs; a rule hit may carry only the per-entry MANIFEST
IDENTIFIERS (never digests) of the policy entries sharing its (path, rule).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass, field, replace
import enum
import hashlib
import json
from math import log2
import os
from pathlib import Path
import re
from stat import S_ISDIR, S_ISLNK, S_ISREG
import subprocess
from typing import Any


#: The closed reason-code enum of every finding. ``match`` is an ordinary
#: rule hit; the rest classify the ``scan-incomplete`` sentinel constructions
#: and the submission gate's own sentinels, so a block is diagnosable without
#: parsing free-form description strings.
class FindingReason(enum.StrEnum):
    MATCH = "match"
    FILE_SIZE_LIMIT = "file_size_limit"
    WALK_BUDGET = "walk_budget"
    ENUMERATION_LIMIT = "enumeration_limit"
    UNREADABLE = "unreadable"
    UNKNOWN_WRITE_FOOTPRINT = "unknown_write_footprint"
    POLICY_MISMATCH = "policy_mismatch"


def strip_control_characters(path: str) -> str:
    """*path* without its non-printable characters.

    A path carrying control characters (a NUL, an escape, a newline smuggled
    into a filename) is never echoed raw: every non-printable character is
    stripped, so no consumer of a finding's path can be fed log-injection or
    terminal-escape content.
    """
    return "".join(character for character in path if character.isprintable())


def opaque_path_identifier(path: str, rule_id: str, line: int, column: int) -> str:
    """The deterministic opaque per-finding identifier for a suspect *path*.

    Used when the path string itself contains rule-matched content: the raw
    path can never be echoed, and the identifier is a pure function of the
    finding's identity (path, rule, location), so the same finding degrades
    to the same identifier on every assembly — without disclosing any byte of
    the path or the match.
    """
    digest = hashlib.sha256(
        f"{path}\x00{rule_id}\x00{line}\x00{column}".encode("utf-8", errors="replace")
    ).hexdigest()
    return f"<opaque-path:{digest[:16]}>"


#: Findings never carry the matched secret value; each carries only the path,
#: the rule that matched, a redacted description, and the match location.
#: The path is always safe to echo (see :func:`strip_control_characters` and
#: :func:`opaque_path_identifier`), ``reason`` is a closed
#: :class:`FindingReason` code, and ``manifest_entries`` holds only per-entry
#: manifest identifiers — never entry digests or matched bytes.
@dataclass(frozen=True)
class SecretFinding:
    """One rule hit, without the matched value."""

    path: str
    rule_id: str
    description: str
    line: int
    column: int
    reason: FindingReason = FindingReason.MATCH
    #: Whether the finding's original path string contained rule-matched
    #: content (the path itself was degraded to an opaque identifier).
    path_suspect: bool = False
    #: The per-entry manifest identifiers of the policy entries sharing this
    #: hit's (path, rule) — identifiers only, never digests.
    manifest_entries: tuple[str, ...] = ()


@dataclass(frozen=True)
class SecretRule:
    """One detection rule.

    ``pattern`` finds candidates on a line. When ``entropy_group`` names a
    capture group, the match only counts when that group's Shannon entropy
    reaches ``entropy_threshold`` bits per character; otherwise every pattern
    match counts. ``approvable`` marks whether the match span covers the
    credential's COMPLETE sensitive material, so that a value-span digest of
    the match can be approved: a partial-coverage rule (the private-key
    header matches only the constant BEGIN line) is not approvable — its hits
    block until remediation, never approval.
    """

    rule_id: str
    description: str
    pattern: re.Pattern[str]
    entropy_group: int | None = None
    entropy_threshold: float = 0.0
    approvable: bool = True


#: One internal value-scoped approval key: canonical project-relative path,
#: rule id, and the SHA-256 digest of the rule's value-span bytes. Keys are
#: computed at the match point and never appear in findings, results, or logs
#: — neither the matched bytes nor the entry digest is ever surfaced. A key
#: whose ``path`` is ``None`` (the identity was uncanonicalizable) can never
#: suppress a finding.
@dataclass(frozen=True)
class ApprovalKey:
    """Internal approval key for one rule hit; never part of a finding."""

    path: str | None
    rule_id: str
    digest: str


#: Minimum Shannon entropy (bits per character) for the generic keyword rule.
#: Human-chosen values ("supersecretvalue123") stay below it; random tokens
#: exceed it.
GENERIC_ENTROPY_THRESHOLD = 4.0

#: Per-file read cap for :func:`scan_paths`; larger files are scanned to the cap.
DEFAULT_MAX_SCAN_BYTES = 4 * 1024 * 1024

#: Per-directory walk budget for :func:`scan_paths`, in enumerated entries
#: (files and directories alike): a crafted tree must not turn the scan into
#: an unbounded traversal.
DEFAULT_MAX_WALK_ENTRIES = 4096

#: Absolute cap on the raw entries a walk enumerates, whatever the walk budget
#: counts. A walk that counts only entries qualifying for inspection still has
#: to bound its scandir work: a tree of structural entries (directories,
#: out-of-window files) would otherwise walk forever. Reaching the cap reports
#: the same ``scan-incomplete`` sentinel as an exhausted budget.
DEFAULT_MAX_WALK_ENUMERATIONS = 65536

#: Rule id of the sentinel finding reported when content could not be fully
#: inspected (truncated at the scan cap, unreadable, a directory the walk
#: cannot enumerate, or a walk past its entry budget). Consumers treat it
#: exactly like a rule hit: an incomplete scan is never a clean scan.
SCAN_INCOMPLETE_RULE_ID = "scan-incomplete"

#: The line terminators :meth:`str.splitlines` recognizes; used to keep a
#: finding's line number file-accurate across NUL-split text runs.
_LINE_TERMINATOR = re.compile(r"\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]")

_RULE_TABLE: tuple[SecretRule, ...] = (
    SecretRule(
        "aws-access-key-id",
        "AWS access key ID",
        re.compile(
            r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}"
        ),
    ),
    SecretRule(
        "aws-secret-access-key",
        "AWS secret access key",
        re.compile(r"(?i)aws.{0,30}?[\"'][0-9a-zA-Z/+]{40}[\"']"),
    ),
    SecretRule(
        "github-token", "GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,255}")
    ),
    SecretRule("gitlab-token", "GitLab token", re.compile(r"glpat-[0-9a-zA-Z_-]{20,}")),
    SecretRule(
        "slack-token", "Slack token", re.compile(r"xox[baprs]-[0-9a-zA-Z-]{10,}")
    ),
    SecretRule(
        "slack-webhook",
        "Slack webhook URL",
        re.compile(
            r"https://hooks\.slack\.com/services/T[A-Za-z0-9_]{8,}"
            r"/B[A-Za-z0-9_]{8,}/[A-Za-z0-9]+"
        ),
    ),
    SecretRule(
        "openai-api-key",
        "OpenAI API key",
        re.compile(r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}"),
    ),
    SecretRule(
        "google-api-key", "Google API key", re.compile(r"AIza[0-9A-Za-z_-]{35}")
    ),
    SecretRule(
        "stripe-secret-key",
        "Stripe secret key",
        re.compile(r"[sr]k_live_[0-9a-zA-Z]{24,}"),
    ),
    SecretRule("npm-token", "npm access token", re.compile(r"npm_[0-9a-zA-Z]{36}")),
    SecretRule(
        "pypi-token",
        "PyPI upload token",
        re.compile(r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{20,}"),
    ),
    SecretRule(
        "private-key-header",
        "Embedded private key",
        re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"),
        # The match is the constant BEGIN-marker line only — no complete
        # sensitive material — so this rule is NOT approvable: its hits
        # block until remediation, never approval.
        approvable=False,
    ),
    # The captured value must terminate as a literal, not head a larger
    # expression: the trailing lookahead rejects a value run followed by "("
    # (e.g. a contextvars Token from `token = var.set(budget)`). The lookahead
    # forbids every character of the value class plus "(", so a backtracked
    # shorter capture inside the run cannot satisfy it either. The keyword
    # family matches identifiers that CONTAIN a keyword: the left boundary is
    # the identifier start (line start, whitespace, or a non-word character),
    # not a boundary between a prefix and the keyword, so provider-prefixed
    # names like MISTRAL_API_KEY match exactly like a bare api_key. The
    # `(?<!tokens)` lookbehind rejects plural names: `max_tokens`,
    # `run_max_session_tokens` and same-class `*_tokens` configuration are
    # token COUNTS, not credentials. The greedy identifier tail cannot
    # backtrack past it — a shorter tail leaves a word character where the
    # assignment separator is required — so `max_tokens=` cannot re-match
    # with the keyword ending mid-identifier. Singular names
    # (`access_token = <high-entropy literal>`) are unaffected. The separator
    # is one ":" or "=" and NEITHER padding side may absorb a colon, so the
    # separator never sits beside another colon: pytest-style
    # `<file>.py::<test>` node ids and `<name>::<value>` pairs do not read as
    # assignments (the colon run is no separator, and the engine's backtrack
    # to the second colon would have to consume the first as padding, which
    # the class forbids), while `password = V` and `api_key: V` still hit.
    SecretRule(
        "generic-keyword-assignment",
        "Credential-like assignment with a high-entropy value",
        re.compile(
            r"(?i)(?<![A-Za-z0-9_])[A-Za-z0-9_]*"
            r"(?:api[_-]?key|apikey|secret(?:[_-]?key)?|token|password|passwd|pwd"
            r"|credential|auth[_-]?(?:key|token)|access[_-]?(?:key|token)"
            r"|private[_-]?key|client[_-]?secret)"
            r"[A-Za-z0-9_]*(?<!tokens)"
            r"[^\w:]{0,3}[:=][^\w:]{0,3}"
            r"([A-Za-z0-9_./+=-]{16,})(?![A-Za-z0-9_./+=\-(])"
        ),
        entropy_group=1,
        entropy_threshold=GENERIC_ENTROPY_THRESHOLD,
    ),
)


def shannon_entropy(text: str) -> float:
    """Shannon entropy of *text* in bits per character (0.0 when empty)."""
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for character in text:
        counts[character] = counts.get(character, 0) + 1
    total = len(text)
    return -sum((count / total) * log2(count / total) for count in counts.values())


def value_span(rule: SecretRule, match: re.Match[str]) -> str | None:
    """The complete sensitive material of *match* under *rule*.

    The capture group the rule names when it defines ``entropy_group`` (the
    generic keyword rule's group 1 is the value alone), otherwise the full
    match — which for the value-shape rules is exactly the material the rule
    captures (the OpenAI rule's full match is the token; the AWS secret
    rule's full match spans keyword + separator + quotes + value, a
    documented reformatting-brittle basis). ``None`` for a non-approvable
    rule: a partial-coverage match covers no complete credential, so it can
    never form an approvable value-span digest.
    """
    if not rule.approvable:
        return None
    if rule.entropy_group is not None:
        return match.group(rule.entropy_group)
    return match.group(0)


def approval_digest(rule: SecretRule, match: re.Match[str]) -> str | None:
    """SHA-256 over the UTF-8 bytes of *rule*'s value-span text in *match*.

    The span is taken from the scanner's UTF-8-replacement-decoded text (see
    :func:`_scan_runs`), so the digest is over the decoded span's UTF-8 bytes
    — a replacement character digests as its ``U+FFFD`` encoding, never as
    the raw invalid byte. ``None`` for a non-approvable rule.
    """
    span = value_span(rule, match)
    if span is None:
        return None
    return hashlib.sha256(span.encode("utf-8")).hexdigest()


def approval_key(
    rule: SecretRule, match: re.Match[str], *, path: str | None
) -> ApprovalKey | None:
    """The internal approval key for one rule hit, or ``None``.

    *path* is the match's canonical project-relative identity (``None`` when
    uncanonicalizable — such a key can never suppress a finding). ``None`` is
    also returned for a non-approvable rule, whose hits block until
    remediation, never approval.
    """
    digest = approval_digest(rule, match)
    if digest is None:
        return None
    return ApprovalKey(path=path, rule_id=rule.rule_id, digest=digest)


def canonical_project_identity(
    path: str | Path, *, project_root: str | Path
) -> str | None:
    """The canonical project-relative identity of *path* under *project_root*.

    Both sides are fully resolved, so every spelling of the same file —
    absolute, relative, or through a symlink in either direction — yields the
    same POSIX-separated relative identity, and both scan gates derive the
    same key for the same file. ``None`` when the identity is
    uncanonicalizable (the path resolves outside the root, or either side
    cannot be resolved): the failure direction is safe, because a key whose
    ``path`` is ``None`` can never suppress a finding.
    """
    try:
        resolved = Path(path).resolve()
        root = Path(project_root).resolve()
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(root):
        return None
    return resolved.relative_to(root).as_posix()


#: Filename of the repo-committed approved-match manifest, at the project
#: root (name PINNED). Both scan gates read it from the COMMITTED HEAD blob
#: (``git show HEAD:<manifest>``) — never the working tree, never the git
#: index — so a worker's mid-run edit and a single commit staging both a
#: secret and its approval cannot self-authorize.
MANIFEST_FILENAME = ".chartreux-secret-scan.json"

#: Schema version of the manifest document.
SCAN_POLICY_SCHEMA_VERSION = 1

#: Timeout for one committed-manifest git probe: the gates run at submission
#: and commit time and must not hang on a wedged git.
_MANIFEST_GIT_TIMEOUT_SECONDS = 10.0

#: A manifest entry digest: SHA-256 hex.
_HEX64 = re.compile(r"[0-9a-f]{64}")


#: The committed scan policy: one admission-frozen or hook-read snapshot of
#: the approved-match manifest. ``revision_digest`` is the SHA-256 of the
#: manifest blob bytes as a whole (``None`` when the manifest is absent) —
#: it is the policy-REVISION digest, never a value fingerprint, and it is the
#: only digest a gate compares across time (frozen at admission vs the
#: current HEAD) to detect a mid-run policy change. ``approved`` holds the
#: approved-match entries (canonical path + rule id + value-span digest);
#: ``residual`` holds the partial-coverage residual exclusions (path + rule,
#: no digest — initially EMPTY by design). ``entry_ids`` maps each
#: (path, rule id) to the per-entry MANIFEST IDENTIFIERS of its approved
#: entries (their position in the manifest's ``approved_matches`` list) —
#: identifiers only, never the entry digests, which never appear in findings,
#: results, or logs.
@dataclass(frozen=True)
class ScanPolicy:
    """One snapshot of the repo-committed approved-match manifest."""

    revision_digest: str | None
    approved: frozenset[ApprovalKey]
    residual: frozenset[tuple[str, str]]
    entry_ids: Mapping[tuple[str, str], tuple[str, ...]] = field(default_factory=dict)


def empty_scan_policy(revision_digest: str | None = None) -> ScanPolicy:
    """The fail-closed policy: no approvals active at all."""
    return ScanPolicy(
        revision_digest=revision_digest, approved=frozenset(), residual=frozenset()
    )


def policy_entry_ids(
    policy: ScanPolicy, path: str | None, rule_id: str
) -> tuple[str, ...]:
    """The per-entry manifest identifiers of *policy*'s entries for (*path*, *rule_id*).

    Identifiers only — never entry digests — so a surfaced rule hit can tell
    the parent WHICH approved entries share its (path, rule) (stale or
    missing), without disclosing any value fingerprint. An
    uncanonicalizable path (``None``) has no identifiers.
    """
    if path is None:
        return ()
    return policy.entry_ids.get((path, rule_id), ())


def is_approved(policy: ScanPolicy, key: ApprovalKey) -> bool:
    """Whether *key* matches an approved (path, rule, digest) entry.

    A key whose ``path`` is ``None`` (the identity was uncanonicalizable) is
    never approved, so it always blocks. Residual (path, rule) exclusions
    are digest-less and checked separately by :func:`is_residually_excluded`.
    """
    if key.path is None:
        return False
    return key in policy.approved


def is_residually_excluded(policy: ScanPolicy, path: str | None, rule_id: str) -> bool:
    """Whether the partial-coverage residual section excludes (*path*, *rule_id*).

    Residual entries carry no digest (a partial-coverage rule captures no
    complete credential), so they key on path + rule alone; an
    uncanonicalizable path is never excluded.
    """
    if path is None:
        return False
    return (path, rule_id) in policy.residual


def _valid_manifest_path(path: object) -> bool:
    """Whether *path* is a canonical project-relative manifest key."""
    if not isinstance(path, str) or not path:
        return False
    if path.startswith("/") or "\\" in path:
        return False
    return all(segment not in {"", ".", ".."} for segment in path.split("/"))


def _load_manifest_document(raw: bytes) -> dict[str, Any] | None:
    """The decoded manifest document, or ``None`` when it is malformed."""
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "approved_matches",
        "residual_exclusions",
    }:
        return None
    version = document["schema_version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != SCAN_POLICY_SCHEMA_VERSION
    ):
        return None
    if not isinstance(document["approved_matches"], list) or not isinstance(
        document["residual_exclusions"], list
    ):
        return None
    return document


def _parse_approved_matches(
    entries: list[Any],
) -> tuple[set[ApprovalKey], dict[tuple[str, str], tuple[str, ...]]] | None:
    """The approved-match entries and their per-entry identifiers.

    Returns ``(approved, entry_ids)``, or ``None`` when any entry is
    malformed. Each entry's identifier is its position in the manifest's
    ``approved_matches`` list (``approved_matches[<index>]``) — an identifier
    only, never the entry's digest.
    """
    rules = {rule.rule_id: rule for rule in _RULE_TABLE}
    approved: set[ApprovalKey] = set()
    entry_ids: dict[tuple[str, str], list[str]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"path", "rule_id", "digest"}:
            return None
        path, rule_id, digest = entry["path"], entry["rule_id"], entry["digest"]
        rule = rules.get(rule_id) if isinstance(rule_id, str) else None
        if (
            not _valid_manifest_path(path)
            or rule is None
            or not rule.approvable
            or not isinstance(digest, str)
            or _HEX64.fullmatch(digest) is None
        ):
            return None
        approved.add(ApprovalKey(path=path, rule_id=rule_id, digest=digest))
        entry_ids.setdefault((path, rule_id), []).append(f"approved_matches[{index}]")
    return approved, {key: tuple(ids) for key, ids in entry_ids.items()}


def _parse_residual_exclusions(entries: list[Any]) -> set[tuple[str, str]] | None:
    """The residual exclusions, or ``None`` when any entry is malformed."""
    rules = {rule.rule_id: rule for rule in _RULE_TABLE}
    residual: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "rule_id",
            "rationale",
        }:
            return None
        path, rule_id, rationale = entry["path"], entry["rule_id"], entry["rationale"]
        rule = rules.get(rule_id) if isinstance(rule_id, str) else None
        if (
            not _valid_manifest_path(path)
            or rule is None
            # The residual section is reserved for partial-coverage rules:
            # an approvable rule there would be a whole-value bypass, not a
            # partial-coverage scoping.
            or rule.approvable
            or not isinstance(rationale, str)
            or not rationale
        ):
            return None
        residual.add((path, rule_id))
    return residual


def parse_scan_policy(raw: bytes) -> ScanPolicy | None:
    """Parse the manifest blob *raw* into a :class:`ScanPolicy`.

    Returns ``None`` when the document is malformed — a missing key, an
    unknown key, a wrong schema version, an unknown or non-approvable rule in
    ``approved_matches`` (a partial-coverage rule can never be approved),
    an approvable rule in ``residual_exclusions`` (that section is reserved
    for partial-coverage fixtures), a non-canonical path, or a non-SHA-256
    digest. Malformed fails closed: the caller turns ``None`` into the empty
    policy, so no approvals are active. The revision digest is computed over
    the raw blob bytes whatever the parse outcome, so a mid-run change of a
    malformed manifest is still detectable.
    """
    revision_digest = hashlib.sha256(raw).hexdigest()
    document = _load_manifest_document(raw)
    if document is None:
        return None
    parsed_approved = _parse_approved_matches(document["approved_matches"])
    residual = _parse_residual_exclusions(document["residual_exclusions"])
    if parsed_approved is None or residual is None:
        return None
    approved, entry_ids = parsed_approved
    return ScanPolicy(
        revision_digest=revision_digest,
        approved=frozenset(approved),
        residual=frozenset(residual),
        entry_ids=entry_ids,
    )


def policy_from_blob(raw: bytes | None) -> ScanPolicy:
    """The :class:`ScanPolicy` of one manifest blob.

    ``None`` — absent manifest, unborn HEAD, not a repository, a failed
    probe — yields the empty policy (no approvals active, fail closed).
    A malformed blob also yields the empty policy, but keeps its revision
    digest so a later change of the manifest file stays detectable.
    """
    if raw is None:
        return empty_scan_policy()
    parsed = parse_scan_policy(raw)
    if parsed is None:
        return empty_scan_policy(hashlib.sha256(raw).hexdigest())
    return parsed


def git_toplevel(
    path: str | Path, *, timeout: float = _MANIFEST_GIT_TIMEOUT_SECONDS
) -> Path | None:
    """The root of the git repository containing *path*, or ``None``.

    Any failure (no git, not a repository, a timeout, a nonexistent path)
    reads as ``None`` — the caller falls back to *path* itself and every
    downstream use fails closed.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(path),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    root = completed.stdout.decode("utf-8", errors="replace").strip()
    return Path(root) if root else None


def committed_manifest_blob(
    project_root: str | Path, *, timeout: float = _MANIFEST_GIT_TIMEOUT_SECONDS
) -> bytes | None:
    """The manifest blob as committed at HEAD of the repository at *project_root*.

    ``git show HEAD:<manifest>`` reads the committed HEAD blob only — never
    the working tree, never the index — so a staged or edited manifest does
    not change what this returns. ``None`` when the manifest is absent from
    HEAD, HEAD is unborn, *project_root* is not a repository, or the probe
    fails: every failure direction reads as absent, which fails closed.
    """
    try:
        completed = subprocess.run(
            ["git", "show", f"HEAD:{MANIFEST_FILENAME}"],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def committed_scan_policy(project_root: str | Path) -> ScanPolicy:
    """The scan policy of the manifest committed at HEAD of *project_root*."""
    return policy_from_blob(committed_manifest_blob(project_root))


def scan_text(
    text: str,
    *,
    label: str = "<text>",
    keys: list[ApprovalKey] | None = None,
    key_path: str | None = None,
    policy: ScanPolicy | None = None,
) -> list[SecretFinding]:
    """Run every rule over *text*, line by line.

    Findings are deduplicated per rule and line; repeated hits of one rule on
    the same line collapse into a single finding at the first match's position.

    *keys*, when supplied, collects one internal :class:`ApprovalKey` per
    finding — rule hits only, never a ``scan-incomplete`` sentinel — carrying
    the canonical identity *key_path* (``None`` when uncanonicalizable). The
    key has no line component: suppression by key is line-agnostic, and the
    digest never leaves this collector.

    *policy*, when supplied, suppresses an approved rule hit INSIDE scanning,
    before the matched bytes are discarded: a hit whose computed key matches
    an approved (path, rule, digest) entry or a residual (path, rule)
    exclusion yields no finding (its key still reaches *keys*). A
    ``scan-incomplete`` sentinel and a non-approvable rule's hit never
    suppress.

    Every surfaced rule hit carries a safe path identifier: when the path
    string itself contains the rule's matched content (a matched credential
    can occur in a filename), the path degrades to a deterministic opaque
    per-finding identifier; otherwise non-printable control characters are
    stripped. A rule hit also carries the per-entry manifest identifiers
    (never digests) of the policy entries sharing its (path, rule), and the
    closed reason code :attr:`FindingReason.MATCH`.
    """
    findings: list[SecretFinding] = []
    seen: set[tuple[str, int, str]] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        for rule in _RULE_TABLE:
            for match in rule.pattern.finditer(line):
                if rule.entropy_group is not None:
                    candidate = match.group(rule.entropy_group)
                    if candidate is None:
                        continue
                    if shannon_entropy(candidate) < rule.entropy_threshold:
                        continue
                key = (rule.rule_id, line_number, match.group(0))
                if key in seen:
                    continue
                seen.add(key)
                approval = (
                    approval_key(rule, match, path=key_path)
                    if keys is not None or policy is not None
                    else None
                )
                suppressed = policy is not None and (
                    (approval is not None and is_approved(policy, approval))
                    or is_residually_excluded(policy, key_path, rule.rule_id)
                )
                if suppressed:
                    if keys is not None and approval is not None:
                        keys.append(approval)
                    continue
                column = match.start() + 1
                needle = value_span(rule, match)
                if needle is None:
                    needle = match.group(0)
                suspect = needle in label
                if suspect:
                    safe_path = opaque_path_identifier(
                        label, rule.rule_id, line_number, column
                    )
                else:
                    safe_path = strip_control_characters(
                        label
                    ) or opaque_path_identifier(
                        label, rule.rule_id, line_number, column
                    )
                findings.append(
                    SecretFinding(
                        path=safe_path,
                        rule_id=rule.rule_id,
                        description=rule.description,
                        line=line_number,
                        column=column,
                        reason=FindingReason.MATCH,
                        path_suspect=suspect,
                        manifest_entries=(
                            policy_entry_ids(policy, key_path, rule.rule_id)
                            if policy is not None
                            else ()
                        ),
                    )
                )
                if keys is not None and approval is not None:
                    keys.append(approval)
    return findings


def scan_paths(  # noqa: PLR0913 - the scan's additive caller-declared surface
    paths: Iterable[str | Path],
    *,
    max_bytes: int = DEFAULT_MAX_SCAN_BYTES,
    max_walk_entries: int = DEFAULT_MAX_WALK_ENTRIES,
    modified_since_ns: int | None = None,
    exclude_roots: Iterable[str | Path] = (),
    prune_names: Container[str] = frozenset(),
    skipped: list[str] | None = None,
    count_qualifying_only: bool = False,
    keys: list[ApprovalKey] | None = None,
    project_root: str | Path | None = None,
    policy: ScanPolicy | None = None,
) -> list[SecretFinding]:
    """Scan each touched path (file or directory) and return all findings.

    Directories are walked recursively under a bounded entry budget; missing
    paths are skipped (touched paths may include artifacts that were deleted
    before submission). Files are scanned to *max_bytes* per file, and
    NUL-containing content is scanned per text run — binary content is
    inspected, never skipped wholesale. A file or directory the scan cannot
    fully inspect — larger than *max_bytes*, unreadable, a directory the walk
    cannot enumerate, or a walk past its entry budget — adds a
    ``scan-incomplete`` sentinel finding naming the path and the reason, so
    an incomplete scan is never mistaken for a clean one. Findings carry no
    matched value, so the returned list is safe to surface to the model.

    *modified_since_ns* is the run's own ``time.time_ns()`` start. A DIRECTORY
    path still enumerates every descendant, but a file is inspected only when
    its mtime is at or after that point, so a tree the session merely wrote
    into does not put pre-existing content in front of the scan. An
    explicitly named file path is inspected whatever its mtime — naming it is
    the session declaring it its own artifact. A caller that omits the cutoff
    keeps the full-walk semantics. The cutoff is a write-window filter, never
    a provenance check: content backdated to before the run (including
    content extracted with preserved timestamps) is not inspected, while a
    file whose metadata cannot be read is reported ``scan-incomplete``
    instead of being assumed old, and the entry budget still counts every
    enumerated entry.

    *exclude_roots* declares path classes the caller owns and the scan must
    leave alone — the submission boundary passes the runtime's scratchpad
    roots. A touched path that RESOLVES under an excluded root is skipped
    before any metadata is read: no walk, no read, and no ``scan-incomplete``
    sentinel, because content the caller has declared transient is not
    uninspected content. This is the single deliberate deviation from "an
    explicitly named file is always inspected": naming a path is normally
    the session declaring it its own artifact, but a path under a
    caller-declared transient root is skipped whether it is named directly
    or reached through a walk. ``skipped``, when supplied, collects those
    paths. Exclusion resolution is fail-closed — a path whose own
    resolution fails is not excluded, so it is scanned.

    *prune_names* names directories a walk does not descend into: the entry
    is not enumerated, is not counted against the budget, and its subtree is
    never inspected. It applies during walks only — a witness root named
    directly on the command line (a top-level path) is not pruned by name,
    though it is still excluded when it resolves under an *exclude_roots*
    entry.

    *count_qualifying_only* changes what the walk budget counts. By default
    every enumerated entry counts, inspected or not, so a large tree of
    pre-window entries can exhaust the budget with nothing inspected. When
    true, only entries that qualify for inspection in the run's write window
    are counted, while structural entries (directories, symlinks to
    directories) are free to walk; a separate absolute enumeration cap
    (:data:`DEFAULT_MAX_WALK_ENUMERATIONS`) still bounds the raw scandir
    work, and reaching it reports the same ``scan-incomplete`` sentinel, so
    the budget stays a bound on the scan, never a way past it.

    *keys*, when supplied, collects one internal :class:`ApprovalKey` per
    rule hit — including hits in files discovered by a directory walk —
    computed at the match point; ``scan-incomplete`` sentinels never produce
    a key, so an incomplete scan can never be suppressed. With *project_root*,
    each key carries the file's canonical project-relative identity
    (:func:`canonical_project_identity`); a file whose identity is
    uncanonicalizable gets a key with ``path=None``, which can never suppress.

    *policy*, when supplied, suppresses approved rule hits INSIDE scanning —
    in explicitly named files and in files discovered by a walk alike — before
    the matched bytes are discarded (see :func:`scan_text`).
    """
    excluded_roots = (
        tuple(Path(root) for root in exclude_roots) if exclude_roots else ()
    )
    findings: list[SecretFinding] = []
    for raw_path in paths:
        path = Path(raw_path)
        if _excluded(path, excluded_roots):
            if skipped is not None:
                skipped.append(str(path))
            continue
        try:
            path_status = path.stat()
        except FileNotFoundError:
            # A touched path deleted before submission — or a sentinel that
            # names no path — has no content left to inspect.
            continue
        except OSError:
            # Fail-closed: what cannot be inspected is uninspected content,
            # never a clean scan.
            findings.append(
                incomplete_finding(
                    path,
                    "the path could not be inspected",
                    reason_code=FindingReason.UNREADABLE,
                )
            )
            continue
        if S_ISDIR(path_status.st_mode):
            findings.extend(
                _scan_directory(
                    path,
                    max_bytes,
                    max_walk_entries,
                    modified_since_ns,
                    prune_names=prune_names,
                    count_qualifying_only=count_qualifying_only,
                    keys=keys,
                    project_root=project_root,
                    policy=policy,
                )
            )
        elif S_ISREG(path_status.st_mode):
            findings.extend(
                _scan_file(
                    path, max_bytes, keys=keys, project_root=project_root, policy=policy
                )
            )
    return findings


def _excluded(path: Path, exclude_roots: tuple[Path, ...]) -> bool:
    """Whether *path* resolves under one of *exclude_roots*.

    Both sides are resolved so a link into the excluded tree is excluded and
    a link out of it is not. An unresolvable path or root never excludes:
    what the scan cannot place under a caller-declared transient root is
    scanned, so the skip stays fail-closed.
    """
    if not exclude_roots:
        return False
    try:
        resolved = path.resolve()
        roots = tuple(root.resolve() for root in exclude_roots)
    except (OSError, RuntimeError):
        return False
    return any(resolved.is_relative_to(root) for root in roots)


def scan_bytes(
    raw: bytes,
    *,
    label: str,
    max_bytes: int = DEFAULT_MAX_SCAN_BYTES,
    keys: list[ApprovalKey] | None = None,
    key_path: str | None = None,
    policy: ScanPolicy | None = None,
) -> list[SecretFinding]:
    """Scan in-memory file content and return all findings.

    Content past *max_bytes* is not inspected and adds a ``scan-incomplete``
    sentinel, exactly like an over-cap file on disk. NUL-containing content is
    split on NUL boundaries and every text run is scanned (see
    :func:`_scan_runs`), so binary content is inspected rather than skipped.
    The pre-commit hook scans the staged blobs it reads from the git index
    through this function.

    *keys*, when supplied, collects one internal :class:`ApprovalKey` per
    rule hit, carrying the canonical identity *key_path* (``None`` when
    uncanonicalizable). *policy*, when supplied, suppresses approved rule
    hits inside scanning (see :func:`scan_text`).
    """
    truncated = len(raw) > max_bytes
    findings = _scan_runs(
        raw[:max_bytes], label=label, keys=keys, key_path=key_path, policy=policy
    )
    if truncated:
        findings.append(
            incomplete_finding(
                label,
                f"content past the {max_bytes}-byte scan cap was not inspected",
                reason_code=FindingReason.FILE_SIZE_LIMIT,
            )
        )
    return findings


def incomplete_finding(
    path: str | Path,
    reason: str,
    *,
    reason_code: FindingReason = FindingReason.UNREADABLE,
) -> SecretFinding:
    """The sentinel finding for content the scan could not inspect.

    The finding's path is sanitized (control characters stripped) like every
    finding's, and it carries the closed *reason_code* classifying the
    sentinel; its ``(1, 1)`` location is an internal sentinel, never a real
    match location.
    """
    return SecretFinding(
        path=strip_control_characters(str(path)),
        rule_id=SCAN_INCOMPLETE_RULE_ID,
        description=f"File could not be fully inspected: {reason}",
        line=1,
        column=1,
        reason=reason_code,
    )


def _scan_runs(
    raw: bytes,
    *,
    label: str,
    keys: list[ApprovalKey] | None = None,
    key_path: str | None = None,
    policy: ScanPolicy | None = None,
) -> list[SecretFinding]:
    """Scan *raw* content, splitting NUL-containing content on NUL boundaries.

    A file whose bytes contain a NUL is not skipped wholesale — the
    artifact's writer controls the content, so one prepended NUL must not
    suppress the scan. Each non-NUL byte run is decoded and scanned on its
    own: a secret appended behind a NUL is still found, while genuine
    binaries stay passable (their text runs rarely match a rule, and the
    entropy-gated generic rule is unchanged). Findings keep file-accurate
    line numbers: a run's lines are offset by the line terminators in the
    runs before it. *keys*, *key_path*, and *policy* are passed through to
    :func:`scan_text`.
    """
    if b"\x00" not in raw:
        return scan_text(
            raw.decode("utf-8", errors="replace"),
            label=label,
            keys=keys,
            key_path=key_path,
            policy=policy,
        )
    findings: list[SecretFinding] = []
    line_offset = 0
    for run in raw.split(b"\x00"):
        text = run.decode("utf-8", errors="replace")
        for finding in scan_text(
            text, label=label, keys=keys, key_path=key_path, policy=policy
        ):
            findings.append(replace(finding, line=finding.line + line_offset))
        line_offset += len(_LINE_TERMINATOR.findall(text))
    return findings


def _scan_directory(
    path: Path,
    max_bytes: int,
    max_walk_entries: int,
    modified_since_ns: int | None = None,
    *,
    prune_names: Container[str] = frozenset(),
    count_qualifying_only: bool = False,
    keys: list[ApprovalKey] | None = None,
    project_root: str | Path | None = None,
    policy: ScanPolicy | None = None,
) -> list[SecretFinding]:
    """Scan every file under *path*, fail-closed on walk errors and budget.

    A directory the walk cannot enumerate yields a sentinel naming it (its
    subtree is uninspected content, never a clean scan), and a walk past
    *max_walk_entries* enumerated entries stops with one sentinel naming
    *path* for the unwalked remainder. With *modified_since_ns*, every
    descendant is still enumerated — and, unless *count_qualifying_only*,
    counted against the budget — while only the files the cutoff qualifies
    are scanned. A *prune_names* directory is never entered, and its entry
    is not counted. *keys* collects the internal approval key of every rule
    hit in the walked files, each carrying its canonical project-relative
    identity when *project_root* is supplied; *policy* suppresses approved
    hits inside scanning.
    """
    findings: list[SecretFinding] = []
    files, exhausted_limit = _walk_files(
        path,
        max_walk_entries,
        findings,
        modified_since_ns,
        prune_names=prune_names,
        count_qualifying_only=count_qualifying_only,
    )
    for file in files:
        findings.extend(
            _scan_file(
                file, max_bytes, keys=keys, project_root=project_root, policy=policy
            )
        )
    if exhausted_limit is not None:
        findings.append(
            incomplete_finding(
                path,
                f"the directory walk exceeded the {exhausted_limit}-entry "
                "budget; the remainder was not inspected",
                reason_code=(
                    FindingReason.ENUMERATION_LIMIT
                    if exhausted_limit == DEFAULT_MAX_WALK_ENUMERATIONS
                    else FindingReason.WALK_BUDGET
                ),
            )
        )
    return findings


def _walk_files(
    root: Path,
    max_entries: int,
    findings: list[SecretFinding],
    modified_since_ns: int | None = None,
    *,
    prune_names: Container[str] = frozenset(),
    count_qualifying_only: bool = False,
) -> tuple[list[Path], int | None]:
    """Collect the files under *root*, bounded by *max_entries* entries.

    Returns the files found (name-sorted within each directory) and the
    limit the walk exhausted (``None`` when it completed). An enumeration
    failure appends a sentinel naming the directory and leaves its subtree
    unwalked; an entry whose metadata cannot be read fails the same way,
    never as an unchanged entry. By default every enumerated entry counts
    against the budget, whether or not *modified_since_ns* qualifies its
    file for inspection; with *count_qualifying_only* only entries that
    qualify for inspection count, while every enumerated entry still counts
    against :data:`DEFAULT_MAX_WALK_ENUMERATIONS`. A directory named in
    *prune_names* is neither entered nor counted. A symlink keeps its
    target's type: a link to a file is collected when the link OR the
    target qualifies (a fresh link over pre-session content is a touched
    artifact too), and a link to a directory is walked under cycle detection,
    so a link back up the tree cannot loop the walk. A top-level *root* is
    never pruned by name — pruning applies to entries a walk descends into.
    """
    files: list[Path] = []
    counted = 0
    enumerated = 0
    exhausted_limit: int | None = None
    visited: set[tuple[int, int]] = set()
    try:
        visited.add(_directory_identity(os.stat(root)))
    except OSError:
        findings.append(
            incomplete_finding(
                root,
                "the directory could not be inspected",
                reason_code=FindingReason.UNREADABLE,
            )
        )
        return files, exhausted_limit
    pending: deque[Path] = deque([root])
    while pending and exhausted_limit is None:
        directory = pending.popleft()
        try:
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError:
            # Fail-closed: a directory the walk cannot enumerate is
            # uninspected content, never a clean scan.
            findings.append(
                incomplete_finding(
                    directory,
                    "the directory could not be enumerated",
                    reason_code=FindingReason.UNREADABLE,
                )
            )
            continue
        for child in children:
            if _pruned(child, prune_names):
                continue
            enumerated += 1
            if enumerated > DEFAULT_MAX_WALK_ENUMERATIONS:
                exhausted_limit = DEFAULT_MAX_WALK_ENUMERATIONS
                break
            if not count_qualifying_only:
                counted += 1
                if counted > max_entries:
                    exhausted_limit = max_entries
                    break
            try:
                child_status = child.stat(follow_symlinks=False)
            except OSError:
                findings.append(
                    incomplete_finding(
                        Path(child.path),
                        "the directory entry could not be inspected",
                        reason_code=FindingReason.UNREADABLE,
                    )
                )
                continue
            qualifying = False
            if S_ISDIR(child_status.st_mode):
                _enter_directory(Path(child.path), child_status, visited, pending)
            elif S_ISLNK(child_status.st_mode):
                qualifying = _collect_symlink(
                    child,
                    child_status,
                    modified_since_ns,
                    visited,
                    pending,
                    files,
                    findings,
                )
            elif S_ISREG(child_status.st_mode) and _qualifies(
                child_status.st_mtime_ns, modified_since_ns
            ):
                files.append(Path(child.path))
                qualifying = True
            if count_qualifying_only and qualifying:
                counted += 1
                if counted > max_entries:
                    exhausted_limit = max_entries
                    break
    return files, exhausted_limit


def _pruned(entry: os.DirEntry[str], prune_names: Container[str]) -> bool:
    """Whether *entry* is a plain directory the walk must not descend into.

    Only a real directory entry is pruned: a symlink keeps the symlink
    policy (its target is walked under cycle detection), and an entry whose
    type cannot be established falls through to the ordinary stat path,
    which fails closed there.
    """
    if not prune_names or entry.name not in prune_names:
        return False
    try:
        return entry.is_dir(follow_symlinks=False)
    except OSError:
        return False


def _directory_identity(status: os.stat_result) -> tuple[int, int]:
    """The (device, inode) pair the walk keys directory cycle detection by."""
    return (status.st_dev, status.st_ino)


def _enter_directory(
    path: Path,
    status: os.stat_result,
    visited: set[tuple[int, int]],
    pending: deque[Path],
) -> None:
    """Queue *path* for the walk unless its directory was already visited."""
    identity = _directory_identity(status)
    if identity in visited:
        return
    visited.add(identity)
    pending.append(path)


def _collect_symlink(
    entry: os.DirEntry[str],
    link_status: os.stat_result,
    modified_since_ns: int | None,
    visited: set[tuple[int, int]],
    pending: deque[Path],
    files: list[Path],
    findings: list[SecretFinding],
) -> bool:
    """Walk or collect a symlinked entry according to its target's type.

    The link's own mtime joins the target's when qualifying a file: a fresh
    link over pre-session content is a touched artifact, and so is a fresh
    target behind an old link. A target that no longer exists has no content
    behind the link; a target whose metadata cannot be read is uninspected
    content, reported rather than assumed unchanged. Returns whether the
    entry qualified for inspection (a file was collected) — what a
    qualifying-only walk budget counts.
    """
    path = Path(entry.path)
    try:
        target_status = entry.stat()
    except FileNotFoundError:
        return False
    except OSError:
        findings.append(
            incomplete_finding(
                path,
                "the symlink target could not be inspected",
                reason_code=FindingReason.UNREADABLE,
            )
        )
        return False
    if S_ISDIR(target_status.st_mode):
        _enter_directory(path, target_status, visited, pending)
        return False
    if S_ISREG(target_status.st_mode) and (
        _qualifies(target_status.st_mtime_ns, modified_since_ns)
        or _qualifies(link_status.st_mtime_ns, modified_since_ns)
    ):
        files.append(path)
        return True
    return False


def _qualifies(mtime_ns: int, modified_since_ns: int | None) -> bool:
    """Whether *mtime_ns* falls inside the run's write window (equality too)."""
    return modified_since_ns is None or mtime_ns >= modified_since_ns


def _scan_file(
    path: Path,
    max_bytes: int,
    *,
    keys: list[ApprovalKey] | None = None,
    project_root: str | Path | None = None,
    policy: ScanPolicy | None = None,
) -> list[SecretFinding]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError:
        # Fail-closed: an unreadable artifact is uninspected content, never a
        # clean scan.
        return [
            incomplete_finding(
                path, "the file could not be read", reason_code=FindingReason.UNREADABLE
            )
        ]
    key_path = (
        canonical_project_identity(path, project_root=project_root)
        if project_root is not None
        else None
    )
    return scan_bytes(
        raw,
        label=str(path),
        max_bytes=max_bytes,
        keys=keys,
        key_path=key_path,
        policy=policy,
    )
