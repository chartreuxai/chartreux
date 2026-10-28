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
submission boundary; there is deliberately no tool registration or other
model-visible surface for this module.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import log2
import os
from pathlib import Path
import re


#: Findings never carry the matched secret value; each carries only the path,
#: the rule that matched, a redacted description, and the match location.
@dataclass(frozen=True)
class SecretFinding:
    """One rule hit, without the matched value."""

    path: str
    rule_id: str
    description: str
    line: int
    column: int


@dataclass(frozen=True)
class SecretRule:
    """One detection rule.

    ``pattern`` finds candidates on a line. When ``entropy_group`` names a
    capture group, the match only counts when that group's Shannon entropy
    reaches ``entropy_threshold`` bits per character; otherwise every pattern
    match counts.
    """

    rule_id: str
    description: str
    pattern: re.Pattern[str]
    entropy_group: int | None = None
    entropy_threshold: float = 0.0


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
    # (`access_token = <high-entropy literal>`) are unaffected.
    SecretRule(
        "generic-keyword-assignment",
        "Credential-like assignment with a high-entropy value",
        re.compile(
            r"(?i)(?<![A-Za-z0-9_])[A-Za-z0-9_]*"
            r"(?:api[_-]?key|apikey|secret(?:[_-]?key)?|token|password|passwd|pwd"
            r"|credential|auth[_-]?(?:key|token)|access[_-]?(?:key|token)"
            r"|private[_-]?key|client[_-]?secret)"
            r"[A-Za-z0-9_]*(?<!tokens)"
            r"\W{0,3}[:=]\W{0,3}"
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


def scan_text(text: str, *, label: str = "<text>") -> list[SecretFinding]:
    """Run every rule over *text*, line by line.

    Findings are deduplicated per rule and line; repeated hits of one rule on
    the same line collapse into a single finding at the first match's position.
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
                findings.append(
                    SecretFinding(
                        path=label,
                        rule_id=rule.rule_id,
                        description=rule.description,
                        line=line_number,
                        column=match.start() + 1,
                    )
                )
    return findings


def scan_paths(
    paths: Iterable[str | Path],
    *,
    max_bytes: int = DEFAULT_MAX_SCAN_BYTES,
    max_walk_entries: int = DEFAULT_MAX_WALK_ENTRIES,
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
    """
    findings: list[SecretFinding] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            findings.extend(_scan_directory(path, max_bytes, max_walk_entries))
        elif path.is_file():
            findings.extend(_scan_file(path, max_bytes))
    return findings


def scan_bytes(
    raw: bytes, *, label: str, max_bytes: int = DEFAULT_MAX_SCAN_BYTES
) -> list[SecretFinding]:
    """Scan in-memory file content and return all findings.

    Content past *max_bytes* is not inspected and adds a ``scan-incomplete``
    sentinel, exactly like an over-cap file on disk. NUL-containing content is
    split on NUL boundaries and every text run is scanned (see
    :func:`_scan_runs`), so binary content is inspected rather than skipped.
    The pre-commit hook scans the staged blobs it reads from the git index
    through this function.
    """
    truncated = len(raw) > max_bytes
    findings = _scan_runs(raw[:max_bytes], label=label)
    if truncated:
        findings.append(
            incomplete_finding(
                label, f"content past the {max_bytes}-byte scan cap was not inspected"
            )
        )
    return findings


def incomplete_finding(path: str | Path, reason: str) -> SecretFinding:
    """The sentinel finding for content the scan could not inspect."""
    return SecretFinding(
        path=str(path),
        rule_id=SCAN_INCOMPLETE_RULE_ID,
        description=f"File could not be fully inspected: {reason}",
        line=1,
        column=1,
    )


def _scan_runs(raw: bytes, *, label: str) -> list[SecretFinding]:
    """Scan *raw* content, splitting NUL-containing content on NUL boundaries.

    A file whose bytes contain a NUL is not skipped wholesale — the
    artifact's writer controls the content, so one prepended NUL must not
    suppress the scan. Each non-NUL byte run is decoded and scanned on its
    own: a secret appended behind a NUL is still found, while genuine
    binaries stay passable (their text runs rarely match a rule, and the
    entropy-gated generic rule is unchanged). Findings keep file-accurate
    line numbers: a run's lines are offset by the line terminators in the
    runs before it.
    """
    if b"\x00" not in raw:
        return scan_text(raw.decode("utf-8", errors="replace"), label=label)
    findings: list[SecretFinding] = []
    line_offset = 0
    for run in raw.split(b"\x00"):
        text = run.decode("utf-8", errors="replace")
        for finding in scan_text(text, label=label):
            findings.append(replace(finding, line=finding.line + line_offset))
        line_offset += len(_LINE_TERMINATOR.findall(text))
    return findings


def _scan_directory(
    path: Path, max_bytes: int, max_walk_entries: int
) -> list[SecretFinding]:
    """Scan every file under *path*, fail-closed on walk errors and budget.

    A directory the walk cannot enumerate yields a sentinel naming it (its
    subtree is uninspected content, never a clean scan), and a walk past
    *max_walk_entries* enumerated entries stops with one sentinel naming
    *path* for the unwalked remainder.
    """
    findings: list[SecretFinding] = []
    files, over_budget = _walk_files(path, max_walk_entries, findings)
    for file in files:
        findings.extend(_scan_file(file, max_bytes))
    if over_budget:
        findings.append(
            incomplete_finding(
                path,
                f"the directory walk exceeded the {max_walk_entries}-entry "
                "budget; the remainder was not inspected",
            )
        )
    return findings


def _walk_files(
    root: Path, max_entries: int, findings: list[SecretFinding]
) -> tuple[list[Path], bool]:
    """Collect the files under *root*, bounded by *max_entries* entries.

    Returns the files found (name-sorted within each directory) and whether
    the walk ran out of budget. An enumeration failure appends a sentinel
    naming the directory and leaves its subtree unwalked; an entry whose type
    cannot be determined fails the same way.
    """
    files: list[Path] = []
    enumerated = 0
    over_budget = False
    pending: deque[Path] = deque([root])
    while pending and not over_budget:
        directory = pending.popleft()
        try:
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError:
            # Fail-closed: a directory the walk cannot enumerate is
            # uninspected content, never a clean scan.
            findings.append(
                incomplete_finding(directory, "the directory could not be enumerated")
            )
            continue
        for child in children:
            enumerated += 1
            if enumerated > max_entries:
                over_budget = True
                break
            try:
                if child.is_dir(follow_symlinks=False):
                    pending.append(Path(child.path))
                elif child.is_file():
                    files.append(Path(child.path))
            except OSError:
                findings.append(
                    incomplete_finding(
                        Path(child.path), "the directory entry could not be inspected"
                    )
                )
    return files, over_budget


def _scan_file(path: Path, max_bytes: int) -> list[SecretFinding]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError:
        # Fail-closed: an unreadable artifact is uninspected content, never a
        # clean scan.
        return [incomplete_finding(path, "the file could not be read")]
    return scan_bytes(raw, label=str(path), max_bytes=max_bytes)
