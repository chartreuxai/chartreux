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
"""

from __future__ import annotations

from collections import deque
from collections.abc import Container, Iterable
from dataclasses import dataclass, replace
from math import log2
import os
from pathlib import Path
import re
from stat import S_ISDIR, S_ISLNK, S_ISREG


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
    modified_since_ns: int | None = None,
    exclude_roots: Iterable[str | Path] = (),
    prune_names: Container[str] = frozenset(),
    skipped: list[str] | None = None,
    count_qualifying_only: bool = False,
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
            findings.append(incomplete_finding(path, "the path could not be inspected"))
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
                )
            )
        elif S_ISREG(path_status.st_mode):
            findings.extend(_scan_file(path, max_bytes))
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
    path: Path,
    max_bytes: int,
    max_walk_entries: int,
    modified_since_ns: int | None = None,
    *,
    prune_names: Container[str] = frozenset(),
    count_qualifying_only: bool = False,
) -> list[SecretFinding]:
    """Scan every file under *path*, fail-closed on walk errors and budget.

    A directory the walk cannot enumerate yields a sentinel naming it (its
    subtree is uninspected content, never a clean scan), and a walk past
    *max_walk_entries* enumerated entries stops with one sentinel naming
    *path* for the unwalked remainder. With *modified_since_ns*, every
    descendant is still enumerated — and, unless *count_qualifying_only*,
    counted against the budget — while only the files the cutoff qualifies
    are scanned. A *prune_names* directory is never entered, and its entry
    is not counted.
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
        findings.extend(_scan_file(file, max_bytes))
    if exhausted_limit is not None:
        findings.append(
            incomplete_finding(
                path,
                f"the directory walk exceeded the {exhausted_limit}-entry "
                "budget; the remainder was not inspected",
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
            incomplete_finding(root, "the directory could not be inspected")
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
                incomplete_finding(directory, "the directory could not be enumerated")
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
                        Path(child.path), "the directory entry could not be inspected"
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
            incomplete_finding(path, "the symlink target could not be inspected")
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


def _scan_file(path: Path, max_bytes: int) -> list[SecretFinding]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError:
        # Fail-closed: an unreadable artifact is uninspected content, never a
        # clean scan.
        return [incomplete_finding(path, "the file could not be read")]
    return scan_bytes(raw, label=str(path), max_bytes=max_bytes)
