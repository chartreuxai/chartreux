"""Pre-commit entry point for the artifact-side secret scan.

The hook scans the STAGED content of the filenames pre-commit passes on the
command line — the blob each path carries in the git index, read with
``git cat-file blob :<path>`` — not the working-tree file: stage a secret,
revert the working tree, commit, and the hook still inspects the staged
secret that ships. The scan itself is :mod:`chartreux.core.tools.secret_scan`;
the commit blocks when any rule matches or when an artifact could not be
fully inspected (the scanner's ``scan-incomplete`` sentinel). A path whose
staged blob cannot be read from the index is a staged deletion when HEAD
carries the path and the index does not — the commit ships no content for
it, so there is nothing to scan; every other unreadable path fails closed
the same way — a sentinel naming the reason, never a silent working-tree
fallback. Findings are printed as path, rule id, and location only: the
scanner's findings never carry the matched value, so the hook's output
cannot disclose a secret. Invocation is harness-side (pre-commit for
user-driven commits), never model-invoked.
"""

from __future__ import annotations

from collections.abc import Iterable
import subprocess
import sys

from chartreux.core.tools.secret_scan import (
    DEFAULT_MAX_SCAN_BYTES,
    SCAN_INCOMPLETE_RULE_ID,
    SecretFinding,
    incomplete_finding,
    scan_bytes,
)

#: Timeout for one staged-blob lookup: the hook runs at commit time and must
#: not hang on a wedged git.
_GIT_LOOKUP_TIMEOUT_SECONDS = 10.0


def staged_blob(path: str, *, max_bytes: int = DEFAULT_MAX_SCAN_BYTES) -> bytes | None:
    """The staged (git index) content of *path*, capped at *max_bytes* + 1.

    The lookup reads the index, not the working tree, so the scanned content
    is exactly what would be committed. The read is bounded: an over-cap blob
    yields its first *max_bytes* + 1 bytes and the caller's truncation
    sentinel reports the uninspected remainder. ``None`` means the index
    holds no readable blob for the path (or the lookup failed); callers
    distinguish a staged deletion from a genuine failure on it.
    """
    try:
        process = subprocess.Popen(
            ["git", "cat-file", "blob", f":{path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    stdout = process.stdout
    if stdout is None:
        process.kill()
        process.wait()
        return None
    try:
        raw = stdout.read(max_bytes + 1)
    except OSError:
        raw = b""
    finally:
        stdout.close()
    try:
        returncode = process.wait(timeout=_GIT_LOOKUP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return None
    if returncode != 0 and len(raw) < max_bytes + 1:
        # The lookup itself failed — no blob at the path, or a broken object
        # mid-stream — never a clean or partially trusted result.
        return None
    return raw


def _git_object_exists(revision: str) -> bool:
    """Whether one bounded git probe finds the object *revision* names.

    ``git cat-file -e <revision>`` exits 0 exactly when the revision names
    an existing object, so it distinguishes "the index holds a blob for the
    path" (``:<path>``) from "HEAD carries the path" (``HEAD:<path>``)
    without reading any content. Any failure of the probe itself (no git, not
    a repository, a timeout) reads as absent, fail closed.
    """
    try:
        completed = subprocess.run(
            ["git", "cat-file", "-e", revision],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_LOOKUP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def scan_staged(filenames: Iterable[str]) -> list[SecretFinding]:
    """Scan the staged content of each filename, fail-closed per path.

    A path whose staged blob cannot be read from the index is examined
    before it is condemned: a STAGED DELETION — the path is tracked in HEAD
    and absent from the index — ships no content for the path, so there is
    nothing to scan and the path passes; it is a legitimate state, not a
    lookup failure. Every other unreadable path is a genuine lookup failure
    and adds a ``scan-incomplete`` sentinel naming it and the reason — the
    hook never falls back to the working tree, so a staged secret cannot be
    laundered by reverting the file on disk.
    """
    findings: list[SecretFinding] = []
    for filename in filenames:
        raw = staged_blob(filename)
        if raw is not None:
            findings.extend(scan_bytes(raw, label=filename))
        elif _git_object_exists(f":{filename}"):
            # The index holds an entry whose blob could not be read: a
            # broken or missing object mid-stream — a genuine failure.
            findings.append(
                incomplete_finding(
                    filename,
                    "the staged blob could not be read from the git object database",
                )
            )
        elif _git_object_exists(f"HEAD:{filename}"):
            # A staged deletion: tracked in HEAD, absent from the index —
            # the commit removes the path and ships no content for it.
            continue
        else:
            findings.append(
                incomplete_finding(
                    filename,
                    "no staged content at the path and it is not a staged deletion",
                )
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    """Scan the staged content of the filenames in *argv*; return 1 on a hit.

    With no filenames the scan is empty and the commit passes. Findings print
    as ``path:line:column: rule-id (description)``; the matched value is never
    printed because :class:`~chartreux.core.tools.secret_scan.SecretFinding`
    does not carry it. An artifact the scan cannot fully inspect blocks the
    commit exactly like a rule hit: the incomplete line names the path and the
    reason, never any file content.
    """
    filenames = sys.argv[1:] if argv is None else argv
    findings = scan_staged(filenames)
    for finding in findings:
        print(
            f"{finding.path}:{finding.line}:{finding.column}: "
            f"{finding.rule_id} ({finding.description})"
        )
    if findings:
        print(
            f"secret scan: {len(findings)} finding(s); "
            "remove or rotate the secret. Commit blocked."
        )
        incomplete = [
            finding
            for finding in findings
            if finding.rule_id == SCAN_INCOMPLETE_RULE_ID
        ]
        if incomplete:
            print(
                f"secret scan: {len(incomplete)} artifact(s) could not be fully "
                "inspected; commit blocked until every artifact can be scanned "
                "in full."
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
