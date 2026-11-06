"""Scanner hardening (v0.7.3): the per-run directory-walk cutoff and the
assignment-separator rule.

Two behaviors are pinned here.

**B — the per-run cutoff.** A touched path that is a DIRECTORY (the resolver
collects `.` for a plain extraction, an in-place patch, or a bare `git apply`,
and the directory destination of a copy) enumerated the entire tree into the
scan: pre-session content, including over-cap binaries, blocked a run that had
merely written into an existing tree. ``scan_paths(..., modified_since_ns=)``
now enumerates every descendant but inspects only the files modified at or
after the run started. Explicitly named file paths stay unconditional; an
omitted cutoff keeps the full walk.

**D — the assignment separator.** The generic keyword rule's separator used to
accept up to three non-word characters on either side, so a pytest node id
(`<file>.py::<test>`) satisfied it: the `.py::` run read as padding plus a
separator, and the test name as the "value". Neither padding side may now
absorb a colon, so no separator ever sits beside another colon.

The seeded fixtures are built by concatenation on purpose: this file is itself
a touched path at a submission boundary, so its source has to scan clean.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chartreux.core.tools import secret_scan
from chartreux.core.tools.secret_scan import DEFAULT_MAX_SCAN_BYTES, scan_paths

AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
HIGH_ENTROPY_VALUE = "r7Bqz9WkPm3XvT2sLd8Yh5Nj4Fg6Ua1Q"


def _seed_file(path: Path) -> None:
    """Write a file the AWS rule flags, with the key assembled at runtime."""
    path.write_text(f"# config\naws_access_key_id = {AWS_KEY}\n", encoding="utf-8")


def _set_mtime_ns(path: Path, mtime_ns: int) -> None:
    """Pin a file's mtime exactly, leaving its access time alone."""
    current = path.stat()
    os.utime(path, ns=(current.st_atime_ns, mtime_ns))


# --- B: the per-run directory walk cutoff ------------------------------------


def test_directory_cutoff_inspects_only_files_modified_since_the_cutoff(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    pre_session = tree / "sub" / "old.py"
    _seed_file(pre_session)
    in_run = tree / "fresh.py"
    _seed_file(in_run)
    cutoff = in_run.stat().st_mtime_ns
    _set_mtime_ns(pre_session, cutoff - 1)

    findings = scan_paths([tree], modified_since_ns=cutoff)

    # The tree was enumerated; only the run's own file was inspected.
    assert {finding.path for finding in findings} == {str(in_run)}


def test_directory_cutoff_includes_a_file_modified_exactly_at_the_cutoff(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    artifact = tree / "at-cutoff.py"
    _seed_file(artifact)
    cutoff = artifact.stat().st_mtime_ns

    findings = scan_paths([tree], modified_since_ns=cutoff)

    assert [finding.path for finding in findings] == [str(artifact)]


def test_directory_cutoff_never_consults_ctime(tmp_path: Path) -> None:
    # The cutoff is the file's MODIFICATION time. Backdating a file leaves a
    # fresh ctime behind; the scan must not treat that as "written in run".
    tree = tmp_path / "tree"
    tree.mkdir()
    artifact = tree / "backdated.py"
    _seed_file(artifact)
    cutoff = artifact.stat().st_mtime_ns + 1
    _set_mtime_ns(artifact, cutoff - 10_000_000)
    status = artifact.stat()
    assert status.st_mtime_ns < cutoff
    assert status.st_ctime_ns >= cutoff

    assert scan_paths([tree], modified_since_ns=cutoff) == []


def test_explicit_file_path_is_inspected_whatever_its_mtime(tmp_path: Path) -> None:
    # Naming a file is the session declaring it its own artifact, so the
    # cutoff does not apply to it.
    artifact = tmp_path / "old.py"
    _seed_file(artifact)

    findings = scan_paths([artifact], modified_since_ns=artifact.stat().st_mtime_ns + 1)

    assert [finding.rule_id for finding in findings] == ["aws-access-key-id"]


def test_directory_walk_without_a_cutoff_scans_every_file(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    first = tree / "first.py"
    _seed_file(first)
    second = tree / "second.py"
    _seed_file(second)
    _set_mtime_ns(second, first.stat().st_mtime_ns - 1)

    findings = scan_paths([tree])

    assert {finding.path for finding in findings} == {str(first), str(second)}


def test_directory_cutoff_drops_a_pre_session_over_cap_file(tmp_path: Path) -> None:
    # The knock-on the cutoff is for: a pre-session binary larger than the
    # per-file cap is not the run's content, so it no longer blocks a run that
    # wrote into the same tree.
    tree = tmp_path / "tree"
    tree.mkdir()
    artifact = tree / "archive.bin"
    artifact.write_bytes(b"x" * (DEFAULT_MAX_SCAN_BYTES + 1))
    run_file = tree / "run.txt"
    run_file.write_text("clean\n", encoding="utf-8")
    cutoff = run_file.stat().st_mtime_ns
    _set_mtime_ns(artifact, cutoff - 1)

    assert scan_paths([tree], modified_since_ns=cutoff) == []
    # The cap itself is unchanged: the same walk without a cutoff still
    # reports the truncation.
    with_cutoff_dropped = scan_paths([tree])
    assert [finding.rule_id for finding in with_cutoff_dropped] == ["scan-incomplete"]
    assert with_cutoff_dropped[0].path == str(artifact)


def test_directory_cutoff_still_counts_pre_session_entries_against_the_budget(
    tmp_path: Path,
) -> None:
    # The entry budget counts every ENUMERATED entry, inspected or not: the
    # cutoff filters what is inspected, never what the walk has to see.
    tree = tmp_path / "tree"
    tree.mkdir()
    for index in range(4):
        entry = tree / f"old-{index}.txt"
        entry.write_text("clean\n", encoding="utf-8")
        _set_mtime_ns(entry, 1_000_000)

    findings = scan_paths([tree], max_walk_entries=2, modified_since_ns=2_000_000)

    assert [finding.rule_id for finding in findings] == ["scan-incomplete"]
    assert findings[0].path == str(tree)
    assert "budget" in findings[0].description


def test_fresh_symlink_over_a_pre_session_target_is_inspected(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    target = tmp_path / "target.py"
    _seed_file(target)
    cutoff = target.stat().st_mtime_ns + 1_000_000
    _set_mtime_ns(target, cutoff - 1)
    link = tree / "link.py"
    link.symlink_to(target)
    # The link is the in-run write, so its own mtime is fresh.
    os.utime(link, ns=(cutoff, cutoff), follow_symlinks=False)

    findings = scan_paths([tree], modified_since_ns=cutoff)

    # A fresh link over pre-session content is still a touched artifact.
    assert [finding.path for finding in findings] == [str(link)]


def test_stale_symlink_to_an_in_run_target_is_inspected(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    target = tmp_path / "target.py"
    _seed_file(target)
    cutoff = target.stat().st_mtime_ns
    link = tree / "link.py"
    link.symlink_to(target)
    os.utime(link, ns=(cutoff - 5_000_000, cutoff - 5_000_000), follow_symlinks=False)

    findings = scan_paths([tree], modified_since_ns=cutoff)

    assert [finding.path for finding in findings] == [str(link)]


def test_stale_symlink_to_a_stale_target_is_skipped(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    target = tmp_path / "target.py"
    _seed_file(target)
    cutoff = target.stat().st_mtime_ns + 1_000_000
    _set_mtime_ns(target, cutoff - 10_000_000)
    link = tree / "link.py"
    link.symlink_to(target)
    os.utime(link, ns=(cutoff - 5_000_000, cutoff - 5_000_000), follow_symlinks=False)

    assert scan_paths([tree], modified_since_ns=cutoff) == []


def test_directory_symlink_is_walked_and_a_cycle_terminates(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    (tree / "real").mkdir(parents=True)
    in_run = tree / "real" / "in-run.py"
    _seed_file(in_run)
    cutoff = in_run.stat().st_mtime_ns
    (tree / "linked").symlink_to(tree / "real")
    (tree / "cycle").symlink_to(tree)

    findings = scan_paths([tree], modified_since_ns=cutoff)

    # The target subtree is reached (once) and the link back to the root does
    # not loop the walk or turn it into an incomplete scan.
    assert [finding.rule_id for finding in findings] == ["aws-access-key-id"]


def test_symlink_loop_reports_scan_incomplete(tmp_path: Path) -> None:
    # A target whose metadata cannot be read is uninspected content, never a
    # clean scan: the link pair resolves to no readable target at all.
    tree = tmp_path / "tree"
    tree.mkdir()
    first = tree / "first"
    second = tree / "second"
    first.symlink_to(second)
    second.symlink_to(first)

    findings = scan_paths([tree])

    # Both links name the same unresolvable run: each is its own entry.
    assert {finding.rule_id for finding in findings} == {"scan-incomplete"}
    assert {finding.path for finding in findings} == {str(first), str(second)}


def test_dangling_symlink_is_skipped(tmp_path: Path) -> None:
    # Nothing sits behind a dangling link, so there is no content to inspect
    # and no reason to block (the same rule as a deleted touched path).
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "gone.py").symlink_to(tmp_path / "never-written.py")

    assert scan_paths([tree]) == []


def test_uninspectable_path_metadata_reports_scan_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.py"
    _seed_file(artifact)
    real_stat = Path.stat

    def refusing_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == artifact:
            raise PermissionError("permission denied")
        return real_stat(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", refusing_stat)

    findings = scan_paths([artifact])

    assert [finding.rule_id for finding in findings] == ["scan-incomplete"]
    assert findings[0].path == str(artifact)


# --- D: the assignment separator ---------------------------------------------


def test_assignment_separator_still_matches_equals_and_colon() -> None:
    for text in (
        f"password = {HIGH_ENTROPY_VALUE}\n",
        f"api_key: {HIGH_ENTROPY_VALUE}\n",
        f"api_key:{HIGH_ENTROPY_VALUE}\n",
        f'client_secret = "{HIGH_ENTROPY_VALUE}"\n',
    ):
        findings = secret_scan.scan_text(text, label="assignment")
        assert [finding.rule_id for finding in findings] == [
            "generic-keyword-assignment"
        ], text


def test_pytest_node_id_is_not_an_assignment() -> None:
    # The node-id false positive: `.py::<test>` used to read as padding, a
    # separator, and a value. The test name is not a credential.
    for text in (
        "tests/core/test_tools.py::test_api_key_resolution_is_exact\n",
        "tests/core/test_secret_scan.py::test_bare_high_entropy_literal_fires\n",
        "tests/cli/textual_ui/test_program_status.py::TestOwnershipTokenCapture"
        "::test_unmapped_typed_id_withholds_without_a_live_client\n",
    ):
        assert secret_scan.scan_text(text, label="node-id") == [], text


def test_double_colon_separator_is_not_an_assignment() -> None:
    # Second-colon backtracking: neither colon of a "::" run can serve as the
    # separator — as padding it is forbidden, as the separator the other colon
    # is adjacent to it.
    for text in (
        f"token::{HIGH_ENTROPY_VALUE}\n",
        f"token:: {HIGH_ENTROPY_VALUE}\n",
        f"api_key::{HIGH_ENTROPY_VALUE}\n",
        f"secret::{HIGH_ENTROPY_VALUE}\n",
        f"password:: {HIGH_ENTROPY_VALUE}\n",
    ):
        assert secret_scan.scan_text(text, label="double-colon") == [], text
