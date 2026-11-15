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

**A1 (v0.7.9) — value-scoped approval keys.** The scanner computes an internal
approval key per rule hit at the match point: SHA-256 over the rule's
value-span bytes (capture group when the rule defines one, else the full
match), plus the file's canonical project-relative identity when a project
root is supplied. The concrete per-rule span/approvability table and the
canonical-identity, symlink, and walk regressions are pinned below.

The seeded fixtures are built by concatenation on purpose: this file is itself
a touched path at a submission boundary, so its source has to scan clean.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from chartreux.core.tools import secret_scan
from chartreux.core.tools.secret_scan import DEFAULT_MAX_SCAN_BYTES, scan_paths

AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
HIGH_ENTROPY_VALUE = "r7Bqz9WkPm3XvT2sLd8Yh5Nj4Fg6Ua1Q"

VALUE40 = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
AWS_SECRET_LINE = "aws_secret_access_key = '" + VALUE40 + "'"
GH_SAMPLE = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
GLPAT_SAMPLE = "glpat-" + "a1B2c3D4e5F6g7H8i9J0k1L2"
XOX_SAMPLE = "xoxb-" + "1234567890ABCDEF"
WEBHOOK_SAMPLE = (
    "https://hooks.slack.com/services/" + "T12345678" + "/B12345678/" + "abcdefgh12"
)
SK_SAMPLE = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
GOOGLE_SAMPLE = "AIza" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R"
STRIPE_SAMPLE = "sk_live_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3"
NPM_SAMPLE = "npm_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
PYPI_SAMPLE = "pypi-AgEIcHlwaS5vcmc" + "a1B2c3D4e5F6g7H8i9J0k1L2"
HEADER_SAMPLE = "-----BEGIN " + "RSA PRIVATE KEY-----"

#: The per-rule digest-basis table: (rule id, sample line, expected value span).
#: ``None`` marks a non-approvable rule — its match covers no complete
#: sensitive material, so no approvable key exists for it.
_RULE_SAMPLES: tuple[tuple[str, str, str | None], ...] = (
    ("aws-access-key-id", f"aws_access_key_id = {AWS_KEY}", AWS_KEY),
    ("aws-secret-access-key", AWS_SECRET_LINE, AWS_SECRET_LINE),
    ("github-token", GH_SAMPLE, GH_SAMPLE),
    ("gitlab-token", GLPAT_SAMPLE, GLPAT_SAMPLE),
    ("slack-token", XOX_SAMPLE, XOX_SAMPLE),
    ("slack-webhook", WEBHOOK_SAMPLE, WEBHOOK_SAMPLE),
    ("openai-api-key", SK_SAMPLE, SK_SAMPLE),
    ("google-api-key", GOOGLE_SAMPLE, GOOGLE_SAMPLE),
    ("stripe-secret-key", STRIPE_SAMPLE, STRIPE_SAMPLE),
    ("npm-token", NPM_SAMPLE, NPM_SAMPLE),
    ("pypi-token", PYPI_SAMPLE, PYPI_SAMPLE),
    ("private-key-header", HEADER_SAMPLE, None),
    (
        "generic-keyword-assignment",
        f'api_key = "{HIGH_ENTROPY_VALUE}"',
        HIGH_ENTROPY_VALUE,
    ),
)


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


# --- A1: per-rule span/approvability table and canonical identities -----------


def test_rule_table_approvability_inventory() -> None:
    # 13 rules at the baseline; exactly one is non-approvable.
    assert len(secret_scan._RULE_TABLE) == 13
    non_approvable = {
        rule.rule_id for rule in secret_scan._RULE_TABLE if not rule.approvable
    }
    assert non_approvable == {"private-key-header"}
    assert sum(rule.approvable for rule in secret_scan._RULE_TABLE) == 12


def test_per_rule_span_and_approvability_table() -> None:
    # The enumeration source is the rule table itself: every rule appears
    # exactly once in the pinned sample table.
    assert {rule.rule_id for rule in secret_scan._RULE_TABLE} == {
        rule_id for rule_id, _, _ in _RULE_SAMPLES
    }
    assert len(_RULE_SAMPLES) == len(secret_scan._RULE_TABLE) == 13
    for rule_id, line, expected_span in _RULE_SAMPLES:
        keys: list[secret_scan.ApprovalKey] = []
        findings = secret_scan.scan_text(line + "\n", label="sample", keys=keys)
        rule_findings = [f for f in findings if f.rule_id == rule_id]
        assert len(rule_findings) == 1, rule_id
        rule_keys = [k for k in keys if k.rule_id == rule_id]
        if expected_span is None:
            # Non-approvable partial-coverage rule: the hit blocks and no
            # approvable value-span key exists for it.
            assert rule_keys == [], rule_id
        else:
            assert len(rule_keys) == 1, rule_id
            assert (
                rule_keys[0].digest
                == hashlib.sha256(expected_span.encode("utf-8")).hexdigest()
            ), rule_id


def test_explicit_path_key_carries_canonical_relative_identity(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "sub").mkdir(parents=True)
    artifact = root / "sub" / "f.py"
    _seed_file(artifact)
    keys: list[secret_scan.ApprovalKey] = []
    findings = scan_paths([artifact], project_root=root, keys=keys)
    assert [finding.rule_id for finding in findings] == ["aws-access-key-id"]
    assert len(keys) == 1
    assert keys[0].path == "sub/f.py"
    assert keys[0].rule_id == "aws-access-key-id"
    assert keys[0].digest == hashlib.sha256(AWS_KEY.encode("utf-8")).hexdigest()


def test_walked_files_carry_canonical_relative_keys(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    tree = root / "tree"
    (tree / "nested").mkdir(parents=True)
    _seed_file(tree / "a.py")
    _seed_file(tree / "nested" / "b.py")
    keys: list[secret_scan.ApprovalKey] = []
    findings = scan_paths([tree], project_root=root, keys=keys)
    assert {finding.path for finding in findings} == {
        str(tree / "a.py"),
        str(tree / "nested" / "b.py"),
    }
    assert {key.path for key in keys} == {"tree/a.py", "tree/nested/b.py"}
    assert len(keys) == 2


def test_relative_spelling_yields_the_same_canonical_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    artifact = root / "f.py"
    _seed_file(artifact)
    monkeypatch.chdir(tmp_path)
    keys_relative: list[secret_scan.ApprovalKey] = []
    keys_absolute: list[secret_scan.ApprovalKey] = []
    scan_paths(["proj/f.py"], project_root="proj", keys=keys_relative)
    scan_paths([artifact], project_root=root, keys=keys_absolute)
    assert keys_relative == keys_absolute
    assert keys_relative[0].path == "f.py"


def test_symlink_into_the_project_canonicalizes_to_the_target(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    artifact = root / "f.py"
    _seed_file(artifact)
    outside = tmp_path / "scratch"
    outside.mkdir()
    link = outside / "link.py"
    link.symlink_to(artifact)
    keys_link: list[secret_scan.ApprovalKey] = []
    keys_direct: list[secret_scan.ApprovalKey] = []
    findings = scan_paths([link], project_root=root, keys=keys_link)
    scan_paths([artifact], project_root=root, keys=keys_direct)
    assert [finding.rule_id for finding in findings] == ["aws-access-key-id"]
    # Both spellings of the same file derive the SAME canonical identity.
    assert keys_link == keys_direct
    assert keys_link[0].path == "f.py"


def test_symlink_escaping_the_project_is_uncanonicalizable(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    outside_target = tmp_path / "outside.py"
    _seed_file(outside_target)
    link = root / "link.py"
    link.symlink_to(outside_target)
    keys: list[secret_scan.ApprovalKey] = []
    findings = scan_paths([link], project_root=root, keys=keys)
    # The hit still blocks — and its key carries no identity, so it can
    # never be suppressed.
    assert [finding.rule_id for finding in findings] == ["aws-access-key-id"]
    assert len(keys) == 1
    assert keys[0].path is None


def test_path_outside_the_project_root_is_uncanonicalizable(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    artifact = tmp_path / "elsewhere.py"
    _seed_file(artifact)
    keys: list[secret_scan.ApprovalKey] = []
    findings = scan_paths([artifact], project_root=root, keys=keys)
    assert [finding.rule_id for finding in findings] == ["aws-access-key-id"]
    assert len(keys) == 1
    assert keys[0].path is None


def test_project_root_reached_through_a_symlink_yields_the_same_identity(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-proj"
    (real_root / "sub").mkdir(parents=True)
    artifact = real_root / "sub" / "f.py"
    _seed_file(artifact)
    alias = tmp_path / "alias-proj"
    alias.symlink_to(real_root)
    keys_real: list[secret_scan.ApprovalKey] = []
    keys_alias: list[secret_scan.ApprovalKey] = []
    scan_paths([artifact], project_root=real_root, keys=keys_real)
    scan_paths([alias / "sub" / "f.py"], project_root=alias, keys=keys_alias)
    assert keys_real == keys_alias
    assert keys_alias[0].path == "sub/f.py"


def test_canonical_project_identity_normalizes_and_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    assert (
        secret_scan.canonical_project_identity(
            root / "a" / ".." / "b.py", project_root=root
        )
        == "b.py"
    )
    assert (
        secret_scan.canonical_project_identity(tmp_path / "other.py", project_root=root)
        is None
    )


def test_scan_incomplete_with_project_root_still_yields_no_key(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    artifact = root / "big.bin"
    artifact.write_bytes(b"x" * (DEFAULT_MAX_SCAN_BYTES + 1))
    keys: list[secret_scan.ApprovalKey] = []
    findings = scan_paths([artifact], project_root=root, keys=keys)
    assert [finding.rule_id for finding in findings] == ["scan-incomplete"]
    # An incomplete scan never yields a key: it can never be suppressed.
    assert keys == []


def test_walk_budget_sentinel_yields_no_key(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    tree = root / "tree"
    tree.mkdir(parents=True)
    for index in range(4):
        (tree / f"clean-{index}.txt").write_text("clean\n", encoding="utf-8")
    keys: list[secret_scan.ApprovalKey] = []
    findings = scan_paths([tree], project_root=root, max_walk_entries=2, keys=keys)
    assert [finding.rule_id for finding in findings] == ["scan-incomplete"]
    assert keys == []
