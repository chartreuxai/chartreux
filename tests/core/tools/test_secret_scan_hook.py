from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from chartreux.core.tools import secret_scan
from chartreux.core.tools.secret_scan import DEFAULT_MAX_SCAN_BYTES

# Seeded secrets are assembled by concatenation (the same pattern the
# submission-boundary fixtures use) so this test file itself stays clean
# under the secret scan.
AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A real repository: the hook scans the staged blobs in its index."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    return tmp_path


def run_hook(
    *filenames: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "chartreux.core.tools.secret_scan_hook", *filenames],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(cwd) if cwd is not None else None,
    )


def stage(repo: Path, name: str, content: str) -> None:
    (repo / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "--", name], cwd=str(repo), check=True)


def commit(repo: Path, message: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=pins@example.com",
            "-c",
            "user.name=Pins",
            "commit",
            "-q",
            "-m",
            message,
        ],
        cwd=str(repo),
        check=True,
    )


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True)


def test_hook_blocks_a_seeded_aws_key(git_repo: Path) -> None:
    stage(git_repo, "aws_config.txt", f"[default]\naws_access_key_id = {AWS_KEY}\n")
    result = run_hook("aws_config.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "aws-access-key-id" in result.stdout
    assert "aws_config.txt:2:" in result.stdout
    assert "Commit blocked" in result.stdout
    assert AWS_KEY not in result.stdout
    assert AWS_KEY not in result.stderr


def test_hook_blocks_a_seeded_github_token(git_repo: Path) -> None:
    stage(git_repo, "deploy.sh", f"GITHUB_TOKEN={GITHUB_TOKEN}\n")
    result = run_hook("deploy.sh", cwd=git_repo)
    assert result.returncode == 1
    assert "github-token" in result.stdout
    assert GITHUB_TOKEN not in result.stdout
    assert GITHUB_TOKEN not in result.stderr


def test_hook_output_never_contains_the_matched_value(git_repo: Path) -> None:
    stage(
        git_repo,
        "config.txt",
        f"# config\naws_access_key_id = {AWS_KEY}\n"
        f"export GITHUB_TOKEN={GITHUB_TOKEN}\n",
    )
    result = run_hook("config.txt", cwd=git_repo)
    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert AWS_KEY not in output
    assert GITHUB_TOKEN not in output
    assert "AKIA" not in output
    assert "ghp_" not in output


def test_hook_passes_a_clean_file(git_repo: Path) -> None:
    stage(
        git_repo,
        "clean.py",
        "def add(left: int, right: int) -> int:\n    return left + right\n",
    )
    result = run_hook("clean.py", cwd=git_repo)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_hook_without_filenames_passes() -> None:
    result = run_hook()
    assert result.returncode == 0
    assert result.stdout == ""


def test_hook_blocks_an_artifact_the_scan_cannot_finish(git_repo: Path) -> None:
    # Fail-closed: an artifact larger than the per-file scan cap blocks the
    # commit with the incomplete reason. The seeded key sits past the cap, so
    # it is neither reported nor disclosed.
    stage(git_repo, "big.txt", "x" * DEFAULT_MAX_SCAN_BYTES + AWS_KEY + "\n")
    result = run_hook("big.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "scan-incomplete" in result.stdout
    assert "big.txt" in result.stdout
    assert "Commit blocked" in result.stdout
    assert AWS_KEY not in result.stdout
    assert AWS_KEY not in result.stderr


def test_hook_blocks_a_secret_only_in_the_staged_blob(git_repo: Path) -> None:
    # The hook scans the STAGED content, not the working tree: stage a
    # secret, revert the working-tree file to clean content, and the commit
    # is still blocked by the staged secret that would ship.
    stage(git_repo, "laundered.txt", f"aws_access_key_id = {AWS_KEY}\n")
    (git_repo / "laundered.txt").write_text("clean = True\n", encoding="utf-8")
    result = run_hook("laundered.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "aws-access-key-id" in result.stdout
    assert "laundered.txt:1:" in result.stdout
    assert "Commit blocked" in result.stdout
    assert AWS_KEY not in result.stdout
    assert AWS_KEY not in result.stderr


def test_hook_blocks_a_secret_staged_then_deleted_from_the_work_tree(
    git_repo: Path,
) -> None:
    # A staged blob survives deletion of the working-tree file: the hook reads
    # the index, so the secret still blocks even with no file on disk to scan.
    stage(git_repo, "gone.txt", f"GITHUB_TOKEN={GITHUB_TOKEN}\n")
    (git_repo / "gone.txt").unlink()
    result = run_hook("gone.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "github-token" in result.stdout
    assert GITHUB_TOKEN not in result.stdout
    assert GITHUB_TOKEN not in result.stderr


def test_hook_fails_closed_when_the_index_lookup_fails(git_repo: Path) -> None:
    # Fail-closed: a path with no staged blob is a scan-incomplete sentinel,
    # never a silent working-tree fallback — the clean file on disk does not
    # launder it.
    (git_repo / "unstaged.txt").write_text("clean = True\n", encoding="utf-8")
    result = run_hook("unstaged.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "scan-incomplete" in result.stdout
    assert "unstaged.txt" in result.stdout
    assert "Commit blocked" in result.stdout


def test_hook_passes_a_staged_deletion(git_repo: Path) -> None:
    # A staged deletion is not a lookup failure: the path is tracked in HEAD
    # and absent from the index, so the commit ships no content for it and
    # there is nothing to scan.
    stage(git_repo, "victim.txt", "nothing sensitive\n")
    commit(git_repo, "add victim")
    git(git_repo, "rm", "-q", "victim.txt")
    result = run_hook("victim.txt", cwd=git_repo)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_hook_passes_a_staged_deletion_of_a_secret_bearing_file(git_repo: Path) -> None:
    # Removing a committed secret is a legitimate commit: the deletion ships
    # no content for the path, so it does not block — and the hook's output
    # still never carries the secret.
    stage(git_repo, "secret.txt", f"GITHUB_TOKEN={GITHUB_TOKEN}\n")
    commit(git_repo, "add secret")
    git(git_repo, "rm", "-q", "secret.txt")
    result = run_hook("secret.txt", cwd=git_repo)
    assert result.returncode == 0
    assert GITHUB_TOKEN not in result.stdout
    assert GITHUB_TOKEN not in result.stderr


def test_hook_scans_the_new_path_of_a_rename(git_repo: Path) -> None:
    # git mv stages a deletion of the old path and the content under the new
    # one: the old path passes (nothing ships for it), the new path is
    # scanned and still blocks on the secret.
    stage(git_repo, "old.txt", f"GITHUB_TOKEN={GITHUB_TOKEN}\n")
    commit(git_repo, "add old")
    git(git_repo, "mv", "old.txt", "new.txt")
    old = run_hook("old.txt", cwd=git_repo)
    assert old.returncode == 0
    new = run_hook("new.txt", cwd=git_repo)
    assert new.returncode == 1
    assert "github-token" in new.stdout
    assert GITHUB_TOKEN not in new.stdout
    assert GITHUB_TOKEN not in new.stderr


def test_hook_fails_closed_when_nothing_is_staged_for_the_path(git_repo: Path) -> None:
    # Staged, deleted from the working tree, and the deletion staged too:
    # no index entry and no HEAD entry — not a staged deletion, so the
    # genuine lookup failure still blocks the commit.
    stage(git_repo, "both.txt", "clean = True\n")
    (git_repo / "both.txt").unlink()
    git(git_repo, "add", "-A")
    result = run_hook("both.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "scan-incomplete" in result.stdout
    assert "both.txt" in result.stdout
    assert "Commit blocked" in result.stdout


# --- A2: the committed-HEAD approved-match manifest --------------------------
#
# Both gates consume one repo-committed manifest (.chartreux-secret-scan.json),
# which the hook reads from the COMMITTED HEAD blob — never the index, never
# the working tree — so a single commit staging both a secret and its
# approval does not self-authorize. Seeded secrets stay concatenation-built
# so this file itself scans clean.

OTHER_TOKEN = "ghp_" + "z9Y8x7W6v5U4t3S2r1Q0p9O8n7M6l5K4j3H2"
SWAPPED_TOKEN = "ghp_" + "b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9"
SECRET_LINE = f"GITHUB_TOKEN = {GITHUB_TOKEN}"


def _digest_of(line: str, rule_id: str) -> str:
    keys: list[secret_scan.ApprovalKey] = []
    secret_scan.scan_text(line + "\n", label="fixture.txt", keys=keys)
    return next(key.digest for key in keys if key.rule_id == rule_id)


def _approval(path: str, line: str, rule_id: str) -> dict[str, str]:
    return {"path": path, "rule_id": rule_id, "digest": _digest_of(line, rule_id)}


def _manifest_document(entries: list[dict[str, str]]) -> str:
    return (
        json.dumps(
            {
                "schema_version": secret_scan.SCAN_POLICY_SCHEMA_VERSION,
                "approved_matches": entries,
                "residual_exclusions": [],
            },
            indent=2,
        )
        + "\n"
    )


def _commit_manifest(repo: Path, entries: list[dict[str, str]]) -> None:
    (repo / secret_scan.MANIFEST_FILENAME).write_text(
        _manifest_document(entries), encoding="utf-8"
    )
    git(repo, "add", "--", secret_scan.MANIFEST_FILENAME)
    commit(repo, "approve synthetic matches")


def _both_approvals(path: str, line: str) -> list[dict[str, str]]:
    # The seeded assignment line hits BOTH github-token and the entropy-gated
    # generic rule; both entries are needed for it to clear.
    return [
        _approval(path, line, "github-token"),
        _approval(path, line, "generic-keyword-assignment"),
    ]


def test_hook_passes_an_approved_match_from_the_committed_head_manifest(
    git_repo: Path,
) -> None:
    _commit_manifest(git_repo, _both_approvals("fixture.txt", SECRET_LINE))
    stage(git_repo, "fixture.txt", SECRET_LINE + "\n")
    result = run_hook("fixture.txt", cwd=git_repo)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert GITHUB_TOKEN not in result.stdout + result.stderr


def test_hook_multi_rule_hit_needs_both_rule_approvals(git_repo: Path) -> None:
    # Only the github-token entry is committed: the same line's
    # generic-keyword-assignment hit is a DIFFERENT (path, rule) pair and
    # still blocks.
    _commit_manifest(git_repo, [_approval("fixture.txt", SECRET_LINE, "github-token")])
    stage(git_repo, "fixture.txt", SECRET_LINE + "\n")
    result = run_hook("fixture.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "generic-keyword-assignment" in result.stdout
    assert "Commit blocked" in result.stdout
    assert GITHUB_TOKEN not in result.stdout + result.stderr


def test_hook_blocks_a_different_credential_on_the_same_line(git_repo: Path) -> None:
    # The approved token sits on the same line as a second, different token:
    # the second token's digest is not approved, so it blocks.
    _commit_manifest(git_repo, _both_approvals("fixture.txt", SECRET_LINE))
    stage(git_repo, "fixture.txt", f"tokens = {GITHUB_TOKEN} and {OTHER_TOKEN}\n")
    result = run_hook("fixture.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "github-token" in result.stdout
    assert "Commit blocked" in result.stdout
    assert GITHUB_TOKEN not in result.stdout + result.stderr
    assert OTHER_TOKEN not in result.stdout + result.stderr


def test_hook_blocks_an_identical_match_at_another_path(git_repo: Path) -> None:
    # Approval is keyed by canonical path: the identical secret at a
    # different path is a different key and blocks.
    _commit_manifest(git_repo, _both_approvals("approved.txt", SECRET_LINE))
    stage(git_repo, "other.txt", SECRET_LINE + "\n")
    result = run_hook("other.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "github-token" in result.stdout


def test_hook_reads_the_manifest_from_committed_head_never_index_or_worktree(
    git_repo: Path,
) -> None:
    # A single commit staging BOTH the secret and its matching approval:
    # HEAD carries no manifest yet, so the hook blocks — no self-authorization.
    stage(git_repo, "fixture.txt", SECRET_LINE + "\n")
    (git_repo / secret_scan.MANIFEST_FILENAME).write_text(
        _manifest_document(_both_approvals("fixture.txt", SECRET_LINE)),
        encoding="utf-8",
    )
    git(git_repo, "add", "--", secret_scan.MANIFEST_FILENAME)
    result = run_hook("fixture.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "github-token" in result.stdout
    assert "Commit blocked" in result.stdout

    # A working-tree-only manifest edit after a committed (empty) manifest
    # does not add approvals either: the hook still reads HEAD.
    _commit_manifest(git_repo, [])
    (git_repo / secret_scan.MANIFEST_FILENAME).write_text(
        _manifest_document(_both_approvals("fixture.txt", SECRET_LINE)),
        encoding="utf-8",
    )
    result = run_hook("fixture.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "github-token" in result.stdout


def test_hook_malformed_manifest_fails_closed(git_repo: Path) -> None:
    # A malformed committed manifest provides no approvals: the seeded
    # secret still blocks, and a clean file still passes.
    for bad in (
        "not json at all\n",
        json.dumps({
            "schema_version": 2,
            "approved_matches": [],
            "residual_exclusions": [],
        })
        + "\n",
        json.dumps({
            "schema_version": 1,
            "approved_matches": [
                {"path": "fixture.txt", "rule_id": "no-such-rule", "digest": "0" * 64}
            ],
            "residual_exclusions": [],
        })
        + "\n",
    ):
        (git_repo / secret_scan.MANIFEST_FILENAME).write_text(bad, encoding="utf-8")
        git(git_repo, "add", "--", secret_scan.MANIFEST_FILENAME)
        commit(git_repo, "bad manifest")
        stage(git_repo, "fixture.txt", SECRET_LINE + "\n")
        result = run_hook("fixture.txt", cwd=git_repo)
        assert result.returncode == 1
        assert "github-token" in result.stdout
        assert GITHUB_TOKEN not in result.stdout + result.stderr
        git(git_repo, "rm", "-q", "--cached", "fixture.txt")


def test_hook_fresh_clone_without_a_manifest(git_repo: Path) -> None:
    # No manifest anywhere in HEAD: a clean file passes, a secret blocks —
    # fresh-clone commits of non-fixture paths work with the policy absent.
    stage(git_repo, "clean.txt", "value = 1\n")
    assert run_hook("clean.txt", cwd=git_repo).returncode == 0
    stage(git_repo, "fixture.txt", SECRET_LINE + "\n")
    result = run_hook("fixture.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "github-token" in result.stdout


def test_hook_value_swap_two_commit_workflow(git_repo: Path) -> None:
    # The documented value-update churn sequence: the new value's approval
    # commit precedes the fixture swap, and the old entry is removed in the
    # same reviewed change — so the swapped fixture clears and the old value
    # blocks.
    old_line = SECRET_LINE
    new_line = f"GITHUB_TOKEN = {SWAPPED_TOKEN}"
    _commit_manifest(git_repo, _both_approvals("fixture.txt", old_line))
    stage(git_repo, "fixture.txt", old_line + "\n")
    assert run_hook("fixture.txt", cwd=git_repo).returncode == 0

    # One reviewed commit: manifest now approves ONLY the new value, and the
    # fixture is swapped to it in the same commit.
    _commit_manifest(git_repo, _both_approvals("fixture.txt", new_line))
    stage(git_repo, "fixture.txt", new_line + "\n")
    result = run_hook("fixture.txt", cwd=git_repo)
    assert result.returncode == 0

    # The swapped-out value's entry is gone, so the old value now blocks.
    stage(git_repo, "fixture.txt", old_line + "\n")
    result = run_hook("fixture.txt", cwd=git_repo)
    assert result.returncode == 1
    assert "github-token" in result.stdout
    assert GITHUB_TOKEN not in result.stdout + result.stderr


def test_real_pre_commit_run_drives_the_hook_in_a_temp_git_repo(tmp_path: Path) -> None:
    # The real invocation node: `pre-commit run` against the hook in a
    # temporary git repository (a local system hook running the same module
    # the repo's .pre-commit-config.yaml entry runs).
    pre_commit = pytest.importorskip("pre_commit")
    del pre_commit  # only the import is needed; the module is driven by -m
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    git(repo, "config", "user.email", "pins@example.com")
    git(repo, "config", "user.name", "Pins")
    chartreux_root = Path(secret_scan.__file__).resolve().parents[3]
    (repo / ".pre-commit-config.yaml").write_text(
        "repos:\n"
        "  - repo: local\n"
        "    hooks:\n"
        "      - id: secret-scan\n"
        "        name: Secret scan\n"
        f"        entry: {sys.executable} -m chartreux.core.tools.secret_scan_hook\n"
        "        language: system\n"
        "        pass_filenames: true\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("value = 1\n", encoding="utf-8")
    git(repo, "add", "-A")
    commit(repo, "init")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(chartreux_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PRE_COMMIT_HOME"] = str(tmp_path / "pre-commit-home")

    def pre_commit_run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "pre_commit", "run", *args],
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    # A staged secret with no manifest in HEAD: the real run blocks the commit.
    stage(repo, "fixture.txt", SECRET_LINE + "\n")
    blocked = pre_commit_run("secret-scan")
    assert blocked.returncode == 1
    assert "github-token" in blocked.stdout
    assert GITHUB_TOKEN not in blocked.stdout + blocked.stderr

    # Approval-first: commit the manifest, then stage the fixture — passes.
    git(repo, "restore", "--staged", "fixture.txt")
    _commit_manifest(repo, _both_approvals("fixture.txt", SECRET_LINE))
    stage(repo, "fixture.txt", SECRET_LINE + "\n")
    passed = pre_commit_run("secret-scan")
    assert passed.returncode == 0
    assert "Passed" in passed.stdout

    # A single commit staging both a NEW secret and its approval: HEAD does
    # not carry the approval yet, so the real run still blocks.
    stage(repo, "other.txt", f"GITHUB_TOKEN = {OTHER_TOKEN}\n")
    (repo / secret_scan.MANIFEST_FILENAME).write_text(
        _manifest_document(
            _both_approvals("fixture.txt", SECRET_LINE)
            + _both_approvals("other.txt", f"GITHUB_TOKEN = {OTHER_TOKEN}")
        ),
        encoding="utf-8",
    )
    git(repo, "add", "-A")
    blocked_together = pre_commit_run("secret-scan")
    assert blocked_together.returncode == 1
    assert "other.txt" in blocked_together.stdout
    assert OTHER_TOKEN not in blocked_together.stdout + blocked_together.stderr
