from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

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
