"""P2-c acceptance pins: recursive-`rm` git-status-aware operand analysis.

Decision 2's saved/unsaved split: recursive ``rm`` over a tree with unsaved
work resolves handoff (the shipped blanket denial — never autonomous, no
approval path); over a git-clean tree it resolves run-class only when every
expanded operand lies entirely within the repository whose clean status is
assessed, does not overlap that repository's Git metadata (neither
encompassing the Git directory nor lying inside it, including linked-worktree
common directories), and recovery status is established for every repository
contained in or overlapped by the operand (nested repositories included).
The analysis is fail-closed on every axis — any unestablishable aspect, and
any mixed operand, resolves the whole command to handoff.

The handoff pins are the plan's currently-true guard rails (each fails only
against a wrong implementation of this package); the run-class pins are the
discriminating acceptance and fail under the blanket denial.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_permission_resolver import (
    ShellPermissionResolver,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.permissions import PermissionContext


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run one git command in a fixture repository."""
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=check, capture_output=True, text=True
    )


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    git(path, "config", "user.email", "pins@example.com")
    git(path, "config", "user.name", "Pins")
    return path


def commit_all(path: Path, message: str) -> None:
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", message)


def make_clean_repo(path: Path) -> Path:
    """A git-clean repository with a tracked ``build/`` directory."""
    repo = init_repo(path)
    (repo / "build").mkdir()
    (repo / "build" / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    commit_all(repo, "tracked content")
    return repo


def make_unsaved_repo(path: Path) -> Path:
    """A repository whose tree carries unsaved work (a modified tracked file)."""
    repo = make_clean_repo(path)
    (repo / "tracked.txt").write_text("unsaved\n", encoding="utf-8")
    return repo


def make_rebase_repo(path: Path) -> Path:
    """A repository with a rebase in progress (recovery unestablishable)."""
    repo = init_repo(path)
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    commit_all(repo, "base")
    default_branch = git(repo, "branch", "--show-current").stdout.strip()
    git(repo, "checkout", "-q", "-b", "feature")
    (repo / "base.txt").write_text("feature\n", encoding="utf-8")
    commit_all(repo, "feature")
    git(repo, "checkout", "-q", default_branch)
    (repo / "base.txt").write_text("mainline\n", encoding="utf-8")
    commit_all(repo, "mainline")
    git(repo, "checkout", "-q", "feature")
    # The conflicting rebase leaves rebase-merge/ in progress.
    git(repo, "rebase", default_branch, check=False)
    assert (repo / ".git" / "rebase-merge").is_dir()
    return repo


def make_nested_repo(
    path: Path, *, remote: str | None, ignore: str = "vendor/\n"
) -> Path:
    """A clean repository ignoring a nested repository under ``vendor/``.

    The nested repository is clean; ``remote`` decides whether its recovery
    reads as established. ``"bare"`` configures a real bare remote and pushes
    the nested commits to it. ``"refs-only"`` authors a remote-tracking ref
    with ``git update-ref`` while no remote is configured — local metadata,
    not a remote copy of the commits, so recovery stays unestablished.
    ``None`` leaves the commits unpushed entirely. ``"dead-local"`` pushes to
    a local-path remote whose directory is then deleted (R7: a dead
    local-path remote holds no copy). ``"file-url"`` configures a ``file://``
    remote whose path exists. ``"https"`` configures a remote-scheme URL,
    which cannot be verified locally and counts as evidence (the gate's
    documented asymmetry). ``"plain-file"`` and ``"plain-dir"`` configure a
    local-path remote whose target exists on disk but is not a repository
    (round-3 F6a: the liveness check must verify a genuinely fetchable
    remote, not merely an existing path). ``"bundle"`` configures a local-path
    remote whose target is a git bundle file holding the commits — a
    genuinely fetchable remote git can read. ``"bundle-inside"``,
    ``"bundle-inside-relative"``, ``"bundle-sibling"`` and
    ``"file-url-repo-inside"`` configure a genuinely fetchable remote whose
    target lies INSIDE the deletion footprint (R10: recovery evidence the
    deletion itself destroys is not recovery evidence) — the first two as a
    bundle file beneath the operand (the second spelled relative to the
    nested repository), the third in a sibling ignored directory a second
    operand deletes, and the last as a ``file://`` URL to a bare repository
    beneath the operand that carries the commits and has an outside backup
    of its own. ``"file-url-pct-inside"`` and ``"file-url-pct-outside"``
    (R12) spell a ``file://`` remote with a percent-encoded path component
    — git percent-decodes ``file://`` URLs at connect time, so the encoded
    spelling names the DECODED path — with the decoded target inside the
    operand (the nested repository's own git dir) and outside it (a bare
    repository beside the outer repository, pushed through the encoded
    URL). ``ignore`` is the outer repository's ignore file.
    """
    outer = init_repo(path)
    (outer / "build").mkdir()
    (outer / "build" / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    (outer / ".gitignore").write_text(ignore, encoding="utf-8")
    commit_all(outer, "outer content")
    nested = init_repo(outer / "vendor" / "nested")
    (nested / "nested.txt").write_text("nested\n", encoding="utf-8")
    commit_all(nested, "nested content")
    if remote == "bare":
        # The bare remote lives beside the outer repository: inside the
        # operand it would be a second nested repository whose own recovery
        # the analysis would have to establish.
        bare = path.parent / f"{path.name}-origin.git"
        bare.mkdir(parents=True)
        git(bare, "init", "-q", "--bare")
        git(nested, "remote", "add", "origin", str(bare))
        git(nested, "push", "-q", "origin", "HEAD:refs/heads/main")
    elif remote == "refs-only":
        git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
    elif remote == "dead-local":
        dead = path.parent / f"{path.name}-dead-origin.git"
        dead.mkdir(parents=True)
        git(dead, "init", "-q", "--bare")
        git(nested, "remote", "add", "origin", str(dead))
        git(nested, "push", "-q", "origin", "HEAD:refs/heads/main")
        # The remote's local path no longer exists: the remote-tracking
        # ref survives, but no copy of the commits exists behind it.
        shutil.rmtree(dead)
    elif remote == "file-url":
        bare = path.parent / f"{path.name}-file-origin.git"
        bare.mkdir(parents=True)
        git(bare, "init", "-q", "--bare")
        git(nested, "remote", "add", "origin", f"file://{bare}")
        git(nested, "push", "-q", "origin", "HEAD:refs/heads/main")
    elif remote == "https":
        git(nested, "remote", "add", "origin", "https://example.invalid/repo.git")
        git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
    elif remote == "plain-file":
        # The remote's fetch URL is an existing REGULAR FILE: not a
        # repository, so no fetch could read a copy from it.
        target = path.parent / f"{path.name}-plain-file.txt"
        target.write_text("not a repository\n", encoding="utf-8")
        git(nested, "remote", "add", "origin", str(target))
        git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
    elif remote == "plain-dir":
        # The remote's fetch URL is an existing NON-REPOSITORY DIRECTORY.
        target = path.parent / f"{path.name}-plain-dir"
        target.mkdir(parents=True)
        (target / "inner.txt").write_text("not a repository\n", encoding="utf-8")
        git(nested, "remote", "add", "origin", str(target))
        git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
    elif remote == "bundle":
        # The remote's fetch URL is a git BUNDLE file holding the nested
        # commits: a genuinely fetchable remote (git fetch reads bundles).
        bundle = path.parent / f"{path.name}-origin.bundle"
        git(nested, "bundle", "create", str(bundle), "HEAD")
        git(nested, "remote", "add", "origin", str(bundle))
        git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
    elif remote == "bundle-inside":
        # The same genuinely fetchable bundle remote, but its file lives
        # INSIDE the operand: the deletion destroys the recovery copy
        # together with the repository.
        bundle = outer / "vendor" / "nested.bundle"
        git(nested, "bundle", "create", str(bundle), "HEAD")
        git(nested, "remote", "add", "origin", str(bundle))
        git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
    elif remote == "bundle-inside-relative":
        # The same inside-footprint bundle, spelled as a path relative to
        # the nested repository, so the URL resolution itself must land
        # inside the deletion footprint.
        bundle = outer / "vendor" / "nested.bundle"
        git(nested, "bundle", "create", str(bundle), "HEAD")
        git(nested, "remote", "add", "origin", "../nested.bundle")
        git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
    elif remote == "bundle-sibling":
        # The bundle lives in a sibling ignored directory: outside the
        # vendor operand, but inside the footprint of a command that
        # deletes both operands.
        backup = outer / "backup"
        backup.mkdir()
        bundle = backup / "nested.bundle"
        git(nested, "bundle", "create", str(bundle), "HEAD")
        git(nested, "remote", "add", "origin", str(bundle))
        git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
    elif remote == "file-url-repo-inside":
        # A file:// remote to a bare repository INSIDE the operand. The
        # bare repository holds the nested commits and carries its own
        # outside backup, so its own recovery is established; only the
        # nested repository's evidence location is at issue.
        bare = outer / "vendor" / "origin.git"
        bare.mkdir(parents=True)
        git(bare, "init", "-q", "--bare")
        git(nested, "remote", "add", "origin", f"file://{bare}")
        git(nested, "push", "-q", "origin", "HEAD:refs/heads/main")
        backup = path.parent / f"{path.name}-inside-backup.git"
        backup.mkdir(parents=True)
        git(backup, "init", "-q", "--bare")
        git(bare, "remote", "add", "backup", str(backup))
        git(bare, "push", "-q", "backup", "refs/heads/main:refs/heads/main")
    elif remote == "file-url-pct-inside":
        # R12: a file:// remote whose path carries a percent-encoded
        # component. git percent-decodes file:// URLs at connect time, so
        # the fetch target is the DECODED path — the nested repository's
        # own git dir, inside the operand the deletion destroys.
        git(nested, "remote", "add", "origin", f"file://{outer}/%76endor/nested/.git")
        git(nested, "update-ref", "refs/remotes/origin/main", "HEAD")
    elif remote == "file-url-pct-outside":
        # R12 control: the same encoded spelling whose decoded target lies
        # outside the footprint — a bare repository beside the outer
        # repository, pushed through the encoded URL (git fetches it).
        bare = path.parent / f"{path.name}-pct-origin.git"
        bare.mkdir(parents=True)
        git(bare, "init", "-q", "--bare")
        git(
            nested,
            "remote",
            "add",
            "origin",
            f"file://{path.parent}/%72epo-pct-origin.git",
        )
        git(nested, "push", "-q", "origin", "HEAD:refs/heads/main")
    return outer


def make_nested_bare_repo(path: Path, *, remote: str | None) -> Path:
    """A clean repository ignoring a BARE repository under ``vendor/``.

    The bare repository carries its Git metadata directly in its directory
    (a ``HEAD`` file plus ``objects/`` and ``refs/`` — git's own heuristic),
    so discovery finds it inside the operand like any nested repository. A
    seed repository pushes its commits into it. ``remote=None`` leaves the
    bare repository with no remote of its own, so its commits are unpushed;
    ``"local"`` configures a live local backup remote on the bare
    repository and pushes everything to it.
    """
    outer = init_repo(path)
    (outer / "build").mkdir()
    (outer / "build" / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    (outer / ".gitignore").write_text("vendor/\n", encoding="utf-8")
    commit_all(outer, "outer content")
    bare = outer / "vendor" / "origin.git"
    bare.mkdir(parents=True)
    git(bare, "init", "-q", "--bare")
    seed = init_repo(path.parent / f"{path.name}-seed")
    (seed / "seed.txt").write_text("seed\n", encoding="utf-8")
    commit_all(seed, "seed content")
    git(seed, "push", "-q", str(bare), "HEAD:refs/heads/main")
    if remote == "local":
        backup = path.parent / f"{path.name}-backup.git"
        backup.mkdir(parents=True)
        git(backup, "init", "-q", "--bare")
        git(bare, "remote", "add", "backup", str(backup))
        # A bare repository's own HEAD is unborn; push the seeded ref.
        git(bare, "push", "-q", "backup", "refs/heads/main:refs/heads/main")
    return outer


def make_linked_worktree(path: Path) -> tuple[Path, Path]:
    """A main repository plus a linked worktree; returns (main, worktree)."""
    main = init_repo(path)
    (main / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    commit_all(main, "tracked content")
    worktree = path.parent / f"{path.name}-wt"
    git(main, "worktree", "add", "-q", str(worktree), "-b", "linked")
    return main, worktree


def resolve(cwd: Path, command: str) -> PermissionContext | None:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)
    return tool.resolve_permission(BashArgs(command=command))


def assert_handoff(context: PermissionContext | None) -> None:
    """The command resolves handoff: the shipped blanket recursive-rm denial."""
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "recursive rm" in (context.reason or "")


def assert_run_class(context: PermissionContext | None) -> None:
    """The command resolves run-class under the full P2-c condition."""
    assert context is not None
    assert context.permission is ToolPermission.ALWAYS


class TestRecursiveRmPins:
    def test_unsaved_tree_resolves_handoff(self, tmp_path: Path) -> None:
        repo = make_unsaved_repo(tmp_path / "repo")
        assert_handoff(resolve(repo, "rm -rf build"))

    def test_clean_tree_resolves_run_class_under_the_full_condition(
        self, tmp_path: Path
    ) -> None:
        repo = make_clean_repo(tmp_path / "repo")
        assert_run_class(resolve(repo, "rm -rf build"))
        # Every expanded operand must satisfy the condition.
        assert_run_class(resolve(repo, "rm -rf build tracked.txt"))

    def test_operand_encompassing_the_git_dir_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        repo = make_clean_repo(tmp_path / "repo")
        assert_handoff(resolve(repo, "rm -rf ."))

    def test_git_metadata_subpath_resolves_handoff(self, tmp_path: Path) -> None:
        repo = make_clean_repo(tmp_path / "repo")
        assert_handoff(resolve(repo, "rm -rf .git/refs"))

    def test_linked_worktree_common_dir_subpath_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        main, worktree = make_linked_worktree(tmp_path / "repo")
        assert_handoff(resolve(worktree, "rm -rf ../repo/.git/refs"))
        assert_handoff(resolve(worktree, f"rm -rf {main / '.git' / 'refs'}"))
        # The work-tree .git marker is metadata too: encompassing it is handoff.
        assert_handoff(resolve(worktree, "rm -rf ."))

    def test_operand_outside_the_assessed_repository_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        repo = make_clean_repo(tmp_path / "repo")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "file.txt").write_text("outside\n", encoding="utf-8")
        assert_handoff(resolve(repo, "rm -rf ../outside"))
        assert_handoff(resolve(repo, f"rm -rf {outside}"))
        # Partially outside: an ancestor of the assessed repository.
        assert_handoff(resolve(repo, "rm -rf .."))

    def test_rebase_in_progress_resolves_handoff(self, tmp_path: Path) -> None:
        repo = make_rebase_repo(tmp_path / "repo")
        assert_handoff(resolve(repo, "rm -rf build"))

    def test_mixed_operands_resolve_the_whole_command_to_handoff(
        self, tmp_path: Path
    ) -> None:
        repo = make_clean_repo(tmp_path / "repo")
        assert_handoff(resolve(repo, "rm -rf build .git/refs"))
        assert_handoff(resolve(repo, "rm -rf build .."))

    def test_nested_repository_with_unestablished_recovery_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        repo = make_nested_repo(tmp_path / "repo", remote=None)
        assert_handoff(resolve(repo, "rm -rf vendor"))

    def test_nested_repository_with_refs_but_no_remote_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # A remote-tracking ref is local metadata: authored with update-ref
        # and no configured remote, no remote copy of the commits exists, so
        # recovery stays unestablished and the operand resolves handoff.
        repo = make_nested_repo(tmp_path / "repo", remote="refs-only")
        assert_handoff(resolve(repo, "rm -rf vendor"))

    def test_nested_repository_with_established_recovery_resolves_run_class(
        self, tmp_path: Path
    ) -> None:
        repo = make_nested_repo(tmp_path / "repo", remote="bare")
        assert_run_class(resolve(repo, "rm -rf vendor"))

    def test_nested_repository_with_dead_local_remote_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # R7: the configured-remote gate inspects the remote's fetch URL,
        # not just its name — a remote whose local path no longer exists
        # leaves the stale remote-tracking ref reading as
        # recovery-established with no copy behind it.
        repo = make_nested_repo(tmp_path / "repo", remote="dead-local")
        assert_handoff(resolve(repo, "rm -rf vendor"))

    def test_nested_repository_with_file_url_remote_resolves_run_class(
        self, tmp_path: Path
    ) -> None:
        # R7: a file:// fetch URL is local-scheme and counts as evidence
        # when its path exists on disk.
        repo = make_nested_repo(tmp_path / "repo", remote="file-url")
        assert_run_class(resolve(repo, "rm -rf vendor"))

    def test_nested_repository_with_remote_scheme_url_resolves_run_class(
        self, tmp_path: Path
    ) -> None:
        # R7: a remote-scheme fetch URL cannot be verified locally and
        # counts as evidence exactly as a name-only check counted it — the
        # gate's documented asymmetry.
        repo = make_nested_repo(tmp_path / "repo", remote="https")
        assert_run_class(resolve(repo, "rm -rf vendor"))

    def test_nested_repository_with_plain_file_remote_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # Round-3 F6a: the liveness check verifies a genuinely fetchable
        # remote, not merely an existing path — a remote whose local path is
        # a plain file is not a repository any fetch could read, so the
        # stale remote-tracking ref reads as recovery-unestablished.
        repo = make_nested_repo(tmp_path / "repo", remote="plain-file")
        assert_handoff(resolve(repo, "rm -rf vendor"))

    def test_nested_repository_with_plain_directory_remote_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # Round-3 F6a: same for a local path that exists as a directory but
        # is not a repository.
        repo = make_nested_repo(tmp_path / "repo", remote="plain-dir")
        assert_handoff(resolve(repo, "rm -rf vendor"))

    def test_nested_repository_with_bundle_remote_resolves_run_class(
        self, tmp_path: Path
    ) -> None:
        # Round-3 F6a control: a git bundle file IS a genuinely fetchable
        # remote (git fetch reads bundles), so the remote gate counts it as
        # evidence exactly like a live repository.
        repo = make_nested_repo(tmp_path / "repo", remote="bundle")
        assert_run_class(resolve(repo, "rm -rf vendor"))

    def test_nested_repository_with_bundle_inside_the_operand_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # R10: recovery evidence the deletion itself destroys is not
        # recovery evidence. The bundle remote is genuinely fetchable and
        # holds the commits, but its file lives inside the operand, so the
        # deletion destroys the copy together with the repository and no
        # surviving evidence remains.
        repo = make_nested_repo(tmp_path / "repo", remote="bundle-inside")
        assert_handoff(resolve(repo, "rm -rf vendor"))

    def test_nested_repository_with_relative_bundle_inside_the_operand_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # R10: the same inside-footprint bundle spelled as a path relative
        # to the nested repository — the remote URL's resolved location
        # lands inside the deletion footprint exactly like an absolute one.
        repo = make_nested_repo(tmp_path / "repo", remote="bundle-inside-relative")
        assert_handoff(resolve(repo, "rm -rf vendor"))

    def test_nested_repository_with_file_url_remote_inside_the_operand_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # R10: the file:// spelling resolves to a filesystem path exactly
        # like a plain one, so a fetchable file:// remote whose target lies
        # inside the operand holds no surviving copy either. The bare
        # repository itself carries an outside backup, so its own recovery
        # is established — only the nested repository's evidence location
        # gates the command.
        repo = make_nested_repo(tmp_path / "repo", remote="file-url-repo-inside")
        assert_handoff(resolve(repo, "rm -rf vendor"))

    def test_recovery_evidence_in_another_operand_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # R10: the survival requirement covers the ENTIRE expanded deletion
        # footprint, not just the operand holding the at-risk repository.
        # The bundle lives in a sibling directory the same command deletes
        # as a second operand.
        repo = make_nested_repo(
            tmp_path / "repo", remote="bundle-sibling", ignore="vendor/\nbackup/\n"
        )
        assert_handoff(resolve(repo, "rm -rf vendor backup"))
        # Control: deleting only the repository's operand leaves the
        # evidence outside the footprint, so recovery stays established.
        assert_run_class(resolve(repo, "rm -rf vendor"))

    def test_ignored_bare_repository_with_unpushed_commits_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # R7: a bare repository carries its Git metadata directly in its
        # directory (a HEAD file plus objects/ and refs/), so discovery
        # finds it inside the operand and its recovery gates the command:
        # unpushed commits with no remote to hold a copy resolve handoff.
        repo = make_nested_bare_repo(tmp_path / "repo", remote=None)
        assert_handoff(resolve(repo, "rm -rf vendor"))
        assert_handoff(resolve(repo, "rm -rf vendor/origin.git"))

    def test_ignored_bare_repository_pushed_to_a_real_remote_resolves_run_class(
        self, tmp_path: Path
    ) -> None:
        # R7: the same bare repository with a configured remote whose
        # local path exists and no unpushed commit is eligible under the
        # rest of the condition.
        repo = make_nested_bare_repo(tmp_path / "repo", remote="local")
        assert_run_class(resolve(repo, "rm -rf vendor"))

    def test_directory_without_a_repository_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # The resolution axis fails closed: no repository's clean status can
        # be assessed, so the blanket denial remains.
        plain = tmp_path / "plain"
        plain.mkdir()
        assert_handoff(resolve(plain, "rm -rf subdir"))

    def test_recovery_evidence_a_second_segment_deletes_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # R12 (deep-review round 5): recovery-evidence survival spans the
        # WHOLE call, not one segment. The bundle lives in a sibling
        # ignored directory a SECOND segment of the same call deletes, so
        # the union of the segments' deletion footprints covers the evidence
        # exactly like the combined command's single footprint.
        repo = make_nested_repo(
            tmp_path / "repo", remote="bundle-sibling", ignore="vendor/\nbackup/\n"
        )
        assert_handoff(resolve(repo, "rm -rf vendor; rm -rf backup"))
        assert_handoff(resolve(repo, "rm -rf vendor && rm -rf backup"))
        assert_handoff(resolve(repo, "rm -rf vendor || rm -rf backup"))
        assert_handoff(resolve(repo, "rm -rf backup; rm -rf vendor"))
        # Control: the combined single command still denies (R10 unchanged).
        assert_handoff(resolve(repo, "rm -rf vendor backup"))
        # Control: a second segment that deletes nothing the evidence needs
        # leaves the union footprint clear of it — the call stays run-class.
        assert_run_class(resolve(repo, "rm -rf vendor; rm -rf build"))
        assert_run_class(resolve(repo, "rm -rf vendor; rm -rf missing"))

    def test_write_before_rm_in_the_same_call_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # R12 (deep-review round 5): a write an EARLIER statement of the
        # same call performs inside a later recursive-rm segment's
        # eligibility scope fails the segment closed — the write dirties
        # the clean-tree state the eligibility relies on, or writes content
        # the deletion destroys. The bundle beside the outer repository
        # keeps the rm segment's baseline run-class.
        repo = make_nested_repo(tmp_path / "repo", remote="bundle")
        # The hazard shape: the write dirties the NESTED repository's
        # tracked file; the rm then destroys the unrecoverable modification.
        assert_handoff(
            resolve(repo, "echo x > vendor/nested/nested.txt; rm -rf vendor")
        )
        # A write anywhere in the assessed repository's work tree fails the
        # segment closed: its clean-tree state is the eligibility's premise.
        assert_handoff(resolve(repo, "echo x > build/artifact.txt; rm -rf vendor"))
        assert_handoff(resolve(repo, "echo x > newfile.txt; rm -rf vendor"))
        # A mutating command's destination write counts the same way.
        assert_handoff(
            resolve(
                repo, "cp build/artifact.txt vendor/nested/nested.txt; rm -rf vendor"
            )
        )
        # Control: a write AFTER the rm segment is not an earlier
        # statement's write — the rm stays run-class.
        assert_run_class(resolve(repo, "rm -rf vendor; echo x > after.txt"))
        # Control: a read-only earlier statement has no write surface.
        assert_run_class(resolve(repo, "git status; rm -rf vendor"))
        # Control: the rm segment's OWN redirect is part of its statement,
        # not an earlier segment's write.
        assert_run_class(resolve(repo, "rm -rf vendor > log.txt"))

    def test_percent_encoded_file_url_remote_inside_the_operand_resolves_handoff(
        self, tmp_path: Path
    ) -> None:
        # R12 (deep-review round 5): git percent-decodes file:// URLs at
        # connect time, so an encoded component names the DECODED path. The
        # survival containment must decode exactly like git or it sees the
        # evidence outside the footprint while git fetches from inside it.
        repo = make_nested_repo(tmp_path / "repo", remote="file-url-pct-inside")
        assert_handoff(resolve(repo, "rm -rf vendor"))

    def test_percent_encoded_file_url_remote_outside_the_operand_resolves_run_class(
        self, tmp_path: Path
    ) -> None:
        # R12 control: the same encoded spelling whose decoded target lies
        # outside the footprint keeps the evidence standing — the decoding
        # does not over-deny.
        repo = make_nested_repo(tmp_path / "repo", remote="file-url-pct-outside")
        assert_run_class(resolve(repo, "rm -rf vendor"))

    @pytest.mark.parametrize(
        "mutation",
        [
            "rm backup/nested.bundle",
            "rm -- backup/nested.bundle",
            "rm backup/*.bundle",
            "mv backup/nested.bundle elsewhere/",
            "mv backup elsewhere/",
            "mv -t elsewhere backup/nested.bundle",
            "mv --target-directory=elsewhere backup/*.bundle",
            "mv backup/nested.bundle elsewhere/ --suffix=.old",
            "printf '' > backup/nested.bundle",
            "cp build/artifact.txt backup/nested.bundle",
            "mv build/artifact.txt backup/nested.bundle",
            "install build/artifact.txt backup/nested.bundle",
            "dd if=build/artifact.txt of=backup/nested.bundle",
            "objcopy build/artifact.txt backup/nested.bundle",
            "sed -n 'w backup/nested.bundle' build/artifact.txt",
            "zip backup/nested.bundle build/artifact.txt",
            "7z a backup/nested.bundle build/artifact.txt",
            "unzip archive.zip -d backup",
            "patch -d backup -i changes.diff",
            "git apply --directory=backup changes.diff",
        ],
    )
    def test_same_call_evidence_mutation_resolves_handoff(
        self, tmp_path: Path, mutation: str
    ) -> None:
        repo = make_nested_repo(
            tmp_path / "repo", remote="bundle-sibling", ignore="vendor/\nbackup/\n"
        )
        assert_run_class(resolve(repo, "rm -rf vendor"))
        assert_handoff(resolve(repo, f"rm -rf vendor; {mutation}"))

    @pytest.mark.parametrize(
        "command",
        [
            "rm backup/nested.bundle; rm -rf vendor",
            "rm -rf vendor && rm backup/nested.bundle",
            "rm -rf vendor || mv backup/nested.bundle elsewhere/",
            "rm -rf vendor\nrm backup/nested.bundle",
            "rm -rf vendor; command rm backup/nested.bundle",
            "rm -rf vendor; bash -c 'mv backup/nested.bundle elsewhere/'",
            "rm -rf vendor > backup/nested.bundle",
        ],
    )
    def test_evidence_mutation_order_wrappers_and_redirects(
        self, tmp_path: Path, command: str
    ) -> None:
        repo = make_nested_repo(
            tmp_path / "repo", remote="bundle-sibling", ignore="vendor/\nbackup/\n"
        )
        assert_handoff(resolve(repo, command))

    @pytest.mark.parametrize(
        "mutation",
        [
            "rm backup/nested.bundle",
            "mv backup/nested.bundle elsewhere/",
            "printf '' > backup/nested.bundle",
            "rm -rf build",
            "echo x > after.txt",
        ],
    )
    def test_evidence_outside_all_mutations_survives(
        self, tmp_path: Path, mutation: str
    ) -> None:
        repo = make_nested_repo(tmp_path / "repo", remote="bundle")
        assert_run_class(resolve(repo, f"rm -rf vendor; {mutation}"))

    @pytest.mark.parametrize("prefix", ["", "file://"])
    def test_symlink_parent_remote_resolution(
        self, tmp_path: Path, prefix: str
    ) -> None:
        repo = make_nested_repo(
            tmp_path / "repo", remote="bundle", ignore="vendor/\nalias\n"
        )
        nested = repo / "vendor" / "nested"
        (repo / "vendor" / "deep").mkdir()
        (repo / "alias").symlink_to("vendor/deep", target_is_directory=True)
        url = f"{prefix}{repo}/alias/../nested/.git"
        assert git(nested, "ls-remote", url).returncode == 0
        git(nested, "remote", "set-url", "origin", url)
        assert_handoff(resolve(repo, "rm -rf vendor"))
        git(nested, "remote", "set-url", "origin", f"{prefix}{nested}/.git")
        assert_handoff(resolve(repo, "rm -rf vendor"))
        outside = tmp_path / "outside.git"
        git(nested, "clone", "-q", "--bare", str(nested), str(outside))
        git(nested, "remote", "set-url", "origin", f"{prefix}{outside}")
        assert_run_class(resolve(repo, "rm -rf vendor"))

    def test_unknown_call_destruction_fails_closed(self, tmp_path: Path) -> None:
        repo = make_nested_repo(tmp_path / "repo", remote="bundle")
        with patch.object(
            ShellPermissionResolver, "_segment_write_paths", return_value=None
        ):
            assert_handoff(resolve(repo, "rm -rf vendor; echo x > after.txt"))


@pytest.mark.parametrize("command", ["command unset PATH", "/usr/bin/unset PATH"])
def test_protected_unset_guard_is_reachable(tmp_path: Path, command: str) -> None:
    context = resolve(tmp_path, command)
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert (
        context.reason
        == "Command denied: unsetting a protected shell environment variable"
    )
