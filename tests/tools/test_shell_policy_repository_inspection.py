"""Repository scope resolution and config-risk tests for the shell-command policy suite (git readers inspecting the repository, repository config risk, pager config vectors, the repository alias, -C cwd relocations)."""

from __future__ import annotations

from pathlib import Path
import shlex

import pytest

from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    analyze_shell_command_policy,
    git_cwd_relocations,
    git_repository_config_risk,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs
from tests.tools._shell_policy_test_support import bash_tool as bash_tool


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "diff"],
        ["git", "log"],
        ["git", "show"],
        ["git", "status"],
        ["git", "blame"],
        ["git", "whatchanged"],
    ],
)
def test_git_readers_inspect_the_repository(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).inspect_git_repository


@pytest.mark.parametrize(
    "tokens", [["git", "commit", "-m", "x"], ["git", "init"], ["git"], ["cat", "file"]]
)
def test_non_readers_do_not_inspect_the_repository(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).inspect_git_repository


def _repository_with_config(root: Path, config: str) -> Path:
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(config)
    return root


def test_git_repository_config_risk_names_the_active_vector(tmp_path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n\tpager = ./pager\n")
    risk = git_repository_config_risk(["git", "log"], cwd=root)
    assert risk is not None
    assert risk.endswith("via core.pager")


def test_git_repository_config_risk_is_none_for_clean_repositories(tmp_path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n\trepositoryformatversion = 0\n")
    assert git_repository_config_risk(["git", "status"], cwd=root) is None


def test_git_repository_config_risk_ignores_inactive_values(tmp_path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n\tpager = false\n")
    assert git_repository_config_risk(["git", "log"], cwd=root) is None


def test_git_repository_config_risk_fails_closed_on_unreadable_config(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").mkdir()
    risk = git_repository_config_risk(["git", "log"], cwd=tmp_path)
    assert risk is not None
    assert "unreadable" in risk


def test_git_repository_config_risk_is_none_outside_any_repository(tmp_path) -> None:
    assert git_repository_config_risk(["git", "log"], cwd=tmp_path) is None


def test_git_repository_config_risk_reads_worktree_pointer(tmp_path) -> None:
    git_dir = tmp_path / "real-git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n\tpager = ./pager\n")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {git_dir}")
    risk = git_repository_config_risk(["git", "log"], cwd=worktree)
    assert risk is not None
    assert risk.endswith("via core.pager")


def test_git_repository_config_risk_treats_empty_config_value_as_unset(
    tmp_path,
) -> None:
    # `core.pager =` (empty value) is FALSE to git, not the implicit true of
    # the valueless bare-key form.
    root = _repository_with_config(tmp_path, "[core]\n\tpager =\n")
    assert git_repository_config_risk(["git", "log"], cwd=root) is None


def test_git_repository_config_risk_treats_bare_config_key_as_true(tmp_path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n\tpager\n")
    risk = git_repository_config_risk(["git", "log"], cwd=root)
    assert risk is not None
    assert risk.endswith("via core.pager")


def test_git_repository_config_risk_fails_closed_on_unreadable_commondir(
    tmp_path,
) -> None:
    # A directory where git expects the commondir file makes the read fail;
    # falling back to the worktree gitdir's own (nonexistent) config would
    # hide the active core.pager in the shared config.
    worktree_git_dir = tmp_path / "real-git" / "worktrees" / "wt"
    worktree_git_dir.mkdir(parents=True)
    (tmp_path / "real-git" / "config").write_text("[core]\n\tpager = ./pager\n")
    (worktree_git_dir / "commondir").mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {worktree_git_dir}")
    risk = git_repository_config_risk(["git", "log"], cwd=worktree)
    assert risk is not None
    assert "unreadable" in risk


def test_git_repository_config_risk_reads_separate_git_dir_config(tmp_path) -> None:
    # A --separate-git-dir checkout has no commondir; its gitdir's own config
    # is the repository config, so the missing file must not fail closed.
    git_dir = tmp_path / "separate-git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n\tpager = ./pager\n")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text(f"gitdir: {git_dir}")
    risk = git_repository_config_risk(["git", "log"], cwd=checkout)
    assert risk is not None
    assert risk.endswith("via core.pager")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            "[include]\n\tpath = ./included\n",
            "via include.path; audit or remove the [include] directive in .git/config",
        ),
        (
            '[includeIf "gitdir:."]\n\tpath = ./included\n',
            "via includeif.path; "
            "audit or remove the [includeIf] directive in .git/config",
        ),
        (
            "[core]\n\tfsmonitor = ./monitor\n",
            "via core.fsmonitor; "
            "audit or remove the core.fsmonitor setting in .git/config",
        ),
        (
            '[filter "lfs"]\n\tclean = ./clean\n',
            "via filter.clean; audit or remove the [filter] command in .git/config",
        ),
        (
            '[filter "lfs"]\n\tprocess = ./filter\n',
            "via filter.process; audit or remove the [filter] command in .git/config",
        ),
    ],
)
def test_git_repository_config_risk_names_remediation_for_broad_vectors(
    tmp_path, config: str, expected: str
) -> None:
    # The resolver is deny-only with no passthrough, so the broadest vectors
    # must tell the reader what to do about them, not just name the key.
    root = _repository_with_config(tmp_path, config)
    risk = git_repository_config_risk(["git", "status"], cwd=root)
    assert risk is not None
    assert risk.endswith(expected)


@pytest.mark.parametrize(
    "command", ["branch", "tag", "grep", "reflog", "stash", "shortlog"]
)
def test_paging_git_subcommands_inspect_repository(command: str) -> None:
    assert analyze_shell_command_policy(["git", command]).inspect_git_repository


def test_git_config_same_line_and_unsupported_syntax_fail_closed(
    tmp_path: Path,
) -> None:
    root = _repository_with_config(tmp_path, '[core] fsmonitor = "./monitor"\n')
    assert "core.fsmonitor" in (
        git_repository_config_risk(["git", "status"], cwd=root) or ""
    )
    (root / ".git" / "config").write_text("[core]\n  nonsense ! syntax\n")
    assert "unreadable" in (
        git_repository_config_risk(["git", "status"], cwd=root) or ""
    )


def test_git_config_invalid_utf8_fails_closed(tmp_path: Path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n")
    (root / ".git" / "config").write_bytes(b"[core]\nfsmonitor = \xff\n")
    assert "unreadable" in (
        git_repository_config_risk(["git", "status"], cwd=root) or ""
    )


def test_no_pager_disables_only_pager_config_vector(tmp_path: Path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n pager = ./pager\n")
    assert git_repository_config_risk(["git", "--no-pager", "log"], cwd=root) is None
    assert git_repository_config_risk(["git", "-P", "log"], cwd=root) is None
    assert git_repository_config_risk(["git", "--no-pager", "-p", "log"], cwd=root)


@pytest.mark.parametrize("globals_", ["", "--no-pager ", "-C . "])
def test_repository_alias_requires_approval(bash_tool: Bash, globals_: str) -> None:
    git_dir = bash_tool.cwd / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[alias]\n planted = !touch payload\n")
    command = f"git {globals_}planted"
    assert analyze_shell_command_policy(shlex.split(command)).requires_approval
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission == ToolPermission.NEVER
    assert not (bash_tool.cwd / "payload").exists()


def test_git_cwd_relocations_models_the_c_chain(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    # No -C (or an empty -C value, a no-op relocation git itself accepts):
    # the tracked cwds resolve unchanged.
    assert git_cwd_relocations(shlex.split("git archive -o b.tar HEAD"), {root}) is None
    assert (
        git_cwd_relocations(shlex.split("git -C '' archive -o b.tar HEAD"), {root})
        is None
    )
    assert (
        git_cwd_relocations(
            shlex.split("git --git-dir=.git archive -o b.tar HEAD"), {root}
        )
        is None
    )
    # -C relocates each tracked cwd through the chain, in turn.
    assert git_cwd_relocations(
        shlex.split("git -C .git archive -o config HEAD"), {root}
    ) == {root / ".git"}
    assert git_cwd_relocations(
        shlex.split("git -C sub -C .. archive -o b.tar HEAD"), {root}
    ) == {root}
    assert git_cwd_relocations(
        shlex.split("git -C /abs archive -o b.tar HEAD"), {root}
    ) == {Path("/abs")}


# The resolver-layer git -C relocated matrices (denied and benign) migrated
# to the shared fixtures in tests/tools/_git_apply_matrix.py (stage 12b-2);
# the resolver-layer assertions live in test_git_metadata_guard.py.
