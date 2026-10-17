"""Git subcommand classification tests for the shell-command policy suite (config-write guards, push/include compounds, program-selection keys, remote mutations, raw-write and work-tree governance, canonical git form, the git denylist, reset/clean/diff classification, git globals, apply/init/worktree write operands, round-3 metadata writes, worktree move/submodule)."""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

import pytest

from chartreux.core.dangerous_commands import DangerousCommandAuthority
from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    analyze_shell_command_policy,
    canonical_git_command,
    git_work_tree_writing_subcommand,
    path_candidates,
    resolve_repository_scope,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from tests.tools._git_apply_matrix import (
    GIT_APPLY_BENIGN_COMMANDS,
    GIT_APPLY_UNSAFE_PATHS_COMMANDS,
)
from tests.tools._shell_policy_test_support import bash_tool as bash_tool


@pytest.mark.parametrize(
    "command",
    [
        "fsck",
        "bundle list-heads",
        "gc --auto",
        "maintenance run",
        "sparse-checkout list",
    ],
)
def test_unmodeled_git_builtins_are_not_described_as_aliases(
    bash_tool: Bash, command: str
) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=f"git {command}"))
    assert context is not None and context.permission is ToolPermission.NEVER
    assert context.reason and "unmodeled git builtin" in context.reason
    assert "denied conservatively" in context.reason
    assert "permission settings cannot override" in context.reason
    assert "aliases" not in context.reason


@pytest.mark.parametrize(
    "command",
    [
        # git resolves repeated booleans in order and accepts --no-
        # spellings, so the final --no-dry-run cancels --dry-run (or -n) and
        # re-enables the destructive clean: the shipped policy layer must
        # require approval for it too, not only the classifier.
        "git clean --force --dry-run --no-dry-run",
        "git clean --force -n --no-dry-run",
        "git clean -f --dry-run --no-dry-run",
        "git clean -fd --no-dry-run",
        "git clean -fd",
        "git clean -f -d",
    ],
)
def test_clean_dry_run_negation_requires_approval(command: str) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert policy.requires_approval


@pytest.mark.parametrize(
    "command",
    [
        # Effective last-wins semantics: --no-force cancels --force (git
        # itself refuses to clean), and a trailing -n wins.
        "git clean --force --no-force",
        "git clean -f --no-force",
        "git clean --no-dry-run -n",
        "git clean -n",
        "git clean -fdn",
        "git clean -n -f",
        "git clean",
    ],
)
def test_negated_clean_flags_stay_permitted(command: str) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert not policy.requires_approval


@pytest.mark.parametrize(
    "command",
    [
        "git config remote.origin.mirror true",
        "git config --global remote.x.push +HEAD:main",
        "git config set remote.origin.mirror true",
        "git config --unset remote.origin.mirror",
        "git config -e",
        # The remote definition, the program a push runs, url rewrites, and
        # the bare push destination are push-effecting too.
        "git config remote.origin.url https://example.com/x.git",
        "git config remote.origin.pushurl ../evil.git",
        "git config remote.origin.receivepack echo",
        "git config remote.origin.uploadpack echo",
        "git config remote.origin.proxy echo",
        "git config url.https://evil/.pushInsteadOf https://github.com/",
        "git config url.https://evil/.insteadOf https://github.com/",
        "git config branch.main.remote origin",
        "git config branch.main.pushremote ../evil.git",
        "git config --unset remote.origin.url",
        "git config set branch.main.remote origin",
        "git config --remove-section url.https://evil/",
        "git config --rename-section branch.old branch.new",
        # The remote's vcs helper: git remote update executes the configured
        # git-remote-<vcs> helper from PATH.
        "git config remote.origin.vcs evil",
        "git config --unset remote.origin.vcs",
        # The program-execution family: the ssh command (even a --dry-run),
        # the git:// proxy, the ask-pass program, and the credential helpers.
        "git config core.sshCommand evil",
        "git config core.gitProxy evil",
        "git config core.askPass evil",
        "git config set core.gitProxy evil",
        "git config credential.helper evil",
        "git config credential.https://example.com.helper evil",
        "git config --unset credential.helper",
        # An authored include folds a file into configuration resolution,
        # which can set any push-effecting key for a verified remote.
        "git config include.path /tmp/evil.cfg",
        "git config includeIf.gitdir:~/work/.path /tmp/evil.cfg",
        "git config set include.path /tmp/evil.cfg",
        "git config --unset include.path",
        # Section operations reaching the credential and include sections.
        "git config --remove-section credential",
        "git config --remove-section include",
        "git config --rename-section includeIf.gitdir:/x includeIf.gitdir:/y",
    ],
)
def test_push_effecting_config_writes_are_denied(bash_tool: Bash, command: str) -> None:
    # A hard guard: denied with no grantable proposal, so no answer can
    # authorize the write.
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert context.reason


def test_config_then_push_compound_is_denied_without_a_proposal(
    bash_tool: Bash,
) -> None:
    # The same-call exploit: writing the config that turns an ordinary push
    # into a mirror push is denied on the config segment, and the whole
    # compound call carries no proposal.
    context = bash_tool.resolve_permission(
        BashArgs(command="git config remote.origin.mirror true && git push origin")
    )
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert context.reason
    assert "push-effecting configuration" in context.reason


@pytest.mark.parametrize(
    "command",
    [
        # core.hooksPath selects the directory git runs hooks from; the write
        # is a hard guard in every scope, including --file and relocation.
        "git config core.hooksPath /tmp/h",
        "git config --global core.hooksPath /tmp/h",
        "git config --worktree core.hooksPath /tmp/h",
        "git config --file cfg core.hooksPath /tmp/h",
        "git config set core.hooksPath /tmp/h",
        "git config --unset core.hooksPath",
        "git -C .git config core.hooksPath /tmp/h",
        "git --git-dir=.git config core.hooksPath /tmp/h",
        "env X=1 git config core.hooksPath /tmp/h",
        # core.editor and sequence.editor select the programs git runs for
        # commit/tag/rebase without -m and interactive rebase.
        "git config core.editor evil",
        "git config --global core.editor evil",
        "git config --worktree core.editor evil",
        "git config --file cfg core.editor evil",
        "git config set core.editor evil",
        "git config --unset core.editor",
        "git -C .git config core.editor evil",
        "git --git-dir=.git config core.editor evil",
        "git config sequence.editor evil",
        "git config --global sequence.editor evil",
        "git config --worktree sequence.editor evil",
        "git config --file cfg sequence.editor evil",
        "git config set sequence.editor evil",
        "git config --unset sequence.editor",
        "git -C .git config sequence.editor evil",
        "git --git-dir=.git config sequence.editor evil",
        "env X=1 git config sequence.editor evil",
        # filter.<name>.clean and filter.<name>.process select the programs
        # git runs for files with that filter (git add, git checkout).
        "git config filter.pwn.clean evil",
        "git config --global filter.pwn.clean evil",
        "git config --worktree filter.pwn.clean evil",
        "git config --file cfg filter.pwn.clean evil",
        "git config set filter.pwn.clean evil",
        "git config --add filter.pwn.clean evil",
        "git config --replace-all filter.pwn.clean evil",
        "git config --unset filter.pwn.clean",
        "git config unset filter.pwn.clean",
        "git -C .git config filter.pwn.clean evil",
        "git --git-dir=.git config filter.pwn.clean evil",
        "git config filter.pwn.process evil",
        "git config --file cfg filter.pwn.process evil",
        "git config set filter.pwn.process evil",
        "git config --unset-all filter.pwn.process",
        "git -C .git config filter.pwn.process evil",
        # merge.<name>.driver selects the program git runs for files with
        # that merge driver (git merge).
        "git config merge.ours.driver evil",
        "git config --global merge.ours.driver evil",
        "git config --worktree merge.ours.driver evil",
        "git config --file cfg merge.ours.driver evil",
        "git config set merge.ours.driver evil",
        "git config --unset merge.ours.driver",
        "git -C .git config merge.ours.driver evil",
        "git --git-dir=.git config merge.ours.driver evil",
        # diff.<name>.command/textconv/external select the programs git
        # runs for diffs with that driver (git diff, git log -p).
        "git config diff.external.command evil",
        "git config --global diff.external.command evil",
        "git config --worktree diff.external.command evil",
        "git config --file cfg diff.external.command evil",
        "git config set diff.external.command evil",
        "git config --unset diff.external.command",
        "git -C .git config diff.external.command evil",
        "git --git-dir=.git config diff.external.command evil",
        "git config diff.external.textconv evil",
        "git config --file cfg diff.external.textconv evil",
        "git config --unset-all diff.external.textconv",
        "git config diff.external.external evil",
        "git config set diff.external.external evil",
        "git config --replace-all diff.external.external evil",
        # Section and key names fold case-insensitively; the guard fails
        # closed on any spelling.
        "git config Filter.Pwn.Clean evil",
        "git config MERGE.Ours.Driver evil",
        "git config Diff.External.Command evil",
        "git config diff.EXTERNAL.Textconv evil",
        # Section renames reaching core/sequence (destination names included).
        "git config --rename-section other core",
        "git config --rename-section core.other core",
        "git config --rename-section other sequence",
        "git config rename-section other core",
        "git config --file cfg --rename-section other core",
        "git -C .git config --rename-section other core",
        "git --git-dir=.git config --rename-section other core",
        # The same bypass class for the driver sections: renaming into
        # (or out of) filter/merge/diff lands an agent-authored driver key.
        "git config --rename-section other filter",
        "git config --rename-section filter other",
        "git config --rename-section filter.lfs filter",
        "git config --rename-section other merge",
        "git config --rename-section merge ours2",
        "git config --rename-section other diff",
        "git config --rename-section diff ours",
        "git config --file cfg --rename-section other filter",
        "git -C .git config --rename-section other merge",
        "git --git-dir=.git config --rename-section other diff",
    ],
)
def test_program_selection_config_writes_are_denied(
    bash_tool: Bash, command: str
) -> None:
    # A hard guard: denied with no grantable proposal, so no answer can
    # authorize the write.
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert context.reason
    assert "program-selection" in context.reason


@pytest.mark.parametrize(
    "command",
    [
        # Reads of the program-selection keys never write.
        "git config --get core.hooksPath",
        "git config --get core.editor",
        "git config --get sequence.editor",
        "git config core.hooksPath",
        "git config core.editor",
        "git config sequence.editor",
        "git config --get filter.pwn.clean",
        "git config --get filter.pwn.process",
        "git config --get merge.ours.driver",
        "git config --get diff.external.command",
        "git config --get diff.external.textconv",
        "git config --get diff.external.external",
        "git config filter.pwn.clean",
        "git config merge.ours.driver",
        "git config diff.external.command",
        "git config --list",
        # Writes to unrelated keys and sections stay routine, including
        # section removals (a removal cannot move a key into a guarded
        # section) and unrelated renames. The driver sections keep their
        # non-program keys: required/smudge are boolean flags and data
        # spellings, not command strings (a documented residual).
        "git config user.name X",
        "git config core.pager less",
        "git config core.autocrlf true",
        "git config sequence.squashWarning false",
        "git config filter.pwn.required true",
        "git config filter.pwn.smudge cat",
        "git config merge.ours.name ours",
        "git config merge.conflictStyle diff3",
        "git config diff.external.foo bar",
        "git config diff.renameLimit 100",
        "git config --remove-section core.old",
        "git config --remove-section sequence.old",
        "git config --remove-section filter.old",
        "git config --remove-section merge.old",
        "git config --remove-section diff.old",
        "git config --rename-section other.one other.two",
    ],
)
def test_program_selection_config_reads_and_unrelated_writes_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        # Read forms do not put the destination in write_path_values: a
        # reading config --file stays an authority-scoped operand.
        "git config --file=.git/config --get user.name",
        "git config --file=.git/config user.name",
        "git config --file=.git/config --list",
    ],
)
def test_git_config_read_file_is_not_a_write_operand(command: str) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert policy.write_path_values == ()
    assert ".git/config" in policy.option_path_values


def test_git_work_tree_writing_subcommand_detection() -> None:
    assert git_work_tree_writing_subcommand(shlex.split("git merge evil")) == "merge"
    assert (
        git_work_tree_writing_subcommand(shlex.split("git -C .git apply p.diff"))
        == "apply"
    )
    assert (
        git_work_tree_writing_subcommand(
            shlex.split("git --git-dir=.git --work-tree=.git stash pop")
        )
        == "stash"
    )
    # Readers, non-writers, and unmodeled forms carry nothing to check.
    assert git_work_tree_writing_subcommand(shlex.split("git status")) is None
    assert git_work_tree_writing_subcommand(shlex.split("git -C .git status")) is None
    assert git_work_tree_writing_subcommand(shlex.split("git fetch origin")) is None
    assert git_work_tree_writing_subcommand(shlex.split("git log")) is None
    assert git_work_tree_writing_subcommand(shlex.split("git commit -m x")) is None
    assert git_work_tree_writing_subcommand(shlex.split("ls")) is None


# --- Git raw-write operand and work-tree governance (delta review, family 1) ---


def _git_dir_with_bare_markers(root: Path) -> Path:
    """A minimal git dir: discovery finds it as a repository git dir."""
    git_dir = root / ".git"
    git_dir.mkdir(exist_ok=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "config").write_text("[core]\n\tbare = false\n")
    (git_dir / "objects").mkdir(exist_ok=True)
    (git_dir / "refs").mkdir(exist_ok=True)
    (root / "seed.txt").write_text("seed\n")
    return root


@pytest.mark.parametrize(
    "command",
    [
        # git archive --output/-o writes the named file directly.
        "git archive --output=.git/config HEAD",
        "git archive --output .git/config HEAD",
        "git archive -o .git/config HEAD",
        "git archive -o.git/config HEAD",
        "git archive --output=.git/hooks/pre-commit HEAD",
        # A writing git config --file/-f names the written file.
        "git config --file=.git/config user.name x",
        "git config --file .git/config user.name x",
        "git config -f.git/config user.name x",
        "git config --file=.git/config set user.name x",
        "git config --file=.git/config --unset user.name",
        # The operand guard fires before the classifier's key guard for a
        # metadata destination, and the key guard still covers the rest.
        "git config --file=.git/config core.hooksPath /tmp/h",
        # Work-tree relocation into the metadata dir.
        "git -C .git apply ../hook.diff",
        "git -C .git merge evil",
        "git --git-dir=.git --work-tree=.git merge evil",
        "git --git-dir=.git --work-tree=.git stash pop",
        "git --git-dir=.git --work-tree=.git checkout main",
        "git --work-tree=.git merge evil",
        "git -C .git/hooks apply hook.diff",
    ],
)
def test_git_raw_write_and_work_tree_bypasses_are_denied(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    assert "protected Git metadata" in (context.reason or ""), command


@pytest.mark.parametrize(
    "command",
    [
        # In-workspace destinations and readers stay permitted.
        "git archive --output=bundle.tar HEAD",
        "git archive -obundle.tar HEAD",
        "git config --file=custom.cfg user.name X",
        "git config --file=custom.cfg --get user.name",
        "git config --file=.git/config --get user.name",
        "git -C .git status",
        "git --git-dir=.git log",
        "git --work-tree=. --git-dir=.git diff",
        # In-worktree tree-writing calls and legitimate relocations.
        "git apply fix.diff",
        "git merge evil",
        "git stash pop",
        "git switch main",
        "git fetch origin",
        "git --git-dir=.git --work-tree=wt merge evil",
        "git -C sub apply ../fix.diff",
    ],
)
def test_git_raw_write_and_work_tree_benign_forms_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.ALWAYS, f"{command}: {context.reason}"


def test_include_then_push_compound_is_denied_without_a_proposal(
    bash_tool: Bash,
) -> None:
    # The F-1 exploit shape: the include write folds an authored file that
    # could set remote.<name>.pushurl for a verified remote, so it is
    # denied on its own segment and the whole compound call — the push
    # included — carries no proposal.
    context = bash_tool.resolve_permission(
        BashArgs(command="git config include.path /tmp/evil.cfg && git push origin")
    )
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert context.reason
    assert "push-effecting configuration" in context.reason


@pytest.mark.parametrize(
    "command",
    [
        "git config user.name X",
        "git config --global user.name X",
        "git config --get remote.origin.mirror",
        "git config --get remote.origin.url",
        "git config --get branch.main.remote",
        "git config --get url.https://x/.pushInsteadOf",
        "git config --get remote.origin.vcs",
        "git config --get core.sshCommand",
        "git config --get core.gitProxy",
        "git config --get core.askPass",
        "git config --get credential.helper",
        "git config --get credential.https://example.com.helper",
        "git config --get include.path",
        "git config --get includeIf.gitdir:/x.path",
        "git config --list",
        # Reads of the newly guarded keys never write: the bare-key form is
        # git's read form too.
        "git config core.sshCommand",
        "git config credential.helper",
        "git config include.path",
        "git config remote.origin.vcs",
        # push.default writes stay routine: they broaden what a bare push
        # publishes, never force, delete, or mirror.
        "git config push.default simple",
        "git config remote.origin.fetch +refs/heads/*:refs/remotes/origin/*",
    ],
)
def test_config_reads_and_unrelated_writes_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "command",
    [
        "git remote add origin ../x",
        "git remote set-url origin ../x",
        "git remote rename origin evil",
        "git remote remove origin",
        "git remote rm origin",
    ],
)
def test_git_remote_mutations_are_denied(bash_tool: Bash, command: str) -> None:
    # A git remote mutation redefines where a granted push lands, so it is
    # a hard guard: denied with no grantable proposal.
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert context.reason
    assert "redefines where a granted push lands" in context.reason


@pytest.mark.parametrize(
    "command",
    [
        "git remote",
        "git remote -v",
        "git remote get-url origin",
        "git remote show origin",
        "git remote prune origin",
    ],
)
def test_git_remote_reads_stay_permitted(bash_tool: Bash, command: str) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.ALWAYS


def test_a_push_grant_does_not_authorize_the_config_write(tmp_path: Path) -> None:
    # The guard's boundary: an approved ordinary push stays publication,
    # but it never extends to the config write that would make the push a
    # mirror push.
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    # The push destination is a bare name, so the repository must resolve
    # it: a configured remote keeps the ordinary push grantable.
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "remote",
            "add",
            "origin",
            "https://example.com/x.git",
        ],
        check=True,
    )
    scope = resolve_repository_scope(repo, GitRepositoryRelocation())
    assert scope is not None
    config = BashToolConfig(denylist=[])
    tool = Bash(config_getter=lambda: config, state=BaseToolState(), cwd=repo)
    tool.dangerous_command_authorities_getter = lambda: (
        DangerousCommandAuthority(
            operations=frozenset({"git.push"}), scope=scope, source="runtime_grant"
        ),
    )
    context = tool.resolve_permission(BashArgs(command="git push origin"))
    assert context is not None
    assert context.permission is ToolPermission.ALWAYS
    context = tool.resolve_permission(
        BashArgs(command="git config remote.origin.mirror true && git push origin")
    )
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None


def test_unknown_git_subcommand_may_be_an_alias(bash_tool: Bash) -> None:
    context = bash_tool.resolve_permission(BashArgs(command="git fixture-alias"))
    assert context is not None and context.permission is ToolPermission.NEVER
    assert context.reason and "aliases and external helpers" in context.reason


@pytest.fixture
def git_denylist_tool(tmp_path, monkeypatch) -> Bash:
    # Explicit git entries only: the shipped defaults are extended elsewhere and
    # must not be pinned by these tests.
    monkeypatch.chdir(tmp_path)
    config = BashToolConfig(
        denylist=[
            "git push",
            "git checkout",
            "git stash drop",
            "git stash clear",
            "git restore",
            "git switch --discard-changes",
            "git switch -f",
            "git reflog expire",
            "git reflog delete",
        ]
    )
    return Bash(config_getter=lambda: config, state=BaseToolState())


@pytest.fixture
def git_denylist_resolver(git_denylist_tool: Bash):
    from chartreux.core.tools.builtins._shell_permission_resolver import (
        ShellPermissionResolver,
    )

    return ShellPermissionResolver(
        git_denylist_tool.config,
        git_denylist_tool.cwd,
        git_denylist_tool.workspace,
        git_denylist_tool.path_authority,
    )


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (["git", "push"], ["git", "push"]),
        (["git", "-C", ".", "push"], ["git", "push"]),
        (["git", "-Csubdir", "push"], ["git", "push"]),
        (["git", "-C.", "push"], ["git", "push"]),
        (["git", "--no-pager", "push"], ["git", "push"]),
        (["git", "-C", ".", "--no-pager", "-C", "sub", "push"], ["git", "push"]),
        (["git", "--git-dir", ".", "push"], ["git", "push"]),
        (["git", "--git-dir=.", "push"], ["git", "push"]),
        # The full subcommand suffix is retained, never truncated.
        (["git", "-C", ".", "stash", "drop"], ["git", "stash", "drop"]),
        (["git", "-C", ".", "stash", "clear"], ["git", "stash", "clear"]),
        (["git", "-C", ".", "checkout", "file"], ["git", "checkout", "file"]),
        (["git", "-C", ".", "restore", "file"], ["git", "restore", "file"]),
        (["/usr/bin/git", "-C", ".", "push"], ["/usr/bin/git", "push"]),
        # The walker stops at the subcommand: later tokens are never stripped.
        (["git", "push", "--no-pager"], ["git", "push", "--no-pager"]),
        (["git", "status", "--", "push"], ["git", "status", "--", "push"]),
        (["git", "stash", "list"], ["git", "stash", "list"]),
        # A global option's operand is consumed even when it names a subcommand.
        (["git", "-C", "push", "status"], ["git", "status"]),
    ],
)
def test_canonical_git_command_elides_recognized_global_options(
    tokens: list[str], expected: list[str]
) -> None:
    assert canonical_git_command(tokens) == expected


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "--unknown", "push"],
        ["git", "-c", "alias.pwn=!touch pwn", "push"],
        ["git", "--config-env", "foo=bar", "push"],
        ["git", "--config-env=foo=bar", "push"],
        ["ls", "-la"],
        ["gitx", "-C", ".", "push"],
        [],
    ],
)
def test_canonical_git_command_fails_closed_on_unrecognized_globals(
    tokens: list[str],
) -> None:
    assert canonical_git_command(tokens) is None


@pytest.mark.parametrize(
    "command, pattern",
    [
        ("git -C . push", "git push"),
        ("git -Csubdir push", "git push"),
        ("git --no-pager push", "git push"),
        ("git -C . --no-pager -C sub push", "git push"),
        ("git --git-dir . push", "git push"),
        ("git --git-dir=. push", "git push"),
        ("git -C . checkout file", "git checkout"),
        ("git -C . restore file", "git restore"),
        ("git -C . stash drop", "git stash drop"),
        ("git -C . stash clear", "git stash clear"),
        ("git -C . switch --discard-changes", "git switch --discard-changes"),
        ("git -C . switch -f main", "git switch -f"),
        ("git -C . reflog expire --all", "git reflog expire"),
        ("git -C . reflog delete main", "git reflog delete"),
        ("/usr/bin/git -C . push", "git push"),
    ],
)
def test_denylist_matches_canonical_git_form(
    git_denylist_resolver, command: str, pattern: str
) -> None:
    assert git_denylist_resolver._find_denylist_match(command) == pattern


@pytest.mark.parametrize(
    "command",
    [
        # Unrecognized globals get no normalization; the policy layer denies them.
        "git --unknown push",
        "git -c alias.pwn=!touch pwn push",
        "git --config-env foo=bar push",
        # A global option's operand is consumed, even when it names a subcommand.
        "git -C push status",
        # Tokens after the subcommand are never stripped.
        "git status -- push",
        "git stash -- drop",
        # A shorter subcommand suffix never matches a longer pattern.
        "git stash list",
        "git stash",
        # Only the destructive switch/reflog forms match; plain forms do not.
        "git switch main",
        "git switch -c feature",
        "git stash pop",
        "git reflog",
        "git reflog show",
    ],
)
def test_denylist_canonical_git_form_rejects_non_matching_commands(
    git_denylist_resolver, command: str
) -> None:
    assert git_denylist_resolver._find_denylist_match(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "git --unknown push",
        "git -c alias.pwn=!touch pwn push",
        "git --config-env foo=bar push",
    ],
)
def test_unparsable_git_globals_stay_denied_by_policy_layer(
    git_denylist_resolver, command: str
) -> None:
    # No denylist normalization applies, and the policy layer hard-denies the
    # spelling, so the command never runs.
    assert git_denylist_resolver._find_denylist_match(command) is None
    assert analyze_shell_command_policy(shlex.split(command)).requires_approval


def test_guardrail_denies_command_builtin_wrapped_git_global_form(
    git_denylist_resolver,
) -> None:
    # The "command" builtin is unwrapped before the denylist lookup, so the
    # canonical git form applies to the wrapped command too.
    context = git_denylist_resolver._resolve_guardrail_permission([
        "command git -C . push"
    ])

    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "matches denylist pattern 'git push'" in context.reason


def test_bash_tool_denies_git_global_form_via_denylist(git_denylist_tool) -> None:
    result = git_denylist_tool.resolve_permission(BashArgs(command="git -C . push"))

    assert result is not None
    assert result.permission == ToolPermission.NEVER
    assert "matches denylist pattern 'git push'" in (result.reason or "")


def test_git_reset_hard_requires_approval() -> None:
    assert analyze_shell_command_policy(["git", "reset", "--hard"]).requires_approval


def test_git_reset_soft_needs_no_approval() -> None:
    assert not analyze_shell_command_policy([
        "git",
        "reset",
        "--soft",
    ]).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "clean", "-f"],
        ["git", "clean", "-fd"],
        ["git", "clean", "-fdx"],
        ["git", "clean", "-f", "-e", "pattern"],
    ],
)
def test_git_clean_force_requires_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "clean", "-n"],
        ["git", "clean", "-n", "-f"],
        ["git", "clean", "-ndx"],
        ["git", "clean", "--dry-run", "-f"],
    ],
)
def test_git_clean_dry_run_needs_no_approval(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).requires_approval


def test_git_global_options_precede_subcommand() -> None:
    assert analyze_shell_command_policy([
        "git",
        "-C",
        "/repo",
        "reset",
        "--hard",
    ]).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "diff", "--output", "/tmp/x"],
        ["git", "diff", "--output=/tmp/x"],
        ["git", "show", "--ext-diff"],
        ["git", "show", "--textconv"],
    ],
)
def test_git_diff_output_and_external_tools_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "status"],
        ["git", "log"],
        ["git", "--no-pager", "diff"],
        ["git", "diff", "--no-ext-diff"],
        ["git", "--help"],
        ["git", "--version"],
    ],
)
def test_git_read_only_commands_need_no_approval(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "-c", "core.editor=x", "status"],
        ["git", "--config-env", "x", "status"],
        ["git", "--unknown-global", "status"],
    ],
)
def test_git_config_override_and_unknown_global_fail_closed(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


def test_git_dir_values_are_collected_as_paths() -> None:
    policy = analyze_shell_command_policy(["git", "--git-dir", "/x", "log"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("/x",)

    attached = analyze_shell_command_policy(["git", "--git-dir=/x", "log"])
    assert attached.option_path_values == ("/x",)


def test_git_diff_no_index_inspects_positional_paths() -> None:
    policy = analyze_shell_command_policy(["git", "diff", "--no-index", "a", "b"])
    assert not policy.requires_approval
    assert policy.inspect_positional_paths
    assert policy.positional_values == ("a", "b")


@pytest.mark.parametrize("command", ["git status", "git diff", "git stash list"])
@pytest.mark.parametrize("custom", [False, True])
def test_git_index_hooks_are_denied(
    bash_tool: Bash, command: str, custom: bool
) -> None:
    git_dir = bash_tool.cwd / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(
        "[core]\n hooksPath = custom-hooks\n" if custom else "[core]\n bare = false\n"
    )
    hooks = bash_tool.cwd / "custom-hooks" if custom else git_dir / "hooks"
    hooks.mkdir()
    hook = hooks / "post-index-change"
    hook.write_text("#!/bin/sh\ntouch payload\n")
    hook.chmod(0o755)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission == ToolPermission.NEVER
    assert (
        "hooksPath" in (context.reason or "")
        if custom
        else "post-index-change" in (context.reason or "")
    )
    assert not (bash_tool.cwd / "payload").exists()


@pytest.mark.parametrize(
    "target", [".git/config", ".git/hooks/post-index-change", "../outside"]
)
@pytest.mark.parametrize("operator", [">", ">>", "2>", "&>"])
def test_git_reader_file_redirections_are_denied(
    bash_tool: Bash, target: str, operator: str
) -> None:
    context = bash_tool.resolve_permission(
        BashArgs(command=f"git log {operator} {target}")
    )
    assert context is not None and context.permission == ToolPermission.NEVER
    assert "require approval" not in (context.reason or "")


@pytest.mark.parametrize(
    "operand", ["/outside/file", "../outside", ".env", ".env.local"]
)
@pytest.mark.parametrize("attached", [False, True])
def test_git_blame_contents_is_workspace_scoped(
    bash_tool: Bash, operand: str, attached: bool
) -> None:
    option = f"--contents={operand}" if attached else f"--contents {operand}"
    command = f"git blame {option} tracked.txt"
    assert operand in path_candidates(
        shlex.split(command), inspect_positional_paths=False
    )
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission == ToolPermission.NEVER


@pytest.mark.parametrize(
    "command",
    [
        "git log",
        "git status",
        "git diff",
        "git blame tracked.txt",
        "git blame --contents=tracked.txt tracked.txt",
        "git log 2>&1",
    ],
)
def test_clean_git_readers_remain_permitted(bash_tool: Bash, command: str) -> None:
    git_dir = bash_tool.cwd / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n bare = false\n")
    hooks = git_dir / "hooks"
    hooks.mkdir()
    # Git ignores non-executable hook files, including the shipped samples.
    (hooks / "post-index-change").write_text("#!/bin/sh\nexit 0\n")
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None and context.permission == ToolPermission.ALWAYS


# --- git apply --directory write roots and the --unsafe-paths hard guard
# --- (delta re-review, round 2) ---


@pytest.mark.parametrize("command", GIT_APPLY_UNSAFE_PATHS_COMMANDS)
def test_git_apply_unsafe_paths_is_a_hard_guard(command: str) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert policy.requires_approval
    assert policy.denial_reason is not None
    assert "--unsafe-paths" in policy.denial_reason


@pytest.mark.parametrize("command", GIT_APPLY_BENIGN_COMMANDS)
def test_git_apply_benign_forms_carry_no_hard_guard(command: str) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert not policy.requires_approval


@pytest.mark.parametrize(
    "command, expected",
    [
        # git apply --directory relocates the patch's write root: the value
        # is a write operand like any other, in every spelling.
        ("git apply --directory=.git/hooks hook.diff", (".git/hooks",)),
        ("git apply --directory .git/hooks hook.diff", (".git/hooks",)),
        ("git apply --directory=.git/hooks/ hook.diff", (".git/hooks/",)),
        ("git apply -p0 --directory=.git/hooks hook.diff", (".git/hooks",)),
        ("git apply --directory=/abs/path hook.diff", ("/abs/path",)),
        ("git apply hook.diff", ()),
        # git init creates its positional directory and the git dir its
        # --separate-git-dir names; --template is a read, and --shared
        # takes an optional attached value only, so it never consumes the
        # positional.
        ("git init newdir", ("newdir",)),
        ("git init --shared newdir", ("newdir",)),
        ("git init --template=/usr/share/git-core/templates newdir", ("newdir",)),
        ("git init --separate-git-dir=.git/evil newdir", ("newdir", ".git/evil")),
        ("git init", ()),
        # git worktree add's first positional is the new work tree;
        # --reason/-b take values, --orphan is bare.
        ("git worktree add wt2", ("wt2",)),
        ("git worktree add --lock --reason why wt2", ("wt2",)),
        ("git worktree add -b topic wt2", ("wt2",)),
        ("git worktree add --orphan wt2", ("wt2",)),
        ("git worktree list", ()),
        ("git worktree add", ()),
    ],
)
def test_git_apply_init_and_worktree_destinations_are_write_operands(
    command: str, expected: tuple[str, ...]
) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert policy.write_path_values == expected, command


@pytest.mark.parametrize(
    "command",
    [
        # --directory relocates the patch's write root into the metadata.
        "git apply --directory=.git/hooks hook.diff",
        "git apply --directory .git/hooks hook.diff",
        "git apply --directory=./.git/hooks hook.diff",
        "git apply --directory=.git/../.git/hooks hook.diff",
        "git apply --directory=.git/hooks/ hook.diff",
        "git apply -p0 --directory=.git/hooks hook.diff",
        "git apply --directory=.git hook.diff",
        "git -C subdir apply --directory=../.git/hooks hook.diff",
        # ...and outside the workspace.
        "git apply --directory=/tmp/outside hook.diff",
        "git apply --directory=../outside hook.diff",
    ],
)
def test_git_apply_directory_write_roots_are_denied(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    (bash_tool.cwd / "subdir").mkdir()
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    reason = context.reason or ""
    if "outside" in command:
        assert "outside the authorized workspace" in reason, command
    else:
        assert "protected Git metadata" in reason, command


# The resolver-layer unsafe-paths and benign-apply matrices migrated to the
# shared fixtures in tests/tools/_git_apply_matrix.py (stage 12b-1); the
# resolver-layer assertions live in test_git_metadata_guard.py.


# --- git init / git worktree add destinations (delta re-review, round 2) ---


@pytest.mark.parametrize(
    "command",
    [
        # Destinations resolving into the metadata.
        "git init .git/evil",
        "git init .git/hooks/evil",
        "git init --separate-git-dir=.git/evil newdir",
        "git init --separate-git-dir .git/evil newdir",
        "git worktree add .git/wt2",
        "git worktree add .git/hooks/wt",
        "git worktree add -b wtb .git/wt2",
        "git worktree add --lock --reason why .git/wt2",
        "git -C subdir init ../.git/evil",
        "git -C subdir worktree add ../.git/wt2",
        # ...and outside the workspace.
        "git init /tmp/outside-init",
        "git worktree add ../outside-wt",
    ],
)
def test_git_init_and_worktree_add_metadata_destinations_are_denied(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    (bash_tool.cwd / "subdir").mkdir()
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    reason = context.reason or ""
    if "outside" in command:
        assert "outside the authorized workspace" in reason, command
    else:
        assert "protected Git metadata" in reason, command


@pytest.mark.parametrize(
    "command",
    [
        # No positional: git init creates .git in the cwd, which stays
        # permitted; in-workspace destinations stay permitted.
        "git init",
        "git init newdir",
        "git init --shared newdir",
        "git init --shared=group newdir",
        "git init -b trunk newdir",
        "git init --template=/usr/share/git-core/templates newdir",
        "git worktree add wt2",
        "git worktree add -b wtb wt2",
        "git worktree add --orphan wt2",
        "git worktree list",
        # git clone keeps its existing treatment (not a metadata destination).
        "git clone -q . /tmp/x-clone",
    ],
)
def test_git_init_and_worktree_add_benign_forms_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.ALWAYS, f"{command}: {context.reason}"


@pytest.mark.parametrize(
    "command",
    [
        # Binutils/assembler output operands into metadata.
        "strip -o .git/hooks/pre-push evil",
        "strip -o .git/config evil",
        "strip -o.git/config evil",
        "strip .git/config",
        "strip -R .text .git/config",
        "objcopy evil .git/hooks/pre-push",
        "objcopy --strip-all evil .git/config",
        "objcopy .git/config",
        "as -o .git/config evil.s",
        "as -o.git/config evil.s",
        # Config-content writers into metadata (gpg/xmllint output, scp
        # local-to-local in both directions).
        "gpg --batch --yes --pinentry-mode loopback --passphrase x --decrypt -o .git/config payload.gpg",
        "gpg --batch --yes --pinentry-mode loopback --passphrase x --decrypt --output .git/config payload.gpg",
        "gpg --batch --yes --pinentry-mode loopback --passphrase x --decrypt --output=.git/config payload.gpg",
        "gpg --batch --yes --pinentry-mode loopback --passphrase x --decrypt -o.git/config payload.gpg",
        "xmllint --output .git/config doc.xml",
        "xmllint -o .git/config doc.xml",
        "xmllint --output=.git/config doc.xml",
        "scp seed.txt .git/config",
        "scp .git/config out.txt",
        "scp -r .git host:x",
        # In-place compression mutators.
        "bzip2 .git/config",
        "bunzip2 .git/config.bz2",
        "bzcat .git/config.bz2",
        "xz .git/config",
        "unxz .git/config.xz",
        "xzcat .git/config.xz",
        "zstd .git/config",
        "zstd -q -o .git/config seed.txt",
        "zstd --output-dir-flat=.git seed.txt",
        "zstd --output-dir-mirror .git seed.txt",
        "zstdmt .git/config",
        "unzstd .git/config.zst",
        "lzma .git/config",
        "unlzma .git/config.lzma",
        "lzcat .git/config.lzma",
        "lzmainfo .git/config.lzma",
        # Archive creation/extraction destinations.
        "zip .git/evil.zip seed.txt",
        "zip -r .git/evil2.zip sub",
        "7z a .git/evil.7z seed.txt",
        "7zr a .git/evilr.7z seed.txt",
        "7za a .git/evil.7z seed.txt",
        "7z x -o.git evil.7z",
        "7z x -o .git evil.7z",
        "unar -f -o .git evil.zip",
        "unar -f -o=.git evil.zip",
        "unar -f -output-directory .git evil.zip",
        "unar -f -output-directory=.git evil.zip",
        # mktemp creation operands.
        "mktemp .git/evil.XXXXXX",
        "mktemp -p .git evil.XXXXXX",
        "mktemp --tmpdir=.git evil.XXXXXX",
        # git worktree move / submodule add destinations.
        "git worktree move wt .git/wt2",
        "git worktree move -f wt .git/wt2",
        "git -C sub worktree move ../wt ../.git/wt2",
        "git submodule add ../src .git/evilmod",
        "git submodule add -b main ../src .git/evilmod",
        "git submodule add --separate-git-dir=.git/evilmod ../src dest",
        "git -C sub submodule add ../../src ../.git/evilmod",
        # ...and outside the workspace (the write-operand authority denies
        # them, exactly like the init/worktree-add flips).
        "git worktree move wt ../outside-wt",
        "git submodule add ../src ../outside-mod",
    ],
)
def test_round3_write_operand_metadata_writes_are_denied(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    (bash_tool.cwd / "sub").mkdir()
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    reason = context.reason or ""
    if "outside" in command:
        assert "outside the authorized workspace" in reason, command
    else:
        assert "protected Git metadata" in reason, command


@pytest.mark.parametrize(
    "command",
    [
        # Binutils/assembler benign forms.
        "strip -o stripped evil",
        "strip evil",
        "strip -R .text -o stripped2 evil",
        # --output abbreviates --output-target in real strip, so the form
        # consumes the operand as a bfdname and errors: no write happens.
        "strip --output .git/config evil",
        "objcopy evil copied",
        "objcopy --strip-all evil copied2",
        "as -o evil.o evil.s",
        # Config-content writers to workspace/scratchpad targets.
        "gpg --batch --yes --pinentry-mode loopback --passphrase x --decrypt -o out.txt payload.gpg",
        "xmllint --output out.xml doc.xml",
        "xmllint -o out2.xml doc.xml",
        "scp seed.txt copied.txt",
        "scp -P 22 seed.txt copied2.txt",
        "scp -i fake-id seed.txt copied3.txt",
        # scp remote forms stay outside the local-metadata scope.
        "scp seed.txt host:x",
        "scp host:x local.txt",
        # In-place compression mutators on workspace files.
        "bzip2 seed.txt",
        "bunzip2 seed.txt.bz2",
        "bzcat seed.txt.bz2",
        "xz seed.txt",
        "unxz seed.txt.xz",
        "xzcat seed.txt.xz",
        "zstd seed.txt",
        "zstd -q -o out.zst seed.txt",
        "zstd --output-dir-flat=odir seed.txt",
        "zstd --output-dir-mirror odm seed.txt",
        "zstdmt seed.txt",
        "unzstd seed.txt.zst",
        "lzma seed.txt",
        "unlzma seed.txt.lzma",
        "lzcat seed.txt.lzma",
        "lzmainfo seed.txt.lzma",
        # Archive destinations in the workspace.
        "zip out.zip seed.txt",
        "zip -r out2.zip sub",
        "zip -q -b . out3.zip seed.txt",
        "zip -P secret out4.zip seed.txt",
        "7z a out.7z seed.txt",
        "7zr a outr.7z seed.txt",
        "7za a out5.7z seed.txt",
        "7z x -od7z evil.7z",
        "7z l out.7z",
        # -o= names a directory literally called "=.git" in real 7z (the
        # value is everything after -o), never the metadata directory.
        "7z x -o=.git evil.7z",
        "unar -f -o unarout evil.zip",
        "unar -f -o=unarout2 evil.zip",
        # mktemp creation in the workspace (no operand: the default temp
        # directory stays permitted).
        "mktemp",
        "mktemp tmpl.XXXXXX",
        "mktemp -p . tmpl2.XXXXXX",
        "mktemp --tmpdir=. tmpl3.XXXXXX",
        # git worktree move / submodule add to workspace destinations.
        "git worktree move wt wt3",
        "git worktree move -f wt wt4",
        "git submodule add ../src submod",
        "git submodule add -b main ../src submod2",
        "git submodule list",
        "git worktree list",
    ],
)
def test_round3_write_operand_benign_forms_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    _git_dir_with_bare_markers(bash_tool.cwd)
    (bash_tool.cwd / "sub").mkdir()
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.ALWAYS, f"{command}: {context.reason}"


@pytest.mark.parametrize(
    "command,expected",
    [
        ("git worktree move wt dest", ("dest",)),
        ("git worktree move -f wt dest", ("dest",)),
        ("git worktree move --force wt dest", ("dest",)),
        ("git worktree add dest", ("dest",)),
        ("git submodule add ../src dest", ("dest",)),
        ("git submodule add -b main --name nm ../src dest", ("dest",)),
        ("git submodule add --separate-git-dir=gd ../src dest", ("dest", "gd")),
        ("git submodule add ../src", ()),
        ("git worktree move wt", ()),
        ("git worktree prune", ()),
        ("git submodule status", ()),
    ],
)
def test_git_worktree_move_and_submodule_add_destinations_are_write_operands(
    command: str, expected: tuple[str, ...]
) -> None:
    policy = analyze_shell_command_policy(shlex.split(command))
    assert not policy.requires_approval, command
    assert policy.write_path_values == expected, command
