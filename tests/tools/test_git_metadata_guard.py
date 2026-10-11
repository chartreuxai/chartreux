"""Operand-level protection of Git metadata (ADR 0019, P10).

The dangerous-git feature originally guarded only shell REDIRECTS into ``.git``
metadata. These tests pin the general write-operand mechanism: every
write/mutate-classified path operand — of every command except git itself,
whose selectors locate the repository and are governed by the classifier — is
resolved against ALL statically tracked working directories (``cd`` tracking
and executor cwd overrides such as ``env -C``) and denied on git-metadata
containment. Hard-link aliases are contained at link-creation time (the
source operand of ``ln``/``cp -l``/``cp -s``) and, for pre-existing aliases
created outside the policy, at write time by (device, inode) comparison
against the critical metadata files.

Each test reproduces a bypass probed against the shipped redirect-only guard.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A real repository: the protected metadata exists on disk."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    hooks = tmp_path / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    (hooks / "pre-push").write_text("#!/bin/sh\nexit 0\n")
    (tmp_path / "seed.txt").write_text("seed\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.txt").write_text("inner\n")
    return tmp_path


@pytest.fixture
def bash_tool(git_repo: Path) -> Bash:
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    return Bash(config_getter=lambda: config, state=BaseToolState(), cwd=git_repo)


def assert_denied(bash: Bash, command: str) -> None:
    context = bash.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, command
    assert "protected Git metadata" in (context.reason or ""), command


def assert_permitted(bash: Bash, command: str) -> None:
    context = bash.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.ALWAYS, f"{command}: {context.reason}"


# --- Direct operand writes into metadata (absolute and relative spellings) ---


@pytest.mark.parametrize(
    "command",
    [
        # The quiet writers the redirect-only guard never saw.
        "cp seed.txt .git/config",
        "cp seed.txt .git/hooks/pre-push",
        "cp .git/config moved-config",
        "cp -r .git copied-git",
        "mv seed.txt .git/config",
        "mv .git/config moved-config",
        "tee .git/config",
        "dd if=seed.txt of=.git/config",
        "dd of=.git/hooks/pre-push if=seed.txt",
        "install seed.txt .git/hooks/pre-push",
        "install -D seed.txt .git/hooks/nested/pre-push",
        "rsync seed.txt .git/config",
        "rsync -a seed.txt .git/",
        "sed -i s/x/y/ .git/config",
        "sed -i.bak s/x/y/ .git/config",
        "sed --in-place s/x/y/ .git/hooks/pre-push",
        "truncate -s 0 .git/config",
        "patch .git/config fix.diff",
        "unzip evil.zip -d .git",
        "unzip -d .git evil.zip",
        "tar -xf evil.tgz -C .git",
        "tar -C .git -xf evil.tgz",
        "tar -xf evil.tgz --directory .git",
        "tar -xf evil.tgz --directory=.git",
        # mkdir/rm/rmdir/unlink on metadata paths.
        "mkdir .git/hooks2",
        "mkdir -p .git/hooks2/nested",
        "rm .git/config",
        "rm .git/hooks/pre-push",
        "rmdir .git/hooks",
        "unlink .git/config",
        # chmod/chown retarget metadata too.
        "chmod +x .git/hooks/pre-push",
        "chown root .git/config",
        # Compiler output operands.
        "gcc -o .git/hooks/pre-push evil.c",
        "gcc -o.git/hooks/pre-push evil.c",
        "clang --output .git/hooks/pre-push evil.c",
        "cc -o .git/config evil.c",
        # The redirect guard's own class stays denied.
        "echo x > .git/config",
        "echo x >> .git/hooks/pre-push",
    ],
)
def test_direct_metadata_operand_writes_are_denied(
    bash_tool: Bash, command: str
) -> None:
    assert_denied(bash_tool, command)


@pytest.mark.parametrize(
    "command",
    [
        # Absolute spellings of the same writes (the workspace root is the
        # repository, so these stay inside the workspace authority).
        "cp seed.txt {repo}/.git/config",
        "mv seed.txt {repo}/.git/hooks/pre-push",
        "tee {repo}/.git/config",
        "dd if=seed.txt of={repo}/.git/config",
        "install seed.txt {repo}/.git/hooks/pre-push",
        "rsync seed.txt {repo}/.git/config",
        "sed -i s/x/y/ {repo}/.git/config",
        "truncate -s 0 {repo}/.git/config",
        "patch {repo}/.git/config fix.diff",
        "unzip evil.zip -d {repo}/.git",
        "tar -xf evil.tgz -C {repo}/.git",
        "mkdir {repo}/.git/hooks2",
        "rm {repo}/.git/config",
        "rmdir {repo}/.git/hooks",
        "unlink {repo}/.git/config",
        "ln {repo}/.git/config hardlink",
        "ln -s {repo}/.git glink",
        "cp -s {repo}/.git gcp",
        "cp -l {repo}/.git/config ccp",
        "gcc -o {repo}/.git/hooks/pre-push evil.c",
        "echo x > {repo}/.git/config",
    ],
)
def test_absolute_metadata_operand_writes_are_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    assert_denied(bash_tool, command.format(repo=git_repo))


# --- The cwd-relative family: no .git in any operand, no redirect ---


@pytest.mark.parametrize(
    "command",
    [
        # Nine variants: the write happens with cwd inside .git, so no operand
        # spells .git and no redirect exists. The operand guard resolves
        # every write operand against ALL tracked possible cwds.
        "cd .git && cp hooks/pre-push config",
        "cd .git && mv hooks/pre-push config",
        "cd .git && tee config",
        "cd .git && dd if=hooks/pre-push of=config",
        "cd .git && install hooks/pre-push config",
        "cd .git && rsync hooks/pre-push config",
        "cd .git && sed -i s/x/y/ config",
        "cd .git && truncate -s 0 config",
        "cd .git && ln hooks/pre-push config2",
        # Semicolon and nested-shell spellings of the same family.
        "cd .git; cp hooks/pre-push config",
        "sh -c 'cd .git && cp hooks/pre-push config'",
        # cd into metadata through a symlink alias.
        "cd g && cp hooks/pre-push config",
        # pushd into metadata.
        "pushd .git && cp hooks/pre-push config",
    ],
)
def test_cwd_relative_metadata_writes_are_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    (git_repo / "g").symlink_to(git_repo / ".git", target_is_directory=True)
    assert_denied(bash_tool, command)


# --- The env -C (executor cwd override) family ---


@pytest.mark.parametrize(
    "command",
    [
        "env X=1 -C .git cp hooks/pre-push config",
        "env X=1 --chdir .git cp hooks/pre-push config",
        "env X=1 -C .git sed -i s/x/y/ config",
        "env X=1 -C .git mv hooks/pre-push config",
        "env X=1 -C .git tee config",
        "env X=1 -C .git dd if=hooks/pre-push of=config",
        "env X=1 -C .git install hooks/pre-push config",
        "env X=1 -C .git rsync hooks/pre-push config",
        "env X=1 -C .git truncate -s 0 config",
        "env X=1 -C .git ln hooks/pre-push config2",
        "env X=1 -C .git sh -c 'cp hooks/pre-push config'",
        # Through a repo-root symlink: -C r/.git resolves into metadata.
        "env X=1 -C r/.git cp hooks/pre-push config",
    ],
)
def test_executor_cwd_override_metadata_writes_are_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    (git_repo / "r").symlink_to(git_repo, target_is_directory=True)
    assert_denied(bash_tool, command)


# --- Extraction tools: tar -C / unzip -d destinations ---


@pytest.mark.parametrize(
    "command",
    [
        "tar -xf evil.tgz -C .git",
        "tar -xf evil.tgz -C r/.git",
        "tar -xf evil.tgz -C g",
        "tar -xf evil.tgz --directory r/.git",
        "tar -C r/.git -xf evil.tgz",
        "unzip evil.zip -d .git",
        "unzip evil.zip -d r/.git",
        "unzip -d g evil.zip",
    ],
)
def test_extraction_destinations_in_metadata_are_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    (git_repo / "r").symlink_to(git_repo, target_is_directory=True)
    (git_repo / "g").symlink_to(git_repo / ".git", target_is_directory=True)
    assert_denied(bash_tool, command)


# --- Link-creation source containment (mechanism B) ---


@pytest.mark.parametrize(
    "command",
    [
        # ln is a mutating path command: both operands (source and link path)
        # are checked, so an alias into metadata cannot be created.
        "ln .git/config hardlink",
        "ln .git/config /tmp/hardlink-out",
        "ln -s .git glink",
        "ln -s .git/config glink-config",
        "ln -s seed.txt .git/config",
        "ln -sf .git/config hardlink",
        # cp's link-making forms: -s (symlink), -l (hard link), -a/-r (copy).
        "cp -s .git gcp",
        "cp -l .git/config ccp",
        "cp -a .git gcp",
        "cp -r .git gcp",
        "cp --link .git/config ccp",
        "cp --symbolic-link .git gcp",
    ],
)
def test_link_creation_from_metadata_sources_is_denied(
    bash_tool: Bash, command: str
) -> None:
    assert_denied(bash_tool, command)


def test_preexisting_hard_link_write_through_is_denied(
    bash_tool: Bash, git_repo: Path
) -> None:
    # A hard link created OUTSIDE the policy (pre-existing, or made by
    # another process) is invisible to path resolution; the write-time
    # (device, inode) comparison against the critical metadata files catches
    # the write-through. This closes the probed live bypass of the shipped
    # redirect guard (echo x > c with c a hard link to .git/config).
    alias = git_repo / "alias-config"
    os.link(git_repo / ".git" / "config", alias)
    assert alias.stat().st_ino == (git_repo / ".git" / "config").stat().st_ino
    assert_denied(bash_tool, "echo x > alias-config")
    redirect_context = bash_tool.resolve_permission(
        BashArgs(command="echo x > alias-config")
    )
    assert redirect_context is not None
    assert "hard link" in (redirect_context.reason or "")
    assert_denied(bash_tool, "echo x >> alias-config")
    assert_denied(bash_tool, "tee alias-config")
    assert_denied(bash_tool, "cp seed.txt alias-config")
    assert_denied(bash_tool, "truncate -s 0 alias-config")
    assert_denied(bash_tool, "sed -i s/x/y/ alias-config")
    assert_denied(bash_tool, "rm alias-config")
    assert_denied(bash_tool, "mv alias-config elsewhere")
    # A hard link to a worktree file is not metadata: writes stay permitted.
    worktree_alias = git_repo / "seed-alias"
    os.link(git_repo / "seed.txt", worktree_alias)
    assert_permitted(bash_tool, "echo x > seed-alias")


def test_preexisting_symlink_into_metadata_is_denied(
    bash_tool: Bash, git_repo: Path
) -> None:
    # The existing at-use symlink resolution keeps covering later writes
    # through pre-existing symlinks.
    (git_repo / "g").symlink_to(git_repo / ".git", target_is_directory=True)
    assert_denied(bash_tool, "echo x > g/config")
    assert_denied(bash_tool, "cp seed.txt g/hooks/pre-push")
    assert_denied(bash_tool, "tee g/config")


def test_hard_link_to_hook_is_denied_at_write_time(
    bash_tool: Bash, git_repo: Path
) -> None:
    hook_alias = git_repo / "hook-alias"
    os.link(git_repo / ".git" / "hooks" / "pre-push", hook_alias)
    assert_denied(bash_tool, "echo x > hook-alias")
    assert_denied(bash_tool, "chmod +x hook-alias")


# --- Containment beyond the literal ".git" spelling (gitfile layouts) ---


@pytest.fixture
def gitfile_repo(tmp_path: Path) -> Path:
    """A repository whose git dir is not named .git (a gitfile pointer)."""
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config").write_text("[core]\n\trepositoryformatversion = 0\n")
    (tmp_path / ".git").write_text("gitdir: metadata\n")
    (tmp_path / "seed.txt").write_text("seed\n")
    return tmp_path


@pytest.fixture
def gitfile_tool(gitfile_repo: Path) -> Bash:
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    return Bash(config_getter=lambda: config, state=BaseToolState(), cwd=gitfile_repo)


@pytest.mark.parametrize(
    "command",
    [
        # No ".git" component in any operand: containment comes from
        # git_metadata_paths resolving the gitfile pointer.
        "cp seed.txt metadata/config",
        "mv metadata/config moved-config",
        "tee metadata/config",
        "rm metadata/config",
        "sed -i s/x/y/ metadata/config",
        "truncate -s 0 metadata/config",
        "tar -xf evil.tgz -C metadata",
        "unzip evil.zip -d metadata",
        "ln metadata/config hardlink",
        "mkdir metadata/hooks2",
        "cd metadata && cp seed.txt config",
        "env X=1 -C metadata cp seed.txt config",
        "echo x > metadata/config",
    ],
)
def test_gitfile_layout_metadata_writes_are_denied(
    gitfile_tool: Bash, command: str
) -> None:
    assert_denied(gitfile_tool, command)


# --- Fail closed ---


def test_unknown_cwd_write_operand_fails_closed(bash_tool: Bash) -> None:
    # An operand whose working directory is not statically known is denied
    # before any metadata check can wave it through.
    context = bash_tool.resolve_permission(
        BashArgs(command="cd - && cp seed.txt out.txt")
    )
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "not statically known" in (context.reason or "")


def test_session_rooted_in_metadata_denies_relative_operand_writes(
    git_repo: Path,
) -> None:
    # Isolation for the tracked-cwd resolution: with the session itself rooted
    # in the git dir there is no cd and no executor selector, so the denial of
    # a bare relative write can only come from resolving the write operand
    # against the tracked working directory and checking containment.
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    tool = Bash(
        config_getter=lambda: config, state=BaseToolState(), cwd=git_repo / ".git"
    )
    assert_denied(tool, "cp hooks/pre-push config")
    assert_denied(tool, "mv hooks/pre-push moved-hook")
    assert_denied(tool, "tee config")
    assert_denied(tool, "sed -i s/x/y/ config")
    assert_denied(tool, "rm config")
    assert_denied(tool, "echo x > config")
    assert_denied(tool, "echo x > hooks/pre-push")
    assert_denied(tool, "echo x > out.txt")
    assert_denied(tool, "cp config backup-config")
    # Read-classified operands stay governed by the workspace authority.
    assert_permitted(tool, "cat config")
    assert_permitted(tool, "ls hooks")


# --- Benign equivalents stay permitted ---


@pytest.mark.parametrize(
    "command",
    [
        "cp seed.txt copy.txt",
        "cp -r sub copied-sub",
        "mv seed.txt moved.txt",
        "tee copy.txt",
        "dd if=seed.txt of=copy.txt",
        "install seed.txt installed.txt",
        "rsync seed.txt copy.txt",
        "sed -i s/x/y/ seed.txt",
        "sed -i.bak s/x/y/ seed.txt",
        "truncate -s 0 seed.txt",
        "patch seed.txt fix.diff",
        "unzip evil.zip -d outdir",
        "tar -xf evil.tgz -C outdir",
        "tar -cf out.tar seed.txt",
        "tar -xf evil.tgz --directory outdir",
        "mkdir newdir",
        "rm seed.txt",
        "rmdir newdir",
        "unlink seed.txt",
        "ln seed.txt hardlink",
        "ln -s seed.txt symlink",
        "cp -s seed.txt cpsym",
        "cp -l seed.txt cplink",
        "gcc -o tool evil.c",
        "gcc -otool evil.c",
        "clang --output tool evil.c",
        "echo x > out.txt",
        "cd sub && cp inner.txt copy.txt",
        "env X=1 -C sub cp inner.txt copy.txt",
        "env X=1 -C sub sh -c 'cp inner.txt copy.txt'",
        # git's own selectors locate the repository and stay governed by the
        # classifier/repository-config flows, not the operand guard.
        "git status",
        "git -C .git status",
        "git --git-dir=.git log",
        "git --work-tree=. --git-dir=.git diff",
        # Routine config writes stay permitted in a real repository.
        "git config user.name X",
        "git config --get user.name",
        "git config --rename-section other.one other.two",
        "git config --remove-section core.old",
        # Read-classified operands are not write targets: reading metadata
        # with cat/ls stays governed by the workspace authority alone.
        "cat .git/config",
        "cat seed.txt",
        "ls .git",
    ],
)
def test_benign_equivalents_stay_permitted(bash_tool: Bash, command: str) -> None:
    assert_permitted(bash_tool, command)


def test_benign_extraction_into_scratchpad_stays_permitted(
    git_repo: Path, tmp_path: Path
) -> None:
    # A scratchpad-configured tool may extract into its scratchpad.
    scratchpad = tmp_path / "scratch"
    scratchpad.mkdir()
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    tool = Bash(
        config_getter=lambda: config,
        state=BaseToolState(),
        cwd=git_repo,
        scratchpad_dir=scratchpad,
    )
    assert_permitted(tool, f"tar -xf evil.tgz -C {scratchpad}")
    assert_permitted(tool, f"cp seed.txt {scratchpad}/copy.txt")
    assert_permitted(tool, "mkdir outdir && tar -xf evil.tgz -C outdir")


# --- Attached-option write operands (delta review, family 2) ---
#
# The positional scan skips option tokens, so an ATTACHED option value
# (--target-directory=.git, -C.git, -out=.git/config) used to bypass the
# guard entirely. Every write operand of a guarded command is now extracted
# in separate, attached-short, and attached-long spellings alike.


def assert_never(bash: Bash, command: str) -> None:
    context = bash.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"


@pytest.mark.parametrize(
    "command",
    [
        # cp/mv/install/ln: --target-directory= and the attached short -t<dir>.
        "cp --target-directory=.git seed.txt",
        "cp --target-directory=.git/hooks pre-commit",
        "cp -t.git seed.txt",
        "mv --target-directory=.git seed.txt",
        "mv -t.git seed.txt",
        "install --target-directory=.git seed.txt",
        "install -t.git seed.txt",
        "ln --target-directory=.git seed.txt",
        "ln -t.git seed.txt",
        # tar: the attached short -C<dir> (also inside a short cluster), the
        # archive operand -f<archive>/--file=, and --directory=.
        "tar -C.git -xf evil.tgz",
        "tar -xf evil.tgz -C.git",
        "tar -xC.git -f evil.tgz",
        "tar -C .git/hooks -xf evil.tgz",
        "tar -cf .git/config seed.txt",
        "tar -cf.git/config seed.txt",
        "tar --file=.git/config -c seed.txt",
        "tar -C.git -cf out.tar seed.txt",
        # unzip: the attached short -d<dir>.
        "unzip -d.git evil.zip",
        "unzip evil.zip -d.git",
        "unzip -d.git/hooks evil.zip",
        # patch: --output=/-o<val>, --reject-file=/-r<val>, the relocated
        # -d/--directory, and the backup-name prefixes.
        "patch --output=.git/config fix.diff",
        "patch -o.git/config fix.diff",
        "patch --reject-file=.git/rej fix.diff",
        "patch -r.git/rej fix.diff",
        "patch -d.git fix.diff",
        "patch --directory=.git fix.diff",
        "patch --prefix=.git/ -b fix.diff",
        "patch -B.git/ -b fix.diff",
        # openssl: -out= and --out= (single- and double-dash spellings).
        "openssl enc -aes-256-cbc -out=.git/config -in seed.txt",
        "openssl enc -aes-256-cbc --out=.git/config -in seed.txt",
        # iconv: -o<val> and --output=.
        "iconv -f UTF-8 -t UTF-8 -o.git/config seed.txt",
        "iconv -f UTF-8 -t UTF-8 --output=.git/config seed.txt",
        # rsync: the attached write-directory options and -T<dir>.
        "rsync --temp-dir=.git seed.txt out.txt",
        "rsync -T.git seed.txt out.txt",
        "rsync --partial-dir=.git seed.txt out.txt",
        "rsync --backup-dir=.git seed.txt out.txt",
        "rsync --compare-dest=.git seed.txt out.txt",
        "rsync --copy-dest=.git seed.txt out.txt",
        "rsync --link-dest=.git seed.txt out.txt",
        "rsync --log-file=.git/log seed.txt out.txt",
        # tee: the attached --output-error= (GNU's error MODE, extracted
        # conservatively as a write operand).
        "tee --output-error=.git/config seed.txt",
        # split: the positional PREFIX names the written output files.
        "split -b 100 seed.txt .git/x",
        "split -b 100 seed.txt .git/hooks/pre-push",
        "split --bytes=100 seed.txt .git/x",
        "split seed.txt .git/x",
        "split -b 100 seed.txt .git/config",
        # csplit: --prefix= and the attached short -f<prefix>.
        "csplit --prefix=.git/cracked seed.txt 1",
        "csplit -f.git/cracked seed.txt 1",
        "csplit -f .git/cracked seed.txt 1",
    ],
)
def test_attached_option_metadata_writes_are_denied(
    bash_tool: Bash, command: str
) -> None:
    assert_denied(bash_tool, command)


@pytest.mark.parametrize(
    "command",
    [
        # The same attached spellings into ordinary in-workspace destinations.
        "cp --target-directory=outdir seed.txt",
        "cp -toutdir seed.txt",
        "mv --target-directory=outdir seed.txt",
        "install --target-directory=outdir seed.txt",
        "ln --target-directory=outdir seed.txt",
        "tar -C outdir -xf evil.tgz",
        "tar -Coutdir -xf evil.tgz",
        "tar -cf out.tar seed.txt",
        "tar -cfout.tar seed.txt",
        "tar --file=out.tar -c seed.txt",
        "tar -xf evil.tgz --directory=outdir",
        "unzip -doutdir evil.zip",
        "unzip -d outdir evil.zip",
        "patch --output=patched.txt fix.diff",
        "patch -opatched.txt fix.diff",
        "patch --reject-file=rej.txt fix.diff",
        "patch -d sub fix.diff",
        "patch --prefix=bak/ -b fix.diff",
        "openssl enc -aes-256-cbc -out=enc.bin -in seed.txt",
        "openssl enc -aes-256-cbc --out=enc.bin -in seed.txt",
        "iconv -f UTF-8 -t UTF-8 -oconv.txt seed.txt",
        "iconv -f UTF-8 -t UTF-8 --output=conv.txt seed.txt",
        "rsync --temp-dir=tmpdir seed.txt out.txt",
        "rsync -T tmpdir seed.txt out.txt",
        "rsync --backup-dir=backups seed.txt out.txt",
        "rsync --log-file=rsync.log seed.txt out.txt",
        "tee --output-error=warn copy.txt",
        "tee --output-error=exit-nopipe copy.txt",
        "split -b 100 seed.txt out",
        "split -b 100 seed.txt",
        "split --bytes=100 seed.txt out2",
        # The precise split modeling keeps option operands (a SIZE, a
        # SEPARATOR) out of the candidates entirely.
        "split -t / seed.txt out3",
        "csplit --prefix=out seed.txt 1",
        "csplit -f out2 seed.txt 1",
        "csplit seed.txt 1",
    ],
)
def test_attached_option_benign_writes_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    assert_permitted(bash_tool, command)


# --- Git raw-write operands (delta review, family 1) ---
#
# git's -C/--git-dir/--work-tree selectors LOCATE the repository and stay
# governed by the classifier and repository-config flows, but git's RAW
# WRITE operands — archive --output/-o and a writing config --file/-f — are
# governed like any other write operand.


@pytest.mark.parametrize(
    "command",
    [
        # git archive --output/-o writes the named file directly.
        "git archive --output=.git/config HEAD",
        "git archive --output .git/config HEAD",
        "git archive --output=.git/hooks/pre-push HEAD",
        "git archive -o .git/config HEAD",
        "git archive -o.git/config HEAD",
        "git archive --output=.git/config --format=tar HEAD",
        # git config --file/-f names the written file of a writing call.
        "git config --file=.git/config user.name x",
        "git config --file .git/config user.name x",
        "git config -f .git/config user.name x",
        "git config -f.git/config user.name x",
        "git config --file=.git/config set user.name x",
        "git config --file=.git/config --unset user.name",
        "git config --file=.git/config --add user.name x",
        "git config --file=.git/config --rename-section a b",
        "git config --file=.git/config --remove-section core.old",
        "git config --file=.git/config --edit",
        "git config --file=.git/config.worktree core.pager evil",
        # The classifier still guards the dangerous KEYS in every scope.
        "git config --file=.git/config core.hooksPath /tmp/h",
        "git config --file=.git/config remote.origin.url /tmp/evil",
    ],
)
def test_git_raw_write_operands_into_metadata_are_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    assert_denied(bash_tool, command.format(repo=git_repo))


@pytest.mark.parametrize(
    "command",
    [
        # Absolute spellings of the same writes.
        "git archive --output={repo}/.git/config HEAD",
        "git archive -o {repo}/.git/hooks/pre-push HEAD",
        "git config --file={repo}/.git/config user.name x",
        "git config -f{repo}/.git/config user.name x",
        # Outside-workspace destinations: the authority denies them even
        # though no metadata is involved.
        "git archive --output=/etc/cron.d/pwn HEAD",
        "git archive --output=../outside.tar HEAD",
        "git config --file=/etc/evil user.name x",
        "git config --file=../evil.cfg user.name x",
    ],
)
def test_git_raw_write_operands_outside_authority_are_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    assert_never(bash_tool, command.format(repo=git_repo))


@pytest.mark.parametrize(
    "command",
    [
        # In-workspace destinations stay permitted, in every spelling.
        "git archive --output=bundle.tar HEAD",
        "git archive --output bundle.tar HEAD",
        "git archive -o bundle.tar HEAD",
        "git archive -obundle.tar HEAD",
        "git config --file=custom.cfg user.name X",
        "git config --file custom.cfg user.name X",
        "git config -f custom.cfg user.name X",
        "git config -fcustom.cfg user.name X",
        "git config --file=custom.cfg --get user.name",
        # A read names a file git only reads: authority-scoped, not
        # metadata-guarded, exactly like cat .git/config.
        "git config --file=.git/config --get user.name",
        "git config --file=.git/config user.name",
        "git config --file=.git/config --list",
        # The bare-key read form of the same destination.
        "git config --file .git/config core.pager",
    ],
)
def test_git_raw_write_operand_benign_forms_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    assert_permitted(bash_tool, command)


def test_git_write_operands_into_scratchpad_stay_permitted(
    git_repo: Path, tmp_path: Path
) -> None:
    # A scratchpad-configured tool may archive and configure into it.
    scratchpad = tmp_path / "scratch"
    scratchpad.mkdir()
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    tool = Bash(
        config_getter=lambda: config,
        state=BaseToolState(),
        cwd=git_repo,
        scratchpad_dir=scratchpad,
    )
    assert_permitted(tool, f"git archive --output={scratchpad}/bundle.tar HEAD")
    assert_permitted(tool, f"git archive -o {scratchpad}/bundle.tar HEAD")
    assert_permitted(tool, f"git config --file={scratchpad}/cfg user.name X")
    assert_permitted(tool, f"git config --file={scratchpad}/cfg --get user.name")


# --- Git work-tree relocation (delta review, family 1) ---
#
# A work tree inside the git metadata dir is never legitimate: routine
# tree-writing subcommands then write tracked (or patched) paths beneath it,
# which is the probed routine ACE chain that planted an executable hook.


@pytest.mark.parametrize(
    "command",
    [
        # -C into the metadata dir: the cwd is the git dir, so git uses it
        # as the work tree.
        "git -C .git apply ../hook.diff",
        "git -C .git merge evil",
        "git -C .git/hooks am hook.diff",
        "git -C .git stash pop",
        "git -C .git switch main",
        "git -C .git checkout main",
        "git -C .git restore seed.txt",
        "git -C .git cherry-pick evil",
        "git -C .git revert evil",
        "git -C .git rebase main",
        "git -C .git pull",
        # --work-tree/--git-dir relocation into the metadata dir.
        "git --git-dir=.git --work-tree=.git merge evil",
        "git --git-dir=.git --work-tree=.git cherry-pick evil",
        "git --git-dir=.git --work-tree=.git/hooks apply hook.diff",
        "git --git-dir=.git --work-tree=.git stash pop",
        "git --git-dir=.git --work-tree=.git switch main",
        "git --git-dir=.git --work-tree=.git checkout main",
        "git --git-dir=.git --work-tree=.git restore seed.txt",
        "git --git-dir=.git --work-tree=.git rebase main",
        "git --git-dir=.git --work-tree=.git revert evil",
        "git --git-dir=.git --work-tree=.git am hook.diff",
        "git --git-dir=.git --work-tree=.git pull",
        "git --work-tree=.git merge evil",
        "git --work-tree=.git/hooks apply hook.diff",
        # Through a symlink alias into the metadata dir.
        "git -C g apply hook.diff",
    ],
)
def test_git_work_tree_relocation_into_metadata_is_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    (git_repo / "g").symlink_to(git_repo / ".git", target_is_directory=True)
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    assert "work tree" in (context.reason or ""), command


@pytest.mark.parametrize(
    "command",
    [
        # The shell cwd family: the cd operand itself reaches metadata, so
        # the pre-existing operand guard denies the chain at the cd segment.
        "cd .git && git apply ../hook.diff",
        "cd .git && git merge evil",
        "cd .git/hooks && git apply ../../hook.diff",
    ],
)
def test_git_tree_writer_after_cd_into_metadata_is_denied(
    bash_tool: Bash, command: str
) -> None:
    assert_denied(bash_tool, command)


def test_git_work_tree_relocation_executor_override_is_denied(
    bash_tool: Bash, git_repo: Path
) -> None:
    # An executor cwd override into metadata is denied: the override itself
    # is a write-classified candidate of the wrapping command.
    (git_repo / "g").symlink_to(git_repo / ".git", target_is_directory=True)
    assert_denied(bash_tool, "env X=1 -C .git git apply ../hook.diff")
    assert_denied(bash_tool, "env X=1 -C g git merge evil")


@pytest.mark.parametrize(
    "command",
    [
        # Absolute spellings of the relocation.
        "git --git-dir={repo}/.git --work-tree={repo}/.git merge evil",
        "git -C {repo}/.git apply ../hook.diff",
    ],
)
def test_git_work_tree_relocation_absolute_spellings_are_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    context = bash_tool.resolve_permission(
        BashArgs(command=command.format(repo=git_repo))
    )
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, command
    assert "work tree" in (context.reason or ""), command


@pytest.mark.parametrize(
    "command",
    [
        # Routine readers rooted in the metadata dir stay permitted: they
        # never write work-tree files.
        "git -C .git status",
        "git -C .git log",
        "git --git-dir=.git log",
        "git --work-tree=. --git-dir=.git diff",
        "git --git-dir=.git --work-tree=.git status",
        # In-worktree tree-writing calls stay permitted.
        "git apply fix.diff",
        "git apply hook.diff",
        "git merge evil",
        "git cherry-pick evil",
        "git revert evil",
        "git rebase main",
        "git am hook.diff",
        "git pull",
        "git stash pop",
        "git switch main",
        # fetch never writes the work tree.
        "git fetch origin",
        # Legitimate relocations: the work tree stays outside the metadata.
        "git -C sub apply ../fix.diff",
        "git --git-dir=.git --work-tree=wt merge evil",
        "git --git-dir=.git --work-tree=sub apply fix.diff",
        "git --work-tree=wt merge evil",
        "git --work-tree=sub apply fix.diff",
        "git -C . merge evil",
        "git -C sub merge evil",
        # A git dir spelled through a symlink with the normal work tree.
        "git --git-dir=g --work-tree=. merge evil",
    ],
)
def test_git_work_tree_benign_forms_stay_permitted(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    (git_repo / "g").symlink_to(git_repo / ".git", target_is_directory=True)
    assert_permitted(bash_tool, command)


def test_git_selector_authority_checks_keep_working(bash_tool: Bash) -> None:
    # A selector outside the workspace is still denied by the path authority.
    context = bash_tool.resolve_permission(BashArgs(command="git -C /etc status"))
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "outside" in (context.reason or "")
    context = bash_tool.resolve_permission(BashArgs(command="git -C /etc merge evil"))
    assert context is not None
    assert context.permission is ToolPermission.NEVER


def test_core_worktree_relocation_into_metadata_is_denied(
    bash_tool: Bash, git_repo: Path
) -> None:
    # core.worktree relocates the work tree without any command-line
    # selector; a relative value resolves against the git dir, so it lands
    # inside the metadata.
    config = git_repo / ".git" / "config"
    config.write_text("[core]\n\tworktree = hooks\n")
    assert_denied(bash_tool, "git merge evil")
    assert_denied(bash_tool, "git apply fix.diff")
    assert_denied(bash_tool, "git stash pop")
    # An absolute core.worktree outside the workspace authority is denied.
    config.write_text("[core]\n\tworktree = /etc\n")
    assert_never(bash_tool, "git merge evil")
    # A core.worktree outside the metadata stays permitted.
    worktree = git_repo / "wt"
    worktree.mkdir()
    config.write_text(f"[core]\n\tworktree = {worktree}\n")
    assert_permitted(bash_tool, "git merge evil")
    assert_permitted(bash_tool, "git apply fix.diff")


def test_git_work_tree_relocation_fails_closed_on_unknown_cwd(bash_tool: Bash) -> None:
    # A tree-writing git call whose working directory is not statically
    # known cannot have its work tree verified.
    context = bash_tool.resolve_permission(BashArgs(command="cd - && git merge evil"))
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "not statically known" in (context.reason or "")


# --- git apply --directory write roots and --unsafe-paths (re-review, round 2) ---
#
# git apply --directory relocates the patch's write root: every patched path
# (after -p stripping) lands beneath it, so the value is a write-classified
# operand, denied on metadata containment and outside-workspace authority.
# --unsafe-paths disables git's own path refusal entirely (it is what turns
# `--directory=.git/hooks` into a routine end-to-end ACE), so it is a hard
# guard with no grantable operation.


@pytest.mark.parametrize(
    "command",
    [
        "git apply --directory=.git/hooks hook.diff",
        "git apply --directory=.git/hooks/ hook.diff",
        "git apply --directory=./.git/hooks hook.diff",
        "git apply --directory=.git/../.git/hooks hook.diff",
        "git apply --directory .git/hooks hook.diff",
        "git apply -p0 --directory=.git/hooks hook.diff",
        "git apply --directory=.git/config fix.diff",
        "git -C sub apply --directory=../.git/hooks ../hook.diff",
        "git --git-dir=.git --work-tree=. apply --directory=.git/hooks hook.diff",
        # Through a symlink alias into the metadata dir.
        "git apply --directory=g/hooks hook.diff",
    ],
)
def test_git_apply_directory_write_roots_in_metadata_are_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    (git_repo / "g").symlink_to(git_repo / ".git", target_is_directory=True)
    assert_denied(bash_tool, command)


@pytest.mark.parametrize(
    "command",
    [
        "git apply --directory={repo}/.git/hooks hook.diff",
        "git apply --directory={repo}/.git/config fix.diff",
    ],
)
def test_git_apply_directory_absolute_write_roots_are_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    assert_denied(bash_tool, command.format(repo=git_repo))


@pytest.mark.parametrize(
    "command",
    [
        # Outside-workspace write roots: the authority denies them even
        # though no metadata is involved (git itself refuses these paths
        # without --unsafe-paths; the operand guard denies them regardless).
        "git apply --directory=/tmp/evil-hook hook.diff",
        "git apply --directory=../outside hook.diff",
        "git apply --directory=../../../../tmp/escape hook.diff",
    ],
)
def test_git_apply_directory_outside_authority_is_denied(
    bash_tool: Bash, command: str
) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    assert "outside" in (context.reason or ""), command


@pytest.mark.parametrize(
    "command",
    [
        "git apply --unsafe-paths hook.diff",
        "git apply --unsafe-paths --directory=.git/hooks hook.diff",
        "git apply --directory=.git/hooks --unsafe-paths hook.diff",
        "git apply -p0 --unsafe-paths hook.diff",
        "git apply --unsafe-paths --directory=/tmp/evil hook.diff",
    ],
)
def test_git_apply_unsafe_paths_is_a_hard_guard(bash_tool: Bash, command: str) -> None:
    # A hard guard: denied with no grantable proposal, so no answer can
    # authorize the call.
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert "--unsafe-paths" in (context.reason or ""), command


@pytest.mark.parametrize(
    "command",
    [
        "git apply fix.diff",
        "git apply --directory=patches fix.diff",
        "git apply --exclude=.git/* hook.diff",
        "git apply --include=seed.txt hook.diff",
        "git apply --build-fake-ancestor=.git/foo hook.diff",
        # git am supports neither --unsafe-paths (git rejects the option)
        # nor a metadata-reaching --directory (git rejects the path), so
        # the spelling stays with git's own refusal.
        "git am --directory=.git/hooks hook.diff",
    ],
)
def test_git_apply_benign_forms_stay_permitted(bash_tool: Bash, command: str) -> None:
    assert_permitted(bash_tool, command)


# --- Git's own -C relocation of raw write operands (re-review, round 2) ---
#
# Git applies its -C sequence before the subcommand runs, so a relative raw
# write operand resolves against the RELOCATED directory: `git -C .git
# archive -o config HEAD` writes .git/config. The operand guard now resolves
# git's write operands against that -C chain, not just the shell's tracked
# cwds.


@pytest.mark.parametrize(
    "command",
    [
        "git -C .git archive -o config HEAD",
        "git -C .git archive --output=config HEAD",
        "git -C .git archive -o hooks/pre-commit HEAD",
        "git -C .git/hooks archive -o pre-commit HEAD",
        "git -C .git archive -o index HEAD",
        "git -C .git config --file=config user.name x",
        "git -C .git config --file=hooks/pre-commit user.name x",
        "git -C .git config -f config --unset user.name",
        "git -C sub archive -o ../.git/config HEAD",
        "git -C sub config --file=../.git/config user.name x",
        "git -C g archive -o config HEAD",
        "git -C . -C .git archive -o config HEAD",
    ],
)
def test_git_c_relocated_write_operands_in_metadata_are_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    (git_repo / "g").symlink_to(git_repo / ".git", target_is_directory=True)
    assert_denied(bash_tool, command)


@pytest.mark.parametrize(
    "command",
    [
        # Legitimate relocations: the write lands outside the metadata.
        "git -C sub config --file=cfg user.name x",
        "git -C sub config --file=../custom.cfg user.name x",
        "git -C sub archive -o bundle.tar HEAD",
        "git -C sub archive -o ../bundle.tar HEAD",
        "git -C . -C . archive -o bundle.tar HEAD",
        "git -C . status",
        "git -C sub status",
        "git -C sub log --oneline",
    ],
)
def test_git_c_relocated_benign_forms_stay_permitted(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    assert_permitted(bash_tool, command)


def test_git_c_relocated_write_operands_to_scratchpad_stay_permitted(
    git_repo: Path, tmp_path: Path
) -> None:
    # A scratchpad-configured tool may archive into its scratchpad even
    # through a metadata-relocating -C: the write lands in the scratchpad,
    # not in the metadata.
    scratchpad = tmp_path / "scratch"
    scratchpad.mkdir()
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    tool = Bash(
        config_getter=lambda: config,
        state=BaseToolState(),
        cwd=git_repo,
        scratchpad_dir=scratchpad,
    )
    assert_permitted(tool, f"git -C .git archive -o {scratchpad}/bundle.tar HEAD")
    assert_permitted(tool, f"git -C .git config --file={scratchpad}/cfg user.name X")


# --- Unmodeled file commands on the write/mutate surface (re-review, round 2) ---
#
# cpio, pax, bsdtar, ar, shred, and fallocate joined the inspected
# write/mutate surface: their write-classified operands get the same
# metadata containment as every other command.


@pytest.mark.parametrize(
    "command",
    [
        # cpio: the pass-through destination positional, the -D directory
        # copy-in changes into before extracting, and the copy-out archive
        # files are all write operands.
        "cpio -p .git/hooks",
        "cpio -p .git",
        "cpio --pass-through .git/hooks",
        "cpio -l .git/hooks",
        "printf 'pre-commit\\n' | cpio -p .git/hooks",
        "cpio -i -D .git",
        "cpio -i --directory=.git",
        "cpio -i -D .git/hooks",
        "cpio -o -F .git/config",
        "cpio -o --file=.git/config",
        "cpio -o -O .git/config",
        # shred and fallocate mutate their positional file in place.
        "shred .git/config",
        "shred -n 1 .git/config",
        "shred -u .git/config",
        "shred --remove .git/config",
        "fallocate -l 100 .git/config",
        "fallocate --length=100 .git/config",
        "fallocate -p -l 100 .git/config",
        # ar: the archive operand is created or replaced, and --output
        # names the extraction destination.
        "ar rcs .git/evil.a x.o",
        "ar rcs .git/hooks/pre-commit x.o",
        "ar x --output=.git lib.a",
        "ar x --output .git lib.a",
        # pax/bsdtar: tar-family archive operands and extraction
        # destinations.
        "pax -w -f .git/bundle.tar seed.txt",
        "pax -wf .git/bundle.tar seed.txt",
        "pax -r -f .git/config",
        "bsdtar -cf .git/bundle.tar seed.txt",
        "bsdtar -C .git -xf evil.tgz",
        "bsdtar --directory=.git -xf evil.tgz",
    ],
)
def test_unmodeled_file_command_metadata_writes_are_denied(
    bash_tool: Bash, command: str
) -> None:
    assert_denied(bash_tool, command)


@pytest.mark.parametrize(
    "command",
    [
        "cpio -p outdir",
        "cpio -i -D outdir",
        "cpio -o -F out.cpio",
        "cpio -t",
        "shred seed.txt",
        "shred -u seed.txt",
        "fallocate -l 100 seed.txt",
        "ar rcs lib.a x.o",
        "ar t lib.a",
        "pax -w -f out.tar seed.txt",
        "pax -r -f out.tar",
        "bsdtar -cf out.tar seed.txt",
        "bsdtar -C outdir -xf good.tgz",
    ],
)
def test_unmodeled_file_command_benign_forms_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    assert_permitted(bash_tool, command)


def test_unmodeled_file_commands_to_scratchpad_stay_permitted(
    git_repo: Path, tmp_path: Path
) -> None:
    # A scratchpad-configured tool may run the newly inspected commands
    # against its scratchpad.
    scratchpad = tmp_path / "scratch"
    scratchpad.mkdir()
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    tool = Bash(
        config_getter=lambda: config,
        state=BaseToolState(),
        cwd=git_repo,
        scratchpad_dir=scratchpad,
    )
    assert_permitted(tool, f"cpio -p {scratchpad}/hooks")
    assert_permitted(tool, f"shred {scratchpad}/junk.bin")
    assert_permitted(tool, f"fallocate -l 100 {scratchpad}/blob")
    assert_permitted(tool, f"ar rcs {scratchpad}/lib.a x.o")


# --- git init / git worktree add positional destinations (re-review, round 2) ---
#
# The positional destination of git init (plus --separate-git-dir's git dir)
# and of git worktree add names a directory the call CREATES, so it is a
# write-classified operand: destinations resolving into protected metadata
# are denied. Benign in-workspace and scratchpad destinations stay as-is.


@pytest.mark.parametrize(
    "command",
    [
        "git init .git/evil",
        "git init .git/hooks/evil",
        "git init --separate-git-dir=.git/evil newdir",
        "git init --separate-git-dir .git/evil newdir",
        "git worktree add .git/wt2",
        "git worktree add .git/hooks/wt",
        "git worktree add -b wtb .git/wt2",
        "git worktree add --detach .git/wt2",
        "git worktree add --lock --reason why .git/wt2",
        "git -C sub worktree add ../.git/wt2",
        "git -C sub init ../.git/evil",
    ],
)
def test_git_init_and_worktree_destinations_in_metadata_are_denied(
    bash_tool: Bash, command: str
) -> None:
    assert_denied(bash_tool, command)


@pytest.mark.parametrize(
    "command", ["git init {repo}/.git/evil", "git worktree add {repo}/.git/wt2"]
)
def test_git_init_and_worktree_absolute_destinations_are_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    assert_denied(bash_tool, command.format(repo=git_repo))


@pytest.mark.parametrize(
    "command",
    [
        # Outside-workspace destinations: the write-operand authority denies
        # them, exactly like every other write operand.
        "git init /tmp/evil-repo",
        "git init ../outside-repo",
        "git worktree add /tmp/evil-wt",
        "git worktree add ../outside-wt",
    ],
)
def test_git_init_and_worktree_outside_authority_are_denied(
    bash_tool: Bash, command: str
) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    assert "outside" in (context.reason or ""), command


@pytest.mark.parametrize(
    "command",
    [
        # No positional: git init creates .git in the cwd, which stays
        # permitted (the guard protects existing metadata, not fresh
        # repositories).
        "git init",
        "git init newdir",
        "git init -q newdir",
        "git init --bare newdir",
        "git init --shared newdir",
        "git init --shared=group newdir",
        "git init -b trunk newdir",
        "git init --template=/usr/share/git-core/templates newdir",
        "git worktree add wt2",
        "git worktree add -b wtb wt2",
        "git worktree add --detach wt2",
        "git worktree add --lock --reason why wt2",
        "git worktree add --orphan wt2",
        "git worktree list",
        # git clone keeps its existing treatment (not a metadata destination).
        "git clone -q . /tmp/x-clone",
    ],
)
def test_git_init_and_worktree_benign_forms_stay_permitted(
    bash_tool: Bash, command: str
) -> None:
    assert_permitted(bash_tool, command)


def test_git_init_and_worktree_to_scratchpad_stay_permitted(
    git_repo: Path, tmp_path: Path
) -> None:
    # A scratchpad-configured tool may create repositories and worktrees in
    # its scratchpad.
    scratchpad = tmp_path / "scratch"
    scratchpad.mkdir()
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    tool = Bash(
        config_getter=lambda: config,
        state=BaseToolState(),
        cwd=git_repo,
        scratchpad_dir=scratchpad,
    )
    assert_permitted(tool, f"git init {scratchpad}/newrepo")
    assert_permitted(tool, f"git worktree add {scratchpad}/wt")


# --- Round-3 closures (P10 final enumeration round) ---
#
# The same-family gaps the closure check named: binutils/assembler output
# operands (exec-bit-preserving hook planters), config-content writers
# (full .git/config content control), in-place compression mutators,
# archive creation/extraction destinations, mktemp, scp local-to-local
# copies, and the worktree-move/submodule-add destinations. Grammar facts
# verified against the shipped tools (binutils 2.46, Info-ZIP zip 3.0,
# 7-Zip 26.00, unar 1.10.8, xz-utils 5.8.3, zstd 1.5.7, uutils mktemp
# 0.8.0, OpenSSH scp 10.2p1, GnuPG 2.4.8, libxml2 xmllint 2.15.2).


@pytest.mark.parametrize(
    "command",
    [
        # strip/objcopy/as write operands: strip and objcopy preserve the
        # executable bit, so -o/positional outputs plant hooks exactly
        # like the modeled ld -o.
        "strip -o .git/hooks/pre-push evil",
        "strip -o .git/config evil",
        "strip -o.git/config evil",
        "strip .git/config",
        "objcopy evil .git/hooks/pre-push",
        "objcopy --strip-all evil .git/config",
        "objcopy .git/config",
        "as -o .git/config evil.s",
        "as -o.git/config evil.s",
        # Config-content writers: gpg/xmllint output operands give full
        # .git/config content control; scp local-to-local overwrites it in
        # both directions (the ln rule).
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
        # In-place compression mutators destroy or replace the config.
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
        # mktemp creates files at its template/-p/--tmpdir operands.
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
    ],
)
def test_round3_metadata_writes_are_denied(bash_tool: Bash, command: str) -> None:
    # The git_repo fixture already provides the sub/ directory.
    assert_denied(bash_tool, command)


@pytest.mark.parametrize(
    "command",
    [
        "git worktree move wt {repo}/.git/wt2",
        "git submodule add ../src {repo}/.git/mod",
    ],
)
def test_round3_absolute_metadata_destinations_are_denied(
    bash_tool: Bash, git_repo: Path, command: str
) -> None:
    assert_denied(bash_tool, command.format(repo=git_repo))


@pytest.mark.parametrize(
    "command",
    [
        # Outside-workspace destinations: the write-operand authority denies
        # them, exactly like every other write operand.
        "git worktree move wt /tmp/outside-wt",
        "git worktree move wt ../outside-wt",
        "git submodule add ../src /tmp/outside-mod",
        "git submodule add ../src ../outside-mod",
    ],
)
def test_round3_outside_authority_destinations_are_denied(
    bash_tool: Bash, command: str
) -> None:
    context = bash_tool.resolve_permission(BashArgs(command=command))
    assert context is not None, command
    assert context.permission is ToolPermission.NEVER, f"{command}: {context.reason}"
    assert "outside" in (context.reason or ""), command


@pytest.mark.parametrize(
    "command",
    [
        # Binutils/assembler benign forms.
        "strip -o stripped evil",
        "strip evil",
        "objcopy evil copied",
        "as -o evil.o evil.s",
        # Config-content writers to workspace targets; scp remote forms
        # stay outside the local-metadata scope.
        "gpg --batch --yes --pinentry-mode loopback --passphrase x --decrypt -o out.txt payload.gpg",
        "xmllint --output out.xml doc.xml",
        "scp seed.txt copied.txt",
        "scp -P 22 seed.txt copied2.txt",
        "scp seed.txt host:x",
        "scp host:x local.txt",
        # In-place compression mutators on workspace files.
        "bzip2 seed.txt",
        "xz seed.txt",
        "zstd seed.txt",
        "zstd -q -o out.zst seed.txt",
        "lzma seed.txt",
        # Archive destinations in the workspace.
        "zip out.zip seed.txt",
        "7z a out.7z seed.txt",
        "7z x -od7z evil.7z",
        "unar -f -o unarout evil.zip",
        # mktemp creation in the workspace (no operand stays permitted).
        "mktemp",
        "mktemp tmpl.XXXXXX",
        "mktemp -p . tmpl2.XXXXXX",
        # git worktree move / submodule add to workspace destinations.
        "git worktree move wt wt3",
        "git worktree move -f wt wt4",
        "git submodule add ../src submod",
        "git submodule list",
        "git worktree list",
    ],
)
def test_round3_benign_forms_stay_permitted(bash_tool: Bash, command: str) -> None:
    assert_permitted(bash_tool, command)


def test_round3_new_commands_to_scratchpad_stay_permitted(
    git_repo: Path, tmp_path: Path
) -> None:
    # A scratchpad-configured tool may run the newly inspected commands
    # against its scratchpad.
    scratchpad = tmp_path / "scratch"
    scratchpad.mkdir()
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    tool = Bash(
        config_getter=lambda: config,
        state=BaseToolState(),
        cwd=git_repo,
        scratchpad_dir=scratchpad,
    )
    assert_permitted(tool, f"strip -o {scratchpad}/stripped evil")
    assert_permitted(tool, f"objcopy evil {scratchpad}/copied")
    assert_permitted(
        tool,
        f"gpg --batch --yes --pinentry-mode loopback --passphrase x --decrypt -o {scratchpad}/out.txt payload.gpg",
    )
    assert_permitted(tool, f"xmllint --output {scratchpad}/out.xml doc.xml")
    assert_permitted(tool, f"scp seed.txt {scratchpad}/copied.txt")
    assert_permitted(tool, f"bzip2 {scratchpad}/f.txt")
    assert_permitted(tool, f"xz {scratchpad}/f.txt")
    assert_permitted(tool, f"zstd -q -o {scratchpad}/out.zst seed.txt")
    assert_permitted(tool, f"zip {scratchpad}/out.zip seed.txt")
    assert_permitted(tool, f"7z a {scratchpad}/out.7z seed.txt")
    assert_permitted(tool, f"7z x -o{scratchpad}/x7z evil.7z")
    assert_permitted(tool, f"unar -f -o {scratchpad}/unarout evil.zip")
    assert_permitted(tool, f"mktemp -p {scratchpad} tmpl.XXXXXX")
    assert_permitted(tool, f"git worktree move wt {scratchpad}/wt")
    assert_permitted(tool, f"git submodule add ../src {scratchpad}/mod")


# --- Shell policy v0.2 coupling (ADR 0021, P10 guards stay green) ---
#
# The scratchpad redirect-target authorization must not weaken the operand-
# level metadata guard: the metadata check sees the ORIGINAL literal redirect
# path (before any scratchpad authorization), and the ln guard still contains
# hard-link/symlink aliases created by a chain that also carries a redirect.


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    scratchpad = tmp_path / "scratch"
    scratchpad.mkdir()
    return scratchpad


def _scratch_tool(git_repo: Path, scratchpad: Path) -> Bash:
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    return Bash(
        config_getter=lambda: config,
        state=BaseToolState(),
        cwd=git_repo,
        scratchpad_dir=scratchpad,
    )


def test_scratchpad_redirect_target_is_allowed(git_repo: Path, scratch: Path) -> None:
    # Positive control: the scratchpad authorization is real.
    assert_permitted(_scratch_tool(git_repo, scratch), f"echo hi > {scratch}/log.txt")
    assert_permitted(
        _scratch_tool(git_repo, scratch), f"cd {scratch} && echo hi > log.txt"
    )


def test_scratchpad_symlink_to_git_metadata_stays_denied(
    git_repo: Path, scratch: Path
) -> None:
    # A pre-existing symlink inside the scratchpad that points at the
    # repository's metadata is not a scratchpad target: the metadata guard
    # resolves the ORIGINAL redirect path through the link and denies it.
    (scratch / "g").symlink_to(git_repo / ".git", target_is_directory=True)
    tool = _scratch_tool(git_repo, scratch)
    assert_denied(tool, f"echo x > {scratch}/g/config")
    assert_denied(tool, f"echo x > {scratch}/g/hooks/pre-push")
    # The same target spelled relative to a cd into the scratchpad.
    assert_denied(tool, f"cd {scratch} && echo x > g/config")


def test_ln_alias_into_scratchpad_with_redirect_stays_denied(
    git_repo: Path, scratch: Path
) -> None:
    # The P10 ln guard contains the alias at link-creation time (its source
    # operand names the metadata directory), redirect or not.
    tool = _scratch_tool(git_repo, scratch)
    assert_denied(
        tool, f"ln -s {git_repo}/.git {scratch}/g && echo x > {scratch}/g/config"
    )
    # The same chain with the cd first: the relative redirect target resolves
    # inside the scratchpad, so only the ln guard can deny it.
    assert_denied(tool, f"cd {scratch} && ln -s {git_repo}/.git g && echo x > g/config")


def test_scratchpad_redirect_does_not_open_metadata_writes(
    git_repo: Path, scratch: Path
) -> None:
    # The device-sink and scratchpad exemptions never turned into file writes:
    # a relative redirect that reaches metadata through a scratchpad symlink
    # component is still denied by the metadata guard.
    (scratch / "hooks").symlink_to(
        git_repo / ".git" / "hooks", target_is_directory=True
    )
    tool = _scratch_tool(git_repo, scratch)
    assert_denied(tool, f"echo x > {scratch}/hooks/pre-push")
    assert_denied(tool, f"cd {scratch} && echo x > hooks/pre-push")


def test_concatenation_redirect_to_metadata_stays_denied(
    git_repo: Path, scratch: Path
) -> None:
    # The broadened _literal_redirect_target attributes concatenation
    # destinations (ADR 0021 fix round): pre-v0.2, only word/string/raw_string
    # destinations were extracted, so these spellings bypassed the redirect
    # metadata guard and wrote protected metadata. Concatenation is still
    # attributed: the metadata guard sees the original literal target.
    tool = _scratch_tool(git_repo, scratch)
    assert_denied(tool, 'echo x > .gi"t"/config')
    assert_denied(tool, 'echo x > ".g"it/config')
    assert_denied(tool, 'echo x > "sub"/../.git/config')
    # The same target through a scratchpad component stays denied.
    (scratch / "alias").symlink_to(git_repo / ".git", target_is_directory=True)
    assert_denied(tool, f'echo x > "{scratch}"/alias/config')
    # Positive control: a benign concatenation target stays permitted.
    assert_permitted(tool, 'echo x > "sub"/inner.txt')
