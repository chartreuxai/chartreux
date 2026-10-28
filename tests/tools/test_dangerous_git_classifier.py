from __future__ import annotations

import dataclasses
from pathlib import Path
import shlex

import pytest

from chartreux.core.dangerous_commands import (
    CONSEQUENTIAL_OPERATION_IDS,
    OPERATION_REGISTRY,
    PUBLICATION_OPERATION_IDS,
    DangerousCommandAuthority,
    DangerousCommandClassification,
    DangerousCommandProposal,
    GitRepositoryRelocation,
    GrantChoice,
    RepositoryScope,
    grant_choices_for,
    is_grant_choice_valid,
    operations_for_ids,
)
from chartreux.core.tools.builtins import _shell_git_classifier, _shell_git_support
from chartreux.core.tools.builtins._shell_command_policy import (
    analyze_shell_command_policy,
    classify_dangerous_git_command,
    is_dangerous_git_subcommand,
)

FROZEN_OPERATION_IDS = (
    "git.push",
    "git.push.force",
    "git.push.delete",
    "git.push.mirror",
    "git.checkout",
    "git.restore",
    "git.switch.discard",
    "git.stash.drop",
    "git.stash.clear",
    "git.reflog.expire",
    "git.reflog.delete",
    # v0.5.9 P2-a: destination-aware egress — publication-class like
    # git.push, so the ask-class rule grants it the same way.
    "network.egress",
)


def classify(command: str) -> DangerousCommandClassification:
    """Classify one shell command string as a whole expanded call."""
    return classify_dangerous_git_command(shlex.split(command))


def assert_grantable(command: str, operation_ids: tuple[str, ...]) -> None:
    """Assert a command classifies as grantable with exactly these IDs."""
    result = classify(command)
    assert result.kind == "grantable"
    assert result.grantable is not None
    assert result.grantable.operation_ids == operation_ids
    assert result.grantable.consequence


def assert_hard_denied(command: str) -> None:
    """Assert a command fails closed: denied with no grantable operation."""
    result = classify(command)
    assert result.kind == "hard_denied"
    assert result.grantable is None
    assert result.reason


@pytest.mark.parametrize(
    "command",
    [
        "git push",
        "git push origin main",
        "git push origin main:main",
        "git push v1.0",
        "git push --all origin",
        "git push --tags origin",
        "git push -u origin main",
        "git push --atomic origin main",
        "git push --thin origin main",
        "git push -q origin main",
        "git push -v origin main",
        "git push --dry-run origin main",
        "git push --porcelain origin main",
        "git push --signed=yes origin main",
        "git push --no-force origin main",
        "git push --progress origin main",
        "git push --verify origin main",
        "git push --follow-tags origin main",
        "git push -4 origin main",
        "git push --ipv6 origin main",
        "git push -o ci.skip origin main",
        # A --repo value that is a bare name stays ordinary publication.
        "git push --repo=origin main",
        "git push --repo upstream main",
    ],
)
def test_ordinary_push_is_grantable_publication(command: str) -> None:
    assert_grantable(command, ("git.push",))
    assert OPERATION_REGISTRY["git.push"].command_class == "publication"


@pytest.mark.parametrize(
    "command",
    [
        "git push --force origin main",
        "git push -f origin main",
        "git push -fq origin main",
        "git push --force-with-lease origin main",
        "git push --force-with-lease=main origin main",
        "git push --force-with-lease=main:0123abcd origin main",
        # An empty attached lease value is a valid git form.
        "git push --force-with-lease= origin main",
        "git push --force-if-includes origin main",
        # Unambiguous abbreviations resolve like git's own parser.
        "git push --force-with-le origin main",
        "git push --force-w origin main",
        "git push origin +main",
        "git push origin +main:dev",
        "git push origin +refs/heads/a:refs/heads/b",
        "git push origin main +dev:dev",
    ],
)
def test_force_push_forms_are_grantable_consequential(command: str) -> None:
    assert_grantable(command, ("git.push.force",))
    assert OPERATION_REGISTRY["git.push.force"].command_class == "consequential"


@pytest.mark.parametrize(
    "command",
    [
        "git push origin :main",
        "git push origin :refs/heads/main",
        "git push --delete origin main",
        "git push --delete origin main topic",
        "git push origin main :dev",
        # --prune removes remote refs without a local counterpart, so it is
        # a deletion mechanism, never ordinary publication.
        "git push --prune origin",
        "git push --prune --all origin",
        "git push --prune --tags origin",
    ],
)
def test_delete_push_forms_are_grantable_consequential(command: str) -> None:
    assert_grantable(command, ("git.push.delete",))
    assert OPERATION_REGISTRY["git.push.delete"].command_class == "consequential"


@pytest.mark.parametrize("command", ["git push --mirror origin", "git push --mirror"])
def test_mirror_push_is_grantable_consequential(command: str) -> None:
    assert_grantable(command, ("git.push.mirror",))
    assert OPERATION_REGISTRY["git.push.mirror"].command_class == "consequential"


@pytest.mark.parametrize(
    "command, operation_ids",
    [
        (
            "git push --force --delete origin main",
            ("git.push.force", "git.push.delete"),
        ),
        # IDs are canonicalized to registry order regardless of flag order.
        (
            "git push --delete --force origin main",
            ("git.push.force", "git.push.delete"),
        ),
        ("git push --mirror --force origin", ("git.push.force", "git.push.mirror")),
        ("git push --delete --mirror origin", ("git.push.delete", "git.push.mirror")),
        (
            "git push --force --delete --mirror origin",
            ("git.push.force", "git.push.delete", "git.push.mirror"),
        ),
        ("git push --force origin :main", ("git.push.force", "git.push.delete")),
        (
            "git push --force-with-lease origin :main",
            ("git.push.force", "git.push.delete"),
        ),
        ("git push origin +main:dev :old", ("git.push.force", "git.push.delete")),
        ("git push origin +main :dev +old:new", ("git.push.force", "git.push.delete")),
        ("git push origin +:main", ("git.push.force", "git.push.delete")),
        ("git push --delete origin main :topic", ("git.push.delete",)),
        # --prune combines with the other mechanisms like --delete does.
        ("git push --prune --force origin main", ("git.push.force", "git.push.delete")),
        ("git push --prune --mirror origin", ("git.push.delete", "git.push.mirror")),
    ],
)
def test_compound_push_requires_all_consequential_ids(
    command: str, operation_ids: tuple[str, ...]
) -> None:
    # A push combining consequential mechanisms requires ALL classified
    # consequential IDs; it never requires or inherits git.push.
    assert_grantable(command, operation_ids)
    assert "git.push" not in operation_ids
    for operation_id in operation_ids:
        operation = OPERATION_REGISTRY[operation_id]
        assert operation.command_class == "consequential"
        assert not operation.persistable


@pytest.mark.parametrize(
    "command",
    [
        # An explicit path or URL destination is never ordinary publication:
        # the grant scopes the source repository, and a push to the
        # destination runs that destination's receive-pack and hooks.
        "git push /tmp/evil.git HEAD:refs/heads/main",
        "git push ../remote.git main",
        "git push ~/remote.git",
        "git push https://user@host/x.git",
        "git push ssh://host/x.git main",
        "git push git://host/x.git",
        # An scp-like colon destination is a URL too: git remote names can
        # never contain ':' (refname validation), so the operand cannot name
        # a configured remote the grant covers.
        "git push host:repo.git",
        "git push user@host:repo.git",
        "git push host:path/to/repo.git",
        # The first positional is the repository, so a refspec-shaped operand
        # there is a destination too, not a refspec.
        "git push HEAD:refs/heads/main",
        "git push :main",
        # --repo names the same destination, in = or separated form.
        "git push --repo=/tmp/evil.git main",
        "git push --repo /tmp/evil.git main",
        "git push --repo=../remote.git main",
        "git push --repo=https://host/x.git main",
        "git push --repo=host:repo.git main",
        "git push --repo host:repo.git main",
    ],
)
def test_explicit_path_and_url_destinations_are_hard_guards(command: str) -> None:
    result = classify(command)
    assert result.kind == "hard_denied"
    assert result.grantable is None
    assert result.reason
    assert "destination" in result.reason
    assert "receive-pack" in result.reason


def test_first_push_positional_is_the_repository_not_a_refspec() -> None:
    # git parses the first positional as the repository, never a refspec: a
    # +-prefixed operand there is a remote name, not a forced refspec, and a
    # leading-colon operand is an explicit destination (a hard guard), not a
    # deletion refspec.
    result = classify("git push +main")
    assert result.kind == "grantable"
    assert result.grantable is not None
    assert result.grantable.operation_ids == ("git.push",)
    assert result.grantable.call.operands == ("+main",)
    assert classify("git push :main").kind == "hard_denied"


@pytest.mark.parametrize(
    "command, cwd_paths, git_dir, work_tree",
    [
        ("git -C . push", (".",), None, None),
        ("git -C. push", (".",), None, None),
        ("git -Csub push", ("sub",), None, None),
        ("git --git-dir=. push", (), ".", None),
        ("git --git-dir . push", (), ".", None),
        ("git --work-tree=/wt push", (), None, "/wt"),
        ("git --work-tree /wt push", (), None, "/wt"),
        ("git -C a -C b push", ("a", "b"), None, None),
        ("git --no-pager -C . push", (".",), None, None),
        ("git -C . --no-pager push", (".",), None, None),
        # The last --git-dir/--work-tree value wins.
        ("git --git-dir=a --git-dir=b push", (), "b", None),
        ("git --work-tree=/a --work-tree=/b push", (), None, "/b"),
        ("/usr/bin/git -C . push", (".",), None, None),
        ("git -C . push origin main", (".",), None, None),
        ("git -C . --git-dir=. --work-tree=/wt push", (".",), ".", "/wt"),
    ],
)
def test_global_option_spellings_are_normalized(
    command: str, cwd_paths: tuple[str, ...], git_dir: str | None, work_tree: str | None
) -> None:
    result = classify(command)
    assert result.kind == "grantable"
    assert result.grantable is not None
    assert result.grantable.operation_ids == ("git.push",)
    assert result.grantable.relocation == GitRepositoryRelocation(
        cwd_paths=cwd_paths, git_dir=git_dir, work_tree=work_tree
    )


def test_consumed_global_operand_is_not_a_flag() -> None:
    # -C consumes the next token even when it looks like a flag.
    result = classify("git -C -v push")
    assert result.kind == "grantable"
    assert result.grantable is not None
    assert result.grantable.relocation.cwd_paths == ("-v",)


@pytest.mark.parametrize(
    "command, operation_id",
    [
        ("git checkout", "git.checkout"),
        ("git checkout main", "git.checkout"),
        ("git checkout -b feat", "git.checkout"),
        ("git checkout -B feat", "git.checkout"),
        ("git checkout --detach main", "git.checkout"),
        ("git checkout --orphan new", "git.checkout"),
        ("git checkout -q main", "git.checkout"),
        ("git checkout --conflict=merge main", "git.checkout"),
        ("git checkout main file.txt", "git.checkout"),
        # Every checkout form is consequential; git itself rejects unmodeled
        # options, so recording them does not weaken the classification.
        ("git checkout --bogus main", "git.checkout"),
        ("git restore", "git.restore"),
        ("git restore file", "git.restore"),
        ("git restore -s HEAD file", "git.restore"),
        ("git restore --source=HEAD --staged file", "git.restore"),
        ("git restore -p file", "git.restore"),
        ("git restore -S -W file", "git.restore"),
        ("git restore --worktree --staged file", "git.restore"),
    ],
)
def test_checkout_and_restore_are_always_consequential(
    command: str, operation_id: str
) -> None:
    assert_grantable(command, (operation_id,))
    assert OPERATION_REGISTRY[operation_id].command_class == "consequential"


@pytest.mark.parametrize(
    "command",
    [
        "git switch main",
        "git switch -c feat",
        "git switch -C feat",
        "git switch -d main",
        "git switch --detach main",
        "git switch -- main",
        "git switch -m main",
        "git switch -q main",
        "git switch --track origin/main",
        "git switch --recurse-submodules main",
        "git switch --no-create main",
    ],
)
def test_ordinary_switch_is_routine(command: str) -> None:
    assert classify(command).kind == "routine"


@pytest.mark.parametrize(
    "command",
    [
        "git switch -f main",
        "git switch --force main",
        "git switch --discard-changes main",
        "git switch -qf main",
        "git switch --force --detach main",
        "git switch -f",
    ],
)
def test_discard_switch_is_consequential(command: str) -> None:
    assert_grantable(command, ("git.switch.discard",))
    assert OPERATION_REGISTRY["git.switch.discard"].command_class == "consequential"


@pytest.mark.parametrize(
    "command, operation_id",
    [
        ("git stash drop", "git.stash.drop"),
        ("git stash drop -q", "git.stash.drop"),
        ("git stash drop stash@{1}", "git.stash.drop"),
        ("git stash drop --quiet stash@{0}", "git.stash.drop"),
        ("git stash clear", "git.stash.clear"),
        ("git stash clear -q", "git.stash.clear"),
        ("git reflog expire", "git.reflog.expire"),
        ("git reflog expire --all", "git.reflog.expire"),
        (
            "git reflog expire --expire=now --expire-unreachable=now --updateref",
            "git.reflog.expire",
        ),
        ("git reflog delete", "git.reflog.delete"),
        ("git reflog delete main@{0}", "git.reflog.delete"),
        ("git reflog delete -n main@{0}", "git.reflog.delete"),
    ],
)
def test_stash_and_reflog_destructive_forms_are_grantable(
    command: str, operation_id: str
) -> None:
    assert_grantable(command, (operation_id,))
    assert OPERATION_REGISTRY[operation_id].command_class == "consequential"


@pytest.mark.parametrize(
    "command",
    [
        "git stash",
        "git stash list",
        "git stash pop",
        "git stash apply",
        "git stash show",
        "git stash branch feat",
        "git stash push -m x",
        # -m/--message consume before the sub-subcommand is read: these are
        # stash pushes with a message, not drops.
        "git stash -m drop",
        "git stash --message drop",
        "git reflog",
        "git reflog show",
        "git reflog show HEAD",
    ],
)
def test_other_stash_and_reflog_forms_are_routine(command: str) -> None:
    # drop/clear and expire/delete are recognized without enabling other
    # unmodeled forms.
    assert classify(command).kind == "routine"


def test_reflog_stays_dispatch_masked_after_the_unmodeled_dedup() -> None:
    """reflog's safety is the dispatch mask, never the reader set.

    Deduplicated out of the unmodeled table (its entry there was dead: the
    dispatch to the mode-aware stash/reflog classifier runs before the
    unmodeled check, so no reflog form ever reached it). reflog can write,
    so its destructive forms must keep their mode-aware classification —
    its reader-set membership only governs the policy layer's read forms.
    """
    assert "reflog" in _shell_git_classifier._DANGEROUS_GIT_SUBCOMMANDS
    assert "reflog" in _shell_git_support._GIT_READER_SUBCOMMANDS
    assert "reflog" not in _shell_git_support._GIT_UNMODELED_BUILTINS
    assert is_dangerous_git_subcommand(["git", "reflog"])
    assert_grantable("git reflog expire --all", ("git.reflog.expire",))
    assert_grantable("git reflog delete main@{0}", ("git.reflog.delete",))
    assert classify("git reflog").kind == "routine"
    assert classify("git reflog show").kind == "routine"


@pytest.mark.parametrize(
    "command",
    [
        "git reset --hard",
        "git reset --hard HEAD",
        # Unambiguous abbreviations resolve like git's own parser, so an
        # abbreviated --hard is still a hard guard.
        "git reset --ha HEAD",
        "git reset -q --hard HEAD",
        "git clean -f",
        "git clean -fd",
        "git clean --f",
        "git clean -x -f",
        "git clean -e .ign -f",
        # Config overrides are hard guards.
        "git -c foo=bar push",
        "git -cfoo=bar push",
        "git --config-env foo=bar push",
        "git --config-env=foo=bar push",
        # Aliases and unmodeled builtins cannot be inspected.
        "git fixture-alias",
        "git fsck",
        "git gc --auto",
        "git bundle list-heads",
    ],
)
def test_hard_guards_and_unmodeled_forms_are_not_grantable(command: str) -> None:
    assert_hard_denied(command)


@pytest.mark.parametrize(
    "command",
    [
        "git reset --soft HEAD~1",
        "git reset --mixed HEAD",
        "git reset --keep HEAD",
        "git reset --merge HEAD",
        "git reset",
        "git reset HEAD~1",
        "git clean",
        "git clean -n",
        "git clean -x",
        "git clean -n -f",
        "git clean -fdn",
        "git clean -e .ign -n",
    ],
)
def test_benign_reset_and_clean_forms_are_routine(command: str) -> None:
    # Non-destructive reset/clean forms stay governed by the existing policy.
    assert classify(command).kind == "routine"


@pytest.mark.parametrize(
    "command",
    [
        # git resolves repeated booleans in order and accepts --no-
        # spellings, so the final --no-dry-run cancels --dry-run (or -n) and
        # re-enables the destructive clean: a hard guard, never a proposal.
        "git clean --force --dry-run --no-dry-run",
        "git clean --force -n --no-dry-run",
        "git clean -f --dry-run --no-dry-run",
        "git clean -fd --no-dry-run",
    ],
)
def test_clean_dry_run_negation_is_last_spelling_wins(command: str) -> None:
    assert_hard_denied(command)


@pytest.mark.parametrize(
    "command",
    [
        # --no-force cancels --force, so git itself refuses to clean.
        "git clean --force --no-force",
        "git clean -f --no-force",
        # A trailing -n wins over an earlier --no-dry-run.
        "git clean --no-dry-run -n",
        "git clean --force --no-force --dry-run --no-dry-run",
    ],
)
def test_clean_negated_flags_resolve_safely(command: str) -> None:
    # Effective last-wins semantics: the cancelled flag leaves the clean
    # inert, so the form stays routine.
    assert classify(command).kind == "routine"


@pytest.mark.parametrize(
    "command",
    [
        # A write to a push-effecting key redefines where an approved push
        # lands or the program it runs (the remote's url, a url rewrite, the
        # bare push destination, the receive-pack program, mirrored refs,
        # configured refspecs, the default push remote), so no grant can
        # authorize it: a hard guard, not a proposal.
        "git config remote.origin.mirror true",
        "git config --global remote.origin.mirror true",
        "git config --local remote.origin.mirror true",
        "git config remote.origin.push +HEAD:main",
        "git config remote.origin.push :main",
        "git config --global remote.x.push +HEAD:main",
        "git config --add remote.origin.push +HEAD:main",
        "git config --replace-all remote.origin.push +HEAD:main",
        "git config remote.pushDefault origin",
        "git config remote.origin.pushDefault origin",
        "git config --global remote.pushDefault origin",
        "git config --file .git/config remote.origin.mirror true",
        "git config --file custom.cfg remote.origin.mirror true",
        "git config --type bool remote.origin.mirror true",
        "git config --includes remote.origin.mirror true",
        # The remote definition itself: url/pushurl point a granted push at
        # a destination of the writer's choosing.
        "git config remote.origin.url https://example.com/x.git",
        "git config remote.origin.url ../evil.git",
        "git config --global remote.origin.pushurl https://example.com/x.git",
        "git config remote.upstream.pushurl ../evil.git",
        # The program a push (or fetch) runs, and the proxy it runs through.
        "git config remote.origin.receivepack echo",
        "git config remote.origin.uploadpack echo",
        "git config --global remote.origin.receivepack echo",
        "git config remote.origin.proxy echo",
        # url rewrites redirect a push to a configured remote's URL.
        "git config url.https://evil/.pushInsteadOf https://github.com/",
        "git config url.https://evil/.insteadOf https://github.com/",
        "git config --global url.https://evil/.pushInsteadOf https://github.com/",
        # The bare push destination (path-capable).
        "git config branch.main.remote origin",
        "git config branch.main.remote ../evil.git",
        "git config branch.main.pushremote origin",
        "git config --global branch.main.pushremote ../evil.git",
        # The remote's vcs helper: git remote update executes the configured
        # git-remote-<vcs> helper from PATH.
        "git config remote.origin.vcs evil",
        "git config --global remote.upstream.vcs hg",
        "git config set remote.origin.vcs git",
        "git config --unset remote.origin.vcs",
        # The program a push runs beyond the remote's own: the ssh command
        # (even a --dry-run), the git:// proxy, the ask-pass program, and
        # the credential-lookup helpers.
        "git config core.sshCommand evil",
        "git config --global core.sshCommand evil",
        "git config core.gitProxy evil",
        "git config core.askPass evil",
        "git config set core.gitProxy evil",
        "git config --unset core.sshCommand",
        "git config credential.helper evil",
        "git config credential.https://example.com.helper evil",
        "git config --global credential.helper evil",
        "git config set credential.https://example.com.helper evil",
        "git config --unset credential.helper",
        # An authored include folds a file into configuration resolution,
        # which can set any push-effecting key for a verified remote.
        "git config include.path /tmp/evil.cfg",
        "git config --global include.path ../evil.cfg",
        "git config includeIf.gitdir:~/work/.path /tmp/evil.cfg",
        "git config includeIf.onbranch:main.path /tmp/evil.cfg",
        "git config includeIf.hasconfig:remote.*.url:https://x/.path /tmp/e.cfg",
        "git config set include.path /tmp/evil.cfg",
        "git config --add includeIf.gitdir:~/work/.path /tmp/evil.cfg",
        "git config --unset include.path",
        "git config --unset includeIf.gitdir:/x.path",
        # The newer subcommand spellings write the same keys.
        "git config set remote.origin.mirror true",
        "git config set remote.origin.push +HEAD:main",
        "git config set remote.origin.url https://example.com/x.git",
        "git config set branch.main.remote origin",
        # Removals and section operations are mutations too: the agent must
        # not touch the push-effecting surface at all.
        "git config --unset remote.origin.mirror",
        "git config --unset-all remote.origin.push",
        "git config --unset remote.origin.url",
        "git config --unset branch.main.pushremote",
        "git config unset remote.origin.mirror",
        "git config --rename-section remote.evil remote.origin",
        "git config --remove-section remote.evil",
        "git config --remove-section remote",
        "git config --remove-section url.https://evil/",
        "git config --rename-section branch.old branch.new",
        "git config --remove-section branch.old",
        # Section operations reaching the credential and include sections
        # relocate or delete their keys too.
        "git config --remove-section credential",
        "git config --remove-section credential.https://example.com",
        "git config --rename-section include include2",
        "git config --remove-section include",
        "git config --rename-section includeIf.gitdir:/x includeIf.gitdir:/y",
        "git config --remove-section includeIf.onbranch:main",
        # An editor write can set any key, so it is never classifiable.
        "git config -e",
        "git config --edit",
        "git config edit",
        # Section and key names fold case-insensitively; the guard fails
        # closed on any spelling.
        "git config REMOTE.ORIGIN.MIRROR true",
        "git config Remote.Origin.Push +HEAD:main",
        "git config REMOTE.ORIGIN.URL https://example.com/x.git",
        "git config Branch.Main.Remote origin",
        "git config URL.HTTPS://EVIL/.PUSHINSTEADOF https://github.com/",
        "git config REMOTE.ORIGIN.VCS evil",
        "git config Core.SshCommand evil",
        "git config CORE.GITPROXY evil",
        "git config Core.AskPass evil",
        "git config CREDENTIAL.HELPER evil",
        "git config Credential.Https://Example.com.Helper evil",
        "git config INCLUDE.PATH /tmp/evil.cfg",
        "git config INCLUDEIF.GITDIR:/X.PATH /tmp/evil.cfg",
    ],
)
def test_push_effecting_config_writes_are_hard_guards(command: str) -> None:
    result = classify(command)
    assert result.kind == "hard_denied"
    assert result.grantable is None
    assert result.reason
    assert "push-effecting configuration" in result.reason
    assert "no grant can authorize" in result.reason
    assert "manually" in result.reason


@pytest.mark.parametrize(
    "command",
    [
        # Reads never write: --get/--list/... and the get/list subcommands,
        # including reads of push-effecting keys and the bare-key form,
        # which git reads.
        "git config --get remote.origin.mirror",
        "git config --get-all remote.origin.push",
        "git config --get-regexp remote.origin",
        "git config --get-urlmatch url.https://x/.insteadof https://x/",
        "git config --get remote.origin.url",
        "git config --get remote.origin.receivepack",
        "git config --get url.https://x/.pushInsteadOf",
        "git config --get branch.main.remote",
        "git config --get branch.main.pushremote",
        "git config --get remote.origin.vcs",
        "git config --get core.sshCommand",
        "git config --get core.gitProxy",
        "git config --get core.askPass",
        "git config --get credential.helper",
        "git config --get credential.https://example.com.helper",
        "git config --get include.path",
        "git config --get includeIf.gitdir:/x.path",
        "git config --list",
        "git config -l",
        "git config --name-only --list",
        "git config get remote.origin.mirror",
        "git config get remote.origin.url",
        "git config get core.sshCommand",
        "git config get include.path",
        "git config list",
        "git config remote.origin.mirror",
        "git config remote.origin.url",
        "git config branch.main.remote",
        "git config remote.origin.vcs",
        "git config core.sshCommand",
        "git config credential.helper",
        "git config include.path",
        "git config --get remote.origin.mirror pattern",
        # Writes to unrelated keys stay routine.
        "git config user.name X",
        "git config --global user.name X",
        "git config --add user.email t@t",
        "git config set user.name X",
        "git config --unset user.name",
        "git config unset user.name",
        "git config --file custom.cfg user.name X",
        "git config remote.origin.fetch +refs/heads/*:refs/remotes/origin/*",
        "git config branch.main.merge refs/heads/main",
        # DELIBERATE REVISION (P10): a rename whose destination section is
        # `core` used to stay routine, which let an agent write hooksPath into
        # an unrelated section and rename it into `core`. Section renames
        # reaching `core` (or `sequence`) are hard guards now; the unrelated
        # renames and section removals below stay routine.
        "git config --remove-section core.old",
        "git config --remove-section sequence.old",
        "git config --rename-section other.one other.two",
        # push.default writes stay routine: they broaden what a bare push
        # publishes, never force, delete, or mirror (documented in ADR 0019).
        "git config push.default simple",
        "git config push.default current",
        # Reads of the program-selection keys never write: the bare-key form
        # is git's read form too.
        "git config --get core.hooksPath",
        "git config --get-all core.editor",
        "git config --get-regexp core.hooksPath",
        "git config --get sequence.editor",
        "git config core.hooksPath",
        "git config core.editor",
        "git config sequence.editor",
        "git config --get core.hooksPath extra",
        "git config --get filter.pwn.clean",
        "git config --get filter.pwn.process",
        "git config --get merge.ours.driver",
        "git config --get diff.external.command",
        "git config --get diff.external.textconv",
        "git config --get diff.external.external",
        "git config filter.pwn.clean",
        "git config merge.ours.driver",
        "git config diff.external.command",
        # Reads of the round-3 program-selection keys never write.
        "git config --get core.fsmonitor",
        "git config --get man.woman.path",
        "git config --get man.viewer",
        "git config --get interactive.diffFilter",
        "git config core.fsmonitor",
        "git config man.woman.path",
        "git config man.viewer",
        "git config interactive.diffFilter",
        # Writes to unrelated core/sequence keys stay routine.
        "git config core.pager less",
        "git config core.autocrlf true",
        "git config sequence.squashWarning false",
        "git config --global core.fileMode true",
        # The driver sections keep their non-program keys: required and
        # smudge are boolean flags and data spellings, not command strings
        # (a documented residual, ADR 0019).
        "git config filter.pwn.required true",
        "git config filter.pwn.smudge cat",
        "git config merge.ours.name ours",
        "git config merge.conflictStyle diff3",
        "git config diff.external.foo bar",
        "git config diff.renameLimit 100",
        "git config --remove-section filter.old",
        "git config --remove-section merge.old",
        "git config --remove-section diff.old",
        # The round-3 sections keep their non-program keys too: the
        # deprecated man.<tool>.cmd is ignored by git 2.53 (warning, never
        # executed — probe-verified), and man.<tool>.tool/other spellings
        # name no executable.
        "git config man.woman.cmd evil",
        "git config man.woman.tool woman",
        "git config interactive.singleKey true",
        "git config --remove-section man.old",
        "git config --remove-section interactive.old",
    ],
)
def test_config_reads_and_unrelated_writes_are_routine(command: str) -> None:
    assert classify(command).kind == "routine"


def test_git_config_routes_through_the_dangerous_flow() -> None:
    # config is routed so a hard-guarded write is denied by the resolver;
    # routine config forms stay governed by the existing policy layer.
    assert is_dangerous_git_subcommand(shlex.split("git config user.name x"))
    assert is_dangerous_git_subcommand(shlex.split("git config --get user.name"))


@pytest.mark.parametrize(
    "command",
    [
        # DELIBERATE LAYERING (P10 fix round 1): the classifier leaves git's
        # raw write operands and the routine tree-writing subcommands
        # routine. The operand guard (archive --output, a writing config
        # --file) and the work-tree relocation guard (apply/merge/... under
        # a metadata-resolving work tree) are resolver-layer denials that
        # run before this flow; classifying these forms as dangerous here
        # would be a deliberate revision of that layering.
        "git archive --output=.git/config HEAD",
        "git archive -o /etc/cron.d/pwn HEAD",
        "git config --file=/etc/evil user.name x",
        "git config --file=.git/config user.name x",
        "git -C .git apply ../hook.diff",
        "git -C .git merge evil",
        "git --git-dir=.git --work-tree=.git merge evil",
        "git --git-dir=.git --work-tree=.git cherry-pick evil",
        "git stash pop",
        "git pull",
        "git fetch origin",
    ],
)
def test_git_raw_write_and_tree_writing_calls_stay_routine(command: str) -> None:
    result = classify(command)
    assert result.kind == "routine"
    assert result.grantable is None


@pytest.mark.parametrize(
    "command",
    [
        # The classifier's key guards still cover the dangerous keys of a
        # --file write in every scope, metadata destination or not.
        "git config --file=.git/config core.hooksPath /tmp/h",
        "git config --file=/etc/evil core.hooksPath /tmp/h",
        "git config --file=/etc/evil remote.origin.url /tmp/evil",
        "git config -f /etc/evil core.editor evil",
        "git config --file=/etc/evil sequence.editor evil",
        "git config --file=/etc/evil --rename-section other core",
    ],
)
def test_config_file_writes_of_guarded_keys_are_hard_guards(command: str) -> None:
    result = classify(command)
    assert result.kind == "hard_denied"
    assert result.grantable is None
    assert result.reason
    assert "no grant can authorize" in result.reason


@pytest.mark.parametrize(
    "command",
    [
        # core.hooksPath selects the directory git runs hooks from: a granted
        # push — or any hook-running subcommand — executes the hooks there.
        # Every write scope is covered, including --file and relocation.
        "git config core.hooksPath /tmp/h",
        "git config --global core.hooksPath /tmp/h",
        "git config --local core.hooksPath /tmp/h",
        "git config --worktree core.hooksPath /tmp/h",
        "git config --file cfg core.hooksPath /tmp/h",
        "git config --file .git/config core.hooksPath /tmp/h",
        "git config set core.hooksPath /tmp/h",
        "git config --add core.hooksPath /tmp/h",
        "git config --replace-all core.hooksPath /tmp/h",
        "git config --unset core.hooksPath",
        "git config --unset-all core.hooksPath",
        "git config unset core.hooksPath",
        "git -C .git config core.hooksPath /tmp/h",
        "git -C . config core.hooksPath /tmp/h",
        "git --git-dir=.git config core.hooksPath /tmp/h",
        "git --git-dir .git config core.hooksPath /tmp/h",
        "git --work-tree=. --git-dir=.git config core.hooksPath /tmp/h",
        # core.editor selects the program git runs for commit/tag/rebase
        # without -m.
        "git config core.editor evil",
        "git config --global core.editor evil",
        "git config --worktree core.editor evil",
        "git config --file cfg core.editor evil",
        "git config set core.editor evil",
        "git config --unset core.editor",
        "git -C .git config core.editor evil",
        "git --git-dir=.git config core.editor evil",
        # sequence.editor selects the program an interactive rebase runs:
        # a section-prefix rule, not a core key.
        "git config sequence.editor evil",
        "git config --global sequence.editor evil",
        "git config --worktree sequence.editor evil",
        "git config --file cfg sequence.editor evil",
        "git config set sequence.editor evil",
        "git config --unset sequence.editor",
        "git -C .git config sequence.editor evil",
        "git --git-dir=.git config sequence.editor evil",
        # Section and key names fold case-insensitively; the guard fails
        # closed on any spelling.
        "git config Core.HooksPath /tmp/h",
        "git config CORE.EDITOR evil",
        "git config Sequence.Editor evil",
        "git config core.HOOKSPATH /tmp/h",
        # filter.<name>.clean and filter.<name>.process select the programs
        # git runs for files with that filter (git add, git checkout).
        "git config filter.pwn.clean evil",
        "git config --global filter.pwn.clean evil",
        "git config --local filter.pwn.clean evil",
        "git config --worktree filter.pwn.clean evil",
        "git config --file cfg filter.pwn.clean evil",
        "git config set filter.pwn.clean evil",
        "git config --add filter.pwn.clean evil",
        "git config --replace-all filter.pwn.clean evil",
        "git config --unset filter.pwn.clean",
        "git config --unset-all filter.pwn.clean",
        "git config unset filter.pwn.clean",
        "git -C .git config filter.pwn.clean evil",
        "git --git-dir=.git config filter.pwn.clean evil",
        "git config filter.pwn.process evil",
        "git config --file cfg filter.pwn.process evil",
        "git config set filter.pwn.process evil",
        "git config --unset filter.pwn.process",
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
        # core.fsmonitor selects the daemon git runs for status-bearing
        # subcommands (git add, git commit): a routine non-reader trigger.
        "git config core.fsmonitor evil",
        "git config --global core.fsmonitor evil",
        "git config --worktree core.fsmonitor evil",
        "git config --file cfg core.fsmonitor evil",
        "git config set core.fsmonitor evil",
        "git config --add core.fsmonitor evil",
        "git config --unset core.fsmonitor",
        "git config unset core.fsmonitor",
        "git -C .git config core.fsmonitor evil",
        "git --git-dir=.git config core.fsmonitor evil",
        # man.<tool>.path selects the man viewer git runs for git help once
        # man.viewer selects its tool; man.viewer itself selects the tool.
        "git config man.viewer woman",
        "git config --global man.viewer woman",
        "git config --worktree man.viewer woman",
        "git config --file cfg man.viewer woman",
        "git config set man.viewer woman",
        "git config --unset man.viewer",
        "git -C .git config man.viewer woman",
        "git --git-dir=.git config man.viewer woman",
        "git config man.woman.path /tmp/evil",
        "git config --global man.woman.path /tmp/evil",
        "git config --worktree man.woman.path /tmp/evil",
        "git config --file cfg man.woman.path /tmp/evil",
        "git config set man.woman.path /tmp/evil",
        "git config --unset man.woman.path",
        "git -C .git config man.woman.path /tmp/evil",
        "git --git-dir=.git config man.woman.path /tmp/evil",
        # interactive.diffFilter selects the filter git runs for git add -p
        # diffs.
        "git config interactive.diffFilter evil",
        "git config --global interactive.diffFilter evil",
        "git config --worktree interactive.diffFilter evil",
        "git config --file cfg interactive.diffFilter evil",
        "git config set interactive.diffFilter evil",
        "git config --unset interactive.diffFilter",
        "git -C .git config interactive.diffFilter evil",
        "git --git-dir=.git config interactive.diffFilter evil",
        # Case-folded spellings fail closed on any capitalization.
        "git config Filter.Pwn.Clean evil",
        "git config MERGE.Ours.Driver evil",
        "git config Diff.External.Command evil",
        "git config diff.EXTERNAL.Textconv evil",
        "git config CORE.FSMONITOR evil",
        "git config Core.FsMonitor evil",
        "git config MAN.WOMAN.PATH /tmp/evil",
        "git config Man.Viewer woman",
        "git config INTERACTIVE.DIFFFILTER evil",
        "git config Interactive.DiffFilter evil",
    ],
)
def test_program_selection_config_writes_are_hard_guards(command: str) -> None:
    result = classify(command)
    assert result.kind == "hard_denied"
    assert result.grantable is None
    assert result.reason
    assert "program-selection configuration" in result.reason
    assert "no grant can authorize" in result.reason
    assert "manually" in result.reason


@pytest.mark.parametrize(
    "command",
    [
        # A section rename reaching `core` moves its keys into the section
        # whose hooksPath/editor keys the key guard blocks: an agent can
        # write hooksPath into an unrelated section and rename it to core.
        # Both operands are checked, so destination section names are covered.
        "git config --rename-section other core",
        "git config --rename-section core.other core",
        "git config --rename-section CORE core",
        "git config --rename-section core core2",
        "git config --rename-section core.worktree core.worktree2",
        "git config rename-section other core",
        "git config rename-section core.other core",
        "git config --file cfg --rename-section other core",
        "git config --global --rename-section other core",
        "git -C .git config --rename-section other core",
        "git --git-dir=.git config --rename-section other core",
        # The same bypass class for sequence.editor: renaming into the
        # sequence section lands an agent-authored editor key.
        "git config --rename-section other sequence",
        "git config --rename-section sequence.old sequence",
        "git config --rename-section sequence sequence2",
        "git config rename-section other sequence",
        # The same bypass class for the driver sections: renaming into
        # (or out of) filter/merge/diff lands an agent-authored driver key.
        "git config --rename-section other filter",
        "git config --rename-section filter other",
        "git config --rename-section filter.lfs filter",
        "git config --rename-section FILTER filter",
        "git config --rename-section other merge",
        "git config --rename-section merge ours2",
        "git config --rename-section other diff",
        "git config --rename-section diff ours",
        "git config rename-section other filter",
        "git config --file cfg --rename-section other filter",
        "git config --global --rename-section other merge",
        "git -C .git config --rename-section other diff",
        "git --git-dir=.git config --rename-section other filter",
        # The same bypass class for the round-3 sections: renaming into
        # (or out of) man lands an agent-authored man viewer key, and
        # interactive lands an agent-authored diffFilter key.
        "git config --rename-section other man",
        "git config --rename-section man other",
        "git config --rename-section man.woman man",
        "git config --rename-section MAN other",
        "git config rename-section other man",
        "git config --file cfg --rename-section other man",
        "git config --global --rename-section other man",
        "git -C .git config --rename-section other man",
        "git --git-dir=.git config --rename-section man other",
        "git config --rename-section other interactive",
        "git config --rename-section interactive other",
        "git config --rename-section interactive.diffFilter interactive",
        "git config --rename-section INTERACTIVE other",
        "git config rename-section other interactive",
        "git config --file cfg --rename-section other interactive",
        "git config --global --rename-section other interactive",
        "git -C .git config --rename-section other interactive",
        "git --git-dir=.git config --rename-section interactive other",
    ],
)
def test_section_renames_reaching_program_selection_sections_are_hard_guards(
    command: str,
) -> None:
    result = classify(command)
    assert result.kind == "hard_denied"
    assert result.grantable is None
    assert result.reason
    assert "program-selection section" in result.reason
    assert "no grant can authorize" in result.reason
    assert "manually" in result.reason


@pytest.mark.parametrize(
    "command, expected",
    [
        # A rename reaching a program-selection section names the
        # destination section it lands in.
        (
            "git config --rename-section other core",
            "renames a section to 'core', a program-selection section",
        ),
        (
            "git config --rename-section other filter",
            "renames a section to 'filter', a program-selection section",
        ),
        (
            "git config --rename-section other merge",
            "renames a section to 'merge', a program-selection section",
        ),
        (
            "git config --rename-section other diff",
            "renames a section to 'diff', a program-selection section",
        ),
        # A rename moving a program-selection section out names the
        # matched (source) section.
        (
            "git config --rename-section core other",
            "renames the program-selection section 'core'",
        ),
        (
            "git config --rename-section filter.lfs other",
            "renames the program-selection section 'filter.lfs'",
        ),
        (
            "git config --rename-section merge ours2",
            "renames the program-selection section 'merge'",
        ),
        (
            "git config --rename-section diff ours",
            "renames the program-selection section 'diff'",
        ),
        # The round-3 sections name their destination/matched section the
        # same way.
        (
            "git config --rename-section other man",
            "renames a section to 'man', a program-selection section",
        ),
        (
            "git config --rename-section man.woman other",
            "renames the program-selection section 'man.woman'",
        ),
        (
            "git config --rename-section other interactive",
            "renames a section to 'interactive', a program-selection section",
        ),
        (
            "git config --rename-section interactive.diffFilter other",
            "renames the program-selection section 'interactive.diffFilter'",
        ),
    ],
)
def test_rename_denial_names_the_destination_or_matched_section(
    command: str, expected: str
) -> None:
    result = classify(command)
    assert result.kind == "hard_denied"
    assert result.reason
    assert expected in result.reason, command


@pytest.mark.parametrize(
    "command",
    [
        # Every git remote mutation redefines where a granted push lands:
        # add/set-url point a bare destination name at an arbitrary URL or
        # path, rename retargets an existing remote, and remove turns a
        # configured bare name back into a URL git resolves as a path.
        "git remote add origin ../x",
        "git remote add origin https://example.com/x.git",
        "git remote add -f origin ../x",
        "git remote add --mirror=push origin ../x",
        "git remote set-url origin ../x",
        "git remote set-url --push origin ../x",
        "git remote rename origin evil",
        "git remote remove origin",
        "git remote rm origin",
        # The global verbose flag does not launder a mutation.
        "git remote -v add origin ../x",
        "git remote --verbose set-url origin ../x",
    ],
)
def test_git_remote_mutations_are_hard_guards(command: str) -> None:
    result = classify(command)
    assert result.kind == "hard_denied"
    assert result.grantable is None
    assert result.reason
    assert "redefines where a granted push lands" in result.reason
    assert "no grant can authorize" in result.reason
    assert "manually" in result.reason


@pytest.mark.parametrize(
    "command",
    [
        # Reads and non-mutating forms stay routine. git remote update and
        # prune stay routine too: with remote.<name>.vcs and core.sshCommand
        # writes barred, the helper and ssh program they run are
        # user-configured, so they are in-contract.
        "git remote",
        "git remote -v",
        "git remote --verbose",
        "git remote get-url origin",
        "git remote show origin",
        "git remote show -n origin",
        "git remote prune origin",
        "git remote update",
        "git remote set-head origin main",
        "git remote set-branches origin main",
    ],
)
def test_git_remote_reads_stay_routine(command: str) -> None:
    assert classify(command).kind == "routine"


def test_git_remote_routes_through_the_dangerous_flow() -> None:
    # remote is routed so a hard-guarded mutation is denied by the
    # resolver; routine remote forms stay governed by the existing policy
    # layer.
    assert is_dangerous_git_subcommand(shlex.split("git remote add origin ../x"))
    assert is_dangerous_git_subcommand(shlex.split("git remote -v"))


@pytest.mark.parametrize(
    "command",
    [
        # A bare-name destination is name-shaped, not a path or URL: the
        # classifier stays pure and cannot inspect the repository, so the
        # resolver verifies the name against the repository's configured
        # remotes (git resolves an unconfigured destination as a URL).
        "git push evil main",
        "git push remote.git main",
        "git push upstream main",
        "git push --repo=evil main",
        "git push --repo evil main",
        "git push --repo=remote.git main",
    ],
)
def test_bare_name_destinations_are_grantable_at_the_pure_layer(command: str) -> None:
    assert_grantable(command, ("git.push",))


def test_bare_name_destinations_carry_the_destination_operands() -> None:
    # The modeled call carries both destination operands — the first
    # positional and every --repo value — so the resolver can verify them.
    result = classify("git push --repo=upstream origin main")
    assert result.kind == "grantable"
    assert result.grantable is not None
    assert result.grantable.call.subcommand == "push"
    assert result.grantable.call.operands == ("origin", "main")
    assert result.grantable.call.option_values == (("--repo", "upstream"),)


def test_config_then_push_compound_denies_on_the_config_segment() -> None:
    # The exploit shape the guard exists for: the config segment is a hard
    # guard, so the whole compound call is denied with no proposal even
    # though the push segment alone is ordinary publication.
    config_segment = classify_dangerous_git_command(
        shlex.split("git config remote.origin.mirror true")
    )
    push_segment = classify_dangerous_git_command(shlex.split("git push origin"))
    assert config_segment.kind == "hard_denied"
    assert config_segment.grantable is None
    assert push_segment.kind == "grantable"


def test_include_then_push_compound_denies_on_the_include_segment() -> None:
    # The F-1 exploit shape at the pure layer: an authored include folds a
    # file that can set remote.<name>.pushurl for a remote the resolver
    # verified, so the include write is a hard guard and the compound call
    # is denied with no proposal even though the push alone is grantable.
    include_segment = classify_dangerous_git_command(
        shlex.split("git config include.path /tmp/evil.cfg")
    )
    push_segment = classify_dangerous_git_command(shlex.split("git push origin"))
    assert include_segment.kind == "hard_denied"
    assert include_segment.grantable is None
    assert push_segment.kind == "grantable"
    assert push_segment.grantable is not None
    assert push_segment.grantable.operation_ids == ("git.push",)


@pytest.mark.parametrize(
    "command",
    [
        "git push --bogus origin main",
        # Ambiguous abbreviations fail closed.
        "git push --force- origin main",
        "git push --f origin main",
        # Values given to boolean options are invalid forms.
        "git push --force=x origin main",
        "git push --mirror=1 origin",
        "git push --delete=1 origin main",
        # Options that select the program git runs are never grantable.
        "git push --receive-pack=echo origin main",
        "git push --receive-p=echo origin main",
        "git push --exec=echo origin main",
        # Unmodeled refspec forms fail closed.
        "git push origin main:",
        "git push origin a:b:c",
        "git push origin refs/heads/*:refs/heads/*",
        "git push origin ''",
        # Missing operands fail closed.
        "git push --repo",
        "git push --repo= origin main",
        "git push -o",
        # Unknown or ambiguous globals fail closed.
        "git --bogus push",
        "git -C",
        "git --git-dir",
        "git --git-dir= push",
        "git -vC . push",
        # Switch classification depends on its options.
        "git switch --bogus main",
        "git switch -c",
    ],
)
def test_unknown_ambiguous_and_invalid_forms_never_grantable(command: str) -> None:
    # Unknown or ambiguous forms never fall into an ordinary git.push.
    assert_hard_denied(command)


@pytest.mark.parametrize(
    "command",
    [
        "git push -- origin main",
        "git push origin -- main",
        "git push --force -- origin main",
        "git -- push",
        "git checkout -- file",
        "git restore -- file",
        "git switch --discard-changes -- main",
        "git stash drop --",
        "git reflog expire --all --",
    ],
)
def test_end_of_options_fails_closed_for_grantable_forms(command: str) -> None:
    result = classify(command)
    assert result.kind == "hard_denied"
    assert result.grantable is None
    assert result.reason == "end-of-options marker is not a modeled form"


@pytest.mark.parametrize(
    "command",
    [
        "git log -- file",
        "git status -- file",
        "git switch -- main",
        "git stash list -- x",
        "git reset --soft -- HEAD",
        "git -- status",
    ],
)
def test_end_of_options_keeps_routine_forms_routine(command: str) -> None:
    # `--` is never grantable, but forms that are routine without it stay
    # routine: readers are not dangerous-command operations.
    assert classify(command).kind == "routine"


@pytest.mark.parametrize(
    "command",
    [
        "git status",
        "git log --oneline",
        "git log -- file",
        "git diff",
        "git show HEAD",
        "git branch",
        "git branch -D x",
        "git tag",
        "git commit -m x",
        "git rebase main",
        "git fetch origin",
        "git clone ../x",
        "git worktree add ../x",
        "git config user.name x",
        "git help push",
        "git",
        "git --version",
        "git -C .",
        "git -- status",
    ],
)
def test_readers_and_unclassified_subcommands_are_routine(command: str) -> None:
    # Readers and recognized non-registry subcommands keep their existing
    # policy treatment: no dangerous-command involvement.
    assert classify(command).kind == "routine"


@pytest.mark.parametrize(
    "command",
    [
        # v0.5.9 P2-d: symbolic-ref is modeled by mode — the read form
        # (exactly one operand, no write marker) carries no dangerous
        # operation.
        "git symbolic-ref HEAD",
        "git symbolic-ref -q HEAD",
        "git symbolic-ref --short HEAD",
        "git symbolic-ref --recurse HEAD",
        "git symbolic-ref --no-recurse HEAD",
        "git -C . symbolic-ref HEAD",
    ],
)
def test_symbolic_ref_read_forms_are_routine(command: str) -> None:
    assert classify(command).kind == "routine"


@pytest.mark.parametrize(
    "command",
    [
        # The write forms (a ref update or deletion) are not granted by the
        # read modeling: hard guards with no grantable operation.
        "git symbolic-ref HEAD refs/heads/main",
        "git symbolic-ref -m reason HEAD refs/heads/main",
        "git symbolic-ref --message reason HEAD refs/heads/main",
        "git symbolic-ref -m reason HEAD",
        "git symbolic-ref -d HEAD",
        "git symbolic-ref --delete HEAD",
        "git symbolic-ref -d -q HEAD",
        "git symbolic-ref HEAD refs/heads/main extra",
    ],
)
def test_symbolic_ref_write_forms_are_hard_denied(command: str) -> None:
    result = classify(command)
    assert result.kind == "hard_denied"
    assert result.grantable is None
    assert "write form" in (result.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        # Bare, operand-less, and unscannable forms fail closed too.
        "git symbolic-ref",
        "git symbolic-ref -q",
        "git symbolic-ref --recurse",
        "git symbolic-ref --bogus HEAD",
        "git symbolic-ref -x HEAD",
    ],
)
def test_symbolic_ref_unmodeled_forms_are_hard_denied(command: str) -> None:
    result = classify(command)
    assert result.kind == "hard_denied"
    assert result.grantable is None
    assert "write form" not in (result.reason or "")


@pytest.mark.parametrize(
    "command",
    [
        "echo hi",
        "ls -la",
        "rm -rf src",
        # Executor-wrapped git is not a git invocation at the top level; the
        # permission flow classifies expanded command parts.
        "env FOO=1 git push",
    ],
)
def test_non_git_calls_are_out_of_classifier_scope(command: str) -> None:
    assert classify(command).kind == "routine"


def test_existing_enforcement_is_unchanged_for_hard_guards() -> None:
    # The classifier adds no enforcement: the shipped policy layer still
    # hard-guards recursive rm and git reset --hard on its own.
    for command in ("rm -rf src", "git reset --hard"):
        policy = analyze_shell_command_policy(shlex.split(command))
        assert policy.requires_approval


def test_grantable_carries_operations_consequence_operands_and_relocation() -> None:
    result = classify("git -C sub --git-dir=. push --force origin main")
    assert result.kind == "grantable"
    assert result.grantable is not None
    grantable = result.grantable
    assert grantable.operation_ids == ("git.push.force",)
    assert "forced push" in grantable.consequence
    assert grantable.call.subcommand == "push"
    assert grantable.call.operands == ("origin", "main")
    assert grantable.relocation == GitRepositoryRelocation(
        cwd_paths=("sub",), git_dir=".", work_tree=None
    )
    operation = grantable.operations[0]
    assert operation.operation_id == "git.push.force"
    assert operation.command_class == "consequential"
    assert not operation.persistable


def test_modeled_operands_and_option_values() -> None:
    push = classify("git push --force --repo=upstream origin main:dev +old:new")
    assert push.kind == "grantable"
    assert push.grantable is not None
    assert push.grantable.call.options == ("--force", "--repo")
    assert push.grantable.call.option_values == (("--repo", "upstream"),)
    assert push.grantable.call.operands == ("origin", "main:dev", "+old:new")

    checkout = classify("git checkout -b feat")
    assert checkout.kind == "grantable"
    assert checkout.grantable is not None
    assert checkout.grantable.call.subcommand == "checkout"
    assert checkout.grantable.call.option_values == (("-b", "feat"),)
    assert checkout.grantable.call.operands == ()

    stash = classify("git stash drop stash@{1}")
    assert stash.kind == "grantable"
    assert stash.grantable is not None
    assert stash.grantable.call.subcommand == "stash"
    assert stash.grantable.call.operands == ("drop", "stash@{1}")


def test_classifier_is_pure_and_repeatable() -> None:
    first = classify("git push --force origin main")
    second = classify("git push --force origin main")
    assert first == second
    assert first.kind == "grantable"


def test_classification_requires_payload_exactly_when_grantable() -> None:
    with pytest.raises(ValueError, match="grantable"):
        DangerousCommandClassification(kind="grantable")
    grantable = classify("git push").grantable
    assert grantable is not None
    with pytest.raises(ValueError, match="grantable"):
        DangerousCommandClassification(kind="routine", grantable=grantable)


def test_registry_matches_the_frozen_contract() -> None:
    assert tuple(OPERATION_REGISTRY) == FROZEN_OPERATION_IDS
    assert PUBLICATION_OPERATION_IDS == frozenset({"git.push", "network.egress"})
    assert CONSEQUENTIAL_OPERATION_IDS == {
        operation_id
        for operation_id in FROZEN_OPERATION_IDS
        if OPERATION_REGISTRY[operation_id].command_class == "consequential"
    }
    persistable = [
        operation_id
        for operation_id, operation in OPERATION_REGISTRY.items()
        if operation.persistable
    ]
    # Only publication is persistable.
    assert persistable == ["git.push", "network.egress"]
    for operation in OPERATION_REGISTRY.values():
        assert operation.consequence
        assert operation.command_class in {"publication", "consequential"}


def test_operations_for_ids_resolves_in_registry_order() -> None:
    operations = operations_for_ids(("git.push.mirror", "git.push.force", "git.push"))
    assert [operation.operation_id for operation in operations] == [
        "git.push",
        "git.push.force",
        "git.push.mirror",
    ]


def test_operations_for_ids_rejects_unregistered_ids() -> None:
    with pytest.raises(ValueError, match="unregistered"):
        operations_for_ids(("git.push", "git.nonsense"))


@pytest.mark.parametrize(
    "choice", ["deny", "allow_once", "allow_session", "allow_project"]
)
def test_publication_offers_every_grant_choice(choice: GrantChoice) -> None:
    assert grant_choices_for("publication") == (
        "deny",
        "allow_once",
        "allow_session",
        "allow_project",
    )
    assert is_grant_choice_valid("git.push", choice)


@pytest.mark.parametrize("choice", ["deny", "allow_once", "allow_session"])
def test_consequential_offers_session_grant_choices(choice: GrantChoice) -> None:
    assert grant_choices_for("consequential") == ("deny", "allow_once", "allow_session")
    assert is_grant_choice_valid("git.push.force", choice)


@pytest.mark.parametrize("operation_id", sorted(CONSEQUENTIAL_OPERATION_IDS))
def test_consequential_never_accepts_a_persistent_grant(operation_id: str) -> None:
    # A forged "always" answer is rejected even if a client sends it.
    assert not is_grant_choice_valid(operation_id, "allow_project")


@pytest.mark.parametrize(
    "choice", ["deny", "allow_once", "allow_session", "allow_project"]
)
def test_unregistered_operation_accepts_no_choice(choice: GrantChoice) -> None:
    assert not is_grant_choice_valid("git.nonsense", choice)


def test_authority_and_proposal_types_carry_the_whole_call() -> None:
    scope = RepositoryScope(git_dir=Path("/repo/.git"), work_tree=Path("/repo"))
    authority = DangerousCommandAuthority(
        operations=frozenset({"git.push"}), scope=scope, source="protected_config"
    )
    assert authority.operations == frozenset({"git.push"})
    assert authority.scope is scope
    assert authority.source == "protected_config"
    proposal = DangerousCommandProposal(
        command=("git", "push", "origin", "main"),
        project=Path("/repo"),
        scope=scope,
        operations=("git.push",),
        consequence="Publish local commits to the remote.",
    )
    assert proposal.command == ("git", "push", "origin", "main")
    assert proposal.project == Path("/repo")
    assert proposal.scope is scope
    assert proposal.operations == ("git.push",)
    # The authority and proposal types are immutable.
    for target, attribute in ((proposal, "consequence"), (authority, "operations")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(target, attribute, None)
