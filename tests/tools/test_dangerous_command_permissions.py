"""Package P3: whole-call dangerous-command resolution, authority, parity.

The resolver classifies the entire expanded shell call (pipelines, lists,
nested shells, supported executor wrappers, scoped cwd changes) with the P1
classifier, collects every dangerous-operation gap, and denies with a typed
whole-call proposal only when the missing dangerous authority is the sole
missing authority. Authority comes from accepted saved grants (the P2 read
side) and an injected runtime provider; with no runtime approval handler
wired, dangerous calls fail closed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import shlex
import subprocess

import pytest

from chartreux.core.config._dangerous_command_authority import GRANTS_FIELD
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.dangerous_commands import (
    DangerousCommandAuthority,
    GitRepositoryRelocation,
    RepositoryScope,
)
from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import resolve_repository_scope
from chartreux.core.tools.builtins._shell_permission_resolver import (
    DangerousCommandAuthorityProvider,
    ShellPermissionResolver,
    _get_default_denylist,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.builtins.bash_start import BashStart, BashStartArgs
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.permissions import PermissionContext
from tests.conftest import build_test_vibe_config


def make_repo(path: Path, *, remotes: tuple[str, ...] = ("origin",)) -> Path:
    """A repository with the named remotes configured (origin by default).

    A bare-name push destination resolves only through a configured
    remote, so a realistic grantable-push fixture configures one; pass
    ``remotes=()`` for a repository git resolves no destination in.
    """
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    for name in remotes:
        subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "remote",
                "add",
                name,
                f"https://example.com/{name}.git",
            ],
            check=True,
        )
    return path


def scope_of(path: Path) -> RepositoryScope:
    scope = resolve_repository_scope(path, GitRepositoryRelocation())
    assert scope is not None
    return scope


def authority_for(
    scope: RepositoryScope, *operation_ids: str, source: str = "runtime_grant"
) -> DangerousCommandAuthority:
    return DangerousCommandAuthority(
        operations=frozenset(operation_ids),
        scope=scope,
        source=source,  # type: ignore[arg-type]
    )


class AuthorityProvider:
    """Injected immutable test authority (the runtime port P4 implements)."""

    def __init__(self, authorities: tuple[DangerousCommandAuthority, ...] = ()):
        self.authorities = authorities
        self.calls = 0

    def dangerous_command_authorities(self) -> tuple[DangerousCommandAuthority, ...]:
        self.calls += 1
        return self.authorities


def make_bash(
    cwd: Path, authorities: tuple[DangerousCommandAuthority, ...] = (), **config: object
) -> Bash:
    config_model = BashToolConfig(**config)  # type: ignore[arg-type]
    tool = Bash(config_getter=lambda: config_model, state=BaseToolState(), cwd=cwd)
    provider = AuthorityProvider(authorities)
    tool.dangerous_command_authorities_getter = provider.dangerous_command_authorities
    return tool


def manager_for(
    cwd: Path,
    *,
    parent: ToolManager | None = None,
    restrictions: tuple[SourceRestrictions, ...] = (),
    provider: DangerousCommandAuthorityProvider | None = None,
    **config: object,
) -> ToolManager:
    settings = build_test_vibe_config(**config)  # type: ignore[arg-type]
    return ToolManager(
        lambda: settings,
        cwd=cwd,
        defer_mcp=True,
        restriction_getter=(lambda: restrictions) if restrictions else None,
        accepted_token_getter=lambda: "test-token",
        parent_authority_getter=(lambda: parent) if parent else None,
        dangerous_command_authority_provider=provider,
    )


def saved_grant_source(
    tmp_path: Path, project: Path, operations: tuple[str, ...]
) -> SourceRestrictions:
    """One accepted saved grant, projected through the P2 read side."""
    user_layer = UserConfigLayer(path=tmp_path / "user.toml")
    return SourceRestrictions.from_raw(
        {GRANTS_FIELD: {str(project): list(operations)}},
        layer_name=user_layer.name,
        locator=user_layer.source_locator,
        kind="source",
        store_fingerprint="test",
        root_layer=user_layer,
    )


def assert_dangerous_denial(
    context: PermissionContext | None, operation_ids: tuple[str, ...]
) -> None:
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind == "dangerous_command"
    assert context.reason
    proposal = context.dangerous_command_proposal
    assert proposal is not None
    assert proposal.operations == operation_ids
    assert proposal.command
    assert proposal.consequence


def assert_other_denial(context: PermissionContext | None) -> None:
    """A non-dangerous denial: never the grantable kind, never a proposal."""
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind is None
    assert context.dangerous_command_proposal is None
    assert context.reason


class TestWholeCallClassification:
    """Every expanded-command shape participates in the classification."""

    @pytest.mark.parametrize(
        ("command", "operation_ids"),
        [
            # Pipelines.
            ("git push | cat", ("git.push",)),
            ("git push origin main | tee out.txt", ("git.push",)),
            # Lists and conditionals.
            ("git push && git status", ("git.push",)),
            ("git push || true", ("git.push",)),
            ("true; git push", ("git.push",)),
            ("git status; git push; git status", ("git.push",)),
            # Nested shells.
            ("sh -c 'git push'", ("git.push",)),
            ("bash -c 'git -C . push origin main'", ("git.push",)),
            # Supported executor wrappers.
            ("command git push", ("git.push",)),
            ("builtin git push", ("git.push",)),
            ("env FOO=1 git push", ("git.push",)),
            ("env FOO=1 command git push", ("git.push",)),
            ("timeout 10 git push", ("git.push",)),
            ("nice -n 5 git push", ("git.push",)),
            # Scoped cwd changes and git relocation.
            ("cd sub && git push", ("git.push",)),
            ("git -C sub push", ("git.push",)),
            ("git -C . push", ("git.push",)),
            ("git --no-pager push", ("git.push",)),
            # Multiple dangerous segments collect every gap.
            ("git push && git checkout main", ("git.push", "git.checkout")),
            ("git checkout main; git stash drop", ("git.checkout", "git.stash.drop")),
            (
                "git push --force && git stash clear",
                ("git.push.force", "git.stash.clear"),
            ),
            ("sh -c 'git push' && git restore file.txt", ("git.push", "git.restore")),
        ],
    )
    def test_shapes_are_classified_per_git_segment(
        self, tmp_path: Path, command: str, operation_ids: tuple[str, ...]
    ):
        repo = make_repo(tmp_path / "project")
        (repo / "sub").mkdir()
        tool = make_bash(repo)
        context = tool.resolve_permission(BashArgs(command=command))
        assert_dangerous_denial(context, operation_ids)
        assert context is not None
        proposal = context.dangerous_command_proposal
        assert proposal is not None
        assert proposal.project == repo
        assert proposal.scope == scope_of(repo)
        assert proposal.command == tuple(shlex.split(command))

    def test_scoped_cwd_change_resolves_the_repository_scope(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        (repo / "sub").mkdir()
        scope = scope_of(repo)
        tool = make_bash(repo, (authority_for(scope, "git.push"),))
        # Both statically possible cwds walk up to the same repository.
        context = tool.resolve_permission(BashArgs(command="cd sub && git push"))
        assert context is not None
        assert context.permission is ToolPermission.ALWAYS

    def test_proposal_summarizes_the_whole_call(self, tmp_path: Path):
        tool = make_bash(tmp_path)
        context = tool.resolve_permission(
            BashArgs(command="git push && git checkout main")
        )
        assert context is not None
        proposal = context.dangerous_command_proposal
        assert proposal is not None
        assert proposal.operations == ("git.push", "git.checkout")
        assert "Publish local commits" in proposal.consequence
        assert "overwrites working-tree" in proposal.consequence
        assert "git.push" in (context.reason or "")
        assert "git.checkout" in (context.reason or "")

    def test_multi_repo_proposal_is_complete_but_the_scope_is_representative(
        self, tmp_path: Path
    ):
        # A whole call spanning repositories collects every missing operation
        # in registry order, while the proposal's scope is representative only:
        # the first missing operation's scope, never the complete scope set.
        repo_a = make_repo(tmp_path / "a")
        repo_b = make_repo(tmp_path / "b")
        scope_a = scope_of(repo_a)
        scope_b = scope_of(repo_b)
        assert scope_a != scope_b
        tool = make_bash(tmp_path)
        for command, representative in (
            ("git -C a push && git -C b checkout", scope_a),
            ("git -C b checkout && git -C a push", scope_b),
        ):
            context = tool.resolve_permission(BashArgs(command=command))
            # The operations are complete and registry-ordered in both orders.
            assert_dangerous_denial(context, ("git.push", "git.checkout"))
            assert context is not None
            proposal = context.dangerous_command_proposal
            assert proposal is not None
            assert proposal.operations == ("git.push", "git.checkout")
            # The scope follows the first missing operation, not the whole set.
            assert proposal.scope == representative


class TestFailClosedCheckpoint:
    """With no runtime approval handler wired, dangerous calls fail closed."""

    @pytest.mark.parametrize(
        "command",
        [
            "git push",
            "git push origin main",
            "git push --force origin main",
            "git checkout main",
            "git restore file.txt",
            "git stash drop",
            "git stash clear",
            "git switch --discard-changes main",
            "git switch -f main",
            "git reflog expire --all",
            "git reflog delete main",
        ],
    )
    def test_dangerous_calls_fail_closed_without_authority(
        self, tmp_path: Path, command: str
    ):
        tool = make_bash(tmp_path)
        context = tool.resolve_permission(BashArgs(command=command))
        assert context is not None
        assert context.permission is ToolPermission.NEVER
        assert context.denial_kind == "dangerous_command"
        assert context.dangerous_command_proposal is not None
        assert context.reason

    def test_routine_commands_are_unaffected(self, tmp_path: Path):
        tool = make_bash(tmp_path)
        for command in (
            "echo hi",
            "git status",
            "git log --oneline",
            "git switch main",
            "git stash pop",
            "git branch -D feature",
            "git tag -d v1",
            "git reflog show",
            "git commit -m x",
        ):
            context = tool.resolve_permission(BashArgs(command=command))
            assert context is not None, command
            assert context.permission is ToolPermission.ALWAYS, command
            assert context.denial_kind is None, command
            assert context.dangerous_command_proposal is None, command

    def test_unmodeled_git_forms_stay_with_the_policy_layer(self, tmp_path: Path):
        # Readers that are unmodeled builtins, unrecognized subcommands, and
        # dangling globals without a subcommand carry no dangerous operation:
        # the existing policy layer governs them, and its enforcement is
        # unchanged.
        tool = make_bash(tmp_path)
        for command in (
            "git describe --tags",
            "git shortlog",
            "git count-objects",
            "git merge-base HEAD HEAD",
            "git -C",
            "git --git-dir",
        ):
            context = tool.resolve_permission(BashArgs(command=command))
            assert context is not None, command
            assert context.permission is ToolPermission.ALWAYS, command
            assert context.dangerous_command_proposal is None, command

    def test_background_job_recovery_commands_are_unaffected(self, tmp_path: Path):
        # Recovery tools never resolve shell source; routine launches stay
        # permitted without any dangerous-command authority.
        manager = manager_for(tmp_path)
        for name in ("bash_start", "bash_read", "bash_stop", "bash_list"):
            context = manager.get(name).resolve_permission(
                BashStartArgs(command="sleep 30")
            )
            assert context is None or context.permission is ToolPermission.ALWAYS


class TestOrderingInvariant:
    """dangerous_command is returned only as the sole missing authority."""

    @pytest.mark.parametrize("order", ["first", "second"])
    def test_denylist_veto_is_never_masked(self, tmp_path: Path, order: str):
        command = "git push && vim x" if order == "first" else "vim x && git push"
        context = make_bash(tmp_path).resolve_permission(BashArgs(command=command))
        assert_other_denial(context)
        assert "matches denylist pattern 'vim'" in (context.reason or "")  # type: ignore[union-attr]

    @pytest.mark.parametrize("order", ["first", "second"])
    def test_sensitive_rule_is_never_masked(self, tmp_path: Path, order: str):
        command = "git push && sudo ls" if order == "first" else "sudo ls && git push"
        context = make_bash(tmp_path).resolve_permission(BashArgs(command=command))
        assert_other_denial(context)
        assert "sensitive command rule" in (context.reason or "")  # type: ignore[union-attr]

    @pytest.mark.parametrize("order", ["first", "second"])
    def test_path_restriction_is_never_masked(self, tmp_path: Path, order: str):
        command = (
            "git push && cat /etc/hostname"
            if order == "first"
            else "cat /etc/hostname && git push"
        )
        context = make_bash(tmp_path).resolve_permission(BashArgs(command=command))
        assert_other_denial(context)
        assert "outside the authorized workspace" in (context.reason or "")  # type: ignore[union-attr]

    @pytest.mark.parametrize("order", ["first", "second"])
    def test_policy_hard_guard_is_never_masked(self, tmp_path: Path, order: str):
        command = (
            "git push && git reset --hard"
            if order == "first"
            else "git reset --hard && git push"
        )
        context = make_bash(tmp_path).resolve_permission(BashArgs(command=command))
        assert_other_denial(context)

    @pytest.mark.parametrize("order", ["first", "second"])
    def test_protected_metadata_is_never_masked(self, tmp_path: Path, order: str):
        repo = make_repo(tmp_path / "project")
        command = (
            "git push && echo x > .git/config"
            if order == "first"
            else "echo x > .git/config && git push"
        )
        context = make_bash(repo).resolve_permission(BashArgs(command=command))
        assert_other_denial(context)
        assert "protected Git metadata" in (context.reason or "")  # type: ignore[union-attr]

    def test_tool_disablement_is_never_masked(self, tmp_path: Path):
        tool = make_bash(tmp_path, permission=ToolPermission.NEVER)
        context = tool.resolve_permission(BashArgs(command="git push"))
        assert_other_denial(context)
        assert "Tool denied: bash" in (context.reason or "")  # type: ignore[union-attr]

    def test_syntax_approval_is_never_masked(self, tmp_path: Path):
        context = make_bash(tmp_path).resolve_permission(
            BashArgs(command="git push origin $(whoami)")
        )
        assert_other_denial(context)
        assert "unsupported shell syntax" in (context.reason or "")  # type: ignore[union-attr]

    def test_hard_denied_git_segment_masks_a_grantable_gap(self, tmp_path: Path):
        # Unmodeled forms fail closed for the whole call: no proposal.
        for command in (
            "git push --bogus origin main && git push",
            "git push && git push --bogus origin main",
            "git checkout -- file.txt && git push",
        ):
            context = make_bash(tmp_path).resolve_permission(BashArgs(command=command))
            assert_other_denial(context)

    def test_unlocatable_repository_is_hard_denied(self, tmp_path: Path):
        (tmp_path / ".git").write_text("not a gitdir pointer")
        context = make_bash(tmp_path).resolve_permission(BashArgs(command="git push"))
        assert_other_denial(context)
        assert "cannot be located" in (context.reason or "")  # type: ignore[union-attr]

    def test_unknown_cwd_is_hard_denied(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        tool = make_bash(repo, (authority_for(scope_of(repo), "git.push"),))
        context = tool.resolve_permission(BashArgs(command="popd extra && git push"))
        assert_other_denial(context)
        assert "not statically known" in (context.reason or "")  # type: ignore[union-attr]


class TestPublicationAuthority:
    """git.push authority permits only ordinary publication."""

    @pytest.mark.parametrize(
        "command",
        [
            "git push",
            "git push origin main",
            "git push origin main:main",
            "git push --all origin",
            "git push -u origin main",
            "git push --dry-run origin main",
        ],
    )
    def test_ordinary_push_is_authorized(self, tmp_path: Path, command: str):
        repo = make_repo(tmp_path / "project")
        tool = make_bash(repo, (authority_for(scope_of(repo), "git.push"),))
        context = tool.resolve_permission(BashArgs(command=command))
        assert context is not None
        assert context.permission is ToolPermission.ALWAYS

    @pytest.mark.parametrize(
        ("command", "operation_ids"),
        [
            ("git push --force origin main", ("git.push.force",)),
            ("git push -f origin main", ("git.push.force",)),
            ("git push --force-with-lease origin main", ("git.push.force",)),
            ("git push origin +main", ("git.push.force",)),
            ("git push origin :main", ("git.push.delete",)),
            ("git push --delete origin main", ("git.push.delete",)),
            ("git push --mirror origin", ("git.push.mirror",)),
            (
                "git push --force --delete origin main",
                ("git.push.force", "git.push.delete"),
            ),
        ],
    )
    def test_consequential_push_never_inherits_git_push(
        self, tmp_path: Path, command: str, operation_ids: tuple[str, ...]
    ):
        repo = make_repo(tmp_path / "project")
        tool = make_bash(repo, (authority_for(scope_of(repo), "git.push"),))
        context = tool.resolve_permission(BashArgs(command=command))
        assert_dangerous_denial(context, operation_ids)
        assert context is not None
        assert context.dangerous_command_proposal is not None
        assert context.dangerous_command_proposal.scope == scope_of(repo)


class TestPushDestinationVerification:
    """A bare-name push destination must resolve to a configured remote.

    git resolves a push destination that is not a configured remote name
    as a URL — a path, for a bare name — so a push to an unconfigured
    name runs the destination's receive-pack and hooks. The resolver
    verifies every destination operand (the first positional and every
    ``--repo`` value) against the configured remotes of every repository
    the command may run in, and fails closed otherwise; the classifier
    stays pure and never inspects the repository.
    """

    def test_configured_remote_destination_is_authorized(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        (repo / "sub").mkdir()
        tool = make_bash(repo, (authority_for(scope_of(repo), "git.push"),))
        for command in (
            "git push origin main",
            "git push origin main:main",
            "git push --all origin",
            "git push --repo=origin",
            "git push --repo origin",
            # Both statically possible cwds resolve the same repository.
            "cd sub && git push origin main",
        ):
            context = tool.resolve_permission(BashArgs(command=command))
            assert context is not None, command
            assert context.permission is ToolPermission.ALWAYS, command

    def test_unconfigured_destination_fails_closed(self, tmp_path: Path):
        # No remote is configured: git would resolve the bare name as a
        # path and run the destination's receive-pack and hooks.
        repo = make_repo(tmp_path / "project", remotes=())
        tool = make_bash(repo, (authority_for(scope_of(repo), "git.push"),))
        for command in (
            "git push origin main",
            "git push evil main",
            "git push remote.git main",
            "git push --repo=origin",
            "git push --repo evil",
        ):
            context = tool.resolve_permission(BashArgs(command=command))
            assert context is not None, command
            assert context.permission is ToolPermission.NEVER, command
            # A hard guard: no grantable proposal, never a
            # dangerous_command denial.
            assert context.denial_kind is None, command
            assert context.dangerous_command_proposal is None, command
            assert "not a configured remote" in (context.reason or ""), command
            assert "receive-pack" in (context.reason or ""), command

    def test_both_destination_operands_are_verified(self, tmp_path: Path):
        # The first positional is the repository and --repo overrides it,
        # so both operands are verified: a configured --repo value never
        # launders an unconfigured positional, and the deprecated --repo
        # spelling is verified conservatively too. (The rendered reason
        # escapes quotes, so the assertions avoid them.)
        repo = make_repo(tmp_path / "project")
        tool = make_bash(repo, (authority_for(scope_of(repo), "git.push"),))
        for command, destination in (
            ("git push --repo=origin evil", "evil"),
            ("git push --repo=evil origin", "evil"),
        ):
            context = tool.resolve_permission(BashArgs(command=command))
            assert context is not None, command
            assert context.permission is ToolPermission.NEVER, command
            assert context.denial_kind is None, command
            assert context.dangerous_command_proposal is None, command
            assert "not a configured remote" in (context.reason or ""), command
            assert destination in (context.reason or ""), command

    def test_remote_without_url_is_not_resolvable(self, tmp_path: Path):
        # A remote with only mirror/push config is NOT resolvable: git
        # resolves the destination as a URL without remote.<name>.url.
        repo = make_repo(tmp_path / "project", remotes=())
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "config",
                "remote.ghost.push",
                "+refs/heads/*:refs/heads/*",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "config", "remote.ghost.mirror", "true"],
            check=True,
        )
        tool = make_bash(repo, (authority_for(scope_of(repo), "git.push"),))
        context = tool.resolve_permission(BashArgs(command="git push ghost main"))
        assert context is not None
        assert context.permission is ToolPermission.NEVER
        assert context.denial_kind is None
        assert context.dangerous_command_proposal is None
        # The rendered reason escapes quotes, so assert without them.
        assert "not a configured remote" in (context.reason or "")
        assert "ghost" in (context.reason or "")

    def test_unreadable_repository_config_fails_closed(self, tmp_path: Path):
        # A config that cannot be read fails closed: the destination
        # cannot be verified as a configured remote.
        repo = make_repo(tmp_path / "project")
        config = repo / ".git" / "config"
        config.unlink()
        config.mkdir()
        tool = make_bash(repo, (authority_for(scope_of(repo), "git.push"),))
        context = tool.resolve_permission(BashArgs(command="git push origin main"))
        assert context is not None
        assert context.permission is ToolPermission.NEVER
        assert context.denial_kind is None
        assert context.dangerous_command_proposal is None
        assert "could not be read" in (context.reason or "")

    def test_no_destination_operand_stays_grantable(self, tmp_path: Path):
        # With no destination operand, git resolves the destination from
        # configuration the agent cannot write (and refuses the push when
        # nothing is configured), so the ordinary push stays grantable.
        repo = make_repo(tmp_path / "project", remotes=())
        tool = make_bash(repo, (authority_for(scope_of(repo), "git.push"),))
        context = tool.resolve_permission(BashArgs(command="git push"))
        assert context is not None
        assert context.permission is ToolPermission.ALWAYS

    def test_nominal_scope_is_not_verified(self, tmp_path: Path):
        # A directory governed by no repository is not verified: git
        # itself refuses the push there, so the call keeps its ordinary
        # dangerous-command proposal.
        plain = tmp_path / "plain"
        plain.mkdir()
        context = make_bash(plain).resolve_permission(
            BashArgs(command="git push origin main")
        )
        assert_dangerous_denial(context, ("git.push",))

    def test_explicit_destination_stays_a_hard_guard(self, tmp_path: Path):
        # Round-1 pin at resolver level: an explicit path or URL
        # destination is never grantable, even with authority.
        repo = make_repo(tmp_path / "project")
        tool = make_bash(repo, (authority_for(scope_of(repo), "git.push"),))
        for command in (
            "git push /tmp/evil.git main",
            "git push ../remote.git main",
            "git push --repo=/tmp/evil.git main",
        ):
            context = tool.resolve_permission(BashArgs(command=command))
            assert context is not None, command
            assert context.permission is ToolPermission.NEVER, command
            assert context.denial_kind is None, command
            assert context.dangerous_command_proposal is None, command
            assert "receive-pack" in (context.reason or ""), command

    def test_destination_verification_is_scoped_per_repository(self, tmp_path: Path):
        # The destination must resolve in every repository the command
        # may run in: authority for one repository never relocates the
        # check to another.
        repo_a = make_repo(tmp_path / "a")
        repo_b = make_repo(tmp_path / "b", remotes=())
        tool = make_bash(
            tmp_path,
            (
                authority_for(scope_of(repo_a), "git.push"),
                authority_for(scope_of(repo_b), "git.push"),
            ),
        )
        context = tool.resolve_permission(BashArgs(command="git -C a push origin main"))
        assert context is not None and context.permission is ToolPermission.ALWAYS
        context = tool.resolve_permission(BashArgs(command="git -C b push origin main"))
        assert context is not None
        assert context.permission is ToolPermission.NEVER
        assert context.denial_kind is None
        assert context.dangerous_command_proposal is None
        # The rendered reason escapes quotes, so assert without them.
        assert "not a configured remote" in (context.reason or "")
        assert "origin" in (context.reason or "")

    def test_git_dir_scope_verifies_against_its_own_config(self, tmp_path: Path):
        # A --git-dir relocation verifies against that repository's own
        # config, not the ambient repository's.
        bare = tmp_path / "bare.git"
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
        scope = RepositoryScope(git_dir=bare, work_tree=None)
        tool = make_bash(tmp_path, (authority_for(scope, "git.push"),))
        context = tool.resolve_permission(
            BashArgs(command=f"git --git-dir={bare} push origin main")
        )
        assert context is not None
        assert context.permission is ToolPermission.NEVER
        assert context.denial_kind is None
        assert context.dangerous_command_proposal is None
        subprocess.run(
            [
                "git",
                "-C",
                str(bare),
                "remote",
                "add",
                "origin",
                "https://example.com/b.git",
            ],
            check=True,
        )
        context = tool.resolve_permission(
            BashArgs(command=f"git --git-dir={bare} push origin main")
        )
        assert context is not None
        assert context.permission is ToolPermission.ALWAYS

    def test_linked_worktree_verifies_against_the_shared_config(self, tmp_path: Path):
        # A linked worktree's scope shares the common repository config
        # through commondir, so its remotes resolve like the main
        # worktree's.
        root = tmp_path
        repo = make_repo(root / "project")
        worktree = root / "wt"
        worktree.mkdir()
        git_dir = repo / ".git" / "worktrees" / "wt"
        git_dir.mkdir(parents=True)
        (git_dir / "commondir").write_text("../..")
        (worktree / ".git").write_text(f"gitdir: {git_dir}")
        scope_wt = scope_of(worktree)
        assert scope_wt != scope_of(repo)
        tool = make_bash(root, (authority_for(scope_wt, "git.push"),))
        context = tool.resolve_permission(
            BashArgs(command="git -C wt push origin main")
        )
        assert context is not None
        assert context.permission is ToolPermission.ALWAYS


class TestPushEffectingConfigWriteGuards:
    """git config writes to the push-effecting surface are hard guards.

    The resolver denies them with no grantable proposal even when ordinary
    publication authority exists, so a compound call that writes the
    configuration an approved push would run is denied on the config
    segment: the F-1 exploit shape (an authored include setting pushurl for
    a verified remote) never reaches the push.
    """

    def test_guarded_writes_are_denied_without_a_proposal(self, tmp_path: Path):
        # Every writer of the round-4 surface: the include directives, the
        # program-execution keys, the credential section, and remote.vcs.
        repo = make_repo(tmp_path / "project")
        tool = make_bash(repo, (authority_for(scope_of(repo), "git.push"),))
        for command in (
            "git config include.path /tmp/evil.cfg",
            "git config includeIf.gitdir:/x.path /tmp/evil.cfg",
            "git config core.sshCommand evil",
            "git config core.gitProxy evil",
            "git config core.askPass evil",
            "git config credential.helper evil",
            "git config credential.https://example.com.helper evil",
            "git config remote.origin.vcs evil",
        ):
            context = tool.resolve_permission(BashArgs(command=command))
            assert context is not None, command
            assert context.permission is ToolPermission.NEVER, command
            assert context.denial_kind is None, command
            assert context.dangerous_command_proposal is None, command
            assert "push-effecting configuration" in (context.reason or ""), command

    def test_include_then_push_compound_is_denied(self, tmp_path: Path):
        # The F-1 exploit shape: the include write is denied on its own
        # segment, so the push never sees the injected pushurl; the push
        # alone stays authorized through the verified remote.
        repo = make_repo(tmp_path / "project")
        tool = make_bash(repo, (authority_for(scope_of(repo), "git.push"),))
        context = tool.resolve_permission(
            BashArgs(command="git config include.path /tmp/evil.cfg && git push origin")
        )
        assert context is not None
        assert context.permission is ToolPermission.NEVER
        assert context.denial_kind is None
        assert context.dangerous_command_proposal is None
        assert "push-effecting configuration" in (context.reason or "")
        context = tool.resolve_permission(BashArgs(command="git push origin"))
        assert context is not None
        assert context.permission is ToolPermission.ALWAYS

    def test_guarded_key_reads_stay_authorized(self, tmp_path: Path):
        # Reads of the guarded keys never write, so they stay routine.
        repo = make_repo(tmp_path / "project")
        tool = make_bash(repo)
        for command in (
            "git config --get include.path",
            "git config --get core.sshCommand",
            "git config --get credential.helper",
            "git config --get remote.origin.vcs",
            "git config include.path",
        ):
            context = tool.resolve_permission(BashArgs(command=command))
            assert context is not None, command
            assert context.permission is ToolPermission.ALWAYS, command


class TestConsequentialSessionAuthority:
    """Session grants keyed by operation and scope; never persisted."""

    def test_consequential_session_authority_is_supported(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        scope = scope_of(repo)
        provider = AuthorityProvider((authority_for(scope, "git.push.force"),))
        manager = manager_for(repo, provider=provider)
        tool = manager.get("bash")
        context = tool.resolve_permission(
            BashArgs(command="git push --force origin main")
        )
        assert context is not None
        assert context.permission is ToolPermission.ALWAYS
        # The same session authority does not cover ordinary publication.
        assert_dangerous_denial(
            tool.resolve_permission(BashArgs(command="git push")), ("git.push",)
        )

    def test_session_authority_is_never_persisted(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        scope = scope_of(repo)
        authority = authority_for(scope, "git.push.force")
        provider = AuthorityProvider((authority,))
        restrictions = ()
        manager = manager_for(repo, restrictions=restrictions, provider=provider)
        tool = manager.get("bash")
        granted = tool.resolve_permission(BashArgs(command="git push --force"))
        assert granted is not None
        assert granted.permission is ToolPermission.ALWAYS
        # The resolver is read-only: the injected snapshot is untouched, no
        # saved grant appeared, and nothing was written to disk.
        assert provider.authorities == (authority,)
        assert manager._dangerous_command_authorities() == (
            DangerousCommandAuthority(
                operations=frozenset({"git.push.force"}),
                scope=scope,
                source="runtime_grant",
            ),
        )
        assert restrictions == ()
        assert not (tmp_path / "user.toml").exists()
        with pytest.raises(dataclasses.FrozenInstanceError):
            authority.operations = frozenset()  # type: ignore[misc]

    def test_broken_provider_fails_closed(self, tmp_path: Path):
        class BrokenProvider:
            def dangerous_command_authorities(self):
                raise RuntimeError("provider down")

        manager = manager_for(
            tmp_path,
            provider=BrokenProvider(),  # type: ignore[arg-type]
        )
        context = manager.get("bash").resolve_permission(BashArgs(command="git push"))
        assert_dangerous_denial(context, ("git.push",))


class TestRepositoryScopeIsolation:
    """Relocation and sibling worktrees cannot consume another scope's grant."""

    @pytest.fixture
    def layout(
        self, tmp_path: Path
    ) -> tuple[Path, Path, RepositoryScope, RepositoryScope]:
        # The workspace root holds both repositories, so relocation operands
        # stay inside the authorized workspace and reach the classifier.
        repo_a = make_repo(tmp_path / "a")
        repo_b = make_repo(tmp_path / "b")
        return tmp_path, repo_b, scope_of(repo_a), scope_of(repo_b)

    def test_other_repository_cannot_consume_the_grant(
        self, layout: tuple[Path, Path, RepositoryScope, RepositoryScope]
    ):
        root, repo_b, scope_a, scope_b = layout
        tool = make_bash(root, (authority_for(scope_a, "git.push"),))
        context = tool.resolve_permission(BashArgs(command="git -C a push"))
        assert context is not None and context.permission is ToolPermission.ALWAYS
        for command, expected_scope in (
            ("git -C b push", scope_b),
            (f"git -C {repo_b} push", scope_b),
            ("git --git-dir=b/.git push", RepositoryScope(repo_b / ".git", None)),
            (
                f"git --git-dir={repo_b / '.git'} push",
                RepositoryScope(repo_b / ".git", None),
            ),
        ):
            other = tool.resolve_permission(BashArgs(command=command))
            assert_dangerous_denial(other, ("git.push",))
            assert other is not None
            assert other.dangerous_command_proposal is not None
            assert other.dangerous_command_proposal.scope == expected_scope

    def test_git_dir_without_work_tree_is_a_distinct_scope(
        self, layout: tuple[Path, Path, RepositoryScope, RepositoryScope]
    ):
        root, repo_b, scope_a, _scope_b = layout
        tool = make_bash(root, (authority_for(scope_a, "git.push"),))
        context = tool.resolve_permission(BashArgs(command="git --git-dir=a/.git push"))
        assert_dangerous_denial(context, ("git.push",))
        assert context is not None
        assert context.dangerous_command_proposal is not None
        assert context.dangerous_command_proposal.scope == RepositoryScope(
            repo_b.parent / "a" / ".git", None
        )

    def test_sibling_worktree_cannot_consume_the_grant(self, tmp_path: Path):
        root = tmp_path
        repo = make_repo(root / "project")
        # Hand-built linked-worktree layout: a .git pointer into the shared
        # repository's worktree metadata.
        worktree = root / "wt"
        worktree.mkdir()
        git_dir = repo / ".git" / "worktrees" / "wt"
        git_dir.mkdir(parents=True)
        (git_dir / "commondir").write_text("../..")
        (worktree / ".git").write_text(f"gitdir: {git_dir}")
        scope_main = scope_of(repo)
        scope_wt = scope_of(worktree)
        assert scope_wt != scope_main

        tool = make_bash(root, (authority_for(scope_main, "git.push"),))
        context = tool.resolve_permission(BashArgs(command="git -C project push"))
        assert context is not None and context.permission is ToolPermission.ALWAYS
        context = tool.resolve_permission(BashArgs(command="git -C wt push"))
        assert_dangerous_denial(context, ("git.push",))
        assert context is not None
        assert context.dangerous_command_proposal is not None
        assert context.dangerous_command_proposal.scope == scope_wt

        # The sibling worktree's own scope is authorizable in its turn.
        granted = make_bash(root, (authority_for(scope_wt, "git.push"),))
        context = granted.resolve_permission(BashArgs(command="git -C wt push"))
        assert context is not None and context.permission is ToolPermission.ALWAYS
        # ... but it does not authorize the main worktree.
        assert_dangerous_denial(
            granted.resolve_permission(BashArgs(command="git -C project push")),
            ("git.push",),
        )

    def test_unknown_repository_identity_stays_denied(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        tool = make_bash(tmp_path, (authority_for(scope_of(repo), "git.push"),))
        # A directory governed by no repository has only a nominal scope,
        # which no real grant covers.
        plain = tmp_path / "plain"
        plain.mkdir()
        context = tool.resolve_permission(BashArgs(command="git -C plain push"))
        assert_dangerous_denial(context, ("git.push",))
        assert context is not None
        assert context.dangerous_command_proposal is not None
        assert context.dangerous_command_proposal.scope == RepositoryScope(
            plain / ".git", plain
        )


class TestSavedProjectGrants:
    """Accepted saved grants (P2 read side) authorize their project's repo."""

    def test_saved_grant_authorizes_ordinary_publication(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        source = saved_grant_source(tmp_path, repo, ("git.push",))
        manager = manager_for(repo, restrictions=(source,))
        tool = manager.get("bash")
        context = tool.resolve_permission(BashArgs(command="git push"))
        assert context is not None and context.permission is ToolPermission.ALWAYS
        # Only ordinary publication: force/delete/mirror never inherit it.
        assert_dangerous_denial(
            tool.resolve_permission(BashArgs(command="git push --force")),
            ("git.push.force",),
        )

    def test_saved_grant_scope_is_the_project_repository(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        other = make_repo(tmp_path / "other")
        source = saved_grant_source(tmp_path, repo, ("git.push",))
        # A manager rooted at the workspace: relocation stays in scope.
        root_manager = manager_for(tmp_path, restrictions=(source,))
        root_tool = root_manager.get("bash")
        context = root_tool.resolve_permission(BashArgs(command="git -C project push"))
        assert context is not None and context.permission is ToolPermission.ALWAYS
        # It never relocates to another repository.
        assert_dangerous_denial(
            root_tool.resolve_permission(BashArgs(command="git -C other push")),
            ("git.push",),
        )
        assert_dangerous_denial(
            root_tool.resolve_permission(BashArgs(command=f"git -C {other} push")),
            ("git.push",),
        )
        # A directory governed by no repository has only a nominal scope.
        assert_dangerous_denial(
            root_tool.resolve_permission(BashArgs(command="git push")), ("git.push",)
        )
        # A manager rooted inside the repository: the grant follows the
        # repository from any directory inside it.
        (repo / "src").mkdir()
        manager = manager_for(repo, restrictions=(source,))
        tool = manager.get("bash")
        context = tool.resolve_permission(BashArgs(command="cd src && git push"))
        assert context is not None and context.permission is ToolPermission.ALWAYS
        context = tool.resolve_permission(BashArgs(command="git -C src push"))
        assert context is not None and context.permission is ToolPermission.ALWAYS

    def test_saved_grant_for_a_subdirectory_covers_its_repository(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        sub = repo / "src"
        sub.mkdir()
        source = saved_grant_source(tmp_path, sub, ("git.push",))
        manager = manager_for(repo, restrictions=(source,))
        context = manager.get("bash").resolve_permission(BashArgs(command="git push"))
        assert context is not None and context.permission is ToolPermission.ALWAYS

    def test_saved_grant_for_another_project_does_not_apply(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        elsewhere = make_repo(tmp_path / "elsewhere")
        source = saved_grant_source(tmp_path, elsewhere, ("git.push",))
        manager = manager_for(repo, restrictions=(source,))
        assert_dangerous_denial(
            manager.get("bash").resolve_permission(BashArgs(command="git push")),
            ("git.push",),
        )

    def test_explicit_denylist_is_absolute_even_with_a_saved_grant(
        self, tmp_path: Path
    ):
        # A denylisted git.push never gets a proposal: the veto is checked
        # before grantability.
        repo = make_repo(tmp_path / "project")
        source = saved_grant_source(tmp_path, repo, ("git.push",))
        manager = manager_for(
            repo, restrictions=(source,), tools={"bash": {"denylist": ["git push"]}}
        )
        context = manager.get("bash").resolve_permission(BashArgs(command="git push"))
        assert_other_denial(context)
        assert "matches denylist pattern 'git push'" in (context.reason or "")  # type: ignore[union-attr]

    def test_restriction_source_denylist_is_absolute_even_with_a_saved_grant(
        self, tmp_path: Path
    ):
        # A denylist contributed through the restriction source (not the tool
        # config) vetoes absolutely: it is checked before grantability, so the
        # saved git.push grant loaded from the same source never yields a
        # proposal.
        repo = make_repo(tmp_path / "project")
        user_layer = UserConfigLayer(path=tmp_path / "user.toml")
        source = SourceRestrictions.from_raw(
            {
                GRANTS_FIELD: {str(repo): ["git.push"]},
                "tools": {"bash": {"denylist": ["git push"]}},
            },
            layer_name=user_layer.name,
            locator=user_layer.source_locator,
            kind="source",
            store_fingerprint="test",
            root_layer=user_layer,
        )
        manager = manager_for(repo, restrictions=(source,))
        # The saved grant is loaded from the restriction source ...
        authorities = manager._dangerous_command_authorities()
        assert len(authorities) == 1
        assert authorities[0].operations == frozenset({"git.push"})
        # ... but the restriction-source denylist vetoes before grantability.
        context = manager.get("bash").resolve_permission(BashArgs(command="git push"))
        assert_other_denial(context)
        assert "matches denylist pattern 'git push'" in (context.reason or "")  # type: ignore[union-attr]

    def test_non_persistable_operations_in_a_saved_grant_do_not_authorize(
        self, tmp_path: Path
    ):
        # P2 validation rejects non-persistable operations at load; the read
        # side filters them again so a consequential operation never persists
        # even if a restriction source carries one.
        from chartreux.core.config._dangerous_command_authority import (
            ProjectDangerousCommandGrant,
        )

        repo = make_repo(tmp_path / "project")
        source = SourceRestrictions(
            layer_name="user-toml",
            locator="user-toml",
            kind="source",
            content_fingerprint="test",
            store_fingerprint="test",
            tools=(),
            dangerous_command_grants=(
                ProjectDangerousCommandGrant(repo, ("git.push", "git.push.force")),
            ),
        )
        manager = manager_for(repo, restrictions=(source,))
        authorities = manager._dangerous_command_authorities()
        assert len(authorities) == 1
        assert authorities[0].operations == frozenset({"git.push"})
        assert authorities[0].source == "protected_config"
        assert_dangerous_denial(
            manager.get("bash").resolve_permission(
                BashArgs(command="git push --force")
            ),
            ("git.push.force",),
        )


class TestParentCeilings:
    """Parent invocation ceilings are checked even on a local gap."""

    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        return make_repo(tmp_path / "project")

    @pytest.mark.parametrize("tool_name", ["bash", "bash_start"])
    def test_parent_gap_classifies_as_parent_ceiling(self, repo: Path, tool_name: str):
        # Neither the child nor the parent holds authority: the parent's
        # dangerous-authority gap is a ceiling, never locally grantable.
        parent = manager_for(repo)
        child = manager_for(repo, parent=parent)
        args = (
            BashArgs(command="git push")
            if tool_name == "bash"
            else BashStartArgs(command="git push")
        )
        context = child.get(tool_name).resolve_permission(args)
        assert context is not None
        assert context.permission is ToolPermission.NEVER
        assert context.denial_kind == "parent_ceiling"
        assert context.dangerous_command_proposal is None

    @pytest.mark.parametrize("tool_name", ["bash", "bash_start"])
    def test_child_grant_cannot_lift_a_parent_gap(self, repo: Path, tool_name: str):
        # The child holds session authority the parent lacks: the parent
        # re-resolves under its own authority, so the gap is still a ceiling.
        parent = manager_for(repo)
        child = manager_for(
            repo,
            parent=parent,
            provider=AuthorityProvider((authority_for(scope_of(repo), "git.push"),)),
        )
        args = (
            BashArgs(command="git push")
            if tool_name == "bash"
            else BashStartArgs(command="git push")
        )
        context = child.get(tool_name).resolve_permission(args)
        assert context is not None
        assert context.permission is ToolPermission.NEVER
        assert context.denial_kind == "parent_ceiling"
        assert context.dangerous_command_proposal is None

    @pytest.mark.parametrize("tool_name", ["bash", "bash_start"])
    def test_local_gap_is_grantable_when_the_parent_allows(
        self, repo: Path, tool_name: str
    ):
        # The parent holds a saved grant the child lacks: the local gap is
        # the sole missing authority and carries the grantable proposal.
        source = saved_grant_source(repo.parent, repo, ("git.push",))
        parent = manager_for(repo, restrictions=(source,))
        child = manager_for(repo, parent=parent)
        args = (
            BashArgs(command="git push")
            if tool_name == "bash"
            else BashStartArgs(command="git push")
        )
        context = child.get(tool_name).resolve_permission(args)
        assert_dangerous_denial(context, ("git.push",))
        assert context is not None
        assert context.dangerous_command_proposal is not None
        assert context.dangerous_command_proposal.project == repo

    @pytest.mark.parametrize("tool_name", ["bash", "bash_start"])
    def test_parent_tool_disablement_is_never_masked(self, repo: Path, tool_name: str):
        parent = manager_for(repo, tools={"bash": {"permission": "never"}})
        child = manager_for(repo, parent=parent)
        args = (
            BashArgs(command="git push")
            if tool_name == "bash"
            else BashStartArgs(command="git push")
        )
        context = child.get(tool_name).resolve_permission(args)
        assert context is not None
        assert context.permission is ToolPermission.NEVER
        assert context.dangerous_command_proposal is None


class TestAuthorityLifetime:
    """Authority changes invalidate cached decisions; retirement fails closed."""

    def test_authority_change_is_visible_without_reconstruction(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        scope = scope_of(repo)
        provider = AuthorityProvider()
        manager = manager_for(repo, provider=provider)
        tool = manager.get("bash")
        assert_dangerous_denial(
            tool.resolve_permission(BashArgs(command="git push")), ("git.push",)
        )
        # A session grant arrives: the same retained tool now resolves allowed.
        provider.authorities = (authority_for(scope, "git.push"),)
        context = tool.resolve_permission(BashArgs(command="git push"))
        assert context is not None and context.permission is ToolPermission.ALWAYS
        assert provider.calls >= 2

    def test_invalidate_dangerous_command_authority_bumps_the_token(
        self, tmp_path: Path
    ):
        manager = manager_for(tmp_path)
        before = manager._effective_authority_token()
        manager._invalidate_dangerous_command_authority()
        after = manager._effective_authority_token()
        assert before != after

    def test_retained_tools_fail_closed_after_retirement(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        scope = scope_of(repo)
        provider = AuthorityProvider((authority_for(scope, "git.push"),))
        manager = manager_for(repo, provider=provider)
        retained = manager.get("bash")
        context = retained.resolve_permission(BashArgs(command="git push"))
        assert context is not None and context.permission is ToolPermission.ALWAYS
        manager._retire_authority()
        context = retained.resolve_permission(BashArgs(command="git push"))
        assert context is not None
        assert context.permission is ToolPermission.NEVER
        assert "retired" in (context.reason or "")
        assert context.dangerous_command_proposal is None
        # Even routine commands fail closed on a retained tool.
        context = retained.resolve_permission(BashArgs(command="echo hi"))
        assert context is not None and context.permission is ToolPermission.NEVER


class TestBashStartParity:
    """Identical resolution for bash and bash_start across config shapes."""

    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        return make_repo(tmp_path / "project")

    def _resolve_all(self, manager: ToolManager, command: str):
        foreground = manager.get("bash").resolve_permission(BashArgs(command=command))
        launch = manager.get("bash_start").resolve_permission(
            BashStartArgs(command=command)
        )
        shell = manager.get("bash")
        shared = ShellPermissionResolver(
            shell.config,
            shell.cwd,
            shell.workspace,
            shell.path_authority,
            shell.dangerous_command_authorities_getter,
        )
        direct = shared.resolve_permission(BashStartArgs(command=command))
        return foreground, launch, direct

    def _assert_parity(self, results, command: str):
        foreground, launch, direct = results
        for context in (foreground, launch, direct):
            assert context is not None, command
        assert foreground.permission == launch.permission == direct.permission  # type: ignore[union-attr]
        assert foreground.denial_kind == launch.denial_kind == direct.denial_kind  # type: ignore[union-attr]
        assert (  # type: ignore[union-attr]
            foreground.dangerous_command_proposal
            == launch.dangerous_command_proposal
            == direct.dangerous_command_proposal
        )

    @pytest.mark.parametrize(
        "command, expected",
        [
            ("git push", ToolPermission.NEVER),
            ("git push --force origin main", ToolPermission.NEVER),
            ("git checkout main", ToolPermission.NEVER),
            ("echo ok", ToolPermission.ALWAYS),
            ("git status", ToolPermission.ALWAYS),
        ],
    )
    def test_parity_under_defaults(
        self, repo: Path, command: str, expected: ToolPermission
    ):
        manager = manager_for(repo)
        self._assert_parity(self._resolve_all(manager, command), command)
        foreground = manager.get("bash").resolve_permission(BashArgs(command=command))
        assert foreground is not None and foreground.permission is expected

    @pytest.mark.parametrize("denylist", [[], ["curl"]])
    @pytest.mark.parametrize("command", ["git push", "echo ok"])
    def test_parity_under_replacement_denylists(
        self, repo: Path, denylist: list[str], command: str
    ):
        manager = manager_for(repo, tools={"bash": {"denylist": denylist}})
        self._assert_parity(self._resolve_all(manager, command), command)

    def test_parity_under_inherited_policies(self, repo: Path):
        # Both bound adapters carry the parent ceiling, so they agree; a
        # child-only resolver sees just the local policy and is not comparable.
        parent = manager_for(repo, tools={"bash": {"denylist": ["echo"]}})
        child = manager_for(repo, parent=parent, tools={"bash": {"denylist": []}})
        for command in ("echo ok", "git push"):
            foreground = child.get("bash").resolve_permission(BashArgs(command=command))
            launch = child.get("bash_start").resolve_permission(
                BashStartArgs(command=command)
            )
            assert foreground is not None and launch is not None
            assert foreground.permission is ToolPermission.NEVER
            assert foreground.permission == launch.permission
            assert foreground.denial_kind == launch.denial_kind
            assert (
                foreground.dangerous_command_proposal
                == launch.dangerous_command_proposal
            )

    def test_parity_under_saved_authority(self, repo: Path):
        source = saved_grant_source(repo.parent, repo, ("git.push",))
        manager = manager_for(repo, restrictions=(source,))
        for command in ("git push", "git push --force", "echo ok"):
            self._assert_parity(self._resolve_all(manager, command), command)
        foreground = manager.get("bash").resolve_permission(
            BashArgs(command="git push")
        )
        assert foreground is not None and foreground.permission is ToolPermission.ALWAYS

    def test_parity_under_session_authority(self, repo: Path):
        scope = scope_of(repo)
        provider = AuthorityProvider((authority_for(scope, "git.push"),))
        manager = manager_for(repo, provider=provider)
        for command in ("git push", "git push --force", "git stash drop"):
            self._assert_parity(self._resolve_all(manager, command), command)
        foreground = manager.get("bash").resolve_permission(
            BashArgs(command="git push")
        )
        assert foreground is not None and foreground.permission is ToolPermission.ALWAYS

    def test_parity_under_once_snapshots(self, repo: Path):
        # A once grant lives in the snapshot for exactly one resolution; the
        # resolver consumes snapshots and never repeats or retains them.
        scope = scope_of(repo)
        once: list[DangerousCommandAuthority] = [authority_for(scope, "git.push")]

        class OnceProvider:
            def dangerous_command_authorities(self):
                return tuple(once)

        manager = manager_for(repo, provider=OnceProvider())
        self._assert_parity(self._resolve_all(manager, "git push"), "git push")
        foreground = manager.get("bash").resolve_permission(
            BashArgs(command="git push")
        )
        assert foreground is not None and foreground.permission is ToolPermission.ALWAYS
        # The runtime (P4) consumes the once grant after the invocation.
        once.clear()
        self._assert_parity(self._resolve_all(manager, "git push"), "git push")
        foreground = manager.get("bash").resolve_permission(
            BashArgs(command="git push")
        )
        assert foreground is not None
        assert foreground.permission is ToolPermission.NEVER
        assert foreground.denial_kind == "dangerous_command"


class TestResolverReadOnly:
    """The resolver never mutates grants, persists, or executes."""

    def test_resolution_leaves_the_authority_snapshot_untouched(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        scope = scope_of(repo)
        authorities = (authority_for(scope, "git.push"),)
        provider = AuthorityProvider(authorities)
        manager = manager_for(repo, provider=provider)
        tool = manager.get("bash")
        tool.resolve_permission(BashArgs(command="git push"))
        tool.resolve_permission(BashArgs(command="git push --force"))
        assert provider.authorities == authorities
        assert provider.authorities[0] is authorities[0]
        assert manager._dangerous_command_authorities() == authorities
        assert not (tmp_path / "user.toml").exists()

    def test_repeated_resolution_is_stable(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        scope = scope_of(repo)
        tool = make_bash(repo, (authority_for(scope, "git.push"),))
        first = tool.resolve_permission(BashArgs(command="git push --force"))
        second = tool.resolve_permission(BashArgs(command="git push --force"))
        assert first == second


class TestScopeResolution:
    """Repository/worktree identity resolution (pure filesystem inspection)."""

    def test_plain_repository(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        assert scope_of(repo) == RepositoryScope(git_dir=repo / ".git", work_tree=repo)

    def test_subdirectory_walks_up(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        sub = repo / "src" / "deep"
        sub.mkdir(parents=True)
        assert scope_of(sub) == scope_of(repo)

    def test_dash_c_chain_applies_in_order(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        (repo / "a" / "b").mkdir(parents=True)
        relocation = GitRepositoryRelocation(cwd_paths=("a", "b"))
        scope = resolve_repository_scope(repo, relocation)
        assert scope is not None
        assert scope.git_dir == repo / ".git"
        assert scope.work_tree == repo

    def test_git_dir_override(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        bare = tmp_path / "bare.git"
        bare.mkdir()
        scope = resolve_repository_scope(
            repo, GitRepositoryRelocation(git_dir=str(bare))
        )
        assert scope == RepositoryScope(git_dir=bare, work_tree=None)

    def test_work_tree_override(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        work_tree = tmp_path / "wt"
        work_tree.mkdir()
        scope = resolve_repository_scope(
            repo, GitRepositoryRelocation(work_tree=str(work_tree))
        )
        assert scope == RepositoryScope(git_dir=repo / ".git", work_tree=work_tree)

    def test_bare_repository_has_no_work_tree(self, tmp_path: Path):
        bare = tmp_path / "bare.git"
        bare.mkdir()
        for name in ("HEAD", "config"):
            (bare / name).write_text("")
        (bare / "objects").mkdir()
        (bare / "refs").mkdir()
        assert scope_of(bare) == RepositoryScope(git_dir=bare, work_tree=None)

    def test_git_file_pointer_is_the_worktree_identity(self, tmp_path: Path):
        repo = make_repo(tmp_path / "project")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        git_dir = repo / ".git" / "worktrees" / "wt"
        git_dir.mkdir(parents=True)
        (worktree / ".git").write_text(f"gitdir: {git_dir}")
        assert scope_of(worktree) == RepositoryScope(
            git_dir=git_dir, work_tree=worktree
        )

    def test_no_repository_yields_the_nominal_scope(self, tmp_path: Path):
        plain = tmp_path / "plain"
        plain.mkdir()
        assert scope_of(plain) == RepositoryScope(
            git_dir=plain / ".git", work_tree=plain
        )

    def test_malformed_git_file_is_unlocatable(self, tmp_path: Path):
        plain = tmp_path / "plain"
        plain.mkdir()
        (plain / ".git").write_text("garbage")
        assert resolve_repository_scope(plain, GitRepositoryRelocation()) is None

    def test_symlinked_git_dir_is_canonicalized(self, tmp_path: Path):
        metadata = tmp_path / "metadata"
        metadata.mkdir()
        repo = tmp_path / "project"
        repo.mkdir()
        (repo / ".git").symlink_to(metadata, target_is_directory=True)
        assert scope_of(repo) == RepositoryScope(git_dir=metadata, work_tree=repo)


class TestShippedDenylistMigration:
    """The recognized git operations left the shipped denylist defaults."""

    def test_shipped_denylist_has_no_git_entries(self):
        denylist = _get_default_denylist()
        assert not any(entry.startswith("git ") for entry in denylist)
        # Non-git entries are kept.
        for kept in ("curl", "wget", "nc", "vim", "nano", "tmux", "gdb"):
            assert kept in denylist

    def test_default_config_denylist_has_no_git_entries(self):
        assert not any(entry.startswith("git ") for entry in BashToolConfig().denylist)

    @pytest.mark.parametrize("denylist", [[], ["curl"]])
    def test_cleared_or_replaced_denylist_does_not_authorize_git(
        self, tmp_path: Path, denylist: list[str]
    ):
        # Breaking change, documented: denylist=[] no longer implicitly
        # authorizes the recognized git operations.
        tool = make_bash(tmp_path, denylist=denylist)
        for command in ("git push", "git checkout main", "git stash drop"):
            context = tool.resolve_permission(BashArgs(command=command))
            assert context is not None
            assert context.permission is ToolPermission.NEVER
            assert context.denial_kind == "dangerous_command"
            assert context.dangerous_command_proposal is not None

    def test_explicit_git_denylist_entry_still_vetoes(self, tmp_path: Path):
        tool = make_bash(tmp_path, denylist=["git push"])
        context = tool.resolve_permission(BashArgs(command="git push"))
        assert_other_denial(context)
        assert "matches denylist pattern 'git push'" in (context.reason or "")  # type: ignore[union-attr]
        # The veto also applies to relocated spellings.
        context = tool.resolve_permission(BashArgs(command="git -C . push"))
        assert_other_denial(context)
        assert "matches denylist pattern 'git push'" in (context.reason or "")  # type: ignore[union-attr]

    def test_hard_guards_survive_the_migration(self, tmp_path: Path):
        tool = make_bash(tmp_path, denylist=[])
        for command in (
            "git reset --hard HEAD",
            "git clean -fd",
            "rm -rf /tmp/something",
        ):
            context = tool.resolve_permission(BashArgs(command=command))
            assert_other_denial(context)


def test_bash_start_resolve_permission_uses_the_live_authority_getter(tmp_path: Path):
    # BashStart shares the resolver and the live authority getter with Bash.
    repo = make_repo(tmp_path / "project")
    scope = scope_of(repo)
    provider = AuthorityProvider((authority_for(scope, "git.push"),))
    manager = manager_for(repo, provider=provider)
    start = manager.get("bash_start")
    assert isinstance(start, BashStart)
    context = start.resolve_permission(BashStartArgs(command="git push"))
    assert context is not None and context.permission is ToolPermission.ALWAYS
    assert_dangerous_denial(
        start.resolve_permission(BashStartArgs(command="git push --force")),
        ("git.push.force",),
    )
