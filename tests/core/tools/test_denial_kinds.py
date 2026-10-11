from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.dangerous_commands import DangerousCommandProposal, RepositoryScope
from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.models import PermissionContext
from chartreux.core.tools.utils import (
    PathAccess,
    PathAuthority,
    resolve_file_tool_permission,
)
from chartreux.core.workspace import Workspace


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("tool_policy", "tool_policy"),
        ("path_rule", "path_rule"),
        ("sensitive", "sensitive"),
        ("plan_scope", "plan_scope"),
        ("out_of_root", "out_of_root"),
        ("parent_ceiling", "parent_ceiling"),
    ],
)
def test_denial_kind_classification(case: str, expected: str, tmp_path: Path) -> None:
    target = tmp_path.parent / f"{tmp_path.name}-outside" / ".env"
    target.parent.mkdir()
    workspace = Workspace.for_session(tmp_path)
    sensitive = []
    denylist = []
    permission = ToolPermission.ALWAYS
    plan_scope = None
    if case == "tool_policy":
        permission = ToolPermission.NEVER
    elif case == "path_rule":
        denylist = [str(target)]
        sensitive = ["**/.env"]
    elif case == "sensitive":
        sensitive = ["**/.env"]
    elif case == "plan_scope":
        plan_scope = tmp_path / "plan.md"
    elif case == "parent_ceiling":
        workspace = Workspace.for_session(tmp_path, authorized_roots=[target.parent])
        workspace = Workspace(
            workspace.cwd, workspace.authorized_roots, Workspace.for_session(tmp_path)
        )
    result = resolve_file_tool_permission(
        str(target),
        tool_name="write_file",
        allowlist=[],
        denylist=denylist,
        config_permission=permission,
        sensitive_patterns=sensitive,
        workspace=workspace,
        plan_file_write_scope=plan_scope,
    )
    assert result is not None
    assert result.permission is ToolPermission.NEVER
    assert result.denial_kind == expected


def test_parent_walk_preserves_sensitive_and_tool_policy(tmp_path: Path) -> None:
    target = tmp_path.parent / f"{tmp_path.name}-outside" / ".env"
    target.parent.mkdir()
    workspace = Workspace.for_session(tmp_path)
    for permission, patterns, expected in (
        (ToolPermission.ALWAYS, ("**/.env",), "sensitive"),
        (ToolPermission.NEVER, (), "tool_policy"),
    ):
        authority = PathAuthority(
            tool_name="read_file",
            permission=permission,
            allowlist=(),
            denylist=(),
            sensitive=patterns,
            workspace=workspace,
        )
        child = PathAuthority(
            tool_name="read_file",
            permission=ToolPermission.ALWAYS,
            allowlist=(),
            denylist=(),
            sensitive=(),
            workspace=workspace,
            parents=(authority,),
        )
        decision = child.resolve(str(target), PathAccess.READ)
        assert decision is not None
        assert decision.denial_kind == expected


def test_sensitive_and_path_rule_precede_out_of_root(tmp_path: Path) -> None:
    target = tmp_path / "outside" / ".env"
    workspace = Workspace.for_session(tmp_path)
    for denylist, patterns, expected in (
        ([], ["**/.env"], "sensitive"),
        ([str(target)], [], "path_rule"),
    ):
        decision = resolve_file_tool_permission(
            str(target),
            tool_name="read_file",
            allowlist=[],
            denylist=denylist,
            config_permission=ToolPermission.ALWAYS,
            sensitive_patterns=patterns,
            workspace=workspace,
        )
        assert decision is not None
        assert decision.denial_kind == expected


def _proposal() -> DangerousCommandProposal:
    return DangerousCommandProposal(
        command=("git", "push"),
        project=Path("/project"),
        scope=RepositoryScope(
            git_dir=Path("/project/.git"), work_tree=Path("/project")
        ),
        operations=("git.push",),
        consequence="Publish local commits to the remote.",
    )


def test_dangerous_command_kind_requires_a_grantable_proposal() -> None:
    # The taxonomy admits exactly one grantable kind, and it always carries
    # the typed whole-call proposal.
    context = PermissionContext(
        permission=ToolPermission.NEVER,
        reason="denied",
        denial_kind="dangerous_command",
        dangerous_command_proposal=_proposal(),
    )
    assert context.dangerous_command_proposal is not None
    with pytest.raises(ValueError, match="requires a grantable proposal"):
        PermissionContext(
            permission=ToolPermission.NEVER,
            reason="denied",
            denial_kind="dangerous_command",
        )


@pytest.mark.parametrize(
    "denial_kind",
    [
        "out_of_root",
        "sensitive",
        "path_rule",
        "plan_scope",
        "parent_ceiling",
        "tool_policy",
    ],
)
def test_other_denial_kinds_never_carry_a_proposal(denial_kind: str) -> None:
    with pytest.raises(ValueError, match="dangerous_command_proposal requires"):
        PermissionContext(
            permission=ToolPermission.NEVER,
            reason="denied",
            denial_kind=denial_kind,  # type: ignore[arg-type]
            dangerous_command_proposal=_proposal(),
        )


def test_shell_dangerous_git_denial_is_the_grantable_kind(tmp_path: Path) -> None:
    # Without any dangerous-command authority, the shell resolver classifies
    # the missing authority as the grantable dangerous_command kind.
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    context = tool.resolve_permission(BashArgs(command="git push"))
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind == "dangerous_command"
    assert context.dangerous_command_proposal is not None
    assert context.dangerous_command_proposal.operations == ("git.push",)


@pytest.mark.parametrize(
    ("command", "expected_kind"),
    [
        # Absolute denylist veto: never the grantable kind, never a proposal.
        ("vim file.txt", None),
        # Hard guards: never the grantable kind, never a proposal.
        ("git reset --hard HEAD", None),
        ("git push --bogus origin main", None),
    ],
)
def test_shell_veto_and_hard_denials_never_use_dangerous_command(
    tmp_path: Path, command: str, expected_kind: str | None
) -> None:
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=tmp_path
    )
    context = tool.resolve_permission(BashArgs(command=command))
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert context.denial_kind == expected_kind
    assert context.dangerous_command_proposal is None
