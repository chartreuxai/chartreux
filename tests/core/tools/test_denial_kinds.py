from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import ToolPermission
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
