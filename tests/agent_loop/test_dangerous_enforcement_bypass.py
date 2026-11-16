"""v0.8.0 A-P1 chunk 1: the DANGEROUS enforcement-layer bypasses.

Every bypass keys on the session authority state (SafetyLevel == DANGEROUS)
through the live context seam in ``core/tools/permissions.py``. At this
chunk's landing no path can produce DANGEROUS (the ceremony lands in chunk
2), so these tests bind the seam explicitly and assert the bypassed and the
guarded behavior side by side.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chartreux.core.agent_loop._loop_types import ToolDecision
from chartreux.core.agent_loop._safety_authority import (
    SafetyLevel,
    SessionAuthorityState,
)
from chartreux.core.agent_loop._tool_decisions import ToolDecisionCoordinator
from chartreux.core.agent_loop_hooks import AgentLoopHooksMixin
from chartreux.core.agents.identities import ToolPosture, get_builtin_identity
from chartreux.core.dangerous_commands import DangerousCommandProposal, RepositoryScope
from chartreux.core.hooks.models import HookToolDenial
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins._shell_permission_resolver import (
    ShellPermissionResolver,
)
from chartreux.core.tools.builtins.bash import BashArgs, BashToolConfig
from chartreux.core.tools.manager import (
    child_withheld_tools,
    identity_allows_name,
    posture_allows_name,
)
from chartreux.core.tools.permissions import bind_dangerous_mode, dangerous_mode_active
from chartreux.core.tools.utils import (
    PathAccess,
    PathAuthority,
    resolve_file_tool_permission,
)
from chartreux.core.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path, (tmp_path,))


@pytest.fixture
def path_authority(workspace: Workspace) -> PathAuthority:
    return PathAuthority(
        tool_name="write_file",
        permission=ToolPermission.ALWAYS,
        allowlist=(),
        denylist=(),
        sensitive=(),
        workspace=workspace,
    )


def _dangerous_state() -> SessionAuthorityState:
    return SessionAuthorityState(session_id="sess-t", level=SafetyLevel.DANGEROUS)


class TestSeam:
    def test_unbound_context_is_guarded(self) -> None:
        assert dangerous_mode_active() is False

    def test_broken_getter_fails_closed(self) -> None:
        def boom() -> bool:
            raise RuntimeError("broken")

        with bind_dangerous_mode(boom):
            assert dangerous_mode_active() is False

    def test_bound_getter_is_live(self) -> None:
        state = SessionAuthorityState(session_id="sess-t", level=SafetyLevel.SAFE)
        with bind_dangerous_mode(state.is_dangerous):
            assert dangerous_mode_active() is False
            state.transition(SafetyLevel.DANGEROUS, actor="dangerous")
            assert dangerous_mode_active() is True


class TestShellPolicyBypass:
    def test_denylisted_command_denied_normally_admitted_dangerous(
        self, tmp_path: Path, workspace: Workspace, path_authority: PathAuthority
    ) -> None:
        config = BashToolConfig(denylist=["shutdown"])
        resolver = ShellPermissionResolver(config, tmp_path, workspace, path_authority)
        args = BashArgs(command="shutdown now")
        denied = resolver.resolve_permission(args)
        assert denied is not None and denied.permission is ToolPermission.NEVER
        with bind_dangerous_mode(_dangerous_state().is_dangerous):
            assert resolver.resolve_permission(args) is None


class TestFileAuthorityBypass:
    def test_out_of_root_denied_normally_always_dangerous(
        self, workspace: Workspace, path_authority: PathAuthority
    ) -> None:
        outside = "/etc/chartreux-bypass-probe"
        denied = path_authority.resolve(outside, PathAccess.WRITE)
        assert denied is not None and denied.permission is ToolPermission.NEVER
        with bind_dangerous_mode(_dangerous_state().is_dangerous):
            admitted = path_authority.resolve(outside, PathAccess.WRITE)
            assert admitted is not None
            assert admitted.permission is ToolPermission.ALWAYS
            direct = resolve_file_tool_permission(
                outside,
                tool_name="write_file",
                allowlist=[],
                denylist=[],
                config_permission=ToolPermission.NEVER,
                sensitive_patterns=[],
                workspace=workspace,
                access=PathAccess.WRITE,
            )
            assert direct is not None and direct.permission is ToolPermission.ALWAYS


class TestToolAvailabilityBypass:
    def test_posture_and_identity_axes_yield(self) -> None:
        reviewer = get_builtin_identity("reviewer")
        assert reviewer is not None
        # "write_file": not in the reviewer's execution carve-out ({bash}).
        assert posture_allows_name(ToolPosture.NON_MUTATING, "write_file") is False
        assert identity_allows_name(reviewer, "write_file") is False
        withheld = child_withheld_tools(reviewer)
        assert "write_file" in withheld
        with bind_dangerous_mode(_dangerous_state().is_dangerous):
            assert posture_allows_name(ToolPosture.NON_MUTATING, "write_file") is True
            assert identity_allows_name(reviewer, "write_file") is True
            # Only the loop-level structural set stays withheld.
            from chartreux.core.tools.manager import CHILD_CONTEXT_UNAVAILABLE_TOOLS

            assert child_withheld_tools(reviewer) == CHILD_CONTEXT_UNAVAILABLE_TOOLS


class TestGrantDecisionBypass:
    def _coordinator(self, level: SafetyLevel) -> ToolDecisionCoordinator:
        state = SessionAuthorityState(session_id="sess-t", level=level)
        coordinator = object.__new__(ToolDecisionCoordinator)
        coordinator._authority_state_getter = lambda: state  # type: ignore[attr-defined]
        return coordinator

    def test_bypass_executes_without_grant(self) -> None:
        coordinator = self._coordinator(SafetyLevel.DANGEROUS)
        decision = coordinator._dangerous_mode_bypass("bash", None)
        assert isinstance(decision, ToolDecision)
        assert decision.verdict.name == "EXECUTE"
        assert decision.approval_type is ToolPermission.ALWAYS
        assert decision.admission is None

    def test_guarded_level_does_not_bypass(self) -> None:
        for level in (SafetyLevel.SAFE, SafetyLevel.SMART):
            assert self._coordinator(level)._dangerous_mode_bypass("bash", None) is None

    def test_bypass_records_decided_call_with_dangerous_actor(self) -> None:
        coordinator = self._coordinator(SafetyLevel.DANGEROUS)
        proposal = DangerousCommandProposal(
            command=("git", "push"),
            project=Path("/tmp"),
            scope=RepositoryScope(git_dir=Path("/tmp/.git"), work_tree=Path("/tmp")),
            operations=("git.push",),
            consequence="publish",
        )
        decision = coordinator._dangerous_mode_bypass("bash", proposal)
        assert decision is not None and decision.verdict.name == "EXECUTE"
        state = coordinator._authority_state_getter()  # type: ignore[attr-defined]
        assert state is not None
        records = state.decided_calls.records()
        assert len(records) == 1
        assert records[0].decision == "dangerous_mode"
        assert records[0].actor == "dangerous"
        assert records[0].authorizer == "dangerous"


class _HookStub:
    def __init__(self, events: list[HookToolDenial]) -> None:
        self._events = events

    async def _run_pre_tool_hooks(self, tool_call, tool_input):  # type: ignore[no-untyped-def]
        for event in self._events:
            yield event


class TestHookAdvisoryBypass:
    @pytest.mark.asyncio
    async def test_pre_tool_veto_ignored_under_dangerous(self) -> None:
        # The guarded (veto blocks) direction is covered by the existing hook
        # pipeline tests; here only the DANGEROUS advisory direction.
        denial = HookToolDenial(hook_name="guard", content="no")
        tool_call = SimpleNamespace(tool_name="bash", call_id="c1")
        with bind_dangerous_mode(_dangerous_state().is_dangerous):
            events, resolution = await AgentLoopHooksMixin._run_pre_tool_pipeline(
                _HookStub([denial]),  # type: ignore[arg-type]
                tool_call,  # type: ignore[arg-type]
                {"command": "echo hi"},
            )
        assert resolution.denial_event is None
        assert events == []  # the veto is logged and ignored, never blocking
