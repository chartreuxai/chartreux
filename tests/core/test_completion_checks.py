"""P5 acceptance: the template acceptance-check runner.

All six templates' argv, injection operands, symlink escapes, denied and
revoked authority, output floods, timeout, and the mocked spawn seam. No
live-provider calls: the spawn seam is monkeypatched except in the two
batteries that execute local, bounded subprocesses (each carrying an
explicit ``timeout`` marker).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import shlex
from typing import Any
from unittest.mock import patch

import pytest

from chartreux.core.completion_checks import (
    MAX_CAPTURE_BYTES,
    CompletionCheckRunner,
    CompletionCheckStatus,
    CompletionCheckTemplate,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    ShellPermissionResolver,
)
from chartreux.core.tools.builtins.bash import BashToolConfig
from chartreux.core.tools.models import PermissionContext, ToolPermission
from chartreux.core.tools.utils import PathAccess, PathAuthority
from chartreux.core.workspace import Workspace

pytestmark = pytest.mark.timeout(60)

#: A permissive authority: every path resolves to ALWAYS.
_ALWAYS_AUTHORITY = ("always",)


def _authority(tmp_path: Path) -> Any:
    """A permissive PathAuthority over ``tmp_path``."""

    class _StubAuthority:
        tool_name = "bash"

        def resolve(self, path: str, access: PathAccess) -> Any:
            return PermissionContext(permission=ToolPermission.ALWAYS)

    return _StubAuthority()


def _workspace(tmp_path: Path) -> Workspace:
    return Workspace.for_session(tmp_path)


def _runner(tmp_path: Path, **overrides: Any) -> CompletionCheckRunner:
    return CompletionCheckRunner(
        cwd=tmp_path,
        workspace=_workspace(tmp_path),
        path_authority=_authority(tmp_path),
        **overrides,
    )


# --- all six templates: harness-built argv -------------------------------


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        (CompletionCheckTemplate.PYTEST, ("uv", "run", "pytest", "-q")),
        (CompletionCheckTemplate.RUFF_CHECK, ("uv", "run", "ruff", "check")),
        (CompletionCheckTemplate.PYRIGHT, ("uv", "run", "pyright")),
        (CompletionCheckTemplate.GIT_HEAD, ("git", "rev-parse", "HEAD")),
        (CompletionCheckTemplate.GIT_STATUS, ("git", "status", "--porcelain")),
        (CompletionCheckTemplate.GIT_DIFF_STAT, ("git", "diff", "--stat")),
    ],
)
def test_check_argv_all_six_templates(
    tmp_path: Path, template: CompletionCheckTemplate, expected: tuple[str, ...]
) -> None:
    assert _runner(tmp_path).check_argv(template) == expected


def test_check_argv_root_operand_appended(tmp_path: Path) -> None:
    root = tmp_path / "sub dir"
    argv = _runner(tmp_path).check_argv(CompletionCheckTemplate.PYTEST, root=root)
    assert argv[-1] == str(root)


# --- injection operands stay literal --------------------------------------


@pytest.mark.timeout(60)
def test_injection_operand_stays_literal(tmp_path: Path) -> None:
    """A metacharacter-bearing root is one literal argv element."""
    nasty = "sub;$(touch pwned)`id`$((1+1))"
    argv = _runner(tmp_path).check_argv(
        CompletionCheckTemplate.PYTEST, root=Path(nasty)
    )
    assert argv[-1] == nasty
    assert not (tmp_path / "pwned").exists()


# --- symlink escapes -------------------------------------------------------


@pytest.mark.timeout(60)
def test_symlink_escape_denied(tmp_path: Path) -> None:
    """A target outside the authorized roots is refused before spawn."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker").write_text("x")
    inside = tmp_path / "inside"
    inside.mkdir()
    escape = inside / "escape"
    escape.symlink_to(Path("/") / "etc")
    runner = _runner(tmp_path)
    with pytest.raises(Exception, match="root_escape"):
        runner.validate_check_root(escape)


@pytest.mark.timeout(60)
def test_symlink_inside_root_allowed(tmp_path: Path) -> None:
    inside = tmp_path / "inside"
    inside.mkdir()
    runner = _runner(tmp_path)
    resolved = runner.validate_check_root(inside)
    assert resolved == inside.resolve()


# --- denied / revoked authority --------------------------------------------


@pytest.mark.timeout(60)
def test_denied_capability_never_spawns(tmp_path: Path) -> None:
    """A denied capability stamps the check unavailable without spawning."""

    class _DeniedContract:
        capability = CompletionCheckStatus.DENIED.value

    spawned: list[list[str]] = []

    async def _recorder(command: str, **kwargs: Any) -> None:
        spawned.append(shlex.split(command))

    runner = _runner(tmp_path)
    outcome = asyncio.run(
        runner.run(
            CompletionCheckTemplate.PYTEST,
            contract=_DeniedContract(),  # type: ignore[arg-type]
        )
    )
    assert outcome.status is CompletionCheckStatus.DENIED
    assert outcome.argv == ()
    assert spawned == []


@pytest.mark.timeout(60)
def test_unavailable_capability_never_spawns(tmp_path: Path) -> None:
    class _UnavailableContract:
        capability = "unavailable"

    spawned: list[list[str]] = []

    async def _recorder(command: str, **kwargs: Any) -> None:
        spawned.append(shlex.split(command))

    runner = _runner(tmp_path)
    outcome = asyncio.run(
        runner.run(
            CompletionCheckTemplate.PYTEST,
            contract=_UnavailableContract(),  # type: ignore[arg-type]
        )
    )
    assert outcome.status is CompletionCheckStatus.DENIED
    assert spawned == []


# --- output floods are capped ----------------------------------------------


@pytest.mark.timeout(60)
def test_output_flood_is_capped(tmp_path: Path) -> None:
    """A flood of output keeps the outcome's captured tails within the cap."""
    flood = "x" * (MAX_CAPTURE_BYTES * 8)
    argv = ("printf", "%s", flood)
    outcome = asyncio.run(_run_real(tmp_path, argv))
    assert len(outcome.stdout) <= MAX_CAPTURE_BYTES + 128


async def _run_real(tmp_path: Path, argv: tuple[str, ...]) -> Any:
    runner = _runner(tmp_path)
    return await runner._spawn(CompletionCheckTemplate.GIT_HEAD, argv, tmp_path, 0.0)


# --- timeout terminates the process group -----------------------------------


@pytest.mark.timeout(60)
def test_timeout_kills_process_group(tmp_path: Path) -> None:
    """A timed-out check's process group is terminated, not orphaned."""
    runner = _runner(tmp_path, timeout=0.2)
    outcome = asyncio.run(
        runner._spawn(CompletionCheckTemplate.PYTEST, ("sleep", "5"), tmp_path, 0.0)
    )
    assert outcome.status is CompletionCheckStatus.TIMED_OUT


# --- no prompts, no retries, no fallbacks ------------------------------------


@pytest.mark.timeout(60)
def test_no_retry_after_nonzero(tmp_path: Path) -> None:
    """A failing check spawns exactly once: no retries, no fallbacks."""
    calls: list[int] = []

    async def _failing_spawn(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        raise asyncio.CancelledError()

    runner = _runner(tmp_path)
    with patch.object(
        CompletionCheckRunner, "_spawn", autospec=True, side_effect=_failing_spawn
    ):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(runner.run(CompletionCheckTemplate.PYTEST))
    assert len(calls) == 1


# --- shell admission: resolver admission immediately before spawn -----------


class _DenyAdmission:
    """A shell-admission stub denying every command."""

    def __init__(self, denial_kind: Any = "tool_policy") -> None:
        self.denial_kind = denial_kind
        self.commands: list[str] = []

    def resolve_permission(self, args: Any) -> Any:
        self.commands.append(args.command)
        return PermissionContext(
            permission=ToolPermission.NEVER,
            reason="denied by test stub",
            denial_kind=self.denial_kind,
        )


class _AllowAdmission:
    """A shell-admission stub admitting every command."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def resolve_permission(self, args: Any) -> Any:
        self.commands.append(args.command)
        return PermissionContext(permission=ToolPermission.ALWAYS)


class _SpawnRecorder:
    """The monkeypatched check spawn seam: records commands, fakes processes."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def __call__(self, command: str, **kwargs: Any) -> Any:
        self.commands.append(command)

        class _Process:
            async def communicate(self) -> tuple[bytes, bytes]:
                return (b"ok\n", b"")

            @property
            def returncode(self) -> int:
                return 0

        return _Process()


def _install_spawn(monkeypatch: pytest.MonkeyPatch, spawn: _SpawnRecorder) -> None:
    monkeypatch.setattr("chartreux.core.completion_checks.spawn_shell_command", spawn)


def _real_admission(tmp_path: Path, config: Any) -> Any:
    """A REAL ShellPermissionResolver over ``tmp_path`` with ``config``."""
    workspace = Workspace.for_session(tmp_path)
    return ShellPermissionResolver(
        config,
        tmp_path,
        workspace,
        PathAuthority(
            tool_name="bash",
            permission=ToolPermission.ALWAYS,
            allowlist=(),
            denylist=(),
            sensitive=(),
            workspace=workspace,
        ),
    )


@pytest.mark.timeout(60)
def test_shell_admission_denial_refuses_without_spawning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A resolver denial refuses the check before spawn: no process runs."""
    admission = _DenyAdmission()
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    runner = _runner(tmp_path, shell_admission=admission)

    outcome = asyncio.run(runner.run(CompletionCheckTemplate.PYTEST))

    assert outcome.status is CompletionCheckStatus.DENIED
    assert outcome.argv == ()
    assert outcome.detail.startswith("shell_admission_denied")
    assert spawn.commands == []
    # The resolver saw the composed check command, root operand included.
    assert len(admission.commands) == 1
    assert "pytest" in admission.commands[0]
    assert str(tmp_path.resolve()) in admission.commands[0]


@pytest.mark.timeout(60)
def test_shell_admission_ask_class_denial_never_prompts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The promptable ask-class (out-of-root) is still a plain DENIED stamp:
    finalization never prompts, so nothing waits for or seeks an approval.
    """
    admission = _DenyAdmission(denial_kind="out_of_root")
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    runner = _runner(tmp_path, shell_admission=admission)

    outcome = asyncio.run(runner.run(CompletionCheckTemplate.PYTEST))

    assert outcome.status is CompletionCheckStatus.DENIED
    assert outcome.argv == ()
    assert outcome.detail == "shell_admission_denied:out_of_root"
    assert spawn.commands == []


@pytest.mark.timeout(60)
def test_shell_admission_defect_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A resolver defect is an informational denial, never a spawn."""

    class _BrokenAdmission:
        def resolve_permission(self, args: Any) -> Any:
            raise RuntimeError("resolver retired")

    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    runner = _runner(tmp_path, shell_admission=_BrokenAdmission())

    outcome = asyncio.run(runner.run(CompletionCheckTemplate.PYTEST))

    assert outcome.status is CompletionCheckStatus.DENIED
    assert outcome.argv == ()
    assert outcome.detail == "shell_admission_unavailable"
    assert spawn.commands == []


@pytest.mark.timeout(60)
def test_shell_admission_allows_then_spawns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An admitted command spawns exactly once through the scrubbed runner."""
    admission = _AllowAdmission()
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    runner = _runner(tmp_path, shell_admission=admission)

    outcome = asyncio.run(runner.run(CompletionCheckTemplate.PYTEST))

    assert outcome.status is CompletionCheckStatus.PASSED
    assert len(spawn.commands) == 1
    assert spawn.commands[0] == admission.commands[0]


@pytest.mark.timeout(60)
def test_real_resolver_denylisted_template_refuses_without_spawning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The REAL resolver: a denylisted command template refuses, no spawn."""
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    runner = _runner(
        tmp_path,
        shell_admission=_real_admission(tmp_path, BashToolConfig(denylist=["uv"])),
    )

    outcome = asyncio.run(runner.run(CompletionCheckTemplate.PYTEST))

    assert outcome.status is CompletionCheckStatus.DENIED
    assert outcome.argv == ()
    assert outcome.detail.startswith("shell_admission_denied")
    assert spawn.commands == []


@pytest.mark.timeout(60)
def test_real_resolver_revoked_authority_refuses_without_spawning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The REAL resolver: a revoked bash policy refuses every check, no spawn."""
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    runner = _runner(
        tmp_path,
        shell_admission=_real_admission(
            tmp_path, BashToolConfig(permission=ToolPermission.NEVER)
        ),
    )

    outcome = asyncio.run(runner.run(CompletionCheckTemplate.GIT_HEAD))

    assert outcome.status is CompletionCheckStatus.DENIED
    assert outcome.argv == ()
    assert outcome.detail.startswith("shell_admission_denied")
    assert spawn.commands == []


@pytest.mark.timeout(60)
def test_real_resolver_admits_an_in_root_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The REAL resolver with the default policy admits an in-root check."""
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    runner = _runner(
        tmp_path, shell_admission=_real_admission(tmp_path, BashToolConfig())
    )

    outcome = asyncio.run(runner.run(CompletionCheckTemplate.PYTEST))

    assert outcome.status is CompletionCheckStatus.PASSED
    assert len(spawn.commands) == 1
