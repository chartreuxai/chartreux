"""P6a/P6b acceptance: the once-only finalization and its cancellation paths.

Covers the P6a acceptance: the once-only finalization awaited immediately
before each ``build_result`` call site in the background monitor (the normal
path and the repeated build after a result-build failure); the finalization
cache on ``SubagentRunAccumulator`` — the P3-declared field P6a consumes
WITHOUT editing ``core/subagents.py`` — with cache creation and reuse
demonstrated without P10; the contract-bound acceptance checks through the
P5 runner; the publication HEAD snapshot only when the contract mandates a
commit artifact, alongside the consumed admission baseline; the unavailable
HEAD; repeated builds running no checks; and no registry lock across the
subprocess. No live-provider calls: the check spawn seam is monkeypatched.

Covers the P6b acceptance: the foreground finalization site, the cancelled-run
disposition (reason recorded, stub finalization cached, no worker credit) for
both cancelled-settled and mid-run cancellation, the drained in-flight check
process, the sibling-scoped drain, and the capacity slot accounted while the
verification runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server import _subagent_execution
from chartreux.app_server._sessions import SessionRuntime, SessionRuntimeRegistry
from chartreux.app_server._subagent_execution import finalize_run_with_drain
from chartreux.app_server._subagent_finalization import (
    RunFinalization,
    finalize_background_run,
)
from chartreux.app_server._subagent_records import (
    CompletionCapability,
    FrozenCompletionContract,
)
from chartreux.app_server.protocol import TurnInterruptParams
from chartreux.core import completion_checks
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.completion_checks import (
    CompletionCheckOutcome,
    CompletionCheckRunner,
    CompletionCheckStatus,
)
from chartreux.core.completion_contract import (
    CompletionArtifactKind,
    CompletionCheckTemplate,
)
from chartreux.core.git.repo import GitRepo
from chartreux.core.llm_models import LLMChunk
from chartreux.core.recovery_models import PackageTier
from chartreux.core.subagents import (
    CancelOutcome,
    EmptySubagentResponseError,
    RunStatus,
    RunStopReason,
    SubagentRunAccumulator,
    TaskArgs,
    TaskResult,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.builtins.bash import BashToolConfig
from chartreux.core.tools.models import PermissionContext, ToolPermission
from chartreux.core.tools.utils import PathAccess
from chartreux.core.utils import kill_async_subprocess
from chartreux.core.utils.shell import spawn_shell_command as _real_shell_spawn
from chartreux.core.workspace import Workspace
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    _background_result,
    _config,
    _retention_registry,
    _wait_result,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.timeout(60)


def _freeze(**overrides: Any) -> FrozenCompletionContract:
    """Freeze a contract with coder defaults plus the given overrides."""
    params: dict[str, Any] = {
        "identity": "coder",
        "identity_revision": "rev-1",
        "seat": None,
        "tier": PackageTier.SUBSTANTIVE,
        "cwd": "/workspace",
        "roots": ("/workspace",),
        "completion_check": None,
        "required_artifacts": None,
        "shell_less": False,
        "audit_mode": "stamp",
        "capability": CompletionCapability.AVAILABLE,
    }
    params.update(overrides)
    return FrozenCompletionContract.freeze(**params)


def _authority() -> Any:
    """A permissive PathAuthority: every path resolves to ALWAYS."""

    class _StubAuthority:
        tool_name = "bash"

        def resolve(self, path: str, access: PathAccess) -> Any:
            return PermissionContext(permission=ToolPermission.ALWAYS)

    return _StubAuthority()


def _child_runtime(cwd: Path) -> SessionRuntime:
    """A stub child runtime exposing cwd, workspace, and path authority."""
    tool_manager = SimpleNamespace(
        workspace=Workspace.for_session(cwd),
        path_authority=lambda tool_name: _authority(),
    )
    return cast(
        SessionRuntime,
        SimpleNamespace(agent_loop=SimpleNamespace(cwd=cwd, tool_manager=tool_manager)),
    )


class _SpawnRecorder:
    """The monkeypatched check spawn seam: records commands, fakes processes."""

    def __init__(self, *, returncode: int = 0) -> None:
        self.commands: list[str] = []
        self.returncode = returncode

    async def __call__(self, command: str, **kwargs: Any) -> Any:
        self.commands.append(command)
        returncode = self.returncode

        class _Process:
            async def communicate(self) -> tuple[bytes, bytes]:
                return (b"ok\n", b"")

            @property
            def returncode(self) -> int:
                return returncode

        return _Process()


def _install_spawn(monkeypatch: pytest.MonkeyPatch, recorder: _SpawnRecorder) -> None:
    monkeypatch.setattr(
        "chartreux.core.completion_checks.spawn_shell_command", recorder
    )


def _install_head_probe(monkeypatch: pytest.MonkeyPatch, head: str) -> list[Path]:
    """Patch ``GitRepo.head_commit_at`` with a recorder returning ``head``."""
    probes: list[Path] = []

    def probe(target: Path) -> str:
        probes.append(target)
        return head

    monkeypatch.setattr(GitRepo, "head_commit_at", staticmethod(probe))
    return probes


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, timeout=30
    )


async def _wait_killed(process: asyncio.subprocess.Process) -> None:
    """Wait for a killed check process's exit status.

    ``Process.wait()`` can hang past transport closure for an
    already-reaped leader (a ``_wait`` waiter appended after the transport
    closed is never resolved); the returncode poll reads the same exit
    evidence without that waiter race.
    """
    for _ in range(100):
        if process.returncode is not None:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("process was never drained")


# ----------------------------------------------------------------------
# Unit level: the once-only finalization and its cache
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalization_runs_bound_checks_and_caches_the_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _freeze(
        cwd=str(tmp_path),
        roots=(str(tmp_path),),
        completion_check=[CompletionCheckTemplate.PYTEST],
    )
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    head_probes = _install_head_probe(monkeypatch, "h" * 40)
    accumulator = SubagentRunAccumulator()
    runtime = _child_runtime(tmp_path)

    first = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head=None,
        runtime=runtime,
    )
    second = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head=None,
        runtime=runtime,
    )

    # ONCE-ONLY: the second await is a cache hit — no check re-runs, no
    # HEAD probe re-fires, and the identical finalization returns.
    assert second is first
    assert accumulator._finalization_cache["run-1"] is first
    assert len(spawn.commands) == 1
    assert "pytest" in spawn.commands[0]
    assert len(first.checks) == 1
    outcome = first.checks[0]
    assert outcome.template is CompletionCheckTemplate.PYTEST
    assert outcome.status is CompletionCheckStatus.PASSED
    assert outcome.exit_code == 0
    # No commit mandate: no HEAD probes at all.
    assert first.publication_head is None
    assert head_probes == []
    assert first.runtime_failure is None


@pytest.mark.asyncio
async def test_finalization_is_run_keyed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _freeze(
        cwd=str(tmp_path),
        roots=(str(tmp_path),),
        completion_check=[CompletionCheckTemplate.PYTEST],
    )
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    accumulator = SubagentRunAccumulator()
    runtime = _child_runtime(tmp_path)

    first = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head=None,
        runtime=runtime,
    )
    other = await finalize_background_run(
        run_id="run-2",
        accumulator=accumulator,
        contract=contract,
        admission_head=None,
        runtime=runtime,
    )

    assert other is not first
    assert set(accumulator._finalization_cache) == {"run-1", "run-2"}
    assert len(spawn.commands) == 2


@pytest.mark.asyncio
async def test_finalization_without_mandates_runs_no_checks_and_no_head_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _freeze(cwd=str(tmp_path), roots=(str(tmp_path),))
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    head_probes = _install_head_probe(monkeypatch, "h" * 40)
    accumulator = SubagentRunAccumulator()

    finalization = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head=None,
        runtime=_child_runtime(tmp_path),
    )

    assert finalization.checks == ()
    assert finalization.publication_head is None
    assert finalization.runtime_failure is None
    assert spawn.commands == []
    assert head_probes == []
    assert accumulator._finalization_cache["run-1"] is finalization


@pytest.mark.asyncio
async def test_finalization_captures_the_publication_head_once(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "file.txt").write_text("one")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=Test",
        "commit",
        "-m",
        "one",
    )
    head = GitRepo.head_commit_at(repo)
    contract = _freeze(
        cwd=str(repo),
        roots=(str(repo),),
        required_artifacts=[CompletionArtifactKind.COMMIT],
    )
    assert contract.requires_commit_artifact is True
    accumulator = SubagentRunAccumulator()

    first = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head="a" * 40,
        runtime=_child_runtime(repo),
    )
    second = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head="a" * 40,
        runtime=_child_runtime(repo),
    )

    # The publication HEAD snapshot is captured at finalization, once; the
    # admission baseline is consumed unchanged.
    assert first.publication_head == head
    assert first.admission_head == "a" * 40
    assert second is first


@pytest.mark.asyncio
async def test_finalization_unavailable_head_is_informational(tmp_path: Path) -> None:
    # tmp_path is not a git repository: the publication HEAD probe fails.
    contract = _freeze(
        cwd=str(tmp_path),
        roots=(str(tmp_path),),
        required_artifacts=[CompletionArtifactKind.COMMIT],
    )
    accumulator = SubagentRunAccumulator()

    finalization = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head="a" * 40,
        runtime=_child_runtime(tmp_path),
    )
    again = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head="a" * 40,
        runtime=_child_runtime(tmp_path),
    )

    # An unavailable HEAD is an evidence-availability fact (None), never a
    # finalization failure — and never a runtime failure of the verifier.
    assert finalization.publication_head is None
    assert finalization.admission_head == "a" * 40
    assert finalization.runtime_failure is None
    assert again is finalization


@pytest.mark.asyncio
async def test_finalization_denied_capability_stamps_checks_without_spawning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _freeze(
        cwd=str(tmp_path),
        roots=(str(tmp_path),),
        completion_check=[CompletionCheckTemplate.PYTEST],
        capability=CompletionCapability.DENIED,
    )
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    accumulator = SubagentRunAccumulator()

    finalization = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head=None,
        runtime=_child_runtime(tmp_path),
    )

    assert len(finalization.checks) == 1
    assert finalization.checks[0].status is CompletionCheckStatus.DENIED
    assert finalization.checks[0].argv == ()
    assert spawn.commands == []
    # A denied authority is a harness-side informational stamp.
    assert finalization.runtime_failure == "denied"


@pytest.mark.asyncio
async def test_finalization_denylisted_check_template_refuses_without_spawning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The P5 admission wiring: the child's bash policy denylists the bound
    check's command, so the acceptance check refuses through the shell
    admission resolver without spawning — an informational stamp.
    """
    contract = _freeze(
        cwd=str(tmp_path),
        roots=(str(tmp_path),),
        completion_check=[CompletionCheckTemplate.PYTEST],
    )
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    accumulator = SubagentRunAccumulator()
    bash_config = BashToolConfig(denylist=["uv"])
    tool_manager = SimpleNamespace(
        workspace=Workspace.for_session(tmp_path),
        path_authority=lambda tool_name: _authority(),
        get_tool_config=lambda name: bash_config,
    )
    runtime = cast(
        SessionRuntime,
        SimpleNamespace(
            agent_loop=SimpleNamespace(cwd=tmp_path, tool_manager=tool_manager)
        ),
    )

    finalization = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head=None,
        runtime=runtime,
    )

    assert len(finalization.checks) == 1
    assert finalization.checks[0].status is CompletionCheckStatus.DENIED
    assert finalization.checks[0].argv == ()
    assert finalization.checks[0].detail.startswith("shell_admission_denied")
    assert spawn.commands == []
    # A shell-admission denial is a harness-side informational stamp.
    assert finalization.runtime_failure == "denied"


@pytest.mark.asyncio
async def test_finalization_broken_child_runtime_stamps_unavailable(
    tmp_path: Path,
) -> None:
    contract = _freeze(
        cwd=str(tmp_path),
        roots=(str(tmp_path),),
        completion_check=[CompletionCheckTemplate.PYTEST],
    )

    class _BrokenToolManager:
        @property
        def workspace(self) -> Any:
            raise RuntimeError("authority retired")

        def path_authority(self, tool_name: str) -> Any:
            raise RuntimeError("authority retired")

    runtime = cast(
        SessionRuntime,
        SimpleNamespace(
            agent_loop=SimpleNamespace(cwd=tmp_path, tool_manager=_BrokenToolManager())
        ),
    )
    accumulator = SubagentRunAccumulator()

    finalization = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head=None,
        runtime=runtime,
    )

    assert finalization.checks == ()
    assert finalization.runtime_failure == "unavailable"
    assert accumulator._finalization_cache["run-1"] is finalization


@pytest.mark.asyncio
async def test_finalization_nonzero_check_is_evidence_not_runtime_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _freeze(
        cwd=str(tmp_path),
        roots=(str(tmp_path),),
        completion_check=[CompletionCheckTemplate.PYTEST],
    )
    spawn = _SpawnRecorder(returncode=1)
    _install_spawn(monkeypatch, spawn)
    accumulator = SubagentRunAccumulator()

    finalization = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head=None,
        runtime=_child_runtime(tmp_path),
    )

    assert finalization.checks[0].status is CompletionCheckStatus.NONZERO
    assert finalization.checks[0].exit_code == 1
    # A failed check is real evidence about the run — never a verifier
    # failure, never an informational runtime stamp.
    assert finalization.runtime_failure is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (CompletionCheckStatus.PASSED, None),
        (CompletionCheckStatus.NONZERO, None),
        (CompletionCheckStatus.DENIED, "denied"),
        (CompletionCheckStatus.TIMED_OUT, "timeout"),
        (CompletionCheckStatus.LAUNCH_FAILURE, "unrunnable"),
    ],
)
async def test_finalization_maps_harness_side_statuses_to_failure_words(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: CompletionCheckStatus,
    expected: str | None,
) -> None:
    contract = _freeze(
        cwd=str(tmp_path),
        roots=(str(tmp_path),),
        completion_check=[CompletionCheckTemplate.PYTEST],
    )

    async def _fake_run(
        self: CompletionCheckRunner, template: CompletionCheckTemplate, **kwargs: Any
    ) -> CompletionCheckOutcome:
        return CompletionCheckOutcome(template=template, status=status, argv=())

    monkeypatch.setattr(CompletionCheckRunner, "run", _fake_run)
    accumulator = SubagentRunAccumulator()

    finalization = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head=None,
        runtime=_child_runtime(tmp_path),
    )

    assert finalization.runtime_failure == expected


@pytest.mark.asyncio
async def test_finalization_defect_stamps_unrunnable_and_never_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _freeze(
        cwd=str(tmp_path),
        roots=(str(tmp_path),),
        completion_check=[CompletionCheckTemplate.PYTEST],
    )

    async def _explode(**kwargs: Any) -> RunFinalization:
        raise RuntimeError("finalization defect")

    monkeypatch.setattr(
        "chartreux.app_server._subagent_finalization._build_finalization", _explode
    )
    accumulator = SubagentRunAccumulator()

    finalization = await finalize_background_run(
        run_id="run-1",
        accumulator=accumulator,
        contract=contract,
        admission_head="a" * 40,
        runtime=_child_runtime(tmp_path),
    )

    assert finalization.checks == ()
    assert finalization.runtime_failure == "unrunnable"
    assert finalization.admission_head == "a" * 40
    assert accumulator._finalization_cache["run-1"] is finalization


@pytest.mark.asyncio
async def test_cancelled_finalization_leaves_the_cache_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _freeze(
        cwd=str(tmp_path),
        roots=(str(tmp_path),),
        completion_check=[CompletionCheckTemplate.PYTEST],
    )
    entered = asyncio.Event()

    async def _hanging_run(
        self: CompletionCheckRunner, template: CompletionCheckTemplate, **kwargs: Any
    ) -> CompletionCheckOutcome:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(CompletionCheckRunner, "run", _hanging_run)
    accumulator = SubagentRunAccumulator()
    task = asyncio.create_task(
        finalize_background_run(
            run_id="run-1",
            accumulator=accumulator,
            contract=contract,
            admission_head=None,
            runtime=_child_runtime(tmp_path),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The cache write happens only once the finalization settles: a
    # cancelled finalization leaves the cache empty (P6b drains the
    # already-started verification).
    assert accumulator._finalization_cache == {}


# ----------------------------------------------------------------------
# Integration: the background monitor threads the finalization
# ----------------------------------------------------------------------


class _NotedBackend(FakeBackend):
    """A parent backend that answers every turn with terminal content."""

    async def complete(self, **kwargs: Any) -> LLMChunk:
        return mock_llm_chunk(content="parent noted")

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        yield mock_llm_chunk(content="parent noted")


async def _registry_with_parent(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> tuple[SessionRuntimeRegistry, AgentLoop]:
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    # The parent grants bash (a real orchestrator does), so its delegation
    # ceiling carries the shell tool and the harness verification capability
    # derives as available.
    parent = build_test_agent_loop(
        config=_config(enabled_tools=["task", "bash"]), backend=_NotedBackend()
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


async def _foreground_registry_with_shell(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> tuple[SessionRuntimeRegistry, AgentLoop]:
    """The foreground registry with the parent granting bash — the shell tool
    carries the harness verification capability (the _registry_with_parent
    precedent; the granted shell is the delegation ceiling).
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(
        config=_config(enabled_tools=["task", "bash"]), backend=FakeBackend()
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


def _find_run(record: Any, run_id: str) -> Any:
    return next(
        item
        for item in [record.current_run, *record.run_history]
        if item is not None and item.run_id == run_id
    )


def _record_finalize_calls(
    monkeypatch: pytest.MonkeyPatch, order: list[str]
) -> list[dict[str, Any]]:
    """Record the monitor's finalization calls (and their order position)."""
    calls: list[dict[str, Any]] = []
    real = _subagent_execution.finalize_background_run

    async def _recording(**kwargs: Any) -> RunFinalization:
        order.append("finalize")
        result = await real(**kwargs)
        calls.append({**kwargs, "result": result})
        return result

    monkeypatch.setattr(_subagent_execution, "finalize_background_run", _recording)
    return calls


def _record_build_order(monkeypatch: pytest.MonkeyPatch, order: list[str]) -> None:
    """Record each build_result call in the shared order list."""
    real = SubagentRunAccumulator.build_result

    def _recording(self: SubagentRunAccumulator, **kwargs: Any) -> Any:
        order.append("build")
        return real(self, **kwargs)

    monkeypatch.setattr(SubagentRunAccumulator, "build_result", _recording)


@pytest.mark.asyncio
async def test_normal_path_finalizes_once_before_the_result_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = FakeBackend([mock_llm_chunk(content="child done")])
    registry, parent = await _registry_with_parent(monkeypatch, child_backend)
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    head_probes = _install_head_probe(monkeypatch, "h" * 40)
    order: list[str] = []
    calls = _record_finalize_calls(monkeypatch, order)
    _record_build_order(monkeypatch, order)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(
            registry,
            TaskArgs(
                task="Mandated work",
                identity="coder",
                background=True,
                retain=True,
                completion_check=[CompletionCheckTemplate.PYTEST],
            ),
            ctx,
        )
        assert ack.agent_id is not None and ack.run_id is not None
        result = await _wait_result(registry, ack.agent_id, ack.run_id)
        # The published result is unchanged: the finalization assembles
        # evidence, it does not change behavior.
        assert result.completed is True
        assert result.response == "child done"
        # The finalization was awaited immediately BEFORE the result build.
        assert order == ["finalize", "build"]
        assert len(calls) == 1
        call = calls[0]
        assert call["run_id"] == ack.run_id
        finalization = call["result"]
        assert isinstance(finalization, RunFinalization)
        # The cache on the accumulator holds exactly this finalization,
        # keyed by the run id — creation demonstrated without P10.
        accumulator = call["accumulator"]
        assert accumulator._finalization_cache[ack.run_id] is finalization
        # The contract-bound check ran through the P5 runner (spawn faked).
        assert len(spawn.commands) == 1
        assert len(finalization.checks) == 1
        assert finalization.checks[0].status is CompletionCheckStatus.PASSED
        # No commit mandate: no HEAD probes at all.
        assert finalization.publication_head is None
        assert head_probes == []
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_repeated_builds_run_no_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A result-build failure rebuilds through the exception path (P6a).

    The first build_result raises the empty-response error the build itself
    raises for a completed run with no response (the gateway rejects empty
    final output upstream, so the failure is simulated at the build seam);
    the monitor rebuilds the result through its exception path. Both build
    sites must be served by ONE finalization.
    """
    child_backend = FakeBackend([mock_llm_chunk(content="child done")])
    registry, parent = await _registry_with_parent(monkeypatch, child_backend)
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    order: list[str] = []
    calls = _record_finalize_calls(monkeypatch, order)
    real_build = SubagentRunAccumulator.build_result
    builds = 0

    def _flaky_build(self: SubagentRunAccumulator, **kwargs: Any) -> Any:
        nonlocal builds
        builds += 1
        order.append("build")
        if builds == 1:
            raise EmptySubagentResponseError(
                "Completed subagent produced an empty response"
            )
        return real_build(self, **kwargs)

    monkeypatch.setattr(SubagentRunAccumulator, "build_result", _flaky_build)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(
            registry,
            TaskArgs(
                task="Mandated work",
                identity="coder",
                background=True,
                retain=True,
                completion_check=[CompletionCheckTemplate.PYTEST],
            ),
            ctx,
        )
        assert ack.agent_id is not None and ack.run_id is not None
        result = await _wait_result(registry, ack.agent_id, ack.run_id)
        # The failed result-build rebuilds through the exception path.
        assert result.completed is False
        record = registry._agent_records[ack.agent_id]
        run = _find_run(record, ack.run_id)
        assert run.status is RunStatus.FAILED
        # Both build sites were served by ONE finalization: the second await
        # is a cache hit and no check re-ran.
        assert order == ["finalize", "build", "finalize", "build"]
        assert len(calls) == 2
        first, second = (call["result"] for call in calls)
        assert second is first
        assert len(spawn.commands) == 1
        accumulator = calls[0]["accumulator"]
        assert list(accumulator._finalization_cache) == [ack.run_id]
        assert accumulator._finalization_cache[ack.run_id] is first
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_commit_mandate_finalization_carries_both_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = FakeBackend([mock_llm_chunk(content="child done")])
    registry, parent = await _registry_with_parent(monkeypatch, child_backend)
    head_probes = _install_head_probe(monkeypatch, "h" * 40)
    order: list[str] = []
    calls = _record_finalize_calls(monkeypatch, order)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(
            registry,
            TaskArgs(
                task="Commit work",
                identity="coder",
                background=True,
                retain=True,
                required_artifacts=[CompletionArtifactKind.COMMIT],
            ),
            ctx,
        )
        assert ack.agent_id is not None and ack.run_id is not None
        result = await _wait_result(registry, ack.agent_id, ack.run_id)
        assert result.completed is True
        assert len(calls) == 1
        finalization = calls[0]["result"]
        # The admission baseline (P4, captured PRE-EXECUTION) is consumed
        # unchanged; the publication snapshot is captured at finalization.
        assert finalization.admission_head == "h" * 40
        assert finalization.publication_head == "h" * 40
        # No check templates bound: no subprocess.
        assert finalization.checks == ()
        assert finalization.runtime_failure is None
        # One admission probe (P4) plus one publication probe (P6a).
        assert len(head_probes) == 2
    finally:
        await registry.drain_children()
        await parent.aclose()


# ----------------------------------------------------------------------
# P6b: the foreground/cancellation finalization
# ----------------------------------------------------------------------


async def _wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    """Poll a condition to truth, with a bounded deadline."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.01)


def _record_finalize_entry_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Record finalization call entries BEFORE the await lands (cancelled
    finalizations stay observable too).
    """
    calls: list[dict[str, Any]] = []
    real = _subagent_execution.finalize_background_run

    async def _recording(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return await real(**kwargs)

    monkeypatch.setattr(_subagent_execution, "finalize_background_run", _recording)
    return calls


@pytest.mark.asyncio
async def test_foreground_finalization_finalizes_before_the_result_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The foreground build site finalizes first (order + cache + capacity)."""
    child_backend = FakeBackend([mock_llm_chunk(content="child done")])
    registry, parent = await _foreground_registry_with_shell(monkeypatch, child_backend)
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    head_probes = _install_head_probe(monkeypatch, "h" * 40)
    order: list[str] = []
    calls = _record_finalize_calls(monkeypatch, order)
    _record_build_order(monkeypatch, order)
    try:
        ctx = InvokeContext(tool_call_id="fg-1", session_id=parent.session_id)
        items = [
            item
            async for item in registry.run(
                TaskArgs(
                    task="Mandated work",
                    identity="coder",
                    background=False,
                    completion_check=[CompletionCheckTemplate.PYTEST],
                ),
                ctx,
            )
        ]
        result = cast(TaskResult, items[-1])
        # The completed result is unchanged: evidence assembles, behavior
        # does not change, and no cancellation reason is stamped.
        assert result.completed is True
        assert result.response == "child done"
        assert result.stop_reason is None
        # The finalization was awaited immediately BEFORE the result build.
        assert order == ["finalize", "build"]
        assert len(calls) == 1
        call = calls[0]
        finalization = call["result"]
        assert isinstance(finalization, RunFinalization)
        # The cache holds exactly this finalization, keyed by the foreground
        # run key (the child turn id).
        assert call["run_id"]
        accumulator = call["accumulator"]
        assert accumulator._finalization_cache[call["run_id"]] is finalization
        assert len(spawn.commands) == 1
        assert finalization.checks[0].status is CompletionCheckStatus.PASSED
        # No commit mandate: no HEAD probes at all.
        assert finalization.publication_head is None
        assert head_probes == []
        # P6b capacity: the slot is released only after the finalization —
        # empty once the run settled.
        assert len(registry._active_work_slots) == 0
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_interrupted_settlement_takes_the_cancellation_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case (a), foreground: a turn settled INTERRUPTED finalizes through
    the cancellation stub — the reason is recorded, no check spawns, no
    worker credit.
    """
    child_backend = BlockingBackend()
    registry, parent = await _foreground_registry_with_shell(monkeypatch, child_backend)
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    calls = _record_finalize_entry_calls(monkeypatch)
    try:
        ctx = InvokeContext(tool_call_id="fg-cancelled", session_id=parent.session_id)
        items: list[Any] = []

        async def consume() -> None:
            async for item in registry.run(
                TaskArgs(
                    task="Mandated work",
                    identity="coder",
                    background=False,
                    completion_check=[CompletionCheckTemplate.PYTEST],
                ),
                ctx,
            ):
                items.append(item)

        task = asyncio.ensure_future(consume())
        await asyncio.wait_for(child_backend.started.wait(), 5)
        child_runtime = next(iter(registry._children.values()))
        active = child_runtime.turns.active_turn
        assert active is not None
        child_runtime.turns.interrupt(
            TurnInterruptParams(
                session_id=active.session_id, expected_turn_id=active.id
            )
        )
        await asyncio.wait_for(task, 10)
        result = cast(TaskResult, items[-1])
        # The cancelled run gets no worker credit and preserves its reason.
        assert result.completed is False
        assert result.stop_reason is RunStopReason.USER_CANCELLED
        assert "run cancelled (user_cancelled)" in result.response
        # The stub was cached and the finalization ran NO checks.
        assert spawn.commands == []
        assert len(calls) == 1
        accumulator = calls[0]["accumulator"]
        cached = list(accumulator._finalization_cache.values())
        assert len(cached) == 1
        assert isinstance(cached[0], RunFinalization)
        assert cached[0].checks == ()
        assert cached[0].runtime_failure == "unavailable"
        # The slot was released by the settled foreground teardown.
        assert len(registry._active_work_slots) == 0
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_cancelled_background_run_finalizes_through_the_cancellation_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case (a), background: a run cancelled before the audit starts takes
    the cancellation disposition in the monitor's normal path — no checks,
    the stub cached, the reason preserved on the result.
    """
    child_backend = BlockingBackend()
    registry, parent = await _registry_with_parent(monkeypatch, child_backend)
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    calls = _record_finalize_entry_calls(monkeypatch)
    try:
        ctx = InvokeContext(tool_call_id="bg-cancelled", session_id=parent.session_id)
        ack = await _background_result(
            registry,
            TaskArgs(
                task="Mandated work",
                identity="coder",
                background=True,
                retain=True,
                completion_check=[CompletionCheckTemplate.PYTEST],
            ),
            ctx,
        )
        assert ack.agent_id is not None and ack.run_id is not None
        record = registry._agent_records[ack.agent_id]
        await asyncio.wait_for(child_backend.started.wait(), 5)
        outcome = await registry.cancel_run(
            record.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        assert outcome.outcome is CancelOutcome.STOP_REQUESTED
        result = await asyncio.wait_for(_wait_result(registry, record.agent_id), 5)
        run = _find_run(record, ack.run_id)
        assert run.status is RunStatus.CANCELLED
        assert result.completed is False
        assert result.stop_reason is RunStopReason.USER_CANCELLED
        assert "run cancelled (user_cancelled)" in result.response
        # A cancelled run never spawns verification: the stub is the cached
        # finalization instead.
        assert spawn.commands == []
        assert len(calls) == 1
        finalization = calls[0]["accumulator"]._finalization_cache[ack.run_id]
        assert isinstance(finalization, RunFinalization)
        assert finalization.checks == ()
        assert finalization.runtime_failure == "unavailable"
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_cancelled_mid_audit_drains_the_in_flight_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case (b), foreground: cancellation during an already-started check
    drains the in-flight check process, discards the partial findings, keeps
    the capacity slot accounted until the finalization settled, and leaves
    the teardown clean.
    """
    child_backend = FakeBackend([mock_llm_chunk(content="child done")])
    registry, parent = await _foreground_registry_with_shell(monkeypatch, child_backend)
    spawned: list[asyncio.subprocess.Process] = []
    finalize_entries = _record_finalize_entry_calls(monkeypatch)

    # The check's real `sleep 30` process IS the in-flight check: the seam
    # returns it and the runner hangs in communicate() until the drain.
    async def delayed_spawn(command: str, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await _real_shell_spawn("sleep 30", cwd=Path(kwargs["cwd"]))
        spawned.append(process)
        return process

    monkeypatch.setattr(
        "chartreux.core.completion_checks.spawn_shell_command", delayed_spawn
    )
    try:
        ctx = InvokeContext(tool_call_id="fg-drain", session_id=parent.session_id)
        consumed: list[Any] = []

        async def consume() -> None:
            async for item in registry.run(
                TaskArgs(
                    task="Mandated work",
                    identity="coder",
                    background=False,
                    completion_check=[CompletionCheckTemplate.PYTEST],
                ),
                ctx,
            ):
                consumed.append(item)

        task = asyncio.ensure_future(consume())
        await _wait_until(lambda: len(spawned) == 1)
        # The capacity slot stays accounted while the verification runs.
        assert len(registry._active_work_slots) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        # The in-flight check process was drained — killed, never orphaned.
        assert len(spawned) == 1
        await _wait_killed(spawned[0])
        assert spawned[0].returncode is not None
        # The stub finalization is the cached evidence: UNAVAILABLE, no
        # checks, no partial findings published.
        assert len(finalize_entries) == 1
        accumulator = finalize_entries[0]["accumulator"]
        cached = list(accumulator._finalization_cache.values())
        assert len(cached) == 1
        assert isinstance(cached[0], RunFinalization)
        assert cached[0].checks == ()
        assert cached[0].runtime_failure == "unavailable"
        # Teardown: the child is detached and the slot released.
        await registry.drain_children()
        assert len(registry._active_work_slots) == 0
        assert not registry._children
    finally:
        for process in spawned:
            if process.returncode is None:
                await kill_async_subprocess(process, kill_exited_process_group=False)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_drain_wrapper_kills_only_this_tasks_check_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The drain is sibling-scoped: only the wrapper's own task's in-flight
    check processes are killed; a concurrent spawn from another task is
    never collateral.
    """
    spawned: list[tuple[asyncio.Task[Any] | None, asyncio.subprocess.Process]] = []

    async def delayed_spawn(command: str, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await _real_shell_spawn("sleep 30", cwd=tmp_path)
        spawned.append((asyncio.current_task(), process))
        return process

    monkeypatch.setattr(
        "chartreux.core.completion_checks.spawn_shell_command", delayed_spawn
    )
    both_spawned = asyncio.Event()
    sibling_tasks: list[asyncio.Task[None]] = []

    async def fake_finalize(**kwargs: Any) -> RunFinalization:
        async def sibling() -> None:
            await completion_checks.spawn_shell_command("ignored", cwd=tmp_path)

        sibling_task = asyncio.ensure_future(sibling())
        sibling_tasks.append(sibling_task)
        await completion_checks.spawn_shell_command("ignored", cwd=tmp_path)
        both_spawned.set()
        await asyncio.sleep(3600)  # the audit never settles in this test
        raise AssertionError("unreachable: both spawns are held in-flight")

    monkeypatch.setattr(_subagent_execution, "finalize_background_run", fake_finalize)
    contract = _freeze(
        cwd=str(tmp_path),
        roots=(str(tmp_path),),
        completion_check=[CompletionCheckTemplate.PYTEST],
    )
    try:
        wrapper_task = asyncio.ensure_future(
            finalize_run_with_drain(
                run_id="run-drain",
                accumulator=SubagentRunAccumulator(),
                contract=contract,
                admission_head=None,
                runtime=_child_runtime(tmp_path),
            )
        )
        await _wait_until(lambda: len(spawned) == 2)
        wrapper_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(wrapper_task, 10)
        own = [process for owner, process in spawned if owner is wrapper_task]
        others = [process for owner, process in spawned if owner is not wrapper_task]
        assert len(own) == 1 and len(others) == 1
        # Own in-flight check: drained. Sibling: never touched.
        await _wait_killed(own[0])
        assert own[0].returncode is not None
        assert others[0].returncode is None
    finally:
        for sibling_task in sibling_tasks:
            sibling_task.cancel()
        if sibling_tasks:
            await asyncio.gather(*sibling_tasks, return_exceptions=True)
        for _, process in spawned:
            if process.returncode is None:
                await kill_async_subprocess(process, kill_exited_process_group=False)
        await asyncio.gather(
            *(process.wait() for _, process in spawned if process.returncode is None),
            return_exceptions=True,
        )
