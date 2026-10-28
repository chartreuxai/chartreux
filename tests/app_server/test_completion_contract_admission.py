"""P4 — the completion contract frozen at admission and rendered to the worker.

Covers the P4 acceptance: the contract freeze at admission (identity/seat/
tier, cwd/roots, mandates, envelope requirement/exemption, audit mode,
capability inputs) stored in the run record; the rendering into the child
prompt (schema version, requirement IDs, check bindings, artifact mandates,
exemptions); reuse = a NEW contract; resume = the failed run's run-scoped
snapshot (rendered into the resumed prompt too); and the admission HEAD
baseline captured PRE-EXECUTION alongside the frozen contract — only when a
commit artifact is required. A worker commit changes the HEAD the captured
baseline is compared against; runs without a commit mandate perform no HEAD
probes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._sessions import (
    SessionRuntime,
    SessionRuntimeRegistry,
    _AgentState,
)
from chartreux.app_server._subagent_execution import (
    _shell_less_launch,
    _verification_capability,
)
from chartreux.app_server._subagent_records import (
    CompletionCapability,
    FrozenCompletionContract,
)
from chartreux.app_server.models import TextContentBlock
from chartreux.app_server.protocol import TurnStartParams
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop.errors import EmptyLLMResponseError
from chartreux.core.agents.identities import get_builtin_identity
from chartreux.core.completion_contract import (
    CompletionArtifactKind,
    CompletionCheckTemplate,
)
from chartreux.core.git.repo import GitRepo
from chartreux.core.llm_models import FunctionCall, LLMChunk, LLMMessage, Role, ToolCall
from chartreux.core.recovery_models import PackageTier
from chartreux.core.subagents import (
    LaunchConfig,
    LaunchToolOverride,
    ResumeOutcome,
    TaskArgs,
    TaskResult,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.models import ToolPermission
from tests.app_server._subagent_test_support import (
    _background_result,
    _config,
    _retention_registry,
    _wait_result,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


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


def _requirement_ids(contract: FrozenCompletionContract) -> list[str]:
    assert contract.envelope is not None
    return [
        requirement.requirement_id for requirement in contract.envelope.requirements
    ]


# ----------------------------------------------------------------------
# The freeze: envelope requirement/exemption matrix (unit level)
# ----------------------------------------------------------------------


def test_freeze_substantive_seat_freezes_mandates_and_requires_envelope() -> None:
    contract = _freeze(
        completion_check=[CompletionCheckTemplate.PYTEST],
        required_artifacts=[
            CompletionArtifactKind.COMMIT,
            CompletionArtifactKind.TOUCHED_PATH,
        ],
    )
    assert contract.schema_version == 1
    assert contract.identity == "coder"
    assert contract.identity_revision == "rev-1"
    assert contract.tier is PackageTier.SUBSTANTIVE
    assert contract.cwd == "/workspace"
    assert contract.roots == ("/workspace",)
    assert contract.audit_mode == "stamp"
    assert contract.capability is CompletionCapability.AVAILABLE
    assert contract.envelope_required is True
    assert contract.exemptions == ()
    assert _requirement_ids(contract) == [
        "check:pytest",
        "artifact:commit",
        "artifact:touched_path",
    ]
    assert contract.requires_commit_artifact is True
    assert contract.envelope is not None
    check, commit, touched = contract.envelope.requirements
    assert check.template is CompletionCheckTemplate.PYTEST
    assert commit.artifact is CompletionArtifactKind.COMMIT
    assert touched.artifact is CompletionArtifactKind.TOUCHED_PATH


def test_freeze_without_mandates_disables_only_the_mandates() -> None:
    contract = _freeze()
    assert contract.envelope is None
    assert contract.requires_commit_artifact is False
    # Fail-open: None mandates disable only themselves — the seat
    # classification (substantive seats require) stands.
    assert contract.envelope_required is True
    assert contract.exemptions == ()


def test_freeze_reviewer_seat_is_exempt() -> None:
    contract = _freeze(
        identity="reviewer",
        completion_check=[CompletionCheckTemplate.PYTEST],
        required_artifacts=[CompletionArtifactKind.COMMIT],
    )
    assert contract.envelope_required is False
    assert contract.exemptions == ("reviewer",)
    # The mandates still freeze as data; the exemption lifts the requirement,
    # so an exempt seat never requires the commit artifact (no HEAD probe).
    assert contract.envelope is not None
    assert contract.requires_commit_artifact is False


def test_freeze_read_only_seat_without_deliverables_is_exempt() -> None:
    contract = _freeze(identity="researcher")
    assert contract.envelope_required is False
    assert contract.exemptions == ("read_only",)
    assert contract.envelope is None
    assert contract.requires_commit_artifact is False


def test_freeze_read_only_seat_with_deliverables_carries_reduced_matrix() -> None:
    contract = _freeze(
        identity="researcher",
        completion_check=[CompletionCheckTemplate.PYTEST],
        required_artifacts=[
            CompletionArtifactKind.COMMIT,
            CompletionArtifactKind.CHECK_PASS,
            CompletionArtifactKind.TOUCHED_PATH,
        ],
    )
    assert contract.envelope_required is True
    assert contract.exemptions == ("reduced_matrix",)
    # The commit artifact is NOT mandated; check_pass rides the named check.
    assert _requirement_ids(contract) == [
        "check:pytest",
        "artifact:check_pass",
        "artifact:touched_path",
    ]
    assert contract.requires_commit_artifact is False


def test_freeze_reduced_matrix_drops_check_pass_without_named_check() -> None:
    contract = _freeze(
        identity="advisor",
        required_artifacts=[
            CompletionArtifactKind.CHECK_PASS,
            CompletionArtifactKind.TOUCHED_PATH,
        ],
    )
    assert contract.envelope_required is True
    assert contract.exemptions == ("reduced_matrix",)
    assert _requirement_ids(contract) == ["artifact:touched_path"]


def test_freeze_shell_less_launch_strips_check_classes() -> None:
    contract = _freeze(
        completion_check=[CompletionCheckTemplate.PYTEST],
        required_artifacts=[
            CompletionArtifactKind.CHECK_PASS,
            CompletionArtifactKind.COMMIT,
        ],
        shell_less=True,
    )
    assert contract.envelope_required is True
    assert contract.exemptions == ("shell_less_checks",)
    assert _requirement_ids(contract) == ["artifact:commit"]
    assert contract.requires_commit_artifact is True


def test_shell_less_launch_classification() -> None:
    coder = get_builtin_identity("coder")
    researcher = get_builtin_identity("researcher")
    assert coder is not None and researcher is not None
    # A custom identity carries no harness-known posture: shell-less.
    assert _shell_less_launch(TaskArgs(task="t"), None) is True
    # The shipped seats are not shell-less by identity (post-B).
    assert _shell_less_launch(TaskArgs(task="t"), coder) is False
    assert _shell_less_launch(TaskArgs(task="t"), researcher) is False
    # Explicitly disabled tools make a shipped seat shell-less.
    assert (
        _shell_less_launch(
            TaskArgs(task="t", config=LaunchConfig(disabled_tools=["bash"])), coder
        )
        is True
    )
    assert (
        _shell_less_launch(
            TaskArgs(
                task="t",
                config=LaunchConfig(
                    tools={"bash": LaunchToolOverride(permission=ToolPermission.NEVER)}
                ),
            ),
            coder,
        )
        is True
    )
    assert (
        _shell_less_launch(
            TaskArgs(
                task="t",
                config=LaunchConfig(
                    tools={"bash": LaunchToolOverride(permission=ToolPermission.ALWAYS)}
                ),
            ),
            coder,
        )
        is False
    )


def _stub_parent(delegable: object) -> SessionRuntime:
    return cast(
        SessionRuntime,
        SimpleNamespace(
            agent_loop=SimpleNamespace(
                tool_manager=SimpleNamespace(delegable_tools=delegable)
            )
        ),
    )


def test_verification_capability_derives_from_parent_delegation() -> None:
    assert (
        _verification_capability(_stub_parent({"bash": object()}))
        is CompletionCapability.AVAILABLE
    )
    assert (
        _verification_capability(_stub_parent({"read_file": object()}))
        is CompletionCapability.DENIED
    )
    assert (
        _verification_capability(_stub_parent(object()))
        is CompletionCapability.UNAVAILABLE
    )


# ----------------------------------------------------------------------
# The rendering into the child prompt (unit level)
# ----------------------------------------------------------------------


def test_render_prompt_section_carries_the_frozen_contract() -> None:
    contract = _freeze(
        completion_check=[CompletionCheckTemplate.PYTEST],
        required_artifacts=[CompletionArtifactKind.COMMIT],
        seat="review.deep",
    )
    section = contract.render_prompt_section()
    assert "schema version: 1" in section
    assert "completion_envelope_v1" in section
    assert "envelope required: yes" in section
    assert "exemptions: none" in section
    assert "audit mode: stamp" in section
    assert "check:pytest" in section
    assert "check `pytest`" in section
    assert "artifact:commit" in section
    assert "artifact `commit`" in section


def test_render_prompt_section_names_exemptions_and_absent_mandates() -> None:
    contract = _freeze(identity="reviewer", seat="review.deep")
    section = contract.render_prompt_section()
    assert "envelope required: no" in section
    assert "exemptions: reviewer" in section
    assert "mandates: none" in section


# ----------------------------------------------------------------------
# Admission integration (real loops through the registry)
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


async def _until(predicate: Any, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


def _find_run(record: Any, run_id: str) -> Any:
    return next(
        item
        for item in [record.current_run, *record.run_history]
        if item is not None and item.run_id == run_id
    )


def _child_prompt(backend: FakeBackend) -> str:
    assert backend.requests_messages
    return next(
        str(message.content)
        for message in backend.requests_messages[0]
        if message.role is Role.user
    )


@pytest.mark.asyncio
async def test_admission_freezes_contract_and_renders_it_into_the_child_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = FakeBackend([mock_llm_chunk(content="child done")])
    registry, parent = await _registry_with_parent(monkeypatch, child_backend)
    probe_calls: list[Path] = []

    def probe(cwd: Path) -> str | None:
        probe_calls.append(cwd)
        return "f" * 40

    monkeypatch.setattr(
        "chartreux.app_server._subagent_execution._capture_admission_head", probe
    )
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
                required_artifacts=[
                    CompletionArtifactKind.COMMIT,
                    CompletionArtifactKind.TOUCHED_PATH,
                ],
            ),
            ctx,
        )
        assert ack.agent_id is not None and ack.run_id is not None
        await registry.wait_for_agent(ack.agent_id, ack.run_id)
        record = registry._agent_records[ack.agent_id]
        run = _find_run(record, ack.run_id)
        contract = run.completion_contract
        assert contract is not None
        assert contract.identity == "coder"
        assert contract.identity_revision == record.identity_revision
        assert contract.seat is None
        assert contract.tier is PackageTier.SUBSTANTIVE
        child_cwd = record.runtime.agent_loop.cwd
        assert contract.cwd == str(child_cwd)
        assert contract.cwd in contract.roots
        assert contract.audit_mode == "stamp"
        assert contract.capability is CompletionCapability.AVAILABLE
        assert contract.envelope_required is True
        assert contract.exemptions == ()
        assert _requirement_ids(contract) == [
            "check:pytest",
            "artifact:commit",
            "artifact:touched_path",
        ]
        # The admission HEAD baseline is captured at admission (the contract
        # requires a commit artifact).
        assert run.admission_head == "f" * 40
        assert probe_calls == [child_cwd]
        # The frozen contract renders into the child prompt.
        prompt = _child_prompt(child_backend)
        assert "Mandated work" in prompt
        assert contract.render_prompt_section() in prompt
        assert "schema version: 1" in prompt
        assert "check:pytest" in prompt
        assert "artifact:commit" in prompt
        assert "artifact:touched_path" in prompt
        assert "exemptions: none" in prompt
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_reviewer_launch_freezes_an_exempt_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = FakeBackend([mock_llm_chunk(content="review done")])
    registry, parent = await _registry_with_parent(monkeypatch, child_backend)
    probe_calls: list[Path] = []

    def probe(cwd: Path) -> str | None:
        probe_calls.append(cwd)
        return "f" * 40

    monkeypatch.setattr(
        "chartreux.app_server._subagent_execution._capture_admission_head", probe
    )
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(
            registry,
            TaskArgs(
                task="Review the change",
                identity="reviewer",
                background=True,
                retain=True,
                completion_check=[CompletionCheckTemplate.PYTEST],
                required_artifacts=[CompletionArtifactKind.COMMIT],
            ),
            ctx,
        )
        assert ack.agent_id is not None and ack.run_id is not None
        await registry.wait_for_agent(ack.agent_id, ack.run_id)
        record = registry._agent_records[ack.agent_id]
        run = _find_run(record, ack.run_id)
        contract = run.completion_contract
        assert contract is not None
        assert contract.identity == "reviewer"
        assert contract.envelope_required is False
        assert contract.exemptions == ("reviewer",)
        # The exemption lifts the commit requirement: no HEAD probe fires.
        assert run.admission_head is None
        assert probe_calls == []
        prompt = _child_prompt(child_backend)
        assert "envelope required: no" in prompt
        assert "exemptions: reviewer" in prompt
    finally:
        await registry.drain_children()
        await parent.aclose()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, timeout=30
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_worker_commit_changes_the_captured_admission_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    head_one = GitRepo.head_commit_at(repo)

    child_backend = FakeBackend([mock_llm_chunk(content="child done")])
    registry, parent = await _registry_with_parent(monkeypatch, child_backend)

    def probe(cwd: Path) -> str | None:
        return GitRepo.head_commit_at(repo)

    monkeypatch.setattr(
        "chartreux.app_server._subagent_execution._capture_admission_head", probe
    )
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
        await registry.wait_for_agent(ack.agent_id, ack.run_id)
        record = registry._agent_records[ack.agent_id]
        run = _find_run(record, ack.run_id)
        # The baseline is captured PRE-EXECUTION: it holds the admission HEAD.
        assert run.admission_head == head_one
        # The worker commits: the publication HEAD moves away from the
        # captured baseline.
        (repo / "file.txt").write_text("two")
        _git(repo, "add", ".")
        _git(
            repo,
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "-m",
            "two",
        )
        head_two = GitRepo.head_commit_at(repo)
        assert head_two != head_one
        # The captured baseline did not move with the worker's commit.
        assert run.admission_head == head_one
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_runs_without_a_commit_mandate_perform_no_head_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = FakeBackend([
        [mock_llm_chunk(content="first")],
        [mock_llm_chunk(content="second")],
    ])
    registry, parent = await _registry_with_parent(monkeypatch, child_backend)
    probe_calls: list[Path] = []

    def probe(cwd: Path) -> str | None:
        probe_calls.append(cwd)
        return "f" * 40

    monkeypatch.setattr(
        "chartreux.app_server._subagent_execution._capture_admission_head", probe
    )
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        # A check mandate without a commit artifact performs no HEAD probe.
        first = await _background_result(
            registry,
            TaskArgs(
                task="Check only",
                identity="coder",
                background=True,
                retain=True,
                completion_check=[CompletionCheckTemplate.PYTEST],
            ),
            ctx,
        )
        assert first.agent_id is not None and first.run_id is not None
        await registry.wait_for_agent(first.agent_id, first.run_id)
        # A run without any mandates performs no HEAD probe either.
        second = await _background_result(
            registry,
            TaskArgs(
                task="No mandates", identity="coder", background=True, retain=True
            ),
            ctx,
        )
        assert second.agent_id is not None and second.run_id is not None
        await registry.wait_for_agent(second.agent_id, second.run_id)
        assert probe_calls == []
        record = registry._agent_records[first.agent_id]
        first_run = _find_run(record, first.run_id)
        assert first_run.admission_head is None
        first_contract = first_run.completion_contract
        assert first_contract is not None
        assert _requirement_ids(first_contract) == ["check:pytest"]
        second_record = registry._agent_records[second.agent_id]
        second_run = _find_run(second_record, second.run_id)
        assert second_run.admission_head is None
        second_contract = second_run.completion_contract
        assert second_contract is not None
        assert second_contract.envelope is None
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_reuse_admission_freezes_a_new_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = FakeBackend([
        [mock_llm_chunk(content="first run")],
        [mock_llm_chunk(content="second run")],
    ])
    registry, parent = await _registry_with_parent(monkeypatch, child_backend)
    probe_calls: list[Path] = []

    def probe(cwd: Path) -> str | None:
        probe_calls.append(cwd)
        return "f" * 40

    monkeypatch.setattr(
        "chartreux.app_server._subagent_execution._capture_admission_head", probe
    )
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        first = await _background_result(
            registry,
            TaskArgs(
                task="First assignment",
                identity="coder",
                background=True,
                retain=True,
                completion_check=[CompletionCheckTemplate.PYTEST],
            ),
            ctx,
        )
        assert first.agent_id is not None and first.run_id is not None
        await registry.wait_for_agent(first.agent_id, first.run_id)
        record = registry._agent_records[first.agent_id]
        await _until(lambda: record.state is _AgentState.IDLE)
        # Reuse: the retask freezes a NEW contract from the new launch args.
        second = await _background_result(
            registry,
            TaskArgs(
                task="Second assignment",
                identity="coder",
                agent_id=first.agent_id,
                background=True,
                completion_check=[CompletionCheckTemplate.RUFF_CHECK],
                required_artifacts=[CompletionArtifactKind.COMMIT],
            ),
            ctx,
        )
        assert second.agent_id is not None and second.agent_id == first.agent_id
        assert second.run_id is not None and second.run_id != first.run_id
        await registry.wait_for_agent(second.agent_id, second.run_id)
        first_run = _find_run(record, first.run_id)
        second_run = _find_run(record, second.run_id)
        first_contract = first_run.completion_contract
        second_contract = second_run.completion_contract
        assert first_contract is not None and second_contract is not None
        assert first_contract is not second_contract
        assert _requirement_ids(first_contract) == ["check:pytest"]
        assert _requirement_ids(second_contract) == [
            "check:ruff_check",
            "artifact:commit",
        ]
        # The first run's contract and baseline are unchanged by the reuse;
        # the second run's commit mandate arms the probe exactly once.
        assert first_run.admission_head is None
        assert second_run.admission_head == "f" * 40
        assert probe_calls == [record.runtime.agent_loop.cwd]
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_admission_freezes_contract_and_renders_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = FakeBackend([mock_llm_chunk(content="foreground done")])
    registry, parent = await _registry_with_parent(monkeypatch, child_backend)
    probe_calls: list[Path] = []

    def probe(cwd: Path) -> str | None:
        probe_calls.append(cwd)
        return "f" * 40

    monkeypatch.setattr(
        "chartreux.app_server._subagent_execution._capture_admission_head", probe
    )
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        result = cast(
            TaskResult,
            [
                item
                async for item in registry.run(
                    TaskArgs(
                        task="Foreground work",
                        identity="coder",
                        background=False,
                        completion_check=[CompletionCheckTemplate.PYTEST],
                        required_artifacts=[CompletionArtifactKind.COMMIT],
                    ),
                    ctx,
                )
            ][-1],
        )
        assert result.completed
        # The foreground admission captured the HEAD baseline (the contract
        # requires a commit artifact).
        assert len(probe_calls) == 1
        prompt = _child_prompt(child_backend)
        assert "Foreground work" in prompt
        assert "Completion contract (frozen at admission):" in prompt
        assert "check:pytest" in prompt
        assert "artifact:commit" in prompt
    finally:
        await registry.drain_children()
        await parent.aclose()


# ----------------------------------------------------------------------
# Resume: the run-scoped contract snapshot, rendered into the resume prompt
# ----------------------------------------------------------------------


class _ScriptedBackend(FakeBackend):
    """One outcome per logical call: chunks, or a fault raised on every call."""

    def __init__(self, outcome: list[LLMChunk] | Exception | None) -> None:
        super().__init__([])
        self._outcome = outcome

    async def complete(self, **kwargs: Any) -> LLMChunk:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        chunk_agg = LLMChunk(message=LLMMessage(role=Role.assistant))
        for chunk in self._outcome or []:
            chunk_agg += chunk
        return chunk_agg

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        for chunk in self._outcome or []:
            yield chunk


class _OutcomeFactory:
    """Scripted child backends: one outcome per logical call, in order.

    Construction and resume-time refresh backends (no ``retry_budget``) are
    throwaway; every logical call draws the next scripted outcome.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outcomes: list[list[LLMChunk] | Exception],
    ) -> None:
        self._outcomes = list(outcomes)
        monkeypatch.setattr(
            "chartreux.core.agent_loop._loop.create_backend", self._create_backend
        )

    def _create_backend(self, **kwargs: Any) -> FakeBackend:
        if "retry_budget" in kwargs:
            return _ScriptedBackend(self._outcomes.pop(0))
        return _ScriptedBackend(None)


def _todo_write_call(call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="todo",
            arguments=json.dumps({
                "action": "write",
                "todos": [{"id": "1", "content": "item", "status": "pending"}],
            }),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_resume_carries_the_run_scoped_contract_snapshot_and_renders_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _OutcomeFactory(
        monkeypatch,
        outcomes=[
            # A settled tool effect before the fault: the envelope's tool
            # effects are SETTLED, so the failure is recovery-ELIGIBLE.
            [mock_llm_chunk(content="", tool_calls=[_todo_write_call("todo-1")])],
            EmptyLLMResponseError("test-provider", "test-model"),
            [mock_llm_chunk(content="resumed done")],
        ],
    )
    parent = build_test_agent_loop(
        config=_config(enabled_tools=["task", "todo"]), backend=_NotedBackend()
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    probe_calls: list[Path] = []

    def probe(cwd: Path) -> str | None:
        probe_calls.append(cwd)
        return "a" * 40

    monkeypatch.setattr(
        "chartreux.app_server._subagent_execution._capture_admission_head", probe
    )
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(
            registry,
            TaskArgs(
                task="Mandated assignment",
                identity="coder",
                background=True,
                retain=True,
                completion_check=[CompletionCheckTemplate.PYTEST],
                required_artifacts=[CompletionArtifactKind.COMMIT],
            ),
            ctx,
        )
        assert ack.agent_id is not None and ack.run_id is not None
        failed = await _wait_result(registry, ack.agent_id, ack.run_id)
        assert not failed.completed
        record = registry._agent_records[ack.agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)
        failed_run = _find_run(record, ack.run_id)
        failed_contract = failed_run.completion_contract
        assert failed_contract is not None
        assert _requirement_ids(failed_contract) == ["check:pytest", "artifact:commit"]
        assert failed_run.admission_head == "a" * 40
        assert len(probe_calls) == 1

        # Capture the resume turn's start params.
        captured: list[TurnStartParams] = []
        original_start = record.runtime.turns.start

        def capture_start(params: TurnStartParams) -> Any:
            captured.append(params)
            return original_start(params)

        monkeypatch.setattr(record.runtime.turns, "start", capture_start)

        resume = await registry.resume_agent(
            ack.agent_id,
            ack.run_id,
            f"{ack.agent_id}:{ack.run_id}",
            requester_session_id=parent.session_id,
        )
        assert resume.outcome is ResumeOutcome.SCHEDULED
        assert resume.successor_run_id is not None
        successor = await _wait_result(registry, ack.agent_id, resume.successor_run_id)
        assert successor.completed
        assert successor.response == "resumed done"

        successor_run = _find_run(record, resume.successor_run_id)
        # Resume = run-scoped snapshot: the successor keeps the failed run's
        # frozen contract and admission HEAD baseline — the reduced resume
        # args carry no mandates, and no HEAD probe runs again.
        assert successor_run.completion_contract is failed_contract
        assert successor_run.admission_head == "a" * 40
        assert successor_run.resumed_from_run_id == ack.run_id
        assert len(probe_calls) == 1
        # The frozen contract renders into the resumed prompt too.
        assert len(captured) == 1
        assert captured[0].injected is True
        resume_text = next(
            block.text
            for block in captured[0].message
            if isinstance(block, TextContentBlock)
        )
        assert "This is a resumed run" in resume_text
        assert "Completion contract (frozen at admission):" in resume_text
        assert "check:pytest" in resume_text
        assert "artifact:commit" in resume_text
    finally:
        await registry.drain_children()
        await parent.aclose()
