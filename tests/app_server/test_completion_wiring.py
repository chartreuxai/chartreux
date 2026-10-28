"""The enforce-activation wiring: the two previously inert production surfaces.

Part 1 of the enforce-activation package wires the composition's two inert
inputs into the production path:

- the parent-shared roots (defect E): the frozen contract carries the roots
  shared with the parent (the session scratchpad, derived at admission from
  the child runtime's inherited scratch authority), and ``build_result``
  threads them into the composition — a failed finalization check observing
  final state under a shared root composes UNVERIFIED, never CONTRADICTED,
  never blocked, where pre-wiring it falsely rejected an honest worker in
  enforce mode;
- the claimed-value surface (P1/P7/C1): the receipt grammar carries the
  bounded ``claimed_values`` object (claimed commit hashes, claimed artifact
  paths), the last-marker scan recognizes it, and ``build_result`` derives it
  from the claim so the artifact audit receives it — an honest artifact claim
  resolves claim_matched through the artifact path, a fabricated hash
  resolves unverified per the truth table, and a claim without values stays
  unverified, never a fault.

No live-provider calls: the check spawn seam is monkeypatched; the child
backends are fakes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server._subagent_execution import (
    _parent_shared_roots,
    extract_completion_claim,
)
from chartreux.app_server._subagent_finalization import RunFinalization
from chartreux.app_server._subagent_records import (
    CompletionCapability,
    FrozenCompletionContract,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.completion_checks import (
    CompletionCheckOutcome,
    CompletionCheckStatus,
    CompletionCheckTemplate,
)
from chartreux.core.completion_contract import (
    COMPLETION_CONTRACT_SCHEMA_VERSION,
    COMPLETION_ENVELOPE_MARKER,
    CompletionArtifactKind,
    CompletionClaimFidelity,
    CompletionEnvelope,
    CompletionReceipt,
    CompletionRejectionReason,
    CompletionRequirement,
)
from chartreux.core.config.models import CompletionSectionConfig
from chartreux.core.events import AssistantEvent, ToolResultEvent
from chartreux.core.llm_models import FunctionCall, LLMChunk, ToolCall
from chartreux.core.recovery_models import PackageTier
from chartreux.core.subagents import RunStatus, SubagentRunAccumulator, TaskArgs
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.builtins.bash import Bash, CapturedShellResult
from chartreux.core.tools.builtins.write_file import WriteFile, WriteFileResult
from tests.app_server._subagent_test_support import (
    _background_result,
    _config,
    _retention_registry,
    _wait_result,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.timeout(60)

AuditMode = Literal["stamp", "enforce"]

CANONICAL_COMMAND = "pytest tests/test_x.py"
RUNNER_ARGV = ["pytest", "tests/test_x.py"]
FULL_HASH = "abc1234" + "0" * 33


# ----------------------------------------------------------------------
# Shared fixtures (the registry harness surface, re-derived here)
# ----------------------------------------------------------------------


class _NotedBackend(FakeBackend):
    """A parent backend that answers every turn with terminal content."""

    async def complete(self, **kwargs: Any) -> LLMChunk:
        return mock_llm_chunk(content="parent noted")

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        yield mock_llm_chunk(content="parent noted")


def _config_with_mode(mode: AuditMode, *, write: bool = False) -> Any:
    """The parent config granting bash (and write_file), with the audit mode."""
    tools = ["task", "bash"] + (["write_file"] if write else [])
    config = _config(enabled_tools=tools)
    return config.model_copy(update={"completion": CompletionSectionConfig(mode=mode)})


async def _registry_with_parent(
    monkeypatch: pytest.MonkeyPatch,
    backend: FakeBackend,
    *,
    mode: AuditMode = "enforce",
) -> tuple[SessionRuntimeRegistry, AgentLoop]:
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(
        config=_config_with_mode(mode, write=True), backend=_NotedBackend()
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


def _record_finalize_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the monitor's finalization calls (with their accumulators)."""
    import chartreux.app_server._subagent_execution as execution

    calls: list[dict[str, Any]] = []
    real = execution.finalize_background_run

    async def _recording(**kwargs: Any) -> RunFinalization:
        result = await real(**kwargs)
        calls.append({**kwargs, "result": result})
        return result

    monkeypatch.setattr(execution, "finalize_background_run", _recording)
    return calls


def _envelope_payload(*requirements: dict[str, Any]) -> dict[str, Any]:
    return {
        "marker": COMPLETION_ENVELOPE_MARKER,
        "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
        "requirements": list(requirements),
    }


def _receipt_payload(
    requirements: list[dict[str, Any]],
    satisfied: list[str],
    claimed_values: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "marker": COMPLETION_ENVELOPE_MARKER,
        "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
        "envelope": _envelope_payload(*requirements),
        "satisfied_requirement_ids": satisfied,
    }
    if claimed_values is not None:
        payload["claimed_values"] = claimed_values
    return payload


def _check_requirement(requirement_id: str = "check:pytest") -> dict[str, Any]:
    return {"requirement_id": requirement_id, "template": "pytest"}


def _commit_requirement() -> dict[str, Any]:
    return {"requirement_id": "artifact:commit", "artifact": "commit"}


def _runner() -> Any:
    return SimpleNamespace(check_argv=lambda template: list(RUNNER_ARGV))


def _observe_write(
    accumulator: SubagentRunAccumulator, path: Path, occurrence_id: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "payload\n"
    path.write_text(content, encoding="utf-8")
    accumulator.observe(AssistantEvent(content="done"), tool_call_id="task")
    accumulator.observe(
        ToolResultEvent(
            tool_name="write_file",
            tool_class=WriteFile,
            result=WriteFileResult(
                file_path=str(path),
                bytes_written=len(content.encode()),
                content=content,
            ),
            tool_call_id="call-1",
            occurrence_id=occurrence_id,
        ),
        tool_call_id="call-1",
    )


def _commit_contract(
    tmp_path: Path, *, audit_mode: str = "enforce"
) -> FrozenCompletionContract:
    return FrozenCompletionContract.freeze(
        identity="coder",
        identity_revision="rev-1",
        seat=None,
        tier=PackageTier.SUBSTANTIVE,
        cwd=str(tmp_path),
        roots=(str(tmp_path),),
        completion_check=None,
        required_artifacts=[CompletionArtifactKind.COMMIT],
        shell_less=False,
        audit_mode=audit_mode,
        capability=CompletionCapability.AVAILABLE,
    )


def _publish_claim(
    accumulator: SubagentRunAccumulator,
    contract: Any,
    report: str,
    *,
    run_id: str = "run-1",
    finalization: RunFinalization | None = None,
) -> Any:
    """Publish one run through the production ``build_result`` call shape."""
    if finalization is not None:
        accumulator._finalization_cache[run_id] = finalization
    accumulator.observe(AssistantEvent(content=report), tool_call_id="final-report")
    claim = extract_completion_claim("".join(accumulator._response))
    return accumulator.build_result(
        turns_used=1,
        completed=True,
        run_id=run_id,
        contract=contract,
        claim=claim,
        runner=_runner(),
    )


def _observe_git_commit(accumulator: SubagentRunAccumulator, stdout: str) -> None:
    """Observe one git-command result through the real collection path."""
    accumulator.observe(
        ToolResultEvent(
            tool_name="bash",
            tool_class=Bash,
            result=CapturedShellResult(command='git commit -m "work"', stdout=stdout),
            tool_call_id="call-1",
            occurrence_id="occ-1",
        ),
        tool_call_id="call-1",
    )


# ----------------------------------------------------------------------
# The parent-shared roots: admission derivation and the dispatched pin
# ----------------------------------------------------------------------


def test_parent_shared_roots_empty_without_scratch_authority() -> None:
    # A loop that cannot report its scratch authority freezes no shared
    # root: external writes are not established as possible.
    assert _parent_shared_roots(SimpleNamespace()) == ()

    class _Broken:
        @property
        def tool_manager(self) -> Any:
            raise RuntimeError("no tool manager")

    assert _parent_shared_roots(_Broken()) == ()

    loop = SimpleNamespace(
        tool_manager=SimpleNamespace(scratchpad_roots=frozenset({Path("/tmp/scratch")}))
    )
    assert _parent_shared_roots(loop) == ("/tmp/scratch",)


@pytest.mark.asyncio
async def test_freeze_run_admission_freezes_the_parent_shared_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission derives the shared roots from the child's inherited scratch
    authority: the session scratchpad the parent and the child both write.
    """
    child_backend = FakeBackend([mock_llm_chunk(content="child done")])
    registry, parent = await _registry_with_parent(monkeypatch, child_backend)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(
            registry,
            TaskArgs(task="Work", identity="coder", background=True, retain=True),
            ctx,
        )
        assert ack.agent_id is not None and ack.run_id is not None
        await registry.wait_for_agent(ack.agent_id, ack.run_id)
        record = registry._agent_records[ack.agent_id]
        run = _find_run(record, ack.run_id)
        contract = run.completion_contract
        assert contract is not None
        scratchpad = parent.scratchpad_dir
        assert scratchpad is not None
        assert contract.shared_roots == (str(scratchpad),)
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_dispatched_child_shared_root_edit_composes_unverified_never_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect-E regression pin, through a real dispatched child run.

    The child writes a shared-root file (the session scratchpad), the parent
    edits it between the worker's report and the finalization, and the
    finalization's bound check FAILS over that final state. Pre-wiring (no
    shared roots) the failed check composed CONTRADICTED and enforce mode
    falsely rejected the honest worker; wired, the check is check-context
    evidence: the composition is UNVERIFIED — never contradicted, never
    blocked — and the run publishes.
    """
    registry, parent = await _registry_with_parent(
        monkeypatch, FakeBackend(), mode="enforce"
    )
    scratchpad = parent.scratchpad_dir
    assert scratchpad is not None
    artifact = scratchpad / "shared-artifact.txt"
    write_call = ToolCall(
        id="write-1",
        index=0,
        function=FunctionCall(
            name="write_file",
            arguments=json.dumps({
                "file_path": str(artifact),
                "content": "worker payload\n",
            }),
        ),
    )
    child_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[write_call])],
        [
            mock_llm_chunk(
                content="Work complete.\n"
                + json.dumps(_receipt_payload([_check_requirement()], ["check:pytest"]))
            )
        ],
    ])
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    # The finalization's bound check FAILS over the final state.
    spawn = _SpawnRecorder(returncode=1)
    _install_spawn(monkeypatch, spawn)
    calls = _record_finalize_calls(monkeypatch)
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
        # The parent edits the shared-root file between the worker's report
        # and the finalization (the interleaving the guard exists for).
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 10
        while not artifact.exists():
            assert loop.time() < deadline, "the child never wrote the shared artifact"
            await asyncio.sleep(0.01)
        artifact.write_text("worker payload\nparent edit\n", encoding="utf-8")
        result = await _wait_result(registry, ack.agent_id, ack.run_id)
        # The honest worker is never blocked: the run publishes completed.
        assert result.completed is True
        assert "Work complete." in result.response
        assert "[Completion audit rejected:" not in result.response
        # The composition is UNVERIFIED — never contradicted.
        audit = result.completion_audit
        assert audit is not None
        assert audit.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
        assert audit.rejection is None
        # The run record agrees: COMPLETED, no failure envelope.
        record = registry._agent_records[ack.agent_id]
        run = _find_run(record, ack.run_id)
        assert run.status is RunStatus.COMPLETED
        assert run.failure is None
        contract = run.completion_contract
        assert contract is not None
        assert contract.shared_roots == (str(scratchpad),)
        # The finalization's check really failed, and the run's collected
        # touched path sits under the wired shared root.
        assert len(calls) == 1
        finalization = calls[0]["result"]
        assert isinstance(finalization, RunFinalization)
        assert len(finalization.checks) == 1
        assert finalization.checks[0].status is CompletionCheckStatus.NONZERO
        accumulator = calls[0]["accumulator"]
        assert str(artifact) in accumulator._touched_paths
    finally:
        await registry.drain_children()
        await parent.aclose()


# ----------------------------------------------------------------------
# build_result: the shared roots and the claimed values ride the claim
# ----------------------------------------------------------------------


def _failed_finalization() -> Any:
    return SimpleNamespace(
        checks=(
            CompletionCheckOutcome(
                template=CompletionCheckTemplate.PYTEST,
                status=CompletionCheckStatus.NONZERO,
                argv=tuple(RUNNER_ARGV),
                exit_code=1,
            ),
        ),
        admission_head=None,
        publication_head=None,
        runtime_failure=None,
    )


def _check_envelope() -> CompletionEnvelope:
    return CompletionEnvelope(
        marker=COMPLETION_ENVELOPE_MARKER,
        schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
        requirements=(
            CompletionRequirement(
                requirement_id="check:pytest", template=CompletionCheckTemplate.PYTEST
            ),
        ),
    )


def test_build_result_reads_the_contract_shared_roots(tmp_path: Path) -> None:
    """build_result derives the shared roots from the contract: the failed
    finalization check over the shared root composes UNVERIFIED and the
    enforce-mode run publishes; without the wired roots the same shape
    composes CONTRADICTED and blocks (the pre-wiring false rejection).
    """
    envelope = _check_envelope()
    report = "Work complete.\n" + json.dumps(
        _receipt_payload([_check_requirement()], ["check:pytest"])
    )

    wired = SubagentRunAccumulator()
    _observe_write(wired, tmp_path / "out.txt", "occ-1")
    wired._finalization_cache["run-wired"] = _failed_finalization()
    wired_result = _publish_claim(
        wired,
        SimpleNamespace(
            envelope=envelope, audit_mode="enforce", shared_roots=(str(tmp_path),)
        ),
        report,
        run_id="run-wired",
    )
    assert wired_result.completed is True
    assert "[Completion audit rejected:" not in wired_result.response
    wired_audit = wired_result.completion_audit
    assert wired_audit is not None
    assert wired_audit.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert wired_audit.rejection is None

    # The pre-wiring shape: no shared roots on the contract — the failed
    # check keeps its full force and enforce mode blocks the submission.
    unwired = SubagentRunAccumulator()
    _observe_write(unwired, tmp_path / "out.txt", "occ-1")
    unwired._finalization_cache["run-unwired"] = _failed_finalization()
    unwired_result = _publish_claim(
        unwired,
        SimpleNamespace(envelope=envelope, audit_mode="enforce"),
        report,
        run_id="run-unwired",
    )
    assert unwired_result.completed is False
    assert unwired_result.response.startswith("[Completion audit rejected:")
    unwired_audit = unwired_result.completion_audit
    assert unwired_audit is not None
    assert unwired_audit.rejection is CompletionRejectionReason.CLAIM_CONTRADICTED


# ----------------------------------------------------------------------
# The claimed values through the production path
# ----------------------------------------------------------------------


def test_extract_completion_claim_recognizes_the_claimed_values_receipt() -> None:
    report = "Work complete.\n" + json.dumps(
        _receipt_payload(
            [_commit_requirement()], ["artifact:commit"], {"artifact:commit": FULL_HASH}
        )
    )
    claim = extract_completion_claim(report)
    assert isinstance(claim, CompletionReceipt)
    assert claim.claimed_values == {"artifact:commit": FULL_HASH}

    # A receipt carrying only the claimed-value surface is still a claim.
    values_only = json.dumps(
        _receipt_payload([_commit_requirement()], [], {"artifact:commit": FULL_HASH})
    )
    claim = extract_completion_claim(values_only)
    assert isinstance(claim, CompletionReceipt)
    assert claim.satisfied_requirement_ids == ()
    assert claim.claimed_values == {"artifact:commit": FULL_HASH}

    # The nested envelope (no claim key) is not a claim.
    envelope_only = json.dumps(_envelope_payload(_commit_requirement()))
    assert extract_completion_claim(envelope_only) is None

    # A grammar-violating claimed-value surface rides the raw payload.
    bad = json.dumps(
        _receipt_payload(
            [_commit_requirement()], ["artifact:commit"], {"artifact:other": FULL_HASH}
        )
    )
    claim = extract_completion_claim(bad)
    assert isinstance(claim, dict)
    assert claim["claimed_values"] == {"artifact:other": FULL_HASH}


def test_worker_claiming_a_genuine_commit_hash_resolves_claim_matched(
    tmp_path: Path,
) -> None:
    """A worker emitting a genuine commit hash for a commit it made resolves
    claim_matched through the artifact path (the changed-HEAD commit proof),
    and the enforce-mode run publishes.
    """
    contract = _commit_contract(tmp_path)
    accumulator = SubagentRunAccumulator()
    _observe_git_commit(accumulator, f"[main {FULL_HASH}] work\n")
    report = "Work complete.\n" + json.dumps(
        _receipt_payload(
            [_commit_requirement()],
            ["artifact:commit"],
            {"artifact:commit": FULL_HASH[:7]},
        )
    )
    result = _publish_claim(
        accumulator,
        contract,
        report,
        finalization=RunFinalization(
            run_id="run-1", admission_head="0" * 40, publication_head=FULL_HASH
        ),
    )
    assert result.completed is True
    audit = result.completion_audit
    assert audit is not None
    assert audit.claim_fidelity is CompletionClaimFidelity.CLAIM_MATCHED
    assert audit.rejection is None


def test_fabricated_commit_hash_resolves_unverified_never_a_fault(
    tmp_path: Path,
) -> None:
    """A fabricated hash resolves per the truth table: the artifact audit's
    vocabulary has no contradicted member, so the claim is unverified
    (check-context) — never a fault, never blocked in enforce mode.
    """
    contract = _commit_contract(tmp_path)
    accumulator = SubagentRunAccumulator()
    _observe_git_commit(accumulator, f"[main {FULL_HASH}] work\n")
    report = "Work complete.\n" + json.dumps(
        _receipt_payload(
            [_commit_requirement()], ["artifact:commit"], {"artifact:commit": "f" * 40}
        )
    )
    result = _publish_claim(
        accumulator,
        contract,
        report,
        finalization=RunFinalization(
            run_id="run-1", admission_head="0" * 40, publication_head="e" * 40
        ),
    )
    assert result.completed is True
    assert "[Completion audit rejected:" not in result.response
    audit = result.completion_audit
    assert audit is not None
    assert audit.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert audit.rejection is None


def test_claim_without_claimed_values_stays_unverified_never_a_fault(
    tmp_path: Path,
) -> None:
    """Fail-open: a claim without values resolves "no commit hash to check"
    (check-context) — unverified, never a fault, never blocked.
    """
    contract = _commit_contract(tmp_path)
    accumulator = SubagentRunAccumulator()
    _observe_git_commit(accumulator, f"[main {FULL_HASH}] work\n")
    report = "Work complete.\n" + json.dumps(
        _receipt_payload([_commit_requirement()], ["artifact:commit"])
    )
    result = _publish_claim(
        accumulator,
        contract,
        report,
        finalization=RunFinalization(
            run_id="run-1", admission_head="0" * 40, publication_head=FULL_HASH
        ),
    )
    assert result.completed is True
    audit = result.completion_audit
    assert audit is not None
    assert audit.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert audit.rejection is None
