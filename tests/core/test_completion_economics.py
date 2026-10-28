"""P16: the completion-audit economics at the publication boundary.

The cost invariants of the audit path, pinned where they are paid:

- ZERO LLM CALLS — the composition, the once-only finalization, and the
  production ``build_result`` construct no backend (the seam is rigged);
- AT MOST ONE ACCEPTANCE ATTEMPT per run — the run-keyed finalization
  cache: a repeated finalization await, a repeated result build, and a
  repeated retrieval run no check twice;
- AT MOST TWO HEAD SNAPSHOTS per run, and only when the frozen contract
  mandates a commit artifact — the admission baseline (P4) and the
  publication snapshot (P6a); a check-only or mandate-free run probes
  nothing;
- BOUNDED PAYLOADS — the envelope, receipt, artifact-snapshot, chronology,
  block-message, and check-capture caps are pinned against silent growth
  and exercised at their boundaries;
- TIMEOUT PATHS NEVER RETRY — one spawn per acceptance attempt, ever;
- ZERO HONEST-REPORT FALSE REJECTIONS under enforcement — honest claims
  with matching evidence, honest reports without a receipt, honest
  aborts, honest red/green retries, and exempt identities all publish.

All evidence is constructed in-process through the real collection and
publication path (the accumulator's observe, the P2 side-channel receipts,
the run-keyed finalization cache, the real claim extraction, the frozen
contract); no subprocess runs and no check executes — the spawn seam is
recorded. The end-to-end pins drive the real background publication path
through the session registry with fake backends.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterable
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError
import pytest

from chartreux.app_server._subagent_execution import (
    _capture_admission_head,
    extract_completion_claim,
)
from chartreux.app_server._subagent_finalization import (
    RunFinalization,
    finalize_background_run,
)
from chartreux.app_server._subagent_records import (
    CompletionCapability,
    FrozenCompletionContract,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop_hooks import CompletionEvidenceReceipt
from chartreux.core.completion_checks import (
    MAX_CAPTURE_BYTES,
    CompletionCheckOutcome,
    CompletionCheckRunner,
    CompletionCheckStatus,
    CompletionCheckTemplate,
)
from chartreux.core.completion_contract import (
    COMPLETION_CONTRACT_SCHEMA_VERSION,
    COMPLETION_ENVELOPE_MARKER,
    MAX_ENVELOPE_REQUIREMENTS,
    MAX_SATISFIED_REQUIREMENTS,
    CompletionArtifactKind,
    CompletionClaimFidelity,
    CompletionEnvelope,
    CompletionReceipt,
    CompletionRejectionReason,
    CompletionRequirement,
)
from chartreux.core.completion_evidence import (
    ARTIFACT_SNAPSHOT_CAP_BYTES,
    MAX_CHRONOLOGY_ENTRIES,
    EvidenceAvailability,
)
from chartreux.core.config.models import CompletionSectionConfig
from chartreux.core.events import AssistantEvent, ToolResultEvent
from chartreux.core.git.errors import GitError
from chartreux.core.git.repo import GitRepo
from chartreux.core.hooks.models import ToolStatus
from chartreux.core.llm_models import LLMChunk
from chartreux.core.recovery_models import PackageTier
from chartreux.core.subagent_submission import (
    BLOCK_MESSAGE_MAX_PATHS,
    blocked_submission_response,
)
from chartreux.core.subagents import SubagentRunAccumulator, TaskArgs, TaskResult
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.builtins.bash import Bash, CapturedShellResult
from chartreux.core.tools.builtins.write_file import WriteFile, WriteFileResult
from chartreux.core.tools.secret_scan import SecretFinding
from chartreux.core.workspace import Workspace
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

#: The canonical pytest acceptance-check argv (the P5 template table).
PYTEST_ARGV = ("uv", "run", "pytest", "-q")
PYTEST_COMMAND = "uv run pytest -q"


# ----------------------------------------------------------------------
# Shared helpers (the real collection and publication path, in-process)
# ----------------------------------------------------------------------


def _runner() -> Any:
    """The acceptance-check runner stub: the canonical argv table only."""
    return SimpleNamespace(check_argv=lambda template: list(PYTEST_ARGV))


def _freeze(
    workspace: Path, *, mode: AuditMode = "enforce", **overrides: Any
) -> FrozenCompletionContract:
    """Freeze a coder contract over the given workspace."""
    params: dict[str, Any] = {
        "identity": "coder",
        "identity_revision": "rev-1",
        "seat": None,
        "tier": PackageTier.SUBSTANTIVE,
        "cwd": str(workspace),
        "roots": (str(workspace),),
        "completion_check": None,
        "required_artifacts": None,
        "shell_less": False,
        "audit_mode": mode,
        "capability": CompletionCapability.AVAILABLE,
    }
    params.update(overrides)
    return FrozenCompletionContract.freeze(**params)


def _observe_write(
    accumulator: SubagentRunAccumulator,
    path: Path,
    occurrence_id: str,
    *,
    content: str = "payload\n",
) -> None:
    """Observe one write_file result through the real collection path."""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _emit_receipt(
    accumulator: SubagentRunAccumulator,
    occurrence_id: str,
    command: str,
    *,
    status: ToolStatus = "success",
    final_text: str = "ok",
) -> None:
    """Emit one side-channel receipt through the accumulator's sink."""
    accumulator.completion_evidence_sink()(
        CompletionEvidenceReceipt(
            occurrence_id=occurrence_id,
            tool_call_id="call-1",
            tool_name="write_file",
            status=status,
            final_text=final_text,
            executed_args={"command": command},
        )
    )


def _observe_bash(
    accumulator: SubagentRunAccumulator,
    command: str,
    exit_code: int,
    stdout: str,
    occurrence_id: str,
) -> None:
    """Observe one bash result and emit its P2 side-channel receipt."""
    accumulator.observe(
        ToolResultEvent(
            tool_name="bash",
            tool_class=Bash,
            result=CapturedShellResult(
                command=command, exit_code=exit_code, stdout=stdout, write_paths=[]
            ),
            tool_call_id="call-bash",
            occurrence_id=occurrence_id,
        ),
        tool_call_id="call-bash",
    )
    accumulator.completion_evidence_sink()(
        CompletionEvidenceReceipt(
            occurrence_id=occurrence_id,
            tool_call_id="call-bash",
            tool_name="bash",
            status="success" if exit_code == 0 else "failure",
            final_text=stdout[:500],
            executed_args={"command": command},
        )
    )


def _claim_payload(
    contract: FrozenCompletionContract, satisfied: list[str]
) -> dict[str, Any]:
    """The contract-era receipt payload the worker's report would carry."""
    payload: dict[str, Any] = {
        "marker": COMPLETION_ENVELOPE_MARKER,
        "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
        "satisfied_requirement_ids": satisfied,
    }
    if contract.envelope is not None:
        payload["envelope"] = contract.envelope.model_dump(mode="json")
    return payload


def _passed_check() -> CompletionCheckOutcome:
    return CompletionCheckOutcome(
        template=CompletionCheckTemplate.PYTEST,
        status=CompletionCheckStatus.PASSED,
        argv=PYTEST_ARGV,
        exit_code=0,
    )


def _finalization(
    run_id: str,
    *,
    checks: Iterable[CompletionCheckOutcome] = (),
    admission_head: str | None = None,
    publication_head: str | None = None,
    runtime_failure: str | None = None,
) -> RunFinalization:
    return RunFinalization(
        run_id=run_id,
        checks=tuple(checks),
        admission_head=admission_head,
        publication_head=publication_head,
        runtime_failure=runtime_failure,
    )


def _publish(
    accumulator: SubagentRunAccumulator,
    *,
    contract: FrozenCompletionContract,
    report: str,
    run_id: str = "run-1",
    declared_completed: bool = True,
    finalization: RunFinalization | None = None,
    mode: AuditMode = "enforce",
) -> TaskResult:
    """Publish one synthetic run through the production ``build_result``."""
    if finalization is not None:
        accumulator._finalization_cache[run_id] = finalization
    accumulator.observe(AssistantEvent(content=report), tool_call_id="final-report")
    # The P12 call-site shape: the claim is operative only on a healthy
    # completed turn.
    claim = (
        extract_completion_claim("".join(accumulator._response))
        if declared_completed
        else None
    )
    if not declared_completed:
        accumulator.record_error("run cancelled (user_cancelled)")
    return accumulator.build_result(
        turns_used=1,
        completed=declared_completed,
        run_id=run_id,
        contract=contract,
        claim=claim,
        runner=_runner(),
        mode=mode,
    )


class _SpawnRecorder:
    """The monkeypatched check spawn seam: records commands, fakes processes."""

    def __init__(self, *, returncode: int = 0, stdout: bytes = b"ok\n") -> None:
        self.commands: list[str] = []
        self.returncode = returncode
        self.stdout = stdout

    async def __call__(self, command: str, **kwargs: Any) -> Any:
        self.commands.append(command)
        returncode = self.returncode
        stdout = self.stdout

        class _Process:
            async def communicate(self) -> tuple[bytes, bytes]:
                return (stdout, b"")

            @property
            def returncode(self) -> int:
                return returncode

        return _Process()


def _install_spawn(monkeypatch: pytest.MonkeyPatch, recorder: _SpawnRecorder) -> None:
    monkeypatch.setattr(
        "chartreux.core.completion_checks.spawn_shell_command", recorder
    )


def _check_runner(cwd: Path, *, timeout: float = 120.0) -> CompletionCheckRunner:
    """A runner whose workspace and path authority always allow."""
    return CompletionCheckRunner(
        cwd=cwd,
        workspace=SimpleNamespace(allows=lambda target: True),
        path_authority=SimpleNamespace(resolve=lambda path, access: None),
        timeout=timeout,
    )


def _runtime(cwd: Path) -> Any:
    """The child-runtime surface the finalization's runner is built from."""
    return SimpleNamespace(
        agent_loop=SimpleNamespace(
            cwd=cwd,
            tool_manager=SimpleNamespace(
                workspace=Workspace.for_session(cwd),
                path_authority=lambda name: SimpleNamespace(
                    resolve=lambda path, access: None
                ),
            ),
        )
    )


def _probe(probes: list[Path]) -> Any:
    """A ``GitRepo.head_commit_at`` replacement that records its target."""

    def _head(target: Path) -> str:
        probes.append(target)
        return "c" * 40

    return _head


# ----------------------------------------------------------------------
# Zero LLM calls
# ----------------------------------------------------------------------


class TestZeroLLMCalls:
    """The audit path constructs no backend — zero LLM calls."""

    @pytest.mark.asyncio
    async def test_composition_finalization_and_publication_construct_no_backend(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[dict[str, Any]] = []

        def _rig(**kwargs: Any) -> Any:
            calls.append(kwargs)
            return FakeBackend()

        monkeypatch.setattr("chartreux.core.agent_loop._loop.create_backend", _rig)
        spawn = _SpawnRecorder()
        _install_spawn(monkeypatch, spawn)
        contract = _freeze(tmp_path, completion_check=[CompletionCheckTemplate.PYTEST])
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, tmp_path / "out.txt", "occ-1")
        _emit_receipt(accumulator, "occ-1", PYTEST_COMMAND)
        await finalize_background_run(
            run_id="run-1",
            accumulator=accumulator,
            contract=contract,
            admission_head=None,
            runtime=_runtime(tmp_path),
        )
        result = _publish(
            accumulator,
            contract=contract,
            report="Work complete.\n"
            + json.dumps(_claim_payload(contract, ["check:pytest"])),
        )
        assert result.completed is True
        assert calls == []


# ----------------------------------------------------------------------
# At most one acceptance attempt per run
# ----------------------------------------------------------------------


class TestOneAcceptanceAttemptPerRun:
    """The run-keyed finalization cache: one acceptance attempt, ever."""

    @pytest.mark.asyncio
    async def test_finalization_is_cached_per_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        spawn = _SpawnRecorder()
        _install_spawn(monkeypatch, spawn)
        contract = _freeze(tmp_path, completion_check=[CompletionCheckTemplate.PYTEST])
        accumulator = SubagentRunAccumulator()
        first = await finalize_background_run(
            run_id="run-1",
            accumulator=accumulator,
            contract=contract,
            admission_head=None,
            runtime=_runtime(tmp_path),
        )
        second = await finalize_background_run(
            run_id="run-1",
            accumulator=accumulator,
            contract=contract,
            admission_head=None,
            runtime=_runtime(tmp_path),
        )
        # The second await is a cache hit: the same object, no check re-run.
        assert second is first
        assert len(spawn.commands) == 1
        assert first.checks[0].status is CompletionCheckStatus.PASSED
        # The cache is run-keyed: a different run finalizes again.
        other = await finalize_background_run(
            run_id="run-2",
            accumulator=accumulator,
            contract=contract,
            admission_head=None,
            runtime=_runtime(tmp_path),
        )
        assert other is not first
        assert len(spawn.commands) == 2

    @pytest.mark.asyncio
    async def test_repeated_builds_consume_the_cached_finalization(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        spawn = _SpawnRecorder()
        _install_spawn(monkeypatch, spawn)
        contract = _freeze(tmp_path, completion_check=[CompletionCheckTemplate.PYTEST])
        accumulator = SubagentRunAccumulator()
        await finalize_background_run(
            run_id="run-1",
            accumulator=accumulator,
            contract=contract,
            admission_head=None,
            runtime=_runtime(tmp_path),
        )
        report = "All done, nothing to claim."
        first = _publish(accumulator, contract=contract, report=report)
        second = _publish(accumulator, contract=contract, report=report)
        assert second.completion_audit == first.completion_audit
        assert second.completed is True
        # The repeated build ran no check: the one spawn is the
        # finalization's, from before the first build.
        assert len(spawn.commands) == 1


# ----------------------------------------------------------------------
# At most two HEAD snapshots, only when commit proof is mandated
# ----------------------------------------------------------------------


class TestHeadSnapshotBudget:
    """HEAD probes: the admission baseline + the publication snapshot."""

    @pytest.mark.asyncio
    async def test_check_only_contract_probes_no_head(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        probes: list[Path] = []
        monkeypatch.setattr(GitRepo, "head_commit_at", _probe(probes))
        spawn = _SpawnRecorder()
        _install_spawn(monkeypatch, spawn)
        contract = _freeze(tmp_path, completion_check=[CompletionCheckTemplate.PYTEST])
        finalization = await finalize_background_run(
            run_id="run-1",
            accumulator=SubagentRunAccumulator(),
            contract=contract,
            admission_head=None,
            runtime=_runtime(tmp_path),
        )
        assert probes == []
        assert finalization.publication_head is None
        assert len(spawn.commands) == 1

    @pytest.mark.asyncio
    async def test_commit_mandate_probes_once_at_finalization(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        probes: list[Path] = []
        monkeypatch.setattr(GitRepo, "head_commit_at", _probe(probes))
        spawn = _SpawnRecorder()
        _install_spawn(monkeypatch, spawn)
        contract = _freeze(tmp_path, required_artifacts=[CompletionArtifactKind.COMMIT])
        finalization = await finalize_background_run(
            run_id="run-1",
            accumulator=SubagentRunAccumulator(),
            contract=contract,
            admission_head="a" * 40,
            runtime=_runtime(tmp_path),
        )
        # Exactly one probe at finalization; the admission baseline passes
        # through unchanged. No check templates bound: no spawn.
        assert len(probes) == 1
        assert finalization.publication_head == "c" * 40
        assert finalization.admission_head == "a" * 40
        assert finalization.checks == ()
        assert spawn.commands == []

    @pytest.mark.asyncio
    async def test_no_contract_probes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        probes: list[Path] = []
        monkeypatch.setattr(GitRepo, "head_commit_at", _probe(probes))
        spawn = _SpawnRecorder()
        _install_spawn(monkeypatch, spawn)
        finalization = await finalize_background_run(
            run_id="run-1",
            accumulator=SubagentRunAccumulator(),
            contract=None,
            admission_head=None,
            runtime=_runtime(tmp_path),
        )
        assert probes == []
        assert spawn.commands == []
        assert finalization.checks == ()
        assert finalization.publication_head is None
        assert finalization.runtime_failure is None

    def test_requires_commit_artifact_truth_table(self, tmp_path: Path) -> None:
        check_only = _freeze(
            tmp_path, completion_check=[CompletionCheckTemplate.PYTEST]
        )
        assert check_only.requires_commit_artifact is False
        commit_only = _freeze(
            tmp_path, required_artifacts=[CompletionArtifactKind.COMMIT]
        )
        assert commit_only.requires_commit_artifact is True
        both = _freeze(
            tmp_path,
            completion_check=[CompletionCheckTemplate.PYTEST],
            required_artifacts=[CompletionArtifactKind.COMMIT],
        )
        assert both.requires_commit_artifact is True
        mandate_free = _freeze(tmp_path)
        assert mandate_free.requires_commit_artifact is False
        # An exempt seat never arms the probe, even with a commit mandate.
        reviewer = _freeze(
            tmp_path,
            identity="reviewer",
            required_artifacts=[CompletionArtifactKind.COMMIT],
        )
        assert reviewer.envelope_required is False
        assert reviewer.requires_commit_artifact is False
        # A read-only identity with mandated artifacts carries the reduced
        # matrix: the commit artifact is stripped, so no probe.
        researcher = _freeze(
            tmp_path,
            identity="researcher",
            required_artifacts=[
                CompletionArtifactKind.COMMIT,
                CompletionArtifactKind.TOUCHED_PATH,
            ],
        )
        assert "reduced_matrix" in researcher.exemptions
        assert researcher.requires_commit_artifact is False

    def test_admission_capture_probes_once_per_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        probes: list[Path] = []
        monkeypatch.setattr(GitRepo, "head_commit_at", _probe(probes))
        assert _capture_admission_head(tmp_path) == "c" * 40
        assert len(probes) == 1

        def _failing(target: Path) -> str:
            raise GitError("not a repository")

        monkeypatch.setattr(GitRepo, "head_commit_at", _failing)
        # A probe failure is informational: the baseline is None, never an
        # admission failure.
        assert _capture_admission_head(tmp_path) is None


# ----------------------------------------------------------------------
# Bounded payloads
# ----------------------------------------------------------------------


class TestBoundedPayloads:
    """The audit path's payload caps, pinned and exercised at the boundary."""

    def test_budget_constants_are_pinned(self) -> None:
        assert MAX_ENVELOPE_REQUIREMENTS == 16
        assert MAX_SATISFIED_REQUIREMENTS == 16
        assert ARTIFACT_SNAPSHOT_CAP_BYTES == 4096
        assert MAX_CHRONOLOGY_ENTRIES == 128
        assert BLOCK_MESSAGE_MAX_PATHS == 8
        assert MAX_CAPTURE_BYTES == 64 * 1024

    def test_artifact_snapshot_is_capped_at_collection(self, tmp_path: Path) -> None:
        content = "x" * (ARTIFACT_SNAPSHOT_CAP_BYTES + 4096)
        accumulator = SubagentRunAccumulator()
        artifact = tmp_path / "big.txt"
        _observe_write(accumulator, artifact, "occ-1", content=content)
        entry = accumulator._chronology.entry("occ-1")
        assert entry is not None
        snapshot = entry.snapshot
        assert snapshot is not None
        # The retained samples stay bounded; the hash covers the whole file.
        assert (
            len(snapshot.head_bytes) + len(snapshot.tail_bytes)
            <= ARTIFACT_SNAPSHOT_CAP_BYTES
        )
        assert snapshot.truncated is True
        assert snapshot.byte_size == len(content)

    def test_chronology_evicts_oldest_at_the_cap(self, tmp_path: Path) -> None:
        accumulator = SubagentRunAccumulator()
        total = MAX_CHRONOLOGY_ENTRIES + 8
        for index in range(total):
            _observe_write(
                accumulator,
                tmp_path / "src" / f"file-{index:03d}.py",
                f"occ-{index}",
                content=f"payload {index}\n",
            )
        assert len(accumulator._chronology._entries) == MAX_CHRONOLOGY_ENTRIES
        # Eviction is an availability fact about the harness, never a
        # worker disposition.
        assert (
            accumulator._chronology.resolution("occ-0") is EvidenceAvailability.EVICTED
        )
        assert (
            accumulator._chronology.resolution(f"occ-{total - 1}")
            is EvidenceAvailability.COLLECTED
        )

    def test_block_message_names_at_most_the_capped_paths(self) -> None:
        findings = [
            SecretFinding(
                path=f"src/file-{index:02d}.py",
                rule_id="aws-access-key-id",
                description="redacted finding detail",
                line=1,
                column=1,
            )
            for index in range(BLOCK_MESSAGE_MAX_PATHS + 5)
        ]
        message = blocked_submission_response(findings)
        for index in range(BLOCK_MESSAGE_MAX_PATHS):
            assert f"file-{index:02d}.py" in message
        assert "file-08.py" not in message
        assert "5 more" in message
        # The message never carries the rule id or the finding detail.
        assert "aws-access-key-id" not in message
        assert "redacted finding detail" not in message

    @pytest.mark.asyncio
    async def test_check_capture_keeps_a_capped_tail(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        spawn = _SpawnRecorder(stdout=b"x" * (3 * MAX_CAPTURE_BYTES))
        _install_spawn(monkeypatch, spawn)
        runner = _check_runner(tmp_path)
        outcome = await runner.run(CompletionCheckTemplate.PYTEST)
        assert outcome.status is CompletionCheckStatus.PASSED
        assert len(spawn.commands) == 1
        assert len(outcome.stdout) < 3 * MAX_CAPTURE_BYTES
        assert "bytes truncated" in outcome.stdout
        assert outcome.stdout.startswith("x" * 64)
        assert outcome.stdout.endswith("x" * 64)

    def test_envelope_and_receipt_grammar_caps(self) -> None:
        requirements = tuple(
            CompletionRequirement(
                requirement_id=f"req-{index}", template=CompletionCheckTemplate.PYTEST
            )
            for index in range(MAX_ENVELOPE_REQUIREMENTS + 1)
        )
        with pytest.raises(ValidationError):
            CompletionEnvelope(
                requirements=requirements,
                schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
                marker=COMPLETION_ENVELOPE_MARKER,
            )
        envelope = CompletionEnvelope(
            requirements=(
                CompletionRequirement(
                    requirement_id="req-1", template=CompletionCheckTemplate.PYTEST
                ),
            ),
            schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
            marker=COMPLETION_ENVELOPE_MARKER,
        )
        with pytest.raises(ValidationError):
            CompletionReceipt(
                marker=COMPLETION_ENVELOPE_MARKER,
                schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
                envelope=envelope,
                satisfied_requirement_ids=tuple(
                    f"req-{index}" for index in range(MAX_SATISFIED_REQUIREMENTS + 1)
                ),
            )


# ----------------------------------------------------------------------
# Timeout paths never retry
# ----------------------------------------------------------------------


class TestTimeoutNeverRetries:
    """A timed-out check spawns exactly once — no retries, no fallbacks."""

    @pytest.mark.asyncio
    async def test_a_timed_out_check_spawns_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        commands: list[str] = []

        class _Hanging:
            async def communicate(self) -> tuple[bytes, bytes]:
                await asyncio.sleep(30)
                return (b"", b"")

            @property
            def returncode(self) -> None:
                return None

        async def _spawn(command: str, **kwargs: Any) -> Any:
            commands.append(command)
            return _Hanging()

        monkeypatch.setattr(
            "chartreux.core.completion_checks.spawn_shell_command", _spawn
        )
        monkeypatch.setattr(
            "chartreux.core.completion_checks.registered_process_group",
            lambda proc: SimpleNamespace(),
        )

        async def _terminate(owner: Any, **kwargs: Any) -> None:
            return None

        monkeypatch.setattr(
            "chartreux.core.completion_checks.terminate_process_group", _terminate
        )
        runner = _check_runner(tmp_path, timeout=0.05)
        outcome = await runner.run(CompletionCheckTemplate.PYTEST)
        assert outcome.status is CompletionCheckStatus.TIMED_OUT
        assert outcome.detail == "timed out after 0.05s"
        # One acceptance attempt, one spawn: the timeout path never retries.
        assert len(commands) == 1


# ----------------------------------------------------------------------
# Zero honest-report false rejections under enforcement
# ----------------------------------------------------------------------


class TestZeroHonestFalseRejectionsUnderEnforcement:
    """Honest reports publish under enforce mode — never blocked."""

    def test_honest_full_claim_with_matching_evidence_publishes(
        self, tmp_path: Path
    ) -> None:
        contract = _freeze(tmp_path, completion_check=[CompletionCheckTemplate.PYTEST])
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, tmp_path / "out.txt", "occ-1")
        _emit_receipt(accumulator, "occ-1", PYTEST_COMMAND)
        report = "Work complete.\n" + json.dumps(
            _claim_payload(contract, ["check:pytest"])
        )
        result = _publish(
            accumulator,
            contract=contract,
            report=report,
            finalization=_finalization("run-1", checks=[_passed_check()]),
        )
        assert result.completed is True
        audit = result.completion_audit
        assert audit is not None
        assert audit.claim_fidelity is CompletionClaimFidelity.CLAIM_MATCHED
        assert audit.rejection is None

    def test_honest_report_without_a_receipt_publishes(self, tmp_path: Path) -> None:
        contract = _freeze(tmp_path, completion_check=[CompletionCheckTemplate.PYTEST])
        # The harness finalizes every run, so the composition carries the
        # check classifications' surface even for a receipt-less report.
        result = _publish(
            SubagentRunAccumulator(),
            contract=contract,
            report="All done, nothing to claim.",
            finalization=_finalization("run-1", checks=[_passed_check()]),
        )
        assert result.completed is True
        assert result.response == "All done, nothing to claim."
        audit = result.completion_audit
        assert audit is not None
        assert audit.rejection is CompletionRejectionReason.RECEIPT_UNVERIFIABLE

    def test_honest_abort_is_never_replaced_by_a_rejection(
        self, tmp_path: Path
    ) -> None:
        contract = _freeze(tmp_path, completion_check=[CompletionCheckTemplate.PYTEST])
        report = "Blocked: the acceptance check could not run here."
        result = _publish(
            SubagentRunAccumulator(),
            contract=contract,
            report=report,
            declared_completed=False,
        )
        assert result.completed is False
        # The honest report survives (the recorded cancellation rides it);
        # no rejection message ever replaces it.
        assert result.response.startswith(report)
        assert "[Completion audit rejected:" not in result.response

    def test_honest_red_green_retry_publishes(self, tmp_path: Path) -> None:
        contract = _freeze(tmp_path, completion_check=[CompletionCheckTemplate.PYTEST])
        accumulator = SubagentRunAccumulator()
        _observe_bash(accumulator, PYTEST_COMMAND, 1, "1 failed", "occ-red")
        _observe_bash(accumulator, PYTEST_COMMAND, 0, "ok", "occ-green")
        report = "The first check run failed; the retry passed.\n" + json.dumps(
            _claim_payload(contract, ["check:pytest"])
        )
        result = _publish(
            accumulator,
            contract=contract,
            report=report,
            finalization=_finalization("run-1", checks=[_passed_check()]),
        )
        assert result.completed is True
        audit = result.completion_audit
        assert audit is not None
        assert audit.rejection is None

    def test_reviewer_exempt_identity_is_never_blocked(self, tmp_path: Path) -> None:
        contract = _freeze(
            tmp_path,
            identity="reviewer",
            completion_check=[CompletionCheckTemplate.PYTEST],
            required_artifacts=[CompletionArtifactKind.COMMIT],
        )
        report = "Review complete.\n" + json.dumps(
            _claim_payload(contract, ["check:pytest", "artifact:commit"])
        )
        result = _publish(SubagentRunAccumulator(), contract=contract, report=report)
        assert result.completed is True
        audit = result.completion_audit
        assert audit is not None
        # The exemption is an admission-surface fact (no envelope
        # requirement, no HEAD probe — pinned in TestHeadSnapshotBudget);
        # an exempt seat's report is never blocked.
        assert contract.envelope_required is False
        assert audit.rejection is None


# ----------------------------------------------------------------------
# The production call site (end to end, through the session registry)
# ----------------------------------------------------------------------


class _NotedBackend(FakeBackend):
    """A parent backend that answers every turn with terminal content."""

    async def complete(self, **kwargs: Any) -> LLMChunk:
        return mock_llm_chunk(content="parent noted")

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        yield mock_llm_chunk(content="parent noted")


def _config_with_mode(mode: AuditMode) -> Any:
    """The parent config granting bash, with the completion audit mode."""
    config = _config(enabled_tools=["task", "bash"])
    return config.model_copy(update={"completion": CompletionSectionConfig(mode=mode)})


def _mandated_args(**overrides: Any) -> TaskArgs:
    params: dict[str, Any] = {
        "task": "Mandated work",
        "identity": "coder",
        "background": True,
        "retain": True,
        "completion_check": [CompletionCheckTemplate.PYTEST],
    }
    params.update(overrides)
    return TaskArgs(**params)


def _registry_with_child_rig(
    monkeypatch: pytest.MonkeyPatch, mode: AuditMode, child: FakeBackend
) -> tuple[Any, AgentLoop, list[dict[str, Any]]]:
    """The registry with a parent granting bash, and a counting backend rig.

    The parent grants bash (a real orchestrator does), so the harness
    verification capability derives as available and the bound check
    actually spawns. The rig counts backend constructions: the child
    loop's construction at launch is the only legitimate call.
    """
    backend_calls: list[dict[str, Any]] = []

    def _rig(**kwargs: Any) -> Any:
        backend_calls.append(kwargs)
        return child

    monkeypatch.setattr("chartreux.core.agent_loop._loop.create_backend", _rig)
    parent = build_test_agent_loop(
        config=_config_with_mode(mode), backend=_NotedBackend()
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent, backend_calls


class TestProductionPublication:
    """The economics pinned at the real background publication call site."""

    @pytest.mark.asyncio
    async def test_enforce_mode_honest_run_publication_economics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An honest enforce-mode run costs exactly one acceptance attempt,
        zero HEAD probes (check-only mandate), and no backend beyond the
        child loop — and it publishes.
        """
        child = FakeBackend([mock_llm_chunk(content="All done, nothing to claim.")])
        registry, parent, backend_calls = _registry_with_child_rig(
            monkeypatch, "enforce", child
        )
        spawn = _SpawnRecorder()
        _install_spawn(monkeypatch, spawn)
        probes: list[Path] = []
        monkeypatch.setattr(GitRepo, "head_commit_at", _probe(probes))
        try:
            ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
            ack = await _background_result(registry, _mandated_args(), ctx)
            assert ack.agent_id is not None and ack.run_id is not None
            result = await _wait_result(registry, ack.agent_id, ack.run_id)
            assert result.completed is True
            assert result.response == "All done, nothing to claim."
            # At most one acceptance attempt: exactly one bound check
            # spawned, and a repeated retrieval re-runs nothing.
            assert len(spawn.commands) == 1
            assert spawn.commands[0].startswith(PYTEST_COMMAND)
            again = await _wait_result(registry, ack.agent_id, ack.run_id)
            assert again.completed is True
            assert len(spawn.commands) == 1
            # No commit mandate: zero HEAD probes across the whole run.
            assert probes == []
            # Zero LLM calls on the publication path: exactly two backend
            # constructions, both at setup (the root runtime's loop and the
            # child's, before the run starts) — publication adds none.
            assert len(backend_calls) == 2
        finally:
            await registry.drain_children()
            await parent.aclose()

    @pytest.mark.asyncio
    async def test_commit_mandated_run_probes_exactly_two_heads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A commit-mandated run pays exactly two HEAD snapshots — the
        admission baseline and the publication snapshot — and still
        publishes an honest report.
        """
        child = FakeBackend([mock_llm_chunk(content="All done, nothing to claim.")])
        registry, parent, _backend_calls = _registry_with_child_rig(
            monkeypatch, "enforce", child
        )
        spawn = _SpawnRecorder()
        _install_spawn(monkeypatch, spawn)
        probes: list[Path] = []
        monkeypatch.setattr(GitRepo, "head_commit_at", _probe(probes))
        try:
            ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
            args = _mandated_args(required_artifacts=[CompletionArtifactKind.COMMIT])
            ack = await _background_result(registry, args, ctx)
            assert ack.agent_id is not None and ack.run_id is not None
            result = await _wait_result(registry, ack.agent_id, ack.run_id)
            assert result.completed is True
            assert len(probes) == 2
            assert len(spawn.commands) == 1
        finally:
            await registry.drain_children()
            await parent.aclose()
