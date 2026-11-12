"""WP16b: the child failure envelope and its derivation inputs (item 2 B.2/B.3).

A process-backed child's measured evidence dies with its process, so the child
derives the typed failure envelope where the evidence still lives and its
terminal frame carries BOTH the derived envelope and the derivation inputs the
coordinator re-derives from. These tests pin that carrier, the
eligibility-EQUIVALENT rebuild (not byte compatibility), the child-side
truncation cases of B.2 end to end, the disk-based outstanding-call
reconciliation of a process-backed child, and the two loss cases: a lost
completion (no terminal frame, no envelope) and a crash that left the derived
envelope in the durable result record.

Every fault here is a SYNTHETIC fixture; no transport reaches a provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.app_server._subagent_recovery import (
    _submission_defect,
    disk_outstanding_tool_call_ids,
    reconcile_outstanding_ids,
)
from chartreux.app_server._subagent_transcript import (
    ChildResultRecord,
    ChildStartMarker,
    ChildTranscriptOutcome,
    ChildTranscriptStore,
    child_result_digest,
)
from chartreux.app_server.agent_wire import (
    AGENT_TURN_ENDED_METHOD,
    SUBAGENT_OPEN_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
)
from chartreux.app_server.child_agent import ChildAgentServer
from chartreux.app_server.child_agent_process import (
    ProcessSubagentCompletionSource,
    persisted_child_failure,
)
from chartreux.app_server.root_agent_process import AgentProcessTurnFailed
from chartreux.app_server.subagent_ports import (
    CHILD_FAILURE_FRAME_KEY,
    ChildFailureEvidence,
    SubagentCompletion,
    child_failure_evidence,
    child_spawn_spec,
    decode_child_failure,
    encode_child_failure,
    failure_decision,
    spawn_spec_to_wire,
    subagent_completion_frame,
)
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.llm.exceptions import (
    BackendError,
    IncompleteStreamError,
    PayloadSummary,
)
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.recovery_models import (
    CallEvidence,
    FailureClass,
    RecoveryEligibility,
    ToolEffectsDisposition,
    attach_call_evidence,
)
from chartreux.core.subagents import EffectVerification

_ROOT_SESSION_ID = "root-session-1"
_SESSION_ID = "child-session-1"


def _measured_call() -> CallEvidence:
    """A measured logical call: the provenance a transient needs (R2)."""
    return CallEvidence(
        logical_call_id="call-measured-1",
        started_at=0.0,
        budget_seconds=900.0,
        response_started=True,
        semantic_output_published=False,
        transport_attempts=2,
        elapsed_seconds=1.5,
        budget_exhausted=False,
    )


def _truncated_stream_error() -> IncompleteStreamError:
    """A provider stream truncation carrying its own measured call evidence."""
    error = IncompleteStreamError("prov", "wire")
    attach_call_evidence(error, _measured_call())
    return error


def _rate_limit_error() -> BackendError:
    error = BackendError(
        provider="prov",
        endpoint="https://api.example.com/v1",
        status=429,
        reason="rate limited",
        headers=None,
        body_text=None,
        parsed_error=None,
        model="wire",
        payload_summary=PayloadSummary(
            model="wire",
            message_count=1,
            approx_chars=2,
            temperature=0.0,
            has_tools=False,
            tool_choice=None,
        ),
    )
    attach_call_evidence(error, _measured_call())
    return error


def _settled_tool_messages() -> list[LLMMessage]:
    """One requested-and-settled transcript-visible tool call."""
    return [
        LLMMessage(role=Role.user, message_id="message-1", content="go"),
        LLMMessage(
            role=Role.assistant,
            message_id="message-2",
            content="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    index=0,
                    function=FunctionCall(name="read_file", arguments="{}"),
                )
            ],
        ),
        LLMMessage(
            role=Role.tool,
            message_id="message-3",
            tool_call_id="call-1",
            name="read_file",
            content="ok",
        ),
    ]


def _call_free_messages() -> list[LLMMessage]:
    """No transcript-visible tool calls: effects stay UNKNOWN, never settled."""
    return [LLMMessage(role=Role.user, message_id="message-1", content="go")]


def _loop(messages: list[LLMMessage]) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=_SESSION_ID,
        messages=messages,
        committed_model=SimpleNamespace(
            base_model="base-a", provider="prov-a", wire_name="wire-a"
        ),
        config=SimpleNamespace(active_model="configured-wire"),
    )


def _evidence(
    messages: list[LLMMessage] | None = None,
    *,
    error: BaseException | None = None,
    stop_reason: str | None = None,
    exclusion_reasons: list[str] | None = None,
) -> ChildFailureEvidence:
    return child_failure_evidence(
        _loop(messages if messages is not None else _settled_tool_messages()),
        error if error is not None else _truncated_stream_error(),
        agent_id="coder:py",
        run_id="turn-1",
        turn_id="turn-1",
        stop_reason=stop_reason,
        exclusion_reasons=exclusion_reasons,
    )


# ---------------------------------------------------------------------------
# The completion frame carries the derived envelope plus its derivation inputs
# ---------------------------------------------------------------------------


def test_the_completion_frame_carries_the_envelope_and_its_derivation_inputs() -> None:
    evidence = _evidence()
    assert evidence.envelope.failure.failure_class is FailureClass.INCOMPLETE_STREAM
    assert evidence.envelope.recovery.eligibility is RecoveryEligibility.ELIGIBLE
    assert evidence.envelope.progress.tool_effects is ToolEffectsDisposition.SETTLED

    frame = subagent_completion_frame(
        "coder:py",
        7,
        SubagentCompletion(
            outcome=TURN_OUTCOME_FAILED,
            error="IncompleteStreamError: truncated",
            failure=evidence,
        ),
    )
    assert frame["method"] == AGENT_TURN_ENDED_METHOD
    block = frame["params"][CHILD_FAILURE_FRAME_KEY]

    # The derived result rides the frame...
    assert block["envelope"]["failure"]["failure_class"] == "incomplete_stream"
    assert block["envelope"]["recovery"]["eligibility"] == "eligible"
    assert "tool_effects_settled" in block["envelope"]["recovery"]["reason_codes"]
    # ...and so do the inputs a re-derivation consumes, not a field subset.
    inputs = block["derivation_inputs"]
    assert inputs["tool_calls_present"] is True
    assert inputs["tool_calls_pending"] is False
    assert inputs["attempted_deployment"] == ["base-a", "prov-a", "wire-a"]
    assert inputs["configured_model"] == "configured-wire"
    assert inputs["stop_reason"] is None
    assert "exclusion_reasons" in inputs
    observation = inputs["observation"]
    assert observation["phase"] == "turn"
    assert observation["call_evidence"]["transport_attempts"] == 2
    assert observation["classification"]["category"] == "incomplete_stream"
    assert "retry_after" in observation["classification"]

    # The frame is wire-safe: a JSON round trip decodes to the same decision.
    decoded = decode_child_failure(
        json.loads(json.dumps(frame))["params"][CHILD_FAILURE_FRAME_KEY]
    )
    assert decoded is not None
    assert failure_decision(decoded.envelope) == failure_decision(evidence.envelope)
    assert decoded.derivation_inputs["tool_calls_present"] is True


def test_a_completed_run_carries_no_failure_block() -> None:
    frame = subagent_completion_frame(
        "coder:py", 7, SubagentCompletion(outcome=TURN_OUTCOME_COMPLETED)
    )
    assert CHILD_FAILURE_FRAME_KEY not in frame["params"]
    assert decode_child_failure(None) is None
    assert decode_child_failure({"derivation_inputs": {}}) is None


# ---------------------------------------------------------------------------
# The rebuild is eligibility-EQUIVALENT, and the inputs make it so
# ---------------------------------------------------------------------------


def test_settled_and_call_free_truncations_rebuild_the_same_decision() -> None:
    settled = _evidence()
    assert settled.rebuild().recovery.eligibility is RecoveryEligibility.ELIGIBLE
    assert settled.eligibility_equivalent is True

    call_free = _evidence(_call_free_messages())
    assert call_free.envelope.failure.failure_class is FailureClass.INCOMPLETE_STREAM
    assert call_free.envelope.progress.tool_effects is ToolEffectsDisposition.UNKNOWN
    assert (
        call_free.envelope.recovery.eligibility
        is RecoveryEligibility.RECONCILE_REQUIRED
    )
    assert "tool_effects_unknown_reconcile_first" in (
        call_free.envelope.recovery.reason_codes
    )
    rebuilt = call_free.rebuild()
    assert failure_decision(rebuilt) == failure_decision(call_free.envelope)
    assert call_free.eligibility_equivalent is True


def test_a_budget_stop_stays_prohibited_because_the_inputs_ride_the_frame() -> None:
    """I2-B3: a field subset without ``stop_reason`` wrongly admits this run."""
    budget = _evidence(_settled_tool_messages(), stop_reason="budget_exceeded")
    assert budget.envelope.failure.failure_class is FailureClass.BUDGET_STOP
    assert budget.envelope.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert budget.eligibility_equivalent is True

    stripped = dict(budget.derivation_inputs)
    stripped["stop_reason"] = None
    misadmitted = ChildFailureEvidence(
        envelope=budget.envelope, derivation_inputs=stripped
    )
    assert misadmitted.rebuild().recovery.eligibility is RecoveryEligibility.ELIGIBLE
    assert misadmitted.eligibility_equivalent is False


def test_a_rebuild_without_the_measured_evidence_fails_closed() -> None:
    """R2: the measured call evidence is an input the rebuild cannot invent."""
    measured = _evidence(_settled_tool_messages(), error=_rate_limit_error())
    assert measured.envelope.recovery.eligibility is RecoveryEligibility.ELIGIBLE

    stripped = ChildFailureEvidence(
        envelope=measured.envelope,
        derivation_inputs={
            **measured.derivation_inputs,
            "observation": {
                **measured.derivation_inputs["observation"],
                "call_evidence": None,
            },
        },
    )
    rebuilt = stripped.rebuild()
    assert rebuilt.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert "transient_cause_unmeasured_no_recovery" in rebuilt.recovery.reason_codes
    assert stripped.eligibility_equivalent is False


# ---------------------------------------------------------------------------
# B.2 end to end through the child server: settled/ELIGIBLE, no-call/RECONCILE
# ---------------------------------------------------------------------------


class _ChildTransport:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        await asyncio.Event().wait()
        yield {}  # pragma: no cover - a child transport never yields here

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        return None


class _FailingPort:
    """An execution port whose turn stream dies on the given fault."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def run_turn(self, request: ExecutionTurnRequest) -> AsyncIterator[Any]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        raise self._error
        yield  # pragma: no cover - keeps the generator shape


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "sessions" / _ROOT_SESSION_ID
    root.mkdir(parents=True)
    return root


async def _run_failed_turn(
    root: Path, messages: list[LLMMessage], error: BaseException
) -> tuple[ChildTranscriptStore, dict[str, Any], SimpleNamespace, _ChildTransport]:
    session_dir = root / "agents" / f"coder-py_{_SESSION_ID}"
    session_dir.mkdir(parents=True)
    loop = _loop(messages)
    loop.session_logger = SimpleNamespace(session_dir=session_dir)
    loop.execution_port = _FailingPort(error)

    class _Factory:
        async def open(self, params: Any) -> Any:
            return loop

        async def close(self) -> None:
            return None

    transport = _ChildTransport()
    server = ChildAgentServer(cast("Any", transport), factory=cast("Any", _Factory()))
    spec = child_spawn_spec(
        agent_id="coder:py",
        parent_session_id=_ROOT_SESSION_ID,
        root_session_dir=root,
        payload={},
    )
    opened = await server._handle_request(
        SUBAGENT_OPEN_METHOD, spawn_spec_to_wire(spec)
    )
    assert opened["session_dir"] == str(session_dir)

    await server._run_turn(
        "turn-1", {"message": "hi", "turn_options": {"turn_id": "turn-1"}}
    )

    ended = [
        message
        for message in transport.sent
        if message.get("method") == AGENT_TURN_ENDED_METHOD
    ]
    assert ended
    store = ChildTranscriptStore(session_dir, root_session_dir=root)
    return store, ended[-1], loop, transport


@pytest.mark.asyncio
async def test_a_settled_truncation_reaches_the_parent_eligible(tmp_path: Path) -> None:
    root = _root(tmp_path)
    store, ended, _loop_obj, _transport = await _run_failed_turn(
        root, _settled_tool_messages(), _truncated_stream_error()
    )

    assert ended["params"]["outcome"] == TURN_OUTCOME_FAILED
    evidence = decode_child_failure(ended["params"][CHILD_FAILURE_FRAME_KEY])
    assert evidence is not None
    assert evidence.envelope.failure.failure_class is FailureClass.INCOMPLETE_STREAM
    assert evidence.envelope.failure.confidence.value == "measured"
    assert evidence.envelope.progress.tool_effects is ToolEffectsDisposition.SETTLED
    assert evidence.envelope.recovery.eligibility is RecoveryEligibility.ELIGIBLE
    assert evidence.eligibility_equivalent is True

    # The save-before-emit record carries the same evidence.
    adjudication = store.adjudicate()
    assert adjudication.outcome is ChildTranscriptOutcome.COMPLETED
    persisted = persisted_child_failure(adjudication)
    assert persisted is not None
    assert failure_decision(persisted.envelope) == failure_decision(evidence.envelope)
    assert persisted.eligibility_equivalent is True


@pytest.mark.asyncio
async def test_a_call_free_truncation_requires_an_explicit_reconcile(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    _store, ended, _loop_obj, _transport = await _run_failed_turn(
        root, _call_free_messages(), _truncated_stream_error()
    )

    evidence = decode_child_failure(ended["params"][CHILD_FAILURE_FRAME_KEY])
    assert evidence is not None
    assert evidence.envelope.progress.tool_effects is ToolEffectsDisposition.UNKNOWN
    assert (
        evidence.envelope.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
    )
    assert evidence.eligibility_equivalent is True


# ---------------------------------------------------------------------------
# The coordinator side of the same carrier
# ---------------------------------------------------------------------------


class _FailedTurnProcess:
    """A child process double whose run failed inside its own process."""

    def __init__(self, terminal: dict[str, Any] | None) -> None:
        self._terminal = terminal
        self.turn = ExecutionTurnRequest(
            turn_options=AgentTurnOptions(turn_id="turn-1")
        )

    def run_turn(self, request: ExecutionTurnRequest) -> AsyncIterator[Any]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        raise AgentProcessTurnFailed("child run failed")
        yield  # pragma: no cover - keeps the generator shape

    def turn_terminal(self, turn_id: str) -> dict[str, Any] | None:
        return self._terminal


async def _completion(process: _FailedTurnProcess) -> SubagentCompletion:
    source = ProcessSubagentCompletionSource("coder:py", cast("Any", process))
    request = process.turn
    frames = [frame async for frame in source.stream(request)]
    assert len(frames) == 1
    completion = frames[0]
    assert isinstance(completion, SubagentCompletion)
    return completion


@pytest.mark.asyncio
async def test_the_coordinator_rebuilds_from_the_children_terminal_frame() -> None:
    evidence = _evidence()
    process = _FailedTurnProcess({
        "turn_id": "turn-1",
        "outcome": TURN_OUTCOME_FAILED,
        "error": "IncompleteStreamError: truncated",
        CHILD_FAILURE_FRAME_KEY: encode_child_failure(evidence),
    })
    completion = await _completion(process)
    assert completion.outcome == TURN_OUTCOME_FAILED
    assert completion.failure is not None
    assert failure_decision(completion.failure.envelope) == failure_decision(
        evidence.envelope
    )
    assert completion.failure.eligibility_equivalent is True


@pytest.mark.asyncio
async def test_a_lost_completion_is_failed_without_an_envelope() -> None:
    completion = await _completion(_FailedTurnProcess(None))
    assert completion.outcome == TURN_OUTCOME_FAILED
    assert completion.error
    assert completion.failure is None


# ---------------------------------------------------------------------------
# A crashed child leaves its derived envelope in the durable record
# ---------------------------------------------------------------------------


def _child_store(root: Path) -> ChildTranscriptStore:
    child_dir = root / "agents" / "coder-py_20260101_000000_abcdef12"
    child_dir.mkdir(parents=True)
    store = ChildTranscriptStore(child_dir, root_session_dir=root)
    store.record_start(
        ChildStartMarker(
            agent_id="coder:py",
            session_id=_SESSION_ID,
            run_id="turn-1",
            started_at="2026-01-01T00:00:00+00:00",
        )
    )
    return store


def _failed_record(failure: dict[str, Any] | None) -> ChildResultRecord:
    return ChildResultRecord(
        agent_id="coder:py",
        session_id=_SESSION_ID,
        run_id="turn-1",
        result_id="coder:py:turn-1",
        digest=child_result_digest(
            agent_id="coder:py",
            session_id=_SESSION_ID,
            run_id="turn-1",
            outcome=TURN_OUTCOME_FAILED,
            error="IncompleteStreamError: truncated",
        ),
        outcome=TURN_OUTCOME_FAILED,
        completed_at="2026-01-01T00:00:05+00:00",
        error="IncompleteStreamError: truncated",
        failure=failure,
    )


def test_a_crash_after_the_terminal_save_keeps_the_envelope(tmp_path: Path) -> None:
    root = _root(tmp_path)
    store = _child_store(root)
    evidence = _evidence()
    store.record_result(_failed_record(encode_child_failure(evidence)))

    adjudication = store.adjudicate()
    assert adjudication.outcome is ChildTranscriptOutcome.COMPLETED
    persisted = persisted_child_failure(adjudication)
    assert persisted is not None
    assert failure_decision(persisted.envelope) == failure_decision(evidence.envelope)
    assert persisted.eligibility_equivalent is True


def test_a_record_without_the_block_reports_no_envelope(tmp_path: Path) -> None:
    root = _root(tmp_path)
    store = _child_store(root)
    store.record_result(_failed_record(None))

    assert persisted_child_failure(store.adjudicate()) is None


# ---------------------------------------------------------------------------
# Disk-based outstanding-call reconciliation (the process-topology equivalent)
# ---------------------------------------------------------------------------


def _write_messages(child_dir: Path, messages: list[LLMMessage]) -> None:
    child_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(message.model_dump(mode="json"), separators=(",", ":"))
        for message in messages
    ]
    (child_dir / "messages.jsonl").write_text("\n".join(lines) + "\n")


def _requested_and_outstanding() -> list[LLMMessage]:
    """One settled call plus one requested-but-unanswered call."""
    return [
        LLMMessage(role=Role.user, message_id="message-1", content="go"),
        LLMMessage(
            role=Role.assistant,
            message_id="message-2",
            content="",
            tool_calls=[
                ToolCall(
                    id="call-settled",
                    index=0,
                    function=FunctionCall(name="read_file", arguments="{}"),
                )
            ],
        ),
        LLMMessage(
            role=Role.tool,
            message_id="message-3",
            tool_call_id="call-settled",
            name="read_file",
            content="ok",
        ),
        LLMMessage(
            role=Role.assistant,
            message_id="message-4",
            content="",
            tool_calls=[
                ToolCall(
                    id="call-outstanding",
                    index=0,
                    function=FunctionCall(name="write_file", arguments="{}"),
                )
            ],
        ),
    ]


def test_disk_records_yield_the_same_outstanding_ids_as_the_live_loop(
    tmp_path: Path,
) -> None:
    messages = _requested_and_outstanding()
    child_dir = tmp_path / "agents" / "coder-py_20260101_000000_abcdef12"
    _write_messages(child_dir, messages)

    from_loop = reconcile_outstanding_ids(messages)
    from_disk = reconcile_outstanding_ids(child_dir=child_dir)
    assert from_loop == {"call-outstanding"}
    assert from_disk == from_loop
    assert disk_outstanding_tool_call_ids(child_dir) == {"call-outstanding"}

    # The read is fail-closed: an absent transcript is None, never an empty set.
    assert disk_outstanding_tool_call_ids(tmp_path / "agents" / "absent") is None


def test_the_disk_accounting_set_decides_a_clearing_submission(tmp_path: Path) -> None:
    messages = _requested_and_outstanding()
    child_dir = tmp_path / "agents" / "coder-py_20260101_000000_abcdef12"
    _write_messages(child_dir, messages)
    outstanding = reconcile_outstanding_ids(child_dir=child_dir)
    assert outstanding is not None

    settled = [
        EffectVerification(tool_call_id="call-outstanding", disposition="settled")
    ]
    assert _submission_defect(messages, settled, outstanding=outstanding) is None

    codes, reason = _submission_defect([], [], outstanding=outstanding) or ([], "")
    assert "unverified_outstanding_tool_calls" in codes
    assert "call-outstanding" in codes
    assert "account" in reason

    codes, _reason = _submission_defect(
        messages,
        [EffectVerification(tool_call_id="call-other", disposition="settled")],
        outstanding=outstanding,
    ) or ([], "")
    assert "unknown_tool_call_ids" in codes
    assert "call-other" in codes
    assert "unverified_outstanding_tool_calls" in codes

    # A verified damage report is never refused over the accounting technicality.
    damage = [
        EffectVerification(
            tool_call_id="call-elsewhere", disposition="unrepeatable_damage"
        )
    ]
    assert _submission_defect(messages, damage, outstanding=outstanding) is None
