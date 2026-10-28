"""P2: typed completion-evidence receipts at the post-tool seam.

The side-channel callback is invoked from ``_run_post_tool_and_finalize``
only — never on the typed event stream — and carries the executor-minted,
run-scoped receipt occurrence ID. Round-9 (binding): the ID is minted
BEFORE the invocation at the tool-call dispatch site and propagates through
BOTH the callback payload and the result events the accumulator collects
(P3 registers it); tool_call_id alone never keys pairing because tool-call
ids are reused across turns.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop_hooks import CompletionEvidenceReceipt
from chartreux.core.events import BaseEvent, ToolResultEvent, ToolStreamEvent
from chartreux.core.hooks.models import HookTextReplacement
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.tools import secret_redaction as sr
from chartreux.core.tools.base import CancellableToolResult, ToolPermission
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_tool import FakeTool, FakeToolArgs, FakeToolResult


@pytest.fixture(autouse=True)
def _clean_redaction_state() -> Any:
    sr.set_env_passthrough(())
    sr.reset_cache()
    yield
    sr.set_env_passthrough(())
    sr.reset_cache()


class _CancelledFakeToolResult(FakeToolResult, CancellableToolResult):
    pass


class _CancellingFakeTool(FakeTool):
    async def run(
        self, args: FakeToolArgs, ctx: Any = None
    ) -> AsyncGenerator[ToolStreamEvent | FakeToolResult, None]:
        yield _CancelledFakeToolResult(
            message=args.text or "cancelled run", cancelled=True
        )


class _JobLikeResult(FakeToolResult):
    job: dict[str, Any]


def _stub_call(call_id: str, arguments: str = "{}") -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name="stub_tool", arguments=arguments),
    )


def _make_stub_loop(
    turns: list[list[Any]],
    sink: _ReceiptSink,
    tool_class: type[FakeTool] = FakeTool,
    tool_permission: ToolPermission = ToolPermission.ALWAYS,
) -> AgentLoop:
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["stub_tool"],
            tools={"stub_tool": {"permission": tool_permission.value}},
        ),
        backend=FakeBackend([*turns, [mock_llm_chunk(content="Done.")]]),
    )
    loop.tool_manager._all_tools["stub_tool"] = tool_class
    loop._completion_evidence_sink = sink
    return loop


class _ReceiptSink:
    """Records receipts and the observation timeline in arrival order."""

    def __init__(self) -> None:
        self.receipts: list[CompletionEvidenceReceipt] = []
        self.timeline: list[tuple[str, str]] = []

    def __call__(self, receipt: CompletionEvidenceReceipt) -> None:
        self.receipts.append(receipt)
        self.timeline.append(("receipt", receipt.occurrence_id))

    def observe(self, event: ToolResultEvent) -> None:
        if event.occurrence_id:
            self.timeline.append(("observed", event.occurrence_id))


async def _act_and_track(
    loop: AgentLoop, prompt: str, sink: _ReceiptSink
) -> list[BaseEvent]:
    events: list[BaseEvent] = []
    async for event in loop.act(prompt):
        events.append(event)
        if isinstance(event, ToolResultEvent):
            sink.observe(event)
    return events


@pytest.mark.asyncio
async def test_successful_call_emits_typed_receipt() -> None:
    """A successful tool call emits exactly one typed receipt at the seam.

    The receipt carries the executor-minted occurrence ID, the effective
    executed-args signature, and the model-visible serialization as its
    final text; the receipt's occurrence ID is identical to the result
    event's (the accumulator's collection path).
    """
    sink = _ReceiptSink()
    loop = _make_stub_loop(
        [
            [
                mock_llm_chunk(
                    content="Calling.",
                    tool_calls=[_stub_call("call_ok", '{"text": "hello"}')],
                )
            ]
        ],
        sink,
    )
    events = await _act_and_track(loop, "Go", sink)

    assert loop.stats.tool_calls_succeeded == 1
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(results) == 1
    assert len(sink.receipts) == 1
    receipt = sink.receipts[0]
    assert isinstance(receipt, CompletionEvidenceReceipt)
    assert receipt.tool_call_id == "call_ok"
    assert receipt.tool_name == "stub_tool"
    assert receipt.status == "success"
    assert receipt.occurrence_id
    assert receipt.executed_args == {"text": "hello"}
    assert "hello" in receipt.final_text
    assert receipt.tool_output is not None
    assert receipt.tool_output.get("message") == "hello"
    assert receipt.duration_ms is not None
    # Round-9: identical occurrence ID across the callback payload and the
    # collected result event.
    assert receipt.occurrence_id == results[0].occurrence_id


@pytest.mark.asyncio
async def test_failed_call_receipt_distinguishes_failure() -> None:
    """A raising tool emits a failure receipt carrying the error text."""
    sink = _ReceiptSink()

    class _ExplodingTool(FakeTool):
        async def run(
            self, args: FakeToolArgs, ctx: Any = None
        ) -> AsyncGenerator[ToolStreamEvent | FakeToolResult, None]:
            raise RuntimeError("tool exploded")
            yield  # pragma: no cover

    loop = _make_stub_loop(
        [[mock_llm_chunk(content="Calling.", tool_calls=[_stub_call("call_fail")])]],
        sink,
        tool_class=_ExplodingTool,
    )
    await _act_and_track(loop, "Go", sink)

    assert len(sink.receipts) == 1
    receipt = sink.receipts[0]
    assert receipt.status == "failure"
    assert "tool exploded" in receipt.final_text
    assert receipt.tool_output is None
    assert receipt.occurrence_id


@pytest.mark.asyncio
async def test_result_cancelled_receipt_distinguishes_cancellation() -> None:
    """A tool returning a cancelled result emits a cancelled receipt."""
    sink = _ReceiptSink()
    loop = _make_stub_loop(
        [[mock_llm_chunk(content="Calling.", tool_calls=[_stub_call("call_cancel")])]],
        sink,
        tool_class=_CancellingFakeTool,
    )
    await _act_and_track(loop, "Go", sink)

    assert len(sink.receipts) == 1
    receipt = sink.receipts[0]
    assert receipt.status == "cancelled"
    assert receipt.occurrence_id


@pytest.mark.asyncio
async def test_post_hook_replacement_is_reflected_in_receipt_final_text() -> None:
    """The receipt's final text is the post-hook text, after replacement."""

    async def replacing_hooks(
        tool_call: Any, **kwargs: Any
    ) -> AsyncGenerator[HookTextReplacement, None]:
        yield HookTextReplacement(text="replaced by hook")

    sink = _ReceiptSink()
    loop = _make_stub_loop(
        [[mock_llm_chunk(content="Calling.", tool_calls=[_stub_call("call_hook")])]],
        sink,
    )
    loop._run_post_tool_hooks = replacing_hooks  # type: ignore[method-assign]
    await _act_and_track(loop, "Go", sink)

    assert len(sink.receipts) == 1
    assert sink.receipts[0].final_text == "replaced by hook"


@pytest.mark.asyncio
async def test_receipt_final_text_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Receipt text and payloads pass the recorder's redaction pass."""
    name, value = "CHARTREUX_TEST_CREDENTIAL_KEY", "exfil-credential-0123456789xyz"
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(sr, "_read_dotenv_entries", lambda: {name: value})
    monkeypatch.setattr(sr, "_catalog_env_var_names", lambda: frozenset())
    sr.reset_cache()

    sink = _ReceiptSink()
    loop = _make_stub_loop(
        [
            [
                mock_llm_chunk(
                    content="Calling.",
                    tool_calls=[_stub_call("call_secret", f'{{"text": "{value}"}}')],
                )
            ]
        ],
        sink,
    )
    await _act_and_track(loop, "Go", sink)

    assert len(sink.receipts) == 1
    receipt = sink.receipts[0]
    assert value not in receipt.final_text
    assert sr.REDACTED_PLACEHOLDER in receipt.final_text
    assert receipt.tool_output is not None
    assert value not in str(receipt.tool_output)


@pytest.mark.asyncio
async def test_bash_start_job_id_rides_receipt() -> None:
    """Managed-job IDs from bash_start-shaped results ride the receipt."""

    class _JobStartTool(FakeTool):
        async def run(
            self, args: FakeToolArgs, ctx: Any = None
        ) -> AsyncGenerator[ToolStreamEvent | FakeToolResult, None]:
            yield _JobLikeResult(
                message="started", job={"job_id": "job-7", "state": "running"}
            )

    sink = _ReceiptSink()
    loop = _make_stub_loop(
        [[mock_llm_chunk(content="Calling.", tool_calls=[_stub_call("call_job")])]],
        sink,
        tool_class=_JobStartTool,
    )
    await _act_and_track(loop, "Go", sink)

    assert len(sink.receipts) == 1
    assert sink.receipts[0].bash_job_ids == ("job-7",)


@pytest.mark.asyncio
async def test_no_receipt_for_never_permission_call() -> None:
    """A call denied before execution (permission NEVER) emits no receipt."""
    sink = _ReceiptSink()
    loop = _make_stub_loop(
        [[mock_llm_chunk(content="Calling.", tool_calls=[_stub_call("call_never")])]],
        sink,
        tool_permission=ToolPermission.NEVER,
    )
    await _act_and_track(loop, "Go", sink)

    assert sink.receipts == []


@pytest.mark.asyncio
async def test_sink_failure_never_breaks_the_tool_flow() -> None:
    """A raising sink is swallowed: the tool still succeeds and records."""

    def exploding_sink(receipt: CompletionEvidenceReceipt) -> None:
        raise RuntimeError("sink boom")

    sink = _ReceiptSink()
    loop = _make_stub_loop(
        [[mock_llm_chunk(content="Calling.", tool_calls=[_stub_call("call_sink")])]],
        sink,
    )
    loop._completion_evidence_sink = exploding_sink
    events: list[BaseEvent] = []
    async for event in loop.act("Go"):
        events.append(event)

    assert loop.stats.tool_calls_succeeded == 1
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(results) == 1
    message = next(m for m in loop.messages if m.tool_call_id == "call_sink")
    assert "fake tool executed" in (message.content or "")


@pytest.mark.asyncio
async def test_reused_tool_call_id_resolves_only_to_its_own_invocation() -> None:
    """Round-9: two turns reusing one tool_call_id — a failed check followed
    by a successful retry — each receipt resolves ONLY to its originating
    invocation, even though the callback fires before the accumulator
    observes the queued result (the timeline records the receipt first).
    """
    sink = _ReceiptSink()
    attempts = {"count": 0}

    class _FlakyTool(FakeTool):
        async def run(
            self, args: FakeToolArgs, ctx: Any = None
        ) -> AsyncGenerator[ToolStreamEvent | FakeToolResult, None]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("first check failed")
            yield FakeToolResult(message="retry ok")

    loop = _make_stub_loop(
        [
            [mock_llm_chunk(content="Try.", tool_calls=[_stub_call("reused")])],
            [mock_llm_chunk(content="Retry.", tool_calls=[_stub_call("reused")])],
        ],
        sink,
        tool_class=_FlakyTool,
    )
    events = await _act_and_track(loop, "Go", sink)

    assert len(sink.receipts) == 2
    failure_receipt = next(r for r in sink.receipts if r.status == "failure")
    success_receipt = next(r for r in sink.receipts if r.status == "success")
    # Same tool_call_id (reused across turns), distinct occurrence IDs.
    assert failure_receipt.tool_call_id == success_receipt.tool_call_id == "reused"
    assert failure_receipt.occurrence_id != success_receipt.occurrence_id

    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(results) == 2
    failed_event = next(e for e in results if e.error is not None)
    succeeded_event = next(e for e in results if e.error is None)
    assert failed_event.tool_call_id == succeeded_event.tool_call_id == "reused"
    # Each receipt resolves ONLY to its originating invocation.
    assert failure_receipt.occurrence_id == failed_event.occurrence_id
    assert success_receipt.occurrence_id == succeeded_event.occurrence_id

    # Callback-before-observation ordering: for each occurrence the receipt
    # was recorded before the result event was observed off the queue.
    for occurrence_id in (failure_receipt.occurrence_id, success_receipt.occurrence_id):
        marks = [kind for kind, occ in sink.timeline if occ == occurrence_id]
        assert marks == ["receipt", "observed"]
