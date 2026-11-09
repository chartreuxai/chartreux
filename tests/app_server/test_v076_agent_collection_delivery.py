"""P2 chunk 2d: digest/full delivery states (marking, refusal, coexistence).

Covers the delivery-state semantics of the two fetch routes
(``get_agent_result`` / ``wait_for_agent``, marking SHARED across routes):

- RunResults hosts the keyword-only ``mode``/``force`` params at ``get_result``
  and ``wait``; the legacy delegate contract (raw ``TaskResult | None`` /
  ``WaitForAgentResult``, the perf call-site shapes) is preserved unchanged
  because ``mode=None`` is the legacy path.
- The marking tracks (persisted-conversation-id, agent_id, run_id,
  representation) as a per-message ENVELOPE key committed atomically with the
  transcript write (session_logger); it survives restart via transcript
  replay (``SessionLogger.delivery_state_scan`` over the live message list,
  or over the persisted records when no live list is bound).
- MECHANICAL REFUSAL LIFT, pinned semantics: the marking PERSISTS; the refusal
  predicate is "delivered AND no compaction boundary between that delivery and
  this fetch"; a post-lift re-delivery writes a fresh marking and re-arms.
- COEXISTENCE: a persisted marking plus an evicted in-memory result both
  authorize "nothing left to deliver"; a forced refetch on an evicted result
  surfaces ``AgentResultExpiredError``.
- Telemetry: digest/full deliveries, post-lift re-deliveries, refusals, per
  route, with replay-weighted payload bytes (content-free, P3 conventions).

Harness: ``tests/app_server/_subagent_test_support.py`` backends plus the
session-persistence fixtures for the transcript-atomicity nodes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import (
    FunctionCall,
    LLMMessage,
    PersistedToolResult,
    Role,
    ToolCall,
)
from chartreux.core.session.session_loader import MESSAGES_FILENAME
from chartreux.core.session.session_logger import (
    RESULT_DELIVERY_KEY,
    RESULT_DELIVERY_OUTPUT_KEY,
    SessionLogger,
    scan_delivery_state,
)
from chartreux.core.session_types import AgentStats
from chartreux.core.subagents import (
    AgentResultExpiredError,
    RunStatus,
    TaskResult,
    WaitForAgentResult,
    WaitStatus,
)
from chartreux.core.tools.base import BaseToolState, InvokeContext, ToolError
from chartreux.core.tools.builtins.get_agent_result import (
    GetAgentResult,
    GetAgentResultArgs,
    GetAgentResultConfig,
)
from chartreux.core.tools.builtins.wait_for_agent import (
    WaitForAgent,
    WaitForAgentArgs,
    WaitForAgentConfig,
    WaitForAgentDeliveryResult,
)
from tests.app_server._subagent_test_support import (
    GatedRegistryLock,
    _ManualClock,
    _record,
    _retention_registry,
    _run,
    _stored_result,
)
from tests.session.persistence._session_persistence_support import (
    mock_tool_manager as mock_tool_manager,
    mock_vibe_config as mock_vibe_config,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)

_CONVERSATION = "conv-1"


def _marking_record(
    *,
    agent_id: str = "agent-1",
    run_id: str = "run-1",
    representation: str = "digest",
    conversation_id: str = _CONVERSATION,
) -> dict[str, Any]:
    return {
        RESULT_DELIVERY_KEY: {
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "representation": representation,
        }
    }


class _ScanLogger:
    """A duck-typed session logger over raw persisted records + a live tail."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def delivery_state_scan(self, live_messages: Any = None) -> Any:
        records = list(self.records)
        if live_messages is not None:
            records.extend(
                SessionLogger._message_record(message, allow_archived=True)
                for message in live_messages
                if message.role != Role.system
            )
        return scan_delivery_state(records)


class _StubLoop:
    def __init__(self, session_logger: _ScanLogger, session_id: str) -> None:
        self.session_logger = session_logger
        self.session_id = session_id
        self.messages: list[LLMMessage] = []


class _StubRoot:
    def __init__(self, loop: _StubLoop) -> None:
        self.agent_loop = loop


def _registry_with_result(
    agent_id: str = "agent-1", run_id: str = "run-1"
) -> SessionRuntimeRegistry:
    registry = _retention_registry(_ManualClock())
    registry._result_store[(agent_id, run_id)] = _stored_result(agent_id, run_id)
    return registry


def _bind_scan(
    registry: SessionRuntimeRegistry,
    records: list[dict[str, Any]],
    *,
    session_id: str = _CONVERSATION,
) -> _StubLoop:
    loop = _StubLoop(_ScanLogger(records), session_id)
    registry._root = _StubRoot(loop)  # type: ignore[assignment]
    return loop


def _delivery_message(
    *,
    conversation_id: str = _CONVERSATION,
    agent_id: str = "agent-1",
    run_id: str = "run-1",
    representation: str = "digest",
    call_id: str = "call-fetch",
) -> LLMMessage:
    return LLMMessage(
        role=Role.tool,
        tool_call_id=call_id,
        name="get_agent_result",
        tool_result=PersistedToolResult(
            output={
                "result": {
                    "response": "big payload",
                    "turns_used": 1,
                    "completed": True,
                },
                "message": "Agent result is available",
                RESULT_DELIVERY_OUTPUT_KEY: {
                    "conversation_id": conversation_id,
                    "agent_id": agent_id,
                    "run_id": run_id,
                    "representation": representation,
                },
            }
        ),
    )


def _fetch_assistant(call_id: str = "call-fetch") -> LLMMessage:
    return LLMMessage(
        role=Role.assistant,
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                index=0,
                function=FunctionCall(
                    name="get_agent_result",
                    arguments=json.dumps({"agent_id": "agent-1"}),
                ),
            )
        ],
    )


class TestLegacyContractPreserved:
    """The registry's public fetch API keeps its raw contract (mode=None)."""

    @pytest.mark.asyncio
    async def test_repeated_legacy_fetches_return_the_result(self) -> None:
        registry = _registry_with_result()
        first = await registry.get_agent_result("agent-1", "run-1")
        second = await registry.get_agent_result("agent-1", "run-1")
        assert first is not None and second is not None
        assert first == second
        # No delivery marking was registered on the legacy path.
        assert getattr(registry, "_delivery_markings", {}) == {}

    @pytest.mark.asyncio
    async def test_perf_call_site_shape_stays_valid(self) -> None:
        # tests/perf call shape: wait_for_agent(agent_id, run_id, timeout=...).
        registry = _registry_with_result()
        disposition = await registry.wait_for_agent("agent-1", "run-1", timeout=1)
        assert isinstance(disposition, WaitForAgentResult)
        assert disposition.status is WaitStatus.COMPLETED
        assert disposition.result is not None
        again = await registry.wait_for_agent("agent-1", "run-1", timeout=1)
        assert again.status is WaitStatus.COMPLETED


class TestGetRouteDeliveryStates:
    """The get route's mode/force delivery states."""

    @pytest.mark.asyncio
    async def test_first_mode_fetch_delivers_with_marking(self) -> None:
        registry = _registry_with_result()
        outcome = await registry._run_results.get_result(
            "agent-1", "run-1", mode="digest"
        )
        assert outcome.status == "delivered"
        assert outcome.result is not None
        assert outcome.marking == {
            "conversation_id": "",
            "agent_id": "agent-1",
            "run_id": "run-1",
            "representation": "digest",
        }
        assert outcome.post_lift is False
        assert ("agent-1", "run-1") in registry._collected_results

    @pytest.mark.asyncio
    async def test_second_same_mode_fetch_is_refused(self) -> None:
        registry = _registry_with_result()
        await registry._run_results.get_result("agent-1", "run-1", mode="digest")
        refused = await registry._run_results.get_result(
            "agent-1", "run-1", mode="digest"
        )
        assert refused.status == "refused"
        assert refused.result is None and refused.marking is None
        # The refusal message carries the force affordance and the
        # refetch-after-compaction disclosure.
        assert "force=True" in refused.message
        assert "compaction" in refused.message

    @pytest.mark.asyncio
    async def test_forced_refetch_redelivers_and_rearms(self) -> None:
        registry = _registry_with_result()
        await registry._run_results.get_result("agent-1", "run-1", mode="digest")
        forced = await registry._run_results.get_result(
            "agent-1", "run-1", mode="digest", force=True
        )
        assert forced.status == "delivered" and forced.forced is True
        assert forced.result is not None
        again = await registry._run_results.get_result(
            "agent-1", "run-1", mode="digest"
        )
        assert again.status == "refused"

    @pytest.mark.asyncio
    async def test_other_representation_shares_the_run_marking(self) -> None:
        registry = _registry_with_result()
        await registry._run_results.get_result("agent-1", "run-1", mode="digest")
        full = await registry._run_results.get_result("agent-1", "run-1", mode="full")
        assert full.status == "delivered"
        assert full.marking is not None and full.marking["representation"] == "full"
        # One marking entry for the run, both representations recorded.
        entry = registry._delivery_markings[("agent-1", "run-1")]
        assert set(entry) == {"digest", "full"}

    @pytest.mark.asyncio
    async def test_pending_when_running_marks_nothing(self) -> None:
        registry = _retention_registry(_ManualClock())
        record = _record("agent-1", clock=_ManualClock())
        run = _run(
            record, "run-1", started_at=0.0, status=RunStatus.RUNNING, done=False
        )
        record.current_run = run
        registry._agent_records["agent-1"] = record
        outcome = await registry._run_results.get_result(
            "agent-1", "run-1", mode="digest"
        )
        assert outcome.status == "pending"
        assert getattr(registry, "_delivery_markings", {}) == {}
        assert ("agent-1", "run-1") not in registry._collected_results


class TestSimultaneousFetchesShareTheMarking:
    @pytest.mark.asyncio
    async def test_simultaneous_digest_and_full_share_one_marking(self) -> None:
        registry = _registry_with_result("agent-2", "run-2")
        results = registry._run_results
        pair = await asyncio.gather(
            results.get_result("agent-2", "run-2", mode="digest"),
            results.get_result("agent-2", "run-2", mode="full"),
        )
        assert [item.status for item in pair] == ["delivered", "delivered"]
        # The in-memory collection marker is shared: exactly one entry.
        assert ("agent-2", "run-2") in registry._collected_results
        entry = registry._delivery_markings[("agent-2", "run-2")]
        assert set(entry) == {"digest", "full"}


class TestExpiryAndCoexistence:
    @pytest.mark.asyncio
    async def test_forced_refetch_on_evicted_raises_expired(self) -> None:
        registry = _registry_with_result("agent-3", "run-3")
        async with registry._registry_lock:
            registry._result_store.pop(("agent-3", "run-3"))
            registry._expired_results.add(("agent-3", "run-3"))
        with pytest.raises(AgentResultExpiredError):
            await registry._run_results.get_result(
                "agent-3", "run-3", mode="digest", force=True
            )
        # The raise registered no delivery marking.
        assert ("agent-3", "run-3") not in getattr(registry, "_delivery_markings", {})

    @pytest.mark.asyncio
    async def test_nonforced_refetch_on_evicted_without_marking_raises(self) -> None:
        # Pinned expiry semantics preserved: without a persisted marking the
        # mode-bearing fetch raises exactly like the legacy path.
        registry = _registry_with_result("agent-3", "run-3")
        async with registry._registry_lock:
            registry._result_store.pop(("agent-3", "run-3"))
            registry._expired_results.add(("agent-3", "run-3"))
        with pytest.raises(AgentResultExpiredError):
            await registry._run_results.get_result("agent-3", "run-3", mode="digest")

    @pytest.mark.asyncio
    async def test_persisted_marking_plus_evicted_result_authorizes_nothing_left(
        self,
    ) -> None:
        registry = _registry_with_result("agent-3", "run-3")
        _bind_scan(registry, [_marking_record(agent_id="agent-3", run_id="run-3")])
        async with registry._registry_lock:
            registry._result_store.pop(("agent-3", "run-3"))
            registry._expired_results.add(("agent-3", "run-3"))
            # The in-memory marker dies with the result...
            registry._collected_results.discard(("agent-3", "run-3"))
        outcome = await registry._run_results.get_result(
            "agent-3", "run-3", mode="digest"
        )
        assert outcome.status == "nothing_left"
        assert "Nothing left to deliver" in outcome.message
        # ...while the persisted marking still authorizes it; a forced refetch
        # surfaces the typed expiry error instead.
        with pytest.raises(AgentResultExpiredError):
            await registry._run_results.get_result(
                "agent-3", "run-3", mode="digest", force=True
            )

    @pytest.mark.asyncio
    async def test_forced_refetch_racing_expiry_never_half_marks(self) -> None:
        registry = _registry_with_result("agent-4", "run-4")
        lock = GatedRegistryLock(registry._registry_lock, gate_entry=1)
        registry._registry_lock = lock  # type: ignore[assignment]
        fetch = asyncio.create_task(
            registry._run_results.get_result(
                "agent-4", "run-4", mode="digest", force=True
            )
        )
        await asyncio.wait_for(lock.entered.wait(), 2)
        # The fetch is parked INSIDE its locked section; evict underneath it.
        registry._result_store.pop(("agent-4", "run-4"))
        registry._expired_results.add(("agent-4", "run-4"))
        lock.release.set()
        with pytest.raises(AgentResultExpiredError):
            await asyncio.wait_for(fetch, 2)
        # The racing forced fetch registered no marking and no collection.
        assert ("agent-4", "run-4") not in getattr(registry, "_delivery_markings", {})
        assert ("agent-4", "run-4") not in registry._collected_results


class TestRefusalLiftAndRearm:
    """The compaction boundary integration: refusal lift + re-arm."""

    @pytest.mark.asyncio
    async def test_boundary_after_persisted_marking_lifts_then_rearms(self) -> None:
        # Post-restart shape: fresh in-memory registry, persisted marking at
        # position 1, compaction boundary at position 2.
        registry = _registry_with_result("agent-5", "run-5")
        loop = _bind_scan(
            registry,
            [
                {"role": "user"},
                _marking_record(agent_id="agent-5", run_id="run-5"),
                {"role": "assistant", "context_boundary": "compaction"},
            ],
        )
        first = await registry._run_results.get_result(
            "agent-5", "run-5", mode="digest"
        )
        assert first.status == "delivered"
        assert first.post_lift is True  # re-delivery after the boundary
        # The fresh marking re-arms the refusal.
        second = await registry._run_results.get_result(
            "agent-5", "run-5", mode="digest"
        )
        assert second.status == "refused"
        assert loop.session_id == _CONVERSATION

    @pytest.mark.asyncio
    async def test_live_boundary_after_in_memory_delivery_lifts_in_session(
        self,
    ) -> None:
        registry = _registry_with_result("agent-6", "run-6")
        loop = _bind_scan(registry, [{"role": "user"}])
        first = await registry._run_results.get_result(
            "agent-6", "run-6", mode="digest"
        )
        assert first.status == "delivered"
        refused = await registry._run_results.get_result(
            "agent-6", "run-6", mode="digest"
        )
        assert refused.status == "refused"
        # The delivery's tool message lands in the live tail, then a
        # compaction boundary after it.
        loop.messages.append(_delivery_message(agent_id="agent-6", run_id="run-6"))
        loop.messages.append(
            LLMMessage(
                role=Role.assistant, content="summary", context_boundary="compaction"
            )
        )
        relifted = await registry._run_results.get_result(
            "agent-6", "run-6", mode="digest"
        )
        assert relifted.status == "delivered"
        assert relifted.post_lift is True
        # Re-armed again by the fresh marking.
        again = await registry._run_results.get_result(
            "agent-6", "run-6", mode="digest"
        )
        assert again.status == "refused"

    @pytest.mark.asyncio
    async def test_boundary_before_marking_does_not_lift(self) -> None:
        registry = _registry_with_result("agent-7", "run-7")
        _bind_scan(
            registry,
            [
                {"role": "assistant", "context_boundary": "compaction"},
                {"role": "user"},
                _marking_record(agent_id="agent-7", run_id="run-7"),
            ],
        )
        # The only boundary is BEFORE the marking (position 2, boundary at 0):
        # no boundary between that delivery and this fetch, so the persisted
        # marking arms the refusal immediately.
        outcome = await registry._run_results.get_result(
            "agent-7", "run-7", mode="digest"
        )
        assert outcome.status == "refused"
        # A forced fetch re-delivers and writes a fresh marking.
        forced = await registry._run_results.get_result(
            "agent-7", "run-7", mode="digest", force=True
        )
        assert forced.status == "delivered" and forced.forced is True


class TestWaitRouteDeliveryStates:
    """The wait route shares the marking with the get route."""

    @pytest.mark.asyncio
    async def test_wait_mode_delivers_with_marking(self) -> None:
        registry = _registry_with_result("agent-8", "run-8")
        outcome = await registry._run_results.wait(
            "agent-8", "run-8", timeout=1, mode="digest"
        )
        assert outcome.status == "delivered"
        assert outcome.disposition is not None
        assert outcome.disposition.status is WaitStatus.COMPLETED
        assert outcome.marking is not None
        assert outcome.marking["representation"] == "digest"

    @pytest.mark.asyncio
    async def test_wait_mode_refusal_shares_the_get_marking(self) -> None:
        registry = _registry_with_result("agent-8", "run-8")
        await registry._run_results.get_result("agent-8", "run-8", mode="digest")
        refused = await registry._run_results.wait(
            "agent-8", "run-8", timeout=1, mode="digest"
        )
        assert refused.status == "refused"
        assert "force=True" in refused.message

    @pytest.mark.asyncio
    async def test_wait_still_running_passes_through_unmarked(self) -> None:
        registry = _retention_registry(_ManualClock())
        record = _record("agent-9", clock=_ManualClock())
        run = _run(
            record, "run-9", started_at=0.0, status=RunStatus.RUNNING, done=False
        )
        record.current_run = run
        registry._agent_records["agent-9"] = record
        outcome = await registry._run_results.wait(
            "agent-9", "run-9", timeout=0.01, mode="digest"
        )
        assert outcome.status == "passthrough"
        assert outcome.disposition is not None
        assert outcome.disposition.status is WaitStatus.STILL_RUNNING
        assert getattr(registry, "_delivery_markings", {}) == {}


class TestMarkingPersistence:
    """The marking is a per-message envelope key committed with the transcript."""

    def test_message_record_hoists_delivery_envelope(self) -> None:
        message = _delivery_message()
        record = SessionLogger._message_record(message)
        assert record[RESULT_DELIVERY_KEY] == {
            "conversation_id": _CONVERSATION,
            "agent_id": "agent-1",
            "run_id": "run-1",
            "representation": "digest",
        }
        # The payload copy is trimmed from the persisted output; the live
        # message keeps it.
        assert RESULT_DELIVERY_OUTPUT_KEY not in record["tool_result"]["output"]
        assert RESULT_DELIVERY_OUTPUT_KEY in message.tool_result.output  # type: ignore[union-attr]

    def test_malformed_delivery_payload_is_never_hoisted(self) -> None:
        message = LLMMessage(
            role=Role.tool,
            tool_call_id="c",
            tool_result=PersistedToolResult(
                output={RESULT_DELIVERY_OUTPUT_KEY: {"agent_id": "agent-1"}}
            ),
        )
        record = SessionLogger._message_record(message)
        assert RESULT_DELIVERY_KEY not in record
        assert RESULT_DELIVERY_OUTPUT_KEY in record["tool_result"]["output"]
        # Non-tool messages are untouched.
        user = SessionLogger._message_record(LLMMessage(role=Role.user, content="hi"))
        assert RESULT_DELIVERY_KEY not in user

    def test_scan_delivery_state_positions(self) -> None:
        scan = scan_delivery_state([
            {"role": "user"},
            _marking_record(),
            {"role": "assistant", "context_boundary": "compaction"},
            {"role": "user"},
        ])
        assert len(scan.markings) == 1
        assert scan.markings[0].position == 1
        assert scan.markings[0].representation == "digest"
        assert scan.compaction_boundaries == (2,)
        assert scan.position_count == 4

    @pytest.mark.asyncio
    async def test_save_commits_marking_with_the_transcript_write(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: Any,
        mock_tool_manager: Any,
    ) -> None:
        logger = SessionLogger(session_config, _CONVERSATION)
        messages = [
            LLMMessage(role=Role.user, content="Delegate"),
            _fetch_assistant(),
            _delivery_message(),
            LLMMessage(
                role=Role.assistant, content="summary", context_boundary="compaction"
            ),
        ]
        await logger.save_interaction(
            messages, AgentStats(steps=1), mock_vibe_config, mock_tool_manager
        )
        # Crash atomicity: the marking IS the envelope key of the tool
        # record's line — one marking, on that line, no sidecar anywhere.
        session_dir = logger.session_dir
        assert session_dir is not None
        lines = [
            json.loads(line)
            for line in (session_dir / MESSAGES_FILENAME).read_text().splitlines()
        ]
        marked = [line for line in lines if RESULT_DELIVERY_KEY in line]
        assert len(marked) == 1
        tool_line = next(line for line in lines if line.get("role") == "tool")
        assert marked[0] is tool_line
        assert RESULT_DELIVERY_OUTPUT_KEY not in tool_line["tool_result"]["output"]
        assert not list(session_dir.glob("*marking*"))

    @pytest.mark.asyncio
    async def test_restart_replay_scans_the_persisted_marking(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: Any,
        mock_tool_manager: Any,
    ) -> None:
        logger = SessionLogger(session_config, _CONVERSATION)
        messages = [
            LLMMessage(role=Role.user, content="Delegate"),
            _fetch_assistant(),
            _delivery_message(),
            LLMMessage(
                role=Role.assistant, content="summary", context_boundary="compaction"
            ),
        ]
        await logger.save_interaction(
            messages, AgentStats(steps=1), mock_vibe_config, mock_tool_manager
        )
        session_dir = logger.session_dir
        # A FRESH logger over the same persisted transcript (restart replay).
        resumed = SessionLogger(session_config, _CONVERSATION, session_dir=session_dir)
        scan = resumed.delivery_state_scan([])
        assert len(scan.markings) == 1
        marking = scan.markings[0]
        assert marking.conversation_id == _CONVERSATION
        assert marking.agent_id == "agent-1"
        assert marking.run_id == "run-1"
        assert marking.representation == "digest"
        assert marking.position == 2
        assert scan.compaction_boundaries == (3,)

    @pytest.mark.asyncio
    async def test_live_list_scans_in_one_position_space_without_double_count(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: Any,
        mock_tool_manager: Any,
    ) -> None:
        logger = SessionLogger(session_config, _CONVERSATION)
        messages = [
            LLMMessage(role=Role.user, content="Delegate"),
            _fetch_assistant(),
            _delivery_message(),
        ]
        await logger.save_interaction(
            messages, AgentStats(steps=1), mock_vibe_config, mock_tool_manager
        )
        # The loop's full live list already contains the saved prefix; the
        # scan runs over the live list alone so no saved record is counted
        # twice. The unsaved tail (a compaction boundary) counts immediately.
        live = [
            *messages,
            LLMMessage(
                role=Role.assistant, content="summary", context_boundary="compaction"
            ),
        ]
        scan = logger.delivery_state_scan(live)
        assert len(scan.markings) == 1
        assert scan.markings[0].position == 2
        assert scan.compaction_boundaries == (3,)
        assert scan.position_count == 4

    def test_disabled_logger_scans_empty(self) -> None:
        disabled = SessionLogger(
            SessionLoggingConfig(
                enabled=False,
                save_dir="/tmp/chartreux-test-disabled",
                session_prefix="t",
            ),
            "disabled",
        )
        scan = disabled.delivery_state_scan([])
        assert scan.markings == () and scan.compaction_boundaries == ()


class TestDeliveryTelemetry:
    """Content-free delivery telemetry per route (P3 measurement fields)."""

    @pytest.mark.asyncio
    async def test_delivered_and_refused_events_carry_route_and_bytes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = _registry_with_result("agent-10", "run-10")
        with caplog.at_level(
            logging.INFO, logger="chartreux.app_server._subagent_results"
        ):
            await registry._run_results.get_result("agent-10", "run-10", mode="digest")
            await registry._run_results.get_result("agent-10", "run-10", mode="digest")
            await registry._run_results.wait(
                "agent-10", "run-10", timeout=1, mode="full"
            )
        events = [
            record
            for record in caplog.records
            if record.getMessage().startswith("result_delivery")
        ]
        assert len(events) == 3
        delivered_get = next(
            r
            for r in events
            if "route=get_agent_result" in r.getMessage()
            and "status=delivered" in r.getMessage()
        )
        assert "representation=digest" in delivered_get.getMessage()
        assert "payload_bytes=" in delivered_get.getMessage()
        refused = next(r for r in events if "status=refused" in r.getMessage())
        assert "route=get_agent_result" in refused.getMessage()
        assert "payload_bytes=0" in refused.getMessage()
        delivered_wait = next(
            r for r in events if "route=wait_for_agent" in r.getMessage()
        )
        assert "representation=full" in delivered_wait.getMessage()

    @pytest.mark.asyncio
    async def test_post_lift_redelivery_is_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = _registry_with_result("agent-11", "run-11")
        _bind_scan(
            registry,
            [
                {"role": "user"},
                _marking_record(agent_id="agent-11", run_id="run-11"),
                {"role": "assistant", "context_boundary": "compaction"},
            ],
        )
        with caplog.at_level(
            logging.INFO, logger="chartreux.app_server._subagent_results"
        ):
            outcome = await registry._run_results.get_result(
                "agent-11", "run-11", mode="digest"
            )
        assert outcome.status == "delivered" and outcome.post_lift is True
        events = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("result_delivery")
        ]
        assert any("status=delivered" in e and "post_lift=True" in e for e in events)


class TestToolSurface:
    """The fetch tools' keyword-only mode/force params over a real registry."""

    def _ctx(self, registry: SessionRuntimeRegistry) -> InvokeContext:
        return InvokeContext(
            tool_call_id="call-fetch",
            session_id=_CONVERSATION,
            subagent_manager=registry,  # type: ignore[arg-type]
        )

    @staticmethod
    def _get_tool() -> GetAgentResult:
        return GetAgentResult(
            config_getter=lambda: GetAgentResultConfig(), state=BaseToolState()
        )

    @staticmethod
    def _wait_tool() -> WaitForAgent:
        return WaitForAgent(
            config_getter=lambda: WaitForAgentConfig(), state=BaseToolState()
        )

    @pytest.mark.asyncio
    async def test_get_tool_delivers_with_conversation_marking(self) -> None:
        registry = _registry_with_result("agent-12", "run-12")
        tool = self._get_tool()
        results = [
            item
            async for item in tool.run(
                GetAgentResultArgs(agent_id="agent-12", run_id="run-12"),
                self._ctx(registry),
            )
        ]
        assert len(results) == 1
        assert results[0].result is not None
        assert results[0].delivery == {
            "conversation_id": _CONVERSATION,
            "agent_id": "agent-12",
            "run_id": "run-12",
            "representation": "digest",
        }

    @pytest.mark.asyncio
    async def test_get_tool_refusal_is_a_tool_error(self) -> None:
        registry = _registry_with_result("agent-12", "run-12")
        tool = self._get_tool()
        ctx = self._ctx(registry)
        _ = [
            item
            async for item in tool.run(
                GetAgentResultArgs(agent_id="agent-12", run_id="run-12"), ctx
            )
        ]
        with pytest.raises(ToolError, match="force=True"):
            _ = [
                item
                async for item in tool.run(
                    GetAgentResultArgs(agent_id="agent-12", run_id="run-12"), ctx
                )
            ]

    @pytest.mark.asyncio
    async def test_wait_tool_delivers_marked_disposition(self) -> None:
        registry = _registry_with_result("agent-13", "run-13")
        tool = self._wait_tool()
        results = [
            item
            async for item in tool.run(
                WaitForAgentArgs(agent_id="agent-13", run_id="run-13", timeout=1),
                self._ctx(registry),
            )
        ]
        assert len(results) == 1
        assert isinstance(results[0], WaitForAgentDeliveryResult)
        assert results[0].status is WaitStatus.COMPLETED
        assert results[0].delivery == {
            "conversation_id": _CONVERSATION,
            "agent_id": "agent-13",
            "run_id": "run-13",
            "representation": "digest",
        }

    @pytest.mark.asyncio
    async def test_wait_tool_refusal_is_a_tool_error(self) -> None:
        registry = _registry_with_result("agent-13", "run-13")
        tool = self._wait_tool()
        ctx = self._ctx(registry)
        _ = [
            item
            async for item in tool.run(
                WaitForAgentArgs(agent_id="agent-13", run_id="run-13", timeout=1), ctx
            )
        ]
        with pytest.raises(ToolError, match="force=True"):
            _ = [
                item
                async for item in tool.run(
                    WaitForAgentArgs(agent_id="agent-13", run_id="run-13", timeout=1),
                    ctx,
                )
            ]

    @pytest.mark.asyncio
    async def test_legacy_manager_without_collaborator_keeps_raw_fetch(self) -> None:
        class _LegacyManager:
            async def get_agent_result(
                self, agent_id: str, run_id: str | None = None
            ) -> TaskResult | None:
                return TaskResult(response="raw", turns_used=1, completed=True)

            async def wait_for_agent(
                self,
                agent_id: str,
                run_id: str | None = None,
                *,
                timeout: float | None = None,
            ) -> WaitForAgentResult:
                return WaitForAgentResult(
                    status=WaitStatus.COMPLETED,
                    result=TaskResult(response="raw", turns_used=1, completed=True),
                )

        ctx = InvokeContext(
            tool_call_id="c",
            session_id=_CONVERSATION,
            subagent_manager=_LegacyManager(),  # type: ignore[arg-type]
        )
        get_results = [
            item
            async for item in self._get_tool().run(
                GetAgentResultArgs(agent_id="agent-1"), ctx
            )
        ]
        assert get_results[0].result is not None
        assert get_results[0].delivery is None
        wait_results = [
            item
            async for item in self._wait_tool().run(
                WaitForAgentArgs(agent_id="agent-1"), ctx
            )
        ]
        assert isinstance(wait_results[0], WaitForAgentResult)
        assert not isinstance(wait_results[0], WaitForAgentDeliveryResult)
