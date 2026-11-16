"""v0.8.0 A-P2 chunk 1: strict parser, fail-closed matrix, decided-call
memory, the typed SYNTAX discriminator, and the _should_execute_tool hook.

v0.8.2 W7: durable smart-approve usage records (tee sink + attribution),
structured gate-outcome log lines, and uphold-marker feedback.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
import logging
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

from pydantic import BaseModel
import pytest

from chartreux.core.agent_loop import _smart_approve as smart_approve_module
from chartreux.core.agent_loop._loop_types import ToolExecutionResponse
from chartreux.core.agent_loop._safety_authority import (
    DecidedCallIdentity,
    SafetyLevel,
    SessionAuthorityState,
)
from chartreux.core.agent_loop._smart_approve import (
    GATE_CAP,
    GATE_CLASSIFIER_ERROR,
    GATE_DECIDED_REPLAY,
    GATE_LEVEL_NOT_SMART,
    GATE_SYNTAX,
    GATE_UNPARSEABLE,
    OVERRIDE_FEEDBACK_MARKER,
    UPHOLD_FEEDBACK_MARKER,
    SmartApproveSession,
    _model_classifier,
    parse_classifier_verdict,
)
from chartreux.core.llm import utility_completion
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm_models import LLMChunk, LLMMessage, LLMUsage, Role
from chartreux.core.tools.base import BaseTool, ToolPermission
from chartreux.core.tools.builtins._shell_diagnostics import DenialClass, Diagnostic
from chartreux.core.tools.builtins._shell_permission_resolver import _denied
from chartreux.core.tools.models import PermissionContext
from chartreux.core.usage import (
    UsageAttribution,
    UsageOutcome,
    UsagePrices,
    UsagePurpose,
    UsageRecord,
    UsageState,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


class _Args(BaseModel):
    command: str


class TestStrictParser:
    """The 15-case parser plan: exactly {verdict, rationale}, one optional
    fence stripped, duplicate keys rejected, verdict by exact equality.
    """

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            # 1-2: the two valid verdicts.
            ('{"verdict": "override", "rationale": "in scope"}', "override"),
            ('{"verdict": "uphold", "rationale": "out of scope"}', "uphold"),
            # 3: one optional surrounding fence is stripped, then parsed.
            (
                '```json\n{"verdict": "override", "rationale": "fenced"}\n```',
                "override",
            ),
            # 4: prose around the object upholds.
            ('Here you go: {"verdict": "override", "rationale": "x"}', None),
            # 5: two objects uphold.
            (
                '{"verdict": "uphold", "rationale": "a"} {"verdict": "override", "rationale": "b"}',
                None,
            ),
            # 6: duplicate keys are rejected (last-wins must not apply).
            ('{"verdict": "uphold", "verdict": "override", "rationale": "dup"}', None),
            # 7: missing rationale upholds.
            ('{"verdict": "override"}', None),
            # 8: an extra key upholds.
            ('{"verdict": "uphold", "rationale": "x", "extra": 1}', None),
            # 9: verdict is case-sensitive (exact equality).
            ('{"verdict": "Override", "rationale": "x"}', None),
            # 10: verdict with padding is not exact.
            ('{"verdict": "override ", "rationale": "x"}', None),
            # 11: a non-string verdict upholds.
            ('{"verdict": true, "rationale": "x"}', None),
            # 12: a non-object response upholds.
            ('["override"]', None),
            # 13: empty / whitespace / None uphold (refused or empty).
            ("", None),
            ("   \n  ", None),
            (None, None),
            # 14: a nested second fence does not parse after one strip.
            ('```\n```\n{"verdict": "override", "rationale": "x"}\n```\n```', None),
            # 15: a non-string rationale upholds.
            ('{"verdict": "uphold", "rationale": 42}', None),
        ],
    )
    def test_parser_plan(self, response: str | None, expected: str | None) -> None:
        verdict = parse_classifier_verdict(response)
        if expected is None:
            assert verdict is None
        else:
            assert verdict is not None
            assert verdict.verdict == expected
            assert isinstance(verdict.rationale, str)


class TestSyntaxDiscriminator:
    def test_syntax_diagnostic_carries_typed_marker(self) -> None:
        ctx = _denied(
            Diagnostic(
                "command",
                detail="unsupported",
                denial_class=DenialClass.SYNTAX,
                alternative="write well-formed quoting",
            )
        )
        assert ctx.permission == ToolPermission.NEVER
        assert ctx.denial_kind == "syntax"

    def test_policy_diagnostic_has_no_syntax_marker(self) -> None:
        ctx = _denied(Diagnostic("sensitive_command"))
        assert ctx.permission == ToolPermission.NEVER
        assert ctx.denial_kind is None


def _smart_state() -> SessionAuthorityState:
    return SessionAuthorityState(session_id="sess", level=SafetyLevel.SMART)


def _policy_ctx() -> PermissionContext:
    return PermissionContext(
        permission=ToolPermission.NEVER,
        reason="Shell path is outside the authorized workspace",
        denial_kind="out_of_root",
    )


class _StubClassifier:
    """Records calls and returns a canned response (or raises)."""

    def __init__(
        self, response: str | None = None, error: Exception | None = None
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        config: Any,
        system_prompt: str,
        user_content: str,
        accounting_sink: Any,
        usage_attribution: Any = None,
    ) -> str | None:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_content": user_content,
            "usage_attribution": usage_attribution,
        })
        if self.error is not None:
            raise self.error
        return self.response


class TestAdjudicationFailClosed:
    async def test_unparseable_response_upholds_and_records_error(self) -> None:
        stub = _StubClassifier(response="not json at all")
        session = SmartApproveSession(
            config_getter=build_test_vibe_config,
            authority_state=_smart_state(),
            classifier=stub,
        )
        verdict = await session.adjudicate(
            tool_name="bash",
            args=_Args(command="rm -rf /"),
            ctx=_policy_ctx(),
            messages=[],
            launch_intent=None,
        )
        assert verdict.verdict == "uphold"
        assert session.last_classifier_error is not None
        assert "unparseable" in session.last_classifier_error
        assert session.uphold_count == 1

    async def test_backend_error_upholds_and_records_error(self) -> None:
        stub = _StubClassifier(error=RuntimeError("backend down"))
        session = SmartApproveSession(
            config_getter=build_test_vibe_config,
            authority_state=_smart_state(),
            classifier=stub,
        )
        verdict = await session.adjudicate(
            tool_name="bash",
            args=_Args(command="ls"),
            ctx=_policy_ctx(),
            messages=[],
            launch_intent=None,
        )
        assert verdict.verdict == "uphold"
        assert session.last_classifier_error is not None
        assert "backend down" in session.last_classifier_error

    async def test_override_verdict_is_returned_and_counted(self) -> None:
        stub = _StubClassifier(
            response='{"verdict": "override", "rationale": "in scope"}'
        )
        session = SmartApproveSession(
            config_getter=build_test_vibe_config,
            authority_state=_smart_state(),
            classifier=stub,
        )
        verdict = await session.adjudicate(
            tool_name="bash",
            args=_Args(command="git status"),
            ctx=_policy_ctx(),
            messages=[],
            launch_intent=None,
        )
        assert verdict.verdict == "override"
        assert session.override_count == 1
        assert session.calls == 1

    async def test_non_smart_level_never_calls_classifier(self) -> None:
        stub = _StubClassifier(response='{"verdict": "override", "rationale": "x"}')
        state = SessionAuthorityState(session_id="sess", level=SafetyLevel.SAFE)
        session = SmartApproveSession(
            config_getter=build_test_vibe_config, authority_state=state, classifier=stub
        )
        verdict = await session.adjudicate(
            tool_name="bash",
            args=_Args(command="ls"),
            ctx=_policy_ctx(),
            messages=[],
            launch_intent=None,
        )
        assert verdict.verdict == "uphold"
        assert stub.calls == []

    async def test_syntax_refusal_never_reaches_classifier(self) -> None:
        stub = _StubClassifier(response='{"verdict": "override", "rationale": "x"}')
        session = SmartApproveSession(
            config_getter=build_test_vibe_config,
            authority_state=_smart_state(),
            classifier=stub,
        )
        syntax_ctx = PermissionContext(
            permission=ToolPermission.NEVER,
            reason="Command denied: unsupported shell syntax",
            denial_kind="syntax",
        )
        verdict = await session.adjudicate(
            tool_name="bash",
            args=_Args(command="echo $(unclosed"),
            ctx=syntax_ctx,
            messages=[],
            launch_intent=None,
        )
        assert verdict.verdict == "uphold"
        assert stub.calls == []
        assert session.calls == 0


class TestDecidedCallMemory:
    async def test_decided_call_is_never_redecided(self) -> None:
        stub = _StubClassifier(response='{"verdict": "override", "rationale": "first"}')
        state = _smart_state()
        session = SmartApproveSession(
            config_getter=build_test_vibe_config, authority_state=state, classifier=stub
        )
        args = _Args(command="git checkout main")
        first = await session.adjudicate(
            tool_name="bash",
            args=args,
            ctx=_policy_ctx(),
            messages=[],
            launch_intent=None,
        )
        second = await session.adjudicate(
            tool_name="bash",
            args=args,
            ctx=_policy_ctx(),
            messages=[],
            launch_intent=None,
        )
        assert first.verdict == "override"
        assert second.verdict == "override"
        assert len(stub.calls) == 1  # the classifier ran only once
        identity = DecidedCallIdentity.of("bash", args)
        record = state.decided_calls.lookup(identity)
        assert record is not None
        assert record.decision == "override"
        assert record.actor == "classifier"

    async def test_distinct_call_identities_are_decided_independently(self) -> None:
        stub = _StubClassifier(response='{"verdict": "uphold", "rationale": "no"}')
        session = SmartApproveSession(
            config_getter=build_test_vibe_config,
            authority_state=_smart_state(),
            classifier=stub,
        )
        await session.adjudicate(
            tool_name="bash",
            args=_Args(command="ls"),
            ctx=_policy_ctx(),
            messages=[],
            launch_intent=None,
        )
        await session.adjudicate(
            tool_name="bash",
            args=_Args(command="pwd"),
            ctx=_policy_ctx(),
            messages=[],
            launch_intent=None,
        )
        assert len(stub.calls) == 2


class _DeniedTool(BaseTool):
    name = "denied_stub"

    def resolve_permission(self, args: _Args) -> PermissionContext | None:
        return _policy_ctx()

    async def run(self, args: _Args, ctx: Any = None) -> AsyncGenerator[Any, None]:
        yield "unreachable"


class _SyntaxTool(BaseTool):
    name = "syntax_stub"

    def resolve_permission(self, args: _Args) -> PermissionContext | None:
        return PermissionContext(
            permission=ToolPermission.NEVER,
            reason="Command denied: unsupported shell syntax",
            denial_kind="syntax",
        )

    async def run(self, args: _Args, ctx: Any = None) -> AsyncGenerator[Any, None]:
        yield "unreachable"


def _make_tool(tool_cls: type[BaseTool]) -> BaseTool:
    from chartreux.core.tools.base import BaseToolConfig, BaseToolState

    return tool_cls(config_getter=lambda: BaseToolConfig(), state=BaseToolState())


class TestLoopHook:
    async def test_smart_override_executes_denied_call(self) -> None:
        loop = build_test_agent_loop()
        loop._authority_state.transition(SafetyLevel.SMART, actor="human")
        stub = _StubClassifier(
            response='{"verdict": "override", "rationale": "requested"}'
        )
        loop._smart_approve = SmartApproveSession(
            config_getter=lambda: loop.config,
            authority_state=loop._authority_state,
            classifier=stub,
        )
        decision = await loop._should_execute_tool(
            _make_tool(_DeniedTool), _Args(command="ls")
        )
        assert decision.verdict == ToolExecutionResponse.EXECUTE
        assert len(stub.calls) == 1
        await loop.aclose()

    async def test_smart_uphold_returns_original_denial(self) -> None:
        loop = build_test_agent_loop()
        loop._authority_state.transition(SafetyLevel.SMART, actor="human")
        stub = _StubClassifier(
            response='{"verdict": "uphold", "rationale": "accident"}'
        )
        loop._smart_approve = SmartApproveSession(
            config_getter=lambda: loop.config,
            authority_state=loop._authority_state,
            classifier=stub,
        )
        decision = await loop._should_execute_tool(
            _make_tool(_DeniedTool), _Args(command="rm -rf /")
        )
        assert decision.verdict == ToolExecutionResponse.SKIP
        assert decision.approval_type == ToolPermission.NEVER
        assert "outside the authorized workspace" in (decision.feedback or "")
        await loop.aclose()

    async def test_safe_level_never_consults_classifier(self) -> None:
        loop = build_test_agent_loop()
        stub = _StubClassifier(response='{"verdict": "override", "rationale": "x"}')
        loop._smart_approve = SmartApproveSession(
            config_getter=lambda: loop.config,
            authority_state=loop._authority_state,
            classifier=stub,
        )
        decision = await loop._should_execute_tool(
            _make_tool(_DeniedTool), _Args(command="ls")
        )
        assert decision.verdict == ToolExecutionResponse.SKIP
        assert stub.calls == []
        await loop.aclose()

    async def test_syntax_refusal_never_reaches_classifier_at_hook(self) -> None:
        loop = build_test_agent_loop()
        loop._authority_state.transition(SafetyLevel.SMART, actor="human")
        stub = _StubClassifier(response='{"verdict": "override", "rationale": "x"}')
        loop._smart_approve = SmartApproveSession(
            config_getter=lambda: loop.config,
            authority_state=loop._authority_state,
            classifier=stub,
        )
        decision = await loop._should_execute_tool(
            _make_tool(_SyntaxTool), _Args(command="echo $(unclosed")
        )
        assert decision.verdict == ToolExecutionResponse.SKIP
        assert stub.calls == []
        await loop.aclose()


class TestContextMessagesShape:
    def test_messages_are_llm_messages(self) -> None:
        # The hook feeds freeze_for_request() output; pin the shape it reads.
        messages = [
            LLMMessage(role=Role.user, content="do the thing"),
            LLMMessage(role=Role.assistant, content="ok"),
        ]
        assert [m.role for m in messages] == [Role.user, Role.assistant]


# --- v0.8.2 W7 -----------------------------------------------------------


class _ListSink:
    """Durable-sink stand-in: collects every settled record."""

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    async def __call__(self, record: UsageRecord, /) -> None:
        self.records.append(record)


def _attribution() -> UsageAttribution:
    return UsageAttribution(
        root_session_id="root-sess",
        session_id="sess",
        agent_role="root",
        model="stale",
        provider="stale",
        wire_name="stale",
        project_key="proj",
    )


def _usage_record(cost_usd: float = 0.0) -> UsageRecord:
    return UsageRecord(
        **_attribution()
        .model_copy(update={"purpose": UsagePurpose.SMART_APPROVE})
        .model_dump(),
        outcome=UsageOutcome.COMPLETED,
        usage_state=UsageState.COMPLETE,
        input_tokens=10,
        output_tokens=2,
        cached_input_tokens=0,
        record_id=str(uuid4()),
        occurred_at=datetime.now(UTC),
        prices_usd_per_million=UsagePrices(),
        known_cost_usd=cost_usd,
        has_unknown_cost=False,
    )


class TestModelClassifierUsageRecords:
    """W7 R1/R2: the real _model_classifier settles purpose:"smart-approve"
    usage records through the passed sink when attribution is supplied.
    """

    async def test_completed_attempt_reaches_durable_sink(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sink = _ListSink()
        backend = FakeBackend()

        async def complete(**_kwargs: Any) -> LLMChunk:
            notify_request_started()
            return LLMChunk(
                message=LLMMessage(
                    role=Role.assistant,
                    content='{"verdict": "override", "rationale": "ok"}',
                ),
                usage=LLMUsage.from_reported(
                    prompt_tokens=10, completion_tokens=4, cached_tokens=2
                ),
            )

        monkeypatch.setattr(backend, "complete", complete)
        monkeypatch.setattr(utility_completion, "create_backend", lambda **_: backend)
        response = await _model_classifier(
            config=build_test_vibe_config(),
            system_prompt="system",
            user_content="packet",
            accounting_sink=sink,
            usage_attribution=_attribution(),
        )
        assert response == '{"verdict": "override", "rationale": "ok"}'
        assert len(sink.records) == 1
        record = sink.records[0]
        assert record.purpose == UsagePurpose.SMART_APPROVE
        assert record.outcome == UsageOutcome.COMPLETED
        assert record.input_tokens == 10
        assert record.output_tokens == 4
        assert record.cached_input_tokens == 2
        # Session identity from the attribution survives settlement.
        assert record.root_session_id == "root-sess"
        assert record.session_id == "sess"

    async def test_transport_failure_settles_failed_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sink = _ListSink()
        backend = FakeBackend()

        async def complete(**_kwargs: Any) -> LLMChunk:
            notify_request_started()
            raise RuntimeError("transport down")

        monkeypatch.setattr(backend, "complete", complete)
        monkeypatch.setattr(utility_completion, "create_backend", lambda **_: backend)
        with pytest.raises(RuntimeError, match="transport down"):
            await _model_classifier(
                config=build_test_vibe_config(),
                system_prompt="system",
                user_content="packet",
                accounting_sink=sink,
                usage_attribution=_attribution(),
            )
        assert len(sink.records) == 1
        record = sink.records[0]
        assert record.purpose == UsagePurpose.SMART_APPROVE
        assert record.outcome == UsageOutcome.FAILED

    async def test_model_selection_failure_writes_no_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # W7 R2: the pre-transport boundary — model selection raises before
        # any backend exists, so no usage record is settled.
        sink = _ListSink()
        monkeypatch.setattr(
            smart_approve_module,
            "select_smart_approve_model",
            Mock(side_effect=ValueError("unknown model")),
        )
        with pytest.raises(ValueError, match="unknown model"):
            await _model_classifier(
                config=build_test_vibe_config(),
                system_prompt="system",
                user_content="packet",
                accounting_sink=sink,
                usage_attribution=_attribution(),
            )
        assert sink.records == []


class _SinkCallingClassifier:
    """Stub classifier that settles one record through the passed sink."""

    def __init__(self, record: UsageRecord) -> None:
        self.record = record
        self.seen_attribution: Any = None

    async def __call__(
        self,
        *,
        config: Any,
        system_prompt: str,
        user_content: str,
        accounting_sink: Any,
        usage_attribution: Any = None,
    ) -> str | None:
        self.seen_attribution = usage_attribution
        if accounting_sink is not None:
            await accounting_sink(self.record)
        return '{"verdict": "uphold", "rationale": "no"}'


class TestTeeSinkAndAttributionPlumbing:
    async def test_tee_sink_accrues_spend_and_settles_ledger(self) -> None:
        ledger = _ListSink()
        record = _usage_record(cost_usd=0.0025)
        session = SmartApproveSession(
            config_getter=build_test_vibe_config,
            authority_state=_smart_state(),
            ledger_sink_getter=lambda: ledger,
            attribution_getter=_attribution,
        )
        await session._tee_sink(record)
        assert session.spend_usd == pytest.approx(0.0025)
        assert ledger.records == [record]

    async def test_adjudicate_passes_attribution_and_tee_sink(self) -> None:
        ledger = _ListSink()
        record = _usage_record(cost_usd=0.001)
        stub = _SinkCallingClassifier(record)
        session = SmartApproveSession(
            config_getter=build_test_vibe_config,
            authority_state=_smart_state(),
            classifier=stub,
            ledger_sink_getter=lambda: ledger,
            attribution_getter=_attribution,
        )
        verdict = await session.adjudicate(
            tool_name="bash",
            args=_Args(command="ls"),
            ctx=_policy_ctx(),
            messages=[],
            launch_intent=None,
        )
        assert verdict.verdict == "uphold"
        assert stub.seen_attribution is not None
        assert stub.seen_attribution.root_session_id == "root-sess"
        assert session.spend_usd == pytest.approx(0.001)
        assert ledger.records == [record]


def _gate_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.message
        for record in caplog.records
        if record.name == "chartreux.core.agent_loop._smart_approve"
        and "smart_approve_adjudication" in record.message
    ]


class TestGateOutcomeLogging:
    """W7 R5: one structured session log line per adjudication gate."""

    async def test_level_gate_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        state = SessionAuthorityState(session_id="sess", level=SafetyLevel.SAFE)
        session = SmartApproveSession(
            config_getter=build_test_vibe_config, authority_state=state
        )
        with caplog.at_level(logging.INFO, logger=smart_approve_module.__name__):
            await session.adjudicate(
                tool_name="bash",
                args=_Args(command="ls"),
                ctx=_policy_ctx(),
                messages=[],
                launch_intent=None,
            )
        lines = _gate_lines(caplog)
        assert len(lines) == 1
        assert f"gate={GATE_LEVEL_NOT_SMART}" in lines[0]
        assert "verdict=uphold" in lines[0]

    async def test_syntax_gate_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        session = SmartApproveSession(
            config_getter=build_test_vibe_config,
            authority_state=_smart_state(),
            classifier=_StubClassifier(response="{}"),
        )
        syntax_ctx = PermissionContext(
            permission=ToolPermission.NEVER, reason="bad syntax", denial_kind="syntax"
        )
        with caplog.at_level(logging.INFO, logger=smart_approve_module.__name__):
            await session.adjudicate(
                tool_name="bash",
                args=_Args(command="echo $(x"),
                ctx=syntax_ctx,
                messages=[],
                launch_intent=None,
            )
        lines = _gate_lines(caplog)
        assert len(lines) == 1
        assert f"gate={GATE_SYNTAX}" in lines[0]

    async def test_replay_gate_logged_on_second_identical_call(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        stub = _StubClassifier(response='{"verdict": "override", "rationale": "first"}')
        session = SmartApproveSession(
            config_getter=build_test_vibe_config,
            authority_state=_smart_state(),
            classifier=stub,
        )
        args = _Args(command="git status")
        with caplog.at_level(logging.INFO, logger=smart_approve_module.__name__):
            first = await session.adjudicate(
                tool_name="bash",
                args=args,
                ctx=_policy_ctx(),
                messages=[],
                launch_intent=None,
            )
            second = await session.adjudicate(
                tool_name="bash",
                args=args,
                ctx=_policy_ctx(),
                messages=[],
                launch_intent=None,
            )
        assert first.gate == "override"
        assert second.gate == GATE_DECIDED_REPLAY
        lines = _gate_lines(caplog)
        assert len(lines) == 2
        assert f"gate={GATE_DECIDED_REPLAY}" in lines[1]

    async def test_cap_gate_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        session = SmartApproveSession(
            config_getter=build_test_vibe_config,
            authority_state=_smart_state(),
            classifier=_StubClassifier(response='{"verdict": "override"}'),
        )
        # Burn the spend cap so the cap gate fires without consulting.
        session.spend_usd = session._caps()[1]
        with caplog.at_level(logging.INFO, logger=smart_approve_module.__name__):
            verdict = await session.adjudicate(
                tool_name="bash",
                args=_Args(command="ls"),
                ctx=_policy_ctx(),
                messages=[],
                launch_intent=None,
            )
        assert verdict.gate == GATE_CAP
        lines = _gate_lines(caplog)
        assert len(lines) == 1
        assert f"gate={GATE_CAP}" in lines[0]

    async def test_classifier_error_gate_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = SmartApproveSession(
            config_getter=build_test_vibe_config,
            authority_state=_smart_state(),
            classifier=_StubClassifier(error=RuntimeError("down")),
        )
        with caplog.at_level(logging.INFO, logger=smart_approve_module.__name__):
            verdict = await session.adjudicate(
                tool_name="bash",
                args=_Args(command="ls"),
                ctx=_policy_ctx(),
                messages=[],
                launch_intent=None,
            )
        assert verdict.gate == GATE_CLASSIFIER_ERROR
        lines = _gate_lines(caplog)
        assert len(lines) == 1
        assert f"gate={GATE_CLASSIFIER_ERROR}" in lines[0]

    async def test_unparseable_gate_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = SmartApproveSession(
            config_getter=build_test_vibe_config,
            authority_state=_smart_state(),
            classifier=_StubClassifier(response="not json"),
        )
        with caplog.at_level(logging.INFO, logger=smart_approve_module.__name__):
            verdict = await session.adjudicate(
                tool_name="bash",
                args=_Args(command="ls"),
                ctx=_policy_ctx(),
                messages=[],
                launch_intent=None,
            )
        assert verdict.gate == GATE_UNPARSEABLE
        lines = _gate_lines(caplog)
        assert len(lines) == 1
        assert f"gate={GATE_UNPARSEABLE}" in lines[0]

    async def test_consulted_verdicts_log_their_verdict_as_gate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        for response, expected_gate in (
            ('{"verdict": "override", "rationale": "x"}', "override"),
            ('{"verdict": "uphold", "rationale": "x"}', "uphold"),
        ):
            caplog.clear()
            session = SmartApproveSession(
                config_getter=build_test_vibe_config,
                authority_state=_smart_state(),
                classifier=_StubClassifier(response=response),
            )
            with caplog.at_level(logging.INFO, logger=smart_approve_module.__name__):
                verdict = await session.adjudicate(
                    tool_name="bash",
                    args=_Args(command=f"cmd-{expected_gate}"),
                    ctx=_policy_ctx(),
                    messages=[],
                    launch_intent=None,
                )
            assert verdict.gate == expected_gate
            lines = _gate_lines(caplog)
            assert len(lines) == 1
            assert f"gate={expected_gate}" in lines[0]


class TestMarkerFeedback:
    """W7 R3/R4: uphold outcomes carry the uphold marker; replays and
    level/syntax gates return the original denial untouched.
    """

    async def test_uphold_carries_uphold_marker(self) -> None:
        loop = build_test_agent_loop()
        loop._authority_state.transition(SafetyLevel.SMART, actor="human")
        loop._smart_approve = SmartApproveSession(
            config_getter=lambda: loop.config,
            authority_state=loop._authority_state,
            classifier=_StubClassifier(
                response='{"verdict": "uphold", "rationale": "accident"}'
            ),
        )
        decision = await loop._should_execute_tool(
            _make_tool(_DeniedTool), _Args(command="rm -rf /")
        )
        assert decision.verdict == ToolExecutionResponse.SKIP
        assert UPHOLD_FEEDBACK_MARKER in (decision.feedback or "")
        assert "outside the authorized workspace" in (decision.feedback or "")
        await loop.aclose()

    async def test_override_carries_override_marker(self) -> None:
        loop = build_test_agent_loop()
        loop._authority_state.transition(SafetyLevel.SMART, actor="human")
        loop._smart_approve = SmartApproveSession(
            config_getter=lambda: loop.config,
            authority_state=loop._authority_state,
            classifier=_StubClassifier(
                response='{"verdict": "override", "rationale": "requested"}'
            ),
        )
        decision = await loop._should_execute_tool(
            _make_tool(_DeniedTool), _Args(command="ls")
        )
        assert decision.verdict == ToolExecutionResponse.EXECUTE
        assert OVERRIDE_FEEDBACK_MARKER in (decision.feedback or "")
        await loop.aclose()

    async def test_decided_call_replay_returns_denial_without_marker(self) -> None:
        loop = build_test_agent_loop()
        loop._authority_state.transition(SafetyLevel.SMART, actor="human")
        stub = _StubClassifier(response='{"verdict": "uphold", "rationale": "no"}')
        loop._smart_approve = SmartApproveSession(
            config_getter=lambda: loop.config,
            authority_state=loop._authority_state,
            classifier=stub,
        )
        args = _Args(command="rm -rf /tmp/x")
        first = await loop._should_execute_tool(_make_tool(_DeniedTool), args)
        second = await loop._should_execute_tool(_make_tool(_DeniedTool), args)
        assert len(stub.calls) == 1
        assert UPHOLD_FEEDBACK_MARKER in (first.feedback or "")
        # The replay carries no marker, so the UI never re-renders it.
        assert UPHOLD_FEEDBACK_MARKER not in (second.feedback or "")
        assert "outside the authorized workspace" in (second.feedback or "")
        await loop.aclose()
