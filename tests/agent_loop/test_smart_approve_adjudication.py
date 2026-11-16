"""v0.8.0 A-P2 chunk 1: strict parser, fail-closed matrix, decided-call
memory, the typed SYNTAX discriminator, and the _should_execute_tool hook.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel
import pytest

from chartreux.core.agent_loop._loop_types import ToolExecutionResponse
from chartreux.core.agent_loop._safety_authority import (
    DecidedCallIdentity,
    SafetyLevel,
    SessionAuthorityState,
)
from chartreux.core.agent_loop._smart_approve import (
    SmartApproveSession,
    parse_classifier_verdict,
)
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.tools.base import BaseTool, ToolPermission
from chartreux.core.tools.builtins._shell_diagnostics import DenialClass, Diagnostic
from chartreux.core.tools.builtins._shell_permission_resolver import _denied
from chartreux.core.tools.models import PermissionContext
from tests.conftest import build_test_agent_loop, build_test_vibe_config

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
    ) -> str | None:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_content": user_content,
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
