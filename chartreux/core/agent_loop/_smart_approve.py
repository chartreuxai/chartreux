"""SMART-mode classifier adjudication (v0.8.0 item A, package A-P2).

At ``SafetyLevel.SMART`` the loop seam in
:meth:`AgentLoop._should_execute_tool` consults this module on RETURNED
denials only: shell-grammar (SYNTAX) refusals carry the typed
``denial_kind="syntax"`` marker and never reach the classifier. The
verdict protocol is strict and fail-closed: the classifier's whole
response is parsed as exactly one JSON object ``{"verdict", "rationale"}``
(one optional surrounding code fence stripped, duplicate keys rejected,
verdict by exact equality); anything unparseable, refused, or errored
upholds the denial and is recorded as the session's
``last_classifier_error``. A decided call identity is never re-decided:
the A-P0 decided-call memory replays the recorded verdict.

The packet is the denial facts (tool, canonical arguments, denial class,
policy reason, launch-intent line, current level) plus a
``<conversation-context>`` block: the current user instruction clamped to
the first 2000 chars, then up to 3 trailing messages clamped to 1000 chars
each, under the hard per-adjudication budget
``safety.smart_approve_context_chars`` (default 3000, enforced in code;
0 = packet-only). The classifier model resolves at use time through
``get_smart_approve_model()`` (thinking pinned OFF) and the prompt through
``smart_approve_prompt`` (bundled ``smart_approve_classifier.md`` by
default); any resolution/backend failure upholds. Per-session call and
spend caps fail closed the same way.

v0.8.2 W7 telemetry: every classifier transport attempt settles a
``purpose="smart-approve"`` usage record through a durable tee sink (the
session ledger writer plus the in-memory spend sink), and every
adjudication gate outcome is emitted as one structured session log line.
Pre-transport failures (model selection, empty content, skip_if_no_key in
``run_utility_completion``) never produce a usage record — they are
covered by the gate-outcome log instead. Classifier rationale is model
content and never enters the usage ledger (core/usage.py) or the log
lines.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from chartreux.core.agent_loop._safety_authority import (
    AuthorityActor,
    DecidedCallIdentity,
    SafetyLevel,
    SessionAuthorityState,
)
from chartreux.core.config.models import (
    DEFAULT_SMART_APPROVE_CONTEXT_CHARS,
    DEFAULT_SMART_APPROVE_MAX_CALLS,
    DEFAULT_SMART_APPROVE_MAX_SPEND_USD,
    DEFAULT_SMART_APPROVE_PROMPT_ID,
)
from chartreux.core.llm.utility_completion import (
    run_utility_completion,
    select_smart_approve_model,
)
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.tools.models import PermissionContext
from chartreux.core.usage import (
    AccountingSink,
    UsageAttribution,
    UsagePurpose,
    UsageRecord,
)

if TYPE_CHECKING:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema

logger = logging.getLogger(__name__)

__all__ = [
    "ClassifierRequest",
    "ClassifierVerdict",
    "SmartApproveSession",
    "build_conversation_context",
    "build_denial_packet",
    "parse_classifier_verdict",
]

VERDICT_OVERRIDE = "override"
VERDICT_UPHOLD = "uphold"

#: v0.8.2 W7: adjudication gate outcomes, each emitted as one structured
#: session log line. A classifier-consulted verdict logs its own verdict
#: ("override"/"uphold") as the gate.
GATE_LEVEL_NOT_SMART = "level-not-smart"
GATE_SYNTAX = "syntax"
GATE_DECIDED_REPLAY = "decided-call-replay"
GATE_CAP = "cap"
GATE_CLASSIFIER_ERROR = "classifier-error"
GATE_UNPARSEABLE = "unparseable"

#: Feedback markers the UI scans for in tool-decision feedback (W7 R3).
OVERRIDE_FEEDBACK_MARKER = "SMART classifier override:"
UPHOLD_FEEDBACK_MARKER = "SMART classifier uphold:"

#: Gates that render a red "denied by smart approve" transcript marker;
#: the override verdict renders the green one. Level/syntax/replay gates
#: never render (replays must not re-render, W7 R4).
UPHOLD_MARKER_GATES = frozenset({
    VERDICT_UPHOLD,
    GATE_CAP,
    GATE_CLASSIFIER_ERROR,
    GATE_UNPARSEABLE,
})

#: Current user instruction clamp (first chars kept).
INSTRUCTION_CLAMP_CHARS = 2000
#: Per-message clamp inside the trailing window.
WINDOW_MESSAGE_CLAMP_CHARS = 1000
#: Up-to-N messages immediately preceding the current user message.
WINDOW_MESSAGE_COUNT = 3
#: Classifier completion budget; the verdict JSON is small.
CLASSIFIER_MAX_TOKENS = 512

CONTEXT_OPEN = "<conversation-context>"
CONTEXT_CLOSE = "</conversation-context>"
_ELISION = "…[truncated]"


@dataclass(frozen=True, slots=True)
class ClassifierVerdict:
    """The parsed classifier verdict; ``verdict`` is exactly override/uphold.

    ``gate`` names the adjudication gate that produced the verdict (W7);
    a classifier-consulted verdict carries its own verdict as the gate.
    """

    verdict: str
    rationale: str
    gate: str = "classifier"


class ClassifierRequest(Protocol):
    """The classifier call: one system prompt, one user string, fail-closed."""

    def __call__(
        self,
        *,
        config: ChartreuxConfigSchema,
        system_prompt: str,
        user_content: str,
        accounting_sink: AccountingSink | None,
        usage_attribution: UsageAttribution | None,
    ) -> Awaitable[str | None]: ...


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in classifier response: {key!r}")
        result[key] = value
    return result


def _strip_one_fence(text: str) -> str:
    """Strip ONE optional surrounding code fence, then parse (pinned behavior:
    strip-one-fence-then-parse, never reject; a second fence simply fails the
    strict ``json.loads`` that follows).
    """
    if text.startswith("```"):
        newline = text.find("\n")
        if newline != -1:
            inner = text[newline + 1 :]
            if inner.endswith("```"):
                return inner[:-3]
    return text


def parse_classifier_verdict(response: str | None) -> ClassifierVerdict | None:
    """Strict whole-response parse; anything else is None (uphold).

    Exactly one JSON object with exactly the keys ``{"verdict", "rationale"}``;
    ``verdict`` by exact equality against "override"/"uphold"; duplicate keys
    rejected via ``object_pairs_hook``.
    """
    if response is None:
        return None
    text = _strip_one_fence(response.strip()).strip()
    try:
        payload: Any = (
            json.loads(text, object_pairs_hook=_reject_duplicate_keys) if text else None
        )
    except ValueError:
        return None
    if (
        not isinstance(payload, dict)
        or set(payload.keys()) != {"verdict", "rationale"}
        or not isinstance(payload["verdict"], str)
        or payload["verdict"] not in {VERDICT_OVERRIDE, VERDICT_UPHOLD}
        or not isinstance(payload["rationale"], str)
    ):
        return None
    return ClassifierVerdict(verdict=payload["verdict"], rationale=payload["rationale"])


def _clamp(text: str, limit: int) -> str:
    if len(text) > limit:
        return text[:limit] + _ELISION
    return text


def build_conversation_context(
    messages: Sequence[LLMMessage], *, budget_chars: int
) -> str:
    """The ``<conversation-context>`` block; "" when the budget is 0.

    First line is the current user instruction (the last ``role=user``
    message) clamped to the first ``min(2000, budget)`` chars; then up to 3
    trailing messages as ``role: text`` clamped to 1000 chars each. The hard
    budget is enforced in code before the call: the whole block (fences
    included) never exceeds ``budget_chars``.
    """
    if budget_chars <= 0:
        return ""
    last_user_idx: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == Role.user:
            last_user_idx = index
            break
    instruction_limit = min(INSTRUCTION_CLAMP_CHARS, budget_chars)
    body_parts: list[str] = []
    if last_user_idx is not None:
        instruction = (messages[last_user_idx].content or "").strip()
        body_parts.append(_clamp(instruction, instruction_limit))
        window = messages[max(0, last_user_idx - WINDOW_MESSAGE_COUNT) : last_user_idx]
        window_lines: list[str] = []
        for message in window:
            text = (message.content or "").strip()
            if not text:
                continue
            window_lines.append(
                _clamp(f"{message.role.value}: {text}", WINDOW_MESSAGE_CLAMP_CHARS)
            )
        if window_lines:
            body_parts.append("\n".join(window_lines))
    body = "\n".join(body_parts)
    overhead = len(CONTEXT_OPEN) + len(CONTEXT_CLOSE) + 2
    if len(body) + overhead > budget_chars:
        body = _clamp(body, max(budget_chars - overhead, 0))
    if not body:
        return ""
    return f"{CONTEXT_OPEN}\n{body}\n{CONTEXT_CLOSE}"


def build_denial_packet(
    *,
    tool_name: str,
    args: BaseModel | None,
    denial_class: str,
    reason: str,
    launch_intent: str | None,
    level: str,
    context_block: str,
) -> str:
    """The denial packet: call identity, denial class + policy reason,
    launch-intent line (when present), current level, and the fenced
    conversation context (when present).
    """
    lines = [
        "Denial packet:",
        f"tool: {tool_name}",
        f"arguments: {args.model_dump_json() if args is not None else ''}",
        f"denial class: {denial_class}",
        f"policy reason: {reason}",
    ]
    if launch_intent:
        lines.append(f"launch intent: {launch_intent}")
    lines.append(f"current safety level: {level}")
    user_content = "\n".join(lines)
    if context_block:
        user_content = f"{user_content}\n\n{context_block}"
    return user_content


async def _model_classifier(
    *,
    config: ChartreuxConfigSchema,
    system_prompt: str,
    user_content: str,
    accounting_sink: AccountingSink | None,
    usage_attribution: UsageAttribution | None,
) -> str | None:
    """The real classifier call: utility seam, SMART model, thinking OFF.

    W7 R1: ``usage_attribution`` is passed through so every transport
    attempt settles a ``purpose="smart-approve"`` usage record (outcome
    completed/failed/refused) into the sink. W7 R2: pre-transport
    failures (model selection, empty content) raise before any backend
    exists and settle NO record; the adjudication site fail-closes to
    uphold and the gate-outcome log covers the attempt.

    Any resolution or backend error propagates to the adjudication site,
    which fail-closes to uphold and records it as ``last_classifier_error``.
    """
    return await run_utility_completion(
        config=config,
        system_prompt=system_prompt,
        user_content=user_content,
        max_tokens=CLASSIFIER_MAX_TOKENS,
        request_timeout_seconds=config.api_timeout,
        retry_budget_seconds=config.api_retry_max_elapsed_time,
        accounting_sink=accounting_sink,
        usage_attribution=usage_attribution,
        purpose=UsagePurpose.SMART_APPROVE,
        model_selector=select_smart_approve_model,
    )


class SmartApproveSession:
    """Per-session SMART-classifier state: caps, counters, decided calls.

    The adjudication itself is fail-closed end to end: any classifier error,
    unparseable/refused response, or exhausted cap upholds the denial.
    """

    def __init__(
        self,
        *,
        config_getter: Callable[[], ChartreuxConfigSchema],
        authority_state: SessionAuthorityState | None,
        classifier: ClassifierRequest | None = None,
        ledger_sink_getter: Callable[[], AccountingSink | None] | None = None,
        attribution_getter: Callable[[], UsageAttribution | None] | None = None,
    ) -> None:
        self._config_getter = config_getter
        self._authority_state = authority_state
        self._classifier: ClassifierRequest = classifier or _model_classifier
        # W7 R1: the durable ledger sink (tee'd with _spend_sink) and the
        # session-identity attribution for the classifier's usage records.
        self._ledger_sink_getter = ledger_sink_getter
        self._attribution_getter = attribution_getter
        self.calls = 0
        self.spend_usd = 0.0
        self.override_count = 0
        self.uphold_count = 0
        self.last_classifier_error: str | None = None

    def _caps(self) -> tuple[int, float, int]:
        section = getattr(self._config_getter(), "safety", None)
        if section is None:
            return (
                DEFAULT_SMART_APPROVE_MAX_CALLS,
                DEFAULT_SMART_APPROVE_MAX_SPEND_USD,
                DEFAULT_SMART_APPROVE_CONTEXT_CHARS,
            )
        return (
            section.smart_approve_max_calls,
            section.smart_approve_max_spend_usd,
            section.smart_approve_context_chars,
        )

    async def _spend_sink(self, record: UsageRecord, /) -> None:
        if record.purpose == UsagePurpose.SMART_APPROVE:
            self.spend_usd += float(record.known_cost_usd)

    async def _tee_sink(self, record: UsageRecord, /) -> None:
        """W7 R1: the durable tee — accrue in-memory spend, then settle to
        the ledger sink. Spend accrues first so a ledger failure never
        masks cap accounting; the finalizer's ``_write`` contains sink
        errors, so neither leg raises into inference control flow.
        """
        await self._spend_sink(record)
        if self._ledger_sink_getter is not None:
            ledger = self._ledger_sink_getter()
            if ledger is not None:
                await ledger(record)

    def _log_gate(
        self, *, gate: str, verdict: str, tool_name: str, identity: dict[str, str]
    ) -> None:
        """W7 R5: one structured session log line per adjudication gate.

        The classifier rationale is model-generated content and stays out
        of this line (and out of the usage ledger entirely, per
        core/usage.py), so only content-free fields are logged.
        """
        logger.info(
            "smart_approve_adjudication gate=%s verdict=%s tool=%s "
            "decided_call=%s calls=%d spend_usd=%.6f",
            gate,
            verdict,
            tool_name,
            identity,
            self.calls,
            self.spend_usd,
        )

    def stats_view(self, config: ChartreuxConfigSchema) -> dict[str, Any]:
        """v0.8.0 A-P3: the posture-report counters and caps for /status, /safety."""
        max_calls, max_spend, context_chars = self._caps()
        section = getattr(config, "safety", None)
        prompt_id = (
            section.smart_approve_prompt_id
            if section is not None
            else DEFAULT_SMART_APPROVE_PROMPT_ID
        )
        model_expression = (
            section.smart_approve_model if section is not None else ""
        ) or "(active model)"
        return {
            "prompt_id": prompt_id,
            "model_expression": model_expression,
            "calls": self.calls,
            "max_calls": max_calls,
            "spend_usd": self.spend_usd,
            "max_spend_usd": max_spend,
            "context_chars": context_chars,
            "override_count": self.override_count,
            "uphold_count": self.uphold_count,
            "last_classifier_error": self.last_classifier_error,
        }

    async def adjudicate(
        self,
        *,
        tool_name: str,
        args: BaseModel | None,
        ctx: PermissionContext | None,
        messages: Sequence[LLMMessage],
        launch_intent: str | None,
    ) -> ClassifierVerdict:
        """Adjudicate one returned denial; always returns a verdict.

        Gates, in order: level must be SMART; SYNTAX refusals never reach the
        classifier; decided-call memory replays a recorded verdict; caps fail
        closed; otherwise the classifier is consulted and its strict verdict
        (or fail-closed uphold) is recorded in the decided-call memory.

        W7: every gate outcome is emitted as one structured session log
        line, and the classifier call settles its usage record through the
        durable tee sink with session-identity attribution.
        """
        state = self._authority_state
        identity = DecidedCallIdentity.of(tool_name, args)
        if state is None or state.level is not SafetyLevel.SMART:
            verdict = ClassifierVerdict(
                verdict=VERDICT_UPHOLD,
                rationale="safety level is not smart",
                gate=GATE_LEVEL_NOT_SMART,
            )
            self._log_gate(
                gate=GATE_LEVEL_NOT_SMART,
                verdict=verdict.verdict,
                tool_name=tool_name,
                identity=identity.encode(),
            )
            return verdict
        if ctx is not None and ctx.denial_kind == "syntax":
            verdict = ClassifierVerdict(
                verdict=VERDICT_UPHOLD,
                rationale="shell-grammar refusals never reach the classifier",
                gate=GATE_SYNTAX,
            )
            self._log_gate(
                gate=GATE_SYNTAX,
                verdict=verdict.verdict,
                tool_name=tool_name,
                identity=identity.encode(),
            )
            return verdict
        record = state.decided_calls.lookup(identity)
        if record is not None:
            verdict = ClassifierVerdict(
                verdict=(
                    VERDICT_OVERRIDE
                    if record.decision == VERDICT_OVERRIDE
                    else VERDICT_UPHOLD
                ),
                rationale="decided-call memory: never re-decided",
                gate=GATE_DECIDED_REPLAY,
            )
            self._log_gate(
                gate=GATE_DECIDED_REPLAY,
                verdict=verdict.verdict,
                tool_name=tool_name,
                identity=identity.encode(),
            )
            return verdict
        max_calls, max_spend, context_chars = self._caps()
        if self.calls >= max_calls or self.spend_usd >= max_spend:
            self.last_classifier_error = (
                "smart_approve call/spend cap reached; upholding"
            )
            verdict = ClassifierVerdict(
                verdict=VERDICT_UPHOLD, rationale="cap reached", gate=GATE_CAP
            )
        else:
            self.calls += 1
            config = self._config_getter()
            context_block = build_conversation_context(
                messages, budget_chars=context_chars
            )
            user_content = build_denial_packet(
                tool_name=tool_name,
                args=args,
                denial_class=(
                    str(ctx.denial_kind)
                    if ctx is not None and ctx.denial_kind
                    else "policy"
                ),
                reason=(ctx.reason or "") if ctx is not None else "",
                launch_intent=launch_intent,
                level=state.level.value,
                context_block=context_block,
            )
            try:
                system_prompt = config.smart_approve_prompt
                response = await self._classifier(
                    config=config,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    accounting_sink=self._tee_sink,
                    usage_attribution=(
                        self._attribution_getter()
                        if self._attribution_getter is not None
                        else None
                    ),
                )
            except Exception as error:
                self.last_classifier_error = f"{type(error).__name__}: {error}"
                verdict = ClassifierVerdict(
                    verdict=VERDICT_UPHOLD,
                    rationale="classifier call failed",
                    gate=GATE_CLASSIFIER_ERROR,
                )
            else:
                parsed = parse_classifier_verdict(response)
                if parsed is None:
                    self.last_classifier_error = (
                        "unparseable or refused classifier response; upholding"
                    )
                    verdict = ClassifierVerdict(
                        verdict=VERDICT_UPHOLD,
                        rationale="unparseable response",
                        gate=GATE_UNPARSEABLE,
                    )
                else:
                    verdict = ClassifierVerdict(
                        verdict=parsed.verdict,
                        rationale=parsed.rationale,
                        gate=parsed.verdict,
                    )
        self._log_gate(
            gate=verdict.gate,
            verdict=verdict.verdict,
            tool_name=tool_name,
            identity=identity.encode(),
        )
        state.record_decided_call(
            identity, decision=verdict.verdict, actor=AuthorityActor.CLASSIFIER.value
        )
        if verdict.verdict == VERDICT_OVERRIDE:
            self.override_count += 1
        else:
            self.uphold_count += 1
        return verdict
