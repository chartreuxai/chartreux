from __future__ import annotations

import pytest

from chartreux.core.events import CompactStartEvent
from tests.agent_loop.e2e.conftest import MistralAPI, build_e2e_agent_loop
from tests.backend.data.mistral import mistral_completion
from tests.conftest import build_test_vibe_config, make_test_models

COMPACTION_MODELS = make_test_models(auto_compact_threshold=1)

# A scripted compaction response carrying a pending-approval handoff: design
# accepted, plan pending, approved scope with evidence, one consumed and one
# unconsumed one-time grant, and active agent/run handles. Pure ASCII so the
# fragments survive the JSON-encoded model-facing request verbatim.
PENDING_APPROVAL_HANDOFF = """\
Goal: implement the retry policy from the user's approved design.
Task classification: implementation (non-trivial)
Current workflow phase: plan
Dispatch policy identity: orchestrated; policy version: 1; snapshot version: 1
Contributor authorship: scope=retry.py; slot=implementor; evidence=changed retry helper; status=partial
Consumed attempt budget: scope=retry.py; used=2; remaining=0
Current recovery route: scope=retry.py; advisor diagnosis pending; no implementation retry
Design acceptance state: accepted
Plan acceptance state: pending
Approved scope: design work only; no implementation until the plan is accepted
Approval evidence: the user's message 'the design looks good, go ahead' approved the design
One-time grants:
- target: scratchpad draft of the plan; state: consumed
- target: one focused test run; state: unconsumed
Active agent/run handles: agent-7 (run-3, idle), agent-9 (run-5, running)
Outstanding dependencies: agent-9's findings feed the plan; run-3's result is uncollected
Next step: present the plan to the user and wait for explicit plan acceptance
"""


@pytest.mark.asyncio
async def test_auto_compaction_preserves_user_message_and_embeds_summary(
    mistral_api: MistralAPI,
) -> None:
    mistral_api.reply(
        # First: the compaction summary
        mistral_completion("<summary>A summary of what happened so far</summary>"),
        # Then: the final answer to user message
        mistral_completion("final answer"),
    )
    agent = build_e2e_agent_loop(
        config=build_test_vibe_config(models=COMPACTION_MODELS)
    )
    agent.stats.context_tokens = 5  # Trigger the auto-compaction immediately

    events = [event async for event in agent.act("Investigate the bug")]

    assert any(isinstance(e, CompactStartEvent) for e in events)
    sent_after_compaction = mistral_api.model_facing_text(1)
    assert "Investigate the bug" in sent_after_compaction
    assert "A summary of what happened so far" in sent_after_compaction


@pytest.mark.asyncio
async def test_repeated_auto_compaction_preserves_earlier_user_messages(
    mistral_api: MistralAPI,
) -> None:
    mistral_api.reply(
        mistral_completion("<summary>summary one</summary>"),
        mistral_completion("reply one"),
        mistral_completion("<summary>summary two</summary>"),
        mistral_completion("reply two"),
    )
    agent = build_e2e_agent_loop(
        config=build_test_vibe_config(models=COMPACTION_MODELS)
    )

    agent.stats.context_tokens = 5
    [_ async for _ in agent.act("first ask")]
    agent.stats.context_tokens = 5
    [_ async for _ in agent.act("second ask")]

    sent_after_second_compaction = mistral_api.model_facing_text(3)
    assert "first ask" in sent_after_second_compaction
    assert "second ask" in sent_after_second_compaction


@pytest.mark.asyncio
async def test_oversized_user_message_is_middle_truncated_in_compaction(
    mistral_api: MistralAPI,
) -> None:
    huge_message = "alpha " * 20_000
    mistral_api.reply(
        mistral_completion("<summary>summary intro</summary>"),
        mistral_completion("reply intro"),
        mistral_completion("<summary>summary huge</summary>"),
        mistral_completion("reply huge"),
    )
    agent = build_e2e_agent_loop(
        config=build_test_vibe_config(models=COMPACTION_MODELS)
    )

    agent.stats.context_tokens = 5
    [_ async for _ in agent.act("intro")]
    agent.stats.context_tokens = 5
    [_ async for _ in agent.act(huge_message)]

    sent_after_second_compaction = mistral_api.model_facing_text(3)
    assert "[... truncated ...]" in sent_after_second_compaction
    assert "intro" not in sent_after_second_compaction


@pytest.mark.asyncio
async def test_compaction_handoff_reaches_next_request_with_pending_approval(
    mistral_api: MistralAPI,
) -> None:
    mistral_api.reply(
        # First: the compaction summary carrying the pending-approval handoff
        mistral_completion(f"<summary>{PENDING_APPROVAL_HANDOFF}</summary>"),
        # Then: the final answer to user message
        mistral_completion("final answer"),
    )
    agent = build_e2e_agent_loop(
        config=build_test_vibe_config(models=COMPACTION_MODELS)
    )
    agent.stats.context_tokens = 5  # Trigger the auto-compaction immediately

    events = [event async for event in agent.act("Investigate the bug")]

    assert any(isinstance(e, CompactStartEvent) for e in events)
    sent_after_compaction = mistral_api.model_facing_text(1)
    for fragment in (
        "Design acceptance state: accepted",
        "Dispatch policy identity:",
        "Contributor authorship: scope=retry.py",
        "status=partial",
        "used=2; remaining=0",
        "advisor diagnosis pending",
        "Plan acceptance state: pending",
        "Approved scope:",
        "Approval evidence:",
        "state: consumed",
        "state: unconsumed",
        "agent-7",
        "run-3",
        "agent-9",
        "run-5",
        "uncollected",
    ):
        assert fragment in sent_after_compaction


@pytest.mark.asyncio
async def test_repeated_compaction_transports_dispatch_handoff(
    mistral_api: MistralAPI,
) -> None:
    mistral_api.reply(
        mistral_completion(f"<summary>{PENDING_APPROVAL_HANDOFF}</summary>"),
        mistral_completion("reply one"),
        mistral_completion(f"<summary>{PENDING_APPROVAL_HANDOFF}</summary>"),
        mistral_completion("reply two"),
    )
    agent = build_e2e_agent_loop(
        config=build_test_vibe_config(models=COMPACTION_MODELS)
    )
    bound = agent.bound_dispatch_policy
    agent.stats.context_tokens = 5
    [_ async for _ in agent.act("first ask")]
    agent.stats.context_tokens = 5
    [_ async for _ in agent.act("continue the same scope")]
    sent = mistral_api.model_facing_text(3)
    for fragment in (
        "used=2; remaining=0",
        "advisor diagnosis pending",
        "scope=retry.py",
        "status=partial",
        "Plan acceptance state: pending",
        "state: consumed",
    ):
        assert fragment in sent
    assert agent.bound_dispatch_policy == bound
