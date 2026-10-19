"""WP6 — gradual enablement: the promoted defaults and their behavior matrix.

The v0.5.6 release PROMOTES all three recovery features to enabled-by-default
(OD4b/OD5, after WP4's, U's, and R2's acceptance matrices all passed):

- ``[recovery.subagents]`` (``RecoveryConfig.enabled``) — the bounded
  automatic recovery controller for child subagents (WP4);
- ``[recovery.root]`` (``RootRecoveryConfig.enabled``) — the bounded
  automatic root-turn recovery controller (R2);
- ``[user_questions]`` (``UserQuestionsConfig.unattended_timeout_enabled``)
  — unattended expiry of explicitly defaultable questions (U).

This file verifies the PUBLISHED BEHAVIOR of the shipped defaults — fresh
controlled worker/session tests, not source assertions alone:

- one default-flip sanity test per section (the shipped default is enabled);
- [user_questions] (from U's round-2 amendment): (a1) default-enabled +
  per-question ``unattended_default`` expires to the typed ``unanswered``
  disposition; (a2) default-enabled + ``requires_user`` or a missing default
  never expires; (a3) default-enabled + a runtime approval (root access)
  never expires and grants nothing; (a4) an EXPLICITLY disabled config still
  suppresses expiry under the new default;
- [recovery.subagents] (from WP4's round-3 symmetry amendment): the
  default-enabled policy automatically recovers a suspended child through
  the SAME ``resume_agent`` admission; disabling cancels the pending
  schedule without discarding evidence;
- [recovery.root] (from R2's round-2 amendment): the default-enabled policy
  schedules an automatic continuation for a terminal transient failure with
  NO user action; exhaustion parks to the manual Retry affordance (R1);
  disabling cancels the pending schedule without discarding evidence.

The explicit-flag disabled-state tests of the three packages keep passing
unchanged (they set their flags explicitly — never relying on the shipped
default); see tests/recovery/test_question_timeout.py,
tests/recovery/test_recovery_controller.py, and
tests/recovery/test_root_recovery_config.py.

Harnesses are reused from the sibling recovery test modules (the established
cross-test import pattern in this directory); all transports are controlled
fakes and all time-based behavior runs on manual clocks behind gated
wakeups — an unmocked provider call cannot occur (fail-closed egress).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from chartreux.app_server._sessions import _AgentState
from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.models import (
    AnsweredCallbackState,
    ExpiredCallbackState,
    OpenCallbackState,
    PublicMessageEntry,
    UserAnswer,
    UserInputCallbackDetail,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import (
    RecoveryConfig,
    RecoverySectionConfig,
    RootRecoveryConfig,
    UserQuestionsConfig,
)
from chartreux.core.recovery_models import RecoveryEligibility
from tests.app_server._session_runtime_support import _consume, _next_event
from tests.app_server._subagent_test_support import (
    _GatedWakeup,
    _ManualClock,
    _wait_result,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.recovery.test_explicit_resume import (
    _empty_fault,
    _final_outcome,
    _write_outcome,
)
from tests.recovery.test_question_timeout import (
    _answer_output,
    _callback_entry,
    _completed_effect_output,
    _defaultable_payload,
    _effect_entries,
    _open_session,
    _patch_timer,
    _patched_session,
    _question_tool_call,
    _read_tool_call,
    _release_gates,
    _requires_user_payload,
    _unanswered_from_effect,
    _wait_until,
)
from tests.recovery.test_recovery_controller import (
    _controller_registry,
    _events_of,
    _Factory,
    _suspend_child,
    _until,
    _until_admitted,
)
from tests.recovery.test_root_recovery import _failure, _GateHarness
from tests.stubs.fake_backend import FakeBackend

#: The shipped [user_questions] default timeout (UserQuestionsConfig).
DEFAULT_TIMEOUT = 300.0


@pytest.fixture(autouse=True)
def _controller_events_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """The child controller's milestone lines are INFO on the ``vibe`` logger."""
    caplog.set_level(logging.INFO, logger="vibe")


# ---------------------------------------------------------------------------
# The promoted defaults (one sanity test per section)
# ---------------------------------------------------------------------------


class TestPromotedDefaults:
    """The shipped default is enabled for all three recovery features."""

    def test_subagent_recovery_is_enabled_by_default(self) -> None:
        assert RecoveryConfig().enabled is True
        assert ChartreuxConfigSchema().recovery.subagents.enabled is True

    def test_root_recovery_is_enabled_by_default(self) -> None:
        assert RootRecoveryConfig().enabled is True
        assert ChartreuxConfigSchema().recovery.root.enabled is True

    def test_user_question_expiry_is_enabled_by_default(self) -> None:
        assert UserQuestionsConfig().unattended_timeout_enabled is True
        assert ChartreuxConfigSchema().user_questions.unattended_timeout_enabled is True


# ---------------------------------------------------------------------------
# [user_questions] — the default-enabled expiry matrix (from U)
# ---------------------------------------------------------------------------


def _default_enabled_config(*, enabled_tools: list[str]) -> ChartreuxConfigSchema:
    """The SHIPPED default config: expiry enabled (OD4b promotion).

    No ``[user_questions]`` override is applied — the test relies on the
    promoted default, which is exactly what the enablement matrix pins.
    """
    config = build_test_vibe_config(enabled_tools=enabled_tools)
    assert config.user_questions.unattended_timeout_enabled is True
    return config


class TestUserQuestionsEnablement:
    """Default-enabled expiry behavior (U's round-2 amendment, cases a1-a4)."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_default_enabled_expiry_expires_a_defaultable_question(self) -> None:
        """(a1) Default-enabled + a per-question unattended_default: eligible
        silence expires into the typed unanswered disposition — exactly one
        unanswered result and one deferred record, nothing approved.
        """
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_defaultable_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _default_enabled_config(enabled_tools=["ask_user_question"]),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            events: list[Any] = []
            callback: Any = None
            async for event in ctx.session.act("ask me"):
                events.append(event)
                if callback is None and isinstance(event, CallbackRequested):
                    callback = event.callback
                    expiry = callback.detail.unattended_expiry
                    assert expiry is not None
                    # The shipped default timeout (300s) arms the timer.
                    assert expiry.timeout_seconds == DEFAULT_TIMEOUT
                    await _wait_until(lambda: bool(ctx.wakeup.calls))
                    assert ctx.wakeup.calls == [DEFAULT_TIMEOUT]
                    # Silence: advance past the deadline and release the timer.
                    ctx.clock.advance(DEFAULT_TIMEOUT + 1.0)
                    _release_gates(ctx.wakeup)
            assert callback is not None
            # Exactly one unanswered result: the server-owned disposition.
            effects = _effect_entries(ctx.session.history)
            assert len(effects) == 1
            unanswered = _unanswered_from_effect(effects[0])
            assert unanswered.disposition == "unanswered"
            assert unanswered.reason == "unanswered_timeout"
            assert unanswered.answers == []
            assert unanswered.cancelled is False
            assert unanswered.timeout_seconds == DEFAULT_TIMEOUT
            assert unanswered.elapsed_seconds == DEFAULT_TIMEOUT + 1.0
            assert unanswered.continuation_guidance
            # The closure is published: terminal expired history, no live
            # callback.
            entry = _callback_entry(ctx.session.history)
            assert isinstance(entry.state, ExpiredCallbackState)
            assert entry.state.reason == "unanswered_timeout"
            assert ctx.session.state.active_callbacks == []
            # Exactly one deferred record, keyed by session and request.
            records = ctx.loop.deferred_questions.records()
            assert len(records) == 1
            assert records[0].request_id == callback.callback_id
            assert records[0].reason == "unanswered_timeout"
            # The turn completed normally after the deferral.
            assert any(
                isinstance(entry, PublicMessageEntry)
                and entry.role == "assistant"
                and "Done" in entry.text
                for entry in ctx.session.history
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_default_enabled_never_expires_a_missing_default(self) -> None:
        """(a2) Default-enabled + a defaultable kind WITHOUT the
        unattended-default marker: the per-question opt-in is absent, so the
        question never expires.
        """
        payload = _defaultable_payload()
        del payload["unattended_default"]
        backend = FakeBackend([
            [mock_llm_chunk(content="", tool_calls=[_question_tool_call([payload])])],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _default_enabled_config(enabled_tools=["ask_user_question"]),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            assert callback.detail.unattended_expiry is None
            await asyncio.sleep(0.05)
            assert ctx.wakeup.calls == []
            # Silence far past any deadline: the question stays open.
            ctx.clock.advance(10_000.0)
            _release_gates(ctx.wakeup)
            await asyncio.sleep(0.05)
            assert [c.callback_id for c in ctx.session.state.active_callbacks] == [
                callback.callback_id
            ]
            assert isinstance(
                _callback_entry(ctx.session.history).state, OpenCallbackState
            )
            assert ctx.loop.deferred_questions.records() == []
            # The user can still answer normally.
            await ctx.session.respond_to_callback(
                callback.callback_id, _answer_output()
            )
            await asyncio.wait_for(_consume(stream), timeout=10)
            result = UserQuestionResult.model_validate(
                _completed_effect_output(ctx.session.history)
            )
            assert result.disposition == "answered"
            assert isinstance(
                _callback_entry(ctx.session.history).state, AnsweredCallbackState
            )
            assert ctx.loop.deferred_questions.records() == []

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_default_enabled_never_expires_a_requires_user_batch(self) -> None:
        """(a2) Default-enabled + a requires_user question: the batch blocks
        expiry for every question in it.
        """
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[
                        _question_tool_call([
                            _defaultable_payload(),
                            _requires_user_payload(),
                        ])
                    ],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _default_enabled_config(enabled_tools=["ask_user_question"]),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            assert callback.detail.unattended_expiry is None
            await asyncio.sleep(0.05)
            assert ctx.wakeup.calls == []
            ctx.clock.advance(10_000.0)
            _release_gates(ctx.wakeup)
            await asyncio.sleep(0.05)
            assert [c.callback_id for c in ctx.session.state.active_callbacks] == [
                callback.callback_id
            ]
            assert isinstance(
                _callback_entry(ctx.session.history).state, OpenCallbackState
            )
            assert ctx.loop.deferred_questions.records() == []
            await ctx.session.respond_to_callback(
                callback.callback_id, _answer_output()
            )
            await asyncio.wait_for(_consume(stream), timeout=10)
            result = UserQuestionResult.model_validate(
                _completed_effect_output(ctx.session.history)
            )
            assert result.disposition == "answered"
            assert ctx.loop.deferred_questions.records() == []

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_default_enabled_never_expires_a_runtime_approval(
        self, tmp_path: Path
    ) -> None:
        """(a3) Default-enabled + a runtime approval (root access): the
        approval shares the callback plumbing but is permanently ineligible —
        it stays unanswered indefinitely and grants nothing.
        """
        project = tmp_path / "project"
        project.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "note.txt"
        target.write_text("granted content")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="", tool_calls=[_read_tool_call(target, "read-1")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        config = _default_enabled_config(enabled_tools=["read_file"])
        loop = build_test_agent_loop(
            config=config, backend=backend, cwd=project, user_input_capability=True
        )
        server, session = await _open_session(loop)
        clock = _ManualClock()
        wakeup = _GatedWakeup()
        _patch_timer(server, clock, wakeup)
        try:
            stream = session.act("read the note")
            callback = (await _next_event(stream, CallbackRequested)).callback
            detail = callback.detail
            assert isinstance(detail, UserInputCallbackDetail)
            # The runtime approval is permanently ineligible: no expiry policy.
            assert detail.unattended_expiry is None
            question = detail.request.questions[0]
            assert question.header == "Grant access"
            # No timer is armed even though the shipped default is enabled.
            await asyncio.sleep(0.05)
            assert wakeup.calls == []
            # Silence far past any deadline: still open, nothing granted.
            clock.advance(10_000.0)
            _release_gates(wakeup)
            await asyncio.sleep(0.05)
            assert [c.callback_id for c in session.state.active_callbacks] == [
                callback.callback_id
            ]
            assert isinstance(_callback_entry(session.history).state, OpenCallbackState)
            assert loop._session_root_grants.roots == set()
            assert not loop.tool_manager.workspace.allows(target)
            assert loop.deferred_questions.records() == []
            # Deny: the refusal is remembered; still nothing was granted.
            await session.respond_to_callback(
                callback.callback_id,
                UserInputCallbackOutput(
                    result=UserQuestionResult(
                        answers=[UserAnswer(question="grant", answer="Deny")]
                    )
                ),
            )
            await asyncio.wait_for(_consume(stream), timeout=10)
            assert loop._session_root_grants.roots == set()
            assert loop._session_root_grants.denied_roots == {outside.resolve()}
            assert loop.deferred_questions.records() == []
        finally:
            await session.close()
            await loop.aclose()

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_explicitly_disabled_config_still_never_expires(self) -> None:
        """(a4) An EXPLICITLY disabled ``[user_questions]`` still suppresses
        expiry under the promoted default (disabling cancels future
        scheduling without discarding anything).
        """
        config = _default_enabled_config(enabled_tools=["ask_user_question"])
        config = config.model_copy(
            update={
                "user_questions": UserQuestionsConfig(unattended_timeout_enabled=False)
            }
        )
        assert config.user_questions.unattended_timeout_enabled is False
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_defaultable_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            config, backend, enable_streaming=True, user_input_capability=True
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            assert callback.detail.unattended_expiry is None
            await asyncio.sleep(0.05)
            assert ctx.wakeup.calls == []
            ctx.clock.advance(10_000.0)
            _release_gates(ctx.wakeup)
            await asyncio.sleep(0.05)
            assert [c.callback_id for c in ctx.session.state.active_callbacks] == [
                callback.callback_id
            ]
            assert isinstance(
                _callback_entry(ctx.session.history).state, OpenCallbackState
            )
            assert ctx.loop.deferred_questions.records() == []
            # The question stays answerable: nothing was discarded.
            await ctx.session.respond_to_callback(
                callback.callback_id, _answer_output()
            )
            await asyncio.wait_for(_consume(stream), timeout=10)
            result = UserQuestionResult.model_validate(
                _completed_effect_output(ctx.session.history)
            )
            assert result.disposition == "answered"
            assert result.answers[0].answer == "Yes"
            assert ctx.loop.deferred_questions.records() == []


# ---------------------------------------------------------------------------
# [recovery.subagents] — the default-enabled controller matrix (from WP4)
# ---------------------------------------------------------------------------


def _default_enabled_subagent_policy() -> RecoveryConfig:
    """The SHIPPED default ``[recovery.subagents]`` policy (OD5 promotion).

    Only the deterministic test cooldowns/jitter are overridden; the enable
    flag is left at its shipped default, which is what the matrix pins.
    """
    config = RecoveryConfig(cooldowns_seconds=[5.0, 10.0], jitter_fraction=0.0)
    assert config.enabled is True
    return config


class TestSubagentRecoveryEnablement:
    """Default-enabled automatic recovery (WP4's round-3 symmetry amendment)."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_default_enabled_policy_automatically_recovers(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The default-enabled policy schedules and admits the successor of a
        recoverably failed child through the SAME resume_agent admission — with
        no user action and one shared identity.
        """
        harness = await _controller_registry(
            monkeypatch,
            _default_enabled_subagent_policy(),
            _Factory([
                [
                    _write_outcome("call-1", "step one"),
                    _empty_fault(),
                    _final_outcome("recovered"),
                ]
            ]),
        )
        resume_keys: list[str] = []
        original_resume = harness.registry.resume_agent

        async def resume_spy(
            agent_id: str,
            expected_run_id: str,
            idempotency_key: str,
            *,
            requester_session_id: str,
        ) -> Any:
            resume_keys.append(idempotency_key)
            return await original_resume(
                agent_id,
                expected_run_id,
                idempotency_key,
                requester_session_id=requester_session_id,
            )

        monkeypatch.setattr(harness.registry, "resume_agent", resume_spy)
        try:
            # The shipped default drove the controller (never an override).
            assert harness.parent.config.recovery.subagents.enabled is True
            agent_id, run_id, failed, record = await _suspend_child(harness)
            assert failed.failure is not None
            assert failed.failure.recovery.eligibility is RecoveryEligibility.ELIGIBLE
            # Suspension still happened (the reaper armed its suspension TTL);
            # the controller scheduled on the first configured cooldown.
            assert record.state is _AgentState.SUSPENDED
            assert 5.0 in harness.wakeup.calls
            assert 1800.0 in harness.wakeup.calls
            scheduled = _events_of(caplog, "scheduled")
            assert len(scheduled) == 1
            # No user action: the timer admits the successor.
            harness.driver.release(5.0)
            successor = await _until_admitted(harness, agent_id, run_id)
            assert successor.run_id != run_id
            assert successor.resumed_from_run_id == run_id
            assert record.latest_run_id == successor.run_id
            assert record.same_agent_resumes_used == 1
            # The automatic admission went through the SAME operation.
            assert len(resume_keys) == 1
            assert resume_keys[0].startswith("auto-recovery:")
            assert len(_events_of(caplog, "successor_start")) == 1
            # The successor completes on the SAME agent identity.
            final = await _wait_result(harness.registry, agent_id, successor.run_id)
            assert final.completed
            assert final.response == "recovered"
            assert len(_events_of(caplog, "success")) == 1
        finally:
            await harness.registry.drain_children()
            await harness.parent.aclose()

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_disabling_cancels_scheduling_without_discarding_evidence(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Disabling automation cancels the pending schedule without an
        admission and without discarding the failed run's evidence.
        """
        harness = await _controller_registry(
            monkeypatch,
            _default_enabled_subagent_policy(),
            _Factory([[_write_outcome("call-1", "step one"), _empty_fault()]]),
        )
        try:
            assert harness.parent.config.recovery.subagents.enabled is True
            agent_id, run_id, failed, record = await _suspend_child(harness)
            assert harness.controller._episodes != {}
            assert 5.0 in harness.wakeup.calls
            # Disable automation mid-schedule (a config reload, as WP4's own
            # reload test drives it).
            object.__setattr__(
                harness.parent.config,
                "recovery",
                RecoverySectionConfig(
                    subagents=RecoveryConfig(
                        enabled=False,
                        cooldowns_seconds=[5.0, 10.0],
                        jitter_fraction=0.0,
                    )
                ),
            )
            assert harness.parent.config.recovery.subagents.enabled is False
            # The timer wakes and refuses: no admission, the episode stops.
            harness.driver.release(5.0)
            await _until(lambda: harness.controller._episodes == {})
            assert [run.run_id for run in harness.runs(agent_id)] == [run_id]
            assert record.latest_run_id == run_id
            assert record.state is _AgentState.SUSPENDED
            exhaustion = _events_of(caplog, "exhaustion")
            assert len(exhaustion) == 1
            assert exhaustion[0]["detail"] == "disabled"
            # The evidence is not discarded: the failed run's typed envelope
            # and its pinned result stay retrievable.
            assert failed.failure is not None
            assert failed.failure.recovery.eligibility is RecoveryEligibility.ELIGIBLE
            again = await _wait_result(harness.registry, agent_id, run_id)
            assert again.run_id == run_id
            assert not again.completed
            assert again.failure is not None
            assert again.failure.recovery.eligibility is RecoveryEligibility.ELIGIBLE
        finally:
            await harness.registry.drain_children()
            await harness.parent.aclose()


# ---------------------------------------------------------------------------
# [recovery.root] — the default-enabled gate matrix (from R2)
# ---------------------------------------------------------------------------


def _default_root_gate() -> _GateHarness:
    """A gate harness whose policy is the SHIPPED default (OD4b promotion):
    the enable flag is left at its default, never overridden.
    """
    harness = _GateHarness()
    harness.config = RootRecoveryConfig()
    assert harness.config.enabled is True
    return harness


class TestRootRecoveryEnablement:
    """Default-enabled automatic root-turn recovery (R2's round-2 amendment)."""

    @pytest.mark.asyncio
    async def test_default_enabled_schedules_with_no_user_action(self) -> None:
        """A terminal transient failure schedules an automatic continuation
        under the DEFAULT config — no user action required.
        """
        harness = _default_root_gate()
        harness.capture(_failure())

        scheduled = await harness.schedule()
        assert scheduled.decision == "scheduled"
        assert scheduled.attempt == 1
        assert scheduled.limit == 2
        # due = the first configured backoff (30s) + zero pinned jitter.
        assert scheduled.delay_seconds == 30.0
        assert scheduled.eligible_class == "timeout"
        assert scheduled.episode_id == "root-auto-turn-1"
        assert scheduled.idempotency_key == "root-auto-recovery:root-auto-turn-1:1"

        ((detail, message),) = harness.notices("scheduled")
        assert detail.episode_id == "root-auto-turn-1"
        assert detail.attempt == 1
        assert detail.limit == 2
        assert detail.delay_seconds == 30.0
        assert "No user action is required" in message

    @pytest.mark.asyncio
    async def test_default_enabled_exhaustion_parks_to_the_manual_fallback(
        self,
    ) -> None:
        """The R1 manual fallback still exists under the default config: the
        shared budget exhausts, further scheduling is declined, and the
        exhausted notice keeps the failed turn and names the manual retry.
        """
        harness = _default_root_gate()
        harness.capture(_failure())
        first = await harness.schedule()
        assert first.decision == "scheduled"
        await harness.admit(first.episode_id, first.idempotency_key)
        harness.gate.observe_turn_started("turn-2", first.idempotency_key)

        harness.capture(_failure(turn_id="turn-2", observed_at=1010.0))
        second = await harness.schedule("turn-2")
        assert second.decision == "scheduled"
        assert second.episode_id == first.episode_id
        assert second.attempt == 2
        await harness.admit(second.episode_id, second.idempotency_key, "turn-2")
        harness.gate.observe_turn_started("turn-3", second.idempotency_key)

        harness.capture(_failure(turn_id="turn-3", observed_at=1020.0))
        declined = await harness.schedule("turn-3")
        assert declined.decision == "declined"
        assert declined.reason == "budget_exhausted"
        assert declined.limit == 2

        ((exhausted, message),) = harness.notices("exhausted")
        assert exhausted.attempt == 2
        assert exhausted.reason == "budget_exhausted"
        assert "manual Retry" in message
        assert "The failed turn is kept" in message
        assert len(harness.notices("exhausted")) == 1

    @pytest.mark.asyncio
    async def test_disabling_cancels_scheduling_without_discarding_evidence(
        self,
    ) -> None:
        """Disabling automation cancels the pending schedule without an
        admission and without discarding the captured evidence.
        """
        harness = _default_root_gate()
        harness.capture(_failure())
        scheduled = await harness.schedule()
        assert scheduled.decision == "scheduled"

        # Disable automation mid-schedule.
        harness.config = RootRecoveryConfig(enabled=False)
        declined = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
        assert declined.decision == "declined"
        assert declined.reason == "disabled"
        ((cancelled, _message),) = harness.notices("cancelled")
        assert cancelled.reason == "disabled"

        # The evidence is not discarded: the captured typed projection for the
        # failed turn survives, and both transcript notices remain.
        assert "turn-1" in harness.gate._projections
        milestones = [entry[0].milestone for entry in harness.staged]
        assert milestones == ["scheduled", "cancelled"]
