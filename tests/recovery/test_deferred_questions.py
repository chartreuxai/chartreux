"""Deferred-question records for unattended question expiry (plan U/OD1, stage 3).

Unit coverage for ``chartreux/core/deferred_questions.py`` — the bounded,
honest record of unanswered question batches — plus the two integration
seams that carry it: the compaction envelope (``CompactionManager``'s
``deferred_questions_getter``) and the resume replay
(``SessionTransitions.restore_deferred_questions``). The end-to-end expiry
path (eligible silence produces exactly one unanswered result and one
deferred record; compaction carries and replay restores) lives in
``tests/recovery/test_question_timeout.py``.

Expiry defers work; it never approves anything. Authorization/approval
questions never expire (``requires_user`` blocks the batch; runtime
approvals are permanently ineligible), so they can never appear in this
store — an open authorization decision is one that never declared a
default, and ``render_summary`` says so explicitly.
"""

from __future__ import annotations

from typing import Any

import pytest

from chartreux.core.deferred_questions import (
    MAX_DEFERRED_QUESTIONS,
    DeferredQuestionRecord,
    DeferredQuestionStore,
)
from chartreux.core.llm_models import LLMMessage, PersistedToolResult, Role
from chartreux.questions import (
    QuestionKind,
    UnansweredQuestion,
    UnansweredQuestions,
    UnattendedDefault,
    UnattendedDefaultAction,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_SESSION = "session-1"
_DESCRIPTION = "Defer the release until the user answers."


def _echoed_question(
    question_id: str,
    *,
    kind: QuestionKind = QuestionKind.PREFERENCE,
    description: str = _DESCRIPTION,
    action: UnattendedDefaultAction = UnattendedDefaultAction.DEFER_AFFECTED_WORK,
) -> UnansweredQuestion:
    return UnansweredQuestion(
        question_id=question_id,
        question="Ship it?",
        kind=kind,
        unattended_default=UnattendedDefault(action=action, description=description),
    )


def _unanswered(
    *questions: UnansweredQuestion, elapsed: float = 61.0, timeout: float = 60.0
) -> UnansweredQuestions:
    return UnansweredQuestions(
        questions=list(questions),
        elapsed_seconds=elapsed,
        timeout_seconds=timeout,
        continuation_guidance="Continue without the user's answer.",
    )


def _record(
    store: DeferredQuestionStore,
    request_id: str,
    *questions: UnansweredQuestion,
    session_id: str = _SESSION,
) -> DeferredQuestionRecord:
    return store.record_unanswered(
        session_id=session_id, request_id=request_id, unanswered=_unanswered(*questions)
    )


def _boundary_message(envelope: dict[str, Any]) -> LLMMessage:
    return LLMMessage(
        role=Role.user,
        content="Compacted context",
        injected=True,
        context_boundary="compaction",
        tool_result=PersistedToolResult(output=envelope),
    )


class TestRecordUnanswered:
    """One record per unanswered batch, keyed by session and request."""

    def test_record_captures_the_batch(self) -> None:
        store = DeferredQuestionStore(clock=lambda: 1_700_000_000.0)
        record = _record(
            store,
            "req-1",
            _echoed_question("req-1:0"),
            _echoed_question("req-1:1", description="Defer the deploy."),
        )
        assert record.session_id == _SESSION
        assert record.request_id == "req-1"
        assert record.question_ids == ["req-1:0", "req-1:1"]
        assert record.deferred_at_unix == 1_700_000_000.0
        assert record.timeout_seconds == 60.0
        assert record.elapsed_seconds == 61.0
        assert record.reason == "unanswered_timeout"
        # The deferred work is the joined per-question default descriptions.
        assert record.affected_work == (
            f"req-1:0: {_DESCRIPTION}; req-1:1: Defer the deploy."
        )
        assert record.follow_up is None
        # The echoed questions carry their stable IDs, kinds, and defaults.
        assert [q["question_id"] for q in record.questions] == ["req-1:0", "req-1:1"]
        assert record.questions[0]["kind"] == QuestionKind.PREFERENCE.value
        assert record.questions[0]["unattended_default"]["action"] == (
            UnattendedDefaultAction.DEFER_AFFECTED_WORK.value
        )
        assert store.records() == [record]

    def test_same_key_replaces_so_one_record_per_expiry(self) -> None:
        store = DeferredQuestionStore()
        first = _record(store, "req-1", _echoed_question("req-1:0"))
        second = _record(store, "req-1", _echoed_question("req-1:0"))
        assert len(store) == 1
        assert store.records() == [second]
        assert store.records()[0] is not first

    def test_distinct_keys_accumulate_oldest_first(self) -> None:
        store = DeferredQuestionStore()
        _record(store, "req-1", _echoed_question("req-1:0"))
        _record(store, "req-2", _echoed_question("req-2:0"))
        _record(store, "req-1", _echoed_question("req-1:0"), session_id="other")
        assert len(store) == 3
        assert [record.request_id for record in store.records()] == [
            "req-1",
            "req-2",
            "req-1",
        ]
        # The key is (session, request): the same request in another session
        # is a different record.
        assert {record.session_id for record in store.records()} == {_SESSION, "other"}


class TestBounding:
    """Bounding never silently drops items: the oldest are evicted and counted."""

    def test_oldest_evicted_and_counted(self) -> None:
        store = DeferredQuestionStore(max_records=3)
        for index in range(4):
            _record(store, f"req-{index}", _echoed_question(f"req-{index}:0"))
        assert len(store) == 3
        assert store.overflowed == 1
        # The evicted record is the OLDEST one.
        assert [record.request_id for record in store.records()] == [
            "req-1",
            "req-2",
            "req-3",
        ]

    def test_overflow_counts_every_eviction(self) -> None:
        store = DeferredQuestionStore(max_records=2)
        for index in range(5):
            _record(store, f"req-{index}", _echoed_question(f"req-{index}:0"))
        assert len(store) == 2
        assert store.overflowed == 3
        assert [record.request_id for record in store.records()] == ["req-3", "req-4"]

    def test_default_cap_is_the_code_invariant(self) -> None:
        store = DeferredQuestionStore()
        for index in range(MAX_DEFERRED_QUESTIONS + 1):
            _record(store, f"req-{index}", _echoed_question(f"req-{index}:0"))
        assert len(store) == MAX_DEFERRED_QUESTIONS
        assert store.overflowed == 1


class TestFollowUp:
    """A later explicit follow-up is recorded against the deferred record."""

    def test_note_follow_up_attaches(self) -> None:
        store = DeferredQuestionStore()
        _record(store, "req-1", _echoed_question("req-1:0"))
        assert store.note_follow_up(
            session_id=_SESSION, request_id="req-1", follow_up="User answered later."
        )
        assert store.records()[0].follow_up == "User answered later."

    def test_note_follow_up_unknown_key_returns_false(self) -> None:
        store = DeferredQuestionStore()
        assert not store.note_follow_up(
            session_id=_SESSION, request_id="missing", follow_up="Nope."
        )
        assert (
            store.note_follow_up(
                session_id="other", request_id="req-1", follow_up="Nope."
            )
            is False
        )


class TestRenderSummary:
    """The WP5 report distinguishes deferred preference vs blocker vs open."""

    def test_empty_store_reports_no_deferred_questions(self) -> None:
        summary = DeferredQuestionStore().render_summary()
        assert "no deferred questions" in summary
        # Still-open authorization decisions never expire and are not listed.
        assert "still-open authorization decisions" in summary

    def test_preference_batch_is_not_a_blocker(self) -> None:
        store = DeferredQuestionStore()
        _record(
            store, "req-1", _echoed_question("req-1:0", kind=QuestionKind.PREFERENCE)
        )
        summary = store.render_summary()
        assert "deferred preference (not a blocker)" in summary
        assert "req-1" in summary
        assert _SESSION in summary
        assert "1 question(s) expired unanswered after 61s (timeout 60s)" in summary
        assert _DESCRIPTION in summary

    def test_informational_batch_is_not_a_blocker(self) -> None:
        store = DeferredQuestionStore()
        _record(
            store, "req-1", _echoed_question("req-1:0", kind=QuestionKind.INFORMATIONAL)
        )
        assert "deferred informational note (not a blocker)" in (store.render_summary())

    def test_defer_affected_work_is_a_possible_blocker(self) -> None:
        store = DeferredQuestionStore()
        _record(
            store,
            "req-1",
            _echoed_question("req-1:0", kind=QuestionKind.PREFERENCE),
            _echoed_question("req-1:1", kind=QuestionKind.INFORMATIONAL),
        )
        # A mixed batch is classified by its declared actions.
        assert "deferred work (possible blocker)" in store.render_summary()

    def test_follow_up_and_overflow_are_disclosed(self) -> None:
        store = DeferredQuestionStore(max_records=1)
        _record(store, "req-1", _echoed_question("req-1:0"))
        _record(store, "req-2", _echoed_question("req-2:0"))
        assert store.note_follow_up(
            session_id=_SESSION, request_id="req-2", follow_up="Asked again."
        )
        summary = store.render_summary()
        assert "follow-up recorded: Asked again." in summary
        assert "overflow: 1 older deferred-question record(s)" in summary
        # The evicted record is not listed...
        assert "req-1" not in summary
        # ...but the surviving one is.
        assert "req-2" in summary


class TestEnvelope:
    """The compaction envelope round-trips; replay takes the last boundary."""

    def test_envelope_round_trip(self) -> None:
        store = DeferredQuestionStore(clock=lambda: 1_700_000_000.0)
        record = _record(store, "req-1", _echoed_question("req-1:0"))
        envelope = store.to_envelope()
        assert set(envelope) == {"deferred_questions"}
        restored = DeferredQuestionStore.from_envelope(envelope)
        assert restored is not None
        assert restored.records() == [record]
        assert restored.records()[0].affected_work == record.affected_work

    def test_from_envelope_rejects_malformed_payloads(self) -> None:
        assert DeferredQuestionStore.from_envelope(None) is None
        assert DeferredQuestionStore.from_envelope("deferred_questions") is None
        assert DeferredQuestionStore.from_envelope({}) is None
        assert DeferredQuestionStore.from_envelope({"deferred_questions": "x"}) is None
        # Invalid items are skipped, valid ones survive.
        restored = DeferredQuestionStore.from_envelope({
            "deferred_questions": [
                {"not": "a record"},
                DeferredQuestionRecord(
                    session_id=_SESSION,
                    request_id="req-1",
                    question_ids=["req-1:0"],
                    deferred_at_unix=1.0,
                    timeout_seconds=60.0,
                    elapsed_seconds=61.0,
                    affected_work="deferred",
                ).model_dump(mode="json"),
            ]
        })
        assert restored is not None
        assert [r.request_id for r in restored.records()] == ["req-1"]

    def test_replay_takes_the_last_compaction_boundary(self) -> None:
        store = DeferredQuestionStore()
        _record(store, "req-old", _echoed_question("req-old:0"))
        older = store.to_envelope()
        store = DeferredQuestionStore()
        _record(store, "req-new", _echoed_question("req-new:0"))
        newer = store.to_envelope()
        messages = [
            LLMMessage(role=Role.user, content="before"),
            _boundary_message(older),
            LLMMessage(role=Role.assistant, content="between"),
            # A boundary without a tool result is not an envelope.
            LLMMessage(
                role=Role.user,
                content="Compacted context",
                injected=True,
                context_boundary="compaction",
            ),
            _boundary_message(newer),
            LLMMessage(role=Role.user, content="after"),
        ]
        replayed = DeferredQuestionStore.replay(messages)
        assert [r.request_id for r in replayed.records()] == ["req-new"]

    def test_replay_of_a_transcript_without_boundaries_is_empty(self) -> None:
        messages = [
            LLMMessage(role=Role.user, content="hello"),
            LLMMessage(role=Role.assistant, content="hi"),
        ]
        replayed = DeferredQuestionStore.replay(messages)
        assert len(replayed) == 0
        assert replayed.overflowed == 0

    def test_replay_ignores_a_corrupt_envelope(self) -> None:
        messages = [
            _boundary_message({"deferred_questions": "corrupt"}),
            LLMMessage(role=Role.user, content="after"),
        ]
        replayed = DeferredQuestionStore.replay(messages)
        assert len(replayed) == 0


class TestMergeReplayed:
    """The resume merge: replayed wins per key; post-boundary records survive."""

    def test_replayed_wins_per_key_and_live_records_survive(self) -> None:
        live = DeferredQuestionStore()
        # A record created AFTER the compaction boundary (the live session
        # expired a question the boundary never saw).
        _record(live, "req-live", _echoed_question("req-live:0"))
        # The live store also holds a stale copy of a replayed key.
        _record(live, "req-shared", _echoed_question("req-shared:0"))
        replayed = DeferredQuestionStore()
        replayed_record = _record(
            replayed, "req-shared", _echoed_question("req-shared:0")
        )
        _record(replayed, "req-replayed", _echoed_question("req-replayed:0"))
        live.merge_replayed(replayed)
        by_request = {record.request_id: record for record in live.records()}
        assert set(by_request) == {"req-live", "req-shared", "req-replayed"}
        # Replayed wins for the shared key...
        assert by_request["req-shared"] is replayed_record
        # ...and the post-boundary live record survives untouched.
        assert by_request["req-live"].session_id == _SESSION

    def test_merge_applies_the_bound(self) -> None:
        live = DeferredQuestionStore(max_records=2)
        replayed = DeferredQuestionStore()
        for index in range(3):
            _record(replayed, f"req-{index}", _echoed_question(f"req-{index}:0"))
        live.merge_replayed(replayed)
        assert len(live) == 2
        assert live.overflowed == 1
        assert [record.request_id for record in live.records()] == ["req-1", "req-2"]


class TestSessionTransitionsRestore:
    """The resume replay merges the transcript into the loop's live store."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_restore_merges_transcript_into_the_live_store(self) -> None:
        loop = build_test_agent_loop()
        try:
            # The live store holds a record created after the boundary.
            _record(loop.deferred_questions, "req-live", _echoed_question("req-live:0"))
            # The transcript carries a boundary envelope with an older record.
            carried = DeferredQuestionStore()
            _record(carried, "req-carried", _echoed_question("req-carried:0"))
            messages = [
                LLMMessage(role=Role.user, content="ask me"),
                _boundary_message(carried.to_envelope()),
                LLMMessage(role=Role.assistant, content="Done"),
            ]
            loop._session_transitions.restore_deferred_questions(messages)
            by_request = {
                record.request_id: record
                for record in loop.deferred_questions.records()
            }
            assert set(by_request) == {"req-live", "req-carried"}
            assert by_request["req-carried"].affected_work == (
                f"req-carried:0: {_DESCRIPTION}"
            )
        finally:
            await loop.aclose()

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_restore_is_fail_safe_on_a_corrupt_envelope(self) -> None:
        loop = build_test_agent_loop()
        try:
            _record(loop.deferred_questions, "req-live", _echoed_question("req-live:0"))
            messages = [_boundary_message({"deferred_questions": "corrupt"})]
            # A corrupt envelope degrades to the live state, never raises.
            loop._session_transitions.restore_deferred_questions(messages)
            assert [r.request_id for r in loop.deferred_questions.records()] == [
                "req-live"
            ]
        finally:
            await loop.aclose()


class TestCompactionCarriesRecords:
    """The compaction boundary carries the store's records in its envelope."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_compact_boundary_carries_the_deferred_records(self) -> None:
        backend = FakeBackend([
            [mock_llm_chunk(content="<summary>summary of the conversation</summary>")]
        ])
        loop = build_test_agent_loop(config=build_test_vibe_config(), backend=backend)
        try:
            loop.messages.append(LLMMessage(role=Role.user, content="ask me"))
            record = _record(
                loop.deferred_questions, "req-1", _echoed_question("req-1:0")
            )
            summary = await loop.compact()
            assert summary == "summary of the conversation"
            boundaries = [
                message
                for message in loop.messages
                if message.context_boundary == "compaction"
            ]
            assert len(boundaries) == 1
            tool_result = boundaries[0].tool_result
            assert tool_result is not None
            carried = tool_result.output["deferred_questions"]
            assert isinstance(carried, list)
            assert len(carried) == 1
            first = carried[0]
            assert isinstance(first, dict)
            assert first["request_id"] == "req-1"
            assert first["session_id"] == record.session_id
            # A replay of the compacted transcript restores the record into a
            # fresh store (the structural-boundary round trip).
            replayed = DeferredQuestionStore.replay(list(loop.messages))
            assert [r.request_id for r in replayed.records()] == ["req-1"]
            assert replayed.records()[0].affected_work == record.affected_work
        finally:
            await loop.aclose()
