"""Bounded deferred-question records for unattended question expiry (plan U/OD1).

When an eligible ``ask_user_question`` call expires unanswered, the app
server's callback-record timer resolves the tool with a server-owned
``UnansweredQuestions`` disposition. This module owns the honest record of
that event: one record per unanswered question batch, keyed by session and
request, bounded, and carried through structural compaction state exactly
like ``TodoState`` and the working-set file state
(``compaction/manager.py``:98-126 pattern — the envelope rides in a
``PersistedToolResult`` output on a compaction-boundary message, and
:meth:`DeferredQuestionStore.replay` restores it from a transcript).

Expiry defers work; it never approves anything. A deferred record is a
REPORT of that deferral for the consolidated blocker report (WP5), which
distinguishes a deferred preference from an actual blocker from a
still-open authorization decision: authorization/approval questions never
expire (``requires_user`` blocks the batch; runtime approvals are
permanently ineligible), so they can never appear in this store — an open
authorization decision is one that never declared a default.

Bounding never silently drops items: the store keeps the most recent
records up to the cap and counts the overflowed ones, so a report can
disclose the overflow honestly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.llm_models import LLMMessage
from chartreux.questions import (
    QuestionKind,
    UnansweredQuestions,
    UnattendedDefaultAction,
)

# Bounded record count (code invariant, never a config escape hatch).
MAX_DEFERRED_QUESTIONS = 100

# The compaction-envelope key, mirroring FileStateStore's ``file_state``.
_ENVELOPE_KEY = "deferred_questions"


class DeferredQuestionRecord(BaseModel):
    """One unanswered question batch and the work it deferred (plan U)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    request_id: str
    question_ids: list[str] = Field(min_length=1)
    questions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Echoed questions with stable IDs, kinds, and declared defaults "
            "(UnansweredQuestion payloads)"
        ),
    )
    deferred_at_unix: float = Field(
        description="Wall-clock seconds when the question expired unanswered"
    )
    timeout_seconds: float = Field(gt=0.0)
    elapsed_seconds: float = Field(ge=0.0)
    reason: str = Field(default="unanswered_timeout")
    affected_work: str = Field(
        description=(
            "What the declared defaults defer, joined from the per-question "
            "default descriptions"
        )
    )
    follow_up: str | None = Field(
        default=None,
        description="A later explicit follow-up recorded against this request",
    )


class DeferredQuestionStore:
    """Bounded, oldest-first-evicted records of unanswered question batches.

    Owned by the root ``AgentLoop`` and read by the app server (which records
    on expiry) and by compaction (which carries the records across the
    structural boundary). Bounding is honest: when the cap is exceeded the
    OLDEST records are evicted and counted in ``overflowed``, never silently
    dropped.
    """

    def __init__(
        self,
        *,
        max_records: int = MAX_DEFERRED_QUESTIONS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._records: dict[tuple[str, str], DeferredQuestionRecord] = {}
        self._max_records = max_records
        self._clock = clock
        self.overflowed = 0

    def __len__(self) -> int:
        return len(self._records)

    def record_unanswered(
        self, *, session_id: str, request_id: str, unanswered: UnansweredQuestions
    ) -> DeferredQuestionRecord:
        """Record one expired-unanswered batch (exactly one per expiry)."""
        affected = "; ".join(
            f"{question.question_id}: {question.unattended_default.description}"
            for question in unanswered.questions
        )
        record = DeferredQuestionRecord(
            session_id=session_id,
            request_id=request_id,
            question_ids=[question.question_id for question in unanswered.questions],
            questions=[
                question.model_dump(mode="json") for question in unanswered.questions
            ],
            deferred_at_unix=self._clock(),
            timeout_seconds=unanswered.timeout_seconds,
            elapsed_seconds=unanswered.elapsed_seconds,
            affected_work=affected,
        )
        key = (session_id, request_id)
        self._records.pop(key, None)
        self._records[key] = record
        while len(self._records) > self._max_records:
            oldest = next(iter(self._records))
            self._records.pop(oldest, None)
            self.overflowed += 1
        return record

    def note_follow_up(
        self, *, session_id: str, request_id: str, follow_up: str
    ) -> bool:
        """Attach a later explicit follow-up to a deferred record."""
        record = self._records.get((session_id, request_id))
        if record is None:
            return False
        record.follow_up = follow_up
        return True

    def records(self) -> list[DeferredQuestionRecord]:
        """All retained records, oldest first."""
        return list(self._records.values())

    def render_summary(self) -> str:
        """Honest report text distinguishing deferred kinds (plan U/WP5).

        Deferred preferences/informational notes are not blockers; a
        ``defer_affected_work`` default is the batch that actually deferred
        work. Still-open authorization decisions never appear here (they
        never expire), so the report says so explicitly.
        """
        lines: list[str] = []
        for record in self.records():
            kinds = {QuestionKind(question["kind"]) for question in record.questions}
            actions = {
                UnattendedDefaultAction(question["unattended_default"]["action"])
                for question in record.questions
            }
            if kinds == {QuestionKind.PREFERENCE}:
                classification = "deferred preference (not a blocker)"
            elif kinds == {QuestionKind.INFORMATIONAL}:
                classification = "deferred informational note (not a blocker)"
            elif UnattendedDefaultAction.DEFER_AFFECTED_WORK in actions:
                classification = "deferred work (possible blocker)"
            else:
                classification = "deferred work (possible blocker)"
            follow_up = (
                f"; follow-up recorded: {record.follow_up}"
                if record.follow_up is not None
                else ""
            )
            lines.append(
                f"- request {record.request_id} in session {record.session_id}: "
                f"{classification}; {len(record.question_ids)} question(s) expired "
                f"unanswered after {record.elapsed_seconds:.0f}s "
                f"(timeout {record.timeout_seconds:.0f}s); deferred: "
                f"{record.affected_work}{follow_up}"
            )
        if self.overflowed:
            lines.append(
                f"- overflow: {self.overflowed} older deferred-question record(s) "
                "were evicted by the bounded store and are not listed above"
            )
        if not lines:
            lines.append(
                "- no deferred questions; still-open authorization decisions "
                "never expire and are not listed here"
            )
        return "\n".join(lines)

    def to_envelope(self) -> dict[str, Any]:
        """Serialize for the compaction envelope (mirrors FileStateStore)."""
        return {
            _ENVELOPE_KEY: [record.model_dump(mode="json") for record in self.records()]
        }

    @classmethod
    def from_envelope(cls, payload: Any) -> DeferredQuestionStore | None:
        """Restore from a compaction-envelope payload; None when absent/invalid."""
        if not isinstance(payload, dict):
            return None
        raw = payload.get(_ENVELOPE_KEY)
        if not isinstance(raw, list):
            return None
        store = cls()
        for item in raw:
            try:
                record = DeferredQuestionRecord.model_validate(item)
            except Exception:
                continue
            store._records[(record.session_id, record.request_id)] = record
        return store

    @classmethod
    def replay(cls, messages: Sequence[LLMMessage]) -> DeferredQuestionStore:
        """Restore the deferred-question records from a transcript.

        Structural replay across compaction, mirroring
        ``FileStateStore.replay``: the last compaction-boundary envelope in
        the transcript wins.
        """
        store = cls()
        for message in messages:
            if message.context_boundary != "compaction":
                continue
            if message.tool_result is None:
                continue
            restored = cls.from_envelope(message.tool_result.output)
            if restored is not None:
                store = restored
        return store

    def merge_replayed(self, replayed: DeferredQuestionStore) -> None:
        """Merge replayed records into the live store (replayed wins per key)."""
        for key, record in replayed._records.items():
            self._records.pop(key, None)
            self._records[key] = record
        while len(self._records) > self._max_records:
            oldest = next(iter(self._records))
            self._records.pop(oldest, None)
            self.overflowed += 1
