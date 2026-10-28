"""P3: bounded per-run completion-evidence chronology.

The accumulator-side half of the completion-evidence model: pair the P2
side-channel receipts with the typed-stream result events of their
originating invocations, keyed by the harness-minted, run-scoped receipt
occurrence ID — never tool_call_id alone (tool-call ids are reused across
turns, so pairing by tool_call_id alone can associate evidence with the
wrong invocation) — and capture a bounded content snapshot of each
collected artifact AT COLLECTION TIME. Snapshot capture happens at
collection, never at finalization: finalization-time capture would
violate the worker-evidence versus shared-root-final-state distinction.
Evidence that outlives the chronology's caps is evicted into typed
markers classified unavailable — never contradictory.
"""

from __future__ import annotations

from enum import StrEnum
import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from chartreux.core.agent_loop_hooks import CompletionEvidenceReceipt
from chartreux.core.events import ToolResultEvent

__all__ = [
    "ARTIFACT_SNAPSHOT_CAP_BYTES",
    "MAX_CHRONOLOGY_ENTRIES",
    "EvictionMarker",
    "EvidenceAvailability",
    "PerRunCompletionChronology",
    "snapshot_bytes",
]


#: Total size cap of one collected artifact's content snapshot (~4KB).
ARTIFACT_SNAPSHOT_CAP_BYTES = 4096

#: Entry cap of one run's chronology before eviction starts.
MAX_CHRONOLOGY_ENTRIES = 128


class EvidenceAvailability(StrEnum):
    """The disposition vocabulary of one chronology entry.

    Eviction is availability accounting, never a worker disposition: an
    evicted entry classifies exactly as an unavailable one (check-context),
    so a cap can never manufacture a contradiction.
    """

    COLLECTED = "collected"
    EVICTED = "evicted"
    UNAVAILABLE = "unavailable"


def snapshot_bytes(
    content: bytes,
    *,
    head_bytes: int = ARTIFACT_SNAPSHOT_CAP_BYTES // 2,
    tail_bytes: int = ARTIFACT_SNAPSHOT_CAP_BYTES // 2,
) -> ContentSnapshot:
    """Hash one artifact's full content and sample its bounded head/tail.

    The hash covers the WHOLE content; the samples are capped so the
    snapshot's retained bytes stay bounded regardless of artifact size.
    """
    return ContentSnapshot(
        content_hash=hashlib.sha256(content).hexdigest(),
        byte_size=len(content),
        head_bytes=content[:head_bytes],
        tail_bytes=content[-tail_bytes:] if len(content) > head_bytes else b"",
        truncated=len(content) > head_bytes + tail_bytes,
    )


class ContentSnapshot(BaseModel):
    """The bounded collection-time witness for one collected artifact.

    Captured once, at collection — never at finalization — so the audit
    consumes what the run's collection observed, not the shared root's
    final state.
    """

    model_config = ConfigDict(frozen=True)

    content_hash: str
    byte_size: int
    head_bytes: bytes
    tail_bytes: bytes
    truncated: bool


class EvictionMarker(BaseModel):
    """The typed replacement for a chronology entry lost to a cap.

    Eviction is an availability fact about the HARNESS (the bounded
    chronology), never a worker fault: a consumer resolves an evicted
    entry to unavailable evidence, never to a contradiction.
    """

    model_config = ConfigDict(frozen=True)

    sequence: int
    occurrence_id: str | None
    availability: Literal[EvidenceAvailability.EVICTED] = EvidenceAvailability.EVICTED


class CollectedInvocation(BaseModel):
    """One invocation's paired evidence inside the chronology.

    The side-channel receipt and the typed-stream result event join here
    by the same run-scoped occurrence ID; either side alone still
    resolves — the chronology records which sides arrived.
    """

    model_config = ConfigDict(frozen=True)

    sequence: int
    occurrence_id: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    receipt: CompletionEvidenceReceipt | None = None
    result_event: ToolResultEvent | None = None
    snapshot: ContentSnapshot | None = None


class PerRunCompletionChronology:
    """Bounded per-run pairing of receipts with their originating invocations.

    Entries key on the harness-minted, run-scoped receipt occurrence ID;
    tool_call_id is carried for display and never keys pairing. Caps
    evict the OLDEST entries first, into typed markers — an eviction is
    an availability fact (unavailable), never a contradiction.
    """

    def __init__(self, *, max_entries: int = MAX_CHRONOLOGY_ENTRIES) -> None:
        self._max_entries = max_entries
        self._sequence = 0
        self._entries: dict[str, CollectedInvocation] = {}
        self._evicted: dict[str, EvictionMarker] = {}

    def observe_receipt(self, receipt: CompletionEvidenceReceipt) -> None:
        """Register one side-channel receipt against its invocation entry."""
        self._register(receipt.occurrence_id, "receipt", receipt)

    def observe_result_event(
        self, event: ToolResultEvent, *, snapshot: ContentSnapshot | None = None
    ) -> None:
        """Register one typed-stream result event against its invocation.

        ``snapshot`` carries the invocation's collection-time content
        witness (captured by the accumulator at collection, never at
        finalization) and is stored on the entry unchanged — the receipt
        audit reads it back as the mutation witness.
        """
        if not event.occurrence_id:
            # No minted key — the invocation cannot be paired and resolves
            # UNAVAILABLE (never contradictory). Fabricating a key would be
            # reminting, which is prohibited.
            return
        self._register(event.occurrence_id, "result_event", event)
        if snapshot is not None:
            entry = self._entries[event.occurrence_id]
            self._entries[event.occurrence_id] = entry.model_copy(
                update={"snapshot": snapshot}
            )

    def _register(self, occurrence_id: str, side: str, payload: Any) -> None:
        self._sequence += 1
        entry = self._entries.get(occurrence_id)
        if entry is None:
            entry = CollectedInvocation(
                sequence=self._sequence, occurrence_id=occurrence_id
            )
            self._entries[occurrence_id] = entry
        self._entries[occurrence_id] = entry.model_copy(update={side: payload})
        while len(self._entries) > self._max_entries:
            oldest_key = next(iter(self._entries))
            oldest = self._entries.pop(oldest_key)
            self._evicted[oldest_key] = EvictionMarker(
                sequence=oldest.sequence, occurrence_id=oldest.occurrence_id
            )

    def resolution(self, occurrence_id: str) -> EvidenceAvailability:
        """Resolve one occurrence ID's evidence availability."""
        if occurrence_id in self._entries:
            return EvidenceAvailability.COLLECTED
        if occurrence_id in self._evicted:
            return EvidenceAvailability.EVICTED
        return EvidenceAvailability.UNAVAILABLE

    def entry(self, occurrence_id: str) -> CollectedInvocation | None:
        """The collected entry for one occurrence ID, if not yet evicted."""
        return self._entries.get(occurrence_id)
