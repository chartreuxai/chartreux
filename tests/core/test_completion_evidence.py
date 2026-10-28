"""P3: bounded per-run completion-evidence chronology tests.

Receipts pair with their originating invocations via the harness-minted,
run-scoped receipt occurrence ID — never tool_call_id alone (tool-call
ids are reused across turns). Caps evict the OLDEST entries into typed
markers that classify as unavailable, never contradictory.
"""

from __future__ import annotations

from chartreux.core.agent_loop_hooks import CompletionEvidenceReceipt
from chartreux.core.completion_evidence import PerRunCompletionChronology
from chartreux.core.events import ToolResultEvent


def _result_event(occurrence_id: str, tool_call_id: str) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="stub_tool",
        tool_class=None,
        tool_call_id=tool_call_id,
        occurrence_id=occurrence_id,
    )


def _receipt(occurrence_id: str, tool_call_id: str) -> CompletionEvidenceReceipt:
    return CompletionEvidenceReceipt(
        occurrence_id=occurrence_id,
        tool_call_id=tool_call_id,
        tool_name="stub_tool",
        status="success",
        final_text="ok",
    )


def test_sequence_is_monotonic_across_registrations() -> None:
    chronology = PerRunCompletionChronology()
    first = "occ-1"
    second = "occ-2"
    chronology.observe_result_event(_result_event(first, "call-a"))
    chronology.observe_result_event(_result_event(second, "call-a"))

    assert chronology.resolution(first) == "collected"
    assert chronology.resolution(second) == "collected"
    assert chronology.resolution("occ-never-seen") == "unavailable"
    first_entry = chronology.entry(first)
    second_entry = chronology.entry(second)
    assert first_entry is not None and second_entry is not None
    assert first_entry.sequence < second_entry.sequence


def test_reused_tool_call_id_resolves_only_to_its_own_invocation() -> None:
    """Two turns reusing one tool_call_id — a failed check followed by a
    successful retry — each receipt resolves ONLY to its originating
    invocation, whichever side of the pair arrived first.
    """
    chronology = PerRunCompletionChronology()
    failed_occurrence = "occ-failed"
    retry_occurrence = "occ-retry"
    failed_event = _result_event(failed_occurrence, "reused")
    retry_event = _result_event(retry_occurrence, "reused")
    failed_receipt = _receipt(failed_occurrence, "reused")
    retry_receipt = _receipt(retry_occurrence, "reused")

    # Callback-before-observation order: the receipts arrive first.
    chronology.observe_receipt(failed_receipt)
    chronology.observe_receipt(retry_receipt)
    chronology.observe_result_event(failed_event)
    chronology.observe_result_event(retry_event)

    failed_entry = chronology.entry(failed_occurrence)
    retry_entry = chronology.entry(retry_occurrence)
    assert failed_entry is not None and retry_entry is not None
    assert failed_entry.receipt is failed_receipt
    assert failed_entry.result_event is failed_event
    assert retry_entry.receipt is retry_receipt
    assert retry_entry.result_event is retry_event


def test_identical_occurrence_id_across_dispatch_and_collection() -> None:
    """The executor-minted occurrence ID is registered UNCHANGED."""
    chronology = PerRunCompletionChronology()
    minted = "run-scoped-occurrence-id"
    chronology.observe_result_event(_result_event(minted, "call-ok"))

    entry = chronology.entry(minted)
    assert entry is not None
    assert entry.occurrence_id == minted


def test_event_without_a_minted_id_resolves_unavailable() -> None:
    """A result event lacking the minted occurrence ID cannot be paired;
    fabricating a key would be reminting, so it resolves unavailable.
    """
    chronology = PerRunCompletionChronology()
    chronology.observe_result_event(_result_event("", "call-nomint"))

    assert chronology.resolution("") == "unavailable"


def test_eviction_marks_the_oldest_entries_unavailable() -> None:
    chronology = PerRunCompletionChronology(max_entries=2)
    for index in range(3):
        chronology.observe_result_event(_result_event(f"occ-{index}", f"call-{index}"))

    assert chronology.resolution("occ-0") == "evicted"
    assert chronology.resolution("occ-1") == "collected"
    assert chronology.resolution("occ-2") == "collected"
    assert chronology.entry("occ-0") is None
