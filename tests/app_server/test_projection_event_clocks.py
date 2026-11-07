from __future__ import annotations

import pytest

from chartreux.app_server._projection import project_message_history
from chartreux.app_server.models import (
    PublicEffectEntry,
    PublicHistoryEntry,
    PublicReasoningEntry,
)
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall

REASONING_AT = 1_700_000_000_000
DISPATCH_AT = REASONING_AT + 90_000


def stamped_messages() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.user, content="question", message_id="u"),
        LLMMessage(
            role=Role.assistant,
            content="answer",
            message_id="a",
            reasoning_content="thinking",
            reasoning_at=REASONING_AT,
            tool_calls=[
                ToolCall(
                    id="c",
                    index=0,
                    function=FunctionCall(name="todo", arguments="{}"),
                    started_at=DISPATCH_AT,
                )
            ],
        ),
        LLMMessage(role=Role.tool, content="done", tool_call_id="c", name="todo"),
    ]


def unstamped_messages() -> list[LLMMessage]:
    """The same conversation as a pre-v0.7.4 record: no per-event clocks."""
    return [
        LLMMessage(role=Role.user, content="question", message_id="u"),
        LLMMessage(
            role=Role.assistant,
            content="answer",
            message_id="a",
            reasoning_content="thinking",
            tool_calls=[
                ToolCall(
                    id="c", index=0, function=FunctionCall(name="todo", arguments="{}")
                )
            ],
        ),
        LLMMessage(role=Role.tool, content="done", tool_call_id="c", name="todo"),
    ]


def entries_of(
    messages: list[LLMMessage],
) -> tuple[PublicReasoningEntry, PublicEffectEntry, list[PublicHistoryEntry]]:
    entries = project_message_history("session", messages, None)
    reasoning = next(e for e in entries if isinstance(e, PublicReasoningEntry))
    effect = next(e for e in entries if isinstance(e, PublicEffectEntry))
    return reasoning, effect, entries


def test_restored_entries_surface_the_persisted_real_clock() -> None:
    reasoning, effect, entries = entries_of(stamped_messages())

    assert reasoning.event_at == REASONING_AT
    assert effect.event_at == DISPATCH_AT
    # created_at stays the synthetic restore ordinal (timestamp + message
    # index): it is never the header source, so it never carries a real clock.
    assert reasoning.created_at != reasoning.event_at
    assert effect.created_at != effect.event_at
    # Every entry projected from the same message shares that message's ordinal.
    ordinal = [entry.created_at - entries[0].created_at for entry in entries]
    assert ordinal == [0, 1, 1, 1]


def test_legacy_records_surface_no_real_clock_despite_the_synthetic_ordinal() -> None:
    reasoning, effect, entries = entries_of(unstamped_messages())

    assert reasoning.event_at is None
    assert effect.event_at is None
    # The synthetic ordinal still exists for every record — a legacy record is
    # distinguishable from a stamped one only by the absent real clock.
    assert [entry.created_at - entries[0].created_at for entry in entries] == [
        0,
        1,
        1,
        1,
    ]


def test_legacy_entry_wire_payload_carries_no_event_clock() -> None:
    reasoning, effect, _ = entries_of(unstamped_messages())
    assert "eventAt" not in reasoning.model_dump(by_alias=True)
    assert "eventAt" not in effect.model_dump(by_alias=True)

    stamped_reasoning, stamped_effect, _ = entries_of(stamped_messages())
    assert stamped_reasoning.model_dump(by_alias=True)["eventAt"] == REASONING_AT
    assert stamped_effect.model_dump(by_alias=True)["eventAt"] == DISPATCH_AT


@pytest.mark.parametrize("entry_type", [PublicReasoningEntry, PublicEffectEntry])
def test_event_clock_is_optional_on_public_entries(
    entry_type: type[PublicReasoningEntry] | type[PublicEffectEntry],
) -> None:
    assert entry_type.model_fields["event_at"].is_required() is False
