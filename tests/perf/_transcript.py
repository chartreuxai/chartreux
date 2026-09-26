from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from chartreux.app_server.models import (
    CompletedEffectState,
    EffectCallDisplay,
    EffectResultDisplay,
    GenericEffectDetail,
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicReasoningEntry,
    ShellEffectDetail,
    ShellEffectInput,
    ShellEffectOutput,
    TextContentBlock,
)

TRANSCRIPT_WIDTH = 120
TRANSCRIPT_HEIGHT = 40
SHELL_LINE_COUNT = 20_000
OVERSIZED_GROUP_LINE_COUNT = 1_601
_SUPPORTED_ENTRY_COUNTS = (300, 400, 500)


class _EntryBase(TypedDict):
    id: str
    session_id: str
    turn_id: str
    created_at: int
    updated_at: int
    generation_status: PublicEntryGenerationStatus


@dataclass(frozen=True, slots=True)
class TranscriptFixture:
    """Deterministic public history used by the transcript-windowing scenario."""

    entries: list[PublicHistoryEntry]
    entry_count: int
    turn_count: int
    checkpoint_count: int
    ordinary_message_count: int
    large_shell_entry_id: str
    oversized_group_entry_id: str
    oversized_group_line_count: int


def _message(
    index: int,
    *,
    role: Literal["user", "assistant"],
    text: str,
    turn_id: str | None = None,
) -> PublicMessageEntry:
    return PublicMessageEntry(
        id=f"transcript-message-{index:04d}",
        session_id="transcript-perf-session",
        turn_id=turn_id,
        created_at=index,
        updated_at=index,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        role=role,
        content=[TextContentBlock(text=text)],
        source="harness",
    )


def _call_display(summary: str) -> EffectCallDisplay:
    return EffectCallDisplay(
        summary=summary,
        verb="Running",
        message=summary,
        settled_verb="Ran",
        settled_message=summary,
        status_text="Running synthetic tool",
    )


def _result_display(message: str) -> EffectResultDisplay:
    return EffectResultDisplay(success=True, verb="Returned", message=message)


def _small_shell_output(turn_number: int) -> str:
    return f"shell-result-{turn_number:03d}\nexit: 0\n"


def _large_shell_output() -> str:
    return "".join(f"{line:05d} {'x' * 74}\n" for line in range(SHELL_LINE_COUNT))


def _oversized_group_output() -> str:
    return "\n".join(
        f"expanded-group-row-{line:04d}" for line in range(OVERSIZED_GROUP_LINE_COUNT)
    )


def _entry_base(entry_id: str, index: int, turn_id: str) -> _EntryBase:
    return {
        "id": entry_id,
        "session_id": "transcript-perf-session",
        "turn_id": turn_id,
        "created_at": index,
        "updated_at": index,
        "generation_status": PublicEntryGenerationStatus.COMPLETED,
    }


def _generic_effect(
    index: int, turn_number: int, *, oversized: bool, output: str
) -> PublicEffectEntry:
    effect_id = f"transcript-generic-effect-{turn_number:03d}"
    return PublicEffectEntry(
        **_entry_base(effect_id, index, f"transcript-turn-{turn_number:03d}"),
        title="Inspect synthetic data",
        detail=GenericEffectDetail(
            tool_name="synthetic_lookup",
            input={"turn": turn_number},
            display=_call_display("Inspect synthetic data"),
        ),
        state=CompletedEffectState(
            output={"content": output},
            output_text=output,
            display=_result_display(
                "Expanded non-shell group fixture" if oversized else "Small result"
            ),
        ),
    )


def _shell_effect(index: int, turn_number: int, *, output: str) -> PublicEffectEntry:
    effect_id = f"transcript-shell-effect-{turn_number:03d}"
    command = f"printf synthetic-shell-{turn_number:03d}"
    shell_output = ShellEffectOutput(stdout=output, stderr="", output=output)
    return PublicEffectEntry(
        **_entry_base(effect_id, index, f"transcript-turn-{turn_number:03d}"),
        title="Run synthetic shell command",
        detail=ShellEffectDetail(
            tool_name="bash",
            input=ShellEffectInput(command=command),
            display=_call_display(command),
        ),
        state=CompletedEffectState(
            output=shell_output.model_dump(mode="json"),
            output_text=output,
            display=_result_display("Synthetic shell output"),
        ),
    )


def build_transcript_fixture(entry_count: int = 400) -> TranscriptFixture:
    """Build the 300/400/500-entry composition with stable entry identities.

    Each turn contributes six entries.  After each three-turn block, one
    compaction checkpoint and one ordinary message are inserted.  The 400-entry
    case is therefore exactly 60 * 6 + 20 + 20, with proportional 300/500
    variants for scaling.
    """
    if entry_count not in _SUPPORTED_ENTRY_COUNTS:
        raise ValueError(f"entry_count must be one of {_SUPPORTED_ENTRY_COUNTS}")

    turn_count = entry_count * 3 // 20
    checkpoint_count = turn_count // 3
    ordinary_message_count = checkpoint_count
    large_shell_turn = turn_count // 4
    oversized_group_turn = turn_count // 2
    large_shell_entry_id = f"transcript-shell-effect-{large_shell_turn:03d}"
    oversized_group_entry_id = f"transcript-generic-effect-{oversized_group_turn:03d}"
    oversized_group_output = _oversized_group_output()
    large_shell_output = _large_shell_output()

    entries: list[PublicHistoryEntry] = []
    index = 0
    for turn_number in range(turn_count):
        turn_id = f"transcript-turn-{turn_number:03d}"
        prose = (
            f"Turn {turn_number:03d} summarizes the synthetic transcript in wrapped "
            "prose. This deliberately uses ordinary sentences rather than narrow "
            "tokens, so the rendered Markdown follows the terminal width and keeps "
            "reflow measurable across resize phases. "
        ) * 3
        entries.extend([
            _message(
                index,
                role="user",
                text=f"Please inspect fixture item {turn_number:03d}.",
                turn_id=turn_id,
            ),
            _message(
                index + 1,
                role="assistant",
                text=(
                    f"## Synthetic turn {turn_number:03d}\n\n"
                    f"Markdown summary with **stable content** and `{turn_id}`."
                ),
                turn_id=turn_id,
            ),
            PublicReasoningEntry(
                **_entry_base(
                    f"transcript-reasoning-{turn_number:03d}", index + 2, turn_id
                ),
                text=f"Reasoning for {turn_id}. {prose}",
            ),
            _generic_effect(
                index + 3,
                turn_number,
                oversized=turn_number == oversized_group_turn,
                output=(
                    oversized_group_output
                    if turn_number == oversized_group_turn
                    else f"small synthetic result for turn {turn_number:03d}"
                ),
            ),
            _shell_effect(
                index + 4,
                turn_number,
                output=(
                    large_shell_output
                    if turn_number == large_shell_turn
                    else _small_shell_output(turn_number)
                ),
            ),
            _message(index + 5, role="assistant", text=prose, turn_id=turn_id),
        ])
        index += 6

        if (turn_number + 1) % 3 == 0:
            checkpoint_number = (turn_number + 1) // 3 - 1
            entries.append(
                PublicCheckpointEntry(
                    **_entry_base(
                        f"transcript-compaction-{checkpoint_number:03d}", index, turn_id
                    ),
                    kind="compaction",
                    message="Synthetic context compaction checkpoint",
                )
            )
            index += 1
            entries.append(
                _message(
                    index,
                    role="user",
                    text=(
                        f"Ordinary inter-turn message {checkpoint_number:03d}: "
                        "continue with the next stable fixture page."
                    ),
                )
            )
            index += 1

    assert len(entries) == entry_count
    return TranscriptFixture(
        entries=entries,
        entry_count=entry_count,
        turn_count=turn_count,
        checkpoint_count=checkpoint_count,
        ordinary_message_count=ordinary_message_count,
        large_shell_entry_id=large_shell_entry_id,
        oversized_group_entry_id=oversized_group_entry_id,
        oversized_group_line_count=OVERSIZED_GROUP_LINE_COUNT,
    )
