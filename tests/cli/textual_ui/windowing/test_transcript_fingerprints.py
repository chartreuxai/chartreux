"""Equivalence oracle for the transcript entry fingerprints (P2a).

The pinned direction is *fingerprint equality implies deep-copy equality*: a
missed invalidation leaves a stale placement on screen, while a spurious
invalidation only costs a rebuild. Every case below is a class of pairs the
later equality cutover must keep safe, and each degenerate-payload case is
pinned against the guard that makes it safe.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from enum import Enum
from typing import Any, Literal, cast
from weakref import WeakKeyDictionary
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ValidationError
import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.geometry import Size
from textual.widget import Widget

from chartreux.app_server.models import (
    CompletedEffectState,
    ContentBlock,
    EffectCallDisplay,
    EffectResultDisplay,
    FailedEffectState,
    FileEditEffectDetail,
    FileImageSource,
    GenericEffectDetail,
    ImageAttachment,
    ImageContentBlock,
    InlineImageSource,
    OpenCallbackState,
    PublicCallbackEntry,
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicNoticeEntry,
    PublicReasoningEntry,
    ResourceContentBlock,
    RunningEffectState,
    SessionTitleUpdatedNoticeDetail,
    TextContentBlock,
    UserInputCallbackDetail,
)
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.windowing.fingerprints import (
    DIGEST_SIZE,
    fingerprint_entry,
)
from chartreux.cli.textual_ui.windowing.transcript import TranscriptWindow
from chartreux.questions import QuestionChoice, UserQuestion, UserQuestionRequest
from chartreux.user_content import UserDisplayContent, UserTextResource


def _base(entry_id: str) -> dict[str, Any]:
    return {
        "id": entry_id,
        "session_id": "session-1",
        "turn_id": "turn-1",
        "created_at": 1,
        "updated_at": 2,
        "generation_status": PublicEntryGenerationStatus.COMPLETED,
        "related_entry_id": "related-1",
    }


def _blocks() -> list[ContentBlock]:
    return [
        TextContentBlock(text="hello"),
        ImageContentBlock(
            attachment=ImageAttachment(
                source=InlineImageSource(data="aaaa"),
                alias="a.png",
                mime_type="image/png",
            )
        ),
        ResourceContentBlock(resource=UserTextResource(uri="file:///x", text="body")),
    ]


def _message(
    index: int = 1,
    *,
    blocks: Sequence[ContentBlock] | None = None,
    role: Literal["system", "user", "assistant"] = "assistant",
) -> PublicMessageEntry:
    return PublicMessageEntry(
        **_base(f"message-{index}"),
        role=role,
        content=list(blocks) if blocks is not None else _blocks(),
        posted_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        source="turn_start",
        turn_duration_ms=12.5,
        user_display_content=UserDisplayContent(
            version="1", host="host", content=[{"prompt": "x"}]
        ),
    )


def _reasoning(index: int = 1) -> PublicReasoningEntry:
    return PublicReasoningEntry(
        **_base(f"reasoning-{index}"),
        text="thinking",
        summary=["one", "two"],
        event_at=1700000000000,
    )


def _effect(
    index: int = 1, *, output: dict[str, Any] | None = None
) -> PublicEffectEntry:
    return PublicEffectEntry(
        **_base(f"effect-{index}"),
        title="tool",
        detail=GenericEffectDetail(
            tool_name="tool",
            display=EffectCallDisplay(summary="tool", status_text="run"),
        ),
        state=CompletedEffectState(
            display=EffectResultDisplay(success=True, message="done"),
            output={"rows": [1, 2]} if output is None else output,
            duration_ms=8.0,
        ),
        event_at=1700000000000,
    )


def _callback(index: int = 1) -> PublicCallbackEntry:
    request = UserQuestionRequest(
        questions=[
            UserQuestion(
                question="Pick one",
                options=[QuestionChoice(label="one"), QuestionChoice(label="two")],
            )
        ]
    )
    return PublicCallbackEntry(
        **_base(f"callback-{index}"),
        callback_id="callback-1",
        title="Input required",
        detail=UserInputCallbackDetail(request=request),
        state=OpenCallbackState(),
    )


def _checkpoint(index: int = 1) -> PublicCheckpointEntry:
    return PublicCheckpointEntry(
        **_base(f"checkpoint-{index}"),
        kind="compaction",
        message="compacted",
        details={"turn": 1, "files": {"a.py": {"lines": 3}}},
    )


def _notice(index: int = 1) -> PublicNoticeEntry:
    return PublicNoticeEntry(
        **_base(f"notice-{index}"),
        level="info",
        message="Title changed",
        detail=SessionTitleUpdatedNoticeDetail(title="new title"),
    )


_NY = ZoneInfo("America/New_York")


class _FoldAwareZone(tzinfo):
    """A DST-style zone: fold selects which of the two offsets applies."""

    def utcoffset(self, dt: datetime | None) -> timedelta:
        return timedelta(hours=-4 if dt is None or dt.fold == 0 else -5)

    def dst(self, dt: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, dt: datetime | None) -> str:
        return "fold-aware"


def _posted(value: datetime) -> PublicMessageEntry:
    """A message entry whose only varied field is its posting datetime."""
    return _message().model_copy(update={"posted_at": value})


def _datetime_variants() -> list[datetime]:
    """Instants that agree across zones, naive/aware drift, and DST fold pairs."""
    return [
        datetime(2026, 1, 1, 12, 0),
        datetime(2026, 1, 1, 12, 0, 0, 1),
        datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 14, 0, tzinfo=timezone(timedelta(hours=2))),
        datetime(2026, 1, 1, 17, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        datetime(2026, 1, 1, 7, 0, tzinfo=_NY),
        datetime(2026, 11, 1, 1, 30, tzinfo=_NY, fold=0),
        datetime(2026, 11, 1, 1, 30, tzinfo=_NY, fold=1),
        datetime(2026, 11, 1, 5, 30, tzinfo=UTC),
        datetime(2026, 11, 1, 6, 30, tzinfo=UTC),
    ]


_BUILDERS: list[Callable[[], PublicHistoryEntry]] = [
    _message,
    _reasoning,
    _effect,
    _callback,
    _checkpoint,
    _notice,
]


def _name(builder: Callable[[], PublicHistoryEntry]) -> str:
    return getattr(builder, "__name__", "entry")


def _mutant(value: object) -> object:
    """A different-but-same-shaped value for the per-field sweep."""
    if isinstance(value, Enum):
        return next(member for member in type(value) if member != value)
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return f"{value}-other"
    if isinstance(value, datetime):
        return value + timedelta(seconds=1)
    if isinstance(value, list):
        return [_mutant(value[0]), *value[1:]] if value else ["tail"]
    if isinstance(value, dict):
        return {**value, "mutant": "tail"}
    if isinstance(value, BaseModel):
        return _mutate_nested(value)
    raise AssertionError(f"no mutant for {type(value).__name__}")


def _mutate_nested(model: BaseModel) -> BaseModel:
    for name, value in model.__dict__.items():
        if value is not None and not isinstance(value, BaseModel | list | dict):
            return model.model_copy(update={name: _mutant(value)})
    raise AssertionError(f"no scalar field to mutate in {type(model).__name__}")


def _payload_entry(payload: object) -> PublicCheckpointEntry:
    """Checkpoint entries reach degenerate payloads only without validation."""
    return _checkpoint().model_copy(update={"details": payload})


def _replacement(entry: PublicHistoryEntry) -> PublicHistoryEntry:
    """A second object holding the same content, as a wire re-parse would give."""
    return entry.model_copy(deep=True)


def _variant_corpus() -> list[PublicHistoryEntry]:
    corpus: list[PublicHistoryEntry] = []
    for build in _BUILDERS:
        entry = build()
        corpus.append(entry)
        for name in type(entry).model_fields:
            corpus.append(
                entry.model_copy(update={name: _mutant(getattr(entry, name))})
            )
    corpus.extend([
        _message(blocks=[TextContentBlock(text="only text")]),
        _message(
            blocks=[
                ImageContentBlock(
                    attachment=ImageAttachment(
                        source=FileImageSource(path="/tmp/a.png"),
                        alias="a.png",
                        mime_type="image/png",
                    )
                )
            ]
        ),
        _message(
            blocks=[ResourceContentBlock(resource=UserTextResource(uri="u", text="t"))]
        ),
        _effect().model_copy(
            update={
                "detail": FileEditEffectDetail(
                    tool_name="edit",
                    display=EffectCallDisplay(summary="edit", status_text="edit"),
                )
            }
        ),
        _effect().model_copy(
            update={
                "state": FailedEffectState(
                    error={"message": "boom"},  # type: ignore[arg-type]
                    display=EffectResultDisplay(success=False, message="boom"),
                )
            }
        ),
        _effect().model_copy(
            update={"state": RunningEffectState(output_text="partial")}
        ),
        _payload_entry({"a": 1}),
        _payload_entry({"a": 1.0}),
        _payload_entry({"a": True}),
        _payload_entry({"a": {"b": [1, {"c": 2}]}}),
        _payload_entry({"a": {"b": [1, {"c": 3}]}}),
        _payload_entry({"b": {"c": 2, "d": 4}}),
        _payload_entry({"a": float("inf")}),
        _payload_entry({"a": float("-inf")}),
        # Fingerprint-equal pairs, so the property below is not vacuous.
        _replacement(_message()),
        _replacement(_reasoning()),
        _replacement(_payload_entry({"a": 1})),
    ])
    return corpus


def _assert_no_missed_invalidation(entries: Sequence[PublicHistoryEntry]) -> None:
    fingerprints = [fingerprint_entry(entry) for entry in entries]
    for left, left_print in zip(entries, fingerprints, strict=True):
        for right, right_print in zip(entries, fingerprints, strict=True):
            if left_print == right_print:
                assert left.model_copy(deep=True) == right, (
                    f"missed invalidation: {left.id} vs {right.id}"
                )


def _assert_consistent(window: TranscriptWindow) -> None:
    """Cached entry fingerprints must equal a fresh derivation from the records.

    Validated entries only: a degenerate payload fingerprints as always-changed,
    so its cached value cannot be re-derived by comparison.
    """
    for unit_id in window.unit_ids:
        unit = window.units[unit_id]
        assert len(unit.entry_fingerprints) == len(unit.entries)
        assert len(unit.entry_fingerprints) == len(unit.member_entry_ids)
        for entry, recorded in zip(unit.entries, unit.entry_fingerprints, strict=True):
            assert recorded == fingerprint_entry(entry)


# Entry-level canonicalization and discrimination.


@pytest.mark.parametrize("build", _BUILDERS, ids=_name)
def test_fingerprints_are_fixed_width_and_deep_copy_stable(
    build: Callable[[], PublicHistoryEntry],
) -> None:
    entry = build()
    first = fingerprint_entry(entry)
    assert len(first) == DIGEST_SIZE
    assert fingerprint_entry(entry) == first
    assert fingerprint_entry(entry.model_copy(deep=True)) == first


@pytest.mark.parametrize("build", _BUILDERS, ids=_name)
def test_equal_replacement_objects_share_fingerprints(
    build: Callable[[], PublicHistoryEntry],
) -> None:
    entry = build()
    replacement = _replacement(entry)
    assert replacement is not entry
    assert replacement == entry
    assert fingerprint_entry(replacement) == fingerprint_entry(entry)


@pytest.mark.parametrize("build", _BUILDERS, ids=_name)
def test_per_field_sweep_discriminates_every_top_level_field(
    build: Callable[[], PublicHistoryEntry],
) -> None:
    entry = build()
    swept: set[str] = set()
    for name in type(entry).model_fields:
        mutant = entry.model_copy(update={name: _mutant(getattr(entry, name))})
        assert mutant != entry, name
        assert fingerprint_entry(mutant) != fingerprint_entry(entry), name
        swept.add(name)
    assert swept == set(type(entry).model_fields)


def test_content_block_and_union_variants_are_distinct() -> None:
    variants = [
        _message(blocks=[TextContentBlock(text="hello")]),
        _message(blocks=[TextContentBlock(text="hello!"), TextContentBlock(text="x")]),
        _message(
            blocks=[
                ImageContentBlock(
                    attachment=ImageAttachment(
                        source=InlineImageSource(data="aaaa"),
                        alias="a.png",
                        mime_type="image/png",
                    )
                )
            ]
        ),
        _message(
            blocks=[
                ImageContentBlock(
                    attachment=ImageAttachment(
                        source=FileImageSource(path="/tmp/a.png"),
                        alias="a.png",
                        mime_type="image/png",
                    )
                )
            ]
        ),
        _message(
            blocks=[ResourceContentBlock(resource=UserTextResource(uri="u", text="t"))]
        ),
    ]
    assert len({fingerprint_entry(entry) for entry in variants}) == len(variants)


def test_cross_type_swap_at_one_id_is_discriminated() -> None:
    entries = [build().model_copy(update={"id": "shared"}) for build in _BUILDERS]
    for left in entries:
        for right in entries:
            if left is not right:
                assert left != right
    fingerprints = [fingerprint_entry(entry) for entry in entries]
    assert len(fingerprints) == len(set(fingerprints))
    _assert_no_missed_invalidation(entries)


def test_nested_in_place_mutation_is_detected() -> None:
    original = _payload_entry({"a": {"b": [1, {"c": 2}]}})
    mutated = original.model_copy(deep=True)
    payload = cast("Any", mutated.details)
    payload["a"]["b"][1]["c"] = 3
    assert mutated != original
    assert fingerprint_entry(mutated) != fingerprint_entry(original)


def test_dict_payload_key_order_is_canonical() -> None:
    left = _payload_entry({"a": 1, "b": {"y": 2, "x": 3}, "c": [{"n": 1, "m": 2}]})
    right = _payload_entry({"c": [{"m": 2, "n": 1}], "b": {"x": 3, "y": 2}, "a": 1})
    assert left == right
    assert fingerprint_entry(left) == fingerprint_entry(right)


def test_timezone_variant_datetimes_are_fingerprint_equal() -> None:
    utc = _posted(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    offset = _posted(datetime(2026, 1, 1, 14, 0, tzinfo=timezone(timedelta(hours=2))))
    zone = _posted(datetime(2026, 1, 1, 7, 0, tzinfo=_NY))
    assert utc == offset == zone
    assert (
        fingerprint_entry(utc) == fingerprint_entry(offset) == fingerprint_entry(zone)
    )


def test_naive_and_aware_datetimes_stay_canonically_distinct() -> None:
    naive = _message().model_copy(update={"posted_at": datetime(2026, 1, 1, 12, 0)})
    aware = _message().model_copy(
        update={"posted_at": datetime(2026, 1, 1, 12, 0, tzinfo=UTC)}
    )
    assert naive != aware
    assert fingerprint_entry(naive) != fingerprint_entry(aware)


def test_fold_variant_timestamps_never_collapse_with_their_instant_twin() -> None:
    early_at = datetime(2026, 11, 1, 1, 30, tzinfo=_NY, fold=0)
    late_at = datetime(2026, 11, 1, 1, 30, tzinfo=_NY, fold=1)
    early, late = _posted(early_at), _posted(late_at)
    early_utc = _posted(datetime(2026, 11, 1, 5, 30, tzinfo=UTC))
    late_utc = _posted(datetime(2026, 11, 1, 6, 30, tzinfo=UTC))

    # 2026-11-01 01:30 happens twice in America/New_York: fold picks EDT or EST.
    assert early_at.utcoffset() != late_at.utcoffset()

    # Each fold variant holds an instant its UTC twin names, but CPython calls
    # the records unequal, so a shared fingerprint would keep the stale record.
    assert late != late_utc
    assert early != early_utc
    assert late != early_utc
    assert fingerprint_entry(late) != fingerprint_entry(late_utc)
    assert fingerprint_entry(early) != fingerprint_entry(early_utc)

    # One tzinfo object makes the two fold variants pydantic-equal (the wall
    # clocks match field-by-field) even though they name different instants:
    # separating them is the accepted spurious-invalidation direction.
    assert early == late
    assert fingerprint_entry(early) != fingerprint_entry(late)


def test_fold_ambiguous_payloads_are_always_changed() -> None:
    # Two records whose every payload value agrees — wall clock, offset, zone
    # name, fold — yet each carries its own tzinfo object, and an ambiguous wall
    # time never equals another tzinfo's. Only an always-changed fingerprint can
    # keep equality implied by fingerprint equality here.
    first_at = datetime(2026, 11, 1, 1, 30, tzinfo=_FoldAwareZone(), fold=0)
    second_at = datetime(2026, 11, 1, 1, 30, tzinfo=_FoldAwareZone(), fold=0)
    assert first_at.isoformat() == second_at.isoformat()
    first, second = _posted(first_at), _posted(second_at)
    assert first != second
    assert fingerprint_entry(first) != fingerprint_entry(second)
    assert fingerprint_entry(first) != fingerprint_entry(first)


def test_datetime_variant_matrix_keeps_fingerprint_equality_sound() -> None:
    entries = [_posted(value) for value in _datetime_variants()]
    _assert_no_missed_invalidation(entries)
    fingerprints = [fingerprint_entry(entry) for entry in entries]
    shared = sum(
        1
        for index, left in enumerate(fingerprints)
        for right in fingerprints[index + 1 :]
        if left == right
    )
    assert shared >= 2, "equal instants across zones must still share fingerprints"


def test_nan_payloads_are_always_changed() -> None:
    first = _payload_entry({"a": float("nan")})
    second = _payload_entry({"a": float("nan")})
    # Distinct NaN objects are pydantic-unequal, so no value digest may match.
    assert first != second
    assert fingerprint_entry(first) != fingerprint_entry(second)
    assert fingerprint_entry(first) != fingerprint_entry(first)
    assert fingerprint_entry(first) != fingerprint_entry(_payload_entry({"a": 1}))


def test_infinite_payloads_are_stable_and_distinct() -> None:
    positive = _payload_entry({"a": float("inf")})
    repeated = _payload_entry({"a": float("inf")})
    negative = _payload_entry({"a": float("-inf")})
    assert positive == repeated
    assert fingerprint_entry(positive) == fingerprint_entry(repeated)
    assert fingerprint_entry(positive) != fingerprint_entry(negative)
    assert fingerprint_entry(positive) != fingerprint_entry(
        _payload_entry({"a": 1e308})
    )


def test_non_string_mapping_keys_are_always_changed() -> None:
    integer_keys = _payload_entry({1: "a"})
    string_keys = _payload_entry({"1": "a"})
    mixed = _payload_entry({1: "a", "b": 2})
    assert integer_keys != string_keys
    assert fingerprint_entry(integer_keys) != fingerprint_entry(string_keys)
    assert fingerprint_entry(mixed) != fingerprint_entry(mixed)


def test_tuple_payload_values_are_always_changed() -> None:
    as_tuple = _payload_entry({"a": (1, 2)})
    as_list = _payload_entry({"a": [1, 2]})
    assert as_tuple != as_list
    assert fingerprint_entry(as_tuple) != fingerprint_entry(as_list)
    assert fingerprint_entry(as_tuple) != fingerprint_entry(as_tuple)


def test_lone_surrogate_payloads_are_always_changed() -> None:
    # A JSON payload can decode to a lone surrogate ("\ud800"), and such text
    # has no UTF-8 encoding. The canonical form cannot be built, so admission
    # must fall back to always-changed instead of raising.
    surrogate = _message(blocks=[TextContentBlock(text="a\ud800b")])
    clean = _message(blocks=[TextContentBlock(text="ab")])
    assert fingerprint_entry(surrogate) != fingerprint_entry(surrogate)
    assert fingerprint_entry(surrogate) != fingerprint_entry(clean)

    window = TranscriptWindow()
    window.admit([surrogate], start_index=0)
    unit = window.units["message-1"]
    first = unit.entry_fingerprints[0]
    window.update_entry(_message(blocks=[TextContentBlock(text="a\ud800b")]))
    # Always-changed, so an update is never mistaken for an unchanged record.
    assert unit.entry_fingerprints[0] != first
    assert unit.geometry_version > 0


def test_extreme_datetimes_fall_back_to_always_changed() -> None:
    # Both extremes are valid datetimes, but normalizing them to UTC leaves
    # the representable range and raises OverflowError inside _encode; the
    # digest must fall back to always-changed instead of failing admission.
    low = _posted(datetime(1, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=1))))
    high = _posted(
        datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone(timedelta(hours=-1)))
    )
    for entry in (low, high):
        assert fingerprint_entry(entry) != fingerprint_entry(entry)
    assert fingerprint_entry(low) != fingerprint_entry(high)
    assert fingerprint_entry(low) != fingerprint_entry(
        _posted(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    )

    window = TranscriptWindow()
    window.admit(
        [
            low.model_copy(update={"id": "message-low"}),
            high.model_copy(update={"id": "message-high"}),
        ],
        start_index=0,
    )
    assert len(window.unit_ids) == 2


def test_numeric_type_drift_is_the_accepted_spurious_direction() -> None:
    integer = _payload_entry({"a": 1})
    floating = _payload_entry({"a": 1.0})
    boolean = _payload_entry({"a": True})
    # Recorded drift: pydantic calls these equal, the type-tagged encoding does
    # not. That is spurious invalidation only, so the pin still holds.
    assert integer == floating == boolean
    assert fingerprint_entry(integer) != fingerprint_entry(floating)
    assert fingerprint_entry(integer) != fingerprint_entry(boolean)
    _assert_no_missed_invalidation([integer, floating, boolean])


def test_fingerprint_equality_implies_deep_copy_equality() -> None:
    corpus = _variant_corpus()
    assert len(corpus) > len(_BUILDERS)
    _assert_no_missed_invalidation(corpus)
    fingerprints = [fingerprint_entry(entry) for entry in corpus]
    equal_pairs = sum(
        1
        for index, left in enumerate(fingerprints)
        for right in fingerprints[index + 1 :]
        if left == right
    )
    assert equal_pairs >= 3, "the property is vacuous without fingerprint-equal pairs"


def test_entry_models_forbid_extra_fields() -> None:
    payload = {
        **_base("message-9"),
        "role": "user",
        "content": [{"type": "text", "text": "x"}],
        "unexpected": 1,
    }
    with pytest.raises(ValidationError):
        PublicMessageEntry.model_validate(payload)


@pytest.mark.parametrize("build", _BUILDERS, ids=_name)
def test_entry_models_carry_no_state_outside_the_field_dict(
    build: Callable[[], PublicHistoryEntry],
) -> None:
    # Fingerprints canonicalize ``__dict__``; pydantic equality additionally
    # compares extras and private attributes, so both must be empty here.
    entry = build()
    assert entry.__pydantic_extra__ is None
    assert not entry.__pydantic_private__


# Registry-level entry-fingerprint maintenance.


def test_admission_records_entry_fingerprints() -> None:
    window = TranscriptWindow()
    window.admit([_message(0), _message(1)], start_index=0, geometry_width=80)
    _assert_consistent(window)
    unit = window.units["message-0"]
    assert unit.entry_fingerprints == [fingerprint_entry(_message(0))]


def test_equal_re_admission_skips_invalidation() -> None:
    window = TranscriptWindow()
    window.admit([_message(0)], start_index=0)
    unit = window.units["message-0"]
    retained = unit.entries[0].model_copy(deep=True)
    prints = list(unit.entry_fingerprints)
    geometry = unit.geometry_version
    window.admit([_message(0)], start_index=0)
    assert unit.entry_fingerprints == prints
    assert unit.entries[0] == retained
    assert unit.geometry_version == geometry
    _assert_consistent(window)


def test_changed_re_admission_invalidates_the_replaced_member() -> None:
    window = TranscriptWindow()
    window.admit([_message(0)], start_index=0)
    unit = window.units["message-0"]
    geometry = unit.geometry_version
    replacement = _message(0, blocks=[TextContentBlock(text="steered")])
    window.admit([replacement], start_index=0)
    assert unit.entry_fingerprints == [fingerprint_entry(replacement)]
    assert unit.entries[0] == replacement
    assert unit.geometry_version != geometry
    _assert_consistent(window)


def test_update_entry_tracks_retained_data_in_entry_fingerprints() -> None:
    window = TranscriptWindow()
    window.admit([_message(0)], start_index=0)
    unit = window.units["message-0"]
    window.update_entry(_message(0))
    assert unit.entry_fingerprints == [fingerprint_entry(_message(0))]
    _assert_consistent(window)

    updated = _message(0, blocks=[TextContentBlock(text="updated")])
    window.update_entry(updated)
    assert unit.entry_fingerprints == [fingerprint_entry(updated)]
    assert unit.entries[0] == updated
    _assert_consistent(window)

    checkpoint = _checkpoint(0).model_copy(update={"id": "message-0"})
    window.update_entry(checkpoint)
    assert unit.entry_fingerprints == [fingerprint_entry(checkpoint)]
    _assert_consistent(window)


def test_equal_update_leaves_the_retained_record_in_place() -> None:
    window = TranscriptWindow()
    window.admit([_message(0)], start_index=0)
    unit = window.units["message-0"]
    retained = unit.entries[0]
    geometry = unit.geometry_version

    equal = _message(0)
    assert equal is not retained
    assert equal == retained
    window.update_entry(equal)

    assert unit.entries[0] is retained
    assert unit.geometry_version == geometry
    _assert_consistent(window)


def test_numeric_type_drift_refreshes_instead_of_skipping() -> None:
    window = TranscriptWindow()
    window.admit([_effect(0, output={"a": 1})], start_index=0)
    unit = window.units["effect-0"]
    retained = unit.entries[0]
    geometry = unit.geometry_version

    drifted = _effect(0, output={"a": 1.0})
    assert drifted == retained
    window.update_entry(drifted)

    # The accepted spurious-invalidation direction: the records are pydantic-equal
    # but fingerprint-unequal, so the cutover refreshes rather than skipping.
    assert unit.entries[0] is drifted
    assert unit.geometry_version != geometry
    _assert_consistent(window)


def test_fold_variant_timestamp_invalidates_the_retained_entry() -> None:
    window = TranscriptWindow()
    window.admit([_message(0)], start_index=0)
    unit = window.units["message-0"]
    retained = unit.entries[0]
    prints = list(unit.entry_fingerprints)
    geometry = unit.geometry_version

    folded = retained.model_copy(
        update={"posted_at": datetime(2026, 11, 1, 1, 30, tzinfo=_NY, fold=1)}
    )
    assert folded != retained
    window.update_entry(folded)

    # Keeping the retained record here would render the pre-fold timestamp.
    # The payload is fold-ambiguous, so it fingerprints as always-changed and
    # the cached value is not re-derivable — hence no consistency sweep.
    assert unit.entries[0] is folded
    assert unit.entry_fingerprints != prints
    assert unit.geometry_version != geometry


def test_identity_adoption_rewrites_membership_and_entry_fingerprints() -> None:
    window = TranscriptWindow()
    pending = _message(0)
    pending.id = ""
    window.admit([pending], start_index=0)
    transient_id = window.unit_ids[0]
    assert transient_id.startswith("transient-")
    prints = list(window.units[transient_id].entry_fingerprints)

    pending.id = "message-adopted"
    window.admit([pending], start_index=0)

    assert window.unit_ids == ["message-adopted"]
    assert window.units["message-adopted"].member_entry_ids == ["message-adopted"]
    assert window.units["message-adopted"].entry_fingerprints != prints
    _assert_consistent(window)


def test_retry_membership_extends_the_last_placement() -> None:
    window = TranscriptWindow()
    window.admit([_message(0)], start_index=0)
    unit = window.units["message-0"]
    root = Widget()
    unit.mounted_roots = [root]
    prints = list(unit.entry_fingerprints)

    assert window.admit_reused_root(_message(1), root, index=1)
    assert unit.member_entry_ids == ["message-0", "message-1"]
    assert unit.entry_fingerprints != prints
    _assert_consistent(window)


def test_prefix_admission_rebases_start_index_only() -> None:
    window = TranscriptWindow()
    window.admit([_message(2), _message(3)], start_index=0)
    unit = window.units["message-2"]
    prints = list(unit.entry_fingerprints)

    window.admit([_message(0), _message(1)], start_index=0)

    assert unit.start_index == 2
    assert window.admitted_start_index == 0
    assert window.admitted_end_index == 4
    assert unit.entry_fingerprints == prints
    _assert_consistent(window)


def test_expansion_invalidates_geometry_without_touching_entry_fingerprints() -> None:
    state = EntryExpansionState()
    window = TranscriptWindow(entry_expansion_state=state)
    window.admit([_message(0)], start_index=0)
    unit = window.units["message-0"]
    prints = list(unit.entry_fingerprints)
    geometry = unit.geometry_version

    state.set_collapsed("message-0", False)
    assert window.sync_expansion() == {"message-0"}

    assert unit.entry_fingerprints == prints
    assert unit.geometry_version != geometry
    _assert_consistent(window)


def test_removal_and_reset_clear_the_registry() -> None:
    window = TranscriptWindow()
    window.admit([_message(0), _message(1)], start_index=0)
    _assert_consistent(window)

    window.remove_unit("message-0")
    assert window.unit_ids == ["message-1"]
    _assert_consistent(window)

    window.reset()
    assert window.unit_ids == []
    assert not window.units


def test_mixed_sequence_keeps_entry_fingerprints_consistent() -> None:
    window = TranscriptWindow()
    window.admit([_effect(0), _effect(1)], start_index=0)
    _assert_consistent(window)
    window.admit([_message(2)], start_index=2)
    _assert_consistent(window)
    window.update_entry(_effect(1, output={"rows": [9]}))
    _assert_consistent(window)
    window.admit([_effect(-2), _effect(-1)], start_index=-2)
    _assert_consistent(window)
    window.remove_unit(window.unit_ids[0])
    _assert_consistent(window)
    window.update_entry(_effect(1))
    _assert_consistent(window)


def test_nan_entry_always_reinvalidates_on_replacement() -> None:
    nan_entry = _effect(0, output={"a": float("nan")})
    window = TranscriptWindow()
    window.admit([nan_entry], start_index=0)
    unit = window.units["effect-0"]
    first = unit.entry_fingerprints[0]

    window.update_entry(_effect(0, output={"a": float("nan")}))
    assert unit.entry_fingerprints[0] != first

    generation = unit.geometry_version
    window.update_entry(_effect(0, output={"a": float("nan")}))
    # A NaN payload is always-changed, so a repeated update whose records are
    # unequal by value can never be mistaken for an unchanged one.
    assert unit.entry_fingerprints[0] != first
    assert unit.geometry_version != generation


def test_infinite_entry_keeps_a_stable_entry_fingerprint() -> None:
    window = TranscriptWindow()
    window.admit([_effect(0, output={"a": float("inf")})], start_index=0)
    unit = window.units["effect-0"]
    prints = list(unit.entry_fingerprints)

    window.update_entry(_effect(0, output={"a": float("inf")}))
    assert unit.entry_fingerprints == prints
    _assert_consistent(window)


class _App(App[None]):
    CSS = "#stream { width: 60; height: 10; layout: stream; overflow-y: auto; }"

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="stream")


class _Measured(Widget):
    def __init__(self, height: int) -> None:
        super().__init__()
        self.height = height

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return self.height


@pytest.mark.asyncio
async def test_eviction_and_restore_interleave_with_update_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = TranscriptWindow()
    window.admit([_message(0), _message(1)], start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = [_Measured(4), _Measured(4)]
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        _assert_consistent(window)

        unit = window.units["message-0"]
        prints = list(unit.entry_fingerprints)
        geometry = unit.geometry_version
        await window.evict_unit("message-0")
        await pilot.pause()
        assert unit.placeholders
        assert unit.entry_fingerprints == prints
        _assert_consistent(window)

        window.update_entry(
            _message(0, blocks=[TextContentBlock(text="while evicted")])
        )
        assert unit.entry_fingerprints != prints
        assert unit.geometry_version != geometry
        _assert_consistent(window)

        restored = _Measured(4)
        monkeypatch.setattr(window, "build_unit", lambda *_: [restored])
        assert await window.restore_unit(
            "message-0", stream, WeakKeyDictionary(), follow_bottom=True
        )
        await pilot.pause()
        assert unit.mounted_roots == [restored]
        assert not unit.placeholders
        _assert_consistent(window)

        prints = list(unit.entry_fingerprints)
        window.update_entry(
            _message(0, blocks=[TextContentBlock(text="after restore")])
        )
        assert unit.entry_fingerprints != prints
        _assert_consistent(window)
