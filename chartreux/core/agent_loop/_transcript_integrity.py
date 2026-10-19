"""Shared transcript message-order integrity validator and repair (PI1/T).

Production incident PI1: mid-turn steering appended and persisted a user
message before the in-flight tool call's cancellation settled, and the
backfill repair (``repair_missing_responses``) could not see the genuine
result behind the steered user message, so it inserted a DUPLICATE synthetic
``TOOL_NO_RESPONSE`` placeholder. The resulting ``[assistant, placeholder,
user, genuine result]`` shape is a permanent provider 400
(``invalid_request_message_order``) that no layer repaired — not at request
build, not at session load, and not across restarts.

This module is the single shared validator/repair for that failure class:

- duplicate tool-result detection (scoped per originating assistant call),
- provider-safe order admission (verified AFTER the request-build reorder),
- deterministic repair of the unambiguous PI1 pattern (drop the synthetic
  placeholder shadowed by a genuine structured result; never delete genuine
  results), and
- the typed LOCAL transcript-integrity diagnostic
  (:class:`TranscriptIntegrityError`).

Three validation contracts are defined:

``LIVE-PREFIX`` (``validate_live_prefix``)
    The live tail of the transcript — the trailing assistant tool-call block,
    staged batch results, the non-injected steered user messages walked over
    on the way to it, and the nested injected assistant/tool pairs (skill and
    mentioned-file expansion, question callbacks, agent-lifecycle injections)
    — is LEGITIMATE even while unresolved. Unresolved calls, staged batch
    results, and nested injected pairs are never violations. A settled
    assistant block joins the tail only when it carries the injector's
    provenance marker (round-5 Y): an unmarked settled provider response
    ends the tail whatever the format of its call ids.

``SETTLED-HISTORY`` (``validate_settled_history``)
    Every assistant tool call that is not part of the live tail must be
    answered by exactly one tool result (after the deterministic dedupe), and
    no tool result may be unclaimed by any assistant call.

``PROVIDER-READY`` (``validate_provider_ready``)
    The order request build produces (dedupe + ``reorder_for_tool_adjacency``)
    must be provider-safe: every tool result must sit in the open result run
    of its owning assistant call, so no ``[user, tool]`` adjacency survives to
    the provider, and no assistant tool call is left unanswered in the
    projected order (round-1 S1(c); the live tail stays exempt — a call
    whose result is still in flight is the designed backfill's job). An
    unrepairable order is reported, never guessed.

The diagnostic is a LOCAL typed failure: its own category
(``transcript_integrity``), never transient, never retried, never scheduled —
it deliberately declares no ``failure_category``, so the failure classifier
keeps it UNKNOWN (report-and-park) through exception wrapping and both the
root and child projections. It grants no recovery authority: integrity repair
never proves external-effect settlement and never replenishes budgets,
clears cooldowns, restores callbacks/grants, or resurrects suspended
children.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from chartreux.core.compaction.context import (
    is_synthetic_tool_placeholder,
    reorder_for_tool_adjacency,
    shadowed_placeholder_positions,
)
from chartreux.core.llm_models import CONTEXT_INJECTION_TOOL_PAIR_V1, LLMMessage, Role

__all__ = [
    "TRANSCRIPT_INTEGRITY_CATEGORY",
    "TRANSCRIPT_INTEGRITY_METADATA_KEY",
    "TranscriptIntegrityError",
    "TranscriptIntegrityReport",
    "TranscriptIntegrityViolation",
    "ViolationKind",
    "check_tool_message_admission",
    "dedupe_shadowed_placeholders",
    "find_duplicate_tool_results",
    "find_persistence_violations",
    "has_structured_tool_result",
    "is_synthetic_tool_placeholder",
    "pop_report_from_metadata",
    "repair_transcript",
    "report_from_metadata",
    "report_to_metadata",
    "validate_live_prefix",
    "validate_provider_ready",
    "validate_settled_history",
    "validate_transcript",
]

#: The diagnostic's own internal category (never a FailureCategory member:
#: classification stays UNKNOWN — never transient, never retried).
TRANSCRIPT_INTEGRITY_CATEGORY = "transcript_integrity"

#: Reserved metadata key under which ``SessionLoader.load_session`` rides the
#: in-memory repair report to write-owning install paths. ``SessionMetadata``
#: ignores unknown keys, so read-only consumers are unaffected.
TRANSCRIPT_INTEGRITY_METADATA_KEY = "transcript_integrity_repair"

#: A call id only becomes a duplicate candidate once at least two results
#: exist for it.
_MIN_RESULTS_FOR_DUPLICATE = 2

ViolationKind = Literal[
    "duplicate_tool_result",
    "unclaimed_tool_result",
    "unresolved_tool_call",
    "misordered_tool_result",
    "transcript_repair_failed",
]


@dataclass(frozen=True, slots=True)
class TranscriptIntegrityViolation:
    """One concrete transcript-integrity defect."""

    kind: ViolationKind
    tool_call_id: str | None
    detail: str

    def __str__(self) -> str:
        location = f" (tool_call_id={self.tool_call_id})" if self.tool_call_id else ""
        return f"{self.kind}{location}: {self.detail}"


class TranscriptIntegrityError(Exception):
    """Typed LOCAL transcript-integrity diagnostic (PI1/T).

    Raised BEFORE any provider call (request-build admission), at the resume
    install / turn admission (fail closed), and at the persistence boundary
    (never silently persisted). Own category, never transient, never retried,
    never scheduled; projected through the existing ``public_error`` mapping
    as an internal error carrying ``category=transcript_integrity``.
    """

    category: ClassVar[str] = TRANSCRIPT_INTEGRITY_CATEGORY

    def __init__(
        self, violations: Sequence[TranscriptIntegrityViolation], *, context: str = ""
    ) -> None:
        self.violations: tuple[TranscriptIntegrityViolation, ...] = tuple(violations)
        rendered = "; ".join(str(violation) for violation in self.violations)
        message = (
            "Session transcript failed message-order integrity checks: "
            f"{rendered}. The transcript was not modified. Reload (resume) "
            "the session to apply the automatic transcript repair, or rewind "
            "to before the damage; retrying the same turn would fail the "
            "same way."
        )
        if context:
            message = f"{context}: {message}"
        super().__init__(message)

    @classmethod
    def migration_failed(
        cls, path: Path | str, exc: BaseException
    ) -> TranscriptIntegrityError:
        """The durable transcript-repair migration failed: fail closed.

        The original transcript is preserved as an artifact (or the artifact
        write itself failed); the live path was not silently replaced with a
        partial migration. Raised only at the write-owning resume install.
        """
        return cls(
            [
                TranscriptIntegrityViolation(
                    kind="transcript_repair_failed",
                    tool_call_id=None,
                    detail=(
                        f"the repaired transcript could not be durably installed "
                        f"at {path}: {exc}"
                    ),
                )
            ],
            context="transcript repair migration",
        )


@dataclass(frozen=True, slots=True)
class TranscriptIntegrityReport:
    """The outcome of one validate + deterministic-repair pass."""

    #: True when the unambiguous PI1 pattern was repaired in memory.
    repaired: bool
    #: Call ids whose shadowed synthetic placeholders were dropped.
    dropped_placeholder_call_ids: tuple[str, ...]
    #: Unrepairable defects (ambiguous cases fail closed, never guessed).
    violations: tuple[TranscriptIntegrityViolation, ...]

    @property
    def clean(self) -> bool:
        return not self.violations


def has_structured_tool_result(message: LLMMessage) -> bool:
    """Whether a tool message carries the structured ``tool_result`` payload."""
    return message.role == Role.tool and message.tool_result is not None


def _call_ids(message: LLMMessage) -> set[str]:
    return {tool_call.id for tool_call in message.tool_calls or [] if tool_call.id}


def _result_groups(messages: Sequence[LLMMessage]) -> dict[tuple[int, str], list[int]]:
    """Group tool-result indexes by (owning assistant index, tool_call_id).

    ONE O(n) pass. Call identity is scoped PER ORIGINATING ASSISTANT CALL
    (PI1/T): tool-call ids are reused across turns, so two results sharing a
    ``tool_call_id`` under DIFFERENT assistant calls are not duplicates. The
    owner of a result is the last assistant message carrying the call id
    BEFORE the result; a result no earlier assistant call carries is
    unclaimed (result-before-call included).
    """
    last_owner: dict[str, int] = {}
    groups: dict[tuple[int, str], list[int]] = {}
    for index, message in enumerate(messages):
        if message.role == Role.assistant:
            for call_id in _call_ids(message):
                last_owner[call_id] = index
        elif message.role == Role.tool and message.tool_call_id:
            owner = last_owner.get(message.tool_call_id)
            if owner is not None:
                groups.setdefault((owner, message.tool_call_id), []).append(index)
    return groups


def _owner_before(
    messages: Sequence[LLMMessage], index: int, call_id: str
) -> int | None:
    """The originating assistant call for a result entering at ``index``.

    The last assistant message carrying the call id before ``index`` (the
    forward-scan owner at that position).
    """
    owner: int | None = None
    for position in range(index):
        message = messages[position]
        if message.role == Role.assistant and call_id in _call_ids(message):
            owner = position
    return owner


def find_duplicate_tool_results(
    messages: Sequence[LLMMessage],
) -> list[TranscriptIntegrityViolation]:
    """Detect duplicate tool results, scoped per originating assistant call.

    One assistant call answered by more than one tool result is a duplicate
    (call-id reuse across turns is NOT: those results have different owners).
    It is REPAIRABLE (not a violation) exactly when one result is genuine and
    every other is a synthetic ``TOOL_NO_RESPONSE`` placeholder — the
    unambiguous PI1 pattern, repaired by :func:`dedupe_shadowed_placeholders`.
    Two genuine results, or placeholders without a genuine counterpart, are
    ambiguous: reported, never guessed.
    """
    violations: list[TranscriptIntegrityViolation] = []
    for (_, call_id), indexes in _result_groups(messages).items():
        if len(indexes) < _MIN_RESULTS_FOR_DUPLICATE:
            continue
        genuine = [
            position
            for position in indexes
            if not is_synthetic_tool_placeholder(messages[position])
        ]
        if len(genuine) == 1:
            continue  # the unambiguous pattern: repaired by the dedupe
        violations.append(
            TranscriptIntegrityViolation(
                kind="duplicate_tool_result",
                tool_call_id=call_id,
                detail=(
                    f"{len(indexes)} tool results for one call "
                    f"({len(genuine)} genuine); the transcript is ambiguous"
                ),
            )
        )
    return violations


def _unclaimed_tool_results(
    messages: Sequence[LLMMessage],
) -> list[TranscriptIntegrityViolation]:
    """Tool results no assistant tool call claims (orphans, incl. result-before-call).

    A tool message without a ``tool_call_id`` is an unscoped boundary marker,
    not an orphan: it cannot be scoped to an originating assistant call, and
    the provider mappers already convert unscoped results defensively — it is
    outside the PI1 duplicate/order scope. Claiming is scoped per originating
    assistant call: the owner is the last assistant carrying the call id
    BEFORE the result.
    """
    violations: list[TranscriptIntegrityViolation] = []
    last_owner: dict[str, int] = {}
    for index, message in enumerate(messages):
        if message.role == Role.assistant:
            for call_id in _call_ids(message):
                last_owner[call_id] = index
        elif message.role == Role.tool:
            call_id = message.tool_call_id
            if call_id is None:
                continue  # unscoped boundary marker: outside the PI1 scope
            if call_id not in last_owner:
                violations.append(
                    TranscriptIntegrityViolation(
                        kind="unclaimed_tool_result",
                        tool_call_id=call_id,
                        detail="tool result matches no assistant tool call",
                    )
                )
    return violations


def dedupe_shadowed_placeholders(messages: Sequence[LLMMessage]) -> list[LLMMessage]:
    """Drop synthetic placeholders shadowed by a genuine result (single pass).

    Pure and order-stable: genuine structured tool results are never deleted;
    synthetic placeholders stay identifiable as synthetic (content + missing
    structured result). The predicate never keys on the cancellation tag, so
    genuine ``TOOL_INTERRUPTED``/``TOOL_SKIPPED`` settlements are never
    dropped. Shadowing is scoped per originating assistant call, so a
    placeholder for a reused call id under a different assistant call is
    kept. The owner-scoped predicate is SHARED with the request-build
    reorder (``shadowed_placeholder_positions`` in ``compaction.context``):
    the validated projection and the emitted projection cannot diverge.
    """
    drop = shadowed_placeholder_positions(messages)
    if not drop:
        return list(messages)
    return [
        message for position, message in enumerate(messages) if position not in drop
    ]


def _dedupe_violations(
    violations: Sequence[TranscriptIntegrityViolation],
) -> list[TranscriptIntegrityViolation]:
    seen: set[tuple[str, str | None, str]] = set()
    unique: list[TranscriptIntegrityViolation] = []
    for violation in violations:
        key = (violation.kind, violation.tool_call_id, violation.detail)
        if key in seen:
            continue
        seen.add(key)
        unique.append(violation)
    return unique


def _genuine_result_call_ids(messages: Sequence[LLMMessage]) -> frozenset[str]:
    """Call ids with a recorded GENUINE (non-placeholder) tool result.

    A synthetic ``TOOL_NO_RESPONSE`` placeholder is the backfill's marker for
    a dangling call, not an answer; only a genuine result means the transcript
    recorded an answer for the id.
    """
    return frozenset(
        message.tool_call_id
        for message in messages
        if message.role == Role.tool
        and message.tool_call_id
        and not is_synthetic_tool_placeholder(message)
    )


def _genuine_result_owners(messages: Sequence[LLMMessage]) -> dict[str, set[int]]:
    """Call ids mapped to the OWNING assistant blocks of their genuine results.

    The owner of a result is the last assistant message carrying the call id
    BEFORE the result (the forward-scan owner — the same
    per-originating-assistant-call scoping the dedupe and the duplicate
    detection use). Keyed by the owning message's IDENTITY so the projected
    pass — which permutes the same message objects — can match owners across
    the projection.
    """
    last_owner: dict[str, int] = {}
    owners: dict[str, set[int]] = {}
    for message in messages:
        if message.role == Role.assistant:
            for call_id in _call_ids(message):
                last_owner[call_id] = id(message)
        elif (
            message.role == Role.tool
            and message.tool_call_id
            and not is_synthetic_tool_placeholder(message)
        ):
            owner = last_owner.get(message.tool_call_id)
            if owner is not None:
                owners.setdefault(message.tool_call_id, set()).add(owner)
    return owners


def _nested_in_assistant_gap(  # noqa: PLR0911 — one guard per provenance/structure step
    messages: Sequence[LLMMessage], index: int
) -> bool:
    """Whether the settled assistant block at ``index`` is nested injection
    material (round-3 W2, round-4 X1/X4, round-5 Y).

    Production skill/mention expansion appends its settled assistant/tool
    pair with NO ``injected`` flag (``_context_injection.py``), so the
    flag-based walk cannot recognize it. Structurally, a settled assistant
    tool-call block is nested in the gap of ANOTHER assistant tool-call
    block — the pair was injected while the outer call was in flight. The
    bounded backward scan skips both tool results and non-injected user
    messages (mid-turn steering), so it reaches the outer block across the
    steered user message of a COMPOSED steer (the steered user message
    followed by the implicit skill/mention pair). A settled block answering
    a user message is a completed turn's response and ends the live tail,
    so the scan recognizes the outer block only through evidence the
    injector — and only the injector — can produce.

    Round-5 Y: nesting is keyed on PROVENANCE, never on call-id format. The
    candidate must carry the injector's private origin marker
    (``_context_injection_origin``: only ``ContextInjector`` originates it
    in production, and only the trusted session loader restores it from
    Chartreux's owned session transcript), and its call ids must be DISJOINT
    from the preceding assistant block's (round-4 X4). Provenance is
    necessary, NOT sufficient: an unmarked settled provider response ends
    the region regardless of its id's format (a uuid4-shaped provider id is
    no authority — the round-5 fail-open), and a marked pair colliding with
    the outer block's ids ends the region too — the adjacent-duplicate-id
    shape keeps an empty region by the walk itself, not by the exemption's
    owner check.

    Assumption (bounded scan): the scan skips only tool messages and
    non-injected user messages, so an ``injected=True`` message inside the
    gap BREAKS the scan — the walk stops at it before reaching the outer
    block. That is not production-reachable: a reviewer's trace of all
    ``injected=True`` append sites (the compaction boundary, the
    question-callback and lifecycle surfaces) shows they append at the
    transcript tail, never between an in-flight assistant block and its
    results, so no injected message can sit inside an assistant call's gap.
    """
    candidate = messages[index]
    if candidate.role != Role.assistant or not candidate.tool_calls:
        return False
    if type(candidate._context_injection_origin) is not str:
        return False
    if candidate._context_injection_origin != CONTEXT_INJECTION_TOOL_PAIR_V1:
        return False
    pair_ids = _call_ids(candidate)
    if not pair_ids:
        return False
    position = index - 1
    while position >= 0:
        message = messages[position]
        if message.role == Role.tool:
            position -= 1
            continue
        if message.role == Role.user and not message.injected:
            position -= 1
            continue
        break
    if position < 0:
        return False
    preceding = messages[position]
    if preceding.role != Role.assistant or not preceding.tool_calls:
        return False
    if pair_ids & _call_ids(preceding):
        return False
    return True


def _live_tail_region_identities(
    messages: Sequence[LLMMessage], *, staged_call_ids: frozenset[str] = frozenset()
) -> frozenset[int]:
    """The assistant tool-call blocks of the live tail region (identities).

    The live tail is the trailing region of the transcript: the trailing
    assistant tool-call block whose calls are still unresolved (their results
    are in flight), the staged batch results behind it, and the nested
    injected assistant/tool pairs and steered user messages in its gap. The
    walk runs backward from the end over that live-tail material and anchors
    at the first non-injected assistant tool-call block with an unresolved
    call; any other non-injected message ends the tail EXCEPT the structural
    live-tail material — a non-injected user message (mid-turn steering) and
    a settled non-injected assistant tool-call block nested in another call's
    gap (see :func:`_nested_in_assistant_gap`) are walked over. A turn
    boundary still ends the tail: a turn-boundary user message is always
    followed by its turn's assistant block, at which the walk has already
    ended. Returns the identities of the assistant tool-call blocks in the
    region (empty when the transcript has no live tail).

    Round-3 W2: the boundary is derived STRUCTURALLY from evidence
    production actually produces, not only from message flags. Production
    steering injects user messages via ``as_message=True`` with
    ``injected=False`` and the settled skill/mention assistant messages also
    lack the flags, so the walk additionally:

    - walks over a non-injected USER message (mid-turn steering): a user
      message is only reachable here when everything behind it (toward the
      end) is live-tail material — a turn-boundary user is always followed
      by its turn's assistant block, at which the walk has already ended;
    - walks over a settled non-injected assistant tool-call block when it is
      NESTED in the gap of another assistant tool-call block
      (:func:`_nested_in_assistant_gap`) — a completed turn's response is
      preceded by a user message and still ends the tail.

    Round-4 X1/X4 + round-5 Y: the nested rule is keyed on PROVENANCE plus
    call-id disjointness, never on call-id format. A settled block is
    nested injection material only when it carries the injector's private
    origin marker AND its call ids are DISJOINT from the outer block's
    (see :func:`_nested_in_assistant_gap`). An unmarked settled provider
    response — whatever the format of its call ids — ends the region: a
    uuid4-shaped provider id is no authority (the round-5 fail-open), and a
    marked pair colliding with the outer block's ids ends the region too,
    so the adjacent-duplicate-id shape keeps an empty region by the walk
    itself.
    """
    answered = set(_result_groups(messages))
    anchor: int | None = None
    index = len(messages) - 1
    while index >= 0:
        message = messages[index]
        if message.role == Role.tool or message.injected:
            index -= 1
            continue
        if message.role == Role.assistant and message.tool_calls:
            if any(
                (index, call_id) not in answered and call_id not in staged_call_ids
                for call_id in _call_ids(message)
            ):
                anchor = index
                break
            # A settled assistant tool-call block: nested injection material
            # when it carries the injector's provenance marker and sits in
            # another assistant call's gap (production skill/mention
            # expansion, unflagged — directly, or behind the steered user
            # message of a composed steer); any other settled block — an
            # unmarked provider response or a completed turn's response —
            # ends the tail.
            if _nested_in_assistant_gap(messages, index):
                index -= 1
                continue
            break
        if message.role == Role.user:
            # Mid-turn steering (production injects it with injected=False):
            # the walk only reaches a user message when the material behind
            # it is live-tail material, so keep walking toward the anchor.
            index -= 1
            continue
        break
    if anchor is None:
        return frozenset()
    return frozenset(
        id(message)
        for message in messages[anchor:]
        if message.role == Role.assistant and message.tool_calls
    )


@dataclass(slots=True)
class _LiveTailExemption:
    """The per-owner live-tail exemption context (round-2 V1)."""

    #: Call ids with a recorded genuine result (the historical global set).
    genuine_result_call_ids: frozenset[str]
    #: Call id -> identities of the assistant blocks owning a genuine result.
    genuine_result_owners: dict[str, set[int]]
    #: Call ids with a genuine result the projection leaves unplaced.
    unclaimed_genuine_call_ids: set[str]
    #: Identities of the assistant tool-call blocks in the live tail region.
    live_tail_region: frozenset[int]


def _live_tail_exempt_call(
    exemption: _LiveTailExemption, call_id: str, owner_identity: int | None
) -> bool:
    """Whether an unanswered call is the legitimate live tail (round-2 V1).

    A call is exempt when NO genuine result is recorded for its id anywhere
    in the transcript (the result is still in flight — the designed
    backfill's job), OR when the id's genuine results all belong to OTHER
    owner blocks: the call's OWNING assistant block sits in the live tail
    region (unanswered only because its result is still in flight), no
    genuine result was recorded FOR it, and every recorded answer for the id
    is placed in the projected order. A result that exists in the transcript
    but was never delivered to ITS owner (the stranded shape) is never exempt,
    and neither is a mid-transcript dangling call outside the live tail.
    """
    if call_id not in exemption.genuine_result_call_ids:
        return True
    if owner_identity is None or owner_identity not in exemption.live_tail_region:
        return False
    if owner_identity in exemption.genuine_result_owners.get(call_id, ()):
        return False
    if call_id in exemption.unclaimed_genuine_call_ids:
        return False
    return True


def _unanswered_tool_call_violations(
    open_calls: set[str] | None,
    answered: set[str],
    *,
    live_tail_exempt: bool,
    genuine_result_call_ids: frozenset[str],
    staged_call_ids: frozenset[str] = frozenset(),
    exemption: _LiveTailExemption | None = None,
    owner_identity: int | None = None,
) -> list[TranscriptIntegrityViolation]:
    """Violations for one assistant block's calls left unanswered.

    Round-1 S1(c): an assistant tool call left unanswered in the projected
    order is the other half of the provider 400 (a ``tool_use`` with no
    ``tool_result``) — reported, never sent. The live tail stays exempt under
    ``live_tail_exempt=True`` (see :func:`_live_tail_exempt_call`).
    ``live_tail_exempt=False`` flags every unanswered call.

    Round-2 V1: the exemption is scoped PER OWNING ASSISTANT BLOCK / live-tail
    region instead of a global id set — a call whose owning block sits in the
    live tail region is exempt even when an EARLIER owner block recorded a
    genuine result for the same id (legitimate id reuse across owner blocks).
    Round-2 V2: calls whose results are staged in an active tool batch are
    satisfied (counted as answered) in both paths, mirroring the
    SETTLED-HISTORY staged-result contract.
    """
    if not open_calls:
        return []
    unanswered = (open_calls - answered) - staged_call_ids
    if not unanswered:
        return []
    violations: list[TranscriptIntegrityViolation] = []
    for call_id in sorted(unanswered):
        if (
            live_tail_exempt
            and exemption is not None
            and _live_tail_exempt_call(exemption, call_id, owner_identity)
        ):
            continue
        detail = (
            "assistant tool call is left unanswered in the provider-bound "
            "order although the transcript records a result for it"
            if live_tail_exempt
            else "assistant tool call is left unanswered in the provider-bound order"
        )
        violations.append(
            TranscriptIntegrityViolation(
                kind="unresolved_tool_call", tool_call_id=call_id, detail=detail
            )
        )
    return violations


def validate_provider_ready(
    messages: Sequence[LLMMessage],
    *,
    live_tail_exempt: bool = True,
    staged_call_ids: frozenset[str] = frozenset(),
) -> list[TranscriptIntegrityViolation]:
    """PROVIDER-READY contract: the order request build produces is safe.

    Runs on the projected sequence: dedupe shadowed placeholders, reorder for
    tool adjacency, then verify in a single pass that every tool result sits
    in the open result run of its owning assistant call. A tool result with no
    open owner would serialize as the ``[user, tool]`` adjacency providers
    reject with ``invalid_request_message_order`` — reported, never sent.

    Round-1 S1(c): an assistant tool call left UNANSWERED in the projected
    order is the other half of the same provider 400 — also reported, never
    sent — with the live tail exempt under ``live_tail_exempt=True`` (see
    :func:`_live_tail_exempt_call`). ``live_tail_exempt=False`` flags every
    unanswered call.

    Round-2 V1: the live-tail exemption is scoped PER OWNING ASSISTANT BLOCK
    / live-tail region, not a global id set — the legitimate mid-turn reuse
    shape ``[U, A1(c1), R1(c1), U, A2(c1)]`` (A2's call in flight; A1's
    genuine result for the same id belongs to the earlier owner block) stays
    exempt, so the persistence boundary and the resume install accept the
    live save. The stranded shape (a result exists in the transcript but was
    never delivered to ITS owner) is still flagged.

    Round-3 W2 + round-5 Y: the live-tail region boundary is derived
    STRUCTURALLY from evidence production actually produces — production
    steering user messages (``as_message=True``, ``injected=False``) are
    walked over on the way to the anchor, and a settled skill/mention
    assistant pair (unflagged) joins the region only when it carries the
    injector's provenance marker, while an unmarked settled provider
    response behind a user message still ends the region whatever the
    format of its call ids (see :func:`_live_tail_region_identities`).

    Round-2 V2: ``staged_call_ids`` — calls whose results are staged in an
    active tool batch — are satisfied (counted as answered) in both paths,
    mirroring the SETTLED-HISTORY staged-result contract.
    """
    projected = reorder_for_tool_adjacency(dedupe_shadowed_placeholders(messages))
    exemption = _LiveTailExemption(
        genuine_result_call_ids=_genuine_result_call_ids(messages),
        genuine_result_owners=_genuine_result_owners(messages),
        unclaimed_genuine_call_ids=set(),
        live_tail_region=_live_tail_region_identities(
            messages, staged_call_ids=staged_call_ids
        ),
    )
    violations: list[TranscriptIntegrityViolation] = []
    open_calls: set[str] | None = None
    answered: set[str] = set()
    owner_identity: int | None = None
    for message in projected:
        if message.role == Role.assistant and message.tool_calls:
            violations.extend(
                _unanswered_tool_call_violations(
                    open_calls,
                    answered,
                    live_tail_exempt=live_tail_exempt,
                    genuine_result_call_ids=exemption.genuine_result_call_ids,
                    staged_call_ids=staged_call_ids,
                    exemption=exemption,
                    owner_identity=owner_identity,
                )
            )
            open_calls = _call_ids(message)
            answered = set()
            owner_identity = id(message)
        elif message.role == Role.tool:
            call_id = message.tool_call_id
            if call_id is None:
                # Unscoped boundary marker: the mappers convert these
                # defensively; outside the PI1 duplicate/order scope.
                continue
            if open_calls is None or call_id not in open_calls:
                violations.append(
                    TranscriptIntegrityViolation(
                        kind="misordered_tool_result",
                        tool_call_id=call_id,
                        detail=(
                            "tool result has no open owning assistant call in "
                            "the provider-bound order"
                        ),
                    )
                )
                if not is_synthetic_tool_placeholder(message):
                    # A GENUINE result the projection leaves unplaced: no
                    # unanswered call for its id may claim the live-tail
                    # exemption (the projection is doubly not provider-safe).
                    exemption.unclaimed_genuine_call_ids.add(call_id)
            else:
                answered.add(call_id)
        else:
            violations.extend(
                _unanswered_tool_call_violations(
                    open_calls,
                    answered,
                    live_tail_exempt=live_tail_exempt,
                    genuine_result_call_ids=exemption.genuine_result_call_ids,
                    staged_call_ids=staged_call_ids,
                    exemption=exemption,
                    owner_identity=owner_identity,
                )
            )
            open_calls = None
            answered = set()
            owner_identity = None
    violations.extend(
        _unanswered_tool_call_violations(
            open_calls,
            answered,
            live_tail_exempt=live_tail_exempt,
            genuine_result_call_ids=exemption.genuine_result_call_ids,
            staged_call_ids=staged_call_ids,
            exemption=exemption,
            owner_identity=owner_identity,
        )
    )
    return violations


def validate_settled_history(
    messages: Sequence[LLMMessage], *, staged_call_ids: frozenset[str] = frozenset()
) -> list[TranscriptIntegrityViolation]:
    """SETTLED-HISTORY contract over the whole transcript.

    Every assistant tool call outside the live tail must be answered by
    exactly one tool result (after the deterministic dedupe), and no tool
    result may be unclaimed. Calls whose results are staged in an active tool
    batch are legitimate (counted as satisfied).
    """
    violations = find_duplicate_tool_results(messages)
    violations.extend(_unclaimed_tool_results(messages))
    # Answered is scoped per originating assistant call: a result recorded for
    # a reused call id under a LATER assistant call satisfies that later call,
    # not the earlier one (call-id reuse across turns is legitimate).
    answered = set(_result_groups(messages))
    for index, message in enumerate(messages):
        if message.role != Role.assistant or not message.tool_calls:
            continue
        for call_id in _call_ids(message):
            if (index, call_id) in answered or call_id in staged_call_ids:
                continue
            violations.append(
                TranscriptIntegrityViolation(
                    kind="unresolved_tool_call",
                    tool_call_id=call_id,
                    detail="settled assistant tool call has no tool result",
                )
            )
    return _dedupe_violations(violations)


def validate_live_prefix(
    messages: Sequence[LLMMessage], *, staged_call_ids: frozenset[str] = frozenset()
) -> list[TranscriptIntegrityViolation]:
    """LIVE-PREFIX contract: the live tail is legitimate.

    Unresolved calls, staged batch results, and nested injected
    assistant/tool pairs (skill/mention expansion, question callbacks,
    agent-lifecycle injections) are LEGITIMATE and never violations. The
    defects no live state can excuse — unrepairable duplicate results and
    unclaimed tool results — are still reported.

    The live tail's boundary (round-3 W2, round-4 X1, round-5 Y): the walk
    runs backward over the staged results and the nested injected pairs in
    the anchor's gap — including across the non-injected steered user
    messages of a composed steer — and a settled assistant block joins the
    tail only when it carries the injector's provenance marker; an unmarked
    settled provider response ends the tail whatever the format of its
    call ids.
    """
    violations = find_duplicate_tool_results(messages)
    violations.extend(_unclaimed_tool_results(messages))
    return _dedupe_violations(violations)


def validate_transcript(
    messages: Sequence[LLMMessage],
    *,
    staged_call_ids: frozenset[str] = frozenset(),
    live_tail_exempt: bool = True,
) -> list[TranscriptIntegrityViolation]:
    """The composite admission check: LIVE-PREFIX + SETTLED-HISTORY + PROVIDER-READY.

    ``live_tail_exempt=True`` (load repair, resume install, turn admission,
    persistence boundary, AND request build): unresolved calls are
    legitimate — the designed backfill repairs dangling calls at the act()
    boundary (and a sideband completion may run over a never-executed tool
    call), so only structural defects (unrepairable duplicates, unclaimed
    results) and order defects are reported. PROVIDER-READY rides the same
    flag (round-1 S1(c)): under the exemption an assistant tool call left
    unanswered in the projected order is reported only when the transcript
    already records a genuine result for its id (the stranded-call shape);
    a call whose result is still in flight stays legitimate.
    ``live_tail_exempt=False`` is the strict SETTLED-HISTORY contract: every
    unresolved call is a violation — for callers that must prove completeness.

    Calls whose results are staged in an active tool batch
    (``staged_call_ids``) count as satisfied in every sub-check, mirroring
    the SETTLED-HISTORY staged-result contract (round-2 V2).
    """
    violations = validate_live_prefix(messages, staged_call_ids=staged_call_ids)
    if not live_tail_exempt:
        violations.extend(
            validate_settled_history(messages, staged_call_ids=staged_call_ids)
        )
    violations.extend(
        validate_provider_ready(
            messages, live_tail_exempt=live_tail_exempt, staged_call_ids=staged_call_ids
        )
    )
    return _dedupe_violations(violations)


def repair_transcript(
    messages: Sequence[LLMMessage],
) -> tuple[list[LLMMessage], TranscriptIntegrityReport]:
    """Validate + deterministically repair the unambiguous PI1 pattern.

    The repair drops synthetic ``TOOL_NO_RESPONSE`` placeholders shadowed by a
    genuine result — never a genuine structured tool result — and changes
    nothing else (the raw ``[assistant, user, genuine result]`` order request
    build already repairs is left as-is). Idempotent: an already-repaired
    transcript is an exact no-op. Ambiguous cases are reported, never guessed.
    The drop rides the shared owner-scoped predicate
    (``shadowed_placeholder_positions``), so the repaired transcript and the
    repair report always agree.
    """
    drop = shadowed_placeholder_positions(messages)
    deduped = (
        [message for position, message in enumerate(messages) if position not in drop]
        if drop
        else list(messages)
    )
    violations = validate_transcript(messages, live_tail_exempt=True)
    dropped_call_ids: set[str] = set()
    for position in drop:
        call_id = messages[position].tool_call_id
        if call_id is not None:
            dropped_call_ids.add(call_id)
    report = TranscriptIntegrityReport(
        repaired=bool(drop),
        dropped_placeholder_call_ids=tuple(sorted(dropped_call_ids)),
        violations=tuple(violations),
    )
    return deduped, report


def find_persistence_violations(
    messages: Sequence[LLMMessage],
) -> list[TranscriptIntegrityViolation]:
    """S1 persistence-boundary check: nothing that would (re)poison the log.

    Stricter than the load/admission contracts: ANY duplicate tool result —
    including the unambiguous placeholder-shadowed-by-genuine pattern the
    load-time repair heals — is a violation here. The repair exists for
    transcripts already on disk; the save boundary never writes new poison.
    Unclaimed results and provider-unsafe order are reported as usual, with
    the live tail exempt (unresolved calls are legitimate mid-turn: the
    designed backfill repairs them before any request). Duplicates are scoped
    per originating assistant call (call-id reuse across turns is legitimate).
    """
    violations: list[TranscriptIntegrityViolation] = []
    for (_, call_id), indexes in _result_groups(messages).items():
        if len(indexes) < _MIN_RESULTS_FOR_DUPLICATE:
            continue
        violations.append(
            TranscriptIntegrityViolation(
                kind="duplicate_tool_result",
                tool_call_id=call_id,
                detail=(
                    f"{len(indexes)} tool results for one call; the persistence "
                    "boundary never writes a duplicate"
                ),
            )
        )
    violations.extend(validate_transcript(messages, live_tail_exempt=True))
    return _dedupe_violations(violations)


def check_tool_message_admission(
    messages: Sequence[LLMMessage], index: int, message: LLMMessage
) -> None:
    """S1 append-boundary check for one tool message about to enter the list.

    No append/insert path may create a duplicate tool result for the same
    tool_call_id (scoped per originating assistant call — tool-call ids are
    reused across turns, and results under different assistant calls are not
    duplicates) or a ``[user, tool]`` adjacency that request build cannot
    repair. Violations fail closed with the typed diagnostic — never silently
    persisted. Appending a genuine result where a synthetic placeholder exists
    is the DESIGNED replacement, not a violation (the caller replaces the
    placeholder in place).

    A result with NO owning assistant call is not a duplicate (there is no
    owner scope to duplicate within); the append is admitted unless it lands
    directly behind a user message, where the unplaceable result would
    serialize as the unrepairable ``[user, tool]`` 400 shape. The ownerless
    append itself (the direct-execution seam records results for calls with
    no recorded assistant message) is policed by the request-build, save,
    load, and admission seams, which fail closed on the resulting
    transcript.
    """
    if message.role != Role.tool or not message.tool_call_id:
        return
    call_id = message.tool_call_id
    owner = _owner_before(messages, index, call_id)
    existing = [
        scanned
        for position, scanned in enumerate(messages)
        if position < index
        and scanned.role == Role.tool
        and scanned.tool_call_id == call_id
        and _owner_before(messages, position, call_id) == owner
    ]
    genuine_incoming = has_structured_tool_result(message)
    if existing:
        genuine_existing = any(
            not is_synthetic_tool_placeholder(existing_message)
            for existing_message in existing
        )
        if genuine_incoming and genuine_existing:
            raise TranscriptIntegrityError(
                [
                    TranscriptIntegrityViolation(
                        kind="duplicate_tool_result",
                        tool_call_id=call_id,
                        detail="appending a second genuine tool result",
                    )
                ],
                context="transcript append boundary",
            )
        if not genuine_incoming and genuine_existing:
            raise TranscriptIntegrityError(
                [
                    TranscriptIntegrityViolation(
                        kind="duplicate_tool_result",
                        tool_call_id=call_id,
                        detail="appending a synthetic placeholder over a genuine result",
                    )
                ],
                context="transcript append boundary",
            )
        if not genuine_incoming and not genuine_existing:
            raise TranscriptIntegrityError(
                [
                    TranscriptIntegrityViolation(
                        kind="duplicate_tool_result",
                        tool_call_id=call_id,
                        detail="appending a duplicate synthetic placeholder",
                    )
                ],
                context="transcript append boundary",
            )
        return  # designed replacement of a synthetic placeholder
    if owner is None:
        # No owning assistant call: not a duplicate (there is no owner scope
        # to duplicate within). The append is admitted UNLESS it lands
        # directly behind a user message — an unplaceable result there
        # would serialize as the unrepairable [user, tool] 400 shape. The
        # ownerless append itself (the direct-execution seam records results
        # for calls with no recorded assistant message) is policed by the
        # request-build, save, load, and admission seams, which fail closed
        # on the resulting transcript.
        if index > 0 and messages[index - 1].role == Role.user:
            raise TranscriptIntegrityError(
                [
                    TranscriptIntegrityViolation(
                        kind="misordered_tool_result",
                        tool_call_id=call_id,
                        detail=(
                            "appending a tool result after a user message with "
                            "no owning assistant call before it"
                        ),
                    )
                ],
                context="transcript append boundary",
            )
        return
    # An owning assistant call exists before the append point: a [user,
    # tool] adjacency here is repairable by request build's reorder
    # ([assistant, user, tool] reorders to [assistant, tool, user]), so the
    # append is admitted.


def report_to_metadata(report: TranscriptIntegrityReport) -> dict[str, object]:
    """Serialize a repair report for the ``load_session`` metadata ride-along."""
    return {
        "repaired": report.repaired,
        "dropped_placeholder_call_ids": list(report.dropped_placeholder_call_ids),
        "violations": [
            {
                "kind": violation.kind,
                "tool_call_id": violation.tool_call_id,
                "detail": violation.detail,
            }
            for violation in report.violations
        ],
    }


def report_from_metadata(value: object) -> TranscriptIntegrityReport | None:
    """Parse a repair report from the ``load_session`` metadata ride-along."""
    if not isinstance(value, dict):
        return None
    dropped = value.get("dropped_placeholder_call_ids")
    raw_violations = value.get("violations")
    violations: list[TranscriptIntegrityViolation] = []
    if isinstance(raw_violations, list):
        for item in raw_violations:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            detail = item.get("detail")
            if not isinstance(kind, str) or not isinstance(detail, str):
                continue
            call_id = item.get("tool_call_id")
            violations.append(
                TranscriptIntegrityViolation(
                    kind=kind,  # type: ignore[arg-type]
                    tool_call_id=call_id if isinstance(call_id, str) else None,
                    detail=detail,
                )
            )
    return TranscriptIntegrityReport(
        repaired=bool(value.get("repaired")),
        dropped_placeholder_call_ids=tuple(
            call_id for call_id in (dropped or []) if isinstance(call_id, str)
        ),
        violations=tuple(violations),
    )


def pop_report_from_metadata(
    metadata: dict[str, object],
) -> TranscriptIntegrityReport | None:
    """Remove and parse the repair report a ``load_session`` call attached."""
    value = metadata.pop(TRANSCRIPT_INTEGRITY_METADATA_KEY, None)
    return report_from_metadata(value)
