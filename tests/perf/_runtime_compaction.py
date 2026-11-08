"""X8a compaction / pre-seal execution-cost cell (v0.7.5 stage 2).

Plan §4 files compaction and pre-seal execution cost under the observation-only
set: the quantity is reported in full with its recorded reasons and is never
gated. The reason is structural. One eligible prefix exists per transcript
state, and publishing a release mutates archive state — a seal receipt, an
:class:`~chartreux.core.session.transcript_archive.ArchivedPrefixV1` descriptor
and a RAM replacement of the sealed messages — so no repeatable >= 100-op unit
exists for either the P3 pre-seal release or the post-compaction trim.

What the cell models is the landed P3 delta (plan §2): the pre-seal release
retains the newest ``PRE_SEAL_RETAIN_RECENT_USER_ROUNDS`` (2) user-started
rounds in full and releases only the longest settled prefix strictly before the
older of them (``transcript_archive.py:301``, ``:342-378``); a release attempt
taken without a compaction boundary carries a BOUNDARY-NONE descriptor
(``manager.py:133-151``) whose attempt key is the ``"pre-seal"`` key derived
from the stable eligible-prefix endpoint — never a bare ``None`` key
(``transcript_archive.py:385-457``).

The cell drives those production primitives over a shortened transcript (plan
§5: in-suite contract nodes never execute the full 200k fixture) and times the
in-process attempt: eligibility, endpoint/key derivation, descriptor build over
the simulated persisted prefix bytes, and the per-message replacement
preparation. Publication is deliberately NOT executed, so the live transcript
keeps every payload — asserted by fingerprint — which is exactly why these
numbers stay observation-only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import random
from statistics import median
from time import perf_counter_ns
from typing import Any

from chartreux.core.compaction.manager import PendingPostTurnTrim
from chartreux.core.llm_models import (
    FunctionCall,
    LLMMessage,
    Role,
    ToolCall,
    archived_replacement,
)
from chartreux.core.session.transcript_archive import (
    PRE_SEAL_RETAIN_RECENT_USER_ROUNDS,
    ArchivedPrefixV1,
    TrimAttemptKey,
    eligible_archive_prefix,
    eligible_pre_seal_prefix,
    pre_seal_endpoint,
    release_key_is_new,
    trim_attempt_key,
)
from tests.perf._runtime_fixture import FIXTURE_TURNS
from tests.perf._runtime_metrics import resolve_perf_paths
from tests.perf._runtime_report import METRIC_VERSION, build_report, validate_report

#: The observation cell's identity; it is not a registry campaign ID (§4's
#: closed registry carries the 17 gated rows only, and this cell is not one).
COMPACTION_CELL = "x8_compaction_pre_seal"
COMPACTION_SCHEMA = "chartreux.runtime_compaction.v1"
MODE_CONTRACT = "contract"

BLOB_SECTION = "compaction_cost"
BLOB_LINE_PREFIX = f"PERF-BLOB:{COMPACTION_CELL}:"

#: Plan §4's observation-only disposition, recorded on every emitted record.
OBSERVATION_ONLY_REASON = (
    "plan §4: compaction/pre-seal execution cost is observation-only"
)
NO_REPEATABLE_UNIT_REASON = (
    "no repeatable 100-op unit exists without mutating archive state"
)
PRE_SEAL_ONE_PREFIX_REASON = (
    "one eligible pre-seal prefix per transcript state; the prefix advances "
    "only when the transcript grows"
)
BOUNDARY_ONE_SEAL_REASON = (
    "a post-compaction trim seals once per compaction boundary (durable "
    "persist + verify of the sealed prefix)"
)
COMPACTION_REASONS: tuple[str, ...] = (
    OBSERVATION_ONLY_REASON,
    NO_REPEATABLE_UNIT_REASON,
    PRE_SEAL_ONE_PREFIX_REASON,
    BOUNDARY_ONE_SEAL_REASON,
)

#: Exactly what the timed span covers, so the reported numbers cannot be read
#: as a full seal.
TIMED_SCOPE: tuple[str, ...] = (
    "pre-seal / archive eligibility over the transcript",
    "stable eligible-prefix endpoint and progressive release key derivation",
    "descriptor build over the simulated persisted prefix bytes (count + sha256)",
    "per-message payload-replacement preparation",
)

#: Deliberately NOT executed: each one mutates archive state or needs durable
#: IO, which is why the quantity is not a repeatable gated unit.
EXCLUDED_SCOPE: tuple[str, ...] = (
    "durable persist and on-disk prefix verification (the seal receipt)",
    "publication of the replacements into the live transcript",
)

#: Deterministic transcript provenance (plan §5: a fixed seed, fixed message
#: sizes and a fixed round shape; nothing clock- or network-dependent enters).
COMPACTION_SEED = 20261108
COMPACTION_SESSION_ID = "x8a-compaction-observation"
COMPACTION_BOUNDARY_ID = "compaction-boundary-0001"
ROUND_WIDTH = 4
MESSAGE_CHARS = 200
BASE_ROUNDS = 4
POST_BOUNDARY_ROUNDS = 2

#: One contract-node measurement: warmups discarded, then the reported attempts.
WARMUP_ATTEMPTS = 3
MEASURED_ATTEMPTS = 12

_CONTENT_STREAM_CHARS = 4_096
_CONTENT_STRIDE = 733
_CONTENT_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_TOOL_NAME = "compaction_probe"


@lru_cache(maxsize=1)
def _content_stream() -> str:
    """Return the seeded content stream, doubled so any window fits in a slice."""
    generator = random.Random(COMPACTION_SEED)
    width = len(_CONTENT_ALPHABET)
    single = "".join(
        _CONTENT_ALPHABET[int(generator.random() * width)]
        for _ in range(_CONTENT_STREAM_CHARS)
    )
    return single + single


def _seeded_text(prefix: str, offset: int, char_count: int) -> str:
    """Build exactly ``char_count`` characters: the prefix plus a seeded window."""
    stream = _content_stream()
    start = offset % _CONTENT_STREAM_CHARS
    return (prefix + stream[start : start + char_count])[:char_count]


def _call_id(round_index: int) -> str:
    """Return the committed tool-call id of one history round."""
    return f"compaction-call-{round_index:04d}"


def _round_messages(round_index: int, *, payloads: bool) -> list[LLMMessage]:
    """Build the four messages of one complete settled history round.

    The round is ``user -> assistant tool call -> tool result -> assistant``, so
    the settled-round rule has a real open/answered tool pair to walk. Both
    assistant messages carry a reasoning payload; without payloads the pre-seal
    prefix has nothing to release and the eligibility rule correctly declines.
    """
    base = round_index * _CONTENT_STRIDE
    call_id = _call_id(round_index)
    tool_call = ToolCall(
        id=call_id, index=0, function=FunctionCall(name=_TOOL_NAME, arguments="{}")
    )
    return [
        LLMMessage(
            role=Role.user,
            content=_seeded_text(
                f"compaction-user-{round_index:04d} ", base, MESSAGE_CHARS
            ),
            message_id=f"compaction-user-{round_index:04d}",
        ),
        LLMMessage(
            role=Role.assistant,
            content=_seeded_text(
                f"compaction-call-{round_index:04d} ", base + 101, MESSAGE_CHARS
            ),
            tool_calls=[tool_call],
            reasoning_payloads=(
                [{"type": "thinking", "text": f"reasoning-call-{round_index:04d}"}]
                if payloads
                else None
            ),
            reasoning_message_id=(
                f"compaction-reasoning-call-{round_index:04d}" if payloads else None
            ),
            message_id=f"compaction-call-message-{round_index:04d}",
        ),
        LLMMessage(
            role=Role.tool,
            content=_seeded_text(
                f"compaction-result-{round_index:04d} ", base + 211, MESSAGE_CHARS
            ),
            tool_call_id=call_id,
        ),
        LLMMessage(
            role=Role.assistant,
            content=_seeded_text(
                f"compaction-answer-{round_index:04d} ", base + 317, MESSAGE_CHARS
            ),
            reasoning_payloads=(
                [{"type": "thinking", "text": f"reasoning-answer-{round_index:04d}"}]
                if payloads
                else None
            ),
            reasoning_message_id=(
                f"compaction-reasoning-answer-{round_index:04d}" if payloads else None
            ),
            message_id=f"compaction-answer-{round_index:04d}",
        ),
    ]


def _boundary_message() -> LLMMessage:
    """Return a compaction boundary envelope (the newest archive boundary)."""
    return LLMMessage(
        role=Role.user,
        content="compaction boundary: summarized prefix",
        injected=True,
        context_boundary="compaction",
        message_id=COMPACTION_BOUNDARY_ID,
    )


def build_transcript(
    rounds: int, *, payloads: bool = True, with_boundary: bool = False
) -> list[LLMMessage]:
    """Build a shortened transcript of ``rounds`` settled history rounds.

    ``with_boundary`` appends a compaction boundary envelope followed by
    ``POST_BOUNDARY_ROUNDS`` post-boundary rounds, which is the active-invocation
    shape the post-compaction trim must never seal.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be positive, got {rounds}")
    messages: list[LLMMessage] = []
    for index in range(rounds):
        messages.extend(_round_messages(index, payloads=payloads))
    if with_boundary:
        messages.append(_boundary_message())
        for index in range(rounds, rounds + POST_BOUNDARY_ROUNDS):
            messages.extend(_round_messages(index, payloads=payloads))
    return messages


def transcript_fingerprint(messages: Sequence[LLMMessage]) -> str:
    """Return a digest of every field a release could change (proof of no trim)."""
    digest = hashlib.sha256()
    for message in messages:
        digest.update(f"{message.role.value}|{message.message_id or '-'}|".encode())
        digest.update((message.content or "").encode("utf-8"))
        for payload in message.reasoning_payloads or ():
            digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        for call in message.tool_calls or ():
            name = call.function.name if call.function is not None else "-"
            digest.update(f"{call.id}|{name}".encode())
    return digest.hexdigest()


def payload_message_count(messages: Sequence[LLMMessage]) -> int:
    """Return the number of messages still carrying a reasoning payload."""
    return sum(1 for message in messages if message.reasoning_payloads)


def _persisted_prefix_bytes(messages: Sequence[LLMMessage]) -> bytes:
    """Serialize ``messages`` exactly as the transcript holds them (JSONL)."""
    lines: list[bytes] = []
    for message in messages:
        record = json.dumps(
            message.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        lines.append(record.encode("utf-8"))
    return b"".join(line + b"\n" for line in lines)


def build_prefix_descriptor(
    messages: Sequence[LLMMessage],
    *,
    session_id: str,
    message_epoch: int,
    boundary_message_id: str | None,
    trim_mode: str = "payloads",
) -> ArchivedPrefixV1:
    """Build the descriptor a seal would issue for ``messages``.

    The byte count and sha256 are computed over the exact persisted prefix
    bytes, which is the work the sealing operation performs; ``boundary_message_id``
    is ``None`` for a P3 pre-seal release, which carries no compaction boundary.
    """
    if trim_mode != "payloads":
        raise ValueError(f"the modelled trim mode is payloads, got {trim_mode!r}")
    blob = _persisted_prefix_bytes(messages)
    return ArchivedPrefixV1(
        session_id=session_id,
        message_epoch=message_epoch,
        archived_count=len(messages),
        prefix_end_byte=len(blob),
        prefix_sha256=hashlib.sha256(blob).hexdigest(),
        boundary_message_id=boundary_message_id,
        trim_mode="payloads",
    )


@dataclass(frozen=True, slots=True)
class PreSealAttempt:
    """One timed P3 pre-seal release attempt over one transcript state."""

    duration_ms: float
    suppressed: bool
    endpoint: str
    attempt_key: TrimAttemptKey
    pending: PendingPostTurnTrim
    descriptor: ArchivedPrefixV1 | None
    retain_recent_user_rounds: int
    user_round_count: int
    retained_round_count: int
    retained_message_count: int
    eligible_message_count: int
    replaced_messages: int
    eligible_payload_messages: int
    retained_payload_messages: int


@dataclass(frozen=True, slots=True)
class BoundaryAttempt:
    """One timed post-compaction trim attempt over the boundary transcript."""

    duration_ms: float
    suppressed: bool
    boundary_message_id: str
    attempt_key: TrimAttemptKey
    pending: PendingPostTurnTrim
    descriptor: ArchivedPrefixV1 | None
    eligible_message_count: int
    post_boundary_message_count: int
    replaced_messages: int


def pre_seal_attempt(
    messages: Sequence[LLMMessage],
    *,
    session_id: str = COMPACTION_SESSION_ID,
    message_epoch: int = 0,
    published: set[TrimAttemptKey] | None = None,
) -> PreSealAttempt:
    """Time one boundary-none pre-seal release attempt over ``messages``.

    The attempt runs eligibility, endpoint/key derivation, the descriptor build
    and the replacement preparation, and publishes its key into ``published``.
    It never places the replacements into the live list: publication is the
    archive mutation that makes this quantity unrepeatable.
    """
    published_keys: set[TrimAttemptKey] = set() if published is None else published
    prefix = eligible_pre_seal_prefix(messages)
    if prefix is None:
        raise ValueError("the transcript carries no eligible pre-seal prefix")
    non_system = [message for message in messages if message.role is not Role.system]
    user_starts = [
        index for index, message in enumerate(non_system) if message.role is Role.user
    ]
    retained_start_index = len(user_starts) - PRE_SEAL_RETAIN_RECENT_USER_ROUNDS
    retained = non_system[user_starts[retained_start_index] :]

    started = perf_counter_ns()
    endpoint = pre_seal_endpoint(prefix)
    # P3 boundary-none descriptor: no compaction boundary exists yet, so both
    # the pending trim and the seal descriptor carry None, and the attempt key
    # is derived from the stable eligible-prefix endpoint.
    pending = PendingPostTurnTrim(
        session_id=session_id, message_epoch=message_epoch, boundary_message_id=None
    )
    key = trim_attempt_key(
        session_id, message_epoch, pending.boundary_message_id, endpoint
    )
    suppressed = not release_key_is_new(published_keys, key)
    descriptor: ArchivedPrefixV1 | None = None
    replaced = 0
    if not suppressed:
        descriptor = build_prefix_descriptor(
            prefix.messages,
            session_id=session_id,
            message_epoch=message_epoch,
            boundary_message_id=pending.boundary_message_id,
        )
        replacements = [
            archived_replacement(message, descriptor, trim_mode=descriptor.trim_mode)
            for message in prefix.messages
        ]
        replaced = len(replacements)
        published_keys.add(key)
    finished = perf_counter_ns()

    return PreSealAttempt(
        duration_ms=(finished - started) / _NANOSECONDS_PER_MILLISECOND,
        suppressed=suppressed,
        endpoint=endpoint,
        attempt_key=key,
        pending=pending,
        descriptor=descriptor,
        retain_recent_user_rounds=PRE_SEAL_RETAIN_RECENT_USER_ROUNDS,
        user_round_count=len(user_starts),
        retained_round_count=len(user_starts) - retained_start_index,
        retained_message_count=len(retained),
        eligible_message_count=len(prefix.messages),
        replaced_messages=replaced,
        eligible_payload_messages=payload_message_count(prefix.messages),
        retained_payload_messages=payload_message_count(retained),
    )


def boundary_trim_attempt(
    messages: Sequence[LLMMessage],
    *,
    session_id: str = COMPACTION_SESSION_ID,
    message_epoch: int = 0,
    published: set[TrimAttemptKey] | None = None,
) -> BoundaryAttempt:
    """Time one post-compaction trim attempt over a boundary transcript.

    Once a real boundary exists, post-seal alone owns the release, so the
    attempt keys on the boundary id (the coalescing rule) and the descriptor
    carries it. The eligible prefix is the closed region strictly before the
    newest boundary; the post-boundary (active-invocation) rounds stay live.
    """
    published_keys: set[TrimAttemptKey] = set() if published is None else published
    prefix = eligible_archive_prefix(messages)
    if prefix is None or prefix.boundary_message_id is None:
        raise ValueError("the transcript carries no eligible archive prefix")
    non_system = [message for message in messages if message.role is not Role.system]
    boundary_position = next(
        index
        for index, message in enumerate(non_system)
        if message.message_id == prefix.boundary_message_id
    )

    started = perf_counter_ns()
    pending = PendingPostTurnTrim(
        session_id=session_id,
        message_epoch=message_epoch,
        boundary_message_id=prefix.boundary_message_id,
    )
    key = trim_attempt_key(
        session_id,
        message_epoch,
        pending.boundary_message_id,
        pre_seal_endpoint_for(prefix.messages),
    )
    suppressed = not release_key_is_new(published_keys, key)
    descriptor: ArchivedPrefixV1 | None = None
    replaced = 0
    if not suppressed:
        descriptor = build_prefix_descriptor(
            prefix.messages,
            session_id=session_id,
            message_epoch=message_epoch,
            boundary_message_id=pending.boundary_message_id,
        )
        replacements = [
            archived_replacement(message, descriptor, trim_mode=descriptor.trim_mode)
            for message in prefix.messages
        ]
        replaced = len(replacements)
        published_keys.add(key)
    finished = perf_counter_ns()

    return BoundaryAttempt(
        duration_ms=(finished - started) / _NANOSECONDS_PER_MILLISECOND,
        suppressed=suppressed,
        boundary_message_id=str(prefix.boundary_message_id),
        attempt_key=key,
        pending=pending,
        descriptor=descriptor,
        eligible_message_count=len(prefix.messages),
        post_boundary_message_count=(len(non_system) - boundary_position - 1),
        replaced_messages=replaced,
    )


def pre_seal_endpoint_for(messages: Sequence[LLMMessage]) -> str:
    """Return the stable endpoint string of a closed prefix (last message id)."""
    last = messages[-1]
    return f"{len(messages)}:{last.message_id or '-'}"


@dataclass(frozen=True, slots=True)
class CompactionObservation:
    """The complete observation-only result of one compaction-cell execution."""

    workload: Mapping[str, Any]
    pre_seal: PreSealAttempt
    boundary_trim: BoundaryAttempt
    pre_seal_samples: tuple[PreSealAttempt, ...]
    suppressed_samples: tuple[PreSealAttempt, ...]
    boundary_samples: tuple[BoundaryAttempt, ...]
    transcript_fingerprint: str
    archive_mutated: bool

    def record(self) -> dict[str, Any]:
        """Return the observation's additive-section record (never gated)."""
        pre_seal = self.pre_seal
        boundary = self.boundary_trim
        descriptor = pre_seal.descriptor
        boundary_descriptor = boundary.descriptor
        return {
            "cell": COMPACTION_CELL,
            "schema": COMPACTION_SCHEMA,
            "metric_version": METRIC_VERSION,
            "mode": MODE_CONTRACT,
            "gated": False,
            "observation_only": True,
            "reasons": list(COMPACTION_REASONS),
            "timed_scope": list(TIMED_SCOPE),
            "excluded_scope": list(EXCLUDED_SCOPE),
            "archive_mutated": self.archive_mutated,
            "transcript_fingerprint": self.transcript_fingerprint,
            "workload": dict(self.workload),
            "pre_seal": {
                "retain_recent_user_rounds": pre_seal.retain_recent_user_rounds,
                "user_round_count": pre_seal.user_round_count,
                "retained_round_count": pre_seal.retained_round_count,
                "retained_message_count": pre_seal.retained_message_count,
                "eligible_message_count": pre_seal.eligible_message_count,
                "eligible_payload_messages": pre_seal.eligible_payload_messages,
                "retained_payload_messages": pre_seal.retained_payload_messages,
                "boundary_message_id": pre_seal.pending.boundary_message_id,
                "descriptor_boundary_message_id": (
                    None if descriptor is None else descriptor.boundary_message_id
                ),
                "attempt_key_kind": pre_seal.attempt_key[0],
                "endpoint": pre_seal.endpoint,
                "descriptor_archived_count": (
                    None if descriptor is None else descriptor.archived_count
                ),
                "descriptor_prefix_end_byte": (
                    None if descriptor is None else descriptor.prefix_end_byte
                ),
                "descriptor_trim_mode": (
                    None if descriptor is None else descriptor.trim_mode
                ),
                "replaced_messages": pre_seal.replaced_messages,
                "suppressed_attempt": pre_seal.suppressed,
            },
            "boundary_trim": {
                "boundary_message_id": boundary.boundary_message_id,
                "attempt_key_kind": boundary.attempt_key[0],
                "eligible_message_count": boundary.eligible_message_count,
                "post_boundary_message_count": (boundary.post_boundary_message_count),
                "replaced_messages": boundary.replaced_messages,
                "descriptor_boundary_message_id": (
                    None
                    if boundary_descriptor is None
                    else boundary_descriptor.boundary_message_id
                ),
                "descriptor_prefix_end_byte": (
                    None
                    if boundary_descriptor is None
                    else boundary_descriptor.prefix_end_byte
                ),
                "descriptor_prefix_sha256": (
                    None
                    if boundary_descriptor is None
                    else boundary_descriptor.prefix_sha256
                ),
            },
            "samples": {
                "pre_seal_advance": _summarize_pre_seal(self.pre_seal_samples),
                "pre_seal_suppressed": _summarize_pre_seal(self.suppressed_samples),
                "boundary_trim": _summarize_boundary(self.boundary_samples),
            },
            "suppressed_attempts": {
                "pre_seal": sum(
                    1 for attempt in self.suppressed_samples if attempt.suppressed
                ),
                "boundary": sum(
                    1 for attempt in self.boundary_samples if attempt.suppressed
                ),
            },
        }


def _summarize_durations(durations: list[float]) -> dict[str, Any]:
    """Return the observation summary of one attempt series (never a gate)."""
    return {
        "count": len(durations),
        "unit": "ms",
        "min_ms": min(durations),
        "median_ms": median(durations),
        "max_ms": max(durations),
        "durations_ms": durations,
    }


def _summarize_pre_seal(attempts: Sequence[PreSealAttempt]) -> dict[str, Any]:
    """Summarize one pre-seal attempt series."""
    summary = _summarize_durations([attempt.duration_ms for attempt in attempts])
    summary["warmup_attempts"] = WARMUP_ATTEMPTS
    summary["replaced_messages"] = [attempt.replaced_messages for attempt in attempts]
    summary["suppressed"] = [attempt.suppressed for attempt in attempts]
    return summary


def _summarize_boundary(attempts: Sequence[BoundaryAttempt]) -> dict[str, Any]:
    """Summarize the boundary-trim attempt series."""
    summary = _summarize_durations([attempt.duration_ms for attempt in attempts])
    summary["warmup_attempts"] = WARMUP_ATTEMPTS
    summary["replaced_messages"] = [attempt.replaced_messages for attempt in attempts]
    summary["suppressed"] = [attempt.suppressed for attempt in attempts]
    return summary


def _workload_record(first_rounds: int, last_rounds: int) -> dict[str, Any]:
    """Return the committed workload block of one observation."""
    return {
        "cell": COMPACTION_CELL,
        "transcript": {
            "shortened": True,
            "seed": COMPACTION_SEED,
            "round_width": ROUND_WIDTH,
            "message_chars": MESSAGE_CHARS,
            "advancing_rounds": [first_rounds, last_rounds],
            "boundary_rounds": BASE_ROUNDS,
            "post_boundary_rounds": POST_BOUNDARY_ROUNDS,
            "full_scale_turns": FIXTURE_TURNS,
        },
        "attempts": {"warmup": WARMUP_ATTEMPTS, "measured": MEASURED_ATTEMPTS},
    }


def observe_compaction_cell() -> CompactionObservation:
    """Run the cell: pre-seal advance, duplicate suppression, boundary trim.

    Every transcript state advances by one round, so each pre-seal attempt
    derives a new endpoint and key (a full attempt). The duplicate series times
    the suppressed path — the same state attempted again, which is what a
    coalesced or retried release does. The boundary series advances the message
    epoch so each attempt keys on the boundary id at a new epoch, and its last
    entry repeats the previous epoch to prove boundary-key suppression.
    """
    published: set[TrimAttemptKey] = set()
    pre_seal_samples: list[PreSealAttempt] = []
    last_messages: list[LLMMessage] = []
    for attempt in range(WARMUP_ATTEMPTS + MEASURED_ATTEMPTS):
        last_messages = build_transcript(BASE_ROUNDS + attempt)
        sample = pre_seal_attempt(last_messages, published=published)
        if attempt >= WARMUP_ATTEMPTS:
            pre_seal_samples.append(sample)

    before = transcript_fingerprint(last_messages)
    suppressed_samples = [
        pre_seal_attempt(last_messages, published=published)
        for _ in range(MEASURED_ATTEMPTS)
    ]
    after = transcript_fingerprint(last_messages)

    boundary_messages = build_transcript(BASE_ROUNDS, with_boundary=True)
    boundary_before = transcript_fingerprint(boundary_messages)
    boundary_published: set[TrimAttemptKey] = set()
    boundary_samples: list[BoundaryAttempt] = []
    for attempt in range(WARMUP_ATTEMPTS + MEASURED_ATTEMPTS):
        sample = boundary_trim_attempt(
            boundary_messages, message_epoch=attempt, published=boundary_published
        )
        if attempt >= WARMUP_ATTEMPTS:
            boundary_samples.append(sample)
    boundary_samples.append(
        boundary_trim_attempt(
            boundary_messages,
            message_epoch=WARMUP_ATTEMPTS + MEASURED_ATTEMPTS - 1,
            published=boundary_published,
        )
    )
    boundary_after = transcript_fingerprint(boundary_messages)

    return CompactionObservation(
        workload=_workload_record(
            BASE_ROUNDS, BASE_ROUNDS + WARMUP_ATTEMPTS + MEASURED_ATTEMPTS - 1
        ),
        pre_seal=pre_seal_samples[-1],
        boundary_trim=boundary_samples[0],
        pre_seal_samples=tuple(pre_seal_samples),
        suppressed_samples=tuple(suppressed_samples),
        boundary_samples=tuple(boundary_samples),
        transcript_fingerprint=before,
        archive_mutated=before != after or boundary_before != boundary_after,
    )


def build_compaction_blob(observation: CompactionObservation) -> dict[str, Any]:
    """Build and validate the WP6-shaped blob carrying the observed cost."""
    document = build_report(())
    document[BLOB_SECTION] = observation.record()
    validate_report(document)
    return document


def emit_compaction_blob(
    blob: Mapping[str, Any], *, environ: Mapping[str, str] | None = None
) -> str:
    """Print the observation blob and append it through the WP1 path guard."""
    paths = resolve_perf_paths(environ)
    blob_text = json.dumps(blob, sort_keys=True, separators=(",", ":"))
    line = f"{BLOB_LINE_PREFIX}{blob_text}"
    print(line)
    if paths.report is not None:
        with paths.report.open("a", encoding="utf-8") as report_file:
            report_file.write(f"{line}\n")
    return line
