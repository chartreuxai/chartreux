"""The committed 200k-equivalent transcript fixture (v0.7.5 stage 2, WP3a).

Every loaded/save/soak cell of the runtime campaign measures against the same
200k-equivalent history, and run eligibility binds a run to its cell's committed
fixture hash (Appendix B ``run_eligibility.fixture_hash``). This module is that
fixture's single source: the seed, the message geometry, and the deterministic
response shape are pinned here together with :data:`FIXTURE_HASH`, and the
digest covers the specification AND the generated transcript, so drift in either
changes the hash.

The fixture is generated in-process and validated in-process on the landed P5
save-boundary path (:meth:`SessionLogger._check_transcript_persistence`,
``chartreux/core/session/session_logger.py:1123-1149``), so a transcript that a
real save would refuse can never enter a measurement run. Nothing here touches
production code, and no clock-dependent or network-dependent content enters the
transcript: every content byte derives from the committed seed.

"200k-equivalent" uses the repository's established token proxy
(``tests/perf/_synthetic.py``): one token is estimated as four content
characters, so the fixture carries exactly 800,000 content characters.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import hashlib
import random
from typing import Any

from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.session.session_logger import SessionLogger

FIXTURE_SCHEMA = "chartreux.runtime_fixture.v1"
FIXTURE_ID = "long_history_200k"
FIXTURE_SEED = 20261107
FIXTURE_APPROX_TOKENS = 200_000
FIXTURE_CHARS_PER_TOKEN = 4
FIXTURE_TURNS = 400
FIXTURE_TURN_WIDTH = 5
FIXTURE_MESSAGE_CHARS = 400
FIXTURE_MESSAGE_COUNT = FIXTURE_TURNS * FIXTURE_TURN_WIDTH
FIXTURE_CONTENT_CHARS = FIXTURE_MESSAGE_COUNT * FIXTURE_MESSAGE_CHARS
FIXTURE_RESPONSE_CHARS = 400

# Committed fixture lineage: the digest below covers this seed, the geometry and
# the response shape together with the transcript itself.
FIXTURE_HASH = "25d08b939a0329f8af0276686ccc468880b03d9ce5079832822391d80c9b9d7d"

_CONTENT_STREAM_CHARS = 8_192
_CONTENT_STRIDE = 977
_RESPONSE_STRIDE = 613
_CONTENT_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_MESSAGE_PREFIX = "fixture-{index:06d} "
_RESPONSE_PREFIX = "response-{turn:06d} "
_TOOL_NAME = "fixture_tool"
_TOOL_ARGUMENTS = "{}"
_TURN_SHAPE = ("user", "assistant_call", "user", "tool_result", "assistant")
_RECORD_MARKER = b"\x1e"
_FIELD_SEPARATOR = "\x1f"


@lru_cache(maxsize=1)
def _content_stream() -> str:
    """Return the seeded content stream, doubled so any window fits in one slice."""
    rng = random.Random(FIXTURE_SEED)
    width = len(_CONTENT_ALPHABET)
    single = "".join(
        _CONTENT_ALPHABET[int(rng.random() * width)]
        for _ in range(_CONTENT_STREAM_CHARS)
    )
    return single + single


def _seeded_text(prefix: str, offset: int, char_count: int) -> str:
    """Build exactly ``char_count`` characters: the prefix plus a seeded window."""
    stream = _content_stream()
    start = offset % _CONTENT_STREAM_CHARS
    return (prefix + stream[start : start + char_count])[:char_count]


def _call_id(turn: int) -> str:
    return f"fixture-call-{turn:06d}"


def _tool_call(call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name=_TOOL_NAME, arguments=_TOOL_ARGUMENTS),
    )


def _turn_message(index: int, content: str) -> LLMMessage:
    """Shape one history message by its committed position inside a turn."""
    shape = _TURN_SHAPE[index % FIXTURE_TURN_WIDTH]
    call_id = _call_id(index // FIXTURE_TURN_WIDTH)
    if shape == "assistant_call":
        return LLMMessage(
            role=Role.assistant, content=content, tool_calls=[_tool_call(call_id)]
        )
    if shape == "tool_result":
        return LLMMessage(role=Role.tool, content=content, tool_call_id=call_id)
    return LLMMessage(
        role=Role.user if shape == "user" else Role.assistant, content=content
    )


def build_fixture() -> list[LLMMessage]:
    """Build the committed 200k-equivalent history, byte-identically every call."""
    messages: list[LLMMessage] = []
    for index in range(FIXTURE_MESSAGE_COUNT):
        content = _seeded_text(
            _MESSAGE_PREFIX.format(index=index),
            index * _CONTENT_STRIDE,
            FIXTURE_MESSAGE_CHARS,
        )
        messages.append(_turn_message(index, content))
    return messages


def response_content(turn: int) -> str:
    """Return the committed deterministic assistant text of one added turn."""
    if turn < 0:
        raise ValueError("turn must be non-negative")
    return _seeded_text(
        _RESPONSE_PREFIX.format(turn=turn),
        turn * _RESPONSE_STRIDE,
        FIXTURE_RESPONSE_CHARS,
    )


def response_message(turn: int) -> LLMMessage:
    """Return the committed deterministic assistant response of one added turn."""
    return LLMMessage(role=Role.assistant, content=response_content(turn))


def fixture_spec() -> dict[str, Any]:
    """Return the committed fixture lineage a run record carries.

    These are the values eligibility compares against the cell's committed
    baseline (fixture hash equal, declared dimensions equal).
    """
    return {
        "fixture_schema": FIXTURE_SCHEMA,
        "fixture_id": FIXTURE_ID,
        "fixture_hash": FIXTURE_HASH,
        "seed": FIXTURE_SEED,
        "approx_tokens": FIXTURE_APPROX_TOKENS,
        "chars_per_token": FIXTURE_CHARS_PER_TOKEN,
        "content_chars": FIXTURE_CONTENT_CHARS,
        "message_count": FIXTURE_MESSAGE_COUNT,
        "message_chars": FIXTURE_MESSAGE_CHARS,
        "turn_count": FIXTURE_TURNS,
        "turn_width": FIXTURE_TURN_WIDTH,
        "turn_shape": list(_TURN_SHAPE),
        "tool_name": _TOOL_NAME,
        "response_chars": FIXTURE_RESPONSE_CHARS,
    }


def _spec_record() -> str:
    """Render the committed specification as one order-fixed record."""
    return "|".join((
        FIXTURE_SCHEMA,
        FIXTURE_ID,
        f"seed={FIXTURE_SEED}",
        f"tokens={FIXTURE_APPROX_TOKENS}",
        f"chars_per_token={FIXTURE_CHARS_PER_TOKEN}",
        f"turns={FIXTURE_TURNS}",
        f"turn_width={FIXTURE_TURN_WIDTH}",
        f"message_chars={FIXTURE_MESSAGE_CHARS}",
        f"turn_shape={','.join(_TURN_SHAPE)}",
        f"stream={_CONTENT_STREAM_CHARS}",
        f"stream_stride={_CONTENT_STRIDE}",
        f"response_stride={_RESPONSE_STRIDE}",
        f"alphabet={_CONTENT_ALPHABET}",
        f"message_prefix={_MESSAGE_PREFIX}",
        f"response_prefix={_RESPONSE_PREFIX}",
        f"tool={_TOOL_NAME}:{_TOOL_ARGUMENTS}",
        f"response_chars={FIXTURE_RESPONSE_CHARS}",
    ))


def _message_record(message: LLMMessage) -> str:
    fields = [message.role.value, message.content or "", message.tool_call_id or ""]
    for call in message.tool_calls or ():
        fields.append(
            f"{call.id}|{call.index}|{call.function.name}|{call.function.arguments}"
        )
    return _FIELD_SEPARATOR.join(fields)


def fixture_digest(messages: Sequence[LLMMessage]) -> str:
    """Return the sha256 digest of the committed specification plus ``messages``."""
    digest = hashlib.sha256()
    digest.update(_spec_record().encode("utf-8"))
    for message in messages:
        digest.update(_RECORD_MARKER)
        digest.update(_message_record(message).encode("utf-8"))
    return digest.hexdigest()


def validate_fixture(messages: Sequence[LLMMessage]) -> None:
    """Validate a transcript on the landed P5 save boundary; raise on any violation.

    The system messages are filtered exactly as ``save_interaction`` filters them
    before the worker scan, so this is the production check rather than a
    reimplementation of it.
    """
    non_system = [message for message in messages if message.role is not Role.system]
    SessionLogger._check_transcript_persistence(non_system)
