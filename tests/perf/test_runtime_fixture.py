"""WP3a contract node: the committed 200k-equivalent transcript fixture.

The node proves the lineage the runtime cells will measure against: the
fixture's committed hash covers its seed, geometry and response shape as well
as the transcript itself, the transcript is reproducible, and it passes the
landed P5 save-boundary scan (``session_logger.py:1123-1149``) that a real save
runs. A negative control corrupts one committed property at a time so the
contract cannot pass vacuously: the digest must change when the specification
drifts, and the save scan must reject a duplicate tool result.
"""

from __future__ import annotations

import sys
from time import perf_counter

import pytest

from chartreux.core.agent_loop._transcript_integrity import TranscriptIntegrityError
from chartreux.core.llm_models import LLMMessage, Role
from tests.perf import _runtime_fixture
from tests.perf._runtime_fixture import (
    FIXTURE_APPROX_TOKENS,
    FIXTURE_CHARS_PER_TOKEN,
    FIXTURE_CONTENT_CHARS,
    FIXTURE_HASH,
    FIXTURE_MESSAGE_CHARS,
    FIXTURE_MESSAGE_COUNT,
    FIXTURE_RESPONSE_CHARS,
    FIXTURE_SEED,
    FIXTURE_TURN_WIDTH,
    FIXTURE_TURNS,
    build_fixture,
    fixture_digest,
    fixture_spec,
    response_content,
    response_message,
    validate_fixture,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux", reason="the runtime performance cells are Linux-only"
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_FIXTURE_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

_FIRST_CALL_ID = "fixture-call-000000"


def _assert_turn_shape(messages: list[LLMMessage]) -> None:
    """Assert every history turn carries the committed five-message shape."""
    for turn in range(FIXTURE_TURNS):
        start = turn * FIXTURE_TURN_WIDTH
        turn_messages = messages[start : start + FIXTURE_TURN_WIDTH]
        user, assistant_call, trailing_user, tool_result, closing = turn_messages
        assert user.role is Role.user
        assert assistant_call.role is Role.assistant
        call = (assistant_call.tool_calls or [])[0]
        assert call.id == f"fixture-call-{turn:06d}"
        assert call.index == 0
        assert call.function.arguments == "{}"
        assert trailing_user.role is Role.user
        assert tool_result.role is Role.tool
        assert tool_result.tool_call_id == call.id
        assert closing.role is Role.assistant


@pytest.mark.timeout(_FIXTURE_TIMEOUT_S)
def test_fixture_integrity_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture is 200k-equivalent, reproducible, hash-pinned and save-clean."""
    started = perf_counter()
    messages = build_fixture()

    spec = fixture_spec()
    assert spec["fixture_hash"] == FIXTURE_HASH
    assert spec["seed"] == FIXTURE_SEED
    assert spec["message_count"] == FIXTURE_MESSAGE_COUNT
    assert spec["message_chars"] == FIXTURE_MESSAGE_CHARS
    assert FIXTURE_CONTENT_CHARS == FIXTURE_APPROX_TOKENS * FIXTURE_CHARS_PER_TOKEN
    assert FIXTURE_MESSAGE_COUNT == FIXTURE_TURNS * FIXTURE_TURN_WIDTH

    assert len(messages) == FIXTURE_MESSAGE_COUNT
    assert sum(len(message.content or "") for message in messages) == (
        FIXTURE_CONTENT_CHARS
    )
    assert {len(message.content or "") for message in messages} == {
        FIXTURE_MESSAGE_CHARS
    }
    _assert_turn_shape(messages)

    # The committed hash is eligibility's drift rule; it covers the transcript
    # and, as the next block proves, the specification that generated it.
    assert fixture_digest(messages) == FIXTURE_HASH
    assert fixture_digest(build_fixture()) == FIXTURE_HASH

    monkeypatch.setattr(_runtime_fixture, "FIXTURE_SEED", FIXTURE_SEED + 1)
    assert fixture_digest(messages) != FIXTURE_HASH
    monkeypatch.undo()

    # Integrity runs on the landed P5 save boundary, never a reimplementation.
    validate_fixture(messages)

    corrupted = [
        *messages,
        LLMMessage(role=Role.tool, content="duplicate", tool_call_id=_FIRST_CALL_ID),
    ]
    with pytest.raises(TranscriptIntegrityError):
        validate_fixture(corrupted)
    assert fixture_digest(corrupted) != FIXTURE_HASH

    # The committed response shape is deterministic and fixed-size.
    response = response_message(0)
    content = response.content or ""
    assert response.role is Role.assistant
    assert content == response_content(0)
    assert len(content) == FIXTURE_RESPONSE_CHARS
    assert content.startswith("response-000000 ")
    assert response_content(0) != response_content(1)

    elapsed = perf_counter() - started
    assert elapsed <= _CONTRACT_WINDOW_S, (
        f"fixture contract took {elapsed:.3f}s, window {_CONTRACT_WINDOW_S}s"
    )
