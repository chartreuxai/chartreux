"""PI1/T — transcript message-order integrity (plan state.md, package T).

Covers the six seams of the transcript message-order integrity repair and
its acceptance items: the shared validator/repair
(``chartreux/core/agent_loop/_transcript_integrity.py``), the S2 amplifier
fix (``repair_missing_responses``), the S1 append/save boundary, the S3
request-build admission (``messages_for_backend``), the S4 self-healing load
(``SessionLoader.load_session`` + the write-owning resume install), the S5
anchored steering reconciliation (``TranscriptRecorder``), and the S6 root
admission pre-check (``RootRecoveryGate`` + ``TurnController.start``).

FIXTURES ARE SYNTHETIC: every transcript shape here is derived from the
documented PI1 pattern (a synthetic ``TOOL_NO_RESPONSE`` placeholder with no
structured ``tool_result``, a steered user message, and a genuine structured
result for the same ``tool_call_id``). A dispatch-time read-only re-scan of
``~/.chartreux/logs/session/*/messages.jsonl`` found 20 poison pairs across
6 sessions; NO raw session content is committed or reproduced here —
transcripts contain user data.

Steering responsiveness (branch 2 pinned, branch 1 FORBIDDEN) is pinned by
the existing green suites this battery keeps green:
``tests/app_server/test_wait_steering.py::test_delayed_skill_finishes_before_wait_cancellation``
(``manager.released == []`` while a skill expansion is in flight — the
expansion-before-wait-cancellation ordering) and
``tests/agent_loop/test_agent_wait_steering.py`` (completion/interrupt races,
unrelated running tools surviving steering). This file adds the PI1 replay
itself: the settled result lands BEFORE the steered user message.

All transports are controlled fakes — an unmocked provider call cannot occur
(fail-closed egress).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from chartreux.app_server._projector import ProjectedUpdate
from chartreux.app_server._root_recovery import (
    RootRecoveryGate,
    RootRecoverySessionView,
    _blocked_message,
)
from chartreux.app_server._runtime import _install_transcript_repair
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server._turns import TurnConflictError
from chartreux.app_server._utils import public_error
from chartreux.app_server.models import (
    PublicError,
    RootRecoveryNoticeDetail,
    TextContentBlock,
    TurnErrorCode,
)
from chartreux.app_server.protocol import (
    RootRecoveryAdmitParams,
    RootRecoveryScheduleParams,
    TurnStartParams,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._transcript_integrity import (
    TRANSCRIPT_INTEGRITY_CATEGORY,
    TRANSCRIPT_INTEGRITY_METADATA_KEY,
    TranscriptIntegrityError,
    TranscriptIntegrityReport,
    TranscriptIntegrityViolation,
    ViolationKind,
    _live_tail_region_identities,
    check_tool_message_admission,
    dedupe_shadowed_placeholders,
    find_duplicate_tool_results,
    find_persistence_violations,
    has_structured_tool_result,
    pop_report_from_metadata,
    repair_transcript,
    report_from_metadata,
    report_to_metadata,
    validate_live_prefix,
    validate_provider_ready,
    validate_settled_history,
    validate_transcript,
)
from chartreux.core.agent_loop._transcript_recorder import TranscriptRecorder
from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    LLMGateway,
    _map_call_error,
    messages_for_backend,
)
from chartreux.core.compaction.context import (
    is_synthetic_tool_placeholder,
    reorder_for_tool_adjacency,
    select_model_context,
)
from chartreux.core.config import ProviderConfig, SessionLoggingConfig
from chartreux.core.config.models import (
    ModelConfig,
    RecoveryEligibleClass,
    RootRecoveryConfig,
)
from chartreux.core.events import UserInputRequestEvent
from chartreux.core.llm.backend.generic import OpenAIAdapter
from chartreux.core.llm.exceptions import BackendError
from chartreux.core.llm.failures import (
    _FAILOVER_CATEGORIES,
    _PROTECTED_CATEGORIES,
    _RETRY_CATEGORIES,
    FailureCategory,
    classify_failure,
)
from chartreux.core.llm.format import APIToolFormatHandler
from chartreux.core.llm_models import (
    CONTEXT_INJECTION_ORIGIN_KEY,
    CONTEXT_INJECTION_TOOL_PAIR_V1,
    FunctionCall,
    LLMMessage,
    PersistedToolResult,
    Role,
    ToolCall,
)
from chartreux.core.message_list import MessageList
from chartreux.core.model_catalog.availability import AvailabilitySnapshot
from chartreux.core.recovery_models import FailureClass
from chartreux.core.session.session_loader import (
    MESSAGES_FILENAME,
    METADATA_FILENAME,
    SessionLoader,
)
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import AgentStats
from chartreux.core.skills.models import ParsedSkillCommand, SkillInfo
from chartreux.core.subagents import SubagentRunnerPort
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.secret_redaction import ScrubPolicy
from chartreux.core.utils.tags import CancellationReason, get_user_cancellation_message
from chartreux.questions import UserAnswer, UserQuestionResult
from tests.agent_loop.test_agent_steer_tool_call import (
    _assert_tool_calls_immediately_paired,
)
from tests.agent_loop.test_agent_wait_steering import WaitManager, make_loop, wait_call
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.recovery.conftest import FakeClock
from tests.recovery.test_root_recovery import _failure, _payload_summary, _timeout_error
from tests.stubs.fake_backend import FakeBackend

_STEER_TEXT = "sorry i meant the python runtime inside it"
_PLACEHOLDER_TEXT = str(
    get_user_cancellation_message(CancellationReason.TOOL_NO_RESPONSE)
)


# ---------------------------------------------------------------------------
# Synthetic transcript builders (derived shapes, never raw session content)
# ---------------------------------------------------------------------------


def _user(text: str, *, injected: bool = False) -> LLMMessage:
    return LLMMessage(role=Role.user, content=text, injected=injected)


def _tool_call(call_id: str, name: str = "read_file", index: int = 0) -> ToolCall:
    return ToolCall(
        id=call_id, index=index, function=FunctionCall(name=name, arguments="{}")
    )


def _assistant(*call_ids: str, content: str = "Working") -> LLMMessage:
    return LLMMessage(
        role=Role.assistant,
        content=content,
        tool_calls=(
            [_tool_call(call_id, index=index) for index, call_id in enumerate(call_ids)]
            or None
        ),
    )


def _genuine_result(
    call_id: str, text: str = "file contents", *, structured: bool = True
) -> LLMMessage:
    return LLMMessage(
        role=Role.tool,
        tool_call_id=call_id,
        name="read_file",
        content=text,
        tool_result=(
            PersistedToolResult(output={"text": text}) if structured else None
        ),
    )


def _placeholder(call_id: str, name: str = "read_file") -> LLMMessage:
    """The synthetic TOOL_NO_RESPONSE backfill placeholder (content + no
    structured tool_result — never the cancellation tag alone).
    """
    return LLMMessage(
        role=Role.tool, tool_call_id=call_id, name=name, content=_PLACEHOLDER_TEXT
    )


def _poison_transcript(call_id: str) -> list[LLMMessage]:
    """The PI1 four-message shape: [assistant, placeholder, user, genuine]."""
    return [
        _assistant(call_id),
        _placeholder(call_id),
        _user("steered mid-turn"),
        _genuine_result(call_id),
    ]


def _model() -> ModelConfig:
    return ModelConfig(name="test-model", provider="test", alias="test")


def _completion_inputs() -> CompletionInputs:
    return CompletionInputs(
        model=_model(),
        provider_name="test",
        emits_finish_reason=True,
        messages=(),
        tools=None,
        tool_choice=None,
        extra_headers={},
        metadata={},
        max_tokens=None,
    )


def _recorder(messages: MessageList) -> tuple[TranscriptRecorder, list[bool]]:
    """A directly-constructed recorder over ``messages`` (stub collaborators)."""
    invalidated: list[bool] = []
    recorder = TranscriptRecorder(
        messages=messages,
        scrub_policy_getter=lambda: ScrubPolicy(),
        format_handler_getter=lambda: APIToolFormatHandler(),
        set_execution_message=lambda message: None,
        note_prose=lambda message: None,
        invalidate_cursor=lambda: invalidated.append(True),
    )
    return recorder, invalidated


def _assert_no_user_tool_adjacency(messages: Sequence[LLMMessage]) -> None:
    """No [user, tool] adjacency — the provider 400 ``message_order`` shape."""
    for index, message in enumerate(messages):
        if message.role == Role.user and index + 1 < len(messages):
            following = messages[index + 1]
            assert following.role != Role.tool, (
                f"a tool result follows a user message at #{index + 1}: "
                f"the order still carries the 400 shape ({following.content!r})"
            )


def _write_session(
    session_dir: Path, messages: Sequence[LLMMessage], *, session_id: str = "pi1"
) -> Path:
    """Write one synthetic session directory (messages.jsonl + meta.json).

    The session-record writer is ``SessionLogger._message_record`` (round-5
    Y): the explicit session serializer, so a marked injector pair's
    provenance key rides the record exactly as a real save persists it —
    otherwise this writer would unintentionally manufacture a pre-marker
    (old-format) session.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        json.dumps(SessionLogger._message_record(message), ensure_ascii=False) + "\n"
        for message in messages
    )
    (session_dir / MESSAGES_FILENAME).write_text(lines, encoding="utf-8")
    (session_dir / METADATA_FILENAME).write_text(
        json.dumps({
            "session_id": session_id,
            "total_messages": len(messages),
            "environment": {"working_directory": "/repo"},
        }),
        encoding="utf-8",
    )
    return session_dir


def _session_logger(tmp_path: Path) -> SessionLogger:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    config = SessionLoggingConfig(
        save_dir=str(save_dir), session_prefix="test", enabled=True
    )
    return SessionLogger(config, "integrity-save")


def _tool_manager() -> ToolManager:
    manager = MagicMock(spec=ToolManager)
    manager.available_tools = {}
    return manager


def _integrity_error(
    kind: ViolationKind = "duplicate_tool_result", call_id: str = "call_pi1"
) -> TranscriptIntegrityError:
    return TranscriptIntegrityError([
        TranscriptIntegrityViolation(kind=kind, tool_call_id=call_id, detail="test")
    ])


# ---------------------------------------------------------------------------
# Acceptance 1 — the three validation contracts (LIVE-PREFIX, SETTLED-HISTORY,
# PROVIDER-READY) with positive pins for the legitimate live shapes
# ---------------------------------------------------------------------------


class TestLivePrefixContract:
    """LIVE-PREFIX: unresolved calls, staged batch results, and nested
    injected assistant/tool pairs are LEGITIMATE; unrepairable duplicates and
    unclaimed results are still reported.
    """

    def test_unresolved_live_tail_is_legitimate(self) -> None:
        messages = [_user("hi"), _assistant("c1")]
        assert validate_live_prefix(messages) == []
        assert validate_transcript(messages, live_tail_exempt=True) == []

    def test_staged_batch_results_are_legitimate(self) -> None:
        messages = [_user("hi"), _assistant("c1", "c2"), _genuine_result("c1")]
        assert validate_live_prefix(messages) == []
        assert validate_transcript(messages, live_tail_exempt=True) == []
        # The strict contract counts a staged batch result as satisfied.
        assert (
            validate_settled_history(messages, staged_call_ids=frozenset({"c2"})) == []
        )
        assert [violation.kind for violation in validate_settled_history(messages)] == [
            "unresolved_tool_call"
        ]

    def test_nested_injected_pairs_are_legitimate(self) -> None:
        # A skill/mention expansion: an injected assistant/tool pair nested in
        # the outer call's gap, with the steered user behind it.
        messages = [
            _user("hi"),
            _assistant("c1"),
            _assistant("c2", content=""),
            _genuine_result("c2"),
            _user("steered", injected=True),
            _genuine_result("c1"),
        ]
        assert validate_live_prefix(messages) == []
        assert validate_transcript(messages, live_tail_exempt=True) == []
        projection = messages_for_backend(messages, _model())
        assert [message.role for message in projection] == [
            Role.user,
            Role.assistant,
            Role.tool,
            Role.assistant,
            Role.tool,
            Role.user,
        ]
        _assert_no_user_tool_adjacency(projection)

    def test_multi_tool_batch_is_legitimate(self) -> None:
        messages = [
            _user("hi"),
            _assistant("c1", "c2"),
            _genuine_result("c1"),
            _genuine_result("c2"),
        ]
        assert validate_live_prefix(messages) == []
        assert validate_settled_history(messages) == []
        assert validate_provider_ready(messages) == []
        assert validate_transcript(messages, live_tail_exempt=True) == []

    def test_compaction_boundary_is_legitimate(self) -> None:
        boundary = LLMMessage(
            role=Role.user,
            content="compaction boundary",
            injected=True,
            context_boundary="compaction",
        )
        messages = [
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
            boundary,
            _user("again"),
            _assistant("c2"),
            _genuine_result("c2"),
        ]
        assert validate_transcript(messages, live_tail_exempt=True) == []
        projection = messages_for_backend(messages, _model())
        assert projection[0] is boundary
        assert validate_transcript(projection, live_tail_exempt=True) == []

    def test_question_callback_settlement_is_genuine(self) -> None:
        # An answered ask_user_question call: the settlement is a genuine
        # result, never mistaken for a synthetic placeholder.
        answer = LLMMessage(
            role=Role.tool,
            tool_call_id="c1",
            name="ask_user_question",
            content="Python",
            tool_result=PersistedToolResult(output={"answer": "Python"}),
        )
        messages = [_user("hi"), _assistant("c1"), answer]
        assert not is_synthetic_tool_placeholder(answer)
        assert validate_transcript(messages, live_tail_exempt=True) == []

    def test_agent_lifecycle_injection_is_legitimate(self) -> None:
        messages = [
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
            _assistant(content="lifecycle note"),
            _user("again"),
        ]
        assert validate_live_prefix(messages) == []
        assert validate_transcript(messages, live_tail_exempt=True) == []

    def test_unrepairable_duplicates_are_flagged(self) -> None:
        messages = [
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
            _genuine_result("c1"),
        ]
        violations = validate_live_prefix(messages)
        assert [violation.kind for violation in violations] == ["duplicate_tool_result"]

    def test_unclaimed_results_are_flagged(self) -> None:
        messages = [_user("hi"), _genuine_result("ghost")]
        violations = validate_live_prefix(messages)
        assert [violation.kind for violation in violations] == ["unclaimed_tool_result"]


class TestSettledHistoryContract:
    """SETTLED-HISTORY: strict completeness — every settled call answered by
    exactly one result (staged batch results satisfy), no unclaimed results.
    """

    def test_unresolved_call_is_flagged_strictly(self) -> None:
        messages = [_user("hi"), _assistant("c1")]
        violations = validate_settled_history(messages)
        assert [violation.kind for violation in violations] == ["unresolved_tool_call"]
        assert violations[0].tool_call_id == "c1"

    def test_duplicates_and_orphans_are_flagged(self) -> None:
        messages = [
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
            _genuine_result("c1"),
            _genuine_result("ghost"),
        ]
        kinds = [violation.kind for violation in validate_settled_history(messages)]
        assert kinds == ["duplicate_tool_result", "unclaimed_tool_result"]

    def test_call_id_reuse_across_turns_is_legitimate(self) -> None:
        # Tool-call ids are reused across turns; per-originating-assistant-call
        # scoping keeps both answered calls legitimate.
        messages = [
            _user("first"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("second"),
            _assistant("c1"),
            _genuine_result("c1"),
        ]
        assert validate_settled_history(messages) == []
        assert validate_transcript(messages, live_tail_exempt=True) == []
        assert find_duplicate_tool_results(messages) == []


class TestProviderReadyContract:
    """PROVIDER-READY: the order request build produces is provider-safe."""

    def test_steered_user_order_is_repaired(self) -> None:
        messages = [
            _user("hi"),
            _assistant("c1"),
            _user("steered"),
            _genuine_result("c1"),
        ]
        assert validate_provider_ready(messages) == []
        reordered = reorder_for_tool_adjacency(messages)
        assert [message.role for message in reordered] == [
            Role.user,
            Role.assistant,
            Role.tool,
            Role.user,
        ]

    def test_poison_shape_dedupes_and_reorders(self) -> None:
        poison = _poison_transcript("c1")
        assert validate_provider_ready(poison) == []
        deduped = dedupe_shadowed_placeholders(poison)
        assert [message.role for message in deduped] == [
            Role.assistant,
            Role.user,
            Role.tool,
        ]
        assert not any(is_synthetic_tool_placeholder(m) for m in deduped)
        reordered = reorder_for_tool_adjacency(poison)
        assert [message.role for message in reordered] == [
            Role.assistant,
            Role.tool,
            Role.user,
        ]
        _assert_no_user_tool_adjacency(reordered)

    def test_result_before_its_call_is_unrepairable(self) -> None:
        messages = [_genuine_result("c1"), _assistant("c1")]
        violations = validate_provider_ready(messages)
        # The tool result has no open owning call (misordered) AND the
        # trailing assistant tool call is left unanswered although the
        # transcript records a genuine result for its id (round-1 S1(c)):
        # the projection is doubly not provider-safe.
        assert [violation.kind for violation in violations] == [
            "misordered_tool_result",
            "unresolved_tool_call",
        ]

    def test_ambiguous_duplicate_is_reported_never_guessed(self) -> None:
        messages = [
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
            _genuine_result("c1"),
        ]
        violations = validate_transcript(messages, live_tail_exempt=True)
        assert [violation.kind for violation in violations] == ["duplicate_tool_result"]


# ---------------------------------------------------------------------------
# Acceptance 1 (S2) — the poison amplifier: repair_missing_responses
# ---------------------------------------------------------------------------


class TestS2AmplifierFix:
    """S2 (landed FIRST): the backfill's lookahead is bounded to the logical
    block, so a genuine result behind a steered user message is SEEN and no
    duplicate placeholder is inserted; the designed dangling-call repair
    (hard crash mid-tool, no genuine result anywhere) still inserts.
    """

    # NOTE: every fixture leads with a non-assistant message because the
    # backfill scan starts at index 1 (the system prompt owns index 0 in a
    # real transcript) — an assistant at index 0 would never be scanned.

    def test_genuine_result_behind_steered_user_gets_no_duplicate(self) -> None:
        messages = MessageList([
            _user("hi"),
            _assistant("c1"),
            _user("steered"),
            _genuine_result("c1"),
        ])
        recorder, _ = _recorder(messages)
        recorder.repair_missing_responses()
        assert [message.role for message in messages] == [
            Role.user,
            Role.assistant,
            Role.user,
            Role.tool,
        ]
        assert not any(is_synthetic_tool_placeholder(m) for m in messages)

    def test_dangling_call_still_gets_its_placeholder(self) -> None:
        # Hard crash mid-tool: no genuine result anywhere in the block.
        messages = MessageList([_user("hi"), _assistant("c1"), _user("later")])
        recorder, invalidated = _recorder(messages)
        recorder.repair_missing_responses()
        assert [message.role for message in messages] == [
            Role.user,
            Role.assistant,
            Role.tool,
            Role.user,
        ]
        placeholder = messages[2]
        assert placeholder.tool_call_id == "c1"
        assert placeholder.name == "read_file"
        assert placeholder.content == _PLACEHOLDER_TEXT
        assert is_synthetic_tool_placeholder(placeholder)
        assert invalidated == [True]  # interior insertion invalidates the cursor

    def test_placeholder_at_list_end_does_not_invalidate_cursor(self) -> None:
        messages = MessageList([_user("hi"), _assistant("c1")])
        recorder, invalidated = _recorder(messages)
        recorder.repair_missing_responses()
        assert [message.role for message in messages] == [
            Role.user,
            Role.assistant,
            Role.tool,
        ]
        assert is_synthetic_tool_placeholder(messages[2])
        assert invalidated == []

    def test_multi_tool_batch_backfills_the_missing_call(self) -> None:
        messages = MessageList([
            _user("hi"),
            _assistant("c1", "c2"),
            _genuine_result("c2"),
        ])
        recorder, _ = _recorder(messages)
        recorder.repair_missing_responses()
        assert [message.role for message in messages] == [
            Role.user,
            Role.assistant,
            Role.tool,
            Role.tool,
        ]
        assert messages[2].tool_call_id == "c2"
        assert messages[3].tool_call_id == "c1"
        assert messages[3].content == _PLACEHOLDER_TEXT

    def test_poison_shape_can_no_longer_be_created(self) -> None:
        # Running the backfill over the already-poisoned shape must not add a
        # third result, and over the raw steered order it must never add a
        # placeholder at all — repeated runs are idempotent.
        poison = MessageList([_user("hi"), *_poison_transcript("c1")])
        recorder, _ = _recorder(poison)
        recorder.repair_missing_responses()
        recorder.repair_missing_responses()
        tool_results = [m for m in poison if m.role == Role.tool]
        assert len(tool_results) == 2
        assert sum(is_synthetic_tool_placeholder(m) for m in tool_results) == 1

        raw = MessageList([
            _user("hi"),
            _assistant("c1"),
            _user("steered"),
            _genuine_result("c1"),
        ])
        recorder, _ = _recorder(raw)
        recorder.repair_missing_responses()
        recorder.repair_missing_responses()
        assert not any(is_synthetic_tool_placeholder(m) for m in raw)

    def test_nested_injection_genuine_result_blocks_a_placeholder(self) -> None:
        # The block scan stops at the nested assistant call; the per-owner
        # defense still sees the genuine result behind the nested pair.
        messages = MessageList([
            _user("hi"),
            _assistant("c1"),
            _assistant("c2", content=""),
            _genuine_result("c2"),
            _user("steered"),
            _genuine_result("c1"),
        ])
        recorder, _ = _recorder(messages)
        recorder.repair_missing_responses()
        assert not any(is_synthetic_tool_placeholder(m) for m in messages)
        assert len([m for m in messages if m.role == Role.tool]) == 2

    def test_reused_call_id_is_scoped_per_originating_call(self) -> None:
        # The earlier turn's dangling call still gets its placeholder; the
        # later turn's genuine result for the reused id is untouched.
        messages = MessageList([
            _user("first"),
            _assistant("c1"),
            _user("between turns"),
            _assistant("c1"),
            _genuine_result("c1"),
        ])
        recorder, _ = _recorder(messages)
        recorder.repair_missing_responses()
        assert [message.role for message in messages] == [
            Role.user,
            Role.assistant,
            Role.tool,
            Role.user,
            Role.assistant,
            Role.tool,
        ]
        assert is_synthetic_tool_placeholder(messages[2])
        assert messages[5].tool_result is not None
        assert validate_transcript(list(messages), live_tail_exempt=True) == []


# ---------------------------------------------------------------------------
# Acceptance 2 (S1) — the append boundary and the save boundary
# ---------------------------------------------------------------------------


class TestS1AppendBoundary:
    """S1: no append/insert path may create a duplicate tool result (scoped
    per originating assistant call) or an unrepairable [user, tool]
    adjacency; violations fail closed; genuine-over-placeholder is the
    designed replacement, NOT a violation.
    """

    def test_genuine_over_placeholder_is_the_designed_replacement(self) -> None:
        messages = [_assistant("c1"), _placeholder("c1")]
        check_tool_message_admission(messages, len(messages), _genuine_result("c1"))

    def test_second_genuine_result_fails_closed(self) -> None:
        messages = [_assistant("c1"), _genuine_result("c1")]
        with pytest.raises(TranscriptIntegrityError) as excinfo:
            check_tool_message_admission(messages, len(messages), _genuine_result("c1"))
        assert excinfo.value.violations[0].kind == "duplicate_tool_result"

    def test_placeholder_over_genuine_fails_closed(self) -> None:
        messages = [_assistant("c1"), _genuine_result("c1")]
        with pytest.raises(TranscriptIntegrityError):
            check_tool_message_admission(messages, len(messages), _placeholder("c1"))

    def test_duplicate_placeholder_fails_closed(self) -> None:
        messages = [_assistant("c1"), _placeholder("c1")]
        with pytest.raises(TranscriptIntegrityError):
            check_tool_message_admission(messages, len(messages), _placeholder("c1"))

    def test_orphan_append_behind_a_user_message_fails_closed(self) -> None:
        # No owning assistant call AND directly behind a user message: the
        # unplaceable result would serialize as the unrepairable
        # [user, tool] 400 shape — fail closed.
        messages = [_user("hi")]
        with pytest.raises(TranscriptIntegrityError) as excinfo:
            check_tool_message_admission(
                messages, len(messages), _genuine_result("ghost")
            )
        assert excinfo.value.violations[0].kind == "misordered_tool_result"

    def test_orphan_append_without_an_owner_is_admitted_and_policed(self) -> None:
        # No owning assistant call is not a duplicate (no owner scope): the
        # append boundary admits it (the direct-execution seam records
        # results for calls with no recorded assistant message), and the
        # validator seams police the resulting transcript.
        messages = [_assistant("c1")]
        check_tool_message_admission(messages, len(messages), _genuine_result("ghost"))
        transcript = [*messages, _genuine_result("ghost")]
        violations = validate_transcript(transcript, live_tail_exempt=True)
        # Unclaimed (no owner scope) AND unplaceable in any provider-safe
        # order (no open owning call) — both seams fail closed on it.
        assert [v.kind for v in violations] == [
            "unclaimed_tool_result",
            "misordered_tool_result",
        ]
        with pytest.raises(TranscriptIntegrityError):
            messages_for_backend(transcript, _model())

    def test_no_call_id_boundary_marker_is_admitted(self) -> None:
        messages = [_assistant("c1"), _genuine_result("c1")]
        marker = LLMMessage(role=Role.tool, content="boundary marker")
        check_tool_message_admission(messages, len(messages), marker)

    def test_repairable_user_tool_adjacency_is_admitted(self) -> None:
        # [assistant, user] + the genuine result: request build repairs the
        # order, so the append boundary admits it.
        messages = [_assistant("c1"), _user("steered")]
        check_tool_message_admission(messages, len(messages), _genuine_result("c1"))

    def test_cross_turn_call_id_reuse_is_admitted(self) -> None:
        messages = [
            _assistant("c1"),
            _genuine_result("c1"),
            _user("second"),
            _assistant("c1"),
        ]
        check_tool_message_admission(messages, len(messages), _genuine_result("c1"))


class TestS1SaveBoundary:
    """S1 save boundary: nothing that would (re)poison the log is ever
    silently persisted — the check fails closed with the typed diagnostic.
    """

    def test_persistence_boundary_rejects_every_duplicate(self) -> None:
        with pytest.raises(TranscriptIntegrityError):
            SessionLogger._check_transcript_persistence([
                _user("hi"),
                _assistant("c1"),
                _genuine_result("c1"),
                _genuine_result("c1"),
            ])
        with pytest.raises(TranscriptIntegrityError):
            SessionLogger._check_transcript_persistence([
                _assistant("c1"),
                _genuine_result("c1"),
                _placeholder("c1"),
            ])
        with pytest.raises(TranscriptIntegrityError):
            SessionLogger._check_transcript_persistence([
                _assistant("c1"),
                _placeholder("c1"),
                _placeholder("c1"),
            ])
        with pytest.raises(TranscriptIntegrityError):
            SessionLogger._check_transcript_persistence([
                _user("hi"),
                _genuine_result("ghost"),
            ])

    def test_persistence_boundary_admits_repairable_shapes(self) -> None:
        # The raw steered order, the designed replacement, and cross-turn
        # call-id reuse are all repairable/legitimate — never re-poison.
        SessionLogger._check_transcript_persistence([
            _user("hi"),
            _assistant("c1"),
            _user("steered"),
            _genuine_result("c1"),
        ])
        SessionLogger._check_transcript_persistence([
            _assistant("c1"),
            _genuine_result("c1"),
            _user("steered"),
        ])
        SessionLogger._check_transcript_persistence([
            _assistant("c1"),
            _genuine_result("c1"),
            _user("second"),
            _assistant("c1"),
            _genuine_result("c1"),
        ])
        assert (
            find_persistence_violations([
                _user("hi"),
                _assistant("c1"),
                _user("steered"),
                _genuine_result("c1"),
            ])
            == []
        )

    @pytest.mark.asyncio
    async def test_save_interaction_never_persists_a_defect(
        self, tmp_path: Path
    ) -> None:
        logger = _session_logger(tmp_path)
        assert logger.session_dir is not None
        poisoned = [
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
            _genuine_result("c1"),
        ]
        with pytest.raises(TranscriptIntegrityError):
            await logger.save_interaction(
                poisoned, AgentStats(), build_test_vibe_config(), _tool_manager()
            )
        assert not (logger.session_dir / MESSAGES_FILENAME).exists()

    @pytest.mark.asyncio
    async def test_save_interaction_persists_the_designed_replacement(
        self, tmp_path: Path
    ) -> None:
        logger = _session_logger(tmp_path)
        assert logger.session_dir is not None
        replaced = [
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("steered"),
        ]
        await logger.save_interaction(
            replaced, AgentStats(), build_test_vibe_config(), _tool_manager()
        )
        messages_path = logger.session_dir / MESSAGES_FILENAME
        on_disk = messages_path.read_text(encoding="utf-8")
        records = [json.loads(line) for line in on_disk.splitlines()]
        tool_records = [
            record for record in records if record.get("tool_call_id") == "c1"
        ]
        assert len(tool_records) == 1
        assert tool_records[0]["tool_result"] is not None
        assert "no response available" not in on_disk


# ---------------------------------------------------------------------------
# Acceptance 3 (S3) — request-build admission (messages_for_backend)
# ---------------------------------------------------------------------------


class TestS3RequestBuildAdmission:
    """S3: the projection runs the shared validator — single-pass, dedupes
    (genuine kept, synthetic placeholder dropped), verifies the provider-safe
    order AFTER reordering, and raises the typed diagnostic BEFORE any
    provider call.
    """

    def test_ab_pin_poison_projection_carries_no_400_shape(self) -> None:
        """THE A/B PIN: [A, T1(placeholder), U, T2(genuine)] projects with NO
        [user, tool] adjacency, the placeholder dropped, the genuine kept.

        FAILS without the fix: the pre-fix reorder emptied its pending set at
        the placeholder, so the [user, tool] adjacency survived to the
        provider (the permanent 400).
        """
        projection = messages_for_backend(_poison_transcript("c1"), _model())
        _assert_no_user_tool_adjacency(projection)
        results = [m for m in projection if m.role == Role.tool]
        assert len(results) == 1
        assert results[0].tool_call_id == "c1"
        assert has_structured_tool_result(results[0])
        assert results[0].content == "file contents"
        assert not any(is_synthetic_tool_placeholder(m) for m in projection)
        assert [m.role for m in projection] == [Role.assistant, Role.tool, Role.user]

    def test_steered_order_is_repaired(self) -> None:
        messages = [
            _user("hi"),
            _assistant("c1"),
            _user("steered"),
            _genuine_result("c1"),
        ]
        projection = messages_for_backend(messages, _model())
        assert [m.role for m in projection] == [
            Role.user,
            Role.assistant,
            Role.tool,
            Role.user,
        ]

    def test_unrepairable_order_raises_the_typed_diagnostic(self) -> None:
        ambiguous = [
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
            _genuine_result("c1"),
        ]
        with pytest.raises(TranscriptIntegrityError) as excinfo:
            messages_for_backend(ambiguous, _model())
        assert excinfo.value.violations[0].kind == "duplicate_tool_result"
        with pytest.raises(TranscriptIntegrityError) as excinfo:
            messages_for_backend([_genuine_result("c1"), _assistant("c1")], _model())
        assert excinfo.value.violations[0].kind == "unclaimed_tool_result"

    def test_interrupted_and_skipped_settlements_are_never_dropped(self) -> None:
        interrupted = LLMMessage(
            role=Role.tool,
            tool_call_id="c1",
            name="read_file",
            content=str(
                get_user_cancellation_message(CancellationReason.TOOL_INTERRUPTED)
            ),
            tool_result=PersistedToolResult(output={"cancelled": True}, cancelled=True),
        )
        skipped = LLMMessage(
            role=Role.tool,
            tool_call_id="c2",
            name="read_file",
            content=str(
                get_user_cancellation_message(
                    CancellationReason.TOOL_SKIPPED, "read_worker"
                )
            ),
        )
        messages = [
            _user("hi"),
            _assistant("c1"),
            interrupted,
            _user("again"),
            _assistant("c2"),
            skipped,
        ]
        projection = messages_for_backend(messages, _model())
        results = [m for m in projection if m.role == Role.tool]
        assert [m.tool_call_id for m in results] == ["c1", "c2"]
        assert results[0].content == interrupted.content
        assert results[1].content == skipped.content
        assert results[0].tool_result is not None
        _assert_no_user_tool_adjacency(projection)

    @pytest.mark.asyncio
    async def test_request_build_rejection_precedes_any_provider_call(self) -> None:
        backend = FakeBackend([[mock_llm_chunk(content="ok")]])
        loop = build_test_agent_loop(backend=backend)
        try:
            for message in (
                _user("hi"),
                _assistant("c1"),
                _genuine_result("c1"),
                _genuine_result("c1"),
            ):
                loop.messages.append(message)
            raised: BaseException | None = None
            try:
                async for _event in loop.act("go"):
                    pass
            except Exception as exc:
                raised = exc
            assert raised is not None
            # ROUND-1 S2 (improved behavior): the gateway's error mapper
            # passes the typed LOCAL diagnostic through unchanged (the
            # ImagesNotSupportedError precedent at llm_gateway.py:907) —
            # the surfaced failure IS the typed diagnostic, never a
            # provider-shaped RuntimeError ("API error from {provider} ...")
            # mislabeling a local transcript defect as a provider failure.
            assert isinstance(raised, TranscriptIntegrityError), (
                f"the typed diagnostic was re-wrapped: {raised!r}"
            )
            assert "API error from" not in str(raised)
            assert backend.requests_messages == []
            # The failed request build never modified the transcript.
            assert find_duplicate_tool_results(list(loop.messages))
            # Acceptance 9 through the production wrapping: the wrapped
            # failure still classifies UNKNOWN — never retried, never
            # scheduled.
            info = classify_failure(raised)
            assert info.category is FailureCategory.UNKNOWN
            assert info.retry_eligible is False
        finally:
            await loop.aclose()


# ---------------------------------------------------------------------------
# Acceptance 4 (S4) — the self-healing load and the write-owning install
# ---------------------------------------------------------------------------


class TestS4LoadRepair:
    """S4: load_session validates + repairs IN MEMORY for every caller
    (never hard-fails, never writes); the durable migration lives only in
    the write-owning resume install and fails closed on backup/write
    failure.
    """

    def test_poisoned_file_heals_in_memory_without_writes(self, tmp_path: Path) -> None:
        session_dir = _write_session(
            tmp_path / "session_pi1", _poison_transcript("call_pi1")
        )
        messages_path = session_dir / MESSAGES_FILENAME
        original_bytes = messages_path.read_bytes()

        messages, metadata = SessionLoader.load_session(session_dir)

        # The d7339819-shaped file heals WITHOUT manual rewind: the
        # placeholder is dropped, the genuine result is kept, order verified.
        assert [message.role for message in messages] == [
            Role.assistant,
            Role.user,
            Role.tool,
        ]
        results = [m for m in messages if m.role == Role.tool]
        assert len(results) == 1
        assert results[0].tool_call_id == "call_pi1"
        assert results[0].tool_result is not None
        assert results[0].content == "file contents"
        assert not any(is_synthetic_tool_placeholder(m) for m in messages)
        assert validate_transcript(messages, live_tail_exempt=True) == []
        projection = messages_for_backend(messages, _model())
        _assert_no_user_tool_adjacency(projection)

        report = report_from_metadata(metadata[TRANSCRIPT_INTEGRITY_METADATA_KEY])
        assert report is not None
        assert report.repaired is True
        assert report.dropped_placeholder_call_ids == ("call_pi1",)
        assert report.violations == ()
        assert report.clean

        # Read-only access performs NO writes: the poisoned bytes are intact.
        assert messages_path.read_bytes() == original_bytes

    def test_repeated_load_is_deterministic(self, tmp_path: Path) -> None:
        session_dir = _write_session(
            tmp_path / "session_pi1", _poison_transcript("call_pi1")
        )
        first, first_metadata = SessionLoader.load_session(session_dir)
        second, second_metadata = SessionLoader.load_session(session_dir)
        assert [m.model_dump() for m in first] == [m.model_dump() for m in second]
        # The disk is still poisoned, so the stamp rides again (deterministic).
        assert TRANSCRIPT_INTEGRITY_METADATA_KEY in second_metadata
        assert report_from_metadata(
            second_metadata[TRANSCRIPT_INTEGRITY_METADATA_KEY]
        ) == report_from_metadata(first_metadata[TRANSCRIPT_INTEGRITY_METADATA_KEY])

    def test_resume_install_migrates_durably_and_is_idempotent(
        self, tmp_path: Path
    ) -> None:
        session_dir = _write_session(
            tmp_path / "session_pi1", _poison_transcript("call_pi1")
        )
        messages_path = session_dir / MESSAGES_FILENAME
        original_bytes = messages_path.read_bytes()

        messages, metadata = SessionLoader.load_session(session_dir)
        installed = _install_transcript_repair(session_dir, messages, metadata)
        assert installed is messages

        artifacts = list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        assert len(artifacts) == 1
        # The ORIGINAL transcript is preserved as an artifact before the
        # replacement (backup-success verified by read-back before the swap).
        assert artifacts[0].read_bytes() == original_bytes

        # The live path now carries the repaired transcript: a repeated load
        # is an idempotent no-op (no stamp, no second artifact).
        reloaded, reloaded_metadata = SessionLoader.load_session(session_dir)
        assert [m.role for m in reloaded] == [Role.assistant, Role.user, Role.tool]
        assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in reloaded_metadata
        again = _install_transcript_repair(session_dir, reloaded, reloaded_metadata)
        assert again is reloaded
        assert len(list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))) == 1

    def test_ambiguous_duplicate_fails_closed_at_install(self, tmp_path: Path) -> None:
        ambiguous = [
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
            _genuine_result("c1"),
        ]
        session_dir = _write_session(tmp_path / "session_ambiguous", ambiguous)
        messages_path = session_dir / MESSAGES_FILENAME
        original_bytes = messages_path.read_bytes()

        messages, metadata = SessionLoader.load_session(session_dir)
        # Ambiguous cases are never guessed: no repair in memory.
        assert [m.role for m in messages] == [
            Role.user,
            Role.assistant,
            Role.tool,
            Role.tool,
        ]
        report = report_from_metadata(metadata[TRANSCRIPT_INTEGRITY_METADATA_KEY])
        assert report is not None
        assert report.repaired is False
        assert [v.kind for v in report.violations] == ["duplicate_tool_result"]

        with pytest.raises(TranscriptIntegrityError):
            _install_transcript_repair(session_dir, messages, metadata)
        # Fail closed: no partial migration, no artifact, disk untouched.
        assert messages_path.read_bytes() == original_bytes
        assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))

    def test_failed_backup_write_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session_dir = _write_session(
            tmp_path / "session_pi1", _poison_transcript("call_pi1")
        )
        messages_path = session_dir / MESSAGES_FILENAME
        original_bytes = messages_path.read_bytes()
        messages, metadata = SessionLoader.load_session(session_dir)

        def failing_replace(target: Path, data: bytes) -> None:
            raise OSError("injected backup failure")

        monkeypatch.setattr(
            "chartreux.core.session.session_logger.durable_replace", failing_replace
        )
        with pytest.raises(TranscriptIntegrityError) as excinfo:
            _install_transcript_repair(session_dir, messages, metadata)
        assert excinfo.value.violations[0].kind == "transcript_repair_failed"
        assert messages_path.read_bytes() == original_bytes
        assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))

    def test_missing_live_transcript_fails_the_migration(self, tmp_path: Path) -> None:
        missing = tmp_path / MESSAGES_FILENAME
        with pytest.raises(TranscriptIntegrityError) as excinfo:
            SessionLogger.migrate_repaired_transcript(missing, [_user("hi")])
        assert excinfo.value.violations[0].kind == "transcript_repair_failed"
        assert not missing.exists()

    def test_clean_session_passes_through_without_writes(self, tmp_path: Path) -> None:
        clean = [_user("hi"), _assistant("c1"), _genuine_result("c1"), _user("again")]
        session_dir = _write_session(tmp_path / "session_clean", clean)
        messages_path = session_dir / MESSAGES_FILENAME
        original_bytes = messages_path.read_bytes()

        messages, metadata = SessionLoader.load_session(session_dir)
        assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
        assert _install_transcript_repair(session_dir, messages, metadata) is messages
        assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        assert messages_path.read_bytes() == original_bytes

    def test_load_repair_is_scoped_per_originating_assistant_call(
        self, tmp_path: Path
    ) -> None:
        # A reused call id: the later turn's shadowed placeholder is dropped;
        # the earlier turn's genuine result for the same id is kept.
        reused = [
            _user("first"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("second"),
            _assistant("c1"),
            _placeholder("c1"),
            _genuine_result("c1"),
        ]
        session_dir = _write_session(tmp_path / "session_reuse", reused)
        messages, metadata = SessionLoader.load_session(session_dir)
        assert [m.role for m in messages] == [
            Role.user,
            Role.assistant,
            Role.tool,
            Role.user,
            Role.assistant,
            Role.tool,
        ]
        results = [m for m in messages if m.role == Role.tool]
        assert len(results) == 2
        assert all(has_structured_tool_result(m) for m in results)
        report = report_from_metadata(metadata[TRANSCRIPT_INTEGRITY_METADATA_KEY])
        assert report is not None
        assert report.repaired is True
        assert report.dropped_placeholder_call_ids == ("c1",)
        assert validate_transcript(messages, live_tail_exempt=True) == []


# ---------------------------------------------------------------------------
# Acceptance 5 (S5) — anchored steering reconciliation (branch 2 pinned)
# ---------------------------------------------------------------------------


class TestS5RecorderPlacement:
    """S5 branch 2 (pinned): the recorder records the settled result at its
    CORRECT position — after the owning assistant message, BEFORE the steered
    user message, replacing a synthetic placeholder in place.
    """

    def test_result_is_anchored_before_the_steered_user(self) -> None:
        messages = MessageList([_assistant("c1")])
        messages.append(_user("steered"))
        recorder, invalidated = _recorder(messages)
        genuine = _genuine_result("c1")
        recorder.append_response("c1", genuine)
        assert [m.role for m in messages] == [Role.assistant, Role.tool, Role.user]
        assert messages[1] is genuine
        assert invalidated == [True]  # interior insertion invalidates the cursor

    def test_genuine_replaces_the_placeholder_in_place(self) -> None:
        messages = MessageList([_assistant("c1"), _placeholder("c1"), _user("steered")])
        recorder, invalidated = _recorder(messages)
        genuine = _genuine_result("c1")
        recorder.append_response("c1", genuine)
        assert len(messages) == 3
        assert messages[1] is genuine
        assert not is_synthetic_tool_placeholder(messages[1])
        assert messages[2].role == Role.user
        assert invalidated == []  # in-place replacement keeps the cursor proof

    def test_batch_stages_results_in_call_order(self) -> None:
        messages = MessageList([_assistant("c1", "c2")])
        recorder, _ = _recorder(messages)
        recorder.begin_batch()
        recorder.append_response("c2", _genuine_result("c2"))
        assert len(messages) == 1  # staged, not yet appended
        recorder.append_response("c1", _genuine_result("c1"))
        assert [m.tool_call_id for m in messages if m.role == Role.tool] == ["c1", "c2"]

    def test_orphan_append_appends_at_the_list_end(self) -> None:
        # No owning assistant call: no anchor exists, so the recorder appends
        # at the list end (the direct-execution seam's pre-existing shape);
        # the validator seams flag the resulting transcript.
        messages = MessageList([_assistant("c1")])
        recorder, _ = _recorder(messages)
        recorder.append_response("ghost", _genuine_result("ghost"))
        assert len(messages) == 2
        assert messages[1].tool_call_id == "ghost"
        violations = validate_transcript(list(messages), live_tail_exempt=True)
        # Unclaimed (no owner scope) AND unplaceable in any provider-safe
        # order (no open owning call) — the validator seams fail closed.
        assert [v.kind for v in violations] == [
            "unclaimed_tool_result",
            "misordered_tool_result",
        ]

    def test_second_genuine_append_fails_closed(self) -> None:
        messages = MessageList([_assistant("c1"), _genuine_result("c1")])
        recorder, _ = _recorder(messages)
        with pytest.raises(TranscriptIntegrityError):
            recorder.append_response("c1", _genuine_result("c1"))
        assert len(messages) == 2

    def test_append_after_the_owning_assistant_at_the_end(self) -> None:
        messages = MessageList([_user("hi"), _assistant("c1")])
        recorder, _ = _recorder(messages)
        recorder.append_response("c1", _genuine_result("c1"))
        assert [m.role for m in messages] == [Role.user, Role.assistant, Role.tool]


class TestS5LoopPins:
    """Loop-level S5 pins: the conversation loop never strands a steered
    user message, and the PI1 replay records the settled result before the
    steer.
    """

    @pytest.mark.asyncio
    async def test_unanswered_user_message_pins(self) -> None:
        loop = build_test_agent_loop()
        try:
            assert loop._has_unanswered_user_message() is False  # []
            loop.messages.append(_user("hi"))
            assert loop._has_unanswered_user_message() is True  # [U]
            loop.messages.append(_assistant("c1"))
            assert loop._has_unanswered_user_message() is False  # [U, A]
            loop.messages.append(_genuine_result("c1"))
            assert loop._has_unanswered_user_message() is False  # [U, A, T]
            loop.messages.append(_assistant(content="note"))
            assert loop._has_unanswered_user_message() is False  # [A, T, A]
            loop.messages.append(_user("again"))
            assert loop._has_unanswered_user_message() is True  # [A, T, U]
        finally:
            await loop.aclose()

    @pytest.mark.asyncio
    async def test_pi1_replay_steering_during_a_tool_call(self) -> None:
        """The PI1 replay: mid-turn steering while a tool call is in flight.

        The settled genuine result lands BEFORE the steered user message (the
        raw transcript carries no 400 shape), exactly one genuine result
        exists for the call, and every request that reached the backend is
        tool-paired. Pre-fix the raw order was [A, U, T] (the steer between
        the call and its result).
        """
        tool_call = ToolCall(
            id="call_1",
            index=0,
            function=FunctionCall(
                name="ask_user_question",
                arguments=(
                    '{"questions": [{"question": "Which runtime?", "options": '
                    '[{"label": "Python"}, {"label": "Node"}]}]}'
                ),
            ),
        )
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="Let me check your todos.", tool_calls=[tool_call]
                )
            ],
            [mock_llm_chunk(content="Done.")],
        ])
        loop = build_test_agent_loop(
            config=build_test_vibe_config(enabled_tools=["ask_user_question"]),
            backend=backend,
        )
        try:
            async for event in loop.act("please check the runtime"):
                if isinstance(event, UserInputRequestEvent):
                    await loop.inject_user_context(_STEER_TEXT, as_message=True)
                    loop.resolve_user_input_request(
                        event.request_id,
                        UserQuestionResult(
                            answers=[
                                UserAnswer(question="Which runtime?", answer="Python")
                            ]
                        ),
                    )
            messages = list(loop.messages)
            results = [
                m
                for m in messages
                if m.role == Role.tool and m.tool_call_id == "call_1"
            ]
            assert len(results) == 1
            assert not is_synthetic_tool_placeholder(results[0])
            tool_index = next(i for i, m in enumerate(messages) if m is results[0])
            steer_index = next(
                i
                for i, m in enumerate(messages)
                if m.role == Role.user and m.content == _STEER_TEXT
            )
            assert tool_index < steer_index
            _assert_no_user_tool_adjacency(messages)
            _assert_no_user_tool_adjacency(messages_for_backend(messages, _model()))
            assert backend.requests_messages, "no backend request was made"
            for request in backend.requests_messages:
                _assert_tool_calls_immediately_paired(request)
        finally:
            await loop.aclose()


# ---------------------------------------------------------------------------
# Acceptance 6 (S6) — the root admission pre-check
# ---------------------------------------------------------------------------


class _IntegrityGateHarness:
    """A directly-constructed gate with the S6 transcript-integrity
    collaborator (the gate unit tests' shape, plus the optional view).
    """

    def __init__(
        self, transcript: list[LLMMessage] | None, *, collaborator_raises: bool = False
    ) -> None:
        self.clock = FakeClock(1000.0)
        self.view = RootRecoverySessionView(
            session_id="session-1",
            active_turn_id=None,
            queued_work=False,
            open_callbacks=False,
            stop_requested=False,
            handoff_pending=False,
            lifecycle_reserved=False,
            closing=False,
        )
        self.snapshot = AvailabilitySnapshot(
            available=True,
            permanent=False,
            endpoint_blocked=False,
            probe_in_flight=False,
            next_check_delay=0.0,
        )
        self.config = RootRecoveryConfig(
            enabled=True,
            auto_retries_max=2,
            deadline_seconds=2400.0,
            cooldowns_seconds=[30.0, 120.0],
            availability_recheck_seconds=1.0,
            eligible_classes=list(RecoveryEligibleClass),
        )
        self.staged: list[tuple[RootRecoveryNoticeDetail, str]] = []
        self.transcript = transcript

        def read_transcript() -> Sequence[LLMMessage]:
            if collaborator_raises:
                raise RuntimeError("transcript read failed")
            assert self.transcript is not None
            return self.transcript

        self.gate = RootRecoveryGate(
            config=lambda: self.config,
            session_view=lambda: self.view,
            availability=lambda: self.snapshot,
            stage_notice=self._stage_notice,
            emit_projected=self._emit_projected,
            transcript_integrity=None if transcript is None else read_transcript,
        )
        self.gate._clock = self.clock
        self.gate._jitter_draw = lambda: 0.5

    def _stage_notice(
        self, detail: RootRecoveryNoticeDetail, message: str
    ) -> ProjectedUpdate | None:
        self.staged.append((detail, message))
        return cast(ProjectedUpdate, object())

    async def _emit_projected(self, update: ProjectedUpdate) -> None:
        return None

    async def schedule(self) -> Any:
        return await self.gate.schedule(
            RootRecoveryScheduleParams(
                session_id="session-1",
                terminal_turn_id="turn-1",
                solicited=True,
                input_generation=0,
            )
        )

    async def admit(self, episode_id: str, idempotency_key: str) -> Any:
        return await self.gate.admit(
            RootRecoveryAdmitParams(
                session_id="session-1",
                episode_id=episode_id,
                idempotency_key=idempotency_key,
                terminal_turn_id="turn-1",
                input_generation=0,
            )
        )

    def notices(self, milestone: str) -> list[tuple[RootRecoveryNoticeDetail, str]]:
        return [entry for entry in self.staged if entry[0].milestone == milestone]


class TestS6GatePrecheck:
    """S6: the R2 gate's schedule/admission pre-check runs the shared
    validator over the live transcript; a poisoned transcript parks to the
    manual fallback WITH a repair offer — never another doomed retry.
    """

    @pytest.mark.asyncio
    async def test_gate_declines_schedule_on_a_poisoned_transcript(self) -> None:
        harness = _IntegrityGateHarness([_user("hi"), _genuine_result("ghost")])
        harness.gate.capture_terminal(_failure())
        response = await harness.schedule()
        assert response.decision == "declined"
        assert response.reason == "transcript_integrity"

    @pytest.mark.asyncio
    async def test_gate_schedules_on_a_clean_transcript(self) -> None:
        harness = _IntegrityGateHarness([
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
        ])
        harness.gate.capture_terminal(_failure())
        response = await harness.schedule()
        assert response.decision == "scheduled"

    @pytest.mark.asyncio
    async def test_collaborator_is_optional_and_defaulted(self) -> None:
        # Constructed WITHOUT the collaborator (the gate unit tests' shape):
        # the pre-check is a no-op and never masks a decision.
        harness = _IntegrityGateHarness(None)
        harness.gate.capture_terminal(_failure())
        response = await harness.schedule()
        assert response.decision == "scheduled"

    @pytest.mark.asyncio
    async def test_collaborator_failure_degrades_to_no_reason(self) -> None:
        harness = _IntegrityGateHarness([_user("hi")], collaborator_raises=True)
        harness.gate.capture_terminal(_failure())
        response = await harness.schedule()
        assert response.decision == "scheduled"

    @pytest.mark.asyncio
    async def test_repairable_pi1_pattern_passes_the_precheck(self) -> None:
        # Pinned semantics: the repairable PI1 pattern is NOT an admission
        # violation (S3 dedupes at request build, S4 heals at load, the save
        # boundary never persists it).
        harness = _IntegrityGateHarness(_poison_transcript("call_pi1"))
        harness.gate.capture_terminal(_failure())
        response = await harness.schedule()
        assert response.decision == "scheduled"

    @pytest.mark.asyncio
    async def test_admission_declines_and_cancels_on_a_poisoned_transcript(
        self,
    ) -> None:
        harness = _IntegrityGateHarness([
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
        ])
        harness.gate.capture_terminal(_failure())
        scheduled = await harness.schedule()
        assert scheduled.decision == "scheduled"
        assert scheduled.episode_id is not None
        assert scheduled.idempotency_key is not None

        # The transcript turns poisoned after the schedule (the pre-check
        # reads the LIVE transcript at admission).
        harness.transcript = [_user("hi"), _genuine_result("ghost")]
        response = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
        assert response.decision == "declined"
        assert response.reason == "transcript_integrity"
        # The pending schedule is cancelled, never silently rearmed.
        assert scheduled.episode_id not in harness.gate._episodes
        cancelled = harness.notices("cancelled")
        assert len(cancelled) == 1
        # The repair offer names an ACTIONABLE path (session reload per S4).
        assert "reload (resume) the session" in cancelled[0][1]

    @pytest.mark.asyncio
    async def test_blocked_message_names_the_session_reload_repair_offer(self) -> None:
        harness = _IntegrityGateHarness([
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
        ])
        harness.gate.capture_terminal(_failure())
        scheduled = await harness.schedule()
        assert scheduled.decision == "scheduled"
        episode = next(iter(harness.gate._episodes.values()))
        message = _blocked_message("transcript_integrity", episode)
        assert "reload (resume) the session" in message
        assert "rewind" in message
        assert "retrying would fail the same way" in message

    @pytest.mark.asyncio
    async def test_the_integrity_diagnostic_category_is_never_scheduled(self) -> None:
        # Acceptance 9 tie-in: the diagnostic classifies UNKNOWN, and UNKNOWN
        # is never an eligible class — it can never become a scheduled retry.
        harness = _IntegrityGateHarness([
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
        ])
        harness.gate.capture_terminal(
            _failure(
                cause_category=FailureCategory.UNKNOWN,
                failure_class=FailureClass.UNKNOWN,
            )
        )
        response = await harness.schedule()
        assert response.decision == "declined"
        assert response.reason == "not_eligible"


async def _until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


async def _waiting_runtime() -> tuple[AgentLoop, WaitManager, Any, str]:
    """One waiting root turn over a minimal WaitManager (the S6 app surface)."""
    loop, _backend = make_loop([wait_call("child")])
    manager = WaitManager()
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    root = registry._build_child_runtime(loop)
    root.turns._subagent_runner = cast(SubagentRunnerPort, manager)
    response, action = root.turns.start(
        TurnStartParams(
            session_id=loop.session_id, message=[TextContentBlock(text="start")]
        )
    )
    action()
    await _until(lambda: loop.is_waiting_only(response.turn.id))
    return loop, manager, root, response.turn.id


class TestS6TurnAdmission:
    """S6 app-level: the manual retry/new-turn admission (TurnController.start)
    runs the shared validator BEFORE scheduling or appending.
    """

    @pytest.mark.asyncio
    async def test_turn_admission_fails_closed_on_unrepairable_transcript(self) -> None:
        loop, _manager, root, _turn_id = await _waiting_runtime()
        try:
            loop.messages.append(_genuine_result("ghost"))
            with pytest.raises(TranscriptIntegrityError):
                root.turns.start(
                    TurnStartParams(
                        session_id=loop.session_id,
                        message=[TextContentBlock(text="again")],
                    )
                )
        finally:
            await root.close()
            await loop.aclose()

    @pytest.mark.asyncio
    async def test_turn_admission_passes_the_repairable_pi1_pattern(self) -> None:
        loop, _manager, root, _turn_id = await _waiting_runtime()
        try:
            for message in _poison_transcript("call_pi1"):
                loop.messages.append(message)
            # The repairable pattern is NOT an admission violation: the check
            # passes and the (already running) turn conflict surfaces instead.
            with pytest.raises(TurnConflictError):
                root.turns.start(
                    TurnStartParams(
                        session_id=loop.session_id,
                        message=[TextContentBlock(text="again")],
                    )
                )
        finally:
            await root.close()
            await loop.aclose()

    @pytest.mark.asyncio
    async def test_gate_is_wired_to_the_live_transcript(self) -> None:
        loop, _manager, root, _turn_id = await _waiting_runtime()
        try:
            gate = root.turns.root_recovery
            assert gate._transcript_integrity is not None
            assert list(gate._transcript_integrity()) == list(loop.messages)
        finally:
            await root.close()
            await loop.aclose()

    @pytest.mark.asyncio
    async def test_app_gate_declines_schedule_on_a_poisoned_transcript(self) -> None:
        loop, _manager, root, _turn_id = await _waiting_runtime()
        try:
            gate = root.turns.root_recovery
            gate.capture_terminal(_failure())
            loop.messages.append(_genuine_result("ghost"))
            response = await gate.schedule(
                RootRecoveryScheduleParams(
                    session_id=loop.session_id,
                    terminal_turn_id="turn-1",
                    solicited=True,
                    input_generation=0,
                )
            )
            assert response.decision == "declined"
            assert response.reason == "transcript_integrity"
        finally:
            await root.close()
            await loop.aclose()


# ---------------------------------------------------------------------------
# Acceptance 7 — regression pins
# ---------------------------------------------------------------------------


class TestRegressionPins:
    """The empirical reorder cases, the adjacent tool→tool duplicate variant
    (defense in depth), the append→request-build→load replay, and repair
    idempotence.
    """

    def test_adjacent_tool_tool_duplicate_variant_is_flagged(self) -> None:
        adjacent = [
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
            _genuine_result("c1"),
        ]
        violations = validate_transcript(adjacent, live_tail_exempt=True)
        assert [v.kind for v in violations] == ["duplicate_tool_result"]
        with pytest.raises(TranscriptIntegrityError):
            messages_for_backend(adjacent, _model())

    def test_adjacent_placeholder_genuine_variant_is_repaired(self) -> None:
        adjacent = [_assistant("c1"), _placeholder("c1"), _genuine_result("c1")]
        projection = messages_for_backend(adjacent, _model())
        results = [m for m in projection if m.role == Role.tool]
        assert len(results) == 1
        assert has_structured_tool_result(results[0])
        assert [m.role for m in projection] == [Role.assistant, Role.tool]

    def test_replay_through_load_and_request_build_emits_no_400_shape(
        self, tmp_path: Path
    ) -> None:
        # The raw steered order the mid-turn steering append produced: the
        # replay through load → request build emits no 400-shape request.
        raw = [_user("hi"), _assistant("c1"), _user("steered"), _genuine_result("c1")]
        session_dir = _write_session(tmp_path / "session_replay", raw)
        messages, metadata = SessionLoader.load_session(session_dir)
        assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata  # nothing to repair
        projection = messages_for_backend(messages, _model())
        _assert_no_user_tool_adjacency(projection)
        assert [m.role for m in projection] == [
            Role.user,
            Role.assistant,
            Role.tool,
            Role.user,
        ]

        # The poisoned on-disk shape heals through the same replay.
        poisoned_dir = _write_session(
            tmp_path / "session_replay_poison", _poison_transcript("c1")
        )
        healed, _healed_metadata = SessionLoader.load_session(poisoned_dir)
        healed_projection = messages_for_backend(healed, _model())
        _assert_no_user_tool_adjacency(healed_projection)
        assert not any(is_synthetic_tool_placeholder(m) for m in healed_projection)

    def test_repeated_repair_is_an_idempotent_no_op(self) -> None:
        once, report = repair_transcript(_poison_transcript("c1"))
        assert report.repaired is True
        twice, second = repair_transcript(once)
        assert twice == once
        assert second.repaired is False
        assert second.violations == ()

    def test_repair_report_metadata_round_trip(self) -> None:
        _deduped, report = repair_transcript(_poison_transcript("call_pi1"))
        payload = report_to_metadata(report)
        assert report_from_metadata(payload) == report
        metadata: dict[str, object] = {TRANSCRIPT_INTEGRITY_METADATA_KEY: payload}
        assert pop_report_from_metadata(metadata) == report
        assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
        assert pop_report_from_metadata(metadata) is None


# ---------------------------------------------------------------------------
# Round-1 release review — package S repairs (S1 a/b/c)
# ---------------------------------------------------------------------------


class TestRound1S1Repair:
    """Round-1 release-review S1 (seats 3+4): the request-build reorder's
    placeholder drop was scoped by CALL ID ONLY, diverging from the
    per-originating-assistant-call scoping used everywhere else in package
    T. On the legitimate shape the designed S2 repair itself produces —
    [user, A1(c1), P(c1 synthetic placeholder), user, A2(c1), R(c1 genuine)]
    (an earlier dangling call backfilled with a placeholder, then a later
    turn reusing the same call id) — every integrity seam passed while the
    reorder DROPPED the placeholder and emitted a projection ending in an
    UNANSWERED assistant tool call (a provider 400 shape no seam detected,
    repaired, or diagnosed; pre-T code served this shape correctly).
    """

    def test_reused_call_id_keeps_the_earlier_placeholder(self) -> None:
        """THE S1(b) REQUEST-BUILD REGRESSION PIN.

        The placeholder answers the EARLIER call; the genuine result answers
        the LATER one. The request-build projection must KEEP the
        placeholder (the drop is scoped per originating assistant call) and
        stay provider-safe: every assistant tool call is answered — no
        unresolved trailing assistant tool call.

        FAILS against the round-1 code: the call-id-scoped drop removed the
        placeholder and the projection ended in an unanswered assistant
        tool call.
        """
        messages = [
            _user("first"),
            _assistant("c1"),
            _placeholder("c1"),
            _user("second"),
            _assistant("c1"),
            _genuine_result("c1"),
        ]
        # The raw shape itself is legitimate (pinned by the S2 amplifier
        # fix): every integrity seam passes.
        assert validate_transcript(messages, live_tail_exempt=True) == []
        projection = list(messages_for_backend(messages, _model()))
        # The placeholder is KEPT: it answers the earlier dangling call.
        results = [m for m in projection if m.role == Role.tool]
        assert [m.tool_call_id for m in results] == ["c1", "c1"]
        assert is_synthetic_tool_placeholder(results[0])
        assert has_structured_tool_result(results[1])
        assert [m.role for m in projection] == [
            Role.user,
            Role.assistant,
            Role.tool,
            Role.user,
            Role.assistant,
            Role.tool,
        ]
        # Provider-safe: no [user, tool] adjacency and every assistant
        # tool_use is immediately paired — no unresolved trailing call.
        _assert_no_user_tool_adjacency(projection)
        _assert_tool_calls_immediately_paired(projection)

    def test_reorder_scopes_the_placeholder_drop_per_originating_call(self) -> None:
        # S1(a): the drop is scoped per originating assistant call. The
        # placeholder under A1 is kept (A1's own group holds no genuine
        # result — the later turn's genuine result answers A2, not A1),
        # while the unambiguous PI1 pattern under ONE call still drops it.
        reuse = [
            _user("first"),
            _assistant("c1"),
            _placeholder("c1"),
            _user("second"),
            _assistant("c1"),
            _genuine_result("c1"),
        ]
        reordered = reorder_for_tool_adjacency(reuse)
        assert [m.role for m in reordered] == [
            Role.user,
            Role.assistant,
            Role.tool,
            Role.user,
            Role.assistant,
            Role.tool,
        ]
        assert is_synthetic_tool_placeholder(reordered[2])
        assert has_structured_tool_result(reordered[5])
        # The mirror shape: a placeholder under the LATER call, shadowed by
        # that call's own genuine result, is dropped (the load-repair pin).
        mirrored = [
            _user("first"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("second"),
            _assistant("c1"),
            _placeholder("c1"),
            _genuine_result("c1"),
        ]
        deduped = dedupe_shadowed_placeholders(mirrored)
        assert [m.role for m in deduped] == [
            Role.user,
            Role.assistant,
            Role.tool,
            Role.user,
            Role.assistant,
            Role.tool,
        ]
        assert all(
            has_structured_tool_result(m) for m in deduped if m.role == Role.tool
        )
        # The unambiguous poison pattern under ONE call still drops it.
        poison = _poison_transcript("c1")
        assert not any(
            is_synthetic_tool_placeholder(m) for m in reorder_for_tool_adjacency(poison)
        )

    def test_provider_ready_fails_closed_on_the_stranded_projection(self) -> None:
        # S1(c), re-anchored by round-2 V1. The GENUINELY stranded shape is
        # the RAW transcript whose projection strands a call: the later call
        # reusing the id has its genuine result recorded AFTER it, but the
        # reorder delivers that result to the earlier call reusing the id —
        # a result exists in the transcript but was never delivered to ITS
        # owner. Reported at every seam that sees the raw order.
        #
        # The original S1(c) pin fed the round-1 defect's PROJECTION shape
        # ([U, A1(c1), R1(c1), U, A2(c1)]) back into the validator. That
        # sequence is identical to the legitimate live tail (the round-1
        # disclosed limitation: "sequence-identical to the stranded
        # projection at the seams"); round-2 V1 resolves the collision in
        # favor of the live reading (the live shape is pinned accepted by
        # TestRound2V1LiveTailScoping below), so the pin now feeds the raw
        # stranded transcript instead. The request-build seam reorders
        # BEFORE validating and therefore cannot distinguish the two
        # readings; the raw order is policed at the append/persistence/
        # load/install/admission seams instead (pinned there).
        stranded = [
            _user("first"),
            _assistant("c1"),
            _user("second"),
            _assistant("c1"),
            _genuine_result("c1"),
        ]
        violations = validate_provider_ready(stranded)
        assert [v.kind for v in violations] == ["unresolved_tool_call"]
        assert violations[0].tool_call_id == "c1"
        # The composite check and the persistence boundary refuse it too: the
        # save fails closed, so the stranded transcript never reaches a
        # request build.
        assert validate_transcript(stranded, live_tail_exempt=True) != []
        assert find_persistence_violations(stranded) != []
        with pytest.raises(TranscriptIntegrityError):
            SessionLogger._check_transcript_persistence(stranded)
        # The load-time repair reports the violation, so the write-owning
        # resume install fails closed as well.
        _deduped, report = repair_transcript(stranded)
        assert [v.kind for v in report.violations] == ["unresolved_tool_call"]

    def test_provider_ready_keeps_the_live_tail_exempt(self) -> None:
        # S1(c)'s exemption: a call whose result is still in flight stays
        # legitimate — no GENUINE result is recorded for its id anywhere in
        # the transcript. The live tail, the staged batch, and a later turn
        # reusing an id whose only recorded result is the earlier call's
        # synthetic placeholder (the backfill's marker, not an answer) all
        # pass.
        assert validate_provider_ready([_user("hi"), _assistant("c1")]) == []
        assert (
            validate_provider_ready([
                _user("hi"),
                _assistant("c1", "c2"),
                _genuine_result("c1"),
            ])
            == []
        )
        in_flight = [
            _user("first"),
            _assistant("c1"),
            _placeholder("c1"),
            _user("second"),
            _assistant("c1"),
        ]
        assert validate_provider_ready(in_flight) == []
        assert validate_transcript(in_flight, live_tail_exempt=True) == []


# ---------------------------------------------------------------------------
# Round-2 release review — package V repairs (V1 live-tail scoping, V2 staged)
# ---------------------------------------------------------------------------


class TestRound2V1LiveTailScoping:
    """Round-2 V1 (MAJOR): the live-tail exemption is scoped per OWNING
    assistant block / live-tail region, not a GLOBAL genuine-result id set.

    The legitimate mid-turn reuse shape [U, A1(c1), R1(c1), U, A2(c1)] — A1
    earlier emitted a call with the same id c1 and obtained a genuine result
    R1(c1) (legitimate id reuse across owner blocks); A2's call to c1 is IN
    FLIGHT at the live tail — is exactly the live tail the exemption exists
    for. Persistence invokes the composite check and mid-turn steering saves
    immediately (_context_injection.py:264), so the live save and the resume
    install (_runtime.py:1757-1759) must ACCEPT it.

    FAILS pre-fix: the global genuine-result id set disqualified the
    exemption (c1 IS in the set because of A1's result), so the live shape was
    reported unresolved_tool_call and the live save/resume install was
    REJECTED — a legitimate live session shape became unrecoverable (the
    same defect class as PI1).
    """

    def test_live_reuse_shape_is_accepted_by_provider_ready_and_composite(self) -> None:
        live = [
            _user("first"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("second"),
            _assistant("c1"),
        ]
        assert validate_provider_ready(live) == []
        assert validate_provider_ready(live, live_tail_exempt=True) == []
        assert validate_transcript(live, live_tail_exempt=True) == []
        # The mid-turn steering save variant: the steered (injected) user
        # message trails the in-flight call — the shape the steering save at
        # _context_injection.py:264 persists.
        steered = [*live, _user(_STEER_TEXT, injected=True)]
        assert validate_provider_ready(steered) == []
        assert validate_transcript(steered, live_tail_exempt=True) == []
        # The request build accepts it too (the projection is provider-safe
        # once the in-flight result lands).
        projection = messages_for_backend(live, _model())
        _assert_no_user_tool_adjacency(projection)

    def test_live_reuse_shape_passes_the_persistence_and_resume_install(
        self, tmp_path: Path
    ) -> None:
        live = [
            _user("first"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("second"),
            _assistant("c1"),
        ]
        steered = [*live, _user(_STEER_TEXT, injected=True)]
        # The persistence boundary accepts the live save (the mid-turn
        # steering save): no violation, no raise.
        for messages in (live, steered):
            assert find_persistence_violations(messages) == []
            SessionLogger._check_transcript_persistence(messages)
        # The resume install accepts the live shape on disk: the load-time
        # repair reports nothing to repair, so the write-owning install
        # (_runtime.py:1757-1759) never rejects it.
        for name, messages in (("live", live), ("steered", steered)):
            session_dir = _write_session(tmp_path / f"session_{name}", messages)
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
            # No durable migration was needed: the live shape is clean.
            assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))

    def test_stranded_shape_still_fails_closed_is_a_regression_guard(self) -> None:
        # Round-2 V1 pin (c): the stranded detection is NOT reopened. A
        # result exists in the transcript but was never delivered to ITS
        # owner: the raw stranded transcript [U, A1(c1), U, A2(c1), R1(c1)]
        # records A2's genuine result after A2, but the reorder delivers it
        # to the earlier call reusing the id. (Pinned in full by
        # test_provider_ready_fails_closed_on_the_stranded_projection above;
        # this guard pins the exemption's per-owner scoping directly.)
        stranded = [
            _user("first"),
            _assistant("c1"),
            _user("second"),
            _assistant("c1"),
            _genuine_result("c1"),
        ]
        violations = validate_provider_ready(stranded)
        assert [v.kind for v in violations] == ["unresolved_tool_call"]
        assert violations[0].tool_call_id == "c1"
        # A mid-transcript dangling call reusing an id whose only genuine
        # results belong to OTHER owner blocks is still flagged (the
        # exemption is scoped to the live tail region, not global).
        mid_transcript = [
            _user("first"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("second"),
            _assistant("c1"),
            _user("third"),
            _assistant("c2"),
            _genuine_result("c2"),
        ]
        violations = validate_provider_ready(mid_transcript)
        assert [v.kind for v in violations] == ["unresolved_tool_call"]
        assert violations[0].tool_call_id == "c1"


class TestRound2V2StagedStrictPath:
    """Round-2 V2 (MINOR): the composite STRICT path (live_tail_exempt=False)
    disregarded staged_call_ids — a call whose result is staged in an active
    tool batch (delivered out-of-band) is satisfied by the staged-result
    contract, so the strict provider-ready check must not flag it.

    FAILS pre-fix: [U, A(c1, c2), R(c1)] with staged_call_ids={'c2'} failed
    provider-ready under live_tail_exempt=False (the staged parameter existed
    only on validate_settled_history/validate_live_prefix).
    """

    def test_strict_path_honors_staged_call_ids(self) -> None:
        messages = [_user("hi"), _assistant("c1", "c2"), _genuine_result("c1")]
        # The staged batch result satisfies the call in the strict path.
        assert (
            validate_provider_ready(
                messages, live_tail_exempt=False, staged_call_ids=frozenset({"c2"})
            )
            == []
        )
        assert (
            validate_transcript(
                messages, staged_call_ids=frozenset({"c2"}), live_tail_exempt=False
            )
            == []
        )
        # Without the staged contract the same call is unresolved (strict).
        assert [
            v.kind for v in validate_provider_ready(messages, live_tail_exempt=False)
        ] == ["unresolved_tool_call"]
        # The strict COMPOSITE reports the unresolved call from BOTH sub-checks
        # (settled-history completeness and provider-ready order carry
        # distinct detail strings, so _dedupe_violations keeps both) — the
        # pre-existing double report, verified against the 862aeb6 module.
        strict = validate_transcript(messages, live_tail_exempt=False)
        assert [v.kind for v in strict] == [
            "unresolved_tool_call",
            "unresolved_tool_call",
        ]
        assert {v.tool_call_id for v in strict} == {"c2"}
        # The exempt path is unchanged (a staged call was already exempt).
        assert (
            validate_provider_ready(
                messages, live_tail_exempt=True, staged_call_ids=frozenset({"c2"})
            )
            == []
        )


# ---------------------------------------------------------------------------
# Round-3 release review — package W repairs (W1 request-build ownership,
# W2 production live-tail boundary)
# ---------------------------------------------------------------------------


class TestRound3W1RequestBuildOwnership:
    """Round-3 W1 (MAJOR): request build erased the ownership evidence
    before validating.

    ``messages_for_backend`` reordered BEFORE validating, so the reorder
    could move a genuine result up to an EARLIER owner block reusing the
    call id; the provider-ready validation then ran on the PROJECTED order,
    read the later owner's unanswered call as a legitimate live-tail
    in-flight call (the exact shape round-2 V1 was designed to accept), and
    DISPATCHED the request with the later call unanswered — the PI1 400
    poison shape (a ``tool_use`` with no ``tool_result``) reached the
    provider. Round-2 V's re-anchoring of the stranded pin had dropped the
    former ``messages_for_backend`` rejection assertion; these pins restore
    the end-to-end request-build stranded rejection.

    FAILS pre-fix: ``messages_for_backend`` accepted the raw stranded
    transcript and a controlled ``LLMGateway.complete`` DISPATCHED it — the
    fake backend received the request (verified: ``requests_messages`` had
    one entry).
    """

    def test_request_build_refuses_the_raw_stranded_transcript(self) -> None:
        # The raw stranded transcript: R2's result follows A2's call, but the
        # reorder delivers it to A1 — a result exists in the transcript but
        # was never delivered to ITS owner. The raw order is refused by
        # direct validation (pinned by the round-1/round-2 stranded pins);
        # the request-build seam must refuse it too — the reorder is only
        # for the wire and must not launder the stranded shape into the
        # legitimate live-tail shape.
        stranded = [
            _user("first"),
            _assistant("c1"),
            _user("second"),
            _assistant("c1"),
            _genuine_result("c1"),
        ]
        with pytest.raises(TranscriptIntegrityError) as excinfo:
            messages_for_backend(stranded, _model())
        assert excinfo.value.violations[0].kind == "unresolved_tool_call"
        assert excinfo.value.violations[0].tool_call_id == "c1"
        # The legitimate live-reuse shape (the reorder's other reading of
        # the same ids) still dispatches at the same seam.
        live = [
            _user("first"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("second"),
            _assistant("c1"),
        ]
        projection = messages_for_backend(live, _model())
        _assert_no_user_tool_adjacency(projection)

    @pytest.mark.asyncio
    async def test_stranded_transcript_never_reaches_the_backend(self) -> None:
        # THE END-TO-END PIN: a controlled LLMGateway.complete over the raw
        # stranded transcript must fail closed BEFORE any provider call.
        stranded = [
            _user("first"),
            _assistant("c1"),
            _user("second"),
            _assistant("c1"),
            _genuine_result("c1"),
        ]
        backend = FakeBackend([[mock_llm_chunk(content="ok")]])
        inputs = CompletionInputs(
            model=_model(),
            provider_name="test",
            emits_finish_reason=True,
            messages=tuple(stranded),
            tools=None,
            tool_choice=None,
            extra_headers={},
            metadata={},
            max_tokens=None,
        )
        resources = CallResources(
            backend=backend, stats=AgentStats(), process_message=lambda message: message
        )
        with pytest.raises(TranscriptIntegrityError) as excinfo:
            await LLMGateway().complete(inputs, resources)
        assert excinfo.value.violations[0].kind == "unresolved_tool_call"
        assert excinfo.value.violations[0].tool_call_id == "c1"
        # The PI1 400 poison shape never reached the provider.
        assert backend.requests_messages == []
        # ...and the legitimate live-reuse shape still dispatches end-to-end.
        live = [
            _user("first"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("second"),
            _assistant("c1"),
        ]
        live_backend = FakeBackend([[mock_llm_chunk(content="ok")]])
        live_inputs = CompletionInputs(
            model=_model(),
            provider_name="test",
            emits_finish_reason=True,
            messages=tuple(live),
            tools=None,
            tool_choice=None,
            extra_headers={},
            metadata={},
            max_tokens=None,
        )
        live_resources = CallResources(
            backend=live_backend,
            stats=AgentStats(),
            process_message=lambda message: message,
        )
        chunk = await LLMGateway().complete(live_inputs, live_resources)
        assert chunk.message.content == "ok"
        assert len(live_backend.requests_messages) == 1


class TestRound3W2ProductionLiveTail:
    """Round-3 W2 (MAJOR): the live-tail region detector did not recognize
    production mid-turn shapes.

    The backward scan relied on message flags, but production steering
    injects user messages via ``as_message=True`` with ``injected=False``
    (``app_server/_turns.py`` steering path -> ``_context_injection.py``
    ``_inject_user_context``), and the settled skill/mention assistant
    messages also lack the flags (``_context_injection.py`` ``inject_skill``
    / ``inject_mentions``). For ``[U, A1(c1), R1(c1), U, A2(c1), Usteer]``
    with A2 in flight the scan found no live region and the check reported
    the legitimate in-flight A2 as an ``unresolved_tool_call`` — the
    save/resume install was rejected (the original round-2 V1 defect for
    REAL steering shapes; V's pin marked the steering message
    ``injected=True``, a combination production never produces).

    The acceptance pins drive the ACTUAL ContextInjector with production
    flag values (no fabricated flag combinations).

    FAILS pre-fix: the steering save raised ``TranscriptIntegrityError``
    and the skill-expansion save raised ``TranscriptIntegrityError`` (both
    verified against the unfixed module).
    """

    @pytest.mark.asyncio
    async def test_production_steering_save_accepts_the_live_reuse_tail(
        self, tmp_path: Path
    ) -> None:
        config = build_test_vibe_config(
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="w-steer", enabled=True
            )
        )
        loop = build_test_agent_loop(
            config=config, backend=FakeBackend([[mock_llm_chunk(content="ok")]])
        )
        try:
            for message in (
                _user("first"),
                _assistant("c1"),
                _genuine_result("c1"),
                _user("second"),
                _assistant("c1"),  # in flight at the live tail
            ):
                loop.messages.append(message)
            # The actual production steering path: as_message=True appends
            # the user message with injected=False and saves immediately.
            await loop.inject_user_context(_STEER_TEXT, as_message=True)
            transcript = list(loop.messages)
            steered = transcript[-1]
            assert steered.role == Role.user
            assert steered.injected is False  # production flag values
            assert steered.content == _STEER_TEXT
            # The save seam accepts the legitimate in-flight call...
            assert find_persistence_violations(transcript) == []
            SessionLogger._check_transcript_persistence(transcript)
            # ...and so does the resume install (load + write-owning install).
            session_dir = _write_session(tmp_path / "session_steer", transcript)
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
            assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        finally:
            await loop.aclose()

    @pytest.mark.asyncio
    async def test_production_skill_expansion_save_accepts_the_live_reuse_tail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = build_test_vibe_config(
            enabled_tools=["skill"],
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="w-skill", enabled=True
            ),
        )
        loop = build_test_agent_loop(
            config=config, backend=FakeBackend([[mock_llm_chunk(content="ok")]])
        )
        try:
            for message in (
                _user("first"),
                _assistant("c1"),
                _genuine_result("c1"),
                _user("second"),
                _assistant("c1"),  # in flight at the live tail
            ):
                loop.messages.append(message)
            skill = SkillInfo(
                name="w-skill", description="test", prompt="Do the thing."
            )
            monkeypatch.setattr(
                loop.skill_manager,
                "parse_skill_command",
                lambda _text: ParsedSkillCommand(name=skill.name, content=skill.prompt),
            )
            monkeypatch.setattr(loop.skill_manager, "get_skill", lambda _name: skill)
            # The actual injector skill expansion: the settled assistant/tool
            # pair is appended with production flag values (injected=False).
            events = [event async for event in loop._inject_invoked_skill("/w-skill")]
            assert len(events) == 2
            transcript = list(loop.messages)
            skill_assistant, skill_result = transcript[-2], transcript[-1]
            assert skill_assistant.role == Role.assistant
            assert skill_assistant.injected is False  # production flag values
            assert skill_assistant.tool_calls
            assert skill_result.role == Role.tool
            assert has_structured_tool_result(skill_result)
            # The save seam accepts the legitimate in-flight call...
            assert find_persistence_violations(transcript) == []
            SessionLogger._check_transcript_persistence(transcript)
            await loop._save_messages()  # the actual save: no raise
            # ...and so does the resume install.
            session_dir = _write_session(tmp_path / "session_skill", transcript)
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
            assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        finally:
            await loop.aclose()

    def test_the_live_region_still_ends_at_a_turn_boundary(self) -> None:
        # The structural boundary must not leak past a turn boundary: the
        # production-flag twin of the pinned mid-transcript dangling shape
        # (a steered user message followed by a settled turn) keeps an EMPTY
        # live region, and the multi-reuse stranded shapes keep failing
        # closed — the exemption never reaches a call whose owner block
        # sits outside the live tail region.
        steer_then_turn = [
            _user("first"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("second"),
            _assistant("c1"),
            _user(_STEER_TEXT),  # production steering user (injected=False)
            _assistant("c2"),
            _genuine_result("c2"),
        ]
        violations = validate_provider_ready(steer_then_turn)
        assert [v.kind for v in violations] == ["unresolved_tool_call"]
        assert violations[0].tool_call_id == "c1"
        assert validate_transcript(steer_then_turn, live_tail_exempt=True) != []
        assert find_persistence_violations(steer_then_turn) != []
        with pytest.raises(TranscriptIntegrityError):
            messages_for_backend(steer_then_turn, _model())
        multi_reuse = [
            _user("first"),
            _assistant("c1"),
            _user("second"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("third"),
            _assistant("c1"),
            _genuine_result("c1"),
        ]
        violations = validate_provider_ready(multi_reuse)
        assert [v.kind for v in violations] == ["unresolved_tool_call"]
        assert violations[0].tool_call_id == "c1"
        assert validate_transcript(multi_reuse, live_tail_exempt=True) != []
        assert find_persistence_violations(multi_reuse) != []
        with pytest.raises(TranscriptIntegrityError):
            messages_for_backend(multi_reuse, _model())


# ---------------------------------------------------------------------------
# Round-4 release review — package X repairs (X1 composed-steer live-region
# walk, X2 production-composition coverage matrix, X4 structural call-id
# disjointness)
# ---------------------------------------------------------------------------


def _live_reuse_tail() -> list[LLMMessage]:
    """The round-2 V1 live-reuse tail: A2's call to the reused id is in flight."""
    return [
        _user("first"),
        _assistant("c1"),
        _genuine_result("c1"),
        _user("second"),
        _assistant("c1"),  # in flight at the live tail
    ]


def _injected_pair(call_id: str, tool_name: str) -> tuple[LLMMessage, LLMMessage]:
    """One settled production injection pair (skill/mention expansion).

    The injector appends the assistant message unflagged with EMPTY content
    and a fresh ``str(uuid4())`` call id, then the structured result
    (``_context_injection.py`` ``inject_skill`` / ``inject_mentions``).
    Round-5 Y: the fixture sets the injector's private provenance marker
    explicitly (tests may fabricate private provenance to represent
    injector-owned fixtures) — the marker is the ONLY nesting authority now,
    never the call id's format.
    """
    assistant = LLMMessage(
        role=Role.assistant,
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                index=0,
                function=FunctionCall(name=tool_name, arguments="{}"),
            )
        ],
    )
    assistant._context_injection_origin = CONTEXT_INJECTION_TOOL_PAIR_V1
    result = LLMMessage(
        role=Role.tool,
        tool_call_id=call_id,
        name=tool_name,
        content=f"{tool_name} result",
        tool_result=PersistedToolResult(output={"text": f"{tool_name} result"}),
    )
    return assistant, result


def _patch_skill_manager(
    loop: AgentLoop, monkeypatch: pytest.MonkeyPatch, name: str = "x-skill"
) -> None:
    """Patch the loop's skill manager the way the W2 pins do (text-sensitive:
    only ``/``-commands parse, mirroring ``parse_skill_command``).
    """
    skill = SkillInfo(name=name, description="test", prompt="Do the thing.")
    monkeypatch.setattr(
        loop.skill_manager,
        "parse_skill_command",
        lambda text: (
            ParsedSkillCommand(name=skill.name, content=skill.prompt)
            if text.strip().startswith("/")
            else None
        ),
    )
    monkeypatch.setattr(loop.skill_manager, "get_skill", lambda _name: skill)


def _assert_live_region_accepted(
    messages: list[LLMMessage], *region_members: LLMMessage
) -> None:
    """An accepted composition: the live region is EXACTLY the expected
    assistant tool-call blocks, and no seam reports a violation.
    """
    region = _live_tail_region_identities(messages)
    assert region == {id(member) for member in region_members}
    assert validate_provider_ready(messages) == []
    assert validate_transcript(messages, live_tail_exempt=True) == []
    assert find_persistence_violations(messages) == []
    projection = messages_for_backend(messages, _model())
    _assert_no_user_tool_adjacency(projection)


def _assert_live_region_refused(messages: list[LLMMessage], call_id: str) -> None:
    """A refused composition: the live region is EMPTY by the walk itself and
    every seam fails closed with the unresolved call.
    """
    assert _live_tail_region_identities(messages) == frozenset()
    violations = validate_provider_ready(messages)
    assert [v.kind for v in violations] == ["unresolved_tool_call"]
    assert {v.tool_call_id for v in violations} == {call_id}
    assert validate_transcript(messages, live_tail_exempt=True) != []
    assert find_persistence_violations(messages) != []
    with pytest.raises(TranscriptIntegrityError):
        messages_for_backend(messages, _model())


class TestRound4X1ComposedSteerVector:
    """Round-4 X1 (MAJOR) + the X2 core pin: the composed production steer
    vector.

    The app_server steer DEFAULT composes the injection:
    ``inject_user_context(prompt, as_message=True,
    inject_implicit=params.inject_invoked_skill)`` (``app_server/_turns.py``
    steer path; ``TurnSteerParams.inject_invoked_skill`` defaults True in
    ``protocol/turns.py``; the same default drives the TUI steer). The
    injector appends the steered user message FIRST
    (``_context_injection.py:233``) and only THEN runs the implicit
    skill/mention expansion (``:239-246``), so the settled pair lands BEHIND
    the steered user message over the in-flight anchor:
    ``[U, A1(c1), R1(c1), U, A2(c1), Usteer, Askill(k), Rskill(k)]``. The
    round-3 W2 walk recognized the pair only when it sat DIRECTLY in the
    anchor's gap, so the composed shape kept an EMPTY live region and the
    immediate save (``:264``) raised ``TranscriptIntegrityError`` on a
    legitimate steer — surfacing to the user as "Steering committed;
    accepted-operation recovery failed".

    These pins drive the ACTUAL ContextInjector with the production steer
    default (no fabricated flag combinations).

    FAILS pre-fix: the four composed saves raised ``TranscriptIntegrityError``
    (verified against the unfixed module). The multiple-consecutive-steers
    pin passes pre-fix — it pins behavior that must not change.
    """

    @pytest.mark.asyncio
    async def test_composed_production_steer_vector_save_accepts_the_live_reuse_tail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = build_test_vibe_config(
            enabled_tools=["skill"],
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="x-steer", enabled=True
            ),
        )
        loop = build_test_agent_loop(
            config=config, backend=FakeBackend([[mock_llm_chunk(content="ok")]])
        )
        try:
            for message in _live_reuse_tail():
                loop.messages.append(message)
            _patch_skill_manager(loop, monkeypatch)
            # THE X2 CORE PIN: the composed vector — as_message=True,
            # inject_implicit=True, the production steer default — appends the
            # steered user message, then the implicit skill pair, then saves.
            await loop.inject_user_context(
                "/x-skill", as_message=True, inject_implicit=True
            )
            transcript = list(loop.messages)
            # [.., A2(c1) in flight, Usteer, Askill, Rskill]
            in_flight = transcript[-4]
            steered = transcript[-3]
            skill_assistant, skill_result = transcript[-2], transcript[-1]
            assert in_flight.role == Role.assistant
            assert [tc.id for tc in in_flight.tool_calls or []] == ["c1"]
            assert steered.role == Role.user
            assert steered.injected is False  # production flag values
            assert steered.content == "/x-skill"
            assert skill_assistant.role == Role.assistant
            assert skill_assistant.injected is False  # production flag values
            assert skill_assistant.tool_calls
            assert skill_result.role == Role.tool
            assert has_structured_tool_result(skill_result)
            # The live region reaches the in-flight anchor THROUGH the
            # steered user message and the settled pair.
            region = _live_tail_region_identities(transcript)
            assert region == {id(in_flight), id(skill_assistant)}
            # The save seam accepts the legitimate in-flight call...
            assert find_persistence_violations(transcript) == []
            SessionLogger._check_transcript_persistence(transcript)
            await loop._save_messages()  # the actual save: no raise
            # ...the composite and the request build too...
            assert validate_transcript(transcript, live_tail_exempt=True) == []
            assert validate_provider_ready(transcript) == []
            projection = messages_for_backend(transcript, _model())
            _assert_no_user_tool_adjacency(projection)
            # ...and so does the resume install (load + write-owning install).
            session_dir = _write_session(tmp_path / "session_x_steer", transcript)
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
            assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        finally:
            await loop.aclose()

    @pytest.mark.asyncio
    async def test_composed_steer_and_mention_save_accepts_the_live_reuse_tail(
        self, tmp_path: Path, tmp_working_directory: Path
    ) -> None:
        (tmp_working_directory / "notes.md").write_text("mention body")
        config = build_test_vibe_config(
            enabled_tools=["read_file"],
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="x-mention", enabled=True
            ),
        )
        loop = build_test_agent_loop(
            config=config,
            backend=FakeBackend([[mock_llm_chunk(content="ok")]]),
            cwd=tmp_working_directory,
        )
        try:
            for message in _live_reuse_tail():
                loop.messages.append(message)
            # The composed vector with a @mention: the steered user message,
            # then the implicit mention pair (read_file through the loop's
            # single-tool path), then the save.
            await loop.inject_user_context(
                "read @notes.md", as_message=True, inject_implicit=True
            )
            transcript = list(loop.messages)
            in_flight = transcript[-4]
            steered = transcript[-3]
            mention_assistant, mention_result = transcript[-2], transcript[-1]
            assert steered.role == Role.user
            assert steered.injected is False  # production flag values
            assert mention_assistant.role == Role.assistant
            assert mention_assistant.injected is False  # production flag values
            assert mention_assistant.tool_calls
            mention_call_id = mention_assistant.tool_calls[0].id
            assert mention_result.role == Role.tool
            assert mention_result.tool_call_id == mention_call_id
            assert mention_result.name == "read_file"
            assert "mention body" in (mention_result.content or "")
            assert has_structured_tool_result(mention_result)
            region = _live_tail_region_identities(transcript)
            assert region == {id(in_flight), id(mention_assistant)}
            assert find_persistence_violations(transcript) == []
            SessionLogger._check_transcript_persistence(transcript)
            await loop._save_messages()  # the actual save: no raise
            assert validate_transcript(transcript, live_tail_exempt=True) == []
            session_dir = _write_session(tmp_path / "session_x_mention", transcript)
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
            assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        finally:
            await loop.aclose()

    @pytest.mark.asyncio
    async def test_composed_steer_skill_then_plain_steer_save_accepts_the_live_reuse_tail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = build_test_vibe_config(
            enabled_tools=["skill"],
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="x-two-steers", enabled=True
            ),
        )
        loop = build_test_agent_loop(
            config=config, backend=FakeBackend([[mock_llm_chunk(content="ok")]])
        )
        try:
            for message in _live_reuse_tail():
                loop.messages.append(message)
            _patch_skill_manager(loop, monkeypatch)
            # A composed steer (skill pair behind the steered user), then a
            # plain steer: the pair sits between two steered user messages
            # over the in-flight anchor.
            await loop.inject_user_context(
                "/x-skill", as_message=True, inject_implicit=True
            )
            await loop.inject_user_context(_STEER_TEXT, as_message=True)
            transcript = list(loop.messages)
            # [.., A2(c1), Usteer1, Askill, Rskill, Usteer2]
            in_flight = transcript[-5]
            steer_one = transcript[-4]
            skill_assistant, skill_result = transcript[-3], transcript[-2]
            steer_two = transcript[-1]
            assert steer_one.role == Role.user and steer_one.injected is False
            assert steer_two.role == Role.user and steer_two.injected is False
            assert steer_two.content == _STEER_TEXT
            assert skill_assistant.role == Role.assistant
            assert skill_assistant.injected is False
            assert skill_result.role == Role.tool
            region = _live_tail_region_identities(transcript)
            assert region == {id(in_flight), id(skill_assistant)}
            assert find_persistence_violations(transcript) == []
            SessionLogger._check_transcript_persistence(transcript)
            await loop._save_messages()  # the actual save: no raise
            assert validate_transcript(transcript, live_tail_exempt=True) == []
            session_dir = _write_session(tmp_path / "session_x_two_steers", transcript)
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
            assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        finally:
            await loop.aclose()

    @pytest.mark.asyncio
    async def test_two_composed_steers_skill_then_mention_save_accepts_the_live_reuse_tail(
        self,
        tmp_path: Path,
        tmp_working_directory: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_working_directory / "notes.md").write_text("mention body")
        config = build_test_vibe_config(
            enabled_tools=["skill", "read_file"],
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="x-skill-mention", enabled=True
            ),
        )
        loop = build_test_agent_loop(
            config=config,
            backend=FakeBackend([[mock_llm_chunk(content="ok")]]),
            cwd=tmp_working_directory,
        )
        try:
            for message in _live_reuse_tail():
                loop.messages.append(message)
            _patch_skill_manager(loop, monkeypatch)
            # Two consecutive composed steers: the skill pair, then the mention
            # pair — two settled pairs behind steered user messages over the
            # in-flight anchor.
            await loop.inject_user_context(
                "/x-skill", as_message=True, inject_implicit=True
            )
            await loop.inject_user_context(
                "read @notes.md", as_message=True, inject_implicit=True
            )
            transcript = list(loop.messages)
            # [.., A2(c1), Usteer1, Askill, Rskill, Usteer2, Amention, Rmention]
            in_flight = transcript[-7]
            steer_one = transcript[-6]
            skill_assistant, skill_result = transcript[-5], transcript[-4]
            steer_two = transcript[-3]
            mention_assistant, mention_result = transcript[-2], transcript[-1]
            assert steer_one.role == Role.user and steer_one.injected is False
            assert steer_two.role == Role.user and steer_two.injected is False
            assert skill_assistant.role == Role.assistant
            assert skill_assistant.injected is False
            assert skill_result.role == Role.tool
            assert mention_assistant.role == Role.assistant
            assert mention_assistant.injected is False
            assert mention_result.role == Role.tool
            region = _live_tail_region_identities(transcript)
            assert region == {id(in_flight), id(skill_assistant), id(mention_assistant)}
            assert find_persistence_violations(transcript) == []
            SessionLogger._check_transcript_persistence(transcript)
            await loop._save_messages()  # the actual save: no raise
            assert validate_transcript(transcript, live_tail_exempt=True) == []
            session_dir = _write_session(
                tmp_path / "session_x_skill_mention", transcript
            )
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
            assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        finally:
            await loop.aclose()

    @pytest.mark.asyncio
    async def test_multiple_consecutive_steers_save_accepts_the_live_reuse_tail(
        self, tmp_path: Path
    ) -> None:
        config = build_test_vibe_config(
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="x-steers", enabled=True
            )
        )
        loop = build_test_agent_loop(
            config=config, backend=FakeBackend([[mock_llm_chunk(content="ok")]])
        )
        try:
            for message in _live_reuse_tail():
                loop.messages.append(message)
            # Two plain steers (as_message=True, inject_implicit=False): the
            # walk's user rule walks over both, so the region still reaches
            # the in-flight anchor. (Passes pre-fix — a characterization pin.)
            await loop.inject_user_context("first steer", as_message=True)
            await loop.inject_user_context("second steer", as_message=True)
            transcript = list(loop.messages)
            in_flight = transcript[-3]
            steer_one, steer_two = transcript[-2], transcript[-1]
            assert steer_one.role == Role.user and steer_one.injected is False
            assert steer_two.role == Role.user and steer_two.injected is False
            assert steer_two.content == "second steer"
            region = _live_tail_region_identities(transcript)
            assert region == {id(in_flight)}
            assert find_persistence_violations(transcript) == []
            SessionLogger._check_transcript_persistence(transcript)
            await loop._save_messages()  # the actual save: no raise
            assert validate_transcript(transcript, live_tail_exempt=True) == []
            session_dir = _write_session(tmp_path / "session_x_steers", transcript)
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
            assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        finally:
            await loop.aclose()


class TestRound4X2CompositionMatrix:
    """Round-4 X2 (MINOR): EVERY production composition of injected content
    over the live-reuse tail, pinned at the walk level.

    The live-reuse tail is ``[U, A1(c1), R1(c1), U, A2(c1)]`` with A2's call
    in flight. Every composition is derived from the production injector
    code (``_context_injection.py``): the steered user message is appended
    with ``injected=False`` (``as_message=True``, ``:212-233``); the settled
    skill/mention pairs are appended unflagged with fresh ``str(uuid4())``
    call ids (``inject_skill`` ``:309/:341-379``, ``inject_mentions``
    ``:397/:398-415``); the compaction boundary and the question-callback /
    agent-lifecycle / middleware surfaces are ``injected=True``. The
    real-injector pins live in :class:`TestRound4X1ComposedSteerVector` and
    the round-3 W2 class; these synthetic twins pin the same shapes at the
    walk level (production flag values and injector-signature call ids — no
    fabricated flag combinations).

    Each accepted composition asserts the live region is EXACTLY the
    expected blocks and no seam reports a violation; each refused
    composition pins the design expectation with its reason.

    FAILS pre-fix: the three composed synthetic twins (the region never
    reached the anchor). Every other pin passes pre-fix — they pin behavior
    that must not change.
    """

    def test_steer_only_composition_is_accepted(self) -> None:
        # as_message=True, inject_implicit=False — the round-3 W2
        # real-injector twin's synthetic shape: [.., A2(c1), Usteer].
        tail = _live_reuse_tail()
        messages = [*tail, _user(_STEER_TEXT)]
        _assert_live_region_accepted(messages, tail[-1])

    def test_skill_only_composition_is_accepted(self) -> None:
        # The round-3 W2 real-injector twin's synthetic shape: the settled
        # skill pair lands DIRECTLY in the in-flight anchor's gap.
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "skill")
        messages = [*tail, pair_assistant, pair_result]
        _assert_live_region_accepted(messages, tail[-1], pair_assistant)

    def test_mention_only_composition_is_accepted(self) -> None:
        # The mention pair (read_file through the single-tool path) lands
        # directly in the in-flight anchor's gap.
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "read_file")
        messages = [*tail, pair_assistant, pair_result]
        _assert_live_region_accepted(messages, tail[-1], pair_assistant)

    def test_composed_steer_and_skill_is_accepted(self) -> None:
        # THE X1 SHAPE (synthetic twin of the real-injector core pin): the
        # steered user message FOLLOWED BY the implicit skill pair.
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "skill")
        messages = [*tail, _user(_STEER_TEXT), pair_assistant, pair_result]
        _assert_live_region_accepted(messages, tail[-1], pair_assistant)

    def test_composed_steer_and_mention_is_accepted(self) -> None:
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "read_file")
        messages = [*tail, _user(_STEER_TEXT), pair_assistant, pair_result]
        _assert_live_region_accepted(messages, tail[-1], pair_assistant)

    def test_composed_steer_skill_and_mention_is_accepted(self) -> None:
        # One steer message carrying both a skill command and a @mention: the
        # injector runs inject_skill then inject_mentions (``:239-246``),
        # landing two settled pairs behind the steered user message.
        tail = _live_reuse_tail()
        skill_assistant, skill_result = _injected_pair(str(uuid4()), "skill")
        mention_assistant, mention_result = _injected_pair(str(uuid4()), "read_file")
        messages = [
            *tail,
            _user("/x-skill read @notes.md"),
            skill_assistant,
            skill_result,
            mention_assistant,
            mention_result,
        ]
        _assert_live_region_accepted(
            messages, tail[-1], skill_assistant, mention_assistant
        )

    def test_multiple_consecutive_steers_are_accepted(self) -> None:
        tail = _live_reuse_tail()
        messages = [*tail, _user("first steer"), _user("second steer")]
        _assert_live_region_accepted(messages, tail[-1])

    def test_skill_pair_before_the_last_user_message_is_accepted(self) -> None:
        # DESIGN (accepted): the pair lands in the anchor's gap, THEN a steer
        # arrives while the anchor's call is still in flight — the pair is
        # directly nested (the round-3 W2 rule) and the trailing steered user
        # is walked over (the user rule), so the region still reaches the
        # anchor.
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "skill")
        messages = [*tail, pair_assistant, pair_result, _user(_STEER_TEXT)]
        _assert_live_region_accepted(messages, tail[-1], pair_assistant)

    def test_implicit_skill_after_a_user_turn_is_accepted(self) -> None:
        # DESIGN (accepted): the normal turn path appends the user message,
        # runs the implicit skill expansion, and the model's response is in
        # flight at the tail — the pair is settled history BEFORE the anchor,
        # which is the trailing block.
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "skill")
        messages = [
            _user("first"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("second"),
            pair_assistant,
            pair_result,
            _assistant("c1"),  # in flight at the live tail
        ]
        _assert_live_region_accepted(messages, messages[-1])

    def test_compaction_marker_over_the_live_reuse_tail_is_accepted(self) -> None:
        # DESIGN (accepted): the compaction boundary is injected=True — the
        # walk skips injected messages unconditionally, so the region still
        # reaches the anchor.
        tail = _live_reuse_tail()
        boundary = LLMMessage(
            role=Role.user,
            content="compaction boundary",
            injected=True,
            context_boundary="compaction",
        )
        messages = [*tail, boundary]
        _assert_live_region_accepted(messages, tail[-1])

    def test_injected_true_question_callback_surface_is_accepted(self) -> None:
        # DESIGN (accepted): the question-callback / middleware / agent
        # lifecycle injected=True user surfaces over the live tail — injected
        # messages are walked over, so the region still reaches the anchor
        # (the round-2 V1 pin's steered variant, pinned here for the matrix).
        tail = _live_reuse_tail()
        messages = [*tail, _user("answer", injected=True)]
        _assert_live_region_accepted(messages, tail[-1])

    def test_skill_pair_before_the_last_user_message_then_a_settled_turn_is_refused(
        self,
    ) -> None:
        # DESIGN (refused): the pair does NOT launder the settled turn behind
        # the new user message. The model answered the new turn, abandoning
        # the anchor's in-flight call: the settled turn's id ("c2") lacks the
        # injector's fresh-uuid4 signature, so it ends the region and the
        # anchor's call is flagged.
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "skill")
        messages = [
            *tail,
            pair_assistant,
            pair_result,
            _user("new turn"),
            _assistant("c2"),
            _genuine_result("c2"),
        ]
        _assert_live_region_refused(messages, "c1")


class TestRound4X4StructuralDisjointness:
    """Round-4 X4 (OBSERVATION): the nested rule's call-id disjointness makes
    the adjacent-duplicate-id refusal STRUCTURAL.

    The shape ``[A2(c2), A3(c2), R2(c2)]`` — a settled block whose call ids
    COLLIDE with the outer block's — gained live-region membership under the
    round-3 W2 walk and was refused only because the reorder delivers the
    result to the first owner while ``genuine_result_owners`` assigns it to
    the last: a fragile fail-closed that a future change to either scan
    could flip. The round-4 rule requires DISJOINTNESS — a nested block whose
    call ids are not disjoint from the outer block's is not injection
    material (production pairs always carry fresh uuid4 ids) — so the shape
    keeps an EMPTY region by the walk itself.

    FAILS pre-fix: the region assertion (the shape gained membership —
    region == 2 blocks — pre-fix). The violation assertions pass pre-fix
    (the fragile fail-closed) and stay green.
    """

    def test_adjacent_duplicate_id_block_keeps_an_empty_live_region(self) -> None:
        adjacent = [_assistant("c2"), _assistant("c2"), _genuine_result("c2")]
        # STRUCTURAL: the walk itself keeps the region empty (pre-fix the
        # shape gained membership and was refused only by the exemption's
        # owner check).
        assert _live_tail_region_identities(adjacent) == frozenset()
        violations = validate_provider_ready(adjacent)
        assert [v.kind for v in violations] == ["unresolved_tool_call"]
        assert violations[0].tool_call_id == "c2"
        assert validate_transcript(adjacent, live_tail_exempt=True) != []
        assert find_persistence_violations(adjacent) != []
        with pytest.raises(TranscriptIntegrityError):
            messages_for_backend(adjacent, _model())

    def test_steered_user_with_a_colliding_settled_turn_keeps_an_empty_region(
        self,
    ) -> None:
        # The guard invariant's colliding case in the steered-user setting:
        # the bounded scan now crosses the steered user message, but the
        # settled turn's ids COLLIDE with the anchor's, so the disjointness
        # key refuses it — a previous turn's response whose call ids collide
        # with the tail's is never walked over.
        tail = _live_reuse_tail()
        messages = [*tail, _user(_STEER_TEXT), _assistant("c1"), _genuine_result("c1")]
        _assert_live_region_refused(messages, "c1")


# ---------------------------------------------------------------------------
# Acceptance 8 — evidence integrity
# ---------------------------------------------------------------------------


class TestEvidenceIntegrity:
    """Repairs never delete genuine structured tool results; synthetic
    placeholders stay identifiable as synthetic (content + missing structured
    result — NO LLMMessage schema change); originals are preserved as
    artifacts before durable writes.
    """

    def test_repair_never_deletes_the_genuine_result(self) -> None:
        poison = _poison_transcript("call_pi1")
        genuine = poison[3]
        deduped, report = repair_transcript(poison)
        assert report.repaired is True
        assert report.dropped_placeholder_call_ids == ("call_pi1",)
        assert any(message is genuine for message in deduped)
        assert not any(is_synthetic_tool_placeholder(m) for m in deduped)
        assert [m.role for m in deduped] == [Role.assistant, Role.user, Role.tool]

    def test_placeholders_stay_identifiable_as_synthetic(self) -> None:
        placeholder = _placeholder("c1")
        interrupted = LLMMessage(
            role=Role.tool,
            tool_call_id="c2",
            name="read_file",
            content=str(
                get_user_cancellation_message(CancellationReason.TOOL_INTERRUPTED)
            ),
        )
        skipped = LLMMessage(
            role=Role.tool,
            tool_call_id="c3",
            name="read_file",
            content=str(
                get_user_cancellation_message(
                    CancellationReason.TOOL_SKIPPED, "read_file"
                )
            ),
        )
        genuine = _genuine_result("c4")
        assert is_synthetic_tool_placeholder(placeholder)
        assert placeholder.tool_result is None
        assert _PLACEHOLDER_TEXT in (placeholder.content or "")
        # The predicate is content + missing structured result — NEVER the
        # cancellation tag alone: genuine settlements never match.
        assert not is_synthetic_tool_placeholder(interrupted)
        assert not is_synthetic_tool_placeholder(skipped)
        assert not is_synthetic_tool_placeholder(genuine)

    def test_repair_of_a_clean_transcript_is_an_exact_no_op(self) -> None:
        clean = [_user("hi"), _assistant("c1"), _genuine_result("c1"), _user("again")]
        deduped, report = repair_transcript(clean)
        assert deduped == clean
        assert report.repaired is False
        assert report.violations == ()

    def test_originals_are_preserved_as_artifacts_before_durable_writes(
        self, tmp_path: Path
    ) -> None:
        # The durable migration preserves the original transcript bytes as a
        # recovery artifact BEFORE the live path is replaced (verified by
        # read-back), so a failed migration never destroys the original.
        session_dir = _write_session(
            tmp_path / "session_pi1", _poison_transcript("call_pi1")
        )
        messages_path = session_dir / MESSAGES_FILENAME
        original_bytes = messages_path.read_bytes()
        messages, metadata = SessionLoader.load_session(session_dir)
        _install_transcript_repair(session_dir, messages, metadata)
        artifact = next(iter(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*")))
        assert artifact.read_bytes() == original_bytes
        assert messages_path.read_bytes() != original_bytes


# ---------------------------------------------------------------------------
# Acceptance 9 — no new recovery authority
# ---------------------------------------------------------------------------


def _invalid_request_error() -> BackendError:
    """The PI1 provider fault: a permanent 400 invalid_request_message_order."""
    return BackendError(
        provider="mistral",
        endpoint="https://api.mistral.test/v1/chat/completions",
        status=400,
        reason="invalid_request_message_order",
        headers=None,
        body_text=(
            '{"error": {"code": 3230, "type": "invalid_request_error", '
            '"message": "Unexpected role tool after role user"}}'
        ),
        parsed_error="invalid_request_message_order",
        model="test-model",
        payload_summary=_payload_summary(),
    )


class TestNoNewRecoveryAuthority:
    """The transcript-integrity diagnostic is a LOCAL typed failure: its own
    category, never transient, never retried, never scheduled — through
    exception wrapping and both projections. The 400/INVALID_REQUEST
    classification stays permanent.
    """

    def test_classification_stays_unknown_and_non_retryable(self) -> None:
        error = _integrity_error()
        info = classify_failure(error)
        assert info.category is FailureCategory.UNKNOWN
        assert info.retry_eligible is False
        assert info.failover_eligible is False
        # The diagnostic's own category is not a FailureCategory member and
        # sits outside every retry/failover/protected set.
        assert TRANSCRIPT_INTEGRITY_CATEGORY not in set(FailureCategory)
        assert FailureCategory.UNKNOWN not in _RETRY_CATEGORIES
        assert FailureCategory.UNKNOWN not in _FAILOVER_CATEGORIES
        assert FailureCategory.UNKNOWN not in _PROTECTED_CATEGORIES
        assert error.category == TRANSCRIPT_INTEGRITY_CATEGORY

    def test_non_retryable_through_an_exception_group(self) -> None:
        group = ExceptionGroup("integrity", [_integrity_error()])
        info = classify_failure(group)
        assert info.category is FailureCategory.UNKNOWN
        assert info.retry_eligible is False
        assert info.failover_eligible is False

    def test_non_retryable_through_a_cause_chain(self) -> None:
        wrapper = RuntimeError("request failed")
        wrapper.__cause__ = _integrity_error()
        info = classify_failure(wrapper)
        assert info.category is FailureCategory.UNKNOWN
        assert info.retry_eligible is False

    def test_public_error_projects_an_internal_error_with_category(self) -> None:
        projected = public_error(_integrity_error())
        assert isinstance(projected, PublicError)
        assert projected.code == TurnErrorCode.INTERNAL_ERROR
        assert isinstance(projected.details, dict)
        assert projected.details["category"] == TRANSCRIPT_INTEGRITY_CATEGORY

    def test_invalid_request_stays_permanent_and_protected(self) -> None:
        info = classify_failure(_invalid_request_error())
        assert info.category is FailureCategory.INVALID_REQUEST
        assert info.retry_eligible is False
        assert info.failover_eligible is False
        assert FailureCategory.INVALID_REQUEST in _PROTECTED_CATEGORIES
        # Protected precedence: a permanent 400 is never masked by a
        # transient sibling anywhere in an exception group.
        group = ExceptionGroup("mixed", [_timeout_error(), _invalid_request_error()])
        grouped = classify_failure(group)
        assert grouped.category is FailureCategory.INVALID_REQUEST
        assert grouped.retry_eligible is False

    def test_migration_failure_is_the_typed_diagnostic(self, tmp_path: Path) -> None:
        error = TranscriptIntegrityError.migration_failed(
            tmp_path / MESSAGES_FILENAME, OSError("injected")
        )
        assert error.violations[0].kind == "transcript_repair_failed"
        info = classify_failure(error)
        assert info.category is FailureCategory.UNKNOWN
        assert info.retry_eligible is False
        projected = public_error(error)
        assert projected.code == TurnErrorCode.INTERNAL_ERROR
        assert isinstance(projected.details, dict)
        assert projected.details["category"] == TRANSCRIPT_INTEGRITY_CATEGORY

    def test_error_mapper_passes_the_typed_diagnostic_through(self) -> None:
        # ROUND-1 S2: the gateway's error mapper passes the typed LOCAL
        # diagnostic through unchanged (the ImagesNotSupportedError
        # precedent) — it is never re-wrapped into a provider-shaped
        # RuntimeError ("API error from {provider} ..."), which mislabeled
        # a local transcript defect as a provider failure.
        error = _integrity_error()
        assert _map_call_error(error, _completion_inputs()) is error


# ---------------------------------------------------------------------------
# Acceptance 10 — non-regression is the battery: the four-path
# injected-retry matrix, the R2 races/root-recovery suites, the WP2
# explicit-resume/reconciliation suites, the session-logger/loader suites,
# the steering suites, the compaction reorder pins, and the rewind suites
# all stay green (run with this file in the verification battery).
# ---------------------------------------------------------------------------


class TestReportType:
    """The report type's contract surface (frozen, value-compared)."""

    def test_report_clean_property(self) -> None:
        _messages, report = repair_transcript(_poison_transcript("c1"))
        assert isinstance(report, TranscriptIntegrityReport)
        assert report.clean is True
        failing = TranscriptIntegrityReport(
            repaired=False,
            dropped_placeholder_call_ids=(),
            violations=(
                TranscriptIntegrityViolation(
                    kind="duplicate_tool_result", tool_call_id="c1", detail="ambiguous"
                ),
            ),
        )
        assert failing.clean is False


# ---------------------------------------------------------------------------
# Round-5 release review — package Y, tranche 1: the fail-open refusals of
# the uuid4-format discriminator (unmarked provider shapes) and the
# old-format session behavior. These pins need no provenance machinery, so
# they run (and FAIL) against the unfixed tree: the current walk accepts
# every one of these shapes with an empty violation set.
# ---------------------------------------------------------------------------


def _provider_pair(
    call_id: str, tool_name: str = "skill"
) -> tuple[LLMMessage, LLMMessage]:
    """One settled UNMARKED provider assistant/result pair (P(k), RP(k)).

    The assistant is built through the ACTUAL OpenAI Chat adapter ingress
    (``OpenAIAdapter.parse_response`` — whole-dictionary validation), so the
    pin exercises the real provider ingestion path, including a
    uuid4-shaped call id when asked. The result is the locally recorded
    genuine settlement (the formatter's shape).
    """
    provider = ProviderConfig(
        name="openai",
        api_base="https://api.openai.test/v1",
        api_key_env_var="OPENAI_API_KEY",
    )
    payload = {
        "id": "chatcmpl-y",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    assistant = OpenAIAdapter().parse_response(payload, provider).message
    assert assistant.role == Role.assistant
    assert assistant.tool_calls and assistant.tool_calls[0].id == call_id
    result = LLMMessage(
        role=Role.tool,
        tool_call_id=call_id,
        name=tool_name,
        content=f"{tool_name} result",
        tool_result=PersistedToolResult(output={"text": f"{tool_name} result"}),
    )
    return assistant, result


class TestRound5YFailOpenRefusals:
    """Round-5 Y: the fail-open the uuid4-format discriminator proved. A
    provider response carrying a uuid4-shaped call id is walked over as if
    it were injector material, so the live region reaches the in-flight
    anchor and the stranded call is EXEMPTED — the provider-400 poison class
    (a request carrying the anchor's unanswered ``tool_use``).

    Notation: T = ``[U, A1(c1), R1(c1), U, A2(c1)]`` with A2's call in
    flight; ``P(k), RP(k)`` is an UNMARKED provider assistant/result pair
    built through the actual adapter ingress.

    FAILS pre-fix: every refusal in this class is ACCEPTED by the unfixed
    walk (non-empty live region, empty violation set, request build
    dispatched) — the round-5 fail-open reproduction.
    """

    def test_uuid4_shaped_provider_response_behind_a_new_user_is_refused(self) -> None:
        # THE EXACT ROUND-5 FAIL-OPEN REPRODUCTION:
        # [U, A1(c1), R1(c1), U, A2(c1), Unew, P(k), RP(k)], k = str(uuid4()).
        tail = _live_reuse_tail()
        provider_assistant, provider_result = _provider_pair(str(uuid4()))
        messages = [
            *tail,
            _user("a brand new user message"),
            provider_assistant,
            provider_result,
        ]
        _assert_live_region_refused(messages, "c1")

    def test_uuid4_shaped_provider_response_behind_a_steered_user_is_refused(
        self,
    ) -> None:
        # The steered-user twin: the production-flag guard shape with a
        # uuid4-shaped provider id — the injector's signature is not
        # authority for a provider response.
        tail = _live_reuse_tail()
        provider_assistant, provider_result = _provider_pair(str(uuid4()))
        messages = [*tail, _user(_STEER_TEXT), provider_assistant, provider_result]
        _assert_live_region_refused(messages, "c1")

    def test_unmarked_direct_disjoint_provider_pair_is_refused(self) -> None:
        # No intervening user: the direct-nested case keeps NO
        # provenance-free structural fallback.
        tail = _live_reuse_tail()
        provider_assistant, provider_result = _provider_pair(str(uuid4()))
        messages = [*tail, provider_assistant, provider_result]
        _assert_live_region_refused(messages, "c1")

    def test_unmarked_non_uuid_direct_provider_pair_is_refused(self) -> None:
        # A non-UUID disjoint provider id is refused too: format is not
        # authority in either direction.
        tail = _live_reuse_tail()
        provider_assistant, provider_result = _provider_pair("provider-call-1")
        messages = [*tail, provider_assistant, provider_result]
        _assert_live_region_refused(messages, "c1")

    def test_provider_id_colliding_with_the_anchor_is_refused(self) -> None:
        # Boundary variant: a provider response reusing the anchor's call id
        # (the disjointness key refuses it — passes pre-fix as a guard).
        tail = _live_reuse_tail()
        provider_assistant, provider_result = _provider_pair("c1")
        messages = [
            *tail,
            _user("a brand new user message"),
            provider_assistant,
            provider_result,
        ]
        _assert_live_region_refused(messages, "c1")

    def test_unmarked_pair_cannot_launder_a_later_provider_response(self) -> None:
        # An earlier settled pair (itself unmarked) does not launder a later
        # provider response behind a new user message.
        tail = _live_reuse_tail()
        first_assistant, first_result = _provider_pair(str(uuid4()))
        later_assistant, later_result = _provider_pair(str(uuid4()))
        messages = [
            *tail,
            first_assistant,
            first_result,
            _user("a brand new user message"),
            later_assistant,
            later_result,
        ]
        _assert_live_region_refused(messages, "c1")


def _write_session_records(
    session_dir: Path, records: Sequence[dict[str, Any]], *, session_id: str = "pi1-y"
) -> Path:
    """Write one synthetic session directory from RAW records.

    Unlike ``_write_session``, the records are written verbatim, so a pin can
    inject arbitrary session-key variants the message model would drop.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    (session_dir / MESSAGES_FILENAME).write_text(lines, encoding="utf-8")
    (session_dir / METADATA_FILENAME).write_text(
        json.dumps({
            "session_id": session_id,
            "total_messages": len(records),
            "environment": {"working_directory": "/repo"},
        }),
        encoding="utf-8",
    )
    return session_dir


class TestRound5YOldFormatSessions:
    """Round-5 Y old-format session behavior (design §8): old records carry
    no session key and restore with no marker, so their settled assistant
    blocks end the walk — while fully settled old history stays valid.

    FAILS pre-fix: the affected old composed-steer shape (the dangling
    REUSED id of T behind a steered user plus an unmarked uuid4 pair,
    persisted without the session key) is ACCEPTED by the unfixed walk, so
    the load reports nothing and the resume install is a no-op.
    """

    def test_old_fully_settled_history_is_accepted(self, tmp_path: Path) -> None:
        # Fully settled old history remains valid (passes pre-fix — guard).
        messages = [
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
            _user("again"),
            _assistant("c2"),
            _genuine_result("c2"),
        ]
        session_dir = _write_session(tmp_path / "session_old_settled", messages)
        loaded, metadata = SessionLoader.load_session(session_dir)
        assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
        assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
        assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))

    @pytest.mark.parametrize(
        "origin_key", ["absent", "null", "unknown", "bool", "object"]
    )
    def test_missing_or_unknown_provenance_session_key_is_not_restored(
        self, tmp_path: Path, origin_key: str
    ) -> None:
        # A composed shape whose pair record carries an absent / null /
        # unknown-string / bool / object session key: no restoration, the
        # region ends, and T is refused at every seam.
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _provider_pair(str(uuid4()))
        messages = [*tail, _user(_STEER_TEXT), pair_assistant, pair_result]
        records = [
            message.model_dump(exclude_none=True, mode="json") for message in messages
        ]
        pair_record = records[-2]
        assert pair_record["role"] == "assistant"
        if origin_key == "absent":
            pair_record.pop("context_injection_origin", None)
        elif origin_key == "null":
            pair_record["context_injection_origin"] = None
        elif origin_key == "unknown":
            pair_record["context_injection_origin"] = "context_injector.tool_pair.v2"
        elif origin_key == "bool":
            pair_record["context_injection_origin"] = True
        else:
            pair_record["context_injection_origin"] = {"version": 1}
        session_dir = _write_session_records(
            tmp_path / f"session_key_{origin_key}", records
        )
        loaded, metadata = SessionLoader.load_session(session_dir)
        # No restoration: the pair's settled block ends the walk, the region
        # is empty, and the reused id is refused.
        assert _live_tail_region_identities(loaded) == frozenset()
        violations = validate_provider_ready(loaded)
        assert [v.kind for v in violations] == ["unresolved_tool_call"]
        assert {v.tool_call_id for v in violations} == {"c1"}
        assert TRANSCRIPT_INTEGRITY_METADATA_KEY in metadata
        with pytest.raises(TranscriptIntegrityError):
            _install_transcript_repair(session_dir, loaded, metadata)
        assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))

    def test_old_interrupted_composed_steer_fails_the_resume_install(
        self, tmp_path: Path
    ) -> None:
        # The affected old shape: the dangling REUSED id of T behind a
        # steered user plus an unmarked uuid4 pair, persisted WITHOUT the
        # session key (pre-marker format). Load returns the messages plus an
        # integrity report; the write-owning install refuses; the disk is
        # unchanged (no migration, no artifact).
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _provider_pair(str(uuid4()))
        messages = [*tail, _user(_STEER_TEXT), pair_assistant, pair_result]
        session_dir = _write_session(tmp_path / "session_old_composed", messages)
        messages_path = session_dir / MESSAGES_FILENAME
        original_bytes = messages_path.read_bytes()
        loaded, metadata = SessionLoader.load_session(session_dir)
        assert TRANSCRIPT_INTEGRITY_METADATA_KEY in metadata
        report = report_from_metadata(metadata[TRANSCRIPT_INTEGRITY_METADATA_KEY])
        assert report is not None
        assert [violation.kind for violation in report.violations] == [
            "unresolved_tool_call"
        ]
        with pytest.raises(TranscriptIntegrityError):
            _install_transcript_repair(session_dir, loaded, metadata)
        assert messages_path.read_bytes() == original_bytes
        assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))


def _saved_records(session_dir: Path) -> list[dict[str, Any]]:
    """The actual records of a saved ``messages.jsonl`` (one JSON per line)."""
    return [
        json.loads(line)
        for line in (session_dir / MESSAGES_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


class TestRound5YProvenanceNesting:
    """Round-5 Y: the replacement discriminator (design §4) — nesting is
    keyed on the injector's PRIVATE PROVENANCE MARKER, never on call-id
    format.

    Notation: T = ``[U, A1(c1), R1(c1), U, A2(c1)]`` with A2's call in
    flight; ``I(k), RI(k)`` is a MARKED injector pair (the fixture sets the
    private marker explicitly — tests may fabricate private provenance to
    represent injector-owned fixtures, design assumptions); ``P(k), RP(k)``
    is an UNMARKED provider pair.

    These pins pass ONLY through the provenance mechanism: pre-fix the
    marked acceptances also pass (the old uuid4 keying walked over them),
    while the unmarked-shape refusals of ``TestRound5YFailOpenRefusals``
    FAIL pre-fix — the two classes together pin both directions of the
    replacement (the marker is necessary, and necessary-plus-structural is
    sufficient).
    """

    def test_direct_marked_pair_is_accepted(self) -> None:
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "skill")
        messages = [*tail, pair_assistant, pair_result]
        _assert_live_region_accepted(messages, tail[-1], pair_assistant)

    def test_composed_marked_pair_is_accepted(self) -> None:
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "skill")
        messages = [*tail, _user(_STEER_TEXT), pair_assistant, pair_result]
        _assert_live_region_accepted(messages, tail[-1], pair_assistant)

    def test_non_uuid_marked_id_is_accepted_direct_and_composed(self) -> None:
        # Format independence (design §12): a NON-UUID marked id nests
        # exactly like a uuid4 one — the marker, not the id's format, is
        # the authority.
        for steered in (False, True):
            tail = _live_reuse_tail()
            pair_assistant, pair_result = _injected_pair("injector-call-7", "skill")
            messages = [*tail]
            if steered:
                messages.append(_user(_STEER_TEXT))
            messages.extend((pair_assistant, pair_result))
            _assert_live_region_accepted(messages, tail[-1], pair_assistant)

    def test_marked_pair_colliding_with_the_preceding_assistant_is_refused(
        self,
    ) -> None:
        # Disjointness is kept even on MARKED assistants (design §10): a
        # marked pair reusing the anchor's id is not injection material.
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _injected_pair("c1", "skill")
        messages = [*tail, pair_assistant, pair_result]
        _assert_live_region_refused(messages, "c1")

    def test_earlier_injection_cannot_launder_a_later_response(self) -> None:
        # ``[T, I(k), RI(k), Unew, P(uuid4), RP(uuid4)]``: an earlier marked
        # injection does not launder a later UNMARKED provider response.
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "skill")
        provider_assistant, provider_result = _provider_pair(str(uuid4()))
        messages = [
            *tail,
            pair_assistant,
            pair_result,
            _user("a brand new user message"),
            provider_assistant,
            provider_result,
        ]
        _assert_live_region_refused(messages, "c1")

    def test_adjacent_duplicate_ids_with_the_second_assistant_marked_keep_an_empty_region(
        self,
    ) -> None:
        # The round-4 X4 adjacent-duplicate shape repeated with the SECOND
        # assistant MARKED: provenance does not bypass the disjointness key
        # — the walk itself keeps the region empty.
        second = _assistant("c2")
        second._context_injection_origin = CONTEXT_INJECTION_TOOL_PAIR_V1
        adjacent = [_assistant("c2"), second, _genuine_result("c2")]
        assert _live_tail_region_identities(adjacent) == frozenset()
        violations = validate_provider_ready(adjacent)
        assert [v.kind for v in violations] == ["unresolved_tool_call"]
        assert violations[0].tool_call_id == "c2"
        assert validate_transcript(adjacent, live_tail_exempt=True) != []
        assert find_persistence_violations(adjacent) != []
        with pytest.raises(TranscriptIntegrityError):
            messages_for_backend(adjacent, _model())

    def test_mixed_history_stops_at_the_older_unmarked_pair(self) -> None:
        # Mixed old/new restored history (design §8): a newer MARKED pair
        # does not confer provenance on an older UNMARKED settled pair —
        # the walk stops at the old pair, no anchor is found, the region is
        # empty, and the reused id is refused.
        tail = _live_reuse_tail()
        old_assistant, old_result = _provider_pair(str(uuid4()))
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "skill")
        messages = [*tail, old_assistant, old_result, pair_assistant, pair_result]
        _assert_live_region_refused(messages, "c1")


class TestRound5YRealInjectorProvenance:
    """Round-5 Y: the ACTUAL injector stamps provenance (design §3) and the
    ACTUAL session save persists it (design §7) — the real composed steer
    over the live-reuse tail, end to end.

    The injector marks only its own assistant message; the immediate save
    persists the session key on exactly that record; the trusted loader
    restores the marker BEFORE integrity validation, so the provenance-keyed
    walk accepts the composition and the resume install is a clean no-op.
    """

    @pytest.mark.asyncio
    async def test_real_composed_skill_steer_persists_and_restores_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = build_test_vibe_config(
            enabled_tools=["skill"],
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="y-steer", enabled=True
            ),
        )
        loop = build_test_agent_loop(
            config=config, backend=FakeBackend([[mock_llm_chunk(content="ok")]])
        )
        try:
            for message in _live_reuse_tail():
                loop.messages.append(message)
            _patch_skill_manager(loop, monkeypatch, name="y-skill")
            await loop.inject_user_context(
                "/y-skill", as_message=True, inject_implicit=True
            )
            transcript = list(loop.messages)
            # [.., A2(c1) in flight, Usteer, Askill(k), Rskill(k)]
            in_flight = transcript[-4]
            steered = transcript[-3]
            pair_assistant, pair_result = transcript[-2], transcript[-1]
            assert steered.role == Role.user
            assert steered.injected is False  # production flag values
            assert steered.content == "/y-skill"
            assert pair_assistant.role == Role.assistant
            assert pair_assistant.injected is False  # production flag values
            assert pair_assistant.tool_calls
            pair_call_id = pair_assistant.tool_calls[0].id
            assert pair_result.role == Role.tool
            assert has_structured_tool_result(pair_result)
            # THE PROVENANCE: only the injector's assistant carries the
            # private origin marker — not the steered user, not the result.
            assert (
                pair_assistant._context_injection_origin
                == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
            assert type(steered._context_injection_origin) is not str
            assert type(pair_result._context_injection_origin) is not str
            # The provenance-keyed walk accepts the composition.
            region = _live_tail_region_identities(transcript)
            assert region == {id(in_flight), id(pair_assistant)}
            assert find_persistence_violations(transcript) == []
            SessionLogger._check_transcript_persistence(transcript)
            await loop._save_messages()  # the actual save: no raise
            assert validate_transcript(transcript, live_tail_exempt=True) == []
            assert validate_provider_ready(transcript) == []
            # The ACTUAL saved record carries the provenance key on exactly
            # the injector's assistant record.
            session_dir = loop.session_logger.session_dir
            assert session_dir is not None
            records = _saved_records(session_dir)
            marked = [
                record
                for record in records
                if record.get(CONTEXT_INJECTION_ORIGIN_KEY) is not None
            ]
            assert len(marked) == 1
            assert marked[0]["role"] == "assistant"
            assert marked[0]["tool_calls"][0]["id"] == pair_call_id
            assert (
                marked[0][CONTEXT_INJECTION_ORIGIN_KEY]
                == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
            # The trusted loader restores the marker before validation.
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            loaded_assistants = [
                message
                for message in loaded
                if message.role == Role.assistant and message.tool_calls
            ]
            # [U, A1(c1), R1(c1), U, A2(c1), Usteer, I(k), RI(k)]
            anchor, restored_pair = loaded_assistants[1], loaded_assistants[2]
            assert anchor.tool_calls and restored_pair.tool_calls
            assert anchor.tool_calls[0].id == "c1"
            assert restored_pair.tool_calls[0].id == pair_call_id
            assert (
                restored_pair._context_injection_origin
                == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
            assert _live_tail_region_identities(loaded) == {
                id(anchor),
                id(restored_pair),
            }
            assert find_persistence_violations(loaded) == []
            assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
            assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        finally:
            await loop.aclose()

    @pytest.mark.asyncio
    async def test_real_composed_mention_steer_persists_and_restores_provenance(
        self, tmp_path: Path, tmp_working_directory: Path
    ) -> None:
        (tmp_working_directory / "notes.md").write_text("mention body")
        config = build_test_vibe_config(
            enabled_tools=["read_file"],
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="y-mention", enabled=True
            ),
        )
        loop = build_test_agent_loop(
            config=config,
            backend=FakeBackend([[mock_llm_chunk(content="ok")]]),
            cwd=tmp_working_directory,
        )
        try:
            for message in _live_reuse_tail():
                loop.messages.append(message)
            # The composed vector with a @mention: the steered user message,
            # then the implicit mention pair (read_file through the loop's
            # single-tool path), then the save.
            await loop.inject_user_context(
                "read @notes.md", as_message=True, inject_implicit=True
            )
            transcript = list(loop.messages)
            in_flight = transcript[-4]
            steered = transcript[-3]
            pair_assistant, pair_result = transcript[-2], transcript[-1]
            assert steered.role == Role.user
            assert steered.injected is False  # production flag values
            assert pair_assistant.role == Role.assistant
            assert pair_assistant.injected is False  # production flag values
            assert pair_assistant.tool_calls
            pair_call_id = pair_assistant.tool_calls[0].id
            assert pair_result.role == Role.tool
            assert pair_result.tool_call_id == pair_call_id
            assert pair_result.name == "read_file"
            assert "mention body" in (pair_result.content or "")
            assert has_structured_tool_result(pair_result)
            # THE PROVENANCE: the injector's assistant is marked; the
            # executor's result is not (a normal provider-issued read_file
            # call would stay unmarked too).
            assert (
                pair_assistant._context_injection_origin
                == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
            assert type(steered._context_injection_origin) is not str
            assert type(pair_result._context_injection_origin) is not str
            region = _live_tail_region_identities(transcript)
            assert region == {id(in_flight), id(pair_assistant)}
            assert find_persistence_violations(transcript) == []
            SessionLogger._check_transcript_persistence(transcript)
            await loop._save_messages()  # the actual save: no raise
            assert validate_transcript(transcript, live_tail_exempt=True) == []
            assert validate_provider_ready(transcript) == []
            session_dir = loop.session_logger.session_dir
            assert session_dir is not None
            records = _saved_records(session_dir)
            marked = [
                record
                for record in records
                if record.get(CONTEXT_INJECTION_ORIGIN_KEY) is not None
            ]
            assert len(marked) == 1
            assert marked[0]["role"] == "assistant"
            assert marked[0]["tool_calls"][0]["id"] == pair_call_id
            assert (
                marked[0][CONTEXT_INJECTION_ORIGIN_KEY]
                == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            restored = [
                message
                for message in loaded
                if message.role == Role.assistant
                and message.tool_calls
                and message.tool_calls[0].id == pair_call_id
            ]
            assert len(restored) == 1
            assert (
                restored[0]._context_injection_origin == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
            assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
            assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        finally:
            await loop.aclose()


class TestRound5YSessionProvenance:
    """Round-5 Y: the copy/serialization split (design §2, §7, §10) —
    ``model_copy`` preserves private provenance; ordinary ``model_dump``
    never emits it (either spelling); ordinary validation never restores
    it; the TRUSTED session load restores it.
    """

    def test_copy_and_serialization_split(self, tmp_path: Path) -> None:
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "skill")
        # Copies retain private provenance (shallow and deep).
        for copy in (pair_assistant.model_copy(), pair_assistant.model_copy(deep=True)):
            assert copy._context_injection_origin == CONTEXT_INJECTION_TOOL_PAIR_V1
        # Ordinary serialization NEVER emits the provenance (either
        # spelling) — the private attribute is absent without an exclusion.
        dump = pair_assistant.model_dump(exclude_none=True, mode="json")
        assert CONTEXT_INJECTION_ORIGIN_KEY not in dump
        assert "_context_injection_origin" not in dump
        dumped_json = pair_assistant.model_dump_json(exclude_none=True)
        assert CONTEXT_INJECTION_ORIGIN_KEY not in dumped_json
        assert "_context_injection_origin" not in dumped_json
        # Ordinary validation does NOT restore provenance — the spellings
        # are not model fields (``extra="ignore"``).
        round_tripped = LLMMessage.model_validate(dump)
        assert type(round_tripped._context_injection_origin) is not str
        forged = LLMMessage.model_validate({
            **dump,
            CONTEXT_INJECTION_ORIGIN_KEY: CONTEXT_INJECTION_TOOL_PAIR_V1,
            "_context_injection_origin": CONTEXT_INJECTION_TOOL_PAIR_V1,
        })
        assert type(forged._context_injection_origin) is not str
        extra: dict[str, Any] = {
            CONTEXT_INJECTION_ORIGIN_KEY: CONTEXT_INJECTION_TOOL_PAIR_V1
        }
        constructed = LLMMessage(
            role=Role.assistant,
            content="",
            tool_calls=pair_assistant.tool_calls,
            **extra,
        )
        assert type(constructed._context_injection_origin) is not str
        # The TRUSTED session load restores the marker (design §7).
        messages = [_user("hi"), pair_assistant, pair_result]
        session_dir = _write_session(tmp_path / "session_y_split", messages)
        loaded, metadata = SessionLoader.load_session(session_dir)
        assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
        assert loaded[1]._context_injection_origin == CONTEXT_INJECTION_TOOL_PAIR_V1
        assert type(loaded[0]._context_injection_origin) is not str
        assert type(loaded[2]._context_injection_origin) is not str
        assert validate_transcript(loaded, live_tail_exempt=True) == []
        assert validate_provider_ready(loaded) == []


class TestRound5YSaveModesAndMigration:
    """Round-5 Y: save modes and repair migration (design §7, §10, §12) —
    the initial save, the incremental append, the forced full verification,
    and the durable repaired-transcript migration ALL preserve recognized
    markers (every serialization path uses the explicit session record),
    and the migration preserves the original artifact.
    """

    @pytest.mark.asyncio
    async def test_save_modes_preserve_recognized_markers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = build_test_vibe_config(
            enabled_tools=["skill"],
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="y-modes", enabled=True
            ),
        )
        loop = build_test_agent_loop(
            config=config, backend=FakeBackend([[mock_llm_chunk(content="ok")]])
        )
        try:
            for message in _live_reuse_tail():
                loop.messages.append(message)
            _patch_skill_manager(loop, monkeypatch, name="y-skill")
            logger = loop.session_logger
            session_dir = logger.session_dir
            assert session_dir is not None
            messages_path = session_dir / MESSAGES_FILENAME
            # The initial save (the tail only): no provenance anywhere yet.
            await loop._save_messages()
            assert all(
                CONTEXT_INJECTION_ORIGIN_KEY not in record
                for record in _saved_records(session_dir)
            )
            # The direct skill injection appends the marked pair; the
            # INCREMENTAL save (the tail-serialization path) persists the key.
            before = len(loop.messages)
            async for _event in loop._inject_invoked_skill("/y-skill"):
                pass
            assert len(loop.messages) == before + 2
            pair_assistant = loop.messages[-2]
            assert pair_assistant.role == Role.assistant
            assert pair_assistant.tool_calls
            pair_call_id = pair_assistant.tool_calls[0].id
            assert (
                pair_assistant._context_injection_origin
                == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
            await loop._save_messages()
            marked = [
                record
                for record in _saved_records(session_dir)
                if record.get(CONTEXT_INJECTION_ORIGIN_KEY) is not None
            ]
            assert len(marked) == 1
            assert marked[0]["tool_calls"][0]["id"] == pair_call_id
            # The FORCED FULL-VERIFICATION save (cursor invalidated) keeps
            # the key on the same record.
            logger.invalidate_transcript_cursor()
            await loop._save_messages()
            marked = [
                record
                for record in _saved_records(session_dir)
                if record.get(CONTEXT_INJECTION_ORIGIN_KEY) is not None
            ]
            assert len(marked) == 1
            assert marked[0]["tool_calls"][0]["id"] == pair_call_id
            # A further save with no new messages is a no-op.
            size_before = messages_path.stat().st_size
            await loop._save_messages()
            assert messages_path.stat().st_size == size_before
            # The fingerprint is consistent: a copy fingerprints identically,
            # an unmarked message does not.
            assert SessionLogger._message_fingerprint(
                pair_assistant
            ) == SessionLogger._message_fingerprint(pair_assistant.model_copy())
            unmarked = pair_assistant.model_copy()
            unmarked._context_injection_origin = None
            assert SessionLogger._message_fingerprint(
                pair_assistant
            ) != SessionLogger._message_fingerprint(unmarked)
            # The saved transcript round-trips through the trusted loader.
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            restored = [
                message
                for message in loaded
                if message.role == Role.assistant
                and message.tool_calls
                and message.tool_calls[0].id == pair_call_id
            ]
            assert len(restored) == 1
            assert (
                restored[0]._context_injection_origin == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
        finally:
            await loop.aclose()

    def test_repaired_transcript_migration_preserves_markers_and_the_original(
        self, tmp_path: Path
    ) -> None:
        # The poison shape with a MARKED pair: the load repair drops the
        # shadowed placeholder, and the durable migration preserves the
        # marker on the surviving owner AND the original artifact.
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "skill")
        assert pair_assistant.tool_calls
        pair_call_id = pair_assistant.tool_calls[0].id
        messages = [
            _user("hi"),
            _assistant("c1"),
            _placeholder("c1"),
            _user(_STEER_TEXT),
            pair_assistant,
            pair_result,
            _genuine_result("c1"),
        ]
        session_dir = _write_session(tmp_path / "session_y_migrate", messages)
        messages_path = session_dir / MESSAGES_FILENAME
        original_bytes = messages_path.read_bytes()
        loaded, metadata = SessionLoader.load_session(session_dir)
        # The load repair dropped the shadowed placeholder and stamped the
        # report; the surviving pair kept its restored marker.
        report = report_from_metadata(metadata[TRANSCRIPT_INTEGRITY_METADATA_KEY])
        assert report is not None
        assert report.repaired is True
        assert report.dropped_placeholder_call_ids == ("c1",)
        assert report.violations == ()
        assert [message.role for message in loaded] == [
            Role.user,
            Role.assistant,
            Role.user,
            Role.assistant,
            Role.tool,
            Role.tool,
        ]
        restored_pair = [
            message
            for message in loaded
            if message.role == Role.assistant
            and message.tool_calls
            and message.tool_calls[0].id == pair_call_id
        ]
        assert len(restored_pair) == 1
        assert (
            restored_pair[0]._context_injection_origin == CONTEXT_INJECTION_TOOL_PAIR_V1
        )
        assert validate_transcript(loaded, live_tail_exempt=True) == []
        assert validate_provider_ready(loaded) == []
        # The write-owning install migrates durably: the ORIGINAL is
        # preserved as an artifact and the live path carries the marker.
        installed = _install_transcript_repair(session_dir, loaded, metadata)
        assert installed is loaded
        artifacts = list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        assert len(artifacts) == 1
        assert artifacts[0].read_bytes() == original_bytes
        migrated = _saved_records(session_dir)
        marked = [
            record
            for record in migrated
            if record.get(CONTEXT_INJECTION_ORIGIN_KEY) is not None
        ]
        assert len(marked) == 1
        assert marked[0]["tool_calls"][0]["id"] == pair_call_id
        assert marked[0][CONTEXT_INJECTION_ORIGIN_KEY] == CONTEXT_INJECTION_TOOL_PAIR_V1
        # A repeated load is clean and restores the marker again.
        reloaded, reloaded_metadata = SessionLoader.load_session(session_dir)
        assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in reloaded_metadata
        again = [
            message
            for message in reloaded
            if message.role == Role.assistant
            and message.tool_calls
            and message.tool_calls[0].id == pair_call_id
        ]
        assert len(again) == 1
        assert again[0]._context_injection_origin == CONTEXT_INJECTION_TOOL_PAIR_V1
        assert validate_transcript(reloaded, live_tail_exempt=True) == []
        assert len(list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))) == 1


class TestRound5YBackfillDirection:
    """Round-5 Y backfill direction (design §8, §12): the ordinary
    missing-response backfill settles the unanswered owner of the OLD
    composed shape with its own placeholder — it never invents provenance,
    and validation then succeeds.
    """

    def test_backfill_settles_the_unanswered_owner_without_inventing_provenance(
        self,
    ) -> None:
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _provider_pair(str(uuid4()))
        assert pair_assistant.tool_calls
        pair_call_id = pair_assistant.tool_calls[0].id
        messages = MessageList([
            LLMMessage(role=Role.system, content="system prompt"),
            *tail,
            _user(_STEER_TEXT),
            pair_assistant,
            pair_result,
        ])
        recorder, _ = _recorder(messages)
        recorder.repair_missing_responses()
        # A2 receives its OWN placeholder; no genuine result is deleted or
        # duplicated.
        placeholders = [m for m in messages if is_synthetic_tool_placeholder(m)]
        assert len(placeholders) == 1
        assert placeholders[0].tool_call_id == "c1"
        tool_messages = [m for m in messages if m.role == Role.tool]
        assert [m.tool_call_id for m in tool_messages] == ["c1", "c1", pair_call_id]
        assert tool_messages[0].tool_result is not None
        assert tool_messages[2].tool_result is not None
        # No provenance is invented: the old pair stays unmarked.
        assert type(pair_assistant._context_injection_origin) is not str
        # Validation then succeeds.
        assert validate_transcript(list(messages), live_tail_exempt=True) == []
        assert validate_provider_ready(list(messages)) == []


class TestRound5YCompactionProvenance:
    """Round-5 Y compaction (design §10): ordinary context selection
    preserves the retained message objects and their private attributes;
    the compaction boundary and the summary never acquire provenance.
    """

    def test_selected_context_preserves_markers_and_summaries_acquire_none(
        self,
    ) -> None:
        tail = _live_reuse_tail()
        pair_assistant, pair_result = _injected_pair(str(uuid4()), "skill")
        boundary = LLMMessage(
            role=Role.user,
            content="compacted context",
            injected=True,
            context_boundary="compaction",
        )
        summary = LLMMessage(
            role=Role.user, content="summarized earlier turns", injected=True
        )
        messages = [
            _user("hi"),
            _assistant("c1"),
            _genuine_result("c1"),
            boundary,
            summary,
            *tail,
            pair_assistant,
            pair_result,
        ]
        selected = select_model_context(messages)
        # The retained pair keeps its identity AND its provenance marker.
        assert any(message is pair_assistant for message in selected)
        retained = next(m for m in selected if m is pair_assistant)
        assert retained._context_injection_origin == CONTEXT_INJECTION_TOOL_PAIR_V1
        # The compaction boundary and the summary are retained too, and
        # neither acquires provenance.
        assert any(message is boundary for message in selected)
        assert any(message is summary for message in selected)
        assert type(boundary._context_injection_origin) is not str
        assert type(summary._context_injection_origin) is not str
        # The selected context keeps the live region and validates.
        assistants = [
            message
            for message in selected
            if message.role == Role.assistant and message.tool_calls
        ]
        anchor, pair = assistants[-2], assistants[-1]
        assert anchor.tool_calls and pair.tool_calls and pair_assistant.tool_calls
        assert anchor.tool_calls[0].id == "c1"
        assert pair.tool_calls[0].id == pair_assistant.tool_calls[0].id
        assert _live_tail_region_identities(selected) == {id(anchor), id(pair)}
        assert validate_transcript(selected, live_tail_exempt=True) == []
        assert validate_provider_ready(selected) == []
