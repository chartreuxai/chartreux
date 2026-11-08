"""P3 (S1-B) pre-seal payload release — unit and integration tests.

C1 (prefix ``test_pre_seal_unit_``): the PRODUCTION helpers run their real
code in every case — ``eligible_pre_seal_prefix``, the
``EligiblePreSealPrefix`` descriptor construction, and the seam-independent
progressive keying functions. No mocks stand in for the units under test
(the helpers are pure functions of their inputs, so no collaborator mocks
are needed at this level).

C2 (prefix ``test_pre_seal_integration_``): the real-logger seam — the
pre-seal plan arm, the widened attempt keys, the coalescing rule (post-seal
alone once a real boundary exists) — and publication through
``MessageList.publish_archive_trim``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chartreux.core.agent_loop._transcript_integrity import (
    archive_reference,
    validate_transcript,
)
from chartreux.core.compaction import CompactionManager
from chartreux.core.compaction.context import select_model_context
from chartreux.core.llm_models import FunctionCall, LLMChunk, LLMMessage, Role, ToolCall
from chartreux.core.message_list import MessageList
from chartreux.core.rewind import (
    RewindArchiveError,
    archived_history_for,
    complete_history,
)
from chartreux.core.session.session_loader import MESSAGES_FILENAME
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session.transcript_archive import (
    PRE_SEAL_RETAIN_RECENT_USER_ROUNDS,
    EligiblePreSealPrefix,
    eligible_pre_seal_prefix,
    pre_seal_endpoint,
    pre_seal_release_key,
    release_key_is_new,
    trim_attempt_key,
    verify_archived_prefix_on_disk,
)
from chartreux.core.session_types import AgentStats
from chartreux.core.tools.manager import ToolManager
from chartreux.core.usage import UsagePurpose
from tests.conftest import build_test_vibe_config, make_test_models
from tests.mock.utils import mock_llm_chunk
from tests.session.persistence._session_persistence_support import (
    mock_tool_manager as mock_tool_manager,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)


def _user(content: str, message_id: str | None = None) -> LLMMessage:
    return LLMMessage(role=Role.user, content=content, message_id=message_id)


def _assistant(
    content: str,
    message_id: str | None = None,
    payload: bool = False,
    tool_call_id: str | None = None,
) -> LLMMessage:
    return LLMMessage(
        role=Role.assistant,
        content=content,
        message_id=message_id,
        reasoning_payloads=[{"opaque": f"payload-{message_id or content}"}]
        if payload
        else None,
        tool_calls=[
            ToolCall(
                id=tool_call_id,
                index=0,
                function=FunctionCall(name="tool", arguments="{}"),
            )
        ]
        if tool_call_id
        else None,
    )


def _tool_result(call_id: str) -> LLMMessage:
    return LLMMessage(role=Role.tool, tool_call_id=call_id, content="result")


class TestPreSealUnitEligibilityRule:
    """The eligibility rule on synthetic round shapes (production helper)."""

    def test_pre_seal_unit_eligibility_needs_more_than_the_retained_rounds(
        self,
    ) -> None:
        # One or two user-started rounds: the newest two cover everything
        # before the older of them, so there is no release candidate.
        assert (
            eligible_pre_seal_prefix([_user("u1"), _assistant("a1", payload=True)])
            is None
        )
        assert (
            eligible_pre_seal_prefix([
                _user("u1", "u1"),
                _assistant("a1", "a1", payload=True),
                _user("u2", "u2"),
            ])
            is None
        )

    def test_pre_seal_unit_eligibility_releases_prefix_before_older_of_two_newest(
        self,
    ) -> None:
        messages = [
            _user("u1", "u1"),
            _assistant("a1", "a1", payload=True),
            _user("u2", "u2"),
            _assistant("a2", "a2", payload=True),
            _user("u3", "u3"),
            _assistant("a3", "a3", payload=True),
        ]
        prefix = eligible_pre_seal_prefix(messages)
        assert prefix is not None
        # The newest two user-started rounds ([u2, a2] and [u3, a3]) are
        # retained in full; only the strictly-older settled prefix releases.
        assert [message.message_id for message in prefix.messages] == ["u1", "a1"]
        assert prefix.endpoint_message_id == "a1"
        assert PRE_SEAL_RETAIN_RECENT_USER_ROUNDS == 2

    def test_pre_seal_unit_eligibility_trims_back_to_last_settled_round(self) -> None:
        # The candidate region holds a settled round, then an UNSETTLED one
        # (assistant tool call left unanswered): the prefix ends at the
        # settled round.
        messages = [
            _user("u1", "u1"),
            _assistant("a1", "a1", payload=True, tool_call_id="c1"),
            _tool_result("c1"),
            _user("u2", "u2"),
            _assistant("a2", "a2", tool_call_id="c2"),  # open call, no result
            _user("u3", "u3"),
            _assistant("a3", "a3", payload=True),
        ]
        prefix = eligible_pre_seal_prefix(messages)
        assert prefix is not None
        assert [message.message_id for message in prefix.messages] == [
            "u1",
            "a1",
            None,  # the tool result carries no message id
        ]
        assert prefix.endpoint_message_id is None

    def test_pre_seal_unit_eligibility_requires_an_eligible_payload(self) -> None:
        messages = [
            _user("u1", "u1"),
            _assistant("a1", "a1"),  # no reasoning payload
            _user("u2", "u2"),
            _assistant("a2", "a2"),
            _user("u3", "u3"),
            _assistant("a3", "a3", payload=True),  # payload only in a RETAINED round
        ]
        assert eligible_pre_seal_prefix(messages) is None

    def test_pre_seal_unit_eligibility_ignores_system_messages(self) -> None:
        messages = [
            LLMMessage(role=Role.system, content="sys"),
            _user("u1", "u1"),
            _assistant("a1", "a1", payload=True),
            LLMMessage(role=Role.system, content="sys2"),
            _user("u2", "u2"),
            _assistant("a2", "a2", payload=True),
            _user("u3", "u3"),
            _assistant("a3", "a3", payload=True),
        ]
        prefix = eligible_pre_seal_prefix(messages)
        assert prefix is not None
        assert [message.message_id for message in prefix.messages] == ["u1", "a1"]

    def test_pre_seal_unit_eligibility_empty_sequences(self) -> None:
        assert eligible_pre_seal_prefix([]) is None
        assert eligible_pre_seal_prefix([_user("u1")]) is None


class TestPreSealUnitDescriptor:
    """The EligiblePreSealPrefix descriptor construction (production type)."""

    def test_pre_seal_unit_descriptor_valid_construction(self) -> None:
        prefix = EligiblePreSealPrefix(
            messages=(_user("u1", "u1"), _assistant("a1", "a1")),
            endpoint_message_id="a1",
        )
        assert prefix.messages[1].message_id == "a1"
        assert prefix.endpoint_message_id == "a1"

    def test_pre_seal_unit_descriptor_rejects_invalid_shapes(self) -> None:
        good = (_user("u1", "u1"),)
        with pytest.raises(ValueError, match="non-empty tuple"):
            EligiblePreSealPrefix(messages=(), endpoint_message_id=None)
        with pytest.raises(ValueError, match="non-empty tuple"):
            EligiblePreSealPrefix(messages=[_user("u1")], endpoint_message_id=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="LLMMessage"):
            EligiblePreSealPrefix(messages=("not-a-message",), endpoint_message_id=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="system"):
            EligiblePreSealPrefix(
                messages=(LLMMessage(role=Role.system, content="s"),),
                endpoint_message_id=None,
            )
        with pytest.raises(ValueError, match="endpoint_message_id"):
            EligiblePreSealPrefix(messages=good, endpoint_message_id="")


class TestPreSealUnitKeying:
    """The progressive keying mechanics (production pure functions)."""

    def test_pre_seal_unit_keying_endpoint_is_stable_per_prefix(self) -> None:
        prefix = EligiblePreSealPrefix(
            messages=(_user("u1", "u1"), _assistant("a1", "a1")),
            endpoint_message_id="a1",
        )
        assert pre_seal_endpoint(prefix) == "2:a1"
        no_id = EligiblePreSealPrefix(
            messages=(_user("u1"), _tool_result("c1")), endpoint_message_id=None
        )
        assert pre_seal_endpoint(no_id) == "2:-"

    def test_pre_seal_unit_keying_advancing_prefixes_yield_new_keys(self) -> None:
        first = pre_seal_release_key("s1", 0, "2:a1")
        second = pre_seal_release_key("s1", 0, "4:a2")
        assert first != second
        assert first[0] == "pre-seal"
        published: set = set()
        assert release_key_is_new(published, first)
        assert release_key_is_new(published, second)
        published.add(first)
        assert release_key_is_new(published, second)
        assert not release_key_is_new(published, first)

    def test_pre_seal_unit_keying_duplicate_suppressed(self) -> None:
        key = pre_seal_release_key("s1", 0, "2:a1")
        published = {key}
        assert not release_key_is_new(published, key)

    def test_pre_seal_unit_keying_retry_after_failure_allowed(self) -> None:
        # A failed attempt never publishes its key, so the same key is new
        # again on retry.
        key = pre_seal_release_key("s1", 0, "2:a1")
        assert release_key_is_new(set(), key)
        published: set = set()
        assert release_key_is_new(published, key)
        published.add(key)  # only a SUCCESSFUL publication lands here
        assert not release_key_is_new(published, key)

    def test_pre_seal_unit_keying_transition_to_real_boundary(self) -> None:
        pre_key = trim_attempt_key("s1", 0, None, "2:a1")
        boundary_key = trim_attempt_key("s1", 0, "boundary-1", "2:a1")
        assert pre_key == ("pre-seal", "s1", 0, "2:a1")
        assert boundary_key == ("boundary", "s1", 0, "boundary-1")
        # Once a real boundary exists, its key supersedes the pre-seal key.
        published = {pre_key}
        assert release_key_is_new(published, boundary_key)
        assert not release_key_is_new(published, pre_key)

    def test_pre_seal_unit_keying_boundary_none_refusal_no_bare_none_key(self) -> None:
        with pytest.raises(ValueError, match="endpoint"):
            trim_attempt_key("s1", 0, None, None)
        with pytest.raises(ValueError, match="endpoint"):
            trim_attempt_key("s1", 0, None, "")
        with pytest.raises(ValueError, match="endpoint"):
            pre_seal_release_key("s1", 0, "")
        with pytest.raises(ValueError, match="session_id"):
            pre_seal_release_key("", 0, "2:a1")
        with pytest.raises(ValueError, match="message_epoch"):
            pre_seal_release_key("s1", -1, "2:a1")
        with pytest.raises(ValueError, match="boundary_message_id"):
            trim_attempt_key("s1", 0, "", "2:a1")


# ---------------------------------------------------------------------------
# C2 — integration through the real logger seam.
# ---------------------------------------------------------------------------


async def _noop() -> None:
    return None


async def _complete(
    *,
    model: Any,
    messages: Any,
    tools: Any = None,
    tool_choice: Any = None,
    call_type: str | None = None,
    account_conversation: bool = False,
    purpose: UsagePurpose = UsagePurpose.CONVERSATION,
) -> LLMChunk:
    """Stand-in for ``AgentLoop._complete``: the pre-seal seam never calls it."""
    return mock_llm_chunk()


def _integration_config() -> Any:
    return build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=999),
        raise_on_compaction_failure=False,
    )


def _round(
    user_id: str, ask: str, assistant_id: str, answer: str, payload: bool = False
) -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.user, content=ask, message_id=user_id),
        LLMMessage(
            role=Role.assistant,
            content=answer,
            message_id=assistant_id,
            reasoning_payloads=[{"opaque": f"payload-{assistant_id}"}]
            if payload
            else None,
        ),
    ]


def _pre_seal_conversation() -> MessageList:
    """Three settled user rounds, no compaction boundary, payloads in the
    two oldest assistant answers (the oldest round is the eligible prefix).
    """
    return MessageList([
        LLMMessage(role=Role.system, content="sys"),
        *_round("u1", "oldest ask", "a1", "work", payload=True),
        *_round("u2", "middle ask", "a2", "work2", payload=True),
        *_round("u3", "newest ask", "a3", "work3"),
    ])


def _boundary(content: str = "compaction boundary") -> LLMMessage:
    return LLMMessage(
        role=Role.user, content=content, injected=True, context_boundary="compaction"
    )


def _messages_path(logger: SessionLogger) -> Path:
    session_dir = logger.session_dir
    assert session_dir is not None
    return session_dir / MESSAGES_FILENAME


def _non_system(messages: MessageList) -> list[LLMMessage]:
    return [message for message in messages if message.role is not Role.system]


def _prefix_endpoint(messages: MessageList) -> str:
    prefix = eligible_pre_seal_prefix(list(messages))
    assert prefix is not None
    return pre_seal_endpoint(prefix)


class _Box:
    def __init__(self) -> None:
        self.value: Any = None

    def getter(self) -> Any:
        return self.value

    def recorder(self, pending: Any) -> None:
        self.value = pending


def _build_manager(
    logger: SessionLogger,
    messages: MessageList,
    stats: AgentStats,
    tool_manager: ToolManager,
    box: _Box,
) -> CompactionManager:
    return CompactionManager(
        messages=messages,
        stats_getter=lambda: stats,
        config_getter=_integration_config,
        complete=_complete,
        available_tools=lambda: [],
        tool_choice=lambda: "auto",
        save=_noop,
        session_logger_getter=lambda: logger,
        tool_manager_getter=lambda: tool_manager,
        has_active_turn=lambda: True,
        trim_enabled_getter=lambda: True,
        pending_trim_getter=box.getter,
        pending_trim_recorder=box.recorder,
    )


async def _seam(
    logger: SessionLogger,
    messages: MessageList,
    stats: AgentStats,
    tool_manager: ToolManager,
    session_id: str,
) -> _Box:
    box = _Box()
    manager = _build_manager(logger, messages, stats, tool_manager, box)
    await manager.capture_post_turn_pending_trim(
        session_id=session_id, message_epoch=messages.message_epoch
    )
    return box


class TestPreSealIntegrationSeam:
    """The real-logger seam: plan arm + widened keys + publication through
    ``publish_archive_trim``.
    """

    @pytest.mark.asyncio
    async def test_pre_seal_integration_release_publishes_cover_all_descriptor(
        self, session_config: Any, mock_tool_manager: ToolManager
    ) -> None:
        session_id = "p3-seam"
        logger = SessionLogger(session_config, session_id)
        messages = _pre_seal_conversation()
        stats = AgentStats()
        config = _integration_config()
        # The durable transcript exists before the seam (saves precede any
        # trim seam): the whole conversation is persisted with its payloads.
        await logger.save_interaction(messages, stats, config, mock_tool_manager)
        expected_key = trim_attempt_key(
            session_id, messages.message_epoch, None, _prefix_endpoint(messages)
        )

        await _seam(logger, messages, stats, mock_tool_manager, session_id)

        descriptor = messages.archive_descriptor
        assert descriptor is not None
        # The pre-seal descriptor is boundary-None and covers the whole
        # non-system snapshot (released prefix + retained rounds).
        assert descriptor.boundary_message_id is None
        assert descriptor.archived_count == len(_non_system(messages)) == 6
        assert descriptor.trim_mode == "payloads"
        assert logger.archive_descriptor is descriptor
        # The progressive pre-seal key (never a bare None) is published.
        assert expected_key in logger._trim_published_boundaries
        assert expected_key[0] == "pre-seal"
        assert logger._trim_in_flight_boundaries == set()
        # Publication went through publish_archive_trim: the released prefix
        # slots are payload-trimmed, the retained rounds keep their payloads.
        non_system = _non_system(messages)
        assert non_system[0].reasoning_payloads is None  # u1 never had any
        assert non_system[1].reasoning_payloads is None  # a1 RELEASED
        assert non_system[3].reasoning_payloads == [{"opaque": "payload-a2"}]
        # The archived prefix on disk verifies against the descriptor.
        verify_archived_prefix_on_disk(descriptor, _messages_path(logger))

    @pytest.mark.asyncio
    async def test_pre_seal_integration_model_visibility_after_release(
        self, session_config: Any, mock_tool_manager: ToolManager
    ) -> None:
        session_id = "p3-visible"
        logger = SessionLogger(session_config, session_id)
        messages = _pre_seal_conversation()
        stats = AgentStats()
        config = _integration_config()
        await logger.save_interaction(messages, stats, config, mock_tool_manager)
        contents_before = [message.content for message in _non_system(messages)]

        await _seam(logger, messages, stats, mock_tool_manager, session_id)

        # Request-build shape: the model context (no compaction boundary, so
        # the whole list) validates WITHOUT any archive authority, and the
        # released slots still carry their model-visible content.
        context = select_model_context(list(messages))
        assert validate_transcript(context, live_tail_exempt=True) == []
        assert [
            message.content for message in context if message.role is not Role.system
        ] == contents_before
        assert _non_system(messages)[1].content == "work"

    @pytest.mark.asyncio
    async def test_pre_seal_integration_memory_retention_disk_keeps_ram_drops(
        self, session_config: Any, mock_tool_manager: ToolManager
    ) -> None:
        # The disk-keeps/RAM-drops pattern (generation-fence :709-716): the
        # persisted record of a released slot still carries its payload on
        # disk, the RAM replacement does not, and a later save leaves the
        # archived prefix bytes byte-identical.
        session_id = "p3-retention"
        logger = SessionLogger(session_config, session_id)
        messages = _pre_seal_conversation()
        stats = AgentStats()
        config = _integration_config()
        await logger.save_interaction(messages, stats, config, mock_tool_manager)

        await _seam(logger, messages, stats, mock_tool_manager, session_id)

        descriptor = messages.archive_descriptor
        assert descriptor is not None
        path = _messages_path(logger)
        prefix_bytes = path.read_bytes()[: descriptor.prefix_end_byte]
        records = [json.loads(line) for line in prefix_bytes.splitlines()]
        # Disk keeps the payload of the released assistant slot (record 1)...
        assert records[1]["reasoning_payloads"] == [{"opaque": "payload-a1"}]
        assert records[1]["role"] == "assistant"
        # ...while the RAM slot dropped it.
        assert _non_system(messages)[1].reasoning_payloads is None
        # A later ordinary save keeps the archived prefix byte-identical.
        messages.append(LLMMessage(role=Role.user, content="later", message_id="u4"))
        await logger.save_interaction(messages, stats, config, mock_tool_manager)
        assert path.read_bytes()[: descriptor.prefix_end_byte] == prefix_bytes
        assert logger.last_save_diagnostic is None

    @pytest.mark.asyncio
    async def test_pre_seal_integration_intact_rewind_over_a_released_prefix(
        self, session_config: Any, mock_tool_manager: ToolManager
    ) -> None:
        session_id = "p3-rewind"
        logger = SessionLogger(session_config, session_id)
        messages = _pre_seal_conversation()
        stats = AgentStats()
        config = _integration_config()
        await logger.save_interaction(messages, stats, config, mock_tool_manager)
        await _seam(logger, messages, stats, mock_tool_manager, session_id)
        assert _non_system(messages)[1].reasoning_payloads is None

        source = archived_history_for(messages, logger)
        assert source is not None
        complete = complete_history(list(messages), source)
        hydrated = next(message for message in complete if message.message_id == "a1")
        # The released prefix hydrates INTACT from the verified disk
        # transcript: the rewind consumer never sees a trimmed payload.
        assert hydrated.reasoning_payloads == [{"opaque": "payload-a1"}]
        assert hydrated.content == "work"

    @pytest.mark.asyncio
    async def test_pre_seal_integration_hydration_failure_fails_closed(
        self, session_config: Any, mock_tool_manager: ToolManager
    ) -> None:
        session_id = "p3-failclosed"
        logger = SessionLogger(session_config, session_id)
        messages = _pre_seal_conversation()
        stats = AgentStats()
        config = _integration_config()
        await logger.save_interaction(messages, stats, config, mock_tool_manager)
        await _seam(logger, messages, stats, mock_tool_manager, session_id)
        path = _messages_path(logger)
        descriptor = messages.archive_descriptor
        assert descriptor is not None

        # A corrupt archived prefix fails closed with the typed error.
        raw = bytearray(path.read_bytes())
        raw[10] ^= 0xFF
        path.write_bytes(bytes(raw))
        source = archived_history_for(messages, logger)
        assert source is not None
        with pytest.raises(RewindArchiveError):
            source.archived_messages()

        # A missing transcript fails closed the same way.
        path.unlink()
        source = archived_history_for(messages, logger)
        assert source is not None
        with pytest.raises(RewindArchiveError):
            source.archived_messages()

    @pytest.mark.asyncio
    async def test_pre_seal_integration_progressive_repeated_release(
        self, session_config: Any, mock_tool_manager: ToolManager
    ) -> None:
        session_id = "p3-progressive"
        logger = SessionLogger(session_config, session_id)
        messages = _pre_seal_conversation()
        stats = AgentStats()
        config = _integration_config()
        await logger.save_interaction(messages, stats, config, mock_tool_manager)
        first_key = trim_attempt_key(
            session_id, messages.message_epoch, None, _prefix_endpoint(messages)
        )
        await _seam(logger, messages, stats, mock_tool_manager, session_id)
        first_descriptor = messages.archive_descriptor
        assert first_descriptor is not None and first_descriptor.archived_count == 6

        # Another settled round arrives (with a payload): the eligible prefix
        # advances and a second release extends the cover-all descriptor.
        messages.extend(_round("u4", "fourth ask", "a4", "work4", payload=True))
        await logger.save_interaction(messages, stats, config, mock_tool_manager)
        second_key = trim_attempt_key(
            session_id, messages.message_epoch, None, _prefix_endpoint(messages)
        )
        assert second_key != first_key
        await _seam(logger, messages, stats, mock_tool_manager, session_id)

        second_descriptor = messages.archive_descriptor
        assert second_descriptor is not None
        assert second_descriptor.archived_count == 8
        assert second_descriptor.prefix_end_byte > first_descriptor.prefix_end_byte
        assert second_descriptor.boundary_message_id is None
        # Both progressive keys are published (advancing prefixes).
        assert first_key in logger._trim_published_boundaries
        assert second_key in logger._trim_published_boundaries
        # The advanced prefix is released; the newest two rounds stay full.
        non_system = _non_system(messages)
        assert [m.reasoning_payloads is None for m in non_system[:4]] == [True] * 4
        assert non_system[7].reasoning_payloads == [{"opaque": "payload-a4"}]
        verify_archived_prefix_on_disk(second_descriptor, _messages_path(logger))

    @pytest.mark.asyncio
    async def test_pre_seal_integration_repeat_seam_without_growth_is_suppressed(
        self,
        session_config: Any,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Duplicate suppression at the seam: with the list unchanged, the
        # released prefix carries no payload in RAM anymore, so the second
        # seam attempt records nothing and never re-seals.
        session_id = "p3-duplicate"
        logger = SessionLogger(session_config, session_id)
        messages = _pre_seal_conversation()
        stats = AgentStats()
        config = _integration_config()
        await logger.save_interaction(messages, stats, config, mock_tool_manager)
        await _seam(logger, messages, stats, mock_tool_manager, session_id)
        descriptor = messages.archive_descriptor
        published = set(logger._trim_published_boundaries)

        seal_calls = 0
        real_seal = SessionLogger.seal_archive

        async def _seal_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
            nonlocal seal_calls
            seal_calls += 1
            return await real_seal(self, *args, **kwargs)

        monkeypatch.setattr(SessionLogger, "seal_archive", _seal_spy)
        box = await _seam(logger, messages, stats, mock_tool_manager, session_id)

        assert seal_calls == 0
        assert box.value is None
        assert messages.archive_descriptor is descriptor
        assert set(logger._trim_published_boundaries) == published

    @pytest.mark.asyncio
    async def test_pre_seal_integration_coalesces_with_post_seal(
        self, session_config: Any, mock_tool_manager: ToolManager
    ) -> None:
        # Coalescing rule: once a real compaction boundary exists, post-seal
        # alone owns the release — the seam publishes the boundary-keyed
        # descriptor (marked replacements), superseding the pre-seal key.
        session_id = "p3-coalesce"
        logger = SessionLogger(session_config, session_id)
        messages = _pre_seal_conversation()
        stats = AgentStats()
        config = _integration_config()
        await logger.save_interaction(messages, stats, config, mock_tool_manager)
        pre_key = trim_attempt_key(
            session_id, messages.message_epoch, None, _prefix_endpoint(messages)
        )
        await _seam(logger, messages, stats, mock_tool_manager, session_id)
        assert pre_key in logger._trim_published_boundaries
        pre_descriptor = messages.archive_descriptor
        assert pre_descriptor is not None and pre_descriptor.boundary_message_id is None

        boundary = _boundary()
        messages.append(boundary)
        await logger.save_interaction(messages, stats, config, mock_tool_manager)
        await _seam(logger, messages, stats, mock_tool_manager, session_id)

        descriptor = messages.archive_descriptor
        assert descriptor is not None
        assert descriptor.boundary_message_id == boundary.message_id
        boundary_key = trim_attempt_key(
            session_id, messages.message_epoch, boundary.message_id, None
        )
        assert boundary_key in logger._trim_published_boundaries
        assert boundary_key[0] == "boundary"
        # Post-seal publication: the sealed slots carry the archive marker.
        sealed = _non_system(messages)[: descriptor.archived_count]
        assert all(archive_reference(message) is descriptor for message in sealed)
        assert _non_system(messages)[1].reasoning_payloads is None
        verify_archived_prefix_on_disk(descriptor, _messages_path(logger))
