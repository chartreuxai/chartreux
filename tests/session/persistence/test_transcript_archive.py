"""P4a1/P4a2: archive descriptor, its ownership/retention, the shared archive-read
interface, and the sealing operation (eligible-prefix rule + durable receipt).

Covers the P4a1 row's invariants: descriptor construction/validation; epoch
semantics (reset/rebind changes the epoch, cursor invalidation does not —
including the rebind path exercised by tests/app_server/test_local_resume.py);
the descriptor retained independently of _TranscriptCursor; save snapshots
capturing the descriptor WITH the message epoch; and the module's exported
archive-READ interface over injected descriptors + archived bytes. The
per-message archive reference is deliberately NOT here (it lands in P4b).

P4a2 adds: the eligible-prefix rule (strictly before the newest boundary;
excluding system messages; ending at a complete settled round; excluding the
active invocation), the receipt-producing seal under the save lock (strict
pre-seal validation of the closed prefix, receipt only after durable persist
and on-disk prefix verification, never on a caught-and-continued failure),
and the two P4a1 open questions: a non-authoritative save snapshot fails only
at sealing (not at save time), and the retained descriptor's session-id check
is re-validated on logger rebind (reset_session / apply_resumed_session).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from typing import Any

import pytest

from chartreux.core.agent_loop._transcript_integrity import (
    ARCHIVE_REFERENCE_ATTRIBUTE,
    TranscriptIntegrityError,
)
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.message_list import MessageList
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session.transcript_archive import (
    ARCHIVE_DESCRIPTOR_VERSION,
    ArchivedPrefixV1,
    ArchiveIntegrityError,
    ArchiveSaveSnapshot,
    ArchiveSealError,
    ArchiveSealReceipt,
    eligible_archive_prefix,
    hydrate_archived_messages,
    iter_archived_records,
    read_archived_records,
    verify_archived_prefix_on_disk,
)
from chartreux.core.session_types import AgentStats
from chartreux.core.tools.manager import ToolManager
from tests.session.persistence._session_persistence_support import (
    disabled_session_config as disabled_session_config,
    mock_tool_manager as mock_tool_manager,
    mock_vibe_config as mock_vibe_config,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)


def _record_line(message: LLMMessage) -> bytes:
    """The exact persisted transcript line for one message."""
    record = SessionLogger._message_record(message)
    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


def _make_descriptor(
    session_id: str,
    messages: list[LLMMessage],
    *,
    message_epoch: int = 0,
    boundary_message_id: str | None = "boundary-1",
    trim_mode: Any = "payloads",
) -> tuple[ArchivedPrefixV1, bytes]:
    """An injected descriptor paired with its exact archived bytes."""
    archived_bytes = b"".join(_record_line(m) for m in messages)
    descriptor = ArchivedPrefixV1(
        session_id=session_id,
        message_epoch=message_epoch,
        archived_count=len(messages),
        prefix_end_byte=len(archived_bytes),
        prefix_sha256=hashlib.sha256(archived_bytes).hexdigest(),
        boundary_message_id=boundary_message_id,
        trim_mode=trim_mode,
    )
    return descriptor, archived_bytes


def _base_kwargs() -> dict[str, Any]:
    return {
        "session_id": "sess-1",
        "message_epoch": 0,
        "archived_count": 1,
        "prefix_end_byte": 10,
        "prefix_sha256": hashlib.sha256(b"0123456789").hexdigest(),
        "boundary_message_id": "boundary-1",
        "trim_mode": "payloads",
    }


class TestDescriptorConstruction:
    def test_valid_descriptor_round_trips_its_fields(self) -> None:
        kwargs = _base_kwargs()
        descriptor = ArchivedPrefixV1(**kwargs)
        assert descriptor.session_id == "sess-1"
        assert descriptor.message_epoch == 0
        assert descriptor.archived_count == 1
        assert descriptor.prefix_end_byte == 10
        assert descriptor.prefix_sha256 == kwargs["prefix_sha256"]
        assert descriptor.boundary_message_id == "boundary-1"
        assert descriptor.trim_mode == "payloads"
        assert descriptor.version == ARCHIVE_DESCRIPTOR_VERSION

    def test_descriptor_is_frozen(self) -> None:
        descriptor = ArchivedPrefixV1(**_base_kwargs())
        with pytest.raises(FrozenInstanceError):
            descriptor.session_id = "other"  # type: ignore[misc]

    def test_empty_archive_is_a_valid_descriptor(self) -> None:
        descriptor = ArchivedPrefixV1(
            session_id="s",
            message_epoch=0,
            archived_count=0,
            prefix_end_byte=0,
            prefix_sha256=hashlib.sha256(b"").hexdigest(),
            boundary_message_id=None,
            trim_mode="tombstones",
        )
        assert descriptor.archived_count == 0
        assert descriptor.boundary_message_id is None
        assert descriptor.trim_mode == "tombstones"

    @pytest.mark.parametrize(
        "field,value",
        [
            ("session_id", ""),
            ("session_id", None),
            ("message_epoch", -1),
            ("message_epoch", True),
            ("message_epoch", "0"),
            ("archived_count", -1),
            ("archived_count", 1.5),
            ("prefix_end_byte", -1),
            ("prefix_sha256", "tooshort"),
            ("prefix_sha256", "A" * 64),
            ("prefix_sha256", "z" * 64),
            ("boundary_message_id", ""),
            ("trim_mode", "everything"),
            ("trim_mode", None),
        ],
    )
    def test_invalid_field_is_rejected(self, field: str, value: Any) -> None:
        kwargs = _base_kwargs()
        kwargs[field] = value
        with pytest.raises(ValueError):
            ArchivedPrefixV1(**kwargs)

    def test_nonzero_count_with_zero_end_byte_is_rejected(self) -> None:
        kwargs = _base_kwargs()
        kwargs["prefix_end_byte"] = 0
        with pytest.raises(ValueError):
            ArchivedPrefixV1(**kwargs)

    def test_zero_count_with_nonzero_end_byte_is_rejected(self) -> None:
        kwargs = _base_kwargs()
        kwargs["archived_count"] = 0
        with pytest.raises(ValueError):
            ArchivedPrefixV1(**kwargs)


class TestArchiveReadInterface:
    def test_read_archived_records_over_injected_descriptor_and_bytes(self) -> None:
        messages = [
            LLMMessage(role=Role.user, content="hello", message_id="u1"),
            LLMMessage(role=Role.assistant, content="hi there", message_id="a1"),
        ]
        descriptor, archived_bytes = _make_descriptor("sess-1", messages)
        records = read_archived_records(descriptor, archived_bytes)
        assert records == [SessionLogger._message_record(m) for m in messages]

    def test_hydrate_archived_messages_round_trips(self) -> None:
        messages = [
            LLMMessage(role=Role.user, content="hello", message_id="u1"),
            LLMMessage(role=Role.assistant, content="hi there", message_id="a1"),
        ]
        descriptor, archived_bytes = _make_descriptor("sess-1", messages)
        hydrated = hydrate_archived_messages(descriptor, archived_bytes)
        assert [m.message_id for m in hydrated] == ["u1", "a1"]
        assert [m.role for m in hydrated] == [Role.user, Role.assistant]
        assert [m.content for m in hydrated] == ["hello", "hi there"]

    def test_iter_archived_records_streams_chunked_bytes(self) -> None:
        messages = [
            LLMMessage(role=Role.user, content="chunked " * 5, message_id="u1"),
            LLMMessage(role=Role.assistant, content="reply", message_id="a1"),
            LLMMessage(role=Role.user, content="again", message_id="u2"),
        ]
        descriptor, archived_bytes = _make_descriptor("sess-1", messages)
        # 7-byte chunks split lines mid-record; the reader reassembles them.
        chunks = [archived_bytes[i : i + 7] for i in range(0, len(archived_bytes), 7)]
        streamed = list(iter_archived_records(descriptor, chunks))
        assert streamed == read_archived_records(descriptor, archived_bytes)

    def test_empty_archive_reads_as_no_records(self) -> None:
        descriptor = ArchivedPrefixV1(
            session_id="s",
            message_epoch=0,
            archived_count=0,
            prefix_end_byte=0,
            prefix_sha256=hashlib.sha256(b"").hexdigest(),
            boundary_message_id=None,
            trim_mode="payloads",
        )
        assert read_archived_records(descriptor, b"") == []
        assert hydrate_archived_messages(descriptor, b"") == []

    def test_tampered_bytes_fail_closed_on_hash(self) -> None:
        descriptor, archived_bytes = _make_descriptor(
            "sess-1", [LLMMessage(role=Role.user, content="hello", message_id="u1")]
        )
        tampered = archived_bytes[:-1] + (
            b"X" if archived_bytes[-2:-1] != b"X" else b"Y"
        )
        with pytest.raises(ArchiveIntegrityError):
            read_archived_records(descriptor, tampered)

    def test_truncated_bytes_fail_closed_on_byte_count(self) -> None:
        descriptor, archived_bytes = _make_descriptor(
            "sess-1", [LLMMessage(role=Role.user, content="hello", message_id="u1")]
        )
        with pytest.raises(ArchiveIntegrityError):
            read_archived_records(descriptor, archived_bytes[:-3])

    def test_extra_bytes_fail_closed_on_prefix_end_byte(self) -> None:
        descriptor, archived_bytes = _make_descriptor(
            "sess-1", [LLMMessage(role=Role.user, content="hello", message_id="u1")]
        )
        with pytest.raises(ArchiveIntegrityError):
            read_archived_records(descriptor, archived_bytes + b"extra\n")

    def test_extra_record_fails_closed_on_archived_count(self) -> None:
        messages = [
            LLMMessage(role=Role.user, content="one", message_id="u1"),
            LLMMessage(role=Role.user, content="two", message_id="u2"),
            LLMMessage(role=Role.user, content="three", message_id="u3"),
        ]
        archived_bytes = b"".join(_record_line(m) for m in messages)
        # Descriptor claims 2 records over the full 3-record byte range.
        descriptor = ArchivedPrefixV1(
            session_id="sess-1",
            message_epoch=0,
            archived_count=2,
            prefix_end_byte=len(archived_bytes),
            prefix_sha256=hashlib.sha256(archived_bytes).hexdigest(),
            boundary_message_id=None,
            trim_mode="payloads",
        )
        with pytest.raises(ArchiveIntegrityError):
            list(iter_archived_records(descriptor, (archived_bytes,)))

    def test_fewer_records_fail_closed_on_archived_count(self) -> None:
        messages = [
            LLMMessage(role=Role.user, content="one", message_id="u1"),
            LLMMessage(role=Role.user, content="two", message_id="u2"),
        ]
        descriptor, _ = _make_descriptor("sess-1", messages)
        one_record = _record_line(messages[0])
        short = ArchivedPrefixV1(
            session_id="sess-1",
            message_epoch=0,
            archived_count=2,
            prefix_end_byte=len(one_record),
            prefix_sha256=hashlib.sha256(one_record).hexdigest(),
            boundary_message_id=None,
            trim_mode="payloads",
        )
        assert descriptor.archived_count == 2
        with pytest.raises(ArchiveIntegrityError):
            read_archived_records(short, one_record)

    def test_non_json_line_fails_closed(self) -> None:
        line = b"this is not json\n"
        descriptor = ArchivedPrefixV1(
            session_id="s",
            message_epoch=0,
            archived_count=1,
            prefix_end_byte=len(line),
            prefix_sha256=hashlib.sha256(line).hexdigest(),
            boundary_message_id=None,
            trim_mode="payloads",
        )
        with pytest.raises(ArchiveIntegrityError):
            read_archived_records(descriptor, line)

    def test_non_object_line_fails_closed(self) -> None:
        line = b"[1, 2]\n"
        descriptor = ArchivedPrefixV1(
            session_id="s",
            message_epoch=0,
            archived_count=1,
            prefix_end_byte=len(line),
            prefix_sha256=hashlib.sha256(line).hexdigest(),
            boundary_message_id=None,
            trim_mode="payloads",
        )
        with pytest.raises(ArchiveIntegrityError):
            read_archived_records(descriptor, line)

    def test_trailing_partial_line_fails_closed(self) -> None:
        full = _record_line(
            LLMMessage(role=Role.user, content="hello", message_id="u1")
        )
        partial = full[:-1]  # drop the newline: the archive ends mid-record
        descriptor = ArchivedPrefixV1(
            session_id="s",
            message_epoch=0,
            archived_count=1,
            prefix_end_byte=len(partial),
            prefix_sha256=hashlib.sha256(partial).hexdigest(),
            boundary_message_id=None,
            trim_mode="payloads",
        )
        with pytest.raises(ArchiveIntegrityError):
            read_archived_records(descriptor, partial)


class TestEpochSemantics:
    def test_epoch_starts_at_zero_and_survives_ordinary_mutations(self) -> None:
        messages = MessageList(
            initial=[LLMMessage(role=Role.user, content="first", message_id="u1")]
        )
        assert messages.message_epoch == 0
        messages.append(
            LLMMessage(role=Role.assistant, content="reply", message_id="a1")
        )
        messages.insert(0, LLMMessage(role=Role.user, content="early", message_id="u0"))
        messages.replace(
            0, LLMMessage(role=Role.user, content="early2", message_id="u0b")
        )
        messages.extend([LLMMessage(role=Role.user, content="more", message_id="u2")])
        messages.update_system_prompt("sys")
        assert messages.message_epoch == 0

    def test_reset_changes_epoch_and_clears_owned_descriptor(self) -> None:
        messages = MessageList(
            initial=[LLMMessage(role=Role.user, content="first", message_id="u1")]
        )
        descriptor, _ = _make_descriptor("sess-1", [messages[0]])
        messages.set_archive_descriptor(descriptor)
        assert messages.archive_descriptor is descriptor

        messages.reset([LLMMessage(role=Role.user, content="rewound", message_id="u0")])

        assert messages.message_epoch == 1
        assert messages.archive_descriptor is None

    def test_rebind_via_reset_preserving_system_changes_epoch(self) -> None:
        # The rebind path covered by tests/app_server/test_local_resume.py:
        # resume_root / the in-place resume commit call
        # ``messages.reset_preserving_system(loaded_messages)``
        # (app_server/_runtime.py:766, agent_loop/_loop.py:3291).
        messages = MessageList(
            initial=[
                LLMMessage(role=Role.system, content="sys"),
                LLMMessage(role=Role.user, content="old", message_id="u1"),
            ]
        )
        descriptor, _ = _make_descriptor("sess-1", [messages[1]])
        messages.set_archive_descriptor(descriptor)
        epoch_before = messages.message_epoch

        loaded_messages = [
            LLMMessage(role=Role.user, content="loaded", message_id="u9"),
            LLMMessage(role=Role.assistant, content="loaded reply", message_id="a9"),
        ]
        messages.reset_preserving_system(loaded_messages)

        assert messages.message_epoch == epoch_before + 1
        assert messages.archive_descriptor is None
        assert [m.role for m in messages] == [Role.system, Role.user, Role.assistant]
        assert messages[1].content == "loaded"

    def test_stale_epoch_descriptor_is_rejected_on_ownership(self) -> None:
        messages = MessageList(
            initial=[LLMMessage(role=Role.user, content="first", message_id="u1")]
        )
        stale, _ = _make_descriptor("sess-1", [messages[0]], message_epoch=0)
        messages.reset([LLMMessage(role=Role.user, content="new", message_id="u2")])
        with pytest.raises(ValueError):
            messages.set_archive_descriptor(stale)
        assert messages.archive_descriptor is None
        fresh, _ = _make_descriptor("sess-1", [messages[0]], message_epoch=1)
        messages.set_archive_descriptor(fresh)
        assert messages.archive_descriptor is fresh

    def test_cursor_invalidation_changes_neither_epoch_nor_descriptors(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "p4a1-cursor-epoch")
        messages = MessageList(
            initial=[LLMMessage(role=Role.user, content="first", message_id="u1")]
        )
        descriptor, _ = _make_descriptor(logger.session_id, [messages[0]])
        logger.retain_archive_descriptor(descriptor)
        messages.set_archive_descriptor(descriptor)

        logger.invalidate_transcript_cursor()

        assert logger._transcript_cursor is None
        assert logger.archive_descriptor is descriptor
        assert messages.message_epoch == 0
        assert messages.archive_descriptor is descriptor


class TestDescriptorOwnershipAndRetention:
    def test_logger_retains_descriptor_independently_of_cursor(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "p4a1-retention")
        assert logger.archive_descriptor is None
        descriptor, _ = _make_descriptor(
            "p4a1-retention", [LLMMessage(role=Role.user, content="x", message_id="u1")]
        )
        logger.retain_archive_descriptor(descriptor)
        assert logger.archive_descriptor is descriptor

        logger.invalidate_transcript_cursor()
        logger.invalidate_transcript_cursor()

        # Cursor invalidation discards append acceleration, not authority.
        assert logger._transcript_cursor is None
        assert logger.archive_descriptor is descriptor

    def test_retain_rejects_descriptor_of_another_session(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "p4a1-mismatch")
        foreign, _ = _make_descriptor(
            "some-other-session",
            [LLMMessage(role=Role.user, content="x", message_id="u1")],
        )
        with pytest.raises(ValueError):
            logger.retain_archive_descriptor(foreign)
        assert logger.archive_descriptor is None

    def test_clear_archive_descriptor_keeps_epoch(self) -> None:
        messages = MessageList(
            initial=[LLMMessage(role=Role.user, content="first", message_id="u1")]
        )
        descriptor, _ = _make_descriptor("sess-1", [messages[0]])
        messages.set_archive_descriptor(descriptor)
        messages.clear_archive_descriptor()
        assert messages.archive_descriptor is None
        assert messages.message_epoch == 0

    def test_save_snapshot_is_none_before_any_save(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "p4a1-no-save")
        assert logger.last_save_archive_snapshot is None


class TestSaveSnapshotCapture:
    @pytest.mark.asyncio
    async def test_save_snapshot_captures_descriptor_with_message_epoch(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "p4a1-save-snapshot")
        messages = MessageList(
            initial=[LLMMessage(role=Role.user, content="hi", message_id="u1")]
        )
        descriptor, _ = _make_descriptor(logger.session_id, [messages[0]])
        logger.retain_archive_descriptor(descriptor)
        messages.set_archive_descriptor(descriptor)

        await logger.save_interaction(
            messages, AgentStats(steps=1), mock_vibe_config, mock_tool_manager
        )

        snapshot = logger.last_save_archive_snapshot
        assert snapshot is not None
        assert snapshot.descriptor is descriptor
        assert snapshot.message_epoch == messages.message_epoch == 0
        assert snapshot.is_authoritative

    @pytest.mark.asyncio
    async def test_save_snapshot_after_rebind_captures_the_new_epoch(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "p4a1-save-rebind")
        messages = MessageList(
            initial=[LLMMessage(role=Role.user, content="hi", message_id="u1")]
        )
        descriptor, _ = _make_descriptor(logger.session_id, [messages[0]])
        logger.retain_archive_descriptor(descriptor)
        messages.set_archive_descriptor(descriptor)
        await logger.save_interaction(
            messages, AgentStats(steps=1), mock_vibe_config, mock_tool_manager
        )

        # Rebind exactly as the local-resume path does, then save again: the
        # snapshot still carries the retained descriptor, now paired WITH the
        # rebound list's new epoch (stale pairing is visible, not hidden).
        loaded = [LLMMessage(role=Role.user, content="loaded", message_id="u9")]
        messages.reset_preserving_system(loaded)
        await logger.save_interaction(
            messages, AgentStats(steps=2), mock_vibe_config, mock_tool_manager
        )

        snapshot = logger.last_save_archive_snapshot
        assert snapshot is not None
        assert snapshot.descriptor is descriptor
        assert snapshot.message_epoch == 1
        assert not snapshot.is_authoritative
        # Logger-side retention survived the rebind's cursor invalidation.
        assert logger.archive_descriptor is descriptor

    @pytest.mark.asyncio
    async def test_save_snapshot_of_plain_sequence_has_no_epoch(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "p4a1-plain-list")
        plain = [LLMMessage(role=Role.user, content="hi", message_id="u1")]

        await logger.save_interaction(
            plain, AgentStats(steps=1), mock_vibe_config, mock_tool_manager
        )

        snapshot = logger.last_save_archive_snapshot
        assert snapshot is not None
        assert snapshot.descriptor is None
        assert snapshot.message_epoch is None
        assert not snapshot.is_authoritative


class TestArchiveSaveSnapshot:
    def test_is_authoritative_requires_matching_epoch(self) -> None:
        descriptor, _ = _make_descriptor(
            "s",
            [LLMMessage(role=Role.user, content="x", message_id="u1")],
            message_epoch=3,
        )
        assert ArchiveSaveSnapshot(descriptor, 3).is_authoritative
        assert not ArchiveSaveSnapshot(descriptor, 4).is_authoritative
        assert not ArchiveSaveSnapshot(descriptor, None).is_authoritative
        assert not ArchiveSaveSnapshot(None, 3).is_authoritative


# ---------------------------------------------------------------------------
# P4a2: sealing — eligible-prefix rule, durable receipt, refusals, rebinds.
# ---------------------------------------------------------------------------


def _user(content: str = "hi") -> LLMMessage:
    return LLMMessage(role=Role.user, content=content)


def _assistant_text(content: str = "done") -> LLMMessage:
    return LLMMessage(role=Role.assistant, content=content)


def _assistant_call(call_id: str) -> LLMMessage:
    return LLMMessage(
        role=Role.assistant,
        content="working",
        tool_calls=[
            ToolCall(
                id=call_id,
                index=0,
                function=FunctionCall(name="read_file", arguments="{}"),
            )
        ],
    )


def _tool_result(call_id: str) -> LLMMessage:
    return LLMMessage(
        role=Role.tool,
        tool_call_id=call_id,
        name="read_file",
        content=f"result for {call_id}",
    )


def _boundary() -> LLMMessage:
    return LLMMessage(
        role=Role.user,
        content="compaction boundary",
        injected=True,
        context_boundary="compaction",
    )


async def _seal(
    logger: SessionLogger,
    messages: Any,
    config: ChartreuxConfigSchema,
    tool_manager: ToolManager,
    **kwargs: Any,
) -> Any:
    return await logger.seal_archive(
        messages, AgentStats(steps=1), config, tool_manager, **kwargs
    )


class TestEligibleArchivePrefix:
    def test_no_boundary_yields_none(self) -> None:
        messages = [_user("u1"), _assistant_text(), _user("u2"), _assistant_text()]
        assert eligible_archive_prefix(messages) is None

    def test_prefix_is_strictly_before_the_newest_boundary(self) -> None:
        old_boundary = _boundary()
        new_boundary = _boundary()
        messages = [
            _user("u1"),
            _assistant_text(),
            old_boundary,
            _user("u2"),
            _assistant_text(),
            new_boundary,
            _user("u3"),
            _assistant_text(),
        ]
        eligible = eligible_archive_prefix(messages)
        assert eligible is not None
        # The older boundary envelope is a settled record INSIDE the prefix;
        # the newest one stays live and names the archive.
        assert eligible.messages == (
            messages[0],
            messages[1],
            old_boundary,
            messages[3],
            messages[4],
        )
        assert eligible.boundary_message_id == new_boundary.message_id

    def test_system_messages_are_excluded(self) -> None:
        system = LLMMessage(role=Role.system, content="prompt")
        mid_system = LLMMessage(role=Role.system, content="update")
        u1, a1, boundary = _user("u1"), _assistant_text(), _boundary()
        eligible = eligible_archive_prefix([
            system,
            u1,
            a1,
            mid_system,
            boundary,
            _user("u2"),
            _assistant_text(),
        ])
        assert eligible is not None
        assert eligible.messages == (u1, a1)
        assert all(m.role is not Role.system for m in eligible.messages)

    def test_settled_tool_round_is_included(self) -> None:
        u1, a1, result, boundary = (
            _user("u1"),
            _assistant_call("c1"),
            _tool_result("c1"),
            _boundary(),
        )
        eligible = eligible_archive_prefix([
            u1,
            a1,
            result,
            boundary,
            _user("u2"),
            _assistant_text(),
        ])
        assert eligible is not None
        assert eligible.messages == (u1, a1, result)

    def test_unsettled_trailing_round_is_trimmed_back(self) -> None:
        # The boundary landed mid-invocation: the trailing round has an
        # unanswered tool call, so the prefix ends at the previous settled round.
        messages = [
            _user("u1"),
            _assistant_text(),
            _user("u2"),
            _assistant_call("c9"),
            _boundary(),
            _user("u3"),
        ]
        eligible = eligible_archive_prefix(messages)
        assert eligible is not None
        assert eligible.messages == (messages[0], messages[1])

    def test_trailing_bare_user_message_is_not_a_complete_round(self) -> None:
        messages = [_user("u1"), _assistant_text(), _user("u2"), _boundary()]
        eligible = eligible_archive_prefix(messages)
        assert eligible is not None
        assert eligible.messages == (messages[0], messages[1])

    def test_active_invocation_after_the_boundary_is_excluded(self) -> None:
        messages = [
            _user("u1"),
            _assistant_text(),
            _boundary(),
            _user("u2"),
            _assistant_call("c9"),  # active invocation, result still in flight
        ]
        eligible = eligible_archive_prefix(messages)
        assert eligible is not None
        assert eligible.messages == (messages[0], messages[1])

    def test_no_settled_round_before_the_boundary_yields_none(self) -> None:
        messages = [_user("u1"), _assistant_call("c1"), _boundary()]
        assert eligible_archive_prefix(messages) is None

    def test_empty_region_before_the_boundary_yields_none(self) -> None:
        assert eligible_archive_prefix([_boundary(), _user("u1")]) is None

    def test_eligible_prefix_rejects_system_messages_and_empty_boundary_id(
        self,
    ) -> None:
        from chartreux.core.session.transcript_archive import EligibleArchivePrefix

        with pytest.raises(ValueError):
            EligibleArchivePrefix(
                messages=(LLMMessage(role=Role.system, content="s"),),
                boundary_message_id="b",
            )
        with pytest.raises(ValueError):
            EligibleArchivePrefix(messages=(), boundary_message_id="")


class TestArchiveSealReceipt:
    def test_receipt_requires_descriptor_and_sealed_at(self) -> None:
        descriptor, _ = _make_descriptor(
            "s", [LLMMessage(role=Role.user, content="x", message_id="u1")]
        )
        receipt = ArchiveSealReceipt(
            descriptor=descriptor, sealed_at="2026-11-01T00:00:00+00:00"
        )
        assert receipt.session_id == "s"
        assert receipt.archived_count == 1
        assert receipt.prefix_end_byte == descriptor.prefix_end_byte
        assert receipt.prefix_sha256 == descriptor.prefix_sha256
        assert receipt.boundary_message_id == "boundary-1"
        with pytest.raises(ValueError):
            ArchiveSealReceipt(descriptor=descriptor, sealed_at="")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            ArchiveSealReceipt(descriptor="x", sealed_at="t")  # type: ignore[arg-type]


class TestSealArchivePositive:
    @pytest.mark.asyncio
    async def test_seal_issues_receipt_after_durable_persist_and_disk_verify(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "p4a2-seal")
        boundary = _boundary()
        messages = MessageList([
            LLMMessage(role=Role.system, content="prompt"),
            _user("u1"),
            _assistant_call("c1"),
            _tool_result("c1"),
            boundary,
            _user("u2"),
            _assistant_text(),
        ])

        receipt = await _seal(logger, messages, mock_vibe_config, mock_tool_manager)

        assert isinstance(receipt, ArchiveSealReceipt)
        descriptor = receipt.descriptor
        assert descriptor.session_id == "p4a2-seal"
        assert descriptor.message_epoch == messages.message_epoch == 0
        # The closed prefix is the settled round strictly before the boundary;
        # the boundary envelope and the live tail stay live on disk.
        assert descriptor.archived_count == 3
        assert descriptor.boundary_message_id == boundary.message_id
        assert descriptor.trim_mode == "payloads"
        assert receipt.sealed_at
        # The descriptor was retained and the save snapshot is authoritative.
        assert logger.archive_descriptor is descriptor
        snapshot = logger.last_save_archive_snapshot
        assert snapshot is not None and snapshot.is_authoritative
        assert logger.persisted
        # Durable persist + on-disk verification: the first prefix_end_byte
        # bytes of messages.jsonl are exactly the sealed prefix.
        assert logger.session_dir is not None
        messages_path = logger.session_dir / "messages.jsonl"
        raw = messages_path.read_bytes()
        assert len(raw) > descriptor.prefix_end_byte
        assert (
            hashlib.sha256(raw[: descriptor.prefix_end_byte]).hexdigest()
            == descriptor.prefix_sha256
        )
        records = [json.loads(line) for line in raw.splitlines()]
        assert len(records) == 6
        # Sealing consumes the shared read interface (single source).
        verify_archived_prefix_on_disk(descriptor, messages_path)

    @pytest.mark.asyncio
    async def test_seal_verify_goes_through_the_shared_read_interface(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import chartreux.core.session.transcript_archive as archive_module

        calls: list[Any] = []
        real = archive_module.iter_archived_records

        def spy(descriptor: Any, chunks: Any) -> Any:
            calls.append(descriptor)
            return real(descriptor, chunks)

        monkeypatch.setattr(archive_module, "iter_archived_records", spy)
        logger = SessionLogger(session_config, "p4a2-seal-spy")
        messages = MessageList([
            _user("u1"),
            _assistant_text(),
            _boundary(),
            _user("u2"),
            _assistant_text(),
        ])
        receipt = await _seal(logger, messages, mock_vibe_config, mock_tool_manager)
        assert calls == [receipt.descriptor]

    @pytest.mark.asyncio
    async def test_seal_records_the_requested_trim_mode(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "p4a2-seal-tombstones")
        messages = MessageList([
            _user("u1"),
            _assistant_text(),
            _boundary(),
            _user("u2"),
            _assistant_text(),
        ])
        receipt = await _seal(
            logger,
            messages,
            mock_vibe_config,
            mock_tool_manager,
            trim_mode="tombstones",
        )
        assert receipt.descriptor.trim_mode == "tombstones"


class TestSealArchiveRefusals:
    @pytest.mark.asyncio
    async def test_malformed_prefix_duplicate_results_yields_no_receipt(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "p4a2-seal-dup")
        messages = MessageList([
            _user("u1"),
            _assistant_call("c1"),
            _tool_result("c1"),
            _tool_result("c1"),  # two genuine results: ambiguous duplicate
            _boundary(),
            _user("u2"),
            _assistant_text(),
        ])
        with pytest.raises(ArchiveSealError) as exc_info:
            await _seal(logger, messages, mock_vibe_config, mock_tool_manager)
        assert exc_info.value.violations
        assert any(
            getattr(v, "kind", None) == "duplicate_tool_result"
            for v in exc_info.value.violations
        )
        assert logger.archive_descriptor is None
        assert not logger.persisted

    @pytest.mark.asyncio
    async def test_unresolved_tool_call_inside_prefix_yields_no_receipt(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "p4a2-seal-unresolved")
        messages = MessageList([
            _user("u1"),
            _assistant_call("c1"),  # never answered, mid-prefix
            _user("u2"),
            _assistant_call("c2"),
            _tool_result("c2"),
            _boundary(),
            _user("u3"),
            _assistant_text(),
        ])
        with pytest.raises(ArchiveSealError) as exc_info:
            await _seal(logger, messages, mock_vibe_config, mock_tool_manager)
        assert any(
            getattr(v, "kind", None) == "unresolved_tool_call"
            for v in exc_info.value.violations
        )
        assert logger.archive_descriptor is None

    @pytest.mark.asyncio
    async def test_unclaimed_tool_result_in_prefix_yields_no_receipt(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "p4a2-seal-unclaimed")
        messages = MessageList([
            _user("u1"),
            _tool_result("c9"),  # no assistant call claims it
            _assistant_text(),
            _boundary(),
            _user("u2"),
            _assistant_text(),
        ])
        with pytest.raises(ArchiveSealError) as exc_info:
            await _seal(logger, messages, mock_vibe_config, mock_tool_manager)
        assert any(
            getattr(v, "kind", None) == "unclaimed_tool_result"
            for v in exc_info.value.violations
        )
        assert logger.archive_descriptor is None

    @pytest.mark.asyncio
    async def test_no_boundary_yields_no_receipt(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "p4a2-seal-noboundary")
        messages = MessageList([_user("u1"), _assistant_text()])
        with pytest.raises(ArchiveSealError, match="no eligible archive prefix"):
            await _seal(logger, messages, mock_vibe_config, mock_tool_manager)
        assert logger.archive_descriptor is None

    @pytest.mark.asyncio
    async def test_plain_sequence_cannot_be_sealed(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "p4a2-seal-plain")
        plain = [
            _user("u1"),
            _assistant_text(),
            _boundary(),
            _user("u2"),
            _assistant_text(),
        ]
        with pytest.raises(ArchiveSealError, match="plain message sequence"):
            await _seal(logger, plain, mock_vibe_config, mock_tool_manager)
        assert logger.archive_descriptor is None

    @pytest.mark.asyncio
    async def test_disabled_logger_cannot_seal(
        self,
        disabled_session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(disabled_session_config, "p4a2-seal-disabled")
        messages = MessageList([_user("u1"), _assistant_text(), _boundary()])
        with pytest.raises(ArchiveSealError, match="disabled"):
            await _seal(logger, messages, mock_vibe_config, mock_tool_manager)

    @pytest.mark.asyncio
    async def test_stale_retained_descriptor_fails_only_at_sealing_not_at_save(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        # P4a2 open question (a): a non-authoritative save snapshot (descriptor
        # epoch != list epoch after a list rebind) is NOT a hard save-time
        # failure — the ordinary save stays correct and succeeds — it fails
        # ONLY at sealing, which refuses with no receipt.
        logger = SessionLogger(session_config, "p4a2-stale")
        messages = MessageList([_user("first")])
        descriptor, _ = _make_descriptor(logger.session_id, [messages[0]])
        logger.retain_archive_descriptor(descriptor)

        # List rebind (resume path): epoch bumps, logger retention survives.
        messages.reset_preserving_system([
            _user("u9"),
            _boundary(),
            _user("u10"),
            _assistant_text(),
        ])
        assert messages.message_epoch == 1
        await logger.save_interaction(
            messages, AgentStats(steps=1), mock_vibe_config, mock_tool_manager
        )
        # The ordinary save did NOT hard-fail on the non-authoritative snapshot.
        assert logger.persisted
        snapshot = logger.last_save_archive_snapshot
        assert snapshot is not None and not snapshot.is_authoritative

        with pytest.raises(
            ArchiveSealError, match="does not match the saved list epoch"
        ):
            await _seal(logger, messages, mock_vibe_config, mock_tool_manager)
        # The refusal issued no receipt and left retention untouched.
        assert logger.archive_descriptor is descriptor


class TestRebindRevalidatesArchiveAuthority:
    def test_reset_session_drops_descriptor_of_the_departed_session(
        self, session_config: SessionLoggingConfig
    ) -> None:
        # P4a2 open question (b): the session-id check is re-validated on
        # logger rebind — a descriptor naming the departed session is dropped
        # fail-closed.
        logger = SessionLogger(session_config, "old-session")
        descriptor, _ = _make_descriptor(
            "old-session", [LLMMessage(role=Role.user, content="x", message_id="u1")]
        )
        logger.retain_archive_descriptor(descriptor)

        logger.reset_session("new-session")

        assert logger.session_id == "new-session"
        assert logger.archive_descriptor is None

    def test_reset_session_keeps_a_matching_descriptor(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "same-session")
        descriptor, _ = _make_descriptor(
            "same-session", [LLMMessage(role=Role.user, content="x", message_id="u1")]
        )
        logger.retain_archive_descriptor(descriptor)

        logger.reset_session("same-session")

        assert logger.archive_descriptor is descriptor

    def test_apply_resumed_session_revalidates_the_session_id(
        self, session_config: SessionLoggingConfig, tmp_path: Any
    ) -> None:
        logger = SessionLogger(session_config, "sess-a")
        assert logger.session_metadata is not None
        metadata = logger.session_metadata
        descriptor_a, _ = _make_descriptor(
            "sess-a", [LLMMessage(role=Role.user, content="x", message_id="u1")]
        )
        logger.retain_archive_descriptor(descriptor_a)

        # Rebind onto a different session: the foreign descriptor is dropped.
        logger.apply_resumed_session("sess-b", tmp_path / "other", metadata)
        assert logger.session_id == "sess-b"
        assert logger.archive_descriptor is None

        # Warm continuation onto the SAME session keeps a matching descriptor.
        descriptor_b, _ = _make_descriptor(
            "sess-b", [LLMMessage(role=Role.user, content="y", message_id="u2")]
        )
        logger.retain_archive_descriptor(descriptor_b)
        logger.apply_resumed_session("sess-b", tmp_path / "other", metadata)
        assert logger.archive_descriptor is descriptor_b


# ---------------------------------------------------------------------------
# P4a4: the generation fence — reset generation on rebind, the stale-descriptor
# guard on retention, the validation-side authority token, and the save
# boundary recognizing ONLY the authorized archived prefix.
# ---------------------------------------------------------------------------


class TestP4a4ResetGeneration:
    def test_archive_generation_starts_at_zero_and_bumps_on_every_rebind(
        self, session_config: SessionLoggingConfig, tmp_path: Any
    ) -> None:
        logger = SessionLogger(session_config, "p4a4-gen")
        assert logger._archive_generation == 0
        logger.reset_session("p4a4-gen")
        assert logger._archive_generation == 1
        assert logger.session_metadata is not None
        logger.apply_resumed_session(
            "p4a4-gen", tmp_path / "resume", logger.session_metadata
        )
        assert logger._archive_generation == 2

    def test_retain_refuses_a_stale_epoch_descriptor(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "p4a4-retain-stale")
        newer, _ = _make_descriptor(
            "p4a4-retain-stale",
            [LLMMessage(role=Role.user, content="x", message_id="u1")],
            message_epoch=1,
        )
        logger.retain_archive_descriptor(newer)
        older, _ = _make_descriptor(
            "p4a4-retain-stale",
            [LLMMessage(role=Role.user, content="x", message_id="u1")],
            message_epoch=0,
        )
        with pytest.raises(ValueError, match="stale"):
            logger.retain_archive_descriptor(older)
        assert logger.archive_descriptor is newer
        # An equal-epoch re-retention (re-seal) is fine.
        same, _ = _make_descriptor(
            "p4a4-retain-stale",
            [LLMMessage(role=Role.user, content="x", message_id="u1")],
            message_epoch=1,
        )
        logger.retain_archive_descriptor(same)
        assert logger.archive_descriptor is same


class TestP4a4ArchiveAuthorityToken:
    def test_authority_is_minted_only_for_the_authoritative_pairing(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "p4a4-auth")
        messages = MessageList([
            LLMMessage(role=Role.user, content="hi", message_id="u1")
        ])
        # Nothing retained -> no authority.
        assert logger._archive_authority_for(messages) is None
        descriptor, _ = _make_descriptor(logger.session_id, [messages[0]])
        logger.retain_archive_descriptor(descriptor)
        authority = logger._archive_authority_for(messages)
        assert authority is not None
        assert authority.archived_count == descriptor.archived_count
        assert authority.boundary_message_id == descriptor.boundary_message_id
        assert authority.message_epoch == 0
        # Epoch mismatch (list rebound) -> no authority.
        messages.reset([LLMMessage(role=Role.user, content="new", message_id="u2")])
        assert logger._archive_authority_for(messages) is None
        # A plain sequence carries no epoch -> no authority.
        assert logger._archive_authority_for(list(messages)) is None

    @pytest.mark.asyncio
    async def test_save_boundary_recognizes_the_authorized_prefix(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """A save under an authoritative descriptor passes the S1 boundary and
        stays archive-aware (the seal's own save already proved this; this pins
        it through the authority token minted at the save boundary).
        """
        logger = SessionLogger(session_config, "p4a4-save-auth")
        messages = MessageList([
            LLMMessage(role=Role.system, content="prompt"),
            _user("u1"),
            _assistant_text(),
            _boundary(),
            _user("u2"),
            _assistant_text(),
        ])
        receipt = await _seal(logger, messages, mock_vibe_config, mock_tool_manager)
        assert logger.archive_descriptor is receipt.descriptor
        await logger.save_interaction(
            messages, AgentStats(steps=2), mock_vibe_config, mock_tool_manager
        )
        assert logger.persisted
        assert logger.last_save_diagnostic is None
        snapshot = logger.last_save_archive_snapshot
        assert snapshot is not None and snapshot.is_authoritative

    @pytest.mark.asyncio
    async def test_save_boundary_fails_closed_on_a_marker_without_authority(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """A snapshot with an archived marker but NO matching authority (no
        retained descriptor) fails closed at the save boundary — it must not
        be mistaken for legacy complete history.
        """
        logger = SessionLogger(session_config, "p4a4-save-marker")
        messages = MessageList([
            LLMMessage(role=Role.user, content="hi", message_id="u1")
        ])
        object.__setattr__(messages[0], ARCHIVE_REFERENCE_ATTRIBUTE, "ref-1")
        with pytest.raises(TranscriptIntegrityError) as exc_info:
            await logger.save_interaction(
                messages, AgentStats(steps=1), mock_vibe_config, mock_tool_manager
            )
        assert any(
            violation.kind == "unauthorized_archived_prefix"
            for violation in exc_info.value.violations
        )
        assert not logger.persisted

    @pytest.mark.asyncio
    async def test_fence_publish_after_seal_publishes_both_owners(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """Seal -> fence+publish with test-double replacements: the descriptor
        lands on the list AND the logger, and the next save is archive-aware.
        """
        logger = SessionLogger(session_config, "p4a4-fence-int")
        messages = MessageList([
            LLMMessage(role=Role.system, content="prompt"),
            _user("u1"),
            LLMMessage(
                role=Role.assistant,
                content="a1",
                message_id="a1",
                reasoning_payloads=[{"opaque": "p1"}],
            ),
            _boundary(),
            _user("u2"),
            _assistant_text(),
        ])
        receipt = await _seal(logger, messages, mock_vibe_config, mock_tool_manager)
        descriptor = receipt.descriptor
        replacements = []
        for index in (1, 2):
            replacement = messages[index].model_copy(
                update={"reasoning_payloads": None}
            )
            object.__setattr__(replacement, ARCHIVE_REFERENCE_ATTRIBUTE, f"ref-{index}")
            replacements.append(replacement)
        logger.fence_and_publish_archive_trim(
            messages,
            descriptor,
            replacements,
            save_generation=logger._transcript_cursor_generation,
            reset_generation=logger._archive_generation,
        )
        assert messages.archive_descriptor is descriptor
        assert logger.archive_descriptor is descriptor
        assert messages[1] is replacements[0] and messages[2] is replacements[1]
        await logger.save_interaction(
            messages, AgentStats(steps=2), mock_vibe_config, mock_tool_manager
        )
        assert logger.last_save_diagnostic is None
        snapshot = logger.last_save_archive_snapshot
        assert snapshot is not None and snapshot.descriptor is descriptor
