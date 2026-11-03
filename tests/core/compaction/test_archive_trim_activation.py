"""P4d: trim ACTIVATION — the M1 payload-only eviction contract, end to end.

Every assertion here is about the ACTIVATED path (P4b's wiring shipped inert
behind ``ARCHIVE_TRIM_ENABLED``; P4d flips it on):

- the exact compaction procedure: the envelope is persisted with the
  originals present, the eligible closed prefix is sealed under the save lock
  WITH a durable receipt, replacements are prepared without mutating the
  originals, queued stale saves are fenced, and the publication happens
  synchronously with no ``reset()`` and no reset hooks;
- failure retention: a persistence/verification failure leaves the originals
  in RAM (compaction still succeeds);
- TRIM-ACTUALLY-RELEASED: the PRE-TRIM ``LLMMessage`` instances themselves are
  collectible after the publication (a weakref to the instance, not a
  ``reasoning_payloads == []`` shape check);
- FIELD-PRESERVATION MATRIX: only ``reasoning_payloads`` is emptied — content,
  tool calls/results, image references and display fields survive intact, so
  over-trimming fails here (the disk bytes are authoritative on resume, which
  is why nothing else catches it);
- resume does not re-trim; the automatic-compaction closed-prefix restriction
  holds with a real sealing surface;
- a re-seal after a real trim EXTENDS the descriptor from the old prefix disk
  bytes plus the new records (P4b open item (a)), and refuses to re-anchor
  changed archived bytes;
- the admission-side authority: a live list mints it from its OWN sealed state,
  while a plain sequence, an unarchived list, or a stale epoch yields none (so
  markers still fail closed); the turn-start admission path itself is pinned by
  tests/app_server/test_session_resume_admission.py::
  test_compaction_keeps_session_and_restores_full_history;
- the integrated compaction -> trim -> save -> full-verify -> resume cycle,
  including a display-identical reconstruction and the cold-resume load.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import weakref

import pytest

from chartreux.app_server._projection import history_messages, project_message_history
from chartreux.core.agent_loop._transcript_integrity import (
    archive_reference,
    live_archive_authority,
    validate_transcript,
)
from chartreux.core.compaction.manager import ARCHIVE_TRIM_ENABLED, CompactionManager
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import (
    FileImageSource,
    FunctionCall,
    ImageAttachment,
    LLMChunk,
    LLMMessage,
    PersistedToolResult,
    Role,
    ToolCall,
)
from chartreux.core.message_list import MessageList
from chartreux.core.session.session_loader import MESSAGES_FILENAME, SessionLoader
from chartreux.core.session.session_logger import SessionDiskFullError, SessionLogger
from chartreux.core.session.transcript_archive import (
    hydrate_archived_messages,
    verify_archived_prefix_on_disk,
)
from chartreux.core.session_types import AgentStats
from chartreux.core.tools.manager import ToolManager
from chartreux.core.usage import UsagePurpose
from chartreux.user_content import UserDisplayContent, UserTextResource
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from tests.mock.utils import mock_llm_chunk
from tests.session.persistence._session_persistence_support import (
    mock_tool_manager as mock_tool_manager,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)


class _ScriptedComplete:
    """Stand-in for ``AgentLoop._complete``: pops one scripted result per call."""

    def __init__(self, results: list[LLMChunk], stats: AgentStats) -> None:
        self._results = list(results)
        self._stats = stats
        self.calls: list[list[LLMMessage]] = []

    async def __call__(
        self,
        *,
        model: Any,
        messages: Any,
        tools: Any,
        tool_choice: Any,
        call_type: str | None,
        account_conversation: bool,
        purpose: UsagePurpose = UsagePurpose.CONVERSATION,
    ) -> LLMChunk:
        assert account_conversation is False
        self.calls.append(list(messages))
        return self._results.pop(0)


def _config() -> ChartreuxConfigSchema:
    return build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=999),
        raise_on_compaction_failure=False,
    )


def _build_logger(
    session_config: SessionLoggingConfig, session_id: str
) -> SessionLogger:
    return SessionLogger(session_config, session_id)


async def _save(
    logger: SessionLogger,
    messages: MessageList,
    stats: AgentStats,
    config: ChartreuxConfigSchema,
    tool_manager: ToolManager,
) -> None:
    await logger.save_interaction(messages, stats, config, tool_manager)


def _manager(
    *,
    logger: SessionLogger,
    messages: MessageList,
    stats: AgentStats,
    config: ChartreuxConfigSchema,
    tool_manager: ToolManager,
    complete: _ScriptedComplete,
    has_active_turn: bool | None = None,
) -> CompactionManager:
    async def save() -> None:
        await _save(logger, messages, stats, config, tool_manager)

    return CompactionManager(
        messages=messages,
        stats_getter=lambda: stats,
        config_getter=lambda: config,
        complete=complete,
        available_tools=lambda: [],
        tool_choice=lambda: "auto",
        save=save,
        session_logger_getter=lambda: logger,
        tool_manager_getter=lambda: tool_manager,
        has_active_turn=(
            (lambda: has_active_turn) if has_active_turn is not None else None
        ),
    )


def _image() -> ImageAttachment:
    return ImageAttachment(
        source=FileImageSource(path=Path("/repo/assets/diagram.png")),
        alias="diagram.png",
        mime_type="image/png",
    )


def _display() -> UserDisplayContent:
    return UserDisplayContent(
        version="1", host="chartreux-test", content=[{"type": "text", "text": "ask"}]
    )


def _tool_result() -> PersistedToolResult:
    return PersistedToolResult(output={"exit_code": 0}, duration=0.25)


def _conversation() -> MessageList:
    """A conversation whose eligible prefix exercises every preserved field."""
    return MessageList([
        LLMMessage(role=Role.system, content="system prompt"),
        LLMMessage(
            role=Role.user,
            content="oldest ask",
            message_id="u1",
            images=[_image()],
            user_display_content=_display(),
            resources=[
                UserTextResource(kind="text", uri="file:///repo/a.py", text="print(1)")
            ],
            input_text="oldest ask",
        ),
        LLMMessage(
            role=Role.assistant,
            content="answer with a tool",
            message_id="a1",
            reasoning_content="visible thinking",
            reasoning_payloads=[{"opaque": "payload-1"}],
            tool_calls=[
                ToolCall(
                    id="t1",
                    index=0,
                    function=FunctionCall(name="bash", arguments='{"cmd": "ls"}'),
                )
            ],
            turn_duration=1.5,
            deployment_identity={"base_model": "m", "deployment": "d"},
        ),
        LLMMessage(
            role=Role.tool,
            tool_call_id="t1",
            name="bash",
            content="file.txt",
            tool_result=_tool_result(),
        ),
        LLMMessage(role=Role.user, content="newest ask", message_id="u2"),
        LLMMessage(
            role=Role.assistant,
            content="newest answer",
            message_id="a2",
            reasoning_payloads=[{"opaque": "payload-2"}],
        ),
    ])


def _messages_path(logger: SessionLogger) -> Path:
    session_dir = logger.session_dir
    assert session_dir is not None
    return session_dir / MESSAGES_FILENAME


def _non_system(messages: MessageList) -> list[LLMMessage]:
    return [message for message in messages if message.role is not Role.system]


def _dumps(messages: Any) -> list[dict[str, Any]]:
    return [message.model_dump(mode="json") for message in messages]


# ---------------------------------------------------------------------------
# Activation wiring
# ---------------------------------------------------------------------------


def test_trim_flag_is_activated_at_p4d() -> None:
    assert ARCHIVE_TRIM_ENABLED is True


def test_loop_wires_the_activated_trim_into_its_compaction_manager() -> None:
    # The activation site in the compaction path: the loop hands the manager
    # every dependency the trim needs, and the manager resolves the flag ON.
    loop = build_test_agent_loop(config=_config())
    manager = loop.compaction_manager
    assert manager._trim_enabled() is True
    assert manager._session_logger_getter is not None
    assert manager._session_logger_getter() is loop.session_logger
    assert manager._tool_manager_getter is not None
    assert manager._tool_manager_getter() is loop.tool_manager
    assert manager._has_active_turn is not None


# ---------------------------------------------------------------------------
# Exact compaction procedure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_seals_with_a_receipt_then_publishes_the_trim(
    session_config: SessionLoggingConfig,
    mock_tool_manager: ToolManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Step 1 (envelope persisted with originals present) is asserted AT the
    # sealing boundary: when seal_archive is entered, the transcript already
    # holds the compaction envelope, still carries the originals' payloads,
    # and no slot carries an archive marker yet.
    config = _config()
    logger = _build_logger(session_config, "p4d-procedure")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [mock_llm_chunk(content="<summary>all done</summary>")], stats
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )

    observed: dict[str, Any] = {}
    real_seal = SessionLogger.seal_archive

    async def _seal_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        raw = _messages_path(self).read_bytes()
        observed["boundary_persisted"] = b'"context_boundary": "compaction"' in raw
        observed["originals_payloads_persisted"] = (
            b'"reasoning_payloads": [{"opaque": "payload-1"}]' in raw
        )
        observed["no_marker_before_seal"] = all(
            archive_reference(message) is None for message in messages
        )
        receipt = await real_seal(self, *args, **kwargs)
        observed["receipt_trim_mode"] = receipt.descriptor.trim_mode
        return receipt

    monkeypatch.setattr(SessionLogger, "seal_archive", _seal_spy)

    summary = await manager.compact()

    assert summary == "all done"
    assert observed == {
        "boundary_persisted": True,
        "originals_payloads_persisted": True,
        "no_marker_before_seal": True,
        "receipt_trim_mode": "payloads",
    }

    descriptor = messages.archive_descriptor
    assert descriptor is not None
    assert descriptor.trim_mode == "payloads"
    assert descriptor.archived_count == 5
    # Open item (b): BOTH owners carry the descriptor after the activated
    # publication — hydration silently no-ops when the list copy is missing.
    assert logger.archive_descriptor is descriptor
    # The sealed prefix's RAM copies are the trimmed replacements.
    sealed = _non_system(messages)[: descriptor.archived_count]
    assert all(message.reasoning_payloads is None for message in sealed)
    assert all(archive_reference(message) is descriptor for message in sealed)
    # The boundary envelope stays live and unmarked (the tail is exactly it).
    tail = _non_system(messages)[descriptor.archived_count :]
    assert [message.context_boundary for message in tail] == ["compaction"]
    assert all(archive_reference(message) is None for message in tail)
    # The disk prefix is the authoritative ORIGINAL bytes.
    verify_archived_prefix_on_disk(descriptor, _messages_path(logger))
    raw = _messages_path(logger).read_bytes()
    archived_records = [
        json.loads(line) for line in raw[: descriptor.prefix_end_byte].splitlines()
    ]
    assert {"reasoning_payloads": [{"opaque": "payload-1"}]} in [
        {"reasoning_payloads": record["reasoning_payloads"]}
        for record in archived_records
        if "reasoning_payloads" in record
    ]


@pytest.mark.asyncio
async def test_trim_publication_calls_no_reset_and_fires_no_reset_hook(
    session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
) -> None:
    config = _config()
    logger = _build_logger(session_config, "p4d-no-reset")
    messages = _conversation()
    stats = AgentStats(steps=1)
    hooks: list[str] = []
    messages.on_reset(lambda: hooks.append("reset-fired"))
    complete = _ScriptedComplete(
        [mock_llm_chunk(content="<summary>all done</summary>")], stats
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )
    epoch_before = messages.message_epoch

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(MessageList, "reset", cast(Any, _forbidden_reset), raising=True)
        await manager.compact()

    # No reset() ran (it would have raised), no hook fired, and the epoch — the
    # checkpoint/rewind identity — is untouched: trimming is same-slot.
    assert hooks == []
    assert messages.message_epoch == epoch_before
    assert messages.archive_descriptor is not None


def _forbidden_reset(self: MessageList, new: list[LLMMessage]) -> None:
    raise AssertionError("trim publication must never call MessageList.reset()")


# ---------------------------------------------------------------------------
# Failure retention
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persistence_failure_retains_the_originals(
    session_config: SessionLoggingConfig,
    mock_tool_manager: ToolManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    logger = _build_logger(session_config, "p4d-retain")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [mock_llm_chunk(content="<summary>all done</summary>")], stats
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )
    before = _dumps(_non_system(messages))

    def _disk_full(*args: Any, **kwargs: Any) -> Any:
        raise SessionDiskFullError(_messages_path(logger), "seal persist")

    # The sealing persist fails: no receipt, so the trim must not publish.
    monkeypatch.setattr(SessionLogger, "_seal_persist_sync", _disk_full)

    summary = await manager.compact()

    assert summary == "all done"
    assert messages.archive_descriptor is None
    assert logger.archive_descriptor is None
    assert all(archive_reference(message) is None for message in messages)
    # The originals are untouched — payloads intact, no tombstoning.
    after = _dumps(_non_system(messages))[:-1]  # minus the appended envelope
    assert after == before
    # The compaction itself succeeded and its envelope was persisted.
    assert messages[-1].context_boundary == "compaction"
    raw = _messages_path(logger).read_bytes()
    assert b'"context_boundary": "compaction"' in raw


@pytest.mark.asyncio
async def test_verification_failure_retains_the_originals(
    session_config: SessionLoggingConfig,
    mock_tool_manager: ToolManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chartreux.core.session.session_logger as logger_module

    config = _config()
    logger = _build_logger(session_config, "p4d-verify-retain")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [mock_llm_chunk(content="<summary>all done</summary>")], stats
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )

    def _unverifiable(*args: Any, **kwargs: Any) -> None:
        raise logger_module.ArchiveIntegrityError("prefix bytes changed")

    monkeypatch.setattr(logger_module, "verify_archived_prefix_on_disk", _unverifiable)

    summary = await manager.compact()

    assert summary == "all done"
    assert messages.archive_descriptor is None
    assert logger.archive_descriptor is None
    assert [message.reasoning_payloads for message in _non_system(messages)[1:5]] == [
        [{"opaque": "payload-1"}],
        None,
        None,
        [{"opaque": "payload-2"}],
    ]


# ---------------------------------------------------------------------------
# Trim-actually-released (the binding weakref mechanism)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trim_actually_releases_the_original_message_graph(
    session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
) -> None:
    config = _config()
    logger = _build_logger(session_config, "p4d-release")
    # Built inline: a local list of the same instances would keep the
    # collectibility targets alive and void the assertion.
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [mock_llm_chunk(content="<summary>all done</summary>")], stats
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )
    # The collectibility target is the PRE-TRIM LLMMessage INSTANCE (a plain
    # dict payload cannot be weak-referenced).
    sealed_indexes = [1, 2, 3, 4, 5]
    targets = [weakref.ref(messages[index]) for index in sealed_indexes]
    assert all(reference() is not None for reference in targets)

    await manager.compact()

    assert messages.archive_descriptor is not None
    assert all(messages[index].reasoning_payloads is None for index in sealed_indexes)
    # Every non-list reference to the originals is gone: the scripted
    # summarizer's recorded copies go with it.
    complete.calls.clear()
    gc.collect()

    assert [reference() for reference in targets] == [None] * len(targets)


# ---------------------------------------------------------------------------
# Field-preservation matrix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_field_preservation_matrix_only_reasoning_payloads_is_emptied(
    session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
) -> None:
    config = _config()
    logger = _build_logger(session_config, "p4d-matrix")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [mock_llm_chunk(content="<summary>all done</summary>")], stats
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )
    before = _dumps(_non_system(messages)[:5])

    await manager.compact()

    descriptor = messages.archive_descriptor
    assert descriptor is not None
    after = _dumps(_non_system(messages)[: descriptor.archived_count])
    assert len(after) == len(before) == descriptor.archived_count
    for original, retained in zip(before, after, strict=True):
        # Exact declared transformation: drop ONLY reasoning_payloads; every
        # other persisted field (content, tool calls/results, image
        # references, display fields, reasoning text) must be byte-identical.
        original.pop("reasoning_payloads", None)
        retained.pop("reasoning_payloads", None)
        assert retained == original
    # Named spot checks for the matrix's required columns.
    user = messages[1]
    assert str(user.content) == "oldest ask"
    assert user.images is not None and user.images[0].alias == "diagram.png"
    assert user.images[0].source.kind == "file"
    assert user.user_display_content == _display()
    assert user.input_text == "oldest ask"
    assert user.resources is not None and user.resources[0].uri == "file:///repo/a.py"
    assistant = messages[2]
    assert assistant.reasoning_content == "visible thinking"
    assert assistant.reasoning_message_id is not None
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].id == "t1"
    assert assistant.turn_duration == 1.5
    assert assistant.deployment_identity == {"base_model": "m", "deployment": "d"}
    tool = messages[3]
    assert tool.tool_call_id == "t1"
    assert tool.tool_result == _tool_result()


# ---------------------------------------------------------------------------
# Resume / closed-prefix restriction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warm_resume_with_a_retained_descriptor_clears_it_by_clear_and_reseal(
    session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
) -> None:
    config = _config()
    logger = _build_logger(session_config, "p4d-resume")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [mock_llm_chunk(content="<summary>all done</summary>")], stats
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )
    await manager.compact()
    descriptor = messages.archive_descriptor
    assert descriptor is not None
    session_dir = logger.session_dir
    assert session_dir is not None
    # The transcript holds non-system records only (the system prompt is
    # session metadata), so the resumed list is [system] + loaded.
    loaded, _metadata = SessionLoader.load_session(session_dir)
    loaded_dumps = _dumps(loaded)

    # Warm resume: the logger rebinds to the SAME session (its descriptor is
    # retained through the rebind) and the list rebinds to the complete
    # loaded history; the resume path then runs the clear-and-reseal
    # transaction (T2b), which proves the rebound list hydrated from disk
    # and CLEARS the retained descriptor — never re-pinned.
    logger.apply_resumed_session(
        logger.session_id, session_dir, SessionLoader.load_metadata(session_dir)
    )
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(SessionLogger, "seal_archive", _forbidden_seal)
        patcher.setattr(
            SessionLogger,
            "fence_and_publish_archive_trim",
            cast(Any, _forbidden_publish),
        )
        messages.reset_preserving_system(loaded)
        resume_txn = await logger.clear_and_reseal_warm_resume(
            messages, session_id=logger.session_id
        )

    assert resume_txn.outcome == "proved_and_transitioned"
    # Nothing re-trimmed: the resumed list holds the COMPLETE records (payloads
    # included) and carries no list-side descriptor. INVERTED (the old node
    # pinned `logger.archive_descriptor is descriptor`): clear-and-reseal
    # cleared the retained descriptor by construction — the resume path never
    # re-seals and never publishes a trim, and trim re-arms only through a
    # later compaction seal (the post-seal positive arm below).
    assert logger.archive_descriptor is None
    assert messages.archive_descriptor is None
    assert _dumps(messages)[1:] == loaded_dumps
    assert (
        messages[0].role is Role.system and str(messages[0].content) == "system prompt"
    )


async def _forbidden_seal(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the resume path must never re-seal (and so never re-trim)")


def _forbidden_publish(self: SessionLogger, *args: Any, **kwargs: Any) -> None:
    raise AssertionError("the resume path must never publish a trim")


@pytest.mark.asyncio
async def test_subsequent_seal_after_warm_resume_mints_a_fresh_receipt_at_the_new_epoch(
    session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
) -> None:
    """The LL post-seal positive arm (ADDED beside the retained forbidden
    seal/publish guards): after the warm-resume clear-and-reseal cleared the
    old authority, the NEXT compaction seal mints a FRESH receipt at the new
    epoch — that seal (outside the resume window) is what re-arms the trim.
    """
    config = _config()
    logger = _build_logger(session_config, "p4d-resume-rearm")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [
            mock_llm_chunk(content="<summary>first</summary>"),
            mock_llm_chunk(content="<summary>second</summary>"),
        ],
        stats,
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )
    await manager.compact()
    old = messages.archive_descriptor
    assert old is not None
    assert logger.archive_descriptor is old
    session_dir = logger.session_dir
    assert session_dir is not None
    loaded, _metadata = SessionLoader.load_session(session_dir)

    # Warm resume + clear-and-reseal: the retained authority is cleared.
    logger.apply_resumed_session(
        logger.session_id, session_dir, SessionLoader.load_metadata(session_dir)
    )
    messages.reset_preserving_system(loaded)
    resume_txn = await logger.clear_and_reseal_warm_resume(
        messages, session_id=logger.session_id
    )
    assert resume_txn.outcome == "proved_and_transitioned"
    assert logger.archive_descriptor is None
    assert messages.archive_descriptor is None
    epoch_after_resume = messages.message_epoch

    # A new settled round lands, then the subsequent seal re-arms: a fresh
    # receipt minted at the NEW epoch, never a re-pin of the old descriptor.
    receipts: list[Any] = []
    real_seal = SessionLogger.seal_archive

    async def _seal_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        receipt = await real_seal(self, *args, **kwargs)
        receipts.append(receipt)
        return receipt

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(SessionLogger, "seal_archive", _seal_spy)
        messages.append(
            LLMMessage(role=Role.user, content="third ask", message_id="u3")
        )
        messages.append(
            LLMMessage(
                role=Role.assistant,
                content="third answer",
                message_id="a3",
                reasoning_payloads=[{"opaque": "payload-3"}],
            )
        )
        await manager.compact()

    new = messages.archive_descriptor
    assert new is not None and new is not old
    assert new.message_epoch == epoch_after_resume == messages.message_epoch
    assert logger.archive_descriptor is new
    assert len(receipts) == 1
    assert receipts[0].descriptor is new, "the fresh receipt names the new descriptor"
    verify_archived_prefix_on_disk(new, _messages_path(logger))


@pytest.mark.asyncio
async def test_automatic_compaction_closed_prefix_restriction_with_a_real_logger(
    session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
) -> None:
    # Requirement: an automatic compaction that fires during an active
    # invocation leaves the active invocation intact — with the REAL sealing
    # surface wired, the guard is what stops the trim (not the flag).
    config = _config()
    logger = _build_logger(session_config, "p4d-closed-prefix")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [mock_llm_chunk(content="<summary>mid-run</summary>")], stats
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
        has_active_turn=True,
    )
    before = _dumps(_non_system(messages))

    summary = await manager.compact()

    assert summary == "mid-run"
    assert messages.archive_descriptor is None
    assert logger.archive_descriptor is None
    assert all(archive_reference(message) is None for message in messages)
    assert _dumps(_non_system(messages))[:-1] == before


# ---------------------------------------------------------------------------
# Re-seal: extension over the sealed prefix (P4b open item (a))
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reseal_after_a_real_trim_extends_the_descriptor_from_disk_bytes(
    session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
) -> None:
    config = _config()
    logger = _build_logger(session_config, "p4d-reseal")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [
            mock_llm_chunk(content="<summary>first</summary>"),
            mock_llm_chunk(content="<summary>second</summary>"),
        ],
        stats,
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )

    await manager.compact()
    first = messages.archive_descriptor
    assert first is not None
    path = _messages_path(logger)
    old_prefix = path.read_bytes()[: first.prefix_end_byte]
    assert hashlib.sha256(old_prefix).hexdigest() == first.prefix_sha256

    # A new settled round lands, then a second compaction seals the extension.
    messages.append(LLMMessage(role=Role.user, content="third ask", message_id="u3"))
    messages.append(
        LLMMessage(
            role=Role.assistant,
            content="third answer",
            message_id="a3",
            reasoning_payloads=[{"opaque": "payload-3"}],
        )
    )
    await manager.compact()

    second = messages.archive_descriptor
    assert second is not None and second is not first
    assert second.archived_count > first.archived_count
    assert second.trim_mode == "payloads"
    # Open item (a): the extension is the OLD PREFIX DISK BYTES plus the new
    # records — the old byte range is preserved exactly, never re-anchored.
    extended = path.read_bytes()[: second.prefix_end_byte]
    assert extended[: first.prefix_end_byte] == old_prefix
    assert hashlib.sha256(extended).hexdigest() == second.prefix_sha256
    verify_archived_prefix_on_disk(second, path)
    # Both owners moved to the extended descriptor; the RAM copies of the
    # previously sealed records stay trimmed (they are not revived from disk).
    assert logger.archive_descriptor is second
    sealed = _non_system(messages)[: second.archived_count]
    assert all(message.reasoning_payloads is None for message in sealed)
    assert all(archive_reference(message) is second for message in sealed)


@pytest.mark.asyncio
async def test_reseal_refuses_to_reanchor_changed_archived_bytes(
    session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
) -> None:
    config = _config()
    logger = _build_logger(session_config, "p4d-reanchor")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [
            mock_llm_chunk(content="<summary>first</summary>"),
            mock_llm_chunk(content="<summary>second</summary>"),
        ],
        stats,
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )
    await manager.compact()
    first = messages.archive_descriptor
    assert first is not None

    # Mutate the archived bytes in place (same size, interior edit).
    path = _messages_path(logger)
    raw = path.read_bytes()
    tampered = raw.replace(b"oldest ask", b"edited ask", 1)
    assert tampered != raw
    path.write_bytes(tampered)

    messages.append(LLMMessage(role=Role.user, content="third ask", message_id="u3"))
    messages.append(
        LLMMessage(
            role=Role.assistant,
            content="third answer",
            message_id="a3",
            reasoning_payloads=[{"opaque": "payload-3"}],
        )
    )
    await manager.compact()

    # No receipt, no publication: the retained descriptor is unchanged (never
    # re-anchored over mutated history) and the new round stays untrimmed.
    assert logger.archive_descriptor is first
    assert messages.archive_descriptor is first
    new_round = _non_system(messages)[-3:-1]
    assert [message.content for message in new_round] == ["third ask", "third answer"]
    assert [message.reasoning_payloads for message in new_round] == [
        None,
        [{"opaque": "payload-3"}],
    ]
    diagnostic = logger.last_save_diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "archive_unavailable"
    assert diagnostic.recoverable_by_restart is False


# ---------------------------------------------------------------------------
# Integrated end-to-end cycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integrated_compaction_trim_save_full_verify_resume_cycle(
    session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
) -> None:
    config = _config()
    logger = _build_logger(session_config, "p4d-e2e")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [mock_llm_chunk(content="<summary>cycle</summary>")], stats
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )
    original_view = _dumps(messages)

    # 1. Compaction -> trim.
    await manager.compact()
    descriptor = messages.archive_descriptor
    assert descriptor is not None
    assert logger.archive_descriptor is descriptor
    assert all(
        message.reasoning_payloads is None
        for message in _non_system(messages)[: descriptor.archived_count]
    )

    # 2. Save -> the publication invalidated the cursor, so this is the
    # archive-aware full verify: prefix hash + retained fields + live tail all
    # checked against disk, with no fail-closed diagnostic.
    await _save(logger, messages, stats, config, mock_tool_manager)
    assert logger.last_save_diagnostic is None
    logger.invalidate_transcript_cursor()
    await _save(logger, messages, stats, config, mock_tool_manager)
    assert logger.last_save_diagnostic is None
    assert logger.last_save_archive_snapshot is not None
    assert logger.last_save_archive_snapshot.descriptor is descriptor

    # 3. Open item (c): the reconstructed view is display-identical — the
    # archived prefix is hydrated from the authoritative disk bytes, so the
    # payload trim does not change what the user sees.
    loop_view = SimpleNamespace(
        messages=messages, session_id=logger.session_id, session_logger=logger
    )
    view = list(history_messages(cast(Any, loop_view)))
    # The view is the original conversation plus the compaction boundary it
    # ended with — byte-identical to the pre-trim messages.
    assert _dumps(view)[:-1] == original_view
    assert view[-1].context_boundary == "compaction"
    projected = project_message_history(
        logger.session_id, view, logger.session_metadata
    )
    projected_live = project_message_history(
        logger.session_id, messages, logger.session_metadata
    )
    assert len(projected) == len(projected_live)

    # 4. Resume: a cold load yields the complete records (disk is
    # authoritative), so the cycle restores exactly the pre-trim conversation.
    session_dir = logger.session_dir
    assert session_dir is not None
    loaded, _metadata = SessionLoader.load_session(session_dir)
    # The resumed transcript (non-system records only) is exactly the
    # reconstructed display view minus its system prompt.
    assert _dumps(loaded) == _dumps(view)[1:]
    assert loaded[-1].context_boundary == "compaction"


# ---------------------------------------------------------------------------
# Admission-side authority (the activated trim's live-prefix recognition)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_archive_authority_gates_the_admission_validators(
    session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
) -> None:
    # An admission validator over the LIVE list must recognize the sealed
    # prefix exactly while the list owns an authoritative descriptor: this is
    # what the turn-start admission / append boundary / root-recovery
    # pre-check now mint. Markers without it still fail closed (M1 §4).
    config = _config()
    logger = _build_logger(session_config, "p4d-authority")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [mock_llm_chunk(content="<summary>all done</summary>")], stats
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )

    assert live_archive_authority(messages) is None
    assert live_archive_authority(list(messages)) is None

    await manager.compact()
    descriptor = messages.archive_descriptor
    assert descriptor is not None
    authority = live_archive_authority(messages)
    assert authority is not None
    assert authority.archived_count == descriptor.archived_count
    assert authority.boundary_message_id == descriptor.boundary_message_id
    assert authority.message_epoch == messages.message_epoch
    # With the minted authority the live transcript is admitted; without it
    # the same markers are rejected (the marker is never the authority).
    assert (
        validate_transcript(messages, live_tail_exempt=True, archived_prefix=authority)
        == []
    )
    violations = validate_transcript(messages, live_tail_exempt=True)
    assert [violation.kind for violation in violations] == [
        "unauthorized_archived_prefix"
    ]

    # A stale pairing (descriptor from a departed list generation) mints
    # nothing: fail closed, even when the descriptor is forced back on.
    messages.clear_archive_descriptor()
    assert live_archive_authority(messages) is None
    messages.reset_preserving_system(list(_non_system(messages)))
    object.__setattr__(messages, "_archive_descriptor", descriptor)
    assert messages.message_epoch != descriptor.message_epoch
    assert live_archive_authority(messages) is None
    assert [
        violation.kind
        for violation in validate_transcript(messages, live_tail_exempt=True)
    ] == ["unauthorized_archived_prefix"]


# ---------------------------------------------------------------------------
# Disk / field fidelity (T3): the archived prefix bytes are exactly what the
# trim claims — the complete original records, never a partial or corrupted
# field state — while the live replacements are exactly the declared trim.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_archived_prefix_disk_bytes_are_the_exact_untrimmed_records(
    session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
) -> None:
    config = _config()
    logger = _build_logger(session_config, "p4d-disk-fidelity")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [mock_llm_chunk(content="<summary>all done</summary>")], stats
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )
    originals = [
        SessionLogger._message_record(message) for message in _non_system(messages)[:5]
    ]

    await manager.compact()

    descriptor = messages.archive_descriptor
    assert descriptor is not None
    path = _messages_path(logger)
    raw = path.read_bytes()
    # The transcript is a complete JSONL file: every record line is terminated,
    # so the archived prefix cannot end inside a record.
    assert raw.endswith(b"\n")
    prefix = raw[: descriptor.prefix_end_byte]
    assert prefix.endswith(b"\n"), "the prefix holds complete record lines"
    lines = prefix.splitlines()
    assert len(lines) == descriptor.archived_count == 5
    records = [json.loads(line) for line in lines]
    # Field fidelity on disk: every archived record equals the ORIGINAL
    # message's full serialized record — reasoning payloads, tool results,
    # image references, display fields and all — byte-for-byte as JSON.
    assert records == originals
    assert records[1]["reasoning_payloads"] == [{"opaque": "payload-1"}]
    assert records[4]["reasoning_payloads"] == [{"opaque": "payload-2"}]
    # The prefix is exactly the bytes the descriptor hashes, and the shared
    # read interface hydrates them back to the originals.
    assert hashlib.sha256(prefix).hexdigest() == descriptor.prefix_sha256
    assert [
        SessionLogger._message_record(message)
        for message in hydrate_archived_messages(descriptor, prefix)
    ] == originals
    # The live tail after the prefix starts on a record boundary: the NEXT
    # complete line is the compaction boundary envelope, not a partial record.
    tail = raw[descriptor.prefix_end_byte :]
    boundary_line = tail.split(b"\n", 1)[0]
    boundary = json.loads(boundary_line)
    assert boundary["context_boundary"] == "compaction"


@pytest.mark.asyncio
async def test_trim_leaves_only_reasoning_payloads_empty_on_disk_and_in_ram(
    session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
) -> None:
    config = _config()
    logger = _build_logger(session_config, "p4d-trim-shape")
    messages = _conversation()
    stats = AgentStats(steps=1)
    complete = _ScriptedComplete(
        [mock_llm_chunk(content="<summary>all done</summary>")], stats
    )
    manager = _manager(
        logger=logger,
        messages=messages,
        stats=stats,
        config=config,
        tool_manager=mock_tool_manager,
        complete=complete,
    )
    originals = [
        SessionLogger._message_record(message) for message in _non_system(messages)[:5]
    ]

    await manager.compact()

    descriptor = messages.archive_descriptor
    assert descriptor is not None
    raw = _messages_path(logger).read_bytes()
    disk_records = [
        json.loads(line) for line in raw[: descriptor.prefix_end_byte].splitlines()
    ]
    live_records = [
        SessionLogger._message_record(message, allow_archived=True)
        for message in _non_system(messages)[: descriptor.archived_count]
    ]
    assert len(disk_records) == len(live_records) == len(originals) == 5

    for disk_record, live_record, original in zip(
        disk_records, live_records, originals, strict=True
    ):
        # The disk record is the complete original; the live replacement is
        # the declared trim and nothing else. `reasoning_payloads` is omitted
        # from the live record exactly because the field is None, never an
        # empty list, a placeholder, or a partial dict.
        assert disk_record == original
        assert live_record == {
            key: value for key, value in original.items() if key != "reasoning_payloads"
        }
        if "reasoning_payloads" in original:
            assert original["reasoning_payloads"]
            assert "reasoning_payloads" not in live_record
            assert set(disk_record) - set(live_record) == {"reasoning_payloads"}
        else:
            assert "reasoning_payloads" not in disk_record
            assert set(disk_record) == set(live_record)
        # Every field the trim promises to preserve is identical on both
        # sides: the emptied field is the only difference, so no partial or
        # corrupted field state can hide here.
        for key in original:
            if key != "reasoning_payloads":
                assert live_record[key] == disk_record[key] == original[key]
    verify_archived_prefix_on_disk(descriptor, _messages_path(logger))
