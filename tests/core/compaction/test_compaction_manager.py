from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chartreux.core.agent_loop._file_state_context import MAX_HANDOFF_BYTES
from chartreux.core.agent_loop.errors import EmptyLLMResponseError
from chartreux.core.agent_loop.llm_gateway import apply_usage
from chartreux.core.compaction import CompactionFailedError, CompactionManager
from chartreux.core.compaction.context import (
    parse_previous_user_messages,
    render_compaction_context,
    select_model_context,
)
from chartreux.core.compaction.manager import PendingPostTurnTrim
from chartreux.core.errors import ContextTooLongError
from chartreux.core.file_outline import outline_source
from chartreux.core.file_revision import full_revision
from chartreux.core.file_state import MAX_TRACKED_FILES, FileStateStore
from chartreux.core.llm_models import (
    FunctionCall,
    LLMChunk,
    LLMMessage,
    Role,
    StopInfo,
    ToolCall,
)
from chartreux.core.message_list import MessageList
from chartreux.core.session.session_loader import MESSAGES_FILENAME
from chartreux.core.session.session_logger import (
    PostTurnTrimResult,
    SessionDiskFullError,
    SessionLogger,
)
from chartreux.core.session.transcript_archive import verify_archived_prefix_on_disk
from chartreux.core.session_types import AgentStats
from chartreux.core.tools.manager import ToolManager
from chartreux.core.usage import UsagePurpose
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
from tests.stubs.fake_backend import FakeBackend


class _FakeComplete:
    """Stand-in for AgentLoop._complete: records calls, scripts results, and
    accounts usage into stats (as the real accounted path does).
    """

    def __init__(self, results: list[LLMChunk | Exception], stats: AgentStats) -> None:
        self._results = list(results)
        self._stats = stats
        self.calls: list[dict] = []

    async def __call__(
        self,
        *,
        model,
        messages,
        tools,
        tool_choice,
        call_type,
        account_conversation,
        purpose: UsagePurpose = UsagePurpose.CONVERSATION,
    ):
        assert account_conversation is False
        self.calls.append({
            "model": model,
            "messages": list(messages),
            "tools": tools,
            "tool_choice": tool_choice,
            "call_type": call_type,
        })
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        if item.usage is not None:
            apply_usage(
                self._stats,
                item.usage,
                time_seconds=1.0,
                model=model,
                account_conversation=account_conversation,
            )
        return item


async def _noop() -> None:
    return None


def _tool_call_chunk() -> LLMChunk:
    return mock_llm_chunk(
        content="",
        tool_calls=[
            ToolCall(
                id="t1", index=0, function=FunctionCall(name="bash", arguments="{}")
            )
        ],
    )


def _build_manager(
    results: list[LLMChunk | Exception],
    *,
    messages: MessageList,
    stats: AgentStats,
    raise_on_failure: bool = False,
    file_state: FileStateStore | None = None,
) -> tuple[CompactionManager, _FakeComplete]:
    cfg = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=999),
        raise_on_compaction_failure=raise_on_failure,
    )
    complete = _FakeComplete(results, stats)
    manager = CompactionManager(
        messages=messages,
        stats_getter=lambda: stats,
        config_getter=lambda: cfg,
        complete=complete,
        available_tools=lambda: [],
        tool_choice=lambda: "auto",
        save=_noop,
        file_state_getter=(lambda: file_state) if file_state is not None else None,
    )
    return manager, complete


def _conversation() -> MessageList:
    return MessageList([
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="oldest ask"),
        LLMMessage(role=Role.assistant, content="work"),
        LLMMessage(role=Role.user, content="newest ask"),
    ])


# A scripted compaction response carrying a pending-approval handoff: design
# accepted, plan pending, approved scope with evidence, one consumed and one
# unconsumed one-time grant, and active agent/run handles.
PENDING_APPROVAL_HANDOFF = """\
Goal: implement the retry policy from the user's approved design.
Task classification: implementation (non-trivial)
Current workflow phase: plan
Dispatch policy identity: orchestrated; policy version: 1; snapshot version: 1
Contributor authorship: scope=retry.py; slot=implementor; evidence=changed retry helper; status=partial
Consumed attempt budget: scope=retry.py; used=2; remaining=0
Current recovery route: scope=retry.py; advisor diagnosis pending; no implementation retry
Design acceptance state: accepted
Plan acceptance state: pending
Approved scope: design work only; no implementation until the plan is accepted
Approval evidence: the user's message 'the design looks good, go ahead' approved the design
One-time grants:
- target: scratchpad draft of the plan; state: consumed
- target: one focused test run; state: unconsumed
Active agent/run handles: agent-7 (run-3, idle), agent-9 (run-5, running)
Outstanding dependencies: agent-9's findings feed the plan; run-3's result is uncollected
Next step: present the plan to the user and wait for explicit plan acceptance
"""


def _assert_pending_approval_handoff_intact(text: str) -> None:
    for fragment in (
        "Design acceptance state: accepted",
        "Dispatch policy identity:",
        "Contributor authorship: scope=retry.py",
        "status=partial",
        "used=2; remaining=0",
        "advisor diagnosis pending",
        "Plan acceptance state: pending",
        "Approved scope:",
        "Approval evidence:",
        "state: consumed",
        "state: unconsumed",
        "agent-7",
        "run-3",
        "agent-9",
        "run-5",
        "uncollected",
    ):
        assert fragment in text


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
async def test_typed_empty_summary_preserves_existing_fallback(strict, monkeypatch):
    stats = AgentStats(context_tokens=80, last_turn_prompt_tokens=70)
    messages = _conversation()
    manager, complete = _build_manager(
        [
            EmptyLLMResponseError("test", "test"),
            mock_llm_chunk(content="<summary>recovered</summary>"),
        ],
        messages=messages,
        stats=stats,
        raise_on_failure=strict,
    )
    success = Mock()
    monkeypatch.setattr(
        "chartreux.core.compaction.manager.log_model_call_success", success
    )
    if strict:
        with pytest.raises(CompactionFailedError, match="empty_summary"):
            await manager.compact()
        assert len(complete.calls) == 1
        assert not success.called
        assert stats.context_tokens == 80
    else:
        assert await manager.compact() == "recovered"
        assert len(complete.calls) == 2
        assert success.call_count == 1
    assert stats.last_turn_prompt_tokens == 70


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
async def test_gateway_empty_primary_replays_then_uses_only_existing_fallback(
    recover, monkeypatch
):
    terminal = "<summary>recovered</summary>" if recover else ""
    backend = FakeBackend([
        [mock_llm_chunk(content=" ")],
        [mock_llm_chunk(content="")],
        [mock_llm_chunk(content=terminal)],
        [mock_llm_chunk(content="")],
    ])
    agent = build_test_agent_loop(
        config=build_test_vibe_config(raise_on_compaction_failure=False),
        backend=backend,
    )
    agent.stats.context_tokens = 80
    agent.stats.last_turn_prompt_tokens = 70
    agent.stats.last_turn_completion_tokens = 10
    success = Mock()
    monkeypatch.setattr(
        "chartreux.core.compaction.manager.log_model_call_success", success
    )
    before = list(agent.messages)
    if recover:
        assert await agent.compact() == "recovered"
        assert success.call_count == 1
        assert len(backend.requests_messages) == 3
    else:
        with pytest.raises(CompactionFailedError, match="empty_summary"):
            await agent.compact()
        assert not success.called
        assert list(agent.messages) == before
        assert agent.stats.context_tokens == 80
        assert len(backend.requests_messages) == 4
        assert backend.requests_messages[2] == backend.requests_messages[3]
    assert backend.requests_messages[0] == backend.requests_messages[1]
    assert backend.requests_metadata[0] == backend.requests_metadata[1]
    assert backend.requests_messages[0] != backend.requests_messages[2]
    assert agent.stats.last_turn_prompt_tokens == 70
    assert agent.stats.last_turn_completion_tokens == 10
    assert agent.stats.session_prompt_tokens == 10 * len(backend.requests_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
async def test_gateway_empty_incomplete_summary_uses_existing_fallback(recover):
    incomplete = mock_llm_chunk(content="").model_copy(
        update={"stop": StopInfo(reason="incomplete")}
    )
    fallback = (
        mock_llm_chunk(content="<summary>recovered</summary>")
        if recover
        else incomplete
    )
    backend = FakeBackend([[incomplete], [fallback]])
    agent = build_test_agent_loop(
        config=build_test_vibe_config(raise_on_compaction_failure=False),
        backend=backend,
    )
    agent.stats.context_tokens = 80
    before = list(agent.messages)
    if recover:
        assert await agent.compact() == "recovered"
    else:
        with pytest.raises(CompactionFailedError) as exc_info:
            await agent.compact()
        assert exc_info.value.reason == "empty_summary"
        assert list(agent.messages) == before
        assert agent.stats.context_tokens == 80
    assert len(backend.requests_messages) == 2
    assert backend.requests_tools[1] is None
    assert backend.requests_messages[0] != backend.requests_messages[1]


@pytest.mark.asyncio
async def test_primary_success_appends_compaction_boundary() -> None:
    messages = _conversation()
    stats = AgentStats()
    manager, complete = _build_manager(
        [mock_llm_chunk(content="<summary>all done</summary>")],
        messages=messages,
        stats=stats,
    )

    summary = await manager.compact()

    assert summary == "all done"
    assert len(complete.calls) == 1  # no fallback
    assert [m.role for m in messages] == [
        Role.system,
        Role.user,
        Role.assistant,
        Role.user,
        Role.user,
    ]
    assert messages[-1].context_boundary == "compaction"
    assert "all done" in (messages[-1].content or "")
    assert parse_previous_user_messages(messages[-1].content or "") == [
        "oldest ask",
        "newest ask",
    ]
    assert stats.context_tokens == -1
    assert messages[-1].tool_result is not None
    assert messages[-1].tool_result.output == {"todo_state": {"todos": []}}


@pytest.mark.asyncio
async def test_fallback_usage_is_accounted() -> None:
    # Regression: the fallback must go through the accounted `complete` path so
    # its tokens land in session stats (the old direct backend.complete bug).
    messages = _conversation()
    stats = AgentStats()
    manager, complete = _build_manager(
        [_tool_call_chunk(), mock_llm_chunk(content="<summary>recovered</summary>")],
        messages=messages,
        stats=stats,
    )

    summary = await manager.compact()

    assert summary == "recovered"
    assert len(complete.calls) == 2  # primary + fallback both via `complete`
    assert complete.calls[1]["tools"] is None  # fallback disables tools
    # Both calls' usage is accounted (10 prompt + 5 completion each).
    assert stats.session_prompt_tokens == 20
    assert stats.session_completion_tokens == 10


@pytest.mark.asyncio
async def test_fallback_transcript_keeps_tool_calls() -> None:
    # The fallback transcript must include the assistant's tool calls (its
    # actions), not just the tool results, so trajectory context survives.
    messages = MessageList([
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="do it"),
        LLMMessage(
            role=Role.assistant,
            content="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    index=0,
                    function=FunctionCall(name="run_tests", arguments='{"path": "x"}'),
                )
            ],
        ),
        LLMMessage(role=Role.tool, content="3 failures", tool_call_id="t1"),
        LLMMessage(role=Role.user, content="fix them"),
    ])
    stats = AgentStats()
    manager, complete = _build_manager(
        [_tool_call_chunk(), mock_llm_chunk(content="<summary>done</summary>")],
        messages=messages,
        stats=stats,
    )

    await manager.compact()

    fallback_prompt = complete.calls[1]["messages"][1].content or ""
    assert "run_tests" in fallback_prompt  # assistant's tool call is preserved
    assert "3 failures" in fallback_prompt  # tool result is preserved


@pytest.mark.asyncio
async def test_overflow_trims_whole_rounds_then_succeeds() -> None:
    messages = _conversation()
    stats = AgentStats()
    manager, complete = _build_manager(
        [
            ContextTooLongError("p", "m"),
            mock_llm_chunk(content="<summary>fit now</summary>"),
        ],
        messages=messages,
        stats=stats,
    )

    summary = await manager.compact()

    assert summary == "fit now"
    # Second attempt sent fewer messages than the first (a round was dropped).
    assert len(complete.calls[1]["messages"]) < len(complete.calls[0]["messages"])
    # Prior user messages are preserved from the pre-trim snapshot.
    assert parse_previous_user_messages(messages[-1].content or "") == [
        "oldest ask",
        "newest ask",
    ]


@pytest.mark.asyncio
async def test_overflow_retry_preserves_existing_compaction_boundary() -> None:
    prior_boundary = LLMMessage(
        role=Role.user,
        content=render_compaction_context([], "prior discoveries"),
        injected=True,
        context_boundary="compaction",
    )
    messages = MessageList([
        LLMMessage(role=Role.system, content="sys"),
        prior_boundary,
        LLMMessage(role=Role.user, content="older post-compaction ask"),
        LLMMessage(role=Role.assistant, content="older work"),
        LLMMessage(role=Role.user, content="latest ask"),
    ])
    stats = AgentStats()
    manager, complete = _build_manager(
        [
            ContextTooLongError("p", "m"),
            mock_llm_chunk(content="<summary>replacement</summary>"),
        ],
        messages=messages,
        stats=stats,
        raise_on_failure=True,
    )

    await manager.compact()

    retry_messages = complete.calls[1]["messages"]
    assert prior_boundary in retry_messages
    assert any(
        "prior discoveries" in (message.content or "") for message in retry_messages
    )
    assert prior_boundary in messages
    assert messages[-1].context_boundary == "compaction"


@pytest.mark.asyncio
async def test_overflow_fails_when_only_compaction_boundary_is_trimmable() -> None:
    prior_boundary = LLMMessage(
        role=Role.user,
        content=render_compaction_context([], "prior discoveries"),
        injected=True,
        context_boundary="compaction",
    )
    messages = MessageList([
        LLMMessage(role=Role.system, content="sys"),
        prior_boundary,
        LLMMessage(role=Role.user, content="latest ask"),
    ])
    original = list(messages)
    stats = AgentStats(context_tokens=1234)
    manager, complete = _build_manager(
        [ContextTooLongError("p", "m")],
        messages=messages,
        stats=stats,
        raise_on_failure=True,
    )

    with pytest.raises(ContextTooLongError):
        await manager.compact()

    assert len(complete.calls) == 1
    assert list(messages) == original
    assert stats.context_tokens == 1234


@pytest.mark.asyncio
async def test_fallback_trims_on_overflow_then_succeeds() -> None:
    # The fallback must also recover from ContextTooLongError by dropping the
    # oldest round and retrying (same overflow handling as the primary).
    messages = _conversation()
    stats = AgentStats()
    manager, complete = _build_manager(
        [
            _tool_call_chunk(),  # primary fails (tool call)
            ContextTooLongError("p", "m"),  # fallback overflows once
            mock_llm_chunk(content="<summary>fit now</summary>"),  # fallback retries
        ],
        messages=messages,
        stats=stats,
    )

    summary = await manager.compact()

    assert summary == "fit now"
    assert len(complete.calls) == 3  # primary + fallback (overflow) + fallback (retry)
    # The fallback embeds the transcript in one user message, so the retry's
    # transcript is shorter than its first attempt (a round was dropped).
    first_transcript = complete.calls[1]["messages"][1].content or ""
    retry_transcript = complete.calls[2]["messages"][1].content or ""
    assert len(retry_transcript) < len(first_transcript)


@pytest.mark.asyncio
async def test_compaction_does_not_consume_turn_budget() -> None:
    # Compaction (primary + fallback + overflow retries) must not bump
    # stats.steps — otherwise overflow recovery could trip the turn limit and
    # abort the very turn the reactive compaction is trying to save.
    messages = _conversation()
    stats = AgentStats()
    steps_before = stats.steps
    manager, _ = _build_manager(
        [
            ContextTooLongError("p", "m"),  # primary overflows once
            _tool_call_chunk(),  # primary fails (tool call)
            mock_llm_chunk(content="<summary>done</summary>"),  # fallback succeeds
        ],
        messages=messages,
        stats=stats,
    )

    await manager.compact()

    assert stats.steps == steps_before


@pytest.mark.asyncio
async def test_live_messages_untouched_on_failure() -> None:
    messages = _conversation()
    original = list(messages)
    stats = AgentStats(context_tokens=1234)
    manager, _ = _build_manager([RuntimeError("boom")], messages=messages, stats=stats)

    with pytest.raises(RuntimeError, match="boom"):
        await manager.compact()

    # Snapshot-based generation must never mutate the live list on failure.
    assert list(messages) == original
    assert stats.context_tokens == 1234


@pytest.mark.asyncio
async def test_strict_mode_raises_on_primary_failure() -> None:
    messages = _conversation()
    original = list(messages)
    stats = AgentStats(context_tokens=1234)
    manager, complete = _build_manager(
        [_tool_call_chunk()], messages=messages, stats=stats, raise_on_failure=True
    )

    with pytest.raises(CompactionFailedError) as exc_info:
        await manager.compact()

    assert exc_info.value.reason == "tool_call"
    assert len(complete.calls) == 1  # strict: no fallback
    assert list(messages) == original
    assert stats.context_tokens == 1234


@pytest.mark.asyncio
async def test_terminal_failure_after_tool_call_preserves_context() -> None:
    messages = _conversation()
    original = list(messages)
    stats = AgentStats(context_tokens=1234)
    manager, complete = _build_manager(
        [_tool_call_chunk(), mock_llm_chunk(content="no tags here")],
        messages=messages,
        stats=stats,
    )

    with pytest.raises(CompactionFailedError) as exc_info:
        await manager.compact()

    assert exc_info.value.reason == "tool_call"
    assert len(complete.calls) == 2
    assert list(messages) == original
    assert stats.context_tokens == 1234


@pytest.mark.asyncio
async def test_terminal_failure_after_empty_summary_preserves_context() -> None:
    messages = _conversation()
    original = list(messages)
    stats = AgentStats(context_tokens=1234)
    manager, complete = _build_manager(
        [mock_llm_chunk(content="no tags"), mock_llm_chunk(content="still none")],
        messages=messages,
        stats=stats,
    )

    with pytest.raises(CompactionFailedError) as exc_info:
        await manager.compact()

    assert exc_info.value.reason == "empty_summary"
    assert list(messages) == original
    assert stats.context_tokens == 1234


@pytest.mark.asyncio
async def test_primary_compaction_preserves_pending_approval_handoff() -> None:
    messages = _conversation()
    stats = AgentStats()
    manager, complete = _build_manager(
        [mock_llm_chunk(content=f"<summary>{PENDING_APPROVAL_HANDOFF}</summary>")],
        messages=messages,
        stats=stats,
    )

    summary = await manager.compact()

    assert summary == PENDING_APPROVAL_HANDOFF.strip()
    assert len(complete.calls) == 1  # primary only; no fallback needed
    envelope = messages[-1]
    assert envelope.context_boundary == "compaction"
    _assert_pending_approval_handoff_intact(envelope.content or "")
    # The handoff reaches the next model-facing request intact.
    model_context = select_model_context(messages)
    assert any(
        PENDING_APPROVAL_HANDOFF.strip() in (message.content or "")
        for message in model_context
    )


@pytest.mark.asyncio
async def test_fallback_compaction_preserves_pending_approval_handoff() -> None:
    messages = _conversation()
    stats = AgentStats()
    manager, complete = _build_manager(
        [
            _tool_call_chunk(),  # primary fails (tool call)
            mock_llm_chunk(content=f"<summary>{PENDING_APPROVAL_HANDOFF}</summary>"),
        ],
        messages=messages,
        stats=stats,
    )

    summary = await manager.compact()

    assert summary == PENDING_APPROVAL_HANDOFF.strip()
    assert len(complete.calls) == 2  # primary + fallback
    envelope = messages[-1]
    assert envelope.context_boundary == "compaction"
    _assert_pending_approval_handoff_intact(envelope.content or "")


@pytest.mark.asyncio
async def test_repeated_compaction_preserves_pending_approval_handoff() -> None:
    # Two compaction rounds back to back: the second round summarizes the
    # first round's handoff, and the acceptance states, grant consumption,
    # and agent/run handles must survive both compactions.
    messages = _conversation()
    stats = AgentStats()
    manager, complete = _build_manager(
        [
            mock_llm_chunk(content=f"<summary>{PENDING_APPROVAL_HANDOFF}</summary>"),
            mock_llm_chunk(content=f"<summary>{PENDING_APPROVAL_HANDOFF}</summary>"),
        ],
        messages=messages,
        stats=stats,
    )

    first_summary = await manager.compact()
    second_summary = await manager.compact()

    assert first_summary == PENDING_APPROVAL_HANDOFF.strip()
    assert second_summary == PENDING_APPROVAL_HANDOFF.strip()
    assert len(complete.calls) == 2  # one primary per compaction round
    # The second round summarized the first round's handoff, not a fresh start.
    second_request = complete.calls[1]["messages"]
    assert any(
        PENDING_APPROVAL_HANDOFF.strip() in (message.content or "")
        for message in second_request
    )
    # Both boundaries remain, with the latest carrying the handoff forward.
    envelopes = [m for m in messages if m.context_boundary == "compaction"]
    assert len(envelopes) == 2
    _assert_pending_approval_handoff_intact(envelopes[-1].content or "")
    # The handoff reaches the next model-facing request intact.
    model_context = select_model_context(messages)
    assert any(
        PENDING_APPROVAL_HANDOFF.strip() in (message.content or "")
        for message in model_context
    )


@pytest.mark.asyncio
async def test_dispatch_handoff_survives_repeated_overflow_and_fallback() -> None:
    messages = _conversation()
    stats = AgentStats()
    manager, complete = _build_manager(
        [
            mock_llm_chunk(content=f"<summary>{PENDING_APPROVAL_HANDOFF}</summary>"),
            ContextTooLongError("p", "m"),
            _tool_call_chunk(),
            mock_llm_chunk(content=f"<summary>{PENDING_APPROVAL_HANDOFF}</summary>"),
        ],
        messages=messages,
        stats=stats,
    )
    await manager.compact()
    messages.extend([
        LLMMessage(role=Role.user, content="old continuation"),
        LLMMessage(role=Role.assistant, content="work"),
        LLMMessage(role=Role.user, content="latest continuation"),
    ])
    await manager.compact()
    for call in complete.calls[1:]:
        text = "\n".join(message.content or "" for message in call["messages"])
        _assert_pending_approval_handoff_intact(text)
        assert "missing budget means consumed" in text.lower()
    _assert_pending_approval_handoff_intact(messages[-1].content or "")


@pytest.mark.asyncio
async def test_compaction_envelope_carries_file_state_alongside_todo_state() -> None:
    messages = _conversation()
    stats = AgentStats()
    file_state = FileStateStore()
    file_state.record_revision("/repo/a.py", full_revision(b"a = 1\n"))
    file_state.record_revision("/repo/b.py", full_revision(b"b = 2\n"))
    manager, _complete = _build_manager(
        [mock_llm_chunk(content="<summary>all done</summary>")],
        messages=messages,
        stats=stats,
        file_state=file_state,
    )

    await manager.compact()

    boundary = messages[-1]
    assert boundary.context_boundary == "compaction"
    assert boundary.tool_result is not None
    output = boundary.tool_result.output
    # The existing TodoState replay keeps working alongside file_state.
    assert output["todo_state"] == {"todos": []}
    assert "file_state" in output
    # Structural replay restores the working-set state (metadata only).
    replayed = FileStateStore.replay(messages)
    assert replayed.known_revision("/repo/a.py") == full_revision(b"a = 1\n")
    assert replayed.known_revision("/repo/b.py") == full_revision(b"b = 2\n")
    assert len(replayed) == 2


@pytest.mark.asyncio
async def test_compaction_trims_file_state_to_cap_at_the_boundary() -> None:
    messages = _conversation()
    stats = AgentStats()
    file_state = FileStateStore()
    for index in range(MAX_TRACKED_FILES + 5):
        file_state.record_revision(
            f"/repo/file_{index}.py", full_revision(f"{index}".encode())
        )
    assert file_state.is_over_cap()
    manager, _complete = _build_manager(
        [mock_llm_chunk(content="<summary>all done</summary>")],
        messages=messages,
        stats=stats,
        file_state=file_state,
    )

    await manager.compact()

    # Eviction happens only here, at the boundary: the store is trimmed to the
    # cap and the envelope carries at most MAX_TRACKED_FILES entries.
    assert not file_state.is_over_cap()
    assert len(file_state) == MAX_TRACKED_FILES
    replayed = FileStateStore.replay(messages)
    assert len(replayed) == MAX_TRACKED_FILES
    # The oldest entries were evicted first.
    assert "/repo/file_0.py" not in replayed
    assert f"/repo/file_{MAX_TRACKED_FILES + 4}.py" in replayed


@pytest.mark.asyncio
async def test_repeated_compaction_last_boundary_wins() -> None:
    messages = _conversation()
    stats = AgentStats()
    file_state = FileStateStore()
    file_state.record_revision("/repo/a.py", full_revision(b"v1"))
    manager, _complete = _build_manager(
        [mock_llm_chunk(content="<summary>first</summary>")],
        messages=messages,
        stats=stats,
        file_state=file_state,
    )
    await manager.compact()
    # Post-compaction observation: the live store moves ahead of the envelope.
    file_state.record_revision("/repo/a.py", full_revision(b"v2"))
    file_state.record_revision("/repo/b.py", full_revision(b"b"))
    manager, _complete = _build_manager(
        [mock_llm_chunk(content="<summary>second</summary>")],
        messages=messages,
        stats=stats,
        file_state=file_state,
    )
    await manager.compact()

    replayed = FileStateStore.replay(messages)
    assert replayed.known_revision("/repo/a.py") == full_revision(b"v2")
    assert replayed.known_revision("/repo/b.py") == full_revision(b"b")


@pytest.mark.asyncio
async def test_fallback_and_overflow_compaction_still_carry_file_state() -> None:
    messages = _conversation()
    stats = AgentStats()
    file_state = FileStateStore()
    file_state.record_revision("/repo/a.py", full_revision(b"a"))
    # Primary overflows once, then returns a tool call (fallback path), and
    # the fallback succeeds: the envelope is built once, after the summary.
    manager, _complete = _build_manager(
        [
            ContextTooLongError("p", "m"),
            _tool_call_chunk(),
            mock_llm_chunk(content="<summary>recovered</summary>"),
        ],
        messages=messages,
        stats=stats,
        file_state=file_state,
    )

    summary = await manager.compact()

    assert summary == "recovered"
    boundary = messages[-1]
    assert boundary.context_boundary == "compaction"
    assert boundary.tool_result is not None
    assert "file_state" in boundary.tool_result.output
    assert "todo_state" in boundary.tool_result.output
    replayed = FileStateStore.replay(messages)
    assert replayed.known_revision("/repo/a.py") == full_revision(b"a")


@pytest.mark.asyncio
async def test_compaction_boundary_content_serves_the_working_set() -> None:
    # B3 regression: the boundary message's CONTENT — the model-visible part;
    # the backend drops tool_result from the converted request — carries the
    # bounded working-set summary: paths, last-seen revisions, outlines.
    messages = _conversation()
    stats = AgentStats()
    file_state = FileStateStore()
    file_state.record_observation(
        "/repo/a.py",
        revision=full_revision(b"def f():\n    return 1\n"),
        outline=outline_source("def f():\n    return 1\n"),
    )
    file_state.record_revision("/repo/b.py", full_revision(b"b = 2\n"))
    manager, _complete = _build_manager(
        [mock_llm_chunk(content="<summary>all done</summary>")],
        messages=messages,
        stats=stats,
        file_state=file_state,
    )

    await manager.compact()

    boundary = messages[-1]
    assert boundary.context_boundary == "compaction"
    content = str(boundary.content)
    # The model-facing content lists the working set with revisions and
    # outlines, without re-reading the files.
    assert "<file_state_summary>" in content
    assert "/repo/a.py" in content
    assert full_revision(b"def f():\n    return 1\n").revision in content
    assert "function f (lines 1-2)" in content
    assert "/repo/b.py" in content
    assert full_revision(b"b = 2\n").revision in content
    # The envelope still carries the structural state alongside todo_state.
    assert boundary.tool_result is not None
    output = boundary.tool_result.output
    assert "file_state" in output
    assert output["todo_state"] == {"todos": []}
    # The boundary (with its content) is the model context after compaction.
    model_context = select_model_context(messages)
    assert model_context[-1] is boundary


@pytest.mark.asyncio
async def test_second_compaction_re_renders_the_working_set_summary() -> None:
    # B3: each compaction boundary re-renders the current working set; the
    # earlier boundary is immutable history.
    messages = _conversation()
    stats = AgentStats()
    file_state = FileStateStore()
    file_state.record_revision("/repo/a.py", full_revision(b"v1"))
    manager, _complete = _build_manager(
        [mock_llm_chunk(content="<summary>first</summary>")],
        messages=messages,
        stats=stats,
        file_state=file_state,
    )
    await manager.compact()
    first_boundary = messages[-1]
    first_content = str(first_boundary.content)
    assert full_revision(b"v1").revision in first_content

    # Post-compaction observation: the live store moves ahead of the envelope.
    file_state.record_revision("/repo/a.py", full_revision(b"v2"))
    manager, _complete = _build_manager(
        [mock_llm_chunk(content="<summary>second</summary>")],
        messages=messages,
        stats=stats,
        file_state=file_state,
    )
    await manager.compact()

    second_boundary = messages[-1]
    assert second_boundary is not first_boundary
    second_content = str(second_boundary.content)
    # The second boundary re-renders the current working set ...
    assert full_revision(b"v2").revision in second_content
    assert full_revision(b"v1").revision not in second_content
    # ... and the first boundary is unchanged (immutable history).
    assert str(first_boundary.content) == first_content


@pytest.mark.asyncio
async def test_compaction_file_state_summary_is_bounded_with_a_marker() -> None:
    # B3: the rendered working-set summary stays bounded (the store cap plus
    # the MAX_HANDOFF_BYTES-style budget); the file list truncates with a
    # degradation marker.
    messages = _conversation()
    stats = AgentStats()
    file_state = FileStateStore()
    big = "class Big:\n" + "".join(
        f"    def method_{i}(self):\n        return {i}\n" for i in range(200)
    )
    for index in range(MAX_TRACKED_FILES):
        file_state.record_observation(
            f"/repo/big_{index}.py",
            revision=full_revision(f"big{index}".encode()),
            outline=outline_source(big),
        )
    manager, _complete = _build_manager(
        [mock_llm_chunk(content="<summary>all done</summary>")],
        messages=messages,
        stats=stats,
        file_state=file_state,
    )

    await manager.compact()

    content = str(messages[-1].content)
    # The store cap was enforced at the boundary (trim_to_cap) ...
    assert not file_state.is_over_cap()
    assert len(file_state) == MAX_TRACKED_FILES
    # ... and the rendered section truncated with a degradation marker.
    assert "summary budget reached" in content
    section = content[content.index("<file_state_summary>") :]
    assert len(section.encode("utf-8")) <= MAX_HANDOFF_BYTES + 200


@pytest.mark.asyncio
async def test_envelope_without_file_state_getter_is_unchanged() -> None:
    # Backward compatibility: a manager built without a file-state getter
    # produces exactly the pre-existing envelope and boundary content.
    messages = _conversation()
    stats = AgentStats()
    manager, _complete = _build_manager(
        [mock_llm_chunk(content="<summary>all done</summary>")],
        messages=messages,
        stats=stats,
    )

    await manager.compact()

    assert messages[-1].tool_result is not None
    assert messages[-1].tool_result.output == {"todo_state": {"todos": []}}
    assert "<file_state_summary>" not in str(messages[-1].content)


@pytest.mark.asyncio
async def test_archive_trim_path_is_entered_after_activation() -> None:
    # P4d ACTIVATES the trim: a successful compaction enters the trim path —
    # proven by a SPY on CompactionManager._trim_archived_prefix. A manager
    # with no sealing surface (no session logger) still leaves the list
    # unmarked: the path is entered, then declines without a sealing surface.
    from chartreux.core.agent_loop._transcript_integrity import archive_reference
    from chartreux.core.compaction.manager import (
        ARCHIVE_TRIM_ENABLED,
        CompactionManager as _Manager,
    )

    assert ARCHIVE_TRIM_ENABLED is True
    messages = _conversation()
    stats = AgentStats()
    manager, _complete = _build_manager(
        [mock_llm_chunk(content="<summary>all done</summary>")],
        messages=messages,
        stats=stats,
    )
    assert manager._trim_enabled() is True

    with patch.object(_Manager, "_trim_archived_prefix", autospec=True) as trim_spy:
        summary = await manager.compact()

    assert summary == "all done"
    trim_spy.assert_awaited_once()
    # No sealing surface was wired, so nothing was sealed or marked; the
    # boundary envelope is appended and the conversation is otherwise intact.
    assert messages[-1].context_boundary == "compaction"
    assert messages.archive_descriptor is None
    assert all(archive_reference(m) is None for m in messages)
    assert len(messages) == 5


@pytest.mark.asyncio
async def test_compaction_during_active_invocation_leaves_the_active_invocation_intact() -> (
    None
):
    # Closed-prefix restriction, compaction-wiring side: an automatic
    # compaction that fires during an active invocation leaves the active
    # invocation intact until a subsequent compaction. Trimming is ACTIVE at
    # P4d (the flag is on by default), so the active-turn guard is what must
    # keep the sealing surface untouched and the list unmarked — finalization
    # still mutates an earlier assistant's duration after this point.
    from chartreux.core.agent_loop._transcript_integrity import archive_reference

    messages = _conversation()
    stats = AgentStats()
    cfg = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=999),
        raise_on_compaction_failure=False,
    )
    complete = _FakeComplete(
        [mock_llm_chunk(content="<summary>mid-run</summary>")], stats
    )
    fake_logger = Mock()
    fake_logger.enabled = True
    fake_logger.seal_archive = AsyncMock()
    manager = CompactionManager(
        messages=messages,
        stats_getter=lambda: stats,
        config_getter=lambda: cfg,
        complete=complete,
        available_tools=lambda: [],
        tool_choice=lambda: "auto",
        save=_noop,
        session_logger_getter=lambda: fake_logger,
        tool_manager_getter=lambda: Mock(),
        has_active_turn=lambda: True,
        trim_enabled_getter=lambda: True,
    )

    summary = await manager.compact()

    assert summary == "mid-run"
    # The compaction itself succeeded (boundary appended), but the trim path
    # never reached the sealing surface: the active invocation is intact.
    fake_logger.seal_archive.assert_not_called()
    assert messages[-1].context_boundary == "compaction"
    assert messages.archive_descriptor is None
    assert all(archive_reference(m) is None for m in messages)
    assert [m.content for m in messages if m.role is not Role.system] == [
        "oldest ask",
        "work",
        "newest ask",
        messages[-1].content,
    ]


def _boundary_message(content: str = "compaction boundary") -> LLMMessage:
    return LLMMessage(
        role=Role.user, content=content, injected=True, context_boundary="compaction"
    )


class _PendingTrimBox:
    """Stand-in for the loop's pending-trim holder (getter + recorder pair)."""

    def __init__(self) -> None:
        self.value: PendingPostTurnTrim | None = None

    def getter(self) -> PendingPostTurnTrim | None:
        return self.value

    def recorder(self, pending: PendingPostTurnTrim | None) -> None:
        self.value = pending


def _build_post_turn_manager(
    *,
    messages: MessageList,
    stats: AgentStats,
    active_turn: bool,
    box: _PendingTrimBox,
    seal_receipt: Mock | None = None,
) -> tuple[CompactionManager, Mock]:
    cfg = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=999),
        raise_on_compaction_failure=False,
    )
    complete = _FakeComplete(
        [mock_llm_chunk(content="<summary>post-turn</summary>")], stats
    )
    fake_logger = Mock()
    fake_logger.enabled = True
    fake_logger.seal_archive = AsyncMock()
    # T1b: the entry completes the recorded pending through the logger's
    # drain/fence/seal/publish method (a Mock intercepts it, so seal_archive and
    # fence_and_publish_archive_trim are never reached through the entry).
    fake_logger.complete_post_turn_trim = AsyncMock(
        return_value=PostTurnTrimResult("published")
    )
    if seal_receipt is not None:
        fake_logger.seal_archive.return_value = seal_receipt
    manager = CompactionManager(
        messages=messages,
        stats_getter=lambda: stats,
        config_getter=lambda: cfg,
        complete=complete,
        available_tools=lambda: [],
        tool_choice=lambda: "auto",
        save=_noop,
        session_logger_getter=lambda: fake_logger,
        tool_manager_getter=lambda: Mock(),
        has_active_turn=lambda: active_turn,
        trim_enabled_getter=lambda: True,
        pending_trim_getter=box.getter,
        pending_trim_recorder=box.recorder,
    )
    return manager, fake_logger


@pytest.mark.asyncio
async def test_post_turn_entry_records_then_completes_the_pending_at_the_seam() -> None:
    # T1b positive arm (mocked logger): at the post-finalization seam (the turn
    # is still registered — after seal_turn, before _active_turn = None) the
    # guarded entry records pending state keyed by session/epoch/boundary (the
    # single recording point) and then makes it LIVE at the same seam by
    # awaiting the logger's drain/fence/seal/publish completion with the
    # recorded key. The holder is consumed at the attempt boundary.
    messages = _conversation()
    boundary = _boundary_message()
    messages.append(boundary)
    stats = AgentStats()
    box = _PendingTrimBox()
    manager, fake_logger = _build_post_turn_manager(
        messages=messages, stats=stats, active_turn=True, box=box
    )

    await manager.capture_post_turn_pending_trim(
        session_id="session-1", message_epoch=messages.message_epoch
    )

    # The completion ran exactly once, with the recorded pending key.
    fake_logger.complete_post_turn_trim.assert_awaited_once()
    call = fake_logger.complete_post_turn_trim.await_args
    assert call.kwargs["session_id"] == "session-1"
    assert call.kwargs["message_epoch"] == messages.message_epoch
    assert call.kwargs["boundary_message_id"] == boundary.message_id
    # The holder is consumed at the attempt boundary.
    assert box.value is None
    # Through the mocked logger the raw sealing surface is never reached by the
    # entry itself (the completion method owns it).
    fake_logger.seal_archive.assert_not_called()
    fake_logger.fence_and_publish_archive_trim.assert_not_called()


@pytest.mark.asyncio
async def test_post_turn_entry_coalesces_to_the_newest_boundary() -> None:
    # Coalescing: a second post-turn attempt with a newer boundary replaces
    # the pending record (newest wins, never stacked) and the completion runs
    # with the newest boundary.
    messages = _conversation()
    first = _boundary_message("first boundary")
    messages.append(first)
    stats = AgentStats()
    box = _PendingTrimBox()
    manager, fake_logger = _build_post_turn_manager(
        messages=messages, stats=stats, active_turn=True, box=box
    )

    await manager.capture_post_turn_pending_trim(
        session_id="session-1", message_epoch=messages.message_epoch
    )
    assert fake_logger.complete_post_turn_trim.await_count == 1
    assert (
        fake_logger.complete_post_turn_trim.await_args.kwargs["boundary_message_id"]
        == first.message_id
    )

    second = _boundary_message("second boundary")
    messages.append(second)
    await manager.capture_post_turn_pending_trim(
        session_id="session-1", message_epoch=messages.message_epoch
    )

    assert fake_logger.complete_post_turn_trim.await_count == 2
    assert (
        fake_logger.complete_post_turn_trim.await_args.kwargs["boundary_message_id"]
        == second.message_id
    )
    assert box.value is None
    fake_logger.seal_archive.assert_not_called()
    fake_logger.fence_and_publish_archive_trim.assert_not_called()


@pytest.mark.asyncio
async def test_post_turn_entry_never_invents_a_boundary() -> None:
    # No compaction boundary in the list: the entry records nothing and never
    # reaches the completion — a boundary is never created merely to trim.
    messages = _conversation()
    stats = AgentStats()
    box = _PendingTrimBox()
    manager, fake_logger = _build_post_turn_manager(
        messages=messages, stats=stats, active_turn=True, box=box
    )

    await manager.capture_post_turn_pending_trim(
        session_id="session-1", message_epoch=messages.message_epoch
    )

    assert box.value is None
    fake_logger.complete_post_turn_trim.assert_not_called()
    fake_logger.seal_archive.assert_not_called()
    fake_logger.fence_and_publish_archive_trim.assert_not_called()


@pytest.mark.asyncio
async def test_post_turn_entry_skips_an_invalid_prefix() -> None:
    # A boundary with nothing settled before it: the eligible prefix is
    # invalid (None), so the entry skips it and records nothing.
    messages = MessageList([
        LLMMessage(role=Role.system, content="sys"),
        _boundary_message(),
    ])
    stats = AgentStats()
    box = _PendingTrimBox()
    manager, fake_logger = _build_post_turn_manager(
        messages=messages, stats=stats, active_turn=True, box=box
    )

    await manager.capture_post_turn_pending_trim(
        session_id="session-1", message_epoch=messages.message_epoch
    )

    assert box.value is None
    fake_logger.complete_post_turn_trim.assert_not_called()
    fake_logger.seal_archive.assert_not_called()
    fake_logger.fence_and_publish_archive_trim.assert_not_called()


@pytest.mark.asyncio
async def test_post_turn_entry_discards_a_stale_pending_on_rebind() -> None:
    # Stale rebind discard — the T1a MUTATION TARGET (design JJJ): the epoch
    # check is what makes this arm discriminating. The discard is observed AT
    # THE ATTEMPT BOUNDARY: after a rebind (new message epoch, no boundary in
    # the rebound list) the stale pending is cleared and no completion runs.
    # Removing the epoch check leaves the stale pending in place, so the
    # cleared-state assertion below fails.
    messages = _conversation()
    boundary = _boundary_message()
    messages.append(boundary)
    stats = AgentStats()
    box = _PendingTrimBox()
    manager, fake_logger = _build_post_turn_manager(
        messages=messages, stats=stats, active_turn=True, box=box
    )

    await manager.capture_post_turn_pending_trim(
        session_id="session-1", message_epoch=messages.message_epoch
    )
    assert fake_logger.complete_post_turn_trim.await_count == 1

    # Rebind (reset/resume): new message epoch, no boundary in the new list.
    messages.reset([
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="fresh ask"),
    ])
    new_epoch = messages.message_epoch
    assert new_epoch != 0

    await manager.capture_post_turn_pending_trim(
        session_id="session-1", message_epoch=new_epoch
    )

    # The discard, observed at the attempt boundary: pending cleared, no
    # completion for the stale key.
    assert box.value is None
    assert fake_logger.complete_post_turn_trim.await_count == 1
    fake_logger.seal_archive.assert_not_called()
    fake_logger.fence_and_publish_archive_trim.assert_not_called()


@pytest.mark.asyncio
async def test_baseline_trim_path_still_seals_when_no_turn_is_active() -> None:
    # The baseline trim path (the seal+publish block in _trim_archived_prefix)
    # is UNCHANGED by T1b — with no active turn it still seals and publishes
    # exactly as before.
    messages = _conversation()
    messages.append(_boundary_message())
    stats = AgentStats()
    box = _PendingTrimBox()
    receipt = Mock()
    receipt.descriptor.archived_count = 2  # eligible prefix of _conversation()+boundary
    receipt.descriptor.trim_mode = "payloads"
    manager, fake_logger = _build_post_turn_manager(
        messages=messages, stats=stats, active_turn=False, box=box, seal_receipt=receipt
    )

    await manager._trim_archived_prefix()

    fake_logger.seal_archive.assert_awaited_once()
    fake_logger.fence_and_publish_archive_trim.assert_called_once()
    # The post-turn entry's seam guard declines here (no turn is registered),
    # so it records nothing and never reaches the completion.
    await manager.capture_post_turn_pending_trim(
        session_id="session-1", message_epoch=messages.message_epoch
    )
    assert box.value is None
    fake_logger.complete_post_turn_trim.assert_not_called()
    fake_logger.seal_archive.assert_awaited_once()


# ---------------------------------------------------------------------------
# T1b completion arms with a REAL SessionLogger (drain/fence/seal/publish)
# ---------------------------------------------------------------------------


def _messages_path(logger: SessionLogger) -> Path:
    session_dir = logger.session_dir
    assert session_dir is not None
    return session_dir / MESSAGES_FILENAME


def _post_turn_config() -> Any:
    return build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=999),
        raise_on_compaction_failure=False,
    )


def _build_real_post_turn_manager(
    *,
    logger: SessionLogger,
    messages: MessageList,
    stats: AgentStats,
    tool_manager: ToolManager,
    box: _PendingTrimBox,
    active_turn: bool = True,
) -> CompactionManager:
    cfg = _post_turn_config()
    complete = _FakeComplete([], stats)
    return CompactionManager(
        messages=messages,
        stats_getter=lambda: stats,
        config_getter=lambda: cfg,
        complete=complete,
        available_tools=lambda: [],
        tool_choice=lambda: "auto",
        save=_noop,
        session_logger_getter=lambda: logger,
        tool_manager_getter=lambda: tool_manager,
        has_active_turn=lambda: active_turn,
        trim_enabled_getter=lambda: True,
        pending_trim_getter=box.getter,
        pending_trim_recorder=box.recorder,
    )


def _non_system(messages: MessageList) -> list[LLMMessage]:
    return [message for message in messages if message.role is not Role.system]


def _outcome_spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every complete_post_turn_trim outcome in call order."""
    outcomes: list[str] = []
    real_complete = SessionLogger.complete_post_turn_trim

    async def _spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        result = await real_complete(self, *args, **kwargs)
        outcomes.append(result.outcome)
        return result

    monkeypatch.setattr(SessionLogger, "complete_post_turn_trim", _spy)
    return outcomes


@pytest.mark.asyncio
async def test_post_turn_entry_seals_and_publishes_at_the_seam_with_a_real_logger(
    session_config: Any, mock_tool_manager: ToolManager
) -> None:
    # T1b positive post-turn arm with a real logger: the guarded entry drains,
    # fences, seals and publishes the recorded pending at the exclusive seam.
    from chartreux.core.agent_loop._transcript_integrity import archive_reference

    config = _post_turn_config()
    logger = SessionLogger(session_config, "t1b-seam")
    messages = _conversation()
    boundary = _boundary_message()
    boundary_id = boundary.message_id
    assert boundary_id is not None
    messages.append(boundary)
    stats = AgentStats()
    # The boundary is persisted with the originals present (the compact() save).
    await logger.save_interaction(messages, stats, config, mock_tool_manager)
    box = _PendingTrimBox()
    manager = _build_real_post_turn_manager(
        logger=logger,
        messages=messages,
        stats=stats,
        tool_manager=mock_tool_manager,
        box=box,
    )
    generation_before = logger.transcript_save_generation

    await manager.capture_post_turn_pending_trim(
        session_id="t1b-seam", message_epoch=messages.message_epoch
    )

    # Descriptor advance: both owners carry the new descriptor for the boundary.
    descriptor = messages.archive_descriptor
    assert descriptor is not None
    assert descriptor.archived_count == 2  # settled round before the boundary
    assert descriptor.boundary_message_id == boundary.message_id
    assert logger.archive_descriptor is descriptor
    # Fenced publication: the save generation moved (the fence ran) and the
    # sealed slots carry the archive marker; the tail stays live.
    assert logger.transcript_save_generation > generation_before
    sealed = _non_system(messages)[: descriptor.archived_count]
    assert all(archive_reference(message) is descriptor for message in sealed)
    tail = _non_system(messages)[descriptor.archived_count :]
    assert [message.context_boundary for message in tail] == [None, "compaction"]
    assert all(archive_reference(message) is None for message in tail)
    # The archived prefix on disk verifies against the descriptor.
    verify_archived_prefix_on_disk(descriptor, _messages_path(logger))
    # The holder is consumed at the attempt boundary.
    assert box.value is None
    # Exclusivity: a second attempt for the same boundary is excluded, not a
    # concurrent seal/publication.
    result = await logger.complete_post_turn_trim(
        messages,
        stats,
        config,
        mock_tool_manager,
        session_id="t1b-seam",
        message_epoch=messages.message_epoch,
        boundary_message_id=boundary_id,
    )
    assert result.outcome == "excluded_already_published"
    assert messages.archive_descriptor is descriptor


@pytest.mark.asyncio
async def test_post_turn_completion_is_exclusive_per_boundary(
    session_config: Any, mock_tool_manager: ToolManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exclusivity arm: while one drain/fence/seal/publish attempt for a boundary
    # is in flight, a concurrent attempt for the SAME boundary is excluded at
    # the attempt boundary — exactly one seal and one publication happen.
    config = _post_turn_config()
    logger = SessionLogger(session_config, "t1b-exclusive")
    messages = _conversation()
    boundary = _boundary_message()
    messages.append(boundary)
    stats = AgentStats()
    await logger.save_interaction(messages, stats, config, mock_tool_manager)

    seal_calls = 0
    real_seal = SessionLogger.seal_archive

    async def _seal_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        nonlocal seal_calls
        seal_calls += 1
        return await real_seal(self, *args, **kwargs)

    fence_calls = 0
    real_fence = SessionLogger.fence_and_publish_archive_trim

    def _fence_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        nonlocal fence_calls
        fence_calls += 1
        return real_fence(self, *args, **kwargs)

    # Gate the sealing plan (runs on a worker thread) so the first attempt is
    # observably in flight inside the seal when the second attempt arrives.
    plan_entered = threading.Event()
    plan_release = threading.Event()
    real_plan = SessionLogger._plan_archive_seal

    def _gated_plan(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        plan_entered.set()
        plan_release.wait(timeout=30)
        return real_plan(self, *args, **kwargs)

    monkeypatch.setattr(SessionLogger, "seal_archive", _seal_spy)
    monkeypatch.setattr(SessionLogger, "fence_and_publish_archive_trim", _fence_spy)
    monkeypatch.setattr(SessionLogger, "_plan_archive_seal", _gated_plan)

    kwargs = {
        "session_id": "t1b-exclusive",
        "message_epoch": messages.message_epoch,
        "boundary_message_id": boundary.message_id,
    }
    first = asyncio.create_task(
        logger.complete_post_turn_trim(
            messages, stats, config, mock_tool_manager, **kwargs
        )
    )
    try:
        deadline = asyncio.get_running_loop().time() + 10
        while not plan_entered.is_set():
            assert asyncio.get_running_loop().time() < deadline, (
                "the first attempt never reached the sealing plan"
            )
            await asyncio.sleep(0.005)
        # The first attempt holds the boundary in flight: a concurrent attempt
        # for the same boundary is excluded at the attempt boundary.
        second = await logger.complete_post_turn_trim(
            messages, stats, config, mock_tool_manager, **kwargs
        )
        assert second.outcome == "excluded_concurrent"
        assert seal_calls == 1
        plan_release.set()
        first_result = await first
    finally:
        plan_release.set()
    assert first_result.outcome == "published"
    # Exactly one seal and one fenced publication for the boundary.
    assert seal_calls == 1
    assert fence_calls == 1
    assert logger._trim_published_boundaries == {boundary.message_id}
    assert logger._trim_in_flight_boundaries == set()
    assert messages.archive_descriptor is not None


@pytest.mark.asyncio
async def test_post_turn_completion_drains_queued_saves_before_fencing(
    session_config: Any, mock_tool_manager: ToolManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Drain arm: a save queued/in-flight when the completion starts is drained
    # (it finishes under the save lock) BEFORE the sealing persist runs, and
    # the publication fence moves the save generation afterwards.
    config = _post_turn_config()
    logger = SessionLogger(session_config, "t1b-drain")
    messages = _conversation()
    boundary = _boundary_message()
    boundary_id = boundary.message_id
    assert boundary_id is not None
    messages.append(boundary)
    stats = AgentStats()
    await logger.save_interaction(messages, stats, config, mock_tool_manager)

    order: list[str] = []
    save_entered = threading.Event()
    save_release = threading.Event()
    real_sync = SessionLogger._save_interaction_sync
    sync_calls = 0

    def _gated_sync(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            # The queued ordinary save's persist: block while holding the lock.
            order.append("save-persist-start")
            save_entered.set()
            save_release.wait(timeout=30)
            order.append("save-persist-end")
        else:
            order.append("seal-persist-start")
        return real_sync(self, *args, **kwargs)

    monkeypatch.setattr(SessionLogger, "_save_interaction_sync", _gated_sync)
    messages.append(LLMMessage(role=Role.assistant, content="extra answer"))
    queued_save = asyncio.create_task(
        logger.save_interaction(messages, stats, config, mock_tool_manager)
    )
    try:
        deadline = asyncio.get_running_loop().time() + 10
        while not save_entered.is_set():
            assert asyncio.get_running_loop().time() < deadline, (
                "the queued save never started its persist"
            )
            await asyncio.sleep(0.005)
        generation_before = logger.transcript_save_generation
        completion = asyncio.create_task(
            logger.complete_post_turn_trim(
                messages,
                stats,
                config,
                mock_tool_manager,
                session_id="t1b-drain",
                message_epoch=messages.message_epoch,
                boundary_message_id=boundary_id,
            )
        )
        # While the queued save holds the lock, the completion is still draining.
        await asyncio.sleep(0.05)
        assert order == ["save-persist-start"]
        assert not completion.done()
        save_release.set()
        await queued_save
        result = await completion
    finally:
        save_release.set()
    assert result.outcome == "published"
    # The queued save drained completely before the sealing persist began.
    assert order == ["save-persist-start", "save-persist-end", "seal-persist-start"]
    # The publication fence moved the save generation.
    assert logger.transcript_save_generation > generation_before


@pytest.mark.asyncio
async def test_post_turn_completion_failed_seal_retains_originals_and_recovers(
    session_config: Any, mock_tool_manager: ToolManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Failed-finalization arm: the sealing persist fails — no receipt, the
    # originals are retained, nothing is published, the holder is consumed, and
    # the turn is still complete (a later seam retries and succeeds).
    from chartreux.core.agent_loop._transcript_integrity import archive_reference

    config = _post_turn_config()
    logger = SessionLogger(session_config, "t1b-failed")
    messages = _conversation()
    boundary = _boundary_message()
    messages.append(boundary)
    stats = AgentStats()
    await logger.save_interaction(messages, stats, config, mock_tool_manager)
    box = _PendingTrimBox()
    manager = _build_real_post_turn_manager(
        logger=logger,
        messages=messages,
        stats=stats,
        tool_manager=mock_tool_manager,
        box=box,
    )
    outcomes = _outcome_spy(monkeypatch)
    real_sync = SessionLogger._seal_persist_sync

    def _disk_full(*args: Any, **kwargs: Any) -> Any:
        raise SessionDiskFullError(_messages_path(logger), "seal persist")

    monkeypatch.setattr(SessionLogger, "_seal_persist_sync", _disk_full)

    await manager.capture_post_turn_pending_trim(
        session_id="t1b-failed", message_epoch=messages.message_epoch
    )

    assert outcomes == ["failed_seal"]
    # Originals retained: no descriptor on either owner, no markers, the
    # transcript on disk still holds the boundary and the originals.
    assert messages.archive_descriptor is None
    assert logger.archive_descriptor is None
    assert all(archive_reference(message) is None for message in messages)
    assert box.value is None
    raw = _messages_path(logger).read_bytes()
    assert b'"context_boundary": "compaction"' in raw
    # The turn is still complete: with the failure removed, a later seam
    # retries and publishes.
    monkeypatch.setattr(SessionLogger, "_seal_persist_sync", real_sync)
    await manager.capture_post_turn_pending_trim(
        session_id="t1b-failed", message_epoch=messages.message_epoch
    )
    assert outcomes == ["failed_seal", "published"]
    assert messages.archive_descriptor is not None
    assert logger.archive_descriptor is messages.archive_descriptor


@pytest.mark.asyncio
async def test_post_turn_completion_cancelled_retains_originals(
    session_config: Any, mock_tool_manager: ToolManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Cancelled-finalization arm: the completion is cancelled mid-seal — the
    # in-flight persist is drained inside seal_archive, no receipt is issued,
    # nothing is published, the originals are retained, the holder is consumed,
    # and the entry absorbs the cancellation so the completed turn does not
    # become a failure (a later seam works).
    from chartreux.core.agent_loop._transcript_integrity import archive_reference

    config = _post_turn_config()
    logger = SessionLogger(session_config, "t1b-cancelled")
    messages = _conversation()
    boundary = _boundary_message()
    messages.append(boundary)
    stats = AgentStats()
    await logger.save_interaction(messages, stats, config, mock_tool_manager)
    box = _PendingTrimBox()
    manager = _build_real_post_turn_manager(
        logger=logger,
        messages=messages,
        stats=stats,
        tool_manager=mock_tool_manager,
        box=box,
    )
    fence_calls = 0
    real_fence = SessionLogger.fence_and_publish_archive_trim

    def _fence_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        nonlocal fence_calls
        fence_calls += 1
        return real_fence(self, *args, **kwargs)

    plan_entered = threading.Event()
    plan_release = threading.Event()
    real_plan = SessionLogger._plan_archive_seal

    def _gated_plan(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        plan_entered.set()
        plan_release.wait(timeout=30)
        return real_plan(self, *args, **kwargs)

    monkeypatch.setattr(SessionLogger, "fence_and_publish_archive_trim", _fence_spy)
    monkeypatch.setattr(SessionLogger, "_plan_archive_seal", _gated_plan)

    entry_task = asyncio.create_task(
        manager.capture_post_turn_pending_trim(
            session_id="t1b-cancelled", message_epoch=messages.message_epoch
        )
    )
    try:
        deadline = asyncio.get_running_loop().time() + 10
        while not plan_entered.is_set():
            assert asyncio.get_running_loop().time() < deadline, (
                "the completion never reached the sealing plan"
            )
            await asyncio.sleep(0.005)
        entry_task.cancel()
        try:
            await entry_task
        except asyncio.CancelledError:
            # The runtime may deliver the cancellation to the awaiter even
            # though the entry absorbed it internally; either way the entry has
            # finished and nothing was published.
            pass
    finally:
        plan_release.set()
    # Nothing partially published: no fence, no descriptor, no markers,
    # holder consumed, originals on disk.
    assert fence_calls == 0
    assert messages.archive_descriptor is None
    assert logger.archive_descriptor is None
    assert all(archive_reference(message) is None for message in messages)
    assert box.value is None
    assert logger._trim_in_flight_boundaries == set()
    raw = _messages_path(logger).read_bytes()
    assert b'"context_boundary": "compaction"' in raw
    # The turn is still complete: a later seam (plan restored) publishes.
    monkeypatch.setattr(SessionLogger, "_plan_archive_seal", real_plan)
    await manager.capture_post_turn_pending_trim(
        session_id="t1b-cancelled", message_epoch=messages.message_epoch
    )
    assert messages.archive_descriptor is not None
    assert fence_calls == 1


@pytest.mark.asyncio
async def test_post_turn_seal_succeeds_but_publish_fails_keeps_originals(
    session_config: Any, mock_tool_manager: ToolManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Seal-succeeds-but-cannot-publish arm: a first boundary is published
    # (pre-seal state = descriptor D1); a second boundary's seal SUCCEEDS but
    # the publication fails — the pre-seal archived bytes and descriptor D1
    # remain effective, nothing is partially published, and the outcome is
    # reported explicitly as failed_publish with the receipt.
    from chartreux.core.agent_loop._transcript_integrity import archive_reference

    config = _post_turn_config()
    logger = SessionLogger(session_config, "t1b-cannot-publish")
    messages = _conversation()
    first_boundary = _boundary_message("first boundary")
    messages.append(first_boundary)
    stats = AgentStats()
    await logger.save_interaction(messages, stats, config, mock_tool_manager)
    box = _PendingTrimBox()
    manager = _build_real_post_turn_manager(
        logger=logger,
        messages=messages,
        stats=stats,
        tool_manager=mock_tool_manager,
        box=box,
    )
    outcomes = _outcome_spy(monkeypatch)

    await manager.capture_post_turn_pending_trim(
        session_id="t1b-cannot-publish", message_epoch=messages.message_epoch
    )
    d1 = messages.archive_descriptor
    assert d1 is not None and logger.archive_descriptor is d1

    # A newer settled round plus a second boundary.
    messages.append(LLMMessage(role=Role.user, content="third ask"))
    messages.append(LLMMessage(role=Role.assistant, content="third answer"))
    second_boundary = _boundary_message("second boundary")
    messages.append(second_boundary)
    await logger.save_interaction(messages, stats, config, mock_tool_manager)

    results: list[Any] = []
    real_complete = SessionLogger.complete_post_turn_trim

    async def _capturing_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        result = await real_complete(self, *args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(SessionLogger, "complete_post_turn_trim", _capturing_spy)

    def _cannot_publish(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("simulated cannot-publish (stale fence)")

    monkeypatch.setattr(
        SessionLogger, "fence_and_publish_archive_trim", _cannot_publish
    )

    await manager.capture_post_turn_pending_trim(
        session_id="t1b-cannot-publish", message_epoch=messages.message_epoch
    )

    assert outcomes == ["published", "failed_publish"]
    assert results[-1].outcome == "failed_publish"
    assert results[-1].receipt is not None  # the seal receipt is reported
    # The pre-seal descriptor remains effective on both owners.
    assert logger.archive_descriptor is d1
    assert messages.archive_descriptor is d1
    # The original archived bytes on disk still verify against D1.
    verify_archived_prefix_on_disk(d1, _messages_path(logger))
    # Nothing partially published: the newly eligible region keeps its
    # originals (no markers beyond D1's sealed slots).
    non_system = _non_system(messages)
    assert all(
        archive_reference(message) is d1 for message in non_system[: d1.archived_count]
    )
    assert all(
        archive_reference(message) is None
        for message in non_system[d1.archived_count :]
    )
    assert box.value is None


@pytest.mark.asyncio
async def test_post_turn_completion_refuses_stale_epoch_and_session_keys(
    session_config: Any, mock_tool_manager: ToolManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The epoch/fence check in the completion path: a pending keyed to a stale
    # message epoch (after a rebind) or another session is refused AT THE
    # ATTEMPT BOUNDARY — never sealed, never published.
    config = _post_turn_config()
    logger = SessionLogger(session_config, "t1b-refused")
    messages = _conversation()
    boundary = _boundary_message()
    boundary_id = boundary.message_id
    assert boundary_id is not None
    messages.append(boundary)
    stats = AgentStats()
    await logger.save_interaction(messages, stats, config, mock_tool_manager)
    seal_calls = 0
    real_seal = SessionLogger.seal_archive

    async def _seal_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        nonlocal seal_calls
        seal_calls += 1
        return await real_seal(self, *args, **kwargs)

    monkeypatch.setattr(SessionLogger, "seal_archive", _seal_spy)

    stale_epoch = await logger.complete_post_turn_trim(
        messages,
        stats,
        config,
        mock_tool_manager,
        session_id="t1b-refused",
        message_epoch=messages.message_epoch + 1,
        boundary_message_id=boundary_id,
    )
    assert stale_epoch.outcome == "refused_epoch_mismatch"

    wrong_session = await logger.complete_post_turn_trim(
        messages,
        stats,
        config,
        mock_tool_manager,
        session_id="another-session",
        message_epoch=messages.message_epoch,
        boundary_message_id=boundary_id,
    )
    assert wrong_session.outcome == "refused_session_mismatch"

    wrong_boundary = await logger.complete_post_turn_trim(
        messages,
        stats,
        config,
        mock_tool_manager,
        session_id="t1b-refused",
        message_epoch=messages.message_epoch,
        boundary_message_id="no-such-boundary",
    )
    assert wrong_boundary.outcome == "refused_boundary_mismatch"

    # No seal ran for any refused key; nothing was published.
    assert seal_calls == 0
    assert messages.archive_descriptor is None
    assert logger.archive_descriptor is None
    assert logger._trim_in_flight_boundaries == set()
    assert logger._trim_published_boundaries == set()
