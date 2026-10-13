from __future__ import annotations

from unittest.mock import Mock

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
from chartreux.core.session_types import AgentStats
from chartreux.core.usage import UsagePurpose
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from tests.mock.utils import mock_llm_chunk
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
