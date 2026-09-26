from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, ClassVar

from chartreux.app_server._agent_transcript import (
    _encode_cursor,
    _project_entries,
    _read_projected_entries,
    read_agent_transcript,
    read_live_agent_transcript,
)
from chartreux.app_server.protocol import (
    MAX_AGENT_TRANSCRIPT_ID_LENGTH,
    AgentTranscriptState,
)
from chartreux.core.llm_models import LLMMessage, Role


def _write_session(path: Path, messages: list[dict[str, object]]) -> None:
    (path / "meta.json").write_text(json.dumps({"total_messages": len(messages)}))
    (path / "messages.jsonl").write_text(
        "".join(json.dumps(message) + "\n" for message in messages)
    )


def _dumped(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    return [
        message.model_dump(exclude_none=True, mode="json")
        for message in messages
        if message.role is not Role.system
    ]


def _oracle(dumped: list[dict[str, Any]], before: str | None, limit: int):
    return _read_projected_entries(_project_entries(dumped), before, limit)


def _random_transcript(rng: random.Random, turns: int) -> list[LLMMessage]:
    messages: list[LLMMessage] = []
    call_counter = 0
    for turn in range(turns):
        if rng.random() < 0.15:
            messages.append(LLMMessage(role=Role.system, content=f"system {turn}"))
        if rng.random() < 0.25:
            user_kwargs: dict[str, Any] = {
                "role": "user",
                "content": rng.choice([
                    None,
                    "",
                    f"question {turn}",
                    f"note {turn} " * 20,
                ]),
            }
            if rng.random() < 0.7:
                user_kwargs["message_id"] = f"user-{turn}"
            if rng.random() < 0.2:
                user_kwargs["images"] = [
                    {
                        "source": {"kind": "inline", "data": "ZA=="},
                        "alias": f"/tmp/img-{turn}.png",
                        "mime_type": "image/png",
                    }
                ]
            messages.append(LLMMessage.model_validate(user_kwargs))

        assistant_kwargs: dict[str, Any] = {"role": "assistant"}
        if rng.random() < 0.7:
            assistant_kwargs["message_id"] = f"assistant-{turn}"
        if rng.random() < 0.4:
            assistant_kwargs["reasoning_content"] = f"thinking {turn}"
            if rng.random() < 0.5:
                assistant_kwargs["reasoning_message_id"] = f"reasoning-{turn}"
        if rng.random() < 0.8:
            assistant_kwargs["content"] = f"answer {turn}"
        calls: list[dict[str, Any]] = []
        for _ in range(rng.choice([0, 0, 1, 2, 4])):
            call_counter += 1
            call: dict[str, Any] = {
                "function": {
                    "name": rng.choice(["bash", "read_file", "custom_tool"]),
                    "arguments": json.dumps({"value": call_counter}),
                }
            }
            if rng.random() < 0.85:
                call["id"] = f"call-{call_counter}"
            if rng.random() < 0.3:
                call["presentation"] = {
                    "kind": "shell",
                    "display": {
                        "summary": f"step {call_counter}",
                        "verb": "Running",
                        "message": f"step {call_counter}",
                        "settled_verb": "Ran",
                        "settled_message": f"step {call_counter}",
                        "status_text": "Running shell",
                    },
                }
            calls.append(call)
        if calls:
            assistant_kwargs["tool_calls"] = calls
        messages.append(LLMMessage.model_validate(assistant_kwargs))

        for call in calls:
            call_id = call.get("id")
            if call_id is None:
                continue
            if rng.random() < 0.85:
                tool_kwargs: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": call["function"]["name"],
                    "content": f"output {call_id}",
                }
                if rng.random() < 0.5:
                    tool_kwargs["tool_result"] = {
                        "output": {"value": call_id},
                        "duration": 0.1,
                    }
                messages.append(LLMMessage.model_validate(tool_kwargs))
        if rng.random() < 0.1:
            messages.append(
                LLMMessage.model_validate({
                    "role": "tool",
                    "tool_call_id": f"orphan-{turn}",
                    "name": "bash",
                    "content": "orphan result",
                })
            )
        if rng.random() < 0.1:
            # A result that arrives before its call exercises last-occurrence-wins.
            messages.append(
                LLMMessage.model_validate({
                    "role": "tool",
                    "tool_call_id": f"call-{call_counter + 1}",
                    "name": "bash",
                    "content": "early result",
                })
            )
    return messages


def _assert_paging_equivalent(
    messages: list[LLMMessage], *, limit: int, tmp_path: Path | None = None
) -> None:
    dumped = _dumped(messages)
    if tmp_path is not None:
        _write_session(tmp_path, dumped)

    def read(before: str | None):
        if tmp_path is None:
            return read_live_agent_transcript(messages, before=before, limit=limit)
        return read_agent_transcript(tmp_path, lambda: True, before=before, limit=limit)

    before = None
    seen_cursors: set[str] = set()
    for _ in range(2000):
        response = read(before)
        assert response == _oracle(dumped, before, limit)
        if not response.has_more or response.oldest_cursor is None:
            break
        before = response.oldest_cursor
        assert before not in seen_cursors, "paging walk repeated a cursor"
        seen_cursors.add(before)
    else:
        raise AssertionError("paging walk did not terminate")

    bogus = _encode_cursor("missing:entry", "0" * 64)
    assert read(bogus).state is AgentTranscriptState.CHANGED
    assert _oracle(dumped, bogus, limit).state is AgentTranscriptState.CHANGED


def test_live_equivalence_on_random_transcripts() -> None:
    for seed in range(12):
        messages = _random_transcript(random.Random(seed), turns=12)
        for limit in (1, 3, 50, 200):
            _assert_paging_equivalent(messages, limit=limit)


def test_saved_equivalence_on_random_transcripts(tmp_path: Path) -> None:
    for seed in range(12, 24):
        messages = _random_transcript(random.Random(seed), turns=12)
        for limit in (1, 3, 50, 200):
            _assert_paging_equivalent(messages, limit=limit, tmp_path=tmp_path)


def test_boundary_cursor_at_every_entry_component(tmp_path: Path) -> None:
    messages = _random_transcript(random.Random(99), turns=12)
    dumped = _dumped(messages)
    _write_session(tmp_path, dumped)
    entries = _project_entries(dumped)
    assert len(entries) > 10

    for boundary in entries:
        cursor = _encode_cursor(boundary.entry_id, boundary.digest)
        live = read_live_agent_transcript(messages, before=cursor, limit=5)
        saved = read_agent_transcript(tmp_path, lambda: True, before=cursor, limit=5)
        oracle = _oracle(dumped, cursor, 5)
        assert live == oracle
        assert saved == oracle


def test_stale_boundary_digest_reports_changed(tmp_path: Path) -> None:
    messages = [
        LLMMessage.model_validate({
            "role": "user",
            "message_id": "user-1",
            "content": "first",
        }),
        LLMMessage.model_validate({
            "role": "assistant",
            "message_id": "assistant-1",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "bash", "arguments": "{}"}}
            ],
        }),
    ]
    dumped = _dumped(messages)
    _write_session(tmp_path, dumped)
    entries = _project_entries(dumped)
    call_entry = next(entry for entry in entries if entry.tool_call_id == "call-1")
    cursor = _encode_cursor(call_entry.entry_id, call_entry.digest)

    completed = [
        *messages,
        LLMMessage.model_validate({
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "bash",
            "content": "done",
        }),
    ]
    changed_live = read_live_agent_transcript(completed, before=cursor, limit=10)
    changed_saved_dumped = _dumped(completed)
    _write_session(tmp_path, changed_saved_dumped)
    changed_saved = read_agent_transcript(
        tmp_path, lambda: True, before=cursor, limit=10
    )
    oracle = _oracle(changed_saved_dumped, cursor, 10)
    assert changed_live.state is AgentTranscriptState.CHANGED
    assert changed_saved.state is AgentTranscriptState.CHANGED
    assert oracle.state is AgentTranscriptState.CHANGED


def test_duplicate_entry_ids_use_first_digest_match(tmp_path: Path) -> None:
    messages = [
        LLMMessage.model_validate({
            "role": "user",
            "message_id": "duplicated",
            "content": "first",
        }),
        LLMMessage.model_validate({
            "role": "user",
            "message_id": "duplicated",
            "content": "second",
        }),
    ]
    dumped = _dumped(messages)
    _write_session(tmp_path, dumped)
    entries = _project_entries(dumped)
    assert entries[0].entry_id == entries[1].entry_id
    assert entries[0].digest != entries[1].digest

    first_cursor = _encode_cursor(entries[0].entry_id, entries[0].digest)
    second_cursor = _encode_cursor(entries[1].entry_id, entries[1].digest)
    for cursor in (first_cursor, second_cursor):
        live = read_live_agent_transcript(messages, before=cursor, limit=10)
        saved = read_agent_transcript(tmp_path, lambda: True, before=cursor, limit=10)
        oracle = _oracle(dumped, cursor, 10)
        assert live == oracle
        assert saved == oracle

    rewritten = [
        LLMMessage.model_validate({
            "role": "user",
            "message_id": "duplicated",
            "content": "rewritten",
        }),
        messages[1],
    ]
    rewritten_dumped = _dumped(rewritten)
    _write_session(tmp_path, rewritten_dumped)
    live = read_live_agent_transcript(rewritten, before=first_cursor, limit=10)
    saved = read_agent_transcript(tmp_path, lambda: True, before=first_cursor, limit=10)
    oracle = _oracle(rewritten_dumped, first_cursor, 10)
    assert live.state is AgentTranscriptState.CHANGED
    assert saved.state is AgentTranscriptState.CHANGED
    assert oracle.state is AgentTranscriptState.CHANGED


def test_colon_and_oversized_identities_page_identically(tmp_path: Path) -> None:
    long_id = "weird:id:with:colons:" + "x" * (MAX_AGENT_TRANSCRIPT_ID_LENGTH + 50)
    messages = [
        LLMMessage.model_validate({
            "role": "user",
            "message_id": long_id,
            "content": "first",
        }),
        LLMMessage.model_validate({
            "role": "assistant",
            "message_id": "assistant-1",
            "reasoning_message_id": "reasoning:weird:id",
            "reasoning_content": "thinking",
            "content": "answer",
        }),
        LLMMessage.model_validate({
            "role": "user",
            "message_id": "message:0",
            "content": "fallback-looking id",
        }),
    ]
    _assert_paging_equivalent(messages, limit=2)
    _assert_paging_equivalent(messages, limit=50, tmp_path=tmp_path)


class _CountingMessage(LLMMessage):
    dump_count: ClassVar[int] = 0

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        _CountingMessage.dump_count += 1
        return super().model_dump(**kwargs)


def test_newest_page_projection_is_bounded_by_selection() -> None:
    messages: list[LLMMessage] = []
    for turn in range(1000):
        messages.append(
            _CountingMessage.model_validate({
                "role": "user",
                "message_id": f"user-{turn}",
                "content": f"question {turn}",
            })
        )
        messages.append(
            _CountingMessage.model_validate({
                "role": "assistant",
                "message_id": f"assistant-{turn}",
                "content": f"answer {turn}",
                "tool_calls": [
                    {
                        "id": f"call-{turn}",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            })
        )
        messages.append(
            _CountingMessage.model_validate({
                "role": "tool",
                "tool_call_id": f"call-{turn}",
                "name": "bash",
                "content": f"output {turn}",
            })
        )

    _CountingMessage.dump_count = 0
    response = read_live_agent_transcript(messages, before=None, limit=50)
    assert response.state is AgentTranscriptState.AVAILABLE
    assert response.entries is not None
    assert len(response.entries) == 50
    assert _CountingMessage.dump_count < 300
