from __future__ import annotations

import json
from pathlib import Path

from chartreux.app_server._agent_transcript import (
    read_agent_transcript,
    read_live_agent_transcript,
)
from chartreux.app_server.protocol import AgentTranscriptState
from chartreux.core.llm_models import LLMMessage


def _write_session(path: Path, messages: list[dict[str, object]]) -> None:
    (path / "meta.json").write_text(json.dumps({"total_messages": len(messages)}))
    (path / "messages.jsonl").write_text(
        "".join(json.dumps(message) + "\n" for message in messages)
    )


def _messages(count: int) -> list[dict[str, object]]:
    return [
        {"role": "user", "message_id": f"message-{index}", "content": str(index)}
        for index in range(count)
    ]


def test_live_projection_matches_saved_view_with_system_and_tool_messages(
    tmp_path: Path,
) -> None:
    messages = [
        LLMMessage.model_validate({"role": "system", "content": "instructions"}),
        LLMMessage.model_validate({
            "role": "user",
            "message_id": "user-1",
            "content": "question",
        }),
        LLMMessage.model_validate({
            "role": "assistant",
            "message_id": "assistant-1",
            "content": "working",
        }),
        LLMMessage.model_validate({
            "role": "tool",
            "tool_call_id": "call-1",
            "tool_result": {"output": {"answer": "done"}},
        }),
    ]
    _write_session(
        tmp_path,
        [
            message.model_dump(exclude_none=True, mode="json")
            for message in messages
            if message.role.value != "system"
        ],
    )

    saved = read_agent_transcript(tmp_path, lambda: True)
    live = read_live_agent_transcript(messages)

    assert live == saved

    _write_session(tmp_path, _messages(205))

    pages = []
    before = None
    while True:
        page = read_agent_transcript(tmp_path, lambda: True, before=before, limit=200)
        assert page.state is AgentTranscriptState.AVAILABLE
        pages.append(page)
        if not page.has_more:
            break
        before = page.oldest_cursor

    assert [
        entry.display_text for page in reversed(pages) for entry in page.entries or []
    ] == [str(index) for index in range(205)]


def test_cursor_detects_rewritten_boundary_and_allows_append(tmp_path: Path) -> None:
    messages = _messages(3)
    _write_session(tmp_path, messages)
    latest = read_agent_transcript(tmp_path, lambda: True, limit=2)
    assert latest.oldest_cursor is not None

    messages.append({"role": "user", "message_id": "message-3", "content": "3"})
    _write_session(tmp_path, messages)
    older = read_agent_transcript(
        tmp_path, lambda: True, before=latest.oldest_cursor, limit=2
    )
    assert [entry.display_text for entry in older.entries or []] == ["0"]

    messages[1]["content"] = "rewritten"
    _write_session(tmp_path, messages)
    changed = read_agent_transcript(
        tmp_path, lambda: True, before=latest.oldest_cursor, limit=2
    )
    assert changed.state is AgentTranscriptState.CHANGED


def test_valid_empty_and_display_truncation(tmp_path: Path) -> None:
    _write_session(tmp_path, [])
    empty = read_agent_transcript(tmp_path, lambda: True)
    assert empty.state is AgentTranscriptState.AVAILABLE
    assert empty.entries == []

    large = "é" * (8 * 1024 + 1)
    _write_session(
        tmp_path,
        [{"role": "tool", "tool_call_id": "call", "tool_result": {"output": large}}],
    )
    page = read_agent_transcript(tmp_path, lambda: True)
    assert page.entries is not None
    assert page.entries[0].truncated


def test_display_limit_and_page_budget_are_wire_bytes(tmp_path: Path) -> None:
    exact = "é" * (8 * 1024 // 2)
    _write_session(tmp_path, [{"role": "user", "content": exact}])
    page = read_agent_transcript(tmp_path, lambda: True)
    assert page.entries is not None
    assert page.entries[0].display_text.encode() == exact.encode()
    assert not page.entries[0].truncated

    overflow = exact + "é"
    messages: list[dict[str, object]] = [
        {"role": "user", "message_id": f"m-{index}", "content": overflow}
        for index in range(200)
    ]
    _write_session(tmp_path, messages)
    page = read_agent_transcript(tmp_path, lambda: True, limit=200)
    assert page.entries
    assert all(len(entry.display_text.encode()) <= 8 * 1024 for entry in page.entries)
    wire = json.dumps(
        page.model_dump(mode="json", by_alias=True), separators=(",", ":")
    ).encode()
    assert len(wire) <= 512 * 1024


def test_missing_or_malformed_saved_files_do_not_crash(tmp_path: Path) -> None:
    assert read_agent_transcript(tmp_path, lambda: True).state is (
        AgentTranscriptState.NO_SAVED_TRANSCRIPT
    )
    (tmp_path / "meta.json").write_text("{")
    (tmp_path / "messages.jsonl").write_text('{"role": "user", "content": "x"}\n')
    assert read_agent_transcript(tmp_path, lambda: True).state is (
        AgentTranscriptState.NO_SAVED_TRANSCRIPT
    )
