from __future__ import annotations

import json
from pathlib import Path

from chartreux.app_server._agent_transcript import (
    _RESPONSE_BYTE_LIMIT,
    read_agent_transcript,
    read_live_agent_transcript,
)
from chartreux.app_server.protocol import AgentTranscriptEntryKind, AgentTranscriptState
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
    assert len(wire) <= _RESPONSE_BYTE_LIMIT


def test_reasoning_entries_expose_stable_per_entry_digests() -> None:
    message = LLMMessage.model_validate({
        "role": "assistant",
        "message_id": "assistant-1",
        "reasoning_message_id": "reasoning-1",
        "reasoning_content": "checking the result",
        "content": "done",
    })

    first = read_live_agent_transcript([message])
    repeated = read_live_agent_transcript([message])
    assert first.entries is not None and repeated.entries is not None
    reasoning = next(
        entry
        for entry in first.entries
        if entry.kind is AgentTranscriptEntryKind.REASONING
    )
    repeated_reasoning = next(
        entry
        for entry in repeated.entries
        if entry.kind is AgentTranscriptEntryKind.REASONING
    )
    assert reasoning.display_text == "checking the result"
    assert reasoning.digest == repeated_reasoning.digest
    assert reasoning.created_at == reasoning.updated_at
    assert reasoning.generation_status.value == "completed"

    changed = message.model_copy(update={"reasoning_content": "new reasoning"})
    changed_page = read_live_agent_transcript([changed])
    assert changed_page.entries is not None
    changed_reasoning = next(
        entry
        for entry in changed_page.entries
        if entry.kind is AgentTranscriptEntryKind.REASONING
    )
    assert changed_reasoning.entry_id == reasoning.entry_id
    assert changed_reasoning.digest != reasoning.digest

    changed_sibling = message.model_copy(update={"content": "a different answer"})
    sibling_page = read_live_agent_transcript([changed_sibling])
    assert sibling_page.entries is not None
    sibling_text = next(
        entry
        for entry in sibling_page.entries
        if entry.kind is AgentTranscriptEntryKind.ASSISTANT_TEXT
    )
    assert (
        sibling_text.digest
        != next(
            entry
            for entry in first.entries
            if entry.kind is AgentTranscriptEntryKind.ASSISTANT_TEXT
        ).digest
    )
    assert (
        next(
            entry
            for entry in sibling_page.entries
            if entry.kind is AgentTranscriptEntryKind.REASONING
        ).digest
        == reasoning.digest
    )

    changed_reasoning_only = message.model_copy(
        update={"reasoning_content": "new reasoning"}
    )
    reasoning_page = read_live_agent_transcript([changed_reasoning_only])
    assert reasoning_page.entries is not None
    assert (
        next(
            entry
            for entry in reasoning_page.entries
            if entry.kind is AgentTranscriptEntryKind.ASSISTANT_TEXT
        ).digest
        == next(
            entry
            for entry in first.entries
            if entry.kind is AgentTranscriptEntryKind.ASSISTANT_TEXT
        ).digest
    )


def test_tool_entries_include_presentations_effect_payloads_and_fallbacks() -> None:
    assistant = LLMMessage.model_validate({
        "role": "assistant",
        "message_id": "assistant-1",
        "tool_calls": [
            {
                "id": "call-1",
                "function": {"name": "shell", "arguments": '{"command":"echo hi"}'},
                "presentation": {
                    "kind": "shell",
                    "display": {
                        "summary": "echo hi",
                        "content": "echo hi",
                        "verb": "Running",
                        "message": "echo hi",
                        "settled_verb": "Ran",
                        "settled_message": "echo hi",
                        "status_text": "Running shell",
                    },
                },
            },
            {
                "id": "call-2",
                "function": {"name": "custom_tool", "arguments": '{"value":1}'},
            },
        ],
    })
    shell_output = {
        "stdout": "hi\\n",
        "stderr": "",
        "output": "hi\\n",
        "truncated": False,
    }
    shell_result = LLMMessage.model_validate({
        "role": "tool",
        "name": "shell",
        "tool_call_id": "call-1",
        "content": "hi\\n",
        "tool_result": {
            "output": shell_output,
            "duration": 0.25,
            "presentation": {
                "kind": "shell",
                "display": {"success": True, "verb": "Ran", "message": "echo hi"},
                "projected_output": shell_output,
            },
        },
    })
    fallback_result = LLMMessage.model_validate({
        "role": "tool",
        "name": "custom_tool",
        "tool_call_id": "call-2",
        "content": "finished",
        "tool_result": {"output": {"answer": "ok"}},
    })

    response = read_live_agent_transcript([assistant, shell_result, fallback_result])
    assert response.entries is not None
    shell_call = next(
        entry
        for entry in response.entries
        if entry.kind is AgentTranscriptEntryKind.TOOL_CALL
        and entry.tool_call_id == "call-1"
    )
    shell_entry = next(
        entry
        for entry in response.entries
        if entry.kind is AgentTranscriptEntryKind.TOOL_RESULT
        and entry.tool_call_id == "call-1"
    )
    assert shell_call.arguments == {"command": "echo hi"}
    assert shell_call.call_presentation is not None
    assert shell_call.call_presentation.display.summary == "echo hi"
    assert shell_call.detail is not None
    assert shell_call.detail.kind.value == "shell"
    assert shell_call.state is not None and shell_call.state.status == "completed"
    assert shell_entry.result == shell_output
    assert shell_entry.result_presentation is not None
    assert shell_entry.result_presentation.display.message == "echo hi"

    changed_result = shell_result.model_copy(update={"content": "updated"})
    changed_page = read_live_agent_transcript([
        assistant,
        changed_result,
        fallback_result,
    ])
    assert changed_page.entries is not None
    changed_call = next(
        entry
        for entry in changed_page.entries
        if entry.kind is AgentTranscriptEntryKind.TOOL_CALL
        and entry.tool_call_id == "call-1"
    )
    assert changed_call.entry_id == shell_call.entry_id
    assert changed_call.digest != shell_call.digest

    changed_call_message = LLMMessage.model_validate({
        "role": "assistant",
        "message_id": "assistant-1",
        "tool_calls": [
            {
                "id": "call-1",
                "function": {"name": "shell", "arguments": '{"command":"echo bye"}'},
                "presentation": {
                    "kind": "shell",
                    "display": {
                        "summary": "echo hi",
                        "content": "echo hi",
                        "verb": "Running",
                        "message": "echo hi",
                        "settled_verb": "Ran",
                        "settled_message": "echo hi",
                        "status_text": "Running shell",
                    },
                },
            }
        ],
    })
    changed_call_page = read_live_agent_transcript([
        changed_call_message,
        shell_result,
        fallback_result,
    ])
    assert changed_call_page.entries is not None
    changed_call_entry = next(
        entry
        for entry in changed_call_page.entries
        if entry.kind is AgentTranscriptEntryKind.TOOL_CALL
        and entry.tool_call_id == "call-1"
    )
    changed_result_entry = next(
        entry
        for entry in changed_call_page.entries
        if entry.kind is AgentTranscriptEntryKind.TOOL_RESULT
        and entry.tool_call_id == "call-1"
    )
    assert changed_call_entry.digest != shell_call.digest
    assert changed_result_entry.digest != shell_entry.digest

    fallback_call = next(
        entry
        for entry in response.entries
        if entry.kind is AgentTranscriptEntryKind.TOOL_CALL
        and entry.tool_call_id == "call-2"
    )
    fallback_entry = next(
        entry
        for entry in response.entries
        if entry.kind is AgentTranscriptEntryKind.TOOL_RESULT
        and entry.tool_call_id == "call-2"
    )
    assert fallback_call.call_presentation is not None
    assert fallback_call.call_presentation.kind.value == "tool"
    assert fallback_call.call_presentation.display.summary == "custom_tool(value=1)"
    assert fallback_call.detail is not None
    assert fallback_call.detail.kind.value == "tool"
    assert fallback_entry.result == {"answer": "ok"}
    assert fallback_entry.result_presentation is not None
    assert fallback_entry.result_presentation.display.message == "finished"
    assert fallback_entry.status is not None
    assert fallback_entry.status.value == "completed"


def test_user_attachment_projection_contains_names_not_image_data() -> None:
    message = LLMMessage.model_validate({
        "role": "user",
        "message_id": "user-1",
        "content": "What is in this picture?",
        "images": [
            {
                "source": {"kind": "inline", "data": "c2VjcmV0LWltYWdlLWRhdGE="},
                "alias": "/private/secret/photo.png",
                "mime_type": "image/png",
            }
        ],
    })

    page = read_live_agent_transcript([message])
    assert page.entries is not None
    user_entry = page.entries[0]
    assert user_entry.kind is AgentTranscriptEntryKind.USER_TEXT
    assert user_entry.attachment_names == ["photo.png"]
    assert user_entry.attachment_count == 1
    wire = json.dumps(page.model_dump(mode="json", by_alias=True))
    assert "photo.png" in wire
    assert "/private/secret" not in wire
    assert "c2VjcmV0LWltYWdlLWRhdGE=" not in wire
    assert "inline" not in wire


def test_rich_transcript_pages_within_response_byte_limit(tmp_path: Path) -> None:
    messages: list[dict[str, object]] = []
    for index in range(80):
        call_id = f"call-{index}"
        arguments = json.dumps({"value": "a" * 1200})
        messages.extend([
            {
                "role": "assistant",
                "message_id": f"assistant-{index}",
                "tool_calls": [
                    {
                        "id": call_id,
                        "function": {"name": "custom_tool", "arguments": arguments},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "custom_tool",
                "content": "done",
                "tool_result": {"output": {"value": "b" * 1200}},
            },
        ])
    _write_session(tmp_path, messages)

    pages = []
    before = None
    while True:
        page = read_agent_transcript(tmp_path, lambda: True, before=before, limit=200)
        assert page.state is AgentTranscriptState.AVAILABLE
        assert page.entries is not None
        wire = json.dumps(
            page.model_dump(mode="json", by_alias=True), separators=(",", ":")
        ).encode()
        assert len(wire) <= _RESPONSE_BYTE_LIMIT
        assert page.entries
        pages.append(page)
        if not page.has_more:
            break
        assert page.oldest_cursor is not None
        before = page.oldest_cursor

    all_entries = [entry for page in reversed(pages) for entry in page.entries or []]
    assert len(pages) > 1
    assert len(all_entries) == 160
    assert (
        sum(entry.kind is AgentTranscriptEntryKind.TOOL_CALL for entry in all_entries)
        == 80
    )
    assert (
        sum(entry.kind is AgentTranscriptEntryKind.TOOL_RESULT for entry in all_entries)
        == 80
    )


def test_oversized_single_result_stays_bounded_and_pages_without_empty_trap(
    tmp_path: Path,
) -> None:
    oversized = "x" * (2 * 1024 * 1024)
    _write_session(
        tmp_path,
        [
            {
                "role": "assistant",
                "message_id": "assistant-1",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "custom_tool",
                            "arguments": '{"path":"src/example.py"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "custom_tool",
                "content": "completed",
                "tool_result": {
                    "output": {"content": oversized},
                    "presentation": {
                        "kind": "tool",
                        "display": {
                            "success": True,
                            "message": "completed",
                            "warnings": [oversized],
                        },
                        "projected_output": {"content": oversized},
                    },
                },
            },
        ],
    )

    newest = read_agent_transcript(tmp_path, lambda: True, limit=1)
    assert newest.state is AgentTranscriptState.AVAILABLE
    assert newest.entries is not None and len(newest.entries) == 1
    assert newest.entries[0].kind is AgentTranscriptEntryKind.TOOL_RESULT
    wire = json.dumps(
        newest.model_dump(mode="json", by_alias=True), separators=(",", ":")
    ).encode()
    assert len(wire) <= _RESPONSE_BYTE_LIMIT
    assert newest.has_more
    assert newest.oldest_cursor is not None

    older = read_agent_transcript(
        tmp_path, lambda: True, before=newest.oldest_cursor, limit=1
    )
    assert older.state is AgentTranscriptState.AVAILABLE
    assert older.entries is not None and len(older.entries) == 1
    assert older.entries[0].kind is AgentTranscriptEntryKind.TOOL_CALL
    assert not older.has_more


def test_missing_or_malformed_saved_files_do_not_crash(tmp_path: Path) -> None:
    assert read_agent_transcript(tmp_path, lambda: True).state is (
        AgentTranscriptState.NO_SAVED_TRANSCRIPT
    )
    (tmp_path / "meta.json").write_text("{")
    (tmp_path / "messages.jsonl").write_text('{"role": "user", "content": "x"}\n')
    assert read_agent_transcript(tmp_path, lambda: True).state is (
        AgentTranscriptState.NO_SAVED_TRANSCRIPT
    )
