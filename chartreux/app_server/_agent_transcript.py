from __future__ import annotations

import base64
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
from typing import Any

from chartreux.app_server.protocol import (
    MAX_AGENT_TRANSCRIPT_ID_LENGTH,
    AgentTranscriptEntry,
    AgentTranscriptEntryKind,
    AgentTranscriptGetResponse,
    AgentTranscriptState,
    AgentTranscriptTruncation,
)
from chartreux.core.session.session_loader import (
    BoundedSessionLoadState,
    SessionFileContainmentError,
    SessionLoader,
)

__all__ = ["read_agent_transcript"]

_INPUT_BYTE_LIMIT = 16 * 1024 * 1024
_RESPONSE_BYTE_LIMIT = 512 * 1024
_DISPLAY_TEXT_LIMIT = 8 * 1024
_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200


def read_agent_transcript(  # noqa: PLR0911
    session_dir: Path,
    validate: Callable[[], bool],
    *,
    before: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    session_dir_fd: int | None = None,
) -> AgentTranscriptGetResponse:
    """Read one disk-backed, chronological transcript page without side effects.

    ``session_dir`` has already been authorized by the caller.  ``validate`` is
    intentionally supplied by that authorization layer so it can reject a
    snapshot that was revoked before this worker started reading.
    """
    if not 1 <= limit <= _MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}")
    try:
        if not validate():
            return _no_saved_transcript()
        loaded = SessionLoader.load_session_bounded(
            session_dir, byte_limit=_INPUT_BYTE_LIMIT, dir_fd=session_dir_fd
        )
        if loaded.state is BoundedSessionLoadState.EXCEEDS_LIMIT:
            return _state(AgentTranscriptState.EXCEEDS_VIEWER_LIMIT)
        if loaded.state is not BoundedSessionLoadState.AVAILABLE:
            return _no_saved_transcript()
        if not validate():
            return _no_saved_transcript()
    except SessionFileContainmentError:
        raise
    except (OSError, ValueError):
        return _no_saved_transcript()

    entries = _project_entries(loaded.messages or [])
    boundary_index = len(entries)
    if before is not None:
        boundary = _decode_cursor(before)
        if boundary is None:
            return _state(AgentTranscriptState.CHANGED)
        boundary_index = _find_boundary(entries, boundary)
        if boundary_index is None:
            return _state(AgentTranscriptState.CHANGED)

    return _page(entries[:boundary_index], limit)


def _no_saved_transcript() -> AgentTranscriptGetResponse:
    return _state(AgentTranscriptState.NO_SAVED_TRANSCRIPT)


def _state(state: AgentTranscriptState) -> AgentTranscriptGetResponse:
    return AgentTranscriptGetResponse(state=state)


def _project_entries(
    messages: list[dict[str, Any]],
) -> list[tuple[AgentTranscriptEntry, str]]:
    entries: list[tuple[AgentTranscriptEntry, str]] = []
    for message_index, message in enumerate(messages):
        role = message.get("role")
        message_id = message.get("message_id")
        identity = message_id if isinstance(message_id, str) and message_id else None
        if role in {"user", "assistant"} and "content" in message:
            text = _text_content(message["content"])
            if text is not None:
                entries.append(
                    _entry(
                        _entry_id(identity, message_index, "text"),
                        AgentTranscriptEntryKind.USER_TEXT
                        if role == "user"
                        else AgentTranscriptEntryKind.ASSISTANT_TEXT,
                        text,
                        message,
                        "text",
                    )
                )
        if role == "assistant":
            for call_index, call in enumerate(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                function = function if isinstance(function, dict) else {}
                call_id = call.get("id")
                call_identity = (
                    call_id if isinstance(call_id, str) and call_id else None
                )
                name = function.get("name")
                tool_name = name if isinstance(name, str) and name else "unknown"
                arguments = function.get("arguments")
                display = arguments if isinstance(arguments, str) else ""
                entries.append(
                    _entry(
                        _entry_id(identity, message_index, f"call:{call_index}"),
                        AgentTranscriptEntryKind.TOOL_CALL,
                        display,
                        message,
                        f"call:{call_index}",
                        tool_name=_bounded_identity(tool_name),
                        tool_call_id=_bounded_identity(call_identity)
                        if call_identity
                        else None,
                    )
                )
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                tool_call_id = f"tool:{message_index}"
            result = message.get("tool_result")
            if isinstance(result, dict) and "output" in result:
                display = _json_text(result["output"])
            else:
                display = _text_content(message.get("content")) or ""
            entries.append(
                _entry(
                    _entry_id(identity, message_index, "result"),
                    AgentTranscriptEntryKind.TOOL_RESULT,
                    display,
                    message,
                    "result",
                    tool_call_id=_bounded_identity(tool_call_id),
                )
            )
    return entries


def _entry(
    entry_id: str,
    kind: AgentTranscriptEntryKind,
    display_text: str,
    message: dict[str, Any],
    component: str,
    *,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
) -> tuple[AgentTranscriptEntry, str]:
    bounded_text, truncated = _truncate_utf8(display_text, _DISPLAY_TEXT_LIMIT)
    entry = AgentTranscriptEntry(
        entry_id=entry_id,
        kind=kind,
        display_text=bounded_text,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        truncated=truncated,
        truncation=AgentTranscriptTruncation.DISPLAY_TEXT_LIMIT if truncated else None,
    )
    # Hash the untruncated source component: changing text beyond the display
    # limit must still invalidate an older-page boundary.
    canonical = _json_text({"message": message, "component": component})
    return entry, hashlib.sha256(canonical.encode()).hexdigest()


def _truncate_utf8(text: str, byte_limit: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text, False
    return encoded[:byte_limit].decode("utf-8", errors="ignore"), True


def _text_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return None


def _json_text(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _entry_id(message_id: str | None, index: int, component: str) -> str:
    return _bounded_identity(f"{message_id or f'message:{index}'}:{component}")


def _bounded_identity(value: str) -> str:
    if len(value) <= MAX_AGENT_TRANSCRIPT_ID_LENGTH:
        return value
    digest = hashlib.sha256(value.encode()).hexdigest()
    return value[: MAX_AGENT_TRANSCRIPT_ID_LENGTH - len(digest) - 1] + ":" + digest


def _encode_cursor(entry_id: str, digest: str) -> str:
    payload = _json_text({"id": entry_id, "digest": digest}).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[str, str] | None:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded.encode()))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or set(value) != {"id", "digest"}:
        return None
    entry_id, digest = value["id"], value["digest"]
    if not isinstance(entry_id, str) or not isinstance(digest, str):
        return None
    return entry_id, digest


def _find_boundary(
    entries: list[tuple[AgentTranscriptEntry, str]], boundary: tuple[str, str]
) -> int | None:
    entry_id, digest = boundary
    for index, (entry, current_digest) in enumerate(entries):
        if entry.entry_id == entry_id:
            return index if current_digest == digest else None
    return None


def _page(
    candidates: list[tuple[AgentTranscriptEntry, str]], limit: int
) -> AgentTranscriptGetResponse:
    selected: list[tuple[AgentTranscriptEntry, str]] = []
    for candidate in reversed(candidates):
        if len(selected) == limit:
            break
        prospective = [candidate, *selected]
        response = _available_response(prospective, len(candidates) > len(prospective))
        if _wire_size(response) > _RESPONSE_BYTE_LIMIT:
            break
        selected = prospective
    # A single display-bounded entry always fits the fixed response ceiling.
    return _available_response(selected, len(candidates) > len(selected))


def _wire_size(response: AgentTranscriptGetResponse) -> int:
    payload = response.model_dump(mode="json", by_alias=True)
    return len(json.dumps(payload, separators=(",", ":")).encode())


def _available_response(
    selected: list[tuple[AgentTranscriptEntry, str]], has_more: bool
) -> AgentTranscriptGetResponse:
    cursor = (
        _encode_cursor(selected[0][0].entry_id, selected[0][1]) if selected else None
    )
    return AgentTranscriptGetResponse(
        state=AgentTranscriptState.AVAILABLE,
        entries=[entry for entry, _digest in selected],
        oldest_cursor=cursor,
        has_more=has_more,
    )
