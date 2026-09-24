from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue, ValidationError

from chartreux.app_server._tool_projection import (
    project_effect_detail,
    project_effect_output_value,
)
from chartreux.app_server.models import (
    CancelledEffectState,
    CompletedEffectState,
    EffectDetail,
    EffectResultDisplay,
    EffectState,
    FailedEffectState,
    PublicEntryGenerationStatus,
    PublicError,
    RunningEffectState,
)
from chartreux.app_server.protocol import (
    MAX_AGENT_TRANSCRIPT_ID_LENGTH,
    AgentTranscriptEntry,
    AgentTranscriptEntryKind,
    AgentTranscriptGetResponse,
    AgentTranscriptState,
    AgentTranscriptToolStatus,
    AgentTranscriptTruncation,
)
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_loader import (
    BoundedSessionLoadState,
    SessionFileContainmentError,
    SessionLoader,
)
from chartreux.core.utils import CANCELLATION_TAG, TOOL_ERROR_TAG, TaggedText
from chartreux.utils.tool_presentation import (
    EffectCallDisplay as ToolEffectCallDisplay,
    EffectResultDisplay as ToolEffectResultDisplay,
    ToolCallPresentation,
    ToolEffectKind,
    ToolResultPresentation,
)

__all__ = ["read_agent_transcript", "read_live_agent_transcript"]

_INPUT_BYTE_LIMIT = 16 * 1024 * 1024
_RESPONSE_BYTE_LIMIT = 512 * 1024
_DISPLAY_TEXT_LIMIT = 8 * 1024
_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200


def read_agent_transcript(
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

    return _read_projected_entries(
        _project_entries(loaded.messages or []), before, limit
    )


def read_live_agent_transcript(
    messages: list[LLMMessage],
    *,
    before: str | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> AgentTranscriptGetResponse:
    """Project an in-memory child transcript through the persisted-view rules."""
    if not 1 <= limit <= _MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}")
    persisted_messages = [
        message.model_dump(exclude_none=True, mode="json")
        for message in messages
        if message.role is not Role.system
    ]
    return _read_projected_entries(_project_entries(persisted_messages), before, limit)


def _read_projected_entries(
    entries: list[tuple[AgentTranscriptEntry, str]], before: str | None, limit: int
) -> AgentTranscriptGetResponse:
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


_TOOL_VALUE_BYTE_LIMIT = 16 * 1024
_PRESENTATION_TEXT_LIMIT = 4 * 1024
_MAX_PRESENTATION_WARNINGS = 8
_MAX_ATTACHMENT_NAMES = 32


@dataclass(frozen=True, slots=True)
class _ToolCallProjection:
    tool_name: str
    arguments: JsonValue
    presentation: ToolCallPresentation
    detail: EffectDetail


@dataclass(frozen=True, slots=True)
class _ToolResultProjection:
    state: EffectState
    status: AgentTranscriptToolStatus
    presentation: ToolResultPresentation | None
    result: dict[str, JsonValue] | None
    output_text: str | None


@dataclass(frozen=True, slots=True)
class _EntryPayload:
    title: str
    tool_name: str | None = None
    tool_call_id: str | None = None
    arguments: JsonValue = None
    result: dict[str, JsonValue] | None = None
    output_text: str | None = None
    status: AgentTranscriptToolStatus | None = None
    call_presentation: ToolCallPresentation | None = None
    result_presentation: ToolResultPresentation | None = None
    detail: EffectDetail | None = None
    state: EffectState | None = None
    attachment_names: list[str] | None = None
    attachment_count: int = 0


def _project_entries(
    messages: list[dict[str, Any]],
) -> list[tuple[AgentTranscriptEntry, str]]:
    calls_by_index, calls_by_id, results_by_id = _tool_message_indexes(messages)
    entries: list[tuple[AgentTranscriptEntry, str]] = []
    for message_index, message in enumerate(messages):
        match message.get("role"):
            case "user":
                entry = _project_user_entry(message, message_index, len(entries))
                if entry is not None:
                    entries.append(entry)
            case "assistant":
                entries.extend(
                    _project_assistant_entries(
                        message,
                        message_index,
                        len(entries),
                        calls_by_index,
                        results_by_id,
                    )
                )
            case "tool":
                entries.append(
                    _project_tool_result_entry(
                        message, message_index, len(entries), calls_by_id
                    )
                )
    return entries


def _tool_message_indexes(
    messages: list[dict[str, Any]],
) -> tuple[
    dict[tuple[int, int], _ToolCallProjection],
    dict[str, _ToolCallProjection],
    dict[str, dict[str, Any]],
]:
    calls_by_index: dict[tuple[int, int], _ToolCallProjection] = {}
    calls_by_id: dict[str, _ToolCallProjection] = {}
    results_by_id: dict[str, dict[str, Any]] = {}
    for message_index, message in enumerate(messages):
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                results_by_id[call_id] = message
            continue
        if message.get("role") != "assistant":
            continue
        for call_index, call in enumerate(message.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            projection = _project_tool_call(call)
            calls_by_index[(message_index, call_index)] = projection
            call_id = call.get("id")
            if isinstance(call_id, str) and call_id:
                calls_by_id[call_id] = projection
    return calls_by_index, calls_by_id, results_by_id


def _project_tool_call(call: dict[str, Any]) -> _ToolCallProjection:
    function = call.get("function")
    function = function if isinstance(function, dict) else {}
    name = function.get("name")
    tool_name = _bounded_identity(name if isinstance(name, str) and name else "unknown")
    arguments = _bounded_json_value(_parse_arguments(function.get("arguments")))
    presentation = _call_presentation(call.get("presentation"), tool_name, arguments)
    detail = project_effect_detail(tool_name, arguments, presentation)
    return _ToolCallProjection(tool_name, arguments, presentation, detail)


def _project_user_entry(
    message: dict[str, Any], message_index: int, timestamp: int
) -> tuple[AgentTranscriptEntry, str] | None:
    text = _text_content(message.get("content"))
    if text is None:
        return None
    names, count = _attachment_placeholders(message)
    return _entry(
        _entry_id(_message_identity(message), message_index, "text"),
        AgentTranscriptEntryKind.USER_TEXT,
        text,
        timestamp=timestamp,
        payload=_EntryPayload(
            title="User message", attachment_names=names, attachment_count=count
        ),
    )


def _project_assistant_entries(
    message: dict[str, Any],
    message_index: int,
    timestamp: int,
    calls_by_index: dict[tuple[int, int], _ToolCallProjection],
    results_by_id: dict[str, dict[str, Any]],
) -> list[tuple[AgentTranscriptEntry, str]]:
    entries: list[tuple[AgentTranscriptEntry, str]] = []
    reasoning_entry = _project_reasoning_entry(message, message_index, timestamp)
    if reasoning_entry is not None:
        entries.append(reasoning_entry)
    text_entry = _project_assistant_text_entry(
        message, message_index, timestamp + len(entries)
    )
    if text_entry is not None:
        entries.append(text_entry)
    for call_index, call in enumerate(message.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        projection = calls_by_index.get((message_index, call_index))
        if projection is None:
            continue
        call_id = call.get("id")
        call_id = call_id if isinstance(call_id, str) and call_id else None
        result_message = results_by_id.get(call_id) if call_id else None
        tool_result = _tool_result_payload(
            projection.tool_name, projection.detail, result_message
        )
        entries.append(
            _project_tool_call_entry(
                message,
                message_index,
                timestamp + len(entries),
                call_index,
                call,
                projection,
                call_id,
                result_message,
                tool_result,
            )
        )
    return entries


def _project_reasoning_entry(
    message: dict[str, Any], message_index: int, timestamp: int
) -> tuple[AgentTranscriptEntry, str] | None:
    reasoning = _text_content(message.get("reasoning_content"))
    if reasoning is None:
        return None
    reasoning_id = message.get("reasoning_message_id")
    identity = reasoning_id if isinstance(reasoning_id, str) and reasoning_id else None
    identity = identity or _message_identity(message)
    return _entry(
        _entry_id(identity, message_index, "reasoning"),
        AgentTranscriptEntryKind.REASONING,
        reasoning,
        timestamp=timestamp,
        payload=_EntryPayload(title="Reasoning"),
    )


def _project_assistant_text_entry(
    message: dict[str, Any], message_index: int, timestamp: int
) -> tuple[AgentTranscriptEntry, str] | None:
    text = _text_content(message.get("content"))
    if text is None:
        return None
    return _entry(
        _entry_id(_message_identity(message), message_index, "text"),
        AgentTranscriptEntryKind.ASSISTANT_TEXT,
        text,
        timestamp=timestamp,
        payload=_EntryPayload(title="Assistant response"),
    )


def _project_tool_call_entry(
    message: dict[str, Any],
    message_index: int,
    timestamp: int,
    call_index: int,
    call: dict[str, Any],
    projection: _ToolCallProjection,
    call_id: str | None,
    result_message: dict[str, Any] | None,
    tool_result: _ToolResultProjection,
) -> tuple[AgentTranscriptEntry, str]:
    function = call.get("function")
    raw_arguments = function.get("arguments") if isinstance(function, dict) else None
    display = raw_arguments if isinstance(raw_arguments, str) else ""
    return _entry(
        _entry_id(_message_identity(message), message_index, f"call:{call_index}"),
        AgentTranscriptEntryKind.TOOL_CALL,
        display,
        timestamp=timestamp,
        payload=_tool_entry_payload(
            projection.tool_name,
            call_id,
            projection.arguments,
            projection.presentation,
            projection.detail,
            tool_result,
        ),
    )


def _project_tool_result_entry(
    message: dict[str, Any],
    message_index: int,
    timestamp: int,
    calls_by_id: dict[str, _ToolCallProjection],
) -> tuple[AgentTranscriptEntry, str]:
    source_id = message.get("tool_call_id")
    call_id = (
        source_id
        if isinstance(source_id, str) and source_id
        else f"tool:{message_index}"
    )
    projection = calls_by_id.get(call_id)
    name = projection.tool_name if projection is not None else message.get("name")
    tool_name = _bounded_identity(name if isinstance(name, str) and name else "unknown")
    arguments = projection.arguments if projection is not None else None
    call_presentation = (
        projection.presentation
        if projection is not None
        else _call_presentation(None, tool_name, arguments)
    )
    detail = (
        projection.detail
        if projection is not None
        else project_effect_detail(tool_name, arguments, call_presentation)
    )
    tool_result = _tool_result_payload(tool_name, detail, message)
    result_value = message.get("tool_result")
    display = (
        _json_text(result_value.get("output"))
        if isinstance(result_value, dict) and "output" in result_value
        else _text_content(message.get("content")) or ""
    )
    return _entry(
        _entry_id(_message_identity(message), message_index, "result"),
        AgentTranscriptEntryKind.TOOL_RESULT,
        display,
        timestamp=timestamp,
        payload=_tool_entry_payload(
            tool_name, call_id, arguments, call_presentation, detail, tool_result
        ),
    )


def _tool_entry_payload(
    tool_name: str,
    call_id: str | None,
    arguments: JsonValue,
    call_presentation: ToolCallPresentation,
    detail: EffectDetail,
    tool_result: _ToolResultProjection,
) -> _EntryPayload:
    return _EntryPayload(
        title=tool_name,
        tool_name=tool_name,
        tool_call_id=_bounded_identity(call_id) if call_id else None,
        arguments=arguments,
        result=tool_result.result,
        output_text=tool_result.output_text,
        status=tool_result.status,
        call_presentation=call_presentation,
        result_presentation=tool_result.presentation,
        detail=detail,
        state=tool_result.state,
    )


def _message_identity(message: dict[str, Any]) -> str | None:
    message_id = message.get("message_id")
    return message_id if isinstance(message_id, str) and message_id else None


def _call_presentation(
    value: Any, tool_name: str, arguments: JsonValue
) -> ToolCallPresentation:
    try:
        if value is not None:
            presentation = ToolCallPresentation.model_validate(value)
            return _bounded_call_presentation(presentation)
    except ValidationError:
        pass
    summary = _generic_call_summary(tool_name, arguments)
    display = ToolEffectCallDisplay(
        summary=summary,
        verb="Running",
        message=summary,
        settled_verb="Ran",
        settled_message=summary,
        status_text=f"Running {tool_name}",
    )
    return _bounded_call_presentation(
        ToolCallPresentation(kind=ToolEffectKind.TOOL, display=display)
    )


def _tool_result_payload(
    tool_name: str, detail: EffectDetail, result_message: dict[str, Any] | None
) -> _ToolResultProjection:
    if result_message is None:
        return _ToolResultProjection(
            RunningEffectState(), AgentTranscriptToolStatus.PENDING, None, None, None
        )
    content = _text_content(result_message.get("content")) or ""
    tagged = TaggedText.from_string(content)
    persisted = result_message.get("tool_result")
    persisted = persisted if isinstance(persisted, dict) else {}
    return _project_present_result(tool_name, detail, persisted, tagged)


def _project_present_result(
    tool_name: str, detail: EffectDetail, persisted: dict[str, Any], tagged: TaggedText
) -> _ToolResultProjection:
    output_value = persisted.get("output")
    output = (
        _bounded_json_value(cast(JsonValue, output_value))
        if output_value is not None
        else None
    )
    output_text = _bounded_text(tagged.message, _DISPLAY_TEXT_LIMIT)
    duration_value = persisted.get("duration")
    duration_ms = (
        max(0.0, duration_value * 1000)
        if isinstance(duration_value, int | float)
        else 0.0
    )
    cancelled = persisted.get("cancelled") is True or tagged.tag == CANCELLATION_TAG
    failed = tagged.tag == TOOL_ERROR_TAG
    presentation = _result_presentation(
        persisted.get("presentation"),
        detail,
        output,
        tool_name,
        tagged.message,
        cancelled=cancelled,
        failed=failed,
    )
    structured_output = _presentation_output(presentation, output)
    state, status = _effect_state(
        presentation, structured_output, output_text, duration_ms, cancelled, failed
    )
    result = output if isinstance(output, dict) else None
    return _ToolResultProjection(state, status, presentation, result, output_text)


def _presentation_output(
    presentation: ToolResultPresentation, output: JsonValue | None
) -> JsonValue | None:
    value = (
        presentation.projected_output
        if presentation.projected_output is not None
        else output
    )
    return _bounded_json_value(value) if value is not None else None


def _effect_state(
    presentation: ToolResultPresentation,
    output: JsonValue | None,
    output_text: str,
    duration_ms: float,
    cancelled: bool,
    failed: bool,
) -> tuple[EffectState, AgentTranscriptToolStatus]:
    display = _public_result_display(presentation.display)
    if cancelled:
        return (
            CancelledEffectState(
                reason=display.message,
                output_text=output_text,
                duration_ms=duration_ms,
                display=display,
            ),
            AgentTranscriptToolStatus.CANCELLED,
        )
    if failed:
        return (
            FailedEffectState(
                error=PublicError(message=display.message),
                output_text=output_text,
                duration_ms=duration_ms,
                display=display,
            ),
            AgentTranscriptToolStatus.FAILED,
        )
    projected_output = project_effect_output_value(
        presentation.kind,
        presentation.projected_output
        if presentation.projected_output is not None
        else output,
    )
    return (
        CompletedEffectState(
            output=projected_output,
            output_text=output_text,
            duration_ms=duration_ms,
            display=display,
        ),
        AgentTranscriptToolStatus.COMPLETED,
    )


def _result_presentation(
    value: Any,
    detail: EffectDetail,
    output: JsonValue | None,
    tool_name: str,
    result_text: str,
    *,
    cancelled: bool,
    failed: bool,
) -> ToolResultPresentation:
    try:
        if value is not None:
            presentation = ToolResultPresentation.model_validate(value)
            return _bounded_result_presentation(presentation)
    except ValidationError:
        pass
    message = _bounded_text(
        result_text
        or (
            f"{tool_name} cancelled"
            if cancelled
            else f"{tool_name} failed"
            if failed
            else f"{tool_name} completed"
        ),
        _PRESENTATION_TEXT_LIMIT,
    )
    display = ToolEffectResultDisplay(
        success=not (cancelled or failed), message=message
    )
    return _bounded_result_presentation(
        ToolResultPresentation(
            kind=detail.kind, display=display, projected_output=output
        )
    )


def _bounded_call_presentation(
    presentation: ToolCallPresentation,
) -> ToolCallPresentation:
    display = presentation.display
    bounded = display.model_copy(
        update={
            field_name: _bounded_text(value, _PRESENTATION_TEXT_LIMIT)
            if isinstance(value, str)
            else value
            for field_name in type(display).model_fields
            for value in (getattr(display, field_name),)
        }
    )
    return presentation.model_copy(update={"display": bounded})


def _bounded_result_presentation(
    presentation: ToolResultPresentation,
) -> ToolResultPresentation:
    display = presentation.display.model_copy(
        update={
            "verb": _bounded_text(presentation.display.verb, _PRESENTATION_TEXT_LIMIT),
            "message": _bounded_text(
                presentation.display.message, _PRESENTATION_TEXT_LIMIT
            ),
            "suffix": _bounded_text(
                presentation.display.suffix, _PRESENTATION_TEXT_LIMIT
            ),
            "warnings": [
                _bounded_text(warning, _PRESENTATION_TEXT_LIMIT)
                for warning in presentation.display.warnings[
                    :_MAX_PRESENTATION_WARNINGS
                ]
            ],
        }
    )
    output = presentation.projected_output
    return presentation.model_copy(
        update={
            "display": display,
            "projected_output": _bounded_json_value(output)
            if output is not None
            else None,
        }
    )


def _public_result_display(display: ToolEffectResultDisplay) -> EffectResultDisplay:
    return EffectResultDisplay(
        success=display.success,
        verb=display.verb,
        message=display.message,
        warnings=display.warnings,
        suffix=display.suffix,
    )


def _generic_call_summary(tool_name: str, arguments: JsonValue) -> str:
    if not isinstance(arguments, dict):
        return tool_name
    rendered = ", ".join(
        f"{key}={value!r}" for key, value in list(arguments.items())[:3]
    )
    return f"{tool_name}({rendered})"


def _parse_arguments(value: Any) -> JsonValue:
    if not isinstance(value, str):
        return None
    try:
        return cast(JsonValue, json.loads(value))
    except json.JSONDecodeError:
        return value


def _bounded_json_value(value: JsonValue) -> JsonValue:
    if len(_json_text(value).encode()) <= _TOOL_VALUE_BYTE_LIMIT:
        return value
    preview, _ = _truncate_utf8(_json_text(value), 2 * 1024)
    return {"_truncated": True, "preview": preview}


def _bounded_text(value: str, byte_limit: int) -> str:
    return _truncate_utf8(value, byte_limit)[0]


def _attachment_placeholders(message: dict[str, Any]) -> tuple[list[str], int]:
    images = message.get("images")
    if not isinstance(images, list):
        return [], 0
    names = [
        _bounded_text(Path(image["alias"]).name or image["alias"], 512)
        for image in images[:_MAX_ATTACHMENT_NAMES]
        if isinstance(image, dict) and isinstance(image.get("alias"), str)
    ]
    return names, len(images)


def _entry(
    entry_id: str,
    kind: AgentTranscriptEntryKind,
    display_text: str,
    *,
    timestamp: int,
    payload: _EntryPayload,
) -> tuple[AgentTranscriptEntry, str]:
    bounded_text, truncated = _truncate_utf8(display_text, _DISPLAY_TEXT_LIMIT)
    entry = AgentTranscriptEntry(
        entry_id=entry_id,
        kind=kind,
        display_text=bounded_text,
        digest="0" * 64,
        created_at=timestamp,
        updated_at=timestamp,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title=_bounded_text(payload.title, MAX_AGENT_TRANSCRIPT_ID_LENGTH),
        tool_name=payload.tool_name,
        tool_call_id=payload.tool_call_id,
        arguments=payload.arguments,
        result=payload.result,
        output_text=payload.output_text,
        status=payload.status,
        call_presentation=payload.call_presentation,
        result_presentation=payload.result_presentation,
        detail=payload.detail,
        state=payload.state,
        attachment_names=payload.attachment_names or [],
        attachment_count=payload.attachment_count,
        truncated=truncated,
        truncation=AgentTranscriptTruncation.DISPLAY_TEXT_LIMIT if truncated else None,
    )
    canonical = _json_text(
        entry.model_dump(
            mode="json", exclude={"entry_id", "digest", "created_at", "updated_at"}
        )
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return entry.model_copy(update={"digest": digest}), digest


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
    # Entry text, structured values, presentations, and attachment names are bounded
    # so one projected entry always fits the fixed response ceiling.
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
