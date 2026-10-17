"""Protocol error codes, error models, and validation redaction helpers."""

from __future__ import annotations

from enum import StrEnum, auto
from functools import cache
from typing import get_origin

from pydantic import JsonValue

from chartreux.app_server._model import ProtocolModel


class ProtocolErrorCode(StrEnum):
    INVALID_REQUEST = auto()
    INVALID_PARAMS = auto()
    NOT_INITIALIZED = auto()
    NOT_FOUND = auto()
    CONFLICT = auto()
    STALE_TURN = auto()
    NOT_STEERABLE = auto()
    CALLBACK_CLOSED = auto()
    COMPACTION_FAILED = auto()
    UNAUTHORIZED = auto()
    FORBIDDEN = auto()
    METHOD_NOT_FOUND = auto()
    NOT_IMPLEMENTED = auto()
    STALE_CURSOR = auto()
    INTERNAL_ERROR = auto()


class InvalidParamsIssue(ProtocolModel):
    path: list[str | int]
    message: str


class InvalidParamsData(ProtocolModel):
    error_count: int
    issues: list[InvalidParamsIssue]


class ProtocolError(ProtocolModel):
    code: ProtocolErrorCode
    message: str
    data: JsonValue = None


_REDACTED_VALIDATION_VALUE = "[redacted]"
_VALIDATION_MESSAGE_TEMPLATES = {
    "missing": "Field required",
    "type_error": "Invalid type",
    "value_error": "Invalid value",
}


@cache
def _schema_field_names() -> frozenset[str]:
    return frozenset(
        name
        for model in ProtocolModel.__subclasses__()
        for field_name, field in model.model_fields.items()
        for name in (field_name, field.alias)
        if name is not None
    )


@cache
def _mapping_schema_field_names() -> frozenset[str]:
    return frozenset(
        name
        for model in ProtocolModel.__subclasses__()
        for field_name, field in model.model_fields.items()
        if get_origin(field.annotation) is dict
        for name in (field_name, field.alias)
        if name is not None
    )


def redact_validation_message(error_type: str) -> str:
    """Render a diagnostic from the Pydantic error type, never its input message."""
    if error_type in _VALIDATION_MESSAGE_TEMPLATES:
        return _VALIDATION_MESSAGE_TEMPLATES[error_type]
    if error_type.endswith("_type") or error_type.endswith("_parsing"):
        return "Invalid type"
    if error_type.startswith("value_error") or error_type.endswith("_value"):
        return "Invalid value"
    return "Invalid value"


def redact_validation_path(path: tuple[str | int, ...]) -> list[str | int]:
    """Keep schema fields and indices while hiding submitted mapping keys."""
    schema_fields = _schema_field_names()
    mapping_fields = _mapping_schema_field_names()
    submitted_mapping = False
    redacted: list[str | int] = []
    for segment in path:
        if isinstance(segment, int):
            redacted.append(segment)
            continue
        if submitted_mapping or segment not in schema_fields:
            redacted.append(_REDACTED_VALIDATION_VALUE)
        else:
            redacted.append(segment)
        if segment in mapping_fields:
            submitted_mapping = True
    return redacted


def format_invalid_params_issues(data: JsonValue) -> str | None:
    """Render field-level detail carried by an INVALID_PARAMS error."""
    if not isinstance(data, dict):
        return None
    issues = data.get("issues")
    if not isinstance(issues, list) or not issues:
        return None
    parts: list[str] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        path = issue.get("path")
        location = (
            ".".join(str(segment) for segment in path)
            if isinstance(path, list) and path
            else "<root>"
        )
        message = issue.get("message")
        parts.append(f"{location}: {message}" if message else location)
    if not parts:
        return None
    return "; ".join(parts)


def _render_protocol_error(error: ProtocolError) -> str:
    detail = format_invalid_params_issues(error.data)
    if detail is None:
        return error.message
    return f"{error.message} ({detail})"
