"""Structured shell denials; render only when producing permission feedback."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from chartreux.core.tools.secret_redaction import (
    REDACTED_PLACEHOLDER,
    known_secret_values,
    redact,
    redact_shell_source as _redact_shell_source,
)

MAX_DIAGNOSTIC_LENGTH = 1024
_PREVIEW_LENGTH = 240
_STATIC_MESSAGES = {
    "protected_env": "Command denied: unsetting a protected shell environment variable",
    "sensitive_command": "Command denied by a sensitive command rule",
    "git_cwd_unknown": "Command denied: the working directory at this git command is not statically known, so its repository git config cannot be inspected",
    "tool_denied": "Tool denied: bash",
    "cwd_outside": "Shell cwd is outside the authorized workspace",
    "operand_cwd_unknown": "Shell file operand has a working directory that is not statically known",
    "path_outside": "Shell path is outside the authorized workspace; only an explicit user scope change can authorize it",
    "metadata_unlocated": "Command denied: protected Git metadata cannot be located",
    "metadata_redirect": "Command denied: redirection to protected Git metadata",
    "byte_budget": "Command denied: shell analysis budget exceeded (bytes)",
    "eval_exec": "Command denied: eval and exec cannot be safely inspected",
}


@dataclass(frozen=True)
class Diagnostic:
    code: str
    offending_token: str | None = None
    command_part: str | None = None
    detail: str | None = None
    related: tuple[Diagnostic, ...] = ()


def _preview(value: str, limit: int = _PREVIEW_LENGTH) -> str:
    # Scan the complete original value first: truncation/escaping must not hide
    # a registered secret from the redactor or expose a prefix of that secret.
    redacted = redact(value)
    pieces: list[str] = []
    length = 0
    for character in redacted:
        if character == "\\":
            piece = "\\\\"
        elif (
            character in "[]<>`*_{}()!#|&'\""
            or unicodedata.category(character).startswith("C")
            or character in "\n\r\t\u2028\u2029"
        ):
            piece = f"\\u{ord(character):04x}"
        else:
            piece = character
        if length + len(piece) > limit - 3:
            pieces.append("...")
            break
        pieces.append(piece)
        length += len(piece)
    return "".join(pieces)


def _command_preview(value: str, limit: int = _PREVIEW_LENGTH) -> str:
    # Wrapper expansion uses lossless shlex.join serialization for policy checks.
    # Nested source needs multiple recovery layers: shell quoting can otherwise
    # split an apostrophe-containing credential before the display redactor.
    sanitized = _redact_shell_source(value)
    return _preview(REDACTED_PLACEHOLDER if sanitized is None else sanitized, limit)


def render_diagnostic(diagnostic: Diagnostic) -> str:
    """The single redaction, escaping and bounding boundary for shell denials."""
    field_preview = _command_preview if known_secret_values() else _preview
    token = field_preview(diagnostic.offending_token or "")
    part = field_preview(diagnostic.command_part or "")
    original_detail = (
        ", ".join(sorted({item.detail or "" for item in diagnostic.related}))
        if diagnostic.related
        else diagnostic.detail or ""
    )
    detail = field_preview(original_detail, 640)
    match diagnostic.code:
        case "recursive_rm":
            message = (
                f"Command denied: recursive rm option '{token}' in '{part}'. "
                "This is a hard guard regardless of target. "
                "Use non-recursive rm -- 'file', then rmdir -- 'dir'."
            )
        case "path_glob":
            message = (
                f"Shell path glob candidate '{token}' in command segment '{part}' "
                "cannot be safely inspected"
            )
        case "denylist":
            message = f"Command denied: '{part}' matches denylist pattern '{token}'. Do not attempt to run this command."
        case "standalone":
            message = f"Command denied: '{part}' is not allowed as a standalone command. Do not attempt to run this command."
        case "policy":
            message = f"Command denied: {detail}: '{part}'"
        case "analysis":
            message = f"Command denied: unsupported shell syntax: {detail}"
        case "nested_analysis":
            message = f"Command denied: nested shell syntax cannot be safely inspected: {detail}"
        case "command":
            message = f"Command denied: {detail}"
        case _:
            message = _STATIC_MESSAGES.get(diagnostic.code, detail)
    return message[:MAX_DIAGNOSTIC_LENGTH]
