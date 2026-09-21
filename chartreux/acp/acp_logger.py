from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import os
import re
from typing import TYPE_CHECKING

from cachetools import TTLCache

from chartreux.core.paths import (
    LITERAL_LOG_DIR,
    LOG_DIR,
    GlobalPath,
    ensure_chartreux_home_private,
)
from chartreux.core.session.session_permissions import ensure_private_directory
from chartreux.observability.logging import _ChartreuxFileHandler

if TYPE_CHECKING:
    from acp.connection import StreamEvent

ACP_LOG_DIR = GlobalPath(lambda: LOG_DIR.path / "acp")
ACP_LOG_FILE = GlobalPath(lambda: ACP_LOG_DIR.path / "messages.jsonl")
LITERAL_ACP_LOG_DIR = GlobalPath(lambda: LITERAL_LOG_DIR.path / "acp")
LITERAL_ACP_LOG_FILE = GlobalPath(lambda: LITERAL_ACP_LOG_DIR.path / "messages.jsonl")
MAX_LOG_SIZE_BYTES = 1_000_000
BACKUP_COUNT = 3

ACP_LOGGING_ENABLED_KEY = "CHARTREUX_ACP_LOGGING_ENABLED"

_session_cache: TTLCache[int | str, str] = TTLCache(maxsize=1000, ttl=3600)
_current_session: str | None = None
_logger: logging.Logger | None = None


def is_acp_logging_enabled() -> bool:
    return os.getenv(ACP_LOGGING_ENABLED_KEY, "").lower() in {"1", "true", "yes"}


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(record.msg, separators=(",", ":"))


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    ensure_chartreux_home_private()
    ensure_private_directory(LITERAL_LOG_DIR.path)
    ensure_private_directory(LITERAL_ACP_LOG_DIR.path)

    logger = logging.getLogger("acp_messages")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = _ChartreuxFileHandler(
        ACP_LOG_FILE.path,
        maxBytes=MAX_LOG_SIZE_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
        repair_path=LITERAL_ACP_LOG_FILE.path,
    )
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)

    _logger = logger
    return _logger


def _extract_session_id(message: dict) -> str | None:
    json_str = json.dumps(message)
    match = re.search(r'"(?:session_id|sessionId)":\s*"([^"]+)"', json_str)
    return match.group(1) if match else None


def _redact_sensitive_values(value: object) -> object:
    sensitive_names = {"authorization", "x-api-key", "set-cookie", "cookie"}
    sensitive_substrings = ("token", "key", "secret", "password")

    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if isinstance(key, str)
                and (
                    key.lower() in sensitive_names
                    or any(
                        substring in key.lower() for substring in sensitive_substrings
                    )
                )
                else _redact_sensitive_values(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive_values(item) for item in value)
    return value


def acp_message_observer(event: StreamEvent) -> None:
    if not is_acp_logging_enabled():
        return

    try:
        global _current_session

        message = event.message
        msg_id = message.get("id", "")

        if msg_id in _session_cache:
            session_id = _session_cache[msg_id]
        else:
            session_id = _extract_session_id(message) or _current_session

        if session_id is not None:
            _current_session = session_id
            if msg_id:
                _session_cache[msg_id] = session_id

        log_entry: dict = {
            "ts": datetime.now(UTC).isoformat(),
            "dir": "in" if event.direction.value == "incoming" else "out",
            "msg": _redact_sensitive_values(message),
            **({"session": session_id} if session_id else {}),
        }

        _get_logger().info(log_entry)
    except Exception:
        pass
