"""OSC 7501 program-status reporting for the terminal front end.

The encoder is pure: it validates and encodes, nothing else. The message
pipeline (sanitize, one redaction pass per stage, codepoint-safe truncation,
cap checks) lives here too so the fail-closed caps cannot drift from the
wire format. The only core dependency is the secret redactor, reached lazily
by module name so this module keeps the Textual layer's clean import closure.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from importlib import import_module
import re
from typing import Any

_REDACTION_MODULE = "chartreux.core.tools.secret_redaction"
_OSC_PREFIX = "\x1b]7501;"
_BEL = "\x07"
_STATES = frozenset({"idle", "working", "done", "blocked", "error"})
_KINDS = frozenset({"question", "permission"})
# Validated whole, not just by length: an unchecked app name could inject a pair.
_APP_PATTERN = re.compile(r"[A-Za-z0-9_.+-]{1,32}")
# Whole OSC 7501 report, its base64 message, and the message's decoded bytes.
_MAX_REPORT_BYTES = 4096
_MAX_ENCODED_MSG_BYTES = 2732
_MAX_MSG_BYTES = 2048
# Pass 1 redacts and scans the RAW input synchronously on the event loop, so an
# unbounded prompt costs the loop directly; the payload is capped far below this
# anyway, and omitting the message can never leak.
_MAX_RAW_INPUT_CHARS = 256_000
# Control characters become spaces so neighbouring words do not merge.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SPACE_RUNS = re.compile(r" {2,}")
# Stripped after the whitespace collapse: a credential split by one of these
# would otherwise be rebuilt verbatim. ZWJ/ZWNJ stay (emoji, Indic, Persian).
_INVISIBLE_CHARS = re.compile(
    "[\u061c\u200b\u200e\u200f\u2060\ufeff\u2028\u2029\u202a-\u202e\u2066-\u2069]"
)


def encode_report(
    state: str,
    *,
    kind: str | None = None,
    msg: str | None = None,
    app: str = "chartreux",
) -> str:
    """Encode one OSC 7501 report, refusing anything the terminal would reject."""
    if state not in _STATES:
        raise ValueError(f"unsupported program-status state: {state!r}")
    if kind is not None:
        if kind not in _KINDS:
            raise ValueError(f"unsupported program-status kind: {kind!r}")
        if state != "blocked":
            raise ValueError(f"program-status kind {kind!r} requires state blocked")
    if not _APP_PATTERN.fullmatch(app):
        raise ValueError(f"invalid program-status app name: {app!r}")
    pairs = [f"state={state}"]
    if kind is not None:
        pairs.append(f"kind={kind}")
    pairs.append(f"app={app}")
    if msg is not None:
        body = msg.encode("utf-8")
        encoded = base64.b64encode(body).decode("ascii")
        if len(body) > _MAX_MSG_BYTES or len(encoded) > _MAX_ENCODED_MSG_BYTES:
            raise ValueError("program-status message exceeds the maximum size")
        pairs.append(f"msg={encoded}")
    report = f"{_OSC_PREFIX}{':'.join(pairs)}{_BEL}"
    if len(report) > _MAX_REPORT_BYTES:
        raise ValueError("program-status report exceeds the sequence cap")
    return report


def encode_clear() -> str:
    """Encode the bare clear that removes every OSC 7501 record on the terminal."""
    return f"{_OSC_PREFIX}state=clear{_BEL}"


def prepare_message(text: str | None) -> str | None:
    """Reduce UI text to a message payload, or None when it must be omitted.

    Three redaction passes (raw, post-sanitize, exact post-truncation string)
    and a per-pass-input candidate check: a pass whose decode scan could have
    been budget-truncated leaves it unproven that every carrier was examined,
    so the message pair is dropped rather than sent. Every failure path,
    including a pass that raises, omits the message instead of sending text.
    An input past ``_MAX_RAW_INPUT_CHARS`` is omitted before pass 1 runs: that
    pass would redact and scan the whole raw text on the event loop, and the
    payload is capped at ``_MAX_MSG_BYTES`` regardless.
    """
    if text is None:
        return None
    if len(text) > _MAX_RAW_INPUT_CHARS:
        return None
    try:
        return _prepare_redacted_message(text)
    except Exception:
        return None


def _prepare_redacted_message(text: str) -> str | None:
    redaction = _redaction_module()
    # With no known values nothing can leak and scan truncation is
    # harmless, so the exact candidate accounting is skipped.
    account = bool(redaction.known_secret_values())
    if account and _scan_saturated(redaction, text):
        return None
    sanitized = _sanitize(redaction.redact(text))
    if not sanitized or (account and _scan_saturated(redaction, sanitized)):
        return None
    truncated = _truncate_utf8(redaction.redact(sanitized), _MAX_MSG_BYTES)
    if account and _scan_saturated(redaction, truncated):
        return None
    final = redaction.redact(truncated)
    body = final.encode("utf-8")
    if len(body) > _MAX_MSG_BYTES or (
        len(base64.b64encode(body)) > _MAX_ENCODED_MSG_BYTES
    ):
        return None
    return final


class ProgramStatusEmitter:
    """Publish OSC 7501 records for the running app; never raises.

    Guards mirror the notification adapter's: nothing is written headless,
    with no driver, while the config gate is off, before the app is running,
    or once chartreux shutdown has started. The running/shutdown checks are
    load-bearing: a write into the dead writer queue can block forever, which
    no try/except can catch, and teardown finalizers must not emit.

    ``clear_record`` is the ONE deliberate exception to that shutdown-write
    rule, and it bypasses three guards on purpose: ``_shutdown_started``
    (already set on every Ctrl+C/Ctrl+D/SIGTERM and force-quit path),
    ``is_running`` (False inside ``_shutdown`` by construction) and the
    config gate (which can flip off mid-session while the record we already
    wrote persists). Its own guards are "we spoke at least once", not
    headless, and a driver. Restoring any bypassed guard here would silently
    disable the exit clear.
    """

    def __init__(self, app: Any, *, get_enabled: Callable[[], bool]) -> None:
        self._app = app
        self._get_enabled = get_enabled
        self._last_report: str | None = None
        self._generation = 0

    @property
    def generation(self) -> int:
        """Count every emit() call, deduped or not.

        A caller that awaits between deciding to report and reporting reads this
        first and drops its report if the value moved: another report (a durable
        terminal outcome, a newer prompt) owned the record in the meantime.
        """
        return self._generation

    def emit(
        self, state: str, *, kind: str | None = None, msg: str | None = None
    ) -> None:
        """Publish one report; a byte-identical repeat of the full report is skipped."""
        self._generation += 1
        try:
            if not self._can_write():
                return
            report = encode_report(state, kind=kind, msg=prepare_message(msg))
            # The encoded report carries the full sanitized message, so a
            # second blocked prompt with new text re-emits.
            if report == self._last_report:
                return
            self._write(report)
        except Exception:
            # A dropped record beats a broken turn: close paths are non-essential.
            return

    def republish(self) -> None:
        """Re-send the last written report, bypassing dedup (terminal resume)."""
        try:
            if self._last_report is not None and self._can_write():
                self._write(self._last_report)
        except Exception:
            return

    def clear_record(self) -> None:
        """Remove every OSC 7501 record on the terminal; the exit clear.

        Writes ``encode_clear()`` straight to the driver: ``emit()`` would
        suppress the write via ``_can_write()`` and bump ``generation``, and
        ``_write()`` would keep it as the last report so a later
        ``republish()`` would re-send the clear.

        Bypasses the shutdown, running, and config-gate guards on purpose (see
        the class docstring), and never raises: the body swallows
        ``BaseException`` because this runs as the first statement of
        ``_shutdown``, where a ``CancelledError`` or ``KeyboardInterrupt``
        escaping would skip the rest of the shutdown.
        """
        try:
            if self._last_report is None:
                return
            if getattr(self._app, "is_headless", True):
                return
            if getattr(self._app, "_driver", None) is None:
                return
            self._app._driver.write(encode_clear())
            self._last_report = None
        except BaseException:
            return

    def _write(self, report: str) -> None:
        self._app._driver.write(report)
        self._last_report = report

    def _can_write(self) -> bool:
        if not self._get_enabled():
            return False
        if getattr(self._app, "is_headless", True):
            return False
        if getattr(self._app, "_driver", None) is None:
            return False
        if not getattr(self._app, "is_running", False):
            return False
        return not getattr(self._app, "_shutdown_started", False)


def _redaction_module() -> Any:
    return import_module(_REDACTION_MODULE)


def _scan_saturated(redaction: Any, text: str) -> bool:
    """Whether the redactor's own decode scan could have stopped early on *text*."""
    candidates = redaction._encoded_candidates(text)
    return len(candidates) >= redaction._MAX_ENCODED_CANDIDATES


def _sanitize(text: str) -> str:
    first_line = text.split("\n", 1)[0]
    replaced = _CONTROL_CHARS.sub(" ", first_line)
    collapsed = _SPACE_RUNS.sub(" ", replaced)
    return _INVISIBLE_CHARS.sub("", collapsed).strip()


def _truncate_utf8(text: str, limit: int) -> str:
    """Longest complete-codepoint prefix whose UTF-8 encoding fits in *limit*."""
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    # The encoded text is valid UTF-8, so ignoring the trailing partial
    # sequence drops exactly the split codepoint and nothing else.
    return data[:limit].decode("utf-8", errors="ignore")


__all__ = ["ProgramStatusEmitter", "encode_clear", "encode_report", "prepare_message"]
