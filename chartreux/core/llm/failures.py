from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import email.utils
from enum import StrEnum, auto
from http import HTTPStatus
import json
import math
import time
from typing import Any

import httpx


class FailureCategory(StrEnum):
    AUTH = auto()
    BILLING = auto()
    QUOTA = auto()
    INVALID_REQUEST = auto()
    CONTEXT_SIZE = auto()
    CANCELLATION = auto()
    CONNECTION = auto()
    TIMEOUT = auto()
    SERVER_ERROR = auto()
    OVERLOAD = auto()
    RATE_LIMIT = auto()
    CONFLICT = auto()
    TOO_EARLY = auto()
    # Typed transient dispositions measured at the response boundary (WP1).
    # A stream that ends without a finish reason, or a provider that marks
    # its terminal response incomplete, is a typed transient form — not an
    # unknown, and not a connection failure. An empty assistant response at
    # the final-chunk boundary is the observed 2026-10-17 main-agent
    # terminal mode; without a typed class the recovery layer's
    # never-retry-unknown rule leaves it manual forever.
    INCOMPLETE_STREAM = auto()
    EMPTY_RESPONSE = auto()
    UNKNOWN = auto()


_OVERLOAD_STATUS = 529

# The low-level retry/failover sets stay EXPLICIT frozensets: the WP1 typed
# transient categories (INCOMPLETE_STREAM, EMPTY_RESPONSE) are deliberately
# NOT members. They are consumed by the recovery layer's eligible_classes
# config instead — joining the low-level sets would silently grant in-call
# retry and same-base failover authority to model-content dispositions, which
# is a separate deliberate decision (recorded in the WP1 report), never an
# automatic consequence of acquiring a label.
_RETRY_CATEGORIES = frozenset({
    FailureCategory.CONNECTION,
    FailureCategory.TIMEOUT,
    FailureCategory.SERVER_ERROR,
    FailureCategory.OVERLOAD,
    FailureCategory.RATE_LIMIT,
    FailureCategory.CONFLICT,
    FailureCategory.TOO_EARLY,
})
_FAILOVER_CATEGORIES = frozenset({
    FailureCategory.CONNECTION,
    FailureCategory.TIMEOUT,
    FailureCategory.SERVER_ERROR,
    FailureCategory.OVERLOAD,
    FailureCategory.RATE_LIMIT,
})

# Categories that must never be masked by a transient sibling anywhere in a
# cause chain or exception group: protected members dominate classification.
_PROTECTED_CATEGORIES = frozenset({
    FailureCategory.CANCELLATION,
    FailureCategory.AUTH,
    FailureCategory.BILLING,
    FailureCategory.QUOTA,
    FailureCategory.INVALID_REQUEST,
    FailureCategory.CONTEXT_SIZE,
})


@dataclass(frozen=True, slots=True)
class FailureInfo:
    category: FailureCategory
    status: int | None = None
    error_code: str | None = None
    error_type: str | None = None
    retry_after: float | None = None

    @property
    def retry_eligible(self) -> bool:
        return self.category in _RETRY_CATEGORIES

    @property
    def failover_eligible(self) -> bool:
        return self.category in _FAILOVER_CATEGORIES


class RequestRetryBudget:
    """A monotonic deadline shared by every attempt of one logical completion."""

    def __init__(
        self, max_elapsed_time: float, *, started_at: float | None = None
    ) -> None:
        if max_elapsed_time < 0:
            raise ValueError("max_elapsed_time must be non-negative")
        self._max_elapsed_time = max_elapsed_time
        self._deadline = (
            time.monotonic() if started_at is None else started_at
        ) + max_elapsed_time

    @property
    def max_elapsed_time(self) -> float:
        """The configured budget, for evidence that reports the bound."""
        return self._max_elapsed_time

    @property
    def deadline(self) -> float:
        return self._deadline

    @property
    def remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def allows_delay(self, delay: float) -> bool:
        return delay <= self.remaining


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _structured_error(error: BaseException) -> tuple[str | None, str | None]:
    code = _text(getattr(error, "error_code", None))
    error_type = _text(getattr(error, "error_type", None))
    body = getattr(error, "body_text", None)
    response = getattr(error, "response", None)
    if response is None:
        response = getattr(error, "raw_response", None)
    if not body and response is not None:
        try:
            response.read()
            body = response.text
        except Exception:
            body = None
    if isinstance(body, bytes):
        body = body.decode(errors="replace")
    if isinstance(body, dict):
        data: Any = body
    elif isinstance(body, str) and body:
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            data = {}
    else:
        data = {}
    detail = data.get("error", data) if isinstance(data, dict) else {}
    if isinstance(detail, dict):
        code = code or _text(detail.get("code"))
        error_type = error_type or _text(detail.get("type"))
    return code, error_type


def _status(error: BaseException) -> int | None:
    value = getattr(error, "status", None)
    if isinstance(value, int):
        return value
    response = getattr(error, "response", None)
    if response is None:
        response = getattr(error, "raw_response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _retry_after(error: BaseException) -> float | None:  # noqa: PLR0911
    """Server-directed delay in seconds, or None when absent or unusable.

    WP4: a non-finite value is never a usable delay — an infinite Retry-After
    (a non-finite attribute or an overflowing delta-seconds header) would arm
    an unbounded cooldown and park recovery silently instead of failing
    deterministically.
    """
    existing = getattr(error, "retry_after", None)
    if isinstance(existing, int | float) and not isinstance(existing, bool):
        if math.isfinite(existing) and existing >= 0:
            return float(existing)
        return None
    headers = getattr(error, "headers", None)
    response = getattr(error, "response", None)
    if response is None:
        response = getattr(error, "raw_response", None)
    if headers is None and response is not None:
        headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = str(headers.get("retry-after", "")).strip()
    if value.isascii() and value.isdigit():
        seconds = float(value)
        return seconds if math.isfinite(seconds) else None
    if not value:
        return None
    try:
        at = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    return max((at - datetime.now(UTC)).total_seconds(), 0.0)


_MAX_FAILURE_WALK = 8


def _walk_failures(error: BaseException) -> tuple[list[BaseException], bool]:
    """Bounded, cycle-safe walk of one failure's cause graph (D3).

    The error itself comes first, then its ``__cause__`` chain and the
    members of any :class:`BaseExceptionGroup` encountered along the way
    (groups reduce to their members). A visited set makes cyclic cause
    graphs terminate, and the depth bound keeps classification cheap, so a
    grouped or cyclic failure can neither crash classification nor mask a
    protected member (cancellation/auth) behind a transient sibling.

    Returns the visited members plus whether the depth bound cut the walk
    short with unvisited candidates still pending. A truncated walk cannot
    prove that no protected member hides beyond the bound, so the caller
    must not classify from it as if the graph were exhausted.
    """
    ordered: list[BaseException] = []
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    truncated = False
    while True:
        # Already-visited candidates contribute nothing new (their causes
        # and group members were enqueued when first visited), so drain
        # them without consuming the depth bound.
        while pending and id(pending[0]) in seen:
            pending.pop(0)
        if not pending:
            break
        if len(ordered) >= _MAX_FAILURE_WALK:
            truncated = True
            break
        current = pending.pop(0)
        seen.add(id(current))
        ordered.append(current)
        cause = current.__cause__
        if cause is not None:
            pending.append(cause)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return ordered, truncated


def _declared_category(error: BaseException) -> FailureCategory | None:
    """The exception's own measured category, when it declares one (WP1).

    Typed dispositions mark their exceptions with a ``failure_category``
    class attribute at the point the condition is measured, so the typed
    provenance survives wrapping instead of being relabeled by whatever
    the classifier can infer from shape alone.
    """
    value = getattr(error, "failure_category", None)
    return value if isinstance(value, FailureCategory) else None


def classify_failure(error: BaseException) -> FailureInfo:
    """Classify one failure, honoring the exception's cached info first."""
    existing = getattr(error, "failure_info", None)
    if isinstance(existing, FailureInfo):
        return existing
    return classify_failure_fresh(error)


def _is_model_refusal(error: BaseException) -> bool:
    """Whether the error is a measured model refusal (a content disposition).

    The import is lazy: ``chartreux.core.errors`` sits below the llm layer,
    and a module-level import risks an import cycle through intermediate
    package modules.
    """
    from chartreux.core.errors import RefusalError

    return isinstance(error, RefusalError)


def _local_hard_stop_category(error: BaseException) -> FailureCategory | None:
    """A local hard stop's protected category, when it is one (review R1).

    Local policy/context stops carry no provider status and no structured
    error code, so shape classification alone reads them as UNKNOWN — and in
    a group an UNKNOWN member is skipped, letting a transient sibling mask
    a stop that must never be retried or failed over. Recognizing them here
    gives them their protected category (never masked, never retried). The
    imports are lazy for the same cycle-avoidance reason as
    :func:`_is_model_refusal`.
    """
    from chartreux.core.errors import ContextTooLongError
    from chartreux.core.tools.models import ToolPermissionError

    if isinstance(error, ContextTooLongError):
        return FailureCategory.CONTEXT_SIZE
    if isinstance(error, ToolPermissionError):
        return FailureCategory.AUTH
    return None


def classify_failure_fresh(error: BaseException) -> FailureInfo:
    """Classify without consulting the error's cached ``failure_info``.

    ``BackendError.failure_info`` resolves lazily through this entry point
    (D2): an eager construction-time classification would freeze a
    ``status=None`` request error before its transport cause is attached by
    ``raise ... from``, and a wrapped READ TIMEOUT would be recorded as a
    plain connection failure, losing the timeout subtype and its retry
    semantics.
    """
    members, truncated = _walk_failures(error)
    inherited: FailureInfo | None = None
    for member in members:
        info = _classify_single(member)
        if info.category is FailureCategory.UNKNOWN:
            if _is_model_refusal(member):
                # A model refusal is a measured content stop: it dominates
                # group reduction so a transient sibling can never mask it,
                # and its UNKNOWN category keeps it report-and-park.
                return info
            continue
        if info.category in _PROTECTED_CATEGORIES:
            # Protected precedence: a chained or grouped cancellation/auth
            # member is never masked by a transient sibling.
            return info
        if inherited is None:
            inherited = info
    if truncated:
        # The bounded walk stopped with unvisited candidates: a protected
        # member may hide beyond the bound, so the first transient sibling
        # is not a safe classification — report-and-park instead (the
        # general error policy's uncertainty rule).
        return FailureInfo(FailureCategory.UNKNOWN)
    if inherited is not None:
        return inherited
    return _classify_single(error)


def _classify_single(error: BaseException) -> FailureInfo:  # noqa: PLR0912
    status = _status(error)
    code, error_type = _structured_error(error)
    token = " ".join(filter(None, (code, error_type))).casefold().replace("-", "_")

    # Local hard stops precede shape classification: a tool-permission
    # denial or an over-limit context carries no provider status/code, and
    # without this recognition it reads UNKNOWN and can be masked by a
    # transient sibling in a group (review R1). Both map to protected
    # categories — never masked, never retried, never failed over.
    if (local := _local_hard_stop_category(error)) is not None:
        category = local
    # Structured permanent classes deliberately precede status classification.
    elif any(
        part in token
        for part in ("authentication", "invalid_api_key", "unauthorized", "auth_error")
    ):
        category = FailureCategory.AUTH
    elif any(
        part in token
        for part in (
            "billing",
            "payment_required",
            "insufficient_funds",
            "credit_balance",
        )
    ):
        category = FailureCategory.BILLING
    elif any(
        part in token
        for part in (
            "insufficient_quota",
            "quota_exceeded",
            "billing_hard_limit",
            "usage_limit",
        )
    ):
        category = FailureCategory.QUOTA
    elif any(
        part in token
        for part in (
            "context_length",
            "context_size",
            "model_context_exceeded",
            "prompt_too_long",
        )
    ):
        category = FailureCategory.CONTEXT_SIZE
    elif any(
        part in token for part in ("invalid_request", "validation_error", "bad_request")
    ):
        category = FailureCategory.INVALID_REQUEST
    elif any(part in token for part in ("cancelled", "canceled", "cancellation")):
        category = FailureCategory.CANCELLATION
    elif isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)):
        category = FailureCategory.CANCELLATION
    elif (declared := _declared_category(error)) is not None:
        # The exception's own measured disposition (WP1 typed transients).
        category = declared
    elif status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
        category = FailureCategory.AUTH
    elif status == HTTPStatus.PAYMENT_REQUIRED:
        category = FailureCategory.BILLING
    elif status in {
        HTTPStatus.BAD_REQUEST,
        HTTPStatus.NOT_FOUND,
        HTTPStatus.METHOD_NOT_ALLOWED,
        HTTPStatus.UNPROCESSABLE_ENTITY,
    }:
        category = FailureCategory.INVALID_REQUEST
    elif status == HTTPStatus.REQUEST_TIMEOUT or isinstance(
        error, (httpx.TimeoutException, TimeoutError)
    ):
        category = FailureCategory.TIMEOUT
    elif status == HTTPStatus.CONFLICT:
        category = FailureCategory.CONFLICT
    elif status == HTTPStatus.TOO_EARLY:
        category = FailureCategory.TOO_EARLY
    elif status == HTTPStatus.TOO_MANY_REQUESTS:
        # Only explicitly transient rate-limit codes are safe to replay.
        category = (
            FailureCategory.RATE_LIMIT
            if any(
                part in token
                for part in ("rate_limit", "too_many_requests", "overload")
            )
            else FailureCategory.UNKNOWN
        )
    elif status is not None and status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        category = (
            FailureCategory.OVERLOAD
            if status in {HTTPStatus.SERVICE_UNAVAILABLE, _OVERLOAD_STATUS}
            or "overload" in token
            else FailureCategory.SERVER_ERROR
        )
    elif isinstance(error, httpx.RequestError):
        # D1: only a real transport error is a connection failure. An
        # exception merely EXPOSING status=None (an unmapped stream error, a
        # permanent local fault — TLS/certificate, proxy, DNS-config,
        # protocol misuse) is NOT relabeled as a connection failure; it
        # stays UNKNOWN and reports-and-parks instead of being retried.
        category = FailureCategory.CONNECTION
    else:
        category = FailureCategory.UNKNOWN
    return FailureInfo(category, status, code, error_type, _retry_after(error))


# Concise public spelling for the attempt coordinator.
classify = classify_failure
