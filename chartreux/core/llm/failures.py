from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import email.utils
from enum import StrEnum, auto
from http import HTTPStatus
import json
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
    UNKNOWN = auto()


_OVERLOAD_STATUS = 529

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
        self._deadline = (
            time.monotonic() if started_at is None else started_at
        ) + max_elapsed_time

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


def _retry_after(error: BaseException) -> float | None:
    existing = getattr(error, "retry_after", None)
    if isinstance(existing, int | float) and existing >= 0:
        return float(existing)
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
        return float(value)
    if not value:
        return None
    try:
        at = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    return max((at - datetime.now(UTC)).total_seconds(), 0.0)


def classify_failure(error: BaseException) -> FailureInfo:  # noqa: PLR0912
    existing = getattr(error, "failure_info", None)
    if isinstance(existing, FailureInfo):
        return existing
    cause = error.__cause__
    if cause is not None and cause is not error:
        inherited = classify_failure(cause)
        if inherited.category is not FailureCategory.UNKNOWN:
            return inherited
    status = _status(error)
    code, error_type = _structured_error(error)
    token = " ".join(filter(None, (code, error_type))).casefold().replace("-", "_")

    # Structured permanent classes deliberately precede status classification.
    if any(
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
    elif (
        isinstance(error, httpx.RequestError)
        or getattr(error, "status", object()) is None
    ):
        category = FailureCategory.CONNECTION
    else:
        category = FailureCategory.UNKNOWN
    return FailureInfo(category, status, code, error_type, _retry_after(error))


# Concise public spelling for the attempt coordinator.
classify = classify_failure
