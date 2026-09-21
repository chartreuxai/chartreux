from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from chartreux.core.llm.exceptions import BackendErrorBuilder
from chartreux.core.llm.failures import FailureCategory, RequestRetryBudget, classify
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.utils.retry import _next_delay, async_retry


def _http_error(
    status: int,
    *,
    code: str | None = None,
    error_type: str | None = None,
    retry_after: str | None = None,
) -> httpx.HTTPStatusError:
    detail = {
        key: value for key, value in {"code": code, "type": error_type}.items() if value
    }
    request = httpx.Request("POST", "https://provider.invalid/chat")
    response = httpx.Response(
        status,
        request=request,
        headers={"Retry-After": retry_after} if retry_after else None,
        json={"error": detail},
    )
    return httpx.HTTPStatusError("provider error", request=request, response=response)


@pytest.mark.parametrize(
    ("error", "category", "retry", "failover"),
    [
        (_http_error(500, code="invalid_api_key"), FailureCategory.AUTH, False, False),
        (
            _http_error(500, code="billing_hard_limit"),
            FailureCategory.BILLING,
            False,
            False,
        ),
        (
            _http_error(429, code="insufficient_quota"),
            FailureCategory.QUOTA,
            False,
            False,
        ),
        (
            _http_error(400, code="invalid_request_error"),
            FailureCategory.INVALID_REQUEST,
            False,
            False,
        ),
        (
            _http_error(400, code="context_length_exceeded"),
            FailureCategory.CONTEXT_SIZE,
            False,
            False,
        ),
        (asyncio.CancelledError(), FailureCategory.CANCELLATION, False, False),
        (httpx.ConnectError("down"), FailureCategory.CONNECTION, True, True),
        (httpx.ReadTimeout("slow"), FailureCategory.TIMEOUT, True, True),
        (_http_error(408), FailureCategory.TIMEOUT, True, True),
        (_http_error(500), FailureCategory.SERVER_ERROR, True, True),
        (
            _http_error(503, error_type="overloaded_error"),
            FailureCategory.OVERLOAD,
            True,
            True,
        ),
        (
            _http_error(429, code="rate_limit_exceeded"),
            FailureCategory.RATE_LIMIT,
            True,
            True,
        ),
        (_http_error(429), FailureCategory.UNKNOWN, False, False),
    ],
)
def test_failure_classification_matrix(
    error: BaseException, category: FailureCategory, retry: bool, failover: bool
) -> None:
    info = classify(error)
    assert info.category is category
    assert info.retry_eligible is retry
    assert info.failover_eligible is failover


def test_adapter_boundary_preserves_structured_failure_fields() -> None:
    raw = _http_error(
        429, code="rate_limit_exceeded", error_type="rate_limit_error", retry_after="7"
    )
    backend_error = BackendErrorBuilder.build_http_error(
        provider="provider",
        endpoint="https://provider.invalid/chat",
        error=raw,
        response=raw.response,
        model="model",
        messages=[LLMMessage(role=Role.user, content="hi")],
        temperature=0.2,
        has_tools=False,
        tool_choice=None,
    )
    assert backend_error.parsed_error != json.dumps({"code": "rate_limit_exceeded"})
    assert backend_error.failure_info.error_code == "rate_limit_exceeded"
    assert backend_error.failure_info.error_type == "rate_limit_error"
    assert backend_error.failure_info.retry_after == 7


def test_mistral_sdk_retry_after_controls_shared_retry_delay() -> None:
    class SDKLikeError(Exception):
        def __init__(self) -> None:
            self.raw_response = httpx.Response(
                429,
                request=httpx.Request("POST", "https://provider.invalid/chat"),
                headers={"Retry-After": "7"},
                json={"error": {"code": "rate_limit_exceeded"}},
            )
            super().__init__("rate limited")

    assert _next_delay(SDKLikeError(), 0, 0.5, 2.0, 60.0) == 7.0


@pytest.mark.asyncio
async def test_shared_budget_prevents_retry_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [10.0]
    monkeypatch.setattr("chartreux.core.llm.failures.time.monotonic", lambda: now[0])
    budget = RequestRetryBudget(0.25)
    calls = 0

    @async_retry(tries=None, delay_seconds=0.5, budget=budget)
    async def call() -> None:
        nonlocal calls
        calls += 1
        raise _http_error(503)

    with pytest.raises(httpx.HTTPStatusError):
        await call()
    assert calls == 1
    assert budget.remaining == pytest.approx(0.25)
