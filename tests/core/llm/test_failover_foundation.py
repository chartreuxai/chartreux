from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import json

import httpx
import pytest

from chartreux.core.llm.exceptions import BackendErrorBuilder
from chartreux.core.llm.failures import FailureCategory, RequestRetryBudget, classify
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.utils.retry import (
    RetryCategory,
    RetryReason,
    _next_delay,
    async_generator_retry,
    async_retry,
)


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
        (_http_error(429), FailureCategory.RATE_LIMIT, True, True),
        (
            _http_error(429, code="unrecognized_gateway_code"),
            FailureCategory.RATE_LIMIT,
            True,
            True,
        ),
        (
            _http_error(429, code="insufficient_funds"),
            FailureCategory.BILLING,
            False,
            False,
        ),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("stage", ["attempt", "observer", "sleep", "retry-after"])
async def test_shared_budget_stops_before_another_attempt(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, stage: str
) -> None:
    now = [0.0]
    monkeypatch.setattr("chartreux.core.llm.failures.time.monotonic", lambda: now[0])
    budget = RequestRetryBudget(1.0)
    calls = 0
    sleeps: list[float] = []
    notices: list[RetryReason] = []
    error = _http_error(503, retry_after="7" if stage == "retry-after" else None)

    async def attempt() -> None:
        nonlocal calls
        calls += 1
        if stage == "attempt":
            now[0] = 1.0
        raise error

    async def observe(reason: RetryReason) -> None:
        notices.append(reason)
        if stage == "observer":
            now[0] = 1.0

    async def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] = 1.0

    monkeypatch.setattr("chartreux.core.utils.retry.asyncio.sleep", sleep)
    if streaming:

        async def stream() -> AsyncGenerator[None]:
            await attempt()
            yield None

        retried = async_generator_retry(tries=None, budget=budget, on_retry=observe)(
            stream
        )
        with pytest.raises(httpx.HTTPStatusError) as caught:
            _ = [item async for item in retried()]
    else:
        call = async_retry(tries=None, budget=budget, on_retry=observe)(attempt)
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await call()

    assert caught.value is error
    assert calls == 1
    assert len(notices) == (1 if stage in {"observer", "sleep"} else 0)
    assert sleeps == ([0.5] if stage == "sleep" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "code",
    [
        "invalid_api_key",
        "billing_hard_limit",
        "insufficient_quota",
        "invalid_request_error",
        "context_length_exceeded",
    ],
)
async def test_permanent_structured_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, code: str
) -> None:
    calls = 0
    error = _http_error(500, code=code)

    async def attempt() -> None:
        nonlocal calls
        calls += 1
        raise error

    async def no_sleep(_delay: float) -> None:
        pytest.fail("permanent failures must not back off")

    monkeypatch.setattr("chartreux.core.utils.retry.asyncio.sleep", no_sleep)
    if streaming:

        async def stream() -> AsyncGenerator[None]:
            await attempt()
            yield None

        retried = async_generator_retry(tries=3)(stream)
        with pytest.raises(httpx.HTTPStatusError) as caught:
            _ = [item async for item in retried()]
    else:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await async_retry(tries=3)(attempt)()
    assert caught.value is error
    assert calls == 1


def test_plain_text_429_body_classifies_rate_limit() -> None:
    """A proxy answers a 429 with a plain-text body: the status alone is the
    provider's rate-limit signal, so the category is RATE_LIMIT and the
    failure is retry/failover eligible.
    """
    request = httpx.Request("POST", "https://provider.invalid/chat")
    response = httpx.Response(429, request=request, text="Too Many Requests")
    info = classify(
        httpx.HTTPStatusError("provider error", request=request, response=response)
    )
    assert info.category is FailureCategory.RATE_LIMIT
    assert info.retry_eligible is True
    assert info.failover_eligible is True


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_bare_429_retries_within_a_sufficient_budget(
    monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    """A bare 429 behaves exactly like a coded one: one deployment with a
    sufficient budget retries in-call and honors Retry-After.
    """
    attempts = 0
    sleeps: list[float] = []
    notices: list[RetryReason] = []
    error = _http_error(429, retry_after="7")

    async def attempt() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error

    async def observe(reason: RetryReason) -> None:
        notices.append(reason)

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("chartreux.core.utils.retry.asyncio.sleep", sleep)
    if streaming:

        async def stream() -> AsyncGenerator[None]:
            await attempt()
            yield None

        retried = async_generator_retry(
            tries=None, budget=RequestRetryBudget(300.0), on_retry=observe
        )(stream)
        assert [item async for item in retried()] == [None]
    else:
        call = async_retry(
            tries=None, budget=RequestRetryBudget(300.0), on_retry=observe
        )(attempt)
        await call()

    assert attempts == 2
    assert sleeps == [7.0]
    assert [notice.category for notice in notices] == [RetryCategory.RATE_LIMITED]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["insufficient_quota", "insufficient_funds"])
async def test_protected_429_never_retries(
    monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    """Quota and billing 429s keep their protected categories: the bare-429
    flip never grants them retry authority.
    """
    calls = 0
    error = _http_error(429, code=code)

    async def attempt() -> None:
        nonlocal calls
        calls += 1
        raise error

    async def no_sleep(_delay: float) -> None:
        pytest.fail("protected 429s must not back off")

    monkeypatch.setattr("chartreux.core.utils.retry.asyncio.sleep", no_sleep)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        await async_retry(tries=3)(attempt)()
    assert caught.value is error
    assert calls == 1
