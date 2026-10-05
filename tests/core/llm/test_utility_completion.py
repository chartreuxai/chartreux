from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from chartreux.core.llm import utility_completion
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm.utility_completion import (
    is_fast_utility_model,
    run_utility_completion,
    select_utility_model,
)
from chartreux.core.llm_models import LLMChunk, LLMMessage, LLMUsage, Role
from chartreux.core.usage import UsageAttribution, UsagePurpose, UsageRecord
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend


def test_utility_completion_uses_shipped_active_model() -> None:
    model, provider = select_utility_model(build_test_vibe_config())
    assert model.alias == "glm-5-3"
    assert model.name == "zai-glm-5-3"
    assert provider.name == "mistral"


def test_shipped_default_is_not_fast_utility_model() -> None:
    assert not is_fast_utility_model(build_test_vibe_config())


ATTRIBUTION = UsageAttribution(
    root_session_id="root",
    session_id="session",
    agent_role="root",
    model="stale",
    provider="stale",
    wire_name="stale",
    project_key="project",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["empty", "missing-key", "config", "preparation"])
async def test_pre_attempt_paths_have_no_records(
    monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    config = build_test_vibe_config()
    records: list[UsageRecord] = []
    backend = FakeBackend()
    factory = Mock(return_value=backend)
    monkeypatch.setattr(utility_completion, "create_backend", factory)
    monkeypatch.setattr(utility_completion, "resolve_api_key", lambda _: None)

    async def sink(record: UsageRecord) -> None:
        records.append(record)

    if scenario == "config":
        monkeypatch.setattr(
            utility_completion, "select_utility_model", Mock(side_effect=ValueError)
        )
    if scenario == "preparation":

        async def fail(**_kwargs: Any) -> LLMChunk:
            raise ValueError("preparation")

        monkeypatch.setattr(backend, "complete", fail)
    kwargs: dict[str, Any] = dict(
        config=config,
        system_prompt="system",
        user_content=" " if scenario == "empty" else "input",
        max_tokens=10,
        request_timeout_seconds=1,
        retry_budget_seconds=2,
        skip_if_no_key=scenario == "missing-key",
        accounting_sink=sink,
        usage_attribution=ATTRIBUTION,
        purpose=UsagePurpose.WORKTREE_NAMING,
    )
    if scenario in {"config", "preparation"}:
        with pytest.raises(ValueError):
            await run_utility_completion(**kwargs)
    else:
        assert await run_utility_completion(**kwargs) is None
    assert not records
    assert factory.call_count == int(scenario == "preparation")


@pytest.mark.asyncio
@pytest.mark.parametrize("sink_fails", [False, True])
async def test_utility_retry_fallback_keeps_budgets_and_one_record_per_candidate(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, sink_fails: bool
) -> None:
    config = build_test_vibe_config()
    model, provider = select_utility_model(config)
    records: list[UsageRecord] = []
    calls: list[dict[str, Any]] = []
    backend = FakeBackend()
    factory = Mock(return_value=backend)
    monkeypatch.setattr(utility_completion, "create_backend", factory)

    async def sink(record: UsageRecord) -> None:
        records.append(record)
        if sink_fails:
            raise OSError("private accounting error")

    async def complete(**kwargs: Any) -> LLMChunk:
        calls.append(kwargs)
        # Internal transport retries still belong to this logical utility call.
        notify_request_started()
        notify_request_started()
        if len(calls) == 1:
            raise TimeoutError("candidate failed")
        return LLMChunk(
            message=LLMMessage(role=Role.assistant, content="fallback title"),
            usage=LLMUsage.from_reported(completion_tokens=3),
        )

    monkeypatch.setattr(backend, "complete", complete)
    kwargs: dict[str, Any] = dict(
        config=config,
        system_prompt="system",
        user_content="input",
        max_tokens=10,
        request_timeout_seconds=1.5,
        retry_budget_seconds=2.5,
        accounting_sink=sink,
        usage_attribution=ATTRIBUTION,
        purpose=UsagePurpose.WORKTREE_NAMING,
    )
    with pytest.raises(TimeoutError):
        await run_utility_completion(**kwargs)
    assert await run_utility_completion(**kwargs) == "fallback title"
    assert len(calls) == len(records) == 2
    assert len({r.record_id for r in records}) == 2
    assert all(r.purpose == UsagePurpose.WORKTREE_NAMING for r in records)
    assert all((r.model, r.provider) == (model.alias, provider.name) for r in records)
    assert records[0].input_tokens is records[0].output_tokens is None
    assert records[1].output_tokens == 3
    assert all(call["temperature"] == 0.0 for call in calls)
    assert all(call["max_tokens"] == 10 for call in calls)
    factory.assert_called_with(
        provider=provider, timeout=1.5, retry_max_elapsed_time=2.5
    )
    if sink_fails:
        assert "coverage is degraded" in caplog.text
        assert "private accounting error" not in caplog.text
    # Accounting resets its task-local request observer after each candidate.
    from chartreux.core.llm.backend.generic import request_start_observer

    assert request_start_observer.get() is None
