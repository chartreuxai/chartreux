from __future__ import annotations

import asyncio
from dataclasses import asdict
import logging
import subprocess
import sys
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.llm import provider_smoke as smoke
from chartreux.core.llm_models import (
    FunctionCall,
    LLMChunk,
    LLMMessage,
    Role,
    StopInfo,
    ToolCall,
)
from chartreux.utils.api_keys import ApiKeyOrigin, ApiKeySource


def response(**kwargs) -> LLMChunk:
    return LLMChunk(message=LLMMessage(role=Role.assistant, **kwargs))


def tool_response(name="chartreux_smoke_echo", arguments='{"value":"smoke-ok"}'):
    return response(
        tool_calls=[ToolCall(function=FunctionCall(name=name, arguments=arguments))]
    )


@pytest.mark.parametrize(
    ("chunk", "reason"),
    [
        (response(content="no"), "missing_tool_call"),
        (tool_response(name="other"), "wrong_tool_name_or_count"),
        (tool_response(arguments="{secret"), "malformed_tool_arguments"),
        (tool_response(arguments='{"value":"other"}'), "unexpected_tool_arguments"),
        (
            tool_response(arguments='{"value":"smoke-ok","extra":true}'),
            "unexpected_tool_arguments",
        ),
        (tool_response(arguments="null"), "unexpected_tool_arguments"),
    ],
)
def test_tool_failure_verdicts(chunk, reason):
    assert smoke._verdict("tool", chunk) == smoke.CapabilityResult("fail", reason)


@pytest.mark.parametrize("capability", ["tool", "thinking", "image"])
@pytest.mark.parametrize(
    "reason", ["length", "max_tokens", "max_output_tokens", "incomplete"]
)
def test_truncation_is_failure(capability, reason):
    chunk = tool_response().model_copy(update={"stop": StopInfo(reason=reason)})
    assert smoke._verdict(capability, chunk) == smoke.CapabilityResult(
        "fail", "output_truncated"
    )


def test_success_and_unobservable_verdicts():
    assert smoke._verdict("tool", tool_response()).status == "pass"
    assert smoke._verdict("image", response(content=" Red ")).status == "pass"
    assert (
        smoke._verdict("image", response(content="blue")).reason == "wrong_image_answer"
    )
    assert (
        smoke._verdict("thinking", response(reasoning_content="calculation")).status
        == "pass"
    )
    for chunk in [
        response(content="323"),
        response(reasoning_payloads=[{"encrypted": "opaque"}]),
    ]:
        assert smoke._verdict("thinking", chunk).status == "unverified"


@pytest.fixture
def deployment():
    return (
        ModelConfig(
            name="test-model", alias="test-alias", provider="test", supports_images=True
        ),
        ProviderConfig(
            name="test", api_base="https://example.test/v1", api_key_env_var="TEST_KEY"
        ),
    )


@pytest.fixture
def mocked_backend():
    backend = MagicMock()
    backend.complete = AsyncMock(
        side_effect=[tool_response(), response(content="323"), response(content="red")]
    )
    backend.__aexit__ = AsyncMock()
    credential = (
        "secret-credential",
        ApiKeyOrigin(ApiKeySource.ENVIRONMENT, "TEST_KEY"),
    )
    with (
        patch.object(smoke, "create_backend", return_value=backend) as factory,
        patch.object(
            smoke, "resolve_api_key_with_origin", return_value=credential
        ) as resolve,
    ):
        yield backend, factory, resolve


@pytest.mark.asyncio
async def test_isolated_requests_budget_limits_identity_and_closure(
    deployment, mocked_backend
):
    model, provider = deployment
    backend, factory, resolve = mocked_backend
    result = await smoke.probe_provider_smoke(
        model=model, provider=provider, max_tokens=111
    )
    assert (result.provider, result.model, result.alias) == (
        provider.name,
        model.name,
        model.alias,
    )
    assert (result.tool.status, result.thinking.status, result.image.status) == (
        "pass",
        "unverified",
        "pass",
    )
    resolve.assert_called_once_with("TEST_KEY")
    kwargs = factory.call_args.kwargs
    assert kwargs["retry_budget"].remaining == 0
    assert kwargs["retry_budget"].exhausted
    assert kwargs["resolved_credential"] == resolve.return_value
    assert 0 < kwargs["timeout"] <= 30
    calls = backend.complete.call_args_list
    assert len(calls) == 3
    assert all(len(call.kwargs["messages"]) == 1 for call in calls)
    assert all(call.kwargs["max_tokens"] == 111 for call in calls)
    assert calls[0].kwargs["tools"][0] == calls[0].kwargs["tool_choice"]
    assert calls[1].kwargs["model"].thinking != "off"
    assert calls[2].kwargs["messages"][0].images[0].source.kind == "inline"
    assert model.thinking == "off"
    backend.__aexit__.assert_awaited_once()
    assert "secret" not in str(asdict(result))


@pytest.mark.asyncio
async def test_skip_rules_send_only_tool(deployment, mocked_backend):
    model, provider = deployment
    model = model.model_copy(
        update={"supports_images": False, "supported_thinking_levels": ["off"]}
    )
    backend, _, _ = mocked_backend
    result = await smoke.probe_provider_smoke(model=model, provider=provider)
    assert result.thinking.status == result.image.status == "unsupported"
    assert backend.complete.await_count == 1


@pytest.mark.asyncio
async def test_deadline_closes_and_preserves_skip(deployment, mocked_backend):
    model, provider = deployment
    backend, _, _ = mocked_backend
    backend.complete.side_effect = lambda **kwargs: None

    async def blocked(**kwargs):
        await asyncio.sleep(1)
        return response()

    backend.complete.side_effect = blocked
    result = await smoke.probe_provider_smoke(
        model=model, provider=provider, deadline_seconds=0.01
    )
    assert (
        result.tool.reason
        == result.thinking.reason
        == result.image.reason
        == "overall_deadline_exceeded"
    )
    assert backend.complete.await_count == 1
    backend.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolution_elapsed_counts_before_async_deadline(
    deployment, mocked_backend
):
    model, provider = deployment
    _, factory, resolve = mocked_backend
    with patch.object(smoke.time, "monotonic", side_effect=[10.0, 12.0]):
        result = await smoke.probe_provider_smoke(
            model=model, provider=provider, deadline_seconds=1
        )
    assert result.tool.reason == "overall_deadline_exceeded"
    resolve.assert_called_once()
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_cancellation_propagates_and_closes(deployment, mocked_backend):
    model, provider = deployment
    backend, _, _ = mocked_backend
    started = asyncio.Event()

    async def blocked(**kwargs):
        started.set()
        await asyncio.Event().wait()

    backend.complete.side_effect = blocked
    task = asyncio.create_task(
        smoke.probe_provider_smoke(model=model, provider=provider)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    backend.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["resolve", "factory", "request", "cleanup"])
async def test_error_redaction(deployment, mocked_backend, stage):
    model, provider = deployment
    backend, factory, resolve = mocked_backend
    error = RuntimeError("secret-credential raw-provider-body")
    if stage == "resolve":
        resolve.side_effect = error
    elif stage == "factory":
        factory.side_effect = error
    elif stage == "request":
        backend.complete.side_effect = error
    else:
        backend.__aexit__.side_effect = error
    result = await smoke.probe_provider_smoke(model=model, provider=provider)
    assert result.tool.status == ("pass" if stage == "cleanup" else "fail")
    if stage == "cleanup":
        assert result.image.status == "pass"
        assert result.thinking.reason == "backend_cleanup_failed"
    assert "secret" not in str(asdict(result))
    assert "raw-provider-body" not in str(asdict(result))
    if stage in {"request", "cleanup"}:
        backend.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"deadline_seconds": 0},
        {"deadline_seconds": float("inf")},
        {"deadline_seconds": float("nan")},
        {"max_tokens": 0},
        {"max_tokens": True},
    ],
)
async def test_invalid_bounds_do_not_resolve(deployment, mocked_backend, options):
    model, provider = deployment
    _, factory, resolve = mocked_backend
    with pytest.raises(ValueError):
        await smoke.probe_provider_smoke(model=model, provider=provider, **options)
    resolve.assert_not_called()
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_wrong_deployment_rejected(deployment, mocked_backend):
    model, provider = deployment
    with pytest.raises(ValueError):
        await smoke.probe_provider_smoke(
            model=model.model_copy(update={"provider": "other"}), provider=provider
        )


@pytest.mark.asyncio
async def test_mistral_low_none_is_not_thinking(deployment, mocked_backend):
    model, provider = deployment
    provider = provider.model_copy(update={"backend": "mistral"})
    backend, _, _ = mocked_backend
    await smoke.probe_provider_smoke(model=model, provider=provider)
    assert backend.complete.call_args_list[1].kwargs["model"].thinking == "medium"


@pytest.mark.asyncio
@pytest.mark.parametrize("default", ["off", "medium"])
async def test_low_only_mistral_baseline(deployment, mocked_backend, default):
    model, provider = deployment
    model = model.model_copy(
        update={"thinking": default, "supported_thinking_levels": ["low"]}
    )
    provider = provider.model_copy(update={"backend": "mistral"})
    backend, _, _ = mocked_backend
    replies = iter([tool_response(), response(content="red")])

    async def complete(**kwargs):
        # Match the adapter's rejection of levels outside deployment support.
        assert kwargs["model"].thinking == "low"
        return next(replies)

    backend.complete.side_effect = complete
    result = await smoke.probe_provider_smoke(model=model, provider=provider)
    assert result.tool.status == result.image.status == "pass"
    assert result.thinking.status == "unsupported"
    assert backend.complete.await_count == 2


@pytest.mark.asyncio
async def test_narrowed_thinking_and_tls_policy(deployment, mocked_backend):
    model, provider = deployment
    model = model.model_copy(update={"supported_thinking_levels": ["high"]})
    backend, factory, _ = mocked_backend
    await smoke.probe_provider_smoke(
        model=model, provider=provider, enable_system_trust_store=True
    )
    assert factory.call_args.kwargs["enable_system_trust_store"] is True
    assert all(
        call.kwargs["model"].thinking == "high"
        for call in backend.complete.call_args_list
    )


@pytest.mark.asyncio
async def test_missing_credential_has_specific_code(deployment, mocked_backend):
    model, provider = deployment
    _, factory, resolve = mocked_backend
    resolve.return_value = None
    result = await smoke.probe_provider_smoke(model=model, provider=provider)
    assert result.tool == smoke.CapabilityResult("fail", "credential_missing")
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_non_backend_failure_is_not_initialization(deployment, mocked_backend):
    model, provider = deployment
    backend, _, _ = mocked_backend
    with patch.object(smoke, "_tool_fixture", side_effect=RuntimeError("secret")):
        result = await smoke.probe_provider_smoke(model=model, provider=provider)
    assert result.tool.reason == "probe_failed"
    backend.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_debug_logs_are_suppressed_and_restored(
    deployment, mocked_backend, monkeypatch, caplog
):
    model, provider = deployment
    backend, factory, resolve = mocked_backend
    monkeypatch.setenv("MISTRAL_DEBUG", "1")
    caplog.set_level(logging.DEBUG)
    previous = logging.root.manager.disable

    def disclose(*args, **kwargs):
        logging.getLogger("mistralai.client").critical(
            "secret-credential raw-provider-body"
        )

    factory.side_effect = lambda **kwargs: (disclose(), backend)[1]
    resolve.side_effect = lambda *args: (disclose(), resolve.return_value)[1]

    async def complete(**kwargs):
        disclose()
        return tool_response()

    backend.complete.side_effect = complete
    backend.__aexit__.side_effect = disclose
    await smoke.probe_provider_smoke(model=model, provider=provider)
    assert logging.root.manager.disable == previous
    assert "secret-credential" not in caplog.text
    assert "raw-provider-body" not in caplog.text
    logging.getLogger("mistralai.client").warning("logging restored")
    assert "logging restored" in caplog.text


@pytest.mark.parametrize("outcome", ["complete", "exception", "cancel"])
def test_sdk_root_logging_restored_in_fresh_process(outcome, monkeypatch):
    monkeypatch.setenv("MISTRAL_DEBUG", "1")
    script = textwrap.dedent(
        """
        import asyncio
        import logging
        import sys
        from unittest.mock import patch

        from mistralai.client import Mistral
        from chartreux.core.config import ModelConfig, ProviderConfig
        from chartreux.core.llm import provider_smoke as smoke

        outcome = sys.argv[1]
        assert logging.root.handlers == []
        logging.root.setLevel(logging.ERROR)
        logging.disable(logging.INFO)
        before = (logging.root.manager.disable, logging.root.level,
                  logging.root.handlers.copy())

        async def main():
            started = asyncio.Event()

            async def probe(**kwargs):
                # Initialize the real SDK without sending any requests.
                with Mistral(api_key="synthetic-smoke-key"):
                    assert logging.root.level == logging.DEBUG
                    assert logging.root.handlers
                    assert logging.root.manager.disable == logging.CRITICAL
                    started.set()
                    if outcome == "exception":
                        raise RuntimeError("probe failed")
                    if outcome == "cancel":
                        await asyncio.Event().wait()

            with patch.object(smoke, "_probe_provider_smoke", side_effect=probe):
                task = asyncio.create_task(smoke.probe_provider_smoke(
                    model=ModelConfig(name="test", provider="test"),
                    provider=ProviderConfig(name="test", api_base="https://example.test"),
                ))
                await started.wait()
                if outcome == "cancel":
                    task.cancel()
                try:
                    await task
                except (RuntimeError, asyncio.CancelledError):
                    assert outcome != "complete"
                else:
                    assert outcome == "complete"

        asyncio.run(main())
        assert (logging.root.manager.disable, logging.root.level,
                logging.root.handlers) == before
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, outcome],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_overlapping_probes_restore_logging_only_after_last_exit(deployment):
    model, provider = deployment
    before = (
        logging.root.manager.disable,
        logging.root.level,
        logging.root.handlers.copy(),
    )
    started = [asyncio.Event(), asyncio.Event()]
    finish = [asyncio.Event(), asyncio.Event()]
    calls = 0

    async def probe(**kwargs):
        nonlocal calls
        index = calls
        calls += 1
        logging.root.setLevel(logging.DEBUG)
        started[index].set()
        await finish[index].wait()

    tasks = []
    try:
        with patch.object(smoke, "_probe_provider_smoke", side_effect=probe):
            for index in range(2):
                tasks.append(
                    asyncio.create_task(
                        smoke.probe_provider_smoke(model=model, provider=provider)
                    )
                )
                await started[index].wait()
            finish[0].set()
            await tasks[0]
            assert logging.root.manager.disable == max(before[0], logging.CRITICAL)
            assert logging.root.level == logging.DEBUG
            finish[1].set()
            await tasks[1]
        assert (
            logging.root.manager.disable,
            logging.root.level,
            logging.root.handlers,
        ) == before
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
