from __future__ import annotations

from collections.abc import Sequence
import cProfile
from io import StringIO
import pstats
import time

import pytest

from chartreux.core.agent_loop.llm_gateway import messages_for_backend
from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.llm.backend.anthropic import AnthropicAdapter
from chartreux.core.llm.backend.base import APIAdapter, PreparedRequest
from chartreux.core.llm.backend.generic import OpenAIAdapter
from chartreux.core.llm_models import AvailableFunction, AvailableTool, LLMMessage
from tests.perf._metrics import machine_context, percentiles, record
from tests.perf._synthetic import synthetic_messages

_TOKEN_CASES = ((100_000, 100), (300_000, 300), (500_000, 500))
_REPETITIONS = 5
_MODEL = ModelConfig(
    name="synthetic-model",
    provider="synthetic-provider",
    alias="synthetic-model",
    supports_images=False,
)
_TOOLS = [
    AvailableTool(
        function=AvailableFunction(
            name="synthetic_tool",
            description="Synthetic request-preparation tool.",
            parameters={"type": "object", "properties": {}},
        )
    )
]
_MACHINE_CONTEXT_RECORDED = False


def _adapter_and_provider(adapter_name: str) -> tuple[APIAdapter, ProviderConfig]:
    if adapter_name == "generic":
        return (
            OpenAIAdapter(),
            ProviderConfig(
                name="synthetic-generic",
                api_base="https://example.invalid/v1",
                api_style="openai",
            ),
        )
    if adapter_name == "anthropic":
        return (
            AnthropicAdapter(),
            ProviderConfig(
                name="synthetic-anthropic",
                api_base="https://example.invalid",
                api_style="anthropic",
            ),
        )
    raise ValueError(f"unknown adapter {adapter_name!r}")


def _prepare_request(
    adapter: APIAdapter, provider: ProviderConfig, messages: Sequence[LLMMessage]
) -> PreparedRequest:
    return adapter.prepare_request(
        model_name=_MODEL.name,
        messages=messages,
        temperature=_MODEL.temperature,
        tools=_TOOLS,
        max_tokens=2048,
        tool_choice=None,
        enable_streaming=False,
        provider=provider,
        thinking=_MODEL.thinking,
    )


def _run_pipeline(
    messages: Sequence[LLMMessage], adapter: APIAdapter, provider: ProviderConfig
) -> PreparedRequest:
    backend_messages = messages_for_backend(messages, _MODEL)
    return _prepare_request(adapter, provider, backend_messages)


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _assert_tool_results_adjacent(messages: Sequence[LLMMessage]) -> None:
    tool_calls = 0
    for index, message in enumerate(messages):
        if not message.tool_calls:
            continue
        tool_calls += len(message.tool_calls)
        assert index + 1 < len(messages)
        assert messages[index + 1].role == "tool"
        assert messages[index + 1].tool_call_id == message.tool_calls[0].id
    assert tool_calls > 0


@pytest.mark.perf
@pytest.mark.timeout(300)
@pytest.mark.parametrize("adapter_name", ("generic", "anthropic"))
@pytest.mark.parametrize(
    ("approx_tokens", "n_messages"), _TOKEN_CASES, ids=("100k", "300k", "500k")
)
def test_request_preparation(
    adapter_name: str, approx_tokens: int, n_messages: int
) -> None:
    global _MACHINE_CONTEXT_RECORDED

    if not _MACHINE_CONTEXT_RECORDED:
        record("request_prep.machine", machine_context())
        _MACHINE_CONTEXT_RECORDED = True

    adapter, provider = _adapter_and_provider(adapter_name)
    messages = synthetic_messages(approx_tokens, n_messages)
    assert len(messages) == n_messages
    assert sum(len(message.content or "") for message in messages) == approx_tokens * 4

    # One complete warmup exercises projection and request preparation before sampling.
    warmup_projection = messages_for_backend(messages, _MODEL)
    assert warmup_projection != messages
    _assert_tool_results_adjacent(warmup_projection)
    warmup_request = _prepare_request(adapter, provider, warmup_projection)

    projection_samples: list[float] = []
    prepare_samples: list[float] = []
    end_to_end_samples: list[float] = []
    stage_sum_samples: list[float] = []
    last_request = warmup_request

    for _ in range(_REPETITIONS):
        # The outer timer measures this unsplit a+b pipeline run; inner timers
        # isolate its two production stages.
        pipeline_started = time.perf_counter()
        started = time.perf_counter()
        backend_messages = messages_for_backend(messages, _MODEL)
        projection_samples.append(_ms_since(started))

        started = time.perf_counter()
        last_request = _prepare_request(adapter, provider, backend_messages)
        prepare_samples.append(_ms_since(started))
        stage_sum_samples.append(projection_samples[-1] + prepare_samples[-1])
        end_to_end_samples.append(_ms_since(pipeline_started))

    projection_summary = percentiles(projection_samples)
    prepare_summary = percentiles(prepare_samples)
    end_to_end_summary = percentiles(end_to_end_samples)
    stage_sum_summary = percentiles(stage_sum_samples)
    split_delta_percent = (
        abs(stage_sum_summary["p50"] - end_to_end_summary["p50"])
        / end_to_end_summary["p50"]
        * 100
    )
    split_mean_delta_percent = (
        abs(stage_sum_summary["mean"] - end_to_end_summary["mean"])
        / end_to_end_summary["mean"]
        * 100
    )
    assert split_delta_percent <= 10, (
        f"p50 split a+b differs from p50 end-to-end by "
        f"{split_delta_percent:.2f}% for {adapter_name} at {approx_tokens} tokens; "
        f"stage={stage_sum_summary}, end_to_end={end_to_end_summary}"
    )

    metrics: dict[str, object] = {
        "approx_tokens_chars_div_4": approx_tokens,
        "n_messages": n_messages,
        "adapter": adapter_name,
        "warmups": 1,
        "repetitions": _REPETITIONS,
        "request_body_bytes": len(last_request.body),
        "stages_ms": {
            "messages_for_backend": {
                "min": projection_summary["min"],
                "p50": projection_summary["p50"],
            },
            "prepare_request_including_conversion_and_serialization": {
                "min": prepare_summary["min"],
                "p50": prepare_summary["p50"],
            },
        },
        "split_a_plus_b_mean_ms": stage_sum_summary["mean"],
        "split_a_plus_b_p50_ms": stage_sum_summary["p50"],
        "end_to_end_ms": {
            "min": end_to_end_summary["min"],
            "mean": end_to_end_summary["mean"],
            "p50": end_to_end_summary["p50"],
        },
        "split_p50_delta_percent": round(split_delta_percent, 3),
        "split_mean_delta_percent": round(split_mean_delta_percent, 3),
        "split_a_plus_b_within_10_percent": True,
    }
    record(f"request_prep.{adapter_name}.{approx_tokens}", metrics)
    print(
        f"request-prep split check: {adapter_name} {approx_tokens} tokens; "
        f"split a+b p50={stage_sum_summary['p50']:.3f}ms, "
        f"end-to-end p50={end_to_end_summary['p50']:.3f}ms, "
        f"delta={split_delta_percent:.2f}% (<=10%: PASS); "
        f"mean delta={split_mean_delta_percent:.2f}%"
    )

    if adapter_name == "generic" and approx_tokens == 500_000:
        profiler = cProfile.Profile()
        profiler.enable()
        _run_pipeline(messages, adapter, provider)
        profiler.disable()
        output = StringIO()
        pstats.Stats(profiler, stream=output).sort_stats(
            pstats.SortKey.CUMULATIVE
        ).print_stats(20)
        print("CPROFILE TOP 20 — 500k generic request-prep pipeline")
        print(output.getvalue())
