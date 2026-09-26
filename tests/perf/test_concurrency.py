from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import threading
import time
from typing import TypedDict

import pytest

from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.core.subagents import AgentEviction
from tests.perf._harness import (
    FanOutHarness,
    GatedSequenceBackend,
    create_fan_out_harness,
    launch_fan_out,
)
from tests.perf._metrics import machine_context, percentiles, record
from tests.stubs.fake_backend import FakeBackend

_HEARTBEAT_INTERVAL_SECONDS = 0.005

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="performance scenarios require Linux /proc readers"
)


class _IterationMetrics(TypedDict):
    launch_latency_ms: list[float]
    loop_lag_ms: list[float]
    agents_update_serialization_proxy_ms: list[float]
    agents_update_serialization_proxy_bytes: list[int]
    agents_update_count: int
    rss_current_before_mb: float
    rss_current_after_mb: float
    rss_current_delta_mb: float
    rss_process_peak_mb: float
    fd_count_before: int
    fd_count_after_release: int
    fd_delta: int
    thread_count_before: int
    thread_count_peak: int
    asyncio_task_count_at_release: int
    asyncio_task_count_peak: int
    teardown_ms: float
    leaked_asyncio_tasks: int


class _IterationSummary(TypedDict):
    launch_latency_ms: dict[str, float]
    loop_lag_ms: dict[str, float]
    agents_update_serialization_proxy_ms: dict[str, float]
    agents_update_serialization_proxy_bytes: dict[str, float]
    agents_update_count: int
    rss_current_before_mb: float
    rss_current_after_mb: float
    rss_current_delta_mb: float
    rss_process_peak_mb: float
    fd_count_before: int
    fd_count_after_release: int
    fd_delta: int
    thread_count_before: int
    thread_count_peak: int
    asyncio_task_count_at_release: int
    asyncio_task_count_peak: int
    teardown_ms: float
    leaked_asyncio_tasks: int


class _ConcurrencySummary(TypedDict):
    n: int
    iterations: int
    launch_latency_ms: dict[str, float]
    launch_latency_iteration_p50_median_ms: float
    loop_lag_ms: dict[str, float]
    agents_update_serialization_proxy_ms: dict[str, float]
    agents_update_serialization_proxy_iteration_p50_median_ms: float
    agents_update_serialization_proxy_bytes: dict[str, float]
    agents_update_count_per_iteration: list[int]
    rss_current_delta_mb_median: float
    rss_current_after_mb_median: float
    rss_process_peak_mb_median: float
    fd_count_before_median: float
    fd_count_after_release_median: float
    fd_delta_median: float
    thread_count_before_median: float
    thread_count_peak_median: float
    asyncio_task_count_at_release_median: float
    asyncio_task_count_peak_median: float
    teardown_ms_median: float
    leaked_asyncio_tasks_per_iteration: list[int]
    machine: dict[str, object]


def _rss_current_mb() -> float:
    resident_pages = int(
        Path("/proc/self/statm").read_text(encoding="utf-8").split()[1]
    )
    page_size = os.sysconf("SC_PAGE_SIZE")
    return resident_pages * page_size / (1024 * 1024)


def _rss_process_peak_mb() -> float:
    """Return the process-lifetime high-water RSS, not an iteration delta."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return max_rss / divisor


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


async def _heartbeat(
    stop: asyncio.Event,
    loop_lag_ms: list[float],
    task_counts: list[int],
    thread_counts: list[int],
) -> None:
    loop = asyncio.get_running_loop()
    tick = 0
    while not stop.is_set():
        expected_wake = loop.time() + _HEARTBEAT_INTERVAL_SECONDS
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        now = loop.time()
        loop_lag_ms.append(max(0.0, now - expected_wake) * 1000)
        tick += 1
        # Task/thread enumeration still adds observer overhead, but only every
        # 40th 5ms tick (about 5 samples per second) rather than on every tick.
        if tick % 40 == 0:
            task_counts.append(len(asyncio.all_tasks()))
            thread_counts.append(threading.active_count())


async def _await_update_count(harness: FanOutHarness, expected: int) -> None:
    deadline = asyncio.get_running_loop().time() + 10
    while len(harness.agents_update_sizes) < expected:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"saw {len(harness.agents_update_sizes)} AgentsUpdate emissions; "
                f"expected at least {expected}"
            )
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)


async def _iteration(
    agent_count: int, monkeypatch: pytest.MonkeyPatch
) -> _IterationMetrics:
    baseline_tasks = set(asyncio.all_tasks())
    role_members = [f"perf-model-{index}" for index in range(agent_count)]
    backend = GatedSequenceBackend()
    for _ in range(agent_count):
        backend.add_gate()

    def create_backend(**_kwargs: object) -> FakeBackend:
        return backend

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    harness: FanOutHarness | None = None
    heartbeat: asyncio.Task[None] | None = None
    heartbeat_results: tuple[None | BaseException, ...] = ()
    stop_heartbeat = asyncio.Event()
    loop_lag_ms: list[float] = []
    sampled_task_counts: list[int] = []
    sampled_thread_counts: list[int] = []
    emit_durations_ms: list[float] = []
    payload_sizes_bytes: list[int] = []
    closed = False
    try:
        harness = await create_fan_out_harness(role_members)
        registry = harness.registry

        # Pin the measurement to SessionRuntimeRegistry._emit_agents_update in
        # chartreux/app_server/_sessions.py:2757. Attribute access intentionally has
        # no fallback: removing or renaming this emission point fails the scenario.
        # The harness callback serializes summaries into an in-memory list, so these
        # serialization-proxy metrics do not measure production AgentsUpdateParams
        # construction or the services.notify path.
        original_emit = registry._emit_agents_update
        original_notify = registry._notify_agents
        assert original_notify is not None

        async def measure_notify(
            agents: list[AgentSummaryModel], evictions: list[AgentEviction]
        ) -> None:
            payload = {
                "agents": [
                    agent.model_dump(mode="json", exclude_none=True) for agent in agents
                ],
                "evictions": [asdict(eviction) for eviction in evictions],
            }
            payload_sizes_bytes.append(
                len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
            )
            await original_notify(agents, evictions)

        async def measure_emit(evictions: list[AgentEviction] | None = None) -> None:
            started_at = time.perf_counter()
            try:
                await original_emit(evictions)
            finally:
                emit_durations_ms.append((time.perf_counter() - started_at) * 1000)

        monkeypatch.setattr(registry, "_notify_agents", measure_notify)
        monkeypatch.setattr(registry, "_emit_agents_update", measure_emit)

        rss_before_mb = _rss_current_mb()
        fds_before = _fd_count()
        threads_before = threading.active_count()
        heartbeat = asyncio.create_task(
            _heartbeat(
                stop_heartbeat, loop_lag_ms, sampled_task_counts, sampled_thread_counts
            ),
            name=f"perf-loop-heartbeat-{agent_count}",
        )

        fan_out_started_at = time.perf_counter()
        result = await launch_fan_out(registry, harness.context, harness.role)
        assert result.members is not None
        assert len(result.members) == agent_count
        assert [member.base_model for member in result.members] == role_members
        assert all(member.status == "running" for member in result.members)

        await asyncio.wait_for(
            asyncio.gather(*(started.wait() for started in backend.started)),
            timeout=120,
        )
        assert all(started.is_set() for started in backend.started), (
            f"only {sum(started.is_set() for started in backend.started)} of "
            f"{agent_count} gates started before release"
        )
        assert all(timestamp is not None for timestamp in backend.started_at)
        launch_latencies_ms = [
            (timestamp - fan_out_started_at) * 1000
            for timestamp in backend.started_at
            if timestamp is not None
        ]
        assert len(launch_latencies_ms) == agent_count
        await _await_update_count(harness, 1)
        assert len(harness.agents_update_sizes) >= 1
        tasks_at_release = len(asyncio.all_tasks())
        for release in backend.releases:
            release.set()

        members: list[tuple[str, str]] = []
        for member in result.members:
            if member.agent_id and member.run_id:
                members.append((member.agent_id, member.run_id))
        assert len(members) == agent_count
        completions = await asyncio.wait_for(
            asyncio.gather(
                *(
                    registry.wait_for_agent(agent_id, run_id, timeout=120)
                    for agent_id, run_id in members
                )
            ),
            timeout=120,
        )
        assert all(completion.completed for completion in completions)
        # Each member publishes both terminal/finalizing and idle updates; the
        # coalesced fan-out launch contributes one additional batch update.
        await _await_update_count(harness, 2 * agent_count + 1)
        agents_update_count = len(harness.agents_update_sizes)

        rss_after_mb = _rss_current_mb()
        rss_process_peak_mb = _rss_process_peak_mb()
        fds_after = _fd_count()
        tasks_peak = max([tasks_at_release, *sampled_task_counts])
        threads_peak = max([threads_before, *sampled_thread_counts])
        stop_heartbeat.set()
        await heartbeat
        heartbeat = None

        # Keep teardown-only emissions out of fan-out publication cost metrics.
        registry._notify_agents = original_notify
        registry._emit_agents_update = original_emit
        teardown_started_at = time.perf_counter()
        await harness.close()
        teardown_ms = (time.perf_counter() - teardown_started_at) * 1000
        closed = True
        await asyncio.sleep(0)

        current_task = asyncio.current_task()
        leaked_tasks = {
            task
            for task in asyncio.all_tasks()
            if task not in baseline_tasks
            and task is not current_task
            and not task.done()
        }
        assert not leaked_tasks, [task.get_name() for task in leaked_tasks]

        return {
            "launch_latency_ms": launch_latencies_ms,
            "loop_lag_ms": loop_lag_ms,
            "agents_update_serialization_proxy_ms": emit_durations_ms,
            "agents_update_serialization_proxy_bytes": payload_sizes_bytes,
            "agents_update_count": agents_update_count,
            "rss_current_before_mb": rss_before_mb,
            "rss_current_after_mb": rss_after_mb,
            "rss_current_delta_mb": rss_after_mb - rss_before_mb,
            "rss_process_peak_mb": rss_process_peak_mb,
            "fd_count_before": fds_before,
            "fd_count_after_release": fds_after,
            "fd_delta": fds_after - fds_before,
            "thread_count_before": threads_before,
            "thread_count_peak": threads_peak,
            "asyncio_task_count_at_release": tasks_at_release,
            "asyncio_task_count_peak": tasks_peak,
            "teardown_ms": teardown_ms,
            "leaked_asyncio_tasks": 0,
        }
    finally:
        for release in backend.releases:
            release.set()
        stop_heartbeat.set()
        if heartbeat is not None:
            heartbeat.cancel()
            heartbeat_results = await asyncio.gather(heartbeat, return_exceptions=True)
        if harness is not None and not closed:
            await harness.close()
        assert all(
            result is None or isinstance(result, asyncio.CancelledError)
            for result in heartbeat_results
        ), f"heartbeat failed: {heartbeat_results}"


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 3)


def _summarize(
    agent_count: int, iterations: list[_IterationMetrics]
) -> _ConcurrencySummary:
    launch_samples = [
        sample for iteration in iterations for sample in iteration["launch_latency_ms"]
    ]
    emit_samples = [
        sample
        for iteration in iterations
        for sample in iteration["agents_update_serialization_proxy_ms"]
    ]
    payload_samples = [
        float(sample)
        for iteration in iterations
        for sample in iteration["agents_update_serialization_proxy_bytes"]
    ]
    iteration_summaries = [
        {
            "launch_latency_ms": percentiles(iteration["launch_latency_ms"]),
            "loop_lag_ms": percentiles(iteration["loop_lag_ms"]),
            "agents_update_serialization_proxy_ms": percentiles(
                iteration["agents_update_serialization_proxy_ms"]
            ),
            "agents_update_serialization_proxy_bytes": percentiles([
                float(sample)
                for sample in iteration["agents_update_serialization_proxy_bytes"]
            ]),
            "agents_update_count": iteration["agents_update_count"],
            "rss_current_before_mb": round(
                float(iteration["rss_current_before_mb"]), 3
            ),
            "rss_current_after_mb": round(float(iteration["rss_current_after_mb"]), 3),
            "rss_current_delta_mb": round(float(iteration["rss_current_delta_mb"]), 3),
            "rss_process_peak_mb": round(float(iteration["rss_process_peak_mb"]), 3),
            "fd_count_before": iteration["fd_count_before"],
            "fd_count_after_release": iteration["fd_count_after_release"],
            "fd_delta": iteration["fd_delta"],
            "thread_count_before": iteration["thread_count_before"],
            "thread_count_peak": iteration["thread_count_peak"],
            "asyncio_task_count_at_release": iteration["asyncio_task_count_at_release"],
            "asyncio_task_count_peak": iteration["asyncio_task_count_peak"],
            "teardown_ms": round(float(iteration["teardown_ms"]), 3),
            "leaked_asyncio_tasks": iteration["leaked_asyncio_tasks"],
        }
        for iteration in iterations
    ]
    return {
        "n": agent_count,
        "iterations": len(iterations),
        "launch_latency_ms": percentiles(launch_samples),
        "launch_latency_iteration_p50_median_ms": _median([
            summary["launch_latency_ms"]["p50"] for summary in iteration_summaries
        ]),
        "loop_lag_ms": {
            "p50": _median([
                summary["loop_lag_ms"]["p50"] for summary in iteration_summaries
            ]),
            "p99": _median([
                summary["loop_lag_ms"]["p99"] for summary in iteration_summaries
            ]),
            "max": _median([
                summary["loop_lag_ms"]["max"] for summary in iteration_summaries
            ]),
        },
        "agents_update_serialization_proxy_ms": percentiles(emit_samples),
        "agents_update_serialization_proxy_iteration_p50_median_ms": _median([
            summary["agents_update_serialization_proxy_ms"]["p50"]
            for summary in iteration_summaries
        ]),
        "agents_update_serialization_proxy_bytes": percentiles(payload_samples),
        "agents_update_count_per_iteration": [
            summary["agents_update_count"] for summary in iteration_summaries
        ],
        "rss_current_delta_mb_median": _median([
            float(summary["rss_current_delta_mb"]) for summary in iteration_summaries
        ]),
        "rss_current_after_mb_median": _median([
            float(summary["rss_current_after_mb"]) for summary in iteration_summaries
        ]),
        "rss_process_peak_mb_median": _median([
            float(summary["rss_process_peak_mb"]) for summary in iteration_summaries
        ]),
        "fd_count_before_median": _median([
            float(summary["fd_count_before"]) for summary in iteration_summaries
        ]),
        "fd_count_after_release_median": _median([
            float(summary["fd_count_after_release"]) for summary in iteration_summaries
        ]),
        "fd_delta_median": _median([
            float(summary["fd_delta"]) for summary in iteration_summaries
        ]),
        "thread_count_before_median": _median([
            float(summary["thread_count_before"]) for summary in iteration_summaries
        ]),
        "thread_count_peak_median": _median([
            float(summary["thread_count_peak"]) for summary in iteration_summaries
        ]),
        "asyncio_task_count_at_release_median": _median([
            float(summary["asyncio_task_count_at_release"])
            for summary in iteration_summaries
        ]),
        "asyncio_task_count_peak_median": _median([
            float(summary["asyncio_task_count_peak"]) for summary in iteration_summaries
        ]),
        "teardown_ms_median": _median([
            float(summary["teardown_ms"]) for summary in iteration_summaries
        ]),
        "leaked_asyncio_tasks_per_iteration": [
            summary["leaked_asyncio_tasks"] for summary in iteration_summaries
        ],
        "machine": machine_context(),
    }


@pytest.mark.perf
@pytest.mark.timeout(600)
@pytest.mark.parametrize("agent_count", [8, 16])
@pytest.mark.asyncio
async def test_real_fan_out_concurrency(
    agent_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    iterations = [await _iteration(agent_count, monkeypatch) for _ in range(3)]
    metrics = _summarize(agent_count, iterations)
    assert all(
        int(count) >= 2 * agent_count + 1
        for count in metrics["agents_update_count_per_iteration"]
    )
    assert metrics["leaked_asyncio_tasks_per_iteration"] == [0, 0, 0]
    record(f"concurrency_n{agent_count}", dict(metrics))
