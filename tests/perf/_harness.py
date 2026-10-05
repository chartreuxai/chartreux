from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import time
from typing import Any
from unittest.mock import AsyncMock

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.subagents import AgentEviction, LaunchConfig, TaskArgs, TaskResult
from chartreux.core.tools.base import InvokeContext
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class GatedSequenceBackend(FakeBackend):
    """Return one response per gate, keeping every selected child run observable."""

    def __init__(self) -> None:
        super().__init__()
        self.started: list[asyncio.Event] = []
        self.releases: list[asyncio.Event] = []
        self.started_at: list[float | None] = []
        self._next_gate_index = 0

    def add_gate(self) -> tuple[asyncio.Event, asyncio.Event]:
        started = asyncio.Event()
        release = asyncio.Event()
        self.started.append(started)
        self.releases.append(release)
        self.started_at.append(None)
        self._streams.append([mock_llm_chunk(content="done")])
        return started, release

    async def complete(
        self,
        *,
        model: str,
        messages: list[Any],
        temperature: float | None,
        tools: list[Any],
        tool_choice: Any,
        extra_headers: dict[str, str] | None,
        max_tokens: int | None,
        metadata: dict[str, str] | None = None,
    ) -> Any:
        # Origin: tests/app_server/test_subagents.py:191-211. The original chose
        # a gate using len(requests_messages), which races because concurrent calls
        # all see the same length while waiting. Reserve a unique index synchronously
        # before the first await so every concurrent call owns a distinct gate.
        gate_index = self._next_gate_index
        self._next_gate_index += 1
        if gate_index >= len(self.started):
            raise AssertionError(f"No gate was prepared for backend call {gate_index}")
        self.started_at[gate_index] = time.perf_counter()
        self.started[gate_index].set()
        await self.releases[gate_index].wait()
        return await super().complete(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            extra_headers=extra_headers,
            max_tokens=max_tokens,
            metadata=metadata,
        )


@dataclass(slots=True)
class SubagentHarness:
    """Real retained-session machinery configured for explicit subagent models."""

    parent: AgentLoop
    registry: SessionRuntimeRegistry
    context: InvokeContext
    model_names: list[str]
    agents_update_sizes: list[int]

    async def close(self) -> None:
        """Drain retained children and close the root agent loop."""
        await self.registry.drain_children()
        await self.parent.aclose()


async def create_subagent_harness(model_names: Sequence[str]) -> SubagentHarness:
    """Create a registry for launching independent tasks with explicit models."""
    if not model_names:
        raise ValueError("at least one explicit model is required")

    providers: dict[str, dict[str, str]] = {
        "test-perf": {"api_base": "https://perf.test", "backend": "generic"}
    }
    models = {
        name: {"deployments": [{"provider": "test-perf", "name": f"{name}-wire"}]}
        for name in model_names
    }
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": providers,
            "models": models,
            "roles": {
                "orchestrator": {
                    "description": "Performance harness main preset",
                    "model": model_names[0],
                    "thinking": "high",
                }
            },
        }),
        "perf-fan-out-test",
    )
    config = build_test_vibe_config(
        enabled_tools=["task"], tools={"task": {"permission": "always"}}
    ).attach_catalog_snapshot(snapshot)
    parent = build_test_agent_loop(
        config=config,
        backend=FakeBackend([
            [mock_llm_chunk(content="parent done")] for _ in model_names
        ]),
    )

    # Origin: tests/app_server/test_subagents.py:301-330 for the parent,
    # SessionRuntimeRegistry, and AgentRuntimeFactory lifecycle; fan-out registry
    # binding follows test_subagents.py:5434-5514. Unlike the direct-child test,
    # this bootstrap binds only the root so measured launches go through registry.run.
    agents_update_sizes: list[int] = []

    async def notify_agents(
        agents: list[AgentSummaryModel], evictions: list[AgentEviction]
    ) -> None:
        del evictions
        agents_update_sizes.append(len(agents))

    registry = SessionRuntimeRegistry(
        AsyncMock(),
        AsyncMock(),
        lambda _session_id: 0,
        runtime_factory=AgentRuntimeFactory(),
        notify_agents=notify_agents,
    )
    root = registry._build_child_runtime(parent)
    root.turns._projector = None
    registry.bind_root(root)

    return SubagentHarness(
        parent=parent,
        registry=registry,
        context=InvokeContext(
            tool_call_id="perf-subagents", session_id=parent.session_id
        ),
        model_names=list(model_names),
        agents_update_sizes=agents_update_sizes,
    )


async def launch_explicit_tasks(
    registry: SessionRuntimeRegistry,
    context: InvokeContext,
    model_names: Sequence[str],
    *,
    task: str = "Run the performance harness task",
) -> list[TaskResult]:
    """Launch one background task per explicit model, concurrently."""

    async def launch_one(index: int, model_name: str) -> TaskResult:
        result: TaskResult | None = None
        args = TaskArgs(
            task=f"{task} ({model_name})",
            background=True,
            config=LaunchConfig(model=model_name),
        )
        task_context = InvokeContext(
            tool_call_id=f"{context.tool_call_id}-{index}",
            session_id=context.session_id,
        )
        async for event in registry.run(args, task_context):
            if isinstance(event, TaskResult):
                result = event
        if result is None:
            raise RuntimeError("registry.run completed without a task launch result")
        return result

    return list(
        await asyncio.gather(
            *(
                launch_one(index, model_name)
                for index, model_name in enumerate(model_names)
            )
        )
    )
