"""P14 acceptance: the orchestrator dispatch policy + advisory budget-fit log.

Covers the P14 acceptance surface:

- The dispatch-policy text served by the orchestrator identity: scope-based
  routing (release/publish/destructive/migration/git-state never ride the
  flash tier), predecessor retention when replacement lineage may be needed,
  first typed audit-rejection escalation through the existing lineage, the
  discriminating-check re-run before consequential reliance, budget-fit as
  advisory log + prose only, and the preserved B1 orchestrator exception
  (probes and trivial state/bookkeeping edits).
- The task tool prompt documents the completion-mandate fields
  (``completion_check`` / ``required_artifacts``) with fail-open semantics.
- The ONE advisory budget-fit log line at background admission: emitted once
  per admission, advisory only — it never gates admission and never rejects
  a launch, with the run budget configured or unconfigured.

All transports are mocked (no live-provider verification). The P14 brief
forbids editing ``tests/tools/test_task.py``; its pins are exercised
unmodified in the acceptance run.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.models import SubagentsConfig
from chartreux.core.llm_models import LLMChunk
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.models import ToolPermission
from tests.app_server._subagent_test_support import (
    _background_result,
    _retention_registry,
    _wait_result,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.timeout(60)

ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_IDENTITY = ROOT / "chartreux/core/prompts/identities/orchestrator.md"
TASK_PROMPT = ROOT / "chartreux/core/tools/builtins/prompts/task.md"

DISPATCH_POLICY_ANCHORS: tuple[str, ...] = (
    "Dispatch policy",
    "never ride the flash tier",
    "retain the implementation predecessor",
    "FIRST typed audit rejection",
    "replacement-claim validation",
    "discriminating check",
    "advisory log line",
    "never a preflight rejection",
)

# B1's orchestrator exception (probes and trivial state/bookkeeping edits):
# the dispatch policy must preserve it, never override it.
B1_EXCEPTION_ANCHORS: tuple[str, ...] = (
    "one-line read-only probes",
    "trivial state/bookkeeping edits",
    "direct-tool reconciliation above stands",
)

COMPLETION_MANDATE_ANCHORS: tuple[str, ...] = (
    "`completion_check`",
    "`required_artifacts`",
    "`pytest`",
    "`ruff_check`",
    "`pyright`",
    "`git_head`",
    "`git_status`",
    "`git_diff_stat`",
    "`commit`",
    "`touched_path`",
    "`check_pass`",
    "`envelope`",
    "fail-open",
)


def test_orchestrator_dispatch_policy_is_served() -> None:
    body = ORCHESTRATOR_IDENTITY.read_text(encoding="utf-8")
    for anchor in DISPATCH_POLICY_ANCHORS:
        assert anchor in body, anchor


def test_dispatch_policy_preserves_the_b1_orchestrator_exception() -> None:
    body = ORCHESTRATOR_IDENTITY.read_text(encoding="utf-8")
    for anchor in B1_EXCEPTION_ANCHORS:
        assert anchor in body, anchor


def test_task_prompt_documents_the_completion_mandate_fields() -> None:
    prompt = TASK_PROMPT.read_text(encoding="utf-8")
    for anchor in COMPLETION_MANDATE_ANCHORS:
        assert anchor in prompt, anchor


# ---------------------------------------------------------------------------
# The advisory budget-fit log line at admission
# ---------------------------------------------------------------------------


class _NotedBackend(FakeBackend):
    """A parent backend that answers every turn with terminal content."""

    async def complete(self, **kwargs: Any) -> LLMChunk:
        return mock_llm_chunk(content="parent noted")

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        yield mock_llm_chunk(content="parent noted")


def _config_with_run_budget(
    *, run_max_price: float | None = None, run_max_session_tokens: int | None = None
) -> Any:
    return build_test_vibe_config(
        enabled_tools=["task", "bash"],
        tools={
            name: {"permission": ToolPermission.ALWAYS.value}
            for name in ("task", "bash")
        },
        session_logging=SessionLoggingConfig(enabled=False),
        subagents=SubagentsConfig(
            run_max_price=run_max_price, run_max_session_tokens=run_max_session_tokens
        ),
    )


async def _registry_with_parent(
    monkeypatch: pytest.MonkeyPatch, config: Any
) -> tuple[SessionRuntimeRegistry, AgentLoop]:
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="child done")]),
    )
    parent = build_test_agent_loop(config=config, backend=_NotedBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


def _budget_fit_records(caplog: pytest.LogCaptureFixture) -> list[Any]:
    return [
        record
        for record in caplog.records
        if "Budget-fit advisory" in record.getMessage()
    ]


@pytest.mark.asyncio
async def test_admission_logs_one_advisory_budget_fit_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    registry, parent = await _registry_with_parent(
        monkeypatch,
        _config_with_run_budget(run_max_price=1.5, run_max_session_tokens=100_000),
    )
    try:
        with caplog.at_level(logging.INFO, logger="vibe"):
            ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
            ack = await _background_result(
                registry,
                TaskArgs(
                    task="Mandated work", identity="coder", background=True, retain=True
                ),
                ctx,
            )
            # Advisory only: the launch is admitted, never rejected.
            assert ack.completed
            assert ack.agent_id is not None and ack.run_id is not None
            result = await _wait_result(registry, ack.agent_id, ack.run_id)
            assert result.completed
        records = _budget_fit_records(caplog)
        # ONE advisory budget-fit log line per admission.
        assert len(records) == 1
        message = records[0].getMessage()
        assert "price<=1.50" in message
        assert "tokens<=100000" in message
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_unconfigured_run_budget_logs_the_unconfigured_advisory(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    registry, parent = await _registry_with_parent(
        monkeypatch, _config_with_run_budget()
    )
    try:
        with caplog.at_level(logging.INFO, logger="vibe"):
            ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
            ack = await _background_result(
                registry,
                TaskArgs(
                    task="Mandated work", identity="coder", background=True, retain=True
                ),
                ctx,
            )
            assert ack.completed
        records = _budget_fit_records(caplog)
        assert len(records) == 1
        assert "unconfigured" in records[0].getMessage()
    finally:
        await registry.drain_children()
        await parent.aclose()
