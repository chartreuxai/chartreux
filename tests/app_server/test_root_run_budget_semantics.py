"""C1: the root run-budget interval (characterization only).

Pins the existing opt-in root semantics of the run budget:

- Session options arm the root loop (``RunBudgetMiddleware`` in the root
  pipeline) but arming does not admit a run: no anchor is written.
- A root with no anchor self-initializes at the FIRST middleware
  evaluation (the lazy fail-safe anchor), so the budget interval means
  "from first evaluation onward".
- Resume preserves persisted run anchors: the middleware reset is a no-op
  and the persisted baseline is restored on the rebind, so the interval
  spans the resume boundary.
- The ``[subagents]`` run-budget config default does NOT arm the root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.config import SessionLoggingConfig
from chartreux.core.middleware import MiddlewareAction, ResetReason, RunBudgetMiddleware
from chartreux.core.session_types import AgentStats
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend


def _capture_anchor_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record the stats snapshot of every anchor_run_baseline call, in order."""
    captured: list[tuple] = []
    original = AgentStats.anchor_run_baseline

    def capture(self: AgentStats) -> None:
        captured.append((
            self.steps,
            self.known_cost_total,
            self.session_total_llm_tokens,
            self.has_unknown_cost,
        ))
        original(self)

    monkeypatch.setattr(AgentStats, "anchor_run_baseline", capture)
    return captured


def _session_config(tmp_path: Path):
    return build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )


def _run_budget(loop) -> RunBudgetMiddleware:
    budgets = [
        mw
        for mw in loop.middleware_pipeline.middlewares
        if isinstance(mw, RunBudgetMiddleware)
    ]
    assert len(budgets) == 1
    return budgets[0]


@pytest.mark.asyncio
async def test_session_options_arm_root_without_child_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_anchor_calls(monkeypatch)
    parent = build_test_agent_loop(
        backend=FakeBackend(), run_max_price=0.5, run_max_session_tokens=2000
    )
    try:
        middleware = _run_budget(parent)
        assert middleware.run_max_price == 0.5
        assert middleware.run_max_session_tokens == 2000
        assert parent.runtime_policy.run_max_price == 0.5
        assert parent.runtime_policy.run_max_session_tokens == 2000

        # Arming is not admission: no run baseline was anchored, and the
        # root's stats carry no anchor yet.
        assert captured == []
        assert parent.stats.run_anchor_steps is None
        assert parent.stats.run_anchor_known_cost_total is None
        assert parent.stats.run_anchor_session_tokens is None
        assert parent.stats.run_anchor_has_unknown_cost is None
    finally:
        await parent.aclose()


@pytest.mark.asyncio
async def test_unanchored_root_self_initializes_on_first_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_anchor_calls(monkeypatch)
    parent = build_test_agent_loop(backend=FakeBackend(), run_max_price=0.5)
    try:
        middleware = _run_budget(parent)
        assert parent.stats.run_anchor_steps is None
        assert captured == []

        # The first evaluation lazily self-anchors at the current stats:
        # the interval starts here, not at loop construction.
        result = middleware.evaluate(parent.stats)
        assert result.action is not MiddlewareAction.STOP
        assert captured == [(0, 0.0, 0, False)]
        assert parent.stats.run_anchor_steps == 0
        assert parent.stats.run_anchor_known_cost_total == 0.0
        assert parent.stats.run_anchor_session_tokens == 0
        assert parent.stats.run_anchor_has_unknown_cost is False

        # The interval is measured from that first evaluation: consumption
        # above the max stops the run. A re-anchor on the second evaluation
        # would swallow the delta and pass.
        parent.stats.known_cost_total = 0.75
        stopped = middleware.evaluate(parent.stats)
        assert stopped.action is MiddlewareAction.STOP
        assert stopped.reason is not None
        assert "Run price limit exceeded" in stopped.reason
        assert len(captured) == 1
    finally:
        await parent.aclose()


async def _persist_anchored_session(tmp_path: Path) -> str:
    """Persist a root whose run anchor sits mid-run, below its consumption."""
    saved = build_test_agent_loop(config=_session_config(tmp_path), run_max_price=5.0)
    saved.stats.steps = 3
    saved.stats.known_cost_total = 1.0
    saved.stats.session_prompt_tokens = 700
    saved.stats.session_completion_tokens = 100
    saved.stats.has_unknown_cost = False
    saved.stats.anchor_run_baseline()
    # Consumption after the anchor: the persisted anchor is NOT a snapshot
    # of the persisted stats, so a resume that re-based it is detectable.
    saved.stats.steps = 5
    saved.stats.known_cost_total = 2.0
    saved.stats.session_prompt_tokens = 900
    saved.stats.session_completion_tokens = 100
    await saved.persist_empty_session()
    session_id = saved.session_id
    await saved.aclose()
    return session_id


@pytest.mark.asyncio
async def test_root_resume_preserves_existing_run_anchors(tmp_path: Path) -> None:
    target_session_id = await _persist_anchored_session(tmp_path)
    source = build_test_agent_loop(
        config=_session_config(tmp_path), backend=FakeBackend(), run_max_price=5.0
    )
    session = await create_test_app_server_session(source)
    try:
        await session.resume(target_session_id)
        assert source.session_id == target_session_id

        # The persisted baseline is restored verbatim on the rebind — not
        # re-based onto the loaded (post-anchor) consumption.
        assert source.stats.run_anchor_steps == 3
        assert source.stats.run_anchor_known_cost_total == 1.0
        assert source.stats.run_anchor_session_tokens == 800
        assert source.stats.run_anchor_has_unknown_cost is False
        assert source.stats.steps == 5
        assert source.stats.known_cost_total == 2.0

        # The middleware reset at the resume boundary does not rebase
        # either: the restored baseline survives it untouched.
        middleware = _run_budget(source)
        middleware.reset(ResetReason.STOP)
        assert source.stats.run_anchor_steps == 3
        assert source.stats.run_anchor_known_cost_total == 1.0

        # Re-evaluation keeps the restored baseline, so the resumed run's
        # interval still spans the resume boundary: pre-resume consumption
        # counts against this run's budget.
        result = middleware.evaluate(source.stats)
        assert result.action is not MiddlewareAction.STOP
        middleware.run_max_price = 0.75
        stopped = middleware.evaluate(source.stats)
        assert stopped.action is MiddlewareAction.STOP
        assert stopped.reason is not None
        assert "Run price limit exceeded" in stopped.reason
        assert "$1.0000" in stopped.reason
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_subagent_config_default_does_not_arm_root() -> None:
    parent = build_test_agent_loop(backend=FakeBackend())
    # Live-config mutation (the shared support pattern): the run-budget
    # defaults arm subagents created after the mutation, never the root.
    parent.config.subagents.run_max_price = 0.25
    parent.config.subagents.run_max_session_tokens = 2000
    try:
        root_types = [type(mw) for mw in parent.middleware_pipeline.middlewares]
        assert RunBudgetMiddleware not in root_types
        assert parent._run_max_price is None
        assert parent._run_max_session_tokens is None
        assert parent.runtime_policy.run_max_price is None
        assert parent.runtime_policy.run_max_session_tokens is None
    finally:
        await parent.aclose()
