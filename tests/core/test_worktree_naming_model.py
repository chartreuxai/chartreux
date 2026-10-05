from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from git import Repo
import pytest

from chartreux.app_server._worktree_session import SessionWorktrees
from chartreux.app_server.protocol import (
    AutoWorktreeInput,
    ExistingWorktreeInput,
    NewWorktreeInput,
    SessionOptions,
)
from chartreux.core._usage_startup import create_startup_accounting_context
from chartreux.core.git.worktree import naming_model as worktree_naming_model
from chartreux.core.git.worktree.naming_model import suggest_worktree_name
from chartreux.core.llm import utility_completion
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm_models import LLMChunk, LLMMessage, LLMUsage, Role
from chartreux.core.usage import UsageOutcome, UsageReader, UsageRecord, capture_prices
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    answer: str | None = None,
    error: Exception | None = None,
    slow: bool = False,
) -> None:
    class _Orchestrator:
        config = object()

    async def build(**_kwargs: Any) -> _Orchestrator:
        return _Orchestrator()

    async def complete(**_kwargs: Any) -> str | None:
        if slow:
            await asyncio.sleep(10)
        if error is not None:
            raise error
        return answer

    monkeypatch.setattr(worktree_naming_model, "build_default_orchestrator", build)
    monkeypatch.setattr(worktree_naming_model, "run_utility_completion", complete)


async def _suggest() -> str | None:
    return await suggest_worktree_name("Fix the login bug", cwd=Path.cwd())


@pytest.mark.asyncio
async def test_returns_the_model_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, answer="fix-login-bug")

    assert await _suggest() == "fix-login-bug"


@pytest.mark.asyncio
async def test_returns_none_when_the_model_answers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, answer="")

    assert await _suggest() is None


@pytest.mark.asyncio
async def test_returns_none_when_the_completion_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, error=RuntimeError("connection reset"))

    assert await _suggest() is None


@pytest.mark.asyncio
async def test_returns_none_when_the_model_is_too_slow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worktree_naming_model, "_TOTAL_TIMEOUT_SECONDS", 0.01)
    _install(monkeypatch, slow=True)

    assert await _suggest() is None


@pytest.mark.asyncio
async def test_opts_into_the_no_key_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Orchestrator:
        config = object()

    async def build(**_kwargs: Any) -> _Orchestrator:
        return _Orchestrator()

    async def complete(**kwargs: Any) -> str | None:
        captured.update(kwargs)
        return "fix-login-bug"

    monkeypatch.setattr(worktree_naming_model, "build_default_orchestrator", build)
    monkeypatch.setattr(worktree_naming_model, "run_utility_completion", complete)

    await _suggest()

    # Naming runs at session start with a deterministic fallback ready, so a
    # keyless provider must short-circuit instead of waiting out the budget.
    assert captured["skip_if_no_key"] is True


@pytest.mark.asyncio
async def test_does_not_load_config_without_a_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(**_kwargs: Any) -> None:
        raise AssertionError("no config should be loaded without a prompt")

    monkeypatch.setattr(worktree_naming_model, "build_default_orchestrator", explode)

    assert await suggest_worktree_name("", cwd=Path.cwd()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "failure",
        "timeout",
        "empty-answer",
        "malformed-answer",
        "sink-failure",
        "unbound",
        "no-prompt",
        "empty-input",
        "missing-key",
        "named",
        "existing",
        "no-worktree",
    ],
)
@pytest.mark.parametrize("use_factory", [False, True])
async def test_startup_naming_accounting_through_worktree_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    scenario: str,
    use_factory: bool,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with Repo.init(project, initial_branch="main") as repo:
        repo.config_writer().set_value("user", "name", "Tester").release()
        repo.config_writer().set_value("user", "email", "t@example.com").release()
        (project / "file.txt").write_text("hello\n")
        repo.index.add(["file.txt"])
        repo.index.commit("initial")

    context = await create_startup_accounting_context(
        project, usage_dir=tmp_path / "usage"
    )
    writer = context.early_writer()
    config = build_test_vibe_config()
    model, provider = utility_completion.select_utility_model(config)
    calls: list[dict[str, Any]] = []
    factory_calls: list[dict[str, Any]] = []
    sink_calls: list[UsageRecord] = []
    backend = FakeBackend()

    class _Orchestrator:
        def __init__(self) -> None:
            self.config = config

    orchestrator = _Orchestrator()

    async def build(**_kwargs: Any) -> _Orchestrator:
        return orchestrator

    def factory(**kwargs: Any) -> FakeBackend:
        factory_calls.append(kwargs)
        return backend

    async def complete(**kwargs: Any) -> LLMChunk:
        calls.append(kwargs)
        notify_request_started()
        if scenario == "failure":
            raise RuntimeError("backend failure")
        if scenario == "timeout":
            await asyncio.sleep(10)
        answer = {"empty-answer": "", "malformed-answer": "!!!"}.get(
            scenario, "repair-oauth-redirect"
        )
        return LLMChunk(
            message=LLMMessage(role=Role.assistant, content=answer),
            usage=LLMUsage.from_reported(prompt_tokens=10, completion_tokens=3),
        )

    async def sink(record: UsageRecord) -> None:
        sink_calls.append(record)
        if scenario == "sink-failure":
            raise OSError("private sink error")
        await writer(record)

    monkeypatch.setattr(worktree_naming_model, "build_default_orchestrator", build)
    monkeypatch.setattr(utility_completion, "create_backend", factory)
    monkeypatch.setattr(
        utility_completion,
        "resolve_api_key",
        lambda _: None if scenario == "missing-key" else "test-key",
    )
    monkeypatch.setattr(backend, "complete", complete)
    if scenario == "timeout":
        monkeypatch.setattr(worktree_naming_model, "_TOTAL_TIMEOUT_SECONDS", 0.01)

    options = SessionOptions(
        cwd=str(project), worktree=AutoWorktreeInput(prompt="Fix the login bug")
    )
    if scenario == "no-prompt":
        options.worktree = AutoWorktreeInput()
    elif scenario == "empty-input":
        options.worktree = AutoWorktreeInput(prompt="   ")
    elif scenario == "named":
        options.worktree = NewWorktreeInput(
            name="chosen-name", branch="feat/chosen-name"
        )
    elif scenario == "existing":
        existing = tmp_path / "existing"
        with Repo(project) as repo:
            repo.git.worktree("add", "-b", "existing", str(existing))
        options.worktree = ExistingWorktreeInput(cwd=str(existing))
    elif scenario == "no-worktree":
        options.worktree = None

    lifecycle = SessionWorktrees()
    try:
        resolution = await lifecycle.resolve_for_start(
            options,
            accounting_sink=None if scenario == "unbound" else sink,
            usage_attribution=None
            if scenario == "unbound" or use_factory
            else context.attribution(model),
            usage_attribution_factory=None
            if scenario == "unbound" or not use_factory
            else context.attribution,
        )
        skipped = scenario in {
            "no-prompt",
            "empty-input",
            "missing-key",
            "named",
            "existing",
            "no-worktree",
        }
        assert len(calls) == len(factory_calls) == int(not skipped)
        assert len(sink_calls) == int(not skipped and scenario != "unbound")
        if calls:
            assert factory_calls == [
                {"provider": provider, "timeout": 1.5, "retry_max_elapsed_time": 0}
            ]
            assert calls[0]["temperature"] == 0.0
            assert calls[0]["max_tokens"] == 24
        if scenario in {"failure", "timeout", "empty-answer", "malformed-answer"}:
            assert resolution.prepared_worktree is not None
            assert resolution.prepared_worktree.name == "fix-the-login-bug"
        elif scenario in {"success", "sink-failure", "unbound"}:
            assert resolution.prepared_worktree is not None
            assert resolution.prepared_worktree.name == "repair-oauth-redirect"
        await writer.drain()
        records = UsageReader(context.identity.usage_dir).reconcile().records
        assert len(records) == int(
            not skipped and scenario not in {"unbound", "sink-failure"}
        )
        if sink_calls:
            record = sink_calls[0]
            assert record.purpose == "worktree-naming"
            assert record.agent_role == "startup"
            assert record.agent_profile is record.parent_session_id is None
            assert (
                record.root_session_id
                == record.session_id
                == context.identity.root_session_id
            )
            assert record.project_key == context.identity.project_key
            assert (record.model, record.provider, record.wire_name) == (
                model.alias,
                provider.name,
                model.name,
            )
            assert record.prices_usd_per_million == capture_prices(model)
            assert record.outcome == {
                "failure": UsageOutcome.FAILED,
                "timeout": UsageOutcome.INTERRUPTED,
            }.get(scenario, UsageOutcome.COMPLETED)
        if scenario == "sink-failure":
            assert "coverage is degraded" in caplog.text
            assert "private sink error" not in caplog.text
        await lifecycle.cleanup(resolution)
    finally:
        await writer.aclose()
