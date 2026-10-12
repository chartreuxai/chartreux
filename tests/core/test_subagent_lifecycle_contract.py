"""Lifecycle-contract tests for the 0.5.3 subagent lifecycle rethink (WP1).

Covers the WP1 contract surface only: strict parsing of the new ``retain`` and
tombstone fields, foreground retention rejection at both the tool and registry
admission boundaries, the frozen ``RetentionPolicy`` snapshot dataclass, shipped
profile defaults, the ``AgentClosedError``/``AgentEvictedError`` hierarchy,
``AgentAvailability.CLOSED``, and standalone construction/discrimination of the
§6 wait-disposition types (not yet activated on the port).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError
import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.core.agents.manager import AgentManager
from chartreux.core.agents.models import ADVISOR, REVIEWER, WORKER, AgentProfile
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.models import SubagentsConfig
from chartreux.core.subagents import (
    AgentAvailability,
    AgentClosedError,
    AgentEvictedError,
    RetentionPolicy,
    RunStatus,
    SubagentManagementError,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
    WaitForAgentResult,
    WaitStatus,
)
from chartreux.core.tools.base import BaseToolState, InvokeContext, ToolError
from chartreux.core.tools.builtins.task import Task, TaskToolConfig
from tests.conftest import ConfigBuilder, OrchestratorLoader
from tests.mock.utils import collect_result


class TestTaskArgsRetainValidation:
    def test_retain_defaults_to_none_and_stays_unset(self) -> None:
        args = TaskArgs(task="work")

        assert args.retain is None
        assert "retain" not in args.model_fields_set

    @pytest.mark.parametrize("retain", [True, False])
    def test_retain_accepts_strict_bools(self, retain: bool) -> None:
        args = TaskArgs(task="work", retain=retain)

        assert args.retain is retain
        assert "retain" in args.model_fields_set

    @pytest.mark.parametrize("retain", [0, 1, "true", "false", "yes", 1.0, [True]])
    def test_retain_rejects_non_bool_values(self, retain: object) -> None:
        with pytest.raises(ValidationError):
            TaskArgs.model_validate({"task": "work", "retain": retain})


class _RecordingRunner:
    """Minimal subagent runner that records the admitted args."""

    def __init__(self) -> None:
        self.calls: list[TaskArgs] = []

    async def run(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[TaskResult, None]:
        self.calls.append(args)
        yield TaskResult(response="done", turns_used=1, completed=True)


class TestForegroundRetentionRejection:
    @pytest.fixture
    def task_tool(self) -> Task:
        return Task(
            config_getter=lambda: TaskToolConfig(allowlist=["worker"]),
            state=BaseToolState(),
        )

    @pytest.fixture
    def ctx(
        self,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> InvokeContext:
        manager = AgentManager(load_orchestrator(build_config()))
        return InvokeContext(tool_call_id="test-call-id", agent_manager=manager)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("retain", [True, False])
    async def test_tool_rejects_explicit_retain_for_foreground(
        self, task_tool: Task, ctx: InvokeContext, retain: bool
    ) -> None:
        args = TaskArgs(task="work", background=False, retain=retain)

        with pytest.raises(
            ToolError, match="retain can only be used with background mode"
        ):
            await collect_result(task_tool.run(args, ctx))

    @pytest.mark.asyncio
    async def test_tool_admits_explicit_retain_for_background(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        runner = _RecordingRunner()
        ctx.subagent_runner = runner
        args = TaskArgs(task="work", background=True, retain=True)

        result = await collect_result(task_tool.run(args, ctx))

        assert result.completed
        assert runner.calls == [args]
        assert runner.calls[0].retain is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("retain", [True, False])
    async def test_registry_rejects_explicit_retain_for_foreground(
        self, retain: bool
    ) -> None:
        registry = SessionRuntimeRegistry(
            AsyncMock(), AsyncMock(), lambda _session_id: 0, max_running_subagents=16
        )
        root = MagicMock()
        root.agent_loop.session_id = "root"
        registry.bind_root(root)
        ctx = InvokeContext(tool_call_id="test-call-id", session_id="root")
        args = TaskArgs(task="work", background=False, retain=retain)

        with pytest.raises(
            ValueError, match="retain is only supported for background tasks"
        ):
            await anext(registry.run(args, ctx))


class TestSubagentsConfigContract:
    def test_defaults_match_the_0_5_3_lifecycle_model(self) -> None:
        config = SubagentsConfig()

        assert config.idle_ttl_seconds == 300
        assert config.max_idle_agents == 4
        assert config.retain_agents is False
        assert config.tombstone_ttl_seconds == 900
        assert config.max_tombstones == 32
        assert config.max_running_subagents == 16

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("idle_ttl_seconds", -1),
            ("idle_ttl_seconds", True),
            ("idle_ttl_seconds", "300"),
            ("max_idle_agents", -1),
            ("max_idle_agents", False),
            ("retain_agents", 0),
            ("retain_agents", 1),
            ("retain_agents", "true"),
            ("tombstone_ttl_seconds", -1),
            ("tombstone_ttl_seconds", True),
            ("tombstone_ttl_seconds", 1.5),
            ("max_tombstones", 0),
            ("max_tombstones", -1),
            ("max_tombstones", False),
            ("max_running_subagents", 0),
        ],
    )
    def test_invalid_values_are_rejected(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError):
            SubagentsConfig.model_validate({field: value})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("idle_ttl_seconds", 0),
            ("max_idle_agents", 0),
            ("retain_agents", True),
            ("tombstone_ttl_seconds", 0),
            ("max_tombstones", 1),
        ],
    )
    def test_boundary_values_are_accepted(self, field: str, value: object) -> None:
        config = SubagentsConfig.model_validate({field: value})

        assert getattr(config, field) == value

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SubagentsConfig.model_validate({"retain_agent": True})


class TestAgentProfileRetainParsing:
    @pytest.mark.parametrize(("raw", "expected"), [("true", True), ("false", False)])
    def test_retain_parses_as_a_strict_bool(
        self, tmp_path: Path, raw: str, expected: bool
    ) -> None:
        profile_path = tmp_path / "retained.toml"
        profile_path.write_text(f"retain = {raw}\n", encoding="utf-8")

        profile = AgentProfile.from_toml(profile_path)

        assert profile.retain is expected
        assert profile.overrides == {}

    def test_retain_defaults_to_none(self, tmp_path: Path) -> None:
        profile_path = tmp_path / "plain.toml"
        profile_path.write_text(
            'description = "No retention setting"\n', encoding="utf-8"
        )

        assert AgentProfile.from_toml(profile_path).retain is None

    @pytest.mark.parametrize("raw", ["1", "0", '"true"', '"yes"', "1.0"])
    def test_non_bool_retain_is_rejected(self, tmp_path: Path, raw: str) -> None:
        profile_path = tmp_path / "invalid.toml"
        profile_path.write_text(f"retain = {raw}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="retain must be a boolean"):
            AgentProfile.from_toml(profile_path)

    @pytest.mark.parametrize("raw", ["true", "false", "-1", '"invalid"'])
    def test_bool_and_range_violations_are_rejected_for_int_fields(
        self, tmp_path: Path, raw: str
    ) -> None:
        # TOML booleans are rejected where integers are expected.
        profile_path = tmp_path / "invalid.toml"
        profile_path.write_text(f"idle_ttl_seconds = {raw}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="idle_ttl_seconds must be an integer"):
            AgentProfile.from_toml(profile_path)

    @pytest.mark.parametrize("raw", ["1", '"true"'])
    def test_non_bool_retain_is_rejected_at_discovery(
        self,
        tmp_path: Path,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
        raw: str,
    ) -> None:
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "invalid.toml").write_text(f"retain = {raw}\n", encoding="utf-8")

        manager = AgentManager(
            load_orchestrator(build_config(agent_paths=[agents_dir]))
        )

        with pytest.raises(ValueError, match="not found"):
            manager.get_agent("invalid")


class TestShippedProfileDefaults:
    def test_advisor_is_retained_with_a_long_idle_ttl(self) -> None:
        assert ADVISOR.retain is True
        assert ADVISOR.idle_ttl_seconds == 3600

    def test_worker_and_reviewer_inherit_the_global_default(self) -> None:
        assert WORKER.retain is None
        assert WORKER.idle_ttl_seconds is None
        assert REVIEWER.retain is None
        assert REVIEWER.idle_ttl_seconds is None


class TestRetentionPolicy:
    def test_named_fields_and_equality(self) -> None:
        policy = RetentionPolicy(
            idle_ttl_seconds=300, max_idle_agents=4, retain_agents=False
        )

        assert policy.idle_ttl_seconds == 300
        assert policy.max_idle_agents == 4
        assert policy.retain_agents is False
        assert policy == RetentionPolicy(
            idle_ttl_seconds=300, max_idle_agents=4, retain_agents=False
        )
        assert policy != RetentionPolicy(
            idle_ttl_seconds=300, max_idle_agents=4, retain_agents=True
        )

    def test_policy_is_frozen(self) -> None:
        policy = RetentionPolicy(
            idle_ttl_seconds=300, max_idle_agents=4, retain_agents=False
        )

        with pytest.raises(FrozenInstanceError):
            policy.retain_agents = True  # type: ignore[misc]

    def test_registry_initializes_the_dataclass_snapshot(self) -> None:
        registry = SessionRuntimeRegistry(
            AsyncMock(), AsyncMock(), lambda _session_id: 0
        )

        assert registry._retention_policy == RetentionPolicy(
            idle_ttl_seconds=0, max_idle_agents=0, retain_agents=False
        )


class TestClosedErrorHierarchy:
    def test_evicted_is_a_closed_error(self) -> None:
        assert issubclass(AgentEvictedError, AgentClosedError)
        assert issubclass(AgentClosedError, SubagentManagementError)
        assert issubclass(AgentEvictedError, SubagentManagementError)

    def test_closed_and_evicted_stay_distinguishable(self) -> None:
        closed = AgentClosedError("closed")
        evicted = AgentEvictedError("evicted")

        assert isinstance(evicted, AgentClosedError)
        assert not isinstance(closed, AgentEvictedError)
        assert type(closed) is not type(evicted)
        assert str(closed) == "closed"
        assert str(evicted) == "evicted"

    def test_unknown_agent_is_not_a_closed_error(self) -> None:
        assert issubclass(UnknownAgentError, SubagentManagementError)
        assert not issubclass(UnknownAgentError, AgentClosedError)


class TestAgentAvailability:
    def test_closed_availability_is_available(self) -> None:
        assert AgentAvailability.CLOSED.value == "closed"
        assert AgentAvailability.CLOSED == "closed"
        assert {availability.value for availability in AgentAvailability} == {
            "running",
            "finalizing",
            "idle",
            "evicted",
            "closed",
        }


class TestWaitDispositionTypes:
    def test_completed_variant_carries_the_task_result(self) -> None:
        result = TaskResult(response="done", turns_used=2, completed=True)

        disposition = WaitForAgentResult(status=WaitStatus.COMPLETED, result=result)

        assert disposition.status is WaitStatus.COMPLETED
        assert disposition.status == "completed"
        assert disposition.result == result
        assert disposition.agent_id is None
        assert disposition.run_id is None
        assert disposition.elapsed_seconds is None
        assert disposition.run_status is None
        assert disposition.turns_used is None
        assert disposition.guidance is None

    def test_completed_variant_requires_a_result(self) -> None:
        with pytest.raises(ValidationError, match="requires a result"):
            WaitForAgentResult(status=WaitStatus.COMPLETED)

    def test_still_running_variant_carries_progress_and_guidance(self) -> None:
        disposition = WaitForAgentResult(
            status=WaitStatus.STILL_RUNNING,
            agent_id="agent-1",
            run_id="run-1",
            elapsed_seconds=12.5,
            run_status=RunStatus.RUNNING,
            turns_used=3,
            guidance="End your turn now; re-invoke wait_for_agent in a later turn.",
        )

        assert disposition.status is WaitStatus.STILL_RUNNING
        assert disposition.status == "still_running"
        assert disposition.result is None
        assert (disposition.agent_id, disposition.run_id) == ("agent-1", "run-1")
        assert disposition.elapsed_seconds == 12.5
        assert disposition.run_status is RunStatus.RUNNING
        assert disposition.turns_used == 3
        assert "End your turn" in (disposition.guidance or "")

    @pytest.mark.parametrize(
        "missing",
        [
            "agent_id",
            "run_id",
            "elapsed_seconds",
            "run_status",
            "turns_used",
            "guidance",
        ],
    )
    def test_still_running_variant_requires_every_progress_field(
        self, missing: str
    ) -> None:
        payload: dict[str, Any] = {
            "status": "still_running",
            "agent_id": "agent-1",
            "run_id": "run-1",
            "elapsed_seconds": 12.5,
            "run_status": "running",
            "turns_used": 3,
            "guidance": "yield",
        }
        del payload[missing]

        with pytest.raises(
            ValidationError, match="still_running wait disposition requires"
        ):
            WaitForAgentResult.model_validate(payload)

    @pytest.mark.parametrize(
        "field",
        [
            "agent_id",
            "run_id",
            "elapsed_seconds",
            "run_status",
            "turns_used",
            "guidance",
        ],
    )
    def test_completed_variant_rejects_still_running_fields(self, field: str) -> None:
        progress_values: dict[str, Any] = {
            "agent_id": "agent-1",
            "run_id": "run-1",
            "elapsed_seconds": 12.5,
            "run_status": "running",
            "turns_used": 3,
            "guidance": "yield",
        }
        payload: dict[str, Any] = {
            "status": "completed",
            "result": {"response": "done", "turns_used": 1, "completed": True},
            field: progress_values[field],
        }

        with pytest.raises(
            ValidationError, match="completed wait disposition cannot carry"
        ):
            WaitForAgentResult.model_validate(payload)

    def test_still_running_variant_rejects_a_result(self) -> None:
        with pytest.raises(
            ValidationError,
            match="still_running wait disposition cannot carry a result",
        ):
            WaitForAgentResult.model_validate({
                "status": "still_running",
                "result": {"response": "done", "turns_used": 1, "completed": True},
                "agent_id": "agent-1",
                "run_id": "run-1",
                "elapsed_seconds": 12.5,
                "run_status": "running",
                "turns_used": 3,
                "guidance": "yield",
            })

    def test_unset_variant_fields_round_trip_as_null(self) -> None:
        # model_dump() serializes the other variant's fields as null; a null
        # carries no data, so the dump -> validate round-trip stays valid.
        completed = WaitForAgentResult(
            status=WaitStatus.COMPLETED,
            result=TaskResult(response="done", turns_used=1, completed=True),
        )
        still_running = WaitForAgentResult(
            status=WaitStatus.STILL_RUNNING,
            agent_id="agent-1",
            run_id="run-1",
            elapsed_seconds=1.0,
            run_status=RunStatus.RUNNING,
            turns_used=1,
            guidance="yield",
        )

        assert WaitForAgentResult.model_validate(completed.model_dump()) == completed
        assert (
            WaitForAgentResult.model_validate(still_running.model_dump())
            == still_running
        )

    def test_variants_discriminate_by_status(self) -> None:
        completed = WaitForAgentResult(
            status=WaitStatus.COMPLETED,
            result=TaskResult(response="done", turns_used=1, completed=True),
        )
        still_running = WaitForAgentResult(
            status=WaitStatus.STILL_RUNNING,
            agent_id="agent-1",
            run_id="run-1",
            elapsed_seconds=1.0,
            run_status=RunStatus.RUNNING,
            turns_used=1,
            guidance="yield",
        )

        assert completed.status is not still_running.status
        assert isinstance(completed.result, TaskResult)
        assert still_running.result is None
        assert (
            WaitForAgentResult.model_validate_json(completed.model_dump_json())
            == completed
        )
        assert (
            WaitForAgentResult.model_validate_json(still_running.model_dump_json())
            == still_running
        )

    def test_unknown_status_and_extra_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WaitForAgentResult.model_validate({"status": "unknown"})

        with pytest.raises(ValidationError):
            WaitForAgentResult.model_validate({
                "status": "completed",
                "result": {"response": "done", "turns_used": 1, "completed": True},
                "unexpected": 1,
            })
