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
import importlib
import inspect
from pathlib import Path
import tomllib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError
import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server.root_agent import AgentChildHost
from chartreux.core.agents.identities import builtin_identities
from chartreux.core.agents.manager import AgentManager
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.harness_files import HarnessFilesManager
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
    prepare_subagent_prompt,
    still_running_guidance,
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
        # A runtime with no session directory: bind_root skips the registry
        # manifest and writer fence, which need a real directory to write to.
        root.agent_loop.session_logger.session_dir = None
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


class TestRetiredProfileTomlSurface:
    """The agents/*.toml profile axis is retired (package 16a).

    Profile TOML files are detected loudly and never parsed or consumed:
    the typing shim carries no parser, and the harness manager reports the
    directories as retired inputs while leaving the files untouched.
    """

    def test_profile_toml_parsing_is_retired(self) -> None:
        # The AgentProfile module retired with the profile axis (16a/16b):
        # no TOML parser surface remains anywhere.
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("chartreux.core.agents.models")

    @pytest.mark.parametrize("raw", ["true", "false", "1"])
    def test_agents_toml_is_detected_loudly_and_never_consumed(
        self, config_dir: Path, raw: str
    ) -> None:
        agents_dir = config_dir / "agents"
        agents_dir.mkdir()
        profile_file = agents_dir / "retained.toml"
        profile_file.write_text(f"retain = {raw}\n", encoding="utf-8")

        manager = HarnessFilesManager(sources=("user",))

        # The retired directory is reported; the file is left untouched and
        # is never parsed into an identity or a config layer.
        assert manager.retired_agent_dirs == [agents_dir]
        assert profile_file.read_text(encoding="utf-8") == f"retain = {raw}\n"


class TestShippedIdentitySurface:
    def test_shipped_profile_registry_is_retired(self) -> None:
        # The shipped profile objects (WORKER/ADVISOR/REVIEWER,
        # BUILTIN_SUBAGENTS) are gone with the registry module; retention is
        # config-driven (SubagentsConfig), never profile-driven.
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("chartreux.core.agents.models")

    def test_builtin_identities_carry_no_retention_defaults(self) -> None:
        for identity in builtin_identities():
            assert not hasattr(identity, "retain")
            assert not hasattr(identity, "idle_ttl_seconds")


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
            # WP2: warm recovery states (handoff §1.8).
            "suspended",
            "restoring",
        }


class TestTaskResultFailureContract:
    """WP1: TaskResult carries the typed failure evidence (0.5.6 recovery)."""

    def test_failure_fields_default_to_none_and_stay_unset(self) -> None:
        result = TaskResult(response="done", turns_used=1, completed=True)

        assert result.run_status is None
        assert result.failure is None
        assert result.recovery is None
        dumped = result.model_dump(mode="json")
        assert "run_status" not in dumped
        assert "failure" not in dumped
        assert "recovery" not in dumped

    def test_failed_result_carries_envelope_run_status_and_reference(self) -> None:
        from chartreux.core.recovery_models import (
            RecoveryEligibility,
            RecoveryReference,
            build_failure_envelope,
        )

        envelope = build_failure_envelope(
            observation=None, agent_id="agent-1", run_id="run-1"
        )
        result = TaskResult(
            response="failed",
            turns_used=1,
            completed=False,
            agent_id="agent-1",
            run_id="run-1",
            run_status=RunStatus.FAILED,
            failure=envelope,
            recovery=RecoveryReference.from_envelope(envelope),
        )

        assert result.run_status is RunStatus.FAILED
        assert result.failure is envelope
        assert result.recovery is not None
        assert result.recovery.event_id == envelope.event_id
        assert result.recovery.eligibility is RecoveryEligibility.PROHIBITED
        restored = TaskResult.model_validate(result.model_dump(mode="json"))
        assert restored == result

    def test_legacy_result_payload_validates_without_failure_fields(self) -> None:
        legacy = TaskResult.model_validate({
            "response": "done",
            "turns_used": 1,
            "completed": True,
        })

        assert legacy.failure is None
        assert legacy.recovery is None
        assert legacy.run_status is None

    def test_run_status_values_match_the_public_contract(self) -> None:
        assert {status.value for status in RunStatus} == {
            "running",
            "completed",
            "failed",
            "cancelled",
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

    def test_still_running_guidance_is_notification_first_and_conditional(self) -> None:
        """v0.7.9 point 4: the shipped guidance leads with yielding, prefers
        the completion notification and a fetch, forbids a same-turn re-wait,
        and conditions any later re-wait on the three wait conditions while
        retaining identifiers, elapsed time, status, turns and modes.
        """
        guidance = still_running_guidance(
            agent_id="agent-1",
            run_id="run-1",
            elapsed_seconds=12.5,
            run_status=RunStatus.RUNNING,
            turns_used=3,
        )

        assert guidance.startswith("Agent agent-1 (run run-1) is still running")
        assert "after 12.5s and 3 turns" in guidance
        assert "End your turn or continue independent work now" in guidance
        assert 'mode="digest"' in guidance and 'mode="full"' in guidance
        assert "Never re-invoke wait_for_agent in the same turn." in guidance
        assert guidance.index("Prefer the completion notification") < guidance.index(
            "In a later turn, re-wait only if"
        )
        assert (
            "re-wait only if the result is a hard dependency of your next "
            "action, no independent work can proceed, and that action must "
            "occur in that turn rather than after notification" in guidance
        )
        assert "if uncertain, end your turn" in guidance
        assert "check_agents" in guidance

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


class TestVerificationArtifactBrief:
    """The verification-artifact discipline rides in every task brief (A4)."""

    def test_a_scratchpad_brief_names_a_unique_invocation_directory(self) -> None:
        prompt = prepare_subagent_prompt(
            "Run the suite.",
            InvokeContext(tool_call_id="call-1", scratchpad_dir=Path("/tmp/sp")),
        )

        assert "fresh, unique directory under this scratchpad directory" in prompt
        assert "basetemps, tool caches, and logs" in prompt
        # A destination outside the scratchpad stays scanned and still blocks.
        assert (
            "Paths outside the scratchpad are still scanned and still block" in prompt
        )
        # Cleanup is the worker's own completed invocation only: no wildcard
        # removal, and no live job's directory is touched.
        assert "Clean up only your own completed invocation's directory" in prompt
        assert "never wildcard-remove sibling invocations" in prompt
        assert "any directory a still-running job is using" in prompt

    def test_a_brief_without_a_scratchpad_stops_the_verification(self) -> None:
        prompt = prepare_subagent_prompt(
            "Run the suite.", InvokeContext(tool_call_id="call-2")
        )

        assert "this run has no scratchpad directory" in prompt
        assert "Do not fall back to ~/.cache" in prompt
        assert "Stop the affected verification" in prompt
        assert "report the unavailable destination" in prompt
        # No scratchpad header is claimed, and the task text still closes the
        # brief.
        assert "Scratchpad directory" not in prompt
        assert prompt.endswith("Run the suite.")

    def test_both_dispatch_paths_route_through_the_brief_seam(self) -> None:
        # The spawned path (AgentChildHost.run_child) and the monitored path
        # (SessionRuntimeRegistry._start_child_turn) both send the same brief,
        # so the discipline reaches workers in either dispatch mode.
        spawned = inspect.getsource(AgentChildHost.run_child)
        monitored = inspect.getsource(SessionRuntimeRegistry._start_child_turn)

        assert "prepare_subagent_prompt(args.task, ctx)" in spawned
        assert "prepare_subagent_prompt(args.task, ctx)" in monitored


class TestTmpPlacementBrief:
    """The temporary-placement discipline rides in the scratchpad brief (C4 chunk ii)."""

    def test_a_scratchpad_brief_carries_the_invocation_dir_discipline(self) -> None:
        prompt = prepare_subagent_prompt(
            "Run the suite.",
            InvokeContext(tool_call_id="call-3", scratchpad_dir=Path("/tmp/sp")),
        )

        assert (
            "own fresh, unique invocation directory under this scratchpad directory"
            in prompt
        )
        assert "basetemp, pytest cache, uv cache, and ruff cache" in prompt
        # An explicit destination under the scratchpad wins and is never reaped.
        assert "takes precedence and is never removed automatically" in prompt
        # A default run lands under the managed root, never the shared tree.
        assert "managed pytest run root" in prompt
        assert "never under the shared /tmp/pytest-of-<user> tree" in prompt
        # Failed-only retention, with the copy-before-teardown debug flow.
        assert "Only failed tests keep their call-phase directories" in prompt
        assert "setup/teardown-phase artifacts are not retained" in prompt
        assert prompt.endswith("Run the suite.")

    def test_a_brief_without_a_scratchpad_has_no_tmp_placement(self) -> None:
        prompt = prepare_subagent_prompt(
            "Run the suite.", InvokeContext(tool_call_id="call-4")
        )

        assert "this run has no scratchpad directory" in prompt
        # Without a scratchpad no destination can be named: the brief stops the
        # verification instead of instructing a placement.
        assert "invocation directory" not in prompt
        assert "managed pytest run root" not in prompt

    def test_repo_pytest_ini_retains_failed_tests_only(self) -> None:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if not pyproject.is_file():
            pytest.skip("repository pyproject.toml not present (installed context)")
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
        ini_options = data["tool"]["pytest"]["ini_options"]
        assert ini_options["tmp_path_retention_policy"] == "failed"

    def test_agents_md_and_safety_guide_carry_the_placement_contract(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        agents_md = repo_root / "AGENTS.md"
        safety = repo_root / "docs" / "guides" / "tools-safety.md"
        if not agents_md.is_file() or not safety.is_file():
            pytest.skip("repository docs not present (installed context)")
        agents_text = agents_md.read_text(encoding="utf-8")
        safety_text = safety.read_text(encoding="utf-8")

        assert "## Temporary-file placement" in agents_text
        assert 'tmp_path_retention_policy = "failed"' in agents_text
        assert "## Managed temporary files" in safety_text
        assert 'tmp_path_retention_policy = "failed"' in safety_text
        assert "never reaped automatically" in safety_text
