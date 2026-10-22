"""Loud, actionable retirement pins for the profile axis (package 16a).

The profile axis is retired: the profile registry and the ``AgentProfile``
runtime surface, the ``AgentProfileLayer`` install path, agents/*.toml
discovery, the legacy ``[agents]``/agent-paths config plumbing, the task
tool's ``agent_type`` bridge, the legacy ``profile`` record field, and the
persisted pre-identity persona formats are gone. These tests pin the
retirement itself:

- the retired modules and names no longer exist (no public aliases);
- retired config fields and environment names fail with specific successor
  guidance that never echoes the supplied value;
- retired on-disk inputs (agents/*.toml, project prompt dirs) are detected
  loudly and never consumed, deleted, or converted;
- old persisted formats fail resume with "start a new session";
- the ML4-style deployed setup fails explicitly at its first gate, with the
  successor identity path still working;
- the utility/compaction prompt surface that shares the prompt machinery is
  NOT retired with the profile axis;
- the dispatch slot/preset/schema modules, the agents-models profile module,
  and the old enum-based prompt surface are gone (package 16b), and a legacy
  ``[dispatch]`` catalog table is reported loudly, never merged.
"""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.app_server._runtime import _resume_envelope
from chartreux.app_server._subagent_records import AgentRecord, RunRecord
from chartreux.app_server.protocol.notifications import AgentSummaryModel
from chartreux.core.agents import AgentManager
from chartreux.core.agents.identities import (
    UnknownIdentityError,
    validate_identity_selection,
)
from chartreux.core.config._source_validation import (
    validate_environment_names,
    validate_source,
)
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.default_orchestrator import build_default_orchestrator
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.dispatch.session import SessionPolicyError, resume_policy_bundle
from chartreux.core.launch_types import LaunchConfig
from chartreux.core.policy.composition import (
    policy_resource_loader,
    user_resource_directories,
)
from chartreux.core.policy.models import ResourceKind
from chartreux.core.prompts import UtilityPrompt
from chartreux.core.session_types import (
    CommittedModelIdentity,
    LaunchMetadataV1,
    LaunchMetadataV2,
    LaunchPersonaV1,
    SessionMetadata,
)
from chartreux.core.subagents import (
    AgentIdentityMismatchError,
    AgentSummary,
    InvalidLaunchConfigError,
    TaskArgs,
)
from chartreux.core.trusted_folders import trusted_folders_manager
from tests.conftest import ConfigBuilder

pytestmark = pytest.mark.asyncio

#: The successor each retired config field's diagnostic must name.
_RETIRED_FIELD_GUIDANCE_FRAGMENTS = {
    "system_prompt_id": "identity composition",
    "agent_paths": "no longer load from any directory",
    "enabled_agents": "identity denylist",
    "disabled_agents": "identity denylist",
}

_RETIRED_CONFIG_FIELDS = (
    "system_prompt_id",
    "agent_paths",
    "enabled_agents",
    "disabled_agents",
)


class TestRetiredModulesAndNames:
    @pytest.mark.parametrize(
        "module",
        [
            "chartreux.agents",
            "chartreux.core.agents.diagnostics",
            "chartreux.core.agents.models",
            "chartreux.core.config.layers.agent_profile",
            "chartreux.core.dispatch.presets",
            "chartreux.core.dispatch.schema",
        ],
    )
    async def test_retired_modules_no_longer_import(self, module: str) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)

    async def test_retired_profile_names_are_gone_from_the_agents_surface(self) -> None:
        # chartreux.core.agents.models is gone (see the module pin above); the
        # retired profile names exist on no surviving surface either — the
        # identity registry (chartreux.core.agents.identities) is the successor.
        agents_pkg = importlib.import_module("chartreux.core.agents")
        identities = importlib.import_module("chartreux.core.agents.identities")
        for surface in (agents_pkg, identities):
            for name in (
                "AgentSafety",
                "AgentType",
                "AgentRegistry",
                "AgentProfile",
                "BUILTIN_SUBAGENTS",
                "WORKER",
                "ADVISOR",
                "REVIEWER",
            ):
                assert not hasattr(surface, name)
        # The successor registry surface is in place.
        assert identities.builtin_identity_names() == (
            "advisor",
            "coder",
            "orchestrator",
            "researcher",
            "reviewer",
        )

    async def test_no_public_profile_aliases_remain_on_the_axis(self) -> None:
        registry = importlib.import_module("chartreux.core.agents.registry")
        assert not hasattr(registry, "apply_profile_overrides")
        assert not hasattr(registry, "AgentRegistry")

        layers = importlib.import_module("chartreux.core.config.layers")
        assert not hasattr(layers, "AgentProfileLayer")

        agents_pkg = importlib.import_module("chartreux.core.agents")
        assert not hasattr(agents_pkg, "AgentRegistry")
        assert agents_pkg.AgentManager is AgentManager

        subagents_module = importlib.import_module("chartreux.core.subagents")
        assert not hasattr(subagents_module, "AgentProfileMismatchError")
        assert not hasattr(subagents_module, "MissingAgentProfileError")
        assert not hasattr(subagents_module, "InvalidLaunchPromptError")
        assert subagents_module.AgentIdentityMismatchError is AgentIdentityMismatchError

    async def test_harness_files_manager_has_no_agent_dir_loaders(self) -> None:
        mgr = HarnessFilesManager(sources=("user",))
        assert not hasattr(mgr, "user_agents_dirs")
        assert not hasattr(mgr, "project_agents_dirs")
        # The loud retirement surface replaces them.
        assert mgr.retired_agent_dirs == []
        assert mgr.retired_project_prompt_dirs == []

    async def test_schema_and_records_lost_their_profile_fields(self) -> None:
        for retired in _RETIRED_CONFIG_FIELDS:
            assert retired not in ChartreuxConfigSchema.model_fields

        run_fields = {f.name for f in dataclasses.fields(RunRecord)}
        assert "profile" not in run_fields
        assert "identity" in run_fields
        agent_fields = {f.name for f in dataclasses.fields(AgentRecord)}
        assert "profile" not in agent_fields
        assert "identity" in agent_fields
        assert "profile" not in AgentSummary.__dataclass_fields__
        assert "profile" not in LaunchMetadataV2.model_fields
        assert "profile" not in AgentSummaryModel.model_fields
        assert "agent_type" not in TaskArgs.model_fields

    async def test_config_loses_the_system_prompt_property_only(
        self, build_config: ConfigBuilder
    ) -> None:
        config = build_config()
        assert not hasattr(config, "system_prompt")
        # The compaction prompt surface is NOT retired with the profile axis.
        assert config.compaction_prompt_id == "compact"


class TestRetiredConfigFields:
    @pytest.mark.parametrize("field", _RETIRED_CONFIG_FIELDS)
    async def test_retired_field_fails_with_successor_guidance(
        self, field: str
    ) -> None:
        source = "user config (/test/config.toml)"
        with pytest.raises(ValidationError) as raised:
            validate_source(
                ChartreuxConfigSchema, {field: "retired-value-abc"}, source=source
            )
        message = str(raised.value)
        assert f"{field} is retired" in message
        assert _RETIRED_FIELD_GUIDANCE_FRAGMENTS[field] in message
        assert source in message
        assert "retired-value-abc" not in message

    @pytest.mark.parametrize(
        "env_name",
        [
            "CHARTREUX_SYSTEM_PROMPT_ID",
            "CHARTREUX_AGENT_PATHS",
            "CHARTREUX_ENABLED_AGENTS",
            "CHARTREUX_DISABLED_AGENTS",
        ],
    )
    async def test_retired_environment_name_fails_with_successor_guidance(
        self, env_name: str
    ) -> None:
        field = env_name.removeprefix("CHARTREUX_").lower()
        with pytest.raises(ValidationError) as raised:
            validate_environment_names(
                ChartreuxConfigSchema,
                {env_name: "retired-value-abc"},
                source="environment",
            )
        message = str(raised.value)
        assert f"{field} is retired" in message
        assert _RETIRED_FIELD_GUIDANCE_FRAGMENTS[field] in message
        assert "retired-value-abc" not in message

    @pytest.mark.parametrize(
        "field,toml",
        [
            ("system_prompt_id", 'system_prompt_id = "orchestrator"\n'),
            ("agent_paths", 'agent_paths = ["/nonexistent/agents"]\n'),
            ("enabled_agents", 'enabled_agents = ["worker"]\n'),
            ("disabled_agents", 'disabled_agents = ["worker"]\n'),
        ],
    )
    async def test_retired_field_in_user_config_fails_at_load(
        self, config_dir: Path, field: str, toml: str
    ) -> None:
        (config_dir / "config.toml").write_text(toml, encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            await build_default_orchestrator()
        message = str(raised.value)
        assert f"{field} is retired" in message
        assert _RETIRED_FIELD_GUIDANCE_FRAGMENTS[field] in message
        assert "config.toml" in message


class TestRetiredFileSurface:
    async def test_user_agents_toml_is_detected_and_never_an_identity(
        self, config_dir: Path
    ) -> None:
        agents_dir = config_dir / "agents"
        agents_dir.mkdir()
        profile = agents_dir / "ml4-worker.toml"
        original = (
            b'role = "route-mistral"\nsystem_prompt_id = "worker-ml4"\nretain = false\n'
        )
        profile.write_bytes(original)

        mgr = HarnessFilesManager(sources=("user",))
        assert mgr.retired_agent_dirs == [agents_dir]

        # The profile file is never consulted as an identity source.
        with pytest.raises(UnknownIdentityError, match="Unknown identity 'ml4-worker'"):
            validate_identity_selection(
                "ml4-worker", user_prompts_dirs=[config_dir / "prompts"]
            )
        assert profile.read_bytes() == original

    async def test_project_agents_and_prompt_dirs_are_detected_never_consumed(
        self, tmp_working_directory: Path
    ) -> None:
        agents_dir = tmp_working_directory / ".chartreux" / "agents"
        agents_dir.mkdir(parents=True)
        profile = agents_dir / "ml4-worker.toml"
        original = b'role = "route-mistral"\n'
        profile.write_bytes(original)
        prompts_dir = tmp_working_directory / ".chartreux" / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "orchestrator.md").write_text("user copy\n", encoding="utf-8")

        trusted_folders_manager.trust_for_session(tmp_working_directory)
        mgr = HarnessFilesManager(sources=("user", "project"))

        assert agents_dir in mgr.retired_agent_dirs
        assert mgr.retired_project_prompt_dirs == [prompts_dir]
        # Project prompt directories are never consumed (the package-16b
        # loader keeps working through the always-empty adapter).
        assert mgr.project_prompts_dirs == []
        assert profile.read_bytes() == original


class TestShadowingDetection:
    async def test_user_prompt_cannot_shadow_a_builtin_identity(
        self, config_dir: Path
    ) -> None:
        user_prompts = config_dir / "prompts"
        user_prompts.mkdir()
        (user_prompts / "orchestrator.md").write_text("user copy\n", encoding="utf-8")

        with pytest.raises(
            UnknownIdentityError, match="cannot shadow built-in identities"
        ):
            validate_identity_selection(
                "user:orchestrator", user_prompts_dirs=[user_prompts]
            )
        # A bare shipped name still resolves to the shipped definition.
        assert (
            validate_identity_selection(
                "orchestrator", user_prompts_dirs=[user_prompts]
            )
            == "orchestrator"
        )

        mgr = HarnessFilesManager(sources=("user",))
        loader = policy_resource_loader(user_resource_directories(mgr))
        resolved = loader.resolve(ResourceKind.IDENTITY, "orchestrator")
        assert resolved.source == "builtin"
        assert resolved.content is not None
        assert "user copy" not in resolved.content


class TestOldPersistedFormats:
    async def test_v1_launch_envelope_is_rejected_with_new_session_guidance(
        self,
    ) -> None:
        envelope = LaunchMetadataV1(
            version=1,
            profile="worker",
            overrides=LaunchConfig(),
            persona=LaunchPersonaV1(system_prompt_id="worker", instructions=None),
        )
        with pytest.raises(
            InvalidLaunchConfigError,
            match="Start a new session.*no migration for the old persisted format",
        ) as raised:
            _resume_envelope(envelope)
        assert raised.value.field == "launch_config.persona"

    async def test_v2_envelope_with_v1_persona_is_rejected_the_same_way(self) -> None:
        envelope = LaunchMetadataV2(
            version=2,
            overrides=LaunchConfig(),
            persona=LaunchPersonaV1(system_prompt_id="worker", instructions=None),
            committed_model=CommittedModelIdentity(
                base_model="glm-5-3",
                provider="mistral",
                wire_name="glm-5-3",
                catalog_revision="rev-1",
            ),
        )
        with pytest.raises(
            InvalidLaunchConfigError,
            match="Start a new session.*no migration for the old persisted format",
        ):
            _resume_envelope(envelope)

    async def test_legacy_envelope_formats_still_parse(self) -> None:
        # The legacy formats are rejected at RESUME, not at parse: a stored
        # old record still loads so the failure is the actionable one.
        v1 = LaunchMetadataV1.model_validate({
            "version": 1,
            "profile": "worker",
            "overrides": {},
            "persona": {"system_prompt_id": "worker", "instructions": None},
        })
        assert v1.profile == "worker"
        v2 = LaunchMetadataV2.model_validate({
            "version": 2,
            "overrides": {},
            "persona": {"system_prompt_id": "worker", "instructions": None},
            "committed_model": {
                "base_model": "m",
                "provider": "p",
                "wire_name": "w",
                "catalog_revision": "r",
            },
        })
        assert isinstance(v2.persona, LaunchPersonaV1)

    async def test_legacy_session_record_without_policy_bundle_fails_new_session(
        self,
    ) -> None:
        metadata = SessionMetadata(
            session_id="s",
            start_time="",
            end_time=None,
            git_commit=None,
            git_branch=None,
            environment={},
            username="tester",
        )
        with pytest.raises(
            SessionPolicyError,
            match="Start a new session.*no migration for the old persisted format",
        ):
            resume_policy_bundle(metadata)


class TestML4StyleSetup:
    """The deployed ML4 prompt-layer strategy (ml4-orchestration-findings §3).

    The setup activated ``prompts/orchestrator.md`` via
    ``system_prompt_id = "orchestrator"`` in config.toml, defined the
    ``ml4-worker`` profile in ``agents/ml4-worker.toml`` (with
    ``system_prompt_id = "worker-ml4"``), and bound the dispatch slot
    ``escalation-implementor`` to ``profile = "ml4-worker"`` in models.toml.
    Package 16a retires every one of those inputs: the setup fails
    explicitly at its first gate, and nothing is migrated.
    """

    async def test_system_prompt_id_activation_fails_at_config_load(
        self, config_dir: Path
    ) -> None:
        (config_dir / "config.toml").write_text(
            'system_prompt_id = "orchestrator"\n', encoding="utf-8"
        )
        with pytest.raises(ValidationError) as raised:
            await build_default_orchestrator()
        message = str(raised.value)
        assert "system_prompt_id is retired" in message
        assert "identity composition" in message
        assert "config.toml" in message

    async def test_deployed_setup_fails_explicitly_and_is_never_migrated(
        self, config_dir: Path
    ) -> None:
        # The deployed file layout, byte-for-byte (all under CHARTREUX_HOME).
        prompts_dir = config_dir / "prompts"
        prompts_dir.mkdir()
        orchestrator_md = prompts_dir / "orchestrator.md"
        orchestrator_md.write_text("# Orchestrator\n", encoding="utf-8")
        worker_md = prompts_dir / "worker-ml4.md"
        worker_md.write_text("# Worker (ML4)\n", encoding="utf-8")
        agents_dir = config_dir / "agents"
        agents_dir.mkdir()
        profile_path = agents_dir / "ml4-worker.toml"
        profile_path.write_bytes(
            b'role = "route-mistral"\nsystem_prompt_id = "worker-ml4"\nretain = false\n'
        )
        models_path = config_dir / "models.toml"
        models_path.write_bytes(
            b'[dispatch.slots.escalation-implementor]\nprofile = "ml4-worker"\n'
        )
        original = {
            path: path.read_bytes()
            for path in (
                config_dir / "config.toml",
                orchestrator_md,
                worker_md,
                profile_path,
                models_path,
            )
        }

        orchestrator = await build_default_orchestrator()
        manager = AgentManager(orchestrator, HarnessFilesManager(sources=("user",)))

        # The legacy [dispatch] table is reported loudly at load with successor
        # guidance and never merged: the slot's profile reference resolves to
        # nothing because the read path is gone (package 16b).
        snapshot = orchestrator.config.catalog_snapshot
        assert snapshot is not None
        assert not hasattr(snapshot, "dispatch")
        diagnostics = "\n".join(snapshot.dispatch_diagnostics)
        assert "The legacy [dispatch] table in" in diagnostics
        assert "models.toml" in diagnostics
        assert "[orchestration.routes.<purpose>]" in diagnostics
        assert "[agent] identity/workflow/strategy selections" in diagnostics
        assert "never merged" in diagnostics
        # The report never echoes the table's content.
        assert "ml4-worker" not in diagnostics
        assert "escalation-implementor" not in diagnostics

        # The profile directory is detected loudly and never consumed: the
        # profile name is not an identity, and the task tool's legacy
        # agent_type bridge rejects it with the successor guidance.
        detected = HarnessFilesManager(sources=("user",))
        assert detected.retired_agent_dirs == [agents_dir]
        with pytest.raises(UnknownIdentityError, match="Unknown identity 'ml4-worker'"):
            validate_identity_selection("ml4-worker", user_prompts_dirs=[prompts_dir])
        with pytest.raises(ValidationError, match="agent_type is retired"):
            TaskArgs.model_validate({"task": "work", "agent_type": "ml4-worker"})

        # A custom prompt cannot shadow a shipped identity, but the deployed
        # worker prompt still resolves as the successor custom identity.
        with pytest.raises(
            UnknownIdentityError, match="cannot shadow built-in identities"
        ):
            validate_identity_selection(
                "user:orchestrator", user_prompts_dirs=[prompts_dir]
            )
        assert (
            validate_identity_selection(
                "user:worker-ml4", user_prompts_dirs=[prompts_dir]
            )
            == "user:worker-ml4"
        )

        # Nothing is migrated: every file is byte-unchanged and no profile is
        # converted into a prompt file.
        for path, content in original.items():
            assert path.read_bytes() == content
        assert sorted(p.name for p in agents_dir.iterdir()) == ["ml4-worker.toml"]
        assert not (prompts_dir / "ml4-worker.md").exists()
        assert manager.config is orchestrator.config


class TestTaskAgentTypeRetired:
    async def test_task_agent_type_fails_with_successor_guidance(self) -> None:
        with pytest.raises(ValidationError) as raised:
            TaskArgs.model_validate({"task": "work", "agent_type": "worker"})
        message = str(raised.value)
        assert "agent_type is retired" in message
        assert "worker -> coder" in message
        assert "identity field" in message

    async def test_task_identity_field_still_works(self) -> None:
        args = TaskArgs(task="work", identity="coder")
        assert args.identity == "coder"

    async def test_launch_config_system_prompt_id_fails_with_successor_guidance(
        self,
    ) -> None:
        with pytest.raises(ValidationError) as raised:
            LaunchConfig.model_validate({"system_prompt_id": "cli"})
        message = str(raised.value)
        assert "system_prompt_id is retired" in message
        assert "identity field" in message


class TestUtilityAndCompactionPromptsSurvive:
    async def test_compact_utility_prompt_and_compaction_prompt_survive(
        self, build_config: ConfigBuilder
    ) -> None:
        compact = UtilityPrompt.COMPACT.read()
        assert compact
        config = build_config()
        assert config.compaction_prompt_id == "compact"
        assert config.compaction_prompt == compact

    async def test_enum_prompt_surface_is_retired_and_utility_prompts_survive(
        self, build_config: ConfigBuilder
    ) -> None:
        # Package 16b retired the enum-based full-skeleton prompt axis
        # (SystemPrompt, load_system_prompt, the cli/worker/... bodies); the
        # utility/compaction surface sharing the prompt machinery is NOT
        # retired with it.
        from chartreux.core import prompts

        assert not hasattr(prompts, "SystemPrompt")
        assert not hasattr(prompts, "load_system_prompt")
        assert not (prompts.PROMPTS_DIR / "cli.md").exists()
        config = build_config()
        assert config.compaction_prompt == UtilityPrompt.COMPACT.read()
