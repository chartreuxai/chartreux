"""The identity-surface agent manager and its retired legacy adapters (16a/16b).

The profile registry and the AgentProfile surface are retired with the
profile axis (package 16a): agents/*.toml discovery, profile lookup, and
the enabled/disabled agent filters are gone, and custom identities resolve
from the user prompts directory through the package-1 strict loader. What
remains is the loop's identity-surface manager — configuration and launch
facts, and loud detection of retired on-disk inputs (never consumed, never
deleted or converted). Package 16b retired the dispatch axis outright: the
shipped profile-name mapping, its identity-keyed inverse, and the dispatch
activation validation are gone, and a legacy ``[dispatch]`` table in
``models.toml`` is reported loudly with successor guidance, never merged.
Review finding 3 removed the last adapter too: the legacy profile-spelling
bridge is gone from the manager, and a legacy ``worker`` reference fails
loudly with identity successor guidance — never a profile path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from pydantic import ValidationError
import pytest

from chartreux.core.agents.identity_launch import InvalidChildIdentityError
from chartreux.core.agents.manager import (
    AgentManager,
    identity_denylist_keys,
    requested_identity_reference,
    resolve_identity_candidate,
)
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.model_catalog.loader import load_catalog
from chartreux.core.subagents import TaskArgs
from chartreux.core.trusted_folders import trusted_folders_manager
from tests.conftest import ConfigBuilder, OrchestratorLoader


class TestLegacyProfileIdentityRetired:
    """The legacy profile-spelling bridge is gone (review finding 3).

    Package 11b bridged a legacy profile spelling onto its identity at the
    task boundary; the alias is retired with the profile axis, so the
    manager carries no bridge and a legacy ``worker`` reference fails
    loudly with identity successor guidance — never a profile path.
    """

    def test_bridge_is_gone_from_the_manager(self) -> None:
        from chartreux.core.agents import manager

        assert not hasattr(manager, "legacy_profile_identity")

    def test_task_boundary_rejects_the_legacy_field_with_successor_guidance(
        self,
    ) -> None:
        with pytest.raises(ValidationError) as raised:
            TaskArgs.model_validate({"task": "work", "agent_type": "worker"})
        message = str(raised.value)
        assert "agent_type is retired" in message
        assert "worker -> coder" in message
        assert "identity field" in message

    def test_legacy_worker_identity_fails_loudly_in_the_canonical_grammar(
        self,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> None:
        # A legacy "worker" spelling is not an identity: the canonical
        # resolution rejects it with the successor identity guidance (no
        # bridge maps the spelling onto "coder").
        orchestrator = load_orchestrator(build_config())
        with pytest.raises(
            InvalidChildIdentityError, match="Unknown identity 'worker'"
        ):
            resolve_identity_candidate(
                TaskArgs(task="work", identity="worker"),
                parent_orchestrator=orchestrator,
                tool_inventory={},
                authorized_tool_names=frozenset(),
            )


class TestRetiredDispatchAdapters:
    """The dispatch adapters retired with the 16b dispatch axis."""

    def test_dispatch_adapters_are_gone_from_the_manager(self) -> None:
        from chartreux.core.agents import manager

        for retired in (
            "SHIPPED_LEGACY_PROFILE_NAMES",
            "legacy_dispatch_profiles",
            "slot_profile_for_identity",
            "validate_dispatch_activation",
            "IDENTITY_SLOT_PROFILES",
        ):
            assert not hasattr(manager, retired), retired


class TestIdentityDenylistKeys:
    """The denylist keys are symmetric across spellings (review finding 1).

    A denylist pattern written in any equivalent spelling blocks the
    identity in every spelling: a ``user:``-qualified custom identity also
    matches its bare name, a ``builtin:``-qualified identity also matches
    its bare name, a bare built-in also matches its ``builtin:``-qualified
    form, and a bare custom id also matches its ``user:``-qualified form.
    """

    @pytest.mark.parametrize(
        ("identity", "keys"),
        [
            ("user:foo", ("user:foo", "foo")),
            ("builtin:coder", ("builtin:coder", "coder")),
            ("coder", ("coder", "builtin:coder")),
            ("researcher", ("researcher", "builtin:researcher")),
            ("advisor", ("advisor", "builtin:advisor")),
            ("reviewer", ("reviewer", "builtin:reviewer")),
            ("orchestrator", ("orchestrator", "builtin:orchestrator")),
            ("foo", ("foo", "user:foo")),
        ],
    )
    def test_equivalent_spellings_match_each_other(
        self, identity: str, keys: tuple[str, ...]
    ) -> None:
        assert identity_denylist_keys(identity) == keys

    @pytest.mark.parametrize(
        ("denied_pattern", "identity"),
        [
            ("coder", "builtin:coder"),
            ("builtin:coder", "coder"),
            ("user:foo", "foo"),
            ("foo", "user:foo"),
        ],
        ids=[
            "bare-builtin-pattern-blocks-builtin-qualified",
            "builtin-qualified-pattern-blocks-bare-builtin",
            "user-qualified-pattern-blocks-bare-custom",
            "bare-custom-pattern-blocks-user-qualified",
        ],
    )
    def test_denying_any_spelling_blocks_every_spelling(
        self, denied_pattern: str, identity: str
    ) -> None:
        # The admission site compares every returned key against each
        # denylist pattern, so a denied spelling is among the identity's
        # equivalent spellings (the four directions of finding 1).
        assert denied_pattern in identity_denylist_keys(identity)


class TestRequestedIdentityReference:
    def test_structured_identity_wins(self) -> None:
        args = TaskArgs(task="do work", identity="researcher")
        assert requested_identity_reference(args) == "researcher"

    def test_model_shorthand_identity_component_names_the_target(self) -> None:
        args = TaskArgs(task="do work", model="@fast:researcher")
        assert requested_identity_reference(args) == "researcher"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"task": "do work"},
            {"task": "do work", "agent_id": "agent-7", "background": True},
            {"task": "do work", "model": "@fast"},
        ],
        ids=["no-target", "retained-continuation", "model-without-identity"],
    )
    def test_calls_without_a_launch_target_name_none(self, kwargs: dict) -> None:
        assert requested_identity_reference(TaskArgs(**kwargs)) is None


class TestAgentManager:
    @pytest.fixture
    def manager(
        self,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> AgentManager:
        return AgentManager(load_orchestrator(build_config()))

    def test_manager_exposes_the_orchestrator_config(
        self,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> None:
        orchestrator = load_orchestrator(build_config())
        manager = AgentManager(orchestrator)
        assert manager.config is orchestrator.config

    def test_profile_lookup_surface_is_retired(self, manager: AgentManager) -> None:
        # The profile registry is gone: no profile lookup or launch roster.
        assert not hasattr(manager, "get_agent")
        assert not hasattr(manager, "get_subagents")

    def test_retired_agent_dirs_are_reported_loudly_and_untouched(
        self,
        config_dir: Path,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        agents_dir = config_dir / "agents"
        agents_dir.mkdir()
        profile_file = agents_dir / "old-profile.toml"
        original = b'display_name = "Old Profile"\n'
        profile_file.write_bytes(original)
        warning = Mock()
        monkeypatch.setattr("chartreux.core.agents.manager.logger.warning", warning)

        orchestrator = load_orchestrator(build_config())
        manager = AgentManager(orchestrator, HarnessFilesManager(sources=("user",)))

        assert manager.config is orchestrator.config
        # The logger is mocked before %-formatting, so rebuild each message.
        messages = [
            call.args[0] % call.args[1:] if call.args else ""
            for call in warning.call_args_list
        ]
        assert any(
            "Agent profiles are retired and no longer load" in message
            and str(agents_dir) in message
            and "task tool's identity field" in message
            for message in messages
        )
        # The profile file is detected, never deleted or converted.
        assert profile_file.read_bytes() == original

    def test_retired_project_prompt_dirs_are_reported_loudly_and_untouched(
        self,
        tmp_working_directory: Path,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        prompts_dir = tmp_working_directory / ".chartreux" / "prompts"
        prompts_dir.mkdir(parents=True)
        prompt_file = prompts_dir / "custom.md"
        original = b"# custom prompt\n"
        prompt_file.write_bytes(original)
        trusted_folders_manager.trust_for_session(tmp_working_directory)
        warning = Mock()
        monkeypatch.setattr("chartreux.core.agents.manager.logger.warning", warning)

        orchestrator = load_orchestrator(build_config())
        manager = AgentManager(
            orchestrator, HarnessFilesManager(sources=("user", "project"))
        )

        assert manager.config is orchestrator.config
        # The logger is mocked before %-formatting, so rebuild each message.
        messages = [
            call.args[0] % call.args[1:] if call.args else ""
            for call in warning.call_args_list
        ]
        assert any(
            "Project-level custom prompt directories are retired" in message
            and str(prompts_dir) in message
            and "user prompts directory" in message
            for message in messages
        )
        assert prompt_file.read_bytes() == original

    def test_legacy_dispatch_report_is_projected_and_recomputed(
        self,
        tmp_path: Path,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> None:
        from chartreux.app_server._projection import project_config_view

        path = tmp_path / "models.toml"
        path.write_text('[dispatch]\nmode = "standalone"\n')
        reported = load_catalog(path)
        assert len(reported.dispatch_diagnostics) == 1
        report = reported.dispatch_diagnostics[0]
        assert "[dispatch]" in report and "retired" in report

        orchestrator = load_orchestrator(build_config())
        snapshot = orchestrator.config.catalog_snapshot
        assert snapshot is not None
        orchestrator.config.attach_catalog_snapshot(reported)
        # The loud report rides the catalog notes into the served
        # configuration's validation warnings and the client projection.
        assert report in orchestrator.config.validation_warnings
        assert report in project_config_view(orchestrator.config).validation_warnings
        # Recomputing from a clean snapshot drops the report.
        orchestrator.config.attach_catalog_snapshot(snapshot)
        assert report not in orchestrator.config.validation_warnings

    def test_legacy_dispatch_table_and_profile_file_are_reported_never_consumed(
        self,
        config_dir: Path,
        tmp_path: Path,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> None:
        """A legacy ``[dispatch]`` table naming a profile that only exists as
        agents/*.toml is reported loudly; the profile file is detected as
        retired input and never consumed or migrated into an identity.
        """
        agents_dir = config_dir / "agents"
        agents_dir.mkdir()
        profile_file = agents_dir / "custom.toml"
        original = b'role = "scout"\n'
        profile_file.write_bytes(original)

        path = tmp_path / "models.toml"
        path.write_text('[dispatch.slots.mechanical]\nprofile = "custom"\n')
        snapshot = load_catalog(path)

        orchestrator = load_orchestrator(build_config())
        orchestrator.config.attach_catalog_snapshot(snapshot)
        AgentManager(orchestrator, HarnessFilesManager(sources=("user",)))

        # The retired table is reported with successor guidance; the catalog
        # it sits beside loads normally and the report reaches the config.
        assert len(snapshot.dispatch_diagnostics) == 1
        report = snapshot.dispatch_diagnostics[0]
        assert "[dispatch]" in report and "retired" in report
        assert "[orchestration.routes.<purpose>]" in report
        assert report in orchestrator.config.validation_warnings
        # The profile file is detected as retired input, never consumed.
        assert profile_file.read_bytes() == original

    def test_legacy_dispatch_load_never_repairs_and_serves_the_catalog(
        self,
        tmp_path: Path,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> None:
        path = tmp_path / "models.toml"
        original = (
            '[dispatch.slots.mechanical]\nprofile = "missing-profile"\n'
            '[providers.mistral]\napi_base = "https://preserved.example/v1"\n'
        )
        path.write_text(original)
        snapshot = load_catalog(path)

        orchestrator = load_orchestrator(build_config())
        orchestrator.config.attach_catalog_snapshot(snapshot)
        AgentManager(orchestrator)

        # The retired table is reported loudly; the load serves the catalog
        # beside it and never repairs or rewrites the file.
        assert len(snapshot.dispatch_diagnostics) == 1
        report = snapshot.dispatch_diagnostics[0]
        assert "[dispatch]" in report and "retired" in report
        assert report in orchestrator.config.validation_warnings
        assert (
            snapshot.catalog.providers["mistral"].api_base
            == "https://preserved.example/v1"
        )
        assert path.read_text() == original
