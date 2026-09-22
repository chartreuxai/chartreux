from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from chartreux.agents import AgentSafety, AgentType
from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._sessions import SessionRuntime, SessionRuntimeRegistry
from chartreux.core.agent_loop._loop import AgentLoop
from chartreux.core.agents.launch import resolve_launch
from chartreux.core.agents.models import AgentProfile
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.models import (
    ModelConfig,
    ProviderConfig,
    SessionLoggingConfig,
)
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.llm.failures import RequestRetryBudget
from chartreux.core.llm_models import LLMChunk, LLMMessage, Role
from chartreux.core.model_catalog.loader import CatalogSnapshot, load_catalog
from chartreux.core.model_catalog.migration import apply_migration, plan_migration
from chartreux.core.model_catalog.resolver import ModelResolutionError, ModelResolver
from chartreux.core.model_catalog.schema import ModelCatalog, RoleDefinition
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.subagents import LaunchConfig, TaskArgs, TaskResult
from chartreux.core.tools.base import InvokeContext
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


def _snapshot(revision: str, *, first_wire: str = "first") -> CatalogSnapshot:
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test/one": {"api_base": "https://one.invalid"},
                "test/two": {"api_base": "https://two.invalid"},
            },
            "models": {
                "first": {
                    "deployments": [
                        {"provider": "test/one", "name": first_wire},
                        {"provider": "test/two", "name": "first-two-wire"},
                    ]
                },
                "later": {
                    "deployments": [{"provider": "test/two", "name": "later-wire"}]
                },
                "compact": {
                    "thinking": "low",
                    "deployments": [
                        {"provider": "test/one", "name": "compact-one"},
                        {"provider": "test/two", "name": "compact-two"},
                    ],
                },
            },
            "roles": {"ordered": {"models": ["first", "later"]}},
        }),
        revision,
    )


async def _orchestrator(
    snapshot: CatalogSnapshot,
) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    layer = OverridesLayer(data={"active_model": "first"})
    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), layer],
        default_layer_resolver=lambda: layer,
        catalog_snapshot=snapshot,
    )


def _profile() -> AgentProfile:
    return AgentProfile(
        name="worker",
        display_name="Worker",
        description="test",
        safety=AgentSafety.NEUTRAL,
        agent_type=AgentType.SUBAGENT,
    )


class _ModelTrackingBackend(FakeBackend):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.requested_models: list[ModelConfig] = []

    async def complete(self, *, model: ModelConfig, **kwargs: object) -> LLMChunk:
        self.requested_models.append(model)
        return await super().complete(model=model, **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_reload_patch_new_child_preserves_old_child_catalog_snapshot() -> None:
    """A root reload is atomic, while child orchestrators retain their captured catalog."""
    revision_a = _snapshot("A", first_wire="a-wire")
    revision_b = _snapshot("B", first_wire="b-wire")
    root = await _orchestrator(revision_a)
    existing_child = root._copy_for_child()
    root._catalog_loader = lambda: revision_b

    await root.reload()
    assert not await root.set_field("/theme", "after-reload", reason="ordinary patch")
    assert root.config.theme == "after-reload"
    new_child = root._copy_for_child()

    assert root.config.catalog_snapshot.revision == "B"
    assert new_child.config.catalog_snapshot.revision == "B"
    assert new_child.config.get_active_model().name == "b-wire"
    assert existing_child.config.catalog_snapshot.revision == "A"
    assert existing_child.config.get_active_model().name == "a-wire"


@pytest.mark.asyncio
async def test_role_bound_child_uses_its_role_instead_of_parent_committed_model() -> (
    None
):
    """A profile role is an explicit model choice, not inherited parent state."""
    snapshot = _snapshot("A")
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "@ordered"}, context={"catalog_snapshot": snapshot}
    ).attach_catalog_snapshot(snapshot)
    parent_loop = build_test_agent_loop(config=config, backend=FakeBackend())
    committed = CommittedModelIdentity(
        base_model="later",
        provider="test/two",
        wire_name="later-wire",
        catalog_revision="A",
    )
    parent_loop.committed_model = committed
    parent_loop.agent_manager._discovered["worker"] = AgentProfile(**{
        **_profile().__dict__,
        "role": "ordered",
    })
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)

    candidate = registry._resolve_launch_candidate(
        cast(SessionRuntime, SimpleNamespace(agent_loop=parent_loop)),
        TaskArgs(task="role-bound", agent="worker", background=True),
    )

    assert candidate.committed_model != committed
    assert (candidate.effective_model.alias, candidate.effective_model.name) == (
        "first",
        "first",
    )
    await parent_loop.aclose()


@pytest.mark.asyncio
async def test_resumed_role_bound_child_reresolves_its_role(tmp_path: Path) -> None:
    """Cross-restart resume follows the current role instead of its saved identity."""
    snapshot = _snapshot("A")
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions"))
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "@ordered", "session_logging": logging.model_dump()},
        context={"catalog_snapshot": snapshot},
    ).attach_catalog_snapshot(snapshot)
    parent = build_test_agent_loop(config=config, backend=FakeBackend())
    profile = AgentProfile(**{**_profile().__dict__, "role": "ordered"})
    parent.agent_manager._discovered["worker"] = profile
    try:
        await parent.persist_empty_session()
        candidate = resolve_launch(
            profile_name="worker",
            config=None,
            parent_orchestrator=parent.config_orchestrator,
            tool_inventory={},
            profile_lookup=lambda _name: profile,
        )
        child = await AgentRuntimeFactory().create_child(parent, candidate)
        await child.wait_until_ready()
        await child.persist_empty_session()
        child_id = child.session_id
        child_dir = child.session_logger.session_dir
        assert child_dir is not None
        await child.aclose()

        updated_catalog = snapshot.catalog.model_copy(
            update={"roles": {"ordered": RoleDefinition(models=("later",))}}
        )
        parent.config.attach_catalog_snapshot(CatalogSnapshot(updated_catalog, "B"))
        resumed = await AgentRuntimeFactory().resume_child(
            parent, "worker", child_id, child_dir
        )
        try:
            assert resumed.config.active_model == "@ordered"
            assert resumed.committed_model is not None
            assert resumed.committed_model.base_model == "later"
        finally:
            await resumed.aclose()
    finally:
        await parent.aclose()


def test_role_thinking_resolves_base_and_rejects_v0_1_expressions() -> None:
    snapshot = _snapshot("A")
    parent = asyncio.run(_orchestrator(snapshot))
    candidate = resolve_launch(
        profile_name="worker",
        config=LaunchConfig(model="@ordered", thinking="low"),
        parent_orchestrator=parent,
        tool_inventory={},
        profile_lookup=lambda _name: _profile(),
    )

    assert (candidate.committed_model.base_model, candidate.effective_model.name) == (
        "first",
        "first",
    )
    assert candidate.effective_thinking == "low"
    resolver = ModelResolver(snapshot)
    for expression in (["first"], "test/one/first"):
        with pytest.raises(ModelResolutionError) as error:
            resolver.expression_bases(expression)  # type: ignore[arg-type]
        assert error.value.code == "invalid_expression"


@pytest.mark.asyncio
async def test_compaction_alias_materializes_destination_deployment_and_thinking() -> (
    None
):
    snapshot = _snapshot("A")
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "first", "compaction_model": "compact"},
        context={"catalog_snapshot": snapshot},
    ).attach_catalog_snapshot(snapshot)
    backend = _ModelTrackingBackend([
        [mock_llm_chunk(content="<summary>done</summary>")]
    ])
    agent = build_test_agent_loop(config=config, backend=backend)
    agent.messages.append(LLMMessage(role=Role.user, content="context"))
    agent.stats.context_tokens = 100

    await agent.compact()

    assert backend.requested_models[0].alias == "compact"
    assert (backend.requested_models[0].provider, backend.requested_models[0].name) == (
        "test/one",
        "compact-one",
    )
    assert backend.requested_models[0].thinking == "low"
    await agent.aclose()


class _GatedFailoverBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, **_kwargs: object) -> LLMChunk:
        self.started.set()
        await self.release.wait()
        raise httpx.ConnectError("first provider unavailable")


class _BlockingCompletionBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, **_kwargs: object) -> LLMChunk:
        self.started.set()
        await self.release.wait()
        return mock_llm_chunk(content="child completed")


class _SiblingCompletionBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__([
            mock_llm_chunk(content="sibling completed"),
            mock_llm_chunk(content="failed sibling recovered"),
        ])
        self.completed_bases: list[str | None] = []
        self.later_completed = asyncio.Event()

    async def complete(self, *, model: ModelConfig, **kwargs: object) -> LLMChunk:
        self.completed_bases.append(model.alias)
        result = await super().complete(model=model, **kwargs)  # type: ignore[arg-type]
        if model.alias == "later":
            self.later_completed.set()
        return result


async def _fan_out_result(
    registry: SessionRuntimeRegistry, args: TaskArgs, context: InvokeContext
) -> TaskResult:
    return cast(TaskResult, [event async for event in registry.run(args, context)][-1])


@pytest.mark.asyncio
async def test_failover_terminal_identity_survives_eviction_and_result_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real fan-out preserves the terminal failover identity after eviction."""
    snapshot = _snapshot("A")
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            **snapshot.catalog.model_dump(),
            "roles": {
                **snapshot.catalog.model_dump()["roles"],
                "single": {"models": ["first"]},
            },
        }),
        snapshot.revision,
    )
    config = build_test_vibe_config(active_model="first").attach_catalog_snapshot(
        snapshot
    )
    failed = _GatedFailoverBackend()

    def create_backend(*, provider: ProviderConfig, **_kwargs: object) -> FakeBackend:
        return (
            failed
            if provider.name == "test/one"
            else FakeBackend([mock_llm_chunk(content="completed after failover")])
        )

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    parent = build_test_agent_loop(config=config, backend=FakeBackend())
    parent.agent_manager._discovered["worker"] = _profile()
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    registry._retention_policy = (0, 0)
    try:
        task = asyncio.create_task(
            _fan_out_result(
                registry,
                TaskArgs(
                    task="fail over",
                    fan_out=True,
                    background=False,
                    config=LaunchConfig(model="@single"),
                ),
                InvokeContext(tool_call_id="fan-out", session_id=parent.session_id),
            )
        )
        await asyncio.wait_for(failed.started.wait(), timeout=1)
        failed.release.set()
        result = await asyncio.wait_for(task, timeout=1)

        assert result.completed and result.members is not None
        member = result.members[0]
        assert (member.base_model, member.provider, member.display_name) == (
            "first",
            "test/two",
            "test/two/first-two-wire",
        )
        assert member.agent_id is not None and member.run_id is not None
        assert registry._agent_records == {}
        delivered = await registry.wait_for_agent(member.agent_id, member.run_id)
        serialized = delivered.model_dump(mode="json")
        assert TaskResult.model_validate(serialized) == delivered
        assert serialized["metadata"]["providers_used"] == [["test/one", "test/two"]]
    finally:
        failed.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_fan_out_sibling_survives_failover_and_shares_root_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing fan-out member neither cancels its sibling nor owns its cooldown."""
    snapshot = _snapshot("A")
    config = build_test_vibe_config(active_model="first").attach_catalog_snapshot(
        snapshot
    )
    failed = _GatedFailoverBackend()
    sibling = _SiblingCompletionBackend()

    def create_backend(*, provider: ProviderConfig, **_kwargs: object) -> FakeBackend:
        return failed if provider.name == "test/one" else sibling

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    parent = build_test_agent_loop(config=config, backend=FakeBackend())
    parent.agent_manager._discovered["worker"] = _profile()
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    try:
        task = asyncio.create_task(
            _fan_out_result(
                registry,
                TaskArgs(
                    task="run siblings",
                    fan_out=True,
                    background=False,
                    config=LaunchConfig(model="@ordered"),
                ),
                InvokeContext(tool_call_id="fan-out", session_id=parent.session_id),
            )
        )
        await asyncio.wait_for(failed.started.wait(), timeout=1)
        await asyncio.wait_for(sibling.later_completed.wait(), timeout=1)
        assert not task.done()
        assert sibling.completed_bases == ["later"]

        failed.release.set()
        result = await asyncio.wait_for(task, timeout=1)

        root_registry = parent.config_orchestrator._availability_registry
        assert root_registry.cooldown_until("first", "test/one") is not None
        assert all(
            record.runtime.agent_loop.config_orchestrator._availability_registry
            is root_registry
            for record in registry._agent_records.values()
        )
        assert result.completed and result.members is not None
        assert {
            (member.base_model, member.provider, member.display_name)
            for member in result.members
        } == {
            ("first", "test/two", "test/two/first-two-wire"),
            ("later", "test/two", "test/two/later-wire"),
        }
        assert sibling.completed_bases == ["later", "first"]
    finally:
        failed.release.set()
        await registry.drain_children()
        await parent.aclose()


_LEGACY_CONFIG = b"""active_model = "friendly"\n[[providers]]\nname = "test"\napi_base = "https://example.test/v1"\n[[models]]\nname = "base"\nprovider = "test"\nalias = "friendly"\n"""


@pytest.mark.asyncio
async def test_migration_legacy_unknown_cost_policy_round_trips_through_stats(
    tmp_path: Path,
) -> None:
    """Migrated catalogs and legacy session resumes keep cost incompleteness durable."""
    config_path = tmp_path / "config.toml"
    catalog_path = tmp_path / "models.toml"
    config_path.write_bytes(_LEGACY_CONFIG)
    apply_migration(plan_migration(config_path, catalog_path))
    migrated_catalog = load_catalog(catalog_path)
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions"))
    config = build_test_vibe_config(
        active_model="base", session_logging=logging
    ).attach_catalog_snapshot(migrated_catalog)
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    session_id = saved.session_id
    session_dir = saved.session_logger.session_dir
    assert session_dir is not None
    await saved.aclose()

    metadata_path = session_dir / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["launch_config"] = {
        "version": 1,
        "profile": "legacy",
        "overrides": {},
        "persona": {"system_prompt_id": "tests", "instructions": None},
    }
    metadata_path.write_text(json.dumps(metadata))

    resumed = build_test_agent_loop(config=config)
    try:
        await AgentRuntimeFactory().resume_root(resumed, session_id)
        assert resumed.committed_model is not None
        assert (
            resumed.committed_model.base_model,
            resumed.committed_model.provider,
            resumed.committed_model.wire_name,
        ) == ("base", "test/default", "base")
        assert (resumed.stats.has_unknown_cost, resumed.stats.known_cost_total) == (
            True,
            0.0,
        )
        await resumed.persist_empty_session()
    finally:
        await resumed.aclose()

    reloaded = build_test_agent_loop(config=config)
    try:
        await AgentRuntimeFactory().resume_root(reloaded, session_id)
        assert reloaded.committed_model == resumed.committed_model
        assert (reloaded.stats.has_unknown_cost, reloaded.stats.known_cost_total) == (
            True,
            0.0,
        )
    finally:
        await reloaded.aclose()


@pytest.mark.asyncio
async def test_reload_while_background_child_runs_keeps_child_snapshot_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retained child launched through the registry keeps its launch snapshot."""
    root_orchestrator = await _orchestrator(_snapshot("A", first_wire="a-wire"))
    parent = AgentLoop(
        config_orchestrator=root_orchestrator,
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )
    parent.agent_manager._discovered["worker"] = _profile()
    child_backend = _BlockingCompletionBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    runtime = registry._build_child_runtime(parent)
    runtime.turns._projector = MagicMock()
    runtime.turns.link_subagent = AsyncMock()
    registry.bind_root(runtime)
    try:
        launch = await _fan_out_result(
            registry,
            TaskArgs(task="snapshot", agent="worker", background=True),
            InvokeContext(tool_call_id="background", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        await asyncio.wait_for(child_backend.started.wait(), timeout=1)
        root_orchestrator._catalog_loader = lambda: _snapshot("B", first_wire="b-wire")
        await root_orchestrator.reload()
        child_backend.release.set()
        await registry.wait_for_agent(launch.agent_id, launch.run_id)

        child = registry._agent_records[launch.agent_id].runtime.agent_loop
        assert root_orchestrator.config.get_active_model().name == "b-wire"
        assert child.config.catalog_snapshot is not None
        assert child.config.catalog_snapshot.revision == "A"
        assert child.config.get_active_model().name == "a-wire"
    finally:
        child_backend.release.set()
        await registry.drain_children()
        await parent.aclose()


class _ClosingFakeBackend(FakeBackend):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.closes = 0

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.closes += 1


@pytest.mark.asyncio
async def test_repeated_attempt_lifecycle_closes_once_and_does_not_extend_deadline() -> (
    None
):
    snapshot = _snapshot("A")
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "first"}, context={"catalog_snapshot": snapshot}
    ).attach_catalog_snapshot(snapshot)
    first = _ClosingFakeBackend(exception_to_raise=httpx.ConnectError("down"))
    second = _ClosingFakeBackend([
        [mock_llm_chunk(content="first completion")],
        [mock_llm_chunk(content="second completion")],
    ])
    agent = build_test_agent_loop(config=config, backend=first)
    deadlines: list[float] = []

    def backend_for_attempt(
        _self: AgentLoop, model: ModelConfig, budget: RequestRetryBudget
    ) -> FakeBackend:
        deadlines.append(budget.deadline)
        return {"test/one": first, "test/two": second}[model.provider]

    agent._backend_for_attempt = backend_for_attempt.__get__(agent, AgentLoop)  # type: ignore[method-assign]
    before = len(os.listdir("/proc/self/fd"))
    assert (await agent._chat()).message.content == "first completion"
    assert (await agent._chat()).message.content == "second completion"
    await agent.aclose()

    assert len(deadlines) == 3
    assert deadlines[0] < deadlines[1]
    assert deadlines[2] >= deadlines[1]
    assert len(os.listdir("/proc/self/fd")) - before <= 2
    assert (first.closes, second.closes) == (1, 1)
