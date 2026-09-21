"""Factory source ownership, distinct from equal effective restriction values."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config._restrictions import (
    SourceRestrictions,
    partition_policy_sources,
)
from chartreux.core.config.layer import RawConfig
from chartreux.core.config.patch import AddOperationPatch
from tests.core.agent_loop.test_accepted_source_snapshot import make_orchestrator
from tests.core.agent_loop.test_policy_replacement import replace as replace_policy
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


def source(loop: AgentLoop, name: str = "user-toml") -> SourceRestrictions:
    return next(
        r for r in loop.config_orchestrator.restrictions if r.layer_name == name
    )


def lineage(loop: AgentLoop) -> tuple[SourceRestrictions, ...]:
    return (
        loop.runtime_policy.inherited_restrictions
        + loop.runtime_policy.inherited_mode_restrictions
        + loop.config_orchestrator.restrictions
    )


@pytest.mark.parametrize("reassert", [False, True])
async def test_factory_copy_and_equal_assertion_have_distinct_owners(
    tmp_path: Path, reassert: bool
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[tools.read_file]\ndenylist = ["A"]\n')
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    loops: list[AgentLoop] = []
    try:
        factory = AgentRuntimeFactory()
        child = await factory.create_child(parent, "worker")
        loops.append(child)
        await child.wait_until_ready()
        parent_source = source(parent)
        assert source(child).identity == parent_source.identity
        assert (
            child.config_orchestrator.policy_owner
            != parent.config_orchestrator.policy_owner
        )
        own, inherited = partition_policy_sources(
            lineage(child), owner=child.config_orchestrator.policy_owner
        )
        assert not any(r.layer_name == "user-toml" for r in own)
        assert parent_source in inherited
        if reassert:
            await replace_policy(child, ["A"])
            assert source(child).tools == parent_source.tools
            assert source(child).identity != parent_source.identity
            child_identity = source(child).identity
            assert child_identity is not None
            assert child_identity.owner == child.config_orchestrator.policy_owner
        child_source = source(child)
        grandchild = await factory.create_child(child, "worker")
        loops.append(grandchild)
        await grandchild.wait_until_ready()
        assert source(grandchild).identity == child_source.identity
        own, inherited = partition_policy_sources(
            lineage(grandchild), owner=grandchild.config_orchestrator.policy_owner
        )
        assert not any(r.layer_name == "user-toml" for r in own)
        assert child_source in inherited

        await replace_policy(parent, [])
        assert source(parent).identity == parent_source.identity
        assert source(parent).tools == ()
        # Upcoming registry transaction can remove exactly this opaque key.
        # This is extraction evidence, not a tree commit or runtime revocation.
        retained = tuple(
            r for r in lineage(grandchild) if r.identity != parent_source.identity
        )
        assert (
            any(r.layer_name == "user-toml" and r.tools for r in retained) == reassert
        )
        if reassert:
            await replace_policy(child, [])
            assert source(child).identity == child_source.identity
            assert source(child).tools == ()
            retained = tuple(r for r in retained if r.identity != child_source.identity)
            assert not any(r.layer_name == "user-toml" and r.tools for r in retained)
        assert "A" in child.tool_manager.get_tool_config("read_file").denylist
        assert path.read_text() == '[tools.read_file]\ndenylist = ["A"]\n'
    finally:
        for loop in reversed(loops):
            await loop.aclose()
        await parent.aclose()


async def test_preview_and_staging_do_not_claim_live_copied_sources(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[tools.read_file]\ndenylist = ["A"]\n')
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    child = None
    try:
        child = await AgentRuntimeFactory().create_child(parent, "worker")
        await child.wait_until_ready()
        orchestrator = child.config_orchestrator
        before = orchestrator.restrictions
        owner = orchestrator.policy_owner
        candidate = await orchestrator.preview_candidate(
            layer_overrides={
                "user-toml": RawConfig.model_validate({
                    "tools": {"read_file": {"denylist": ["preview"]}}
                })
            }
        )
        assert (
            next(
                r for r in candidate.restrictions if r.layer_name == "user-toml"
            ).identity
            == source(parent).identity
        )
        copied = orchestrator.copy()
        assert copied.policy_owner == owner
        assert copied.restrictions == before
        staged = await orchestrator._stage_policy_replacement(
            source="user-toml",
            tools={"read_file": {"denylist": ["A"]}},
            expected_token=orchestrator.accepted_token,
        )
        assert (
            next(r for r in staged.restrictions if r.layer_name == "user-toml").identity
            != source(parent).identity
        )
        assert orchestrator.restrictions is before
        assert orchestrator.policy_owner == owner
        # A fresh session-only assertion has its own source, even if equal to
        # a restriction inherited from a differently named parent source.
        prepared = await child._prepare_policy_replacement(
            source="overrides",
            tools={"read_file": {"denylist": ["A"]}},
            expected_token=orchestrator.accepted_token,
        )
        child._commit_policy_replacement(prepared)
        local = source(child, "overrides")
        assert local.identity is not None and local.identity.owner == owner
        assert local.identity != source(child).identity
        own, inherited = partition_policy_sources(lineage(child), owner=owner)
        assert local in own and source(child) in inherited
        before = orchestrator.restrictions
        token = orchestrator.accepted_token
        errors = await orchestrator.set_field(
            "/tools/read_file/denylist", ["A"], target_layer="user-toml"
        )
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert str(errors[0]) == "Child sessions cannot persist configuration"
        assert orchestrator.accepted_token is token
        assert path.read_text() == '[tools.read_file]\ndenylist = ["A"]\n'
        with pytest.raises(ValueError, match="explicit policy replacement"):
            await orchestrator._preview_patch([
                AddOperationPatch(
                    path="/tools/read_file/denylist",
                    value=["A"],
                    target_layer_name="user-toml",
                )
            ])
        assert orchestrator.restrictions is before
        with pytest.raises(ValueError, match="provenance unavailable"):
            partition_policy_sources(
                (replace(source(child), identity=None),), owner=owner
            )
    finally:
        if child is not None:
            await child.aclose()
        await parent.aclose()
