"""Private local policy transactions; server descendant publication is not wired."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop import AgentLoopStateError
from chartreux.core.session_types import SessionMetadata
from chartreux.core.tools.manager import ToolManager
from tests.core.agent_loop.test_accepted_source_snapshot import make_orchestrator
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


async def make_loop(tmp_path: Path) -> AgentLoop:
    path = tmp_path / "settings.toml"
    path.write_text('[tools.read_file]\ndenylist = ["A"]\n')
    orchestrator = await make_orchestrator(path)
    await orchestrator.set_field("/tools/read_file/denylist", ["other"])
    return AgentLoop(
        config_orchestrator=orchestrator, cwd=tmp_path, backend=FakeBackend()
    )


async def replace(loop: AgentLoop, patterns: list[str]) -> None:
    prepared = await loop._prepare_policy_replacement(
        source="user-toml",
        tools={"read_file": {"denylist": patterns}},
        expected_token=loop.config_orchestrator.accepted_token,
    )
    loop._commit_policy_replacement(prepared)


async def test_replacement_not_union_and_source_scoped(tmp_path: Path) -> None:
    loop = await make_loop(tmp_path)
    try:
        for values, expected in [(["B"], {"B", "other"}), ([], {"other"})]:
            old = loop.tool_manager.get("read_file")
            assert not hasattr(loop, "_permission_store")
            await replace(loop, values)
            assert (
                set(loop.tool_manager.get_tool_config("read_file").denylist) == expected
            )
            assert loop.tool_manager.get("read_file") is not old
            assert not hasattr(loop, "_permission_store")
            await loop.reload_with_initial_messages(reload_config=True)
            assert (
                set(loop.tool_manager.get_tool_config("read_file").denylist) == expected
            )
        assert '"A"' in (tmp_path / "settings.toml").read_text()
    finally:
        await loop.aclose()


@pytest.mark.parametrize("failure", ["stale", "prepare", "invalid"])
async def test_failed_replacement_preserves_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    loop = await make_loop(tmp_path)
    config = loop.config
    restrictions = loop.config_orchestrator.restrictions
    manager = loop.tool_manager
    tool = manager.get("read_file")
    token = loop.config_orchestrator.accepted_token
    assert not hasattr(loop, "_permission_store")
    try:

        def reject(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("prepare failed")

        if failure == "prepare":
            monkeypatch.setattr(ToolManager, "get_tool_config", reject)
        with pytest.raises((ValueError, RuntimeError)):
            await loop._prepare_policy_replacement(
                source="user-toml",
                tools={
                    "read_file": {"denylist": [23] if failure == "invalid" else ["B"]}
                },
                expected_token=object() if failure == "stale" else token,
            )
        assert loop.config is config
        assert loop.config_orchestrator.restrictions is restrictions
        assert loop.config_orchestrator.accepted_token is token
        assert loop.tool_manager is manager
        assert manager.get("read_file") is tool
        assert not hasattr(loop, "_permission_store")
    finally:
        await loop.aclose()


async def test_staged_commit_stale_and_child_inheritance(tmp_path: Path) -> None:
    root = await make_loop(tmp_path)
    child = AgentLoop(
        config_orchestrator=root.config_orchestrator.copy(),
        cwd=tmp_path,
        backend=FakeBackend(),
        inherited_restrictions=root.child_runtime_policy.inherited_restrictions,
    )
    try:
        prepared = await root._prepare_policy_replacement(
            source="user-toml",
            tools={},
            expected_token=root.config_orchestrator.accepted_token,
        )
        await root.config_orchestrator.set_field("/auto_compact_threshold", 60000)
        with pytest.raises(ValueError, match="Stale"):
            root._commit_policy_replacement(prepared)
        await replace(child, [])
        assert "A" in child.tool_manager.get_tool_config("read_file").denylist
        await replace(root, [])
        assert "A" not in root.tool_manager.get_tool_config("read_file").denylist
        # Deliberately fixed until the server-owned descendant adoption stage.
        assert "A" in child.tool_manager.get_tool_config("read_file").denylist
    finally:
        await child.aclose()
        await root.aclose()


@pytest.mark.parametrize(
    "field,value", [("permission", "never"), ("sensitive_patterns", ["*.private"])]
)
async def test_complete_replacement_clears_all_source_fields(
    tmp_path: Path, field: str, value: Any
) -> None:
    loop = await make_loop(tmp_path)
    try:
        prepared = await loop._prepare_policy_replacement(
            source="user-toml",
            tools={"read_file": {field: value}},
            expected_token=loop.config_orchestrator.accepted_token,
        )
        loop._commit_policy_replacement(prepared)
        source = next(
            r
            for r in loop.config_orchestrator.restrictions
            if r.layer_name == "user-toml"
        )
        assert source.tools and source.store_fingerprint is None
        assert not source.tools[0].denylist
        await replace(loop, [])
        source = next(
            r
            for r in loop.config_orchestrator.restrictions
            if r.layer_name == "user-toml"
        )
        assert source.tools == ()
        assert "other" in loop.tool_manager.get_tool_config("read_file").denylist
    finally:
        await loop.aclose()


async def test_generic_revoke_rejected_and_no_approval_api_remains(
    tmp_path: Path,
) -> None:
    loop = await make_loop(tmp_path)
    try:
        token = loop.config_orchestrator.accepted_token
        with pytest.raises(ValueError, match="cannot weaken"):
            await loop.config_orchestrator.set_field(
                "/tools/read_file/denylist", [], target_layer="user-toml"
            )
        assert loop.config_orchestrator.accepted_token is token
        assert not hasattr(loop, "set_tool_permission")
        await loop.config_orchestrator.set_field("/tools/todo/permission", "never")
        with pytest.raises(ValueError, match="cannot weaken"):
            await loop.config_orchestrator.set_field("/tools/todo/permission", "always")
    finally:
        await loop.aclose()


@pytest.mark.parametrize("phase", ["prepare", "commit"])
@pytest.mark.parametrize("operation", ["relocate", "plan"])
async def test_held_session_rejects_policy_replacement_without_effects(
    tmp_path: Path, phase: str, operation: str
) -> None:
    loop = await make_loop(tmp_path)
    manager = loop.tool_manager
    tool = manager.get("read_file")
    config = loop.config
    restrictions = loop.config_orchestrator.restrictions
    token = loop.config_orchestrator.accepted_token
    assert not hasattr(loop, "_permission_store")
    try:
        prepared = (
            await loop._prepare_policy_replacement(
                source="user-toml", tools={}, expected_token=token
            )
            if phase == "commit"
            else None
        )
        loop._take_session(operation)
        try:
            with pytest.raises(AgentLoopStateError, match="idle"):
                if prepared is None:
                    await loop._prepare_policy_replacement(
                        source="user-toml", tools={}, expected_token=token
                    )
                else:
                    loop._commit_policy_replacement(prepared)
            assert loop._holders == [operation]
            assert loop.config is config
            assert loop.config_orchestrator.restrictions is restrictions
            assert loop.config_orchestrator.accepted_token is token
            assert loop.tool_manager is manager
            assert manager.get("read_file") is tool
            assert tool.config == manager.get_tool_config("read_file")
            assert not hasattr(loop, "_permission_store")
        finally:
            loop._release_session(operation)
    finally:
        await loop.aclose()


@pytest.mark.parametrize("transition", ["rebind", "same-session-rebind", "reset"])
async def test_session_transition_invalidates_prepared_policy_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str
) -> None:
    old_scratch = tmp_path / "old-scratch"
    new_scratch = tmp_path / "new-scratch"
    old_scratch.mkdir()
    new_scratch.mkdir()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.init_scratchpad", lambda _: old_scratch
    )
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.cleanup_scratchpad", lambda _: None
    )
    loop = await make_loop(tmp_path)
    manager = loop.tool_manager
    token = loop.config_orchestrator.accepted_token
    config = loop.config
    restrictions = loop.config_orchestrator.restrictions
    try:
        manager.get("read_file")
        prepared = await loop._prepare_policy_replacement(
            source="user-toml", tools={}, expected_token=token
        )
        if transition == "reset":
            await loop._reset_session()
        else:
            monkeypatch.setattr(
                "chartreux.core.agent_loop._loop.init_scratchpad", lambda _: new_scratch
            )
            session_id = (
                loop.session_id if transition == "same-session-rebind" else "rebound"
            )
            loop.rebind_to_session(
                session_id,
                tmp_path / "session",
                [],
                session_metadata=SessionMetadata(
                    session_id=session_id,
                    start_time="2026-01-01T00:00:00",
                    end_time=None,
                    git_commit=None,
                    git_branch=None,
                    environment={"working_directory": str(tmp_path)},
                    username="fixture",
                    config={},
                ),
            )
            assert loop.scratchpad_dir == new_scratch
        current_scratch = loop.scratchpad_dir
        current_tool = manager.get("read_file")
        assert not hasattr(loop, "_permission_store")
        # Neither token nor manager identity changed: the session epoch must reject it.
        assert loop.tool_manager is manager
        assert loop.config_orchestrator.accepted_token is token
        with pytest.raises(ValueError, match="Stale policy runtime"):
            loop._commit_policy_replacement(prepared)
        assert loop.tool_manager is manager
        assert manager.get("read_file") is current_tool
        assert current_tool.scratchpad_dir == current_scratch
        assert current_tool.config == manager.get_tool_config("read_file")
        assert loop.scratchpad_dir == current_scratch
        assert loop.config is config
        assert loop.config_orchestrator.restrictions is restrictions
        assert loop.config_orchestrator.accepted_token is token
        assert not hasattr(loop, "_permission_store")
    finally:
        await loop.aclose()
