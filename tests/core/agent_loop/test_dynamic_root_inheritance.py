"""Child inheritance and child-originated session root grants."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from chartreux.core.agent_loop import AgentLoop, AgentRuntimePolicy
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.events import BaseEvent, ToolResultEvent, UserInputRequestEvent
from chartreux.core.llm_models import FunctionCall, LLMChunk, Role, ToolCall
from chartreux.questions import UserAnswer, UserQuestionResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio

VISION_MODEL = ModelConfig(
    name="vision", provider="mistral", alias="vision", supports_images=True
)


def _config() -> ChartreuxConfigSchema:
    return build_test_vibe_config(active_model="vision", models=[VISION_MODEL])


def _read_call(target: Path, call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="read_file", arguments=json.dumps({"file_path": str(target)})
        ),
    )


def _read_turn(target: Path, call_id: str) -> list[list[LLMChunk]]:
    """One scripted turn: a read_file call stream, then a closing stream."""
    return [
        [mock_llm_chunk(tool_calls=[_read_call(target, call_id)])],
        [mock_llm_chunk(content="done")],
    ]


async def _drive(
    agent: AgentLoop, answer: str | None = None
) -> tuple[list[BaseEvent], list[UserInputRequestEvent]]:
    """Run one turn, answering every root-grant prompt with *answer*."""
    events: list[BaseEvent] = []
    requests: list[UserInputRequestEvent] = []
    async with asyncio.timeout(10):
        async for event in agent.act("Exercise root grants"):
            if isinstance(event, UserInputRequestEvent):
                requests.append(event)
                agent.resolve_user_input_request(
                    event.request_id,
                    UserQuestionResult(
                        answers=[UserAnswer(question="grant", answer=answer or "")]
                    ),
                )
            events.append(event)
    return events, requests


def _results(events: list[BaseEvent]) -> list[ToolResultEvent]:
    return [event for event in events if isinstance(event, ToolResultEvent)]


def _tool_texts(loop: AgentLoop) -> list[str]:
    return [
        message.content or "" for message in loop.messages if message.role == Role.tool
    ]


class _SessionGrantPort:
    """Registry-style port: resolves loops by session id, applies locally."""

    def __init__(self) -> None:
        self._loops: dict[str, AgentLoop] = {}

    def register(self, loop: AgentLoop) -> None:
        self._loops[loop.session_id] = loop

    async def grant_root(self, session_id: str, root: Path) -> None:
        loop = self._loops.get(session_id)
        if loop is None:
            raise RuntimeError(
                f"Root grant requires a registered session: {session_id}"
            )
        loop.apply_root_grant(root)

    async def save_root(
        self, session_id: str, root: Path, expected_revision: str
    ) -> ConfigSaveResult:
        raise AssertionError("These tests never select 'Always for this project'")


def _child_loop(
    parent: AgentLoop,
    policy: AgentRuntimePolicy,
    backend: FakeBackend,
    *,
    cwd: Path | None = None,
) -> AgentLoop:
    """Build a child the way the runtime factory does, from a captured policy."""
    return AgentLoop(
        config_orchestrator=parent.config_orchestrator.copy(),
        cwd=cwd or parent.cwd,
        backend=backend,
        is_subagent=True,
        inherited_workspace=policy.inherited_workspace,
        inherited_restrictions=policy.inherited_restrictions,
        inherited_mode_restrictions=policy.inherited_mode_restrictions,
        inherited_plan_write_scopes=policy.inherited_plan_write_scopes,
        inherited_root_grants=policy.inherited_root_grants,
        user_input_capability=policy.user_input_capability,
        parent_authority_getter=policy.parent_authority_getter,
        parent_authority_revision_getter=policy.parent_authority_revision_getter,
    )


async def test_child_dispatched_after_a_parent_grant_inherits_it(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    parent = build_test_agent_loop(
        config=_config(), cwd=project, backend=FakeBackend(), user_input_capability=True
    )
    try:
        early_policy = parent.child_runtime_policy
        assert early_policy.inherited_root_grants == ()
        before = _child_loop(
            parent,
            early_policy,
            FakeBackend([
                *_read_turn(target, "early"),
                *_read_turn(target, "early-retry"),
            ]),
        )
        try:
            events, requests = await _drive(before, "Allow this session")
            assert requests == []
            assert _results(events)[0].skipped
            assert before._session_root_grants.roots == set()

            parent.apply_root_grant(outside)
            assert parent._session_root_grants.roots == {outside.resolve()}

            after_policy = parent.child_runtime_policy
            assert after_policy.inherited_root_grants == (outside.resolve(),)
            after = _child_loop(
                parent, after_policy, FakeBackend([*_read_turn(target, "late")])
            )
            try:
                assert after._session_root_grants.roots == {outside.resolve()}
                assert after.tool_manager.workspace.allows(target)
                events, requests = await _drive(after, "Allow this session")
                assert requests == []
                results = _results(events)
                assert len(results) == 1
                assert not results[0].skipped
                assert results[0].result is not None
                assert "granted content" in results[0].result.model_dump_json()
            finally:
                await after.aclose()

            # The child launched before the grant keeps its frozen authority.
            events, requests = await _drive(before, "Allow this session")
            assert requests == []
            assert _results(events)[0].skipped
            assert before._session_root_grants.roots == set()
            assert not before.tool_manager.workspace.allows(target)
        finally:
            await before.aclose()
    finally:
        await parent.aclose()


async def test_child_local_grant_under_a_permitting_ceiling_is_child_only(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    nested = project / "nested"
    nested.mkdir()
    target = project / "note.txt"
    target.write_text("parent project content")
    parent = build_test_agent_loop(
        config=_config(), cwd=project, backend=FakeBackend(), user_input_capability=True
    )
    port = _SessionGrantPort()
    port.register(parent)
    try:
        policy = parent.child_runtime_policy
        assert policy.inherited_root_grants == ()
        child = _child_loop(
            parent, policy, FakeBackend([*_read_turn(target, "child-read")]), cwd=nested
        )
        sibling = _child_loop(
            parent,
            policy,
            FakeBackend([*_read_turn(target, "sibling-read")]),
            cwd=nested,
        )
        for loop in (child, sibling):
            loop.bind_root_grant_port(port)
            port.register(loop)
        try:
            # The child's own roots stop at its cwd; the frozen ceiling (the
            # parent's workspace) permits the parent project directory.
            assert not child.tool_manager.workspace.allows(target)
            ceiling = child.tool_manager.workspace.ceiling
            assert ceiling is not None and ceiling.allows(target)
            events, requests = await _drive(child, "Allow this session")
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert not results[0].skipped
            assert results[0].result is not None
            assert "parent project content" in results[0].result.model_dump_json()
            assert child._session_root_grants.roots == {project.resolve()}
            # The grant is child-local: parent and sibling stores stay empty.
            assert parent._session_root_grants.roots == set()
            assert sibling._session_root_grants.roots == set()
            assert not sibling.tool_manager.workspace.allows(target)
            # A sibling asks for itself; the parent still receives nothing.
            events, requests = await _drive(sibling, "Deny")
            assert len(requests) == 1
            assert _results(events)[0].skipped
            assert sibling._session_root_grants.roots == set()
            assert sibling._session_root_grants.denied_roots == {project.resolve()}
            assert parent._session_root_grants.roots == set()
        finally:
            await child.aclose()
            await sibling.aclose()
    finally:
        await parent.aclose()


async def test_parent_ceiling_denial_from_a_child_never_prompts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    parent = build_test_agent_loop(
        config=_config(), cwd=project, backend=FakeBackend(), user_input_capability=True
    )
    port = _SessionGrantPort()
    port.register(parent)
    try:
        child = _child_loop(
            parent,
            parent.child_runtime_policy,
            FakeBackend([
                *_read_turn(target, "frozen"),
                *_read_turn(target, "frozen-retry"),
            ]),
        )
        child.bind_root_grant_port(port)
        port.register(child)
        try:
            # The frozen ceiling is the parent workspace without the grant, so
            # the denial is a parent ceiling: no grant can lift it.
            workspace = child.tool_manager.workspace
            assert not workspace.allows(target)
            assert workspace.ceiling is not None
            assert not workspace.ceiling.allows(target)
            events, requests = await _drive(child, "Allow this session")
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert child._session_root_grants.roots == set()
            # Even a later parent grant does not widen the frozen child.
            parent.apply_root_grant(outside)
            events, requests = await _drive(child, "Allow this session")
            assert requests == []
            assert _results(events)[0].skipped
            assert child._session_root_grants.roots == set()
            assert not child.tool_manager.workspace.allows(target)
        finally:
            await child.aclose()
    finally:
        await parent.aclose()


async def test_grandchild_inherits_a_child_local_grant(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    nested = project / "nested"
    nested.mkdir()
    target = project / "note.txt"
    target.write_text("parent project content")
    parent = build_test_agent_loop(
        config=_config(), cwd=project, backend=FakeBackend(), user_input_capability=True
    )
    port = _SessionGrantPort()
    port.register(parent)
    try:
        policy = parent.child_runtime_policy
        child = _child_loop(
            parent,
            policy,
            FakeBackend([*_read_turn(target, "child-grant")]),
            cwd=nested,
        )
        child.bind_root_grant_port(port)
        port.register(child)
        try:
            events, requests = await _drive(child, "Allow this session")
            assert len(requests) == 1
            assert not _results(events)[0].skipped
            assert child._session_root_grants.roots == {project.resolve()}
            # A grandchild dispatched by the granted child inherits the grant.
            grandchild = _child_loop(
                child,
                child.child_runtime_policy,
                FakeBackend([*_read_turn(target, "grandchild")]),
            )
            try:
                assert grandchild._session_root_grants.roots == {project.resolve()}
                assert grandchild.tool_manager.workspace.allows(target)
                events, requests = await _drive(grandchild, "Allow this session")
                assert requests == []
                results = _results(events)
                assert len(results) == 1
                assert not results[0].skipped
                assert results[0].result is not None
                assert "parent project content" in results[0].result.model_dump_json()
            finally:
                await grandchild.aclose()
        finally:
            await child.aclose()
    finally:
        await parent.aclose()


async def test_inherited_grants_are_bare_paths_across_a_differing_cwd(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    parent = build_test_agent_loop(
        config=_config(), cwd=project, backend=FakeBackend(), user_input_capability=True
    )
    try:
        parent.apply_root_grant(outside)
        # A child rooted at a different project identity still receives the
        # grant: it is a bare canonical path, never keyed to the parent project.
        child = _child_loop(
            parent,
            parent.child_runtime_policy,
            FakeBackend([*_read_turn(target, "relocated")]),
            cwd=relocated,
        )
        try:
            assert child.cwd == relocated.resolve()
            assert child._session_root_grants.roots == {outside.resolve()}
            workspace = child.tool_manager.workspace
            assert workspace.cwd == relocated.resolve()
            assert outside.resolve() in workspace.authorized_roots
            assert workspace.allows(target)
            events, requests = await _drive(child, "Allow this session")
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert not results[0].skipped
            assert results[0].result is not None
            assert "granted content" in results[0].result.model_dump_json()
        finally:
            await child.aclose()
    finally:
        await parent.aclose()


async def test_retask_and_reload_do_not_widen_a_retained_child(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    parent = build_test_agent_loop(
        config=_config(), cwd=project, backend=FakeBackend(), user_input_capability=True
    )
    port = _SessionGrantPort()
    port.register(parent)
    try:
        child = _child_loop(
            parent,
            parent.child_runtime_policy,
            FakeBackend([
                *_read_turn(target, "first"),
                *_read_turn(target, "retask"),
                *_read_turn(target, "reload"),
            ]),
        )
        child.bind_root_grant_port(port)
        port.register(child)
        try:
            frozen_workspace = child.tool_manager.workspace
            events, _requests = await _drive(child, "Allow this session")
            assert _results(events)[0].skipped
            parent.apply_root_grant(outside)
            # A retask of the retained child reuses its frozen authority.
            events, requests = await _drive(child, "Allow this session")
            assert requests == []
            assert _results(events)[0].skipped
            assert child._session_root_grants.roots == set()
            # A reload rebuilds consumers around the same store and ceiling.
            await child.reload_with_initial_messages(reload_config=True)
            assert child.tool_manager.workspace is not frozen_workspace
            assert child.tool_manager.workspace.ceiling is frozen_workspace.ceiling
            events, requests = await _drive(child, "Allow this session")
            assert requests == []
            assert _results(events)[0].skipped
            assert child._session_root_grants.roots == set()
            assert not child.tool_manager.workspace.allows(target)
            assert len(_tool_texts(child)) == 3
            assert all(
                "outside authorized workspace" in text for text in _tool_texts(child)
            )
        finally:
            await child.aclose()
    finally:
        await parent.aclose()
