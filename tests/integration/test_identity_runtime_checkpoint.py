"""Package 11d C-runtime checkpoint: the end-to-end identity lifecycle.

One fake-backend root session exercises the identity runtime surface
continuously: canonical delegation through the task tool (the canonical
identity fields of packages 8/11b), the identity-keyed delegated record and
its persisted launch envelope (11a/11c), in-band result retrieval, a config
reload that retains the frozen bundle, a fresh-runtime resume from disk
(persisted bundle content survives; the identity binding is restored), and
a /clear that binds a fresh bundle.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from pathlib import Path
import time
from typing import Any, cast

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server.server import AppServer
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session_types import LaunchMetadataV2, LaunchPersonaV2
from chartreux.core.subagents import WaitStatus
from tests.app_server._subagent_test_support import BlockingBackend, _config, _consume
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    set_agent_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend


def _canonical_task_call() -> ToolCall:
    arguments = {
        "task": "Trace the identity runtime end to end",
        "identity": "researcher",
        "background": True,
        "retain": True,
        "task_summary": "identity checkpoint",
    }
    return ToolCall(
        id="task-checkpoint",
        index=0,
        function=FunctionCall(name="task", arguments=json.dumps(arguments)),
    )


async def _settle_notification_turn(
    server: AppServer, parent_backend: FakeBackend, *, timeout: float = 30.0
) -> None:
    """Wait out the queued background-completion notification turn."""
    backend = cast(SessionBackendImpl, server._root)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(parent_backend.requests_messages) >= 3:
            root = backend.session
            if root.turns._active_task is None and not root.turns.has_queued_turns:
                return
        await asyncio.sleep(0.05)
    raise AssertionError("background notification turn did not settle")


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_identity_runtime_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    parent_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_canonical_task_call()])],
        [mock_llm_chunk(content="Delegation launched")],
        [mock_llm_chunk(content="Notification received")],
    ])
    child_backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    created: list[tuple[AgentLoop, Path | None]] = []
    create_child = AgentRuntimeFactory.create_child

    async def capture_child(
        factory: AgentRuntimeFactory, loop: AgentLoop, agent_name: str, **kwargs: Any
    ) -> AgentLoop:
        child = await create_child(factory, loop, agent_name, **kwargs)
        created.append((child, child.session_logger.session_dir))
        return child

    monkeypatch.setattr(AgentRuntimeFactory, "create_child", capture_child)

    config = _config(logging)
    parent = build_test_agent_loop(
        config=config, backend=parent_backend, enable_streaming=True
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    registry = legacy_backend(server).children

    root_bound = parent._bound_policy
    assert root_bound is not None
    assert parent._bound_identity == "coder"
    assert "You are the coder" in (parent.messages[0].content or "")

    action: asyncio.Task[None] | None = None
    try:
        action = asyncio.create_task(_consume(session.act("Delegate the research")))
        await asyncio.wait_for(child_backend.started.wait(), timeout=30)
        await asyncio.wait_for(action, timeout=30)
        action = None

        # The delegated record is identity-keyed end to end: the launch
        # resolved the canonical identity field and the record carries the
        # child's committed (identity, identity_revision) binding.
        summaries = await registry.check_agents()
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.identity == "researcher"
        assert summary.identity_revision is not None
        assert summary.availability == "running"
        assert summary.current_run_id is not None
        agent_handle = summary.agent_id
        run_handle = summary.current_run_id

        child_backend.release.set()

        # The parent retrieves the delegated result.
        waited = await registry.wait_for_agent(agent_handle, run_handle, timeout=30)
        assert waited.status is WaitStatus.COMPLETED
        assert waited.result is not None
        assert waited.result.completed
        assert "Child completed" in waited.result.response
        stored = await registry.get_agent_result(agent_handle, run_handle)
        assert stored is not None
        assert "Child completed" in stored.response

        # The background completion notification drives one final parent
        # turn; let it settle before touching the root session.
        await _settle_notification_turn(server, parent_backend)

        # The retained record keeps the identity binding after completion.
        summaries = await registry.check_agents()
        assert len(summaries) == 1
        retained = summaries[0]
        assert retained.availability == "idle"
        assert retained.identity == "researcher"
        assert retained.identity_revision == summary.identity_revision

        # The launch envelope persisted the child's resolved identity
        # binding alongside the frozen policy bundle.
        child, child_dir = created[0]
        assert child_dir is not None
        assert child._bound_identity == "researcher"
        child_meta = SessionLoader.load_metadata(child_dir)
        child_envelope = child_meta.launch_config
        assert isinstance(child_envelope, LaunchMetadataV2)
        child_persona = child_envelope.persona
        assert isinstance(child_persona, LaunchPersonaV2)
        assert child_persona.identity == "researcher"
        assert child_persona.identity_revision == summary.identity_revision
        assert child_meta.policy_bundle is not None
    finally:
        child_backend.release.set()
        if action is not None:
            with suppress(Exception):
                await asyncio.wait_for(action, timeout=30)
    await legacy_backend(server).children.drain_children()

    root_dir = parent.session_logger.session_dir
    assert root_dir is not None
    root_meta = SessionLoader.load_metadata(root_dir)
    root_envelope = root_meta.launch_config
    assert isinstance(root_envelope, LaunchMetadataV2)
    root_persona = root_envelope.persona
    assert isinstance(root_persona, LaunchPersonaV2)
    assert root_persona.identity == "coder"

    saved_session_id = parent.session_id
    root_prompt = parent.messages[0].content

    # Reload: the frozen bundle is retained byte-for-byte and the saved
    # identity selection change is reported as pending.
    set_agent_config(parent, build_test_vibe_config(agent={"identity": "reviewer"}))
    await parent.reload_with_initial_messages()
    assert parent._bound_policy is root_bound
    assert parent.messages[0].content == root_prompt
    assert any(
        "identity selection change" in note for note in parent.pending_policy_changes
    )
    await session.close()
    await parent.aclose()

    # Resume from disk in a fresh runtime: the persisted bundle content
    # survives and the identity binding is restored from the record.
    source = build_test_agent_loop(config=_config(logging))
    try:
        await AgentRuntimeFactory().resume_root(source, saved_session_id)
        await source.wait_until_ready()
        assert source.session_id == saved_session_id
        assert source._bound_identity == "coder"
        assert source._identity_binding() == (
            root_persona.identity,
            root_persona.identity_revision,
        )
        assert source.messages[0].content == root_prompt
        await source.refresh_system_prompt()
        assert source.messages[0].content == root_prompt

        # /clear binds a fresh bundle from the current selections.
        source.config.agent.identity = "reviewer"
        await source.clear_history()
        assert source.session_id != saved_session_id
        assert source._bound_policy is not None
        assert source._bound_policy.manifest.identity.selected == "reviewer"
        assert source._bound_identity == "reviewer"
        new_prompt = source.messages[0].content or ""
        assert "You are the reviewer" in new_prompt
        assert "You are the coder" not in new_prompt
    finally:
        await source.aclose()
