"""Dangerous-command lifecycle — session boundaries concern (stage 20c split target).

Runtime-grant lifetime across session boundaries and child sessions: a
reload preserves runtime grants, a resume does not restore them, a resume
rebind clears them, a history clear clears them, and a parent's dangerous
gap is a ceiling for a child session. Moved here from
``test_dangerous_command_lifecycle.py`` in the 20c-ii-4 concern commit,
together with the single-concern ``_logged_config`` helper. The module
re-registers the marks the population had in the original —
``pytestmark = pytest.mark.asyncio``; the original file defines no
module-local fixtures, so there is no fixture registration to carry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._dangerous_command_grants import (
    DangerousCommandCallIdentity,
)
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.tools.base import ToolPermission
from chartreux.core.workspace import Workspace
from tests.agent_loop._lifecycle_test_support import (
    VISION_MODEL,
    _authorities,
    _bash_args,
    _call,
    _config,
    _drive,
    _results,
    make_repo,
    scope_of,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


def _logged_config(save_dir: Path) -> ChartreuxConfigSchema:
    """A config with session logging on, so a resume has a session to rebind."""
    return build_test_vibe_config(
        active_model="vision",
        models=[VISION_MODEL],
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(save_dir)),
    )


class TestSessionLifetime:
    """Runtime grants live exactly as long as the session."""

    async def test_reload_preserves_runtime_grants(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "grant")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        try:
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            old_manager = agent.tool_manager
            await agent.reload_with_initial_messages()
            # The reload rebuilds the manager around the same runtime store.
            assert agent.tool_manager is not old_manager
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            tool = agent.tool_manager.get("bash")
            decision = tool.resolve_permission(
                tool.validate_arguments({"command": "git push"})
            )
            assert decision is not None
            assert decision.permission == ToolPermission.ALWAYS
        finally:
            await agent.aclose()

    async def test_resume_does_not_restore_runtime_grants(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "grant")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        try:
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert _authorities(agent) != []
            session_id = agent.session_id
        finally:
            await agent.aclose()
        # A resumed conversation is a new loop: the runtime store starts empty.
        resumed = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[_call("bash", _bash_args("git push"), "resume")]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
            session_id=session_id,
        )
        try:
            assert _authorities(resumed) == []
            events, requests = await _drive(resumed, "Deny")
            assert len(requests) == 1
            assert _results(events)[0].skipped
            assert _authorities(resumed) == []
        finally:
            await resumed.aclose()

    async def test_resume_rebind_clears_runtime_grants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resume reuses the loop in place; session grants do not survive it."""
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "grant")]
                )
            ],
            [mock_llm_chunk(content="done")],
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "resume")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_logged_config(tmp_path / "logs"),
            cwd=project,
            backend=backend,
            user_input_capability=True,
        )
        try:
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            # Refusal memory and a pending once permit are session-scoped too.
            refused = DangerousCommandCallIdentity(
                scope=scope_of(project), command=("git", "push", "--force")
            )
            agent._dangerous_command_grants.record_refusal(refused)
            agent._dangerous_command_grants.install_once(
                frozenset({("git.push", scope_of(project))})
            )
            # The runtime reuses this loop in place on resume.
            logger = agent.session_logger
            assert logger.session_dir is not None
            assert logger.session_metadata is not None
            agent.rebind_to_session(
                agent.session_id,
                logger.session_dir,
                list(agent.messages),
                session_metadata=logger.session_metadata,
                prepared_scratchpad=None,
            )
            assert _authorities(agent) == []
            assert not agent._dangerous_command_grants.is_refused(refused)
            # The same call re-prompts instead of resuming on stale authority.
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert not _results(events)[0].skipped
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
        finally:
            await agent.aclose()

    async def test_clear_history_clears_runtime_grants(self, tmp_path: Path) -> None:
        """A history clear starts a new conversation; session grants expire."""
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "grant")]
                )
            ],
            [mock_llm_chunk(content="done")],
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "again")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        try:
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            refused = DangerousCommandCallIdentity(
                scope=scope_of(project), command=("git", "push", "--force")
            )
            agent._dangerous_command_grants.record_refusal(refused)
            agent._dangerous_command_grants.install_once(
                frozenset({("git.push", scope_of(project))})
            )
            await agent.clear_history()
            assert _authorities(agent) == []
            assert not agent._dangerous_command_grants.is_refused(refused)
            # The cached authority the grant fed is dropped too: the push is
            # denied again and proposes a fresh approval.
            tool = agent.tool_manager.get("bash")
            decision = tool.resolve_permission(
                tool.validate_arguments({"command": "git push"})
            )
            assert decision is not None
            assert decision.permission == ToolPermission.NEVER
            assert decision.dangerous_command_proposal is not None
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert not _results(events)[0].skipped
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
        finally:
            await agent.aclose()


class TestChildSessions:
    """A child session never prompts on its own; parent gaps are ceilings."""

    async def test_parent_dangerous_gap_is_a_ceiling_for_the_child(
        self, tmp_path: Path
    ) -> None:
        project = make_repo(tmp_path / "project")
        parent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        child = AgentLoop(
            config_orchestrator=parent.config_orchestrator.copy(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[_call("bash", _bash_args("git push"), "child")]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            is_subagent=True,
            inherited_workspace=Workspace.for_session(project),
            parent_authority_getter=lambda: parent.tool_manager,
            user_input_capability=True,
        )
        try:
            events, requests = await _drive(child, "Allow this session")
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            # The parent's dangerous gap is a ceiling: no local grant, no prompt.
            assert _authorities(child) == []
            assert _authorities(parent) == []
        finally:
            await child.aclose()
            await parent.aclose()
