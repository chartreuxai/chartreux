"""Shared test-support surface for the app-server local-harness suite (W10, stage 19d).

The definitions here are the shared module-level helper surface of
``tests/app_server/test_local.py``, relocated AST-verbatim in the 19d
phase-(i) helper-relocation commit: the accounting-configuration helper
``_configure_local_accounting`` (used by the startup-context tests of the
spawn concern and the accounting close/resume tests of the shutdown and
resume concerns), the worktree helpers ``_prepare_auto`` and ``_linked``
(session-start tests of the spawn concern plus session-stop and
worktree-remove tests of the shutdown concern), the repository factory
``_init_repo`` (34 tests across the spawn, shutdown and resume
concerns) and the session factory ``_open_local_session`` (session-start
tests of the spawn concern plus session-stop tests of the shutdown
concern). No test bodies live here; the test population stays in
``test_local.py`` until the phase-(ii) concern moves. Importer:
``test_local.py`` itself (the seam enumeration found zero external
importers of that file).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from git import Repo
import pytest

from chartreux.app_server import _runtime as runtime
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import ClientCapabilities, ClientInfo, SessionOptions
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.git.worktree import (
    LinkedWorktree,
    PreparedWorktree,
    WorktreeRepository,
)
import chartreux.core.git.worktree.record as worktree_record
from chartreux.core.llm import utility_completion
from tests.app_server.test_usage_attribution import AttemptBackend
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import create_legacy_app_server
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


def _configure_local_accounting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, logging_enabled: bool = False
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=logging_enabled,
            save_dir=str(tmp_path / "sessions"),
            generate_titles=False,
        )
    )
    monkeypatch.setattr(
        runtime,
        "build_default_orchestrator",
        AsyncMock(return_value=FakeConfigOrchestrator(config)),
    )
    monkeypatch.setattr(
        runtime.HarnessProcess,
        "_build_mcp_registry_impl",
        AsyncMock(return_value=FakeMCPRegistry()),
    )
    real_loop = runtime.AgentLoop

    def build_loop(**kwargs: Any) -> AgentLoop:
        return real_loop(
            backend=AttemptBackend([mock_llm_chunk(content="answer")]), **kwargs
        )

    monkeypatch.setattr(runtime, "AgentLoop", build_loop)
    monkeypatch.setattr(
        utility_completion,
        "create_backend",
        lambda **_: AttemptBackend([mock_llm_chunk(content="fix-bug")]),
    )


def _prepare_auto(base: Path, *, prompt: str | None = None) -> PreparedWorktree:
    with WorktreeRepository.open(base) as repository:
        prepared = repository.prepare_auto(prompt=prompt)
    if prepared.pending_token is not None:
        worktree_record.release_holder(prepared.pending_token)
    return prepared


def _linked(base: Path) -> tuple[LinkedWorktree, ...]:
    with WorktreeRepository.open(base) as repository:
        return repository.linked()


def _init_repo(root: Path) -> Repo:
    repo = Repo.init(root, initial_branch="main")
    repo.config_writer().set_value("user", "name", "Tester").release()
    repo.config_writer().set_value("user", "email", "t@example.com").release()
    (root / "file.txt").write_text("hello\n")
    repo.index.add(["file.txt"])
    repo.index.commit("initial")
    return repo


async def _open_local_session(
    name: str, session_options: SessionOptions
) -> tuple[AppServerSession, AppServerClient]:
    async def open_root(request: runtime.RootOpenRequest) -> AgentLoop:
        assert request.options.cwd is not None
        return build_test_agent_loop(cwd=Path(request.options.cwd))

    client_transport, server_transport = memory_transport_pair()
    server = create_legacy_app_server(server_transport, open_root=open_root)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name=name, version="1"),
        capabilities=ClientCapabilities(),
        session_options=session_options,
    )
    return session, client
