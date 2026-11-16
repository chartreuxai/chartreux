"""Wire regression test for ``agent/session/plan_mode/present`` (plan-mode removal).

The root agent's ``_plan_mode_present`` answers a one-bit flag from the RAW
session metadata only. It must never call ``SessionLoader.load_session``:
that reads the whole ``messages.jsonl`` and raises ``ValueError`` when the
file does not exist yet (a fresh resumed session), erroring the wire call
instead of returning ``{"plan_mode_present": False}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from chartreux.app_server.root_agent import RootAgentServer
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from tests.conftest import build_test_agent_loop, build_test_vibe_config


async def _persisted_loop(tmp_path: Path) -> AgentLoop:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        )
    )
    loop = build_test_agent_loop(config=config)
    await loop.persist_empty_session()
    return loop


def _server_over(loop: AgentLoop) -> RootAgentServer:
    server = RootAgentServer(cast(Any, None), factory=cast(Any, None))
    server._loop = loop
    return server


@pytest.mark.asyncio
async def test_plan_mode_present_is_false_when_messages_log_is_absent(
    tmp_path: Path,
) -> None:
    loop = await _persisted_loop(tmp_path)
    try:
        session_dir = loop.session_logger.session_dir
        assert session_dir is not None
        # Fresh resumed session: metadata exists, the transcript does not.
        (session_dir / "messages.jsonl").unlink()
        server = _server_over(loop)
        result = await server._plan_mode_present({"session_id": loop.session_id})
        assert result == {"plan_mode_present": False}
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_plan_mode_present_is_true_when_raw_metadata_carries_the_key(
    tmp_path: Path,
) -> None:
    loop = await _persisted_loop(tmp_path)
    try:
        session_dir = loop.session_logger.session_dir
        assert session_dir is not None
        metadata_path = session_dir / "meta.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["plan_mode"] = {
            "plan_path": "/tmp/workspaces/tasks/plan-x/plan.md",
            "state": "drafting",
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        server = _server_over(loop)
        result = await server._plan_mode_present({"session_id": loop.session_id})
        assert result == {"plan_mode_present": True}
    finally:
        await loop.aclose()
