from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory, close_agent_loop
from chartreux.app_server.registry_manifest import (
    RegistryManifest,
    SessionDeletedError,
    SessionWriterFence,
)
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.session.session_lease import SessionBusyError
from tests.conftest import build_test_agent_loop, build_test_vibe_config

pytestmark = pytest.mark.asyncio


def test_writer_fence_refuses_competitors_and_preserves_inode(tmp_path: Path) -> None:
    fence = SessionWriterFence(tmp_path, "root")
    with fence:
        inode = fence.path.stat().st_ino
        with pytest.raises(SessionBusyError):
            SessionWriterFence(tmp_path, "root").acquire()
        RegistryManifest(tmp_path).initialize()
        assert fence.path.stat().st_ino == inode
        script = (
            "from pathlib import Path\n"
            "from chartreux.app_server.registry_manifest import SessionWriterFence\n"
            "from chartreux.core.session.session_lease import SessionBusyError\n"
            "import sys\n"
            "try:\n"
            "    SessionWriterFence(Path(sys.argv[1]), 'root').acquire()\n"
            "except SessionBusyError:\n"
            "    sys.exit(0)\n"
            "sys.exit(1)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            check=False,
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr.decode()
    assert fence.path.stat().st_ino == inode
    with SessionWriterFence(tmp_path, "root"):
        assert fence.path.stat().st_ino == inode


def test_writer_fence_reports_missing_session(tmp_path: Path) -> None:
    with pytest.raises(SessionDeletedError):
        SessionWriterFence(tmp_path / "missing", "root").acquire()


def _logging_loop(tmp_path: Path):
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
        )
    )
    return build_test_agent_loop(config=config)


async def test_factory_fence_lifecycle_refuses_competitors_until_release(
    tmp_path: Path,
) -> None:
    loop = _logging_loop(tmp_path)
    session_dir = loop.session_logger.session_dir
    assert session_dir is not None
    factory = AgentRuntimeFactory()
    try:
        # The bind path creates the session dir (and the lockfile) first.
        factory.initialize_registry_manifest(loop, session_dir)
        fence = factory.ensure_writer_fence(loop, session_dir)
        assert factory.writer_fence(loop) is fence
        assert fence.path == session_dir / ".writer-fence.lock"
        inode = fence.path.stat().st_ino
        with pytest.raises(SessionBusyError):
            SessionWriterFence(session_dir, loop.session_id).acquire()
        # Re-ensuring the same directory reuses the held fence (stable inode).
        assert factory.ensure_writer_fence(loop, session_dir) is fence
        assert fence.path.stat().st_ino == inode
        factory.release_writer_fence(loop)
        assert factory.writer_fence(loop) is None
        # The lockfile is never unlinked while in scope: release keeps it.
        assert fence.path.stat().st_ino == inode
        SessionWriterFence(session_dir, loop.session_id).acquire().release()
    finally:
        await close_agent_loop(loop)


async def test_close_agent_loop_releases_the_writer_fence(tmp_path: Path) -> None:
    loop = _logging_loop(tmp_path)
    session_dir = loop.session_logger.session_dir
    assert session_dir is not None
    factory = AgentRuntimeFactory()
    factory.initialize_registry_manifest(loop, session_dir)
    factory.ensure_writer_fence(loop, session_dir)
    with pytest.raises(SessionBusyError):
        SessionWriterFence(session_dir, loop.session_id).acquire()
    await close_agent_loop(loop)
    assert factory.writer_fence(loop) is None
    SessionWriterFence(session_dir, loop.session_id).acquire().release()


async def test_resume_root_holds_the_fence_for_the_resumed_session(
    tmp_path: Path,
) -> None:
    loop = _logging_loop(tmp_path)
    await loop.persist_empty_session()
    session_dir = loop.session_logger.session_dir
    assert session_dir is not None
    factory = AgentRuntimeFactory()
    try:
        await factory.resume_root(loop, loop.session_id)
        fence = factory.writer_fence(loop)
        assert fence is not None
        assert fence.path.parent == session_dir
        inode = fence.path.stat().st_ino
        with pytest.raises(SessionBusyError):
            SessionWriterFence(session_dir, loop.session_id).acquire()
        # A later bind of the same loop reuses the held fence (no self-conflict,
        # stable inode).
        assert factory.ensure_writer_fence(loop, session_dir) is fence
        assert fence.path.stat().st_ino == inode
    finally:
        await close_agent_loop(loop)
    SessionWriterFence(session_dir, loop.session_id).acquire().release()
