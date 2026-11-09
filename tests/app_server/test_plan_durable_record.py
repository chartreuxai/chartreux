"""The forced durable plan-record writes: C2's retirement and C3's marker.

Both handoff writes land exactly where the ordinary plan-mode persistence is
silently uninterested: ``_persist_metadata_field_locked`` patches an EXISTING
metadata file, and a fresh planning session (no saved turn yet) and a session
just rotated by the reset have none. A skipped write there is indistinguishable
from a durable one, so the handoff uses the forced variant: the file is created,
the bytes are published through the plan-scoped durable wrapper, and every
failure raises. These tests pin that contract, including the fault rows — a
failed publish may not repair the artifact into looking successful.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
from typing import Any

import pytest

from chartreux.app_server._plan_record import plan_mode_record
from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server._session_backend_port import SessionBackendError
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    ProtocolErrorCode,
    SessionPlanRecord,
    SessionPlanRecordWriteParams,
)
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.models import SessionLoggingConfig
import chartreux.core.plan_episode as plan_episode
from chartreux.core.plan_episode import (
    PLAN_TEMP_PREFIX,
    LoggingDomain,
    PlanEpisodeScanError,
    PlanEpisodeWriteError,
    scan_episode_executor,
)
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session.session_logger import SessionDiskFullError
from chartreux.core.session_types import PlanModeRecord
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import attach_test_app_server_session, build_test_app_server

_EPISODE_ID = "ep-0123456789abcdef"
_CONSENT_SHA256 = "c0" * 32
_SNAPSHOT = f"accepted-{_EPISODE_ID}.md"
_METADATA = "meta.json"


def _loop(tmp_path: Path, *, enabled: bool = True) -> AgentLoop:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=enabled, save_dir=str(tmp_path / "sessions")
        )
    )
    return build_test_agent_loop(config=config, session_id="initial")


def _draft() -> SessionPlanRecord:
    return SessionPlanRecord(
        plan_path="/workspaces/chartreux/tasks/plan-abc123/plan.md",
        theme="Fix the widget",
        planner_model="glm-5-3",
        planner_thinking="high",
        pre_entry_active_model="mistral-large-4",
        episode_id=_EPISODE_ID,
        sidecar_path="/workspaces/chartreux/tasks/plan-abc123/.sidecar.json",
        draft_path="/workspaces/chartreux/tasks/plan-abc123/plan.md",
        state="draft",
    )


def _accepted() -> SessionPlanRecord:
    return _draft().model_copy(
        update={
            "state": "accepted",
            "consent_sha256": _CONSENT_SHA256,
            "snapshot_path": _SNAPSHOT,
            "attempt_generation": 2,
        }
    )


def _marker() -> SessionPlanRecord:
    return _draft().model_copy(
        update={
            "state": "accepting",
            "consent_sha256": _CONSENT_SHA256,
            "snapshot_path": _SNAPSHOT,
            "attempt_generation": 2,
        }
    )


def _persisted(record: SessionPlanRecord) -> PlanModeRecord:
    """The persisted-model form a direct session-logger call takes."""
    return plan_mode_record(record)


async def _serve(loop: AgentLoop) -> tuple[AppServer, AppServerSession]:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    return server, session


def _backend(server: AppServer) -> SessionBackendImpl:
    root = server._require_root()
    assert isinstance(root, SessionBackendImpl)
    return root


def _session_dir(loop: AgentLoop) -> Path:
    session_dir = loop.session_logger.session_dir
    assert session_dir is not None
    return session_dir


def _metadata_path(loop: AgentLoop) -> Path:
    return _session_dir(loop) / _METADATA


def _plan_temporaries(directory: Path) -> list[str]:
    return sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.name.startswith(PLAN_TEMP_PREFIX)
    )


class _ReplaceFailingOS:
    """The plan module's ``os`` binding with its atomic replace disabled."""

    def __init__(self, error: OSError) -> None:
        self._error = error

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    def replace(self, source: Any, destination: Any) -> None:
        raise self._error


class _ReplaceRecordingOS:
    """The plan module's ``os`` binding, recording each published temp name."""

    def __init__(self) -> None:
        self.published: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    def replace(self, source: Any, destination: Any) -> None:
        self.published.append(Path(source).name)
        os.replace(source, destination)


@contextmanager
def _publish_fails(monkeypatch: pytest.MonkeyPatch, error: OSError) -> Iterator[None]:
    """Fault injection at the plan writer's atomic replace.

    Patched on the episode module's own ``os`` binding, so the injection cannot
    reach any other writer and the plan writer itself cannot compensate for it.
    """
    monkeypatch.setattr(plan_episode, "os", _ReplaceFailingOS(error))
    yield


class TestC2ForcedRetirement:
    @pytest.mark.asyncio
    async def test_the_forced_retirement_creates_the_missing_metadata_file(
        self, tmp_path: Path
    ) -> None:
        # A fresh planning session with no saved turn is the real lazy hole: no
        # metadata file exists yet, so the retirement must create it.
        loop = _loop(tmp_path)
        assert not _metadata_path(loop).exists()
        server, _ = await _serve(loop)
        backend = _backend(server)

        result = await backend.write_plan_record(
            SessionPlanRecordWriteParams(
                session_id=backend.session_id, record=_accepted()
            ),
            force_durable=True,
        )

        assert result.response.record == _accepted()
        assert _metadata_path(loop).is_file()
        persisted = SessionLoader.load_metadata(_session_dir(loop))
        assert persisted.plan_mode is not None
        assert persisted.plan_mode.state == "accepted"
        assert persisted.plan_mode.episode_id == _EPISODE_ID
        assert persisted.plan_mode.consent_sha256 == _CONSENT_SHA256
        assert persisted.plan_mode.snapshot_path == _SNAPSHOT
        assert persisted.plan_mode.attempt_generation == 2

    @pytest.mark.asyncio
    async def test_the_lazy_write_reports_a_record_it_never_persisted(
        self, tmp_path: Path
    ) -> None:
        # The hole the forced variant closes, stated as behavior: the lazy shape
        # answers with the in-memory record while nothing reaches disk.
        loop = _loop(tmp_path)
        server, _ = await _serve(loop)
        backend = _backend(server)

        result = await backend.write_plan_record(
            SessionPlanRecordWriteParams(
                session_id=backend.session_id, record=_accepted()
            )
        )

        assert result.response.record == _accepted()
        assert not _metadata_path(loop).exists()

    @pytest.mark.asyncio
    async def test_the_forced_retirement_publishes_through_the_plan_durable_wrapper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loop = _loop(tmp_path)
        recording = _ReplaceRecordingOS()
        monkeypatch.setattr(plan_episode, "os", recording)

        await loop.session_logger.persist_plan_mode_durable(_persisted(_accepted()))

        assert len(recording.published) == 1
        assert recording.published[0].startswith(PLAN_TEMP_PREFIX)
        assert _plan_temporaries(_session_dir(loop)) == []
        assert _metadata_path(loop).is_file()

    @pytest.mark.asyncio
    async def test_the_retirement_preserves_the_rest_of_the_metadata(
        self, tmp_path: Path
    ) -> None:
        loop = _loop(tmp_path)
        await loop.persist_empty_session()
        await loop.session_logger.persist_plan_mode(plan_mode_record(_draft()))
        before = json.loads(_metadata_path(loop).read_text(encoding="utf-8"))

        await loop.session_logger.persist_plan_mode_durable(_persisted(_accepted()))

        after = json.loads(_metadata_path(loop).read_text(encoding="utf-8"))
        assert after["plan_mode"]["state"] == "accepted"
        assert {k: v for k, v in after.items() if k != "plan_mode"} == {
            k: v for k, v in before.items() if k != "plan_mode"
        }

    @pytest.mark.asyncio
    async def test_the_retirement_keeps_the_root_session_guard(
        self, tmp_path: Path
    ) -> None:
        loop = _loop(tmp_path)
        server, _ = await _serve(loop)
        backend = _backend(server)

        with pytest.raises(SessionBackendError) as refused:
            await backend.write_plan_record(
                SessionPlanRecordWriteParams(
                    session_id="some-other-session", record=_accepted()
                ),
                force_durable=True,
            )

        assert refused.value.code is ProtocolErrorCode.INVALID_PARAMS
        assert not _metadata_path(loop).exists()

    @pytest.mark.asyncio
    async def test_a_failed_publish_surfaces_loudly_through_the_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loop = _loop(tmp_path)
        server, _ = await _serve(loop)
        backend = _backend(server)

        with _publish_fails(monkeypatch, OSError(errno.EIO, "injected")):
            with pytest.raises(PlanEpisodeWriteError) as failed:
                await backend.write_plan_record(
                    SessionPlanRecordWriteParams(
                        session_id=backend.session_id, record=_accepted()
                    ),
                    force_durable=True,
                )

        assert "meta.json" in str(failed.value)
        assert _plan_temporaries(_session_dir(loop)) == []

    @pytest.mark.asyncio
    async def test_a_failed_publish_leaves_the_existing_metadata_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loop = _loop(tmp_path)
        await loop.persist_empty_session()
        await loop.session_logger.persist_plan_mode(_persisted(_draft()))
        before = _metadata_path(loop).read_bytes()

        with _publish_fails(monkeypatch, OSError(errno.EIO, "injected")):
            with pytest.raises(PlanEpisodeWriteError):
                await loop.session_logger.persist_plan_mode_durable(
                    _persisted(_accepted())
                )

        # No repair, no partial artifact: the pre-failure bytes are exactly what
        # remains on disk, and the failed write left no temporary sibling.
        assert _metadata_path(loop).read_bytes() == before
        assert _plan_temporaries(_session_dir(loop)) == []
        persisted = SessionLoader.load_metadata(_session_dir(loop))
        assert persisted.plan_mode is not None
        assert persisted.plan_mode.state == "draft"

    @pytest.mark.asyncio
    async def test_a_disk_full_publish_keeps_its_own_loud_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loop = _loop(tmp_path)

        with _publish_fails(monkeypatch, OSError(errno.ENOSPC, "injected")):
            with pytest.raises(SessionDiskFullError):
                await loop.session_logger.persist_plan_mode_durable(
                    _persisted(_accepted())
                )

        assert not _metadata_path(loop).exists()

    @pytest.mark.asyncio
    async def test_a_publish_whose_parent_barrier_fails_is_never_durable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The bytes may land; the durability barrier did not, so the call must
        # fail rather than report a durable write it cannot prove.
        loop = _loop(tmp_path)

        def _barrier_fails(path: Path) -> None:
            raise OSError(errno.EIO, "injected barrier failure")

        monkeypatch.setattr(plan_episode, "fsync_directory", _barrier_fails)

        with pytest.raises(PlanEpisodeWriteError):
            await loop.session_logger.persist_plan_mode_durable(_persisted(_accepted()))

        assert _plan_temporaries(_session_dir(loop)) == []

    @pytest.mark.asyncio
    async def test_an_unmakeable_session_directory_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        loop = _loop(tmp_path)
        session_dir = _session_dir(loop)
        session_dir.parent.mkdir(parents=True, exist_ok=True)
        session_dir.write_text("not a directory", encoding="utf-8")

        with pytest.raises(PlanEpisodeWriteError) as failed:
            await loop.session_logger.persist_plan_mode_durable(_persisted(_accepted()))

        assert str(session_dir) in str(failed.value)
        assert session_dir.is_file()


class TestC3MarkerDurableWrite:
    @pytest.mark.asyncio
    async def test_the_marker_lands_on_the_rotated_session_with_no_metadata_file(
        self, tmp_path: Path
    ) -> None:
        # C3 runs after clear_history rotated the session: the executor id and
        # save folder are new, and the marker write is the session's FIRST
        # durable act, so there is no metadata file to patch.
        loop = _loop(tmp_path)
        await loop.persist_empty_session()
        planning_dir = _session_dir(loop)
        loop.session_logger.reset_session("executor-session")
        executor_dir = _session_dir(loop)

        assert executor_dir != planning_dir
        assert not executor_dir.exists()

        await loop.session_logger.persist_plan_mode_durable(_persisted(_marker()))

        assert executor_dir.is_dir()
        persisted = SessionLoader.load_metadata(executor_dir)
        assert persisted.session_id == "executor-session"
        assert persisted.plan_mode is not None
        assert persisted.plan_mode.state == "accepting"
        assert persisted.plan_mode.episode_id == _EPISODE_ID
        assert persisted.plan_mode.consent_sha256 == _CONSENT_SHA256
        assert persisted.plan_mode.snapshot_path == _SNAPSHOT

    @pytest.mark.asyncio
    async def test_the_recovery_scan_finds_the_executor_from_the_marker(
        self, tmp_path: Path
    ) -> None:
        loop = _loop(tmp_path)
        await loop.persist_empty_session()
        loop.session_logger.reset_session("executor-session")
        await loop.session_logger.persist_plan_mode_durable(_persisted(_marker()))

        found = scan_episode_executor(
            LoggingDomain(
                save_dir=str(tmp_path / "sessions"), session_prefix="session"
            ),
            _EPISODE_ID,
        )

        assert found.executor_session_id == "executor-session"

    @pytest.mark.asyncio
    async def test_a_failed_marker_leaves_no_executor_meta_to_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Row 3: the marker is the executor predicate, so a failed marker write
        # must not leave an executor meta behind — and the scan fails closed
        # rather than silently reading the marker's absence as "not executed".
        loop = _loop(tmp_path)
        await loop.persist_empty_session()
        loop.session_logger.reset_session("executor-session")
        executor_dir = _session_dir(loop)

        with _publish_fails(monkeypatch, OSError(errno.EIO, "injected")):
            with pytest.raises(PlanEpisodeWriteError):
                await loop.session_logger.persist_plan_mode_durable(
                    _persisted(_marker())
                )

        assert not (executor_dir / _METADATA).exists()
        with pytest.raises(PlanEpisodeScanError):
            scan_episode_executor(
                LoggingDomain(
                    save_dir=str(tmp_path / "sessions"), session_prefix="session"
                ),
                _EPISODE_ID,
            )

    @pytest.mark.asyncio
    async def test_a_marker_cannot_be_written_durably_without_session_logging(
        self, tmp_path: Path
    ) -> None:
        loop = _loop(tmp_path, enabled=False)

        with pytest.raises(PlanEpisodeWriteError) as failed:
            await loop.session_logger.persist_plan_mode_durable(_persisted(_marker()))

        assert "logging" in str(failed.value)
