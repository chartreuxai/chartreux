"""Restore-path lifecycle interpretation under the episode sidecar (P10 chunk 7).

A fork inherits the plan-mode record, a rewind fork re-installs it, and a
compaction re-injects the binding. All three are RESTORE sites: the record each
one holds is a copy the source session wrote, so the sidecar, never the copy,
decides the lifecycle (design §4.5). Accepted and cancelled episodes never
restore as active planning, drafts restore as drafts, and an accepting episode
restores its accepting status.

The fork's shared arbiter is exercised through the real claim and terminal
refusals: the second fork consults the SAME sidecar and cannot double-accept.
Every provider call goes to the in-memory FakeBackend; no provider is contacted.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chartreux.app_server._runtime import _inherit_plan_mode
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.plan_episode import (
    RECOVERY_ROW_UNRESOLVABLE,
    STATE_ACCEPTED,
    STATE_ACCEPTING,
    STATE_CANCELLED,
    STATE_DRAFT,
    TERMINAL_STATES,
    LoggingDomain,
    PlanEpisodeSidecar,
    PlanEpisodeStateError,
    PlanEpisodeTerminalError,
    PlanSidecarError,
    claim_accept,
    create_sidecar,
    ensure_episode_fence,
    observe_episode,
    read_sidecar,
    sidecar_path,
    write_sidecar,
)
from chartreux.core.session_types import PlanModeRecord, PlanScanDomain
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

SESSION_ID = "01234567-89ab-cdef-0123-deadbeef1234"
FORK_SESSION = "11111111-2222-3333-4444-555555555555"
SECOND_FORK_SESSION = "66666666-7777-8888-9999-aaaaaaaaaaaa"
EXECUTOR_SESSION = "fedcba98-7654-3210-fedc-ba9876543210"
EPISODE_ID = "ep-deadbeef"
SUFFIX = "deadbeef1234"
SUMMARY = "<summary>fixed summary that omits the notice</summary>"
NOTICE_MARKER = "Plan mode is active. Plan document:"


def _logging(save_dir: Path) -> SessionLoggingConfig:
    return SessionLoggingConfig(enabled=True, save_dir=str(save_dir))


def _loop(
    save_dir: Path,
    *,
    session_id: str = SESSION_ID,
    backend: FakeBackend | None = None,
    cwd: Path | None = None,
):
    return build_test_agent_loop(
        config=build_test_vibe_config(session_logging=_logging(save_dir)),
        backend=backend or FakeBackend([]),
        cwd=cwd,
        session_id=session_id,
    )


@pytest.fixture
def save_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "sessions"
    directory.mkdir()
    return directory


@pytest.fixture
def episode(tmp_path: Path) -> Path:
    directory = tmp_path / "tasks" / f"plan-{SUFFIX}"
    directory.mkdir(parents=True)
    ensure_episode_fence(directory)
    create_sidecar(
        directory,
        PlanEpisodeSidecar(episode_id=EPISODE_ID, draft_path=directory / "plan.md"),
    )
    return directory


def _domain(save_dir: Path) -> LoggingDomain:
    return LoggingDomain(save_dir=str(save_dir), session_prefix="session")


def _record(
    directory: Path, *, scan_domain: Path | None = None, state: str = STATE_DRAFT
) -> PlanModeRecord:
    return PlanModeRecord(
        plan_path=str(directory / "plan.md"),
        theme="Restore the binding",
        planner_model="alpha",
        planner_thinking="medium",
        episode_id=EPISODE_ID,
        sidecar_path=str(sidecar_path(directory)),
        draft_path=str(directory / "plan.md"),
        state=state,
        scan_domain=(
            None
            if scan_domain is None
            else PlanScanDomain(save_dir=str(scan_domain), session_prefix="session")
        ),
    )


def _write_state(
    directory: Path, state: str, *, executor_session_id: str | None = None
) -> None:
    sidecar = read_sidecar(directory)
    assert sidecar is not None
    write_sidecar(
        directory,
        replace(sidecar, state=state, executor_session_id=executor_session_id),
    )


def _plan_mode(loop) -> PlanModeRecord | None:
    metadata = loop.session_logger.session_metadata
    assert metadata is not None
    return metadata.plan_mode


def _boundary(loop) -> str:
    return next(
        message.content or ""
        for message in reversed(loop.messages)
        if message.context_boundary == "compaction"
    )


@pytest.mark.asyncio
class TestForkRestore:
    async def test_the_fork_is_rebound_to_the_source_sidecar(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        source = _loop(save_dir, cwd=tmp_path)
        await source.session_logger.persist_plan_mode(_record(episode))

        forked = _loop(save_dir, session_id=FORK_SESSION, cwd=tmp_path)
        _inherit_plan_mode(source, forked)

        inherited = _plan_mode(forked)
        assert inherited is not None
        assert inherited.episode_id == EPISODE_ID
        # The same shared-disk arbiter: no fork-local identity copy exists.
        assert inherited.sidecar_path == str(sidecar_path(episode))
        assert inherited.state == STATE_DRAFT
        assert inherited is not _plan_mode(source)

    async def test_an_accepted_episode_forks_as_terminal_history(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        source = _loop(save_dir, cwd=tmp_path)
        # The source's copy is stale-active; the sidecar is the authority.
        await source.session_logger.persist_plan_mode(_record(episode))
        _write_state(episode, STATE_ACCEPTED, executor_session_id=EXECUTOR_SESSION)

        forked = _loop(save_dir, session_id=FORK_SESSION, cwd=tmp_path)
        _inherit_plan_mode(source, forked)

        inherited = _plan_mode(forked)
        assert inherited is not None
        assert inherited.state == STATE_ACCEPTED
        assert inherited.state in TERMINAL_STATES
        observation = observe_episode(
            episode, record=inherited, session_id=FORK_SESSION
        )
        assert observation.terminal
        assert observation.executor_session_id == EXECUTOR_SESSION

    async def test_a_cancelled_episode_forks_as_terminal_history(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        source = _loop(save_dir, cwd=tmp_path)
        await source.session_logger.persist_plan_mode(_record(episode))
        _write_state(episode, STATE_CANCELLED)

        forked = _loop(save_dir, session_id=FORK_SESSION, cwd=tmp_path)
        _inherit_plan_mode(source, forked)

        inherited = _plan_mode(forked)
        assert inherited is not None
        assert inherited.state == STATE_CANCELLED
        assert inherited.state in TERMINAL_STATES

    async def test_an_accepting_episode_forks_with_the_accepting_status(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        claim_accept(episode, session_id=SESSION_ID, owner_domain=_domain(save_dir))
        source = _loop(save_dir, cwd=tmp_path)
        # The record's own label is draft; the sidecar says accepting.
        await source.session_logger.persist_plan_mode(_record(episode))

        forked = _loop(save_dir, session_id=FORK_SESSION, cwd=tmp_path)
        _inherit_plan_mode(source, forked)

        inherited = _plan_mode(forked)
        assert inherited is not None
        assert inherited.state == STATE_ACCEPTING

    async def test_a_second_fork_cannot_double_accept(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        source = _loop(save_dir, cwd=tmp_path)
        await source.session_logger.persist_plan_mode(_record(episode))

        first = _loop(save_dir, session_id=FORK_SESSION, cwd=tmp_path)
        _inherit_plan_mode(source, first)
        claim_accept(episode, session_id=FORK_SESSION, owner_domain=_domain(save_dir))

        # The second fork re-observes the SAME sidecar and inherits accepting.
        second = _loop(save_dir, session_id=SECOND_FORK_SESSION, cwd=tmp_path)
        _inherit_plan_mode(source, second)
        inherited = _plan_mode(second)
        assert inherited is not None
        assert inherited.state == STATE_ACCEPTING
        with pytest.raises(PlanEpisodeStateError):
            claim_accept(
                episode, session_id=SECOND_FORK_SESSION, owner_domain=_domain(save_dir)
            )

        _write_state(episode, STATE_ACCEPTED, executor_session_id=EXECUTOR_SESSION)
        third = _loop(save_dir, session_id=SECOND_FORK_SESSION, cwd=tmp_path)
        _inherit_plan_mode(source, third)
        accepted = _plan_mode(third)
        assert accepted is not None
        assert accepted.state == STATE_ACCEPTED
        with pytest.raises(PlanEpisodeTerminalError):
            claim_accept(
                episode, session_id=SECOND_FORK_SESSION, owner_domain=_domain(save_dir)
            )

    async def test_a_missing_sidecar_keeps_the_terminal_record_as_history(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        sidecar_path(episode).unlink()
        source = _loop(save_dir, cwd=tmp_path)
        await source.session_logger.persist_plan_mode(
            _record(episode, state=STATE_ACCEPTED)
        )

        forked = _loop(save_dir, session_id=FORK_SESSION, cwd=tmp_path)
        _inherit_plan_mode(source, forked)

        inherited = _plan_mode(forked)
        assert inherited is not None
        assert inherited.state == STATE_ACCEPTED

    async def test_an_untrusted_sidecar_does_not_fail_the_fork(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        sidecar_path(episode).write_bytes(b"{not json")
        source = _loop(save_dir, cwd=tmp_path)
        await source.session_logger.persist_plan_mode(_record(episode))

        forked = _loop(save_dir, session_id=FORK_SESSION, cwd=tmp_path)
        _inherit_plan_mode(source, forked)

        # The copy travels; the restore surfaces observe and report the failure.
        inherited = _plan_mode(forked)
        assert inherited is not None
        assert inherited.episode_id == EPISODE_ID
        with pytest.raises(PlanSidecarError):
            observe_episode(episode, record=inherited, session_id=FORK_SESSION)


async def _rewind_to_start(loop) -> None:
    backend_done = [event async for event in loop.act("plan the widget")]
    assert backend_done
    rewindable = loop.rewind_manager.get_rewindable_messages()
    assert len(rewindable) == 1
    await loop.rewind_manager.rewind_to_message(rewindable[0][0], restore_files=False)


@pytest.mark.asyncio
class TestRewindRestore:
    async def _prepare(self, tmp_path: Path, save_dir: Path, record: PlanModeRecord):
        loop = _loop(
            save_dir,
            backend=FakeBackend([[mock_llm_chunk(content="noted")]]),
            cwd=tmp_path,
        )
        await loop.session_logger.persist_plan_mode(record)
        return loop

    async def test_the_rewind_fork_rotates_the_id_and_rebinds_the_draft_record(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        loop = await self._prepare(tmp_path, save_dir, _record(episode))
        departing_id = loop.session_id

        await _rewind_to_start(loop)

        assert loop.session_id != departing_id
        metadata = loop.session_logger.session_metadata
        assert metadata is not None
        restored = metadata.plan_mode
        assert restored is not None
        assert restored.episode_id == EPISODE_ID
        assert restored.state == STATE_DRAFT
        assert restored.sidecar_path == str(sidecar_path(episode))

    async def test_a_pre_accept_checkpoint_of_an_accepted_episode_shows_accepted(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        _write_state(episode, STATE_ACCEPTED, executor_session_id=EXECUTOR_SESSION)
        loop = await self._prepare(tmp_path, save_dir, _record(episode))

        await _rewind_to_start(loop)

        metadata = loop.session_logger.session_metadata
        assert metadata is not None
        assert metadata.plan_mode is not None
        assert metadata.plan_mode.state == STATE_ACCEPTED

    async def test_a_cancelled_episode_never_resurrects_as_active(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        _write_state(episode, STATE_CANCELLED)
        loop = await self._prepare(tmp_path, save_dir, _record(episode))

        await _rewind_to_start(loop)

        metadata = loop.session_logger.session_metadata
        assert metadata is not None
        assert metadata.plan_mode is not None
        assert metadata.plan_mode.state == STATE_CANCELLED
        assert metadata.plan_mode.state in TERMINAL_STATES

    async def test_a_terminal_record_without_a_sidecar_stays_terminal_history(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        sidecar_path(episode).unlink()
        loop = await self._prepare(
            tmp_path, save_dir, _record(episode, state=STATE_ACCEPTED)
        )

        await _rewind_to_start(loop)

        metadata = loop.session_logger.session_metadata
        assert metadata is not None
        assert metadata.plan_mode is not None
        assert metadata.plan_mode.state == STATE_ACCEPTED

    async def test_a_non_terminal_record_without_a_sidecar_is_never_restored_as_draft(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        sidecar_path(episode).unlink()
        # A clean scan domain: no session meta marks the episode, so the mirror
        # scan misses rather than surfacing a recorded executor.
        empty_domain = tmp_path / "scan-sessions"
        empty_domain.mkdir()
        loop = await self._prepare(
            tmp_path, save_dir, _record(episode, scan_domain=empty_domain)
        )

        await _rewind_to_start(loop)

        metadata = loop.session_logger.session_metadata
        assert metadata is not None
        restored = metadata.plan_mode
        assert restored is not None
        # The surfaces observe the restored record sidecar-first: a clean scan
        # miss is unresolvable, never an armed draft.
        observation = observe_episode(
            episode, record=restored, session_id=loop.session_id
        )
        assert observation.row == RECOVERY_ROW_UNRESOLVABLE


@pytest.mark.asyncio
class TestCompactionReinjection:
    async def _compacted(
        self, tmp_path: Path, save_dir: Path, record: PlanModeRecord
    ) -> str:
        loop = _loop(
            save_dir,
            backend=FakeBackend([[mock_llm_chunk(content=SUMMARY)]]),
            cwd=tmp_path,
        )
        loop.messages.append(LLMMessage(role=Role.user, content="Hello"))
        loop.stats.context_tokens = 100
        await loop.session_logger.persist_plan_mode(record)
        await loop.compact()
        return _boundary(loop)

    async def test_a_draft_episode_reinjects_the_document_and_its_status(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        content = await self._compacted(
            tmp_path, save_dir, _record(episode, scan_domain=save_dir)
        )

        assert NOTICE_MARKER in content
        assert f"Plan episode {EPISODE_ID}: {STATE_DRAFT}." in content

    async def test_an_accepting_episode_reinjects_the_accepting_status(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        claim_accept(episode, session_id=SESSION_ID, owner_domain=_domain(save_dir))
        content = await self._compacted(
            tmp_path, save_dir, _record(episode, scan_domain=save_dir)
        )

        assert NOTICE_MARKER in content
        assert f"Plan episode {EPISODE_ID}: {STATE_ACCEPTING}." in content

    async def test_an_accepted_episode_reinjects_as_terminal_not_active(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        _write_state(episode, STATE_ACCEPTED, executor_session_id=EXECUTOR_SESSION)
        content = await self._compacted(
            tmp_path, save_dir, _record(episode, scan_domain=save_dir)
        )

        assert NOTICE_MARKER not in content
        assert f"Plan episode {EPISODE_ID} is {STATE_ACCEPTED} (terminal)" in content
        assert "plan mode is not active in this session" in content

    async def test_a_cancelled_episode_reinjects_as_terminal_not_active(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        _write_state(episode, STATE_CANCELLED)
        content = await self._compacted(
            tmp_path, save_dir, _record(episode, scan_domain=save_dir)
        )

        assert NOTICE_MARKER not in content
        assert f"Plan episode {EPISODE_ID} is {STATE_CANCELLED} (terminal)" in content

    async def test_a_missing_sidecar_with_a_clean_scan_is_reinjected_as_unresolvable(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        sidecar_path(episode).unlink()
        content = await self._compacted(
            tmp_path, save_dir, _record(episode, scan_domain=save_dir)
        )

        assert NOTICE_MARKER not in content
        assert f"Plan episode {EPISODE_ID} is unresolvable" in content

    async def test_an_untrusted_scan_is_reported_not_read_as_a_clean_miss(
        self, tmp_path: Path, save_dir: Path, episode: Path
    ) -> None:
        sidecar_path(episode).unlink()
        missing_domain = tmp_path / "moved-away"
        content = await self._compacted(
            tmp_path, save_dir, _record(episode, scan_domain=missing_domain)
        )

        assert NOTICE_MARKER not in content
        assert "Plan episode state could not be validated" in content
