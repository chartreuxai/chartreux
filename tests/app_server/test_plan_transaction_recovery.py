"""Transaction crash-recovery suite (P10 chunk 8, design §4.4).

Every reachable boundary of the acceptance transaction's crash table is
exercised here: both accepting rows with their re-offer and cancel legs and
both in-fence executor re-scans, the possibly-executed row with its owner
adjudication, the accepted-with-pending-rename intermediate with the completion
rule at every observation surface, the cancel-branch rename window with its
either-name locate and verified-no-op continuation, the attempt-scoped
``c1_verified`` lifecycle, the C2/C3/C4 failure rows, competing owners, foreign
sessions, and the cleanup/provenance interactions.

Crash residue is produced the honest way: the real transaction runs up to its
boundary, or that boundary's write or scan is made to fail. No test repairs the
artifact it just observed — the assertions read the state the boundary left
behind and the recovery legs then act on it.

``tests/app_server/test_plan_handoff_integrity.py`` exercises the transaction's
own paths; this suite exercises the crash boundaries between them. Every
handler exercise runs over the memory transport with the fake backend (a
failing provider is a scripted transport failure), so no provider is contacted.
"""

from __future__ import annotations

import base64
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, NoReturn

import httpx
import pytest

from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import ProtocolErrorCode, SessionPlanHandoffParams
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core import plan_episode
from chartreux.core.config.models import ModelConfig, SessionLoggingConfig
from chartreux.core.plan_episode import (
    DOMAIN_REFUSAL,
    RECOVERY_ROW_ACCEPTED,
    RECOVERY_ROW_ACCEPTING_NO_SNAPSHOT,
    RECOVERY_ROW_ACCEPTING_SNAPSHOT,
    RECOVERY_ROW_CANCELLED,
    RECOVERY_ROW_EXECUTED,
    RECOVERY_ROW_POSSIBLY_EXECUTED,
    RECOVERY_ROW_UNRESOLVABLE,
    SEED_DOCUMENT_BEGIN,
    SNAPSHOT_RENAME_NOOP,
    SNAPSHOT_RENAME_PENDING,
    SNAPSHOT_RENAMED,
    STATE_ACCEPTED,
    STATE_ACCEPTING,
    STATE_CANCELLED,
    STATE_DRAFT,
    CommitAttempt,
    CommittedEpisode,
    EpisodeObservation,
    LoggingDomain,
    PlanArbiterUnresolvableError,
    PlanDigestMismatchError,
    PlanEpisodeOwnerError,
    PlanEpisodePossiblyExecutedError,
    PlanEpisodeScanError,
    PlanEpisodeSidecar,
    PlanEpisodeStateError,
    PlanEpisodeTerminalError,
    PlanEpisodeWriteError,
    PlanFenceBusyError,
    PlanTransitionError,
    PreparedAccept,
    accepted_snapshot_path,
    adjudicate_possibly_executed,
    begin_commit,
    claim_accept,
    cleanup_episode_workspace,
    complete_accepted_rename,
    create_sidecar,
    ensure_episode_fence,
    fence_path,
    locate_episode_snapshot,
    observe_episode,
    prepare_accept,
    provenance_snapshot_path,
    publish_consent_snapshot,
    read_sidecar,
    rebind_draft,
    recover_cancel_accepting,
    recover_release_claim,
    release_claim,
    rename_accepted_snapshot,
    reopen_accepting_episode,
    restore_episode_record,
    restored_plan_record,
    sha256_bytes,
    sidecar_path,
    surviving_snapshot_name,
)
from chartreux.core.plan_mode import PLAN_MARKER_FILENAME
from chartreux.core.session_types import PlanModeRecord, PlanScanDomain
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    legacy_backend,
)
from tests.stubs.fake_backend import FakeBackend

SESSION_ID = "01234567-89ab-cdef-0123-deadbeef1234"
OTHER_SESSION = "fedcba98-7654-3210-fedc-ba9876543210"
EXECUTOR_SESSION = "exec-session-0001"
EPISODE_ID = "ep-deadbeef"
DOCUMENT = b"# Plan\r\n\r\n## Summary\r\nKeep these exact bytes \xe2\x80\x94 raw.\r\n"


# --- Fixtures and helpers ---------------------------------------------------


class _CrashBoundary(Exception):
    """A simulated process death at a named transaction boundary."""


def _make_episode(root: Path, name: str = "plan-deadbeef1234") -> Path:
    directory = root / name
    directory.mkdir()
    ensure_episode_fence(directory)
    create_sidecar(
        directory,
        PlanEpisodeSidecar(episode_id=EPISODE_ID, draft_path=directory / "plan.md"),
    )
    (directory / "plan.md").write_bytes(DOCUMENT)
    return directory


@pytest.fixture
def episode(tmp_path: Path) -> Path:
    return _make_episode(tmp_path)


def _sidecar(directory: Path) -> PlanEpisodeSidecar:
    sidecar = read_sidecar(directory)
    assert sidecar is not None
    return sidecar


def _persisted(directory: Path) -> dict[str, Any]:
    return json.loads(sidecar_path(directory).read_text(encoding="utf-8"))


def _snapshot(directory: Path) -> Path:
    return accepted_snapshot_path(directory, EPISODE_ID)


def _provenance(directory: Path, generation: int = 1) -> Path:
    return provenance_snapshot_path(directory, EPISODE_ID, generation)


def _observe(
    directory: Path,
    *,
    domain: LoggingDomain | None = None,
    session_id: str = SESSION_ID,
    record: PlanModeRecord | None = None,
) -> EpisodeObservation:
    return observe_episode(
        directory, record=record, session_id=session_id, owner_domain=domain
    )


def _scan_domain(root: Path, name: str = "sessions") -> LoggingDomain:
    save_dir = root / name
    save_dir.mkdir(parents=True, exist_ok=True)
    return LoggingDomain(save_dir=str(save_dir), session_prefix="session")


def _session_with_marker(
    save_dir: Path, *, suffix: str, session_id: str, episode_id: str = EPISODE_ID
) -> Path:
    """Write a C3 EXECUTOR marker: the executor's accepting record, not a mirror."""
    directory = save_dir / f"session_20260101_120000_{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(
        json.dumps({
            "session_id": session_id,
            "plan_mode": {
                "episode_id": episode_id,
                "state": STATE_ACCEPTING,
                "consent_sha256": "c" * 64,
                "snapshot_path": f"accepted-{episode_id}.md",
            },
        }),
        encoding="utf-8",
    )
    return directory


def _session_with_planning_mirror(
    save_dir: Path, *, suffix: str, session_id: str, state: str = STATE_DRAFT
) -> Path:
    """Write a PLANNING session's own episode mirror (same id, its own state)."""
    directory = save_dir / f"session_20260101_120000_{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(
        json.dumps({
            "session_id": session_id,
            "plan_mode": {"episode_id": EPISODE_ID, "state": state},
        }),
        encoding="utf-8",
    )
    return directory


def _session_metas(save_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(meta_path.read_text(encoding="utf-8"))
        for meta_path in sorted(save_dir.glob("session_*/meta.json"))
    ]


def _session_meta(save_dir: Path, session_id: str) -> dict[str, Any] | None:
    for payload in _session_metas(save_dir):
        if payload.get("session_id") == session_id:
            return payload
    return None


def _executor_marker(save_dir: Path) -> dict[str, Any] | None:
    for payload in _session_metas(save_dir):
        record = payload.get("plan_mode")
        if (
            isinstance(record, dict)
            and record.get("episode_id") == EPISODE_ID
            and record.get("state") == STATE_ACCEPTING
        ):
            return payload
    return None


def _seeded_messages(loop: Any) -> list[str]:
    return [
        message.content or ""
        for message in loop.messages
        if SEED_DOCUMENT_BEGIN in (message.content or "")
    ]


def _logging_domain(loop: Any) -> LoggingDomain:
    logger = loop.session_logger
    assert logger.save_dir is not None
    return LoggingDomain(save_dir=str(logger.save_dir), session_prefix="session")


def _prepare(
    directory: Path,
    *,
    domain: LoggingDomain,
    session_id: str = SESSION_ID,
    previewed: bytes = DOCUMENT,
) -> PreparedAccept:
    return prepare_accept(
        directory,
        episode_id=EPISODE_ID,
        session_id=session_id,
        owner_domain=domain,
        previewed_bytes=previewed,
        previewed_sha256=sha256_bytes(previewed),
    )


def _begin(
    directory: Path,
    *,
    domain: LoggingDomain,
    prepared: PreparedAccept | None = None,
    session_id: str = SESSION_ID,
) -> CommitAttempt:
    if prepared is None:
        prepared = _prepare(directory, domain=domain, session_id=session_id)
    return begin_commit(
        directory,
        episode_id=EPISODE_ID,
        session_id=session_id,
        owner_domain=domain,
        carried_bytes=prepared.retained_bytes,
        snapshot_path=prepared.snapshot_path,
    )


def _commit(
    directory: Path,
    *,
    domain: LoggingDomain,
    prepared: PreparedAccept | None = None,
    executor_session_id: str = EXECUTOR_SESSION,
) -> CommittedEpisode:
    attempt = _begin(directory, domain=domain, prepared=prepared)
    try:
        return attempt.finalize(executor_session_id=executor_session_id)
    finally:
        attempt.close()


def _crash_at_the_rename(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise _CrashBoundary(
        "simulated process death between the terminal flip and the derived rename"
    )


def _pending_rename(*_args: Any, **_kwargs: Any) -> str:
    raise PlanTransitionError(
        "injected: the derived rename could not run at this observation"
    )


def _draft_write_refuses(directory: Path, sidecar: PlanEpisodeSidecar) -> None:
    """Fail the return-to-draft state write; a non-draft write passes through."""
    if sidecar.state == STATE_DRAFT:
        raise OSError("injected: the return-to-draft state write failed")
    plan_episode.durable_replace_plan(
        sidecar_path(directory), plan_episode.encode_sidecar(sidecar)
    )


def _crashed_after_flip(
    episode: Path, domain: LoggingDomain, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive a commit to its terminal flip, then die before the derived rename."""
    attempt = _begin(episode, domain=domain)
    with monkeypatch.context() as patch:
        patch.setattr(plan_episode, "rename_accepted_snapshot", _crash_at_the_rename)
        try:
            with pytest.raises(_CrashBoundary):
                attempt.finalize(executor_session_id=EXECUTOR_SESSION)
        finally:
            # A crashed holder's fence is released by the OS; here, explicitly.
            attempt.close()


async def _cancel_window_crash(
    episode: Path, domain: LoggingDomain, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crash between the cancel-branch rename and the return-to-draft write."""
    attempt = _begin(episode, domain=domain)
    attempt.close()
    with monkeypatch.context() as patch:
        patch.setattr(plan_episode, "write_sidecar", _draft_write_refuses)
        with pytest.raises(PlanEpisodeWriteError):
            await recover_cancel_accepting(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=_RecordSink(),
            )


def _committed_with_an_occupied_provenance(
    episode: Path, domain: LoggingDomain
) -> CommittedEpisode:
    """A commit whose attempt provenance name is already occupied."""
    _provenance(episode, 1).write_bytes(b"survivor\n")
    return _commit(episode, domain=domain)


def _source_record(
    directory: Path, *, state: str = STATE_DRAFT, episode_id: str = EPISODE_ID
) -> PlanModeRecord:
    return PlanModeRecord(
        plan_path=str(directory / "plan.md"),
        theme="Keep these exact bytes",
        planner_model="alpha",
        planner_thinking="medium",
        episode_id=episode_id,
        sidecar_path=str(sidecar_path(directory)),
        draft_path=str(directory / "plan.md"),
        state=state,
    )


class _RecordSink:
    def __init__(self) -> None:
        self.records: list[PlanModeRecord] = []

    async def __call__(self, record: PlanModeRecord) -> None:
        self.records.append(record)


async def _refuse_record(_record: PlanModeRecord) -> None:
    raise PlanEpisodeWriteError("injected restoration failure")


# --- The handler harness ----------------------------------------------------


def _handler_plan_record(plan_dir: Path, save_dir: Path) -> PlanModeRecord:
    return PlanModeRecord(
        plan_path=str(plan_dir / "plan.md"),
        theme="Keep these exact bytes",
        planner_model="alpha",
        planner_thinking="medium",
        episode_id=EPISODE_ID,
        sidecar_path=str(sidecar_path(plan_dir)),
        draft_path=str(plan_dir / "plan.md"),
        state=STATE_DRAFT,
        scan_domain=PlanScanDomain(save_dir=str(save_dir), session_prefix="session"),
    )


async def _handler_server(
    tmp_path: Path, plan_dir: Path, *, backend: FakeBackend | None = None
) -> tuple[Any, Any, Any]:
    """A logging-enabled app server whose root session carries the plan record."""
    save_dir = tmp_path / "sessions"
    config = build_test_vibe_config(
        active_model="alpha",
        models=[ModelConfig(name="alpha", provider="mistral", alias="alpha")],
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(save_dir)),
    )
    loop = build_test_agent_loop(
        config=config, session_id=SESSION_ID, backend=backend or FakeBackend()
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    await loop.session_logger.persist_plan_mode_durable(
        _handler_plan_record(plan_dir, save_dir)
    )
    return legacy_backend(server).handler, session, loop


def _handoff_params(
    directory: Path, *, content: bytes = DOCUMENT
) -> SessionPlanHandoffParams:
    return SessionPlanHandoffParams(
        session_id=SESSION_ID,
        executor_model="alpha",
        executor_thinking="medium",
        executor_identity="orchestrator",
        plan_path=str(directory / "plan.md"),
        episode_id=EPISODE_ID,
        snapshot_path=str(_snapshot(directory)),
        snapshot_sha256=sha256_bytes(content),
        snapshot_bytes_b64=base64.b64encode(content).decode("ascii"),
    )


def _ignore_selection_mismatch(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    monkeypatch.setattr(
        handler, "_plan_handoff_selection_mismatch", lambda *args, **kwargs: None
    )


# --- Row 2: accepting, own session, NO snapshot -----------------------------


class TestRowAcceptingNoSnapshot:
    """A crash before (or during) P3 leaves ``accepting`` with no snapshot."""

    def test_a_crash_after_p1_leaves_the_accepting_no_snapshot_row(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        claim_accept(episode, session_id=SESSION_ID, owner_domain=domain)

        observation = _observe(episode, domain=domain)

        assert observation.row == RECOVERY_ROW_ACCEPTING_NO_SNAPSHOT
        assert observation.state == STATE_ACCEPTING
        assert observation.recoverable
        assert observation.owner
        assert observation.snapshot_name is None
        assert observation.executor_session_id is None
        assert not _snapshot(episode).exists()
        assert fence_path(episode).exists()

    def test_a_partial_snapshot_left_by_the_crash_is_unlinked_on_release(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        claim_accept(episode, session_id=SESSION_ID, owner_domain=domain)
        # The crash residue: a torn P3 publication.
        _snapshot(episode).write_bytes(DOCUMENT[:12])

        released = recover_release_claim(
            episode, episode_id=EPISODE_ID, session_id=SESSION_ID, owner_domain=domain
        )

        assert released.state == STATE_DRAFT
        assert released.claiming_session_id is None
        assert released.c1_verified is False
        assert released.consent_sha256 is None
        assert not _snapshot(episode).exists()
        assert fence_path(episode).exists()

    def test_the_released_episode_re_offers_from_scratch_with_a_new_generation(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        claim_accept(episode, session_id=SESSION_ID, owner_domain=domain)
        recover_release_claim(
            episode, episode_id=EPISODE_ID, session_id=SESSION_ID, owner_domain=domain
        )

        prepared = _prepare(episode, domain=domain)

        assert prepared.sidecar.state == STATE_ACCEPTING
        assert prepared.sidecar.attempt_generation == 2
        assert prepared.sidecar.consent_sha256 == sha256_bytes(DOCUMENT)
        assert _snapshot(episode).read_bytes() == DOCUMENT

    def test_a_foreign_session_cannot_release_the_crashed_claim(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        claim_accept(episode, session_id=SESSION_ID, owner_domain=domain)

        with pytest.raises(PlanEpisodeOwnerError):
            recover_release_claim(
                episode,
                episode_id=EPISODE_ID,
                session_id=OTHER_SESSION,
                owner_domain=domain,
            )
        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _sidecar(episode).claiming_session_id == SESSION_ID

        # The owner's recovery still proceeds: the refusal moved nothing.
        released = recover_release_claim(
            episode, episode_id=EPISODE_ID, session_id=SESSION_ID, owner_domain=domain
        )
        assert released.state == STATE_DRAFT

    def test_a_foreign_logging_domain_cannot_release_the_crashed_claim(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        elsewhere = LoggingDomain(
            save_dir=str(tmp_path / "elsewhere"), session_prefix="session"
        )
        claim_accept(episode, session_id=SESSION_ID, owner_domain=domain)

        with pytest.raises(PlanEpisodeOwnerError) as caught:
            recover_release_claim(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=elsewhere,
            )

        assert DOMAIN_REFUSAL in str(caught.value)
        assert _sidecar(episode).state == STATE_ACCEPTING


# --- Row 3: accepting, snapshot present, NO executor meta -------------------


class TestRowAcceptingSnapshotNoExecutor:
    """A crash after P3 (and possibly after C2) leaves the snapshot row."""

    def test_a_crash_after_p3_lands_in_the_snapshot_row(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)

        observation = _observe(episode, domain=domain)

        assert observation.row == RECOVERY_ROW_ACCEPTING_SNAPSHOT
        assert observation.state == STATE_ACCEPTING
        assert observation.recoverable
        assert observation.owner
        assert observation.snapshot_name == _snapshot(episode).name
        assert observation.executor_session_id is None
        assert observation.record_mismatch is False

    @pytest.mark.asyncio
    async def test_the_re_offer_restores_the_record_from_the_sidecar(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        prepared = _prepare(episode, domain=domain)
        sink = _RecordSink()

        offer = await reopen_accepting_episode(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            # A C2 crash may leave the record at accepted while the sidecar is
            # still accepting; the sidecar is authoritative.
            record=_source_record(episode, state=STATE_ACCEPTED),
            write_record=sink,
        )

        assert offer.content == DOCUMENT
        assert offer.snapshot_path == _snapshot(episode)
        restored = sink.records[0]
        assert restored.state == STATE_ACCEPTING
        assert restored.episode_id == EPISODE_ID
        assert restored.consent_sha256 == sha256_bytes(DOCUMENT)
        assert restored.attempt_generation == prepared.sidecar.attempt_generation
        assert restored.scan_domain == PlanScanDomain(
            save_dir=domain.save_dir, session_prefix="session"
        )
        assert restored.snapshot_path == _snapshot(episode).name
        # Never auto-executes: the claim stays held for the confirm.
        sidecar = _sidecar(episode)
        assert sidecar.state == STATE_ACCEPTING
        assert sidecar.c1_verified is False
        assert sidecar.executor_session_id is None
        assert _session_metas(Path(domain.save_dir)) == []

    @pytest.mark.asyncio
    async def test_the_re_offer_fails_closed_when_the_bytes_are_absent(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        _snapshot(episode).unlink()

        with pytest.raises(PlanArbiterUnresolvableError):
            await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=_RecordSink(),
            )

        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _sidecar(episode).consent_sha256 == sha256_bytes(DOCUMENT)

    @pytest.mark.asyncio
    async def test_the_re_offer_refuses_a_torn_snapshot(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        # A torn survivor: the bytes are not the consented bytes.
        _snapshot(episode).write_bytes(DOCUMENT[:12])

        with pytest.raises(PlanDigestMismatchError):
            await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=_RecordSink(),
            )

        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _snapshot(episode).read_bytes() == DOCUMENT[:12]

    @pytest.mark.asyncio
    async def test_a_restoration_write_failure_surfaces_unresolvable(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)

        with pytest.raises(PlanArbiterUnresolvableError):
            await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=_refuse_record,
            )

        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _snapshot(episode).read_bytes() == DOCUMENT

    @pytest.mark.asyncio
    async def test_the_cancel_leg_returns_the_episode_and_record_to_draft(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        sink = _RecordSink()

        released = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(episode, state=STATE_ACCEPTED),
            write_record=sink,
        )

        assert released.state == STATE_DRAFT
        assert _sidecar(episode) == released
        assert _persisted(episode)["state"] == STATE_DRAFT
        assert _persisted(episode)["c1_verified"] is False
        assert released.consent_sha256 is None
        assert not _snapshot(episode).exists()
        restored = sink.records[0]
        assert restored.state == STATE_DRAFT
        assert restored.snapshot_path is None
        assert restored.consent_sha256 is None

    @pytest.mark.asyncio
    async def test_the_cancel_leg_refuses_a_record_it_cannot_restore(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)

        with pytest.raises(PlanArbiterUnresolvableError):
            await recover_cancel_accepting(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=_refuse_record,
            )

        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _snapshot(episode).is_file()

    @pytest.mark.asyncio
    async def test_the_re_offer_continuation_commits_from_the_held_claim(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        assert _sidecar(episode).c1_verified is False

        offer = await reopen_accepting_episode(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(episode),
            write_record=_RecordSink(),
        )
        # The re-offer keeps the claim held: no settle, no fresh attempt.
        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _sidecar(episode).attempt_generation == 1
        # Post-consent the draft is never consulted; the located snapshot is the
        # carried artifact.
        (episode / "plan.md").unlink()
        attempt = begin_commit(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            carried_bytes=offer.content,
            snapshot_path=offer.snapshot_path,
        )
        try:
            committed = attempt.finalize(executor_session_id=EXECUTOR_SESSION)
        finally:
            attempt.close()

        assert committed.sidecar.state == STATE_ACCEPTED
        assert committed.rename_outcome == SNAPSHOT_RENAMED
        assert _provenance(episode, 1).read_bytes() == DOCUMENT
        assert not _snapshot(episode).exists()

    @pytest.mark.asyncio
    async def test_the_re_offer_continuation_re_enters_commit_end_to_end(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler, session, loop = await _handler_server(tmp_path, episode)
        _ignore_selection_mismatch(monkeypatch, handler)
        try:
            # The crash residue at the end of P3: the claim is held and the
            # snapshot holds the consented bytes.
            await handler._plan_handoff_prepare(_handoff_params(episode))
            assert _sidecar(episode).state == STATE_ACCEPTING
            offer = await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=handler._plan_handoff_logging_domain(),
                record=_source_record(episode),
                write_record=_RecordSink(),
            )
            # Post-consent the draft is never consulted again.
            (episode / "plan.md").unlink()
            await handler._plan_handoff(_handoff_params(episode, content=offer.content))
        finally:
            await session.close()

        sidecar = _sidecar(episode)
        assert sidecar.state == STATE_ACCEPTED
        assert sidecar.executor_session_id == loop.session_id
        # One attempt, continued: the re-offer did not start a second one.
        assert sidecar.attempt_generation == 1
        assert _provenance(episode, 1).read_bytes() == DOCUMENT
        assert not _snapshot(episode).exists()
        assert DOCUMENT.decode().strip() in _seeded_messages(loop)[0]


# --- The possibly-executed row ----------------------------------------------


class TestRowPossiblyExecuted:
    """An episode-marked executor meta: never auto-quit, never auto-retry."""

    def test_an_executor_marker_moves_the_surface_to_possibly_executed(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        _session_with_marker(
            Path(domain.save_dir), suffix="exec", session_id=EXECUTOR_SESSION
        )

        observation = _observe(episode, domain=domain)

        assert observation.row == RECOVERY_ROW_POSSIBLY_EXECUTED
        assert observation.executor_session_id == EXECUTOR_SESSION
        assert observation.possibly_executed
        assert not observation.recoverable
        assert not observation.terminal
        assert observation.snapshot_name == _snapshot(episode).name

    @pytest.mark.asyncio
    async def test_the_re_offer_refuses_once_the_executor_is_found(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        _session_with_marker(
            Path(domain.save_dir), suffix="exec", session_id=EXECUTOR_SESSION
        )

        with pytest.raises(PlanEpisodePossiblyExecutedError) as caught:
            await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=_RecordSink(),
            )

        assert caught.value.executor_session_id == EXECUTOR_SESSION
        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _snapshot(episode).read_bytes() == DOCUMENT

    @pytest.mark.asyncio
    async def test_the_cancel_rescan_discovers_a_marker_and_moves_the_surface(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        assert _observe(episode, domain=domain).row == RECOVERY_ROW_ACCEPTING_SNAPSHOT
        # The marker appears AFTER the observation and inside the cancel's fence.
        _session_with_marker(
            Path(domain.save_dir), suffix="late", session_id=EXECUTOR_SESSION
        )

        with pytest.raises(PlanEpisodePossiblyExecutedError) as caught:
            await recover_cancel_accepting(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=_RecordSink(),
            )

        assert caught.value.executor_session_id == EXECUTOR_SESSION
        # Nothing moved, and the possibly-executed row owns the episode now.
        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _snapshot(episode).is_file()
        assert _observe(episode, domain=domain).row == RECOVERY_ROW_POSSIBLY_EXECUTED

    @pytest.mark.asyncio
    async def test_the_cancel_rescan_is_bounded_to_the_recorded_domain(
        self, episode: Path, tmp_path: Path
    ) -> None:
        recorded = _scan_domain(tmp_path, "recorded")
        elsewhere = _scan_domain(tmp_path, "elsewhere")
        _prepare(episode, domain=recorded)
        _session_with_marker(
            Path(elsewhere.save_dir), suffix="out", session_id=EXECUTOR_SESSION
        )

        released = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=recorded,
            record=_source_record(episode),
            write_record=_RecordSink(),
        )

        assert released.state == STATE_DRAFT
        assert not _snapshot(episode).exists()

    def test_mark_accepted_flips_and_completes_the_rename(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        _session_with_marker(
            Path(domain.save_dir), suffix="exec", session_id=EXECUTOR_SESSION
        )

        committed = adjudicate_possibly_executed(
            episode, episode_id=EPISODE_ID, session_id=SESSION_ID, owner_domain=domain
        )

        assert committed.sidecar.state == STATE_ACCEPTED
        assert committed.sidecar.executor_session_id == EXECUTOR_SESSION
        assert committed.sidecar.accepted_sha256 == sha256_bytes(DOCUMENT)
        assert committed.rename_outcome == SNAPSHOT_RENAMED
        assert _provenance(episode, 1).read_bytes() == DOCUMENT
        assert not _snapshot(episode).exists()

    def test_mark_accepted_forecloses_and_names_the_post_completion_remedy(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        _session_with_marker(
            Path(domain.save_dir), suffix="exec", session_id=EXECUTOR_SESSION
        )
        adjudicate_possibly_executed(
            episode, episode_id=EPISODE_ID, session_id=SESSION_ID, owner_domain=domain
        )

        # The episode is foreclosed: never re-accepted, never re-armed at P3.
        with pytest.raises(PlanEpisodeTerminalError):
            claim_accept(episode, session_id=SESSION_ID, owner_domain=domain)
        assert (
            surviving_snapshot_name(episode, _sidecar(episode))
            == _provenance(episode, 1).name
        )

        # The remedy: a FRESH episode re-binds the surviving bytes, whose
        # recorded name is their post-completion actual state.
        fresh = _make_episode(tmp_path, "plan-fresh")
        rebound = rebind_draft(
            fresh,
            draft_path=_provenance(episode, 1),
            authorized_roots=[tmp_path],
            session_id=SESSION_ID,
        )
        assert rebound.draft_path == _provenance(episode, 1).resolve()

    def test_mark_accepted_is_refused_without_a_discovered_executor(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)

        with pytest.raises(PlanEpisodeStateError):
            adjudicate_possibly_executed(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
            )

        assert _sidecar(episode).state == STATE_ACCEPTING

    def test_an_ambiguous_double_executor_fails_closed(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        _session_with_marker(Path(domain.save_dir), suffix="one", session_id="exec-one")
        _session_with_marker(Path(domain.save_dir), suffix="two", session_id="exec-two")

        with pytest.raises(PlanEpisodeScanError):
            _observe(episode, domain=domain)
        with pytest.raises(PlanEpisodeScanError):
            adjudicate_possibly_executed(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
            )

        assert _sidecar(episode).state == STATE_ACCEPTING


# --- The executor-scan discriminator (F7 residual) --------------------------


class TestExecutorScanDiscriminator:
    """A recorded executor id is the arbiter; its absence fails closed.

    A planning session that restores an ``accepting`` episode writes a
    marker-shaped mirror (``record_from_sidecar`` copies the sidecar's state,
    consent digest, and snapshot name), so marker shape alone cannot identify
    the launch. Once the sidecar records its executor at C4 the scan narrows on
    that session id. C4 runs after C3's marker write, so a recorded id with NO
    matching meta means execution may have begun — that fails closed (design
    §4.4) instead of re-offering the handoff. Only the reclaimed-sidecar record
    path still relies on marker shape.
    """

    def _crashed_after_c4(self, episode: Path, domain: LoggingDomain) -> None:
        """The real C4 residue: accepting, C1 verified, the executor recorded."""
        attempt = _begin(episode, domain=domain)
        try:
            attempt.record_executor_session(EXECUTOR_SESSION)
        finally:
            attempt.close()
        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _sidecar(episode).executor_session_id == EXECUTOR_SESSION

    def test_a_recorded_executor_with_no_matching_meta_fails_closed(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        self._crashed_after_c4(episode, domain)
        # Only the planner's restored mirror is on disk: marker-shaped, its own
        # session id. The C4-recorded executor has NO meta, and C4 runs after
        # C3's marker write — execution may have begun, so this is never a
        # clean miss that would let the handoff be re-offered.
        _session_with_marker(
            Path(domain.save_dir), suffix="planner", session_id=SESSION_ID
        )

        with pytest.raises(PlanEpisodeScanError):
            _observe(episode, domain=domain, session_id=OTHER_SESSION)
        with pytest.raises(PlanEpisodeScanError):
            adjudicate_possibly_executed(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
            )

        # The failure is a refusal, not a classification: nothing moved.
        assert _sidecar(episode).state == STATE_ACCEPTING

    @pytest.mark.asyncio
    async def test_the_re_offer_refuses_when_the_recorded_executor_is_missing(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        self._crashed_after_c4(episode, domain)
        sink = _RecordSink()

        with pytest.raises(PlanEpisodeScanError):
            await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_record(episode, state=STATE_ACCEPTING),
                write_record=sink,
            )

        # No re-offer was built and the restored record was never written.
        assert sink.records == []
        assert _sidecar(episode).state == STATE_ACCEPTING

    def test_the_recorded_executor_is_still_found_beside_the_mirror(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        self._crashed_after_c4(episode, domain)
        save_dir = Path(domain.save_dir)
        _session_with_marker(save_dir, suffix="planner", session_id=SESSION_ID)
        _session_with_marker(save_dir, suffix="exec", session_id=EXECUTOR_SESSION)

        observation = _observe(episode, domain=domain, session_id=OTHER_SESSION)

        assert observation.row == RECOVERY_ROW_POSSIBLY_EXECUTED
        assert observation.executor_session_id == EXECUTOR_SESSION

    def test_the_reclaimed_sidecar_fallback_still_matches_the_marker(
        self, episode: Path, tmp_path: Path
    ) -> None:
        # No C4 id was ever recorded, so the record path keeps the marker-shape
        # predicate — the documented, narrower fallback.
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        _session_with_marker(
            Path(domain.save_dir), suffix="exec", session_id=EXECUTOR_SESSION
        )
        sidecar_path(episode).unlink()
        record = _source_record(episode, state=STATE_ACCEPTING).model_copy(
            update={
                "scan_domain": PlanScanDomain(
                    save_dir=domain.save_dir, session_prefix="session"
                )
            }
        )

        observation = observe_episode(
            episode, record=record, session_id="observer-session"
        )

        assert observation.row == RECOVERY_ROW_EXECUTED
        assert observation.executor_session_id == EXECUTOR_SESSION


# --- The accepted-with-pending-rename intermediate --------------------------


class TestAcceptedPendingRename:
    """The crash-between-the-flip-and-the-rename intermediate."""

    def test_a_crash_between_the_flip_and_the_rename_leaves_the_intermediate(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        _crashed_after_flip(episode, domain, monkeypatch)

        sidecar = _sidecar(episode)
        assert sidecar.state == STATE_ACCEPTED
        assert sidecar.executor_session_id == EXECUTOR_SESSION
        assert sidecar.accepted_sha256 == sha256_bytes(DOCUMENT)
        assert sidecar.c1_verified is True
        # The flip is authoritative; the rename is a derived follow-on.
        assert _snapshot(episode).read_bytes() == DOCUMENT
        assert not _provenance(episode, 1).exists()

    def test_the_next_observation_completes_the_pending_rename(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        _crashed_after_flip(episode, domain, monkeypatch)

        observation = _observe(episode)

        assert observation.row == RECOVERY_ROW_ACCEPTED
        assert observation.rename_outcome == SNAPSHOT_RENAMED
        assert observation.snapshot_name == _provenance(episode, 1).name
        assert _provenance(episode, 1).read_bytes() == DOCUMENT
        assert not _snapshot(episode).exists()

    def test_a_pending_rename_never_re_offers_the_handoff(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        _crashed_after_flip(episode, domain, monkeypatch)

        observation = _observe(episode)
        assert observation.terminal
        assert not observation.recoverable

        with pytest.raises(PlanEpisodeTerminalError):
            claim_accept(episode, session_id=SESSION_ID, owner_domain=domain)
        with pytest.raises(PlanEpisodeStateError):
            adjudicate_possibly_executed(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
            )

    def test_the_completion_is_idempotent_across_observations(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        _crashed_after_flip(episode, domain, monkeypatch)

        first = _observe(episode)
        second = _observe(episode)

        assert first.rename_outcome == SNAPSHOT_RENAMED
        assert second.rename_outcome == SNAPSHOT_RENAME_NOOP
        assert second.snapshot_name == first.snapshot_name
        assert _provenance(episode, 1).read_bytes() == DOCUMENT

    def test_the_adjudication_flip_follows_the_same_completion_rule(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        _session_with_marker(
            Path(domain.save_dir), suffix="exec", session_id=EXECUTOR_SESSION
        )
        with monkeypatch.context() as patch:
            patch.setattr(plan_episode, "rename_accepted_snapshot", _pending_rename)
            committed = adjudicate_possibly_executed(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
            )

        # The flip stands even though its very own completion could not run.
        assert committed.sidecar.state == STATE_ACCEPTED
        assert committed.rename_outcome == SNAPSHOT_RENAME_PENDING
        assert _snapshot(episode).read_bytes() == DOCUMENT

        completed = _observe(episode)
        assert completed.rename_outcome == SNAPSHOT_RENAMED
        assert _provenance(episode, 1).read_bytes() == DOCUMENT

    def test_the_pending_intermediate_never_re_arms_p3(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        _crashed_after_flip(episode, domain, monkeypatch)

        with pytest.raises(PlanEpisodeTerminalError):
            publish_consent_snapshot(
                episode,
                session_id=SESSION_ID,
                owner_domain=domain,
                content=DOCUMENT,
                consent_sha256=sha256_bytes(DOCUMENT),
            )

        # The occupied live path is collision-harmless while foreclosed.
        assert _snapshot(episode).read_bytes() == DOCUMENT


# --- The C2/C3/C4 failure rows (handler-level, end-to-end) ------------------


class TestC2C3Failures:
    """Failures injected at the handler's durable-write and replacement seams."""

    @pytest.mark.asyncio
    async def test_a_c2_write_failure_preserves_the_planning_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
    ) -> None:
        handler, session, loop = await _handler_server(tmp_path, episode)
        save_dir = Path(str(loop.session_logger.save_dir))

        async def _refuse(_record: PlanModeRecord) -> None:
            raise PlanEpisodeWriteError("injected C2 retirement failure")

        monkeypatch.setattr(loop.session_logger, "persist_plan_mode_durable", _refuse)
        _ignore_selection_mismatch(monkeypatch, handler)
        try:
            with pytest.raises(RequestFailure) as caught:
                await handler._plan_handoff(_handoff_params(episode))
        finally:
            await session.close()

        assert caught.value.code is ProtocolErrorCode.CONFLICT
        assert caught.value.data is None
        assert "could not retire the planning record" in str(caught.value)
        # No destructive step ran: the root id is unchanged and the record is
        # still the draft, never a reported success.
        assert loop.session_id == SESSION_ID
        source = _session_meta(save_dir, SESSION_ID)
        assert source is not None
        assert source["plan_mode"]["state"] == STATE_DRAFT
        assert _executor_marker(save_dir) is None
        assert not _seeded_messages(loop)
        # C1 already ran, so the attempt is committed: the recovery rows govern.
        sidecar = _sidecar(episode)
        assert sidecar.state == STATE_ACCEPTING
        assert sidecar.c1_verified is True
        assert sidecar.executor_session_id is None
        assert _snapshot(episode).read_bytes() == DOCUMENT
        assert not _provenance(episode, 1).exists()

    @pytest.mark.asyncio
    async def test_the_c2_failure_state_re_offers_and_completes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
    ) -> None:
        handler, session, loop = await _handler_server(tmp_path, episode)
        logger = loop.session_logger
        real_write = logger.persist_plan_mode_durable

        async def _refuse(_record: PlanModeRecord) -> None:
            raise PlanEpisodeWriteError("injected C2 retirement failure")

        monkeypatch.setattr(logger, "persist_plan_mode_durable", _refuse)
        _ignore_selection_mismatch(monkeypatch, handler)
        try:
            with pytest.raises(RequestFailure):
                await handler._plan_handoff(_handoff_params(episode))
            # The sidecar is the authority: the episode re-offers from row 3.
            domain = _logging_domain(loop)
            sink = _RecordSink()
            offer = await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=sink,
            )
            assert offer.content == DOCUMENT
            assert sink.records[0].state == STATE_ACCEPTING
            settled = await recover_cancel_accepting(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=sink.records[0],
                write_record=_RecordSink(),
            )
            assert settled.state == STATE_DRAFT
            # C1 had run, so the attempt's bytes are permanent provenance.
            assert _provenance(episode, 1).read_bytes() == DOCUMENT
            # The whole transaction then runs again from the draft.
            monkeypatch.setattr(logger, "persist_plan_mode_durable", real_write)
            await handler._plan_handoff(_handoff_params(episode))
        finally:
            await session.close()

        sidecar = _sidecar(episode)
        assert sidecar.state == STATE_ACCEPTED
        assert sidecar.executor_session_id == loop.session_id
        assert _provenance(episode, 1).read_bytes() == DOCUMENT
        assert _provenance(episode, 2).read_bytes() == DOCUMENT
        assert not _snapshot(episode).exists()

    @pytest.mark.asyncio
    async def test_a_c2_write_that_published_before_failing_is_reconciled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
    ) -> None:
        handler, session, loop = await _handler_server(tmp_path, episode)
        save_dir = Path(str(loop.session_logger.save_dir))
        logger = loop.session_logger
        real_write = logger.persist_plan_mode_durable

        async def _write_then_fail(record: PlanModeRecord) -> None:
            await real_write(record)
            raise PlanEpisodeWriteError(
                "injected: the retirement failed after publication"
            )

        monkeypatch.setattr(logger, "persist_plan_mode_durable", _write_then_fail)
        _ignore_selection_mismatch(monkeypatch, handler)
        try:
            with pytest.raises(RequestFailure) as caught:
                await handler._plan_handoff(_handoff_params(episode))
            assert "could not retire the planning record" in str(caught.value)
        finally:
            await session.close()

        # The failure is honest, and the repair is the sidecar's authority: the
        # record says accepted while the sidecar still says accepting.
        source = _session_meta(save_dir, SESSION_ID)
        assert source is not None
        assert source["plan_mode"]["state"] == STATE_ACCEPTED
        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _sidecar(episode).c1_verified is True

        sink = _RecordSink()
        offer = await reopen_accepting_episode(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=_logging_domain(loop),
            record=_source_record(episode, state=STATE_ACCEPTED),
            write_record=sink,
        )
        assert offer.content == DOCUMENT
        assert sink.records[0].state == STATE_ACCEPTING

    @pytest.mark.asyncio
    async def test_a_c3_marker_failure_recovers_through_the_no_executor_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
    ) -> None:
        handler, session, loop = await _handler_server(tmp_path, episode)
        save_dir = Path(str(loop.session_logger.save_dir))
        logger = loop.session_logger
        real_write = logger.persist_plan_mode_durable
        writes: list[int] = []

        async def _fail_the_marker(record: PlanModeRecord) -> None:
            writes.append(len(writes) + 1)
            # C2 is the first write; C3's marker is the second.
            if len(writes) == 2:
                raise PlanEpisodeWriteError("injected C3 marker failure")
            await real_write(record)

        monkeypatch.setattr(logger, "persist_plan_mode_durable", _fail_the_marker)
        _ignore_selection_mismatch(monkeypatch, handler)
        try:
            with pytest.raises(RequestFailure) as caught:
                await handler._plan_handoff(_handoff_params(episode))
            assert caught.value.code is ProtocolErrorCode.INTERNAL_ERROR
            assert "mark the executor session durably" in str(caught.value)
        finally:
            await session.close()

        # Row-3 semantics: no executor meta, no injection, no reported success.
        assert _executor_marker(save_dir) is None
        assert not _seeded_messages(loop)
        sidecar = _sidecar(episode)
        assert sidecar.state == STATE_ACCEPTING
        assert sidecar.c1_verified is True
        assert sidecar.executor_session_id is None
        observation = _observe(episode)
        assert observation.row == RECOVERY_ROW_ACCEPTING_SNAPSHOT
        assert not observation.possibly_executed

        settled = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=_logging_domain(loop),
            record=_source_record(episode, state=STATE_ACCEPTED),
            write_record=_RecordSink(),
        )
        assert settled.state == STATE_DRAFT
        assert _provenance(episode, 1).read_bytes() == DOCUMENT

    @pytest.mark.asyncio
    async def test_a_c3_injection_failure_lands_in_the_executor_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
    ) -> None:
        handler, session, loop = await _handler_server(tmp_path, episode)
        save_dir = Path(str(loop.session_logger.save_dir))

        async def _refuse_injection(_content: str, **_kwargs: Any) -> None:
            raise RuntimeError("injected injection failure")

        monkeypatch.setattr(loop, "inject_user_context", _refuse_injection)
        _ignore_selection_mismatch(monkeypatch, handler)
        try:
            with pytest.raises(RequestFailure) as caught:
                await handler._plan_handoff(_handoff_params(episode))
            assert caught.value.code is ProtocolErrorCode.INTERNAL_ERROR
            assert "possibly executed" in str(caught.value)
        finally:
            await session.close()

        rotated = loop.session_id
        assert rotated != SESSION_ID
        assert _executor_marker(save_dir) is not None
        sidecar = _sidecar(episode)
        assert sidecar.state == STATE_ACCEPTING
        assert sidecar.c1_verified is True
        assert sidecar.executor_session_id is None

        observation = _observe(episode)
        assert observation.row == RECOVERY_ROW_POSSIBLY_EXECUTED
        assert observation.executor_session_id == rotated

        committed = adjudicate_possibly_executed(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=_logging_domain(loop),
        )
        assert committed.sidecar.state == STATE_ACCEPTED
        assert committed.sidecar.executor_session_id == rotated

    @pytest.mark.asyncio
    async def test_a_c4_replacement_failure_is_never_a_preserved_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
    ) -> None:
        handler, session, loop = await _handler_server(tmp_path, episode)
        save_dir = Path(str(loop.session_logger.save_dir))

        async def _fail_the_reset() -> None:
            raise RuntimeError("injected C4 turn reset failure")

        monkeypatch.setattr(handler._turns, "reset", _fail_the_reset)
        _ignore_selection_mismatch(monkeypatch, handler)
        try:
            with pytest.raises(RuntimeError):
                await handler._plan_handoff(_handoff_params(episode))
        finally:
            await session.close()

        # The id has rotated: this is never reported as a preserved session.
        assert loop.session_id != SESSION_ID
        assert _executor_marker(save_dir) is not None
        assert _seeded_messages(loop)
        observation = _observe(episode)
        assert observation.row == RECOVERY_ROW_POSSIBLY_EXECUTED
        assert observation.executor_session_id == loop.session_id

    @pytest.mark.asyncio
    async def test_the_transaction_never_contacts_a_provider(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
    ) -> None:
        # Fail closed at the transport level: any completion attempt would raise.
        backend = FakeBackend(exception_to_raise=httpx.ConnectError("no live provider"))
        handler, session, loop = await _handler_server(
            tmp_path, episode, backend=backend
        )
        _ignore_selection_mismatch(monkeypatch, handler)
        try:
            await handler._plan_handoff(_handoff_params(episode))
        finally:
            await session.close()

        assert backend.requests_messages == []
        assert _sidecar(episode).state == STATE_ACCEPTED


# --- Competing owners and foreign sessions ----------------------------------


class TestCompetingOwnersForeignSessions:
    """Every recovery mutation is owner-only, fenced, and freshly re-verified."""

    @pytest.mark.asyncio
    async def test_every_recovery_leg_refuses_a_foreign_session(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)

        with pytest.raises(PlanEpisodeOwnerError):
            recover_release_claim(
                episode,
                episode_id=EPISODE_ID,
                session_id=OTHER_SESSION,
                owner_domain=domain,
            )
        with pytest.raises(PlanEpisodeOwnerError):
            await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=OTHER_SESSION,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=_RecordSink(),
            )
        with pytest.raises(PlanEpisodeOwnerError):
            await recover_cancel_accepting(
                episode,
                episode_id=EPISODE_ID,
                session_id=OTHER_SESSION,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=_RecordSink(),
            )
        with pytest.raises(PlanEpisodeOwnerError):
            adjudicate_possibly_executed(
                episode,
                episode_id=EPISODE_ID,
                session_id=OTHER_SESSION,
                owner_domain=domain,
            )

        assert _observe(episode, domain=domain, session_id=OTHER_SESSION).owner is False
        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _snapshot(episode).is_file()

        released = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(episode),
            write_record=_RecordSink(),
        )
        assert released.state == STATE_DRAFT

    @pytest.mark.asyncio
    async def test_a_foreign_logging_domain_is_refused_with_the_domain_refusal(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        elsewhere = LoggingDomain(
            save_dir=str(tmp_path / "elsewhere"), session_prefix="session"
        )
        _prepare(episode, domain=domain)

        with pytest.raises(PlanEpisodeOwnerError) as caught:
            recover_release_claim(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=elsewhere,
            )
        assert DOMAIN_REFUSAL in str(caught.value)
        with pytest.raises(PlanEpisodeOwnerError) as caught:
            await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=elsewhere,
                record=_source_record(episode),
                write_record=_RecordSink(),
            )
        assert DOMAIN_REFUSAL in str(caught.value)
        assert _sidecar(episode).state == STATE_ACCEPTING

    @pytest.mark.asyncio
    async def test_a_second_claim_cannot_displace_the_held_claim(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)

        with pytest.raises(PlanEpisodeStateError):
            claim_accept(episode, session_id=OTHER_SESSION, owner_domain=domain)

        assert _sidecar(episode).claiming_session_id == SESSION_ID
        assert _sidecar(episode).state == STATE_ACCEPTING
        released = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(episode),
            write_record=_RecordSink(),
        )
        assert released.state == STATE_DRAFT

    @pytest.mark.asyncio
    async def test_a_held_commit_fence_blocks_every_recovery_mutation(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        attempt = _begin(episode, domain=domain)
        try:
            with pytest.raises(PlanFenceBusyError):
                recover_release_claim(
                    episode,
                    episode_id=EPISODE_ID,
                    session_id=SESSION_ID,
                    owner_domain=domain,
                )
            with pytest.raises(PlanFenceBusyError):
                await reopen_accepting_episode(
                    episode,
                    episode_id=EPISODE_ID,
                    session_id=SESSION_ID,
                    owner_domain=domain,
                    record=_source_record(episode),
                    write_record=_RecordSink(),
                )
            with pytest.raises(PlanFenceBusyError):
                await recover_cancel_accepting(
                    episode,
                    episode_id=EPISODE_ID,
                    session_id=SESSION_ID,
                    owner_domain=domain,
                    record=_source_record(episode),
                    write_record=_RecordSink(),
                )
        finally:
            attempt.close()

        released = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(episode),
            write_record=_RecordSink(),
        )
        assert released.state == STATE_DRAFT
        assert _provenance(episode, 1).read_bytes() == DOCUMENT

    @pytest.mark.asyncio
    async def test_a_recovery_mutation_reverifies_the_claim_after_a_competing_change(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        assert _observe(episode, domain=domain).row == RECOVERY_ROW_ACCEPTING_SNAPSHOT

        # A competing observed exit returns the episode to draft first.
        release_claim(episode, session_id=SESSION_ID, owner_domain=domain)

        with pytest.raises(PlanEpisodeStateError):
            await recover_cancel_accepting(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=_RecordSink(),
            )
        with pytest.raises(PlanEpisodeStateError):
            await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=_RecordSink(),
            )
        assert _sidecar(episode).state == STATE_DRAFT

    @pytest.mark.asyncio
    async def test_an_executor_marker_for_another_episode_is_never_matched(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        _session_with_marker(
            Path(domain.save_dir),
            suffix="other",
            session_id=EXECUTOR_SESSION,
            episode_id="ep-someone-else",
        )

        assert _observe(episode, domain=domain).row == RECOVERY_ROW_ACCEPTING_SNAPSHOT
        released = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(episode),
            write_record=_RecordSink(),
        )
        assert released.state == STATE_DRAFT

    @pytest.mark.asyncio
    async def test_a_predecessor_planning_mirror_is_never_a_discovered_executor(
        self, episode: Path, tmp_path: Path
    ) -> None:
        # F7: the departed planner's own record carries the same episode id
        # under its own lifecycle state. It is a clean scan miss, so both the
        # observation and the cancel leg proceed as if no executor existed.
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        _session_with_planning_mirror(
            Path(domain.save_dir), suffix="planner", session_id=SESSION_ID
        )
        _session_with_planning_mirror(
            Path(domain.save_dir),
            suffix="planner-accepted",
            session_id="planner-accepted",
            state=STATE_ACCEPTED,
        )

        assert _observe(episode, domain=domain).row == RECOVERY_ROW_ACCEPTING_SNAPSHOT
        released = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(episode),
            write_record=_RecordSink(),
        )
        assert released.state == STATE_DRAFT

    def test_a_reclaimed_sidecar_with_a_planner_mirror_surfaces_unresolvable(
        self, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        directory = tmp_path / "plan-reclaimed"
        directory.mkdir()
        record = _source_record(directory).model_copy(
            update={
                "scan_domain": PlanScanDomain(
                    save_dir=domain.save_dir, session_prefix="session"
                )
            }
        )
        _session_with_planning_mirror(
            Path(domain.save_dir), suffix="planner", session_id=SESSION_ID
        )
        _session_with_planning_mirror(
            Path(domain.save_dir),
            suffix="planner-accepted",
            session_id="planner-accepted",
            state=STATE_ACCEPTED,
        )

        observation = observe_episode(
            directory, record=record, session_id="observer-session"
        )

        assert observation.row == RECOVERY_ROW_UNRESOLVABLE
        assert observation.possibly_executed is False
        assert observation.executor_session_id is None


# --- The attempt-scoped c1_verified lifecycle -------------------------------


class TestC1VerifiedLifecycle:
    """C1's durable mark distinguishes pre-commit from post-commit attempts."""

    def test_a_crash_after_c1_lands_in_the_snapshot_row_with_the_flag_set(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        attempt = _begin(episode, domain=domain)
        attempt.close()

        sidecar = _sidecar(episode)
        assert sidecar.state == STATE_ACCEPTING
        assert sidecar.c1_verified is True
        assert _persisted(episode)["c1_verified"] is True
        observation = _observe(episode, domain=domain)
        assert observation.row == RECOVERY_ROW_ACCEPTING_SNAPSHOT
        assert observation.executor_session_id is None
        assert _snapshot(episode).read_bytes() == DOCUMENT

    @pytest.mark.asyncio
    async def test_the_flag_alone_selects_rename_over_unlink(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        plain = _make_episode(tmp_path, "plan-plain")
        flagged = _make_episode(tmp_path, "plan-flagged")
        _prepare(plain, domain=domain)
        attempt = _begin(flagged, domain=domain)
        attempt.close()

        released_plain = await recover_cancel_accepting(
            plain,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(plain),
            write_record=_RecordSink(),
        )
        released_flagged = await recover_cancel_accepting(
            flagged,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(flagged),
            write_record=_RecordSink(),
        )

        # The PRE-COMMIT attempt's live name is not provenance: it is unlinked.
        assert released_plain.c1_verified is False
        assert not _snapshot(plain).exists()
        assert not _provenance(plain, 1).exists()
        # The POST-C1 attempt's bytes are provenance: renamed, never unlinked.
        assert released_flagged.c1_verified is False
        assert not _snapshot(flagged).exists()
        assert _provenance(flagged, 1).read_bytes() == DOCUMENT

    @pytest.mark.asyncio
    async def test_the_flag_is_durable_before_the_retirement_begins(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
    ) -> None:
        handler, session, loop = await _handler_server(tmp_path, episode)
        logger = loop.session_logger
        real_write = logger.persist_plan_mode_durable
        observed: list[bool] = []

        async def _spy(record: PlanModeRecord) -> None:
            # C2's write is the first: the flag must already be on disk.
            observed.append(_sidecar(episode).c1_verified)
            await real_write(record)

        monkeypatch.setattr(logger, "persist_plan_mode_durable", _spy)
        _ignore_selection_mismatch(monkeypatch, handler)
        try:
            await handler._plan_handoff(_handoff_params(episode))
        finally:
            await session.close()

        assert observed[0] is True

    @pytest.mark.asyncio
    async def test_a_pre_commit_cancel_clears_the_flag_in_one_state_write(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)
        seen: list[dict[str, Any]] = []
        real_write = plan_episode.write_sidecar

        def _spy(directory: Path, sidecar: PlanEpisodeSidecar) -> None:
            real_write(directory, sidecar)
            seen.append(_persisted(directory))

        monkeypatch.setattr(plan_episode, "write_sidecar", _spy)
        await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(episode),
            write_record=_RecordSink(),
        )

        assert len(seen) == 1
        assert seen[0]["state"] == STATE_DRAFT
        assert seen[0]["c1_verified"] is False
        assert seen[0]["claiming_session_id"] is None

    @pytest.mark.asyncio
    async def test_a_post_c1_cancel_keeps_the_bytes_and_clears_the_flag(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        attempt = _begin(episode, domain=domain)
        attempt.close()
        generation = _sidecar(episode).attempt_generation

        released = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(episode),
            write_record=_RecordSink(),
        )

        assert released.state == STATE_DRAFT
        assert released.c1_verified is False
        assert released.consent_sha256 is None
        assert released.attempt_generation == generation
        assert _provenance(episode, generation).read_bytes() == DOCUMENT
        assert not _snapshot(episode).exists()

    @pytest.mark.asyncio
    async def test_a_fresh_claim_clears_the_previous_attempts_flag_and_consent(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        attempt = _begin(episode, domain=domain)
        attempt.close()
        released = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(episode),
            write_record=_RecordSink(),
        )
        assert released.c1_verified is False

        claimed = claim_accept(episode, session_id=SESSION_ID, owner_domain=domain)

        assert claimed.attempt_generation == released.attempt_generation + 1
        assert claimed.c1_verified is False
        assert claimed.consent_sha256 is None
        assert not _snapshot(episode).exists()

    @pytest.mark.asyncio
    async def test_the_cancel_window_survivor_is_located_by_either_name(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        await _cancel_window_crash(episode, domain, monkeypatch)

        # The crash landed between the cancel rename and the draft write.
        sidecar = _sidecar(episode)
        assert sidecar.state == STATE_ACCEPTING
        assert sidecar.c1_verified is True
        assert not _snapshot(episode).exists()
        assert _provenance(episode, sidecar.attempt_generation).read_bytes() == DOCUMENT

        observation = _observe(episode, domain=domain)
        assert observation.row == RECOVERY_ROW_ACCEPTING_SNAPSHOT
        assert (
            observation.snapshot_name
            == _provenance(episode, sidecar.attempt_generation).name
        )
        path, content = locate_episode_snapshot(episode, sidecar)
        assert path == _provenance(episode, sidecar.attempt_generation)
        assert content == DOCUMENT

    @pytest.mark.asyncio
    async def test_the_re_offer_continuation_completes_c5_as_a_verified_no_op(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        await _cancel_window_crash(episode, domain, monkeypatch)
        generation = _sidecar(episode).attempt_generation
        sink = _RecordSink()

        offer = await reopen_accepting_episode(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_record(episode),
            write_record=sink,
        )

        assert offer.snapshot_path == _provenance(episode, generation)
        assert offer.content == DOCUMENT
        assert sink.records[0].snapshot_path == _provenance(episode, generation).name
        # The desired end state already holds: C5's continuation is a no-op.
        assert (
            rename_accepted_snapshot(
                episode, episode_id=EPISODE_ID, generation=generation
            )
            == SNAPSHOT_RENAME_NOOP
        )
        released = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=sink.records[0],
            write_record=_RecordSink(),
        )
        assert released.state == STATE_DRAFT
        assert released.c1_verified is False
        assert _provenance(episode, generation).read_bytes() == DOCUMENT


# --- The completion rule across every observation surface -------------------


class TestCompletionRuleAcrossSurfaces:
    """The flip-agnostic completion rule at recovery, restore, cleanup, status."""

    def test_the_restore_surface_completes_the_pending_rename(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        _crashed_after_flip(episode, domain, monkeypatch)

        restored = restore_episode_record(
            _source_record(episode, state=STATE_ACCEPTING), session_id=SESSION_ID
        )

        assert restored is not None
        assert restored.observation.row == RECOVERY_ROW_ACCEPTED
        assert restored.observation.rename_outcome == SNAPSHOT_RENAMED
        assert restored.record.state == STATE_ACCEPTED
        assert restored.record.snapshot_path == _provenance(episode, 1).name
        assert _provenance(episode, 1).read_bytes() == DOCUMENT

    def test_the_installed_restore_record_names_the_completed_provenance(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        _crashed_after_flip(episode, domain, monkeypatch)

        installed = restored_plan_record(
            _source_record(episode, state=STATE_ACCEPTING), session_id=SESSION_ID
        )

        assert installed.state == STATE_ACCEPTED
        assert installed.snapshot_path == _provenance(episode, 1).name
        assert not _snapshot(episode).exists()

    @pytest.mark.asyncio
    async def test_every_surface_reports_the_episode_as_terminal(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        _crashed_after_flip(episode, domain, monkeypatch)

        observed = _observe(episode)
        assert observed.terminal
        assert not observed.recoverable
        restored = restore_episode_record(
            _source_record(episode), session_id=SESSION_ID
        )
        assert restored is not None
        assert restored.observation.terminal

        with pytest.raises(PlanEpisodeStateError):
            await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_record(episode),
                write_record=_RecordSink(),
            )
        with pytest.raises(PlanEpisodeStateError):
            adjudicate_possibly_executed(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
            )

    def test_cleanup_keeps_both_names_when_the_completion_cannot_run(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        committed = _committed_with_an_occupied_provenance(episode, domain)
        assert committed.rename_outcome == SNAPSHOT_RENAME_PENDING
        (episode / PLAN_MARKER_FILENAME).write_text("marker\n", encoding="utf-8")

        removed = cleanup_episode_workspace(episode, marker_name=PLAN_MARKER_FILENAME)

        # The completion cannot run, and the cleanup deletes nothing of
        # uncertain ownership: both names are permanent provenance.
        assert removed is False
        assert _snapshot(episode).read_bytes() == DOCUMENT
        assert _provenance(episode, 1).read_bytes() == b"survivor\n"
        assert not sidecar_path(episode).exists()
        assert not (episode / PLAN_MARKER_FILENAME).exists()
        assert fence_path(episode).exists()

    def test_the_status_observation_reports_the_live_name_while_pending(
        self, episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        domain = _scan_domain(tmp_path)
        _crashed_after_flip(episode, domain, monkeypatch)

        with monkeypatch.context() as patch:
            patch.setattr(plan_episode, "rename_accepted_snapshot", _pending_rename)
            observation = _observe(episode)

        assert observation.row == RECOVERY_ROW_ACCEPTED
        assert observation.rename_outcome == SNAPSHOT_RENAME_PENDING
        assert observation.snapshot_name == _snapshot(episode).name
        assert observation.terminal
        # The artifact was not repaired: the live name still holds the bytes.
        assert _snapshot(episode).read_bytes() == DOCUMENT
        assert not _provenance(episode, 1).exists()

        completed = _observe(episode)
        assert completed.rename_outcome == SNAPSHOT_RENAMED

    def test_a_cancelled_episode_is_never_completed_or_re_offered(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        plan_episode.write_sidecar(
            episode, replace(_sidecar(episode), state=STATE_CANCELLED)
        )

        cancelled = _observe(episode, domain=domain)
        assert cancelled.row == RECOVERY_ROW_CANCELLED
        assert cancelled.terminal
        with pytest.raises(PlanEpisodeStateError):
            complete_accepted_rename(episode, _sidecar(episode))
        with pytest.raises(PlanEpisodeTerminalError):
            claim_accept(episode, session_id=SESSION_ID, owner_domain=domain)
        assert not _snapshot(episode).exists()

    def test_completion_never_applies_before_the_flip(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, domain=domain)

        with pytest.raises(PlanEpisodeStateError):
            complete_accepted_rename(episode, _sidecar(episode))

        # An accepting episode never renames: its live snapshot stays live.
        assert _snapshot(episode).read_bytes() == DOCUMENT
        assert not _provenance(episode, 1).exists()
