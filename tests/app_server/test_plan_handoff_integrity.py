"""Plan handoff integrity, PREPARE and COMMIT halves (P10 chunk 5a/5b).

P1 claims the draft episode under the transition fence; P2 reads and retains
the document's RAW bytes; P2.5 re-reads and byte-compares them against the
preview; P3 publishes the retained bytes by exclusive create and records their
digest in the sidecar. Every observed failure releases the claim, unlinks the
PRE-COMMIT snapshot, and returns the episode to draft.

COMMIT (C1-C5) then acquires the same fence and holds it through the terminal
flip: C1 dual-verifies the carried bytes against the sidecar's consent digest
and the snapshot file against the same digest and durably marks the attempt,
C2 retires the source record as accepted, C3 records the discovery domain,
writes the executor's durable marker, and injects the carried bytes, C4 binds
the rotated executor session, and C5 flips to accepted before renaming the
snapshot to its derived provenance name.

The core exercises are plain filesystem and ``flock`` interactions. The handler
exercises run the app server over a memory transport with the fake backend: no
provider is contacted.
"""

from __future__ import annotations

import base64
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import pytest

from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import ProtocolErrorCode, SessionPlanHandoffParams
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core import plan_episode
from chartreux.core.config.models import ModelConfig, SessionLoggingConfig
from chartreux.core.plan_episode import (
    AUTHORITY_RECORD,
    AUTHORITY_RECORD_TERMINAL,
    AUTHORITY_SIDECAR,
    CARRIED_DIGEST_MISMATCH_MESSAGE,
    LOGGING_REQUIRED_MESSAGE,
    RECOVERY_ROW_ACCEPTED,
    RECOVERY_ROW_ACCEPTING_SNAPSHOT,
    RECOVERY_ROW_CANCELLED,
    RECOVERY_ROW_DRAFT,
    RECOVERY_ROW_EXECUTED,
    RECOVERY_ROW_LEGACY,
    RECOVERY_ROW_POSSIBLY_EXECUTED,
    RECOVERY_ROW_UNRESOLVABLE,
    REPREVIEW_REQUIRED_MESSAGE,
    SEED_DOCUMENT_BEGIN,
    SEED_DOCUMENT_END,
    SNAPSHOT_DIGEST_MISMATCH_MESSAGE,
    SNAPSHOT_RENAME_NOOP,
    SNAPSHOT_RENAME_PENDING,
    SNAPSHOT_RENAMED,
    STATE_ACCEPTED,
    STATE_ACCEPTING,
    STATE_DRAFT,
    LoggingDomain,
    PlanArbiterUnresolvableError,
    PlanDigestMismatchError,
    PlanDocumentChangedError,
    PlanDraftReadError,
    PlanEpisodeOwnerError,
    PlanEpisodePossiblyExecutedError,
    PlanEpisodeScanError,
    PlanEpisodeSidecar,
    PlanEpisodeStateError,
    PlanEpisodeTerminalError,
    PlanFenceBusyError,
    PlanSnapshotCollisionError,
    abort_accept,
    accepted_snapshot_path,
    adjudicate_possibly_executed,
    begin_commit,
    build_handoff_context,
    claim_accept,
    complete_accepted_rename,
    create_sidecar,
    ensure_episode_fence,
    locate_episode_snapshot,
    observe_episode,
    observe_logging_domain,
    prepare_accept,
    provenance_snapshot_path,
    publish_consent_snapshot,
    publish_episode_snapshot,
    read_sidecar,
    recover_cancel_accepting,
    recover_release_claim,
    release_claim,
    rename_accepted_snapshot,
    reopen_accepting_episode,
    require_no_executor,
    sha256_bytes,
    surviving_snapshot_name,
    write_sidecar,
)
from chartreux.core.plan_mode import PLAN_MARKER_FILENAME
from chartreux.core.session_types import (
    LEGACY_UNVERIFIED_STATE,
    PlanModeRecord,
    PlanScanDomain,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    legacy_backend,
)

SESSION_ID = "01234567-89ab-cdef-0123-deadbeef1234"
OTHER_SESSION = "fedcba98-7654-3210-fedc-ba9876543210"
EPISODE_ID = "ep-deadbeef"
DOCUMENT = b"# Plan\r\n\r\n## Summary\r\nKeep these exact bytes \xe2\x80\x94 raw.\r\n"
CHANGED_DOCUMENT = b"# Plan\r\n\r\n## Summary\r\nEdited after the preview.\r\n"

DOMAIN = LoggingDomain(
    save_dir="/tmp/chartreux-plan-prepare-a", session_prefix="session"
)
OTHER_DOMAIN = LoggingDomain(
    save_dir="/tmp/chartreux-plan-prepare-b", session_prefix="session"
)


# --- Fixtures and helpers ---------------------------------------------------


@pytest.fixture
def episode(tmp_path: Path) -> Path:
    directory = tmp_path / "plan-deadbeef1234"
    directory.mkdir()
    ensure_episode_fence(directory)
    create_sidecar(
        directory,
        PlanEpisodeSidecar(episode_id=EPISODE_ID, draft_path=directory / "plan.md"),
    )
    (directory / "plan.md").write_bytes(DOCUMENT)
    return directory


def _sidecar(directory: Path) -> PlanEpisodeSidecar:
    sidecar = read_sidecar(directory)
    assert sidecar is not None
    return sidecar


def _snapshot(directory: Path) -> Path:
    return accepted_snapshot_path(directory, EPISODE_ID)


def _prepare(
    directory: Path,
    *,
    session_id: str = SESSION_ID,
    owner_domain: LoggingDomain = DOMAIN,
    previewed: bytes = DOCUMENT,
    episode_id: str = EPISODE_ID,
) -> plan_episode.PreparedAccept:
    return prepare_accept(
        directory,
        episode_id=episode_id,
        session_id=session_id,
        owner_domain=owner_domain,
        previewed_bytes=previewed,
        previewed_sha256=sha256_bytes(previewed),
    )


def _assert_back_at_draft(directory: Path) -> None:
    sidecar = _sidecar(directory)
    assert sidecar.state == STATE_DRAFT
    assert sidecar.claiming_session_id is None
    assert sidecar.c1_verified is False
    assert sidecar.consent_sha256 is None
    assert not _snapshot(directory).exists()


class TestPrepareP1ToP3:
    """The PREPARE transaction: claim, retain, re-compare, publish."""

    def test_p1_claims_the_episode_and_records_the_attempt(self, episode: Path) -> None:
        prepared = _prepare(episode)

        sidecar = _sidecar(episode)
        assert sidecar.state == STATE_ACCEPTING
        assert sidecar.claiming_session_id == SESSION_ID
        assert sidecar.owner_domain == DOMAIN
        assert sidecar.scan_domain == DOMAIN
        assert sidecar.attempt_generation == 1
        assert sidecar.c1_verified is False
        assert prepared.sidecar == sidecar

    def test_each_prepare_mutation_releases_the_fence(self, episode: Path) -> None:
        _prepare(episode)

        # A held fence would refuse this immediate non-blocking acquire: P1 and
        # P3 each take the fence for their own duration only.
        with plan_episode.held_transition_fence(
            episode, holder_session_id=OTHER_SESSION
        ):
            assert _sidecar(episode).state == STATE_ACCEPTING

    def test_p1_takes_the_fence_and_never_races_a_holder(self, episode: Path) -> None:
        with plan_episode.held_transition_fence(
            episode, holder_session_id=OTHER_SESSION
        ):
            with pytest.raises(PlanFenceBusyError):
                _prepare(episode)

        assert _sidecar(episode).state == STATE_DRAFT
        assert _sidecar(episode).claiming_session_id is None

    def test_p2_retains_the_raw_bytes_and_never_a_normalized_read(
        self, episode: Path
    ) -> None:
        prepared = _prepare(episode)

        assert prepared.retained_bytes == DOCUMENT
        assert _snapshot(episode).read_bytes() == DOCUMENT
        # The shared document reader normalizes CRLF; a normalized digest would
        # differ, so this pins the byte authority.
        assert _snapshot(episode).read_bytes() != DOCUMENT.replace(b"\r\n", b"\n")

    def test_p3_records_the_consent_digest_in_the_sidecar(self, episode: Path) -> None:
        prepared = _prepare(episode)

        sidecar = _sidecar(episode)
        assert sidecar.consent_sha256 == sha256_bytes(DOCUMENT)
        assert prepared.consent_sha256 == sidecar.consent_sha256
        assert prepared.snapshot_path == _snapshot(episode)
        # Only the snapshot itself is created; no durable-write temporary stays.
        assert sorted(entry.name for entry in episode.iterdir()) == sorted([
            ".chartreux-plan-claim.lock",
            ".chartreux-plan-episode.json",
            f"accepted-{EPISODE_ID}.md",
            "plan.md",
        ])

    def test_p25_a_changed_draft_demands_a_re_preview(self, episode: Path) -> None:
        (episode / "plan.md").write_bytes(CHANGED_DOCUMENT)

        with pytest.raises(PlanDocumentChangedError) as caught:
            _prepare(episode, previewed=DOCUMENT)

        assert str(caught.value) == REPREVIEW_REQUIRED_MESSAGE
        _assert_back_at_draft(episode)

    def test_p25_a_change_during_confirm_is_caught_by_the_re_read(
        self, episode: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = iter([DOCUMENT, CHANGED_DOCUMENT, CHANGED_DOCUMENT])

        def _sequence(_path: Path) -> bytes:
            return next(reads)

        monkeypatch.setattr(plan_episode, "read_draft_bytes", _sequence)

        with pytest.raises(PlanDocumentChangedError, match="re-preview"):
            _prepare(episode)
        _assert_back_at_draft(episode)

    def test_p25_a_missing_draft_refuses_and_releases(self, episode: Path) -> None:
        (episode / "plan.md").unlink()

        with pytest.raises(PlanDraftReadError):
            _prepare(episode)
        _assert_back_at_draft(episode)

    def test_the_live_name_survivor_is_never_overwritten(self, episode: Path) -> None:
        survivor = _snapshot(episode)
        survivor.write_bytes(b"a stale survivor\n")

        with pytest.raises(FileExistsError):
            publish_episode_snapshot(survivor, DOCUMENT)
        assert survivor.read_bytes() == b"a stale survivor\n"

        # P3 refuses through the exclusive create; the release then unlinks the
        # PRE-COMMIT survivor (DR-20), so the retry's create is clean.
        with pytest.raises(PlanSnapshotCollisionError, match="never overwritten"):
            _prepare(episode)
        _assert_back_at_draft(episode)

    def test_p3_refuses_a_post_commit_claim(self, episode: Path) -> None:
        claim_accept(episode, session_id=SESSION_ID, owner_domain=DOMAIN)
        write_sidecar(episode, replace(_sidecar(episode), c1_verified=True))

        with pytest.raises(PlanEpisodeStateError, match="post-commit"):
            publish_consent_snapshot(
                episode,
                session_id=SESSION_ID,
                owner_domain=DOMAIN,
                content=DOCUMENT,
                consent_sha256=sha256_bytes(DOCUMENT),
            )

        assert not _snapshot(episode).exists()

    def test_a_competing_second_accept_is_refused_with_the_status(
        self, episode: Path
    ) -> None:
        _prepare(episode)

        with pytest.raises(PlanEpisodeStateError, match=STATE_ACCEPTING):
            _prepare(episode, session_id=OTHER_SESSION)

        sidecar = _sidecar(episode)
        assert sidecar.claiming_session_id == SESSION_ID
        assert sidecar.consent_sha256 == sha256_bytes(DOCUMENT)
        assert _snapshot(episode).read_bytes() == DOCUMENT

    def test_a_foreign_logging_domain_never_claims(self, episode: Path) -> None:
        _prepare(episode)
        abort_accept(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

        # owner_domain is immutable once recorded: a different logging root
        # refuses even with the episode back at draft.
        with pytest.raises(PlanEpisodeOwnerError, match="logging domain"):
            claim_accept(episode, session_id=SESSION_ID, owner_domain=OTHER_DOMAIN)

        sidecar = _sidecar(episode)
        assert sidecar.state == STATE_DRAFT
        assert sidecar.owner_domain == DOMAIN
        assert sidecar.claiming_session_id is None

    def test_a_mismatched_episode_identity_refuses_and_releases(
        self, episode: Path
    ) -> None:
        with pytest.raises(PlanEpisodeStateError, match="does not match"):
            _prepare(episode, episode_id="ep-someone-else")
        _assert_back_at_draft(episode)

    def test_abort_accept_returns_a_prepared_episode_to_draft(
        self, episode: Path
    ) -> None:
        _prepare(episode)

        aborted = abort_accept(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

        assert aborted.state == STATE_DRAFT
        assert aborted.attempt_generation == 1
        assert not _snapshot(episode).exists()
        assert _sidecar(episode) == aborted

    def test_release_is_guarded_by_the_sidecar_claiming_identity(
        self, episode: Path
    ) -> None:
        _prepare(episode)

        with pytest.raises(PlanEpisodeOwnerError, match="owning session"):
            release_claim(episode, session_id=OTHER_SESSION, owner_domain=DOMAIN)

        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _snapshot(episode).is_file()
        assert (
            release_claim(episode, session_id=SESSION_ID, owner_domain=DOMAIN).state
            == STATE_DRAFT
        )

    def test_a_fresh_attempt_gets_a_new_generation_and_digest(
        self, episode: Path
    ) -> None:
        _prepare(episode)
        abort_accept(episode, session_id=SESSION_ID, owner_domain=DOMAIN)
        (episode / "plan.md").write_bytes(CHANGED_DOCUMENT)

        prepared = _prepare(episode, previewed=CHANGED_DOCUMENT)

        assert prepared.sidecar.attempt_generation == 2
        assert prepared.consent_sha256 == sha256_bytes(CHANGED_DOCUMENT)
        assert _snapshot(episode).read_bytes() == CHANGED_DOCUMENT


# --- The handler's PREPARE wiring -------------------------------------------


def _episode_params(
    directory: Path,
    *,
    content: bytes = DOCUMENT,
    digest: str | None = None,
    snapshot_path: Path | None = None,
) -> SessionPlanHandoffParams:
    return SessionPlanHandoffParams(
        session_id=SESSION_ID,
        executor_model="alpha",
        executor_thinking="medium",
        executor_identity="orchestrator",
        plan_path=str(directory / "plan.md"),
        episode_id=EPISODE_ID,
        snapshot_path=str(snapshot_path or _snapshot(directory)),
        snapshot_sha256=sha256_bytes(content) if digest is None else digest,
        snapshot_bytes_b64=base64.b64encode(content).decode("ascii"),
    )


def _legacy_params(directory: Path) -> SessionPlanHandoffParams:
    return SessionPlanHandoffParams(
        session_id=SESSION_ID,
        executor_model="alpha",
        executor_thinking="medium",
        executor_identity="orchestrator",
        plan_path=str(directory / "plan.md"),
    )


async def _handler(tmp_path: Path, *, logging_enabled: bool) -> tuple[Any, Any]:
    config = build_test_vibe_config(
        active_model="alpha",
        models=[ModelConfig(name="alpha", provider="mistral", alias="alpha")],
        session_logging=SessionLoggingConfig(
            enabled=logging_enabled, save_dir=str(tmp_path / "sessions")
        ),
    )
    loop = build_test_agent_loop(config=config, session_id=SESSION_ID)
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    return legacy_backend(server).handler, session


def test_handoff_params_carry_the_verifiable_body() -> None:
    params = SessionPlanHandoffParams(
        session_id=SESSION_ID,
        executor_model="alpha",
        executor_thinking="medium",
        executor_identity="orchestrator",
        episode_id=EPISODE_ID,
        snapshot_path=f"/tmp/{EPISODE_ID}/accepted-{EPISODE_ID}.md",
        snapshot_sha256="a" * 64,
        snapshot_bytes_b64=base64.b64encode(DOCUMENT).decode("ascii"),
    )

    dumped = params.model_dump()
    assert dumped["snapshotBytesB64"] == base64.b64encode(DOCUMENT).decode("ascii")
    assert dumped["snapshotSha256"] == "a" * 64
    with pytest.raises(ValidationError):
        SessionPlanHandoffParams.model_validate({
            "session_id": SESSION_ID,
            "executor_model": "alpha",
            "executor_thinking": "medium",
            "executor_identity": "orchestrator",
            "not_a_field": True,
        })


@pytest.mark.asyncio
async def test_handler_prepare_decodes_publishes_and_records_the_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session = await _handler(tmp_path, logging_enabled=True)
    try:
        await handler._plan_handoff_prepare(_episode_params(episode))
    finally:
        await session.close()

    sidecar = _sidecar(episode)
    assert sidecar.state == STATE_ACCEPTING
    assert sidecar.claiming_session_id == SESSION_ID
    assert sidecar.consent_sha256 == sha256_bytes(DOCUMENT)
    assert _snapshot(episode).read_bytes() == DOCUMENT


@pytest.mark.asyncio
async def test_handler_prepare_refuses_session_logging_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session = await _handler(tmp_path, logging_enabled=False)
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff_prepare(_episode_params(episode))
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.CONFLICT
    assert str(caught.value) == LOGGING_REQUIRED_MESSAGE
    _assert_back_at_draft(episode)


@pytest.mark.asyncio
async def test_handler_prepare_refuses_a_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session = await _handler(tmp_path, logging_enabled=True)
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff_prepare(
                _episode_params(episode, digest="0" * 64)
            )
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.INVALID_PARAMS
    assert "does not match its bytes" in str(caught.value)
    _assert_back_at_draft(episode)


@pytest.mark.asyncio
async def test_handler_prepare_reports_a_changed_draft_as_a_conflict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session = await _handler(tmp_path, logging_enabled=True)
    (episode / "plan.md").write_bytes(CHANGED_DOCUMENT)
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff_prepare(_episode_params(episode))
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.CONFLICT
    assert str(caught.value) == REPREVIEW_REQUIRED_MESSAGE
    _assert_back_at_draft(episode)


@pytest.mark.asyncio
async def test_handler_prepare_refuses_an_unexpected_snapshot_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session = await _handler(tmp_path, logging_enabled=True)
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff_prepare(
                _episode_params(episode, snapshot_path=episode / "snapshot.md")
            )
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.INVALID_PARAMS
    assert "must be the episode" in str(caught.value)
    _assert_back_at_draft(episode)


@pytest.mark.asyncio
async def test_handler_prepare_refuses_a_partial_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session = await _handler(tmp_path, logging_enabled=True)
    params = _episode_params(episode).model_copy(update={"snapshot_bytes_b64": None})
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff_prepare(params)
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.INVALID_PARAMS
    assert "together" in str(caught.value)
    _assert_back_at_draft(episode)


@pytest.mark.asyncio
async def test_handler_prepare_is_a_noop_on_the_legacy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session = await _handler(tmp_path, logging_enabled=True)
    try:
        await handler._plan_handoff_prepare(_legacy_params(episode))
    finally:
        await session.close()

    _assert_back_at_draft(episode)
    assert not _snapshot(episode).exists()


@pytest.mark.asyncio
async def test_plan_handoff_prepares_before_any_destructive_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session = await _handler(tmp_path, logging_enabled=False)
    calls: list[str] = []
    real_prepare = handler._plan_handoff_prepare
    real_clear = handler._agent_loop.clear_history

    async def _spy_prepare(params: SessionPlanHandoffParams) -> None:
        calls.append("prepare")
        await real_prepare(params)

    async def _spy_clear(*args: Any, **kwargs: Any) -> Any:
        calls.append("clear")
        return await real_clear(*args, **kwargs)

    monkeypatch.setattr(handler, "_plan_handoff_prepare", _spy_prepare)
    monkeypatch.setattr(handler._agent_loop, "clear_history", _spy_clear)
    # The apply-exactly check needs a real config orchestrator; this node only
    # pins the ORDERING of the prepare call against the destructive reset.
    monkeypatch.setattr(
        handler, "_plan_handoff_selection_mismatch", lambda *args, **kwargs: None
    )
    try:
        await handler._plan_handoff(_legacy_params(episode))
    finally:
        await session.close()

    assert calls == ["prepare", "clear"]


# --- The COMMIT half (C1-C5) ------------------------------------------------


def _prepared_commit(
    episode: Path, *, content: bytes = DOCUMENT, retry_timeout: float = 0.0
) -> plan_episode.CommitAttempt:
    """Prepare the episode, then enter COMMIT with the carried body."""
    prepared = _prepare(episode)
    return begin_commit(
        episode,
        episode_id=EPISODE_ID,
        session_id=SESSION_ID,
        owner_domain=DOMAIN,
        carried_bytes=content,
        snapshot_path=prepared.snapshot_path,
        retry_timeout=retry_timeout,
    )


class TestCommitC1ToC5:
    """The COMMIT transaction: verify, retire, mark, inject, flip, rename."""

    def test_c1_marks_the_attempt_durably_and_holds_the_fence(
        self, episode: Path
    ) -> None:
        attempt = _prepared_commit(episode)
        try:
            sidecar = _sidecar(episode)
            assert sidecar.state == STATE_ACCEPTING
            assert sidecar.c1_verified is True

            # The commit holds the fence through C5: an immediate competing
            # acquire must lose, not wait.
            with pytest.raises(PlanFenceBusyError):
                begin_commit(
                    episode,
                    episode_id=EPISODE_ID,
                    session_id=OTHER_SESSION,
                    owner_domain=DOMAIN,
                    carried_bytes=DOCUMENT,
                    snapshot_path=_snapshot(episode),
                )
        finally:
            attempt.close()

        # Closing the attempt releases the fence; the episode is still
        # accepting, and a fresh acquire proves the release.
        with plan_episode.held_transition_fence(
            episode, holder_session_id=OTHER_SESSION
        ):
            assert _sidecar(episode).state == STATE_ACCEPTING

    def test_c1_refuses_carried_bytes_that_are_not_the_consented_bytes(
        self, episode: Path
    ) -> None:
        with pytest.raises(plan_episode.PlanDigestMismatchError) as caught:
            _prepared_commit(episode, content=CHANGED_DOCUMENT)

        assert str(caught.value) == CARRIED_DIGEST_MISMATCH_MESSAGE
        sidecar = _sidecar(episode)
        assert sidecar.state == STATE_ACCEPTING
        assert sidecar.c1_verified is False
        assert _snapshot(episode).read_bytes() == DOCUMENT

    def test_c1_refuses_a_snapshot_that_is_not_the_consented_bytes(
        self, episode: Path
    ) -> None:
        _prepare(episode)
        _snapshot(episode).write_bytes(CHANGED_DOCUMENT)

        with pytest.raises(plan_episode.PlanDigestMismatchError) as caught:
            begin_commit(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=DOMAIN,
                carried_bytes=DOCUMENT,
                snapshot_path=_snapshot(episode),
            )

        assert str(caught.value).startswith(SNAPSHOT_DIGEST_MISMATCH_MESSAGE)
        assert _sidecar(episode).c1_verified is False

    def test_c1_refuses_a_missing_snapshot(self, episode: Path) -> None:
        _prepare(episode)
        _snapshot(episode).unlink()

        with pytest.raises(plan_episode.PlanDigestMismatchError):
            begin_commit(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=DOMAIN,
                carried_bytes=DOCUMENT,
                snapshot_path=_snapshot(episode),
            )

        assert _sidecar(episode).c1_verified is False

    def test_a_competing_commit_busy_retries_then_refuses_after_the_flip(
        self, episode: Path
    ) -> None:
        attempt = _prepared_commit(episode)
        try:
            # A busy fence retries up to its bound and then refuses: it never
            # blocks a running commit out of its sequence.
            with pytest.raises(PlanFenceBusyError):
                _prepared_commit_while_held(episode)
        finally:
            attempt.finalize(executor_session_id=OTHER_SESSION)
            attempt.close()

        assert _sidecar(episode).state == STATE_ACCEPTED
        # Once the commit finished, the next attempt is refused by the
        # expected-state verify, not by the fence.
        with pytest.raises(PlanEpisodeTerminalError, match=STATE_ACCEPTED):
            begin_commit(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=DOMAIN,
                carried_bytes=DOCUMENT,
                snapshot_path=_snapshot(episode),
            )

    def test_c3_records_the_discovery_domain_without_touching_the_owner(
        self, episode: Path
    ) -> None:
        attempt = _prepared_commit(episode)
        try:
            recorded = attempt.record_scan_domain(OTHER_DOMAIN)
        finally:
            attempt.close()

        assert recorded.scan_domain == OTHER_DOMAIN
        assert recorded.owner_domain == DOMAIN
        assert recorded.state == STATE_ACCEPTING

    def test_a_draft_deleted_after_consent_never_fails_the_commit(
        self, episode: Path
    ) -> None:
        attempt = _prepared_commit(episode)
        (episode / "plan.md").unlink()
        try:
            committed = attempt.finalize(executor_session_id=OTHER_SESSION)
        finally:
            attempt.close()

        assert committed.sidecar.state == STATE_ACCEPTED
        assert committed.rename_outcome == SNAPSHOT_RENAMED

    def test_a_draft_mutated_after_consent_never_fails_the_commit(
        self, episode: Path
    ) -> None:
        attempt = _prepared_commit(episode)
        (episode / "plan.md").write_bytes(CHANGED_DOCUMENT)
        try:
            committed = attempt.finalize(executor_session_id=OTHER_SESSION)
        finally:
            attempt.close()

        assert committed.sidecar.state == STATE_ACCEPTED
        assert committed.sidecar.consent_sha256 == sha256_bytes(DOCUMENT)

    def test_c5_flips_to_accepted_before_the_derived_rename(
        self, episode: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempt = _prepared_commit(episode)
        states: list[str] = []
        real_rename = plan_episode.rename_accepted_snapshot

        def _spy(directory: Path, *, episode_id: str, generation: int) -> str:
            states.append(_sidecar(directory).state)
            return real_rename(directory, episode_id=episode_id, generation=generation)

        monkeypatch.setattr(plan_episode, "rename_accepted_snapshot", _spy)
        try:
            committed = attempt.finalize(executor_session_id=OTHER_SESSION)
        finally:
            attempt.close()

        # The rename observed the ALREADY-ACCEPTED sidecar: the flip is the last
        # arbiter-visible change and the rename is strictly after it.
        assert states == [STATE_ACCEPTED]
        assert committed.rename_outcome == SNAPSHOT_RENAMED

    def test_c5_renames_to_the_attempt_provenance_name(self, episode: Path) -> None:
        attempt = _prepared_commit(episode)
        try:
            committed = attempt.finalize(executor_session_id=OTHER_SESSION)
        finally:
            attempt.close()

        provenance = provenance_snapshot_path(episode, EPISODE_ID, 1)
        assert committed.rename_outcome == SNAPSHOT_RENAMED
        assert provenance.read_bytes() == DOCUMENT
        assert not _snapshot(episode).exists()

    def test_the_rename_is_an_idempotent_no_op_when_already_renamed(
        self, episode: Path
    ) -> None:
        attempt = _prepared_commit(episode)
        try:
            attempt.finalize(executor_session_id=OTHER_SESSION)
        finally:
            attempt.close()

        # The cancel-window continuation locates the bytes at the provenance
        # name; the same end state already holds, so this is a verified no-op.
        outcome = rename_accepted_snapshot(episode, episode_id=EPISODE_ID, generation=1)

        assert outcome == SNAPSHOT_RENAME_NOOP
        assert provenance_snapshot_path(episode, EPISODE_ID, 1).is_file()

    def test_accepting_means_the_rename_has_not_run(self, episode: Path) -> None:
        attempt = _prepared_commit(episode)
        try:
            attempt.record_scan_domain(DOMAIN)
            assert _snapshot(episode).read_bytes() == DOCUMENT
            assert not provenance_snapshot_path(episode, EPISODE_ID, 1).exists()
        finally:
            attempt.close()

    def test_a_rename_that_cannot_run_is_honest_tolerated(self, episode: Path) -> None:
        attempt = _prepared_commit(episode)
        _snapshot(episode).unlink()
        try:
            committed = attempt.finalize(executor_session_id=OTHER_SESSION)
        finally:
            attempt.close()

        # The flip is authoritative; the completion rule restores the end state
        # at the episode's next observation. Nothing is reported as failed.
        assert committed.sidecar.state == STATE_ACCEPTED
        assert committed.rename_outcome == SNAPSHOT_RENAME_PENDING

    def test_a_provenance_collision_leaves_the_flip_standing(
        self, episode: Path
    ) -> None:
        provenance_snapshot_path(episode, EPISODE_ID, 1).write_bytes(b"survivor\n")
        attempt = _prepared_commit(episode)
        try:
            committed = attempt.finalize(executor_session_id=OTHER_SESSION)
        finally:
            attempt.close()

        assert committed.sidecar.state == STATE_ACCEPTED
        assert committed.rename_outcome == SNAPSHOT_RENAME_PENDING
        # Neither name is ever deleted: both are permanent provenance.
        assert provenance_snapshot_path(episode, EPISODE_ID, 1).read_bytes() == (
            b"survivor\n"
        )
        assert _snapshot(episode).read_bytes() == DOCUMENT

    def test_the_accepted_flip_records_the_executor_and_the_digest(
        self, episode: Path
    ) -> None:
        attempt = _prepared_commit(episode)
        try:
            committed = attempt.finalize(executor_session_id=OTHER_SESSION)
        finally:
            attempt.close()

        sidecar = _sidecar(episode)
        assert sidecar == committed.sidecar
        assert sidecar.accepted_sha256 == sha256_bytes(DOCUMENT)
        assert sidecar.consent_sha256 == sha256_bytes(DOCUMENT)
        assert sidecar.executor_session_id == OTHER_SESSION
        assert sidecar.c1_verified is True

    def test_an_untrusted_scan_fails_closed(self, tmp_path: Path) -> None:
        missing = LoggingDomain(
            save_dir=str(tmp_path / "absent"), session_prefix="session"
        )

        with pytest.raises(plan_episode.PlanEpisodeScanError):
            require_no_executor(missing, EPISODE_ID)

    def test_the_pre_launch_rescan_refuses_a_discovered_executor(
        self, tmp_path: Path
    ) -> None:
        save_dir = tmp_path / "sessions"
        executor_dir = save_dir / "session_20260101_120000_abcd"
        executor_dir.mkdir(parents=True)
        (executor_dir / "meta.json").write_text(
            json.dumps({
                "session_id": "executor-session",
                "plan_mode": {
                    "episode_id": EPISODE_ID,
                    "state": STATE_ACCEPTING,
                    "consent_sha256": "c" * 64,
                    "snapshot_path": f"accepted-{EPISODE_ID}.md",
                },
            }),
            encoding="utf-8",
        )
        domain = LoggingDomain(save_dir=str(save_dir), session_prefix="session")

        with pytest.raises(PlanEpisodeStateError, match="already records"):
            require_no_executor(domain, EPISODE_ID, excluding_session_id=SESSION_ID)

        # The source session is the caller, never a launched executor.
        require_no_executor(domain, EPISODE_ID, excluding_session_id="executor-session")

    def test_the_observation_binds_the_domain_to_the_session_directory(
        self, tmp_path: Path
    ) -> None:
        save_dir = tmp_path / "sessions"
        observation = observe_logging_domain(
            save_dir=save_dir,
            session_prefix="session",
            session_dir=save_dir / "session_20260101_000000_abcdef",
        )

        assert observation.domain == LoggingDomain(
            save_dir=str(save_dir), session_prefix="session"
        )
        assert observation.session_dir.parent == save_dir

    def test_an_observation_outside_the_domain_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(plan_episode.PlanEpisodeWriteError, match="derive"):
            observe_logging_domain(
                save_dir=tmp_path / "sessions",
                session_prefix="session",
                session_dir=tmp_path / "elsewhere" / "session_20260101_000000_abc",
            )


def _prepared_commit_while_held(episode: Path) -> plan_episode.CommitAttempt:
    """A short bounded retry, so a held fence surfaces without a long wait."""
    return begin_commit(
        episode,
        episode_id=EPISODE_ID,
        session_id=SESSION_ID,
        owner_domain=DOMAIN,
        carried_bytes=DOCUMENT,
        snapshot_path=_snapshot(episode),
        retry_timeout=0.1,
    )


# --- The seed prose ---------------------------------------------------------


class TestExecutorSeedProvenance:
    """The injected context: the consented bytes plus documentation-only prose."""

    def _seed(self) -> str:
        return build_handoff_context(
            content=DOCUMENT,
            episode_id=EPISODE_ID,
            snapshot_path=Path(
                f"/workspaces/plan-{EPISODE_ID}/accepted-{EPISODE_ID}.md"
            ),
            consent_sha256=sha256_bytes(DOCUMENT),
        )

    def test_the_seed_records_snapshot_provenance_and_digest(self) -> None:
        text = self._seed()

        assert f"accepted-{EPISODE_ID}.md" in text
        assert sha256_bytes(DOCUMENT) in text
        assert "documentation only" in text

    def test_the_injected_document_region_is_exactly_the_snapshot_bytes(self) -> None:
        text = self._seed()

        region = text.split(f"{SEED_DOCUMENT_BEGIN}\n")[1].split(
            f"\n{SEED_DOCUMENT_END}"
        )[0]
        # Raw bytes, not a reader's newline-normalized view.
        assert region == DOCUMENT.decode()
        assert "\r\n" in region

    def test_the_seed_is_documentation_not_a_tool_invocation(self) -> None:
        text = self._seed()

        assert text.startswith("Execute the accepted plan for episode ")
        assert "begin_commit" not in text
        assert "tool" not in text.lower()

    def test_the_seed_does_not_summarize_the_document(self) -> None:
        text = self._seed()

        # The whole document travels; the summary line is not its stand-in.
        assert DOCUMENT.decode().strip() in text
        assert "Planner summary" not in text


# --- The handler's COMMIT wiring --------------------------------------------


def _plan_record(plan_dir: Path, save_dir: Path) -> PlanModeRecord:
    """The planning session's record for the episode under test."""
    return PlanModeRecord(
        plan_path=str(plan_dir / "plan.md"),
        theme="Keep these exact bytes",
        planner_model="alpha",
        planner_thinking="medium",
        episode_id=EPISODE_ID,
        sidecar_path=str(plan_dir / plan_episode.EPISODE_SIDECAR_FILENAME),
        draft_path=str(plan_dir / "plan.md"),
        state=STATE_DRAFT,
        scan_domain=PlanScanDomain(save_dir=str(save_dir), session_prefix="session"),
    )


async def _episode_handler(tmp_path: Path, plan_dir: Path) -> tuple[Any, Any, Any]:
    """A logging-enabled app server whose root session carries the plan record."""
    save_dir = tmp_path / "sessions"
    config = build_test_vibe_config(
        active_model="alpha",
        models=[ModelConfig(name="alpha", provider="mistral", alias="alpha")],
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(save_dir)),
    )
    loop = build_test_agent_loop(config=config, session_id=SESSION_ID)
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    await loop.session_logger.persist_plan_mode_durable(
        _plan_record(plan_dir, save_dir)
    )
    return legacy_backend(server).handler, session, loop


def _sessions_dir(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


def _session_metas(save_dir: Path) -> list[dict[str, Any]]:
    """Every session meta under the save dir, parsed."""
    metas: list[dict[str, Any]] = []
    for meta_path in sorted(save_dir.glob("session_*/meta.json")):
        metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
    return metas


def _executor_marker(save_dir: Path, episode_id: str) -> dict[str, Any] | None:
    """The executing session's meta payload, if one records the episode."""
    for payload in _session_metas(save_dir):
        record = payload.get("plan_mode")
        if (
            isinstance(record, dict)
            and record.get("episode_id") == episode_id
            and record.get("state") == STATE_ACCEPTING
        ):
            return payload
    return None


def _source_record(save_dir: Path, episode_id: str) -> dict[str, Any] | None:
    """The planning session's meta payload for the episode."""
    for payload in _session_metas(save_dir):
        record = payload.get("plan_mode")
        if isinstance(record, dict) and record.get("episode_id") == episode_id:
            if payload.get("session_id") != SESSION_ID:
                continue
            return payload
    return None


def _seeded_messages(loop: Any) -> list[str]:
    return [
        message.content or ""
        for message in loop.messages
        if SEED_DOCUMENT_BEGIN in (message.content or "")
    ]


@pytest.mark.asyncio
async def test_handler_episode_handoff_marks_the_executor_and_renames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    save_dir = _sessions_dir(tmp_path)
    observed: list[dict[str, Any] | None] = []
    real_inject = loop.inject_user_context

    async def _spy(content: str, **kwargs: Any) -> Any:
        # Read the durable marker AT injection time: the file proves the marker
        # write already ran, not merely that it is planned.
        observed.append(_executor_marker(save_dir, EPISODE_ID))
        return await real_inject(content, **kwargs)

    monkeypatch.setattr(loop, "inject_user_context", _spy)
    # The apply-exactly readback is its own node; this one exercises the
    # transaction around it.
    monkeypatch.setattr(
        handler, "_plan_handoff_selection_mismatch", lambda *args, **kwargs: None
    )
    try:
        await handler._plan_handoff(_episode_params(episode))
    finally:
        await session.close()

    # C3: the marker was already durable when the bytes were injected.
    assert observed and observed[0] is not None
    marker = observed[0]["plan_mode"]
    assert marker["state"] == STATE_ACCEPTING
    assert marker["episode_id"] == EPISODE_ID
    assert marker["consent_sha256"] == sha256_bytes(DOCUMENT)
    assert marker["snapshot_path"] == f"accepted-{EPISODE_ID}.md"

    # C2: the source record was retired durably as accepted.
    source = _source_record(save_dir, EPISODE_ID)
    assert source is not None
    assert source["plan_mode"]["state"] == STATE_ACCEPTED
    assert source["plan_mode"]["consent_sha256"] == sha256_bytes(DOCUMENT)

    # C4/C5: the executor session is bound and the flip precedes the rename.
    sidecar = _sidecar(episode)
    assert sidecar.state == STATE_ACCEPTED
    assert sidecar.executor_session_id == loop.session_id
    assert sidecar.executor_session_id != SESSION_ID
    assert sidecar.accepted_sha256 == sha256_bytes(DOCUMENT)
    assert provenance_snapshot_path(episode, EPISODE_ID, 1).read_bytes() == DOCUMENT
    assert not _snapshot(episode).exists()

    # The injected initial context carries the whole document, not a summary.
    seeds = _seeded_messages(loop)
    assert len(seeds) == 1
    assert DOCUMENT.decode().strip() in seeds[0]


@pytest.mark.asyncio
async def test_handler_c1_mismatch_refuses_and_releases_the_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    params = _episode_params(episode)
    try:
        await handler._plan_handoff_prepare(params)
        with pytest.raises(RequestFailure) as caught:
            handler._plan_handoff_begin_commit(
                params, (EPISODE_ID, _snapshot(episode), CHANGED_DOCUMENT)
            )
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.CONFLICT
    # No data payload: the refusing layer is the pre-replacement one, so the
    # planning conversation is provably still in place.
    assert caught.value.data is None
    assert CARRIED_DIGEST_MISMATCH_MESSAGE in str(caught.value)
    assert "released" in str(caught.value)
    _assert_back_at_draft(episode)
    assert not _seeded_messages(loop)


@pytest.mark.asyncio
async def test_handler_c2_failure_preserves_the_planning_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    save_dir = _sessions_dir(tmp_path)

    async def _refuse(_record: PlanModeRecord) -> None:
        raise plan_episode.PlanEpisodeWriteError("injected retirement failure")

    monkeypatch.setattr(loop.session_logger, "persist_plan_mode_durable", _refuse)
    monkeypatch.setattr(
        handler, "_plan_handoff_selection_mismatch", lambda *args, **kwargs: None
    )
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff(_episode_params(episode))
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.CONFLICT
    assert caught.value.data is None
    assert "could not retire the planning record" in str(caught.value)
    # The reset never ran: the root session id is unchanged and its record is
    # still the draft, never a reported success.
    assert loop.session_id == SESSION_ID
    source = _source_record(save_dir, EPISODE_ID)
    assert source is not None
    assert source["plan_mode"]["state"] == STATE_DRAFT
    assert _executor_marker(save_dir, EPISODE_ID) is None
    assert not _seeded_messages(loop)
    # C1 already ran, so the attempt is committed: the recovery rows govern it.
    sidecar = _sidecar(episode)
    assert sidecar.state == STATE_ACCEPTING
    assert sidecar.c1_verified is True
    assert _snapshot(episode).read_bytes() == DOCUMENT


@pytest.mark.asyncio
async def test_a_marker_failure_never_injects_and_never_reports_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    save_dir = _sessions_dir(tmp_path)
    logger = loop.session_logger
    real_write = logger.persist_plan_mode_durable
    writes: list[int] = []

    async def _fail_the_marker(record: PlanModeRecord) -> None:
        writes.append(len(writes) + 1)
        # C2 is the first write; C3's marker is the second.
        if len(writes) == 2:
            raise plan_episode.PlanEpisodeWriteError("injected marker failure")
        await real_write(record)

    monkeypatch.setattr(logger, "persist_plan_mode_durable", _fail_the_marker)
    monkeypatch.setattr(
        handler, "_plan_handoff_selection_mismatch", lambda *args, **kwargs: None
    )
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff(_episode_params(episode))
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.INTERNAL_ERROR
    assert "mark the executor session durably" in str(caught.value)
    # Row-3 semantics: no episode-marked executor meta exists, so nothing was
    # injected and the episode is not accepted.
    assert _executor_marker(save_dir, EPISODE_ID) is None
    assert not _seeded_messages(loop)
    sidecar = _sidecar(episode)
    assert sidecar.state == STATE_ACCEPTING
    assert sidecar.c1_verified is True
    assert sidecar.executor_session_id is None


@pytest.mark.asyncio
async def test_handler_injection_follows_the_marker_write_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    order: list[str] = []
    logger = loop.session_logger
    real_write = logger.persist_plan_mode_durable
    real_inject = loop.inject_user_context

    async def _spy_write(record: PlanModeRecord) -> None:
        await real_write(record)
        order.append(f"marker:{record.state}")

    async def _spy_inject(content: str, **kwargs: Any) -> Any:
        order.append("inject")
        return await real_inject(content, **kwargs)

    monkeypatch.setattr(logger, "persist_plan_mode_durable", _spy_write)
    monkeypatch.setattr(loop, "inject_user_context", _spy_inject)
    monkeypatch.setattr(
        handler, "_plan_handoff_selection_mismatch", lambda *args, **kwargs: None
    )
    try:
        await handler._plan_handoff(_episode_params(episode))
    finally:
        await session.close()

    assert order == [f"marker:{STATE_ACCEPTED}", f"marker:{STATE_ACCEPTING}", "inject"]


@pytest.mark.asyncio
async def test_handler_prepare_never_touches_the_commit_side(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    try:
        await handler._plan_handoff_prepare(_episode_params(episode))
    finally:
        await session.close()

    # PREPARE alone leaves the attempt uncommitted: the flag, the domain record
    # the marker, and the injection all belong to COMMIT.
    sidecar = _sidecar(episode)
    assert sidecar.state == STATE_ACCEPTING
    assert sidecar.c1_verified is False
    assert sidecar.scan_domain == sidecar.owner_domain
    assert not _seeded_messages(loop)


# --- The re-offer continuation (row 3's COMMIT re-entry) --------------------


def _ignore_selection_mismatch(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    monkeypatch.setattr(
        handler, "_plan_handoff_selection_mismatch", lambda *args, **kwargs: None
    )


@pytest.mark.asyncio
async def test_handler_continuation_commits_a_deleted_draft_after_consent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    _ignore_selection_mismatch(monkeypatch, handler)
    try:
        # The crash residue: P1-P3 ran, so the claim is held and the snapshot
        # holds the consented bytes, but no commit followed.
        await handler._plan_handoff_prepare(_episode_params(episode))
        (episode / "plan.md").unlink()
        # Post-consent the draft is NEVER consulted: the re-offer continuation
        # re-enters COMMIT with the snapshot's bytes and cannot fail on the
        # missing draft.
        await handler._plan_handoff(_episode_params(episode))
    finally:
        await session.close()

    sidecar = _sidecar(episode)
    assert sidecar.state == STATE_ACCEPTED
    assert sidecar.executor_session_id == loop.session_id
    # The same attempt continued: PREPARE did not run again.
    assert sidecar.attempt_generation == 1
    assert provenance_snapshot_path(episode, EPISODE_ID, 1).read_bytes() == DOCUMENT
    assert not _snapshot(episode).exists()
    seeds = _seeded_messages(loop)
    assert len(seeds) == 1
    assert DOCUMENT.decode().strip() in seeds[0]


@pytest.mark.asyncio
async def test_handler_continuation_commits_a_draft_changed_after_consent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    _ignore_selection_mismatch(monkeypatch, handler)
    try:
        await handler._plan_handoff_prepare(_episode_params(episode))
        (episode / "plan.md").write_bytes(CHANGED_DOCUMENT)
        await handler._plan_handoff(_episode_params(episode))
    finally:
        await session.close()

    sidecar = _sidecar(episode)
    assert sidecar.state == STATE_ACCEPTED
    assert sidecar.attempt_generation == 1
    # The consented bytes travel, never the edited draft.
    assert provenance_snapshot_path(episode, EPISODE_ID, 1).read_bytes() == DOCUMENT
    assert CHANGED_DOCUMENT.decode().strip() not in _seeded_messages(loop)[0]


@pytest.mark.asyncio
async def test_handler_continuation_verifies_the_carried_bytes_against_consent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    _ignore_selection_mismatch(monkeypatch, handler)
    try:
        await handler._plan_handoff_prepare(_episode_params(episode))
        (episode / "plan.md").unlink()
        # C1's one authority: a body whose own digest is consistent but is not
        # the consented bytes refuses, and the refusal releases the claim.
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff(
                _episode_params(episode, content=CHANGED_DOCUMENT)
            )
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.CONFLICT
    assert caught.value.data is None
    assert CARRIED_DIGEST_MISMATCH_MESSAGE in str(caught.value)
    _assert_back_at_draft(episode)
    assert not _seeded_messages(loop)


@pytest.mark.asyncio
async def test_handler_continuation_accepts_the_provenance_named_survivor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    _ignore_selection_mismatch(monkeypatch, handler)
    provenance = provenance_snapshot_path(episode, EPISODE_ID, 1)
    try:
        await handler._plan_handoff_prepare(_episode_params(episode))
        # The cancel-window crash residue: the snapshot survives under the
        # attempt's provenance name while the sidecar still says accepting.
        _snapshot(episode).rename(provenance)
        await handler._plan_handoff(_episode_params(episode, snapshot_path=provenance))
    finally:
        await session.close()

    sidecar = _sidecar(episode)
    assert sidecar.state == STATE_ACCEPTED
    assert sidecar.attempt_generation == 1
    assert provenance.read_bytes() == DOCUMENT
    assert not _snapshot(episode).exists()
    assert DOCUMENT.decode().strip() in _seeded_messages(loop)[0]


# --- PREPARE's refusal of an unusable document (defense in depth) -----------


@pytest.mark.asyncio
async def test_handler_prepare_refuses_an_empty_draft(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session = await _handler(tmp_path, logging_enabled=True)
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff_prepare(_episode_params(episode, content=b""))
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.CONFLICT
    assert caught.value.data is None
    assert "empty" in str(caught.value)
    assert "re-consent" in str(caught.value)
    # Refused BEFORE the claim: the episode was never touched.
    assert _sidecar(episode).attempt_generation == 0
    _assert_back_at_draft(episode)


@pytest.mark.asyncio
async def test_handler_prepare_refuses_a_summary_less_draft(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session = await _handler(tmp_path, logging_enabled=True)
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff_prepare(
                _episode_params(episode, content=b"# Plan\r\n\r\nNo summary.\r\n")
            )
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.CONFLICT
    assert "no ## Summary section" in str(caught.value)
    assert "re-consent" in str(caught.value)
    assert _sidecar(episode).attempt_generation == 0
    _assert_back_at_draft(episode)


@pytest.mark.asyncio
async def test_handler_prepare_refuses_an_empty_summary_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session = await _handler(tmp_path, logging_enabled=True)
    empty_summary = b"# Plan\r\n\r\n## Summary\r\n\r\n## Steps\r\nOne.\r\n"
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff_prepare(
                _episode_params(episode, content=empty_summary)
            )
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.CONFLICT
    assert "## Summary section is empty" in str(caught.value)
    assert _sidecar(episode).attempt_generation == 0
    _assert_back_at_draft(episode)


# --- The record decides the handoff route (F5) ------------------------------


def _legacy_record(plan_dir: Path) -> PlanModeRecord:
    """A pre-amendment record: no episode fields exist on it at all."""
    return PlanModeRecord(
        plan_path=str(plan_dir / "plan.md"),
        theme="Legacy draft",
        planner_model="alpha",
        planner_thinking="medium",
        draft_path=str(plan_dir / "plan.md"),
    )


@pytest.mark.asyncio
async def test_handler_refuses_the_legacy_route_for_a_legacy_unverified_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    _ignore_selection_mismatch(monkeypatch, handler)
    await loop.session_logger.persist_plan_mode_durable(_legacy_record(episode))
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff(_legacy_params(episode))
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.CONFLICT
    assert caught.value.data is None
    assert "legacy-unverified" in str(caught.value)
    assert "re-bind" in str(caught.value)
    # Nothing ran: no claim, no snapshot, no reset.
    assert loop.session_id == SESSION_ID
    _assert_back_at_draft(episode)
    assert not _seeded_messages(loop)


@pytest.mark.asyncio
async def test_handler_refuses_the_legacy_route_for_an_episode_bound_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    _ignore_selection_mismatch(monkeypatch, handler)
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff(_legacy_params(episode))
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.INVALID_PARAMS
    assert caught.value.data is None
    assert "integrity handoff" in str(caught.value)
    assert loop.session_id == SESSION_ID
    _assert_back_at_draft(episode)
    assert not _seeded_messages(loop)


@pytest.mark.asyncio
async def test_handler_refuses_a_foreign_episode_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    handler, session, loop = await _episode_handler(tmp_path, episode)
    _ignore_selection_mismatch(monkeypatch, handler)
    foreign = _episode_params(episode).model_copy(
        update={
            "episode_id": "ep-foreign",
            "snapshot_path": str(episode / "accepted-ep-foreign.md"),
        }
    )
    try:
        with pytest.raises(RequestFailure) as caught:
            await handler._plan_handoff(foreign)
    finally:
        await session.close()

    assert caught.value.code is ProtocolErrorCode.INVALID_PARAMS
    assert "is not this session's episode" in str(caught.value)
    assert loop.session_id == SESSION_ID
    _assert_back_at_draft(episode)


@pytest.mark.asyncio
async def test_handler_legacy_handoff_still_runs_without_any_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: Path
) -> None:
    # A pre-P10 client shape: no record at all, so the route check has nothing
    # to contradict and the old behavior stands.
    handler, session = await _handler(tmp_path, logging_enabled=True)
    _ignore_selection_mismatch(monkeypatch, handler)
    try:
        await handler._plan_handoff(_legacy_params(episode))
    finally:
        await session.close()

    assert handler._agent_loop.session_id != SESSION_ID
    _assert_back_at_draft(episode)


# --- The §4.4 recovery implementation ---------------------------------------


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


def _source_plan_record(directory: Path, *, state: str = STATE_DRAFT) -> PlanModeRecord:
    return PlanModeRecord(
        plan_path=str(directory / "plan.md"),
        theme="Keep these exact bytes",
        planner_model="alpha",
        planner_thinking="medium",
        episode_id=EPISODE_ID,
        sidecar_path=str(directory / plan_episode.EPISODE_SIDECAR_FILENAME),
        draft_path=str(directory / "plan.md"),
        state=state,
    )


class _RecordSink:
    def __init__(self) -> None:
        self.records: list[PlanModeRecord] = []

    async def __call__(self, record: PlanModeRecord) -> None:
        self.records.append(record)


async def _refuse_record(_record: PlanModeRecord) -> None:
    raise plan_episode.PlanEpisodeWriteError("injected restoration failure")


def _accepted_with_pending_rename(episode: Path) -> None:
    """The crash-between-the-flip-and-the-rename intermediate."""
    write_sidecar(
        episode,
        replace(
            _sidecar(episode),
            state=STATE_ACCEPTED,
            accepted_sha256=sha256_bytes(DOCUMENT),
            executor_session_id=OTHER_SESSION,
            c1_verified=True,
        ),
    )


class TestRecoveryImplementation:
    """The §4.4 recovery rows at the implementation level.

    Each node exercises the recovery implementation's own paths — the
    sidecar-first observation and its recorded-domain scan, the fenced claim
    release, the re-offer and cancel legs with their in-fence re-scans, the
    either-name locate, the completion rule, and the attempt-scoped
    ``c1_verified`` lifecycle. The exhaustive fault-injection matrix is the
    transaction crash-recovery suite's, not this class's.
    """

    def test_the_executor_scan_reads_session_metas_directly(
        self, tmp_path: Path
    ) -> None:
        save_dir = tmp_path / "sessions"
        save_dir.mkdir()
        _session_with_marker(save_dir, suffix="abcd", session_id="exec-1")
        # The listing index is a display cache: an index that knows no sessions
        # must not hide the meta the recovery scan reads.
        (save_dir / ".session_index.json").write_text(
            json.dumps({"version": 1, "entries": {}}), encoding="utf-8"
        )

        scan = plan_episode.scan_episode_executor(
            LoggingDomain(save_dir=str(save_dir), session_prefix="session"), EPISODE_ID
        )

        assert scan.executor_session_id == "exec-1"

    def test_the_scan_is_bounded_to_the_recorded_domain(
        self, episode: Path, tmp_path: Path
    ) -> None:
        recorded = _scan_domain(tmp_path, "recorded")
        elsewhere = _scan_domain(tmp_path, "elsewhere")
        _prepare(episode, owner_domain=recorded)
        _session_with_marker(
            Path(elsewhere.save_dir), suffix="other", session_id="exec-elsewhere"
        )

        observation = observe_episode(
            episode, record=None, session_id=SESSION_ID, owner_domain=recorded
        )
        assert observation.row == RECOVERY_ROW_ACCEPTING_SNAPSHOT

        _session_with_marker(
            Path(recorded.save_dir), suffix="here", session_id="exec-recorded"
        )
        observation = observe_episode(
            episode, record=None, session_id=SESSION_ID, owner_domain=recorded
        )
        assert observation.row == RECOVERY_ROW_POSSIBLY_EXECUTED
        assert observation.executor_session_id == "exec-recorded"

    def test_an_unreadable_recorded_domain_fails_closed(
        self, episode: Path, tmp_path: Path
    ) -> None:
        missing = LoggingDomain(
            save_dir=str(tmp_path / "gone"), session_prefix="session"
        )
        _prepare(episode, owner_domain=missing)

        with pytest.raises(PlanEpisodeScanError):
            observe_episode(
                episode, record=None, session_id=SESSION_ID, owner_domain=missing
            )

    def test_a_corrupt_meta_fails_closed(self, episode: Path, tmp_path: Path) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        broken = Path(domain.save_dir) / "session_20260101_000000_bad"
        broken.mkdir()
        (broken / "meta.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(PlanEpisodeScanError):
            observe_episode(
                episode, record=None, session_id=SESSION_ID, owner_domain=domain
            )

    def test_a_sidecar_loss_clean_miss_is_unresolvable_not_draft(
        self, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        directory = tmp_path / "plan-gone"
        directory.mkdir()
        record = _source_plan_record(directory).model_copy(
            update={
                "scan_domain": PlanScanDomain(
                    save_dir=domain.save_dir, session_prefix="session"
                )
            }
        )

        observation = observe_episode(directory, record=record, session_id=SESSION_ID)

        assert observation.row == RECOVERY_ROW_UNRESOLVABLE
        assert observation.authority == AUTHORITY_RECORD
        assert not observation.recoverable
        assert observation.row != RECOVERY_ROW_DRAFT

    def test_a_sidecar_loss_with_a_marker_surfaces_the_recorded_executor(
        self, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        directory = tmp_path / "plan-gone"
        directory.mkdir()
        record = _source_plan_record(directory).model_copy(
            update={
                "scan_domain": PlanScanDomain(
                    save_dir=domain.save_dir, session_prefix="session"
                )
            }
        )
        _session_with_marker(
            Path(domain.save_dir), suffix="exec", session_id="exec-reclaimed"
        )

        observation = observe_episode(directory, record=record, session_id=SESSION_ID)

        assert observation.row == RECOVERY_ROW_EXECUTED
        assert observation.executor_session_id == "exec-reclaimed"
        assert observation.possibly_executed
        assert not observation.terminal

    def test_a_terminal_record_stays_display_authority_after_reclamation(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "plan-gone"
        directory.mkdir()

        accepted = observe_episode(
            directory,
            record=_source_plan_record(directory, state=STATE_ACCEPTED),
            session_id=SESSION_ID,
        )
        cancelled = observe_episode(
            directory,
            record=_source_plan_record(directory, state=plan_episode.STATE_CANCELLED),
            session_id=SESSION_ID,
        )

        assert accepted.row == RECOVERY_ROW_ACCEPTED
        assert accepted.authority == AUTHORITY_RECORD_TERMINAL
        assert accepted.terminal and not accepted.recoverable
        assert cancelled.row == RECOVERY_ROW_CANCELLED

    def test_a_legacy_record_surfaces_as_legacy_unverified(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "plan-gone"
        directory.mkdir()
        legacy = PlanModeRecord(
            plan_path=str(directory / "plan.md"),
            planner_model="alpha",
            planner_thinking="medium",
        )

        observation = observe_episode(directory, record=legacy, session_id=SESSION_ID)

        assert observation.row == RECOVERY_ROW_LEGACY
        assert observation.state == LEGACY_UNVERIFIED_STATE
        assert not observation.recoverable

    def test_row_two_releases_the_claim_and_unlinks_the_partial_snapshot(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        _snapshot(episode).unlink()
        observation = observe_episode(
            episode, record=None, session_id=SESSION_ID, owner_domain=domain
        )
        assert observation.row == plan_episode.RECOVERY_ROW_ACCEPTING_NO_SNAPSHOT
        assert observation.recoverable

        # A crashed PREPARE may leave a partial snapshot behind.
        _snapshot(episode).write_bytes(b"partial")
        released = recover_release_claim(
            episode, episode_id=EPISODE_ID, session_id=SESSION_ID, owner_domain=domain
        )

        assert released.state == STATE_DRAFT
        assert released.claiming_session_id is None
        assert released.c1_verified is False
        assert released.consent_sha256 is None
        assert not _snapshot(episode).exists()

    def test_row_two_refuses_a_foreign_session(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)

        with pytest.raises(PlanEpisodeOwnerError):
            recover_release_claim(
                episode,
                episode_id=EPISODE_ID,
                session_id=OTHER_SESSION,
                owner_domain=domain,
            )

        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _snapshot(episode).is_file()

    @pytest.mark.asyncio
    async def test_the_re_offer_restores_the_source_record_from_the_sidecar(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        sink = _RecordSink()

        offer = await reopen_accepting_episode(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            # A C2 crash may leave the record at accepted while the sidecar is
            # still accepting; the sidecar is authoritative.
            record=_source_plan_record(episode, state=STATE_ACCEPTED),
            write_record=sink,
        )

        assert offer.content == DOCUMENT
        assert offer.snapshot_path == _snapshot(episode)
        restored = sink.records[0]
        assert restored.state == STATE_ACCEPTING
        assert restored.episode_id == EPISODE_ID
        assert restored.consent_sha256 == sha256_bytes(DOCUMENT)
        assert restored.attempt_generation == 1
        assert restored.scan_domain == PlanScanDomain(
            save_dir=domain.save_dir, session_prefix="session"
        )
        assert restored.snapshot_path == f"accepted-{EPISODE_ID}.md"
        # The claim STAYS held for the confirm.
        assert _sidecar(episode).state == STATE_ACCEPTING

    @pytest.mark.asyncio
    async def test_a_restoration_failure_surfaces_unresolvable(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)

        with pytest.raises(PlanArbiterUnresolvableError):
            await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_plan_record(episode),
                write_record=_refuse_record,
            )

        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _snapshot(episode).read_bytes() == DOCUMENT

    @pytest.mark.asyncio
    async def test_the_re_offer_locates_the_provenance_named_survivor(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        # The cancel-window state: an accepting sidecar whose bytes were already
        # renamed before the return-to-draft write.
        write_sidecar(episode, replace(_sidecar(episode), c1_verified=True))
        provenance = provenance_snapshot_path(episode, EPISODE_ID, 1)
        _snapshot(episode).rename(provenance)

        assert locate_episode_snapshot(episode, _sidecar(episode))[0] == provenance
        sink = _RecordSink()
        offer = await reopen_accepting_episode(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_plan_record(episode),
            write_record=sink,
        )

        assert offer.snapshot_path == provenance
        assert offer.content == DOCUMENT
        assert sink.records[0].snapshot_path == provenance.name
        # C5's continuation treats the completed end state as a verified no-op.
        assert (
            rename_accepted_snapshot(episode, episode_id=EPISODE_ID, generation=1)
            == SNAPSHOT_RENAME_NOOP
        )

    @pytest.mark.asyncio
    async def test_the_re_offer_fails_closed_on_absent_or_mismatched_bytes(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        _snapshot(episode).unlink()

        with pytest.raises(PlanArbiterUnresolvableError):
            await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_plan_record(episode),
                write_record=_RecordSink(),
            )

        _snapshot(episode).write_bytes(CHANGED_DOCUMENT)
        with pytest.raises(PlanDigestMismatchError):
            await reopen_accepting_episode(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_plan_record(episode),
                write_record=_RecordSink(),
            )

        assert _sidecar(episode).state == STATE_ACCEPTING

    @pytest.mark.asyncio
    async def test_a_discovered_executor_refuses_the_cancel(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        _session_with_marker(Path(domain.save_dir), suffix="exec", session_id="exec-1")

        with pytest.raises(PlanEpisodePossiblyExecutedError) as caught:
            await recover_cancel_accepting(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                record=_source_plan_record(episode),
                write_record=_RecordSink(),
            )

        assert caught.value.executor_session_id == "exec-1"
        # DR-57: the refused cancel moved nothing — the possibly-executed row
        # owns the episode now.
        assert _sidecar(episode).state == STATE_ACCEPTING
        assert _snapshot(episode).is_file()
        observation = observe_episode(
            episode, record=None, session_id=SESSION_ID, owner_domain=domain
        )
        assert observation.row == RECOVERY_ROW_POSSIBLY_EXECUTED

    @pytest.mark.asyncio
    async def test_the_cancel_leg_returns_the_episode_and_record_to_draft(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        sink = _RecordSink()

        released = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_plan_record(episode, state=STATE_ACCEPTED),
            write_record=sink,
        )

        assert released.state == STATE_DRAFT
        assert released.c1_verified is False
        assert released.consent_sha256 is None
        assert _sidecar(episode) == released
        assert not _snapshot(episode).exists()
        restored = sink.records[0]
        assert restored.state == STATE_DRAFT
        assert restored.snapshot_path is None
        assert restored.consent_sha256 is None

    @pytest.mark.asyncio
    async def test_the_cancel_leg_renames_a_post_c1_survivor_to_attempt_provenance(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        abort_accept(episode, session_id=SESSION_ID, owner_domain=domain)
        _prepare(episode, owner_domain=domain)
        write_sidecar(episode, replace(_sidecar(episode), c1_verified=True))

        released = await recover_cancel_accepting(
            episode,
            episode_id=EPISODE_ID,
            session_id=SESSION_ID,
            owner_domain=domain,
            record=_source_plan_record(episode),
            write_record=_RecordSink(),
        )

        provenance = provenance_snapshot_path(episode, EPISODE_ID, 2)
        assert provenance.read_bytes() == DOCUMENT
        assert not _snapshot(episode).exists()
        assert released.state == STATE_DRAFT
        assert released.attempt_generation == 2
        # The attempt-scoped flag is cleared in the SAME durable state write.
        persisted = json.loads(
            plan_episode.sidecar_path(episode).read_text(encoding="utf-8")
        )
        assert persisted["state"] == STATE_DRAFT
        assert persisted["c1_verified"] is False

    def test_the_completion_rule_completes_the_pending_rename_at_the_status_surface(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        _accepted_with_pending_rename(episode)

        observation = observe_episode(
            episode, record=None, session_id=SESSION_ID, owner_domain=domain
        )

        provenance = provenance_snapshot_path(episode, EPISODE_ID, 1)
        assert observation.row == RECOVERY_ROW_ACCEPTED
        assert observation.rename_outcome == SNAPSHOT_RENAMED
        assert provenance.read_bytes() == DOCUMENT
        assert not _snapshot(episode).exists()
        assert observation.snapshot_name == provenance.name

    def test_the_completion_rule_completes_at_the_cleanup_surface(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        _accepted_with_pending_rename(episode)
        (episode / PLAN_MARKER_FILENAME).write_text("marker\n", encoding="utf-8")

        removed = plan_episode.cleanup_episode_workspace(
            episode, marker_name=PLAN_MARKER_FILENAME
        )

        provenance = provenance_snapshot_path(episode, EPISODE_ID, 1)
        assert removed is False
        assert provenance.read_bytes() == DOCUMENT
        assert not _snapshot(episode).exists()

    def test_mark_accepted_flips_and_forecloses_the_episode(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        _session_with_marker(Path(domain.save_dir), suffix="exec", session_id="exec-1")

        committed = adjudicate_possibly_executed(
            episode, episode_id=EPISODE_ID, session_id=SESSION_ID, owner_domain=domain
        )

        assert committed.sidecar.state == STATE_ACCEPTED
        assert committed.sidecar.executor_session_id == "exec-1"
        assert committed.sidecar.accepted_sha256 == sha256_bytes(DOCUMENT)
        assert committed.sidecar.c1_verified is True
        assert committed.rename_outcome == SNAPSHOT_RENAMED
        assert provenance_snapshot_path(episode, EPISODE_ID, 1).read_bytes() == DOCUMENT
        assert not _snapshot(episode).exists()
        # The episode is foreclosed: never re-accepted, never re-armed at P3.
        with pytest.raises(PlanEpisodeTerminalError):
            claim_accept(episode, session_id=SESSION_ID, owner_domain=domain)
        with pytest.raises(PlanEpisodeTerminalError):
            publish_consent_snapshot(
                episode,
                session_id=SESSION_ID,
                owner_domain=domain,
                content=DOCUMENT,
                consent_sha256=sha256_bytes(DOCUMENT),
            )
        with pytest.raises(PlanEpisodeTerminalError):
            begin_commit(
                episode,
                episode_id=EPISODE_ID,
                session_id=SESSION_ID,
                owner_domain=domain,
                carried_bytes=DOCUMENT,
                snapshot_path=_snapshot(episode),
            )

    def test_an_occupied_provenance_is_collision_harmless(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        _accepted_with_pending_rename(episode)
        provenance = provenance_snapshot_path(episode, EPISODE_ID, 1)
        provenance.write_bytes(b"survivor\n")

        outcome = complete_accepted_rename(episode, _sidecar(episode))

        assert outcome == SNAPSHOT_RENAME_PENDING
        # Both names are permanent provenance; neither is deleted, and the
        # foreclosed episode never re-arms P3 to collide with either.
        assert _snapshot(episode).read_bytes() == DOCUMENT
        assert provenance.read_bytes() == b"survivor\n"
        with pytest.raises(PlanEpisodeTerminalError):
            publish_consent_snapshot(
                episode,
                session_id=SESSION_ID,
                owner_domain=domain,
                content=DOCUMENT,
                consent_sha256=sha256_bytes(DOCUMENT),
            )

    def test_the_observation_is_sidecar_authoritative_and_owner_only(
        self, episode: Path, tmp_path: Path
    ) -> None:
        domain = _scan_domain(tmp_path)
        _prepare(episode, owner_domain=domain)
        foreign_record = _source_plan_record(episode).model_copy(
            update={"episode_id": "ep-someone-else"}
        )

        foreign = observe_episode(
            episode,
            record=foreign_record,
            session_id=OTHER_SESSION,
            owner_domain=domain,
        )
        owner = observe_episode(
            episode, record=foreign_record, session_id=SESSION_ID, owner_domain=domain
        )

        assert foreign.row == RECOVERY_ROW_ACCEPTING_SNAPSHOT
        assert foreign.authority == AUTHORITY_SIDECAR
        assert foreign.record_mismatch is True
        assert foreign.owner is False
        assert owner.owner is True
        assert (
            surviving_snapshot_name(episode, _sidecar(episode))
            == _snapshot(episode).name
        )
