"""Episode allocation, transition-fence, and cleanup semantics (P10 chunk 1).

Every exercise here is a plain filesystem and ``flock`` interaction in a
temporary directory: no provider, transport, or session runtime is involved.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from threading import Thread
import time
from typing import Any

import pytest

from chartreux.core import plan_episode
from chartreux.core.plan_episode import (
    DOMAIN_REFUSAL,
    EPISODE_FENCE_FILENAME,
    EPISODE_SIDECAR_FILENAME,
    LOGGING_REQUIRED_MESSAGE,
    PLAN_TEMP_PREFIX,
    RECOVERY_ROW_CANCELLED,
    STATE_ACCEPTED,
    STATE_ACCEPTING,
    STATE_CANCELLED,
    STATE_DRAFT,
    LoggingDomain,
    PlanArbiterUnresolvableError,
    PlanDraftReadError,
    PlanEpisodeBindError,
    PlanEpisodeOwnerError,
    PlanEpisodeScanError,
    PlanEpisodeSidecar,
    PlanEpisodeStateError,
    PlanEpisodeTerminalError,
    PlanEpisodeWriteError,
    PlanFenceBusyError,
    PlanLoggingDisabledError,
    TransitionFence,
    abort_accept,
    accepted_snapshot_path,
    cancel_episode,
    claim_accept,
    cleanup_episode_workspace,
    clear_during_claim,
    create_sidecar,
    ensure_episode_fence,
    fence_path,
    guard_plan_entry,
    guard_plan_record_resume,
    new_episode_id,
    observe_episode,
    provenance_snapshot_path,
    publish_episode_snapshot,
    read_draft_bytes,
    read_sidecar,
    rebind_draft,
    release_claim,
    restore_episode_record,
    scan_episode_executor,
    scan_sidecar_executor,
    sidecar_path,
    write_sidecar,
)
from chartreux.core.plan_mode import (
    PLAN_MARKER_FILENAME,
    PlanWorkspaceError,
    allocate_plan_workspace,
    remove_empty_plan_workspace,
)
from chartreux.core.session_types import PlanModeRecord

SESSION_ID = "01234567-89ab-cdef-0123-deadbeef1234"
OTHER_SESSION = "fedcba98-7654-3210-fedc-ba9876543210"
EPISODE_ID = "ep-deadbeef"
SUFFIX = "deadbeef1234"
MARKER_CONTENT = "chartreux plan-mode workspace\n"

DOMAIN = LoggingDomain(save_dir="/tmp/chartreux-plan-a", session_prefix="session")
OTHER_DOMAIN = LoggingDomain(save_dir="/tmp/chartreux-plan-b", session_prefix="session")


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "myproject"
    root.mkdir()
    return root


@pytest.fixture
def episode(tmp_path: Path) -> Path:
    directory = tmp_path / f"plan-{SUFFIX}"
    directory.mkdir()
    (directory / PLAN_MARKER_FILENAME).write_text(MARKER_CONTENT, encoding="utf-8")
    ensure_episode_fence(directory)
    create_sidecar(
        directory,
        PlanEpisodeSidecar(episode_id=EPISODE_ID, draft_path=directory / "plan.md"),
    )
    return directory


@pytest.fixture
def workspaces_root(tmp_path: Path) -> Path:
    return tmp_path / "workspaces"


def _claimed(directory: Path) -> PlanEpisodeSidecar:
    return claim_accept(directory, session_id=SESSION_ID, owner_domain=DOMAIN)


def _sidecar(directory: Path) -> PlanEpisodeSidecar:
    sidecar = read_sidecar(directory)
    assert sidecar is not None
    return sidecar


def _session_meta(
    path: Path,
    *,
    session_id: str,
    episode_id: str | None,
    state: str | None = STATE_ACCEPTING,
    marker_fields: bool = True,
) -> Path:
    """Write one session meta; the default shape is a C3 EXECUTOR marker.

    An episode-carrying record is written with its lifecycle state, and the
    marker's consent/snapshot fields are attached exactly where C3 attaches
    them (an ``accepting`` record written by the handoff).
    """
    payload: dict[str, object] = {"session_id": session_id}
    if episode_id is not None:
        record: dict[str, object] = {"episode_id": episode_id}
        if state is not None:
            record["state"] = state
        if marker_fields and state == STATE_ACCEPTING:
            record["consent_sha256"] = "c" * 64
            record["snapshot_path"] = f"accepted-{episode_id}.md"
        payload["plan_mode"] = record
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _session_directory(save_dir: Path, suffix: str) -> Path:
    directory = save_dir / f"session_20260101_120000_{suffix}"
    directory.mkdir(parents=True)
    return directory


# --- Allocation -------------------------------------------------------------


def test_allocation_claims_the_directory_with_an_exclusive_sidecar(
    project_root: Path, workspaces_root: Path
) -> None:
    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
    )

    directory = workspaces_root / "myproject" / "tasks" / f"plan-{SUFFIX}"
    assert workspace.directory == directory.absolute()
    assert workspace.episode_id is not None
    sidecar = read_sidecar(directory)
    assert sidecar is not None
    assert sidecar.episode_id == workspace.episode_id
    assert sidecar.state == STATE_DRAFT
    assert sidecar.draft_path == workspace.document_path
    assert sidecar.claiming_session_id is None
    assert workspace.sidecar_path == sidecar_path(directory)
    assert (directory / PLAN_MARKER_FILENAME).is_file()
    assert not workspace.document_path.exists()


def test_two_sequential_entries_get_distinct_episodes(
    project_root: Path, workspaces_root: Path
) -> None:
    first = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
    )
    second = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
    )

    assert first.directory != second.directory
    assert first.episode_id != second.episode_id
    assert second.directory.name == f"plan-{SUFFIX}-2"
    assert _sidecar(second.directory).episode_id == second.episode_id


def test_the_probe_skips_a_directory_holding_only_a_sidecar(
    project_root: Path, workspaces_root: Path
) -> None:
    tasks = workspaces_root / "myproject" / "tasks"
    reserved = tasks / f"plan-{SUFFIX}"
    reserved.mkdir(parents=True)
    create_sidecar(
        reserved, PlanEpisodeSidecar(episode_id=EPISODE_ID, draft_path=reserved)
    )

    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
    )

    assert workspace.directory == tasks / f"plan-{SUFFIX}-2"
    assert _sidecar(reserved).episode_id == EPISODE_ID


def test_the_probe_still_skips_a_directory_holding_a_document(
    project_root: Path, workspaces_root: Path
) -> None:
    tasks = workspaces_root / "myproject" / "tasks"
    drafted = tasks / f"plan-{SUFFIX}"
    drafted.mkdir(parents=True)
    (drafted / "plan.md").write_text("# Existing draft\n", encoding="utf-8")

    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
    )

    assert workspace.directory == tasks / f"plan-{SUFFIX}-2"
    assert (drafted / "plan.md").read_text(encoding="utf-8") == "# Existing draft\n"
    assert read_sidecar(drafted) is None


def test_allocation_opens_the_fence_so_the_directory_is_kept(
    project_root: Path, workspaces_root: Path
) -> None:
    workspace = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
    )

    fence = fence_path(workspace.directory)
    assert fence.is_file()
    assert not fence.is_symlink()
    assert sorted(entry.name for entry in workspace.directory.iterdir()) == sorted([
        PLAN_MARKER_FILENAME,
        EPISODE_SIDECAR_FILENAME,
        EPISODE_FENCE_FILENAME,
    ])
    assert remove_empty_plan_workspace(workspace.directory) is False
    assert workspace.directory.is_dir()
    assert fence.is_file()


def test_allocation_fails_loudly_at_the_name_cap(
    project_root: Path, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("chartreux.core.plan_mode._MAX_NAME_ATTEMPTS", 2)
    for _ in range(2):
        allocate_plan_workspace(
            SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
        )

    with pytest.raises(PlanWorkspaceError, match="no free plan workspace name"):
        allocate_plan_workspace(
            SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
        )


# --- The transition fence ---------------------------------------------------


def test_a_second_acquirer_is_refused_as_busy(episode: Path) -> None:
    holder = TransitionFence(episode)
    holder.acquire(holder_session_id=SESSION_ID)
    contender = TransitionFence(episode)
    try:
        with pytest.raises(PlanFenceBusyError, match="busy"):
            contender.acquire(holder_session_id=OTHER_SESSION)
    finally:
        contender.close()
        holder.close()


def test_release_is_unlock_only_and_keeps_the_file(episode: Path) -> None:
    fence = TransitionFence(episode)
    fence.acquire(holder_session_id=SESSION_ID)
    fence.release()
    fence.close()

    assert fence_path(episode).is_file()
    again = TransitionFence(episode)
    again.acquire(holder_session_id=SESSION_ID)
    again.close()


def test_the_holder_payload_is_diagnostic_only(episode: Path) -> None:
    fence = TransitionFence(episode)
    fence.acquire(holder_session_id=SESSION_ID)
    payload = json.loads(fence_path(episode).read_bytes())
    fence.release()
    fence.close()

    assert payload["holder_session_id"] == SESSION_ID
    assert isinstance(payload["pid"], int)
    assert isinstance(payload["acquired_at"], str)
    # A stale payload naming another session proves nothing and blocks nothing.
    successor = TransitionFence(episode)
    successor.acquire(holder_session_id=OTHER_SESSION)
    successor.release()
    successor.close()
    assert json.loads(fence_path(episode).read_bytes())["holder_session_id"] == (
        OTHER_SESSION
    )


def test_a_bounded_retry_succeeds_once_the_holder_releases(episode: Path) -> None:
    holder = TransitionFence(episode)
    holder.acquire(holder_session_id=SESSION_ID)
    started = time.monotonic()

    def _release_soon() -> None:
        time.sleep(0.1)
        holder.release()
        holder.close()

    thread = Thread(target=_release_soon)
    thread.start()
    contender = TransitionFence(episode)
    try:
        contender.acquire(holder_session_id=OTHER_SESSION, retry_timeout=5.0)
    finally:
        contender.close()
        thread.join(timeout=5)

    assert time.monotonic() - started >= 0.1
    assert fence_path(episode).is_file()


# --- Transitions ------------------------------------------------------------


def test_claim_records_the_owner_and_clears_the_commit_flag(episode: Path) -> None:
    sidecar = _claimed(episode)

    assert sidecar.state == STATE_ACCEPTING
    assert sidecar.claiming_session_id == SESSION_ID
    assert sidecar.owner_domain == DOMAIN
    assert sidecar.scan_domain == DOMAIN
    assert sidecar.attempt_generation == 1
    assert sidecar.c1_verified is False
    assert _sidecar(episode) == sidecar


def test_a_second_claim_while_accepting_refuses_with_the_status(episode: Path) -> None:
    _claimed(episode)

    with pytest.raises(PlanEpisodeStateError, match=STATE_ACCEPTING):
        claim_accept(episode, session_id=OTHER_SESSION, owner_domain=DOMAIN)
    assert _sidecar(episode).claiming_session_id == SESSION_ID


def test_a_claim_on_a_terminal_episode_refuses(episode: Path) -> None:
    for state in (STATE_ACCEPTED, STATE_CANCELLED):
        write_sidecar(episode, replace(_sidecar(episode), state=state))
        with pytest.raises(PlanEpisodeTerminalError):
            claim_accept(episode, session_id=SESSION_ID, owner_domain=DOMAIN)
        assert _sidecar(episode).state == state


def test_release_returns_the_episode_to_draft_and_unlinks_the_snapshot(
    episode: Path,
) -> None:
    _claimed(episode)
    snapshot = accepted_snapshot_path(episode, EPISODE_ID)
    publish_episode_snapshot(snapshot, b"# Plan\n")
    write_sidecar(episode, replace(_sidecar(episode), consent_sha256="a" * 64))

    released = release_claim(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

    assert released.state == STATE_DRAFT
    assert released.claiming_session_id is None
    assert released.c1_verified is False
    assert released.consent_sha256 is None
    assert released.attempt_generation == 1
    assert not snapshot.exists()
    assert _sidecar(episode) == released


def test_release_refuses_a_post_commit_claim_and_keeps_the_snapshot(
    episode: Path,
) -> None:
    _claimed(episode)
    write_sidecar(episode, replace(_sidecar(episode), c1_verified=True))
    snapshot = accepted_snapshot_path(episode, EPISODE_ID)
    publish_episode_snapshot(snapshot, b"# Plan\n")

    with pytest.raises(PlanEpisodeStateError, match="post-commit"):
        release_claim(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

    assert snapshot.is_file()
    assert _sidecar(episode).state == STATE_ACCEPTING


def test_release_refuses_a_foreign_session(episode: Path) -> None:
    _claimed(episode)

    with pytest.raises(PlanEpisodeOwnerError, match="owning session"):
        release_claim(episode, session_id=OTHER_SESSION, owner_domain=DOMAIN)
    assert _sidecar(episode).state == STATE_ACCEPTING


def test_abort_accept_is_the_release_procedure(episode: Path) -> None:
    _claimed(episode)
    snapshot = accepted_snapshot_path(episode, EPISODE_ID)
    publish_episode_snapshot(snapshot, b"# Plan\n")

    aborted = abort_accept(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

    assert aborted.state == STATE_DRAFT
    assert aborted.claiming_session_id is None
    assert not snapshot.exists()


def test_a_fresh_claim_clears_the_previous_commit_flag(episode: Path) -> None:
    write_sidecar(
        episode, replace(_sidecar(episode), c1_verified=True, attempt_generation=4)
    )

    sidecar = _claimed(episode)

    assert sidecar.c1_verified is False
    assert sidecar.attempt_generation == 5


def test_a_pre_rename_write_failure_leaves_the_previous_state(
    episode: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _refuse(_directory: Path, _sidecar: PlanEpisodeSidecar) -> None:
        raise OSError("the disk went away")

    monkeypatch.setattr(plan_episode, "write_sidecar", _refuse)

    with pytest.raises(PlanEpisodeWriteError) as caught:
        claim_accept(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

    assert "possibly published" not in str(caught.value)
    assert _sidecar(episode).state == STATE_DRAFT


def test_a_write_failure_after_publication_reports_a_possibly_published_state(
    episode: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _write_then_fail(directory: Path, sidecar: PlanEpisodeSidecar) -> None:
        sidecar_path(directory).write_bytes(plan_episode.encode_sidecar(sidecar))
        raise OSError("the directory barrier failed")

    monkeypatch.setattr(plan_episode, "write_sidecar", _write_then_fail)

    with pytest.raises(PlanEpisodeWriteError, match="possibly published"):
        claim_accept(episode, session_id=SESSION_ID, owner_domain=DOMAIN)
    assert _sidecar(episode).state == STATE_ACCEPTING


def test_a_write_failure_with_an_unexpected_state_surfaces_unresolvable(
    episode: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _write_garbage(directory: Path, _sidecar: PlanEpisodeSidecar) -> None:
        sidecar_path(directory).write_bytes(b'{"kind":"something-else"}\n')
        raise OSError("the write went strange")

    monkeypatch.setattr(plan_episode, "write_sidecar", _write_garbage)

    with pytest.raises(PlanArbiterUnresolvableError, match="unresolvable"):
        claim_accept(episode, session_id=SESSION_ID, owner_domain=DOMAIN)


def test_rebind_rewrites_the_bound_draft_path_under_the_fence(
    episode: Path, tmp_path: Path
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "other-plan.md"
    target.write_text("# Other\n", encoding="utf-8")
    link = root / "linked-plan.md"
    link.symlink_to(target)

    rebound = rebind_draft(
        episode, draft_path=link, authorized_roots=[root], session_id=SESSION_ID
    )

    assert rebound.draft_path == target.resolve()
    assert _sidecar(episode).draft_path == target.resolve()


def test_rebind_refuses_a_path_outside_the_authorized_roots(
    episode: Path, tmp_path: Path
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    with pytest.raises(PlanEpisodeBindError, match="outside the authorized roots"):
        rebind_draft(
            episode,
            draft_path=tmp_path / "elsewhere.md",
            authorized_roots=[root],
            session_id=SESSION_ID,
        )
    assert _sidecar(episode).draft_path == episode / "plan.md"


def test_rebind_refuses_while_a_claim_is_held(episode: Path, tmp_path: Path) -> None:
    _claimed(episode)
    root = tmp_path / "project"
    root.mkdir()

    with pytest.raises(PlanEpisodeStateError, match="abort-accept first"):
        rebind_draft(
            episode,
            draft_path=root / "other.md",
            authorized_roots=[root],
            session_id=SESSION_ID,
            owner_domain=DOMAIN,
        )


# --- The terminal cancel (/plan cancel outside the accept flow) --------------


def test_cancel_outside_the_accept_flow_writes_the_terminal_sidecar(
    episode: Path,
) -> None:
    cancelled = cancel_episode(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

    assert cancelled.state == STATE_CANCELLED
    assert cancelled.claiming_session_id is None
    assert cancelled.c1_verified is False
    assert _sidecar(episode) == cancelled
    assert _sidecar(episode).state == STATE_CANCELLED
    # The durable write goes through the plan-mode temporary-name wrapper.
    assert [
        entry.name
        for entry in episode.iterdir()
        if entry.name.startswith(PLAN_TEMP_PREFIX)
    ] == []


def test_cancel_takes_the_transition_fence(episode: Path) -> None:
    holder = TransitionFence(episode)
    holder.acquire(holder_session_id=SESSION_ID)
    try:
        with pytest.raises(PlanFenceBusyError, match="busy"):
            cancel_episode(episode, session_id=SESSION_ID, owner_domain=DOMAIN)
    finally:
        holder.close()

    assert _sidecar(episode).state == STATE_DRAFT
    assert (
        cancel_episode(episode, session_id=SESSION_ID, owner_domain=DOMAIN).state
        == STATE_CANCELLED
    )


def test_cancel_is_terminal_with_no_transition_out(episode: Path) -> None:
    for state in (STATE_CANCELLED, STATE_ACCEPTED):
        write_sidecar(episode, replace(_sidecar(episode), state=state))

        with pytest.raises(PlanEpisodeTerminalError):
            cancel_episode(episode, session_id=SESSION_ID, owner_domain=DOMAIN)
        with pytest.raises(PlanEpisodeTerminalError):
            claim_accept(episode, session_id=SESSION_ID, owner_domain=DOMAIN)
        with pytest.raises(PlanEpisodeTerminalError):
            rebind_draft(
                episode,
                draft_path=episode / "other.md",
                authorized_roots=[episode],
                session_id=SESSION_ID,
                owner_domain=DOMAIN,
            )
        assert _sidecar(episode).state == state


def test_cancel_refuses_a_foreign_logging_domain(episode: Path) -> None:
    _claimed(episode)
    release_claim(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

    with pytest.raises(PlanEpisodeOwnerError) as caught:
        cancel_episode(episode, session_id=SESSION_ID, owner_domain=OTHER_DOMAIN)

    assert str(caught.value) == DOMAIN_REFUSAL
    assert _sidecar(episode).state == STATE_DRAFT


def test_cancel_while_accepting_is_abort_accept(episode: Path) -> None:
    _claimed(episode)
    snapshot = accepted_snapshot_path(episode, EPISODE_ID)
    publish_episode_snapshot(snapshot, b"# Plan\n")

    aborted = cancel_episode(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

    assert aborted.state == STATE_DRAFT
    assert aborted.claiming_session_id is None
    assert not snapshot.exists()
    assert _sidecar(episode).state == STATE_DRAFT


def test_cancel_while_accepting_refuses_a_foreign_session(episode: Path) -> None:
    _claimed(episode)

    with pytest.raises(PlanEpisodeOwnerError, match="owning session"):
        cancel_episode(episode, session_id=OTHER_SESSION, owner_domain=DOMAIN)

    assert _sidecar(episode).state == STATE_ACCEPTING


def test_cancel_while_accepting_refuses_a_post_commit_claim(episode: Path) -> None:
    _claimed(episode)
    write_sidecar(episode, replace(_sidecar(episode), c1_verified=True))

    with pytest.raises(PlanEpisodeStateError, match="post-commit"):
        cancel_episode(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

    assert _sidecar(episode).state == STATE_ACCEPTING


def test_a_cancelled_episode_observes_as_terminal_history(episode: Path) -> None:
    cancel_episode(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

    observation = observe_episode(
        episode, record=None, session_id=SESSION_ID, owner_domain=DOMAIN
    )

    assert observation.row == RECOVERY_ROW_CANCELLED
    assert observation.state == STATE_CANCELLED
    assert observation.terminal is True
    assert observation.recoverable is False
    assert observation.possibly_executed is False


def test_a_cancelled_episode_never_rearms_a_restored_record(episode: Path) -> None:
    record = PlanModeRecord(
        plan_path=str(episode / "plan.md"),
        planner_model="alpha",
        planner_thinking="medium",
        episode_id=EPISODE_ID,
        sidecar_path=str(sidecar_path(episode)),
        draft_path=str(episode / "plan.md"),
        state=STATE_DRAFT,
    )
    cancel_episode(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

    restored = restore_episode_record(
        record, session_id=SESSION_ID, owner_domain=DOMAIN
    )

    assert restored is not None
    assert restored.record.state == STATE_CANCELLED
    assert restored.observation.row == RECOVERY_ROW_CANCELLED
    assert restored.observation.terminal is True


# --- Re-bind containment (F6) ----------------------------------------------


def test_rebind_revalidates_under_the_fence_and_refuses_a_swap(
    episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    # The swap lands on another AUTHORIZED path, so only the re-validation —
    # not the roots check — can refuse it.
    other = root / "other-plan.md"
    other.write_text("# Other\n", encoding="utf-8")
    link = root / "linked-plan.md"
    link.write_text("# Plan\n", encoding="utf-8")

    real = plan_episode.canonicalize_authorized_path
    calls: list[Path] = []

    def _swap_after_validation(path: Path, roots: Any) -> Path:
        canonical = real(path, roots)
        if not calls:
            calls.append(canonical)
            link.unlink()
            link.symlink_to(other)
        return canonical

    monkeypatch.setattr(
        plan_episode, "canonicalize_authorized_path", _swap_after_validation
    )

    with pytest.raises(PlanEpisodeBindError, match="moved while it was being"):
        rebind_draft(
            episode, draft_path=link, authorized_roots=[root], session_id=SESSION_ID
        )

    assert _sidecar(episode).draft_path == episode / "plan.md"


def test_a_bound_read_refuses_a_leaf_swapped_for_a_symlink(
    episode: Path, tmp_path: Path
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "plan.md"
    target.write_text("# Plan\n", encoding="utf-8")
    outside = tmp_path / "secret.md"
    outside.write_text("# Secret\n", encoding="utf-8")

    rebind_draft(
        episode, draft_path=target, authorized_roots=[root], session_id=SESSION_ID
    )
    bound = _sidecar(episode).draft_path
    assert read_draft_bytes(bound) == b"# Plan\n"

    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(PlanDraftReadError, match="moved after it was bound"):
        read_draft_bytes(bound)


def test_a_bound_read_refuses_an_intermediate_directory_swap(
    episode: Path, tmp_path: Path
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    documents = root / "documents"
    documents.mkdir()
    target = documents / "plan.md"
    target.write_text("# Plan\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "plan.md").write_text("# Secret\n", encoding="utf-8")

    rebind_draft(
        episode, draft_path=target, authorized_roots=[root], session_id=SESSION_ID
    )
    bound = _sidecar(episode).draft_path

    documents.rename(root / "documents-away")
    documents.symlink_to(outside)

    with pytest.raises(PlanDraftReadError, match="moved after it was bound"):
        read_draft_bytes(bound)


def test_a_bound_read_refuses_a_swap_that_lands_mid_read(
    episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "plan.md"
    target.write_text("# Plan\n", encoding="utf-8")
    rebind_draft(
        episode, draft_path=target, authorized_roots=[root], session_id=SESSION_ID
    )
    bound = _sidecar(episode).draft_path

    real = plan_episode._bound_resolution
    resolutions: list[Path] = []

    def _swapping_resolution(path: Path) -> Path:
        resolutions.append(path)
        if len(resolutions) == 1:
            return real(path)
        return Path(path).parent / "elsewhere.md"

    monkeypatch.setattr(plan_episode, "_bound_resolution", _swapping_resolution)

    with pytest.raises(PlanDraftReadError, match="moved while it was being read"):
        read_draft_bytes(bound)


# --- /clear during an accept claim -----------------------------------------


def test_clear_releases_the_claim_before_the_record_clears(episode: Path) -> None:
    _claimed(episode)
    snapshot = accepted_snapshot_path(episode, EPISODE_ID)
    publish_episode_snapshot(snapshot, b"# Plan\n")
    observed: list[tuple[str, bool]] = []

    released = clear_during_claim(
        episode,
        session_id=SESSION_ID,
        owner_domain=DOMAIN,
        clear_record=lambda: observed.append((
            _sidecar(episode).state,
            snapshot.exists(),
        )),
    )

    assert released is not None
    assert released.state == STATE_DRAFT
    assert observed == [(STATE_DRAFT, False)]
    assert _sidecar(episode).state == STATE_DRAFT


def test_clear_without_a_claim_only_clears_the_record(episode: Path) -> None:
    cleared: list[bool] = []

    released = clear_during_claim(
        episode,
        session_id=SESSION_ID,
        owner_domain=DOMAIN,
        clear_record=lambda: cleared.append(True),
    )

    assert released is None
    assert cleared == [True]
    assert _sidecar(episode).state == STATE_DRAFT


def test_clear_refuses_a_foreign_claim_and_keeps_the_record(episode: Path) -> None:
    _claimed(episode)
    cleared: list[bool] = []

    with pytest.raises(PlanEpisodeOwnerError):
        clear_during_claim(
            episode,
            session_id=OTHER_SESSION,
            owner_domain=DOMAIN,
            clear_record=lambda: cleared.append(True),
        )

    assert cleared == []
    assert _sidecar(episode).state == STATE_ACCEPTING


def test_clear_refuses_a_post_commit_claim_and_keeps_the_record(episode: Path) -> None:
    _claimed(episode)
    write_sidecar(episode, replace(_sidecar(episode), c1_verified=True))
    cleared: list[bool] = []

    with pytest.raises(PlanEpisodeStateError, match="post-commit"):
        clear_during_claim(
            episode,
            session_id=SESSION_ID,
            owner_domain=DOMAIN,
            clear_record=lambda: cleared.append(True),
        )

    assert cleared == []
    assert _sidecar(episode).state == STATE_ACCEPTING


# --- The logging-disabled entry guard --------------------------------------


def test_plan_entry_refuses_without_session_logging() -> None:
    with pytest.raises(PlanLoggingDisabledError) as caught:
        guard_plan_entry(False)

    assert str(caught.value) == LOGGING_REQUIRED_MESSAGE
    assert "session logging" in str(caught.value)


def test_plan_record_resume_refuses_without_session_logging() -> None:
    with pytest.raises(PlanLoggingDisabledError) as caught:
        guard_plan_record_resume(False)

    assert str(caught.value) == LOGGING_REQUIRED_MESSAGE


def test_both_surfaces_admit_an_enabled_logging_configuration() -> None:
    assert guard_plan_entry(True) is None
    assert guard_plan_record_resume(True) is None


# --- The owner-domain guard -------------------------------------------------


def test_the_owner_domain_is_recorded_at_p1_and_survives_a_release(
    episode: Path,
) -> None:
    claimed = _claimed(episode)
    released = release_claim(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

    assert claimed.owner_domain == DOMAIN
    assert claimed.scan_domain == DOMAIN
    assert released.owner_domain == DOMAIN


def test_a_claim_from_another_logging_domain_refuses(episode: Path) -> None:
    _claimed(episode)
    release_claim(episode, session_id=SESSION_ID, owner_domain=DOMAIN)

    with pytest.raises(PlanEpisodeOwnerError) as caught:
        claim_accept(episode, session_id=OTHER_SESSION, owner_domain=OTHER_DOMAIN)

    assert str(caught.value) == DOMAIN_REFUSAL
    assert _sidecar(episode).state == STATE_DRAFT


def test_a_release_requires_the_matching_domain_for_the_right_session(
    episode: Path,
) -> None:
    _claimed(episode)

    with pytest.raises(PlanEpisodeOwnerError) as caught:
        release_claim(episode, session_id=SESSION_ID, owner_domain=OTHER_DOMAIN)

    assert str(caught.value) == DOMAIN_REFUSAL
    assert _sidecar(episode).state == STATE_ACCEPTING


def test_a_rebind_from_another_logging_domain_refuses(
    episode: Path, tmp_path: Path
) -> None:
    _claimed(episode)
    release_claim(episode, session_id=SESSION_ID, owner_domain=DOMAIN)
    root = tmp_path / "project"
    root.mkdir()

    with pytest.raises(PlanEpisodeOwnerError) as caught:
        rebind_draft(
            episode,
            draft_path=root / "other.md",
            authorized_roots=[root],
            session_id=SESSION_ID,
            owner_domain=OTHER_DOMAIN,
        )

    assert str(caught.value) == DOMAIN_REFUSAL


# --- The bounded executor scan ---------------------------------------------


def test_the_scan_reports_a_clean_miss(tmp_path: Path) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    directory = _session_directory(save_dir, "aaaa1111")
    _session_meta(directory / "meta.json", session_id="aaaa1111", episode_id=None)

    scan = scan_episode_executor(
        LoggingDomain(save_dir=str(save_dir), session_prefix="session"), EPISODE_ID
    )

    assert scan.found is False
    assert scan.executor_session_id is None


def test_the_scan_finds_the_episode_marker(tmp_path: Path) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    directory = _session_directory(save_dir, "bbbb2222")
    _session_meta(
        directory / "meta.json", session_id="bbbb2222-session", episode_id=EPISODE_ID
    )

    scan = scan_episode_executor(
        LoggingDomain(save_dir=str(save_dir), session_prefix="session"), EPISODE_ID
    )

    assert scan.found is True
    assert scan.executor_session_id == "bbbb2222-session"


def test_the_scan_fails_closed_on_an_unreadable_domain(tmp_path: Path) -> None:
    with pytest.raises(PlanEpisodeScanError, match="unreadable"):
        scan_episode_executor(
            LoggingDomain(save_dir=str(tmp_path / "missing"), session_prefix="session"),
            EPISODE_ID,
        )


def test_the_scan_fails_closed_on_a_missing_or_corrupt_meta(tmp_path: Path) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    missing = _session_directory(save_dir, "cccc3333")
    corrupt = _session_directory(save_dir, "dddd4444")
    (corrupt / "meta.json").write_bytes(b"{not json")

    domain = LoggingDomain(save_dir=str(save_dir), session_prefix="session")
    with pytest.raises(PlanEpisodeScanError, match="cannot read the session meta"):
        scan_episode_executor(domain, EPISODE_ID)

    (missing / "meta.json").write_bytes(b"{not json")
    with pytest.raises(PlanEpisodeScanError, match="corrupt"):
        scan_episode_executor(domain, EPISODE_ID)


def test_the_scan_fails_closed_on_two_matching_executors(tmp_path: Path) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    for suffix in ("eeee5555", "ffff6666"):
        directory = _session_directory(save_dir, suffix)
        _session_meta(directory / "meta.json", session_id=suffix, episode_id=EPISODE_ID)

    with pytest.raises(PlanEpisodeScanError, match="several executor sessions"):
        scan_episode_executor(
            LoggingDomain(save_dir=str(save_dir), session_prefix="session"), EPISODE_ID
        )


def test_the_scan_never_matches_a_predecessor_planning_mirror(tmp_path: Path) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    # The departed planner's own record: same episode id, its own lifecycle
    # state (draft before C2's retirement, accepted after it), no marker fields.
    for suffix, state in (
        ("plan-draft", STATE_DRAFT),
        ("plan-accepted", STATE_ACCEPTED),
        ("plan-cancelled", STATE_CANCELLED),
    ):
        directory = _session_directory(save_dir, suffix)
        _session_meta(
            directory / "meta.json",
            session_id=suffix,
            episode_id=EPISODE_ID,
            state=state,
            marker_fields=False,
        )

    scan = scan_episode_executor(
        LoggingDomain(save_dir=str(save_dir), session_prefix="session"), EPISODE_ID
    )

    assert scan.found is False
    assert scan.executor_session_id is None


def test_the_scan_still_finds_the_executor_beside_a_planning_mirror(
    tmp_path: Path,
) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    mirror = _session_directory(save_dir, "plan-accepted")
    _session_meta(
        mirror / "meta.json",
        session_id="planner-session",
        episode_id=EPISODE_ID,
        state=STATE_ACCEPTED,
        marker_fields=False,
    )
    executor = _session_directory(save_dir, "exec")
    _session_meta(
        executor / "meta.json", session_id="executor-session", episode_id=EPISODE_ID
    )

    scan = scan_episode_executor(
        LoggingDomain(save_dir=str(save_dir), session_prefix="session"), EPISODE_ID
    )

    assert scan.executor_session_id == "executor-session"


def test_the_excluded_session_is_dropped_before_the_ambiguity_rule(
    tmp_path: Path,
) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    for suffix in ("one", "two"):
        directory = _session_directory(save_dir, suffix)
        _session_meta(directory / "meta.json", session_id=suffix, episode_id=EPISODE_ID)

    domain = LoggingDomain(save_dir=str(save_dir), session_prefix="session")
    with pytest.raises(PlanEpisodeScanError):
        scan_episode_executor(domain, EPISODE_ID)

    scan = scan_episode_executor(domain, EPISODE_ID, excluding_session_id="one")

    assert scan.executor_session_id == "two"


def test_a_non_marker_shaped_episode_meta_is_a_clean_miss(tmp_path: Path) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    domain = LoggingDomain(save_dir=str(save_dir), session_prefix="session")

    # An accepting record with no marker fields, and one with no lifecycle
    # state at all: neither is C3's marker, so neither is an executor.
    markerless = _session_directory(save_dir, "markerless")
    _session_meta(
        markerless / "meta.json",
        session_id="markerless",
        episode_id=EPISODE_ID,
        marker_fields=False,
    )
    stateless = _session_directory(save_dir, "stateless")
    _session_meta(
        stateless / "meta.json",
        session_id="stateless",
        episode_id=EPISODE_ID,
        state=None,
    )

    scan = scan_episode_executor(domain, EPISODE_ID)

    assert scan.found is False


def test_the_scan_narrows_to_the_recorded_executor(tmp_path: Path) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    domain = LoggingDomain(save_dir=str(save_dir), session_prefix="session")

    # Two marker-shaped metas: the launch and a planning session that restored
    # the SAME accepting episode (record_from_sidecar rebuilds that shape).
    for suffix in ("exec", "planner"):
        directory = _session_directory(save_dir, suffix)
        _session_meta(directory / "meta.json", session_id=suffix, episode_id=EPISODE_ID)

    # Shape alone cannot decide between them; it fails closed as ambiguous.
    with pytest.raises(PlanEpisodeScanError, match="several executor sessions"):
        scan_episode_executor(domain, EPISODE_ID)

    scan = scan_episode_executor(domain, EPISODE_ID, expected_session_id="exec")

    assert scan.executor_session_id == "exec"


def test_a_recorded_executor_with_no_matching_meta_fails_closed(tmp_path: Path) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    domain = LoggingDomain(save_dir=str(save_dir), session_prefix="session")
    # The restored planning mirror is marker-shaped with its own session id;
    # the sidecar's C4-recorded executor has no meta at all. C4 runs AFTER the
    # C3 marker write, so execution may have begun — never a clean miss.
    planner = _session_directory(save_dir, "planner")
    _session_meta(
        planner / "meta.json", session_id="planner-session", episode_id=EPISODE_ID
    )

    with pytest.raises(PlanEpisodeScanError, match="no session meta matches"):
        scan_episode_executor(
            domain, EPISODE_ID, expected_session_id="executor-session"
        )


def test_a_markerless_recorded_executor_meta_fails_closed(tmp_path: Path) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    domain = LoggingDomain(save_dir=str(save_dir), session_prefix="session")
    # The recorded executor's own meta exists but is not marker-shaped: a
    # marker that is not durable is as fail-closed as a missing meta.
    directory = _session_directory(save_dir, "executor")
    _session_meta(
        directory / "meta.json",
        session_id="executor-session",
        episode_id=EPISODE_ID,
        marker_fields=False,
    )

    with pytest.raises(PlanEpisodeScanError, match="no session meta matches"):
        scan_episode_executor(
            domain, EPISODE_ID, expected_session_id="executor-session"
        )


def test_a_present_recorded_executor_meta_is_found(tmp_path: Path) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    domain = LoggingDomain(save_dir=str(save_dir), session_prefix="session")
    for suffix, session_id in (
        ("exec", "executor-session"),
        ("planner", "planner-session"),
    ):
        directory = _session_directory(save_dir, suffix)
        _session_meta(
            directory / "meta.json", session_id=session_id, episode_id=EPISODE_ID
        )

    scan = scan_episode_executor(
        domain, EPISODE_ID, expected_session_id="executor-session"
    )

    assert scan.executor_session_id == "executor-session"


def test_the_recorded_executor_is_a_miss_only_when_it_is_the_observer(
    tmp_path: Path,
) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    domain = LoggingDomain(save_dir=str(save_dir), session_prefix="session")
    directory = _session_directory(save_dir, "exec")
    _session_meta(
        directory / "meta.json", session_id="executor-session", episode_id=EPISODE_ID
    )

    # The observer excluding its own recorded-executor meta is the one
    # legitimate miss; any other empty match fails closed.
    scan = scan_episode_executor(
        domain,
        EPISODE_ID,
        expected_session_id="executor-session",
        excluding_session_id="executor-session",
    )

    assert scan.found is False
    with pytest.raises(PlanEpisodeScanError, match="no session meta matches"):
        scan_episode_executor(
            domain, EPISODE_ID, expected_session_id="another-executor"
        )


def test_the_pre_c4_fallback_still_clean_misses_without_a_marker(
    tmp_path: Path,
) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    domain = LoggingDomain(save_dir=str(save_dir), session_prefix="session")
    planner = _session_directory(save_dir, "planner")
    _session_meta(
        planner / "meta.json",
        session_id="planner-session",
        episode_id=EPISODE_ID,
        state=STATE_DRAFT,
        marker_fields=False,
    )

    # No C4 id recorded: the marker shape alone decides, exactly as before.
    scan = scan_episode_executor(domain, EPISODE_ID, expected_session_id=None)

    assert scan.found is False


def test_the_sidecar_scan_threads_its_recorded_executor(
    episode: Path, tmp_path: Path
) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    domain = LoggingDomain(save_dir=str(save_dir), session_prefix="session")
    write_sidecar(
        episode,
        replace(
            _sidecar(episode),
            state=STATE_ACCEPTING,
            claiming_session_id=SESSION_ID,
            owner_domain=domain,
            scan_domain=domain,
            consent_sha256="c" * 64,
            executor_session_id="exec-session",
        ),
    )
    for suffix in ("exec", "planner"):
        directory = _session_directory(save_dir, suffix)
        _session_meta(
            directory / "meta.json",
            session_id="exec-session" if suffix == "exec" else "planner-session",
            episode_id=EPISODE_ID,
        )

    scan = scan_sidecar_executor(_sidecar(episode), excluding_session_id="observer")

    assert scan.executor_session_id == "exec-session"


# --- Cleanup ----------------------------------------------------------------


def test_cleanup_removes_the_marker_the_sidecar_and_the_temps(episode: Path) -> None:
    temp = episode / f"{PLAN_TEMP_PREFIX}1234-abcd"
    temp.write_bytes(b"partial")

    removed = cleanup_episode_workspace(episode, marker_name=PLAN_MARKER_FILENAME)

    assert removed is False
    assert episode.is_dir()
    assert fence_path(episode).is_file()
    assert not (episode / PLAN_MARKER_FILENAME).exists()
    assert not sidecar_path(episode).exists()
    assert not temp.exists()


def test_cleanup_removes_a_pre_commit_snapshot(episode: Path) -> None:
    snapshot = accepted_snapshot_path(episode, EPISODE_ID)
    publish_episode_snapshot(snapshot, b"# Plan\n")

    cleanup_episode_workspace(episode, marker_name=PLAN_MARKER_FILENAME)

    assert not snapshot.exists()


def test_cleanup_keeps_a_post_commit_snapshot(episode: Path) -> None:
    write_sidecar(episode, replace(_sidecar(episode), c1_verified=True))
    snapshot = accepted_snapshot_path(episode, EPISODE_ID)
    publish_episode_snapshot(snapshot, b"# Plan\n")

    cleanup_episode_workspace(episode, marker_name=PLAN_MARKER_FILENAME)

    assert snapshot.read_bytes() == b"# Plan\n"


def test_cleanup_never_removes_a_provenance_named_snapshot(episode: Path) -> None:
    write_sidecar(episode, replace(_sidecar(episode), state=STATE_ACCEPTED))
    provenance = provenance_snapshot_path(episode, EPISODE_ID, 2)
    publish_episode_snapshot(provenance, b"# Plan\n")

    cleanup_episode_workspace(episode, marker_name=PLAN_MARKER_FILENAME)

    assert provenance.read_bytes() == b"# Plan\n"
    assert episode.is_dir()


def test_cleanup_keeps_a_snapshot_when_the_executor_scan_matches(
    episode: Path, tmp_path: Path
) -> None:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    directory = _session_directory(save_dir, "9999aaaa")
    _session_meta(directory / "meta.json", session_id="9999aaaa", episode_id=EPISODE_ID)
    write_sidecar(
        episode,
        replace(
            _sidecar(episode),
            scan_domain=LoggingDomain(save_dir=str(save_dir), session_prefix="session"),
        ),
    )
    snapshot = accepted_snapshot_path(episode, EPISODE_ID)
    publish_episode_snapshot(snapshot, b"# Plan\n")

    cleanup_episode_workspace(episode, marker_name=PLAN_MARKER_FILENAME)

    assert snapshot.is_file()


def test_cleanup_keeps_a_snapshot_it_cannot_classify(
    episode: Path, tmp_path: Path
) -> None:
    write_sidecar(
        episode,
        replace(
            _sidecar(episode),
            scan_domain=LoggingDomain(
                save_dir=str(tmp_path / "gone"), session_prefix="session"
            ),
        ),
    )
    snapshot = accepted_snapshot_path(episode, EPISODE_ID)
    publish_episode_snapshot(snapshot, b"# Plan\n")

    cleanup_episode_workspace(episode, marker_name=PLAN_MARKER_FILENAME)

    assert snapshot.is_file()


def test_cleanup_refuses_while_a_claim_is_held(episode: Path) -> None:
    _claimed(episode)

    with pytest.raises(PlanEpisodeStateError, match="accepting"):
        cleanup_episode_workspace(episode, marker_name=PLAN_MARKER_FILENAME)

    assert (episode / PLAN_MARKER_FILENAME).is_file()
    assert sidecar_path(episode).is_file()


def test_cleanup_keeps_a_foreign_entry(episode: Path) -> None:
    (episode / "notes.txt").write_text("mine\n", encoding="utf-8")

    removed = cleanup_episode_workspace(episode, marker_name=PLAN_MARKER_FILENAME)

    assert removed is False
    assert (episode / "notes.txt").read_text(encoding="utf-8") == "mine\n"
    assert not sidecar_path(episode).exists()


def test_cleanup_is_a_noop_without_the_marker(episode: Path) -> None:
    (episode / PLAN_MARKER_FILENAME).unlink()

    assert cleanup_episode_workspace(episode, marker_name=PLAN_MARKER_FILENAME) is False
    assert sidecar_path(episode).is_file()


def test_cleanup_removes_a_directory_that_holds_nothing_else(tmp_path: Path) -> None:
    directory = tmp_path / f"plan-{SUFFIX}"
    directory.mkdir()
    (directory / PLAN_MARKER_FILENAME).write_text(MARKER_CONTENT, encoding="utf-8")

    assert (
        cleanup_episode_workspace(directory, marker_name=PLAN_MARKER_FILENAME) is True
    )
    assert not directory.exists()


# --- Accumulation -----------------------------------------------------------


def test_episode_directories_accumulate_and_never_reuse_a_draft(
    project_root: Path, workspaces_root: Path
) -> None:
    first = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
    )
    first.document_path.write_text("# First plan\n", encoding="utf-8")
    second = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
    )

    assert first.directory != second.directory
    assert remove_empty_plan_workspace(first.directory) is False
    assert first.document_path.read_text(encoding="utf-8") == "# First plan\n"
    assert first.directory.is_dir()

    third = allocate_plan_workspace(
        SESSION_ID, project_root=project_root, workspaces_root=workspaces_root
    )
    assert third.directory not in (first.directory, second.directory)
    assert new_episode_id() != new_episode_id()
