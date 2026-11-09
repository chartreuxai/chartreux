"""The plan-episode flow: resume surfaces, pin restore, rebind, and /clear.

The sidecar is the arbiter for every surface here; the TUI never re-arms a mode
on anything but a draft episode.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chartreux.app_server.protocol import ConfigFieldKind, ConfigFieldWire
from chartreux.cli.textual_ui import plan_mode as tui_plan_mode
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.plan_mode import (
    EPISODE_ROW_ACCEPTED,
    EPISODE_ROW_ACCEPTING_NO_SNAPSHOT,
    EPISODE_ROW_ACCEPTING_SNAPSHOT,
    EPISODE_ROW_CANCELLED,
    EPISODE_ROW_DRAFT,
    EPISODE_ROW_EXECUTED,
    EPISODE_ROW_LEGACY,
    EPISODE_ROW_POSSIBLY_EXECUTED,
    EPISODE_ROW_UNRESOLVABLE,
    PLAN_MODE_TINT_CLASS,
    ConfirmedPlan,
    EpisodeView,
    PlanWorkspacePaths,
    digest_bytes,
    episode_actions,
    episode_status_lines,
)
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.plan_episode import (
    EPISODE_FENCE_FILENAME,
    RECOVERY_ROW_ACCEPTED,
    RECOVERY_ROW_ACCEPTING_NO_SNAPSHOT,
    RECOVERY_ROW_ACCEPTING_SNAPSHOT,
    RECOVERY_ROW_CANCELLED,
    RECOVERY_ROW_DRAFT,
    RECOVERY_ROW_EXECUTED,
    RECOVERY_ROW_LEGACY,
    RECOVERY_ROW_POSSIBLY_EXECUTED,
    RECOVERY_ROW_UNRESOLVABLE,
    STATE_ACCEPTED,
    STATE_ACCEPTING,
    STATE_CANCELLED,
    STATE_DRAFT,
    LoggingDomain,
    PlanEpisodeSidecar,
    accepted_snapshot_path,
    claim_accept,
    ensure_episode_fence,
    read_sidecar,
    sidecar_path,
    write_sidecar,
)
from chartreux.core.plan_mode import PLAN_MARKER_FILENAME, allocate_plan_workspace
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session_types import PlanModeRecord, PlanScanDomain
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
)

TEST_SESSION_ID = "01234567-89ab-cdef-0123-deadbeef1234"
OTHER_SESSION_ID = "fedcba98-7654-3210-fedc-ba9876543210"
EPISODE_ID = "ep-flow"
EXECUTOR_SESSION_ID = "abcdef12-3456-7890-abcd-ef1234567890"
SESSION_PREFIX = "session"
SNAPSHOT_BYTES = b"# Plan\n\n## Summary\n\nDo the thing carefully.\n"
PLAN_TEXT = SNAPSHOT_BYTES.decode("utf-8")


# --- Fixtures and builders ---------------------------------------------------


def _domain(tmp_path: Path) -> LoggingDomain:
    return LoggingDomain(
        save_dir=str(tmp_path / "sessions"), session_prefix=SESSION_PREFIX
    )


def _scan_domain(tmp_path: Path) -> PlanScanDomain:
    return PlanScanDomain(
        save_dir=str(tmp_path / "sessions"), session_prefix=SESSION_PREFIX
    )


def _fields(tmp_path: Path) -> list[ConfigFieldWire]:
    return [
        ConfigFieldWire(
            name="plan",
            kind=ConfigFieldKind.COMPLEX,
            description="",
            value={
                "planner": {"model": "glm-5-3", "thinking": "high"},
                "executor": {
                    "model": "mistral-large-4",
                    "thinking": "medium",
                    "identity": "orchestrator",
                },
            },
            path="/plan",
        ),
        ConfigFieldWire(
            name="session_logging",
            kind=ConfigFieldKind.COMPLEX,
            description="",
            value={
                "enabled": True,
                "save_dir": str(tmp_path / "sessions"),
                "session_prefix": SESSION_PREFIX,
            },
            path="/session_logging",
        ),
    ]


def _episode(
    tmp_path: Path,
    *,
    state: str = STATE_DRAFT,
    holder: str | None = None,
    with_sidecar: bool = True,
    snapshot: bool = False,
    executor_session_id: str | None = None,
    draft_path: Path | None = None,
    document: str | None = None,
) -> Path:
    """One episode directory in the requested on-disk state."""
    directory = tmp_path / "tasks" / f"plan-{TEST_SESSION_ID[-12:]}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PLAN_MARKER_FILENAME).write_text(
        "chartreux plan-mode workspace\n", encoding="utf-8"
    )
    ensure_episode_fence(directory)
    if with_sidecar:
        write_sidecar(
            directory,
            PlanEpisodeSidecar(
                episode_id=EPISODE_ID,
                draft_path=draft_path or (directory / "plan.md"),
                state=state,
                claiming_session_id=holder,
                owner_domain=_domain(tmp_path) if holder is not None else None,
                scan_domain=(
                    _domain(tmp_path)
                    if state == STATE_ACCEPTING or executor_session_id
                    else None
                ),
                attempt_generation=1 if state != STATE_DRAFT else 0,
                executor_session_id=executor_session_id,
            ),
        )
    if snapshot:
        accepted_snapshot_path(directory, EPISODE_ID).write_bytes(SNAPSHOT_BYTES)
    if document is not None:
        (directory / "plan.md").write_text(document, encoding="utf-8")
    return directory


def _record(
    directory: Path,
    *,
    state: str,
    episode_id: str | None = EPISODE_ID,
    scan_domain: PlanScanDomain | None = None,
    theme: str | None = "Fix the widget",
    pre_entry_active_model: str | None = "mistral-large-4",
    pre_entry_planner_thinking: str | None = "low",
) -> PlanModeRecord:
    document = directory / "plan.md"
    return PlanModeRecord(
        plan_path=str(document),
        theme=theme,
        planner_model="glm-5-3",
        planner_thinking="high",
        pre_entry_active_model=pre_entry_active_model,
        pre_entry_planner_thinking=pre_entry_planner_thinking,
        episode_id=episode_id,
        sidecar_path=(None if episode_id is None else str(sidecar_path(directory))),
        draft_path=None if episode_id is None else str(document),
        state=None if episode_id is None else state,
        scan_domain=scan_domain,
    )


def _sidecar(directory: Path) -> PlanEpisodeSidecar:
    """The persisted sidecar, asserted present."""
    sidecar = read_sidecar(directory)
    assert sidecar is not None
    return sidecar


async def _session_with_plan(
    tmp_path: Path, build_record: Callable[[str], PlanModeRecord]
) -> tuple[str, SessionLoggingConfig]:
    """Persist a session whose record is built from the session's own id."""
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions"))
    loop = build_test_agent_loop(
        config=build_test_vibe_config(session_logging=logging), cwd=tmp_path
    )
    await loop.persist_empty_session()
    await loop.session_logger.persist_plan_mode(build_record(loop.session_id))
    session_id = loop.session_id
    await loop.aclose()
    return session_id, logging


def _resumed_app(
    logging: SessionLoggingConfig, tmp_path: Path
) -> tuple[ChartreuxApp, Any]:
    loop = build_test_agent_loop(
        config=build_test_vibe_config(session_logging=logging), cwd=tmp_path
    )
    return build_test_chartreux_app(agent_loop=loop), loop


class _Harness:
    """The app's external collaborators, patched for one plan command."""

    def __init__(
        self,
        app: ChartreuxApp,
        *,
        fields: list[ConfigFieldWire],
        confirmed: ConfirmedPlan | None = None,
    ) -> None:
        self.app = app
        self.mounted: list[Any] = []
        self.writes: list[tuple[list[Any], str]] = []
        self._patchers: list[Any] = []
        self._fields = fields
        self._confirmed = confirmed
        self.handoff = AsyncMock(return_value=None)
        self.consent = AsyncMock(return_value=confirmed)
        self.reload = AsyncMock(return_value=None)
        self.seed = AsyncMock()

    def __enter__(self) -> _Harness:
        config = self.app.app_server.resources.config
        self._patchers = [
            patch.object(
                config,
                "read_fields",
                AsyncMock(return_value=SimpleNamespace(fields=self._fields)),
            ),
            patch.object(config, "write", AsyncMock(side_effect=self._write)),
            patch.object(self.app.app_server, "plan_handoff", self.handoff),
            patch.object(
                self.app, "_mount_and_scroll", AsyncMock(side_effect=self._mount)
            ),
            patch.object(self.app, "_plan_request_consent", self.consent),
            patch.object(self.app, "_reload_config", self.reload),
            patch.object(self.app, "_handle_user_message", self.seed),
            patch.object(self.app, "_reset_message_widgets", AsyncMock()),
            patch.object(self.app, "_sync_terminal_title", MagicMock()),
            patch.object(self.app, "_refresh_context_progress", MagicMock()),
            patch.object(self.app, "_refresh_banner", MagicMock()),
            patch.object(self.app._queue, "clear_server_queue", AsyncMock()),
        ]
        for patcher in self._patchers:
            patcher.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for patcher in reversed(self._patchers):
            patcher.stop()

    async def _write(self, ops: list[Any], *, reason: str) -> SimpleNamespace:
        self.writes.append((ops, reason))
        return SimpleNamespace(rejected=False, failures=[], application="applied")

    async def _mount(self, widget: Any, **kwargs: Any) -> None:
        self.mounted.append(widget)

    def messages(self, kind: type) -> list[Any]:
        return [widget for widget in self.mounted if isinstance(widget, kind)]

    def posted(self) -> str:
        return "\n".join(str(widget._content) for widget in self.mounted)


def _tint_classes(app: ChartreuxApp) -> set[str]:
    container = app._chat_input_container
    assert container is not None
    return set(container.classes)


def _persisted_record(loop: Any) -> PlanModeRecord | None:
    session_dir = loop.session_logger.session_dir
    assert session_dir is not None
    return SessionLoader.load_metadata(session_dir).plan_mode


def _executor_marker(tmp_path: Path, episode_id: str = EPISODE_ID) -> None:
    """Record an executor session for the episode in the P1 domain.

    The shape is C3's marker: the executor's ``accepting`` record carrying the
    episode's consent digest and snapshot name — never a planning session's own
    mirror, which the executor scan does not match.
    """
    session_dir = tmp_path / "sessions" / f"{SESSION_PREFIX}_20260101_120000_abc"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "meta.json").write_text(
        json.dumps({
            "session_id": EXECUTOR_SESSION_ID,
            "plan_mode": {
                "episode_id": episode_id,
                "state": "accepting",
                "consent_sha256": "c" * 64,
                "snapshot_path": f"accepted-{episode_id}.md",
            },
        }),
        encoding="utf-8",
    )


# --- The resume surfaces -----------------------------------------------------


class TestResumeSurfaces:
    @pytest.mark.asyncio
    async def test_a_draft_episode_resumes_as_active_planning(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)
        session_id, logging = await _session_with_plan(
            tmp_path, lambda _sid: _record(directory, state=STATE_DRAFT)
        )
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)

            assert app._plan_mode is not None
            assert app._plan_mode.workspace.episode_id == EPISODE_ID
            assert PLAN_MODE_TINT_CLASS in _tint_classes(app)
            posted = harness.posted()
            assert "Plan mode resumed" in posted
            assert f"Episode: {EPISODE_ID}" in posted

    @pytest.mark.asyncio
    async def test_an_owned_accepting_episode_offers_only_recovery(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)

        def _build(session_id: str) -> PlanModeRecord:
            claim_accept(
                directory, session_id=session_id, owner_domain=_domain(tmp_path)
            )
            return _record(directory, state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)

            assert app._plan_mode is None
            assert PLAN_MODE_TINT_CLASS not in _tint_classes(app)
            assert "no consent snapshot survives" in harness.posted()
            assert "/plan accept re-offers the accept transaction" in harness.posted()
            assert "/plan cancel aborts the claim" in harness.posted()
            [(ops, reason)] = harness.writes
            assert reason == "plan mode cancel restore"
            assert [(op.op, op.value) for op in ops] == [
                ("set", "mistral-large-4"),
                ("set", "low"),
            ]

    @pytest.mark.asyncio
    async def test_an_accepting_episode_names_its_surviving_snapshot(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)

        def _build(session_id: str) -> PlanModeRecord:
            claim_accept(
                directory, session_id=session_id, owner_domain=_domain(tmp_path)
            )
            accepted_snapshot_path(directory, EPISODE_ID).write_bytes(SNAPSHOT_BYTES)
            return _record(directory, state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)

            assert app._plan_mode is None
            assert "its consent snapshot survives" in harness.posted()

    @pytest.mark.asyncio
    async def test_a_foreign_accepting_claim_refuses_the_owner_actions(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)

        def _build(_session_id: str) -> PlanModeRecord:
            claim_accept(
                directory, session_id=OTHER_SESSION_ID, owner_domain=_domain(tmp_path)
            )
            return _record(directory, state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)

            assert app._plan_mode is None
            posted = harness.posted()
            assert "held by another session" in posted
            assert "/plan accept re-offers" not in posted

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("state", "label"),
        [
            (STATE_ACCEPTED, "accepted (terminal)"),
            (STATE_CANCELLED, "cancelled (terminal)"),
        ],
    )
    async def test_a_terminal_episode_never_restores_as_active(
        self, tmp_path: Path, state: str, label: str
    ) -> None:
        directory = _episode(tmp_path, state=state)
        session_id, logging = await _session_with_plan(
            tmp_path, lambda _sid: _record(directory, state=state)
        )
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)

            assert app._plan_mode is None
            assert PLAN_MODE_TINT_CLASS not in _tint_classes(app)
            posted = harness.posted()
            assert label in posted
            assert "/plan to start a new plan" in posted
            assert "cancel with the pre-entry selections restored" in posted
            assert harness.writes[0][1] == "plan mode cancel restore"

    @pytest.mark.asyncio
    async def test_a_legacy_record_shows_the_unverified_surface(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT, with_sidecar=False)
        session_id, logging = await _session_with_plan(
            tmp_path,
            lambda _sid: _record(directory, state=STATE_DRAFT, episode_id=None),
        )
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)

            assert app._plan_mode is None
            posted = harness.posted()
            assert "legacy-unverified" in posted
            assert "accept is disabled" in posted
            assert "/plan rebind <path>" in posted

    @pytest.mark.asyncio
    async def test_a_missing_sidecar_surfaces_unresolvable(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT, with_sidecar=False)
        (tmp_path / "sessions").mkdir(exist_ok=True)
        session_id, logging = await _session_with_plan(
            tmp_path,
            lambda _sid: _record(
                directory, state=STATE_DRAFT, scan_domain=_scan_domain(tmp_path)
            ),
        )
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)

            assert app._plan_mode is None
            posted = harness.posted()
            assert "unresolvable" in posted
            assert "/plan rebind <path>" in posted
            assert "/plan to start a new plan" in posted

    @pytest.mark.asyncio
    async def test_a_possibly_executed_episode_offers_inspect_or_adjudicate(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)

        def _build(session_id: str) -> PlanModeRecord:
            claim_accept(
                directory, session_id=session_id, owner_domain=_domain(tmp_path)
            )
            accepted_snapshot_path(directory, EPISODE_ID).write_bytes(SNAPSHOT_BYTES)
            return _record(directory, state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        _executor_marker(tmp_path)
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)

            assert app._plan_mode is None
            view = app._plan_view
            assert view is not None
            assert view.row == EPISODE_ROW_POSSIBLY_EXECUTED
            posted = harness.posted()
            assert "possibly executed" in posted
            assert EXECUTOR_SESSION_ID in posted
            assert "inspect-executor" in posted
            assert "/plan mark-accepted" in posted
            # The remedy names the surviving proof path by its actual name.
            remedy = accepted_snapshot_path(directory, EPISODE_ID)
            assert f"`{remedy}`" in posted
            assert f"/plan rebind {remedy}" in posted

    @pytest.mark.asyncio
    async def test_the_executor_surface_stays_observable_and_never_rearms(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)

        def _build(session_id: str) -> PlanModeRecord:
            claim_accept(
                directory, session_id=session_id, owner_domain=_domain(tmp_path)
            )
            return _record(directory, state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        _executor_marker(tmp_path)
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)
                harness.mounted.clear()
                await app._plan_command("status")

            assert app._plan_mode is None
            posted = harness.posted()
            assert "possibly executed" in posted
            assert "inspect-executor" in posted


# --- Never-re-armed surfaces restore the pre-entry pins ----------------------


class TestNeverRearmedPinRestore:
    @pytest.mark.asyncio
    async def test_an_accepted_resume_restores_the_pre_entry_selections(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_ACCEPTED)
        session_id, logging = await _session_with_plan(
            tmp_path, lambda _sid: _record(directory, state=STATE_ACCEPTED)
        )
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)

            [(ops, reason)] = harness.writes
            assert reason == "plan mode cancel restore"
            assert [(op.op, op.path, op.value) for op in ops] == [
                ("set", "/active_model", "mistral-large-4"),
                ("set", "/thinking_overrides/glm-5-3", "low"),
            ]

    @pytest.mark.asyncio
    async def test_an_absent_pre_entry_snapshot_removes_both_keys(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_CANCELLED)
        session_id, logging = await _session_with_plan(
            tmp_path,
            lambda _sid: _record(
                directory,
                state=STATE_CANCELLED,
                pre_entry_active_model=None,
                pre_entry_planner_thinking=None,
            ),
        )
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)

            [(ops, _reason)] = harness.writes
            assert [(op.op, op.path) for op in ops] == [
                ("remove", "/active_model"),
                ("remove", "/thinking_overrides/glm-5-3"),
            ]

    @pytest.mark.asyncio
    async def test_a_legacy_resume_restores_the_pins(self, tmp_path: Path) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT, with_sidecar=False)
        session_id, logging = await _session_with_plan(
            tmp_path,
            lambda _sid: _record(directory, state=STATE_DRAFT, episode_id=None),
        )
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)

            assert [reason for _, reason in harness.writes] == [
                "plan mode cancel restore"
            ]

    @pytest.mark.asyncio
    async def test_mark_accepted_retires_the_surface_and_restores_the_pins(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)

        def _build(session_id: str) -> PlanModeRecord:
            claim_accept(
                directory, session_id=session_id, owner_domain=_domain(tmp_path)
            )
            accepted_snapshot_path(directory, EPISODE_ID).write_bytes(SNAPSHOT_BYTES)
            return _record(directory, state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        _executor_marker(tmp_path)
        app, loop = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)
                harness.mounted.clear()
                harness.writes.clear()
                await app._plan_command("mark-accepted")

            assert app._plan_mode is None
            assert _sidecar(directory).state == STATE_ACCEPTED
            [(ops, _reason)] = harness.writes
            assert [(op.op, op.value) for op in ops] == [
                ("set", "mistral-large-4"),
                ("set", "low"),
            ]
            posted = harness.posted()
            assert "accepted (terminal)" in posted
            assert "The surviving proof path is" in posted
            assert "/plan rebind" in posted

        record = _persisted_record(loop)
        assert record is not None
        assert record.state == STATE_ACCEPTED

    @pytest.mark.asyncio
    async def test_cancel_on_a_possibly_executed_episode_refuses(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)

        def _build(session_id: str) -> PlanModeRecord:
            claim_accept(
                directory, session_id=session_id, owner_domain=_domain(tmp_path)
            )
            accepted_snapshot_path(directory, EPISODE_ID).write_bytes(SNAPSHOT_BYTES)
            return _record(directory, state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        _executor_marker(tmp_path)
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)
                harness.writes.clear()
                harness.mounted.clear()
                await app._plan_command("cancel")

            # Only the design-named actions exist: inspect, or adjudicate.
            assert harness.writes == []
            [error] = harness.messages(ErrorMessage)
            assert "possibly executed" in str(error._error)
            assert "/plan mark-accepted" in str(error._error)
            assert _sidecar(directory).state == STATE_ACCEPTING

    @pytest.mark.asyncio
    async def test_cancel_on_a_terminal_episode_restores_the_pins_and_keeps_it(
        self, tmp_path: Path
    ) -> None:
        # An accepted episode is TERMINAL history: cancel-with-pin-restore does
        # not rewrite it to a weaker state (design :494-495).
        directory = _episode(tmp_path, state=STATE_ACCEPTED)
        session_id, logging = await _session_with_plan(
            tmp_path, lambda _sid: _record(directory, state=STATE_ACCEPTED)
        )
        app, loop = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)
                harness.writes.clear()
                await app._plan_command("cancel")

            assert app._plan_mode is None
            assert app._plan_record is not None
            assert app._plan_record.state == STATE_ACCEPTED
            assert [reason for _, reason in harness.writes] == [
                "plan mode cancel restore"
            ]

        record = _persisted_record(loop)
        assert record is not None
        assert record.state == STATE_ACCEPTED


# --- Rebind and the authorized roots ----------------------------------------


def _armed_episode(tmp_path: Path) -> PlanWorkspacePaths:
    """An episode-bound draft workspace with a real, exclusive claim."""
    workspace = allocate_plan_workspace(
        TEST_SESSION_ID, project_root=tmp_path, workspaces_root=tmp_path / "workspaces"
    )
    paths = PlanWorkspacePaths(
        directory=workspace.directory,
        document_path=workspace.document_path,
        episode_id=workspace.episode_id,
        sidecar_path=workspace.sidecar_path,
    )
    return paths


def _armed_state(paths: PlanWorkspacePaths) -> Any:
    return tui_plan_mode.PlanModeState(
        workspace=paths,
        planner=tui_plan_mode.PlannerSelection(model="glm-5-3", thinking="high"),
        pre_entry_active_model="mistral-large-4",
        pre_entry_planner_thinking="low",
        theme="Fix the widget",
    )


class TestRebindAndAuthorizedRoots:
    @pytest.mark.asyncio
    async def test_rebind_of_an_armed_episode_resolves_inside_the_session_cwd(
        self, tmp_path: Path
    ) -> None:
        paths = _armed_episode(tmp_path)
        target = tmp_path / "PLAN.md"
        target.write_text(PLAN_TEXT, encoding="utf-8")
        app, _ = _resumed_app(
            SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions")),
            tmp_path,
        )
        app._plan_mode = _armed_state(paths)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._plan_command(f"rebind {target}")

            assert app._plan_mode is not None
            bound = app._plan_mode.workspace.document_path
            assert bound == target.resolve()
            assert _sidecar(paths.directory).draft_path == target.resolve()
            assert harness.writes == []
            assert "re-bound" in harness.posted()

    @pytest.mark.asyncio
    async def test_rebind_refuses_a_document_outside_the_authorized_roots(
        self, tmp_path: Path
    ) -> None:
        paths = _armed_episode(tmp_path)
        outside = tmp_path.parent / "chartreux-outside-roots" / "PLAN.md"
        outside.parent.mkdir(exist_ok=True)
        outside.write_text(PLAN_TEXT, encoding="utf-8")
        app, _ = _resumed_app(
            SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions")),
            tmp_path,
        )
        app._plan_mode = _armed_state(paths)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._plan_command(f"rebind {outside}")

            assert app._plan_mode is not None
            assert app._plan_mode.workspace.document_path == paths.document_path
            assert _sidecar(paths.directory).draft_path == paths.document_path
            [error] = harness.messages(ErrorMessage)
            assert "authorized" in str(error._error)

    @pytest.mark.asyncio
    async def test_rebind_resolves_symlinks_before_the_roots_check(
        self, tmp_path: Path
    ) -> None:
        paths = _armed_episode(tmp_path)
        outside = tmp_path.parent / "chartreux-outside-roots" / "PLAN.md"
        outside.parent.mkdir(exist_ok=True)
        outside.write_text(PLAN_TEXT, encoding="utf-8")
        escaping = tmp_path / "escape.md"
        escaping.symlink_to(outside)
        inside_link = tmp_path / "inside.md"
        inside_link.symlink_to(paths.document_path)
        paths.document_path.write_text(PLAN_TEXT, encoding="utf-8")

        app, _ = _resumed_app(
            SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions")),
            tmp_path,
        )
        app._plan_mode = _armed_state(paths)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._plan_command(f"rebind {escaping}")
                [error] = harness.messages(ErrorMessage)
                assert "authorized" in str(error._error)
                assert _sidecar(paths.directory).draft_path == paths.document_path

                harness.mounted.clear()
                await app._plan_command(f"rebind {inside_link}")

            assert app._plan_mode is not None
            assert (
                app._plan_mode.workspace.document_path == paths.document_path.resolve()
            )
            assert _sidecar(paths.directory).draft_path == paths.document_path.resolve()

    @pytest.mark.asyncio
    async def test_rebind_of_a_legacy_surface_allocates_a_fresh_episode(
        self, tmp_path: Path
    ) -> None:
        legacy_dir = _episode(tmp_path, state=STATE_DRAFT, with_sidecar=False)
        target = tmp_path / "PLAN.md"
        target.write_text(PLAN_TEXT, encoding="utf-8")
        session_id, logging = await _session_with_plan(
            tmp_path,
            lambda _sid: _record(legacy_dir, state=STATE_DRAFT, episode_id=None),
        )
        app, _ = _resumed_app(logging, tmp_path)
        with patch(
            "chartreux.core.plan_mode.allocate_plan_workspace",
            MagicMock(
                side_effect=lambda session_id, *, project_root, workspaces_root=None: (
                    allocate_plan_workspace(
                        session_id,
                        project_root=project_root,
                        workspaces_root=tmp_path / "workspaces",
                    )
                )
            ),
        ):
            async with app.run_test():
                with _Harness(app, fields=_fields(tmp_path)) as harness:
                    await app._resume_local_session(session_id)
                    assert app._plan_mode is None
                    assert app._plan_record is not None
                    await app._plan_command(f"rebind {target}")

        state = app._plan_mode
        assert state is not None
        assert state.workspace.episode_id not in (None, EPISODE_ID)
        assert state.workspace.document_path == target.resolve()
        assert state.workspace.directory != legacy_dir
        assert _sidecar(state.workspace.directory).draft_path == target.resolve()
        assert "bound to" in harness.posted()

    @pytest.mark.asyncio
    async def test_accept_refuses_an_absent_bound_draft_and_names_rebind(
        self, tmp_path: Path
    ) -> None:
        paths = _armed_episode(tmp_path)
        app, _ = _resumed_app(
            SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions")),
            tmp_path,
        )
        app._plan_mode = _armed_state(paths)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._plan_accept()

            assert app._plan_mode is not None
            assert harness.consent.await_count == 0
            harness.handoff.assert_not_called()
            [error] = harness.messages(ErrorMessage)
            assert "Cannot accept the plan" in str(error._error)
            assert "/plan rebind <path>" in str(error._error)

    @pytest.mark.asyncio
    async def test_accept_after_rebind_re_previews_the_new_bytes(
        self, tmp_path: Path
    ) -> None:
        paths = _armed_episode(tmp_path)
        target = tmp_path / "PLAN.md"
        target.write_text(
            "# Bound\n\n## Summary\n\nThe re-bound document.\n", encoding="utf-8"
        )
        app, _ = _resumed_app(
            SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions")),
            tmp_path,
        )
        app._plan_mode = _armed_state(paths)
        content = target.read_bytes()
        confirmed = ConfirmedPlan(
            content=content,
            sha256=digest_bytes(content),
            summary="The re-bound document.",
        )
        async with app.run_test():
            with _Harness(
                app, fields=_fields(tmp_path), confirmed=confirmed
            ) as harness:
                await app._plan_command(f"rebind {target}")
                await app._plan_accept()

            consent_call = harness.consent.await_args
            assert consent_call is not None
            assert consent_call.args[1] == content
            harness.handoff.assert_awaited_once()
            handoff_call = harness.handoff.await_args
            assert handoff_call is not None
            kwargs = handoff_call.kwargs
            assert kwargs["snapshot_bytes"] == content
            assert kwargs["snapshot_sha256"] == digest_bytes(content)


# --- /clear during a claim ---------------------------------------------------


class TestClearDuringClaimIntegration:
    @pytest.mark.asyncio
    async def test_clear_aborts_the_claim_before_the_record_clears(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)
        order: list[str] = []
        real_abort = tui_plan_mode.abort_accept_claim

        def _abort(*args: Any, **kwargs: Any) -> str | None:
            order.append("abort")
            return real_abort(*args, **kwargs)

        def _build(session_id: str) -> PlanModeRecord:
            claim_accept(
                directory, session_id=session_id, owner_domain=_domain(tmp_path)
            )
            accepted_snapshot_path(directory, EPISODE_ID).write_bytes(SNAPSHOT_BYTES)
            return _record(directory, state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        snapshot = accepted_snapshot_path(directory, EPISODE_ID)
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            app.app_server.clear_history = AsyncMock()
            record_clear = AsyncMock(side_effect=lambda: order.append("record"))
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                with (
                    patch("chartreux.cli.textual_ui.app.abort_accept_claim", _abort),
                    patch.object(app, "_plan_clear_record", record_clear),
                ):
                    await app._resume_local_session(session_id)
                    await app._clear_history()

            assert order == ["abort", "record"]
            assert app._plan_mode is None
            assert app._plan_record is None
            assert "Plan mode ended" in harness.posted()

        sidecar = read_sidecar(directory)
        assert sidecar is not None
        assert sidecar.state == STATE_DRAFT
        assert sidecar.claiming_session_id is None
        assert sidecar.c1_verified is False
        assert not snapshot.exists()

    @pytest.mark.asyncio
    async def test_clear_refuses_when_another_session_holds_the_claim(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)

        def _build(_session_id: str) -> PlanModeRecord:
            claim_accept(
                directory, session_id=OTHER_SESSION_ID, owner_domain=_domain(tmp_path)
            )
            return _record(directory, state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            clear = AsyncMock()
            app.app_server.clear_history = clear
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)
                await app._clear_history()

            clear.assert_not_called()
            [error] = harness.messages(ErrorMessage)
            assert "Clear refused" in str(error._error)
            assert "held by another session" in str(error._error)
            assert app._plan_record is not None

        assert _sidecar(directory).state == STATE_ACCEPTING

    @pytest.mark.asyncio
    async def test_clear_refuses_a_possibly_executed_episode(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)

        def _build(session_id: str) -> PlanModeRecord:
            claim_accept(
                directory, session_id=session_id, owner_domain=_domain(tmp_path)
            )
            return _record(directory, state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        _executor_marker(tmp_path)
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            clear = AsyncMock()
            app.app_server.clear_history = clear
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)
                await app._clear_history()

            clear.assert_not_called()
            [error] = harness.messages(ErrorMessage)
            assert "an executor session records plan episode" in str(error._error)
            assert "mark the episode accepted first" in str(error._error)

    @pytest.mark.asyncio
    async def test_clear_removes_the_record_after_a_real_abort(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)

        def _build(session_id: str) -> PlanModeRecord:
            claim_accept(
                directory, session_id=session_id, owner_domain=_domain(tmp_path)
            )
            accepted_snapshot_path(directory, EPISODE_ID).write_bytes(SNAPSHOT_BYTES)
            return _record(directory, state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        app, loop = _resumed_app(logging, tmp_path)
        async with app.run_test():
            app.app_server.clear_history = AsyncMock()
            with _Harness(app, fields=_fields(tmp_path)):
                await app._resume_local_session(session_id)
                await app._clear_history()

        assert _sidecar(directory).state == STATE_DRAFT
        assert _persisted_record(loop) is None


# --- The status surface ------------------------------------------------------

_MIRRORED_ROWS = {
    "EPISODE_ROW_DRAFT": (EPISODE_ROW_DRAFT, RECOVERY_ROW_DRAFT),
    "EPISODE_ROW_ACCEPTING_NO_SNAPSHOT": (
        EPISODE_ROW_ACCEPTING_NO_SNAPSHOT,
        RECOVERY_ROW_ACCEPTING_NO_SNAPSHOT,
    ),
    "EPISODE_ROW_ACCEPTING_SNAPSHOT": (
        EPISODE_ROW_ACCEPTING_SNAPSHOT,
        RECOVERY_ROW_ACCEPTING_SNAPSHOT,
    ),
    "EPISODE_ROW_ACCEPTED": (EPISODE_ROW_ACCEPTED, RECOVERY_ROW_ACCEPTED),
    "EPISODE_ROW_CANCELLED": (EPISODE_ROW_CANCELLED, RECOVERY_ROW_CANCELLED),
    "EPISODE_ROW_EXECUTED": (EPISODE_ROW_EXECUTED, RECOVERY_ROW_EXECUTED),
    "EPISODE_ROW_POSSIBLY_EXECUTED": (
        EPISODE_ROW_POSSIBLY_EXECUTED,
        RECOVERY_ROW_POSSIBLY_EXECUTED,
    ),
    "EPISODE_ROW_UNRESOLVABLE": (EPISODE_ROW_UNRESOLVABLE, RECOVERY_ROW_UNRESOLVABLE),
    "EPISODE_ROW_LEGACY": (EPISODE_ROW_LEGACY, RECOVERY_ROW_LEGACY),
}


class TestEpisodeStatusLines:
    def test_the_tui_rows_mirror_the_core_recovery_rows(self) -> None:
        for name, (mirror, core) in _MIRRORED_ROWS.items():
            assert mirror == core, name

    def test_every_row_names_its_design_actions(self) -> None:
        expected = {
            EPISODE_ROW_DRAFT: ("continue planning", "/plan accept to execute"),
            EPISODE_ROW_ACCEPTING_SNAPSHOT: (
                "/plan accept re-offers the accept transaction",
                "/plan cancel aborts the claim",
            ),
            EPISODE_ROW_ACCEPTED: (
                "/plan to start a new plan",
                "cancel with the pre-entry selections restored",
            ),
            EPISODE_ROW_CANCELLED: (
                "/plan to start a new plan",
                "cancel with the pre-entry selections restored",
            ),
            EPISODE_ROW_LEGACY: (
                "/plan rebind <path>",
                "cancel with the pre-entry selections restored",
            ),
            EPISODE_ROW_UNRESOLVABLE: (
                "/plan rebind <path>",
                "/plan to start a new plan",
            ),
            EPISODE_ROW_EXECUTED: (
                "/plan to start a new plan",
                "cancel with the pre-entry selections restored",
            ),
            EPISODE_ROW_POSSIBLY_EXECUTED: ("inspect-executor", "/plan mark-accepted"),
        }
        for row, fragments in expected.items():
            # The owner's view is the one that offers the recovery actions.
            joined = " ".join(episode_actions(EpisodeView(row=row, owner=True)))
            for fragment in fragments:
                assert fragment in joined, (row, fragment)

    def test_the_accepting_actions_are_owner_scoped(self) -> None:
        owner = " ".join(
            episode_actions(
                EpisodeView(row=EPISODE_ROW_ACCEPTING_NO_SNAPSHOT, owner=True)
            )
        )
        foreign = " ".join(
            episode_actions(
                EpisodeView(row=EPISODE_ROW_ACCEPTING_NO_SNAPSHOT, owner=False)
            )
        )
        assert "owner session only" in owner
        assert "held by another session" in foreign

    def test_the_mark_accepted_action_states_its_foreclosure(self) -> None:
        joined = " ".join(
            episode_actions(EpisodeView(row=EPISODE_ROW_POSSIBLY_EXECUTED))
        )
        assert "handoff is foreclosed" in joined
        assert "re-execution of the same episode is refused" in joined
        assert "fresh episode re-binding the surviving proof path" in joined

    def test_the_status_lines_carry_the_state_path_and_actions(self) -> None:
        lines = episode_status_lines(
            EpisodeView(row=EPISODE_ROW_ACCEPTED, episode_id=EPISODE_ID),
            plan_path="/plans/plan.md",
        )
        assert f"Plan episode {EPISODE_ID}: accepted (terminal)" in lines
        assert "Plan document: /plans/plan.md." in lines
        assert "Actions: " in lines

    def test_the_possibly_executed_lines_name_the_executor_and_the_remedy(self) -> None:
        lines = episode_status_lines(
            EpisodeView(
                row=EPISODE_ROW_POSSIBLY_EXECUTED,
                episode_id=EPISODE_ID,
                executor_session_id=EXECUTOR_SESSION_ID,
            ),
            remedy_path="/plans/accepted-ep-flow.md",
        )
        assert EXECUTOR_SESSION_ID in lines
        assert "/plans/accepted-ep-flow.md" in lines
        assert "/plan rebind /plans/accepted-ep-flow.md" in lines

    @pytest.mark.asyncio
    async def test_status_reports_a_terminal_episode_without_arming(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_ACCEPTED)
        session_id, logging = await _session_with_plan(
            tmp_path, lambda _sid: _record(directory, state=STATE_ACCEPTED)
        )
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)
                harness.mounted.clear()
                await app._plan_command("status")

            assert app._plan_mode is None
            posted = harness.posted()
            assert "accepted (terminal)" in posted
            assert "/plan to start a new plan" in posted
            assert PLAN_MODE_TINT_CLASS not in _tint_classes(app)

    def test_the_fence_survives_a_terminal_cancel(self, tmp_path: Path) -> None:
        directory = _episode(tmp_path, state=STATE_ACCEPTED)
        removed = tui_plan_mode.release_workspace(
            PlanWorkspacePaths(
                directory=directory,
                document_path=directory / "plan.md",
                episode_id=EPISODE_ID,
                sidecar_path=sidecar_path(directory),
            )
        )
        assert removed is False
        assert [entry.name for entry in directory.iterdir()] == [EPISODE_FENCE_FILENAME]


# --- The terminal cancel surface (design :477, :494-500) ---------------------


class TestTerminalCancelSurface:
    @pytest.mark.asyncio
    async def test_a_draft_cancel_writes_the_terminal_sidecar_and_keeps_the_record(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)
        session_id, logging = await _session_with_plan(
            tmp_path, lambda _sid: _record(directory, state=STATE_DRAFT)
        )
        states_at_cleanup: list[str] = []
        real_release = tui_plan_mode.release_workspace

        def _release(workspace: PlanWorkspacePaths) -> bool:
            sidecar = read_sidecar(workspace.directory)
            states_at_cleanup.append("gone" if sidecar is None else sidecar.state)
            return real_release(workspace)

        app, loop = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                with patch("chartreux.cli.textual_ui.app.release_workspace", _release):
                    await app._resume_local_session(session_id)
                    harness.writes.clear()
                    await app._plan_command("cancel")

                assert app._plan_mode is None
                assert PLAN_MODE_TINT_CLASS not in _tint_classes(app)
                assert app._plan_record is not None
                assert app._plan_record.state == STATE_CANCELLED
                [(ops, reason)] = harness.writes
                assert reason == "plan mode cancel restore"
                assert [(op.op, op.value) for op in ops] == [
                    ("set", "mistral-large-4"),
                    ("set", "low"),
                ]

        # The durable transition ran BEFORE the cleanup whitelist, which then
        # left the retained fence as the directory's only entry.
        assert states_at_cleanup == [STATE_CANCELLED]
        assert [entry.name for entry in directory.iterdir()] == [EPISODE_FENCE_FILENAME]
        record = _persisted_record(loop)
        assert record is not None
        assert record.state == STATE_CANCELLED

    @pytest.mark.asyncio
    async def test_a_record_write_failure_keeps_the_terminal_sidecar(
        self, tmp_path: Path
    ) -> None:
        # design :494-495: when the cancelled record cannot be written, the
        # sidecar is the LAST terminal authority — so the cleanup whitelist
        # (which removes it) must not run.
        directory = _episode(tmp_path, state=STATE_DRAFT)
        session_id, logging = await _session_with_plan(
            tmp_path, lambda _sid: _record(directory, state=STATE_DRAFT)
        )
        app, loop = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)
                harness.writes.clear()
                with patch.object(
                    app,
                    "_plan_write_record",
                    AsyncMock(side_effect=RuntimeError("record write refused")),
                ):
                    await app._plan_command("cancel")

                # The pins are restored whatever the record write did.
                [(ops, reason)] = harness.writes
                assert reason == "plan mode cancel restore"
                assert [(op.op, op.value) for op in ops] == [
                    ("set", "mistral-large-4"),
                    ("set", "low"),
                ]
                [error] = harness.messages(ErrorMessage)
                assert "stored record could not be updated" in str(error._error)

        assert app._plan_mode is None
        assert app._plan_record is not None
        assert app._plan_record.state == STATE_CANCELLED
        # The terminal arbiter state survived: the sidecar still reads cancelled
        # and the directory still holds the plan-owned entries.
        assert _sidecar(directory).state == STATE_CANCELLED
        assert (directory / PLAN_MARKER_FILENAME).exists()
        # The record is honestly unchanged, which the error above reports.
        persisted = _persisted_record(loop)
        assert persisted is not None
        assert persisted.state == STATE_DRAFT

    @pytest.mark.asyncio
    async def test_a_cancelled_record_with_a_reclaimed_sidecar_still_shows_history(
        self, tmp_path: Path
    ) -> None:
        # design :493-495 — a terminal record stays display authority after the
        # cleanup reclaims the sidecar; it never re-arms.
        directory = _episode(tmp_path, state=STATE_DRAFT, with_sidecar=False)
        session_id, logging = await _session_with_plan(
            tmp_path, lambda _sid: _record(directory, state=STATE_CANCELLED)
        )
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)

            assert app._plan_mode is None
            assert app._plan_record is not None
            posted = harness.posted()
            assert "cancelled (terminal)" in posted
            assert "/plan to start a new plan" in posted
            assert "cancel with the pre-entry selections restored" in posted
            [(_, reason)] = harness.writes
            assert reason == "plan mode cancel restore"

    @pytest.mark.asyncio
    async def test_clear_outside_the_accept_flow_cancels_the_episode(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)
        session_id, logging = await _session_with_plan(
            tmp_path, lambda _sid: _record(directory, state=STATE_DRAFT)
        )
        app, loop = _resumed_app(logging, tmp_path)
        async with app.run_test():
            app.app_server.clear_history = AsyncMock()
            with _Harness(app, fields=_fields(tmp_path)):
                await app._resume_local_session(session_id)
                await app._clear_history()

        # /clear runs no cleanup whitelist (design :532), so the terminal
        # sidecar survives as the episode's durable cancelled record.
        assert _sidecar(directory).state == STATE_CANCELLED
        assert _persisted_record(loop) is None

    @pytest.mark.asyncio
    async def test_a_foreign_session_cannot_cancel_the_accepting_episode(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)

        def _build(_session_id: str) -> PlanModeRecord:
            claim_accept(
                directory, session_id=OTHER_SESSION_ID, owner_domain=_domain(tmp_path)
            )
            return _record(directory, state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        app, loop = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path)) as harness:
                await app._resume_local_session(session_id)
                harness.mounted.clear()
                harness.writes.clear()
                await app._plan_command("cancel")

            [error] = harness.messages(ErrorMessage)
            assert "held by another session" in str(error._error)
            assert harness.writes == []
            assert app._plan_record is not None

        sidecar = _sidecar(directory)
        assert sidecar.state == STATE_ACCEPTING
        assert sidecar.claiming_session_id == OTHER_SESSION_ID
        record = _persisted_record(loop)
        assert record is not None
        assert record.state == STATE_ACCEPTING


# --- The recovery re-offer continuation (design :449, :246-249) --------------


def _claimed_episode(
    tmp_path: Path, session_id: str, *, snapshot_name: str | None = None
) -> Path:
    """A claimed episode whose sidecar records the snapshot's consent digest."""
    directory = _episode(tmp_path, state=STATE_DRAFT)
    sidecar = claim_accept(
        directory, session_id=session_id, owner_domain=_domain(tmp_path)
    )
    write_sidecar(
        directory, replace(sidecar, consent_sha256=digest_bytes(SNAPSHOT_BYTES))
    )
    name = snapshot_name or accepted_snapshot_path(directory, EPISODE_ID).name
    (directory / name).write_bytes(SNAPSHOT_BYTES)
    return directory


def _snapshot_plan() -> ConfirmedPlan:
    return ConfirmedPlan(
        content=SNAPSHOT_BYTES,
        sha256=digest_bytes(SNAPSHOT_BYTES),
        summary="Do the thing carefully.",
    )


class TestRecoveryReOfferContinuation:
    @pytest.mark.asyncio
    async def test_the_re_offer_confirms_the_snapshot_not_the_changed_draft(
        self, tmp_path: Path
    ) -> None:
        built: dict[str, Path] = {}

        def _build(session_id: str) -> PlanModeRecord:
            built["directory"] = _claimed_episode(tmp_path, session_id)
            return _record(built["directory"], state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        directory = built["directory"]
        (directory / "plan.md").write_text(
            "# Plan\n\n## Summary\n\nA completely different plan.\n", encoding="utf-8"
        )
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(
                app, fields=_fields(tmp_path), confirmed=_snapshot_plan()
            ) as harness:
                await app._resume_local_session(session_id)
                assert app._plan_mode is None
                await app._plan_accept()

            consent_call = harness.consent.await_args
            assert consent_call is not None
            assert consent_call.args[1] == SNAPSHOT_BYTES
            assert consent_call.kwargs["snapshot_consent"] is True
            harness.handoff.assert_awaited_once()
            handoff_call = harness.handoff.await_args
            assert handoff_call is not None
            kwargs = handoff_call.kwargs
            assert kwargs["snapshot_bytes"] == SNAPSHOT_BYTES
            assert kwargs["snapshot_sha256"] == digest_bytes(SNAPSHOT_BYTES)
            assert kwargs["snapshot_path"] == str(
                accepted_snapshot_path(directory, EPISODE_ID)
            )

        # The claim STAYS held: the server re-enters COMMIT on this attempt
        # instead of preparing a new one from the draft.
        sidecar = _sidecar(directory)
        assert sidecar.state == STATE_ACCEPTING
        assert sidecar.claiming_session_id == session_id

    @pytest.mark.asyncio
    async def test_the_re_offer_survives_a_deleted_draft(self, tmp_path: Path) -> None:
        built: dict[str, Path] = {}

        def _build(session_id: str) -> PlanModeRecord:
            built["directory"] = _claimed_episode(tmp_path, session_id)
            return _record(built["directory"], state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        directory = built["directory"]
        draft = directory / "plan.md"
        draft.write_bytes(SNAPSHOT_BYTES)
        draft.unlink()
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(
                app, fields=_fields(tmp_path), confirmed=_snapshot_plan()
            ) as harness:
                await app._resume_local_session(session_id)
                await app._plan_accept()

            handoff_call = harness.handoff.await_args
            assert handoff_call is not None
            kwargs = handoff_call.kwargs
            assert kwargs["snapshot_bytes"] == SNAPSHOT_BYTES

        assert _sidecar(directory).state == STATE_ACCEPTING

    @pytest.mark.asyncio
    async def test_the_re_offer_carries_the_surviving_provenance_name(
        self, tmp_path: Path
    ) -> None:
        name = f"provenance-{EPISODE_ID}-1.md"
        built: dict[str, Path] = {}

        def _build(session_id: str) -> PlanModeRecord:
            built["directory"] = _claimed_episode(
                tmp_path, session_id, snapshot_name=name
            )
            return _record(built["directory"], state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        directory = built["directory"]
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(
                app, fields=_fields(tmp_path), confirmed=_snapshot_plan()
            ) as harness:
                await app._resume_local_session(session_id)
                await app._plan_accept()

            handoff_call = harness.handoff.await_args
            assert handoff_call is not None
            kwargs = handoff_call.kwargs
            # The either-name locator supplies the locator: the live name is
            # absent, so the attempt's provenance name travels instead.
            assert kwargs["snapshot_path"] == str(directory / name)
            assert kwargs["snapshot_bytes"] == SNAPSHOT_BYTES

        assert not accepted_snapshot_path(directory, EPISODE_ID).exists()

    @pytest.mark.asyncio
    async def test_an_abort_at_the_re_offer_releases_the_claim(
        self, tmp_path: Path
    ) -> None:
        built: dict[str, Path] = {}

        def _build(session_id: str) -> PlanModeRecord:
            built["directory"] = _claimed_episode(tmp_path, session_id)
            return _record(built["directory"], state=STATE_ACCEPTING)

        session_id, logging = await _session_with_plan(tmp_path, _build)
        directory = built["directory"]
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(app, fields=_fields(tmp_path), confirmed=None) as harness:
                await app._resume_local_session(session_id)
                await app._plan_accept()

            harness.handoff.assert_not_called()
            assert "The accept claim was released" in harness.posted()

        sidecar = _sidecar(directory)
        assert sidecar.state == STATE_DRAFT
        assert sidecar.claiming_session_id is None
        assert not accepted_snapshot_path(directory, EPISODE_ID).exists()


# --- A rebound record resumes against the episode's sidecar directory --------


class TestReboundResumeDirectory:
    @pytest.mark.asyncio
    async def test_a_rebound_record_resumes_against_the_sidecar_directory(
        self, tmp_path: Path
    ) -> None:
        directory = _episode(tmp_path, state=STATE_DRAFT)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        rebound = elsewhere / "PLAN.md"
        rebound.write_text(PLAN_TEXT, encoding="utf-8")
        write_sidecar(directory, replace(_sidecar(directory), draft_path=rebound))

        def _build(_session_id: str) -> PlanModeRecord:
            return _record(directory, state=STATE_DRAFT).model_copy(
                update={"draft_path": str(rebound)}
            )

        session_id, logging = await _session_with_plan(tmp_path, _build)
        content = rebound.read_bytes()
        confirmed = ConfirmedPlan(
            content=content,
            sha256=digest_bytes(content),
            summary="Do the thing carefully.",
        )
        app, _ = _resumed_app(logging, tmp_path)
        async with app.run_test():
            with _Harness(
                app, fields=_fields(tmp_path), confirmed=confirmed
            ) as harness:
                await app._resume_local_session(session_id)
                state = app._plan_mode
                assert state is not None
                # The document moved with the re-bind; the episode did not.
                assert state.workspace.document_path == rebound
                assert state.workspace.directory == directory
                await app._plan_accept()

            handoff_call = harness.handoff.await_args
            assert handoff_call is not None
            kwargs = handoff_call.kwargs
            assert kwargs["plan_path"] == str(rebound)
            assert kwargs["snapshot_path"] == str(
                accepted_snapshot_path(directory, EPISODE_ID)
            )
