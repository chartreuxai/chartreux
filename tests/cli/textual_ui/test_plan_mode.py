"""The /plan TUI surface: commands, the consent modal, and the handoff wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from textual.app import App, ComposeResult
from textual.color import Color
from textual.widgets import Button

from chartreux.app_server.protocol import (
    AppServerResponseError,
    ConfigFieldKind,
    ConfigFieldWire,
    ConfigLayerValueWire,
    ProtocolError,
    ProtocolErrorCode,
)
from chartreux.cli.commands import CommandRegistry
from chartreux.cli.textual_ui.app import ChartreuxApp, StartupOptions
from chartreux.cli.textual_ui.plan_mode import (
    PLAN_MODE_LABEL,
    PLAN_MODE_TINT_CLASS,
    ConfirmedPlan,
    ExecutorSelection,
    PlanDocumentBrief,
    PlanModeState,
    PlannerSelection,
    PlanWorkspacePaths,
    allocate_workspace,
    clear_ops,
    digest_bytes,
    entry_ops,
    handoff_failed_before_replacement,
    plan_section,
    planner_seed,
    pre_entry_snapshot,
    read_document,
    record_from_state,
    restore_ops,
    resume_notice,
    state_from_record,
    thinking_path,
)
from chartreux.cli.textual_ui.screens.plan_consent import (
    CANCEL_BUTTON_ID,
    START_BUTTON_ID,
    PlanConsentScreen,
)
from chartreux.cli.textual_ui.widgets.messages import (
    ErrorMessage,
    UserCommandMessage,
    WarningMessage,
)
from chartreux.cli.textual_ui.widgets.session_picker import SessionPickerApp
from chartreux.config_values import ThinkingLevel
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.models import PlanSectionConfig
from chartreux.core.plan_episode import EPISODE_FENCE_FILENAME, STATE_CANCELLED
from chartreux.core.plan_mode import (
    PLAN_MARKER_FILENAME,
    PlanWorkspaceError,
    allocate_plan_workspace,
)
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session_types import PlanModeRecord
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
)

_PLAN_TEXT = """# Plan

## Summary

Do the thing carefully.

## Steps

1. Read the code.
2. Change it.
"""


def _fields(
    *,
    plan: dict[str, Any] | None = None,
    active_model: tuple[str, str] | None = None,
    thinking_overrides: dict[str, Any] | None = None,
) -> list[ConfigFieldWire]:
    fields = [
        ConfigFieldWire(
            name="plan",
            kind=ConfigFieldKind.COMPLEX,
            description="",
            value=(
                plan
                if plan is not None
                else PlanSectionConfig().model_dump(mode="json")
            ),
            path="/plan",
        ),
        ConfigFieldWire(
            name="session_logging",
            kind=ConfigFieldKind.COMPLEX,
            description="",
            value={
                "enabled": True,
                "save_dir": "/tmp/chartreux-plan-test-sessions",
                "session_prefix": "session",
            },
            path="/session_logging",
        ),
    ]
    if active_model is not None:
        value, layer = active_model
        fields.append(
            ConfigFieldWire(
                name="active_model",
                kind=ConfigFieldKind.STR,
                description="",
                value=value,
                path="/active_model",
                layer_values=[ConfigLayerValueWire(layer=layer, value=value)],
            )
        )
    if thinking_overrides is not None:
        fields.append(
            ConfigFieldWire(
                name="thinking_overrides",
                kind=ConfigFieldKind.COMPLEX,
                description="",
                value=thinking_overrides,
                path="/thinking_overrides",
                layer_values=[
                    ConfigLayerValueWire(layer="overrides", value=thinking_overrides)
                ],
            )
        )
    return fields


def _workspace(tmp_path: Path) -> PlanWorkspacePaths:
    directory = tmp_path / "plan-test-session"
    directory.mkdir()
    return PlanWorkspacePaths(directory=directory, document_path=directory / "plan.md")


def _marked_workspace(tmp_path: Path) -> PlanWorkspacePaths:
    """A workspace as the allocator leaves it: the directory and its marker."""
    workspace = _workspace(tmp_path)
    (workspace.directory / PLAN_MARKER_FILENAME).write_text(
        "chartreux plan-mode workspace\n", encoding="utf-8"
    )
    return workspace


_EPISODE_SESSION_ID = "01234567-89ab-cdef-0123-deadbeef1234"


def _episode_workspace(
    tmp_path: Path, *, text: str | None = None
) -> PlanWorkspacePaths:
    """An episode-bound workspace: the allocator's exclusive claim, on disk.

    Its sidecar makes the episode observable, so accept binds the claim rather
    than a legacy path.
    """
    allocated = allocate_plan_workspace(
        _EPISODE_SESSION_ID,
        project_root=tmp_path,
        workspaces_root=tmp_path / "workspaces",
    )
    workspace = PlanWorkspacePaths(
        directory=allocated.directory,
        document_path=allocated.document_path,
        episode_id=allocated.episode_id,
        sidecar_path=allocated.sidecar_path,
    )
    if text is not None:
        workspace.document_path.write_text(text, encoding="utf-8")
    return workspace


def _write_plan(workspace: PlanWorkspacePaths, text: str = _PLAN_TEXT) -> None:
    workspace.document_path.write_text(text, encoding="utf-8")


def _confirmed(workspace: PlanWorkspacePaths) -> ConfirmedPlan:
    """The consent result for the document as it stands on disk."""
    content = workspace.document_path.read_bytes()
    return ConfirmedPlan(
        content=content, sha256=digest_bytes(content), summary="Do the thing carefully."
    )


def _state(
    workspace: PlanWorkspacePaths,
    *,
    planner_model: str = "glm-5-3",
    planner_thinking: ThinkingLevel = "high",
    pre_entry_active_model: str | None = None,
    pre_entry_planner_thinking: str | None = None,
    theme: str | None = None,
) -> PlanModeState:
    return PlanModeState(
        workspace=workspace,
        planner=PlannerSelection(model=planner_model, thinking=planner_thinking),
        pre_entry_active_model=pre_entry_active_model,
        pre_entry_planner_thinking=pre_entry_planner_thinking,
        theme=theme,
    )


def _applied() -> SimpleNamespace:
    return SimpleNamespace(rejected=False, failures=[], application="applied")


class _Harness:
    """The app's external collaborators, patched for one /plan command."""

    def __init__(
        self,
        app: ChartreuxApp,
        *,
        fields: list[ConfigFieldWire],
        workspace: PlanWorkspacePaths,
        real_reset: bool = False,
    ) -> None:
        self.app = app
        self._real_reset = real_reset
        self.mounted: list[Any] = []
        self.writes: list[tuple[list[Any], str]] = []
        self._patchers: list[Any] = []
        self.handoff = AsyncMock(return_value=None)
        self.seed = AsyncMock()
        self.consent = AsyncMock(
            return_value=ConfirmedPlan(content=b"", sha256="0" * 64, summary="")
        )
        self.reload = AsyncMock(return_value=None)
        self._fields = fields
        self._workspace = workspace

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
            patch(
                "chartreux.cli.textual_ui.app.allocate_workspace",
                MagicMock(return_value=(self._workspace, None)),
            ),
            patch.object(self.app, "_reset_message_widgets", AsyncMock()),
            patch.object(self.app, "_sync_terminal_title", MagicMock()),
            patch.object(self.app, "_refresh_context_progress", MagicMock()),
            patch.object(self.app, "_refresh_banner", MagicMock()),
            patch.object(self.app._queue, "clear_server_queue", AsyncMock()),
        ]
        if self._real_reset:
            self._patchers = [
                patcher
                for patcher in self._patchers
                if patcher.attribute != "_reset_message_widgets"
            ]
        for patcher in self._patchers:
            patcher.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for patcher in reversed(self._patchers):
            patcher.stop()

    async def _write(self, ops: list[Any], *, reason: str) -> SimpleNamespace:
        self.writes.append((ops, reason))
        return _applied()

    async def _mount(self, widget: Any, **kwargs: Any) -> None:
        self.mounted.append(widget)

    def messages(self, kind: type) -> list[Any]:
        return [widget for widget in self.mounted if isinstance(widget, kind)]


@pytest.fixture
def app() -> ChartreuxApp:
    return build_test_chartreux_app()


def _contents(widgets: list[Any]) -> str:
    return "\n".join(str(widget._content) for widget in widgets)


# --- Command surface ---------------------------------------------------------


class TestCommandRegistration:
    def test_plan_command_is_registered(self) -> None:
        registry = CommandRegistry()
        assert registry.get_command_name("/plan") == "plan"
        result = registry.parse_command("/plan accept")
        assert result is not None
        name, command, args = result
        assert name == "plan"
        assert command.handler == "_plan_command"
        assert command.side_channel is True
        assert args == "accept"
        assert "`/plan`" in registry.get_help_text()

    def test_bare_plan_passes_no_arguments(self) -> None:
        registry = CommandRegistry()
        result = registry.parse_command("/plan")
        assert result == ("plan", registry.commands["plan"], "")


# --- Entry -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_entry_writes_planner_pins_and_posts_the_seed(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._plan_command("")

        assert app._plan_mode is not None
        assert app._plan_mode.plan_path == str(workspace.document_path)
        [(ops, reason)] = harness.writes
        assert reason == "plan mode entry"
        assert [(op.op, op.path, op.value, op.target_layer) for op in ops] == [
            ("set", "/active_model", "glm-5-3", "overrides"),
            ("set", "/thinking_overrides/glm-5-3", "high", "overrides"),
        ]
        harness.reload.assert_awaited_once_with()
        harness.seed.assert_awaited_once_with(planner_seed(workspace))


@pytest.mark.asyncio
async def test_plan_entry_during_active_turn_queues_seed(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    prepared = MagicMock()
    enqueue = AsyncMock()
    async with app.run_test():
        with (
            _Harness(app, fields=_fields(), workspace=workspace) as harness,
            patch.object(app, "_is_busy", return_value=True),
            patch.object(
                app, "_prepare_prompt_or_abort", AsyncMock(return_value=prepared)
            ),
            patch.object(app._queue, "enqueue_prompt", enqueue),
        ):
            await app._dispatch_submitted_value("/plan")
            task = app._side_channel._task
            assert task is not None
            await task
            assert app._plan_mode is not None
            harness.seed.assert_not_called()
            enqueue.assert_awaited_once_with(
                planner_seed(workspace),
                skill_name=None,
                prepared_prompt=prepared,
                optimistic_start=False,
            )


@pytest.mark.asyncio
async def test_plan_entry_reload_failure_reverts_and_reports_error(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            harness.reload.return_value = "reload refused"
            await app._plan_command("")

        harness.reload.assert_awaited_once_with()
        assert [reason for _, reason in harness.writes] == [
            "plan mode entry",
            "plan mode entry revert",
        ]
        assert app._plan_mode is None
        [error] = harness.messages(ErrorMessage)
        assert "runtime reload failed (reload refused)" in str(error._error)
        assert "restored" in str(error._error)
        assert not harness.messages(UserCommandMessage)


@pytest.mark.asyncio
async def test_plan_entry_reports_an_already_active_mode(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    app._plan_mode = _state(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._plan_command("")

        assert harness.writes == []
        posted = harness.messages(UserCommandMessage)
        assert len(posted) == 1
        assert "already active" in posted[0]._content
        assert str(workspace.document_path) in posted[0]._content


@pytest.mark.asyncio
async def test_plan_entry_refuses_when_the_model_is_admin_enforced(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    fields = _fields(active_model=("admin-model", "admin"))
    async with app.run_test():
        with _Harness(app, fields=fields, workspace=workspace) as harness:
            await app._plan_command("")

        assert app._plan_mode is None
        assert harness.writes == []
        [error] = harness.messages(ErrorMessage)
        assert "enforced by your administrator" in str(error._error)


@pytest.mark.asyncio
async def test_plan_entry_failure_reverts_and_reports_the_outcome(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            config = app.app_server.resources.config
            with patch.object(
                config,
                "write",
                AsyncMock(side_effect=[RuntimeError("write refused"), _applied()]),
            ):
                await app._plan_command("")

        assert app._plan_mode is None
        [error] = harness.messages(ErrorMessage)
        message = str(error._error)
        assert "Failed to enter plan mode: write refused" in message
        assert "restored" in message


@pytest.mark.asyncio
async def test_plan_entry_failure_names_a_possible_leftover_selection(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    failed = SimpleNamespace(
        rejected=True, failures=["patch failed"], application="failed"
    )
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            config = app.app_server.resources.config
            with patch.object(config, "write", AsyncMock(return_value=failed)):
                await app._plan_command("")

        assert app._plan_mode is None
        [error] = harness.messages(ErrorMessage)
        assert "may remain active" in str(error._error)
        assert "/thinking" in str(error._error)


@pytest.mark.asyncio
async def test_plan_entry_reports_a_workspace_failure(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            with patch(
                "chartreux.cli.textual_ui.app.allocate_workspace",
                MagicMock(return_value=(None, "no free plan workspace name")),
            ):
                await app._plan_command("")

        assert app._plan_mode is None
        assert harness.writes == []
        [error] = harness.messages(ErrorMessage)
        assert "Failed to create the plan workspace" in str(error._error)
        assert "no free plan workspace name" in str(error._error)


# --- Status ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_status_reports_the_mode_theme_and_document(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    app._plan_mode = _state(workspace, theme="Fix the widget")
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._plan_command("  Status  ")

        # A status report is read-only: no pins, no seed, no record write.
        assert harness.writes == []
        harness.seed.assert_not_called()
        [posted] = harness.messages(UserCommandMessage)
        lines = posted._content.splitlines()
        assert lines[0] == (
            "Plan mode is active. "
            f"Plan document: {workspace.document_path}. Theme: Fix the widget."
        )
        # The episode surface is reported too: a legacy record never arms accept.
        assert "Plan episode (unidentified): legacy-unverified" in posted._content
        assert "/plan rebind <path>" in posted._content


@pytest.mark.asyncio
async def test_plan_status_omits_the_theme_clause_when_themeless(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    app._plan_mode = _state(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._plan_command("status")

        [posted] = harness.messages(UserCommandMessage)
        assert posted._content.splitlines()[0] == (
            f"Plan mode is active. Plan document: {workspace.document_path}."
        )
        assert "Theme:" not in posted._content
        assert "None" not in posted._content


@pytest.mark.asyncio
async def test_plan_status_outside_plan_mode_reports_not_in_plan_mode(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=_workspace(tmp_path)) as harness:
            await app._plan_command("status")

        assert harness.writes == []
        assert not harness.messages(UserCommandMessage)
        [error] = harness.messages(ErrorMessage)
        assert "Not in plan mode" in str(error._error)
        assert app._plan_mode is None


# --- The standing indicator --------------------------------------------------


def _input_box(app: ChartreuxApp) -> Any:
    container = app._chat_input_container
    assert container is not None
    return container.get_widget_by_id("input-box")


@pytest.mark.asyncio
async def test_plan_entry_applies_the_amber_tint_and_the_label(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    async with app.run_test() as pilot:
        with _Harness(app, fields=_fields(), workspace=workspace):
            await app._plan_command("")
        await pilot.pause()

        assert PLAN_MODE_TINT_CLASS in _tint_classes(app)
        input_box = _input_box(app)
        assert input_box.border_title == f" {PLAN_MODE_LABEL} "
        amber = Color.parse(app.theme_variables["warning"])
        assert input_box.styles.border_top == ("solid", amber)
        assert input_box.styles.border_bottom == ("solid", amber)


@pytest.mark.asyncio
async def test_plan_cancel_clears_the_tint_and_the_label(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    app._plan_mode = _state(workspace)
    async with app.run_test() as pilot:
        baseline = _input_box(app).styles.border_top
        app._set_plan_tint(True)
        await pilot.pause()
        assert _input_box(app).border_title == f" {PLAN_MODE_LABEL} "
        assert _input_box(app).styles.border_top != baseline

        with _Harness(app, fields=_fields(), workspace=workspace):
            await app._plan_command("cancel")
        await pilot.pause()

        assert PLAN_MODE_TINT_CLASS not in _tint_classes(app)
        input_box = _input_box(app)
        assert input_box.border_title == ""
        assert input_box.styles.border_top == baseline


@pytest.mark.asyncio
async def test_resumed_plan_mode_shows_the_styled_tint_and_the_label(
    tmp_path: Path,
) -> None:
    # The resume-restored session carries the same standing indicator as an
    # entered one: 01D restores the state, 01B-3 supplies the styling.
    session_id, logging, plan_directory = await _planning_session(tmp_path)
    app, _ = _resumed_app(logging, tmp_path)
    async with app.run_test() as pilot:
        await app._resume_local_session(session_id)
        await pilot.pause()

        assert app._plan_mode is not None
        assert PLAN_MODE_TINT_CLASS in _tint_classes(app)
        input_box = _input_box(app)
        assert input_box.border_title == f" {PLAN_MODE_LABEL} "
        amber = Color.parse(app.theme_variables["warning"])
        assert input_box.styles.border_top == ("solid", amber)
        assert input_box.styles.border_bottom == ("solid", amber)


# --- Accept ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_accept_outside_plan_mode_reports_not_in_plan_mode(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=_workspace(tmp_path)) as harness:
            await app._plan_command("accept")

        assert harness.consent.await_count == 0
        [error] = harness.messages(ErrorMessage)
        assert "Not in plan mode" in str(error._error)


@pytest.mark.asyncio
async def test_plan_accept_requires_a_readable_plan_with_a_summary(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _episode_workspace(tmp_path, text="# Plan\n\nNo summary here.\n")
    app._plan_mode = _state(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._plan_accept()

        assert app._plan_mode is not None
        assert harness.consent.await_count == 0
        harness.handoff.assert_not_called()
        [error] = harness.messages(ErrorMessage)
        assert "Cannot accept the plan" in str(error._error)
        assert "Summary" in str(error._error)


@pytest.mark.asyncio
async def test_plan_accept_dismissal_keeps_planning(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _episode_workspace(tmp_path, text=_PLAN_TEXT)
    app._plan_mode = _state(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            harness.consent.return_value = None
            await app._plan_accept()

        assert app._plan_mode is not None
        harness.handoff.assert_not_called()
        harness.seed.assert_not_called()
        posted = harness.messages(UserCommandMessage)
        assert any("still active" in str(widget._content) for widget in posted)


@pytest.mark.asyncio
async def test_plan_accept_hands_over_the_confirmed_bytes_without_seeding(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    # The handoff is the ONE byte authority: the confirmed bytes travel with the
    # episode identity and the snapshot locator, and the TUI seeds nothing.
    workspace = _episode_workspace(tmp_path, text=_PLAN_TEXT)
    plan = {
        "planner": {"model": "glm-5-3", "thinking": "high"},
        "executor": {
            "model": "mistral-large-4",
            "thinking": "medium",
            "identity": "orchestrator",
        },
    }
    app._plan_mode = _state(workspace, pre_entry_planner_thinking="low")
    confirmed = _confirmed(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(plan=plan), workspace=workspace) as harness:
            harness.consent.return_value = confirmed
            await app._plan_accept()
            harness.handoff.assert_awaited_once_with(
                executor_model="mistral-large-4",
                executor_thinking="medium",
                executor_identity="orchestrator",
                plan_path=str(workspace.document_path),
                planner_thinking_alias="glm-5-3",
                planner_thinking_value="low",
                episode_id=workspace.episode_id,
                snapshot_path=str(
                    workspace.directory / f"accepted-{workspace.episode_id}.md"
                ),
                snapshot_sha256=confirmed.sha256,
                snapshot_bytes=confirmed.content,
            )
            consent_call = harness.consent.await_args
            assert consent_call is not None
            assert consent_call.args[1] == confirmed.content
            assert consent_call.args[2] == ExecutorSelection(
                model="mistral-large-4", thinking="medium", identity="orchestrator"
            )
            assert consent_call.args[4].episode_id == workspace.episode_id
            harness.seed.assert_not_called()
            assert app._plan_mode is None


@pytest.mark.asyncio
async def test_plan_accept_shows_the_theme_without_exporting_the_discussion(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    # The theme row is display-only: the consent sees the document's own bytes,
    # with no discussion-context export riding along.
    workspace = _episode_workspace(tmp_path, text=_PLAN_TEXT)
    app._plan_mode = _state(workspace, theme="Fix the widget")
    confirmed = _confirmed(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            harness.consent.return_value = confirmed
            await app._plan_accept()
            consent_call = harness.consent.await_args
            assert consent_call is not None
            assert consent_call.args[3].theme == "Fix the widget"
            assert consent_call.args[1] == _PLAN_TEXT.encode("utf-8")
            assert b"Fix the widget" not in consent_call.args[1]
            harness.seed.assert_not_called()


@pytest.mark.asyncio
async def test_plan_accept_admission_conflict_keeps_planning(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _episode_workspace(tmp_path, text=_PLAN_TEXT)
    app._plan_mode = _state(workspace)
    conflict = AppServerResponseError(
        ProtocolError(
            code=ProtocolErrorCode.CONFLICT,
            message="Cannot start plan execution while a turn is active; "
            "finish or stop the work first",
        )
    )
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            harness.handoff.side_effect = conflict
            await app._plan_accept()

        assert app._plan_mode is not None
        harness.seed.assert_not_called()
        [error] = harness.messages(ErrorMessage)
        assert "retry /plan accept" in str(error._error)


@pytest.mark.asyncio
async def test_plan_accept_post_bind_failure_names_the_plan_path(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _episode_workspace(tmp_path, text=_PLAN_TEXT)
    app._plan_mode = _state(workspace)
    failure = AppServerResponseError(
        ProtocolError(
            code=ProtocolErrorCode.INTERNAL_ERROR,
            message="Plan handoff could not apply the consented executor selection",
        )
    )
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            harness.handoff.side_effect = failure
            await app._plan_accept()

        assert app._plan_mode is None
        harness.seed.assert_not_called()
        [error] = harness.messages(ErrorMessage)
        assert str(workspace.document_path) in str(error._error)
        assert "may already be active" in str(error._error)


@pytest.mark.asyncio
async def test_plan_accept_clears_the_tint_and_the_label(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    # The handoff ends the mode, so the standing indicator ends with it: the
    # replaced session must not keep the amber border /plan entry applied.
    workspace = _episode_workspace(tmp_path, text=_PLAN_TEXT)
    app._plan_mode = _state(workspace)
    async with app.run_test() as pilot:
        app._set_plan_tint(True)
        await pilot.pause()
        assert PLAN_MODE_TINT_CLASS in _tint_classes(app)
        assert _input_box(app).border_title == f" {PLAN_MODE_LABEL} "

        with _Harness(app, fields=_fields(), workspace=workspace, real_reset=True):
            await app._plan_accept()
        await pilot.pause()

        assert app._plan_mode is None
        assert PLAN_MODE_TINT_CLASS not in _tint_classes(app)
        assert _input_box(app).border_title == ""


# --- Cancel ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_cancel_restores_the_pre_entry_values(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    app._plan_mode = _state(
        workspace, pre_entry_active_model="some-model", pre_entry_planner_thinking="low"
    )
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._plan_command("cancel")

        assert app._plan_mode is None
        [(ops, reason)] = harness.writes
        assert reason == "plan mode cancel restore"
        assert [(op.op, op.path, op.value) for op in ops] == [
            ("set", "/active_model", "some-model"),
            ("set", "/thinking_overrides/glm-5-3", "low"),
        ]
        posted = harness.messages(UserCommandMessage)
        assert "restored" in str(posted[0]._content)
        assert str(workspace.document_path) in posted[0]._content


@pytest.mark.asyncio
async def test_plan_cancel_removes_keys_that_were_absent_pre_entry(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    app._plan_mode = _state(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._plan_command("cancel")

        assert app._plan_mode is None
        [(ops, _)] = harness.writes
        assert [(op.op, op.path) for op in ops] == [
            ("remove", "/active_model"),
            ("remove", "/thinking_overrides/glm-5-3"),
        ]


@pytest.mark.asyncio
async def test_plan_cancel_reports_a_restore_failure(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    app._plan_mode = _state(workspace)
    failed = SimpleNamespace(
        rejected=True, failures=["layer write failed"], application="failed"
    )
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            config = app.app_server.resources.config
            with patch.object(config, "write", AsyncMock(return_value=failed)):
                await app._plan_command("cancel")

        assert app._plan_mode is None
        [error] = harness.messages(ErrorMessage)
        assert "could not be fully restored" in str(error._error)
        assert "layer write failed" in str(error._error)


@pytest.mark.asyncio
async def test_plan_cancel_removes_the_marker_only_workspace(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _marked_workspace(tmp_path)
    app._plan_mode = _state(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._plan_command("cancel")

        assert app._plan_mode is None
        assert not workspace.directory.exists()
        posted = harness.messages(UserCommandMessage)
        assert "removed" in str(posted[0]._content)


@pytest.mark.asyncio
async def test_plan_cancel_removes_the_allocated_workspaces_plan_owned_entries(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    allocated = allocate_plan_workspace(
        "01234567-89ab-cdef-0123-deadbeef1234",
        project_root=tmp_path,
        workspaces_root=tmp_path / "workspaces",
    )
    assert (allocated.directory / PLAN_MARKER_FILENAME).is_file()
    workspace = PlanWorkspacePaths(
        directory=allocated.directory, document_path=allocated.document_path
    )
    app._plan_mode = _state(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace):
            await app._plan_command("cancel")

        # The declared accumulation rule: the marker and the sidecar go, while
        # the never-unlinked transition fence keeps an ever-allocated episode
        # directory rmdir-ineligible.
        assert allocated.directory.is_dir()
        assert [entry.name for entry in allocated.directory.iterdir()] == [
            EPISODE_FENCE_FILENAME
        ]


@pytest.mark.asyncio
async def test_plan_cancel_keeps_the_workspace_holding_the_plan(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _marked_workspace(tmp_path)
    _write_plan(workspace)
    app._plan_mode = _state(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._plan_command("cancel")

        assert workspace.directory.is_dir()
        assert workspace.document_path.is_file()
        posted = harness.messages(UserCommandMessage)
        assert str(workspace.document_path) in posted[0]._content


@pytest.mark.asyncio
async def test_plan_cancel_keeps_a_workspace_holding_a_foreign_file(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _marked_workspace(tmp_path)
    (workspace.directory / "notes.txt").write_text("mine\n", encoding="utf-8")
    app._plan_mode = _state(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace):
            await app._plan_command("cancel")

        assert workspace.directory.is_dir()
        assert (workspace.directory / "notes.txt").read_text(encoding="utf-8") == (
            "mine\n"
        )


@pytest.mark.asyncio
async def test_plan_cancel_never_removes_an_unmarked_reused_workspace(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    # The path matches the convention, but no allocation ever claimed it.
    workspace = _workspace(tmp_path)
    app._plan_mode = _state(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace):
            await app._plan_command("cancel")

        assert workspace.directory.is_dir()


@pytest.mark.asyncio
async def test_plan_cancel_is_a_noop_when_the_workspace_is_gone(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _marked_workspace(tmp_path)
    (workspace.directory / PLAN_MARKER_FILENAME).unlink()
    workspace.directory.rmdir()
    app._plan_mode = _state(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._plan_command("cancel")

        assert app._plan_mode is None
        assert not workspace.directory.exists()
        assert not harness.messages(ErrorMessage)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["Fix The Widget", "accept later", "resume"])
async def test_plan_command_enters_with_theme(
    app: ChartreuxApp, tmp_path: Path, theme: str
) -> None:
    workspace = _workspace(tmp_path)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._plan_command(f"  {theme}  ")
        assert app._plan_mode is not None
        assert app._plan_mode.theme == theme
        harness.seed.assert_awaited_once_with(planner_seed(workspace, theme=theme))


@pytest.mark.asyncio
async def test_plan_command_theme_does_not_reenter_active_mode(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    state = _state(workspace)
    app._plan_mode = state
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._plan_command("new theme")
        assert app._plan_mode is state
        assert not harness.writes
        harness.seed.assert_not_awaited()
        assert "already active" in _contents(harness.messages(UserCommandMessage))


# --- /clear while planning ---------------------------------------------------


@pytest.mark.asyncio
async def test_clear_while_planning_removes_the_pins_and_ends_the_mode(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    app._plan_mode = _state(workspace, pre_entry_active_model="some-model")
    async with app.run_test():
        app.app_server.clear_history = AsyncMock()
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            await app._clear_history()

        assert app._plan_mode is None
        [(ops, reason)] = harness.writes
        assert reason == "plan mode cleared"
        assert [(op.op, op.path) for op in ops] == [
            ("remove", "/active_model"),
            ("remove", "/thinking_overrides/glm-5-3"),
        ]
        posted = _contents(harness.messages(UserCommandMessage))
        assert "Plan mode ended" in posted
        assert str(workspace.document_path) in posted


@pytest.mark.asyncio
async def test_clear_while_planning_keeps_the_workspace(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    # /clear keeps everything: it ends the mode without touching the workspace.
    workspace = _marked_workspace(tmp_path)
    app._plan_mode = _state(workspace)
    async with app.run_test():
        app.app_server.clear_history = AsyncMock()
        with _Harness(app, fields=_fields(), workspace=workspace):
            await app._clear_history()

        assert workspace.directory.is_dir()
        assert (workspace.directory / PLAN_MARKER_FILENAME).is_file()


# --- Module seams ------------------------------------------------------------


def test_entry_ops_pin_the_planner_and_escape_the_alias() -> None:
    planner = PlannerSelection(model="vendor/model", thinking="high")
    ops = entry_ops(planner)
    assert [op.path for op in ops] == [
        "/active_model",
        "/thinking_overrides/vendor~1model",
    ]
    assert {op.target_layer for op in ops} == {"overrides"}


def test_restore_ops_cover_present_and_absent_keys(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    restored = restore_ops(
        _state(
            workspace, pre_entry_active_model="kept", pre_entry_planner_thinking="off"
        )
    )
    assert [(op.op, op.value) for op in restored] == [("set", "kept"), ("set", "off")]
    removed = restore_ops(_state(workspace))
    assert [op.op for op in removed] == ["remove", "remove"]
    cleared = clear_ops(
        _state(
            workspace, pre_entry_active_model="kept", pre_entry_planner_thinking="off"
        )
    )
    assert [op.op for op in cleared] == ["remove", "remove"]


def test_pre_entry_snapshot_reads_the_overrides_layer_only() -> None:
    fields = _fields(
        active_model=("pinned-model", "overrides"),
        thinking_overrides={"glm-5-3": "low", "other": "max"},
    )
    active_model, thinking = pre_entry_snapshot(fields, "glm-5-3", [])
    assert (active_model, thinking) == ("pinned-model", "low")


def test_pre_entry_snapshot_reports_absent_keys() -> None:
    fields = _fields(active_model=("admin-model", "admin"), thinking_overrides={})
    active_model, thinking = pre_entry_snapshot(fields, "glm-5-3", [])
    assert (active_model, thinking) == (None, None)


def test_handoff_failure_classification() -> None:
    admission = ProtocolError(code=ProtocolErrorCode.CONFLICT, message="busy")
    mismatch = ProtocolError(
        code=ProtocolErrorCode.CONFLICT, message="mismatch", data={"appliedModel": "x"}
    )
    internal = ProtocolError(
        code=ProtocolErrorCode.INTERNAL_ERROR, message="reset failed"
    )
    assert handoff_failed_before_replacement(admission) is True
    assert handoff_failed_before_replacement(mismatch) is False
    assert handoff_failed_before_replacement(internal) is False


def test_thinking_path_escapes_json_pointer_tokens() -> None:
    assert thinking_path("a/b~c") == "/thinking_overrides/a~1b~0c"


def test_plan_section_reads_the_projected_selection() -> None:
    fields = _fields(
        plan={
            "planner": {"model": "glm-5-3", "thinking": "max"},
            "executor": {
                "model": "mistral-large-4",
                "thinking": "low",
                "identity": "reviewer",
            },
        }
    )
    selection = plan_section(fields)
    assert selection.planner == PlannerSelection(model="glm-5-3", thinking="max")
    assert selection.executor == ExecutorSelection(
        model="mistral-large-4", thinking="low", identity="reviewer"
    )


def test_plan_section_falls_back_to_the_shipped_defaults() -> None:
    assert plan_section([]) == plan_section(_fields())


def test_allocate_workspace_maps_the_core_workspace(tmp_path: Path) -> None:
    directory = tmp_path / "plan-session"
    document = directory / "plan.md"
    sidecar = directory / "episode.json"
    core_workspace = SimpleNamespace(
        directory=directory,
        document_path=document,
        episode_id="ep-1",
        sidecar_path=sidecar,
    )
    with patch(
        "chartreux.core.plan_mode.allocate_plan_workspace",
        MagicMock(return_value=core_workspace),
    ) as allocate:
        paths, error = allocate_workspace("sess-1", project_root=tmp_path)

    assert error is None
    assert paths == PlanWorkspacePaths(
        directory=directory,
        document_path=document,
        episode_id="ep-1",
        sidecar_path=sidecar,
    )
    allocate.assert_called_once_with("sess-1", project_root=tmp_path)


def test_allocate_workspace_reports_the_core_failure(tmp_path: Path) -> None:
    with patch(
        "chartreux.core.plan_mode.allocate_plan_workspace",
        MagicMock(side_effect=PlanWorkspaceError("no free name")),
    ):
        paths, error = allocate_workspace("sess-1", project_root=tmp_path)

    assert paths is None
    assert error == "no free name"


def test_read_document_returns_the_accepted_brief(tmp_path: Path) -> None:
    path = tmp_path / "plan.md"
    path.write_text(_PLAN_TEXT, encoding="utf-8")
    document, error = read_document(path)
    assert error is None
    assert document == PlanDocumentBrief(
        path=path, content=_PLAN_TEXT, summary="Do the thing carefully."
    )


def test_read_document_reports_an_unusable_document(tmp_path: Path) -> None:
    path = tmp_path / "plan.md"
    path.write_text("# Plan\n\nNo summary here.\n", encoding="utf-8")
    document, error = read_document(path)
    assert document is None
    assert error is not None
    assert "Summary" in error


def test_seed_builders_carry_the_path_and_summary(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    planner = planner_seed(workspace)
    assert "Plan mode is active" in planner
    assert str(workspace.document_path) in planner


# --- Consent modal -----------------------------------------------------------


_DOCUMENT = PlanDocumentBrief(
    path=Path("/plans/plan-test/plan.md"),
    content=_PLAN_TEXT,
    summary="Do the thing carefully.",
)
_EXECUTOR = ExecutorSelection(
    model="mistral-large-4", thinking="medium", identity="orchestrator"
)


class _ModalApp(App[None]):
    def __init__(self, screen: PlanConsentScreen) -> None:
        super().__init__()
        self.config = SimpleNamespace(ascii_chrome=False)  # pyright: ignore[reportAttributeAccessIssue]
        self.screen_under_test = screen
        self.result: bool | None = None

    def compose(self) -> ComposeResult:
        yield NoMarkupStatic("host")

    def on_mount(self) -> None:
        self.push_screen(self.screen_under_test, callback=self._remember)

    def _remember(self, result: bool | None) -> None:
        self.result = result


def _screen(theme: str | None = None) -> PlanConsentScreen:
    return PlanConsentScreen(
        plan_path=str(_DOCUMENT.path),
        summary=_DOCUMENT.summary,
        executor_model=_EXECUTOR.model,
        executor_thinking=_EXECUTOR.thinking,
        executor_identity=_EXECUTOR.identity,
        executor_display_name="Mistral Large",
        theme=theme,
    )


def _text(screen: PlanConsentScreen, selector: str) -> str:
    return str(screen.query_one(selector, NoMarkupStatic).content)


def _labels(screen: PlanConsentScreen) -> list[str]:
    return [
        str(widget.content)
        for widget in screen.query(NoMarkupStatic)
        if widget.has_class("plan-consent-label")
    ]


@pytest.mark.asyncio
async def test_consent_modal_shows_path_summary_and_the_exact_executor() -> None:
    app = _ModalApp(_screen())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = app.screen_under_test
        assert _text(screen, "#plan-consent-path") == "/plans/plan-test/plan.md"
        assert _text(screen, "#plan-consent-summary") == "Do the thing carefully."
        executor = _text(screen, "#plan-consent-executor")
        assert "Mistral Large" in executor
        assert "(mistral-large-4)" in executor
        assert "thinking medium" in executor
        assert "identity orchestrator" in executor
        assert str(screen.query_one(f"#{START_BUTTON_ID}", Button).label) == (
            "Start execution"
        )
        assert str(screen.query_one(f"#{CANCEL_BUTTON_ID}", Button).label) == "Cancel"


@pytest.mark.asyncio
async def test_consent_modal_shows_the_theme_row_when_themed() -> None:
    app = _ModalApp(_screen("Fix the widget"))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = app.screen_under_test
        assert _text(screen, "#plan-consent-theme") == "Fix the widget"
        assert _labels(screen).count("Theme") == 1
        assert screen.query_one("#plan-consent-dialog").size.height <= 24


@pytest.mark.asyncio
async def test_consent_modal_has_no_theme_row_when_themeless() -> None:
    app = _ModalApp(_screen())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = app.screen_under_test
        assert not screen.query("#plan-consent-theme")
        assert "Theme" not in _labels(screen)


@pytest.mark.asyncio
async def test_consent_modal_defaults_to_cancel_and_escape_dismisses_false() -> None:
    app = _ModalApp(_screen())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = app.screen_under_test
        assert app.focused is screen.query_one(f"#{CANCEL_BUTTON_ID}", Button)
        await pilot.press("escape")
        await pilot.pause()
        assert app.result is False


@pytest.mark.asyncio
async def test_consent_modal_start_button_confirms() -> None:
    app = _ModalApp(_screen())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{START_BUTTON_ID}")
        await pilot.pause()
        assert app.result is True


@pytest.mark.asyncio
async def test_consent_modal_fits_the_compatibility_floor() -> None:
    app = _ModalApp(_screen())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        dialog = app.screen_under_test.query_one("#plan-consent-dialog")
        assert dialog.size.height <= 24
        assert dialog.size.width <= 80
        hint = _text(app.screen_under_test, "#plan-consent-hint")
        assert "Esc" in hint


@pytest.mark.asyncio
async def test_plan_accept_worker_guard_schedules_one_accept(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    app._plan_mode = _state(workspace)
    flow = AsyncMock()
    async with app.run_test():
        with patch.object(app, "_plan_accept", flow):
            await app._plan_start_accept()
            worker = app._plan_accept_worker
            assert worker is not None
            await app.workers.wait_for_complete([worker])
        flow.assert_awaited_once()


# --- The persisted record ---------------------------------------------------


async def _planning_session(
    tmp_path: Path,
    *,
    theme: str | None = "Fix the widget",
    pre_entry_active_model: str | None = "mistral-large-4",
) -> tuple[str, SessionLoggingConfig, Path]:
    """A persisted session whose record marks it as a DRAFT plan episode.

    The directory is allocated through the real allocator, so it carries the
    ownership marker, an episode claim, and no plan document: a resume observes
    a draft episode and continues planning.
    """
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions"))
    loop = build_test_agent_loop(
        config=build_test_vibe_config(session_logging=logging), cwd=tmp_path
    )
    await loop.persist_empty_session()
    workspace = allocate_plan_workspace(
        loop.session_id, project_root=tmp_path, workspaces_root=tmp_path / "workspaces"
    )
    plan_directory = workspace.directory
    await loop.session_logger.persist_plan_mode(
        PlanModeRecord(
            plan_path=str(workspace.document_path),
            theme=theme,
            planner_model="glm-5-3",
            planner_thinking="high",
            pre_entry_active_model=pre_entry_active_model,
            pre_entry_planner_thinking="low",
            episode_id=workspace.episode_id,
            sidecar_path=str(workspace.sidecar_path),
            draft_path=str(workspace.document_path),
            state="draft",
        )
    )
    session_id = loop.session_id
    await loop.aclose()
    return session_id, logging, plan_directory


def _resumed_app(
    logging: SessionLoggingConfig, tmp_path: Path, *, initial_prompt: str | None = None
) -> tuple[ChartreuxApp, Any]:
    loop = build_test_agent_loop(
        config=build_test_vibe_config(session_logging=logging), cwd=tmp_path
    )
    app = build_test_chartreux_app(agent_loop=loop, initial_prompt=initial_prompt)
    return app, loop


def _notices(loop: Any) -> list[str]:
    return [
        message.content
        for message in loop.messages
        if message.content and "Plan mode resumed" in message.content
    ]


def _resume_notice(
    plan_directory: Path, episode_id: str | None, theme: str | None
) -> str:
    notice = f"Plan mode resumed. Plan document: {plan_directory / 'plan.md'}."
    if theme:
        notice = f"{notice} Theme: {theme}."
    if episode_id:
        notice = f"{notice} Episode: {episode_id}."
    return notice


def _tint_classes(app: ChartreuxApp) -> set[str]:
    container = app._chat_input_container
    assert container is not None
    return set(container.classes)


def _persisted_record(loop: Any) -> PlanModeRecord | None:
    session_dir = loop.session_logger.session_dir
    assert session_dir is not None
    return SessionLoader.load_metadata(session_dir).plan_mode


@pytest.mark.asyncio
async def test_plan_entry_persists_the_record(tmp_path: Path) -> None:
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions"))
    loop = build_test_agent_loop(
        config=build_test_vibe_config(session_logging=logging), cwd=tmp_path
    )
    await loop.persist_empty_session()
    app = build_test_chartreux_app(agent_loop=loop)
    workspace = _workspace(tmp_path)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace):
            await app._plan_command("  Fix the widget  ")

    record = _persisted_record(loop)
    assert record is not None
    assert record.plan_path == str(workspace.document_path)
    assert record.theme == "Fix the widget"
    assert record.planner_model == "glm-5-3"
    assert record.planner_thinking == "high"
    assert record.pre_entry_active_model is None
    assert app._plan_mode is not None


@pytest.mark.asyncio
async def test_plan_entry_reports_a_record_write_failure(tmp_path: Path) -> None:
    # The pins are written and the mode works in memory; only the restart
    # restore is lost, so entry stands and says so.
    workspace = _workspace(tmp_path)
    app = build_test_chartreux_app()
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            with patch.object(
                app.app_server.resources.sessions,
                "write_plan_record",
                AsyncMock(side_effect=RuntimeError("the disk went away")),
            ):
                await app._plan_command("")

        assert app._plan_mode is not None
        harness.seed.assert_awaited_once()
        warnings = "\n".join(
            widget._message for widget in harness.messages(WarningMessage)
        )
        assert "will not be restored on resume" in warnings
        assert "the disk went away" in warnings


@pytest.mark.asyncio
async def test_resumed_cancel_marks_the_record_cancelled_and_restores_the_pins(
    tmp_path: Path,
) -> None:
    session_id, logging, plan_directory = await _planning_session(tmp_path)
    app, loop = _resumed_app(logging, tmp_path)
    async with app.run_test():
        await app._resume_local_session(session_id)
        state = app._plan_mode
        assert state is not None
        with _Harness(app, fields=_fields(), workspace=state.workspace) as harness:
            await app._plan_command("cancel")

        [(ops, reason)] = harness.writes
        assert reason == "plan mode cancel restore"
        assert [(op.op, op.path, op.value) for op in ops] == [
            ("set", "/active_model", "mistral-large-4"),
            ("set", "/thinking_overrides/glm-5-3", "low"),
        ]

    assert app._plan_mode is None
    # The episode is cancelled TERMINALLY; the record keeps that as history.
    record = _persisted_record(loop)
    assert record is not None
    assert record.state == STATE_CANCELLED
    # This test built the directory itself, so it carries no ownership marker:
    # an unmarked directory is foreign and stays.
    assert plan_directory.is_dir()


@pytest.mark.asyncio
async def test_resumed_cancel_marks_the_record_and_cleans_the_workspace_entries(
    tmp_path: Path,
) -> None:
    # The full restart -> resume -> cancel composite: the record's terminal
    # state, the allocated workspace's plan-owned entries, and the pre-entry
    # pins all settle.
    session_id, logging, plan_directory = await _planning_session(tmp_path)
    assert (plan_directory / PLAN_MARKER_FILENAME).is_file()
    assert not (plan_directory / "plan.md").exists()
    app, loop = _resumed_app(logging, tmp_path)
    async with app.run_test():
        await app._resume_local_session(session_id)
        state = app._plan_mode
        assert state is not None
        assert state.workspace.directory == plan_directory
        with _Harness(app, fields=_fields(), workspace=state.workspace) as harness:
            await app._plan_command("cancel")

        [(ops, reason)] = harness.writes
        assert reason == "plan mode cancel restore"
        assert [(op.op, op.path, op.value) for op in ops] == [
            ("set", "/active_model", "mistral-large-4"),
            ("set", "/thinking_overrides/glm-5-3", "low"),
        ]

    assert app._plan_mode is None
    record = _persisted_record(loop)
    assert record is not None
    assert record.state == STATE_CANCELLED
    # The retained transition fence keeps the episode directory (the declared
    # accumulation rule); everything else plan-owned is gone.
    assert [entry.name for entry in plan_directory.iterdir()] == [
        EPISODE_FENCE_FILENAME
    ]


@pytest.mark.asyncio
async def test_clear_while_planning_removes_the_record(tmp_path: Path) -> None:
    session_id, logging, _ = await _planning_session(tmp_path)
    app, loop = _resumed_app(logging, tmp_path)
    async with app.run_test():
        await app._resume_local_session(session_id)
        app.app_server.clear_history = AsyncMock()
        state = app._plan_mode
        assert state is not None
        with _Harness(app, fields=_fields(), workspace=state.workspace):
            await app._clear_history()

    assert app._plan_mode is None
    assert _persisted_record(loop) is None


@pytest.mark.asyncio
async def test_continue_resume_restores_plan_mode_before_the_startup_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id, logging, plan_directory = await _planning_session(tmp_path)
    app, loop = _resumed_app(logging, tmp_path, initial_prompt="continue the work")
    dispatched: list[bool] = []

    def _dispatch() -> None:
        dispatched.append(app._plan_mode is not None)

    monkeypatch.setattr(app, "_process_initial_prompt", _dispatch)

    async with app.run_test():
        app._continue_latest = True
        app._startup_command_availability_ready.set()
        with patch.object(
            app.app_server.resources.sessions,
            "resolve_continue_session",
            AsyncMock(return_value=session_id),
        ):
            await app._auto_resume_on_startup()

        # The startup prompt dispatch sees the restored mode: resume restore
        # completes before any prompt dispatch, on this path.
        assert dispatched == [True]
        state = app._plan_mode
        assert state is not None
        assert state.plan_path == str(plan_directory / "plan.md")
        assert state.theme == "Fix the widget"
        assert state.planner == PlannerSelection(model="glm-5-3", thinking="high")
        assert state.pre_entry_active_model == "mistral-large-4"
        assert state.pre_entry_planner_thinking == "low"
        assert PLAN_MODE_TINT_CLASS in _tint_classes(app)
        assert _notices(loop) == [
            _resume_notice(plan_directory, state.workspace.episode_id, "Fix the widget")
        ]
        # The notice is server-side history, not only a UI mount.
        assert [
            message.injected
            for message in loop.messages
            if message.content and "Plan mode resumed" in message.content
        ] == [True]

        # `-c -P` composition (design §8): the restored record makes the entry
        # an already-active guard, so the extra theme is ignored.
        with _Harness(app, fields=_fields(), workspace=state.workspace) as harness:
            await app._plan_command("some other theme")

        assert harness.writes == []
        assert app._plan_mode is state
        assert "already active" in _contents(harness.messages(UserCommandMessage))


@pytest.mark.asyncio
async def test_picker_resume_restores_plan_mode_before_the_startup_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id, logging, plan_directory = await _planning_session(tmp_path)
    app, loop = _resumed_app(logging, tmp_path, initial_prompt="continue the work")
    dispatched: list[bool] = []

    def _dispatch() -> None:
        dispatched.append(app._plan_mode is not None)

    monkeypatch.setattr(app, "_process_initial_prompt", _dispatch)

    async with app.run_test():
        app._show_resume_picker = True
        app._startup_command_availability_ready.set()
        await app.on_session_picker_app_session_selected(
            SessionPickerApp.SessionSelected(f"local:{session_id}", session_id)
        )

    # This path dispatches the startup prompt before marking the session ready;
    # the invariant is only that the restore came first.
    assert dispatched == [True]
    assert app._session_ready.is_set()
    state = app._plan_mode
    assert state is not None
    assert state.plan_path == str(plan_directory / "plan.md")
    assert PLAN_MODE_TINT_CLASS in _tint_classes(app)
    assert _notices(loop) == [
        _resume_notice(plan_directory, state.workspace.episode_id, "Fix the widget")
    ]


@pytest.mark.asyncio
async def test_resumed_themeless_plan_mode_posts_the_notice_without_a_theme(
    tmp_path: Path,
) -> None:
    session_id, logging, plan_directory = await _planning_session(tmp_path, theme=None)
    app, loop = _resumed_app(logging, tmp_path)
    async with app.run_test():
        await app._resume_local_session(session_id)

    state = app._plan_mode
    assert state is not None
    assert state.theme is None
    assert _notices(loop) == [
        _resume_notice(plan_directory, state.workspace.episode_id, None)
    ]


@pytest.mark.asyncio
async def test_resume_without_a_record_leaves_plan_mode_off(tmp_path: Path) -> None:
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions"))
    saved = build_test_agent_loop(
        config=build_test_vibe_config(session_logging=logging), cwd=tmp_path
    )
    await saved.persist_empty_session()
    session_id = saved.session_id
    await saved.aclose()

    app, loop = _resumed_app(logging, tmp_path)
    async with app.run_test():
        await app._resume_local_session(session_id)

    assert app._plan_mode is None
    assert PLAN_MODE_TINT_CLASS not in _tint_classes(app)
    assert _notices(loop) == []


@pytest.mark.asyncio
async def test_a_resumed_restore_failure_does_not_abort_the_resume(
    tmp_path: Path,
) -> None:
    session_id, logging, _ = await _planning_session(tmp_path)
    app, _ = _resumed_app(logging, tmp_path)
    async with app.run_test():
        with patch(
            "chartreux.cli.textual_ui.app.state_from_record",
            side_effect=RuntimeError("unusable record"),
        ):
            await app._resume_local_session(session_id)

        assert app.app_server.session_id == session_id
        assert app._resume_ui_ready.is_set()
        assert app._plan_mode is None


# --- Startup entry (CLI -P/--plan, design §6/§8) -----------------------------


@pytest.mark.asyncio
async def test_post_ready_startup_ensures_the_startup_plan_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app(startup=StartupOptions(plan=True))
    await app.prepare()
    ensured = AsyncMock()
    monkeypatch.setattr(app, "_ensure_startup_plan_mode", ensured)
    monkeypatch.setattr(
        app, "_show_custom_tools_deprecation_warning_after_initial_history", AsyncMock()
    )
    monkeypatch.setattr(app, "_refresh_banner", Mock())

    await app._complete_post_ready_startup()

    ensured.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("plan", "theme"), [(True, None), ("  Fix the widget  ", "Fix the widget")]
)
async def test_startup_plan_entry_passes_the_requested_theme(
    plan: bool | str, theme: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test():
        entry = AsyncMock()
        monkeypatch.setattr(app, "_plan_enter", entry)
        app._startup_plan = plan
        app._startup_plan_started = False

        await app._ensure_startup_plan_mode()

        entry.assert_awaited_once_with(theme=theme)


@pytest.mark.asyncio
async def test_startup_plan_entry_stays_off_without_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test():
        entry = AsyncMock()
        monkeypatch.setattr(app, "_plan_enter", entry)

        await app._ensure_startup_plan_mode()

        entry.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_plan_entry_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_test_chartreux_app()
    async with app.run_test():
        entry = AsyncMock()
        monkeypatch.setattr(app, "_plan_enter", entry)
        app._startup_plan = True
        app._startup_plan_started = False

        await app._ensure_startup_plan_mode()
        await app._ensure_startup_plan_mode()

        entry.assert_awaited_once_with(theme=None)


@pytest.mark.asyncio
async def test_startup_plan_entry_applies_the_theme_and_the_pins(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    app = build_test_chartreux_app()
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            app._startup_plan = "Fix the widget"
            app._startup_plan_started = False

            await app._ensure_startup_plan_mode()

            assert app._plan_mode is not None
            assert app._plan_mode.theme == "Fix the widget"
            assert [reason for _, reason in harness.writes] == ["plan mode entry"]
            harness.seed.assert_awaited_once_with(
                planner_seed(workspace, theme="Fix the widget")
            )
            assert PLAN_MODE_TINT_CLASS in _tint_classes(app)


@pytest.mark.asyncio
async def test_continue_resume_restores_plan_mode_before_the_startup_plan_entry(
    tmp_path: Path,
) -> None:
    session_id, logging, _ = await _planning_session(tmp_path)
    app, _ = _resumed_app(logging, tmp_path)
    entries: list[bool] = []

    async def _entry(theme: str | None = None) -> None:
        entries.append(app._plan_mode is not None)

    async with app.run_test():
        app._continue_latest = True
        app._startup_plan = "another theme"
        app._startup_plan_started = False
        app._startup_command_availability_ready.set()
        with (
            patch.object(app, "_plan_enter", _entry),
            patch.object(
                app.app_server.resources.sessions,
                "resolve_continue_session",
                AsyncMock(return_value=session_id),
            ),
        ):
            await app._auto_resume_on_startup()

    # Resume restore lands first; the entry then sees the restored mode.
    assert entries == [True]


@pytest.mark.asyncio
async def test_continue_with_plan_keeps_the_resumed_mode_and_ignores_the_theme(
    tmp_path: Path,
) -> None:
    session_id, logging, plan_directory = await _planning_session(tmp_path)
    app, _ = _resumed_app(logging, tmp_path)
    async with app.run_test():
        app._continue_latest = True
        app._startup_command_availability_ready.set()
        with _Harness(app, fields=_fields(), workspace=_workspace(tmp_path)) as harness:
            app._startup_plan = "another theme"
            app._startup_plan_started = False
            with patch.object(
                app.app_server.resources.sessions,
                "resolve_continue_session",
                AsyncMock(return_value=session_id),
            ):
                await app._auto_resume_on_startup()

            assert harness.writes == []
            assert "already active" in _contents(harness.messages(UserCommandMessage))
    state = app._plan_mode
    assert state is not None
    assert state.plan_path == str(plan_directory / "plan.md")
    assert state.theme == "Fix the widget"


@pytest.mark.asyncio
async def test_named_resume_with_plan_enters_plan_mode_on_the_landed_session(
    tmp_path: Path,
) -> None:
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions"))
    saved = build_test_agent_loop(
        config=build_test_vibe_config(session_logging=logging), cwd=tmp_path
    )
    await saved.persist_empty_session()
    session_id = saved.session_id
    await saved.aclose()

    app, _ = _resumed_app(logging, tmp_path)
    workspace = _workspace(tmp_path)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            app._resume_session_id = session_id
            app._startup_plan = "Fix the widget"
            app._startup_plan_started = False
            app._startup_command_availability_ready.set()

            await app._auto_resume_on_startup()

            assert app.app_server.session_id == session_id
            assert app._plan_mode is not None
            assert app._plan_mode.theme == "Fix the widget"
            harness.seed.assert_awaited_once_with(
                planner_seed(workspace, theme="Fix the widget")
            )


@pytest.mark.asyncio
async def test_continue_with_plan_and_no_sessions_still_enters_plan_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test():
        themes: list[str | None] = []

        async def _entry(theme: str | None = None) -> None:
            themes.append(theme)

        monkeypatch.setattr(app, "_plan_enter", _entry)
        monkeypatch.setattr(
            app,
            "_show_custom_tools_deprecation_warning_after_initial_history",
            AsyncMock(),
        )
        app._continue_latest = True
        app._startup_plan = True
        app._startup_plan_started = False
        app._startup_command_availability_ready.set()
        with patch.object(
            app.app_server.resources.sessions,
            "resolve_continue_session",
            AsyncMock(return_value=None),
        ):
            await app._auto_resume_on_startup()

    assert themes == [None]


@pytest.mark.asyncio
async def test_picker_selection_with_plan_enters_plan_mode_after_the_resume(
    tmp_path: Path,
) -> None:
    session_id, logging, _ = await _planning_session(tmp_path)
    app, _ = _resumed_app(logging, tmp_path)
    entries: list[bool] = []

    async def _entry(theme: str | None = None) -> None:
        entries.append(app._plan_mode is not None)

    async with app.run_test():
        app._show_resume_picker = True
        app._startup_plan = "another theme"
        app._startup_plan_started = False
        app._startup_command_availability_ready.set()
        with patch.object(app, "_plan_enter", _entry):
            await app.on_session_picker_app_session_selected(
                SessionPickerApp.SessionSelected(f"local:{session_id}", session_id)
            )

    assert entries == [True]
    assert app._session_ready.is_set()


@pytest.mark.asyncio
async def test_picker_cancel_with_plan_enters_plan_mode_on_the_current_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test():
        entry = AsyncMock()
        monkeypatch.setattr(app, "_plan_enter", entry)
        monkeypatch.setattr(app, "_switch_to_input_app", AsyncMock())
        monkeypatch.setattr(
            app,
            "_show_custom_tools_deprecation_warning_after_initial_history",
            AsyncMock(),
        )
        app._show_resume_picker = True
        app._startup_plan = True
        app._startup_plan_started = False
        app._startup_command_availability_ready.set()

        await app._exit_picker_to_input()

        entry.assert_awaited_once_with(theme=None)


def test_record_and_state_round_trip(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    state = _state(
        workspace, pre_entry_active_model="some-model", pre_entry_planner_thinking="low"
    )
    record = record_from_state(state)
    assert record.plan_path == str(workspace.document_path)
    assert state_from_record(record) == state


def test_record_from_state_carries_the_theme(tmp_path: Path) -> None:
    state = PlanModeState(
        workspace=_workspace(tmp_path),
        planner=PlannerSelection(model="glm-5-3", thinking="high"),
        pre_entry_active_model=None,
        pre_entry_planner_thinking=None,
        theme="Fix the widget",
    )
    assert record_from_state(state).theme == "Fix the widget"


def test_resume_notice_appends_the_theme_only_when_one_exists(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    record = record_from_state(_state(workspace))
    assert resume_notice(record) == (
        f"Plan mode resumed. Plan document: {record.plan_path}."
    )
    themed = record_from_state(
        PlanModeState(
            workspace=workspace,
            planner=PlannerSelection(model="glm-5-3", thinking="high"),
            pre_entry_active_model=None,
            pre_entry_planner_thinking=None,
            theme="Fix the widget",
        )
    )
    assert resume_notice(themed) == (
        f"Plan mode resumed. Plan document: {themed.plan_path}. Theme: Fix the widget."
    )
