"""The /plan TUI surface: commands, the consent modal, and the handoff wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.app import App, ComposeResult
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
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.plan_mode import (
    ExecutorSelection,
    PlanDocumentBrief,
    PlanModeState,
    PlannerSelection,
    PlanWorkspacePaths,
    allocate_workspace,
    clear_ops,
    entry_ops,
    executor_seed,
    handoff_failed_before_replacement,
    plan_section,
    planner_seed,
    pre_entry_snapshot,
    read_document,
    restore_ops,
    thinking_path,
)
from chartreux.cli.textual_ui.screens.plan_consent import (
    CANCEL_BUTTON_ID,
    START_BUTTON_ID,
    PlanConsentScreen,
)
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from chartreux.config_values import ThinkingLevel
from chartreux.core.config.models import PlanSectionConfig
from chartreux.core.plan_mode import PlanWorkspaceError
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.conftest import build_test_chartreux_app

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
        )
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


def _write_plan(workspace: PlanWorkspacePaths, text: str = _PLAN_TEXT) -> None:
    workspace.document_path.write_text(text, encoding="utf-8")


def _state(
    workspace: PlanWorkspacePaths,
    *,
    planner_model: str = "glm-5-3",
    planner_thinking: ThinkingLevel = "high",
    pre_entry_active_model: str | None = None,
    pre_entry_planner_thinking: str | None = None,
) -> PlanModeState:
    return PlanModeState(
        workspace=workspace,
        planner=PlannerSelection(model=planner_model, thinking=planner_thinking),
        pre_entry_active_model=pre_entry_active_model,
        pre_entry_planner_thinking=pre_entry_planner_thinking,
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
    ) -> None:
        self.app = app
        self.mounted: list[Any] = []
        self.writes: list[tuple[list[Any], str]] = []
        self._patchers: list[Any] = []
        self.handoff = AsyncMock(return_value=None)
        self.seed = AsyncMock()
        self.consent = AsyncMock(return_value=True)
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
    workspace = _workspace(tmp_path)
    workspace.document_path.write_text("# Plan\n\nNo summary here.\n", encoding="utf-8")
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
    workspace = _workspace(tmp_path)
    _write_plan(workspace)
    app._plan_mode = _state(workspace)
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=workspace) as harness:
            harness.consent.return_value = False
            await app._plan_accept()

        assert app._plan_mode is not None
        harness.handoff.assert_not_called()
        harness.seed.assert_not_called()
        posted = harness.messages(UserCommandMessage)
        assert any("still active" in str(widget._content) for widget in posted)


@pytest.mark.asyncio
async def test_plan_accept_applies_the_displayed_snapshot_and_seeds_the_executor(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    _write_plan(workspace)
    plan = {
        "planner": {"model": "glm-5-3", "thinking": "high"},
        "executor": {
            "model": "mistral-large-4",
            "thinking": "medium",
            "identity": "orchestrator",
        },
    }
    app._plan_mode = _state(workspace, pre_entry_planner_thinking="low")
    async with app.run_test():
        with _Harness(app, fields=_fields(plan=plan), workspace=workspace) as harness:
            await app._plan_accept()
            harness.handoff.assert_awaited_once_with(
                executor_model="mistral-large-4",
                executor_thinking="medium",
                executor_identity="orchestrator",
                plan_path=str(workspace.document_path),
                planner_thinking_alias="glm-5-3",
                planner_thinking_value="low",
            )
            consent_call = harness.consent.await_args
            assert consent_call is not None
            snapshot = consent_call.args[1]
            assert snapshot == ExecutorSelection(
                model="mistral-large-4", thinking="medium", identity="orchestrator"
            )
            harness.seed.assert_awaited_once()
            seed_call = harness.seed.await_args
            assert seed_call is not None
            seed = seed_call.args[0]
            assert "Do the thing carefully." in seed
            assert str(workspace.document_path) in seed
            assert app._plan_mode is None


@pytest.mark.asyncio
async def test_plan_accept_admission_conflict_keeps_planning(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    _write_plan(workspace)
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
    workspace = _workspace(tmp_path)
    _write_plan(workspace)
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
async def test_plan_command_rejects_an_unknown_action(
    app: ChartreuxApp, tmp_path: Path
) -> None:
    async with app.run_test():
        with _Harness(app, fields=_fields(), workspace=_workspace(tmp_path)) as harness:
            await app._plan_command("resume")

        [error] = harness.messages(ErrorMessage)
        assert "Unknown /plan action: resume" in str(error._error)


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
    core_workspace = SimpleNamespace(directory=directory, document_path=document)
    with patch(
        "chartreux.core.plan_mode.allocate_plan_workspace",
        MagicMock(return_value=core_workspace),
    ) as allocate:
        paths, error = allocate_workspace("sess-1", project_root=tmp_path)

    assert error is None
    assert paths == PlanWorkspacePaths(directory=directory, document_path=document)
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
    executor = executor_seed(
        PlanDocumentBrief(
            path=workspace.document_path,
            content=_PLAN_TEXT,
            summary="Do the thing carefully.",
        )
    )
    assert "Do the thing carefully." in executor
    assert str(workspace.document_path) in executor


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


def _screen() -> PlanConsentScreen:
    return PlanConsentScreen(
        plan_path=str(_DOCUMENT.path),
        summary=_DOCUMENT.summary,
        executor_model=_EXECUTOR.model,
        executor_thinking=_EXECUTOR.thinking,
        executor_identity=_EXECUTOR.identity,
        executor_display_name="Mistral Large",
    )


def _text(screen: PlanConsentScreen, selector: str) -> str:
    return str(screen.query_one(selector, NoMarkupStatic).content)


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
