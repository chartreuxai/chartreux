"""H1a/H1b: editor commands are validated before the app suspends, and
execution errors are contained inside the suspend body.

Every scenario runs on a suspend-capable ``Driver`` subclass injected through
``ChartreuxApp._build_driver``. The real headless driver cannot suspend at
all, which would make the "no suspend entered" assertions vacuous.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

import pytest
from textual import events
from textual.driver import Driver
from textual.geometry import Size
from textual.pilot import Pilot

from chartreux.cli.textual_ui import external_editor
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.external_editor import ExternalEditor
from chartreux.cli.textual_ui.program_status import ProgramStatusEmitter, encode_clear
from chartreux.cli.textual_ui.widgets.chat_input import ChatTextArea
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage, PlanFileMessage
from chartreux.utils.paths import get_chartreux_home
from tests.conftest import build_test_chartreux_app, build_test_vibe_config

INVALID_EDITOR_TEXT = "Invalid editor command: check the quotes in $VISUAL or $EDITOR"
EDITOR_RESOLUTION_PREFIX = "Could not start editor: "
CONFIG_READ_ERROR_PREFIX = "Could not open or read config file: "
PLAN_OPEN_ERROR_PREFIX = "Could not open plan in editor: "
MALFORMED_EDITOR = "unmatched '"
WHITESPACE_EDITOR = "   "
MISSING_EDITOR = "chartreux-editor-that-does-not-exist"
WORKING_EDITOR = "/bin/true"
EXIT_CODE = 3


class _SuspendRecordingDriver(Driver):
    """A suspend-capable driver: records suspend/resume and captures writes."""

    def __init__(
        self, app: ChartreuxApp, *, mouse: bool, size: tuple[int, int] | None
    ) -> None:
        super().__init__(app, mouse=mouse, size=size)
        self._terminal_size = size
        self.suspend_calls = 0
        self.resume_calls = 0
        self.writes: list[str] = []

    @property
    def can_suspend(self) -> bool:
        return True

    def write(self, data: str) -> None:
        self.writes.append(data)

    def start_application_mode(self) -> None:
        terminal_size = self._terminal_size or (80, 24)
        size = Size(*terminal_size)
        self.send_message(events.Resize(size, size))

    def disable_input(self) -> None:
        pass

    def stop_application_mode(self) -> None:
        pass

    def suspend_application_mode(self) -> None:
        self.suspend_calls += 1
        super().suspend_application_mode()

    def resume_application_mode(self) -> None:
        self.resume_calls += 1
        super().resume_application_mode()


class _DriverSlot:
    """Holds the driver built inside ``run_test`` for the test to inspect."""

    def __init__(self) -> None:
        self.driver: _SuspendRecordingDriver | None = None


def _install_suspend_driver(monkeypatch: pytest.MonkeyPatch) -> _DriverSlot:
    slot = _DriverSlot()

    def _build_driver(
        _self: ChartreuxApp,
        headless: bool,
        inline: bool,
        mouse: bool,
        size: tuple[int, int] | None,
    ) -> Driver:
        slot.driver = _SuspendRecordingDriver(_self, mouse=mouse, size=size)
        return slot.driver

    monkeypatch.setattr(ChartreuxApp, "_build_driver", _build_driver)
    return slot


def _app_with_substrate(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ChartreuxApp, _DriverSlot]:
    slot = _install_suspend_driver(monkeypatch)
    app = build_test_chartreux_app(config=build_test_vibe_config())
    app._program_status = ProgramStatusEmitter(app, get_enabled=lambda: True)
    return app, slot


async def _run_config_command(app: ChartreuxApp, pilot: Pilot[None]) -> list[str]:
    await app._handle_command("/open-config-file")
    await pilot.pause()
    return [str(error._error) for error in app.query(ErrorMessage)]


async def _mount_plan(
    app: ChartreuxApp, pilot: Pilot[None], tmp_path: Path
) -> PlanFileMessage:
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("# plan\n", "utf-8")
    widget = PlanFileMessage(file_path=plan_file)
    await app._mount_and_scroll(widget)
    await pilot.pause()
    return widget


def _fail_run(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Fail at the spawn layer, after ``resolve_command`` has already passed."""

    def _run(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(subprocess, "run", _run)


def _fail_mkstemp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail temp-file allocation inside ``ExternalEditor.edit``."""
    real_mkstemp = tempfile.mkstemp

    def _mkstemp(
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | os.PathLike[str] | None = None,
        text: bool = False,
    ) -> tuple[int, str]:
        if prefix == "vibe_":
            raise OSError("cannot allocate temp file")
        return real_mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)

    monkeypatch.setattr(external_editor.tempfile, "mkstemp", _mkstemp)


def _fail_unlink(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the cleanup unlink inside ``ExternalEditor.edit``."""
    real_unlink = os.unlink

    def _unlink(path: str | os.PathLike[str], *, dir_fd: int | None = None) -> None:
        if Path(path).name.startswith("vibe_"):
            raise OSError("cannot remove temp file")
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(external_editor.os, "unlink", _unlink)


async def _open_prompt_editor(app: ChartreuxApp, pilot: Pilot[None]) -> None:
    """Drive the prompt site: the suspend body that calls ``ExternalEditor.edit``."""
    text_area = app.query_one(ChatTextArea)
    text_area.action_open_external_editor()
    await pilot.pause()


def _config_path() -> Path:
    return get_chartreux_home() / "config.toml"


def _exit_clears(slot: _DriverSlot) -> list[str]:
    driver = slot.driver
    assert driver is not None
    return [write for write in driver.writes if write == encode_clear()]


@pytest.mark.asyncio
async def test_config_site_malformed_quoting_never_suspends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISUAL", MALFORMED_EDITOR)
    app, slot = _app_with_substrate(monkeypatch)

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        assert app._program_status._app is app
        assert app._program_status._app._driver is driver
        errors = await _run_config_command(app, pilot)
        assert driver.suspend_calls == 0

    assert errors == [INVALID_EDITOR_TEXT]


@pytest.mark.asyncio
async def test_config_site_whitespace_only_editor_never_suspends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISUAL", WHITESPACE_EDITOR)
    app, slot = _app_with_substrate(monkeypatch)

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        errors = await _run_config_command(app, pilot)
        assert driver.suspend_calls == 0

    assert errors == [INVALID_EDITOR_TEXT]


@pytest.mark.asyncio
async def test_config_site_missing_editor_never_suspends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISUAL", MISSING_EDITOR)
    app, slot = _app_with_substrate(monkeypatch)

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        errors = await _run_config_command(app, pilot)
        assert driver.suspend_calls == 0

    assert errors == [f"{EDITOR_RESOLUTION_PREFIX}{MISSING_EDITOR}"]
    assert all("Could not open or read config file" not in text for text in errors)


@pytest.mark.asyncio
async def test_plan_site_malformed_quoting_never_suspends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VISUAL", MALFORMED_EDITOR)
    app, slot = _app_with_substrate(monkeypatch)
    notifications: list[tuple[str, dict[str, object]]] = []

    def _record_notify(message: str, **kwargs: object) -> None:
        notifications.append((message, kwargs))

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        widget = await _mount_plan(app, pilot, tmp_path)
        monkeypatch.setattr(app, "notify", _record_notify)
        widget.open_in_editor()
        assert driver.suspend_calls == 0
        widget.stop_watching()

    assert [message for message, _ in notifications] == [
        f"{EDITOR_RESOLUTION_PREFIX}{MALFORMED_EDITOR}"
    ]
    assert notifications[0][1]["severity"] == "error"


@pytest.mark.asyncio
async def test_plan_site_whitespace_only_editor_never_suspends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VISUAL", WHITESPACE_EDITOR)
    app, slot = _app_with_substrate(monkeypatch)
    notifications: list[tuple[str, dict[str, object]]] = []

    def _record_notify(message: str, **kwargs: object) -> None:
        notifications.append((message, kwargs))

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        widget = await _mount_plan(app, pilot, tmp_path)
        monkeypatch.setattr(app, "notify", _record_notify)
        widget.open_in_editor()
        assert driver.suspend_calls == 0
        widget.stop_watching()

    assert [message for message, _ in notifications] == [
        f"{EDITOR_RESOLUTION_PREFIX}{WHITESPACE_EDITOR}"
    ]


@pytest.mark.asyncio
async def test_plan_site_missing_editor_never_suspends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VISUAL", MISSING_EDITOR)
    app, slot = _app_with_substrate(monkeypatch)
    notifications: list[tuple[str, dict[str, object]]] = []

    def _record_notify(message: str, **kwargs: object) -> None:
        notifications.append((message, kwargs))

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        widget = await _mount_plan(app, pilot, tmp_path)
        monkeypatch.setattr(app, "notify", _record_notify)
        widget.open_in_editor()
        assert driver.suspend_calls == 0
        widget.stop_watching()

    assert [message for message, _ in notifications] == [
        f"{EDITOR_RESOLUTION_PREFIX}{MISSING_EDITOR}"
    ]


@pytest.mark.asyncio
async def test_config_site_well_formed_command_still_suspends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISUAL", WORKING_EDITOR)
    monkeypatch.setattr(
        ExternalEditor, "edit_file", staticmethod(lambda *_a, **_k: None)
    )
    app, slot = _app_with_substrate(monkeypatch)

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        errors = await _run_config_command(app, pilot)
        assert driver.suspend_calls == 1
        assert driver.resume_calls == 1

    assert errors == []


@pytest.mark.asyncio
async def test_plan_site_well_formed_command_still_suspends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VISUAL", WORKING_EDITOR)
    monkeypatch.setattr(
        ExternalEditor, "edit_file", staticmethod(lambda *_a, **_k: None)
    )
    app, slot = _app_with_substrate(monkeypatch)
    notifications: list[tuple[str, dict[str, object]]] = []

    def _record_notify(message: str, **kwargs: object) -> None:
        notifications.append((message, kwargs))

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        widget = await _mount_plan(app, pilot, tmp_path)
        monkeypatch.setattr(app, "notify", _record_notify)
        widget.open_in_editor()
        assert driver.suspend_calls == 1
        assert driver.resume_calls == 1
        widget.stop_watching()

    assert notifications == []


@pytest.mark.asyncio
async def test_config_site_spawn_failure_restores_driver_and_clear_delivers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H1b (iii): the spawn fails after resolution, inside the suspend body."""
    monkeypatch.setenv("VISUAL", WORKING_EDITOR)
    _fail_run(monkeypatch, OSError("editor failed to spawn"))
    app, slot = _app_with_substrate(monkeypatch)

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        errors = await _run_config_command(app, pilot)
        assert driver.suspend_calls == 1
        assert driver.resume_calls == 1
        app._program_status.emit("working")

    assert errors == [f"{CONFIG_READ_ERROR_PREFIX}`{_config_path()}`"]
    assert _exit_clears(slot) == [encode_clear()]


@pytest.mark.asyncio
async def test_plan_site_spawn_failure_restores_driver_and_clear_delivers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """H1b (iii): the same spawn failure at the plan editor site."""
    monkeypatch.setenv("VISUAL", WORKING_EDITOR)
    _fail_run(monkeypatch, OSError("editor failed to spawn"))
    app, slot = _app_with_substrate(monkeypatch)
    notifications: list[tuple[str, dict[str, object]]] = []

    def _record_notify(message: str, **kwargs: object) -> None:
        notifications.append((message, kwargs))

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        widget = await _mount_plan(app, pilot, tmp_path)
        monkeypatch.setattr(app, "notify", _record_notify)
        widget.open_in_editor()
        assert driver.suspend_calls == 1
        assert driver.resume_calls == 1
        widget.stop_watching()
        app._program_status.emit("working")

    assert [message for message, _ in notifications] == [
        f"{PLAN_OPEN_ERROR_PREFIX}{tmp_path / 'plan.md'}"
    ]
    assert notifications[0][1]["severity"] == "error"
    assert _exit_clears(slot) == [encode_clear()]


@pytest.mark.asyncio
async def test_config_site_nonzero_exit_restores_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H1b (iv): a nonzero editor exit keeps the existing CalledProcessError arm."""
    monkeypatch.setenv("VISUAL", WORKING_EDITOR)
    _fail_run(monkeypatch, subprocess.CalledProcessError(EXIT_CODE, "editor"))
    app, slot = _app_with_substrate(monkeypatch)

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        errors = await _run_config_command(app, pilot)
        assert driver.suspend_calls == 1
        assert driver.resume_calls == 1
        app._program_status.emit("working")

    assert errors == [
        f"Editor exited with an error (code {EXIT_CODE}): `{_config_path()}`"
    ]
    assert _exit_clears(slot) == [encode_clear()]


@pytest.mark.asyncio
async def test_config_site_editor_error_surfaces_before_changed_file_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H1b ordering: the edit error is raised before the changed-file read.

    The config file stays readable for the initial ``before`` read and turns
    unreadable only as the editor is about to fail. Re-raising the captured
    error after ``changed = config_path.read_text(...)`` would report the
    file-read message instead of the editor message.
    """
    monkeypatch.setenv("VISUAL", WORKING_EDITOR)
    config_path = _config_path()

    def _run(*_args: object, **_kwargs: object) -> None:
        config_path.chmod(0o000)
        raise subprocess.CalledProcessError(EXIT_CODE, "editor")

    monkeypatch.setattr(subprocess, "run", _run)
    app, slot = _app_with_substrate(monkeypatch)

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        errors = await _run_config_command(app, pilot)
        assert driver.suspend_calls == 1
        assert driver.resume_calls == 1
        app._program_status.emit("working")

    config_path.chmod(0o600)
    assert errors == [
        f"Editor exited with an error (code {EXIT_CODE}): `{config_path}`"
    ]
    assert all(CONFIG_READ_ERROR_PREFIX not in text for text in errors)
    assert _exit_clears(slot) == [encode_clear()]


@pytest.mark.asyncio
async def test_prompt_site_mkstemp_failure_restores_driver_and_clear_delivers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H1c (v): a temp-file allocation failure is contained inside ``edit``."""
    _fail_mkstemp(monkeypatch)
    app, slot = _app_with_substrate(monkeypatch)

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        await _open_prompt_editor(app, pilot)
        assert driver.suspend_calls == 1
        assert driver.resume_calls == 1
        app._program_status.emit("working")

    assert _exit_clears(slot) == [encode_clear()]


@pytest.mark.asyncio
async def test_prompt_site_unlink_failure_restores_driver_and_clear_delivers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H1c (v): a cleanup unlink failure is contained inside ``edit``."""
    monkeypatch.setenv("VISUAL", WORKING_EDITOR)
    monkeypatch.setattr(
        ExternalEditor, "edit_file", staticmethod(lambda *_a, **_k: None)
    )
    _fail_unlink(monkeypatch)
    app, slot = _app_with_substrate(monkeypatch)

    async with app.run_test() as pilot:
        driver = slot.driver
        assert driver is not None
        await _open_prompt_editor(app, pilot)
        assert driver.suspend_calls == 1
        assert driver.resume_calls == 1
        app._program_status.emit("working")

    assert _exit_clears(slot) == [encode_clear()]
