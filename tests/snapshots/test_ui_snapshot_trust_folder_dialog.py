from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from textual._compositor import CompositorUpdate
from textual.containers import VerticalScroll

from chartreux.setup.trusted_folders.trust_folder_dialog import TrustFolderApp
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.snapshots.snap_compare import SnapCompare

_DIALOG_CSS = str(
    Path(__file__).parents[2]
    / "chartreux"
    / "setup"
    / "trusted_folders"
    / "trust_folder_dialog.tcss"
)
_SETTINGS_PATH = "/home/user/.chartreux/trusted_folders.toml"


class TrustFolderDialogSnapshotApp(TrustFolderApp):
    """cwd is itself the trust target (two-option dialog)."""

    CSS_PATH = _DIALOG_CSS

    def __init__(self) -> None:
        super().__init__(
            cwd=Path("/home/user/projects/my-project"),
            repo_root=None,
            detected_files=["AGENTS.md", ".chartreux/"],
            settings_path=_SETTINGS_PATH,
        )


class TrustFolderDialogWithRepoSnapshotApp(TrustFolderApp):
    """cwd inside a git repo (three-option dialog)."""

    CSS_PATH = _DIALOG_CSS

    def __init__(self) -> None:
        super().__init__(
            cwd=Path("/home/user/projects/my-project/src/pkg"),
            repo_root=Path("/home/user/projects/my-project"),
            detected_files=["AGENTS.md"],
            repo_detected_files=[".chartreux/", "src/AGENTS.md"],
            offer_repo_trust=True,
            settings_path=_SETTINGS_PATH,
        )


class TrustFolderDialogUntrustedRepoSnapshotApp(TrustFolderApp):
    """cwd inside a git repo that was previously marked untrusted."""

    CSS_PATH = _DIALOG_CSS

    def __init__(self) -> None:
        super().__init__(
            cwd=Path("/home/user/projects/my-project/src/pkg"),
            repo_root=Path("/home/user/projects/my-project"),
            offer_repo_trust=False,
            repo_explicitly_untrusted=True,
            detected_files=["AGENTS.md"],
            settings_path=_SETTINGS_PATH,
        )


class TrustFolderDialogManyFilesSnapshotApp(TrustFolderApp):
    CSS_PATH = _DIALOG_CSS

    def __init__(self) -> None:
        detected = [f"sub{i}/AGENTS.md" for i in range(20)] + [
            ".chartreux/",
            ".agents/",
        ]
        super().__init__(
            cwd=Path("/home/user/projects/my-project"),
            repo_root=None,
            detected_files=detected,
            settings_path=_SETTINGS_PATH,
        )


def test_snapshot_trust_folder_dialog(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_trust_folder_dialog.py:TrustFolderDialogSnapshotApp",
        terminal_size=(80, 40),
    )


def test_snapshot_trust_folder_dialog_with_repo(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_trust_folder_dialog.py:TrustFolderDialogWithRepoSnapshotApp",
        terminal_size=(80, 40),
    )


def test_snapshot_trust_folder_dialog_untrusted_repo(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_trust_folder_dialog.py:TrustFolderDialogUntrustedRepoSnapshotApp",
        terminal_size=(80, 40),
    )


def test_snapshot_trust_folder_dialog_small_terminal(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_trust_folder_dialog.py:TrustFolderDialogSnapshotApp",
        terminal_size=(80, 24),
    )


def test_snapshot_trust_folder_dialog_many_files(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_trust_folder_dialog.py:TrustFolderDialogManyFilesSnapshotApp",
        terminal_size=(80, 40),
    )


@pytest.mark.asyncio
async def test_detected_files_can_be_scrolled_by_keyboard_without_choosing_trust() -> (
    None
):
    app = TrustFolderDialogManyFilesSnapshotApp()

    async with app.run_test(size=(80, 24)) as pilot:
        files = app.query_one("#trust-dialog-content", VerticalScroll)
        await pilot.press("tab")
        assert files.has_focus
        for _ in range(12):
            await pilot.press("down")
        await pilot.pause()
        assert files.scroll_offset.y > 0
        assert app.return_value is None
        await pilot.press("enter")
        assert not files.has_focus
        assert app.return_value is None


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_size", [(80, 24), (120, 36)])
async def test_trust_dialog_footer_and_options_fit_terminal(
    terminal_size: tuple[int, int],
) -> None:
    cwd = Path("/very/long/work/path/" + "x" * 40)
    repo_root = Path("/very/long/repository/path/" + "y" * 40)
    app = TrustFolderApp(
        cwd=cwd,
        repo_root=repo_root,
        offer_repo_trust=True,
        repo_explicitly_untrusted=True,
        detected_files=["AGENTS.md", ".chartreux/", "config.toml"],
        repo_detected_files=[".chartreux/", "src/AGENTS.md", "pyproject.toml"],
        settings_path="/very/long/settings/path/trusted_folders.toml",
    )

    async with app.run_test(size=terminal_size):
        dialog = app.query_one("#trust-dialog")
        content = app.query_one("#trust-dialog-content", VerticalScroll)
        widgets = [
            *app.query(".trust-option"),
            app.query_one("#trust-dialog-path"),
            app.query_one("#trust-dialog-repo-untrusted"),
            app.query_one("#trust-dialog-save-info"),
            app.query_one(".trust-dialog-help"),
        ]
        assert all(widget.region.y >= 0 for widget in widgets)
        assert all(widget.region.bottom <= app.size.height for widget in widgets)

        warning_widget = app.query_one("#trust-dialog-repo-untrusted")
        assert str(repo_root) in str(cast(NoMarkupStatic, warning_widget).content)
        assert "marked untrusted" in str(cast(NoMarkupStatic, warning_widget).content)

        help_widget = app.query_one(".trust-dialog-help")
        if terminal_size == (80, 24):
            assert help_widget.region.height == 2
        assert content.max_scroll_y > 0
        assert dialog.max_scroll_y == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("app_factory", "option_range"),
    [
        (TrustFolderDialogSnapshotApp, "1-2"),
        (TrustFolderDialogWithRepoSnapshotApp, "1-3"),
    ],
)
async def test_trust_dialog_help_shows_available_option_range(
    app_factory: Callable[[], TrustFolderApp], option_range: str
) -> None:
    app = app_factory()

    async with app.run_test():
        help_widget = app.query_one(".trust-dialog-help")
        assert f"{option_range} Choose" in str(
            cast(NoMarkupStatic, help_widget).content
        )


@pytest.mark.asyncio
async def test_trust_dialog_no_color_uses_grayscale_theme_with_contrast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    app = TrustFolderDialogSnapshotApp()
    app.configured_theme = "dark"

    async with app.run_test(size=(80, 24)):
        dialog = app.query_one("#trust-dialog")
        warning = app.query_one("#trust-dialog-warning")

        assert app.theme == "textual-dark"
        assert app.no_color is True
        assert app.native_ansi_color is True

        warning_color = warning.styles.color
        dialog_background = dialog.styles.background
        warning_luminance = (
            0.2126 * warning_color.r
            + 0.7152 * warning_color.g
            + 0.0722 * warning_color.b
        )
        background_luminance = (
            0.2126 * dialog_background.r
            + 0.7152 * dialog_background.g
            + 0.0722 * dialog_background.b
        )
        assert warning_luminance > background_luminance

        rendered = app.screen._compositor.render_update(full=True)
        assert rendered is not None
        rendered_text = cast(CompositorUpdate, rendered).render_segments(app.console)
        assert "Trusting grants this folder permission" in rendered_text
        assert "\x1b[38;" not in rendered_text
        assert "\x1b[48;" not in rendered_text


@pytest.mark.asyncio
@pytest.mark.parametrize("ascii_chrome", [False, True])
async def test_trust_dialog_ascii_chrome_changes_rendered_border(
    ascii_chrome: bool,
) -> None:
    app = TrustFolderApp(
        cwd=Path("/home/user/projects/my-project"),
        repo_root=None,
        detected_files=["AGENTS.md"],
        ascii_chrome=ascii_chrome,
    )

    async with app.run_test(size=(80, 24)):
        dialog = app.query_one("#trust-dialog")
        border_line = app.screen._compositor.render_strips()[dialog.region.y].text

        if ascii_chrome:
            assert "+" in border_line
            assert "┌" not in border_line
        else:
            assert "┌" in border_line


@pytest.mark.asyncio
async def test_trust_dialog_compact_height_does_not_fill_terminal() -> None:
    app = TrustFolderDialogSnapshotApp()

    async with app.run_test(size=(80, 40)):
        dialog = app.query_one("#trust-dialog")
        content = app.query_one("#trust-dialog-content", VerticalScroll)

        assert dialog.region.height < app.size.height
        assert content.max_scroll_y == 0


@pytest.mark.asyncio
async def test_escape_cancels_trust_dialog() -> None:
    app = TrustFolderDialogSnapshotApp()

    async with app.run_test() as pilot:
        await pilot.press("escape")
        assert app._quit_without_saving is True
        assert app.return_value is None
        assert "Exit without starting" in str(
            cast(NoMarkupStatic, app.query_one(".trust-dialog-help")).content
        )


@pytest.mark.asyncio
async def test_trust_dialog_uses_configured_theme() -> None:
    app = TrustFolderApp(
        cwd=Path("/workspace"), repo_root=None, detected_files=[], theme="light"
    )

    async with app.run_test():
        assert app.theme == "ansi-light"


@pytest.mark.asyncio
async def test_number_shortcuts_remain_visible_and_options_fit_narrow_width() -> None:
    app = TrustFolderDialogWithRepoSnapshotApp()

    async with app.run_test(size=(40, 24)) as pilot:
        options = [
            cast(NoMarkupStatic, option) for option in app.query(".trust-option")
        ]

        assert [
            str(option.content).endswith(f"{idx}. {label}")
            for idx, label, option in [
                (1, "Trust full repo", options[0]),
                (2, "Trust folder", options[1]),
                (3, "Don't trust (save as untrusted)", options[2]),
            ]
        ] == [True, True, True]
        assert all(option.region.width > 0 for option in options)
        assert all(option.region.right <= app.size.width for option in options)

        await pilot.press("3")
        assert app.return_value == "decline"


@pytest.mark.asyncio
async def test_trust_options_activate_by_mouse_and_disclose_saved_refusal() -> None:
    app = TrustFolderDialogWithRepoSnapshotApp()

    async with app.run_test(size=(80, 24)) as pilot:
        options = list(app.query(".trust-option"))
        await pilot.click(options[2])

        assert app.return_value == "decline"
        assert "Don't trust (save as untrusted)" in str(
            cast(NoMarkupStatic, options[2]).content
        )


@pytest.mark.asyncio
async def test_trust_footer_keeps_actions_in_two_intentional_lines() -> None:
    app = TrustFolderDialogSnapshotApp()

    async with app.run_test(size=(80, 24)):
        help_widget = cast(NoMarkupStatic, app.query_one(".trust-dialog-help"))
        lines = str(help_widget.content).splitlines()

        assert help_widget.region.height == 2
        assert "Navigate/scroll" in lines[0]
        assert "1-2 Choose" in lines[0]
        assert "Tab Inspect files" in lines[0]
        assert "Enter Select" in lines[1]
        assert "Esc Exit without starting" in lines[1]
