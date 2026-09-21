from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

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
async def test_escape_cancels_trust_dialog() -> None:
    app = TrustFolderDialogSnapshotApp()

    async with app.run_test() as pilot:
        await pilot.press("escape")
        assert app._quit_without_saving is True
        assert app.return_value is None


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
                (3, "Don't trust", options[2]),
            ]
        ] == [True, True, True]
        assert all(option.region.width > 0 for option in options)
        assert all(option.region.right <= app.size.width for option in options)

        await pilot.press("3")
        assert app.return_value == "decline"
