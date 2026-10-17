"""Identity and cycle contracts for the settings screen facade (stage 11, Tier B).

The W5 extraction (v0.5.5 stage 11) moves the settings screen's value/draft
helpers, list widgets, and help widgets into ``widgets/settings_lists.py`` and
``widgets/settings_help.py``, and its write/collections/editor/confirmation
controllers into the four ``screens/settings_*.py`` modules, while
``screens/settings.py`` re-exports the moved names. These contracts are the
move guards: every test-imported name resolves through the facade to the same
object as its canonical home, and the widget/controller modules import without
pulling ``screens.settings`` into ``sys.modules`` (the cycle-breaking
contract — controllers may reference the screen only under ``TYPE_CHECKING``).
"""

from __future__ import annotations

import subprocess
import sys

import chartreux.cli.textual_ui.screens.settings as settings_facade
from chartreux.cli.textual_ui.widgets import settings_help, settings_lists


def test_facade_reexports_the_ten_test_imported_names_as_identical_objects() -> None:
    assert settings_facade.ConfirmationText is settings_lists.ConfirmationText
    assert settings_facade.SettingsChecklist is settings_lists.SettingsChecklist
    assert settings_facade.SettingsHints is settings_help.SettingsHints
    assert settings_facade.SettingsOptionList is settings_lists.SettingsOptionList
    assert settings_facade._draft_values is settings_lists._draft_values
    assert settings_facade.inventory_item_state is settings_lists.inventory_item_state
    assert (
        settings_facade.inventory_name_matches is settings_lists.inventory_name_matches
    )
    assert settings_facade.parse_setting_value is settings_lists.parse_setting_value
    assert settings_facade.toggle_inventory_name is settings_lists.toggle_inventory_name


def test_facade_defines_the_screen_and_reexports_the_editor_context() -> None:
    assert (
        settings_facade.SettingsScreen.__module__
        == "chartreux.cli.textual_ui.screens.settings"
    )
    assert settings_facade._ListEditorContext is settings_lists._ListEditorContext


def test_widget_and_controller_modules_import_without_the_screen() -> None:
    code = (
        "import sys;"
        "import chartreux.cli.textual_ui.widgets.settings_help;"
        "import chartreux.cli.textual_ui.widgets.settings_lists;"
        "import chartreux.cli.textual_ui.screens.settings_collections;"
        "import chartreux.cli.textual_ui.screens.settings_confirmation;"
        "import chartreux.cli.textual_ui.screens.settings_editor;"
        "import chartreux.cli.textual_ui.screens.settings_write;"
        "assert 'chartreux.cli.textual_ui.screens.settings' not in sys.modules;"
        "print('cycle-free')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "cycle-free" in result.stdout
