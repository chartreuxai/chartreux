from __future__ import annotations

from importlib.util import find_spec

from chartreux.cli.commands import CommandRegistry
from chartreux.cli.textual_ui.app import BottomApp, ChartreuxApp


def test_registry_browser_is_not_an_available_command_or_widget() -> None:
    registry = CommandRegistry()
    assert registry.get_command_name("/skills") is None
    assert registry.parse_command("/skills") is None
    assert "/skills" not in registry.get_help_text()
    assert "skills" not in registry._build_commands()
    assert not hasattr(ChartreuxApp, "_show_skills")
    assert not hasattr(BottomApp, "SkillsBrowser")
    assert find_spec("chartreux.cli.textual_ui.widgets.skills_browser") is None
