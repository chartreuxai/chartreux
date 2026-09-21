from __future__ import annotations

from chartreux.acp import utils


def test_routine_approval_options_are_not_exposed() -> None:
    assert not hasattr(utils, "build_permission_options")
    assert not hasattr(utils, "ToolOption")
    assert not hasattr(utils, "build_mode_state")
