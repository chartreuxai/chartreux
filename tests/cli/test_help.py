from __future__ import annotations

import pytest

from chartreux.cli.entrypoint import parse_arguments


def test_help_omits_updater_surface(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["chartreux", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        parse_arguments()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--check-upgrade" not in output
    assert "update         " not in output
