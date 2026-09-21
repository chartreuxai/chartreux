from __future__ import annotations

import importlib
import io
import sys

import pytest

from chartreux.core import config as config_mod
from chartreux.setup import onboarding as onboarding_mod


class _InputWithReconfigure(io.StringIO):
    def reconfigure(self, **_kwargs: object) -> None:
        pass


def test_setup_without_tty_does_not_run_onboarding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", _InputWithReconfigure())
    entrypoint_mod = importlib.import_module("chartreux.acp.entrypoint")
    onboarding_called = False

    def fail_onboarding(*_args: object, **_kwargs: object) -> None:
        nonlocal onboarding_called
        onboarding_called = True

    monkeypatch.setattr(
        entrypoint_mod, "init_harness_files_manager", lambda *_args: None
    )
    monkeypatch.setattr(
        entrypoint_mod, "init_file_logging", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(config_mod, "load_dotenv_values", lambda: None)
    monkeypatch.setattr(entrypoint_mod, "bootstrap_config_files", lambda: None)
    monkeypatch.setattr(
        entrypoint_mod, "parse_arguments", lambda: entrypoint_mod.Arguments(setup=True)
    )
    monkeypatch.setattr(entrypoint_mod, "has_usable_terminal", lambda: False)
    monkeypatch.setattr(onboarding_mod, "run_onboarding", fail_onboarding)

    with pytest.raises(SystemExit) as exc_info:
        entrypoint_mod.main()

    assert exc_info.value.code == 1
    assert onboarding_called is False
    assert "Interactive setup requires a terminal" in capsys.readouterr().err
