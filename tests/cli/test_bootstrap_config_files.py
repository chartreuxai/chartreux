from __future__ import annotations

import tomllib

import pytest

from chartreux.cli import cli as cli_mod
from chartreux.core.config import harness_files
from chartreux.core.paths import GLOBAL_ENV_FILE, HISTORY_FILE


def test_bootstrap_exports_empty_selections_config_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_file = harness_files.get_harness_files_manager().user_config_file
    config_file.unlink(missing_ok=True)

    cli_mod.bootstrap_config_files()

    assert config_file.exists()
    with config_file.open("rb") as stream:
        assert tomllib.load(stream) == {}
    captured = capsys.readouterr()
    assert "Created selections config" not in captured.out
    assert "Created selections config" in captured.err


def test_bootstrap_seeds_history_greeting() -> None:
    history_file = HISTORY_FILE.path
    history_file.unlink(missing_ok=True)

    cli_mod.bootstrap_config_files()

    assert history_file.exists()
    assert history_file.read_text(encoding="utf-8") == "Hello Chartreux!\n"


def test_bootstrap_creates_empty_env_file() -> None:
    env_file = GLOBAL_ENV_FILE.path
    env_file.unlink(missing_ok=True)

    cli_mod.bootstrap_config_files()

    assert env_file.exists()
    assert env_file.read_text(encoding="utf-8") == ""
