from __future__ import annotations

import stat
import tomllib

import pytest

from chartreux.cli import cli as cli_mod
from chartreux.core.config import ChartreuxConfigSchema, harness_files
from chartreux.core.config.settings_catalog import (
    EDITABLE_SETTINGS,
    VISIBLE_SETTINGS,
    render_initial_user_config,
)
from chartreux.core.paths import GLOBAL_ENV_FILE, HISTORY_FILE


def test_bootstrap_exports_complete_editable_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_file = harness_files.get_harness_files_manager().user_config_file
    config_file.unlink(missing_ok=True)

    cli_mod.bootstrap_config_files()

    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert "# Interface" in content
    assert "# Project Context" in content
    assert "[project_context]" in content
    assert "[subagents]" in content
    assert "[session_logging]" in content
    parsed = tomllib.loads(content)
    defaults = ChartreuxConfigSchema.model_construct().model_dump(mode="json")
    for item in EDITABLE_SETTINGS:
        path = item.path.split(".")
        actual = parsed
        expected = defaults
        for component in path:
            actual = actual[component]
            expected = expected[component]
        assert actual == expected, item.path
        assert f"# {item.description}\n{path[-1]} = " in content
    assert ChartreuxConfigSchema.model_validate(parsed)
    captured = capsys.readouterr()
    assert "Created selections config" not in captured.out
    assert "Created selections config" in captured.err


def test_bootstrap_ignores_ambient_nested_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tomllib.loads(render_initial_user_config())
    monkeypatch.setenv("DEFAULT_COMMIT_COUNT", "99")
    monkeypatch.setenv("TIMEOUT_SECONDS", "8.5")
    monkeypatch.setenv("SESSION_PREFIX", "ambient")
    monkeypatch.setenv("ENABLED", "false")
    assert tomllib.loads(render_initial_user_config()) == baseline
    assert baseline["project_context"]["default_commit_count"] == 5
    assert baseline["session_logging"]["session_prefix"] == "session"
    assert "pins today's defaults explicitly" in render_initial_user_config()


def test_settings_help_describes_effective_limits() -> None:
    descriptions = {item.path: item.description for item in EDITABLE_SETTINGS}
    assert "0 disables automatic compaction" in descriptions["auto_compact_threshold"]
    assert "capped at 10 seconds" in descriptions["project_context.timeout_seconds"]
    assert "including enabled_tools matches" in descriptions["disabled_tools"]
    initial_config = render_initial_user_config()
    assert "# " + descriptions["disabled_tools"] in initial_config
    assert "ignored when enabled_tools" not in initial_config
    inventory_help = {item.path: item.description for item in VISIBLE_SETTINGS}
    assert (
        "disabled_tools still blocks matching tools"
        in inventory_help["inventory_tools"]
    )
    for category in ("skills", "agents"):
        assert (
            f"disabled_{category} is ignored then"
            in inventory_help[f"inventory_{category}"]
        )


def test_bootstrap_does_not_overwrite_existing_config() -> None:
    config_file = harness_files.get_harness_files_manager().user_config_file
    existing = 'theme = "dark"\n'
    config_file.write_text(existing, encoding="utf-8")

    cli_mod.bootstrap_config_files()

    assert config_file.read_text(encoding="utf-8") == existing


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
    # The file will hold API keys once filled in, so creation must restrict
    # it to owner-only right away rather than waiting for the next load.
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
