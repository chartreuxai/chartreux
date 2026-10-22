"""Identity settings: invocation precedence, resolver choices, and doctor.

Covers the package-13 settings surface end to end: CLI > environment > config >
default precedence for the three ``[agent]`` selections with omission
distinguished from explicit selection, resolver-sourced settings choices,
and doctor's offline identity/resource/retirement diagnostics.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterator
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from chartreux.cli import cli as cli_mod
import chartreux.cli.doctor_command as doctor
from chartreux.cli.entrypoint import parse_arguments


def _parse(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> argparse.Namespace:
    monkeypatch.setattr("sys.argv", ["chartreux", *argv])
    return parse_arguments()


def _clear_selection_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env in cli_mod.AGENT_SELECTION_ENV.values():
        monkeypatch.delenv(env, raising=False)


@pytest.fixture(autouse=True)
def _restore_selection_env() -> Iterator[None]:
    """Restore the process environment around every test in this module.

    ``apply_invocation_selections`` publishes explicit CLI selections into
    ``os.environ`` on purpose (that is how a selection outranks the environment
    and config layers at startup), so tests exercising it mutate the worker's
    process environment. Snapshot-and-restore keeps that mutation from leaking
    into later tests in the same xdist worker.
    """
    saved = {env: os.environ.get(env) for env in cli_mod.AGENT_SELECTION_ENV.values()}
    yield
    for env, value in saved.items():
        if value is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = value


def test_invocation_selection_flags_parse_and_omit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(monkeypatch, [])
    assert args.identity is None
    assert args.workflow is None
    assert args.orchestration_strategy is None

    args = _parse(
        monkeypatch,
        [
            "--identity",
            "orchestrator",
            "--workflow",
            "ceremony",
            "--orchestration-strategy",
            "default",
        ],
    )
    assert args.identity == "orchestrator"
    assert args.workflow == "ceremony"
    assert args.orchestration_strategy == "default"


def test_apply_invocation_selections_publishes_only_explicit_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_selection_env(monkeypatch)
    cli_mod.apply_invocation_selections(
        argparse.Namespace(identity=None, workflow=None, orchestration_strategy=None)
    )
    # Omission publishes nothing: the environment and config stay in charge.
    assert not any(os.environ.get(env) for env in cli_mod.AGENT_SELECTION_ENV.values())

    cli_mod.apply_invocation_selections(
        argparse.Namespace(
            identity="researcher", workflow=None, orchestration_strategy="default"
        )
    )
    assert os.environ["CHARTREUX_AGENT__IDENTITY"] == "researcher"
    assert os.environ["CHARTREUX_AGENT__ORCHESTRATION_STRATEGY"] == "default"
    assert "CHARTREUX_AGENT__WORKFLOW" not in os.environ


def test_cli_selection_outranks_inherited_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHARTREUX_AGENT__IDENTITY", "reviewer")
    cli_mod.apply_invocation_selections(
        argparse.Namespace(
            identity="advisor", workflow=None, orchestration_strategy=None
        )
    )
    assert os.environ["CHARTREUX_AGENT__IDENTITY"] == "advisor"


@pytest.mark.parametrize("mode", ["interactive", "programmatic"])
def test_run_cli_carries_all_three_selections_into_both_startup_modes(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    captured: dict[str, dict[str, str]] = {}

    def capture(args: argparse.Namespace, stdin_prompt: Any = None) -> None:
        captured["env"] = {
            key: os.environ.get(key, "") for key in cli_mod.AGENT_SELECTION_ENV.values()
        }

    for runner in ("_run_interactive_mode", "_run_programmatic_mode"):
        monkeypatch.setattr(cli_mod, runner, capture)
    monkeypatch.setattr(cli_mod, "bootstrap_config_files", Mock())
    monkeypatch.setattr(cli_mod, "get_prompt_from_stdin", lambda: None)
    monkeypatch.setattr(cli_mod, "require_api_key_or_onboard", lambda *a, **k: None)
    monkeypatch.setattr(
        cli_mod, "load_config_orchestrator_or_exit", lambda: Mock(config=Mock())
    )
    _clear_selection_env(monkeypatch)

    args = _parse(
        monkeypatch,
        [
            "--identity",
            "orchestrator",
            "--workflow",
            "ceremony",
            "--orchestration-strategy",
            "default",
            *(("--prompt", "hi") if mode == "programmatic" else ()),
        ],
    )
    cli_mod.run_cli(args)
    assert captured["env"] == {
        "CHARTREUX_AGENT__IDENTITY": "orchestrator",
        "CHARTREUX_AGENT__WORKFLOW": "ceremony",
        "CHARTREUX_AGENT__ORCHESTRATION_STRATEGY": "default",
    }


async def _identity_from_config() -> str:
    from chartreux.core.config.default_orchestrator import build_default_orchestrator

    return (await build_default_orchestrator()).config.agent.identity


def test_selection_precedence_cli_env_config_default(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    (config_dir / "config.toml").write_text(
        '[agent]\nidentity = "reviewer"\n', encoding="utf-8"
    )
    _clear_selection_env(monkeypatch)

    # Config beats the shipped default.
    assert asyncio.run(_identity_from_config()) == "reviewer"

    # Environment beats config; the omitted selections stay omitted.
    monkeypatch.setenv("CHARTREUX_AGENT__IDENTITY", "researcher")
    assert asyncio.run(_identity_from_config()) == "researcher"

    # The CLI flag outranks the environment.
    cli_mod.apply_invocation_selections(
        argparse.Namespace(
            identity="advisor", workflow=None, orchestration_strategy=None
        )
    )
    assert asyncio.run(_identity_from_config()) == "advisor"

    # An omitted CLI selection falls back to the next layer.
    monkeypatch.delenv("CHARTREUX_AGENT__IDENTITY")
    assert asyncio.run(_identity_from_config()) == "reviewer"


def test_invalid_invocation_selection_fails_config_load_loudly(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    from chartreux.core.config.layer import LayerImplementationError

    monkeypatch.setenv("CHARTREUX_AGENT__IDENTITY", "!!invalid!!")
    with pytest.raises(LayerImplementationError) as exc_info:
        asyncio.run(_identity_from_config())
    assert "agent.identity" in str(exc_info.value.__cause__)


def test_agent_selection_choices_come_from_the_resolver(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    from chartreux.app_server._resources import agent_selection_choices

    prompts = config_dir / "prompts"
    prompts.mkdir()
    (prompts / "my-scribe.md").write_text("Custom identity body")
    (prompts / "empty.md").write_text("   ")
    workflows = Path(os.environ["CHARTREUX_HOME"]) / "workflows"
    workflows.mkdir()
    (workflows / "solo.md").write_text("Custom workflow body")

    choices = agent_selection_choices()
    assert choices["agent.identity"] == (
        "advisor",
        "coder",
        "orchestrator",
        "researcher",
        "reviewer",
        "my-scribe",
    )
    assert "empty" not in choices["agent.identity"]
    assert choices["agent.workflow"] == ("ceremony", "minimal", "solo")
    assert choices["agent.orchestration_strategy"] == ("none", "default")


def test_settings_catalog_wire_supplies_identity_choices() -> None:
    from chartreux.app_server._resources import _settings_catalog_wire

    catalog = {item.path: item for item in _settings_catalog_wire()}
    assert "system_prompt_id" not in catalog
    assert catalog["agent.identity"].choices == (
        "advisor",
        "coder",
        "orchestrator",
        "researcher",
        "reviewer",
    )
    assert catalog["agent.workflow"].choices == ("ceremony", "minimal")
    assert catalog["agent.orchestration_strategy"].choices == ("none", "default")
    assert catalog["inventory_identities"].inventory == "identities"
    assert catalog["inventory_identities"].control == "toggle_inventory"
    assert "agent_paths" not in catalog
    assert "enabled_agents" not in catalog
    assert catalog["agent.identity"].timing == "next-conversation"


def test_denied_roster_states_block_without_granting() -> None:
    from chartreux.app_server._resources import _inventory_item_states

    states = _inventory_item_states(
        {"identities": ["coder", "researcher"]},
        {"tools.task.denylist": ["researcher", "co*"]},
    )
    assert states["identities"]["coder"].effective is False
    assert states["identities"]["coder"].pattern_driven is True
    assert states["identities"]["researcher"].effective is False
    assert states["identities"]["researcher"].pattern_driven is False
    assert states["identities"]["coder"].default_effective is True
    all_on = _inventory_item_states({"identities": ["coder"]}, {})
    assert all_on["identities"]["coder"].effective is True
    assert all_on["identities"]["coder"].default_effective is True


@pytest.mark.asyncio
async def test_doctor_reports_unresolvable_selection_with_actionable_error(
    config_dir: Path,
) -> None:
    from chartreux.core.config.harness_files import get_harness_files_manager

    (config_dir / "config.toml").write_text(
        '[agent]\nidentity = "user:missing"\n', encoding="utf-8"
    )
    config, _manager, checks = await doctor._load_config()
    selection = next(check for check in checks if check.name == "agent selections")
    assert selection.status == "fail"
    assert "missing" in selection.reason
    assert "config.toml" in selection.reason
    assert (
        doctor._agent_selections_check(config, get_harness_files_manager()).status
        == "fail"
    )


@pytest.mark.asyncio
async def test_doctor_selections_pass_on_defaults(config_dir: Path) -> None:
    _, _, checks = await doctor._load_config()
    selection = next(check for check in checks if check.name == "agent selections")
    assert selection.status == "pass"


def test_doctor_flags_retired_settings_with_successors(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """All four retired config fields fail load together, each named with its
    successor: doctor surfaces the loader's source-aware validation message.
    """
    (config_dir / "config.toml").write_text(
        'system_prompt_id = "worker"\n'
        'agent_paths = ["retired-value-abc"]\n'
        'enabled_agents = ["worker"]\n'
        'disabled_agents = ["advisor"]\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        doctor.run_doctor_cli(["--json"])
    assert exc.value.code == 1
    result = json.loads(capsys.readouterr().out)
    checks = {check["name"]: check for check in result["checks"]}
    check = checks["config"]
    assert check["status"] == "fail"
    reason = check["reason"]
    for field in (
        "system_prompt_id",
        "agent_paths",
        "enabled_agents",
        "disabled_agents",
    ):
        assert f"{field} is retired" in reason
    assert "identity composition" in reason
    assert "identity denylist" in reason
    # Source-aware: the failing source file is named.
    assert "config.toml" in reason
    # The supplied values are never echoed (the source locator may name the
    # config.toml path, so the probe value must be path-independent).
    assert "retired-value-abc" not in json.dumps(result)


@pytest.mark.asyncio
async def test_doctor_reports_no_retired_inputs_on_clean_config(
    config_dir: Path,
) -> None:
    _, manager, _ = await doctor._load_config()
    assert doctor._retired_inputs_checks(manager) == []
