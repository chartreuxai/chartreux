from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import pytest

from chartreux.app_server import local as local_harness_mod
from chartreux.cli import cli as cli_mod
from chartreux.cli.textual_ui import app as textual_app_mod
from chartreux.core._usage_startup import create_startup_accounting_context


def _make_args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "initial_prompt": None,
        "prompt": None,
        "agent": None,
        "False": False,
        "enabled_tools": None,
        "disabled_tools": None,
        "add_dir": [],
        "trust": False,
        "worktree": None,
        "teleport": False,
        "continue_session": False,
        "resume": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def captured_startup(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Records what the interactive launch hands the TUI, without running it."""
    call: dict[str, Any] = {}

    def fake_run_textual_ui(**kwargs: Any) -> None:
        call.update(kwargs)
        return None

    real_harness = local_harness_mod.LocalHarness

    def recording_harness(options: Any) -> Any:
        call["harness_options"] = options
        return real_harness(options)

    monkeypatch.setattr(textual_app_mod, "run_textual_ui", fake_run_textual_ui)
    monkeypatch.setattr(local_harness_mod, "LocalHarness", recording_harness)
    return call


def _run(args: argparse.Namespace) -> None:
    cli_mod._run_interactive_mode(args=args, stdin_prompt=None)


def test_owned_harness_close_is_forwarded_to_textual_ui(
    captured_startup: dict[str, Any],
) -> None:
    _run(_make_args())

    connect = captured_startup["start_app_server"]
    close = captured_startup["close_app_server"]
    assert close.__self__ is connect.__self__
    assert close == connect.__self__.close


def test_trust_prompt_is_shown_by_default(captured_startup: dict[str, Any]) -> None:
    _run(_make_args())

    assert captured_startup["startup"].prompt_for_workspace_trust is True


def test_worktree_skips_the_trust_prompt(captured_startup: dict[str, Any]) -> None:
    # A worktree is a fresh directory every session, so prompting would ask
    # again on every launch for trust the session grants itself anyway.
    _run(_make_args(worktree=True))

    assert captured_startup["startup"].prompt_for_workspace_trust is False
    options = captured_startup["harness_options"]
    assert options.session_options.trust_workspace is True


def test_trust_flag_skips_the_trust_prompt(captured_startup: dict[str, Any]) -> None:
    _run(_make_args(trust=True))

    assert captured_startup["startup"].prompt_for_workspace_trust is False


def test_session_uses_cwd_captured_by_entrypoint(
    tmp_path: Path, captured_startup: dict[str, Any]
) -> None:
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()

    _run(_make_args(session_cwd=launch_dir))

    options = captured_startup["harness_options"]
    assert options.session_options.cwd == str(launch_dir.resolve())


@pytest.mark.parametrize("resume", [None, True, "saved-session"])
def test_startup_accounting_is_forwarded_independently_of_resume(
    tmp_path: Path, captured_startup: dict[str, Any], resume: str | bool | None
) -> None:
    context = asyncio.run(create_startup_accounting_context(tmp_path))
    _run(_make_args(startup_accounting=context, resume=resume))

    assert captured_startup["harness_options"].startup_accounting is context
    assert context.state == "available"
    assert captured_startup["startup"].show_resume_picker is (resume is True)
    assert captured_startup["startup"].resume_session_id == (
        resume if isinstance(resume, str) else None
    )


def test_experimental_harness_is_forwarded_to_local_harness(
    captured_startup: dict[str, Any],
) -> None:
    _run(_make_args(experimental_harness=True))
