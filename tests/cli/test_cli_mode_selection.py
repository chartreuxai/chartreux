from __future__ import annotations

import argparse
import io

import pytest

from chartreux.cli import cli as cli_mod
from chartreux.core.config import MissingAPIKeyError


class _PipedStdin(io.StringIO):
    def isatty(self) -> bool:
        return False


def _make_args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "initial_prompt": None,
        "prompt": None,
        "max_turns": None,
        "max_price": None,
        "max_tokens": None,
        "enabled_tools": None,
        "disabled_tools": None,
        "output": "text",
        "setup": False,
        "worktree": None,
        "add_dir": [],
        "trust": False,
        "continue_session": False,
        "resume": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_piped_prompt_selects_programmatic_mode_before_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr("sys.stdin", _PipedStdin("hello from pipe\n"))
    monkeypatch.setattr(cli_mod, "load_dotenv_values", lambda: calls.append("dotenv"))
    monkeypatch.setattr(
        cli_mod, "bootstrap_config_files", lambda: calls.append("bootstrap")
    )
    monkeypatch.setattr(cli_mod, "load_config_orchestrator_or_exit", lambda: object())
    monkeypatch.setattr(
        cli_mod,
        "require_api_key_or_onboard",
        lambda _orchestrator, *, interactive: calls.append((
            "credentials",
            interactive,
        )),
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_programmatic_mode",
        lambda *, args, stdin_prompt: calls.append(("programmatic", stdin_prompt)),
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_interactive_mode",
        lambda *, args, stdin_prompt: calls.append(("interactive", stdin_prompt)),
    )

    cli_mod.run_cli(_make_args())

    assert calls == [
        "dotenv",
        "bootstrap",
        ("credentials", False),
        ("programmatic", "hello from pipe"),
    ]


def test_empty_pipe_stays_interactive_after_mode_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr("sys.stdin", _PipedStdin(" \n"))
    monkeypatch.setattr(cli_mod, "load_dotenv_values", lambda: None)
    monkeypatch.setattr(cli_mod, "bootstrap_config_files", lambda: None)
    monkeypatch.setattr(cli_mod, "load_config_orchestrator_or_exit", lambda: object())
    monkeypatch.setattr(
        cli_mod,
        "require_api_key_or_onboard",
        lambda _orchestrator, *, interactive: calls.append((
            "credentials",
            interactive,
        )),
    )
    monkeypatch.setattr(cli_mod, "has_usable_terminal", lambda: True)
    monkeypatch.setattr(
        cli_mod,
        "restore_interactive_stdin",
        lambda *, stdin_isatty: calls.append(("restore", stdin_isatty)),
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_interactive_mode",
        lambda *, args, stdin_prompt: calls.append(("interactive", stdin_prompt)),
    )

    cli_mod.run_cli(_make_args())

    assert calls == [("restore", False), ("credentials", True), ("interactive", None)]


def test_setup_wins_over_piped_programmatic_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("sys.stdin", _PipedStdin("hello from pipe\n"))
    monkeypatch.setattr(cli_mod, "load_dotenv_values", lambda: calls.append("dotenv"))
    monkeypatch.setattr(
        cli_mod, "bootstrap_config_files", lambda: calls.append("bootstrap")
    )
    monkeypatch.setattr(cli_mod, "load_config_orchestrator_or_exit", lambda: object())
    monkeypatch.setattr(
        "chartreux.setup.onboarding.run_onboarding",
        lambda *, orchestrator: calls.append("onboarding"),
    )
    monkeypatch.setattr(
        cli_mod,
        "_run_programmatic_mode",
        lambda **_kwargs: calls.append("programmatic"),
    )
    monkeypatch.setattr(cli_mod, "has_usable_terminal", lambda: True)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.run_cli(_make_args(setup=True))

    assert exc_info.value.code == 0
    assert calls == ["dotenv", "bootstrap", "onboarding"]


def test_implicit_start_without_tty_does_not_run_onboarding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Config:
        def require_active_provider_api_key(self) -> None:
            raise MissingAPIKeyError("MISTRAL_API_KEY", "mistral")

    class _Orchestrator:
        config = _Config()

    onboarding_called = False

    def fail_onboarding(*_args: object, **_kwargs: object) -> None:
        nonlocal onboarding_called
        onboarding_called = True

    monkeypatch.setattr(cli_mod, "load_dotenv_values", lambda: None)
    monkeypatch.setattr(cli_mod, "bootstrap_config_files", lambda: None)
    monkeypatch.setattr(cli_mod, "get_prompt_from_stdin", lambda: None)
    monkeypatch.setattr(cli_mod, "restore_interactive_stdin", lambda **_kwargs: None)
    monkeypatch.setattr(cli_mod, "has_usable_terminal", lambda: False)
    monkeypatch.setattr(
        cli_mod, "load_config_orchestrator_or_exit", lambda: _Orchestrator()
    )
    monkeypatch.setattr("chartreux.setup.onboarding.run_onboarding", fail_onboarding)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.run_cli(_make_args())

    assert exc_info.value.code == 1
    assert onboarding_called is False
    assert "run `chartreux --setup` once interactively" in capsys.readouterr().err


def test_setup_without_tty_does_not_run_onboarding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    onboarding_called = False

    def fail_onboarding(*_args: object, **_kwargs: object) -> None:
        nonlocal onboarding_called
        onboarding_called = True

    monkeypatch.setattr(cli_mod, "load_dotenv_values", lambda: None)
    monkeypatch.setattr(cli_mod, "bootstrap_config_files", lambda: None)
    monkeypatch.setattr(cli_mod, "has_usable_terminal", lambda: False)
    monkeypatch.setattr("chartreux.setup.onboarding.run_onboarding", fail_onboarding)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.run_cli(_make_args(setup=True))

    assert exc_info.value.code == 1
    assert onboarding_called is False
    assert "Interactive setup requires a terminal" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("prompt", "initial_prompt", "expected"),
    [("explicit", "positional", "explicit"), (None, "positional", "positional")],
)
def test_explicit_prompt_takes_precedence_over_piped_prompt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prompt: str | None,
    initial_prompt: str | None,
    expected: str,
) -> None:
    monkeypatch.setattr(
        "chartreux.cli.programmatic.run_programmatic", lambda **kwargs: kwargs["prompt"]
    )
    args = _make_args(prompt=prompt, initial_prompt=initial_prompt)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod._run_programmatic_mode(args, "piped")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"{expected}\n"
