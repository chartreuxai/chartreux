from __future__ import annotations

import io
import os
from pathlib import Path
import time
import tomllib

import pexpect
import pytest

from tests import TESTS_ROOT
from tests.e2e.common import (
    ansi_tolerant_pattern,
    drain_child_output,
    send_ctrl_c_until_quit_confirmation,
    strip_ansi,
    wait_for_rendered_text,
    wait_for_request_count_while_draining_child_output,
)
from tests.e2e.mock_server import StreamingMockServer


def _send_and_wait_for_text(
    child: pexpect.spawn, keys: str, text: str, *, timeout: float = 10
) -> None:
    """Send keyboard input and wait for the resulting screen state."""
    if keys.startswith(("\x1b[A", "\x1b[B", "\x1b[F")) and keys.endswith("\r"):
        child.send(keys[:-1])
        time.sleep(0.1)
        child.send("\r")
    else:
        child.send(keys)
    try:
        child.expect(ansi_tolerant_pattern(text), timeout=timeout)
    except pexpect.TIMEOUT as exc:
        raise AssertionError(strip_ansi(str(child.before))[-2400:]) from exc


def _advance_welcome_with_keyboard(child: pexpect.spawn, timeout: float) -> None:
    """Press Enter until Welcome opens the Providers screen."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        child.send("\r")
        try:
            child.expect(ansi_tolerant_pattern("Add custom provider"), timeout=0.2)
        except pexpect.TIMEOUT:
            continue
        return
    raise AssertionError(
        f"Timed out advancing the welcome screen with Enter.\n{str(child.before)[-1200:]}"
    )


@pytest.mark.timeout(60)
def test_empty_home_onboarding_reaches_first_streaming_turn(
    streaming_mock_server: StreamingMockServer,
    e2e_workdir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the public first-run flow with terminal key presses only."""
    chartreux_home = tmp_path / "empty-chartreux-home"
    api_key_env_var = "E2E_ONBOARDING_API_KEY"
    api_key_value = "e2e-onboarding-dummy-key"

    assert not chartreux_home.exists()
    monkeypatch.delenv(api_key_env_var, raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("CHARTREUX_HOME", str(chartreux_home))
    monkeypatch.setenv("CHARTREUX_TEST_DISABLE_KEYRING", "1")
    monkeypatch.setenv("CHARTREUX_TEST_DISABLE_AUTO_TITLE", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("UV_OFFLINE", "1")

    captured = io.StringIO()
    child = pexpect.spawn(
        "uv",
        ["run", "chartreux", "--workdir", str(e2e_workdir)],
        cwd=str(TESTS_ROOT.parent),
        env=os.environ,
        encoding="utf-8",
        timeout=30,
        dimensions=(24, 80),
    )
    child.logfile_read = captured

    try:
        _advance_welcome_with_keyboard(child, timeout=10)

        wait_for_rendered_text(child, captured, "Provider Settings", timeout=10)
        wait_for_rendered_text(child, captured, "Add custom provider", timeout=10)
        # Root actions are a separate group from the suggested Mistral row.
        # Configure a custom provider for the local OpenAI-compatible server.
        _send_and_wait_for_text(child, "\t", "▸ Add custom provider")
        _send_and_wait_for_text(
            child, "\r", "New provider: unnamed — connection", timeout=25
        )
        _send_and_wait_for_text(child, "\r", "Provider Name *")
        _send_and_wait_for_text(
            child, "Local e2e provider\r", "Name  Local e2e provider"
        )
        _send_and_wait_for_text(child, "\x1b[B\r", "API Base *")
        _send_and_wait_for_text(
            child, streaming_mock_server.api_base + "\r", "API base"
        )
        child.send("\x1b[B\x1b[B")
        _send_and_wait_for_text(child, "\r", "Credential Env Var")
        _send_and_wait_for_text(child, api_key_env_var + "\r", "Credential env var")
        _send_and_wait_for_text(child, "\x1b[B\r", "API Key *")
        _send_and_wait_for_text(child, api_key_value + "\r", "Saved: Credential")
        _send_and_wait_for_text(child, "\t", "▸ Save and configure models")
        _send_and_wait_for_text(
            child, "\r", "Models for Local e2e provider", timeout=25
        )
        _send_and_wait_for_text(child, "\t", "▸ Retry discovery")
        _send_and_wait_for_text(child, "\x1b[B", "▸ Edit connection")
        _send_and_wait_for_text(child, "\x1b[B", "▸ Add model manually")
        _send_and_wait_for_text(child, "\r", "Model ID *")
        child.send("onboarding-mock-model\r")
        wait_for_rendered_text(child, captured, "onboarding-mock-model", timeout=10)
        # Closing the model editor restores its action-row opener.
        _send_and_wait_for_text(child, "\x1b[B", "▸ Save and add another provider")
        _send_and_wait_for_text(child, "\x1b[B", "▸ Save and continue to presets")
        _send_and_wait_for_text(child, "\r", "Choose default presets", timeout=25)
        catalog_path = chartreux_home / "models.toml"
        deadline = time.monotonic() + 8
        while not catalog_path.is_file() and time.monotonic() < deadline:
            drain_child_output(child, captured, idle_sleep=0.1)
        assert catalog_path.is_file(), strip_ansi(captured.getvalue())[-1200:]
        presets = (
            ("orchestrator", "Main Assistant"),
            ("large", "Large"),
            ("medium", "Medium"),
            ("small", "Small"),
        )
        for index, (role, title) in enumerate(presets):
            if index:
                _send_and_wait_for_text(
                    child, "\x1b[B", f"▸ {title} (@{role})", timeout=10
                )
            _send_and_wait_for_text(child, "\r", f"Edit {title} preset", timeout=10)
            _send_and_wait_for_text(child, "\r", f"Choose model for {role}", timeout=10)
            _send_and_wait_for_text(
                child, "\x1b[F\r", "Model  onboarding-mock-model", timeout=10
            )
            _send_and_wait_for_text(child, "\x1b[B", "▸ Thinking")
            _send_and_wait_for_text(
                child, "\r", f"Choose thinking for {role}", timeout=10
            )
            _send_and_wait_for_text(child, "\x1b[B", "▸ low")
            _send_and_wait_for_text(child, "\x1b[B", "▸ medium")
            _send_and_wait_for_text(child, "\r", f"Edit {title} preset", timeout=10)
            _send_and_wait_for_text(child, "\t", "▸ Apply model and thinking")
            _send_and_wait_for_text(child, "\r", "Choose default presets", timeout=10)
        _send_and_wait_for_text(child, "\t", "▸ Save presets and continue")
        _send_and_wait_for_text(
            child,
            "\r",
            "Choose Exa, Brave or DuckDuckGo to set up web search",
            timeout=25,
        )
        # From the provider list, Shift+Tab visits the form, then Skip for now.
        # Pace the keys so focus moves before Enter reaches the button.
        child.send("\x1b[Z")
        drain_child_output(child, captured, idle_sleep=0.1)
        child.send("\x1b[Z")
        drain_child_output(child, captured, idle_sleep=0.1)
        _send_and_wait_for_text(child, "\r", "Setup complete", timeout=25)
        config_path = chartreux_home / "config.toml"
        env_path = chartreux_home / ".env"
        catalog_path = chartreux_home / "models.toml"
        assert env_path.is_file()
        assert catalog_path.is_file()

        config = (
            tomllib.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        catalog = tomllib.loads(catalog_path.read_text(encoding="utf-8"))
        assert "active_model" not in config
        assert "theme" not in config
        assert config.get("theme", "auto") == "auto"
        provider = catalog["providers"]["Local e2e provider"]
        assert provider["api_base"] == streaming_mock_server.api_base
        assert provider["api_key_env_var"] == api_key_env_var
        deployment = catalog["models"]["onboarding-mock-model"]["deployments"][0]
        assert deployment["provider"] == "Local e2e provider"
        assert deployment["name"] == "onboarding-mock-model"
        assert set(catalog["roles"]) == {"orchestrator", "large", "medium", "small"}
        assert all(
            role["model"] == "onboarding-mock-model"
            for role in catalog["roles"].values()
        )
        assert f"{api_key_env_var}=" in env_path.read_text(encoding="utf-8")

        # Wait for the main TUI to finish starting up before typing, otherwise
        # the message keystrokes arrive while the app is still entering its
        # input screen and are lost.
        wait_for_rendered_text(child, captured, "F1 Help · /help", timeout=25)
        child.send("Greet from onboarding")
        child.send("\r")
        wait_for_request_count_while_draining_child_output(
            child,
            captured,
            lambda: len(streaming_mock_server.requests),
            expected_count=1,
            timeout=10,
        )
        child.expect(ansi_tolerant_pattern("Hello from mock server"), timeout=10)

        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    finally:
        if child.isalive():
            child.terminate(force=True)
        if not child.closed:
            child.close()

    request_payload = streaming_mock_server.requests[-1]
    assert request_payload.get("model") == "onboarding-mock-model"
    messages = request_payload.get("messages")
    assert messages is not None
    assert any(
        message.get("role") == "user"
        and message.get("content") == "Greet from onboarding"
        for message in messages
    )
