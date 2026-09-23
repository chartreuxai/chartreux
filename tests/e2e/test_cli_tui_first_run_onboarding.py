from __future__ import annotations

import io
import os
from pathlib import Path
import time
import tomllib

import pexpect
import pytest

from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
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
    child.send(keys)
    child.expect(ansi_tolerant_pattern(text), timeout=timeout)


def _select_option_with_keyboard(
    child: pexpect.spawn, captured: io.StringIO, option_text: str
) -> None:
    """Select a newly added model from the fresh-home active-model picker."""
    wait_for_rendered_text(child, captured, "Labels identify deployments;", timeout=10)
    assert option_text in strip_ansi(captured.getvalue())
    # A fresh home highlights Default first. Canonical model rows are sorted,
    # followed by role rows, so derive this model's offset from the shipped set.
    model_names = sorted((*SHIPPED_CATALOG.models, option_text))
    child.send("j" * (model_names.index(option_text) + 1))
    drain_child_output(child, idle_sleep=0.03)
    child.send("\r")


def _advance_welcome_with_keyboard(child: pexpect.spawn, timeout: float) -> None:
    """Press Enter on output events until the typed welcome accepts it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        child.send("\r")
        try:
            child.expect(
                ansi_tolerant_pattern("Select your preferred theme"), timeout=0.2
            )
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
        child.send("\r")

        wait_for_rendered_text(child, captured, "Choose a provider", timeout=10)
        _send_and_wait_for_text(child, "\t\r", "Preset", timeout=25)

        _send_and_wait_for_text(child, "\t\t", "Generic OpenAI-style")
        child.sendcontrol("a")
        child.sendcontrol("k")
        child.send("Local e2e provider")
        child.send("\t")
        child.sendcontrol("a")
        child.sendcontrol("k")
        child.send(streaming_mock_server.api_base)
        child.send("\t\t")
        child.sendcontrol("a")
        child.sendcontrol("k")
        child.send(api_key_env_var)
        _send_and_wait_for_text(child, "\t\t\r", "Credential")

        child.send("\t")
        child.send(api_key_value)
        _send_and_wait_for_text(child, "\t\r", "Manual entry")

        _send_and_wait_for_text(child, "\t\t\r", "Enter the provider wire name")
        child.send("\x1b[Z")
        child.send("onboarding-mock-model")
        _send_and_wait_for_text(child, "\t\t\r", "Selected models")
        _send_and_wait_for_text(child, "\t" * 6 + "\r", "Provider saved.")
        time.sleep(0.1)
        _send_and_wait_for_text(child, "\r", "Choose Active Model")
        time.sleep(0.1)

        _select_option_with_keyboard(child, captured, "onboarding-mock-model")
        wait_for_rendered_text(child, captured, "Setup complete", timeout=25)

        config_path = chartreux_home / "config.toml"
        env_path = chartreux_home / ".env"
        catalog_path = chartreux_home / "models.toml"
        assert config_path.is_file()
        assert env_path.is_file()
        assert catalog_path.is_file()

        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        catalog = tomllib.loads(catalog_path.read_text(encoding="utf-8"))
        assert config["active_model"] == "onboarding-mock-model"
        assert config["theme"] == "auto"
        provider = catalog["providers"]["local-e2e-provider/default"]
        assert provider["api_base"] == streaming_mock_server.api_base
        assert provider["api_key_env_var"] == api_key_env_var
        deployment = catalog["models"]["onboarding-mock-model"]["deployments"][0]
        assert deployment["provider"] == "local-e2e-provider/default"
        assert deployment["name"] == "onboarding-mock-model"
        assert f"{api_key_env_var}=" in env_path.read_text(encoding="utf-8")

        # Wait for the main TUI to finish starting up before typing, otherwise
        # the message keystrokes arrive while the app is still entering its
        # input screen and are lost.
        wait_for_rendered_text(
            child, captured, "Type /help for more information", timeout=25
        )
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
