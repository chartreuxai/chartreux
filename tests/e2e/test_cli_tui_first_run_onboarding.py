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
    if keys.startswith(("\x1b[A", "\x1b[B", "\x1b[F", "\x1b[H")) and keys.endswith(
        "\r"
    ):
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


@pytest.mark.timeout(90)
@pytest.mark.parametrize("customize", [False, True], ids=["auto-seeded", "customize"])
def test_empty_home_onboarding_reaches_first_streaming_turn(
    customize: bool,
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
        _send_and_wait_for_text(child, "\r", "Roles bound automatically", timeout=25)
        catalog_path = chartreux_home / "models.toml"
        # The model stage has already committed provider/model state; presets
        # remain staged until Finish. This checks transitions, not just headings.
        catalog = tomllib.loads(catalog_path.read_text(encoding="utf-8"))
        assert "onboarding-mock-model" in catalog["models"]
        assert "Local e2e provider" in catalog["providers"]
        summary_output = strip_ansi(captured.getvalue())
        assert "(*) Standalone" in summary_output
        assert "Edit Main Assistant preset" not in summary_output
        assert "Choose thinking for" not in summary_output
        assert "Choose model for" not in summary_output
        if customize:
            _send_and_wait_for_text(child, "\r", "Main assistant (@orchestrator)")
            _send_and_wait_for_text(child, "\r", "Edit Main Assistant preset")
            _send_and_wait_for_text(
                child, "\x1b[B\r", "Choose thinking for orchestrator"
            )
            _send_and_wait_for_text(child, "\x1b[H\r", "Thinking  off")
            _send_and_wait_for_text(child, "\t", "▸ Apply model and thinking")
            _send_and_wait_for_text(child, "\r", "Choose default presets")
            wait_for_rendered_text(
                child, captured, "Mode changes apply next session", timeout=10
            )
        _send_and_wait_for_text(child, "\t", "▸ Finish setup")
        _send_and_wait_for_text(child, "\r", "Setup complete", timeout=25)
        assert "Choose Exa, Brave or DuckDuckGo" not in strip_ansi(captured.getvalue())
        config_path = chartreux_home / "config.toml"
        assert catalog_path.is_file()
        env_path = chartreux_home / ".env"
        assert env_path.is_file()
        assert f"{api_key_env_var}='{api_key_value}'" in env_path.read_text(
            encoding="utf-8"
        )

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
        # Fresh setup keeps the standalone default without role questions.
        assert catalog.get("dispatch", {}).get("mode", "standalone") == "standalone"
        if customize:
            assert catalog["roles"]["orchestrator"]["thinking"] == "off"
        else:
            assert "Choose thinking for" not in strip_ansi(captured.getvalue())

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
