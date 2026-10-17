"""First-run onboarding journeys driven through a real spawned CLI.

The retained scenarios are this file's PTY-only population (design v3 §6.11):
the public first-run onboarding flow driven by terminal key presses only, end
to end through the first streaming turn, with the on-disk catalog and .env
artifacts asserted, plus a pure unit test of the module-local highlight
helper. The mirrored journeys (env-key skip, interactive-key materialization,
welcome decline, sparse-overlay merge, multi-key credential selection,
invalid-credential reporting) are retired with acceptance-layer twins; see
tests/e2e/SURVIVAL_MAP.md (stage 12d-xii).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys
import time
import tomllib
from typing import cast

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


def _highlighted_row(captured: io.StringIO) -> str | None:
    """The most recently rendered highlighted row, if the screen shows one."""
    text = strip_ansi(captured.getvalue())
    start = text.rfind("▸ ")
    if start < 0:
        return None
    return text[start + 2 :].split("\n", 1)[0].rstrip()


def _wait_for_moved_highlight(
    child: pexpect.spawn, captured: io.StringIO, highlighted: str | None, timeout: float
) -> None:
    """Wait until the highlight marker moves to another row.

    Enter must not race the arrow key: the moved highlight has to render
    first, the same way the autocomplete popup gates its Enter.
    """
    start = time.monotonic()
    saw_parseable_marker = False
    last_row: str | None = None
    while time.monotonic() - start < timeout:
        row = _highlighted_row(captured)
        if row is not None:
            saw_parseable_marker = True
            last_row = row
            if row != highlighted:
                return
        if drain_child_output(child, captured):
            raise AssertionError(
                "Child exited while waiting for the highlight to move. "
                f"Last observed row: {last_row!r}."
            )
    if not child.isalive():
        case = "child exited"
    elif not saw_parseable_marker:
        case = "no parseable highlight marker appeared in any captured frame"
    else:
        case = f"highlight marker stayed on its original row {highlighted!r}"
    raise AssertionError(
        "Timed out waiting for the highlight to move after an arrow keypress: "
        f"{case}. Last observed row: {last_row!r}."
    )


def test_wait_for_moved_highlight_tolerates_marker_absence_before_movement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient unparsable frame does not block later observed movement."""
    rows = iter(("Original row", None, "Moved row"))
    monkeypatch.setattr(
        sys.modules[__name__], "_highlighted_row", lambda _captured: next(rows)
    )
    monkeypatch.setattr(
        sys.modules[__name__], "drain_child_output", lambda _child, _captured: False
    )
    _wait_for_moved_highlight(
        cast(pexpect.spawn, type("Child", (), {"isalive": lambda _self: True})()),
        io.StringIO(),
        "Original row",
        timeout=1,
    )

    # The helper consumed the unparsable frame and returned only on the
    # actual move; an early return on the None frame would leave a row.
    assert list(rows) == []


def _send_and_wait_for_text(
    child: pexpect.spawn,
    captured: io.StringIO,
    keys: str,
    text: str,
    *,
    timeout: float = 10,
) -> None:
    """Send keyboard input and wait for the resulting screen state."""
    if keys.startswith(("\x1b[A", "\x1b[B", "\x1b[F", "\x1b[H")) and keys.endswith(
        "\r"
    ):
        # The arrow moves the highlight; wait for the moved row to render
        # before pressing Enter so Enter activates the intended row.
        drain_child_output(child, captured)
        highlighted = _highlighted_row(captured)
        child.send(keys[:-1])
        _wait_for_moved_highlight(child, captured, highlighted, timeout)
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
        _send_and_wait_for_text(child, captured, "\t", "▸ Add custom provider")
        _send_and_wait_for_text(
            child, captured, "\r", "New provider: unnamed — connection", timeout=25
        )
        _send_and_wait_for_text(child, captured, "\r", "Provider Name *")
        _send_and_wait_for_text(
            child, captured, "Local e2e provider\r", "Name  Local e2e provider"
        )
        _send_and_wait_for_text(child, captured, "\x1b[B\r", "API Base *")
        _send_and_wait_for_text(
            child, captured, streaming_mock_server.api_base + "\r", "API base"
        )
        child.send("\x1b[B\x1b[B")
        _send_and_wait_for_text(child, captured, "\r", "Credential Env Var")
        _send_and_wait_for_text(
            child, captured, api_key_env_var + "\r", "Credential env var"
        )
        _send_and_wait_for_text(child, captured, "\x1b[B\r", "API Key *")
        _send_and_wait_for_text(
            child, captured, api_key_value + "\r", "Saved: Credential"
        )
        _send_and_wait_for_text(child, captured, "\t", "▸ Save and configure models")
        _send_and_wait_for_text(
            child, captured, "\r", "Models for Local e2e provider", timeout=25
        )
        _send_and_wait_for_text(child, captured, "\t", "▸ Retry discovery")
        _send_and_wait_for_text(child, captured, "\x1b[B", "▸ Edit connection")
        _send_and_wait_for_text(child, captured, "\x1b[B", "▸ Add model manually")
        _send_and_wait_for_text(child, captured, "\r", "Model ID *")
        child.send("onboarding-mock-model\r")
        wait_for_rendered_text(child, captured, "onboarding-mock-model", timeout=10)
        # Closing the model editor restores its action-row opener.
        _send_and_wait_for_text(
            child, captured, "\x1b[B", "▸ Save and add another provider"
        )
        _send_and_wait_for_text(
            child, captured, "\x1b[B", "▸ Save and continue to presets"
        )
        _send_and_wait_for_text(
            child, captured, "\r", "Roles bound automatically", timeout=25
        )
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
            _send_and_wait_for_text(
                child, captured, "\r", "Main assistant (@orchestrator)"
            )
            _send_and_wait_for_text(child, captured, "\r", "Edit Main Assistant preset")
            _send_and_wait_for_text(
                child, captured, "\x1b[B\r", "Choose thinking for orchestrator"
            )
            _send_and_wait_for_text(child, captured, "\x1b[H\r", "Thinking  off")
            _send_and_wait_for_text(child, captured, "\t", "▸ Apply model and thinking")
            _send_and_wait_for_text(child, captured, "\r", "Choose default presets")
            wait_for_rendered_text(
                child, captured, "Mode changes apply next session", timeout=10
            )
        _send_and_wait_for_text(child, captured, "\t", "▸ Finish setup")
        _send_and_wait_for_text(child, captured, "\r", "Setup complete", timeout=25)
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
        assert set(catalog["roles"]) == {"orchestrator", "worker", "scout", "heavy"}
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
