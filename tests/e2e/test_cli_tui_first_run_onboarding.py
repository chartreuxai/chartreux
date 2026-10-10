from __future__ import annotations

import io
import os
from pathlib import Path
import socket
import sys
import time
import tomllib
from typing import cast

import pexpect
import pytest

from chartreux.core.model_catalog.materialization import render_default_template
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


def _prepare_first_run_home(
    monkeypatch: pytest.MonkeyPatch, chartreux_home: Path
) -> None:
    """Isolate the child CLI to a Chartreux home with test-safe settings."""
    for key in ("MISTRAL_API_KEY", "EXA_API_KEY", "BRAVE_SEARCH_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CHARTREUX_HOME", str(chartreux_home))
    monkeypatch.setenv("CHARTREUX_TEST_DISABLE_KEYRING", "1")
    monkeypatch.setenv("CHARTREUX_TEST_DISABLE_AUTO_TITLE", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("UV_OFFLINE", "1")


def _spawn_first_run(e2e_workdir: Path) -> tuple[pexpect.spawn, io.StringIO]:
    """Spawn the CLI at 80x24 for a first-run journey."""
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
    return child, captured


def _stop_child(child: pexpect.spawn) -> None:
    if child.isalive():
        child.terminate(force=True)
    if not child.closed:
        child.close()


def _quit_and_wait(child: pexpect.spawn, captured: io.StringIO) -> None:
    send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
    child.expect(pexpect.EOF, timeout=10)


def _open_provider_settings(
    child: pexpect.spawn, captured: io.StringIO, *, timeout: float = 15
) -> None:
    """Open Provider Settings from the chat input.

    The slash-command autocomplete must render before Enter submits, so the
    command text and Enter are sent separately.
    """
    child.send("/providers")
    wait_for_rendered_text(
        child, captured, "Open Provider Settings to add or manage", timeout=10
    )
    child.send("\r")
    wait_for_rendered_text(child, captured, "Add custom provider", timeout=timeout)


def _close_provider_settings(child: pexpect.spawn, captured: io.StringIO) -> None:
    """Close Provider Settings and wait for the dismissal notice.

    The first Escape can land while the freshly opened screen is still busy,
    so Escape is retried until the workbench actually dismisses.
    """
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        child.send("\x1b")
        inner = time.monotonic() + 3
        while time.monotonic() < inner:
            if "Provider Settings closed" in strip_ansi(captured.getvalue()):
                return
            drain_child_output(child, captured)
            time.sleep(0.05)
    raise AssertionError("Provider Settings did not close after Escape.")


def _write_sparse_mistral_overlay(
    chartreux_home: Path, api_base: str
) -> tuple[Path, str]:
    """Write a user models.toml overriding only the shipped Mistral base URL."""
    chartreux_home.mkdir(parents=True, exist_ok=True)
    sparse_overlay = f'[providers.mistral]\napi_base = "{api_base}"\n'
    catalog_path = chartreux_home / "models.toml"
    catalog_path.write_text(sparse_overlay, encoding="utf-8")
    return catalog_path, sparse_overlay


def _unused_local_port() -> int:
    """A local TCP port with nothing listening on it."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.mark.timeout(90)
def test_env_key_first_run_skips_onboarding_without_creating_a_catalog(
    e2e_workdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolvable key in the environment starts the TUI directly.

    First-run class 1: MISTRAL_API_KEY present in the environment means no
    onboarding flow, no materialized models.toml, and no graduation nudge; the
    shipped catalog serves both default models immediately.
    """
    chartreux_home = tmp_path / "env-key-home"
    assert not chartreux_home.exists()
    _prepare_first_run_home(monkeypatch, chartreux_home)
    monkeypatch.setenv("MISTRAL_API_KEY", "e2e-env-key")

    child, captured = _spawn_first_run(e2e_workdir)
    try:
        wait_for_rendered_text(child, captured, "F1 Help · /help", timeout=25)
        output = strip_ansi(captured.getvalue())
        assert "Let's get you started" not in output
        assert "Provider Settings" not in output
        # The startup banner already counts both shipped default models.
        assert "2 models" in output
        assert "zai-glm-5-3" in output

        # Both shipped default models are usable without any onboarding.
        _open_provider_settings(child, captured)
        wait_for_rendered_text(child, captured, "Key Set", timeout=15)
        wait_for_rendered_text(child, captured, "2 runnable models", timeout=15)
        _close_provider_settings(child, captured)
        _quit_and_wait(child, captured)
    finally:
        _stop_child(child)

    # First run with a resolvable key creates no catalog file.
    assert not (chartreux_home / "models.toml").exists()
    output = strip_ansi(captured.getvalue())
    assert "More models can share the work" not in output


@pytest.mark.timeout(120)
def test_interactive_mistral_key_onboarding_materializes_the_default_catalog(
    e2e_workdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entering the Mistral key interactively completes onboarding.

    First-run class 2: no key in the environment, the key entered in
    onboarding. Onboarding completes, the role summary binds both default
    models, the shipped default template is materialized, and both default
    models are usable in the running TUI.
    """
    chartreux_home = tmp_path / "interactive-key-home"
    assert not chartreux_home.exists()
    _prepare_first_run_home(monkeypatch, chartreux_home)
    # The shipped Mistral preset lists its models over HTTPS. Point the child's
    # HTTPS egress at a closed local port so the test never reaches
    # api.mistral.ai: the preset, the keystrokes, and the saved catalog are
    # unchanged (a failed discovery stages no pending models), and the mock
    # server is reached over plain HTTP, which an HTTPS proxy does not touch.
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{_unused_local_port()}")
    api_key_value = "e2e-interactive-key"

    child, captured = _spawn_first_run(e2e_workdir)
    try:
        _advance_welcome_with_keyboard(child, timeout=10)
        wait_for_rendered_text(child, captured, "Provider Settings", timeout=10)
        wait_for_rendered_text(child, captured, "Add custom provider", timeout=10)
        # The shipped Mistral row is the suggested setup and shows the missing
        # credential badge.
        wait_for_rendered_text(child, captured, "Key Required", timeout=10)
        _send_and_wait_for_text(
            child, captured, "\r", "New provider: mistral — connection", timeout=25
        )
        # Navigate the connection fields to the API key row and enter the key.
        _send_and_wait_for_text(
            child, captured, "\x1b[B\x1b[B\x1b[B\x1b[B", "▸ API key"
        )
        _send_and_wait_for_text(child, captured, "\r", "API Key *")
        _send_and_wait_for_text(
            child, captured, api_key_value + "\r", "Saved: Credential"
        )
        _send_and_wait_for_text(child, captured, "\t", "▸ Save and configure models")
        _send_and_wait_for_text(child, captured, "\r", "Models for mistral", timeout=25)
        # Both shipped default models are listed for the provider.
        wait_for_rendered_text(child, captured, "zai-glm-5-3", timeout=10)
        wait_for_rendered_text(child, captured, "mistral-large-4", timeout=10)
        # Continue to the role summary without changing the model selections.
        _send_and_wait_for_text(child, captured, "\t", "▸ Retry discovery")
        _send_and_wait_for_text(
            child,
            captured,
            "\x1b[B\x1b[B\x1b[B\x1b[B",
            "▸ Save and continue to presets",
        )
        _send_and_wait_for_text(
            child, captured, "\r", "Roles bound automatically", timeout=25
        )
        wait_for_rendered_text(
            child,
            captured,
            "Roles bound automatically to glm-5-3, mistral-large-4",
            timeout=10,
        )
        _send_and_wait_for_text(child, captured, "\t", "▸ Finish setup")
        _send_and_wait_for_text(child, captured, "\r", "Setup complete", timeout=25)
        wait_for_rendered_text(child, captured, "F1 Help · /help", timeout=25)

        # Both default models are usable in the running TUI.
        _open_provider_settings(child, captured)
        wait_for_rendered_text(child, captured, "Key Set", timeout=15)
        wait_for_rendered_text(child, captured, "2 runnable models", timeout=15)
        _close_provider_settings(child, captured)
        _quit_and_wait(child, captured)
    finally:
        _stop_child(child)

    # Completing onboarding on the shipped defaults materializes the template.
    catalog_path = chartreux_home / "models.toml"
    assert catalog_path.is_file()
    assert catalog_path.read_text(encoding="utf-8") == render_default_template()
    env_path = chartreux_home / ".env"
    assert env_path.is_file()
    assert f"MISTRAL_API_KEY='{api_key_value}'" in env_path.read_text(encoding="utf-8")


@pytest.mark.timeout(60)
def test_no_key_welcome_decline_exits_without_materializing(
    e2e_workdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining the no-key welcome exits cleanly and writes no catalog."""
    chartreux_home = tmp_path / "decline-home"
    assert not chartreux_home.exists()
    _prepare_first_run_home(monkeypatch, chartreux_home)

    child, captured = _spawn_first_run(e2e_workdir)
    try:
        wait_for_rendered_text(child, captured, "Let's get you started", timeout=15)
        child.send("\x1b")
        child.expect(ansi_tolerant_pattern("Setup cancelled"), timeout=10)
        child.expect(pexpect.EOF, timeout=10)
    finally:
        _stop_child(child)

    assert child.exitstatus == 0
    assert not (chartreux_home / "models.toml").exists()
    output = strip_ansi(captured.getvalue())
    assert "Provider Settings" not in output


@pytest.mark.timeout(90)
def test_sparse_overlay_first_run_merges_without_rewriting_the_user_catalog(
    streaming_mock_server: StreamingMockServer,
    e2e_workdir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sparse user models.toml merges onto the shipped catalog at first run.

    The overlay overrides only the Mistral base URL; the credential wiring is
    inherited from the shipped provider, so first run skips onboarding, serves
    inference through the merged catalog, and never rewrites the user file.
    """
    chartreux_home = tmp_path / "sparse-overlay-home"
    catalog_path, sparse_overlay = _write_sparse_mistral_overlay(
        chartreux_home, streaming_mock_server.api_base
    )
    _prepare_first_run_home(monkeypatch, chartreux_home)
    monkeypatch.setenv("MISTRAL_API_KEY", "e2e-sparse-key")

    child, captured = _spawn_first_run(e2e_workdir)
    try:
        wait_for_rendered_text(child, captured, "F1 Help · /help", timeout=25)
        output = strip_ansi(captured.getvalue())
        assert "Let's get you started" not in output

        child.send("Greet from the sparse overlay")
        child.send("\r")
        wait_for_request_count_while_draining_child_output(
            child,
            captured,
            lambda: len(streaming_mock_server.requests),
            expected_count=1,
            timeout=10,
        )
        child.expect(ansi_tolerant_pattern("Hello from mock server"), timeout=10)
        _quit_and_wait(child, captured)
    finally:
        _stop_child(child)

    # The user's sparse overlay is never rewritten by first run.
    assert catalog_path.read_text(encoding="utf-8") == sparse_overlay
    request_payload = streaming_mock_server.requests[-1]
    assert request_payload.get("model") == "zai-glm-5-3"
    messages = request_payload.get("messages")
    assert messages is not None
    assert any(
        message.get("role") == "user"
        and message.get("content") == "Greet from the sparse overlay"
        for message in messages
    )


@pytest.mark.timeout(90)
def test_multi_key_env_first_run_sends_the_selected_provider_credential(
    streaming_mock_server: StreamingMockServer,
    e2e_workdir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several keys in the environment send the selected provider's credential.

    First run with multiple API keys in the environment skips onboarding, and
    the request carries the Mistral credential — never an unrelated key.
    """
    chartreux_home = tmp_path / "multi-key-home"
    catalog_path, sparse_overlay = _write_sparse_mistral_overlay(
        chartreux_home, streaming_mock_server.api_base
    )
    _prepare_first_run_home(monkeypatch, chartreux_home)
    monkeypatch.setenv("MISTRAL_API_KEY", "e2e-mistral-key")
    monkeypatch.setenv("EXA_API_KEY", "e2e-exa-key")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "e2e-brave-key")

    child, captured = _spawn_first_run(e2e_workdir)
    try:
        wait_for_rendered_text(child, captured, "F1 Help · /help", timeout=25)
        output = strip_ansi(captured.getvalue())
        assert "Let's get you started" not in output

        child.send("Greet with several keys set")
        child.send("\r")
        wait_for_request_count_while_draining_child_output(
            child,
            captured,
            lambda: len(streaming_mock_server.requests),
            expected_count=1,
            timeout=10,
        )
        child.expect(ansi_tolerant_pattern("Hello from mock server"), timeout=10)
        _quit_and_wait(child, captured)
    finally:
        _stop_child(child)

    assert catalog_path.read_text(encoding="utf-8") == sparse_overlay
    headers = streaming_mock_server.request_headers[-1]
    assert headers.get("authorization") == "Bearer e2e-mistral-key"
    header_values = " ".join(headers.values())
    assert "e2e-exa-key" not in header_values
    assert "e2e-brave-key" not in header_values


@pytest.mark.timeout(90)
def test_invalid_credential_first_run_reports_invalid_api_key(
    e2e_workdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected credential is reported honestly on the first turn.

    First run with a key the provider rejects skips onboarding (the credential
    is present), the first turn surfaces the invalid-key error, and the app
    stays up and exits cleanly.
    """
    server = StreamingMockServer(error_status=401)
    server.start()
    try:
        chartreux_home = tmp_path / "invalid-credential-home"
        catalog_path, sparse_overlay = _write_sparse_mistral_overlay(
            chartreux_home, server.api_base
        )
        _prepare_first_run_home(monkeypatch, chartreux_home)
        monkeypatch.setenv("MISTRAL_API_KEY", "e2e-rejected-key")

        child, captured = _spawn_first_run(e2e_workdir)
        try:
            wait_for_rendered_text(child, captured, "F1 Help · /help", timeout=25)
            output = strip_ansi(captured.getvalue())
            assert "Let's get you started" not in output

            child.send("Greet with a rejected key")
            child.send("\r")
            wait_for_request_count_while_draining_child_output(
                child,
                captured,
                lambda: len(server.requests),
                expected_count=1,
                timeout=10,
            )
            # The rejected credential is reported; the app stays alive.
            wait_for_rendered_text(child, captured, "Invalid API key", timeout=15)
            _quit_and_wait(child, captured)
        finally:
            _stop_child(child)

        assert catalog_path.read_text(encoding="utf-8") == sparse_overlay
        assert server.requests
    finally:
        server.stop()
