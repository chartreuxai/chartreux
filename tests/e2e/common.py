from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
import io
import os
from pathlib import Path
import re
import time
from typing import Protocol

import pexpect


class SpawnedChartreuxProcessFixture(Protocol):
    def __call__(
        self, workdir: Path, extra_args: Sequence[str] | None = None
    ) -> AbstractContextManager[tuple[pexpect.spawn, io.StringIO]]: ...


def ansi_tolerant_pattern(text: str) -> re.Pattern[str]:
    ansi = r"(?:\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\r|\n)*"
    return re.compile(ansi.join(re.escape(char) for char in text))


def write_e2e_config(
    vibe_home: Path, api_base: str, *, provider_name: str = "mock-provider"
) -> None:
    vibe_home.mkdir(parents=True, exist_ok=True)
    (vibe_home / "config.toml").write_text(
        "\n".join([
            'active_model = "mock-model"',
            "disable_welcome_banner_animation = true",
        ]),
        encoding="utf-8",
    )
    (vibe_home / "models.toml").write_text(
        "\n".join([
            f'[providers."{provider_name}/default"]',
            f'api_base = "{api_base}"',
            'api_key_env_var = "MISTRAL_API_KEY"',
            'backend = "generic"',
            "",
            '[models."mock-model"]',
            'thinking = "off"',
            "temperature = 0.2",
            "",
            '[[models."mock-model".deployments]]',
            f'provider = "{provider_name}/default"',
            'name = "mock-model"',
            "supports_images = false",
            "",
            '[roles."small-worker"]',
            'description = "E2E worker role"',
            'models = ["mock-model"]',
        ]),
        encoding="utf-8",
    )


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07", "", text)


def _rendered_text_failure_details(captured: str) -> str:
    raw_tail = captured[-2000:]
    stripped_tail = strip_ansi(captured)[-2000:]
    if any(char.isprintable() for char in stripped_tail):
        printable_tail = "".join(
            char if char.isprintable() else repr(char)[1:-1] for char in stripped_tail
        )
    else:
        printable_tail = "(no printable PTY output captured)"
    return (
        f"Raw captured tail ({len(raw_tail)} chars): {raw_tail!r}\n"
        f"ANSI-stripped tail ({len(stripped_tail)} chars): {stripped_tail!r}\n"
        f"Printable tail:\n{printable_tail}"
    )


def _app_log_tail() -> str | None:
    home = os.environ.get("CHARTREUX_HOME")
    if home is None:
        return None
    try:
        lines = (
            (Path(home) / "logs" / "chartreux.log")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    except (OSError, UnicodeError):
        return None
    return "\n".join(lines[-50:]) or None


def poll_until(predicate: Callable[[], bool], timeout: float, message: str) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(message)


def wait_for_request_count(
    request_count_getter: Callable[[], int], expected_count: int, timeout: float
) -> None:
    poll_until(
        lambda: request_count_getter() >= expected_count,
        timeout,
        f"Timed out waiting for {expected_count} backend request(s).",
    )


def wait_for_request_count_while_draining_child_output(
    child: pexpect.spawn,
    captured: io.StringIO,
    request_count_getter: Callable[[], int],
    *,
    expected_count: int,
    timeout: float,
) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if request_count_getter() >= expected_count:
            return
        if drain_child_output(child, captured):
            rendered_tail = strip_ansi(captured.getvalue())[-1200:]
            raise AssertionError(
                "Child exited while waiting for "
                f"{expected_count} backend request(s).\n\nRendered tail:\n{rendered_tail}"
            )
    rendered_tail = strip_ansi(captured.getvalue())[-1200:]
    raise AssertionError(
        f"Timed out waiting for {expected_count} backend request(s).\n\n"
        f"Rendered tail:\n{rendered_tail}"
    )


def wait_for_main_screen(child: pexpect.spawn, timeout: float = 20.0) -> None:
    child.expect(ansi_tolerant_pattern("Chartreux v"), timeout=timeout)


def drain_child_output(
    child: pexpect.spawn,
    captured: io.StringIO | None = None,
    *,
    idle_sleep: float = 0.05,
) -> bool:
    """Read all currently-available child output into the capture log.

    ``pexpect.spawn.expect`` only reads until its pattern matches, so it leaves
    large render bursts sitting in the kernel PTY buffer. A Textual app driven
    through a pexpect PTY can re-render whole screens on each cursor move; once
    that output exceeds the PTY buffer the app's writer thread blocks on
    ``write()``, and the app then deadlocks on shutdown
    (``driver.close`` -> ``WriterThread.stop`` -> ``join``). Draining the PTY
    keeps the writer thread unblocked.

    Returns ``True`` if the child reached EOF while draining. When ``captured`` is
    provided, output is retained there even if pexpect is not logging reads to it.
    """
    eof = False
    while True:
        try:
            output = child.read_nonblocking(size=65536, timeout=0)
            # pexpect logs direct reads to logfile_read. Avoid writing the same
            # output twice when that logger is already the requested capture.
            if captured is not None and child.logfile_read is not captured:
                captured.write(output)
        except pexpect.TIMEOUT:
            break
        except pexpect.EOF:
            eof = True
            break
    if not eof:
        time.sleep(idle_sleep)
    return eof


def wait_for_rendered_text(
    child: pexpect.spawn, captured: io.StringIO, needle: str, timeout: float
) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if needle in strip_ansi(captured.getvalue()):
            return
        if drain_child_output(child, captured):
            details = _rendered_text_failure_details(captured.getvalue())
            if app_log_tail := _app_log_tail():
                details = f"{details}\n\nApp log tail (last 50 lines):\n{app_log_tail}"
            raise AssertionError(
                f"Child exited while waiting for rendered text: {needle!r}\n\n{details}"
            )
    details = _rendered_text_failure_details(captured.getvalue())
    if app_log_tail := _app_log_tail():
        details = f"{details}\n\nApp log tail (last 50 lines):\n{app_log_tail}"
    raise AssertionError(
        f"Timed out waiting for rendered text: {needle!r}\n\n{details}"
    )


def send_ctrl_c_until_quit_confirmation(
    child: pexpect.spawn, captured: io.StringIO, timeout: float = 3
) -> None:
    """Send Ctrl+C and wait for quit confirmation prompt. Retries if first Ctrl+C interrupts."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        child.sendcontrol("c")
        try:
            child.expect(ansi_tolerant_pattern("Press Ctrl+C again to quit"), timeout=2)
            # Confirmation prompt appeared, send second Ctrl+C
            child.sendcontrol("c")
            return
        except pexpect.TIMEOUT:
            # First Ctrl+C may have interrupted something, try again
            continue
    rendered_tail = strip_ansi(captured.getvalue())[-1200:]
    raise AssertionError(
        f"Timed out waiting for quit confirmation prompt.\n\nRendered tail:\n{rendered_tail}"
    )
