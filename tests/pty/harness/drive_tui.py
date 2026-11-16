"""Durable PTY driver for a real chartreux TUI on the stdio transport.

Port of the archived B1 probe driver (``tasks/v0.7.9/pty-probe-archive/harness/
drive_tui.py``) into the repository's own e2e idiom (``tests/e2e/common.py``):
an isolated ``CHARTREUX_HOME``/``HOME`` tree under the caller's per-test
directory, the deterministic :class:`StreamingMockServer` on ``127.0.0.1`` as
the only model provider, a fake API key, and a PTY driven through pexpect.

Two properties matter for the stdio gate:

* the provider is started INSIDE the gate's process tree. The sandbox loopback
  is not the host's, so an endpoint outside the namespace is unreachable by
  construction -- the mock provider must live on the same loopback the spawned
  TUI can see;
* the spawned child's transport is asserted, not assumed. The selection is
  published into the spawn environment, re-resolved from that environment, and
  the stdio topology is proven executed by finding the separate root-agent
  process the TUI spawned.

Everything the driver writes stays under ``root`` (a pytest tmp dir, i.e. the
gate's in-namespace tmpfs). The repository is only read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import cast

import pexpect

from chartreux.app_server.transport import (
    STDIO_TRANSPORT,
    TRANSPORT_ENV_VAR,
    resolve_transport_selection,
)
from tests.e2e.common import (
    ansi_tolerant_pattern,
    drain_child_output,
    strip_ansi,
    write_e2e_config,
)
from tests.e2e.mock_server import StreamingMockServer

#: The module the coordinator spawns as the root agent on the stdio route. Its
#: presence in the child's process tree is what proves the stdio topology ran.
ROOT_AGENT_MODULE = "chartreux.app_server.root_agent"

#: Rendered lines that mean a flow hit a wire refusal or a crash. Kept narrow on
#: purpose: a broad match turns ordinary output into a false alarm.
ERROR_MARKERS = (
    "NOT_IMPLEMENTED",
    "not carried",
    "Traceback (most recent call last)",
    "Internal error",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def chartreux_command() -> list[str]:
    """The argv that starts the CLI without a `uv` sync in the sandbox."""
    console_script = Path(sys.executable).with_name("chartreux")
    if console_script.exists():
        return [str(console_script)]
    return ["uv", "run", "--project", str(_REPO_ROOT), "--no-sync", "chartreux"]


def process_tree(pid: int) -> list[tuple[int, list[str]]]:
    """Every live descendant of ``pid`` with its argv, depth-first.

    Reads ``/proc/<pid>/task/<pid>/children``: the sandbox keeps the host PID
    namespace, so the spawned TUI's tree stays observable from the test process.
    """
    found: list[tuple[int, list[str]]] = []
    pending = [pid]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            raw = Path(f"/proc/{current}/cmdline").read_bytes()
        except OSError:
            continue
        argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        found.append((current, argv))
        try:
            children = (
                Path(f"/proc/{current}/task/{current}/children")
                .read_text(encoding="utf-8")
                .split()
            )
        except OSError:
            continue
        pending.extend(int(child) for child in children if child.isdigit())
    return found


def root_agent_pids(pid: int) -> list[int]:
    """The pids in ``pid``'s tree running the root agent module."""
    return [
        descendant
        for descendant, argv in process_tree(pid)
        if any(ROOT_AGENT_MODULE in argument for argument in argv)
    ]


class TuiDriver:
    """One spawned TUI on the stdio transport, plus its isolated home tree."""

    def __init__(
        self,
        root: Path,
        provider: StreamingMockServer,
        *,
        transport: str = STDIO_TRANSPORT,
        dimensions: tuple[int, int] = (40, 120),
        timeout: float = 30.0,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.root = root
        self.provider = provider
        self.transport = transport
        self.dimensions = dimensions
        self.timeout = timeout
        self.extra_args = list(extra_args)
        self.home = root / "home"
        self.workdir = root / "workdir"
        self.cwd = root / "cwd"
        self.captured = io.StringIO()
        self.events: list[dict[str, object]] = []
        self.child: pexpect.spawn | None = None
        for directory in (self.home, self.workdir, self.cwd):
            directory.mkdir(parents=True, exist_ok=True)

    # -- lifecycle --------------------------------------------------------

    def environment(self) -> dict[str, str]:
        """The spawn environment: isolated home, fake key, stdio selection."""
        env = os.environ.copy()
        env.update({
            "CHARTREUX_HOME": str(self.home),
            "HOME": str(self.home),
            "XDG_CONFIG_HOME": str(self.home / "xdg-config"),
            "XDG_DATA_HOME": str(self.home / "xdg-data"),
            "XDG_STATE_HOME": str(self.home / "xdg-state"),
            "XDG_CACHE_HOME": str(self.home / "xdg-cache"),
            "MISTRAL_API_KEY": "fake-key",
            "CHARTREUX_TEST_DISABLE_KEYRING": "1",
            "CHARTREUX_TEST_DISABLE_AUTO_TITLE": "1",
            "TERM": "xterm-256color",
            "PYTHONDONTWRITEBYTECODE": "1",
            TRANSPORT_ENV_VAR: self.transport,
        })
        return env

    def spawn(self) -> TuiDriver:
        """Write the isolated config and start the TUI on a PTY."""
        # v0.8.0 plan-mode removal retired the ``[plan]`` config block (it is a
        # config error now), so no plan-role pin is written into the home config.
        write_e2e_config(self.home, self.provider.api_base)
        env = self.environment()
        assert env[TRANSPORT_ENV_VAR] == self.transport
        command = chartreux_command()
        self.events.append({
            "tag": "spawn",
            "command": command,
            "transport": self.transport,
            "provider": self.provider.api_base,
        })
        child = pexpect.spawn(
            command[0],
            [*command[1:], "--workdir", str(self.workdir), *self.extra_args],
            cwd=str(self.cwd),
            env=cast("os._Environ[str]", env),
            encoding="utf-8",
            timeout=self.timeout,
            dimensions=self.dimensions,
        )
        child.logfile_read = self.captured
        self.child = child
        return self

    def close(self) -> None:
        """Kill the child's whole process group, wait, and close the PTY.

        The TUI is a session leader (pexpect forks with ``setsid``), so the
        process group is the spawned tree: the coordinator, the root agent and
        any shell an MCP server started. Killing the group leaves no orphaned
        spawned root behind, and the bounded wait gives the kernel time to
        release the PTY and its ports before the caller asserts on them.
        """
        child = self.child
        if child is None:
            return
        try:
            pid = child.pid
            if child.isalive() and pid is not None:
                with suppress(ProcessLookupError, PermissionError):
                    os.killpg(pid, signal.SIGKILL)
        finally:
            with suppress(Exception):
                child.terminate(force=True)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if not child.isalive():
                    break
                time.sleep(0.05)
            with suppress(Exception):
                child.close(force=True)
            self.child = None

    def __enter__(self) -> TuiDriver:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- reading ----------------------------------------------------------

    def text(self) -> str:
        """Everything the child rendered so far, ANSI-stripped."""
        return strip_ansi(self.captured.getvalue())

    def mark(self) -> int:
        """A position in the rendered transcript, for "since here" checks."""
        return len(self.text())

    def drain(self, idle: float = 0.4) -> bool:
        """Read available output; returns ``True`` at child EOF."""
        if self.child is None:
            return True
        if drain_child_output(self.child, self.captured):
            return True
        if idle > 0:
            time.sleep(idle)
        return False

    def wait_for(self, needle: str, timeout: float) -> float:
        """Wait for rendered text, returning how long it took.

        Drains the PTY on every poll: a Textual app re-renders whole screens,
        and an undrained PTY blocks the app's writer thread.
        """
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if needle in self.text():
                elapsed = time.monotonic() - start
                self.events.append({
                    "tag": "wait",
                    "needle": needle,
                    "t": round(elapsed, 2),
                })
                return elapsed
            if self.drain(idle=0):
                if needle in self.text():
                    return time.monotonic() - start
                raise AssertionError(
                    f"child exited while waiting for {needle!r}\n\n{self._failure_details()}"
                )
        raise AssertionError(
            f"timed out after {timeout}s waiting for {needle!r}\n\n{self._failure_details()}"
        )

    def wait_until(
        self, predicate: Callable[[], bool], timeout: float, description: str
    ) -> None:
        """Poll a transcript predicate, draining the PTY, until it holds."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if predicate():
                return
            if self.drain(idle=0):
                break
        if predicate():
            return
        raise AssertionError(
            f"timed out after {timeout}s waiting until {description}\n\n"
            f"{self._failure_details()}"
        )

    def wait_for_root_agent(self, timeout: float) -> int:
        """Wait for the spawned root-agent process and return its pid."""
        child = self.child
        if child is None:
            raise AssertionError("the TUI was not spawned")
        pid = child.pid
        if pid is None:
            raise AssertionError("the TUI was not spawned")
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            pids = root_agent_pids(pid)
            if pids:
                self.events.append({"tag": "root-agent", "pids": pids})
                return pids[0]
            time.sleep(0.1)
        raise AssertionError(
            "no separate root agent process appeared in the spawned tree; "
            f"tree: {process_tree(pid)}\n\n{self._failure_details()}"
        )

    def tail(self, mark: int, limit: int = 600) -> str:
        """Rendered text since ``mark``, tail-limited for assertions."""
        return self.text()[mark:][-limit:]

    def errors_since(self, mark: int) -> list[str]:
        """Rendered lines since ``mark`` carrying a wire refusal or a crash."""
        found: list[str] = []
        for line in self.text()[mark:].splitlines():
            if any(marker in line for marker in ERROR_MARKERS):
                found.append(line.strip()[:300])
        return found

    def _failure_details(self) -> str:
        rendered = self.text()[-2000:]
        return f"rendered tail:\n{rendered}"

    # -- writing ----------------------------------------------------------

    def send(self, keys: str) -> None:
        """Send raw keys to the child."""
        if self.child is None:
            raise AssertionError("the TUI was not spawned")
        self.child.send(keys)

    def send_line(self, text: str) -> None:
        """Type a line and submit it."""
        self.send(text)
        self.send("\r")

    def expect(self, needle: str, timeout: float) -> None:
        """Wait on the ANSI-tolerant pattern for ``needle``, draining as we go."""
        if self.child is None:
            raise AssertionError("the TUI was not spawned")
        pattern = ansi_tolerant_pattern(needle)
        try:
            self.child.expect(pattern, timeout=timeout)
        except pexpect.TIMEOUT as exc:
            raise AssertionError(
                f"timed out waiting for pattern {needle!r}\n\n{self._failure_details()}"
            ) from exc
        except pexpect.EOF as exc:
            raise AssertionError(
                f"child exited while waiting for {needle!r}\n\n{self._failure_details()}"
            ) from exc
        self.drain(idle=0)

    def wait_for_exit(self, timeout: float) -> None:
        """Wait for the child to reach EOF: a clean shutdown, not a kill."""
        child = self.child
        if child is None:
            raise AssertionError("the TUI was not spawned")
        try:
            child.expect(pexpect.EOF, timeout=timeout)
        except pexpect.TIMEOUT as exc:
            raise AssertionError(
                f"the TUI did not exit within {timeout}s\n\n{self._failure_details()}"
            ) from exc

    def quit(self, timeout: float = 20.0) -> None:
        """Leave the TUI the way a user does: Ctrl+C twice."""
        child = self.child
        if child is None:
            return
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            self.send("\x03")
            try:
                self.expect("Press Ctrl+C again to quit", timeout=2)
            except AssertionError:
                if self.drain(idle=0.2):
                    return
                continue
            self.send("\x03")
            with suppress(pexpect.EOF, pexpect.TIMEOUT):
                child.expect(pexpect.EOF, timeout=10)
            return

    # -- artifacts --------------------------------------------------------

    def snapshot(self) -> Path:
        """Persist the transcript and the event stream under the run root."""
        (self.root / "transcript.raw.txt").write_text(
            self.captured.getvalue(), encoding="utf-8"
        )
        (self.root / "transcript.txt").write_text(self.text(), encoding="utf-8")
        (self.root / "events.jsonl").write_text(
            "\n".join(json.dumps(event) for event in self.events), encoding="utf-8"
        )
        return self.root

    def event(self, **payload: object) -> None:
        """Record one event in the stream this driver emits."""
        self.events.append(payload)

    def effective_transport(self) -> str:
        """The transport the spawn environment resolves to."""
        return resolve_transport_selection(environ=self.environment()).kind


def assert_stdio_environment(environ: Mapping[str, str]) -> None:
    """Assert the spawn environment selects stdio, as the harness contract says."""
    assert environ.get(TRANSPORT_ENV_VAR) == STDIO_TRANSPORT, environ.get(
        TRANSPORT_ENV_VAR
    )
    assert resolve_transport_selection(environ=environ).kind == STDIO_TRANSPORT


def assistant_text_chunks(text: str, *, created: int = 200) -> list[dict[str, object]]:
    """A one-shot reply chunk set: the text, then a stop with usage."""
    return [
        StreamingMockServer.build_chunk(
            created=created,
            delta={"role": "assistant", "content": text},
            finish_reason=None,
        ),
        StreamingMockServer.build_chunk(
            created=created + 1,
            delta={},
            finish_reason="stop",
            usage={"prompt_tokens": 3, "completion_tokens": 4},
        ),
    ]
