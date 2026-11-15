"""Gate-scoped fixtures for the spawned stdio / PTY legs.

These tests drive real spawned processes, so they only ever run inside the
named isolation mechanism (``tests/pty/run_gate.sh`` -> bubblewrap network
namespace). Collected without that gate they skip with the gate command in the
reason: a green unisolated PTY run is exactly the false green the gate exists to
prevent, and a skipped item is never evidence for a capability receipt.

Inside the gate the ordering matters and is pinned here:

* ``stdio_after_autouse`` sets ``CHARTREUX_TRANSPORT=stdio`` AFTER the suite-wide
  autouse (``tests/conftest.py``) pinned ``in_process``, re-resolves the
  selection from the resulting environment, and stamps the executed-receipt
  recorder with the transport the node actually runs on;
* ``local_mock_provider`` starts the deterministic provider on ``127.0.0.1``
  inside the gate's process tree -- the sandbox loopback is not the host's, so a
  provider started outside it would be unreachable by construction;
* ``pty_driver`` owns the spawned TUI: its cleanup kills the child's process
  group, waits, closes the PTY, and the provider fixture then asserts the port
  was released.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import os
from pathlib import Path
import socket
from urllib.parse import urlsplit

import pytest

from chartreux.app_server.transport import (
    STDIO_TRANSPORT,
    TRANSPORT_ENV_VAR,
    resolve_transport_selection,
)
from tests.app_server import executed_receipts
from tests.e2e.mock_server import StreamingMockServer
from tests.pty.harness.drive_tui import TuiDriver

#: Set by ``tests/pty/run_gate.sh`` inside the sandbox. Deliberately not a
#: ``CHARTREUX_*`` name: that prefix is the runtime's configuration namespace,
#: and an unknown name there fails every config read.
GATE_ENV_VAR = "PYTEST_CHARTREUX_PTY_GATE"

GATE_COMMAND = 'bash tests/pty/run_gate.sh "uv run --no-sync pytest tests/pty"'

#: The receipt nodes a whole-directory gate run must have executed and passed.
_RECEIPT_NODES = (
    "tests/pty/test_isolation_receipts.py::test_native_nonloopback_denied",
    "tests/pty/test_isolation_receipts.py::test_python_child_nonloopback_denied_by_kernel",
    "tests/pty/test_isolation_receipts.py::test_loopback_permitted",
    "tests/pty/test_isolation_receipts.py::test_no_nonloopback_interface_exists",
)


def isolation_gate_active() -> bool:
    """Whether this run was launched by the named isolation mechanism."""
    return os.environ.get(GATE_ENV_VAR) == "1"


@pytest.fixture(scope="session", autouse=True)
def _require_isolation_gate() -> None:
    """Refuse to run the spawned legs outside the gate (skip, never pass)."""
    if isolation_gate_active():
        return
    pytest.skip(
        "spawned stdio/PTY legs run only under the isolation gate: " + GATE_COMMAND
    )


@pytest.fixture
def stdio_after_autouse(monkeypatch: pytest.MonkeyPatch) -> str:
    """Select stdio after the suite autouse and stamp the executed receipt."""
    monkeypatch.setenv(TRANSPORT_ENV_VAR, STDIO_TRANSPORT)
    assert resolve_transport_selection(environ=os.environ).kind == STDIO_TRANSPORT
    executed_receipts.stamp_transport(STDIO_TRANSPORT)
    return STDIO_TRANSPORT


@pytest.fixture
def local_mock_provider(
    request: pytest.FixtureRequest,
) -> Iterator[StreamingMockServer]:
    """The deterministic model provider, on the gate's own loopback.

    A module may pin a scripted reply sequence by defining the module-level
    ``PROVIDER_CHUNK_SCRIPT`` (the mock server's chunk-factory contract); the
    script module owns routing and answers every provider request the spawned
    tree can make, so no gate run ever falls back to a live endpoint.
    """
    factory = getattr(request.module, "PROVIDER_CHUNK_SCRIPT", None)
    server = StreamingMockServer(chunk_factory=factory)
    server.start()
    port = urlsplit(server.api_base).port
    assert port is not None
    try:
        yield server
    finally:
        server.stop()
        assert_port_released(port)


@pytest.fixture
def pty_driver(
    tmp_path: Path, stdio_after_autouse: str, local_mock_provider: StreamingMockServer
) -> Iterator[TuiDriver]:
    """A spawned TUI on the stdio transport, isolated under the test's tmp dir."""
    driver = TuiDriver(tmp_path / "tui", local_mock_provider).spawn()
    try:
        yield driver
    finally:
        driver.close()
        driver.snapshot()


@pytest.fixture
def pty_driver_factory(
    tmp_path: Path, stdio_after_autouse: str, local_mock_provider: StreamingMockServer
) -> Iterator[Callable[[Sequence[str] | None], TuiDriver]]:
    """Build spawned TUIs under one shared home, closing every one at teardown.

    Flows that need more than one TUI process against the same isolated home
    (a resume flow replaces the process rather than reusing it) spawn through
    this factory; the extra arguments land verbatim on the CLI argv, so a
    ``["--continue"]`` spawn resumes the home's most recent saved session.
    """
    drivers: list[TuiDriver] = []

    def _spawn(extra_args: Sequence[str] | None = None) -> TuiDriver:
        driver = TuiDriver(
            tmp_path / "tui", local_mock_provider, extra_args=extra_args or ()
        ).spawn()
        drivers.append(driver)
        return driver

    yield _spawn
    for driver in reversed(drivers):
        driver.close()
        driver.snapshot()


def assert_port_released(port: int) -> None:
    """The provider's listening port must be free once the run is over."""
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as exc:
        raise AssertionError(
            f"the in-gate provider port {port} was not released: {exc}"
        ) from exc
    finally:
        probe.close()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail a gate run whose isolation receipts did not execute and pass.

    The verdict is read from the executed-receipt journal (C5b machinery), which
    is the only place a controller can see what its workers actually ran. A
    receipt that was skipped, deselected, failed, or never collected leaves no
    passed entry, so a gate run cannot go green without proving its isolation.

    A narrowed ``-k`` flow run is the caller's explicit selection (the C2e
    restoration command selects one flow); the constraint enforced here is that
    a whole-directory gate run cannot quietly drop the receipts. The gate's own
    pre-launch probes still prove isolation for a narrowed run.
    """
    if hasattr(session.config, "workerinput"):
        return
    if not isolation_gate_active():
        return
    if session.config.option.keyword:
        return
    failures = receipt_failures()
    if not failures:
        return
    session.exitstatus = 1
    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_sep("-", "isolation receipt gate failed", red=True)
        for failure in failures:
            terminal.write_line(failure, red=True)


def receipt_failures() -> tuple[str, ...]:
    """The receipt nodes this gate run did not execute-and-pass."""
    path = executed_receipts.journal_path()
    if path is None:
        return (
            "the gate run has no executed-receipt journal "
            f"({executed_receipts.JOURNAL_ENV_VAR} is unset): the isolation "
            "receipts cannot be reconciled",
        )
    journal = executed_receipts.read_journal(path)
    passed = {
        _base(receipt.node_id)
        for receipt in journal
        if receipt.outcome == executed_receipts.PASSED and not receipt.shared
    }
    recorded = {_base(receipt.node_id): receipt.outcome for receipt in journal}
    return tuple(
        f"isolation receipt did not execute and pass: {node_id} "
        f"(recorded: {recorded.get(node_id, 'nothing')})"
        for node_id in _RECEIPT_NODES
        if node_id not in passed
    )


def _base(node_id: str) -> str:
    """A node id without xdist's ``@group`` suffix or a parametrization."""
    return node_id.split("@", 1)[0].partition("[")[0]
