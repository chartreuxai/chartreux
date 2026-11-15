"""In-gate isolation receipts: the gate proves it is isolating, per run.

These execute INSIDE the network namespace that ``tests/pty/run_gate.sh``
created, so a runner whose isolation silently degraded cannot produce a green
gate receipt. Four properties are re-proved at test time, mirroring the gate's
pre-launch probes:

* (a) a native executable cannot reach a non-loopback destination;
* (b) a Python child cannot either;
* (c) loopback is usable -- the in-gate deterministic provider depends on it;
* (d) no non-loopback interface exists at all.

Outside the gate the suite-level fixture in ``tests/pty/conftest.py`` skips
these: unisolated, (a) and (b) would fail for the wrong reason (the host has
network access) and a skip is never evidence. Inside the gate they must pass.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading

import pytest

from tests.differential.egress import EGRESS_BLOCK_ENV, EGRESS_GUARD_ENV

pytestmark = [pytest.mark.timeout(60), pytest.mark.xdist_group(name="pty_gate")]

#: RFC 5737 documentation address: unroutable by definition, so a connection
#: attempt measures route availability inside the namespace, not a real peer.
DOCUMENTATION_ADDRESS = "192.0.2.1"

#: Exit codes of the kernel-layer probe below.
_PROBE_DENIED = 0
_PROBE_ESCAPED = 17

_KERNEL_PROBE = """
import socket
import sys

probe = socket.socket()
probe.settimeout(2)
try:
    probe.connect(("192.0.2.1", 80))
except OSError as exc:
    sys.stderr.write(f"connect denied: errno={exc.errno} {exc}\\n")
    sys.exit(0 if exc.errno in (101, 113) else 2)
sys.exit(17)
"""


def test_native_nonloopback_denied() -> None:
    """A native executable gets no route out of the namespace."""
    result = subprocess.run(
        [
            "curl",
            "--noproxy",
            "*",
            "-sS",
            "--connect-timeout",
            "3",
            "-o",
            "/dev/null",
            f"http://{DOCUMENTATION_ADDRESS}/",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, (
        f"native egress escaped: {result.stdout} {result.stderr}"
    )
    assert DOCUMENTATION_ADDRESS in result.stderr, result.stderr


def test_python_child_nonloopback_denied_by_kernel() -> None:
    """A python child gets no kernel route, independent of the in-process guard.

    The suite installs an in-process guard (``tests/differential/egress.py``)
    that denies non-loopback connects inside any python process carrying its
    marker, so an in-process probe would pass on a host with real network
    access too -- it proves the guard, not the namespace. This child drops the
    marker and its ``sitecustomize``, leaving the kernel's routing table as the
    only thing between it and the address.
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {EGRESS_BLOCK_ENV, EGRESS_GUARD_ENV, "PYTHONPATH"}
    }
    result = subprocess.run(
        [sys.executable, "-c", _KERNEL_PROBE],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != _PROBE_ESCAPED, (
        f"python egress escaped the namespace: {result.stdout} {result.stderr}"
    )
    assert result.returncode == _PROBE_DENIED, result.stderr


def test_loopback_permitted() -> None:
    """The gate's own loopback serves the deterministic provider."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve() -> None:
        connection, _ = server.accept()
        connection.send(b"ok")
        connection.close()

    threading.Thread(target=serve, daemon=True).start()
    client = socket.socket()
    client.settimeout(2)
    try:
        client.connect(("127.0.0.1", port))
        assert client.recv(2) == b"ok"
    finally:
        client.close()
        server.close()


def test_no_nonloopback_interface_exists() -> None:
    """The namespace has exactly one interface: loopback."""
    names: list[str] = []
    with open("/proc/net/dev", encoding="utf-8") as handle:
        for line in list(handle)[2:]:
            names.append(line.split(":", 1)[0].strip())
    assert names == ["lo"], names
