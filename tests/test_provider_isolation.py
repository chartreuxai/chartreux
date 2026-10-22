from __future__ import annotations

import errno
import os
import socket
import subprocess
import sys
import threading

import pytest

CREDENTIALS = (
    "MISTRAL_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


def test_provider_network_and_credentials_are_isolated():
    with pytest.raises(OSError):
        socket.create_connection(("example.com", 443), timeout=0.1)
    sock = socket.socket()
    try:
        assert sock.connect_ex(("93.184.216.34", 443)) != 0
    finally:
        sock.close()
    assert all(os.environ.get(name) in (None, "mock") for name in CREDENTIALS)

    code = (
        "import os,socket; "
        "s=socket.socket(); "
        "r=s.connect_ex(('93.184.216.34',443)); s.close(); "
        "assert r != 0; "
        f"assert all(os.environ.get(n) in (None, 'mock') for n in {CREDENTIALS!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_loopback_and_unix_sockets_are_allowed(tmp_path):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    accepted = []

    def accept_one():
        conn, _ = server.accept()
        accepted.append(True)
        conn.close()

    thread = threading.Thread(target=accept_one)
    thread.start()
    client = socket.create_connection(("localhost", server.getsockname()[1]), timeout=2)
    client.close()
    thread.join(timeout=2)
    server.close()
    assert accepted

    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("AF_UNIX unavailable")
    path = str(tmp_path / "local.sock")
    unix_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    unix_server.bind(path)
    unix_server.listen()
    thread = threading.Thread(target=lambda: unix_server.accept()[0].close())
    thread.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(path)
    client.close()
    thread.join(timeout=2)
    unix_server.close()


def test_ipv6_four_tuple_addresses_are_rejected_before_the_transport_call(tmp_path):
    """Both provider-transport guards reject an IPv6 4-tuple address before
    the original transport call: the guard's own malformed-address OSError
    on both connect and connect_ex (an AF_INET transport would raise
    TypeError on a 4-tuple, so the guard's signature proves the rejection
    happened first), while 2-tuples and non-tuple Unix addresses pass the
    guard.
    """
    sock = socket.socket()
    try:
        with pytest.raises(OSError, match="malformed socket address"):
            sock.connect(("::1", 443, 0, 0))
        with pytest.raises(OSError, match="malformed socket address"):
            sock.connect_ex(("::1", 443, 0, 0))
    finally:
        sock.close()

    # A 2-tuple passes the guard: a real loopback connection succeeds.
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    try:
        client = socket.create_connection(
            ("127.0.0.1", server.getsockname()[1]), timeout=2
        )
        client.close()
    finally:
        server.close()

    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("AF_UNIX unavailable")
    # A non-tuple Unix address passes the guard: the real transport answers
    # (ENOENT / FileNotFoundError), never the guard's malformed-address error.
    missing = str(tmp_path / "missing.sock")
    unix_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        assert unix_sock.connect_ex(missing) == errno.ENOENT
        with pytest.raises(FileNotFoundError):
            unix_sock.connect(missing)
    finally:
        unix_sock.close()

    # The same pins hold in a fresh interpreter under the sitecustomize guard.
    code = (
        "import errno, os, socket, tempfile\n"
        "s = socket.socket()\n"
        "try:\n"
        "    s.connect(('::1', 443, 0, 0))\n"
        "    raise SystemExit('4-tuple connect was not rejected')\n"
        "except OSError as exc:\n"
        "    assert 'malformed socket address' in str(exc), exc\n"
        "try:\n"
        "    s.connect_ex(('::1', 443, 0, 0))\n"
        "    raise SystemExit('4-tuple connect_ex was not rejected')\n"
        "except OSError as exc:\n"
        "    assert 'malformed socket address' in str(exc), exc\n"
        "s.close()\n"
        "srv = socket.socket()\n"
        "srv.bind(('127.0.0.1', 0))\n"
        "srv.listen()\n"
        "client = socket.socket()\n"
        "client.connect(('127.0.0.1', srv.getsockname()[1]))\n"
        "client.close()\n"
        "srv.close()\n"
        "missing = os.path.join(tempfile.mkdtemp(), 'missing.sock')\n"
        "u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "assert u.connect_ex(missing) == errno.ENOENT\n"
        "u.close()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
