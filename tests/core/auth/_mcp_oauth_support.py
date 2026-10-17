"""Shared test-support surface for the MCP OAuth suite (W10, stage 19j).

The definitions here are the shared module-level surface of
``tests/core/auth/test_mcp_oauth.py``, relocated AST-verbatim in the 19j
phase-(i) support-seam commit: the ``memory_keyring`` fixture (61 tests
request it, spanning the storage, fingerprint and flow clusters), the
``_MemoryKeyring`` in-memory keyring backend class (used by the storage,
fingerprint and flow clusters and instantiated by the ``memory_keyring``
fixture), the ``_KEYRING_SERVICE`` keyring service constant (3 storage
tests, 1 flow test and the ``_MemoryKeyring`` methods), and the server /
loopback helpers ``_oauth_server`` (33 users across the storage,
fingerprint and flow clusters), ``_free_port`` (12 users, callback and
flow), ``_send_callback`` (7 users, callback and flow) and
``_connect_when_bound`` (the dependency closure of ``_send_callback``;
5 direct callback users). The remaining module-level surface is
single-concern and stays in ``test_mcp_oauth.py`` until the phase-(ii)
concern moves: the ``headless_keyring`` fixture (2 users, storage), the
``_mock_login_transport`` helper (5 users, flow), the ``_mcp_responses``
helper (2 users, flow) and the ``_RaisingStream`` class (2 users, flow).
No test bodies or marks live here; the test population stays in
``test_mcp_oauth.py`` until the phase-(ii) concern moves. Importer:
``test_mcp_oauth.py`` itself (the seam enumeration found zero external
importers of that file).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import suppress
import socket

import keyring
from keyring.backend import KeyringBackend
import keyring.errors
import pytest

from chartreux.core.config import MCPHttp, MCPOAuth

_KEYRING_SERVICE = "chartreux"


class _MemoryKeyring(KeyringBackend):
    priority = 100  # type: ignore[assignment]

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        assert service == _KEYRING_SERVICE
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        assert service == _KEYRING_SERVICE
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        assert service == _KEYRING_SERVICE
        if (service, username) not in self.store:
            raise keyring.errors.PasswordDeleteError()
        del self.store[(service, username)]


@pytest.fixture
def memory_keyring() -> Iterator[_MemoryKeyring]:
    original = keyring.get_keyring()
    fake = _MemoryKeyring()
    keyring.set_keyring(fake)
    try:
        yield fake
    finally:
        keyring.set_keyring(original)


def _oauth_server(
    *,
    name: str = "demo",
    url: str = "https://mcp.example.com/mcp",
    scopes: list[str] | None = None,
    client_id: str | None = None,
    client_metadata_url: str | None = None,
    redirect_port: int = 47823,
) -> MCPHttp:
    auth = MCPOAuth(
        type="oauth",
        scopes=scopes if scopes is not None else ["read", "write"],
        client_id=client_id,
        client_metadata_url=client_metadata_url,  # type: ignore[arg-type]
        redirect_port=redirect_port,
    )
    return MCPHttp(transport="streamable-http", name=name, url=url, auth=auth)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _connect_when_bound(
    port: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    async def connect() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        while True:
            try:
                return await asyncio.open_connection("127.0.0.1", port)
            except ConnectionRefusedError:
                await asyncio.sleep(0)

    return await asyncio.wait_for(connect(), 1)


async def _send_callback(port: int, query: str, *, timeout: float = 5.0) -> bytes:
    deadline = asyncio.get_event_loop().time() + timeout
    last_err: BaseException | None = None
    while asyncio.get_event_loop().time() < deadline:
        try:
            reader, writer = await _connect_when_bound(port)
        except (ConnectionRefusedError, OSError) as exc:
            last_err = exc
            await asyncio.sleep(0.02)
            continue
        try:
            request = (
                f"GET /callback?{query} HTTP/1.0\r\n"
                "Host: 127.0.0.1\r\n"
                "Connection: close\r\n"
                "\r\n"
            )
            writer.write(request.encode("ascii"))
            await writer.drain()
            return await reader.read()
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
    raise RuntimeError(f"loopback never bound on port {port}: {last_err}")
