"""MCP OAuth login fingerprint tests (scope-order stability, whitespace stripping and dedupe, client-id/metadata-url/DCR markers, static-auth rejection, url/scope/marker change detection, missing load, save/load round trip)."""

from __future__ import annotations

import pytest

from chartreux.core.auth.mcp_oauth import Fingerprint
from chartreux.core.config import MCPHttp
from tests.core.auth._mcp_oauth_support import (
    _MemoryKeyring,
    _oauth_server,
    memory_keyring as memory_keyring,
)


class TestFingerprint:
    def test_compute_stable_across_scope_order(
        self, memory_keyring: _MemoryKeyring
    ) -> None:
        a = _oauth_server(scopes=["read", "write", "admin"])
        b = _oauth_server(scopes=["admin", "write", "read"])
        assert Fingerprint.compute(a) == Fingerprint.compute(b)

    def test_compute_strips_whitespace_and_dedupes(
        self, memory_keyring: _MemoryKeyring
    ) -> None:
        a = _oauth_server(scopes=["read", "write"])
        b = _oauth_server(scopes=[" read ", "write", "write", ""])
        assert Fingerprint.compute(a) == Fingerprint.compute(b)

    def test_compute_marker_for_client_id(self, memory_keyring: _MemoryKeyring) -> None:
        srv = _oauth_server(client_id="pre-registered-id")
        assert Fingerprint.compute(srv).client_marker == "pre-registered-id"

    def test_compute_marker_for_client_metadata_url(
        self, memory_keyring: _MemoryKeyring
    ) -> None:
        srv = _oauth_server(client_metadata_url="https://vibe.example/cm.json")
        fp = Fingerprint.compute(srv)
        assert fp.client_marker.startswith("https://vibe.example/cm.json")

    def test_compute_marker_for_dcr(self, memory_keyring: _MemoryKeyring) -> None:
        assert Fingerprint.compute(_oauth_server()).client_marker == "<dcr>"

    def test_compute_rejects_static_auth(self, memory_keyring: _MemoryKeyring) -> None:
        from chartreux.core.config import MCPStaticAuth

        srv = MCPHttp(
            transport="streamable-http",
            name="x",
            url="https://x/mcp",
            auth=MCPStaticAuth(),
        )
        with pytest.raises(TypeError, match="OAuth"):
            Fingerprint.compute(srv)

    def test_matches_detects_url_change(self, memory_keyring: _MemoryKeyring) -> None:
        a = Fingerprint.compute(_oauth_server(url="https://a/mcp"))
        b = Fingerprint.compute(_oauth_server(url="https://b/mcp"))
        assert a != b

    def test_matches_detects_scope_change(self, memory_keyring: _MemoryKeyring) -> None:
        a = Fingerprint.compute(_oauth_server(scopes=["read"]))
        b = Fingerprint.compute(_oauth_server(scopes=["read", "write"]))
        assert a != b

    def test_matches_detects_marker_change(
        self, memory_keyring: _MemoryKeyring
    ) -> None:
        a = Fingerprint.compute(_oauth_server(client_id="x"))
        b = Fingerprint.compute(_oauth_server(client_id="y"))
        assert a != b

    @pytest.mark.asyncio
    async def test_load_returns_none_when_missing(
        self, memory_keyring: _MemoryKeyring
    ) -> None:
        assert await Fingerprint.load("nope") is None

    @pytest.mark.asyncio
    async def test_save_and_load_round_trip(
        self, memory_keyring: _MemoryKeyring
    ) -> None:
        fp = Fingerprint.compute(_oauth_server(name="linear"))
        await fp.save("linear")
        loaded = await Fingerprint.load("linear")
        assert loaded == fp
