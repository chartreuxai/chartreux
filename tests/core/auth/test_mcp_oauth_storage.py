"""MCP OAuth keyring token storage tests (token and client-info round trips, absolute and legacy expiry restore, per-alias isolation, headless init errors, credential deletion and its failure wrapping, snapshot/restore)."""

from __future__ import annotations

from collections.abc import Iterator
import json
import time
from unittest.mock import AsyncMock, patch

import keyring
import keyring.backends.fail
import keyring.errors
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
import pytest

from chartreux.core.auth.mcp_oauth import (
    Fingerprint,
    KeyringTokenStorage,
    MCPOAuthCredentialCleanupFailed,
    MCPOAuthHeadlessError,
    delete_oauth_credentials,
    restore_oauth_credentials,
    snapshot_oauth_credentials,
)
from tests.core.auth._mcp_oauth_support import (
    _KEYRING_SERVICE,
    _MemoryKeyring,
    _oauth_server,
    memory_keyring as memory_keyring,
)


@pytest.fixture
def headless_keyring() -> Iterator[None]:
    original = keyring.get_keyring()
    keyring.set_keyring(keyring.backends.fail.Keyring())
    try:
        yield
    finally:
        keyring.set_keyring(original)


class TestKeyringTokenStorage:
    @pytest.mark.asyncio
    async def test_round_trip_tokens(self, memory_keyring: _MemoryKeyring) -> None:
        storage = KeyringTokenStorage(alias="linear")
        assert await storage.get_tokens() is None

        tokens = OAuthToken(
            access_token="at",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="rt",
            scope="read write",
        )
        await storage.set_tokens(tokens)

        loaded = await storage.get_tokens()
        assert loaded is not None
        assert loaded.access_token == "at"
        assert loaded.refresh_token == "rt"
        assert loaded.scope == "read write"
        raw = memory_keyring.store[(_KEYRING_SERVICE, "mcp-oauth:linear:tokens")]
        saved_at = json.loads(raw)["expires_at"]
        assert saved_at == pytest.approx(time.time() + 3600, abs=1)

    @pytest.mark.asyncio
    async def test_loading_tokens_restores_absolute_expiry(
        self, memory_keyring: _MemoryKeyring
    ) -> None:
        storage = KeyringTokenStorage(alias="linear")
        tokens = OAuthToken(access_token="at", expires_in=3600, refresh_token="rt")

        with patch("chartreux.core.auth.mcp_oauth.time.time", return_value=1000):
            await storage.set_tokens(tokens)
        with patch("chartreux.core.auth.mcp_oauth.time.time", return_value=1600):
            loaded = await storage.get_tokens()

        assert loaded is not None
        assert storage.token_expiry_time == 4600

    @pytest.mark.asyncio
    async def test_legacy_expiring_tokens_are_considered_expired(
        self, memory_keyring: _MemoryKeyring
    ) -> None:
        storage = KeyringTokenStorage(alias="linear")
        memory_keyring.store[(_KEYRING_SERVICE, "mcp-oauth:linear:tokens")] = (
            OAuthToken(
                access_token="at", expires_in=3600, refresh_token="rt"
            ).model_dump_json()
        )

        loaded = await storage.get_tokens()

        assert loaded is not None
        assert storage.token_expiry_time == -1

    @pytest.mark.asyncio
    async def test_round_trip_client_info(self, memory_keyring: _MemoryKeyring) -> None:
        storage = KeyringTokenStorage(alias="linear")
        assert await storage.get_client_info() is None

        info = OAuthClientInformationFull(
            client_id="abc123",
            redirect_uris=["http://127.0.0.1:47823/callback"],  # type: ignore[list-item]
            token_endpoint_auth_method="none",
        )
        await storage.set_client_info(info)

        loaded = await storage.get_client_info()
        assert loaded is not None
        assert loaded.client_id == "abc123"
        assert (
            _KEYRING_SERVICE,
            "mcp-oauth:linear:client_info",
        ) in memory_keyring.store

    @pytest.mark.asyncio
    async def test_server_issued_secret_is_stored_verbatim(
        self, memory_keyring: _MemoryKeyring
    ) -> None:
        # Supabase-style DCR response: registration asked for "none", the server
        # issued a confidential client with a client_secret and omitted the
        # token_endpoint_auth_method. The storage keeps the server's literal
        # response; the provider derives the auth method per token request.
        storage = KeyringTokenStorage(alias="linear")
        info = OAuthClientInformationFull(
            client_id="confidential",
            client_secret="issued-secret",
            redirect_uris=["http://127.0.0.1:47823/callback"],  # type: ignore[list-item]
            token_endpoint_auth_method=None,
        )
        await storage.set_client_info(info)

        loaded = await storage.get_client_info()

        assert loaded is not None
        assert loaded.token_endpoint_auth_method is None
        assert loaded.client_secret == "issued-secret"

    @pytest.mark.asyncio
    async def test_per_alias_isolation(self, memory_keyring: _MemoryKeyring) -> None:
        a = KeyringTokenStorage(alias="linear")
        b = KeyringTokenStorage(alias="notion")
        await a.set_tokens(
            OAuthToken(access_token="A", token_type="Bearer", expires_in=60)
        )
        await b.set_tokens(
            OAuthToken(access_token="B", token_type="Bearer", expires_in=60)
        )
        loaded_a = await a.get_tokens()
        loaded_b = await b.get_tokens()
        assert loaded_a is not None and loaded_a.access_token == "A"
        assert loaded_b is not None and loaded_b.access_token == "B"

    def test_headless_init_raises(self, headless_keyring: None) -> None:
        with pytest.raises(MCPOAuthHeadlessError) as exc_info:
            KeyringTokenStorage(alias="linear")
        msg = str(exc_info.value)
        assert "linear" in msg
        assert "api_key_env" in msg
        assert exc_info.value.server_alias == "linear"

    def test_unloadable_backend_raises_headless(self) -> None:
        # PYTHON_KEYRING_BACKEND can point at a backend module that is absent from
        # Chartreux's isolated venv (e.g. flyte._keyring.file in Flyte/Slurm envs), so
        # get_keyring() raises ModuleNotFoundError instead of returning a backend.
        with patch(
            "chartreux.core.auth.mcp_oauth.keyring.get_keyring",
            side_effect=ModuleNotFoundError("No module named 'flyte'"),
        ):
            with pytest.raises(MCPOAuthHeadlessError) as exc_info:
                KeyringTokenStorage(alias="notion")
        assert exc_info.value.server_alias == "notion"

    @pytest.mark.asyncio
    async def test_delete_oauth_credentials_wraps_keyring_failure(
        self, memory_keyring: _MemoryKeyring
    ) -> None:
        delete = AsyncMock(side_effect=keyring.errors.KeyringError("keyring locked"))

        with patch("chartreux.core.auth.mcp_oauth._kr_delete", delete):
            with pytest.raises(MCPOAuthCredentialCleanupFailed, match="keyring locked"):
                await delete_oauth_credentials("linear")

    @pytest.mark.asyncio
    async def test_delete_oauth_credentials_is_noop_when_headless(
        self, headless_keyring: None
    ) -> None:
        # No keyring backend means nothing was ever stored, so removing an OAuth
        # server (e.g. added with `--no-login` on CI) must not fail.
        assert await delete_oauth_credentials("linear") is None

    @pytest.mark.asyncio
    async def test_oauth_credential_snapshot_restores_exact_keyring_state(
        self, memory_keyring: _MemoryKeyring
    ) -> None:
        """*Prepare*: One OAuth source has tokens, client registration, and fingerprint.
        *Do*: Snapshot its keyring entries, delete them, and restore the snapshot.
        *Assert*: Every opaque credential entry returns byte-for-byte unchanged.
        """
        # Prepare
        storage = KeyringTokenStorage(alias="linear")
        await storage.set_tokens(
            OAuthToken(
                access_token="access",
                token_type="Bearer",
                expires_in=3600,
                refresh_token="refresh",
            )
        )
        await storage.set_client_info(
            OAuthClientInformationFull(
                client_id="client",
                redirect_uris=["http://127.0.0.1:47823/callback"],  # type: ignore[list-item]
                token_endpoint_auth_method="none",
            )
        )
        await Fingerprint.compute(_oauth_server(name="linear")).save("linear")
        expected = dict(memory_keyring.store)

        # Do
        backup = await snapshot_oauth_credentials("linear")
        await delete_oauth_credentials("linear")
        await restore_oauth_credentials("linear", backup)

        # Assert
        assert memory_keyring.store == expected
