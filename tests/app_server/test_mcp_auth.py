from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import replace
from functools import partial
import gc
import time
from unittest.mock import AsyncMock, patch

import httpx
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyUrl
import pytest

from chartreux.app_server import _mcp_auth
from chartreux.app_server._mcp_auth import MCPAuthenticationService
from chartreux.app_server._session_backend_port import (
    MCPAuthorizationRequired,
    MCPAuthorizationSnapshot,
)
from chartreux.core.auth import mcp_oauth
from chartreux.core.auth.mcp_oauth import (
    Fingerprint,
    KeyringTokenStorage,
    MCPOAuthCredentialRestoreFailed,
    MCPOAuthError,
    MCPOAuthHeadlessError,
    MCPOAuthInvalidGrant,
    MCPOAuthTransientRefreshError,
)
from chartreux.core.config import MCPHttp, MCPOAuth, MCPStaticAuth, MCPStdio
from chartreux.core.config.types import ConcurrencyConflictError


def _static_server(*, url: str = "https://mcp.example.test") -> MCPHttp:
    return MCPHttp(
        name="linear",
        transport="streamable-http",
        url=url,
        auth=MCPStaticAuth(
            headers={"X-Tenant": "workspace"}, api_key_env="LINEAR_TOKEN"
        ),
    )


def _oauth_server() -> MCPHttp:
    return MCPHttp(
        name="linear",
        transport="streamable-http",
        url="https://mcp.example.test",
        auth=MCPOAuth(type="oauth", scopes=["read"]),
    )


@pytest.mark.asyncio
async def test_static_authorization_resolves_headers_and_environment_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LINEAR_TOKEN", "secret")
    service = MCPAuthenticationService()
    server = _static_server()
    await service.bind_catalog([server])

    result = await service.resolve(service.reference_for(server))

    assert isinstance(result, MCPAuthorizationSnapshot)
    assert result.headers == {"X-Tenant": "workspace", "Authorization": "Bearer secret"}


@pytest.mark.asyncio
async def test_environment_token_change_advances_only_connection_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MCPAuthenticationService()
    server = _static_server()
    await service.bind_catalog([server])
    reference = service.reference_for(server)
    monkeypatch.setenv("LINEAR_TOKEN", "one")
    first = await service.resolve(reference)
    monkeypatch.setenv("LINEAR_TOKEN", "two")
    second = await service.resolve(reference)

    assert isinstance(first, MCPAuthorizationSnapshot)
    assert isinstance(second, MCPAuthorizationSnapshot)
    assert second.connection_revision != first.connection_revision
    assert second.descriptor_revision == first.descriptor_revision
    assert second.headers["Authorization"] == "Bearer two"
    assert first._descriptor_context != second._descriptor_context
    repeated = await service.resolve(reference)
    assert isinstance(repeated, MCPAuthorizationSnapshot)
    assert second._descriptor_context == repeated._descriptor_context
    assert "Bearer two" not in repr(second)
    assert second._descriptor_context not in repr(second)


@pytest.mark.asyncio
async def test_stale_rejection_returns_newer_authorization_without_invalidating_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MCPAuthenticationService()
    server = _static_server()
    await service.bind_catalog([server])
    reference = service.reference_for(server)
    monkeypatch.setenv("LINEAR_TOKEN", "one")
    stale = await service.resolve(reference)
    monkeypatch.setenv("LINEAR_TOKEN", "two")
    current = await service.resolve(reference)
    assert isinstance(stale, MCPAuthorizationSnapshot)
    assert isinstance(current, MCPAuthorizationSnapshot)

    rejected = await service.reject(
        reference,
        observed_connection_revision=stale.connection_revision,
        reason="http_unauthorized",
    )

    assert rejected == current


@pytest.mark.asyncio
async def test_current_static_rejection_advances_descriptor_and_requires_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LINEAR_TOKEN", "one")
    service = MCPAuthenticationService()
    server = _static_server()
    await service.bind_catalog([server])
    reference = service.reference_for(server)
    current = await service.resolve(reference)
    assert isinstance(current, MCPAuthorizationSnapshot)

    rejected = await service.reject(
        reference,
        observed_connection_revision=current.connection_revision,
        reason="mcp_unauthorized",
    )

    assert isinstance(rejected, MCPAuthorizationRequired)
    assert rejected.reason == "rejected"
    assert rejected.observed_connection_revision == current.connection_revision
    assert rejected.descriptor_revision != current.descriptor_revision
    assert isinstance(await service.resolve(reference), MCPAuthorizationRequired)


@pytest.mark.asyncio
async def test_successful_login_with_rejected_token_clears_binding_rejection() -> None:
    service = MCPAuthenticationService()
    server = _oauth_server()
    await service.bind_catalog([server])
    reference = service.reference_for(server)
    fingerprint = Fingerprint.compute(server)
    storage = AsyncMock()
    storage.get_tokens.return_value = OAuthToken(
        access_token="unchanged", token_type="Bearer"
    )
    storage.token_expiry_time = None

    with (
        patch.object(Fingerprint, "load", new=AsyncMock(return_value=fingerprint)),
        patch(
            "chartreux.app_server._mcp_auth.KeyringTokenStorage", return_value=storage
        ),
    ):
        accepted = await service.resolve(reference)
        assert isinstance(accepted, MCPAuthorizationSnapshot)
        rejected = await service.reject(
            reference,
            observed_connection_revision=accepted.connection_revision,
            reason="http_unauthorized",
        )
        assert isinstance(rejected, MCPAuthorizationRequired)
        assert rejected.reason == "rejected"
        with patch(
            "chartreux.app_server._mcp_auth.perform_oauth_login", new=AsyncMock()
        ):
            await service.login(server.name, on_url=AsyncMock())
        resolved = await service.resolve(reference)

    assert isinstance(resolved, MCPAuthorizationSnapshot)
    assert resolved.headers["Authorization"] == "Bearer unchanged"


@pytest.mark.asyncio
async def test_changed_catalog_fingerprint_rejects_stale_reference() -> None:
    service = MCPAuthenticationService()
    original = _static_server()
    await service.bind_catalog([original])
    stale_reference = service.reference_for(original)
    changed = _static_server(url="https://changed.example.test")
    await service.bind_catalog([changed])

    result = await service.resolve(stale_reference)

    assert isinstance(result, MCPAuthorizationRequired)
    assert result.reason == "invalid"


@pytest.mark.asyncio
async def test_credential_removal_rolls_back_keyring_and_authorization_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: An OAuth source with an opaque keyring backup and accepted revision.
    *Do*: Delete credentials, then abort the enclosing config removal with a conflict.
    *Assert*: Credentials and the prior authorization revision are restored.
    """
    # Prepare
    service = MCPAuthenticationService()
    server = _oauth_server()
    await service.bind_catalog([server])
    reference = service.reference_for(server)
    fingerprint = Fingerprint.compute(server)
    storage = AsyncMock()
    storage.get_tokens.return_value = OAuthToken(
        access_token="accepted", token_type="Bearer"
    )
    storage.token_expiry_time = None
    backup = object()
    snapshot = AsyncMock(return_value=backup)
    cleanup = AsyncMock()
    restore = AsyncMock()
    monkeypatch.setattr(_mcp_auth, "snapshot_oauth_credentials", snapshot)
    monkeypatch.setattr(_mcp_auth, "delete_oauth_credentials", cleanup)
    monkeypatch.setattr(_mcp_auth, "restore_oauth_credentials", restore)

    # Do
    with (
        patch.object(Fingerprint, "load", new=AsyncMock(return_value=fingerprint)),
        patch(
            "chartreux.app_server._mcp_auth.KeyringTokenStorage", return_value=storage
        ),
    ):
        previous = await service.resolve(reference)
        with pytest.raises(ConcurrencyConflictError):
            async with service.credential_removal(server.name):
                assert service.descriptor_revision(server.name) != (
                    previous.descriptor_revision
                )
                raise ConcurrencyConflictError("expected", "actual")
        restored = await service.resolve(reference)

    # Assert
    assert isinstance(previous, MCPAuthorizationSnapshot)
    assert restored == previous
    snapshot.assert_awaited_once_with(server.name)
    cleanup.assert_awaited_once_with(server.name)
    restore.assert_awaited_once_with(server.name, backup)


@pytest.mark.asyncio
async def test_credential_removal_restores_authorization_state_when_keyring_restore_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: An OAuth source whose keyring restore fails after config removal aborts.
    *Do*: Exit credential removal through the config failure path.
    *Assert*: In-process authorization state rolls back and both failures remain chained.
    """
    # Prepare
    service = MCPAuthenticationService()
    server = _oauth_server()
    await service.bind_catalog([server])
    previous_revision = service.descriptor_revision(server.name)
    backup = object()
    restore_failure = MCPOAuthCredentialRestoreFailed(
        server_alias=server.name, reason="injected restore failure"
    )
    monkeypatch.setattr(
        _mcp_auth, "snapshot_oauth_credentials", AsyncMock(return_value=backup)
    )
    monkeypatch.setattr(_mcp_auth, "delete_oauth_credentials", AsyncMock())
    monkeypatch.setattr(
        _mcp_auth, "restore_oauth_credentials", AsyncMock(side_effect=restore_failure)
    )

    # Do
    with pytest.raises(MCPOAuthCredentialRestoreFailed) as exc_info:
        async with service.credential_removal(server.name):
            assert service.descriptor_revision(server.name) != previous_revision
            raise ConcurrencyConflictError("expected", "actual")

    # Assert
    assert service.descriptor_revision(server.name) == previous_revision
    assert isinstance(exc_info.value.__context__, ConcurrencyConflictError)


@pytest.mark.asyncio
async def test_stdio_authorization_never_exposes_environment() -> None:
    service = MCPAuthenticationService()
    server = MCPStdio(
        name="local", transport="stdio", command="server", env={"SECRET": "value"}
    )
    await service.bind_catalog([server])

    result = await service.resolve(service.reference_for(server))

    assert isinstance(result, MCPAuthorizationSnapshot)
    assert result.headers == {}


@pytest.mark.asyncio
async def test_oauth_missing_credentials_returns_typed_requirement() -> None:
    service = MCPAuthenticationService()
    server = _oauth_server()
    await service.bind_catalog([server])
    fingerprint = Fingerprint.compute(server)
    storage = AsyncMock()
    storage.get_tokens.return_value = None
    storage.token_expiry_time = None

    with (
        patch.object(Fingerprint, "load", new=AsyncMock(return_value=fingerprint)),
        patch(
            "chartreux.app_server._mcp_auth.KeyringTokenStorage", return_value=storage
        ),
    ):
        result = await service.resolve(service.reference_for(server))

    assert isinstance(result, MCPAuthorizationRequired)
    assert result.reason == "missing"


@pytest.mark.asyncio
async def test_oauth_refresh_publishes_fresh_token_and_expiry() -> None:
    service = MCPAuthenticationService()
    server = _oauth_server()
    await service.bind_catalog([server])
    fingerprint = Fingerprint.compute(server)
    storage = AsyncMock()
    storage.get_tokens.side_effect = [
        OAuthToken(access_token="old", token_type="Bearer"),
        OAuthToken(access_token="fresh", token_type="Bearer"),
    ]
    storage.token_expiry_time = time.time() - 1

    with (
        patch.object(Fingerprint, "load", new=AsyncMock(return_value=fingerprint)),
        patch(
            "chartreux.app_server._mcp_auth.KeyringTokenStorage", return_value=storage
        ),
        patch.object(service, "_refresh_oauth", new=AsyncMock()) as refresh,
    ):
        result = await service.resolve(service.reference_for(server))

    assert isinstance(result, MCPAuthorizationSnapshot)
    assert result.headers["Authorization"] == "Bearer fresh"
    assert result.expires_at is not None
    refresh.assert_awaited_once_with(server)


@pytest.mark.parametrize(
    ("failure", "advances_descriptor"),
    [
        (MCPOAuthInvalidGrant(server_alias="linear", reason="invalid_grant"), True),
        (MCPOAuthTransientRefreshError(server_alias="linear", reason="503"), False),
    ],
)
@pytest.mark.asyncio
async def test_oauth_refresh_failure_is_typed_and_only_invalid_grant_invalidates(
    failure: Exception, advances_descriptor: bool
) -> None:
    service = MCPAuthenticationService()
    server = _oauth_server()
    await service.bind_catalog([server])
    reference = service.reference_for(server)
    before = service.descriptor_revision(server.name)
    fingerprint = Fingerprint.compute(server)
    storage = AsyncMock()
    storage.get_tokens.return_value = OAuthToken(
        access_token="old", token_type="Bearer"
    )
    storage.token_expiry_time = time.time() - 1

    with (
        patch.object(Fingerprint, "load", new=AsyncMock(return_value=fingerprint)),
        patch(
            "chartreux.app_server._mcp_auth.KeyringTokenStorage", return_value=storage
        ),
        patch.object(service, "_refresh_oauth", new=AsyncMock(side_effect=failure)),
    ):
        result = await service.resolve(reference)

    assert isinstance(result, MCPAuthorizationRequired)
    assert result.reason == "expired"
    assert (result.descriptor_revision != before) is advances_descriptor


@pytest.mark.asyncio
async def test_a_config_server_that_declares_static_auth_is_never_late_bound() -> None:
    # Prepare
    service = MCPAuthenticationService()
    await service.bind_catalog([_static_server()])

    # Do / Assert
    with pytest.raises(ValueError, match="not configured for OAuth"):
        await service.login("linear", on_url=AsyncMock())


class _Session:
    """Weak-referenceable owner of a session's configured MCP servers."""


def _configured_figma(
    *, oauth: bool, url: str = "https://figma.internal.test"
) -> MCPHttp:
    return MCPHttp(
        name="figma",
        transport="streamable-http",
        url=url,
        auth=MCPOAuth(type="oauth", scopes=["read"])
        if oauth
        else MCPStaticAuth(headers={"X-Tenant": "workspace"}),
    )


@pytest.mark.asyncio
async def test_same_alias_and_header_names_never_resolve_other_sessions_credentials() -> (
    None
):
    service = MCPAuthenticationService()
    first, second = _Session(), _Session()
    server = MCPHttp(
        name="shared",
        transport="streamable-http",
        url="https://mcp.example.test",
        auth=MCPStaticAuth(headers={"Authorization": "Bearer synthetic-session-a"}),
    )
    other = server.model_copy(
        update={
            "auth": MCPStaticAuth(
                headers={"Authorization": "Bearer synthetic-session-b"}
            )
        }
    )
    await service.bind_catalog([server], owner=first)
    reference = service.reference_for(server, owner=first)
    before = await service.resolve(reference)
    assert isinstance(before, MCPAuthorizationSnapshot)
    assert before.headers == server.http_headers()

    await service.bind_catalog([other], owner=second)
    other_reference = service.reference_for(other, owner=second)
    assert reference.server_fingerprint == other_reference.server_fingerprint
    assert reference.descriptor_revision == other_reference.descriptor_revision

    assert reference.binding_id != other_reference.binding_id
    result = await service.resolve(reference)
    assert isinstance(result, MCPAuthorizationSnapshot)
    assert result.headers == server.http_headers()
    assert result.headers != other.http_headers()


@pytest.mark.asyncio
@pytest.mark.parametrize("other_servers", [[], [_oauth_server()]])
async def test_another_sessions_catalog_preserves_configured_login(
    other_servers: list[MCPHttp],
) -> None:
    service = MCPAuthenticationService()
    first, second = _Session(), _Session()
    server = _configured_figma(oauth=True)
    await service.bind_catalog([server], owner=first)
    reference = service.reference_for(server, owner=first)
    await service.bind_catalog(other_servers, owner=second)
    assert service.reference_for(server, owner=first) == reference
    with patch(
        "chartreux.app_server._mcp_auth.perform_oauth_login", new=AsyncMock()
    ) as login:
        await service.login(server.name, on_url=AsyncMock(), owner=first)
    assert login.await_args is not None
    assert login.await_args.args[0] == server


@pytest.mark.asyncio
async def test_a_shared_definition_outlives_one_session_binding() -> None:
    service = MCPAuthenticationService()
    first, second = _Session(), _Session()
    server = _configured_figma(oauth=False)
    await service.bind_catalog([server], owner=first)
    await service.bind_catalog([server], owner=second)
    reference = service.reference_for(server, owner=second)
    await service.bind_catalog([], owner=first)
    result = await service.resolve(reference)
    assert isinstance(result, MCPAuthorizationSnapshot)
    assert result.headers == {"X-Tenant": "workspace"}
    await service.bind_catalog([], owner=second)
    result = await service.resolve(reference)
    assert isinstance(result, MCPAuthorizationRequired)
    assert result.reason == "invalid"


@pytest.mark.asyncio
async def test_ended_session_revokes_its_configured_login_and_reference() -> None:
    service = MCPAuthenticationService()
    server = _configured_figma(oauth=True)
    ended = _Session()
    await service.bind_catalog([server], owner=ended)
    reference = service.reference_for(server, owner=ended)
    del ended
    gc.collect()
    with pytest.raises(ValueError, match="not configured for OAuth"):
        await service.login(server.name, on_url=AsyncMock())
    result = await service.resolve(reference)
    assert isinstance(result, MCPAuthorizationRequired)
    assert result.reason == "invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize("other_oauth", [True, False])
async def test_login_uses_its_sessions_definition_despite_shared_alias(
    other_oauth: bool,
) -> None:
    service = MCPAuthenticationService()
    first, second = _Session(), _Session()
    server = _configured_figma(oauth=True, url="https://first.example.test")
    other = _configured_figma(oauth=other_oauth)
    await service.bind_catalog([server], owner=first)
    original = service.reference_for(server, owner=first)
    await service.bind_catalog([other], owner=second)
    assert (
        service.reference_for(other, owner=second).server_fingerprint
        != original.server_fingerprint
    )
    assert service.descriptor_revision(server.name) != original.descriptor_revision
    with patch(
        "chartreux.app_server._mcp_auth.perform_oauth_login", new=AsyncMock()
    ) as login:
        await service.login(server.name, on_url=AsyncMock(), owner=first)
    assert login.await_args is not None
    assert login.await_args.args[0] == server
    assert login.await_args.kwargs["headers"] == server.http_headers()
    if not other_oauth:
        with pytest.raises(ValueError, match="not configured for OAuth"):
            await service.login(other.name, on_url=AsyncMock(), owner=second)


@pytest.mark.asyncio
async def test_same_alias_different_endpoint_keeps_each_sessions_snapshot() -> None:
    service = MCPAuthenticationService()
    first, second = _Session(), _Session()
    server = _configured_figma(oauth=False, url="https://first.example.test")
    other = _configured_figma(oauth=False)
    await service.bind_catalog([server], owner=first)
    original = service.reference_for(server, owner=first)
    before = await service.resolve(original)
    await service.bind_catalog([other], owner=second)
    replacement = service.reference_for(other, owner=second)

    first_current = await service.resolve(original)
    second_current = await service.resolve(replacement)

    assert isinstance(before, MCPAuthorizationSnapshot)
    assert isinstance(first_current, MCPAuthorizationSnapshot)
    assert isinstance(second_current, MCPAuthorizationSnapshot)
    assert first_current.descriptor_revision == original.descriptor_revision
    assert first_current.headers == server.http_headers()
    assert second_current.headers == other.http_headers()


@pytest.mark.asyncio
async def test_same_endpoint_credential_rebind_invalidates_old_binding() -> None:
    service = MCPAuthenticationService()
    owner = _Session()
    server = _configured_figma(oauth=False)
    await service.bind_catalog([server], owner=owner)
    original = service.reference_for(server, owner=owner)
    assert isinstance(await service.resolve(original), MCPAuthorizationSnapshot)

    rotated = server.model_copy(
        update={"auth": MCPStaticAuth(headers={"X-Tenant": "rotated"})}
    )
    await service.bind_catalog([rotated], owner=owner)

    stale = await service.resolve(original)
    current = await service.resolve(service.reference_for(rotated, owner=owner))
    assert isinstance(stale, MCPAuthorizationRequired)
    assert stale.reason == "invalid"
    assert isinstance(current, MCPAuthorizationSnapshot)
    assert current.headers == {"X-Tenant": "rotated"}


@pytest.mark.asyncio
async def test_a_second_sessions_configured_servers_do_not_evict_the_firsts() -> None:
    service = MCPAuthenticationService()
    first, second = _Session(), _Session()
    server = _static_server()
    await service.bind_catalog([server], owner=first)
    reference = service.reference_for(server, owner=first)
    await service.bind_catalog([], owner=second)
    assert isinstance(await service.resolve(reference), MCPAuthorizationSnapshot)


@pytest.mark.asyncio
async def test_a_shared_name_never_returns_or_deletes_the_other_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MCPAuthenticationService()
    first, second = _Session(), _Session()
    original = _configured_figma(oauth=True, url="https://first.example.test")
    configured = _configured_figma(oauth=True)
    await service.bind_catalog([original], owner=first)
    stale_reference = service.reference_for(original, owner=first)
    await service.bind_catalog([configured], owner=second)
    reference = service.reference_for(configured, owner=second)
    assert reference.server_fingerprint != stale_reference.server_fingerprint
    storage = AsyncMock()
    storage.get_tokens.return_value = OAuthToken(
        access_token="other-session-grant", token_type="Bearer"
    )
    storage.token_expiry_time = None
    cleanup = AsyncMock()
    monkeypatch.setattr(_mcp_auth, "delete_oauth_credentials", cleanup)
    with (
        patch.object(
            Fingerprint,
            "load",
            new=AsyncMock(return_value=Fingerprint.compute(original)),
        ),
        patch(
            "chartreux.app_server._mcp_auth.KeyringTokenStorage", return_value=storage
        ),
    ):
        result = await service.resolve(reference)
        stale = await service.resolve(stale_reference)
    assert isinstance(result, MCPAuthorizationRequired)
    assert result.reason == "invalid"
    assert isinstance(stale, MCPAuthorizationSnapshot)
    assert stale.headers == {"Authorization": "Bearer other-session-grant"}
    cleanup.assert_not_awaited()
    storage.delete_tokens.assert_not_awaited()


@pytest.mark.asyncio
async def test_shared_alias_rejection_revokes_reference_without_deleting_other_grant() -> (
    None
):
    service = MCPAuthenticationService()
    first, second = _Session(), _Session()
    original = _configured_figma(oauth=True, url="https://first.example.test")
    server = _configured_figma(oauth=True)
    await service.bind_catalog([original], owner=first)
    await service.bind_catalog([server], owner=second)
    reference = service.reference_for(server, owner=second)
    storage = AsyncMock()
    storage.get_tokens.return_value = OAuthToken(
        access_token="accepted", token_type="Bearer"
    )
    storage.token_expiry_time = None
    with (
        patch.object(
            Fingerprint, "load", new=AsyncMock(return_value=Fingerprint.compute(server))
        ),
        patch(
            "chartreux.app_server._mcp_auth.KeyringTokenStorage", return_value=storage
        ),
    ):
        accepted = await service.resolve(reference)
        assert isinstance(accepted, MCPAuthorizationSnapshot)
        rejected = await service.reject(
            reference,
            observed_connection_revision=accepted.connection_revision,
            reason="http_unauthorized",
        )
        assert isinstance(rejected, MCPAuthorizationRequired)
        assert rejected.reason == "rejected"
        assert rejected.descriptor_revision != reference.descriptor_revision
        stale = await service.resolve(reference)
        assert isinstance(stale, MCPAuthorizationRequired)
        assert stale.reason == "rejected"
        current = await service.resolve(service.reference_for(server, owner=second))
        assert isinstance(current, MCPAuthorizationRequired)
        assert current.reason == "rejected"
    storage.delete_tokens.assert_not_awaited()


@pytest.mark.asyncio
async def test_static_configured_server_keeps_headers_without_headless_oauth() -> None:
    service = MCPAuthenticationService()
    server = _configured_figma(oauth=False)
    await service.bind_catalog([server])
    with patch(
        "chartreux.app_server._mcp_auth.KeyringTokenStorage",
        side_effect=MCPOAuthHeadlessError(server_alias=server.name),
    ) as storage:
        result = await service.resolve(service.reference_for(server))
    assert isinstance(result, MCPAuthorizationSnapshot)
    assert result.headers == {"X-Tenant": "workspace"}
    assert "Authorization" not in result.headers
    storage.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["headers", "declaration", "mutation", "remove", "gc"]
)
async def test_binding_revocation_never_resolves_or_rejects_surviving_owner(
    change: str,
) -> None:
    service = MCPAuthenticationService()
    owner, survivor = _Session(), _Session()
    server = _configured_figma(oauth=False)
    other = server.model_copy(deep=True)
    await service.bind_catalog([server], owner=owner)
    reference = service.reference_for(server, owner=owner)
    accepted = await service.resolve(reference)
    assert isinstance(accepted, MCPAuthorizationSnapshot)
    await service.bind_catalog([other], owner=survivor)
    surviving_reference = service.reference_for(other, owner=survivor)
    if change in {"headers", "mutation"}:
        assert isinstance(server.auth, MCPStaticAuth)
        server.auth.headers["X-Tenant"] = "replacement"
        if change == "headers":
            await service.bind_catalog([server], owner=owner)
            replacement = service.reference_for(server, owner=owner)
            assert replacement.binding_id != reference.binding_id
            assert replacement.server_fingerprint == reference.server_fingerprint
            assert replacement.descriptor_revision == reference.descriptor_revision
    elif change == "declaration":
        await service.bind_catalog(
            [server.model_copy(update={"disabled": True})], owner=owner
        )
    elif change == "remove":
        await service.bind_catalog([], owner=owner)
        await service.bind_catalog([server], owner=owner)
        assert (
            service.reference_for(server, owner=owner).binding_id
            != reference.binding_id
        )
    else:
        del owner
        gc.collect()
    result = await service.resolve(reference)
    assert isinstance(result, MCPAuthorizationRequired)
    assert result.reason == "invalid"
    rejected = await service.reject(
        reference,
        observed_connection_revision=accepted.connection_revision,
        reason="http_unauthorized",
    )
    assert rejected == result
    surviving = await service.resolve(surviving_reference)
    assert isinstance(surviving, MCPAuthorizationSnapshot)
    assert surviving.headers == other.http_headers()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["resolve", "reject", "reject_current"])
@pytest.mark.parametrize("change", ["remove", "replacement", "gc", "mutation"])
@pytest.mark.parametrize(
    "pause",
    [
        "tokens",
        "expired_tokens",
        "refresh",
        "fingerprint",
        "refreshed_tokens",
        "invalid_grant",
        "transient_refresh",
        "mismatched_tokens",
    ],
)
async def test_oauth_await_revalidates_live_binding(
    operation: str, change: str, pause: str
) -> None:
    service = MCPAuthenticationService()
    owner, survivor = _Session(), _Session()
    server = _oauth_server()
    other = server.model_copy(deep=True)
    await service.bind_catalog([server], owner=owner)
    await service.bind_catalog([other], owner=survivor)
    reference = service.reference_for(server, owner=owner)
    entered, release = asyncio.Event(), asyncio.Event()
    token = OAuthToken(access_token="synthetic", token_type="Bearer")
    reads = 0
    refreshing = pause in {
        "refresh",
        "refreshed_tokens",
        "invalid_grant",
        "transient_refresh",
    }

    async def paused_tokens() -> OAuthToken:
        nonlocal reads
        reads += 1
        if pause in {"tokens", "expired_tokens", "mismatched_tokens"} or (
            pause == "refreshed_tokens" and reads == 2
        ):
            entered.set()
            await release.wait()
        return token

    async def paused_fingerprint(_name: str) -> Fingerprint:
        if pause == "fingerprint":
            entered.set()
            await release.wait()
        if pause == "mismatched_tokens":
            return Fingerprint.compute(
                other.model_copy(update={"url": "https://other.test"})
            )
        return Fingerprint.compute(other)

    async def paused_refresh(_server: MCPHttp) -> None:
        if pause in {"refresh", "invalid_grant", "transient_refresh"}:
            entered.set()
            await release.wait()
        if pause == "invalid_grant":
            raise MCPOAuthInvalidGrant(server_alias=server.name, reason="invalid_grant")
        if pause == "transient_refresh":
            raise MCPOAuthTransientRefreshError(server_alias=server.name, reason="503")

    storage = AsyncMock()
    storage.get_tokens.side_effect = paused_tokens
    storage.token_expiry_time = (
        time.time() - 1 if pause == "expired_tokens" or refreshing else None
    )
    with (
        patch.object(
            Fingerprint, "load", new=AsyncMock(side_effect=paused_fingerprint)
        ),
        patch(
            "chartreux.app_server._mcp_auth.KeyringTokenStorage", return_value=storage
        ),
        patch.object(
            service, "_refresh_oauth", new=AsyncMock(side_effect=paused_refresh)
        ) as refresh,
        patch(
            "chartreux.app_server._mcp_auth.delete_oauth_credentials", new=AsyncMock()
        ) as cleanup,
    ):
        work = asyncio.create_task(
            service.resolve(reference)
            if operation == "resolve"
            else service.reject(
                reference,
                observed_connection_revision=(
                    "mcp-auth-connection:linear:1"
                    if operation == "reject_current"
                    else "older"
                ),
                reason="http_unauthorized",
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            if change == "remove":
                await service.bind_catalog([], owner=owner)
            elif change == "replacement":
                await service.bind_catalog(
                    [server.model_copy(update={"disabled": True})], owner=owner
                )
            elif change == "mutation":
                server.disabled = True
            else:
                del owner
                gc.collect()
            assert (
                service.descriptor_revision(server.name)
                == reference.descriptor_revision
            )
        finally:
            release.set()
        result = await asyncio.wait_for(work, 2)
        assert isinstance(result, MCPAuthorizationRequired)
        assert result.reason == "invalid"
        assert service.connection_revision(server.name).endswith(":0")
        assert service.descriptor_revision(server.name) == reference.descriptor_revision
        storage.delete_tokens.assert_not_awaited()
        cleanup.assert_not_awaited()
        assert refresh.await_count == int(refreshing)
        if pause == "fingerprint":
            storage.get_tokens.assert_not_awaited()


@pytest.mark.asyncio
async def test_equal_rebind_does_not_mutate_inflight_oauth_declaration() -> None:
    service = MCPAuthenticationService()
    owner = _Session()
    server = _oauth_server()
    expected = server.model_copy(deep=True)
    await service.bind_catalog([server], owner=owner)
    reference = service.reference_for(server, owner=owner)
    entered, release = asyncio.Event(), asyncio.Event()

    async def paused_tokens() -> OAuthToken:
        entered.set()
        await release.wait()
        return OAuthToken(access_token="synthetic", token_type="Bearer")

    storage = AsyncMock()
    storage.get_tokens.side_effect = paused_tokens
    storage.token_expiry_time = time.time() - 1
    with (
        patch.object(
            Fingerprint, "load", new=AsyncMock(return_value=Fingerprint.compute(server))
        ),
        patch(
            "chartreux.app_server._mcp_auth.KeyringTokenStorage", return_value=storage
        ),
        patch.object(service, "_refresh_oauth", new=AsyncMock()) as refresh,
    ):
        pending = asyncio.create_task(service.resolve(reference))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await service.bind_catalog([expected], owner=owner)
            assert service.reference_for(expected, owner=owner) == reference
            server.url = "https://detached.example.test"
        finally:
            release.set()
        result = await asyncio.wait_for(pending, 2)
        assert isinstance(result, MCPAuthorizationSnapshot)
        refresh.assert_awaited_once_with(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["remove", "replacement"])
@pytest.mark.parametrize("operation", ["login", "logout", "removal"])
async def test_credential_work_revalidates_binding_after_await(
    operation: str, change: str
) -> None:
    service = MCPAuthenticationService()
    owner, survivor = _Session(), _Session()
    server = _oauth_server()
    await service.bind_catalog([server], owner=owner)
    await service.bind_catalog([server.model_copy(deep=True)], owner=survivor)
    before = service.descriptor_revision(server.name)
    entered, release = asyncio.Event(), asyncio.Event()

    async def paused(*_args: object, **_kwargs: object) -> None:
        entered.set()
        await release.wait()

    async def work() -> None:
        if operation == "login":
            await service.login(server.name, owner=owner, on_url=AsyncMock())
        elif operation == "logout":
            await service.logout(server.name, owner=owner)
        else:
            async with service.credential_removal(server.name, owner=owner):
                pytest.fail("Revoked removal must not reach config persistence")

    with (
        patch(
            "chartreux.app_server._mcp_auth.perform_oauth_login",
            new=AsyncMock(side_effect=paused),
        ) as login,
        patch(
            "chartreux.app_server._mcp_auth.snapshot_oauth_credentials",
            new=AsyncMock(side_effect=paused),
        ),
        patch(
            "chartreux.app_server._mcp_auth.delete_oauth_credentials",
            new=AsyncMock(side_effect=paused),
        ) as cleanup,
        patch(
            "chartreux.app_server._mcp_auth.restore_oauth_credentials", new=AsyncMock()
        ) as restore,
    ):
        pending = asyncio.create_task(work())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await service.bind_catalog(
                []
                if change == "remove"
                else [server.model_copy(update={"disabled": True})],
                owner=owner,
            )
        finally:
            release.set()
        with pytest.raises(ValueError, match="Unknown MCP server"):
            await asyncio.wait_for(pending, 2)
        assert service.descriptor_revision(server.name) == before
        assert login.await_count == int(operation == "login")
        assert cleanup.await_count == int(operation == "logout")
        restore.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_login_does_not_authorize_replacement_and_other_alias_progresses() -> (
    None
):
    service = MCPAuthenticationService()
    owner = _Session()
    server = _oauth_server()
    await service.bind_catalog([server], owner=owner)
    entered = asyncio.Event()

    async def login() -> str:
        entered.set()
        return await service.login(server.name, owner=owner, on_url=AsyncMock())

    with patch(
        "chartreux.app_server._mcp_auth.perform_oauth_login", new=AsyncMock()
    ) as authorize:
        async with service._lock(server.name):
            pending = asyncio.create_task(login())
            await asyncio.wait_for(entered.wait(), 2)
            await service.bind_catalog(
                [server.model_copy(update={"disabled": True})], owner=owner
            )
            unrelated = _static_server().model_copy(update={"name": "unrelated"})
            await asyncio.wait_for(service.bind_catalog([unrelated]), 2)
            result = await asyncio.wait_for(
                service.resolve(service.reference_for(unrelated)), 2
            )
            assert isinstance(result, MCPAuthorizationSnapshot)
        with pytest.raises(ValueError, match="Unknown MCP server"):
            await asyncio.wait_for(pending, 2)
        authorize.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_oauth_resolution_releases_alias_lock() -> None:
    service = MCPAuthenticationService()
    server = _oauth_server()
    await service.bind_catalog([server])
    reference = service.reference_for(server)
    entered = asyncio.Event()

    async def paused() -> None:
        entered.set()
        await asyncio.Event().wait()

    storage = AsyncMock()
    storage.get_tokens.side_effect = paused
    with (
        patch.object(
            Fingerprint, "load", new=AsyncMock(return_value=Fingerprint.compute(server))
        ),
        patch(
            "chartreux.app_server._mcp_auth.KeyringTokenStorage", return_value=storage
        ),
        patch.object(service, "_refresh_oauth", new=AsyncMock()) as refresh,
    ):
        pending = asyncio.create_task(service.resolve(reference))
        await asyncio.wait_for(entered.wait(), 2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not service._lock(server.name).locked()
        await service.bind_catalog([])
        result = await asyncio.wait_for(service.resolve(reference), 2)
        assert isinstance(result, MCPAuthorizationRequired)
        assert result.reason == "invalid"
        refresh.assert_not_awaited()
        storage.delete_tokens.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse_owners", [False, True])
@pytest.mark.parametrize("same_identity", [False, True])
@pytest.mark.parametrize(
    "outcome",
    ["http_error", "cancel", "remove", "replacement", "write_cancel", "write_remove"],
)
async def test_failed_shared_alias_login_preserves_other_credential_identity(
    reverse_owners: bool,
    same_identity: bool,
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only the keyring boundary is fake; login, expiry restoration and SDK refresh
    # all run normally. No host keyring, browser, callback socket or HTTP is used.
    entries: dict[str, str] = {}
    monkeypatch.setattr(mcp_oauth.keyring, "get_keyring", lambda: object())
    monkeypatch.setattr(mcp_oauth, "_kr_get", AsyncMock(side_effect=entries.get))
    monkeypatch.setattr(
        mcp_oauth, "_kr_set", AsyncMock(side_effect=entries.__setitem__)
    )
    monkeypatch.setattr(
        mcp_oauth,
        "_kr_delete",
        AsyncMock(side_effect=lambda key: entries.pop(key, None)),
    )
    service = MCPAuthenticationService()
    owner, survivor = _Session(), _Session()
    original = _oauth_server()
    candidate = original.model_copy(
        update={} if same_identity else {"url": "https://candidate.example.test/mcp"}
    )
    bindings = [(original, survivor), (candidate, owner)]
    for server, bound_owner in reversed(bindings) if reverse_owners else bindings:
        await service.bind_catalog([server], owner=bound_owner)
    storage = KeyringTokenStorage(alias=original.name)
    with patch.object(mcp_oauth.time, "time", return_value=1000):
        await storage.set_tokens(
            OAuthToken(
                access_token="synthetic-other-access",
                refresh_token="synthetic-other-refresh",
                expires_in=60,
            )
        )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="synthetic-other-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:47823/callback")],
            token_endpoint_auth_method="none",
        )
    )
    await Fingerprint.compute(original).save(original.name)
    before = dict(entries)
    entered, release = asyncio.Event(), asyncio.Event()
    requests: list[httpx.Request] = []

    if outcome.startswith("write_"):

        async def paused_write(key: str, value: str) -> None:
            entries[key] = value
            if "synthetic-candidate-access" in value:
                entered.set()
                await release.wait()

        monkeypatch.setattr(mcp_oauth, "_kr_set", paused_write)

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "synthetic-candidate-access",
                    "refresh_token": "synthetic-candidate-refresh",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        assert str(request.url) == candidate.url
        if not outcome.startswith("write_"):
            entered.set()
            await release.wait()
        return httpx.Response(503 if outcome == "http_error" else 200)

    monkeypatch.setattr(mcp_oauth, "build_ssl_context", lambda: True)
    monkeypatch.setattr(
        mcp_oauth,
        "ChartreuxAsyncHTTPClient",
        partial(
            httpx.AsyncClient, transport=httpx.MockTransport(respond), trust_env=False
        ),
    )

    async def login() -> str | None:
        try:
            return await service.login(candidate.name, owner=owner, on_url=AsyncMock())
        except (MCPOAuthError, ValueError):
            return None

    pending = asyncio.create_task(login())
    arrived = asyncio.create_task(entered.wait())
    rebind: asyncio.Task[None] | None = None
    try:
        # A safe implementation may refuse the identity conflict before any I/O.
        done, _ = await asyncio.wait(
            {pending, arrived}, timeout=2, return_when=asyncio.FIRST_COMPLETED
        )
        assert done, "Login neither refused the conflict nor reached the fake transport"
        if not pending.done():
            if outcome in {"cancel", "write_cancel"}:
                pending.cancel()
            elif outcome in {"remove", "replacement", "write_remove"}:
                rebinding = asyncio.Event()

                async def change_catalog() -> None:
                    rebinding.set()
                    # bind_catalog revokes the binding before awaiting the alias lock.
                    await service.bind_catalog(
                        []
                        if outcome in {"remove", "write_remove"}
                        else [candidate.model_copy(update={"disabled": True})],
                        owner=owner,
                    )

                rebind = asyncio.create_task(change_catalog())
                await asyncio.wait_for(rebinding.wait(), 2)
            release.set()
        if (
            pending.cancelled()
            or outcome in {"cancel", "write_cancel"}
            and entered.is_set()
        ):
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 2)
        else:
            assert await asyncio.wait_for(pending, 2) is None
        if rebind is not None:
            await asyncio.wait_for(rebind, 2)
    finally:
        release.set()
        tasks = [pending, arrived]
        if rebind is not None:
            tasks.append(rebind)
        for task in tasks:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    assert entries == before
    if same_identity:
        assert entered.is_set()
        assert any(
            b"synthetic-other-refresh" in request.content for request in requests
        )
        assert (
            requests[-1].headers["authorization"] == "Bearer synthetic-candidate-access"
        )
    else:
        assert all(
            b"synthetic-other-refresh" not in request.content for request in requests
        )
        assert all(
            "synthetic-other-access" not in request.headers.get("authorization", "")
            for request in requests
        )


@pytest.mark.asyncio
async def test_default_owner_and_unknown_owner_never_fall_back_to_session() -> None:
    service = MCPAuthenticationService()
    owner, unknown = _Session(), _Session()
    server = _oauth_server()
    await service.bind_catalog([server], owner=owner)
    for candidate in (None, unknown):
        with pytest.raises(ValueError, match="Unknown MCP server"):
            service.reference_for(server, owner=candidate)
        with pytest.raises(ValueError, match="not configured for OAuth"):
            await service.login(server.name, on_url=AsyncMock(), owner=candidate)
    await service.bind_catalog([server])
    reference = service.reference_for(server)
    session_reference = service.reference_for(server, owner=owner)
    assert reference.binding_id != session_reference.binding_id
    await service.bind_catalog([server.model_copy(deep=True)])
    assert service.reference_for(server) == reference
    for invalid in (
        replace(reference, binding_id=""),
        replace(reference, binding_id="unknown"),
    ):
        result = await service.resolve(invalid)
        assert isinstance(result, MCPAuthorizationRequired)
        assert result.reason == "invalid"
    separate = MCPAuthenticationService()
    await separate.bind_catalog([server])
    result = await separate.resolve(reference)
    assert isinstance(result, MCPAuthorizationRequired)
    assert result.reason == "invalid"
