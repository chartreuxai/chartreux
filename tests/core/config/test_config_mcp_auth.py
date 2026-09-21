from __future__ import annotations

from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
import pytest

from chartreux.core.config import MCPHttp, MCPOAuth, MCPStaticAuth
from chartreux.core.config._source_validation import validate_source
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.tools.mcp import AuthStatus
from chartreux.core.tools.mcp.registry import MCPRegistry

HTTP_TRANSPORTS = [pytest.param(MCPHttp, "streamable-http", id="streamable-http")]


@pytest.mark.parametrize(
    "server",
    [
        {"name": "remote", "transport": "http", "url": "https://mcp.example.com"},
        {
            "name": "remote",
            "transport": "streamable-http",
            "url": "https://mcp.example.com",
            "api_key_env": "MCP_TOKEN",
        },
    ],
    ids=["legacy-http-transport", "legacy-top-level-auth"],
)
def test_retired_mcp_configuration_is_rejected_with_source_diagnostics(
    server: dict[str, object],
) -> None:
    with pytest.raises(ValidationError) as caught:
        validate_source(
            ChartreuxConfigSchema, {"mcp_servers": [server]}, source="user-toml"
        )

    assert "user-toml" in str(caught.value)
    assert "mcp_servers" in str(caught.value)


@pytest.mark.parametrize(
    "field", ["headers", "api_key_env", "api_key_header", "api_key_format"]
)
def test_retired_top_level_auth_field_is_not_promoted(field: str) -> None:
    with pytest.raises(ValidationError):
        MCPHttp.model_validate({
            "name": "remote",
            "transport": "streamable-http",
            "url": "https://mcp.example.com",
            field: {} if field == "headers" else "MCP_TOKEN",
        })


@pytest.mark.parametrize("transport", ["http", "stdio"])
def test_mcp_http_requires_canonical_streamable_transport(transport: str) -> None:
    with pytest.raises(ValidationError):
        MCPHttp.model_validate({
            "name": "remote",
            "transport": transport,
            "url": "https://mcp.example.com",
        })


def test_mcp_http_rejects_url_userinfo_before_auth_resolution() -> None:
    with pytest.raises(ValidationError, match="userinfo|credentials"):
        MCPHttp.model_validate({
            "name": "remote",
            "transport": "streamable-http",
            "url": "https://user:password@mcp.example.com/mcp",
            "auth": {"type": "oauth", "scopes": []},
        })


def test_mcp_http_userinfo_source_diagnostics() -> None:
    with pytest.raises(ValidationError) as caught:
        validate_source(
            ChartreuxConfigSchema,
            {
                "mcp_servers": [
                    {
                        "name": "remote",
                        "transport": "streamable-http",
                        "url": "https://user:password@mcp.example.com/mcp",
                    }
                ]
            },
            source="user-toml",
        )
    assert "user-toml" in str(caught.value)
    assert "mcp_servers" in str(caught.value)


def test_api_key_format_accepts_format_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_TOKEN", "k")
    auth = MCPStaticAuth.model_validate({
        "api_key_env": "MCP_TOKEN",
        "api_key_header": "X-API-Key",
        "api_key_format": "{token:>3}",
    })

    assert auth.http_headers() == {"X-API-Key": "  k"}


def test_explicit_static_auth_header_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_TOKEN", "environment-token")
    auth = MCPStaticAuth.model_validate({
        "headers": {"Authorization": "Bearer fixed-token"},
        "api_key_env": "MCP_TOKEN",
    })

    assert auth.http_headers() == {"Authorization": "Bearer fixed-token"}


@pytest.mark.parametrize(("cls", "transport"), HTTP_TRANSPORTS)
def test_explicit_static_auth_round_trips(cls: type[MCPHttp], transport: str) -> None:
    srv = cls.model_validate({
        "name": "remote",
        "transport": transport,
        "url": "https://mcp.example.com",
        "auth": {"type": "static", "api_key_env": "X", "api_key_header": "X-API-Key"},
    })

    dumped = srv.model_dump()
    rebuilt = cls.model_validate(dumped)

    assert isinstance(rebuilt.auth, MCPStaticAuth)
    assert rebuilt.auth.api_key_env == "X"
    assert rebuilt.auth.api_key_header == "X-API-Key"


@pytest.mark.parametrize(("cls", "transport"), HTTP_TRANSPORTS)
def test_oauth_auth_parses(cls: type[MCPHttp], transport: str) -> None:
    srv = cls.model_validate({
        "name": "linear",
        "transport": transport,
        "url": "https://mcp.linear.app/mcp",
        "auth": {"type": "oauth", "scopes": ["read", "write"]},
    })

    assert isinstance(srv.auth, MCPOAuth)
    assert srv.auth.scopes == ["read", "write"]
    assert srv.auth.redirect_port == 47823
    assert srv.http_headers() == {}


@pytest.mark.parametrize(("cls", "transport"), HTTP_TRANSPORTS)
def test_top_level_auth_keys_are_rejected_even_with_auth_block(
    cls: type[MCPHttp], transport: str
) -> None:
    with pytest.raises(ValidationError):
        cls.model_validate({
            "name": "remote",
            "transport": transport,
            "url": "https://mcp.example.com",
            "api_key_env": "LEGACY",
            "auth": {"type": "static", "api_key_env": "NEW"},
        })


@pytest.mark.parametrize(("cls", "transport"), HTTP_TRANSPORTS)
def test_default_auth_is_static(cls: type[MCPHttp], transport: str) -> None:
    srv = cls.model_validate({
        "name": "remote",
        "transport": transport,
        "url": "https://mcp.example.com",
    })

    assert isinstance(srv.auth, MCPStaticAuth)
    assert srv.http_headers() == {}


def test_oauth_client_id_and_metadata_url_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        MCPOAuth.model_validate({
            "type": "oauth",
            "scopes": ["read"],
            "client_id": "abc",
            "client_metadata_url": "https://example.com/cm.json",
        })


def test_oauth_client_metadata_url_must_be_http_url() -> None:
    with pytest.raises(ValidationError):
        MCPOAuth.model_validate({
            "type": "oauth",
            "scopes": ["read"],
            "client_metadata_url": "not-a-url",
        })


def test_oauth_client_id_rejects_empty_string() -> None:
    with pytest.raises(ValidationError):
        MCPOAuth.model_validate({"type": "oauth", "scopes": ["read"], "client_id": ""})


@pytest.mark.parametrize("port", [80, 1023, 0, 65536, 70000])
def test_oauth_redirect_port_out_of_range(port: int) -> None:
    with pytest.raises(ValidationError):
        MCPOAuth.model_validate({
            "type": "oauth",
            "scopes": ["read"],
            "redirect_port": port,
        })


def test_oauth_redirect_port_inside_range() -> None:
    auth = MCPOAuth.model_validate({
        "type": "oauth",
        "scopes": ["read"],
        "redirect_port": 1024,
    })
    assert auth.redirect_port == 1024


def test_static_auth_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        MCPStaticAuth.model_validate({"type": "static", "headerz": {}})


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"api_key_env": "NOT VALID"}, "valid environment variable name"),
        (
            {"api_key_env": "TOKEN", "api_key_header": "Bad Header"},
            "valid HTTP header name",
        ),
        (
            {"api_key_env": "TOKEN", "api_key_format": "Bearer token"},
            "must contain the `{token}` placeholder",
        ),
        (
            {"api_key_env": "TOKEN", "api_key_format": "Bearer {token} {other}"},
            "may only reference the token placeholder",
        ),
        (
            {"api_key_env": "TOKEN", "api_key_format": "Bearer {{token}}"},
            "must contain the `{token}` placeholder",
        ),
        ({"headers": {"X-Tenant": "a", "x-tenant": "b"}}, "Duplicate HTTP header"),
    ],
)
def test_static_auth_rejects_invalid_configuration(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        MCPStaticAuth.model_validate(values)


def test_api_key_header_without_env_loads_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        auth = MCPStaticAuth.model_validate({
            "api_key_header": "X-API-Key",
            "api_key_format": "{token}",
        })

    assert auth.api_key_header == "X-API-Key"
    assert auth.http_headers() == {}
    assert any("api_key_env" in record.message for record in caplog.records)


def test_default_static_auth_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        MCPStaticAuth.model_validate({})

    assert caplog.records == []


def test_oauth_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        MCPOAuth.model_validate({"type": "oauth", "scopes": ["read"], "scope": "x"})


def test_oauth_scopes_required() -> None:
    with pytest.raises(ValidationError):
        MCPOAuth.model_validate({"type": "oauth"})


def test_oauth_scopes_empty_list_allowed() -> None:
    auth = MCPOAuth.model_validate({"type": "oauth", "scopes": []})
    assert auth.scopes == []


@pytest.mark.parametrize(("cls", "transport"), HTTP_TRANSPORTS)
def test_registry_sync_preserves_resolved_oauth_state_without_guessing(
    cls: type[MCPHttp], transport: str
) -> None:
    oauth = cls.model_validate({
        "name": "linear",
        "transport": transport,
        "url": "https://mcp.linear.app/mcp",
        "auth": {"type": "oauth", "scopes": ["read"]},
    })
    registry = MCPRegistry()

    registry.sync_active_servers([oauth])

    assert registry.needs_auth == set()

    registry.mark_needs_auth("linear")
    registry.sync_active_servers([oauth])

    assert registry.needs_auth == {"linear"}

    static = oauth.model_copy(update={"auth": MCPStaticAuth()})
    registry.sync_active_servers([static])

    assert registry.needs_auth == set()
    assert registry.status()["linear"] == AuthStatus.STATIC


@pytest.mark.asyncio
@pytest.mark.parametrize(("cls", "transport"), HTTP_TRANSPORTS)
async def test_unconfigured_registry_marks_oauth_servers_as_needing_auth(
    cls: type[MCPHttp], transport: str
) -> None:
    srv = cls.model_validate({
        "name": "linear",
        "transport": transport,
        "url": "https://mcp.linear.app/mcp",
        "auth": {"type": "oauth", "scopes": ["read"]},
    })
    registry = MCPRegistry()

    with patch(
        "chartreux.core.tools.mcp.registry.list_tools_http", new=AsyncMock()
    ) as discover:
        first = await registry.get_tools_async([srv])
        second = await registry.get_tools_async([srv])

    assert first == {}
    assert second == {}
    discover.assert_not_awaited()
    assert registry.needs_auth == {"linear"}
    assert registry.status()["linear"] == AuthStatus.NEEDS_AUTH
