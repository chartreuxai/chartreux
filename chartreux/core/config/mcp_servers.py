from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import SplitResult, urlsplit, urlunsplit

from pydantic import TypeAdapter, ValidationError

from chartreux.core.config._mcp_save import (
    MCPApply,
    MCPPreflight,
    MCPSaveError,
    accepted_mcp_servers,
    save_mcp_servers,
)
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import (
    MCPHttp,
    MCPOAuth,
    MCPServer,
    MCPStdio,
    normalize_mcp_server_name,
)
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.types import ConfigSaveResult
from chartreux.utils.mcp import MCPAddTransport


class MCPServerAddError(MCPSaveError):
    pass


class MCPServerRemoveError(MCPSaveError):
    pass


@dataclass(frozen=True)
class PersistedMCPServerResult[ServerT: MCPHttp | MCPStdio]:
    server: ServerT
    created: bool
    save_result: ConfigSaveResult | None = None


@dataclass(frozen=True)
class RemovedMCPServerResult:
    name: str
    server: MCPServer | None
    removed: bool
    save_result: ConfigSaveResult | None = None


_DEFAULT_PORTS = {"http": 80, "https": 443}
_HOST_PREFIXES_TO_DROP = {"mcp", "www"}
_GENERIC_ALIAS_SEGMENTS = {"api", "mcp", "server"}
_LEADING_HOST_PREFIX_LABEL_MIN_COUNT = 3
_MCP_SERVER_ADAPTER = TypeAdapter(MCPServer)


async def persist_oauth_mcp_server(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    *,
    url: str,
    name: str | None = None,
    scopes: list[str] | None = None,
    transport: MCPAddTransport = "streamable-http",
    preflight: MCPPreflight | None = None,
    apply: MCPApply | None = None,
) -> PersistedMCPServerResult[MCPHttp]:
    normalized_url = normalize_mcp_server_url(url)
    requested_name = normalize_mcp_server_name(name) if name is not None else None
    if name is not None and not requested_name:
        raise MCPServerAddError("MCP server name must contain letters or numbers.")

    active_servers = list(orchestrator.config.mcp_servers)
    if existing := _find_server_url(active_servers, normalized_url):
        if not isinstance(existing.auth, MCPOAuth):
            raise MCPServerAddError(
                f"MCP server URL is already configured as `{existing.name}` "
                "with static auth. `/mcp add` only supports OAuth MCP servers."
            )
        if requested_name is not None and requested_name != existing.name:
            raise MCPServerAddError(
                f"MCP server URL is already configured as `{existing.name}`."
            )
        return PersistedMCPServerResult(server=existing, created=False)

    server_name = _resolve_new_server_name(
        requested_name, normalized_url, {server.name for server in active_servers}
    )
    model = MCPHttp
    try:
        server = model.model_validate({
            "name": server_name,
            "transport": transport,
            "url": normalized_url,
            "auth": {"type": "oauth", "scopes": scopes or []},
        })
    except ValidationError:
        raise MCPServerAddError("Invalid MCP server configuration.") from None
    return await persist_remote_mcp_server(
        orchestrator, server, preflight=preflight, apply=apply
    )


async def persist_remote_mcp_server(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    server: MCPHttp,
    *,
    preflight: MCPPreflight | None = None,
    apply: MCPApply | None = None,
) -> PersistedMCPServerResult[MCPHttp]:
    normalized_server = server.model_copy(
        update={"url": normalize_mcp_server_url(server.url)}
    )
    current_servers = list(orchestrator.config.mcp_servers)
    if existing := _find_server_name(current_servers, normalized_server.name):
        if isinstance(existing, MCPHttp):
            if _remote_servers_equivalent(existing, normalized_server):
                return PersistedMCPServerResult(server=existing, created=False)
            if _url_key(existing.url) == _url_key(normalized_server.url):
                raise MCPServerAddError(
                    f"MCP server `{normalized_server.name}` is already configured "
                    "with different options."
                )
        raise MCPServerAddError(
            f"MCP server name `{normalized_server.name}` is already configured."
        )
    if existing := _find_server_url(current_servers, normalized_server.url):
        raise MCPServerAddError(
            f"MCP server URL is already configured as `{existing.name}`."
        )

    saved = await _append_persisted_mcp_server(
        orchestrator, normalized_server, preflight=preflight, apply=apply
    )
    return PersistedMCPServerResult(
        server=normalized_server, created=True, save_result=saved
    )


async def persist_stdio_mcp_server(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    server: MCPStdio,
    *,
    preflight: MCPPreflight | None = None,
    apply: MCPApply | None = None,
) -> PersistedMCPServerResult[MCPStdio]:
    if existing := _find_server_name(
        list(orchestrator.config.mcp_servers), server.name
    ):
        if isinstance(existing, MCPStdio) and existing == server:
            return PersistedMCPServerResult(server=existing, created=False)
        raise MCPServerAddError(
            f"MCP server name `{server.name}` is already configured."
        )
    saved = await _append_persisted_mcp_server(
        orchestrator, server, preflight=preflight, apply=apply
    )
    return PersistedMCPServerResult(server=server, created=True, save_result=saved)


async def _append_persisted_mcp_server(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    server: MCPServer,
    *,
    preflight: MCPPreflight | None = None,
    apply: MCPApply | None = None,
) -> ConfigSaveResult:
    raw_servers, revision = accepted_mcp_servers(orchestrator, MCPServerAddError)
    return await save_mcp_servers(
        orchestrator,
        [*raw_servers, _serialize_mcp_server(server)],
        revision=revision,
        reason="Add MCP server",
        error_type=MCPServerAddError,
        preflight=preflight,
        apply=apply,
    )


async def remove_mcp_server(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    name: str,
    *,
    preflight: MCPPreflight | None = None,
    apply: MCPApply | None = None,
) -> RemovedMCPServerResult:
    normalized_name = normalize_mcp_server_name(name)
    if not normalized_name:
        raise MCPServerRemoveError("MCP server name must contain letters or numbers.")

    raw_servers, revision = accepted_mcp_servers(orchestrator, MCPServerRemoveError)
    persisted = _find_raw_server(raw_servers, normalized_name)
    if persisted is None:
        return RemovedMCPServerResult(name=normalized_name, server=None, removed=False)
    try:
        removed_server = _parse_raw_mcp_server(persisted)
    except ValidationError:
        raise MCPServerRemoveError(
            "MCP server in the persistence layer is invalid."
        ) from None

    saved = await save_mcp_servers(
        orchestrator,
        [
            server
            for server in raw_servers
            if _raw_server_name(server) != normalized_name
        ],
        revision=revision,
        reason="Remove MCP server",
        error_type=MCPServerRemoveError,
        preflight=preflight,
        apply=apply,
    )
    return RemovedMCPServerResult(
        name=normalized_name, server=removed_server, removed=True, save_result=saved
    )


def _serialize_mcp_server(server: MCPServer) -> dict[str, object]:
    data = server.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    if isinstance(server, MCPHttp):
        # exclude_defaults drops MCPStaticAuth's `type="static"`, but the persisted
        # auth union is discriminated by `type`, so re-add it.
        data["auth"] = {**data.get("auth", {}), "type": server.auth.type}
    return data


def normalize_mcp_server_url(value: str) -> str:
    parsed = _parse_mcp_server_url(value)
    return _url_with_normalized_host(parsed, trim_trailing_slash=False)


def _resolve_new_server_name(
    requested_name: str | None, normalized_url: str, active_names: set[str]
) -> str:
    if requested_name is None:
        return _dedupe_server_name(_suggest_server_name(normalized_url), active_names)
    if requested_name in active_names:
        raise MCPServerAddError(
            f"MCP server name `{requested_name}` is already configured."
        )
    return requested_name


def _parse_mcp_server_url(value: str) -> SplitResult:
    raw_url = value.strip()
    if not raw_url:
        raise MCPServerAddError("MCP server URL is required.")
    try:
        parsed = urlsplit(raw_url)
        host = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise MCPServerAddError("MCP server URL must be a valid HTTP(S) URL.") from None
    scheme = parsed.scheme.lower()
    if not scheme:
        raise MCPServerAddError("MCP server URL must include a scheme.")
    if scheme not in {"http", "https"}:
        raise MCPServerAddError("MCP server URL must use https.")
    if not host:
        raise MCPServerAddError("MCP server URL must include a host.")
    if parsed.fragment:
        raise MCPServerAddError("MCP server URL must not include a fragment.")
    if parsed.username is not None or parsed.password is not None:
        raise MCPServerAddError("MCP server URL must not include credentials.")
    if scheme == "http" and not _is_loopback_host(host):
        raise MCPServerAddError(
            "MCP server URL must use https unless it points to localhost."
        )
    return parsed


def _find_server_name(servers: list[MCPServer], name: str) -> MCPServer | None:
    return next((server for server in servers if server.name == name), None)


def _raw_server_name(raw_server: object) -> str | None:
    if not isinstance(raw_server, dict):
        return None
    name = raw_server.get("name")
    return normalize_mcp_server_name(name) if isinstance(name, str) else None


def _find_raw_server(raw_servers: object, name: str) -> object | None:
    if not isinstance(raw_servers, list):
        return None
    return next(
        (server for server in raw_servers if _raw_server_name(server) == name), None
    )


def _parse_raw_mcp_server(raw_server: object) -> MCPServer:
    return _MCP_SERVER_ADAPTER.validate_python(raw_server)


def _find_server_url(servers: list[MCPServer], url: str) -> MCPHttp | None:
    url_key = _url_key(url)
    return next(
        (
            server
            for server in servers
            if isinstance(server, MCPHttp) and _url_key(server.url) == url_key
        ),
        None,
    )


def _remote_servers_equivalent(existing: MCPHttp, requested: MCPHttp) -> bool:
    if _url_key(existing.url) != _url_key(requested.url):
        return False
    return existing.model_copy(update={"url": requested.url}) == requested


def _url_key(value: str) -> str:
    return _url_with_normalized_host(urlsplit(value.strip()), trim_trailing_slash=True)


def _url_with_normalized_host(parsed: SplitResult, *, trim_trailing_slash: bool) -> str:
    scheme = parsed.scheme.lower()
    host = parsed.hostname
    if host is None:
        return parsed.geturl()
    hostname = host.lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None and parsed.port != _DEFAULT_PORTS.get(scheme):
        netloc = f"{netloc}:{parsed.port}"
    path = parsed.path.rstrip("/") if trim_trailing_slash else parsed.path
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _suggest_server_name(url: str) -> str:
    parsed = urlsplit(url)
    labels = [label for label in (parsed.hostname or "mcp").lower().split(".") if label]
    if (
        len(labels) >= _LEADING_HOST_PREFIX_LABEL_MIN_COUNT
        and labels[0] in _HOST_PREFIXES_TO_DROP
    ):
        labels = labels[1:]
    candidate = labels[0] if labels else ""
    if candidate in _GENERIC_ALIAS_SEGMENTS:
        candidate = _path_alias_candidate(parsed.path)
    return normalize_mcp_server_name(candidate) or "mcp"


def _path_alias_candidate(path: str) -> str:
    for segment in path.split("/"):
        normalized = normalize_mcp_server_name(segment.lower())
        if normalized and normalized not in _GENERIC_ALIAS_SEGMENTS:
            return normalized
    return ""


def _dedupe_server_name(base: str, existing_names: set[str]) -> str:
    if base not in existing_names:
        return base
    index = 2
    while f"{base}_{index}" in existing_names:
        index += 1
    return f"{base}_{index}"


__all__ = [
    "MCPServerAddError",
    "MCPServerRemoveError",
    "PersistedMCPServerResult",
    "RemovedMCPServerResult",
    "normalize_mcp_server_url",
    "persist_oauth_mcp_server",
    "persist_remote_mcp_server",
    "persist_stdio_mcp_server",
    "remove_mcp_server",
]
