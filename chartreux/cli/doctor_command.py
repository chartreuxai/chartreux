"""Read-only diagnostics; all runtime activity requires explicit opt-in."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
    from chartreux.core.config.models import MCPHttp, MCPServer, ModelConfig
    from chartreux.core.model_catalog.schema import ProviderDefinition


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    reason: str


def run_doctor_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="chartreux doctor", description="Read-only, non-interactive diagnostics."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="List provider metadata and launch configured MCP servers.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Send billable inference probes to one selected deployment.",
    )
    parser.add_argument("--provider", metavar="ID")
    parser.add_argument("--model", metavar="BASE")
    parser.add_argument("--json", action="store_true", help="Emit JSON only on stdout.")
    args = parser.parse_args(argv)
    if args.smoke and not (args.provider or args.model):
        parser.error("--smoke requires --provider ID and/or --model BASE")
    if not args.smoke and (args.provider or args.model):
        parser.error("--provider and --model require --smoke")
    if args.live or args.smoke:
        from chartreux.core.config.chartreux_schema import load_dotenv_values

        load_dotenv_values()
    try:
        checks = asyncio.run(_diagnose(args))
    except InvalidTarget as exc:
        parser.error(str(exc))
    code = int(any(check.status == "fail" for check in checks))
    if args.json:
        print(
            json.dumps({
                "checks": [asdict(check) for check in checks],
                "exit_code": code,
            })
        )
    else:
        for check in checks:
            print(f"{check.status.upper()}: {check.name}: {check.reason}")
    raise SystemExit(code)


class InvalidTarget(ValueError):
    pass


async def _load_config() -> tuple[ChartreuxConfigSchema, list[Check]]:
    from chartreux.core.config.default_orchestrator import build_default_orchestrator
    from chartreux.core.config.harness_files import HarnessFilesManager
    from chartreux.core.config.layers.project import ProjectConfigLayer
    from chartreux.core.trusted_folders import TrustedFoldersManager

    trust_store = TrustedFoldersManager()
    manager = HarnessFilesManager(
        sources=("user", "project"), cwd=Path.cwd(), trust_store=trust_store
    )
    orchestrator = await build_default_orchestrator(harness_files=manager)
    checks = [
        Check("project config", "skipped", "untrusted project configuration ignored")
        for layer in orchestrator.layers
        if isinstance(layer, ProjectConfigLayer) and layer.is_trusted is False
    ]
    if trust_store.load_error:
        checks.append(Check("trust store", "unverified", trust_store.load_error))
    return orchestrator.config, checks


def _target(config: ChartreuxConfigSchema, args: argparse.Namespace) -> ModelConfig:
    from chartreux.core.model_catalog.resolver import ResolvedModel

    snapshot = config.catalog_snapshot
    candidates = []
    for base, definition in snapshot.catalog.models.items():
        if definition.disabled or (args.model and base != args.model):
            continue
        for deployment in definition.deployments:
            provider = snapshot.catalog.providers[deployment.provider]
            if (
                deployment.disabled
                or provider.disabled
                or (args.provider and deployment.provider != args.provider)
            ):
                continue
            candidates.append(
                ResolvedModel(base, definition, deployment, provider, snapshot.revision)
            )
    if len(candidates) != 1:
        names = (
            ", ".join(
                sorted(f"{c.base_model} ({c.deployment.provider})" for c in candidates)
            )
            or "none"
        )
        raise InvalidTarget(
            f"Select exactly one enabled deployment; candidates: {names}"
        )
    return candidates[0].materialize(
        auto_compact_threshold=config.auto_compact_threshold, validate_thinking=False
    )


async def _diagnose(args: argparse.Namespace) -> list[Check]:
    from chartreux.core.model_catalog.loader import load_catalog

    checks: list[Check] = []
    try:
        load_catalog()
        checks.append(Check("catalog", "pass", "catalog validated"))
    except Exception:
        checks.append(
            Check(
                "catalog",
                "fail",
                "invalid or unreadable models.toml; check catalog syntax and schema",
            )
        )
    try:
        config, source_checks = await _load_config()
        checks.extend(source_checks)
        checks.append(Check("config", "pass", "trusted configuration validated"))
    except Exception:
        checks.append(
            Check(
                "config",
                "fail",
                "invalid or unreadable configuration; check TOML syntax and schema",
            )
        )
        return checks
    selected = _target(config, args) if args.smoke else None
    providers: set[str] = set()
    for name, resolve in (
        ("default model", config.resolve_default_model_alias),
        ("active model", config.get_active_model),
    ):
        try:
            value = resolve()
            if isinstance(value, str):
                from chartreux.core.model_catalog.resolver import resolver_for

                value = (
                    resolver_for(config)
                    .resolve("@orchestrator", allowed_models=config.allowed_models)
                    .materialize(auto_compact_threshold=config.auto_compact_threshold)
                )
            config.get_provider_for_model(value)
            providers.add(value.provider)
            checks.append(
                Check(name, "pass", "model and provider configuration validated")
            )
        except Exception:
            checks.append(
                Check(name, "fail", "model or provider configuration invalid")
            )
    if selected:
        providers.add(selected.provider)
    snapshot = config.catalog_snapshot
    providers.update(
        name
        for name in snapshot.overlaid_providers
        if not snapshot.catalog.providers[name].disabled
    )
    for name in sorted(providers):
        provider = snapshot.catalog.providers[name]
        checks.append(
            _credential(
                name, provider.api_key_env_var, dotenv_loaded=args.live or args.smoke
            )
        )
        if args.live:
            try:
                checks.append(
                    await _listing(name, provider, config.enable_system_trust_store)
                )
            except Exception:
                checks.append(
                    Check(f"provider {name} listing", "fail", "metadata listing failed")
                )
        else:
            checks.append(
                Check(f"provider {name} listing", "skipped", "requires --live")
            )
    for server in config.mcp_servers:
        try:
            checks.append(await _mcp(server, live=args.live))
        except Exception:
            checks.append(Check(f"MCP {server.name}", "fail", "readiness check failed"))
    if selected:
        from chartreux.core.llm.provider_smoke import probe_provider_smoke

        try:
            result = await probe_provider_smoke(
                model=selected,
                provider=config.get_provider_for_model(selected),
                enable_system_trust_store=config.enable_system_trust_store,
            )
            for capability in ("tool", "thinking", "image"):
                verdict = getattr(result, capability)
                checks.append(
                    Check(
                        f"smoke {selected.alias} ({selected.provider}) {capability}",
                        verdict.status,
                        verdict.reason,
                    )
                )
        except Exception:
            checks.append(Check("smoke", "fail", "inference probe failed"))
    return checks


def _credential(name: str, env: str, *, dotenv_loaded: bool = False) -> Check:
    if not env:
        return Check(f"credential {name}", "pass", "no API key configured")
    if os.environ.get(env):
        return Check(f"credential {name}", "pass", "process-env credential present")
    return Check(
        f"credential {name}",
        "unverified",
        "keyring-configured, not inspected; dotenv loaded; process-env credential absent"
        if dotenv_loaded
        else "keyring-configured, not inspected; dotenv-configured, not inspected",
    )


async def _listing(
    name: str, provider: ProviderDefinition, system_trust: bool
) -> Check:
    import httpx

    from chartreux.core.model_catalog.contracts import (
        DiscoveryError,
        ProviderDraft,
        TLSConfig,
    )
    from chartreux.core.model_catalog.discovery import discover_models
    from chartreux.utils.api_keys import resolve_api_key
    from chartreux.utils.http import ChartreuxAsyncHTTPClient

    credential = resolve_api_key(provider.api_key_env_var)
    draft = ProviderDraft(
        None,
        name,
        name,
        provider.api_base,
        provider.api_style,
        provider.api_key_env_var,
        None,
        backend=provider.backend,
        extra_headers=provider.extra_headers,
    )
    statuses: list[int] = []

    async def record(response: httpx.Response) -> None:
        statuses.append(response.status_code)

    async with ChartreuxAsyncHTTPClient(
        timeout=httpx.Timeout(15, connect=5),
        follow_redirects=False,
        enable_system_trust_store=system_trust,
        event_hooks={"response": [record]},
    ) as client:
        result = await discover_models(
            draft, credential, TLSConfig(system_trust), http_client=client
        )
    if not isinstance(result, DiscoveryError):
        return Check(f"provider {name} listing", "pass", "metadata listing succeeded")
    status = statuses[-1] if statuses else None
    if status in {401, 403} or result.code == "auth_rejected":
        return Check(f"provider {name} listing", "fail", "authentication rejected")
    if status == httpx.codes.NOT_FOUND or (
        status is None and result.listing_unsupported
    ):
        return Check(
            f"provider {name} listing", "unverified", "model listing unsupported"
        )
    return Check(f"provider {name} listing", "fail", "metadata listing unsuccessful")


async def _mcp(server: MCPServer, *, live: bool) -> Check:
    if not live:
        return await _mcp_readiness(server, live=False)
    # SDK warnings can include raw initialization responses and credentials.
    # This non-interactive diagnostic runs checks sequentially; restore even on
    # failure/cancellation, including root handlers installed by SDK imports.
    previous = logging.root.manager.disable
    level = logging.root.level
    handlers = logging.root.handlers.copy()
    logging.disable(max(previous, logging.CRITICAL))
    try:
        return await _mcp_readiness(server, live=True)
    finally:
        logging.root.handlers[:] = handlers
        logging.root.setLevel(level)
        logging.disable(previous)


async def _mcp_readiness(server: MCPServer, *, live: bool) -> Check:
    from chartreux.core.config.models import MCPHttp, MCPOAuth
    from chartreux.core.tools.mcp.tools import list_tools_http, list_tools_stdio

    name = f"MCP {server.name}"
    if server.disabled:
        return Check(name, "skipped", "server disabled")
    if not live:
        return Check(
            name,
            "unverified",
            "configuration validated; runtime readiness requires --live; OAuth keyring-configured, not inspected"
            if isinstance(server, MCPHttp) and isinstance(server.auth, MCPOAuth)
            else "configuration validated; runtime readiness requires --live",
        )
    if isinstance(server, MCPHttp):
        if isinstance(server.auth, MCPOAuth):
            return await _oauth_metadata(server)
        headers = dict(server.auth.headers)
        has_explicit_api_key_header = any(
            header.lower() == server.auth.api_key_header.lower() for header in headers
        )
        if server.auth.api_key_env and not has_explicit_api_key_header:
            token = os.getenv(server.auth.api_key_env)
            if not token:
                return Check(name, "fail", "static authentication credential missing")
            headers[server.auth.api_key_header] = server.auth.api_key_format.format(
                token=token
            )
        tools = await list_tools_http(
            server.url, headers=headers, startup_timeout_sec=server.startup_timeout_sec
        )
    else:
        tools = await list_tools_stdio(
            server.argv(),
            env=server.env or None,
            cwd=server.cwd,
            startup_timeout_sec=server.startup_timeout_sec,
        )
    return Check(name, "pass", f"initialized; {len(tools)} tools listed")


async def _oauth_metadata(server: MCPHttp) -> Check:
    from chartreux.core.auth.mcp_oauth import Fingerprint, KeyringTokenStorage

    name = f"MCP {server.name} OAuth"
    fingerprint = await Fingerprint.load(server.name)
    storage = KeyringTokenStorage(server.name)
    tokens = await storage.get_tokens()
    if fingerprint != Fingerprint.compute(server) or tokens is None:
        return Check(
            name, "fail", "stored OAuth credentials absent or stale; no login attempted"
        )
    if (
        storage.token_expiry_time is not None
        and storage.token_expiry_time <= time.time()
    ):
        return Check(
            name, "fail", "stored OAuth credentials expired; no refresh attempted"
        )
    return Check(
        name,
        "unverified",
        "stored fingerprint/expiry inspected; OAuth runtime listing skipped (no refresh/login)",
    )
