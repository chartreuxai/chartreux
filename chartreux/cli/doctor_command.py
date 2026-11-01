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
    from chartreux.core.config.harness_files import HarnessFilesManager
    from chartreux.core.config.models import MCPHttp, MCPServer, ModelConfig
    from chartreux.core.dispatch.route_inspection import RouteReport
    from chartreux.core.model_catalog.schema import ProviderDefinition
    from chartreux.core.skills.manager import ShadowEntry, SkillLoadFailure


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


async def _load_config() -> tuple[
    ChartreuxConfigSchema, HarnessFilesManager, list[Check]
]:
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
    checks.append(_agent_selections_check(orchestrator.config, manager))
    return orchestrator.config, manager, checks


def _agent_selections_check(
    config: ChartreuxConfigSchema, manager: HarnessFilesManager
) -> Check:
    """Resolve the [agent] selections offline through the strict resolver.

    Config validation alone accepts unknown bare/user selections; they fail
    later, at conversation binding. Doctor reports them here with the actual
    resolver error — no live probes, no conversation is started.
    """
    from chartreux.core.policy.composition import (
        policy_resource_loader,
        resolve_policy_resources,
        user_resource_directories,
    )

    try:
        resolve_policy_resources(
            config, policy_resource_loader(user_resource_directories(manager))
        )
    except ValueError as exc:
        return Check(
            "agent selections",
            "fail",
            f"{exc} Fix [agent] identity/workflow/orchestration_strategy in "
            "config.toml (custom identities are prompt files in the user "
            "prompts directory).",
        )
    return Check(
        "agent selections", "pass", "identity/workflow/strategy selections resolve"
    )


def _retired_inputs_checks(manager: HarnessFilesManager) -> list[Check]:
    """Report retired on-disk inputs loudly; user files are never touched.

    Retired config.toml fields (``system_prompt_id``, ``agent_paths``,
    ``enabled_agents``, ``disabled_agents``) fail config load itself with
    their successor guidance — doctor surfaces that validation message via
    :func:`_config_failure_check`. What can still load is the retired file
    surface: agents/*.toml profile directories and project-level prompt
    directories. They are detected here, never consumed, and never deleted
    or converted (design §12).
    """
    checks: list[Check] = []
    retired_agents = manager.retired_agent_dirs
    if retired_agents:
        checks.append(
            Check(
                "retired agent profiles",
                "fail",
                "agents/*.toml profile files are retired and no longer load: "
                f"found {', '.join(str(path) for path in retired_agents)}. Use "
                "identities instead — select [agent] identity in config.toml or "
                "pass the task tool's identity field; custom identities are "
                "prompt files in the user prompts directory. The files are left "
                "untouched.",
            )
        )
    retired_prompts = manager.retired_project_prompt_dirs
    if retired_prompts:
        checks.append(
            Check(
                "retired project prompts",
                "fail",
                "project-level custom prompt directories are retired and no "
                f"longer load: found {', '.join(str(path) for path in retired_prompts)}. "
                "Move custom identity prompts to the user prompts directory; "
                "workflows and orchestration strategies are user-level too. "
                "The files are left untouched.",
            )
        )
    return checks


def _skills_checks(
    config: ChartreuxConfigSchema, manager: HarnessFilesManager
) -> list[Check]:
    """Advisory skill checks from the ONE shared skill inventory (K1/K2).

    Read-only: the inventory reports the loader's own selection and the
    typed load-failure records its discovery captured. ``warn`` never
    changes doctor's exit code; only ``fail`` does.
    """
    from chartreux.core.skills import manager as skills_manager

    inventory = skills_manager.build_shadow_inventory(
        lambda: config, harness_files=manager
    )
    checks = [
        Check(
            f"skill {failure.path.parent.name}", "warn", _load_failure_reason(failure)
        )
        for failure in inventory.load_failures
    ]
    checks.extend(
        Check(f"user skill {entry.name}", "warn", _shadow_reason(entry))
        for entry in inventory.entries
    )
    if not checks:
        return [Check("user skills", "pass", "no user skill shadows a skill copy")]
    return checks


def _routes_checks(config: ChartreuxConfigSchema) -> list[Check]:
    """Report the purpose-route table this configuration independently loads.

    Same advisory mechanism as the skill checks: one ``Check`` per purpose
    route plus one heading, all in doctor's single check list. ``warn`` never
    changes doctor's exit code; only ``fail`` does. The section renders the
    table loaded for THIS run — never a running conversation's live table
    (the ``/routes`` browser owns that one).
    """
    from chartreux.core.dispatch.route_inspection import (
        DOCTOR_SECTION_LABEL,
        inspect_config_routes,
    )

    report = inspect_config_routes(config)
    if report.error is not None:
        return [
            Check(
                "routes",
                "unverified",
                f"{DOCTOR_SECTION_LABEL}: [orchestration.routes] read from the "
                "trusted configuration on this machine, not from a running "
                f"conversation; the table could not be inspected: {report.error}",
            )
        ]
    statuses = [_route_check_status(route) for route in report.routes]
    counts = {status: statuses.count(status) for status in ("pass", "warn", "fail")}
    return [
        Check(
            "routes",
            _worst_status(statuses),
            f"{DOCTOR_SECTION_LABEL}: [orchestration.routes] read from the trusted "
            "configuration on this machine, not from a running conversation. "
            f"{len(report.routes)} purpose route(s): {counts['pass']} pass, "
            f"{counts['warn']} warn, {counts['fail']} fail. Model bindings are "
            "previews, not launch guarantees.",
        ),
        *(
            Check(
                f"route {route.purpose}",
                _route_check_status(route),
                _route_reason(route),
            )
            for route in report.routes
        ),
    ]


def _route_check_status(route: RouteReport) -> str:
    """``fail`` for a rejected reload or an unresolvable binding; ``warn`` for
    a launch-time diversity fact (the shipped all-marker case included).
    """
    if route.reload_status == "fail" or any(
        target.error is not None for target in route.targets
    ):
        return "fail"
    if route.launch_status != "pass":
        return "warn"
    return "pass"


def _worst_status(statuses: list[str]) -> str:
    for status in ("fail", "warn", "pass"):
        if status in statuses:
            return status
    return "pass"


def _route_reason(route: RouteReport) -> str:
    return "; ".join([route.summary(), *(target.summary() for target in route.targets)])


def _load_failure_reason(failure: SkillLoadFailure) -> str:
    return (
        f"cannot load {failure.path} ({failure.origin}): {failure.diagnostic}; "
        "fix the file and rerun chartreux doctor"
    )


def _shadow_reason(entry: ShadowEntry) -> str:
    from chartreux.core.skills.manager import ShadowState

    match entry.state:
        case ShadowState.IDENTICAL:
            return (
                f"identical to the shipped copy at {entry.shipped_path}; "
                "suggest removal after reviewing support files — never automatic"
            )
        case ShadowState.DIVERGED:
            return (
                f"differs from the shipped copy at {entry.shipped_path}; "
                f"review it with `chartreux skills diff {entry.name}`"
            )
        case ShadowState.INACTIVE:
            winner = (
                f"{entry.winner_path} ({entry.winner_scope})"
                if entry.winner_path is not None
                else f"the {entry.winner_scope} skill"
            )
            return (
                f"inactive: {winner} loads instead; editing this copy at "
                f"{entry.skill_path} has no effect"
            )
        case _:
            return f"unverified: {entry.reason}"


def _config_failure_check(exc: BaseException) -> Check:
    """Surface the loader's own diagnostic instead of a generic one.

    Retired settings fail config load with their successor guidance in the
    validation message; doctor shows that message (source-aware, values
    never echoed) so the user learns the successor from the failure itself.
    """
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_url=False, include_input=False)
        )
        if details:
            return Check("config", "fail", details)
    return Check(
        "config",
        "fail",
        "invalid or unreadable configuration; check TOML syntax and schema",
    )


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
        config, manager, source_checks = await _load_config()
        checks.extend([
            *source_checks,
            Check("config", "pass", "trusted configuration validated"),
            *_retired_inputs_checks(manager),
            *_skills_checks(config, manager),
            *_routes_checks(config),
        ])
    except Exception as exc:
        checks.append(_config_failure_check(exc))
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
                    .resolve("@main", allowed_models=config.allowed_models)
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
