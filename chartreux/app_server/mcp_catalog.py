"""Backend-independent MCP catalog persistence and session convergence."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, cast, runtime_checkable

from chartreux.app_server._dispatch import (
    DispatchResult,
    RequestFailure,
    method_not_found,
)
from chartreux.app_server._mcp_auth import MCPAuthenticationService
from chartreux.app_server._model import ProtocolModel, validate_wire
from chartreux.app_server._session_backend_port import (
    MCPAuthorizationProvider,
    MCPAuthorizationRequired,
    ResolvedMCPCatalog,
    ResolvedMCPServerConfig,
    SessionBackend,
    SessionBackendError,
    SessionMCPSourceState,
    SessionMCPState,
    SessionMCPToolDescriptor,
)
from chartreux.app_server.models import (
    MCPSourceStatus,
    MCPSourceSummary,
    MCPState,
    MCPToolSummary,
)
from chartreux.app_server.protocol import (
    MCPAddParams,
    MCPAddResponse,
    MCPAuthRequiredParams,
    MCPAuthUrlParams,
    MCPCatalogMutationResponse,
    MCPLoginParams,
    MCPLogoutParams,
    MCPReadParams,
    MCPReadResponse,
    MCPRefreshParams,
    MCPRemoveParams,
    MCPRemoveResponse,
    MCPToggleParams,
    ProtocolErrorCode,
    RuntimeSnapshot,
    RuntimeUpdatedParams,
)
from chartreux.core.auth.mcp_oauth import MCPOAuthError
from chartreux.core.config import MCPHttp, MCPOAuth, MCPServer, MCPStdio
from chartreux.core.config._mcp_save import MCPApply, MCPPreflight, MCPSaveError
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.mcp_servers import (
    MCPServerAddError,
    MCPServerRemoveError,
    PersistedMCPServerResult,
    RemovedMCPServerResult,
    persist_oauth_mcp_server,
    persist_remote_mcp_server,
    persist_stdio_mcp_server,
    remove_mcp_server,
)
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.types import ConcurrencyConflictError, ConfigSaveResult
from chartreux.core.tools.mcp.authorization import (
    MCPAuthorizationRequired as MCPAuthorizationRequiredImpl,
)
from chartreux.core.tools.mcp_settings import persist_mcp_toggle
from chartreux.utils.mcp import format_tool_display_description

if TYPE_CHECKING:
    from chartreux.app_server._resources import ResourceRequestHandler
    from chartreux.core.agent_loop import AgentLoop
    from chartreux.core.agent_loop._loop import _PreparedReload

type Notify = Callable[[str, ProtocolModel], Awaitable[None]]
type SessionlessCatalogFactory = Callable[
    [], Awaitable[ConfigOrchestrator[ChartreuxConfigSchema]]
]

# Not ``None``, which already means something here: that the caller has no
# owner to name and wants the slot every sessionless one shares.
_OWNED_BY_ORCHESTRATOR = object()


@runtime_checkable
class SessionMCPCatalogBinding(Protocol):
    @property
    def mcp_config_orchestrator(self) -> ConfigOrchestrator[ChartreuxConfigSchema]: ...


@runtime_checkable
class SessionMCPConfigSave(Protocol):
    def reserve_mcp_config(
        self,
    ) -> AbstractContextManager[tuple[MCPPreflight, MCPApply]]: ...


@runtime_checkable
class SessionMCPProjectionSink(Protocol):
    def update_mcp_projection(self, state: MCPState) -> None: ...


class MCPCatalogService:
    """Own app configuration, OAuth coordination, and catalog-to-session calls."""

    def __init__(
        self,
        authentication: MCPAuthenticationService,
        *,
        sessionless_catalog_factory: SessionlessCatalogFactory | None = None,
    ) -> None:
        self._authentication = authentication
        self._sessionless_catalog_factory = sessionless_catalog_factory
        self._auth_required_seen: set[tuple[str, str, str, str | None]] = set()
        self._convergence_errors: dict[tuple[str, str], str] = {}

    @staticmethod
    def handles(method: str) -> bool:
        return method.startswith("mcp_catalog/")

    @property
    def authentication(self) -> MCPAuthenticationService:
        return self._authentication

    async def resolve_catalog(
        self,
        orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        *,
        owner: object | None = _OWNED_BY_ORCHESTRATOR,
    ) -> ResolvedMCPCatalog:
        # The orchestrator stands for the session by default, which is what a
        # caller reading its own config wants. One whose orchestrator it does
        # not keep -- the session's is handed on as a copy -- says so instead
        # and names something it holds: the authentication service files the
        # binding under it weakly and would otherwise drop it mid-session.
        key = orchestrator if owner is _OWNED_BY_ORCHESTRATOR else owner
        return await self._resolved(orchestrator, owner=key)

    async def sessionless_orchestrator(
        self,
    ) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        if self._sessionless_catalog_factory is None:
            raise RequestFailure(
                ProtocolErrorCode.NOT_IMPLEMENTED,
                "Sessionless MCP catalog mutations are not configured",
            )
        return await self._sessionless_catalog_factory()

    def _owner_handle(self, owner_key: str | None) -> object:
        """Resolve one typed wire owner key to the handle it strong-holds."""
        if owner_key is None:
            return _OWNED_BY_ORCHESTRATOR
        return self._authentication.acquire_owner_key(owner_key)

    async def accept_auth_required(
        self,
        params: MCPAuthRequiredParams,
        root: SessionBackend | None,
        *,
        owner_key: str | None = None,
    ) -> RuntimeUpdatedParams | None:
        if (
            root is None
            or root.session_id != params.session_id
            or not isinstance(root, SessionMCPCatalogBinding)
        ):
            return None
        state = await root.read_mcp()
        source = next(
            (candidate for candidate in state.sources if candidate.name == params.name),
            None,
        )
        if (
            source is None
            or source.status != "needs_auth"
            or source.descriptor_revision != params.descriptor_revision
        ):
            return None
        key = (
            params.session_id,
            params.name,
            params.descriptor_revision,
            params.observed_connection_revision,
        )
        if key in self._auth_required_seen:
            return None
        self._auth_required_seen.add(key)
        context = _CatalogContext.for_session(root, owner=self._owner_handle(owner_key))
        self._runtime(
            context, self._project(context, state), preserve_auth_required=key
        )
        return root.runtime_updated_params()

    async def prepare_config_reload(
        self, root: SessionBackend, *, owner_key: str | None = None
    ) -> _CatalogReloadPlan | None:
        if not isinstance(root, SessionMCPCatalogBinding):
            return None
        context = _CatalogContext.for_session(root, owner=self._owner_handle(owner_key))
        previous = await self._resolved(
            context.orchestrator, owner=context.catalog_owner
        )
        await context.orchestrator.reload()
        candidate = await self._resolved(
            context.orchestrator, owner=context.catalog_owner
        )
        restrictive = await self._suspend_restrictive_changes(
            context, previous, candidate
        )
        return _CatalogReloadPlan(previous=previous, restrictive_names=restrictive)

    async def finish_config_reload(
        self,
        root: SessionBackend,
        plan: _CatalogReloadPlan | None,
        *,
        owner_key: str | None = None,
    ) -> RuntimeSnapshot | None:
        if plan is None or not isinstance(root, SessionMCPCatalogBinding):
            return None
        context = _CatalogContext.for_session(root, owner=self._owner_handle(owner_key))
        candidate = await self._resolved(
            context.orchestrator, owner=context.catalog_owner
        )
        restrictive = await self._suspend_restrictive_changes(
            context, plan.previous, candidate
        )
        affected = frozenset(server.name for server in candidate.servers) | frozenset(
            server.name for server in plan.previous.servers
        )
        try:
            state = await context.require_control().reconfigure_mcp(
                candidate, force_remote_discovery=False
            )
        except Exception:
            self._record_convergence_error(context, affected | restrictive)
            raise
        self._clear_convergence_errors(context, affected)
        return self._runtime(context, self._project(context, state))

    async def fail_config_reload(
        self,
        root: SessionBackend,
        plan: _CatalogReloadPlan | None,
        *,
        owner_key: str | None = None,
    ) -> None:
        if plan is None or not plan.restrictive_names:
            return
        if not isinstance(root, SessionMCPCatalogBinding):
            return
        context = _CatalogContext.for_session(root, owner=self._owner_handle(owner_key))
        await self._restore_after_restrictive_failure(
            context, plan.previous, affected_names=plan.restrictive_names
        )

    async def dispatch(
        self,
        method: str,
        raw_params: dict[str, Any],
        *,
        root: SessionBackend | None,
        notify: Notify,
        owner_key: str | None = None,
    ) -> DispatchResult:
        owner = self._owner_handle(owner_key)
        match method:
            case "mcp_catalog/read":
                params = validate_wire(MCPReadParams, raw_params)
                context = await self._target(params.session_id, root, owner=owner)
                session_state = await context.require_control().read_mcp()
                response: ProtocolModel = MCPReadResponse(
                    mcp=self._project(context, session_state)
                )
                runtime_updated = False
            case "mcp_catalog/refresh":
                params = validate_wire(MCPRefreshParams, raw_params)
                context = await self._target(params.session_id, root, owner=owner)
                await context.orchestrator.reload()
                state = await self._converge(
                    context,
                    force_remote_discovery=True,
                    affected_names=frozenset(
                        server.name
                        for server in context.orchestrator.config.mcp_servers
                    ),
                )
                if state is None:
                    raise RuntimeError("A targeted MCP refresh did not converge")
                response = MCPCatalogMutationResponse(
                    runtime=self._runtime(context, self._project(context, state))
                )
                runtime_updated = True
            case "mcp_catalog/toggle" | "mcp_catalog/add" | "mcp_catalog/remove":
                response, runtime_updated = await self._config_mutation(
                    method, raw_params, root, owner=owner
                )
            case "mcp_catalog/login":
                params = validate_wire(MCPLoginParams, raw_params)
                response, runtime_updated = await self._login(
                    params, root, notify, owner=owner
                )
            case "mcp_catalog/logout":
                params = validate_wire(MCPLogoutParams, raw_params)
                response, runtime_updated = await self._logout(
                    params, root, owner=owner
                )
            case _:
                raise method_not_found(method)
        return DispatchResult(response=response, runtime_updated=runtime_updated)

    async def _config_mutation(
        self,
        method: str,
        raw_params: dict[str, Any],
        root: SessionBackend | None,
        *,
        owner: object = _OWNED_BY_ORCHESTRATOR,
    ) -> tuple[ProtocolModel, bool]:
        params = (
            validate_wire(MCPAddParams, raw_params)
            if method == "mcp_catalog/add"
            else validate_wire(MCPRemoveParams, raw_params)
            if method == "mcp_catalog/remove"
            else validate_wire(MCPToggleParams, raw_params)
        )
        context = await self._mutation_target(params.session_id, root, owner=owner)
        if context.control is not None and not isinstance(
            context.control, SessionMCPConfigSave
        ):
            raise RequestFailure(
                ProtocolErrorCode.NOT_IMPLEMENTED,
                "The selected session backend cannot prepare checked MCP configuration saves",
            )
        reservation = (
            context.control.reserve_mcp_config()
            if isinstance(context.control, SessionMCPConfigSave)
            else nullcontext((None, None))
        )
        # Admission covers preparation, checked writer drain, and convergence.
        # There is no eager suspension: the reservation excludes all tree work,
        # and synchronous application retires the old routes after replacement.
        with reservation as hooks:
            context.preflight, context.apply = hooks
            try:
                if isinstance(params, MCPAddParams):
                    return await self._add(params, context)
                if isinstance(params, MCPRemoveParams):
                    return await self._remove(params, context)
                return await self._toggle(params, context)
            except RequestFailure as failure:
                await self._finish_save_failure(context, params.name, failure)
                raise

    async def _finish_save_failure(
        self, context: _CatalogContext, name: str | None, failure: RequestFailure
    ) -> None:
        cause = failure.__cause__
        if not isinstance(cause, MCPSaveError) or cause.save_result is None:
            return
        saved = cause.save_result
        failure.data = {
            "persistence": saved.persistence,
            "application": saved.application,
            "error": saved.error,
        }
        if saved.persistence == "not_saved":
            return
        names = frozenset(
            server.name for server in context.orchestrator.config.mcp_servers
        ) | frozenset({name} if name else ())
        if saved.application != "applied":
            self._record_convergence_error(context, names)
            return
        context.save_result = saved
        try:
            await self._converge(
                context, force_remote_discovery=False, affected_names=names
            )
        except Exception:
            failure.data["convergence"] = "failed"
            self._record_convergence_error(context, names)

    async def _add(
        self, params: MCPAddParams, context: _CatalogContext
    ) -> tuple[MCPAddResponse, bool]:
        try:
            result = await persist_oauth_mcp_server(
                context.orchestrator,
                url=params.url,
                name=params.name,
                scopes=params.scopes,
                transport=params.transport,
                preflight=context.preflight,
                apply=context.apply,
            )
        except ConcurrencyConflictError as exc:
            raise RequestFailure(ProtocolErrorCode.CONFLICT, str(exc)) from exc
        except MCPServerAddError as exc:
            raise RequestFailure(ProtocolErrorCode.INVALID_PARAMS, str(exc)) from exc
        context.save_result = result.save_result
        state = await self._converge(
            context,
            force_remote_discovery=False,
            affected_names=frozenset({result.server.name}),
        )
        runtime = (
            self._runtime(context, self._project(context, state))
            if state is not None
            else None
        )
        return (
            MCPAddResponse(
                name=result.server.name,
                url=result.server.url,
                created=result.created,
                runtime=runtime,
            ),
            state is not None,
        )

    async def _toggle(
        self, params: MCPToggleParams, context: _CatalogContext
    ) -> tuple[MCPCatalogMutationResponse, bool]:
        try:
            context.save_result = await persist_mcp_toggle(
                context.orchestrator,
                name=params.name,
                disabled=params.disabled,
                tool_name=params.tool_name,
                candidate_preflight=context.preflight,
                apply=context.apply,
            )
        except ConcurrencyConflictError as exc:
            raise RequestFailure(ProtocolErrorCode.CONFLICT, str(exc)) from exc
        except ValueError as exc:
            raise RequestFailure(ProtocolErrorCode.INVALID_PARAMS, str(exc)) from exc
        state = await self._converge(
            context,
            force_remote_discovery=False,
            affected_names=frozenset({params.name}),
        )
        runtime = (
            self._runtime(context, self._project(context, state))
            if state is not None
            else None
        )
        return MCPCatalogMutationResponse(runtime=runtime), state is not None

    async def _remove(
        self, params: MCPRemoveParams, context: _CatalogContext
    ) -> tuple[MCPRemoveResponse, bool]:
        configured = _server_named(context.orchestrator, params.name)
        try:
            result = await _remove_server_with_credentials(
                self._authentication,
                context.orchestrator,
                configured=configured,
                name=params.name,
                owner=context.catalog_owner,
                preflight=context.preflight,
                apply=context.apply,
            )
        except ConcurrencyConflictError as exc:
            raise RequestFailure(ProtocolErrorCode.CONFLICT, str(exc)) from exc
        except MCPServerRemoveError as exc:
            raise RequestFailure(ProtocolErrorCode.INVALID_PARAMS, str(exc)) from exc
        except (MCPOAuthError, ValueError) as exc:
            raise RequestFailure(ProtocolErrorCode.INTERNAL_ERROR, str(exc)) from exc
        context.save_result = result.save_result
        state = await self._converge(
            context,
            force_remote_discovery=False,
            affected_names=frozenset({params.name}),
        )
        runtime = (
            self._runtime(context, self._project(context, state))
            if state is not None
            else None
        )
        return (
            MCPRemoveResponse(
                name=result.name, removed=result.removed, runtime=runtime
            ),
            state is not None,
        )

    async def _login(
        self,
        params: MCPLoginParams,
        root: SessionBackend | None,
        notify: Notify,
        *,
        owner: object = _OWNED_BY_ORCHESTRATOR,
    ) -> tuple[MCPCatalogMutationResponse, bool]:
        context = await self._mutation_target(params.session_id, root, owner=owner)
        catalog_owner = context.catalog_owner
        await self._authentication.bind_catalog(
            context.orchestrator.config.mcp_servers, owner=catalog_owner
        )

        async def publish_url(url: str) -> None:
            payload = MCPAuthUrlParams(name=params.name, url=url)
            await notify("mcp_catalog/authUrl", payload)

        try:
            descriptor_revision = await self._authentication.login(
                params.name, on_url=publish_url, owner=catalog_owner
            )
        except (MCPOAuthError, ValueError) as exc:
            raise RequestFailure(ProtocolErrorCode.INVALID_PARAMS, str(exc)) from exc
        state = None
        if context.control is not None:
            try:
                state = await context.control.authorization_changed(
                    name=params.name, descriptor_revision=descriptor_revision
                )
            except Exception:
                self._record_convergence_error(context, frozenset({params.name}))
                raise
            self._clear_convergence_errors(context, frozenset({params.name}))
        runtime = (
            self._runtime(context, self._project(context, state))
            if state is not None
            else None
        )
        return MCPCatalogMutationResponse(runtime=runtime), state is not None

    async def _logout(
        self,
        params: MCPLogoutParams,
        root: SessionBackend | None,
        *,
        owner: object = _OWNED_BY_ORCHESTRATOR,
    ) -> tuple[MCPCatalogMutationResponse, bool]:
        context = await self._mutation_target(params.session_id, root, owner=owner)
        catalog_owner = context.catalog_owner
        await self._authentication.bind_catalog(
            context.orchestrator.config.mcp_servers, owner=catalog_owner
        )
        original = (
            await self._resolved(context.orchestrator, owner=context.catalog_owner)
            if context.control is not None
            else None
        )
        if context.control is not None:
            await context.control.suspend_mcp(
                name=params.name, tool_name=None, reason="logout"
            )
        try:
            descriptor_revision = await self._authentication.logout(
                params.name, owner=catalog_owner
            )
        except (MCPOAuthError, ValueError) as exc:
            await self._restore_after_restrictive_failure(
                context, original, affected_names=frozenset({params.name})
            )
            raise RequestFailure(ProtocolErrorCode.INVALID_PARAMS, str(exc)) from exc
        state = None
        if context.control is not None:
            try:
                state = await context.control.authorization_changed(
                    name=params.name, descriptor_revision=descriptor_revision
                )
            except Exception:
                self._record_convergence_error(context, frozenset({params.name}))
                raise
            self._clear_convergence_errors(context, frozenset({params.name}))
        runtime = (
            self._runtime(context, self._project(context, state))
            if state is not None
            else None
        )
        return MCPCatalogMutationResponse(runtime=runtime), state is not None

    async def _converge(
        self,
        context: _CatalogContext,
        *,
        force_remote_discovery: bool,
        affected_names: frozenset[str],
    ) -> SessionMCPState | None:
        try:
            configuration = await self._resolved(
                context.orchestrator, owner=context.catalog_owner
            )
            if context.control is None:
                return None
            state = await context.control.reconfigure_mcp(
                configuration, force_remote_discovery=force_remote_discovery
            )
        except (Exception, asyncio.CancelledError):
            self._record_convergence_error(context, affected_names)
            if context.save_result is not None:
                raise RequestFailure(
                    ProtocolErrorCode.INTERNAL_ERROR,
                    f"MCP save: persistence={context.save_result.persistence}; "
                    "application=failed; error=convergence. Reload before retrying.",
                    data={
                        "persistence": context.save_result.persistence,
                        "application": "failed",
                        "error": "convergence",
                    },
                ) from None
            raise
        self._clear_convergence_errors(context, affected_names)
        return state

    async def _restore_after_restrictive_failure(
        self,
        context: _CatalogContext,
        configuration: ResolvedMCPCatalog | None,
        *,
        affected_names: frozenset[str],
    ) -> None:
        if context.control is None or configuration is None:
            return
        try:
            await context.control.reconfigure_mcp(
                configuration, force_remote_discovery=False
            )
        except Exception:
            self._record_convergence_error(context, affected_names)
            return
        self._clear_convergence_errors(context, affected_names)

    async def _suspend_restrictive_changes(
        self,
        context: _CatalogContext,
        previous: ResolvedMCPCatalog,
        candidate: ResolvedMCPCatalog,
    ) -> frozenset[str]:
        if context.control is None:
            return frozenset()
        next_by_name = {server.name: server for server in candidate.servers}
        suspended: set[str] = set()
        for old in previous.servers:
            new = next_by_name.get(old.name)
            if (
                new is None
                or new.disabled
                or old.authorization.server_fingerprint
                != new.authorization.server_fingerprint
            ):
                await context.control.suspend_mcp(
                    name=old.name,
                    tool_name=None,
                    reason="replace" if new is not None else "remove",
                )
                suspended.add(old.name)
                continue
            for tool_name in new.disabled_tools - old.disabled_tools:
                await context.control.suspend_mcp(
                    name=old.name, tool_name=tool_name, reason="disable"
                )
                suspended.add(old.name)
        return frozenset(suspended)

    def _record_convergence_error(
        self, context: _CatalogContext, affected_names: frozenset[str]
    ) -> None:
        if context.root is None:
            return
        for name in affected_names:
            self._convergence_errors[(context.root.session_id, name)] = (
                "MCP source configuration did not converge in this session"
            )

    def _clear_convergence_errors(
        self, context: _CatalogContext, affected_names: frozenset[str]
    ) -> None:
        if context.root is None:
            return
        for name in affected_names:
            self._convergence_errors.pop((context.root.session_id, name), None)

    async def _resolved(
        self,
        orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        *,
        owner: object | None,
    ) -> ResolvedMCPCatalog:
        servers = orchestrator.config.mcp_servers
        await self._authentication.bind_catalog(servers, owner=owner)
        resolved = tuple(
            self._resolve_server(server, owner=owner) for server in servers
        )
        payload = [
            {
                "name": server.name,
                "transport": server.transport,
                "url": server.url,
                "command": server.command,
                "args": server.args,
                "cwd": str(server.cwd) if server.cwd is not None else None,
                "env": dict(server.env),
                "authorization": {
                    "server_name": server.authorization.server_name,
                    "server_fingerprint": server.authorization.server_fingerprint,
                    "kind": server.authorization.kind,
                },
                "prompt": server.prompt,
                "startup_timeout_s": server.startup_timeout_s,
                "tool_timeout_s": server.tool_timeout_s,
                "disabled": server.disabled,
                "disabled_tools": sorted(server.disabled_tools),
            }
            for server in resolved
        ]
        revision = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return ResolvedMCPCatalog(revision=revision, servers=resolved)

    def _resolve_server(
        self, server: MCPServer, *, owner: object | None
    ) -> ResolvedMCPServerConfig:
        reference = self._authentication.reference_for(server, owner=owner)
        if isinstance(server, MCPStdio):
            argv = server.argv()
            return ResolvedMCPServerConfig(
                name=server.name,
                transport="stdio",
                url=None,
                command=argv[0] if argv else None,
                args=tuple(argv[1:]),
                cwd=Path(server.cwd).expanduser().resolve() if server.cwd else None,
                env=server.env,
                authorization=reference,
                prompt=server.prompt,
                startup_timeout_s=server.startup_timeout_sec,
                tool_timeout_s=server.tool_timeout_sec,
                disabled=server.disabled,
                disabled_tools=frozenset(server.disabled_tools),
            )
        return ResolvedMCPServerConfig(
            name=server.name,
            transport=server.transport,
            url=server.url,
            command=None,
            args=(),
            cwd=None,
            env={},
            authorization=reference,
            prompt=server.prompt,
            startup_timeout_s=server.startup_timeout_sec,
            tool_timeout_s=server.tool_timeout_sec,
            disabled=server.disabled,
            disabled_tools=frozenset(server.disabled_tools),
        )

    async def _target(
        self,
        session_id: str,
        root: SessionBackend | None,
        *,
        owner: object = _OWNED_BY_ORCHESTRATOR,
    ) -> _CatalogContext:
        if root is None or root.session_id != session_id:
            raise RequestFailure(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {session_id}"
            )
        return _CatalogContext.for_session(root, owner=owner)

    async def _mutation_target(
        self,
        session_id: str | None,
        root: SessionBackend | None,
        *,
        owner: object = _OWNED_BY_ORCHESTRATOR,
    ) -> _CatalogContext:
        if session_id is not None:
            return await self._target(session_id, root, owner=owner)
        if root is not None:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT,
                "A sessionless MCP catalog mutation cannot target an active session implicitly",
            )
        orchestrator = await self.sessionless_orchestrator()
        return _CatalogContext(orchestrator, None, None)

    def _project(self, context: _CatalogContext, state: SessionMCPState) -> MCPState:
        orchestrator = context.orchestrator
        session_id = context.root.session_id if context.root is not None else None
        convergence_errors = {
            name: error
            for (failed_session_id, name), error in self._convergence_errors.items()
            if failed_session_id == session_id
        }
        sources, errors = project_mcp_sources(
            orchestrator, state, convergence_errors=convergence_errors
        )
        return MCPState(sources=[*sources], discovery_errors=errors)

    def _runtime(
        self,
        context: _CatalogContext,
        state: MCPState,
        *,
        preserve_auth_required: tuple[str, str, str, str | None] | None = None,
    ) -> RuntimeSnapshot:
        root = context.root
        if root is None:
            raise SessionBackendError(
                ProtocolErrorCode.NOT_IMPLEMENTED,
                "The selected session backend cannot project runtime state",
            )
        if isinstance(root, SessionMCPProjectionSink):
            root.update_mcp_projection(state)
        self._clear_resolved_auth_required(
            root.session_id, preserve=preserve_auth_required
        )
        return root.runtime_updated_params().runtime

    def _clear_resolved_auth_required(
        self,
        session_id: str,
        *,
        preserve: tuple[str, str, str, str | None] | None = None,
    ) -> None:
        self._auth_required_seen = {
            key
            for key in self._auth_required_seen
            if key[0] != session_id or key == preserve
        }


class SessionlessMCPCatalog:
    """Narrow command facade that never constructs a Harness session."""

    def __init__(self, service: MCPCatalogService, notify: Notify) -> None:
        self._service = service
        self._notify = notify

    async def dispatch(self, method: str, params: ProtocolModel) -> ProtocolModel:
        result = await self._service.dispatch(
            method,
            params.model_dump(mode="json", by_alias=True),
            root=None,
            notify=self._notify,
        )
        return result.response

    async def add_server(
        self,
        server: MCPHttp | MCPStdio,
        *,
        login: bool,
        on_oauth_url: Callable[[str], Awaitable[None]],
        on_persisted: Callable[[PersistedMCPServerResult[Any]], None] | None = None,
    ) -> PersistedMCPServerResult[Any]:
        orchestrator = await self._service.sessionless_orchestrator()
        result = (
            await persist_stdio_mcp_server(orchestrator, server)
            if isinstance(server, MCPStdio)
            else await persist_remote_mcp_server(orchestrator, server)
        )
        await self._service.authentication.bind_catalog(orchestrator.config.mcp_servers)
        if on_persisted is not None:
            on_persisted(result)
        if login and _is_oauth(result.server):
            await self._service.authentication.login(
                result.server.name, on_url=on_oauth_url
            )
        return result

    async def remove_server(self, name: str) -> RemovedMCPServerResult:
        orchestrator = await self._service.sessionless_orchestrator()
        server = _server_named(orchestrator, name)
        await self._service.authentication.bind_catalog(orchestrator.config.mcp_servers)
        try:
            return await _remove_server_with_credentials(
                self._service.authentication,
                orchestrator,
                configured=server,
                name=name,
                owner=None,
            )
        except (MCPOAuthError, ValueError) as exc:
            raise MCPServerRemoveError(
                f"Failed to remove OAuth credentials for `{name}`: {exc}"
            ) from exc


def create_sessionless_mcp_catalog(
    factory: SessionlessCatalogFactory, *, notify: Notify | None = None
) -> SessionlessMCPCatalog:
    async def ignore_notification(_method: str, _params: ProtocolModel) -> None:
        return None

    service = MCPCatalogService(
        MCPAuthenticationService(), sessionless_catalog_factory=factory
    )
    return SessionlessMCPCatalog(service, notify or ignore_notification)


class _CatalogContext:
    def __init__(
        self,
        orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        control: SessionBackend | None,
        root: SessionBackend | None,
        *,
        owner: object = _OWNED_BY_ORCHESTRATOR,
    ) -> None:
        self.orchestrator = orchestrator
        self.control = control
        self.root = root
        self._owner = owner
        self.preflight: MCPPreflight | None = None
        self.apply: MCPApply | None = None
        self.save_result: ConfigSaveResult | None = None

    @classmethod
    def for_session(
        cls, root: SessionBackend, *, owner: object = _OWNED_BY_ORCHESTRATOR
    ) -> _CatalogContext:
        if not isinstance(root, SessionMCPCatalogBinding):
            raise RequestFailure(
                ProtocolErrorCode.NOT_IMPLEMENTED,
                "The selected session backend does not support MCP catalog control",
            )
        return cls(root.mcp_config_orchestrator, root, root, owner=owner)

    def require_control(self) -> SessionBackend:
        if self.control is None:
            raise RuntimeError("A targeted MCP catalog operation has no control port")
        return self.control

    @property
    def catalog_owner(self) -> object | None:
        # What the authentication service files this session's configured
        # servers under. None for a sessionless mutation: its orchestrator is
        # built fresh per call, so keying on it would file the binding under
        # something nothing holds, and it has no session to be confused with.
        # A wire owner names its typed key's handle instead of the orchestrator
        # object: the handle is held strongly for exactly as long as the key.
        if self._owner is not _OWNED_BY_ORCHESTRATOR:
            return self._owner
        return self.orchestrator if self.root is not None else None


@dataclass(frozen=True, slots=True)
class _CatalogReloadPlan:
    previous: ResolvedMCPCatalog
    restrictive_names: frozenset[str]


def _project_source(
    server: MCPServer, state: SessionMCPSourceState | None, *, convergence_failed: bool
) -> MCPSourceSummary:
    if convergence_failed:
        status = MCPSourceStatus.UNAVAILABLE
        tools = [] if state is None else _project_tools(state)
    elif state is None:
        status = (
            MCPSourceStatus.DISABLED if server.disabled else MCPSourceStatus.UNAVAILABLE
        )
        tools: list[MCPToolSummary] = []
    else:
        status = MCPSourceStatus(state.status)
        tools = _project_tools(state)
    return MCPSourceSummary(
        name=server.name, transport=server.transport, status=status, tools=tools
    )


def project_mcp_sources(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    state: SessionMCPState,
    *,
    convergence_errors: Mapping[str, str] | None = None,
) -> tuple[list[MCPSourceSummary], dict[str, str]]:
    convergence_errors = convergence_errors or {}
    states = {source.name: source for source in state.sources}
    sources = [
        _project_source(
            server,
            states.get(server.name),
            convergence_failed=server.name in convergence_errors,
        )
        for server in orchestrator.config.mcp_servers
    ]
    errors = dict(state.discovery_errors)
    for server in orchestrator.config.mcp_servers:
        if not server.disabled and server.name not in states:
            errors.setdefault(
                server.name, "MCP source configuration is not active in this session"
            )
        if error := convergence_errors.get(server.name):
            errors[server.name] = error
    return sources, errors


def _project_tools(state: SessionMCPSourceState) -> list[MCPToolSummary]:
    return [
        MCPToolSummary(
            name=tool.remote_name,
            description=format_tool_display_description(
                tool.description, source_name=state.name
            ),
            enabled=tool.enabled,
        )
        for tool in state.tools
    ]


def _server_named(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], name: str
) -> MCPServer | None:
    return next(
        (server for server in orchestrator.config.mcp_servers if server.name == name),
        None,
    )


def _is_oauth(server: object) -> TypeGuard[MCPHttp]:
    return isinstance(server, MCPHttp) and isinstance(server.auth, MCPOAuth)


async def _remove_server_with_credentials(
    authentication: MCPAuthenticationService,
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    *,
    configured: MCPServer | None,
    name: str,
    owner: object | None,
    preflight: MCPPreflight | None = None,
    apply: MCPApply | None = None,
) -> RemovedMCPServerResult:
    if not _is_oauth(configured):
        return await remove_mcp_server(
            orchestrator, name, preflight=preflight, apply=apply
        )
    saved_error: MCPSaveError | None = None
    async with authentication.credential_removal(configured.name, owner=owner):
        try:
            return await remove_mcp_server(
                orchestrator, name, preflight=preflight, apply=apply
            )
        except MCPSaveError as exc:
            if exc.save_result is None or exc.save_result.persistence == "not_saved":
                raise
            # Replacement happened. Do not restore credentials on top of the
            # removed disk entry just because durability/application is uncertain.
            saved_error = exc
    assert saved_error is not None
    raise saved_error


class AgentLoopMCPCatalogRoot:
    """The agent process's own loop as the MCP catalog's control port.

    On the stdio topology the coordinator proxies ``mcp_catalog/*`` control
    over the agent wire and every leg runs here against the loop this process
    owns, so no coordinator-side orchestrator or catalog owner exists to
    drift. The typed owner key the wire carries files the auth bindings under
    a handle held for exactly as long as the key lives, never dropped by GC
    timing mid-session.
    """

    def __init__(
        self,
        loop: AgentLoop,
        *,
        resources: ResourceRequestHandler,
        authorization_provider: MCPAuthorizationProvider,
        is_busy: Callable[[], bool],
        authorization_required_sink: (
            Callable[
                [str, MCPAuthorizationRequired | MCPAuthorizationRequiredImpl],
                Awaitable[None],
            ]
            | None
        ) = None,
        descriptor_cache_root: Path | None = None,
    ) -> None:
        self._loop = loop
        self._resources = resources
        self._authorization_provider = authorization_provider
        self._is_busy = is_busy
        self._authorization_required_sink = authorization_required_sink
        self._descriptor_cache_root = descriptor_cache_root
        self.mcp_catalog: ResolvedMCPCatalog | None = None
        self._mcp_catalog_revision = ""
        self._mcp_route_revision = 0
        self._mcp_save_task: asyncio.Task[Any] | None = None
        self._projection: MCPState | None = None

    @property
    def session_id(self) -> str:
        return self._loop.session_id

    @property
    def mcp_config_orchestrator(self) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        return self._loop.config_orchestrator

    def _configure_mcp_registry(self) -> None:
        from chartreux.app_server._session_backend_impl import (
            configure_mcp_registry_impl,
        )
        from chartreux.core.tools.mcp.registry import MCPRegistry

        catalog = self.mcp_catalog
        if catalog is None:
            return
        registry = self._loop.mcp_registry
        if registry is None and self._loop.config.mcp_servers:
            registry = MCPRegistry()
            self._loop.mcp_registry = registry
            self._loop.tool_manager.set_mcp_registry(registry)
        if registry is None:
            return
        configure_mcp_registry_impl(
            registry,
            catalog,
            self._authorization_provider,
            required_sink=self._authorization_required_sink,
            descriptor_cache_root=self._descriptor_cache_root,
        )

    @contextmanager
    def reserve_mcp_config(self) -> Iterator[tuple[MCPPreflight, MCPApply]]:
        """Stage one checked MCP save against the loop this process owns.

        Preparation may discover configured endpoints, so it runs against a
        fresh registry that never mutates the accepted one; application swaps
        that registry in, exactly as the in-process save does.
        """
        from chartreux.core.tools.mcp.registry import MCPRegistry

        loop = self._loop
        prepared: _PreparedReload | None = None
        registry = MCPRegistry()

        async def prepare(candidate: Any) -> None:
            nonlocal prepared
            prepared = loop._prepare_reload(
                candidate.config, True, mcp_registry=registry
            )

        def apply(candidate: Any) -> None:
            assert prepared is not None
            loop._commit_reload(prepared, True)
            loop.mcp_registry = registry

        self._mcp_save_task = asyncio.current_task()
        try:
            yield prepare, apply
        finally:
            self._mcp_save_task = None
            if prepared is not None and prepared.backend is not loop.backend:
                loop._backend_lifetime.retire(
                    prepared.backend, whole_turn_active=loop._active_turn is not None
                )

    async def read_mcp(self) -> SessionMCPState:
        return self._session_mcp_state()

    async def reconfigure_mcp(
        self, configuration: ResolvedMCPCatalog, *, force_remote_discovery: bool
    ) -> SessionMCPState:
        self._require_mcp_idle()
        current_names = tuple(server.name for server in self._loop.config.mcp_servers)
        if current_names != tuple(server.name for server in configuration.servers):
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT,
                "The resolved MCP catalog does not match the session configuration",
            )
        self.mcp_catalog = configuration
        self._configure_mcp_registry()
        self._mcp_catalog_revision = configuration.revision
        manager = self._loop.tool_manager

        async def converge() -> None:
            if force_remote_discovery:
                await manager.refresh_remote_tools_async()
            else:
                await manager.reconfigure_mcp_async()
            await self._loop.refresh_system_prompt()

        if self._mcp_save_task is asyncio.current_task():
            # A replaced source must not outlive its admission reservation
            # while endpoint/cache work is still running. Join, never retry.
            pending = asyncio.create_task(converge())
            while True:
                try:
                    await asyncio.shield(pending)
                    break
                except asyncio.CancelledError:
                    if pending.cancelled():
                        raise
        else:
            await converge()
        self._mcp_route_revision += 1
        return self._session_mcp_state()

    async def authorization_changed(
        self, *, name: str, descriptor_revision: str
    ) -> SessionMCPState:
        self._require_mcp_idle()
        registry = self._loop.mcp_registry
        if registry is None:
            raise SessionBackendError(
                ProtocolErrorCode.NOT_FOUND, "No MCP servers are configured"
            )
        catalog = self.mcp_catalog
        resolved = (
            next((server for server in catalog.servers if server.name == name), None)
            if catalog is not None
            else None
        )
        if catalog is None or resolved is None:
            raise SessionBackendError(
                ProtocolErrorCode.NOT_FOUND, f"MCP server not found: {name}"
            )
        resolved = replace(
            resolved,
            authorization=replace(
                resolved.authorization, descriptor_revision=descriptor_revision
            ),
        )
        self.mcp_catalog = replace(
            catalog,
            servers=tuple(
                resolved if server.name == name else server
                for server in catalog.servers
            ),
        )
        self._configure_mcp_registry()
        manager = self._loop.tool_manager
        if resolved.disabled:
            registry.record_disabled_authorization(name, descriptor_revision)
            manager.suspend_mcp(name)
            await self._loop.refresh_system_prompt()
            self._mcp_route_revision += 1
            return self._session_mcp_state()
        authorization = await self._authorization_provider.resolve(
            resolved.authorization
        )
        if authorization.descriptor_revision != descriptor_revision:
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT,
                "The MCP authorization descriptor revision is stale",
            )
        if isinstance(authorization, MCPAuthorizationRequired):
            registry.mark_needs_auth(name, descriptor_revision)
            manager.suspend_mcp(name)
            if self._authorization_required_sink is not None:
                await self._authorization_required_sink(name, authorization)
            await self._loop.refresh_system_prompt()
            self._mcp_route_revision += 1
            return self._session_mcp_state()
        registry.invalidate(name)
        await manager.reconfigure_mcp_async()
        await self._loop.refresh_system_prompt()
        self._mcp_route_revision += 1
        return self._session_mcp_state()

    async def suspend_mcp(
        self, *, name: str, tool_name: str | None, reason: str
    ) -> SessionMCPState:
        if reason not in {"logout", "remove", "disable", "replace"}:
            raise SessionBackendError(
                ProtocolErrorCode.INVALID_PARAMS, "Invalid MCP suspension reason"
            )
        self._require_mcp_idle()
        self._loop.tool_manager.suspend_mcp(name, tool_name)
        await self._loop.refresh_system_prompt()
        self._mcp_route_revision += 1
        return self._session_mcp_state()

    def _require_mcp_idle(self) -> None:
        # Convergence against an active turn (mid-tool-call) is fenced: the
        # owner knows its own live turn, and reconfiguring under one would race
        # the run in flight, so the carrier answers the partial receipt.
        if (
            self._mcp_save_task is not None
            and self._mcp_save_task is asyncio.current_task()
        ):
            return
        if self._is_busy():
            raise SessionBackendError(
                ProtocolErrorCode.CONFLICT,
                "Session is busy running a turn; MCP convergence is fenced "
                "until it settles",
            )

    def update_mcp_projection(self, state: MCPState) -> None:
        self._projection = state

    def runtime_updated_params(self) -> RuntimeUpdatedParams:
        runtime = self._resources.runtime_snapshot()
        if self._projection is not None:
            runtime = runtime.model_copy(update={"mcp": self._projection})
        return RuntimeUpdatedParams(session_id=self.session_id, runtime=runtime)

    def _session_mcp_state(self) -> SessionMCPState:
        public = self._resources.runtime_snapshot().mcp
        registry = self._loop.mcp_registry
        transports = {
            server.name: server.transport for server in self._loop.config.mcp_servers
        }
        sources = tuple(
            SessionMCPSourceState(
                name=source.name,
                transport=cast(Any, transports[source.name]),
                status=cast(Any, source.status.value),
                tools=tuple(
                    SessionMCPToolDescriptor(
                        remote_name=tool.name,
                        description=tool.description,
                        enabled=tool.enabled,
                        display_name=f"{source.name}_{tool.name}",
                    )
                    for tool in source.tools
                ),
                descriptor_revision=(
                    registry.descriptor_revision(source.name)
                    if registry is not None
                    else ""
                ),
                error=public.discovery_errors.get(source.name),
            )
            for source in public.sources
            if source.name in transports
        )
        return SessionMCPState(
            catalog_revision=self._mcp_catalog_revision,
            # Preserve this persisted route-revision prefix for compatibility.
            route_revision=f"legacy-mcp-routes:{self._mcp_route_revision}",
            sources=sources,
            discovery_errors=public.discovery_errors,
        )


__all__ = [
    "AgentLoopMCPCatalogRoot",
    "MCPCatalogService",
    "SessionMCPCatalogBinding",
    "SessionMCPConfigSave",
    "SessionMCPProjectionSink",
    "SessionlessMCPCatalog",
    "create_sessionless_mcp_catalog",
    "project_mcp_sources",
]
