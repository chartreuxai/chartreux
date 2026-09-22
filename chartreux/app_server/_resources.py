from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from typing import Any

from chartreux.app_server._config_introspect import (
    HIDDEN_SETTINGS,
    POPULAR_SETTINGS,
    build_field_wires,
    collect_layer_values,
)
from chartreux.app_server._config_write import (
    config_field_write_targets,
    config_write_ops_to_patches,
    config_write_projection,
    config_write_targets,
)
from chartreux.app_server._dispatch import (
    DispatchResult,
    RequestFailure,
    method_not_found,
)
from chartreux.app_server._execution import SessionExecution, SessionExecutionKind
from chartreux.app_server._model import ProtocolModel, validate_wire
from chartreux.app_server._projection import (
    project_config,
    project_debug_logs,
    project_diagnostics,
    project_installed_skills,
    project_mcp,
    project_session_log,
    project_skills,
    project_stats,
    project_tools,
)
from chartreux.app_server.config import ProxySettingsView
from chartreux.app_server.models import MCPState, ScheduledLoop
from chartreux.app_server.protocol import (
    ConfigFieldsReadParams,
    ConfigFieldsReadResponse,
    ConfigMutationResponse,
    ConfigProxyReadParams,
    ConfigProxyReadResponse,
    ConfigProxyWriteParams,
    ConfigReadParams,
    ConfigReadResponse,
    ConfigReloadParams,
    ConfigWriteParams,
    ConfigWriteResponse,
    DiagnosticsListParams,
    DiagnosticsListResponse,
    DiagnosticsLogsReadParams,
    DiagnosticsLogsReadResponse,
    EmptyResponse,
    LoopsClearParams,
    LoopsClearResponse,
    LoopsCreateParams,
    LoopsCreateResponse,
    LoopsDeleteParams,
    LoopsDeleteResponse,
    LoopsListParams,
    LoopsListResponse,
    ProtocolErrorCode,
    RuntimeReadParams,
    RuntimeReadResponse,
    RuntimeSnapshot,
    SkillsInstalledParams,
    SkillsInstalledResponse,
    SkillsListParams,
    SkillsListResponse,
    StatsReadParams,
    StatsReadResponse,
    ToolsListParams,
    ToolsListResponse,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop import _PreparedReload
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layer import ConfigLayerError
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.log_reader import LogReader
from chartreux.core.loop import LoopError, LoopManager
from chartreux.core.proxy_setup import (
    SUPPORTED_PROXY_VARS,
    ProxySetupError,
    get_current_proxy_settings,
    set_proxy_var,
    unset_proxy_var,
    validate_proxy_var,
)
from chartreux.core.session_types import ScheduledLoop as CoreScheduledLoop


class ResourceRequestHandler:
    def __init__(
        self,
        agent_loop: AgentLoop,
        execution: SessionExecution,
        notify: Callable[[str, ProtocolModel], Awaitable[None]],
        current_event_id: Callable[[str], int] | None = None,
        initial_loops: list[CoreScheduledLoop] | None = None,
        reserve_config: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> None:
        self._agent_loop = agent_loop
        self._reserve_config = reserve_config or (
            lambda: execution.reserve(SessionExecutionKind.LIFECYCLE, "configuration")
        )
        self._execution = execution
        self._notify = notify
        self._current_event_id = current_event_id or (lambda _session_id: 0)
        self._loops = LoopManager(agent_loop.session_logger)
        self._logs = LogReader()
        self._mcp_discovery_errors: dict[str, str] = {}
        self.restore_loops(initial_loops)

    async def dispatch(self, method: str, raw_params: dict[str, Any]) -> DispatchResult:
        namespace = method.partition("/")[0]
        match namespace:
            case "runtime":
                result = self._dispatch_runtime(method, raw_params)
            case "config":
                result = await self._dispatch_config(method, raw_params)
            case "skills":
                result = await self._dispatch_skills(method, raw_params)
            case "tools" | "stats" | "diagnostics":
                result = self._dispatch_catalog(method, raw_params)
            case "loops":
                result = await self._dispatch_loops(method, raw_params)
            case _:
                raise method_not_found(method)
        return result

    def _dispatch_runtime(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        if method != "runtime/read":
            raise method_not_found(method)
        params = validate_wire(RuntimeReadParams, raw_params)
        self._require_session(params.session_id)
        return DispatchResult(
            RuntimeReadResponse(
                runtime=self.runtime_snapshot(),
                session_log=project_session_log(self._agent_loop),
                ready=self._agent_loop.is_initialized,
            )
        )

    def restore_loops(self, loops: list[CoreScheduledLoop] | None = None) -> None:
        metadata = self._agent_loop.session_logger.session_metadata
        restored = (
            loops
            if loops is not None
            else list(metadata.loops)
            if metadata is not None
            else []
        )
        self._loops.restore([loop.model_copy(deep=True) for loop in restored])
        if loops is not None and metadata is not None:
            metadata.loops = self._loops.loops

    def loop_snapshot(self) -> list[CoreScheduledLoop]:
        return [loop.model_copy(deep=True) for loop in self._loops.loops]

    async def persist_loops(self) -> None:
        await self._agent_loop.session_logger.persist_loops()

    async def copy_loops_to(self, target: AgentLoop) -> None:
        metadata = target.session_logger.session_metadata
        if metadata is not None:
            metadata.loops = self.loop_snapshot()
        await target.session_logger.persist_loops()

    def transfer_loops(self) -> None:
        metadata = self._agent_loop.session_logger.session_metadata
        if metadata is not None:
            metadata.loops = self._loops.loops

    def next_loop_due_in(self) -> float:
        return self._loops.next_due_in()

    def due_loop(self) -> CoreScheduledLoop | None:
        return self._loops.due()

    async def mark_loop_fired(self, loop_id: str) -> None:
        await self._loops.mark_fired(loop_id)

    def runtime_snapshot(self) -> RuntimeSnapshot:
        issues, hooks_count = project_diagnostics(self._agent_loop)
        return RuntimeSnapshot(
            config=project_config(self._agent_loop),
            skills=project_skills(self._agent_loop),
            tools=project_tools(self._agent_loop),
            stats=project_stats(self._agent_loop),
            context_window=self._context_window(),
            issues=issues,
            hooks_count=hooks_count,
            mcp=self._mcp_state(),
        )

    def _mcp_state(self) -> MCPState:
        self._mcp_discovery_errors.update(
            self._agent_loop.tool_manager.pop_mcp_errors()
        )
        return project_mcp(
            self._agent_loop, discovery_errors=self._mcp_discovery_errors
        )

    def _clear_mcp_discovery_errors(self) -> None:
        self._mcp_discovery_errors.clear()

    async def _dispatch_config(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        match method:
            case "config/read":
                response: ProtocolModel = self._config_read(
                    validate_wire(ConfigReadParams, raw_params)
                )
                runtime_updated = False
            case "config/write":
                write_response = await self._config_write(
                    validate_wire(ConfigWriteParams, raw_params)
                )
                response = write_response
                runtime_updated = write_response.application == "applied"
            case "config/fields/read":
                response = await self._config_fields_read(
                    validate_wire(ConfigFieldsReadParams, raw_params)
                )
                runtime_updated = False
            case "config/reload":
                response = await self._config_reload(
                    validate_wire(ConfigReloadParams, raw_params)
                )
                runtime_updated = True
            case "config/proxy/read":
                response = await self._config_proxy_read(
                    validate_wire(ConfigProxyReadParams, raw_params)
                )
                runtime_updated = False
            case "config/proxy/write":
                response = await self._config_proxy_write(
                    validate_wire(ConfigProxyWriteParams, raw_params)
                )
                runtime_updated = False
            case _:
                raise method_not_found(method)
        return DispatchResult(response, runtime_updated=runtime_updated)

    def _dispatch_catalog(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        match method:
            case "tools/list":
                response: ProtocolModel = self._tools_list(
                    validate_wire(ToolsListParams, raw_params)
                )
            case "stats/read":
                response = self._stats_read(validate_wire(StatsReadParams, raw_params))
            case "diagnostics/list":
                response = self._diagnostics_list(
                    validate_wire(DiagnosticsListParams, raw_params)
                )
            case "diagnostics/logs/read":
                response = self._diagnostics_logs_read(
                    validate_wire(DiagnosticsLogsReadParams, raw_params)
                )
            case _:
                raise method_not_found(method)
        return DispatchResult(response)

    async def _dispatch_loops(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        try:
            match method:
                case "loops/list":
                    params = validate_wire(LoopsListParams, raw_params)
                    self._require_session(params.session_id)
                    response: ProtocolModel = LoopsListResponse(
                        loops=[_project_loop(loop) for loop in self._loops.loops]
                    )
                case "loops/create":
                    self._execution.require_idle()
                    params = validate_wire(LoopsCreateParams, raw_params)
                    self._require_session(params.session_id)
                    response = LoopsCreateResponse(
                        loop=_project_loop(
                            await self._loops.create(params.interval, params.prompt)
                        )
                    )
                case "loops/delete":
                    self._execution.require_idle()
                    params = validate_wire(LoopsDeleteParams, raw_params)
                    self._require_session(params.session_id)
                    response = LoopsDeleteResponse(
                        loop=_project_loop(await self._loops.delete(params.loop_id))
                    )
                case "loops/clear":
                    self._execution.require_idle()
                    params = validate_wire(LoopsClearParams, raw_params)
                    self._require_session(params.session_id)
                    response = LoopsClearResponse(count=await self._loops.clear())
                case _:
                    raise method_not_found(method)
        except LoopError as exc:
            raise RequestFailure(ProtocolErrorCode.INVALID_PARAMS, str(exc)) from exc
        return DispatchResult(response)

    def _config_read(self, params: ConfigReadParams) -> ConfigReadResponse:
        if params.session_id is not None:
            self._require_session(params.session_id)
        config = project_config(self._agent_loop)
        skills_count = sum(
            1 for skill in project_skills(self._agent_loop) if skill.source == "local"
        )
        _, hooks_count = project_diagnostics(self._agent_loop)
        mcp_servers_total = len(self._agent_loop.config.mcp_servers)
        mcp_servers_enabled = sum(
            1 for server in self._agent_loop.config.mcp_servers if not server.disabled
        )
        return ConfigReadResponse(
            config=config,
            skills_count=skills_count,
            hooks_count=hooks_count,
            mcp_servers_total=mcp_servers_total,
            mcp_servers_enabled=mcp_servers_enabled,
        )

    async def _config_write(self, params: ConfigWriteParams) -> ConfigWriteResponse:
        self._require_session(params.session_id)
        with self._reserve_config():
            response = await self._config_write_reserved(params)
            fields, saved = config_write_projection(
                self._agent_loop.config_orchestrator, params.ops
            )
            response.fields = fields
            if response.persistence != "not_saved":
                response.saved_values = saved
            return response

    async def _config_write_reserved(
        self, params: ConfigWriteParams
    ) -> ConfigWriteResponse:
        loop = self._agent_loop
        orchestrator = loop.config_orchestrator
        prepared: _PreparedReload | None = None

        async def prepare(config: ChartreuxConfigSchema) -> None:
            nonlocal prepared
            # Do not leave an unjoined preparation thread behind on cancellation.
            prepared = loop._prepare_reload(config, True)

        def apply(config: ChartreuxConfigSchema) -> None:
            assert prepared is not None
            loop._commit_reload(prepared, True)

        try:
            operations = config_write_ops_to_patches(params.ops)
            if params.target != "session":
                result = await orchestrator.save(
                    operations,
                    target=params.target,
                    expected_revision=params.expected_revision or "",
                    reason=params.reason,
                    preflight=lambda candidate: prepare(candidate.config),
                    apply=lambda candidate: apply(candidate.config),
                )
                return ConfigWriteResponse(
                    runtime=self.runtime_snapshot(),
                    target=params.target,
                    persistence=result.persistence,
                    application=result.application,
                    revision=result.revision,
                    rejected=result.persistence == "not_saved",
                    failures=[result.error] if result.error else [],
                )
            if params.expected_revision is not None or any(
                op.target_layer_name not in {None, "overrides"} for op in operations
            ):
                raise ValueError("Session writes require the session target")
            failures = await orchestrator.apply_session_patch(
                operations, reason=params.reason, preflight=prepare, apply=apply
            )
            if failures:
                return ConfigWriteResponse(
                    runtime=self.runtime_snapshot(),
                    failures=[
                        orchestrator.thinking_patch_failure_detail([
                            (op.path, op.target_layer_name) for op in operations
                        ])
                        or "Configuration preparation or application failed"
                    ],
                )
            return ConfigWriteResponse(
                runtime=self.runtime_snapshot(), application="applied"
            )
        except Exception:
            return ConfigWriteResponse(
                runtime=self.runtime_snapshot(),
                target=params.target,
                rejected=True,
                failures=[
                    orchestrator.thinking_patch_failure_detail([
                        (op.path, op.target_layer) for op in params.ops
                    ])
                    or "Configuration preparation or application failed"
                ],
            )
        finally:
            if prepared is not None and prepared.backend is not loop.backend:
                loop._backend_lifetime.retire(
                    prepared.backend, whole_turn_active=loop._active_turn is not None
                )

    async def _config_reload(
        self, params: ConfigReloadParams
    ) -> ConfigMutationResponse:
        self._require_session(params.session_id)
        with self._reserve_config():
            orchestrator = self._agent_loop.config_orchestrator
            prepared: _PreparedReload | None = None

            async def prepare(config: ChartreuxConfigSchema) -> None:
                nonlocal prepared
                prepared = self._agent_loop._prepare_reload(config, True)

            def apply(config: ChartreuxConfigSchema) -> None:
                assert prepared is not None
                self._agent_loop._commit_reload(prepared, True)

            failed = False
            try:
                await orchestrator.reload(preflight=prepare, apply=apply)
            except Exception:
                failed = True
            finally:
                if (
                    prepared is not None
                    and prepared.backend is not self._agent_loop.backend
                ):
                    self._agent_loop._backend_lifetime.retire(
                        prepared.backend,
                        whole_turn_active=self._agent_loop._active_turn is not None,
                    )
            if failed:
                # Raise outside the handler: even __context__ must not retain
                # source values or credentials from a preparation exception.
                raise RequestFailure(
                    ProtocolErrorCode.INVALID_PARAMS,
                    "Configuration reload preparation or application failed",
                )
            self._clear_mcp_discovery_errors()
            try:
                await self._agent_loop.persist_launch_metadata()
            except Exception:
                # Runtime publication cannot be safely undone after the prepared
                # authority is committed. Report the applied-but-undurable result
                # explicitly so callers do not mistake it for a failed reload.
                return self._config_mutation_response(launch_metadata_persisted=False)
            return self._config_mutation_response()

    async def _config_proxy_read(
        self, params: ConfigProxyReadParams
    ) -> ConfigProxyReadResponse:
        self._require_session(params.session_id)
        values = await asyncio.to_thread(get_current_proxy_settings)
        return ConfigProxyReadResponse(
            settings=ProxySettingsView(values=values, descriptions=SUPPORTED_PROXY_VARS)
        )

    async def _config_proxy_write(
        self, params: ConfigProxyWriteParams
    ) -> EmptyResponse:
        self._execution.require_idle()
        self._require_session(params.session_id)

        def write() -> None:
            for key, value in params.changes.items():
                validate_proxy_var(key, value)
            for key, value in params.changes.items():
                if value:
                    set_proxy_var(key, value)
                else:
                    unset_proxy_var(key)

        try:
            await asyncio.to_thread(write)
        except ProxySetupError as exc:
            raise RequestFailure(ProtocolErrorCode.INVALID_PARAMS, str(exc)) from exc
        return EmptyResponse()

    async def _config_fields_read(
        self, params: ConfigFieldsReadParams
    ) -> ConfigFieldsReadResponse:
        self._require_session(params.session_id)
        orchestrator = self._agent_loop.config_orchestrator
        config = orchestrator.config
        layer_values = await collect_layer_values(orchestrator.layers)
        fields = [
            wire
            for wire in build_field_wires(
                config,
                layer_values,
                popular=POPULAR_SETTINGS,
                writable_targets={
                    name: config_field_write_targets(orchestrator, name)
                    for name in type(config).model_fields
                },
            )
            if wire.name not in HIDDEN_SETTINGS
        ]
        revisions: dict[str, str] = {}
        for layer in orchestrator.copy().layers:
            if isinstance(layer, (UserConfigLayer, ProjectConfigLayer)):
                try:
                    await layer.load(force=True)
                except ConfigLayerError:
                    # An unavailable source has no usable save revision, but must
                    # not prevent inspecting or saving the other sources.
                    continue
                if layer.fingerprint is not None:
                    revisions[layer.name] = layer.fingerprint
        return ConfigFieldsReadResponse(
            fields=fields, targets=self._config_targets(), revisions=revisions
        )

    def _config_targets(self) -> list[str]:
        return config_write_targets(self._agent_loop.config_orchestrator)

    def _skills_list(self, params: SkillsListParams) -> SkillsListResponse:
        self._require_session(params.session_id)
        return SkillsListResponse(skills=project_skills(self._agent_loop))

    async def _dispatch_skills(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        match method:
            case "skills/list":
                response: ProtocolModel = self._skills_list(
                    validate_wire(SkillsListParams, raw_params)
                )
                runtime_updated = False
            case "skills/installed":
                response = self._skills_installed(
                    validate_wire(SkillsInstalledParams, raw_params)
                )
                runtime_updated = False
            case _:
                raise method_not_found(method)
        return DispatchResult(response, runtime_updated=runtime_updated)

    def _skills_installed(
        self, params: SkillsInstalledParams
    ) -> SkillsInstalledResponse:
        self._require_session(params.session_id)
        return SkillsInstalledResponse(
            skills=project_installed_skills(self._agent_loop)
        )

    def _tools_list(self, params: ToolsListParams) -> ToolsListResponse:
        self._require_session(params.session_id)
        return ToolsListResponse(tools=project_tools(self._agent_loop))

    def _stats_read(self, params: StatsReadParams) -> StatsReadResponse:
        self._require_session(params.session_id)
        return StatsReadResponse(
            stats=project_stats(self._agent_loop), context_window=self._context_window()
        )

    def _context_window(self) -> int:
        try:
            return self._agent_loop.config.get_active_model().auto_compact_threshold
        except ValueError:
            return 0

    def _diagnostics_list(
        self, params: DiagnosticsListParams
    ) -> DiagnosticsListResponse:
        self._require_session(params.session_id)
        issues, hooks_count = project_diagnostics(self._agent_loop)
        return DiagnosticsListResponse(issues=issues, hooks_count=hooks_count)

    def _diagnostics_logs_read(
        self, params: DiagnosticsLogsReadParams
    ) -> DiagnosticsLogsReadResponse:
        self._require_session(params.session_id)
        logs = self._logs.get_logs(limit=params.limit, offset=params.offset)
        return DiagnosticsLogsReadResponse(logs=project_debug_logs(logs))

    def _config_mutation_response(
        self, *, launch_metadata_persisted: bool = True
    ) -> ConfigMutationResponse:
        return ConfigMutationResponse(
            runtime=self.runtime_snapshot(),
            stripped_history_images=(
                self._agent_loop.count_history_images_unsupported_by_active_model()
            ),
            launch_metadata_persisted=launch_metadata_persisted,
        )

    def _require_session(self, session_id: str) -> None:
        if session_id != self._agent_loop.session_id:
            raise RequestFailure(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {session_id}"
            )


def _project_loop(loop: CoreScheduledLoop) -> ScheduledLoop:
    return ScheduledLoop(
        id=loop.id,
        prompt=loop.prompt,
        interval_seconds=loop.interval_seconds,
        next_fire_at=loop.next_fire_at,
    )
