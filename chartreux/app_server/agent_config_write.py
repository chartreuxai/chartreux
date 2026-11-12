"""The mode-(d) config transaction, replayed where the loop lives.

``config/write`` is executed by the process that owns the session, so the
reconfigure request carries the config-write candidate and this adapter runs
the shipped transaction there: PREPARE on a candidate backend, the atomic
COMMIT that publishes it, the shipped compensation that retires a prepared
backend the commit did not adopt, and the changed-field projection the
coordinator reads.

The receipt is the SHIPPED response, carried wholesale and verbatim: the
persistence x application dimensions, the rejected flag, the failure details,
the revision, the saved values, the changed fields, and the launch-metadata
outcome. Nothing here re-classifies them into a smaller set --
APPLIED-BUT-UNDURABLE is the composition it already is, and every "nothing
applied" answer is scoped to RUNTIME application, never to durable disk state:
an unchanged or failed runtime can coexist with a source that was saved.

The ``reload_runtime`` flag is not carried on this path: mode (d) IS the
runtime-reload leg, triggered by its own request rather than by reading a flag,
so the adapter always runs the prepare/commit legs.
"""

from __future__ import annotations

from chartreux.app_server._config_write import (
    config_write_ops_to_patches,
    config_write_projection,
    touches_web_search,
    validate_web_search_candidate,
    validate_web_search_write_ops,
)
from chartreux.app_server._projection import (
    project_config,
    project_diagnostics,
    project_mcp,
    project_skills,
    project_stats,
    project_tools,
)
from chartreux.app_server.agent_wire import AgentSessionNotFoundError
from chartreux.app_server.protocol import (
    ConfigWriteParams,
    ConfigWriteResponse,
    RuntimeSnapshot,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop import _PreparedReload
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema


class AgentConfigWriteService:
    """Serve one config-write transaction against the loop this agent owns."""

    def __init__(self, loop: AgentLoop) -> None:
        self._loop = loop

    async def write(self, params: ConfigWriteParams) -> ConfigWriteResponse:
        """Run the transaction and answer with the shipped response, projected."""
        self._require_session(params.session_id)
        response = await self._write_reserved(params)
        fields, saved = config_write_projection(
            self._loop.config_orchestrator, params.ops
        )
        response.fields = fields
        if response.persistence != "not_saved":
            response.saved_values = saved
        return response

    async def _write_reserved(self, params: ConfigWriteParams) -> ConfigWriteResponse:
        """PREPARE, persist, apply, and compensate -- the shipped order."""
        loop = self._loop
        orchestrator = loop.config_orchestrator
        prepared: _PreparedReload | None = None
        validate_search = touches_web_search(params.ops)

        async def prepare(config: ChartreuxConfigSchema) -> None:
            nonlocal prepared
            if validate_search:
                validate_web_search_candidate(config)
            # Do not leave an unjoined preparation thread behind on cancellation.
            prepared = loop._prepare_reload(config, True)

        def apply(config: ChartreuxConfigSchema) -> None:
            assert prepared is not None
            loop._commit_reload(prepared, True)

        try:
            if validate_search:
                validate_web_search_write_ops(params.ops)
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
                return self._response(
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
                return self._response(
                    failures=[
                        orchestrator.thinking_patch_failure_detail([
                            (op.path, op.target_layer_name) for op in operations
                        ])
                        or "Configuration preparation or application failed"
                    ]
                )
            return self._response(application="applied")
        except Exception:
            return self._response(
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
            # The shipped compensation: a prepared backend the commit did not
            # adopt is retired here, under the backend lifetime's dispose order.
            if prepared is not None and prepared.backend is not loop.backend:
                loop._backend_lifetime.retire(
                    prepared.backend, whole_turn_active=loop._active_turn is not None
                )

    def _response(self, **kwargs: object) -> ConfigWriteResponse:
        """The shipped response shape: the runtime projection plus the outcome."""
        return ConfigWriteResponse(
            runtime=self._runtime_snapshot(),
            stripped_history_images=(
                self._loop.count_history_images_unsupported_by_active_model()
            ),
            **kwargs,  # type: ignore[arg-type]
        )

    def _runtime_snapshot(self) -> RuntimeSnapshot:
        """The runtime view this agent publishes, built after the transaction."""
        loop = self._loop
        issues, hooks_count = project_diagnostics(loop)
        try:
            context_window = loop.config.get_active_model().context_limit
        except ValueError:
            context_window = None
        return RuntimeSnapshot(
            config=project_config(loop),
            skills=project_skills(loop),
            tools=project_tools(loop),
            stats=project_stats(loop),
            context_window=context_window,
            issues=issues,
            hooks_count=hooks_count,
            mcp=project_mcp(loop),
        )

    def _require_session(self, session_id: str) -> None:
        if session_id != self._loop.session_id:
            raise AgentSessionNotFoundError(f"Session not found: {session_id}")


__all__ = ["AgentConfigWriteService"]
