"""Server-side plan handoff: admission, consent-snapshot apply-exactly-or-fail,
and the prompt-freshness ordering (the PMD6R-W01(G) clause fix: the executor
install sits between the unpin and the reload, and the reload is the named
refresh step for the unpinned case too).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

import chartreux.app_server._handler as handler_module
from chartreux.app_server._session_model import (
    override_active_model,
    override_agent_identity,
    override_thinking_level,
    set_session_thinking_override,
)
from chartreux.app_server.agent_reconfigure import AgentReconfigureService
from chartreux.app_server.agent_snapshot import build_session_snapshot
from chartreux.app_server.agent_wire import (
    RECONFIGURE_MODE_PLAN_HANDOFF,
    RECONFIGURE_OUTCOME_ACCEPTED,
    ReconfigureRequest,
)
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.models import PublicTurn
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ProtocolErrorCode,
    SessionPlanHandoffParams,
    SessionPlanHandoffResponse,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop, AgentLoopStateError
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import ModelConfig, SessionLoggingConfig
from chartreux.core.config.orchestrator import ConfigOrchestrator
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    legacy_backend,
)

PLANNER = "alpha"
EXECUTOR = "beta"


async def _layered_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    session_logging: SessionLoggingConfig | None = None,
    planner_model_pin: bool = False,
) -> AgentLoop:
    """A loop on a real orchestrator with a session-scoped overrides layer.

    The user layer pins the planner model so the effective selection starts
    on the planner without a session pin; ``planner_model_pin`` additionally
    simulates the /plan-era ``/active_model`` override.
    """
    models = [
        ModelConfig(name="alpha", provider="mistral", alias="alpha"),
        ModelConfig(name="beta", provider="mistral", alias="beta"),
    ]
    config = build_test_vibe_config(
        active_model=PLANNER,
        models=models,
        session_logging=session_logging or SessionLoggingConfig(enabled=False),
    )

    async def load(
        cfg: ChartreuxConfigSchema,
    ) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        user_path = tmp_path / "user-config.toml"
        user_path.write_text('theme = "dark"\n')
        user = UserConfigLayer(path=user_path)
        launch = OverridesLayer(
            name="launch",
            data={
                "active_model": PLANNER,
                "session_logging": cfg.session_logging.model_dump(mode="json"),
            },
        )
        session_data: dict[str, Any] = {}
        if planner_model_pin:
            session_data["active_model"] = PLANNER
        return await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=[user, launch, OverridesLayer(data=session_data)],
            default_layer_resolver=lambda: user,
            catalog_snapshot=cfg.catalog_snapshot,
        )

    monkeypatch.setattr("tests.conftest._load_orchestrator", load)
    return build_test_agent_loop(config=config, session_id="initial")


def _params(**overrides: Any) -> SessionPlanHandoffParams:
    base: dict[str, Any] = {
        "session_id": "initial",
        "executor_model": EXECUTOR,
        "executor_thinking": "medium",
        "executor_identity": "orchestrator",
        "plan_path": "/tmp/plan-workspace/plan.md",
    }
    return SessionPlanHandoffParams(**(base | overrides))


async def _serve(loop: AgentLoop) -> tuple[AppServerClient, Any, Any]:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    return client, session, legacy_backend(server)


@pytest.mark.asyncio
async def test_plan_handoff_applies_consent_snapshot_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _layered_loop(monkeypatch, tmp_path)
    orch = loop.config_orchestrator
    # /plan entry wrote the planner-alias thinking key (absent pre-entry).
    assert (
        await set_session_thinking_override(orch, PLANNER, "high", reason="plan entry")
        == []
    )
    calls: list[str] = []
    real_reload = loop.reload_with_initial_messages

    async def _spy_reload(*args: Any, **kwargs: Any) -> Any:
        calls.append("reload")
        return await real_reload(*args, **kwargs)

    real_set_model = handler_module.set_session_active_model_override

    async def _spy_set_model(*args: Any, **kwargs: Any) -> Any:
        calls.append("install-model")
        return await real_set_model(*args, **kwargs)

    monkeypatch.setattr(loop, "reload_with_initial_messages", _spy_reload)
    monkeypatch.setattr(
        handler_module, "set_session_active_model_override", _spy_set_model
    )

    client, session, _backend = await _serve(loop)
    try:
        raw = await client.request(
            "session/plan/handoff",
            _params(planner_thinking_alias=PLANNER, planner_thinking_value=None),
        )
        response = SessionPlanHandoffResponse.model_validate(raw)
        assert response.state.session.id != "initial"
        assert response.state.session.id == loop.session_id
        # The consented snapshot is applied exactly.
        assert override_active_model(orch) == EXECUTOR
        assert override_thinking_level(orch, EXECUTOR) == "medium"
        assert loop._bound_identity == "orchestrator"
        assert loop.config.get_active_model().alias == EXECUTOR
        # The identity pin is removed after the fresh bind.
        assert override_agent_identity(orch) is None
        # The planner-alias key written at entry is removed (absent pre-entry).
        assert override_thinking_level(orch, PLANNER) is None
        # PMD6R-W01(G): the executor install sits between the unpin and the
        # reload, and the reload runs in the unpinned case too (session
        # logging disabled here, so the clear route's logger-gated branch
        # would never have refreshed the prompt).
        assert calls == ["install-model", "reload"]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_plan_handoff_pinned_case_unpins_installs_then_reloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logging_config = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions")
    )
    loop = await _layered_loop(
        monkeypatch, tmp_path, session_logging=logging_config, planner_model_pin=True
    )
    await loop.persist_empty_session()
    # The persisted session log carries the planner-era active model, which is
    # what gates the clear route's unpin branch.
    assert loop.session_logger.active_model == PLANNER
    orch = loop.config_orchestrator
    assert override_active_model(orch) == PLANNER

    calls: list[str] = []
    real_unpin = handler_module.clear_session_active_model_override

    async def _spy_unpin(*args: Any, **kwargs: Any) -> Any:
        calls.append("unpin")
        return await real_unpin(*args, **kwargs)

    real_set_model = handler_module.set_session_active_model_override

    async def _spy_set_model(*args: Any, **kwargs: Any) -> Any:
        calls.append("install-model")
        return await real_set_model(*args, **kwargs)

    real_reload = loop.reload_with_initial_messages

    async def _spy_reload(*args: Any, **kwargs: Any) -> Any:
        calls.append("reload")
        return await real_reload(*args, **kwargs)

    monkeypatch.setattr(
        handler_module, "clear_session_active_model_override", _spy_unpin
    )
    monkeypatch.setattr(
        handler_module, "set_session_active_model_override", _spy_set_model
    )
    monkeypatch.setattr(loop, "reload_with_initial_messages", _spy_reload)

    client, session, _backend = await _serve(loop)
    try:
        raw = await client.request("session/plan/handoff", _params())
        SessionPlanHandoffResponse.model_validate(raw)
        assert calls == ["unpin", "install-model", "reload"]
        assert override_active_model(orch) == EXECUTOR
        assert loop.config.get_active_model().alias == EXECUTOR
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_plan_handoff_refuses_active_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _layered_loop(monkeypatch, tmp_path)
    client, session, backend = await _serve(loop)
    try:
        backend.handler._turns._active_turn = cast(PublicTurn, object())
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request("session/plan/handoff", _params())
        assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT
        assert "turn is active" in exc_info.value.error.message
        assert "finish or stop the work first" in exc_info.value.error.message
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_plan_handoff_refuses_active_background_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _layered_loop(monkeypatch, tmp_path)

    def _raise_jobs(*args: Any, **kwargs: Any) -> None:
        raise AgentLoopStateError(
            "Stop live background jobs before changing execution authority"
        )

    monkeypatch.setattr(loop, "require_background_jobs_idle", _raise_jobs)
    client, session, _backend = await _serve(loop)
    try:
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request("session/plan/handoff", _params())
        assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT
        assert "background work is active" in exc_info.value.error.message
        assert "Finish or stop the work first" in exc_info.value.error.message
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_plan_handoff_apply_mismatch_fails_without_starting_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _layered_loop(monkeypatch, tmp_path)
    orch = loop.config_orchestrator

    async def _noop_install(*args: Any, **kwargs: Any) -> list[BaseException]:
        # Simulate an install that reported success but did not take.
        return []

    monkeypatch.setattr(handler_module, "set_session_thinking_override", _noop_install)
    client, session, backend = await _serve(loop)
    try:
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request("session/plan/handoff", _params())
        assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT
        assert (
            "could not apply the consented executor selection exactly"
            in exc_info.value.error.message
        )
        data = cast(dict[str, Any], exc_info.value.error.data)
        assert data["appliedThinking"] is None
        # Execution was not started: no idle replacement was published.
        assert backend.coordinator.attached_session_id == "initial"
    finally:
        await session.close()
    assert override_active_model(orch) == EXECUTOR


@pytest.mark.asyncio
async def test_plan_handoff_reset_failure_reports_and_cleans_identity_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _layered_loop(monkeypatch, tmp_path)
    orch = loop.config_orchestrator

    async def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom during reset")

    monkeypatch.setattr(loop, "clear_history", _boom)
    client, session, _backend = await _serve(loop)
    try:
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request("session/plan/handoff", _params())
        assert exc_info.value.error.code is ProtocolErrorCode.INTERNAL_ERROR
        assert "Plan handoff reset failed" in exc_info.value.error.message
        assert "boom during reset" in exc_info.value.error.message
        assert "leftover identity pin removed" in exc_info.value.error.message
    finally:
        await session.close()
    assert override_agent_identity(orch) is None


@pytest.mark.asyncio
async def test_plan_handoff_reset_cancellation_propagates_after_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _layered_loop(monkeypatch, tmp_path)
    orch = loop.config_orchestrator

    async def _cancel(*args: Any, **kwargs: Any) -> None:
        assert override_agent_identity(orch) == "orchestrator"
        raise asyncio.CancelledError("cancelled during reset")

    monkeypatch.setattr(loop, "clear_history", _cancel)
    client, session, backend = await _serve(loop)
    try:
        with pytest.raises(asyncio.CancelledError, match="cancelled during reset"):
            await backend.handler._plan_handoff(_params())
        assert override_agent_identity(orch) is None
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_plan_handoff_executor_install_is_final_write_on_shared_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _layered_loop(monkeypatch, tmp_path)
    orch = loop.config_orchestrator
    # /plan entry wrote the planner thinking key on the SAME alias the
    # executor will use.
    assert (
        await set_session_thinking_override(orch, PLANNER, "high", reason="plan entry")
        == []
    )
    client, session, _backend = await _serve(loop)
    try:
        raw = await client.request(
            "session/plan/handoff",
            _params(
                executor_model=PLANNER,
                executor_thinking="medium",
                planner_thinking_alias=PLANNER,
                planner_thinking_value="low",
            ),
        )
        SessionPlanHandoffResponse.model_validate(raw)
        # The planner-key restore ran first, then the executor install is the
        # final write on the shared alias.
        assert override_thinking_level(orch, PLANNER) == "medium"
        assert override_active_model(orch) == PLANNER
        assert loop._bound_identity == "orchestrator"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_plan_handoff_restores_planner_key_to_pre_entry_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _layered_loop(monkeypatch, tmp_path)
    orch = loop.config_orchestrator
    assert (
        await set_session_thinking_override(orch, PLANNER, "high", reason="plan entry")
        == []
    )
    client, session, _backend = await _serve(loop)
    try:
        raw = await client.request(
            "session/plan/handoff",
            _params(planner_thinking_alias=PLANNER, planner_thinking_value="low"),
        )
        SessionPlanHandoffResponse.model_validate(raw)
        assert override_thinking_level(orch, PLANNER) == "low"
        assert override_thinking_level(orch, EXECUTOR) == "medium"
    finally:
        await session.close()


# --- Item 1 op 10: the handoff carried over the process wire ---------------
#
# The pipe carries no deferred action, so the post-response step the in-process
# handler answers with runs inside the operation boundary, on the live loop,
# before the reply is written; its outcome rides the receipt instead of a
# promise the peer cannot keep.


def _reconfigure_service(loop: AgentLoop, after: Any) -> AgentReconfigureService:
    return AgentReconfigureService(
        loop,
        turn_active=lambda: False,
        snapshot=lambda: build_session_snapshot(
            loop, markers=(), incarnation_id="inc-reconfigure"
        ),
        after_response=after,
    )


def _handoff_request() -> ReconfigureRequest:
    return ReconfigureRequest(
        mode=RECONFIGURE_MODE_PLAN_HANDOFF, session_id="initial", handoff=_params()
    )


@pytest.mark.asyncio
async def test_the_wire_handoff_runs_the_post_response_step_before_answering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _layered_loop(monkeypatch, tmp_path)
    observed: list[str] = []

    async def after(mode: str) -> str:
        # Read the LIVE selection at call time: the step runs after the
        # executor install, inside the operation boundary.
        observed.append(f"{mode}:{override_active_model(loop.config_orchestrator)}")
        return "post-response-ran"

    reply = await _reconfigure_service(loop, after).plan_handoff(_handoff_request())

    # The step already ran when the reply was built, and its outcome is part of
    # the receipt: there is no deferred action for the peer to run later.
    assert observed == [f"{RECONFIGURE_MODE_PLAN_HANDOFF}:{EXECUTOR}"]
    assert reply.outcome == RECONFIGURE_OUTCOME_ACCEPTED
    assert reply.followup == "post-response-ran"
    assert reply.snapshot is not None
    assert reply.episode_id is None


@pytest.mark.asyncio
async def test_a_failed_post_response_step_rides_the_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = await _layered_loop(monkeypatch, tmp_path)

    async def after(_mode: str) -> str:
        raise RuntimeError("the queued prompt refused")

    reply = await _reconfigure_service(loop, after).plan_handoff(_handoff_request())

    # The handoff itself landed, so it is not reported as a failed accept; the
    # step that could not finish is named in the receipt.
    assert reply.outcome == RECONFIGURE_OUTCOME_ACCEPTED
    assert "the post-response step failed" in reply.followup
    assert "the queued prompt refused" in reply.followup
    assert override_active_model(loop.config_orchestrator) == EXECUTOR
