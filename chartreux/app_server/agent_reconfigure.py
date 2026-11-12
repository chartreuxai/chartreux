"""Agent-side adapters for the reconfigure modes this release carries.

The coordinator reaches the root session through one reconfigure request per
invocation mode, and the modes that mutate the session -- the settings
apply-and-ack, the /clear history reset, the /plan accept handoff, and the
config write with its prepare/commit-reload legs -- are served here, next to
the loop that owns the state they change. Nothing the coordinator owns moves
across the boundary: the plan-episode settlement and the pin removal that must
precede a /clear stay the coordinator's own step, and a refused settlement
means no clear frame is ever sent.

The plan handoff is the in-process transaction (``_handler._plan_handoff``)
carried to this side: PREPARE claims the episode and publishes the consent
snapshot, COMMIT holds the fence through the durable retirement, the identity
and selection install, the read-back, the executor marker, the injection, the
terminal flip and the derived rename. The one shape change is the deferred
post-response callable: the pipe carries no deferred action, so whatever it
would do runs here, inside the operation boundary and before the reply is
written, and its outcome rides the receipt.

The clear acks success only after verifying, against LIVE runtime state, that
the history is gone and no session model override remains -- the override clear
is driven from that state rather than from the session metadata a reset has
just rewritten, and the receipt's ``stage`` names exactly the steps this call
completed, so a retry after a partial failure re-enters and finishes the rest.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from chartreux.app_server._session_model import (
    clear_session_active_model_override,
    clear_session_agent_identity_override,
    override_active_model,
    override_agent_identity,
    override_thinking_level,
    remove_session_thinking_override,
    set_session_active_model_override,
    set_session_agent_identity_override,
    set_session_thinking_override,
)
from chartreux.app_server.agent_config_write import AgentConfigWriteService
from chartreux.app_server.agent_snapshot import AgentSessionSnapshot, RootOpenAccounting
from chartreux.app_server.agent_wire import (
    CLEAR_STAGE_ALREADY_CLEAR,
    CLEAR_STAGE_HISTORY,
    CLEAR_STAGE_MODEL_OVERRIDE,
    RECONFIGURE_OUTCOME_ACCEPTED,
    RECONFIGURE_OUTCOME_APPLIED,
    RECONFIGURE_OUTCOME_CLEARED,
    RECONFIGURE_OUTCOME_CLEARED_WITH_FOLLOWUP_FAILURE,
    AgentReconfigureConflictError,
    AgentReconfigureFailedError,
    AgentReconfigureRefusedError,
    AgentSessionNotFoundError,
    ReconfigureReply,
    ReconfigureRequest,
)
from chartreux.app_server.protocol import SessionPlanHandoffParams
from chartreux.core.agent_loop import AgentLoop, AgentLoopStateError
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.plan_episode import (
    LOGGING_REQUIRED_MESSAGE,
    STATE_ACCEPTED,
    STATE_ACCEPTING,
    CommitAttempt,
    LoggingDomain,
    LoggingObservation,
    PlanDigestMismatchError,
    PlanEpisodeError,
    PlanEpisodeWriteError,
    PlanFenceBusyError,
    PlanTransitionError,
    begin_commit,
    build_handoff_context,
    observe_logging_domain,
    prepare_accept,
    read_sidecar,
    release_claim,
    require_no_executor,
    sha256_bytes,
)
from chartreux.core.plan_mode import extract_plan_summary
from chartreux.core.session.session_logger import SessionDiskFullError
from chartreux.core.session_types import PlanModeRecord, PlanScanDomain
from chartreux.utils.io import decode_safe

PLAN_COMMIT_FENCE_RETRY_SECONDS = 5.0
"""A competing accept busy-retries a bounded moment, then loses the fence."""


type AfterResponse = Callable[[str], Awaitable[str | None]]
type SnapshotSource = Callable[[], AgentSessionSnapshot]


class AgentReconfigureService:
    """Serve the reconfigure modes against the loop this agent owns."""

    def __init__(
        self,
        loop: AgentLoop,
        *,
        turn_active: Callable[[], bool],
        snapshot: SnapshotSource,
        after_response: AfterResponse | None = None,
    ) -> None:
        self._loop = loop
        self._turn_active = turn_active
        self._snapshot = snapshot
        self._after_response = after_response

    def settings_update(self, request: ReconfigureRequest) -> ReconfigureReply:
        """Apply the requested bounds and acknowledge what was applied."""
        self._require_session(request.session_id)
        if request.max_turns is not None:
            self._loop.set_max_turns(request.max_turns)
        if request.max_tokens is not None:
            self._loop.set_max_tokens(request.max_tokens)
        return ReconfigureReply(
            mode=request.mode,
            outcome=RECONFIGURE_OUTCOME_APPLIED,
            max_turns=request.max_turns,
            max_tokens=request.max_tokens,
        )

    async def history_clear(self, request: ReconfigureRequest) -> ReconfigureReply:
        """Clear the session history and its model pin, or report what failed."""
        self._require_session(request.session_id)
        if self._turn_active():
            raise AgentReconfigureConflictError(
                "Cannot clear history while a turn is active"
            )
        loop = self._loop
        orchestrator = loop.config_orchestrator
        stages: list[str] = []
        if _conversation_messages(loop):
            try:
                await loop.clear_history()
            except Exception as exc:
                raise AgentReconfigureFailedError(
                    f"Failed to clear history: {exc}"
                ) from exc
            stages.append(CLEAR_STAGE_HISTORY)
        # The override clear is driven from LIVE runtime state: the metadata a
        # reset just rewrote is not the authority for whether a pin remains.
        if override_active_model(orchestrator) is not None:
            failures = await clear_session_active_model_override(
                orchestrator, reason="clear session active model"
            )
            if failures:
                return self._cleared_with_followup(
                    request,
                    stages,
                    f"Failed to clear session active model: {failures[0]}",
                )
            stages.append(CLEAR_STAGE_MODEL_OVERRIDE)
            try:
                await loop.reload_with_initial_messages()
            except Exception as exc:
                return self._cleared_with_followup(
                    request, stages, f"The prompt refresh after the clear failed: {exc}"
                )
        # The terminal-state check that gates the success ack (S6-B5).
        if _conversation_messages(loop):
            raise AgentReconfigureFailedError(
                "The session history is still present after the clear"
            )
        if override_active_model(orchestrator) is not None:
            return self._cleared_with_followup(
                request,
                stages,
                "The session active model override is still active after the "
                "clear; retry the clear to remove it",
            )
        followup = await self._run_after_response(request.mode)
        return ReconfigureReply(
            mode=request.mode,
            outcome=RECONFIGURE_OUTCOME_CLEARED,
            stage=",".join(stages) or CLEAR_STAGE_ALREADY_CLEAR,
            followup=followup,
            snapshot=self._snapshot(),
        )

    async def config_write(self, request: ReconfigureRequest) -> ReconfigureReply:
        """Run the config transaction here and carry its shipped result back.

        The receipt is the shipped response (persistence x application, the
        rejected flag, the failure details, the revision, the saved values and
        the launch-metadata outcome), so the outcome this reply names is that
        response's own runtime application dimension -- nothing is collapsed
        into a smaller class, and an applied-but-undurable transaction stays the
        composition it is.
        """
        candidate = request.config_write
        if candidate is None:
            raise AgentReconfigureRefusedError(
                "The config write carries no candidate transaction"
            )
        if self._turn_active():
            raise AgentReconfigureConflictError(
                "Cannot write the configuration while a turn is active; "
                "finish or stop the work first"
            )
        response = await AgentConfigWriteService(self._loop).write(candidate)
        return ReconfigureReply(
            mode=request.mode,
            outcome=response.application,
            snapshot=self._snapshot(),
            accounting=RootOpenAccounting.from_loop(self._loop),
            config_write=response,
        )

    async def plan_handoff(self, request: ReconfigureRequest) -> ReconfigureReply:
        """Start the consented executor session for an accepted plan.

        Mirrors the in-process transaction step for step; the refusals keep
        their classes (a conflicting state is a conflict, a malformed body a
        param refusal) and every durable write that cannot be proven loud
        fails the handoff instead of reporting a clean success.
        """
        params = request.handoff
        if params is None:
            raise AgentReconfigureRefusedError(
                "The plan handoff carries no consent snapshot"
            )
        self._require_session(params.session_id)
        if self._turn_active():
            raise AgentReconfigureConflictError(
                "Cannot start plan execution while a turn is active; "
                "finish or stop the work first"
            )
        loop = self._loop
        try:
            loop.require_background_jobs_idle()
        except AgentLoopStateError as exc:
            raise AgentReconfigureConflictError(
                f"Cannot start plan execution while background work is "
                f"active: {exc}. Finish or stop the work first."
            ) from exc
        body = _handoff_verified_body(params)
        self._require_route(body)
        if body is None or not self._continuation(params, body):
            await self._prepare(params)
        with ExitStack() as stack:
            commit: CommitAttempt | None = None
            retired: PlanModeRecord | None = None
            if body is not None:
                commit = self._begin_commit(params, body)
                stack.callback(commit.close)
                retired = await self._retire_source(params, commit)
            followup = await self._hand_off(params, request.mode, commit, retired)
            executor_session_id = loop.session_id
            if commit is not None:
                self._record_executor(commit, executor_session_id)
                self._finalize(commit, executor_session_id)
        return ReconfigureReply(
            mode=request.mode,
            outcome=RECONFIGURE_OUTCOME_ACCEPTED,
            followup=followup,
            snapshot=self._snapshot(),
            accounting=RootOpenAccounting.from_loop(loop),
            episode_id=None if commit is None else commit.episode_id,
            attempt_generation=None if commit is None else commit.attempt_generation,
        )

    async def _hand_off(
        self,
        params: SessionPlanHandoffParams,
        mode: str,
        commit: CommitAttempt | None,
        retired: PlanModeRecord | None,
    ) -> str:
        """Install the consented selection, mark the executor, and answer.

        Runs inside COMMIT's held fence: the identity pin, the reset, the
        unpins, the planner-key restore, the executor install, the read-back
        that refuses a selection that did not take, then the marker, the
        injection, the prompt refresh and the post-response step -- in that
        order, because each later step reads what an earlier one wrote.
        """
        loop = self._loop
        orchestrator = loop.config_orchestrator
        # The executor identity rides as a session-scoped override written
        # BEFORE the fresh root bind, so the reset binds it; the bound bundle
        # freezes it and the pin is removed again after it.
        identity_failures = await set_session_agent_identity_override(
            orchestrator,
            params.executor_identity,
            reason="plan handoff executor identity",
        )
        if identity_failures:
            raise AgentReconfigureFailedError(
                f"Failed to install plan executor identity: {identity_failures[0]}"
            )
        session_model_pinned = loop.session_logger.active_model is not None
        try:
            await loop.clear_history()
        except BaseException as exc:
            cleanup = await clear_session_agent_identity_override(
                orchestrator, reason="plan handoff reset failure cleanup"
            )
            if not isinstance(exc, Exception):
                raise
            detail = (
                "leftover identity pin removed"
                if not cleanup
                else f"leftover identity pin could not be removed: {cleanup[0]}"
            )
            raise AgentReconfigureFailedError(
                f"Plan handoff reset failed: {exc} ({detail})"
            ) from exc
        unpin_failures = await clear_session_agent_identity_override(
            orchestrator, reason="plan handoff identity unpin after bind"
        )
        if unpin_failures:
            raise AgentReconfigureFailedError(
                f"Failed to remove plan executor identity pin: {unpin_failures[0]}"
            )
        if session_model_pinned:
            failures = await clear_session_active_model_override(
                orchestrator, reason="plan handoff active model unpin"
            )
            if failures:
                raise AgentReconfigureFailedError(
                    f"Failed to clear session active model: {failures[0]}"
                )
        # Restore the planner-alias thinking key /plan wrote at entry BEFORE the
        # executor install, so the install stays the final write on its alias
        # even when both share it.
        restore_failures = await self._restore_planner_key(orchestrator, params)
        if restore_failures:
            raise AgentReconfigureFailedError(
                f"Failed to restore planner thinking key: {restore_failures[0]}"
            )
        model_failures = await set_session_active_model_override(
            orchestrator, params.executor_model, reason="plan handoff executor model"
        )
        thinking_failures = await set_session_thinking_override(
            orchestrator,
            params.executor_model,
            params.executor_thinking,
            reason="plan handoff executor thinking",
        )
        install_failures = [*model_failures, *thinking_failures]
        if install_failures:
            raise AgentReconfigureFailedError(
                f"Failed to install plan executor selection: {install_failures[0]}"
            )
        mismatch = self._selection_mismatch(orchestrator, params)
        if mismatch is not None:
            applied = ", ".join(
                f"{name}={value!r}" for name, value in sorted(mismatch.items())
            )
            raise AgentReconfigureConflictError(
                "Plan handoff could not apply the consented executor selection "
                f"exactly ({applied}); execution was not started"
            )
        if commit is not None:
            assert retired is not None
            await self._mark_and_inject(commit, retired)
        # Prompt refresh: reload_with_initial_messages is the only refresh hook,
        # and it runs after the executor install.
        await loop.reload_with_initial_messages()
        return await self._run_after_response(mode)

    async def _prepare(self, params: SessionPlanHandoffParams) -> None:
        """PREPARE (P1-P3) of an episode-bound handoff."""
        body = _handoff_verified_body(params)
        if body is None:
            return
        episode_id, snapshot_path, content = body
        refusal = _draft_refusal(content)
        if refusal is not None:
            raise AgentReconfigureConflictError(refusal)
        try:
            prepare_accept(
                snapshot_path.parent,
                episode_id=episode_id,
                session_id=params.session_id,
                owner_domain=self._logging_domain(),
                previewed_bytes=content,
                previewed_sha256=sha256_bytes(content),
            )
        except PlanTransitionError as exc:
            raise AgentReconfigureConflictError(str(exc)) from exc
        except PlanEpisodeError as exc:
            raise AgentReconfigureFailedError(str(exc)) from exc

    def _require_route(self, body: tuple[str, Path, bytes] | None) -> None:
        """Refuse a handoff whose shape contradicts this session's plan record."""
        record = self._plan_record()
        if record is None:
            return
        if record.episode_id is None:
            raise AgentReconfigureConflictError(
                "This plan record is legacy-unverified, so it cannot accept: "
                "re-bind the draft into a new plan episode and accept again, or "
                "cancel plan mode"
            )
        if body is None:
            raise AgentReconfigureRefusedError(
                f"The plan record for episode {record.episode_id} admits only the "
                "integrity handoff: the request must carry the episode id, the "
                "snapshot path and digest, and the snapshot bytes"
            )
        if body[0] != record.episode_id:
            raise AgentReconfigureRefusedError(
                f"The handoff episode {body[0]} is not this session's episode "
                f"{record.episode_id}; execution was not started"
            )

    def _continuation(
        self, params: SessionPlanHandoffParams, body: tuple[str, Path, bytes]
    ) -> bool:
        """Whether this request RE-ENTERS the commit of a claim already held."""
        episode_id, snapshot_path, _content = body
        try:
            sidecar = read_sidecar(snapshot_path.parent)
        except PlanEpisodeError:
            return False
        if sidecar is None:
            return False
        return (
            sidecar.episode_id == episode_id
            and sidecar.state == STATE_ACCEPTING
            and sidecar.claiming_session_id == params.session_id
            and sidecar.consent_sha256 is not None
        )

    def _plan_record(self) -> PlanModeRecord | None:
        """The root session's own plan record, or ``None`` when it has none."""
        metadata = self._loop.session_logger.session_metadata
        return None if metadata is None else metadata.plan_mode

    def _logging_domain(self) -> LoggingDomain:
        """The episode's PREPARE-side logging domain (the P1 owner domain)."""
        logger = self._loop.session_logger
        save_dir = logger.save_dir
        session_prefix = logger.session_prefix
        if not logger.enabled or save_dir is None or session_prefix is None:
            raise AgentReconfigureConflictError(LOGGING_REQUIRED_MESSAGE)
        return LoggingDomain(save_dir=str(save_dir), session_prefix=session_prefix)

    def _begin_commit(
        self, params: SessionPlanHandoffParams, body: tuple[str, Path, bytes]
    ) -> CommitAttempt:
        """COMMIT entry: acquire the fence and run C1, or refuse."""
        episode_id, snapshot_path, content = body
        domain = self._logging_domain()
        try:
            return begin_commit(
                snapshot_path.parent,
                episode_id=episode_id,
                session_id=params.session_id,
                owner_domain=domain,
                carried_bytes=content,
                snapshot_path=snapshot_path,
                retry_timeout=PLAN_COMMIT_FENCE_RETRY_SECONDS,
            )
        except PlanFenceBusyError as exc:
            raise AgentReconfigureConflictError(
                f"{exc} — an accept transaction is in progress; retry after it "
                "completes, or inspect the episode status"
            ) from exc
        except PlanDigestMismatchError as exc:
            detail = self._release_refused_claim(
                snapshot_path.parent, params.session_id, domain
            )
            raise AgentReconfigureConflictError(f"{exc}{detail}") from exc
        except PlanTransitionError as exc:
            raise AgentReconfigureConflictError(str(exc)) from exc
        except PlanEpisodeError as exc:
            raise AgentReconfigureFailedError(str(exc)) from exc

    def _release_refused_claim(
        self, directory: Path, session_id: str, domain: LoggingDomain
    ) -> str:
        """Release a refused C1 claim; report, never hide, a release failure."""
        try:
            release_claim(directory, session_id=session_id, owner_domain=domain)
        except PlanEpisodeError as exc:
            return (
                f" (the accept claim could not be released: {exc}; inspect the "
                "episode status before retrying)"
            )
        return " The accept claim was released and planning can continue."

    async def _retire_source(
        self, params: SessionPlanHandoffParams, commit: CommitAttempt
    ) -> PlanModeRecord:
        """C2: retire the planning session's record durably as accepted."""
        if params.session_id != self._loop.session_id:
            raise AgentReconfigureConflictError(
                "The plan record belongs to the root session"
            )
        logger = self._loop.session_logger
        metadata = logger.session_metadata
        source = None if metadata is None else metadata.plan_mode
        if source is None:
            raise AgentReconfigureConflictError(
                "The planning session records no plan mode to retire"
            )
        retired = source.model_copy(
            update={
                "episode_id": commit.episode_id,
                "state": STATE_ACCEPTED,
                "consent_sha256": commit.sidecar.consent_sha256,
                "snapshot_path": commit.snapshot_path.name,
                "attempt_generation": commit.attempt_generation,
                "scan_domain": _record_scan_domain(commit.sidecar.owner_domain),
            }
        )
        try:
            await logger.persist_plan_mode_durable(retired)
        except (PlanEpisodeWriteError, SessionDiskFullError) as exc:
            raise AgentReconfigureConflictError(
                "Plan handoff could not retire the planning record durably: "
                f"{exc}. The planning session is unchanged, but this accept "
                "attempt is committed — inspect the episode status before "
                "retrying or cancelling."
            ) from exc
        return retired

    async def _mark_and_inject(
        self, commit: CommitAttempt, retired: PlanModeRecord
    ) -> None:
        """C3: one logging observation, the domain record, the marker, injection."""
        logger = self._loop.session_logger
        observation: LoggingObservation = observe_logging_domain(
            save_dir=logger.save_dir,
            session_prefix=logger.session_prefix,
            session_dir=logger.session_dir,
        )
        try:
            commit.record_scan_domain(observation.domain)
            require_no_executor(
                observation.domain,
                commit.episode_id,
                excluding_session_id=commit.sidecar.claiming_session_id,
            )
        except PlanEpisodeError as exc:
            raise AgentReconfigureFailedError(str(exc)) from exc
        if logger.session_dir != observation.session_dir:
            raise AgentReconfigureFailedError(
                "The executor session dir moved away from the observed logging "
                "configuration; the episode marker was not written"
            )
        marker = retired.model_copy(
            update={
                "state": STATE_ACCEPTING,
                "episode_id": commit.episode_id,
                "consent_sha256": commit.sidecar.consent_sha256,
                "snapshot_path": commit.snapshot_path.name,
                "attempt_generation": commit.attempt_generation,
                "scan_domain": _record_scan_domain(observation.domain),
            }
        )
        try:
            await logger.persist_plan_mode_durable(marker)
        except (PlanEpisodeWriteError, SessionDiskFullError) as exc:
            raise AgentReconfigureFailedError(
                f"Plan handoff could not mark the executor session durably: "
                f"{exc}. The episode was not reported as accepted; inspect its "
                "status before retrying."
            ) from exc
        try:
            await self._loop.inject_user_context(
                build_handoff_context(
                    content=commit.carried_bytes,
                    episode_id=commit.episode_id,
                    snapshot_path=commit.snapshot_path,
                    consent_sha256=commit.sidecar.consent_sha256 or "",
                ),
                as_message=True,
            )
        except Exception as exc:
            raise AgentReconfigureFailedError(
                "Plan handoff could not inject the accepted plan into the "
                f"executor session: {exc}. The executor marker is durable, so "
                "the episode is possibly executed — inspect its status."
            ) from exc

    def _record_executor(self, commit: CommitAttempt, executor_session_id: str) -> None:
        """C4: record the rotated executor id while the episode is accepting."""
        try:
            commit.record_executor_session(executor_session_id)
        except PlanEpisodeError as exc:
            raise AgentReconfigureFailedError(
                f"Plan handoff could not record the executor session: {exc}. The "
                "executor marker is durable, so the episode is possibly executed "
                "— inspect its status."
            ) from exc

    def _finalize(self, commit: CommitAttempt, executor_session_id: str) -> None:
        """C5: the terminal flip once the executor exists, then the rename."""
        try:
            commit.finalize(executor_session_id=executor_session_id)
        except PlanEpisodeError as exc:
            raise AgentReconfigureFailedError(
                f"Plan handoff could not record the accepted episode: {exc}"
            ) from exc

    async def _restore_planner_key(
        self,
        orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        params: SessionPlanHandoffParams,
    ) -> list[BaseException]:
        """Restore the planner-alias thinking key to its pre-entry value.

        Per-key only: a key absent pre-entry (``planner_thinking_value=None``)
        is removed, never whole-map replaced.
        """
        if not params.planner_thinking_alias:
            return []
        if params.planner_thinking_value is None:
            return await remove_session_thinking_override(
                orchestrator,
                params.planner_thinking_alias,
                reason="plan handoff planner thinking key removal",
            )
        return await set_session_thinking_override(
            orchestrator,
            params.planner_thinking_alias,
            params.planner_thinking_value,
            reason="plan handoff planner thinking key restore",
        )

    def _selection_mismatch(
        self,
        orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        params: SessionPlanHandoffParams,
    ) -> dict[str, Any] | None:
        """Read back the installed selection; None when it matches the consent."""
        applied_model = override_active_model(orchestrator)
        applied_thinking = override_thinking_level(orchestrator, params.executor_model)
        bound_identity = self._loop._bound_identity
        if (
            applied_model == params.executor_model
            and applied_thinking == params.executor_thinking
            and bound_identity == params.executor_identity
            and override_agent_identity(orchestrator) is None
        ):
            return None
        return {
            "appliedModel": applied_model,
            "appliedThinking": applied_thinking,
            "boundIdentity": bound_identity,
        }

    async def _run_after_response(self, mode: str) -> str:
        """Run the post-response step inside the operation boundary.

        The in-process handlers answer with a callable the coordinator runs
        after the reply; the pipe carries no deferred action, so the work runs
        here, before the reply is written, and a failure becomes the receipt's
        ``followup`` instead of a promise the peer cannot keep.
        """
        if self._after_response is None:
            return ""
        try:
            outcome = await self._after_response(mode)
        except Exception as exc:
            return f"the post-response step failed ({type(exc).__name__}: {exc})"
        return outcome or ""

    def _cleared_with_followup(
        self, request: ReconfigureRequest, stages: list[str], detail: str
    ) -> ReconfigureReply:
        """The receipt for a clear whose history landed but a stage did not."""
        return ReconfigureReply(
            mode=request.mode,
            outcome=RECONFIGURE_OUTCOME_CLEARED_WITH_FOLLOWUP_FAILURE,
            stage=",".join(stages),
            detail=detail,
            snapshot=self._snapshot(),
        )

    def _require_session(self, session_id: str) -> None:
        if session_id != self._loop.session_id:
            raise AgentSessionNotFoundError(f"Session not found: {session_id}")


def _conversation_messages(loop: AgentLoop) -> list[LLMMessage]:
    """The non-system messages the live loop still carries."""
    return [message for message in loop.messages if message.role is not Role.system]


def _record_scan_domain(domain: LoggingDomain | None) -> PlanScanDomain | None:
    """The persisted record's scan-domain mirror of a logging domain."""
    if domain is None:
        return None
    return PlanScanDomain(
        save_dir=domain.save_dir, session_prefix=domain.session_prefix
    )


def _handoff_verified_body(
    params: SessionPlanHandoffParams,
) -> tuple[str, Path, bytes] | None:
    """Decode and validate an episode-bound handoff's consent body.

    The episode identity, snapshot path, digest, and bytes travel together or
    not at all. The bytes are the authority: their digest must equal the
    carried digest, and the snapshot path must be one of the episode's own names
    in the episode's own directory. ``None`` means the legacy request.
    """
    episode_id = params.episode_id
    snapshot_path = params.snapshot_path
    snapshot_sha256 = params.snapshot_sha256
    encoded = params.snapshot_bytes_b64
    if (
        episode_id is None
        and snapshot_path is None
        and snapshot_sha256 is None
        and encoded is None
    ):
        return None
    if (
        episode_id is None
        or snapshot_path is None
        or snapshot_sha256 is None
        or encoded is None
    ):
        raise AgentReconfigureRefusedError(
            "An episode handoff carries its episode id, snapshot path, "
            "snapshot digest, and snapshot bytes together"
        )
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AgentReconfigureRefusedError(
            "The plan handoff snapshot bytes are not valid base64"
        ) from exc
    if sha256_bytes(content) != snapshot_sha256:
        raise AgentReconfigureRefusedError(
            "The plan handoff snapshot digest does not match its bytes"
        )
    path = Path(snapshot_path)
    if not _is_episode_snapshot_name(path.name, episode_id):
        raise AgentReconfigureRefusedError(
            f"The plan handoff snapshot path must be the episode's "
            f"accepted-{episode_id}.md, or its surviving "
            f"provenance-{episode_id}-<generation>.md"
        )
    return episode_id, path, content


def _is_episode_snapshot_name(name: str, episode_id: str) -> bool:
    """True for either of the episode's own snapshot names."""
    if name == f"accepted-{episode_id}.md":
        return True
    prefix = f"provenance-{episode_id}-"
    if not name.startswith(prefix) or not name.endswith(".md"):
        return False
    generation = name[len(prefix) : len(name) - len(".md")]
    return generation.isdigit() and int(generation) >= 1


def _draft_refusal(content: bytes) -> str | None:
    """The refusal for consented bytes that are not a usable plan document."""
    recovery = (
        " Bind another document (re-bind the draft and re-consent) or cancel plan mode."
    )
    if not content.strip():
        return f"The plan document is empty; execution was not started.{recovery}"
    summary = extract_plan_summary(decode_safe(content).text)
    if summary is None:
        return (
            "The plan document has no ## Summary section; execution was not "
            f"started.{recovery}"
        )
    if not summary:
        return (
            "The plan document's ## Summary section is empty; execution was not "
            f"started.{recovery}"
        )
    return None


__all__ = ["AfterResponse", "AgentReconfigureService"]
