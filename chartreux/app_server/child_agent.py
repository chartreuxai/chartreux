"""Child agent process: one subagent loop served over its own standard pipes.

The coordinator spawns this module for each subagent (WP5 [5A]). The child
announces readiness with the WP1 bootstrap frame, opens through the
``subagent/open`` frame — which carries the spawn spec — and then serves the
same agent-wire turn protocol as the root agent process: turn events stream
back in production order and the terminal ``agent/turn/ended`` frame closes
the run, which the coordinator's completion source surfaces as its final
frame.

A child process is always fresh, never resumed as a session
(design.md:176-180): its prior transcripts persist as records under the root
session and are reconciled by the coordinator registry ([5C]). Child-owned
transcripts under ``<root>/agents/*`` and the propagated policy tree land with
[5C]/[5D]; this entrypoint only serves the wire.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import aclosing, suppress
from dataclasses import replace
from datetime import UTC, datetime
import itertools
from pathlib import Path
import sys
from typing import Any, cast
from uuid import uuid4

from chartreux.app_server._agent_transcript import read_live_agent_transcript
from chartreux.app_server._subagent_transcript import (
    AGENT_CONFIG_PAYLOAD_KEY,
    AGENT_NAME_PAYLOAD_KEY,
    AUTHORITY_SNAPSHOT_PAYLOAD_KEY,
    INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY,
    INHERITED_RESTRICTIONS_PAYLOAD_KEY,
    INHERITED_ROOT_GRANTS_PAYLOAD_KEY,
    INHERITED_WORKSPACE_PAYLOAD_KEY,
    ChildResultRecord,
    ChildStartMarker,
    ChildTranscriptError,
    ChildTranscriptStore,
    child_result_digest,
    child_transcript_logging,
    child_transcript_root_for,
    decode_inherited_root_grants,
    decode_inherited_workspace,
    decode_source_restrictions,
    require_child_policy_payload_keys,
    root_session_dir_from_spec,
)
from chartreux.app_server.agent_wire import (
    AGENT_AUTHORITY_ALLOW,
    AGENT_AUTHORITY_CHANGED_METHOD,
    AGENT_AUTHORITY_CHECK_METHOD,
    AGENT_AUTHORITY_DECISION_KIND_TOOL_AVAILABILITY,
    AGENT_AUTHORITY_DECISION_KIND_TOOL_INVOCATION,
    AGENT_CHILD_TEARDOWN_METHOD,
    AGENT_TURN_ENDED_METHOD,
    AGENT_TURN_EVENT_METHOD,
    AUTHORITY_REVISION_COMPONENTS,
    CHILD_RUN_TURNS_FRAME_KEY,
    PARENT_INCARNATION_PAYLOAD_KEY,
    SUBAGENT_OPEN_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
    TURN_OUTCOME_INTERRUPTED,
    AgentAuthorityCheck,
    AgentWireError,
    UnsupportedAgentRequestError,
    agent_request_id,
    decode_authority_changed,
    decode_authority_decision,
    decode_parent_incarnation,
    decode_turn_request,
    encode_authority_check,
    encode_event,
    encode_transcript_page,
    implementation_identity,
)
from chartreux.app_server.process_bootstrap import (
    bootstrap_requested,
    write_ready_frame,
)
from chartreux.app_server.root_agent import RootAgentServer, RootLoopFactory
from chartreux.app_server.subagent_ports import (
    CHILD_FAILURE_FRAME_KEY,
    CHILD_HANDOFF_PAYLOAD_KEY,
    ChildFailureEvidence,
    child_failure_evidence,
    encode_child_failure,
    spawn_spec_from_wire,
)
from chartreux.app_server.transport import (
    BinaryLineReader,
    BinaryLineWriter,
    JsonRpcTransport,
    StdioJsonRpcTransport,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.paths import LITERAL_LOG_FILE, LOG_FILE
from chartreux.observability.logging import init_file_logging, logger

CHILD_AGENT_MODULE = "chartreux.app_server.child_agent"

CHILD_TURNS_USED_FRAME_KEY = "turns_used"
"""The child's own logical turn count on its terminal ``agent/turn/ended`` frame.

The value numbers the assistant messages the run committed, the same measure
the in-process route derives from its loop, and never the events the run
relayed: a streaming loop emits one ``AssistantEvent`` per chunk and a
tool-only step emits none, so the relayed frames cannot count turns.
"""

CHILD_LIVE_TURNS_FRAME_KEY = CHILD_RUN_TURNS_FRAME_KEY
"""The same count, carried ahead of the terminal frame on each relayed event (H1).

A running child's row has no turn count until its run ends, because the
terminal frame is the only place the final number travels. This key carries
the run's progress so far beside every event the run relays: the owner reads
it off the decoded event and projects it, and the terminal frame still settles
the final value. The key is the wire codec's own (``agent_wire``), so the
encoder that stamps it and the decoder that surfaces it agree by construction;
the measure is the terminal carrier's -- committed assistant messages, never
relayed frames.
"""

DEFAULT_CHILD_AUTHORITY_TIMEOUT = 10.0
"""How long one authority decision may take before the child denies (G5b).

The ask is one round trip over the child's own pipe to the process that
spawned it, so the bound is generous for a busy owner and still short enough
that a wedged owner cannot hang a tool call. A timeout is a DENY: a decision is
never manufactured out of the absence of an answer.
"""


class ChildAuthorityUnavailable(Exception):
    """The owner could not be asked at all (G5b): a typed, fail-closed signal.

    Raised for a transport failure or a bounded timeout. A decision that
    arrived but is unusable -- foreign, late, reordered, undecodable -- is NOT
    this class: it is a refusal for the call it answers, so the pass that was
    asking can still finish. This class aborts a catalog pass instead, because
    every later question would hit the same dead pipe.
    """


class ChildAuthorityClient:
    """Ask this child's immediate owner for one live decision per question (G5b).

    A process child's policy is not its own: F11 hands it the parent's policy
    as it stood at spawn, while the owner keeps deciding from its CURRENT
    policy. This client replaces every spawn-time decision that matters -- tool
    invocation admission, the catalog axis, and therefore launch admission for
    a nested ``task`` call -- with a typed question to the process that
    spawned it, answered allow/deny only. The child never receives policy it
    could execute on its own, and it never caches an allow into an execution:
    every invocation asks again, and the execution gate asks again after the
    file-snapshot await so a revocation landing inside that window wins.

    Fencing: every question names this child's run generation and the fresh
    random epoch minted for it, the parent session and incarnation it believes
    it is talking to, its last-known owner revision vector, and an
    agent-origin request id that only ever advances. A reply must echo all of
    them back, carry a revision vector that is componentwise non-decreasing,
    and decode as an allow/deny; anything else is a refusal. No wall-clock
    value takes part in any of it.

    An UNBOUND client -- one whose spawn payload carried no parent incarnation,
    i.e. an owner older than this carrier -- denies every decision and withholds
    the whole catalog. That is the fail-closed answer: such a child runs no
    tools rather than running them ungated.
    """

    def __init__(
        self,
        call_host: Any,
        *,
        agent_id: str,
        session_id: str,
        parent_incarnation: str | None,
        timeout: float = DEFAULT_CHILD_AUTHORITY_TIMEOUT,
    ) -> None:
        self._call_host = call_host
        self._agent_id = agent_id
        self._session_id = session_id
        self._parent_incarnation = parent_incarnation
        self._timeout = timeout
        self._requests = itertools.count(1)
        self._generation = 0
        self._epoch = uuid4().hex
        self._revision: dict[str, int] = {
            component: 0 for component in sorted(AUTHORITY_REVISION_COMPONENTS)
        }
        # This end's questions are serialized: the owner fences each child's
        # request sequence, so two questions must never overtake each other on
        # the pipe. The lock is the CLIENT's own and is held across its own
        # wire await; no manager, loop or store lock is ever held across it.
        self._lock = asyncio.Lock()
        self._catalog: dict[str, bool] = {}
        self._catalog_stale = False
        # Bumped wherever the cache is dropped; a catalog pass publishes only
        # when this epoch survived the whole pass, so a mid-pass invalidation
        # is never republished over by the pass it interrupted.
        self._catalog_epoch = 0
        self._asks = 0
        self._refusals = 0
        self._changes = 0

    @property
    def bound(self) -> bool:
        """Whether this child has an owner incarnation it can ask."""
        return self._parent_incarnation is not None

    @property
    def generation(self) -> int:
        """The run generation this client's questions are fenced with."""
        return self._generation

    @property
    def revision(self) -> dict[str, int]:
        """The last owner revision vector this child accepted."""
        return dict(self._revision)

    @property
    def ask_count(self) -> int:
        return self._asks

    @property
    def refusal_count(self) -> int:
        return self._refusals

    @property
    def change_count(self) -> int:
        """How many owner invalidations reached this child."""
        return self._changes

    @property
    def catalog_stale(self) -> bool:
        return self._catalog_stale

    async def begin_run(self) -> None:
        """Start one run: a fresh generation and a fresh epoch for it.

        A retained child serves several runs, and the owner's fence treats a
        new generation as a fresh incarnation: the epoch is minted here and is
        never reused, so an earlier run's question can never be replayed into a
        later one.
        """
        self._generation += 1
        self._epoch = uuid4().hex

    async def denial(
        self, *, tool_name: str, tool_class: type[Any], arguments: Mapping[str, Any]
    ) -> str | None:
        """The refusal for one invocation under the owner's current policy.

        ``None`` means the owner allowed this exact invocation; every other
        outcome -- unbound, unavailable, unreadable, foreign, reordered, stale
        or a plain deny -- is a refusal string the caller turns into a skip.
        """
        return await self._decide(
            AGENT_AUTHORITY_DECISION_KIND_TOOL_INVOCATION,
            tool_name,
            tool_class,
            dict(arguments),
        )

    def catalog_withholds(self, tool_name: str) -> bool:
        """Whether this child's catalog withholds *tool_name* right now.

        Read from the last catalog pass, never from a notification: a name the
        pass did not see allowed -- including every name while the cache is
        empty, as it is right after an invalidation -- is withheld.
        """
        if not self.bound:
            return True
        return self._catalog.get(tool_name) is not True

    async def prime_catalog(self, tools: Mapping[str, type[Any]]) -> None:
        """Ask the owner one catalog question per name and cache the verdicts.

        Every name is asked under its own implementation identity, so the owner
        compares the class this child would run, never a name alone. A refusal
        withholds that name; an unavailable owner aborts the pass with the
        verdicts gathered so far, and every unasked name stays withheld. A
        pass interrupted by an invalidation publishes nothing at all: the
        dropped cache belongs to the already-scheduled refresh.
        """
        if not self.bound:
            self._catalog = {}
            self._catalog_stale = False
            return
        epoch = self._catalog_epoch
        verdicts: dict[str, bool] = {}
        try:
            for name in sorted(tools):
                refusal = await self._decide(
                    AGENT_AUTHORITY_DECISION_KIND_TOOL_AVAILABILITY,
                    name,
                    tools[name],
                    {},
                )
                verdicts[name] = refusal is None
        finally:
            if self._catalog_epoch == epoch:
                self._catalog = verdicts
                self._catalog_stale = False

    def receive_changed(self, changed: Any) -> None:
        """Apply one owner invalidation: the catalog cache is dropped.

        A foreign session, a superseded owner incarnation and a rolled-back
        vector are all ignored, so an old or spoofed frame can never widen this
        child's catalog. A moved vector DROPS the cache rather than rewriting
        it -- an invalidation carries no policy, so the next assembly asks
        again. The execution gate does not depend on this at all: it asks on
        every invocation, so a dropped invalidation costs a stale catalog, never
        a stale allow.
        """
        self._changes += 1
        if not self.bound:
            return
        if changed.session_id != self._session_id:
            return
        if changed.incarnation != self._parent_incarnation:
            return
        if any(
            value < self._revision.get(component, 0)
            for component, value in changed.revision.items()
        ):
            return
        self._fold_revision(changed.revision)
        self._catalog = {}
        self._catalog_stale = True
        self._catalog_epoch += 1

    def invalidate_catalog(self) -> None:
        """Drop the catalog cache after a pass that could not be completed."""
        self._catalog = {}
        self._catalog_stale = True
        self._catalog_epoch += 1

    async def _decide(
        self,
        decision_kind: str,
        tool_name: str,
        tool_class: type[Any],
        arguments: dict[str, Any],
    ) -> str | None:
        """Ask one question and answer the refusal, or ``None`` for an allow."""
        if not self.bound:
            self._refusals += 1
            return (
                f"{tool_name!r} denied: this child was published without a "
                "live authority owner, so no decision can be asked for"
            )
        implementation = implementation_identity(tool_class)
        async with self._lock:
            request_id = agent_request_id(next(self._requests))
            check = AgentAuthorityCheck(
                decision_kind=decision_kind,
                tool_name=tool_name,
                implementation=implementation,
                arguments=arguments,
                child_agent_id=self._agent_id,
                child_generation=self._generation,
                epoch=self._epoch,
                parent_session_id=self._session_id,
                parent_incarnation=self._parent_incarnation or "",
                request_id=request_id,
                revision=dict(self._revision),
            )
            self._asks += 1
            try:
                reply = await asyncio.wait_for(
                    self._call_host(
                        AGENT_AUTHORITY_CHECK_METHOD, encode_authority_check(check)
                    ),
                    timeout=self._timeout,
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                raise ChildAuthorityUnavailable(
                    f"the owner could not be asked: {type(exc).__name__}: {exc}"
                ) from exc
            try:
                decision = decode_authority_decision(reply)
            except AgentWireError as exc:
                return self._refuse(
                    tool_name, f"the owner answered an unreadable decision ({exc})"
                )
            refusal = self._fence_reply(
                decision,
                request_id=request_id,
                generation=self._generation,
                epoch=self._epoch,
            )
            if refusal is not None:
                return self._refuse(tool_name, refusal)
            rolled_back = self._fold_revision(decision.revision)
            if rolled_back is not None:
                return self._refuse(tool_name, rolled_back)
            if decision.decision != AGENT_AUTHORITY_ALLOW:
                return self._refuse(
                    tool_name,
                    decision.reason or "the owner's live policy denies this call",
                )
            return None

    def _fence_reply(
        self, decision: Any, *, request_id: str, generation: int, epoch: str
    ) -> str | None:
        """The refusal for a reply minted for another question (or none)."""
        if decision.request_id != request_id:
            return "the owner answered a different request"
        if decision.epoch != epoch:
            return "the owner answered a superseded child epoch"
        if decision.child_generation != generation:
            return "the owner answered a superseded child generation"
        if decision.parent_session_id != self._session_id:
            return "the owner answered for another session"
        if decision.parent_incarnation != self._parent_incarnation:
            return "the owner answered for another owner incarnation"
        return None

    def _fold_revision(self, revision: Mapping[str, int]) -> str | None:
        """Fold one accepted vector in; a componentwise decrease is a refusal."""
        rolled_back = [
            component
            for component, value in revision.items()
            if value < self._revision.get(component, 0)
        ]
        if rolled_back:
            return (
                "the owner answered a rolled-back authority revision "
                f"({', '.join(sorted(rolled_back))})"
            )
        self._revision = {
            component: max(value, self._revision.get(component, 0))
            for component, value in revision.items()
        }
        return None

    def _refuse(self, tool_name: str, reason: str) -> str:
        self._refusals += 1
        logger.warning("Live authority refused %s: %s", tool_name, reason)
        return f"Tool '{tool_name}' denied: {reason}"


class ChildAgentServer(RootAgentServer):
    """Serve one child agent loop; ``subagent/open`` carries the spawn spec.

    The child OWNS its transcript ([5C]): its loop's SessionLogger writes
    ``<root>/agents/<agent>_<ts>_<id>``, and this server writes the durable
    start marker before a run executes and the terminal result record (result
    id + digest) save-before-emit. The coordinator never writes these records;
    it reads them to adjudicate a crash.
    """

    def __init__(
        self, transport: JsonRpcTransport, *, factory: RootLoopFactory
    ) -> None:
        super().__init__(transport, factory=factory)
        self._agent_id: str | None = None
        self._attempt_id: str | None = None
        self._root_session_dir: Path | None = None
        self._transcript_store: ChildTranscriptStore | None = None
        # G5b: the live-authority client this child asks its owner through. It
        # is installed at open and attached to the loop's tool manager, which
        # is what makes every tool invocation and the catalog ask again.
        self._authority_client: ChildAuthorityClient | None = None

    async def _handle_request(
        self, method: str, params: Mapping[str, Any], request_id: object = None
    ) -> dict[str, Any]:
        if method == SUBAGENT_OPEN_METHOD:
            if self._loop is not None:
                raise UnsupportedAgentRequestError("the child agent is already open")
            # Validate the spec before building the loop: a malformed open
            # frame fails typed instead of opening a nameless child.
            spec = spawn_spec_from_wire(params)
            self._loop = await self._factory.open(params)
            self._agent_id = spec.agent_id
            # The publication attempt this child was spawned under rides every
            # durable record it writes, so a lost aggregate reply stays
            # attributable to its attempt (item 1 child seam).
            self._attempt_id = spec.attempt_id
            session_dir = self._loop.session_logger.session_dir
            self._root_session_dir = child_transcript_root_for(
                session_dir, root_session_dir_from_spec(spec.payload)
            )
            self._transcript_store = self._open_transcript_store()
            self._install_authority_client(spec)
            return {
                "agent_id": spec.agent_id,
                "session_id": self._loop.session_id,
                "session_dir": str(session_dir) if session_dir is not None else None,
            }
        if method == AGENT_CHILD_TEARDOWN_METHOD:
            # A child agent owns no publication attempts: its own attempt is
            # settled by the process that spawned it, never by its own pipe.
            raise UnsupportedAgentRequestError(
                "this agent owns no child publication attempts to tear down"
            )
        return await super()._handle_request(method, params, request_id)

    def _install_authority_client(self, spec: Any) -> None:
        """Install the live-authority client this child asks its owner through.

        The parent incarnation rides the spawn payload under its exact key. A
        value that is present but unusable refuses the open typed; an ABSENT
        key is an owner older than this carrier, and the client is then left
        unbound -- every decision denies and the whole catalog is withheld, so
        such a child runs no tools rather than running them ungated.
        """
        payload = spec.payload
        parent_incarnation: str | None = None
        if PARENT_INCARNATION_PAYLOAD_KEY in payload:
            parent_incarnation = decode_parent_incarnation(payload)
        loop = self._loop
        if loop is None:
            return
        client = ChildAuthorityClient(
            self.call_host,
            agent_id=self._agent_id or spec.agent_id,
            session_id=spec.parent_session_id,
            parent_incarnation=parent_incarnation,
        )
        self._authority_client = client
        attach = getattr(loop, "attach_live_authority", None)
        if attach is not None:
            attach(client)
            return
        manager = getattr(loop, "tool_manager", None)
        manager_attach = getattr(manager, "attach_live_authority", None)
        if manager_attach is not None:
            manager_attach(client)

    async def _begin_authority_run(self) -> None:
        """Start a run's authority lineage and refresh the advertised catalog.

        The generation rotates per run (a retained child serves several), and
        the catalog pass runs BEFORE the model sees any tool definition, so a
        withheld name is never advertised as invocable. An owner that cannot be
        asked leaves the cache empty: the child then advertises nothing rather
        than advertising what it cannot have re-authorized.
        """
        client = self._authority_client
        if client is None:
            return
        await client.begin_run()
        await self._refresh_authority_catalog()

    async def _refresh_authority_catalog(self) -> None:
        """Ask the owner for the catalog verdict on every candidate tool name."""
        client = self._authority_client
        loop = self._loop
        if client is None or loop is None:
            return
        manager = getattr(loop, "tool_manager", None)
        tools = dict(getattr(manager, "available_tools", {}) or {})
        try:
            await client.prime_catalog(tools)
        except asyncio.CancelledError:
            raise
        except ChildAuthorityUnavailable as exc:
            # Every later question would hit the same dead pipe: the pass stops
            # with its partial verdicts and every unasked name stays withheld.
            logger.warning("Child authority catalog pass aborted: %s", exc)
            client.invalidate_catalog()
        except BaseException as exc:
            logger.warning("Child authority catalog pass failed: %s", exc)
            client.invalidate_catalog()

    def _receive_authority_changed(self, params: Mapping[str, Any]) -> None:
        """Apply one owner invalidation and re-ask the catalog (G5b)."""
        client = self._authority_client
        if client is None:
            self._record_diagnostic(
                "this agent holds no live child-authority client",
                None,
                method=AGENT_AUTHORITY_CHANGED_METHOD,
            )
            return
        try:
            changed = decode_authority_changed(params)
        except AgentWireError as exc:
            self._record_diagnostic(
                f"an authority-change frame could not be decoded: {exc}",
                None,
                method=AGENT_AUTHORITY_CHANGED_METHOD,
            )
            return
        client.receive_changed(changed)
        if client.catalog_stale:
            self._track(self._refresh_authority_catalog())

    def _open_transcript_store(self) -> ChildTranscriptStore | None:
        """The child-owned marker store for this transcript dir, when usable."""
        loop = self._loop
        root = self._root_session_dir
        if loop is None or root is None:
            return None
        session_dir = loop.session_logger.session_dir
        if session_dir is None:
            return None
        try:
            return ChildTranscriptStore(session_dir, root_session_dir=root)
        except ChildTranscriptError as exc:
            logger.warning("Child transcript markers disabled: %s", exc)
            return None

    async def _run_turn(self, turn_id: str, params: Mapping[str, Any]) -> None:
        """Run one turn with the child's durable start and terminal records."""
        await self._begin_authority_run()
        self._record_start(turn_id)
        # The run's turn count is a difference: a retained child serves several
        # runs, so only the messages THIS run commits belong to its count.
        start_turns = self._committed_turn_count()
        outcome = TURN_OUTCOME_COMPLETED
        error: str | None = None
        failure: ChildFailureEvidence | None = None
        try:
            loop = self._require_loop()
            request = decode_turn_request(params)
            runner, tool_io = self._host_ports_for(params)
            options = request.turn_options or AgentTurnOptions()
            request = replace(
                request,
                subagent_runner=runner,
                tool_io=tool_io,
                turn_options=replace(
                    options, turn_id=turn_id, retry_sink=self._retry_sink(turn_id)
                ),
            )
            async with aclosing(loop.execution_port.run_turn(request)) as events:
                async for event in events:
                    await self._notify(
                        AGENT_TURN_EVENT_METHOD,
                        {
                            "turn_id": turn_id,
                            "event": encode_event(
                                event,
                                live_turns_used=self._run_turn_progress(start_turns),
                            ),
                        },
                    )
        except asyncio.CancelledError:
            outcome = TURN_OUTCOME_INTERRUPTED
        except BaseException as exc:
            outcome = TURN_OUTCOME_FAILED
            error = f"{type(exc).__name__}: {exc}"
            failure = self._failure_evidence(turn_id, exc)
        finally:
            self._active_turn_id = None
            # Save-before-emit: the terminal result record is durable before
            # the completion frame is announced, so a crash right after this
            # point reconciles as COMPLETED, not as an unknown outcome. The
            # failure block is part of the record: the measured evidence stays
            # on disk even when the completion frame is lost with the pipe.
            failure_block = (
                encode_child_failure(failure) if failure is not None else None
            )
            self._record_result(
                turn_id, outcome=outcome, error=error, failure=failure_block
            )
            ended: dict[str, Any] = {
                "turn_id": turn_id,
                "outcome": outcome,
                "error": error,
            }
            if failure_block is not None:
                ended[CHILD_FAILURE_FRAME_KEY] = failure_block
            turns_used = self._run_turn_progress(start_turns)
            if turns_used is not None:
                ended[CHILD_TURNS_USED_FRAME_KEY] = turns_used
            with suppress(Exception):
                await self._notify(AGENT_TURN_ENDED_METHOD, ended)

    def _run_turn_progress(self, start_turns: int | None) -> int | None:
        """The assistant messages THIS run has committed, as a running count (H1).

        The same measure the terminal carrier reports, taken mid-run for the
        row a live child shows: a retained child serves several runs, so the
        count is the difference from the run's own start and never the loop's
        total. Only the child can count it -- the events it relays are one per
        streamed chunk and none for a tool-only step -- so the count is stamped
        beside each relayed event. ``None`` when no run-start baseline exists,
        or when the loop reports fewer messages than it started with: an
        unobserved anomaly stays unknown rather than becoming a negative count.
        """
        if start_turns is None:
            return None
        current = self._committed_turn_count()
        if current is None or current < start_turns:
            return None
        return current - start_turns

    def _committed_turn_count(self) -> int | None:
        """The assistant messages this child's loop has committed so far.

        The child's own count, read where the messages live; ``None`` when the
        loop never opened, so an unopenable child reports no count at all
        instead of a fabricated zero.
        """
        loop = self._loop
        if loop is None:
            return None
        return sum(
            message.role is Role.assistant
            for message in cast("list[LLMMessage]", loop.messages)
        )

    def _failure_evidence(
        self, turn_id: str, exc: BaseException
    ) -> ChildFailureEvidence | None:
        """Derive the run's typed envelope here, where its evidence still lives.

        Total by contract: a derivation failure is logged and reported as
        ``None`` (no envelope), never masking the run's primary failure.
        """
        loop = self._loop
        if loop is None:
            return None
        try:
            return child_failure_evidence(
                loop,
                exc,
                agent_id=self._agent_id or loop.session_id,
                run_id=turn_id,
                turn_id=turn_id,
            )
        except Exception as failure_exc:
            logger.warning(
                "Failed to derive the child failure envelope", exc_info=failure_exc
            )
            return None

    async def _read_page(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Serve a live page; the child answers to its agent id and session id."""
        loop = self._require_loop()
        agent_id = params.get("agent_id")
        if agent_id not in {loop.session_id, self._agent_id}:
            raise UnsupportedAgentRequestError("unknown agent")
        response = await asyncio.to_thread(
            read_live_agent_transcript,
            cast("list[LLMMessage]", loop.messages),
            before=params.get("before"),
            limit=int(params.get("limit") or 50),
        )
        return encode_transcript_page(response)

    def _record_start(self, turn_id: str) -> None:
        store = self._transcript_store
        loop = self._loop
        if store is None or loop is None:
            return
        try:
            store.record_start(
                ChildStartMarker(
                    agent_id=self._agent_id or loop.session_id,
                    session_id=loop.session_id,
                    run_id=turn_id,
                    started_at=datetime.now(UTC).isoformat(),
                    # The run-time marker keeps its shipped semantics (written
                    # immediately before the run, never at publication) and
                    # carries the attempt id when the spawn named one, so the
                    # marker bridges to the attempt that created this child.
                    attempt_id=self._attempt_id,
                )
            )
        except ChildTranscriptError as exc:
            logger.warning("Failed to record the child start marker: %s", exc)

    def _record_result(
        self,
        turn_id: str,
        *,
        outcome: str,
        error: str | None,
        failure: dict[str, Any] | None = None,
    ) -> None:
        store = self._transcript_store
        loop = self._loop
        if store is None or loop is None:
            return
        agent_id = self._agent_id or loop.session_id
        try:
            store.record_result(
                ChildResultRecord(
                    agent_id=agent_id,
                    session_id=loop.session_id,
                    run_id=turn_id,
                    result_id=f"{agent_id}:{turn_id}",
                    digest=child_result_digest(
                        agent_id=agent_id,
                        session_id=loop.session_id,
                        run_id=turn_id,
                        outcome=outcome,
                        error=error,
                    ),
                    outcome=outcome,
                    completed_at=datetime.now(UTC).isoformat(),
                    error=error,
                    failure=failure,
                )
            )
        except ChildTranscriptError as exc:
            logger.warning("Failed to record the child terminal result: %s", exc)


class HarnessChildLoopFactory:
    """The production factory: one child process owns one child loop."""

    def __init__(self) -> None:
        from chartreux.app_server._runtime import HarnessProcess

        self._process = HarnessProcess()

    async def open(self, params: Mapping[str, Any]) -> AgentLoop:
        from chartreux.app_server._connection_protocol import (
            ClientCapabilities,
            ClientInfo,
        )
        from chartreux.app_server._runtime import RootOpenRequest
        from chartreux.app_server.protocol import AgentConfig

        try:
            spec = spawn_spec_from_wire(params)
            payload = spec.payload
            root_session_dir = root_session_dir_from_spec(payload)
            if root_session_dir is None:
                # Fail closed: a child's transcript belongs under the root
                # session's agents/ subtree. Falling back to the configured
                # save dir would put it outside the root and break
                # reconciliation.
                raise UnsupportedAgentRequestError(
                    "a child spawn spec carries no root session directory"
                )
            options = AgentConfig.model_validate(
                payload.get(AGENT_CONFIG_PAYLOAD_KEY) or {}
            )
            client_info = ClientInfo.model_validate(payload.get("client_info") or {})
            capabilities = ClientCapabilities.model_validate(
                payload.get("client_capabilities") or {}
            )
            raw_name = payload.get(AGENT_NAME_PAYLOAD_KEY)
            agent_name = (
                raw_name
                if isinstance(raw_name, str) and raw_name
                else spec.agent_id.replace(":", "-")
            )
            # A child loop is constructed as a subagent, so its open is only
            # legal with the parent workspace: no fallback decode here -- a
            # payload without the key keeps inherited_workspace None and the
            # loop construction answers the loud typed failure an old-version
            # peer deserves.
            workspace_payload = payload.get(INHERITED_WORKSPACE_PAYLOAD_KEY)
            inherited_workspace = (
                None
                if workspace_payload is None
                else decode_inherited_workspace(workspace_payload)
            )
            # F11: the parent's accepted policy is equally load-bearing -- a
            # payload a pre-F11 peer sends (no policy keys) never opens the
            # child with an empty policy: it fails loudly here instead, so a
            # restricted or budget-capped parent cannot spawn an unrestricted,
            # unbudgeted child.
            require_child_policy_payload_keys(payload)
            inherited_restrictions = decode_source_restrictions(
                payload[INHERITED_RESTRICTIONS_PAYLOAD_KEY]
            )
            inherited_mode_restrictions = decode_source_restrictions(
                payload[INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY]
            )
            inherited_root_grants = decode_inherited_root_grants(
                payload[INHERITED_ROOT_GRANTS_PAYLOAD_KEY]
            )
            # v0.8.0 A-P0: the parent's safety-authority launch snapshot rides
            # the spawn payload the same way. A legacy payload without the key
            # decodes to None, and the child falls back to the hard-coded SAFE
            # initial level (initial_level(config)) — never a silent elevation.
            loop = await self._process.open_root(
                RootOpenRequest(
                    options=options,
                    client_info=client_info,
                    client_capabilities=capabilities,
                    # The child's own transcript dir, under the root session's
                    # agents/ subtree: the child owns the only writer.
                    session_logging=child_transcript_logging(
                        root_session_dir, agent_name
                    ),
                    is_subagent=True,
                    parent_session_id=spec.parent_session_id,
                    inherited_workspace=inherited_workspace,
                    inherited_restrictions=inherited_restrictions,
                    inherited_mode_restrictions=inherited_mode_restrictions,
                    inherited_root_grants=inherited_root_grants,
                    inherited_authority=payload.get(AUTHORITY_SNAPSHOT_PAYLOAD_KEY),
                )
            )
        except asyncio.CancelledError:
            # Cancellation is control flow, not an open failure: the
            # traceback log is noise the rotating file does not need.
            raise
        except BaseException as exc:
            # The wire error this open failure answers carries the typed
            # detail but no traceback, so the child logs the full failure
            # here -- a malformed payload or a workspace decode error
            # included, which run before the loop opens -- before its
            # dispatch replies over the pipe.
            logger.exception("Child agent loop open failed: %s", exc)
            raise
        # Publication-time bridge: the attempt id lands in the child's FIRST
        # durable write, so a torn persist-to-link window (or a lost aggregate
        # reply) is attributable to the attempt that produced it.
        await loop.persist_empty_session(attempt_id=spec.attempt_id)
        handoff = payload.get(CHILD_HANDOFF_PAYLOAD_KEY)
        if isinstance(handoff, str) and handoff:
            # Spawn handoff: the parent's working set arrives ahead of the
            # task brief, redacted on the parent's side like every message.
            loop.messages.append(
                LLMMessage(role=Role.user, content=handoff, injected=True)
            )
        return loop

    async def close(self) -> None:
        await self._process.close()


async def serve_child_agent(
    *,
    factory: RootLoopFactory,
    reader: BinaryLineReader | None = None,
    writer: BinaryLineWriter | None = None,
    announce_ready: bool = False,
) -> None:
    """Serve a child agent over the given pipes, announcing readiness first."""
    if reader is None or writer is None:
        transport: JsonRpcTransport = StdioJsonRpcTransport.from_standard_streams()
        ready_writer: BinaryLineWriter = sys.stdout.buffer
    else:
        transport = StdioJsonRpcTransport(reader, writer)
        ready_writer = writer
    if announce_ready:
        write_ready_frame(ready_writer)
    try:
        await ChildAgentServer(transport, factory=factory).serve()
    finally:
        with suppress(Exception):
            await transport.close()


def main() -> None:
    from chartreux.core.config import load_dotenv_values

    init_harness_files_manager("user", "project")
    init_file_logging(LOG_FILE.path, repair_path=LITERAL_LOG_FILE.path)
    load_dotenv_values()
    logger.info("Child agent process starting")
    asyncio.run(
        serve_child_agent(
            factory=HarnessChildLoopFactory(), announce_ready=bootstrap_requested()
        )
    )


if __name__ == "__main__":
    main()
