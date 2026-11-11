"""Root-side permissions, approval, expiry, and cancellation differential oracle (WP6 [6B]).

The oracle is a *component*, not a trace: it hosts the production decision
surfaces behind the [6A] :class:`~tests.differential.replay.Channel` contract,
so the same function runs with the process boundary removed and inside the
spawned replay child. Every decision it records comes from production code:

* permission checks run through :func:`resolve_file_tool_permission` over a
  :class:`Workspace` built from the session grant store, which is the authority
  the loop rebuilds after an approved grant;
* grants, deny memory, and answer parsing come from
  :mod:`chartreux.core.agent_loop._root_grants`;
* approval/open-request admission, arbitration, unattended expiry, and
  settlement publication come from the agent-side
  :class:`~chartreux.core.agent_loop._request_broker.InteractionRequestBroker`
  and :class:`~chartreux.core.agent_loop._tool_decisions.LoopUserInputAdmission`;
* save-before-response persistence runs through the revision-checked
  :func:`~tests.differential.policy_store.write_project_root`.

Two pieces are harness-local on purpose: the manual clock and gated wakeup make
expiry deterministic instead of wall-clock timed, and the revision labels map
the store's volatile ``dev:ino:mtime_ns:size`` tokens onto ordinals so two
topologies on different temporary paths compare frame-for-frame. Paths that
reach a user-visible note are scrubbed to ``<base>`` for the same reason.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import os
from pathlib import Path

from chartreux.core.agent_loop._request_broker import (
    InteractionRequestBroker,
    UserInputSettlement,
)
from chartreux.core.agent_loop._root_grants import (
    ROOT_GRANT_DECLINED_FEEDBACK,
    ROOT_GRANT_DENY,
    ROOT_GRANT_DENY_LABEL,
    ROOT_GRANT_PROJECT,
    ROOT_GRANT_PROJECT_LABEL,
    ROOT_GRANT_SESSION_LABEL,
    SessionRootGrants,
    parse_root_grant_result,
    root_grant_request,
    root_grant_save_note,
)
from chartreux.core.agent_loop._tool_decisions import LoopUserInputAdmission
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import UserQuestionsConfig
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.deferred_questions import DeferredQuestionStore
from chartreux.core.events import UserInputOrigin, UserInputRequestEvent
from chartreux.core.tools.models import ToolPermission
from chartreux.core.tools.utils import PathAccess, resolve_file_tool_permission
from chartreux.core.workspace import Workspace
from chartreux.questions import (
    QuestionChoice,
    QuestionKind,
    UnansweredQuestions,
    UnattendedDefault,
    UnattendedDefaultAction,
    UserAnswer,
    UserQuestion,
    UserQuestionRequest,
    UserQuestionResult,
)
from tests.differential.policy_store import read_revision, write_project_root
from tests.differential.replay import (
    Channel,
    ReplayError,
    ReplayOutcome,
    build_replay_script,
)

#: The unattended expiry this trace configures and advances the clock past.
EXPIRY_TIMEOUT_SECONDS = 60.0
EXPIRY_MAX_SECONDS = 120.0

ORACLE_SESSION_ID = "oracle-session"
ORACLE_TURN_ID = "oracle-turn"

#: Scenarios whose target paths the driver materializes under the trace base.
PROJECT_DIR = "project"
GRANTED_DIR = "granted"
OTHER_DIR = "other"
INNER_FILE = "inner.txt"

_EXPIRY_PUMP_LIMIT = 200


@dataclass(frozen=True)
class RootOracleOp:
    """One scripted root-side operation inside a turn."""

    kind: str
    subject: str
    detail: str = ""


@dataclass(frozen=True)
class RootOracleTurn:
    """One prompted turn and the root-side operations it carries."""

    prompt: str
    ops: tuple[RootOracleOp, ...]


@dataclass(frozen=True)
class RootOracleScript:
    """A seeded root-side trace: prompts plus their scripted operations."""

    seed: int
    turns: tuple[RootOracleTurn, ...]


#: One full cycle of scripted operations; a seeded script rotates into it.
_OP_CYCLE: tuple[tuple[str, str, str], ...] = (
    ("grant", GRANTED_DIR, ""),
    ("permission", f"{GRANTED_DIR}/{INNER_FILE}", ""),
    ("approval-grant-session", GRANTED_DIR, ""),
    ("save", GRANTED_DIR, "fresh"),
    ("revoke", GRANTED_DIR, ""),
    ("permission", f"{GRANTED_DIR}/{INNER_FILE}", ""),
    ("approval-expire", GRANTED_DIR, ""),
    ("approval-deny", OTHER_DIR, ""),
    ("approval-cancel", OTHER_DIR, ""),
    ("approval-keep-waiting", OTHER_DIR, ""),
    ("approval-grant-project", GRANTED_DIR, ""),
    ("permission", f"{PROJECT_DIR}/{INNER_FILE}", ""),
)


def build_root_oracle_script(
    seed: int, *, turns: int = 6, words: int = 3, ops_per_turn: int = 2
) -> RootOracleScript:
    """Build the deterministic root-side script for ``seed``.

    The prompts come from the [6A] seeded script builder, so one campaign seed
    drives both the echo and the root traces. The operation offset is derived
    from the seed, and each turn carries ``ops_per_turn`` consecutive cycle
    entries, so a six-turn script exercises every operation kind once.
    """
    prompts = build_replay_script(seed, turns=turns, words=words).turns
    offset = seed % len(_OP_CYCLE)
    script_turns = tuple(
        RootOracleTurn(
            prompt=prompt,
            ops=tuple(
                RootOracleOp(
                    *_OP_CYCLE[(offset + index * ops_per_turn + step) % len(_OP_CYCLE)]
                )
                for step in range(ops_per_turn)
            ),
        )
        for index, prompt in enumerate(prompts)
    )
    return RootOracleScript(seed=seed, turns=script_turns)


def _oracle_config() -> ChartreuxConfigSchema:
    return ChartreuxConfigSchema(
        user_questions=UserQuestionsConfig(
            unattended_timeout_enabled=True,
            default_timeout_seconds=EXPIRY_TIMEOUT_SECONDS,
            max_timeout_seconds=EXPIRY_MAX_SECONDS,
        )
    )


def _eligible_question_request() -> UserQuestionRequest:
    """A defaultable model question: the only expiry-eligible request shape."""
    return UserQuestionRequest(
        questions=[
            UserQuestion(
                question="Ship it?",
                options=[QuestionChoice(label="Yes"), QuestionChoice(label="No")],
                kind=QuestionKind.PREFERENCE,
                unattended_default=UnattendedDefault(
                    action=UnattendedDefaultAction.DEFER_AFFECTED_WORK,
                    description="Defer the release until the user answers.",
                ),
            )
        ]
    )


def _root_answer(label: str) -> UserQuestionResult:
    return UserQuestionResult(
        answers=[UserAnswer(question="Grant access?", answer=label)]
    )


class _ManualClock:
    """The deterministic monotonic clock both topologies advance by command."""

    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _GatedWakeup:
    """Expiry wakeups that fire when the scripted command releases them."""

    def __init__(self) -> None:
        self._gates: list[asyncio.Event] = []

    @property
    def armed(self) -> bool:
        """True once the armed timer has reached its wakeup."""
        return bool(self._gates)

    async def __call__(self, delay: float) -> None:
        gate = asyncio.Event()
        self._gates.append(gate)
        await gate.wait()

    def release(self) -> None:
        while self._gates:
            self._gates.pop().set()


class _FramePublisher:
    """Record every settlement publication as an ordered frame."""

    def __init__(self, oracle: RootOracle) -> None:
        self._oracle = oracle

    async def publish_user_input_settlement(
        self, request_id: str, settlement: UserInputSettlement
    ) -> None:
        self._oracle.emit(
            "settlement",
            request=request_id,
            settlement_kind=settlement.kind,
            was_open=settlement.was_open,
        )

    async def after_user_input_settlement(
        self, request_id: str, settlement: UserInputSettlement
    ) -> None:
        self._oracle.emit(
            "settlement_after", request=request_id, settlement_kind=settlement.kind
        )


class RootOracle:
    """The root-side component both topologies host."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel
        self._run = f"run-{os.getpid()}"
        self._sequence = 0
        self._base: Path | None = None
        self._project: Path | None = None
        self._config_path: Path | None = None
        self._initial_revision: str | None = None
        self._revisions: dict[str, str] = {}
        self._grants = SessionRootGrants()
        self._clock = _ManualClock()
        self._gate = _GatedWakeup()
        self._deferred = DeferredQuestionStore(clock=self._clock)
        self._admission = LoopUserInputAdmission(
            is_subagent_getter=lambda: False,
            user_input_capability_getter=lambda: True,
            config_getter=_oracle_config,
            session_id_getter=lambda: ORACLE_SESSION_ID,
            session_generation_getter=lambda: 0,
            active_turn_id_getter=lambda: ORACLE_TURN_ID,
            deferred_questions=self._deferred,
        )
        self._broker = InteractionRequestBroker()
        self._broker.bind_admission(self._admission)
        self._broker.bind_settlement_publisher(_FramePublisher(self))
        self._broker.bind_timer_seams(clock=self._clock, wakeup=lambda: self._gate)
        self._callbacks = 0

    # -- frame stream ----------------------------------------------------
    def emit(self, kind: str, **fields: object) -> None:
        frame: dict[str, object] = {
            "type": kind,
            "run": self._run,
            "sequence": self._sequence,
        }
        frame.update(fields)
        self._channel.send(frame)
        self._sequence += 1

    # -- scenario paths --------------------------------------------------
    def _base_dir(self) -> Path:
        if self._base is None:
            raise ReplayError("root oracle has no trace base")
        return self._base

    def _project_dir(self) -> Path:
        if self._project is None:
            raise ReplayError("root oracle has no project directory")
        return self._project

    def _config_file(self) -> Path:
        if self._config_path is None:
            raise ReplayError("root oracle has no policy store")
        return self._config_path

    def _resolve(self, subject: str) -> Path:
        return self._base_dir() / subject

    def _logical(self, path: Path) -> str:
        return path.resolve().relative_to(self._base_dir().resolve()).as_posix()

    # -- session authority -----------------------------------------------
    def _effective_roots(self) -> tuple[Path, ...]:
        """The session's authorized roots, as the loop's policy view builds them."""
        project = self._project_dir().resolve()
        roots = {
            root.resolve() for root in self._grants.roots if root.resolve() != project
        }
        return tuple(sorted(roots, key=self._logical))

    def _workspace(self) -> Workspace:
        return Workspace(
            cwd=self._project_dir().resolve(), authorized_roots=self._effective_roots()
        )

    def _authorized(self) -> list[str]:
        return [self._logical(root) for root in self._effective_roots()]

    # -- policy store ----------------------------------------------------
    def _label(self, revision: str | None) -> str | None:
        if revision is None:
            return None
        return self._revisions.setdefault(revision, f"rev-{len(self._revisions)}")

    def _scrub(self, text: str) -> str:
        return text.replace(str(self._base_dir()), "<base>")

    async def _save(self, *, subject: str, root: Path, stale: bool) -> ConfigSaveResult:
        expected = (
            self._initial_revision
            if stale
            else await read_revision(self._config_file())
        )
        if expected is None:
            raise ReplayError("root oracle captured no initial revision")
        result = await write_project_root(
            config=self._config_file(),
            project=self._project_dir(),
            root=root,
            expected_revision=expected,
        )
        self.emit(
            "save",
            subject=subject,
            basis="initial" if stale else "current",
            target=result.target,
            persistence=result.persistence,
            application=result.application,
            error=result.error,
            revision=self._label(result.revision),
        )
        return result

    # -- operations ------------------------------------------------------
    def _op_grant(self, subject: str) -> None:
        root = self._resolve(subject)
        cleared = root.resolve() in self._grants.denied_roots
        fresh = self._grants.grant(root)
        self.emit(
            "grant",
            subject=subject,
            fresh=fresh,
            deny_memory_cleared=cleared,
            authorized=self._authorized(),
        )

    def _op_revoke(self, subject: str) -> None:
        """Narrow the session's authorized roots (an explicit user scope change)."""
        root = self._resolve(subject).resolve()
        denied = set(self._grants.denied_roots)
        rebuilt = SessionRootGrants()
        for entry in sorted(item for item in self._grants.roots if item != root):
            rebuilt.grant(entry)
        for entry in sorted(denied):
            rebuilt.deny(entry)
        self._grants = rebuilt
        self.emit("revoke", subject=subject, authorized=self._authorized())

    def _op_permission(self, subject: str) -> None:
        context = resolve_file_tool_permission(
            str(self._resolve(subject)),
            tool_name="read_file",
            allowlist=[],
            denylist=[],
            config_permission=ToolPermission.ALWAYS,
            sensitive_patterns=[],
            workspace=self._workspace(),
            access=PathAccess.READ,
        )
        self.emit(
            "permission",
            subject=subject,
            tool="read_file",
            permission="unresolved" if context is None else str(context.permission),
            denial_kind=None if context is None else context.denial_kind,
        )

    async def _op_save(self, subject: str, *, stale: bool) -> None:
        await self._save(subject=subject, root=self._resolve(subject), stale=stale)

    async def _op_approval(self, mode: str, subject: str) -> None:
        request_id = self._open_approval(mode, subject)
        if mode == "grant-session":
            await self._settle_grant(request_id, subject, ROOT_GRANT_SESSION_LABEL)
        elif mode == "grant-project":
            await self._settle_grant(request_id, subject, ROOT_GRANT_PROJECT_LABEL)
        elif mode == "deny":
            await self._settle_deny(request_id, subject)
        elif mode == "duplicate":
            await self._settle_duplicate(request_id, subject)
        elif mode == "expire":
            await self._expire(request_id)
        elif mode == "keep-waiting":
            await self._keep_waiting(request_id)
        elif mode == "cancel":
            await self._cancel(request_id)
        else:
            raise ReplayError(f"unknown approval mode: {mode!r}")

    def _open_approval(self, mode: str, subject: str) -> str:
        self._callbacks += 1
        request_id = f"callback-{self._callbacks}"
        eligible = mode in {"expire", "keep-waiting"}
        event = UserInputRequestEvent(
            request_id=request_id,
            args=(
                _eligible_question_request()
                if eligible
                else root_grant_request(self._resolve(subject))
            ),
            tool_call_id="call-1",
            origin=(
                UserInputOrigin.MODEL_QUESTION
                if eligible
                else UserInputOrigin.RUNTIME_APPROVAL
            ),
        )
        # The admission decision is the production one: runtime approvals are
        # permanently expiry-ineligible, defaultable model questions arm it.
        policy = self._admission.expiry_policy(event)
        self._broker.relay_user_input_request(event)
        record = self._broker.record(request_id)
        self.emit(
            "approval_open",
            request=request_id,
            subject=subject,
            mode=mode,
            origin=str(event.origin),
            expiry_seconds=None if policy is None else policy.timeout_seconds,
            armed=record is not None and record.expiry is not None,
        )
        return request_id

    async def _settle_grant(self, request_id: str, subject: str, label: str) -> None:
        answer = _root_answer(label)
        outcome = await self._broker.settle_answer(request_id, answer)
        decision = parse_root_grant_result(answer)
        root = self._resolve(subject)
        applied = self._grants.grant(root) if decision != ROOT_GRANT_DENY else False
        self.emit(
            "approval_answer",
            request=request_id,
            subject=subject,
            outcome=outcome,
            decision=decision,
            applied=applied,
            authorized=self._authorized(),
            waiter_closed=self._waiter_closed(request_id),
        )
        if decision != ROOT_GRANT_PROJECT:
            return
        # Save-before-response: the session grant is already applied above and
        # the persistence outcome is reported apart from it.
        result = await self._save(subject=subject, root=root, stale=False)
        note = root_grant_save_note(root, self._project_dir(), result)
        self.emit("note", subject=subject, text=self._scrub(note))

    async def _settle_deny(self, request_id: str, subject: str) -> None:
        answer = _root_answer(ROOT_GRANT_DENY_LABEL)
        outcome = await self._broker.settle_answer(request_id, answer)
        decision = parse_root_grant_result(answer)
        self._grants.deny(self._resolve(subject))
        self.emit(
            "approval_answer",
            request=request_id,
            subject=subject,
            outcome=outcome,
            decision=decision,
            applied=False,
            feedback=ROOT_GRANT_DECLINED_FEEDBACK,
            waiter_closed=self._waiter_closed(request_id),
        )

    async def _settle_duplicate(self, request_id: str, subject: str) -> None:
        answer = _root_answer(ROOT_GRANT_SESSION_LABEL)
        accepted = await self._broker.settle_answer(request_id, answer)
        duplicate = await self._broker.settle_answer(request_id, answer)
        conflict = await self._broker.settle_answer(
            request_id, _root_answer(ROOT_GRANT_DENY_LABEL)
        )
        applied = self._grants.grant(self._resolve(subject))
        self.emit(
            "approval_arbitration",
            request=request_id,
            subject=subject,
            accepted=accepted,
            duplicate=duplicate,
            conflict=conflict,
            decision=parse_root_grant_result(answer),
            applied=applied,
            authorized=self._authorized(),
        )

    async def _expire(self, request_id: str) -> None:
        record = self._broker.record(request_id)
        if record is None:
            raise ReplayError(f"no broker record for {request_id}")
        # Let the armed timer task reach its wakeup before releasing it: the
        # gate is what the timer awaits, so releasing it early would leave the
        # task waiting on a gate nobody ever sets.
        for _ in range(_EXPIRY_PUMP_LIMIT):
            if self._gate.armed:
                break
            await asyncio.sleep(0)
        self._clock.advance(EXPIRY_MAX_SECONDS)
        self._gate.release()
        for _ in range(_EXPIRY_PUMP_LIMIT):
            if record.expired:
                break
            await asyncio.sleep(0)
        if not record.expired:
            raise ReplayError(f"unattended expiry did not fire for {request_id}")
        expired = record.future.result()
        if not isinstance(expired, UnansweredQuestions):
            raise ReplayError(f"expiry settled with {type(expired).__name__}")
        self.emit(
            "approval_expiry",
            request=request_id,
            disposition=expired.disposition,
            reason=expired.reason,
            timeout_seconds=expired.timeout_seconds,
            elapsed_seconds=expired.elapsed_seconds,
            question_ids=[question.question_id for question in expired.questions],
            deferred=len(self._deferred),
            decision=parse_root_grant_result(expired),
        )
        late = await self._broker.settle_answer(
            request_id, _root_answer(ROOT_GRANT_SESSION_LABEL)
        )
        self.emit(
            "approval_late_answer",
            request=request_id,
            outcome=late,
            waiter_closed=self._waiter_closed(request_id),
        )

    async def _keep_waiting(self, request_id: str) -> None:
        first = await self._broker.keep_waiting(request_id)
        second = await self._broker.keep_waiting(request_id)
        answered = await self._broker.settle_answer(
            request_id,
            UserQuestionResult(answers=[UserAnswer(question="Ship it?", answer="Yes")]),
        )
        self.emit(
            "approval_keep_waiting",
            request=request_id,
            first=first,
            second=second,
            answered=answered,
        )

    async def _cancel(self, request_id: str) -> None:
        record = self._broker.record(request_id)
        if record is None:
            raise ReplayError(f"no broker record for {request_id}")
        await self._broker.cancel(request_id, RuntimeError("turn ended"))
        late = await self._broker.settle_answer(
            request_id, _root_answer(ROOT_GRANT_SESSION_LABEL)
        )
        # The cancelled waiter's exception is the cancellation itself; consume
        # it so the trace's stderr carries no unretrieved-future noise.
        if record.future.done() and not record.future.cancelled():
            record.future.exception()
        self.emit(
            "approval_cancel",
            request=request_id,
            closed=record.core_resolved,
            late_answer=late,
        )

    def _waiter_closed(self, request_id: str) -> bool:
        record = self._broker.record(request_id)
        return record is not None and record.future.done()

    async def _run_op(self, op: Mapping[str, object]) -> None:
        kind = str(op.get("kind", ""))
        subject = str(op.get("subject", ""))
        detail = str(op.get("detail", ""))
        if kind == "grant":
            self._op_grant(subject)
        elif kind == "revoke":
            self._op_revoke(subject)
        elif kind == "permission":
            self._op_permission(subject)
        elif kind == "save":
            await self._op_save(subject, stale=detail == "stale")
        elif kind.startswith("approval-"):
            await self._op_approval(kind.removeprefix("approval-"), subject)
        else:
            raise ReplayError(f"unknown root oracle operation: {op!r}")

    # -- command loop ----------------------------------------------------
    async def serve(self) -> None:
        try:
            while (message := self._channel.receive()) is not None:
                await self._dispatch(message)
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        for record in self._broker.records().values():
            expiry = record.expiry
            if expiry is not None and expiry.timer is not None:
                expiry.timer.cancel()
                expiry.timer = None
        self._gate.release()
        await asyncio.sleep(0)

    async def _dispatch(self, message: Mapping[str, object]) -> None:
        kind = message.get("type")
        if kind == "init":
            await self._initialize(message)
        elif kind == "turn":
            await self._turn(message)
        else:
            raise ReplayError(f"unknown root oracle message: {message!r}")

    async def _initialize(self, message: Mapping[str, object]) -> None:
        self._base = Path(str(message["base"]))
        self._project = Path(str(message["project"]))
        self._config_path = Path(str(message["config"]))
        self._initial_revision = await read_revision(self._config_file())
        self.emit(
            "ready",
            project=self._logical(self._project_dir()),
            revision=self._label(self._initial_revision),
            expiry_default_seconds=EXPIRY_TIMEOUT_SECONDS,
            authorized=self._authorized(),
        )

    async def _turn(self, message: Mapping[str, object]) -> None:
        text = str(message.get("text", ""))
        words = tuple(word for word in text.split(" ") if word)
        for index, word in enumerate(words):
            self.emit("delta", index=index, text=word)
        ops = message.get("ops", [])
        if not isinstance(ops, list):
            raise ReplayError(f"turn operations are not a list: {ops!r}")
        for op in ops:
            if not isinstance(op, dict):
                raise ReplayError(f"turn operation is not an object: {op!r}")
            await self._run_op(op)
        self.emit("response", text=text, words=len(words))
        self.emit("turn_end", text=text)


def run_root_oracle_loop(channel: Channel) -> None:
    """Component entry point shared by both topologies."""
    asyncio.run(RootOracle(channel).serve())


class RootOracleDriver:
    """Drive a root oracle script over any channel."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel

    def initialize(
        self, *, base: Path, project: Path, config: Path
    ) -> dict[str, object]:
        self._channel.send({
            "type": "init",
            "base": str(base),
            "project": str(project),
            "config": str(config),
        })
        ready = self._channel.receive()
        if ready is None:
            raise ReplayError("root oracle closed before it was ready")
        if ready.get("type") != "ready":
            raise ReplayError(f"unexpected root oracle frame: {ready!r}")
        return ready

    def run(self, script: RootOracleScript) -> ReplayOutcome:
        frames: list[dict[str, object]] = []
        texts: list[str] = []
        for turn in script.turns:
            self._channel.send({
                "type": "turn",
                "text": turn.prompt,
                "ops": [asdict(op) for op in turn.ops],
            })
            while True:
                frame = self._channel.receive()
                if frame is None:
                    raise ReplayError("root oracle closed mid-turn")
                frames.append(frame)
                if frame.get("type") == "turn_end":
                    texts.append(str(frame.get("text", "")))
                    break
        return ReplayOutcome(frames=tuple(frames), texts=tuple(texts))
