from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import logging
import time
from typing import TYPE_CHECKING, Any, Literal, Protocol

from chartreux.core.agent_loop._file_state_context import (
    render_file_state_compaction_summary,
)
from chartreux.core.agent_loop.errors import EmptyLLMResponseError
from chartreux.core.compaction.context import (
    collect_prior_user_messages,
    drop_oldest_round,
    extract_summary,
    render_compaction_context,
    select_model_context,
)
from chartreux.core.deferred_questions import DeferredQuestionStore
from chartreux.core.errors import ContextTooLongError
from chartreux.core.file_state import FileStateStore
from chartreux.core.llm.thinking_levels import (
    get_thinking_levels,
    resolve_thinking_level,
)
from chartreux.core.llm_models import (
    LLMMessage,
    PersistedToolResult,
    Role,
    archived_replacement,
)
from chartreux.core.prompts import UtilityPrompt
from chartreux.core.session.transcript_archive import eligible_archive_prefix
from chartreux.core.tools.builtins.todo import TodoState
from chartreux.core.usage import UsagePurpose
from chartreux.observability.logging import log_model_call_success

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema, ModelConfig
    from chartreux.core.llm_models import AvailableTool, LLMChunk, StrToolChoice
    from chartreux.core.message_list import MessageList
    from chartreux.core.policy.bundle import ActivatedPolicy
    from chartreux.core.session.session_logger import SessionLogger
    from chartreux.core.session_types import AgentStats
    from chartreux.core.tools.manager import ToolManager

logger = logging.getLogger(__name__)

_COMPACTION_PTL_RETRIES = 3
CompactionFailureReason = Literal["tool_call", "empty_summary"]

#: P4d ACTIVATES the M1 payload-only trim: after a successful compaction the
#: sealed prefix's ``reasoning_payloads`` are released from RAM. The flag stays
#: a module-level switch so a package gate can prove the activation (P4b pinned
#: it False; P4d's tests pin it True and prove the trim path runs). Structural
#: tombstones remain out of scope.
ARCHIVE_TRIM_ENABLED: bool = True


def _pending_cancellation() -> bool:
    """True when the running task has a cancellation request outstanding.

    ``Task.cancelling()`` keeps counting after a provider (or any other
    awaiter) swallowed the delivered ``CancelledError``. That is exactly the
    state a cancelled manual compaction must not publish through: the summary
    may be complete while the request it was computed for is already gone.
    """
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


class CompactionFailedError(Exception):
    """Raised when a compaction turn did not produce a usable summary."""

    def __init__(self, reason: str) -> None:
        self.reason = reason  # a CompactionFailureReason
        super().__init__(f"Compaction did not produce a summary (reason={reason}).")


class CompactionSettlementError(Exception):
    """Raised when a published compaction boundary could not be settled.

    The boundary — and with it the summary — is already in the live message
    list; what failed is the persistence and the archive trim behind it. A
    published boundary that failed to settle is a persistence failure, never a
    compaction failure, so the caller classifies on ``published``.
    """

    def __init__(self, published: bool, summary_length: int) -> None:
        self.published = published
        self.summary_length = summary_length
        super().__init__(
            "Compaction summary was published but could not be settled "
            f"(summary_length={summary_length})."
        )


@dataclass(slots=True)
class CompactionOutcome:
    """Publication/settlement carrier of the most recent manual compaction.

    The manager resets it at entry, marks ``published`` at the boundary append
    and records an observed cancellation instead of abandoning an
    already-published boundary. The summary itself travels through
    ``compact``'s return value, so the ``str`` contract is unchanged.
    """

    published: bool = False
    cancellation_observed: bool = False


class CompletionFn(Protocol):
    """One accounted, non-streaming model call (e.g. ``AgentLoop._complete``)."""

    async def __call__(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        call_type: str | None,
        account_conversation: bool,
        purpose: UsagePurpose = UsagePurpose.CONVERSATION,
    ) -> LLMChunk: ...


@dataclass(frozen=True, slots=True)
class PendingPostTurnTrim:
    """A post-turn trim captured at the exclusive idle seam (T1a).

    Keyed by session / message epoch / boundary: the record is meaningful only
    while the session, the ``MessageList`` epoch and the compaction boundary it
    was captured at all still hold. A list rebind (reset/resume bumps the
    epoch) or a new session makes it stale, and the next attempt discards it.
    The capture is the single recording point; the same guarded entry makes the
    recorded pending LIVE — drain/fence/seal/publish through
    ``SessionLogger.complete_post_turn_trim`` (T1b).
    """

    session_id: str
    message_epoch: int
    boundary_message_id: str


class CompactionManager:
    """Summarizes a conversation into a compact-context envelope.

    Every model call goes through the injected ``complete`` callable (so usage
    is always accounted), and the summary is generated on a local copy of the
    messages — the live ``MessageList`` is only mutated by appending the final
    compaction boundary on success, so a failure never leaves a partially
    trimmed conversation behind.
    """

    def __init__(  # noqa: PLR0913 - runtime-owned dependency injection
        self,
        *,
        messages: MessageList,
        stats_getter: Callable[[], AgentStats],
        config_getter: Callable[[], ChartreuxConfigSchema],
        complete: CompletionFn,
        available_tools: Callable[[], list[AvailableTool]],
        tool_choice: Callable[[], StrToolChoice | AvailableTool],
        save: Callable[[], Awaitable[None]],
        file_state_getter: Callable[[], FileStateStore] | None = None,
        deferred_questions_getter: Callable[[], DeferredQuestionStore] | None = None,
        bound_policy_getter: Callable[[], ActivatedPolicy] | None = None,
        session_logger_getter: Callable[[], SessionLogger | None] | None = None,
        tool_manager_getter: Callable[[], ToolManager | None] | None = None,
        has_active_turn: Callable[[], bool] | None = None,
        trim_enabled_getter: Callable[[], bool] | None = None,
        pending_trim_getter: Callable[[], PendingPostTurnTrim | None] | None = None,
        pending_trim_recorder: Callable[[PendingPostTurnTrim | None], None]
        | None = None,
    ) -> None:
        self._messages = messages
        self._stats = stats_getter
        self._config = config_getter
        self._complete = complete
        self._available_tools = available_tools
        self._tool_choice = tool_choice
        self._save = save
        self._file_state_getter = file_state_getter
        self._deferred_questions_getter = deferred_questions_getter
        self._bound_policy_getter = bound_policy_getter
        # M1 trim wiring (wired in P4b, ACTIVE since P4d behind
        # ARCHIVE_TRIM_ENABLED): the sealing surface and the closed-prefix
        # (active-invocation) guard.
        self._session_logger_getter = session_logger_getter
        self._tool_manager_getter = tool_manager_getter
        self._has_active_turn = has_active_turn
        self._trim_enabled_getter = trim_enabled_getter
        # T1a pending post-turn trim holder wiring: the loop owns the holder;
        # the guarded post-finalization entry reads and records through these.
        self._pending_trim_getter = pending_trim_getter
        self._pending_trim_recorder = pending_trim_recorder

    async def compact(
        self,
        extra_instructions: str = "",
        *,
        manual: bool = False,
        outcome: CompactionOutcome | None = None,
        mode_notice: str | None = None,
    ) -> str:
        """Summarize the conversation and publish the compaction boundary.

        ``manual`` selects the request-driven path (a manual compaction whose
        request can be cancelled by the server's own deadline) and defaults
        off, so the automatic caller is unchanged; ``outcome`` is the carrier
        that path reads after a cancellation and is reset here at entry.
        ``mode_notice`` rides the published boundary CONTENT: it is the
        model-visible carrier, whereas ``extra_instructions`` reach only the
        summarizer request.
        """
        if outcome is not None:
            outcome.published = False
            outcome.cancellation_observed = False
        summary_prefix = UtilityPrompt.COMPACT_SUMMARY_PREFIX.read()
        snapshot = select_model_context(self._messages)
        prior_user_messages = collect_prior_user_messages(snapshot, summary_prefix)

        try:
            summary = await self._summarize(snapshot, extra_instructions)
        except asyncio.CancelledError:
            if outcome is not None:
                outcome.cancellation_observed = True
            raise
        if manual and _pending_cancellation():
            # The summarizer may have swallowed the cancellation and returned a
            # usable summary, so the finished summary proves nothing about the
            # request. No await follows until after the append, which means this
            # synchronous check is the only place a manual compaction can still
            # observe the cancellation: nothing below it may evict file state or
            # append a boundary.
            if outcome is not None:
                outcome.cancellation_observed = True
            raise asyncio.CancelledError
        todo_state = TodoState.replay(self._messages).model_dump(mode="json")
        output: dict[str, Any] = {"todo_state": todo_state}
        file_state_section: str | None = None
        if self._file_state_getter is not None:
            # The working-set file state rides the envelope exactly like the
            # todo state (metadata only: path -> last-seen revision + outline).
            # Trimming happens only here — a compaction boundary is the one
            # place a working-set eviction is cache-safe, because the boundary
            # rewrites the model-facing prefix anyway.
            file_store = self._file_state_getter()
            file_store.trim_to_cap()
            output.update(file_store.to_envelope())
            # The envelope is structural only (replayed on resume): the
            # backend drops tool_result from the converted LLM request, so
            # the model would otherwise lose all pre-compaction file knowledge
            # (unchanged files generate no new state block). The bounded
            # working-set summary — paths, last-seen revisions, current
            # outlines — is therefore rendered into the boundary CONTENT once,
            # at this boundary.
            file_state_section = render_file_state_compaction_summary(file_store)
        if self._deferred_questions_getter is not None:
            # Deferred-question records (unattended question expiry, plan U)
            # ride the envelope exactly like the todo and file state, so a
            # reload can restore them; bounding never silently drops items.
            output.update(self._deferred_questions_getter().to_envelope())

        envelope = LLMMessage(
            role=Role.user,
            content=render_compaction_context(
                prior_user_messages, summary, file_state_section, mode_notice
            ),
            injected=True,
            context_boundary="compaction",
            tool_result=PersistedToolResult(output=output),
        )
        self._messages.append(envelope)
        if outcome is not None:
            # Publication happens here and nowhere else, so the carrier is the
            # single source a cancelled caller classifies on.
            outcome.published = True
        # Context size is unknown without an API call; the next LLM turn
        # recomputes it accurately from real usage.
        self._stats().context_tokens = -1
        if manual:
            await self._settle_manual(summary, outcome)
        else:
            await self._settle_published()
        return summary

    async def _settle_published(self) -> None:
        """Persist the published boundary, then run the optional archive trim."""
        await self._save()
        if self._trim_enabled():
            # P4d ACTIVATION SITE: the envelope above is persisted with the
            # originals still in RAM, so the trim may now seal and evict the
            # pre-boundary payloads. A failed trim always leaves the originals.
            await self._trim_archived_prefix()

    async def _settle_manual(
        self, summary: str, outcome: CompactionOutcome | None
    ) -> None:
        """Settle a published manual boundary, absorbing further cancellation.

        The boundary is already in the live message list, so abandoning the
        save or the trim here would leave it half-settled, and the caller's own
        bookkeeping runs only after a normal return. A cancellation delivered
        during settlement is therefore recorded on the carrier and settlement
        continues; repeated cancellations must not tear it either. A settlement
        failure is typed instead of raw, because the caller has to report a
        persistence failure for a boundary that is already published.
        """
        settlement = asyncio.create_task(self._settle_published())
        while True:
            try:
                await asyncio.shield(settlement)
                return
            except asyncio.CancelledError:
                if settlement.cancelled():
                    # Only a cancelled settlement has no outcome left to
                    # report. One that already finished is awaited again
                    # below, so its summary or its own failure still surfaces
                    # instead of being replaced by this cancellation.
                    raise
                if outcome is not None:
                    outcome.cancellation_observed = True
            except Exception as exc:
                raise CompactionSettlementError(
                    published=True, summary_length=len(summary)
                ) from exc

    def _trim_enabled(self) -> bool:
        """Whether the sealed-prefix trim path may run (active since P4d)."""
        if self._trim_enabled_getter is not None:
            return self._trim_enabled_getter()
        return ARCHIVE_TRIM_ENABLED

    async def _trim_archived_prefix(self) -> None:
        """Seal the eligible closed prefix and publish its RAM replacements (M1 §1).

        ACTIVE since P4d. Exact procedure, each step owned by the merged
        substrate: (1) the envelope is already appended and persisted with the
        originals present (``compact`` saved before calling this); (2) the
        eligible prefix is the closed, settled region strictly before the
        newest boundary, system messages excluded; (3) sealing runs under the
        logger's save lock and issues a durable receipt — a returned save call
        is not authorization, the receipt is; (4) replacement copies are
        prepared WITHOUT mutating the originals; (5) queued stale saves are
        fenced and (6) replacements plus the descriptor are published
        synchronously, with no intervening await, no ``reset()`` and no reset
        hooks. Any sealing, persistence, or verification failure retains the
        originals: compaction may succeed without RAM trimming. Closed-prefix
        restriction (wiring side): an automatic compaction that fires during
        an active invocation leaves the active invocation intact until a
        subsequent compaction.
        """
        logger_ = self._session_logger_getter() if self._session_logger_getter else None
        tool_manager = (
            self._tool_manager_getter() if self._tool_manager_getter else None
        )
        if logger_ is None or tool_manager is None or not logger_.enabled:
            return
        if self._has_active_turn is not None and self._has_active_turn():
            # The active invocation stays intact until a subsequent compaction.
            return
        try:
            receipt = await logger_.seal_archive(
                self._messages,
                self._stats(),
                self._config(),
                tool_manager,
                trim_mode="payloads",
            )
            eligible = eligible_archive_prefix(list(self._messages))
            if (
                eligible is None
                or len(eligible.messages) != receipt.descriptor.archived_count
            ):
                # The list moved between sealing and publication; retain the
                # originals (the fence would refuse a stale descriptor anyway).
                return
            replacements = [
                archived_replacement(
                    message, receipt.descriptor, trim_mode=receipt.descriptor.trim_mode
                )
                for message in eligible.messages
            ]
            logger_.fence_and_publish_archive_trim(
                self._messages,
                receipt.descriptor,
                replacements,
                save_generation=logger_.transcript_save_generation,
                reset_generation=logger_.archive_reset_generation,
            )
        except Exception as exc:
            # Compaction may succeed without RAM trimming: retain the originals.
            logger.debug("archived-prefix trim skipped: %s", exc)

    async def capture_post_turn_pending_trim(
        self, *, session_id: str, message_epoch: int
    ) -> None:
        """GUARDED post-finalization entry — the T1 post-turn seal (T1a capture
        + T1b drain/fence/publication, landed together at one seam).

        ``AgentLoop.act`` awaits this after ``checkpoint_recorder.seal_turn()``
        and before ``_active_turn = None``: the genuinely exclusive idle seam
        (a new act(), resume, the relocator and config transactions are all
        refused while the turn is still registered). The entry records the
        pending trim in the loop's holder — the single recording point — and
        then makes the recorded pending LIVE at the same exclusive seam: it
        drains queued/in-flight saves, fences, seals the archive for the
        pending boundary and publishes, through
        ``SessionLogger.complete_post_turn_trim``.

        Guards: the post-turn seam only (a turn must still be registered);
        trim enabled; a live sealing surface; the holder wired. A pending
        captured at another session or message epoch is stale after a rebind
        and is discarded at this attempt boundary. The newest compaction
        boundary coalesces (an existing same-key pending is replaced, never
        stacked). A boundary is never invented merely to trim, and an invalid
        eligible prefix is skipped. The holder is consumed at the attempt
        boundary. Any failure — including cancellation, which the logger
        drains without publishing — is absorbed: a failed or cancelled
        finalization never turns a completed turn into a failure, never
        authorizes eviction, and never partially publishes; the originals are
        retained.
        """
        try:
            if self._pending_trim_getter is None or self._pending_trim_recorder is None:
                return
            if self._has_active_turn is not None and not self._has_active_turn():
                # Not the post-finalization seam: no turn still registered.
                return
            if not self._trim_enabled():
                return
            logger_ = (
                self._session_logger_getter() if self._session_logger_getter else None
            )
            if logger_ is None or not logger_.enabled:
                return
            current = self._pending_trim_getter()
            if current is not None and (
                current.session_id != session_id
                or current.message_epoch != message_epoch
            ):
                # Stale on rebind/new session: discard at the attempt boundary.
                self._pending_trim_recorder(None)
            eligible = eligible_archive_prefix(list(self._messages))
            if eligible is None or eligible.boundary_message_id is None:
                # Never invent a boundary merely to trim; invalid prefixes skip.
                return
            # Coalesce to the newest boundary (the single recording point).
            pending = PendingPostTurnTrim(
                session_id=session_id,
                message_epoch=message_epoch,
                boundary_message_id=eligible.boundary_message_id,
            )
            self._pending_trim_recorder(pending)
            # T1b: make the recorded pending LIVE at the same exclusive seam —
            # drain/fence/seal/publish. The holder is consumed at this attempt
            # boundary whatever the outcome.
            try:
                await self._complete_post_turn_pending_trim(logger_, pending)
            except asyncio.CancelledError:
                # The logger drained its in-flight persist and published
                # nothing; absorb so the completed turn does not become a
                # failure and the loop can unregister the turn normally.
                logger.debug("post-turn trim completion cancelled; originals retained")
            finally:
                self._pending_trim_recorder(None)
        except Exception as exc:
            # A failed finalization never turns a completed turn into a failure.
            logger.debug("post-turn pending trim capture skipped: %s", exc)

    async def _complete_post_turn_pending_trim(
        self, logger_: SessionLogger, pending: PendingPostTurnTrim
    ) -> None:
        """Drain/fence/seal/publish the recorded pending through the logger (T1b)."""
        tool_manager = (
            self._tool_manager_getter() if self._tool_manager_getter else None
        )
        if tool_manager is None:
            return
        result = await logger_.complete_post_turn_trim(
            self._messages,
            self._stats(),
            self._config(),
            tool_manager,
            session_id=pending.session_id,
            message_epoch=pending.message_epoch,
            boundary_message_id=pending.boundary_message_id,
        )
        logger.debug(
            "post-turn trim outcome=%s detail=%r", result.outcome, result.detail
        )

    async def _summarize(
        self, snapshot: list[LLMMessage], extra_instructions: str
    ) -> str:
        config = self._config()
        request = config.compaction_prompt
        if self._bound_policy_getter is not None:
            # The frozen bundle binding (design §10): the bound selections AND
            # their content revisions are session-bound facts the summarizer
            # cannot see anywhere else in the transcript, so they ride the
            # request verbatim — saved selection or resource edits never apply.
            manifest = self._bound_policy_getter().manifest
            request += (
                f"\n\nBound policy bundle: identity {manifest.identity.selected} "
                f"(revision {manifest.identity.revision}); workflow "
                f"{manifest.workflow.selected} (revision {manifest.workflow.revision}); "
                f"orchestration strategy {manifest.strategy.selected} "
                f"(revision {manifest.strategy.revision or 'none'}); "
                f"frozen policy hash {manifest.frozen_policy_hash}. "
                "Copy these session-bound fields into the handoff; do not adopt "
                "saved selection or resource edits."
            )
        if extra_instructions:
            request += f"\n\n## Additional Instructions\n{extra_instructions}"

        summary, working, reason = await self._primary(snapshot, request)
        if summary is not None:
            return summary

        # Strict mode has no fallback and is terminal; otherwise try the dedicated
        # fallback before giving up. Either way, report the primary's failure mode
        # (the fallback can only ever fail as an empty summary, so the primary
        # reason — e.g. tool_call — is the informative one).
        assert reason is not None  # a None summary always carries a reason
        if not self._config().raise_on_compaction_failure:
            recovered = await self._fallback(working, request)
            if recovered is not None:
                return recovered
        raise CompactionFailedError(reason)

    async def _primary(
        self, snapshot: list[LLMMessage], request: str
    ) -> tuple[str | None, list[LLMMessage], CompactionFailureReason | None]:
        # Cache-friendly: the request rides on the live conversation's token
        # prefix. Works on a local copy, so the live message list is untouched.
        request_message = LLMMessage(role=Role.user, content=request)
        result, working = await self._summarize_call(
            list(snapshot),
            lambda history: [*history, request_message],
            model=self._config().get_compaction_model(),
            tools=self._available_tools(),
            tool_choice=self._tool_choice(),
        )
        if result is None:
            return None, working, "empty_summary"
        if result.message.tool_calls:
            return None, working, "tool_call"
        summary = extract_summary(result.message.content or "")
        if summary is None:
            return None, working, "empty_summary"
        return summary, working, None

    async def _fallback(self, history: list[LLMMessage], request: str) -> str | None:
        # Dedicated summarizer call: fresh system prompt, no tools, and no more
        # thinking than the model can encode. Models that cannot disable thinking
        # keep the level nearest their configured one among the declared levels.
        config = self._config()
        model = config.get_compaction_model()
        provider = config.get_provider_for_model(model)
        levels = get_thinking_levels(
            str(provider.backend), provider.api_style, model.name
        )
        fallback_thinking = "off"
        if levels is not None:
            available_levels = (
                levels
                if model.supported_thinking_levels is None
                else {
                    level: levels[level]
                    for level in model.supported_thinking_levels
                    if level in levels
                }
            )
            if "off" not in available_levels:
                fallback_thinking = resolve_thinking_level(
                    available_levels, model.thinking
                ).effective
        model = model.model_copy(update={"thinking": fallback_thinking})
        result, _ = await self._summarize_call(
            list(history),
            lambda working: [
                LLMMessage(
                    role=Role.system, content=UtilityPrompt.COMPACT_SYSTEM.read()
                ),
                LLMMessage(
                    role=Role.user,
                    content=f"{request}\n\n## Conversation Transcript\n"
                    f"{self._render_transcript(working)}",
                ),
            ],
            model=model,
            tools=None,
            tool_choice=None,
        )
        return (
            extract_summary(result.message.content or "")
            if result is not None
            else None
        )

    async def _summarize_call(
        self,
        working: list[LLMMessage],
        build_messages: Callable[[list[LLMMessage]], list[LLMMessage]],
        *,
        model: ModelConfig,
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
    ) -> tuple[LLMChunk | None, list[LLMMessage]]:
        # The gateway imports compaction context, so defer this import to avoid a cycle.
        from chartreux.core.agent_loop.llm_gateway import IncompleteLLMResponseError

        # Summarize `working`; on overflow drop the oldest round and retry.
        # Returns the model result and the (possibly trimmed) history it used.
        tries_left = _COMPACTION_PTL_RETRIES
        while True:
            try:
                start_time = time.perf_counter()
                result = await self._complete(
                    model=model,
                    messages=build_messages(working),
                    tools=tools,
                    tool_choice=tool_choice,
                    call_type="secondary_call",
                    account_conversation=False,
                    purpose=UsagePurpose.COMPACTION,
                )
                _usage = result.usage
                log_model_call_success(
                    model.alias,
                    int((time.perf_counter() - start_time) * 1000),
                    prompt_tokens=_usage.prompt_tokens if _usage else 0,
                    completion_tokens=_usage.completion_tokens if _usage else 0,
                    cached_tokens=_usage.cached_tokens if _usage else 0,
                )
                return result, working
            except (EmptyLLMResponseError, IncompleteLLMResponseError):
                return None, working
            except ContextTooLongError:
                trimmed = drop_oldest_round(working)
                if tries_left == 0 or trimmed is None:
                    raise
                working = trimmed
                tries_left -= 1

    def _render_transcript(self, messages: Sequence[LLMMessage]) -> str:
        # Render each turn to text, keeping tool calls (the agent's actions) so
        # the summarizer sees what invoked the tool results it also sees.
        # Reasoning is intentionally dropped — it's ephemeral and would bloat the
        # transcript compaction is trying to shrink.
        blocks: list[str] = []
        for message in messages:
            if message.role == Role.system:
                continue
            parts: list[str] = []
            text = (message.content or "").strip()
            if text:
                parts.append(text)
            for call in message.tool_calls or []:
                if call.function is not None:
                    parts.append(
                        f"[calls {call.function.name}({call.function.arguments})]"
                    )
            if not parts:
                continue
            blocks.append(f"### {message.role.value}\n" + "\n".join(parts))
        return "\n\n".join(blocks)
