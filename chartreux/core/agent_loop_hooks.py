"""Hook orchestration mixin for AgentLoop.

Provides pre_tool, post_tool, and post_agent hook lifecycle
methods. Extracted from the AgentLoop implementation module to keep it
focused on the core conversation loop and tool execution flow.

Explicit host seams (stage 14.3c): the mixin reads its host
dependencies through late-bound callbacks wired by AgentLoop.__init__,
never through host attributes directly, so post-construction
reassignment (hook reload, session reset, relocation) and instance
patches keep intercepting. The mixin stays a mixin; converting it to
a standalone hook service is separate work.

Callbacks:
    _hooks_manager_getter() -> HooksManager | None
    _hook_session_context() -> HookSessionContext
    _stats_getter() -> AgentStats
    _messages_getter() -> MessageList

Methods (host seams, satisfied by AgentLoop):
    _handle_tool_response(tool_call, text, persisted_result, images)
    _serialize_tool_input(tool_call) -> dict[str, Any]
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chartreux.core.events import ToolResultEvent
from chartreux.core.hooks.models import (
    HookEvent,
    HookTextReplacement,
    HookToolDenial,
    HookToolInputRewrite,
    HookUserMessage,
    PostAgentInvocation,
    PostToolInvocation,
    PreToolInvocation,
    ToolStatus,
)
from chartreux.core.llm.format import ResolvedToolCall
from chartreux.core.llm_models import ImageAttachment, PersistedToolResult
from chartreux.core.tools import secret_redaction
from chartreux.core.utils import (
    CANCELLATION_TAG,
    TOOL_ERROR_TAG,
    CancellationReason,
    get_user_cancellation_message,
)
from chartreux.observability.logging import logger
from chartreux.utils.tool_presentation import ToolResultPresentation

if TYPE_CHECKING:
    from chartreux.core.agent_loop import ToolDecision
    from chartreux.core.hooks.manager import HooksManager
    from chartreux.core.hooks.models import HookSessionContext
    from chartreux.core.llm_models import LLMMessage
    from chartreux.core.message_list import MessageList
    from chartreux.core.session_types import AgentStats


class _PreToolResolution(NamedTuple):
    # ``denial_event`` is non-None when the pipeline ended in a denial
    # (explicit or synthesized from a failed rewrite re-validation);
    # callers yield it and stop.  Otherwise tool_call / tool_input hold
    # the (possibly rewritten) values to use for permission + execution.
    tool_call: ResolvedToolCall
    tool_input: dict[str, Any]
    denial_event: ToolResultEvent | None


@dataclass(frozen=True, slots=True)
class PostToolFinalization:
    tool_input: dict[str, Any]
    tool_status: ToolStatus
    response_status: Literal["success", "failure", "skipped"]
    tool_output: dict[str, Any] | None = None
    tool_presentation: ToolResultPresentation | None = None
    images: list[ImageAttachment] | None = None
    tool_error: str | None = None
    duration_ms: float | None = None
    # Preserve the frozen seconds verbatim rather than round-tripping via ms.
    duration: float | None = None
    initial_text: str = ""
    # P2 (round-9): the executor-minted, run-scoped receipt occurrence ID,
    # minted BEFORE the invocation at the tool-call dispatch site and carried
    # here so the post-tool seam's receipt resolves to this invocation and
    # not another one reusing the same tool_call_id.
    occurrence_id: str | None = None


class CompletionEvidenceReceipt(BaseModel):
    """Typed completion-evidence receipt (P2): the side-channel payload.

    Emitted from the post-tool finalization seam only — never forwarded on
    the typed event stream, and never sent to the projector/outward path.
    One receipt per invocation that reached post-tool finalization;
    pre-invocation denials and skips never produced one (the tool never
    ran), and a hard cancellation outside the seam leaves the invocation's
    evidence derivable as unavailable — never contradictory — by the
    collector (P3), which registers the executor-minted occurrence ID
    unchanged from both this receipt and the result events.

    ``final_text`` is the post-hook final text under the recorder's
    redaction (the same ``secret_redaction`` pass that guards the transcript
    choke point, _transcript_recorder.py:233-259); ``executed_args`` is the
    effective executed-args signature — the serialized args the tool
    actually ran with, after any rewrite — under the same redaction;
    ``tool_output`` is the model-visible result serialization
    (_tool_executor.py's ``result_dict`` join) likewise redacted.
    ``bash_job_ids`` carries managed-job IDs from bash_start-shaped results
    (a nested ``job.job_id``) for lineage.
    """

    model_config = ConfigDict(extra="forbid")

    occurrence_id: str
    tool_call_id: str
    tool_name: str
    status: ToolStatus
    final_text: str
    executed_args: dict[str, Any] = Field(default_factory=dict)
    tool_output: dict[str, Any] | None = None
    bash_job_ids: tuple[str, ...] = ()
    duration_ms: float | None = None


def _redacted_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """A payload under the recorder's redaction (persisted-result shape)."""
    if payload is None:
        return None
    return secret_redaction.redact_persisted_result(
        PersistedToolResult(output=payload)
    ).output


def _bash_job_ids(output: dict[str, Any] | None) -> tuple[str, ...]:
    """Managed-job IDs from a bash_start-shaped result serialization."""
    if not output:
        return ()
    job = output.get("job")
    if isinstance(job, dict):
        job_id = job.get("job_id")
        if isinstance(job_id, str) and job_id:
            return (job_id,)
    return ()


def build_completion_evidence_receipt(
    tool_call: ResolvedToolCall, finalization: PostToolFinalization, final_text: str
) -> CompletionEvidenceReceipt:
    """Assemble one typed receipt from the post-tool finalization inputs.

    The executor supplies the effective executed-args signature and the
    model-visible serialization through ``finalization``; the seam owns the
    post-hook final text and the redaction pass.
    """
    return CompletionEvidenceReceipt(
        occurrence_id=finalization.occurrence_id or "",
        tool_call_id=tool_call.call_id,
        tool_name=tool_call.tool_name,
        status=finalization.tool_status,
        final_text=secret_redaction.redact(final_text),
        executed_args=_redacted_payload(dict(finalization.tool_input)) or {},
        tool_output=_redacted_payload(finalization.tool_output),
        bash_job_ids=_bash_job_ids(finalization.tool_output),
        duration_ms=finalization.duration_ms,
    )


class AgentLoopHooksMixin:
    """Mixin that adds hook orchestration to AgentLoop.

    See module docstring for the implicit contract with the host class.
    """

    # Explicit host seams (stage 14.3c): wired by AgentLoop.__init__ as
    # late-bound callbacks; set by AgentLoop.__init__, never read from host
    # attributes directly.
    _hooks_manager_getter: Callable[[], HooksManager | None]
    _hook_session_context: Callable[[], HookSessionContext]
    _stats_getter: Callable[[], AgentStats]
    _messages_getter: Callable[[], MessageList]
    # P2: the completion-evidence side-channel sink — a late-bound callback
    # receiving the typed receipt emitted from the post-tool finalization
    # seam, or None while no consumer (P3's accumulator) is registered.
    _completion_evidence_sink_getter: Callable[
        [], Callable[[CompletionEvidenceReceipt], None] | None
    ]

    def _handle_tool_response(
        self,
        tool_call: ResolvedToolCall,
        text: str,
        persisted_result: PersistedToolResult | None = None,
        images: list[ImageAttachment] | None = None,
    ) -> None: ...

    def _serialize_tool_input(self, tool_call: ResolvedToolCall) -> dict[str, Any]:
        return tool_call.validated_args.model_dump(mode="json", exclude_unset=True)

    # ------------------------------------------------------------------
    # Hook runners
    # ------------------------------------------------------------------

    async def _run_post_agent_hooks(
        self,
    ) -> AsyncGenerator[HookEvent | HookUserMessage]:
        manager = self._hooks_manager_getter()
        if not manager:
            return
        invocation = PostAgentInvocation(**self._hook_session_context().model_dump())
        async for ev in manager.run(invocation):
            if isinstance(ev, (HookEvent, HookUserMessage)):
                yield ev

    async def _run_pre_tool_hooks(
        self, tool_call: ResolvedToolCall, tool_input: dict[str, Any]
    ) -> AsyncGenerator[HookEvent | HookToolDenial | HookToolInputRewrite]:
        manager = self._hooks_manager_getter()
        if not manager:
            return
        invocation = PreToolInvocation(
            **self._hook_session_context().model_dump(),
            tool_name=tool_call.tool_name,
            tool_call_id=tool_call.call_id,
            tool_input=tool_input,
        )
        async for ev in manager.run(invocation):
            if isinstance(ev, (HookEvent, HookToolDenial, HookToolInputRewrite)):
                yield ev

    async def _run_post_tool_hooks(
        self,
        tool_call: ResolvedToolCall,
        *,
        tool_input: dict[str, Any],
        tool_status: ToolStatus,
        tool_output: dict[str, Any] | None = None,
        tool_error: str | None = None,
        duration_ms: float = 0.0,
        initial_text: str = "",
    ) -> AsyncGenerator[HookEvent | HookTextReplacement]:
        manager = self._hooks_manager_getter()
        if not manager:
            return
        invocation = PostToolInvocation(
            **self._hook_session_context().model_dump(),
            tool_name=tool_call.tool_name,
            tool_call_id=tool_call.call_id,
            tool_input=tool_input,
            tool_status=tool_status,
            tool_output=tool_output,
            tool_output_text=initial_text,
            tool_error=tool_error,
            duration_ms=duration_ms,
        )
        async for ev in manager.run(invocation):
            if isinstance(ev, (HookEvent, HookTextReplacement)):
                yield ev

    # ------------------------------------------------------------------
    # Post-tool collection helpers
    # ------------------------------------------------------------------

    async def _collect_post_tool_events(
        self, tool_call: ResolvedToolCall, **kwargs: Any
    ) -> tuple[str, list[HookEvent]]:
        """List-returning variant for shielded paths (cancel / exception)
        where an async generator cannot be iterated inline.
        """
        final_text: str = kwargs.get("initial_text", "")
        events: list[HookEvent] = []
        async for ev in self._run_post_tool_hooks(tool_call, **kwargs):
            if isinstance(ev, HookTextReplacement):
                final_text = ev.text
            elif isinstance(ev, HookEvent):
                events.append(ev)
        return final_text, events

    async def _run_post_tool_and_finalize(
        self, tool_call: ResolvedToolCall, finalization: PostToolFinalization
    ) -> AsyncGenerator[HookEvent]:
        """Run post-tool hooks, apply text replacements, and record the response.

        Yields ``HookEvent`` instances for the caller to forward to the UI.
        The final text (after any ``HookTextReplacement``) is passed to
        ``_handle_tool_response`` together with the given *response_status*
        and *decision*.
        """
        final_text = finalization.initial_text
        async for ev in self._run_post_tool_hooks(
            tool_call,
            tool_input=finalization.tool_input,
            tool_status=finalization.tool_status,
            tool_output=finalization.tool_output,
            tool_error=finalization.tool_error,
            duration_ms=finalization.duration_ms or 0.0,
            initial_text=finalization.initial_text,
        ):
            if isinstance(ev, HookTextReplacement):
                final_text = ev.text
            else:
                yield ev
        persisted_result = (
            PersistedToolResult(
                output=finalization.tool_output or {},
                duration=(
                    finalization.duration
                    if finalization.duration is not None
                    else finalization.duration_ms / 1000.0
                    if finalization.duration_ms is not None
                    else None
                ),
                cancelled=finalization.tool_status == "cancelled",
                presentation=finalization.tool_presentation,
            )
            if finalization.tool_output is not None
            or finalization.duration_ms is not None
            or finalization.duration is not None
            else None
        )
        self._handle_tool_response(
            tool_call, final_text, persisted_result, finalization.images
        )
        self._emit_completion_evidence(tool_call, finalization, final_text)

    def _emit_completion_evidence(
        self,
        tool_call: ResolvedToolCall,
        finalization: PostToolFinalization,
        final_text: str,
    ) -> None:
        """Side-channel emission of the typed completion-evidence receipt.

        Invoked from the post-tool finalization seam only; never yields an
        event, so the typed event-stream annotations (executor :143/:300,
        batch runner :99, loop :2691) stay unchanged and the receipt never
        reaches the projector/outward path. A sink failure is logged and
        swallowed: the receipt is side-channel evidence and must never break
        the tool flow it observes.
        """
        sink = self._completion_evidence_sink_getter()
        if sink is None or finalization.occurrence_id is None:
            return
        try:
            sink(build_completion_evidence_receipt(tool_call, finalization, final_text))
        except Exception:
            logger.warning(
                "Completion-evidence sink failed tool=%s tool_call_id=%s",
                tool_call.tool_name,
                tool_call.call_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Pre-tool pipeline
    # ------------------------------------------------------------------

    async def _run_pre_tool_pipeline(
        self, tool_call: ResolvedToolCall, tool_input: dict[str, Any]
    ) -> tuple[list[HookEvent], _PreToolResolution]:
        """Validate each rewrite as it arrives; first invalid one aborts the chain.

        Events are buffered (not streamed) because pre_tool hooks are
        gating checks expected to complete quickly.
        """
        events: list[HookEvent] = []
        async for ev in self._run_pre_tool_hooks(tool_call, tool_input):
            if isinstance(ev, HookToolDenial):
                return events, _PreToolResolution(
                    tool_call=tool_call,
                    tool_input=tool_input,
                    denial_event=self._handle_pre_tool_denial(tool_call, ev),
                )
            if isinstance(ev, HookToolInputRewrite):
                rewritten = self._apply_tool_input_rewrite(tool_call, ev)
                if isinstance(rewritten, HookToolDenial):
                    return events, _PreToolResolution(
                        tool_call=tool_call,
                        tool_input=tool_input,
                        denial_event=self._handle_pre_tool_denial(tool_call, rewritten),
                    )
                tool_call, tool_input = rewritten
                continue
            events.append(ev)

        return events, _PreToolResolution(
            tool_call=tool_call, tool_input=tool_input, denial_event=None
        )

    def _apply_tool_input_rewrite(
        self, tool_call: ResolvedToolCall, rewrite: HookToolInputRewrite
    ) -> tuple[ResolvedToolCall, dict[str, Any]] | HookToolDenial:
        """Re-validate a rewrite against the tool's args model.

        Rebuilds ``ResolvedToolCall``, patches the assistant message so the
        LLM sees the rewritten args next turn.  Returns a synthesized
        denial on validation failure.
        """
        tool_class = tool_call.tool_class
        args_model, _ = tool_class._get_tool_args_results()
        try:
            new_validated = args_model.model_validate(rewrite.tool_input)
        except ValidationError as e:
            logger.warning(
                "Hook %s produced invalid tool_input for '%s': %s",
                rewrite.hook_name,
                tool_call.tool_name,
                e,
            )
            return HookToolDenial(
                hook_name=rewrite.hook_name,
                content=(
                    f"Hook '{rewrite.hook_name}' rewrote tool_input but the"
                    f" result failed validation against"
                    f" {tool_call.tool_name}: {e}"
                ),
            )

        new_tool_call = tool_call.model_copy(update={"validated_args": new_validated})
        new_tool_input = self._serialize_tool_input(new_tool_call)
        self._patch_assistant_tool_call_args(tool_call.call_id, new_tool_input)
        return new_tool_call, new_tool_input

    def _patch_assistant_tool_call_args(
        self, call_id: str, new_args: dict[str, Any]
    ) -> None:
        """Mutate the assistant message's tool_calls so the transcript reflects
        what the tool actually ran with (not the model's original args).
        """
        if not call_id:
            return
        encoded = json.dumps(new_args)
        for message in reversed(self._messages_getter()):
            if not message.tool_calls:
                continue
            for tc in message.tool_calls:
                if tc.id == call_id:
                    tc.function.arguments = encoded
                    return

    def _handle_pre_tool_denial(
        self, tool_call: ResolvedToolCall, denial: HookToolDenial
    ) -> ToolResultEvent:
        self._stats_getter().tool_calls_hook_denied += 1
        denial_text = (
            f"<{TOOL_ERROR_TAG}>Tool '{tool_call.tool_name}' was denied by "
            f"hook '{denial.hook_name}': {denial.content}</{TOOL_ERROR_TAG}>"
        )
        self._handle_tool_response(tool_call, denial_text)
        return ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            skipped=True,
            skip_reason=denial_text,
            cancelled=False,
            tool_call_id=tool_call.call_id,
        )

    # ------------------------------------------------------------------
    # Skip / cancel helpers
    # ------------------------------------------------------------------

    async def _handle_tool_skip(
        self, tool_call: ResolvedToolCall, decision: ToolDecision
    ) -> AsyncGenerator[ToolResultEvent | HookEvent]:
        self._stats_getter().tool_calls_rejected += 1
        skip_reason = decision.feedback or str(
            get_user_cancellation_message(
                CancellationReason.TOOL_SKIPPED, tool_call.tool_name
            )
        )
        yield ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            skipped=True,
            skip_reason=skip_reason,
            cancelled=f"<{CANCELLATION_TAG}>" in skip_reason,
            tool_call_id=tool_call.call_id,
        )
        self._handle_tool_response(tool_call, skip_reason)

    async def _finalize_cancelled_tool(
        self,
        tool_call: ResolvedToolCall,
        tool_input: dict[str, Any],
        cancel_text: str,
        *,
        tool_started: bool,
        duration: float | None = None,
    ) -> AsyncGenerator[HookEvent]:
        """Shield post-tool hooks from cancellation so audit/redaction hooks
        still observe the cancelled call.  Yields ``HookEvent`` instances.

        Skips post_tool entirely when ``tool_started`` is False (cancel
        landed before the tool body ran — e.g. during the approval prompt).
        That matches the pre_tool denial path, which also doesn't fire
        post_tool: hooks never observe a phantom completion for a tool
        that never executed.
        """
        if not tool_started:
            self._handle_tool_response(tool_call, cancel_text)
            return
        persisted_result = (
            PersistedToolResult(output={}, duration=duration, cancelled=True)
            if duration is not None
            else None
        )
        collector = asyncio.create_task(
            self._collect_post_tool_events(
                tool_call,
                tool_input=tool_input,
                tool_status="cancelled",
                tool_error=cancel_text,
                duration_ms=duration * 1000.0 if duration is not None else 0.0,
                initial_text=cancel_text,
            )
        )
        while True:
            try:
                final_text, hook_events = await asyncio.shield(collector)
                break
            except asyncio.CancelledError:
                if collector.cancelled():
                    raise
        self._handle_tool_response(tool_call, final_text, persisted_result)
        for ev in hook_events:
            yield ev

    # ------------------------------------------------------------------
    # Post-turn hook dispatch
    # ------------------------------------------------------------------

    async def _dispatch_post_turn_hooks(
        self,
    ) -> tuple[LLMMessage | None, list[HookEvent]]:
        """Run post-agent-turn hooks and separate retry injection from events.

        Returns a ``(retry_message, events)`` tuple.  ``retry_message`` is
        an injected ``LLMMessage`` when a hook requests a retry, else ``None``.
        """
        from chartreux.core.llm_models import LLMMessage, Role

        events: list[HookEvent] = []
        retry_msg: LLMMessage | None = None
        async for hook_event in self._run_post_agent_hooks():
            if isinstance(hook_event, HookUserMessage):
                retry_msg = LLMMessage(
                    role=Role.user, content=hook_event.content, injected=True
                )
            else:
                events.append(hook_event)
        return retry_msg, events
