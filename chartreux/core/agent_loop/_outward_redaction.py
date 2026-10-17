from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Sequence

from chartreux.core.events import (
    BaseEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
)
from chartreux.core.hooks.models import HookEvent
from chartreux.core.llm.format import ResolvedToolCall
from chartreux.core.tools import secret_redaction
from chartreux.observability.logging import logger


class OutwardEventRedactor:
    """Own admission redaction snapshots and sanitize the outward event stream.

    Retains admitted names and resolved values only until outward emission,
    combining them with the live scrub policy at each per-event boundary.
    Execution policies stay separate from these redaction snapshots: a tool
    task is bound to the admission policy's names/passthrough only, while the
    resolved-value snapshot is never bound in a tool task or placed in a
    queue/event.
    """

    def __init__(
        self,
        *,
        scrub_policy_getter: Callable[[], secret_redaction.ScrubPolicy],
        record_tool_call_event: Callable[[ToolCallEvent], ToolCallEvent],
    ) -> None:
        # Live getter: the loop's scrub policy is re-read at every redaction
        # boundary, because reload and policy replacement reassign it (even
        # mid-turn, while tools are still running).
        self._scrub_policy_getter = scrub_policy_getter
        # Late-bound seam: instance patches of the loop's recording method
        # keep intercepting outward ToolCallEvent sanitation.
        self._record_tool_call_event = record_tool_call_event
        self._admitted_tool_policies: dict[str, secret_redaction.ScrubPolicy] = {}

    @property
    def admitted_tool_policies(self) -> dict[str, secret_redaction.ScrubPolicy]:
        """Read-only compatibility view of the retained admission snapshots."""
        return self._admitted_tool_policies

    def register_batch_admissions(
        self,
        tool_calls: Sequence[ResolvedToolCall],
        policy: secret_redaction.ScrubPolicy,
    ) -> None:
        """Resolve the admission policy's values once, at batch admission.

        The snapshot is retained per call ID until outward emission so a tool
        that outlives a reload still redacts against the values it was
        admitted with; it is never bound for execution.
        """
        redaction_snapshot = policy.capture_for_redaction()
        self._admitted_tool_policies.update(
            (tc.call_id, redaction_snapshot) for tc in tool_calls
        )

    def clear_admissions(self) -> None:
        """Drop retained admission snapshots (turn start and turn cleanup)."""
        self._admitted_tool_policies.clear()

    async def sanitize(
        self, events: AsyncGenerator[BaseEvent]
    ) -> AsyncGenerator[BaseEvent]:
        # Hold events behind the earliest unfinished stream so concurrent tool
        # completions cannot reorder the events seen by consumers.
        marker = "[Unchecked tool stream remainder suppressed]"
        pending: list[BaseEvent] = []
        streams: dict[str, list[ToolStreamEvent]] = {}
        stream_sizes: dict[str, int] = {}
        truncated: set[str] = set()
        async for event in events:
            ready: list[BaseEvent] = []
            with secret_redaction.bind_policy(
                secret_redaction.ScrubPolicy.for_redaction(
                    self._admitted_tool_policies.values(), self._scrub_policy_getter()
                )
            ):
                if isinstance(event, ToolStreamEvent):
                    call_id = event.tool_call_id
                    is_first = call_id not in streams
                    used = stream_sizes.get(call_id, 0)
                    remaining = max(0, 8192 * 128 - used)
                    if len(event.message) > remaining:
                        truncated.add(call_id)
                    if remaining and event.message:
                        streams.setdefault(call_id, []).append(
                            event.model_copy(
                                update={"message": event.message[:remaining]}
                            )
                        )
                    elif call_id not in streams:
                        streams[call_id] = [event.model_copy(update={"message": ""})]
                    stream_sizes[call_id] = used + min(len(event.message), remaining)
                    if is_first:
                        pending.append(streams[call_id][0])
                elif isinstance(event, ToolResultEvent):
                    chunks = streams.pop(event.tool_call_id, [])
                    stream_sizes.pop(event.tool_call_id, None)
                    if chunks:
                        replacement = self._sanitize_stream(
                            chunks, marker, event.tool_call_id in truncated
                        )
                        first = pending.index(chunks[0])
                        pending[first] = replacement
                    truncated.discard(event.tool_call_id)
                    pending.append(self._sanitize_outward_event(event))
                    self._admitted_tool_policies.pop(event.tool_call_id, None)
                else:
                    pending.append(self._sanitize_outward_event(event))
                while pending and not (
                    isinstance(pending[0], ToolStreamEvent)
                    and pending[0].tool_call_id in streams
                ):
                    ready.append(pending.pop(0))
            for item in ready:
                yield item
        # A stream with no result is still output; never expose its unchecked
        # remainder if the source generator ended unexpectedly.
        for item in pending:
            if isinstance(item, ToolStreamEvent) and item.tool_call_id in streams:
                with secret_redaction.bind_policy(
                    secret_redaction.ScrubPolicy.for_redaction(
                        self._admitted_tool_policies.values(),
                        self._scrub_policy_getter(),
                    )
                ):
                    item = item.model_copy(
                        update={
                            "message": marker,
                            "tool_name": secret_redaction.redact(item.tool_name),
                            "tool_call_id": secret_redaction.redact(item.tool_call_id),
                        }
                    )
            yield item

    @staticmethod
    def _sanitize_stream(
        chunks: list[ToolStreamEvent], marker: str, truncated: bool
    ) -> ToolStreamEvent:
        body = "".join(chunk.message for chunk in chunks)
        # Streams are buffered, so scan the entire retained body once. A window
        # cannot prove that a credential crossing its edge was checked (even at
        # the documented 16 KiB maximum); never emit independently checked
        # windows. The 1 MiB retention cap still bounds this whole-body scan.
        sanitized = secret_redaction.redact(body)
        if truncated:
            sanitized += marker
        return chunks[0].model_copy(
            update={
                "message": sanitized,
                "tool_name": secret_redaction.redact(chunks[0].tool_name),
                "tool_call_id": secret_redaction.redact(chunks[0].tool_call_id),
            }
        )

    def _sanitize_outward_event(self, event: BaseEvent) -> BaseEvent:
        """Sanitize tool and hook payloads before UI/ACP consumers see them."""
        if isinstance(event, ToolCallEvent):
            return self._record_tool_call_event(event)
        if not isinstance(event, (ToolResultEvent, HookEvent)):
            return event
        try:
            return secret_redaction.redact_model(event)
        except Exception:
            logger.warning(
                "Tool event sanitation failed; suppressing payload", exc_info=True
            )
            if isinstance(event, ToolResultEvent):
                return ToolResultEvent(
                    tool_name="tool",
                    tool_class=None,
                    tool_call_id=secret_redaction.redact(event.tool_call_id),
                    error="Tool result unavailable",
                    duration=event.duration,
                    cancelled=event.cancelled,
                )
            return HookEvent()
