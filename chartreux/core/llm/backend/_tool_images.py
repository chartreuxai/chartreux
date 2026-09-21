from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from chartreux.core.llm_models import LLMMessage, Role


def has_tool_images(messages: Sequence[LLMMessage]) -> bool:
    """Return whether a projection is needed for this request."""
    return any(msg.role == Role.tool and msg.images for msg in messages)


def project_tool_images(  # noqa: PLR0912 - explicit fail-closed malformed-batch cases
    messages: Sequence[LLMMessage],
) -> list[LLMMessage]:
    """Project tool-result images into user turns for backends without tool images.

    A synthetic turn is emitted only after every identified result belonging to an
    assistant tool-call batch has arrived. This keeps incomplete histories from
    placing a user turn between outstanding tool results. An image-bearing result
    that cannot be associated with a complete batch raises rather than silently
    dropping its attachment.
    """
    if not has_tool_images(messages):
        return list(messages)

    projected: list[LLMMessage] = []
    outstanding_ids: Counter[str] | None = None
    batch_images: list[tuple[str, LLMMessage]] = []

    for msg in messages:
        if msg.role == Role.assistant:
            if batch_images:
                raise ValueError(
                    "Cannot faithfully project tool-result images: "
                    "tool-call batch is incomplete"
                )
            tool_call_ids = [call.id for call in msg.tool_calls or []]
            if tool_call_ids and all(call_id is not None for call_id in tool_call_ids):
                outstanding_ids = Counter(
                    call_id for call_id in tool_call_ids if call_id is not None
                )
            else:
                outstanding_ids = None
            batch_images = []
            projected.append(msg.model_copy(deep=True))
            continue

        if msg.role == Role.tool:
            if outstanding_ids is None or msg.tool_call_id is None:
                if msg.images:
                    detail = (
                        "image-bearing result has no associated tool-call batch"
                        if outstanding_ids is None
                        else "image-bearing result has no tool-call ID"
                    )
                    raise ValueError(
                        f"Cannot faithfully project tool-result images: {detail}"
                    )
                projected.append(msg.model_copy(deep=True, update={"images": None}))
                continue
            if not outstanding_ids.get(msg.tool_call_id, 0):
                if msg.images:
                    raise ValueError(
                        "Cannot faithfully project tool-result images: "
                        "image-bearing result has unexpected tool-call ID "
                        f"{msg.tool_call_id!r}"
                    )
                projected.append(msg.model_copy(deep=True, update={"images": None}))
                continue

            projected.append(msg.model_copy(deep=True, update={"images": None}))
            outstanding_ids[msg.tool_call_id] -= 1
            if outstanding_ids[msg.tool_call_id] == 0:
                del outstanding_ids[msg.tool_call_id]
            if msg.images:
                batch_images.append((msg.tool_call_id, msg))
            if not outstanding_ids:
                if batch_images:
                    images = [
                        image
                        for _, result in batch_images
                        for image in result.images or []
                    ]
                    labels = [
                        f"Image from tool call {tool_call_id}"
                        for tool_call_id, result in batch_images
                        for _ in result.images or []
                    ]
                    projected.append(
                        LLMMessage(
                            role=Role.user, content="\n".join(labels), images=images
                        )
                    )
                outstanding_ids = None
                batch_images = []
            continue

        # A non-tool message cannot follow an incomplete image-bearing batch.
        if batch_images:
            raise ValueError(
                "Cannot faithfully project tool-result images: "
                "tool-call batch is incomplete"
            )
        outstanding_ids = None
        batch_images = []
        projected.append(msg.model_copy(deep=True))

    if batch_images:
        raise ValueError(
            "Cannot faithfully project tool-result images: tool-call batch is incomplete"
        )
    return projected
