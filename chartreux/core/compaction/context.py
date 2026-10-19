from __future__ import annotations

from collections.abc import Sequence
from html import escape
import re

from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.utils.tags import CancellationReason, get_user_cancellation_message
from chartreux.core.utils.tokens import approx_token_count, truncate_middle_to_tokens

COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000
_PREVIOUS_USER_MESSAGES_OPEN = "<previous_user_messages>"
_PREVIOUS_USER_MESSAGES_CLOSE = "</previous_user_messages>"
_COMPACTION_SUMMARY_OPEN = "<compaction_summary>"
_COMPACTION_SUMMARY_CLOSE = "</compaction_summary>"
_PREVIOUS_USER_MESSAGE_OPEN = "<previous_user_message>"
_PREVIOUS_USER_MESSAGE_CLOSE = "</previous_user_message>"
_RESERVED_PREVIOUS_USER_MESSAGE_TAGS = (
    _PREVIOUS_USER_MESSAGES_OPEN,
    _PREVIOUS_USER_MESSAGES_CLOSE,
    _PREVIOUS_USER_MESSAGE_OPEN,
    _PREVIOUS_USER_MESSAGE_CLOSE,
)
_PREVIOUS_USER_MESSAGE_RE = re.compile(
    rf"{re.escape(_PREVIOUS_USER_MESSAGE_OPEN)}\n(.*?)\n"
    rf"{re.escape(_PREVIOUS_USER_MESSAGE_CLOSE)}",
    re.DOTALL,
)

_SUMMARY_OPEN = "<summary>"
_SUMMARY_CLOSE = "</summary>"
_SUMMARY_RE = re.compile(
    rf"{re.escape(_SUMMARY_OPEN)}(.*?){re.escape(_SUMMARY_CLOSE)}", re.DOTALL
)

#: The message text of the synthetic TOOL_NO_RESPONSE placeholder the backfill
#: repair inserts for a tool call that never produced a result (PI1/T). The
#: predicate below is content + missing-structured-result — NEVER the
#: cancellation tag alone, so genuine TOOL_INTERRUPTED/TOOL_SKIPPED
#: settlements are never mistaken for placeholders.
_TOOL_NO_RESPONSE_TEXT = str(
    get_user_cancellation_message(CancellationReason.TOOL_NO_RESPONSE)
)


def is_synthetic_tool_placeholder(message: LLMMessage) -> bool:
    """Whether a tool message is a synthetic TOOL_NO_RESPONSE placeholder.

    A placeholder carries the TOOL_NO_RESPONSE message text and NO structured
    ``tool_result`` payload. A genuine settlement always keeps its content
    (and, when recorded with one, its structured result), so this predicate
    never matches a genuine result.
    """
    if message.role != Role.tool or message.tool_result is not None:
        return False
    content = message.content
    return isinstance(content, str) and _TOOL_NO_RESPONSE_TEXT in content


def extract_summary(text: str) -> str | None:
    match = _SUMMARY_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip() or None


def drop_oldest_round(messages: Sequence[LLMMessage]) -> list[LLMMessage] | None:
    """Copy ``messages`` without the oldest safely droppable round.

    A round is a run of user messages (the real turn plus any injected ones)
    together with everything they trigger — assistant replies and tool results
    — up to the next user message. The system prompt and most recent compaction
    boundary are protected: only complete rounds after that boundary may be
    dropped. Returns None when the protected prefix and most recent round are all
    that remain.
    """
    boundary_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if _is_compaction_context_message(messages[index])
        ),
        None,
    )
    prefix_end = boundary_index + 1 if boundary_index is not None else 1
    index = prefix_end
    while index < len(messages) and messages[index].role == Role.user:
        index += 1  # the round's opening user message(s)
    while index < len(messages) and messages[index].role != Role.user:
        index += 1  # the assistant replies and tool results they triggered
    if index >= len(messages):
        return None
    return [*messages[:prefix_end], *messages[index:]]


def render_compaction_context(
    previous_user_messages: Sequence[LLMMessage],
    summary: str,
    file_state_section: str | None = None,
) -> str:
    lines = [
        "You are continuing a trajectory after a context compaction.",
        "",
        "Here are some of the most recent previous user messages, preserved "
        "verbatim where possible. Treat them as prior context, not as new requests.",
        "",
        _PREVIOUS_USER_MESSAGES_OPEN,
    ]
    for message in previous_user_messages:
        content = _escape_reserved_previous_user_message_tags(message.content or "")
        lines.append(
            f"{_PREVIOUS_USER_MESSAGE_OPEN}\n{content}\n{_PREVIOUS_USER_MESSAGE_CLOSE}"
        )
    lines.extend([
        _PREVIOUS_USER_MESSAGES_CLOSE,
        "",
        "Here is a summary of what has happened so far:",
        "",
        _COMPACTION_SUMMARY_OPEN,
        summary,
        _COMPACTION_SUMMARY_CLOSE,
    ])
    if file_state_section:
        # The working-set file state rides the envelope structurally (replayed
        # on resume), but the envelope is not model-visible — the backend
        # drops tool_result from the converted request. The bounded
        # working-set summary is therefore rendered into the boundary content,
        # so the model sees the working-set paths and their revisions right
        # after compaction, without re-reading the files.
        lines.extend(["", file_state_section])
    return "\n".join(lines)


def _escape_reserved_previous_user_message_tags(content: str) -> str:
    for tag in _RESERVED_PREVIOUS_USER_MESSAGE_TAGS:
        content = content.replace(tag, escape(tag, quote=False))
    return content


def parse_previous_user_messages(content: str) -> list[str]:
    block_start = content.find(_PREVIOUS_USER_MESSAGES_OPEN)
    if block_start < 0:
        return []

    block_start += len(_PREVIOUS_USER_MESSAGES_OPEN)
    block_end = content.find(_PREVIOUS_USER_MESSAGES_CLOSE, block_start)
    if block_end < 0:
        return []

    block = content[block_start:block_end]
    return [match.group(1) for match in _PREVIOUS_USER_MESSAGE_RE.finditer(block)]


def _is_compaction_context_message(message: LLMMessage) -> bool:
    if message.context_boundary == "compaction":
        return True

    content = message.content or ""
    return (
        message.role == Role.user
        and message.injected
        and _PREVIOUS_USER_MESSAGES_OPEN in content
        and _PREVIOUS_USER_MESSAGES_CLOSE in content
        and _COMPACTION_SUMMARY_OPEN in content
        and _COMPACTION_SUMMARY_CLOSE in content
    )


def select_model_context(messages: Sequence[LLMMessage]) -> list[LLMMessage]:
    boundary_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if _is_compaction_context_message(messages[index])
        ),
        None,
    )
    if boundary_index is None:
        return list(messages)

    system_messages = [
        message for message in messages[:boundary_index] if message.role == Role.system
    ]
    return [*system_messages, *messages[boundary_index:]]


#: A call id only becomes a duplicate candidate once at least two results
#: exist for it (mirrors ``_MIN_RESULTS_FOR_DUPLICATE`` in the integrity
#: module — the same owner-scoped pattern).
_MIN_RESULTS_FOR_DUPLICATE = 2


def _tool_result_owner_groups(
    messages: Sequence[LLMMessage],
) -> dict[tuple[int, str], list[int]]:
    """Group tool-result positions by (owning assistant index, tool_call_id).

    ONE O(n) pass. Call identity is scoped PER ORIGINATING ASSISTANT CALL
    (PI1/T): tool-call ids are reused across turns, so a placeholder under
    one assistant call is only shadowed by a genuine result in the SAME
    call's group. The owner of a result is the last assistant message
    carrying the call id BEFORE the result; a result no earlier assistant
    call carries is unclaimed (result-before-call included).
    """
    last_owner: dict[str, int] = {}
    groups: dict[tuple[int, str], list[int]] = {}
    for index, message in enumerate(messages):
        if message.role == Role.assistant:
            for tool_call in message.tool_calls or []:
                if tool_call.id:
                    last_owner[tool_call.id] = index
        elif message.role == Role.tool and message.tool_call_id:
            owner = last_owner.get(message.tool_call_id)
            if owner is not None:
                groups.setdefault((owner, message.tool_call_id), []).append(index)
    return groups


def shadowed_placeholder_positions(messages: Sequence[LLMMessage]) -> frozenset[int]:
    """Positions of the synthetic placeholders shadowed by a genuine result.

    Owner-scoped (PI1/T, round-1 S1): a placeholder is shadowed only when
    its OWN (originating assistant call, tool_call_id) group holds exactly
    one genuine result besides it — the unambiguous PI1 pattern. A
    placeholder whose call id is merely REUSED by a later assistant turn
    (its own group holds no genuine result) is NOT shadowed: the later
    turn's result answers the later call, and the placeholder is the
    earlier dangling call's designed backfill — dropping it would strand
    the later call unanswered in the provider-bound order (the round-1 S1
    provider 400 shape).

    This is the single owner-scoped predicate shared by the request-build
    reorder (:func:`_drop_shadowed_placeholders` below) and the integrity
    module's dedupe, so the validated projection and the emitted projection
    cannot diverge.
    """
    drop: set[int] = set()
    for indexes in _tool_result_owner_groups(messages).values():
        # The unambiguous PI1 pattern: at least two results for one call,
        # exactly one of them genuine (repeated amplifier runs included).
        if len(indexes) < _MIN_RESULTS_FOR_DUPLICATE:
            continue
        genuine = [
            position
            for position in indexes
            if not is_synthetic_tool_placeholder(messages[position])
        ]
        if len(genuine) == 1:
            drop.update(
                position
                for position in indexes
                if is_synthetic_tool_placeholder(messages[position])
            )
    return frozenset(drop)


def _drop_shadowed_placeholders(messages: Sequence[LLMMessage]) -> list[LLMMessage]:
    """Drop synthetic placeholders shadowed by a genuine result (PI1/T).

    The reorder's pending set empties at the FIRST matching result, so a
    synthetic TOOL_NO_RESPONSE placeholder sitting in front of the genuine
    result made the order unrepairable (the poisoned ``[assistant,
    placeholder, user, genuine result]`` shape reached the provider as
    ``[user, tool]``). Dropping the shadowed placeholders first — keeping the
    genuine structured result — restores the repairable ``[assistant, user,
    genuine result]`` shape. Genuine results are never dropped, and the pass
    is a no-op for transcripts without the pattern.

    The drop is scoped PER ORIGINATING ASSISTANT CALL (round-1 S1): a
    placeholder is dropped only when a genuine result exists in its own
    call's group. Scoping by call id alone dropped a placeholder that
    answered an EARLIER dangling call while a later turn reused the same
    call id, stranding the later call unanswered in the emitted projection.
    """
    drop = shadowed_placeholder_positions(messages)
    if not drop:
        return list(messages)
    return [
        message for position, message in enumerate(messages) if position not in drop
    ]


def reorder_for_tool_adjacency(messages: Sequence[LLMMessage]) -> list[LLMMessage]:
    """Move messages that fell between a tool_use and its tool_result.

    Providers such as Anthropic require every assistant ``tool_use`` to be
    followed immediately by the matching ``tool_result``. A message steered
    into a running turn while a tool call is in flight can land in that gap --
    on its own, or (with skill/file injection) as a nested tool_use/tool_result
    pair. Only a tool_result whose id matches the open call satisfies that
    call; everything else in the gap is displaced together to just after the
    call's results, so nested pairs are moved intact rather than split.

    PI1/T: synthetic TOOL_NO_RESPONSE placeholders shadowed by a genuine
    result are dropped first (the duplicate blindness fix) — the genuine
    structured result is kept and satisfies its call. Order-stable and pure,
    so it is safe to run on every outbound request.
    """
    messages = _drop_shadowed_placeholders(messages)
    result: list[LLMMessage] = []
    index = 0
    total = len(messages)
    while index < total:
        message = messages[index]
        result.append(message)
        index += 1
        if message.role != Role.assistant or not message.tool_calls:
            continue
        pending = {tc.id for tc in message.tool_calls if tc.id is not None}
        if not pending:
            continue
        displaced: list[LLMMessage] = []
        while index < total and pending:
            following = messages[index]
            index += 1
            call_id = following.tool_call_id
            if following.role == Role.tool and call_id in pending:
                result.append(following)
                pending.discard(call_id)
            else:
                displaced.append(following)
        # Recurse so a nested pair displaced from this gap is itself ordered.
        result.extend(reorder_for_tool_adjacency(displaced))
    return result


def collect_prior_user_messages(
    messages: list[LLMMessage],
    summary_prefix: str,
    max_tokens: int = COMPACT_USER_MESSAGE_MAX_TOKENS,
) -> list[LLMMessage]:
    """Pick user messages to preserve through compaction.

    Walks newest-first within a token budget, dropping system-internal
    injections and prior compaction summaries, middle-truncating the message
    that spills over. Previously preserved user messages are parsed from the
    compaction context envelope and merged with newer real user turns.
    """
    candidates: list[str] = []
    for message in messages:
        content = message.content or ""
        if not content or message.role != Role.user:
            continue

        if _is_compaction_context_message(message):
            candidates.extend(parse_previous_user_messages(content))
            continue

        if message.injected and content.startswith(summary_prefix):
            continue

        if message.injected:
            continue

        candidates.append(content)

    selected: list[LLMMessage] = []
    remaining = max_tokens
    for content in reversed(candidates):
        if remaining <= 0:
            break
        cost = approx_token_count(content)
        if cost <= remaining:
            selected.append(LLMMessage(role=Role.user, content=content, injected=True))
            remaining -= cost
        else:
            truncated = truncate_middle_to_tokens(content, remaining)
            selected.append(
                LLMMessage(role=Role.user, content=truncated, injected=True)
            )
            remaining = 0

    selected.reverse()
    return selected
