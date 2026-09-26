from __future__ import annotations

from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall


def synthetic_messages(approx_tokens: int, n_messages: int) -> list[LLMMessage]:
    """Build a chars/4 token-estimate transcript with displaced tool results.

    There is no local tokenizer available to this performance harness, so the
    requested token size is represented by exactly ``approx_tokens * 4`` content
    characters. Each complete five-message turn contains an assistant tool call,
    an intervening user message, and its matching tool response; the projection
    path must reorder that response next to its call.
    """
    if approx_tokens < 1:
        raise ValueError("approx_tokens must be positive")
    if n_messages < 1:
        raise ValueError("n_messages must be positive")

    total_chars = approx_tokens * 4
    chars_per_message, remainder = divmod(total_chars, n_messages)
    messages: list[LLMMessage] = []

    for index in range(n_messages):
        char_count = chars_per_message + (index < remainder)
        prefix = f"transcript-{index:06d} "
        repetitions = (char_count + len(prefix) - 1) // len(prefix)
        content = (prefix * repetitions)[:char_count]
        position_in_turn = index % 5

        if index < n_messages - n_messages % 5:
            turn_number = index // 5
            call_id = f"synthetic-call-{turn_number:06d}"
            if position_in_turn == 0:
                message = LLMMessage(role=Role.user, content=content)
            elif position_in_turn == 1:
                message = LLMMessage(
                    role=Role.assistant,
                    content=content,
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            index=0,
                            function=FunctionCall(
                                name="synthetic_tool", arguments="{}"
                            ),
                        )
                    ],
                )
            elif position_in_turn == 2:
                message = LLMMessage(role=Role.user, content=content)
            elif position_in_turn == 3:
                message = LLMMessage(
                    role=Role.tool, content=content, tool_call_id=call_id
                )
            else:
                message = LLMMessage(role=Role.assistant, content=content)
        else:
            role = Role.user if position_in_turn % 2 == 0 else Role.assistant
            message = LLMMessage(role=role, content=content)

        messages.append(message)

    return messages
