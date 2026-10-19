"""Review R3 — malformed tool-argument JSON is rejected, never substituted.

WP1 acceptance: a model tool call whose argument payload is not a JSON
object is REJECTED via the structured invalid-call path — never silently
replaced with ``{}`` and never invoked with synthesized arguments. Both
substitution sites are pinned:

- the response boundary (``APIToolFormatHandler.parse_message`` /
  ``resolve_tool_calls``) routes the call to ``FailedToolCall``, which the
  loop turns into a tagged tool-error result (``_emit_failed_tool_events``);
- the request boundary (``AnthropicMapper._convert_tool_call``) wraps the
  raw text under a reserved key instead of sending an empty object.

An ABSENT payload is different: it is a faithful no-args call (``{}``),
validated and invoked normally.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, ClassVar, cast

from pydantic import BaseModel

from chartreux.core.events import ToolStreamEvent
from chartreux.core.llm.backend.anthropic import AnthropicMapper
from chartreux.core.llm.format import APIToolFormatHandler
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
)


class _OptionalArgs(BaseModel):
    """Every field optional: the shape that used to be invoked with a
    silently synthesized ``{}`` when the model's JSON was malformed.
    """

    note: str | None = None


class _OptionalResult(BaseModel):
    ok: bool = True


class _OptionalArgsTool(
    BaseTool[_OptionalArgs, _OptionalResult, BaseToolConfig, BaseToolState]
):
    description = "A test tool whose arguments are all optional"

    async def run(
        self, args: _OptionalArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | _OptionalResult, None]:
        yield _OptionalResult()


class _OptionalToolManager:
    """Duck-typed stand-in for ToolManager (available_tools is all the
    resolver consults).
    """

    available_tools: ClassVar[dict[str, type[BaseTool]]] = {
        "optional": _OptionalArgsTool
    }


def _manager() -> Any:
    return cast(Any, _OptionalToolManager())


def _message(arguments: str | None) -> LLMMessage:
    return LLMMessage(
        role=Role.assistant,
        content=None,
        tool_calls=[
            ToolCall(
                id="call-1", function=FunctionCall(name="optional", arguments=arguments)
            )
        ],
    )


def test_malformed_arguments_are_rejected_not_invoked() -> None:
    """R3: malformed argument JSON takes the structured invalid-call path.

    Before: ``JSONDecodeError`` was swallowed and the call was invoked with
    a synthesized ``{}`` — an all-optional tool would run as if the model
    had sent an empty object. After: the call is a ``FailedToolCall`` and
    never reaches invocation.
    """
    handler = APIToolFormatHandler()
    parsed = handler.parse_message(_message('{"note": "unterminated'))

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].arguments_error is not None
    assert "Malformed tool arguments JSON" in parsed.tool_calls[0].arguments_error

    resolved = handler.resolve_tool_calls(parsed, _manager())
    assert resolved.tool_calls == []
    assert len(resolved.failed_calls) == 1
    failed = resolved.failed_calls[0]
    assert failed.tool_name == "optional"
    assert failed.call_id == "call-1"
    assert "Malformed tool arguments JSON" in failed.error


def test_non_object_arguments_are_rejected_not_invoked() -> None:
    """R3: well-formed JSON that is not an object is equally rejected."""
    handler = APIToolFormatHandler()
    parsed = handler.parse_message(_message('["not", "an", "object"]'))

    assert parsed.tool_calls[0].arguments_error is not None
    assert "JSON object" in parsed.tool_calls[0].arguments_error

    resolved = handler.resolve_tool_calls(parsed, _manager())
    assert resolved.tool_calls == []
    assert len(resolved.failed_calls) == 1


def test_absent_arguments_are_a_faithful_no_args_call() -> None:
    """An ABSENT payload is a faithful no-args call: ``{}``, validated and
    invoked — the substitution ban is about MALFORMED payloads only.
    """
    handler = APIToolFormatHandler()
    parsed = handler.parse_message(_message(None))

    assert parsed.tool_calls[0].arguments_error is None
    assert parsed.tool_calls[0].raw_args == {}

    resolved = handler.resolve_tool_calls(parsed, _manager())
    assert len(resolved.tool_calls) == 1
    assert resolved.failed_calls == []


def test_wellformed_arguments_still_resolve() -> None:
    """Control: a well-formed object payload resolves and validates."""
    handler = APIToolFormatHandler()
    parsed = handler.parse_message(_message('{"note": "hi"}'))

    assert parsed.tool_calls[0].arguments_error is None

    resolved = handler.resolve_tool_calls(parsed, _manager())
    assert len(resolved.tool_calls) == 1
    assert resolved.tool_calls[0].args_dict == {"note": "hi"}


def test_anthropic_mapper_never_sends_empty_object_for_malformed_args() -> None:
    """R3 (request boundary): a malformed payload is wrapped under a
    reserved key — the provider sees the failure, not synthesized args.
    """
    mapper = AnthropicMapper()
    block = mapper._convert_tool_call(
        ToolCall(
            id="call-1", function=FunctionCall(name="optional", arguments="{bad json")
        )
    )

    assert block["type"] == "tool_use"
    assert block["input"] == {"malformed_arguments": "{bad json"}


def test_anthropic_mapper_wraps_non_object_args() -> None:
    mapper = AnthropicMapper()
    block = mapper._convert_tool_call(
        ToolCall(
            id="call-1", function=FunctionCall(name="optional", arguments="[1, 2]")
        )
    )

    assert block["input"] == {"malformed_arguments": "[1, 2]"}


def test_anthropic_mapper_sends_empty_object_for_absent_args() -> None:
    """An ABSENT payload stays a faithful no-args call on the wire too."""
    mapper = AnthropicMapper()
    block = mapper._convert_tool_call(
        ToolCall(id="call-1", function=FunctionCall(name="optional", arguments=None))
    )

    assert block["input"] == {}


def test_anthropic_mapper_passes_wellformed_args_through() -> None:
    """Control: a well-formed object payload converts unchanged."""
    mapper = AnthropicMapper()
    block = mapper._convert_tool_call(
        ToolCall(
            id="call-1",
            function=FunctionCall(name="optional", arguments='{"note": "hi"}'),
        )
    )

    assert block["input"] == {"note": "hi"}
