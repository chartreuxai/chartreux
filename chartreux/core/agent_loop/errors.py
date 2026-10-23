from __future__ import annotations

from typing import ClassVar

from chartreux.core.llm.failures import FailureCategory


class AgentLoopError(Exception):
    """Base exception for AgentLoop errors."""


class AgentLoopStateError(AgentLoopError):
    """Raised when agent loop is in an invalid state."""


class AgentLoopLLMResponseError(AgentLoopError):
    """Raised when LLM response is malformed or missing expected data."""


class EmptyLLMResponseError(AgentLoopLLMResponseError):
    """Raised when the model returns no substantive text or tool calls.

    Typed transient (WP1): raised at the final-chunk boundary — the observed
    2026-10-17 main-agent terminal mode whose in-call empty replay also
    returned empty. Without a typed class the recovery layer's
    never-retry-unknown rule leaves this exact mode manual forever.
    """

    failure_category: ClassVar[FailureCategory] = FailureCategory.EMPTY_RESPONSE

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__("The model returned an empty assistant response.")


class ImagesNotSupportedError(AgentLoopError):
    """Raised when the active model does not support image attachments."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(model)


class EmptyReplayBudgetStop(EmptyLLMResponseError):
    """Internal signal: an empty response whose in-call replay was denied.

    Raised at the logical-call terminal handlers when the model returned an
    empty assistant response AND the synchronous replay admission denied the
    bounded replay. The payload carries the original empty-response error, so
    the provider/model provenance is retained; the loop re-evaluates the full
    middleware pipeline for the authoritative stop reason. The signal is an
    ordinary Exception — the completion coordinator's rollback still runs —
    and it fabricates no middleware result, budget reason, usage settlement,
    or pending state.

    It subclasses ``EmptyLLMResponseError`` (itself an ``AgentLoopError``) so
    every existing ``isinstance`` consumer — the compaction fallback, the
    terminal error projection, and the admission characterization tests —
    keeps treating it exactly like the empty response it replaces on the
    sideband paths the conversation-loop catch boundary does not cover.
    """

    failure_category: ClassVar[FailureCategory] = FailureCategory.EMPTY_RESPONSE

    def __init__(self, empty_error: EmptyLLMResponseError) -> None:
        self.empty_error = empty_error
        super().__init__(empty_error.provider, empty_error.model)
