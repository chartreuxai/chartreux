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
