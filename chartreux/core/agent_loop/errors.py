from __future__ import annotations


class AgentLoopError(Exception):
    """Base exception for AgentLoop errors."""


class AgentLoopStateError(AgentLoopError):
    """Raised when agent loop is in an invalid state."""


class AgentLoopLLMResponseError(AgentLoopError):
    """Raised when LLM response is malformed or missing expected data."""


class EmptyLLMResponseError(AgentLoopLLMResponseError):
    """Raised when the model returns no substantive text or tool calls."""

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__("The model returned an empty assistant response.")


class ImagesNotSupportedError(AgentLoopError):
    """Raised when the active model does not support image attachments."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(model)
