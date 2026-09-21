from __future__ import annotations


class AgentLoopError(Exception):
    """Base exception for AgentLoop errors."""


class AgentLoopStateError(AgentLoopError):
    """Raised when agent loop is in an invalid state."""


class AgentLoopLLMResponseError(AgentLoopError):
    """Raised when LLM response is malformed or missing expected data."""


class ImagesNotSupportedError(AgentLoopError):
    """Raised when the active model does not support image attachments."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(model)
