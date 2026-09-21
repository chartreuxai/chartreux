from __future__ import annotations


class RateLimitError(Exception):
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(
            "Rate limits exceeded. Please wait a moment before trying again."
        )


class ContextTooLongError(Exception):
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(
            "The conversation context exceeds the model's maximum limit. "
            "Use /rewind to undo recent actions, then /compact to summarize the conversation."
        )


class ResponseTooLongError(Exception):
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(
            "The model's response exceeded the maximum output token limit."
        )


class RefusalError(Exception):
    def __init__(
        self,
        provider: str,
        model: str,
        category: str | None = None,
        explanation: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.category = category
        self.explanation = explanation
        super().__init__(self._fmt())

    def _fmt(self) -> str:
        lead = "The model declined to respond to this request and stopped early."
        if self.category:
            lead += f" (category: {self.category})"
        detail = self.explanation or (
            "Try rephrasing your request or starting a new conversation."
        )
        return f"{lead} {detail}"
