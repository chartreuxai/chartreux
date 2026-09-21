"""Structured ACP error classes for the Chartreux agent.

Error codes follow JSON-RPC 2.0 (https://www.jsonrpc.org/specification#error_object)
and ACP error handling (https://agentclientprotocol.com/protocol/overview#error-handling):

  -32700            Parse error (JSON-RPC standard)
  -32600            Invalid request (JSON-RPC standard)
  -32601            Method not found (JSON-RPC standard)
  -32602            Invalid params (JSON-RPC standard)
  -32603            Internal error (JSON-RPC standard)
  -32000 to -32099  Server errors (JSON-RPC implementation-defined)
  -31xxx            Application errors (Chartreux-specific, outside reserved range)

Chartreux application codes:
  -31001            Rate limited
  -31002            Configuration error
  -31003            Conversation limit
  -31004            Context too long
  -31005            Refusal
  -31006            Compaction failed
  -31007            Invalid image attachment
  -31008            Images not supported by the active model
"""

from __future__ import annotations

from typing import Any

from acp import RequestError

from chartreux.app_server.models import PublicError, TurnErrorCode
from chartreux.app_server.protocol import ProtocolError, ProtocolErrorCode

# JSON-RPC 2.0 standard codes
UNAUTHENTICATED = -32000
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Chartreux application codes (outside JSON-RPC reserved range)
RATE_LIMITED = -31001
CONFIGURATION_ERROR = -31002
CONVERSATION_LIMIT = -31003
CONTEXT_TOO_LONG = -31004
REFUSAL = -31005
COMPACTION_FAILED = -31006
INVALID_IMAGE_ATTACHMENT = -31007
IMAGES_NOT_SUPPORTED = -31008


class ChartreuxRequestError(RequestError):
    code: int

    def __init__(self, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(self.code, message, data)


class UnauthenticatedError(ChartreuxRequestError):
    code = UNAUTHENTICATED

    def __init__(self, detail: str, *, provider_name: str | None = None) -> None:
        self.provider_name = provider_name
        data = {"provider": provider_name} if provider_name is not None else None
        super().__init__(message=detail, data=data)

    @classmethod
    def for_provider(cls, provider_name: str) -> UnauthenticatedError:
        return cls(
            f"Missing API key for {provider_name} provider.",
            provider_name=provider_name,
        )


class NotImplementedMethodError(ChartreuxRequestError):
    code = METHOD_NOT_FOUND

    def __init__(self, method: str) -> None:
        super().__init__(
            message=f"Method not implemented: {method}", data={"method": method}
        )


class InvalidRequestError(ChartreuxRequestError):
    code = INVALID_PARAMS

    def __init__(self, detail: str) -> None:
        super().__init__(message=detail)


class SessionNotFoundError(ChartreuxRequestError):
    code = INVALID_PARAMS

    def __init__(self, session_id: str) -> None:
        super().__init__(
            message=f"Session not found: {session_id}", data={"session_id": session_id}
        )


class SessionLoadError(ChartreuxRequestError):
    code = INVALID_PARAMS

    def __init__(self, session_id: str, detail: str) -> None:
        super().__init__(
            message=f"Failed to load session {session_id}: {detail}",
            data={"session_id": session_id},
        )


class RateLimitError(ChartreuxRequestError):
    code = RATE_LIMITED

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(
            message=f"Rate limit exceeded for {provider} (model: {model}).",
            data={"provider": provider, "model": model},
        )


class ContextTooLongError(ChartreuxRequestError):
    code = CONTEXT_TOO_LONG

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(
            message=f"Context too long for {provider} (model: {model}). "
            "Use /rewind to undo recent actions, then /compact to summarize.",
            data={"provider": provider, "model": model},
        )


class RefusalError(ChartreuxRequestError):
    code = REFUSAL

    def __init__(
        self,
        provider: str,
        model: str,
        category: str | None = None,
        explanation: str | None = None,
    ) -> None:
        category_suffix = f" (category: {category})" if category else ""
        detail = explanation or (
            "Try rephrasing your request or starting a new conversation."
        )
        super().__init__(
            message=f"The model declined to respond for {provider} "
            f"(model: {model}){category_suffix}. {detail}",
            data={
                "provider": provider,
                "model": model,
                "category": category,
                "explanation": explanation,
            },
        )


class CompactionError(ChartreuxRequestError):
    code = COMPACTION_FAILED

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(message=detail, data={"reason": reason})


class ConfigurationError(ChartreuxRequestError):
    code = CONFIGURATION_ERROR

    def __init__(self, detail: str) -> None:
        super().__init__(message=detail)


class InvalidImageAttachmentError(ChartreuxRequestError):
    code = INVALID_IMAGE_ATTACHMENT

    def __init__(self, detail: str, reason: str) -> None:
        super().__init__(message=detail, data={"reason": reason})


class ImagesNotSupportedError(ChartreuxRequestError):
    code = IMAGES_NOT_SUPPORTED

    def __init__(self, model: str) -> None:
        super().__init__(
            message=f"Model `{model}` does not support images. "
            f"Switch model, or ask me to enable the support for this model."
        )


class ConversationLimitError(ChartreuxRequestError):
    code = CONVERSATION_LIMIT

    def __init__(self, detail: str) -> None:
        super().__init__(message=detail)


class InternalError(ChartreuxRequestError):
    code = INTERNAL_ERROR

    def __init__(self, detail: str) -> None:
        super().__init__(message=detail or "Internal error")


def from_public_error(error: PublicError) -> ChartreuxRequestError:
    details = error.details if isinstance(error.details, dict) else {}
    provider = details.get("provider")
    model = details.get("model")
    reason = details.get("reason")
    match error.code:
        case TurnErrorCode.RATE_LIMIT if isinstance(provider, str) and isinstance(
            model, str
        ):
            mapped: ChartreuxRequestError = RateLimitError(provider, model)
        case TurnErrorCode.CONTEXT_TOO_LONG if isinstance(provider, str) and isinstance(
            model, str
        ):
            mapped = ContextTooLongError(provider, model)
        case TurnErrorCode.RESPONSE_TOO_LONG:
            mapped = ConversationLimitError(error.message)
        case TurnErrorCode.REFUSAL if isinstance(provider, str) and isinstance(
            model, str
        ):
            category = details.get("category")
            explanation = details.get("explanation")
            mapped = RefusalError(
                provider,
                model,
                category if isinstance(category, str) else None,
                explanation if isinstance(explanation, str) else None,
            )
        case TurnErrorCode.INVALID_API_KEY:
            mapped = UnauthenticatedError(
                error.message,
                provider_name=provider if isinstance(provider, str) else None,
            )
        case TurnErrorCode.INVALID_IMAGE_ATTACHMENT:
            mapped = InvalidImageAttachmentError(
                error.message, reason=TurnErrorCode.INVALID_IMAGE_ATTACHMENT.value
            )
        case TurnErrorCode.IMAGES_NOT_SUPPORTED if isinstance(model, str):
            mapped = ImagesNotSupportedError(model)
        case TurnErrorCode.COMPACTION_FAILED if isinstance(reason, str):
            mapped = CompactionError(reason, error.message)
        case TurnErrorCode.INVALID_MODEL:
            mapped = ConfigurationError(error.message)
        case _:
            mapped = InternalError(error.message)
    return mapped


def from_app_server_error(error: ProtocolError) -> ChartreuxRequestError:
    details = error.data if isinstance(error.data, dict) else {}
    reason = details.get("reason")
    match error.code:
        case ProtocolErrorCode.COMPACTION_FAILED if isinstance(reason, str):
            return CompactionError(reason, error.message)
        case _:
            return InternalError(error.message)
