from __future__ import annotations

from chartreux.acp.exceptions import (
    COMPACTION_FAILED,
    IMAGES_NOT_SUPPORTED,
    UNAUTHENTICATED,
    CompactionError,
    ImagesNotSupportedError,
    UnauthenticatedError,
    from_app_server_error,
    from_public_error,
)
from chartreux.app_server.models import PublicError, TurnErrorCode
from chartreux.app_server.protocol import ProtocolError, ProtocolErrorCode


def test_public_compaction_error_maps_to_acp_error() -> None:
    error = from_public_error(
        PublicError(
            code=TurnErrorCode.COMPACTION_FAILED,
            message="Compaction failed",
            details={"reason": "tool_call"},
        )
    )

    assert isinstance(error, CompactionError)
    assert error.code == COMPACTION_FAILED
    assert error.data == {"reason": "tool_call"}


def test_app_server_compaction_error_maps_to_acp_error() -> None:
    error = from_app_server_error(
        ProtocolError(
            code=ProtocolErrorCode.COMPACTION_FAILED,
            message="Compaction failed",
            data={"reason": "empty_summary"},
        )
    )

    assert isinstance(error, CompactionError)
    assert error.code == COMPACTION_FAILED
    assert error.data == {"reason": "empty_summary"}


def test_public_invalid_api_key_maps_to_unauthenticated_error() -> None:
    error = from_public_error(
        PublicError(
            code=TurnErrorCode.INVALID_API_KEY,
            message="Invalid API key",
            details={"provider": "mistral", "api_key": "secret"},
        )
    )

    assert isinstance(error, UnauthenticatedError)
    assert error.code == UNAUTHENTICATED
    assert error.data == {"provider": "mistral"}


def test_public_images_not_supported_maps_to_acp_error() -> None:
    error = from_public_error(
        PublicError(
            code=TurnErrorCode.IMAGES_NOT_SUPPORTED,
            message="model does not support images",
            details={"model": "text-model"},
        )
    )

    assert isinstance(error, ImagesNotSupportedError)
    assert error.code == IMAGES_NOT_SUPPORTED
