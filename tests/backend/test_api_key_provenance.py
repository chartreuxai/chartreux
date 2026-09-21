"""End-to-end credential provenance and retry-classification contracts."""

from __future__ import annotations

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from chartreux.app_server._runtime import RuntimeAuthenticationError
from chartreux.app_server._utils import public_error
from chartreux.app_server.models import TurnErrorCode
from chartreux.app_server.session import AppServerTurnError
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.llm.backend.generic import GenericBackend
from chartreux.core.llm.backend.mistral import MistralBackend
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.utils.api_keys import ApiKeyOrigin, ApiKeySource

_SECRET = "credential-value-must-not-appear"
_ENV_VAR = "CHARTREUX_PROVENANCE_TEST_KEY"
_PROVIDER = ProviderConfig(
    name="provenance-provider",
    api_base="https://provenance.example/v1",
    api_key_env_var=_ENV_VAR,
)
_MODEL = ModelConfig(name="provenance-model", provider=_PROVIDER.name, alias="test")
_MESSAGES = [LLMMessage(role=Role.user, content="hello")]


def _auth_error(
    *, status: int = 401, origin: ApiKeyOrigin | None = None
) -> BackendError:
    return BackendError(
        provider=_PROVIDER.name,
        endpoint="https://provenance.example/v1/chat/completions",
        status=status,
        reason="Unauthorized",
        headers={},
        body_text="",
        parsed_error=None,
        model=_MODEL.name,
        payload_summary=PayloadSummary(
            model=_MODEL.name,
            message_count=1,
            approx_chars=5,
            temperature=0.2,
            has_tools=False,
            tool_choice=None,
        ),
        api_key_origin=origin,
    )


async def _rejected_request(
    monkeypatch: pytest.MonkeyPatch, *, streaming: bool = False, status: int = 401
) -> BackendError:
    monkeypatch.setenv(_ENV_VAR, _SECRET)
    backend = GenericBackend(provider=_PROVIDER, retry_max_elapsed_time=0)
    with respx.mock(base_url="https://provenance.example") as mock_api:
        mock_api.post("/v1/chat/completions").mock(
            return_value=httpx.Response(status, json={"message": "rejected"})
        )
        with pytest.raises(BackendError) as raised:
            if streaming:
                _ = [
                    chunk
                    async for chunk in backend.complete_streaming(
                        model=_MODEL, messages=_MESSAGES
                    )
                ]
            else:
                await backend.complete(model=_MODEL, messages=_MESSAGES)
    return raised.value


@pytest.mark.asyncio
async def test_rejected_key_flows_from_producer_through_server_to_tui_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_error = await _rejected_request(monkeypatch)

    server_error = public_error(backend_error)
    turn_error = AppServerTurnError(server_error)
    app = MagicMock()
    app._retry_hint = ChartreuxApp._retry_hint
    app._tools_collapsed = False
    app._mount_and_scroll = AsyncMock()
    app.event_handler = MagicMock()

    message = ChartreuxApp._resolve_turn_error_message(app, turn_error)
    await ChartreuxApp._mount_turn_error(app, turn_error, message)

    assert backend_error.api_key_origin == ApiKeyOrigin(
        ApiKeySource.ENVIRONMENT, _ENV_VAR
    )
    assert server_error.code == TurnErrorCode.INVALID_API_KEY
    assert f"env var {_ENV_VAR}" in server_error.message
    assert "/retry" not in message
    app.event_handler.offer_retry.assert_not_called()
    app.event_handler.cancel_retry_presentation.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN])
async def test_streaming_rejected_key_wraps_status_and_origin(
    monkeypatch: pytest.MonkeyPatch, status: HTTPStatus
) -> None:
    error = await _rejected_request(monkeypatch, streaming=True, status=status)

    assert error.status == status
    assert error.api_key_origin == ApiKeyOrigin(ApiKeySource.ENVIRONMENT, _ENV_VAR)
    assert f"env var {_ENV_VAR}" in str(error)


def test_wrapped_rejected_error_retains_origin_at_server_boundary() -> None:
    origin = ApiKeyOrigin(ApiKeySource.ENVIRONMENT, _ENV_VAR)
    backend_error = _auth_error(origin=origin)
    wrapped = RuntimeError(f"API error from provider: {backend_error}")
    wrapped.__cause__ = backend_error

    server_error = public_error(wrapped)

    assert server_error.code == TurnErrorCode.INVALID_API_KEY
    assert f"env var {_ENV_VAR}" in server_error.message


@pytest.mark.asyncio
async def test_credential_diagnostics_never_include_the_key_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_error = await _rejected_request(monkeypatch)
    wrapped = RuntimeError(f"API error: {backend_error}")
    server_error = public_error(wrapped)

    assert _SECRET not in str(backend_error)
    assert _SECRET not in str(wrapped)
    assert _SECRET not in server_error.message
    assert f"env var {_ENV_VAR}" in server_error.message


def test_mistral_keeps_captured_credential_source_after_environment_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_ENV_VAR, _SECRET)
    backend = MistralBackend(provider=_PROVIDER)
    monkeypatch.delenv(_ENV_VAR)

    error = backend._attach_api_key_origin(_auth_error())

    assert backend._api_key == _SECRET
    assert error.api_key_origin == ApiKeyOrigin(ApiKeySource.ENVIRONMENT, _ENV_VAR)
    assert _SECRET not in str(error)
    assert f"env var {_ENV_VAR}" in str(error)


def test_missing_key_and_rejected_key_follow_distinct_error_paths() -> None:
    missing = RuntimeAuthenticationError(_PROVIDER.name)
    rejected = public_error(
        _auth_error(origin=ApiKeyOrigin(ApiKeySource.ENVIRONMENT, _ENV_VAR))
    )

    assert missing.classification is None
    assert str(missing) == f"Authentication is required for provider: {_PROVIDER.name}"
    assert rejected.code == TurnErrorCode.INVALID_API_KEY
    assert "Invalid API key" in rejected.message
    assert str(missing) != rejected.message
