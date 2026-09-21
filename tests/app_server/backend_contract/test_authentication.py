"""Provider authentication errors and credential-free session listing."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
import respx

from chartreux.app_server.client import AppServerClient
from chartreux.app_server.models import TurnErrorCode
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ProtocolErrorCode,
    SessionListParams,
    SessionListResponse,
    SessionOptions,
    SessionStartParams,
)
from chartreux.app_server.session import AppServerSession, AppServerTurnError
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from tests.app_server.backend_contract.conftest import connect_backend_contract_client

# The sentence the legacy backend produces for a rejected credential. Derived
# from ``BackendError`` rather than typed out, so a reworded message fails here
# instead of leaving the two backends saying different things. The Harness
# holds its own copy of the string, pinned to this same source by
# ``tests/app_server/test_provider_credentials.py``.
_INVALID_API_KEY_MESSAGE = str(
    BackendError(
        provider="mistral",
        endpoint="/chat/completions",
        status=401,
        reason="Unauthorized",
        headers={},
        body_text="",
        parsed_error=None,
        model="mistral-vibe-cli-latest",
        payload_summary=PayloadSummary(
            model="mistral-vibe-cli-latest",
            message_count=1,
            approx_chars=0,
            temperature=0.0,
            has_tools=False,
            tool_choice=None,
        ),
    )
)


@pytest_asyncio.fixture
async def unauthenticated_client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AppServerClient]:
    """A connected client for a user with no resolvable Mistral key."""
    monkeypatch.delenv("MISTRAL_API_KEY")
    client = await connect_backend_contract_client(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    try:
        yield client
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_session_start_without_a_key_is_unauthorized(
    unauthenticated_client,
) -> None:
    """*Prepare*: A connected client whose provider has no resolvable key.
    *Do*: Ask for a session.
    *Assert*: A typed ``UNAUTHORIZED`` naming the provider.

    The code and the provider are what ACP reads to offer a sign-in; a
    configuration error here would send the user to edit ``config.toml``.
    """
    # Do
    with pytest.raises(AppServerResponseError) as exc_info:
        await unauthenticated_client.request("session/start", SessionStartParams())

    # Assert
    assert exc_info.value.error.code is ProtocolErrorCode.UNAUTHORIZED
    assert exc_info.value.error.data == {"provider": "mistral/default"}


@pytest.mark.asyncio
async def test_session_list_without_a_key_succeeds(unauthenticated_client) -> None:
    """*Prepare*: A connected client whose provider has no resolvable key.
    *Do*: List sessions.
    *Assert*: The listing answers.

    ``chartreux --resume`` has to work for a signed-out user: reading what is on
    disk needs no credential, and demanding one would hide their own history.
    """
    # Do
    response = SessionListResponse.model_validate(
        await unauthenticated_client.request("session/list", SessionListParams())
    )

    # Assert
    assert response.items == []


@pytest.mark.asyncio
async def test_a_provider_401_mid_turn_yields_the_legacy_message(
    backend_contract_mistral_api: respx.Route,
    backend_contract_session: AppServerSession,
) -> None:
    """*Prepare*: A provider that answers the completion with ``401``.
    *Do*: Run a turn.
    *Assert*: The turn fails carrying the sentence the legacy backend produces.

    The Unified path builds this message in the Harness, which cannot import
    Chartreux; this is the assertion that keeps the two copies the same sentence.

    Containment rather than equality: the legacy loop wraps a retried backend
    failure in ``API error from <provider> (model: <model>): ...``
    (``_loop.py:2969``) before it reaches the turn. Pinning the whole string
    would pin that wrapper too, and reproducing it on the Unified side means
    the Harness formatting a Chartreux model name it has no business knowing.
    """
    # Prepare
    backend_contract_mistral_api.mock(
        return_value=httpx.Response(401, json={"message": "Unauthorized"})
    )

    # Do
    with pytest.raises(AppServerTurnError) as exc_info:
        _ = [event async for event in backend_contract_session.act("hello")]

    # Assert
    assert exc_info.value.error.code == TurnErrorCode.INVALID_API_KEY
    assert "Invalid API key" in exc_info.value.error.message
    assert "Please check your API key and try again." in exc_info.value.error.message
