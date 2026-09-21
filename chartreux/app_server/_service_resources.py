from __future__ import annotations

from chartreux.app_server._model import validate_wire
from chartreux.app_server.client_state import ClientSessionState
from chartreux.app_server.connection import AppServerResourceConnection
from chartreux.app_server.protocol import (
    NarrationSummarizeParams,
    NarrationSummarizeResponse,
)


class NarrationResource:
    def __init__(
        self, connection: AppServerResourceConnection, state: ClientSessionState
    ) -> None:
        self._connection = connection
        self._state = state

    async def summarize(
        self,
        *,
        user_message: str,
        assistant_text: str,
        error: str | None,
        message_id: str | None,
    ) -> str | None:
        client = await self._connection.connect()
        response = validate_wire(
            NarrationSummarizeResponse,
            await client.request(
                "narration/summarize",
                NarrationSummarizeParams(
                    session_id=self._state.session_id,
                    user_message=user_message,
                    assistant_text=assistant_text,
                    error=error,
                    message_id=message_id,
                ),
            ),
        )
        return response.summary
