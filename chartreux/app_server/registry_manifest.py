from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any, BinaryIO, Self

from chartreux.app_server.protocol import AgentTranscriptGetResponse
from chartreux.core.session.session_lease import SessionBusyError


@dataclass(frozen=True, slots=True)
class AgentRoute:
    connection_id: str
    incarnation_id: str


@dataclass(frozen=True, slots=True)
class CallbackTarget:
    """The agent and live route a delivered callback prompt belongs to."""

    agent_id: str
    route: AgentRoute


class LiveAgentUnavailableError(RuntimeError):
    pass


class LiveAgentExitedError(LiveAgentUnavailableError):
    """The agent's process exited; paging has no live route for it any more.

    The coordinator never reads transcript files for interactive paging, so a
    confirmed exit is a typed failure: the transcript is available again after
    a session reload, never through a silent disk fallback.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        super().__init__(
            f"Agent {agent_id} exited; transcript available after session reload"
        )


class RegistryManifestInitializationError(RuntimeError):
    pass


class SessionDeletedError(RuntimeError):
    pass


class SessionWriterFence:
    def __init__(self, session_dir: Path, session_id: str) -> None:
        self.path = session_dir / ".writer-fence.lock"
        self.session_id = session_id
        self._file: BinaryIO | None = None

    def acquire(self) -> Self:
        if self._file is not None:
            raise RuntimeError("session writer fence is already acquired")
        try:
            file = self.path.open("a+b")
        except FileNotFoundError as exc:
            raise SessionDeletedError(
                f"Session directory unavailable: {self.path.parent}"
            ) from exc
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            file.close()
            raise SessionBusyError(self.session_id) from exc
        except BaseException:
            file.close()
            raise
        self._file = file
        return self

    def release(self) -> None:
        file = self._file
        if file is not None:
            self._file = None
            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            finally:
                file.close()

    def __enter__(self) -> Self:
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()


PageReader = Callable[[str, str | None, int], Awaitable[AgentTranscriptGetResponse]]


class RegistryManifest:
    def __init__(self, session_dir: Path) -> None:
        self.path = session_dir.parent / f"{session_dir.name}.registry.json"
        self.fence_path = session_dir / ".writer-fence.lock"
        self._agents: dict[str, AgentRoute] = {}
        self._callbacks: dict[str, str] = {}
        self._retired_callbacks: set[str] = set()
        self._readers: dict[str, PageReader] = {}
        # WP5 [5D]: policy-tree transaction records live in this manifest
        # (the existing persistent artifact), so a crashed transaction is
        # reconciled from the same file a reload already reads. Loading is
        # lazy: construction performs no IO, and a missing or unusable file
        # simply carries no records.
        self._policy_transactions: dict[str, dict[str, Any]] | None = None

    def initialize(self) -> None:
        try:
            with self.fence_path.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            pass
        previous = (
            self._agents,
            self._callbacks,
            self._retired_callbacks,
            self._readers,
        )
        self._agents, self._callbacks, self._readers = {}, {}, {}
        self._retired_callbacks = set()
        try:
            self._write()
        except OSError as exc:
            (self._agents, self._callbacks, self._retired_callbacks, self._readers) = (
                previous
            )
            raise RegistryManifestInitializationError(
                f"Cannot initialize registry manifest at {self.path}"
            ) from exc

    def register_agent(
        self, agent_id: str, route: AgentRoute, reader: PageReader | None = None
    ) -> None:
        previous_route = self._agents.get(agent_id)
        previous_reader = self._readers.get(agent_id)
        previous_callbacks = self._callbacks.copy()
        previous_retired = self._retired_callbacks.copy()
        self._agents[agent_id] = route
        if previous_route is not None and previous_route != route:
            self._retire_callbacks(agent_id)
        self._readers.pop(agent_id, None)
        if reader is not None:
            self._readers[agent_id] = reader
        try:
            self._write()
        except OSError:
            if previous_route is None:
                self._agents.pop(agent_id, None)
            else:
                self._agents[agent_id] = previous_route
            if previous_reader is None:
                self._readers.pop(agent_id, None)
            else:
                self._readers[agent_id] = previous_reader
            self._callbacks = previous_callbacks
            self._retired_callbacks = previous_retired
            raise

    def remove_agent(self, agent_id: str) -> None:
        agents, callbacks, retired, readers = (
            self._agents.copy(),
            self._callbacks.copy(),
            self._retired_callbacks.copy(),
            self._readers.copy(),
        )
        self._agents.pop(agent_id, None)
        self._readers.pop(agent_id, None)
        self._retire_callbacks(agent_id)
        try:
            self._write()
        except OSError:
            self._agents, self._callbacks, self._readers = agents, callbacks, readers
            self._retired_callbacks = retired
            raise

    def route_callback(self, callback_id: str, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise LiveAgentUnavailableError(agent_id)
        previous = self._callbacks.copy()
        was_retired = callback_id in self._retired_callbacks
        self._callbacks[callback_id] = agent_id
        self._retired_callbacks.discard(callback_id)
        try:
            self._write()
        except OSError:
            self._callbacks = previous
            if was_retired:
                self._retired_callbacks.add(callback_id)
            raise

    def remove_callback(self, callback_id: str) -> None:
        previous = self._callbacks.copy()
        self._callbacks.pop(callback_id, None)
        try:
            self._write()
        except OSError:
            self._callbacks = previous
            raise

    def callback_route(self, callback_id: str) -> AgentRoute:
        target = self.delivery_target(callback_id)
        if target is None:
            raise LiveAgentUnavailableError(callback_id)
        return target.route

    def delivery_target(self, callback_id: str) -> CallbackTarget | None:
        """The routed agent a callback prompt must be delivered for.

        ``None`` means the callback holds no routing row at all: an in-process
        callback, delivered from memory exactly as before. A callback whose
        row was retired — its agent's incarnation was replaced or removed —
        raises, so a prompt for an incarnation that can never answer it is not
        delivered. Re-routing the callback clears the retirement.
        """
        agent_id = self._callbacks.get(callback_id)
        if agent_id is None:
            if callback_id in self._retired_callbacks:
                raise LiveAgentUnavailableError(callback_id)
            return None
        route = self._agents.get(agent_id)
        if route is None:
            raise LiveAgentUnavailableError(callback_id)
        return CallbackTarget(agent_id=agent_id, route=route)

    def is_live(self, agent_id: str) -> bool:
        """Whether the agent currently holds a registered route."""
        return agent_id in self._agents

    async def read_live_page(
        self, agent_id: str, *, before: str | None = None, limit: int = 50
    ) -> AgentTranscriptGetResponse:
        """Serve one live page through the agent's registered page reader.

        An agent with no registered route has exited (or was never live): the
        failure is typed, and its transcript is available after a session
        reload rather than through a coordinator disk read.
        """
        if agent_id not in self._agents:
            raise LiveAgentExitedError(agent_id)
        reader = self._readers.get(agent_id)
        if reader is None:
            raise LiveAgentUnavailableError(agent_id)
        return await reader(agent_id, before, limit)

    def _retire_callbacks(self, agent_id: str) -> None:
        """Drop an agent's callback rows and remember them as retired."""
        retired = {
            callback for callback, owner in self._callbacks.items() if owner == agent_id
        }
        self._retired_callbacks |= retired
        self._callbacks = {
            callback: owner
            for callback, owner in self._callbacks.items()
            if owner != agent_id
        }

    def record_policy_transaction(
        self, transaction_id: str, payload: Mapping[str, Any]
    ) -> None:
        """Persist one policy-tree transaction record atomically."""
        rows = self._policy_rows()
        previous = rows.get(transaction_id)
        rows[transaction_id] = dict(payload)
        try:
            self._write()
        except OSError:
            if previous is None:
                rows.pop(transaction_id, None)
            else:
                rows[transaction_id] = previous
            raise

    def policy_transaction(self, transaction_id: str) -> dict[str, Any] | None:
        """Read one policy-tree transaction record back, if it exists."""
        payload = self._policy_rows().get(transaction_id)
        return None if payload is None else dict(payload)

    def _policy_rows(self) -> dict[str, dict[str, Any]]:
        if self._policy_transactions is None:
            self._policy_transactions = self._read_policy_transactions()
        return self._policy_transactions

    def _read_policy_transactions(self) -> dict[str, dict[str, Any]]:
        """Reload surviving policy records so a crash is reconcilable."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return {}
        if not isinstance(payload, Mapping):
            return {}
        rows = payload.get("policy_tree")
        if not isinstance(rows, Mapping):
            return {}
        return {
            str(transaction_id): dict(record)
            for transaction_id, record in rows.items()
            if isinstance(record, Mapping)
        }

    def _write(self) -> None:
        data: dict[str, Any] = {
            "agents": {
                agent_id: {
                    "connection_id": route.connection_id,
                    "incarnation_id": route.incarnation_id,
                }
                for agent_id, route in self._agents.items()
            },
            "callbacks": self._callbacks,
        }
        if self._policy_transactions is None:
            # Preserve records written by an earlier process before the first
            # write of this one; initialize() rewrites the whole file.
            self._policy_rows()
        if self._policy_transactions:
            data["policy_tree"] = self._policy_transactions
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(json.dumps(data, sort_keys=True) + "\n")
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
