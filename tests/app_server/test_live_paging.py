"""Live-agent transcript paging through the registry manifest page readers.

Producer emission is WP3/WP5 work, so these tests drive the coordinator side
against a test-local producer stand-in: it registers the manifest rows a
producer will register and answers pages from an in-memory message list.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.app_server._agent_transcript import read_live_agent_transcript
from chartreux.app_server._handler import CoreRequestHandler
from chartreux.app_server._session_backend_impl import LiveTranscriptPagesImpl
from chartreux.app_server._session_backend_port import LiveTranscriptSourceRefused
from chartreux.app_server._sessions import TranscriptReadSnapshot
from chartreux.app_server.protocol import (
    AgentTranscriptGetResponse,
    AgentTranscriptState,
    ProtocolErrorCode,
)
from chartreux.app_server.registry_manifest import (
    AgentRoute,
    PageReader,
    RegistryManifest,
)
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.subagents import UnknownAgentError


def _messages(*texts: str) -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.user, message_id=f"message-{index}", content=text)
        for index, text in enumerate(texts, start=1)
    ]


def _manifest(tmp_path: Path) -> RegistryManifest:
    session_dir = tmp_path / "sessions" / "root"
    session_dir.mkdir(parents=True)
    manifest = RegistryManifest(session_dir)
    manifest.initialize()
    return manifest


class _ProducerStandIn:
    """Test-local producer publishing live agents and their in-memory pages."""

    def __init__(self, manifest: RegistryManifest) -> None:
        self.manifest = manifest
        self._pages: dict[str, list[LLMMessage]] = {}
        self.calls: list[tuple[str, str | None, int]] = []

    def publish(
        self, agent_id: str, messages: list[LLMMessage], *, with_page: bool = True
    ) -> None:
        self._pages[agent_id] = messages
        reader = self._page_reader(agent_id) if with_page else None
        self.manifest.register_agent(
            agent_id, AgentRoute("connection-1", "incarnation-1"), reader
        )

    def retire(self, agent_id: str) -> None:
        self._pages.pop(agent_id, None)
        self.manifest.remove_agent(agent_id)

    def _page_reader(self, agent_id: str) -> PageReader:
        async def read(
            bound_agent_id: str, before: str | None, limit: int
        ) -> AgentTranscriptGetResponse:
            assert bound_agent_id == agent_id
            self.calls.append((bound_agent_id, before, limit))
            return await asyncio.to_thread(
                read_live_agent_transcript,
                self._pages[bound_agent_id],
                before=before,
                limit=limit,
            )

        return read


class _Sessions:
    """Transcript-read collaborator without a resident child runtime."""

    def __init__(self, snapshot: TranscriptReadSnapshot | None = None) -> None:
        self.snapshot = snapshot
        self.resolved: list[str] = []

    async def resolve_resident_transcript_read(self, _agent_id: str) -> None:
        return None

    async def resident_transcript_read_is_current(self, _snapshot: object) -> bool:
        return True

    async def resolve_transcript_read(self, agent_id: str) -> TranscriptReadSnapshot:
        self.resolved.append(agent_id)
        if self.snapshot is None:
            raise UnknownAgentError(agent_id)
        return self.snapshot

    async def transcript_read_is_current(self, _snapshot: object) -> bool:
        return True


class _AgentLoop:
    """Hashable stand-in for the root loop keying the registry manifests."""

    session_id = "parent-1"
    _session_generation = 1


def _handler(
    sessions: _Sessions, manifest: RegistryManifest | None
) -> CoreRequestHandler:
    handler = object.__new__(CoreRequestHandler)
    test_handler = cast(Any, handler)
    test_handler._agent_loop = _AgentLoop()
    test_handler._sessions = sessions
    test_handler._runtime_factory = SimpleNamespace(
        registry_manifests={test_handler._agent_loop: manifest}
        if manifest is not None
        else {}
    )
    test_handler._require_attached = lambda _session_id: None
    return handler


async def _page(
    handler: CoreRequestHandler, agent_id: str, **params: object
) -> AgentTranscriptGetResponse:
    result = await handler._dispatch_agent(
        "agent/transcript/get", {"agentId": agent_id, **params}
    )
    return cast(AgentTranscriptGetResponse, result.response)


@pytest.mark.asyncio
async def test_live_agent_page_is_served_from_the_registered_reader(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    producer.publish("agent-1", _messages("first", "second"))
    sessions = _Sessions()

    response = await _page(_handler(sessions, manifest), "agent-1")

    assert response.state is AgentTranscriptState.AVAILABLE
    assert [entry.display_text for entry in response.entries or []] == [
        "first",
        "second",
    ]
    assert producer.calls == [("agent-1", None, 50)]
    # The live page never consults the disk-backed view of a live agent.
    assert sessions.resolved == []


@pytest.mark.asyncio
async def test_live_agent_page_honours_the_before_cursor(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    producer.publish("agent-1", _messages("one", "two", "three"))
    handler = _handler(_Sessions(), manifest)

    newest = await _page(handler, "agent-1", limit=2)
    assert [entry.display_text for entry in newest.entries or []] == ["two", "three"]
    assert newest.has_more is True
    cursor = newest.oldest_cursor
    assert cursor is not None

    older = await _page(handler, "agent-1", before=cursor, limit=2)

    assert [entry.display_text for entry in older.entries or []] == ["one"]
    assert older.has_more is False
    assert producer.calls[-1] == ("agent-1", cursor, 2)


@pytest.mark.asyncio
async def test_live_agent_empty_snapshot_installs_an_empty_page(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    producer.publish("agent-1", [])

    response = await _page(_handler(_Sessions(), manifest), "agent-1")

    assert response.state is AgentTranscriptState.AVAILABLE
    assert response.entries == []
    assert response.has_more is False
    assert response.oldest_cursor is None
    assert producer.calls == [("agent-1", None, 50)]


@pytest.mark.asyncio
async def test_page_install_replaces_the_served_page(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    handler = _handler(_Sessions(), manifest)
    producer.publish("agent-1", _messages("old"))

    before_install = await _page(handler, "agent-1")
    producer.publish("agent-1", _messages("installed", "tail"))
    after_install = await _page(handler, "agent-1")

    assert [entry.display_text for entry in before_install.entries or []] == ["old"]
    assert [entry.display_text for entry in after_install.entries or []] == [
        "installed",
        "tail",
    ]


@pytest.mark.asyncio
async def test_live_agent_disk_read_is_refused_typed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    producer.publish("agent-1", _messages("live"))
    sessions = _Sessions()
    handler = _handler(sessions, manifest)

    with pytest.raises(LiveTranscriptSourceRefused) as exc_info:
        await handler.read_agent_transcript_page("agent-1", source="disk")

    assert exc_info.value.reason == "disk_read_for_live_agent"
    assert exc_info.value.code is ProtocolErrorCode.CONFLICT
    assert sessions.resolved == []


@pytest.mark.asyncio
async def test_live_agent_without_a_page_reader_is_refused_typed(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    producer.publish("agent-1", _messages("unreachable"), with_page=False)
    sessions = _Sessions()

    with pytest.raises(LiveTranscriptSourceRefused) as exc_info:
        await _page(_handler(sessions, manifest), "agent-1")

    assert exc_info.value.reason == "live_agent_unavailable"
    assert sessions.resolved == []


@pytest.mark.asyncio
async def test_finished_agent_transcript_still_reads_from_disk(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    producer.publish("agent-1", _messages("live"))
    producer.retire("agent-1")
    root = tmp_path / "sessions" / "root"
    child = root / "agents" / "agent-1"
    child.mkdir(parents=True)
    (child / "meta.json").write_text(json.dumps({"total_messages": 1}))
    (child / "messages.jsonl").write_text(
        json.dumps({"role": "assistant", "message_id": "saved-1", "content": "saved"})
        + "\n"
    )
    snapshot = TranscriptReadSnapshot(
        agent_id="agent-1",
        child_session_id="agent-1",
        child_dir=child,
        parent_dir=root,
        parent_identity=("parent-1", 1),
        parent_runtime_token=1,
        link_identity=("agent-1", "tool-1", "worker", "child"),
    )
    sessions = _Sessions(snapshot)

    response = await _page(_handler(sessions, manifest), "agent-1")

    assert response.state is AgentTranscriptState.AVAILABLE
    assert [entry.display_text for entry in response.entries or []] == ["saved"]
    assert sessions.resolved == ["agent-1"]
    assert producer.calls == []


@pytest.mark.asyncio
async def test_live_page_port_serves_the_registered_page(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    producer.publish("agent-1", _messages("port"))
    pages = LiveTranscriptPagesImpl(_handler(_Sessions(), manifest))

    response = await pages.read_page("agent-1", source="live")

    assert pages.is_live("agent-1") is True
    assert pages.is_live("agent-2") is False
    assert [entry.display_text for entry in response.entries or []] == ["port"]
    with pytest.raises(LiveTranscriptSourceRefused):
        await pages.read_page("agent-1", source="disk")


@pytest.mark.asyncio
async def test_live_paging_adds_no_persistent_artifact(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    producer = _ProducerStandIn(manifest)
    producer.publish("agent-1", _messages("one"))

    await _page(_handler(_Sessions(), manifest), "agent-1")

    assert {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == {"sessions/root.registry.json", "sessions/root/.writer-fence.lock"}
