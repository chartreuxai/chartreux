"""Round-5 Y: the ACTUAL persistence round-trip (design §7, §12).

NOT an in-memory JSON fixture: the real composed steer saves through the
real ``SessionLogger``, the trusted ``SessionLoader`` restores the
provenance marker BEFORE integrity validation, and the write-owning
resume install accepts the clean report without migration. The
incremental-append save path preserves the marker too.

All transports are controlled fakes — an unmocked provider call cannot
occur (fail-closed egress).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chartreux.app_server._runtime import _install_transcript_repair
from chartreux.core.agent_loop._transcript_integrity import (
    TRANSCRIPT_INTEGRITY_METADATA_KEY,
    _live_tail_region_identities,
    find_persistence_violations,
    has_structured_tool_result,
    validate_provider_ready,
    validate_transcript,
)
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import (
    CONTEXT_INJECTION_ORIGIN_KEY,
    CONTEXT_INJECTION_TOOL_PAIR_V1,
    Role,
)
from chartreux.core.session.session_loader import MESSAGES_FILENAME, SessionLoader
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.recovery.test_transcript_integrity import (
    _live_reuse_tail,
    _patch_skill_manager,
)
from tests.stubs.fake_backend import FakeBackend


def _saved_records(session_dir: Path) -> list[dict[str, Any]]:
    """The actual records of a saved ``messages.jsonl`` (one JSON per line)."""
    return [
        json.loads(line)
        for line in (session_dir / MESSAGES_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


class TestRound5YPersistenceRoundTrip:
    """The real save → trusted load → resume install, end to end: the
    provenance marker travels atomically with its owning message in
    ``messages.jsonl`` and is restored before integrity validation.
    """

    @pytest.mark.asyncio
    async def test_real_save_load_round_trip_restores_provenance_before_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = build_test_vibe_config(
            enabled_tools=["skill"],
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="y-roundtrip", enabled=True
            ),
        )
        loop = build_test_agent_loop(
            config=config, backend=FakeBackend([[mock_llm_chunk(content="ok")]])
        )
        try:
            for message in _live_reuse_tail():
                loop.messages.append(message)
            _patch_skill_manager(loop, monkeypatch, name="y-skill")
            # The real composed steer over the live-reuse tail: the steered
            # user message, then the implicit skill pair, then the save.
            await loop.inject_user_context(
                "/y-skill", as_message=True, inject_implicit=True
            )
            transcript = list(loop.messages)
            # [.., A2(c1) in flight, Usteer, Askill(k), Rskill(k)]
            in_flight = transcript[-4]
            steered = transcript[-3]
            pair_assistant, pair_result = transcript[-2], transcript[-1]
            assert steered.role == Role.user
            assert steered.injected is False  # production flag values
            assert steered.content == "/y-skill"
            assert pair_assistant.role == Role.assistant
            assert pair_assistant.injected is False  # production flag values
            assert pair_assistant.tool_calls
            pair_call_id = pair_assistant.tool_calls[0].id
            assert pair_result.role == Role.tool
            assert has_structured_tool_result(pair_result)
            assert (
                pair_assistant._context_injection_origin
                == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
            await loop._save_messages()  # the actual save: no raise
            session_dir = loop.session_logger.session_dir
            assert session_dir is not None
            # The actual saved record carries the key on the pair record.
            records = _saved_records(session_dir)
            marked = [
                record
                for record in records
                if record.get(CONTEXT_INJECTION_ORIGIN_KEY) is not None
            ]
            assert len(marked) == 1
            assert marked[0]["role"] == "assistant"
            assert marked[0]["tool_calls"][0]["id"] == pair_call_id
            assert (
                marked[0][CONTEXT_INJECTION_ORIGIN_KEY]
                == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
            # The trusted loader restores the marker BEFORE validation.
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            loaded_assistants = [
                message
                for message in loaded
                if message.role == Role.assistant and message.tool_calls
            ]
            # [U, A1(c1), R1(c1), U, A2(c1), Usteer, I(k), RI(k)]
            anchor, restored_pair = loaded_assistants[1], loaded_assistants[2]
            assert anchor is not in_flight  # hydration creates new objects
            assert anchor.tool_calls and restored_pair.tool_calls
            assert anchor.tool_calls[0].id == "c1"
            assert restored_pair.tool_calls[0].id == pair_call_id
            assert (
                restored_pair._context_injection_origin
                == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
            # The provenance-keyed walk reaches the anchor through the
            # restored marker.
            assert _live_tail_region_identities(loaded) == {
                id(anchor),
                id(restored_pair),
            }
            assert find_persistence_violations(loaded) == []
            assert validate_transcript(loaded, live_tail_exempt=True) == []
            assert validate_provider_ready(loaded) == []
            # The resume install is a clean no-op (no migration, no artifact).
            assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
            assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        finally:
            await loop.aclose()

    @pytest.mark.asyncio
    async def test_incremental_append_round_trip_preserves_the_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = build_test_vibe_config(
            enabled_tools=["skill"],
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="y-incremental", enabled=True
            ),
        )
        loop = build_test_agent_loop(
            config=config, backend=FakeBackend([[mock_llm_chunk(content="ok")]])
        )
        try:
            for message in _live_reuse_tail():
                loop.messages.append(message)
            _patch_skill_manager(loop, monkeypatch, name="y-skill")
            session_dir = loop.session_logger.session_dir
            assert session_dir is not None
            # The initial save (the tail only): no provenance anywhere yet.
            await loop._save_messages()
            assert all(
                CONTEXT_INJECTION_ORIGIN_KEY not in record
                for record in _saved_records(session_dir)
            )
            # The direct skill injection + the INCREMENTAL save (the
            # tail-serialization path) persists the marker.
            before = len(loop.messages)
            async for _event in loop._inject_invoked_skill("/y-skill"):
                pass
            assert len(loop.messages) == before + 2
            pair_assistant = loop.messages[-2]
            assert pair_assistant.role == Role.assistant
            assert pair_assistant.tool_calls
            pair_call_id = pair_assistant.tool_calls[0].id
            assert (
                pair_assistant._context_injection_origin
                == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
            await loop._save_messages()
            marked = [
                record
                for record in _saved_records(session_dir)
                if record.get(CONTEXT_INJECTION_ORIGIN_KEY) is not None
            ]
            assert len(marked) == 1
            assert marked[0]["tool_calls"][0]["id"] == pair_call_id
            # The round-trip restores the marker; the install is clean.
            loaded, metadata = SessionLoader.load_session(session_dir)
            assert TRANSCRIPT_INTEGRITY_METADATA_KEY not in metadata
            restored = [
                message
                for message in loaded
                if message.role == Role.assistant
                and message.tool_calls
                and message.tool_calls[0].id == pair_call_id
            ]
            assert len(restored) == 1
            assert (
                restored[0]._context_injection_origin == CONTEXT_INJECTION_TOOL_PAIR_V1
            )
            assert find_persistence_violations(loaded) == []
            assert _install_transcript_repair(session_dir, loaded, metadata) is loaded
            assert not list(session_dir.glob(f"{MESSAGES_FILENAME}.recovered-*"))
        finally:
            await loop.aclose()
