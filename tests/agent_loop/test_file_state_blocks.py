"""Working-set state blocks wired into the agent loop.

A state block is appended when and only when a tracked file's revision changes:
one immutable, last-positioned injected user message per working-set revision
change, carrying the changed path, its new revision, and the current outline.
External bumps (revisions published to the shared registry by another worker)
are reported by the next appended block. Denied reads enroll nothing, sensitive
files enroll no outline, and revision-less (client-mediated) observations
degrade without breaking the turn.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chartreux.core.agent_loop._file_state_context import FILE_STATE_BLOCK_TAG
from chartreux.core.events import ToolResultEvent
from chartreux.core.file_revision import FileRevision, full_revision
from chartreux.core.file_state import FileStateStore
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.tools.builtins.read_file import ReadFileResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _matches_revision(known: FileRevision | None, expected: FileRevision) -> bool:
    return known is not None and known.matches(expected)


class _DictRegistry:
    """Minimal revision-registry stand-in shared by loop tests."""

    def __init__(self) -> None:
        self.revisions: dict[str, FileRevision] = {}

    def publish(
        self, path: str, revision: FileRevision, *, sequence: int | None = None
    ) -> None:
        del sequence  # the stand-in keeps last-publish-wins semantics
        self.revisions[path] = revision

    def known_revision(self, path: str) -> FileRevision | None:
        return self.revisions.get(path)

    def external_bump(
        self, path: str, last_seen: FileRevision | str | None
    ) -> FileRevision | None:
        current = self.revisions.get(path)
        if current is None or last_seen is None or current.matches(last_seen):
            return None
        return current


def _call(name: str, args: dict, call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name=name, arguments=json.dumps(args)),
    )


def _state_blocks(loop) -> list[LLMMessage]:
    return [
        message
        for message in loop.messages
        if message.injected
        and message.role == Role.user
        and FILE_STATE_BLOCK_TAG in str(message.content)
    ]


def _read_call(target: Path, call_id: str = "read") -> ToolCall:
    return _call("read_file", {"file_path": str(target)}, call_id)


@pytest.mark.asyncio
async def test_block_appended_only_when_revision_changes(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("def f():\n    return 1\n")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_read_call(target)])],
        [mock_llm_chunk(content="read done")],
        # Second turn: re-read the unchanged file.
        [mock_llm_chunk(tool_calls=[_read_call(target, "read2")])],
        [mock_llm_chunk(content="read again")],
        # Third turn: edit the file (a revision change).
        [
            mock_llm_chunk(
                tool_calls=[
                    _call(
                        "edit",
                        {
                            "file_path": str(target),
                            "old_string": "return 1",
                            "new_string": "return 2",
                        },
                        "edit",
                    )
                ]
            )
        ],
        [mock_llm_chunk(content="edited")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            tools={
                "read_file": {"permission": "always"},
                "edit": {"permission": "always"},
            }
        ),
        cwd=tmp_path,
        backend=backend,
    )
    try:
        events = [event async for event in loop.act("read the file")]
        read_results = [event for event in events if isinstance(event, ToolResultEvent)]
        assert len(read_results) == 1
        read_result = read_results[0].result
        assert isinstance(read_result, ReadFileResult)
        first_revision = read_result.revision
        assert first_revision is not None

        blocks = _state_blocks(loop)
        assert len(blocks) == 1
        block = blocks[0]
        assert block.role == Role.user and block.injected is True
        content = str(block.content)
        assert str(target) in content
        assert first_revision in content
        assert "function f (lines 1-2)" in content
        # The path is enrolled with the observed revision.
        entry = loop.file_state.entry(str(target))
        assert entry is not None
        assert entry.revision is not None
        assert entry.revision.revision == first_revision
        assert entry.outline is not None

        # Turn 2: the file is unchanged — no new block.
        async for _event in loop.act("read it again"):
            pass
        assert len(_state_blocks(loop)) == 1

        # Turn 3: the edit bumps the revision — exactly one new block.
        async for _event in loop.act("edit it"):
            pass
        blocks = _state_blocks(loop)
        assert len(blocks) == 2
        assert "return 2" not in str(blocks[1].content)  # metadata only
        assert "function f (lines 1-2)" in str(blocks[1].content)
        known = loop.file_state.known_revision(str(target))
        assert known is not None
        assert known.matches(full_revision(b"def f():\n    return 2\n"))
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_blocks_are_immutable_and_last_positioned(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("a = 1\n")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_read_call(target)])],
        [mock_llm_chunk(content="one")],
        [
            mock_llm_chunk(
                tool_calls=[
                    _call(
                        "edit",
                        {
                            "file_path": str(target),
                            "old_string": "a = 1",
                            "new_string": "a = 2",
                        },
                        "edit",
                    )
                ]
            )
        ],
        [mock_llm_chunk(content="two")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            tools={
                "read_file": {"permission": "always"},
                "edit": {"permission": "always"},
            }
        ),
        cwd=tmp_path,
        backend=backend,
    )
    try:
        async for _event in loop.act("read"):
            pass
        after_first = [message.model_dump(mode="json") for message in loop.messages]
        first_block = _state_blocks(loop)[0]
        # The block sits right after the read's tool result (it was the last
        # message at append time) and before the assistant's reply.
        block_index = next(
            index
            for index, message in enumerate(loop.messages)
            if message is first_block
        )
        assert loop.messages[block_index - 1].role == Role.tool
        assert loop.messages[block_index + 1].role == Role.assistant

        async for _event in loop.act("edit"):
            pass
        # No per-turn rewrite: the whole earlier transcript is unchanged, and
        # the new block was appended after the previous last message.
        after_second = [message.model_dump(mode="json") for message in loop.messages]
        assert after_second[: len(after_first)] == after_first
        blocks = _state_blocks(loop)
        assert len(blocks) == 2
        assert str(blocks[0].content) == str(first_block.content)
        assert blocks[1] is not first_block
        # The second block was also last at append time (before the final
        # assistant message).
        second_index = next(
            index for index, message in enumerate(loop.messages) if message is blocks[1]
        )
        assert loop.messages[second_index - 1].role == Role.tool
        assert loop.messages[second_index + 1].role == Role.assistant
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_denied_read_enrolls_nothing(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside\n")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_read_call(outside)])],
        [mock_llm_chunk(content="done")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(tools={"read_file": {"permission": "always"}}),
        cwd=tmp_path,
        backend=backend,
    )
    try:
        events = [event async for event in loop.act("read outside")]
        results = [event for event in events if isinstance(event, ToolResultEvent)]
        assert len(results) == 1
        assert results[0].skip_reason  # denied: out of root
        assert len(loop.file_state) == 0
        assert _state_blocks(loop) == []
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_sensitive_file_enrolls_no_outline(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text("SUPER_SECRET_VALUE=1\n")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_read_call(target)])],
        [mock_llm_chunk(content="done")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            # The read itself is allowed (patterns emptied), but the working
            # set must still never outline a sensitive path.
            tools={"read_file": {"permission": "always", "sensitive_patterns": []}}
        ),
        cwd=tmp_path,
        backend=backend,
    )
    try:
        events = [event async for event in loop.act("read the env")]
        results = [event for event in events if isinstance(event, ToolResultEvent)]
        assert len(results) == 1
        assert not results[0].skip_reason and not results[0].error
        entry = loop.file_state.entry(str(target))
        assert entry is not None
        assert entry.outline is None
        blocks = _state_blocks(loop)
        assert len(blocks) == 1
        content = str(blocks[0].content)
        assert "No symbol outline" in content
        assert "SUPER_SECRET_VALUE" not in content
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_windowed_read_keeps_baseline_and_appends_no_second_block(
    tmp_path: Path,
) -> None:
    # W1 regression at the loop seam: a bounded window read publishes
    # revision=None; it must not wipe the vouched baseline, so the next full
    # read of the unchanged file appends no second block (the pre-fix
    # behavior appended a phantom source='self' block here) and the path
    # stays enrolled at its observed revision.
    target = tmp_path / "mod.py"
    target.write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_read_call(target)])],
        [mock_llm_chunk(content="read done")],
        # Second turn: a bounded window read (offset/limit) — revision=None.
        [
            mock_llm_chunk(
                tool_calls=[
                    _call(
                        "read_file",
                        {"file_path": str(target), "offset": 2, "limit": 2},
                        "read-window",
                    )
                ]
            )
        ],
        [mock_llm_chunk(content="window read")],
        # Third turn: a full re-read of the unchanged file.
        [mock_llm_chunk(tool_calls=[_read_call(target, "read3")])],
        [mock_llm_chunk(content="read again")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(tools={"read_file": {"permission": "always"}}),
        cwd=tmp_path,
        backend=backend,
    )
    try:
        async for _event in loop.act("read the file"):
            pass
        assert len(_state_blocks(loop)) == 1
        first_revision = loop.file_state.known_revision(str(target))
        assert first_revision is not None

        async for _event in loop.act("window read"):
            pass
        # The windowed read is non-vouching: no block, baseline intact.
        assert len(_state_blocks(loop)) == 1
        assert loop.file_state.known_revision(str(target)) == first_revision

        async for _event in loop.act("read it again"):
            pass
        # The full re-read of the unchanged file is not a revision change:
        # no second block.
        assert len(_state_blocks(loop)) == 1
        assert loop.file_state.known_revision(str(target)) == first_revision
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_external_bump_reported_by_next_block(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("def f():\n    return 1\n")
    registry = _DictRegistry()
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_read_call(target)])],
        [mock_llm_chunk(content="read done")],
        # Second turn: no tool calls — the flush still reports the bump.
        [mock_llm_chunk(content="continue")],
        # Third turn: still nothing changed — no repeated report.
        [mock_llm_chunk(content="once more")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(tools={"read_file": {"permission": "always"}}),
        cwd=tmp_path,
        backend=backend,
        file_revision_registry=registry,
    )
    try:
        async for _event in loop.act("read the file"):
            pass
        assert len(_state_blocks(loop)) == 1
        assert registry.known_revision(str(target)) is not None
        # Another worker edits the file and publishes its revision.
        target.write_text("def f():\n    return 2\n\n\ndef g():\n    return 3\n")
        registry.publish(
            str(target),
            full_revision(b"def f():\n    return 2\n\n\ndef g():\n    return 3\n"),
        )
        async for _event in loop.act("continue"):
            pass
        blocks = _state_blocks(loop)
        assert len(blocks) == 2
        bump = blocks[1]
        content = str(bump.content)
        assert "source='external'" in content
        assert str(target) in content
        # The fresh on-disk outline is included.
        assert "function g (lines 5-6)" in content
        # The bumped revision is adopted: no repeated report.
        assert _matches_revision(
            loop.file_state.known_revision(str(target)),
            full_revision(b"def f():\n    return 2\n\n\ndef g():\n    return 3\n"),
        )
        # The block was in the model request (appended before the LLM call).
        request = backend.requests_messages[-1]
        assert any(
            message.injected
            and message.role == Role.user
            and "source='external'" in str(message.content)
            for message in request
        )
        async for _event in loop.act("once more"):
            pass
        assert len(_state_blocks(loop)) == 2
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_compaction_envelope_carries_file_state(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("def f():\n    return 1\n")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_read_call(target)])],
        [mock_llm_chunk(content="read done")],
        [mock_llm_chunk(content="<summary>compacted</summary>")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(tools={"read_file": {"permission": "always"}}),
        cwd=tmp_path,
        backend=backend,
    )
    try:
        async for _event in loop.act("read the file"):
            pass
        observed = loop.file_state.known_revision(str(target))
        assert observed is not None
        await loop.compact()
        boundary = loop.messages[-1]
        assert boundary.context_boundary == "compaction"
        assert boundary.tool_result is not None
        output = boundary.tool_result.output
        # The existing TodoState replay keeps working alongside file_state.
        assert "todo_state" in output
        assert "file_state" in output
        # Structural replay restores the working-set state (metadata only).
        replayed = FileStateStore.replay(loop.messages)
        assert _matches_revision(replayed.known_revision(str(target)), observed)
        assert replayed.outline_for(str(target)) is not None
        # The live store is untouched by the replay.
        assert _matches_revision(loop.file_state.known_revision(str(target)), observed)
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_compaction_boundary_serves_file_state_in_the_request(
    tmp_path: Path,
) -> None:
    # B3 regression: the compaction boundary's model-visible content carries
    # the bounded working-set summary (paths, revisions, outlines), so the
    # converted LLM request after compaction contains them — the envelope's
    # tool_result.output alone is not model-visible (the backend drops it).
    target = tmp_path / "mod.py"
    target.write_text("def f():\n    return 1\n")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_read_call(target)])],
        [mock_llm_chunk(content="read done")],
        [mock_llm_chunk(content="<summary>compacted</summary>")],
        [mock_llm_chunk(content="after compaction")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(tools={"read_file": {"permission": "always"}}),
        cwd=tmp_path,
        backend=backend,
    )
    try:
        async for _event in loop.act("read the file"):
            pass
        observed = loop.file_state.known_revision(str(target))
        assert observed is not None

        await loop.compact()

        boundary = loop.messages[-1]
        assert boundary.context_boundary == "compaction"
        content = str(boundary.content)
        assert str(target) in content
        assert observed.revision in content
        assert "function f (lines 1-2)" in content

        # The section reached the converted LLM request: the next turn's
        # request carries the boundary message's content.
        async for _event in loop.act("continue"):
            pass
        request = backend.requests_messages[-1]
        boundary_in_request = next(
            message
            for message in request
            if message.role == Role.user and str(target) in str(message.content)
        )
        request_content = str(boundary_in_request.content)
        assert observed.revision in request_content
        assert "function f (lines 1-2)" in request_content
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_post_compaction_boundary_keeps_system_prompt_and_summary_structure(
    tmp_path: Path,
) -> None:
    # Package-14 precedent pins at the boundary seam: the captured post-
    # compaction request still opens with the FULL system prompt (the frozen
    # policy's composed text), the summary rides the boundary verbatim inside
    # its structural wrapper, and the file-state section sits alongside it.
    target = tmp_path / "mod.py"
    target.write_text("def f():\n    return 1\n")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_read_call(target)])],
        [mock_llm_chunk(content="read done")],
        [mock_llm_chunk(content="<summary>no agents were launched</summary>")],
        [mock_llm_chunk(content="after compaction")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(tools={"read_file": {"permission": "always"}}),
        cwd=tmp_path,
        backend=backend,
    )
    try:
        async for _event in loop.act("read the file"):
            pass
        system_prompt = loop.messages[0]

        await loop.compact()

        boundary = loop.messages[-1]
        assert boundary.context_boundary == "compaction"
        content = str(boundary.content)
        assert "<previous_user_messages>" in content
        assert "<compaction_summary>" in content
        assert "no agents were launched" in content
        assert str(target) in content

        async for _event in loop.act("continue"):
            pass
        request = backend.requests_messages[-1]
        assert request[0].role == Role.system
        assert request[0].content == system_prompt.content
        request_system = str(request[0].content)
        assert "## Resolved contract" in request_system
        boundary_in_request = next(
            message
            for message in request
            if message.role == Role.user and str(target) in str(message.content)
        )
        assert "no agents were launched" in str(boundary_in_request.content)
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_clear_history_resets_working_set(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("a = 1\n")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_read_call(target)])],
        [mock_llm_chunk(content="done")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(tools={"read_file": {"permission": "always"}}),
        cwd=tmp_path,
        backend=backend,
    )
    try:
        async for _event in loop.act("read the file"):
            pass
        assert len(loop.file_state) == 1
        assert len(_state_blocks(loop)) == 1
        await loop.clear_history()
        assert len(loop.file_state) == 0
        assert _state_blocks(loop) == []
    finally:
        await loop.aclose()
