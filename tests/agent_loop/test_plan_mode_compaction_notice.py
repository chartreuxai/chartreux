"""The plan-mode notice rides the compaction boundary on every trigger path.

``AgentLoop.compact`` is the sole funnel of the three compaction triggers — the
TUI's manual ``/compact``, the server's proactive auto-compaction middleware,
and the server's reactive overflow recovery — so a mode notice carried there is
re-injected after ANY compaction, not only the manual one. The carrier is the
published boundary CONTENT: a summarizer echoes nothing (``extra_instructions``
reach its request and nothing else), so every positive below uses a fixed
summary that provably omits the notice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.config import SessionLoggingConfig
from chartreux.core.events import CompactEndEvent, CompactStartEvent
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.plan_mode import build_plan_mode_compaction_notice
from chartreux.core.session_types import PlanModeRecord
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_SUMMARY = "<summary>fixed summary that omits the notice</summary>"
_NOTICE_MARKER = "Plan mode is active. Plan document:"


def _logging(tmp_path: Path) -> SessionLoggingConfig:
    return SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions"))


def _logging_at(tmp_path: Path, name: str) -> SessionLoggingConfig:
    return SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / f"{name}-sessions")
    )


def _ctx_too_long_error() -> BackendError:
    return BackendError(
        provider="mistral",
        endpoint="/chat/completions",
        status=400,
        reason="Bad Request",
        headers={},
        body_text="context too long",
        parsed_error=None,
        model="test",
        payload_summary=PayloadSummary(
            model="test",
            message_count=1,
            approx_chars=1,
            temperature=0.0,
            has_tools=False,
            tool_choice=None,
        ),
    )


class _RaisingBackend(FakeBackend):
    """FakeBackend that raises a scripted exception at one call index."""

    def __init__(self, streams=None, *, raises_at=None):
        super().__init__(streams)
        self._raises_at = dict(raises_at or {})
        self._calls = 0

    def _claim_call(self) -> None:
        index = self._calls
        self._calls += 1
        if index in self._raises_at:
            raise self._raises_at[index]

    async def complete(self, **kwargs):
        self._claim_call()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs):
        self._claim_call()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


def _loop(
    tmp_path: Path,
    backend: FakeBackend,
    *,
    logging: SessionLoggingConfig | None = None,
    auto_compact_threshold: int = 999,
):
    return build_test_agent_loop(
        config=build_test_vibe_config(
            models=make_test_models(auto_compact_threshold=auto_compact_threshold),
            session_logging=logging or _logging(tmp_path),
        ),
        backend=backend,
        cwd=tmp_path,
    )


async def _record_plan_mode(loop, *, theme: str | None = "Fix the widget") -> str:
    plan_path = str(tmp_path_plan_path(loop))
    await loop.session_logger.persist_plan_mode(
        PlanModeRecord(
            plan_path=plan_path,
            theme=theme,
            planner_model="glm-5-3",
            planner_thinking="high",
        )
    )
    return plan_path


def tmp_path_plan_path(loop) -> Path:
    return loop.cwd / "tasks" / "plan-x" / "plan.md"


def _boundary(loop) -> LLMMessage:
    return next(
        message
        for message in reversed(loop.messages)
        if message.context_boundary == "compaction"
    )


@pytest.mark.asyncio
async def test_manual_compaction_boundary_carries_the_plan_notice(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([[mock_llm_chunk(content=_SUMMARY)]])
    loop = _loop(tmp_path, backend)
    loop.messages.append(LLMMessage(role=Role.user, content="Hello"))
    loop.stats.context_tokens = 100
    plan_path = await _record_plan_mode(loop)

    await loop.compact()

    content = _boundary(loop).content or ""
    assert build_plan_mode_compaction_notice(plan_path, "Fix the widget") in content
    assert "fixed summary that omits the notice" in content
    # The carrier is the boundary, not the summarizer: the request the
    # summarizer saw never held the notice.
    assert _NOTICE_MARKER not in (backend.requests_messages[0][-1].content or "")


@pytest.mark.asyncio
async def test_proactive_compaction_boundary_carries_the_plan_notice(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content=_SUMMARY)],
        [mock_llm_chunk(content="<final>")],
    ])
    loop = _loop(tmp_path, backend, auto_compact_threshold=1)
    plan_path = await _record_plan_mode(loop)
    loop.stats.context_tokens = 2

    events = [event async for event in loop.act("Hello")]

    kinds = [type(event) for event in events]
    assert CompactStartEvent in kinds
    assert CompactEndEvent in kinds
    content = _boundary(loop).content or ""
    assert build_plan_mode_compaction_notice(plan_path, "Fix the widget") in content


@pytest.mark.asyncio
async def test_reactive_compaction_boundary_carries_the_plan_notice(
    tmp_path: Path,
) -> None:
    backend = _RaisingBackend(
        [[mock_llm_chunk(content=_SUMMARY)], [mock_llm_chunk(content="<final>")]],
        raises_at={0: _ctx_too_long_error()},
    )
    loop = _loop(tmp_path, backend)
    plan_path = await _record_plan_mode(loop)
    loop.stats.context_tokens = 0

    events = [event async for event in loop.act("Hello")]

    kinds = [type(event) for event in events]
    assert CompactStartEvent in kinds
    assert CompactEndEvent in kinds
    content = _boundary(loop).content or ""
    assert build_plan_mode_compaction_notice(plan_path, "Fix the widget") in content


@pytest.mark.asyncio
async def test_themeless_plan_record_notice_has_no_theme_clause(tmp_path: Path) -> None:
    backend = FakeBackend([[mock_llm_chunk(content=_SUMMARY)]])
    loop = _loop(tmp_path, backend)
    loop.messages.append(LLMMessage(role=Role.user, content="Hello"))
    loop.stats.context_tokens = 100
    plan_path = await _record_plan_mode(loop, theme=None)

    await loop.compact()

    content = _boundary(loop).content or ""
    assert build_plan_mode_compaction_notice(plan_path) in content
    assert "Theme:" not in content
    assert "Theme: None" not in content


@pytest.mark.asyncio
async def test_a_session_without_a_record_gets_no_notice(tmp_path: Path) -> None:
    backend = FakeBackend([[mock_llm_chunk(content=_SUMMARY)]])
    loop = _loop(tmp_path, backend)
    loop.messages.append(LLMMessage(role=Role.user, content="Hello"))
    loop.stats.context_tokens = 100

    await loop.compact()

    content = _boundary(loop).content or ""
    assert _NOTICE_MARKER not in content
    # No stray empty heading either: the boundary is the pre-change render.
    assert content.endswith("</compaction_summary>")


@pytest.mark.asyncio
async def test_logging_disabled_session_gets_no_notice_and_no_error(
    tmp_path: Path,
) -> None:
    # With session logging disabled SessionMetadata is None by construction, so
    # the mode can be neither persisted nor read back: the accepted designed
    # behavior is a compaction boundary with no notice.
    backend = FakeBackend([[mock_llm_chunk(content=_SUMMARY)]])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            models=make_test_models(auto_compact_threshold=999)
        ),
        backend=backend,
        cwd=tmp_path,
    )
    assert loop.session_logger.session_metadata is None
    loop.messages.append(LLMMessage(role=Role.user, content="Hello"))
    loop.stats.context_tokens = 100

    await loop.compact()

    content = _boundary(loop).content or ""
    assert _NOTICE_MARKER not in content
    assert loop.session_logger.session_metadata is None


@pytest.mark.asyncio
async def test_a_separate_loop_does_not_inherit_the_plan_record(tmp_path: Path) -> None:
    # Executor and subagent loops hold their own SessionMetadata: the planner's
    # record is not visible to them, so the same no-record negative keeps them
    # quiet without a special case.
    planner = _loop(tmp_path, FakeBackend([]), logging=_logging_at(tmp_path, "planner"))
    await _record_plan_mode(planner)

    backend = FakeBackend([[mock_llm_chunk(content=_SUMMARY)]])
    other = _loop(tmp_path, backend, logging=_logging_at(tmp_path, "executor"))
    other.messages.append(LLMMessage(role=Role.user, content="Hello"))
    other.stats.context_tokens = 100

    await other.compact()

    assert other.session_logger.session_metadata is not None
    assert other.session_logger.session_metadata.plan_mode is None
    assert _NOTICE_MARKER not in (_boundary(other).content or "")
