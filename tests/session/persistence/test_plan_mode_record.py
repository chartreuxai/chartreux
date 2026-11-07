"""The persisted plan-mode record: schema tolerance, write and removal.

The record is the single source of truth for plan mode across a restart
(design §7). It is OPTIONAL: a session written before the field existed loads
unchanged and restores as not-planning, and the record only ever appears when
entry wrote it — a persisted mode without a restore path cannot be produced
because both land together.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import AgentStats, PlanModeRecord
from chartreux.core.tools.manager import ToolManager
from tests.session.persistence._session_persistence_support import (
    disabled_session_config as disabled_session_config,
    mock_tool_manager as mock_tool_manager,
    mock_vibe_config as mock_vibe_config,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)

_RECORD = PlanModeRecord(
    plan_path="/workspaces/chartreux/tasks/plan-abc123/plan.md",
    theme="Fix the widget",
    planner_model="glm-5-3",
    planner_thinking="high",
    pre_entry_active_model="mistral-large-4",
    pre_entry_planner_thinking="low",
)


def _metadata(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "session_id": "plan-record-session",
        "start_time": "2026-01-01T00:00:00+00:00",
        "end_time": None,
        "git_commit": None,
        "git_branch": None,
        "username": "tester",
        "environment": {"working_directory": "/repo"},
    }
    return base | overrides


def _write_metadata(session_dir: Path, metadata: dict[str, object]) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "meta.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


async def _save(
    logger: SessionLogger, config: ChartreuxConfigSchema, tool_manager: ToolManager
) -> None:
    await logger.save_interaction(
        messages=[
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(role=Role.user, content="Hello"),
        ],
        stats=AgentStats(steps=1),
        config=config,
        tool_manager=tool_manager,
    )


# --- Schema tolerance --------------------------------------------------------


def test_metadata_written_before_the_record_loads_as_not_planning(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "sessions" / "test_plan-record-session_abc123"
    metadata = _metadata()
    assert "plan_mode" not in metadata
    _write_metadata(session_dir, metadata)

    loaded = SessionLoader.load_metadata(session_dir)

    assert loaded.plan_mode is None
    assert loaded.session_id == "plan-record-session"


def test_an_undeclared_top_level_key_is_ignored_on_load(tmp_path: Path) -> None:
    # The reader is model_validate_json over a model without extra="forbid":
    # an unknown key never fails a load, which is why the record had to be
    # declared to be visible at all.
    session_dir = tmp_path / "sessions" / "test_unknown-key_abc123"
    metadata = _metadata()
    metadata["retired_metadata_key"] = {"anything": 1}
    _write_metadata(session_dir, metadata)

    loaded = SessionLoader.load_metadata(session_dir)

    assert loaded.plan_mode is None
    assert not hasattr(loaded, "retired_metadata_key")


def test_a_plan_record_round_trips_through_the_metadata_file(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "test_round-trip_abc123"
    _write_metadata(session_dir, _metadata(plan_mode=_RECORD.model_dump(mode="json")))

    loaded = SessionLoader.load_metadata(session_dir)

    assert loaded.plan_mode == _RECORD


def test_a_themeless_record_round_trips_without_a_theme() -> None:
    themeless = PlanModeRecord(
        plan_path="/workspaces/chartreux/tasks/plan-abc123/plan.md",
        planner_model="glm-5-3",
        planner_thinking="high",
    )

    dumped = themeless.model_dump(mode="json")

    assert dumped["theme"] is None
    assert dumped["pre_entry_active_model"] is None
    assert dumped["pre_entry_planner_thinking"] is None
    assert PlanModeRecord.model_validate(dumped) == themeless


# --- Logging helpers ---------------------------------------------------------


class TestPersistPlanMode:
    @pytest.mark.asyncio
    async def test_a_fresh_session_carries_the_record_into_the_first_save(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        # Entry can precede the first message, so there is no metadata file to
        # patch yet: only the in-memory record can carry it until the first
        # full save writes it out.
        logger = SessionLogger(session_config, "plan-fresh-session")
        assert logger.session_dir is not None
        assert not (logger.session_dir / "meta.json").exists()

        await logger.persist_plan_mode(_RECORD)

        assert not (logger.session_dir / "meta.json").exists()
        assert logger.session_metadata is not None
        assert logger.session_metadata.plan_mode == _RECORD

        await _save(logger, mock_vibe_config, mock_tool_manager)

        reloaded = SessionLoader.load_metadata(logger.session_dir)
        assert reloaded.plan_mode == _RECORD

    @pytest.mark.asyncio
    async def test_an_existing_metadata_file_is_patched_without_a_full_save(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "plan-resumed-session")
        await _save(logger, mock_vibe_config, mock_tool_manager)

        await logger.persist_plan_mode(_RECORD)

        assert logger.session_dir is not None
        reloaded = SessionLoader.load_metadata(logger.session_dir)
        assert reloaded.plan_mode == _RECORD

    @pytest.mark.asyncio
    async def test_clear_removes_the_record_from_the_file_and_memory(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "plan-cleared-session")
        await _save(logger, mock_vibe_config, mock_tool_manager)
        await logger.persist_plan_mode(_RECORD)

        await logger.clear_plan_mode()

        assert logger.session_dir is not None
        assert logger.session_metadata is not None
        assert logger.session_metadata.plan_mode is None
        reloaded = SessionLoader.load_metadata(logger.session_dir)
        assert reloaded.plan_mode is None

    @pytest.mark.asyncio
    async def test_clearing_without_a_record_is_a_noop(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "plan-clean-session")
        await _save(logger, mock_vibe_config, mock_tool_manager)

        await logger.clear_plan_mode()

        assert logger.session_dir is not None
        reloaded = SessionLoader.load_metadata(logger.session_dir)
        assert reloaded.plan_mode is None

    @pytest.mark.asyncio
    async def test_disabled_logging_keeps_the_record_out_of_storage(
        self, disabled_session_config: SessionLoggingConfig
    ) -> None:
        # SessionMetadata is None by construction when logging is off: no
        # record is written and no error is raised (the designed behavior).
        logger = SessionLogger(disabled_session_config, "plan-disabled-session")
        assert logger.session_metadata is None

        await logger.persist_plan_mode(_RECORD)
        await logger.clear_plan_mode()

        assert logger.session_metadata is None
