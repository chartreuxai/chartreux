"""Mapping between the persisted plan-mode record and its wire model."""

from __future__ import annotations

from chartreux.app_server.protocol import SessionPlanRecord
from chartreux.core.session_types import PlanModeRecord


def session_plan_record(record: PlanModeRecord) -> SessionPlanRecord:
    return SessionPlanRecord(
        plan_path=record.plan_path,
        theme=record.theme,
        planner_model=record.planner_model,
        planner_thinking=record.planner_thinking,
        pre_entry_active_model=record.pre_entry_active_model,
        pre_entry_planner_thinking=record.pre_entry_planner_thinking,
    )


def plan_mode_record(record: SessionPlanRecord) -> PlanModeRecord:
    return PlanModeRecord(
        plan_path=record.plan_path,
        theme=record.theme,
        planner_model=record.planner_model,
        planner_thinking=record.planner_thinking,
        pre_entry_active_model=record.pre_entry_active_model,
        pre_entry_planner_thinking=record.pre_entry_planner_thinking,
    )
