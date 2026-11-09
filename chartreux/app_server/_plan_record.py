"""Mapping between the persisted plan-mode record and its wire model."""

from __future__ import annotations

from chartreux.app_server.protocol import SessionPlanRecord, SessionPlanScanDomain
from chartreux.core.session_types import PlanModeRecord, PlanScanDomain


def _scan_domain_wire(domain: PlanScanDomain | None) -> SessionPlanScanDomain | None:
    if domain is None:
        return None
    return SessionPlanScanDomain(
        save_dir=domain.save_dir, session_prefix=domain.session_prefix
    )


def _scan_domain_record(domain: SessionPlanScanDomain | None) -> PlanScanDomain | None:
    if domain is None:
        return None
    return PlanScanDomain(
        save_dir=domain.save_dir, session_prefix=domain.session_prefix
    )


def session_plan_record(record: PlanModeRecord) -> SessionPlanRecord:
    return SessionPlanRecord(
        plan_path=record.plan_path,
        theme=record.theme,
        planner_model=record.planner_model,
        planner_thinking=record.planner_thinking,
        pre_entry_active_model=record.pre_entry_active_model,
        pre_entry_planner_thinking=record.pre_entry_planner_thinking,
        episode_id=record.episode_id,
        sidecar_path=record.sidecar_path,
        draft_path=record.draft_path,
        snapshot_path=record.snapshot_path,
        state=record.state,
        consent_sha256=record.consent_sha256,
        attempt_generation=record.attempt_generation,
        scan_domain=_scan_domain_wire(record.scan_domain),
    )


def plan_mode_record(record: SessionPlanRecord) -> PlanModeRecord:
    return PlanModeRecord(
        plan_path=record.plan_path,
        theme=record.theme,
        planner_model=record.planner_model,
        planner_thinking=record.planner_thinking,
        pre_entry_active_model=record.pre_entry_active_model,
        pre_entry_planner_thinking=record.pre_entry_planner_thinking,
        episode_id=record.episode_id,
        sidecar_path=record.sidecar_path,
        draft_path=record.draft_path,
        snapshot_path=record.snapshot_path,
        state=record.state,
        consent_sha256=record.consent_sha256,
        attempt_generation=record.attempt_generation,
        scan_domain=_scan_domain_record(record.scan_domain),
    )
