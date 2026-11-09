"""The plan-record episode extension: legacy compatibility, direction-explicit.

The persisted ``PlanModeRecord`` and the wire ``SessionPlanRecord`` carry the
same episode fields, and the two converters in ``_plan_record`` move them
across the boundary whole. The two directions behave DELIBERATELY differently
(design §3, §4.1, DR-42):

- persisted -> new reader: the record model does not forbid extra fields, so a
  v0.7.5-era record loads as ``legacy_unverified`` and a new record's extension
  is ignored — never corrupting — by an old reader;
- wire -> old reader: the protocol model is ``extra=forbid``, so an OLD reader
  REJECTS an extended record. That rejection is the declared legacy surface and
  is pinned FAIL-CLOSED here: rejection, never silent truncation.

The fixtures below are literal on purpose. A record from the shipped build is
not constructed through the model under test: it is the payload that build
actually wrote.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError
import pytest

from chartreux.app_server._model import ProtocolModel, validate_wire
from chartreux.app_server._plan_record import plan_mode_record, session_plan_record
from chartreux.app_server.protocol import (
    SessionPlanRecord,
    SessionPlanScanDomain,
    sessions as protocol_sessions,
)
from chartreux.core.plan_episode import LoggingDomain
from chartreux.core.session_types import (
    LEGACY_UNVERIFIED_STATE,
    PlanModeRecord,
    PlanScanDomain,
)

# Exactly the six keys the shipped (pre-amendment) record model wrote.
LEGACY_PERSISTED: dict[str, object] = {
    "plan_path": "/workspaces/chartreux/tasks/plan-abc123/plan.md",
    "theme": "Fix the widget",
    "planner_model": "glm-5-3",
    "planner_thinking": "high",
    "pre_entry_active_model": "mistral-large-4",
    "pre_entry_planner_thinking": "low",
}

# The same shipped record as it crossed the wire (camelCase, extra=forbid).
LEGACY_WIRE: dict[str, object] = {
    "planPath": "/workspaces/chartreux/tasks/plan-abc123/plan.md",
    "theme": "Fix the widget",
    "plannerModel": "glm-5-3",
    "plannerThinking": "high",
    "preEntryActiveModel": "mistral-large-4",
    "preEntryPlannerThinking": "low",
}

TASK_DIR = "/workspaces/chartreux/tasks/plan-abc123"
EPISODE_ID = "ep-0123456789abcdef0123456789abcdef"
SAVE_DIR = "/home/operator/.chartreux/sessions"
SESSION_PREFIX = "chartreux"

OLD_FIELDS = set(LEGACY_PERSISTED)
EXTENSION_FIELDS = {
    "episode_id",
    "sidecar_path",
    "draft_path",
    "snapshot_path",
    "state",
    "consent_sha256",
    "attempt_generation",
    "scan_domain",
}
# Late sidecar fields the record must NOT mirror (DR-69: P1 mirror only; C3
# re-records the SIDECAR, never the record).
LATE_SIDECAR_FIELDS = {
    "accepted_sha256",
    "executor_session_id",
    "claiming_session_id",
    "owner_domain",
    "c1_verified",
}

_EXTENDED_DOMAIN = PlanScanDomain(save_dir=SAVE_DIR, session_prefix=SESSION_PREFIX)

_EXTENDED = PlanModeRecord(
    plan_path=f"{TASK_DIR}/plan.md",
    theme="Fix the widget",
    planner_model="glm-5-3",
    planner_thinking="high",
    pre_entry_active_model="mistral-large-4",
    pre_entry_planner_thinking="low",
    episode_id=EPISODE_ID,
    sidecar_path=f"{TASK_DIR}/.chartreux-plan-episode.json",
    draft_path=f"{TASK_DIR}/plan.md",
    snapshot_path=f"{TASK_DIR}/accepted-{EPISODE_ID}.md",
    state="accepting",
    consent_sha256="a" * 64,
    attempt_generation=2,
    scan_domain=_EXTENDED_DOMAIN,
)


class _V075PersistedReader(BaseModel):
    """The pre-amendment persisted reader, replicated field-for-field.

    Pydantic's default ``extra='ignore'`` is what the shipped model had, so
    this replica is the old build's tolerance, not a wish about the new one.
    """

    plan_path: str
    theme: str | None = None
    planner_model: str
    planner_thinking: str
    pre_entry_active_model: str | None = None
    pre_entry_planner_thinking: str | None = None


class _V075WireReader(ProtocolModel):
    """The pre-amendment wire reader: ``ProtocolModel`` is ``extra=forbid``."""

    plan_path: str
    theme: str | None = None
    planner_model: str
    planner_thinking: str
    pre_entry_active_model: str | None = None
    pre_entry_planner_thinking: str | None = None


def _wire_payload(record: PlanModeRecord) -> dict[str, object]:
    return session_plan_record(record).model_dump(by_alias=True)


class TestPersistedRecordCompat:
    """Direction (a): the persisted record tolerates extension both ways."""

    def test_a_v075_era_record_loads_as_legacy_unverified(self) -> None:
        record = PlanModeRecord.model_validate(LEGACY_PERSISTED)

        assert record.plan_path == LEGACY_PERSISTED["plan_path"]
        assert record.planner_model == "glm-5-3"
        assert record.pre_entry_active_model == "mistral-large-4"
        assert record.pre_entry_planner_thinking == "low"
        assert record.episode_bound is False
        assert record.legacy_unverified is True
        assert record.display_state == LEGACY_UNVERIFIED_STATE
        assert record.episode_id is None
        assert record.scan_domain is None

    def test_an_old_reader_ignores_the_extension_without_corruption(self) -> None:
        payload = _EXTENDED.model_dump(mode="json")

        old = _V075PersistedReader.model_validate(payload)

        # The extension is dropped on read; the shipped fields survive intact.
        assert old.model_dump() == LEGACY_PERSISTED
        assert set(old.model_dump()) == OLD_FIELDS
        assert not hasattr(old, "episode_id")

    def test_the_persisted_model_does_not_forbid_extra_fields(self) -> None:
        payload = dict(LEGACY_PERSISTED)
        payload["future_episode_hint"] = {"anything": ["a future build writes"]}

        record = PlanModeRecord.model_validate(payload)

        assert PlanModeRecord.model_config.get("extra") != "forbid"
        assert record.plan_path == LEGACY_PERSISTED["plan_path"]
        assert record.legacy_unverified is True

    def test_the_extended_record_round_trips_through_its_json(self) -> None:
        payload = _EXTENDED.model_dump(mode="json")

        restored = PlanModeRecord.model_validate(payload)

        assert restored == _EXTENDED
        assert restored.episode_bound is True
        assert restored.display_state == "accepting"


class TestWireRecordFailClosed:
    """Direction (b): the wire rejects extension in the OLD-reader direction."""

    def test_an_old_wire_reader_rejects_an_extended_record(self) -> None:
        payload = _wire_payload(_EXTENDED)

        with pytest.raises(ValidationError) as excinfo:
            _V075WireReader.model_validate(payload, by_alias=True)

        # Rejection is about the extra fields, not about a broken payload.
        assert "extra_forbidden" in str(excinfo.value)
        assert "episodeId" in str(excinfo.value)

        # Negative control: the same old reader accepts the shipped payload, so
        # the rejection is caused by the extension and nothing else.
        old = _V075WireReader.model_validate(LEGACY_WIRE, by_alias=True)
        assert old.plan_path == LEGACY_PERSISTED["plan_path"]

    def test_the_wire_model_rejects_an_unknown_key(self) -> None:
        payload = dict(LEGACY_WIRE)
        payload["episodeRecordVersion"] = 9

        with pytest.raises(ValidationError) as excinfo:
            validate_wire(SessionPlanRecord, payload)

        assert "extra_forbidden" in str(excinfo.value)

    def test_wire_forbids_extra_while_persisted_tolerates_it(self) -> None:
        assert SessionPlanRecord.model_config.get("extra") == "forbid"
        assert SessionPlanScanDomain.model_config.get("extra") == "forbid"
        # The asymmetry is the design: persisted tolerates, wire fails closed.
        assert PlanModeRecord.model_config.get("extra") != "forbid"

    def test_an_old_wire_payload_still_loads_as_legacy(self) -> None:
        record = validate_wire(SessionPlanRecord, LEGACY_WIRE)

        assert record.legacy_unverified is True
        assert record.display_state == LEGACY_UNVERIFIED_STATE
        assert record.episode_id is None
        # The two layers name one display state; a drift fails here.
        assert protocol_sessions.LEGACY_UNVERIFIED_STATE == LEGACY_UNVERIFIED_STATE


class TestRecordConvertersBidirectional:
    """Both converters move every extension field, direction-explicit (PR-11)."""

    def test_persisted_to_wire_preserves_every_extension_field(self) -> None:
        wire = session_plan_record(_EXTENDED)

        assert wire.episode_id == EPISODE_ID
        assert wire.sidecar_path == _EXTENDED.sidecar_path
        assert wire.draft_path == _EXTENDED.draft_path
        assert wire.snapshot_path == _EXTENDED.snapshot_path
        assert wire.state == "accepting"
        assert wire.consent_sha256 == "a" * 64
        assert wire.attempt_generation == 2
        assert isinstance(wire.scan_domain, SessionPlanScanDomain)
        assert wire.scan_domain.save_dir == SAVE_DIR
        assert wire.scan_domain.session_prefix == SESSION_PREFIX
        assert wire.legacy_unverified is False
        assert wire.display_state == "accepting"

    def test_wire_to_persisted_preserves_every_extension_field(self) -> None:
        wire = session_plan_record(_EXTENDED)

        record = plan_mode_record(wire)

        assert record == _EXTENDED
        assert isinstance(record.scan_domain, PlanScanDomain)
        assert record.display_state == "accepting"

    def test_the_round_trip_is_the_identity_in_both_directions(self) -> None:
        assert plan_mode_record(session_plan_record(_EXTENDED)) == _EXTENDED
        wire = session_plan_record(_EXTENDED)
        assert session_plan_record(plan_mode_record(wire)) == wire

    def test_the_wire_payload_serializes_the_extension_in_camel_case(self) -> None:
        payload = _wire_payload(_EXTENDED)

        for key in (
            "episodeId",
            "sidecarPath",
            "draftPath",
            "snapshotPath",
            "state",
            "consentSha256",
            "attemptGeneration",
            "scanDomain",
        ):
            assert key in payload
        assert payload["scanDomain"] == {
            "saveDir": SAVE_DIR,
            "sessionPrefix": SESSION_PREFIX,
        }

        # And the transport's own read path reconstructs the same record.
        assert validate_wire(SessionPlanRecord, payload) == session_plan_record(
            _EXTENDED
        )

    def test_a_legacy_persisted_record_converts_to_a_legacy_wire_record(self) -> None:
        wire = session_plan_record(PlanModeRecord.model_validate(LEGACY_PERSISTED))

        assert wire.legacy_unverified is True
        assert wire.display_state == LEGACY_UNVERIFIED_STATE
        for field in EXTENSION_FIELDS:
            assert getattr(wire, field) is None


class TestDomainRecordMirror:
    """The record mirrors the P1 scan domain only (DR-69)."""

    def test_the_record_mirrors_the_p1_logging_domain_pair(self) -> None:
        sidecar_domain = LoggingDomain(save_dir=SAVE_DIR, session_prefix=SESSION_PREFIX)

        mirrored = PlanScanDomain(
            save_dir=sidecar_domain.save_dir,
            session_prefix=sidecar_domain.session_prefix,
        )
        record = _EXTENDED.model_copy(update={"scan_domain": mirrored})

        domain = record.scan_domain
        assert domain is not None
        assert domain == mirrored
        assert domain.save_dir == sidecar_domain.save_dir
        assert domain.session_prefix == sidecar_domain.session_prefix
        assert set(PlanScanDomain.model_fields) == {"save_dir", "session_prefix"}
        assert set(SessionPlanScanDomain.model_fields) == {"save_dir", "session_prefix"}

    def test_the_domain_mirror_round_trips_through_the_wire(self) -> None:
        wire = session_plan_record(_EXTENDED)

        assert wire.scan_domain is not None
        assert wire.scan_domain.save_dir == _EXTENDED_DOMAIN.save_dir
        assert wire.scan_domain.session_prefix == _EXTENDED_DOMAIN.session_prefix
        assert plan_mode_record(wire).scan_domain == _EXTENDED.scan_domain

    def test_the_record_carries_the_p1_mirror_and_no_late_sidecar_fields(self) -> None:
        record_fields = set(PlanModeRecord.model_fields)

        # The two models carry the same episode vocabulary.
        assert record_fields == set(SessionPlanRecord.model_fields)
        assert EXTENSION_FIELDS <= record_fields
        assert not (LATE_SIDECAR_FIELDS & record_fields)
