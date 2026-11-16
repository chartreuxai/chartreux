from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel
import pytest

from chartreux.core.agent_loop._safety_authority import (
    AUTHORITY_TRANSITION_EVENT,
    AUTHORITY_TRANSITION_PAIRS,
    AuthorityActor,
    AuthorityLaunchSnapshot,
    AuthorityTransition,
    DecidedCallIdentity,
    SafetyLevel,
    SessionAuthorityState,
    initial_level,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config


class _Args(BaseModel):
    command: str


class TestSafetyLevelSurface:
    def test_single_surface_values(self) -> None:
        assert [level.value for level in SafetyLevel] == ["safe", "smart", "dangerous"]

    def test_exactly_six_ordered_pairs(self) -> None:
        assert len(AUTHORITY_TRANSITION_PAIRS) == 6
        assert set(AUTHORITY_TRANSITION_PAIRS) == {
            (a, b) for a in SafetyLevel for b in SafetyLevel if a != b
        }


class TestInitialLevel:
    def test_default_config_is_safe(self) -> None:
        # A-P2 lands the [safety] section; its default_level is "safe".
        config = build_test_vibe_config()
        assert config.safety.default_level == "safe"
        assert initial_level(config) is SafetyLevel.SAFE

    def test_smart_default_level_reads_through_the_seam(self) -> None:
        config = build_test_vibe_config(safety={"default_level": "smart"})
        assert initial_level(config) is SafetyLevel.SMART

    def test_unexpected_values_fail_closed_to_safe(self) -> None:
        # DANGEROUS is ceremony-only and never an initial level; anything
        # unexpected (or a config without the section) fails closed to SAFE.
        assert initial_level(build_test_vibe_config()) is SafetyLevel.SAFE

        class _NoSection:
            pass

        assert initial_level(_NoSection()) is SafetyLevel.SAFE  # type: ignore[arg-type]


class TestAuthorityTransitions:
    def test_all_six_ordered_pairs_emit_events(self) -> None:
        emitted: list[AuthorityTransition] = []
        state = SessionAuthorityState(
            session_id="sess-1", level=SafetyLevel.SAFE, sink=emitted.append
        )
        path = [
            SafetyLevel.SMART,
            SafetyLevel.DANGEROUS,
            SafetyLevel.SAFE,
            SafetyLevel.DANGEROUS,
            SafetyLevel.SMART,
            SafetyLevel.SAFE,
        ]
        for to in path:
            assert state.transition(to, actor=AuthorityActor.HUMAN) is not None
        assert len(emitted) == 6
        assert {(e.from_level, e.to_level) for e in emitted} == set(
            AUTHORITY_TRANSITION_PAIRS
        )
        for event in emitted:
            assert event.session_id == "sess-1"
            assert event.actor == "human"
            assert isinstance(event.ts, datetime)
            assert event.event_name == AUTHORITY_TRANSITION_EVENT
        assert state.transitions == tuple(emitted)

    def test_event_serializes_with_from_to_fields(self) -> None:
        state = SessionAuthorityState(session_id="sess-2", level=SafetyLevel.SAFE)
        event = state.transition(SafetyLevel.DANGEROUS, actor="dangerous")
        assert event is not None
        payload = event.model_dump(by_alias=True, mode="json")
        assert payload["from"] == "safe"
        assert payload["to"] == "dangerous"
        assert payload["actor"] == "dangerous"
        assert payload["session_id"] == "sess-2"
        assert "ts" in payload

    def test_same_level_is_a_noop(self) -> None:
        emitted: list[AuthorityTransition] = []
        state = SessionAuthorityState(
            session_id="sess-3", level=SafetyLevel.SMART, sink=emitted.append
        )
        assert state.transition(SafetyLevel.SMART, actor="human") is None
        assert emitted == []
        assert state.level is SafetyLevel.SMART


class TestDecidedCallMemory:
    def test_identity_hashes_canonical_args(self) -> None:
        same_a = DecidedCallIdentity.of("bash", _Args(command="ls"))
        same_b = DecidedCallIdentity.of("bash", _Args(command="ls"))
        other = DecidedCallIdentity.of("bash", _Args(command="rm -rf /"))
        other_tool = DecidedCallIdentity.of("write_file", _Args(command="ls"))
        assert same_a == same_b
        assert same_a != other
        assert same_a != other_tool

    def test_record_lookup_and_authorizer_tag(self) -> None:
        state = SessionAuthorityState(session_id="sess-4", level=SafetyLevel.SMART)
        identity = DecidedCallIdentity.of("bash", _Args(command="ls"))
        assert identity not in state.decided_calls
        record = state.record_decided_call(
            identity, decision="override", actor="classifier"
        )
        assert record.authorizer == "smart"
        assert record.decision == "override"
        assert state.decided_calls.lookup(identity) is record
        assert identity in state.decided_calls

    def test_authorizer_tag_tracks_level_in_force(self) -> None:
        state = SessionAuthorityState(session_id="sess-5", level=SafetyLevel.SAFE)
        identity = DecidedCallIdentity.of("bash", _Args(command="ls"))
        record = state.record_decided_call(identity, decision="refused", actor="human")
        assert record.authorizer == "safe"
        state.transition(SafetyLevel.DANGEROUS, actor="human")
        again = state.record_decided_call(
            DecidedCallIdentity.of("bash", _Args(command="rm -rf /")),
            decision="granted",
            actor="human",
        )
        assert again.authorizer == "dangerous"


class TestLaunchSnapshot:
    def test_capture_encode_decode_round_trip(self) -> None:
        state = SessionAuthorityState(session_id="parent", level=SafetyLevel.DANGEROUS)
        state.record_decided_call(
            DecidedCallIdentity.of("bash", _Args(command="ls")),
            decision="granted",
            actor="human",
        )
        snapshot = state.launch_snapshot()
        assert snapshot.level is SafetyLevel.DANGEROUS
        assert len(snapshot.decided_calls) == 1
        assert AuthorityLaunchSnapshot.decode(snapshot.encode()) == snapshot

    def test_decode_none_and_malformed(self) -> None:
        assert AuthorityLaunchSnapshot.decode(None) is None
        with pytest.raises(ValueError):
            AuthorityLaunchSnapshot.decode({"level": "chaotic"})
        with pytest.raises(ValueError):
            AuthorityLaunchSnapshot.decode("safe")
        with pytest.raises(ValueError):
            AuthorityLaunchSnapshot.decode({"level": "safe", "decided_calls": {}})

    def test_child_inherits_level_without_any_ceremony(self) -> None:
        # Children inherit the parent's level as an explicit launch snapshot:
        # construction from the snapshot emits no transition event (children
        # never run the DANGEROUS ceremony).
        emitted: list[AuthorityTransition] = []
        parent = SessionAuthorityState(
            session_id="parent", level=SafetyLevel.DANGEROUS, sink=emitted.append
        )
        snapshot = parent.launch_snapshot()
        child = SessionAuthorityState(
            session_id="child", level=snapshot.level, sink=emitted.append
        )
        assert child.level is SafetyLevel.DANGEROUS
        assert emitted == []

    def test_audit_tag(self) -> None:
        state = SessionAuthorityState(session_id="sess-6", level=SafetyLevel.SAFE)
        assert state.audit_tag() == {"authorizer": "safe"}


class TestAgentLoopAuthorityState:
    def test_loop_exposes_authority_state_at_hard_safe(self) -> None:
        loop = build_test_agent_loop()
        assert loop.safety_level is SafetyLevel.SAFE
        assert loop.authority_state.session_id == loop.session_id
        assert loop.authority_state.transitions == ()

    def test_transition_rides_the_out_of_band_event_queue(self) -> None:
        loop = build_test_agent_loop()
        event = loop.authority_state.transition(SafetyLevel.SMART, actor="human")
        assert loop.safety_level is SafetyLevel.SMART
        queued: list[AuthorityTransition] = []
        while not loop._out_of_band_events.empty():
            item = loop._out_of_band_events.get_nowait()
            if isinstance(item, AuthorityTransition):
                queued.append(item)
        assert queued == [event]

    def test_child_runtime_policy_captures_current_level(self) -> None:
        loop = build_test_agent_loop()
        loop.authority_state.transition(SafetyLevel.DANGEROUS, actor="human")
        policy = loop.child_runtime_policy
        assert policy.authority_snapshot is not None
        assert policy.authority_snapshot.level is SafetyLevel.DANGEROUS

    def test_loop_constructs_from_inherited_snapshot(self) -> None:
        loop = build_test_agent_loop(
            inherited_authority=AuthorityLaunchSnapshot(level=SafetyLevel.SMART)
        )
        assert loop.safety_level is SafetyLevel.SMART
