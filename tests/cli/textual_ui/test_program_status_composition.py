"""Composed-ownership OSC status matrix (v0.7.9 point 5, package C7a).

Design of record: tasks/v0.7.9/session-status-design-final.md §3.1-§3.3 and
§5 acceptance 1-3, 5 and 8. The pure ownership classes are exercised on the
composition record directly; every observable sequence below is a wire
report captured from the real app over the real emitter writing to the
recording fake driver — ``W``/``D``/``I`` are the exact bytes of design §5.0.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import pytest
from textual.app import App, ComposeResult

from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.cli.textual_ui.app import _BashSettlement
from chartreux.cli.textual_ui.program_status_composition import (
    CallbackOwner,
    ComposedDecision,
    DeferredRecord,
    ProgramStatusComposition,
    ReleaseOutcome,
    compose_status,
)
from chartreux.cli.textual_ui.turn_supervisor import TurnOutcome
from chartreux.cli.textual_ui.widgets.agent_bar import (
    AgentBar,
    agent_group,
    agent_is_active,
    agent_state,
)
from chartreux.core.subagents import derive_agent_name
from tests.cli.textual_ui.test_program_status import (
    _callback,
    _StaticLivenessInventory,
    _stub_supervisor_ui,
    _wire_app,
    _wire_states,
)

_W = "\x1b]7501;state=working:app=chartreux\x07"
_D = "\x1b]7501;state=done:app=chartreux\x07"
_I = "\x1b]7501;state=idle:app=chartreux\x07"


def _agent(agent_id: str = "agent-1", **kwargs: Any) -> AgentSummaryModel:
    values: dict[str, Any] = {"identity": "coder", "availability": "idle"}
    values.update(kwargs)
    values.setdefault(
        "name",
        derive_agent_name(
            agent_id=agent_id,
            identity=values["identity"],
            identity_display_label=None,
            base_model=values.get("base_model"),
            effective_model=values.get("effective_model"),
        ),
    )
    return AgentSummaryModel(agent_id=agent_id, **values)


class _BarApp(App):
    def compose(self) -> ComposeResult:
        yield AgentBar()


def _children(app: Any, *availabilities: str) -> None:
    """Inject child activity through the liveness seam (C7b): composition
    consumes ONLY the snapshot-carried inventory, so the tests pin the
    poller's inventory directly.
    """
    app._agent_liveness = _StaticLivenessInventory(
        agents=[
            _agent(f"agent-{index}", availability=availability)
            for index, availability in enumerate(availabilities)
        ]
    )


def _wire_reports(host: Any) -> list[str]:
    return [call.args[0] for call in host._driver.write.call_args_list]


async def _finalize(app: Any, outcome: TurnOutcome, turn_id: str) -> None:
    await app._turn_supervisor.finalize_turn_ui(outcome=outcome, turn_id=turn_id)


# ---------------------------------------------------------------------------
# Ownership classes on the composition record (design §3.1)
# ---------------------------------------------------------------------------


class TestCompositionOwnershipClasses:
    def test_durable_take_advances_the_epoch_and_invalidates_the_record(self) -> None:
        composition = ProgramStatusComposition()
        composition.note_release(ReleaseOutcome("done"))
        assert composition.deferred == DeferredRecord("done", epoch=0)
        composition.take()
        assert composition.epoch == 1
        assert composition.deferred is None

    def test_transient_taking_saves_the_deferred_record(self) -> None:
        composition = ProgramStatusComposition()
        composition.note_release(ReleaseOutcome("done"))
        epoch = composition.epoch
        composition.begin_transient()
        assert composition.epoch == epoch
        assert composition.deferred == DeferredRecord("done", epoch=epoch)

    def test_a_dismissal_restores_the_saved_record(self) -> None:
        composition = ProgramStatusComposition()
        composition.note_release(ReleaseOutcome("done"))
        composition.note_release(None)
        assert composition.deferred == DeferredRecord("done", epoch=0)

    def test_the_error_release_takes_the_persistent_rule_2_owner(self) -> None:
        composition = ProgramStatusComposition()
        composition.note_release(ReleaseOutcome("done"))
        composition.note_release(ReleaseOutcome("error", message="boom"))
        assert composition.deferred is None
        assert composition.error_owner_message == "boom"
        # It persists through child activity until a later owner replaces it.
        decision = composition.decision(turn_active=True, child_active=True)
        assert decision == ComposedDecision("error", message="boom")

    def test_the_stale_deferred_record_cannot_speak_after_its_epoch(self) -> None:
        decision = compose_status(
            callback_owner=None,
            error_owner_message=None,
            turn_active=False,
            child_active=False,
            deferred=DeferredRecord("done", epoch=1),
            composed_epoch=2,
        )
        assert decision == ComposedDecision("idle")

    def test_rule_table_precedence(self) -> None:
        blocked = CallbackOwner(kind="question", message="A?")
        # Rule 1 beats rules 2-4.
        assert compose_status(
            callback_owner=blocked,
            error_owner_message="boom",
            turn_active=True,
            child_active=True,
            deferred=DeferredRecord("done", epoch=0),
            composed_epoch=0,
        ) == ComposedDecision("blocked", kind="question", message="A?")
        # Rule 2 beats rules 3-4.
        assert compose_status(
            callback_owner=None,
            error_owner_message="boom",
            turn_active=True,
            child_active=True,
            deferred=DeferredRecord("done", epoch=0),
            composed_epoch=0,
        ) == ComposedDecision("error", message="boom")
        # Rule 3 beats rule 4.
        assert compose_status(
            callback_owner=None,
            error_owner_message=None,
            turn_active=False,
            child_active=True,
            deferred=DeferredRecord("done", epoch=0),
            composed_epoch=0,
        ) == ComposedDecision("working")
        # Rule 4: the freeing outcome, then the saved record, then the prompt.
        assert compose_status(
            callback_owner=None,
            error_owner_message=None,
            turn_active=False,
            child_active=False,
            deferred=DeferredRecord("done", epoch=0),
            composed_epoch=0,
            releasing_outcome=ReleaseOutcome("idle"),
        ) == ComposedDecision("idle")
        assert compose_status(
            callback_owner=None,
            error_owner_message=None,
            turn_active=False,
            child_active=False,
            deferred=DeferredRecord("done", epoch=0),
            composed_epoch=0,
        ) == ComposedDecision("done")
        assert compose_status(
            callback_owner=None,
            error_owner_message=None,
            turn_active=False,
            child_active=False,
            deferred=None,
            composed_epoch=0,
        ) == ComposedDecision("idle")


# ---------------------------------------------------------------------------
# Observable sequences (design §5 acceptance 1-3)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
class TestComposedWireSequences:
    # The generous per-node timeout replaces the repo's 10s global for these
    # nodes only: each one boots a real Textual app and under the mandated
    # parallel -n auto suite that boot can be descheduled well past 10s on a
    # fully loaded host (the C7b parallel-load flake) — the composition
    # assertions themselves are deterministic and wall-clock-free.
    @pytest.mark.asyncio
    async def test_completed_turn_defers_done_until_no_child_is_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Acceptance 1: with one active child and a responsive owner, main
        # completion produces no D or I — the sequence stays W — and D lands
        # only once the child is no longer active.
        app, host = _wire_app()
        async with app.run_test():
            await app._session_ready.wait()
            _stub_supervisor_ui(app, monkeypatch)
            host._driver.write.reset_mock()
            _children(app, "running")
            app._record_turn_owner("turn-1")
            app._refresh_composed_status(turn_active=True)
            await _finalize(app, TurnOutcome.COMPLETED, "turn-1")
            assert _wire_states(host) == ["working"]
            # The child finishes; fresh activity drives the same shared
            # recompute (the C7b snapshot-completion site).
            _children(app)
            app._refresh_composed_status(turn_active=False)
            assert _wire_reports(host) == [_W, _D]

    @pytest.mark.asyncio
    async def test_two_children_finishing_separately_emit_done_after_the_second(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Acceptance 1: D only after the second child becomes inactive.
        app, host = _wire_app()
        async with app.run_test():
            await app._session_ready.wait()
            _stub_supervisor_ui(app, monkeypatch)
            host._driver.write.reset_mock()
            _children(app, "running", "running")
            app._record_turn_owner("turn-1")
            app._refresh_composed_status(turn_active=True)
            await _finalize(app, TurnOutcome.COMPLETED, "turn-1")
            _children(app, "running")
            app._refresh_composed_status(turn_active=False)
            assert _wire_states(host) == ["working"]
            _children(app)
            app._refresh_composed_status(turn_active=False)
            assert _wire_reports(host) == [_W, _D]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("active_child", [True, False])
    async def test_interrupt_emits_idle_only_after_the_child_finishes(
        self, active_child: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Acceptance 2: interruption with an active child produces W, I only
        # after the child finishes; with no child the immediate I stands.
        app, host = _wire_app()
        async with app.run_test():
            await app._session_ready.wait()
            _stub_supervisor_ui(app, monkeypatch)
            host._driver.write.reset_mock()
            if active_child:
                _children(app, "running")
            app._record_turn_owner("turn-1")
            app._refresh_composed_status(turn_active=True)
            app._turn_supervisor.agent_task = asyncio.create_task(
                asyncio.Event().wait()
            )
            try:
                await app._turn_supervisor.interrupt_turn()
                if active_child:
                    # The interrupt's idle is deferred: still no I.
                    assert _wire_states(host) == ["working"]
                    _children(app)
                    app._refresh_composed_status(turn_active=False)
            finally:
                app._turn_supervisor.agent_task.cancel()
                with suppress(asyncio.CancelledError):
                    await app._turn_supervisor.agent_task
            assert _wire_reports(host) == [_W, _I]

    @pytest.mark.asyncio
    async def test_successor_turn_never_emits_the_predecessors_deferred_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Acceptance 3: a successor's durable taking invalidated the old
        # deferred done; the sequence never shows it.
        app, host = _wire_app()
        async with app.run_test():
            await app._session_ready.wait()
            _stub_supervisor_ui(app, monkeypatch)
            host._driver.write.reset_mock()
            _children(app, "running")
            app._record_turn_owner("turn-1")
            app._refresh_composed_status(turn_active=True)
            await _finalize(app, TurnOutcome.COMPLETED, "turn-1")
            # The successor starts before the last child settles.
            _children(app, "running")
            app._record_turn_owner("turn-2")
            app._refresh_composed_status(turn_active=True)
            _children(app)
            await _finalize(app, TurnOutcome.INTERRUPTED, "turn-2")
            # The predecessor's done never emitted; idle settled instead.
            assert _wire_reports(host) == [_W, _I]

    @pytest.mark.asyncio
    async def test_an_open_callback_masks_children_until_dismissed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Acceptance 3: while a callback owns blocked, child start/finish
        # cannot overwrite it; the dismissal restores the deferred done.
        app, host = _wire_app()
        async with app.run_test():
            await app._session_ready.wait()
            _stub_supervisor_ui(app, monkeypatch)
            host._driver.write.reset_mock()
            _children(app, "running")
            app._record_turn_owner("turn-1")
            app._refresh_composed_status(turn_active=True)
            await _finalize(app, TurnOutcome.COMPLETED, "turn-1")
            callback = _callback(origin="model_question")
            app._turn_supervisor.active_callback = callback
            app._turn_supervisor._emit_blocked_callback(callback)
            # Child updates while the prompt owns the record change nothing.
            _children(app)
            app._refresh_composed_status(turn_active=False)
            _children(app, "running")
            app._refresh_composed_status(turn_active=False)
            assert _wire_states(host) == ["working", "blocked"]
            # The dismissal still masks behind the active child: working.
            app._turn_supervisor.active_callback = None
            app._settle_composed_status(outcome=None, turn_active=False)
            assert _wire_states(host) == ["working", "blocked", "working"]
            # With the child finished, the dismissal restores the saved
            # deferred done: not idle.
            _children(app)
            app._refresh_composed_status(turn_active=False)
            assert _wire_states(host) == ["working", "blocked", "working", "done"]

    @pytest.mark.asyncio
    async def test_error_ownership_survives_child_updates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Acceptance 3 / rule 2: the error release persists through child
        # updates until a later owner replaces it.
        app, host = _wire_app()
        async with app.run_test():
            await app._session_ready.wait()
            _stub_supervisor_ui(app, monkeypatch)
            host._driver.write.reset_mock()
            app._settle_bash(_BashSettlement(), "error", "Command failed: badcmd")
            assert _wire_states(host) == ["error"]
            _children(app, "running")
            app._refresh_composed_status(turn_active=False)
            _children(app)
            app._refresh_composed_status(turn_active=False)
            assert _wire_states(host) == ["error"]
            # A subsequent valid turn replaces the error owner.
            app._record_turn_owner("turn-1")
            app._refresh_composed_status(turn_active=True)
            assert _wire_states(host) == ["error", "working"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("deferred_state", "shell_state", "shell_msg", "expected"),
        [
            # Deferred idle, then a successful `!ls`: done.
            ("idle", "done", None, ["working", "done"]),
            # Deferred done, then a failed `!badcmd`: error(msg).
            ("done", "error", "Command failed: badcmd", ["working", "error"]),
            # Deferred done, then a cancelled shell: the shell's own idle.
            ("done", "idle", None, ["working", "idle"]),
        ],
    )
    async def test_shell_release_precedence_replaces_the_saved_record(
        self,
        deferred_state: str,
        shell_state: str,
        shell_msg: str | None,
        expected: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Release precedence (design §3.1): the shell settle REPLACES the
        # saved record with its own settlement outcome — the most recent
        # owner's terminal record wins.
        app, host = _wire_app()
        async with app.run_test():
            await app._session_ready.wait()
            _stub_supervisor_ui(app, monkeypatch)
            host._driver.write.reset_mock()
            _children(app, "running")
            app._settle_composed_status(
                outcome=ReleaseOutcome(deferred_state), turn_active=False
            )
            assert _wire_states(host) == ["working"]
            _children(app)
            app._settle_bash(_BashSettlement(), shell_state, shell_msg)
            assert _wire_states(host) == expected
            if shell_state == "error":
                # The error-kind release keeps owning past the settle.
                app._refresh_composed_status(turn_active=False)
                assert _wire_states(host) == expected


@pytest.mark.timeout(60)
class TestCompositionSource:
    """§3.2 composition source: snapshot-carried inventories only — the
    event-populated browser store (agent_browser.py:154) stays a refresh
    hint and never an emission input.
    """

    @pytest.mark.asyncio
    async def test_event_store_agents_never_drive_composition(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, host = _wire_app()
        async with app.run_test():
            await app._session_ready.wait()
            _stub_supervisor_ui(app, monkeypatch)
            host._driver.write.reset_mock()
            # The browser hint store says a child is running; the poller's
            # snapshot inventory is empty. The settle composes done.
            app._agent_summaries = [_agent(availability="running")]
            app._settle_composed_status(
                outcome=ReleaseOutcome("done"), turn_active=False
            )
            assert _wire_states(host) == ["done"]


# ---------------------------------------------------------------------------
# Lifecycle matrix (design §5 acceptance 5)
# ---------------------------------------------------------------------------

# (agent fields, OSC-active, expected collapsed-summary, bar group)
_LIFECYCLE_ROWS: list[tuple[dict[str, Any], bool, str, str]] = [
    ({"availability": "running"}, True, "1 running", "active"),
    # A compacting agent's availability stays running: the bar counts it —
    # ALIGNED with OSC (design §3.1).
    ({"availability": "running", "compacting": True}, True, "1 running", "active"),
    # RESTORING: OSC-active, but the collapsed bar's active counts key on
    # running/finalizing availability, so it counts neither — the documented
    # divergence (design §3.1).
    ({"availability": "restoring"}, True, "", "active"),
    # FINALIZING: the bar counts it as finishing; OSC does not — the
    # documented divergence (design §3.1).
    ({"availability": "finalizing"}, False, "1 finishing", "active"),
    ({"availability": "suspended"}, False, "", "active"),
    ({"availability": "idle"}, False, "1 idle", "idle"),
    ({"availability": "closed"}, False, "1 recent", "recent"),
    ({"availability": "failed"}, False, "1 idle", "idle"),
    ({"availability": "evicted"}, False, "1 recent", "recent"),
]


@pytest.mark.timeout(60)
class TestLifecycleMatrix:
    @pytest.mark.parametrize(
        ("fields", "osc_active", "bar_summary", "group"), _LIFECYCLE_ROWS
    )
    def test_osc_active_and_bar_activity_per_lifecycle_state(
        self, fields: dict[str, Any], osc_active: bool, bar_summary: str, group: str
    ) -> None:
        agent = _agent(**fields)
        assert agent_is_active(agent) is osc_active
        assert (
            agent_state(agent) in {"running", "compacting", "restoring"}
        ) is osc_active
        assert agent_group(agent) == group

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fields", "osc_active", "bar_summary", "group"), _LIFECYCLE_ROWS
    )
    async def test_every_lifecycle_state_drives_the_composed_sites_the_same_way(
        self,
        fields: dict[str, Any],
        osc_active: bool,
        bar_summary: str,
        group: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Only running, compacting, and restoring count (acceptance 5): a
        # done settle defers to working for those three and lands D for the
        # rest. Startup, abort, reset, recovery, shell settle, interrupt,
        # both finalizers and both callback dismissals all compute through
        # the one shared table, so this settle stands for the site set.
        app, host = _wire_app()
        async with app.run_test():
            await app._session_ready.wait()
            _stub_supervisor_ui(app, monkeypatch)
            host._driver.write.reset_mock()
            _children(app, fields["availability"])
            app._settle_composed_status(
                outcome=ReleaseOutcome("done"), turn_active=False
            )
            assert _wire_states(host) == ["working" if osc_active else "done"]

        bar_app = _BarApp()
        async with bar_app.run_test():
            bar = bar_app.query_one(AgentBar)
            bar.update_agents((_agent(**fields),))
            # The collapsed bar's activity signal per lifecycle state, so
            # OSC-active and bar-visible activity cannot silently disagree.
            assert bar.collapsed_summary(80) == bar_summary


# ---------------------------------------------------------------------------
# Emitter preservation (design §5 acceptance 8)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
class TestEmitterPreservation:
    # Same parallel-load rationale as TestComposedWireSequences: these boot
    # the real app over the real emitter.
    @pytest.mark.asyncio
    async def test_no_change_refresh_neither_calls_emit_nor_advances_generation(
        self,
    ) -> None:
        # A no-change refresh keeps the wire AND the emitter generation: it
        # never calls emit at all (acceptance 8).
        app, host = _wire_app()
        async with app.run_test():
            await app._session_ready.wait()
            generation = app._program_status.generation
            writes = host._driver.write.call_count
            app._refresh_composed_status(turn_active=False)
            app._refresh_composed_status(turn_active=False)
            assert host._driver.write.call_count == writes
            assert app._program_status.generation == generation
            # A changed refresh does emit.
            app._refresh_composed_status(turn_active=True)
            assert app._program_status.generation == generation + 1
            assert _wire_states(host)[-1] == "working"

    @pytest.mark.asyncio
    async def test_the_deferral_clock_is_the_composed_epoch_not_the_generation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every emit call advances the emitter generation, deduplicated and
        # guard-suppressed ones included (program_status emitter). The
        # generation is NOT the deferral clock: raw emits never orphan a
        # pending deferred outcome.
        app, host = _wire_app()
        async with app.run_test():
            await app._session_ready.wait()
            _stub_supervisor_ui(app, monkeypatch)
            host._driver.write.reset_mock()
            _children(app, "running")
            app._record_turn_owner("turn-1")
            app._refresh_composed_status(turn_active=True)
            await _finalize(app, TurnOutcome.INTERRUPTED, "turn-1")
            assert _wire_states(host) == ["working"]
            assert app._status_composition.deferred == DeferredRecord(
                "idle", epoch=app._status_composition.epoch
            )
            generation = app._program_status.generation
            app._program_status.emit("blocked", kind="question", msg="noise")
            assert app._program_status.generation > generation
            # The deferral survived the raw emits and settles at the
            # composed epoch: idle, once no child remains active.
            _children(app)
            app._settle_composed_status(outcome=None, turn_active=False)
            assert _wire_states(host) == ["working", "blocked", "idle"]
            assert _wire_reports(host)[-1] == _I
