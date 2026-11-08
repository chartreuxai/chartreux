"""WP5a + WP5b contract nodes: the five landed runtime ablations.

The nodes prove the ablations do the three things r12 demands of the whole
family: they ACTIVATE (a bracket that silently never replaced anything would
report a fabricated upper bound), they stay OBSERVATION-ONLY with the pinned
declaration — no equivalence claim beyond the coalescing one, no gated
quantity, no production change — and they RESTORE the prior objects so the
next cell is untouched.

Their checkable properties are asserted on the real machinery rather than on
prose: the transport ablation preserves the serialized event stream it delivers
(it removes cost, not events) and performs no transport-level JSON encoding at
all, while the real path performs exactly one per notification; the integrity
stub installs disposable data that carries no committed fixture lineage, so it
can never stand in for a gated run, and never touches the committed fixture;
the render ablation streams a real turn whose text arrives intact while its
markdown widget stays empty; the GC ablation allocates past the gen-0 threshold
with zero collections and restores the collector; and the coalescing ablation
delivers one write per turn instead of one per delta while reproducing the
per-delta run's final text in the same order — the one equivalence plan §3
WP5b pins. The cost difference itself is recorded as a ``gated: False``
observation — never asserted, because an upper bound on a compound ablation is
not a threshold result.

Negative controls keep the nodes from passing vacuously: a drifted declaration,
an equivalence claim outside the pinned one, a combined ablation selection and
an unknown ablation id all have to fail.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import gc
import os
from pathlib import Path
import socket
import sys
from time import perf_counter

import pytest
from textual.containers import Vertical
from textual.pilot import Pilot

from chartreux.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    StreamingMessageBase,
    UserMessage,
)
from tests.perf._runtime_ablations import (
    ABLATION_APPLICATION_POLICY,
    ABLATION_COMBINATIONS,
    ABLATION_GC_DISABLED,
    ABLATION_INTEGRITY_STUB,
    ABLATION_MARKDOWN_NO_RENDER,
    ABLATION_SCHEMA,
    ABLATION_SOURCE_COALESCE,
    ABLATION_SPECS,
    ABLATION_TRANSPORT_OBJECT,
    FINAL_TEXT_ORDERING,
    INTEGRITY_STUB_MESSAGES,
    CollectionCounter,
    SerializationCounter,
    ablate_gc_disabled,
    ablate_integrity_stub,
    ablate_markdown_no_render,
    ablate_source_coalesce,
    ablate_transport_object,
    ablation_spec,
    ablation_wall_observation,
    assert_ablation_declaration,
    assert_single_ablation,
    build_integrity_stub,
    fixture_immutability_probe,
    integrity_verdict,
    rendered_markdown,
)
from tests.perf._runtime_fixture import (
    FIXTURE_HASH,
    FIXTURE_MESSAGE_COUNT,
    fixture_digest,
)
from tests.perf._runtime_pilot import (
    PILOT_ADAPTER,
    PILOT_FRAME_DELAY_S,
    PILOT_HISTORY_CONTAINER_ID,
    PILOT_SETTLE_POLL_S,
    PilotScenarioApp,
    pilot_render_observer,
)
from tests.perf._runtime_scenario import (
    CELL_FRESH_HEADLESS,
    CONFIG_HOME_ENV,
    PROVIDER_CREDENTIAL_NAMES,
    SHORTENED_SCALE,
    WORKLOAD_HASHES,
    EgressGuard,
    build_headless_scenario,
    response_chunks,
    root_user_content,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux", reason="the runtime ablation cells are Linux-only"
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_ABLATION_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

_CONTRACT_TURNS = 2
_EgressProbe = "192.0.2.1"
_SETTLE_TIMEOUT_S = 5.0

# The app-path comparison: two turns, because turn ORDERING is one of the two
# equivalences the coalescing ablation asserts and a single turn could not show
# it. The render bracket needs no ordering, so it streams one turn.
_RENDER_TURNS = 2
_RESTORED_SUFFIX = "restored frame"
_SETTLE_DEADLINE_S = 1.0
# Past gen-0's default threshold (700 allocations), so a collector that was
# actually left enabled would have collected inside the disabled window.
_GC_ALLOCATIONS = 2000


def _assert_ablation_declarations() -> None:
    """Every ablation carries its pinned declaration, and never more than it."""
    assert set(ABLATION_SPECS) == {
        ABLATION_TRANSPORT_OBJECT,
        ABLATION_INTEGRITY_STUB,
        ABLATION_MARKDOWN_NO_RENDER,
        ABLATION_GC_DISABLED,
        ABLATION_SOURCE_COALESCE,
    }
    assert ABLATION_COMBINATIONS is None
    assert ABLATION_APPLICATION_POLICY == "one_at_a_time"
    assert ABLATION_SCHEMA == "chartreux.runtime_ablation.v1"

    transport = assert_ablation_declaration(ABLATION_TRANSPORT_OBJECT)
    assert transport.interpretation == "compound_upper_bound"
    assert transport.equivalence == "not_asserted"
    assert transport.equivalence_reason == "compound_upper_bound"
    assert transport.observed_only is True
    assert transport.production is False
    assert "equivalence NOT asserted" in transport.declaration

    stub = assert_ablation_declaration(ABLATION_INTEGRITY_STUB)
    assert stub.interpretation == "disposable_data"
    assert stub.equivalence == "not_asserted"
    assert stub.equivalence_reason == "disposable_data"
    assert stub.observed_only is True
    assert stub.production is False

    render = assert_ablation_declaration(ABLATION_MARKDOWN_NO_RENDER)
    assert render.interpretation == "render_component"
    assert render.equivalence == "not_asserted"
    assert render.equivalence_reason == "rendering_removed"
    assert render.observed_only is True
    assert render.production is False
    assert "equivalence NOT asserted" in render.declaration

    gc_ablation = assert_ablation_declaration(ABLATION_GC_DISABLED)
    assert gc_ablation.interpretation == "collector_disabled"
    assert gc_ablation.equivalence == "not_asserted"
    assert gc_ablation.equivalence_reason == "process_state"
    assert gc_ablation.observed_only is True
    assert gc_ablation.production is False
    assert "equivalence NOT asserted" in gc_ablation.declaration

    coalesce = assert_ablation_declaration(ABLATION_SOURCE_COALESCE)
    assert coalesce.interpretation == "delta_coalescing"
    assert coalesce.equivalence == FINAL_TEXT_ORDERING
    assert coalesce.equivalence == "final_text_ordering"
    assert coalesce.equivalence_reason == "final_text_ordering"
    assert coalesce.observed_only is True
    assert coalesce.production is False
    assert "final-text/ordering equivalence" in coalesce.declaration
    assert "ASSERTED" in coalesce.declaration

    # One at a time: a combined selection can never be attributed to a
    # component, so it is refused rather than recorded.
    for single in (
        ABLATION_TRANSPORT_OBJECT,
        ABLATION_INTEGRITY_STUB,
        ABLATION_MARKDOWN_NO_RENDER,
        ABLATION_GC_DISABLED,
        ABLATION_SOURCE_COALESCE,
    ):
        assert assert_single_ablation([single]) == single
    for selection in (
        [],
        [ABLATION_TRANSPORT_OBJECT, ABLATION_INTEGRITY_STUB],
        [ABLATION_INTEGRITY_STUB, ABLATION_INTEGRITY_STUB],
        [ABLATION_MARKDOWN_NO_RENDER, ABLATION_SOURCE_COALESCE],
    ):
        with pytest.raises(AssertionError):
            assert_single_ablation(selection)

    # Negative controls: a declaration that claims more than r12 allows fails.
    with pytest.raises(AssertionError):
        assert_ablation_declaration(replace(transport, equivalence="asserted"))
    with pytest.raises(AssertionError):
        assert_ablation_declaration(replace(transport, production=True))
    with pytest.raises(AssertionError):
        assert_ablation_declaration(replace(stub, observed_only=False))
    # Equivalence is pinned per ablation: a bracket that does not reproduce its
    # component's output may not claim the coalescing one, and the coalescing
    # bracket may not drop the claim S1-E(a)'s re-entry rests on.
    with pytest.raises(AssertionError):
        assert_ablation_declaration(replace(render, equivalence=FINAL_TEXT_ORDERING))
    with pytest.raises(AssertionError):
        assert_ablation_declaration(replace(gc_ablation, equivalence="asserted"))
    with pytest.raises(AssertionError):
        assert_ablation_declaration(replace(coalesce, equivalence="not_asserted"))
    with pytest.raises(ValueError):
        ablation_spec("not_an_ablation")

    # The recorded observation is marked as such, whatever it measures.
    observation = ablation_wall_observation(
        cell=CELL_FRESH_HEADLESS, real_ms=10.0, ablated_ms=4.0
    )
    assert observation["gated"] is False
    assert observation["observed_only"] is True
    assert observation["interpretation"] == "compound_upper_bound"
    assert observation["combinations"] is None
    assert observation["observations"]["upper_bound_ms"] == 6.0


def _connect(host: str) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.connect((host, 80))
    finally:
        probe.close()


def _install_guard(tmp_path: Path) -> EgressGuard:
    """Fail-closed egress on the ablation paths too (plan §1)."""
    guard = EgressGuard(tmp_path / "egress")
    guard.install()
    guard.assert_closed()
    assert Path(os.environ[CONFIG_HOME_ENV]) == guard.config_home
    assert not [name for name in PROVIDER_CREDENTIAL_NAMES if name in os.environ]
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    with pytest.raises(OSError, match="blocked"):
        _connect(_EgressProbe)
    return guard


def _real_methods(delivered: list[dict[str, object]]) -> list[str]:
    return [
        str(message["method"])
        for message in delivered
        if isinstance(message.get("method"), str)
    ]


async def _settle_growth(
    client: object, delivered_before: int, *, timeout: float = _SETTLE_TIMEOUT_S
) -> None:
    """Wait for the restored client end to receive more than it had."""
    deadline = perf_counter() + timeout
    while len(client.delivered) <= delivered_before:  # type: ignore[attr-defined]
        if perf_counter() >= deadline:
            raise AssertionError("the restored client end received nothing")
        await asyncio.sleep(0)


async def _exercise_transport_object_ablation() -> None:
    """Replace the transport object and price its compound upper bound."""
    real = await build_headless_scenario(CELL_FRESH_HEADLESS, scale=SHORTENED_SCALE)
    ablated = await build_headless_scenario(CELL_FRESH_HEADLESS, scale=SHORTENED_SCALE)
    try:
        # The reference run: the real transport object, counted.
        real_counter = SerializationCounter()
        with real_counter.install():
            real_turns = [
                await real.run_root_turn(turn) for turn in range(_CONTRACT_TURNS)
            ]
        await real.settle_transport()
        real_methods = _real_methods(real.delivered_notifications)
        assert len(real_methods) == len(real.sink.serialized) > 0
        assert len(real_methods) >= 2 * _CONTRACT_TURNS
        # One transport-level encoding per notification — the second dump.
        assert real_counter.dumps_calls == len(real_methods)

        # The paired comparison only means anything if both scenarios drive the
        # same committed geometry.
        assert real.spec.digest() == ablated.spec.digest()

        original_sink_transport = ablated.sink._transport
        original_client_end = ablated._client_end

        ablated_counter = SerializationCounter()
        async with ablate_transport_object(ablated) as ablation:
            # Proof of activation: the degenerate object is the live one.
            assert ablated.sink._transport is ablation.stub
            assert ablated._client_end is ablation.client
            with ablated_counter.install():
                ablated_turns = [
                    await ablated.run_root_turn(turn) for turn in range(_CONTRACT_TURNS)
                ]
            await ablation.settle()

            assert ablation.stub_sends > 0
            assert ablation.stub_sends == len(ablation.delivered_methods())
            # Delivery neutrality: the ablation removes cost, not events.
            assert ablation.delivered_methods() == real_methods
            # Negative control: the neutrality comparison is load-bearing.
            assert ablation.delivered_methods()[:-1] != real_methods
            # No transport-level encoding happened at all on the ablated path.
            assert ablated_counter.dumps_calls == 0

            real_ms = sum(turn.wall_ms for turn in real_turns)
            ablated_ms = sum(turn.wall_ms for turn in ablated_turns)
            assert real_ms > 0.0
            assert ablated_ms > 0.0
            ablation.observations.update({
                "cell": CELL_FRESH_HEADLESS,
                "workload_hash": ablated.spec.digest(),
                "turns": len(ablated_turns),
                "real_wall_ms": round(real_ms, 3),
                "ablated_wall_ms": round(ablated_ms, 3),
                "upper_bound_ms": round(real_ms - ablated_ms, 3),
                "notification_count": len(real_methods),
                "gated": False,
            })

        # Restoration: the exact prior objects are back in the path.
        assert ablation.restored is True
        assert ablated.sink._transport is original_sink_transport
        assert ablated._client_end is original_client_end

        record = ablation.record()
        assert record["ablation"] == ABLATION_TRANSPORT_OBJECT
        assert record["interpretation"] == "compound_upper_bound"
        assert record["equivalence"] == "not_asserted"
        assert record["observed_only"] is True
        assert record["production"] is False
        assert record["combinations"] is None
        assert record["activated"] is True
        assert record["restored"] is True
        activation = record["activation"]
        assert activation["stub_sends"] == len(real_methods)
        assert activation["delivered_notifications"] == len(real_methods)
        observations = record["observations"]
        assert observations["workload_hash"] == WORKLOAD_HASHES[CELL_FRESH_HEADLESS]
        assert observations["turns"] == _CONTRACT_TURNS
        assert observations["notification_count"] == len(real_methods)
        assert observations["gated"] is False
        assert isinstance(observations["upper_bound_ms"], float)

        # The restored path is functional, not merely reference-equal: the real
        # transport encodes again and its notifications reach the client end.
        restored_counter = SerializationCounter()
        delivered_before = len(ablated.delivered_notifications)
        with restored_counter.install():
            restored = await ablated.run_root_turn(_CONTRACT_TURNS)
        assert restored.status == "completed"
        assert restored_counter.dumps_calls > 0
        await _settle_growth(original_client_end, delivered_before)
    finally:
        await real.close()
        await ablated.close()

    real.assert_cleanup()
    ablated.assert_cleanup()


async def _exercise_integrity_stub_ablation() -> None:
    """Install disposable history and prove the committed fixture is untouched."""
    scenario = await build_headless_scenario(CELL_FRESH_HEADLESS, scale=SHORTENED_SCALE)
    try:
        before = list(scenario.root_loop.messages)
        with ablate_integrity_stub(scenario) as ablation:
            installed = list(scenario.root_loop.messages)
            assert ablation.installed is True
            assert len(installed) == ablation.stub_messages == INTEGRITY_STUB_MESSAGES
            assert [id(message) for message in installed] != [
                id(message) for message in before
            ]
            # Disposable, not the committed fixture: no fixture lineage, so no
            # eligible baseline can ever be measured on this data.
            assert ablation.stub_digest != FIXTURE_HASH
            assert len(installed) != FIXTURE_MESSAGE_COUNT
            # The data stays admissible (the ablation removes the fixture
            # requirement, not the transcript's validity), so the cell runs.
            assert ablation.validation == {"valid": True, "error": None}

            turn = await scenario.run_root_turn(0)
            assert turn.status == "completed"
            assert turn.history_messages == ablation.stub_messages + 2
            ablation.observations.update({
                "cell": CELL_FRESH_HEADLESS,
                "workload_hash": scenario.spec.digest(),
                "stub_messages": ablation.stub_messages,
                "turn_history_messages": turn.history_messages,
                "committed_fixture_hash": FIXTURE_HASH,
                "gated": False,
            })

        # Restoration: the scenario's own transcript is exactly what it was.
        restored = list(scenario.root_loop.messages)
        assert [id(message) for message in restored] == [
            id(message) for message in before
        ]

        record = ablation.record()
        assert record["ablation"] == ABLATION_INTEGRITY_STUB
        assert record["interpretation"] == "disposable_data"
        assert record["equivalence"] == "not_asserted"
        assert record["observed_only"] is True
        assert record["production"] is False
        assert record["combinations"] is None
        assert record["activated"] is True
        assert record["restored"] is True
        activation = record["activation"]
        assert activation["stub_messages"] == INTEGRITY_STUB_MESSAGES
        assert activation["stub_digest"] != FIXTURE_HASH
        assert activation["data_disposable"] is True
        assert activation["fixture_lineage"] is False
        assert activation["integrity_boundary_skipped"] == 1
        assert activation["save_boundary_verdict"] == {"valid": True, "error": None}
        assert record["observations"]["gated"] is False
    finally:
        await scenario.close()
    scenario.assert_cleanup()

    # The committed fixture itself never moved, and it is still integrity-valid —
    # which is what makes the stub's disposal a real ablation and not corruption.
    probe = fixture_immutability_probe()
    assert probe["fixture_digest"] == probe["fixture_hash"] == FIXTURE_HASH
    assert probe["integrity"] == {"valid": True, "error": None}
    # The stub is admissible but carries no lineage, so it can never stand in
    # for the committed fixture a gated run must measure.
    stub = build_integrity_stub()
    assert fixture_digest(stub) != FIXTURE_HASH
    assert integrity_verdict(stub) == {"valid": True, "error": None}


async def _exercise_ablations() -> None:
    await _exercise_transport_object_ablation()
    await _exercise_integrity_stub_ablation()


@pytest.mark.timeout(_ABLATION_TIMEOUT_S)
def test_ablation_transport_integrity_contract(tmp_path: Path) -> None:
    """The two ablations activate, stay observation-only and restore state."""
    started = perf_counter()
    _assert_ablation_declarations()
    guard = _install_guard(tmp_path)
    try:
        asyncio.run(_exercise_ablations())
        guard.assert_closed()
    finally:
        guard.restore()
    # After restore the process is the caller's again: no hooks, no forced home.
    assert not guard._installed
    assert socket.socket.connect is guard._original_connect
    assert os.environ.get(CONFIG_HOME_ENV) != str(guard.config_home)
    elapsed = perf_counter() - started
    assert elapsed <= _CONTRACT_WINDOW_S, (
        f"ablation contract took {elapsed:.3f}s, window {_CONTRACT_WINDOW_S}s"
    )


# --------------------------------------------------------------------------
# WP5b: markdown_no_render, gc_disabled, source_coalesce
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _AppRun:
    """One whole-app run: per-turn delivered text, rendered markdown, flushes."""

    texts: list[str]
    rendered: list[str]
    flushes: int


def _expected_text(turn: int) -> str:
    """The committed response text of one workload turn."""
    return "".join(response_chunks(turn))


async def _mount_turn(
    pilot: Pilot[None], history: Vertical, turn: int
) -> AssistantMessage:
    """Mount one production user/assistant pair and settle the app."""
    await history.mount(UserMessage(root_user_content(turn)))
    message = AssistantMessage("")
    await history.mount(message)
    # Steer this widget's own frame pacing down to the production floor, as the
    # WP4 contract run does; the flush boundary and the render path stay
    # production's.
    message._frame_delay = PILOT_FRAME_DELAY_S
    await pilot.pause()
    return message


async def _settle_rendered(
    pilot: Pilot[None],
    message: AssistantMessage,
    expected: str,
    *,
    deadline_s: float = _SETTLE_DEADLINE_S,
) -> None:
    """Drive the app until the widget has rendered exactly ``expected``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    while rendered_markdown(message) != expected:
        if loop.time() >= deadline:
            raise AssertionError(
                f"render never settled: {len(rendered_markdown(message))} chars"
                f" of {len(expected)}"
            )
        await pilot.pause(PILOT_SETTLE_POLL_S)


async def _reference_app_run(turns: int) -> _AppRun:
    """Drive ``turns`` through production's per-delta streaming path."""
    app = PilotScenarioApp()
    texts: list[str] = []
    rendered: list[str] = []
    flushes = 0
    with pilot_render_observer(warmup_flushes=0) as observer:
        async with app.run_test() as pilot:
            history = app.query_one(f"#{PILOT_HISTORY_CONTAINER_ID}", Vertical)
            for turn in range(turns):
                message = await _mount_turn(pilot, history, turn)
                expected = _expected_text(turn)
                flushes_before = observer.log.non_empty_flushes
                for chunk in response_chunks(turn):
                    await message.append_content(chunk)
                    await pilot.pause(PILOT_SETTLE_POLL_S)
                await message.stop_stream()
                await _settle_rendered(pilot, message, expected)
                flushes += observer.log.non_empty_flushes - flushes_before
                texts.append(message.get_content())
                rendered.append(rendered_markdown(message))
    return _AppRun(texts=texts, rendered=rendered, flushes=flushes)


async def _markdown_no_render_ablation(reference: _AppRun) -> None:
    """Stream one turn with the renderer removed and prove the text survives."""
    expected = _expected_text(0)
    app = PilotScenarioApp()
    async with app.run_test() as pilot:
        history = app.query_one(f"#{PILOT_HISTORY_CONTAINER_ID}", Vertical)
        message = await _mount_turn(pilot, history, 0)
        original = StreamingMessageBase._write_chunk
        with ablate_markdown_no_render(
            message, cell=CELL_FRESH_HEADLESS, adapter=PILOT_ADAPTER
        ) as ablation:
            assert StreamingMessageBase._write_chunk is not original
            # The widget is live and has a markdown surface, and that surface
            # stays empty for the whole ablated turn.
            assert rendered_markdown(message) == ""
            for chunk in response_chunks(0):
                await message.append_content(chunk)
                await pilot.pause(PILOT_SETTLE_POLL_S)
            await message.stop_stream()
            await pilot.pause()
            assert message.get_content() == expected
            assert ablation.rendered_chars == 0
        # Restoration: production's write path is back, and functional — the
        # ablated stretch left the widget unrendered, so the frame written now
        # is exactly what it renders.
        assert StreamingMessageBase._write_chunk is original
        assert ablation.restored is True
        await message.append_content(_RESTORED_SUFFIX)
        await message.stop_stream()
        await _settle_rendered(pilot, message, _RESTORED_SUFFIX)
        assert message.get_content() == expected + _RESTORED_SUFFIX

    record = ablation.record()
    assert record["ablation"] == ABLATION_MARKDOWN_NO_RENDER
    assert record["interpretation"] == "render_component"
    assert record["equivalence"] == "not_asserted"
    assert record["observed_only"] is True
    assert record["production"] is False
    assert record["combinations"] is None
    assert record["activated"] is True
    assert record["restored"] is True
    activation = record["activation"]
    assert activation["stream_writes"] > 0
    assert activation["stream_chars"] == len(expected)
    assert activation["markdown_rendered_chars"] == 0
    assert ablation.stream.text == expected
    # The contrast that makes the emptiness meaningful: the same committed turn
    # on production's path DID render its markdown, and its flush count is not
    # zero either.
    assert reference.rendered[0] == expected
    assert reference.flushes > 0


async def _source_coalesce_ablation(reference: _AppRun, turns: int) -> None:
    """Coalesce a run's deltas and assert final-text/ordering equivalence."""
    app = PilotScenarioApp()
    texts: list[str] = []
    rendered: list[str] = []
    async with app.run_test() as pilot:
        history = app.query_one(f"#{PILOT_HISTORY_CONTAINER_ID}", Vertical)
        first = await _mount_turn(pilot, history, 0)
        original = StreamingMessageBase.append_content
        with ablate_source_coalesce(
            first, cell=CELL_FRESH_HEADLESS, adapter=PILOT_ADAPTER
        ) as ablation:
            assert StreamingMessageBase.append_content is not original
            for turn in range(turns):
                message = (
                    first if turn == 0 else await _mount_turn(pilot, history, turn)
                )
                # The bracket follows the class-wide write path; flush() targets
                # this turn's widget.
                ablation.widget = message
                expected = _expected_text(turn)
                for chunk in response_chunks(turn):
                    await message.append_content(chunk)
                await pilot.pause()
                await ablation.flush()
                await message.stop_stream()
                await _settle_rendered(pilot, message, expected)
                texts.append(message.get_content())
                rendered.append(rendered_markdown(message))
        assert StreamingMessageBase.append_content is original
        assert ablation.restored is True

    deltas = sum(len(response_chunks(turn)) for turn in range(turns))
    record = ablation.record()
    assert record["ablation"] == ABLATION_SOURCE_COALESCE
    assert record["interpretation"] == "delta_coalescing"
    assert record["equivalence"] == FINAL_TEXT_ORDERING
    assert record["observed_only"] is True
    assert record["production"] is False
    assert record["combinations"] is None
    assert record["activated"] is True
    assert record["restored"] is True
    activation = record["activation"]
    assert activation["deltas"] == deltas
    assert activation["writes"] == turns
    assert activation["coalesced"] is True
    assert ablation.writes < ablation.deltas
    # Final-text AND ordering equivalence against the per-delta reference.
    assert texts == reference.texts
    assert rendered == reference.rendered
    assert ablation.turns == reference.texts
    # Negative controls: the comparison is load-bearing — the reference turns
    # differ from each other, so a reordered or truncated run would not match.
    assert reference.texts[0] != reference.texts[1]
    assert texts[::-1] != reference.texts
    assert _expected_text(0)[:16] != _expected_text(1)[:16]


def _allocation_workload(count: int = _GC_ALLOCATIONS) -> int:
    """Allocate and drop containers — past gen-0's threshold when enabled."""
    total = 0
    for index in range(count):
        total += len({"index": index, "text": f"ablation-{index:04d}"})
    return total


def _gc_disabled_ablation() -> None:
    """Disable the collector, allocate past gen-0's threshold, restore it."""
    assert gc.isenabled() is True
    # Positive control: the counter does see collections when they happen.
    positive = CollectionCounter()
    with positive.install():
        gc.collect()
    assert positive.collections >= 1

    with ablate_gc_disabled(
        cell=CELL_FRESH_HEADLESS, adapter=PILOT_ADAPTER
    ) as ablation:
        assert gc.isenabled() is False
        assert ablation.activated is True
        allocations = _allocation_workload()
        assert allocations == 2 * _GC_ALLOCATIONS
        assert ablation.counter.collections == 0
        ablation.observations.update({
            "cell": CELL_FRESH_HEADLESS,
            "allocations": allocations,
            "gen0_threshold": gc.get_threshold()[0],
            "gated": False,
        })
    assert gc.isenabled() is True
    assert ablation.restored is True

    record = ablation.record()
    assert record["ablation"] == ABLATION_GC_DISABLED
    assert record["interpretation"] == "collector_disabled"
    assert record["equivalence"] == "not_asserted"
    assert record["observed_only"] is True
    assert record["production"] is False
    assert record["combinations"] is None
    assert record["activated"] is True
    assert record["restored"] is True
    activation = record["activation"]
    assert activation["collector_was_enabled"] is True
    assert activation["collector_disabled"] is True
    assert activation["collections_during"] == 0
    assert record["observations"]["allocations"] == 2 * _GC_ALLOCATIONS
    assert record["observations"]["gated"] is False


async def _exercise_render_gc_coalesce() -> None:
    reference = await _reference_app_run(_RENDER_TURNS)
    await _markdown_no_render_ablation(reference)
    await _source_coalesce_ablation(reference, _RENDER_TURNS)
    _gc_disabled_ablation()


@pytest.mark.timeout(_ABLATION_TIMEOUT_S)
def test_ablation_render_gc_coalesce_contract(tmp_path: Path) -> None:
    """The render, GC and coalescing ablations activate, restore and agree."""
    started = perf_counter()
    _assert_ablation_declarations()
    guard = _install_guard(tmp_path)
    try:
        asyncio.run(_exercise_render_gc_coalesce())
        guard.assert_closed()
    finally:
        guard.restore()
    # After restore the process is the caller's again: no hooks, no forced home.
    assert not guard._installed
    assert socket.socket.connect is guard._original_connect
    assert os.environ.get(CONFIG_HOME_ENV) != str(guard.config_home)
    elapsed = perf_counter() - started
    assert elapsed <= _CONTRACT_WINDOW_S, (
        f"render/GC/coalesce contract took {elapsed:.3f}s, window {_CONTRACT_WINDOW_S}s"
    )
