"""Synchronized completion-storm cell for the v0.7.5 stage-2 runtime gate (X8b).

The cell is E1 of the plan's §4 table and Appendix B: **8 child agents behind a
synchronized barrier, released together, measured until all finalization and
drain complete**. One storm is one cohort; the run's gated sample series is the
per-storm drain wall, and the run's floor is >= 100 measured storms after >= 10
warmup storms (Appendix B row E1: the unit is a storm, not a turn).

What the cell pins:

* **Fresh cleaned cohort per storm.** Every storm builds its own cohort of
  ``STORM_AGENTS`` children in one real app-server registry, and the previous
  cohort is drained (``SessionRuntimeRegistry.drain_children``) — which clears
  the children, their links, agent records, the result store and its
  collection/expiry markers — before the next storm is built. Cleaning is
  fixture work, never the metric: the run asserts an all-zero residual before
  each cohort instead of measuring the cleaning pass. No completion-history
  entry survives from one storm into the next, and no agent id is reused.
* **The measured window.** The storm trigger is the barrier release; the window
  closes when the last of the eight completions has been delivered AND the
  cohort's finalization tasks (``_pending_child_closes``) have settled. Cohort
  construction (child loops, seeded history, empty-session persistence, barrier
  arrival) is priced as fixture time and recorded per storm, never gated.
* **Real machinery.** The children are launched and awaited through
  :class:`~chartreux.app_server._sessions.SessionRuntimeRegistry`, on the real
  event loop, with the deterministic seeded history of the WP3a fixture and the
  committed child task texts of the WP3b scenario harness. The barrier backend
  is the harness's :class:`~tests.perf._runtime_scenario.GatedChildBackend`, so
  every child request starts before any child is released.
* **Chunks coverage (plan §3).** :meth:`StormRun.chunks_coverage` reports the
  storms per run — measured storms, warmup storms and the cohort count — and
  :meth:`StormRun.run_record` emits the E1 run record of
  ``chartreux.runtime_perf.v1`` with those fields populated.
* **Fail-closed egress (plan §1).** :func:`run_storm_campaign` installs and
  asserts the harness :class:`~tests.perf._runtime_scenario.EgressGuard` around
  the whole run, so a bare-script campaign is covered the same way the pytest
  fixtures cover the in-suite node.

Nothing here changes production behaviour: the cohort is driven from the test
side through public registry entry points, and the only instrumented pieces are
test-side observers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter, perf_counter_ns
from typing import Any

from chartreux.core.subagents import LaunchConfig, TaskArgs, TaskResult, WaitStatus
from chartreux.core.tools.base import InvokeContext
from tests.perf._runtime_fixture import FIXTURE_HASH, FIXTURE_TURN_WIDTH
from tests.perf._runtime_metrics import current_perf_paths
from tests.perf._runtime_report import (
    COUNT_STORMS,
    MODE_MEASURE,
    build_run_record,
    row_spec,
    validate_run_record,
)
from tests.perf._runtime_scenario import (
    CELL_LOADED_HEADLESS,
    EgressGuard,
    HeadlessScenario,
    ScenarioScale,
    _patched_create_backend,
    build_headless_scenario,
    child_task_text,
)

#: The storm cell's schema id (a run wrapper, never a rival of the row identity).
STORM_SCHEMA = "chartreux.runtime_storm.v1"

#: E1's campaign id; the row spec owns the cell/metric/unit/producer identity.
STORM_CAMPAIGN_ID = "E1"
STORM_CELL = row_spec(STORM_CAMPAIGN_ID).cell

#: The synchronized cohort dimension of the cell (plan §4 E1: 8 child agents).
STORM_AGENTS = 8

#: The floors of Appendix B row E1, restated here for local validation.
STORM_WARMUP_STORMS = 10
STORM_MEASURED_FLOOR = 100

#: A run targets the floor: 100 measured storms after the 10 warmup storms.
STORM_MEASURED_STORMS = STORM_MEASURED_FLOOR

#: One storm is one fresh cohort: the cohort count equals the storm count.
STORM_COHORTS_PER_STORM = 1

#: The cohort's committed history fixture: the first two committed turns.
STORM_COHORT_HISTORY_MESSAGES = FIXTURE_TURN_WIDTH * 2

#: The harness fixture cell: the real headless registry/transport/projector.
STORM_HARNESS_CELL = CELL_LOADED_HEADLESS

#: The barrier discipline: every child starts before any child is released.
STORM_SYNC_STARTED_BEFORE_RELEASE = "all_children_started_before_release"

#: The cohort-cleaning discipline recorded with every storm.
STORM_CLEANING_DRAIN = "registry_drain_between_cohorts"

_STORM_SECONDS = 60.0
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_SAMPLE_PRECISION = 4


class StormError(AssertionError):
    """A storm violated its cohort, synchronization or cleaning contract."""


@dataclass(frozen=True, slots=True)
class CohortSpec:
    """One storm cohort's committed geometry and its cleaning discipline."""

    agents: int
    history_messages: int
    synchronization: str
    cleaning: str
    fixture_hash: str

    def record(self) -> dict[str, Any]:
        return {
            "agents": self.agents,
            "history_messages": self.history_messages,
            "synchronization": self.synchronization,
            "cleaning": self.cleaning,
            "fixture_hash": self.fixture_hash,
            "cohorts_per_storm": STORM_COHORTS_PER_STORM,
        }

    def digest(self) -> str:
        """Return the sha256 digest of the committed cohort definition."""
        encoded = json.dumps(
            self.record(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(f"{STORM_SCHEMA}|{STORM_CELL}|".encode())
        digest.update(encoded)
        return digest.hexdigest()


def cohort_spec() -> CohortSpec:
    """Return the committed cohort definition of the storm cell."""
    return CohortSpec(
        agents=STORM_AGENTS,
        history_messages=STORM_COHORT_HISTORY_MESSAGES,
        synchronization=STORM_SYNC_STARTED_BEFORE_RELEASE,
        cleaning=STORM_CLEANING_DRAIN,
        fixture_hash=FIXTURE_HASH,
    )


def storm_scale() -> ScenarioScale:
    """Return the harness scale one storm cohort is built at.

    The cohort is the full 8-child dimension of the cell: only the storm count
    is shortened in-suite (plan §5), never the cohort.
    """
    return ScenarioScale(
        name="storm",
        root_turns=0,
        history_messages=STORM_COHORT_HISTORY_MESSAGES,
        child_agents=STORM_AGENTS,
        burst_agents=STORM_AGENTS,
        burst_interval_turns=0,
        mode="contract",
    )


async def build_storm_harness(session_root: Path | None = None) -> HeadlessScenario:
    """Build the real registry/transport/projector fixture one run storms on."""
    return await build_headless_scenario(
        STORM_HARNESS_CELL, scale=storm_scale(), session_root=session_root
    )


# --------------------------------------------------------------------------
# Cohort state: freshness, cleaning and the residual record
# --------------------------------------------------------------------------


#: The registry state a cleaned cohort leaves behind, one zero count per field.
RESIDUAL_FIELDS: tuple[str, ...] = (
    "children",
    "child_links",
    "agent_records",
    "run_results",
    "collected_results",
    "expired_results",
    "latest_run_ids",
    "pending_child_closes",
    "stored_children",
    "readable_children",
)


def empty_residual() -> dict[str, int]:
    """Return the all-zero residual a cleaned cohort must leave."""
    return dict.fromkeys(RESIDUAL_FIELDS, 0)


def cohort_residual(scenario: HeadlessScenario) -> dict[str, int]:
    """Return the registry's in-memory cohort and completion-history counts.

    Every count must be zero before a cohort is built: a non-zero entry is
    residual completion history from an earlier storm, which the cell forbids.
    """
    registry = scenario.registry
    counts = {
        "children": len(registry._children),
        "child_links": len(registry._child_links),
        "agent_records": len(registry._agent_records),
        "run_results": len(registry._result_store),
        "collected_results": len(registry._collected_results),
        "expired_results": len(registry._expired_results),
        "latest_run_ids": len(registry._latest_run_ids),
        "pending_child_closes": len(registry._pending_child_closes),
        "stored_children": len(registry._stored_children),
        "readable_children": len(registry._readable_children),
    }
    if set(counts) != set(RESIDUAL_FIELDS):
        raise StormError("the cohort residual record drifted from RESIDUAL_FIELDS")
    return counts


async def clean_cohort(scenario: HeadlessScenario) -> dict[str, int]:
    """Drain the previous cohort and return the residual it left behind.

    The drain is the registry's own finalization path: it joins the tracked
    teardowns, closes the remaining child runtimes and clears the completion
    history (records, result store and its collection markers). The returned
    residual must be all-zero; the caller fails closed otherwise.
    """
    await scenario._await_root_idle()
    await scenario.registry.drain_children()
    await scenario._settle_child_closes()
    if scenario.projector is not None:
        raise StormError("a projector outlived the cleaned cohort")
    return cohort_residual(scenario)


def assert_clean(residual: Mapping[str, int], *, where: str) -> None:
    """Fail closed unless every residual count is zero."""
    if set(residual) != set(RESIDUAL_FIELDS):
        raise StormError(f"incomplete residual record {where}: {sorted(residual)}")
    leftover = {key: value for key, value in residual.items() if value}
    if leftover:
        raise StormError(f"residual completion history {where}: {leftover}")


# --------------------------------------------------------------------------
# One storm
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StormRecord:
    """One storm: its cohort, its synchronization and its drain sample."""

    index: int
    cohort_size: int
    completed: int
    drain_ms: float
    construct_ms: float
    warmup: bool
    agent_ids: tuple[str, ...]
    residual_before: dict[str, int]

    def record(self) -> dict[str, Any]:
        return {
            "storm": self.index,
            "cohort_size": self.cohort_size,
            "completed": self.completed,
            "drain_ms": round(self.drain_ms, _SAMPLE_PRECISION),
            "construct_ms": round(self.construct_ms, _SAMPLE_PRECISION),
            "warmup": self.warmup,
            "agent_ids": list(self.agent_ids),
            "residual_before": dict(self.residual_before),
        }


async def _launch_cohort(scenario: HeadlessScenario, storm: int) -> list[TaskResult]:
    """Launch one storm's children; they block on the barrier backend next."""
    launches: list[TaskResult] = []
    for position in range(STORM_AGENTS):
        args = TaskArgs(
            task=child_task_text(position),
            background=True,
            config=LaunchConfig(model=scenario.model_names[position]),
        )
        context = InvokeContext(
            tool_call_id=f"perf-storm-{storm:04d}-{position:02d}",
            session_id=scenario.session_id,
        )
        result: TaskResult | None = None
        async for event in scenario.registry.run(args, context):
            if isinstance(event, TaskResult):
                result = event
        if result is None:
            raise StormError(f"storm {storm} launched no task result at {position}")
        launches.append(result)
    return launches


async def run_storm(
    scenario: HeadlessScenario, storm: int, *, warmup: bool
) -> StormRecord:
    """Construct one fresh cohort, release the barrier, time the drain.

    The measured window opens at the barrier release and closes when the last
    completion is delivered and the cohort's finalization tasks have settled.
    Cohort construction and the preceding cleaning pass stay outside it.
    """
    residual_before = cohort_residual(scenario)
    assert_clean(residual_before, where=f"before storm {storm}")
    backend = scenario._burst_backend
    backend.prepare()
    completed_before = backend.completed
    construct_started = perf_counter()
    with _patched_create_backend(backend):
        launches = await asyncio.wait_for(
            _launch_cohort(scenario, storm), timeout=_STORM_SECONDS
        )
        # The barrier: every child request has started before any release.
        await asyncio.wait_for(
            asyncio.gather(*(gate.wait() for gate in backend.started)),
            timeout=_STORM_SECONDS,
        )
        construct_ms = (perf_counter() - construct_started) * 1000
        awaited = [
            (result.agent_id, result.run_id)
            for result in launches
            if result.agent_id is not None and result.run_id is not None
        ]
        if len(awaited) != STORM_AGENTS:
            raise StormError(
                f"storm {storm} launched {len(awaited)} of {STORM_AGENTS} children"
            )
        agent_ids = tuple(agent_id for agent_id, _run_id in awaited)
        trigger_ns = perf_counter_ns()
        for release in backend.releases:
            release.set()
        completions = await asyncio.wait_for(
            asyncio.gather(
                *(
                    scenario.registry.wait_for_agent(
                        agent_id, run_id, timeout=_STORM_SECONDS
                    )
                    for agent_id, run_id in awaited
                )
            ),
            timeout=_STORM_SECONDS,
        )
        await scenario._settle_child_closes()
        drain_ms = (perf_counter_ns() - trigger_ns) / _NANOSECONDS_PER_MILLISECOND
    completed = sum(
        1
        for completion in completions
        if completion.status is WaitStatus.COMPLETED
        and completion.result is not None
        and completion.result.completed
    )
    if completed != STORM_AGENTS:
        raise StormError(
            f"storm {storm} drained {completed} of {STORM_AGENTS} completions"
        )
    if backend.completed - completed_before != STORM_AGENTS:
        raise StormError(
            f"storm {storm} backend served {backend.completed - completed_before}"
            f" child requests"
        )
    if len(backend.context_chars) != STORM_AGENTS:
        raise StormError(
            f"storm {storm} recorded {len(backend.context_chars)} child contexts"
        )
    return StormRecord(
        index=storm,
        cohort_size=STORM_AGENTS,
        completed=completed,
        drain_ms=drain_ms,
        construct_ms=construct_ms,
        warmup=warmup,
        agent_ids=agent_ids,
        residual_before=residual_before,
    )


# --------------------------------------------------------------------------
# The run: storms, their coverage and the E1 run record
# --------------------------------------------------------------------------


@dataclass(slots=True)
class StormRun:
    """The storms of one run, their coverage and the E1 run record."""

    warmup_storms: int = STORM_WARMUP_STORMS
    cohort: CohortSpec = field(default_factory=cohort_spec)
    records: list[StormRecord] = field(default_factory=list)

    def add(self, record: StormRecord) -> None:
        """Record one storm, rejecting a cohort that is not fresh."""
        if record.index != len(self.records):
            raise StormError(
                f"storm {record.index} arrived out of order at {len(self.records)}"
            )
        if record.cohort_size != self.cohort.agents:
            raise StormError(
                f"storm {record.index} ran a {record.cohort_size}-agent cohort"
            )
        seen = {agent_id for earlier in self.records for agent_id in earlier.agent_ids}
        reused = sorted(seen.intersection(record.agent_ids))
        if reused:
            raise StormError(f"storm {record.index} reused cohort agents {reused}")
        assert_clean(record.residual_before, where=f"before storm {record.index}")
        self.records.append(record)

    @property
    def warmups(self) -> list[StormRecord]:
        return [record for record in self.records if record.warmup]

    @property
    def measured(self) -> list[StormRecord]:
        return [record for record in self.records if not record.warmup]

    @property
    def samples(self) -> list[float]:
        """Return the gated series: one drain sample per measured storm."""
        return [record.drain_ms for record in self.measured]

    def chunks_coverage(self) -> dict[str, Any]:
        """Return the run's storms-per-run coverage (plan §3, row E1)."""
        missing: list[str] = []
        if not self.records:
            missing.append("storms")
        return {
            "unit": "storms",
            "storms_per_run": len(self.records),
            "measured_storms": len(self.measured),
            "warmup_storms": len(self.warmups),
            "cohorts_per_run": len(self.records) * STORM_COHORTS_PER_STORM,
            "agents_per_cohort": self.cohort.agents,
            "drain_ms_per_storm": [
                round(record.drain_ms, _SAMPLE_PRECISION) for record in self.records
            ],
            "cohort_digest": self.cohort.digest(),
            "complete": not missing,
            "missing": missing,
        }

    def storm_section(self) -> dict[str, Any]:
        """Return the additive run-record section with the per-storm rows."""
        return {
            "cell": STORM_CELL,
            "cohort": self.cohort.record(),
            "cohort_digest": self.cohort.digest(),
            "storms": [record.record() for record in self.records],
            "measured_storms": len(self.measured),
            "warmup_storms": len(self.warmups),
            "cohorts": len(self.records) * STORM_COHORTS_PER_STORM,
            "fresh_cohort_per_storm": True,
        }

    def run_record(
        self,
        *,
        run_id: str,
        run_start_utc: str,
        closure_status: str | None = None,
        closure_note: str | None = None,
        mode: str = MODE_MEASURE,
        fixture_hash: str | None = None,
        machine: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the validated E1 run record of this run's storms."""
        record = build_run_record(
            STORM_CAMPAIGN_ID,
            run_id=run_id,
            run_start_utc=run_start_utc,
            mode=mode,
            measured_ops=len(self.measured),
            warmup_ops=len(self.warmups),
            samples=self.samples,
            unit_counts={
                COUNT_STORMS: len(self.measured),
                "storms_per_run": len(self.records),
                "cohorts": len(self.records) * STORM_COHORTS_PER_STORM,
            },
            closure_status=closure_status,
            closure_note=closure_note,
            fixture_hash=fixture_hash,
            machine=machine,
        )
        record["storm"] = self.storm_section()
        validate_run_record(record)
        return record


async def run_storms(
    scenario: HeadlessScenario,
    *,
    warmup_storms: int = STORM_WARMUP_STORMS,
    measured_storms: int = STORM_MEASURED_STORMS,
) -> StormRun:
    """Run one storm sequence: the warmups first, then the measured storms."""
    if warmup_storms < 0 or measured_storms < 0:
        raise ValueError("storm counts must be non-negative")
    run = StormRun(warmup_storms=warmup_storms)
    total = warmup_storms + measured_storms
    for storm in range(total):
        assert_clean(await clean_cohort(scenario), where=f"after storm {storm - 1}")
        record = await run_storm(scenario, storm, warmup=storm < warmup_storms)
        run.add(record)
    assert_clean(await clean_cohort(scenario), where=f"after storm {total - 1}")
    return run


async def run_storm_campaign(
    *,
    measured_storms: int = STORM_MEASURED_STORMS,
    warmup_storms: int = STORM_WARMUP_STORMS,
    run_id: str = "storm-run",
    run_start_utc: str = "",
    session_root: Path | None = None,
) -> dict[str, Any]:
    """Run one externally invoked storm campaign run and return its record.

    The egress guard is installed for the whole run (a bare-script campaign is
    not covered by the pytest fixtures) and the run record is written to the
    boundary-checked ``CHARTREUX_PERF_REPORT`` path when the run environment
    declares one.
    """
    paths = current_perf_paths()
    guard_root = None if paths.run_dir is None else paths.run_dir / "storm-egress"
    guard = EgressGuard(guard_root).install()
    try:
        guard.assert_closed()
        scenario = await build_storm_harness(session_root)
        try:
            run = await run_storms(
                scenario, warmup_storms=warmup_storms, measured_storms=measured_storms
            )
            record = run.run_record(
                run_id=run_id,
                run_start_utc=run_start_utc,
                closure_note="campaign open: this run is one attempt of the E1 campaign",
                fixture_hash=FIXTURE_HASH,
            )
        finally:
            await scenario.close()
        scenario.assert_cleanup()
        guard.assert_closed()
    finally:
        guard.restore()
    if paths.report is not None:
        paths.report.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return record


def storm_samples(records: Sequence[StormRecord]) -> list[float]:
    """Return the drain samples of the measured storms in ``records``."""
    return [record.drain_ms for record in records if not record.warmup]


def rebuild_warmups(run: StormRun, *, warmup_storms: int) -> StormRun:
    """Re-mark a run's first ``warmup_storms`` storms as its warmups.

    A run's warmup boundary is a property of the run record, not of the storm
    itself, so re-marking keeps the samples and the order untouched.
    """
    if not 0 <= warmup_storms <= len(run.records):
        raise ValueError(
            f"{warmup_storms} warmups is outside a {len(run.records)}-storm run"
        )
    rebuilt = StormRun(warmup_storms=warmup_storms, cohort=run.cohort)
    for position, record in enumerate(run.records):
        rebuilt.add(replace(record, warmup=position < warmup_storms))
    return rebuilt


__all__ = [
    "RESIDUAL_FIELDS",
    "STORM_AGENTS",
    "STORM_CAMPAIGN_ID",
    "STORM_CELL",
    "STORM_CLEANING_DRAIN",
    "STORM_COHORTS_PER_STORM",
    "STORM_COHORT_HISTORY_MESSAGES",
    "STORM_HARNESS_CELL",
    "STORM_MEASURED_FLOOR",
    "STORM_MEASURED_STORMS",
    "STORM_SCHEMA",
    "STORM_SYNC_STARTED_BEFORE_RELEASE",
    "STORM_WARMUP_STORMS",
    "CohortSpec",
    "StormError",
    "StormRecord",
    "StormRun",
    "assert_clean",
    "build_storm_harness",
    "clean_cohort",
    "cohort_residual",
    "cohort_spec",
    "empty_residual",
    "rebuild_warmups",
    "run_storm",
    "run_storm_campaign",
    "run_storms",
    "storm_samples",
    "storm_scale",
]
