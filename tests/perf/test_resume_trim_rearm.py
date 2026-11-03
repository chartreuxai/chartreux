"""Warm-resume clear-and-reseal perf gate — T2b part 1 (resume harness + runner).

The resume path now runs the warm-resume clear-and-reseal transaction
(``SessionLogger.clear_and_reseal_warm_resume``) on every resume: it proves
the rebound list fully hydrated from the target disk under the save lock and
then CLEARS the retained archive descriptor (never re-pinned), so the next
compaction seal mints a fresh receipt at the new epoch. This file owns the
resume harness: a scripted loop whose second turn crosses the auto-compact
threshold, so a retained archive descriptor exists before the measured warm
resume (``AgentRuntimeFactory.resume_root`` onto the same session).

Measured quantity (design W/glm F2, effect-based per Amendment EE): the
resume-transaction duration — the clear-and-reseal window inside the warm
resume — measured with THIS harness. Every trial also proves the RESUME-SCOPE
effects (retained descriptor cleared, NO re-seal of the old authority during
the resume, transaction transitioned), so a broken clear-and-reseal cannot
measure as fast. The fresh receipt at the new epoch is minted by the NEXT
seal and is asserted OUTSIDE the measured window (Amendment LL), in
tests/core/compaction/test_archive_trim_activation.py.

The ceiling constant below is FROZEN by T2b part 2 (2026-11-02): the
provenance-frozen pre-change denominator (same harness — as a minimal
denominator variant measuring the identical whole-warm-resume window over the
identical workload — on the pinned pre-change commit f8fd5823, in a worktree)
plus the bounded transaction overhead, both recorded in the constant's
comment block. The smoke trial's numbers remain NOT a denominator — the
denominator is the pinned-worktree freeze recorded in the comment block.

Serial only: run with ``-n0`` on a quiet machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
import time
from typing import Any

import pytest

from chartreux.app_server import _runtime as runtime
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.session.session_logger import SessionLogger
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from tests.mock.utils import mock_llm_chunk
from tests.perf._metrics import machine_context, record
from tests.stubs.fake_backend import FakeBackend

#: Number of measured trials part 2 runs on a quiet machine.
TRIALS = 5

#: The harness's auto-compact threshold; turn 2 is driven past it so a
#: compaction seal retains an archive descriptor before the resume.
AUTO_COMPACT_THRESHOLD = 1_000

# ---------------------------------------------------------------------------
# FROZEN CEILING — landed by T2b part 2 on 2026-11-02 (quiet machine, no
# concurrent worker tests; the orchestrator held all other lanes).
#
# RESUME_TRIM_REARM_OVERHEAD_CEILING_S = 0.029387 s (29.387 ms) =
# frozen pre-change denominator median 28.484 ms + bounded overhead 0.903 ms.
#
# Measured quantity (both sides): the whole warm-resume window —
# resume_started -> resume_root return around
# ``AgentRuntimeFactory().resume_root(loop, loop.session_id)`` — the
# resume-scope quantity that exists on both sides of the change. The
# post-change side additionally records the clear-and-reseal transaction
# window (resume_transaction_s) per trial for attribution.
#
# Denominator provenance: the shipped harness as a minimal DENOMINATOR
# VARIANT (identical workload and window endpoints; post-change-only pieces
# removed: no clear_and_reseal_warm_resume monkeypatch — the API does not
# exist at the baseline — and no post-change-only effect assertions; one
# baseline-only setup delta: an explicit ``loop.compact()`` outside any turn
# because the baseline P4d trim is active-turn-gated, so an in-turn
# auto-compaction never retains a descriptor at f8fd5823, while at 216e6c6b
# the post-turn seal retains it after turn 2 — probe-verified), run in a
# worktree PINNED at pre-change commit
# f8fd58238aa820df43c9f0808b201fe385a5b52f ("Release v0.7.0"), with -n0 from
# the worktree root (cwd-first sys.path shadows the editable install;
# import probe: chartreux.__file__ = <worktree>/chartreux/__init__.py):
# freeze batch of 5 trials gave whole-resume windows 31.920/28.962/28.446/
# 28.484/28.233 ms (median 28.484 ms), a confirmation batch of 5 more trials
# gave 33.605/27.654/27.204/26.906/27.383 ms (median 27.383 ms, drift
# 1.101 ms); all 10 trials retained the descriptor across the baseline warm
# resume with 0 reseal calls during the resume. Post-change side: main tree
# at 216e6c6b0db35cf447533df8c694f28819ac1c73. Machine context (both sides):
# AMD Ryzen 9 5900X 12-Core Processor, 24 cores,
# Linux-7.3.0-9-generic-x86_64 glibc 2.44, Python 3.12.13; loadavg 1-min
# ~1.0-2.4 on 24 cores before/during/after all runs.
#
# Overhead bound: 3 x the maximum measured post-change in-window transaction
# cost. The only work the post-change side adds inside the measured window is
# the clear-and-reseal transaction (proof + staged repair + authority
# transition under one save-lock hold), measured on 216e6c6b via the smoke
# node over 5 single-trial runs (-n0): transaction windows
# 0.230/0.262/0.292/0.301/0.279 ms, max 0.301 ms; 3 x 0.301 ms = 0.903 ms.
# The transaction's workload is bounded by the sealed archived prefix, not by
# turn count; the 3x factor covers trial-level variance and scheduling jitter.
# (Single-trial fresh-process smoke whole-resume windows were 32.9-35.1 ms,
# but those are first-trial warm-up numbers — the baseline batches show the
# same first-trial inflation, 31.920/33.605 ms, with trials 2-5 at 26.9-29.0
# ms — so they are not the denominator.)
#
# A ceiling miss is investigated with per-trial attribution, never silently
# raised.
# ---------------------------------------------------------------------------
RESUME_TRIM_REARM_OVERHEAD_CEILING_S: float | None = 0.029387


@dataclass(slots=True)
class ResumeTrimRearmTrial:
    """One measured warm-resume trial plus its proven resume-scope effects."""

    trial: int
    resume_transaction_s: float
    whole_resume_s: float
    descriptor_retained_before: bool
    descriptor_cleared_after: bool
    reseal_calls_during_resume: int
    txn_outcome: str
    save_generation_moved: bool
    reset_generation_moved: bool


def _build_resume_harness(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[AgentLoop, FakeBackend, dict[str, Any]]:
    """Build a loop whose turn 2 produces a retained archive descriptor.

    Returns the loop, its scripted backend, and the instrumentation dict
    (clear-and-reseal window + outcome, seal calls observed while the resume
    runs).
    """
    config = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=AUTO_COMPACT_THRESHOLD),
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(root / "sessions"), session_prefix="perf-resume"
        ),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="first answer")],
        [mock_llm_chunk(content="<summary>perf compacted</summary>")],
        [mock_llm_chunk(content="second answer")],
    ])
    loop = build_test_agent_loop(config=config, backend=backend)
    instrumentation: dict[str, Any] = {
        "txn_window": None,
        "txn_outcome": None,
        "seal_calls_during_resume": 0,
        "in_resume": False,
    }

    # The warm-resume clear-and-reseal transaction window (the measured
    # quantity) plus its reported outcome (the effect assertion).
    real_txn = SessionLogger.clear_and_reseal_warm_resume

    async def _timed_txn(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = await real_txn(self, *args, **kwargs)
        finally:
            instrumentation["txn_window"] = (start, time.perf_counter())
        instrumentation["txn_outcome"] = result.outcome
        return result

    monkeypatch.setattr(SessionLogger, "clear_and_reseal_warm_resume", _timed_txn)

    # No re-seal of the old authority may happen during the resume.
    real_seal = SessionLogger.seal_archive

    async def _seal_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        if instrumentation["in_resume"]:
            instrumentation["seal_calls_during_resume"] += 1
        return await real_seal(self, *args, **kwargs)

    monkeypatch.setattr(SessionLogger, "seal_archive", _seal_spy)
    return loop, backend, instrumentation


async def _run_trial(
    root: Path, monkeypatch: pytest.MonkeyPatch, trial: int
) -> ResumeTrimRearmTrial:
    """Run one warm-resume trial and prove the resume-scope EFFECTS inside it.

    The effect assertions (retained descriptor cleared, no re-seal of the old
    authority, transaction transitioned) fire in every trial — smoke
    included — so the measured window can never be produced by a resume whose
    clear-and-reseal did not run.
    """
    loop, _backend, instrumentation = _build_resume_harness(root, monkeypatch)
    try:
        # Turn 1: one settled round (user + assistant).
        async for _event in loop.act("first question"):
            pass

        # Turn 2 crosses the auto-compact threshold: the compaction seal
        # retains an archive descriptor before the measured resume.
        loop.stats.context_tokens = AUTO_COMPACT_THRESHOLD + 1
        async for _event in loop.act("second question"):
            pass

        logger = loop.session_logger
        descriptor_before = logger.archive_descriptor
        assert descriptor_before is not None, (
            "the harness must retain an archive descriptor before the resume"
        )
        save_generation_before = logger.transcript_save_generation
        reset_generation_before = logger.archive_reset_generation

        # The measured warm resume (same session, in-place rebind).
        instrumentation["in_resume"] = True
        resume_started = time.perf_counter()
        try:
            await runtime.AgentRuntimeFactory().resume_root(loop, loop.session_id)
        finally:
            instrumentation["in_resume"] = False
        whole_resume_s = time.perf_counter() - resume_started

        # --- resume-scope EFFECT assertions (they fire in the smoke trial too)
        txn_start, txn_end = instrumentation["txn_window"]
        assert instrumentation["txn_outcome"] == "proved_and_transitioned", (
            "the warm resume must run the clear-and-reseal transaction to its "
            "transition outcome"
        )
        assert logger.archive_descriptor is None, (
            "clear-and-reseal: the retained descriptor is cleared on resume"
        )
        assert loop.messages.archive_descriptor is None
        assert instrumentation["seal_calls_during_resume"] == 0, (
            "the resume path must never re-seal the old authority"
        )
        assert logger.transcript_save_generation > save_generation_before, (
            "the transaction fence must move the save generation"
        )
        assert logger.archive_reset_generation > reset_generation_before, (
            "the transaction rebind must move the reset generation"
        )

        return ResumeTrimRearmTrial(
            trial=trial,
            resume_transaction_s=txn_end - txn_start,
            whole_resume_s=whole_resume_s,
            descriptor_retained_before=descriptor_before is not None,
            descriptor_cleared_after=logger.archive_descriptor is None,
            reseal_calls_during_resume=instrumentation["seal_calls_during_resume"],
            txn_outcome=instrumentation["txn_outcome"],
            save_generation_moved=(
                logger.transcript_save_generation > save_generation_before
            ),
            reset_generation_moved=(
                logger.archive_reset_generation > reset_generation_before
            ),
        )
    finally:
        await loop.aclose()


@pytest.mark.perf
@pytest.mark.timeout(300)
@pytest.mark.asyncio
async def test_resume_trim_rearm_smoke_single_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SMOKE CHECK ONLY: a single trial proving the harness executes and its
    resume-scope effect assertions fire (descriptor cleared, no old re-seal,
    transaction transitioned). The numbers printed here are NOT denominators —
    part 2 freezes the pre-change denominator and the ceiling value on a quiet
    machine; nothing is frozen or recorded as a denominator here.
    """
    trial = await _run_trial(tmp_path / "smoke", monkeypatch, trial=0)
    print(
        f"resume-trim-rearm SMOKE trial (NOT a denominator): "
        f"resume transaction {trial.resume_transaction_s * 1_000:.2f} ms, "
        f"whole warm resume {trial.whole_resume_s * 1_000:.2f} ms, "
        f"descriptor retained before {trial.descriptor_retained_before} -> "
        f"cleared after {trial.descriptor_cleared_after}, "
        f"reseal_calls_during_resume {trial.reseal_calls_during_resume}, "
        f"txn_outcome {trial.txn_outcome}, "
        f"save_generation_moved {trial.save_generation_moved}, "
        f"reset_generation_moved {trial.reset_generation_moved}"
    )
    print(f"resume-trim-rearm SMOKE machine context: {machine_context()}")
    record(
        "resume_trim_rearm_smoke",
        {
            "status": (
                "SMOKE ONLY — the harness executes and the resume-scope effect "
                "assertions fired (descriptor cleared, no re-seal of the old "
                "authority, transaction transitioned); this output is NOT a "
                "denominator and no value is frozen here (part 2 owns the "
                "freeze)"
            ),
            "trial": trial.trial,
            "resume_transaction_ms": round(trial.resume_transaction_s * 1_000, 3),
            "whole_resume_ms": round(trial.whole_resume_s * 1_000, 3),
            "descriptor_retained_before": trial.descriptor_retained_before,
            "descriptor_cleared_after": trial.descriptor_cleared_after,
            "reseal_calls_during_resume": trial.reseal_calls_during_resume,
            "txn_outcome": trial.txn_outcome,
            "save_generation_moved": trial.save_generation_moved,
            "reset_generation_moved": trial.reset_generation_moved,
            "ceiling_s": RESUME_TRIM_REARM_OVERHEAD_CEILING_S,
            "ceiling_enforced": RESUME_TRIM_REARM_OVERHEAD_CEILING_S is not None,
            "machine": machine_context(),
        },
    )
    assert trial.descriptor_retained_before is True
    assert trial.descriptor_cleared_after is True
    assert trial.reseal_calls_during_resume == 0
    assert trial.txn_outcome == "proved_and_transitioned"
    assert trial.save_generation_moved is True
    assert trial.reset_generation_moved is True
    if RESUME_TRIM_REARM_OVERHEAD_CEILING_S is None:
        print(
            "resume-trim-rearm ceiling is the PLACEHOLDER (None): not enforced, "
            "no denominator recorded — part 2 freezes the value with provenance"
        )


@pytest.mark.perf
@pytest.mark.timeout(300)
@pytest.mark.asyncio
async def test_resume_trim_rearm_trials_within_frozen_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured trials (part 2 runs these on a quiet machine with -n0).

    Each trial proves the resume-scope EFFECTS (retained descriptor cleared,
    no re-seal of the old authority during the resume, transaction
    transitioned); the median of the whole warm-resume windows is enforced
    against the frozen bounded-overhead ceiling read from
    ``RESUME_TRIM_REARM_OVERHEAD_CEILING_S``, with the clear-and-reseal
    transaction window recorded per trial for attribution. While that
    constant is the PLACEHOLDER (None) the ceiling is not enforced and no
    denominator is recorded — part 2 replaces the placeholder with the
    provenance-frozen value. A ceiling miss is investigated, never silently
    raised. A failed-proof/discard trial is recorded as excluded-negative
    honestly — the effect assertions are never weakened.
    """
    trials = [
        await _run_trial(tmp_path / f"trial-{index}", monkeypatch, trial=index)
        for index in range(TRIALS)
    ]
    windows = [t.whole_resume_s for t in trials]
    txns = [t.resume_transaction_s for t in trials]
    median_window = statistics.median(windows)
    for trial in trials:
        print(
            f"resume-trim-rearm trial {trial.trial + 1}/{TRIALS}: "
            f"whole warm resume {trial.whole_resume_s * 1_000:.3f} ms, "
            f"resume transaction {trial.resume_transaction_s * 1_000:.3f} ms, "
            f"descriptor retained before {trial.descriptor_retained_before} -> "
            f"cleared after {trial.descriptor_cleared_after}, "
            f"reseal_calls_during_resume {trial.reseal_calls_during_resume}, "
            f"txn_outcome {trial.txn_outcome}, "
            f"save_generation_moved {trial.save_generation_moved}, "
            f"reset_generation_moved {trial.reset_generation_moved}"
        )
    print(
        f"resume-trim-rearm MEDIAN of {TRIALS} trials: "
        f"{median_window * 1_000:.3f} ms whole warm resume "
        f"(transaction median {statistics.median(txns) * 1_000:.3f} ms)"
    )
    print(f"resume-trim-rearm machine context: {machine_context()}")
    record(
        "resume_trim_rearm_ceiling",
        {
            "trials": TRIALS,
            "whole_resume_ms": [round(w * 1_000, 3) for w in windows],
            "resume_transaction_ms": [round(t * 1_000, 3) for t in txns],
            "median_whole_resume_ms": round(median_window * 1_000, 3),
            "median_resume_transaction_ms": round(statistics.median(txns) * 1_000, 3),
            "descriptor_retained_before": [
                t.descriptor_retained_before for t in trials
            ],
            "descriptor_cleared_after": [t.descriptor_cleared_after for t in trials],
            "reseal_calls_during_resume": [
                t.reseal_calls_during_resume for t in trials
            ],
            "txn_outcome": [t.txn_outcome for t in trials],
            "save_generation_moved": [t.save_generation_moved for t in trials],
            "reset_generation_moved": [t.reset_generation_moved for t in trials],
            "ceiling_s": RESUME_TRIM_REARM_OVERHEAD_CEILING_S,
            "ceiling_enforced": RESUME_TRIM_REARM_OVERHEAD_CEILING_S is not None,
            "status": (
                "measured trials with per-trial resume-scope effect assertions; "
                "the ceiling constant is the part-1 PLACEHOLDER until part 2 "
                "freezes it with provenance"
                if RESUME_TRIM_REARM_OVERHEAD_CEILING_S is None
                else "measured against the provenance-frozen ceiling"
            ),
            "machine": machine_context(),
        },
    )
    # Effect assertions already fired per trial inside the runner; restate the
    # invariants here so the measured test cannot pass on a no-op resume.
    assert all(t.descriptor_retained_before for t in trials)
    assert all(t.descriptor_cleared_after for t in trials)
    assert all(t.reseal_calls_during_resume == 0 for t in trials)
    assert all(t.txn_outcome == "proved_and_transitioned" for t in trials)
    assert all(t.save_generation_moved for t in trials)
    assert all(t.reset_generation_moved for t in trials)
    if RESUME_TRIM_REARM_OVERHEAD_CEILING_S is None:
        print(
            "resume-trim-rearm ceiling is the PLACEHOLDER (None): not enforced, "
            "no denominator recorded — part 2 freezes the value with provenance"
        )
    else:
        assert median_window <= RESUME_TRIM_REARM_OVERHEAD_CEILING_S, (
            f"resume-trim-rearm whole warm-resume median "
            f"{median_window * 1_000:.3f} ms exceeds the frozen ceiling "
            f"{RESUME_TRIM_REARM_OVERHEAD_CEILING_S * 1_000:.3f} ms; investigate "
            "the miss with per-trial attribution, never raise the constant"
        )
