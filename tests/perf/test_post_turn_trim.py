"""Post-turn trim perf gate — T1b part 1 (boundary-producing harness + runner).

The shipped combined-load harness never compacts, so this file owns the
boundary-producing harness build: a scripted loop whose second turn crosses
the auto-compact threshold, producing a compaction boundary, so the post-turn
seam's drain/fence/seal/publish actually runs inside the measured window.

Measured quantity (design XX): the enclosing turn-completion ->
next-turn-admission latency, measured with THIS harness on the pinned
pre-change commit (frozen with provenance in T1b part 2). Every trial also
proves the trim EFFECT — descriptor advance, fenced publication, archive
bytes removed — so a broken trim cannot measure as fast.

The ceiling constant below is FROZEN by T1b part 2 (2026-11-02): the
provenance-frozen pre-change denominator (same harness on the pinned
pre-change commit f8fd5823) plus the bounded seam overhead, both recorded in
the constant's comment block. The smoke trial's numbers remain NOT a
denominator — the denominator is the pinned-worktree freeze recorded in the
comment block.

Serial only: run with ``-n0`` on a quiet machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
import time
from typing import Any

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._transcript_integrity import archive_reference
from chartreux.core.compaction import CompactionManager
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import Role
from chartreux.core.session.session_loader import MESSAGES_FILENAME
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session.transcript_archive import verify_archived_prefix_on_disk
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

#: The harness's auto-compact threshold; turn 2 is driven past it.
AUTO_COMPACT_THRESHOLD = 1_000

#: Payload bytes seeded onto the sealed round's assistant message, so the trim
#: observably removes archive bytes from the serialized live transcript.
PAYLOAD_BYTES = 4096

# ---------------------------------------------------------------------------
# FROZEN CEILING — landed by T1b part 2 on 2026-11-02 (quiet machine, no
# concurrent worker tests; the orchestrator held all other lanes).
#
# POST_TURN_TRIM_OVERHEAD_CEILING_S = 0.004525 s (4.525 ms) =
# frozen pre-change denominator median 1.867 ms + bounded overhead 2.658 ms.
#
# Denominator provenance: the same harness — as a minimal denominator variant
# measuring the identical turn-completion -> next-turn-admission window over
# the identical boundary-producing workload (no post-change-only trim-effect
# assertions; the baseline tree has no post-turn trim, so its window IS the
# denominator) — run in a worktree PINNED at pre-change commit
# f8fd58238aa820df43c9f0808b201fe385a5b52f ("Release v0.7.0"), with -n0:
# freeze batch of 5 trials gave windows 1.886/1.626/1.597/1.884/1.867 ms
# (median 1.867 ms), a confirmation batch of 5 more trials gave
# 1.903/1.913/1.896/1.932/1.876 ms (median 1.903 ms, drift 0.036 ms); every
# trial produced the compaction boundary; baseline fence_calls were 0 in all
# 10 trials. Post-change side: main tree at f8584f05. Machine context (both
# sides): AMD Ryzen 9 5900X 12-Core Processor, 24 cores,
# Linux-7.3.0-9-generic-x86_64 glibc 2.44, Python 3.12.13; loadavg ~1.5-1.7
# (1 min) on 24 cores before/during/after all runs.
#
# Overhead bound: 3 x the maximum measured post-turn seam cost. The only work
# the post-change side adds inside the window is the post-turn seam
# (drain/fence/publish of the 2-record archived prefix, ~4.1 KB seeded payload
# removed, one fenced save), measured on f8584f05 via the smoke node over 5
# single-trial runs (-n0): seams 0.794/0.847/0.832/0.886/0.875 ms, max
# 0.886 ms; 3 x 0.886 ms = 2.658 ms. The seam's workload is bounded by the
# archived prefix, not by turn count; the 3x factor covers trial-level seam
# variance and scheduling jitter. Smoke windows were 2.818-3.050 ms, so this
# ceiling carries ~1.5 ms headroom over the worst pre-trial smoke window.
#
# A ceiling miss is investigated with per-trial attribution, never silently
# raised.
# ---------------------------------------------------------------------------
POST_TURN_TRIM_OVERHEAD_CEILING_S: float | None = 0.004525


@dataclass(slots=True)
class PostTurnTrimTrial:
    """One measured trial plus its proven trim effects."""

    trial: int
    turn_completion_to_next_admission_s: float
    seam_s: float
    archived_count: int
    serialized_bytes_before: int
    serialized_bytes_after: int
    save_generation_moved: bool
    fence_calls: int


def _serialized_live_bytes(loop: AgentLoop) -> int:
    non_system = [m for m in loop.messages if m.role is not Role.system]
    return len(
        SessionLogger._transcript_bytes([
            SessionLogger._message_record(m, allow_archived=True) for m in non_system
        ])
    )


def _build_boundary_harness(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[AgentLoop, FakeBackend, dict[str, Any]]:
    """Build a loop that produces a compaction boundary on turn 2.

    Returns the loop, its scripted backend, and the instrumentation dict
    (seal_turn timestamps, post-turn entry windows, backend request
    timestamps, fence call count).
    """
    config = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=AUTO_COMPACT_THRESHOLD),
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(root / "sessions"), session_prefix="perf-trim"
        ),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="first answer")],
        [mock_llm_chunk(content="<summary>perf compacted</summary>")],
        [mock_llm_chunk(content="second answer")],
        [mock_llm_chunk(content="third answer")],
    ])
    loop = build_test_agent_loop(config=config, backend=backend)
    instrumentation: dict[str, Any] = {
        "seal_turn_done": [],
        "entry": [],
        "requests": [],
        "fence_calls": 0,
    }

    # Turn completion: the moment seal_turn() returns at the end of a turn.
    recorder = loop.checkpoint_recorder
    real_seal_turn = recorder.seal_turn

    def _timed_seal_turn() -> None:
        real_seal_turn()
        instrumentation["seal_turn_done"].append(time.perf_counter())

    monkeypatch.setattr(recorder, "seal_turn", _timed_seal_turn)

    # The post-finalization seam itself (the guarded entry).
    real_entry = CompactionManager.capture_post_turn_pending_trim

    async def _timed_entry(self: CompactionManager, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return await real_entry(self, **kwargs)
        finally:
            instrumentation["entry"].append((start, time.perf_counter()))

    monkeypatch.setattr(
        CompactionManager, "capture_post_turn_pending_trim", _timed_entry
    )

    # Fenced publication count.
    real_fence = SessionLogger.fence_and_publish_archive_trim

    def _fence_spy(self: SessionLogger, *args: Any, **kwargs: Any) -> Any:
        instrumentation["fence_calls"] += 1
        return real_fence(self, *args, **kwargs)

    monkeypatch.setattr(SessionLogger, "fence_and_publish_archive_trim", _fence_spy)

    # Backend request timestamps (turn 3's first request = next admission).
    real_complete = backend.complete

    async def _timed_complete(**kwargs: Any) -> Any:
        instrumentation["requests"].append(time.perf_counter())
        return await real_complete(**kwargs)

    monkeypatch.setattr(backend, "complete", _timed_complete)
    return loop, backend, instrumentation


async def _run_trial(
    root: Path, monkeypatch: pytest.MonkeyPatch, trial: int
) -> PostTurnTrimTrial:
    """Run one boundary-producing trial and prove the trim EFFECT inside it.

    The effect assertions (descriptor advance, fenced publication, archive
    bytes removed) fire in every trial — smoke included — so the measured
    window can never be produced by a trim that did not run.
    """
    loop, _backend, instrumentation = _build_boundary_harness(root, monkeypatch)
    try:
        # Turn 1: one settled round (user + assistant).
        async for _event in loop.act("first question"):
            pass
        # Seed payload bytes onto the sealed round's assistant message; the
        # post-turn trim must remove them from the live serialized transcript.
        assistant = next(m for m in loop.messages if m.role is Role.assistant)
        assistant.reasoning_payloads = [{"opaque": "x" * PAYLOAD_BYTES}]
        logger = loop.session_logger
        serialized_before = _serialized_live_bytes(loop)
        generation_before = logger.transcript_save_generation
        descriptor_before = logger.archive_descriptor

        # Turn 2 crosses the auto-compact threshold: the boundary is produced
        # mid-turn and the post-turn seam drains/fences/seals/publishes it.
        loop.stats.context_tokens = AUTO_COMPACT_THRESHOLD + 1
        async for _event in loop.act("second question"):
            pass

        # --- trim EFFECT assertions (they fire in the smoke trial too) ---
        assert any(m.context_boundary == "compaction" for m in loop.messages), (
            "the harness must produce a compaction boundary on turn 2"
        )
        descriptor = loop.messages.archive_descriptor
        assert descriptor is not None, "the post-turn trim must publish a descriptor"
        assert descriptor is not descriptor_before, "the descriptor must advance"
        assert descriptor.archived_count == 2, "turn 1's settled round is sealed"
        assert logger.archive_descriptor is descriptor
        assert instrumentation["fence_calls"] == 1, "exactly one fenced publication"
        assert logger.transcript_save_generation > generation_before, (
            "the publication fence must move the save generation"
        )
        non_system = [m for m in loop.messages if m.role is not Role.system]
        sealed = non_system[: descriptor.archived_count]
        assert all(archive_reference(m) is descriptor for m in sealed)
        assert all(m.reasoning_payloads is None for m in sealed), (
            "the sealed slots' payload bytes are removed from RAM"
        )
        # Archive bytes removed: the sealed region's serialized form shrinks by
        # at least the seeded payload bytes (compared slot-for-slot).
        sealed_bytes_after = len(
            SessionLogger._transcript_bytes([
                SessionLogger._message_record(m, allow_archived=True) for m in sealed
            ])
        )
        assert serialized_before - sealed_bytes_after >= PAYLOAD_BYTES, (
            "archive bytes are removed from the sealed region's serialized form"
        )
        session_dir = logger.session_dir
        assert session_dir is not None
        verify_archived_prefix_on_disk(descriptor, session_dir / MESSAGES_FILENAME)

        # Timing: turn completion (seal_turn end) -> next-turn admission.
        turn_completion_at = instrumentation["seal_turn_done"][-1]
        entry_start, entry_end = instrumentation["entry"][-1]
        requests_after_turn2 = len(instrumentation["requests"])

        # Turn 3: the next admission; its first model request timestamps it.
        async for _event in loop.act("third question"):
            pass
        next_admission_at = instrumentation["requests"][requests_after_turn2]

        return PostTurnTrimTrial(
            trial=trial,
            turn_completion_to_next_admission_s=next_admission_at - turn_completion_at,
            seam_s=entry_end - turn_completion_at,
            archived_count=descriptor.archived_count,
            serialized_bytes_before=serialized_before,
            serialized_bytes_after=sealed_bytes_after,
            save_generation_moved=logger.transcript_save_generation > generation_before,
            fence_calls=instrumentation["fence_calls"],
        )
    finally:
        await loop.aclose()


@pytest.mark.perf
@pytest.mark.timeout(300)
@pytest.mark.asyncio
async def test_post_turn_trim_smoke_single_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SMOKE CHECK ONLY: a single trial proving the harness executes and its
    effect assertions fire. The numbers printed here are NOT denominators —
    part 2 freezes the pre-change denominator and the ceiling value on a
    quiet machine; nothing is frozen or recorded as a denominator here.
    """
    trial = await _run_trial(tmp_path / "smoke", monkeypatch, trial=0)
    print(
        f"post-turn-trim SMOKE trial (NOT a denominator): "
        f"turn-completion->next-admission {trial.turn_completion_to_next_admission_s * 1_000:.2f} ms, "
        f"seam {trial.seam_s * 1_000:.2f} ms, archived_count {trial.archived_count}, "
        f"serialized {trial.serialized_bytes_before} -> {trial.serialized_bytes_after} bytes "
        f"({trial.serialized_bytes_before - trial.serialized_bytes_after} removed), "
        f"fence_calls {trial.fence_calls}, save_generation_moved {trial.save_generation_moved}"
    )
    print(f"post-turn-trim SMOKE machine context: {machine_context()}")
    record(
        "post_turn_trim_smoke",
        {
            "status": (
                "SMOKE ONLY — the harness executes and the trim effect "
                "assertions fired; this output is NOT a denominator and no "
                "value is frozen here (part 2 owns the freeze)"
            ),
            "trial": trial.trial,
            "turn_completion_to_next_admission_ms": round(
                trial.turn_completion_to_next_admission_s * 1_000, 3
            ),
            "seam_ms": round(trial.seam_s * 1_000, 3),
            "archived_count": trial.archived_count,
            "serialized_bytes_removed": trial.serialized_bytes_before
            - trial.serialized_bytes_after,
            "fence_calls": trial.fence_calls,
            "machine": machine_context(),
        },
    )
    assert trial.archived_count == 2
    assert trial.fence_calls == 1
    assert trial.save_generation_moved is True


@pytest.mark.perf
@pytest.mark.timeout(300)
@pytest.mark.asyncio
async def test_post_turn_trim_trials_within_frozen_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured trials (part 2 runs these on a quiet machine with -n0).

    Each trial proves the trim effect (descriptor advance, fenced
    publication, archive bytes removed); the median of the
    turn-completion -> next-turn-admission windows is enforced against the
    frozen bounded-overhead ceiling read from
    ``POST_TURN_TRIM_OVERHEAD_CEILING_S``. While that constant is the
    PLACEHOLDER (None) the ceiling is not enforced and no denominator is
    recorded — part 2 replaces the placeholder with the provenance-frozen
    value. A ceiling miss is investigated, never silently raised.
    """
    trials = [
        await _run_trial(tmp_path / f"trial-{index}", monkeypatch, trial=index)
        for index in range(TRIALS)
    ]
    windows = [t.turn_completion_to_next_admission_s for t in trials]
    seams = [t.seam_s for t in trials]
    median_window = statistics.median(windows)
    for trial in trials:
        print(
            f"post-turn-trim trial {trial.trial + 1}/{TRIALS}: "
            f"turn-completion->next-admission {trial.turn_completion_to_next_admission_s * 1_000:.2f} ms, "
            f"seam {trial.seam_s * 1_000:.2f} ms, archived_count {trial.archived_count}, "
            f"serialized {trial.serialized_bytes_before} -> {trial.serialized_bytes_after} bytes, "
            f"fence_calls {trial.fence_calls}"
        )
    print(
        f"post-turn-trim MEDIAN of {TRIALS} trials: "
        f"{median_window * 1_000:.2f} ms turn-completion->next-admission "
        f"(seam median {statistics.median(seams) * 1_000:.2f} ms)"
    )
    print(f"post-turn-trim machine context: {machine_context()}")
    record(
        "post_turn_trim_ceiling",
        {
            "trials": TRIALS,
            "window_ms": [round(w * 1_000, 3) for w in windows],
            "seam_ms": [round(s * 1_000, 3) for s in seams],
            "median_window_ms": round(median_window * 1_000, 3),
            "median_seam_ms": round(statistics.median(seams) * 1_000, 3),
            "archived_count": [t.archived_count for t in trials],
            "serialized_bytes_removed": [
                t.serialized_bytes_before - t.serialized_bytes_after for t in trials
            ],
            "fence_calls": [t.fence_calls for t in trials],
            "ceiling_s": POST_TURN_TRIM_OVERHEAD_CEILING_S,
            "ceiling_enforced": POST_TURN_TRIM_OVERHEAD_CEILING_S is not None,
            "status": (
                "measured trials with per-trial effect assertions; the ceiling "
                "constant is the part-1 PLACEHOLDER until part 2 freezes it "
                "with provenance"
                if POST_TURN_TRIM_OVERHEAD_CEILING_S is None
                else "measured against the provenance-frozen ceiling"
            ),
            "machine": machine_context(),
        },
    )
    # Effect assertions already fired per trial inside the runner; restate the
    # invariants here so the measured test cannot pass on a no-trim run.
    assert all(t.archived_count == 2 for t in trials)
    assert all(t.fence_calls == 1 for t in trials)
    assert all(t.save_generation_moved for t in trials)
    assert all(t.serialized_bytes_after < t.serialized_bytes_before for t in trials)
    if POST_TURN_TRIM_OVERHEAD_CEILING_S is None:
        print(
            "post-turn-trim ceiling is the PLACEHOLDER (None): not enforced, "
            "no denominator recorded — part 2 freezes the value with provenance"
        )
    else:
        assert median_window <= POST_TURN_TRIM_OVERHEAD_CEILING_S, (
            f"post-turn turn-completion->next-admission median "
            f"{median_window * 1_000:.2f} ms exceeds the frozen ceiling "
            f"{POST_TURN_TRIM_OVERHEAD_CEILING_S * 1_000:.2f} ms; investigate "
            "the miss with per-trial attribution, never raise the constant"
        )
