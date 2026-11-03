"""M1 trim savings measurement — T3 (records what the trim actually frees).

The 0.7.0 "M1" claim — that transcript-archive trimming saves memory — was
never measured. This harness measures it against the LANDED post-turn trim
(``SessionLogger.complete_post_turn_trim``, T1b) and the payload-only
replacement model (``archived_replacement(..., trim_mode="payloads")``), which
releases ONLY ``reasoning_payloads`` from the sealed slots; content, tool
outputs, tool structure and display fields stay in RAM.

Two deterministic, realistic transcript mixes (tool-output-heavy and
reasoning-heavy) plus a deliberately large-effect POSITIVE CONTROL mix are each
measured in two arms with IDENTICAL fixture construction:

- ``trim`` — the shipped flag on: the turn's post-turn seam seals and publishes
  the archive trim;
- ``control`` — the shipped flag off: the same turn runs, nothing is trimmed.

Per trial the harness records, separately:

(a) the tracemalloc post-GC snapshot delta of the in-process retained set
    across the turn (``tests/perf/_metrics.py`` ``alloc_snapshot``), in bytes;
(b) the serialized bytes attributable to ``reasoning_payloads`` in the live
    transcript (its own serializer pass — evidence distinct from (a));
(c) RSS as a SIDE observation only (recorded, never asserted).

Every measured ``trim`` trial asserts the trim EFFECT (descriptor published and
advanced, sealed slots carry the archive reference, payloads gone, archived
prefix bytes verified on disk, serialized reasoning bytes actually removed), so
a trim that did not run cannot measure as a saving. Every ``control`` trial
asserts the matched no-trim state (no descriptor, payloads retained) so the
control cannot silently become a trim.

The test asserts only harness-level invariants: the effects fired, and the
POSITIVE CONTROL's trim-minus-control saving exceeds the control's own
run-to-run spread (the noise band) — proving the harness can detect an effect.
Whether either realistic mix shows an effect is DATA, recorded honestly in
``docs/project/trim-savings.md``: within the noise band means "no effect", and
no minimum effect is invented.

Serial only: run with ``-n0`` on a quiet machine.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import statistics
import tracemalloc
from typing import Any

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._transcript_integrity import archive_reference
from chartreux.core.compaction import manager as compaction_manager_module
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import (
    FunctionCall,
    LLMMessage,
    PersistedToolResult,
    Role,
    ToolCall,
)
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session.transcript_archive import verify_archived_prefix_on_disk
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from tests.mock.utils import mock_llm_chunk
from tests.perf._metrics import alloc_snapshot, machine_context, record, rss_mib
from tests.stubs.fake_backend import FakeBackend

#: Measured trials per (mix, arm) cell.
TRIALS = 5

#: The harness's auto-compact threshold; the measured turn is driven past it so
#: a compaction boundary exists and the post-turn seam seals the mix.
AUTO_COMPACT_THRESHOLD = 1_000

#: Bytes removed per payload-bearing assistant message (deterministic filler).
_KIB = 1024
_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class Mix:
    """One deterministic transcript mix (identical construction in both arms)."""

    key: str
    description: str
    rounds: int
    reasoning_payload_bytes: int
    tool_output_bytes: int


MIXES: tuple[Mix, ...] = (
    Mix(
        key="tool-output-heavy",
        description=(
            "8 rounds whose tool results dominate (96 KiB per tool result, "
            "8 KiB reasoning payload per tool-calling assistant)"
        ),
        rounds=8,
        reasoning_payload_bytes=8 * _KIB,
        tool_output_bytes=96 * _KIB,
    ),
    Mix(
        key="reasoning-heavy",
        description=(
            "6 rounds whose reasoning payloads dominate (256 KiB per "
            "tool-calling assistant, 4 KiB per tool result)"
        ),
        rounds=6,
        reasoning_payload_bytes=256 * _KIB,
        tool_output_bytes=4 * _KIB,
    ),
    Mix(
        key="positive-control",
        description=(
            "deliberately large-effect arm: 4 rounds with 2 MiB reasoning "
            "payloads each, engineered far outside the noise band"
        ),
        rounds=4,
        reasoning_payload_bytes=2 * _MIB,
        tool_output_bytes=_KIB,
    ),
)

#: The arm measured as the positive control (must show a detectable effect).
POSITIVE_CONTROL_KEY = "positive-control"


@dataclass(slots=True)
class TrialResult:
    """One measured trial with its proven state."""

    mix: str
    arm: str
    trial: int
    tracemalloc_delta_bytes: int
    reasoning_bytes_before: int
    reasoning_bytes_after: int
    rss_before_mib: float
    rss_after_mib: float
    archived_count: int
    descriptor_published: bool
    loadavg: tuple[float, float, float] | None


def _filler(label: str, size: int) -> str:
    """Deterministic, realistically shaped filler of exactly ``size`` chars."""
    line = f"{label}: artifact line with ordinary transcript text 0123456789\n"
    repeats = size // len(line) + 1
    return (line * repeats)[:size]


def _seed_mix(loop: AgentLoop, mix: Mix) -> int:
    """Append the mix to the loop's message list; return its message count.

    Identical construction in both arms: this never reads the trim flag.
    """
    count = 0
    for round_index in range(mix.rounds):
        call_id = f"call-{round_index}"
        loop.messages.append(
            LLMMessage(
                role=Role.user,
                content=f"Round {round_index}: inspect artifact {round_index}.",
                message_id=f"u{round_index}",
                input_text=f"Round {round_index}: inspect artifact {round_index}.",
            )
        )
        loop.messages.append(
            LLMMessage(
                role=Role.assistant,
                content=f"Checking artifact {round_index}.",
                message_id=f"a{round_index}-call",
                reasoning_content=f"Thinking about artifact {round_index}.",
                reasoning_payloads=[
                    {
                        "opaque": _filler(
                            f"reasoning-{round_index}", mix.reasoning_payload_bytes
                        )
                    }
                ],
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        index=0,
                        function=FunctionCall(
                            name="bash",
                            arguments=json.dumps({
                                "command": f"cat artifact-{round_index}"
                            }),
                        ),
                    )
                ],
            )
        )
        loop.messages.append(
            LLMMessage(
                role=Role.tool,
                tool_call_id=call_id,
                name="bash",
                content=_filler(f"tool-output-{round_index}", mix.tool_output_bytes),
                tool_result=PersistedToolResult(
                    output={"exit_code": 0, "truncated": False}
                ),
            )
        )
        loop.messages.append(
            LLMMessage(
                role=Role.assistant,
                content=f"Artifact {round_index} inspected.",
                message_id=f"a{round_index}-done",
            )
        )
        count += 4
    return count


def _build_harness(
    root: Path, mix: Mix, *, trim_enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[AgentLoop, FakeBackend]:
    """Build the loop for one measured trial (identical fixture in both arms)."""
    config = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=AUTO_COMPACT_THRESHOLD),
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(root / "sessions"), session_prefix="perf-savings"
        ),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="<summary>perf compacted</summary>")],
        [mock_llm_chunk(content="final answer")],
    ])
    loop = build_test_agent_loop(config=config, backend=backend)
    # Tracing starts BEFORE the mix is allocated, so freeing the payload
    # allocations is observable in the post-GC snapshot delta.
    alloc_snapshot()
    _seed_mix(loop, mix)
    # The shipped flag is the arm selector: BOTH arms set it explicitly so a
    # previous control trial can never leak its disabled flag into a trim trial.
    monkeypatch.setattr(compaction_manager_module, "ARCHIVE_TRIM_ENABLED", trim_enabled)
    return loop, backend


def _reasoning_payload_bytes(messages: Any) -> int:
    """Serialized transcript bytes attributable to ``reasoning_payloads``.

    Evidence (b): a dedicated serializer pass over the live transcript, using
    the real record/transcript serializers, independent of the tracemalloc
    number in evidence (a).
    """
    records = [
        SessionLogger._message_record(message, allow_archived=True)
        for message in messages
        if message.role is not Role.system
    ]
    with_payloads = len(SessionLogger._transcript_bytes(records))
    stripped: list[dict[str, Any]] = []
    for entry in records:
        if "reasoning_payloads" in entry:
            entry = dict(entry)
            entry.pop("reasoning_payloads")
        stripped.append(entry)
    without_payloads = len(SessionLogger._transcript_bytes(stripped))
    return with_payloads - without_payloads


def _delta_bytes(before: tracemalloc.Snapshot, after: tracemalloc.Snapshot) -> int:
    """Net traced-bytes delta between two snapshots (the retained-set change)."""
    return sum(
        statistic.size_diff for statistic in after.compare_to(before, "filename")
    )


def _non_system(messages: Any) -> list[LLMMessage]:
    return [message for message in messages if message.role is not Role.system]


async def _run_trial(
    root: Path,
    mix: Mix,
    *,
    trim_enabled: bool,
    trial: int,
    monkeypatch: pytest.MonkeyPatch,
) -> TrialResult:
    """Run one trial and prove the arm's state inside it.

    ``trim`` trials assert the trim EFFECTS; ``control`` trials assert the
    matched no-trim state. No trial may pass without proving which state it
    produced, so neither a silent no-trim "saving" nor a trimmed "control" can
    be recorded.
    """
    loop, backend = _build_harness(
        root, mix, trim_enabled=trim_enabled, monkeypatch=monkeypatch
    )
    try:
        # The fixture already exists (tracing started before it was allocated);
        # its payload bytes are the baseline the measured turn runs against.
        reasoning_before = _reasoning_payload_bytes(loop.messages)
        gc.collect()
        before = alloc_snapshot()
        rss_before = rss_mib()

        # The measured turn: driven past the auto-compact threshold, so a
        # boundary exists and the post-turn seam owns the trim opportunity.
        loop.stats.context_tokens = AUTO_COMPACT_THRESHOLD + 1
        async for _event in loop.act("final check"):
            pass

        # Drop the only non-list references to the originals (the backend's
        # recorded request snapshots) before the post-GC retained-set reading.
        backend.requests_messages.clear()
        reasoning_after = _reasoning_payload_bytes(loop.messages)
        rss_after = rss_mib()
        gc.collect()
        after = alloc_snapshot()
        delta = _delta_bytes(before, after)

        # --- effect assertions: which state did this trial actually produce? ---
        descriptor = loop.messages.archive_descriptor
        logger = loop.session_logger
        expected_count = mix.rounds * 4
        if trim_enabled:
            assert descriptor is not None, "the trim arm must publish a descriptor"
            assert descriptor.archived_count == expected_count, (
                "the whole settled mix is sealed"
            )
            assert logger.archive_descriptor is descriptor
            sealed = _non_system(loop.messages)[: descriptor.archived_count]
            assert all(archive_reference(message) is descriptor for message in sealed)
            assert all(message.reasoning_payloads is None for message in sealed), (
                "the sealed slots' payload bytes are removed from RAM"
            )
            assert reasoning_after == 0, "no reasoning payload bytes remain live"
            assert reasoning_before >= mix.rounds * mix.reasoning_payload_bytes, (
                "the fixture really carried the engineered payload volume"
            )
            session_dir = logger.session_dir
            assert session_dir is not None
            verify_archived_prefix_on_disk(descriptor, session_dir / "messages.jsonl")
        else:
            assert descriptor is None, "the control arm must never trim"
            assert logger.archive_descriptor is None
            retained = _non_system(loop.messages)[:expected_count]
            # Every fourth record is the payload-bearing tool-calling assistant.
            assert all(
                retained[index].reasoning_payloads is not None
                for index in range(1, expected_count, 4)
            ), "the control retains every payload"
            assert reasoning_after == reasoning_before, (
                "the control removes no serialized reasoning bytes"
            )
            assert reasoning_before >= mix.rounds * mix.reasoning_payload_bytes

        return TrialResult(
            mix=mix.key,
            arm="trim" if trim_enabled else "control",
            trial=trial,
            tracemalloc_delta_bytes=delta,
            reasoning_bytes_before=reasoning_before,
            reasoning_bytes_after=reasoning_after,
            rss_before_mib=rss_before,
            rss_after_mib=rss_after,
            archived_count=descriptor.archived_count if descriptor else 0,
            descriptor_published=descriptor is not None,
            loadavg=os.getloadavg(),
        )
    finally:
        await loop.aclose()
        tracemalloc.stop()


def _summarize_cell(results: list[TrialResult]) -> dict[str, Any]:
    """Median + spread for one (mix, arm) cell."""
    deltas = [result.tracemalloc_delta_bytes for result in results]
    reasoning = [
        result.reasoning_bytes_before - result.reasoning_bytes_after
        for result in results
    ]
    return {
        "trials": len(results),
        "delta_bytes": deltas,
        "delta_median_bytes": int(statistics.median(deltas)),
        "delta_spread_bytes": max(deltas) - min(deltas),
        "reasoning_bytes_removed": reasoning,
        "reasoning_bytes_removed_median": int(statistics.median(reasoning)),
        "rss_mib_delta": [
            round(result.rss_after_mib - result.rss_before_mib, 3) for result in results
        ],
        "loadavg": [result.loadavg for result in results],
    }


# ---------------------------------------------------------------------------
# Smoke: the harness executes and both arm states are proven
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.timeout(300)
@pytest.mark.asyncio
async def test_trim_savings_smoke_single_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SMOKE CHECK ONLY: one trim trial and one control trial on the
    reasoning-heavy mix; the numbers are NOT results (the measured test and
    docs/project/trim-savings.md own those).
    """
    mix = next(item for item in MIXES if item.key == "reasoning-heavy")
    trim = await _run_trial(
        tmp_path / "smoke-trim",
        mix,
        trim_enabled=True,
        trial=0,
        monkeypatch=monkeypatch,
    )
    control = await _run_trial(
        tmp_path / "smoke-control",
        mix,
        trim_enabled=False,
        trial=0,
        monkeypatch=monkeypatch,
    )
    print(
        f"trim-savings SMOKE trim: tracemalloc delta "
        f"{trim.tracemalloc_delta_bytes} B, reasoning removed "
        f"{trim.reasoning_bytes_before - trim.reasoning_bytes_after} B, "
        f"archived_count {trim.archived_count}"
    )
    print(
        f"trim-savings SMOKE control: tracemalloc delta "
        f"{control.tracemalloc_delta_bytes} B, reasoning removed "
        f"{control.reasoning_bytes_before - control.reasoning_bytes_after} B, "
        f"archived_count {control.archived_count}"
    )
    print(f"trim-savings SMOKE machine context: {machine_context()}")
    assert trim.descriptor_published is True
    assert trim.archived_count == mix.rounds * 4
    assert trim.reasoning_bytes_after == 0
    assert control.descriptor_published is False
    assert control.reasoning_bytes_after == control.reasoning_bytes_before


# ---------------------------------------------------------------------------
# The measured trials
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.timeout(900)
@pytest.mark.asyncio
async def test_trim_savings_measured_cells_report_effects_and_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measure every mix in both arms and record the honest result.

    Asserted harness invariants: every trim trial fired its effects (asserted
    inside the runner), every control trial stayed untrimmed, and the POSITIVE
    CONTROL's trim-minus-control saving exceeds the control's own run-to-run
    spread. The realistic mixes' outcomes are data, not assertions: a
    within-noise cell is recorded as no effect.
    """
    results: list[TrialResult] = []
    for mix_index, mix in enumerate(MIXES):
        for arm_index, trim_enabled in enumerate((True, False)):
            for trial in range(TRIALS):
                results.append(
                    await _run_trial(
                        tmp_path / f"{mix_index}-{arm_index}-{trial}",
                        mix,
                        trim_enabled=trim_enabled,
                        trial=trial,
                        monkeypatch=monkeypatch,
                    )
                )

    cells: dict[str, dict[str, Any]] = {}
    verdicts: dict[str, str] = {}
    for mix in MIXES:
        trim_cell = _summarize_cell([
            result
            for result in results
            if result.mix == mix.key and result.arm == "trim"
        ])
        control_cell = _summarize_cell([
            result
            for result in results
            if result.mix == mix.key and result.arm == "control"
        ])
        noise = control_cell["delta_spread_bytes"]
        saving = control_cell["delta_median_bytes"] - trim_cell["delta_median_bytes"]
        cells[mix.key] = {
            "description": mix.description,
            "trim": trim_cell,
            "control": control_cell,
            "noise_band_bytes": noise,
            "trim_minus_control_bytes": saving,
            "effect_exceeds_noise": saving > noise,
        }
        verdicts[mix.key] = (
            "effect exceeds the control noise band"
            if saving > noise
            else "no effect distinguishable from the control's noise band"
        )

    print(f"trim-savings measured results ({TRIALS} trials per cell):")
    for mix in MIXES:
        cell = cells[mix.key]
        print(
            f"  {mix.key}: trim median delta "
            f"{cell['trim']['delta_median_bytes']} B "
            f"(spread {cell['trim']['delta_spread_bytes']} B), control median "
            f"{cell['control']['delta_median_bytes']} B "
            f"(spread {cell['control']['delta_spread_bytes']} B), "
            f"saving {cell['trim_minus_control_bytes']} B vs noise "
            f"{cell['noise_band_bytes']} B -> {verdicts[mix.key]}"
        )
        print(
            f"  {mix.key}: serialized reasoning bytes removed "
            f"{cell['trim']['reasoning_bytes_removed']} (trim) vs "
            f"{cell['control']['reasoning_bytes_removed']} (control)"
        )
    print(f"trim-savings machine context: {machine_context()}")
    record(
        "trim_savings",
        {
            "status": (
                "measured: per-trial trim effects asserted; realistic-mix "
                "outcomes reported honestly against the control noise band"
            ),
            "trials_per_cell": TRIALS,
            "serials_only": True,
            "trial_details": [
                {
                    "mix": result.mix,
                    "arm": result.arm,
                    "trial": result.trial,
                    "tracemalloc_delta_bytes": result.tracemalloc_delta_bytes,
                    "reasoning_bytes_before": result.reasoning_bytes_before,
                    "reasoning_bytes_after": result.reasoning_bytes_after,
                    "rss_before_mib": result.rss_before_mib,
                    "rss_after_mib": result.rss_after_mib,
                    "archived_count": result.archived_count,
                    "descriptor_published": result.descriptor_published,
                    "loadavg": result.loadavg,
                }
                for result in results
            ],
            "cells": cells,
            "verdicts": verdicts,
            "machine": machine_context(),
        },
    )

    # Harness invariants only.
    for result in results:
        if result.arm == "trim":
            assert result.descriptor_published is True
            assert result.reasoning_bytes_after == 0
        else:
            assert result.descriptor_published is False
            assert result.reasoning_bytes_before == result.reasoning_bytes_after
    positive = cells[POSITIVE_CONTROL_KEY]
    assert positive["effect_exceeds_noise"] is True, (
        "the POSITIVE CONTROL must show a trim saving larger than the "
        "control's own run-to-run spread; if it does not, the harness cannot "
        "detect an effect and the harness — not the feature — is broken"
    )
