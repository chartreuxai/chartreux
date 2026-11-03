# Trim savings measurement

Chartreux archives a sealed prefix of the session transcript and, since 0.7.0,
releases the sealed slots' `reasoning_payloads` from memory ("payload-only"
trim). The post-turn trim seam (0.7.1) makes that trim run after a completed
turn instead of only during a compaction. This page records what that trim
actually saves, measured rather than asserted. It is the prerequisite evidence
for deciding whether a structural tombstone (which would also release message
content) is warranted.

The measurement is honest about null results: a saving within the control's
noise band would be reported as "no effect", and no minimum effect is claimed
here.

## Environment

- Machine: AMD Ryzen 9 5900X 12-Core Processor, 24 logical cores
- Platform: Linux-7.3.0-9-generic-x86_64 (glibc 2.44), Python 3.12.13
- Checkout: branch `development` at `5fc15d2d` (post-turn trim landed)
- Load: 1-minute load average 1.40-1.97 on 24 cores, recorded per trial
- Trials: 5 per cell (3 transcript mixes x 2 arms), run serially
  (`pytest -n0`) on a quiet machine with no other worker test runs active

## Method

The harness is `tests/perf/test_trim_savings.py`. Each trial builds a fresh
agent loop with session logging enabled, seeds one deterministic transcript
mix, and drives a single turn past the auto-compact threshold so a compaction
boundary exists. The landed post-turn seam then seals and publishes the settled
prefix.

Each mix is measured in two arms with identical fixture construction:

| Arm | Shipped trim flag | Effect |
| --- | --- | --- |
| `trim` | on | the turn's post-turn seam seals the settled mix and publishes payload-only replacements |
| `control` | off | the same turn runs; nothing is trimmed |

Three transcript mixes are measured:

- `tool-output-heavy`: 8 rounds, 96 KiB of tool output per round, 8 KiB
  reasoning payload per tool-calling assistant (768 KiB tool output, 64 KiB
  reasoning).
- `reasoning-heavy`: 6 rounds, 256 KiB reasoning payload per tool-calling
  assistant, 4 KiB tool output per round (1.5 MiB reasoning, 24 KiB tool
  output).
- `positive-control`: 4 rounds with 2 MiB reasoning payloads each (8 MiB
  engineered removable volume), far outside the noise band; it proves the
  harness can detect an effect.

Per trial the harness records two independent evidences plus one side
observation:

1. **Retained-set delta (tracemalloc).** A post-GC `tracemalloc` snapshot
   delta of the in-process retained set across the turn, using the shared
   `tests/perf/_metrics.py` snapshot helpers. Tracing starts before the mix is
   allocated, so freeing payload allocations is observable. `tracemalloc`
   inflates ordinary allocation accounting, so the delta is a relative
   measurement.
2. **Serialized reasoning bytes retained.** A separate serializer pass with
   the real transcript serializers computes the bytes attributable to
   `reasoning_payloads` in the live transcript. This is independent of the
   tracemalloc number. Serialized bytes include JSON escaping, so a payload
   with N newlines serializes slightly larger than its in-memory string.
3. **RSS.** Recorded before and after as a noisy side observation only; it is
   not asserted.

Every `trim` trial asserts the trim effect: a descriptor is published,
`archived_count` equals the whole settled mix, sealed slots carry the archive
reference with `reasoning_payloads is None`, the archived prefix verifies on
disk, and the serialized reasoning bytes drop to zero. Every `control` trial
asserts the matched no-trim state. A trim that did not run cannot measure as a
saving.

The noise band is the control arm's own run-to-run spread (max minus min of
per-trial tracemalloc deltas) in the same cell. The saving is the control
median delta minus the trim median delta (positive means the trim frees more).
The test asserts only harness invariants: the effects fired and the positive
control's saving exceeds its cell's noise band. The realistic mixes' outcomes
are data, reported below.

## Results

Tracemalloc deltas are in bytes; a positive delta means the turn retained more
than it freed, a negative delta means it freed more than it retained.

| Mix | Seeded reasoning | Trim median delta | Trim spread | Control median delta | Control spread (noise) | Saving | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tool-output-heavy | 65,536 B | +408,686 B | 8,469 B | +474,476 B | 1,347 B | 65,790 B | effect exceeds noise |
| reasoning-heavy | 1,572,864 B | -1,097,972 B | 2,610 B | +474,635 B | 3,391 B | 1,572,607 B | effect exceeds noise |
| positive-control | 8,388,608 B | -7,914,277 B | 1,961 B | +475,012 B | 3,747 B | 8,389,289 B | effect exceeds noise |

Per-trial tracemalloc deltas (bytes, trials in run order):

| Mix | Trim arm | Control arm |
| --- | --- | --- |
| tool-output-heavy | 416,414 / 408,686 / 408,344 / 409,404 / 407,945 | 475,078 / 474,333 / 475,463 / 474,476 / 474,116 |
| reasoning-heavy | -1,097,659 / -1,100,163 / -1,098,797 / -1,097,553 / -1,097,972 | 475,522 / 474,539 / 472,131 / 474,635 / 474,685 |
| positive-control | -7,914,554 / -7,914,277 / -7,914,085 / -7,915,451 / -7,913,490 | 472,214 / 475,012 / 475,161 / 475,961 / 474,561 |

Serialized reasoning bytes removed were identical across trials in each cell:
66,816 B for `tool-output-heavy`, 1,596,234 B for `reasoning-heavy`, and
8,512,128 B for `positive-control`; the control cells removed 0 B. The
serialized figures exceed the in-memory payload volume by the JSON newline
escaping in the payload text.

## Verdict

The payload-only trim does save memory, reproducibly. The measured tracemalloc
saving equals the sealed slots' reasoning-payload volume (within a few hundred
bytes of dictionary overhead): about 65.8 KB for the 64 KiB reasoning in the
tool-output-heavy mix, about 1.57 MB for the 1.5 MiB in the reasoning-heavy
mix, and about 8.39 MB for the 8 MiB positive control. The positive control
saving is roughly two thousand times its cell's noise band, so the harness
detects an effect when one exists.

The decisive limitation is scope, not magnitude. The trim releases only
`reasoning_payloads`. In the tool-output-heavy mix the transcript carries
768 KiB of tool output and 64 KiB of reasoning; the trim saved about 65.8 KB
and the 768 KiB of tool output stayed resident. For transcripts dominated by
tool output, payload-only trimming saves little of the transcript's weight.

No effect is invented: both realistic mixes exceeded their control noise bands
and are reported as effects, but the tool-output-heavy effect is small in
absolute terms and covers only reasoning payloads. A structural tombstone would
address the remaining tool-output and content weight; this measurement does not
claim that it would improve user-visible memory without its own evidence.

## Reproducing

From the repository root, on a quiet machine and serially:

```bash
uv run pytest tests/perf/test_trim_savings.py -n0
```

The test prints a `PERF-BLOB:trim_savings:` line with per-trial and per-cell
numbers; set `CHARTREUX_PERF_REPORT=<path>` to append it to a file.
