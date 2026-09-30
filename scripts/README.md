# Project Management Scripts

This directory contains scripts that support import-time correctness checks and startup-cost analysis.

## Import checks

Run both before merging any `TYPE_CHECKING` / lazy-import change.

### `check_import_contracts.py` — runtime cross-file gate

```bash
uv run python scripts/check_import_contracts.py
```

Imports every `from <mod> import <name>` across `chartreux/` and `tests/` and checks that each import resolves at runtime, including fallback module resolution for missing attributes. It also rebuilds Pydantic models to catch field types that fail lazily. This complements Ruff's static, per-file `TC004` check rather than duplicating it. Missing non-chartreux deps are non-blocking warnings.

### `suggest_lazy_imports.py` — informational

```bash
uv run scripts/suggest_lazy_imports.py          # flat listing
uv run scripts/suggest_lazy_imports.py --stats  # per-rule counts
uv run scripts/suggest_lazy_imports.py --tree   # directory tree
uv run scripts/suggest_lazy_imports.py --check  # CI gate (exit 1 on findings)
```

Reports deferral candidates: `TC001`–`TC003` (annotation-only) and `[lazy]` (single-function heuristic). Not gated.

## Import Analysis

`check_startup_import_cost.py` builds the `chartreux` wheel, installs it into a
fresh venv, and reports cold import cost for each target declared in
`startup_import_cost.chartreux.toml`:

- wall time and total imported module count,
- the slowest modules by self time (via `python -X importtime`),
- file-operation call count under `strace` (Linux only; skipped elsewhere),
- installed wheel size.

Each command may carry an optional `budget`. Commands without a budget are
measured and reported but never fail the run, so a config can ship budget-free
and be filled in from a baseline run (observed count + ~10% headroom). Once a
`budget` is set, exceeding it exits non-zero. The shipped `startup_import_cost.chartreux.toml`
already carries baselined budgets, so a regression overshoot fails the step.

### Usage

```bash
# Run the measurement (enforces budgets when set; exits non-zero on overshoot)
uv run scripts/check_startup_import_cost.py

# Override the project or config
uv run scripts/check_startup_import_cost.py --project chartreux --config path/to/config.toml
```

## TUI vision captures

`capture_tui.py` renders deterministic provider, settings, session-picker, and
model-picker surfaces with Textual's test pilot. It uses the existing fake
fixtures, so it does not start Chartreux's app-server or contact an LLM or
network service. The `chat` surface uses a fake backend and the snapshot app's
in-process test app-server to render a deterministic conversation.

SVG files are always written. If `inkscape` is installed, matching PNG files
are written beside them. The default sizes are `80x24`, `120x36`, and
`120x72`; pass `--size` more than once to choose sizes. For a compact ASCII
capture:

```sh
env -u NO_COLOR .venv/bin/python scripts/capture_tui.py \
  --surface all --size 80x24 --theme textual-dark --ascii
```

Use `--surface provider|settings|session|model|chat`, `--theme textual-dark|textual-light|ansi-dark|ansi-light`, and `--output PATH` as needed. `NO_COLOR=1` disables color for monochrome captures; leave it unset for normal color output.

Capture the full fake-backend chat surface with:

```sh
env -u NO_COLOR .venv/bin/python scripts/capture_tui.py \
  --surface chat --size 120x36 --theme textual-dark
```

The chat capture uses `SnapshotTestAppWithConversation`, fake credentials, and
the test harness file manager. It submits one prompt and waits 0.4 seconds for
the deterministic fake response; it never contacts a real model or network.
The full chat is bottom-anchored in tall viewports by design, so blank space
above the conversation is expected.
