# Post-fix TUI captures

`captures.jsonl` indexes 210 SVG/PNG pairs from 51 selected fixture states.
All 51 states were captured at 80×24 and 80×48 in ANSI dark and light. Three
provider states also have ASCII dark captures at both sizes. Every indexed PNG
is a valid Textual/Inkscape render. These files are separate from the frozen
`evidence/live/` baseline.

The manifest's `source_commit` is the repository HEAD, not a commit containing
the uncommitted TUI repairs. The captures reflect the working tree at the time
each fixture was run; preparation steps and resulting focus/selection state are
recorded per row. Re-run a selected fixture from the repository root with:

```sh
UV_CACHE_DIR=/tmp/chartreux-review-uv-cache uv run python docs/reviews/tui-affordances/capture_review.py --output docs/reviews/tui-affordances/evidence/after --fixture settings-confirm
```

The `provider-onboarding-model-detail` fixture sets the in-memory workbench to
the onboarding models stage before opening a real model detail view. It verifies
the stage-specific help copy, not the full add-provider transition. The
`question-other-text` fixture is a standalone widget host; use
`question-full-host` for production shell geometry. Browser OAuth completion,
real credential or proxy writes, and actual terminal NO_COLOR contrast remain
outside these offline captures. No-color SVG/PNG export is invalid for visual
contrast, as documented in `../../CAPTURE.md`.
