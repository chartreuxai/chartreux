# Offline TUI capture fixtures

`capture_review.py` runs real Textual widgets and `ChartreuxApp` shells against
in-process fake backends, MCP registries, and settings services. It reads the
existing snapshot fixtures and `scripts/capture_tui.py` without changing their
tests or baseline snapshots. Inkscape converts Textual's SVG output to PNG.

From the repository root:

```sh
UV_CACHE_DIR=/tmp/chartreux-review-uv-cache uv run python docs/reviews/tui-affordances/capture_review.py
```

The default matrix is 80×24 and 80×48 in `ansi-dark` and `ansi-light`, which
are Chartreux's explicit dark and light themes. Narrow captures can select a
fixture, family, size, or theme:

```sh
UV_CACHE_DIR=/tmp/chartreux-review-uv-cache uv run python docs/reviews/tui-affordances/capture_review.py --fixture oauth-open-failed --size 80x24 --theme ansi-dark
UV_CACHE_DIR=/tmp/chartreux-review-uv-cache uv run python docs/reviews/tui-affordances/capture_review.py --fixture settings-confirm --size 80x24 --theme ansi-dark --ascii --no-color
```

`--no-color` sets `NO_COLOR=1` before Rich or Textual imports. All other runs
remove an inherited `NO_COLOR` first. `--ascii` sets Chartreux's ASCII chrome
flag. The manifest records the effective environment value. The explicit
`ansi-light` app theme is exported with Textual's light terminal palette;
Rich otherwise substitutes its own dark SVG palette. The manifest records the
palette's RGB foreground and background separately from the app theme.

Rich's SVG path currently paints `NO_COLOR` widget cells black-on-black,
including after passing the correct terminal palette to the exporter. These
monochrome PNG images are marked `png_visual_valid: false` in the manifest and must
not be used to judge runtime contrast. Each monochrome capture also writes a
plain `.txt` file from the running Textual compositor; use that for control
text and verify contrast in an actual PTY before closing any no-color finding.

Evidence is written to `evidence/live/` as paired SVG and PNG files. The
`captures.jsonl` manifest records the commit, route, fixture, viewport, theme,
terminal mode, source fixture, host context, and preparation actions for each
PNG. Repeating a capture refreshes its evidence and provenance record. A
successful PNG is rendered by Textual and converted by Inkscape; SVGs are not
hand-authored mockups.

Fixtures whose `host_context` is `full ChartreuxApp shell` use the production
transcript and bottom-app container geometry. Other fixtures are standalone
widget hosts and should be used for local affordance content and focus review,
not production vertical placement. In particular, `question-single` is a
standalone snapshot host; use `question-full-host` for production placement.
Some states are synthetic by design: `oauth-open-failed` patches
`webbrowser.open` to return `False` while the fake login yields an
`example.invalid` URL, and `proxy-error` submits an invalid fake URL. No
credentials, external services, or browser are needed.

For live interaction, import `fixtures()` and `capture()` from this module or
reuse the fixture factories under `scripts/capture_tui.py`. In a `run_test`
pilot, call `pilot.click(widget, offset=(x, y))` for the center, text, and
padding cells separately, then inspect `app.screen`, `app.focused`, the
highlighted option, and draft/service state after `pilot.pause()`. Keep mouse
outcomes in the interaction ledger, separate from the blind screenshot reads.
The harness's preparation actions are setup steps, not behavior-oracle results.

This seed does not certify complete route or state coverage. The surface ledger
and evidence index must reconcile every source route and meaningful state.
The current fork has no routine tool/edit execution approval dialog: the old
approval snapshot names refer to effect-result renderers, reviewed here as
transcript content. A QuestionApp request is captured separately as an actual
decision panel.

The finite capture seed still leaves distinct behavior for later direct
evidence: first-run welcome/repair, OAuth external-browser outcome on an actual
desktop, submitted attachment rendering, session preview and history load-more,
agent task transitions beyond the browser/details states, update notices, and
mouse padding on every control. Existing baseline snapshots cover some adjacent
content but cannot establish those transitions. NO_COLOR text is captured from
the real compositor, while visual contrast remains unverified in an actual
terminal because the SVG exporter renders it black-on-black. These omissions
must remain open in the surface ledger; matching sizes/themes for current
fixtures do not close them.
