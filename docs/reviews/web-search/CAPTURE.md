# Web search settings capture

`capture_web_search.py` exercises `WebSearchScreen` in Textual's real Pilot at
80×24 and 80×48 with ANSI dark and light terminal palettes. It renders an SVG
from each live screen and converts that SVG to PNG with Inkscape. The manifest
at `evidence/captures.jsonl` records the fixture, route actions, viewport,
palette, focused widget, visible state flags, fake call counts, and artifact
paths. Each preparation asserts its expected state before capture.

Run the full finite matrix from the repository root:

```sh
.venv/bin/python docs/reviews/web-search/capture_web_search.py
```

To refresh one state, for example:

```sh
.venv/bin/python docs/reviews/web-search/capture_web_search.py \
  --fixture dirty-back --size 80x24 --theme ansi-dark
```

The harness imports `tests/cli/textual_ui/web_search_fixture.py` and uses only
in-memory settings and credential services. The key is synthetic, its input is
masked, and the script rejects an SVG that contains the synthetic key value.
Manifest metadata records only whether the key input is nonempty, never its
value. No network, real credential store, or user config file is accessed.

The host is an isolated `WebSearchHarness` that pushes the production modal.
It preserves the modal's own layout, focus, and actions, but does not show the
underlying chat transcript. The source commit in metadata is the repository
HEAD at capture time; uncommitted source changes are not represented by that
hash. Captures exercise ANSI dark and light only. They do not validate a real
terminal's monochrome mode or external provider availability.

`provisional/` contains early layout previews. The original
`evidence/captures.jsonl` retains the accepted 18-state, 72-image review
matrix from before the provider-choice policy change. The additional retry
states distinguish a
successful runtime reload followed by a failed settings read from a successful
runtime reload followed by a failed UI refresh.

Historical onboarding evidence is separate at
`evidence/onboarding/captures.jsonl`:
nine Web search decisions and one non-Mistral preset suggestion, each at the
same four viewport/palette combinations (40 SVG and 40 PNG images). Reproduce
them with:

```sh
.venv/bin/python docs/reviews/web-search/capture_onboarding.py
.venv/bin/python docs/reviews/web-search/capture_preset_suggestion.py
```

The onboarding Web search host pushes the production modal with fake services.
Its manifest records asserted `finish`, `skip`, and `back` results after the
pictured state is captured. It does not render the surrounding provider
workbench after returning `back`; the application host loop is covered by its
declared UI tests. The preset suggestion uses the real provider workbench in
an in-memory test host with a configured custom model and synthetic credential
resolver. All capture processes redirect `CHARTREUX_HOME` to a fresh temporary
directory before importing application code.

The current provider-choice policy is captured in
`evidence/choice-policy/captures.jsonl` (five states, 20 SVG and 20 PNG
images). It shows a no-Mistral-key fallback with Exa, Brave, and DuckDuckGo
only and no selected alternative; an explicit Exa selection; ready
DuckDuckGo; the single Mistral radio used for raw Automatic in standalone
settings; and the generic Save presets and continue row. The manifests
record raw and visible provider selection separately, so a focused cursor
does not imply a selected radio. Run the selected fixtures with `--output
docs/reviews/web-search/evidence/choice-policy` and the fixture names in
that manifest. The first-run host bypass for an already-ready Mistral setup
has no Web search modal to photograph; its behavior is verified by the
onboarding tests.
