# Provider Settings

Status: Reviewed design target

Provider Settings (`/providers`) connects providers, configures their models,
and chooses default presets. This is the target interaction contract; an
in-progress screen may differ until implementation is complete. Follow the
visual rules in [DESIGN.md](DESIGN.md) and catalog semantics in the
[Models guide](docs/guides/models.md).

## Concepts

- A **provider** identifies an endpoint, protocol, and credential reference.
- A **canonical model** is the catalog identity; a **deployment** is that
  model's provider-specific wire name and capabilities.
- A **default preset** is a named role with exactly one canonical model and one
  thinking level. Distinct presets may use the same model at different levels.
- **Configured** means a definition is structurally valid and saved.
  **Runnable** also requires an enabled deployment and usable credential.
- `@orchestrator` is the one saved default for the main assistant. `/model`
  and `/thinking` can override it for the current session without changing it.

## First-run journey

First launch opens **Welcome**, then continues directly to **Providers**. The
setup flow applies the theme already configured for Chartreux; it does not ask
for or persist a theme choice. The default theme remains `auto`.

**Welcome → Connect provider → Configure models → Add another provider or continue →
Choose default presets → [Web search when needed] → Finish.**

Each screen presents a focused, visible next action. A user advances by saving
the current step; revisiting an earlier screen is for changing an earlier
choice. Provider and model saves accept structurally valid definitions even
when credentials or presets are incomplete. After presets, setup checks the
effective Web search readiness. When Mistral automatic search is ready, or an
explicit saved search choice is already ready, keep that choice and skip the
editor. Readiness uses credential resolution, including the provider's
configured credential variable; catalog presence alone is insufficient. If
the editor is needed, it offers Exa, Brave, and DuckDuckGo, not `auto` or
Mistral fallback choices. **Skip for now** leaves existing search settings and
tool enablement unchanged. The editor's **Finish setup** action requires valid
web-search configuration and any required key; this is a configuration check,
not a live connection test. Finish also validates required presets against
configured, runnable deployments and supported thinking levels. Its failure
message names the affected preset and opens a direct repair route.

| Screen | Primary action | Other forward action | Esc returns to |
| --- | --- | --- | --- |
| Connect provider | Save and configure models | — | Its opener |
| Configure models | Save and continue to presets | Save and add another provider | Its opener |
| Default presets | Save presets and continue | Edit a named preset | Its opener |
| Web search (when needed) | Finish setup or Save and finish | Back to presets; Skip for now | Presets |
| Model detail | Save model edits | — | Model list or catalog opener |

After **Save and add another provider**, the next Connection screen opens
immediately. After **Save and continue to presets**, the preset editor opens
immediately. A saved first provider remains saved while a second is configured.
The root browser remains a management entry and orientation point; setup does
not require returning to it between steps.

## Connection and credentials

Connection edits provider name, endpoint, protocol, and credential reference.
Existing provider names are read-only; new names are editable. Credential save
is immediate and separate from catalog edits. A nonempty process environment
credential takes precedence; the screen reports whether a key was saved
durably or is available for this session. No secret value appears in
validation text. **Save and configure models** validates the provider's
structure, persists it, then advances. A provider may be saved with a missing
credential, but required presets using it need repair before Finish.

## Model configuration

The provider Models screen lists canonical models and deployments with clear
enabled, selected, and runnable states. Space toggles an applicable checkbox;
Enter opens detail or activates the focused action. Discovery distinguishes
loading, no results, and failure and offers retry or manual configuration.
Model detail keeps canonical and wire IDs read-only. Its eight navigation
rows are Default thinking, Temperature, Auto compact threshold, Input price,
Output price, Cached input price, Image support, and Save model edits. Model
Default thinking applies to direct model selection; each role's thinking is
chosen separately in the preset editor. Roles are assigned in the preset
editor, not by model checkboxes.
**Save model edits** records the local detail and returns to its opener. The
two forward actions on Models commit structurally valid provider/model changes
and open their stated destinations.

The global Model catalog remains available in management mode for inspecting
and editing canonical models across providers. Its provider and model filters
are independent. Opening and closing detail restores the selected object,
filter, scroll, and focus. Pending catalog changes are visibly marked; Apply
persists them, and Discard names its exact scope. A saved credential is outside
that discard scope.

## Presets and main assistant

Show the four built-in presets as **Main**, **Large**, **Medium**, and
**Small**. Their role keys are `orchestrator`, `large`, `medium`, and `small`.
Worker and Reviewer agent profiles use `medium` by default; Advisor uses
`large`. Show every preset as a named choice with a model and thinking level.
Editing one preset selects one configured canonical model and a thinking level
the chosen deployment can encode. A credential-missing or disabled model may
be shown with a precise reason; it never silently substitutes a different
canonical model. Custom role definitions remain available through
`models.toml`; the UI must not imply custom role creation if it cannot do so.

Keep each already runnable preset unchanged during onboarding. If a preset is
not runnable, suggest a ready model from configured providers with an enabled
deployment. Read credential readiness with the credential resolver, including
the configured credential-variable name, and suggest only a thinking level the
deployment supports. This is a preset draft; it becomes the saved default only
when the user saves the presets. Do not treat catalog presence alone as
readiness or suggest an unconfigured catalog model as ready.

The orchestrator preset is the main assistant's saved default. There is no
separate Active picker or persisted active-model field. The current session's
model and thinking, if overridden, are labelled separately from the saved
default. Explicit spawn `config.thinking` overrides its role preset for that
launch. A retained child keeps its committed model and thinking pair; changing
a preset does not silently retarget that child. Deployment choice within one
canonical model follows the catalog's existing selection rules.

The optional Web search step reuses the settings editor. After presets, keep
ready automatic Mistral search and skip the editor. Otherwise, preserve an
already-ready explicit search choice and skip it. If neither is ready, show
the editor with Exa, Brave, and DuckDuckGo; the onboarding provider list does
not contain `auto` or Mistral. In standalone Settings, show Mistral once;
retain `auto` as a configuration alias for Mistral rather than a second
provider choice. **Save API key** is separate from search-setting saves; a key
saved there survives discarding a settings draft. **Save and finish** persists
and applies changed search settings before completing setup when they are
ready; **Finish setup** completes without a write when current settings are
ready. **Skip for now** leaves search settings and tool enablement unchanged.
**Back to presets** returns to the preset editor. Going back or skipping with
unsaved search edits or a key still in the input requires choosing **Keep
editing** or **Discard edits**; an already saved key is outside that discard
scope.

Finish validates required presets as runnable. If a preset is incomplete,
references an unknown or disabled model, needs a credential, or chooses an
unsupported thinking level, show the exact issue and focus a repair action.
Earlier valid provider/model saves stay saved. Old `models = [...]` role tables,
old persisted Active values, and `fan_out: true` receive actionable errors
instead of a migration or implicit fallback.

The Web search step uses the dedicated search-settings editor after presets.
**Save API key** is independent from **Save and finish**; a key saved in this
step survives discarding the search-settings draft. **Finish setup** completes
without changing search settings when their current configuration and any
required key are ready. **Skip for now** leaves search settings and tool
enablement unchanged. **Back to presets** returns to preset selection. If the
search draft or an unsaved key input is present, leaving requires choosing
**Keep editing** or **Discard edits**; a previously saved key is preserved.
Search readiness checks configuration and key availability only, not provider
connectivity.

## Keyboard and transitions

- Up/Down move through the visible rows, fields, and actions from the first
  focused control. Tab and Shift-Tab are optional alternatives, never the only
  route between a filter, list, form field, and action.
- Enter opens or activates the selected control. In a text field, arrow keys
  edit the text and Enter accepts the field, returning to screen navigation.
- Space toggles a focused checkbox or selects a radio value. Cursor movement
  alone changes no draft or saved value.
- Esc closes the local editor, confirmation, help, filter, or nested view one
  level at a time and returns to the exact opener. It restores the opener's
  focus, row, filter, and scroll. A dirty local editor offers scoped Save,
  Continue editing, or Discard; a clean one closes directly.

The current screen names its primary action and Esc destination. No save
requires an Escape/Tab detour. A busy atomic save keeps the draft and tells
the user to wait; an error preserves their work and offers a direct retry.
Stale discovery results cannot replace newer edits.

## Geometry and acceptance

At 80×24, show title, current step, at least five list rows when five exist,
the focused action, and a bounded help/status footer without clipping. At
80×48, use the same reading order with more visible rows. Long validation and
detail text remains reachable by keyboard. Light, dark, no-color, and
ASCII-border rendering retain meaning through words and selection markers.

- [ ] One provider can reach Finish by forward actions only; a saved key and
  provider remain saved after visiting later steps.
- [ ] Two providers can be connected consecutively without revisiting the
  first; Models shows **Add another provider** and **Continue to presets**.
- [ ] Provider/model save accepts partial preset readiness; Finish names and
  routes to repair for each required preset failure.
- [ ] After presets, onboarding opens Web search with **Back to presets**,
  **Skip for now**, and finish actions; Skip leaves saved search settings and
  tool enablement unchanged.
- [ ] Ready automatic or explicit search settings are preserved without
  opening the onboarding editor; a shown editor offers only Exa, Brave, and
  DuckDuckGo. Standalone Settings shows Mistral once while accepting `auto` as
  its configuration alias.
- [ ] Unready current preset suggestions use only enabled, runnable models
  from configured providers; saving the preset step persists the suggestion.
- [ ] Presets store one model and thinking level, including different levels
  for two roles using one model; `@orchestrator` is the sole saved main default.
- [ ] Arrow-only navigation reaches all controls; Enter, Space, Esc, and
  optional Tab obey the contract in both modes.
- [ ] Saved state, draft state, selected row, and current-session overrides
  are distinct in 80×24 and 80×48 light/dark captures.
