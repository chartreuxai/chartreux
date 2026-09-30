# Provider Settings mockups

Status: Revised concept — implementation and live screenshots are tracked in
[PLAN.md](PLAN.md)

These repo-native SVG frames illustrate the forward setup journey in
[PROVIDER_SETTINGS_SPEC.md](PROVIDER_SETTINGS_SPEC.md). They are design
artifacts, not captured application output. Model identifiers and endpoints
are illustrative; no credential value is shown.

First launch shows Welcome and then opens Providers directly. Onboarding uses
the configured theme without asking for a new choice or saving a theme value;
`auto` remains the default.

## Frames

| Step | Frame | Decision shown |
| --- | --- | --- |
| Connect provider | [Connection](docs/assets/provider-settings/connection-forward.svg) | Save a valid provider and advance; credential save is separate. |
| Configure models | [Models and next action](docs/assets/provider-settings/models-next.svg) | Save and add another provider, or save and continue to presets. |
| Edit model detail | [Model detail at 80×24](docs/assets/provider-settings/model-detail-current.svg) | Eight arrow-reachable rows; model Default thinking is distinct from role preset thinking. |
| Choose default presets | [Presets at 80×24, dark](docs/assets/provider-settings/presets-compact.svg) | Main, Large, Medium, and Small each have one model and thinking level. |
| Edit a preset | [Preset at 80×48, light](docs/assets/provider-settings/presets-tall-light.svg) | Two presets can share a model with different thinking levels. |
| Finish needs repair | [Direct repair](docs/assets/provider-settings/finish-repair.svg) | Saved providers remain saved; a missing credential identifies the affected preset and repair route. |

The first provider's Models screen is a branch point. **Save and add another
provider** opens a fresh Connection screen. **Save and continue to presets**
opens the preset list. Neither action requires a return to the root browser.
Management can still open provider, model catalog, and preset editors directly.

Each screen has one current row, a visible primary action, and an advertised
Esc destination. Up/Down can reach fields, rows, and actions; Enter edits or
activates; Space toggles applicable choices; Tab is optional. Text-editing
arrows stay inside an open text field. Esc closes one local layer and restores
its opener's selection, focus, filter, and scroll.

Saving a provider or model requires structural validity only. The model detail
has six scalar rows, Image support, and Save model edits; its Default thinking
belongs to direct model selection, while each role's thinking is chosen in
Default presets. Finish checks
that required presets are runnable. A role is a model plus thinking level;
there is no membership list, model fallback, or separate persisted Active
picker. `/model` and `/thinking` are session choices and appear separately
from the saved `@orchestrator` default.

The older role-membership and Active-picker SVGs in this directory are
historical design explorations. The linked frames above are the current
concept. Live 80×24 and 80×48 captures in light and dark themes remain the
implementation acceptance evidence, as specified in [PLAN.md](PLAN.md).
