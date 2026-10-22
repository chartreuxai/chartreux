# Models

Chartreux separates current-session model choices from the saved provider,
deployment, and role-preset catalog. Define or patch saved entries in the
optional user `$CHARTREUX_HOME/models.toml` overlay; use `/model` or
`/thinking` for a session choice.

## Providers

A provider describes an endpoint, credential variable, and protocol. The shipped catalog is neutral and publicly reachable only: it defines the Mistral public provider (`mistral`) and models Mistral actually serves, such as `glm-5-3`. Personal setups — local proxies, LAN endpoints, private model pins — belong in your `models.toml` overlay, not in the shipped catalog. Additional providers can use these supported protocol styles:

- Mistral (`backend = "mistral"`), using the OpenAI-style protocol surface.
- Codex or another Responses-compatible endpoint (`api_style = "openai-responses"`).
- A generic OpenAI-style endpoint (`api_style = "openai"`, `backend = "generic"`).
- An Anthropic-style endpoint (`api_style = "anthropic"`, `backend = "generic"`).

Provider names are the keys you choose (for example, `example-openai`); leading and trailing whitespace is trimmed, and `/`, `@`, control characters, and empty names are rejected. Names retain their spelling and case and must be unique. A provider’s `api_key_env_var` identifies its credential; see [Configuration](configuration.md) for `.env` and process-environment precedence. If the selected provider's credential is missing, startup names that provider and variable. Repair it with `chartreux --setup` or **Settings > Providers** (`/providers`); another provider's credential does not change routing.

## Catalog overlays

The shipped catalog is the baseline. Create `$CHARTREUX_HOME/models.toml` to add providers, add models, or patch shipped entries. Omitted fields inherit from the shipped catalog; scalar values replace values, lists replace whole lists, and a deployment is matched by its base model and provider.

```toml
[providers."example-openai"]
api_base = "https://api.example.com/v1"
api_key_env_var = "EXAMPLE_API_KEY"
api_style = "openai"
backend = "generic"

[models."example-model"]
thinking = "medium"

[[models."example-model".deployments]]
provider = "example-openai"
name = "example-model-v1"
supports_images = false
supported_thinking_levels = ["low", "medium", "high"]

[roles.custom-review]
description = "Default for independent review"
model = "example-model"
thinking = "high"
```

Provider definitions may supply extra headers and mark an endpoint as not emitting a finish reason. A model definition has semantic defaults and one or more deployments. A deployment identifies the provider-specific wire name and can declare image support, supported thinking levels, compaction threshold, and prices. Disable a retained provider, model, or deployment with `disabled = true`.

A deployment's `auto_compact_threshold` must be a positive whole integer token
count; fractional values and `0` are rejected. If omitted, it uses the global
`auto_compact_threshold` fallback in `config.toml`, where `0` disables automatic
compaction. The TUI context denominator shows this effective threshold, not the
model's maximum context window.

Legacy catalog tables in `config.toml` are rejected. Preview migration with `chartreux models migrate`; apply it with `chartreux models migrate --apply`.

## Smoke-testing a deployment

After configuring a provider and model, use doctor to test inference capability:

```bash
chartreux doctor --smoke --provider example-openai --model example-model
```

Replace the example names with your provider ID and canonical base-model name,
not the deployment's wire name or a role expression. `--provider` and/or
`--model` must select exactly one enabled deployment; ambiguous targets are
rejected with candidates. This explicit diagnostic selection bypasses
`allowed_models` by design.

**Smoke probes send billable inference requests.** They load the app's `.env`
as normal startup does and may resolve keyring credentials. The probe uses
synthetic prompts, a synthetic tool that is never executed, and an in-memory
image fixture, not project files. It tests the selected deployment directly,
without retries, failover, or saving catalog changes or sessions. It does not
list provider metadata or launch MCP servers unless `--live` is also supplied.

Each capability (`tool`, `thinking`, and `image`) has a separate verdict:

| Verdict | Meaning |
| --- | --- |
| `pass` | The expected tool call or image answer was observed, or thinking returned observable reasoning. |
| `fail` | The request failed, timed out, was truncated, or did not return the expected tool/image result. |
| `unsupported` | The deployment declares no image support, or no non-off thinking level can be encoded within its declared supported levels. That capability is not requested. |
| `unverified` | The thinking request was accepted without observable reasoning; opaque/encrypted reasoning alone does not verify it. |

Image and thinking checks respect the deployment's declared capabilities; a
smoke result does not discover or update them. A successful metadata listing
with `chartreux doctor --live` is not evidence that these inference capabilities
work. See [Troubleshooting](troubleshooting.md#run-diagnostics) for local and
live checks, JSON output, and exit codes.

## Linked deployments, priority, and failover

Give one base model deployments at more than one provider to represent the same model across endpoints. Their order is the provider-preference order. A completion remains committed to its selected base model and may fail over only to another eligible deployment of that same model after a transient failure.

Authentication, billing, quota, invalid-request, context, and cancellation failures do not fail over. Cooldowns are held in memory and respect longer retry hints. Automatic retry occurs only before any content, reasoning, or tool-call output is exposed; after partial output, use `/retry`.

## Selecting models and roles

The `@main` preset is the main assistant's saved default. Set it in
`models.toml`; there is no separate persisted active-model selection. Use
`/model` and `/thinking` for current-session overrides; both leave the saved
`@main` preset unchanged. `compaction_model` accepts a canonical base name or
role expression; an empty value uses the current main model:

```toml
compaction_model = "example-model"
allowed_models = ["example-model"]
```

The shipped presets are `main` (the main assistant), `worker`, `scout`,
and `heavy`. The presets screen labels them **Main assistant (@main)**, **Worker
(@worker)**, **Scout (@scout)**, and **Heavy (@heavy)**.

A role is one default preset: a canonical model and a thinking level. A role
expression such as `@worker` selects that exact pair, and the `task` tool's
top-level `model` field accepts one — use it to aim a delegation at a preset.
Different roles can use the same model with different thinking levels. Roles are
model/thinking presets used as launch targets; they carry no agent behavior —
the identity does that (see [Identity
architecture](../reference/identity-architecture.md)). An unavailable model or
credential produces an explicit error; Chartreux does not select another
canonical model from the role. A session or retained child preserves its
committed model, provider, and thinking identity on resume. To launch multiple
independent agents, make separate `task` calls with explicit presets, purposes,
or identities; `fan_out: true` is rejected with guidance. See
[Subagents](subagents.md).

Shipped roles are the four presets above; they can be patched one role at a
time in `models.toml`. Custom roles are TOML-only: define `[roles.<name>]` with
`description`, `model`, and `thinking`; the preset editor selects built-in
roles. Old `models = [...]` role tables are rejected with a conversion hint.

Orchestration is not configured through the catalog. Delegated routing lives in
`[orchestration.routes.*]` in `config.toml`, and who performs the work is
selected by the `[agent]` identity/workflow/orchestration_strategy selections —
see [Configuration](configuration.md) and the [identity
architecture](../reference/identity-architecture.md).

## The presets screen

The presets screen (Provider Settings, or the **Customize role presets** step of
setup) edits role presets: one row per role showing its model and thinking
level, with a per-role editor for both fields. It offers no dispatch modes —
there is nothing to switch, because routing is configured by purpose routes and
agent selections, not by the catalog. Explicit existing selections are
preserved.

Setup seeds the presets automatically: usable existing role selections are
kept, and unavailable bindings are assigned a ready configured model. No
per-role questions are required. **Finish setup** saves the seeded presets and
completes setup; **Customize role presets** opens the model/thinking editor for
optional changes or repairs. Invalid or unavailable bindings block completion
with repair guidance.

After two deduplicated, budget-exceeded implementation attempts fail on the same
task, a passive graduation nudge at an idle boundary suggests reviewing model
presets or delegating differently. It opens the presets screen, never changes
anything automatically, and can be dismissed permanently; headless sessions do
not show it. Saving a second canonical model no longer triggers it —
orchestration is never inferred from the roster size.

## Thinking levels

Base-model definitions set a default `thinking` level for direct model
selection; role presets each carry their own level. Deployments may restrict
levels with `supported_thinking_levels`. Available levels are `off`, `low`,
`medium`, `high`, and `max`; a deployment's supported list limits what
Chartreux can encode, not what a remote provider will ultimately accept.
Use `/thinking` during a session or an explicit spawn `config.thinking` to
override the selected preset in that scope. Persisted
`[thinking_overrides]` in user or project `config.toml` are rejected; set the
desired role's `thinking` in `models.toml` instead.

## Upgrading from 0.5.6

Chartreux is pre-release alpha; breaking changes may occur without migration
paths. Every retirement below fails loudly with an actionable error — nothing
is silently migrated, and no compatibility aliases are kept:

- **The `@orchestrator` model role is `@main`.** Selecting `@orchestrator` fails
  with rename guidance and no alias, and a `[roles.orchestrator]` overlay over
  the shipped base fails loudly at load — rename the table to `[roles.main]`;
  the catalog file is never rewritten automatically. `active_model` in
  `config.toml` is rejected with guidance to edit the `[roles.main]` preset.
  Sessions persisted with `active_model = "@orchestrator"` fail resume with the
  same guidance.
- **The `[dispatch]` catalog overlay is superseded.** It no longer appears in the
  catalog template or the presets editor; orchestration is selected through the
  `[agent]` identity/workflow/orchestration_strategy selections and configured
  through `[orchestration.routes.*]`. A legacy `[dispatch]` table in
  `models.toml` is reported loudly at load with successor guidance — it is never
  merged — and catalog writes preserve its values (comments and formatting are
  not preserved), never rewriting it.
- **The former thinking-tier roles** `small`, `medium`, `large`, and `peer` are
  no longer shipped; the current model-tier roles are `main`, `worker`,
  `scout`, and `heavy`. Review every `[roles.<name>]` entry in an existing
  overlay after upgrading:
  - A sparse patch of a removed role loses the shipped fields it inherited and
    fails validation. Replace it with a complete role definition including
    `description`, `model`, and `thinking`, or bind it to an available shipped
    role.
  - Complete definitions of old roles remain as custom roles; they are not
    deleted. Rename or remove them yourself if you no longer want them.

For field definitions, merge behavior, and the full validation rules, see the [Configuration reference](../reference/configuration.md).
