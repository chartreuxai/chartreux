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

Provider names are the keys you choose (for example, `example-openai`); leading and trailing whitespace is trimmed, and `/`, `@`, control characters, and empty names are rejected. Names retain their spelling and case and must be unique. A provider’s `api_key_env_var` identifies its credential; see [Configuration](configuration.md) for `.env` and process-environment precedence.

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

The `@orchestrator` preset is the main assistant's saved default. Set it in
`models.toml`; there is no separate persisted active-model selection. Use
`/model` and `/thinking` for current-session overrides. `compaction_model`
accepts a canonical base name or role expression; an empty value uses the
current main model:

```toml
compaction_model = "example-model"
allowed_models = ["example-model"]
```

The shipped presets are `orchestrator` (the main assistant), `large`,
`medium`, and `small`. Built-in Worker and Reviewer profiles use `medium` by
default; Advisor uses `large`. The UI labels `orchestrator` as **Main**.

A role is one default preset: a canonical model and a thinking level. A role
expression such as `@medium` selects that exact pair. Different roles can use
the same model with different thinking levels. An unavailable model or
credential produces an explicit error; Chartreux does not select another
canonical model from the role. A session or retained child preserves its
committed model, provider, and thinking identity on resume. To launch multiple
independent agents, make separate `task` calls with explicit presets or models;
`fan_out: true` is rejected with guidance.

Shipped roles are the four presets above; they can be patched one role at a
time in `models.toml`. Custom roles are TOML-only: define `[roles.<name>]` with
`description`, `model`, and `thinking`; the preset editor selects built-in
roles. Old `models = [...]` role tables are rejected with a conversion hint.
See [Subagents](subagents.md).

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

For field definitions, merge behavior, and the full validation rules, see the [Configuration reference](../reference/configuration.md).
