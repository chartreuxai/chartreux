# Models

Chartreux separates model **selection** from the provider and deployment **catalog**. Select a model in `config.toml`; define or patch catalog entries only in the optional user `$CHARTREUX_HOME/models.toml` overlay.

## Providers

A provider describes an endpoint, credential variable, and protocol. The shipped catalog includes the Mistral backend and a local Codex Responses-compatible endpoint. Additional providers can use these supported protocol styles:

- Mistral (`backend = "mistral"`), using the OpenAI-style protocol surface.
- Codex or another Responses-compatible endpoint (`api_style = "openai-responses"`).
- A generic OpenAI-style endpoint (`api_style = "openai"`, `backend = "generic"`).
- An Anthropic-style endpoint (`api_style = "anthropic"`, `backend = "generic"`).

Provider IDs contain a slash, such as `example/openai`. The provider’s `api_key_env_var` identifies its credential; see [Configuration](configuration.md) for `.env` and process-environment precedence.

## Catalog overlays

The shipped catalog is the baseline. Create `$CHARTREUX_HOME/models.toml` to add providers, add models, or patch shipped entries. Omitted fields inherit from the shipped catalog; scalar values replace values, lists replace whole lists, and a deployment is matched by its base model and provider.

```toml
[providers."example/openai"]
api_base = "https://api.example.com/v1"
api_key_env_var = "EXAMPLE_API_KEY"
api_style = "openai"
backend = "generic"

[models."example-model"]
thinking = "medium"

[[models."example-model".deployments]]
provider = "example/openai"
name = "example-model-v1"
supports_images = false
supported_thinking_levels = ["low", "medium", "high"]

[roles.reviewers]
description = "Models used for independent review"
models = ["example-model"]
```

Provider definitions may supply extra headers and mark an endpoint as not emitting a finish reason. A model definition has semantic defaults and one or more deployments. A deployment identifies the provider-specific wire name and can declare image support, supported thinking levels, compaction threshold, and prices. Disable a retained provider, model, or deployment with `disabled = true`.

Legacy catalog tables in `config.toml` are rejected. Preview migration with `chartreux models migrate`; apply it with `chartreux models migrate --apply`.

## Linked deployments, priority, and failover

Give one base model deployments at more than one provider to represent the same model across endpoints. Their order is the provider-preference order. A completion remains committed to its selected base model and may fail over only to another eligible deployment of that same model after a transient failure.

Authentication, billing, quota, invalid-request, context, and cancellation failures do not fail over. Cooldowns are held in memory and respect longer retry hints. Automatic retry occurs only before any content, reasoning, or tool-call output is exposed; after partial output, use `/retry`.

## Selecting models and roles

`active_model` and `compaction_model` accept a canonical base name or a role expression such as `@reviewers`. An empty `active_model` selects `@orchestrator`:

```toml
active_model = "@reviewers"
compaction_model = "example-model"
allowed_models = ["example-model"]
```

A role is an ordered list of canonical base models with a description. Normal resolution chooses its first eligible member. For fan-out subagent work, `task(..., fan_out: true, config: {model: "@role"})` requires an explicit role, checks every member in advance, and returns results in member order without replacing or cancelling siblings. A session or retained child preserves its committed model/provider identity on resume instead of resolving the role again; changing a role does not change that committed identity when the agent is reused within a session; on cross-restart resume a role-bound agent re-resolves its role.

Built-in roles ship with the catalog and can be patched one role at a time in `models.toml`. Custom roles are TOML-only: define `[roles.<name>]` with `description` and `models`; v1 has no role-creation UI. The model edit screen lists every role as a checkbox. Existing members keep their order, while a newly checked model is appended; unchecking and rechecking a model therefore appends it to the end. See [Subagents](subagents.md).

## Thinking levels

Base-model definitions set a default `thinking` level, and deployments may restrict it with `supported_thinking_levels`. Available levels are `off`, `low`, `medium`, `high`, and `max`; a deployment’s supported list limits what Chartreux can encode, not what a remote provider will ultimately accept. Use `/thinking` during a session or `[thinking_overrides]` in `config.toml` to select an allowed level.

For field definitions, merge behavior, and the full validation rules, see the [Configuration reference](../reference/configuration.md).
