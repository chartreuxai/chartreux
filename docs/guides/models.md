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
aliases = ["example"]
thinking = "medium"

[[models."example-model".deployments]]
provider = "example/openai"
name = "example-model-v1"
supports_images = false
supported_thinking_levels = ["low", "medium", "high"]

[tags]
reviewers = ["example-model"]
```

Provider definitions may supply extra headers and mark an endpoint as not emitting a finish reason. A model definition has semantic defaults and one or more deployments. A deployment identifies the provider-specific wire name and can declare image support, supported thinking levels, compaction threshold, and prices. Disable a retained provider, model, or deployment with `disabled = true`.

Legacy catalog tables in `config.toml` are rejected. Preview migration with `chartreux models migrate`; apply it with `chartreux models migrate --apply`.

## Linked deployments, priority, and failover

Give one base model deployments at more than one provider to represent the same model across endpoints. Their order is the provider-preference order. A completion remains committed to its selected base model and may fail over only to another eligible deployment of that same model after a transient failure.

Authentication, billing, quota, invalid-request, context, and cancellation failures do not fail over. Cooldowns are held in memory and respect longer retry hints. Automatic retry occurs only before any content, reasoning, or tool-call output is exposed; after partial output, use `/retry`.

## Selecting models, tags, and roles

`active_model` and `compaction_model` accept a canonical base name, a unique alias, or a tag expression such as `@reviewers`:

```toml
active_model = "example"
compaction_model = "@reviewers"
allowed_models = ["example-model"]
```

A tag is an ordered list of base models. It chooses an eligible member when assigned. For fan-out subagent work, `task(..., fan_out: true, config: {model: "@tag"})` requires an explicit tag, checks every member in advance, and returns results in member order without replacing or cancelling siblings. A session or retained child preserves its committed model/provider identity on resume instead of resolving the tag again. See [Subagents](subagents.md).

## Thinking levels

Base-model definitions set a default `thinking` level, and deployments may restrict it with `supported_thinking_levels`. Available levels are `off`, `low`, `medium`, `high`, and `max`; a deployment’s supported list limits what Chartreux can encode, not what a remote provider will ultimately accept. Use `/thinking` during a session or `[thinking_overrides]` in `config.toml` to select an allowed level.

For field definitions, merge behavior, and the full validation rules, see the [Configuration reference](../reference/configuration.md).
