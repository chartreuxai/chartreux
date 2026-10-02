# Migrating from Mistral Vibe

Chartreux is an independent, local-first derivative of Mistral Vibe v2.25.5. It keeps the coding-agent workflow but deliberately narrows the hosted-product surface. This page describes the Chartreux-specific differences; consult the [changelog](changelog.md) for the release summary.

## Removed or no longer offered

- There is no Chartreux provider account flow or application-level sign-in/sign-out. MCP OAuth login and logout remain available for MCP servers.
- Configuration remains file-based; `/settings` edits curated settings. Some keys, including `subagents.max_running_subagents`, require TOML or environment overrides.
- The Vertex provider and the dedicated reasoning adapter are removed. Use a supported Mistral, generic OpenAI-style, OpenAI Responses, or Anthropic-style provider definition instead.
- OpenRouter and OpenCode Zen presets are not shipped; the OpenCode Go preset remains available.
- Extra upstream themes are gone: use `auto`, `light`, or `dark`.
- The selectable built-in `explore` subagent is gone. `worker` is the neutral built-in subagent profile; `explore` remains a system-prompt ID, and local TOML agent profiles can still define a profile of that name.

## What changed

- **Credentials and onboarding.** Provider keys are supplied through environment variables or `$CHARTREUX_HOME/.env` (normally `~/.chartreux/.env`), rather than an OS-keychain onboarding flow. Setup writes ordinary settings to `config.toml` and saved providers, models, and presets to `models.toml`. A non-empty process environment value takes precedence; if it is unset or empty, Chartreux may load a non-empty `.env` value. The runtime keyring fallback is not an onboarding store.
- **Configuration and models.** `config.toml` holds ordinary settings. Provider, deployment, and role presets live in the shipped catalog plus the optional user `models.toml` overlay. Each role is one model and thinking level; it does not route among models. A canonical model can still have multiple provider deployments under the existing deployment-selection rules. See [Models](../guides/models.md) and the [configuration reference](../reference/configuration.md).
- **Subagents.** `task` runs asynchronously by default. Retained subagents can be checked, waited for, read, released, or retasked, and their transcripts can be browsed. Retention defaults to one hour, with up to 16 idle agents retained.
- **Tools and permissions.** The permission model was reworked around typed tools and server-side policy. Chartreux also adds provider-configurable `web_search` and vision-gated `read_image`; see [Tools and safety](../guides/tools-safety.md).
- **Delivery boundary.** The Textual CLI, ACP, and programmatic clients share the app-server harness and its typed public state rather than using separate live runtimes. See [ACP](../integrations/acp.md) and the [app server](../integrations/app-server.md).

## Configuration migration

1. Move provider credentials to the environment or `$CHARTREUX_HOME/.env`; do not depend on an interactive account session.
2. Set the main assistant default through `[roles.orchestrator]` in `models.toml`; `active_model` and persisted `thinking_overrides` in `config.toml` are rejected. Use `/model` and `/thinking` for session changes. Use `CHARTREUX_*` variables for other supported environment overrides; nested fields use `__`.
3. Move legacy provider/model catalog tables out of `config.toml`. Run `chartreux models migrate` to preview, then `chartreux models migrate --apply` to create a backup and write `models.toml`.
4. Replace retired provider styles and presets with explicit supported provider definitions in `models.toml`. Retired configuration keys fail validation rather than being silently translated.
5. Replace retired `[tags]` tables, model `aliases`, and ordered role `models` lists with presets. Define each role as `[roles.<name>]` with one `model` and `thinking`; select it with `@<name>`. Launch separate tasks for parallel work; `fan_out: true` is rejected.
6. Recheck tool permissions and MCP configuration. Remote MCP transport is `streamable-http`, local process transport is `stdio`; the old `http` spelling is rejected.

Chartreux combines configuration in this order: defaults, user TOML, trusted project TOML, `CHARTREUX_*` environment values, active agent profile, then session overrides. Project configuration can select catalog entries but cannot define the catalog.
