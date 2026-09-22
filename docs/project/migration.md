# Migrating from Mistral Vibe

Chartreux is an independent, local-first derivative of Mistral Vibe v2.25.5. It keeps the coding-agent workflow but deliberately narrows the hosted-product surface. This page describes the Chartreux-specific differences; consult the [changelog](changelog.md) for the release summary.

## Removed or no longer offered

- There is no Chartreux provider account flow or application-level sign-in/sign-out. MCP OAuth login and logout remain available for MCP servers.
- Configuration is file-based; Chartreux has no general settings UI.
- The Vertex provider and the dedicated reasoning adapter are removed. Use a supported Mistral, generic OpenAI-style, OpenAI Responses, or Anthropic-style provider definition instead.
- OpenRouter and OpenCode Zen presets are not shipped; the OpenCode Go preset remains available.
- Extra upstream themes are gone: use `auto`, `light`, or `dark`.
- The selectable built-in `explore` subagent is gone. `worker` is the neutral built-in subagent profile; `explore` remains a system-prompt ID, and local TOML agent profiles can still define a profile of that name.

## What changed

- **Credentials and onboarding.** Provider keys are supplied through environment variables or `$CHARTREUX_HOME/.env` (normally `~/.chartreux/.env`), rather than an OS-keychain onboarding flow. Setup creates a selections-only `config.toml`. A non-empty process environment value takes precedence; if it is unset or empty, Chartreux may load a non-empty `.env` value. The runtime keyring fallback is not an onboarding store.
- **Configuration and models.** `config.toml` selects models and ordinary settings. Provider and deployment definitions live in the shipped catalog plus the optional user `models.toml` overlay. Models can have ordered deployment failover; roles are ordered lists of canonical models that select an eligible model or explicitly fan out subagent work. See [Models](../guides/models.md) and the [configuration reference](../reference/configuration.md).
- **Subagents.** `task` runs asynchronously by default. Retained subagents can be checked, waited for, read, released, or retasked, and their transcripts can be browsed. Retention defaults to one hour, with up to 16 idle agents retained.
- **Tools and permissions.** The permission model was reworked around typed tools and server-side policy. Chartreux also adds provider-configurable `web_search` and vision-gated `read_image`; see [Tools and safety](../guides/tools-safety.md).
- **Delivery boundary.** The Textual CLI, ACP, and programmatic clients share the app-server harness and its typed public state rather than using separate live runtimes. See [ACP](../integrations/acp.md) and the [app server](../integrations/app-server.md).

## Configuration migration

1. Move provider credentials to the environment or `$CHARTREUX_HOME/.env`; do not depend on an interactive account session.
2. Put selections such as `active_model` in `config.toml`. Use `CHARTREUX_*` variables for environment overrides; nested fields use `__`.
3. Move legacy provider/model catalog tables out of `config.toml`. Run `chartreux models migrate` to preview, then `chartreux models migrate --apply` to create a backup and write `models.toml`.
4. Replace retired provider styles and presets with explicit supported provider definitions in `models.toml`. Retired configuration keys fail validation rather than being silently translated.
5. Replace retired `[tags]` tables and model `aliases` with roles. Define each role as `[roles.<name>]` with `description` and an ordered `models` list of canonical model names; select it with `@<name>`.
6. Recheck tool permissions and MCP configuration. Remote MCP transport is `streamable-http`, local process transport is `stdio`; the old `http` spelling is rejected.

Chartreux combines configuration in this order: defaults, user TOML, trusted project TOML, `CHARTREUX_*` environment values, active agent profile, then session overrides. Project configuration can select catalog entries but cannot define the catalog.
