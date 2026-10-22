# Migrating from Mistral Vibe

Chartreux is an independent, local-first derivative of Mistral Vibe v2.25.5. It keeps the coding-agent workflow but deliberately narrows the hosted-product surface. This page describes the Chartreux-specific differences; consult the [changelog](changelog.md) for the release summary.

## Removed or no longer offered

- There is no Chartreux provider account flow or application-level sign-in/sign-out. MCP OAuth login and logout remain available for MCP servers.
- Configuration remains file-based; `/settings` edits curated settings. Some keys, including `subagents.max_running_subagents`, require TOML or environment overrides.
- The Vertex provider and the dedicated reasoning adapter are removed. Use a supported Mistral, generic OpenAI-style, OpenAI Responses, or Anthropic-style provider definition instead.
- OpenRouter and OpenCode Zen presets are not shipped; the OpenCode Go preset remains available.
- Extra upstream themes are gone: use `auto`, `light`, or `dark`.
- The selectable built-in `explore` subagent is gone. The built-in identities are `coder` (the neutral default child identity), `orchestrator`, `researcher`, `reviewer`, and `advisor`; the retired `explore` and `tests` prompt IDs no longer resolve, and custom identities are prompt files in `~/.chartreux/prompts/`.

## What changed

- **Credentials and onboarding.** Provider keys are supplied through environment variables or `$CHARTREUX_HOME/.env` (normally `~/.chartreux/.env`), rather than an OS-keychain onboarding flow. Setup writes ordinary settings to `config.toml` and saved providers, models, and presets to `models.toml`. A non-empty process environment value takes precedence; if it is unset or empty, Chartreux may load a non-empty `.env` value. The runtime keyring fallback is not an onboarding store.
- **Configuration and models.** `config.toml` holds ordinary settings. Provider, deployment, and role presets live in the shipped catalog plus the optional user `models.toml` overlay. Each role is one model and thinking level; it does not route among models. A canonical model can still have multiple provider deployments under the existing deployment-selection rules. See [Models](../guides/models.md) and the [configuration reference](../reference/configuration.md).
- **Subagents.** `task` runs asynchronously by default and launches child identities under purpose routes. Retained subagents can be checked, waited for, read, released, or retasked, and their transcripts can be browsed. Retention is opt-in: agents close on completion unless retained (the global `[subagents].retain_agents` default or the per-launch `retain` argument), retained idle agents are kept for `idle_ttl_seconds` (default 300) with at most `max_idle_agents` (default 4) resident, and at most 16 subagent runs can be active at once.
- **Tools and permissions.** The permission model was reworked around typed tools and server-side policy. Chartreux also adds provider-configurable `web_search` and vision-gated `read_image`; see [Tools and safety](../guides/tools-safety.md).
- **Delivery boundary.** The Textual CLI, ACP, and programmatic clients share the app-server harness and its typed public state rather than using separate live runtimes. See [ACP](../integrations/acp.md) and [the app server](../integrations/app-server.md).
- **Shell policy breaking change.** The recognized dangerous git operations (`git push`, `git checkout`, `git restore`, `git stash drop`, `git stash clear`, `git switch --discard-changes`, `git switch -f`, `git reflog expire`, `git reflog delete`) left the shipped `[tools.bash]` denylist and are now governed by typed runtime approval: interactive sessions prompt the user (Deny default), noninteractive sessions deny with remediation, and only ordinary `git push` is persistable ("Always for this project"). Explicit denylist entries remain an absolute veto, and `denylist = []` no longer implicitly authorizes these commands — publication now requires a grant. Hard guards (`git reset --hard`, forced non-dry-run `git clean`, recursive `rm`) remain denied with no grant path. See [dangerous git commands](../guides/tools-safety.md#dangerous-git-commands).

## Configuration migration

1. Move provider credentials to the environment or `$CHARTREUX_HOME/.env`; do not depend on an interactive account session.
2. Set the main assistant default through `[roles.main]` in `models.toml`; `active_model` and persisted `thinking_overrides` in `config.toml` are rejected. Use `/model` and `/thinking` for session changes. Use `CHARTREUX_*` variables for other supported environment overrides; nested fields use `__`.
3. Move legacy provider/model catalog tables out of `config.toml`. Run `chartreux models migrate` to preview, then `chartreux models migrate --apply` to create a backup and write `models.toml`.
4. Replace retired provider styles and presets with explicit supported provider definitions in `models.toml`. Retired configuration keys fail validation rather than being silently translated.
5. Replace retired `[tags]` tables, model `aliases`, and ordered role `models` lists with presets. Define each role as `[roles.<name>]` with one `model` and `thinking`; select it with `@<name>`. Launch separate tasks for parallel work; `fan_out: true` is rejected.
6. Recheck tool permissions and MCP configuration. Remote MCP transport is `streamable-http`, local process transport is `stdio`; the old `http` spelling is rejected.

Chartreux combines configuration in this order: defaults, user TOML, trusted project TOML, `CHARTREUX_*` environment values, then session overrides; the `[agent]` identity/workflow/orchestration-strategy selections resolve CLI > environment > config > shipped default. Project configuration can select catalog entries but cannot define the catalog.
