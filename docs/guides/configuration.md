# Configuration

## Configuration layers

Chartreux resolves normal settings in this order, from lower to higher precedence:

1. Built-in defaults.
2. User configuration: `$CHARTREUX_HOME/config.toml` (normally `~/.chartreux/config.toml`).
3. Trusted project configuration: `.chartreux/config.toml`, discovered by searching upward from the working directory.
4. `CHARTREUX_*` configuration environment variables.
5. The active agent profile.
6. Runtime overrides.

An untrusted project configuration is not loaded. Project configuration selects catalog entries but cannot define providers or deployments; those belong in the user model catalog. See [Models](models.md).

`CHARTREUX_` setting names are case-insensitive and use `__` for nesting, for example:

```bash
export CHARTREUX_PROJECT_CONTEXT__TIMEOUT_SECONDS=5
```

Configuration environment variables do not discard empty values: a valid empty string or empty list remains a setting, while an invalid empty number, boolean, or structured value is an error. `CHARTREUX_HOME` is a separate location control, not a configuration-field prefix.

## User files and custom home

By default, user-owned files are under `~/.chartreux`:

- `config.toml` holds selections and other general settings.
- `models.toml` is the optional overlay for the shipped model catalog.
- `.env` can hold provider API keys.
- `logs/` holds local logs.

Set `CHARTREUX_HOME` before starting Chartreux to use another home directory:

```bash
export CHARTREUX_HOME="/path/to/chartreux-home"
```

This changes the locations of the files above as well as user prompts, agent profiles, and related user data.

## API keys and `.env`

A provider declares the name of its credential variable. For example, the shipped Mistral provider uses `MISTRAL_API_KEY`:

```bash
export MISTRAL_API_KEY="your-api-key"
```

Run `chartreux --setup` from an interactive terminal to configure a provider,
credentials, and models, then choose the default presets. It uses the theme
already configured for Chartreux (`auto` by default); setup does not ask for or
save a theme choice. Provider and model screens show explicit Save and Continue
actions, so advancing does not require returning to an earlier screen. After
presets, setup checks whether web search is already ready. If Mistral's
automatic search or an explicit saved search provider is ready, it preserves
that choice and skips the Web search step. Otherwise the step offers Exa,
Brave, and DuckDuckGo; it does not offer `auto` or a second Mistral choice.
Standalone Settings presents one Mistral choice, while `auto` remains an
accepted Mistral configuration alias. Readiness uses the configured credential
variable through the credential resolver; a provider or model merely appearing
in the catalog does not establish readiness. **Save and finish** saves and
applies edited search settings, then completes only when the resulting
configuration and any required key are ready; otherwise setup stays open with
a repair cue. **Finish setup** completes when the current configuration and any
required key are ready. **Skip for now** leaves web-search settings unchanged and does not
disable the tool.
**Back to presets** returns to the preset editor. Unsaved search edits require
an explicit discard before leaving, while an API key already saved separately
remains saved. Setup saves configured provider keys in `$CHARTREUX_HOME/.env`;
if disk persistence fails, it reports when a key is available only for the
current session.
Without an interactive terminal, setup prints actionable guidance instead of
launching the TUI. You can also create that file yourself:

```dotenv
MISTRAL_API_KEY=your-api-key
```

Credential precedence is precise: a **non-empty** value already present in the process environment wins. If that variable is unset or empty, Chartreux loads a non-empty value from `.env`. At runtime, API-key lookup falls back to the keyring only when no non-empty environment value is available. Setup and the Web Search editor save keys to `.env` when possible, with a session-only outcome if disk persistence fails; they do not save onboarding credentials to the keyring. The keyring is only a runtime lookup fallback. Keep `.env` private and do not commit it.

For web search, use Settings > Web search or `/web-search` to select a provider and manage its credential separately from search settings. A key may be saved or kept for the current session; see [Tools and safety](tools-safety.md) for provider, readiness, and retry behavior.

## Logging and diagnostics

Chartreux writes structured local logs to `$CHARTREUX_HOME/logs/chartreux.log`. Set `log_level` in `config.toml`, use `/log-level` for a session override or persisted setting, or set `LOG_LEVEL`. Valid levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`; the default is `WARNING`.

The effective level is chosen in this order: session override, `DEBUG_MODE=true` or a valid `LOG_LEVEL`, `log_level` in configuration, then the default. For local data and network-traffic policy, see [Privacy](../project/privacy.md). For the complete setting schema, see the [Configuration reference](../reference/configuration.md).
