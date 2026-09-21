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

Run `chartreux --setup` to bootstrap setup and save a configured provider key in `$CHARTREUX_HOME/.env`, or create that file yourself:

```dotenv
MISTRAL_API_KEY=your-api-key
```

Credential precedence is precise: a **non-empty** value already present in the process environment wins. If that variable is unset or empty, Chartreux loads a non-empty value from `.env`. At runtime, API-key lookup falls back to the keyring only when no non-empty environment value is available. Keep `.env` private and do not commit it.

## Logging and diagnostics

Chartreux writes structured local logs to `$CHARTREUX_HOME/logs/chartreux.log`. Set `log_level` in `config.toml`, use `/log-level` for a session override or persisted setting, or set `LOG_LEVEL`. Valid levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`; the default is `WARNING`.

The effective level is chosen in this order: session override, `DEBUG_MODE=true` or a valid `LOG_LEVEL`, `log_level` in configuration, then the default. For local data and network-traffic policy, see [Privacy](../project/privacy.md). For the complete setting schema, see the [Configuration reference](../reference/configuration.md).
