# Troubleshooting

## Run diagnostics

Start with local checks from the project directory:

```bash
chartreux doctor
```

This validates the catalog, trusted configuration, model resolution, credential
provenance, and MCP configuration without network requests, subprocesses, or
paid inference. It never reads `.env` or keyring credentials. A credential marked
`unverified` may still be available through one of those sources at startup.
Untrusted project configuration is ignored and reported; doctor does not grant
trust or repair configuration.

To check provider metadata and MCP runtime readiness, opt in explicitly:

```bash
chartreux doctor --live
```

This loads the app's `.env` as normal startup does and may resolve keyring
credentials. It makes provider metadata requests and initializes/lists tools
from enabled MCP servers, including launching configured stdio processes.
OAuth servers only have stored fingerprint/expiry checked: doctor does not log
in, refresh tokens, or list their tools. An empty MCP tool list is healthy;
unsupported provider model listing is `unverified`, not a failure.

Metadata listing does not establish inference capability. For a billable probe
of one configured deployment, see [Smoke-testing a deployment](models.md#smoke-testing-a-deployment).
`--live` and `--smoke` are independent; neither implies the other.

Use `--json` for machine-readable output. Exit codes are `0` for no failed
checks, `1` for failed checks, and `2` for an invalid invocation or ambiguous
smoke target. Exit `0` does not turn skipped, unsupported, or unverified checks
into passes. See the [doctor reference](../reference/commands.md#doctor) for
options and side effects.

## Check the local state

Chartreux keeps its local state under `~/.chartreux` by default (or the path in
`CHARTREUX_HOME`). Check `config.toml` for selections and
`~/.chartreux/.env` for provider credentials. Do not share the `.env` file or
its contents.

Structured diagnostic logs are written to:

```text
$CHARTREUX_HOME/logs/chartreux.log
```

The default level is `WARNING`. Set `LOG_LEVEL=DEBUG` for a diagnostic run, or
set `DEBUG_MODE=true` to force debug logging at startup. Valid levels are
`DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`; effective precedence is a
session override, then `DEBUG_MODE`/`LOG_LEVEL`, then `log_level` in
`config.toml`, then the default.

## Common failures

### Missing provider credential

Run `chartreux --setup` from an interactive terminal, or export `MISTRAL_API_KEY`
for the default provider before launching. A non-empty shell value takes precedence
over `~/.chartreux/.env`; an unset or empty shell value allows a non-empty `.env`
value to be used. Non-interactive and programmatic environments never launch the
onboarding TUI; setup prints guidance when an interactive terminal is unavailable.

### Project configuration is ignored

A project `.chartreux/config.toml` is loaded only from a trusted project. Read
the warning on standard error and rerun with `--trust` for a one-time trusted
invocation after verifying the project. See the [configuration guide](configuration.md).

### Invalid configuration or unexpected model selection

Check the reported field and configuration layer, then compare your settings
with the [configuration reference](../reference/configuration.md). Keep model
catalog definitions in `~/.chartreux/models.toml`, not `config.toml`; for a
legacy catalog, preview `chartreux models migrate` and apply it only when the
reported changes are correct.

### TLS or corporate-network failures

Chartreux uses Certifi roots by default. If your organization supplies a system
trust store, set `enable_system_trust_store = true`; `SSL_CERT_FILE` and
`SSL_CERT_DIR` add certificate anchors. See [networking](../integrations/networking.md)
for the full policy.

Chartreux does not send product analytics or OpenTelemetry telemetry. Local
logs, configured provider and MCP requests, and requested web operations remain
separate from that policy; see [privacy](../project/privacy.md).
