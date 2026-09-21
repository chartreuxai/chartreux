# Troubleshooting

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
