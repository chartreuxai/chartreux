# Privacy and local data

Chartreux is a local-first harness. It does not create product analytics or OpenTelemetry spans, configure telemetry exporters, or send product telemetry to a remote service. The Mistral SDK client is explicitly configured with telemetry disabled.

This policy applies to Chartreux instrumentation. It does not mean that provider SDK dependencies contain no telemetry code, nor does it control network traffic initiated by extensions you configure.

## Local records

Chartreux stores data under `$CHARTREUX_HOME`, which defaults to `~/.chartreux`, and in trusted project storage where applicable. Local records can include:

- configuration and model selections in `config.toml`, the optional `models.toml` catalog overlay, and `.env` provider credentials;
- durable session metadata in `meta.json` and session transcripts in `messages.jsonl`, including tool activity and token/cost accounting;
- custom agents, prompts, skills, hooks, and other local extension configuration; and
- diagnostic logs, including `$CHARTREUX_HOME/logs/chartreux.log`.

Session history shown to clients is a public projection; private session files are read and written by the local app server. Session persistence supports resume and transcript inspection. See [Sessions and workspaces](../guides/sessions-workspaces.md) for operational details.

## Network activity

Chartreux makes network requests only when a configured or requested feature needs one:

- model completions go to the selected provider API endpoint;
- remote MCP servers use their configured endpoint, while local `stdio` MCP servers are subprocesses;
- `web_search` contacts its configured search provider; and
- `web_fetch` retrieves the URL requested by the user or model and can therefore access arbitrary URLs.

MCP OAuth can also open a browser for the configured server's authorization flow. Tool permissions and workspace safety policy govern tool use, but they do not turn these network destinations into Chartreux telemetry. For transport and certificate details, see [Networking](../integrations/networking.md).

## Local diagnostics

Local session and diagnostic logs, plus token and cost accounting, remain available so you can inspect failures and usage without a hosted analytics service. Adjust logging through `log_level` in `config.toml` or the `/log-level` command. Treat logs and session transcripts as potentially sensitive project and prompt data when choosing backups, sharing diagnostics, or deleting local storage.

For the governing design decision, see [ADR 0008: Feature instrumentation](../adr/0008-feature-instrumentation.md).
