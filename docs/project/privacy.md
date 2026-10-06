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

## Usage ledger

The append-only ledger at `$CHARTREUX_HOME/usage/<root-session-id>/usage.jsonl`
retains content-free per-call accounting: schema and record IDs, UTC completion
timestamps, root/session/parent IDs, project identity, model/provider/wire name,
agent role and profile, purpose (conversation, compaction, title, or worktree
naming), outcome and usage state, reported input/output/cached token counts,
prices captured at call time, known cost, cost-completeness flags, and USD currency.
Missing token counts remain unknown rather than becoming reported zero.

Ledger records never contain prompts, messages, request or response headers,
credentials, or exception text. They have a lifetime independent of transcripts:
recording continues with session logging disabled, and deleting a saved session
does not delete its ledger. Pre-session naming records survive failed startup.
There is currently no automatic retention or compaction. Include the ledger when
considering local backups or removing accounting data; it retains call identities
and activity times even though it contains no conversation content.

Costs are USD estimates using catalog prices captured for each call, not provider
invoices or remote billing data. Historical costs are not repriced when the catalog
changes. Unreported usage, missing prices, and unreadable records can leave totals
incomplete; clients expose unknown-cost flags and coverage warnings. Reading usage
is local and does not contact a provider billing service.

## Network activity

Chartreux makes network requests only when a configured or requested feature needs one:

- model completions go to the selected provider API endpoint;
- remote MCP servers use their configured endpoint, while local `stdio` MCP servers are subprocesses;
- `web_search` contacts its configured search provider; and
- `web_fetch` retrieves the URL requested by the user or model and can therefore access arbitrary URLs.

MCP OAuth can also open a browser for the configured server's authorization flow. Tool permissions and workspace safety policy govern tool use, but they do not turn these network destinations into Chartreux telemetry. For transport and certificate details, see [Networking](../integrations/networking.md).

## Instruction-file access

Instruction files loaded into an agent's prompt, including user and trusted-project
`AGENTS.md` files, are sent to that agent's selected model provider as prompt
content. Subagents can also reread these exact injected files with `read_file`
or search them with `grep` targeting the exact file, even when the file is outside
workspace roots. A reread sees the file's current contents, not necessarily the
version originally injected.

This is an exact-file read capability, not access to the containing directory,
recursive search, shell commands, or writes. Tool denials, path denylists, and
sensitive-file protections still apply. Other out-of-workspace paths require an
explicit user scope change. Treat injected instructions as provider-visible data
and keep secrets out of them.

## Shell-output retention

Managed jobs capture merged stdout/stderr locally in a root-lifetime rolling
buffer: at most 64 records or 256 KiB per job, with at most 4096 UTF-8 bytes per
record and 32 retained job summaries. Finished records may be evicted; live owned
jobs are not evicted to make room. Buffers are released after successful root
cleanup and are not restored from disk or adopted after restart.

Registered credentials are redacted incrementally before output records enter
retention, with a further outward check on reads. Commands, labels, errors, and
child completion summaries are bounded and redacted too. Unresolved credential
candidates are masked conservatively; redaction and capture loss can reduce the
available output. These checks do not recognize every arbitrary project secret.
Credential-environment scrubbing and its configured passthrough policy still
apply to launched commands.

Job-state notifications carry no command or output bodies. Reading output with
`bash_read` does put the returned page into tool results and model-visible
history; session logging can persist it in transcripts. Replaying retained pages
can therefore grow transcript storage independently of the rolling-buffer bound.
Deleting or evicting a live-buffer record does not erase earlier transcript
copies, provider-visible reads, or a program's own files and network effects.
Foreground `bash` remains a separate finite-call output path. Treat both kinds
of shell output, transcripts, and logs as potentially sensitive.

## Local diagnostics

Bare `chartreux doctor` makes no network requests, launches no subprocesses,
and never reads `.env` or keyring credentials. Opt-in `--live` checks request
provider metadata and initialize/list tools from configured MCP servers;
`--smoke` sends billable synthetic inference probes to one selected deployment.
Both opt-in modes load the app's `.env` as normal startup does and may read
keyring credentials. Doctor does not write sessions or persist trust. See
[Troubleshooting](../guides/troubleshooting.md#run-diagnostics) for the limits
and OAuth exceptions.

Local session and diagnostic logs, plus token and cost accounting, remain available so you can inspect failures and usage without a hosted analytics service. Adjust logging through `log_level` in `config.toml` or the `/log-level` command. Treat logs and session transcripts as potentially sensitive project and prompt data when choosing backups, sharing diagnostics, or deleting local storage.

Registered credential values are redacted from tool events and diagnostic
feedback, and credential variables are scrubbed from child shell environments.
Fallback logging uses safe categories rather than credential-bearing exception
text. This is not a guarantee that arbitrary project or prompt secrets are
recognized; continue treating transcripts and logs as sensitive.

For the governing design decision, see [ADR 0008: Feature instrumentation](../adr/0008-feature-instrumentation.md).
