# App server

The app server is Chartreux's serialized harness boundary. It owns live
sessions, runtime configuration, tools, persistence access, and cleanup, while
delivery surfaces such as the terminal client, ACP, and programmatic mode act as
clients. For the architectural decision, see [ADR 0009](../adr/0009-app-server-boundary.md).

## Connecting

Run `chartreux-app-server` with no arguments. It speaks newline-delimited
JSON-RPC on standard input and output; stdout is reserved for protocol messages.
A client initializes the connection, sends `initialized`, then starts, resumes,
or continues a session. The protocol also has an in-process serialized transport;
there is no HTTP listener or multi-client network service exposed by this
entrypoint.

External Python clients should use the public `chartreux.app_server` package,
not private server modules. Its exported surface is:

- `AppServerHost` for passive pre-session operations and opening sessions;
- `AppServerSession` for an attached session, turns, resources, and live events;
- `ClientToolHandler` for client-hosted operations explicitly advertised by a
  client;
- `AppServerConnectionClosed` for attached-session connection closure; and
- `SessionExitSummary` for shutdown presentation.

## RPC surface

`chartreux.app_server.protocol.SERVER_METHODS` is the authoritative current RPC
method list, including typed parameter and result models. It covers:

- initialization, identity, runtime/configuration, policy roots, diagnostics,
  statistics, and event reads;
- session creation, continuation, resume, reading, history, fork, rename,
  compact, stop, delete, rewind, settings, shell commands, and turns;
- queued-turn enqueue/read/remove/replace/resume operations;
- agent transcripts, skills, tools, scheduled loops, reviews, and workspace
  trust and Git worktree operations; and
- MCP catalog read/add/remove/toggle/refresh/login/logout operations.

## Usage accounting

`usage/read` is a host-level method available after initialization, before session
attachment and during active turns. It does not create a session or require an
idle turn. Parameters are `window` (`day`, `week`, or `month`, default `day`) and
nullable `projectKey` (default `null`, meaning all projects). Reads reconcile local
ledger changes. Wire field names use camelCase.

```json
{"jsonrpc":"2.0","id":12,"method":"usage/read","params":{"window":"day","projectKey":null}}
```

The result contains:

| Field | Contract |
| --- | --- |
| `asOf`, `revision` | One aware timestamp and non-negative revision shared by all summaries and detail. |
| `summaries` | `day`, `week`, and `month` totals, all subject to the request's project filter. |
| `window`, `models`, `components` | Selected window, rows grouped by `(model, provider, wireName)`, and `uncachedInput`/`cachedInput`/`output` breakdown. |
| `projectKey` | Attached root's resolved project identity, not the request filter; `null` before attachment. Use it for a later Current project read. |
| `warnings` | Coverage warnings with `code` and nullable `rootSessionId`/`recordId`, never paths or exception text. Codes: `write-failed`, `unreadable`, `malformed-record`, `torn-tail`, `unsupported-schema`. |

Each window summary has `state` (`loading`, `ready`, or `unavailable`), `requests`,
`inputTokens`, `outputTokens`, `cachedInputTokens`, `hasUnknownTokens`,
`knownCostUsd`, `hasKnownCost`, and `hasUnknownCost`. It also has aware
`startLocal`/`endLocal` and `startUtc`/`endUtc` boundaries, `timezone`, and
`currency` (`USD`). Windows are half-open `[start, end)`, using the system-local
calendar day, Monday-start week, and calendar month. Calls count at completion.

Model rows carry the same totals plus `model`, `provider`, and `wireName`.
Components carry `tokens`, `hasUnknownTokens`, `knownCostUsd`, `hasKnownCost`,
and `hasUnknownCost`. Cached input is a subset of summary/model input counts;
component `uncachedInput` excludes it. Known amounts and token counts are lower
bounds when their unknown flags are set. A ready empty window is zero; unpriced
calls are not known zero. Costs use captured catalog rates, not provider invoices.

For example, this is the `summaries.day` portion of a ready response with one
partially priced call (the full result also includes week/month and detail):

```json
{
  "state": "ready", "requests": 1,
  "inputTokens": 1000, "outputTokens": 200, "cachedInputTokens": 0,
  "hasUnknownTokens": false,
  "knownCostUsd": 0.01, "hasKnownCost": true, "hasUnknownCost": true,
  "startLocal": "2026-10-01T00:00:00+00:00",
  "endLocal": "2026-10-02T00:00:00+00:00",
  "startUtc": "2026-10-01T00:00:00Z",
  "endUtc": "2026-10-02T00:00:00Z",
  "timezone": "UTC", "currency": "USD"
}
```

`usage/updated` is a coalesced host-level notification containing only `asOf`,
`revision`, and `summaries`. Its day/week/month summaries are always global,
regardless of the last read's filter. It is independent of session event sequencing
and has no `sessionId` or event ID. Initialized unattached clients can receive it
unless notifications are disabled. Updates follow local appends, reconciliation,
and calendar rollover; they are not a per-call event stream. Clients deduplicate
by revision and read afresh after reconnecting. To discover external-process
appends, issue `usage/read` on open/refresh; the TUI also polls about every 60
seconds while spend segments are enabled.

For example, a notification envelope uses the same three summary objects as a
read response. In this Python example, `response` is a `UsageReadResponse`:

```python
notification = {
    "jsonrpc": "2.0",
    "method": "usage/updated",
    "params": {
        "asOf": response.as_of.isoformat(),
        "revision": response.revision,
        "summaries": response.summaries.model_dump(mode="json", by_alias=True),
    },
}
```

Build this example from an unfiltered response, not a Current project read.
`stats/read` and `session/statsUpdated` remain session statistics; the ledger
contracts do not change their conversation-scoped accounting.

## Steering during subagent waits

`PublicTurn.waiting_only` identifies an active turn whose remaining work is only
subagent waits. Clients can submit `turn/steer` with `requireWaitingOnly: true`,
the `expectedTurnId`, and a nonempty `idempotencyKey`. The server rechecks the
live turn and admission fences rather than trusting the client's snapshot.
Accepted steering cancels the waits, not child runs, and keeps the current turn
so the model receives the input immediately.

Retry the same payload with the same key to recover its delivery receipt without
injecting it twice; reusing a key with a different payload is rejected. Receipts
are bounded and runtime-local, not durable recovery records. A transport failure
or expired receipt is not proof of nondelivery: clients must preserve uncertainty
instead of automatically queueing a possible duplicate.

## Background run cancellation

`agents/cancel` accepts `{"agentId": "…", "runId": "…"}`. `runId` may be
omitted to snapshot the active run; interactive confirmations should pin it to
avoid stopping a newer run. `AppServerSession.cancel_agent(agent_id, run_id)`
returns a typed public response with `outcome`, nullable `runId`, and nullable
`stopReason`.

The outcomes are `stop_requested`, `already_stopping`, `already_finishing`,
`not_running`, `unknown_run`, and `forbidden`. Acceptance requests a stop; it
does not confirm terminal cancellation. Observe `agents/update` for settlement.
The winning stop reason is first-writer-wins, so a racing orchestrator stop may
retain its reason rather than `user_cancelled`.

This user-authorized operation stops only the selected background run, not the
root turn or siblings. It remains available during root lifecycle work while
attachment and shutdown fences still apply. Retained agent identity, transcript,
and partial results are preserved subject to retention and release policy.
User-cancelled completion is injected into the parent: an idle parent learns at
its next turn and is never auto-started. Root interruption remains the separate
`turn/interrupt` operation.

## Settings projection and writes

`config/settings/read` returns the curated settings view and the current user
configuration revision. Its optional `web_search` projection contains only
supported search-setting leaves, with effective and saved values, their
origins, selected credential-variable names, readiness, and safe repair
metadata. It never returns a resolved API-key value. Readiness reflects local
configuration and credential availability; it is not a provider connectivity
probe.

Clients save web-search settings through `config/write` with the user target
and the revision from the read. The server validates the complete candidate
search configuration before applying the write. The response reports
persistence and runtime application separately, so a client can explain a
successful save whose runtime reload failed and retry application without
writing the settings again. Credential persistence remains a separate client
action and is not part of the settings projection.

The server emits typed live notifications and can send typed callbacks or
advertised `clientTool/*` requests to a capable client. It does not offer a
generic command-execution RPC: clients map their own presentation commands to
specific typed methods.

## Limits

One app-server instance owns one attached root runtime and currently does not
model several simultaneous attached observers of that runtime. `events/read`
returns an empty batch in v0.1; clients consume live notifications and recover
state with a session read when needed. Restarting a stdio server ends its live
runtime and discards in-flight work and queued turns; persisted sessions can be
resumed normally. Public protocol views are projections, not the private session
format, and do not expose credentials or internal runtime objects.
