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

## Process topology

An interactive run is three process classes: the terminal client, one thin
app-server coordinator, and one agent process per agent — the root agent plus
each retained subagent identity. They speak the same typed JSON-RPC over stdio,
so a delivery surface sees one connection while agent work happens outside its
event loop.

| Process | Owns | Must never own |
| --- | --- | --- |
| Client (terminal, ACP, programmatic) | Presentation, input history, a bounded disposable backfill cache | Tools, permission evaluation, transcripts, approval authority, credentials on the wire |
| Coordinator (app server) | User input ingress, relay with per-producer sequences and incarnation IDs, process registry and liveness, approval-prompt routing, paged-read proxying, secret-free usage aggregation | Tools, permission policy or enforcement, transcript writes, provider calls, compaction, MCP, credentials, approval futures |
| Root agent process | Root loop, turn controller, projector, tool batch runner, permission policy and grant store, root transcript and saves, provider calls, compaction, MCP, hooks, subagent admission and supervision, config writes | UI state |
| Subagent process | Its own loop, tools, permission stack, transcript under `<root>/agents/<agent_id>/`, projection, provider calls, MCP clients, compaction | Root history, other children's transcripts |

A subagent transcript is scoped to the root session; only root sessions are
independently resumable.

### Transport selection

`CHARTREUX_TRANSPORT` accepts two selectors:

- `stdio` selects the multi-process topology and is the interactive default.
- `in_process` is the escape: the same components run in one process with every
  process boundary removed.

Omitting the variable leaves an interactive run on `stdio` and a programmatic
(`-p`/`--prompt`) run in-process. The config key `multi_process = false`
selects the same escape as `in_process`, and an explicit environment selector
outranks it. The legacy `multi_process` spelling is rejected as unusable;
diagnostics name the supported selectors and never echo the supplied value.

### Spawn and readiness

The coordinator spawns each agent process and reads exactly one
newline-delimited readiness frame carrying `chartreuxBootstrap` before any
JSON-RPC frame flows. The installing protocol version must match; EOF, a
foreign frame, or a version mismatch is a typed failure with no retry
negotiation, and the whole handshake is one startup deadline on a monotonic
clock. Physical spawns draw on a process-tree budget: an exhausted budget is a
typed spawn denial surfaced to the user, and the coordinator never queues a
spawn silently.

### Credentials

Each agent process reads credentials at spawn from the standard local sources
(environment and dotenv), exactly as the single-process route does. The
coordinator never reads credential variables, and no provider key is sent over
the wire. A key changed mid-session reaches agents spawned afterwards; a
running agent keeps its spawn-time value.

### Presentation loss

Presentation frames carry a per-producer sequence and an opaque incarnation ID.
A detected shed, sequence gap, presentation buffer or outbox overflow, or
snapshot capture or delivery failure invalidates that generation at the
detecting hop. The client shows transient stale or refreshing state, then
installs a fresh producer snapshot atomically at its final page, or the stream
ends in a persistent typed failure after the bounded re-registration attempt
limit. Frames from a retired incarnation are discarded, and a notice for an
already-retired incarnation cannot affect its replacement. Presentation
refresh alone never cancels agent compute, approvals, or results.

### Shutdown

Process-tree shutdown is ordered under one 10-second bound: stop admission,
interrupt, drain saves, send SIGTERM then SIGKILL, close IPC, and reap. No new
tree starts before the old tree is reaped, and a tree that cannot be reaped
inside the bound reports a typed timeout rather than success.

### Persistent artifacts

The topology adds exactly two persistent artifacts:

- The **registry manifest** (`<session>.registry.json`) sits beside the session
  directory, never inside it, because the root agent is the fenced directory's
  only writer. It holds live routes, callbacks, and policy transaction records
  — no credentials and no pending approvals — and it is disposable: a restart
  rebuilds it.
- The **writer-fence lockfile** (`.writer-fence.lock`) lives inside the fenced
  session directory and serializes that session's writer.

## RPC surface

`chartreux.app_server.protocol.SERVER_METHODS` is the authoritative current RPC
method list, including typed parameter and result models. It covers:

- initialization, identity, runtime/configuration, policy roots, diagnostics,
  statistics, and event reads;
- session creation, continuation, resume, reading, history, fork, rename,
  compact, stop, delete, rewind, settings, shell commands, and turns;
- dangerous-command authority inspection and revocation
  (`policy/dangerous_commands/read` and `policy/dangerous_commands/revoke`);
- queued-turn enqueue/read/remove/replace/resume operations;
- subagent resume and effect reconciliation (`agents/resume`, `agents/reconcile`);
- root automatic-recovery scheduling/admission/cancellation
  (`recovery/root/schedule`, `recovery/root/admit`, `recovery/root/cancel`);
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

## Background agent release

`agents/release` accepts `{"agentId": "…", "expectedRunId": "…"}`.
`expectedRunId` may be omitted to release whatever run is current; interactive
confirmations should pin it so a newer assignment is not released by mistake.
`AppServerSession.release_agent(agent_id, expected_run_id)` returns a typed
public response with `outcome` and a nullable `reason`.

The outcomes are `released` (a resident record was released), `evicted` (a
terminal-metadata record was removed), `conflict` (the agent is active,
reserved for replacement, or its latest run changed — nothing was released), and
`unknown` (no such agent — nothing was released). `conflict` and `unknown` never
stopped or purged anything; the reason explains which guard fired.

Release is a guarded, user-authorized removal. The registry verifies
attached-root and current-generation authority, the expected latest-run
identity, a non-active state, and the absence of a replacement reservation
atomically before any destructive work, so a stale idle-to-running race reports
`conflict` instead of stopping the new run. A successful release discards the
agent's retained results, expiry markers, and wait leases, and ends its browser
and transcript access; observe `agents/update` for the resulting inventory.
Unlike `agents/cancel`, release gets no lifecycle-transition exemption: it is
rejected while a root lifecycle transition is active, while attachment and
shutdown fences still apply. The model tool `release_agent(agent_id)` stays
unguarded and may still release an active instance.

## Agent inventory updates

`agents/update` is the server's inventory event: it carries one
`AgentSummaryModel` per visible agent plus any evictions, and each summary
describes that agent's LATEST run. Two fields added in this release carry the
run-collection and audit facts, and the stamp is what lets a client show a
completed verdict without claiming more than the audit says:

| Field | Contract |
| --- | --- |
| `completionAudit` | The typed completion-audit stamp of the latest run, resolved against that run's retained result. `null` when the latest run has no retained result, has not finished, or its result expired, so a previous run's stamp never rides a new run's summary. A stamp means the claim matched evidence; it never means the task is correct, and an unsupported audit version is neutral rather than a verdict. |
| `uncollectedRunIds` | Always present, `[]` included: the runs whose results are still retrievable but were never returned by a successful collection call, newest first. `get_agent_result` returning a result and a `wait_for_agent` that returns one mark a run collected; `check_agents`, transcript reads, a still-running wait, and a cancelled or errored wait do not. Because the key's presence is part of the contract, a client must not read an empty list as an absent field. |

Deliveries are serialized and coalesced: at most one `agents/update` is in
flight per registry, the payload is projected immediately before it is sent
rather than from an older snapshot, and a caller that arrives during a suspended
delivery queues its eviction data for the next pass. A client can apply each
payload as the current inventory, and `check_agents` stays the authoritative
read for an agent that has just changed.

### Child callback closure

`callback/childClosed` (`sessionId`, `callbackId`, `reason`) reports that a
callback opened by a CHILD session reached a terminal state. Child callbacks
never enter the root projection, so their history entries stay in the child
session and this event is the parent client's only route to the closure: on
receipt the client dismisses the question widget it opened for that
`callbackId`. Ids are matched exactly on `callbackId` — a root callback id, or a
closure naming a different owner, is ignored, a duplicate closure is a no-op,
and a late closure never dismisses a newer widget or clears its input.

### Next-request thinking estimate

The effective-thinking estimate rides the configuration view, not the inventory:
`ConfigView.effectiveThinking` carries `requested`, `semantic`, `apiEffort`,
`known`, and `matchesRequested`, computed server-side from the committed
deployment and the message context the next request resolves against. When
`known` is false a client shows `unknown` instead of implying that the requested
level is effective, and when `matchesRequested` is false it should show both
values with the second labeled as the API effort. The estimate describes the
next request only: an in-flight call keeps the encoding it started with, and a
failover may pick another deployment.

## Subagent resume and reconciliation

A recoverably failed background agent is suspended (warm, non-terminal; see
[Subagents](../guides/subagents.md#failure-recovery-suspension-and-resume)).
Two root-session operations operate on it; the automatic recovery controller
and the model tools both go through the same `agents/resume` admission, so
they cannot create duplicate successors.

`agents/resume` accepts
`{"agentId": "…", "expectedRunId": "…", "idempotencyKey": "…"}`.
`expectedRunId` pins the failed run to resume from; `idempotencyKey`
single-flights concurrent calls. `AppServerSession.resume_agent(agent_id,
expected_run_id, idempotency_key)` returns a typed public response with
`outcome`, `agentId`, nullable `runId`, `successorRunId`,
`recoveryOperationId`, `retryAt`, `eligibility`, `reasonCodes`, and
`sameAgentResumesUsed`.

The outcomes are `scheduled` (a successor run of the SAME agent identity was
admitted; `successorRunId` is published on admission), `already_scheduled`
(same key, same run: the pending admission is returned),
`already_resumed` (a different key while an admission is in flight: refused —
exactly one successor), `reconcile_required` (unknown tool effects block
replay), `expired` (the warm suspension window closed),
`not_resumable` (closed, evicted, released, or otherwise ineligible), and
`forbidden` (authority). The failed run's result is immutable and stays
retrievable; waits pinned to it return ITS failed result.

`agents/reconcile` accepts
`{"agentId": "…", "expectedRunId": "…", "transcriptBoundary": "…", "verifiedEffects": [{"toolCallId": "…", "disposition": "settled"|"unrepeatable_damage", "note": "…"}]}`.
It is pinned to the failed run and its transcript boundary, so stale or
duplicate evidence from an earlier run cannot clear a later run's barrier.
`AppServerSession.reconcile_result(...)` returns a typed public response with
`outcome`, `agentId`, nullable `runId` and `transcriptBoundary`,
`eligibility`, and `reasonCodes`.

The outcomes are `cleared` (the barrier flips to eligible), `prohibited`
(verification shows unrepeatable damage, with reason codes),
`stale`/`duplicate` (never change the barrier), `not_pending` (no barrier),
and `forbidden` (authority). The barrier never auto-clears; replacement with
`task(agent_id=..., replace_run=True)` remains permitted at any point. A run
that left no tool call outstanding submits `verifiedEffects: []` — the
complete zero-entry accounting — which clears the barrier exactly when zero
tool calls are outstanding.

Both operations are rejected while attachment and shutdown fences apply, and
release, cancellation, a session change, or shutdown invalidate pending
recovery without discarding the failed run's evidence.

## Root turn recovery

`recovery/root/schedule`, `recovery/root/admit`, and `recovery/root/cancel`
are the wire surface of the bounded automatic root-turn recovery gate (the
TUI's `RootRecoveryController` is the only production caller). They are
root-session operations: `sessionId` must be the attached root session.

- `recovery/root/schedule` accepts
  `{"sessionId": "…", "terminalTurnId": "…", "solicited": false, "inputGeneration": 0}`.
  The gate decides from the typed terminal projection of `terminalTurnId`:
  `scheduled` (with `episodeId`, `idempotencyKey`, `attempt`, `limit`,
  `delaySeconds`, `eligibleClass`, `affectedDeployment`),
  `already_scheduled`, or `declined` (with `reason`).
- `recovery/root/admit` accepts
  `{"sessionId": "…", "episodeId": "…", "idempotencyKey": "…", "terminalTurnId": "…", "inputGeneration": 0}`
  and is the single-flight admission: `admitted` (counted once per episode),
  `already_admitted` (an ambiguous redelivery reconciles to the original
  successor via the idempotency key — never a second one), `deferred` (a
  read-only availability wait with `delaySeconds`; it never consumes a
  retry), or `declined` (with `reason`). Every decision rechecks the session,
  turn, input generation, queued work, callbacks, stop state, budget,
  deadline, and live availability.
- `recovery/root/cancel` accepts `{"sessionId": "…", "episodeId": "…", "reason": "…"}`
  and cancels a pending schedule; it never cancels an already-admitted
  successor in flight (that settles via normal interruption rules).

The gate owns authoritative eligibility, episode accounting, and
single-flight admission; the caller owns the local timer and local
cancellation. Milestone notices (`RootRecoveryNoticeDetail`, kind
`root_recovery`) ride the normal history as `PublicNoticeEntry` records with
deterministic ids. See
[Automatic root recovery](../guides/configuration.md#automatic-root-recovery)
for the policy.

## Managed-job state

`PublicSessionState.active_background_job_count` (wire
`activeBackgroundJobCount`) is a nonnegative, default-zero count of committed
managed jobs whose owned cleanup has not settled. It includes child-created jobs
even after the child runtime closes; stored and inactive sessions report zero.
Clients should use this projected state rather than scan processes or count agent
summaries. History-less snapshots update the count without replacing existing
history or turn pages.

Admission and finalization publish state-only updates, including changes between
turns and count changes such as 1 → 2. Output writes do not publish snapshots;
these job-state updates contain no commands or output bodies. Output is obtained
through `bash_read` tool results, with ordinary transcript projection and replay.

`is_quiescent` also accounts for unsettled managed jobs alongside existing
background work. Non-quiescence is not turn ownership: a long-running server does
not block new prompts, scheduled turns, or compaction. A bounded `bash_read` wait
is an ordinary tool call, not a waiting-only subagent wait. Root close/replacement
cleans up jobs; resumed or forked transcripts do not adopt historical handles.

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

## Dangerous-command authority

`policy/dangerous_commands/read` inspects one registered session's
dangerous-command authority. It takes `{"sessionId": "…"}` (no project key: the
project is always the session's own working directory) and returns:

| Field | Contract |
| --- | --- |
| `project` | The registered session's project identity. |
| `liveAuthorityGeneration` | Live revision of the session's grant store; the expected value for a live revocation. |
| `liveGrants` | Runtime-owned authority: session grants and pending once permits, each `{"operationId", "scope": {"gitDir", "workTree"}}`. |
| `acceptedSavedGrants` | Saved user-config grants the session's accepted restrictions contribute for the registered project. |
| `savedOperations` / `diskRevision` | Fresh disk read (`null` when unavailable); `diskRevision` is the expected revision for a saved removal. |
| `blockingDenylist` | Explicit shell-denylist entries vetoing dangerous git operations, each `{"pattern", "operations"}` — reported apart from grants because no grant can lift them. |

`policy/dangerous_commands/revoke` removes one operation's authority. It takes
`{"sessionId", "operationId", "expectedLiveGeneration", "expectedDiskRevision",
"userInitiated: true}` — `userInitiated` must be the literal `true`; lax truthy
coercion is rejected. The response reports the live and saved sides separately
(`live` and `saved`, each with `status` of `removed`, `conflict`, `skipped`,
`not_applicable`, or `failed`), the post-attempt `liveAuthorityGeneration`, and a
`warning` when a failed saved removal means the authority can return on the next
reload or start. Live removal is immediate (pending once permits are invalidated),
but a question already awaiting an answer is not cancelled: answering it after the
revocation still grants fresh authority for the decided scope. A saved removal
does not change the running session's accepted config until a reload, exactly as
a save never applies to the running session.

There is deliberately **no public grant endpoint**: no RPC grants an arbitrary
dangerous operation. Grants come only from completing the runtime approval
question; these two methods only inspect and revoke. See
[dangerous git commands](../guides/tools-safety.md#dangerous-git-commands).

## Limits

One app-server instance owns one attached root runtime and currently does not
model several simultaneous attached observers of that runtime. `events/read`
returns an empty batch in v0.1; clients consume live notifications and recover
state with a session read when needed. Restarting a stdio server ends its live
runtime and discards in-flight work and queued turns; persisted sessions can be
resumed normally. Public protocol views are projections, not the private session
format, and do not expose credentials or internal runtime objects.
