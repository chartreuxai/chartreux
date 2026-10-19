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

### Provider outages and the root agent

The main agent has built-in protection against provider failures; the behavior
is the same for subagents, which share the main agent's availability state.

- **Same-model failover.** Before any output is produced, a failed deployment
  is retried within the call's retry budget, and the turn can fail over to
  another deployment of the same model. Once streamed output has begun there is
  no transparent replay: the partial response stays in the transcript and the
  failure is reported instead of silently restarting on another deployment.
- **Shared cooldowns.** One availability registry is shared by the main agent
  and every subagent. A failure anywhere arms a cooldown for that deployment,
  and while it is armed no turn — main or child — is admitted to that
  deployment. Only a manual retry (in the TUI or over ACP) clears the model's
  cooldowns; automatic continuations never do.
- **All deployments unavailable.** When every deployment of the model is
  excluded, the turn ends with a visible error that names the model and each
  deployment's exclusion reason. There is no automatic retry storm and no
  hidden continuation behind it — but the error offers the manual retry
  affordance described next.
- **Automatic root recovery (default on).** Since v0.5.6 `[recovery.root]` is
  enabled by default (see
  [configuration](configuration.md#automatic-root-recovery)); a terminal root
  model-call failure — a connection error, timeout, server error, overload,
  rate limit, launch failure, interrupted stream, or empty response — is
  continued AUTOMATICALLY: the harness schedules the continuation turn itself,
  at most `auto_retries_max` (default 2) times per user-turn episode and within
  a monotonic `deadline_seconds` (default 2400) anchored to the first failed
  call. The schedule is cooldown-aware: it waits out the deployment and
  endpoint cooldowns (plus any `Retry-After`) with bounded jitter, re-reads
  live availability at admission, and never resets a cooldown — an automatic
  continuation is an injected turn, never a user request. Each milestone
  (scheduled, started, recovered, exhausted, cancelled) is a notice line in
  the transcript, so it stays readable after an overnight failure and across a
  reconnect. Typing, submitting, answering a question, or interrupting cancels
  a pending continuation; a manual retry starts a new user-directed episode
  and never replenishes the automatic budget. When every deployment is
  excluded for a transient reason the recovery waits and rechecks; permanent
  incompatibility (disabled, allowlist, images, thinking, compaction) is not
  auto-retried. When the budget or deadline is exhausted, the failure surfaces
  with the manual retry affordance described next.
- **Interrupted streams.** A stream that ends without a stop reason is
  continued by that same automatic root recovery — at most twice per user
  turn, each continuation a new injected turn — but ONLY while
  `[recovery.root]` is enabled. With it disabled (`enabled = false`), an
  interrupted stream is NOT continued automatically: the failure is visible
  immediately, and the manual retry affordance starts a new turn with the
  usual warning that completed actions may repeat.
- **Manual retry affordance.** When a turn ends with a retryable terminal
  failure — a backend error, a rate limit, an over-long response, an
  interrupted stream, or every deployment being unavailable — the error shows
  a "Retry — new turn" button and offers the same action on `Ctrl+R` (the
  error text names both, plus `/retry`). Either one starts exactly one new
  turn through the existing manual retry: completed tool results are kept and
  recent actions may repeat. The affordance is manual — it never starts a turn
  on its own — and it is suppressed while a turn is stopping or active, or
  while a runtime approval is still unresolved. A manual retry is a genuine
  user request and is the only action that resets the model's cooldowns, which
  is what makes retrying meaningful once every deployment was excluded; the
  error text says so honestly.
- **Transport disconnects.** If the connection to the app server is lost (or a
  turn stalls on the transport), the error offers inspection and reconnect
  guidance instead of a retry: the client reconnects automatically, and a new
  turn can be started once the connection is back. Nothing is continued
  automatically.
- **Same endpoint is not independence.** Deployments that share one endpoint
  are not independent of each other: an endpoint-wide throttle excludes all of
  them at once, and failing over between them is not outage protection.
- **Resuming a session.** Opening a persisted session in a fresh process
  restores the committed model and the transcript. It never restores in-flight
  questions or callbacks, and it never restores session-only grants such as
  root-access or dangerous-command approvals — those are per-session and must
  be granted again after resuming.

Honest limits: the main agent does not fall back to a different model
automatically — automatic recovery always continues on the same base model,
possibly a different deployment of it. Automatic root recovery requires a
healthy process and event loop; a process crash stays on the manual
fresh-process resume. With `[recovery.root]` disabled (`enabled = false`) there
is no automatic continuation at all — the manual retry affordance, `/retry`,
or a new message is the recovery. A retry also cannot interrupt a request
that is still in flight; interrupting stops the turn instead. The affordance
is not a watchdog: a turn that stalls without a terminal failure shows no
affordance at all, and a retry promises no reconciliation of remote effects —
check completed tool results before acting, because recent actions may repeat.

### Session transcript message-order errors

A provider may reject a malformed transcript with a permanent
`invalid_request_message_order` error ("Unexpected role 'tool' after role
'user'"). Since v0.5.6 the transcript's message order is guarded end to end,
so this error no longer poisons a session permanently:

- **Self-healing load.** Loading a session validates the transcript and
  repairs the unambiguous duplicate-placeholder pattern IN MEMORY for every
  caller. When a resumed session owns the write, the repair is also migrated
  durably: the original transcript is preserved as a
  `messages.jsonl.recovered-<timestamp>-<id>` artifact next to the live file
  before the live file is replaced, so a failed migration never destroys the
  original. A poisoned session heals on resume — no manual rewind needed.
- **No new duplicates.** The missing-response backfill sees a genuine result
  behind a steered user message (its lookahead is bounded to the logical
  block), the recorder anchors a settled result after its owning assistant
  message, and the save boundary refuses to persist a duplicate — the
  poisoning path can no longer create the shape.
- **Typed diagnostic.** An unrepairable transcript defect fails closed with
  a LOCAL `transcript_integrity` error: never transient, never retried, and
  never scheduled by automatic recovery. It is projected as an internal error
  carrying `category=transcript_integrity`, and automatic root recovery parks
  to the manual fallback with an offer to reload (resume) the session, which
  applies the repair. If the diagnostic ever appears, resuming the session (or
  rewinding to before the damage) is the actionable path — retrying the same
  turn would fail the same way.

Chartreux does not send product analytics or OpenTelemetry telemetry. Local
logs, configured provider and MCP requests, and requested web operations remain
separate from that policy; see [privacy](../project/privacy.md).
