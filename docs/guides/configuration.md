# Configuration

## Configuration layers

Chartreux resolves normal settings in this order, from lower to higher precedence:

1. Built-in defaults.
2. User configuration: `$CHARTREUX_HOME/config.toml` (normally `~/.chartreux/config.toml`).
3. Trusted project configuration: `.chartreux/config.toml`, discovered by searching upward from the working directory.
4. `CHARTREUX_*` configuration environment variables.
5. Runtime overrides.

The `[agent]` selections — identity, workflow, and orchestration strategy — have their own new-conversation precedence: CLI flag (`--identity`, `--workflow`, `--orchestration-strategy`) > environment (`CHARTREUX_AGENT__IDENTITY`, `CHARTREUX_AGENT__WORKFLOW`, `CHARTREUX_AGENT__ORCHESTRATION_STRATEGY`) > `config.toml` > shipped default. An omitted flag is always distinguished from an explicit selection. The selections and their resource content freeze per conversation; saved changes apply to new conversations only, and `/reload` reports them as pending. See the [identity architecture](../reference/identity-architecture.md).

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
- `prompts/`, `workflows/`, and `orchestration_strategies/` hold custom identity prompts, workflow bodies, and strategy bodies.
- `skills/` and `tools/` hold user skills and custom tools.
- `.env` can hold provider API keys.
- `logs/` holds local logs.

Set `CHARTREUX_HOME` before starting Chartreux to use another home directory:

```bash
export CHARTREUX_HOME="/path/to/chartreux-home"
```

This changes the locations of the files above as well as user prompts, workflows, orchestration strategies, and related user data.

## Agent selections and purpose routes

Who performs the work is selected by the `[agent]` table, not by the model
catalog. The three selections — identity, workflow, and orchestration
strategy — are documented in the [identity
architecture](../reference/identity-architecture.md); the accepted values and
the Settings > Identity & Orchestration group live in the
[configuration reference](../reference/configuration.md#agent-selections).
The default conversation runs the `coder` identity under the `minimal`
workflow with no orchestration strategy, and orchestration is a one-line
upgrade:

```toml
[agent]
identity = "orchestrator"
```

Delegated routing lives in `[orchestration.routes.*]` in `config.toml` — the
executable launch bindings for `task` purposes and seats. Sparse entries
override the shipped route of the same purpose, and new keys introduce custom
purposes. Every shipped target binds `$session` (the session model), so
default routing behaves identically with one configured model or ten:

```toml
[orchestration.routes.research]
targets = [{ model = "$session", identity = "researcher", tier = "substantive" }]

[orchestration.routes."review.deep"]
requires_distinct_models = 3

[[orchestration.routes."review.deep".targets]]
model = "$session"
identity = "reviewer"
tier = "substantive"
seat = "analytical"
```

Routes deep-merge across sparse layers. The retired `[dispatch]` catalog
overlay is superseded by this section: a legacy `[dispatch]` table in
`models.toml` is reported loudly at load with successor guidance naming
`[orchestration.routes.*]` — it is never merged — and catalog writes preserve
its values (comments and formatting are not preserved), never rewriting it.
See the [configuration reference](../reference/configuration.md#orchestration-routes)
for the full field list, seats, and diversity semantics, and
[Subagents](subagents.md) for the delegation guide.

## API keys and `.env`

A provider declares the name of its credential variable. For example, the shipped Mistral provider uses `MISTRAL_API_KEY`:

```bash
export MISTRAL_API_KEY="your-api-key"
```

Run `chartreux --setup` from an interactive terminal to configure a provider,
credentials, and models. Provider and model saves remain explicit. Setup then
shows an automatically seeded summary: usable existing role selections are
preserved, and unavailable bindings are assigned a ready configured model.
No per-role questions are required. **Finish setup** saves the seeded presets
and completes setup; **Customize role presets** opens the existing model/thinking
editor for optional changes or repairs. Invalid or unavailable bindings block
completion with repair guidance.

The presets screen edits role presets — one row per role with its model and
thinking level, and a per-role editor for both fields. It offers no dispatch
modes: routing is configured by purpose routes and the `[agent]` selections,
not by the catalog. Explicit existing selections are preserved. After two
deduplicated, budget-exceeded implementation attempts fail on the same task, a
passive graduation nudge at an idle boundary suggests reviewing model presets
or delegating differently. It opens the presets screen, never changes anything
automatically, and can be dismissed permanently; headless sessions do not show
it. Saving a second canonical model no longer triggers it — orchestration is
never inferred from the roster size. Setup uses
the configured theme (`auto` by default), without asking for a theme choice.
Web search is not a required setup stage: saved choices remain unchanged, and
Settings > Web search or `/web-search` remains available later.
Setup saves configured provider keys in `$CHARTREUX_HOME/.env`; if persistence
fails, it reports when a key is available only for the current session.
Without an interactive terminal, setup prints actionable guidance instead of
launching the TUI. You can also create that file yourself:

```dotenv
MISTRAL_API_KEY=your-api-key
```

Credential precedence is precise: a **non-empty** value already present in the process environment wins. If that variable is unset or empty, Chartreux loads a non-empty value from `.env`. At runtime, API-key lookup falls back to the keyring only when no non-empty environment value is available. Setup and the Web Search editor save keys to `.env` when possible, with a session-only outcome if disk persistence fails; they do not save onboarding credentials to the keyring. The keyring is only a runtime lookup fallback. Keep `.env` private and do not commit it.

For web search, use Settings > Web search or `/web-search` to select a provider and manage its credential separately from search settings. A key may be saved or kept for the current session; see [Tools and safety](tools-safety.md) for provider, readiness, and retry behavior.

## Status line and message timing

Open `/settings` and select **Status line** to customize the bottom session row.
`Enter` or `Space` cycles a segment's state; directory and context variants are
part of that cycle, and those two segments cannot be disabled. Use `[` and `]`
to reorder segments. The live example preview reflects pending edits, not live
usage. **Apply changes** saves the batch to the user configuration; leaving with
unsaved edits asks whether to discard them. `Ctrl+R` removes saved status-line
overrides after confirmation, allowing lower-precedence values to apply. `D`
shows details and `F1` opens local help.

The same settings can be written in `config.toml`:

```toml
[status_line]
segments = ["directory", "git-branch", "context", "background-jobs"]
directory_style = "name"
context_style = "tokens-percent"
separator = "pipe"
```

Enable **Background jobs** (`background-jobs`) for `Jobs N`, including `Jobs 0`
when none are active. This optional count covers managed shell jobs in the current
root session and all its children, including jobs that survive child completion;
it does not count background agent runs or finished retained job records. It
refreshes from live session state between turns. The default row still contains
only directory, PID, and context. At narrow widths, PID yields first, then
optional segments from the end, so place Jobs earlier if it should outlast other
optional details.

Context displays as, for example, `135k/400k (34%)`; the denominator is the
effective automatic-compaction threshold, not the model's maximum context window.
Git branch lookup is asynchronous and cached. Enable `spend-today`, `spend-week`,
or `spend-month` to show recorded USD spend across **all projects**, for example
`Today $12.34`, `Week $12.34`, or `Month $12.34`. Windows are local calendar day,
Monday-start week, and month,
not rolling intervals. `—` means loading or unavailable, `$12.34+` is a known
lower bound with some cost unknown, and `Unknown` means nothing is priced.
`$0.00` can mean a valid empty window or recorded zero-cost calls. Open `/usage`
for Details or a Current project filter; that filter never changes status-line scope. See the
[status-line reference](../reference/configuration.md#status-line) for accepted
segments and width handling.

`show_message_timestamps` controls known posting times and settled durations in
main and child transcripts. Headers show local `HH:MM`, with a date for messages
not posted today, without role labels. Tool summaries show per-call durations;
the last assistant message shows the whole-turn total. These values do not tick
while work runs. Turning the setting off hides the displays but does not stop
capture or persistence; older messages without timestamps stay unstamped.

## User questions and unattended expiry

The `ask_user_question` tool can mark a question as safe to leave unanswered.
Each question gains a `kind` — `informational`, `preference`, or `requires_user`
(the default) — and an optional `unattended_default` describing what to do when
no answer arrives: `defer_affected_work`, or `preserve_existing_choice` naming an
already-existing option to keep. A question without an `unattended_default`
never expires, and any `requires_user` question blocks expiry for its whole
batch.

When every question in a batch is explicitly defaultable and the feature is
enabled, an unanswered question expires into an honest **unanswered**
disposition after the configured timeout: the dependent work is deferred and
nothing is approved. Expiry never selects an option, never fabricates an
answer, and never applies to runtime approvals such as dangerous-command or
root-access grants — those always wait for an explicit decision. While an
eligible question is open, the TUI shows a countdown with the declared outcome
and a **Keep waiting** action (`Ctrl+K`) that lifts the deadline so the question
stays open without one; answering before the deadline always wins, and a late
answer is rejected with a definite expired response.

Configure the feature with the `[user_questions]` table:

```toml
[user_questions]
unattended_timeout_enabled = true
default_timeout_seconds = 300
max_timeout_seconds = 3600
```

`unattended_timeout_enabled` defaults to `true` (enabled since v0.5.6; set it
to `false` to return to always-wait behavior). The per-question opt-in marker
remains the primary safety gate regardless of this flag: only questions that
declare an `unattended_default` can ever expire. The timeouts must be finite
numbers between 60 and 3600 seconds with `default_timeout_seconds <=
max_timeout_seconds`; invalid values are rejected, not clamped. A model may
pass a per-request `timeout_seconds` override, which must fit the configured
range and is rejected otherwise. Configuration changes apply to newly opened
questions only, and configuration can never make a protected question expire.
Headless and programmatic modes are unchanged: the ask tool stays disabled
there.

Unanswered batches are recorded as deferred questions (bounded, oldest first,
with the overflow counted rather than silently dropped) and carried through
compaction, so a resumed session still reports what was deferred. See the
[configuration reference](../reference/configuration.md) for the complete
setting schema.

## Automatic root recovery

When a root turn ends with a terminal model-call failure — a connection error,
timeout, server error, overload, rate limit, launch failure, interrupted
stream, or empty response — the harness can schedule the continuation turn
itself, bounded and cooldown-aware, instead of waiting for a manual retry.
The manual retry affordance (button, `Ctrl+R`, `/retry`) remains the fallback
when the automatic budget or deadline is exhausted.

Configure it with the `[recovery.root]` table (symmetric with
`[recovery.subagents]`):

```toml
[recovery.root]
enabled = true
auto_retries_max = 2
deadline_seconds = 2400
cooldowns_seconds = [30, 120]
jitter_fraction = 0.2
availability_recheck_seconds = 1
eligible_classes = [
  "connection",
  "timeout",
  "server_error",
  "overload",
  "rate_limit",
  "launch_failure",
  "incomplete_stream",
  "empty_response",
]
```

- `enabled` defaults to `true` (enabled since v0.5.6). While it is false NO
  root automatic continuation runs at all — including the interrupted-stream
  continuation — and the manual retry affordance is the only recovery (a
  breaking change for the interrupted-stream path, which used to continue
  unconditionally). Setting it to `false` returns the root agent to
  manual-retry-only recovery. Enabling it requires `auto_retries_max >= 1`
  and a non-empty `eligible_classes`.
- `auto_retries_max` (0–10) is the ONE shared automatic-turn budget per
  user-turn episode: the interrupted-stream continuation and provider
  failures share one count. It counts admitted continuations, not request
  attempts. A manual user retry starts a new user-directed episode and never
  replenishes this budget.
- `deadline_seconds` (1–86400) is a monotonic, advisory bound anchored to the
  first failed logical call's start. It bounds scheduling and admission; it
  never kills an in-flight request.
- `cooldowns_seconds` is the backoff schedule (finite, non-negative,
  non-decreasing, at least `auto_retries_max` entries). The continuation is
  due at `max(configured backoff, deployment cooldown, endpoint cooldown,
  Retry-After)` plus `jitter_fraction` (0–1) of bounded jitter. Automatic
  continuations NEVER reset cooldowns.
- `availability_recheck_seconds` (positive, at most `deadline_seconds`) is the
  floor for read-only availability rechecks while a probe is in flight or a
  cooldown gap remains; such rechecks never consume a retry.
- `eligible_classes` is restricted to the transient allowlist above.
  Preflight, permission or authorization denial, budget stop, cancellation,
  task blocker, permanent config/auth/billing/quota, context-size, and
  unknown-internal failures never schedule, and neither does a tool's network
  exception without measured model-call provenance.

Typing, submitting, answering a question, or interrupting cancels a pending
continuation; each milestone (scheduled, started, recovered, exhausted,
cancelled) is a notice line in the transcript. See
[Troubleshooting](troubleshooting.md#provider-outages-and-the-root-agent) for
the behavior and its honest limits.

## Automatic subagent recovery

When a background subagent's run fails recoverably, the agent is suspended
(see [Subagents](subagents.md#failure-recovery-suspension-and-resume)) and the
harness can resume the SAME agent itself — a successor run with a new `run_id`
on the same identity, admitted through the same `resume_agent` operation an
explicit request uses, bounded and cooldown-aware, running independently of
parent turns (an overnight parent idle does not block it). Recovery is enabled
in stages — observation (always on), explicit resume (always available),
automatic recovery (default ON since v0.5.6) — and the controller below is the
third stage.

Configure it with the `[recovery.subagents]` table (symmetric with
`[recovery.root]`):

```toml
[recovery.subagents]
enabled = true
auto_retries_max = 2
same_agent_resumes_max = 2
cooldowns_seconds = [30, 120]
jitter_fraction = 0.2
episode_deadline_seconds = 2400
suspension_ttl_seconds = 1800
max_suspended_agents = 8
max_suspended_bytes = 67108864
recovery_concurrency = 2
eligible_classes = [
  "connection",
  "timeout",
  "server_error",
  "overload",
  "rate_limit",
  "launch_failure",
  "incomplete_stream",
  "empty_response",
]
```

- `enabled` defaults to `true` (enabled since v0.5.6). While it is false the
  controller is inert: suspension and the explicit `resume_agent` /
  `reconcile_result` ports are untouched either way, and disabling cancels
  future scheduling without discarding evidence (failed results, envelopes,
  and counters are kept). Enabling it requires `auto_retries_max >= 1`.
- `auto_retries_max` (0–10) bounds the automatic resumes per recovery chain;
  `same_agent_resumes_max` bounds explicit plus automatic resumes per chain.
  Spend/turn/permission budgets never reset on resume, and a package-lineage
  recurrence counter spans replacements.
- `cooldowns_seconds` is the backoff schedule (finite, non-negative,
  non-decreasing, at least `auto_retries_max` entries) with `jitter_fraction`
  (0–1) of bounded jitter; a longer `Retry-After` and any deployment or
  endpoint cooldown postpone admission. An automatic resume NEVER resets
  cooldowns.
- `episode_deadline_seconds` is a monotonic, advisory bound anchored to the
  first failure: no new admission after it, never an in-flight kill (there is
  no hard in-flight timeout; the retry budget is consulted only between
  attempts).
- The suspension caps (`suspension_ttl_seconds`, `max_suspended_agents`,
  `max_suspended_bytes`) bound warm suspension separately from idle retention
  and tombstones; `0` disables each.
- `eligible_classes` is restricted to the transient allowlist above.
  Cancellation, budget stop, authorization denial, preflight, task blocker,
  permanent config errors, and unknown-internal failures never auto-resume;
  unknown tool effects suspend with `reconcile_required` until reconciled.
- Endpoint-correlated throttling blocks recovery onto ALL affected-endpoint
  deployments regardless of model/provider naming; the eligibility view is
  read-only and never claims another caller's probe.

## Logging and diagnostics

Use `chartreux doctor` to validate trusted configuration, catalog entries, and
model selection locally. Bare doctor does not load `.env`, inspect keyring
credentials, start subprocesses, or make network requests. For opt-in live
checks and billable provider smoke probes, see [Troubleshooting](troubleshooting.md#run-diagnostics).

Chartreux writes structured local logs to `$CHARTREUX_HOME/logs/chartreux.log`. Set `log_level` in `config.toml`, use `/log-level` for a session override or persisted setting, or set `LOG_LEVEL`. Valid levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`; the default is `WARNING`.

The `/log-level` picker edits session and saved config levels as drafts. Use
Tab/Shift+Tab to move between scope controls, levels, and **Apply changes**;
Up/Down moves within the levels, and Enter/Space or a badge click toggles the
selected scope's draft level. Only **Apply changes** or `Ctrl+S` applies the
batch. Escape asks before discarding unsaved edits.

The effective level is chosen in this order: session override, `DEBUG_MODE=true` or a valid `LOG_LEVEL`, `log_level` in configuration, then the default. For local data and network-traffic policy, see [Privacy](../project/privacy.md). For the complete setting schema, see the [Configuration reference](../reference/configuration.md).
