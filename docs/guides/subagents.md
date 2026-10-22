# Subagents

Use subagents to delegate bounded work or run independent investigations in
parallel. The primary agent remains responsible for the conversation and can
continue working while subagents run.

Delegation is identity-based: the `task` tool launches a child **identity** —
who performs the work — under a purpose **route** that binds the launch's
model and work class. The model behind the whole design, including the frozen
per-conversation lifecycle, lives in the [identity
architecture](../reference/identity-architecture.md); this guide shows how to
use it day to day.

## Launching work

The `task` tool creates a new agent instance when `agent_id` is omitted. A
launch is selected with four optional top-level fields — `model`, `identity`,
`purpose`, and `seat` — and every one of them is validated before dispatch:

```text
task(task="Implement the bounded change in the issue", identity="coder")
task(task="Recommend an approach and identify risks", identity="advisor")
task(task="Review the authentication changes", purpose="review.standard")
task(task="Investigate the parser's callers", purpose="research")
task(task="Audit the retry logic", model="@heavy", identity="reviewer")
```

- **`identity`** names who performs the work: a bare built-in (`coder`,
  `researcher`, `reviewer`, `advisor`, `orchestrator`), a `builtin:`- or
  `user:`-qualified id, or a bare custom prompt id. The identity's prompt body
  is the child's contract.
- **`model`** takes one model expression: a canonical model name, an `@role`
  preset, or the `@<model>[:<identity>]` shorthand — `@heavy:advisor` selects
  the `heavy` preset for an advisor child. In the shorthand, the identity
  component accepts a bare built-in or a `user:`-qualified id.
- **`purpose`** selects a configured launch route (see [Purpose
  routes](#purpose-routes)); **`seat`** picks one target of a multi-seat
  purpose and is meaningless without one.

Missing components are never guessed. A new child defaults to the `coder`
identity under the bounded-child execution context, runs the session model, and
inherits neither the parent's workflow nor its orchestration strategy — children
never restart the parent's process. Conflicts are rejected instead of resolved
by precedence: supplying `model` together with `config.model` is a validation
error (the top-level field is the canonical location), a shorthand identity
component together with the structured `identity` field is rejected, and a
`seat` without a `purpose` is rejected.

An agent instance's retained identity is its `agent_id`. Handles of the form
`agent-N` (digits after `agent-`) are reserved for instances and rejected as
launch targets; an unknown retained handle reports `Unknown agent_id instance
handle`. The retired `agent_type` field fails validation with successor
guidance (the shipped profile names map onto identities: `worker` → `coder`,
`advisor` → `advisor`, `reviewer` → `reviewer`; a custom profile name becomes a
`user:<name>` custom identity prompt), and the old `fan_out: true` flag is
rejected with guidance to launch separate tasks.

A task launches in the background by default (`background: true`). Prefer
background launches even when you need the result: background agents stay
inspectable in the agent browser and can be amended mid-flight, and you collect
the outcome with `wait_for_agent`. Set `background: false` only when blocking
is truly required. The call returns a successful launch acknowledgment with
`status: "launched"`, non-empty guidance, and stable `agent_id` and `run_id`
handles rather than waiting for the answer; it is not the subagent's terminal
result. The acknowledgment also carries the instance's human-facing `name` and
states whether the agent is retained or closes on completion. Subagents cannot
launch other subagents: child launches are depth-limited to one, and background
agents cannot be launched from subagents.

The `coder` identity dispatches skill-first. When the task names a skill, the
child loads it first and follows its methodology and output format; otherwise
the child selects the applicable task skill itself (`sub-implementor` for edits,
`sub-finder` for searches, `sub-verifier` for verification, `sub-explorer` for
exploration). Name a skill only when a specific methodology or output format is
required:

```text
task(task="Load the sub-finder skill and locate all callers of the parser.", identity="coder")
```

Background agents are ephemeral by default: pass `retain: true` to keep the
instance resident after the run completes, or set
`[subagents].retain_agents` globally. `retain` is only valid for background
launches. See [Retention](#retention).

## Built-in identities and postures

Chartreux ships five built-in identities. Each carries a harness-known posture
that validation and the tool layer both enforce:

| Identity | Delegation | Personal tools | Delegation tools | Posture |
| --- | --- | --- | --- | --- |
| `coder` | optional | all granted | all granted | Direct execution; may delegate |
| `orchestrator` | required | non-mutating only | all granted | Must orchestrate from the root session |
| `researcher` | optional | non-mutating only | non-mutating only | Read-only analysis |
| `reviewer` | optional | non-mutating only | non-mutating only | Read-only analysis |
| `advisor` | optional | non-mutating only | non-mutating only | Read-only analysis |

The read-only postures are honest: `bash` classifies as mutating regardless of
command text, and unclassified or custom tools (including MCP tools) classify
as mutating by default, so a read-only identity genuinely excludes them. A
required-delegation identity (`orchestrator`) is rejected as a child with an
actionable error — it must orchestrate from the root session. Custom identities
carry no harness-known posture: their prompt file is the whole contract, and
their delegation set is the full granted ceiling of the session.

Three independent axes govern a child's tools, and narrowing is monotonic down
the tree:

- **Granted authority** — the session's configured tool ceiling; it never
  widens down the agent tree.
- **Personal tool set** — what the identity may execute itself, enforced below
  the configuration layer: launch-level tool overrides cannot loosen an
  identity's personal posture.
- **Delegation tool set** — what an identity may pass to children.

A child's effective tools are the granted ceiling intersected with every
ancestor's delegation set intersected with the child's personal set. The
consequences are exercised at the root, since children cannot launch at all: a
`researcher` cannot obtain a writing `coder` through `task` or a direct launch,
while an `orchestrator` cannot write personally but can delegate writes within
the granted ceiling.

## Purpose routes

Routes are the executable launch bindings for `task` purposes, configured in
`[orchestration.routes.*]` in `config.toml` (full syntax in the [configuration
reference](../reference/configuration.md#orchestration-routes)):

```toml
[orchestration.routes.research]
targets = [{ model = "$session", identity = "researcher", tier = "substantive" }]
```

Each route carries `targets` — a required, non-empty list of
`{ model, identity, tier, seat?, overlay?, replacement_capability? }` bindings
— and `requires_distinct_models` (default 1). `$session` binds the session
model, and every shipped route target uses it, which is why one configured
model and ten produce identical default routing: multi-model routing exists only
through explicit route overrides.

For one launch, precedence is: structured explicit fields (`model`, `identity`)
override the selected route's corresponding fields; a shorthand expression sits
between them; conflicts between shorthand and structured fields are rejected.
The effective tier (`mechanical` or `substantive`) is the route target's and
feeds the failure-time replacement policy; `replacement_capability` is marked
separately. A direct unrouted launch takes the identity/purpose default tier.
An unavailable model binding fails with an error telling you to configure it
explicitly — another canonical model is never substituted.

Multi-seat purposes require an explicit `seat` selector; there is no implicit
fan-out. The shipped example is `review.deep`:

| Seat | Identity | Notes |
| --- | --- | --- |
| `analytical` | `reviewer` | Read-only analysis seat |
| `peer` | `reviewer` | Read-only analysis seat |
| `execution` | `coder` | The executing seat, with the `review-execution` overlay |

```text
task(task="Deep-review the identity launch module", purpose="review.deep", seat="execution")
```

The execution seat runs as `coder` with the shipped `review-execution` overlay —
full-artifact review with `file:line` citations and declared check-only
verification — because it was never read-only in spirit. Overlay bodies are
harness-owned: there is no user overlay directory, and an unknown overlay
reference fails loudly. The overlay rides the frozen persona instructions at
launch; it grants no authority and never rewrites the identity binding.

`review.deep` declares `requires_distinct_models = 3`. Diversity is evaluated
over **canonical-model cardinality** — the same base model on two providers
counts once — counting partial duplication (A/A/B fails a requirement of 3).
An unmet requirement is reported instead of pretending diversity; a
fresh-context reviewer on the same model is context separation, honestly
labeled. Configure distinct models explicitly via per-seat model fields.

The shipped routes are `implementation` (coder), `mechanical-edit` (coder,
mechanical), `research` (researcher), `search` (researcher, mechanical),
`exploration` (researcher), `verification` (coder, mechanical),
`design-analysis` (advisor), `planning-analysis` (advisor),
`implementation-demanding-settled` (coder), `review.quick` (reviewer),
`review.standard` (reviewer), and `review.deep` as above. Route keys may also
introduce custom purposes. Purpose labels organize work; they never grant
permission.

## The task allowlist and denylist

Launch admission enforces an identity allowlist and denylist under
`[tools.task]`:

```toml
[tools.task]
allowlist = ["coder", "reviewer"]
denylist = ["user:contractor"]
```

The allowlist (default `["coder"]`) merely auto-permits launches of matching
identities. The denylist is an absolute veto, applied **after** resolution —
shorthand, route, and retained-instance launches are all checked against the
resolved identity, not the raw spelling. Entries are exact names, shell-style
globs, or `re:`-prefixed regular expressions, and matching is symmetric across
equivalent spellings: denying `coder` also blocks `builtin:coder`, and denying
`user:foo` also blocks the bare `foo`. A denied launch reports `Task denied for
launch identity`.

The Settings > Identities inventory is denylist-only: unchecking an identity
writes this denylist. It can only block launches; it never grants roots or tool
authority.

## Dynamic launch configuration

The nested `config` argument carries per-launch overrides rather than a
profile-specific default:

```text
task(
  task="Review the authentication changes",
  identity="reviewer",
  config={
    thinking="max",
    instructions="Focus on authorization boundaries.",
    enabled_tools=["read_file", "grep"]
  }
)
```

At launch or when retasking an idle subagent, `config` can override the
thinking level, enabled and disabled tools, and a tool's permission or
allowlist; `instructions` is an additive per-launch instruction overlay on the
identity's prompt body. Supply the model at exactly one level — the top-level
`model` field is canonical; `config.model` is the legacy nested location, and a
dual supply is a validation error. Overrides are session-local; they do not
modify the resolved identity or `config.toml`. The launch persona (the identity
prompt and base instructions) is immutable for a retained subagent, so a changed
persona requires launching a new subagent. A subagent cannot gain tool authority
beyond its parent's delegation ceiling. Retasking is available only for
background subagents.

Subagents can reread exact instruction files injected into their context,
including `AGENTS.md`, using `read_file` or exact-file `grep` even outside the
workspace roots. This does not grant directory searches or write access, and
tool denials and sensitive-file protections still apply. See
[Privacy](../project/privacy.md#instruction-file-access) for the data boundary.

Subagents inherit access to the session scratchpad through their
parent-authority chain rather than receiving a new workspace grant. File tools
and shell checks resolve that access against the ancestor scratchpad roots.
Broken authority chains, symlink escapes, and retargeted roots fail closed; tool
denials and sensitive-file protections still apply.

## Retention

Retention is opt-in and layered: an explicit `task` `retain` argument wins over
the global `[subagents].retain_agents` default (off). There are no per-identity
retention settings. `retain` is only valid for background launches; reuse with
`agent_id` inherits the instance's committed retention policy unless `retain`
is passed.

Retained idle agents are kept for `idle_ttl_seconds` (default 300) with at most
`max_idle_agents` (default 4) resident; `0` disables each limit, so a retained
agent with both limits at zero stays resident without a time or count bound.
TTL and idle-cap eviction remove the runtime but preserve terminal metadata and
stored results, so `get_agent_result` remains usable after eviction.

Terminal metadata feeds the browser's bounded **Recent** group: it expires after
`tombstone_ttl_seconds` (default 900) and is capped at `max_tombstones`
(default 32, oldest first); `tombstone_ttl_seconds = 0` retains no terminal
metadata, and `max_tombstones` must stay positive. Reads do not extend the
deadline. Metadata expiry is independent of result expiry: an expired record's
result can remain retrievable after its metadata disappears. Release is the
opposite: it discards the record, stored results, expiry markers, and wait
leases immediately, so a released handle has nothing left to retrieve. Because
ephemeral agents leave no idle row and the status line counts active work only,
completion and eviction notifications — which carry the agent name, handle,
run, and outcome — are the primary failure surface for ephemeral agents.
Release subagents when you are done.

## Lifecycle and results

A background subagent starts **running**, may be **compacting**, then passes
through **finishing** while cleanup and result publication settle. What happens
next is the retention policy: a retained instance becomes **idle** and remains
available for inspection or reuse, while an unretained instance **closes** — its
runtime is removed and only terminal metadata plus the stored result remain.
**Closed** records cover normal completion, including failed and cancelled runs;
**evicted** records cover retained runtimes removed by the idle TTL or idle
cap. Availability is separate from the run outcome: the browser shows the last
run's outcome, including **Failed**, **Cancelled**, or **Budget stopped**, on
idle, closed, and evicted rows alike. **Stopping** is a local TUI presentation
state for a pending stop, not a registry availability state or a terminal
outcome. The parent receives a completion notification that names the agent and
points it to `get_agent_result`.

A would-be successful run without final prose is reported as **Failed**, with
`completed: false` and stop reason `error`, not as an empty successful result.
Empty model responses get at most one bounded replay; terminal failures use
`EmptyLLMResponseError` or `IncompleteLLMResponseError`. Published streaming
output is never replayed automatically.

If the parent is only waiting on subagents, submitting a new message in the TUI
cancels its waits, not the child runs, and steers the same parent turn
immediately. See [Input and queueing](terminal.md#input-and-queueing).

A subagent's `ask_user_question` calls never expire: unattended-expiry
admission is limited to the root agent in an interactive session, so a
background agent's question waits for an explicit answer (or the run's
cancellation) instead of deferring on a timer. Only the root agent's
explicitly defaultable questions can expire unanswered; see
[User questions and unattended expiry](configuration.md#user-questions-and-unattended-expiry).

- `check_agents` lists resident and Recent subagents with their names, handles,
  availability (`running`, `finalizing`, `idle`, `suspended`, `restoring`,
  `closed`, or `evicted`), and effective model and thinking information.
- `get_agent_result(agent_id, run_id)` returns a completed result without
  waiting; it returns no result while the run is active. It keeps working for
  closed and evicted records while their stored result is retained; a released
  or never-seen handle reports an unknown-run error, and an expired result
  raises a result-expired error.
- `wait_for_agent(agent_id, run_id, timeout=...)` waits for completion. The
  default timeout is 1500 seconds; a timeout returns a `still_running` result —
  a normal outcome, not a failure — reporting the elapsed wait, run status, and
  turns used, and instructing you to end your turn or continue other work now
  and re-invoke `wait_for_agent` in a later turn (queued user input is only
  delivered between turns). A timeout never cancels the subagent. Pass an
  explicit larger timeout to wait longer, or `timeout: null` to wait
  indefinitely.
- Pass a retained idle instance's `agent_id` to
  `task(agent_id=..., background=true, task=...)` to give it another assignment;
  omit the launch fields to keep its committed identity. Omitting `retain`
  preserves the instance's committed policy; pass `retain: false` for a final
  continuation that closes afterward. Both launch acknowledgments and
  `check_agents.reuse_guidance` include this reuse guidance. Reusing a closed
  handle is rejected with guidance to launch a new task (use `retain: true` when
  a resident continuation is required); an evicted handle is rejected as
  evicted, and a released or unknown handle reports unknown.
- `release_agent(agent_id)` closes and removes a subagent — resident or
  terminal — when it is no longer useful, purging its stored results and wait
  leases.

### Failure recovery, suspension, and resume

A recoverable failure — a typed transient provider fault with settled or
verifiable tool effects — moves a background agent to **suspended** instead of
closing it. Suspension is an explicit, bounded exception before closure: it is
not retention under another name, and it is not resurrection. A suspended
agent keeps its identity, conversation, tool state, and cumulative budgets
warm while its provider connection is released; **closed** and **evicted**
agents remain non-reusable exactly as before. Suspension is warm-only: a
parent-process restart forfeits suspended agents (they convert to closed,
non-resumable) with no dangling scheduled recovery, and scratch artifacts
remain the salvage path across restarts. Suspension expires under its own
bounds (`[recovery.subagents]` `suspension_ttl_seconds`,
`max_suspended_agents`, `max_suspended_bytes` — separate from the
idle-retention and tombstone caps); expiry removes recovery eligibility and
emits a notification without deleting the transcript, artifacts, or stored
results. Cancellation, release, a session change, and shutdown invalidate
pending recovery; normal successful completion still closes an ephemeral
agent.

Two management tools operate on a suspended agent:

- `resume_agent(agent_id, expected_run_id, idempotency_key)` admits a
  successor run of the SAME agent identity: a new `run_id`, the same
  `agent_id`, session, persona, and task scope. Outcomes: `scheduled`,
  `already_scheduled`, `already_resumed`, `reconcile_required`, `expired`,
  `not_resumable`, `forbidden`. Concurrent calls admit exactly one successor
  (single-flight, run-pinned). The failed run's result is immutable and stays
  retrievable; waits pinned to it return ITS failed result, never silently
  redirecting to the successor.
- `reconcile_result(agent_id, expected_run_id, transcript_boundary,
  verified_effects)` clears a `reconcile_required` barrier after the unknown
  tool outcomes are verified externally (inspect git, filesystem, or job
  state). It is pinned to the failed run and its transcript boundary, so
  stale or duplicate evidence from an earlier run cannot clear a later run's
  barrier. Outcomes: `cleared`, `prohibited` (verification shows unrepeatable
  damage, with reason codes), `stale`, `duplicate`, `not_pending`,
  `forbidden`. Replacement with `task(agent_id=..., replace_run=True)` remains
  permitted at any point; the barrier never auto-clears.

Reconciliation barriers: when a failed run's tool effects are unknown, no
action is replayed — the agent suspends with `reconcile_required` until the
effects are verified. Recovery promises no exactly-once tool execution:
recent actions may have completed despite the interruption, so a resumed
agent inspects completed tool results and current workspace state before
acting rather than repeating them blindly. A run that left no tool call
outstanding submits the empty `verified_effects` list — the complete
zero-entry accounting — which clears the barrier exactly when zero tool
calls are outstanding.

Recovery is enabled in stages, and the release default is automatic:

1. **Observation** (always on): every terminal failure carries a typed
   failure envelope — class, phase, attempted deployment and endpoint,
   request evidence, progress, and recovery eligibility — through the
   result, wait, and notification surfaces.
2. **Explicit resume** (always available): `resume_agent` and
   `reconcile_result` are not gated on any flag.
3. **Automatic recovery** (default ON since v0.5.6): the bounded controller
   in `[recovery.subagents]` schedules and admits a same-agent resume itself
   after an eligible failure — bounded by `auto_retries_max` per recovery
   chain, cooldown-aware through the shared availability registry (an
   automatic resume never resets cooldowns), and running independently of
   parent turns. Disabling the flag cancels future scheduling without
   discarding evidence. See
   [Automatic subagent recovery](configuration.md#automatic-subagent-recovery).

### Cancelling and replacing background work

The orchestrator can stop one run with `cancel_agent(agent_id, run_id=None)`.
Omitting `run_id` targets the current run; supplying it pins the request to
that run and cannot stop a newer assignment. This tool manages only the
orchestrator's own background agents. Foreground work uses the normal parent
interruption controls.

A response of `stop_requested` means the stop was accepted, not that cleanup is
finished. `already_stopping` means a stop is already pending;
`already_finishing` preserves work that has finished executing. `not_running`,
`unknown_run`, and `forbidden` report an inactive, unknown, or unauthorized
target. Use `wait_for_agent` to wait for the terminal result. Cancellation does
not interrupt the parent or sibling agents and does not release the agent.

Cancelled results carry `completed: false`, accumulator partial output, and a
`stop_reason`: `user_cancelled`, `orchestrator_cancelled`, or `retasked` for
these operations. The first accepted stop reason wins. A genuine completion or
error is not rewritten by a late stop. Stopping does not undo file, shell, or
remote side effects, generate a summary, or salvage additional output.

The conversation, transcript, partial result, and existing waits remain
available subject to the retention policy: an unretained agent closes at
finalization (the cancelled run's result stays retrievable), while a retained
agent stays idle and may later be evicted on TTL or idle-cap limits. Eviction
keeps terminal metadata within the Recent bounds and stored results until
separate result expiry. Saved transcript inspection depends on session
logging and disk availability, and only within the metadata window. Explicit
`release_agent` removes the agent and purges its stored results and wait
leases.

To replace a busy assignment in the same retained conversation:

```text
task(agent_id="<agent-id>", task="Investigate the revised requirement", replace_run=True)
```

`replace_run=True` requires an `agent_id` and background mode. It stops the old
run with reason `retasked`, waits for its cleanup, and launches the replacement
without overlapping execution. The replacement prompt states that it supersedes
the interrupted task; the acknowledgment metadata includes both run IDs.
Busy replacement cannot change the identity or launch configuration. It
preserves the instance's committed retention policy and rejects a conflicting
`retain` override before stopping anything. An idle target uses ordinary reuse,
which still permits runtime configuration overrides. Ordinary reuse of a busy
agent without `replace_run=True` is rejected. Already stopping or finishing
targets are not replaced; if replacement admission fails after the old run was
stopped, an unretained instance still closes rather than lingering resident, and
the error is reported.

A user stop notifies the parent as an injected message **without starting a
parent turn**. An idle parent learns of the cancellation on its next turn. The
notification attributes the cancellation to the user; orchestrator cancellation
and retask are attributed separately.

For a cross-model replacement after a failure, omit `agent_id` (the
replacement gets a NEW handle) and claim the lineage through `config`: the model
goes in the top-level `model` field (or the legacy nested `config.model` —
supply exactly one), plus `config.work_package_id`,
`config.predecessor_agent_id`, and `config.package_tier` (`mechanical` or
`substantive`). Launch admission validates the claim at failure time —
endpoint independence from the attempted failed endpoint, the reconciliation
barrier, and the per-package replacement budget — and rejects it with a typed
reason otherwise. Budgets and recurrence counters span replacements and are
never reset.

## Frozen lifecycle for children

Every child is its own conversation with its own frozen policy bundle. The
child's identity prompt content is captured at launch — the frozen persona is
re-keyed on the identity's content revision — so later edits to a custom
identity's prompt file apply to new launches only, and a child keeps its
committed identity even if the underlying file is edited or removed. A route
overlay rides the frozen persona instructions at launch and never rewrites the
identity binding. Children never inherit the parent's workflow or strategy
bodies: they run the bounded-child execution context under their own identity.

The parent's `/reload` refreshes the parent's reloadable machinery — AGENTS.md,
the skills catalog, and the model catalog — and reports policy edits as pending;
it never rewrites a child's frozen identity content, and a child's committed
model and concrete provider deployment are revalidated on resume rather than
re-resolved from current configuration. See [Frozen
lifecycle](../reference/identity-architecture.md#frozen-lifecycle).

## Agent names

Every instance gets a human-facing name of the form
`{identity}-{model}-{number}` — for example `coder-mistral-large-4-3` or
`reviewer-glm-5-3-7`. The identity segment is the resolved identity reference
(a custom identity uses its display label with whitespace runs collapsed to
hyphens), the model segment is the canonical base model, and the number is the
handle's sequence digits (`agent-15` ↔ `...-15`). Names appear in browser rows,
the details panel, the transcript header, completion and eviction notifications,
the launch acknowledgment, and `check_agents` output. Names are display-only:
`agent_id` is the only programmatic handle, there is no name-to-handle lookup,
and a name never parses as a handle.

## Managed shell jobs

Managed shell jobs (`bash_start`, `bash_read`, `bash_stop`, `bash_list`) are
processes, not background agent runs. `cancel_agent`, `wait_for_agent`, and the
agent browser manage agent runs, not these jobs. A child can access only jobs
created by its own runtime incarnation; guessing a root or sibling job ID
grants no access. A resumed child does not inherit its previous incarnation's
job access. The root can list, read, and stop all jobs, including those created
by children.

All children share the root's eight job allocations; there is no separate
per-child capacity pool. Completion metadata carries bounded, redacted job
summaries and handles for foreground and background child runs. Committed jobs
survive child completion, cancellation, eviction, and release. Recover them from
the root with `bash_list`, then use `bash_read` or `bash_stop` as needed, even if
the child runtime or its stored result is no longer available.

While jobs or pending launches retain execution authority, authority-reducing
shell policy/filter, workspace/scratchpad, credential-environment, and relevant
retained-child reconfiguration changes are rejected before publication. Jobs
whose creator was evicted still count. Finished retained records do not impose
this restriction; unrelated display/model/status-line edits remain available.
Stop active jobs before reducing authority. The root's lifetime, not the
child's retention policy, controls job cleanup; see [Tools and
safety](tools-safety.md#managed-shell-jobs).

## Active-work admission cap

By default, at most 16 subagent runs can be active under a root session. The cap
counts foreground and background work, including retasking idle agents and
pending child creation; idle retained agents do not count. A new launch or idle
reuse at capacity is rejected immediately rather than queued. A busy
replacement with `replace_run=True` proceeds at capacity: it reserves the old
run's existing slot through cleanup and transfers it to the replacement, adding
no concurrency. Unrelated launches cannot take that reserved slot.

Set the cap in `config.toml`:

```toml
[subagents]
max_running_subagents = 16
```

The value must be a strict positive integer (at least 1); booleans, strings,
and fractional values are invalid in TOML. The environment override is
`CHARTREUX_SUBAGENTS__MAX_RUNNING_SUBAGENTS`. This setting is deliberately not
available in the settings UI.

The root session's effective configuration controls admission, not a child's
configuration. Accepted changes apply to later admissions without stopping
existing runs. Completion, cancellation, release, or failed creation frees
capacity. The active-work cap is separate from the resident-runtime and
Recent-metadata limits above; see the
[configuration reference](../reference/configuration.md#agents-and-skills).

## Parallel tasks

To run independent investigations in parallel, make separate background `task`
calls with explicit identities, purposes, or models. Each launched agent has its
own handle and result. The old `fan_out: true` flag is rejected with guidance to
launch separate tasks. See the [models guide](models.md) for role presets and
the [identity architecture](../reference/identity-architecture.md) for route
diversity semantics.

## TUI monitoring

The TUI shows a one-line agent summary above the input. The collapsed segment
counts active work only — running and finalizing instances, each once. Failed,
cancelled, and budget outcomes surface as completion and eviction notifications,
not status-line counts. With zero active agents, the bar shows a dim note
instead of hiding entirely: `N recent` when Recent entries exist, otherwise
`N idle` when retained-idle agents exist. Click the bar, type
`/agents`, or press `Ctrl+Shift+A` to open the docked agent browser.

The browser keeps **Main agent** pinned at the top, followed by subagents grouped
by lifecycle: **Active** (running, stopping, compacting, finalizing), **Retained
Idle**, and **Recent** (closed and evicted records with their outcomes). The
server orders each group deterministically — Active oldest launch first, Retained
Idle most recently completed first, Recent newest closure first — and the browser
consumes that order. Filter tabs **Resident** (the default: Active plus Retained
Idle), **Active**, **Idle**, **Recent**, and **All** show truthful counts;
`Left`/`Right` cycles them. Group headers are labels, never selections. When the
Resident view is empty but Recent entries exist, the browser shows the Recent
group with a note — never an empty screen. Its ten-row sheet scrolls internally.
Rows show the agent's name with the `agent_id` handle dimmed beside it, the
identity, and a static state marker: **Running**, **Compacting**, **Stopping**,
**Finishing** (with the run outcome), **Idle**, **Failed**, **Cancelled**,
**Budget stopped**, **Closed**, or **Evicted**. Compacting is a presentation
substate of active work, not a separate lifecycle state. Expired metadata and
released agents have no rows; a Recent row's details show **result preserved** or
**result expired** for its stored result.

Use `Up`/`Down` to select a row, then `Enter` or a single click to open its output.
Two detail rows show the selection's state, running or last-run duration, idle
duration where applicable, context usage, model, and turns. Running and idle
elapsed times are anchored to receipt of the server snapshot; closed and evicted
values remain frozen at the last recorded measurement. Context usage uses the
effective compaction threshold as its denominator, not the model's maximum
context window.

Press `D` for scrollable full metadata, including the agent's `Name` and
explicitly labeled `Agent ID`, task, provider, model, thinking, run ID, stop
reason, and retention information. Details also works on **Main agent**. `F1`
opens local help; `Escape` returns from help, then details, then closes the
browser and restores input focus. The footer lists available actions.
`/agents` and `Ctrl+Shift+A` toggle the browser.

On a highlighted **Running** or **Compacting** child with a current run ID,
press `C` to open inline **[Stop run] [Cancel]** controls. The confirmation names
the agent and captured run and explains retention-qualified preservation.
**Cancel** has default focus; use `Tab`, arrow keys, or a click to choose an
action, then `Enter` to activate it. `PageUp`/`PageDown` scroll the scope text.
`Escape` dismisses the confirmation first, without closing the browser or
interrupting the parent. Dismissal and submission return focus to the browser
list; closing inspection restores the conversation's focus and reading position.

Choosing **Stop run** sends only that pinned run and shows **Stopping** until
an authoritative update settles it; repeat stops are disabled while pending.
Accepted or already-pending stops do not immediately claim **Cancelled**.
Inactive or unknown targets clear the pending display; finishing targets reconcile
to their actual state. Forbidden requests and transport failures surface an error
toast while leaving retained output inspectable. Updates that terminate, remove,
or replace the captured run invalidate the confirmation or pending display; late
responses cannot mark a replacement run as Stopping. There is no `T` retask key:
ask the orchestrator in prose to use `task(..., replace_run=True)`.

On a highlighted **Idle** or **Recent** entry, press `R` for a confirmed
**[Release] [Cancel]** action: the confirmation names the agent and explains that
release discards its retained results and browser access; **Cancel** has default
focus. Release is the browser counterpart of `release_agent`, guarded against
races: an agent that started running, is reserved for replacement, or changed its
latest run reports a conflict instead of being released. Release keeps the
lifecycle guards (it is not exempt during root lifecycle work), and it is
available for resident idle and Recent entries; **Stop** (`C`) stays for active
entries. Both confirmations use the same keyboard pattern (`Tab` or arrow keys to
choose, `Enter` to activate, `Escape` to dismiss).

Selecting an agent replaces the conversation area with a bordered pane titled
`Subagent: <name> (<agent_id>) · <identity>`. Running or finalizing agents show a
live, append-only transcript refreshed about once per second; the view uses the
same entry structure as the saved transcript, so it converges when the run
finishes. Older pages do not jump during refresh: use `PageUp` to paginate and
`r` for a manual refresh. The pane identifies saved transcripts when the agent
is no longer running. Press `Escape`, or select **Main agent**, to return to the
conversation. Main-agent activity does not close inspection; releasing an agent
closes its pane, and transcript access ends with the metadata window, while a
closed or evicted agent remains inspectable within that window when its saved
transcript is available.

Completion notifications still arrive in the parent session; use the result
tools to consume the machine-readable outcome.

## Upgrading from the retired profile axis

The v0.5.6 profile surfaces are retired and fail loudly with successor
guidance; nothing is silently migrated:

- The `agents/*.toml` profile directory convention is retired: those files are
  detected and reported by `chartreux doctor`, never loaded. Custom identities
  are prompt files in `~/.chartreux/prompts/` instead.
- The task tool's `agent_type` field is removed; use `identity` (the shipped
  profile names map onto identities: `worker` → `coder`, `advisor` →
  `advisor`, `reviewer` → `reviewer`).
- Profile-level `retain` and `idle_ttl_seconds` settings are gone; retention is
  the global `[subagents].retain_agents` default plus the per-launch `retain`
  argument.
- Project-level prompt directories no longer supply identities; custom
  resources are user-level only.

See [Upgrading from 0.5.6](../reference/identity-architecture.md#upgrading-from-056)
for the full retirement list.
