# Subagents

Use subagents to delegate bounded work or run independent investigations in
parallel. The primary agent remains responsible for the conversation and can
continue working while subagents run.

## Launching work

The `task` tool selects an agent type by profile name through `agent_type` and
creates a new agent instance when `agent_id` is omitted. Chartreux ships three
built-in agent types: `worker`, `advisor`, and `reviewer`. A new task without an
`agent_type` name uses the default `worker` profile. There is no built-in
`explore` profile; `explore` remains a system-prompt ID.

An agent instance's retained identity is its `agent_id`. Handles of the form
`agent-N` (digits after `agent-`) are reserved for instances and cannot be profile
names; supplying one as `agent_type` is rejected before dispatch. Unknown profile
names report `Unknown agent_type profile`, while unknown retained handles report
`Unknown agent_id instance handle`. The old Task field `agent` is not accepted.

For the built-in role profiles, dispatch the task directly. Their role prompts
already contain the noninteractive subagent contract and role guidance:

```text
task(task="Implement the bounded change in the issue", agent_type="worker")
task(task="Recommend an approach and identify risks", agent_type="advisor")
task(task="Review the authentication changes", agent_type="reviewer")
```

Workers dispatch skill-first. When the task names a skill, the worker loads it
first and follows its methodology and output format; otherwise the worker
selects the applicable task skill itself (`sub-implementor` for edits,
`sub-finder` for searches, `sub-verifier` for verification, `sub-explorer` for
exploration). Name a skill only when a specific methodology or output format is
required; the role profile and the task skill serve different purposes:

```text
task(task="Load the sub-finder skill and locate all callers of the parser.", agent_type="worker")
```

A task launches in the background by default (`background: true`). The call
returns a successful launch acknowledgment with `status: "launched"`, non-empty
guidance, and stable `agent_id` and `run_id` handles rather than waiting for the
answer; it is not the subagent's terminal result. Set `background: false` only
when the parent must have the result before it can proceed. Subagents cannot
launch other subagents.

## Built-in profiles and role prompts

The built-in profiles are presets for common delegated work:

| Profile | Model role | Prompt ID | Use it for |
| --- | --- | --- | --- |
| `worker` | `medium` | `worker` | General-purpose bounded implementation. |
| `advisor` | `large` | `advisor` | Independent architectural guidance, second opinions, and risk analysis. |
| `reviewer` | `medium` | `reviewer` | Independent read-only reviews of code, documentation, specifications, and plans. |

`advisor` is restricted to the read-only tools `read_file`, `grep`,
`web_search`, `web_fetch`, and `skill`. It is configured with no idle-TTL eviction so it
can be retained across related design and planning iterations. The normal idle
cap and explicit `release_agent` lifecycle still apply.

Each role prompt includes the shared noninteractive contract plus guidance for
that role. A profile's built-in prompt can be customized by placing a prompt
file with the same ID in a trusted project prompt directory or the user prompt
directory: custom prompts take precedence over the shipped prompt, with the
project prompt taking precedence over the user prompt. Likewise, a local TOML
profile with a built-in name overrides that built-in profile according to
profile discovery precedence. This lets users replace a profile or its prompt
without changing Chartreux's shipped defaults.

Profiles and dispatch slots are different concepts. A profile supplies an agent's
prompt, tools, and defaults; a slot names a launch binding (profile plus a model
role) and the purposes it serves. Roles remain saved model-and-thinking pairs.
Fresh sessions default to `standalone`: the main assistant implements within
approved scope, while verification and independent review remain delegated.
All approval gates still apply. Select `orchestrated` explicitly in user
`models.toml` (or the presets screen):

```toml
[dispatch]
mode = "orchestrated"
```

This delegates implementation too, even with one model. Legacy tier entries do
not select a mode. The rendered slot table and purpose vocabulary are
authoritative for routing. A
single-model roster renders task-kind routing without tier names; multi-model
orchestrated routing retains the compatibility guidance. Saved dispatch changes
apply to the next session, not the running session.

## Dynamic launch configuration

The following example uses a runtime configuration override rather than a
profile-specific default:

```text
task(
  task="Review the authentication changes",
  config={
    model="example-model",
    thinking="max",
    instructions="Focus on authorization boundaries.",
    enabled_tools=["read_file", "grep"]
  }
)
```

At launch or when retasking an idle subagent, `config` can override the model,
thinking level, enabled and disabled tools, and a tool's permission or
allowlist. Additional instructions are passed at the task level. Overrides are
session-local; they do not modify a profile or `config.toml`. The launch persona
(system prompt and base instructions) is immutable for a retained subagent, so
a changed persona requires launching a new subagent. A subagent cannot gain
tool authority beyond its parent. Retasking is available only for background
subagents.

Subagents can reread exact instruction files injected into their context,
including `AGENTS.md`, using `read_file` or exact-file `grep` even outside the
workspace roots. This does not grant directory searches or write access, and
tool denials and sensitive-file protections still apply. See
[Privacy](../project/privacy.md#instruction-file-access) for the data boundary.

Subagents inherit access to the session scratchpad through their parent-authority
chain rather than receiving a new workspace grant. File tools and shell checks
resolve that access against the ancestor scratchpad roots. Broken authority
chains, symlink escapes, and retargeted roots fail closed; tool denials and
sensitive-file protections still apply.

## Lifecycle and results

A background subagent starts **running**, may be **compacting**, then passes
through **finishing** while cleanup and result publication settle. It becomes
**idle** and remains available for inspection or reuse, subject to retention
policy. The browser shows the last run's outcome, including **Cancelled**, even
when the retained agent is idle. **Stopping** is a local TUI presentation state
for a pending stop, not a registry availability state or a terminal outcome.
The parent receives a completion notification that points it to
`get_agent_result`.

A would-be successful run without final prose is reported as **Failed**, with
`completed: false` and stop reason `error`, not as an empty successful result.
Empty model responses get at most one bounded replay; terminal failures use
`EmptyLLMResponseError` or `IncompleteLLMResponseError`. Published streaming
output is never replayed automatically.

If the parent is only waiting on subagents, submitting a new message in the TUI
cancels its waits, not the child runs, and steers the same parent turn immediately.
See [Input and queueing](terminal.md#input-and-queueing).

- `check_agents` lists retained subagents, including status and effective model
  and thinking information.
- `get_agent_result(agent_id, run_id)` returns a completed result without
  waiting; it returns no result while the run is active.
- `wait_for_agent(agent_id, run_id, timeout=...)` waits for completion. A
  timeout only stops the wait; it does not cancel the subagent.
- Pass an idle agent instance's `agent_id` to
  `task(agent_id=..., background=true, task=...)` to give it another assignment;
  omit `agent_type` to retain its profile. Both launch acknowledgments and
  `check_agents.reuse_guidance` include this reuse guidance.
- `release_agent(agent_id)` closes and removes a retained subagent when it is no
  longer useful.

Idle subagents are retained by default for 3,600 seconds, with at most 16 idle
subagents kept. TTL and idle-cap eviction remove the runtime but preserve a
tombstone and stored results, so `get_agent_result` remains usable after
eviction. Result expiry is separate: each root generation retains at most 32
unreferenced stored results. Release subagents when you are done.

### Cancelling and replacing background work

The orchestrator can stop one run with `cancel_agent(agent_id, run_id=None)`.
Omitting `run_id` targets the current run; supplying it pins the request to that
run and cannot stop a newer assignment. This tool manages only the
orchestrator's own background agents. Foreground work uses the normal parent
interruption controls.

A response of `stop_requested` means the stop was accepted, not that cleanup is
finished. `already_stopping` means a stop is already pending;
`already_finishing` preserves work that has finished executing. `not_running`,
`unknown_run`, and `forbidden` report an inactive, unknown, or unauthorized
target. Use `wait_for_agent` to wait for the terminal result. Cancellation does
not interrupt the parent or sibling agents and does not release the agent.

Cancelled results carry `completed: false`, accumulator partial output, and a
`stop_reason`: `user_cancelled`, `orchestrator_cancelled`, or `retasked` for these
operations. The first accepted stop reason wins. A genuine completion or error
is not rewritten by a late stop. Stopping does not undo file, shell, or remote
side effects, generate a summary, or salvage additional output.

The retained conversation, transcript, partial result, and existing waits remain
available subject to retention policy. Zero-retention agents evict at
finalization; other agents may evict on TTL or idle-cap limits. Eviction keeps
stored results until separate result expiry. Saved transcript inspection depends
on session logging and disk availability. Explicit `release_agent` removes the
agent and purges its stored results and wait leases.

To replace a busy assignment in the same retained conversation:

```text
task(agent_id="<agent-id>", task="Investigate the revised requirement", replace_run=True)
```

`replace_run=True` requires an `agent_id` and background mode. It stops the old
run with reason `retasked`, waits for its cleanup, and launches the replacement
without overlapping execution. The replacement prompt states that it supersedes
the interrupted task; the acknowledgment metadata includes both run IDs.
Busy replacement cannot change the profile or launch configuration. An idle
target uses ordinary reuse, which still permits runtime configuration overrides.
Ordinary reuse of a busy agent without `replace_run=True` is rejected. Already
stopping or finishing targets are not replaced; if replacement admission fails,
the old run remains stopped and the error is reported.

A user stop notifies the parent as an injected message **without starting a
parent turn**. An idle parent learns of the cancellation on its next turn. The
notification attributes the cancellation to the user; orchestrator cancellation
and retask are attributed separately.

## Managed shell jobs

Managed shell jobs (`bash_start`, `bash_read`, `bash_stop`, `bash_list`) are
processes, not background agent runs. `cancel_agent`, `wait_for_agent`, and the
agent browser manage agent runs, not these jobs. A child can access only jobs
created by its own runtime incarnation; guessing a root or sibling job ID grants
no access. A resumed child does not inherit its previous incarnation's job access.
The root can list, read, and stop all jobs, including those created by children.

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
Stop active jobs before reducing authority. The root's lifetime, not the child's
retention policy, controls job cleanup; see [Tools and safety](tools-safety.md#managed-shell-jobs).

## Active-work admission cap

By default, at most 16 subagent runs can be active under a root session. The cap
counts foreground and background work, including retasking idle agents and
pending child creation; idle retained agents do not count. A new launch or idle
reuse at capacity is rejected immediately rather than queued. A busy replacement
with `replace_run=True` proceeds at capacity: it reserves the old run's existing
slot through cleanup and transfers it to the replacement, adding no concurrency.
Unrelated launches cannot take that reserved slot.

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
capacity. The active-work cap is separate from the idle-retention limits above;
see the [configuration reference](../reference/configuration.md#agents-and-skills).

## Parallel tasks

A role selects one model and thinking level. To run independent investigations
in parallel, make separate background `task` calls with explicit roles or
models. Each launched agent has its own handle and result. The old
`fan_out: true` flag is rejected with guidance to launch separate tasks. See
the [configuration reference](../reference/configuration.md) for role presets.

## TUI monitoring

The TUI shows a one-line agent summary above the input with aggregate counts by
state. Click it, type `/agents`, or press `Ctrl+Shift+A` to open the docked
agent browser.

The browser keeps **Main agent** pinned at the top, followed by retained agents
in stable order. Its ten-row sheet scrolls internally. Rows show identity,
profile, and a static state marker: **Running**, **Compacting**, **Stopping**, **Finishing**
(with the run outcome), **Idle**, **Failed**, **Cancelled**, **Budget stopped**,
or **Evicted**. Compacting is a presentation substate of active work, not a
separate lifecycle state. Released agents disappear; evicted tombstones remain
browsable with their stored-result metadata.

Use `Up`/`Down` to select a row, then `Enter` or a single click to open its output.
Two detail rows show the selection's state, running or last-run duration, idle
duration where applicable, context usage, model, and turns. Running and idle
elapsed times are anchored to receipt of the server snapshot; evicted values
remain frozen at the last recorded measurement. Context usage uses the effective
compaction threshold as its denominator, not the model's maximum context window.

Press `D` for scrollable full metadata, including task, provider, model, thinking,
run ID, stop reason, and retention information. Details also works on **Main
agent**. `F1` opens local help; `Escape` returns from help, then details, then
closes the browser and restores input focus. The footer lists available actions.
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

Selecting an agent replaces the conversation area with a bordered pane titled
`Subagent: <id> · <profile>`. Running or finalizing agents show a live,
append-only transcript refreshed about once per second; the view uses the same
entry structure as the saved transcript, so it converges when the run finishes.
Older pages do not jump during refresh: use `PageUp` to paginate and `r` for a
manual refresh. The pane identifies saved transcripts when the agent is no
longer running. Press `Escape`, or select **Main agent**, to return to the
conversation. Main-agent activity does not close inspection; releasing an
agent closes its pane, while an evicted agent remains inspectable when its saved
transcript is available.

Completion notifications still arrive in the parent session; use the result
tools to consume the machine-readable outcome.

## TOML profiles

Dynamic `task` configuration and TOML profiles coexist. The task tool creates
subagents dynamically and accepts runtime overrides; profiles provide defaults
that the orchestrator can apply when a named profile is launched.

Chartreux discovers `*.toml` profiles from configured `agent_paths`, trusted
project agent directories, and user agent directories. A custom profile may
override the `worker` name. For example, save this as
`~/.chartreux/agents/reviewer.toml`:

```toml
display_name = "Reviewer"
description = "Read-only review work"
agent_type = "subagent"
instructions = "Report concrete findings with file and line references."
role = "medium"
disabled_tools = ["edit", "write_file"]

[tools.bash]
permission = "always"
```

The TOML `agent_type = "subagent"` field classifies the profile internally; it is
not the Task `agent_type` argument, which takes the profile name (`reviewer` in
this example). Profile names matching `agent-N` are reserved and rejected during
discovery.

Profile fields are validated during discovery; an invalid profile is not made
available. Use `role` to bind a profile to a role; `active_model` is rejected in
profile TOML. A retained subagent keeps its committed model and thinking level
when reused, even if its profile's role has since been edited, unless the task
supplies an explicit model or thinking override. Project profiles are subject to the same trusted-project
rules as other local configuration. See [configuration](configuration.md) for the
configuration model and [tools and safety](tools-safety.md) for tool policy.
