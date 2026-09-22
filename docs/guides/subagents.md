# Subagents

Use subagents to delegate bounded work or run independent investigations in
parallel. The primary agent remains responsible for the conversation and can
continue working while subagents run.

## Launching work

The `task` tool uses a named agent profile when `agent` is supplied. Chartreux
ships three built-in role profiles: `worker`, `advisor`, and `reviewer`. A task
without an `agent` name uses the default `worker` profile. There is no built-in
`explore` profile; `explore` remains a system-prompt ID.

For the built-in role profiles, dispatch the task directly. Their role prompts
already contain the noninteractive subagent contract and role guidance; do not
add a role-specific skill-loading instruction:

```text
task(task="Implement the bounded change in the issue", agent="worker")
task(task="Recommend an approach and identify risks", agent="advisor")
task(task="Review the authentication changes", agent="reviewer")
```

Task-type skills remain explicit. For example, a search assignment can ask the
worker to load `sub-finder`; the role profile and the task skill serve different
purposes:

```text
task(task="Load the sub-finder skill and locate all callers of the parser.", agent="worker")
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
| `worker` | `small-worker` | `worker` | General-purpose bounded implementation or miscellaneous work. |
| `advisor` | `advisor` | `advisor` | Independent architectural guidance, second opinions, and risk analysis. |
| `reviewer` | `medium-reviewer` | `reviewer` | Independent read-only reviews of code, documentation, specifications, and plans. |

`advisor` is restricted to the read-only tools `read_file`, `grep`,
`web_search`, and `web_fetch`. It is configured with no idle-TTL eviction so it
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

Use the profile's role binding for model selection, for example
`config={model="@advisor"}` or `config={model="@medium-reviewer"}` when a
specific role tier is required. Runtime configuration is an override for that
launch; it does not change the profile or its role prompt.

## Dynamic launch configuration

The following example uses a runtime configuration override rather than a
profile-specific default:

```text
task(
  task="Review the authentication changes",
  config={
    model="gpt-5.6-terra",
    thinking="high",
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

## Lifecycle and results

A background subagent is first **running**. When its run reaches a terminal
outcome it becomes **idle** and remains available for inspection or reuse.
The parent receives a completion notification that points it to
`get_agent_result`.

- `check_agents` lists retained subagents, including status and effective model
  and thinking information.
- `get_agent_result(agent_id, run_id)` returns a completed result without
  waiting; it returns no result while the run is active.
- `wait_for_agent(agent_id, run_id, timeout=...)` waits for completion. A
  timeout only stops the wait; it does not cancel the subagent.
- Pass an idle `agent_id` to `task` to give that subagent another assignment.
- `release_agent(agent_id)` closes and removes a retained subagent when it is no
  longer useful.

Idle subagents are retained by default for 3,600 seconds, with at most 16 idle
subagents kept. TTL and idle-cap eviction remove the runtime but preserve a
tombstone and stored results, so `get_agent_result` remains usable after
eviction. Result expiry is separate: each root generation retains at most 32
unreferenced stored results. Release subagents when you are done.

## Parallel fan-out

For an explicit model role, `fan_out: true` starts one retained subagent per role
member. Each member has its own handle and outcome; the ordered member results
identify the selected model and provider. Fan-out does not substitute a failed
member or cancel its siblings. See the [configuration reference](../reference/configuration.md)
for model roles.

## TUI monitoring

The TUI shows a one-line background-agent statusline above the input. It includes
an activity spinner while agents are running and aggregate counts by state (for
example, `3 agents: 2 running · 1 idle`). Click the statusline, type `/agents`, or
press `Ctrl+Shift+A` to expand it into an in-place list.

The expanded list is keyboard- and mouse-navigable. It keeps a pinned **Main
agent** entry at the top, followed by one row per retained agent. Rows show the
agent ID, profile, status, model, turns used for the current run, run ID, and
idle/TTL information. Released agents are removed; evicted tombstones remain
browsable with their stored-result metadata. The list is capped to roughly ten
rows and scrolls internally when more agents are available. Use `Up`/`Down` to
move the selection and `Enter` to open it. `/agents` and `Ctrl+Shift+A` toggle the
list; collapsing it restores focus to the chat input.

Selecting an agent replaces the conversation area with a bordered pane titled
`Subagent: <id> · <profile>`. Running or finalizing agents show a live,
append-only transcript refreshed about once per second; the view uses the same
entry structure as the saved transcript, so it converges when the run finishes.
Older pages do not jump during refresh: use `PageUp` to paginate and `r` for a
manual refresh. The pane identifies saved transcripts when the agent is no
longer running. Press `Escape`, or select **Main agent**, to return to the
conversation. A new main-agent turn closes the pane automatically; releasing an
agent closes its pane, while an evicted agent remains available from disk.

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
role = "medium-reviewer"
disabled_tools = ["edit", "write_file"]

[tools.bash]
permission = "ask"
```

Profile fields are validated during discovery; an invalid profile is not made
available. Use `role` to bind a profile to a role; `active_model` is rejected in
profile TOML. A retained subagent keeps its committed model when reused, even if
its profile's role has since been edited, unless the task supplies an explicit
`config.model` override. Project profiles are subject to the same trusted-project
rules as other local configuration. See [configuration](configuration.md) for the
configuration model and [tools and safety](tools-safety.md) for tool policy.
