# Subagents

Use subagents to delegate bounded work or run independent investigations in
parallel. The primary agent remains responsible for the conversation and can
continue working while subagents run.

## Launching work

The `task` tool launches the neutral built-in `worker` preset unless an
`agent` profile is named. `worker` deliberately supplies no model, tool, or
persona settings, so it inherits the parent session's effective configuration.
There is no built-in `explore` subagent.

A task launches in the background by default (`background: true`). The call
returns stable `agent_id` and `run_id` handles rather than waiting for the
answer. Set `background: false` only when the parent must have the result
before it can proceed. Subagents cannot launch other subagents.

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

For an explicit model tag, `fan_out: true` starts one retained subagent per tag
member. Each member has its own handle and outcome; the ordered member results
identify the selected model and provider. Fan-out does not substitute a failed
member or cancel its siblings. See the [configuration reference](../reference/configuration.md)
for model tags.

## TUI monitoring

Press `Ctrl+Shift+A` to open the background-agent sidebar. Select an agent and
press `Enter` to open its transcript. The viewer loads or refreshes the saved,
paginated transcript on demand, so a running agent's display can lag its most
recent activity. Completion notifications still arrive in the parent session;
use the result tools to consume the machine-readable outcome.

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
active_model = "gpt-5.6-terra"
disabled_tools = ["edit", "write_file"]

[tools.bash]
permission = "ask"
```

Profile fields are validated during discovery; an invalid profile is not made
available. Project profiles are subject to the same trusted-project rules as
other local configuration. See [configuration](configuration.md) for the
configuration model and [tools and safety](tools-safety.md) for tool policy.
