Launch a subagent for bounded delegated work of any size. Use `task` to delegate when a specialist is useful; do not treat delegation as limited to exceptional work. Provide a self-contained description and an optional `task_summary`: a concise action plus subsystem or feature (up to 240 characters) used to identify retained agents in `check_agents`. By default it runs in the background and returns a successful launch acknowledgment with `status: "launched"`, non-empty guidance, and `agent_id` and `run_id` handles immediately — it is not the subagent's terminal result, so you can continue working while the subagent runs.

Create a new agent instance with `task(agent_type="worker", task=...)`: `agent_type` selects an agent type by profile name (default `worker`), not a model or an instance handle. Handles of the form `agent-N` (digits after `agent-`) are reserved for agent instances and rejected in `agent_type` before dispatch. An unknown profile reports `Unknown agent_type profile`; an unknown retained handle reports `Unknown agent_id instance handle`. The launch acknowledgment includes guidance to continue the instance with `task(agent_id=..., background=true, task=...)` once it is idle, omitting `agent_type`.

Set `background: false` only when you need the result before proceeding; the subagent blocks until completion and returns its final message.

Use `config` for per-launch configuration: `model` (a single model expression: canonical name or `@role`), `instructions`, `system_prompt_id`, `thinking`, `enabled_tools`, `disabled_tools`, and `tools` (per-tool `permission`/`allowlist`). These typed fields are validated; they cannot expand the parent agent's effective tool authority. `config` is call-level state, not a `[task.config]` setting: it is not written to user or project configuration, but the child commits its resolved base model and concrete provider deployment in session metadata; resume revalidates that identity rather than re-resolving a role.

A child completion may fail over between providers for its committed base model before producing semantic output. Inspect `metadata.switch_notices` for each switch's base model, old and new provider, and reason, and `providers_used` for providers involved; failover never changes the child's committed base model.

$dispatch_reuse

$dispatch_review

$dispatch_routing

To supersede busy work, set `replace_run: true` with `agent_id` and `background: true`. Only the owning parent may replace a run. Busy replacement forbids `config` and profile changes; idle reuse still supports launch reconfiguration. The tool stops the old run, joins its cleanup and result publication, then launches in the same conversation, automatically prepending: "This task supersedes the interrupted task." Its existing capacity slot is held through the handoff, so replacement works at capacity. The acknowledgment has `launch_outcome: "launched"` and `metadata.replaced_run_id` / `metadata.replacement_run_id`. Refusals have `launch_outcome: "already_stopping"`, `"already_finishing"`, or `"rejected_reservation"`, not a successful launch status. Retrieve the old run's partial result by its ID; cancellation does not undo side effects. If replacement admission fails after stopping, the old run remains terminal, not resumed. On an idle agent, `replace_run: true` is ordinary reuse with no stop or supersession frame.

Use `cancel_agent(agent_id, run_id)` to stop background work without replacing it. Acceptance means stop requested, not terminal cancellation. Use `check_agents` to list retained agents, `get_agent_result(agent_id, run_id)` to retrieve a completed result without waiting, `wait_for_agent(agent_id, run_id, timeout)` to wait for completion, and `release_agent(agent_id)` to release an agent when its work is complete. Releasing an agent discards its retained results and identity—retrieve results before releasing. Background agents cannot be launched from subagents.

A slot binds a profile, a role, and purposes. A slot's role is one default model and thinking level. An explicit
`config.thinking` overrides that level for this launch. If a role's model or
credential is unavailable, repair the preset or credential; no other canonical
model is substituted. To run independent agents in parallel, make separate
background `task` calls with explicit presets or models, and collect each
result by its handle. The retired `fan_out: true` flag returns: “Roles are
single presets. Launch separate tasks with explicit presets/models for
multiple agents.”
