Launch a subagent for bounded delegated work of any size. Use `task` to delegate when a specialist is useful; do not treat delegation as limited to exceptional work. Provide a self-contained description and an optional `task_summary`: a concise action plus subsystem or feature (up to 240 characters) used to identify retained agents in `check_agents`. By default it runs in the background and returns a successful launch acknowledgment with `status: "launched"`, non-empty guidance, and `agent_id` and `run_id` handles immediately — it is not the subagent's terminal result, so you can continue working while the subagent runs.

Create a new agent instance with `task(agent_type="worker", task=...)`: `agent_type` selects an agent type by profile name (default `worker`), not a model or an instance handle. Handles of the form `agent-N` (digits after `agent-`) are reserved for agent instances and rejected in `agent_type` before dispatch. An unknown profile reports `Unknown agent_type profile`; an unknown retained handle reports `Unknown agent_id instance handle`. The launch acknowledgment includes guidance to continue the instance with `task(agent_id=..., background=true, task=...)` once it is idle, omitting `agent_type`.

Set `background: false` only when you need the result before proceeding; the subagent blocks until completion and returns its final message.

Use `config` for per-launch configuration: `model` (a single model expression: canonical name or `@role`), `instructions`, `system_prompt_id`, `thinking`, `enabled_tools`, `disabled_tools`, and `tools` (per-tool `permission`/`allowlist`). These typed fields are validated; they cannot expand the parent agent's effective tool authority. `config` is call-level state, not a `[task.config]` setting: it is not written to user or project configuration, but the child commits its resolved base model and concrete provider deployment in session metadata; resume revalidates that identity rather than re-resolving a role.

A child completion may fail over between providers for its committed base model before producing semantic output. Inspect `metadata.switch_notices` for each switch's base model, old and new provider, and reason, and `providers_used` for providers involved; failover never changes the child's committed base model.

To continue genuine work on the same feature, first use `check_agents` to inspect each idle agent's effective model, thinking, and retained context. Reuse an idle handle with `agent_id` and `background: true`; omit `agent_type` to retain its profile. Omit `config`, use `{}`, or omit nested fields to retain effective launch choices; supplied lists replace earlier launch-layer lists and supplied per-tool fields patch that tool. Re-task the same idle agent for a genuine continuation, with a different `model`, `thinking`, or tool configuration when the work calls for it. A retained agent keeps its last model and thinking unless overridden; re-state the tier explicitly on every reuse, so substantive work on a previously `@small` agent runs at `@medium`. Persona is immutable on reuse: `instructions` and `system_prompt_id` may only repeat their existing effective values. Launch a new agent when its persona, role, or stack must change, or when work is unrelated or needs independent judgment. Include the current scope and intervening changes rather than restating its full context. A retained conversation is not proof that the working tree is unchanged. A running agent is busy; wait when its result is a dependency, and spawn another only for independent, non-conflicting work. An evicted agent cannot be reused; retrieve its retained result with `get_agent_result` if needed, then start a new agent.

For example, launch a deep review with `config: {"model": "@large", "instructions": "Review carefully.", "enabled_tools": ["read_file"]}`. For a second-round deep review, launch a fresh reviewer with a `@large` model override — `task(agent_type="reviewer", task="Review the retry changes.", config={"model": "@large"})` — rather than re-tasking an idle implementor: a fresh reviewer must not inherit an implementor's persona or history.

Route by the work, not the profile default: pass the tier explicitly in `config`. Mechanical delegated work — single-file edits, bounded searches, verification runs — launches with a model override, for example `task(agent_type="worker", task="Rename add to plus in utils.py and update its call sites.", config={"model": "@small"})`. Substantive implementation launches at the `@medium` worker default with no override, for example `task(agent_type="worker", task="Add a retry helper with exponential backoff to utils.py and use it in app.py.")`. `@large` launches are for architecture, design, planning, and deep review only — never implementation.

### Authoritative active slot bindings
The active table overrides compatibility examples and profile defaults. Pass the model and thinking explicitly; unavailable slots fail closed.

| Slot | Profile | Role | Model | Thinking | Purposes | Implements | Review eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mechanical` | `worker` | `@small` | `alpha-model` | low | search, exploration, verification, mechanical-edit | routine | False |
| `implementor` | `worker` | `@medium` | `alpha-model` | medium | implementation | routine | False |
| `escalation-implementor` | `worker` | `@medium` | `alpha-model` | medium | implementation-demanding-settled | escalation | False |
| `advisor` | `advisor` | `@large` | `beta-model` | high | design-analysis, planning-analysis | never | False |
| `reviewer` | `reviewer` | `@medium` | `alpha-model` | medium | review.quick, review.standard | never | True |
| `analytical-reviewer` | `reviewer` | `@large` | `beta-model` | high | review.deep | never | True |
| `peer-reviewer` | `reviewer` | `@large` | `beta-model` | high | review.deep | never | True |
| `execution-reviewer` | `reviewer` | `@medium` | `alpha-model` | medium | review.deep | never | True |

### Purpose meanings
- `search`: Bounded searches, grep, and symbol or reference lookups.
- `exploration`: Targeted code investigation and project exploration.
- `verification`: Run proportionate tests, builds, and other verification; report only checks actually run.
- `mechanical-edit`: Mechanical single-file edits with a known, bounded transformation.
- `implementation`: All substantive implementation, including novel algorithmic reasoning, difficult refactoring, and broad-impact work.
- `implementation-demanding-settled`: Demanding execution with a settled approach; a proactive escalation-implementor route, not only a failure-triggered route. State the reason for this route.
- `design-analysis`: Architecture, cross-subsystem design, design refinements, and destructive-operation analysis; read-only advice, never implementation. Keep an advisor across related refinements.
- `planning-analysis`: Analyze an approved design into bounded steps, dependencies, and acceptance checks; read-only advice, never implementation.
- `review.quick`: A quick independent judgment in a fresh reviewer context, never the author or a reused advisor.
- `review.standard`: Independent review against the approved design, plan, and acceptance checks; use a fresh reviewer, never the author. Do not give second-round reviewers earlier conclusions.
- `review.deep`: Deep independent review through the configured composition of fresh reviewer contexts, with authorship-aware substitution. Never reuse an advisor as a reviewer or give second-round reviewers earlier conclusions; fail closed when a required independent seat is unavailable.

### Review compositions
review.quick and review.standard: use slot `reviewer` in a fresh context, never the author.
review.deep: use slots `analytical-reviewer`, `peer-reviewer`, and `execution-reviewer` in fresh contexts. If `execution-reviewer` authored the work, substitute slot `reviewer`; if `reviewer` authored the work, substitute slot `execution-reviewer`. Never reuse an advisor as a reviewer or give a second-round reviewer earlier conclusions.
For one canonical model, including multiple thinking levels, use fresh-context review on the available model instead of claiming model diversity. Fail closed if a required independent slot is unavailable; never silently substitute.

### Failure routes
Select a tier afresh for each task. Uncertainty, file count, session length, or wanting a better answer are not escalation criteria. If an implementation attempt fails, diagnose the failure, retry at `@medium` with a different approach, or dispatch a `@large` advisor for read-only analysis; never re-dispatch implementation to `@large`. Escalate architectural blockers to an advisor and authorization or scope blockers to the user.

### Contrasts
Pass the tier explicitly — the profile default is not the routing decision. Mechanical work (single-file edits, bounded searches, verification runs) launches with a `config` model override: `task(agent_type="worker", task="Rename add to plus in utils.py and update its call sites.", config={"model": "@small"})`. Substantive implementation launches at the `@medium` worker default with no override: `task(agent_type="worker", task="Add a retry helper with exponential backoff to utils.py and use it in app.py.")`. `@large` launches are for architecture, design, planning, and deep review only — never implementation.

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
