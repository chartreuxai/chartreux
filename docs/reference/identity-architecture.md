# Identity architecture

Since v0.5.7, Chartreux configures agents with three orthogonal selections — **identity**, **workflow**, and **orchestration strategy** — replacing the profile, dispatch-slot, and system-prompt-ID axes of earlier versions. This page documents the model: the selections, the deterministic prompt assembly, the frozen per-conversation lifecycle, the canonical task-call grammar, purpose routes, and the authority model. Exact key names and accepted values live in the [configuration reference](configuration.md).

## The three selections

| Concept | Question it answers | Owns |
| --- | --- | --- |
| Identity | Who performs the work | The main prompt, execution posture, delegation behavior |
| Workflow | What process and approval stages apply | Phases, gates, review requirements |
| Orchestration strategy | How delegated work is packaged and routed | Brief discipline, parallelism, packaging preferences |

```toml
[agent]
identity = "coder"                  # default
workflow = "minimal"                # default
orchestration_strategy = "none"     # default
```

All three are optional. New-conversation precedence, low to high, is shipped default, `config.toml`, environment, CLI flag — and an omitted flag is always distinguished from an explicit selection:

- CLI: `chartreux --identity <id>`, `--workflow <id>`, `--orchestration-strategy <id>`
- Environment: `CHARTREUX_AGENT__IDENTITY`, `CHARTREUX_AGENT__WORKFLOW`, `CHARTREUX_AGENT__ORCHESTRATION_STRATEGY`
- Config: the `[agent]` table; the Settings > Identity & Orchestration group edits the same keys

Each selection accepts a built-in name (bare, or `builtin:`-qualified), a `user:`-qualified custom id, or — for identity only, where the prompt file is the whole contract — a bare custom name that resolves to a prompt file in the user prompts directory. Malformed qualifications, empty components, and reserved shipped names fail loudly; reserved built-in names cannot be shadowed by custom files.

Selections — and the **content** of the selected resources — freeze at conversation start (see [Frozen lifecycle](#frozen-lifecycle)). Saved changes apply to new conversations only, and `/reload` reports them as pending.

## Zero-config solo default

A new conversation with no agent configuration at all runs the `coder` identity under the `minimal` workflow with no orchestration strategy. The default performs direct implementation itself, may delegate when a specialist is useful, and carries no forced review thresholds, acceptance gates, or model-tier doctrine. Only a provider credential and one configured model are required.

The default is model-count agnostic: shipped purpose routes all bind the session model, so default routing behaves identically whether the catalog holds one model or ten. No agent profiles, dispatch slots, or prompt IDs need to be configured — there is nothing to coordinate for a single-model setup.

## One-line orchestrator

```toml
[agent]
identity = "orchestrator"
```

or `chartreux --identity orchestrator`. The orchestrator must delegate: it may not perform mutating work itself (its personal tool set is non-mutating only) but can delegate any work within the session's granted authority. The delegation obligation lives in the identity prompt itself, never in a selectable strategy — the orchestrator is self-sufficient with `orchestration_strategy = "none"`.

The `ceremony` workflow (`workflow = "ceremony"`) and the shipped orchestration strategy (`orchestration_strategy = "builtin:default"`) remain selectable independently. The strategy body carries packaging and routing preferences only — brief discipline, parallelism, reuse — never ceremony, model-tier doctrine, or authority claims.

## Built-in identities

| Identity | Delegation | Personal tools | Delegation tools | Posture |
| --- | --- | --- | --- | --- |
| `coder` | optional | all granted | all granted | Direct execution; may delegate |
| `orchestrator` | required | non-mutating only | all granted | Must orchestrate from the root session |
| `researcher` | optional | non-mutating only | non-mutating only | Read-only analysis |
| `reviewer` | optional | non-mutating only | non-mutating only | Read-only analysis |
| `advisor` | optional | non-mutating only | non-mutating only | Read-only analysis |

The read-only postures are honest: `bash` classifies as mutating regardless of command text, and unclassified or custom tools (including MCP tools) classify as mutating by default, so a read-only identity genuinely excludes them. A read-only reviewer does not run shell-based verification; check-only verification is expressed through the execution-seat overlay instead (see [Purpose routes](#purpose-routes-seats-and-diversity)).

Custom identities carry no harness-known posture. Their prompt file is the whole contract, and their delegation set is the full granted ceiling of the session. An identity whose future typed posture declares `delegation = none` combined with an explicit orchestration strategy is a configuration error, not a silent no-op; no shipped identity sets `none`.

## Deterministic prompt assembly

The system prompt is assembled once, at startup, in a fixed nine-step order:

1. The identity opening, at character zero (a short, direct self-policy).
2. Universal critical instructions and the instruction hierarchy — harness-guaranteed; a custom identity cannot drop them.
3. Identity behavior and direct-tool posture.
4. Execution context: interactive main, headless main, or bounded child.
5. The selected workflow body.
6. The selected orchestration strategy body, only if one is selected; `none` is absent, and it resolves concretely without loading a strategy body.
7. Shared references: tool semantics, the rendered launch-target table, model catalog, skills catalog, communication style, scratchpad, environment.
8. AGENTS.md layers — discovery and injection unchanged; when the user and project layers resolve to the same file, the canonical duplicate renders once, keeping the higher-precedence (project) label with both provenance positions recorded.
9. A short closing re-anchor reinforcing the resolved contract, adding no new rules.

Gate and acceptance ceremony lives wholly in the `ceremony` workflow; the composed default (coder + minimal + none) carries none of it. Shipped workflow skills align with the workflow axis: ceremony-bearing skills (the `main-design`, `main-plan`, and `main-review` kin) load only for workflow/identity combinations that carry the ceremony, and the minimal default loads only identity-neutral skills. There are no runtime instruction-file reads for the assembled policy — everything is resolved and composed up front.

## Frozen lifecycle

Identity, workflow, and orchestration-strategy selection **and content** are frozen per conversation — a deliberate behavior change relative to earlier versions, where `/reload` and resume re-read custom prompt content from disk.

- **Startup** binds the conversation once to a policy bundle carrying the resolved resource content, not just hashes. No mixed generation is observable.
- **`/reload`** refreshes reloadable machinery — the tool list, skills catalog, model catalog, and AGENTS.md — and keeps the frozen identity/workflow/strategy bodies byte-for-byte. Differences between the saved and frozen state are reported explicitly as pending changes: a pending selection change states that the conversation stays bound and the saved selection applies to new conversations only; a pending resource content change (or a bound resource that no longer resolves on disk) states that the conversation keeps the frozen content.
- **Resume** restores the persisted bundle, including its content: a conversation resumes identically after the underlying prompt files have been edited or removed. `none` is recorded as its concrete resolution.
- **`/clear` and new conversations** compose a fresh bundle from the current selections; only resume consumes the persisted bundle.
- **Compaction** preserves the bound identity/workflow/strategy ids and revisions, the actual delegation precedent (never invented), the current phase and approval evidence, resident and pending agents with their dependencies, and settled decisions with consumed budgets. Absent delegation evidence is represented explicitly.

Sessions persisted before v0.5.7 fail resume with a loud diagnostic that says to start a new session; there is no silent migration (see [Upgrading from 0.5.6](#upgrading-from-056)).

## Canonical task calls

The `task` tool takes `model`, `identity`, `purpose`, and `seat` as top-level fields:

```
task(model="@fast", identity="researcher", purpose="research", task="...")
```

- **`model`** accepts a canonical model name, an `@role` preset, or the shorthand `@<preset-or-canonical-model>[:<identity>]` — `@fast:researcher` for a bare built-in identity, `@fast:user:foo` for a `user:`-qualified custom id. In the shorthand, the identity component accepts a bare built-in or a `user:`-qualified id; `builtin:` qualification is a structured-field-only form. A plain canonical model name is not shorthand.
- **`identity`** accepts a bare built-in (`coder`, `researcher`, `reviewer`, `advisor`, `orchestrator`), a `builtin:`- or `user:`-qualified id, or a bare custom prompt id.
- **`purpose`** selects a launch route (see below); **`seat`** picks one target of a multi-seat purpose and is meaningless without one.

Missing components are never guessed, and conflicts are rejected instead of resolved by precedence: supplying `model` together with `config.model` is a validation error (the top-level field is the canonical location), a shorthand identity component together with the structured `identity` field is rejected, and a `seat` without a `purpose` is rejected. Ambiguous colon-bearing model expressions are rejected with a pointer to the structured fields.

Defaults for a new child: identity `coder` with the bounded-child execution context, model the session model, no inherited workflow or strategy — children never inherit the parent's workflow or strategy bodies. Bounded children return blockers rather than escalating, never restart the parent's process, and cannot spawn nested agents: child launches are depth-limited to one, and background agents cannot be launched from subagents. Required-delegation identities (`orchestrator`) are rejected as children with an actionable error — they must orchestrate from the root session.

`agent_id` is the only instance handle. Handles of the form `agent-N` are reserved for agent instances and rejected as launch targets; lifecycle fields (`background`, `retain`, `replace_run`, `task_summary`, lineage fields) and the nested `config` (additive `instructions`, `thinking`, tool overrides) are unchanged.

## Purpose routes, seats, and diversity

Routes are executable launch bindings, keyed by purpose, in `[orchestration.routes.*]` (see the [configuration reference](configuration.md#orchestration-routes)):

```toml
[orchestration.routes.research]
targets = [{ model = "$session", identity = "researcher", tier = "substantive" }]
```

Each route carries `targets` — a non-empty list of `{ model, identity, tier, seat?, overlay?, replacement_capability? }` — and `requires_distinct_models` (default 1). `$session` binds the session model; that is what every shipped route target uses, which is why one model and ten models produce identical defaults. Route keys may also introduce custom purposes.

Precedence for one launch: structured explicit fields (`model`, `identity`) override the selected route's corresponding fields; a shorthand expression sits between; conflicts between shorthand and structured fields are rejected. The effective tier is the route target's; a direct unrouted launch takes the identity/purpose default. Replacement capability is marked separately from the tier and feeds the failure-time replacement policy. An unavailable model binding fails with an error telling you to configure it explicitly — another canonical model is never substituted.

Multi-seat purposes require an explicit `seat` selector; there is no implicit fan-out. The shipped example is `review.deep`:

| Seat | Identity | Notes |
| --- | --- | --- |
| `analytical` | `reviewer` | Read-only analysis seat |
| `peer` | `reviewer` | Read-only analysis seat |
| `execution` | `coder` | The executing seat, with the `review-execution` overlay |

The execution seat was never read-only in spirit: it runs as `coder` with a shipped review-instructions overlay (full-artifact review, `file:line` citations, declared check-only verification with exit codes, never implying checks that were not run). The overlay rides the frozen persona instructions at launch; it grants no authority and never rewrites the identity binding or its revision. Overlay bodies are harness-owned: there is no user overlay directory, and an unknown overlay reference fails loudly.

`review.deep` declares `requires_distinct_models = 3`. Diversity is evaluated over **canonical-model cardinality** — the same base model on two providers counts once — counting partial duplication (A/A/B fails a requirement of 3), at the resolution of the seat composition. An unmet requirement is reported instead of pretending diversity; a fresh-context reviewer on the same model is context separation, honestly labeled. Configure distinct models explicitly via per-seat model fields.

Shipped routes: `implementation` (coder), `mechanical-edit` (coder, mechanical), `research` (researcher), `search` (researcher, mechanical), `exploration` (researcher), `verification` (coder, mechanical), `design-analysis` (advisor), `planning-analysis` (advisor), `implementation-demanding-settled` (coder), `review.quick` (reviewer), `review.standard` (reviewer), and `review.deep` as above. Purpose labels organize work; they never grant permission.

## Authority axes

Three independent axes govern tools:

- **Granted authority** — the session's configured tool ceiling. It never widens down the agent tree.
- **Personal tool set** — what an identity may execute itself, enforced below the configuration layer (launch-level tool overrides cannot loosen an identity's personal posture).
- **Delegation tool set** — what an identity may pass to children.

A child's effective tools are the granted ceiling intersected with every ancestor's delegation set intersected with the child's personal set. Narrowing is monotonic per component, and the granted ceiling never widens.

Consequences, exercised at the root (children cannot launch at all):

- A `researcher` cannot obtain a writing `coder` — through `task` or a direct launch.
- An `orchestrator` cannot write personally, but can delegate writes within the granted ceiling.
- A read-only identity's delegation set is non-mutating only, so its children are non-mutating too.
- An explicit root denial still prevents delegated writes.

The tool classification that powers the mutating/non-mutating split is harness-owned and name-keyed: a custom tool named `read_file` does not inherit the built-in's classification, `bash` stays mutating regardless of command text, and reclassification invalidates cached child authority.

## The identity denylist

Launch admission enforces an identity denylist, `tools.task.denylist` in `config.toml`, applied **after** resolution — shorthand, route, and retained-instance launches are all checked against the resolved identity, not the raw spelling. Entries are exact names, shell-style globs, or `re:`-prefixed regular expressions, and they match both `user:foo` and `foo` spellings of a custom identity.

The Settings > Identities inventory is denylist-only: unchecking an identity writes this denylist. It can only block launches; it never grants roots or tool authority. The companion `tools.task.allowlist` (default `["coder"]`) merely auto-permits launches of matching identities; the denylist is an absolute veto checked before any grant.

## Custom identities, workflows, and strategies

| Resource | User directory | Selection |
| --- | --- | --- |
| Identity prompt | the user prompts directory (`~/.chartreux/prompts/`) | `identity = "user:foo"` or bare `"foo"` → `prompts/foo.md` |
| Workflow body | `~/.chartreux/workflows/` | `workflow = "user:foo"` or bare `"foo"` → `workflows/foo.md` |
| Orchestration strategy | `~/.chartreux/orchestration_strategies/` | `orchestration_strategy = "user:foo"` or bare `"foo"` → `orchestration_strategies/foo.md` |

One strict loading contract governs all three kinds: ids are plain file stems (letters, digits, `_`, `-`, starting with a letter or digit — no traversal, nested, or extra-colon forms), files must be non-empty UTF-8 no larger than 64 KiB, and `builtin:`/`user:` qualification must be well-formed. Reserved shipped names (`minimal`, `ceremony` for workflows; `default` for strategies; the five built-in identities) cannot be shadowed — a custom file reusing a shipped name is rejected with a rename instruction, never silently preferred. Built-in definitions are shipped with the harness; custom files exist only in the user directories above (project-level directories do not supply these resources).

A custom identity has no metadata and no enforced posture — the prompt is the whole contract. There is no `extends` or inheritance between identities in v1.

## Upgrading from 0.5.6

Chartreux is pre-release alpha; breaking changes may occur without migration paths. Every retirement below fails loudly with an actionable error — nothing is silently migrated, and no compatibility aliases are kept:

- **`system_prompt_id`** no longer selects the composed prompt; the `[agent]` selections do. The key is retired from the settings catalog, and `chartreux doctor` reports it with successor guidance (a custom identity plus workflow/strategy composition replaces full-skeleton prompt replacement). `compaction_prompt_id` is unchanged.
- **`agent_paths`, `enabled_agents`, `disabled_agents`** are retired from the settings catalog and flagged by `chartreux doctor`; block identities with the task denylist (Settings > Identities) instead of filtering agent profiles. The `agents/*.toml` profile directory convention is retired too: files are detected and reported, never loaded.
- **The `@orchestrator` model role is `@main`.** Selecting `@orchestrator` fails with rename guidance and no alias; a `[roles.orchestrator]` overlay in `models.toml` fails loudly at load — rename the table to `[roles.main]`; the catalog file is never rewritten automatically. Sessions persisted with `active_model = "@orchestrator"` fail resume with the same guidance.
- **Sessions persisted before v0.5.7** (old persona and policy formats) fail resume with a diagnostic that says to start a new session — a deliberate, versioned break, not a generic version error.
- **The task tool's `agent_type` field is removed.** The read-only legacy bridge onto the canonical `identity` field is gone: setting it fails validation with successor guidance (the old bridge mapped shipped profile names onto identities — `worker` → `coder`; a custom profile name → `user:<name>` — and the canonical field won), and the legacy `AgentProfileMismatchError` name is removed with it; the error is an identity-mismatch error.
- **Dispatch slots and modes are removed.** The `[dispatch]` catalog overlay no longer appears in the catalog template or the presets editor; orchestration is selected through identity/workflow/strategy and configured through `[orchestration.routes.*]`. A legacy `[dispatch]` table in `models.toml` is reported loudly at load with successor guidance naming `[orchestration.routes.*]` — it is never merged — and catalog writes preserve its values (comments and formatting are not preserved), never rewriting it.
- **Project-level prompt directories no longer supply identities.** Custom identities are user-level files only; a project `.chartreux/prompts/` directory does not make a prompt selectable.

Run `chartreux doctor` after upgrading: it reports retired settings with their successors and resolves the `[agent]` selections offline, before any conversation starts.
