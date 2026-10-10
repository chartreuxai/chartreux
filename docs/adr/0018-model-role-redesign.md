# 0018 Model-role dispatch redesign

## Status

Accepted.

## Context

The current three-tier routing model couples work classification to a small fixed roster of models. It assigns every substantive implementation task to a middle tier regardless of demand, reserves the highest tier for read-only work, and encodes tier decisions in prompt prose and task descriptions. This rigidity conflicts with roster churn: the shipped catalog may contain only one canonical model, while users may configure different providers, roles, and thinking levels. The resulting instructions can ask for unavailable tiers and duplicate routing policy across prompts and skills.

The redesign makes dispatch a typed, rendered, session-bound policy. The contracts below are normative and are the committed source of truth for WP1–WP6.

## Decision

### A. Mode enum and skeleton invariants

Modes are `standalone` and `orchestrated`, the shipped presets only. The mode is a session-level enum; free-prose values are rejected at load, and mode cannot be changed mid-session.

Both modes bake in these skeleton invariants: STOP opening gate and exemptions (cli.md:93–95); acceptance lattice (:97–101); phase loop (:103–106); delegated verification (“never run tests yourself”, :79–83 and :108); independent fresh-context review; self-verification honesty rule (:172 area); shell/safety rules (:141–143); communication rules (:145–175); and the headless exemption (system_prompt.py:315–323). Mode-variable paragraphs are rendered per mode: cli.md:62 (reading delegation), :107 (“never edit repo files yourself”), :114 (orchestrator identity and context rationale), :116–123 tool-discipline lines, :127–139 routing prose; and task.md routing region :11, :13, :15 (the whole region, not only :15). The baked invariant range is gates = cli.md:93–106 plus :108 (107 is mode-variable). Verification delegation is never mode-variable; standalone instructions must not invite the main agent to run tests.

### B. Standalone preset sentences

The standalone preset uses these sentences (validated by the WP10 behavioral gate):

- Implementation (replaces :107): “You may implement directly: make the approved repo edits yourself. Stay within the approved plan; change minimally.”
- Verification (keeps :108, :79–83 intact): “Verification stays delegated. Dispatch checks to a subagent, scaled to the change, then dispatch an independent review against the approved plan — a fresh reviewer, never the author. Never run tests, builds, or other verification yourself; never review your own work; never claim a check you did not run.”
- Context discipline (replaces :114): “Direct execution grows your context: everything you read or run is re-sent on every call. Keep reads targeted, bound shell output, move large artifacts to the scratchpad, and say when your context is getting long.”
- Tool discipline (replaces :116–123): “write_file/edit — repo files within the approved scope; scratchpad for temporary artifacts. read_file — files the task names or cites. bash — orchestration metadata only, always with timeouts; delegate searches, exploration, tests, and builds.”
- Routing (replaces :127–139): “Delegation is optional. When you delegate, route by purpose through the configured slots and pass the tier explicitly; the slot table is authoritative. When you implement directly, the tier rules govern only work you delegate.”

- Scenario-7 context valve (appends to context discipline): “When context is already long (including when the user says so), or a step would read or produce more than a few hundred lines, delegate that bounded piece to a worker before searching or reading source yourself; keep only its concise result in the main context. This context-hygiene requirement overrides optional delegation; keep the session in standalone mode.”

The opening gate and acceptance lattice remain unchanged, and the preset explicitly says so.

### C. Purpose vocabulary

Purpose identifiers are typed and developer-owned; descriptions render from these entries, not per-slot prose. Initial entries, extendable in an overlay by defining the entry and its rendering:

- `search`, `exploration`, `verification`, `mechanical-edit` — @small-class work.
- `implementation` — all substantive implementation.
- `implementation-demanding-settled` — demanding execution with a settled approach (the escalation implementor’s proactive route, not failure-triggered only).
- `design-analysis`, `planning-analysis` — advisor work, never implementation.
- `review.quick`, `review.standard`, `review.deep` — review tiers; deep is a composition list (default: sol-high-class + glm-class + large-4-class slots).

### D. Slot model

A slot is `{ profile (agent_type), role (@model ref), purposes (typed list), implements: never|routine|escalation (lint and rendered constraint), review eligibility }`. Slots are launch-time bindings only; retained-agent persona is immutable, and reuse re-states the tier explicitly. Roster shape is the set of (canonical model, thinking) pairs. One canonical model is a single-model roster, including when thinking levels differ; the same model through two providers is not multi-model.

### E. Lint spine

All lint runs at load time with stable diagnostic IDs.

Reference integrity: R1 every slot/profile/purpose reference in rendered blocks resolves; R2 contrast examples reference at least two distinct slots; R3 (warning) a contrast purpose should appear in the referenced slot’s `dispatch_for`.

Structural: S1 slot `@role` resolves against the merged catalog; S2 overlays must not shadow builtin profile names; S3 shipped vocabulary entries referenced by shipped skills cannot be removed (additions are allowed); S4 implementation purpose must not bind `implements: never`; S5 escalation slots are reachable only via reason-requiring routes; S6 review compositions resolve, are review-eligible, use an authorship substitution whose replacement differs from the replaced slot, and are non-degenerate (no duplicate seats, non-empty); S7 failure-table targets resolve and classes are unique; S8 rendered-output anchor lint per mode (mandatory verification prohibition and honesty sentences in both modes; “never edit repo files yourself” in orchestrated only); S9 mode selector accepts shipped names only; S10 overlay load failure fails closed — refuse start and print errors, never silently fall back to a shipped preset; never auto-fallback across broken provider/model config; invalid file is never overwritten.

S1–S5 and S8–S10 are the day-one activation-safety spine. S6, S7, and the R-rules land with the curated blocks they lint.

### F. Curated verbatim blocks and promotion triggers

Failure-routing table, review compositions (including authorship substitution), and contrastive examples ship as verbatim text blocks inside preset definitions and are lint-parsed for slot references (R1). Promote a block to schema when: (1) a second consumer needs machine reading (for example, a dispatch tool enforcing failure routes); (2) launch-time authorship enforcement is needed; or (3) a block is edited twice in overlays. Promotion is atomic with golden regeneration.

### G. Frozen WP0 exit decisions

1. Golden compatibility means the orchestrated multi-model fixture preserves today’s routing bytes; single-model rendering is intentionally tier-free. The shipped catalog is single-model, so default-path rendered text does change in Batch A; document this explicitly in the CHANGELOG/docs.
2. Reverse lint prohibits baked tier-routing in prompt templates and skills; it allows catalog role bindings, historical goldens, and the isolated orchestrated compatibility block. The Chartreux reference skill is exempt only after its WP4 update and is in scope.
3. Invalid dispatch means atomic rejection, visible standalone fallback, and a diagnostic; never overwrite the invalid file.
4. A single-model roster means one canonical model, even if roles differ by thinking level.
5. Single-model review is fresh-context review on the available model; fail closed if a required independent slot is unavailable.
6. Nudge predicate: latch on saving a second usable canonical model (readiness-known; linked deployments do not count). Signals are a compaction event or at least two deduplicated implementation failures of the same task (attempt-budgeted; replayed history, cancelled runs, and transient provider retries excluded). Display only at an idle boundary; persist one-time dismissal; suppress headless.

### H. Session policy semantics

Policy is bound at session creation. Resolve slot bindings rather than deferring them, and capture the resolved bindings in the policy snapshot so later catalog edits cannot silently alter launches. Persist identity/version and snapshot to prevent silent rebinding on resume. Children inherit the parent’s bound policy while preserving launch-time model/thinking overrides. Saved edits apply to the next session, not on refresh, reload, compaction, retask, or replacement. Old sessions without snapshots resolve live with a visible note; unsupported versions fail closed to error; removed bound roles/profiles at resume produce an explicit error, never substitution.

## Consequences

A dangling catalog role used by dispatch is a reference defect even when no
`[dispatch]` overlay exists. Reject the active candidate with a diagnostic naming
the catalog role binding to repair; the standalone recovery preset may itself
have unavailable slots. Render those slots as unavailable and fail launches
closed, without inventing bindings. Provider, deployment, model, or credential
unavailability is not dispatch-content invalidity: retain the session's mode
and committed bindings, and surface each degraded slot through configuration
validation warnings.

### I. Compaction handoff

Add policy identity/version, scope-specific contributor authorship (evidence-based), consumed attempt budget (missing state never renews), and current recovery route. Structure these as summary fields using the same transport as acceptance/grant fields, not as separate runtime state.

### J. Behavioral gate rubric (WP10)

Use nine scenarios (the plan’s WP10 table). Each scenario has a fixed input conversation and declared expected and forbidden trace events. Run two trials; a scenario passes only if both pass. A persistent failure after two trials fails the gate; do not reroll to green. Pin the overlay in fixtures; use declarative trace assertions as judge, not model opinion; retain traces in the workspace. Budget at most 12 real-model turns per trial. Unavailable credentials mean BLOCKED and block merge, not advisory. After the WP11 default flip, rerun scenarios 1–4 (the direct-implementation set) against the flipped default revision as the activation smoke.

### K. Reverse-lint breakage inventory

Repoint owners in WP4: tests/test_system_prompt.py:582–609 (routes-by-task), :338–369 (delegation protocol and “$role” templating constraint), :212–230 (defaults.py descriptions), :46–98 (catalog section rendering); tests/tools/test_task.py:46–124 (six tests reading raw task.md); tests/cli/test_installed_contracts.py:146 (task.md path reference, CI-covered); and the two SVGs in `test_ui_snapshot_provider_workbench`. Survivors verified: gate tests :488–579, retire-escalation negatives :623–635, and untrusted-content hardening.

### L. Rendering and cache mechanics

The cli.md read path strips the whole file; task.md read path does not strip, so golden capture must normalize to what the tool serves. `$current_date` is substituted at cli.md:2, outside the routing block; pin the date for deterministic goldens. Custom prompt directories take precedence, so capture must pin the shipped prompt. Task description cache: `Task.get_tool_prompt` is `@classmethod @functools.cache` (process-global); bypass or rekey it by policy identity. `ToolManager._tool_descriptions` is an instance dictionary populated at construction with a `discovery_source` copy (manager.py:164–172, 996, 1007–1015); the four construction sites in `_loop.py` (540, 1251, 4265, 4427) are injection points. Extend the existing per-instance spec cache; do not build parallel caching.

### M. Known follow-ups

- Dogfood the default path for a week after WP11. Roll back if any gate-relevant failure (direct implementation violating approved scope, skipped verification, gate bypass) occurs twice in the week: flip default back to orchestrated while investigating.
- User-level skill copies diverge from shipped; re-sync before the dogfooding week.

## Amendments

### N. Roster release R1 (2026-10-10)

- **G.1 superseded.** The orchestrated multi-model rendering no longer preserves the WP0 legacy routing bytes: the roster rename (`large`/`medium`/`small` -> `worker`/`scout`/`heavy`) rewrote the curated prose blocks, so the regenerated active goldens under `tests/fixtures/dispatch/{standalone,orchestrated,orchestrated-singlemodel}/` are the byte baseline now. The `legacy-orchestrated/` captures are preserved untouched as historical artifacts, and the single-model rendering fixtures use an explicit one-model catalog rather than the (now two-model) shipped catalog.
- **G.6 amended (re-gate landed in R2, 2026-10-10).** The graduation latch keyed on saving a second usable canonical model would fire immediately for fresh installs, because the shipped catalog now carries two canonical models (`glm-5-3` and `mistral-large-4`). The latch is now a bound-roster transition: it sets only when the roster the runtime would select moves from exactly one distinct canonical identity to two or more across a save, with a saved identity part of the multi-model post-save roster, so saves against an already multi-model roster and empty-roster transitions never latch; latches persisted by older stores remain honored. Repointing a role to an already-usable second model without saving or newly readying a model does not latch the notice. This is accepted for this release because the latch is save-driven, and the discoverability nudge is not guaranteed on every path to a multi-model roster.
