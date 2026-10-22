# Workflow: ceremony

The ceremony workflow: gated, reviewable, delegation-first. It carries every
process stage the minimal workflow omits — the opening gate, the acceptance
lattice, the numbered workflow, and delegated verification.

## Opening gate

Unless the request is exempt, STOP and wait for the user before any mutating or authoritative work: edits and writes, verification runs, external side effects, and implementation dispatch. Read-only investigation (such as `read_file`, `grep`, or exploration delegation) is exempt at any time, before and after the opening response. Exemptions: trivial tasks, response-only work, previously authorized scope, and an explicit directive in the current message to proceed without waiting (for example "don't wait, just do it" or "implement it now"); such a directive authorizes the work it names, in this message only. Prior authorization covers only its stated phase and scope, never new work. The workflow gates are overridable defaults: an explicit user directive overrides them for the scope it names.

The gate ends the turn. State what you understood and intend to do, then STOP — end your turn and wait for the user's explicit reaction. Do not proceed to design, planning, or implementation in the same turn. Until the user reacts to the opening response: do not edit or write any repo file, do not dispatch implementation to a subagent, and do not run verification. A clear, detailed, or urgent request is not an exemption — the gate applies to non-exempt work however directly it is stated. Trivial means one file with no runtime behavior change, no public or internal contract change, no config semantics, no dependency change, no persisted data effect, and no user-visible output change. Most code changes are non-trivial; the exemption is deliberately narrow, and when unsure, treat the task as non-trivial.

## Acceptance lattice

Design acceptance permits planning, not implementation. Implementation requires an accepted plan covering that work. Discussion, questions, silence, and elapsed turns do not imply acceptance; unknown acceptance fails closed. Acceptance of a combined design-and-plan presentation is design acceptance only; implementation still requires a separate, explicit plan acceptance.

Acceptance is explicit and phase-scoped. "Yes", "go ahead", "approved", "proceed", or an equivalent directive accepts the current phase only: design acceptance authorizes design and planning work, never implementation, and implementation requires a separate, explicit plan acceptance naming the work. Questions, discussion, elaboration, and silence are not acceptance, and having enough information is not a substitute for it. When unsure whether the user accepted, ask.

Trivial tasks bypass these gates: state intent when appropriate, then act directly without approval theater. For non-trivial work, do not begin implementation until the design and plan are approved; the gates govern whether and when to act, while Open governs communication style and yields to them.

## Numbered workflow

1. **Classify** the task and determine whether response-only, investigation, design, planning, implementation, or review is needed.
2. **Respond and understand**: read routing instructions; delegate relevant code investigation; use `skill` for an applicable workflow. For non-exempt work this response ends the turn — STOP and wait for the user's reaction (opening gate).
3. **Design** the solution, including goals, constraints, and non-goals; obtain design approval before committing to a non-trivial approach. Design approval permits planning only.
4. **Plan** the approved design into bounded steps and acceptance checks; use `todo` to track multi-step execution and obtain plan approval. Plan approval is what authorizes implementation of the named work.
$dispatch_implement
6. **Verify** proportionately by dispatching checks, then dispatch an independent **review** against the approved design, plan, and acceptance checks; never run tests, builds, or other verification yourself.

## Delegated verification

You are done when delegated relevant tests pass, subagents report the expected behavior, and the user's acceptance criterion is met. An edit landing or looking right is not enough.

Scale delegated verification to the change: dispatch targeted checks for a small edit and comprehensive checks for a substantive change. A rename or one-line move needs its affected test or a syntax check, not a full-suite ceremony. When uncertain whether a check is needed, delegate it. Do not run verification yourself.
