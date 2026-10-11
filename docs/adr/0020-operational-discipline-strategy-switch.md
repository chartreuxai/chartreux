# 0020 Operational-discipline rewrite: strategy switch replaces stop rules

## Status

Accepted.

## Context

The worker, reviewer, and advisor role prompts carry an `## Operational
discipline` section added on 2026-10-10, tuned for mistral-large-4 (ML4), the
model that fills the escalation-implementor and deep-review seats. An audit of
six post-change ML4 sessions found the section half-working:

- Worked: verbatim denied-command resubmits were eliminated (0 in all six
  sessions), denial counts roughly halved (5-12 per session vs a 13-28
  baseline), workers kept persistent denial records, and oversized tool
  outputs dropped (~780KB to ~256KB in sampled bytes).
- Failed: the whole-assignment stop rule ("stop and return a blocker after
  two same-reason denials") never fired. Same-rule recurrence continued past
  the second denial every time (2x, 3x, 5x, 7x). Workers complied with every
  action-demanding rule; the only abandon-the-assignment rule lost to ML4's
  task-completion drive.
- Role split: workers recorded denials and reconciled scope in their final
  reports; reviewers did neither. The reviewer/advisor report format
  (target/intent/status/coverage/findings) had no slot for denial records, so
  the recording rule had nowhere to land — the leading explanation for
  reviewer non-compliance.
- One worker spontaneously performed the desired method switch: after its
  second cd-chain denial it switched to `uv run --project <abs-path>` and kept
  the method for the rest of the session. The model can do the behavior; the
  prompt did not demand it.

The rewrite ports the operational-discipline block from the Vibe
`generic-large-4.md` prompt (the 13-rule "MISTRAL LARGE 4 MITIGATIONS" block,
rules 2, 3, and 13) into the three Chartreux role prompts.

## Decision

### Worker prompt

The 6-rule section becomes 7 rules:

1. **Denials** — Vibe rule 2 ported verbatim except that the sentence
   cross-referencing failure retries is deleted (that rule is dropped):
   persistent denial recording, consult-the-record before the next tool call,
   hardened permission/policy refusal language (never retry a refused action
   through changed syntax, another tool, a wrapper, or another model),
   ambiguous denials treated as permission/policy refusals, and at most one
   supported reformulation for syntax-only rejections. Amended after the
   2026-10-11 log dive: the refusal language gains a same-effect
   clarification — the refused action's identity is its effect on the same
   files or state, not the command spelling, so producing the same refused
   effect through a different tool, script, or mechanism is still a retry
   (observed instance: a denied `git checkout` followed by restoring the same
   files' content via a Python script + `git show`).
2. **Strategy switch** — Vibe rule 3 ported verbatim plus three adaptations:
   the "same reason" definition is retained, a bridge clause states that
   adopting a different supported method is not a retry of the refused action,
   and the adopted method is recorded alongside the denial so keeping it stays
   verifiable after compaction. This rule replaces the stop rule.
3-6. **Destinations, provider isolation, edited paths, scope reconciliation** —
   carried over, except Destinations, which is amended after the 2026-10-11
   log dive: the "unless the task explicitly requires repository outputs"
   exception is sharpened to mean the deliverable itself requires repository
   outputs (incidental tooling instructions such as where to point a log
   redirection do not count), and an explicit precedence clause is added —
   these rules take precedence over assignment text, and an instruction in the
   task message does not authorize a destination the rule forbids; report the
   conflict in the result instead of complying silently. Two repo-local
   artifact violations rode the old escape hatch, one of them instructed by
   the assignment text itself. Provider isolation, edited paths, and scope
   reconciliation carry over unchanged.
7. **Efficiency** — Vibe rule 13 with two carve-outs: "unless the loaded skill
   requires otherwise", and a freshness exception permitting bounded re-reads
   when the file's current state matters.

### Reviewer and advisor prompts

The denials rule is rewritten as a read-only variant: denied commands and the
runtime's rejection reasons are reported in the result (not persisted to
task-note files — these roles stay read-only), with the same hardened refusal
language plus the same-effect retry clarification as the worker prompt,
and no whole-assignment stop rule. The strategy-switch rule is
adapted to record in the result. Provider isolation and scope carry over
unchanged. The failures rule is deleted entirely. Both report formats gain a
`**Denials encountered:**` slot — in the reviewer report template after the
Coverage line, and in the advisor Advice template, stated to apply to every
permitted output form (the concise paragraph and the Blocker form included) —
so the recording rule has an output destination.

### Dropped rule

The failure-retries rule is deleted entirely by maintainer decision.

## Consequences

Action-demanding rules replace abandon-the-assignment rules: after the second
denial citing the same runtime rule/pattern, the agent must switch to a
different supported method and keep it, which is observable in the command
stream. Reviewer/advisor denial records now have a format slot, closing the
role gap. The shared subagent-contract block is untouched and remains
byte-identical across the three prompts, pinned by a new contract test along
with presence/absence assertions for the rewritten rules.

Two amendments from the 2026-10-11 log-dive addendum land on top of the
rewrite. First, the worker destinations rule now takes precedence over
assignment text: the "repository outputs" exception means the deliverable
itself requires them, not incidental tooling instructions such as where to
point a log redirection, and a task-message instruction does not authorize a
destination the rule forbids — the conflict is reported instead. This closes
the escape hatch two audited sessions rode to write repo-local artifacts, one
of them at the assignment text's own instruction. Second, the denials rule in
all three prompts now names the same-effect-other-tool retry pattern: a
denied action's identity is its effect on the same files or state, not the
command spelling, so reproducing the refused effect through a different tool,
script, or mechanism is still a retry. Both amendments are pinned by new
presence assertions in `tests/test_system_prompt.py`.

## Accepted risk: no prompt-level bound on repeated failing-command retries

With the failure-retries rule dropped, nothing at the prompt level bounds how
often an agent may repeat the same failing command within a session; the
harness turn and cost budgets are the only remaining bound. This is accepted
deliberately by maintainer decision. The audit evidence does not bear on this
rule: the observed stop-rule failure (the stop rule never fired, with 31
same-reason recurrences in a single session) concerns denials — commands that
never executed — not executed-command failures, and the failure-retry
recording duty was distinct from the denial recording duty (failure records
covered executed commands with exit codes and diagnostics; denial records
cover commands that never executed). The dropped rule simply had no audit
pathology; dropping it loses a recording discipline, not a working control.
The risk is watched through the validation metrics below (repeated unchanged
failing commands per session).

## Validation metrics

The rewrite will be audited against new ML4 sessions with these metrics. The
baselines below are the final pre-change baselines from the 2026-10-11 log
dive (sessions 16:03-20:10Z on 2026-10-10, all under the old rules); they
supersede the earlier six-session baselines where they differ:

1. Final-report denial disclosure rate: baseline 0/4 workers disclosed
   denials in their final reports (target: all disclose).
2. Same-reason denial recurrences per session: baseline 31 in one session
   (target: an observable method switch after the 2nd same-reason denial,
   then near-zero recurrence).
3. Max re-reads of one file per session: baseline 8 (target: 2-3 under the
   efficiency rule).
4. Repo-local artifact writes: baseline 2 sessions (target: 0, contingent on
   the new destinations precedence clause).
5. Verbatim retries of denied commands: baseline 0, keep at 0.
6. Repeated unchanged failing commands per session: watches the accepted risk
   above, since no prompt-level retry bound remains.
7. Claimed-vs-actual test totals: verify claimed test counts against tool
   output (performative-adaptation check — lower denial counts must not come
   from skipped verification).
8. Reviewer-slot disclosure: audit once a denial-bearing reviewer session
   exists. The slot is untested: the one reviewer with a denial ran before
   the slot was added and omitted it, and the one reviewer since had zero
   denials.
