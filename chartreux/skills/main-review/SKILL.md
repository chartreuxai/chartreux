---
name: main-review
description: Review code, diffs, branches, pull requests, docs, specifications, and plans through bounded reviewer presets and synthesize the findings.
user-invocable: true
allowed-tools:
  - task
  - bash
  - read_file
---

# Main Review

Review is a main-agent orchestration procedure. Reviewers are independent, read-only subagents. The main agent owns target selection, intent, purpose choice, synthesis, and any follow-up.

## Target and intent

Use an explicitly named file, directory, diff, branch, pull request, or plan. If no target is named, use the uncommitted diff when one exists; otherwise ask the user for a target. Use the user's stated intent, the commit message, or the artifact goal. Ask one question only when the target or intent is genuinely unresolved.

For a pull request, fetch its branch before delegating, then review the branch comparison. Do not modify the target while reviewing.

## Review purposes

- **Quick:** launch one reviewer with `purpose="review.quick"`. Use for quick, fast, trivial, or rename-only reviews. Model-line substitution applies only to quick single-seat reviews: when the implementor's model family is known, the reviewer may be pointed at a different family — `zai-glm*`, `gpt-6*`, `mistral*` — via an explicit `model`; it never applies to standard or deep reviews.
- **Standard:** launch one reviewer with `purpose="review.standard"`. Use for most reviews; do not add a second reviewer by default.
- **Deep:** launch fresh, independent reviewers with `purpose="review.deep"`, one separate task per configured seat — the shipped `seat="analytical"`, `seat="peer"`, and `seat="execution"` seats plus every supplementary seat the route configures — deep panels comprise all configured seats, so every configured seat launches. Dispatch each seat through the route with no per-seat overrides; deep panels use no authorship substitution regardless of the implementor. Synthesize their reports.
- **Plans:** use the Deep procedure for plans, specifications, and designs.

Every launch resolves its reviewer through the purpose route: the analytical and peer seats bind the read-only `reviewer` identity, and the execution seat binds the `coder` identity with the review-execution instructions overlay (the executing seat — it runs the checks; it is not a read-only seat). Supplementary seats, when the route configures them, bind the read-only `reviewer` identity, whose tool surface carries only the bash verification carve-out (non-mutating verification commands): a supplementary seat audits intent, regression coverage, and evidence quality, or probes negative cases and boundary questions — a probe seat's final message must be the report itself. Do not run `/usage` automatically. Before round 1 record the selected convergence rule, convergence-panel scope, required seats, and round cap. Unless the user chooses otherwise, allow one initial panel and at most two refinement panels; any larger cap needs explicit user authorization. A panel dispatched with an incomplete seat counts toward the cap; completing that same pinned panel with an authorized fresh replacement seat does not consume another round.

If a seat's launch fails or is unavailable, report the failure as a blocker in the synthesis with per-seat status. Do not silently substitute another identity or model for a seat; a substitution is a scope change to propose to the user. The route's declared requirements (for example `review.deep`'s distinct canonical models) are validated by the launch itself: an unmet requirement is reported, never papered over — a fresh reviewer context on the same model is context separation, not model diversity.

## Route prerequisites

Launch admission resolves the conversation's LIVE route table: a validated route edit applies to subsequent launches in the same conversation — the applied notice enumerates each edited purpose's old → new target — while children already admitted keep their bindings. A rejection such as "requires 3 distinct canonical models; resolved 1" evaluates the route's declared requirement over the current table, not a validator defect: an edited table that violates a declared distinct-model requirement is rejected at reload (the prior table stays effective), so fix the configuration and reload, or dispatch the seat directly with an explicit model.

The shipped route binds every seat to the session model, so a fresh install needs a configured multi-model route before routed deep dispatch works: configure enough distinct canonical models to satisfy the route's declared requirement. Without one, use the documented direct fallback — an explicit `model` and `identity` on the launch, with the seat's overlay body copied into `config.instructions`.

## Execution seat brief

Brief the execution seat with the parent-ranked probe list as its primary scope and allow bounded self-discovery only (one hypothesis, or about five turns). Target a 25-30 turn schedule, stop polishing established blockers, and escalate an unfinished security assessment explicitly as INCOMPLETE — unfinished investigation is never converted into a pass.

## Dispatch

Each task string must be self-contained:

```text
task(task="Review: <target>. Intent: <intent>. Purpose: <review purpose>. Return a complete review report with findings, evidence, and verification status.", purpose="<review purpose>"[, seat="<seat>"])
```

Pass the purpose — and the seat for `review.deep` — as top-level fields; the route resolves the model and the seat's identity. For Deep and Plans, launch each seat separately with distinct tasks and independent instructions so each reviewer makes its own judgment. They may run in parallel; collect every result before synthesis. Use fresh independent contexts for every refinement, without earlier reviewers' conclusions or votes; include only the neutral ledger projection specified below.

## Delta rounds and convergence

Before round 1 the main agent records the artifact set, acceptance criteria, content-based baseline and target revision pins, required seats, convergence rule, convergence-panel scope, and round cap. The selected convergence rule is: one complete clean panel on the exact current revision, with all prior blockers independently verified resolved or explicitly user-accepted and no unresolved in-scope blockers; any artifact edit after that panel invalidates convergence. The convergence-granting panel scope is a full read of the exact candidate revision. Do not change these retrospectively to claim a pass. An initial clean full-read panel may itself grant convergence.

Pin content rather than a branch name: canonical ordered artifact paths and content hashes for the complete normative target and acceptance inputs, the exact last-reviewed baseline and target hashes, the effective delta, and the immutable reviewer-visible ledger projection. The mutable ledger stays outside the target digest, linked by that projection and its revision. Freeze the target while a panel runs, and recompute the pins at synthesis and before any convergence claim; a changed artifact or criterion leaves the target stale, and ledger-only status updates change nothing.

The initial panel and the convergence-granting panel full-read the complete target and acceptance criteria. Refinement panels are delta-scoped: they inspect prior blocker fixes, regressions and interactions introduced by the exact effective delta, and evidenced acceptance-criterion contradictions anywhere, with relevant surrounding context. Repetition of an adjudicated claim without cited new evidence is advisory; cited material new evidence can reopen it, and a fresh independent seat may find that evidence on its own judgment.

The main agent maintains an artifact-local adjudication ledger; for an inline artifact retain an inline record. Its minimum columns, in this order, are ID | round/seat | severity | claim+source | disposition | decision evidence | amendment/verification. Allowed dispositions are open, accepted-for-fix, resolved, rejected-needs-contrary-evidence, deferred, and user-accepted. Rejection requires cited contrary evidence, not a majority vote. A fix remains accepted-for-fix until independently verified on the target revision by a sufficiently discriminating observed check. Deferred blockers remain blocking; only explicit user acceptance records a risk as user-accepted. Reopen only with cited material new evidence, recording the old and new state, and preserve IDs across consolidation, mapping split and merged claims. A refinement brief gives the round number, pinned baseline and target, exact delta, criteria, open blocker claims to test, and a neutral ledger projection containing only IDs, claims, disposition labels and references — never prior reports, counts, votes or reviewer conclusions. A label can still bias judgment, so a seat keeps its own judgment and may cite new evidence itself.

Record each panel's scope, criterion, pins, required seats with their canonical models and identities, actual seat identities and models, run IDs or timestamps, evidence pointers, and each required seat's disposition: collected, incomplete, garbled, submission-blocked, or cancelled. Assess the actual report for completeness, intelligibility, scope and evidence; a completion-audit stamp is not a substitute, and a failed launch is recorded as incomplete with its reason. A missing or stale baseline, target, ledger, required seat, or panel makes the review INCOMPLETE, never converged — an honest incomplete report is not a zero-finding seat. Replace a missing seat only within the same pinned panel, with an authorized fresh independent seat on the same route, model, and identity; record the failed launch and the replacement, and never silently substitute.

Amend accepted blockers only; record warnings without amendment unless new evidence shows one meets the declared blocker rule or the user directs the change. After every panel consolidate accepted amendments into the one authoritative artifact, preserving ledger IDs and history by reference; consolidation creates a new target that needs the review the selected rule requires. After a clean panel, a consolidation that keeps the reviewed normative content byte-for-byte may rely on that panel by updating only the separately pinned ledger, while any normative edit makes a new candidate needing a new qualifying full-read panel within the budget. At the cap, consolidate for reporting and claim no convergence.

When the cap is reached, report NOT CONVERGED with the consolidated ledger, remaining blockers, missing evidence, seat states, revisions, and user decisions; do not auto-extend and do not imply a pass. A stale or incomplete panel cannot be retroactively made converged.

In every review brief require the reviewer to return the report directly without file writes; permit only authorized scratchpad-only test outputs and probes for the execution seat. Never copy the whole repository with cp -r; use git archive for a necessary tracked-tree copy and disclose any omitted untracked inputs. Give bounded ranked probes and stop polishing established findings. Require each seat's complete denial record with runtime reasons and its cumulative touched-path disclosure, including reverted paths; an honest empty set stays explicit.

## Synthesis

Return one convergence view:

- target, intent, purpose, and reviewer count
- the resolved canonical-model identity per seat, per round
- consensus findings with reviewer counts and file/line locations
- divergent findings and confidence
- blocking issues
- per-reviewer status
- prioritized next steps
- the selected convergence rule, current pinned revision, seat dispositions, remaining budget, and whether another authorized bounded round is warranted

Completed-panel acceptance: gate acceptance requires status for every selected seat, with missing or incomplete reports treated explicitly — never dropped. Record the actual resolved canonical-model identities for every round. A finding from two or more reviewers is consensus, not proof. A blocker remains open until independently verified resolved, rejected on cited contrary evidence, or explicitly accepted by the user. Apply the selected convergence rule; a clean review is not user acceptance. Do not claim a runtime or verification pass unless a command or behavioral test was actually observed.
