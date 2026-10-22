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

- **Quick:** launch one reviewer with `purpose="review.quick"`. Use for quick, fast, trivial, or rename-only reviews.
- **Standard:** launch one reviewer with `purpose="review.standard"`. Use for most reviews; do not add a second reviewer by default.
- **Deep:** launch fresh, independent reviewers with `purpose="review.deep"`, one separate task per seat — `seat="analytical"`, `seat="peer"`, and `seat="execution"` — honoring the review composition's authorship substitutions. Synthesize their reports.
- **Plans:** use the Deep procedure for plans, specifications, and designs.

Every launch resolves its reviewer through the purpose route: the analytical and peer seats bind the read-only `reviewer` identity, and the execution seat binds the `coder` identity with the review-execution instructions overlay (the executing seat — it runs the checks; it is not a read-only seat). Do not call the usage tool automatically. Keep each phase bounded to one dispatch per listed seat and at most two refinement rounds unless the user approves a larger budget.

If a seat's launch fails or is unavailable, report the failure as a blocker in the synthesis with per-seat status. Do not silently substitute another identity or model for a seat; a substitution is a scope change to propose to the user. The route's declared requirements (for example `review.deep`'s distinct canonical models) are validated by the launch itself: an unmet requirement is reported, never papered over — a fresh reviewer context on the same model is context separation, not model diversity.

## Dispatch

Each task string must be self-contained:

```text
task(task="Review: <target>. Intent: <intent>. Purpose: <review purpose>. Return a complete review report with findings, evidence, and verification status.", purpose="<review purpose>"[, seat="<seat>"])
```

Pass the purpose — and the seat for `review.deep` — as top-level fields; the route resolves the model and the seat's identity. For Deep and Plans, launch each seat separately with distinct tasks and independent instructions so each reviewer makes its own judgment. They may run in parallel; collect every result before synthesis. Use fresh contexts for second-round reviews, without earlier reviewers' conclusions.

## Synthesis

Return one convergence view:

- target, intent, purpose, and reviewer count
- consensus findings with reviewer counts and file/line locations
- divergent findings and confidence
- blocking issues
- per-reviewer status
- prioritized next steps
- whether another bounded round is warranted

A finding from two or more reviewers is consensus. Any blocker reported by a reviewer is blocking until resolved or explicitly accepted by the user. Do not claim a runtime or verification pass unless a command or behavioral test was actually observed.
