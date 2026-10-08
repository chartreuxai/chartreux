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

- **Quick:** launch one reviewer for `review.quick`; use the eligible slot the rendered slot table lists for that purpose. Use for quick, fast, trivial, or rename-only reviews.
- **Standard:** launch one reviewer for `review.standard` through its configured eligible slot. Use for most reviews; do not add a second reviewer by default.
- **Deep:** launch fresh, independent reviewers for `review.deep`, one separate task per seat in the rendered review composition, honoring its authorship substitutions. Synthesize their reports.
- **Plans:** use the Deep procedure for plans, specifications, and designs.

Use only the rendered eligible slots and their launch bindings. Do not call the usage tool automatically. Keep each phase bounded to one dispatch per listed reviewer and at most two refinement rounds unless the user approves a larger budget.

If a listed reviewer fails or is unavailable, report the failure as a blocker in the synthesis with per-reviewer status. Do not silently substitute an unlisted agent for an approved purpose slot; a substitution is a scope change to propose to the user.

## Dispatch

Each task string must be self-contained:

```text
task(task="Review: <target>. Intent: <intent>. Purpose: <review purpose>. Return a complete review report with findings, evidence, and verification status.", agent_type="<slot.profile>", config={model="<slot launch binding>"})
```

Select the profile and launch binding from the rendered eligible slot for the
purpose; do not infer routing from a profile default. For Deep and Plans,
launch each configured composition seat separately with distinct tasks and
independent instructions so each reviewer makes its own judgment. They may run
in parallel; collect every result before synthesis. Use fresh contexts for
second-round reviews, without earlier reviewers' conclusions. On a single-model
roster, use fresh-context review on the available model; if a required independent
slot is unavailable, report a blocker rather than silently substituting.

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
