---
name: main-plan
description: "Turn an approved design into an executable, bounded plan with work packages, dependencies, safe parallelism, acceptance checks, and verification. Delegates deep planning analysis to the advisor identity via the planning-analysis purpose route."
user-invocable: true
allowed-tools:
  - read_file
  - grep
  - write_file
  - edit
  - task
---

# Main Plan

Plan how to deliver an approved design without expanding scope. The main agent reads the design and frames the planning question; the advisor identity does the deep planning analysis. The main agent synthesizes and owns the final plan.

## Process

1. **Read the design.** Read the approved design artifact (inline or workspace `state.md`) and applicable AGENTS.md files. Delegate investigation of relevant code or configuration to a `search` or `exploration` child, naming the `sub-finder` or `sub-explorer` skill in the task brief.
2. **Frame the planning question.** Define: the design goal, the files and subsystems involved, any cross-cutting concerns, and what the plan must address (dependencies, parallelism, verification, rollback).
3. **Dispatch the advisor for planning analysis.** Request the `planning-analysis` purpose; the launch resolves the advisor identity and the model binding from the purpose route — pass the purpose as a top-level field:

```text
task(task="Planning analysis for: <design goal>. Design: <approved approach>. Files and subsystems: <list>. Constraints: <dependencies, shared interfaces, preserved behavior>. Return: bounded work packages, dependency order, safe parallelism, acceptance checks per package, verification commands, and rollback approach.", purpose="planning-analysis")
```

The launch runs in the background by default — a completion notification arrives when the advisor's run finishes; fetch the result then with `get_agent_result` (`mode="digest"` by default, `mode="full"` for the full body) before synthesizing, or block on the run with `wait_for_agent` (same modes).

For cross-cutting changes that span multiple subsystems, first dispatch a bounded exploration child (`purpose="exploration"`) that loads `sub-architecture-mapper` to map interfaces and dependencies, then use that output to frame the planning question.

4. **Synthesize.** Review the advisor's plan. The main agent owns the final plan — adjust work package boundaries, fix missing dependencies, and ensure the plan matches the approved design. If the advisor's plan contradicts the design, trust the design. When reviewing a plan, apply main-review's predeclared convergence rule, cap, fresh-seat scopes, completeness checks, and consolidation. After each panel the main agent updates the artifact-local adjudication ledger before any further dispatch; for an inline plan keep the ledger inline. Link the ledger from task state rather than duplicating it. A clean review is not plan acceptance.
5. **Present to the user.** Share the plan inline in the conversation. Obtain the user's explicit plan acceptance before proceeding to implementation; continued discussion, questions, or silence are not acceptance.

## Work package quality

Every work package must be:
- **Concrete**: an exact file or artifact list, one primary behavior change, and one runnable acceptance check. A package described as "implement X and integrate it everywhere" is not a package — split it along a natural seam (a function, a module, a route, a test file).
- **Independently verifiable**: its acceptance check runs without later packages and yields a pass/fail signal the main agent can evaluate without re-deriving the work. An implementation package's check must exercise its primary behavior, with the needed regression tests inside the package; static checks (lint, format) suffice only for genuinely mechanical or documentation work, and the package states that limitation. A package's acceptance check may also be frozen as a completion contract on the launch — `task(completion_check=[...], required_artifacts=[...])` — so the harness itself runs the mandated checks at finalization and stamps claim fidelity independently of the worker's report.
- **Independently recoverable**: each package names its recovery entry — the prerequisite checkpoint it builds on, the changes it owns, and how to undo or salvage them without disturbing completed packages. A failed package is retried, replaced, or re-planned alone.
- **Routed**: each package names its purpose route so it runs on the route its work class needs: `mechanical-edit` for bounded single-file edits with a known transformation, `implementation` for ordinary implementation, `verification` for packages whose deliverable is running checks, and `search`/`exploration`/`research` for investigation. Reserve `implementation-demanding-settled` for genuinely hard execution whose approach is already settled, and name its qualifying category — such a launch must carry `config.difficulty` from the qualifying set (`novel-algorithm`, `concurrency`, `subtle-invariant`) or admission fails.
- **Disjoint or serialized**: no two parallel packages may modify the same file or the opposite sides of the same interface; packages that must touch the same file are serialized by dependency instead.

## Plan content

The plan must define:
- bounded work packages, each naming its purpose route
- dependencies and only safe parallel work
- exact files or artifact paths in scope
- one runnable acceptance check per package, with concrete verification commands
- behavioral checks, with source evidence separated from runtime evidence
- each package's recovery entry (prerequisite checkpoint, owned changes, undo or salvage procedure)
- non-destructive rollback or inverse edits that preserve unrelated changes
- explicit blockers, assumptions, and non-goals

Write the plan to the task workspace (`~/.chartreux/workspaces/<project>/tasks/<task>/state.md`) only for large or cross-cutting changes. For most tasks, the plan is inline in the conversation.

## Cross-cutting architecture changes

When the change spans multiple subsystems or touches shared interfaces, additionally require:
- **Independently verifiable stages**: each stage has its own acceptance check runnable without later stages.
- **Explicit ownership**: every stage names the exact files it may modify and any shared interfaces it touches; no two parallel stages may modify the same file or the opposite sides of the same interface.
- **Preserved invariants**: list the behavior, contracts, and persisted-state compatibility each stage must not break, with source references.
- **Runnable checkpoints**: after each dependency boundary, a command or behavioral check that confirms the system still works before the next stage starts.
- **Safe parallelism**: parallel stages only when their file sets and interfaces are disjoint; otherwise serialize.

## Constraints

- Do not implement the plan — planning only.
- The main agent owns the plan — the advisor advises, the main agent decides.
- Do not use `ask_user_question` — discuss planning topics inline in the conversation.
