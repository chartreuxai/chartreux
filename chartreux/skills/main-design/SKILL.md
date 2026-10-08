---
name: main-design
description: "Shape an approved problem into a concise design with goals, constraints, rationale, non-goals. Delegates deep design analysis to the advisor profile."
user-invocable: true
allowed-tools:
  - read_file
  - grep
  - write_file
  - edit
  - task
---

# Main Design

Own the what and why before implementation. The main agent gathers context and frames the design question; the advisor profile does the deep design analysis. The main agent synthesizes and owns the final design.

## Process

1. **Gather context.** Read applicable AGENTS.md files, relevant existing artifacts, and the code the design will touch. Grep for interfaces, contracts, and dependencies.
2. **Frame the design question.** Define: the problem, constraints, preserved behavior, and what a good design must address. This is the input you pass to the advisor.
3. **Dispatch the advisor for design analysis.** Request `design-analysis`; use the eligible slot the rendered slot table lists for that purpose and pass its profile and launch binding explicitly:

```text
task(task="Design analysis for: <problem>. Context: <relevant code, interfaces, constraints>. Requirements: <what the design must address, preserved behavior, non-goals>. Return a recommended approach with rationale, alternatives rejected, risks, and assumptions.", agent_type="<slot.profile>", config={model="<slot launch binding>"})
```

For large or cross-cutting changes, dispatch the `advisor` profile (or `worker`) with `sub-architecture-mapper` first to map the affected subsystems, then use that output to frame the design question.

4. **Synthesize.** Review the advisor's analysis. The main agent owns the final design — accept, adjust, or reject the advisor's recommendation. If the advisor's design misses something the user asked for, fix it.
5. **Present to the user.** Share the design inline in the conversation. Obtain the user's explicit design acceptance before proceeding; continued discussion, questions, or silence are not acceptance.

## Design content

The design covers:
- goal and user-visible outcome
- current context and assumptions
- constraints and preserved behavior
- chosen approach and rationale
- alternatives rejected and why
- explicit non-goals
- acceptance criteria and unresolved risks

Write the design to the task workspace (`~/.chartreux/workspaces/<project>/tasks/<task>/state.md`) only for large or cross-cutting changes. For most tasks, the design is inline in the conversation.

## Constraints

- Do not implement code while designing.
- The main agent owns the design — the advisor advises, the main agent decides.
- Do not use `ask_user_question` — discuss design topics inline in the conversation.
