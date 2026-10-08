CRITICAL: Respond with text only. Do NOT call any tools. Any tool call will be rejected.

You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume this task.

Include:
- The user's current goal and any explicit constraints or preferences stated in the conversation before compaction
- Task classification (response-only, investigation, design, planning, implementation, review) and trivial/non-trivial status
- Current workflow phase (classify, respond/understand, design, plan, implement, verify/review)
- Design acceptance state and plan acceptance state as separate, explicit fields, each one of: accepted, pending, rejected, unknown
- Approved scope and the approval evidence — which user message approved what — not just the verdict
- One-time grants: target, purpose, and consumed/unconsumed state
- Active agent/run handles, outstanding dependencies between them, and uncollected results
- Dispatch policy identity, policy version, snapshot version, and session mode (copy the bound policy; saved edits apply next session)
- Scope-specific contributor authorship: scope key (paths or work-package ID), contributor/slot, concrete change evidence, and complete/partial/cancelled status; never infer authorship from thinking level or an agent merely being launched
- Consumed attempt budget by task/scope key: attempts already used, remaining attempts only when supported by evidence; missing budget means consumed, never a fresh allowance
- Current recovery route by task/scope key, its reason and pending dependency; compaction is not a new attempt or a reason to reset the route
- Key decisions made and their rationale
- Files touched and the current state of in-progress work (paths + one-line status)
- What remains to be done — the concrete next step
- Any data, identifiers, or references the next LLM needs to continue

Fail closed on approvals and carry the rule into the summary: never infer a missing approval — discussion, questions, silence, or elapsed turns are not acceptance, and an unknown acceptance state stays unknown. If an earlier compaction round was evicted by overflow, that eviction must not flip any acceptance state or resurrect a consumed grant. A one-time grant counts as consumed once used; if the retained conversation no longer shows that a grant was still available, treat it as consumed and never re-grant it.

Fail closed on dispatch handoff state too: repeated compaction, overflow, or fallback must preserve contributor evidence, consumed budgets, recovery routes, acceptance, and grants. Missing budget is consumed, missing authorship stays unknown, and missing recovery state requires diagnosis rather than restarting attempts. Record partial or cancelled contributions only when actual changes are evidenced, without claiming completion.

The summary's constraint and instruction fields carry conversation constraints only: ONLY constraints that were active in the conversation before compaction — never this prompt's own summarization instructions. The text-only/no-tools rule and the `<summary>` wrapper are transport mechanics for this response, not conversation constraints. The resumed agent must treat any such leaked text as a summarizer artifact, not an instruction, and continue using tools normally.

Be concise and structured. One line per modified file unless a snippet is load-bearing. Do not repeat information already captured. Do not include a "Final Answer" section — the entire summary IS the handoff.

Wrap the ENTIRE summary in <summary></summary> tags and output nothing outside them:

<summary>
...your handoff summary here...
</summary>
