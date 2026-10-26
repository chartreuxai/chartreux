# 0022 Live route editing

## Status

Accepted; implemented in v0.5.9 — the core live semantics (P1-i-a) and the
resume semantics (P1-i-b) shipped on the v0.5.9 permission-realignment
branch. The record below is the accepted design; the implementation note at
the end marks what changed at implementation time.

## Context

The v0.5.9 plan settled live route editing as in-scope (P0 settlement decision 6), reversing the earlier pending-only position. Shipped semantics were pending-only (`chartreux/core/dispatch/session.py:227-241`): route edits applied to new conversations only, and the per-launch model override was the sanctioned interim mechanism. The design pass of 2026-10-24 (the plan's P1-i analysis) settled the frozen-policy semantics against the shipped code.

## Decision

Routes are bindings, not content. The frozen-policy bundle freezes content (identity, workflow, and strategy bodies, plus fragments); the route table is a selector for the next launch, the same class as the model catalog, which is already a refreshable live section.

The conversation's admission source becomes the in-memory route table as of the last reload: a route edit that passes reload validation applies to subsequent launch admissions in the same conversation, while every already-admitted launch keeps its pinned binding. The manifest's bind-time route snapshot stays frozen as the integrity and diff anchor.

Admission and the rendered launch-target table read the same live source, switched in the same synchronous reload commit.

## The three semantic questions, settled

1. **In-flight vs subsequent launch.** Admission is the binding point: `resolve_identity_launch` (`chartreux/core/agents/identity_launch.py:580-607`) is pure and single-read. Edits never touch admitted launches in any state. Every admission after the reload commit resolves against the new table, so no mixed-generation launch is possible.
2. **Mid-conversation visibility.** The pending-route branch (`chartreux/core/dispatch/session.py:227-241`) is replaced by an applied notice at the reload commit. The launch-target table re-renders, and a conversation-visible notice enumerates each edited purpose with its old and new target, stating that in-flight children keep their bindings. The identity, workflow, and strategy pending branches are unchanged, because those are frozen content.
3. **Reload semantics.** Route edits are config-file edits accepted at reload — no watcher and no runtime-only mutation surface. `frozen_purpose_routes` (`chartreux/core/agent_loop/_loop.py:1651-1667`) becomes a live read of the in-memory table. Consumers switch with it: fresh admission and retained-retask validation in `app_server/_subagent_launch.py` (lines 245-251 and 294-309), and resume binding in `chartreux/core/agent_loop/_session_transitions.py:395-397` — on resume, the conversation adopts the current saved table and the resume notice reports the diff against the persisted snapshot. The per-launch model override's precedence over route selection is unchanged (`chartreux/core/dispatch/routes.py:107-127`). An invalid edited table rejects the whole reload, fail-closed: the prior table stays effective and in-flight children are untouched. `requires_distinct_models` is validated at reload.

## What stays frozen

- Identity, workflow, and strategy selections and bodies.
- Per-launch bindings after admission: committed model, identity revision, overlay, tier, pinned by the child record.
- The manifest's bind-time route snapshot, as the integrity and diff anchor.

The rendered launch-target table becomes live.

## Distinguishing checks

The accepted design lets verification distinguish preserved in-flight bindings from newly edited routes:

1. A running child's record keeps its committed model and identity across an edit.
2. A fresh launch after the reload resolves to the new target; the pinned record and the new record in the same conversation are the distinguishing pair.
3. The served launch-target table equals the admission table after the reload.
4. The applied notice is published, and the "applies to new conversations only" route-pending text is gone.
5. A resumed conversation launches on the current table, with the resume notice reporting the diff.
6. An invalid table rejects the reload as a whole: the prior table stays effective and in-flight children are unaffected.

## Implementation

Two substantive packages, shipped in order:

- **P1-i-a (core live semantics):** the live route source, the applied notice, and the prompt-text and pin updates — including inverting the frozen-route pins in `tests/integration/test_identity_policy_lifecycle.py` (lines 261-263, 285-296, 427-429, 522-524, 566-568), `tests/agent_loop/test_agent_loop_refresh_system_prompt.py`, `tests/app_server/test_tree_policy.py`, and `tests/test_token_efficiency_release_contract.py:84-86`.
- **P1-i-b (resume semantics):** resume adopts the current table, with the snapshot's reduced role documented.

## Consequences and residual risks

- Mixed-generation conversations become sanctioned: early launches resolve on the old table and later ones on the new. This is mitigated by the applied notice and the per-launch records, and the distinguishing checks make the split auditable.
- Resume no longer restores bind-time routes; a user who wants the old bindings restores them in config. That is the deliberate cost of live editing, mitigated by the resume notice reporting the diff.
- Diversity validation at reload is new surface. It must be implemented at reload, not per launch, to stay fail-closed.
- The notice channel's exact transcript-insertion mechanism must be located during implementation; the pending tuple is verified unread today.
- Test-pin breadth is the largest mechanical part of the work. A missed pin surfaces as a red suite, not as silent drift.
