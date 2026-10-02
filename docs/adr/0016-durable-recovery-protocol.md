# 0016 Durable Recovery Protocol

> **Status: SUPERSEDED by [0017 Remove Durable Recovery](0017-remove-durable-recovery.md).**
> The protocol was removed before the v0.4.0 release. The original decision below
> is retained as historical context, not an active implementation contract.
> Transcript generation fencing and independent fixes remain in use.

## Decision

Chartreux v0.4 uses a durable session journal and a recoverable, compensated
rewind transaction protocol. This document is the WP6a design gate for WP7–WP13.
The durable-resume and file-restoring-rewind contracts must remain disabled until
their complete recovery paths pass verification. Foundation code may land unused;
an incomplete contract must be disabled or explicitly labeled non-durable, not
presented as safe recovery.

“Must” and “must not” below are requirements. Record tables define logical fields
and relationships, not a byte-level encoding or a storage filename convention.

### Current implementation limits

- Safety gates follow journal enrollment, not the current feature flag. Disabling
  the flag does not weaken an enrolled session. Enabling it on a legacy resume
  raises an explicit degraded-mode error: legacy-baseline enrollment remains a
  follow-up. Start a new session for durable guarantees, or disable the flag to
  continue that legacy session without them.
- Ordinary app-server forks of enrolled sessions are rejected before publication.
  Strict ordinary-fork enrollment and lineage publication remain a follow-up;
  transactional rewind forks use their existing strict publication path.
- Durable sessions reject opaque shell, task, manual-shell, and command-hook
  launches before execution. A successful hook exit is not descendant drainage.
- File-restoring rewind uses memory-only checkpoints. After resume it cannot
  restore edits from earlier processes; an empty historical restore plan does
  not establish that historical workspace files were restored.
- Flag-off workspace admission still creates private registry directories and
  root-scoped sidecars to exclude overlapping transactions. Admission claims are
  effect reservations, not mutually exclusive workspace transactions.
- Rewind preflight rejects targets/replacements over 64 MiB and plans over
  256 MiB. Journal record copies and transaction-manifest protection scans remain
  uncached: caching requires an integrity/invalidation design.
- Cutovers record both the retained message index and the source journal
  watermark; superseded identities describe the retained non-contiguous history.

### Durability authority and failure boundary

Each session must have one serialized journal append authority, one monotonically
ordered record sequence, and one durable watermark. Concurrent tool calls must
use that authority rather than independently allocating sequence numbers or
publishing watermarks. The watermark identifies the last complete record whose
required persistence barriers succeeded; an attempted write is not a watermark.
A transcript watermark identifies the journal prefix reconciled into that
transcript, not merely the largest observed outcome sequence.

Strict persistence operations must raise on failure. Durable append and
same-directory durable replacement must perform the applicable flush, file
fsync, replacement, and directory fsync operations, including creation of files
and directories. A failed barrier must not advance a watermark or publish durable
acceptance, terminal success, a closed turn, or rewind success. Best-effort
persistence is permitted only outside correctness barriers.

The persistence-failure latch has these transitions:

```text
healthy -> pre-dispatch failure -> admission stopped; no effect starts
healthy -> post-effect failure -> admission stopped; no next model step
                                  -> already-started effects drain
latched -> recovery/reconciliation -> continue only after durable resolution
```

Cancellation is not rollback. Repeated cancellation must not release persistence
serialization while a write remains in flight. Durability failures must use a
fatal path outside the loop's ordinary tool-error/failure-finalization handlers.
No later tool, hook, or other effect may start after the latch is set.

### Journal schema and invocation identity

Every record must contain the following common fields:

| Field | Required meaning |
|---|---|
| `version` | Journal schema version; unsupported versions fail closed. |
| `type` | One of the record types below. |
| `session_id` | Stable owning session identity. |
| `generation` | Conversation generation to which the record belongs. |
| `sequence` | Unique ordered position allocated by the append authority. |
| `payload` | Type-specific fields below; required fields must be validated. |

| Record type | Required payload |
|---|---|
| `session-descriptor` | Workspace identity and complete canonical authorized-root set, cwd, origin directory, session/fork lineage, journal-version capability marker, and optional legacy-import boundary. Durable before first acceptance or effect intent. |
| `accepted-turn` | Stable `turn_id`, accepted user content, queue position/state, and any admission idempotency key and receipt. |
| `queue-transition` | Affected turn identities, prior/resulting queue state and order, operation (replacement, removal, pause, or promotion), replacement content where applicable, and any idempotency key and receipt. |
| `model-envelope` | `turn_id`, `model_step_ordinal`, immutable original assistant payload, and complete ordered sibling calls with call ordinals and external call IDs. |
| `call-resolution` | Model-envelope/call reference, ordered hook outcome/rewrite references, validated effective arguments and assistant-argument patch, or denial/no-dispatch evidence; final resolution for this attempt. |
| `effect-intent` | General effect-unit identity and attempt number, optional turn/model-call parent, executor configuration/input, cwd, recovery classification, and envelope/final call-resolution references for model tools. |
| `effect-outcome` | Effect-unit/attempt identity, outcome state, durable result or error evidence, ordered validated rewrite patch/denial evidence for hooks, descendant identifiers when launched, and corresponding intent reference. |
| `turn-close` | `turn_id`, terminal disposition (`completed`, `failed`, or `cancelled`), transcript commit receipt/revision, and incorporated journal watermark covering all turn effects, resolutions, and queue transitions. |
| `recovery-decision` | Evidence references, affected identities/generation, decision and resulting gate/disposition; a decision must not fabricate an effect outcome. |
| `rewind-cutover` | Transaction identity, old/new session and generation relationship, retained conversation/journal boundary, superseded work, pending-queue disposition, inherited settled-outcome references, and fork lineage where applicable. |

#### General effect-unit identity and attempts

Every effect, not only a model tool, must have a durable identity:

| Field | Required meaning |
|---|---|
| `effect_unit_id` | `(session_id, allocation_ordinal)`; ordinal allocated monotonically by the append authority, never reused across generations. |
| `executor_kind` | `model-tool`, `hook`, `manual-shell`, or `background`; opaque/child launchers declare their actual kind and recovery class. |
| `turn_id` | Owning turn when applicable; absent for out-of-turn effects. |
| `model_call_parent` | Optional `(session_id, turn_id, model_step_ordinal, call_ordinal)` reference; required for model tools, optional for hooks/descendants. |
| `external_call_id` | Required correlation evidence for model tools only; not a primary key. |
| `hook_phase`, `hook_ordinal` | Required for hooks: phase (`pre-tool`, `post-tool`, `post-turn`, or declared other phase) and ordered position within that phase and owner scope. Post-turn hooks need no model-call parent. |
| `non_model_owner` | For manual shell/background and hooks without a model-call parent: durable request/launch identity or turn/phase owner reference, never a fabricated model call. |
| `attempt` | Monotonically allocated attempt number within the effect unit; allocated and persisted with each new intent before execution. |
| `retry_of` | Prior attempt reference and durable authorization/decision for a retry, if applicable. |

A model call's correlation identity is
`(session_id, turn_id, model_step_ordinal, call_ordinal)`. Reused external IDs must
not alias calls. Each dispatched effect attempt has exactly one intent and at
most one terminal outcome; recovery decisions preserve that evidence. Retry
allocation must be serialized and durable, including after proven no dispatch.
Recovery must never dispatch under an existing attempt identity. Forks allocate
under their new session ID; cutover must never reset allocation ordinals, reuse
superseded identities, or reactivate a superseded attempt. Gaps from abandoned
allocations are permitted; reuse is not.

#### Two-stage batch persistence and reconstruction

```text
immutable model-envelope durable
  -> each pre-tool hook: durable effect-intent -> execution -> durable outcome/rewrite
  -> final call-resolution durable (effective arguments or denial)
  -> model-tool effect-intent durable -> tool dispatch -> durable effect-outcome
```

The immutable envelope must be durable BEFORE any pre-tool hook runs. It records
the model's original payload, not an assistant message later mutated by hooks.
Each hook rewrite must durably record the validated patch and ordered chain
references before a subsequent hook uses it. The final call resolution and tool
intent must be durable BEFORE tool dispatch; cwd/classification describe actual
execution. Denial must durably establish no tool dispatch, without erasing hook
effects. Post-tool and post-turn hooks have separate intents/outcomes.

Concurrent siblings may finalize independently: one sibling may dispatch while
another is still running hooks. No barrier requires all effective arguments to
be finalized before the first hook or sibling dispatch. A shared envelope may be
referenced rather than duplicated, but must always include all original siblings.

Reconciliation reconstructs from the immutable model envelope, then applies only
durable validated rewrite patches in hook order, keyed by call ordinal rather
than external ID. It uses the final call resolution for dispatched/denied calls
and reconstructs settled results by effect identity. Missing final resolution
leaves that call unresolved; a persisted intermediate rewrite is not dispatch
authorization. It must not fabricate a finalized assistant batch from mutable
memory or infer a hook outcome from a tool result. This preserves a valid sibling
structure even when outcomes precede the batch's transcript save.

Accepted content and queue transitions must be durable before acknowledging
durable acceptance or the transition. Replacement, acknowledged removal, pause,
promotion, and same-key retries must recover their recorded state and receipt;
recovery must not reaccept an already accepted turn or restore an acknowledged
removed turn.

A torn final record must not be interpreted as a completed record. Preserve the
valid prefix and torn-tail evidence, and conservatively block continuation until
the missing evidence is reconciled and a recovery decision is durable. Interior
corruption, identity mismatch, and unsupported versions must fail closed; they
must not be skipped as if they were a torn tail. Record limits and retention must
never discard unresolved recovery evidence. Journals, backups, and manifests
must use private storage; diagnostics must not dump credentials or sensitive
record contents.

### Conversation generations and commit linearization

Rewind, fork, and clear establish a new conversation generation. An in-place
rewind or clear advances the existing session's generation. A fork establishes a
new session/generation with explicit parent lineage; the parent's retained
conversation is not silently rewritten. Generation transitions and journal
boundaries must be durable, including conversation-only or empty-conversation
transitions. They must supersede removed work so that recovery cannot resurrect
it.

All persistence writes must check their captured generation at write time, under
the persistence serialization boundary, before touching the current transcript.
A pre-transition snapshot that waited for the save lock must be discarded after
the new generation commits. Checking only the in-memory cursor after writing is
insufficient. Cutover, transcript publication, and watermark publication must use
the same generation-aware authority.

#### Positive commit receipts and the WP6 fence

Strict transcript commit must return a positive receipt only after all required
barriers succeed. A normal return without such a receipt is not commit success.

| Receipt field | Required meaning |
|---|---|
| `session_id`, `generation` | Owning session and durable conversation generation committed. |
| `transcript_revision` | Immutable revision identity/hash of the published transcript; intermediate saves have distinct revisions. |
| `incorporated_watermark` | Exact reconciled journal prefix incorporated into that revision. |
| `commit_reference` | Durable validated commit evidence binding the preceding fields for recovery. |
| `terminal_turn` | For a terminal revision: `turn_id` and terminal disposition; absent for intermediate saves. Must be bound by the commit evidence, not inferred from message shape. |

A stale STRICT save must raise a stale-generation/epoch error, never silently
return success or issue a receipt for another snapshot. Disabled logging, empty
save suppression, or failed persistence likewise cannot produce a receipt.
WP6's `_transcript_cursor_generation` is an in-memory cursor epoch: synthetic
repair or cursor invalidation may advance it without a durable generation change.
It remains a write-fencing aid, not a durable conversation generation or receipt.
Writes must validate both their captured cursor epoch and durable generation under
the save authority; recovery reads the durable generation from commit/cutover
evidence, not from that counter.

#### Durable pending-turn closure

Normal turn ordering is:

```text
durable acceptance/queue promotion
  -> immutable model envelope
  -> pre-tool hook intents/outcomes and final per-call resolutions
  -> tool intents -> dispatch -> durable outcomes
  -> remaining model steps and post-tool/post-turn hook intents/outcomes
  -> durable terminal transcript + incorporated journal watermark + commit receipt
  -> durable turn-close -> publish closed turn
```

Outcomes must be durable before terminal-success publication and before the next
model step. Pending-turn close linearizes at the durable `turn-close` append,
only after its bound transcript receipt is durable. An intermediate transcript
save never closes a turn. Closure must verify all turn-owned effects (including
post-turn hooks and descendants) have durable settlement or an explicit gated
disposition, and that the receipt incorporates their evidence. Closing a turn
does not clear any uncertainty/descendant gate. No later effect may be attributed
to a closed turn; new out-of-turn work requires its own owner identity.

| Recovery evidence | Required queue/turn action |
|---|---|
| Valid close referencing a valid terminal commit in the active generation | Recover closed disposition; never promote/reexecute the turn. The close record may follow the transcript watermark it binds. |
| Terminal transcript receipt, no durable close | Keep turn open; reconcile effects and durably append an idempotent close only after terminal settlement is proven. Do not rerun to manufacture closure. |
| Intermediate save or missing/invalid terminal evidence | Recover promoted/open state and apply the recovery gate; never infer closure from transcript presence. |
| Superseded turn in a committed cutover | Apply cutover disposition before closure/reconstruction; do not resurrect it. |

There is no claimed multi-file atomic write: interruption between transcript and
close remains an open turn for reconciliation. Same-key repeated closure must
return the recorded disposition/receipt, not append contradictory closure.
Post-tool hook failure must not erase a known tool outcome. Outcome persistence
failure must suppress terminal success and preserve intent/sibling evidence.

### Outcome states and retry policy

```text
durable intent -> proven no dispatch
               -> durable result
               -> error with possible partial effects
               -> unknown (dispatch/completion cannot be established)
```

| State | Meaning | Recovery consequence |
|---|---|---|
| Durable result | Completion and result were durably recorded. | Reconstruct once; do not rerun the effect to recover its response. |
| Proven no dispatch | Evidence establishes that execution never started. | No effect is attributed to this invocation; any new dispatch still needs the normal barriers and policy. |
| Error with possible partial effects | Execution returned an error, but may have changed external state. | Preserve error/effect uncertainty; never relabel as no dispatch. |
| Unknown | Available durable evidence cannot establish the effect's outcome. | Apply the conservative class policy below; absence of an outcome is not proof of no effect. |

| Recovery classification | Policy for incomplete or uncertain execution |
|---|---|
| Shell | No automatic rerun. |
| File edit | Conservative gate; no automatic assumption that failure left files unchanged. |
| Known read-only built-in | Retry permitted after reconciliation and a durable recovery decision. |
| MCP, hook, subagent, child, or unclassified executor | Unknown by default; block autonomous continuation. |

A recovery decision may record a retry authorization or explicit disposition of
uncertainty. It must keep the original evidence and must not invent successful
completion, no dispatch, or external rollback. Permission to retry is not a
promise of exactly-once execution.

### Resume gate, discovery, and compaction

Every load/resume surface (CLI, app-server, and ACP) must use this order:

```text
discover session/journal and workspace transactions
  -> claim/reconcile interrupted rewind transactions
  -> reconcile active generation, journal identity, transcript, and watermarks
  -> durably persist required recovery decisions
  -> synthetic missing-response repair, if still applicable
  -> continuation only when the shared gate permits it
```

Journal-only sessions must be discoverable independently of valid transcript
metadata, including death after first-turn acceptance before transcript
publication. Discovery through the actual CLI/app-server listing paths must not
require the user to supply a journal path manually.

Unresolved effectful calls, unknown child effects, unresolved rewind transactions,
corrupt evidence, identity/version failures, and failed recovery-decision writes
block autonomous continuation. Repeated resume must neither duplicate settled
results nor reintroduce superseded work. Synthetic repair is not a substitute for
journal reconciliation and must not turn an unknown effect into a known outcome.

#### Durable session descriptors and legacy transition

A `session-descriptor` must be the first durable journal record, before first
acceptance or any effect intent. It supplies workspace, complete authorized roots,
cwd, origin, lineage, and the journal-version capability marker independently of
transcript metadata. Descriptor creation and store discovery registration must
be durable before acknowledging acceptance. Listing must enumerate registered
journal stores/descriptors as well as legacy transcript stores, and apply workspace
filters using descriptor cwd/origin/root identities; missing transcript metadata
must not hide a descriptor-backed session. Custom logging stores must register
in the shared discovery namespace before durable operation.

| Discovery evidence | Classification/action |
|---|---|
| Supported descriptor/capability marker, with or without transcript | Journal-capable session; reconcile before repair. |
| Journal/registration/transaction evidence but missing or corrupt descriptor | Incomplete journal-capable evidence; list as gated, never classify as legacy. |
| Unsupported capability/version | List as unsupported/gated; fail closed on resume. |
| Valid pre-v0.4 transcript, no journal capability or transaction evidence | Legacy; retain synthetic repair without durable accepted-turn/effect guarantees. |

Legacy-to-journal enrollment must claim the session and durably register an
`enrolling` capability marker before changing its baseline. It must complete
legacy repair under the legacy contract, strictly persist a baseline transcript
revision, then write a descriptor binding that revision/import boundary and
initial durable generation. The registration must become `journal-capable`
durably before accepting new journal work.
Interrupted enrollment is gated until baseline/descriptor/registration reconcile;
it must not fall back to legacy repair. The imported past gains no retrospective
effect guarantee. A journal-capable session whose journal disappears must fail
closed, not downgrade to legacy. Workspace/cwd relocation requires a durable
descriptor update and the registry procedure below before new effects.

Before automatic or explicit compaction proceeds, unresolved effectful intents
must be durably resolved or explicitly gated. A gate is not permission to forget
evidence: compaction must preserve the unresolved identities, envelopes, outcomes,
and decisions needed for later reconciliation, and must not enable continuation
past them. If that cannot be preserved, compaction must stop.

### Child-session limitation

For v0.4, the parent journal does not observe child dispatch. Child loops/loggers
are not reconciled into a complete parent effect history. Every launch intent
must allocate a durable descendant ID before launch and bind parent effect/attempt,
child session ID (when applicable), foreground/background mode, authorized roots,
and process launch token. A PID alone is not stable identity: process identity
must include a start-instance token, and ownership must account for descendants,
not only the launcher's exit. Launch outcomes and later completion observations
must reference this ID. Opaque launchers unable to supply this evidence must be
classified opaque before dispatch; they must not advertise drainability.

| Child event/evidence | Continuation/resume settlement | Rewind obligation |
|---|---|---|
| Foreground child joined; result and descendant completion observation durable | Launch/result is settled, but child effect history is still outside parent coverage. Even a fully joined foreground child remains conservatively gated on resume until an explicit durable recovery disposition; join is not proof of external effects. | With all descendant processes proven drained and no unresolved child gate (or explicit disposition), preflight may proceed. |
| Background launch succeeds | Launch outcome alone does not settle descendant work; retain durable descendant tracking and gate resume. | Drain all owned descendants and persist observation, or reject before staging. |
| Completion observed | Record observer, descendant ID, result/evidence and proof that owned processes drained; do not infer child dispatch/outcomes. Explicit durable disposition is required to clear the child recovery gate in v0.4. | Completion of launcher alone is insufficient; drain descendants or reject. |
| Parent dies while child may be live | Recover descendant IDs; reattach/observe where provable, otherwise preserve uncertainty and gate. Never rerun launch automatically. | Claim workspace exclusion; prove drainage of every owned descendant or reject. |
| Opaque launch or untracked background process | Unknown effects/liveness; launch success cannot settle it. Require explicit durable uncertainty disposition for resume. | Reject while drainage cannot be established; uncertainty disposition alone cannot prove process death. |

These obligations include shell-spawned background processes and grandchildren.
Mutating descendants must retain or inherit workspace admission reservations
until proven drained; parent death must not drop their exclusion. A launcher that
cannot track or exclude surviving writers must be rejected in durable mode before
launch, or explicitly run outside the durable contract while still respecting
workspace transaction gates. Cancellation is not drainage or rollback.

Rewind preflight must drain live descendants or reject before staging/mutation,
and must not assume cancellation rolled back effects. The child limitation must
be exposed as clearly as the logging-disabled contract; it avoids claiming parent
recovery coverage that independent child loops do not provide.

### Workspace registry and cooperative exclusion

Session leases lock session IDs, not workspaces. A durable workspace transaction
registry must provide discovery and cooperative exclusion across sessions and
competing recovery processes, including forks not yet published as sessions.

#### Shared namespace and conflict domains

All cooperating processes for one OS user must use one registry namespace at
`$XDG_STATE_HOME/chartreux/recovery-registry` (default
`~/.local/state/chartreux/recovery-registry`), independent of session log/store
configuration and logging enablement. Different session stores must register
there, not create private workspace registries. Processes sharing a workspace
must agree on this namespace; unavailable/mismatched registry authority must
reject durable mutation, not fall back to a local store. This is a same-user
cooperative contract; cross-user writers are outside it.

A reservation/claim covers the complete canonical authorized-root set (including
cwd), not just a selected project root. Two sets conflict if any roots are equal
or either is a path-component ancestor of the other. Aliases must resolve to the
same accepted root identity; nested roots such as `/repo` and `/repo/subdir` must
conflict even when sessions use different primary workspace keys. Claims retain
accepted path and filesystem identity; root replacement/relocation must not
silently retarget authorization. Hard-link aliases across disjoint roots and
untracked mount aliases are unsupported; mutating such aliases must be rejected
unless the implementation can include their shared identity in the conflict set.

Acquire/check all roots as one serialized registry operation, sorting canonical
root identities lexicographically for any per-root locks. Never acquire an added
root while retaining only a partial reservation; release/retry the entire set.
Overlap checking must occur under a shared registry coordinator, not independent
locks keyed only by exact root strings. Then follow the session lock order below.

Relocation or authorized-root changes must stop admission, drain reservations,
and acquire the union of old/new root sets using the same lexicographic ordering.
Durably update the registry mapping and session descriptor before admitting
effects at new roots.
Unresolved transactions retain their original paths/identities and claims; a
move must not hide them or reinterpret manifest targets. If their identity/target
mapping cannot be verified, reject relocation and gate recovery. Interrupted
mapping updates must preserve exclusion over both old and new sets until resolved.

| Registry/claim field | Meaning |
|---|---|
| Canonical workspace identity | Stable workspace mapping and complete accepted canonical root set with filesystem identities; aliases and overlapping/nested sets share conflict exclusion. |
| Transaction identity and manifest reference | Discoverable recovery evidence independent of transcript publication. |
| Claim owner | Session/transaction and current recovery owner; ownership must be durably recorded before mutation or recovery. |
| Gate state | Active transaction or unresolved recovery requiring exclusion. |

Registry checks, claim acquisition, and mutation admission must be serialized so
that a check cannot admit a cooperating mutator after a transaction claim starts.
A cooperating mutation must retain its admission reservation until its effects
have drained; transaction preparation must drain or reject existing reservations
before reading target states. A one-time gate check without this exclusion is
insufficient.
Use this lock order when multiple authorities are required: workspace registry
exclusion, then session ownership in stable session-ID order, then session
persistence serialization. Never wait for a workspace claim while holding a
session persistence lock. An existing long-lived session lease must not create a
reverse-order wait: contested ownership must be rejected or retried outside the
critical section. Recheck claims/generation after acquiring ownership;
there must be only one active recovery owner for a transaction. Durable claims
outlive process locks: a dead owner's transaction must be reconciled, not erased
merely because its process lock was released.

Tools, manual shell, review application, and rewind must check the workspace gate
before mutation. New sessions must scan unresolved transactions before admitting
mutation, even when their own session ID has no lease conflict. An unresolved
transaction blocks cooperative workspace mutation until finalized or verified
compensated. Read-only access must not bypass the session continuation gate to
launch effects. The registry does not control arbitrary external editors or
uncooperative processes.

### Rewind preparation, application, and publication

```text
prepare/manifest -> apply -> commit -> cleanup
                       \-> compensate -> verified compensated -> cleanup
                       \-> ambiguous/failed compensation -> mutation gate
```

Preparation must validate the entire restore plan and target policy, read current
target states, reserve storage, capture durable backups (including absent-file
markers), and stage replacement bytes on each target filesystem. Unsupported
symlinks, special files, permission requirements, or new-parent-directory cases
must be rejected before mutation unless their recovery handling is explicitly
implemented. Any preflight/read/stage/manifest failure must leave target files and
the active conversation unchanged. Unresolved artifacts must be preserved.

| Manifest field | Required meaning |
|---|---|
| Version, transaction identity, workspace identity, owner | Validated durable evidence tied to the registry claim. |
| Session/fork lineage | Parent and destination identities, including an unpublished fork. |
| Old/new conversation generation | Expected source and intended destination. |
| Conversation preparation | Durable retained/new snapshot reference; durable parent snapshot reference for a fork. |
| Journal relationship | Source watermark/boundary, intended cutover, superseded work, inherited settled outcomes, queue disposition. |
| Ordered target plan | Target paths/policy, original and prepared hashes or absence markers, backup and stage locations, expected applied state. |
| Phase and progress | Preparation/application/commit/completion evidence and per-target durable progress. |

For a fork, persist the parent conversation snapshot during preparation, before
file mutation. Reserve the destination identity and record its lineage in the
discoverable manifest/registry before publication. New-session publication must
not make the fork resumable before strict conversation and cutover durability;
recovery must be able to find the transaction even if publication never occurred.
The parent must remain recoverable throughout preparation and interrupted apply.

Apply targets in recorded order. Recheck each target's expected state immediately
before atomic single-file replacement or recoverable deletion. Record progress
durably and fail fast on conflict or write failure. Progress records alone are
not conclusive: deletion/replacement may occur before its progress write.

After all file operations succeed, recheck their expected applied states at
conversation commit. Then strictly persist the new-generation transcript and
watermark, the journal cutover relationship, and durable transaction completion
evidence. Rewind success and checkpoint-history drop are allowed only after that
commit is durable. Cleanup is idempotent and outside the correctness-critical
commit path. Conversation-only rewind uses the same generation/cutover barriers
without file application.

Cutover must enumerate the retained conversation boundary and superseded turns,
model steps, and outcomes. Pending work tied to removed history must be removed,
not silently promoted in the new generation; any retained pending work must be
explicitly identified with its queue state and idempotency receipts. A fork must
explicitly map inherited settled outcomes to lineage references, not alias new
calls to parent call identities. Recovery must use committed cutovers before
reconstructing outcomes, so removed work never returns on repeated recovery.

Uncommitted mutations must be compensated in reverse application order using
preserved backups. Before each compensation, recheck that the target still
matches the transaction's applied state. Restore originally absent targets as
absent; do not overwrite unexpected external contents. Drain in-flight
persistence on cancellation before determining whether commit occurred and
whether compensation is valid. Failed compensation or ambiguous commit must
retain evidence and establish a workspace mutation gate, not report rollback or
rewind success.

Concurrent-editor conflicts are detected by hashes at apply, commit, and
compensation; they are not prevented. The check-to-replace window remains. This
is a recoverable compensated transaction with crash detection, not all-or-nothing
multi-file visibility or unconditional rollback.

### Interrupted rewind decision table

Recovery must scan transactions before normal journal reconciliation, synthetic
repair, continuation, or workspace mutation.

#### Committed transaction authority

Validate durable commit evidence FIRST: the transaction completion record must
bind transaction identity, new session/generation, transcript commit receipt,
watermark, and cutover reference. A phase label alone is not commit evidence.
Once that binding proves commit, it is authoritative regardless of current
target bytes, later transcript revisions, or cleanup progress. Preserve durable
commit references across later saves/cutovers so historical completion remains
verifiable. Legitimate later mutations must never retroactively invalidate a
committed transaction or trigger compensation. Missing cleanup backups/stages
are not required commit evidence and must not gate a proven committed transaction.

Only when commit is not proven do manifest phase, target hashes, and backups
govern unresolved application/compensation. Contradictory or corrupt required
commit evidence still gates recovery; current-target mismatch by itself cannot
contradict proven commit.

| Recorded phase | Observed durable/current evidence | Required action and rationale |
|---|---|---|
| Prepare/manifest | No commit; all targets match originals; preparation evidence is valid. | Finalize as uncommitted/no mutation (or verified compensated), then idempotent cleanup; there is nothing to replay. |
| Prepare or apply | No commit; changed targets unambiguously match prepared/applied states and required backups are valid. | Compensate changed targets in reverse order, rechecking each; progress may lag actual replacement/deletion. |
| Any phase | All required new transcript receipt, generation, watermark, cutover, and completion evidence is durable and consistent; current targets may have changed since commit. | Finalize committed transaction and cleanup; never compensate or reapply a proven committed rewind. |
| Commit | Evidence is incomplete, contradictory, or cannot distinguish committed from uncommitted. | Gate mutation and continuation; do not guess from a phase label or reported write success. |
| Cleanup/completed | Commit is proven; only cleanup remains, even if target bytes differ or disposable backups are gone. | Repeat cleanup idempotently; do not replay application or gate later mutations because of historical target hashes. |
| Compensation interrupted | No commit; each target is unambiguously original or applied, with valid backups. | Continue reverse-order compensation only for applied targets; record verified completion durably. |
| Unresolved transaction | No proven commit and unexpected target contents or corrupt/missing required manifest/backups; or any required evidence corruption, ownership conflict, or failed recovery write. | Preserve evidence and gate; external edits and uncertain ownership must not be silently overwritten. |

Competing recovery processes must acquire the registry claim before acting.
Recovery writes use strict barriers; a failed decision/completion write leaves
the gate closed. Repeated recovery must be idempotent. Cleanup failure must not
undo a proven commit, but must not discard unresolved evidence.

### Logging-disabled contract and non-goals

When session logging is disabled, strict journal/transcript persistence requests
must raise rather than silently succeed. Ordinary non-durable operation may use
best-effort paths, but must not advertise durable acceptance or durable resume.
File-restoring rewind requires a configured durable transaction store; otherwise
it must reject before mutation. Disabling logging must not bypass an existing
workspace transaction gate.

This protocol does not provide exactly-once external effects, multi-file atomic
visibility, arbitrary shell/MCP effect reconciliation, or persistent checkpoint
history. Process-death restart guarantees and filesystem/power-loss durability
claims are distinct: SIGKILL tests establish only the former; sync-ordering and
syscall failure-window evidence is required for the latter.

## Rationale

The journal closes the gap between accepted turns, actual effects, and a
transcript saved later. Stable invocation identity and complete batch envelopes
permit reconstruction without rerunning a settled effect. Conservative gates
preserve uncertainty rather than disguising it as synthetic tool failure.

Generation fencing prevents queued stale saves from undoing rewind. Journal
cutovers prevent durable outcomes from resurrecting intentionally removed turns.
Preparing and backing up files before application, then committing conversation
only after application, reverses the unsafe commit-before-restore ordering.
Compensation is allowed only where expected-state evidence makes it safe.

The workspace registry is necessary because independent sessions can mutate the
same files despite holding different session leases. Durable parent preparation
and registry discovery make interrupted forks recoverable before session
publication. The child limitation deliberately avoids expanding v0.4 into a
cross-session effect reconciliation system that its parent journal cannot prove.

## Agent Guidance

- Treat this as the shared contract for WP7–WP13; use one integration owner for
  session identity/generation, persistence, strict commit, and the resume gate.
- Inventory tools, manual shell, hooks (including post-turn hooks), background
  spawning, and alternate ToolIO executors before enabling intent barriers.
- Test write/flush/fsync/replace/directory-sync failures, including short writes,
  rename followed by failed directory sync, and deletion before progress logging.
- Use deterministic process handshakes for deaths around acceptance, queue
  transitions, dispatch, outcomes, compaction, fork publication, and every rewind
  phase; resume in a fresh process and repeat recovery.
- Verify stale-writer rejection, reused external IDs, sibling outcomes, legacy
  repair, actual journal-only discovery, descendant drain/reject, competing
  recovery ownership, and concurrent-editor check-to-replace windows.
- Record commands and observed results; do not equate a source review or SIGKILL
  test with power-loss durability. Keep both contracts disabled until complete.

## Flag To User When

- Evidence is corrupt, unsupported, ambiguous, or cannot safely establish an
  effect outcome, transaction ownership, or commit state.
- A shell/file/MCP/hook/child effect needs an explicit recovery disposition;
  child effects are outside parent observation in v0.4.
- Logging or transaction storage is disabled, a target case is unsupported, or a
  durability/recovery write fails; do not claim safe continuation or rollback.
- External changes prevent safe application/compensation, or a requested
  guarantee exceeds cooperative exclusion and compensated recovery.
