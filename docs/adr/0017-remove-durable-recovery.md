# 0017 Remove Durable Recovery

## Decision

Remove the durable-recovery protocol before the v0.4.0 release. This decision
supersedes [0016 Durable Recovery Protocol](0016-durable-recovery-protocol.md).
The session journal, transactional rewind, workspace admission registry, and
operator recovery command are not part of the release.

Retain transcript generation fencing: stale queued saves must not overwrite the
retained transcript after rewind, reset, or session rebind. Retain the independent
burst-completion freeze fix, prompt-history serialization, thinking-override
restore, checkpoint restoration staging and permissions fixes, shell/git security
hardening, hygiene changes, ASK removal, subagent cap, Linux clipboard support,
and test-stability and logging-isolation fixes.

## Rationale

The crash-mid-tool-call loss that the protocol addressed had never been
experienced in Chartreux use. No upstream or peer implements an equivalent
protocol. Its maintenance cost outweighed the latent benefit of covering that
unobserved failure mode.

## Consequences

Sessions continue to use saved transcripts and memory-only file checkpoints.
Rewind saves or forks the conversation before restoring files individually;
staged writes and per-path failure containment do not provide transactional
compensation or crash recovery. The retained transcript fence protects against
stale saves, not loss of unsaved tool outcomes on process death.

The `~/.local/state/chartreux/recovery-registry/` directory and any
`workspace-admission*` files under the Chartreux state directory are now dead
state with no readers and are safe to delete manually.

## Agent Guidance

- Do not use ADR 0016 as an active implementation contract.
- Describe checkpoint and rewind coverage using the existing non-transactional
  semantics; do not claim durable tool acceptance or recovery guarantees.
