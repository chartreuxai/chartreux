# Changelog

All notable changes to Chartreux are documented in this file.

## 0.7.6 (2026-11-08)

Plan-mode handoff becomes a fenced, crash-recoverable episode transaction; a
bounded child is offered only the tool definitions it can invoke, and subagent
completion is synchronized rather than assumed; compaction thresholds are
validated against a provider-documented context limit; the cache keep-alive
pilot ships off; and the edit and read tools gain bounded echoes and
revision-aware re-reads.

### Added

- **A bounded child's catalog carries only the definitions it can invoke.** A
  child is no longer offered `ask_user_question`, `task`, or the owner-only run
  inventory (`cancel_agent`, `check_agents`, `get_agent_result`,
  `reconcile_result`, `release_agent`, `resume_agent`, `wait_for_agent`) — the
  loop-level restriction is structural, not a posture — nor anything its
  identity's personal posture excludes. Withholding is advertisement-only: a
  forced or stale call still resolves to its definition and hits the tool's own
  explicit blocker instead of failing as an unknown tool. Authority is
  recomputed at every spec assembly, so a delegation grant restores the
  definition and a parent revocation drops it. MCP servers are gated per server
  on the same axis: a bounded child sees no configured server unless its parent
  opts it in, and an opt-in never exceeds the parent's gate. The bounded-child
  prompt fragment was compressed to pay for the catalog, and the pinned
  task-definition saving gate was re-based to a 23,600-byte floor, accepting a
  347-byte shortfall against the original 24,000.
- **Subagent completion is synchronized.** Every finished run surfaces exactly
  once through a reconciliation sweep with five triggers — continuation start,
  drain completion, every run-state terminal transition, a bounded periodic
  timer while the continuation is active, and the wall-clock cap boundary — so
  a dropped or suppressed notification is surfaced without duplicating a
  delivered one, and the sweep never writes a delivery marking. Results arrive
  as a digest by default and in full on demand, completion and wait results
  mark the transcript boundary they belong to, and the programmatic
  continuation lifecycle was rebuilt on that run state. The shipped guidance
  for `wait_for_agent`, `get_agent_result`, `check_agents`, `task`,
  `cancel_agent`, and the design and plan skills follows it.
- **A compaction threshold is validated against a documented context limit.**
  A deployment may declare the provider-documented `context_limit` (a positive
  whole integer token count, unknown unless the provider states it), and any
  raise of `auto_compact_threshold` above its shipped value must satisfy the
  headroom invariant at config load and at launch: the threshold plus a
  217,000-token one-turn growth allowance may not exceed that limit less the
  documented output reserve. Only raises are validated, so a deployment the
  provider does not document still loads at its prior threshold, and a user
  overlay raising one on an unknown-limit deployment must populate the limit or
  fail validation, leaving the last accepted configuration active. Every
  shipped deployment receiving the 500k threshold ships with a known limit and
  reserve; `glm-5-3` moves to 500k and `mistral-large-4` keeps 400k.
- **The cache keep-alive pilot, off by default.** A `[keepalive]` section
  carries the pilot's surface — `enabled` (false unless set), the absolute
  `max_attempts` count cap, the `max_spend_usd` spend cap, and the
  `min_idle_gap_seconds` idle floor — validated like every other section, with
  unknown keys rejected and an enabled pilot requiring spend room. While
  disabled the pilot has no behavior at all: no snapshot, no timer, no ledger
  row. When enabled, a strict accountant applies the caps to attempts rather
  than completions, the snapshot is invalidated against the conversation it was
  taken from, and keep-alive spend is isolated from the turn budget on its own
  usage purpose.
- **Request-shape telemetry on the usage row.** Every attempt of a logical call
  records content-free measurements of its request build — the serialized
  system-prompt and tool-definition byte counts, a prefix-stability hash, the
  idle gap since the loop's previous build, the process-wide in-flight request
  count, a toolset and prompt fingerprint, and the declared output limit — on
  its ledger row. Unset telemetry is omitted from serialization, so rows
  written before the fields existed keep their exact shape.
- **Plan mode hands off through a fenced episode transaction.** A planning
  episode is an identity with an on-disk arbiter: one directory holding one
  sidecar, every state change serialized through a single kernel `flock`, and a
  durable plan write whose temporary siblings carry the
  `.chartreux-plan-tmp-*` prefix. Accept is a two-phase transaction — PREPARE
  claims the episode, retains the draft's raw bytes, re-compares them against
  the consented digest, and publishes the consent snapshot; COMMIT
  dual-verifies the digest, durably marks the attempt, records the discovery
  domain with a one-observation binding, then flips the state and renames the
  snapshot to its provenance name. Crash recovery is keyed on the sidecar, the
  accepted snapshot is retained as provenance, and a record written before this
  release surfaces as `legacy_unverified` rather than being trusted. The
  consent modal, the TUI episode flows, and the restore paths were rebuilt on
  that arbiter, the shipped skill reference was synchronized with it, and the
  release carries a 55-node crash-recovery suite.

### Changed

- `edit` gained `multi_anchor` — an ordered list of `old_string`/`new_string`
  pairs, each matching exactly once, listed in file order, and non-overlapping,
  with every anchor validated before one atomic write lands — and
  `expected_site_count`, which pairs with `replace_all` and writes nothing when
  the anchor's site count differs. The result no longer echoes unbounded
  context: the window cap drops from 8 KiB to 2 KiB, and sites past the hunk
  cap are named by line number in a trailing note bounded at 512 bytes that
  always states the count and total.
- `read_file` gained `expected_revision`: a read conditioned on a content
  revision returns a short unchanged confirmation naming the checks it
  performed instead of the file, and serves the normal read when the content
  differs. It describes whole-file state and cannot be combined with `offset`,
  `limit`, `traversal_id`, or `outline`.
- The status line's context denominator is the deployment's documented context
  limit, not the effective compaction threshold, and renders `—` when the
  provider documents no limit.

### Fixed

- **Plan-handoff integrity, review rounds 1-3.** A terminal cancel writes the
  cancelled state and keeps its record as display history; a continuation
  commit is re-offered from the held claim together with the snapshot bytes
  rather than re-reading a draft that may have moved; the consent modal and the
  server both refuse an empty draft and a draft with no summary; the episode
  directory is resolved from the sidecar after a rebind, and the rebind's reads
  are anchored on a directory descriptor rather than a re-resolved path; the
  server refuses the legacy handoff route; and the executor scan is narrowed to
  the recorded `executor_session_id` and fails closed when that id cannot be
  matched, instead of matching another run.

## 0.7.5 (2026-11-07)

Stage 2 lands the runtime performance measurement machinery under `tests/perf/` — no production code changes.

### Added

- **Runtime measurement core**: observer primitives plus a fail-closed report/run-dir path guard (scratchpad-descendant `CHARTREUX_PERF_RUN_DIR`, guarded `CHARTREUX_PERF_REPORT`), the `chartreux.runtime_perf.v1` run-record/report schema with per-run chunks-coverage for all 17 gated campaign IDs, and a matrix runner with per-cell subprocess isolation, self-installed egress guards, and a py-spy readiness handshake.
- **Scenario harnesses**: integrity-valid 200k-equivalent fixture, headless transport/projector harness, Pilot (Textual) adapter harness with cross-adapter fingerprint parity, and five test-side ablations (transport_object, integrity_stub, markdown_no_render, gc_disabled, source_coalesce with final-text/ordering equivalence).
- **Producer cells**: end-to-end turn latency on both adapter paths with the injected-delay matrix (A1-A4, B1-B2), per-root-turn notification counts/bytes (B3-B4), save-loaded scan/lock-await/heartbeat cell on the P5 code path (C1-C3), Pilot render-flush cell (D1), completion storm with fresh cleaned cohorts (E1), deterministic permission round trips (E2), loaded latency and memory soaks (F1-F2, G1), plus observation-only startup, compaction, and process-tree cells.
- **Gating machinery**: byte-exact pinned metric registry and determinants documents, per-metric baseline validation with shared-set conjunction eligibility and bounded campaign closure, a comparison tool that writes only to the run dir, deterministic threshold construction with the pinned reason order, and committed baselines for seven eligible cells. The release contract now asserts every registry metric is ACTIVE-eligible or DORMANT-with-reason: 16 ACTIVE, G1 DORMANT `insufficient_baseline_data` at first activation.
- **Dev dependencies**: memray and pytest-memray (pyinstrument remains an external tool; declaring it is blocked by the retired-dev-root guard pending a plan-level decision).

## 0.7.4 (2026-11-07)

Plan mode becomes an orientation-then-standby workflow that survives a restart
and can be entered at launch, every transcript event gains a real clock with
per-minute group headers, the stale-margin repaint after a resize is fixed, and
a delegated subagent's tool catalog is gated against its own committed model
rather than its parent's.

### Added

- **Plan mode is an orientation-then-standby workflow.** The entry seed has the
  planner read the state of the repository — current branch, recent releases,
  existing workspace tasks — present a brief orientation, and then stand by for
  the user's plan agenda. It forbids inferring or inventing a task, proposing
  agenda items, and beginning design, planning, or implementation on its own,
  and it defers writing the plan document until the document's design and the
  plan itself have been accepted. `/plan <text>` treats any text that is not an
  action as a theme instead of an error: `accept`, `cancel`, and the new
  `status` remain exact-match actions, and a theme is a heading, never an
  instruction or an authorization.
- `/plan status` reports the active mode, its theme, and the plan document path
  in the same wording as the post-compaction notice, so the status line and what
  the model is told after a compaction cannot drift apart. While the mode is
  active the prompt area carries an amber tint and a `plan` border label; both
  are applied on entry and on resume restore, and dropped by `/plan cancel`,
  `/clear`, and `/plan accept`. The accept consent screen gained a Theme row,
  present only when the mode has a theme.
- **Plan mode survives a restart.** Entry persists a record — plan document
  path, theme, planner selection, and the pre-entry model and thinking — into
  the session metadata, and `-c`/`--resume` restores the mode before the session
  is marked ready, reapplying the tint and posting one resume notice. `/plan
  cancel` and `/clear` remove the record; the planner session checkpointed by
  `/plan accept` keeps its record, so resuming that checkpoint is the way to
  undo a handoff.
- **Compaction re-injects the mode.** After any compaction — `/compact`,
  proactive auto-compaction, or the reactive context-too-long path — the
  published boundary opens with "Plan mode is active. Plan document: …",
  appending the theme clause only when a theme exists. The notice is read from
  the persisted record at the single compaction funnel, and the boundary stays
  byte-identical when there is no record.
- `-P [THEME]` / `--plan [THEME]` starts the interactive client in plan mode,
  with the same heading-only theme semantics as `/plan THEME`. With
  `-c`/`--resume` the mode is ensured after the session lands: a restored
  plan-mode record keeps its own theme and ignores the flag's, while a session
  that was not planning enters plan mode on the resumed conversation. The flag
  is interactive-only and is rejected with `-p`/`--prompt`, a positional prompt,
  piped stdin, and `--setup`; a theme beginning with `-` is read as an unknown
  option. Planner model and thinking come from `[plan.planner]`, and the
  environment bridge (`CHARTREUX_PLAN__PLANNER__MODEL`,
  `CHARTREUX_PLAN__PLANNER__THINKING`) is the only override — there is no
  `--plan-model` flag.
- `/plan cancel` removes the plan workspace when it holds nothing but
  chartreux's ownership marker (`.chartreux-plan-mode`, written only into a
  directory the allocation itself created). A reused directory that happens to
  match the naming convention never receives the marker and is never removable,
  and a directory holding a plan document or any other content is kept. `/clear`
  keeps everything, as before.
- **Every transcript event carries a real clock.** Reasoning blocks, tool and
  effect calls and results, hook notices, compaction records, schedule, route and
  root-recovery notices, interrupt markers, warnings, errors, and session
  records now persist a per-event wall-clock time in the session log and surface
  it in restored history. Headers render only where a real clock exists and are
  never backfilled, so logs written before this release keep their unstamped
  events.
- **Minute group headers.** Stamped events are bucketed by wall-clock minute and
  render a `14:20`-style header (dated off-today, matching the message-header
  format) only when the minute's first stamped event would otherwise show no
  time of its own. A user or assistant message header counts as its minute's
  marker, so events continuing in the same minute add no redundant header; an
  effect stamps its start clock and never re-stamps when its result lands in a
  later minute. The headers are a pure function of entry clocks, regenerated
  inside the windowing path, so they survive rewind, transcript rebuild, and
  virtualization eviction, and they compose with tool-group folding without
  touching its expansion stores. The whole surface stays under
  `show_message_timestamps`.

### Fixed

- **The stale-margin repaint after a resize.** A resize that leaves cells
  vacated — the wider layout's right strip and bottom band still holding old
  lines, durations, and separators — no longer survives the settle. A settled
  resize repaints the screen, which re-blanks every cell with the app
  conversation background the transparent Screen, `#chat`, and `#messages`
  layers never own; any resize also clears the width-keyed streaming strip
  caches rather than letting a pure resize keep serving strips wrapped at the
  old width; and a resize arriving while the single-flight reflow runs is
  re-armed instead of dropped, so the pass re-checks the terminal size when it
  ends and runs again for the newer geometry. A chain of passes that exhausts
  its attempt cap on drifting geometry hands the slot to one successor chain
  rather than abandoning the resize it saw.
- **`read_image` for vision-capable subagents.** A subagent's catalog was
  filtered through the parent's delegation ceiling, which re-evaluated
  capability against the parent's model: a non-vision parent stripped
  `read_image` from every child, including children whose own deployment
  supports images. Capability gating now evaluates against the child's own
  committed active model, so a vision-capable child sees the tool and a
  non-vision child still does not. Authority is unchanged — granted
  configuration, live policy, delegation posture, and every ancestor's ceiling
  still filter, and a name the identity does not register stays denied — as are
  the real enforcement points: the send boundary's `ImagesNotSupportedError`,
  the launch-time rejection of image history for a non-vision model, and the
  tool's own run-time re-check.

### Changed

- Delegation evaluates authority and capability on separate axes. The parent's
  delegation ceiling decides what a child may reach, never what a child's model
  can do, and a capability gate is the caller's axis: an ancestor whose own
  model fails a capability gate no longer narrows a descendant's catalog. This
  is the semantics the v0.8.0 unguarded-mode authority inventory has to assume
  for the same filter chain — the ceiling walk and the launched-child checks
  share it, so an unguarded inventory must not read a capability-gate rejection
  as an authority decision, nor an authority grant as a capability guarantee.
- `/plan cancel` wording follows the outcome: it reports that the empty
  workspace was removed when it was, and otherwise that the plan document was
  left in place at its path.
- Review amendments across the four items: minute headers bucket per entry
  inside folded tool groups, live notice records carry their own stamps and
  re-derive their residency from the DOM instead of assuming it, and a
  `show_message_timestamps` change reconciles both header surfaces with the gate
  off. The resize settle repairs drift past its second attempt cap instead of
  ending on a half-applied reflow, and the seed states the executor handoff
  explicitly so a planner never starts the implementation itself.

### Retained limitations

- A plan-mode session with session logging disabled gets neither the resume
  restore nor the compaction notice: the metadata that carries the record does
  not exist without logging, and no live-record fallback was added. The mode
  itself still works for the session.
- `/branch` inherits the persisted plan-mode record, so a fork shares its
  parent's plan document with last-writer-wins semantics and no locking. That is
  accepted behavior, not a defect: the plan document is the only artifact the
  handoff crosses on.
- The suite retains `tests/tools/test_ui_bash_execution.py::test_ui_handles_non_utf8_stderr`
  and eight snapshot mismatches, which reproduce at the v0.7.3 base and are not
  v0.7.4 regressions.

## 0.7.3 (2026-11-05)

Thinking levels gain a six-level vocabulary declared per deployment, `/plan`
turns planning into a consent-gated mode with a real executor handoff, and the
harness secret scan that systematically blocked subagent submissions on ordinary
repository content is fixed.

### Added

- The thinking vocabulary now has six levels: `off`, `low`, `medium`, `high`,
  `xhigh`, and `max`. A deployment may declare the levels it can encode with
  `supported_thinking_levels` (Mistral Large 4 declares `low`/`medium`/`high`,
  GLM-5.3 declares `low`/`high`/`max`); the declaration constrains admission and
  the effort encoded on the wire, so a level the deployment does not declare is
  never sent to the provider.
- `/plan` mode: `/plan` applies the configured planner's model and thinking as
  session overrides, reloads the runtime, and posts the planner seed, which
  queues behind an active turn. `/plan accept` reads the plan document,
  snapshots the configured executor selection, and shows a consent screen with
  the plan path, summary, and that selection before handing off to a fresh
  executor session seeded with the planner summary and absolute plan path.
  `/plan cancel` restores the pre-entry model and thinking selections and leaves
  the plan document in place.
- `ask_user_question` settlement is server-authoritative: the terminal callback
  patch is published inside the resolution lock before the settlement RPC
  answers, so the client reconciles its own in-flight answer by callback
  identity and exact output instead of reporting a spurious closure. A closure
  the user did not cause is presented quietly with outcome-specific copy; the
  unchanged expiry warning still reads "Unanswered — deferred".
- Inline edit and write results collapse by default, with `+N`/`-M` headers
  computed from the same diff rows the expanded view renders; `write_file` keeps
  its plain call line. Expansion state is session-local: rebuilds and rewinds
  preserve it, while `/resume` and restart reset it.

### Fixed

- **The harness secret-scan defect behind the systematic submission blocks.**
  A touched path that is a directory — what a plain extraction, an in-place
  patch, a bare `git apply`, or a copy to a directory collects as `.` or the
  destination — is still enumerated in full, but only files modified at or after
  the run began are inspected; an explicitly named file path is inspected
  unconditionally. A session that merely wrote into an existing tree therefore
  no longer blocks on pre-session content such as over-cap binaries or the
  repository's own seeded-secret tests. The generic-keyword assignment rule no
  longer accepts `::` as a separator, so pytest node ids are not read as
  credentials. Symlink handling is tightened — a link's own mtime joins its
  target's, directory links are walked under cycle detection, and unreadable
  metadata fails closed as `scan-incomplete` — while the per-file cap, binary
  inspection, and block wording are unchanged.
- The submission scan is path-class aware (S4). Runtime-owned transient
  content — the parent-shared scratchpad and tempdir paths outside the
  session's authorized roots — is skipped with a single harness log notice
  instead of blocking the submission, vendored and cache directories (`.git`,
  `.venv`, `__pycache__`, the tool caches, `node_modules`, `dist`, `site`) are
  no longer walked, and the walk budget counts only the entries the run's
  write window can inspect, with an absolute enumeration cap still bounding
  the walk. Repository content stays scanned and fail-closed, an unestablished
  write surface still blocks, and the pre-commit hook, which scans staged
  blobs, is unchanged.
- Declared-set thinking reaches the wire (REV5). Backend encoding narrows by
  the deployment's declared levels, so an undeclared effort is never sent; on
  the Anthropic-style path, retained reasoning resolves the forced level among
  the declared enabled levels (nearest by vocabulary position), and a
  deployment that declares no enabled level raises a typed configuration error
  instead of emitting an undeclared `medium`.
- Manual compaction settles a `CancelledError` in the post-publication hand-off
  path: the published boundary (checkpoint or replacement publication and the
  lifecycle reserve release) is settled before the exception is re-raised, never
  swallowed.
- A terminal retryable provider failure now mounts a plain, non-retryable error
  message with a one-line hint — "/retry or Ctrl+R to retry this turn" — instead
  of a retry button whose handle could be stranded in the transcript when the
  user typed a new prompt. Retry stays available through `/retry` and Ctrl+R,
  armed by the presentation rather than the widget.
- Review amendments: the plan hand-off re-raises `CancelledError` after its
  cleanup instead of converting it to an internal error (REV1), the `/plan`
  entry path reloads the runtime after its write and reverts on failure (REV1),
  and `/plan` is a side-channel command whose entry seed is enqueued (REV3).

### Changed

- A requested level the deployment does not declare is no longer a typed error.
  Admission resolves it to the declared level nearest to it in vocabulary order
  — a tie resolves upward, toward more thinking — and reports the substitution
  once per deployment, requested level, and effective level. Mistral Large 4's
  `off` therefore resolves to `low`, and GLM-5.3's `medium` resolves to `high`.
  Unknown level names remain typed errors, as do a declaration naming a level
  outside the vocabulary and a declaration the selected model cannot encode.

### Retained limitations

- The scan's run cutoff is a write-window filter, not a provenance check:
  content backdated before the run — including content extracted with preserved
  timestamps — is not inspected (the documented blind spot), and a directory
  tree past the walk's entry budget still fails closed as `scan-incomplete`.
- The projected effective thinking level can still diverge from the Anthropic
  wire in a narrow corner: when the declared set's nearest level disables
  thinking and retained reasoning forces it on, the wire re-enables within the
  declared set while the projection reports the disabled level (deferred to
  0.8.0).
- The suite retains one pre-existing failure,
  `tests/tools/test_ui_bash_execution.py::test_ui_handles_non_utf8_stderr`,
  which fails identically at the v0.7.2 base and is not a v0.7.3 regression.

## 0.7.2 (2026-11-03)

Reliability and workflow follow-ups: a manual `/compact` can no longer wedge a
session behind a timed-out lifecycle transition, and the orchestrator identity
and review workflow are tightened against unbounded review loops.

### Fixed

- Manual `/compact` no longer leaves the session permanently locked. The
  `session/compact` deadline now runs inside the server-side LIFECYCLE reserve
  (720 s) instead of the generic request timeout, and the client sends a
  validated 750 s per-request override, so a slow compaction ends in a typed
  `COMPACTION_FAILED` (`reason=timeout`) instead of a bare "App server request
  timed out" that left `Session lifecycle transition is active: compact:<id>`
  stuck with no retry possible. Post-publication handoff failures (turn reset,
  replacement, reset callback) are classified `reason=persistence` with the
  settlement error kept as `__cause__`; the reset callback is guarded so it can
  no longer mask the typed error; replacement state is adopted only when
  settled; the lifecycle reserve is released and the next admission starts
  clean.

### Changed

- The orchestrator identity delegates reading, exploration, search, and
  verification to subagents instead of doing them inline; its direct tools are
  read-only probes, orchestration metadata, and persistence/bookkeeping edits.
- The review workflow (main-review, main-design, and main-plan skills, the
  reviewer and review-execution identities, and the ceremony workflow) now
  carries a predeclared convergence protocol: one complete clean panel on the
  exact current revision grants convergence; the round cap is the initial panel
  plus at most two refinements; the convergence-granting panel full-reads the
  exact candidate; every panel is consolidated into one authoritative artifact
  with a stable-ID adjudication ledger and a neutral reviewer-visible ledger
  projection; incomplete seats are recorded and replaced only by an authorized
  fresh seat within the same pinned panel; at the cap the review reports NOT
  CONVERGED instead of implying a pass. Shipped-text drift guards pin these
  contracts in `tests/skills/`, and the two orchestrated CLI golden fixtures
  were regenerated for the new orchestrator prose.
- AGENTS.md keeps `mkdocs build --strict` in the release procedure only, not in
  the verification instructions.

### Retained limitations

- A `CancelledError` arriving inside the post-publication compaction handoff
  window still escapes untyped (accepted plan residual), and a partial
  turn-reset failure leaves replacement state to be reconciled through session
  resume.

## 0.7.1 (2026-11-03)

Follow-up completion of the 0.7.0 surface rather than a new headline feature:
remote execution behind a closed destination grammar with destination-scoped
revocation, durable trust-store writes and a `trust` command line, the executor
shell-option contract with its three glob follow-ups, HTTPie's `https` entry
point, and trimming that now runs after a completed turn and re-arms across a
warm resume.

### Added

- `chartreux trust list [PATH]` and `chartreux trust remove PATH` inspect and
  edit the user trust store without starting a session. `list` reports the
  stored entries with their origin and names the closest ancestor entry that
  governs an untrusted path (a nested entry is marked as overriding its
  ancestor). `remove` deletes exactly the named entry, so its siblings and
  descendants survive, asks for confirmation defaulting to no, and refuses
  non-interactive use. There is deliberately no force flag: `--yes`,
  `--force`, and `-y` are rejected as unknown arguments. A removal goes
  through the same checked write as the interactive decision and reports that
  a restart is needed; a durability-uncertain write is an explicit error,
  never a silent success, and an unusable store is refused with exit code 2
  and a byte-identical store. The untrusted-config warning now names these
  commands instead of describing a manual edit.

- Remote execution over `ssh` is admitted for one exact destination. `ssh`
  left the default denylist under a closed grammar: a single, standalone
  invocation with a positive destination — an IP literal, or (with the
  hostname work below) a uniquely and non-locally resolved hostname — is
  classified `network.remote-exec` and asks for approval with the exact
  destination attached. The spawned command pins `UserKnownHostsFile`,
  `HostKeyAlias`, `StrictHostKeyChecking=yes`, and `BatchMode=yes` as
  leading options at every execution transport, derived from the approved
  destination; a hostile or unmodeled option, a composed or wrapped
  invocation, an unparseable destination, and a loopback, unspecified, or
  otherwise local literal all hard-deny with no approval path (see the
  retained limitations below: the pinned known-hosts file is not
  provisionable yet, so an approved spawn fails host-key verification
  closed until provisioning lands).

- Hostname destinations resolve, validate, and pin. A
  `user@hostname:port` destination is admitted only when the system resolver
  (honoring `/etc/hosts`) answers with exactly one non-local address; an
  ambiguous or multi-answer resolution, a local answer, and a resolution
  failure stay denied. The resolved address rides into the spawned argv as
  the destination operand while `HostKeyAlias` stays bound to the granted
  hostname. The resolution is re-observed once at spawn: a hostname whose
  answer changes between approval and execution is denied rather than
  contacted, so a rebound answer never reaches the command line.

- Bounded remote transfers. `scp` and `rsync` left the default denylist
  under a closed transfer grammar: a single standalone transfer with exactly
  two operands, exactly one of them a `user@host:path` remote operand, the
  host an IP literal or a uniquely non-locally resolved hostname, and the
  path inside a closed character set with no `..` segment. Only safe option
  clusters are admitted (plus `scp -P` for the port; `rsync` injects a
  pinned `-e ssh` transport), the call is classified `network.remote-exec`
  with the exact destination on the approval, and the admitted transfer pins
  the same host-key identity and resolved address into the actual argv at
  every spawn point. Remote-to-remote copies, a remote spec with no user,
  the daemon `host::module` form, user-supplied transports (`-e`/`--rsh`,
  `-S`), symlink following (`-L`), destination-mutating options (`--delete`,
  `--remove-source-files`, `rsync -P`, `--inplace`, the
  backup/partial/temp-dir families), the legacy scp protocol and
  server-chosen names (`-O`, `-T`), and any local, ambiguous, or failed
  destination all hard-deny. A locality-only attestation entry —
  network-namespace locality without the SSH trust context — hard-denies
  transfers even on a client that admits `ssh` execution.

- Destination-scoped revocation. `policy/dangerous_commands/revoke` now
  carries the exact canonical destination beside the registered session, the
  operation, the expected live generation, and the expected disk revision,
  and reports the live and saved outcomes apart. Revoking a destination
  removes only that destination's session grants, invalidates whole every
  pending once permit containing it, and removes the same destination from
  the saved project list. **Settings > Shell Permissions** lists each exact
  `user@host:port` under the section that holds it (live, accepted config,
  or disk only), Enter on a destination row revokes exactly it, and the
  whole-scope revoke stays reachable as one separate row behind an explicit
  `y` confirmation.

- Glob admissions for the pinned executor. A path component whose trailing
  star run is exactly two stars (`src/**`, never the recursive `**/` form)
  is admitted for `cat` and `wc` with no options when the live executor
  contract proves the run single-level (the POSIX family, or `bash` with
  `globstar` held off); the plan enumerates the directory and the actual
  spawn receives its literal words. A whole double-quoted `"$NAME"` or
  `"${NAME}"` word in a provably read-only position of a whitelisted reader
  resolves to one literal argv element only when a literal assignment to
  that name precedes the command unchanged in the same invocation.
  Everything else stays conservatively denied, and the recursive `**/`
  fall-out control keeps its denial.

- HTTPie's `https` entry point left the default denylist under the same
  restricted, positional-only grammar as `http`. It admits a literal
  `https://` destination and a schemeless operand (`https host/path`), which
  the client turns into `https://`; a literal loopback destination resolves
  run-class and a non-loopback schemeless destination resolves ask-class as
  `network.egress`. An explicit non-https scheme (`https http://host/`)
  would issue a plain-HTTP request and hard-denies as a cross-scheme
  spelling, as do options, file-reading request items, and every other
  spelling outside the grammar. The localhost shorthand
  (`https :8443/status`) is not a literal URL the recognizer parses, so it
  resolves ask-class rather than riding a loopback auto-allow.

- Trim savings are measured rather than asserted
  (`docs/project/trim-savings.md`): three deterministic transcript mixes,
  five trials per arm, on a quiet machine. The payload-only trim freed
  65,790 B against 1,347 B of control noise in the tool-output-heavy mix
  (64 KiB reasoning alongside 768 KiB of tool output), 1,572,607 B against
  3,391 B in the reasoning-heavy mix (1.5 MiB reasoning), and 8,389,289 B
  in the 8 MiB positive control against 3,747 B of noise; the independently
  serialized reasoning bytes removed were 66,816 B, 1,596,234 B, and
  8,512,128 B. The decisive limitation is scope, not magnitude: the 768 KiB
  of tool output stayed resident.

### Changed

- Trust decisions persist through one checked atomic write. Trusting or
  untrusting a folder or workspace now writes a fsynced temporary file,
  atomically replaces the store, fsyncs the directory, and re-reads the
  revision under the lock before replacing; the revision is a content hash,
  so a same-size write inside one mtime tick is still a conflict. The
  outcomes are explicit — saved, not saved, durability uncertain, conflict,
  malformed, or locked — a malformed store is never rewritten, and a refused
  write never publishes live trust for the decision that failed. A store
  that cannot be answered is refused instead of being read as "untrusted":
  `chartreux doctor` renders one check per store state (missing skipped,
  unreadable unverified, malformed or invalid failed with a repair remedy
  and exit code 1), `chartreux skills` exits 2, and the workspace,
  relocation, project-env, and harness consumers raise a typed error
  instead of silently resolving untrusted. A missing store remains the
  ordinary first-run default.

- Every spawned shell runs under a pinned executor contract.
  Expansion-relevant option families are held off (bash `extglob`,
  `globstar`, `nullglob`, `dotglob`, `failglob`, `nocaseglob`; zsh
  `extendedglob`, `globdots`, `kshglob`, `nullglob`, `caseglob`, `shglob`),
  startup-file environment variables (`BASH_ENV`, `ENV`, `ZSHENV`,
  `KSH_ENV`) are scrubbed, `ZDOTDIR` is redirected, the locale is pinned,
  and the contract is applied last so a project `.env` cannot re-introduce
  an option. The shell invocation admits only `-c` and
  `--noprofile`/`--norc`/`--noediting`; an option-changing builtin
  (`setopt`, `emulate`, top-level `set`/`shopt`) denies; a nested shell
  re-entry with an expansion-bearing payload denies; and an unknown executor
  fails closed at resolution and again at spawn. Extended-glob syntax
  without glob metacharacters (`^foo`, `foo#`, `a~b`) is detected
  quote-aware and stays literal, and the same pinned plan is what the argv,
  shell-string, background, and terminal transports consume — the
  background launcher re-derives the bound plan at execution and hard-denies
  rather than falling back to the original shell text.

- Trimming now runs after a completed turn. The post-turn seam captures the
  settled boundary during the turn's own completion, and the logger then
  seals and publishes it in one synchronous fenced step after the pending
  saves drain and the archive lock is held; compaction is no longer the only
  trigger. The trim is best-effort: a failed seal or a failed publish keeps
  the originals effective, carries a receipt, and never fails the completed
  turn.

- A warm in-process resume re-arms trimming. All resume paths clear the
  old-epoch descriptor and reseal after a full streaming hydration proof —
  session and path identity, archived prefix bytes, and every record on disk
  parsed equal to the loaded records, exactly once each. A failed proof
  keeps the old descriptor and the retained authority with byte-identical
  bytes and loud logging, and there is no silent re-pin fallback: the
  resume still succeeds, but that session stays un-armed rather than
  risking a rewind it cannot prove.

### Fixed

- The question widget's expired-draft preservation store was dead machinery
  and is removed with the widget lifecycle unchanged (a characterization
  test pins the close message, queue advance, and surface handoff before and
  after removal). A model question that originates in a bounded child is now
  refused at admission with the same typed refusal the tool-level guard
  already produced, so no callback is opened that nobody could answer and
  nothing is granted.

- `chartreux doctor` keeps its typed trust row when a corrupt store and a
  discovered project configuration are both present; the config-load
  failure path no longer returns before rendering the store state.

- The remote-exec wire symbols moved to a shared leaf outside both the core
  and app-server packages, so the app-server client boundary no longer
  imports `chartreux.core.dangerous_commands`; the import-boundary tests
  pass again. This is an internal layout fix; the wire models and the
  exported names are unchanged.

- Investigation and test-debt outcomes. The stale-lock anomaly is the
  session-lease filesystem lock, whose inert leftovers are designed to be
  re-acquired by the next process; it is not a live defect, and new
  subprocess-held lease tests pin both the live-owner block and the
  owner-death re-acquisition. A trace of the suite's git traffic found no
  offender: the `index.lock` churn observed during this cycle was an
  external application's status poll running without `--no-optional-locks`,
  not a test writing into the main repository. The shipped-skill dedup audit
  found no duplicate sources, the wheelhouse requirement for the
  installed-contract tests is documented in the README, the stale release
  worktrees and hash-matched branches were removed, and pyright is pinned to
  1.1.414.

### Retained limitations

- Intermediate states, recorded because each was its own integration point:
  before the `trust` CLI landed, the untrusted-config warning still
  described the manual edit; before the trailing-star and quoted-word
  follow-ups, the executor contract narrowed glob admissions further (no
  trailing `**` and no quoted-variable word); before hostname pinning
  landed, only IP-literal `ssh` destinations were admitted; while transfers
  were still denied, pinned hostname execution worked but `scp` and `rsync`
  did not; before the destination-scoped backend landed, revocation was
  whole-grant only; and between the backend and the Settings control,
  destination revocation existed at the protocol without a panel control.
  No intermediate state claims a later state's guarantees.

- An approved `ssh` spawn cannot complete host-key verification yet. The
  pinned `UserKnownHostsFile` is the fixed, unprovisionable path
  `/dev/null/chartreux-pinned-known-hosts`, so with
  `StrictHostKeyChecking=yes` and `BatchMode=yes` the connection fails
  closed instead of trusting an unknown key. This is fail-closed by
  construction rather than a working end-to-end remote session: the
  destination grammar, authority model, and argv pinning are exercised, and
  provisioning the pinned known-hosts file is the remaining step.

- Remote execution from the terminal transports and the ACP path hard-denies
  without an execution-bound attestation entry, and ACP clients default to
  the non-attestation posture. The attestation is client-self-reported: it
  closes accidental misrouting — a client that cannot prove it shares the
  approved network namespace and SSH trust context is denied — but it
  cannot stop a client that lies about its own context. A locality-only
  entry is not enough for transfers.

- Strict-client compatibility is unchanged from the 0.7.0 convention: the
  destination request field and the destination-aware projection follow the
  strict wire models, so an older strict app server rejects the new request
  (the whole-scope request included, because the defaulted field still
  serializes) and an older strict client rejects the new response keys.
  Client and app server must run the same protocol version.

- Revocation undoes no remote effect. A command already running keeps
  running, an already-asked approval question is not cancelled (and
  answering it after the revocation grants fresh authority for the decided
  scope, which the revoke reports as a concurrent-regrant warning), the
  running session keeps whatever its accepted config already granted until
  a reload or start, a child's or another process's store is outside the
  transaction, and the panel reports the server's response rather than
  independently verifying it.

- The automatic post-turn trim is best-effort by design: it is a memory
  optimization on the completion path, so a failure to seal or publish
  leaves the original history effective and is reported as a receipt, never
  as a failed turn or a lost conversation. A trim therefore may simply not
  happen on a given turn.

- The warm-resume re-arm depends on its hydration proof. When the proof
  fails — a foreign session, a trimmed prefix, changed archived bytes, a
  paused in-flight save — the resume keeps the old descriptor and the
  session stays un-armed for trimming; the alternative considered,
  re-pinning an unproven archive, was rejected as the riskier failure.

- Trim scope is unchanged and is the decisive limitation: only the sealed
  messages' reasoning payloads are released, so tool output and visible
  text stay resident. The measurement shows magnitude for reasoning-heavy
  transcripts but little for tool-output-heavy ones, and the 0.8.0
  structural-tombstone decision is the place for the rest.

- `ascii_chrome` is not a full-ASCII mode. A native 80x24 capture measured
  the chrome glyphs replaced, but widget borders (U+2500), the literal
  header separator (U+00B7), the status-line em dash (U+2014), and the
  braille loading spinner (U+28xx) remain non-ASCII. The remaining shell
  over-denials documented in the tools-safety guide stay in force: a star
  run of three or more, a run that does not end the component, a wildcard
  in a middle component, every command outside the read whitelist, and
  every quoted variable outside the one admitted word shape; the work-budget
  caps are unchanged, and system-wide startup files and the shell binary
  remain a root-owned trust base.

- Native verification found no desktop-availability blocker and recorded
  pass verdicts for the OAuth browser launch (a real `xdg-open`/Firefox
  dispatch to a loopback URL), NO_COLOR contrast at compact dimensions and
  across a light palette (grayscale-only SGR with structure retained), and
  trust persistence (the store written on disk and honored across a
  restart). Those checks are native, not hermetic-only; the
  `ascii_chrome` finding above is the one measured negative.

## 0.7.0 (2026-11-01)

Everything shipped since v0.6.1: the skills and route inspection tooling, the
completion-audit and effective-thinking displays, a smaller composed prompt and
a faster streaming transcript, the shell-policy admission work, payload-only
history trimming, and the question-widget fixes.

### Added

- `chartreux skills diff [NAME] [--json]` compares the user copies of a skill
  under `~/.chartreux/skills` and `~/.agents/skills` with the shipped copy of the
  same name and renders deterministic sorted unified diffs from the one skill
  inventory `chartreux doctor` also reads. Each case is named plainly: an
  identical copy could be removed after reviewing support files — never
  automatically; a diverged copy; an inactive copy, naming the path and source
  of the copy that wins; or unverified, when the comparison itself fails.
  `NAME` narrows the report to one skill and an unknown name is an error;
  `--json` emits `entries` and `exit_code`. Exit codes are `0` nothing differs,
  `1` a loaded user copy differs, `2` operational error (invalid UTF-8
  included).

- Broken skills are now visible instead of silent. That same inventory records
  typed load failures; `chartreux doctor` prints one `warn` per failure with the
  remediation "fix the file and rerun chartreux doctor", and the interactive
  interface lists the same records as configuration issues. Diagnostics are
  sanitized — no frontmatter values, and no parsed name is needed for the
  message — both surfaces project one list instead of building their own, a
  broken skill never blocks startup, and `/reload` clears the warning once the
  file parses again.

- `/routes` browses the conversation's live `[orchestration.routes]` table, and
  the doctor route section reports the independently loaded configuration of
  the same table. A row names the purpose, seat, configured binding, identity,
  tier, and overlay, the model a marker binding resolves to today (labeled a
  preview, never a launch guarantee), and `requires_distinct_models` beside the
  resolved distinct-model count. Reload validation and launch-time diversity
  stay separate facts, so an all-marker route shows a passing reload beside a
  failing launch-time diversity check, and one unresolvable target keeps the
  other rows intact. A reload that swaps the table updates the open browser.

- The agent transcript's metadata pane carries the completion audit stamp of the
  selected agent's latest run: the audit version, the worker's claim class, its
  satisfied mandates, its declared lines, a runtime-failure line only when the
  run is not healthy, and a rejection reason only when one exists. The wording
  is "Claims matched" and never "Task verified" — a stamp means the claim
  matched evidence, not that the task is correct. The stamp resolves against the
  latest retained run, so a previous run's stamp never rides a new one, and the
  section disappears on re-task, release, expiry, or a missing result; an
  unsupported audit version renders as unsupported, not as a verdict.

- Effective-thinking display. The banner annotates the model as
  `model[medium; API: high]` when the selected level and the provider's API
  effort differ, and shows `API: unknown` when the estimate cannot be resolved
  rather than implying that the requested level is effective. The thinking
  picker explains the difference, and `/status` shows requested, semantic, and
  API effort as three labeled values. The projection is computed server-side
  from the committed deployment and the same message context the next request
  resolves against — the interface keeps no mapping table of its own — and it
  refreshes after completed reasoning, compaction, and rewind. It describes the
  next request: an in-flight call keeps the encoding it started with, and a
  failover may pick another deployment.

- Uncollected background results stay visible. A completed run whose result is
  still retrievable but was never returned by a successful collection call is
  marked uncollected per `(agent, run)`: `get_agent_result` with a result and a
  `wait_for_agent` that returns one count as collection; `check_agents`, the
  browser, a still-running wait, and a cancelled or errored wait do not. The
  agents browser gains an `uncollected: N` row suffix that abbreviates and then
  drops before it can squeeze the state label, `/status` lists the runs in a
  Background Agent Results section, and `check_agents` carries the same run ids.
  The list covers every retrievable uncollected result for that agent, newest
  first, so an earlier result survives reuse and replacement and can be
  collected by pinning its run id; collecting one run clears only that run, and
  the marker expires, releases, or drains with its result.

- Path completions resolve against the session's workspace root rather than the
  process working directory, so a completion stays correct when the two differ.

- Remote command execution is now its own dangerous-command operation,
  `network.remote-exec`, granted by destination rather than in general: an
  approval authorizes one exact `user@host:port` (the user is mandatory, the
  host is lowercased, a missing port canonicalizes to 22, and wildcards,
  ranges, and alias groups never parse). Deny, allow once, allow this session,
  and always for this project all apply; the persisted form is the protected
  user-layer `remote_exec_grants_by_project` field. An egress grant never admits
  a remote command and a remote-exec grant never admits egress or publication;
  session grants do not propagate to children, a persisted destination grant
  authorizes the parent and its children, and children never prompt. The
  approval states its honest scope: remote commands and their effects are
  outside Chartreux's local guards, so the grant trusts the destination, not a
  contained payload. The approval path, the four-choice lifecycle, the
  persistence field, and the child semantics are all in place, but no shipped
  client classifies a destination yet, so the authority is groundwork until the
  remote-command admission lands (see the retained limitations below).

- Two previously denied shell shapes are admitted with guards. A word whose
  leading `=` is neutralized by quoting or escaping and which proves exactly one
  literal argument is admitted as written (`echo '=ls'`, `echo \=ls`), while an
  unquoted `echo ===X===` keeps its equals-expansion denial and its "quote the
  marker" remediation. One same-invocation status probe is admitted — a plain
  simple command, `;`, then `echo $?` or `printf $?` as the entire last
  statement (`ls; echo $?`) — keeping the `$?` source token so the analyzed argv
  never diverges from the executed command. Pipelines, conditionals,
  assignment-only statements, redirects, wrapper payloads, and any statement
  after the probe stay denied.

- Egress modeling for HTTPie's `http` entry point. It leaves the denylist under
  a restricted positional grammar — verb, literal `http(s)` URL, literal request
  items, quoted `==` items — where a loopback destination runs, a non-loopback
  or unparseable destination resolves ask-class `network.egress`, and any
  option, local-file item, non-`http(s)` scheme, missing destination, or
  startup-config indirection is a hard denial with no approval path. Local reads
  and writes are collected and gated before egress authority.

- Public protocol additions: `AgentSummaryModel` gains `completionAudit` (the
  typed audit stamp of the latest retained run) and `uncollectedRunIds` (always
  serialized, `[]` included); `ConfigView` gains `effectiveThinking` and the
  route-inspection fields `routes` and `routesError`; and one server event,
  `callback/childClosed` (`sessionId`, `callbackId`, `reason`), reports a child
  callback's closure to the parent client. Both wire models are strict
  (`extra="forbid"`), so the additions are forward-compatible only: a new client
  accepts payloads from an older app server, while an older strict client
  rejects the new keys and the new event method. Clients and the app server must
  run the same version.

### Changed

- The launch-target route table now renders exactly once, in the `task` tool
  description, instead of in the composed system prompt. Root guidance points at
  that table, and bounded-child guidance names neither delegation nor a table.
  Measured against the pinned tokenizer baseline, the dedup removes 854 tokens —
  about 15% of the fixture prompt. A custom task description receives the live
  table as its own appended section, so a custom skeleton cannot zero it out,
  and a resumed conversation whose persisted bundle predates the change serves
  the deduped prompt.

- Compaction trims the in-memory history it no longer needs to replay: after the
  envelope is saved with the originals present, the pre-boundary prefix is
  sealed under the save lock with a durable receipt, and the reasoning payloads
  of those messages are dropped while text, tool structure, image references,
  and display fields stay. The transcript's own leading bytes — the full
  records, byte for byte — are the archive the descriptor pins, ordinary saves
  rewrite only the live tail, rewind and `/branch` hydrate the archive into a
  complete operation-scoped snapshot before crossing into a new session, and the
  read paths stay display-identical. A failed trim keeps the originals:
  compaction succeeds either way.

- `cat` and `wc` calls whose operands include eligible globs now bind to a
  literal argv plan re-derived at execution time and run through a no-shell
  transport, with every planned operand re-opened through a component-wise
  `O_NOFOLLOW` walk and its device and inode re-checked. The shell cannot expand
  an operand between the permission check and the read: mutation, symlinks,
  non-regular files, enumeration errors, and platforms without those primitives
  fail closed, and the plan never falls back to the command string. The
  transport is scoped to that glob whitelist — `ls` listing globs, `grep`
  pattern globs, and every admission above keep shell execution.

- `ask_user_question` is denied in a bounded child and withheld from the child
  tool catalog, so a child returns a blocker naming what it needs instead of
  opening a question nobody can answer. Root behavior, runtime approvals, and
  the parent's own question flow are unchanged.

- The shipped skills were corrected against the runtime they describe: route
  terminology in the design, plan, and review skills; the sub-agent skills'
  assigned-root search boundary; the sub-implementor's bounded-read and
  edit-recovery guidance; the sub-verifier's runtime corrections; the
  sub-summarizer's Common Lisp typo; the git-workflow skill's clean-tree
  checkout exemption; the test generator's optional verification dispatch; the
  debugging skill's runtime sync; the web-search usage wording; and the builtin
  command skill's managed-job claim.

### Performance

- Streaming a long bullet-list answer no longer grows the widget tree with the
  list: completed items become cached strips served by one canvas widget keyed
  by effective width and style generation, only the active item stays a live
  widget subtree, and any document that is not a simple shape keeps the
  whole-document path (rendered-text equivalence is asserted at every prefix).
  On the 400-item fixture the frozen ceilings — 30 s wall, 250 ms frame p95,
  20 ms loop-lag p50 — are now enforced and measure 1.9 s, 64 ms, and 1.5 ms.

- Persisted-transcript verification releases the GIL every 128 records in both
  its comparison phase and the drain that follows, so a full verify no longer
  starves the render loop. The release ceiling on combined-phase flush-p95
  degradation was set at 65% by user adjudication: every save now also runs the
  archived-marker admission scan of the history-trimming chain — required
  fail-closed work on the same measured window, not removable by the verify
  cadence — and the redesign that would move that guard off the render loop is
  deferred to the next release.

- Session metadata is serialized once per content: a process-lifetime content
  hash gates the prompt and tool metadata dump, so repeated saves with unchanged
  metadata stop re-serializing it, a changed hash still rewrites, and a restart
  re-serializes once before the hash rebuilds.

- Usage-ledger appends stop rescanning the ledger: a per-ledger attempted set
  and a confirmed-durable seen set make a first append read nothing and a
  confirmed retry short-circuit, while a retry after an unconfirmed append keeps
  the full-scan path (a landed record is reported as already present, never
  duplicated). The 1k-record burst drops from 2472.73 ms to 134.85 ms.

### Fixed

- A question widget whose callback died on the server is dismissed instead of
  trapping the pending queue: `NOT_FOUND` and `CALLBACK_CLOSED` close the
  widget, advance the queue, and offer no resend hint, while `INVALID_PARAMS`
  keeps the draft-and-retry branch and `CONFLICT` keeps its own boundary (now
  pinned). The interface also unmounts question widgets on any non-open
  projected status, not on an expired one only, and consumes the new
  child-closure event identity-matched on `callbackId` — root ids and mismatched
  owners are ignored, a duplicate closure is idempotent, and a late closure
  never dismisses a newer widget or clears its input.

- Agent inventory updates are serialized and coalesced: one delivery is in
  flight per registry, the payload is projected immediately before it is sent
  rather than from an older snapshot, and a caller that arrives during a
  suspended delivery queues its eviction data for the next pass, so an
  `agents/update` can no longer publish a stale inventory or drop an eviction.

### Breaking

- HTTPie's `https` entry point is denied fail-closed by the shipped
  `[tools.bash]` denylist. It is the request client with the https scheme
  defaulted — an unmodeled twin beside the now-modeled `http` — and it has no
  documented, corpus-tested grammar to admit under, so a plain remote
  destination such as `https http://attacker.example/` resolves denied with
  "matches denylist pattern 'https'" instead of resolving beside the `http`
  destination model. Loopback destinations are denied too: no auto-allow rides
  beside the modeled client. Grammar modeling is deferred to the v0.8.0
  backlog; removing the entry re-enables the client as an ordinary denylist
  edit.

- The composed system prompt no longer carries the launch-target route table; it
  lives in the `task` tool description. Anything that compared, pinned, or
  patched the prompt text must read the table there, and a conversation resumed
  from a bundle composed before this change serves the deduped prompt.

Pre-release alpha: there is no migration path, and this release carries no
separate upgrade guide. The user-visible narrowing is described in this section
and in the retained limitations below.

### Retained limitations

- Egress narrowing stays where it was: `httpie`, `wget2`, `https`, `ftp`,
  `telnet`, `aria2c`, `lftp`, `watch`, `ssh`, `scp`, and `rsync` are all denied,
  loopback destinations included. `wget2` stays denied because no documented,
  corpus-tested redirect-disable exists for it, and its recursive retrieval,
  input and config files, and output paths are pinned as denials rather than
  admitted. The FTP PASV residual stays disclosed for the modeled `wget` rather
  than gated.

- Remote-exec authority is a destination grant, not a payload guarantee: the
  prompt says so, and the grant trusts the destination. No remote-command client
  is admitted under it in this release — `ssh`, `scp`, and `rsync` remain
  denied, so no grant can be exercised yet — and the hostname-destination
  admission with its resolve-validate-pin guarantee is not shipped, so nothing
  may be assumed about hostname pinning. Loopback and local-machine destinations
  cannot be reached through this path because every spelling is denied.

- The admission boundaries are exact. S1 admits only the quoted or escaped `=`
  marker that proves one literal word, and one same-invocation `$?` probe with
  nothing after it. Glob execution admits only `cat` and `wc` with no options and
  an operand that is either literal or an eligible glob — an unquoted `*` or `?`
  only in the final component under a literal authorized parent — and it denies
  brackets, `**`, braces, tilde, mixed quoting, a no-match, option-like results,
  matches beginning with `-`, directories, non-regular files, and symlinks. The
  work budgets stay frozen at 4096 directory entries, 524,288 match states, 256
  operands, and 64 KiB of argv, and exhaustion denies instead of truncating.
  `ls` listing globs and `grep` pattern globs keep their existing shell
  execution.

- Structural tombstones are deferred: this release trims only the pre-boundary
  `reasoning_payloads`. Dropping text, tool structure, or image references from
  old messages waits until the retained-memory benefit is measured and the loss
  of pre-boundary conversation rewind is a deliberate product decision.

- File-only restore does not exist. Historical checkpoints are memory-only and
  the restore operation is coupled to conversation rewind; trimming keeps the
  checkpoint data but ships no separate file-only restore command or UI.

- Cold-resume memory peaks are unchanged: a cold resume loads the complete
  records through the unchanged legacy path, so a resumed session's peak is its
  full history's. Trimming reduces what the running session holds, not what a
  resume loads.

- No process-wide memory bound is claimed. The trim releases the payloads of the
  sealed prefix; it is not a statement that a session's or the process's total
  memory is bounded.

- Trim savings are unmeasured and workload-dependent, and they come only from
  the reasoning payloads: tool outputs and visible text remain retained.

- Archive unavailability is classified, and one class is terminal. An
  authority-state failure — a descriptor whose epoch, prefix length, or retained
  fields no longer match — fails the save with a typed diagnostic and recovers
  by restarting the session. Missing, truncated, or changed archived bytes are
  unrecoverable in place: the history is not rebuilt from the trimmed messages,
  the diagnostic says so, and a new conversation is required.

- Two accepted behaviors of trimming in this release: it re-arms only after a
  cold resume or a new session, because a warm in-process resume keeps the
  old-epoch descriptor and the next seal refuses as stale with the originals
  retained; and only the manual `/compact` trims, because an automatic mid-turn
  compaction never does.

## 0.6.1 (2026-10-31)

### Added

- OSC 7501 program-status reporting for the interactive interface: while the
  TUI runs, it reports `idle` at the prompt, `working` during an agent turn,
  `blocked` while waiting on a user question or a permission prompt, and `done`
  or `error` at turn end, so a supporting terminal can show what the program
  is doing. Reporting is on by default and controlled by the new
  `emit_program_status` key (or `CHARTREUX_EMIT_PROGRAM_STATUS`); headless,
  programmatic, and ACP runs emit nothing, and terminals without OSC 7501
  support ignore the sequence.

- The OSC 7501 record Chartreux writes is the terminal's root record, so a
  supporting terminal shows program status as the status of the pane Chartreux
  runs in. Chartreux also clears that record from the terminal on exit —
  including `done` and `error` records — as the first statement of shutdown,
  with a bare `state=clear` that removes every OSC 7501 record on the terminal,
  Chartreux's own root record and any child records other programs wrote
  included. That is a deliberate, documented divergence from spec rev 0.4,
  which recommends (as a SHOULD, not a MUST) that `done` and `error` records
  survive the exit of the program that wrote them: a stale status at the prompt
  is the worse outcome. A `SIGKILL` never reaches shutdown, so the record
  survives it.

- Public protocol additions: `UserInputCallbackDetail` gains `origin`
  (serialized as an explicit `"origin": null` when unset) and `ConfigView`
  gains `emit_program_status`. Both wire models are strict (`extra="forbid"`),
  so the additions are forward-compatible only: a new client accepts payloads
  from an older app server, but an older strict client rejects the new keys.
  Clients and the app server must run the same version.

### Changed

- Generation retirement now drops the read-only snapshots of
  logging-disabled subagent children (`_readable_children`): when a root
  session generation is retired, the closed in-memory snapshots belonging to
  it are released instead of being retained for the registry's lifetime.
  Stored results, tombstones, and children with on-disk transcripts (read
  through their stored links) are unaffected. Accepted degradation:
  logging-disabled children have no disk transcript, so after their
  generation is retired their transcripts become unreadable — transcript
  reads fall to "no transcript", child lookups raise, and the grant ports'
  session resolution returns None.

- Adaptive streaming cadence for live assistant output: the write delay is
  no longer a fixed 50 ms. Under sustained load the pump can run a frame
  late, which doubled the visible frame period, so each frame now measures
  the period it actually achieved and steers the next delay toward the
  50 ms target, clamped to 25-150 ms. Streaming output therefore paces
  itself instead of waiting a constant interval: the scheduled interval can
  drop toward the 25 ms floor when a frame arrives late (back-pressure
  shortens the next delay so the achieved period stays near the 50 ms
  target) and can rise toward the 150 ms cap when frames arrive early.
  Visible text can appear sooner than the old fixed cadence allowed, and
  the worst-case scheduled interval is now 150 ms rather than 50 ms.

### Performance

- Long-session transcript windowing: the per-tick reconcile no longer scans
  the whole registry. Admitted entries resolve without bulk admission, the
  visible window resolves from a scroll-invariant ordered placement index,
  and the reconcile's membership indexes (mounted, live, live-dirty,
  neighbours, placeholders) are maintained incrementally instead of rebuilt
  each tick. Measured on the registry fixture, unchanged-tick cost is flat
  across 1k/10k/50k entries (the perf suite asserts the frozen bounds:
  scaling <= 2.0x, tick <= 5 ms, allocation plateau <= 0.05 MiB). The shadow
  index sets also shed their peak-sized hash tables after membership spikes.
- Persistence memory: full transcript verification streams the persisted
  records and compares them one by one with exact parsed-dict equality,
  instead of loading the whole transcript and deep-comparing message lists;
  at 20k records the verify worker's peak dropped from about 20 MiB to
  about 8 MiB. Repair rewrites now stream through the atomic-replace writer
  instead of materializing the rewritten payload, and checkpoint file
  states are interned by (path, digest) with byte-equality confirmation.

### Fixed

- Full-verify transcript line splitting: the streamed reader reads records
  through the file's universal-newline iteration (`\n`, `\r\n`, `\r`)
  instead of `str.splitlines()`, which additionally splits on U+2028,
  U+2029, `\x0b`, `\x0c`, `\x1c`-`\x1e`, and `\x85`. A persisted
  transcript whose records embed one of those characters inside a JSON
  string is no longer misread as corrupt on save: it verifies clean and is
  not quarantined and rewritten. This changes corrupt-transcript
  classification for such transcripts, which previously lost their
  persisted boundary; genuinely unparseable lines still quarantine exactly
  as before.

- Full-verify transcript encoding detection: the streamed reader opens the
  transcript as raw bytes and decodes with the codec its byte-order mark
  selects (UTF-8, UTF-16, or UTF-32), as the bulk reader's `read_safe` did,
  instead of forcing UTF-8. A BOM-marked UTF-16/UTF-32 transcript verifies
  clean and keeps its persisted boundary instead of being quarantined as
  corrupt. Residual difference: text without a BOM is still read as UTF-8,
  so `read_safe`'s heuristic fallbacks (locale preference, charset
  detection) are not reproduced — a non-UTF-8 transcript carrying no BOM
  that the old bulk reader could guess still classifies as corrupt.
  Mixed-encoding window: the writer still emits UTF-8, so a BOM-marked
  non-UTF-8 transcript that verifies clean read-only becomes a
  mixed-encoding file as soon as the next incremental append is written to
  it, and the following full verify quarantines that file exactly as
  before.

- Transcript fingerprint admission no longer raises on a payload string
  holding a lone surrogate (what JSON-decoding an escaped `\ud800`
  produces): the canonical UTF-8 encoding failure now maps to the
  always-changed fallback, so the entry is admitted and re-rendered
  instead of aborting the window update. Persistence is forfeited for
  such payloads: the writer's UTF-8 serialization still cannot encode a
  lone surrogate, so while such a payload is in history every save fails;
  the writer's format is unchanged.

- Transcript fingerprints no longer collide for fold-ambiguous datetimes:
  two datetimes with the same wall-clock time inside a DST fold (`fold=0`
  vs `fold=1`) are different to pydantic but normalize to the same UTC
  instant, so they previously shared a fingerprint and a window update could
  be skipped for a changed entry. Fold-ambiguous values now compare
  always-changed, and datetimes whose UTC conversion overflows (extreme
  offsets near year 1 or 9999) fall back to always-changed instead of
  aborting admission.

## 0.6.0 (2026-10-28)

This is a combined release: everything shipped since v0.5.8 — the
permission realignment, the backlog design set, and the subagent
completion audit — ships together as v0.6.0.

### Security

- Destination-aware egress is the settled exfiltration decision for the
  modeled network clients. Modeled `curl` and `wget` calls are no longer
  denylisted outright; the destination decides. A literal loopback
  destination (`localhost`, `127.0.0.0/8`, `::1`) runs with no dialog; any
  other destination — another IP literal, a DNS name, or one the recognizer
  cannot parse — resolves ask-class (`network.egress`) and prompts in
  interactive sessions, denying with remediation in headless and child
  sessions. The destination-changing behavior classes resolve ask-class even
  for a literal loopback destination: redirect-following flags (`curl -L`),
  `wget`'s default redirect behavior, and destination-overriding options
  (`--resolve`, `--connect-to`). The development-egress carve-out stays
  run-class (`git fetch`/`clone`/`pull`, `pip`/`uv` installs), and a user
  denylist entry for these clients remains an absolute veto.
- Probed egress clients move onto the shipped default denylist: `ssh`,
  `scp`, `rsync`, `ftp`, `telnet`, `http`/`httpie`, `wget2`, `aria2c`,
  `lftp`, and `watch` are denied until they are modeled; remove an entry to
  re-enable a client deliberately.
- Recursive `rm` gains a git-status-aware carve-out. Over a tree with
  unsaved changes the blanket handoff denial remains. Over a git-clean tree
  a recursive `rm` runs on its own only when every expanded operand lies
  within the assessed repository, clear of its Git metadata (neither
  encompassing it nor lying inside it, linked-worktree common directories
  included), with recovery established for every repository involved, nested
  ones included. The analysis fails closed on every axis — operand scope,
  metadata overlap, recovery status, expansion, resolution — and any mixed
  operand set fails closed to the handoff denial.
- Artifact secret scan: a gitleaks-rule-based scan with entropy checks over
  the session's touched paths runs automatically at the agent submission
  boundary — a finding blocks the submission, the match is redacted from the
  transcript, and the agent is told the artifact failed the scan (not what
  it matched) — and runs as a pre-commit hook for user-driven commits.
- Child sessions deny ask-class dangerous commands outright, regardless of
  user-input capability: the dangerous-command approval dialog is never
  surfaced in a child, a denied ask-class call returns a blocker to the
  parent, and `dangerous_command_grants_by_project` is the only child
  pre-authorization path (saved project grants still run without a prompt).
  The same blanket child semantics apply to out-of-root access grants.

### Added

- Completion audit (fabricant-proof) at the subagent submission
  boundary. Every background subagent run now carries a typed, frozen
  completion contract, admitted at launch and rendered into the worker's
  prompt. The contract binds closed vocabularies of acceptance-check
  templates (`pytest`, `ruff_check`, `pyright`, `git_head`, `git_status`,
  `git_diff_stat`) and artifact kinds (`commit`, `touched_path`,
  `check_pass`, `envelope`), a strict versioned envelope/receipt grammar
  (unknown fields, duplicate requirement ids, and oversized payloads are
  grammar violations, never worker faults; substantive seats must answer
  with a claim, read-only seats are exempt), and a claim-evidence truth
  table on five separate axes — claim fidelity, disclosure completeness,
  mandate satisfaction, declared outcome, and primary runtime failure.
  The audit composes at `build_result` from three evidence audits: the
  artifact audit (claimed commit hashes and artifact paths resolved
  against the run's evidence, with admission-versus-publication HEAD
  snapshots and hash-prefix matching confined to git-command results),
  the receipt audit (envelope-linked receipts only; no prose
  extraction), and the check audit (contract-bound requirement ids and
  their canonical commands; exit-zero evidence after the last mutation).
  Verifier failures (denied, timeout, unavailable, unrunnable) are
  informational, never worker misconduct; unavailable evidence is never
  a contradiction.
- Completion-evidence chronology with collection-time snapshots. Each
  run keeps a bounded, per-run chronology of post-tool evidence
  receipts, paired to their originating invocations by a harness-minted,
  run-scoped receipt occurrence id (never `tool_call_id` alone —
  tool-call ids are reused across turns), with collection-time content
  snapshots (hash, byte size, sampled head/tail) captured at collection,
  never at finalization; evicted evidence is marked unavailable, never
  contradictory.
- Once-only finalization and the acceptance-check runner. Acceptance
  checks run exactly once per run, awaited immediately before each
  `build_result` call site and cached run-keyed; cancellation and
  exception paths never start new verification, publish no completed
  result before the audit completes, and leave no orphaned verifier. The
  runner builds its argv from the frozen contract's templates, validates
  paths under the child's authorized roots, and passes full shell
  command admission — guardrails, path permissions, dangerous-command
  classification, denylist — immediately before spawn (resolve once,
  spawn the resolved target; no prompts, retries, or fallbacks; a
  denial stamps the check denied without spawning).
- Enforce mode is the default: `[completion].mode = "enforce"`. A
  contradicted claim blocks publication at the submission boundary: the
  run finalizes `completed=False` with a typed `completion_rejection`
  failure class (mirrored on the public wire), a measured rejection
  envelope, and the audit result riding `TaskResult.completion_audit`.
  Enforce-mode activation wired the composition's two inert-input
  surfaces into production: parent-shared roots frozen on the contract
  at admission (a failed finalization check observing final state under
  a shared root — the session scratchpad the parent and the child both
  write — composes as check-context unverified, never contradicted,
  never a block), and the receipt's bounded claimed-value surface
  (claimed commit hashes and claimed artifact paths, keyed by
  requirement id), which the artifact audit resolves against the
  evidence. The activation gates — the offline replay corpus, the
  post-contract observation gate, and the fresh-runtime enforce-mode
  test — ran green with zero honest false rejections; `stamp` remains
  available and only stamps the composed classification onto the result
  without blocking.
- Un-stick machinery for stalled orchestrator endpoints. An explicit
  `override_cooldown` on `task` and `resume_agent` launches lets an
  accepted request force a completion attempt past an endpoint's
  cooldown, metered by a root-shared, endpoint-episode cooldown ledger
  (carried unchanged through every `ConfigOrchestrator.copy()`): at most
  2 forced starts per endpoint episode, at least 30 seconds apart, at
  most one attempt in flight, with the pressure bound
  `min(2, 1 + floor(T/30s))`. An episode opens only on a measured
  endpoint failure and closes only on a measured endpoint recovery —
  agent lifecycle, replacement, release, timer expiry, and elapsed time
  never replenish the counters, and automatic paths never acquire
  permits. A permit bypasses only its matching cooldown predicates at
  candidate selection and is consumed once at the actual transport
  start — no retries, no empty-response replay, no failover replay for
  the forced attempt. Denied overrides return bounded diagnostics
  (selected deployment, bypassed exclusions, remaining allowance, next
  permitted time) and consume neither budget nor allowance. The `task`
  and `resume_agent` tool contracts document the bounded override, the
  deadline exception, the diagnostics, and the no-replay rule.
- Resume-progress deadline for stalled resumed runs. A resumed subagent
  run that stalls on its first model call (the provider accepts the
  request and never streams a byte) now trips the one hard in-flight
  kill in the recovery policy: a run-scoped, run-armed
  first-semantic-progress deadline
  (`[recovery.subagents].resume_progress_deadline_seconds`, 180s by
  default), armed at the run's first conversation model call and disarmed
  at the observation of the first semantic delta (content, reasoning, or
  tool-call; heartbeat and usage-only chunks never disarm). Expiry
  cancels the in-flight call and classifies as the measured
  `resume_progress_deadline` timeout reason; automatic same-payload
  continuation is prohibited.
- Transactional explicit resume. Explicit resume admission is now
  transactional: an unavailable admission is rejected before consuming
  a resume or publishing RESTORING; on committed admission the pending
  automatic timer is cancelled and the resumed-progress context is
  installed.
- Resume lineage in the agent bar: the details pane and the agent
  browser render `resumed_from_run_id` with full metadata.
- Worker and orchestrator contracts. The bounded-child and coder prompts
  carry the evidence-linked completion clauses (cite commit hashes and
  receipt references at completion; never discard a result you issued;
  re-issue on conflict; the no-edit proof obligation; file-state metadata
  is non-authoritative), and the orchestrator's dispatch policy retains
  implementation predecessors for replacement lineage, escalates the
  first typed audit rejection through the existing lineage, reruns the
  discriminating check before consequential reliance, and routes
  release/publish/destructive/migration/git-state scopes off the flash
  tier (budget fit is advisory-log only).
- Documented residuals and scope exclusions: a stamp means "claim
  matched evidence," never "task correct"; the acceptance check executes
  project code (conftest, test bodies) and authorized-root validation is
  not containment; managed-job (`bash_start`) and custom/MCP writes are
  excluded from the evidence scope.

- Shell mutation-roots footprint. `bash` and `bash_start` surface the
  resolved work-tree roots of their commands' checkout/switch invocations
  (`mutation_roots`) alongside the write-path footprint — on success and,
  copied onto the error event by the tool executor, on failed, timed-out,
  or cancelled runs alike. The footprint is collected into mutation-scope
  accounting next to the touched paths: it never joins the artifact secret
  scan's path list (a work-tree root would scan the whole repository and
  burn the bounded walk's budget), and the completion audit consumes it
  for its touched-path comparisons. A footprint that cannot be
  established records an unknown-mutation-root sentinel, never an
  established empty one.

- Local tool trust: tools authored under `~/.chartreux/tools/` are trusted
  by file location and admitted to the non-mutating posture; project-discovered
  tools, `tool_paths` entries, and MCP proxies stay fail-closed. The
  provenance-stamp machinery is removed; shipped builtins are classified by
  their defining module.
- Project environment injection: inside a trusted folder, the project `.env`
  and `$CHARTREUX_HOME/.env` load into `bash`/`bash_start` subprocess
  environments (re-read on mtime change so rotation takes effect without a
  restart), with known secret values masked in tool output as a redaction
  backstop; document expected key names in a committable `.env.example`.
- Live route editing (ADR 0022, promoted to `docs/adr/`): route edits apply
  to subsequent launches in the same conversation instead of new
  conversations only; in-flight children keep their admitted bindings, and a
  resumed conversation adopts the current saved route table with a diff
  notice.

### Changed

- Pure cooldown/endpoint-cooldown launch-exclusion sets are normalized
  as transient launch exclusions (an additive classification; existing
  exclusion records are unchanged).

- Identity tool grants: the read-only-posture identities that bear
  delegation (`orchestrator`, `researcher`, `advisor`) gain a settled
  personal execution carve-out — `bash` for probes and verification,
  `bash_start` for managed jobs, and `write_file`/`edit` for scratchpad
  and deliverable persistence. Repository-source edits remain delegated
  by role contract, not by tool denial: the carve-out is personal-axis
  only and never reaches a delegation set. The shipped identity prompts
  are reconciled to match — "read-only: never modify repository source
  files" replaces the blanket "never modify files".

- State-aware clean-tree checkout exemption (ADR 0019 amendment).
  `git checkout <branch>` to an existing local branch and
  `git checkout -b <new> [<start>]` run without the consequential
  approval flow when the entire composed shell call is exactly that one
  eligible invocation and every candidate repository scope passes the
  state guardrails — a fully clean work tree, no in-progress operation
  marker, and no repository configuration that selects a program the
  invocation could run — with the plain branch form additionally
  requiring the target to exist as a local branch (a sanitized
  `show-ref` probe, so a pathspec fallback, a tag, a remote-tracking
  name, or a commit-ish never rides the exemption). Because the bash
  tool executes a command string, the executed command is rewritten at
  execution time with `--no-overwrite-ignore` inserted after the
  `checkout` subcommand, so an ignored-file collision fails closed
  instead of silently clobbering local content. Every ineligible shape
  and every unestablishable state fails closed to the existing
  consequential approval flow; the exemption never bypasses the denylist
  veto, parent ceilings, or any other recheck, and the pure classifier's
  `git.checkout` classification is unchanged.

- Git workflow policy (AGENTS.md, layered on ADR 0019's runtime gates):
  rewrites of unpublished history — rebase, `commit --amend`, and
  `reset --soft`/`--mixed` on a branch whose HEAD no remote-tracking ref
  contains — are agent-executable; rewrites of published history remain
  handoff-only and approval-gated, with the publication gate checking
  the operation's entire affected commit range against remote-reachable
  history; when a branch's publication status is uncertain the agent
  stops and hands the operation to the user rather than rewriting on
  uncertainty; a destructive checkout is one that discards uncommitted
  changes (a clean-tree checkout of an existing local branch is not
  destructive); and the hard guards — `git update-ref`, forced and
  deleting pushes, and history rewrites of published commits — keep
  their existing gates.

- Wrapper and flag recognition: `sg` and `sudo -u <user>` are transparent
  wrappers whose payloads are re-analyzed (including `sg`'s unquoted
  multi-token `-c` payloads, closing a fail-open); `git symbolic-ref`
  resolves in read mode and is classified by mode; `.venv/bin/python
  --version` resolves while bare interpreter invocation stays denied; a
  `VIRTUAL_ENV=...` prefix no longer denies.
- Every grammar-rejection denial names a compliant alternative inline —
  across the variable/subshell-expansion, glob-inspection, zsh
  equals-expansion, heredoc, and for-loop rejection categories.
- The `skill` tool is reclassified as session-scoped instruction loading
  (not a mutation) and `todo` as control-plane, so read-only and orchestrator
  identities admit them; brief-writing and skills-preamble guidance now
  directs children to load named skills via the `skill` tool, and the manual
  skill-file read fallback is retired.
- The reviewer identity gains a bash execution carve-out for non-mutating
  verification commands, so a review exit gate can run the checks it names.
- The shipped `implementation`, `mechanical-edit`, `search`, and
  `verification` route defaults are cost-tier aware: they no longer bind
  `$session` and default to the cheapest capable catalog model.
- The demanding route triggers on difficulty with a structured difficulty
  category field carried in launch metadata; record-only categories fail
  admission.
- The ceremony acceptance lattice gains the plan-granularity clause: an
  accepted plan authorizes its named work packages in sequence, and a
  completed package does not require fresh acceptance to begin the next.
- The git reader subcommand table and the unmodeled-builtin table no longer
  overlap (dispatch-order-aware dedup).
- `bash_read` bounds (`wait_seconds` at most 30, `max_bytes` at least 4096)
  are named in the parameter-schema descriptions and surfaced in the
  validation-error remediation.
- The wrapper-class audit closed with an explicit disposition for every
  payload-runner and egress-client entry (modeled transparent, denylisted,
  or fail-closed), one corpus row per entry.

### Fixed

- The completion-evidence receipt sink is wired into production runs:
  it is installed on the child loop at run admission on both paths
  (background and foreground) and cleared at run teardown — previously
  every post-tool receipt was minted and discarded, so the receipt audit
  found no matching command.
- A denied `override_cooldown` mint on a retained agent no longer
  strands the record as RUNNING: the minted permit is released and the
  reused claim rolled back on any failure.
- Shell-permission tests fail loudly on stray git repositories above
  the tmp tree (a `/tmp/.git` left by a crashed tool run, for example):
  an autouse guard in `tests/tools/conftest.py` detects the pollution
  that silently re-labeled scope-derived assertions under parallel
  (xdist) schedules. The guard only detects; it never deletes or
  modifies anything.

- Oversized TUI error messages are capped behind the `+N lines`
  disclosure: an error widget estimated taller than 12 wrapped rows
  renders as a one-line summary plus the disclosure with a height-capped
  body, and the turn supervisor truncates the error payload at 4000
  characters before it reaches the TUI — a multi-kilobyte API-error
  excerpt can no longer fold-wrap into dozens of rows and stretch or
  obscure the pane.

- Test-suite flake fix: the sustained-ticks transcript memory test's
  timeout bound is raised from the repo-global 10 seconds to 30 — its
  legitimate runtime under load can exceed the global bound while
  remaining bounded.

- Mouse capture is released before the agent-transcript viewer disposes, so
  exiting the viewer no longer leaves the UI mouse-dead with the keyboard
  alive.
- Three latent bugs in the same teardown path: a dispose exception can no
  longer leave the chat permanently hidden, detached-widget hit-race delivery
  uses the filtered widget result, and the detach-for-reparenting registry no
  longer leaks (`is_mounted` is false for detached widgets).
- The agent-transcript live tick is bounded — streaming teardown and O(page)
  page sizing — so a large transcript cannot saturate the Textual message
  pump or grow memory monotonically.
- Generic-keyword assignment no longer false-positives on a function-call
  right-hand side.

### Breaking

- Breaking (pre-release alpha; loud, actionable errors; no silent
  migration): modeled `curl` and `wget` now resolve per destination instead
  of denying outright — a non-loopback destination surfaces a
  `network.egress` ask in interactive sessions and denies with remediation
  in headless and child sessions, and the probed egress clients listed above
  are newly denied by the shipped default denylist. Tooling or workflows
  that relied on the outright denial must re-authorize deliberately.
- Child sessions no longer surface the dangerous-command approval dialog
  under any capability: an ask-class call in a child resolves to a blocker
  returned to the parent. Pre-authorize children through
  `dangerous_command_grants_by_project`.
- The provenance-stamp machinery is removed. Tools that relied on a
  provenance stamp for admission must instead live under
  `~/.chartreux/tools/` (file-location trust) or ship as builtins classified
  by their defining module.

### Retained limitations

- The probed egress clients remain denylisted until they are modeled;
  modeling them is the fast-follow.
- The bare-interpreter gate is retained: bare `.venv/bin/python` still
  denies while `.venv/bin/pytest` (a tool, not an interpreter) passes.

## 0.5.8 (2026-10-23)

### Security

- Redirect targets that contain shell glob characters are denied fail-closed:
  the shell would pathname-expand an unquoted redirect destination before any
  later check, so the denial now happens at permission resolution with a
  message naming the offending target. Quoted redirect targets are restored
  as literal operands and are never pathname-expanded.
- Unquoted glob-bearing `find` pattern operands are denied fail-closed: the
  shell pathname-expands an unquoted pattern word before `find` runs, so a
  decoy file named `-exec` (or `-a`, `-delete`, ...) in the working directory
  injected an execution predicate through the pattern-operand carve-out —
  `find . -name -* printf X ';'` ran the injected `printf` despite the
  execution-predicate guard. The carve-out now covers quoted and literal
  pattern operands only; an unquoted glob-bearing operand is denied with the
  generic path-glob diagnostic, exactly like a redirect target that looks
  like a glob.
- The inspection surface of the `grep` family (`grep`, `egrep`, `fgrep`,
  `rg`) is narrowed: unmodeled options are denied, option-like words are
  never mistaken for file operands, and file operands are inspected like any
  other path, quote-aware.
- `grep` output caps are UTF-8-byte-honest: `max_output_bytes` truncates on
  encoded byte boundaries and never emits a partial character, in plain,
  files-only, and path output alike.

### Added

- Per-run budgets for subagent runs: `[subagents].run_max_price` and
  `[subagents].run_max_session_tokens` (with per-session overrides) arm a
  per-run budget middleware measured relative to each run's admission
  baseline; an over-cap run stops with the budget-exceeded outcome and a
  response with unknown pricing stops with budget-unverifiable — unless the
  run's admission baseline already carries unknown cost: a reused or
  replacement run that inherits a latched unknown-cost baseline cannot
  detect subsequent unknown usage through that latch, so unknown pricing
  recorded after such a baseline does not stop the run. Run budgets are
  checkpoints, not reservations: nothing is reserved in advance. The
  root-loop arming semantics, resume anchor preservation, and the nullable
  foreground attribution are documented in the subagents guide.
- Usage ledger schema v2: content-class attribution, an attempt-ordinal axis
  for replay weighting, and a run-id join key on ledger attribution. v1
  ledgers still parse.
- Opt-in tool-output shapes, both default off and identical to the plain
  modes when off: `read_file` outline mode returns a symbol outline with
  revision pairing for Python files (falling back to a plain read
  elsewhere), and `grep` files-only mode returns deduplicated paths under
  the same byte caps.

### Changed

- Expansion-failure provenance: a denied path glob that failed during
  expansion inspection names the specific limit it reached — the
  directory-entry budget, the match-state budget, or unsupported glob
  syntax — in a bounded, redacted diagnostic field. Denials decided outside
  the expansion inspection and tokens the inspector never expanded render
  the generic path-glob message; verdicts and caps are unchanged.
- Empty-response replay exhaustion propagates through the middleware
  budget-stop path: an empty model response whose in-call replay the run
  budget denies now settles as a terminal budget stop (budget-exceeded or
  budget-unverifiable with a `budget-stopped` parent notification) — not a
  suspension and not a provider failure.
- Root run-budget interval semantics are pinned by characterization tests:
  session options arm the root loop, an unanchored root self-initializes its
  anchor on the first evaluation, a resume preserves existing anchors, and
  the subagent defaults alone do not arm the root.
- Review-system alignment: deep review panels compose all configured seats
  (the shipped route composes three seats; a configured five-seat panel
  launches five), no deep authorship substitution — the material-author
  model-line fallback applies only to quick single-seat reviews — the
  review-execution overlay pins a receipt policy for consumed versus
  independent evidence, and rendered purpose guidance is roster-neutral.

### Breaking

- Breaking (pre-release alpha; loud, actionable errors; no silent
  migration): `zai-glm-5-3-flash` now uses the exact GLM family
  thinking-level table. Selecting thinking `off` for flash is now a typed
  launch-time configuration error (`InvalidLaunchThinkingError`) instead of
  mapping to the provider table's no-thinking value. Selecting thinking
  `low` now wires real thinking (`reasoning_effort: "low"`): flash runs
  consume reasoning tokens where the previous mapping sent none — a cost
  change.
- The glob-denial message wording changed: a denied path glob that failed
  expansion inspection now appends the provenance detail described above
  instead of rendering the generic path-glob message alone. Tooling that
  matched the exact prior wording must accept the appended detail.

### Retained limitations

- Retained shell-policy trade-offs, each with its fast-follow pointer (see
  the retained-limitations section of the tools-safety guide): the
  trailing-`**` over-denial (until the executor shell-option contract
  lands), the extglob residual hazard (detection and modeling fast-follow
  behind the same contract), the quoted-variable conservative denial
  (resolution fast-follow), and the find pattern-operand carve-out, now
  narrowed to quoted and literal pattern operands (the unquoted-glob
  predicate-injection hazard is closed in the security group above; the
  quoted/literal bypass is retained because the shell never expands
  them). Glob-expansion work-budget caps are unchanged in this release;
  raising them is deferred and remains a separate decision.

## 0.5.7 (2026-10-22)

### Added

- Identity-based agent architecture: the profile/slot/prompt-ID
  configuration axes are replaced by three orthogonal `[agent]` selections —
  `identity` (default `coder`), `workflow` (default `minimal`), and
  `orchestration_strategy` (default `none`) — with five built-in identities
  (coder, orchestrator, researcher, reviewer, advisor) carrying typed
  delegation and tool-posture metadata. A single-model setup needs no agent
  configuration at all, and default routing is identical with one model or
  ten: every shipped purpose route binds the session model, never a
  roster-size guess. Selecting the orchestrator is one line (`[agent]
  identity = "orchestrator"` or `chartreux --identity orchestrator`): it must
  delegate — non-mutating personal tools, full-granted delegation — and is
  self-sufficient without a strategy body; the delegation obligation lives
  in the identity, never in a selectable strategy. Custom identities are new
  prompt files in the user prompts directory (`user:`-qualified or bare);
  workflow bodies live in `~/.chartreux/workflows/` and orchestration
  strategies in `~/.chartreux/orchestration_strategies/`, all under one
  strict loader contract (plain file stems, non-empty UTF-8, 64 KiB cap, no
  traversal, `builtin:`/`user:` qualification, reserved shipped names cannot
  be shadowed).
- Deterministic startup prompt assembly in a fixed nine-step order: the
  identity opening at character zero, harness-guaranteed universal critical
  instructions and instruction hierarchy (a custom identity cannot drop
  them), identity behavior, execution context (interactive main, headless
  main, bounded child), the workflow body, the strategy body (absent for
  `none`), shared references, AGENTS.md layers with canonical-file
  deduplication (the surviving entry keeps the project label with both
  provenance positions recorded), and a short closing re-anchor. Gate and
  acceptance ceremony moved wholly into the `ceremony` workflow — the
  composed default carries none of it — and shipped workflow skills align
  with the axis: ceremony-bearing skills load only for workflow/identity
  combinations that carry the ceremony, and the minimal default loads only
  identity-neutral skills.
- Frozen per-conversation policy bundle: selection AND resource content
  freeze at conversation start, a deliberate behavior change — `/reload`
  refreshes the tool list, skills catalog, model catalog, and AGENTS.md
  while keeping the frozen identity/workflow/strategy bodies byte-for-byte,
  and reports pending selection and content changes explicitly (edits apply
  to new conversations only). Resume restores the persisted content, so a
  conversation resumes identically after the underlying files are edited or
  removed. Compaction summaries preserve the bound identity/workflow/strategy
  ids and revisions, the actual delegation precedent (never invented), the
  current phase and approval evidence, resident and pending agents with
  their dependencies, and settled decisions with consumed budgets.
- Canonical task grammar: `model`, `identity`, `purpose`, and `seat` are
  top-level task fields; `model` also accepts the
  `@<preset-or-canonical-model>[:<identity>]` shorthand (`@fast:researcher`,
  `@fast:user:foo`). Missing components are never guessed: dual
  `model`/`config.model` supply, shorthand/structured identity conflicts,
  and a seat without a purpose are validation errors with no precedence
  guess. The default child is `coder` with the bounded-child context and the
  session model; children never inherit the parent's workflow or strategy;
  required-delegation identities (orchestrator) are rejected as children with
  an actionable error; `agent_id` remains the only instance handle and
  nested spawning stays depth-limited to one.
- Purpose routes, seats, and diversity: `[orchestration.routes.<purpose>]` in
  `config.toml` carries
  `targets = [{ model, identity, tier, seat?, overlay?, replacement_capability? }]`
  and `requires_distinct_models`; sparse entries override shipped routes and
  new keys introduce custom purposes; `research` joins the purpose
  vocabulary. Multi-seat purposes require an explicit seat (`review.deep`:
  the analytical and peer seats bind the read-only reviewer; the execution
  seat runs coder with the shipped `review-execution` overlay, which rides
  the frozen persona instructions and grants no authority). Diversity is
  validated over canonical-model cardinality — the same model on two
  providers counts once, and A/A/B fails a requirement of 3 — and unmet
  requirements are reported instead of faked.
- Per-axis authority enforcement: granted, personal, and delegation tool
  axes with a harness-owned mutating/non-mutating tool classification (bash
  mutating regardless of command text, custom and MCP tools mutating by
  default, no classification-by-name inheritance, reclassification
  invalidates child authority caches). A child's effective tools are the
  granted ceiling intersected with every ancestor's delegation set and the
  child's personal set, enforced below the configuration layer: a researcher
  cannot obtain a writing coder through any path, an orchestrator cannot
  write personally but can delegate writes within the granted ceiling, and
  an explicit root denial still blocks delegated writes.
- Identity denylist and invocation selections: `tools.task.denylist` blocks
  launches by resolved identity (exact names, globs, or `re:` patterns;
  shorthand, route, and retained launches are all checked post-resolution;
  the Settings > Identities inventory is its denylist-only editing surface —
  it can only block). The `--identity`, `--workflow`, and
  `--orchestration-strategy` CLI flags and the `CHARTREUX_AGENT__*`
  environment variables carry new-conversation selections with CLI > env >
  config > default precedence and omission distinguished from explicit
  selection; `chartreux doctor` resolves the selections offline and reports
  retired settings with their successors.

### Changed

- The saved main-assistant preset is `@main` (was `@orchestrator`):
  selecting `@orchestrator` fails with rename guidance and no alias, a
  `[roles.orchestrator]` overlay over the shipped base fails loudly at load
  (rename the table to `[roles.main]`; the catalog file is never rewritten),
  and `active_model` in `config.toml` is rejected with guidance to edit
  `[roles.main]`.
- The task tool's default allowlist is `["coder"]` and its denylist keys on
  the resolved launch identity; agent records, names, and browser surfaces
  present identity-keyed fields. The legacy `profile` record field, task
  `agent_type` argument bridge, and `AgentProfileMismatchError` alias are
  removed; the error is now identity-mismatch-specific.
- The `agents/*.toml` agent-profile directory convention is retired; those
  files are detected and reported, but are never loaded.
- `[subagents]` retention is the global default plus the per-launch
  `retain` argument only — profile-level retain and idle-TTL settings are
  gone.

### Removed

- Breaking (pre-release alpha; loud, actionable errors; no silent
  migration): `system_prompt_id` no longer selects the composed prompt (the
  `[agent]` selections do; a custom identity plus workflow/strategy
  composition replaces full-skeleton prompt replacement), and together with
  `agent_paths`, `enabled_agents`, and `disabled_agents` it is retired from
  the settings catalog and flagged by `chartreux doctor` with successor
  guidance — block launchable identities with `tools.task.denylist` instead.
  Project-level prompt directories no longer supply identities; custom
  resources are user-level only. The `[dispatch]` catalog overlay is
  superseded: it no longer appears in the catalog template or the presets
  editor, orchestration is selected through identity/workflow/strategy and
  configured through `[orchestration.routes.*]`, and a legacy `[dispatch]`
  table's values are preserved by catalog writes (comments and formatting are
  not preserved), never rewritten. Sessions
  persisted before v0.5.7 — including any persisted with
  `active_model = "@orchestrator"` — fail resume with a diagnostic that says
  to start a new session.

## 0.5.6 (2026-10-17)

### Added

- Bounded automatic root-turn recovery (recovery program R2/OD3): after a
  terminal root model-call failure — a connection error, timeout, server
  error, overload, rate limit, launch failure, interrupted stream, or empty
  response — the harness schedules the continuation turn itself, with no
  manual `/retry` or new message required. Two collaborators implement it:
  the TUI's `RootRecoveryController` owns an in-process timer and local
  cancellation via an input-generation token (invalidated synchronously on
  typing, submission, question actions, manual retry, interrupt, or session
  change), and the app-server gate (`recovery/root/schedule`,
  `recovery/root/admit`, `recovery/root/cancel`) owns authoritative
  eligibility, episode accounting, and single-flight admission. The
  continuation is due at `max(configured backoff, deployment cooldown,
  endpoint cooldown, Retry-After)` plus bounded jitter, re-reads live
  availability at admission, and never resets a cooldown — an automatic
  continuation is an injected turn, never a user request, and never switches
  base model. One shared automatic-turn budget (default 2 per user-turn
  episode, merging the interrupted-stream continuation) and one monotonic
  advisory deadline (default 2400s, never an in-flight kill) bound the
  episode; a manual retry starts a new user-directed episode and never
  replenishes it. Eligibility is restricted to the typed transient
  allowlist; preflight, permission/authorization denial, budget stop,
  cancellation, task blocker, permanent config/auth/billing/quota,
  context-size, and unknown-internal failures never schedule. Each milestone
  (scheduled, started, recovered, exhausted, cancelled) is a notice line in
  the transcript with a deterministic id — readable after an overnight
  failure and across a reconnect, never injected as a fresh instruction.
  Policy lives in the new `[recovery.root]` config table; exhaustion parks to
  the existing manual retry affordance (button, `Ctrl+R`, `/retry`).
- Typed failure envelope (recovery program WP1): every terminal model-call
  failure now carries a typed envelope — a failure class drawn from an
  authoritative transient allowlist (connection, timeout, server error,
  overload, rate limit, launch failure, interrupted stream, empty response)
  with the protected classes (cancellation, auth, billing, quota, invalid
  request, context size) explicitly ineligible — plus one deduplicatable
  evidence event per logical call, bounded and redacted public messages, and
  typed failure data on task results and agent summaries. Recovery
  eligibility, suspension, and resume/reconcile all consume this envelope;
  old results without it stay immutable.
- Warm subagent suspension and explicit same-agent resume (WP2/WP3): after a
  recoverable failure a subagent can be suspended warm — process-local,
  never persisted, bounded by a TTL (default 1800s), a count cap (8), and a
  byte cap (64 MiB) with oldest-first expiry — and resumed explicitly through
  the new `agents/resume` route with typed dispositions (scheduled,
  already_scheduled, already_resumed, reconcile_required, expired,
  not_resumable, forbidden). `agents/reconcile` settles a pending tool-outcome
  barrier (cleared, prohibited, stale, duplicate, not_pending, forbidden);
  stale or duplicate evidence can never clear a newer barrier. Suspension is
  not retention and not resurrection: a parent restart forfeits suspended
  agents (they close, non-resumable), and a resumed run gets a new run id
  with the old failed results left immutable. There are no exactly-once
  guarantees.
- Bounded automatic subagent recovery (WP4): the new `[recovery.subagents]`
  controller schedules automatic recovery for eligible child-subagent
  failures — jittered backoff over the configured cooldowns, an advisory
  episode deadline (never an in-flight kill), single-flight admission, and
  one bounded startup retry that converts the launch-path cooldown block into
  a cancellable wait instead of an instant terminal error.
- Failure-time replacement policy and the unattended-resilience rule (WP5):
  when a package fails, the orchestrator selects a replacement at failure
  time from validated eligible sets — endpoint independence is enforced
  against the attempted failed endpoint (same-endpoint aliases and
  same-endpoint different-model substitutions are rejected; the worker route
  is rejected for substantive packages), there is no dispatch-time
  substitution table and no automatic cross-model fallback — and every
  substitution emits a milestone notice with an honest cost implication, a
  new agent id with predecessor/package lineage, and a per-package
  replacement budget. Infrastructure and harness failures never produce an
  `ask_user_question`: the failed package is parked and independent work
  continues, ending at worst in a consolidated resumable blocker report;
  authorization, scope, safety, and destructive-action decisions remain
  genuine user questions.
- Unattended question expiry (U): with `[user_questions]`
  `unattended_timeout_enabled`, a non-protected question left unanswered past
  its deadline expires server-side — exactly one typed unanswered disposition
  and one deferred record, drafts preserved locally and never
  auto-delivered, and a detached client sees the expiry on reconnect.
  Runtime approvals (dangerous-command grants, root-access) never arm a
  timer and never expire.
- Manual root retry affordance (R1): after a terminal root-turn failure the
  error widget offers "Retry — new turn" (button, `Ctrl+R`, or `/retry`),
  with an honest exhaustion hint that states both the
  deployments-were-unavailable fact and the cooldown-reset consequence. It is
  a manual fallback, not a watchdog.
- Worker prompt and tool-description rules (track P): the worker brief now
  states the literal-shell rule explicitly (no variable expansion, command
  substitution, or loops; literal commands; uppercase env assignments only;
  no leading `=` words; explicit interpreter, no implicit `uv`), and the
  `bash_read` tool description documents its line limit.
- Transcript message-order integrity validation (PI1/T): a shared
  validator/repair now guards the session transcript's message order —
  duplicate tool-result detection scoped per originating assistant call,
  provider-safe order admission verified after the request-build reorder,
  deterministic repair of the unambiguous steering/backfill duplicate
  pattern, and a typed LOCAL `transcript_integrity` diagnostic (never
  transient, never retried, never scheduled; projected as an internal error
  carrying `category=transcript_integrity`). The append/save boundary
  refuses to persist a defect, request build rejects an unrepairable order
  before any provider call, and the root recovery gate's schedule/admission
  pre-check parks a poisoned transcript to the manual fallback with a repair
  offer.

### Changed

- Default promotion (OD4b/OD5): automatic recovery is now enabled by
  default in all three places — `[recovery.root]` `enabled`,
  `[recovery.subagents]` `enabled`, and `[user_questions]`
  `unattended_timeout_enabled`. Set `enabled = false` (or
  `unattended_timeout_enabled = false`) for the previous manual-only /
  always-wait behavior; disabling cancels future scheduling without
  discarding evidence.
- Cooldown behavior (F6): injected and automatic turns no longer clear
  shared availability cooldowns — only a manual `/retry` (TUI or ACP)
  resets them.
- Breaking (OD4a removal-by-default): the legacy unconditional
  interrupted-stream continuation no longer exists as a hidden second
  scheduler. Automatic root continuation — including the interrupted-stream
  path — runs only through the bounded `[recovery.root]` controller, and
  when `enabled` is false NO root automatic continuation runs at all. The
  flag shipped disabled during development; it is enabled by default in this
  release. The manual retry affordance remains the fallback in disabled
  mode.

### Fixed

- Shell-command policy: a quoted grep PATTERN operand containing glob
  metacharacters (`*`, `?`, or a bracket class, including alternations
  containing `.*`) was falsely rejected as a path-glob candidate. The policy
  now treats the first positional as the pattern when no `-e`/`--regexp`/`-f`
  supplies the regex; all nineteen pinned protections (genuine file-operand
  globs, `-f`/`--exclude-from` values, unscoped `ls`/`cat` globs, `find`
  start points, and the rest) remain denied. Accepted limitation: an
  unquoted glob in pattern position is regex data under the same rule and
  is no longer path-inspected.
- Managed-job output redaction now covers shell-hex-escaped secrets (`\xHH`
  variants, both hex cases, multi-byte byte-wise): the escaped spelling is a
  known secret value, so job output summaries redact it at capture and at
  page time. This closes a real redaction leak intermittently exposed by
  the background-jobs sanitization test.
- Session transcript message-order poisoning (PI1): mid-turn steering could
  persist a user message before the in-flight tool call's cancellation
  settled, and the missing-response backfill — blind to the genuine result
  behind the steered message — inserted a DUPLICATE synthetic
  `TOOL_NO_RESPONSE` placeholder, producing a permanent provider 400
  (`invalid_request_message_order`) that survived restarts and could not be
  reached by manual rewind. The backfill's lookahead is now bounded to the
  logical block (no duplicate is ever created), the recorder anchors the
  settled result after its owning assistant message (before the steered
  user message, replacing a placeholder in place), the save boundary never
  persists a duplicate, and loading a session validates and repairs the
  transcript in memory — healing a poisoned file without manual rewind,
  with the durable migration preserving the original as a recovery artifact
  before replacement.

## 0.5.5 (2026-10-17)

### Changed

- Maintainability program: a twenty-stage decomposition of the largest
  source and test modules, with public names, behavior, and the
  collected-test ledger reconciled at every stage gate. Removed tests were
  strengthened or survival-mapped; the ledger grew from 20,211 to 23,551
  collected, the growth being added boundary, corpus, and identity
  coverage.
- Source decompositions: the app-server `protocol` module (2,342 lines)
  became a thirteen-submodule package behind an identity re-export layer;
  the shell command-policy facade shed its argv corpus (5,366 -> 4,206
  lines) and its git-command family (4,206 -> 1,714); `app.py` shed startup
  support leaves and five controllers - transcript_presenter,
  bottom_panel, agent_browser, session_flow, turn_supervisor (7,316 ->
  4,894); `settings.py` shed four controllers (2,638 -> 1,569); the agent
  loop shed a leaf-types module and nine collaborators (5,462 -> 3,033);
  the sessions registry shed records/authority/transcript extraction and
  eight area modules (5,146 -> 2,091); the provider workbench screen became
  a composition root with ten satellite modules (5,793 -> 1,420).
- Test-tree work: mechanical-assertion cleanup (735 mechanical assertions
  reduced to 23 genuinely weak, 25 strengthened, three dead-contract sites
  disposed); twin-fixture and e2e consolidation behind shared harnesses with
  `tests/e2e/SURVIVAL_MAP.md`; a grant-lifecycle harness unifying the
  dynamic-grant and dangerous-command-lifecycle suites; ten oversized test
  modules split per concern into concern modules with support seams,
  node-count-neutral; and the test-tree reunification below.
- Test-tree reunification moved test modules, changing the pytest node IDs
  that downstream consumers (CI selectors, scripts, issue references) may
  cite: `tests/core/agent_loop/` moved to `tests/agent_loop/` (18 modules,
  347 collected nodes); the former `tests/session/` population nested under
  `tests/session/persistence/` (15 modules, 245 nodes) while
  `tests/core/session/` moved to `tests/session/` (12 modules, 119 nodes;
  the merged root collects 364); and
  `tests/agent_loop/test_dangerous_command_lifecycle.py` (45 defs, 56 nodes)
  was split per concern into `test_dangerous_command_lifecycle_grants.py`
  (41 nodes), `test_dangerous_command_lifecycle_prompts.py` (6 nodes),
  `test_dangerous_command_lifecycle_reprompt.py` (4 nodes), and
  `test_dangerous_command_lifecycle_session_boundaries.py` (5 nodes), with
  the shared helpers in `_lifecycle_test_support.py`; the original module is
  retired. Test names, counts, and behavior are unchanged - only the
  node-ID paths moved.

## 0.5.4 (2026-10-12)

### Added

- Worker report disclosure across every output path: the denial record
  (each denied command and the runtime's stated reason, or none) and the
  complete cumulative touched-path set (including later-reverted edits) are
  required in every report format layer - task-specified, skill-specified,
  the JSON fallback, and blocker reports. The requirement is carried by the
  worker prompt, rule 5, the dispatch scaffolding in both dispatch modes,
  the task tool description, and the task-brief composition seam (with and
  without a scratchpad, on launch and reuse).
- Read continuation contract: an unbounded read that would exceed the
  output cap returns the first chunk with its line range, total lines, a
  traversal id, the content revision, and the next offset to continue -
  a big file is one intentional traversal instead of an error-and-window
  habit. Mid-traversal revision changes restart the traversal with a
  notice; explicitly bounded offset/limit requests are honored exactly and
  never converted; a 32 KiB chunk target leaves headroom under the cap.
- Content revision contract (`chartreux.core.file_revision`): a
  blake2b-128 revision computed by the tool's own read path identifies a
  file state; client-mediated reads and writes that the host cannot attest
  publish no revision (a documented degradation), and traversal state is
  carried by a bounded LRU registry.
- Grep default context: matches return +/- 2 context lines with merged
  overlaps and per-file grouping; `match_count` and parsed matches count
  actual matches only, never context or separator lines; the context
  expansion applies identically to both grep backends.
- Post-edit evidence in `EditResult`: the applied diff (per-site, capped),
  a post-edit context window with CURRENT line numbers, and the post-edit
  file revision - computed after the applied replacement, serialized to
  the model; the UI-only occurrence projection is unchanged.
- Workspace state: a stdlib-AST outline generator (symbol -> current line
  ranges, size-capped, degrading to revision-only tracking), a bounded
  working-set store, a tree-shared dispatcher revision registry, and
  per-revision state blocks - one injected message per working-set
  revision change, appended at LLM-turn start, immutable, trimmed only at
  compaction boundaries. Children receive a spawn handoff with named
  working-set files and current outlines; external bumps by sibling
  workers are reported; the compaction boundary serves the bounded
  working-set summary back to the model and replays state structurally.
- Shell policy tier 2: statically modeled, never-executed whitelisted shell
  idioms - the sed locate-then-print idiom (recognized only when the
  `cut -d: -f1` numeric-extraction stage guarantees a literal numeric
  address; the fully expanded script is authorized under the normal sed
  policy), zsh `=command` equals expansion (resolved against PATH without
  executing; the resolved command is authorized under its normal policy),
  and variable/arithmetic expansion in provably read-only argument
  positions with literal bounds (statement-level bindings only; prefix and
  guarded assignments never bind). Unrecognized-but-recurring shapes are
  drift-logged for tier-3 candidates.
- Typed denial classes: every shell denial carries a SYNTAX or POLICY
  class decided where the denial is created, with fail-closed aggregation
  (mixed or unknown -> POLICY). SYNTAX denials name the compliant
  alternative; POLICY denials state the blocked requirement and suggest
  nothing (the recursive-rm alternative suggestion is removed); the whole
  call's class is decided before rendering so a mixed syntax-plus-policy
  denial never suggests alternatives.

### Fixed

- Dangerous-command once-permit leak: an allow-once permit installed before
  the post-approval recheck is now destroyed on every exit path (resolver
  exception, cancellation, failed recheck), so no authorization outlives a
  failed recheck.
- Dangerous-command revocation window: permission is revalidated after the
  file-snapshot await and immediately before execution (covering both
  local execution and client-terminal delegation), so a grant revoked
  during that window blocks the call; the once permit stays
  invocation-bound and synchronous.
- Static-expansion authorization bypasses found by the deep review: zsh
  `:s`/`:gs`/`:h`-family modifier and `[...]` subscript suffixes after a
  bare `$NAME` no longer resolve (they re-interpret the operand at
  runtime); variable bindings are invalidated by any unmodeled
  reassignment (`export`/`typeset`/`readonly`/`declare`/`local`, guarded
  or non-literal forms); whitespace and empty-string values never resolve;
  the sed simulation stats its source (regular files only, 8 MiB cap)
  before reading and the wiring pre-checks the source against the
  workspace/scratchpad roots, so analysis never blocks on FIFOs or reads
  out-of-workspace or oversized files; NUL-containing (binary) sources are
  refused.
- Workspace-state correctness from the deep review: non-vouching
  observations (windowed and port-path reads) never erase a previously
  vouched revision, so unchanged files no longer produce phantom
  revision-change blocks and external-bump detection stays armed;
  registry publication is ordered per path (superseded deliveries are
  rejected instead of regressing tree truth); client-mediated writes
  publish no fabricated revision.

### Changed

- The worker efficiency rule (Operational discipline rule 7) is deleted
  entirely, including its batching clause; rules now run 1-6 with no
  dangling references.
- The recursive-rm denial message states the hard guard without suggesting
  an alternative deletion method; the approval label is class-aware
  (`unsupported shell syntax:` only for SYNTAX aggregates).


### Added

- Ephemeral-by-default subagents: unretained background agents close their
  runtime on completion and keep only a bounded terminal entry; their
  results stay retrievable until result expiry. Retention is opt-in through
  a per-launch `retain` argument, profile-level `retain`, or
  `[subagents].retain_agents` (explicit argument, then profile, then the
  generation-snapshotted default); reuse inherits the committed policy, an
  explicit `retain=false` admits one final continuation, and the shipped
  advisor profile is retained by default with a 3600s idle TTL.
- Bounded terminal metadata: tombstones are unified for closed (ephemeral
  completion, including failed and cancelled runs) and evicted (retained
  agents removed by idle TTL or cap) agents, expire after
  `tombstone_ttl_seconds` (900s; 0 stores none) plus an oldest-first
  `max_tombstones` cap (32) enforced at insertion, and are pruned by the
  extended single reaper with lazy deadline checks at every listing and
  transcript authorization surface. Metadata expiry never expires results;
  release purges them. `/clear` removes old-generation metadata and
  publishes the reduced inventory even with no live records.
- Agent display names: every surface (browser rows, transcript header,
  completion and eviction notifications, launch acknowledgment,
  `check_agents`) shows a server-derived `{profile}-{model}-{number}` name
  built from the canonical base model; names are display-only,
  handle-shaped derivations are sanitized, and `agent_id` remains the only
  handle.
- Typed `wait_for_agent` results: waits return a `completed` or
  `still_running` disposition instead of raising on timeout; the default
  timeout is 1500s, explicit `null` still means indefinite, timeouts never
  cancel the run, and `still_running` guidance instructs the model to end
  its turn (queued user input drains between turns) and re-invoke the wait
  in a later turn.
- `agents/release` RPC and a browser release action: idle and recent agents
  can be released from the `/agents` browser through a confirmed,
  race-guarded route that verifies attached-root authority, expected run
  identity, and non-active state atomically under the registry lock;
  conflict and unknown outcomes never stop or purge anything.

### Changed

- The `/agents` browser shows Active / Retained Idle / Recent groups with
  Resident / Active / Idle / Recent / All filter tabs and truthful counts,
  consumes server-side deterministic ordering, renders names with dimmed
  handles, and drives keyboard, mouse, scrolling, and highlighting through
  one shared visible-row mapping; the collapsed bar counts active work only
  (with dim recent or idle notes at zero active).
- Subagent defaults are tighter: `idle_ttl_seconds` 3600->300 and
  `max_idle_agents` 16->4; the previous both-zero close-on-completion switch
  now means retained with no limits (`max_running_subagents` is unchanged
  and stays excluded from the settings catalog).
- Task guidance is background-first: background launches are recommended
  even when the result is a dependency (launch, then `wait_for_agent`),
  keeping agents inspectable in the browser and amendable mid-flight;
  foreground is reserved for calls that must block.
- Prompts, dispatch renderer text, dispatch goldens, the builtin skill, the
  settings catalog, and the docs set (configuration reference, subagents
  guide, app-server protocol reference, commands reference) are synchronized
  to the new lifecycle.

## 0.5.2 (2026-10-11)

### Added

- Dangerous-command runtime approvals for git operations: a typed classifier
  resolves dangerous git commands with resolver-side authority binding and
  configured-remote verification; approvals grant per operation and repository
  scope with Allow This Session / Once / Always choices. Only `git.push`
  persists, through a protected user-layer map; app-server endpoints expose
  read and revoke (no public grant), the TUI question renders safely with a
  redaction notice, and a Shell Permissions panel lists grants. Grants survive
  compaction and reload, are cleared by in-place session resets, and are never
  restored on resume (ADR 0019).
- `.git` metadata hardening: every write/mutate path operand is resolved
  against tracked working-directory sets (including git `-C` relocations) and
  denied on metadata containment, fail closed; hard links are contained at
  creation and write time via inode comparison; program-selection config keys
  (`core.hooksPath`, `core.editor`, `sequence.editor`, `filter.*` clean and
  process, `merge.*.driver`, `diff.*` drivers, `core.fsmonitor`, `man.*.path`,
  `interactive.diffFilter`) are hard-guarded in every write scope including
  section renames; `git apply --unsafe-paths` is a hard guard; work-tree
  relocations into metadata are denied for tree-writing subcommands; write
  operands of archive, extraction, binary, and compression tools are inspected
  in attached and separate option forms. The enumeration stop rule and the
  accepted residual boundary are recorded in ADR 0019.
- Shell redirect analysis v0.2 (ADR 0021): the order-blind cd+redirect veto is
  replaced by occurrence-local redirect modeling on the existing resolver
  cd-chain tracker; fd duplication and device sinks are structurally exempt;
  compound chains (semicolons, pipelines, nested sources) resolve; absolute
  redirect targets inside the session scratchpad are authorized at the
  resolver layer after the metadata check, with component-wise root matching.

### Changed

- Subagent operational discipline is rewritten for ML4 (ADR 0020): the
  never-firing denial stop rule is replaced by a strategy-switch rule;
  permission-refusal language is hardened, including the
  same-effect-other-tool retry pattern; an efficiency rule is added; the
  destinations rule takes precedence over assignment text; reviewer and
  advisor report formats gain a Denials encountered slot. The failure-retries
  rule is dropped and recorded as an accepted risk with validation baselines.
- Denylist break: an empty `bash` denylist no longer implicitly authorizes
  dangerous commands; entries must be removed explicitly to permit them.
- The bash tool prompt documents scratchpad-absolute redirect targets.

### Security

- Push integrity: granted pushes are verified against configured remotes;
  push destinations, remote redefinition, and redirecting configuration
  writes (remote URLs, `url.*.insteadOf` rewrites, credential and include
  sections, `core.sshCommand` and exec-program keys) are hard-guarded so an
  approval cannot land at an attacker-chosen repository.
- Closes pre-existing redirect-target bypasses: concatenated literal targets
  (`> "sub"/../.git/config`, `> "sub"/../../outside.log`) were never
  path-checked before and now are, and `source`/`.` operands are
  authority-checked.

## 0.5.1 (2026-10-10)

### Added

- The shipped catalog now includes `mistral-large-4` on the Mistral provider
  (runtime-confirmed wire ID `mistral-large-4`, thinking `high`, image support,
  400000-token compaction threshold) at launch-sale prices below the published
  list prices (input 0.68, output 2.09, cached input 0.07 per million tokens).
- Completing onboarding without saving a catalog now writes
  `~/.chartreux/models.toml` from a shipped template: both default models
  (`glm-5-3` and `mistral-large-4`) with their deployments, the four role
  presets, and sparse `[dispatch]` slot role bindings, with no provider tables
  (they stay inherited from the shipped catalog, so `overlaid_providers`
  provenance is unchanged). The template is identified by a content
  fingerprint rather than a comment marker, materializing is revision-neutral,
  and publication is an exclusive create — an existing `models.toml`, whether
  migrated, saved, or created concurrently, always wins.

### Changed

- The shipped role roster is now `orchestrator`, `worker`, `scout`, and `heavy`:
  `orchestrator` (glm-5-3, high) is the main assistant, `worker` (glm-5-3,
  medium) covers implementation and quick/standard review, `scout` (glm-5-3,
  low) covers mechanical work, and `heavy` (mistral-large-4, high) covers the
  advisor, analytical and peer review, and escalation-implementor seats. The
  tier roles `large`, `medium`, and `small` are removed; dispatch slot
  bindings, the rendered routing prose, and the built-in profile defaults
  (`worker` -> `@worker`, `advisor` -> `@heavy`, `reviewer` -> `@worker`)
  follow the new roster. Deep review renders as ML4 analytical + ML4 peer + GLM
  execution seats, with no three-model diversity claim.
- Removed the dead `show_greeting` setting (the greeting never displayed);
  existing config files that still contain the removed TOML key load cleanly,
  with the key ignored. In contrast, `CHARTREUX_SHOW_GREETING` is rejected at
  environment validation, breaking for anyone who set that variable; remove it.
- Session resume break: pre-upgrade sessions with a saved dispatch policy
  referencing the removed `large`/`medium`/`small` roles fail resume with
  `SessionPolicyError` ("Bound dispatch role removed"). No substitution is
  performed; committed model identities are role-free and still resolve.
- Onboarding seeds role presets from the selected model: when first-run setup
  runs with a selected model, replacement presets for unready roles are drawn
  from that model's ready (model, thinking) pairs instead of any configured
  model.
- The non-interactive missing-credential error now includes repair guidance:
  it names the required environment variable and points at `chartreux --setup`
  or Settings > Providers (/providers), and notes that credentials do not
  change provider routing.

### Fixed

- The graduation nudge ("More models can share the work") no longer latches for
  fresh installs. The latch is now a bound-roster transition: it sets only when
  the roster the runtime would select moves from exactly one distinct canonical
  model to two or more across a save, with the saved model part of the new
  roster. Entering a provider key that makes both shipped default models usable
  at once, and saves against an already multi-model roster, no longer trigger
  the nudge; latches persisted by earlier releases remain honored.
- Workbench readiness now validates the deployment the runtime would select:
  preset validation, finish checks, and the graduation roster resolve the
  staged candidate catalog through the model resolver — deployment ordering,
  thinking-level compatibility, and `allowed_models` restrictions — and gate on
  the selected deployment's credential instead of any deployment's.
- Saving an unchanged shipped Mistral preset no longer force-writes a
  `[providers.mistral]` table into `models.toml`: an unchanged preset over an
  existing shipped provider persists nothing, so `overlaid_providers`
  provenance stays clean and first-run materialization of the default template
  is not blocked.

## 0.5.0 (2026-10-07)

### Added

- Rendered, session-bound dispatch policies with named launch slots, task purposes,
  configurable model-roster rendering, and linted dispatch overlays. `standalone`
  is the default, with direct implementation and delegated verification and review;
  `orchestrated` remains selectable. A dismissible graduation nudge helps
  multi-model users consider the transition.
- Compaction handoffs preserve dispatch policy identity, contributor authorship,
  consumed attempt budgets, and recovery routes.
- The default `bash` denylist blocks destructive Git forms including `git push`,
  `git checkout`, `git stash drop`, `git stash clear`, `git restore`, forced
  `git switch`, and `git reflog` deletion or expiration. Denylist entries are
  user-removable through `[tools.bash]`; hard guards remain non-overridable.
  Git global-option spellings are normalized, and `bash_start` applies the same
  policy without creating a job when launch is denied.

### Changed

- Subagent dispatch is routed by task rather than difficulty. `@small` handles
  search, exploration, verification, and mechanical edits; `@medium` handles
  substantive implementation; `@large` handles architecture, design, planning,
  and deep review. Single-model rosters use task-kind routing. The reviewer stays
  `@medium`; deep review uses a `@large` model override.
- The system prompt requires approval before mutating or authoritative work and
  preserves approval evidence and workflow state through compaction. The worker
  persona is skill-first, and design and planning require explicit user acceptance.
- `standalone` is the shipped dispatch default; onboarding is collapsed, with a
  graduation nudge for multi-model users. `orchestrated` remains available.
- The default model temperature is now `1.0`; the shipped GLM-5.3 temperature pin
  was removed.

### Fixed

### Security

- Runtime denylist denials cannot be lifted through conversational approval.
  Agent guidance and the Git workflow skill reflect the policy.

## 0.4.3 (2026-10-05)

### Added

- Root-owned managed shell jobs through `bash_start`, `bash_read`, `bash_stop`,
  and `bash_list`, with bounded/redacted merged output, replayable sequence
  cursors, gap reporting, shared root/child capacity, and creator-scoped child
  access. Jobs survive turns and child completion, use canonical Bash launch
  policy, and stop through TERM, a grace period, then KILL. Root-session retirement
  cleans up jobs without cross-restart adoption; relocation and authority guards
  prevent retargeting live jobs or signaling unowned processes.
- An opt-in **Background jobs** status-line segment showing `Jobs N` (including
  `Jobs 0`), backed by projected active-job state and between-turn updates without
  output-body publication.

### Changed

- Settings separates catalog, entries, choices, membership, and actions into
  composite-browser groups; the status-line editor separates configuration and
  actions. Both use bounded arrows, priority Tab/Shift+Tab cycling, identity
  bookmarks with nearest-surviving-neighbor repair, and a single-owner Escape
  chain that restores the opener. Pointer actions follow the same guards and
  outcomes as keyboard actions.
- Enter no longer saves a Settings collection: saving requires explicit Apply.
  Failed Apply and reset preserve drafts; boolean rows retain immediate saving.
- The Providers workbench separates root actions, provider operations, connection
  actions, conditional catalog actions, visible detail actions, and preset and
  preset-editor actions. Groups use bounded navigation, remembered focus, and
  equivalent pointer actions. Back handles busy state before confirmation, then
  help; confirmation overlays support keyboard scrolling. The Settings and
  Providers navigation deferral in `DESIGN.md` section 10 is closed.

### Fixed

- Provider discovery is bound to its requester, with per-request feedback tokens
  preventing stale results from updating another view. Connection actions no
  longer include a duplicate Continue row, and the inactive save-key affordance
  is removed.
- Incremental job-output redaction preserves benign text at normal end-of-output
  and releases completed short words without waiting for process exit, while
  still masking split secrets and unresolved encoded prefixes.

## 0.4.2

### Added

- A customizable bottom status line with ordered directory, PID, model, context,
  cached Git branch, and spend segments. Settings includes a state-cycler editor
  with reordering, a live example preview, batch Apply changes, and saved-override
  removal.
- An independent local usage ledger for conversation, subagent, compaction, title,
  and pre-session worktree-naming calls, with captured USD catalog prices and
  explicit unknown-cost and coverage states. Records survive session deletion
  and logging-disabled sessions; linked worktrees count as one project.
- A read-only `/usage` browser with local calendar Day/Week/Month windows,
  All projects/Current project filters, token breakdowns, deployment details,
  and explicit refresh. Status-line spend segments now show global recorded
  estimates; host-level `usage/read` and `usage/updated` expose the same accounting.
- Persisted message posting times, shown as absolute local times with a date for
  older messages, plus settled per-tool durations and whole-turn totals on the
  last assistant message. Timing displays are static and can be hidden together.
- Run-scoped subagent cancellation through the orchestrator's `cancel_agent` tool
  and retasking through `task(replace_run=True)`, retaining the same conversation
  even at capacity. The agent browser offers C-key stopping with inline
  `[Stop run][Cancel]` confirmation and a Stopping state. Typed stop reasons
  distinguish user cancellation, orchestrator cancellation, and retasking, with
  partial output preserved and no parent auto-wake on user cancellation.
- Subagents can read their injected instruction files, including `AGENTS.md`,
  through `read_file` and exact-file `grep`, even outside workspace roots. This
  grants no directory or write access and preserves tool denials and sensitive-file
  protections.
- Immediate steering when the main agent is only waiting on subagents: submitted
  messages cancel outstanding waits, not the subagents, and reach the model in
  the same turn. Admission is server-enforced, delivery receipts are idempotent,
  and recovery treats uncertain delivery conservatively.

### Changed

- The agent browser is a docked ten-row sheet with a retained Main row, stable
  ordering, single-click output opening, two detail rows, full Details for Main
  and subagents, local F1 help, and a keyboard-action footer.
- Agent states use static markers and distinguish Running, Compacting, Finishing
  with outcome, Idle, Failed, Cancelled, Budget stopped, and Evicted. Details
  show running/last-run and idle durations.
- Context usage displays without a label, for example `135k/400k (34%)`, using
  the effective compaction threshold as the denominator. Deployment thresholds
  must be positive whole integers.
- Message headers contain metadata only, without role labels. User messages use
  a `>` marker and accent tint; tool-group folding includes timing when enabled.
- Chrome uses `+`/`-` disclosure glyphs only on actionable controls and `v` for
  ASCII success markers.
- The Task tool selects profiles through `agent_type` instead of `agent`, with
  separate instance handles and explicit reuse guidance.
- The navigation contract specifies bounded arrows within composite-browser
  groups, Tab between groups, per-control Enter roles, Space toggles, single-owner
  Escape, equivalent mouse outcomes, and identity-based focus repair.
- Exit confirmation defaults to Cancel and requires explicit Exit. MCP has
  bounded navigation, pointer enable/disable controls, and opener restoration;
  Provider Settings adds Tab/Shift+Tab groups and separate Details click targets.
- The log-level picker edits session and config drafts symmetrically with a
  visible Apply action. The debug console supports keyboard row selection and
  scoped Escape; question acceptance clicks follow the same guards as Enter.

### Fixed

- Compaction summary calls no longer overwrite conversation context usage.
- Test failures no longer dump process environments.
- Empty assistant responses receive one bounded replay before typed
  `EmptyLLMResponseError` or `IncompleteLLMResponseError` terminal failures.
  Streaming never replays after publication, and subagent runs without final
  prose report `FAILED` with stop reason `ERROR` rather than empty success.
- Subagents inherit scratchpad access through the parent-authority chain,
  failing closed on broken authority, symlink escapes, or root retargeting.
- Shell policy checks modeled executor boundaries through a shared registry.
  Recursive `rm` remains gated, with diagnostics naming the offending option
  and the file-by-file removal plus `rmdir` alternative.
- Credential redaction covers tool events and child environments; fallback
  logging never emits credential-bearing exception text.

## 0.4.1

### Added

- A `[subagents].max_running_subagents` configuration setting for the active-work
  admission cap, also available through the standard environment override.
- `chartreux doctor` with local bare checks, opt-in `--live` readiness checks,
  billable `--smoke` probes, and `--json` output. Exit codes are 0 for no failed
  checks, 1 for failed checks, and 2 for invalid invocations.
- Provider smoke probes with tool, thinking, and image capability verdicts.
- Retry-budget audit regression tests.

### Changed

- Backends accept an optional pre-resolved credential for isolated diagnostics.

### Fixed

- OpenAI Responses non-streaming truncation is surfaced as `StopInfo`.
- Doctor validates runtime provider configuration, suppresses credential-bearing
  MCP SDK logs, and matches runtime environment-only MCP static authentication.
- Successful trust-store repairs clear stale load-error state.

## 0.4.0 (unreleased)

### Added

- A fail-fast cap on active subagent work, defaulting to 16.
- Linux clipboard image paste through `wl-paste` or `xclip`.
- Checkpoint and rewind coverage documentation describing memory-only snapshots,
  staged per-file restoration, and non-transactional failure handling.
- Docs section indexes and strict link validation, and a strict docs build in
  pre-commit.
- CI runs on `development`.

### Changed

- Tool permissions now use only `always` and `never`; the vestigial `ask` mode
  was removed. Configurations containing `"ask"` fail validation with guidance.
- Model-catalog discovery contracts moved from UI to core, and CLI-to-UI
  compatibility shims were retired.
- Shell-command policy now checks npm, sed, dd and git operands, package URLs
  and extras, git aliases, hooks and redirections, promisor fetches, and
  repository-bound environment and trust boundaries more conservatively.

### Fixed

- Shell cancellation drains descendant processes by killing the process group.
- Burst-completion freezes caused by a bounded app-server dispatch queue; the
  queue is now unbounded.
- Prompt-history persistence is serialized across processes, and per-session
  thinking overrides restore without leaking into other sessions.
- Checkpoint file-store failures are contained per path, with restoration
  permissions preserved and writes staged before replacing targets.
- Transcript generation fencing prevents stale queued saves from overwriting
  the retained transcript after rewind, reset, or session rebind.
- Tests use load-tolerant completion bounds and global logging-state isolation;
  history-manager child spawns tolerate import delays.

## 0.3.0 (2026-09-30)

### Added

- A `/settings` UI for all `config.toml` settings, with inline editing, bounded
  inputs, checkboxes and dropdowns, and a single screen to enable or disable
  tools, skills and agents. Changes are saved to `config.toml`.
- A delegation protocol and configured model catalog in the system prompt, so
  the orchestrator can route work to subagents using available roles and models
  rather than guessing model names or doing everything itself.
- Guided web-search configuration and model presets for simpler setup.

### Changed

- Provider Settings is now the single provider configuration surface, shared
  with streamlined onboarding.
- Applied a consistent design language across the TUI, with clearer controls,
  keyboard and mouse interactions, and fewer navigation traps.
- The advisor profile can now load skills, and task prompt examples use shipped
  roles. Setup guidance and the prompt list now match available capabilities.

### Fixed

- TUI freezes under heavy subagent usage, including agent-bar updates and
  switching between agent transcripts.
- Credential names accepted by a session's model catalog remain scrubbed from
  subprocess environments even after catalog reloads.
- Shell timeouts, interrupts and cancellation terminate descendant process
  groups with bounded waits, including cancellation immediately after spawning;
  detached background jobs still survive successful commands.
- Interrupted Anthropic thinking no longer leaves an invalid empty assistant
  message in the next request.
- A full disk while creating a session directory now takes the fail-soft path
  instead of crashing the agent loop.
- Setup no longer imports the CLI, and settings tool-filter help and inventory
  views now match runtime behavior.

## 0.2.0 (2026-09-25)

Initial pre-release of Chartreux. This is the first Chartreux release, forked from
[Mistral Vibe](https://github.com/mistralai/mistral-vibe) v2.25.5; the changes
below describe Chartreux-specific differences rather than inherited capabilities.

### Added

- A provider-agnostic `models.toml` catalog with linked deployments, priority
  failover, model tags and roles, and per-model thinking-level collapse.
- Background subagents with non-blocking execution, fan-out, completion
  notifications, retention, reuse and retasking with per-run model, thinking,
  and permission overrides, plus transcript browsing and an agent sidebar in
  the TUI.
- Provider-configurable `web_search` and image-capable `read_image` tools.
- Fan-out launches runnable role members and reports unavailable or forbidden
  members as skipped.
- A per-server circuit breaker for MCP tools: repeated request timeouts put a
  server in a short cooldown during which calls fail fast instead of burning
  the full tool timeout, and a single-flight recovery probe re-admits it on
  the next successful response.
- A distinct `SessionDiskFullError` when session persistence hits ENOSPC: the
  session stays authoritative in memory and the next save retries, instead of
  a full disk crashing the agent loop.
- A dedicated `ALL_DEPLOYMENTS_UNAVAILABLE` error code with per-deployment
  exclusion reasons when every deployment of the committed model is
  unavailable, surfaced to ACP clients as application code `-31009` instead of
  a generic internal error.

### Changed

- Path allowlist globs in tool permissions are now segment-aware, matching
  upstream: `*` in an absolute pattern no longer crosses separators, so
  `/home/u/proj/*` authorizes only direct children of `proj` instead of its
  entire subtree. Relative patterns still require the whole path to match, and
  denylist patterns are unchanged (a denylist `*` keeps covering a whole
  subtree, keeping denials fail-safe). Allowlist entries that relied on `*`
  matching across separators silently narrow to a single level, and absolute
  `**` patterns narrow the same way; no absolute pattern shape grants recursive
  subtree access anymore.
- Updated the shipped catalog to be neutral and publicly reachable only: it now
  defines the Mistral public provider and models Mistral actually serves (such
  as glm-5-3), while personal setups belong in the user-level `models.toml`
  overlay; raised glm-5-3's default thinking to high and removed
  the former medium worker and reviewer roles.
- Rebound the built-in reviewer profile to `small-reviewer` and reworked the
  main-review tiers around a blocking parallel Deep review fan-out.
- An independent, local-first harness: no sign-in/sign-out or provider accounts;
  provider API keys are supplied through `.env`, onboarding writes a default
  `config.toml`, and configuration is file-based rather than a settings UI.
- A simplified provider layer with generic OpenAI-style and Anthropic-style APIs,
  alongside Mistral and Codex/Responses adapters; Vertex and the reasoning adapter
  were removed.
- A Chartreux-inspired light/dark TUI palette.
- Tool permissions and the shared app-server boundary for the Textual CLI, ACP,
  and programmatic clients were reworked; Chartreux sends no product telemetry.
- MCP stdio servers now receive the MCP SDK default environment merged with the
  per-server `env` table; previously a non-empty `env` replaced the inherited
  default environment wholesale. This is a behavior change: servers that relied
  on replacing the default environment now see it merged underneath their own
  values.
- The automatic project context no longer runs `git status`, whose working-tree
  scan can push file contents through repository-configured clean and process
  filters. The `gitStatus` prompt field is renamed `gitContext` and now carries
  only the branch and recent commit metadata.
- Listing worktrees no longer opens a repository object and spawns a
  `git rev-parse` subprocess per worktree; listings keep their filesystem
  checks but trust the repository's own `git worktree list` records, so
  project resolution no longer costs ~29ms per worktree on every listing.

### Fixed

- MCP OAuth clients whose registration response issues a `client_secret` while
  omitting the auth method (e.g. Supabase) now authenticate with it, picking
  `client_secret_basic` or `client_secret_post` from what the server advertises.
- An abandoned MCP OAuth login no longer pins the loopback port until process
  exit; the callback wait is bounded and times out with the port released.
- A `config.toml` layer using `[mcp_servers.<name>]` now surfaces the
  `Use [[mcp_servers]] instead of [mcp_servers.<name>]` guidance when the
  layer is validated, instead of a generic field-type error.
- A session lease that cannot publish its diagnostic rolls back its lock file
  so the session can be retried, and lease release no longer masks an unlink
  failure behind the release.
- Root sessions that resume with an unusable committed model now resume with a
  model-choice gate instead of hard-failing; child and blueprint launches fail
  fast with actionable errors.
- Batched tool responses normally append in tool-call order even when tools
  complete out of order; if a call aborts before producing a response, its
  backfilled response can follow later calls' responses. Tool output stream
  events are buffered until each tool completes, then sanitized as a whole
  before reaching the UI; output is not displayed live while a tool is running.
- Retiring an MCP stdio pool drains already admitted requests for at most about
  five seconds before cancelling still-running workers.

### Security

- The default `bash` denylist now includes the network clients `curl`, `wget`,
  `nc`, `ncat`, and `socat`, and inline interpreter code via `python -c`,
  `python3 -c`, `pypy -c`, `pypy3 -c`, `node -e`, `perl -e`, and `ruby -e`.
- Chartreux credential environment variables are scrubbed from the environments
  of child processes it spawns (shell commands, MCP stdio servers, hooks, and
  client terminals). The new `credential_env_passthrough` setting lists variable
  names to pass through.
- Known secret values are redacted from tool outputs before they reach the model
  or the session log, including base64-encoded forms.
- `~/.chartreux/.env` is created owner-only (`0600`) and tightened when found
  more permissive.
- Web and MCP tool results are framed as untrusted content, delimited and
  marked as data rather than instructions.
- Git reader commands (`git diff`, `log`, `show`, `status`, `blame`,
  `whatchanged`) are denied in repositories whose configuration activates
  execution vectors (`core.pager`, `pager.*`, `include`/`includeIf`,
  `core.fsmonitor`, `filter.*`, diff drivers, `merge.driver`, `gpg.program`),
  failing closed when the configuration cannot be read.
- The shell policy now covers `less`/`more` startup commands, `checksum
  --check`, `uniq`'s output operand, `date`'s clock-setting operands, `du`/`wc
  --files0-from`, `file --files-from`, and the path-value options of
  `grep`, `diff`, and `tree`; each is denied or requires approval as its
  file-content or command-loading behavior demands.
- `git fetch` resolves the remote through a hardened path that neutralizes
  repository-owned configuration execution vectors (credential helpers, SSH
  commands, proxies, URL rewrites, hooks); only explicit SSH and HTTPS remote
  URLs may be fetched, and fetch configuration that cannot be classified
  fails closed.
- Correction on the Wave 1 description above: its earlier wording about closing
  the prompt-injection-to-exfiltration chain overstates the result. The
  remediation waves add shell-string analysis, per-session scrub policy,
  emission sanitization, credential collection, untrusted-content framing, and
  OAuth/callback hardening. These controls harden against known and naive
  payload shapes; they do not provide complete network-egress containment.
  Accepted residuals include images, arbitrary transforms (including
  encoding/compression or splitting secrets across content), sub-threshold
  secrets, ACP advisory environment scrubbing, user-operated editor/theme/
  clipboard environment inheritance, and filesystem races. Encoded-carrier
  decoding examines at most 128 candidate windows per event; after that limit,
  later encoded candidates may pass unchanged, although direct known-value
  matching still runs over the full text. This redaction layer is best-effort
  transcript control, not complete exfiltration prevention. The shell analyzer
  fails closed on any unrecognized wrapper-shaped prefix before a shell `-c`;
  only `nohup`, `setsid`, `stdbuf`, `timeout`, `nice`, `ionice`, `taskset`,
  `time`, and `flock` are recognized, each with modeled option grammars.
  Command-lookup manipulation (`hash`, `alias`, and `expand_aliases`) in
  analyzed sources also denies the command.
