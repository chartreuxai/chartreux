# Changelog

All notable changes to Chartreux are documented in this file.

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

## Unreleased

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
