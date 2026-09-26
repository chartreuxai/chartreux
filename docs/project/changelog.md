# Changelog

All notable changes to Chartreux are documented in this file.

## 0.1.1 (unreleased)

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
