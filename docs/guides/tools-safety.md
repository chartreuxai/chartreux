# Tools and safety

Chartreux exposes file, search, shell, delegation, web, image, and session-management tools. Runtime policy still applies when a tool is visible.

## Built-in tools

The built-ins are `ask_user_question`, `bash`, `bash_start`, `bash_read`, `bash_stop`, `bash_list`, `cancel_agent`, `check_agents`, `edit`, `get_agent_result`, `grep`, `read_file`, `read_image`, `release_agent`, `skill`, `task`, `todo`, `wait_for_agent`, `web_fetch`, `web_search`, and `write_file`. MCP servers can add tools; their names use `<server>_<tool>`.

`web_search` supports `auto`, `mistral`, `exa`, `brave`, and `duckduckgo`. The runtime default remains `auto`, which selects Mistral only; it does not fall back to another provider. Configure the provider, optional credential environment-variable name, base URL, timeout, result limit, and Mistral search model in `[tools.web_search]`. An unset or blank `base_url` uses the selected provider's default endpoint; the blank value saved by the editor is not treated as a URL override. For a keyed provider, Chartreux selects exactly one credential-variable name: `api_key_env_var` when configured, otherwise the provider default (`EXA_API_KEY` or `BRAVE_SEARCH_API_KEY`); `auto` and `mistral` otherwise use the configured Mistral provider's credential variable or `MISTRAL_API_KEY`. If that selected variable is unavailable, web search reports the missing key. DuckDuckGo needs no key.

An unset or blank Exa or Brave endpoint resolves to that provider's default
absolute endpoint. Transport failures use safe, categorized messages such as
an invalid endpoint URL, DNS or TLS failure, proxy/connect failure, or a
transfer/protocol error; untrusted exception text is not shown.

In standalone Settings, the provider list shows Mistral once. The saved `auto`
value remains a supported Mistral alias for existing configuration, not a
second provider choice. First-run setup finishes after default presets and does
not read or rewrite web-search configuration.

Use Settings > Web search or `/web-search` to edit these values. `Save search settings` saves the settings draft; `Save API key` is a separate credential action, so a saved key remains saved if you later discard the settings draft. The key action reports whether the key was saved or is available only for the current session. Switching providers resets custom credential-variable and endpoint overrides to the new provider defaults and clears an unsaved key. The `Configured; connection not verified` readiness label means the selected configuration and any required key are available; Chartreux does not test live connectivity or switch providers automatically. See the [Configuration guide](configuration.md) and [command reference](../reference/commands.md).

Search-settings feedback reports persistence and runtime application separately. If settings were saved but not applied, use **Retry runtime reload**; if runtime applied but the screen did not refresh, use **Retry UI refresh**. These retries apply already saved settings without repeating the write. If a revision conflict occurs or the saved state could not be read, refresh before editing again; **Refresh (discard draft)** discards unsaved search settings. The editor also identifies fields controlled by a higher-priority configuration layer.

```toml
[tools.web_search]
provider = "brave"
api_key_env_var = "BRAVE_SEARCH_API_KEY"
timeout = 30
max_results = 5
```

`read_image` is available only when the active model deployment supports images. It is subject to the ordinary file and sensitive-path policy. See [Models](models.md) for deployment capabilities.

`web_fetch` is the preferred route for fetching URL content: it is read-only, converts HTML pages to markdown, and frames fetched content as untrusted data. Prefer it over shell `curl`/`wget` for page fetches. This is routing guidance, not an enforced rule. When a fetch does go through the shell, the destination-aware egress rule applies — the settled exfiltration decision of the v0.5.9 permission realignment. Modeled `curl` and `wget` calls (including through modeled wrappers) resolve by destination, not by client name: literal loopback destinations resolve run-class with no dialog — unless a proxy binding is in effect: a proxy-family variable in the invocation's effective environment — the trust-gated injected project environment plus the invocation's own command-line assignments (`http_proxy`, `https_proxy`, `all_proxy`, and the ftp/socks variants, matched case-insensitively) — set to a non-loopback proxy makes the destination effectively unestablishable, since the connection egresses through that proxy, and resolves ask-class regardless of the destination. That effective environment is the whole inspected set: the AMBIENT (non-injected) environment — a user-set ambient proxy variable included — is not inspected and remains the user's own trust boundary. An ambient client startup config is likewise the user's own trust boundary: a pre-existing `~/.curlrc`, or the `WGETRC`-named startup file the user's own environment points `wget` at, is not inspected, while an agent-set `CURL_HOME`, `WGETRC`, or `SYSTEM_WGETRC` assignment is denied as a dangerous environment assignment. A lowercase proxy assignment spelled in the command itself never reaches that environment: the shell policy denies any environment assignment whose name is not uppercase, whether as a command prefix (`http_proxy=... curl ...`) or as an `env` operand (`env http_proxy=... curl ...`) — a fail-closed denial, not an approval question. The ask-class proxy binding is observable through the spellings that do carry the variable: uppercase command-prefix and `env` assignments, the modeled `sudo -u` wrapper's `NAME=value` operands (recovered whatever their case), and the injected project environment. Any other destination — another IP literal, a DNS name, or a destination the recognizer cannot parse (fail-closed, never loopback) — resolves ask-class (`network.egress`, the same grantable class as git push) and requires approval. Three destination-changing behavior classes resolve ask-class even for a literal loopback URL: redirect-following flags (`curl -L` and equivalents), clients whose default behavior follows redirects (`wget`, unless the invocation disables it), and destination-overriding options (`curl --resolve`, `--connect-to`, and equivalents). The recognized loopback spellings are the name `localhost` (case-insensitive, with or without the root-anchored trailing dot), the whole `127.0.0.0/8` range, every spelling that normalizes to `::1`, the IPv4-mapped IPv6 spellings of that range (`::ffff:127.0.0.1`, compressed or hexadecimal variants included — the standard library's loopback test unwraps the IPv4-mapped address, and Linux routes the connect through IPv4), and the unspecified address `0.0.0.0`, which Linux connects to loopback; the IPv6 unspecified address `::` is not in the recognized set (only the IPv4 unspecified `0.0.0.0` is) and resolves ask-class like any other non-loopback destination; the `inet_aton` shorthand spellings of one to four dotted ASCII-decimal components are recognized too, but only when every reading — plain decimal and octal-aware — parses in range and lands in the loopback range or on `0.0.0.0` (`127.1`, `127.000.000.001`, `0`). An ambiguous shorthand such as `0177.0.0.1`, whose octal and decimal readings disagree, stays unrecognized and resolves ask-class, as do hex components and out-of-range ones: fail-closed, never loopback. Ordinary development egress stays run-class by client, not ask-class: `git fetch`/`clone`/`pull` and `pip`/`uv` installs do not route through dialogs. A user-configured denylist entry for one of these clients remains an absolute veto above this rule.

`web_fetch` follows redirects only through a redirect-target allowlist that contains exactly the originally requested netloc (host and port, compared case-insensitively). Redirects are followed manually with the client's redirect limit, and before each hop the resolved target's netloc is compared against that allowlist. An absolute redirection to a different host is denied with `Redirect target is not permitted: <url>`, and exceeding the redirect limit is denied with `Too many redirects`. The allowlist is not user-configurable and has no additional entries: to reach content served from a different host, fetch that host's URL directly.

## Background stop and replacement

`cancel_agent(agent_id, run_id=None)` requests a stop for the owning parent's background run. Pin `run_id` to avoid stopping newer work; omission selects the active run atomically. Its dispositions are `stop_requested`, `already_stopping`, `already_finishing`, `not_running`, `unknown_run`, and `forbidden`. Acceptance is not terminal cancellation: use `wait_for_agent` or `get_agent_result` after cleanup. Partial accumulated output and the winning stop reason are preserved subject to retention; zero-retention may evict the agent at finalization. Foreground runs are unsupported.

`task(agent_id=..., replace_run=True)` stops busy work and launches its replacement in the same conversation only after old cleanup and result publication join. It holds the existing capacity allocation, works at capacity, and injects a deterministic supersession frame. Busy replacement forbids launch-config and identity changes; idle reuse keeps reconfiguration. Check `launch_outcome`: `already_stopping`, `already_finishing`, and `rejected_reservation` mean no launch. Successful replacement reports `metadata.replaced_run_id` and `metadata.replacement_run_id`. Caller cancellation or authority revocation prevents replacement admission; admission failure never resumes the old run.

Neither operation rolls back file, shell, or remote side effects. `release_agent` is destructive: it discards identity and results. Stop and replacement use ordinary tool filtering and permissions, with Task's effective permissions checked before any busy run is stopped.

## Filtering and permissions

Use `enabled_tools` to narrow the available set and `disabled_tools` to remove from that result. Patterns can be exact names, globs, or case-insensitive full-match regular expressions prefixed with `re:`.

```toml
enabled_tools = ["read_file", "grep", "web_*"]
disabled_tools = ["web_fetch"]

[tools.bash]
permission = "always"
```

The current per-tool permission values are `always` and `never`:

- `always` executes the tool automatically, subject to permission policy and runtime safety checks.
- `never` prevents the tool from running.

These are tool permissions, not application-wide operating modes. The former `plan`, `ask`, `accept-edits`, and `auto-approve` mode set is not part of the current configuration. Routine tool calls do not request interactive approval: execution remains governed by permission policy, denylists, sensitive-file and workspace checks, and other runtime safeguards. Child agents cannot receive authority beyond their parent. MCP tools use the same filtering and permission mechanisms; see [MCP](mcp.md).

## Trusted folders

Chartreux asks before trusting a new workspace when it detects material that can influence agent behavior. The prompt identifies `AGENTS.md`, `.chartreux/` and `.agents/` configuration directories, and relevant repository-context files, including instruction files between the repository root and the current directory.

Trust can apply to the current folder, the repository root when offered, or the current session. Declining records the folder as untrusted. Persisted decisions are stored in `~/.chartreux/trusted_folders.toml`. Trusted project configuration, prompts, skills, hooks, and instructions are then eligible to load; untrusted project configuration is not. Passing `--add-dir` is an explicit trust grant for that session.

Inside a trusted folder, the project `.env` (the trusted folder's `.env` plus `$CHARTREUX_HOME/.env`) is injected into `bash` and `bash_start` subprocess environments, and its values are masked in tool output; see [Project environment injection](configuration.md#project-environment-injection).

## Shell safety

`bash` runs a finite command in a fresh POSIX shell. Standard input is closed, a timeout is enforced, and stdout and stderr are returned separately. Shell state, process handles, continued stdin, polling, and cursor-based output do not persist between calls. Configure its `permission`, `max_output_bytes`, `default_timeout`, `denylist`, `denylist_standalone`, and `sensitive_patterns` under `[tools.bash]`; runtime path and sensitive-file protections still apply.

The default `denylist` no longer contains the recognized dangerous git operations (`git push`, `git checkout`, `git restore`, `git stash drop`, `git stash clear`, `git switch --discard-changes`, `git switch -f`, `git reflog expire`, `git reflog delete`): they are classified by the [dangerous-command policy](#dangerous-git-commands) and denied pending typed authority. Explicit `denylist` entries remain an absolute veto: a matching entry denies the command before any grant is even considered, and only the user removing the entry lifts it. Hard guards are separate and non-overridable: `git reset --hard` and forced non-dry-run `git clean` stay denied regardless of `denylist`, `permission`, or approval settings, and `git -c ...`/`--config-env` spellings are denied by the policy layer even with the denylist cleared. Recursive `rm` is denied on the same fail-closed terms except where the git-status-aware operand analysis establishes the clean-tree carve-out (see [Recursive rm analysis](#recursive-rm-analysis)). The denylist covers direct command spellings and modeled wrappers (for example `git -C . push`); it does not cover arbitrary unmodeled executors such as make targets, scripts, or inline interpreter code.

## Dangerous git commands

Recognized dangerous git operations have first-class typed policy instead of denylist membership. A pure classifier assigns each whole shell call an operation set from a frozen registry, and execution is denied unless authority exists for every required operation in the command's exact repository scope (`git_dir`/`work_tree`; sibling worktrees are isolated).

There are three policy classes:

- **Publication** (`git.push`, ordinary `git push` only; `network.egress`, non-loopback egress through a modeled client) — grantable in an interactive session with the choices Deny (default), Allow once, Allow this session, or Always for this project.
- **Consequential** (`git.push.force`, `git.push.delete`, `git.push.mirror`, `git.checkout`, `git.restore`, `git.switch.discard`, `git.stash.drop`, `git.stash.clear`, `git.reflog.expire`, `git.reflog.delete`) — grantable with Deny (default), Allow once, or Allow this session. **Never** "Always": a push combining consequential mechanisms requires all classified consequential operations, and `git.push` authority never covers them.
- **Hard guards** (`git reset --hard`, forced non-dry-run `git clean`, unmodeled or ambiguous forms; recursive `rm` everywhere outside the clean-tree carve-out below) — no operation ID and no grant path: the user runs the command themselves.

Grant and denial semantics:

- A dangerous command denied for lack of authority triggers a runtime approval question in interactive sessions. Approval is **permission authority, not standing intent to publish**: a grant lets the tool call run; it is not a standing instruction that the project should be published or that further dangerous commands are pre-approved.
- **Allow once** authorizes exactly one invocation of the original call, including every listed occurrence of the command; it is never shared across concurrent identical calls.
- **Allow this session** installs session authority keyed by operation and repository scope; it survives compaction and `/reload` and expires with the session, a new conversation or branch, or resume (session grants are not auto-restored).
- **Always for this project** persists the grant to the protected user-layer `dangerous_command_grants_by_project` field (only the publication operations, `git.push` and `network.egress`, are persistable). The saved grant keys to the session's project (`cwd`), matching the authorized-roots flow; for a cross-project command targeting an authorized root, the saved grant still keys to the session's own project.
- **Refusal memory**: a denial records the decided call's repository-bound canonical identity, so the exact decided call returns an honest denial instead of re-prompting. Whitespace and quoting variants that tokenize identically match; a reformulated command re-prompts. It is per decided call, not a blanket block.
- **Recheck is the final gate**: an approval installs runtime authority and then the full permission pipeline rechecks (tool disablement, denylist, sensitivity, path and parent-ceiling restrictions, repository identity). On a recheck failure nothing executes and an unused once authority is destroyed. Child approval never lifts a missing parent dangerous-command authority.
- **Noninteractive sessions** deny dangerous commands promptly with remediation naming the actual path: publication operations point to the persistent knob (`Add git.push to dangerous_command_grants_by_project for this project in your user config.toml`); consequential operations state they cannot be pre-authorized and require approval in an interactive session; a mixed call's remediation covers only the persistable subset, naming the operations that still need interactive approval. The denial carries the resolver's reason. This is the headless statement for every ask-class operation: dangerous git commands and `network.egress` (non-loopback egress through modeled `curl`/`wget`) alike are denied with remediation in headless/programmatic mode, and `dangerous_command_grants_by_project` is the pre-authorization path for both.
- **Child sessions** deny dangerous commands outright regardless of user-input capability: the denial with remediation returns a blocker to the parent, and `dangerous_command_grants_by_project` is the only child pre-authorization path — a saved project grant runs in a child with no prompt. Main interactive sessions keep their approval dialogs. The same blanket denial covers ask-class `network.egress` egress and the out-of-root grant flow: a child never surfaces the interactive root-grant prompt, and out-of-root scope comes only from already-issued child-local grants, parent-directed scope changes, or `authorized_roots_by_project`.
- Revocation and inspection: the app server exposes `policy/dangerous_commands/read` and `policy/dangerous_commands/revoke` (see [App server](../integrations/app-server.md)); there is deliberately no public "grant an arbitrary operation" endpoint — grants come only from the runtime approval flow. The Textual settings panel **Settings > Shell Permissions** (in the Advanced & Tools group) shows live and saved dangerous-command authority and lets the user revoke it.

### Breaking change

Recognized git operations left the shipped denylist, and `denylist = []` no longer implicitly authorizes them: an empty explicit denylist used to leave the agent free to push; now publication needs a runtime grant (or a saved project grant) exactly as before. Explicit denylist entries remain an absolute veto, so an explicit `denylist = ["git push"]` still blocks pushes outright.

## Local tool trust

A tool authored under the user's own tools directory (`~/.chartreux/tools/`) is trusted by file location: its source file is the user's own code, so the harness admits it to the non-mutating posture — it resolves for read-only identities — without a further provenance check. Trust is granted only by the gated discovery path when the tool's source file resolves under the tools directory. Everything else keeps the fail-closed default: project-discovered tools, `tool_paths` entries, direct registrations without a source file, and MCP proxies (even one whose own code lives in the tools directory) classify as mutating and stay denied to non-mutating identities. Shipped builtins are classified by their defining shipped module; the separate provenance-stamp machinery was removed in v0.5.9. Name-keyed launch-admission guards stay fail-closed, and ordinary tool filtering still applies to location-trusted tools.

## Artifact secret scan

Chartreux scans the session's touched paths — the repository paths an agent session created or edited through the file edit/write tools — for credentials with a gitleaks-style rule set (patterns plus entropy checks). The scan runs automatically at two boundaries: when an agent session reports its work complete, before the result is surfaced as final, and as a pre-commit hook for user-driven commits. A failed scan blocks the submission; the finding is redacted from the transcript, and the agent is told the artifact failed the scan, not what it matched. An artifact the scan cannot fully inspect (larger than the per-file scan cap, unreadable, or in a directory the walk cannot enumerate or that exceeds its entry budget) blocks exactly like a finding: the scan is fail-closed, never silently clean. Binary content is not skipped: NUL-containing files are split on NUL boundaries and every text run is scanned, so a secret appended behind a NUL byte is still found. The scan is harness-side automatic and is never model-invoked.

Scope note: files written through the shell — redirects, `tee`, `sed -i`, heredocs, `cp`/`mv`/`install` destinations, `dd` output files — are not touched paths, but they are no longer unseen: the shell permission resolver collects each shell command's write-classified paths (`shell_command_write_paths`) and the submission gate adds them to the scan scope. The modeled network clients' downloads join that surface: `curl -o`/`--output` and `wget -O`/`--output-document` were always collected, and the derivable default downloads are too — `curl -O`/`--remote-name` and a plain `wget <url>` write the URL's file part in the working directory (under `--output-dir`/`-P` when given), so the scan sees them. The derivation is per URL for curl and per invocation for wget: curl's `-o`/`--output` and `-O`/`--remote-name` operands pair with the URLs one-for-one in command-line order (curl's man: "the first -o is for the first URL and so on"), so a paired `-o -` writes to stdout, only a URL with no paired output takes the `--remote-name-all` default, and every derivable destination of a multi-URL invocation is collected — with `--output-dir` composed into each relative `-o` operand, the collected write is the path curl actually writes; wget's `-O`/`--output-document` concatenates all documents into its one file, so it is itself the collected write and suppresses the derived names for every URL. Several download names are not derivable before the run and stay accepted scan limits: `curl -J`/`--remote-header-name` and `wget --content-disposition` save under the server-chosen `Content-Disposition` filename, which only exists once the response arrives, so such a download's file is not in the collected write surface (the URL-derived name the resolver does collect is not the file those flags write); `wget -E`/`--adjust-extension` renames the downloaded file after the transfer to match its content type, so the derived URL name is not the final file; and `wget -i`/`--input-file` reads its URLs from a file, so no download names are derived at all. Permitted inline interpreters are a named accepted residual on the other side: `python -c` and same-class `code -c`-style invocations, admitted by pinned policy, execute opaque code whose writes cannot be statically collected, so their write surface is outside the scan's guarantee. When a command's write surface cannot be established (an unmodeled construct, a failed analysis), the gate records an unknown-write-footprint sentinel and blocks the submission exactly like a scan-incomplete artifact, so the shell's write surface is covered fail-closed too. The backstops for what remains are the redaction backstop, which masks known secret values in tool output, and the pre-commit hook, which scans the staged content of the files actually being committed (read from the git index, not the working tree, so reverting a file on disk cannot launder a staged secret). A file staged for deletion ships no content, so the hook has nothing to scan for it; a path whose staged content cannot be read for any other reason blocks the commit fail-closed.

## Managed shell jobs

Use managed jobs for a server, watcher, or other command that must continue across
turns. Run the target in the foreground inside the managed command; do not add
`&`, `nohup`, or other detachment syntax. Standard input is closed, there is no
PTY or continued stdin, and stdout and stderr are merged into one captured stream.
Launch-time working-directory, environment, and capacity overrides are not accepted.

| Tool | Arguments | Result |
| --- | --- | --- |
| `bash_start` | Nonempty `command` (at most 64 KiB UTF-8); optional nullable, single-line `label` (at most 64 characters). | `job` summary and `next_cursor = 0`. Launch acknowledgment is not readiness. |
| `bash_read` | `job_id`; `cursor = 0`; `max_bytes = 4096` (4096–64000); `wait_seconds = 0` (0–30). | `job`, `records` (`sequence`, `text`), `first_cursor`, `next_cursor`, `end_cursor`, and `lost_records`. |
| `bash_stop` | `job_id`. | Final `job` summary and `already_finished`. |
| `bash_list` | `include_finished = false`. | Visible `jobs` summaries, without output bodies. |

Summaries contain an opaque `job_id`, bounded/redacted `command` and `label`,
`state`, nullable `exit_code` and `error`, and `output_complete` and
`output_incomplete`. States are `starting`, `running`, `stopping`, `exited`,
`failed`, and `stopped`. A nonzero command exit is still `exited`; `failed`
indicates infrastructure or capture failure. `output_complete` means no more
records will arrive; `output_incomplete` indicates capture loss or abandonment.
Both flags may be true. Neither flag reports cursor eviction.

Cursors are record sequences starting at zero, not byte offsets. Forward each
read's `next_cursor` to the next read. Independent readers can replay retained
records without consuming another reader's position. A cursor below
`first_cursor` reports the exact gap in `lost_records` and resumes at the first
retained record; a cursor beyond `end_cursor` is rejected. `max_bytes` bounds
returned, redacted output-body UTF-8 bytes, not the result envelope. Empty reads
at the end can wait for output or completion; a timeout does not stop the job.
Use modest budgets and waits (for example, 5–15 seconds), not busy polling or
repeated reads of unchanged windows. A bounded read wait is an ordinary tool
call, not a steerable subagent wait.

One root session and all its children share eight allocations, including pending
launches and jobs whose cleanup has not settled. The registry retains at most 32
job records, evicting finished records only. Each job retains at most 64 immutable
output records or 256 KiB, whichever binds first; each record is at most 4096
UTF-8 bytes. This is a rolling buffer, not an output archive. Returned pages can
also be persisted in the transcript, so repeated reads grow history independently
of these bounds. After compaction, recover handles with `bash_list`.

A bounded workflow (also useful for checking the tool contract) is:

```text
bash_start(command="printf 'ready\\n'; sleep 30", label="workflow check")
bash_read(job_id="<returned-job-id>", cursor=0, max_bytes=4096, wait_seconds=5)
# Continue normal work; use the read result's next_cursor below.
bash_read(job_id="<returned-job-id>", cursor=<next_cursor>, max_bytes=4096, wait_seconds=0)
bash_stop(job_id="<returned-job-id>")
bash_list(include_finished=true)
```

For a server, start a foreground target such as `python -m http.server 8000`
instead. Inspect its output and use a separate foreground check to establish
readiness before relying on it; a launch acknowledgment or a printed marker is
not itself proof that a service is ready. Stop the job when it is no longer
needed. Stop requests terminate the owned process group, allowing about two
seconds for SIGTERM before SIGKILL; stopping does not undo file or remote effects.
A late stop preserves natural completion.

`bash_start` uses the canonical `[tools.bash]` permission, denylist, sensitive-path,
workspace, and inherited-parent restrictions. It also applies the same
dangerous-command policy as `bash`: `bash` and `bash_start` behave identically, a
dangerous launch prompts the user for approval in interactive sessions (and is
denied with remediation otherwise), and a denied launch creates no active job.
The hard guards (`git reset --hard`, forced non-dry-run
`git clean`, and recursive `rm` outside the clean-tree carve-out) are not
grantable. There is no separate
`[tools.bash_start]` permission key and no separate interactive launch approval
beyond the dangerous-command flow. Ordinary
tool filtering still controls availability; read, stop, and list have their own
ordinary tool permissions and ownership checks. Disabling launch does not itself
disable recovery tools. Live jobs or pending launches restrict execution-authority
changes and worktree relocation; unrelated display settings remain editable.

Managed jobs survive normal turns, interruption after launch commit, compaction,
and their creator child's completion. That survival is also a scan-scope residual:
a background job started with `bash_start` can outlive its child's submission and
write its artifact after the submission scan has passed. The scan's guarantee
covers writes up to submission; reads are still scanned at access time. This is
documented, not gated. They end with the root runtime; see
[session lifetime](sessions-workspaces.md#managed-job-lifetime) and
[child access](subagents.md#managed-shell-jobs). This is neither a sandbox nor a
daemon manager: arbitrary program effects are not contained, escaped descendants
may survive, and crashes or forced process termination cannot guarantee cleanup.
Jobs are not adopted after a restart.

## Shell authority and diagnostics

Shell authority is bounded, not sandboxed. Chartreux inspects modeled command
arguments, redirects, working directories, and known executor boundaries;
unmodeled syntax and analysis-budget overflow are denied. These checks do not
contain arbitrary program effects, package scripts, network access, or inline
interpreter code to the workspace.

Inline interpreter code (for example, `python -c`, `node -e`, and their modeled
`uv run` forms) is allowed by default. Its source is not treated as a file operand
or a shell-expanded glob. Users can deny these forms through `[tools.bash]`
`denylist` entries such as `"python -c"` or `"node -e"`; modeled inner commands
also receive those checks. Bare interactive interpreter commands remain denied
by the default standalone denylist. Inline-code recognition is not code inspection.

### Wrapper and flag recognition (v0.5.9)

`sg` is modeled as a transparent wrapper: its `-c` payload is re-analyzed as if
invoked directly, including unquoted multi-token payloads (an earlier spelling
left those unanalyzed). `sudo -u <user>` is likewise a transparent wrapper —
the wrapped command is re-analyzed rather than denied as a sensitive prefix.
`git symbolic-ref` is modeled by mode: read invocations resolve, and
ref-writing invocations are classified like other ref-writing git commands.
Executor option parsing recognizes the interpreter's own flags, so
`.venv/bin/python --version` resolves while bare `.venv/bin/python` stays
denied, and `VIRTUAL_ENV` is no longer a dangerous environment assignment (it
is a benign venv hint), so a `VIRTUAL_ENV=...` prefix no longer denies.

Probed egress-capable clients without destination modeling — `ssh`, `scp`,
`rsync`, `ftp`, `telnet`, `http`/`httpie`, `wget2`, `aria2c`, `lftp` — ship on
the default `[tools.bash]` denylist until they are modeled; removing an entry
re-enables that client as an ordinary denylist edit.

### Recursive rm analysis

Recursive `rm` is denied fail-closed on the hard-guard terms unless the
git-status-aware operand analysis establishes the clean-tree carve-out. Over a
tree with unsaved changes it stays a hard-guard denial — never autonomous, no
approval path. Over a git-clean tree it resolves run-class only when every
expanded operand lies entirely within the repository whose clean status was
assessed, does not overlap its Git metadata (neither encompassing the Git
directory nor lying inside it, including linked-worktree common directories),
and recovery is established for every repository contained in or overlapped by
the operand, nested repositories included. Every unestablishable aspect —
operand scope, metadata overlap, recovery status, glob expansion, path
resolution — and any mixed operand (an eligible one alongside an ineligible
one) resolves the whole command to the hard-guard denial.

Its hard-guard denial names the original
option or cluster, such as `-r`, `-rf`, or `--recursive`, and recommends
`rm -- 'file'`, then `rmdir -- 'dir'`. Rejected path-glob diagnostics identify the
candidate token and command segment and explain that they cannot be safely
inspected. Feedback previews are bounded, registered secrets are redacted before
escaping or truncation, and control characters and markup delimiters are escaped.
These messages explain existing denials; they do not grant authority.

### Retained shell-policy limitations

Some over-denials and inspection gaps are retained, documented trade-offs in
this release; each names its fast-follow pointer:

- **Trailing-`**` over-denial.** A run of two or more consecutive stars in a
  path glob — including a trailing `**` — fails closed even when the shell's
  own expansion is safe: zsh's recursive globstar spans directory levels the
  one-component model cannot expand faithfully. The over-denial is retained
  until the executor shell-option contract (fast-follow) lands.
- **Extglob residual hazard.** The static model inspects plain POSIX globbing
  only. The executor's shell normally runs with extended globbing off, but
  under an extendedglob-active environment (for example one activated in
  `~/.zshenv`) extended patterns can match more than the static model
  inspects — an under-inspection hazard. Detection and modeling are
  fast-follow, behind the executor shell-option contract.
- **Quoted-variable conservative denial.** Static expansion intentionally
  resolves only plain unquoted words in provably read-only argument positions
  of whitelisted commands, bound only by literal assignments proven unchanged
  through the invocation. A quoted variable — or a variable in any other
  position — is conservatively denied rather than resolved. Resolution is
  fast-follow.
- **Find pattern-operand carve-out.** Quoted or literal `find -name`,
  `-iname`, `-path`, and `-ipath` operand tokens bypass path-glob inspection:
  the shell never pathname-expands them, so they remain pure pattern data,
  and the carve-out is retained for them only. The earlier assessment that
  the bypass reaches "out-of-root directory names only, never file contents"
  was wrong: an unquoted glob-bearing pattern operand IS pathname-expanded
  by the shell before find runs, and the expansion products re-enter find's
  argv — a decoy file named `-exec` (or `-a`, `-delete`, ...) in the working
  directory turned `find . -name -* ...` into predicate injection,
  executing an arbitrary command through an injected `-exec`. The inspector
  cannot expand pattern operands, so it fails closed: an unquoted
  glob-bearing pattern operand is denied with the generic path-glob
  diagnostic, exactly like a redirect target that looks like a glob.
  Quoting the pattern (`find . -name '*.py'`) or writing a literal operand
  keeps the command allowed.
- **Glob-expansion budgets.** Pathname-expansion inspection runs under work
  budgets — a directory-entry cap and a match-state cap. Exceeding either
  fails closed; the caps are the accepted fail-closed trade-off that keeps a
  crafted pattern from turning permission resolution into an unbounded walk.
  The entry cap binds after enumeration — it caps continued matching work,
  not enumeration. Cap values are unchanged in this release; cap changes are
  deferred and remain a separate user decision with a stated bound rationale.

### Expansion-failure provenance

A denied path glob that failed during expansion inspection names the specific
limit it reached: the directory-entry budget, the match-state budget, or
unsupported glob syntax this inspector cannot expand. Denials decided outside
the expansion inspection — a match the caller rejects (an option-like word, a
match outside the authorized workspace) or a token the inspector never
expanded (an unknown cwd, a home-relative word, a partially quoted
metacharacter) — render the generic path-glob message and are never
attributed to a budget. The provenance detail rides the bounded, redacted
diagnostic field, and every failure kind denies exactly as before.

## Interactive questions

`ask_user_question` works only in an interactive UI. Each question requires two or more options; two to four is recommended, not a maximum. An `Other` free-text choice is added unless `hide_other` is true, and questions may permit multiple selections. Programmatic runs deny interactive callbacks rather than displaying them.

For multi-select questions, Space toggles membership. Enter on an ordinary
unchecked option adds it before accepting the answer. Clicking the acceptance
row follows the same validation and acceptance rules as Enter; navigation alone
does not submit an answer.

For reusable instructions that guide tool use, see [Instructions and skills](instructions-skills.md). For non-interactive workflows, see [Automation](automation.md).
