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

`web_fetch` is the preferred route for fetching URL content: it is read-only, converts HTML pages to markdown, and frames fetched content as untrusted data. Prefer it over shell `curl`/`wget` for page fetches. This is routing guidance, not an enforced rule. When a fetch does go through the shell, the destination-aware egress rule applies — the settled exfiltration decision of the v0.6.0 permission realignment. Modeled `curl` and `wget` calls (including through modeled wrappers) resolve by destination, not by client name: literal loopback destinations resolve run-class with no dialog — unless a proxy binding is in effect: a proxy-family variable in the invocation's effective environment — the trust-gated injected project environment plus the invocation's own command-line assignments (`http_proxy`, `https_proxy`, `all_proxy`, and the ftp/socks variants, matched case-insensitively) — set to a non-loopback proxy makes the destination effectively unestablishable, since the connection egresses through that proxy, and resolves ask-class regardless of the destination. That effective environment is the whole inspected set: the AMBIENT (non-injected) environment — a user-set ambient proxy variable included — is not inspected and remains the user's own trust boundary. An ambient client startup config is likewise the user's own trust boundary: a pre-existing `~/.curlrc`, or the `WGETRC`-named startup file the user's own environment points `wget` at, is not inspected, while an agent-set `CURL_HOME`, `WGETRC`, or `SYSTEM_WGETRC` assignment is denied as a dangerous environment assignment. A lowercase proxy assignment spelled in the command itself never reaches that environment: the shell policy denies any environment assignment whose name is not uppercase, whether as a command prefix (`http_proxy=... curl ...`) or as an `env` operand (`env http_proxy=... curl ...`) — a fail-closed denial, not an approval question. The ask-class proxy binding is observable through the spellings that do carry the variable: uppercase command-prefix and `env` assignments, the modeled `sudo -u` wrapper's `NAME=value` operands (recovered whatever their case), and the injected project environment. Any other destination — another IP literal, a DNS name, or a destination the recognizer cannot parse (fail-closed, never loopback) — resolves ask-class (`network.egress`, the same grantable class as git push) and requires approval. Three destination-changing behavior classes resolve ask-class even for a literal loopback URL: redirect-following flags (`curl -L` and equivalents), clients whose default behavior follows redirects (`wget`, unless the invocation disables it), and destination-overriding options (`curl --resolve`, `--connect-to`, and equivalents). The recognized loopback spellings are the name `localhost` (case-insensitive, with or without the root-anchored trailing dot), the whole `127.0.0.0/8` range, every spelling that normalizes to `::1`, the IPv4-mapped IPv6 spellings of that range (`::ffff:127.0.0.1`, compressed or hexadecimal variants included — the standard library's loopback test unwraps the IPv4-mapped address, and Linux routes the connect through IPv4), and the unspecified address `0.0.0.0`, which Linux connects to loopback; the IPv6 unspecified address `::` is not in the recognized set (only the IPv4 unspecified `0.0.0.0` is) and resolves ask-class like any other non-loopback destination; the `inet_aton` shorthand spellings of one to four dotted ASCII-decimal components are recognized too, but only when every reading — plain decimal and octal-aware — parses in range and lands in the loopback range or on `0.0.0.0` (`127.1`, `127.000.000.001`, `0`). An ambiguous shorthand such as `0177.0.0.1`, whose octal and decimal readings disagree, stays unrecognized and resolves ask-class, as do hex components and out-of-range ones: fail-closed, never loopback. Ordinary development egress stays run-class by client, not ask-class: `git fetch`/`clone`/`pull` and `pip`/`uv` installs do not route through dialogs. A user-configured denylist entry for one of these clients remains an absolute veto above this rule.

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

`chartreux trust list` prints every persisted decision, labeled `trusted` or `untrusted`. `chartreux trust list PATH` prints the decision that governs one path: the closest recorded decision wins, so a child path inherits its nearest recorded ancestor rather than the entry above it. `chartreux trust remove PATH` removes exactly the one entry recorded for `PATH` — sibling and descendant entries and every decision recorded for another path are untouched, and removing a path that only inherits trust from an ancestor is refused because nothing is recorded for it. The removal asks for confirmation and defaults to no; without an interactive terminal it is denied, and there is no force or assume-yes flag. A running session keeps the decision it loaded, so restart Chartreux after a removal for it to take effect.

Inside a trusted folder, the project `.env` (the trusted folder's `.env` plus `$CHARTREUX_HOME/.env`) is injected into `bash` and `bash_start` subprocess environments, and its values are masked in tool output; see [Project environment injection](configuration.md#project-environment-injection).

## Shell safety

`bash` runs a finite command in a fresh POSIX shell. Standard input is closed, a timeout is enforced, and stdout and stderr are returned separately. Shell state, process handles, continued stdin, polling, and cursor-based output do not persist between calls. Configure its `permission`, `max_output_bytes`, `default_timeout`, `denylist`, `denylist_standalone`, and `sensitive_patterns` under `[tools.bash]`; runtime path and sensitive-file protections still apply.

The default `denylist` no longer contains the recognized dangerous git operations (`git push`, `git checkout`, `git restore`, `git stash drop`, `git stash clear`, `git switch --discard-changes`, `git switch -f`, `git reflog expire`, `git reflog delete`): they are classified by the [dangerous-command policy](#dangerous-git-commands) and denied pending typed authority. Explicit `denylist` entries remain an absolute veto: a matching entry denies the command before any grant is even considered, and only the user removing the entry lifts it. Hard guards are separate and non-overridable: `git reset --hard` and forced non-dry-run `git clean` stay denied regardless of `denylist`, `permission`, or approval settings, and `git -c ...`/`--config-env` spellings are denied by the policy layer even with the denylist cleared. Recursive `rm` is denied on the same fail-closed terms except where the git-status-aware operand analysis establishes the clean-tree carve-out, or where every operand resolves inside the session scratchpad (see [Recursive rm analysis](#recursive-rm-analysis)). The denylist covers direct command spellings and modeled wrappers (for example `git -C . push`); it does not cover arbitrary unmodeled executors such as make targets, scripts, or inline interpreter code.

## Dangerous git commands

Recognized dangerous git operations have first-class typed policy instead of denylist membership. A pure classifier assigns each whole shell call an operation set from a frozen registry, and execution is denied unless authority exists for every required operation in the command's exact repository scope (`git_dir`/`work_tree`; sibling worktrees are isolated) — except for the routine branch creation and the narrow state-aware clean-tree checkout exemption described below.

There are four policy classes:

- **Publication** (`git.push`, ordinary `git push` only; `network.egress`, non-loopback egress through a modeled client) — grantable in an interactive session with the choices Deny (default), Allow once, Allow this session, or Always for this project.
- **Remote-exec** (`network.remote-exec`, a command run on a remote machine) — grantable with the same four choices, but the persistent grant is a destination grant: it authorizes one exact `user@host:port` destination, never the operation in general, and never a contained payload. See [remote-exec destination grants](#remote-exec-destination-grants-v070).
- **Consequential** (`git.push.force`, `git.push.delete`, `git.push.mirror`, `git.checkout`, `git.restore`, `git.switch.discard`, `git.stash.drop`, `git.stash.clear`, `git.reflog.expire`, `git.reflog.delete`) — grantable with Deny (default), Allow once, or Allow this session. **Never** "Always": a push combining consequential mechanisms requires all classified consequential operations, and `git.push` authority never covers them.
- **Hard guards** (`git reset --hard`, forced non-dry-run `git clean`, unmodeled or ambiguous forms; recursive `rm` everywhere outside the clean-tree and session-scratchpad carve-outs below) — no operation ID and no grant path: the user runs the command themselves.

Grant and denial semantics:

- A dangerous command denied for lack of authority triggers a runtime approval question in interactive sessions. Approval is **permission authority, not standing intent to publish**: a grant lets the tool call run; it is not a standing instruction that the project should be published or that further dangerous commands are pre-approved.
- **Allow once** authorizes exactly one invocation of the original call, including every listed occurrence of the command; it is never shared across concurrent identical calls.
- **Allow this session** installs session authority keyed by operation and repository scope; it survives compaction and `/reload` and expires with the session, a new conversation or branch, or resume (session grants are not auto-restored).
- **Always for this project** persists the grant to the protected user-layer `dangerous_command_grants_by_project` field (the persistable operations are `git.push`, `network.egress`, and `network.remote-exec`; a remote-exec approval persists its exact destination to `remote_exec_grants_by_project` instead). The saved grant keys to the session's project (`cwd`), matching the authorized-roots flow; for a cross-project command targeting an authorized root, the saved grant still keys to the session's own project.
- **Refusal memory**: a denial records the decided call's repository-bound canonical identity, so the exact decided call returns an honest denial instead of re-prompting. Whitespace and quoting variants that tokenize identically match; a reformulated command re-prompts. It is per decided call, not a blanket block.
- **Recheck is the final gate**: an approval installs runtime authority and then the full permission pipeline rechecks (tool disablement, denylist, sensitivity, path and parent-ceiling restrictions, repository identity). On a recheck failure nothing executes and an unused once authority is destroyed. Child approval never lifts a missing parent dangerous-command authority.
- **Noninteractive sessions** deny dangerous commands promptly with remediation naming the actual path: publication operations point to the persistent knob (`Add git.push to dangerous_command_grants_by_project for this project in your user config.toml`); consequential operations state they cannot be pre-authorized and require approval in an interactive session; a mixed call's remediation covers only the persistable subset, naming the operations that still need interactive approval. The denial carries the resolver's reason. This is the headless statement for every ask-class operation: dangerous git commands, `network.egress` (non-loopback egress through modeled `curl`/`wget`), and `network.remote-exec` alike are denied with remediation in headless/programmatic mode, and `dangerous_command_grants_by_project` is the pre-authorization path (a remote-exec denial names `remote_exec_grants_by_project` and the exact destination instead).
- **Child sessions** deny dangerous commands outright regardless of user-input capability: the denial with remediation returns a blocker to the parent, and the persisted grant fields are the only child pre-authorization paths — `dangerous_command_grants_by_project` for the scope-keyed operations and `remote_exec_grants_by_project` for a remote-exec destination. A saved project grant runs in a child with no prompt, while a session grant never propagates: grants are store-local to the loop that owns them. Main interactive sessions keep their approval dialogs. The same blanket denial covers ask-class `network.egress` egress, remote-exec, and the out-of-root grant flow: a child never surfaces the interactive root-grant prompt, and out-of-root scope comes only from already-issued child-local grants, parent-directed scope changes, or `authorized_roots_by_project`.
- Revocation and inspection: the app server exposes `policy/dangerous_commands/read` and `policy/dangerous_commands/revoke` (see [App server](../integrations/app-server.md)); there is deliberately no public "grant an arbitrary operation" endpoint — grants come only from the runtime approval flow. The Textual settings panel **Settings > Shell Permissions** (in the Advanced & Tools group) shows live and saved dangerous-command authority and lets the user revoke it — destination-scoped for remote-exec, with the whole-scope revoke kept as its own confirmed row (see [remote-exec destination revocation](#remote-exec-destination-revocation-v071)). There is no sessionless CLI and no approval-prompt control: a revocation needs a live registered session.

### Routine branch creation (v0.7.8)

`git checkout -b <new-branch> [<start-point>]` is routine: the classifier assigns it no operation, so it neither needs nor prompts for consequential authority, and the repository state does not gate it. A dirty work tree, an in-progress operation marker, and a start point are all admitted — `git` refuses rather than discards when the start point's content conflicts with unsaved work. Rejected forms stay consequential: `-B`, `-f`/`--force`/`--discard-changes`, `-m`/`--merge`, `--orphan`/`--conflict`/`--pathspec-from-file`/`--recurse-submodules`, a pathspec or `--` tail, any other option, inline `-c` configuration, and repository relocation. Discard-shaped checkouts and the clean-tree exemption for `git checkout <existing-local-branch>` are unchanged.

Two protections remain load-bearing on the routine path. The executed command carries an injected `--no-overwrite-ignore` immediately after the `checkout` token — the command as written stays written for display and result surfaces — so an ignored-file collision fails closed instead of silently overwriting local content. And a repository whose configuration selects a program (`core.hooksPath` and the other program-selection keys) keeps the consequential approval class: the branch switch runs that program's `post-checkout` hook, which is unreviewed program execution in a child session that can never prompt. The configuration check is fail-closed: a repository that cannot be located, and a config that cannot be read, both keep the approval flow.

### State-aware clean-tree checkout exemption

`git checkout` keeps its consequential classification, but the permission resolver grants a narrow, state-aware exemption from the approval flow for the existing-branch switch. Exempt shape only: `git checkout <branch>` where the target is an existing local branch — no pathspec, no `--`, no `-f`/`--force`/`--discard-changes`/`--merge`/`--orphan`/`--conflict`/`--pathspec-from-file`/`--recurse-submodules`, no `-B`, and no other option. The exemption applies only when the entire composed shell call is exactly one eligible git invocation: no preceding or chained commands, no pipeline segments, no environment assignments, no wrapper commands, and no inline configuration options (any `git -c <key>=<value>` form is ineligible). Display-only global options (`--no-pager` and equivalents) do not break eligibility; repository relocation (`-C`/`--git-dir`/`--work-tree`) fails closed. Every ineligible shape falls back to the consequential approval flow above.

The exemption holds only when every candidate repository scope passes the state guardrails, each fail-closed on any unestablishable state: a fully clean work tree (no unstaged, staged, or untracked changes), no in-progress operation marker (merge, rebase, cherry-pick, revert), and no repository configuration that selects a program the invocation could run (`core.hooksPath` and the other program-selection keys). The switch form additionally requires the target to exist as a local branch (a sanitized `show-ref` probe), so a pathspec fallback, a tag, a remote-tracking name, or a commit-ish never rides the exemption.

Because the bash tool executes a command string, the executed command of every modeled whole-call checkout — the exempt switch and the routine branch creation alike — is rewritten at execution time: `--no-overwrite-ignore` is inserted immediately after the `checkout` subcommand token, leaving every other byte of the command untouched, so an ignored-file collision fails closed (checkout refuses) instead of silently clobbering local content; the original string stays original for display and result surfaces. Neither admission path bypasses the denylist veto, parent ceilings, or any other recheck. Branch merging, switches to remote-tracking branches, and operations on branches other than the current working branch remain user-owned. See [ADR 0019](../adr/0019-dangerous-git-permissions.md).

### Remote-exec destination grants (v0.7.0)

A command run on a remote machine is a separate operation, `network.remote-exec`, distinct from `network.egress`: an egress grant never admits a remote command, and a remote-exec destination grant never admits non-remote egress or publication. The grant model is destination-keyed, so authorization is a tuple of project scope, the operation, and one exact destination.

- **Exact identity, no patterns.** A destination is `user@host:port`. The user is mandatory; the host is lowercased; a missing port canonicalizes to 22. Grants match the exact canonical `(user, host, port)` tuple, so another user, another host, or another port is a different grant. Wildcards (`*`, `?`), ranges, and comma-separated alias groups never parse, and v1 has no alias-group support: an SSH-config alias is not a destination identity.
- **Lifecycle.** Deny records the decided call so it is not re-prompted. Allow once installs one destination-bound permit for exactly this invocation. Allow this session installs destination authority in the session store. Always for this project persists the destination to the protected user-layer `remote_exec_grants_by_project` field, keyed to the session's project. Session grants do not propagate to children; a persisted destination grant authorizes the parent and its children; children never prompt.
- **Honest scope.** The approval question states that remote commands and their effects are outside chartreux's local guards — trusting the destination, not approving a contained payload. The grant authorizes reaching the host; it does not contain or review what runs there.
- **Resolver boundary.** Loopback and local-machine destinations are hard-denied regardless of any grant, and the SSH grammar that produces a proposal is admitted separately from this grant model: a grant can only authorize a destination the resolver has already validated.

### Remote-exec destination revocation (v0.7.1)

A destination grant is revoked selectively. **Settings > Shell Permissions** lists each exact `user@host:port` destination as its own row under the section that holds it — live session, accepted config, or disk only — and revoking that row removes only that destination's session grants, invalidates every pending once permit containing it whole, and removes the same destination from the saved project list. The whole-scope revoke stays reachable as one separate row that requires an explicit `y` confirmation, because it drops every live destination grant in every scope and every saved destination of the project rather than one destination. The revoke rides the existing `policy/dangerous_commands/revoke` verb: the request carries the registered session, the operation, the exact canonical destination, the expected live generation, and the expected disk revision, and the response reports the live and saved outcomes apart. The read projection carries the destination on every remote-exec entry and reports the disk destinations separately from the disk operations.

A revocation is not a containment guarantee. Four limits are disclosed rather than implied:

- **An in-flight command is not undone.** A remote command already running keeps running: the grant stops the next invocation, not the current one. A question already awaiting an answer is not cancelled either, and answering it after the revocation still grants fresh authority for the decided scope; the revocation rechecks the live generation after its saved-side write and reports that concurrent regrant as a warning.
- **The accepted config can keep authorizing until a reload or start.** Saved authority is removed from the user config on disk for future sessions, while the running session keeps whatever its accepted config already granted until the next reload or session start. The response carries that warning and the panel shows it instead of claiming the authority is gone.
- **Children and other processes are outside any atomic picture.** Session grants are store-local to the loop that owns them, so revoking this session's grants cannot reach a child's store, another running process, or a session that is not registered with this app server. Only the registered session's live authority and the project's saved destinations are touched; there is no cross-process atomicity.
- **The client self-attests the outcome.** The panel reports what the revocation response says; it cannot independently verify that the server removed the authority, so a transport or server fault surfaces as an error or an uncertain outcome instead of a blind retry. The inspection that follows a revocation is a fresh read, so the list shows what the server reports next, not what the panel assumes.

Strict-client note: the destination parameter and the destination-aware projection follow this release's strict convention — an older strict app server rejects the new request field (the whole-scope request included, because the defaulted field still serializes as an explicit null) and an older strict client rejects the new response keys, so client and server must run the same protocol version.

### Breaking change

Recognized git operations left the shipped denylist, and `denylist = []` no longer implicitly authorizes them: an empty explicit denylist used to leave the agent free to push; now publication needs a runtime grant (or a saved project grant) exactly as before. Explicit denylist entries remain an absolute veto, so an explicit `denylist = ["git push"]` still blocks pushes outright.

## v0.7.1 remote-exec and shell-policy summary

This section is the release-level reading map; the clause-level passages above
and in [Network-client egress modeling](#network-client-egress-modeling-v070)
and [Retained shell-policy limitations](#retained-shell-policy-limitations)
remain authoritative for their own clauses.

Remote execution in v0.7.1 is a closed destination grammar behind the
`network.remote-exec` grant model. `ssh`, `scp`, and `rsync` are the only
clients classified `network.remote-exec`: one standalone invocation each, one
exact `user@host:port` destination per call, ask-class with that destination
attached to the approval, and the approved address plus the fixed host-key
identity pinned into the spawned argv at every transport. `ssh` admits an IP
literal or a hostname the system resolver answers uniquely and non-locally;
`scp` and `rsync` add the bounded-transfer grammar (two operands, one remote
operand, closed path charset, safe options only). A resolution that changes
between validation and spawn is never contacted, and a locality-only
attestation entry denies transfers even where it admits `ssh` execution.
Revocation is destination-scoped in **Settings > Shell Permissions** and on the
`policy/dangerous_commands/revoke` wire, with the whole-scope revoke kept as a
separate confirmed action; it stops future invocations and undecided future
authority, and it never claims to undo a running command or reach another
process.

The shell-policy side of the release is the pinned executor contract plus two
narrow glob admissions. Every spawned shell runs with the expansion-relevant
option families off, startup files redirected, and the locale pinned, while an
unknown executor, an option-changing invocation, and a nested shell re-entry
with an expansion-bearing payload fail closed; the contract is read at
execution, and the argv, shell-string, background, and terminal transports
either consume the bound plan or hard-deny. The executor itself is resolved
once at startup — the `[tools.bash] executor` config value, else bash when the
host has it, else the ambient `$SHELL`, else the kernel default — and that one
resolution feeds every spawn and the permission analysis. The shipped bash
default therefore outranks an ambient zsh login shell, and the zsh-only
equals-expansion rules fire only under a zsh executor, because bash never
expands a leading `=` word. The two admissions are a
single-component trailing `**` for the read whitelist and one whole
double-quoted variable word preceded by a literal assignment; everything else
retains its conservative denial. HTTPie's `https` entry point is modeled under
the `http` grammar with the https scheme forced only onto a schemeless operand.

The retained limitations are the same in both areas: a remote-exec destination
grant trusts the destination, not a payload; the pinned known-hosts file is not
provisionable yet, so an admitted `ssh` spawn fails host-key verification
closed; attestation is client-self-reported, so it stops accidental misrouting
rather than a lying client; strict clients must run the same protocol version
as the app server; revocation undoes no remote effect; the executor contract's
remaining trust base is root-owned system startup files and the shell binary;
and the shell work-budget caps are unchanged in this release.

## Local tool trust

A tool authored under the user's own tools directory (`~/.chartreux/tools/`) is trusted by file location: its source file is the user's own code, so the harness admits it to the non-mutating posture — it resolves for read-only identities — without a further provenance check. Trust is granted only by the gated discovery path when the tool's source file resolves under the tools directory. Everything else keeps the fail-closed default: project-discovered tools, `tool_paths` entries, direct registrations without a source file, and MCP proxies (even one whose own code lives in the tools directory) classify as mutating and stay denied to non-mutating identities. Shipped builtins are classified by their defining shipped module; the separate provenance-stamp machinery was removed in v0.6.0. Name-keyed launch-admission guards stay fail-closed, and ordinary tool filtering still applies to location-trusted tools.

## Artifact secret scan

Chartreux scans the session's touched paths — the repository paths an agent session created or edited through the file edit/write tools — for credentials with a gitleaks-style rule set (patterns plus entropy checks). The scan runs automatically at two boundaries: when an agent session reports its work complete, before the result is surfaced as final, and as a pre-commit hook for user-driven commits. A failed scan blocks the submission; the finding is redacted from the transcript, and the agent is told the artifact failed the scan, not what it matched. An artifact the scan cannot fully inspect (larger than the per-file scan cap, unreadable, or in a directory the walk cannot enumerate or that exceeds its entry budget) blocks exactly like a finding: the scan is fail-closed, never silently clean. Binary content is not skipped: NUL-containing files are split on NUL boundaries and every text run is scanned, so a secret appended behind a NUL byte is still found. The scan is harness-side automatic and is never model-invoked.

Within a run the scan is filtered by a per-run cutoff taken when the run's artifact accumulator is constructed. A touched path that names a file explicitly is inspected unconditionally — naming it is the session declaring it its own artifact (the one exception is the transient-root exclusion below) — while a touched directory is enumerated in full but only the files modified at or after the run began are inspected, so a session that merely wrote into an existing tree does not put pre-session content in front of the scan. The cutoff is a write-window filter, never a provenance check: content backdated before the run started, including content extracted with preserved timestamps, is not inspected — that is the documented blind spot — while a file whose metadata cannot be read is reported `scan-incomplete` rather than assumed old, and the walk's entry budget still counts every entry the walk enumerates unless the submission boundary's path-class scope is in effect (see below). A scan that supplies no cutoff keeps full-walk semantics. Symlinks follow their target's type: a link to a file is inspected when the link or its target falls inside the write window, a link to a directory is walked under cycle detection, and an unreadable target is uninspected content, reported rather than assumed unchanged.

### Scope by path class (submission boundary)

An agent that archives a log into its scratchpad, writes into a temp directory, or touches a repository root that carries `.git` and `.venv` should not have its submission blocked by content it did not author. At the submission boundary the touched paths are partitioned by path class before the scan runs.

| Path class | What it covers | Scan effect |
| --- | --- | --- |
| Repository | Everything else — the session workspace, files under its authorized roots, and anything a classification cannot establish as transient | Scanned and fail-closed exactly as above; vendored and cache directories are pruned from its walks |
| Transient | Paths under a runtime-owned scratchpad root (the session scratchpad shared with subagent children) and paths under the OS temp directory that lie outside every authorized workspace root | Skipped before any read; one bounded `INFO` log line names the paths and the counts. Never a finding, never a block, never model-visible |

Precedence and failure rules:

- The path is resolved (symlinks included) before it is classified, so a scratchpad symlink pointing into a repository classifies as repository content and stays scanned.
- A session workspace rooted under the temp directory keeps its own subtree in the scan: an authorized root's subtree always classifies as repository content, and its temp-directory siblings are transient.
- With no authorized root established there is nothing to compare against, so a temp-directory path stays in the scan rather than being skipped. A classification failure (an unresolvable path or root, an OS error) also classifies as repository content. The skip is fail-closed, never fail-open.

| Directory name pruned from walks | Reason |
| --- | --- |
| `.git`, `.hg`, `.svn` | Version-control metadata: packed objects are large binaries no session authored |
| `.venv`, `venv` | Virtualenvs: mostly binaries, no session-authored content |
| `__pycache__`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `.tox` | Tool caches |
| `node_modules`, `dist`, `site` | Vendored dependencies and generated output |

Pruning applies during walks only: a touched path named at the top level is never pruned by name (though it is still skipped when it resolves under an excluded root), and the pre-commit hook scans staged blobs whatever directory name they sit in.

Budget semantics: inside this scoped submission scan the walk budget counts only the entries the run's write window can inspect, so a repository tree of pre-session files no longer exhausts it with nothing left to inspect. A separate absolute enumeration cap (65536 entries) bounds the raw walk, and reaching either limit still reports `scan-incomplete`, which blocks. A scan invoked without the scope keeps the count-every-entry budget.

Accepted residual: content under a transient root is not inspected at submission (the report text channel is likewise unscanned), so a secret can pass a submission by living in a scratchpad or temp file. The pre-commit hook is untouched and still scans every staged blob, so a staged secret cannot ship through it.

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
`git clean`, and recursive `rm` outside the clean-tree and session-scratchpad
carve-outs) are not
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

### Wrapper and flag recognition (v0.6.0)

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

Probed egress-capable clients without destination modeling — `ftp`,
`telnet`, `httpie`, `wget2`, `aria2c`, `lftp` — ship
on the default `[tools.bash]` denylist until they are modeled; removing an
entry re-enables that client as an ordinary denylist edit. HTTPie's `http`
left that denylist in v0.7.0 under the restricted grammar described in
[Network-client egress modeling](#network-client-egress-modeling-v070), as
`curl` and `wget` did under the destination-aware egress rule; its `https`
entry point left it in v0.7.1 under the same grammar with the https scheme
forced onto a schemeless operand. `ssh` left that
denylist in v0.7.1 under the closed destination grammar: only a single
standalone `ssh` invocation with a positive destination is admitted — an IP
literal, or a hostname the system resolver (which honors `/etc/hosts`) answers
with exactly one non-local address — classified `network.remote-exec` with the
exact granted destination attached to the approval — while every other
spelling, and any hostname whose resolution is ambiguous, local, or failed,
hard-denies with no approval path; the admitted address plus a fixed host-key
identity are pinned into the spawned argv at every execution transport, and for
a hostname destination the pinned operand is the resolved address while the
host-key identity stays bound to the granted hostname. `scp` and `rsync` left
that denylist in v0.7.1 under the closed bounded transfer grammar: only a
single standalone transfer of exactly two operands with exactly one
`user@host:path` remote operand is admitted, classified `network.remote-exec`
with the exact granted destination attached to the approval, while every other
remote-bearing spelling hard-denies with no approval path (the bounded-transfer
clauses in the retained-denials passage below).

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

The session scratchpad is a second, independent carve-out (v0.7.8): a
recursive `rm` resolves run-class when EVERY expanded operand resolves by
canonical path strictly inside a session scratchpad root, because that content
is throwaway, session-owned data. The scratchpad root itself is not a deletable
operand, and a symlink that resolves outside the root is refused — the
containment test is the resolved path, never the spelling. The scratchpad
family skips the git-tree analysis (clean-tree state, metadata overlap,
repository recovery) but never the footprint analysis: the whole call's
destruction surface must still be establishable, so a statement whose writes
cannot be modeled (`eval`, an unresolvable operand) keeps the hard-guard
denial. Sensitive-file patterns and parent ceilings still deny through the
path layer, which gates every operand independently of this admission.

Its hard-guard denial names the original
option or cluster, such as `-r`, `-rf`, or `--recursive`, and recommends
`rm -- 'file'`, then `rmdir -- 'dir'`. Rejected path-glob diagnostics identify the
candidate token and command segment and explain that they cannot be safely
inspected. Feedback previews are bounded, registered secrets are redacted before
escaping or truncation, and control characters and markup delimiters are escaped.
These messages explain existing denials; they do not grant authority.

### Safe-pattern admissions (v0.7.0)

Two previously blanket-denied shell shapes are admitted with guards:

- **Quoted or escaped `=` marker.** A word whose leading `=` is neutralized by
  quoting or escaping and which proves exactly one literal argument is
  admitted as written: `echo '=ls'` and `echo \=ls` pass, and the analysis
  emits the same source word the shell executes. The proof rejects any word
  that can still expand past one argument — an unquoted glob metacharacter
  (`echo '='*`) stays denied — and an unquoted leading `=` keeps its live zsh
  equals-expansion denial whose remediation says "quote the marker" (an
  unquoted `echo ===X===` is never admitted). Zsh equals expansion at the
  command position is unchanged.
- **Same-invocation `$?` probe.** Exactly one shape is admitted: a plain
  simple command, a `;` separator, then `echo $?` or `printf $?` as the
  entire last statement (`ls; echo $?`). The probe reports the completed
  command's status, and the analysis keeps the `$?` source token — the shell
  substitutes the number — so the analyzed argv never diverges from the
  executed command. A pipeline, conditional, assignment-only statement,
  redirect, wrapper (`env ls; echo $?`), or any statement after the probe
  keeps the ordinary expansion denial, and the admission never suppresses a
  sibling denial of the preceding command.

KEEP-DENIED unchanged: command substitution (except the existing sed
recognizer), loops and `if` statements, general variable and arithmetic
expansion, and lowercase assignments. Dangerous-git and hard-guard
classification is untouched, and a probe cannot launder a dangerous command
(`git reset --hard; echo $?` keeps its hard-guard denial).

### Network-client egress modeling (v0.7.0)

HTTPie's request client `http` leaves the default denylist under a restricted,
positional-only grammar. The admitted forms are exactly:

- `http URL` — one literal `http://` or `https://` URL.
- `http METHOD URL` — `METHOD` is one of `GET`, `POST`, `PUT`, `PATCH`,
  `DELETE`, `HEAD`, `OPTIONS`.
- `http [METHOD] URL ITEM...` — request items that carry literal data only:
  `name=value` (data), `name:=value` (raw JSON), `name==value` (query
  parameter; it must be quoted, because zsh's magic-equals expansion would
  otherwise split the word and the shell layer denies the unquoted spelling),
  and `name:value` (header).

A literal loopback destination resolves run-class. A non-loopback destination,
and any destination the recognizer cannot parse — a schemeless operand such as
`localhost:8080`, a bare name, an unparseable URL — resolves ask-class as
`network.egress`, the same grantable operation as a non-loopback `curl` call,
and so does a loopback invocation whose effective environment carries a
proxy-family variable set to a non-loopback proxy. Everything else is a hard
denial with no operation ID and no approval path: any option (`--follow`,
`--proxy`, `-o FILE`, `--session`, `--download`), a request item that reads a
local file (`field@upload.txt`, `@body.json`, `payload:=@data.json`), a scheme
other than `http` or `https`, a missing destination, a stray operand, and an
invocation whose own environment sets `HTTPIE_CONFIG_DIR` — an uninspectable
startup config whose `default_options` can add options the grammar never saw
(the `XDG_CONFIG_HOME` and `HOME` spellings of that same chain are already
denied as dangerous environment assignments). HTTPie follows no redirects by
default and `--follow` is not admitted, so a literal loopback destination is
enough for run-class. As with every modeled client, a call's local reads and
writes are collected and gated first: a redirect target or an explicit output
file outside policy is denied by the path layer before any egress authority is
considered.

`httpie` and `wget2` stay on the default denylist:

- **`httpie`.** Admission is gated on a test confirming that the INSTALLED
  entry point carries HTTPie's request grammar. That confirmation does not
  hold: HTTPie 3.x ships its plugin/CLI manager (`httpie plugins install ...`)
  under this name, and the request client is `http`. The gate fails closed, so
  every `httpie` spelling stays denied.
- **`https`.** HTTPie's `https` entry point is the same request client as
  `http` and left the default denylist in v0.7.1 under the same restricted,
  positional-only grammar: `https [METHOD] URL ITEM...` with the same closed
  method set, the same literal request items, and no option admitted. The one
  difference is the scheme. HTTPie forces `https://` onto a SCHEMELESS operand
  for this program name — its parser reads the program name and prefixes the
  scheme before `--default-scheme` is ever consulted, and no option is admitted
  anyway — so a schemeless operand is a bounded destination the model can name:
  a literal loopback one (`https localhost:8443/status`) resolves run-class and
  a non-loopback one resolves ask-class as `network.egress`. An explicit scheme
  is never rewritten, so `https http://host/` keeps its `http://` and really
  does issue a plain-HTTP request: it hard-denies as a cross-scheme spelling,
  with no operation ID and no approval path, and so does every other non-https
  scheme (`ftp://`, `file://`). Every other rule of the `http` grammar applies
  unchanged: an option, a request item that reads a local file, a missing
  destination, a stray operand, and an invocation whose own environment sets
  `HTTPIE_CONFIG_DIR` all hard-deny. HTTPie's localhost shorthand
  (`https :8443/status`, i.e. `localhost:8443`) is not a literal URL the
  recognizer parses, so it resolves ask-class rather than riding a loopback
  auto-allow.
- **`wget2`.** Loopback auto-allow requires a documented, corpus-tested effective redirect-disable, and no effective redirect-disable is established for wget2 in this release, so the denylist entry stays and every `wget2` spelling resolves denied — loopback `http(s)://` operands included. The remaining gates are pinned the same way rather than inherited from wget: recursive retrieval (`-r`/`--recursive`, `-p`, `-m`), input and config files (`-i FILE`/`--input-file`, `--config=FILE`, `--post-file`, `--body-file`), and an `-O` output path outside policy are all denied. wget2's user manual documents `--max-redirect=number` (default 20) with no documented disable value, and its `-e`/`--execute` option runs a startup command straight from the command line while `--local-plugin`/`--plugin` load code — effects a destination model would have to bound before admission. No wget2 binary is installed in the environment this release was verified in, so none of these flags were checked against a running client; the spellings come from the manual and the denial corpus pins the outcome, not flag parsing.

#### FTP PASV residual

The wget FTP PASV residual is not inherited by wget2. wget's default passive
FTP honors the address in the server's PASV response and has no
`--ftp-skip-pasv-ip` equivalent, so that destination-choosing behavior is
DISCLOSED for the modeled `wget` rather than gated; a future wget2 admission
must gate or disclose it again instead of inheriting it silently. Until then a `wget2 ftp://...` operand is denied with the retained entry, and the corpus row that pins it records that denial rather than a loopback auto-allow.

The `ftp`, `telnet`, `aria2c`, `lftp`, and `watch` denials
stand: destination parsing cannot bound remote command execution, secondary
connections, or scripting, and `watch` is a command repeater rather than an
egress client. `ssh` is the exception since v0.7.1: its closed grammar admits
only a positive destination — an IP literal, or a hostname the system resolver
(which honors `/etc/hosts`) answers uniquely and non-locally — and the approved
address plus a fixed `HostKeyAlias`/`UserKnownHostsFile` host-key identity are
pinned into the actual argv every spawn point receives; for a hostname
destination the resolved address rides as the destination operand while the
`HostKeyAlias` pin stays bound to the granted hostname, and an ambiguous, local,
or failed hostname resolution stays denied with no approval path. `scp` and
`rsync` are the same exception since v0.7.1 under the closed bounded transfer
grammar: only a single standalone transfer of exactly two operands, exactly
one of them a `user@host:path` remote operand, is admitted — the host an IP
literal or a uniquely and non-locally resolved hostname, the path inside a
closed character set with no `..` segment — with only safe option clusters
(and the `-P` port) admitted, classified `network.remote-exec` with the exact
granted destination attached to the approval. Remote-to-remote copies,
user-supplied transports (`-e`/`--rsh`, `-S`), symlink-following (`-L`),
destination-mutating options (`--delete`, `--remove-source-files`, `-P`,
`--inplace`, the backup/partial/temp-dir families), the legacy scp protocol
and server-chosen names (`-O`, `-T`), the rsync daemon form (`host::module`),
a remote spec with no user, and any local, ambiguous, or failed destination
all hard-deny with no approval path; a transfer with no remote operand is an
ordinary local copy governed by the path layer. The admitted transfer pins the
same fixed `HostKeyAlias`/`UserKnownHostsFile` host-key identity and the
resolved address into the actual argv every spawn point receives — for a
hostname destination the resolved address rides as the remote operand's host
while the `HostKeyAlias` pin stays bound to the granted hostname — and a
changed resolution at spawn time never reaches the spawn. A locality-only
attestation entry (network-namespace locality without the SSH trust context)
does not admit transfers: it hard-denies `scp`/`rsync` even on a client that
admits `ssh` execution. In particular
`ssh`, `scp`, and `rsync` are never classified as `network.egress`, so the
separate `network.remote-exec` path can be modeled on top of them without this
rule shadowing it.

### Retained shell-policy limitations

Some over-denials and inspection gaps are retained, documented trade-offs in
this release; each names its fast-follow pointer:

- **Trailing-`**` admission.** A path component whose trailing star run is
  exactly two stars — `src/**`, never the recursive `**/` form — is admitted
  for the whitelisted read commands (`cat` and `wc` with no options) through
  the same literal argv plan a single `*` binds: the plan enumerates the real
  directory, the words the OS receives are the plan's words, and the shell
  never re-expands the original text. The admission is limited to that one
  spelling and that one transport, and it requires an executor whose pinned
  contract proves the run is single-level, which is why it does not reach
  zsh: zsh's recursive search is not governed by any option the contract
  holds off, and an unknown executor has no contract at all. The remaining
  over-denial is the same spelling everywhere else — a run of three or more
  stars, a run that does not end the component (`a**b`), a wildcard in a
  middle component, and every command outside the whitelist still fail closed
  on the path glob, and the retained per-call budgets, no-match, symlink, and
  containment checks are unchanged.
- **Extglob environment residual.** The static model inspects plain POSIX
  globbing only, but the executor shell-option contract now pins the executing
  shell itself: every spawned shell runs with the expansion-relevant option
  families held off, startup-file environment variables scrubbed or redirected
  away (so a `~/.zshenv` cannot switch extendedglob on), and the locale
  pinned, while an unknown shell, an option-changing invocation, or a nested
  shell re-entry with an expansion-bearing payload fails closed. Extended-glob
  syntax without glob metacharacters (`^foo`, `foo#`, `a~b`) is detected
  quote-aware and stays literal under the pinned contract. The residual is
  root-owned: system-wide startup files (`/etc/zshenv`, `/etc/profile`) and the
  shell binary itself remain a root-owned trust base, exactly like the
  interpreter the model runs on.
- **Quoted-variable conservative denial.** Static expansion intentionally
  resolves only plain unquoted words in provably read-only argument positions
  of whitelisted commands, bound only by literal assignments proven unchanged
  through the invocation. One quoted spelling joins them: a whole
  double-quoted `"$NAME"` or `"${NAME}"` word resolves by that same static
  expansion — as exactly one literal argv element bound to the actual spawn —
  but only when a literal assignment to that name precedes the command
  unchanged in the same invocation. Every other quoted variable is still
  conservatively denied rather than resolved: an empty value, an ambient
  (unassigned) name, a prefixed assignment value, a modifier form such as
  `${NAME:-default}`, and a concatenated or nested word such as `"pre$NAME"`
  or `"$NAME$OTHER"`.
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
