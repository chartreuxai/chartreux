# Tools and safety

Chartreux exposes file, search, shell, delegation, web, image, and session-management tools. Runtime policy still applies when a tool is visible.

## Built-in tools

The built-ins are `ask_user_question`, `bash`, `check_agents`, `edit`, `get_agent_result`, `grep`, `read_file`, `read_image`, `release_agent`, `skill`, `task`, `todo`, `wait_for_agent`, `web_fetch`, `web_search`, and `write_file`. MCP servers can add tools; their names use `<server>_<tool>`.

`web_search` supports `auto`, `mistral`, `exa`, `brave`, and `duckduckgo`. The runtime default remains `auto`, which selects Mistral only; it does not fall back to another provider. Configure the provider, optional credential environment-variable name, base URL, timeout, result limit, and Mistral search model in `[tools.web_search]`. An unset or blank `base_url` uses the selected provider's default endpoint; the blank value saved by the editor is not treated as a URL override. For a keyed provider, Chartreux selects exactly one credential-variable name: `api_key_env_var` when configured, otherwise the provider default (`EXA_API_KEY` or `BRAVE_SEARCH_API_KEY`); `auto` and `mistral` otherwise use the configured Mistral provider's credential variable or `MISTRAL_API_KEY`. If that selected variable is unavailable, web search reports the missing key. DuckDuckGo needs no key.

An unset or blank Exa or Brave endpoint resolves to that provider's default
absolute endpoint. Transport failures use safe, categorized messages such as
an invalid endpoint URL, DNS or TLS failure, proxy/connect failure, or a
transfer/protocol error; untrusted exception text is not shown.

In standalone Settings, the provider list shows Mistral once. The saved `auto`
value remains a supported Mistral alias for existing configuration, not a
second provider choice. First-run setup preserves any already-ready search
configuration and skips that editor. If it must open the editor, its choices
are Exa, Brave, and DuckDuckGo; it does not offer `auto` or Mistral as fallback
choices. Readiness comes from resolved settings and credential lookup, including
a custom configured credential-variable name, rather than catalog presence.

Use Settings > Web search or `/web-search` to edit these values. `Save search settings` saves the settings draft; `Save API key` is a separate credential action, so a saved key remains saved if you later discard the settings draft. The key action reports whether the key was saved or is available only for the current session. Switching providers resets custom credential-variable and endpoint overrides to the new provider defaults and clears an unsaved key. The `Configured; connection not verified` readiness label means the selected configuration and any required key are available; Chartreux does not test live connectivity or switch providers automatically. See the [Configuration guide](configuration.md) and [command reference](../reference/commands.md).

After default presets, first-run onboarding preserves ready automatic Mistral
search or an already-ready explicit provider and skips the editor. If neither
is ready, the editor offers Exa, Brave, and DuckDuckGo only. **Skip for now**
keeps the existing web-search settings and tool enablement unchanged; unsaved
edits must be discarded before leaving, and a key already saved with **Save API
key** remains saved. The **Finish setup** and **Save and finish** actions
require valid search settings and any required credential; **Skip for now**
can complete setup without them. Readiness does not include a live provider
request.

Search-settings feedback reports persistence and runtime application separately. If settings were saved but not applied, use **Retry runtime reload**; if runtime applied but the screen did not refresh, use **Retry UI refresh**. These retries apply already saved settings without repeating the write. If a revision conflict occurs or the saved state could not be read, refresh before editing again; **Refresh (discard draft)** discards unsaved search settings. The editor also identifies fields controlled by a higher-priority configuration layer.

```toml
[tools.web_search]
provider = "brave"
api_key_env_var = "BRAVE_SEARCH_API_KEY"
timeout = 30
max_results = 5
```

`read_image` is available only when the active model deployment supports images. It is subject to the ordinary file and sensitive-path policy. See [Models](models.md) for deployment capabilities.

## Filtering and permissions

Use `enabled_tools` to narrow the available set and `disabled_tools` to remove from that result. Patterns can be exact names, globs, or case-insensitive full-match regular expressions prefixed with `re:`.

```toml
enabled_tools = ["read_file", "grep", "web_*"]
disabled_tools = ["web_fetch"]

[tools.bash]
permission = "ask"
```

The current per-tool permission values are `always`, `ask`, and `never`:

- `always` permits the tool subject to its other safety checks.
- `ask` does not display an approval prompt; the tool executes automatically, subject to permission policy and runtime safety checks.
- `never` prevents the tool from running.

These are tool permissions, not application-wide operating modes. The former `plan`, `ask`, `accept-edits`, and `auto-approve` mode set is not part of the current configuration. `ask` does not request interactive approval: execution remains governed by permission policy, denylists, sensitive-file and workspace checks, and other runtime safeguards. Child agents cannot receive authority beyond their parent. MCP tools use the same filtering and permission mechanisms; see [MCP](mcp.md).

## Trusted folders

Chartreux asks before trusting a new workspace when it detects material that can influence agent behavior. The prompt identifies `AGENTS.md`, `.chartreux/` and `.agents/` configuration directories, and relevant repository-context files, including instruction files between the repository root and the current directory.

Trust can apply to the current folder, the repository root when offered, or the current session. Declining records the folder as untrusted. Persisted decisions are stored in `~/.chartreux/trusted_folders.toml`. Trusted project configuration, prompts, skills, hooks, and instructions are then eligible to load; untrusted project configuration is not. Passing `--add-dir` is an explicit trust grant for that session.

## Shell safety

`bash` runs a finite command in a fresh POSIX shell. Standard input is closed, a timeout is enforced, and stdout and stderr are returned separately. Shell state, process handles, continued stdin, polling, and cursor-based output do not persist between calls. Configure its `permission`, `max_output_bytes`, `default_timeout`, `denylist`, `denylist_standalone`, and `sensitive_patterns` under `[tools.bash]`; runtime path and sensitive-file protections still apply.

## Interactive questions

`ask_user_question` works only in an interactive UI. Each question requires two or more options; two to four is recommended, not a maximum. An `Other` free-text choice is added unless `hide_other` is true, and questions may permit multiple selections. Programmatic runs deny interactive callbacks rather than displaying them.

For reusable instructions that guide tool use, see [Instructions and skills](instructions-skills.md). For non-interactive workflows, see [Automation](automation.md).
