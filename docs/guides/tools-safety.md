# Tools and safety

Chartreux exposes file, search, shell, delegation, web, image, and session-management tools. Runtime policy still applies when a tool is visible.

## Built-in tools

The built-ins are `ask_user_question`, `bash`, `check_agents`, `edit`, `get_agent_result`, `grep`, `read_file`, `read_image`, `release_agent`, `skill`, `task`, `todo`, `wait_for_agent`, `web_fetch`, `web_search`, and `write_file`. MCP servers can add tools; their names use `<server>_<tool>`.

`web_search` supports `auto`, `mistral`, `exa`, `brave`, and `duckduckgo`. Configure its provider, optional credential environment-variable name, timeout, and result limit in `[tools.web_search]`. For a keyed provider, Chartreux selects exactly one credential-variable name: `api_key_env_var` when configured, otherwise the provider default (`EXA_API_KEY` or `BRAVE_SEARCH_API_KEY`); `auto` and `mistral` otherwise use the configured Mistral provider's credential variable or `MISTRAL_API_KEY`. If that selected variable is unavailable, web search reports the missing key; it does not try other variables. DuckDuckGo needs no key.

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
- `ask` requests approval when the caller is interactive.
- `never` prevents the tool from running.

These are tool permissions, not application-wide operating modes. The former `plan`, `ask`, `accept-edits`, and `auto-approve` mode set is not part of the current configuration. In programmatic mode, tool calls are auto-approved, but path, sensitive-file, and other runtime policy checks remain in force. Child agents cannot receive authority beyond their parent. MCP tools use the same filtering and permission mechanisms; see [MCP](mcp.md).

## Trusted folders

Chartreux asks before trusting a new workspace when it detects material that can influence agent behavior. The prompt identifies `AGENTS.md`, `.chartreux/` and `.agents/` configuration directories, and relevant repository-context files, including instruction files between the repository root and the current directory.

Trust can apply to the current folder, the repository root when offered, or the current session. Declining records the folder as untrusted. Persisted decisions are stored in `~/.chartreux/trusted_folders.toml`. Trusted project configuration, prompts, skills, hooks, and instructions are then eligible to load; untrusted project configuration is not. Passing `--add-dir` is an explicit trust grant for that session.

## Shell safety

`bash` runs a finite command in a fresh POSIX shell. Standard input is closed, a timeout is enforced, and stdout and stderr are returned separately. Shell state, process handles, continued stdin, polling, and cursor-based output do not persist between calls. Configure its `permission`, `max_output_bytes`, `default_timeout`, `denylist`, `denylist_standalone`, and `sensitive_patterns` under `[tools.bash]`; runtime path and sensitive-file protections still apply.

## Interactive questions

`ask_user_question` works only in an interactive UI. Each question requires two or more options; two to four is recommended, not a maximum. An `Other` free-text choice is added unless `hide_other` is true, and questions may permit multiple selections. Programmatic runs deny interactive callbacks rather than displaying them.

For reusable instructions that guide tool use, see [Instructions and skills](instructions-skills.md). For non-interactive workflows, see [Automation](automation.md).
