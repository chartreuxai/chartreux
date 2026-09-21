# MCP

Model Context Protocol (MCP) servers extend Chartreux with externally provided tools. Configure servers in `config.toml`; their tools are named `<server_name>_<tool_name>` and use the normal tool filters and permissions described in [Tools and safety](tools-safety.md).

## Server configuration

Only two transport values are accepted: `streamable-http` for remote servers and `stdio` for local processes. The retired `http` alias is invalid. Startup and tool timeouts default to 10 and 60 seconds.

```toml
[[mcp_servers]]
name = "docs"
transport = "streamable-http"
url = "https://example.invalid/mcp"
startup_timeout_sec = 10
tool_timeout_sec = 60

[mcp_servers.auth]
type = "static"
headers = { "X-Client" = "chartreux" }
api_key_env = "DOCS_API_KEY"
api_key_header = "Authorization"
api_key_format = "Bearer {token}"

[[mcp_servers]]
name = "local-fetch"
transport = "stdio"
command = "uvx"
args = ["mcp-server-fetch"]
env = { "LOG_LEVEL" = "info" }
```

The HTTP `auth` block is nested. Static authentication may supply fixed headers and an environment variable whose token is formatted into a header. Do not put `headers` or `api_key_env` at the server top level. A stdio server uses `command`, optional `args`, and optional environment variables.

Set `disabled = true` to hide all tools from a server, or use `disabled_tools` with unprefixed server tool names to hide selected tools.

## Adding and authenticating servers

Add a remote server from the shell:

```bash
chartreux mcp add docs \
  --url https://example.invalid/mcp \
  --transport streamable-http \
  --api-key-env DOCS_API_KEY
```

Providing `--api-key-env` or `--header` selects static authentication. Without either, the command configures OAuth and starts its login flow unless `--no-login` is supplied. The interactive equivalent, `/mcp add`, is OAuth-only and also defaults to login.

MCP OAuth remains supported. Use `/mcp login <name>` to authenticate or retry authentication and `/mcp logout <name>` to revoke the stored OAuth credentials. OAuth grants are bound to the configured server identity; use a new alias for a different URL or client identity, or explicitly log out before deliberately reusing an alias. Removing a server removes its configuration; it is not an identity-migration mechanism.

For command options, use `chartreux mcp add --help` and see the [command reference](../reference/commands.md).

## Runtime behavior

HTTP calls are one-shot. Stdio connections are persistent and serialized for the current session. A transport failure is ambiguous: Chartreux does not automatically replay the operation, and cancellation cannot undo a remote effect that already happened. Anonymous HTTP descriptor information may be cached; authenticated HTTP and all stdio contexts are session-only.

MCP results preserve remote error status, structured content, and text. Chartreux does not implement MCP sampling: a server's sampling request fails closed with an unsupported error rather than invoking a model.
