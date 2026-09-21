# ACP

Chartreux provides `chartreux-acp` for clients implementing the [Agent Client
Protocol](https://agentclientprotocol.com/overview/clients). It is a local stdio
server: install Chartreux, configure the selected provider credential, then have
your ACP client launch the command.

## Client configuration

No launch argument is required:

```json
{
  "command": "chartreux-acp",
  "args": []
}
```

The client and server exchange ACP messages on standard input and output. Do not
write ordinary output to the server's stdout. Environment variables may be
provided by the client process or loaded from `$CHARTREUX_HOME/.env`; see the
[configuration reference](../reference/configuration.md#environment-variables-and-env).

For development from a checkout, install the scripts with:

```console
uv tool install --editable .
```

Ensure `uv tool dir --bin` is on `PATH`. The installed package also supplies the
main `chartreux` and `chartreux-app-server` commands.

## Setup and diagnostics

Run the server directly to use stdio:

```console
chartreux-acp
```

The available flags are:

| Flag | Meaning |
| --- | --- |
| `-h`, `--help` | Show help. |
| `-v`, `--version` | Show the version. |
| `--setup` | Run API-key setup and exit instead of serving ACP. |

ACP proxy configuration uses the in-session `/proxy-setup` command or the
process environment. Changes to `$CHARTREUX_HOME/.env` take effect only when the
ACP process starts, so restart the ACP process (or have the client start a new
one) after changing it. See [networking](networking.md).
