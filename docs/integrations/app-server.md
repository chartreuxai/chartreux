# App server

The app server is Chartreux's serialized harness boundary. It owns live
sessions, runtime configuration, tools, persistence access, and cleanup, while
delivery surfaces such as the terminal client, ACP, and programmatic mode act as
clients. For the architectural decision, see [ADR 0009](../adr/0009-app-server-boundary.md).

## Connecting

Run `chartreux-app-server` with no arguments. It speaks newline-delimited
JSON-RPC on standard input and output; stdout is reserved for protocol messages.
A client initializes the connection, sends `initialized`, then starts, resumes,
or continues a session. The protocol also has an in-process serialized transport;
there is no HTTP listener or multi-client network service exposed by this
entrypoint.

External Python clients should use the public `chartreux.app_server` package,
not private server modules. Its exported surface is:

- `AppServerHost` for passive pre-session operations and opening sessions;
- `AppServerSession` for an attached session, turns, resources, and live events;
- `ClientToolHandler` for client-hosted operations explicitly advertised by a
  client;
- `AppServerConnectionClosed` for attached-session connection closure; and
- `SessionExitSummary` for shutdown presentation.

## RPC surface

`chartreux.app_server.protocol.SERVER_METHODS` is the authoritative current RPC
method list, including typed parameter and result models. It covers:

- initialization, identity, runtime/configuration, policy roots, diagnostics,
  statistics, and event reads;
- session creation, continuation, resume, reading, history, fork, rename,
  compact, stop, delete, rewind, settings, shell commands, and turns;
- queued-turn enqueue/read/remove/replace/resume operations;
- agent transcripts, skills, tools, scheduled loops, reviews, and workspace
  trust and Git worktree operations; and
- MCP catalog read/add/remove/toggle/refresh/login/logout operations.

The server emits typed live notifications and can send typed callbacks or
advertised `clientTool/*` requests to a capable client. It does not offer a
generic command-execution RPC: clients map their own presentation commands to
specific typed methods.

## Limits

One app-server instance owns one attached root runtime and currently does not
model several simultaneous attached observers of that runtime. `events/read`
returns an empty batch in v0.1; clients consume live notifications and recover
state with a session read when needed. Restarting a stdio server ends its live
runtime and discards in-flight work and queued turns; persisted sessions can be
resumed normally. Public protocol views are projections, not the private session
format, and do not expose credentials or internal runtime objects.
