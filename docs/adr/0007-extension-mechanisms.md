# 0007 Extension Mechanisms

## Decision

Chartreux extends through explicit local mechanisms: agent profiles and subagents,
Markdown skills, executable hooks, directly configured MCP servers (including
OAuth), custom tools, provider configuration, and config layers. The built-in POSIX
`bash` tool is a finite captured shell surface: each model-facing call starts a
fresh shell, closes stdin with EOF, enforces a timeout, and returns separate local
stdout and stderr. It retains no process handle across calls. Managed background
sessions, continued stdin, polling, cursor-based output, and managed shell storage
APIs are intentionally removed; existing stored session/output records are left
untouched. User-authored `!` commands remain a separate manual
shell path. ACP client-terminal delegation remains an explicit client capability; its
client-produced output may be merged and is not promised to have the local shell's
separate streams.

Extensions should be discoverable, filterable, typed where possible, and isolated
from core startup and control flow unless actively configured. Local project and
user extensions remain supported; built-in names are not silently overridden.

For attached sessions, the app server owns extension discovery results, lifecycle,
authentication state, process cleanup, and public projections. Clients use typed
agent, skill, MCP, and tool resource methods; they do not receive managers or
registries. Subagents are server-owned child sessions with independent IDs and
public projections; the client never constructs or stores a child `AgentLoop`.

The standalone harness does not expose account, cloud, registry, plugin-package,
or hosted-connector APIs. Those removed public methods are not compatibility
shims: the protocol reports an unknown method, and obsolete explicit configuration
keys fail with source-aware diagnostics rather than being silently migrated.

MCP's canonical transports are `streamable-http` and `stdio`; the local `http`
alias is rejected. HTTP authentication is nested and supports static or OAuth
forms. MCP sampling is unsupported and fails closed; reload or replacement retires
its prior authority and connections. Stdio connections are persistent and
serialized only within a session; ambiguous operations are never automatically
replayed, and cancellation cannot undo remote side effects.

## Rationale

Explicit local mechanisms preserve useful customization without requiring a
service account, hosted catalog, or product-specific extension package. Keeping
ownership at the app-server boundary prevents delivery surfaces from developing
parallel discovery, authentication, or lifecycle implementations.

## Agent Guidance

- Prefer an existing local mechanism before adding a new extension path.
- Keep discovery deterministic and cheap; defer expensive work until needed.
- Bound hooks and external processes with timeouts and typed invocation/response models.
- Preserve direct MCP descriptor, OAuth, cancellation, and revision-safe save behavior.
  Keep private descriptor contexts session-only, all stdio cache use non-persistent,
  and anonymous HTTP persistence separate.
- Keep the model-facing shell finite and fresh per call; preserve manual `!` and
  explicit ACP client-terminal delegation without claiming managed continuation or
  split streams for client-produced output.
- Return canonical public resource views after mutations and refresh client state
  through app-server notifications.
- Do not add compatibility facades for removed account, registry, plugin, cloud,
  or hosted-connector operations.

## Flag To User When

- A feature adds a new extension path instead of using local agents, skills, hooks,
  MCP, custom tools, providers, or config.
- Extension discovery would run expensive work during startup.
- Local project behavior could override built-ins without an explicit rule.
- A delivery surface needs separate extension discovery or lifecycle logic.
- A change would silently accept an obsolete explicit config key or emulate a removed
  public API instead of returning its existing diagnostic or unknown-method error.
- A change reintroduces managed shell sessions, cross-call process handles, or
  model-facing polling/input/output controls without a separately approved decision.
