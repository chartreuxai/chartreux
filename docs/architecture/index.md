# Architecture

Chartreux is a local coding-agent harness with a reusable engine and several delivery surfaces. The design keeps model and tool decisions in the core, while terminal, protocol, filesystem, network, and provider concerns stay at the edges.

## Map

- **Core engine.** `chartreux.core` contains the agent loop, LLM backends, typed tools, configuration, sessions, and extension mechanisms. It does not depend on Textual, ACP, or the app server.
- **Delivery surfaces.** The Textual CLI, `chartreux-acp`, and programmatic clients all use the app server rather than constructing an agent loop themselves. ACP is a thin protocol adapter; the CLI owns terminal presentation.
- **App-server harness.** `chartreux.app_server` owns live sessions, turns, callbacks, resources, persistence access, and public projections. Its typed JSON-RPC boundary keeps delivery clients separate from private runtime state.
- **Event-driven turns.** The core loop streams typed asynchronous events for assistant output, reasoning, tool work, compaction, and lifecycle changes. The app server projects that stream into public client events, so every surface can observe the same turn.
- **Tools and safety.** Tools have typed arguments, results, configuration, and state. Runtime permission and safety policy covers side effects such as files, processes, network access, and MCP calls; client-hosted operations remain server-validated.
- **Configuration.** A validated effective configuration layers defaults, user TOML, trusted project TOML, `CHARTREUX_*` environment values, an active agent profile, and runtime overrides. The model catalog is separately owned in `models.toml`; see the [configuration reference](../reference/configuration.md).
- **Local sessions.** Conversations are durable local records. Session transcripts use `messages.jsonl` and metadata uses `meta.json`; public session history is a redacted projection, not the storage format.

For practical usage, start with [configuration](../guides/configuration.md), [tools and safety](../guides/tools-safety.md), [subagents](../guides/subagents.md), or the [app-server integration](../integrations/app-server.md).

## ADR reading guide

The ADRs record the constraints behind the map above. Read them in order for the broadest picture, or use this guide to jump to a decision.

- [0001 — Architecture principles](../adr/0001-architecture-principles.md): establishes pragmatic hexagonal boundaries and ownership of side effects.
- [0002 — Core engine and delivery surfaces](../adr/0002-core-engine-and-delivery-surfaces.md): separates the reusable core, the app-server harness, and client-specific adapters.
- [0003 — Event-driven agent loop](../adr/0003-event-driven-agent-loop.md): makes typed async events the contract between execution and delivery.
- [0004 — Typed permissioned tools](../adr/0004-typed-permissioned-tools.md): defines typed tools, permission enforcement, and public effect projection.
- [0005 — Layered configuration](../adr/0005-layered-configuration.md): defines validated configuration layers, source ownership, and explicit persistence.
- [0006 — Local sessions](../adr/0006-local-sessions.md): defines the local session format, public projections, compaction, and rewind behavior.
- [0007 — Extension mechanisms](../adr/0007-extension-mechanisms.md): chooses local agents, skills, hooks, MCP, tools, providers, and config as extension paths.
- [0008 — Feature instrumentation](../adr/0008-feature-instrumentation.md): prohibits product analytics and telemetry export while retaining local diagnostics.
- [0009 — App server as the harness boundary](../adr/0009-app-server-boundary.md): makes the serialized app server the sole runtime boundary for all clients.
- [0010 — Textual content rendering](../adr/0010-textual-content-rendering.md): standardizes safe, theme-aware Textual content rendering.
- [0011 — Unified harness backend](../adr/0011-unified-harness-backend.md): separates process-level session catalogue ownership from a bound session backend.
- [0012 — Slash commands while busy](../adr/0012-two-phase-slash-command-execution.md): distinguishes safe side-channel commands from idle-only commands.
- [0013 — Queue selection and edit mode](../adr/0013-queue-edit-mode.md): defines selection and editing of queued prompts in the CLI.
- [0014 — Backend contract compatibility](../adr/0014-backend-contract-compatibility.md): requires tolerant inference and MCP response parsing without reviving removed APIs.
- [0015 — Outbound TLS trust policy](../adr/0015-outbound-tls-trust-policy.md): applies configured certificate trust policy to Chartreux-owned TLS connections.
