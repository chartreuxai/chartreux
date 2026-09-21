# 0005 Layered Configuration

## Decision

Configuration is layered, validated as a coherent snapshot, and model-driven.
`ChartreuxConfigSchema` is the canonical effective server schema. Every field has an
explicit merge strategy, and external data is parsed through Pydantic rather
than ad-hoc dictionary walks. The approved standalone policy supersedes previous persistent defaults and model pinning; explicit layer and session rules remain authoritative.

For an attached session, the app server owns the `ConfigOrchestrator`, effective
configuration, persistence target, reloads, and config-derived runtime state.
Textual, ACP, and programmatic clients use typed app-server resources. They do
not read or edit `config.toml`, receive the orchestrator, or mutate a live config
object.

The selected `SessionBackend` is the application boundary for changes to that
live runtime. Agent switches, session-limit updates, config writes, and reloads
use distinct typed backend methods after Host-side validation and persistence;
resource handlers do not reach into an implementation-specific runtime object.

## Current layer stack

The effective order is:

1. `DefaultConfigLayer`, materialized from `ChartreuxConfigSchema` defaults;
2. the user TOML layer (`~/.chartreux/config.toml`) when the `user` source is enabled;
3. the trusted project TOML layer (`.chartreux/config.toml`) when the `project` source is enabled;
4. `CHARTREUX_*` environment values;
5. the active `AgentProfileLayer`; and
6. session/runtime overrides.

The agent-profile slot is installed empty and filled by the agent lifecycle. The
runtime override layer is the default destination for ordinary live edits. Loading
or inspecting a persistence layer is pure: it uses an independent snapshot and does
not write, publish, or cache a session change. Historical config migration is not
part of loading; obsolete keys fail with source-aware validation.

MCP configuration has one canonical local spelling: `streamable-http` for remote
HTTP and `stdio` for local process transport. The retired `http` transport alias
and legacy top-level HTTP auth keys are rejected rather than migrated. HTTP
credentials are nested in the server's `auth` block, which is either static or
OAuth. An OAuth identity conflict cannot consume or relabel an existing grant:
use a separate alias, or make an explicit logout decision before deliberate alias
reuse.

MCP descriptor caching is non-authoritative. Anonymous HTTP declarations may use
persistent cache records; private contexts are session-only, and every stdio
context is session-only with no persistence because inherited environment and
working-directory state may affect credentials and configuration. Cache identities
must not publish or derive from secrets.

The user and project TOML layers are installed together so a trusted project
config inherits unspecified values from the user config. Per-field merge
strategies decide how overlapping fields combine, with the project layer taking
priority. An untrusted or absent project layer is skipped while the user layer
still contributes.

Catalog definitions (`models` and `providers`) come from the shipped catalog plus the
user-owned `models.toml` overlay, not a user or project `config.toml` layer.
Project, environment, profile, and session sources may select aliases and set
execution behavior, but cannot define catalogs. A wrong-scope definition fails
before shadowing or persistence. `authorized_roots_by_project` is likewise
accepted only from an installed user layer; an ordinary config write cannot grant
or change roots.

## Explicit persistence

A persistent save is an explicit single-target operation. It must name exactly one
`user` or `project` target and provide that target's expected source revision.
Mixed targets, implicit persistent routing, and stale revisions are rejected. A
save reports persistence independently from runtime application:

- persistence is `not_saved`, `saved`, or `durability_uncertain`; and
- application is `unchanged`, `applied`, or `failed`.

This exposes saved-but-not-applied and uncertain-durability outcomes without
pretending that the effective session changed. A persistent save never mirrors its
values into the session override layer. Session-only edits explicitly target the
runtime override and are not durable.

TOML persistence uses temporary-file, `fsync`, and atomic replacement semantics.
The operation is optimistic: the expected revision detects external changes before
the replacement. The general orchestrator does not pretend that several persistent
targets form one transaction.

An explicit model selection is session state until it is recorded in session
metadata by the subsequent user turn. Default resolution creates no model pin;
once a session is assigned, its committed base model and concrete deployment
identity govern subsequent turns and resume. `/clear` returns to normal
current-configuration resolution.

## App-server config boundary

`ConfigView` is a redacted public projection, not a second writable config
schema. It contains only values a client must render or apply and never contains
resolved API keys, tokens, or arbitrary environment
values. Clients must not infer writable paths from its shape.

The current resource methods are defined by `chartreux.app_server.protocol`:

- `config/read` returns the effective redacted view;
- `config/reload` re-reads configured sources and optionally rebuilds runtime state;
- `config/write` applies a session patch or one explicit user/project save, with
  source revision and separate persistence/application results;
- `config/proxy/read` and `config/proxy/write` manage the supported global proxy and
  certificate `.env` entries; and
- `config/schema` exposes the live schema used by ACP settings clients.

The proxy resource is deliberately separate from the TOML orchestrator.
`config/schema` is configuration-form metadata; it is not a list of valid
`config/write` paths and is not the public app-server protocol schema.

For a write, the server validates the prospective merged config before changing
runtime state. A session patch applies only to the override layer. A persistent save
writes the selected layer once, then reports whether the new effective snapshot was
applied. Clients must use the returned authoritative state rather than assuming
persistence implies application.

## Client-local application

Persistence ownership and runtime application are separate:

- model, agent, permission, tool, MCP, hook, workspace, and session
  settings are applied by the server;
- committed theme, clipboard, and terminal-notification settings are applied from
  accepted server state; temporary presentation previews may remain local.

Before a session is attached, CLI and ACP launchers still load dotenv values,
create initial files, run onboarding, and read startup config for process-level
setup. This is bootstrap staging, not a second attached runtime. After
attachment, live config reads, writes, reloads, trust decisions, and derived
resource refreshes are server operations.

## Rationale

Chartreux must combine defaults, persisted preferences, trusted project policy,
environment values, session options, agents, tools, MCP, and other
extensions without making delivery surfaces understand persistence.
Schema-aware layering provides deterministic merge behavior and one validated
effective snapshot. App-server ownership prevents the UI, ACP, and runtime from
becoming competing sources of truth.

## Agent Guidance

- Add fields to the relevant Pydantic config model with explicit defaults,
  validation, and merge metadata.
- Preserve deterministic layer ordering and keep session overrides separate from
  persisted defaults; ordinary live edits target the override layer.
- Make persistent saves explicit: one `user` or `project` target, its expected
  revision, and separate persistence/application outcomes.
- Keep catalog definitions and `authorized_roots_by_project` restricted to the
  sources that own them; never use a generic config write to grant roots.
- Treat source loading and inspection as pure; migration and persistence are
  explicit operations.
- Mutate attached-session config through app-server resources or server-owned
  orchestrator calls, never from Textual.
- Return canonical server state after a mutation; clients replace their cache
  instead of optimistically merging arbitrary dictionaries.
- Keep redaction in the server projector. Public views expose only what the
  client needs.
- Keep config models strict and source-aware; obsolete configuration keys should fail
  validation rather than being silently migrated.
- Avoid loading optional integrations during startup unless active config
  requires them.

## Flag To User When

- A feature needs hidden global state instead of config or session state.
- A config value is parsed manually or persisted from more than one owner.
- A public write would weaken explicit target, source-revision conflict detection,
  separate persistence/application outcomes, or source ownership.
- A client needs a private config object, TOML path, secret, or orchestrator.
- A new config path would make startup slower for users who do not use the
  feature.
