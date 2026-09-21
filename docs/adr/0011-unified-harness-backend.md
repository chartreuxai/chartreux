# 0011 Unified Harness Backend

## Decision

The app server accesses session implementations through two interfaces in
`chartreux.app_server._session_backend_port`.

`SessionBackendHost` owns the process-level session catalogue and lifecycle. It
starts and resumes sessions, selects the latest resumable session, lists and
reads stored sessions, creates forks, tracks live backends, and shuts them down
with the process. Fork belongs to the Host because it creates and registers a
new session identity, even though its implementation reads a source session.

The current standalone app server composes its Python `SessionBackendHost`
implementation through `HarnessProcess.create_session_backend_host`, which
calls `create_session_backend_host_impl`. The wire protocol does not expose
backend selection, and session IDs resolve in the current local store.

`SessionBackend` represents one bound live session. It owns public reads,
atomic event subscriptions, typed runtime configuration mutations, turn
control, context injection, callbacks, compaction, and runtime shutdown. Every
operation that reads or mutates live session state goes through the bound
backend or a narrow backend capability interface.

The app server owns transport, request validation, and application-specific
Host APIs. A backend owns its complete session state and execution lifecycle.
Neither side reaches into the other's private implementation state.

Backend failures cross the port as `SessionBackendError` with a stable
`ProtocolErrorCode`, message, and optional data. The transport layer maps that
error to its wire response. Adapters translate implementation-specific errors
at their boundary.

Subscriptions atomically return a current snapshot, its event watermark, and a
live stream. Events after that watermark are strictly ordered: duplicates are
ignored and a gap fails with `stale_cursor`. The port retains no historical
events. A restored backend may begin a new event sequence, so its subscription
snapshot replaces every earlier watermark and projection. A standalone read
followed by a subscription is not gap-free; callers use the snapshot returned
by `subscribe` when they need live updates.

Runtime shutdown and transport detachment are distinct. `shutdown` releases a
bound runtime and its owned resources. Disconnecting a client only detaches its
transport and subscriptions.

Backend-specific capabilities do not belong in the common session interfaces.
They use narrow capability interfaces so an adapter can report them as
unsupported without requiring another adapter to emulate them.

## Rationale

Process and session lifetimes are different: one process locates and creates
many sessions, while each live mutation and event stream has one session owner.
The split prevents app-server handlers, storage readers, and execution runtimes
from becoming competing sources of session state.

## Agent Guidance

- Add session creation, lookup, listing, or forking to `SessionBackendHost`.
- Add common live-session reads or mutations to `SessionBackend` with typed
  parameters and results.
- Construct `AppServer` with an explicit Host factory. Do not add a legacy
  default inside `AppServer`.
- Put backend-specific behavior behind a narrow capability interface.
- Translate adapter errors to `SessionBackendError` at the port boundary.
- Keep snapshot creation and event subscription atomic.
- Keep Host composition explicit and Python-only; do not add an alternate backend
  or implicit backend selector.
- Acquire the session lease before reading or mutating the local session store.

## Flag To User When

- A live-session read or mutation bypasses its backend.
- A handler reads backend-owned session storage directly.
- Two components can mutate or persist the same session state.
- Detach, shutdown, and interruption are treated as the same action.
- A backend-specific capability is being added to the common contract.
- A new delivery path needs a second session backend or a cross-store migration.
