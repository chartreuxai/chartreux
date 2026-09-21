# 0006 Local Sessions

## Decision

Sessions are durable local records of conversation state, metadata, tool availability, stats, and resumability data.

Compaction keeps the current session identity and transcript. The compacted
context is stored as an injected message marked with
`context_boundary = "compaction"`. Model requests include the current system
message, the latest marked compaction message, and everything written after it.
Public history still projects the complete transcript and renders each marked
message as a compaction checkpoint.

Session persistence uses the current split store: append-friendly `messages.jsonl`
for transcript entries and atomic `meta.json` for session metadata. It is local to
Chartreux and does not promise migration from historical Vibe transcript formats.
An explicitly selected in-place rewind is the exception: it rewrites the current
session transcript to an earlier boundary.

Private session storage and the public app-server projection are different
contracts. Only the server reads or writes session files. Clients receive a
lossy `PublicSessionState`, page public history through opaque cursors, and use
stable session, turn, entry, callback, effect, and child-session IDs. Public
events are not a persistence format.

The current local format restores completed transcript state, metadata,
statistics, and persisted child links. A reconnect to the same live harness can
recover its snapshot and open callbacks; a new process does not restore an
in-flight turn, open callback future, or live event sequence from JSONL. Do not
present live reconnect behavior as crash recovery.

Rewind has two explicit persistence modes:

- A forked rewind creates a replacement session derived from the selected history
  prefix and leaves the source session untouched.
- An explicitly selected in-place rewind keeps the current session identity and
  persistently truncates that session to the selected prefix. The discarded suffix
  is intentionally removed from durable session history and cannot be recovered
  by resuming that session.

Clear is likewise a replacement-session operation: it creates a new session and
leaves the original session untouched. In-place rewind is the sole destructive
mode described here and must be explicitly selected.

In both modes, the app-server response contains the authoritative public state
after the rewind. Clients replace their projection from that state instead of
editing the visible history locally. File restoration is an independent rewind
choice and does not determine the persistence mode.

## Rationale

Users rely on resume, rewind, titles, transcript inspection, and continuity
across runs. Forked rewind supports exploration without losing the original;
in-place rewind supports users who deliberately want to discard the abandoned
tail without creating another session. The split store is the current format, so
format changes require an explicit product decision rather than implicit migration.

## Agent Guidance

- Persist messages and metadata through the session layer, not directly from UI code.
- Route list, stored reads, resume, continue, and fork through the app-server
  session backend Host. Route live reads, rewind, clear, compact, and history
  mutations through the bound session backend.
- Keep session data serializable in the current split-store format.
- Do not add implicit migrations or compatibility readers for retired transcript
  formats.
- Do not store surface-only widget state in core session transcripts.
- Keep image/session attachment behavior explicit about what is persisted and what remains memory-only.
- Keep compaction in the current session and append its marked context message
  through the normal session logger.
- Treat context-clear session replacement as an explicit handoff: atomically
  adopt the returned session ID, public state, event watermark, and session-log
  summary.
- Treat every rewind result as an authoritative state replacement, even when
  the session ID does not change.
- For a forked rewind, preserve the source session and adopt the returned
  replacement session identity.
- For an in-place rewind, retain the current session identity and persist the
  truncated transcript, including an empty conversation when rewinding to the
  first user message.
- Never infer in-place rewind from a missing option. The destructive persistence
  mode must be selected explicitly; callers that do not expose a choice should
  preserve the source session.
- Treat clear as a replacement-session operation when it returns a new identity.
  The replacement may derive from an earlier history prefix, but the original
  stored session remains intact.
- Represent subagents as linked child sessions. Do not embed a live child
  runtime in a public parent model.

## Flag To User When

- A change requires a new session format or asks users to discard current sessions.
- Compaction changes the session identity, rewrites earlier transcript entries,
  or sends messages before the latest compaction boundary to the model.
- A rewind would destructively update a session without an explicit in-place
  choice.
- UI state is being added to core transcript data.
- Metadata updates are no longer atomic or ordinary message writes are no
  longer append-safe outside the explicit in-place rewind path.
- A client needs to read `messages.jsonl`, session metadata, or private loader
  APIs to render or control a session.
