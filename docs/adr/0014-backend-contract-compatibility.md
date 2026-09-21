# 0014 Backend Contract Compatibility

## Decision

Chartreux remains tolerant of version skew in the inference-provider and direct MCP
wire responses it can run against. Self-hosted providers, MCP servers, and the
Chartreux app server can be upgraded independently, so response parsing must not
assume one exact remote version.

Wire-contract changes are additive and tolerant by default:

- Parsing inference or MCP responses ignores unknown fields and tolerates missing
  fields.
- A newly used response field has a safe default or a capability/version gate; it
  is never a hard requirement.
- One malformed response item is skipped when the surrounding protocol permits
  per-item recovery, rather than failing the whole payload.
- The live app-server architecture remains the serialized, typed JSON-RPC boundary;
  compatibility handling must not create a delivery-surface runtime or bypass it.

Direct MCP result parsing preserves the remote `isError` status, structured
content, text, and explicit null/omitted argument distinction. Unsupported
content blocks receive bounded type-only omission notices. Discovery follows
`nextCursor` pages with finite limits and rejects repeated or invalid cursors.
These response tolerances do not extend to retired local configuration aliases:
`streamable-http` and `stdio` are canonical, while `http` and legacy top-level
HTTP auth keys fail source-aware validation without migration.
- Removed account, cloud, registry, plugin, and hosted-connector methods are not a
  compatibility surface: they return the generic unknown-method error. Obsolete
  explicit configuration keys produce source-aware diagnostics, not silent ignores
  or migrations.
- A change that cannot stay backward compatible is flagged in the PR under
  "Self-hosted compatibility" with the skew window and fallback, and needs
  sign-off. It is never landed silently.

## Rationale

Client, provider, MCP, and self-hosted backend versions drift across deployments.
Tolerant response parsing preserves useful inference and MCP behavior across that
skew while keeping the app-server lifecycle and public request boundary explicit.
Removed product APIs are intentionally not revived as shims, because pretending
that an unsupported service still exists would make configuration and failures
ambiguous.

## Agent Guidance

- Never add a required, no-default field to an inference or MCP response model.
- Skip bad items per-item where the response contract allows it; do not let one
  malformed item fail the whole payload.
- Keep parsing and any local descriptor cache tolerant of old and new response shapes.
  The cache is non-authoritative: anonymous HTTP may persist, private contexts are
  session-only, and all stdio contexts have no persistent cache.
- Never automatically replay an ambiguous MCP operation after transport failure.
  Cancellation may retire local transport state but cannot undo remote effects.
- Add parser tests for older-backend shapes with absent fields and for unknown fields.
- Keep app-server request models and method dispatch typed and serialized; do not
  weaken strict public request validation to preserve a removed product API.
- Document any unavoidable incompatibility with its skew window and fallback.

## Flag To User When

- A response field becomes required, or parsing fails when it is absent.
- A new behavior needs a backend field with no default and no capability gate.
- A change would reintroduce a removed account, registry, plugin, cloud, or
  hosted-connector method as a compatibility shim.
- A wire-contract change cannot be made backward compatible.
