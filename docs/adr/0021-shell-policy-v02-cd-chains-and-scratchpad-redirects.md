# 0021 Shell policy v0.2: occurrence-local cd chains and scratchpad redirect targets

## Status

Accepted.

## Context

An audited ML4 (mistral-large-4) log dive found the dominant shell-denial
class was a parser-modeling limitation, not a security finding. The shell
analysis layer carried an order-blind veto: it rejected a whole command
whenever ANY `cd`/`pushd`/`popd` and ANY `file_redirect` node coexisted,
without modeling where in the chain each occurred. 17 of 17 sampled denials
in that class were benign. The veto fired on:

- pure `2>&1` merges with no file target (`cd sub && grep x f 2>&1 | head`),
  because fd duplication parses as a `file_redirect` node;
- `> /dev/null` and other device sinks, because the analysis's device-sink
  exemption rewrote the destination to a relative literal
  (`__standard_device_sink__`) that still parsed as a `file_redirect`;
- redirects that occur BEFORE a cd (`make > out.log && cd sub`), which are
  written to the pre-cd working directory, not the post-cd one;
- absolute redirect targets into the session scratchpad (`cmd > /abs/sp/log`),
  which the path layer would allow anyway;
- ML4's natural multi-part investigation shape —
  `cd repo && grep ... | head; echo ---; git branch -a` — which the flattened
  command model could not express.

A deep-review panel verified that the resolver's cd-chain tracker already
existed (`_update_guardrail_cwds`/`_scoped_guardrail_cwds`) and that redirect
destinations already became synthetic WRITE operands resolved against the
tracked cwds — so the correct fix was removing the coarse veto and leaning on
that model, not building a second tracker.

## Decision

### 1. Remove the order-blind cd+redirect veto (occurrence-local modeling)

The analysis layer (`_shell_permission_analysis.py`) no longer denies a
command merely because a working-directory change and a redirect coexist. The
cd-chain rule is now the resolver's existing model, upgraded to occurrence
positions:

- The analysis attributes every literal redirect target to the command part
  it precedes (`ShellPermissionAnalysis.redirect_operands`, one group per part
  plus a trailing group for targets after the last part). A redirect attached
  to a list or pipeline anchors to its last command (`cd sub && make >
  out.log` redirects make's output); a redirect-only statement anchors to the
  next command, or to the end of the chain.
- `_expand_guardrail_parts` places each target's synthetic `cat <target>`
  WRITE operand immediately before the command part it redirects, so it
  resolves against the working directory in force at that point. A redirect
  that occurs before a cd resolves against the pre-cd directory; a redirect
  attached to the cd command itself likewise (Bash opens the file before
  running `cd`). A redirect after a cd that cannot be statically resolved
  still fails closed through the existing unknown-cwd operand denial.
- The exemption criterion is structural, not a string hack: fd duplication
  (`2>&1`) and device sinks (`_REDIRECTION_DEVICES`: `/dev/null`,
  `/dev/stdout`, `/dev/stderr`) never count as filesystem-target redirects.
  `_redirect_reason` now exempts device sinks natively, which retires the
  `__standard_device_sink__` source rewrite (the rewrite made the destination
  relative but the node still parsed as a `file_redirect`, which is why
  `/dev/null` merges fired the old veto).
- Compound chains are in scope: semicolons, `echo ---` separators, pipelines,
  and nested `bash -c` sources all resolve occurrence-locally, because the
  resolver already tracks cwds across chain separators and the per-part
  operand placement applies inside nested sources too.
- The `<redirect>` marker is preserved: `python > log.txt` stays allowed while
  bare `python` stays standalone-denied, and `curl ... > log.txt` stays denied
  via prefix matching. The marker is also stripped before cd/pushd/popd token
  counting, so a redirect attached to a cd no longer makes the cwd unknown.

### 2. Authorize absolute redirect targets inside the session scratchpad

Absolute redirect targets that resolve inside the session scratchpad roots
are now allowed. The authorization is placed at the resolver layer, after the
metadata check:

- The analysis layer has no session context, so the scratchpad roots are
  threaded through the resolver's call sites: `resolve_permission` and the
  nested-source analysis inside `_expand_guardrail_parts` both pass the
  authority's roots. `derive_call_requirements` in the dangerous-command
  grants module accepts the same parameter for uniformity, but its callers
  pass no roots today, and it never reads `approval_reasons` — the only
  roots-sensitive analysis output — so the re-derivation is roots-neutral in
  practice. This is a documented deviation from the original threading
  intent: safety rests on the path layer's independent WRITE gating of every
  literal redirect target against the caller's own authority, and a future
  refactor that reads approval reasons from the re-derivation must thread
  the roots first.
- Matching is component-wise against the path authority's scratchpad roots
  (`is_canonical_scratch_path` over the full realpath) — never a string
  prefix, so a sibling directory that merely spells like a scratchpad root, a
  `..` escape, or a symlink escaping the root stays denied.
- The metadata guard still sees the ORIGINAL literal target: the metadata
  check runs before the analysis reasons are consulted, and a scratchpad
  symlink to `.git` stays denied (`ln -s <repo>/.git <sp>/g && echo x >
  <sp>/g/config` and `cd <sp> && ln -s <repo>/.git g && echo x > g/config`
  are both denied).
- The pure-analysis cache key stays `(command, nested_source)`; the roots are
  deliberately not part of it. Safety does not rest on the cache: the path
  layer independently gates every literal redirect target as a WRITE, so a
  cached scratchpad exemption can never authorize a write outside
  workspace+scratchpad (the exemption is friction relief only).

### 3. Load-bearing rules are unchanged

The rm -rf guard, workspace boundaries, protected Git metadata (including the
P10 operand-level extensions: inode defense, work-tree relocation guard,
write-operand tables), dangerous environment names, the denylist, and the
eval/exec denial are untouched, and all P10 guard tests stay green through
the rework.

## Consequences

Allowed now (the audited benign dominant class): `cd sub && grep x f 2>&1 |
head; echo ---; git branch -a`, `cd sub && grep x f > out.log`,
`grep x f > out.log && cd sub`, `cd sub > log`, `cd sub; grep x f > out.log`,
`cd sub && grep x f >> out.log 2> err.log`, `cmd > /abs/scratchpad/log`, and
the same shapes inside nested `bash -c` sources.

Also allowed now: literal redirect targets that merely look like globs
(`echo x > f[0]`, `echo x > f"*"`, `echo x > "f"[0]`). Bash never performs
pathname expansion on a redirect destination, so the path-glob heuristic —
which stays fail-closed for command operands (`cat f[0]`, `cp f* out`), where
the shell really globs — does not fire for redirect destinations; the literal
target is still gated as a WRITE by the path layer like any other.

The broadened `_literal_redirect_target` also closes two real pre-v0.2
bypasses: at HEAD the deleted `_extract_redirect_paths` accepted only
`word`/`string`/`raw_string` destinations, so a concatenation destination
was never extracted and never path-checked. `echo x > "sub"/../../outside.log`
was an arbitrary write outside the workspace (the target never reached the
path layer), and `echo x > "sub"/../.git/config` wrote protected Git
metadata, bypassing the P10 redirect metadata guard. Both are denied now:
the concatenation target is attributed to its occurrence position and gated
like any other literal target, and the metadata guard still sees the
original literal spelling.

Still denied: redirect targets outside workspace+scratchpad (`/etc/x`,
`<sp>/../outside.log`, workspace-absolute targets outside the scratchpad),
`cd` to directories outside the workspace, dynamic cd forms (`cd $HOME`,
`cd $(pwd)`), redirects after an unresolvable cd (`popd +1 && ... > out.log`
fails closed on the unknown cwd), heredocs, sourcing a script outside
workspace+scratchpad (`source /outside/evil.sh`, `. /outside/evil.sh`, and the
cd+redirect variant `cd sub && source ../evil.sh > out` — `source`/`.` read
their operand, so it is path-inspected as a READ operand), and every P10
metadata/ln-guard shape.

## Accepted residuals (documented, not fixed)

- **Heredocs and herestrings** stay fail-closed (`heredocs are unsupported in
  v0.1`). v0.3 candidate.
- **Variable expansion and `$(...)` substitution** stay fail-closed: a
  redirect target or cd operand containing them is denied (as an unmodeled
  expansion, or through the unknown-cwd path). v0.3 candidate.
- **`cd sub && cd ..` over-approximation.** The tracked-cwd set is a monotonic
  union over the whole chain, so a redirect target is checked against every
  statically possible cwd, not only the real one. This can produce false
  denials (e.g. a target that would be outside the workspace from one
  candidate cwd is denied even if the real cwd keeps it inside). Documented;
  not fixed — narrowing the union would require modeling command success.
- **CDPATH / AUTO_CD assumption.** The cd-chain model assumes the runtime
  shell resolves `cd <dir>` to `<cwd>/<dir>`. In-command `CDPATH=...`
  assignments are denied by the existing `DANGEROUS_ENV_NAMES` machinery
  (`CDPATH` is in that set), and AUTO_CD is inert because the runtime shell
  is non-interactive. An *inherited* `CDPATH` from the user's profile is not
  scrubbed by the runtime (`scrub_child_env` removes credential names only)
  and is not visible to the policy: with CDPATH set, a bare-relative `cd`
  could resolve elsewhere and the tracked cwd could diverge. This residual
  predates v0.2 (the tracked-cwd model always assumed it); v0.3 candidates
  are scrubbing CDPATH in the child shell environment or failing closed on
  bare-relative cd targets when CDPATH is inherited.
- **Workspace-absolute redirect targets** (`cmd > /abs/workspace/file`) stay
  denied: only scratchpad absolute targets are authorized. Friction-only;
  the path layer would allow them.
- **Analysis cache is roots-blind (false-denial direction only).** The
  pure-analysis cache key stays `(command, nested_source)` and deliberately
  excludes the scratchpad roots, so within one `bash_start` launch the local
  resolver and the parent invocation guard can hold different authorities
  under the same key. Probed safe: the path layer independently gates every
  literal redirect target as a WRITE against the caller's own authority, so
  a stale roots-sensitive analysis reason can only add a spurious approval
  prompt (the false-denial direction), never authorize a write. The cache is
  deliberately not re-keyed.
- **pushd/popd operands are not path-checked (named follow-up).** `pushd`
  and `popd` track the working directory like `cd`, but their directory
  operand is not authority-checked the way `cd`'s operand is (asymmetric
  with cd): a `pushd` outside the workspace only fails closed through later
  operands that resolve against the tracked cwd. Not fixed.
- **`python3 -c '...' > out` is allowed (named follow-up).** The standalone
  denylist matches bare command names only, so `python3` with any operands —
  including `-c` — stays runnable (with or without a redirect; the
  `<redirect>` marker semantics are preserved deliberately). Not fixed.

## Validation metrics

- Live proof battery (scratchpad script, before/after table): every audited
  benign shape flips from denied to allowed; every load-bearing denial
  (outside-workspace redirect, cd outside workspace, dynamic cd, unknown-cwd
  redirect, `.git` targets, scratchpad symlink escape, ln-guard chains,
  standalone/prefix denylist, sensitive paths) stays denied.
- Targeted suites: `tests/tools/test_shell_permission_analysis.py`,
  `test_shell_command_policy.py`, `test_git_metadata_guard.py`,
  `test_dangerous_git_classifier.py`, plus `test_bash.py`,
  `test_scratchpad_permissions.py`, and the grants tests — all green, with
  updated and new tests only (no test count reductions).
- Full suite, pyright, ruff (check + format), and `mkdocs build --strict`
  all pass.

Fix round (package review verdict PASSED WITH WARNINGS), all four findings
closed without reverting any v0.2 behavior:

- F1: the roots-threading claim above is corrected — callers pass no roots
  and the re-derivation never reads the roots-sensitive output, so the
  parameter is uniformity-only today; the `derive_call_requirements`
  docstring states the same, with the thread-roots-first warning.
- F2/F3: the concatenation-destination coverage change is documented in
  Consequences with the two pre-v0.2 bypasses it closes; redirect
  destinations are exempt from the path-glob heuristic (bash never globs
  them) while command operands keep it, with tests pinning both directions
  (`> f[0]` allowed, `> .gi"t"/config` still denied, `cat f[0]` and
  `cp f* out` still denied).
- F4: `source`/`.` operands join the path-inspected surface as READ
  operands, so outside scripts are denied (`source /outside/evil.sh > out`,
  `cd sub && source ../evil.sh > out`) while `source sub/setup.sh`,
  `source <scratchpad>/env.sh`, and `. ./lib.sh` stay allowed.
- F5: recorded above as named residual follow-ups (not chased).
- F6: the cache-key docstring and the residual above state the roots-blind
  truth (false-denial direction only); the cache is not re-keyed.
