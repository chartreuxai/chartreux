# 0019 Dangerous Git Command Permissions

## Status

Accepted.

## Context

Before this decision, dangerous git operations were protected only by shipped
`[tools.bash]` denylist entries: `git push`, `git checkout`, `git restore`,
`git stash drop`, `git stash clear`, `git switch --discard-changes`, `git switch -f`,
`git reflog expire`, and `git reflog delete` were denied by default and remediation
was "the user runs the command or removes the denylist entry". This had three
defects. Protection and permission were conflated: removing an entry to let the
agent push removed all protection, and a cleared denylist implicitly authorized
everything, including forced pushes. There was no notion of scope: a removed
entry authorized the operation in every repository the session could touch. And
there was no user-decided, per-operation authority: "run it yourself" was the
only remediation, even for the ordinary, routinely desired publication push.

## Decision

Dangerous git commands get a typed classifier, a frozen operation registry, a
runtime approval lifecycle, protected persistence, and an absolute denylist veto.

### Operation registry

Each recognized git operation carries a stable operation ID, a class, a
consequence summary, and a persistable flag:

- Publication: `git.push` (ordinary push only; persistable).
- Consequential: `git.push.force`, `git.push.delete`, `git.push.mirror`,
  `git.checkout`, `git.restore`, `git.switch.discard`, `git.stash.drop`,
  `git.stash.clear`, `git.reflog.expire`, `git.reflog.delete` (never persistable).
- Hard guards have no operation ID and no grant path: recursive `rm` (resolving
  run-class under the git-status-aware clean-tree carve-out where it
  establishes it), `git reset --hard`, forced non-dry-run `git clean`, and
  unmodeled or ambiguous forms. Every writer of the push-effecting surface is a hard guard too: an
  explicit path or URL push destination, a `git remote` mutation (`add`,
  `set-url`, `rename`, `remove`), and any `git config` write to
  `remote.<name>.url`, `remote.<name>.pushurl`, `remote.<name>.receivepack`,
  `remote.<name>.uploadpack`, `remote.<name>.proxy`, `remote.<name>.vcs`,
  `remote.<name>.mirror`, `remote.<name>.push`, `remote.pushDefault` (also
  spelled `remote.<name>.pushDefault`), `url.<base>.insteadOf`,
  `url.<base>.pushInsteadOf`, `branch.<name>.remote`,
  `branch.<name>.pushremote`, `core.sshCommand`, `core.gitProxy`,
  `core.askPass`, the whole `credential` section (`credential.helper` and
  `credential.<url>.helper`), and the include directives (`include.path`,
  `includeIf.<condition>.path`) — set, add, unset, and section-operation
  forms alike; reads of the same keys stay routine. Every writer of the
  program-selection surface is a hard guard the same way, in every write
  scope: `core.hooksPath`, `core.editor`, `core.fsmonitor`, and
  `sequence.editor`, the driver keys `filter.<name>.clean`,
  `filter.<name>.process`, `merge.<name>.driver`, `diff.<name>.command`,
  `diff.<name>.textconv`, and `diff.<name>.external`, the man-viewer keys
  `man.viewer` and `man.<tool>.path`, and `interactive.diffFilter`, plus
  section renames reaching `core`, `sequence`, `filter`, `merge`, `diff`,
  `man`, or `interactive` (destination section names included); reads of
  those keys stay routine.

A push combining consequential mechanisms requires all classified consequential
IDs; it never requires or inherits `git.push`, and `git.push` authority permits
only ordinary publication.

### Runtime approval lifecycle

A pure classifier (`classify_dangerous_git_command`) maps each whole shell call to
`routine`, `hard_denied`, or `grantable` with the required operation set and
repository scope (`git_dir`/`work_tree`; sibling worktrees isolated). A grantable
call without full authority is denied with kind `dangerous_command` and a proposal
(literal command, project, scope, operations, consequence). Interactive sessions
turn that denial into a user question:

- Publication: Deny (default) / Allow once / Allow this session / Always for
  this project.
- Consequential: Deny (default) / Allow once / Allow this session — never Always;
  a forged consequential "Always" is rejected.
- Escape or cancel is Deny.

Session grants are keyed by operation and repository scope; once permits authorize
exactly one invocation of the original call (never shared across concurrent
identical calls); refusals record the decided call's repository-bound canonical
identity so the decided call returns an honest denial instead of re-prompting.
An approval installs runtime authority and then the full permission pipeline
rechecks (disablement, denylist, sensitivity, paths, parent ceilings, repository
identity); on recheck failure nothing executes and unused once authority is
destroyed. Noninteractive sessions deny promptly with remediation. Child approval
never lifts a missing parent dangerous-command authority (`parent_ceiling`).

### State-aware clean-tree checkout exemption

The `git.checkout` operation keeps its consequential classification and its
approval flow, but the permission resolver grants a narrow, state-aware
exemption from that flow without user approval. Exempt shapes only:
`git checkout <branch>` where the target is an existing LOCAL branch, and
`git checkout -b <new> [<start>]` for branch creation — no pathspec, no
`--`, no `-f`/`--force`/`--discard-changes`/`--merge`/`--orphan`/
`--conflict`/`--pathspec-from-file`/`--recurse-submodules`, no `-B`, and no
other option. The exemption applies only when the ENTIRE composed shell call
is exactly one eligible git invocation: no preceding or chained commands, no
pipeline segments, no environment assignments, no wrapper commands, and no
inline configuration options (any `git -c <key>=<value>` form is ineligible —
it can alter the actual invocation's behavior, `core.hooksPath` included).
Display-only global options (`--no-pager` and equivalents) do not break
eligibility; repository relocation (`-C`/`--git-dir`/`--work-tree`) fails
closed. Every ineligible shape falls back to the existing approval class.

The exemption holds only when every candidate repository scope passes the
state guardrails, each fail-closed on any unestablishable state: a fully
clean work tree (no unstaged, staged, or untracked changes), no in-progress
operation marker (merge, rebase, cherry-pick, revert), and no repository
configuration that selects a program the invocation could run
(`core.hooksPath` and the other program-selection keys). The plain branch
form additionally requires the target to exist as a local branch (a
sanitized rev-parse probe), so a pathspec fallback, a tag, a
remote-tracking name, or a commit-ish never rides the exemption.

The bash tool executes a command STRING, and the execution path has no
access to the permission decision, so the executed string of every exempt
checkout is rewritten at execution time — the exemption predicate is
re-derived through the same whole-call eligibility check — by inserting
`--no-overwrite-ignore` immediately after the `checkout` subcommand token,
by source span, leaving every other byte of the command untouched; the
original string stays original for display and result surfaces. The flag
makes an ignored-file collision fail closed (checkout refuses) instead of
silently clobbering local content; it is the execution-time protection for
the adjudicated race between the permission-time state probe and execution
(a concurrent actor can change the tree after the probe: conflicting
concurrent changes cause a refusal, non-conflicting ones may survive a
successful non-forcing checkout — the invariant is preservation of
concurrent user changes, never a destructive outcome).

Classifier divergence, deliberate and pinned: the pure classifier
(`classify_dangerous_git_command`) is unchanged — `git checkout` keeps its
classification and the git_pure corpus bytes stay frozen — while the
resolver's state-aware layer grants the exemption; the divergence is
pinned by the shell-policy corpus, which observes resolver permission
under a discriminator without changing old record shapes.
The approval classes are preserved: a non-exempt checkout still
demands the consequential approval flow (Deny default; Allow once /
Allow this session; never Always), and the exemption never bypasses the
denylist veto, parent ceilings, or any other recheck.

Sanctioned delegated switch shapes (amended branch-ownership rule): the
two exempt shapes above are the sanctioned delegated switch operations
available to any bash-capable seat. Branch MERGING, switches to
remote-tracking branches, and operations on branches other than the
current working branch remain USER-OWNED.

Deferred alternatives, documented as non-goals: exempting
conflict-resolution reads (`checkout --ours`/`--theirs`; the sanctioned
paths are `git show :2:<path>`/`git show :3:<path>` and `git merge -X`
strategy options), a published-HEAD gate in the resolver (a branch's
publication status is a prompt/policy contract, not a resolver gate),
and any `git update-ref` relaxation (update-ref stays a hard guard).

### Protected persistence and absolute veto

Only the publication operations may persist — `git.push` and
`network.egress` — through the protected user-layer
`dangerous_command_grants_by_project` map (project key to operation IDs). The
field is accepted only from an installed user source; programmatic and generic
config writes to it are rejected, and saves use dedicated revision-checked
operations with locked checked writes. The explicit `[tools.bash]` denylist is
an absolute veto checked before grantability: no live or saved grant overrides a
blocking entry, and dangerous resolution runs last in permission resolution so it
never masks other restrictions. The shipped denylist defaults no longer contain
the recognized git entries, and an explicit `denylist = []` does not authorize
anything.

The app server exposes only inspection and revocation
(`policy/dangerous_commands/read`, `policy/dangerous_commands/revoke`, with
revision-checked live and saved outcomes); there is no public "grant an
arbitrary operation" endpoint. The Textual Settings panel (Shell Permissions)
uses these endpoints. `bash` and `bash_start` apply identical policy.

### Write-operand metadata guard

The metadata protection started as a redirect-only guard (shell redirections
into `.git` were denied) and is now a general write-operand mechanism. Every
write- or mutate-classified path operand of every command is resolved against
ALL statically tracked working directories — the `cd`/`pushd` tracking union
plus executor cwd overrides such as `env -C`/`--chdir` — and denied on
git-metadata containment, reusing `git_metadata_paths` (gitfile pointers,
`--git-dir`/`--work-tree` overrides, and `commondir` are all resolved without
invoking git). The literal `.git` path component is denied on its own, so
containment also holds for spellings that never name the metadata directory.
A nested BARE repository's metadata is gated the same way: a bare repository
carries its Git metadata directly in its directory (git's own heuristic — a
`HEAD` file plus `objects/` and `refs/`), so the operand guard also probes
each write/mutate-classified operand's own resolved ancestry for those
markers (`path_reaches_bare_repository`). An operand that is or reaches
inside a nested bare repository — through a symlinked directory component,
or a marker entry that is itself a symlink, both resolved exactly as the
operand's own resolution resolves them — denies like a `.git` operand, and a
redirection into it denies like a `.git` redirect. Reads of bare metadata
(`cat backup.git/config`) stay governed by the workspace authority alone,
and the recursive-`rm` analysis is unchanged: an operand encompassing a
nested bare repository still resolves through its recovery gate, while an
operand reaching inside its metadata is denied by this operand guard.

`git` itself is governed in two distinct ways. Its `-C`/`--git-dir`/`--work-tree`
SELECTORS still locate the repository rather than write it: they are classified
by the dangerous-command flow and risk-inspected by the repository-config
reader check, and they stay outside the operand guard. But git's RAW WRITE
operands are governed like any other write operand: `git archive --output`/`-o`
names the written archive, a writing `git config --file`/`-f` names the
written configuration file (the classifier already guards the dangerous KEYS
in every scope; the operand guard closes the arbitrary-DESTINATION write for
routine keys), `git apply --directory` names the patch's write root (every
patched path lands beneath it after `-p` stripping), and the positional
destination of `git init` (plus its `--separate-git-dir` git dir), of
`git worktree add`, of `git worktree move` (its SECOND positional — the first
selects the work tree being moved), and of `git submodule add` (its SECOND
positional, plus its `--separate-git-dir` git dir) names a directory the call
CREATES or RELOCATES its writes to. A read form (`--get`,
`--list`, a bare key) names a file git only reads, so its destination stays
authority-scoped like a selector. Git's own `-C` sequence relocates the
subcommand's cwd before it runs, so its raw write operands resolve against
the RELOCATED directory chain (each `-C` relative to the previous one), not
just the shell's tracked cwds: `git -C .git archive -o config HEAD` writes
`.git/config`, and the operand guard denies it on metadata containment.
Routine tree-writing subcommands (`apply`, `am`, `merge`, `cherry-pick`,
`revert`, `rebase`, `stash`, `switch`, `checkout`, `restore`, `pull`) are
denied when the resolved WORK TREE — after
`-C`/`--work-tree`/`--git-dir`/`core.worktree` relocation — lands inside the
git metadata dir: a work tree inside `.git` is never legitimate, and the
subcommand writes tracked (or patched) paths beneath it. The resolved work
tree is governed like a write operand: workspace authority first, then
metadata containment. Separately, `git apply --unsafe-paths` is a hard guard
with no grant path: it disables the path refusal git itself applies outside
the work tree — including paths inside `.git` metadata, which git otherwise
rejects as "invalid path" — so `git apply --unsafe-paths --directory=.git/hooks
hook.diff` plants an executable hook, and a patch with literal `.git/...`
paths needs nothing else. No legitimate agent workflow needs the flag, and
`git am` cannot be turned into the same vector: it supports no
`--unsafe-paths`, so its `--directory` root stays under git's own
invalid-path refusal for metadata paths.

The inspected command surface grew with the mechanism: `ln` (both operands —
source and link path), `rmdir`, `unlink`, `truncate`, `patch`, `unzip`, the
extraction destinations of `tar -C`/`--directory` and `unzip -d`, and the
`-o`/`--output` operands of compilers and linkers (`gcc`, `cc`, `clang`,
`g++`, `c++`, `clang++`, `rustc`, `ld`) and of the assembler (`as`, whose `-o`
names the written object; it spells no long form, so `--output` abbreviates
`--output-target` and errors). The destination-operand and in-place-mutation
family joined the same surface: `shred` and `fallocate` mutate their
positional file in place, `ar`'s positional archive is created or replaced
(and `--output` names the extraction destination), `bsdtar` is modeled with
the tar-compatible table (`-f`/`--file`, `-C`/`--directory`), `pax` with
`-f`/`--file`, and `cpio` mode-aware — its `-p`/`-l` pass-through and link
destination positionals, the `-D`/`--directory` copy-in directory, and the
`-F`/`--file`/`-O` copy-out archive files are write operands, in copy-in
(`-i`) extraction as well. `strip` joins with both write-operand classes at
once: its `-o` operand names the written result (strip preserves the
executable bit, so `strip -o .git/hooks/pre-commit evil` plants a hook
exactly like `ld -o`), and every positional operand is mutated in place.
`objcopy` is an in-place mutator whose positional operands name the copied
input and the written output. The config-content writers joined too: `gpg`'s
`-o`/`--output` (every spelling real, including attached `-oFILE` and
`--output=FILE`), `xmllint`'s `--output`/`-o`, and `scp` — a local-to-local
copy is a mutating path command like `ln`, so BOTH operands are
write-classified and inspected (a destination resolving into metadata is
denied on containment, and a source resolving into metadata is denied the
same way; an operand carrying a colon before any slash is a REMOTE spec and
stays outside the local-metadata scope). The in-place compression families
joined the file-content surface: `bzip2`/`bunzip2`/`bzcat`,
`xz`/`unxz`/`xzcat`, and `lzma`/`unlzma`/`lzcat`/`lzmainfo` replace or
annotate their positional operand in place (the `gzip`/`gunzip`/`zcat`
family was already modeled), and `zstd` (with its `zstdmt`/`unzstd`/`zstdcat`
twins — the same binary) is governed by a policy that also extracts its
written destinations: `-o` (separate form only) and
`--output-dir-flat`/`--output-dir-mirror` in both the `=` and the separate
spellings. The archive tools joined: `zip`'s positional archive operand is
created or replaced (its value-taking options — `-b`, `-i`, `-x`, `-n`, `-P`
— are consumed so their operands never surface as positional candidates),
`7z`/`7zr`/`7za` classify every positional operand (the archive is created
or replaced by the mutating commands and read by the listing ones —
write-classified either way, matching `tar -f`) and extract the attached
`-o` extraction destination (`7z` rejects the separate spelling, so its
value stays a positional candidate the scan still inspects; `-o=DIR` names
a directory literally called `=DIR`, never the metadata), and `unar` extracts
its four real `-o` spellings (`-o DIR`, `-o=DIR`, `-output-directory DIR`,
`-output-directory=DIR`). `mktemp` creates a file at every template
positional and in the directory `-p`/`--tmpdir` names (the long form takes
its directory only attached), so the template positionals join the
file-content surface and the `-p`/`--tmpdir` directories are extracted as
write operands. Every write-operand option of a guarded command is extracted
in the separate form AND the attached forms —
`-C.git`/`-C <dir>`, `--target-directory=.git`, `-out=.git/config` — with the
same semantics, so the workspace-authority check benefits too:
`cp`/`mv`/`install`/`ln` `-t`/`--target-directory`, `tar` `-f`/`--file`,
`-C`/`--directory`, `--add-file`, `unzip` `-d`, `patch` `-o`/`--output`,
`-r`/`--reject-file`, `-d`/`--directory`, `-B`/`--prefix`, `-Y`/`--basename-prefix`,
`openssl` `-out`/`--out`, `iconv` `-o`/`--output`, `gpg` `-o`/`--output`, and
`xmllint` `--output`/`-o`, `rsync`
`--temp-dir`/`--partial-dir`/`--backup-dir`/`--compare-dest`/`--copy-dest`/`--link-dest`/`--log-file`
(and `-T`), and `tee --output-error`. `split`'s trailing PREFIX operand and
`csplit`'s `--prefix`/`-f` operand name the written output files and are
modeled as write operands as well. If a write operand cannot be statically
resolved, the command is denied (fail closed).

Hard links are contained at both ends. Link creation is denied when the SOURCE
resolves into git metadata (`ln .git/config c`, `ln -s .git g`, `cp -s/-l/-a/-r`),
so the policy cannot be asked to create an alias through which a later write
reaches `.git` content; the existing at-use symlink resolution keeps covering
later writes through pre-existing SYMLINKS. A hard link created OUTSIDE the
policy (pre-existing, or made by another process) is invisible to path
resolution — `Path.resolve()` follows symlinks but never hard links — so
write-time containment additionally compares (device, inode) identities of the
write target against the critical metadata files (`config`, `config.worktree`,
`HEAD`, `commondir`, `index`, and every file under `hooks/`). That closes the
probed write-through (`echo x > c` with `c` a pre-existing hard link to
`.git/config`) at the moment it happens.

### Program-selection config guards

`git config` writes to `core.hooksPath`, `core.editor`, `core.fsmonitor`, and
`sequence.editor` are hard guards in every write scope (local, `--global`,
`--file`, `--worktree`, and `-C`/`--git-dir` relocation): `core.hooksPath`
selects the directory git runs hooks from (a granted push — or any
hook-running subcommand, including readers that refresh the index — executes
the hooks there), `core.editor` the program git runs for `commit`/`tag`/`rebase`
without `-m`, `sequence.editor` the program an interactive rebase runs, and
`core.fsmonitor` the daemon git runs for status-bearing subcommands
(`git add`, `git commit`). The driver sections join the same guard on their
command-string keys, in every write scope and case-folded:
`filter.<name>.clean` and `filter.<name>.process` select the programs git
runs for files with that filter (`git add`, `git checkout`),
`merge.<name>.driver` the program git runs for files with that merge driver
(`git merge`), and `diff.<name>.command`, `diff.<name>.textconv`, and
`diff.<name>.external` the programs git runs for diffs with that driver
(`git diff`, `git log -p`). The `man` section joins the same guard on its
program keys, in every write scope and case-folded: `man.<tool>.path` names
the man viewer's executable and `man.viewer` selects the viewer tool, so
together they select the program `git help` runs (`git help` is a routine
NON-reader subcommand with no repository-config risk check). The
`interactive` section joins on `interactive.diffFilter`, the filter git runs
for `git add -p` diffs (probe-verified to execute under a color-capable
tty; the shipped no-tty tool context never fires it, and every routine pty
route is denied — see Accepted limitations). Reads of the same keys stay
routine, as do writes to the driver sections' non-program keys
(`filter.<name>.required`, `filter.<name>.smudge`, `merge.<name>.name`, and
data spellings — a documented residual: `smudge` is a command string git runs
on checkout, but it is not in the guarded key set), and to the round-3
sections' non-program keys (the deprecated `man.<tool>.cmd`, which git 2.53
ignores with a warning, and data spellings such as `man.<tool>.tool`).
Because guarding key sets alone is bypassable, section RENAMES reaching
`core`, `sequence`, `filter`, `merge`, `diff`, `man`, or `interactive` —
either operand, so destination section names are covered — are hard guards
too (an agent can write `hooksPath` into an unrelated section and rename it
to `core`, or a driver key into a section renamed to `filter`, or a man-viewer
path key into a section renamed to `man`). The denial names the section it
lands in: the destination section for a rename reaching a guarded section, the
matched (source) section for a rename moving one out. Section removals of
`core.*`/`sequence.*`/`filter.*`/`merge.*`/`diff.*`/`man.*`/`interactive.*`
stay routine because a removal cannot move a key into a guarded section.

## Consequences

This is a documented breaking change: configurations that cleared the denylist
to let the agent push now need an approval flow or a saved project grant, and
configurations relying on shipped git entries no longer get denylist enforcement
for them (they get the stricter approval flow instead). Hard guards are
unchanged for `git reset --hard` and forced non-dry-run `git clean`: both remain
denied with no approval path. Recursive `rm` is no longer a flat hard guard:
over a git-clean tree it resolves run-class under the git-status-aware clean-tree
carve-out — every operand inside the assessed repository, no Git metadata
overlap, recovery established for every repository the operand touches — and
every unestablishable aspect fail-closes back to the hard-guard denial (see the
Recursive rm analysis section of the tools-safety guide). An approval is
permission authority, not standing intent to publish; prompts and documentation
say so.

## Security invariants

- Nothing the model triggers is itself a grant: a grant exists only after the
  user decides the approval question (or a prior saved project grant for
  `git.push`).
- The recheck after approval is the final gate; it re-evaluates every ordinary
  restriction, so dangerous authority never bypasses disablement, denylist,
  sensitivity, path, or ceiling policy.
- Authority is per loop, per operation, and per exact repository scope; child
  authority is bounded by the parent ceiling and never shared across sibling
  worktrees.

## Accepted limitations

- File tools can still write `.git/` paths: `write_file`/`edit` resolve their
  target against the workspace authority, which permits any path inside the
  workspace, metadata included. Closing that is a scheduled follow-up feature;
  the shell operand guard does not cover the file tools.
- Hard links to metadata files OUTSIDE the critical set (`objects/`, `refs/`,
  `logs/`, `packed-refs`, worktree state, and so on) created outside the policy
  (pre-existing, or made by another process or the file tools) still write
  through: path resolution cannot see a hard link, and the write-time inode
  comparison covers only the critical files whose content selects programs or
  repository identity. Link creation through the shell is denied at
  source-containment time, so the policy itself never creates such an alias.
- Archive MEMBER names are not inspected: the operand guard checks operands
  (including extraction destinations), not archive contents, so `tar -xf`/`unzip`
  of an archive whose members are named `.git/config` still writes metadata when
  extracting into a worktree, and `cpio -i` (without `-D`) extracts member
  names into the copy-in directory unchecked. Heredoc bodies, script contents,
  and variable expansions remain opaque for the same reason.
- The arbitrary-code-execution class (`python -c`, `awk`, `ed`, the `ext::`
  transport) is pre-existing surface: those forms remain ALLOWED by the shipped
  policy — the shell analysis layer gates unsupported syntax, not these
  commands — so the metadata guard is not a containment boundary against
  arbitrary code execution. An agent that can run `python -c` can write
  `.git/config` or plant a hook by other means; closing that class is a
  separate feature.
- The work-tree relocation guard covers the routine tree-writing subcommands
  listed above. Other subcommands that can write work-tree files under a
  metadata-resolving relocation — `git reset --keep`/`--merge`, `git rm`,
  `git mv`, `git bisect`, `git submodule update` — stay routine there (a
  documented residual; `git reset --hard` and forced `git clean` are already
  hard guards). Tree-writing forms relocated with `-C` into the metadata dir
  but no `--work-tree` are denied by the guard even where git itself would
  refuse them for lack of a work tree — a deliberate fail-closed
  over-strictness.
- The attached-option extraction is a heuristic superset: an unmodeled
  value-taking option leaves its value as a positional candidate, which can
  only over-block, never hide a modeled write operand. `tee --output-error`
  takes an error MODE (warn/exit/...), not a file, in GNU coreutils; its
  attached value is still extracted as a write operand, and the valid modes
  are workspace-relative strings that stay permitted.
- `git init`/`git clone`/`git worktree add` create fresh repositories: they
  carry no configured remotes, bare pushes are denied, and explicit push
  destinations are hard-denied, so a fresh repository is not a push vector.
  Their destinations are now write-classified operands, so `git init .git/evil`
  and `git worktree add .git/wt2` are denied on metadata containment, and
  destinations resolving outside the workspace are denied by the write-operand
  authority (exactly like `git archive -o ../x`); `git clone` keeps its
  previous treatment (its destination is not modeled as a metadata write).
  The same destination classification now covers `git worktree move <wt>
  <dest>` (moving a worktree INTO `.git` rewrites
  `.git/worktrees/<name>/gitdir`, probe-verified) and `git submodule add
  <url> <dest>` (creating a submodule worktree inside `.git` and writing a
  submodule entry into `.git/config`, probe-verified), including their
  `--separate-git-dir` git dirs: metadata containment and outside-workspace
  authority deny both, consistent with the documented init/worktree-add
  flips.
- `gpg.program`/`push.gpgSign` writes and `filter.<name>.smudge` writes stay
  routine (they are not in the guarded key set), and USER-configured drivers
  still execute on `git add`/`git checkout`/`git merge`/`git diff`: those
  subcommands are not readers, so the repository-config risk check does not
  inspect them. The agent-write guard closes the agent-authored driver
  (`filter.<name>.clean`/`.process`, `merge.<name>.driver`,
  `diff.<name>.command`/`.textconv`/`.external`); what remains is the
  previously-assessed indirection class — repository-configured programs,
  which the reader-side risk check already denies readers for.
- `core.pager` is neutralized at runtime via `GIT_PAGER=cat`, so a
  `core.pager` write does not take effect; it stays a routine write.
- Enumeration stop rule (maintainer decision, P10 fix round 3): the write-
  operand and exec-key enumerations are CLOSED. This was the last enumeration
  round; same-class stragglers discovered later are ACCEPTED RESIDUALS under
  the documented rationale: the destination-operand heuristic is bounded and
  is not a containment promise (an unmodeled value-taking option can only
  over-block, never hide a modeled write operand), and `python -c` is
  policy-ALWAYS (pinned by test), so every same-class ACE chain is
  polymorphism of a capability the policy deliberately grants with zero
  marginal privilege. No further fix rounds on this surface without a new
  maintainer decision. The accepted residuals, named:
  - Commands with write-destination operands not yet in the tables:
    `sponge` (the tee twin), `sqlite3` (DB-creation positional), `uudecode`
    (writes member-named files), `dos2unix`/`unix2dos` (in-place mutators),
    `yq -i` (in-place mutator), `cabextract -d` (extraction destination).
    These are not installed in the probe environment, so they are
    policy-surface residuals, not runtime-confirmed chains.
  - `curl`/`wget` left the denylist and joined the modeled surface, so the
    earlier denylist-protection framing no longer applies: they are
    destination-aware network clients (non-loopback egress resolves ask-class,
    `network.egress`), and their `-o`/`--output` (curl) and
    `-O`/`--output-document` (wget) operands are modeled write operands —
    collected, path-inspected, and metadata-guarded like every other write
    operand, with the derivable default downloads (`curl -O`, a plain
    `wget <url>` writing the URL's file part under `--output-dir`/`-P`)
    collected too. The shipped default denylist no longer names them: the
    operand guard and the egress rule protect these writes, not a removable
    denylist entry.
  - `toybox` (busybox's multi-call twin) is now handled by the busybox-style
    applet admission (shell applets `sh`/`ash` are analyzed as shell sources;
    every other applet fails closed with "unsupported toybox applet"), which
    closes both its unmodeled-applet hole (`toybox cp seed.txt .git/config`)
    and its shell route. It is not installed in the probe environment, so
    this is policy-only, verdict-tested, not execution-tested (same status
    as `pax`/`bsdtar`).
  - `interactive.diffFilter` writes are now hard-guarded, and the key really
    executes on `git add -p` under a color-capable tty (probe-verified raw).
    The trigger is currently unreachable through routine commands: the
    filter only runs when git renders a color diff to a tty, every routine
    pty route is denied (`script` in any form, tmux/screen/socat denylisted,
    expect/unbuffer not installed), and python/expect-style pty allocation is
    the documented ACE class. The write guard closes the exec-class key; the
    trigger-unreachability is a property of the shipped tool context, not a
    guarantee.
  - `core.fsmonitor` presence IS detected at-use like `core.hooksPath` was:
    the repository-config reader risk check denies readers (`git status`,
    probe-verified) when the repository's own config sets an active
    `core.fsmonitor`. The gap the round-3 write guard closed is the
    non-reader side: `git add`/`git commit` are not readers, so they get no
    repository-config risk check, and the write itself was routine in every
    scope. `man.viewer`/`man.<tool>.path` have no reader-side coverage at
    all (their trigger `git help` is a non-reader), so the write guard is
    their only protection.
- Non-cooperating-writer window: persistence uses locked checked writes, but an
  external editor racing a save can still conflict; the revision check detects it
  rather than preventing it.
- Refusal canonicalization is shlex-token level: whitespace and quoting variants
  that tokenize identically share refusal memory, but a reformulated command
  re-prompts. Refusal memory is per decided call, not a blanket block.
- Revocation does not cancel an in-flight approval question: answering it after
  a revocation still grants fresh authority for the decided scope.
- In-place rewind keeps session grants: an in-place rewind truncates the
  conversation under the same session identity, so the grant store is not
  cleared and the grants stay visible and revocable in the Shell Permissions
  panel, while a fork rewind starts a new session and clears them. The
  asymmetry is deliberate: the store is keyed to the session, not to the
  message history.
- Environment-plumbed git config (`GIT_CONFIG_*` assignments, including
  `GIT_CONFIG_COUNT`-keyed pushes) is caught by the pre-existing
  dangerous-environment assignment guard and routed to generic approval, not
  the typed dangerous-command flow.
- `push.default` writes remain routine: they broaden what a bare push
  publishes (which refs, under which remote names), never force, delete, or
  mirror, so they cannot turn an ordinary `git.push` grant into a
  consequential one.
- Destination verification reads the repository's own config only: a remote
  configured solely in global or system config is denied even though git
  would resolve it — a deliberate fail-closed over-strictness, because the
  grant scopes the repository whose own config defines its remotes.
- Verification checks `remote.<name>.url` presence while git's actual landing
  destination is `remote.<name>.pushurl` when set: a user-authored `pushurl`
  is in-contract (the user's own config defines push), and the divergence is
  why agent-authored includes are hard-guarded — an included file could set
  `pushurl` for a remote the resolver verified.
- Probe provenance: the git behaviors behind the push-destination and
  config-write guards (destination resolution, receive-pack and hooks
  execution, include folding, program execution via `core.sshCommand`,
  `core.gitProxy`, `core.askPass`, and `credential.helper`, and
  `git remote update` helper execution) were probe-verified on git 2.53.0.
  The attached-option extraction and the work-tree relocation guard were
  probe-verified against the shipped tool set (git 2.53.0, GNU tar 1.35,
  GNU patch 2.8, UnZip 6.00, OpenSSL 3.5.5, rsync 3.4.1, glibc iconv 2.43,
  uutils split/csplit/tee 0.8.0): every modeled spelling is real, a relative
  `core.worktree` resolves against the git dir, and `git apply` applies
  relative to the resolved work tree. The round-2 closures were probe-verified
  the same way (git 2.53.0, GNU cpio 2.15, GNU ar 2.46, uutils shred 0.8.0,
  util-linux fallocate 2.41.3; `pax` and `bsdtar` are not installed in the
  probe environment, so their modeling is policy-only and verdict-tested, not
  execution-tested): git refuses literal `.git` patch paths ("invalid path")
  and `git apply --directory=.git/hooks` without `--unsafe-paths`, so those
  forms never reach the metadata — but `--unsafe-paths` disables that refusal,
  and `git apply --unsafe-paths --directory=.git/hooks hook.diff` plants an
  executable hook (as does a patch with literal `.git/...` paths under
  `--unsafe-paths`, with no `--directory` at all); `git am` supports no
  `--unsafe-paths`, so its `--directory` root stays refused. Git's own `-C`
  relocation moves the write root for every relative operand:
  `git -C .git archive -o config HEAD` overwrites `.git/config`,
  `git -C .git/hooks archive -o pre-commit` and `git -C .git config
  --file=hooks/pre-commit user.name x` plant files in the metadata,
  `git -C sub archive -o ../.git/config HEAD` escapes a legitimate relocation
  into the metadata, and an empty `-C ''` is a no-op. `printf 'pre-commit\n'
  | cpio -p .git/hooks` plants an executable hook that runs on the next
  commit, `cpio -i -D .git` overwrites `.git/config`, `cpio -o -F
  .git/config` writes the archive into the config, `ar rcs .git/evil.a x.o`
  creates a metadata file, and `shred .git/config`/`fallocate -l 100
  .git/config` mutate the config in place. `git init
  --separate-git-dir=.git/evil` creates `.git/evil`, and a configured
  `filter.pwn.clean` executes on `git add` of a filtered file, a configured
  `merge.ours.driver` executes on a conflicting `git merge`. Git also refuses
  `git -C .git status` ("must be run in a work tree"), which the policy still
  permits because the denial criterion is the resolved work tree, not the
  selector spelling. The round-3 closures were probe-verified the same way
  (git 2.53.0, GNU binutils strip/objcopy/as 2.46, Info-ZIP zip 3.0, 7-Zip
  7z/7zr/7za 26.00, unar 1.10.8, xz-utils 5.8.3, zstd 1.5.7, uutils coreutils
  mktemp 0.8.0, OpenSSH scp 10.2p1, GnuPG 2.4.8, libxml2 xmllint 2.15.2;
  `toybox` is not installed, so its applet admission is policy-only): `strip -o` and `objcopy`'s
  positional output plant an EXECUTABLE hook in `.git/hooks` (both preserve
  the exec bit), `as -o` clobbers `.git/config`, `gpg --decrypt -o` and
  `xmllint --output` give full `.git/config` content control, `scp` local-to-
  local overwrites it, `bzip2`/`xz`/`lzma` destroy `.git/config` in place and
  `zstd` writes `.zst` files into `.git`, `zip`/`7z a` create archives inside
  `.git` and `7z x -o.git`/`unar -o .git` extract into it, `mktemp` creates
  files inside `.git`, `git worktree move <wt> .git/wt2` moves a worktree
  into `.git` and rewrites the `gitdir` pointer, and `git submodule add
  <url> .git/evilmod` creates a worktree inside `.git` and writes a
  submodule entry into `.git/config`. The grammar facts the modeling relies
  on were verified against those binaries: `as` takes no positional output
  (multiple inputs, `-o` only, no long form), `strip` spells no long
  `--output` (`--output` abbreviates `--output-target` and errors), 7z's `-o`
  is attached-only (the separate spelling is rejected; `-o=DIR` names a
  directory literally called `=DIR`), unar takes `-o DIR`/`-o=DIR`/
  `-output-directory DIR`/`-output-directory=DIR` (attached `-oDIR` is
  rejected), xmllint's `--output=`/`-oFILE` attached spellings are rejected,
  zstd's `-o` is separate-only and its `--output-dir-*` take `=` and separate
  spellings, mktemp's `--tmpdir` takes its directory only attached, and
  `core.fsmonitor`/`man.<tool>.path`+`man.viewer`/`interactive.diffFilter`
  really execute on `git add`/`git help <cmd>`/`git add -p` under a tty.
