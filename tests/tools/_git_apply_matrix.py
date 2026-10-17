"""Shared git-apply matrix fixtures for the shell-policy twin suites (W6a, 12b).

``test_shell_command_policy.py`` (policy layer: ``analyze_shell_command_policy``)
and ``test_git_metadata_guard.py`` (resolver layer: ``bash_tool.resolve_permission``)
both parametrize the git-apply hard-guard matrices below. The matrices live here so
each matrix has a single definition; each suite keeps exactly one layer-level
assertion per matrix per layer. Where the twin suites' resolver-layer matrices
diverged, the shared matrix is the UNION of both sides' commands (additions are
marked with their origin suite).

Migration commits (stage 12b): 12b-1 introduces this module with the git-apply
unsafe-paths family; 12b-2 adds the git -C relocated family; 12b-3 adds the
unmodeled-file family. The per-deletion survival mappings are recorded in the
stage-12 report.
"""

from __future__ import annotations

# --- git apply --unsafe-paths hard guard, policy layer ---
# --unsafe-paths disables the path refusal git itself applies outside the work
# tree, including paths inside .git metadata: a hard guard, whatever else the
# call carries.
GIT_APPLY_UNSAFE_PATHS_COMMANDS: tuple[str, ...] = (
    "git apply --unsafe-paths hook.diff",
    "git apply --unsafe-paths --directory=.git/hooks hook.diff",
    "git apply --directory=.git/hooks --unsafe-paths hook.diff",
    "git apply --unsafe-paths --directory=fixes hook.diff",
    "git apply --unsafe-paths --directory=/abs/path hook.diff",
)

# --- git apply --unsafe-paths hard guard, resolver layer (union) ---
# The metadata-guard suite's matrix, plus the policy suite's resolver twin's
# in-workspace --directory spelling.
GIT_APPLY_UNSAFE_PATHS_RESOLVER_COMMANDS: tuple[str, ...] = (
    "git apply --unsafe-paths hook.diff",
    "git apply --unsafe-paths --directory=.git/hooks hook.diff",
    "git apply --directory=.git/hooks --unsafe-paths hook.diff",
    "git apply -p0 --unsafe-paths hook.diff",
    "git apply --unsafe-paths --directory=/tmp/evil hook.diff",
    # Added from the policy suite's resolver twin.
    "git apply --unsafe-paths --directory=fixes hook.diff",
)

# --- git apply benign forms, policy layer ---
# Benign apply forms carry no hard guard: an in-workspace --directory is an
# ordinary write operand, and git am's --directory names the mailbox it reads.
GIT_APPLY_BENIGN_COMMANDS: tuple[str, ...] = (
    "git apply hook.diff",
    "git apply --directory=fixes hook.diff",
    "git apply --exclude=*.txt --include=*.py hook.diff",
    "git am hook.diff",
    "git am --directory=.git/hooks hook.diff",
)

# --- git apply benign forms, resolver layer (union) ---
# The metadata-guard suite's matrix, plus the policy suite's resolver twin's
# spellings.
GIT_APPLY_BENIGN_RESOLVER_COMMANDS: tuple[str, ...] = (
    "git apply fix.diff",
    "git apply --directory=patches fix.diff",
    "git apply --exclude=.git/* hook.diff",
    "git apply --include=seed.txt hook.diff",
    "git apply --build-fake-ancestor=.git/foo hook.diff",
    # git am supports neither --unsafe-paths (git rejects the option) nor a
    # metadata-reaching --directory (git rejects the path), so the spelling
    # stays with git's own refusal.
    "git am --directory=.git/hooks hook.diff",
    # Added from the policy suite's resolver twin.
    "git apply hook.diff",
    "git apply --directory=fixes hook.diff",
    "git apply --exclude=*.txt hook.diff",
    "git am hook.diff",
)

# --- Git's own -C relocation of write operands, resolver layer (union) ---
# Git applies its -C sequence before the subcommand runs, so a relative write
# operand resolves against the RELOCATED directory: `git -C .git archive -o
# config HEAD` writes .git/config. The metadata-guard suite's matrix, plus the
# policy suite's resolver twin's spellings (its `subdir` directory; the guard
# suite's `git_repo` fixture provides `sub` and the tests create `subdir` and
# the `g` symlink alias).
GIT_C_RELOCATED_DENIED_COMMANDS: tuple[str, ...] = (
    "git -C .git archive -o config HEAD",
    "git -C .git archive --output=config HEAD",
    "git -C .git archive -o hooks/pre-commit HEAD",
    "git -C .git/hooks archive -o pre-commit HEAD",
    "git -C .git archive -o index HEAD",
    "git -C .git config --file=config user.name x",
    "git -C .git config --file=hooks/pre-commit user.name x",
    "git -C .git config -f config --unset user.name",
    "git -C sub archive -o ../.git/config HEAD",
    "git -C sub config --file=../.git/config user.name x",
    # Through a symlink alias into the metadata dir.
    "git -C g archive -o config HEAD",
    "git -C . -C .git archive -o config HEAD",
    # Added from the policy suite's resolver twin.
    "git -C .git/hooks config --file=pre-commit user.name x",
    "git -C subdir archive -o ../.git/config HEAD",
    "git -C subdir config --file=../.git/config user.name x",
)

# --- Git's own -C relocation, benign forms, resolver layer (union) ---
# Legitimate relocations: the write lands outside the metadata.
GIT_C_RELOCATED_BENIGN_COMMANDS: tuple[str, ...] = (
    "git -C sub config --file=cfg user.name x",
    "git -C sub config --file=../custom.cfg user.name x",
    "git -C sub archive -o bundle.tar HEAD",
    "git -C sub archive -o ../bundle.tar HEAD",
    "git -C . -C . archive -o bundle.tar HEAD",
    "git -C . status",
    "git -C sub status",
    "git -C sub log --oneline",
    # Added from the policy suite's resolver twin.
    "git -C subdir archive -o bundle.tar HEAD",
    "git -C subdir archive -o ../bundle.tar HEAD",
    "git -C subdir config --file=custom.cfg user.name x",
    "git -C subdir config --file=../custom.cfg user.name x",
    "git -C .git status",
    "git -C .git log",
)

# --- Unmodeled file commands on the write/mutate surface, resolver layer ---
# cpio, pax, bsdtar, ar, shred, and fallocate joined the inspected
# write/mutate surface: their write-classified operands get the same
# metadata containment as every other command. The metadata-guard suite's
# matrix is a strict superset of the policy suite's resolver twin matrix,
# so the shared matrix is the guard suite's list unchanged.
UNMODELED_FILE_DENIED_COMMANDS: tuple[str, ...] = (
    # cpio: the pass-through destination positional, the -D directory
    # copy-in changes into before extracting, and the copy-out archive
    # files are all write operands.
    "cpio -p .git/hooks",
    "cpio -p .git",
    "cpio --pass-through .git/hooks",
    "cpio -l .git/hooks",
    "printf 'pre-commit\\n' | cpio -p .git/hooks",
    "cpio -i -D .git",
    "cpio -i --directory=.git",
    "cpio -i -D .git/hooks",
    "cpio -o -F .git/config",
    "cpio -o --file=.git/config",
    "cpio -o -O .git/config",
    # shred and fallocate mutate their positional file in place.
    "shred .git/config",
    "shred -n 1 .git/config",
    "shred -u .git/config",
    "shred --remove .git/config",
    "fallocate -l 100 .git/config",
    "fallocate --length=100 .git/config",
    "fallocate -p -l 100 .git/config",
    # ar: the archive operand is created or replaced, and --output
    # names the extraction destination.
    "ar rcs .git/evil.a x.o",
    "ar rcs .git/hooks/pre-commit x.o",
    "ar x --output=.git lib.a",
    "ar x --output .git lib.a",
    # pax/bsdtar: tar-family archive operands and extraction
    # destinations.
    "pax -w -f .git/bundle.tar seed.txt",
    "pax -wf .git/bundle.tar seed.txt",
    "pax -r -f .git/config",
    "bsdtar -cf .git/bundle.tar seed.txt",
    "bsdtar -C .git -xf evil.tgz",
    "bsdtar --directory=.git -xf evil.tgz",
)

# --- Unmodeled file commands, benign forms, resolver layer ---
UNMODELED_FILE_BENIGN_COMMANDS: tuple[str, ...] = (
    "cpio -p outdir",
    "cpio -i -D outdir",
    "cpio -o -F out.cpio",
    "cpio -t",
    "shred seed.txt",
    "shred -u seed.txt",
    "fallocate -l 100 seed.txt",
    "ar rcs lib.a x.o",
    "ar t lib.a",
    "pax -w -f out.tar seed.txt",
    "pax -r -f out.tar",
    "bsdtar -cf out.tar seed.txt",
    "bsdtar -C outdir -xf good.tgz",
)
