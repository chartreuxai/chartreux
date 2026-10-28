"""The shell write-surface extraction (the submission secret-scan scope).

``shell_command_write_paths`` collects the paths a shell command writes so
the submission gate scans them alongside the file tools' touched paths.
Beyond the R6 surface (redirect targets, tee/mkdir/touch operands, git's raw
writes, the modeled network clients' and compilers' write options), the
destinations the mutating commands write themselves are collected too: a
cp/mv/install final positional, a dd of= output, and every sed -i input
file. The R9 surface adds the modeled clients' remaining writes: the write
options --libcurl and the HSTS options name, and the derivable download
destinations (curl's -O/--remote-name and a plain wget write the URL's
file part in the working directory). The R11 surface makes curl's download
derivation per URL (its output options pair with its URLs one-for-one) and
composes --output-dir with every relative -o operand, so the collected
write is the path curl actually writes. That is scan-scope coverage only —
the permission layer already scopes those operands as a whole, so no
permission decision changes — and it closes the copy-channel scan gap: a
file written through cp/mv/install/dd/sed -i or a client download now
reaches the submission scan exactly like a redirect or tee target does.
Reads, chmod/rm operands, and non-in-place sed stay uncollected. The R6
surface adds the remaining established-EMPTY write shapes: a sed w/W
script target (a write with or without -i; the r/R targets stay reads),
the archive operands (zip's first positional, 7z's archive operand after
an archive-rewriting command letter), the extraction and patching roots
(an unzip extraction, a patch application, a plain git apply — the cwd
tree root when no -d/-o/--directory/--output relocates it), and
objcopy's written-output positional. The destination-position tables
consume each command's value-taking options, so a trailing option value
(cp's -S .bak, install's -m 644) no longer displaces the true final
positional.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_permission_resolver import (
    shell_command_write_paths,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.utils import DEFAULT_SENSITIVE_PATTERNS, PathAuthority
from chartreux.core.workspace import Workspace


@pytest.fixture
def write_scope(tmp_path: Path) -> tuple[Path, Workspace, PathAuthority]:
    """A workspace rooted at a scratch directory, with the default authority."""
    cwd = tmp_path.resolve()
    workspace = Workspace.for_session(cwd)
    config = BashToolConfig()
    authority = PathAuthority(
        "bash",
        config.permission,
        tuple(config.allowlist),
        tuple(config.denylist),
        tuple(DEFAULT_SENSITIVE_PATTERNS),
        workspace,
    )
    return cwd, workspace, authority


def write_paths(
    command: str, scope: tuple[Path, Workspace, PathAuthority]
) -> tuple[list[str], bool]:
    cwd, workspace, authority = scope
    return shell_command_write_paths(
        command, cwd=cwd, workspace=workspace, authority=authority
    )


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # cp/mv destinations: the final positional is the write; every
        # earlier positional is a read and stays uncollected.
        ("cp src.txt dst.txt", ["dst.txt"]),
        ("cp -r src-dir dst-dir", ["dst-dir"]),
        ("cp a.txt b.txt dst-dir/", ["dst-dir"]),
        ("mv a.txt b.txt", ["b.txt"]),
        ("mv -f a.txt b.txt", ["b.txt"]),
        # With -t/--target-directory every positional is a source and the
        # option value is the write (collected through the option table).
        ("cp -t destdir src.txt", ["destdir"]),
        ("cp --target-directory destdir src.txt", ["destdir"]),
        ("cp --target-directory=destdir src.txt", ["destdir"]),
        ("mv -t destdir a.txt b.txt", ["destdir"]),
        # install shares cp/mv's destination shape: the final positional is
        # the write (-D only creates the destination's leading components),
        # unless -t/--target-directory names the destination directory.
        ("install src.txt dst.txt", ["dst.txt"]),
        ("install -D src.txt dst.txt", ["dst.txt"]),
        ("install a.txt b.txt dst-dir/", ["dst-dir"]),
        ("install -t destdir src.txt", ["destdir"]),
        ("install --target-directory destdir src.txt", ["destdir"]),
        ("install --target-directory=destdir src.txt", ["destdir"]),
        # dd parses no getopt options: its operands are native name=value
        # assignments. of= names the written output file; if= names the
        # read input and is not a write.
        ("dd if=src.img of=dst.img", ["dst.img"]),
        ("dd if=src.img of=dst.img bs=512 count=1", ["dst.img"]),
        ("dd of=dst.img", ["dst.img"]),
        # sed -i rewrites every input file in place. (The backup spelling's
        # derived file.txt.bak is an option-value write the R6 surface never
        # collected; only the in-place target is this fix's scope.)
        ("sed -i s/x/y/ file.txt", ["file.txt"]),
        ("sed --in-place s/x/y/ file.txt", ["file.txt"]),
        ("sed -i s/x/y/ a.txt b.txt", ["a.txt", "b.txt"]),
        ("sed -i.bak s/x/y/ file.txt", ["file.txt"]),
        ("gsed -i s/x/y/ file.txt", ["file.txt"]),
        ("sed -e s/x/y/ -i file.txt", ["file.txt"]),
    ],
)
def test_mutating_destinations_are_collected(
    write_scope: tuple[Path, Workspace, PathAuthority],
    command: str,
    expected: list[str],
) -> None:
    cwd, _workspace, _authority = write_scope
    paths, unknown = write_paths(command, write_scope)
    assert unknown is False
    assert paths == [str(cwd / name) for name in expected]


@pytest.mark.parametrize(
    "command",
    [
        # Reads and non-writing operands stay uncollected.
        "cat file.txt",
        "grep pattern file.txt",
        "sed s/x/y/ file.txt",
        "sed -n p file.txt",
        "chmod +x file.txt",
        "chown root file.txt",
        "rm file.txt",
        "rmdir somedir",
        # A cp/mv SOURCE is a read, not a write: only the destination is
        # collected.
        "cp src.txt dst.txt",
        # dd without of= writes to stdout: the if= input is a read.
        "dd if=file.txt",
    ],
)
def test_reads_and_non_writes_stay_uncollected(
    write_scope: tuple[Path, Workspace, PathAuthority], command: str
) -> None:
    cwd, _workspace, _authority = write_scope
    paths, unknown = write_paths(command, write_scope)
    assert unknown is False
    if command == "cp src.txt dst.txt":
        assert paths == [str(cwd / "dst.txt")]
    else:
        assert paths == []


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # The R6 surface is unchanged: redirects, tee, heredocs, and the
        # write-option commands keep collecting exactly what they did.
        ("echo x > out.txt", ["out.txt"]),
        ("echo x >> out.txt", ["out.txt"]),
        ("echo x > sub/dir/out.txt", ["sub/dir/out.txt"]),
        ("echo x | tee out.txt", ["out.txt"]),
        ("echo x | tee -a out.txt extra.txt", ["out.txt", "extra.txt"]),
        ("mkdir newdir", ["newdir"]),
        ("touch newfile.txt", ["newfile.txt"]),
        ("curl -o out.txt http://127.0.0.1:8080/x", ["out.txt"]),
        ("gcc -o prog main.c", ["prog"]),
        ("tar -cf archive.tar dir", ["archive.tar"]),
        ("git archive --output=bundle.tar HEAD", ["bundle.tar"]),
        ("cp src.txt dst.txt && echo y > second.txt", ["dst.txt", "second.txt"]),
    ],
)
def test_existing_write_surface_is_unchanged(
    write_scope: tuple[Path, Workspace, PathAuthority],
    command: str,
    expected: list[str],
) -> None:
    cwd, _workspace, _authority = write_scope
    paths, unknown = write_paths(command, write_scope)
    assert unknown is False
    assert paths == [str(cwd / name) for name in expected]


def test_unestablishable_write_surface_stays_fail_closed(
    write_scope: tuple[Path, Workspace, PathAuthority],
) -> None:
    # A glob-bearing write operand fails closed at the path-glob rule, so
    # the footprint reports unknown rather than a partial path set.
    paths, unknown = write_paths("cp src.txt 'dst*.txt'", write_scope)
    assert paths == []
    assert unknown is True
    paths, unknown = write_paths("echo x > 'glob*.txt'", write_scope)
    assert paths == []
    assert unknown is True


def test_install_dd_permission_decisions_are_unchanged(
    write_scope: tuple[Path, Workspace, PathAuthority],
) -> None:
    """Collecting install/dd destinations is scan-scope coverage only.

    The permission layer's decisions for install and dd are exactly what
    they were before the collection: in-workspace calls resolve run-class,
    and the positionals stay MUTATE-inspected — an operand reaching
    protected Git metadata is denied on containment, whether it is the
    install destination, a dd of= output, or a dd if= input.
    """
    cwd, _workspace, _authority = write_scope
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)
    for command in ("install a b", "install -t dir a b", "dd if=a of=b"):
        context = tool.resolve_permission(BashArgs(command=command))
        assert context is not None
        assert context.permission is ToolPermission.ALWAYS
        assert context.denial_kind is None
    for command in (
        "install a .git/config",
        "install -D a .git/hooks/pre-push",
        "dd of=.git/config",
        "dd if=.git/config of=b",
    ):
        context = tool.resolve_permission(BashArgs(command=command))
        assert context is not None
        assert context.permission is ToolPermission.NEVER
        assert "file operand reaches protected Git metadata" in (context.reason or "")


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # R9: the derivable download destinations join the write surface.
        # curl's -O/--remote-name writes the URL's file part (the path is
        # cut off) in the working directory; wget's default download does
        # the same, so the submission scan sees them exactly like an
        # explicit -o/-O operand.
        ("curl -O http://127.0.0.1:8080/x", ["x"]),
        ("curl --remote-name http://127.0.0.1:8080/x", ["x"]),
        ("curl --remote-name-all http://127.0.0.1:8080/x", ["x"]),
        ("curl -O http://127.0.0.1:8080/dir/file.tar.gz", ["file.tar.gz"]),
        ("curl -O --url http://127.0.0.1:8080/x", ["x"]),
        ("wget --max-redirect=0 http://127.0.0.1:8080/x", ["x"]),
        (
            "wget --max-redirect=0 http://127.0.0.1:8080/dir/file.tar.gz",
            ["file.tar.gz"],
        ),
        # The directory selectors relocate the derived file (and are
        # themselves collected as write operands).
        ("wget --max-redirect=0 -P sub http://127.0.0.1:8080/x", ["sub", "sub/x"]),
        (
            "wget --max-redirect=0 --directory-prefix sub http://127.0.0.1:8080/x",
            ["sub", "sub/x"],
        ),
        ("curl -O --output-dir sub http://127.0.0.1:8080/x", ["sub", "sub/x"]),
        # A directory URL names the default page: wget saves index.html
        # (or --default-page's value) in the working directory, while curl
        # fails before any write ("Remote file name has no length"), so
        # nothing is derived for it.
        ("wget --max-redirect=0 http://127.0.0.1:8080/", ["index.html"]),
        (
            "wget --max-redirect=0 --default-page home.html http://127.0.0.1:8080/",
            ["home.html"],
        ),
        ("curl -O http://127.0.0.1:8080/", []),
        # curl pairs its output options with its URLs one-for-one: the -O
        # pairs with the single URL and derives x, while the -o out.txt
        # operand is itself collected (real curl drops that extra output,
        # but the raw operand stays collected fail-closed); wget's -O
        # concatenates into its one file, and curl without -O writes to
        # stdout.
        ("curl -O -o out.txt http://127.0.0.1:8080/x", ["out.txt", "x"]),
        ("wget --max-redirect=0 -O out.txt http://127.0.0.1:8080/x", ["out.txt"]),
        ("curl http://127.0.0.1:8080/x", []),
        # The R9 write options join the surface too: --libcurl writes the
        # libcurl-equivalent source, and the HSTS options write the cache
        # they also read.
        ("curl --libcurl client.c http://127.0.0.1:8080/x", ["client.c"]),
        ("curl --libcurl=client.c http://127.0.0.1:8080/x", ["client.c"]),
        ("curl --hsts hsts-cache.txt http://127.0.0.1:8080/x", ["hsts-cache.txt"]),
        (
            "wget --max-redirect=0 --hsts-file hsts.txt http://127.0.0.1:8080/x",
            ["hsts.txt", "x"],
        ),
    ],
)
def test_network_client_writes_are_collected(
    write_scope: tuple[Path, Workspace, PathAuthority],
    command: str,
    expected: list[str],
) -> None:
    cwd, _workspace, _authority = write_scope
    paths, unknown = write_paths(command, write_scope)
    assert unknown is False
    assert paths == [str(cwd / name) for name in expected]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # R11 (deep-review round 5): curl's download derivation is per URL.
        # curl pairs its output options with its URLs one-for-one in
        # command-line order (curl's man: "the first -o is for the first URL
        # and so on"): a URL paired with -O writes its derived name, a URL
        # paired with -o writes that file, a paired "-o -" writes to stdout,
        # and only a URL with no paired output takes the --remote-name-all
        # default — so every derivable destination of a multi-URL
        # invocation is collected.
        (
            "curl --remote-name-all -o - http://127.0.0.1/first http://127.0.0.1/second.txt",
            ["-", "second.txt"],
        ),
        (
            "curl -O http://127.0.0.1/first.txt -o x http://127.0.0.1/second.txt",
            ["x", "first.txt"],
        ),
        # A plain -O is itself a per-URL slot, not a default: the second
        # URL has no paired output and no --remote-name-all, so it writes
        # to stdout and nothing is derived for it.
        (
            "curl -O http://127.0.0.1/first.txt http://127.0.0.1/second.txt",
            ["first.txt"],
        ),
        # A paired "-o -" writes to stdout; the raw operand is still
        # collected.
        ("curl -o - http://127.0.0.1:8080/x", ["-"]),
        # The URL list is command-line order: --url values interleave with
        # the positionals.
        (
            "curl --remote-name-all -o - --url http://127.0.0.1/first.txt "
            "http://127.0.0.1/second.txt",
            ["-", "second.txt"],
        ),
        (
            "curl -O --url http://127.0.0.1/first.txt -o x http://127.0.0.1/second.txt",
            ["x", "first.txt"],
        ),
        # Short clusters expand exactly like the scanner: "-Oo" is -O -o.
        ("curl -Oo out.txt http://127.0.0.1:8080/x", ["out.txt", "x"]),
        # --output-dir composes with every relative -o/--output operand:
        # the collected write is the composed path curl actually writes.
        # An absolute -o operand and "-" (stdout) are not composed.
        (
            "curl --output-dir sub -o out.txt http://127.0.0.1:8080/x",
            ["sub", "out.txt", "sub/out.txt"],
        ),
        (
            "curl --output-dir sub --output out.txt http://127.0.0.1:8080/x",
            ["sub", "out.txt", "sub/out.txt"],
        ),
        ("curl --output-dir sub -o - http://127.0.0.1:8080/x", ["sub", "-"]),
        (
            "curl --output-dir sub -o /tmp/out.txt http://127.0.0.1:8080/x",
            ["sub", "/tmp/out.txt"],
        ),
    ],
)
def test_r11_curl_per_url_download_writes_are_collected(
    write_scope: tuple[Path, Workspace, PathAuthority],
    command: str,
    expected: list[str],
) -> None:
    cwd, _workspace, _authority = write_scope
    paths, unknown = write_paths(command, write_scope)
    assert unknown is False
    assert paths == [str(cwd / name) for name in expected]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # A sed w/W script target is a write with or without -i, spelled as a
        # plain command, in a substitution's flag list, and under an address.
        ("sed -n 'w artifact.txt' input.txt", ["artifact.txt"]),
        ("sed 's/a/b/w out.txt' input.txt", ["out.txt"]),
        ("sed -i 'w out.txt' input.txt", ["out.txt", "input.txt"]),
        ("sed -n '2w first.txt' a.txt b.txt", ["first.txt"]),
        ("sed -n 'W second.txt' a.txt", ["second.txt"]),
        # An r/R target names a file sed only reads: it stays in the
        # inspected surface but is never a write.
        ("sed -n 'r header.txt' input.txt", []),
    ],
)
def test_sed_script_write_targets_are_collected(
    write_scope: tuple[Path, Workspace, PathAuthority],
    command: str,
    expected: list[str],
) -> None:
    cwd, _workspace, _authority = write_scope
    paths, unknown = write_paths(command, write_scope)
    assert unknown is False
    assert paths == [str(cwd / name) for name in expected]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # zip's first positional operand is the archive every invocation
        # creates, replaces, or rewrites (the tar -f treatment:
        # write-classified either way). Its value-taking options consume
        # their values and never displace the archive.
        ("zip archive.zip a.txt", ["archive.zip"]),
        ("zip -r archive.zip dir", ["archive.zip"]),
        ("zip -d archive.zip entry.txt", ["archive.zip"]),
        ("zip -P secret archive.zip a.txt", ["archive.zip"]),
        ("zip -b scratch archive.zip a.txt", ["archive.zip"]),
        # 7z's archive operand follows the command letter, and only the
        # archive-rewriting commands (a/d/u/rn) name it as a write: the
        # extracting, listing, and testing ones leave it unchanged.
        ("7z a archive.7z a.txt", ["archive.7z"]),
        ("7z a archive.7z", ["archive.7z"]),
        ("7z d archive.7z entry.txt", ["archive.7z"]),
        ("7z u archive.7z a.txt", ["archive.7z"]),
        ("7z rn archive.7z old new", ["archive.7z"]),
        ("7z l archive.7z", []),
        ("7z x archive.7z", []),
        ("7z t archive.7z", []),
    ],
)
def test_archive_operands_are_collected(
    write_scope: tuple[Path, Workspace, PathAuthority],
    command: str,
    expected: list[str],
) -> None:
    cwd, _workspace, _authority = write_scope
    paths, unknown = write_paths(command, write_scope)
    assert unknown is False
    assert paths == [str(cwd / name) for name in expected]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # unzip extracts member-named files the operands never spell: the
        # extraction root is derivable, and the scan walks a collected
        # directory root. Without -d the root is the working directory;
        # -d's value is collected through the write-operand table.
        ("unzip archive.zip", ["."]),
        ("unzip -o archive.zip", ["."]),
        ("unzip archive.zip -d outdir", ["outdir"]),
        ("unzip -d outdir archive.zip", ["outdir"]),
        # The listing, test, pipe, and comment modes write nothing outside
        # -d: no root is collected.
        ("unzip -l archive.zip", []),
        ("unzip -t archive.zip", []),
        ("unzip -p archive.zip member.txt", []),
        ("unzip -v archive.zip", []),
        ("unzip -z archive.zip", []),
        # patch writes the target files the diff names in place, under the
        # working directory; -o/-d's values are collected through the
        # write-operand table, and -r's reject file is collected with them.
        ("patch file.c -i fix.diff", ["."]),
        ("patch -i fix.diff", ["."]),
        ("patch -p1 file.c -i fix.diff", ["."]),
        ("patch -d subdir file.c -i fix.diff", ["subdir"]),
        ("patch -o out.c file.c -i fix.diff", ["out.c"]),
        # -r's reject file is collected through the write-operand table, and
        # the in-place patching under the cwd root is collected with it.
        ("patch -r rejects.txt file.c -i fix.diff", ["rejects.txt", "."]),
        # A plain git apply patches the paths the diff names, relative to
        # the current directory: the cwd tree root is the write root.
        # --directory relocates the root and --output redirects the result;
        # both values are collected through git's write-path values.
        ("git apply fix.diff", ["."]),
        ("git apply --directory=sub fix.diff", ["sub"]),
        ("git apply --output=out.c fix.diff", ["out.c"]),
        # git's own -C sequence relocates the cwd the cwd-root token (and
        # the --output/--directory values) resolve against.
        ("git -C sub apply fix.diff", ["sub"]),
        ("git -C sub apply --output=out.c fix.diff", ["sub/out.c"]),
        # objcopy's written output is the cp-destination analog: the final
        # positional is the write, and a lone positional is the
        # in-place-rewritten input. The format options (-O binary, ...)
        # consume their values and never displace it.
        ("objcopy input.bin output.bin", ["output.bin"]),
        ("objcopy input.bin", ["input.bin"]),
        ("objcopy -O binary input.bin output.bin", ["output.bin"]),
        ("objcopy input.bin output.bin -O binary", ["output.bin"]),
    ],
)
def test_derivable_write_roots_are_collected(
    write_scope: tuple[Path, Workspace, PathAuthority],
    command: str,
    expected: list[str],
) -> None:
    cwd, _workspace, _authority = write_scope
    paths, unknown = write_paths(command, write_scope)
    assert unknown is False
    assert paths == [str(cwd / name) for name in expected]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # A value-taking option spelled AFTER the operands consumes its
        # value: the true final positional stays the collected destination.
        ("cp src dst -S .bak", ["dst"]),
        ("cp -S .bak src dst", ["dst"]),
        ("mv src dst -S .bak", ["dst"]),
        ("install src dst -m 644", ["dst"]),
        ("install -D src dst -m 644", ["dst"]),
        ("install src dst -o root -g root", ["dst"]),
        # Options before the operands keep collecting the destination.
        ("cp src dst", ["dst"]),
        ("install src dst", ["dst"]),
        # A long option that takes no separate token (--backup, --context)
        # never displaces anything in either position.
        ("cp src dst --backup", ["dst"]),
        ("cp --backup src dst", ["dst"]),
    ],
)
def test_trailing_option_values_do_not_displace_the_destination(
    write_scope: tuple[Path, Workspace, PathAuthority],
    command: str,
    expected: list[str],
) -> None:
    cwd, _workspace, _authority = write_scope
    paths, unknown = write_paths(command, write_scope)
    assert unknown is False
    assert paths == [str(cwd / name) for name in expected]


def test_archive_and_patch_permission_decisions_are_unchanged(
    write_scope: tuple[Path, Workspace, PathAuthority],
) -> None:
    """Collecting the archive operands and write roots is scan-scope only.

    The permission layer's decisions are exactly what they were before the
    collection: in-workspace calls resolve run-class, and the operands stay
    MUTATE-inspected — an operand reaching protected Git metadata is denied
    on containment whether it is an archive operand, an unzip -d root, or a
    sed script target.
    """
    cwd, _workspace, _authority = write_scope
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)
    for command in (
        "zip archive.zip a.txt",
        "7z a archive.7z a.txt",
        "unzip archive.zip",
        "unzip archive.zip -d outdir",
        "patch file.c -i fix.diff",
        "git apply fix.diff",
        "objcopy input.bin output.bin",
        "sed -n 'w out.txt' input.txt",
    ):
        context = tool.resolve_permission(BashArgs(command=command))
        assert context is not None
        assert context.permission is ToolPermission.ALWAYS
        assert context.denial_kind is None
    for command in (
        "zip .git/config archive.zip",
        "7z a .git/config archive.7z",
        "unzip archive.zip -d .git",
        "sed -n 'w .git/config' input.txt",
    ):
        context = tool.resolve_permission(BashArgs(command=command))
        assert context is not None
        assert context.permission is ToolPermission.NEVER
