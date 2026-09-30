from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import threading

import pytest

from chartreux.core.tools.base import BaseToolState, ToolError, ToolPermission
from chartreux.core.tools.builtins import grep as grep_module
from chartreux.core.tools.builtins.grep import (
    Grep,
    GrepArgs,
    GrepBackend,
    GrepResult,
    GrepToolConfig,
)
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.permissions import PermissionContext
from chartreux.utils import io as io_utils
from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result


@pytest.fixture
def grep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = GrepToolConfig()
    return Grep(config_getter=lambda: config, state=BaseToolState())


@pytest.fixture
def grep_gnu_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original_which = shutil.which

    def mock_which(cmd):
        if cmd == "rg":
            return None
        return original_which(cmd)

    monkeypatch.setattr("shutil.which", mock_which)
    config = GrepToolConfig()
    return Grep(config_getter=lambda: config, state=BaseToolState())


@pytest.mark.asyncio
async def test_cancellation_kills_search_subprocess(grep, monkeypatch):
    class SlowProcess:
        returncode = None

        async def communicate(self):
            await asyncio.Event().wait()

    process = SlowProcess()
    cleanup_calls = []

    async def create_subprocess(*_args, **_kwargs):
        return process

    async def kill_subprocess(proc, *, kill_process_group):
        cleanup_calls.append((proc, kill_process_group))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(
        "chartreux.core.tools.builtins.grep.kill_async_subprocess", kill_subprocess
    )

    task = asyncio.create_task(grep._execute_search(["grep", "needle"]))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleanup_calls == [(process, False)]


def test_detects_ripgrep_when_available(grep):
    if shutil.which("rg"):
        assert grep._detect_backend() == GrepBackend.RIPGREP


def test_falls_back_to_gnu_grep(grep, monkeypatch):
    original_which = shutil.which

    def mock_which(cmd):
        if cmd == "rg":
            return None
        return original_which(cmd)

    monkeypatch.setattr("shutil.which", mock_which)

    if shutil.which("grep"):
        assert grep._detect_backend() == GrepBackend.GNU_GREP


def test_raises_error_if_no_grep_available(grep, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)

    with pytest.raises(ToolError) as err:
        grep._detect_backend()

    assert "Neither ripgrep (rg) nor grep is installed" in str(err.value)


@pytest.mark.asyncio
async def test_finds_pattern_in_file(grep, tmp_path):
    (tmp_path / "test.py").write_text("def hello():\n    print('world')\n")

    result = await collect_result(grep.run(GrepArgs(pattern="hello")))

    assert result.match_count == 1
    assert "hello" in result.matches
    assert "test.py" in result.matches
    assert not result.was_truncated


@pytest.mark.asyncio
async def test_finds_multiple_matches(grep, tmp_path):
    (tmp_path / "test.py").write_text("foo\nbar\nfoo\nbaz\nfoo\n")

    result = await collect_result(grep.run(GrepArgs(pattern="foo")))

    assert result.match_count == 3
    assert result.matches.count("foo") == 3
    assert not result.was_truncated


@pytest.mark.asyncio
async def test_returns_empty_on_no_matches(grep, tmp_path):
    (tmp_path / "test.py").write_text("def hello():\n    pass\n")

    result = await collect_result(grep.run(GrepArgs(pattern="nonexistent")))

    assert result.match_count == 0
    assert result.matches == ""
    assert not result.was_truncated


@pytest.mark.asyncio
async def test_preserves_accents_when_matching_latin1_encoded_file(
    grep, tmp_path, monkeypatch
):
    # Pin a UTF-8 locale (production reality on Linux CI) and a deterministic
    # charset_normalizer result. Without this, decode_safe falls back to
    # charset_normalizer's heuristic, which is unreliable for the single
    # non-ASCII byte ripgrep emits — it can misdetect (e.g. cp1006, which
    # decodes \xe9 to ﻠ instead of é) depending on the platform wheel.
    monkeypatch.setattr(
        io_utils.locale, "getpreferredencoding", lambda _do_setlocale: "utf-8"
    )
    monkeypatch.setattr(io_utils, "_encoding_from_best_match", lambda _raw: "cp1252")
    (tmp_path / "menu.txt").write_bytes("café au lait\nthé glacé\n".encode("latin-1"))

    result = await collect_result(
        grep.run(GrepArgs(pattern="caf"))  # typos:disable-line
    )

    assert result.match_count == 1
    assert "\ufffd" not in result.matches
    assert "café au lait" in result.matches


@pytest.mark.asyncio
async def test_fails_with_empty_pattern(grep):
    with pytest.raises(ToolError) as err:
        await collect_result(grep.run(GrepArgs(pattern="")))

    assert "Empty search pattern" in str(err.value)


@pytest.mark.asyncio
async def test_fails_with_nonexistent_path(grep):
    with pytest.raises(ToolError) as err:
        await collect_result(grep.run(GrepArgs(pattern="test", path="nonexistent")))

    assert "Path does not exist" in str(err.value)


@pytest.mark.asyncio
async def test_searches_in_specific_path(grep, tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "test.py").write_text("match here\n")
    (tmp_path / "other.py").write_text("match here too\n")

    result = await collect_result(grep.run(GrepArgs(pattern="match", path="subdir")))

    assert result.match_count == 1
    assert "subdir" in result.matches and "test.py" in result.matches
    assert "other.py" not in result.matches


@pytest.mark.asyncio
async def test_truncates_to_max_matches(grep, tmp_path):
    (tmp_path / "test.py").write_text("\n".join(f"line {i}" for i in range(200)))

    result = await collect_result(grep.run(GrepArgs(pattern="line", max_matches=50)))

    assert result.match_count == 50
    assert result.was_truncated


@pytest.mark.asyncio
async def test_truncates_to_max_output_bytes(grep, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = GrepToolConfig(max_output_bytes=100)
    grep_tool = Grep(config_getter=lambda: config, state=BaseToolState())
    (tmp_path / "test.py").write_text("\n".join("x" * 100 for _ in range(10)))

    result = await collect_result(grep_tool.run(GrepArgs(pattern="x")))

    assert len(result.matches) <= 100
    assert result.was_truncated


@pytest.mark.asyncio
async def test_respects_default_ignore_patterns(grep, tmp_path):
    (tmp_path / "included.py").write_text("match\n")
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "excluded.js").write_text("match\n")

    result = await collect_result(grep.run(GrepArgs(pattern="match")))

    assert "included.py" in result.matches
    assert "excluded.js" not in result.matches


@pytest.mark.asyncio
async def test_broad_search_does_not_leak_sensitive_file_contents(grep, tmp_path):
    (tmp_path / ".env").write_text("SECRET_TOKEN=supersecret\n")
    (tmp_path / "app.py").write_text("uses SECRET_TOKEN here\n")

    result = await collect_result(grep.run(GrepArgs(pattern="SECRET_TOKEN", path=".")))

    assert "app.py" in result.matches
    assert "supersecret" not in result.matches


@pytest.mark.asyncio
async def test_respects_vibeignore_file(grep, tmp_path):
    (tmp_path / ".vibeignore").write_text("custom_dir/\n*.tmp\n")
    custom_dir = tmp_path / "custom_dir"
    custom_dir.mkdir()
    (custom_dir / "excluded.py").write_text("match\n")
    (tmp_path / "excluded.tmp").write_text("match\n")
    (tmp_path / "included.py").write_text("match\n")

    result = await collect_result(grep.run(GrepArgs(pattern="match")))

    assert "included.py" in result.matches
    assert "excluded.py" not in result.matches
    assert "excluded.tmp" not in result.matches


@pytest.mark.asyncio
async def test_ignores_comments_in_vibeignore(grep, tmp_path):
    (tmp_path / ".vibeignore").write_text("# comment\npattern/\n# another comment\n")
    (tmp_path / "file.py").write_text("match\n")

    result = await collect_result(grep.run(GrepArgs(pattern="match")))

    assert result.match_count >= 1


@pytest.mark.asyncio
async def test_uses_effective_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = GrepToolConfig()
    grep_tool = Grep(config_getter=lambda: config, state=BaseToolState())
    (tmp_path / "test.py").write_text("match\n")

    result = await collect_result(grep_tool.run(GrepArgs(pattern="match", path=".")))

    assert result.match_count == 1
    assert "test.py" in result.matches


@pytest.mark.asyncio
async def test_single_file_match_includes_filename_in_output(grep, tmp_path):
    # Without --with-filename / -H, rg and grep omit the filename when
    # searching a single file, causing GrepMatch.from_output_line to
    # misinterpret the line number as a path. See VIBE-2772.
    (tmp_path / "only.py").write_text("hit one\nnope\nhit two\n")

    result = await collect_result(grep.run(GrepArgs(pattern="hit", path="only.py")))

    assert result.match_count == 2
    for parsed in result.parsed_matches:
        assert parsed.path.endswith("only.py")
        assert parsed.line is not None


@pytest.mark.asyncio
async def test_parsed_match_paths_anchor_on_search_cwd_not_process_cwd(
    tmp_path, monkeypatch
):
    # rg/grep emit paths relative to the search cwd; parsed_matches must anchor
    # them on the tool's cwd, not the process cwd. These differ when the agent
    # is launched from a directory other than the workspace (e.g. `uv run`).
    search_dir = tmp_path / "workspace"
    search_dir.mkdir()
    (search_dir / "target.py").write_text("NEEDLE\n")

    process_dir = tmp_path / "elsewhere"
    process_dir.mkdir()
    monkeypatch.chdir(process_dir)

    config = GrepToolConfig()
    grep_tool = Grep(
        config_getter=lambda: config, state=BaseToolState(), cwd=search_dir
    )

    result = await collect_result(grep_tool.run(GrepArgs(pattern="NEEDLE", path=".")))

    assert result.match_count == 1
    parsed = result.parsed_matches
    assert len(parsed) == 1
    assert parsed[0].path == str((search_dir / "target.py").resolve())


def test_cwd_is_not_serialized_into_the_model_facing_result():
    result = GrepResult(
        matches="target.py:1:NEEDLE",
        match_count=1,
        pattern="NEEDLE",
        was_truncated=False,
        cwd="/private/workspace",
    )

    dumped = result.model_dump(mode="json")
    result_text = "\n".join(f"{key}: {value}" for key, value in dumped.items())

    assert "cwd" not in dumped
    assert "/private/workspace" not in result_text
    assert result.parsed_matches[0].path == str(
        (Path("/private/workspace") / "target.py").resolve()
    )


class TestCollectExcludePatterns:
    def _grep(self, tmp_path, monkeypatch, **config_kwargs):
        monkeypatch.chdir(tmp_path)
        config = GrepToolConfig(**config_kwargs)
        return Grep(config_getter=lambda: config, state=BaseToolState())

    def test_configured_exclude_patterns_preserved(self, tmp_path, monkeypatch):
        grep = self._grep(tmp_path, monkeypatch)
        patterns = grep._collect_exclude_patterns()
        assert "node_modules/" in patterns
        assert ".git/" in patterns

    def test_sensitive_patterns_not_added_as_cli_excludes(self, tmp_path, monkeypatch):
        # Sensitive files are enforced by filtering output, not CLI excludes: a
        # case-sensitive basename glob would miss `.ENV`/`.Env`, and a path glob
        # like `**/secrets/**` would collapse to a meaningless `**` exclude.
        grep = self._grep(
            tmp_path, monkeypatch, sensitive_patterns=["**/.env", "**/secrets/**"]
        )
        patterns = grep._collect_exclude_patterns()
        assert ".env" not in patterns
        assert "**" not in patterns

    def test_vibeignore_patterns_still_collected(self, tmp_path, monkeypatch):
        (tmp_path / ".vibeignore").write_text("custom_dir/\n*.tmp\n")
        grep = self._grep(tmp_path, monkeypatch)
        patterns = grep._collect_exclude_patterns()
        assert "custom_dir/" in patterns
        assert "*.tmp" in patterns

    def test_chartreuxignore_takes_precedence_over_legacy_fallback(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / ".chartreuxignore").write_text("current/\n")
        (tmp_path / ".vibeignore").write_text("legacy/\n")
        grep = self._grep(tmp_path, monkeypatch)

        patterns = grep._collect_exclude_patterns()

        assert "current/" in patterns
        assert "legacy/" not in patterns

    def test_explicit_default_codeignore_does_not_load_legacy_fallback(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / ".vibeignore").write_text("legacy/\n")
        grep = self._grep(tmp_path, monkeypatch, codeignore_file=".chartreuxignore")

        patterns = grep._collect_exclude_patterns()

        assert "legacy/" not in patterns

    def test_unset_codeignore_uses_legacy_fallback(self, tmp_path, monkeypatch):
        (tmp_path / ".vibeignore").write_text("legacy/\n")
        grep = self._grep(tmp_path, monkeypatch)

        patterns = grep._collect_exclude_patterns()

        assert "legacy/" in patterns

    def test_configured_codeignore_overrides_default_and_legacy_files(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / ".chartreuxignore").write_text("current/\n")
        (tmp_path / ".vibeignore").write_text("legacy/\n")
        (tmp_path / ".customignore").write_text("configured/\n")
        grep = self._grep(tmp_path, monkeypatch, codeignore_file=".customignore")

        patterns = grep._collect_exclude_patterns()

        assert "configured/" in patterns
        assert "current/" not in patterns
        assert "legacy/" not in patterns


class TestDropSensitiveMatches:
    def _grep(self, tmp_path, monkeypatch, **config_kwargs):
        monkeypatch.chdir(tmp_path)
        config = GrepToolConfig(**config_kwargs)
        return Grep(config_getter=lambda: config, state=BaseToolState())

    def test_drops_lowercase_sensitive_file(self, tmp_path, monkeypatch):
        grep = self._grep(tmp_path, monkeypatch)
        lines = ["app.py:1:x", ".env:1:SECRET=1"]
        assert grep._drop_sensitive_matches(lines) == ["app.py:1:x"]

    def test_drops_case_variant_sensitive_files(self, tmp_path, monkeypatch):
        grep = self._grep(tmp_path, monkeypatch)
        lines = ["app.py:1:x", ".ENV:1:SECRET=1", "sub/.Env:2:SECRET=2"]
        assert grep._drop_sensitive_matches(lines) == ["app.py:1:x"]

    def test_drops_path_glob_sensitive_matches(self, tmp_path, monkeypatch):
        grep = self._grep(tmp_path, monkeypatch, sensitive_patterns=["**/secrets/**"])
        lines = ["app.py:1:x", "secrets/token.txt:1:abc"]
        assert grep._drop_sensitive_matches(lines) == ["app.py:1:x"]

    def test_keeps_all_when_no_sensitive_patterns(self, tmp_path, monkeypatch):
        grep = self._grep(tmp_path, monkeypatch, sensitive_patterns=[])
        lines = ["app.py:1:x", ".env:1:SECRET=1"]
        assert grep._drop_sensitive_matches(lines) == lines


@pytest.mark.parametrize(
    ("lines", "expected", "checks", "truncated"),
    [
        (["a.py:1:a", "bad line", ".env:1:secret"], ["a.py:1:a"], 2, False),
        (
            [".env:1:secret", "bad line", "b.py:2:b", "a.py:1:a", ".env:2:secret"],
            ["b.py:2:b", "a.py:1:a"],
            4,
            False,
        ),
        (
            [
                ".env:1:secret",
                "bad line",
                "b.py:2:b",
                "a.py:1:a",
                ".env:2:secret",
                "c.py:3:c",
                ".env:3:unchecked",
                "d.py:4:unchecked",
            ],
            ["b.py:2:b", "a.py:1:a"],
            5,
            True,
        ),
    ],
    ids=["n-minus-one-denied-tail", "exact-n-denied-tail", "n-plus-one-early-stop"],
)
def test_output_authorizes_through_allowed_lookahead(
    grep, monkeypatch, lines, expected, checks, truncated
):
    seen = []

    def can_read(path):
        seen.append(path.name)
        return path.name != ".env"

    monkeypatch.setattr(grep, "_can_read", can_read)
    result = grep._parse_output("\n".join(lines), max_matches=2)

    assert result.matches == "\n".join(expected)
    assert result.match_count == len(expected)
    assert result.was_truncated is truncated
    assert len(seen) == checks
    assert seen == [line.split(":", 1)[0] for line in lines if ":" in line][:checks]


def test_output_size_cap_counts_characters_after_match_selection(grep, monkeypatch):
    monkeypatch.setattr(grep.config, "max_output_bytes", 10)
    result = grep._parse_output("a.py:1:ééé\nb.py:2:ok", max_matches=2)

    assert result.matches == "a.py:1:ééé"
    assert len(result.matches.encode()) > 10
    assert result.match_count == 2
    assert result.was_truncated


def test_output_size_cap_does_not_truncate_at_exact_character_length(grep, monkeypatch):
    monkeypatch.setattr(grep.config, "max_output_bytes", len("a.py:1:é"))
    result = grep._parse_output("a.py:1:é", max_matches=1)

    assert result.matches == "a.py:1:é"
    assert result.match_count == 1
    assert not result.was_truncated


@pytest.mark.parametrize(
    "fixture_name, executable", [("grep", "rg"), ("grep_gnu_only", "grep")]
)
@pytest.mark.parametrize("limit", [None, 0, -1, 1, 2, 3])
@pytest.mark.asyncio
async def test_match_limit_on_both_backends(
    request, fixture_name, executable, limit, tmp_path
):
    if not shutil.which(executable):
        pytest.skip(f"{executable} not available")
    tool = request.getfixturevalue(fixture_name)
    (tmp_path / "test.py").write_text("hit one\nhit two\nhit three\n")
    tool.config.default_max_matches = 2

    result = await collect_result(tool.run(GrepArgs(pattern="hit", max_matches=limit)))

    expected_count = 1 if limit == 1 else 3 if limit == 3 else 2
    assert result.matches.splitlines() == [
        f"{tmp_path / 'test.py'}:{index}:hit {word}"
        for index, word in enumerate(("one", "two", "three")[:expected_count], 1)
    ]
    assert result.match_count == expected_count
    assert result.was_truncated is (expected_count < 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["_collect_paths", "_parse_frozen", "_prepare_batch"])
async def test_worker_stage_yields_loop_and_runs_on_other_thread(
    grep, tmp_path, monkeypatch, stage
):
    (tmp_path / "file.py").write_text("hit\n")
    entered = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()
    seen = []
    original = getattr(grep_module, stage)

    def gated(*args):
        seen.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(grep_module, stage, gated)
    task = asyncio.create_task(collect_result(grep.run(GrepArgs(pattern="hit"))))
    try:
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), 6)
        progressed = asyncio.Event()
        asyncio.get_running_loop().call_soon(progressed.set)
        await asyncio.wait_for(progressed.wait(), 1)
        assert seen == [seen[0]] and seen[0] != loop_thread
    finally:
        release.set()
    assert (await task).match_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["_collect_paths", "_parse_frozen"])
async def test_cancellation_during_worker_never_publishes_or_spawns_later(
    grep, tmp_path, monkeypatch, stage
):
    (tmp_path / "file.py").write_text("hit\n")
    entered = threading.Event()
    release = threading.Event()
    original = getattr(grep_module, stage)
    spawns = []
    execute = grep._execute_search

    async def record(cmd):
        spawns.append(cmd)
        return await execute(cmd)

    def gated(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(grep, "_execute_search", record)
    monkeypatch.setattr(grep_module, stage, gated)
    task = asyncio.create_task(collect_result(grep.run(GrepArgs(pattern="hit"))))
    try:
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), 6)
        count = len(spawns)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    await asyncio.sleep(0)
    assert len(spawns) == count


@pytest.mark.asyncio
async def test_authority_changes_at_publication_restart_instead_of_underfilling(
    grep, tmp_path, monkeypatch
):
    (tmp_path / "first.py").write_text("hit\n")
    (tmp_path / "second.py").write_text("hit\n")
    original = grep._verify_paths
    calls = 0
    config = grep.config

    async def flip(paths, token, *, regular=False):
        nonlocal calls
        calls += 1
        if not regular and calls == 2:
            config.denylist.append(str(tmp_path / "first.py"))
        return await original(paths, token, regular=regular)

    monkeypatch.setattr(grep, "_verify_paths", flip)
    result = await collect_result(grep.run(GrepArgs(pattern="hit", max_matches=1)))
    assert "first.py" not in result.matches
    assert "second.py" in result.matches
    assert calls >= 3


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["verification", "spawn", "parsing"])
async def test_authority_change_discards_stale_run_at_boundaries(
    grep, tmp_path, monkeypatch, boundary
):
    (tmp_path / "first.py").write_text("hit\n")
    (tmp_path / "second.py").write_text("hit\n")
    config = grep.config
    original_verify = grep._verify_paths
    original_current = grep._ensure_current
    original_parse = grep_module._parse_frozen
    changed = False

    def deny_first():
        nonlocal changed
        if not changed:
            changed = True
            config.denylist.append(str(tmp_path / "first.py"))

    async def verify(paths, token, *, regular=False):
        if boundary == "verification" and regular and not changed:
            # Change across a cooperative verification yield.
            asyncio.get_running_loop().call_soon(deny_first)
            await asyncio.sleep(0)
        return await original_verify(paths, token, regular=regular)

    def current(token):
        if boundary == "spawn" and not changed and getattr(grep, "_after_batch", False):
            deny_first()
        return original_current(token)

    if boundary == "spawn":

        async def mark_batch(paths, token, *, regular=False):
            result = await verify(paths, token, regular=regular)
            if regular:
                grep._after_batch = True
            return result

        monkeypatch.setattr(grep, "_verify_paths", mark_batch)
        monkeypatch.setattr(grep, "_ensure_current", current)
    elif boundary == "parsing":
        # Signal on the loop as the parse stage returns, before publication.
        original_stage = grep._stage

        async def stage(fn, *values, stop):
            result = await original_stage(fn, *values, stop=stop)
            if fn is original_parse and not changed:
                deny_first()
            return result

        monkeypatch.setattr(grep, "_stage", stage)
    else:
        monkeypatch.setattr(grep, "_verify_paths", verify)
    result = await collect_result(grep.run(GrepArgs(pattern="hit", max_matches=1)))
    assert changed
    assert "first.py" not in result.matches
    assert "second.py" in result.matches


@pytest.mark.asyncio
async def test_run_cancellation_during_subprocess_prevents_next_spawn_and_publication(
    grep, tmp_path, monkeypatch
):
    (tmp_path / "one.py").write_text("hit\n")
    started = asyncio.Event()
    cleanup = []
    spawns = []

    class SlowProcess:
        returncode = None

        async def communicate(self):
            started.set()
            await asyncio.Event().wait()

    async def spawn(*args, **kwargs):
        spawns.append(args)
        return SlowProcess()

    async def kill(proc, *, kill_process_group):
        cleanup.append(proc)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(grep_module, "kill_async_subprocess", kill)
    task = asyncio.create_task(collect_result(grep.run(GrepArgs(pattern="hit"))))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert len(spawns) == 1
    assert len(cleanup) == 1


@pytest.mark.asyncio
async def test_authority_change_while_worker_parses_restarts(
    tmp_path, grep, monkeypatch
):
    (tmp_path / "first.py").write_text("hit\n")
    (tmp_path / "second.py").write_text("hit\n")
    entered = threading.Event()
    release = threading.Event()
    original = grep_module._parse_frozen
    calls = 0

    def gated(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(grep_module, "_parse_frozen", gated)
    task = asyncio.create_task(
        collect_result(grep.run(GrepArgs(pattern="hit", max_matches=1)))
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), 6)
        grep.config.denylist.append(str(tmp_path / "first.py"))
    finally:
        release.set()
    result = await task
    assert calls == 2
    assert "second.py" in result.matches
    assert "first.py" not in result.matches


@pytest.mark.asyncio
async def test_custom_resolver_uses_live_fallback_not_generic_grep(tmp_path):
    (tmp_path / "blocked.py").write_text("hit\n")
    (tmp_path / "allowed.py").write_text("hit\n")

    class CustomGrep(Grep):
        def resolve_permission(self, args):
            if str(args.path).endswith("blocked.py"):
                return PermissionContext(permission=ToolPermission.NEVER)
            return super().resolve_permission(args)

    tool = CustomGrep(config_getter=GrepToolConfig, state=BaseToolState(), cwd=tmp_path)
    token, frozen, _ = tool._snapshot()
    assert token is None and frozen is None
    result = await collect_result(tool.run(GrepArgs(pattern="hit")))
    assert "allowed.py" in result.matches
    assert "blocked.py" not in result.matches


@pytest.mark.asyncio
async def test_custom_ancestor_resolver_is_never_replaced_by_generic_snapshot(tmp_path):
    blocked = tmp_path / "guarded.py"
    allowed = tmp_path / "allowed.py"
    blocked.write_text("hit\n")
    allowed.write_text("hit\n")

    class GuardedGrep(Grep):
        selection_priority = 10

        @classmethod
        def get_name(cls):
            return "grep"

        def resolve_permission(self, args):
            if str(args.path).endswith("guarded.py"):
                return PermissionContext(permission=ToolPermission.NEVER)
            return super().resolve_permission(args)

    config = build_test_vibe_config()
    grandparent = ToolManager(
        lambda: config,
        cwd=tmp_path,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
    )
    grandparent._register_discovered_tool_variant(GuardedGrep, is_custom=True)
    parent = ToolManager(
        lambda: config,
        cwd=tmp_path,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
        parent_authority_getter=lambda: grandparent,
    )
    child = ToolManager(
        lambda: config,
        cwd=tmp_path,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
        parent_authority_getter=lambda: parent,
    )
    tool = child.get("grep")
    assert isinstance(tool, Grep)
    _, frozen, _ = tool._snapshot()
    assert frozen is None
    result = await collect_result(tool.run(GrepArgs(pattern="hit")))
    assert "allowed.py" in result.matches
    assert "guarded.py" not in result.matches


@pytest.mark.asyncio
async def test_frozen_authority_preserves_two_ancestor_denials(tmp_path):
    denied_parent = tmp_path / "parent.py"
    denied_grandparent = tmp_path / "grandparent.py"
    allowed = tmp_path / "allowed.py"
    for path in (denied_parent, denied_grandparent, allowed):
        path.write_text("hit\n")

    def manager(denied=(), parent=None):
        config = build_test_vibe_config(
            tools={"grep": {"denylist": [str(p) for p in denied]}}
        )
        return ToolManager(
            lambda: config,
            cwd=tmp_path,
            defer_mcp=True,
            accepted_token_getter=lambda: "accepted",
            parent_authority_getter=(lambda: parent) if parent else None,
        )

    grandparent = manager((denied_grandparent,))
    parent = manager((denied_parent,), grandparent)
    child = manager(parent=parent)
    tool = child.get("grep")
    assert isinstance(tool, Grep)
    token, frozen, _ = tool._snapshot()
    assert token is not None and frozen is not None
    assert not frozen(denied_parent)
    assert not frozen(denied_grandparent)
    assert frozen(allowed)
    result = await collect_result(tool.run(GrepArgs(pattern="hit")))
    assert result.match_count == 1
    assert "allowed.py" in result.matches


@pytest.mark.skipif(not shutil.which("grep"), reason="GNU grep not available")
class TestGnuGrepBackend:
    @pytest.mark.asyncio
    async def test_finds_pattern_in_file(self, grep_gnu_only, tmp_path):
        (tmp_path / "test.py").write_text("def hello():\n    print('world')\n")

        result = await collect_result(grep_gnu_only.run(GrepArgs(pattern="hello")))

        assert result.match_count == 1
        assert "hello" in result.matches
        assert "test.py" in result.matches

    @pytest.mark.asyncio
    async def test_finds_multiple_matches(self, grep_gnu_only, tmp_path):
        (tmp_path / "test.py").write_text("foo\nbar\nfoo\nbaz\nfoo\n")

        result = await collect_result(grep_gnu_only.run(GrepArgs(pattern="foo")))

        assert result.match_count == 3
        assert result.matches.count("foo") == 3

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_matches(self, grep_gnu_only, tmp_path):
        (tmp_path / "test.py").write_text("def hello():\n    pass\n")

        result = await collect_result(
            grep_gnu_only.run(GrepArgs(pattern="nonexistent"))
        )

        assert result.match_count == 0
        assert result.matches == ""

    @pytest.mark.asyncio
    async def test_case_insensitive_for_lowercase_pattern(
        self, grep_gnu_only, tmp_path
    ):
        (tmp_path / "test.py").write_text("Hello\nHELLO\nhello\n")

        result = await collect_result(grep_gnu_only.run(GrepArgs(pattern="hello")))

        assert result.match_count == 3

    @pytest.mark.asyncio
    async def test_case_sensitive_for_mixed_case_pattern(self, grep_gnu_only, tmp_path):
        (tmp_path / "test.py").write_text("Hello\nHELLO\nhello\n")

        result = await collect_result(grep_gnu_only.run(GrepArgs(pattern="Hello")))

        assert result.match_count == 1

    @pytest.mark.asyncio
    async def test_respects_exclude_patterns(self, grep_gnu_only, tmp_path):
        (tmp_path / "included.py").write_text("match\n")
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "excluded.js").write_text("match\n")

        result = await collect_result(grep_gnu_only.run(GrepArgs(pattern="match")))

        assert "included.py" in result.matches
        assert "excluded.js" not in result.matches

    @pytest.mark.asyncio
    async def test_searches_in_specific_path(self, grep_gnu_only, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "test.py").write_text("match here\n")
        (tmp_path / "other.py").write_text("match here too\n")

        result = await collect_result(
            grep_gnu_only.run(GrepArgs(pattern="match", path="subdir"))
        )

        assert result.match_count == 1
        assert "other.py" not in result.matches

    @pytest.mark.asyncio
    async def test_respects_vibeignore_file(self, grep_gnu_only, tmp_path):
        (tmp_path / ".vibeignore").write_text("custom_dir/\n*.tmp\n")
        custom_dir = tmp_path / "custom_dir"
        custom_dir.mkdir()
        (custom_dir / "excluded.py").write_text("match\n")
        (tmp_path / "excluded.tmp").write_text("match\n")
        (tmp_path / "included.py").write_text("match\n")

        result = await collect_result(grep_gnu_only.run(GrepArgs(pattern="match")))

        assert "included.py" in result.matches
        assert "excluded.py" not in result.matches
        assert "excluded.tmp" not in result.matches

    @pytest.mark.asyncio
    async def test_truncates_to_max_matches(self, grep_gnu_only, tmp_path):
        (tmp_path / "test.py").write_text("\n".join(f"line {i}" for i in range(200)))

        result = await collect_result(
            grep_gnu_only.run(GrepArgs(pattern="line", max_matches=50))
        )

        assert result.match_count == 50
        assert result.was_truncated

    @pytest.mark.asyncio
    async def test_does_not_leak_sensitive_file_contents(self, grep_gnu_only, tmp_path):
        (tmp_path / ".env").write_text("SECRET_TOKEN=supersecret\n")
        (tmp_path / "app.py").write_text("uses SECRET_TOKEN here\n")

        result = await collect_result(
            grep_gnu_only.run(GrepArgs(pattern="SECRET_TOKEN", path="."))
        )

        assert "app.py" in result.matches
        assert "supersecret" not in result.matches

    @pytest.mark.asyncio
    async def test_does_not_leak_case_variant_sensitive_files(
        self, grep_gnu_only, tmp_path
    ):
        (tmp_path / ".ENV").write_text("SECRET_TOKEN=uppercase\n")
        (tmp_path / ".Env").write_text("SECRET_TOKEN=mixedcase\n")
        (tmp_path / "app.py").write_text("uses SECRET_TOKEN here\n")

        result = await collect_result(
            grep_gnu_only.run(GrepArgs(pattern="SECRET_TOKEN", path="."))
        )

        assert "app.py" in result.matches
        assert "uppercase" not in result.matches
        assert "mixedcase" not in result.matches


@pytest.mark.skipif(not shutil.which("rg"), reason="ripgrep not available")
class TestRipgrepBackend:
    @pytest.mark.asyncio
    async def test_smart_case_lowercase_pattern(self, grep, tmp_path):
        (tmp_path / "test.py").write_text("Hello\nHELLO\nhello\n")

        result = await collect_result(grep.run(GrepArgs(pattern="hello")))

        assert result.match_count == 3

    @pytest.mark.asyncio
    async def test_smart_case_mixed_case_pattern(self, grep, tmp_path):
        (tmp_path / "test.py").write_text("Hello\nHELLO\nhello\n")

        result = await collect_result(grep.run(GrepArgs(pattern="Hello")))

        assert result.match_count == 1

    @pytest.mark.asyncio
    async def test_searches_ignored_files_when_use_default_ignore_false(
        self, grep, tmp_path
    ):
        (tmp_path / ".ignore").write_text("ignored_by_rg/\n")

        ignored_dir = tmp_path / "ignored_by_rg"
        ignored_dir.mkdir()
        (ignored_dir / "file.py").write_text("match\n")
        (tmp_path / "included.py").write_text("match\n")

        result_with_ignore = await collect_result(grep.run(GrepArgs(pattern="match")))
        assert "included.py" in result_with_ignore.matches
        assert "ignored_by_rg" not in result_with_ignore.matches

        result_without_ignore = await collect_result(
            grep.run(GrepArgs(pattern="match", use_default_ignore=False))
        )
        assert "included.py" in result_without_ignore.matches
        assert "ignored_by_rg/file.py" in result_without_ignore.matches
