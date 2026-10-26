from __future__ import annotations

import ast
import gzip
import json
import os
from pathlib import Path
import struct
from typing import Any
import zlib

import pytest

from chartreux.core.tools import secret_scan
from chartreux.core.tools.secret_scan import SecretFinding

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET = "aws_secret_access_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'"
GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
HIGH_ENTROPY_VALUE = "r7Bqz9WkPm3XvT2sLd8Yh5Nj4Fg6Ua1Q"
LOW_ENTROPY_VALUE = "supersecretvalue123"

SEEDED_SECRETS = (
    AWS_KEY,
    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    GITHUB_TOKEN,
    HIGH_ENTROPY_VALUE,
)


def _minimal_png() -> bytes:
    """A real 1x1 PNG: NUL bytes and a non-UTF-8 signature byte included."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    scanline = b"\x00\xff\x00\x00"  # filter byte, then one RGB pixel
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(scanline))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def seeded_files(tmp_path: Path) -> tuple[Path, Path]:
    aws_file = tmp_path / "aws_config.py"
    aws_file.write_text(
        f"[default]\naws_access_key_id = {AWS_KEY}\n{AWS_SECRET}\n", encoding="utf-8"
    )
    token_file = tmp_path / "deploy.sh"
    token_file.write_text(
        f'GITHUB_TOKEN={GITHUB_TOKEN}\napi_key = "{HIGH_ENTROPY_VALUE}"\n',
        encoding="utf-8",
    )
    return aws_file, token_file


def test_detects_aws_access_key_and_secret(seeded_files: tuple[Path, Path]) -> None:
    aws_file, _ = seeded_files
    findings = secret_scan.scan_paths([aws_file])
    rule_ids = {finding.rule_id for finding in findings}
    assert "aws-access-key-id" in rule_ids
    assert "aws-secret-access-key" in rule_ids


def test_detects_github_token(seeded_files: tuple[Path, Path]) -> None:
    _, token_file = seeded_files
    findings = secret_scan.scan_paths([token_file])
    assert any(finding.rule_id == "github-token" for finding in findings)


def test_detects_generic_high_entropy_assignment(
    seeded_files: tuple[Path, Path],
) -> None:
    _, token_file = seeded_files
    findings = secret_scan.scan_paths([token_file])
    assert any(finding.rule_id == "generic-keyword-assignment" for finding in findings)


def test_bare_high_entropy_literal_still_fires() -> None:
    text = f"password = {HIGH_ENTROPY_VALUE}\n"
    findings = secret_scan.scan_text(text, label="bare")
    assert [finding.rule_id for finding in findings] == ["generic-keyword-assignment"]


def test_provider_prefixed_api_key_assignment_is_flagged() -> None:
    # The keyword family matches identifiers that merely CONTAIN a keyword:
    # MISTRAL_API_KEY ends with api_key but has no word boundary before it.
    # The fixture value is built by concatenation so this file stays clean.
    text = "MISTRAL_API_KEY = " + '"' + HIGH_ENTROPY_VALUE + '"\n'
    findings = secret_scan.scan_text(text, label="prefixed")
    assert [finding.rule_id for finding in findings] == ["generic-keyword-assignment"]
    # The contextvars negative survives the broadened boundary: a function
    # call on the right-hand side is not a literal value.
    retry_like = (
        "token = _LOGICAL_RETRY_BUDGET.set(budget)\n"
        "    _LOGICAL_RETRY_BUDGET.reset(token)\n"
    )
    assert secret_scan.scan_text(retry_like, label="retry-like") == []


def test_contextvars_token_assignment_is_not_flagged() -> None:
    text = (
        "token = _LOGICAL_RETRY_BUDGET.set(budget)\n"
        "    _LOGICAL_RETRY_BUDGET.reset(token)\n"
    )
    assert secret_scan.scan_text(text, label="retry-like") == []
    retry_path = Path(secret_scan.__file__).parents[1] / "utils" / "retry.py"
    assert secret_scan.scan_paths([retry_path]) == []


def test_token_count_config_assignment_is_not_flagged() -> None:
    # Plural ``*_tokens`` names are token COUNTS, not credentials: the
    # attribute-access right-hand side is a config reference, and its dotted
    # identifier chain can drift past the entropy threshold.
    text = (
        "max_tokens=self.policy.max_tokens\n"
        "max_session_tokens=self.policy.max_session_tokens\n"
        "run_max_session_tokens=self.policy.run_max_session_tokens\n"
        "total_tokens = request.usage.total_tokens\n"
        'MAX_TOKENS = "32768"\n'
    )
    assert secret_scan.scan_text(text, label="counts") == []
    runtime_path = Path(secret_scan.__file__).parents[2] / "app_server" / "_runtime.py"
    assert secret_scan.scan_paths([runtime_path]) == []


def test_plural_token_name_with_high_entropy_literal_is_not_flagged() -> None:
    # The plural criterion is deliberate: value-shape rules (ghp_, sk-, AKIA,
    # ...) still catch real secrets assigned to plural names.
    text = 'api_tokens = "' + HIGH_ENTROPY_VALUE + '"\n'
    assert secret_scan.scan_text(text, label="plural") == []


def test_singular_token_assignment_still_fires() -> None:
    text = "access_token = " + '"' + HIGH_ENTROPY_VALUE + '"\n'
    assert [
        finding.rule_id for finding in secret_scan.scan_text(text, label="singular")
    ] == ["generic-keyword-assignment"]


def test_clean_file_yields_no_findings(tmp_path: Path) -> None:
    clean = tmp_path / "clean.py"
    clean.write_text(
        "def add(left: int, right: int) -> int:\n    return left + right\n",
        encoding="utf-8",
    )
    assert secret_scan.scan_paths([clean]) == []


def test_low_entropy_assignment_is_not_flagged(tmp_path: Path) -> None:
    file = tmp_path / "config.py"
    file.write_text(f'password = "{LOW_ENTROPY_VALUE}"\n', encoding="utf-8")
    assert secret_scan.scan_paths([file]) == []


def test_findings_do_not_carry_the_secret(seeded_files: tuple[Path, Path]) -> None:
    aws_file, token_file = seeded_files
    findings = secret_scan.scan_paths([aws_file, token_file])
    assert findings
    for finding in findings:
        rendered = json.dumps(finding.__dict__)
        for secret in SEEDED_SECRETS:
            assert secret not in rendered
            for value in vars(finding).values():
                assert secret not in str(value)


def test_finding_reports_line_and_rule(seeded_files: tuple[Path, Path]) -> None:
    aws_file, _ = seeded_files
    findings = secret_scan.scan_paths([aws_file])
    aws_ids = [f for f in findings if f.rule_id == "aws-access-key-id"]
    assert len(aws_ids) == 1
    assert aws_ids[0].line == 2
    assert isinstance(aws_ids[0], SecretFinding)
    assert aws_ids[0].path == str(aws_file)


def test_missing_and_directory_paths_are_handled(
    seeded_files: tuple[Path, Path], tmp_path: Path
) -> None:
    aws_file, token_file = seeded_files
    findings = secret_scan.scan_paths([
        tmp_path,
        tmp_path / "deleted-artifact.bin",
        AWS_KEY,
    ])
    rule_ids = {finding.rule_id for finding in findings}
    assert {
        "aws-access-key-id",
        "github-token",
        "generic-keyword-assignment",
    } <= rule_ids


def test_secret_behind_a_nul_byte_is_detected(tmp_path: Path) -> None:
    # The artifact's writer controls the content: one prepended NUL must not
    # suppress the scan. The file is split on NUL boundaries and every text
    # run is scanned, so the secret appended behind the NUL is found.
    artifact = tmp_path / "blob.bin"
    artifact.write_bytes(b"\x00\x01" + AWS_KEY.encode("utf-8"))
    findings = secret_scan.scan_paths([artifact])
    assert [finding.rule_id for finding in findings] == ["aws-access-key-id"]
    assert findings[0].path == str(artifact)


def test_nul_split_runs_keep_file_line_numbers(tmp_path: Path) -> None:
    artifact = tmp_path / "mixed.bin"
    artifact.write_bytes(
        b"line one\n\x00line two\n" + GITHUB_TOKEN.encode("utf-8") + b"\n"
    )
    findings = [
        finding
        for finding in secret_scan.scan_paths([artifact])
        if finding.rule_id == "github-token"
    ]
    assert len(findings) == 1
    assert findings[0].line == 3


def test_genuine_binary_png_passes_clean(tmp_path: Path) -> None:
    png = tmp_path / "pixel.png"
    png.write_bytes(_minimal_png())
    assert secret_scan.scan_paths([png]) == []


def test_genuine_binary_gzip_passes_clean(tmp_path: Path) -> None:
    archive = tmp_path / "blob.gz"
    archive.write_bytes(gzip.compress(b"hello world\n"))
    assert secret_scan.scan_paths([archive]) == []


def test_repeated_rule_hits_on_one_line_deduplicate() -> None:
    text = f"export AWS_KEY={AWS_KEY} {AWS_KEY}\n"
    findings = secret_scan.scan_text(text, label="one")
    assert [f.rule_id for f in findings].count("aws-access-key-id") == 1


def test_max_bytes_caps_the_scanned_prefix(tmp_path: Path) -> None:
    file = tmp_path / "tail.py"
    file.write_text(f"x = 1\ntoken = '{HIGH_ENTROPY_VALUE}'\n", encoding="utf-8")
    capped = secret_scan.scan_paths([file], max_bytes=4)
    # The secret past the cap is not reported — but the truncation is: an
    # incomplete scan is never a clean scan.
    assert [finding.rule_id for finding in capped] == ["scan-incomplete"]
    assert capped[0].path == str(file)


def test_file_exactly_at_the_cap_is_complete(tmp_path: Path) -> None:
    file = tmp_path / "exact.py"
    file.write_text("x = 1\n", encoding="utf-8")
    assert secret_scan.scan_paths([file], max_bytes=6) == []


def test_scan_bytes_scans_in_memory_content() -> None:
    findings = secret_scan.scan_bytes(b"\x00" + AWS_KEY.encode("utf-8"), label="blob")
    assert [finding.rule_id for finding in findings] == ["aws-access-key-id"]
    assert [finding.path for finding in findings] == ["blob"]


def test_scan_bytes_caps_content_at_max_bytes() -> None:
    findings = secret_scan.scan_bytes(
        b"x = 1\n" + AWS_KEY.encode("utf-8"), label="blob", max_bytes=6
    )
    assert [finding.rule_id for finding in findings] == ["scan-incomplete"]
    assert findings[0].path == "blob"


def test_unreadable_file_reports_scan_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fail-closed: an unreadable artifact is uninspected content, never a
    # clean scan.
    file = tmp_path / "locked.txt"
    file.write_text("token = '" + HIGH_ENTROPY_VALUE + "'\n", encoding="utf-8")
    real_open = Path.open

    def refusing_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == file:
            raise OSError("permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refusing_open)
    findings = secret_scan.scan_paths([file])
    assert [finding.rule_id for finding in findings] == ["scan-incomplete"]
    assert findings[0].path == str(file)


def test_unreadable_directory_reports_scan_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fail-closed: a touched directory the walk cannot enumerate is
    # uninspected content, never a clean scan — its subtree is not walked.
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "inner.txt").write_text("clean\n", encoding="utf-8")
    real_scandir = os.scandir

    def refusing_scandir(path: Any) -> Any:
        if Path(path) == tree / "sub":
            raise PermissionError("permission denied")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", refusing_scandir)
    findings = secret_scan.scan_paths([tree])
    assert [finding.rule_id for finding in findings] == ["scan-incomplete"]
    assert findings[0].path == str(tree / "sub")


def test_over_budget_directory_walk_reports_scan_incomplete(tmp_path: Path) -> None:
    # A crafted tree must not turn the scan into an unbounded traversal: the
    # walk stops at the entry budget and the unwalked remainder is reported
    # as a sentinel, never silently clean.
    tree = tmp_path / "tree"
    tree.mkdir()
    for index in range(4):
        (tree / f"clean-{index}.txt").write_text("clean\n", encoding="utf-8")
    for index in range(4):
        (tree / f"seeded-{index}.txt").write_text(
            f"aws_access_key_id = {AWS_KEY}\n", encoding="utf-8"
        )
    findings = secret_scan.scan_paths([tree], max_walk_entries=4)
    # The first four entries (the clean files) were scanned; the walk stopped
    # before the seeded files, so their secrets are not reported — the
    # sentinel for the unwalked remainder is.
    assert [finding.rule_id for finding in findings] == ["scan-incomplete"]
    assert findings[0].path == str(tree)
    assert "budget" in findings[0].description


def test_shannon_entropy_basics() -> None:
    assert secret_scan.shannon_entropy("") == 0.0
    assert secret_scan.shannon_entropy("aaaaaaaa") == 0.0
    assert secret_scan.shannon_entropy("ab") == pytest.approx(1.0)
    assert secret_scan.shannon_entropy(HIGH_ENTROPY_VALUE) > 4.0
    assert secret_scan.shannon_entropy(LOW_ENTROPY_VALUE) < 4.0


def test_scan_text_uses_the_label() -> None:
    findings = secret_scan.scan_text(f"key = {AWS_KEY}\n", label="artifact-3")
    assert [finding.path for finding in findings] == ["artifact-3"]


def test_scanner_module_is_pure_local() -> None:
    source = Path(secret_scan.__file__).read_text(encoding="utf-8")
    imported = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    banned = {"socket", "ssl", "urllib", "http", "requests", "httpx", "aiohttp"}
    assert not imported & banned


def test_rules_table_is_extensible() -> None:
    rule = secret_scan._RULE_TABLE[0]
    assert rule.rule_id
    assert rule.description
    rule_ids = {rule.rule_id for rule in secret_scan._RULE_TABLE}
    assert "aws-access-key-id" in rule_ids
    assert "generic-keyword-assignment" in rule_ids
