from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any
import zlib

import pytest

from chartreux.core.tools import secret_scan
from chartreux.core.tools.secret_scan import DEFAULT_MAX_SCAN_BYTES, SecretFinding

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET = "aws_secret_access_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'"
GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
OPENAI_TOKEN = "sk-proj-" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
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


# --- A1a: internal value-scoped approval keys --------------------------------
#
# The scan computes one ApprovalKey per rule hit at the match point: the
# digest is SHA-256 over the UTF-8 bytes of the rule's value span as decoded
# by the scanner (UTF-8 with replacement). The fixtures below are built by
# concatenation on purpose: this file is itself a touched path at a
# submission boundary, so its source has to scan clean.


def test_generic_key_digest_is_sha256_over_group_one() -> None:
    keys: list[secret_scan.ApprovalKey] = []
    text = f'api_key = "{HIGH_ENTROPY_VALUE}"\n'
    findings = secret_scan.scan_text(text, label="k", keys=keys)
    assert [finding.rule_id for finding in findings] == ["generic-keyword-assignment"]
    assert len(keys) == 1
    key = keys[0]
    assert key.rule_id == "generic-keyword-assignment"
    # The digest basis is capture group 1 — the value alone — not the
    # keyword+separator full match.
    assert key.digest == hashlib.sha256(HIGH_ENTROPY_VALUE.encode("utf-8")).hexdigest()
    full_match = 'api_key = "' + HIGH_ENTROPY_VALUE
    assert key.digest != hashlib.sha256(full_match.encode("utf-8")).hexdigest()
    # No canonical identity was supplied: the key carries None.
    assert key.path is None


def test_openai_key_digest_is_sha256_over_the_full_token() -> None:
    keys: list[secret_scan.ApprovalKey] = []
    findings = secret_scan.scan_text(OPENAI_TOKEN + "\n", label="k", keys=keys)
    assert [finding.rule_id for finding in findings] == ["openai-api-key"]
    assert len(keys) == 1
    # The OpenAI rule's full match IS the token: the digest basis is the
    # complete token.
    assert keys[0].digest == hashlib.sha256(OPENAI_TOKEN.encode("utf-8")).hexdigest()


def test_aws_secret_key_digest_is_sha256_over_the_full_match() -> None:
    keys: list[secret_scan.ApprovalKey] = []
    findings = secret_scan.scan_text(AWS_SECRET + "\n", label="k", keys=keys)
    # The raw assignment line doubly matches: aws-secret-access-key and the
    # generic rule (whose group 1 here is the same 40-char value).
    assert "aws-secret-access-key" in {finding.rule_id for finding in findings}
    aws_keys = [key for key in keys if key.rule_id == "aws-secret-access-key"]
    assert len(aws_keys) == 1
    # ADJ-1: the digest basis is the FULL MATCH — keyword + separator +
    # quotes + value — deliberately reformatting-brittle, NOT the bare value.
    assert aws_keys[0].digest == hashlib.sha256(AWS_SECRET.encode("utf-8")).hexdigest()
    value_only = AWS_SECRET.split("'")[1]
    assert aws_keys[0].digest != hashlib.sha256(value_only.encode("utf-8")).hexdigest()


def test_aws_secret_full_match_basis_is_reformatting_brittle() -> None:
    reformatted = AWS_SECRET.replace(" = ", "  =  ")
    keys_plain: list[secret_scan.ApprovalKey] = []
    keys_reformatted: list[secret_scan.ApprovalKey] = []
    secret_scan.scan_text(AWS_SECRET + "\n", label="a", keys=keys_plain)
    secret_scan.scan_text(reformatted + "\n", label="b", keys=keys_reformatted)
    plain_aws = [k for k in keys_plain if k.rule_id == "aws-secret-access-key"]
    reformatted_aws = [
        k for k in keys_reformatted if k.rule_id == "aws-secret-access-key"
    ]
    assert len(plain_aws) == len(reformatted_aws) == 1
    # Any edit to the keyword/separator/quote bytes changes the full-match
    # digest: the documented brittleness of the ADJ-1 basis.
    assert plain_aws[0].digest != reformatted_aws[0].digest


def test_approval_keys_are_line_independent() -> None:
    keys: list[secret_scan.ApprovalKey] = []
    text = f"{AWS_KEY}\nclean line\n{AWS_KEY}\n"
    findings = secret_scan.scan_text(text, label="k", keys=keys)
    assert [finding.line for finding in findings] == [1, 3]
    assert len(keys) == 2
    # The key has no line component: the same value at any line yields the
    # same key.
    assert keys[0] == keys[1]
    assert keys[0].digest == hashlib.sha256(AWS_KEY.encode("utf-8")).hexdigest()


def test_multiple_matches_on_one_line_yield_distinct_keys() -> None:
    other_key = "AKIA" + "ZZZZZZZZZZZZZZZZ"
    keys: list[secret_scan.ApprovalKey] = []
    text = f"{AWS_KEY} {other_key}\n"
    findings = secret_scan.scan_text(text, label="k", keys=keys)
    assert [finding.rule_id for finding in findings] == ["aws-access-key-id"] * 2
    assert len(keys) == 2
    assert keys[0].digest == hashlib.sha256(AWS_KEY.encode("utf-8")).hexdigest()
    assert keys[1].digest == hashlib.sha256(other_key.encode("utf-8")).hexdigest()
    assert keys[0] != keys[1]


def test_repeated_match_on_one_line_yields_one_key() -> None:
    keys: list[secret_scan.ApprovalKey] = []
    text = f"{AWS_KEY} {AWS_KEY}\n"
    findings = secret_scan.scan_text(text, label="k", keys=keys)
    assert len(findings) == 1
    assert len(keys) == 1


def test_private_key_header_hit_yields_no_approval_key() -> None:
    keys: list[secret_scan.ApprovalKey] = []
    header = "-----BEGIN " + "RSA PRIVATE KEY-----"
    findings = secret_scan.scan_text(header + "\n", label="k", keys=keys)
    assert [finding.rule_id for finding in findings] == ["private-key-header"]
    # Partial-coverage rule: the match is the constant header line only, so
    # no approvable value-span digest exists — the hit can never be approved.
    assert keys == []


def test_value_span_follows_the_rule_schema() -> None:
    generic = next(
        rule
        for rule in secret_scan._RULE_TABLE
        if rule.rule_id == "generic-keyword-assignment"
    )
    match = generic.pattern.search(f'token = "{HIGH_ENTROPY_VALUE}"')
    assert match is not None
    assert secret_scan.value_span(generic, match) == HIGH_ENTROPY_VALUE
    header = next(
        rule for rule in secret_scan._RULE_TABLE if rule.rule_id == "private-key-header"
    )
    header_match = header.pattern.search("-----BEGIN " + "PRIVATE KEY-----")
    assert header_match is not None
    assert secret_scan.value_span(header, header_match) is None
    assert secret_scan.approval_digest(header, header_match) is None
    assert secret_scan.approval_key(header, header_match, path="p") is None


def test_digest_is_over_utf8_replacement_decoded_span_bytes() -> None:
    value = AWS_SECRET.split("'")[1]
    raw_span = b"aws\xff\xfe'" + value.encode("utf-8") + b"'"
    keys: list[secret_scan.ApprovalKey] = []
    findings = secret_scan.scan_bytes(raw_span, label="blob", keys=keys)
    assert [finding.rule_id for finding in findings] == ["aws-secret-access-key"]
    assert len(keys) == 1
    # The scanner decodes with errors="replace": the invalid bytes become
    # U+FFFD inside the match, and the digest is over the DECODED span's
    # UTF-8 bytes (U+FFFD encodes as EF BF BD), never over the raw bytes.
    decoded_span = "aws\ufffd\ufffd'" + value + "'"
    assert keys[0].digest == hashlib.sha256(decoded_span.encode("utf-8")).hexdigest()
    assert keys[0].digest != hashlib.sha256(raw_span).hexdigest()


def test_findings_carry_neither_matched_bytes_nor_entry_digests(
    seeded_files: tuple[Path, Path],
) -> None:
    aws_file, token_file = seeded_files
    keys: list[secret_scan.ApprovalKey] = []
    findings = secret_scan.scan_paths([aws_file, token_file], keys=keys)
    assert findings and keys
    # One key per rule-hit finding, in scan order.
    assert sorted(f.rule_id for f in findings) == sorted(k.rule_id for k in keys)
    rendered = json.dumps([vars(finding) for finding in findings])
    for secret in (*SEEDED_SECRETS, AWS_SECRET):
        assert secret not in rendered
    for finding in findings:
        for field in vars(finding).values():
            for key in keys:
                assert key.digest not in str(field)


def test_scan_paths_collects_keys_for_explicit_and_walked_files(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    explicit = tmp_path / "explicit.py"
    walked = tree / "walked.py"
    for file in (explicit, walked):
        file.write_text(f"aws_access_key_id = {AWS_KEY}\n", encoding="utf-8")
    keys: list[secret_scan.ApprovalKey] = []
    findings = secret_scan.scan_paths([explicit, tree], keys=keys)
    assert {finding.path for finding in findings} == {str(explicit), str(walked)}
    assert len(keys) == 2
    # No canonical identity is supplied at this level yet: path is None.
    assert all(key.path is None for key in keys)
    assert {key.digest for key in keys} == {
        hashlib.sha256(AWS_KEY.encode("utf-8")).hexdigest()
    }


def test_scan_incomplete_produces_no_approval_key(tmp_path: Path) -> None:
    artifact = tmp_path / "big.bin"
    artifact.write_bytes(b"x" * (DEFAULT_MAX_SCAN_BYTES + 1))
    keys: list[secret_scan.ApprovalKey] = []
    findings = secret_scan.scan_paths([artifact], keys=keys)
    assert [finding.rule_id for finding in findings] == ["scan-incomplete"]
    # An incomplete scan never yields a key: it can never be suppressed.
    assert keys == []


# --- A3a: closed reason codes and safe path identifiers ----------------------


def test_reason_code_enum_is_closed() -> None:
    # The closed enum of every finding's reason: the seven codes, no more.
    assert {member.value for member in secret_scan.FindingReason} == {
        "match",
        "file_size_limit",
        "walk_budget",
        "enumeration_limit",
        "unreadable",
        "unknown_write_footprint",
        "policy_mismatch",
    }
    assert len(secret_scan.FindingReason) == 7


def test_rule_hits_carry_the_match_reason_code() -> None:
    findings = secret_scan.scan_text(f"key = {AWS_KEY}\n", label="artifact")
    assert [finding.reason for finding in findings] == [secret_scan.FindingReason.MATCH]
    assert all(finding.reason == "match" for finding in findings)


def test_file_size_limit_sentinel_reason_code(tmp_path: Path) -> None:
    # Both truncation constructions: an over-cap file on disk and in-memory
    # content past the cap.
    artifact = tmp_path / "big.bin"
    artifact.write_bytes(b"x" * (DEFAULT_MAX_SCAN_BYTES + 1))
    findings = secret_scan.scan_paths([artifact])
    assert [finding.reason for finding in findings] == [
        secret_scan.FindingReason.FILE_SIZE_LIMIT
    ]
    in_memory = secret_scan.scan_bytes(
        b"x = 1\n" + AWS_KEY.encode("utf-8"), label="blob", max_bytes=6
    )
    assert [finding.reason for finding in in_memory] == [
        secret_scan.FindingReason.FILE_SIZE_LIMIT
    ]


def test_walk_budget_sentinel_reason_code(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    for index in range(4):
        (tree / f"clean-{index}.txt").write_text("clean\n", encoding="utf-8")
    findings = secret_scan.scan_paths([tree], max_walk_entries=2)
    assert [finding.reason for finding in findings] == [
        secret_scan.FindingReason.WALK_BUDGET
    ]


def test_enumeration_limit_sentinel_reason_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    for index in range(4):
        (tree / f"dir-{index}").mkdir()
    monkeypatch.setattr(secret_scan, "DEFAULT_MAX_WALK_ENUMERATIONS", 2)
    findings = secret_scan.scan_paths([tree], count_qualifying_only=True)
    assert [finding.reason for finding in findings] == [
        secret_scan.FindingReason.ENUMERATION_LIMIT
    ]


def test_unreadable_sentinels_carry_the_unreadable_reason_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unreadable file.
    file = tmp_path / "locked.txt"
    file.write_text(f"token = '{HIGH_ENTROPY_VALUE}'\n", encoding="utf-8")
    real_open = Path.open

    def refusing_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == file:
            raise OSError("permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refusing_open)
    findings = secret_scan.scan_paths([file])
    assert [finding.reason for finding in findings] == [
        secret_scan.FindingReason.UNREADABLE
    ]
    monkeypatch.undo()

    # A directory the walk cannot enumerate.
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    real_scandir = os.scandir

    def refusing_scandir(path: Any) -> Any:
        if Path(path) == tree / "sub":
            raise PermissionError("permission denied")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", refusing_scandir)
    findings = secret_scan.scan_paths([tree])
    assert {finding.reason for finding in findings} == {
        secret_scan.FindingReason.UNREADABLE
    }

    # The constructor's default is the unreadable code (the hook's sentinels).
    sentinel = secret_scan.incomplete_finding("some/path", "why")
    assert sentinel.reason == secret_scan.FindingReason.UNREADABLE


def test_gate_sentinels_carry_their_own_reason_codes() -> None:
    # The submission gate's two sentinels construct their findings with their
    # closed codes.
    unknown = secret_scan.incomplete_finding(
        "<shell command with an unestablished write surface>",
        "why",
        reason_code=secret_scan.FindingReason.UNKNOWN_WRITE_FOOTPRINT,
    )
    assert unknown.reason == secret_scan.FindingReason.UNKNOWN_WRITE_FOOTPRINT
    mismatch = secret_scan.incomplete_finding(
        secret_scan.MANIFEST_FILENAME,
        "why",
        reason_code=secret_scan.FindingReason.POLICY_MISMATCH,
    )
    assert mismatch.reason == secret_scan.FindingReason.POLICY_MISMATCH


def test_path_containing_rule_matched_content_degrades_to_opaque_identifier() -> None:
    # A matched credential can occur in a FILENAME: the finding's path then
    # degrades to a deterministic opaque per-finding identifier that carries
    # no byte of the path or the match.
    label = f"/repo/leak-{GITHUB_TOKEN}.py"
    findings = secret_scan.scan_text(f"GITHUB_TOKEN = {GITHUB_TOKEN}\n", label=label)
    assert len(findings) == 2  # github-token and the generic rule
    for finding in findings:
        assert finding.path_suspect is True
        assert finding.path.startswith("<opaque-path:")
        assert GITHUB_TOKEN not in finding.path
        assert label not in finding.path
    # Deterministic: the same finding degrades to the same identifier, and
    # per-finding: the two rules' hits get distinct identifiers.
    again = secret_scan.scan_text(f"GITHUB_TOKEN = {GITHUB_TOKEN}\n", label=label)
    assert [finding.path for finding in again] == [finding.path for finding in findings]
    assert len({finding.path for finding in findings}) == 2


def test_everyday_paths_stay_echoable() -> None:
    # The degradation trigger is rule-matched content in the path ONLY: an
    # everyday legitimate path — even one that names the scanner's own test
    # file — stays echoable, and a crafted naming convention cannot force
    # every diagnostic opaque.
    findings = secret_scan.scan_text(
        f"GITHUB_TOKEN = {GITHUB_TOKEN}\n",
        label="/repo/tests/core/tools/test_secret_scan.py",
    )
    assert all(
        finding.path == "/repo/tests/core/tools/test_secret_scan.py"
        for finding in findings
    )
    assert all(finding.path_suspect is False for finding in findings)


def test_control_characters_are_stripped_from_finding_paths() -> None:
    label = "/repo/we\x00ird\x1bname\n.py"
    findings = secret_scan.scan_text(f"GITHUB_TOKEN = {GITHUB_TOKEN}\n", label=label)
    for finding in findings:
        assert finding.path == "/repo/weirdname.py"
        assert finding.path_suspect is False
        assert all(character.isprintable() for character in finding.path)
    # The sentinel constructor strips too.
    sentinel = secret_scan.incomplete_finding("/repo/bi\x00g.bin", "why")
    assert sentinel.path == "/repo/big.bin"


def test_strip_control_characters_and_opaque_identifier_helpers() -> None:
    assert secret_scan.strip_control_characters("a\x00b\x1fc\nd") == "abcd"
    assert secret_scan.strip_control_characters("plain/path.py") == "plain/path.py"
    identifier = secret_scan.opaque_path_identifier("/repo/x", "rule", 3, 7)
    assert identifier == secret_scan.opaque_path_identifier("/repo/x", "rule", 3, 7)
    assert identifier != secret_scan.opaque_path_identifier("/repo/x", "rule", 3, 8)
    assert identifier.startswith("<opaque-path:") and identifier.endswith(">")


def _policy_with_two_entries_sharing_path_and_rule() -> secret_scan.ScanPolicy:
    """A manifest whose approved_matches holds two entries for one (path, rule)."""
    document = {
        "schema_version": secret_scan.SCAN_POLICY_SCHEMA_VERSION,
        "approved_matches": [
            {
                "path": "fixture.txt",
                "rule_id": "github-token",
                "digest": hashlib.sha256(b"value-one").hexdigest(),
            },
            {
                "path": "fixture.txt",
                "rule_id": "github-token",
                "digest": hashlib.sha256(b"value-two").hexdigest(),
            },
        ],
        "residual_exclusions": [],
    }
    policy = secret_scan.parse_scan_policy(
        json.dumps(document, indent=2).encode("utf-8")
    )
    assert policy is not None
    return policy


def test_rule_hit_carries_manifest_entry_identifiers_never_digests() -> None:
    policy = _policy_with_two_entries_sharing_path_and_rule()
    # A THIRD value at the same (path, rule) is not approved: it surfaces,
    # carrying the identifiers of the entries that share its (path, rule) —
    # never their digests.
    findings = secret_scan.scan_text(
        f"GITHUB_TOKEN = {GITHUB_TOKEN}\n",
        label="fixture.txt",
        key_path="fixture.txt",
        policy=policy,
    )
    github_hits = [f for f in findings if f.rule_id == "github-token"]
    assert len(github_hits) == 1
    assert github_hits[0].manifest_entries == (
        "approved_matches[0]",
        "approved_matches[1]",
    )
    rendered = json.dumps(vars(github_hits[0]))
    for entry in policy.approved:
        assert entry.digest not in rendered
    assert GITHUB_TOKEN not in rendered


def test_policy_entry_ids_lookup() -> None:
    policy = _policy_with_two_entries_sharing_path_and_rule()
    assert secret_scan.policy_entry_ids(policy, "fixture.txt", "github-token") == (
        "approved_matches[0]",
        "approved_matches[1]",
    )
    assert secret_scan.policy_entry_ids(policy, "fixture.txt", "openai-api-key") == ()
    assert secret_scan.policy_entry_ids(policy, None, "github-token") == ()
    # The fail-closed empty policy carries no identifiers.
    empty = secret_scan.empty_scan_policy("digest")
    assert secret_scan.policy_entry_ids(empty, "fixture.txt", "github-token") == ()


def test_scan_incomplete_carries_no_manifest_entries() -> None:
    # Even with a policy carrying entries for the path, a sentinel finding
    # carries none.
    _policy_with_two_entries_sharing_path_and_rule()
    sentinel = secret_scan.incomplete_finding("fixture.txt", "why")
    assert sentinel.manifest_entries == ()
    assert sentinel.reason == secret_scan.FindingReason.UNREADABLE
