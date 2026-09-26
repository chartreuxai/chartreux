"""Secret-value redaction and credential environment scrubbing.

Redaction does not catch rot13, compression, Unicode interleaving, splitting a
secret across fields, cross-field reconstruction, or images. Decode scans are
budget-bounded per event; after exhaustion later encoded carriers may pass
unchanged. Single-character percent variants of values over 1024 characters
are not enumerated. Direct known-value matching still runs. Secrets shorter
than the 8-character direct-match threshold are not redacted.

Chartreux loads its own credentials: API keys stored in ``~/.chartreux/.env``,
keys resolved through the keyring, model-catalog and web-search provider keys,
and known MCP OAuth access/refresh token records. Two controls reduce exposure
of those values:

1. :func:`redact` replaces every known secret value in tool response text with
   a placeholder. Outward tool events are sanitized before emission, and the
   agent loop also sanitizes model input and persisted session messages.
2. The scrub helpers remove chartreux's credential environment variables from
   the environments of child processes (shell commands, MCP stdio servers, the
   project-context git subprocess, and client terminals) unless a variable is
   listed in the user-configured passthrough.

Heavy imports (dotenv, the model catalog, the keyring) are deferred to first
use so importing this module stays cheap for early-startup call sites.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import os
import re
import threading
import time
from typing import Any
from urllib.parse import unquote
import weakref

from chartreux.observability.logging import logger

REDACTED_PLACEHOLDER = "[REDACTED]"
#: Supported credential length for buffered stream redaction (16 KiB). Direct
#: matching scans the entire retained body, including longer values; encoded
#: carriers remain subject to the bounded decode-candidate budget.
MAX_SUPPORTED_CREDENTIAL_LENGTH = 16 * 1024
_MAX_SINGLE_PERCENT_VARIANT_LENGTH = 1024
#: Direct known-value matching includes short (but not trivial) credentials.
MIN_SECRET_LENGTH = 8
#: Decode matching uses the same minimum secret length as direct matching.
_MIN_ENCODED_SECRET_LENGTH = MIN_SECRET_LENGTH
#: Line width GNU coreutils ``base64`` wraps its output at by default.
_B64_WRAP_WIDTH = 76
#: Minimum candidate length matches the direct secret threshold; ordinary
#: decodable text is kept unless its decoded content reveals a known secret.
_MIN_ENCODED_RUN = MIN_SECRET_LENGTH
#: A whitespace-delimited word made only of base64-alphabet characters (the
#: hex and base32 alphabets are subsets).
_ENCODED_WORD = re.compile(r"[A-Za-z0-9+/_=-]+")
_PERCENT_RUN = re.compile(r"(?:%[0-9a-fA-F]{2}){8,}")
_PERCENT_MIXED = re.compile(r"(?:%[0-9a-fA-F]{2}|[A-Za-z0-9._~-])+")
_OD_ROW = re.compile(
    r"(?m)^[ \t]*(?:(?:[0-7]{7,8}[ \t]+(?:[0-9a-fA-F]{2}[ \t]*)+)"
    r"|(?:(?:[0-9a-fA-F]{2}[ \t]+){7,}[0-9a-fA-F]{2}[ \t]*))$"
)
#: Cheap pre-filter: the per-word candidate scan runs only when the text
#: contains an alphabet run long enough to hide a secret on its own.
_ENCODED_RUN_SEARCH = re.compile(f"[A-Za-z0-9+/_=-]{{{_MIN_ENCODED_RUN}}}")
_MAX_JSON_REDACTION_DEPTH = 64
#: Decode at most 128 windows of at most 8192 characters per event.
_MAX_ENCODED_CANDIDATES = 128
_MAX_ENCODED_CANDIDATE_CHARS = 8192
_ENCODED_WINDOW_STEP = 7168  # overlap covers a secret crossing a window edge
_NEGATIVE_KEYRING_TTL = 30.0
_OD_OFFSET_WIDTH = 7


@dataclass(frozen=True)
class ScrubPolicy:
    """Immutable execution policy; admitted values are captured for output only."""

    passthrough: frozenset[str] = frozenset()
    credential_names: frozenset[str] = frozenset()
    oauth_records: frozenset[str] = frozenset()
    # Only outward redaction policies carry admitted values; never register or
    # bind these snapshots for tool execution, logging, or event serialization.
    redaction_credentials: tuple[tuple[str, str], ...] = field(
        default=(), repr=False, compare=False
    )
    redaction_oauth_values: frozenset[str] = field(
        default=frozenset(), repr=False, compare=False
    )

    def capture_for_redaction(self) -> ScrubPolicy:
        """Resolve values at admission, before a running tool can outlive a reload."""
        with bind_policy(self):
            return replace(
                self,
                redaction_credentials=tuple(_loaded_credentials()),
                redaction_oauth_values=_oauth_secret_values(),
            )

    @classmethod
    def from_config(cls, config: Any) -> ScrubPolicy:
        return cls(
            frozenset(config.credential_env_passthrough),
            mcp_static_auth_env_names(config) | _configured_search_env_names(config),
            frozenset(
                f"mcp-oauth:{server.name}:tokens"
                for server in getattr(config, "mcp_servers", None) or ()
                if getattr(getattr(server, "auth", None), "type", None) == "oauth"
            ),
        )

    @classmethod
    def for_redaction(
        cls, admitted: Iterable[ScrubPolicy], live: ScrubPolicy
    ) -> ScrubPolicy:
        """Union admitted and live names/values, never execution passthrough."""
        policies = (*admitted, live)
        return cls(
            credential_names=frozenset().union(*(p.credential_names for p in policies)),
            oauth_records=frozenset().union(*(p.oauth_records for p in policies)),
            redaction_credentials=tuple(
                pair for p in policies for pair in p.redaction_credentials
            ),
            redaction_oauth_values=frozenset().union(
                *(p.redaction_oauth_values for p in policies)
            ),
        )


_current_policy: ContextVar[ScrubPolicy | None] = ContextVar("chartreux_scrub_policy")


def current_policy() -> ScrubPolicy:
    return _current_policy.get(None) or ScrubPolicy()


@contextmanager
def bind_policy(policy: ScrubPolicy) -> Iterator[None]:
    token = _current_policy.set(policy)
    try:
        yield
    finally:
        _current_policy.reset(token)


_lock = threading.Lock()
_credential_names: frozenset[str] | None = None
_credential_names_token: object = None
_dotenv_entries: dict[str, str | None] | None = None
_keyring_values: dict[str, tuple[str | None, float]] = {}
_oauth_tokens: dict[str, frozenset[str]] = {}
_mcp_static_auth_env_names: frozenset[str] = frozenset()
_live_policies: weakref.WeakKeyDictionary[Any, ScrubPolicy] = (
    weakref.WeakKeyDictionary()
)


def register_session_policy(owner: Any, policy: ScrubPolicy) -> None:
    """Track credential names for all live sessions without sharing passthrough."""
    with _lock:
        _live_policies[owner] = policy


def unregister_session_policy(owner: Any) -> None:
    with _lock:
        _live_policies.pop(owner, None)


def _env_file_token() -> object:
    """A cheap identity token for the credential .env file, if present."""
    try:
        from chartreux.core.paths import GLOBAL_ENV_FILE

        stat = GLOBAL_ENV_FILE.path.stat()
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_ctime_ns, stat.st_mtime_ns, stat.st_size)


def _read_dotenv_entries() -> dict[str, str | None]:
    """Names and stored values in ``~/.chartreux/.env`` (empty when absent)."""
    try:
        from dotenv import dotenv_values

        from chartreux.core.paths import GLOBAL_ENV_FILE

        path = GLOBAL_ENV_FILE.path
        if not path.is_file() and not path.is_fifo():
            return {}
        return dict(dotenv_values(path))
    except Exception:
        return {}


def _catalog_env_var_names() -> frozenset[str]:
    """``api_key_env_var`` names declared by catalog providers."""
    try:
        from chartreux.core.model_catalog.loader import load_catalog

        snapshot = load_catalog()
    except Exception:
        return frozenset()
    return frozenset(
        provider.api_key_env_var
        for provider in snapshot.catalog.providers.values()
        if provider.api_key_env_var
    )


def _configured_search_env_names(config: Any) -> frozenset[str]:
    """Use the same tool override field as web-search provider resolution."""
    override = (getattr(config, "tools", None) or {}).get("web_search") or {}
    name = override.get("api_key_env_var") if isinstance(override, dict) else None
    return frozenset({name}) if isinstance(name, str) and name else frozenset()


def _search_provider_env_names() -> frozenset[str]:
    """Built-in names from the web-search provider's actual key mapping."""
    from chartreux.core.tools.builtins.web_search import _SEARCH_DEFAULT_KEY_ENVS

    return frozenset(_SEARCH_DEFAULT_KEY_ENVS.values())


def mcp_static_auth_env_names(config: Any) -> frozenset[str]:
    """MCP static-auth token variable names declared by a config object.

    ``MCPStaticAuth.api_key_env`` names an environment variable whose value is
    sent as an HTTP header token, so it is a credential exactly like a provider
    API key. The config is inspected duck-typed so this module never needs to
    import the config models; servers without static auth contribute nothing.
    """
    names: set[str] = set()
    for server in getattr(config, "mcp_servers", None) or ():
        auth = getattr(server, "auth", None)
        name = getattr(auth, "api_key_env", "")
        if name:
            names.add(name)
    return frozenset(names)


def set_mcp_static_auth_env_names(names: Iterable[str]) -> None:
    """Set names in this context only (legacy API; prefer binding a policy)."""
    policy = current_policy()
    _current_policy.set(
        ScrubPolicy(policy.passthrough, frozenset(names), policy.oauth_records)
    )


def credential_env_var_names(policy: ScrubPolicy | None = None) -> frozenset[str]:
    """Names of environment variables chartreux treats as credentials.

    Every key stored in ``~/.chartreux/.env`` is a credential by definition,
    together with model-catalog and web-search provider key names and MCP
    static-auth token names declared by the live config. The name set is
    cached and re-read when the .env file changes on disk. Session-derived
    names are never stored in the process-wide cache.
    """
    global _credential_names, _credential_names_token, _dotenv_entries
    token = _env_file_token()
    with _lock:
        if _credential_names is None or token != _credential_names_token:
            entries = _read_dotenv_entries()
            _credential_names = (
                frozenset(entries)
                | _catalog_env_var_names()
                | _search_provider_env_names()
            )
            _credential_names_token = token
            _dotenv_entries = entries
        names = _credential_names
        session_names = frozenset(
            name
            for active in _live_policies.values()
            for name in active.credential_names
        )
    return names | session_names | (policy or current_policy()).credential_names


def invalidate_keyring_credential(name: str) -> None:
    """Invalidate only a named record after a successful keyring write/delete."""
    with _lock:
        _keyring_values.pop(name, None)


def register_mcp_oauth_record(username: str, raw: str | None) -> None:
    """Track a known OAuth token record, never enumerate unrelated keyring data."""
    if not username.startswith("mcp-oauth:") or not username.endswith(":tokens"):
        return
    import json

    tokens: frozenset[str] = frozenset()
    if raw:
        try:
            record = json.loads(raw)
            tokens = frozenset(
                value
                for field in ("access_token", "refresh_token")
                if isinstance((value := record.get(field)), str)
                and len(value) >= MIN_SECRET_LENGTH
            )
        except (ValueError, AttributeError):
            pass
    with _lock:
        if tokens:
            _oauth_tokens[username] = tokens
        else:
            _oauth_tokens.pop(username, None)


def _keyring_value(name: str) -> str | None:
    """Keyring fallback for a credential name missing from the environment.

    Keyring hits remain cached until invalidation/reload; misses expire after
    a short bounded interval to detect out-of-process credential changes.
    """
    now = time.monotonic()
    with _lock:
        entry = _keyring_values.get(name)
        if entry is not None and (entry[0] is not None or now < entry[1]):
            return entry[0]
    try:
        from chartreux.utils.keyring import get_api_key_from_keyring

        value = get_api_key_from_keyring(name)
    except Exception:
        value = None
    with _lock:
        _keyring_values[name] = (value, now + _NEGATIVE_KEYRING_TTL)
    return value


def _loaded_credentials() -> list[tuple[str, str]]:
    """``(name, value)`` pairs for every loaded secret value worth redacting.

    Sources are the live environment, the values stored in
    ``~/.chartreux/.env`` (even when an explicit environment value shadowed
    them at load time), and the keyring. Only values at least
    :data:`MIN_SECRET_LENGTH` characters long are included; shorter strings
    are too prone to false positives.
    """
    # Refresh names and stored values together before taking this operation's
    # snapshot: a rotated .env value must be visible on the first redaction.
    names = credential_env_var_names()
    with _lock:
        stored = dict(_dotenv_entries) if _dotenv_entries is not None else {}
    pairs: list[tuple[str, str]] = []
    for name in sorted(names):
        for value in (os.environ.get(name), stored.get(name), _keyring_value(name)):
            if value and len(value) >= MIN_SECRET_LENGTH:
                pairs.append((name, value))
    return pairs


def _oauth_secret_values() -> frozenset[str]:
    # Consult only configured OAuth token usernames, not the whole user keyring.
    for username in current_policy().oauth_records:
        register_mcp_oauth_record(username, _keyring_value(username))
    with _lock:
        return frozenset(value for record in _oauth_tokens.values() for value in record)


def known_secret_values() -> frozenset[str]:
    """The concrete secret values chartreux has loaded, plus base64 forms.

    The base64 encodings of the bare value and of the ``NAME=value`` .env line
    shapes are included so obfuscated exfiltration (``base64 ~/.chartreux/.env``)
    is caught by the fast path even without the decode-and-scan backstop, as is
    the reversed value (``rev`` on the value or the .env file).
    """
    values: set[str] = set()
    policy = current_policy()
    for name, value in (*_loaded_credentials(), *policy.redaction_credentials):
        values.add(value)
        values.add(value[::-1])
        values.update(_percent_variants(value))
        values.update(_b64_variants(value))
        for line in (f"{name}={value}", f"{name}={value}\n"):
            values.update(_b64_variants(line))
    for value in _oauth_secret_values() | policy.redaction_oauth_values:
        values.update((value, value[::-1]))
        values.update(_b64_variants(value))
        values.update(_percent_variants(value))
    return frozenset(values)


def _percent_variants(value: str) -> set[str]:
    """Recognize full and single-character percent escaping (either hex case)."""
    variants = {"".join(f"%{byte:02X}" for byte in value.encode("utf-8"))}
    # Building one whole-string copy per character is quadratic in the secret
    # length; large credentials remain covered by direct/whole-body matching.
    if len(value) > _MAX_SINGLE_PERCENT_VARIANT_LENGTH:
        return variants
    for index, character in enumerate(value):
        encoded = "".join(f"%{byte:02X}" for byte in character.encode("utf-8"))
        for spelling in (encoded, encoded.lower()):
            variants.add(value[:index] + spelling + value[index + 1 :])
    return variants


def _b64_variants(text: str) -> set[str]:
    """Standard base64 of *text*, plus the wrapped form ``base64`` emits."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    variants = {encoded}
    if len(encoded) > _B64_WRAP_WIDTH:
        wrapped = "\n".join(
            encoded[i : i + _B64_WRAP_WIDTH]
            for i in range(0, len(encoded), _B64_WRAP_WIDTH)
        )
        variants.add(wrapped)
    return variants


def redact(text: str) -> str:
    """Replace every known secret value in *text* with a placeholder.

    Cheap when no secret is loaded: the value set is empty and the text is
    returned unchanged. The decode-and-scan backstop runs first so an encoded
    run is examined whole — a fast-path replacement inside it would garble the
    alignment — then the direct-value variants (including the base64 forms
    chartreux computed itself) replace whatever is left.
    """
    values = known_secret_values()
    if not values:
        return text
    text = _redact_encoded_runs(text)
    return _replace_values(text, values)


def _redact_encoded_runs(text: str) -> str:
    """Decode-and-scan backstop for encoded secret carriers.

    The fast path only matches encodings chartreux computed itself, so a
    whole-file or wrapped encoding whose lines do not align with those
    variants slips through. Instead of enumerating encodings, candidate runs
    of base64/hex/base32 alphabet characters are decoded, and a run whose
    decoded bytes reveal a known secret value — or a ``NAME=value`` line whose
    NAME is a known credential name — is redacted entirely. Reversed payloads
    (``rev chartreux.env | base64``) are caught the same way.

    Out of scope: gzip|base64 (compressed payloads; recovering the plaintext
    would need a decompression pipeline, not just a decode) and wrap widths
    below ``_MIN_ENCODED_RUN``.
    """
    spans: list[tuple[int, int]] = []
    # Loaded lazily: only a decodable candidate needs the raw value set.
    reveals: tuple[frozenset[str], frozenset[str]] | None = None
    for start, end, body in _encoded_candidates(text):
        decoded = _decode_candidate(body)
        if not decoded:
            continue
        if reveals is None:
            reveals = (
                frozenset(value for _, value in _loaded_credentials())
                | frozenset(
                    value for _, value in current_policy().redaction_credentials
                )
                | _oauth_secret_values()
                | current_policy().redaction_oauth_values,
                credential_env_var_names(),
            )
        if any(_decoded_reveals_secret(payload, *reveals) for payload in decoded):
            spans.append((start, end))
    if not spans:
        return text
    # Merge overlapping candidates (a wrapped run and its individual lines)
    # and replace right-to-left so earlier spans keep their offsets.
    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for start, end in reversed(merged):
        text = text[:start] + REDACTED_PLACEHOLDER + text[end:]
    return text


def _encoded_candidates(text: str) -> list[tuple[int, int, str]]:
    """Candidate encoded runs as ``(start, end, whitespace-stripped body)``.

    A candidate is a run of base64-alphabet characters (hex and base32 are
    subsets), or several such runs joined across whitespace — which is how
    ``base64``/``base32``/``openssl`` wrap long payloads. Wrapped encodings are
    anchored at runs of long words (a wrap line is 60-76 characters) so
    adjacent prose words cannot shift the decode alignment; a single short
    trailing word is also tried because a wrap's final line can be short. The
    list is capped at ``_MAX_ENCODED_CANDIDATES`` so a huge output cannot blow
    up the scan.

    Out of scope: wrap widths below ``_MIN_ENCODED_RUN`` (e.g. ``base64 -w 8``)
    produce no run long enough to trip the pre-filter below.
    """
    if (
        not _ENCODED_RUN_SEARCH.search(text)
        and "%" not in text
        and not _OD_ROW.search(text)
    ):
        # No run long enough to hide a secret: skip the per-word scan.
        return []
    candidates: list[tuple[int, int, str]] = []
    group: list[tuple[int, int, str]] = []

    def emit(start: int, end: int, body: str) -> None:
        if len(body) < _MIN_ENCODED_RUN:
            return
        percent = body.startswith("%") and _PERCENT_RUN.fullmatch(body) is not None
        step = 7167 if percent else _ENCODED_WINDOW_STEP
        width = 8190 if percent else _MAX_ENCODED_CANDIDATE_CHARS
        for offset in range(0, len(body), step):
            if len(candidates) >= _MAX_ENCODED_CANDIDATES:
                break
            # Four-character alignment preserves base64/base32; percent runs
            # instead require three-character alignment at both window edges.
            window = body[offset : offset + width]
            if len(window) < _MIN_ENCODED_RUN:
                break
            candidates.append((start, end, window))
            if offset + width >= len(body):
                break

    def flush() -> None:
        if not group:
            return
        for start, end, word in group:
            emit(start, end, word)
        i = 0
        while i < len(group):
            if len(group[i][2]) < _MIN_ENCODED_RUN:
                i += 1
                continue
            anchor = i
            while i + 1 < len(group) and len(group[i + 1][2]) >= _MIN_ENCODED_RUN:
                i += 1
            if i > anchor:
                emit(
                    group[anchor][0],
                    group[i][1],
                    "".join(w for _, _, w in group[anchor : i + 1]),
                )
            if i + 1 < len(group):
                emit(
                    group[anchor][0],
                    group[i + 1][1],
                    "".join(w for _, _, w in group[anchor : i + 2]),
                )
            i += 1
        group.clear()

    if _ENCODED_RUN_SEARCH.search(text):
        for match in _ENCODED_WORD.finditer(text):
            if len(candidates) >= _MAX_ENCODED_CANDIDATES:
                break
            start, end = match.span()
            if group and not text[group[-1][1] : start].isspace():
                # The gap holds punctuation, so the run is only part of a word.
                flush()
            group.append((start, end, match.group()))
        flush()
    # od -tx1 prints byte pairs separated by spaces, optionally with octal
    # offsets. Only complete rows are admitted, never arbitrary prose words.
    for start, end, body in _spaced_hex_candidates(text):
        if len(candidates) >= _MAX_ENCODED_CANDIDATES:
            break
        emit(start, end, body)
    if "%" in text:
        for match in _PERCENT_MIXED.finditer(text):
            if len(candidates) >= _MAX_ENCODED_CANDIDATES:
                break
            if "%" in match.group():
                emit(*match.span(), match.group())
    return candidates


def _spaced_hex_candidates(text: str) -> Iterator[tuple[int, int, str]]:
    row_group: list[re.Match[str]] = []
    for row in _OD_ROW.finditer(text):
        if row_group and text[row_group[-1].end() : row.start()] != "\n":
            yield (
                row_group[0].start(),
                row_group[-1].end(),
                "".join(_od_row_hex(item.group()) for item in row_group),
            )
            row_group.clear()
        row_group.append(row)
    if row_group:
        yield (
            row_group[0].start(),
            row_group[-1].end(),
            "".join(_od_row_hex(item.group()) for item in row_group),
        )


def _od_row_hex(row: str) -> str:
    words = row.split()
    if (
        words
        and len(words[0]) >= _OD_OFFSET_WIDTH
        and all(c in "01234567" for c in words[0])
    ):
        words = words[1:]
    return "".join(words)


def _decode_candidate(body: str) -> list[str]:
    """Best-effort percent, base64(url), hex, base32, at most two layers."""

    def once(candidate: str) -> list[str]:
        decoded: list[str] = []
        if "%" in candidate and not re.search(r"%(?![0-9a-fA-F]{2})", candidate):
            decoded.append(unquote(candidate))
        if len(candidate) % 4 != 1:
            padded = candidate + "=" * (-len(candidate) % 4)
            for alphabet in (False, True):
                try:
                    data = (
                        padded.translate(str.maketrans("-_", "+/"))
                        if alphabet
                        else padded
                    )
                    decoded.append(
                        base64.b64decode(data, validate=True).decode(
                            "utf-8", errors="ignore"
                        )
                    )
                except ValueError:
                    pass
        try:
            decoded.append(bytes.fromhex(candidate).decode("utf-8", errors="ignore"))
        except ValueError:
            pass
        try:
            decoded.append(
                base64.b32decode(candidate, casefold=True).decode(
                    "utf-8", errors="ignore"
                )
            )
        except ValueError:
            pass
        return [item for item in decoded if item]

    first = once(body)
    return first + [second for item in first for second in once(item)]


def _decoded_reveals_secret(
    payload: str, raw_values: frozenset[str], names: frozenset[str]
) -> bool:
    # Reversed too: `rev chartreux.env | base64` decodes to reversed lines.
    if any(
        (secret in payload or secret[::-1] in payload)
        for secret in raw_values
        if len(secret) >= _MIN_ENCODED_SECRET_LENGTH
    ):
        return True
    # A whole-file dump (``base64 ~/.chartreux/.env``) decodes to NAME=value
    # lines; any line naming a known credential is a leak even when that
    # particular value was never loaded.
    for line in payload.splitlines():
        name, sep, _ = line.partition("=")
        if sep and name.strip() in names:
            return True
        if sep and name.strip()[::-1] in names:
            return True
    return False


def _replace_values(text: str, values: frozenset[str]) -> str:
    # Longest first so overlapping occurrences collapse to one placeholder.
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, REDACTED_PLACEHOLDER)
    return text


def redact_json_value(value: Any) -> Any:
    """Sanitize JSON payload values and keys without modifying schema field names."""
    if not known_secret_values():
        return value
    return _redact_json(value)


def _redact_json(value: Any, depth: int = 0) -> Any:
    # Drop the entire subtree, including its keys, before recursion can fail.
    if depth >= _MAX_JSON_REDACTION_DEPTH and isinstance(value, (dict, list)):
        return REDACTED_PLACEHOLDER
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        changed = False
        items: dict[Any, Any] = {}
        # Reserve unchanged keys first so a redacted key never overwrites one.
        reserved = set(value)
        for key, item in value.items():
            sanitized_key = redact(key) if isinstance(key, str) else key
            sanitized_item = _redact_json(item, depth + 1)
            if sanitized_key != key:
                reserved.discard(key)
                base = sanitized_key
                suffix = 2
                while sanitized_key in reserved or sanitized_key in items:
                    sanitized_key = f"{base}#{suffix}"
                    suffix += 1
            changed |= sanitized_key != key or sanitized_item is not item
            items[sanitized_key] = sanitized_item
        return items if changed else value
    if isinstance(value, list):
        elements = [_redact_json(item, depth + 1) for item in value]
        return (
            elements
            if any(a is not b for a, b in zip(value, elements, strict=True))
            else value
        )
    return value


def redact_model(model: Any) -> Any:
    """Sanitize model data recursively, keeping model field names intact.

    Reconstruction failures must be handled by callers with a generic safe
    replacement; they must never emit the untrusted original.
    """
    from pydantic import BaseModel

    if not isinstance(model, BaseModel):
        return redact_json_value(model)
    updates: dict[str, Any] = {}
    for name in type(model).model_fields:
        original = getattr(model, name)
        if isinstance(original, BaseModel):
            replacement = redact_model(original)
        elif isinstance(original, list):
            replacement = [redact_model(item) for item in original]
        else:
            replacement = redact_json_value(original)
        if replacement is not original and replacement != original:
            updates[name] = replacement
    if not updates:
        return model
    return type(model).model_validate({
        **{name: getattr(model, name) for name in type(model).model_fields},
        **updates,
    })


def redact_persisted_result(result: Any) -> Any:
    """Redact secret values from a ``PersistedToolResult``-shaped object.

    The result is sanitized recursively, preserving fixed model field names.
    Reconstruction failures use a sanitized base result or a generic safe
    replacement; untrusted content is never returned on failure.
    """
    values = known_secret_values()
    if not values:
        return result
    try:
        result.model_dump(mode="json")
        # Model field names are schema, not payload dictionary keys.
        return redact_model(result)
    except Exception:
        logger.debug("redact_persisted_result reconstruction failed", exc_info=True)
        from chartreux.core.llm_models import PersistedToolResult

        try:
            return PersistedToolResult(
                output=redact_json_value(result.output),
                duration=result.duration,
                cancelled=result.cancelled,
                presentation=redact_model(result.presentation)
                if result.presentation is not None
                else None,
            )
        except Exception:
            logger.debug("redact_persisted_result fallback failed", exc_info=True)
            return PersistedToolResult(output={"error": "Tool result unavailable"})


def set_env_passthrough(names: Iterable[str]) -> None:
    """Set passthrough in this context only (legacy API; prefer binding a policy)."""
    policy = current_policy()
    _current_policy.set(
        ScrubPolicy(frozenset(names), policy.credential_names, policy.oauth_records)
    )


def env_passthrough(policy: ScrubPolicy | None = None) -> frozenset[str]:
    return (policy or current_policy()).passthrough


def credential_env_scrub_names(policy: ScrubPolicy | None = None) -> frozenset[str]:
    """Credential variable names to remove from child environments."""
    policy = policy or current_policy()
    return credential_env_var_names(policy) - policy.passthrough


def scrub_child_env(
    env: Mapping[str, str], policy: ScrubPolicy | None = None
) -> dict[str, str]:
    """Copy *env* without chartreux's credential variables (minus passthrough).

    The policy only captures names. Rotated values are read at use time via the
    shared .env/keyring cache; an already inherited child environment cannot be
    scrubbed retroactively, so persistent connections must be retired on reload.
    """
    scrub = credential_env_scrub_names(policy)
    if not scrub:
        return dict(env)
    return {name: value for name, value in env.items() if name not in scrub}


def child_env_scrub_list(policy: ScrubPolicy | None = None) -> list[str]:
    """Scrub names as a sorted list, for protocols that carry a scrub list."""
    return sorted(credential_env_scrub_names(policy))


def reset_cache() -> None:
    """Drop cached credential names, stored .env values, and keyring lookups.

    Used by tests and by the agent loop when a config reload goes live.
    """
    global _credential_names, _credential_names_token, _dotenv_entries
    with _lock:
        _credential_names = None
        _credential_names_token = None
        _dotenv_entries = None
        _keyring_values.clear()
        _oauth_tokens.clear()


__all__ = [
    "MIN_SECRET_LENGTH",
    "REDACTED_PLACEHOLDER",
    "ScrubPolicy",
    "bind_policy",
    "child_env_scrub_list",
    "credential_env_scrub_names",
    "credential_env_var_names",
    "current_policy",
    "env_passthrough",
    "known_secret_values",
    "mcp_static_auth_env_names",
    "redact",
    "redact_json_value",
    "redact_persisted_result",
    "reset_cache",
    "scrub_child_env",
    "set_env_passthrough",
    "set_mcp_static_auth_env_names",
]
