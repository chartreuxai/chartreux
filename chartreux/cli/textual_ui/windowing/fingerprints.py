"""Compact deterministic fingerprints for transcript history entries.

A fingerprint answers one question cheaply: *could this record differ?* The
correctness direction is one-way. Fingerprint equality must imply deep-copy
equality, because a missed invalidation leaves a stale placement on screen;
the converse (equal records sharing a fingerprint) is an efficiency goal only.

Canonicalization rules, each recorded against the equality behaviour it keeps:

* Model records canonicalize from ``__dict__`` recursively with every mapping
  sorted at each level. A top-level ``sort_keys`` alone would leave nested
  ``JsonValue`` payload order significant and turn equal entries into spurious
  invalidations.
* Scalars are type-tagged. ``1``/``1.0`` and ``True``/``1`` are pydantic-equal
  yet reconstruct differently, so they fingerprint differently — the accepted
  spurious-invalidation direction.
* Timezone-aware datetimes normalize to their UTC instant: equal instants in
  different zones are pydantic-equal but serialize differently. That holds only
  for a wall time naming a single instant, and only while the instant stays
  representable: an extreme-but-valid datetime such as
  0001-01-01T00:00:00+01:00 or 9999-12-31T23:59:59-01:00 overflows the
  datetime range during UTC normalization and is always-changed.
* A fold-ambiguous wall time (``fold`` selects between two offsets, as in the
  repeated hour of a DST fall-back) is always-changed. CPython compares equal
  wall clocks field-by-field only while both sides share one ``tzinfo`` object,
  and otherwise never equates an ambiguous wall time at all — not even when
  instant, wall clock, offset, zone name and fold all agree. Two records whose
  every payload value matches can therefore be unequal, which no value digest
  can preserve.
* Naive and aware datetimes stay canonically distinct: a naive 12:00 and an
  aware 12:00+00:00 are pydantic-unequal, so normalizing them together would be
  a missed invalidation.
* Entries containing NaN are always-changed: NaN is unequal to a distinct NaN,
  so any value-based encoding would collide across unequal entries.
* Entries with non-string mapping keys are always-changed: ``{1: 'a'}`` and
  ``{'1': 'a'}`` stringify identically (a collision) and mixed key types cannot
  be ordered at all.
* Tuple payload values are always-changed: unreachable through wire validation
  (entries arrive forced to lists) but deep-unequal to the list they would
  encode like.
* Payload strings holding a lone surrogate (what JSON-decoding an escaped
  surrogate such as ``\ud800`` produces) are always-changed: UTF-8 cannot
  encode them, so no canonical byte form exists.
* Any other unencodable value is always-changed, which is the safe direction.

Two structural classes cannot collide through these rules: ``extra='forbid'``
on the entry models keeps unknown fields out, and the discriminator fields
(``type``, ``status``, ``kind``) are ordinary model fields, so they always
serialize.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
import hashlib
from itertools import count
import math
from typing import Final

from pydantic import BaseModel

type Fingerprint = bytes

DIGEST_SIZE: Final[int] = 16
"""Bytes in every fingerprint, so all domains compare at one fixed width."""

_ALWAYS_CHANGED_TAG: Final[bytes] = b"\xff"
_NULL: Final[bytes] = b"\x00"
_FALSE: Final[bytes] = b"\x01"
_TRUE: Final[bytes] = b"\x02"
_INT: Final[bytes] = b"\x03"
_FLOAT: Final[bytes] = b"\x04"
_STR: Final[bytes] = b"\x05"
_BYTES: Final[bytes] = b"\x06"
_MAPPING: Final[bytes] = b"\x07"
_SEQUENCE: Final[bytes] = b"\x08"
_MODEL: Final[bytes] = b"\x09"
_NAIVE_DATETIME: Final[bytes] = b"\x0a"
_AWARE_DATETIME: Final[bytes] = b"\x0b"

_always_changed = count(1)


class _Unfingerprintable(Exception):
    """A value whose equality no value-based digest can preserve."""


def fingerprint_entry(entry: BaseModel) -> Fingerprint:
    """Fingerprint one history entry's canonical content.

    Entries whose payload cannot be canonicalized soundly (NaN, non-string
    mapping keys, tuples, unknown types) fingerprint as always-changed, so
    they force a refresh instead of risking a stale one.
    """
    return _digest(entry)


def _digest(value: object) -> Fingerprint:
    canonical = bytearray()
    try:
        _encode(value, canonical)
    except (_Unfingerprintable, ValueError, OverflowError):
        # The canonical form is UTF-8 bytes, so a payload string the codec
        # rejects (a lone surrogate raises UnicodeEncodeError, a ValueError)
        # has no fingerprint at all, and an extreme-but-valid datetime whose
        # UTC instant leaves the representable range (0001-01-01T00:00:00+01:00,
        # 9999-12-31T23:59:59-01:00) raises OverflowError during normalization.
        # Either way fall back instead of failing admission.
        return _always_changed_fingerprint()
    return hashlib.blake2b(canonical, digest_size=DIGEST_SIZE).digest()


def _always_changed_fingerprint() -> Fingerprint:
    nonce = next(_always_changed).to_bytes(DIGEST_SIZE - 1, "big")
    return _ALWAYS_CHANGED_TAG + nonce


def _encode(value: object, out: bytearray) -> None:
    if value is None:
        out += _NULL
    elif isinstance(value, BaseModel):
        _encode_model(value, out)
    elif isinstance(value, Enum):
        # StrEnum/IntEnum members compare equal to their values, so encode the
        # value rather than the member or the class.
        _encode(value.value, out)
    elif isinstance(value, bool):
        out += _TRUE if value else _FALSE
    elif isinstance(value, int):
        _encode_tagged(out, _INT, str(value))
    elif isinstance(value, float):
        _encode_float(value, out)
    elif isinstance(value, str):
        _encode_tagged(out, _STR, value)
    elif isinstance(value, bytes):
        _encode_tagged(out, _BYTES, value)
    elif isinstance(value, datetime):
        _encode_datetime(value, out)
    elif isinstance(value, list):
        _encode_sequence(value, out)
    elif isinstance(value, dict):
        _encode_mapping(value, out)
    else:
        raise _Unfingerprintable(f"unsupported payload type: {type(value).__name__}")


def _encode_model(model: BaseModel, out: bytearray) -> None:
    out += _MODEL
    # Model equality also requires the same class, so the fully qualified class
    # name belongs in the canonical form.
    name = f"{type(model).__module__}.{type(model).__qualname__}"
    _encode_tagged(out, _STR, name)
    _encode_mapping(model.__dict__, out)


def _encode_float(value: float, out: bytearray) -> None:
    if math.isnan(value):
        raise _Unfingerprintable("NaN payload")
    _encode_tagged(out, _FLOAT, repr(value))


def _encode_datetime(value: datetime, out: bytearray) -> None:
    if value.utcoffset() is None:
        _encode_tagged(out, _NAIVE_DATETIME, value.isoformat())
        return
    if _is_fold_ambiguous(value):
        raise _Unfingerprintable("fold-ambiguous datetime")
    _encode_tagged(out, _AWARE_DATETIME, value.astimezone(UTC).isoformat())


def _is_fold_ambiguous(value: datetime) -> bool:
    """Report whether fold picks between two offsets, so the wall time repeats.

    Datetimes with a single offset compare by instant across zones, which the
    UTC normalization preserves. Fold-ambiguous ones do not: CPython refuses to
    equate them with a datetime holding another ``tzinfo`` object even when
    every field agrees, so their equality is not a function of their values.
    """
    return value.replace(fold=0).utcoffset() != value.replace(fold=1).utcoffset()


def _encode_sequence(values: list[object], out: bytearray) -> None:
    out += _SEQUENCE
    out += len(values).to_bytes(4, "big")
    for value in values:
        _encode(value, out)


def _encode_mapping(mapping: Mapping[str, object], out: bytearray) -> None:
    keys: list[str] = []
    for key in mapping:
        # Payload dicts can hold non-string keys at runtime even though the
        # annotation is str-only; equal-looking keys would collide.
        if not isinstance(key, str):
            raise _Unfingerprintable("non-string mapping key")
        keys.append(key)
    keys.sort()
    out += _MAPPING
    out += len(keys).to_bytes(4, "big")
    for key in keys:
        _encode_tagged(out, _STR, key)
        _encode(mapping[key], out)


def _encode_tagged(out: bytearray, tag: bytes, payload: str | bytes) -> None:
    out += tag
    data = payload.encode() if isinstance(payload, str) else payload
    out += len(data).to_bytes(4, "big")
    out += data
