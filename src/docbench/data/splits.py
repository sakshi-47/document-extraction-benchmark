"""Deterministic, order-independent train/validation/test assignment.

Why not `random.shuffle(ids)` with a fixed seed
-----------------------------------------------
A seeded shuffle is reproducible only if the input list is in exactly the same
order every time. That order depends on filesystem iteration, dataset library
version, and whether anything was added. When it changes, documents silently
migrate between train and test, the test set is quietly contaminated, and the
reported accuracy goes up for a reason nobody can see.

Hashing each document id into a bucket removes the dependency entirely. A given
id lands in the same split forever, regardless of iteration order, how many
other documents exist, or whether new ones are added later.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum

_BUCKETS = 10_000


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def _bucket(doc_id: str, salt: str) -> int:
    """Map a document id to a stable bucket in [0, _BUCKETS).

    SHA-256 is used as a stable, well-distributed hash across platforms and
    Python versions -- not as a security primitive. `hash()` would be wrong
    here: it is randomised per process by default.
    """
    digest = hashlib.sha256(f"{salt}:{doc_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % _BUCKETS


def assign_split(
    doc_id: str,
    salt: str,
    train: float = 0.8,
    val: float = 0.1,
) -> Split:
    """Assign one document to a split. Same id and salt always give the same result.

    `test` takes the remainder, so the test set can never be accidentally
    emptied by ratios that fail to sum to one.
    """
    if not doc_id:
        raise ValueError("doc_id must be non-empty")
    if train < 0 or val < 0 or train + val >= 1.0:
        raise ValueError(f"train + val must be in [0, 1); got {train} + {val}")

    bucket = _bucket(doc_id, salt)
    if bucket < train * _BUCKETS:
        return Split.TRAIN
    if bucket < (train + val) * _BUCKETS:
        return Split.VAL
    return Split.TEST


def split_counts(
    doc_ids: Iterable[str],
    salt: str,
    train: float = 0.8,
    val: float = 0.1,
) -> dict[str, int]:
    """Count documents per split. Report this in the README; splits are a claim."""
    counter = Counter(assign_split(d, salt, train, val).value for d in doc_ids)
    return dict(sorted(counter.items()))
