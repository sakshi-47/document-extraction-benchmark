"""Split assignment tests.

These guarantee the property that makes the benchmark trustworthy: a document's
split never changes. Test-set contamination via a reshuffle is invisible in
every metric you would think to check, so it is prevented structurally.
"""

from __future__ import annotations

import random

import pytest

from docbench.data import assign_split, split_counts

SALT = "docbench-v1"
IDS = [f"doc-{i:05d}" for i in range(5000)]


def test_independent_of_iteration_order():
    """A seeded shuffle is reproducible only if input order is. Hashing is not."""
    forward = {d: assign_split(d, SALT) for d in IDS}
    shuffled = IDS.copy()
    random.Random(0).shuffle(shuffled)
    assert {d: assign_split(d, SALT) for d in shuffled} == forward


def test_adding_documents_does_not_move_existing_ones():
    before = {d: assign_split(d, SALT) for d in IDS[:100]}
    _ = [assign_split(f"new-{i}", SALT) for i in range(10_000)]
    assert {d: assign_split(d, SALT) for d in IDS[:100]} == before


def test_ratios_are_approximately_respected():
    counts = split_counts(IDS, SALT, train=0.8, val=0.1)
    total = sum(counts.values())
    assert total == len(IDS)
    assert counts["train"] / total == pytest.approx(0.8, abs=0.02)
    assert counts["test"] / total == pytest.approx(0.1, abs=0.02)


def test_impossible_ratios_are_rejected():
    with pytest.raises(ValueError):
        assign_split("doc-1", SALT, train=0.9, val=0.2)
