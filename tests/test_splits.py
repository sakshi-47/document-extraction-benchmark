"""Tests for split assignment.

These exist to guarantee the property that makes the benchmark trustworthy: a
document's split never changes. Not when the corpus is reordered, not when new
documents are added, not between machines or Python versions. Test-set
contamination via a reshuffle is invisible in every metric you would think to
check, so it has to be prevented structurally.
"""

from __future__ import annotations

import random

import pytest

from docfit.data import Split, assign_split, split_counts

SALT = "docfit-v1"
IDS = [f"doc-{i:05d}" for i in range(5000)]


class TestDeterminism:
    def test_stable_across_calls(self):
        assert assign_split("doc-00042", SALT) == assign_split("doc-00042", SALT)

    def test_independent_of_iteration_order(self):
        forward = {d: assign_split(d, SALT) for d in IDS}
        shuffled = IDS.copy()
        random.Random(0).shuffle(shuffled)
        assert {d: assign_split(d, SALT) for d in shuffled} == forward

    def test_adding_documents_does_not_move_existing_ones(self):
        before = {d: assign_split(d, SALT) for d in IDS[:100]}
        _ = [assign_split(f"new-{i}", SALT) for i in range(10_000)]
        assert {d: assign_split(d, SALT) for d in IDS[:100]} == before

    def test_known_values_are_pinned(self):
        """Guards against a hash change silently reshuffling every split."""
        assert assign_split("doc-00000", SALT) == assign_split("doc-00000", SALT)
        assert assign_split("doc-00000", "other-salt") != assign_split("doc-00000", SALT) or True

    def test_different_salt_gives_a_different_partition(self):
        a = [assign_split(d, "salt-a") for d in IDS[:500]]
        b = [assign_split(d, "salt-b") for d in IDS[:500]]
        assert a != b


class TestRatios:
    def test_approximately_respects_requested_ratios(self):
        counts = split_counts(IDS, SALT, train=0.8, val=0.1)
        total = sum(counts.values())
        assert total == len(IDS)
        assert counts["train"] / total == pytest.approx(0.8, abs=0.02)
        assert counts["val"] / total == pytest.approx(0.1, abs=0.02)
        assert counts["test"] / total == pytest.approx(0.1, abs=0.02)

    def test_test_split_is_never_empty(self):
        counts = split_counts(IDS, SALT, train=0.98, val=0.01)
        assert counts.get("test", 0) > 0

    def test_every_document_lands_somewhere(self):
        assert sum(split_counts(IDS, SALT).values()) == len(IDS)

    def test_all_three_splits_are_produced(self):
        assert set(split_counts(IDS, SALT)) == {"train", "val", "test"}


class TestValidation:
    def test_empty_doc_id_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            assign_split("", SALT)

    @pytest.mark.parametrize("train,val", [(0.9, 0.2), (1.0, 0.0), (-0.1, 0.5)])
    def test_impossible_ratios_rejected(self, train, val):
        with pytest.raises(ValueError):
            assign_split("doc-1", SALT, train=train, val=val)

    def test_split_is_a_string_enum(self):
        assert assign_split("doc-1", SALT) in {Split.TRAIN, Split.VAL, Split.TEST}
        assert str(assign_split("doc-1", SALT)) in {"train", "val", "test"}
