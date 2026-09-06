"""Scoring layer tests.

Deliberately small. Each test here covers a decision that would be expensive
to get wrong, not a line of code.
"""

from __future__ import annotations

import pytest

from docbench.metrics import (
    bootstrap_ci,
    canonicalize,
    critical_field_error_rate,
    score_field,
    silent_failure_rate,
)
from docbench.types import (
    Criticality,
    FieldKind,
    FieldOutcome,
    FieldPrediction,
    FieldSpec,
    MatchStatus,
)

DOB = FieldSpec(name="dob", kind=FieldKind.DATE, criticality=Criticality.CRITICAL)
NAME = FieldSpec(name="name", kind=FieldKind.NAME, criticality=Criticality.CRITICAL)


def pred(value, confidence=None):
    return FieldPrediction(name="dob", value=value, confidence=confidence)


def outcome(status, criticality=Criticality.CRITICAL):
    return FieldOutcome(name="f", kind=FieldKind.TEXT, criticality=criticality, status=status)


class TestBareStringGuard:
    """A str satisfies Sequence[str], so it iterates as characters and scores
    silently wrong. No type checker catches this; the runtime guard does."""

    def test_bare_str_gold_is_rejected(self):
        with pytest.raises(TypeError, match="not a bare str"):
            score_field(NAME, "Sakshi Raut", pred("Sakshi Raut"))

    def test_sequence_gold_is_accepted(self):
        assert score_field(NAME, ["Sakshi Raut"], pred("Sakshi Raut")).status is MatchStatus.EXACT


class TestMatchStatus:
    def test_normalized_match_ignores_formatting(self):
        assert score_field(DOB, ["01/02/1990"], pred("1990-02-01")).status is MatchStatus.NORMALIZED

    def test_wrong_value(self):
        assert score_field(DOB, ["1990-02-01"], pred("1991-02-01")).status is MatchStatus.WRONG

    def test_missing_when_nothing_predicted(self):
        assert score_field(DOB, ["1990-02-01"], None).status is MatchStatus.MISSING


class TestSilentFailure:
    def test_unparseable_value_is_not_well_formed(self):
        """An impossible date fails validation loudly, so it is never silent."""
        result = score_field(DOB, ["1990-02-01"], pred("1991-13-45", confidence=0.99))
        assert result.status is MatchStatus.WRONG
        assert not result.well_formed
        assert silent_failure_rate([result], 0.7) == 0.0

    def test_confident_wrong_and_well_formed_is_silent(self):
        result = score_field(DOB, ["1990-02-01"], pred("1991-02-01", confidence=0.95))
        assert silent_failure_rate([result], 0.7) == 1.0

    def test_low_confidence_is_not_silent(self):
        """Low confidence routes the case to review, so it is caught."""
        result = score_field(DOB, ["1990-02-01"], pred("1991-02-01", confidence=0.2))
        assert silent_failure_rate([result], 0.7) == 0.0


class TestCFER:
    def test_missing_costs_less_than_wrong(self):
        """The central design claim: a loud failure is cheaper than a silent one."""
        assert critical_field_error_rate(
            [outcome(MatchStatus.MISSING)]
        ) < critical_field_error_rate([outcome(MatchStatus.WRONG)])

    def test_criticality_is_weighted(self):
        critical = critical_field_error_rate(
            [
                outcome(MatchStatus.WRONG, Criticality.CRITICAL),
                outcome(MatchStatus.EXACT, Criticality.COSMETIC),
            ]
        )
        cosmetic = critical_field_error_rate(
            [
                outcome(MatchStatus.EXACT, Criticality.CRITICAL),
                outcome(MatchStatus.WRONG, Criticality.COSMETIC),
            ]
        )
        assert critical > cosmetic


class TestDateCanonicalisation:
    def test_iso_dates_are_not_day_flipped(self):
        """Regression: dateutil with dayfirst=True reads 1990-02-01 as 2 January.

        A confident, well-formed, wrong date -- the exact failure class this
        project measures, found in its own canonicaliser.
        """
        assert canonicalize("1990-02-01", FieldKind.DATE) == "1990-02-01"


class TestBootstrap:
    def test_small_samples_give_a_wide_interval(self):
        """n=18 without an interval is not a finding. This is why."""
        small = [outcome(MatchStatus.WRONG)] * 5 + [outcome(MatchStatus.EXACT)] * 13
        point, low, high = bootstrap_ci(small, "cfer", n_resamples=800)
        assert low <= point <= high
        assert high - low > 0.2
