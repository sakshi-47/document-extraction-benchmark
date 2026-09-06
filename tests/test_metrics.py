"""Tests for the scoring layer.

The first test class is the reason this file exists. A scorer that takes a
sequence of accepted answers will silently iterate a bare string as characters,
scoring "Yes" against 'Y', 'e', 's'. Nothing raises and every resulting number
is meaningless. One test catches it.
"""

from __future__ import annotations

import pytest

from docbench.metrics import (
    bootstrap_ci,
    canonicalize,
    critical_field_error_rate,
    error_profile,
    score_document,
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
ADDRESS = FieldSpec(name="address", kind=FieldKind.TEXT, criticality=Criticality.STANDARD)
DOCNO = FieldSpec(name="doc_no", kind=FieldKind.ALNUM_ID, criticality=Criticality.CRITICAL)


def outcome(status: MatchStatus, criticality=Criticality.CRITICAL, **kw) -> FieldOutcome:
    return FieldOutcome(
        name=kw.pop("name", "f"),
        kind=kw.pop("kind", FieldKind.TEXT),
        criticality=criticality,
        status=status,
        **kw,
    )


class TestBareStringGuard:
    """Regression tests for the bare-string character-iteration hazard."""

    def test_bare_str_gold_is_rejected(self):
        with pytest.raises(TypeError, match="not a bare str"):
            score_field(NAME, "Sakshi Raut", FieldPrediction(name="name", value="Sakshi Raut"))

    def test_error_message_shows_the_fix(self):
        with pytest.raises(TypeError, match=r"pass \['Yes'\] instead"):
            score_field(ADDRESS, "Yes", FieldPrediction(name="address", value="Yes"))

    def test_sequence_gold_is_accepted(self):
        result = score_field(
            NAME, ["Sakshi Raut"], FieldPrediction(name="name", value="Sakshi Raut")
        )
        assert result.status is MatchStatus.EXACT

    def test_the_bug_would_have_shown_here(self):
        """Under the old behaviour this scored as a partial character match."""
        result = score_field(ADDRESS, ["Yes"], FieldPrediction(name="address", value="No"))
        assert result.status is MatchStatus.WRONG


class TestMatchStatus:
    def test_exact(self):
        assert (
            score_field(DOCNO, ["AB1234"], FieldPrediction(name="doc_no", value="AB1234")).status
            is MatchStatus.EXACT
        )

    def test_normalized_date(self):
        r = score_field(DOB, ["01/02/1990"], FieldPrediction(name="dob", value="1990-02-01"))
        assert r.status is MatchStatus.NORMALIZED

    def test_normalized_name_order(self):
        r = score_field(NAME, ["Raut, Sakshi"], FieldPrediction(name="name", value="Sakshi Raut"))
        assert r.status is MatchStatus.NORMALIZED

    def test_wrong(self):
        r = score_field(DOB, ["1990-02-01"], FieldPrediction(name="dob", value="1991-02-01"))
        assert r.status is MatchStatus.WRONG

    def test_missing(self):
        assert score_field(DOB, ["1990-02-01"], None).status is MatchStatus.MISSING

    def test_missing_when_value_is_blank(self):
        r = score_field(DOB, ["1990-02-01"], FieldPrediction(name="dob", value="   "))
        assert r.status is MatchStatus.MISSING

    def test_spurious(self):
        r = score_field(DOB, [], FieldPrediction(name="dob", value="1990-02-01"))
        assert r.status is MatchStatus.SPURIOUS

    def test_true_negative(self):
        assert score_field(DOB, [], None).status is MatchStatus.TRUE_NEGATIVE

    def test_any_accepted_variant_matches(self):
        r = score_field(
            NAME, ["S Raut", "Sakshi Raut"], FieldPrediction(name="name", value="Sakshi Raut")
        )
        assert r.status is MatchStatus.EXACT


class TestWellFormed:
    def test_parseable_date_is_well_formed(self):
        r = score_field(DOB, ["1990-02-01"], FieldPrediction(name="dob", value="1991-13-45"))
        assert r.status is MatchStatus.WRONG
        assert not r.well_formed, "an impossible date fails validation loudly"

    def test_plausible_wrong_date_is_well_formed(self):
        r = score_field(DOB, ["1990-02-01"], FieldPrediction(name="dob", value="1991-02-01"))
        assert r.well_formed and r.is_silent_failure


class TestSilentFailureRate:
    def test_confident_wrong_and_well_formed_counts(self):
        o = [
            score_field(
                DOB,
                ["1990-02-01"],
                FieldPrediction(name="dob", value="1991-02-01", confidence=0.95),
            )
        ]
        assert silent_failure_rate(o, 0.7) == 1.0

    def test_low_confidence_does_not_count(self):
        o = [
            score_field(
                DOB, ["1990-02-01"], FieldPrediction(name="dob", value="1991-02-01", confidence=0.2)
            )
        ]
        assert silent_failure_rate(o, 0.7) == 0.0

    def test_malformed_does_not_count(self):
        o = [
            score_field(
                DOB,
                ["1990-02-01"],
                FieldPrediction(name="dob", value="not a date", confidence=0.99),
            )
        ]
        assert silent_failure_rate(o, 0.7) == 0.0

    def test_missing_never_counts(self):
        assert silent_failure_rate([score_field(DOB, ["1990-02-01"], None)], 0.7) == 0.0

    def test_absent_confidence_is_treated_as_confident_by_default(self):
        o = [score_field(DOB, ["1990-02-01"], FieldPrediction(name="dob", value="1991-02-01"))]
        assert silent_failure_rate(o, 0.7) == 1.0
        assert silent_failure_rate(o, 0.7, treat_missing_confidence_as_confident=False) == 0.0


class TestCFER:
    def test_all_correct_is_zero(self):
        assert critical_field_error_rate([outcome(MatchStatus.EXACT)] * 5) == 0.0

    def test_empty_is_zero(self):
        assert critical_field_error_rate([]) == 0.0

    def test_all_wrong_critical_is_one(self):
        assert critical_field_error_rate([outcome(MatchStatus.WRONG)] * 4) == 1.0

    def test_missing_is_penalised_less_than_wrong(self):
        wrong = critical_field_error_rate([outcome(MatchStatus.WRONG)])
        missing = critical_field_error_rate([outcome(MatchStatus.MISSING)])
        assert missing < wrong, "a loud failure must cost less than a silent one"

    def test_critical_outweighs_cosmetic(self):
        crit = critical_field_error_rate(
            [
                outcome(MatchStatus.WRONG, Criticality.CRITICAL),
                outcome(MatchStatus.EXACT, Criticality.COSMETIC),
            ]
        )
        cosm = critical_field_error_rate(
            [
                outcome(MatchStatus.EXACT, Criticality.CRITICAL),
                outcome(MatchStatus.WRONG, Criticality.COSMETIC),
            ]
        )
        assert crit > cosm

    def test_bounded(self):
        for st in MatchStatus:
            assert 0.0 <= critical_field_error_rate([outcome(st)]) <= 1.0


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self):
        o = [outcome(MatchStatus.WRONG)] * 30 + [outcome(MatchStatus.EXACT)] * 70
        point, low, high = bootstrap_ci(o, "cfer", n_resamples=400)
        assert low <= point <= high

    def test_deterministic_under_seed(self):
        o = [outcome(MatchStatus.WRONG)] * 10 + [outcome(MatchStatus.EXACT)] * 10
        assert bootstrap_ci(o, "cfer", n_resamples=200, seed=7) == bootstrap_ci(
            o, "cfer", n_resamples=200, seed=7
        )

    def test_small_n_gives_a_wide_interval(self):
        """n=18 with no interval is not a finding. This is why."""
        small = [outcome(MatchStatus.WRONG)] * 5 + [outcome(MatchStatus.EXACT)] * 13
        _, low, high = bootstrap_ci(small, "cfer", n_resamples=800)
        assert high - low > 0.2

    def test_rejects_unknown_statistic(self):
        with pytest.raises(ValueError):
            bootstrap_ci([outcome(MatchStatus.EXACT)], "f1")


class TestCanonicalize:
    @pytest.mark.parametrize("raw", ["01/02/1990", "1 Feb 1990", "1990-02-01"])
    def test_date_forms_agree(self, raw):
        assert canonicalize(raw, FieldKind.DATE) == "1990-02-01"

    def test_iso_input_is_not_day_flipped(self):
        """Regression: dayfirst=True overrides unambiguous ISO-8601.

        dateutil.parse("1990-02-01", dayfirst=True) returns 1990-01-02 -- a
        confident, well-formed, wrong date, which is the exact failure class
        this project measures. ISO parsing must run before the heuristic.
        """
        assert canonicalize("1990-02-01", FieldKind.DATE) == "1990-02-01"
        assert canonicalize("1990-12-05", FieldKind.DATE) == "1990-12-05"

    def test_ambiguous_input_still_uses_day_first(self):
        assert canonicalize("01/02/1990", FieldKind.DATE) == "1990-02-01"

    @pytest.mark.parametrize("raw", ["banana", "", "  ", "99/99/9999"])
    def test_unparseable_dates_are_none(self, raw):
        assert canonicalize(raw, FieldKind.DATE) is None

    def test_money_separators(self):
        assert canonicalize("1,234.50", FieldKind.MONEY) == canonicalize("1234.5", FieldKind.MONEY)

    def test_alnum_id_ignores_separators(self):
        assert canonicalize("ab-1234 56", FieldKind.ALNUM_ID) == "AB123456"

    def test_length_is_bounded(self):
        assert len(canonicalize("a" * 10_000, FieldKind.TEXT) or "") <= 512

    def test_none_passes_through(self):
        assert canonicalize(None, FieldKind.TEXT) is None

    def test_non_str_is_rejected(self):
        with pytest.raises(TypeError):
            canonicalize(19900201, FieldKind.DATE)  # type: ignore[arg-type]


class TestScoreDocument:
    def test_scores_every_schema_field(self):
        out = score_document(
            [NAME, DOB, ADDRESS],
            {"name": ["Sakshi Raut"], "dob": ["1990-02-01"]},
            [FieldPrediction(name="name", value="Sakshi Raut")],
        )
        assert [o.status for o in out] == [
            MatchStatus.EXACT,
            MatchStatus.MISSING,
            MatchStatus.TRUE_NEGATIVE,
        ]

    def test_off_schema_predictions_are_ignored(self):
        out = score_document(
            [NAME],
            {"name": ["Sakshi Raut"]},
            [
                FieldPrediction(name="name", value="Sakshi Raut"),
                FieldPrediction(name="nationality", value="IN"),
            ],
        )
        assert len(out) == 1

    def test_profile_row_is_serialisable(self):
        out = score_document(
            [NAME, DOB],
            {"name": ["A B"], "dob": ["1990-02-01"]},
            [FieldPrediction(name="name", value="A B")],
        )
        row = error_profile(out, 0.7).as_row()
        assert row["n"] == 2 and "cfer" in row
