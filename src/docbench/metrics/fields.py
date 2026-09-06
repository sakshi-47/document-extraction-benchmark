"""Scoring a predicted field against ground truth.

Design note -- why `gold` is typed as a Sequence and a bare `str` is rejected:

A scorer that accepts several acceptable answers and iterates them has a
failure mode that is silent and total. Pass one answer as a plain string
instead of a one-element sequence, and iteration yields characters: "Yes" is
scored against 'Y', 'e', 's'. Nothing raises, no output looks malformed, and
every number the scorer produces is meaningless.

A str IS a valid Sequence[str], so type checkers do not catch this. The runtime
guard below does. It is three lines and it is the reason this module can be
trusted.
"""

from __future__ import annotations

from collections.abc import Sequence

from docbench.metrics.normalize import canonicalize
from docbench.types import (
    FieldOutcome,
    FieldPrediction,
    FieldSpec,
    MatchStatus,
)


def _as_variants(gold: Sequence[str]) -> tuple[str, ...]:
    """Coerce accepted-answer variants to a tuple, rejecting a bare str.

    A bare str would iterate character-by-character and score silently wrong.
    Callers with a single accepted answer must pass ["answer"], not "answer".
    """
    if isinstance(gold, str):
        raise TypeError(
            "gold must be a sequence of accepted variants, not a bare str -- "
            f"pass [{gold!r}] instead. A str iterates as characters and would "
            "score against 'a', 'b', 'c'."
        )
    return tuple(gold)


def score_field(
    spec: FieldSpec,
    gold: Sequence[str],
    prediction: FieldPrediction | None,
) -> FieldOutcome:
    """Compare one predicted field against its accepted ground-truth variants.

    `gold` may hold several acceptable surface forms of the same value. An
    empty sequence means the field is genuinely absent from the document, which
    makes correct behaviour an abstention rather than an extraction.
    """
    variants = _as_variants(gold)
    predicted_value = prediction.value if prediction is not None else None
    confidence = prediction.confidence if prediction is not None else None

    has_gold = any(v.strip() for v in variants)
    has_prediction = bool(predicted_value and predicted_value.strip())

    canon_pred = canonicalize(predicted_value, spec.kind) if has_prediction else None
    # A value the canonicaliser cannot interpret as this field kind fails
    # format validation loudly, so it can never be a silent failure.
    well_formed = canon_pred is not None if has_prediction else True

    if not has_gold and not has_prediction:
        status = MatchStatus.TRUE_NEGATIVE
    elif not has_gold:
        status = MatchStatus.SPURIOUS
    elif not has_prediction:
        status = MatchStatus.MISSING
    elif any(predicted_value == v for v in variants):
        status = MatchStatus.EXACT
    elif canon_pred is not None and any(canon_pred == canonicalize(v, spec.kind) for v in variants):
        status = MatchStatus.NORMALIZED
    else:
        status = MatchStatus.WRONG

    return FieldOutcome(
        name=spec.name,
        kind=spec.kind,
        criticality=spec.criticality,
        status=status,
        gold=variants,
        predicted=predicted_value,
        confidence=confidence,
        well_formed=well_formed,
    )


def score_document(
    schema: Sequence[FieldSpec],
    gold: dict[str, Sequence[str]],
    predictions: Sequence[FieldPrediction],
) -> list[FieldOutcome]:
    """Score every field in a document schema.

    The schema is the authority on which fields exist. Predictions naming a
    field outside the schema are ignored rather than scored: a hallucinated
    field name is a different failure class from a hallucinated field value,
    and conflating them would inflate the spurious rate.
    """
    by_name = {p.name: p for p in predictions}
    return [score_field(spec, gold.get(spec.name, ()), by_name.get(spec.name)) for spec in schema]
