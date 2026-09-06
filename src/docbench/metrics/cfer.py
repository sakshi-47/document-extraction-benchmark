"""Critical Field Error Rate (CFER) and Silent Failure Rate (SFR).

Why not just report field accuracy
----------------------------------
Flat accuracy treats every field and every failure mode alike. In a
verification pipeline neither is true.

Fields differ in consequence. A wrong date of birth is a compliance failure; a
mangled street suffix is noise. CFER weights each field by its criticality.

Failure modes differ in consequence far more sharply, and this is the point of
the metric. A MISSING field fails loudly: it is null, validation catches it,
the case routes to a human. It costs money. A WRONG but well-formed field
fails silently: it satisfies every format check and reaches a decision carrying
a value nobody verified. It costs a wrong decision. Weighting the two equally
-- which flat accuracy does -- hides the failure mode that actually matters.

So CFER weights by criticality AND by severity, and SFR isolates the dangerous
quadrant on its own.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from docbench.types import Criticality, FieldOutcome, MatchStatus

#: How much each field class matters to the verification decision.
CRITICALITY_WEIGHTS: dict[Criticality, float] = {
    Criticality.CRITICAL: 1.0,
    Criticality.STANDARD: 0.3,
    Criticality.COSMETIC: 0.05,
}

#: How damaging each failure mode is, given that it occurred.
#: WRONG and SPURIOUS pass validation and reach a decision; MISSING does not.
SEVERITY_WEIGHTS: dict[MatchStatus, float] = {
    MatchStatus.EXACT: 0.0,
    MatchStatus.NORMALIZED: 0.0,
    MatchStatus.TRUE_NEGATIVE: 0.0,
    MatchStatus.MISSING: 0.3,
    MatchStatus.WRONG: 1.0,
    MatchStatus.SPURIOUS: 1.0,
}


def critical_field_error_rate(
    outcomes: Sequence[FieldOutcome],
    criticality_weights: dict[Criticality, float] | None = None,
    severity_weights: dict[MatchStatus, float] | None = None,
) -> float:
    """Criticality- and severity-weighted error rate. 0.0 is perfect.

    Returns 0.0 for an empty input: no fields scored means no errors observed.
    Callers reporting this must carry `n` alongside it.
    """
    if not outcomes:
        return 0.0
    crit = criticality_weights or CRITICALITY_WEIGHTS
    sev = severity_weights or SEVERITY_WEIGHTS

    weighted_errors = sum(crit[o.criticality] * sev[o.status] for o in outcomes)
    total_weight = sum(crit[o.criticality] for o in outcomes)
    return weighted_errors / total_weight if total_weight else 0.0


def is_silent_failure(
    outcome: FieldOutcome,
    confidence_threshold: float,
    treat_missing_confidence_as_confident: bool = True,
) -> bool:
    """True if this outcome is wrong, well-formed, and confidently asserted.

    An extractor reporting no confidence at all is treated as confident by
    default: with no confidence signal there is nothing to route the case to
    review on, so the failure is just as silent in practice. Set the flag to
    False to score only explicitly-confident predictions.
    """
    if not (outcome.status is MatchStatus.WRONG and outcome.well_formed):
        return False
    if outcome.confidence is None:
        return treat_missing_confidence_as_confident
    return outcome.confidence >= confidence_threshold


def silent_failure_rate(
    outcomes: Sequence[FieldOutcome],
    confidence_threshold: float,
    treat_missing_confidence_as_confident: bool = True,
) -> float:
    """Fraction of scored fields that fail silently.

    The denominator is every scored field, not just the erroneous ones, so this
    reads directly as "how often does a field reach a decision carrying an
    unverified wrong value".
    """
    if not outcomes:
        return 0.0
    hits = sum(
        is_silent_failure(o, confidence_threshold, treat_missing_confidence_as_confident)
        for o in outcomes
    )
    return hits / len(outcomes)


@dataclass(frozen=True)
class ErrorProfile:
    """Full breakdown for one experimental condition."""

    n: int
    cfer: float
    silent_failure_rate: float
    status_counts: dict[str, int]

    def as_row(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "n": self.n,
            "cfer": round(self.cfer, 4),
            "silent_failure_rate": round(self.silent_failure_rate, 4),
        }
        row.update({f"n_{k}": v for k, v in sorted(self.status_counts.items())})
        return row


def error_profile(
    outcomes: Sequence[FieldOutcome],
    confidence_threshold: float,
) -> ErrorProfile:
    """Summarise a set of outcomes into the numbers a results table needs."""
    return ErrorProfile(
        n=len(outcomes),
        cfer=critical_field_error_rate(outcomes),
        silent_failure_rate=silent_failure_rate(outcomes, confidence_threshold),
        status_counts=dict(Counter(o.status.value for o in outcomes)),
    )


def bootstrap_ci(
    outcomes: Sequence[FieldOutcome],
    statistic: str = "cfer",
    confidence_threshold: float = 0.7,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap interval: (point, low, high).

    Every headline number in this project ships with one of these. The prior
    work reported a 27.8% vs 20.9% difference at n=18 with no interval, which
    is not a difference anyone can act on. Reporting the interval makes the
    claim falsifiable, and sometimes retracts it -- which is the point.
    """
    if statistic not in {"cfer", "silent_failure_rate"}:
        raise ValueError(f"unknown statistic: {statistic!r}")
    if not outcomes:
        return (0.0, 0.0, 0.0)

    def compute(sample: Sequence[FieldOutcome]) -> float:
        if statistic == "cfer":
            return critical_field_error_rate(sample)
        return silent_failure_rate(sample, confidence_threshold)

    rng = np.random.default_rng(seed)
    n = len(outcomes)
    indices = rng.integers(0, n, size=(n_resamples, n))
    replicates = np.fromiter(
        (compute([outcomes[i] for i in row]) for row in indices),
        dtype=float,
        count=n_resamples,
    )
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(replicates, [alpha, 1.0 - alpha])
    return (compute(outcomes), float(low), float(high))
