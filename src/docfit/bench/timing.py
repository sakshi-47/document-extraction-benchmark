"""Latency measurement and percentile reporting.

Two rules, both there to stop a number being reported that cannot support the
claim made from it:

1. A percentile needs enough samples to mean anything. p95 from 10 requests is
   the second-slowest request with extra decimal places. `summarise` refuses to
   report a percentile it does not have the sample size for.
2. Cold and warm requests are different populations. A first call that loads
   weights and a hundredth call that does not should never be pooled into one
   median. They are reported separately.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

#: Minimum samples before a percentile is reportable. p95 needs at least 20
#: observations for the estimate to sit inside the data rather than on its edge.
MIN_SAMPLES = {50: 5, 95: 20, 99: 100}


class InsufficientSamplesError(RuntimeError):
    """Raised when a percentile is requested without the sample size to support it."""


@dataclass(frozen=True)
class Sample:
    """One timed request."""

    wall_ms: float
    cold: bool = False
    #: Time to first token, where the transport reports it.
    ttft_ms: float | None = None

    def __post_init__(self) -> None:
        if self.wall_ms < 0:
            raise ValueError("wall_ms must be non-negative")


@dataclass(frozen=True)
class LatencyReport:
    n: int
    n_cold: int
    percentiles_ms: dict[int, float]
    mean_ms: float
    min_ms: float
    max_ms: float

    def as_row(self) -> dict[str, float | int]:
        row: dict[str, float | int] = {
            "n": self.n,
            "n_cold": self.n_cold,
            "mean_ms": round(self.mean_ms, 1),
            "min_ms": round(self.min_ms, 1),
            "max_ms": round(self.max_ms, 1),
        }
        row.update({f"p{p}_ms": round(v, 1) for p, v in sorted(self.percentiles_ms.items())})
        return row


def percentile(values: Sequence[float], p: int) -> float:
    """Linear-interpolation percentile, matching numpy's default method.

    Stated explicitly because percentile conventions differ, and a p95 computed
    by nearest-rank is not comparable to one computed by interpolation.
    """
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0 <= p <= 100:
        raise ValueError(f"percentile out of range: {p}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarise(
    samples: Sequence[Sample],
    percentiles: Sequence[int] = (50, 95, 99),
    include_cold: bool = False,
    strict: bool = True,
) -> LatencyReport:
    """Summarise warm-path latency.

    Cold samples are excluded by default: they measure weight loading and
    connection setup, not steady-state serving, and pooling them inflates the
    tail of whichever system happened to be measured from a cold start.

    With `strict`, a percentile without the sample size to support it raises
    rather than being reported. Set it False only for exploratory runs whose
    numbers will not be published.
    """
    considered = list(samples) if include_cold else [s for s in samples if not s.cold]
    if not considered:
        raise ValueError(
            "no samples to summarise" + ("" if include_cold else " -- every sample was marked cold")
        )

    values = [s.wall_ms for s in considered]
    reported: dict[int, float] = {}
    for p in percentiles:
        needed = MIN_SAMPLES.get(p, 1)
        if len(values) < needed:
            if strict:
                raise InsufficientSamplesError(
                    f"p{p} needs at least {needed} samples, got {len(values)}. "
                    f"Collect more, or drop p{p} from the report."
                )
            continue
        reported[p] = percentile(values, p)

    return LatencyReport(
        n=len(considered),
        n_cold=sum(1 for s in samples if s.cold),
        percentiles_ms=reported,
        mean_ms=statistics.fmean(values),
        min_ms=min(values),
        max_ms=max(values),
    )
