"""Token accounting and cost, measured rather than estimated.

The rule this module enforces
-----------------------------
A reported cost must come from token counts the provider returned, and from a
price that carries a source and a date. Nothing else may be published.

This is not pedantry. It is common to see "token/cost tracking" implemented as
`len(text.split()) * 4 // 3` -- a word count with a fudge factor, presented in
a README as measurement. It is wrong by a factor that varies with language,
formatting and tokeniser, and for a vision model it ignores image tokens
entirely, which are usually the dominant term. A cost comparison built on it
cannot support any conclusion.

So `Usage` records whether it was measured, and `cost_usd` refuses to price
anything that was not. An estimate is still available for capacity planning --
it just cannot be reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date


class EstimatedUsageError(RuntimeError):
    """Raised when an estimated token count is used where a measured one is required."""


@dataclass(frozen=True)
class Usage:
    """Token counts for a single request.

    Construct via `from_api_usage` so `measured` is set correctly. The default
    constructor is available for tests and for capacity estimates, which are
    marked unmeasured and cannot be priced.
    """

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    measured: bool = False

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cached_input_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @classmethod
    def from_api_usage(cls, usage: object) -> Usage:
        """Build from a provider's usage object or dict.

        Accepts either an object with attributes or a mapping, since providers
        differ. Missing counts are an error rather than a zero: silently
        pricing a request at zero is how a cost comparison goes wrong without
        anyone noticing.
        """

        def read(*names: str) -> int | None:
            for name in names:
                if isinstance(usage, dict):
                    if name in usage:
                        return int(usage[name])
                elif hasattr(usage, name):
                    return int(getattr(usage, name))
            return None

        input_tokens = read("input_tokens", "prompt_tokens")
        output_tokens = read("output_tokens", "completion_tokens")
        if input_tokens is None or output_tokens is None:
            raise ValueError(
                "provider usage is missing input/output token counts; refusing to fabricate them"
            )
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=read("cache_read_input_tokens", "cached_tokens") or 0,
            measured=True,
        )

    @classmethod
    def estimate(cls, text: str, chars_per_token: float = 3.6) -> Usage:
        """Rough capacity estimate. Marked unmeasured; cannot be priced.

        Retained deliberately so nobody reaches for a word-count heuristic and
        quietly reports the result.
        """
        approx = max(1, int(len(text) / chars_per_token))
        return cls(input_tokens=approx, output_tokens=0, measured=False)

    @property
    def billable_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens)


@dataclass(frozen=True)
class Pricing:
    """Published price for one model, with provenance.

    `source` and `retrieved` are required. A benchmark that reports dollars
    without saying which price list and which day is not reproducible, because
    the prices move.
    """

    model: str
    input_per_mtok: float
    output_per_mtok: float
    source: str
    retrieved: date
    cached_input_per_mtok: float | None = None
    notes: str = field(default="")

    def __post_init__(self) -> None:
        if not self.source.startswith("https://"):
            raise ValueError("pricing source must be an https URL to the published price list")
        for name in ("input_per_mtok", "output_per_mtok"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


_PER_MILLION = 1_000_000


def cost_usd(usage: Usage, pricing: Pricing) -> float:
    """Cost of one request in USD. Refuses unmeasured usage."""
    if not usage.measured:
        raise EstimatedUsageError(
            "refusing to price estimated token counts -- build Usage via "
            "Usage.from_api_usage() with the provider's returned usage object"
        )
    cached_rate = (
        pricing.cached_input_per_mtok
        if pricing.cached_input_per_mtok is not None
        else pricing.input_per_mtok
    )
    return (
        usage.billable_input_tokens * pricing.input_per_mtok
        + usage.cached_input_tokens * cached_rate
        + usage.output_tokens * pricing.output_per_mtok
    ) / _PER_MILLION


def cost_per_1k_docs(usages: Sequence[Usage], pricing: Pricing) -> float:
    """Mean cost per 1,000 documents, the unit the frontier chart is drawn in.

    Averaging per-document cost and scaling is deliberate: it stays correct
    when the sample is not exactly 1,000 documents, which it never is.
    """
    if not usages:
        raise ValueError("cannot compute a rate from zero documents")
    total = sum(cost_usd(u, pricing) for u in usages)
    return (total / len(usages)) * 1_000
