"""Measurement layer tests.

The point of these is narrow: make it impossible to publish a number the
measurement does not support.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from docbench.bench import Pricing, Sample, Usage, cost_per_1k_docs, cost_usd, summarise
from docbench.bench.cost import EstimatedUsageError
from docbench.bench.timing import InsufficientSamplesError, percentile

PRICING = Pricing(
    model="test-model",
    input_per_mtok=1.0,
    output_per_mtok=5.0,
    source="https://example.com/pricing",
    retrieved=date(2026, 9, 6),
)


class TestMeasuredOnly:
    def test_estimated_usage_cannot_be_priced(self):
        """A word-count heuristic must never reach a published cost."""
        with pytest.raises(EstimatedUsageError, match="refusing to price"):
            cost_usd(Usage.estimate("some text"), PRICING)

    def test_measured_usage_prices_correctly(self):
        usage = Usage.from_api_usage({"input_tokens": 1000, "output_tokens": 100})
        assert cost_usd(usage, PRICING) == pytest.approx((1000 * 1.0 + 100 * 5.0) / 1e6)

    def test_missing_counts_are_an_error_not_a_zero(self):
        """Silently pricing a request at zero is how a comparison goes wrong."""
        with pytest.raises(ValueError, match="refusing to fabricate"):
            Usage.from_api_usage({"input_tokens": 100})

    def test_pricing_requires_a_cited_source(self):
        """Dollars without a price list and a date are not reproducible."""
        with pytest.raises(ValueError, match="https URL"):
            Pricing(
                model="m",
                input_per_mtok=1,
                output_per_mtok=1,
                source="the website",
                retrieved=date(2026, 9, 6),
            )

    def test_cost_per_1k_docs_scales_from_any_sample_size(self):
        usages = [Usage.from_api_usage({"input_tokens": 1000, "output_tokens": 0})] * 7
        assert cost_per_1k_docs(usages, PRICING) == pytest.approx(1.0)


class TestLatency:
    def test_percentile_matches_numpy(self):
        """Percentile conventions differ; interpolation is the stated one."""
        values = [float(x) for x in range(1, 101)]
        assert percentile(values, 95) == pytest.approx(float(np.percentile(values, 95)))

    def test_p95_refused_below_twenty_samples(self):
        """p95 from ten requests is the second-slowest request with decimals."""
        with pytest.raises(InsufficientSamplesError, match="p95 needs at least 20"):
            summarise([Sample(wall_ms=10.0)] * 10, percentiles=(50, 95))

    def test_cold_starts_excluded_from_warm_path(self):
        samples = [Sample(wall_ms=5000.0, cold=True)] + [Sample(wall_ms=10.0)] * 30
        report = summarise(samples, percentiles=(50, 95))
        assert report.n == 30 and report.n_cold == 1
        assert report.max_ms == 10.0
