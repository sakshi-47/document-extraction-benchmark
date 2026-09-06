"""Tests for the measurement layer.

The point of these is narrow: make it impossible to publish a number the
measurement does not support. Estimated tokens cannot be priced, and a
percentile cannot be reported without the samples behind it.
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
    retrieved=date(2026, 8, 31),
)


class TestMeasuredOnly:
    def test_estimated_usage_cannot_be_priced(self):
        with pytest.raises(EstimatedUsageError, match="refusing to price"):
            cost_usd(Usage.estimate("some text"), PRICING)

    def test_measured_usage_prices_fine(self):
        usage = Usage.from_api_usage({"input_tokens": 1000, "output_tokens": 100})
        assert cost_usd(usage, PRICING) == pytest.approx((1000 * 1.0 + 100 * 5.0) / 1e6)

    def test_default_constructor_is_unmeasured(self):
        assert not Usage(input_tokens=10, output_tokens=1).measured

    def test_missing_counts_are_an_error_not_a_zero(self):
        with pytest.raises(ValueError, match="refusing to fabricate"):
            Usage.from_api_usage({"input_tokens": 100})

    def test_openai_style_key_names(self):
        usage = Usage.from_api_usage({"prompt_tokens": 50, "completion_tokens": 10})
        assert usage.input_tokens == 50 and usage.measured

    def test_reads_from_an_object_not_just_a_dict(self):
        class Resp:
            input_tokens = 7
            output_tokens = 3

        assert Usage.from_api_usage(Resp()).input_tokens == 7

    def test_negative_counts_rejected(self):
        with pytest.raises(ValueError):
            Usage(input_tokens=-1, output_tokens=0)


class TestPricingProvenance:
    def test_source_must_be_a_url(self):
        with pytest.raises(ValueError, match="https URL"):
            Pricing(
                model="m",
                input_per_mtok=1,
                output_per_mtok=1,
                source="the website",
                retrieved=date(2026, 8, 31),
            )

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError):
            Pricing(
                model="m",
                input_per_mtok=-1,
                output_per_mtok=1,
                source="https://example.com",
                retrieved=date(2026, 8, 31),
            )


class TestCachedTokens:
    def test_cached_tokens_are_not_double_charged(self):
        cached = Pricing(
            model="m",
            input_per_mtok=1.0,
            output_per_mtok=5.0,
            cached_input_per_mtok=0.1,
            source="https://example.com",
            retrieved=date(2026, 8, 31),
        )
        usage = Usage.from_api_usage(
            {"input_tokens": 1000, "output_tokens": 0, "cache_read_input_tokens": 900}
        )
        # 100 billed at full rate, 900 at the cached rate.
        assert cost_usd(usage, cached) == pytest.approx((100 * 1.0 + 900 * 0.1) / 1e6)


class TestCostPer1kDocs:
    def test_scales_from_any_sample_size(self):
        usages = [Usage.from_api_usage({"input_tokens": 1000, "output_tokens": 0})] * 7
        assert cost_per_1k_docs(usages, PRICING) == pytest.approx(1.0)

    def test_empty_is_an_error(self):
        with pytest.raises(ValueError):
            cost_per_1k_docs([], PRICING)


class TestPercentile:
    @pytest.mark.parametrize("p", [50, 90, 95, 99])
    def test_matches_numpy_default_method(self, p):
        values = [float(x) for x in range(1, 101)]
        assert percentile(values, p) == pytest.approx(float(np.percentile(values, p)))

    def test_single_value(self):
        assert percentile([42.0], 95) == 42.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            percentile([], 50)


class TestSummarise:
    def test_refuses_p95_without_enough_samples(self):
        with pytest.raises(InsufficientSamplesError, match="p95 needs at least 20"):
            summarise([Sample(wall_ms=10.0)] * 10, percentiles=(50, 95))

    def test_non_strict_drops_unsupported_percentiles(self):
        report = summarise([Sample(wall_ms=10.0)] * 10, percentiles=(50, 95), strict=False)
        assert 50 in report.percentiles_ms and 95 not in report.percentiles_ms

    def test_cold_samples_excluded_by_default(self):
        samples = [Sample(wall_ms=5000.0, cold=True)] + [Sample(wall_ms=10.0)] * 30
        report = summarise(samples, percentiles=(50, 95))
        assert report.n == 30 and report.n_cold == 1
        assert report.max_ms == 10.0, "a cold start must not set the warm-path maximum"

    def test_cold_can_be_included_explicitly(self):
        samples = [Sample(wall_ms=5000.0, cold=True)] + [Sample(wall_ms=10.0)] * 30
        assert summarise(samples, percentiles=(50, 95), include_cold=True).max_ms == 5000.0

    def test_all_cold_gives_a_clear_error(self):
        with pytest.raises(ValueError, match="every sample was marked cold"):
            summarise([Sample(wall_ms=1.0, cold=True)] * 5)

    def test_row_is_serialisable(self):
        samples = [Sample(wall_ms=float(i)) for i in range(1, 31)]
        row = summarise(samples, percentiles=(50, 95)).as_row()
        assert row["n"] == 30 and "p95_ms" in row

    def test_p99_needs_a_hundred_samples(self):
        """30 samples cannot support a p99; the guard is the point of the module."""
        samples = [Sample(wall_ms=float(i)) for i in range(1, 31)]
        with pytest.raises(InsufficientSamplesError, match="p99 needs at least 100"):
            summarise(samples, percentiles=(50, 95, 99))
        assert summarise(samples, percentiles=(50, 95, 99), strict=False).percentiles_ms.keys() == {
            50,
            95,
        }

    def test_negative_latency_rejected(self):
        with pytest.raises(ValueError):
            Sample(wall_ms=-1.0)
