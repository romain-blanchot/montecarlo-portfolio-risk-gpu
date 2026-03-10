"""Tests for VaR and Expected Shortfall on distributions with known analytical values.

Analytical benchmarks for N(0, 1):
    VaR(95%)  ≈  1.6449
    ES(95%)   ≈  2.0627  (= phi(1.6449) / 0.05, where phi is the standard normal PDF)

We use n=100_000 paths with a fixed seed so that the Monte Carlo error is small
enough to stay within the asserted tolerances, and the test remains deterministic.
"""

import numpy as np
import pytest

from portfolio_risk_engine.domain.expected_shortfall import compute_es
from portfolio_risk_engine.domain.var import compute_var

RNG = np.random.default_rng(0)
LOSSES = RNG.standard_normal(100_000)


class TestComputeVar:
    def test_95th_percentile_approx_1645(self) -> None:
        var = compute_var(LOSSES, confidence=0.95)
        assert abs(var - 1.6449) < 0.05

    def test_50th_percentile_near_zero(self) -> None:
        """Median of N(0,1) should be close to 0."""
        var = compute_var(LOSSES, confidence=0.50)
        assert abs(var) < 0.02

    def test_at_least_95_pct_below_var(self) -> None:
        var = compute_var(LOSSES, confidence=0.95)
        fraction_below = float(np.mean(LOSSES <= var))
        assert fraction_below >= 0.94

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            compute_var(LOSSES, confidence=1.0)

    def test_invalid_confidence_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            compute_var(LOSSES, confidence=0.0)


class TestComputeEs:
    def test_95th_es_approx_2063(self) -> None:
        es = compute_es(LOSSES, confidence=0.95)
        assert abs(es - 2.0627) < 0.10

    def test_es_greater_than_var(self) -> None:
        var = compute_var(LOSSES, confidence=0.95)
        es = compute_es(LOSSES, confidence=0.95)
        assert es >= var

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            compute_es(LOSSES, confidence=1.5)
