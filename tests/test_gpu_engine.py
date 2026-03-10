"""Tests for MonteCarloGPU.

Strategy
--------
* If Numba is not installed → entire module is skipped via ``pytest.importorskip``.
* If Numba is installed but no CUDA GPU is present → the ``needs_gpu`` mark
  skips all simulation tests; only the "no-GPU error" test runs.
* If both Numba and a CUDA GPU are present → all tests run.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if numba is not installed at all.
numba = pytest.importorskip("numba", reason="numba not installed — skipping GPU tests")
numba_cuda = pytest.importorskip(
    "numba.cuda", reason="numba.cuda not importable — skipping GPU tests"
)

# Detect whether a real GPU is available for the current process.
try:
    _cuda_available: bool = bool(numba_cuda.is_available())
except Exception:
    _cuda_available = False

needs_gpu = pytest.mark.skipif(
    not _cuda_available,
    reason="No CUDA-capable GPU detected — skipping GPU simulation tests",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from portfolio_risk_engine.domain.market_model import MarketModel  # noqa: E402
from portfolio_risk_engine.domain.portfolio import Portfolio  # noqa: E402
from portfolio_risk_engine.infrastructure.simulation.monte_carlo_gpu import (  # noqa: E402
    MonteCarloGPU,
    _MAX_ASSETS,
)


def _make_inputs(n_assets: int = 2) -> tuple[Portfolio, MarketModel, np.ndarray]:
    s0 = np.full(n_assets, 100.0)
    weights = np.ones(n_assets) / n_assets
    portfolio = Portfolio(S0=s0, weights=weights)
    market_model = MarketModel(
        mu=np.full(n_assets, 0.05),
        sigma=np.full(n_assets, 0.20),
        dt=1.0 / 252,
        n_steps=252,
    )
    corr_matrix = np.eye(n_assets)
    return portfolio, market_model, corr_matrix


# ---------------------------------------------------------------------------
# Tests that run even without a GPU
# ---------------------------------------------------------------------------


class TestMonteCarloGPUNoDevice:
    def test_raises_when_no_cuda(self) -> None:
        """MonteCarloGPU must raise RuntimeError when no GPU is found."""
        if _cuda_available:
            pytest.skip("GPU is present — this test targets no-GPU environments")
        with pytest.raises(RuntimeError, match="CUDA"):
            MonteCarloGPU()

    def test_invalid_threads_per_block_raises(self) -> None:
        """threads_per_block must be a multiple of 32."""
        if not _cuda_available:
            pytest.skip("GPU required to instantiate MonteCarloGPU for this check")
        with pytest.raises(ValueError, match="multiple of 32"):
            MonteCarloGPU(threads_per_block=100)

    def test_too_many_assets_raises(self) -> None:
        """Exceeding _MAX_ASSETS must raise ValueError."""
        if not _cuda_available:
            pytest.skip("GPU required to instantiate MonteCarloGPU for this check")
        engine = MonteCarloGPU()
        n = _MAX_ASSETS + 1
        s0 = np.full(n, 100.0)
        weights = np.ones(n) / n
        portfolio = Portfolio(S0=s0, weights=weights)
        market_model = MarketModel(
            mu=np.full(n, 0.05),
            sigma=np.full(n, 0.20),
            dt=1 / 252,
            n_steps=1,
        )
        corr_matrix = np.eye(n)
        with pytest.raises(ValueError, match="MAX_ASSETS"):
            engine.run(portfolio, market_model, corr_matrix, n_paths=10)


# ---------------------------------------------------------------------------
# Tests that require a real CUDA GPU
# ---------------------------------------------------------------------------


@needs_gpu
class TestMonteCarloGPU:
    def test_output_shape(self) -> None:
        portfolio, mm, corr = _make_inputs()
        losses = MonteCarloGPU().run(portfolio, mm, corr, n_paths=500, seed=0)
        assert losses.shape == (500,)

    def test_output_is_1d(self) -> None:
        portfolio, mm, corr = _make_inputs()
        losses = MonteCarloGPU().run(portfolio, mm, corr, n_paths=100, seed=1)
        assert losses.ndim == 1

    def test_all_values_finite(self) -> None:
        portfolio, mm, corr = _make_inputs()
        losses = MonteCarloGPU().run(portfolio, mm, corr, n_paths=1_000, seed=2)
        assert np.all(np.isfinite(losses))

    def test_deterministic_with_same_seed(self) -> None:
        portfolio, mm, corr = _make_inputs()
        engine = MonteCarloGPU()
        a = engine.run(portfolio, mm, corr, n_paths=300, seed=42)
        b = engine.run(portfolio, mm, corr, n_paths=300, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_diverge(self) -> None:
        portfolio, mm, corr = _make_inputs()
        engine = MonteCarloGPU()
        a = engine.run(portfolio, mm, corr, n_paths=500, seed=1)
        b = engine.run(portfolio, mm, corr, n_paths=500, seed=2)
        assert not np.array_equal(a, b)

    def test_three_assets_shape(self) -> None:
        portfolio, mm, corr = _make_inputs(n_assets=3)
        losses = MonteCarloGPU().run(portfolio, mm, corr, n_paths=200, seed=0)
        assert losses.shape == (200,)

    def test_correlated_assets(self) -> None:
        corr = np.array([[1.0, 0.7], [0.7, 1.0]])
        s0 = np.array([100.0, 100.0])
        weights = np.array([0.5, 0.5])
        portfolio = Portfolio(S0=s0, weights=weights)
        mm = MarketModel(
            mu=np.array([0.05, 0.05]),
            sigma=np.array([0.20, 0.20]),
            dt=1.0 / 252,
            n_steps=252,
        )
        losses = MonteCarloGPU().run(portfolio, mm, corr, n_paths=500, seed=7)
        assert losses.shape == (500,)
        assert np.all(np.isfinite(losses))

    def test_distribution_mean_near_zero(self) -> None:
        """With zero drift the expected loss should be near zero."""
        portfolio, mm, corr = _make_inputs()
        # mu=0 means E[S(T)] = S(0), so the expected portfolio loss is ~0
        mm_neutral = MarketModel(
            mu=np.zeros_like(mm.sigma),
            sigma=mm.sigma,
            dt=mm.dt,
            n_steps=mm.n_steps,
        )
        losses = MonteCarloGPU().run(
            portfolio, mm_neutral, corr, n_paths=50_000, seed=0
        )
        assert abs(losses.mean()) < 1.0  # loose bound; distributional check only

    def test_dimension_mismatch_raises(self) -> None:
        portfolio, mm, _ = _make_inputs(n_assets=2)
        wrong_corr = np.eye(3)
        with pytest.raises(ValueError):
            MonteCarloGPU().run(portfolio, mm, wrong_corr, n_paths=10)
