import numpy as np
import pytest

from portfolio_risk_engine.domain.market_model import MarketModel
from portfolio_risk_engine.domain.portfolio import Portfolio
from portfolio_risk_engine.infrastructure.simulation.monte_carlo_cpu import (
    MonteCarloCPU,
)


def _make_inputs(n_assets: int = 2) -> tuple[Portfolio, MarketModel, np.ndarray]:
    """Return a minimal valid (portfolio, market_model, corr_matrix) triple."""
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


class TestMonteCarloCPU:
    def test_output_shape(self) -> None:
        portfolio, market_model, corr = _make_inputs()
        losses = MonteCarloCPU().run(portfolio, market_model, corr, n_paths=500, seed=0)
        assert losses.shape == (500,)

    def test_output_is_1d(self) -> None:
        portfolio, market_model, corr = _make_inputs()
        losses = MonteCarloCPU().run(portfolio, market_model, corr, n_paths=100, seed=1)
        assert losses.ndim == 1

    def test_all_values_finite(self) -> None:
        portfolio, market_model, corr = _make_inputs()
        losses = MonteCarloCPU().run(
            portfolio, market_model, corr, n_paths=1_000, seed=2
        )
        assert np.all(np.isfinite(losses))

    def test_deterministic_with_same_seed(self) -> None:
        portfolio, market_model, corr = _make_inputs()
        engine = MonteCarloCPU()
        losses_a = engine.run(portfolio, market_model, corr, n_paths=300, seed=42)
        losses_b = engine.run(portfolio, market_model, corr, n_paths=300, seed=42)
        np.testing.assert_array_equal(losses_a, losses_b)

    def test_different_seeds_produce_different_results(self) -> None:
        portfolio, market_model, corr = _make_inputs()
        engine = MonteCarloCPU()
        losses_a = engine.run(portfolio, market_model, corr, n_paths=500, seed=1)
        losses_b = engine.run(portfolio, market_model, corr, n_paths=500, seed=2)
        assert not np.array_equal(losses_a, losses_b)

    def test_three_assets_shape(self) -> None:
        portfolio, market_model, corr = _make_inputs(n_assets=3)
        losses = MonteCarloCPU().run(portfolio, market_model, corr, n_paths=200, seed=0)
        assert losses.shape == (200,)

    def test_correlated_assets(self) -> None:
        """Engine must accept a non-identity valid correlation matrix."""
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        s0 = np.array([100.0, 100.0])
        weights = np.array([0.5, 0.5])
        portfolio = Portfolio(S0=s0, weights=weights)
        market_model = MarketModel(
            mu=np.array([0.05, 0.05]),
            sigma=np.array([0.20, 0.20]),
            dt=1.0 / 252,
            n_steps=252,
        )
        losses = MonteCarloCPU().run(portfolio, market_model, corr, n_paths=500, seed=7)
        assert losses.shape == (500,)
        assert np.all(np.isfinite(losses))

    def test_dimension_mismatch_raises(self) -> None:
        portfolio, market_model, _ = _make_inputs(n_assets=2)
        wrong_corr = np.eye(3)  # 3x3 but portfolio has 2 assets
        with pytest.raises(ValueError):
            MonteCarloCPU().run(portfolio, market_model, wrong_corr, n_paths=10)
