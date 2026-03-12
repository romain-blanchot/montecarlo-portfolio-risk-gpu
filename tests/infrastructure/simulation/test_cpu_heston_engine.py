import numpy as np

from portfolio_risk_engine.domain.models.heston_model import (
    HestonAssetParams,
    HestonModel,
)
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.infrastructure.simulation.cpu_heston_engine import (
    CpuHestonEngine,
)


def _make_model() -> HestonModel:
    return HestonModel(
        tickers=(Ticker("AAPL"),),
        drift_vector=(0.05,),
        asset_params=(
            HestonAssetParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04),
        ),
        correlation_cholesky=((1.0,),),
        annualization_factor=252,
    )


def _make_two_asset_model() -> HestonModel:
    return HestonModel(
        tickers=(Ticker("AAPL"), Ticker("MSFT")),
        drift_vector=(0.05, 0.08),
        asset_params=(
            HestonAssetParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04),
            HestonAssetParams(kappa=1.5, theta=0.05, xi=0.25, rho=-0.5, v0=0.05),
        ),
        correlation_cholesky=((1.0, 0.0), (0.3, 0.9539)),
        annualization_factor=252,
    )


class TestCpuHestonEngine:
    def test_simulate_returns_correct_shape(self) -> None:
        engine = CpuHestonEngine(seed=42)
        result = engine.simulate(
            model=_make_model(),
            initial_prices=(100.0,),
            num_simulations=1000,
            time_horizon_days=21,
        )
        assert result.num_simulations == 1000
        assert result.time_horizon_days == 21
        assert len(result.terminal_prices[Ticker("AAPL")]) == 1000

    def test_terminal_prices_positive(self) -> None:
        engine = CpuHestonEngine(seed=42)
        result = engine.simulate(
            model=_make_model(),
            initial_prices=(100.0,),
            num_simulations=5000,
            time_horizon_days=21,
        )
        assert all(p > 0 for p in result.terminal_prices[Ticker("AAPL")])

    def test_two_asset_simulation(self) -> None:
        engine = CpuHestonEngine(seed=42)
        result = engine.simulate(
            model=_make_two_asset_model(),
            initial_prices=(100.0, 200.0),
            num_simulations=1000,
            time_horizon_days=21,
        )
        assert len(result.terminal_prices[Ticker("AAPL")]) == 1000
        assert len(result.terminal_prices[Ticker("MSFT")]) == 1000

    def test_mean_reversion_effect(self) -> None:
        """With high kappa, variance should mean-revert, producing less extreme tails."""
        high_kappa = HestonModel(
            tickers=(Ticker("AAPL"),),
            drift_vector=(0.05,),
            asset_params=(
                HestonAssetParams(kappa=10.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04),
            ),
            correlation_cholesky=((1.0,),),
            annualization_factor=252,
        )
        engine = CpuHestonEngine(seed=42)
        result = engine.simulate(
            model=high_kappa,
            initial_prices=(100.0,),
            num_simulations=10_000,
            time_horizon_days=63,
        )
        prices = np.array(result.terminal_prices[Ticker("AAPL")])
        returns = prices / 100.0 - 1.0
        vol = float(np.std(returns))
        # Volatility should be reasonable (not exploded)
        assert vol < 1.0

    def test_seed_reproducibility(self) -> None:
        model = _make_model()
        r1 = CpuHestonEngine(seed=123).simulate(
            model=model,
            initial_prices=(100.0,),
            num_simulations=100,
            time_horizon_days=5,
        )
        r2 = CpuHestonEngine(seed=123).simulate(
            model=model,
            initial_prices=(100.0,),
            num_simulations=100,
            time_horizon_days=5,
        )
        assert r1.terminal_prices[Ticker("AAPL")] == r2.terminal_prices[Ticker("AAPL")]
