import numpy as np

from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.models.student_t_gbm import StudentTGBM
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.infrastructure.simulation.cpu_student_t_engine import (
    CpuStudentTEngine,
)


def _make_model() -> StudentTGBM:
    params = MarketParameters(
        tickers=(Ticker("AAPL"), Ticker("MSFT")),
        drift_vector=(0.10, 0.12),
        covariance_matrix=((0.04, 0.01), (0.01, 0.05)),
        annualization_factor=252,
    )
    return StudentTGBM(
        market_parameters=params,
        cholesky_factor=((1.0, 0.0), (0.25, 0.9682)),
        degrees_of_freedom=5.0,
    )


class TestCpuStudentTEngine:
    def test_simulate_returns_correct_shape(self) -> None:
        engine = CpuStudentTEngine(seed=42)
        result = engine.simulate(
            model=_make_model(),
            initial_prices=(150.0, 300.0),
            num_simulations=1000,
            time_horizon_days=21,
        )
        assert result.num_simulations == 1000
        assert result.time_horizon_days == 21
        assert len(result.terminal_prices[Ticker("AAPL")]) == 1000
        assert len(result.terminal_prices[Ticker("MSFT")]) == 1000

    def test_terminal_prices_positive(self) -> None:
        engine = CpuStudentTEngine(seed=42)
        result = engine.simulate(
            model=_make_model(),
            initial_prices=(150.0, 300.0),
            num_simulations=5000,
            time_horizon_days=21,
        )
        assert all(p > 0 for p in result.terminal_prices[Ticker("AAPL")])
        assert all(p > 0 for p in result.terminal_prices[Ticker("MSFT")])

    def test_heavier_tails_than_gbm(self) -> None:
        """Student-t with low df should produce heavier tails than Gaussian."""
        engine = CpuStudentTEngine(seed=42)
        result = engine.simulate(
            model=_make_model(),
            initial_prices=(100.0, 100.0),
            num_simulations=50_000,
            time_horizon_days=21,
        )
        returns = np.array(result.terminal_prices[Ticker("AAPL")]) / 100.0 - 1.0
        kurtosis = (
            float(np.mean(((returns - returns.mean()) / returns.std()) ** 4)) - 3.0
        )
        # Student-t with df=5 should produce excess kurtosis > 0
        assert kurtosis > 0.1

    def test_seed_reproducibility(self) -> None:
        model = _make_model()
        r1 = CpuStudentTEngine(seed=123).simulate(
            model=model,
            initial_prices=(100.0, 200.0),
            num_simulations=100,
            time_horizon_days=5,
        )
        r2 = CpuStudentTEngine(seed=123).simulate(
            model=model,
            initial_prices=(100.0, 200.0),
            num_simulations=100,
            time_horizon_days=5,
        )
        assert r1.terminal_prices[Ticker("AAPL")] == r2.terminal_prices[Ticker("AAPL")]
