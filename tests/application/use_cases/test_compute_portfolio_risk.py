import numpy as np
import pytest

from portfolio_risk_engine.application.use_cases.compute_portfolio_risk import (
    ComputePortfolioRisk,
)
from portfolio_risk_engine.domain.models.asset import Asset
from portfolio_risk_engine.domain.models.portfolio import Portfolio
from portfolio_risk_engine.domain.models.position import Position
from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)
from portfolio_risk_engine.domain.value_objects.currency import Currency
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.domain.value_objects.weight import Weight

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")
USD = Currency("USD")


def _make_portfolio(*ticker_weights: tuple[Ticker, float]) -> Portfolio:
    return Portfolio(
        positions=tuple(
            Position(
                asset=Asset(ticker=t, currency=USD),
                weight=Weight(w),
            )
            for t, w in ticker_weights
        )
    )


def _make_sim_result(
    tickers: tuple[Ticker, ...],
    initial_prices: tuple[float, ...],
    terminal_prices: dict[Ticker, tuple[float, ...]],
) -> MonteCarloSimulationResult:
    n_sims = len(next(iter(terminal_prices.values())))
    return MonteCarloSimulationResult(
        tickers=tickers,
        initial_prices=initial_prices,
        terminal_prices=terminal_prices,
        num_simulations=n_sims,
        time_horizon_days=21,
    )


class TestComputePortfolioRiskSingleTicker:
    def test_basic_metrics(self):
        # 100% AAPL, S0=100, terminal prices known
        portfolio = _make_portfolio((AAPL, 1.0))
        sim = _make_sim_result(
            tickers=(AAPL,),
            initial_prices=(100.0,),
            terminal_prices={AAPL: (110.0, 90.0, 105.0, 95.0, 100.0)},
        )
        metrics = ComputePortfolioRisk.execute(portfolio, sim)

        # Returns: 0.10, -0.10, 0.05, -0.05, 0.00
        expected_mean = (0.10 - 0.10 + 0.05 - 0.05 + 0.0) / 5
        assert metrics.mean_return == pytest.approx(expected_mean)

    def test_var_loss_positive_convention(self):
        portfolio = _make_portfolio((AAPL, 1.0))
        # Large losses dominate → VaR should be positive
        sim = _make_sim_result(
            tickers=(AAPL,),
            initial_prices=(100.0,),
            terminal_prices={AAPL: tuple(70.0 for _ in range(100))},
        )
        metrics = ComputePortfolioRisk.execute(portfolio, sim)
        # All returns = -0.30, losses = +0.30
        assert metrics.var_95 == pytest.approx(0.30)
        assert metrics.var_99 == pytest.approx(0.30)
        assert metrics.es_95 == pytest.approx(0.30)
        assert metrics.es_99 == pytest.approx(0.30)

    def test_volatility_positive(self):
        portfolio = _make_portfolio((AAPL, 1.0))
        sim = _make_sim_result(
            tickers=(AAPL,),
            initial_prices=(100.0,),
            terminal_prices={AAPL: (110.0, 90.0, 105.0, 95.0, 100.0)},
        )
        metrics = ComputePortfolioRisk.execute(portfolio, sim)
        assert metrics.volatility > 0


class TestComputePortfolioRiskMultiTicker:
    def test_weighted_returns(self):
        # 60% AAPL, 40% MSFT, replicate to avoid ddof=1 warning
        portfolio = _make_portfolio((AAPL, 0.6), (MSFT, 0.4))
        sim = _make_sim_result(
            tickers=(AAPL, MSFT),
            initial_prices=(100.0, 200.0),
            terminal_prices={
                AAPL: (110.0, 110.0),  # +10%
                MSFT: (220.0, 220.0),  # +10%
            },
        )
        metrics = ComputePortfolioRisk.execute(portfolio, sim)
        # Portfolio return: 0.6*0.10 + 0.4*0.10 = 0.10
        assert metrics.mean_return == pytest.approx(0.10)

    def test_diversification(self):
        # One asset up, one down
        portfolio = _make_portfolio((AAPL, 0.5), (MSFT, 0.5))
        sim = _make_sim_result(
            tickers=(AAPL, MSFT),
            initial_prices=(100.0, 100.0),
            terminal_prices={
                AAPL: (120.0, 120.0),  # +20%
                MSFT: (80.0, 80.0),  # -20%
            },
        )
        metrics = ComputePortfolioRisk.execute(portfolio, sim)
        # Portfolio return: 0.5*0.20 + 0.5*(-0.20) = 0.0
        assert metrics.mean_return == pytest.approx(0.0)


class TestComputePortfolioRiskValidation:
    def test_ticker_not_in_portfolio_raises(self):
        portfolio = _make_portfolio((AAPL, 1.0))
        sim = _make_sim_result(
            tickers=(MSFT,),
            initial_prices=(100.0,),
            terminal_prices={MSFT: (110.0,)},
        )
        with pytest.raises(ValueError, match="not found in portfolio"):
            ComputePortfolioRisk.execute(portfolio, sim)


class TestComputePortfolioRiskStatistical:
    def test_var_ordering(self):
        rng = np.random.default_rng(42)
        portfolio = _make_portfolio((AAPL, 1.0))
        # Generate returns with known distribution
        terminal = tuple(float(x) for x in 100.0 * np.exp(rng.normal(0.0, 0.2, 10_000)))
        sim = _make_sim_result(
            tickers=(AAPL,),
            initial_prices=(100.0,),
            terminal_prices={AAPL: terminal},
        )
        metrics = ComputePortfolioRisk.execute(portfolio, sim)
        # VaR99 >= VaR95, ES99 >= ES95
        assert metrics.var_99 >= metrics.var_95
        assert metrics.es_99 >= metrics.es_95
        # ES >= VaR (by definition)
        assert metrics.es_95 >= metrics.var_95
        assert metrics.es_99 >= metrics.var_99
