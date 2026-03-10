import numpy as np
import pytest

from portfolio_risk_engine.domain.models.gbm_model import MultivariateGBM
from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.services.cholesky import cholesky
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.infrastructure.simulation.cpu_monte_carlo_engine import (
    CpuMonteCarloEngine,
)

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")


def _make_model(n: int = 1) -> MultivariateGBM:
    if n == 1:
        params = MarketParameters(
            tickers=(AAPL,),
            drift_vector=(0.10,),
            covariance_matrix=((0.04,),),
            annualization_factor=252,
        )
    else:
        params = MarketParameters(
            tickers=(AAPL, MSFT),
            drift_vector=(0.10, 0.12),
            covariance_matrix=((0.04, 0.01), (0.01, 0.06)),
            annualization_factor=252,
        )
    return MultivariateGBM(
        market_parameters=params,
        cholesky_factor=cholesky(params.covariance_matrix),
    )


class TestCpuMonteCarloEngineSingleTicker:
    def test_returns_correct_number_of_simulations(self):
        engine = CpuMonteCarloEngine()
        model = _make_model(1)
        result = engine.simulate(
            model=model,
            initial_prices=(150.0,),
            num_simulations=1000,
            time_horizon_days=21,
        )
        assert len(result.terminal_prices[AAPL]) == 1000

    def test_terminal_prices_are_positive(self):
        engine = CpuMonteCarloEngine()
        model = _make_model(1)
        result = engine.simulate(
            model=model,
            initial_prices=(150.0,),
            num_simulations=5000,
            time_horizon_days=252,
        )
        assert all(p > 0 for p in result.terminal_prices[AAPL])

    def test_result_metadata(self):
        engine = CpuMonteCarloEngine()
        model = _make_model(1)
        result = engine.simulate(
            model=model,
            initial_prices=(150.0,),
            num_simulations=100,
            time_horizon_days=21,
        )
        assert result.tickers == (AAPL,)
        assert result.initial_prices == (150.0,)
        assert result.num_simulations == 100
        assert result.time_horizon_days == 21

    def test_mean_terminal_price_reasonable(self):
        np.random.seed(42)
        engine = CpuMonteCarloEngine()
        model = _make_model(1)
        # drift=0.10, T=1 year, S0=100
        result = engine.simulate(
            model=model,
            initial_prices=(100.0,),
            num_simulations=50_000,
            time_horizon_days=252,
        )
        mean_price = np.mean(result.terminal_prices[AAPL])
        # E[S_T] = S_0 * exp(mu*T) = 100 * exp(0.10) ~ 110.52
        expected = 100.0 * np.exp(0.10)
        assert mean_price == pytest.approx(expected, rel=0.05)


class TestCpuMonteCarloEngineMultiTicker:
    def test_returns_all_tickers(self):
        engine = CpuMonteCarloEngine()
        model = _make_model(2)
        result = engine.simulate(
            model=model,
            initial_prices=(150.0, 300.0),
            num_simulations=1000,
            time_horizon_days=21,
        )
        assert AAPL in result.terminal_prices
        assert MSFT in result.terminal_prices

    def test_all_prices_positive(self):
        engine = CpuMonteCarloEngine()
        model = _make_model(2)
        result = engine.simulate(
            model=model,
            initial_prices=(150.0, 300.0),
            num_simulations=5000,
            time_horizon_days=252,
        )
        for ticker in (AAPL, MSFT):
            assert all(p > 0 for p in result.terminal_prices[ticker])

    def test_correlation_structure(self):
        np.random.seed(42)
        engine = CpuMonteCarloEngine()
        model = _make_model(2)
        result = engine.simulate(
            model=model,
            initial_prices=(100.0, 100.0),
            num_simulations=50_000,
            time_horizon_days=252,
        )
        # Log returns should be positively correlated (cov=0.01 > 0)
        log_returns_aapl = np.log(np.array(result.terminal_prices[AAPL]) / 100.0)
        log_returns_msft = np.log(np.array(result.terminal_prices[MSFT]) / 100.0)
        corr = np.corrcoef(log_returns_aapl, log_returns_msft)[0, 1]
        assert corr > 0


class TestCpuMonteCarloEngineReproducibility:
    def test_seed_produces_same_result(self):
        engine = CpuMonteCarloEngine()
        model = _make_model(1)

        np.random.seed(123)
        r1 = engine.simulate(
            model=model,
            initial_prices=(100.0,),
            num_simulations=10,
            time_horizon_days=21,
        )

        np.random.seed(123)
        r2 = engine.simulate(
            model=model,
            initial_prices=(100.0,),
            num_simulations=10,
            time_horizon_days=21,
        )

        assert r1.terminal_prices[AAPL] == r2.terminal_prices[AAPL]
