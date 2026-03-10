import numpy as np
import pytest

from portfolio_risk_engine.domain.models.gbm_model import MultivariateGBM
from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.services.cholesky import cholesky
from portfolio_risk_engine.domain.value_objects.ticker import Ticker

try:
    from portfolio_risk_engine.infrastructure.simulation.gpu_monte_carlo_engine import (
        GpuMonteCarloEngine,
    )

    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False

pytestmark = pytest.mark.gpu

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


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="CuPy/CUDA not available")
class TestGpuMonteCarloEngineSingleTicker:
    def test_returns_correct_number_of_simulations(self):
        engine = GpuMonteCarloEngine()
        model = _make_model(1)
        result = engine.simulate(
            model=model,
            initial_prices=(150.0,),
            num_simulations=1000,
            time_horizon_days=21,
        )
        assert len(result.terminal_prices[AAPL]) == 1000

    def test_terminal_prices_are_positive(self):
        engine = GpuMonteCarloEngine()
        model = _make_model(1)
        result = engine.simulate(
            model=model,
            initial_prices=(150.0,),
            num_simulations=5000,
            time_horizon_days=252,
        )
        assert all(p > 0 for p in result.terminal_prices[AAPL])

    def test_result_metadata(self):
        engine = GpuMonteCarloEngine()
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
        engine = GpuMonteCarloEngine(seed=42)
        model = _make_model(1)
        result = engine.simulate(
            model=model,
            initial_prices=(100.0,),
            num_simulations=50_000,
            time_horizon_days=252,
        )
        mean_price = np.mean(result.terminal_prices[AAPL])
        expected = 100.0 * np.exp(0.10)
        assert mean_price == pytest.approx(expected, rel=0.05)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="CuPy/CUDA not available")
class TestGpuMonteCarloEngineMultiTicker:
    def test_returns_all_tickers(self):
        engine = GpuMonteCarloEngine()
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
        engine = GpuMonteCarloEngine()
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
        engine = GpuMonteCarloEngine(seed=42)
        model = _make_model(2)
        result = engine.simulate(
            model=model,
            initial_prices=(100.0, 100.0),
            num_simulations=50_000,
            time_horizon_days=252,
        )
        log_returns_aapl = np.log(np.array(result.terminal_prices[AAPL]) / 100.0)
        log_returns_msft = np.log(np.array(result.terminal_prices[MSFT]) / 100.0)
        corr = np.corrcoef(log_returns_aapl, log_returns_msft)[0, 1]
        assert corr > 0


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="CuPy/CUDA not available")
class TestGpuMonteCarloEngineReproducibility:
    def test_seed_produces_same_result(self):
        model = _make_model(1)

        r1 = GpuMonteCarloEngine(seed=123).simulate(
            model=model,
            initial_prices=(100.0,),
            num_simulations=10,
            time_horizon_days=21,
        )
        r2 = GpuMonteCarloEngine(seed=123).simulate(
            model=model,
            initial_prices=(100.0,),
            num_simulations=10,
            time_horizon_days=21,
        )

        assert r1.terminal_prices[AAPL] == r2.terminal_prices[AAPL]

    def test_different_seeds_produce_different_results(self):
        model = _make_model(1)

        r1 = GpuMonteCarloEngine(seed=1).simulate(
            model=model,
            initial_prices=(100.0,),
            num_simulations=10,
            time_horizon_days=21,
        )
        r2 = GpuMonteCarloEngine(seed=2).simulate(
            model=model,
            initial_prices=(100.0,),
            num_simulations=10,
            time_horizon_days=21,
        )

        assert r1.terminal_prices[AAPL] != r2.terminal_prices[AAPL]


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="CuPy/CUDA not available")
class TestGpuMonteCarloEngineCompatibility:
    """Verify GPU produces statistically compatible results with CPU."""

    def test_same_result_type(self):
        from portfolio_risk_engine.domain.models.simulation_result import (
            MonteCarloSimulationResult,
        )

        engine = GpuMonteCarloEngine(seed=42)
        model = _make_model(1)
        result = engine.simulate(
            model=model,
            initial_prices=(100.0,),
            num_simulations=100,
            time_horizon_days=21,
        )
        assert isinstance(result, MonteCarloSimulationResult)

    def test_returns_pure_python_types(self):
        engine = GpuMonteCarloEngine(seed=42)
        model = _make_model(1)
        result = engine.simulate(
            model=model,
            initial_prices=(100.0,),
            num_simulations=10,
            time_horizon_days=21,
        )
        # terminal_prices must be pure Python, not cupy arrays
        prices = result.terminal_prices[AAPL]
        assert isinstance(prices, tuple)
        assert all(isinstance(p, float) for p in prices)

    def test_compatible_with_run_monte_carlo(self):
        from portfolio_risk_engine.application.use_cases.run_monte_carlo import (
            RunMonteCarlo,
        )

        engine = GpuMonteCarloEngine(seed=42)
        use_case = RunMonteCarlo(engine=engine)
        params = MarketParameters(
            tickers=(AAPL, MSFT),
            drift_vector=(0.10, 0.12),
            covariance_matrix=((0.04, 0.01), (0.01, 0.06)),
            annualization_factor=252,
        )
        result = use_case.execute(
            market_params=params,
            initial_prices=(150.0, 300.0),
            num_simulations=1000,
            time_horizon_days=21,
        )
        assert result.num_simulations == 1000
        assert len(result.terminal_prices[AAPL]) == 1000
        assert len(result.terminal_prices[MSFT]) == 1000
