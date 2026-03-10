import numpy as np
import pytest

from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.models.portfolio_risk_metrics import (
    PortfolioRiskMetrics,
)
from portfolio_risk_engine.domain.value_objects.ticker import Ticker

try:
    from portfolio_risk_engine.infrastructure.simulation.gpu_accelerated_pipeline import (
        GpuAcceleratedPipeline,
    )

    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False

pytestmark = pytest.mark.gpu

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")


def _make_params(n: int = 1) -> MarketParameters:
    if n == 1:
        return MarketParameters(
            tickers=(AAPL,),
            drift_vector=(0.10,),
            covariance_matrix=((0.04,),),
            annualization_factor=252,
        )
    return MarketParameters(
        tickers=(AAPL, MSFT),
        drift_vector=(0.10, 0.12),
        covariance_matrix=((0.04, 0.01), (0.01, 0.06)),
        annualization_factor=252,
    )


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="CuPy/CUDA not available")
class TestGpuAcceleratedPipelineRun:
    def test_returns_portfolio_risk_metrics(self):
        pipeline = GpuAcceleratedPipeline(seed=42)
        metrics = pipeline.run(
            market_params=_make_params(1),
            initial_prices=(100.0,),
            weights=(1.0,),
            num_simulations=10_000,
            time_horizon_days=21,
        )
        assert isinstance(metrics, PortfolioRiskMetrics)

    def test_returns_pure_python_floats(self):
        pipeline = GpuAcceleratedPipeline(seed=42)
        metrics = pipeline.run(
            market_params=_make_params(1),
            initial_prices=(100.0,),
            weights=(1.0,),
            num_simulations=1000,
            time_horizon_days=21,
        )
        assert isinstance(metrics.mean_return, float)
        assert isinstance(metrics.var_95, float)

    def test_var_loss_positive(self):
        pipeline = GpuAcceleratedPipeline(seed=42)
        metrics = pipeline.run(
            market_params=_make_params(1),
            initial_prices=(100.0,),
            weights=(1.0,),
            num_simulations=50_000,
            time_horizon_days=21,
        )
        # VaR should be positive for a typical equity position
        assert metrics.var_95 > 0
        assert metrics.var_99 > 0

    def test_var_ordering(self):
        pipeline = GpuAcceleratedPipeline(seed=42)
        metrics = pipeline.run(
            market_params=_make_params(2),
            initial_prices=(100.0, 100.0),
            weights=(0.6, 0.4),
            num_simulations=50_000,
            time_horizon_days=21,
        )
        assert metrics.var_99 >= metrics.var_95
        assert metrics.es_99 >= metrics.es_95
        assert metrics.es_95 >= metrics.var_95

    def test_mean_return_reasonable(self):
        pipeline = GpuAcceleratedPipeline(seed=42)
        # drift=0.10, T=1 year, E[return] ~ mu*T = 0.10
        metrics = pipeline.run(
            market_params=_make_params(1),
            initial_prices=(100.0,),
            weights=(1.0,),
            num_simulations=100_000,
            time_horizon_days=252,
        )
        assert metrics.mean_return == pytest.approx(0.10, abs=0.02)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="CuPy/CUDA not available")
class TestGpuAcceleratedPipelineMultiTicker:
    def test_weighted_portfolio(self):
        pipeline = GpuAcceleratedPipeline(seed=42)
        metrics = pipeline.run(
            market_params=_make_params(2),
            initial_prices=(150.0, 300.0),
            weights=(0.6, 0.4),
            num_simulations=10_000,
            time_horizon_days=21,
        )
        assert metrics.volatility > 0

    def test_equal_weight(self):
        pipeline = GpuAcceleratedPipeline(seed=42)
        metrics = pipeline.run(
            market_params=_make_params(2),
            initial_prices=(100.0, 100.0),
            weights=(0.5, 0.5),
            num_simulations=10_000,
            time_horizon_days=21,
        )
        assert isinstance(metrics, PortfolioRiskMetrics)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="CuPy/CUDA not available")
class TestGpuAcceleratedPipelineReproducibility:
    def test_seed_produces_same_result(self):
        params = _make_params(2)
        r1 = GpuAcceleratedPipeline(seed=123).run(
            market_params=params,
            initial_prices=(100.0, 100.0),
            weights=(0.5, 0.5),
            num_simulations=10_000,
            time_horizon_days=21,
        )
        r2 = GpuAcceleratedPipeline(seed=123).run(
            market_params=params,
            initial_prices=(100.0, 100.0),
            weights=(0.5, 0.5),
            num_simulations=10_000,
            time_horizon_days=21,
        )
        assert r1.var_95 == r2.var_95
        assert r1.es_99 == r2.es_99

    def test_different_seeds_produce_different_results(self):
        params = _make_params(2)
        r1 = GpuAcceleratedPipeline(seed=1).run(
            market_params=params,
            initial_prices=(100.0, 100.0),
            weights=(0.5, 0.5),
            num_simulations=10_000,
            time_horizon_days=21,
        )
        r2 = GpuAcceleratedPipeline(seed=2).run(
            market_params=params,
            initial_prices=(100.0, 100.0),
            weights=(0.5, 0.5),
            num_simulations=10_000,
            time_horizon_days=21,
        )
        assert r1.var_95 != r2.var_95


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="CuPy/CUDA not available")
class TestGpuAcceleratedPipelineWithSummary:
    def test_returns_metrics_and_summary(self):
        pipeline = GpuAcceleratedPipeline(seed=42)
        metrics, summary = pipeline.run_with_summary(
            market_params=_make_params(2),
            initial_prices=(100.0, 200.0),
            weights=(0.6, 0.4),
            num_simulations=10_000,
            time_horizon_days=21,
        )
        assert isinstance(metrics, PortfolioRiskMetrics)
        assert "AAPL" in summary
        assert "MSFT" in summary
        assert all(isinstance(v, float) for v in summary.values())

    def test_summary_prices_reasonable(self):
        pipeline = GpuAcceleratedPipeline(seed=42)
        _, summary = pipeline.run_with_summary(
            market_params=_make_params(1),
            initial_prices=(100.0,),
            weights=(1.0,),
            num_simulations=50_000,
            time_horizon_days=252,
        )
        # E[S_T] = S_0 * exp(mu*T) ~ 110.5
        expected = 100.0 * np.exp(0.10)
        assert summary["AAPL"] == pytest.approx(expected, rel=0.05)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="CuPy/CUDA not available")
class TestGpuAcceleratedPipelineConsistency:
    """Compare accelerated pipeline results with standard CPU pipeline."""

    def test_consistent_with_cpu_pipeline(self):
        from portfolio_risk_engine.application.use_cases.compute_portfolio_risk import (
            ComputePortfolioRisk,
        )
        from portfolio_risk_engine.application.use_cases.run_monte_carlo import (
            RunMonteCarlo,
        )
        from portfolio_risk_engine.domain.models.asset import Asset
        from portfolio_risk_engine.domain.models.portfolio import Portfolio
        from portfolio_risk_engine.domain.models.position import Position
        from portfolio_risk_engine.domain.value_objects.currency import Currency
        from portfolio_risk_engine.domain.value_objects.weight import Weight
        from portfolio_risk_engine.infrastructure.simulation.cpu_monte_carlo_engine import (
            CpuMonteCarloEngine,
        )

        params = _make_params(2)
        initial_prices = (100.0, 100.0)
        weights = (0.6, 0.4)
        n_sims = 100_000

        # CPU standard pipeline
        cpu_engine = CpuMonteCarloEngine(seed=99)
        cpu_result = RunMonteCarlo(engine=cpu_engine).execute(
            market_params=params,
            initial_prices=initial_prices,
            num_simulations=n_sims,
            time_horizon_days=21,
        )
        portfolio = Portfolio(
            positions=tuple(
                Position(
                    asset=Asset(ticker=t, currency=Currency("USD")),
                    weight=Weight(w),
                )
                for t, w in zip(params.tickers, weights)
            )
        )
        cpu_metrics = ComputePortfolioRisk.execute(portfolio, cpu_result)

        # GPU accelerated pipeline (different RNG, so check magnitude not exact)
        gpu_metrics = GpuAcceleratedPipeline(seed=99).run(
            market_params=params,
            initial_prices=initial_prices,
            weights=weights,
            num_simulations=n_sims,
            time_horizon_days=21,
        )

        # Same order of magnitude (different RNGs, so rel=0.2 tolerance)
        assert gpu_metrics.var_95 == pytest.approx(cpu_metrics.var_95, rel=0.2)
        assert gpu_metrics.var_99 == pytest.approx(cpu_metrics.var_99, rel=0.2)
        assert gpu_metrics.mean_return == pytest.approx(
            cpu_metrics.mean_return, abs=0.01
        )
