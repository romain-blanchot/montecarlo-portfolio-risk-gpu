"""End-to-end GPU simulation + risk computation.

Keeps all data on GPU throughout the pipeline. Only 6 scalar floats
are transferred back to CPU. No intermediate tuple conversion.

Architecture: this is an infrastructure-level optimization.
Domain models and application use cases are not modified.
"""

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.models.portfolio_risk_metrics import (
    PortfolioRiskMetrics,
)


class GpuAcceleratedPipeline:
    """Simulation + risk in a single GPU pass — zero tuple allocation."""

    def __init__(self, seed: int | None = None) -> None:
        if not _CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not installed. Install with: pip install cupy-cuda12x"
            )
        try:
            cp.cuda.runtime.getDeviceCount()
        except (RuntimeError, cp.cuda.runtime.CUDARuntimeError) as e:
            raise RuntimeError(f"No CUDA-capable GPU detected: {e}") from e
        self._seed = seed

    def run(
        self,
        market_params: MarketParameters,
        initial_prices: tuple[float, ...],
        weights: tuple[float, ...],
        num_simulations: int,
        time_horizon_days: int,
    ) -> PortfolioRiskMetrics:
        n = len(market_params.tickers)
        T = time_horizon_days / market_params.annualization_factor

        # Transfer inputs to GPU (small, N-sized vectors + NxN matrix)
        drift = cp.array(market_params.drift_vector, dtype=cp.float64)
        cov = cp.array(market_params.covariance_matrix, dtype=cp.float64)
        variances = cp.array(
            [market_params.covariance_matrix[i][i] for i in range(n)], dtype=cp.float64
        )
        S0 = cp.array(initial_prices, dtype=cp.float64)
        w = cp.array(weights, dtype=cp.float64)

        # Cholesky on GPU
        L = cp.linalg.cholesky(cov)

        # --- Simulation (stays on GPU) ---
        rng = cp.random.RandomState(self._seed)
        Z = rng.standard_normal((n, num_simulations), dtype=cp.float64)
        correlated_Z = L @ Z

        drift_adj = (drift - 0.5 * variances) * T
        log_returns = drift_adj[:, cp.newaxis] + cp.sqrt(T) * correlated_Z
        terminal_prices = S0[:, cp.newaxis] * cp.exp(log_returns)

        # --- Risk computation (stays on GPU) ---
        asset_returns = terminal_prices / S0[:, cp.newaxis] - 1.0  # (n, num_sims)
        portfolio_returns = w @ asset_returns  # (num_sims,)
        losses = -portfolio_returns

        mean_return = float(cp.mean(portfolio_returns))
        volatility = float(cp.std(portfolio_returns, ddof=1))
        var_95 = float(cp.percentile(losses, 95))
        var_99 = float(cp.percentile(losses, 99))
        es_95 = float(cp.mean(losses[losses >= var_95]))
        es_99 = float(cp.mean(losses[losses >= var_99]))

        return PortfolioRiskMetrics(
            mean_return=mean_return,
            volatility=volatility,
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
        )

    def run_with_summary(
        self,
        market_params: MarketParameters,
        initial_prices: tuple[float, ...],
        weights: tuple[float, ...],
        num_simulations: int,
        time_horizon_days: int,
    ) -> tuple[PortfolioRiskMetrics, dict[str, float]]:
        """Like run(), but also returns per-ticker mean terminal prices."""
        n = len(market_params.tickers)
        T = time_horizon_days / market_params.annualization_factor

        drift = cp.array(market_params.drift_vector, dtype=cp.float64)
        cov = cp.array(market_params.covariance_matrix, dtype=cp.float64)
        variances = cp.array(
            [market_params.covariance_matrix[i][i] for i in range(n)], dtype=cp.float64
        )
        S0 = cp.array(initial_prices, dtype=cp.float64)
        w = cp.array(weights, dtype=cp.float64)

        L = cp.linalg.cholesky(cov)

        rng = cp.random.RandomState(self._seed)
        Z = rng.standard_normal((n, num_simulations), dtype=cp.float64)
        correlated_Z = L @ Z

        drift_adj = (drift - 0.5 * variances) * T
        log_returns = drift_adj[:, cp.newaxis] + cp.sqrt(T) * correlated_Z
        terminal_prices = S0[:, cp.newaxis] * cp.exp(log_returns)

        # Per-ticker summary (N floats, cheap)
        mean_terminal = cp.mean(terminal_prices, axis=1)
        summary = {}
        for i, ticker in enumerate(market_params.tickers):
            summary[ticker.value] = float(mean_terminal[i])

        # Risk
        asset_returns = terminal_prices / S0[:, cp.newaxis] - 1.0
        portfolio_returns = w @ asset_returns
        losses = -portfolio_returns

        mean_return = float(cp.mean(portfolio_returns))
        volatility = float(cp.std(portfolio_returns, ddof=1))
        var_95 = float(cp.percentile(losses, 95))
        var_99 = float(cp.percentile(losses, 99))
        es_95 = float(cp.mean(losses[losses >= var_95]))
        es_99 = float(cp.mean(losses[losses >= var_99]))

        metrics = PortfolioRiskMetrics(
            mean_return=mean_return,
            volatility=volatility,
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
        )
        return metrics, summary
