try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)
from portfolio_risk_engine.domain.models.student_t_gbm import StudentTGBM


class GpuStudentTEngine:
    """CuPy-based GPU implementation for Student-t GBM simulation."""

    def __init__(self, seed: int | None = None) -> None:
        if not _CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not installed. Install with: pip install cupy-cuda12x"
            )
        try:
            cp.cuda.runtime.getDeviceCount()
        except Exception as e:
            raise RuntimeError(f"No CUDA-capable GPU detected: {e}") from e
        self._seed = seed

    def simulate(
        self,
        model: StudentTGBM,
        initial_prices: tuple[float, ...],
        num_simulations: int,
        time_horizon_days: int,
    ) -> MonteCarloSimulationResult:
        params = model.market_parameters
        n = len(params.tickers)
        nu = model.degrees_of_freedom
        T = time_horizon_days / params.annualization_factor

        drift = cp.array(params.drift_vector, dtype=cp.float64)
        variances = cp.array(
            [params.covariance_matrix[i][i] for i in range(n)], dtype=cp.float64
        )
        S0 = cp.array(initial_prices, dtype=cp.float64)
        L = cp.array(model.cholesky_factor, dtype=cp.float64)

        rng = cp.random.RandomState(self._seed)

        # Multivariate Student-t: N(0,1) / sqrt(chi2(nu)/nu)
        Z_normal = rng.standard_normal((n, num_simulations), dtype=cp.float64)
        chi2 = rng.chisquare(df=nu, size=num_simulations).astype(cp.float64)
        scaling = cp.sqrt(nu / chi2)
        Z_t = Z_normal * scaling[cp.newaxis, :]
        Z_t *= cp.sqrt((nu - 2) / nu)

        correlated_Z = L @ Z_t

        drift_adj = (drift - 0.5 * variances) * T
        log_returns = drift_adj[:, cp.newaxis] + cp.sqrt(T) * correlated_Z
        terminal_prices_array = S0[:, cp.newaxis] * cp.exp(log_returns)

        terminal_prices_cpu = cp.asnumpy(terminal_prices_array)

        terminal_prices = {}
        for i, ticker in enumerate(params.tickers):
            terminal_prices[ticker] = tuple(terminal_prices_cpu[i].tolist())

        return MonteCarloSimulationResult(
            tickers=params.tickers,
            initial_prices=initial_prices,
            terminal_prices=terminal_prices,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
        )
