import numpy as np

from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)
from portfolio_risk_engine.domain.models.student_t_gbm import StudentTGBM


class CpuStudentTEngine:
    """NumPy-based CPU implementation for Student-t GBM simulation."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

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

        drift = np.array(params.drift_vector)
        variances = np.array([params.covariance_matrix[i][i] for i in range(n)])
        S0 = np.array(initial_prices)
        L = np.array(model.cholesky_factor)

        # Multivariate Student-t via normal/chi2 mixing
        Z_normal = self._rng.standard_normal((n, num_simulations))
        chi2 = self._rng.chisquare(df=nu, size=num_simulations)
        scaling = np.sqrt(nu / chi2)  # (num_simulations,)

        # Scale all components by same chi2 draw (preserves correlation)
        Z_t = Z_normal * scaling[np.newaxis, :]

        # Rescale to unit variance: Var(t/sqrt(nu/(nu-2))) = 1
        Z_t *= np.sqrt((nu - 2) / nu)

        # Apply Cholesky correlation
        correlated_Z = L @ Z_t

        # GBM terminal price
        drift_adj = (drift - 0.5 * variances) * T
        log_returns = drift_adj[:, np.newaxis] + np.sqrt(T) * correlated_Z
        terminal_prices_array = S0[:, np.newaxis] * np.exp(log_returns)

        terminal_prices = {}
        for i, ticker in enumerate(params.tickers):
            terminal_prices[ticker] = tuple(terminal_prices_array[i].tolist())

        return MonteCarloSimulationResult(
            tickers=params.tickers,
            initial_prices=initial_prices,
            terminal_prices=terminal_prices,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
        )
