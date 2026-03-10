import numpy as np

from portfolio_risk_engine.domain.models.gbm_model import MultivariateGBM
from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)


class CpuMonteCarloEngine:
    """NumPy-based CPU implementation of the MonteCarloEngine port."""

    def simulate(
        self,
        model: MultivariateGBM,
        initial_prices: tuple[float, ...],
        num_simulations: int,
        time_horizon_days: int,
    ) -> MonteCarloSimulationResult:
        params = model.market_parameters
        n = len(params.tickers)
        T = time_horizon_days / params.annualization_factor

        drift = np.array(params.drift_vector)
        variances = np.array([params.covariance_matrix[i][i] for i in range(n)])
        S0 = np.array(initial_prices)
        L = np.array(model.cholesky_factor)

        # Independent standard normals: shape (n, num_simulations)
        Z = np.random.standard_normal((n, num_simulations))

        # Correlated normals via Cholesky: shape (n, num_simulations)
        correlated_Z = L @ Z

        # GBM terminal price: S_T = S_0 * exp((mu - sigma^2/2)*T + sqrt(T)*L*Z)
        drift_adj = (drift - 0.5 * variances) * T
        log_returns = drift_adj[:, np.newaxis] + np.sqrt(T) * correlated_Z
        terminal_prices_array = S0[:, np.newaxis] * np.exp(log_returns)

        terminal_prices = {}
        for i, ticker in enumerate(params.tickers):
            terminal_prices[ticker] = tuple(float(x) for x in terminal_prices_array[i])

        return MonteCarloSimulationResult(
            tickers=params.tickers,
            initial_prices=initial_prices,
            terminal_prices=terminal_prices,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
        )
