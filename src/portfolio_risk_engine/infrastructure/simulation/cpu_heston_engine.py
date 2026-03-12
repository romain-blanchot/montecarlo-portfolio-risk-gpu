import numpy as np

from portfolio_risk_engine.domain.models.heston_model import HestonModel
from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)


class CpuHestonEngine:
    """NumPy-based CPU implementation for Heston stochastic volatility simulation.

    Uses Euler-Maruyama discretization with full truncation scheme
    (variance floored at zero).
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def simulate(
        self,
        model: HestonModel,
        initial_prices: tuple[float, ...],
        num_simulations: int,
        time_horizon_days: int,
    ) -> MonteCarloSimulationResult:
        n = len(model.tickers)
        dt = 1.0 / model.annualization_factor
        sqrt_dt = np.sqrt(dt)

        drift = np.array(model.drift_vector)
        kappa = np.array([p.kappa for p in model.asset_params])
        theta = np.array([p.theta for p in model.asset_params])
        xi = np.array([p.xi for p in model.asset_params])
        rho = np.array([p.rho for p in model.asset_params])
        L_corr = np.array(model.correlation_cholesky)

        # Initialize
        log_S = np.log(
            np.tile(np.array(initial_prices), (num_simulations, 1)).T
        )  # (n, num_sims)
        v = np.tile(
            np.array([p.v0 for p in model.asset_params]), (num_simulations, 1)
        ).T  # (n, num_sims)

        # Pre-compute reshaped parameters for broadcasting
        kappa_r = kappa[:, np.newaxis]
        theta_r = theta[:, np.newaxis]
        xi_r = xi[:, np.newaxis]
        rho_r = rho[:, np.newaxis]
        sqrt_1_rho2 = np.sqrt(1.0 - rho**2)[:, np.newaxis]
        drift_r = drift[:, np.newaxis]

        for _ in range(time_horizon_days):
            # Generate independent normals for variance
            Z_v = self._rng.standard_normal((n, num_simulations))

            # Generate correlated normals for price (inter-asset correlation)
            Z_indep = self._rng.standard_normal((n, num_simulations))
            Z_corr = L_corr @ Z_indep

            # Combine with leverage effect per asset
            Z_s = rho_r * Z_v + sqrt_1_rho2 * Z_corr

            # Truncated variance (floor at 0)
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)

            # Update log-price: log(S) += (mu - v/2)*dt + sqrt(v)*sqrt(dt)*Z_s
            log_S += (drift_r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z_s

            # Update variance: v += kappa*(theta - v)*dt + xi*sqrt(v)*sqrt(dt)*Z_v
            v = v + kappa_r * (theta_r - v) * dt + xi_r * sqrt_v * sqrt_dt * Z_v
            v = np.maximum(v, 0.0)

        terminal_prices_array = np.exp(log_S)

        terminal_prices = {}
        for i, ticker in enumerate(model.tickers):
            terminal_prices[ticker] = tuple(terminal_prices_array[i].tolist())

        return MonteCarloSimulationResult(
            tickers=model.tickers,
            initial_prices=initial_prices,
            terminal_prices=terminal_prices,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
        )
