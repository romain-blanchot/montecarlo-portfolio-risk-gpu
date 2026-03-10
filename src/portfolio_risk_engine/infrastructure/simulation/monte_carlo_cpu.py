import numpy as np

from portfolio_risk_engine.domain.correlation import compute_cholesky
from portfolio_risk_engine.domain.market_model import MarketModel
from portfolio_risk_engine.domain.portfolio import Portfolio
from portfolio_risk_engine.infrastructure.simulation.base import SimulationEngine


class MonteCarloCPU(SimulationEngine):
    """NumPy Monte Carlo engine for multi-asset GBM with Cholesky correlations.

    Vectorised over paths: all n_paths prices are advanced together at each
    step, so memory stays O(n_paths * n_assets) rather than storing full histories.

    Each time step:
        Z_indep ~ N(0, 1)  shape (n_paths, n_assets)
        Z_corr = Z_indep @ L.T           # introduce correlations via Cholesky
        S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_corr)

    Final loss per path: loss = V0 - weights @ S_T
    """

    def run(
        self,
        portfolio: Portfolio,
        market_model: MarketModel,
        corr_matrix: np.ndarray,
        n_paths: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """See SimulationEngine.run for the full parameter docs."""
        n_assets = portfolio.S0.shape[0]

        # dimension checks
        if corr_matrix.shape != (n_assets, n_assets):
            raise ValueError(
                f"corr_matrix shape {corr_matrix.shape} is inconsistent with "
                f"n_assets={n_assets}"
            )
        if market_model.mu.shape != (n_assets,):
            raise ValueError(
                f"market_model.mu shape {market_model.mu.shape} must be ({n_assets},)"
            )

        chol = compute_cholesky(corr_matrix)
        dt = market_model.dt
        mu = market_model.mu
        sigma = market_model.sigma
        drift = (mu - 0.5 * sigma**2) * dt  # (n_assets,)
        diffusion_scale = sigma * np.sqrt(dt)  # (n_assets,)

        rng = np.random.default_rng(seed)

        # S: (n_paths, n_assets) — current prices, advanced in-place each step
        S = np.tile(portfolio.S0, (n_paths, 1)).astype(np.float64)

        for _ in range(market_model.n_steps):
            z_indep = rng.standard_normal((n_paths, n_assets))
            z_corr = z_indep @ chol.T
            S = S * np.exp(drift + diffusion_scale * z_corr)

        v0 = portfolio.initial_value  # scalar
        vt = S @ portfolio.weights  # (n_paths,)
        return v0 - vt
