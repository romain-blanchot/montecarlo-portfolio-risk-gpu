import numpy as np

from portfolio_risk_engine.domain.correlation import compute_cholesky
from portfolio_risk_engine.domain.market_model import MarketModel
from portfolio_risk_engine.domain.portfolio import Portfolio
from portfolio_risk_engine.infrastructure.simulation.base import SimulationEngine


class MonteCarloCPU(SimulationEngine):
    """Pure-NumPy Monte Carlo engine using multi-asset GBM with Cholesky correlations.

    The simulation loop is vectorised over paths — all ``n_paths`` are
    advanced one time-step at a time, keeping memory proportional to
    ``n_paths * n_assets`` rather than the full path history.

    Algorithm (per time-step)
    -------------------------
    1. Draw independent innovations ``Z_indep ~ N(0, 1)``,
       shape ``(n_paths, n_assets)``.
    2. Correlate: ``Z_corr = Z_indep @ L.T`` where ``L`` is the lower
       Cholesky factor of the correlation matrix.
    3. Advance each asset price::

           S_t = S_{t-1} * exp((mu - 0.5*sigma²)*dt + sigma*sqrt(dt)*Z_corr)

    Terminal loss per path::

        loss = V0 - weights @ S_T    where V0 = weights @ S0
    """

    def run(
        self,
        portfolio: Portfolio,
        market_model: MarketModel,
        corr_matrix: np.ndarray,
        n_paths: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """See :class:`SimulationEngine` for the full parameter documentation."""
        n_assets = portfolio.S0.shape[0]

        # --- validate dimensions ------------------------------------------------
        if corr_matrix.shape != (n_assets, n_assets):
            raise ValueError(
                f"corr_matrix shape {corr_matrix.shape} is inconsistent with "
                f"n_assets={n_assets}"
            )
        if market_model.mu.shape != (n_assets,):
            raise ValueError(
                f"market_model.mu shape {market_model.mu.shape} must be ({n_assets},)"
            )

        # --- pre-compute constants ----------------------------------------------
        chol = compute_cholesky(corr_matrix)  # (n_assets, n_assets)
        dt = market_model.dt
        mu = market_model.mu  # (n_assets,)
        sigma = market_model.sigma  # (n_assets,)
        drift = (mu - 0.5 * sigma**2) * dt  # (n_assets,)
        diffusion_scale = sigma * np.sqrt(dt)  # (n_assets,)

        rng = np.random.default_rng(seed)

        # --- simulate -----------------------------------------------------------
        # S: (n_paths, n_assets) — current prices for all paths
        S = np.tile(portfolio.S0, (n_paths, 1)).astype(np.float64)

        for _ in range(market_model.n_steps):
            z_indep = rng.standard_normal((n_paths, n_assets))  # (n_paths, n_assets)
            z_corr = z_indep @ chol.T  # (n_paths, n_assets)
            S = S * np.exp(drift + diffusion_scale * z_corr)

        # --- compute losses -----------------------------------------------------
        v0 = portfolio.initial_value  # scalar
        vt = S @ portfolio.weights  # (n_paths,)
        return v0 - vt
