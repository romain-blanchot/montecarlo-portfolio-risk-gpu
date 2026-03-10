from abc import ABC, abstractmethod

import numpy as np

from portfolio_risk_engine.domain.market_model import MarketModel
from portfolio_risk_engine.domain.portfolio import Portfolio


class SimulationEngine(ABC):
    """Base class for Monte Carlo simulation engines.

    Both CPU and GPU engines expose the same run() method so they can be
    swapped without changing any other code.
    """

    @abstractmethod
    def run(
        self,
        portfolio: Portfolio,
        market_model: MarketModel,
        corr_matrix: np.ndarray,
        n_paths: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Simulate portfolio paths and return per-path losses.

        Parameters
        ----------
        portfolio : initial asset prices and weights
        market_model : GBM parameters (drift, vol, time-step, n_steps)
        corr_matrix : asset correlation matrix, shape (n_assets, n_assets)
        n_paths : number of Monte Carlo paths
        seed : optional integer seed for reproducibility

        Returns
        -------
        np.ndarray of shape (n_paths,).
        Positive values mean the portfolio lost money over the horizon.
        """
