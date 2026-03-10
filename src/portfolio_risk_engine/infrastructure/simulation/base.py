from abc import ABC, abstractmethod

import numpy as np

from portfolio_risk_engine.domain.market_model import MarketModel
from portfolio_risk_engine.domain.portfolio import Portfolio


class SimulationEngine(ABC):
    """Abstract interface for Monte Carlo simulation engines.

    All concrete engines — CPU or GPU — must implement :meth:`run`.
    The signature is intentionally identical so callers can swap engines
    without touching application code.
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
        portfolio:
            Portfolio definition (initial prices and weights).
        market_model:
            GBM parameters (drift, volatility, time-step, number of steps).
        corr_matrix:
            Asset correlation matrix of shape ``(n_assets, n_assets)``.
        n_paths:
            Number of Monte Carlo paths to simulate.
        seed:
            Optional integer seed for reproducible results.

        Returns
        -------
        np.ndarray
            1-D array of simulated losses, shape ``(n_paths,)``.
            A positive value means the portfolio lost money.
        """
