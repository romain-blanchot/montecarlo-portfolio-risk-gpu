from typing import TypedDict

import numpy as np

from portfolio_risk_engine.domain.expected_shortfall import compute_es
from portfolio_risk_engine.domain.market_model import MarketModel
from portfolio_risk_engine.domain.portfolio import Portfolio
from portfolio_risk_engine.domain.var import compute_var
from portfolio_risk_engine.infrastructure.simulation.monte_carlo_cpu import (
    MonteCarloCPU,
)


class SimulationResult(TypedDict):
    """Return type for :func:`run`."""

    losses: np.ndarray
    var: float
    es: float


def run(
    portfolio: Portfolio,
    market_model: MarketModel,
    corr_matrix: np.ndarray,
    n_paths: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> SimulationResult:
    """Orchestrate a CPU Monte Carlo simulation and return risk metrics.

    Parameters
    ----------
    portfolio:
        Portfolio definition (initial prices and weights).
    market_model:
        GBM model parameters (drift, vol, time-step, number of steps).
    corr_matrix:
        Asset correlation matrix, shape ``(n_assets, n_assets)``.
    n_paths:
        Number of simulated paths.  Default is ``10_000``.
    confidence:
        VaR / ES confidence level.  Default is ``0.95``.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    SimulationResult
        Dict with keys:

        * ``"losses"`` — ``np.ndarray`` of shape ``(n_paths,)``
        * ``"var"``    — Value at Risk (scalar ``float``)
        * ``"es"``     — Expected Shortfall (scalar ``float``)
    """
    engine = MonteCarloCPU()
    losses = engine.run(portfolio, market_model, corr_matrix, n_paths, seed=seed)
    return SimulationResult(
        losses=losses,
        var=compute_var(losses, confidence),
        es=compute_es(losses, confidence),
    )
