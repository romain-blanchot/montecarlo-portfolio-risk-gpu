"""Compare CPU and GPU Monte Carlo engines on identical inputs.

This module provides :func:`compare`, which runs both engines with the same
seed, measures wall-clock time, and returns risk metrics side-by-side so the
caller can verify numerical agreement and measure GPU speedup.

Notes
-----
The first call to :func:`compare` triggers Numba JIT compilation of the CUDA
kernel (typically 2–5 s).  Pass ``warmup=True`` (the default) to fire a cheap
warm-up run that absorbs the compilation time before the timed measurement.
"""

from __future__ import annotations

import time
from typing import TypedDict

import numpy as np

from portfolio_risk_engine.domain.expected_shortfall import compute_es
from portfolio_risk_engine.domain.market_model import MarketModel
from portfolio_risk_engine.domain.portfolio import Portfolio
from portfolio_risk_engine.domain.var import compute_var
from portfolio_risk_engine.infrastructure.simulation.monte_carlo_cpu import (
    MonteCarloCPU,
)
from portfolio_risk_engine.infrastructure.simulation.monte_carlo_gpu import (
    MonteCarloGPU,
)


class ComparisonResult(TypedDict):
    """Return type of :func:`compare`."""

    cpu_var: float
    cpu_es: float
    gpu_var: float
    gpu_es: float
    cpu_time_s: float
    gpu_time_s: float
    speedup: float


def compare(
    portfolio: Portfolio,
    market_model: MarketModel,
    corr_matrix: np.ndarray,
    n_paths: int = 100_000,
    confidence: float = 0.95,
    seed: int = 42,
    warmup: bool = True,
) -> ComparisonResult:
    """Run both CPU and GPU engines and return timing + risk metrics.

    Parameters
    ----------
    portfolio:
        Portfolio definition (initial prices and weights).
    market_model:
        GBM model parameters.
    corr_matrix:
        Asset correlation matrix, shape ``(n_assets, n_assets)``.
    n_paths:
        Number of Monte Carlo paths for the timed run.
    confidence:
        VaR / ES confidence level.
    seed:
        RNG seed used by both engines (ensures same statistical population
        for a fair comparison of risk-metric estimates).
    warmup:
        If ``True`` (default), run a cheap GPU pass (1 000 paths) before
        the timed measurement to absorb Numba JIT compilation overhead.

    Returns
    -------
    ComparisonResult
        Dict with ``cpu_var``, ``cpu_es``, ``gpu_var``, ``gpu_es``,
        ``cpu_time_s``, ``gpu_time_s``, ``speedup``.

    Raises
    ------
    RuntimeError
        If Numba is not installed or no CUDA-capable GPU is detected
        (propagated from :class:`MonteCarloGPU`).
    """
    cpu_engine = MonteCarloCPU()
    gpu_engine = MonteCarloGPU()  # raises RuntimeError if GPU unavailable

    # --- optional JIT warm-up -----------------------------------------------
    if warmup:
        gpu_engine.run(portfolio, market_model, corr_matrix, n_paths=1_000, seed=seed)

    # --- CPU timed run -------------------------------------------------------
    t0 = time.perf_counter()
    cpu_losses = cpu_engine.run(
        portfolio, market_model, corr_matrix, n_paths, seed=seed
    )
    cpu_time = time.perf_counter() - t0

    # --- GPU timed run -------------------------------------------------------
    t0 = time.perf_counter()
    gpu_losses = gpu_engine.run(
        portfolio, market_model, corr_matrix, n_paths, seed=seed
    )
    gpu_time = time.perf_counter() - t0

    return ComparisonResult(
        cpu_var=compute_var(cpu_losses, confidence),
        cpu_es=compute_es(cpu_losses, confidence),
        gpu_var=compute_var(gpu_losses, confidence),
        gpu_es=compute_es(gpu_losses, confidence),
        cpu_time_s=cpu_time,
        gpu_time_s=gpu_time,
        speedup=cpu_time / gpu_time,
    )
