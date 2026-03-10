"""Timing comparison between the CPU and GPU Monte Carlo engines.

Note: the first run compiles the CUDA kernel via Numba JIT (~2–5 s).
Pass warmup=True (the default) to absorb that cost in a cheap dry run
before the actual timed measurement.
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
    """Run both engines on the same inputs and return timing + risk metrics.

    Both engines use the same seed so risk-metric estimates should be close
    (small differences come from their different RNG implementations).

    Parameters
    ----------
    warmup : if True, fire a cheap GPU pass first to absorb JIT compile time.

    Raises
    ------
    RuntimeError : propagated from MonteCarloGPU if no GPU/Numba is found.
    """
    cpu_engine = MonteCarloCPU()
    gpu_engine = MonteCarloGPU()  # raises RuntimeError if GPU unavailable

    if warmup:
        gpu_engine.run(portfolio, market_model, corr_matrix, n_paths=1_000, seed=seed)

    # timed CPU run
    t0 = time.perf_counter()
    cpu_losses = cpu_engine.run(
        portfolio, market_model, corr_matrix, n_paths, seed=seed
    )
    cpu_time = time.perf_counter() - t0

    # timed GPU run
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
