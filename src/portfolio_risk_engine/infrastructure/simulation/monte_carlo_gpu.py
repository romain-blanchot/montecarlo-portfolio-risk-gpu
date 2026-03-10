"""GPU Monte Carlo engine — one CUDA thread per simulation path.

Requires Numba and a CUDA-capable GPU:

    pip install portfolio-risk-engine[gpu]

Each call to MonteCarloGPU.run() launches a CUDA kernel where every thread
independently simulates one complete price path for all assets over the full
time horizon. Read-only constants (Cholesky matrix, drift, volatility) are
loaded into shared memory once per block so the inner simulation loop reads
from fast on-chip memory instead of slow global DRAM.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from portfolio_risk_engine.domain.correlation import compute_cholesky
from portfolio_risk_engine.domain.market_model import MarketModel
from portfolio_risk_engine.domain.portfolio import Portfolio
from portfolio_risk_engine.infrastructure.simulation.base import SimulationEngine

# _MAX_ASSETS must be a plain integer literal because Numba needs to know the
# size of shared/local memory arrays at JIT-compile time (before any Python
# object exists). If you need more than 32 assets, raise this value and re-run.
_MAX_ASSETS: int = 32

# Number of CUDA threads launched per block.
# 256 = 8 warps of 32 threads — a standard sweet spot: large enough for the GPU
# to overlap memory latency with arithmetic, small enough to let the scheduler
# run multiple blocks on the same Streaming Multiprocessor simultaneously.
_THREADS_PER_BLOCK: int = 256

# We try to import Numba at module load time. If it isn't installed the module
# still loads fine — MonteCarloGPU will raise a clear RuntimeError when you try
# to use it, rather than crashing on import with a confusing traceback.
_NUMBA_AVAILABLE: bool = False
_CUDA_IS_AVAILABLE: bool = False

cuda: Any = None
create_xoroshiro128p_states: Any = None
xoroshiro128p_normal_float64: Any = None

try:
    from numba import cuda
    from numba.cuda.random import (
        create_xoroshiro128p_states,
        xoroshiro128p_normal_float64,
    )

    _NUMBA_AVAILABLE = True
    try:
        _CUDA_IS_AVAILABLE = bool(cuda.is_available())
    except Exception:
        _CUDA_IS_AVAILABLE = False

except ImportError:
    pass


# ------------------------------------------------------------------------------
# CUDA kernel
# ------------------------------------------------------------------------------

if _NUMBA_AVAILABLE:

    @cuda.jit
    def _gbm_kernel(
        rng_states,  # one RNG state per path (xoroshiro128p),      shape (n_paths,)
        s0,  # initial asset prices,                         shape (n_assets,)
        weights,  # portfolio allocation weights,                  shape (n_assets,)
        drift,  # (mu - 0.5 * sigma²) * dt  — pre-computed,    shape (n_assets,)
        diff_scale,  # sigma * sqrt(dt)           — pre-computed,    shape (n_assets,)
        n_steps,  # number of daily time steps (252 for one year)
        chol,  # lower-triangular Cholesky factor,             shape (n_assets, n_assets)
        n_assets,  # actual number of assets in the portfolio
        losses,  # OUTPUT: portfolio loss for each path,         shape (n_paths,)
    ):
        """Simulate one complete Monte Carlo path per CUDA thread.

        Execution flow per thread:
          1. All threads in the block cooperatively load the read-only constants
             (s0, weights, drift, diff_scale, Cholesky) into shared memory.
          2. Each thread runs the full GBM simulation loop for its own path.
          3. Each thread computes the final portfolio loss and writes it to
             global memory.

        Why shared memory for the constants?
          The Cholesky matrix and drift vectors are read at every single time
          step by every thread. Shared memory is on-chip (~4 cycle latency vs
          ~400 cycles for global DRAM), so loading these constants once per
          block rather than fetching them from global memory at every step
          gives a significant reduction in memory traffic.
        """

        # ── Part 1: load constants into shared memory ──────────────────────────
        #
        # "sm_" prefix marks shared memory arrays throughout this kernel.
        # They live on-chip and are visible to all threads in the same block.
        # Sizes must be compile-time literals — _MAX_ASSETS satisfies that.

        sm_s0 = cuda.shared.array(_MAX_ASSETS, dtype=np.float64)
        sm_weights = cuda.shared.array(_MAX_ASSETS, dtype=np.float64)
        sm_drift = cuda.shared.array(_MAX_ASSETS, dtype=np.float64)
        sm_diff_scale = cuda.shared.array(_MAX_ASSETS, dtype=np.float64)
        sm_chol = cuda.shared.array((_MAX_ASSETS, _MAX_ASSETS), dtype=np.float64)

        local_tid = (
            cuda.threadIdx.x
        )  # this thread's index within the block (0 … block_size-1)
        block_size = cuda.blockDim.x  # total number of threads in the block

        # Each thread loads one slice of each 1-D vector, then jumps by block_size.
        # The while-loop handles the case where n_assets > block_size.
        i = local_tid
        while i < n_assets:
            sm_s0[i] = s0[i]
            sm_weights[i] = weights[i]
            sm_drift[i] = drift[i]
            sm_diff_scale[i] = diff_scale[i]
            i += block_size

        # Load the Cholesky matrix by treating it as a flat array of n_assets²
        # values and striding through it with the same pattern.
        flat_idx = local_tid
        flat_size = n_assets * n_assets
        while flat_idx < flat_size:
            row = flat_idx // n_assets
            col = flat_idx % n_assets
            sm_chol[row, col] = chol[row, col]
            flat_idx += block_size

        # Synchronisation barrier: no thread may continue past this point until
        # every thread in the block has finished writing to shared memory.
        # Without this, a fast thread could start reading sm_chol before a slow
        # thread has finished filling it.
        cuda.syncthreads()

        # ── Part 2: find which path this thread is responsible for ─────────────

        path_id = cuda.grid(
            1
        )  # global thread index across the entire grid = path index
        n_paths = losses.shape[0]

        # The grid is rounded up to a full multiple of block_size, so the last
        # block may contain threads that don't correspond to any path. They exit
        # immediately without touching the output array.
        if path_id >= n_paths:
            return

        # ── Part 3: allocate per-thread working arrays (in registers) ──────────
        #
        # cuda.local.array lives in GPU registers — the fastest memory tier on
        # the GPU. Each thread gets its own private copy, completely isolated
        # from every other thread.

        prices = cuda.local.array(_MAX_ASSETS, dtype=np.float64)  # current asset prices
        z_indep = cuda.local.array(
            _MAX_ASSETS, dtype=np.float64
        )  # independent N(0,1) draws
        z_corr = cuda.local.array(
            _MAX_ASSETS, dtype=np.float64
        )  # correlated draws (after Cholesky)

        # Start every asset at its initial price.
        for i in range(n_assets):
            prices[i] = sm_s0[i]

        # ── Part 4: simulate n_steps daily price moves ─────────────────────────

        for _ in range(n_steps):
            # Draw one independent standard normal for each asset.
            for i in range(n_assets):
                z_indep[i] = xoroshiro128p_normal_float64(rng_states, path_id)

            # Introduce correlations via the Cholesky transform: z_corr = L @ z_indep.
            # Because L is lower-triangular, row i only depends on columns 0..i,
            # which saves roughly half the multiplications compared to a full matmul.
            for i in range(n_assets):
                total = 0.0
                for j in range(i + 1):
                    total += sm_chol[i, j] * z_indep[j]
                z_corr[i] = total

            # GBM update for this time step:
            #   price(t+dt) = price(t) * exp( drift[i] + diff_scale[i] * z_corr[i] )
            # drift and diff_scale were pre-computed on the CPU before the kernel launch
            # to avoid redundant arithmetic inside each thread.
            for i in range(n_assets):
                prices[i] *= math.exp(sm_drift[i] + sm_diff_scale[i] * z_corr[i])

        # ── Part 5: compute portfolio loss and write to global memory ──────────
        #
        # Loss = initial portfolio value − final portfolio value.
        # A positive value means the portfolio lost money; negative means a gain.
        #
        # Writing to losses[path_id]: consecutive threads have consecutive path_ids,
        # so their writes land on consecutive memory addresses. The GPU hardware
        # merges these into a single wide memory transaction (coalesced write),
        # which is much more efficient than scattered random-access writes.

        initial_value = 0.0
        final_value = 0.0
        for i in range(n_assets):
            initial_value += sm_weights[i] * sm_s0[i]
            final_value += sm_weights[i] * prices[i]

        losses[path_id] = initial_value - final_value

else:
    _gbm_kernel = None


# ------------------------------------------------------------------------------
# Public engine class
# ------------------------------------------------------------------------------


class MonteCarloGPU(SimulationEngine):
    """GPU-accelerated Monte Carlo engine backed by the _gbm_kernel CUDA kernel.

    One CUDA thread handles one complete simulation path. The class checks
    for GPU availability at construction time and exposes the same run()
    interface as MonteCarloCPU, so the two engines can be swapped freely
    without changing any calling code.
    """

    def __init__(self, threads_per_block: int = _THREADS_PER_BLOCK) -> None:
        if not _NUMBA_AVAILABLE:
            raise RuntimeError(
                "Numba is not installed. "
                "Install the GPU extras with: pip install portfolio-risk-engine[gpu]"
            )
        if not _CUDA_IS_AVAILABLE:
            raise RuntimeError(
                "No CUDA-capable GPU was detected. "
                "MonteCarloGPU requires an NVIDIA GPU with CUDA support and the "
                "CUDA toolkit installed (see environment.yml)."
            )
        # CUDA executes threads in groups of 32 called warps. A threads_per_block
        # value that isn't a multiple of 32 leaves the last warp of each block
        # partially idle, wasting GPU resources.
        if threads_per_block % 32 != 0:
            raise ValueError(
                f"threads_per_block must be a multiple of 32 (warp size), "
                f"got {threads_per_block}"
            )
        self.threads_per_block = threads_per_block

    def run(
        self,
        portfolio: Portfolio,
        market_model: MarketModel,
        corr_matrix: np.ndarray,
        n_paths: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Simulate n_paths price trajectories on the GPU and return the loss per path.

        The very first call triggers JIT compilation of the CUDA kernel (~3–5 s).
        All subsequent calls reuse the compiled version and run at full speed.
        If you need accurate timing, do a small warm-up run first to absorb
        that compilation cost.
        """
        n_assets = portfolio.S0.shape[0]

        if n_assets > _MAX_ASSETS:
            raise ValueError(
                f"GPU kernel supports up to {_MAX_ASSETS} assets, got {n_assets}. "
                f"Increase _MAX_ASSETS and recompile the kernel if needed."
            )
        if corr_matrix.shape != (n_assets, n_assets):
            raise ValueError(
                f"corr_matrix shape {corr_matrix.shape} does not match "
                f"n_assets={n_assets}"
            )

        # Pre-compute the constant GBM terms on the CPU once, before the kernel
        # launch. This avoids repeating these calculations inside every thread
        # at every time step (would be n_paths * n_steps redundant operations).
        chol = compute_cholesky(corr_matrix)
        dt = market_model.dt
        drift = (market_model.mu - 0.5 * market_model.sigma**2) * dt
        diff_scale = market_model.sigma * np.sqrt(dt)

        # Initialise one independent RNG state per path.
        # xoroshiro128p is fast, passes statistical quality tests, and is the
        # generator recommended by Numba for GPU Monte Carlo simulations.
        rng_seed = seed if seed is not None else int(time.time_ns() % (2**31))
        rng_states = create_xoroshiro128p_states(n_paths, seed=rng_seed)

        # Copy all input arrays to GPU memory ("d_" prefix = device array).
        d_s0 = cuda.to_device(portfolio.S0.astype(np.float64))
        d_weights = cuda.to_device(portfolio.weights.astype(np.float64))
        d_drift = cuda.to_device(drift.astype(np.float64))
        d_diff_scale = cuda.to_device(diff_scale.astype(np.float64))
        d_chol = cuda.to_device(chol.astype(np.float64))

        # Allocate the output array directly on the GPU so the kernel can write
        # results straight into device memory without an extra copy.
        d_losses = cuda.device_array(n_paths, dtype=np.float64)

        # Round up the block count so that every path gets exactly one thread,
        # even when n_paths is not a multiple of threads_per_block.
        threads = self.threads_per_block
        blocks = (n_paths + threads - 1) // threads

        # Launch the kernel. Numba's [blocks, threads] syntax is the Python
        # equivalent of CUDA C's <<<blocks, threads>>> launch configuration.
        _gbm_kernel[blocks, threads](
            rng_states,
            d_s0,
            d_weights,
            d_drift,
            d_diff_scale,
            market_model.n_steps,
            d_chol,
            n_assets,
            d_losses,
        )

        # Transfer results from GPU memory back to a regular NumPy array on the CPU.
        return d_losses.copy_to_host()
