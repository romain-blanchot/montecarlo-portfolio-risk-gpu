"""GPU-accelerated Monte Carlo engine via Numba CUDA.

This module implements :class:`MonteCarloGPU`, a drop-in replacement for
:class:`~portfolio_risk_engine.infrastructure.simulation.monte_carlo_cpu.MonteCarloCPU`
that offloads the simulation to an NVIDIA GPU using Numba CUDA JIT kernels.

Requires the ``gpu`` optional dependency group::

    pip install portfolio-risk-engine[gpu]

If numba is not installed or no CUDA-capable GPU is detected, instantiating
:class:`MonteCarloGPU` raises a descriptive :exc:`RuntimeError`.

GPU Optimisations Summary
-------------------------
1. **One thread per path** — zero inter-thread communication; perfectly parallel.
2. **Shared memory for read-only constants** — Cholesky matrix, drift, diffusion
   scale, S0, and weights are loaded once per thread-block from global memory into
   fast shared memory (~100× lower latency than global).
3. **Per-thread register arrays** — current price vector ``S`` and the innovation
   vectors ``Z_indep`` / ``Z_corr`` live in registers, the fastest memory tier.
4. **Pre-computed drift & diffusion scale** — ``(mu - ½σ²)·dt`` and ``σ·√dt`` are
   calculated on the CPU once before kernel launch, eliminating these operations
   from the hot inner loop (saved: 4·n_assets ops × n_paths × n_steps).
5. **xoroshiro128p RNG** — per-thread state means no atomic operations or global
   synchronisation; each thread advances its own 128-bit state independently.
6. **256 threads/block** — 8 full warps per block; hides memory-latency behind
   arithmetic while staying within shared-memory budget (≈ 9 KB / block for
   MAX_ASSETS = 32).
7. **Coalesced global-memory write** — ``losses[tid]`` is written once per thread
   with consecutive thread IDs → maximises global-memory bandwidth.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from portfolio_risk_engine.domain.correlation import compute_cholesky
from portfolio_risk_engine.domain.market_model import MarketModel
from portfolio_risk_engine.domain.portfolio import Portfolio
from portfolio_risk_engine.infrastructure.simulation.base import SimulationEngine

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Compile-time constants (must be resolvable by the Numba JIT compiler)
# ---------------------------------------------------------------------------
_MAX_ASSETS: int = 32  # maximum number of assets supported by the kernel
_TPB: int = 256  # threads per block (= 8 warps; good occupancy heuristic)

# ---------------------------------------------------------------------------
# Optional Numba import — module still loads without numba installed
# ---------------------------------------------------------------------------
_NUMBA_AVAILABLE: bool = False
_CUDA_IS_AVAILABLE: bool = False

# Always-bound references — overwritten by the successful import below.
cuda: Any = None
create_xoroshiro128p_states: Any = None
xoroshiro128p_normal_float64: Any = None

try:
    from numba import cuda  # type: ignore[import-untyped,import-not-found,no-redef,assignment]
    from numba.cuda.random import (  # type: ignore[import-untyped,import-not-found,no-redef]
        create_xoroshiro128p_states,
        xoroshiro128p_normal_float64,
    )

    _NUMBA_AVAILABLE = True
    try:
        _CUDA_IS_AVAILABLE = bool(cuda.is_available())  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        _CUDA_IS_AVAILABLE = False

except ImportError:
    pass

# ---------------------------------------------------------------------------
# CUDA kernel  (compiled only when Numba is present)
# ---------------------------------------------------------------------------
if _NUMBA_AVAILABLE:

    @cuda.jit  # type: ignore[name-defined]
    def _gbm_kernel(
        rng_states,  # xoroshiro128p states  (n_paths,)
        S0,  # initial prices         (n_assets,)  float64
        weights,  # portfolio weights      (n_assets,)  float64
        drift,  # (mu - ½σ²)·dt          (n_assets,)  float64  [pre-computed CPU-side]
        diff_scale,  # σ·√dt                  (n_assets,)  float64  [pre-computed CPU-side]
        n_steps,  # int  — number of time-steps
        chol,  # Cholesky factor        (n_assets, n_assets) float64 lower-triangular
        n_assets,  # int  — actual number of assets (≤ _MAX_ASSETS)
        losses,  # OUTPUT                 (n_paths,)   float64
    ):
        """One CUDA thread simulates one Monte Carlo path end-to-end.

        Memory layout
        -------------
        Global   → shared  : S0, weights, drift, diff_scale, chol  (one load per block)
        Shared   → registers: per-step arithmetic on Z and S arrays
        Registers→ global  : losses[tid]  (single coalesced write at the end)
        """
        # ------------------------------------------------------------------
        # Shared memory: read-only constants, loaded cooperatively by the block
        # Shape must be a compile-time literal — _MAX_ASSETS satisfies this.
        # ------------------------------------------------------------------
        s_S0 = cuda.shared.array(_MAX_ASSETS, dtype=np.float64)  # noqa: N806
        s_weights = cuda.shared.array(_MAX_ASSETS, dtype=np.float64)  # noqa: N806
        s_drift = cuda.shared.array(_MAX_ASSETS, dtype=np.float64)  # noqa: N806
        s_diff_scale = cuda.shared.array(_MAX_ASSETS, dtype=np.float64)  # noqa: N806
        s_chol = cuda.shared.array((_MAX_ASSETS, _MAX_ASSETS), dtype=np.float64)  # noqa: N806

        local_tid = cuda.threadIdx.x
        block_size = cuda.blockDim.x

        # --- Cooperative load of 1-D constant vectors ----------------------
        # Each thread loads one element; threads cycle until all n_assets loaded.
        # This gives up to block_size-way parallelism for the load.
        i = local_tid
        while i < n_assets:
            s_S0[i] = S0[i]
            s_weights[i] = weights[i]
            s_drift[i] = drift[i]
            s_diff_scale[i] = diff_scale[i]
            i += block_size

        # --- Cooperative load of the Cholesky matrix -----------------------
        # Flatten (n_assets × n_assets) into a 1-D index for easy striding.
        flat = local_tid
        flat_size = n_assets * n_assets
        while flat < flat_size:
            row = flat // n_assets
            col = flat % n_assets
            s_chol[row, col] = chol[row, col]
            flat += block_size

        # All threads must see the fully loaded shared memory before proceeding.
        cuda.syncthreads()

        # ------------------------------------------------------------------
        # Early-exit for out-of-bounds threads (last block may be incomplete)
        # ------------------------------------------------------------------
        tid = cuda.grid(1)  # global thread index = path index
        n_paths = losses.shape[0]
        if tid >= n_paths:
            return

        # ------------------------------------------------------------------
        # Per-thread local arrays — live in registers (fastest tier)
        # Unused entries (index ≥ n_assets) are simply never accessed.
        # ------------------------------------------------------------------
        S = cuda.local.array(_MAX_ASSETS, dtype=np.float64)
        Z_indep = cuda.local.array(_MAX_ASSETS, dtype=np.float64)
        Z_corr = cuda.local.array(_MAX_ASSETS, dtype=np.float64)

        # Initialise current prices from shared S0
        for i in range(n_assets):
            S[i] = s_S0[i]

        # ==================================================================
        # Main simulation loop  O(n_steps × n_assets²) per thread
        # ==================================================================
        for _ in range(n_steps):
            # 1. Draw n_assets independent N(0,1) samples using thread-local RNG
            for i in range(n_assets):
                Z_indep[i] = xoroshiro128p_normal_float64(rng_states, tid)  # type: ignore[name-defined]

            # 2. Correlate innovations: Z_corr = L @ Z_indep  (L lower-triangular)
            #    Inner loop runs at most i+1 iterations — no wasted iterations.
            for i in range(n_assets):
                acc = 0.0
                for j in range(i + 1):
                    acc += s_chol[i, j] * Z_indep[j]
                Z_corr[i] = acc

            # 3. GBM update: S_t = S_{t-1} · exp(drift_i + diff_scale_i · Z_corr_i)
            #    drift and diff_scale already incorporate dt (pre-computed on CPU)
            for i in range(n_assets):
                S[i] *= math.exp(s_drift[i] + s_diff_scale[i] * Z_corr[i])

        # ==================================================================
        # Compute portfolio loss for this path and write to global memory
        # Consecutive tids write to consecutive addresses → coalesced write
        # ==================================================================
        v0 = 0.0
        vt = 0.0
        for i in range(n_assets):
            v0 += s_weights[i] * s_S0[i]
            vt += s_weights[i] * S[i]

        losses[tid] = v0 - vt

else:
    _gbm_kernel = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public engine class
# ---------------------------------------------------------------------------


class MonteCarloGPU(SimulationEngine):
    """GPU-accelerated Monte Carlo engine (Numba CUDA).

    Each CUDA thread simulates exactly one Monte Carlo path.  Read-only
    constants (Cholesky matrix, drift, diffusion scale) are cooperatively
    loaded into shared memory by the thread block so that the per-step
    arithmetic exclusively hits fast on-chip memory.

    Parameters
    ----------
    threads_per_block:
        Number of CUDA threads per block.  Must be a multiple of the warp
        size (32).  Default is 256 (8 warps), which is a good starting
        point for most modern GPUs.

    Raises
    ------
    RuntimeError
        On instantiation if Numba is not installed or no CUDA-capable GPU
        is detected.
    """

    def __init__(self, threads_per_block: int = _TPB) -> None:
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
        if threads_per_block % 32 != 0:
            raise ValueError(
                f"threads_per_block must be a multiple of 32 (warp size), "
                f"got {threads_per_block}"
            )
        self.threads_per_block = threads_per_block

    # ------------------------------------------------------------------
    def run(
        self,
        portfolio: Portfolio,
        market_model: MarketModel,
        corr_matrix: np.ndarray,
        n_paths: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """See :class:`~portfolio_risk_engine.infrastructure.simulation.base.SimulationEngine`.

        Notes
        -----
        The *first* call triggers Numba JIT compilation of the CUDA kernel
        (typically a few seconds).  Subsequent calls reuse the compiled PTX.
        Pass a warm-up call with a small ``n_paths`` if you need accurate
        timing of the actual simulation.
        """
        n_assets = portfolio.S0.shape[0]

        if n_assets > _MAX_ASSETS:
            raise ValueError(
                f"GPU kernel supports up to {_MAX_ASSETS} assets, got {n_assets}. "
                f"Increase _MAX_ASSETS and recompile the kernel if needed."
            )
        if corr_matrix.shape != (n_assets, n_assets):
            raise ValueError(
                f"corr_matrix shape {corr_matrix.shape} is inconsistent with "
                f"n_assets={n_assets}"
            )

        # --- CPU-side pre-computation (avoids repeated arithmetic in kernel) ---
        chol = compute_cholesky(corr_matrix)
        dt = market_model.dt
        drift = (market_model.mu - 0.5 * market_model.sigma**2) * dt  # (n_assets,)
        diff_scale = market_model.sigma * np.sqrt(dt)  # (n_assets,)

        # --- Initialise per-thread RNG states on the device ------------------
        rng_seed = seed if seed is not None else int(time.time_ns() % (2**31))
        rng_states = create_xoroshiro128p_states(n_paths, seed=rng_seed)  # type: ignore[name-defined]

        # --- Transfer constant arrays to device (small; negligible overhead) --
        d_S0 = cuda.to_device(portfolio.S0.astype(np.float64))  # type: ignore[name-defined]
        d_weights = cuda.to_device(portfolio.weights.astype(np.float64))  # type: ignore[name-defined]
        d_drift = cuda.to_device(drift.astype(np.float64))  # type: ignore[name-defined]
        d_diff_scale = cuda.to_device(diff_scale.astype(np.float64))  # type: ignore[name-defined]
        d_chol = cuda.to_device(chol.astype(np.float64))  # type: ignore[name-defined]

        # --- Allocate output directly on device (avoids an extra host copy) ---
        d_losses = cuda.device_array(n_paths, dtype=np.float64)  # type: ignore[name-defined]

        # --- Compute grid dimensions -----------------------------------------
        # Round up so the last (potentially partial) block is still launched.
        threads = self.threads_per_block
        blocks = (n_paths + threads - 1) // threads

        # --- Launch kernel ---------------------------------------------------
        _gbm_kernel[blocks, threads](  # type: ignore[index]
            rng_states,
            d_S0,
            d_weights,
            d_drift,
            d_diff_scale,
            market_model.n_steps,
            d_chol,
            n_assets,
            d_losses,
        )

        # --- Copy result back to host and return as NumPy array --------------
        return d_losses.copy_to_host()  # type: ignore[union-attr]
