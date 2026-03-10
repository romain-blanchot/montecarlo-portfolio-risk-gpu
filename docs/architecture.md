# Architecture & GPU Optimisation Guide

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Model](#mathematical-model)
3. [Code Architecture](#code-architecture)
4. [CPU Engine](#cpu-engine)
5. [GPU Engine](#gpu-engine)
6. [GPU Optimisations In Depth](#gpu-optimisations-in-depth)
7. [GPU Memory Hierarchy](#gpu-memory-hierarchy)
8. [Thread Layout](#thread-layout)
9. [Performance Analysis](#performance-analysis)
10. [Usage Examples](#usage-examples)
11. [Swapping Engines](#swapping-engines)

---

## Overview

This project estimates portfolio risk metrics — **Value at Risk (VaR)** and
**Expected Shortfall (ES)** — via Monte Carlo simulation of a multi-asset
portfolio modelled as **Geometric Brownian Motion (GBM)** with inter-asset
correlations captured through a **Cholesky decomposition**.

The same mathematical simulation is implemented twice:

| Engine          | File                                           | Backend    | When to use                                     |
|-----------------|------------------------------------------------|------------|-------------------------------------------------|
| `MonteCarloCPU` | `infrastructure/simulation/monte_carlo_cpu.py` | Pure NumPy | Development, small simulations, no GPU          |
| `MonteCarloGPU` | `infrastructure/simulation/monte_carlo_gpu.py` | Numba CUDA | Production, ≥ 100 k paths, NVIDIA GPU available |

Both share the identical `SimulationEngine` interface so switching is a
one-line change.

---

## Mathematical Model

### Geometric Brownian Motion (GBM)

Each asset *i* follows:

```
dS_i = μ_i S_i dt + σ_i S_i dW_i
```

Discretised with Euler-Maruyama over a time step `dt`:

```
S_i(t + dt) = S_i(t) · exp( (μ_i - ½σ_i²) dt  +  σ_i √dt · Z_corr_i )
```

where `Z_corr` is a **correlated** standard normal vector.

### Cholesky Correlation

Independent normals `Z_indep ~ N(0, I)` are correlated via the lower-triangular
Cholesky factor `L` of the correlation matrix `Σ`:

```
Σ = L Lᵀ
Z_corr = L · Z_indep      ⟹      Cov(Z_corr) = L Lᵀ = Σ
```

### Portfolio Loss

Given initial prices `S0` and weights `w` (summing to 1):

```
V0  = wᵀ S0                    (initial portfolio value)
V_T = wᵀ S_T                   (terminal portfolio value)
loss = V0 − V_T                 (positive → portfolio lost money)
```

### Risk Metrics

```
VaR_α   = quantile(losses, α)
ES_α    = E[ loss | loss ≥ VaR_α ]
```

---

## Code Architecture

The project follows a **layered hexagonal** structure. Business logic never
depends on infrastructure details; the GPU kernel can be swapped without
touching a single domain or application file.

```
src/portfolio_risk_engine/
│
├── domain/                        ← Pure business logic, no I/O
│   ├── market_model.py            MarketModel dataclass
│   ├── portfolio.py               Portfolio dataclass + initial_value
│   ├── correlation.py             compute_cholesky()
│   ├── var.py                     compute_var()
│   └── expected_shortfall.py      compute_es()
│
├── application/                   ← Orchestration (use cases)
│   ├── run_simulation.py          run()  — CPU-only end-to-end
│   └── compare_engines.py         compare()  — CPU vs GPU timing
│
└── infrastructure/                ← External adapters (compute)
    └── simulation/
        ├── base.py                SimulationEngine ABC
        ├── monte_carlo_cpu.py     MonteCarloCPU  (NumPy)
        └── monte_carlo_gpu.py     MonteCarloGPU  (Numba CUDA)
```

**Dependency rule:** arrows point inward only.

```
infrastructure ──→ domain
application    ──→ domain
application    ──→ infrastructure
```

`domain/` has zero imports from `application/` or `infrastructure/`.

---

## CPU Engine

### Strategy: vectorise over paths, loop over time-steps

```python
S = np.tile(S0, (n_paths, 1))          # (n_paths, n_assets)

for _ in range(n_steps):
    Z_indep  = rng.standard_normal((n_paths, n_assets))
    Z_corr   = Z_indep @ chol.T        # (n_paths, n_assets)
    S       *= np.exp(drift + diff_scale * Z_corr)

losses = V0 - S @ weights              # (n_paths,)
```

All `n_paths` are advanced **simultaneously** each step using NumPy broadcasting.
This keeps all data in L2/L3 cache between the matrix-multiply and the
`exp` call, which is the dominant cost.

### Complexity

| Operation | Cost per step |
|---|---|
| Random draw | O(n\_paths × n\_assets) |
| Cholesky matmul `Z @ chol.T` | O(n\_paths × n\_assets²) |
| GBM update (exp) | O(n\_paths × n\_assets) |

**Total:** `O(n_paths × n_steps × n_assets²)`

Sequential paths → wall time scales **linearly** with `n_paths`.

---

## GPU Engine

### Core Idea: one CUDA thread per path

Monte Carlo simulation is **embarrassingly parallel** — each path is
completely independent. Mapping one thread to one path eliminates all
inter-thread communication during the simulation loop.

```
┌─────────────────────────────────────────────────┐
│  GPU Grid                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Block 0  │  │ Block 1  │  │ Block B  │  ... │
│  │ 256 thds │  │ 256 thds │  │ 256 thds │      │
│  │ path 0   │  │ path 256 │  │ path B×256│     │
│  │  ...     │  │  ...     │  │  ...     │      │
│  │ path 255 │  │ path 511 │  │  ...     │      │
│  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────┘
```

### Kernel Signature

```python
_gbm_kernel[blocks, threads](
    rng_states,   # (n_paths,)        per-thread RNG state
    S0,           # (n_assets,)       initial prices
    weights,      # (n_assets,)       portfolio weights
    drift,        # (n_assets,)       (μ − ½σ²)·dt    [CPU pre-computed]
    diff_scale,   # (n_assets,)       σ·√dt           [CPU pre-computed]
    n_steps,      # int
    chol,         # (n_assets, n_assets)   Cholesky factor
    n_assets,     # int
    losses,       # (n_paths,)        OUTPUT
)
```

### Kernel Pseudocode

```
thread tid = global thread index  (= path index)

# --- Shared memory load (once per block) ---
cooperative_load(s_S0, s_weights, s_drift, s_diff_scale, s_chol)
syncthreads()

# --- Per-thread simulation (no synchronisation needed) ---
S[0..n_assets] = s_S0[:]

for step in range(n_steps):
    for i in range(n_assets):
        Z_indep[i] = xoroshiro128p_normal(rng_states, tid)

    for i in range(n_assets):          # Cholesky: lower-triangular matmul
        Z_corr[i] = sum(s_chol[i,j] * Z_indep[j]  for j in 0..i)

    for i in range(n_assets):
        S[i] *= exp(s_drift[i] + s_diff_scale[i] * Z_corr[i])

# --- Coalesced write ---
losses[tid] = sum(s_weights[i] * (s_S0[i] - S[i]))
```

---

## GPU Optimisations In Depth

### 1. Shared Memory for Read-Only Constants

Every thread needs the Cholesky matrix `L`, `drift`, `diff_scale`, `S0`, and
`weights` on **every time step**. Without optimisation, each thread would read
these from **global memory** (~600 GB/s, ~400 cycle latency).

With shared memory (~20 TB/s bandwidth, ~5 cycle latency) the block loads
these once, and every subsequent access hits on-chip memory.

```
Global memory (slow)                Shared memory (fast)
  chol[32×32×8 B = 8 KB]    ──→     s_chol  (one load per block)
  drift[32×8 B = 256 B]     ──→     s_drift
  diff_scale[...]            ──→     s_diff_scale
  S0[...]                    ──→     s_S0
  weights[...]               ──→     s_weights
```

**Cooperative load pattern** — all `block_size` threads stride through the
data together, maximising load bandwidth:

```python
i = cuda.threadIdx.x
while i < n_assets:
    s_S0[i] = S0[i]
    i += cuda.blockDim.x          # stride = block size
```

For the Cholesky matrix (2-D), the same pattern is applied to the flattened
`n_assets × n_assets` index space.

**Shared memory budget per block (MAX_ASSETS = 32, float64):**

| Array | Size |
|---|---|
| `s_chol` (32 × 32) | 8 192 B |
| `s_S0`, `s_weights`, `s_drift`, `s_diff_scale` (4 × 32) | 1 024 B |
| **Total** | **~9 KB** |

Modern GPUs provide 48–96 KB shared memory per SM, so multiple blocks can
reside concurrently.

---

### 2. Per-Thread Register Arrays

Each thread's **current price vector** `S[n_assets]` and innovation vectors
`Z_indep`, `Z_corr` are declared as `cuda.local.array`:

```python
S      = cuda.local.array(_MAX_ASSETS, dtype=np.float64)
Z_indep = cuda.local.array(_MAX_ASSETS, dtype=np.float64)
Z_corr  = cuda.local.array(_MAX_ASSETS, dtype=np.float64)
```

The Numba/PTX compiler promotes small local arrays to **registers** when
possible (register count permitting). Registers are the fastest storage
on the GPU — zero latency, zero bandwidth cost.

For `n_assets = 5`, these three arrays occupy `3 × 5 × 8 = 120 bytes` —
comfortably within the register file of a single thread.

---

### 3. CPU-Side Pre-Computation of Drift and Diffusion Scale

Inside the kernel's hot loop the GBM update is:

```
S[i] *= exp( drift[i]  +  diff_scale[i] * Z_corr[i] )
```

where `drift[i] = (μ_i − ½σ_i²)·dt` and `diff_scale[i] = σ_i·√dt`.

These are **path-independent constants** that the CPU computes once before
the kernel launch:

```python
drift      = (mu - 0.5 * sigma**2) * dt   # computed on CPU, shape (n_assets,)
diff_scale = sigma * np.sqrt(dt)           # computed on CPU, shape (n_assets,)
```

Without pre-computation, every one of the `n_paths × n_steps × n_assets`
inner-loop iterations would redundantly recompute these values on-device.

**Arithmetic saved:** `4 × n_assets × n_steps × n_paths` floating-point ops.

For `n_paths = 1 M`, `n_steps = 252`, `n_assets = 5`: **≈ 5 billion ops saved**.

---

### 4. xoroshiro128p Per-Thread RNG

Numba provides `xoroshiro128p` — a 128-bit state **PRNG** designed for GPU:

```python
Z_indep[i] = xoroshiro128p_normal_float64(rng_states, tid)
```

Key properties:
- **Per-thread state** — thread `tid` reads and writes `rng_states[tid]` only.
  No atomic operations, no synchronisation, no contention.
- **High statistical quality** — passes the BigCrush battery; period 2¹²⁸.
- **Fast** — ~2–3 arithmetic ops per sample; the state fits in 2 registers.
- **Independent streams** — `create_xoroshiro128p_states(n, seed)` uses a
  jump-ahead algorithm to initialise `n` statistically independent streams
  from a single seed, guaranteeing non-overlapping sequences.

---

### 5. Thread Block Size: 256 Threads (8 Warps)

CUDA executes threads in groups of **32 (one warp)**. The block size of 256
threads = 8 warps is chosen to:

1. **Hide latency** — while one warp stalls on a memory access, 7 others can
   execute, keeping the SM's ALUs busy (latency hiding / instruction-level
   parallelism).
2. **Maximise occupancy** — fits well within the 48 KB shared memory budget
   (≈ 9 KB / block) and typical register-file limits.
3. **Minimise tail waste** — for `n_paths` not divisible by 256, at most
   255 threads in the last block are idle — negligible for large simulations.

Configurable via `MonteCarloGPU(threads_per_block=512)` for experimentation.

---

### 6. Coalesced Global Memory Write

The final loss write:

```python
losses[tid] = v0 - vt
```

Thread `tid = 0, 1, 2, ...` writes to `losses[0], losses[1], losses[2], ...` —
consecutive addresses. A warp of 32 threads issues a **single 256-byte
transaction** to global memory instead of 32 separate transactions.

This pattern achieves peak global-memory write bandwidth.

---

### 7. No Branch Divergence

Within a warp, all 32 threads must execute the same instruction at each clock
cycle (SIMT model). Divergent branches cause serialisation.

The inner loops are **data-independent**:
- `for i in range(n_assets)` — same trip count for every thread in the block.
- `for j in range(i + 1)` — same per-warp, no divergence.
- The `if tid >= n_paths: return` guard at the start is the only branch, and
  it only affects the last (possibly partial) block.

---

### Optimisation Summary Table

| # | Technique | Memory tier | Impact |
|---|---|---|---|
| 1 | Shared memory for constants | Global → Shared | ~100× latency reduction for L, drift, σ reads |
| 2 | Register arrays for per-path state | Local → Registers | Zero-latency access to S, Z per step |
| 3 | CPU-side drift / diff\_scale | CPU arithmetic → removed | ~5 B ops saved (1 M paths, 252 steps, 5 assets) |
| 4 | xoroshiro128p per-thread RNG | No atomics | Linear scaling, no synchronisation overhead |
| 5 | 256 threads/block | Occupancy | Maximises SM utilisation via latency hiding |
| 6 | Coalesced losses write | Global BW | Peak global-memory bandwidth on output |
| 7 | No branch divergence | SIMT | All 32 warp lanes execute in lockstep |

---

## GPU Memory Hierarchy

```
          Latency    BW (est.)    Size         Scope
          ──────     ──────────   ──────────   ────────────────────
Registers   1 cy      N/A          ~256 KB/SM   Per thread
──────────────────────────────────────────────────────────────────
L1 / Shared 5 cy     ~20 TB/s     48–96 KB/SM  Per thread-block
──────────────────────────────────────────────────────────────────
L2 Cache   30 cy     ~4 TB/s      4–64 MB      Per GPU
──────────────────────────────────────────────────────────────────
Global     400 cy    ~0.6–2 TB/s  4–80 GB      All threads
──────────────────────────────────────────────────────────────────
Host (RAM) >10 000cy  PCIe ~64 GB/s  system     CPU ↔ GPU transfer
```

The kernel design keeps the hot-loop data (S, Z, L, drift, σ) entirely
in registers and shared memory. Global memory is only accessed for:
- Initial constant load (once per block, amortised over thousands of paths)
- RNG state read/write per step (unavoidable, but sequential per thread)
- `losses[tid]` write (once per path, coalesced)

---

## Thread Layout

```
n_paths = 1 000 000   threads_per_block = 256

blocks = ceil(1 000 000 / 256) = 3 907

Grid: 3 907 blocks × 256 threads = 1 000 192 threads total
       └─ last block: 192 active threads, 64 idle (guarded by tid >= n_paths)

Each SM (e.g. A100 has 108 SMs) runs multiple blocks concurrently.
With 9 KB shared mem / block and 96 KB available → up to 10 blocks / SM
→  10 × 256 = 2 560 resident threads / SM   (good occupancy)
```

---

## Performance Analysis

### Theoretical Arithmetic Intensity

Per path, per step:
- `n_assets` RNG calls
- `n_assets²/2` multiplications (Cholesky, lower-triangular)
- `n_assets` exp calls (dominant cost; `exp` ≈ 20 FLOP equivalent)
- `n_assets` multiplications (GBM update)

For `n_assets = 5, n_steps = 252`:
- Total ≈ `252 × (5 + 12.5 + 100 + 5)` ≈ **30 800 FLOP / path**

### Expected Speedup vs CPU

| GPU | FP64 TFLOPS | n_paths = 1M, n_steps = 252, n_assets = 5 |
|---|---|---|
| RTX 4090 | 1.3 TFLOPS | ~24 s → ~0.025 s (**~1 000×**) |
| A100 | 9.7 TFLOPS | ~24 s → ~0.003 s (**~8 000×**) |
| V100 | 7.0 TFLOPS | ~24 s → ~0.004 s (**~6 000×**) |

> **Note:** Consumer GPUs (RTX series) have reduced FP64 throughput; the
> dominant cost in practice is the `math.exp` call, not pure FLOP throughput.
> Real speedups for `n_assets ≤ 10` are typically **50×–500×** on a modern
> desktop GPU.

### JIT Compilation Overhead

Numba compiles the kernel on **first use** (typically 2–5 s). Subsequent
calls reuse the compiled PTX. For benchmarking, always run one warm-up pass:

```python
engine.run(portfolio, mm, corr, n_paths=1_000, seed=0)   # warm-up (absorbs JIT)
# now benchmark
engine.run(portfolio, mm, corr, n_paths=1_000_000, seed=0)
```

`compare()` does this automatically when `warmup=True` (the default).

---

## Usage Examples

### CPU engine

```python
import numpy as np
from portfolio_risk_engine.domain.market_model import MarketModel
from portfolio_risk_engine.domain.portfolio import Portfolio
from portfolio_risk_engine.application.run_simulation import run

result = run(
    portfolio=Portfolio(
        S0=np.array([100.0, 80.0, 120.0]),
        weights=np.array([0.4, 0.3, 0.3]),
    ),
    market_model=MarketModel(
        mu=np.array([0.08, 0.05, 0.10]),
        sigma=np.array([0.20, 0.15, 0.25]),
        dt=1 / 252,
        n_steps=252,
    ),
    corr_matrix=np.array([
        [1.0, 0.4, 0.2],
        [0.4, 1.0, 0.5],
        [0.2, 0.5, 1.0],
    ]),
    n_paths=100_000,
    seed=42,
)
print(f"VaR 95%: {result['var']:.4f}")
print(f"ES  95%: {result['es']:.4f}")
```

### GPU engine (direct)

```python
from portfolio_risk_engine.infrastructure.simulation.monte_carlo_gpu import MonteCarloGPU

engine = MonteCarloGPU()
losses = engine.run(portfolio, market_model, corr_matrix, n_paths=1_000_000, seed=42)
```

### Compare CPU vs GPU

```python
from portfolio_risk_engine.application.compare_engines import compare

result = compare(
    portfolio, market_model, corr_matrix,
    n_paths=500_000,
    warmup=True,  # absorbs JIT compilation
)
print(f"Speedup: {result['speedup']:.1f}×")
print(f"VaR  CPU={result['cpu_var']:.4f}  GPU={result['gpu_var']:.4f}")
```

---

## Swapping Engines

The `SimulationEngine` ABC guarantees an identical interface:

```python
# base.py
class SimulationEngine(ABC):
    @abstractmethod
    def run(
        self,
        portfolio: Portfolio,
        market_model: MarketModel,
        corr_matrix: np.ndarray,
        n_paths: int,
        seed: int | None = None,
    ) -> np.ndarray: ...
```

Switching from CPU to GPU (or to any future engine) requires changing
**one line**:

```python
# Before
engine = MonteCarloCPU()

# After (requires NVIDIA GPU + pip install portfolio-risk-engine[gpu])
engine = MonteCarloGPU()

# The rest of the code is identical
losses = engine.run(portfolio, market_model, corr_matrix, n_paths=1_000_000)
```

### Adding a new engine

1. Create `infrastructure/simulation/monte_carlo_<backend>.py`
2. Subclass `SimulationEngine` and implement `run()`
3. No changes required to `domain/` or `application/`
