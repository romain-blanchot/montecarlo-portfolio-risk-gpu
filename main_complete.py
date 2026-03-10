"""
Monte Carlo Portfolio Risk Engine — Guided Tour
================================================

This script is an interactive walkthrough of the project.
Run it with:

    python main.py

It demonstrates every layer of the architecture, from low-level domain
objects all the way up to the GPU benchmark — explaining what happens at
each step so you can quickly understand how to build on top of the engine.

Table of contents
-----------------
  PART 1 — Domain layer
    1. Portfolio & MarketModel
    2. Correlation matrix + Cholesky decomposition

  PART 2 — Simulation (CPU)
    3. High-level one-liner via run()
    4. Low-level direct engine usage
    5. Loss distribution & risk metrics
    6. Multiple confidence levels

  PART 3 — Portfolio analysis
    7. Comparing three allocation strategies
    8. Stress test (market-crisis correlations)

  PART 4 — GPU (if an NVIDIA GPU is present)
    9.  GPU availability check
    10. GPU simulation
    11. CPU vs GPU benchmark via compare()

  PART 5 — Architecture summary
"""

from __future__ import annotations

import sys
import time

# Force UTF-8 output so Unicode characters (arrows, Greek letters, etc.)
# render correctly on Windows terminals and when piping to a file.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except AttributeError:
    pass

import numpy as np

from portfolio_risk_engine.application.compare_engines import compare
from portfolio_risk_engine.application.run_simulation import run
from portfolio_risk_engine.domain.correlation import compute_cholesky
from portfolio_risk_engine.domain.expected_shortfall import compute_es
from portfolio_risk_engine.domain.market_model import MarketModel
from portfolio_risk_engine.domain.portfolio import Portfolio
from portfolio_risk_engine.domain.var import compute_var
from portfolio_risk_engine.infrastructure.simulation.monte_carlo_cpu import (
    MonteCarloCPU,
)
from portfolio_risk_engine.infrastructure.simulation.monte_carlo_gpu import (
    MonteCarloGPU,
    _CUDA_IS_AVAILABLE,
    _NUMBA_AVAILABLE,
)

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

_W = 76  # console width


def _banner(title: str) -> None:
    print()
    print("=" * _W)
    print(f"  {title}")
    print("=" * _W)


def _section(title: str) -> None:
    print()
    line_fill = "-" * max(0, _W - len(title) - 5)
    print(f"  --- {title} {line_fill}")


def _note(text: str) -> None:
    """Print an explanatory note, indented and word-wrapped."""
    prefix = "  | "
    for line in text.strip().splitlines():
        print(prefix + line)


def _row(label: str, value: str, width: int = 36) -> None:
    print(f"  {label:<{width}} {value}")


# ===========================================================================
# INTRODUCTION
# ===========================================================================

_banner("Monte Carlo Portfolio Risk Engine — Guided Tour")

print("""
  This project simulates multi-asset portfolio dynamics using Geometric
  Brownian Motion (GBM) and computes standard risk metrics: Value-at-Risk
  (VaR) and Expected Shortfall (ES / CVaR).

  The code is organised in three layers:

    domain/          Pure business logic — no I/O, no compute backend.
                     Portfolio, MarketModel, compute_cholesky(), compute_var(),
                     compute_es().

    infrastructure/  Compute backends behind a common SimulationEngine ABC.
                     MonteCarloCPU (NumPy) and MonteCarloGPU (Numba CUDA).
                     Swap engines by changing a single line.

    application/     High-level use-cases that wire domain + infrastructure.
                     run()     — one-liner CPU simulation.
                     compare() — side-by-side CPU/GPU timing.

  Let's walk through each layer now.
""")

# ===========================================================================
# PART 1 — Domain layer
# ===========================================================================

_banner("PART 1  —  Domain Layer")

# ---------------------------------------------------------------------------
# Step 1 — Portfolio & MarketModel
# ---------------------------------------------------------------------------

_section("Step 1 — Portfolio & MarketModel")

_note("""
Portfolio holds the initial asset prices (S0) and their allocation weights.
The weights must sum to 1.0; __post_init__ validates this automatically.
initial_value is a @cached_property: weights @ S0.
""")

portfolio = Portfolio(
    S0=np.array([150.0, 80.0, 60.0]),  # initial prices in USD
    weights=np.array([0.50, 0.30, 0.20]),  # 50% Tech / 30% Finance / 20% Energy
)

print()
_row("Assets", "Tech (50%)  |  Finance (30%)  |  Energy (20%)")
_row("Initial prices  (USD)", "150.00  /  80.00  /  60.00")
_row("Initial portfolio value", f"${portfolio.initial_value:.2f}")

_note("""
MarketModel describes the GBM parameters for each asset:
  mu        — annualised drift rates (expected return)
  sigma     — annualised volatility (standard deviation)
  dt        — time-step in years  (1/252 ≈ one trading day)
  n_steps   — number of steps per simulation path
""")

market_model = MarketModel(
    mu=np.array([0.12, 0.07, 0.09]),  # 12% / 7% / 9% annualised drift
    sigma=np.array([0.28, 0.18, 0.22]),  # 28% / 18% / 22% annualised vol
    dt=1.0 / 252,  # daily steps
    n_steps=252,  # 1 full trading year
)

print()
_row("Drift  μ  (annualised)", "12%  /  7%  /  9%")
_row("Vol    σ  (annualised)", "28%  /  18%  /  22%")
_row("Horizon", "252 daily steps  →  1 trading year")

# ---------------------------------------------------------------------------
# Step 2 — Correlation matrix + Cholesky decomposition
# ---------------------------------------------------------------------------

_section("Step 2 — Correlation matrix + Cholesky decomposition")

_note("""
compute_cholesky() validates the matrix (symmetry, unit diagonal,
positive semi-definiteness) then returns the lower-triangular factor L
such that L @ L.T == corr_matrix.

The simulation engine uses L to correlate the independent N(0,1) shocks:
    Z_corr = Z_indep @ L.T
so the resulting paths respect the asset correlations.
""")

corr_matrix = np.array(
    [
        [1.00, 0.45, 0.25],
        [0.45, 1.00, 0.40],
        [0.25, 0.40, 1.00],
    ]
)

chol = compute_cholesky(corr_matrix)

print()
print("  Correlation matrix:")
for row in corr_matrix:
    print("    " + "  ".join(f"{v:6.2f}" for v in row))

print()
print("  Lower Cholesky factor  L  (L @ Lᵀ = Σ):")
for row in chol:
    print("    " + "  ".join(f"{v:8.5f}" for v in row))

print()
print("  Verification  max|L @ Lᵀ - Σ| =", np.max(np.abs(chol @ chol.T - corr_matrix)))

# ===========================================================================
# PART 2 — Simulation (CPU)
# ===========================================================================

_banner("PART 2  —  Simulation  (CPU / NumPy)")

N_PATHS = 50_000
SEED = 42

# ---------------------------------------------------------------------------
# Step 3 — High-level one-liner via run()
# ---------------------------------------------------------------------------

_section("Step 3 — High-level one-liner via run()")

_note("""
application.run_simulation.run() is the quickest way to get results.
It creates a MonteCarloCPU engine internally, runs the simulation, and
returns a SimulationResult TypedDict: { losses, var, es }.

One call is all you need for a standard CPU simulation.
""")

print(f"  Paths    : {N_PATHS:,}")
print(f"  Steps    : {market_model.n_steps}  (daily, 1 year)")
print(f"  Seed     : {SEED}")
print()

t0 = time.perf_counter()
result = run(
    portfolio=portfolio,
    market_model=market_model,
    corr_matrix=corr_matrix,
    n_paths=N_PATHS,
    confidence=0.95,
    seed=SEED,
)
cpu_elapsed = time.perf_counter() - t0

losses = result["losses"]

print(f"  Completed in {cpu_elapsed:.3f} s")
print()
_row("result['losses']", f"np.ndarray  shape={losses.shape}  dtype={losses.dtype}")
_row("result['var']", f"{result['var']:.4f}")
_row("result['es']", f"{result['es']:.4f}")

# ---------------------------------------------------------------------------
# Step 4 — Low-level direct engine usage
# ---------------------------------------------------------------------------

_section("Step 4 — Direct engine usage  (infrastructure layer)")

_note("""
You can bypass the application layer and drive a SimulationEngine directly.
This is useful when you need fine-grained control or want to hot-swap the
backend without changing any other code — GPU and CPU expose the same .run()
interface through the SimulationEngine ABC.

    engine = MonteCarloCPU()   # or MonteCarloGPU()
    losses = engine.run(portfolio, market_model, corr_matrix, n_paths, seed)

Then feed the raw losses array to the domain functions.
""")

engine = MonteCarloCPU()
raw_losses = engine.run(
    portfolio, market_model, corr_matrix, n_paths=N_PATHS, seed=SEED
)

var_95 = compute_var(raw_losses, confidence=0.95)
es_95 = compute_es(raw_losses, confidence=0.95)

print()
_row("engine.run() → losses shape", str(raw_losses.shape))
_row("compute_var(losses, 0.95)", f"{var_95:.4f}")
_row("compute_es (losses, 0.95)", f"{es_95:.4f}")
print()
print(
    "  Same values as result['var'] / result['es']:",
    f"var match={np.isclose(var_95, result['var'])},",
    f"es match={np.isclose(es_95, result['es'])}",
)

# ---------------------------------------------------------------------------
# Step 5 — Loss distribution
# ---------------------------------------------------------------------------

_section("Step 5 — Loss distribution")

_note("""
Losses are defined as:  loss = V0 - V_T
  positive value → portfolio lost money
  negative value → portfolio gained money  (a "gain" = negative loss)

The distribution characterises how widely outcomes are spread.
""")

print()
_row("Mean loss  (negative = avg gain)", f"{losses.mean():+.4f}")
_row("Std  deviation", f"{losses.std():.4f}")
_row("Min  (best-case outcome)", f"{losses.min():+.4f}")
_row("Max  (worst-case outcome)", f"{losses.max():+.4f}")
_row("% paths with a gain  (loss < 0)", f"{100 * float(np.mean(losses < 0)):.1f} %")

# ---------------------------------------------------------------------------
# Step 6 — Risk metrics at multiple confidence levels
# ---------------------------------------------------------------------------

_section("Step 6 — VaR and ES at multiple confidence levels")

_note("""
VaR(α)  =  α-quantile of the loss distribution.
           "With probability α we will NOT lose more than VaR."

ES(α)   =  mean of losses that exceed VaR(α).
           Also called CVaR or Tail Loss. It measures the severity of
           losses beyond the VaR threshold — a more conservative metric.

Both are computed purely from the losses array; they are engine-agnostic.
""")

print()
print(f"  {'Confidence':>12}  {'VaR':>10}  {'ES':>10}  {'Tail paths':>12}")
print("  " + "-" * 50)
for conf in (0.90, 0.95, 0.99):
    v = compute_var(losses, conf)
    e = compute_es(losses, conf)
    tail_pct = 100.0 * float(np.mean(losses > v))
    print(f"  {conf * 100:>10.0f}%  {v:>10.4f}  {e:>10.4f}  {tail_pct:>10.1f}%")

print()
_note(f"""
Reading the 95% row:
  VaR = {result["var"]:.4f}  →  95% of simulated days the portfolio
         loses ≤ ${abs(result["var"]):.2f}
  ES  = {result["es"]:.4f}  →  on the worst 5% of days the average
         loss is ${abs(result["es"]):.2f}  (the tail beyond VaR)
""")

# ===========================================================================
# PART 3 — Portfolio analysis
# ===========================================================================

_banner("PART 3  —  Portfolio Analysis")

# ---------------------------------------------------------------------------
# Step 7 — Comparing allocation strategies
# ---------------------------------------------------------------------------

_section("Step 7 — Comparing three allocation strategies")

_note("""
By changing only the 'weights' vector in Portfolio we can compare how
different allocations affect risk — all other parameters stay the same.
The same MonteCarloCPU engine instance is reused across all three runs.
""")

portfolios = {
    "Concentrated  (80/10/10)": Portfolio(
        S0=portfolio.S0,
        weights=np.array([0.80, 0.10, 0.10]),
    ),
    "Diversified   (50/30/20)": portfolio,
    "Equal-weight  (33/33/33)": Portfolio(
        S0=portfolio.S0,
        weights=np.array([1 / 3, 1 / 3, 1 / 3]),
    ),
}

print()
print(f"  {'Strategy':<30}  {'V0':>8}  {'VaR 95%':>9}  {'ES 95%':>9}  {'σ losses':>9}")
print("  " + "-" * 74)

for name, p in portfolios.items():
    ls = engine.run(p, market_model, corr_matrix, N_PATHS, seed=SEED)
    print(
        f"  {name:<30}  "
        f"{p.initial_value:>8.2f}  "
        f"{compute_var(ls, 0.95):>9.4f}  "
        f"{compute_es(ls, 0.95):>9.4f}  "
        f"{ls.std():>9.4f}"
    )

_note("""
Concentrating in the most volatile asset (Tech, σ=28%) pushes VaR and ES
up significantly — the equal-weight spread lowers tail risk the most.
""")

# ---------------------------------------------------------------------------
# Step 8 — Stress test: high-correlation crisis regime
# ---------------------------------------------------------------------------

_section("Step 8 — Stress test  (market-crisis correlations)")

_note("""
During a market crisis, asset correlations spike because investors liquidate
broadly. We re-run the same simulation with a stressed correlation matrix
(all pairs near 0.85–0.90) to quantify the impact on risk metrics.

This is a common regulatory/risk-management exercise (e.g. Basel III).
""")

corr_stress = np.array(
    [
        [1.00, 0.90, 0.85],
        [0.90, 1.00, 0.88],
        [0.85, 0.88, 1.00],
    ]
)

result_base = run(portfolio, market_model, corr_matrix, N_PATHS, seed=SEED)
result_stress = run(portfolio, market_model, corr_stress, N_PATHS, seed=SEED)

print()
print(f"  {'Scenario':<22}  {'VaR 95%':>9}  {'ES 95%':>9}  {'Std dev':>9}")
print("  " + "-" * 57)
print(
    f"  {'Normal (base)  corr':<22}  "
    f"{result_base['var']:>9.4f}  "
    f"{result_base['es']:>9.4f}  "
    f"{result_base['losses'].std():>9.4f}"
)
print(
    f"  {'Crisis (stress) corr':<22}  "
    f"{result_stress['var']:>9.4f}  "
    f"{result_stress['es']:>9.4f}  "
    f"{result_stress['losses'].std():>9.4f}"
)

pct_var = 100 * (result_stress["var"] - result_base["var"]) / abs(result_base["var"])
pct_es = 100 * (result_stress["es"] - result_base["es"]) / abs(result_base["es"])

print()
_row("VaR increase under stress", f"{pct_var:+.1f}%")
_row("ES  increase under stress", f"{pct_es:+.1f}%")

# ===========================================================================
# PART 4 — GPU  (optional — requires NVIDIA CUDA)
# ===========================================================================

_banner("PART 4  —  GPU Simulation  (Numba CUDA — optional)")

_section("Step 9 — GPU availability check")

_note("""
MonteCarloGPU requires:
  - numba installed  (pip install portfolio-risk-engine[gpu])
  - an NVIDIA GPU with the CUDA toolkit

It exposes the same SimulationEngine.run() interface as MonteCarloCPU,
so you swap it in with a single line — all downstream code is identical.

The first call triggers Numba JIT compilation (~2-5 s); subsequent calls
run at full GPU speed. Use warmup=True in compare() to absorb that cost.
""")

print()

if not _NUMBA_AVAILABLE:
    print("  Numba is not installed — GPU engine unavailable.")
    print()
    print("  To enable it:")
    print()
    print("      pip install 'portfolio-risk-engine[gpu]'")
    print("      # or, with conda:")
    print("      conda env update -f environment.yml --prune")

elif not _CUDA_IS_AVAILABLE:
    print("  Numba is installed but no CUDA GPU was detected.")
    print()
    print("  Verify your setup from Python:")
    print()
    print("      from numba import cuda")
    print("      print(cuda.gpus)   # lists available CUDA devices")
    print()
    print("  Ensure the CUDA toolkit version matches your driver.")

else:
    print("  CUDA GPU detected — running GPU steps.")

    # -----------------------------------------------------------------------
    # Step 10 — GPU simulation
    # -----------------------------------------------------------------------

    _section("Step 10 — GPU simulation  (50 000 paths)")

    _note("""
    MonteCarloGPU launches one CUDA thread per path.  Key GPU optimisations:
      - Cholesky / drift / diffusion constants in shared memory (read once)
      - Per-thread register arrays for the price vector S
      - xoroshiro128p independent per-thread RNG (no atomics)
      - 256 threads/block for good warp occupancy
    """)

    gpu_engine = MonteCarloGPU()  # default: 256 threads/block

    print()
    print("  Warming up Numba JIT compiler (may take 2–5 s on first run) ...")
    gpu_engine.run(portfolio, market_model, corr_matrix, n_paths=1_000, seed=SEED)
    print("  JIT compilation done.")
    print()

    t0 = time.perf_counter()
    gpu_losses = gpu_engine.run(
        portfolio, market_model, corr_matrix, n_paths=N_PATHS, seed=SEED
    )
    gpu_elapsed = time.perf_counter() - t0

    _row("GPU VaR 95%", f"{compute_var(gpu_losses, 0.95):.4f}")
    _row("GPU ES  95%", f"{compute_es(gpu_losses, 0.95):.4f}")
    _row("GPU time   ", f"{gpu_elapsed:.4f} s  ({N_PATHS:,} paths)")

    # -----------------------------------------------------------------------
    # Step 11 — CPU vs GPU benchmark via compare()
    # -----------------------------------------------------------------------

    _section("Step 11 — CPU vs GPU benchmark via compare()")

    _note("""
    application.compare_engines.compare() runs both engines under identical
    conditions and measures wall-clock time.  It returns a ComparisonResult
    TypedDict with cpu_var, gpu_var, cpu_time_s, gpu_time_s, and speedup.

    warmup=False here because we already compiled the kernel above.
    """)

    print()
    print("  Running comparison (100 000 paths) ...")
    print()

    bench = compare(
        portfolio,
        market_model,
        corr_matrix,
        n_paths=100_000,
        confidence=0.95,
        seed=SEED,
        warmup=False,  # kernel already warmed up in Step 10
    )

    print(f"  {'Metric':<22}  {'CPU':>12}  {'GPU':>12}")
    print("  " + "-" * 50)
    print(f"  {'VaR 95%':<22}  {bench['cpu_var']:>12.4f}  {bench['gpu_var']:>12.4f}")
    print(f"  {'ES  95%':<22}  {bench['cpu_es']:>12.4f}  {bench['gpu_es']:>12.4f}")
    print(
        f"  {'Time (s)':<22}  {bench['cpu_time_s']:>12.4f}  {bench['gpu_time_s']:>12.4f}"
    )
    print(f"  {'Speedup':<22}  {'—':>12}  {bench['speedup']:>11.1f}x")

    _note("""
    VaR and ES values should be close (same seed, same model) — small
    differences come from the different RNG implementations (NumPy vs
    xoroshiro128p in CUDA).  The speedup grows with n_paths; try 1 000 000
    to see the GPU pull further ahead.
    """)

# ===========================================================================
# PART 5 — Architecture summary
# ===========================================================================

_banner("PART 5  —  Architecture Summary")

print("""
  Layered (hexagonal-ish) structure
  ----------------------------------

  domain/
    Portfolio              dataclass — S0, weights, initial_value
    MarketModel            dataclass — mu, sigma, dt, n_steps
    compute_cholesky()     validates + decomposes correlation matrix
    compute_var()          α-quantile of losses array
    compute_es()           tail-mean beyond VaR (CVaR)

  infrastructure/simulation/
    SimulationEngine       ABC — defines .run(portfolio, model, corr,
                           n_paths, seed) → np.ndarray
    MonteCarloCPU          NumPy vectorised GBM
    MonteCarloGPU          Numba CUDA — one thread per path

  application/
    run_simulation.run()       one-liner: CPU sim + VaR + ES → TypedDict
    compare_engines.compare()  CPU vs GPU timing → TypedDict

  Swapping engines is ONE line:

      engine = MonteCarloCPU()    # pure NumPy
      engine = MonteCarloGPU()    # Numba CUDA — identical interface

  Running from the command line:

      portfolio-sim               # CLI entry point (cli.py)
      python -m portfolio_risk_engine

  For the full GPU optimisation guide see:  docs/architecture.md
""")

print("=" * _W)
print("  Guided tour complete.  Happy simulating!")
print("=" * _W)
print()
