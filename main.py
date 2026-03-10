"""
Monte Carlo Portfolio Risk Engine
==================================
Edit the CONFIGURATION section below, then run:

    python main.py
"""

from __future__ import annotations

import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except AttributeError:
    pass

import numpy as np

from portfolio_risk_engine.application.compare_engines import compare
from portfolio_risk_engine.application.run_simulation import run
from portfolio_risk_engine.domain.market_model import MarketModel
from portfolio_risk_engine.domain.portfolio import Portfolio
from portfolio_risk_engine.domain.var import compute_var
from portfolio_risk_engine.domain.expected_shortfall import compute_es
from portfolio_risk_engine.infrastructure.simulation.monte_carlo_cpu import (
    MonteCarloCPU,
)
from portfolio_risk_engine.infrastructure.simulation.monte_carlo_gpu import (
    MonteCarloGPU,
    _CUDA_IS_AVAILABLE,
    _NUMBA_AVAILABLE,
)

# ===========================================================================
# CONFIGURATION — 8-asset multi-sector portfolio (realistic benchmark)
# ===========================================================================

# Asset names — 8 assets across 5 sectors (tech, finance, energy, healthcare, safe haven)
ASSET_NAMES = ["AAPL", "MSFT", "JPM", "GS", "XOM", "JNJ", "GLD", "BND"]

# Initial prices (USD — approximate real values)
S0 = np.array([195.0, 415.0, 205.0, 510.0, 118.0, 155.0, 215.0, 73.0])

# Portfolio weights — diversified, sums to 1.0
WEIGHTS = np.array([0.18, 0.17, 0.12, 0.10, 0.10, 0.10, 0.13, 0.10])

# --- Market parameters (annualised, based on 2015-2024 historical estimates) ---

# Expected annual return per asset
MU = np.array([
    0.22,   # AAPL  — Big Tech, high growth
    0.25,   # MSFT  — Big Tech, high growth
    0.12,   # JPM   — Finance, moderate
    0.10,   # GS    — Finance, moderate
    0.07,   # XOM   — Energy, low growth
    0.08,   # JNJ   — Healthcare, defensive
    0.05,   # GLD   — Gold, inflation hedge
    0.03,   # BND   — Bonds, safe haven
])

# Annual volatility per asset
SIGMA = np.array([
    0.28,   # AAPL
    0.26,   # MSFT
    0.24,   # JPM
    0.27,   # GS
    0.26,   # XOM
    0.15,   # JNJ  — low vol, defensive
    0.15,   # GLD  — low vol
    0.06,   # BND  — very low vol
])

# Simulation horizon
N_STEPS = 252       # 1 trading year
DT = 1.0 / 252

# --- Correlation matrix (8×8) ---
# Tech stocks highly correlated, bonds/gold negatively correlated to equities
CORR = np.array([
    #  AAPL   MSFT   JPM    GS     XOM    JNJ    GLD    BND
    [ 1.00,  0.78,  0.48,  0.42,  0.22,  0.28, -0.05, -0.22],  # AAPL
    [ 0.78,  1.00,  0.45,  0.40,  0.18,  0.30, -0.08, -0.25],  # MSFT
    [ 0.48,  0.45,  1.00,  0.72,  0.35,  0.32,  0.05, -0.15],  # JPM
    [ 0.42,  0.40,  0.72,  1.00,  0.30,  0.28,  0.08, -0.18],  # GS
    [ 0.22,  0.18,  0.35,  0.30,  1.00,  0.25,  0.35, -0.10],  # XOM
    [ 0.28,  0.30,  0.32,  0.28,  0.25,  1.00,  0.15,  0.10],  # JNJ
    [-0.05, -0.08,  0.05,  0.08,  0.35,  0.15,  1.00,  0.35],  # GLD
    [-0.22, -0.25, -0.15, -0.18, -0.10,  0.10,  0.35,  1.00],  # BND
])

# --- Stress scenario : crise financière systémique ---
# Toutes les corrélations equity montent (flight to quality),
# gold/bonds décrochent des actions encore plus fortement
CORR_STRESS = np.array([
    #  AAPL   MSFT   JPM    GS     XOM    JNJ    GLD    BND
    [ 1.00,  0.95,  0.82,  0.80,  0.70,  0.65, -0.30, -0.55],  # AAPL
    [ 0.95,  1.00,  0.80,  0.78,  0.68,  0.63, -0.32, -0.57],  # MSFT
    [ 0.82,  0.80,  1.00,  0.92,  0.72,  0.68, -0.10, -0.40],  # JPM
    [ 0.80,  0.78,  0.92,  1.00,  0.70,  0.65, -0.08, -0.42],  # GS
    [ 0.70,  0.68,  0.72,  0.70,  1.00,  0.60,  0.20, -0.30],  # XOM
    [ 0.65,  0.63,  0.68,  0.65,  0.60,  1.00,  0.05, -0.10],  # JNJ
    [-0.30, -0.32, -0.10, -0.08,  0.20,  0.05,  1.00,  0.55],  # GLD
    [-0.55, -0.57, -0.40, -0.42, -0.30, -0.10,  0.55,  1.00],  # BND
])

# --- Simulation settings ---
N_PATHS = 1_000_000     # 1M paths — baseline
SEED = 42

# --- Confidence levels ---
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

# --- Benchmark : liste de paths pour courbe de speedup CPU vs GPU ---
# Permet de tracer speedup = f(N_paths) — le GPU écrase le CPU à partir de ~50k
BENCHMARK_PATHS = 200_000

# ===========================================================================
# END OF CONFIGURATION
# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

W = 60


def _header(title: str) -> None:
    print()
    print("=" * W)
    print(f"  {title}")
    print("=" * W)


def _sep() -> None:
    print("  " + "-" * (W - 2))


def _row(label: str, value: str, width: int = 30) -> None:
    print(f"  {label:<{width}} {value}")


# ---------------------------------------------------------------------------
# Build domain objects from config
# ---------------------------------------------------------------------------

portfolio = Portfolio(S0=S0, weights=WEIGHTS)
market_model = MarketModel(mu=MU, sigma=SIGMA, dt=DT, n_steps=N_STEPS)
engine = MonteCarloCPU()

# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------

_header("Portfolio")

assets_str = " / ".join(ASSET_NAMES)
prices_str = " / ".join(f"{p:.2f}" for p in S0)
weights_str = " / ".join(f"{w * 100:.1f}%" for w in WEIGHTS)
mu_str = " / ".join(f"{v * 100:.1f}%" for v in MU)
sigma_str = " / ".join(f"{v * 100:.1f}%" for v in SIGMA)

print()
_row("Assets", assets_str)
_row("Initial prices", prices_str)
_row("Weights", weights_str)
_row("Initial portfolio value", f"${portfolio.initial_value:.2f}")
print()
_row("Annual drift  μ", mu_str)
_row("Annual volatility  σ", sigma_str)
_row("Horizon", f"{N_STEPS} steps  (dt = {DT:.6f} yr)")
print()
_row("Paths", f"{N_PATHS:,}")
_row("Seed", str(SEED))

# ---------------------------------------------------------------------------
# Base simulation
# ---------------------------------------------------------------------------

_header("Risk Metrics — Base Scenario (CPU)")

t0 = time.perf_counter()
result = run(
    portfolio=portfolio,
    market_model=market_model,
    corr_matrix=CORR,
    n_paths=N_PATHS,
    confidence=0.95,
    seed=SEED,
)
cpu_time = time.perf_counter() - t0

losses = result["losses"]

print()
_row("Simulation time", f"{cpu_time:.3f} s")
print()
print(f"  {'Confidence':>12}  {'VaR':>10}  {'ES (CVaR)':>11}  {'Tail paths':>12}")
_sep()
for conf in CONFIDENCE_LEVELS:
    v = compute_var(losses, conf)
    e = compute_es(losses, conf)
    tail_pct = 100.0 * float(np.mean(losses > v))
    print(f"  {conf * 100:>10.0f}%  {v:>10.4f}  {e:>11.4f}  {tail_pct:>10.1f}%")

print()
_sep()
_row("Mean loss  (negative = avg gain)", f"{losses.mean():+.4f}")
_row("Std deviation", f"{losses.std():.4f}")
_row("Best outcome  (min loss)", f"{losses.min():+.4f}")
_row("Worst outcome (max loss)", f"{losses.max():+.4f}")
_row("% paths with a gain  (loss < 0)", f"{100 * float(np.mean(losses < 0)):.1f}%")

# ---------------------------------------------------------------------------
# Allocation strategies
# ---------------------------------------------------------------------------

_header("Allocation Strategies")

n = len(ASSET_NAMES)
strategies: dict[str, Portfolio] = {
    "Your portfolio": portfolio,
    f"Concentrated (first asset 80%)": Portfolio(
        S0=S0,
        weights=np.array([0.80] + [0.20 / (n - 1)] * (n - 1)),
    ),
    f"Equal-weight (1/{n} each)": Portfolio(
        S0=S0,
        weights=np.full(n, 1.0 / n),
    ),
}

print()
print(f"  {'Strategy':<30}  {'V0':>8}  {'VaR 95%':>9}  {'ES 95%':>9}  {'σ losses':>9}")
_sep()
for name, p in strategies.items():
    ls = engine.run(p, market_model, CORR, N_PATHS, seed=SEED)
    print(
        f"  {name:<30}  "
        f"{p.initial_value:>8.2f}  "
        f"{compute_var(ls, 0.95):>9.4f}  "
        f"{compute_es(ls, 0.95):>9.4f}  "
        f"{ls.std():>9.4f}"
    )

# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------

_header("Stress Test — Crisis Correlations")

result_stress = run(portfolio, market_model, CORR_STRESS, N_PATHS, seed=SEED)

pct_var = 100 * (result_stress["var"] - result["var"]) / abs(result["var"])
pct_es = 100 * (result_stress["es"] - result["es"]) / abs(result["es"])

print()
print(f"  {'Scenario':<22}  {'VaR 95%':>9}  {'ES 95%':>9}  {'Std dev':>9}")
_sep()
print(
    f"  {'Base (normal corr)':<22}  "
    f"{result['var']:>9.4f}  "
    f"{result['es']:>9.4f}  "
    f"{result['losses'].std():>9.4f}"
)
print(
    f"  {'Stress (crisis corr)':<22}  "
    f"{result_stress['var']:>9.4f}  "
    f"{result_stress['es']:>9.4f}  "
    f"{result_stress['losses'].std():>9.4f}"
)
print()
_row("VaR change under stress", f"{pct_var:+.1f}%")
_row("ES  change under stress", f"{pct_es:+.1f}%")

# ---------------------------------------------------------------------------
# GPU benchmark
# ---------------------------------------------------------------------------

_header("GPU Benchmark (Numba CUDA)")

print()

if not _NUMBA_AVAILABLE:
    print("  Numba not installed — GPU unavailable.")
    print("  Install with:  pip install 'portfolio-risk-engine[gpu]'")

elif not _CUDA_IS_AVAILABLE:
    print("  Numba installed but no CUDA GPU detected.")
    print('  Check:  python -c "from numba import cuda; print(cuda.gpus)"')

else:
    gpu_engine = MonteCarloGPU()

    print(f"  Warming up JIT compiler (first run only) ...")
    gpu_engine.run(portfolio, market_model, CORR, n_paths=1_000, seed=SEED)
    print(f"  Ready. Running benchmark with {BENCHMARK_PATHS:,} paths ...\n")

    bench = compare(
        portfolio,
        market_model,
        CORR,
        n_paths=BENCHMARK_PATHS,
        confidence=0.95,
        seed=SEED,
        warmup=False,
    )

    print(f"  {'Metric':<22}  {'CPU':>12}  {'GPU':>12}")
    _sep()
    print(f"  {'VaR 95%':<22}  {bench['cpu_var']:>12.4f}  {bench['gpu_var']:>12.4f}")
    print(f"  {'ES  95%':<22}  {bench['cpu_es']:>12.4f}  {bench['gpu_es']:>12.4f}")
    print(
        f"  {'Time (s)':<22}  {bench['cpu_time_s']:>12.4f}  {bench['gpu_time_s']:>12.4f}"
    )
    print(f"  {'Speedup':<22}  {'—':>12}  {bench['speedup']:>11.1f}x")

print()
print("=" * W)
print()
