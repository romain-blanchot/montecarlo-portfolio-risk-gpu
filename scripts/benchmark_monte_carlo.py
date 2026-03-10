"""Benchmark CPU vs GPU Monte Carlo engine.

Measures end-to-end simulate() time including GPU->CPU transfer.
GPU timing uses explicit CUDA synchronization for accuracy.
Each scenario runs N_RUNS times after a warm-up; reports median, mean, min, max.

Usage:
    python scripts/benchmark_monte_carlo.py
"""

import statistics
import time

import numpy as np

from portfolio_risk_engine.application.use_cases.compute_portfolio_risk import (
    ComputePortfolioRisk,
)
from portfolio_risk_engine.application.use_cases.run_monte_carlo import RunMonteCarlo
from portfolio_risk_engine.domain.models.asset import Asset
from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.models.portfolio import Portfolio
from portfolio_risk_engine.domain.models.position import Position
from portfolio_risk_engine.domain.value_objects.currency import Currency
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.domain.value_objects.weight import Weight
from portfolio_risk_engine.infrastructure.simulation.cpu_monte_carlo_engine import (
    CpuMonteCarloEngine,
)

try:
    import cupy as cp

    from portfolio_risk_engine.infrastructure.simulation.gpu_monte_carlo_engine import (
        GpuMonteCarloEngine,
    )

    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False

USD = Currency("USD")
SEED = 42
TIME_HORIZON_DAYS = 21
N_RUNS = 5

SCENARIOS = [
    {"label": "Small", "n_assets": 2, "n_simulations": 10_000},
    {"label": "Medium", "n_assets": 10, "n_simulations": 100_000},
    {"label": "Large", "n_assets": 50, "n_simulations": 1_000_000},
]


def _make_synthetic_params(n_assets: int) -> MarketParameters:
    rng = np.random.default_rng(SEED)
    tickers = tuple(Ticker(f"T{i:03d}") for i in range(n_assets))
    drift = tuple(float(x) for x in rng.normal(0.08, 0.05, n_assets))

    # Random positive-definite covariance matrix
    A = rng.normal(0, 0.1, (n_assets, n_assets))
    cov = A @ A.T + np.eye(n_assets) * 0.02
    covariance_matrix = tuple(tuple(float(x) for x in row) for row in cov)

    return MarketParameters(
        tickers=tickers,
        drift_vector=drift,
        covariance_matrix=covariance_matrix,
        annualization_factor=252,
    )


def _make_portfolio(params: MarketParameters) -> Portfolio:
    n = len(params.tickers)
    w = 1.0 / n
    positions = tuple(
        Position(asset=Asset(ticker=t, currency=USD), weight=Weight(w))
        for t in params.tickers
    )
    return Portfolio(positions=positions)


def _make_initial_prices(n_assets: int) -> tuple[float, ...]:
    rng = np.random.default_rng(SEED + 1)
    return tuple(float(x) for x in rng.uniform(50, 500, n_assets))


def _timed_simulate(
    use_case: RunMonteCarlo,
    params: MarketParameters,
    initial_prices: tuple[float, ...],
    n_simulations: int,
    is_gpu: bool,
) -> tuple[float, object]:
    """Run one simulation and return (elapsed_seconds, result)."""
    if is_gpu:
        cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()
    result = use_case.execute(
        market_params=params,
        initial_prices=initial_prices,
        num_simulations=n_simulations,
        time_horizon_days=TIME_HORIZON_DAYS,
    )
    if is_gpu:
        cp.cuda.Stream.null.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed, result


def _run_benchmark(
    engine: object,
    backend_name: str,
    params: MarketParameters,
    initial_prices: tuple[float, ...],
    portfolio: Portfolio,
    n_simulations: int,
    is_gpu: bool,
) -> dict:
    use_case = RunMonteCarlo(engine=engine)  # type: ignore[arg-type]

    # Warm-up run (not counted)
    _, _ = _timed_simulate(use_case, params, initial_prices, n_simulations, is_gpu)

    # Timed runs
    timings = []
    last_result = None
    for _ in range(N_RUNS):
        elapsed, result = _timed_simulate(
            use_case, params, initial_prices, n_simulations, is_gpu
        )
        timings.append(elapsed)
        last_result = result

    risk = ComputePortfolioRisk.execute(portfolio, last_result)  # type: ignore[arg-type]

    return {
        "backend": backend_name,
        "n_assets": len(params.tickers),
        "n_simulations": n_simulations,
        "timings": timings,
        "median": statistics.median(timings),
        "mean": statistics.mean(timings),
        "min": min(timings),
        "max": max(timings),
        "mean_return": risk.mean_return,
        "volatility": risk.volatility,
        "var_95": risk.var_95,
        "var_99": risk.var_99,
        "es_95": risk.es_95,
        "es_99": risk.es_99,
    }


def _print_result(r: dict) -> None:
    print(f"  Backend:      {r['backend']}")
    print(f"  Assets:       {r['n_assets']}")
    print(f"  Simulations:  {r['n_simulations']:,}")
    print(f"  Runs:         {N_RUNS} (after 1 warm-up)")
    print(f"  Median:       {r['median']:.4f} s")
    print(f"  Mean:         {r['mean']:.4f} s")
    print(f"  Min:          {r['min']:.4f} s")
    print(f"  Max:          {r['max']:.4f} s")
    print(f"  Mean return:  {r['mean_return']:+.4%}")
    print(f"  Volatility:   {r['volatility']:.4%}")
    print(f"  VaR 95%:      {r['var_95']:.4%}")
    print(f"  VaR 99%:      {r['var_99']:.4%}")
    print(f"  ES 95%:       {r['es_95']:.4%}")
    print(f"  ES 99%:       {r['es_99']:.4%}")


def main() -> None:
    print("=" * 60)
    print("  Monte Carlo Benchmark: CPU vs GPU")
    print(f"  {N_RUNS} runs per scenario + 1 warm-up")
    print("  Timing: end-to-end simulate() incl. GPU->CPU transfer")
    if _GPU_AVAILABLE:
        print("  GPU sync: cp.cuda.Stream.null.synchronize()")
    print("=" * 60)

    if not _GPU_AVAILABLE:
        print("\nWARNING: GPU not available. Running CPU-only benchmark.\n")

    for scenario in SCENARIOS:
        label = scenario["label"]
        n_assets = scenario["n_assets"]
        n_sims = scenario["n_simulations"]

        print(f"\n{'─' * 60}")
        print(f"  Scenario: {label} ({n_assets} assets, {n_sims:,} simulations)")
        print(f"{'─' * 60}")

        params = _make_synthetic_params(n_assets)
        initial_prices = _make_initial_prices(n_assets)
        portfolio = _make_portfolio(params)

        # CPU benchmark
        cpu_result = _run_benchmark(
            engine=CpuMonteCarloEngine(seed=SEED),
            backend_name="CPU (NumPy)",
            params=params,
            initial_prices=initial_prices,
            portfolio=portfolio,
            n_simulations=n_sims,
            is_gpu=False,
        )
        print("\n  [CPU]")
        _print_result(cpu_result)

        # GPU benchmark
        if _GPU_AVAILABLE:
            gpu_result = _run_benchmark(
                engine=GpuMonteCarloEngine(seed=SEED),
                backend_name="GPU (CuPy)",
                params=params,
                initial_prices=initial_prices,
                portfolio=portfolio,
                n_simulations=n_sims,
                is_gpu=True,
            )
            print("\n  [GPU]")
            _print_result(gpu_result)

            speedup = cpu_result["median"] / gpu_result["median"]
            print(
                f"\n  Speedup (median): {speedup:.2f}x"
                f" {'(GPU faster)' if speedup > 1 else '(CPU faster)'}"
            )
        else:
            print("\n  [GPU] Skipped (not available)")

    print(f"\n{'=' * 60}")
    print("  Note: CPU and GPU use different RNGs, so statistical")
    print("  results differ slightly. This is expected.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
