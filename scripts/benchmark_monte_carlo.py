"""Benchmark CPU vs GPU Monte Carlo engine.

Compares three pipelines end-to-end (simulate + risk):
  1. CPU:              NumPy simulate -> tuple -> NumPy risk
  2. GPU (via domain): CuPy simulate -> tuple -> NumPy risk
  3. GPU accelerated:  CuPy simulate + risk, zero tuple allocation

GPU timing uses explicit CUDA synchronization.
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

    from portfolio_risk_engine.infrastructure.simulation.gpu_accelerated_pipeline import (
        GpuAcceleratedPipeline,
    )
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
    return Portfolio(
        positions=tuple(
            Position(asset=Asset(ticker=t, currency=USD), weight=Weight(w))
            for t in params.tickers
        )
    )


def _make_initial_prices(n_assets: int) -> tuple[float, ...]:
    rng = np.random.default_rng(SEED + 1)
    return tuple(float(x) for x in rng.uniform(50, 500, n_assets))


# ── Timing helpers ──────────────────────────────────────────


def _time_cpu_pipeline(
    params: MarketParameters,
    initial_prices: tuple[float, ...],
    portfolio: Portfolio,
    n_sims: int,
) -> tuple[float, dict]:
    use_case = RunMonteCarlo(engine=CpuMonteCarloEngine(seed=SEED))
    start = time.perf_counter()
    sim = use_case.execute(
        market_params=params,
        initial_prices=initial_prices,
        num_simulations=n_sims,
        time_horizon_days=TIME_HORIZON_DAYS,
    )
    risk = ComputePortfolioRisk.execute(portfolio, sim)
    elapsed = time.perf_counter() - start
    return elapsed, _metrics_dict(risk)


def _time_gpu_domain_pipeline(
    params: MarketParameters,
    initial_prices: tuple[float, ...],
    portfolio: Portfolio,
    n_sims: int,
) -> tuple[float, dict]:
    use_case = RunMonteCarlo(engine=GpuMonteCarloEngine(seed=SEED))
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    sim = use_case.execute(
        market_params=params,
        initial_prices=initial_prices,
        num_simulations=n_sims,
        time_horizon_days=TIME_HORIZON_DAYS,
    )
    risk = ComputePortfolioRisk.execute(portfolio, sim)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, _metrics_dict(risk)


def _time_gpu_accelerated(
    params: MarketParameters,
    initial_prices: tuple[float, ...],
    weights: tuple[float, ...],
    n_sims: int,
) -> tuple[float, dict]:
    pipeline = GpuAcceleratedPipeline(seed=SEED)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    risk = pipeline.run(
        market_params=params,
        initial_prices=initial_prices,
        weights=weights,
        num_simulations=n_sims,
        time_horizon_days=TIME_HORIZON_DAYS,
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, _metrics_dict(risk)


def _metrics_dict(risk: object) -> dict:
    return {
        "mean_return": risk.mean_return,
        "volatility": risk.volatility,
        "var_95": risk.var_95,
        "var_99": risk.var_99,
        "es_95": risk.es_95,
        "es_99": risk.es_99,
    }


# ── Benchmark runner ────────────────────────────────────────


def _bench(name: str, fn, *args) -> dict:
    """Warm-up + N_RUNS timed executions."""
    # Warm-up
    fn(*args)

    timings = []
    last_metrics = None
    for _ in range(N_RUNS):
        elapsed, metrics = fn(*args)
        timings.append(elapsed)
        last_metrics = metrics

    return {
        "name": name,
        "timings": timings,
        "median": statistics.median(timings),
        "mean": statistics.mean(timings),
        "min": min(timings),
        "max": max(timings),
        **last_metrics,
    }


def _print_result(r: dict) -> None:
    print(f"    Median: {r['median']:.4f} s  |  Mean: {r['mean']:.4f} s")
    print(f"    Min:    {r['min']:.4f} s  |  Max:  {r['max']:.4f} s")
    print(f"    VaR95={r['var_95']:.4%}  VaR99={r['var_99']:.4%}")
    print(f"    ES95={r['es_95']:.4%}   ES99={r['es_99']:.4%}")


def main() -> None:
    print("=" * 64)
    print("  Monte Carlo Benchmark: full pipeline (simulate + risk)")
    print(f"  {N_RUNS} runs + 1 warm-up per backend per scenario")
    if _GPU_AVAILABLE:
        print("  GPU sync: cp.cuda.Stream.null.synchronize()")
    else:
        print("  WARNING: GPU not available — CPU only")
    print("=" * 64)

    for scenario in SCENARIOS:
        label = scenario["label"]
        n_assets = scenario["n_assets"]
        n_sims = scenario["n_simulations"]

        print(f"\n{'─' * 64}")
        print(f"  {label}: {n_assets} assets x {n_sims:,} simulations")
        print(f"{'─' * 64}")

        params = _make_synthetic_params(n_assets)
        initial_prices = _make_initial_prices(n_assets)
        portfolio = _make_portfolio(params)
        weights = tuple(1.0 / n_assets for _ in range(n_assets))

        # 1. CPU
        cpu = _bench(
            "CPU (NumPy)", _time_cpu_pipeline, params, initial_prices, portfolio, n_sims
        )
        print("\n  [CPU]")
        _print_result(cpu)

        if _GPU_AVAILABLE:
            # 2. GPU via domain (with tuple conversion)
            gpu_domain = _bench(
                "GPU via domain",
                _time_gpu_domain_pipeline,
                params,
                initial_prices,
                portfolio,
                n_sims,
            )
            print("\n  [GPU via domain] (simulate -> tuples -> risk)")
            _print_result(gpu_domain)

            # 3. GPU accelerated (zero tuple allocation)
            gpu_accel = _bench(
                "GPU accelerated",
                _time_gpu_accelerated,
                params,
                initial_prices,
                weights,
                n_sims,
            )
            print("\n  [GPU accelerated] (all-GPU, no tuples)")
            _print_result(gpu_accel)

            # Speedup table
            print("\n  Speedup vs CPU (median):")
            s1 = cpu["median"] / gpu_domain["median"]
            s2 = cpu["median"] / gpu_accel["median"]
            print(f"    GPU via domain:    {s1:.2f}x")
            print(f"    GPU accelerated:   {s2:.2f}x")

    print(f"\n{'=' * 64}")
    print("  Note: RNGs differ across backends — small statistical")
    print("  differences are expected.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
