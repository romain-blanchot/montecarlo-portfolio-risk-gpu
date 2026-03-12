# CLI Reference

The engine provides an interactive menu-driven CLI.

## Launch

```bash
# Via entry point
portfolio-sim

# Via module
python -m portfolio_risk_engine
```

## Menu

```
============================================
  Portfolio Monte Carlo Risk Simulator
============================================

  1. Define portfolio
  2. Fetch market data
  3. Estimate parameters
  4. Run Monte Carlo simulation
  5. Compute risk metrics
  6. Full pipeline (CPU)
  7. Full pipeline (GPU accelerated)    # only if CuPy + CUDA available
  0. Exit
```

## Workflow

The CLI is **stateful** — each step builds on the previous one. The state bar shows current progress:

```
State: Portfolio: [AAPL, MSFT] | Prices: 251 obs | Params: ready | Simulation: 50000 paths | Risk: computed
```

### Option 1: Define Portfolio

Prompts for ticker symbols and allocation weights.

```
Tickers (comma-separated, e.g. AAPL,MSFT,GOOGL): AAPL,MSFT,GOOGL
Weights for ['AAPL', 'MSFT', 'GOOGL'] (comma-separated, must sum to 1): 0.5,0.3,0.2
```

!!! note
    All assets are assigned USD currency. Weights must sum to exactly 1.0.

### Option 2: Fetch Market Data

Prompts for a date range and fetches adjusted close prices from Yahoo Finance.

```
Start date (YYYY-MM-DD): 2023-01-01
End date (YYYY-MM-DD): 2024-01-01
```

Requires an active network connection.

### Option 3: Estimate Parameters

Computes annualized drift vector and covariance matrix from the fetched price history. Displays the annualization factor, per-asset drift, and full covariance matrix.

### Option 4: Run Monte Carlo Simulation

Prompts for simulation parameters:

```
Number of simulations [10000]: 50000
Time horizon in trading days [21]: 21
```

Uses the CPU engine (`CpuMonteCarloEngine`).

### Option 5: Compute Risk Metrics

Computes and displays portfolio-level risk:

```
Mean return:            +0.0083%
Volatility:             0.0512%
VaR 95%:                0.0721%
VaR 99%:                0.1068%
Expected Shortfall 95%: 0.0893%
Expected Shortfall 99%: 0.1198%
```

### Option 6: Full Pipeline (CPU)

Runs steps 1–5 sequentially, prompting for inputs as needed.

### Option 7: Full Pipeline (GPU)

Available only when CuPy and a CUDA GPU are detected. Uses `GpuAcceleratedPipeline` — the entire simulation and risk computation stays on GPU. Only 6 scalar metrics are transferred back to CPU.
