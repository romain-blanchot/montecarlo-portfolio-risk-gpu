# Monte Carlo Portfolio Risk Engine (GPU Accelerated)

| | |
| --- | --- |
| CI/CD | [![CI](https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu/actions/workflows/ci.yml) [![CD](https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu/actions/workflows/cd.yml/badge.svg)](https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu/actions/workflows/cd.yml) [![Release](https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu/actions/workflows/release.yml/badge.svg)](https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu/actions/workflows/release.yml) |
| Docs | [![Docs](https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu/actions/workflows/docs.yml/badge.svg)](https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu/actions/workflows/docs.yml) |
| Quality | [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=romain-blanchot_montecarlo-portfolio-risk-gpu&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=romain-blanchot_montecarlo-portfolio-risk-gpu) [![Coverage](https://sonarcloud.io/api/project_badges/measure?project=romain-blanchot_montecarlo-portfolio-risk-gpu&metric=coverage)](https://sonarcloud.io/summary/new_code?id=romain-blanchot_montecarlo-portfolio-risk-gpu) [![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=romain-blanchot_montecarlo-portfolio-risk-gpu&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=romain-blanchot_montecarlo-portfolio-risk-gpu) [![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=romain-blanchot_montecarlo-portfolio-risk-gpu&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=romain-blanchot_montecarlo-portfolio-risk-gpu) [![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=romain-blanchot_montecarlo-portfolio-risk-gpu&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=romain-blanchot_montecarlo-portfolio-risk-gpu) |
| Metrics | [![Bugs](https://sonarcloud.io/api/project_badges/measure?project=romain-blanchot_montecarlo-portfolio-risk-gpu&metric=bugs)](https://sonarcloud.io/summary/new_code?id=romain-blanchot_montecarlo-portfolio-risk-gpu) [![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=romain-blanchot_montecarlo-portfolio-risk-gpu&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=romain-blanchot_montecarlo-portfolio-risk-gpu) [![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=romain-blanchot_montecarlo-portfolio-risk-gpu&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=romain-blanchot_montecarlo-portfolio-risk-gpu) [![Technical Debt](https://sonarcloud.io/api/project_badges/measure?project=romain-blanchot_montecarlo-portfolio-risk-gpu&metric=sqale_index)](https://sonarcloud.io/summary/new_code?id=romain-blanchot_montecarlo-portfolio-risk-gpu) [![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=romain-blanchot_montecarlo-portfolio-risk-gpu&metric=duplicated_lines_density)](https://sonarcloud.io/summary/new_code?id=romain-blanchot_montecarlo-portfolio-risk-gpu) |
| Meta | [![Hatch project](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pypa/hatch/master/docs/assets/badge/v0.json)](https://github.com/pypa/hatch) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white)](https://www.python.org/) [![CUDA](https://img.shields.io/badge/CUDA-supported-76B900?logo=nvidia&logoColor=white)](#benchmarks) [![License - BSD 3](https://img.shields.io/badge/License-BSD_3-yellow.svg)](./LICENSE) |

GPU-accelerated Monte Carlo engine for portfolio risk simulation and market risk analytics.

## Overview


This project implements a GPU-accelerated Monte Carlo engine to simulate multi-asset portfolio dynamics and estimate risk metrics such as Value at Risk (VaR) and Expected Shortfall.


It is designed for:
- quantitative finance experimentation
- portfolio risk analysis
- CPU vs GPU performance comparison
- extension toward more advanced market models

## Financial Context

Portfolio risk cannot be assessed from a single forecast. This engine simulates many market scenarios to estimate the distribution of future portfolio values and derive downside risk measures.

Typical use cases:
- Value-at-Risk
- Expected Shortfall
- scenario analysis
- stress testing
- distribution analysis

## Simulation Models

Three stochastic models are available, selectable via the `"model"` field in JSON input:

| Model | Key | Description |
|---|---|---|
| **Geometric Brownian Motion** | `gbm` | Baseline log-normal model: `dS = μS dt + σS dW` |
| **Student-t GBM** | `student_t` | GBM with multivariate Student-t innovations for fat tails. Degrees of freedom estimated via method of moments |
| **Heston Stochastic Volatility** | `heston` | Mean-reverting stochastic variance: `dS = μS dt + √v S dW_S`, `dv = κ(θ−v)dt + ξ√v dW_v` with leverage correlation ρ. Euler-Maruyama discretization with full truncation |

All models support multi-asset portfolios with inter-asset correlation and automatic GPU acceleration when CUDA is available.

## Risk Metrics

- **VaR** (95%, 99%) — Value at Risk (loss-positive convention)
- **ES** (95%, 99%) — Expected Shortfall / CVaR
- **Skewness** and **Excess Kurtosis** — tail shape diagnostics
- **Prob(Loss)** — probability of negative portfolio return
- Per-asset terminal price distributions with percentiles (P5, P50, P95)

## Architecture

Hexagonal (ports-and-adapters) structure:

```text
src/portfolio_risk_engine/
├── domain/
│   ├── value_objects/       # Ticker, Currency, Weight, DateRange
│   ├── models/              # Asset, Portfolio, MarketParameters, MultivariateGBM,
│   │                        # StudentTGBM, HestonModel, SimulationResult, RiskMetrics
│   ├── ports/               # MarketDataProvider, MonteCarloEngine (Protocols)
│   └── services/            # Cholesky decomposition
├── application/
│   └── use_cases/           # FetchMarketData, ComputeLogReturns, EstimateMarketParameters,
│                            # EstimateStudentTParameters, EstimateHestonParameters,
│                            # RunMonteCarlo, ComputePortfolioRisk, SimulatePortfolio
├── infrastructure/
│   ├── market_data/         # YahooFinanceMarketDataProvider
│   ├── simulation/          # CpuMonteCarloEngine, GpuMonteCarloEngine,
│   │                        # CpuStudentTEngine, GpuStudentTEngine,
│   │                        # CpuHestonEngine, GpuHestonEngine
│   └── rendering/           # Terminal scenario renderer (histograms, risk dashboard)
├── cli.py
└── __main__.py
```

## Installation

```bash
git clone git@github.com:romain-blanchot/montecarlo-portfolio-risk-gpu.git
cd montecarlo-portfolio-risk-gpu
conda env create -f environment.yml
conda activate portfolio-risk-engine
```

```bash
pre-commit install
```

## Usage

### JSON CLI (recommended)

Pass a JSON config directly to `portfolio-sim` for quick scenario analysis:

```bash
# GBM (default model)
portfolio-sim '{"assets":[{"ticker":"AAPL","weight":0.6},{"ticker":"MSFT","weight":0.4}],"start_date":"2024-01-01","end_date":"2025-01-01"}'

# Student-t GBM (fat tails)
portfolio-sim '{"assets":[{"ticker":"AAPL","weight":0.6},{"ticker":"MSFT","weight":0.4}],"start_date":"2024-01-01","end_date":"2025-01-01","model":"student_t","num_simulations":100000}'

# Heston stochastic volatility
portfolio-sim '{"assets":[{"ticker":"AAPL","weight":0.6},{"ticker":"MSFT","weight":0.4}],"start_date":"2024-01-01","end_date":"2025-01-01","model":"heston","num_simulations":100000}'

# Custom horizon (63 trading days ≈ 3 months)
portfolio-sim '{"assets":[{"ticker":"INTC","weight":1}],"start_date":"2024-01-01","end_date":"2025-01-01","model":"heston","num_simulations":500000,"time_horizon_days":63}'
```

JSON input can also be piped via stdin:

```bash
echo '{"assets":[{"ticker":"AAPL","weight":1}],"start_date":"2024-01-01","end_date":"2025-01-01"}' | portfolio-sim
```

#### JSON fields

| Field | Required | Default | Description |
|---|---|---|---|
| `assets` | yes | — | Array of `{"ticker": "...", "weight": ...}` |
| `start_date` | yes | — | Historical data start (YYYY-MM-DD) |
| `end_date` | yes | — | Historical data end (YYYY-MM-DD) |
| `model` | no | `"gbm"` | `"gbm"`, `"student_t"`, or `"heston"` |
| `num_simulations` | no | `10000` | Number of Monte Carlo paths |
| `time_horizon_days` | no | `21` | Simulation horizon in trading days |

GPU acceleration is automatic when CUDA + CuPy are available.

### Interactive CLI

```bash
portfolio-sim   # launches interactive menu
```

### Python API

```python
from datetime import date
from portfolio_risk_engine.application.use_cases.fetch_market_data import FetchMarketData
from portfolio_risk_engine.infrastructure.market_data.yahoo_finance_market_data_provider import (
    YahooFinanceMarketDataProvider,
)
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.domain.value_objects.date_range import DateRange

provider = YahooFinanceMarketDataProvider()
use_case = FetchMarketData(provider)

result = use_case.execute(
    tickers=(Ticker("AAPL"), Ticker("MSFT")),
    date_range=DateRange(start=date(2024, 1, 1), end=date(2024, 3, 1)),
)

print(len(result.dates))                       # 40 (trading days)
print(result.prices_by_ticker[Ticker("AAPL")][:3])  # (183.73, 182.35, 180.03)
```

## Benchmarks

The project includes benchmark scenarios to compare:
- CPU vs GPU runtime
- scalability with number of paths
- scalability with number of assets
- numerical consistency across backends

## Experiments

The `notebooks/` directory contains research and validation work, including:
- volatility sensitivity
- correlation studies
- CPU/GPU comparisons
- reproducibility checks

## Testing

Run the unit test suite:
```bash
pytest
```

With coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

Run integration tests (requires network):
```bash
pytest -m integration
```
## CI/CD

The CI/CD pipeline covers:
- linting
- formatting
- type checking
- tests
- coverage
- package build
- documentation and release workflows

## Roadmap

- [x] Domain model (Asset, Portfolio, HistoricalPrices)
- [x] Market data provider (Yahoo Finance)
- [x] GBM simulation (CPU + GPU)
- [x] Multi-asset correlation (Cholesky decomposition)
- [x] VaR and ES analytics
- [x] Student-t GBM (fat tails, CPU + GPU)
- [x] Heston stochastic volatility (CPU + GPU)
- [x] JSON CLI for quick scenario testing
- [x] Terminal renderer (histograms, risk dashboard, tail metrics)
- [x] GPU auto-detection (CuPy/CUDA)
- [ ] Benchmark suite
- [ ] Regime-switching models
- [ ] Jump-diffusion (Merton)
- [ ] Short selling costs / funding rates

## License

[BSD 3](./LICENSE)