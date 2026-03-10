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

## Mathematical Model

The baseline implementation assumes asset prices follow a Geometric Brownian Motion (GBM):

dS_t = μS_t dt + σS_t dW_t

This model is used as a simple and extensible starting point for portfolio risk simulation.

## Numerical Method

The engine uses Monte Carlo simulation:

1. simulate market paths
2. revalue the portfolio under each path
3. aggregate results into risk metrics

This approach is flexible, scalable, and well suited for GPU acceleration.

## Architecture

```text
src/portfolio_risk_engine/
├── domain/
│   ├── value_objects/       # Ticker, Currency, Weight, DateRange
│   ├── models/              # Asset, Position, Portfolio, HistoricalPrices
│   └── ports/               # MarketDataProvider (Protocol)
├── application/
│   └── use_cases/           # FetchMarketData
├── infrastructure/
│   └── market_data/         # YahooFinanceMarketDataProvider
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

### Fetch asset metadata

```python
from portfolio_risk_engine.infrastructure.market_data.yahoo_finance_market_data_provider import (
    YahooFinanceMarketDataProvider,
)
from portfolio_risk_engine.domain.value_objects.ticker import Ticker

provider = YahooFinanceMarketDataProvider()
asset = provider.get_asset(Ticker("AAPL"))

print(asset.ticker.value)    # AAPL
print(asset.currency.code)   # USD
print(asset.name)            # Apple Inc.
```

### Fetch historical prices

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
- [ ] Baseline GBM simulation
- [ ] GPU acceleration
- [ ] Multi-asset correlation support
- [ ] VaR and ES analytics
- [ ] Benchmark suite
- [ ] Advanced stochastic models

## License

[BSD 3](./LICENSE)