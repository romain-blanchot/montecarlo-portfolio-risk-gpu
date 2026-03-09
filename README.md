# Monte Carlo Portfolio Risk Engine (GPU Accelerated)

| | |
| --- | --- |
| CI/CD | [![CI - Test](https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu/actions/workflows/ci.yml) [![CD - Build](https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu/actions/workflows/cd.yml/badge.svg)](https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu/actions/workflows/cd.yml) |
| Meta | [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) [![code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/ambv/black) [![License - MIT](https://img.shields.io/badge/license-MIT-9400d3.svg)](https://spdx.org/licenses/) [![GitHub Sponsors](https://img.shields.io/github/sponsors/ofek?logo=GitHub%20Sponsors&style=social)](https://github.com/sponsors/ofek) |

CI CD ci cd sonar 
Docs Dev docs 
Meta Hatch Ruff Mypy  BSD3 License NerionSoft GitHub Sponsors

Lint
Tests
Security
Docs
Build
Docker
Release
CI/CD

Coverage

Python version

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](#ci-cd)
[![Python](https://img.shields.io/badge/Python-3.13%2B-blue)](#installation)
[![CUDA](https://img.shields.io/badge/CUDA-supported-green)](#benchmarks)
[![License](https://img.shields.io/badge/License-BSD_3-yellow.svg)](#license)

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
src/  
└── portfolio_simulator/  
    ├── market/  
    ├── portfolio/  
    ├── simulation/  
    │   ├── monte_carlo_cpu.py  
    │   └── monte_carlo_gpu.py  
    ├── risk/  
    └── analytics/
```

## Installation

```bash
git clone git@github.com:romain-blanchot/montecarlo-portfolio-risk-gpu.git
cd montecarlo-portfolio-risk-gpu  
conda env create -f environment.yml 
conda activate portfolio-risk-engine
```

## Usage

```python
print("Hello World!")
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

Run the test suite with:
```bash
pytest
```
With coverage:
```bash
pytest --cov=src --cov-report=term-missing
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

- [ ] Baseline GBM simulation
- [ ] GPU acceleration
- [ ] Multi-asset correlation support
- [ ] VaR and ES analytics
- [ ] Benchmark suite
- [ ] Advanced stochastic models

## License

[BSD 3](./LICENSE)