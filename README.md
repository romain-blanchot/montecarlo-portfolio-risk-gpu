# Monte Carlo Portfolio Risk Engine (GPU Accelerated)

This project implements a GPU-accelerated Monte Carlo engine to simulate multi-asset portfolio dynamics and estimate risk metrics such as Value at Risk (VaR) and Expected Shortfall.


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

-----

## Overview

This project explores the design of a Monte Carlo risk engine capable of simulating portfolio behavior under stochastic market scenarios.

It combines:
- quantitative finance abstractions,
- GPU-oriented computation,
- clean software architecture,
- reproducible development workflows,
- and production-oriented packaging.

## Why this project exists

Quantitative finance systems often become difficult to understand because several layers of abstraction are mixed together:
market assumptions, simulation logic, instrument valuation, portfolio aggregation, and infrastructure.

This repository aims to make those layers explicit.

## Project goals

### Financial goals
- simulate market scenarios,
- analyze portfolio behavior,
- estimate risk-related outputs from Monte Carlo paths.

### Engineering goals
- accelerate compute-intensive workloads with GPU support,
- isolate domain logic from infrastructure,
- provide a reproducible development environment,
- support lean production deployment,
- maintain strong code quality standards.

## Conceptual architecture

At a high level, the engine follows this flow:

1. define market assumptions,
2. generate stochastic scenarios,
3. simulate paths,
4. value positions,
5. aggregate results,
6. compute portfolio-level analytics.

### Main abstractions

#### Market model
Defines how market variables evolve through time.

#### Scenario generator
Produces stochastic trajectories from the selected model.

#### Portfolio and instruments
Represents financial positions to be evaluated.

#### Valuation layer
Computes instrument or portfolio value under simulated states.

#### Risk aggregation
Transforms raw simulation outputs into risk metrics.

#### Compute backend
Executes the workload on CPU or GPU.

## Why GPU acceleration

Monte Carlo simulation is highly parallel by nature, making GPU execution a natural fit for large-scale scenario generation and valuation workloads.

The goal is not only to run faster, but to do so within a maintainable and well-structured system.

## Tech stack

- Python
- CUDA / GPU acceleration
- `pyproject.toml`
- Conda for development
- Docker + CUDA base image + venv for production
- MkDocs + GitHub Pages
- SonarQube
- pytest / ruff / mypy

## Environment strategy

### Development
Development uses Conda together with `pyproject.toml` and a local CUDA Toolkit.

This setup is convenient for iterative work, experimentation, and local GPU development.

### Production
Production uses Docker, an NVIDIA CUDA base image, a Python virtual environment, and the same Python package definition via `pyproject.toml`.

This keeps deployment leaner and more operationally predictable.

## Development setup

### Prerequisites
- Conda
- NVIDIA drivers
- CUDA Toolkit

### Install
```bash
conda env create -f environment.yml
conda activate portfolio-risk-engine
pip install -e ".[dev]"