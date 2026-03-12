# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated Monte Carlo engine for simulating multi-asset portfolio dynamics and estimating risk metrics (VaR, Expected Shortfall). Python 3.13+, built with Hatch.

## Development Setup

```bash
conda env update -f environment.yml --prune
conda activate portfolio-risk-engine
python -m pip install -U pip
python -m pip install --group dev
```

The conda environment provides CUDA toolkit (nvidia channel) + Python 3.13. The `pip: -e .[dev]` in `environment.yml` installs the package in editable mode with dev dependencies.

## Common Commands

```bash
# Run the CLI
portfolio-sim                    # entry point defined in pyproject.toml
python -m portfolio_risk_engine  # alternative via __main__.py

# Linting & formatting (ruff)
ruff check .
ruff format .

# Type checking
mypy src/

# Tests
pytest                          # runs unit tests only (integration/gpu excluded by default)
pytest tests/test_foo.py        # single file
pytest tests/test_foo.py::test_bar  # single test
pytest -x                       # stop on first failure
pytest -m integration           # integration tests (require network)
pytest -m gpu                   # GPU tests (require CUDA)

# Build
python -m build                  # produces sdist + wheel in dist/

# Pre-commit (runs ruff, ruff-format, mypy, pytest)
pre-commit run --all-files
```

## Architecture

The source lives in `src/portfolio_risk_engine/` and follows a hexagonal (ports-and-adapters) structure:

- **`domain/`** - Core business logic, no external dependencies
  - `value_objects/` - Immutable primitives: `Ticker`, `Currency`, `Weight`, `DateRange`
  - `models/` - Frozen dataclasses: `Asset`, `Portfolio`, `Position`, `HistoricalPrices`, `HistoricalReturns`, `MarketParameters`, `MultivariateGBM`, `MonteCarloSimulationResult`, `PortfolioRiskMetrics`
  - `ports/` - `Protocol`-based interfaces: `MarketDataProvider`, `MonteCarloEngine`
  - `services/` - Pure domain logic (e.g. `cholesky` decomposition)
- **`application/use_cases/`** - Orchestration layer, each use case is a class with an `execute()` method
- **`infrastructure/`** - External adapters implementing the domain ports
  - `market_data/` - `YahooFinanceMarketDataProvider` (implements `MarketDataProvider`)
  - `simulation/` - `CpuMonteCarloEngine`, `GpuMonteCarloEngine` (implement `MonteCarloEngine`)
- **`cli.py`** / **`__main__.py`** - CLI entry point

### Simulation Pipeline

The end-to-end flow chains use cases sequentially:

```
FetchMarketData → ComputeLogReturns → EstimateMarketParameters → RunMonteCarlo → ComputePortfolioRisk
```

1. **FetchMarketData**: pulls historical prices via a `MarketDataProvider` port → `HistoricalPrices`
2. **ComputeLogReturns**: converts prices to log returns → `HistoricalReturns`
3. **EstimateMarketParameters**: estimates annualized drift vector + covariance matrix → `MarketParameters`
4. **RunMonteCarlo**: computes Cholesky factor, builds `MultivariateGBM` model, delegates to a `MonteCarloEngine` port → `MonteCarloSimulationResult` (terminal prices per ticker)
5. **ComputePortfolioRisk**: aggregates terminal prices into portfolio-level risk metrics (VaR 95/99, ES 95/99) using loss-positive convention → `PortfolioRiskMetrics`

### Key Patterns

- **Domain models** are `@dataclass(frozen=True)` with invariant validation in `__post_init__`. All collections use immutable `tuple` types rather than `list`.
- **Ports** are `typing.Protocol` classes — adapters implement them structurally (no inheritance required).
- **Use cases** take port dependencies via constructor injection and expose a single `execute()` method.

## Branch Model & Workflow

- `main` - stable/production
- `integration` - staging/pre-production
- Working branches: `feat/...`, `fix/...`, `chore/...`, `docs/...`, `refactor/...`, `hotfix/...`
- PRs target `integration` for standard changes; hotfixes may target `main`
- Commit messages follow Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`, `ci:`

## Test Structure

Tests mirror the source layout under `tests/` (e.g. `tests/domain/models/test_asset.py` tests `domain/models/asset.py`). Integration tests live in `tests/integration/` and are marked with `@pytest.mark.integration`. The default pytest config in `pyproject.toml` excludes `integration` and `gpu` markers, so `pytest` runs only unit tests.

## CI/CD

- **CI** (`ci.yml`): Runs on push/PR to `main` and `integration`. Parallel jobs: lint, format, typecheck, test (with coverage), build, Docker (GHCR), SonarQube
- **Release** (`release.yml`): release-please-action on pushes to `main`
- **CD** (`cd.yml`): Triggered on tag push (v*), re-tags Docker image
- **Docs** (`docs.yml`): Deploys versioned documentation with mike (mkdocs-material)

## Key Dependencies

- **Runtime**: numpy, pandas, yfinance (numba, cupy, scipy planned but currently commented out)
- **Dev**: ruff, mypy, pytest, pytest-cov, pre-commit, mkdocs + mkdocs-material + mkdocstrings[python] + mike
- **Build**: hatchling + hatch-vcs (version from git tags)
