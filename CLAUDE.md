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
pytest tests/                    # full suite
pytest tests/test_foo.py         # single file
pytest tests/test_foo.py::test_bar  # single test
pytest -x                       # stop on first failure

# Build
python -m build                  # produces sdist + wheel in dist/

# Pre-commit (runs ruff, ruff-format, mypy, pytest)
pre-commit run --all-files
```

## Architecture

The source lives in `src/portfolio_risk_engine/` and follows a layered (hexagonal-ish) structure:

- **`domain/`** - Core business logic and domain models (risk metrics, portfolio entities)
- **`application/`** - Use cases and orchestration (simulation workflows)
- **`infrastructure/`** - External adapters (GPU compute, data sources, I/O)
- **`cli.py`** - CLI entry point (`main()`)
- **`__main__.py`** - Enables `python -m portfolio_risk_engine` execution

## Branch Model & Workflow

- `main` - stable/production
- `integration` - staging/pre-production
- Working branches: `feat/...`, `fix/...`, `chore/...`, `docs/...`, `refactor/...`, `hotfix/...`
- PRs target `integration` for standard changes; hotfixes may target `main`
- Commit messages follow Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`, `ci:`

## CI/CD

- **CI** (`ci.yml`): Runs on push/PR to `main` and `integration`. Parallel jobs:
  - `ci-lint`: ruff check
  - `ci-format`: ruff format --check
  - `ci-typecheck`: mypy
  - `ci-test`: pytest with coverage
  - `ci-build`: python -m build (depends on all checks)
  - `ci-docker`: Build and push Docker image to GHCR (depends on all checks)
  - `ci-sonarqube`: SonarQube analysis (depends on test)
- **Release** (`release.yml`): Uses release-please-action on pushes to `main` to manage changelog and tags
- **CD** (`cd.yml`): Triggered on tag push (v*), re-tags Docker image with release version and `latest`
- **Docs** (`docs.yml`): Deploys versioned documentation with mike:
  - `docs-release`: Deploys versioned docs on release tags
  - `docs-dev`: Deploys dev docs on push to `main`/`integration`

## Key Dependencies

- **Runtime**: numpy, pandas (numba, cupy, scipy planned but currently commented out)
- **Dev**: ruff, mypy, pytest, pytest-cov, pre-commit, mkdocs + mkdocs-material + mkdocstrings[python] + mike
- **Build**: hatchling + hatch-vcs (version from git tags)
