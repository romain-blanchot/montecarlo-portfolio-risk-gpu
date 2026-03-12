# Development

## Commands

### Linting & Formatting

```bash
# Lint (ruff)
ruff check .

# Auto-fix lint issues
ruff check . --fix

# Format
ruff format .

# Check format without modifying
ruff format . --check
```

### Type Checking

```bash
mypy src/
```

### Tests

```bash
# Full unit test suite (excludes integration and GPU tests by default)
pytest tests/

# Single file
pytest tests/domain/models/test_portfolio.py

# Single test
pytest tests/domain/models/test_portfolio.py::test_portfolio_weights_must_sum_to_one

# Stop on first failure
pytest -x

# With coverage
pytest --cov=src --cov-report=term-missing

# Integration tests (require network access)
pytest -m integration

# GPU tests (require CUDA)
pytest -m gpu
```

!!! note "Default Test Selection"
    `pyproject.toml` sets `addopts = "-m 'not integration and not gpu'"` so integration and GPU tests are excluded by default.

### Pre-commit

Runs ruff (lint + format), mypy, and pytest:

```bash
pre-commit run --all-files
```

### Build

```bash
python -m build
```

Produces sdist + wheel in `dist/`. Version is derived from git tags via `hatch-vcs`.

### Documentation

```bash
# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

## Test Organization

Tests mirror the source structure:

```
tests/
├── domain/
│   ├── value_objects/    # Ticker, Currency, Weight, DateRange
│   ├── models/           # Asset, Portfolio, MarketParameters, ...
│   └── services/         # Cholesky
├── application/
│   └── use_cases/        # Each use case has its own test file
├── infrastructure/
│   ├── market_data/      # YahooFinance provider
│   └── simulation/       # CPU + GPU engines
├── integration/          # End-to-end tests (marked @pytest.mark.integration)
└── test_cli.py           # CLI interface tests
```

## Branch Model

| Branch | Purpose |
|--------|---------|
| `main` | Stable / production |
| `integration` | Staging / pre-production |
| `feat/...` | New features |
| `fix/...` | Bug fixes |
| `chore/...` | Maintenance |
| `docs/...` | Documentation |
| `refactor/...` | Refactoring |
| `hotfix/...` | Urgent fixes (target `main`) |

PRs target `integration` for standard changes; hotfixes may target `main`.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add stress testing use case
fix: handle single-ticker covariance edge case
docs: update architecture diagram
refactor: extract cholesky into domain service
chore: update ruff to v0.15
test: add GPU engine reproducibility test
ci: add SonarQube analysis job
```
