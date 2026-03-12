# Getting Started

## Prerequisites

- Python 3.13+
- Conda (for CUDA toolkit)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:romain-blanchot/montecarlo-portfolio-risk-gpu.git
cd montecarlo-portfolio-risk-gpu
```

### 2. Create the Conda Environment

The conda environment provides CUDA toolkit (nvidia channel) and Python 3.13:

```bash
conda env create -f environment.yml
conda activate portfolio-risk-engine
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

Pre-commit runs ruff (lint + format), mypy, and pytest on every commit.

### 4. Verify Installation

```bash
# Run the test suite
pytest tests/

# Launch the CLI
portfolio-sim
```

## GPU Support (Optional)

For GPU acceleration, install CuPy matching your CUDA version:

```bash
pip install cupy-cuda12x
```

Verify GPU availability:

```python
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s) found')"
```

## Project Layout

```
montecarlo-portfolio-risk-gpu/
├── src/portfolio_risk_engine/    # Source code
│   ├── domain/                   # Business logic (models, ports, services)
│   ├── application/              # Use cases
│   ├── infrastructure/           # Adapters (Yahoo Finance, CPU/GPU engines)
│   └── cli.py                    # Interactive CLI
├── tests/                        # Unit + integration tests
├── notebooks/                    # Jupyter demos and benchmarks
├── scripts/                      # Benchmark scripts
├── docs/                         # MkDocs documentation source
├── pyproject.toml                # Build config (hatchling + hatch-vcs)
└── environment.yml               # Conda environment
```
