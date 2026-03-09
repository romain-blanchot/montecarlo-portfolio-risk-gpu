# Portfolio Risk Engine

GPU-accelerated Monte Carlo engine for simulating multi-asset portfolio dynamics and estimating risk metrics.

## Features

- **Monte Carlo Simulation**: Simulate thousands of portfolio scenarios
- **Risk Metrics**: Calculate VaR (Value at Risk) and Expected Shortfall
- **GPU Acceleration**: Leverage CUDA for high-performance computations
- **Multi-Asset Support**: Handle portfolios with multiple correlated assets

## Quick Start

### Installation

```bash
pip install portfolio-risk-engine
```

### Basic Usage

```python
from portfolio_risk_engine import PortfolioSimulator

# Create simulator
simulator = PortfolioSimulator()

# Run simulation
results = simulator.run()

# Get risk metrics
var_95 = results.value_at_risk(0.95)
es_95 = results.expected_shortfall(0.95)
```

## CLI

```bash
portfolio-sim --help
```

## Development

```bash
# Clone and setup
git clone https://github.com/romain-blanchot/montecarlo-portfolio-risk-gpu.git
cd montecarlo-portfolio-risk-gpu

# Install with dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Run linting
ruff check .
```
