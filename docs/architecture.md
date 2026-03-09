# Architecture

The Portfolio Risk Engine follows a layered (hexagonal) architecture for clean separation of concerns.

## Project Structure

```
src/portfolio_risk_engine/
├── domain/           # Core business logic
├── application/      # Use cases and orchestration
├── infrastructure/   # External adapters (GPU, I/O)
├── cli.py           # CLI entry point
└── __main__.py      # Module execution support
```

## Layers

### Domain Layer

Core business logic and domain models:

- **Risk Metrics**: VaR, Expected Shortfall calculations
- **Portfolio Entities**: Asset, Portfolio, Position models
- **Simulation Parameters**: Configuration for Monte Carlo runs

### Application Layer

Use cases and orchestration:

- **Simulation Workflows**: Coordinate simulation execution
- **Result Aggregation**: Combine and process simulation outputs

### Infrastructure Layer

External adapters and integrations:

- **GPU Compute**: CUDA/CuPy acceleration
- **Data Sources**: Market data ingestion
- **I/O**: File and network operations

## Data Flow

```
CLI → Application → Domain ← Infrastructure
         ↓            ↑
    Use Cases    GPU Compute
```

1. CLI parses user input and invokes application use cases
2. Application layer orchestrates domain operations
3. Infrastructure provides GPU acceleration and data access
4. Results flow back through the layers to the user
