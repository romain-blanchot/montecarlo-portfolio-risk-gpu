# Architecture

The Portfolio Risk Engine follows a **hexagonal (ports & adapters) architecture** with three layers. Domain logic is fully isolated from infrastructure concerns.

## Layer Overview

```mermaid
graph TB
    subgraph CLI["CLI Layer"]
        cli[portfolio-sim<br/>Interactive menu]
    end

    subgraph APP["Application Layer"]
        direction LR
        uc1[FetchMarketData]
        uc2[ComputeLogReturns]
        uc3[EstimateMarketParameters]
        uc4[RunMonteCarlo]
        uc5[ComputePortfolioRisk]
    end

    subgraph DOMAIN["Domain Layer"]
        direction TB
        subgraph VO["Value Objects"]
            ticker[Ticker]
            currency[Currency]
            weight[Weight]
            daterange[DateRange]
        end
        subgraph MODELS["Models"]
            asset[Asset]
            position[Position]
            portfolio[Portfolio]
            hp[HistoricalPrices]
            hr[HistoricalReturns]
            mp[MarketParameters]
            gbm[MultivariateGBM]
            sr[SimulationResult]
            rm[RiskMetrics]
        end
        subgraph PORTS["Ports"]
            mdp[MarketDataProvider<br/>Protocol]
            mce[MonteCarloEngine<br/>Protocol]
        end
        subgraph SERVICES["Services"]
            chol[cholesky]
        end
    end

    subgraph INFRA["Infrastructure Layer"]
        yahoo[YahooFinance<br/>MarketDataProvider]
        cpu[CpuMonteCarlo<br/>Engine]
        gpu[GpuMonteCarlo<br/>Engine]
        gpuaccel[GpuAccelerated<br/>Pipeline]
    end

    cli --> APP
    APP --> DOMAIN
    yahoo -.->|implements| mdp
    cpu -.->|implements| mce
    gpu -.->|implements| mce

    style DOMAIN fill:#e8eaf6,stroke:#3f51b5
    style APP fill:#e3f2fd,stroke:#2196f3
    style INFRA fill:#fce4ec,stroke:#e91e63
    style CLI fill:#e8f5e9,stroke:#4caf50
```

## Dependency Rule

Dependencies point **inward** — infrastructure and application depend on domain, never the reverse.

```mermaid
graph LR
    INFRA[Infrastructure] -->|depends on| DOMAIN[Domain]
    APP[Application] -->|depends on| DOMAIN
    CLI -->|depends on| APP
    CLI -->|depends on| INFRA

    DOMAIN -.-x|never depends on| INFRA
    DOMAIN -.-x|never depends on| APP

    style DOMAIN fill:#e8eaf6,stroke:#3f51b5,stroke-width:3px
```

- **Domain** defines `Protocol` interfaces (ports) with zero external dependencies
- **Infrastructure** provides concrete implementations (adapters)
- **Application** orchestrates domain operations via use cases

## Data Pipeline

The simulation flow chains five use cases in sequence. Each use case transforms one domain model into the next:

```mermaid
flowchart LR
    A["📊 FetchMarketData"] --> B["📈 ComputeLogReturns"]
    B --> C["📐 EstimateMarket<br/>Parameters"]
    C --> D["🎲 RunMonteCarlo"]
    D --> E["⚠️ ComputePortfolio<br/>Risk"]

    A1[/"HistoricalPrices"/] --> B
    B1[/"HistoricalReturns"/] --> C
    C1[/"MarketParameters"/] --> D
    D1[/"SimulationResult"/] --> E
    E1[/"PortfolioRiskMetrics"/]

    A --> A1
    B --> B1
    C --> C1
    D --> D1
    E --> E1

    style A fill:#bbdefb
    style B fill:#bbdefb
    style C fill:#bbdefb
    style D fill:#bbdefb
    style E fill:#bbdefb
    style A1 fill:#fff9c4
    style B1 fill:#fff9c4
    style C1 fill:#fff9c4
    style D1 fill:#fff9c4
    style E1 fill:#c8e6c9
```

### Step Details

| Step | Use Case | Input | Output | Key Operation |
|------|----------|-------|--------|---------------|
| 1 | `FetchMarketData` | Tickers + DateRange | `HistoricalPrices` | Yahoo Finance API call |
| 2 | `ComputeLogReturns` | `HistoricalPrices` | `HistoricalReturns` | $r_t = \ln(S_t / S_{t-1})$ |
| 3 | `EstimateMarketParameters` | `HistoricalReturns` | `MarketParameters` | Annualized drift + covariance |
| 4 | `RunMonteCarlo` | `MarketParameters` | `MonteCarloSimulationResult` | Cholesky → GBM → terminal prices |
| 5 | `ComputePortfolioRisk` | Simulation + Portfolio | `PortfolioRiskMetrics` | Weighted returns → VaR/ES |

## Domain Model

```mermaid
classDiagram
    class Ticker {
        +str value
        __post_init__() validates
    }
    class Currency {
        +str code
    }
    class Weight {
        +float value
        0.0 ≤ value ≤ 1.0
    }
    class DateRange {
        +date start
        +date end
        start < end
    }

    class Asset {
        +Ticker ticker
        +Currency currency
        +str? name
    }
    class Position {
        +Asset asset
        +Weight weight
    }
    class Portfolio {
        +tuple~Position~ positions
        +tickers() list~Ticker~
        +weights() list~float~
        Σ weights = 1.0
    }

    class HistoricalPrices {
        +tuple~Ticker~ tickers
        +tuple~date~ dates
        +dict prices_by_ticker
    }
    class HistoricalReturns {
        +tuple~Ticker~ tickers
        +tuple~date~ dates
        +dict returns_by_ticker
    }

    class MarketParameters {
        +tuple~Ticker~ tickers
        +tuple~float~ drift_vector
        +tuple~tuple~float~~ covariance_matrix
        +int annualization_factor
    }
    class MultivariateGBM {
        +MarketParameters market_parameters
        +tuple~tuple~float~~ cholesky_factor
    }

    class MonteCarloSimulationResult {
        +tuple~Ticker~ tickers
        +tuple~float~ initial_prices
        +dict terminal_prices
        +int num_simulations
        +int time_horizon_days
    }
    class PortfolioRiskMetrics {
        +float mean_return
        +float volatility
        +float var_95
        +float var_99
        +float es_95
        +float es_99
    }

    Asset --> Ticker
    Asset --> Currency
    Position --> Asset
    Position --> Weight
    Portfolio --> Position

    HistoricalPrices --> Ticker
    HistoricalReturns --> Ticker
    MarketParameters --> Ticker
    MultivariateGBM --> MarketParameters
    MonteCarloSimulationResult --> Ticker
```

## Compute Backends

The engine supports two execution backends through the `MonteCarloEngine` protocol:

```mermaid
graph TB
    subgraph PROTOCOL["Port (Protocol)"]
        MCE[MonteCarloEngine<br/>simulate]
    end

    subgraph CPU["CPU Backend"]
        CPUE[CpuMonteCarloEngine<br/>NumPy + np.random]
    end

    subgraph GPU["GPU Backends"]
        GPUE[GpuMonteCarloEngine<br/>CuPy arrays]
        GPUA[GpuAcceleratedPipeline<br/>Fused simulate + risk]
    end

    CPUE -.->|implements| MCE
    GPUE -.->|implements| MCE
    GPUA -.->|bypasses port| MCE

    style PROTOCOL fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    style CPU fill:#e8f5e9,stroke:#4caf50
    style GPU fill:#fff3e0,stroke:#ff9800
```

| Backend | Library | Data Path | Best For |
|---------|---------|-----------|----------|
| `CpuMonteCarloEngine` | NumPy | CPU → tuples → CPU | Development, small-scale |
| `GpuMonteCarloEngine` | CuPy | GPU → tuples → CPU | Medium scale, compatible with domain pipeline |
| `GpuAcceleratedPipeline` | CuPy | GPU → GPU → 6 scalars | Production, large-scale (zero tuple allocation) |

The `GpuAcceleratedPipeline` fuses simulation and risk computation into a single GPU pass — only 6 scalar floats (the risk metrics) are transferred back to CPU. This avoids the overhead of materializing millions of terminal prices as Python tuples.

## Project Structure

```
src/portfolio_risk_engine/
├── domain/
│   ├── value_objects/          # Ticker, Currency, Weight, DateRange
│   ├── models/                 # Asset, Position, Portfolio, ...
│   ├── ports/                  # MarketDataProvider, MonteCarloEngine (Protocol)
│   └── services/               # cholesky()
├── application/
│   └── use_cases/              # FetchMarketData, ComputeLogReturns, ...
├── infrastructure/
│   ├── market_data/            # YahooFinanceMarketDataProvider
│   └── simulation/             # CpuMonteCarloEngine, Gpu*
├── cli.py                      # Interactive CLI (PortfolioSimulatorCLI)
└── __main__.py                 # python -m portfolio_risk_engine
```
