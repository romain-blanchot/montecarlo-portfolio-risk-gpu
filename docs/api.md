# API Reference

Auto-generated documentation from source code docstrings.

## Domain Layer

### Value Objects

::: portfolio_risk_engine.domain.value_objects.ticker

::: portfolio_risk_engine.domain.value_objects.currency

::: portfolio_risk_engine.domain.value_objects.weight

::: portfolio_risk_engine.domain.value_objects.date_range

### Models

::: portfolio_risk_engine.domain.models.asset

::: portfolio_risk_engine.domain.models.position

::: portfolio_risk_engine.domain.models.portfolio

::: portfolio_risk_engine.domain.models.historical_prices

::: portfolio_risk_engine.domain.models.historical_returns

::: portfolio_risk_engine.domain.models.market_parameters

::: portfolio_risk_engine.domain.models.gbm_model

::: portfolio_risk_engine.domain.models.simulation_result

::: portfolio_risk_engine.domain.models.portfolio_risk_metrics

### Ports

::: portfolio_risk_engine.domain.ports.market_data_provider

::: portfolio_risk_engine.domain.ports.monte_carlo_engine

### Services

::: portfolio_risk_engine.domain.services.cholesky

## Application Layer

### Use Cases

::: portfolio_risk_engine.application.use_cases.fetch_market_data

::: portfolio_risk_engine.application.use_cases.compute_log_returns

::: portfolio_risk_engine.application.use_cases.estimate_market_parameters

::: portfolio_risk_engine.application.use_cases.run_monte_carlo

::: portfolio_risk_engine.application.use_cases.compute_portfolio_risk

## Infrastructure Layer

### Market Data

::: portfolio_risk_engine.infrastructure.market_data.yahoo_finance_market_data_provider

### Simulation Engines

::: portfolio_risk_engine.infrastructure.simulation.cpu_monte_carlo_engine

::: portfolio_risk_engine.infrastructure.simulation.gpu_monte_carlo_engine

::: portfolio_risk_engine.infrastructure.simulation.gpu_accelerated_pipeline
