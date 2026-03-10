from portfolio_risk_engine.domain.models.gbm_model import MultivariateGBM
from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)
from portfolio_risk_engine.domain.ports.monte_carlo_engine import MonteCarloEngine
from portfolio_risk_engine.domain.services.cholesky import cholesky


class RunMonteCarlo:
    def __init__(self, engine: MonteCarloEngine) -> None:
        self._engine = engine

    def execute(
        self,
        market_params: MarketParameters,
        initial_prices: tuple[float, ...],
        num_simulations: int,
        time_horizon_days: int,
    ) -> MonteCarloSimulationResult:
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive.")

        if time_horizon_days <= 0:
            raise ValueError("time_horizon_days must be positive.")

        if len(initial_prices) != len(market_params.tickers):
            raise ValueError(
                f"initial_prices length ({len(initial_prices)}) "
                f"must match number of tickers ({len(market_params.tickers)})."
            )

        for price in initial_prices:
            if price <= 0:
                raise ValueError("All initial prices must be positive.")

        cholesky_factor = cholesky(market_params.covariance_matrix)
        model = MultivariateGBM(
            market_parameters=market_params,
            cholesky_factor=cholesky_factor,
        )

        return self._engine.simulate(
            model=model,
            initial_prices=initial_prices,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
        )
