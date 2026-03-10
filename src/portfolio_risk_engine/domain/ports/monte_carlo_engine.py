from typing import Protocol

from portfolio_risk_engine.domain.models.gbm_model import MultivariateGBM
from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)


class MonteCarloEngine(Protocol):
    def simulate(
        self,
        model: MultivariateGBM,
        initial_prices: tuple[float, ...],
        num_simulations: int,
        time_horizon_days: int,
    ) -> MonteCarloSimulationResult: ...
