import numpy as np

from portfolio_risk_engine.domain.models.portfolio import Portfolio
from portfolio_risk_engine.domain.models.portfolio_risk_metrics import (
    PortfolioRiskMetrics,
)
from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)


class ComputePortfolioRisk:
    @staticmethod
    def execute(
        portfolio: Portfolio,
        simulation_result: MonteCarloSimulationResult,
    ) -> PortfolioRiskMetrics:
        weight_map = dict(zip(portfolio.tickers, portfolio.weights))
        initial_price_map = dict(
            zip(simulation_result.tickers, simulation_result.initial_prices)
        )

        for ticker in simulation_result.tickers:
            if ticker not in weight_map:
                raise ValueError(
                    f"Ticker {ticker.value} from simulation not found in portfolio."
                )

        n_sims = simulation_result.num_simulations
        portfolio_returns = np.zeros(n_sims)

        for ticker in simulation_result.tickers:
            w = weight_map[ticker]
            s0 = initial_price_map[ticker]
            st = np.array(simulation_result.terminal_prices[ticker])
            asset_returns = st / s0 - 1.0
            portfolio_returns += w * asset_returns

        # Loss-positive convention
        losses = -portfolio_returns

        mean_return = float(np.mean(portfolio_returns))
        volatility = float(np.std(portfolio_returns, ddof=1))

        var_95 = float(np.percentile(losses, 95))
        var_99 = float(np.percentile(losses, 99))

        es_95 = float(np.mean(losses[losses >= var_95]))
        es_99 = float(np.mean(losses[losses >= var_99]))

        return PortfolioRiskMetrics(
            mean_return=mean_return,
            volatility=volatility,
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
        )
