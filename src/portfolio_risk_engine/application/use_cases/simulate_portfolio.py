from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from portfolio_risk_engine.application.use_cases.compute_log_returns import (
    ComputeLogReturns,
)
from portfolio_risk_engine.application.use_cases.compute_portfolio_risk import (
    ComputePortfolioRisk,
)
from portfolio_risk_engine.application.use_cases.estimate_heston_parameters import (
    EstimateHestonParameters,
)
from portfolio_risk_engine.application.use_cases.estimate_market_parameters import (
    EstimateMarketParameters,
)
from portfolio_risk_engine.application.use_cases.estimate_student_t_parameters import (
    EstimateStudentTParameters,
)
from portfolio_risk_engine.application.use_cases.fetch_market_data import (
    FetchMarketData,
)
from portfolio_risk_engine.application.use_cases.run_monte_carlo import RunMonteCarlo
from portfolio_risk_engine.domain.models.historical_prices import HistoricalPrices
from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.models.portfolio import Portfolio
from portfolio_risk_engine.domain.models.portfolio_risk_metrics import (
    PortfolioRiskMetrics,
)
from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)
from portfolio_risk_engine.domain.models.student_t_gbm import StudentTGBM
from portfolio_risk_engine.domain.ports.market_data_provider import MarketDataProvider
from portfolio_risk_engine.domain.ports.monte_carlo_engine import MonteCarloEngine
from portfolio_risk_engine.domain.services.cholesky import cholesky
from portfolio_risk_engine.domain.value_objects.date_range import DateRange

SUPPORTED_MODELS = ("gbm", "student_t", "heston")


@dataclass(frozen=True)
class ModelInfo:
    name: str
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioResult:
    portfolio: Portfolio
    historical_prices: HistoricalPrices
    market_parameters: MarketParameters
    simulation_result: MonteCarloSimulationResult
    risk_metrics: PortfolioRiskMetrics
    model_info: ModelInfo | None = None


class SimulatePortfolio:
    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        monte_carlo_engine: MonteCarloEngine | None = None,
        *,
        use_gpu: bool = False,
    ) -> None:
        self._provider = market_data_provider
        self._engine = monte_carlo_engine
        self._use_gpu = use_gpu

    def execute(
        self,
        portfolio: Portfolio,
        date_range: DateRange,
        num_simulations: int = 10_000,
        time_horizon_days: int = 21,
        model: str = "gbm",
    ) -> ScenarioResult:
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model: {model}. Choose from: {', '.join(SUPPORTED_MODELS)}"
            )

        tickers = tuple(portfolio.tickers)

        prices = FetchMarketData(self._provider).execute(
            tickers=tickers,
            date_range=date_range,
        )

        returns = ComputeLogReturns.execute(prices)
        market_params = EstimateMarketParameters().execute(returns)

        initial_prices = tuple(
            prices.prices_by_ticker[t][-1] for t in market_params.tickers
        )

        if model == "gbm":
            sim_result, model_info = self._run_gbm(
                market_params,
                initial_prices,
                num_simulations,
                time_horizon_days,
            )
        elif model == "student_t":
            sim_result, model_info = self._run_student_t(
                market_params,
                returns,
                initial_prices,
                num_simulations,
                time_horizon_days,
            )
        else:  # heston
            sim_result, model_info = self._run_heston(
                market_params,
                returns,
                initial_prices,
                num_simulations,
                time_horizon_days,
            )

        risk_metrics = ComputePortfolioRisk.execute(portfolio, sim_result)

        return ScenarioResult(
            portfolio=portfolio,
            historical_prices=prices,
            market_parameters=market_params,
            simulation_result=sim_result,
            risk_metrics=risk_metrics,
            model_info=model_info,
        )

    def _run_gbm(
        self,
        market_params: MarketParameters,
        initial_prices: tuple[float, ...],
        num_simulations: int,
        time_horizon_days: int,
    ) -> tuple[MonteCarloSimulationResult, ModelInfo]:
        if self._engine is not None:
            engine = self._engine
        else:
            engine = self._make_gbm_engine()

        result = RunMonteCarlo(engine=engine).execute(
            market_params=market_params,
            initial_prices=initial_prices,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
        )

        return result, ModelInfo(name="GBM")

    def _run_student_t(
        self,
        market_params: MarketParameters,
        returns: object,
        initial_prices: tuple[float, ...],
        num_simulations: int,
        time_horizon_days: int,
    ) -> tuple[MonteCarloSimulationResult, ModelInfo]:
        from portfolio_risk_engine.domain.models.historical_returns import (
            HistoricalReturns,
        )

        assert isinstance(returns, HistoricalReturns)

        df = EstimateStudentTParameters().execute(returns)
        cholesky_factor = cholesky(market_params.covariance_matrix)
        model = StudentTGBM(
            market_parameters=market_params,
            cholesky_factor=cholesky_factor,
            degrees_of_freedom=df,
        )

        engine = self._make_student_t_engine()
        result = engine.simulate(
            model=model,
            initial_prices=initial_prices,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
        )

        return result, ModelInfo(name="Student-t GBM", parameters={"df": df})

    def _run_heston(
        self,
        market_params: MarketParameters,
        returns: object,
        initial_prices: tuple[float, ...],
        num_simulations: int,
        time_horizon_days: int,
    ) -> tuple[MonteCarloSimulationResult, ModelInfo]:
        from portfolio_risk_engine.domain.models.historical_returns import (
            HistoricalReturns,
        )

        assert isinstance(returns, HistoricalReturns)

        heston_model = EstimateHestonParameters().execute(returns, market_params)

        engine = self._make_heston_engine()
        result = engine.simulate(
            model=heston_model,
            initial_prices=initial_prices,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
        )

        params: dict[str, float] = {}
        for i, ticker in enumerate(heston_model.tickers):
            p = heston_model.asset_params[i]
            prefix = ticker.value
            params[f"{prefix}_kappa"] = p.kappa
            params[f"{prefix}_theta"] = p.theta
            params[f"{prefix}_xi"] = p.xi
            params[f"{prefix}_rho"] = p.rho
            params[f"{prefix}_v0"] = p.v0
            params[f"{prefix}_vol0"] = sqrt(p.v0)
            params[f"{prefix}_vol_lr"] = sqrt(p.theta)
            params[f"{prefix}_feller"] = 1.0 if p.feller_satisfied else 0.0

        return result, ModelInfo(name="Heston", parameters=params)

    def _make_gbm_engine(self) -> MonteCarloEngine:
        if self._use_gpu:
            try:
                from portfolio_risk_engine.infrastructure.simulation.gpu_monte_carlo_engine import (
                    GpuMonteCarloEngine,
                )

                return GpuMonteCarloEngine()
            except (RuntimeError, ImportError):
                pass
        from portfolio_risk_engine.infrastructure.simulation.cpu_monte_carlo_engine import (
            CpuMonteCarloEngine,
        )

        return CpuMonteCarloEngine()

    def _make_student_t_engine(self) -> Any:
        if self._use_gpu:
            try:
                from portfolio_risk_engine.infrastructure.simulation.gpu_student_t_engine import (
                    GpuStudentTEngine,
                )

                return GpuStudentTEngine()
            except (RuntimeError, ImportError):
                pass
        from portfolio_risk_engine.infrastructure.simulation.cpu_student_t_engine import (
            CpuStudentTEngine,
        )

        return CpuStudentTEngine()

    def _make_heston_engine(self) -> Any:
        if self._use_gpu:
            try:
                from portfolio_risk_engine.infrastructure.simulation.gpu_heston_engine import (
                    GpuHestonEngine,
                )

                return GpuHestonEngine()
            except (RuntimeError, ImportError):
                pass
        from portfolio_risk_engine.infrastructure.simulation.cpu_heston_engine import (
            CpuHestonEngine,
        )

        return CpuHestonEngine()
