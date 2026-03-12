from datetime import date

from portfolio_risk_engine.application.use_cases.simulate_portfolio import (
    ModelInfo,
    ScenarioResult,
)
from portfolio_risk_engine.domain.models.asset import Asset
from portfolio_risk_engine.domain.models.historical_prices import HistoricalPrices
from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.models.portfolio import Portfolio
from portfolio_risk_engine.domain.models.portfolio_risk_metrics import (
    PortfolioRiskMetrics,
)
from portfolio_risk_engine.domain.models.position import Position
from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)
from portfolio_risk_engine.domain.value_objects.currency import Currency
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.domain.value_objects.weight import Weight
from portfolio_risk_engine.infrastructure.rendering.scenario_renderer import (
    render_scenario,
)


def _make_scenario_result() -> ScenarioResult:
    aapl = Ticker("AAPL")
    msft = Ticker("MSFT")

    portfolio = Portfolio(
        positions=(
            Position(
                asset=Asset(ticker=aapl, currency=Currency("USD")),
                weight=Weight(0.6),
            ),
            Position(
                asset=Asset(ticker=msft, currency=Currency("USD")),
                weight=Weight(0.4),
            ),
        )
    )

    prices = HistoricalPrices(
        tickers=(aapl, msft),
        dates=(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)),
        prices_by_ticker={
            aapl: (150.0, 152.0, 151.0),
            msft: (300.0, 305.0, 303.0),
        },
    )

    market_params = MarketParameters(
        tickers=(aapl, msft),
        drift_vector=(0.10, 0.12),
        covariance_matrix=(
            (0.04, 0.01),
            (0.01, 0.05),
        ),
        annualization_factor=252,
    )

    terminal = {
        aapl: (155.0, 148.0, 152.0, 150.0, 153.0, 149.0, 156.0, 151.0, 154.0, 147.0),
        msft: (310.0, 295.0, 305.0, 300.0, 308.0, 298.0, 312.0, 303.0, 307.0, 294.0),
    }

    simulation_result = MonteCarloSimulationResult(
        tickers=(aapl, msft),
        initial_prices=(151.0, 303.0),
        terminal_prices=terminal,
        num_simulations=10,
        time_horizon_days=21,
    )

    risk_metrics = PortfolioRiskMetrics(
        mean_return=0.0185,
        volatility=0.0423,
        var_95=0.0512,
        var_99=0.0784,
        es_95=0.0645,
        es_99=0.0921,
    )

    return ScenarioResult(
        portfolio=portfolio,
        historical_prices=prices,
        market_parameters=market_params,
        simulation_result=simulation_result,
        risk_metrics=risk_metrics,
    )


class TestScenarioRenderer:
    def test_returns_string(self) -> None:
        output = render_scenario(_make_scenario_result())
        assert isinstance(output, str)
        assert len(output) > 0

    def test_contains_header(self) -> None:
        output = render_scenario(_make_scenario_result())
        assert "PORTFOLIO SCENARIO SIMULATION" in output

    def test_contains_ticker_names(self) -> None:
        output = render_scenario(_make_scenario_result())
        assert "AAPL" in output
        assert "MSFT" in output

    def test_contains_asset_summary(self) -> None:
        output = render_scenario(_make_scenario_result())
        assert "ASSET SUMMARY" in output

    def test_contains_risk_metrics(self) -> None:
        output = render_scenario(_make_scenario_result())
        assert "PORTFOLIO RISK" in output
        assert "VaR" in output
        assert "ES" in output

    def test_contains_histogram_bars(self) -> None:
        output = render_scenario(_make_scenario_result())
        assert "\u2588" in output  # full block character
        assert "\u2524" in output  # box drawing character

    def test_contains_percentiles(self) -> None:
        output = render_scenario(_make_scenario_result())
        assert "P5" in output
        assert "P50" in output
        assert "P95" in output

    def test_contains_portfolio_return_distribution(self) -> None:
        output = render_scenario(_make_scenario_result())
        assert "Return Distribution" in output

    def test_contains_observation_count(self) -> None:
        output = render_scenario(_make_scenario_result())
        assert "3 observations" in output

    def test_contains_simulation_count(self) -> None:
        output = render_scenario(_make_scenario_result())
        assert "10 paths" in output

    def test_contains_tail_metrics(self) -> None:
        output = render_scenario(_make_scenario_result())
        assert "Skewness" in output
        assert "Excess Kurtosis" in output
        assert "Prob(Loss)" in output

    def test_contains_model_info_gbm(self) -> None:
        result = _make_scenario_result()
        # Default has no model_info (None) -> should show "GBM"
        output = render_scenario(result)
        assert "GBM" in output

    def test_contains_model_info_student_t(self) -> None:
        base = _make_scenario_result()
        result = ScenarioResult(
            portfolio=base.portfolio,
            historical_prices=base.historical_prices,
            market_parameters=base.market_parameters,
            simulation_result=base.simulation_result,
            risk_metrics=base.risk_metrics,
            model_info=ModelInfo(name="Student-t GBM", parameters={"df": 5.3}),
        )
        output = render_scenario(result)
        assert "Student-t GBM" in output
        assert "5.3" in output

    def test_contains_heston_params(self) -> None:
        base = _make_scenario_result()
        params = {
            "AAPL_kappa": 2.0,
            "AAPL_vol_lr": 0.2,
            "AAPL_xi": 0.3,
            "AAPL_rho": -0.7,
            "AAPL_vol0": 0.2,
            "AAPL_feller": 1.0,
            "MSFT_kappa": 1.5,
            "MSFT_vol_lr": 0.22,
            "MSFT_xi": 0.25,
            "MSFT_rho": -0.5,
            "MSFT_vol0": 0.22,
            "MSFT_feller": 0.0,
        }
        result = ScenarioResult(
            portfolio=base.portfolio,
            historical_prices=base.historical_prices,
            market_parameters=base.market_parameters,
            simulation_result=base.simulation_result,
            risk_metrics=base.risk_metrics,
            model_info=ModelInfo(name="Heston", parameters=params),
        )
        output = render_scenario(result)
        assert "Heston" in output
        assert "AAPL" in output
        assert "Feller" in output
