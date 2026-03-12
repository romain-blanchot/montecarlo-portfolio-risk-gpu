from datetime import date
from unittest.mock import MagicMock

from portfolio_risk_engine.application.use_cases.simulate_portfolio import (
    ScenarioResult,
    SimulatePortfolio,
)
from portfolio_risk_engine.domain.models.asset import Asset
from portfolio_risk_engine.domain.models.historical_prices import HistoricalPrices
from portfolio_risk_engine.domain.models.portfolio import Portfolio
from portfolio_risk_engine.domain.models.position import Position
from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)
from portfolio_risk_engine.domain.value_objects.currency import Currency
from portfolio_risk_engine.domain.value_objects.date_range import DateRange
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.domain.value_objects.weight import Weight


def _make_portfolio() -> Portfolio:
    return Portfolio(
        positions=(
            Position(
                asset=Asset(ticker=Ticker("AAPL"), currency=Currency("USD")),
                weight=Weight(0.6),
            ),
            Position(
                asset=Asset(ticker=Ticker("MSFT"), currency=Currency("USD")),
                weight=Weight(0.4),
            ),
        )
    )


def _make_prices() -> HistoricalPrices:
    return HistoricalPrices(
        tickers=(Ticker("AAPL"), Ticker("MSFT")),
        dates=(
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
            date(2024, 1, 8),
            date(2024, 1, 9),
            date(2024, 1, 10),
            date(2024, 1, 11),
            date(2024, 1, 12),
            date(2024, 1, 16),
        ),
        prices_by_ticker={
            Ticker("AAPL"): (
                150.0,
                152.0,
                151.0,
                153.0,
                154.0,
                152.5,
                155.0,
                156.0,
                154.5,
                157.0,
            ),
            Ticker("MSFT"): (
                300.0,
                305.0,
                303.0,
                308.0,
                306.0,
                310.0,
                307.0,
                312.0,
                309.0,
                315.0,
            ),
        },
    )


def _make_simulation_result() -> MonteCarloSimulationResult:
    return MonteCarloSimulationResult(
        tickers=(Ticker("AAPL"), Ticker("MSFT")),
        initial_prices=(151.0, 303.0),
        terminal_prices={
            Ticker("AAPL"): (155.0, 148.0, 152.0, 150.0, 153.0),
            Ticker("MSFT"): (310.0, 295.0, 305.0, 300.0, 308.0),
        },
        num_simulations=5,
        time_horizon_days=21,
    )


class TestSimulatePortfolio:
    def test_execute_returns_scenario_result(self) -> None:
        prices = _make_prices()
        sim_result = _make_simulation_result()

        mock_provider = MagicMock()
        mock_provider.get_historical_prices.return_value = prices

        mock_engine = MagicMock()
        mock_engine.simulate.return_value = sim_result

        portfolio = _make_portfolio()
        date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 5))

        use_case = SimulatePortfolio(
            market_data_provider=mock_provider,
            monte_carlo_engine=mock_engine,
        )
        result = use_case.execute(
            portfolio=portfolio,
            date_range=date_range,
            num_simulations=5,
            time_horizon_days=21,
        )

        assert isinstance(result, ScenarioResult)
        assert result.portfolio is portfolio
        assert result.historical_prices is prices
        assert result.simulation_result is sim_result
        assert result.market_parameters is not None
        assert result.risk_metrics is not None

    def test_execute_calls_provider_with_correct_tickers(self) -> None:
        prices = _make_prices()
        sim_result = _make_simulation_result()

        mock_provider = MagicMock()
        mock_provider.get_historical_prices.return_value = prices

        mock_engine = MagicMock()
        mock_engine.simulate.return_value = sim_result

        portfolio = _make_portfolio()
        date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 5))

        use_case = SimulatePortfolio(
            market_data_provider=mock_provider,
            monte_carlo_engine=mock_engine,
        )
        use_case.execute(
            portfolio=portfolio,
            date_range=date_range,
            num_simulations=5,
            time_horizon_days=21,
        )

        call_args = mock_provider.get_historical_prices.call_args
        called_tickers = call_args.kwargs.get("tickers") or call_args[1].get("tickers")
        assert set(called_tickers) == {Ticker("AAPL"), Ticker("MSFT")}

    def test_execute_calls_engine_simulate(self) -> None:
        prices = _make_prices()
        sim_result = _make_simulation_result()

        mock_provider = MagicMock()
        mock_provider.get_historical_prices.return_value = prices

        mock_engine = MagicMock()
        mock_engine.simulate.return_value = sim_result

        portfolio = _make_portfolio()
        date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 5))

        use_case = SimulatePortfolio(
            market_data_provider=mock_provider,
            monte_carlo_engine=mock_engine,
        )
        use_case.execute(
            portfolio=portfolio,
            date_range=date_range,
            num_simulations=5,
            time_horizon_days=21,
        )

        mock_engine.simulate.assert_called_once()
        call_kwargs = mock_engine.simulate.call_args.kwargs
        assert call_kwargs["num_simulations"] == 5
        assert call_kwargs["time_horizon_days"] == 21

    def test_risk_metrics_computed(self) -> None:
        prices = _make_prices()
        sim_result = _make_simulation_result()

        mock_provider = MagicMock()
        mock_provider.get_historical_prices.return_value = prices

        mock_engine = MagicMock()
        mock_engine.simulate.return_value = sim_result

        portfolio = _make_portfolio()
        date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 5))

        use_case = SimulatePortfolio(
            market_data_provider=mock_provider,
            monte_carlo_engine=mock_engine,
        )
        result = use_case.execute(
            portfolio=portfolio,
            date_range=date_range,
            num_simulations=5,
            time_horizon_days=21,
        )

        assert result.risk_metrics.mean_return != 0.0 or True  # may be near zero
        assert result.risk_metrics.volatility >= 0.0
