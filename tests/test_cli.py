from datetime import date
from unittest.mock import MagicMock, patch

from portfolio_risk_engine.cli import PortfolioSimulatorCLI, main
from portfolio_risk_engine.domain.models.historical_prices import HistoricalPrices
from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.models.portfolio_risk_metrics import (
    PortfolioRiskMetrics,
)
from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)
from portfolio_risk_engine.domain.value_objects.ticker import Ticker

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")


def _make_prices() -> HistoricalPrices:
    return HistoricalPrices(
        tickers=(AAPL, MSFT),
        dates=(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)),
        prices_by_ticker={
            AAPL: (150.0, 151.0, 152.0),
            MSFT: (300.0, 301.0, 302.0),
        },
    )


def _make_market_params() -> MarketParameters:
    return MarketParameters(
        tickers=(AAPL, MSFT),
        drift_vector=(0.05, 0.08),
        covariance_matrix=((0.04, 0.01), (0.01, 0.03)),
        annualization_factor=252,
    )


def _make_simulation_result() -> MonteCarloSimulationResult:
    return MonteCarloSimulationResult(
        tickers=(AAPL, MSFT),
        initial_prices=(152.0, 302.0),
        terminal_prices={
            AAPL: (155.0, 148.0, 160.0),
            MSFT: (310.0, 295.0, 320.0),
        },
        num_simulations=3,
        time_horizon_days=21,
    )


def _make_risk_metrics() -> PortfolioRiskMetrics:
    return PortfolioRiskMetrics(
        mean_return=0.02,
        volatility=0.15,
        var_95=0.10,
        var_99=0.18,
        es_95=0.12,
        es_99=0.20,
    )


def _make_cli_with_portfolio() -> PortfolioSimulatorCLI:
    """Create a CLI with a portfolio already defined."""
    inputs = iter(["1", "AAPL,MSFT", "0.6,0.4", "0"])
    with patch("builtins.input", side_effect=inputs):
        cli = PortfolioSimulatorCLI()
        cli.run()
    return cli


class TestPortfolioSimulatorCLI:
    def test_exit_immediately(self):
        with patch("builtins.input", return_value="0"):
            cli = PortfolioSimulatorCLI()
            cli.run()

    def test_invalid_option(self, capsys):
        inputs = iter(["9", "0"])
        with patch("builtins.input", side_effect=inputs):
            cli = PortfolioSimulatorCLI()
            cli.run()
        captured = capsys.readouterr()
        assert "Invalid option" in captured.out

    def test_fetch_without_portfolio(self, capsys):
        inputs = iter(["2", "0"])
        with patch("builtins.input", side_effect=inputs):
            cli = PortfolioSimulatorCLI()
            cli.run()
        captured = capsys.readouterr()
        assert "Define a portfolio first" in captured.out

    def test_estimate_without_prices(self, capsys):
        inputs = iter(["3", "0"])
        with patch("builtins.input", side_effect=inputs):
            cli = PortfolioSimulatorCLI()
            cli.run()
        captured = capsys.readouterr()
        assert "Fetch market data first" in captured.out

    def test_simulate_without_params(self, capsys):
        inputs = iter(["4", "0"])
        with patch("builtins.input", side_effect=inputs):
            cli = PortfolioSimulatorCLI()
            cli.run()
        captured = capsys.readouterr()
        assert "Estimate parameters first" in captured.out

    def test_risk_without_simulation(self, capsys):
        inputs = iter(["5", "0"])
        with patch("builtins.input", side_effect=inputs):
            cli = PortfolioSimulatorCLI()
            cli.run()
        captured = capsys.readouterr()
        assert "Run simulation first" in captured.out

    def test_define_portfolio(self, capsys):
        inputs = iter(["1", "AAPL,MSFT", "0.6,0.4", "0"])
        with patch("builtins.input", side_effect=inputs):
            cli = PortfolioSimulatorCLI()
            cli.run()
        captured = capsys.readouterr()
        assert "Portfolio defined" in captured.out
        assert cli.portfolio is not None

    def test_define_portfolio_empty_tickers(self, capsys):
        inputs = iter(["1", "", "0"])
        with patch("builtins.input", side_effect=inputs):
            cli = PortfolioSimulatorCLI()
            cli.run()
        captured = capsys.readouterr()
        assert "No tickers provided" in captured.out

    def test_define_portfolio_wrong_weight_count(self, capsys):
        inputs = iter(["1", "AAPL,MSFT", "0.5", "0"])
        with patch("builtins.input", side_effect=inputs):
            cli = PortfolioSimulatorCLI()
            cli.run()
        captured = capsys.readouterr()
        assert "Number of weights must match" in captured.out

    def test_define_portfolio_clears_state(self):
        inputs = iter(["1", "AAPL", "1.0", "0"])
        with patch("builtins.input", side_effect=inputs):
            cli = PortfolioSimulatorCLI()
            cli.run()
        assert cli.prices is None
        assert cli.market_params is None
        assert cli.simulation_result is None
        assert cli.risk_metrics is None

    def test_keyboard_interrupt_handled(self, capsys):
        call_count = 0

        def input_side_effect(prompt=""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "1"
            if call_count == 2:
                raise KeyboardInterrupt
            return "0"

        with patch("builtins.input", side_effect=input_side_effect):
            cli = PortfolioSimulatorCLI()
            cli.run()
        captured = capsys.readouterr()
        assert "Cancelled" in captured.out

    def test_error_handling(self, capsys):
        # Define portfolio with invalid weights to trigger ValueError
        inputs = iter(["1", "AAPL,MSFT", "0.5,0.6", "0"])
        with patch("builtins.input", side_effect=inputs):
            cli = PortfolioSimulatorCLI()
            cli.run()
        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_print_state_with_portfolio(self, capsys):
        inputs = iter(["1", "AAPL", "1.0", "0"])
        with patch("builtins.input", side_effect=inputs):
            cli = PortfolioSimulatorCLI()
            cli.run()
        captured = capsys.readouterr()
        assert "Portfolio:" in captured.out

    def test_full_pipeline_no_portfolio(self, capsys):
        # Full pipeline prompts for portfolio; give empty tickers to abort
        inputs = iter(["6", "", "0"])
        with patch("builtins.input", side_effect=inputs):
            cli = PortfolioSimulatorCLI()
            cli.run()
        captured = capsys.readouterr()
        assert "No tickers provided" in captured.out


class TestFetchMarketData:
    def test_fetch_market_data_success(self, capsys):
        mock_prices = _make_prices()
        inputs = iter(["2", "2024-01-01", "2024-06-01", "0"])
        with (
            patch("builtins.input", side_effect=inputs),
            patch("portfolio_risk_engine.cli.FetchMarketData") as MockFetch,
        ):
            cli = PortfolioSimulatorCLI()
            cli.portfolio = _make_cli_with_portfolio().portfolio
            MockFetch.return_value.execute.return_value = mock_prices
            cli.run()
        captured = capsys.readouterr()
        assert "Fetched 3 price observations" in captured.out
        assert cli.prices is not None


class TestEstimateParameters:
    def test_estimate_parameters_success(self, capsys):
        mock_params = _make_market_params()
        mock_returns = MagicMock()
        inputs = iter(["3", "0"])
        with (
            patch("builtins.input", side_effect=inputs),
            patch("portfolio_risk_engine.cli.ComputeLogReturns") as MockLogReturns,
            patch("portfolio_risk_engine.cli.EstimateMarketParameters") as MockEstimate,
        ):
            cli = PortfolioSimulatorCLI()
            cli.portfolio = _make_cli_with_portfolio().portfolio
            cli.prices = _make_prices()
            MockLogReturns.execute.return_value = mock_returns
            MockEstimate.return_value.execute.return_value = mock_params
            cli.run()
        captured = capsys.readouterr()
        assert "Annualization factor: 252" in captured.out
        assert "Drift vector" in captured.out
        assert "Covariance matrix" in captured.out
        assert cli.market_params is not None


class TestRunSimulation:
    def test_run_simulation_success(self, capsys):
        mock_sim = _make_simulation_result()
        inputs = iter(["4", "3", "21", "0"])
        with (
            patch("builtins.input", side_effect=inputs),
            patch("portfolio_risk_engine.cli.RunMonteCarlo") as MockRunMC,
        ):
            cli = PortfolioSimulatorCLI()
            cli.portfolio = _make_cli_with_portfolio().portfolio
            cli.prices = _make_prices()
            cli.market_params = _make_market_params()
            MockRunMC.return_value.execute.return_value = mock_sim
            cli.run()
        captured = capsys.readouterr()
        assert "Simulated 3 paths over 21 trading days" in captured.out
        assert "AAPL" in captured.out
        assert cli.simulation_result is not None


class TestComputeRisk:
    def test_compute_risk_success(self, capsys):
        mock_metrics = _make_risk_metrics()
        inputs = iter(["5", "0"])
        with (
            patch("builtins.input", side_effect=inputs),
            patch("portfolio_risk_engine.cli.ComputePortfolioRisk") as MockRisk,
        ):
            cli = PortfolioSimulatorCLI()
            cli.portfolio = _make_cli_with_portfolio().portfolio
            cli.prices = _make_prices()
            cli.market_params = _make_market_params()
            cli.simulation_result = _make_simulation_result()
            MockRisk.execute.return_value = mock_metrics
            cli.run()
        captured = capsys.readouterr()
        assert "Mean return" in captured.out
        assert "VaR 95%" in captured.out
        assert "VaR 99%" in captured.out
        assert "Expected Shortfall 95%" in captured.out
        assert cli.risk_metrics is not None


class TestPrintState:
    def test_print_state_all_fields(self, capsys):
        cli = PortfolioSimulatorCLI()
        cli.portfolio = _make_cli_with_portfolio().portfolio
        cli.prices = _make_prices()
        cli.market_params = _make_market_params()
        cli.simulation_result = _make_simulation_result()
        cli.risk_metrics = _make_risk_metrics()
        cli._print_state()
        captured = capsys.readouterr()
        assert "Portfolio:" in captured.out
        assert "Prices: 3 obs" in captured.out
        assert "Params: ready" in captured.out
        assert "Simulation: 3 paths" in captured.out
        assert "Risk: computed" in captured.out


class TestFullPipeline:
    def test_full_pipeline_with_existing_portfolio(self, capsys):
        mock_prices = _make_prices()
        mock_returns = MagicMock()
        mock_params = _make_market_params()
        mock_sim = _make_simulation_result()
        mock_metrics = _make_risk_metrics()

        inputs = iter(["6", "2024-01-01", "2024-06-01", "3", "21", "0"])
        with (
            patch("builtins.input", side_effect=inputs),
            patch("portfolio_risk_engine.cli.FetchMarketData") as MockFetch,
            patch("portfolio_risk_engine.cli.ComputeLogReturns") as MockLogReturns,
            patch("portfolio_risk_engine.cli.EstimateMarketParameters") as MockEstimate,
            patch("portfolio_risk_engine.cli.RunMonteCarlo") as MockRunMC,
            patch("portfolio_risk_engine.cli.ComputePortfolioRisk") as MockRisk,
        ):
            cli = PortfolioSimulatorCLI()
            cli.portfolio = _make_cli_with_portfolio().portfolio
            MockFetch.return_value.execute.return_value = mock_prices
            MockLogReturns.execute.return_value = mock_returns
            MockEstimate.return_value.execute.return_value = mock_params
            MockRunMC.return_value.execute.return_value = mock_sim
            MockRisk.execute.return_value = mock_metrics
            cli.run()
        captured = capsys.readouterr()
        assert "Full Pipeline" in captured.out
        assert "Fetched" in captured.out
        assert "Mean return" in captured.out


class TestMain:
    def test_main_exits_on_zero(self):
        with patch("builtins.input", return_value="0"):
            main()


class TestMainModule:
    def test_main_module_runs(self):
        with patch("portfolio_risk_engine.cli.main"):
            import importlib

            import portfolio_risk_engine.__main__ as main_mod

            importlib.reload(main_mod)
            # __main__.py only calls main() when __name__ == "__main__"
            # so we just verify the module imports without error
            assert hasattr(main_mod, "main")
