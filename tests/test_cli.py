import json
from unittest.mock import MagicMock, patch

from portfolio_risk_engine.cli import PortfolioSimulatorCLI, _run_scenario, main


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


class TestRunScenario:
    def test_run_scenario_with_valid_json(self, capsys) -> None:
        mock_result = MagicMock()
        mock_result.portfolio = MagicMock()
        mock_result.historical_prices = MagicMock()
        mock_result.market_parameters = MagicMock()
        mock_result.simulation_result = MagicMock()
        mock_result.risk_metrics = MagicMock()

        with (
            patch("portfolio_risk_engine.cli.SimulatePortfolio") as mock_cls,
            patch(
                "portfolio_risk_engine.cli.render_scenario",
                return_value="RENDERED OUTPUT",
            ),
        ):
            mock_cls.return_value.execute.return_value = mock_result

            raw = json.dumps(
                {
                    "assets": [
                        {"ticker": "AAPL", "weight": 0.6},
                        {"ticker": "MSFT", "weight": 0.4},
                    ],
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "num_simulations": 100,
                    "time_horizon_days": 5,
                }
            )
            _run_scenario(raw)

        captured = capsys.readouterr()
        assert "RENDERED OUTPUT" in captured.out

    def test_run_scenario_uses_defaults(self) -> None:
        mock_result = MagicMock()

        with (
            patch("portfolio_risk_engine.cli.SimulatePortfolio") as mock_cls,
            patch(
                "portfolio_risk_engine.cli.render_scenario",
                return_value="",
            ),
        ):
            mock_cls.return_value.execute.return_value = mock_result

            raw = json.dumps(
                {
                    "assets": [{"ticker": "AAPL", "weight": 1.0}],
                    "start_date": "2024-01-01",
                    "end_date": "2024-06-01",
                }
            )
            _run_scenario(raw)

            call_kwargs = mock_cls.return_value.execute.call_args.kwargs
            assert call_kwargs["num_simulations"] == 10_000
            assert call_kwargs["time_horizon_days"] == 21
            assert call_kwargs["model"] == "gbm"

    def test_run_scenario_with_model_selection(self, capsys) -> None:
        mock_result = MagicMock()

        with (
            patch("portfolio_risk_engine.cli.SimulatePortfolio") as mock_cls,
            patch(
                "portfolio_risk_engine.cli.render_scenario",
                return_value="RENDERED",
            ),
        ):
            mock_cls.return_value.execute.return_value = mock_result

            raw = json.dumps(
                {
                    "assets": [{"ticker": "AAPL", "weight": 1.0}],
                    "start_date": "2024-01-01",
                    "end_date": "2024-06-01",
                    "model": "heston",
                }
            )
            _run_scenario(raw)

            call_kwargs = mock_cls.return_value.execute.call_args.kwargs
            assert call_kwargs["model"] == "heston"

        captured = capsys.readouterr()
        assert "heston" in captured.out

    def test_run_scenario_invalid_model(self, capsys) -> None:
        raw = json.dumps(
            {
                "assets": [{"ticker": "AAPL", "weight": 1.0}],
                "start_date": "2024-01-01",
                "end_date": "2024-06-01",
                "model": "invalid_model",
            }
        )
        _run_scenario(raw)

        captured = capsys.readouterr()
        assert "Unknown model" in captured.out

    def test_run_scenario_invalid_json_raises(self) -> None:
        import pytest

        with pytest.raises(json.JSONDecodeError):
            _run_scenario("not valid json")


class TestMain:
    def test_main_exits_on_zero(self):
        with (
            patch("builtins.input", return_value="0"),
            patch("sys.argv", ["portfolio-sim"]),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = True
            main()

    def test_main_json_arg(self, capsys) -> None:
        raw = json.dumps(
            {
                "assets": [{"ticker": "AAPL", "weight": 1.0}],
                "start_date": "2024-01-01",
                "end_date": "2024-06-01",
            }
        )
        with (
            patch("sys.argv", ["portfolio-sim", raw]),
            patch("portfolio_risk_engine.cli._run_scenario") as mock_run,
        ):
            main()
            mock_run.assert_called_once_with(raw)
