from unittest.mock import patch

from portfolio_risk_engine.cli import PortfolioSimulatorCLI, main


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


class TestMain:
    def test_main_exits_on_zero(self):
        with patch("builtins.input", return_value="0"):
            main()
