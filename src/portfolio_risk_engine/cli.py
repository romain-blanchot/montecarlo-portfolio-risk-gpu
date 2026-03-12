from __future__ import annotations

from datetime import date

from portfolio_risk_engine.application.use_cases.compute_log_returns import (
    ComputeLogReturns,
)
from portfolio_risk_engine.application.use_cases.compute_portfolio_risk import (
    ComputePortfolioRisk,
)
from portfolio_risk_engine.application.use_cases.estimate_market_parameters import (
    EstimateMarketParameters,
)
from portfolio_risk_engine.application.use_cases.fetch_market_data import (
    FetchMarketData,
)
from portfolio_risk_engine.application.use_cases.run_monte_carlo import RunMonteCarlo
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
from portfolio_risk_engine.domain.value_objects.date_range import DateRange
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.domain.value_objects.weight import Weight
from portfolio_risk_engine.infrastructure.market_data.yahoo_finance_market_data_provider import (
    YahooFinanceMarketDataProvider,
)
from portfolio_risk_engine.infrastructure.simulation.cpu_monte_carlo_engine import (
    CpuMonteCarloEngine,
)

try:
    from portfolio_risk_engine.infrastructure.simulation.gpu_accelerated_pipeline import (
        GpuAcceleratedPipeline,
    )

    _GPU_AVAILABLE = True
except (ImportError, RuntimeError):
    _GPU_AVAILABLE = False


def _build_menu() -> str:
    lines = [
        "",
        "============================================",
        "  Portfolio Monte Carlo Risk Simulator",
        "============================================",
        "",
        "  1. Define portfolio",
        "  2. Fetch market data",
        "  3. Estimate parameters",
        "  4. Run Monte Carlo simulation",
        "  5. Compute risk metrics",
        "  6. Full pipeline (CPU)",
    ]
    if _GPU_AVAILABLE:
        lines.append("  7. Full pipeline (GPU accelerated)")
    lines += ["  0. Exit", ""]
    return "\n".join(lines)


MENU = _build_menu()


class PortfolioSimulatorCLI:
    def __init__(self) -> None:
        self._provider = YahooFinanceMarketDataProvider()
        self._engine = CpuMonteCarloEngine()

        self.portfolio: Portfolio | None = None
        self.prices: HistoricalPrices | None = None
        self.market_params: MarketParameters | None = None
        self.simulation_result: MonteCarloSimulationResult | None = None
        self.risk_metrics: PortfolioRiskMetrics | None = None

    def run(self) -> None:
        from collections.abc import Callable

        actions: dict[str, Callable[[], None]] = {
            "1": self.define_portfolio,
            "2": self.fetch_market_data,
            "3": self.estimate_parameters,
            "4": self.run_simulation,
            "5": self.compute_risk,
            "6": self.full_pipeline,
        }
        if _GPU_AVAILABLE:
            actions["7"] = self.full_pipeline_gpu

        while True:
            print(MENU)
            self._print_state()
            choice = input("Select an option: ").strip()

            if choice == "0":
                print("Goodbye.")
                break

            action = actions.get(choice)
            if action is None:
                print("Invalid option.")
                continue

            try:
                action()
            except KeyboardInterrupt:
                print("\nCancelled.")
            except (ValueError, RuntimeError, OSError) as e:
                print(f"\nError: {e}")

    def _print_state(self) -> None:
        flags = []
        if self.portfolio:
            tickers = ", ".join(t.value for t in self.portfolio.tickers)
            flags.append(f"Portfolio: [{tickers}]")
        if self.prices:
            flags.append(f"Prices: {len(self.prices.dates)} obs")
        if self.market_params:
            flags.append("Params: ready")
        if self.simulation_result:
            flags.append(f"Simulation: {self.simulation_result.num_simulations} paths")
        if self.risk_metrics:
            flags.append("Risk: computed")
        if _GPU_AVAILABLE:
            flags.append("GPU: available")
        if flags:
            print("  State: " + " | ".join(flags))
            print()

    def define_portfolio(self) -> None:
        print("\n--- Define Portfolio ---")
        raw_tickers = input("Tickers (comma-separated, e.g. AAPL,MSFT,GOOGL): ").strip()
        if not raw_tickers:
            print("No tickers provided.")
            return

        ticker_strs = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        tickers = [Ticker(t) for t in ticker_strs]

        raw_weights = input(
            f"Weights for {ticker_strs} (comma-separated, must sum to 1): "
        ).strip()
        weights = [float(w.strip()) for w in raw_weights.split(",")]

        if len(weights) != len(tickers):
            print("Number of weights must match number of tickers.")
            return

        positions = tuple(
            Position(
                asset=Asset(ticker=t, currency=Currency("USD")),
                weight=Weight(w),
            )
            for t, w in zip(tickers, weights)
        )

        self.portfolio = Portfolio(positions=positions)
        self.prices = None
        self.market_params = None
        self.simulation_result = None
        self.risk_metrics = None

        print(f"Portfolio defined: {[t.value for t in self.portfolio.tickers]}")
        print(f"Weights: {self.portfolio.weights}")

    def fetch_market_data(self) -> None:
        if not self.portfolio:
            print("Define a portfolio first (option 1).")
            return

        print("\n--- Fetch Market Data ---")
        start_str = input("Start date (YYYY-MM-DD): ").strip()
        end_str = input("End date (YYYY-MM-DD): ").strip()

        start = date.fromisoformat(start_str)
        end = date.fromisoformat(end_str)
        date_range = DateRange(start=start, end=end)

        tickers = tuple(self.portfolio.tickers)
        fetch = FetchMarketData(self._provider)
        self.prices = fetch.execute(tickers=tickers, date_range=date_range)

        self.market_params = None
        self.simulation_result = None
        self.risk_metrics = None

        print(f"Fetched {len(self.prices.dates)} price observations.")

    def estimate_parameters(self) -> None:
        if not self.prices:
            print("Fetch market data first (option 2).")
            return

        print("\n--- Estimate Parameters ---")
        returns = ComputeLogReturns.execute(self.prices)
        self.market_params = EstimateMarketParameters().execute(returns)

        self.simulation_result = None
        self.risk_metrics = None

        print(f"Annualization factor: {self.market_params.annualization_factor}")
        print("Drift vector:")
        for ticker, drift in zip(
            self.market_params.tickers, self.market_params.drift_vector
        ):
            print(f"  {ticker.value}: {drift:+.6f}")
        print("Covariance matrix:")
        header = "         " + "  ".join(
            f"{t.value:>10}" for t in self.market_params.tickers
        )
        print(header)
        for ticker, row in zip(
            self.market_params.tickers, self.market_params.covariance_matrix
        ):
            vals = "  ".join(f"{v:>10.6f}" for v in row)
            print(f"  {ticker.value:>5}  {vals}")

    def run_simulation(self) -> None:
        if not self.market_params or not self.prices:
            print("Estimate parameters first (option 3).")
            return

        print("\n--- Run Monte Carlo Simulation ---")
        n_sims = int(input("Number of simulations [10000]: ").strip() or "10000")
        t_days = int(input("Time horizon in trading days [21]: ").strip() or "21")

        initial_prices = tuple(
            self.prices.prices_by_ticker[t][-1] for t in self.market_params.tickers
        )

        use_case = RunMonteCarlo(engine=self._engine)
        self.simulation_result = use_case.execute(
            market_params=self.market_params,
            initial_prices=initial_prices,
            num_simulations=n_sims,
            time_horizon_days=t_days,
        )

        self.risk_metrics = None

        print(f"Simulated {n_sims} paths over {t_days} trading days.")
        for ticker in self.simulation_result.tickers:
            prices = self.simulation_result.terminal_prices[ticker]
            s0 = dict(
                zip(
                    self.simulation_result.tickers,
                    self.simulation_result.initial_prices,
                )
            )[ticker]
            mean_p = sum(prices) / len(prices)
            print(f"  {ticker.value}: S0={s0:.2f}, mean(S_T)={mean_p:.2f}")

    def compute_risk(self) -> None:
        if not self.portfolio or not self.simulation_result:
            print("Run simulation first (option 4).")
            return

        print("\n--- Compute Risk Metrics ---")
        self.risk_metrics = ComputePortfolioRisk.execute(
            self.portfolio, self.simulation_result
        )
        self._print_risk_metrics()

    def full_pipeline(self) -> None:
        print("\n--- Full Pipeline (CPU) ---")

        if not self.portfolio:
            self.define_portfolio()
            if not self.portfolio:
                return

        if not self.prices:
            self.fetch_market_data()
            if not self.prices:
                return

        self.estimate_parameters()
        self.run_simulation()
        self.compute_risk()

    def full_pipeline_gpu(self) -> None:  # pragma: no cover
        print("\n--- Full Pipeline (GPU accelerated) ---")

        if not self.portfolio:
            self.define_portfolio()
            if not self.portfolio:
                return

        if not self.prices:
            self.fetch_market_data()
            if not self.prices:
                return

        if not self.market_params:
            self.estimate_parameters()
            if not self.market_params:
                return

        n_sims = int(input("Number of simulations [10000]: ").strip() or "10000")
        t_days = int(input("Time horizon in trading days [21]: ").strip() or "21")

        initial_prices = tuple(
            self.prices.prices_by_ticker[t][-1] for t in self.market_params.tickers
        )
        weights = tuple(self.portfolio.weights)

        pipeline = GpuAcceleratedPipeline()
        self.risk_metrics, summary = pipeline.run_with_summary(
            market_params=self.market_params,
            initial_prices=initial_prices,
            weights=weights,
            num_simulations=n_sims,
            time_horizon_days=t_days,
        )

        self.simulation_result = None  # no per-path data in GPU mode

        print(f"Simulated {n_sims} paths over {t_days} trading days (GPU).")
        ip_map = dict(zip(self.market_params.tickers, initial_prices))
        for ticker_str, mean_p in summary.items():
            s0 = ip_map.get(Ticker(ticker_str))
            print(f"  {ticker_str}: S0={s0:.2f}, mean(S_T)={mean_p:.2f}")

        self._print_risk_metrics()

    def _print_risk_metrics(self) -> None:
        if not self.risk_metrics:
            return
        print(f"  Mean return:            {self.risk_metrics.mean_return:+.4%}")
        print(f"  Volatility:             {self.risk_metrics.volatility:.4%}")
        print(f"  VaR 95%:                {self.risk_metrics.var_95:.4%}")
        print(f"  VaR 99%:                {self.risk_metrics.var_99:.4%}")
        print(f"  Expected Shortfall 95%: {self.risk_metrics.es_95:.4%}")
        print(f"  Expected Shortfall 99%: {self.risk_metrics.es_99:.4%}")


def main() -> None:
    cli = PortfolioSimulatorCLI()
    cli.run()
