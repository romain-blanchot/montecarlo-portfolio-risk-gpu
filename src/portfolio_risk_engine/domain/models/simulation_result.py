from dataclasses import dataclass

from portfolio_risk_engine.domain.value_objects.ticker import Ticker


@dataclass(frozen=True)
class MonteCarloSimulationResult:
    tickers: tuple[Ticker, ...]
    initial_prices: tuple[float, ...]
    terminal_prices: dict[Ticker, tuple[float, ...]]
    num_simulations: int
    time_horizon_days: int

    def __post_init__(self) -> None:
        n = len(self.tickers)

        if n == 0:
            raise ValueError("Must contain at least one ticker.")

        if len(self.initial_prices) != n:
            raise ValueError(
                f"initial_prices length ({len(self.initial_prices)}) "
                f"must match number of tickers ({n})."
            )

        for price in self.initial_prices:
            if price <= 0:
                raise ValueError("All initial prices must be positive.")

        if set(self.terminal_prices.keys()) != set(self.tickers):
            raise ValueError("terminal_prices keys must match tickers.")

        for ticker in self.tickers:
            if len(self.terminal_prices[ticker]) != self.num_simulations:
                raise ValueError(
                    f"terminal_prices for {ticker.value} has "
                    f"{len(self.terminal_prices[ticker])} values, "
                    f"expected {self.num_simulations}."
                )

        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive.")

        if self.time_horizon_days <= 0:
            raise ValueError("time_horizon_days must be positive.")
