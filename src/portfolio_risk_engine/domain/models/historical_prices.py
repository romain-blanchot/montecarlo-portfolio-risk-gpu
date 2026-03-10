from dataclasses import dataclass
from datetime import date

from portfolio_risk_engine.domain.value_objects.ticker import Ticker


@dataclass(frozen=True)
class HistoricalPrices:
    tickers: tuple[Ticker, ...]
    dates: tuple[date, ...]
    prices_by_ticker: dict[Ticker, tuple[float, ...]]

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("HistoricalPrices must contain at least one ticker.")

        if not self.dates:
            raise ValueError("HistoricalPrices must contain at least one date.")

        if len(self.dates) != len(set(self.dates)):
            raise ValueError("Dates must be unique.")

        if self.dates != tuple(sorted(self.dates)):
            raise ValueError("Dates must be in chronological order.")

        if set(self.prices_by_ticker.keys()) != set(self.tickers):
            raise ValueError("prices_by_ticker keys must match tickers.")

        expected_length = len(self.dates)

        for ticker in self.tickers:
            prices = self.prices_by_ticker[ticker]

            if len(prices) != expected_length:
                raise ValueError(
                    f"Price series length mismatch for ticker {ticker.value}."
                )

            if any(price <= 0 for price in prices):
                raise ValueError(
                    f"Historical prices must be strictly positive for ticker {ticker.value}."
                )
