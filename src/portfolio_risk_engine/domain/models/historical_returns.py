from dataclasses import dataclass
from datetime import date

from portfolio_risk_engine.domain.value_objects.ticker import Ticker


@dataclass(frozen=True)
class HistoricalReturns:
    tickers: tuple[Ticker, ...]
    dates: tuple[date, ...]
    returns_by_ticker: dict[Ticker, tuple[float, ...]]

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("HistoricalReturns must contain at least one ticker.")

        if not self.dates:
            raise ValueError("HistoricalReturns must contain at least one date.")

        if len(self.dates) != len(set(self.dates)):
            raise ValueError("Dates must be unique.")

        if self.dates != tuple(sorted(self.dates)):
            raise ValueError("Dates must be in chronological order.")

        if set(self.returns_by_ticker.keys()) != set(self.tickers):
            raise ValueError("returns_by_ticker keys must match tickers.")

        expected_length = len(self.dates)

        for ticker in self.tickers:
            returns = self.returns_by_ticker[ticker]

            if len(returns) != expected_length:
                raise ValueError(
                    f"Returns series length mismatch for ticker {ticker.value}."
                )
