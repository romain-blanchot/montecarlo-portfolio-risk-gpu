import math

from portfolio_risk_engine.domain.models.historical_prices import HistoricalPrices
from portfolio_risk_engine.domain.models.historical_returns import HistoricalReturns


class ComputeLogReturns:
    @staticmethod
    def execute(prices: HistoricalPrices) -> HistoricalReturns:
        for ticker in prices.tickers:
            if len(prices.prices_by_ticker[ticker]) < 2:
                raise ValueError(
                    f"Need at least 2 prices to compute returns for {ticker.value}."
                )

        dates = prices.dates[1:]

        returns_by_ticker = {}
        for ticker in prices.tickers:
            ticker_prices = prices.prices_by_ticker[ticker]
            log_returns = tuple(
                math.log(ticker_prices[i] / ticker_prices[i - 1])
                for i in range(1, len(ticker_prices))
            )
            returns_by_ticker[ticker] = log_returns

        return HistoricalReturns(
            tickers=prices.tickers,
            dates=dates,
            returns_by_ticker=returns_by_ticker,
        )
