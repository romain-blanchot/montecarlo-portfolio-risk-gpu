from datetime import timedelta

import pandas as pd
import yfinance as yf

from portfolio_risk_engine.domain.models.asset import Asset
from portfolio_risk_engine.domain.models.historical_prices import HistoricalPrices
from portfolio_risk_engine.domain.value_objects.currency import Currency
from portfolio_risk_engine.domain.value_objects.date_range import DateRange
from portfolio_risk_engine.domain.value_objects.ticker import Ticker


class YahooFinanceMarketDataProvider:
    def get_asset(self, ticker: Ticker) -> Asset:
        yf_ticker = yf.Ticker(ticker.value)
        info = yf_ticker.info

        if not info or info.get("trailingPegRatio") is None and len(info) <= 1:
            raise ValueError(f"Ticker not found on Yahoo Finance: {ticker.value}")

        currency_str = info.get("currency")
        if not currency_str:
            raise ValueError(f"Currency not available for ticker: {ticker.value}")

        name = info.get("shortName") or info.get("longName")

        return Asset(
            ticker=ticker,
            currency=Currency(currency_str),
            name=name,
        )

    def get_historical_prices(
        self,
        tickers: tuple[Ticker, ...],
        date_range: DateRange,
    ) -> HistoricalPrices:
        symbols = [t.value for t in tickers]

        # yfinance end date is exclusive, add 1 day to include end date
        end_exclusive = date_range.end + timedelta(days=1)

        df: pd.DataFrame = yf.download(
            symbols,
            start=date_range.start.isoformat(),
            end=end_exclusive.isoformat(),
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            raise ValueError(
                f"No price data returned for tickers: {', '.join(symbols)}"
            )

        # Normalize: yfinance returns different structures for 1 vs N tickers
        close = self._extract_close(df, symbols)

        # Drop dates with any NaN across requested tickers
        close = close.dropna()

        if close.empty:
            raise ValueError(
                "No complete price data after removing missing values "
                f"for tickers: {', '.join(symbols)}"
            )

        dates = tuple(d.date() for d in close.index)
        prices_by_ticker = {
            Ticker(symbol): tuple(close[symbol].tolist()) for symbol in symbols
        }

        return HistoricalPrices(
            tickers=tickers,
            dates=dates,
            prices_by_ticker=prices_by_ticker,
        )

    @staticmethod
    def _extract_close(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            # Multiple tickers: columns are (metric, ticker)
            close = df["Close"][symbols]
        else:
            # Single ticker: columns are just metrics
            close = df[["Close"]].rename(columns={"Close": symbols[0]})
        return close
