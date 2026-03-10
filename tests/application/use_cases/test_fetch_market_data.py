from datetime import date

import pytest

from portfolio_risk_engine.application.use_cases.fetch_market_data import (
    FetchMarketData,
)
from portfolio_risk_engine.domain.models.asset import Asset
from portfolio_risk_engine.domain.models.historical_prices import HistoricalPrices
from portfolio_risk_engine.domain.value_objects.currency import Currency
from portfolio_risk_engine.domain.value_objects.date_range import DateRange
from portfolio_risk_engine.domain.value_objects.ticker import Ticker


AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")
DATES = (date(2024, 1, 1), date(2024, 1, 2))
DATE_RANGE = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 3))


class FakeMarketDataProvider:
    def __init__(
        self,
        prices: HistoricalPrices | None = None,
    ) -> None:
        self._prices = prices or HistoricalPrices(
            tickers=(AAPL,),
            dates=DATES,
            prices_by_ticker={AAPL: (150.0, 151.0)},
        )
        self.call_count = 0

    def get_asset(self, ticker: Ticker) -> Asset:
        return Asset(ticker=ticker, currency=Currency("USD"))

    def get_historical_prices(
        self,
        tickers: tuple[Ticker, ...],
        date_range: DateRange,
    ) -> HistoricalPrices:
        self.call_count += 1
        return self._prices


class TestFetchMarketDataExecute:
    def test_returns_historical_prices(self):
        provider = FakeMarketDataProvider()
        use_case = FetchMarketData(provider)

        result = use_case.execute(tickers=(AAPL,), date_range=DATE_RANGE)

        assert result.tickers == (AAPL,)
        assert result.dates == DATES
        assert result.prices_by_ticker[AAPL] == (150.0, 151.0)

    def test_delegates_to_provider(self):
        provider = FakeMarketDataProvider()
        use_case = FetchMarketData(provider)

        use_case.execute(tickers=(AAPL,), date_range=DATE_RANGE)

        assert provider.call_count == 1

    def test_multiple_tickers(self):
        prices = HistoricalPrices(
            tickers=(AAPL, MSFT),
            dates=DATES,
            prices_by_ticker={
                AAPL: (150.0, 151.0),
                MSFT: (300.0, 301.0),
            },
        )
        provider = FakeMarketDataProvider(prices=prices)
        use_case = FetchMarketData(provider)

        result = use_case.execute(tickers=(AAPL, MSFT), date_range=DATE_RANGE)

        assert len(result.tickers) == 2


class TestFetchMarketDataValidation:
    def test_empty_tickers_raises(self):
        provider = FakeMarketDataProvider()
        use_case = FetchMarketData(provider)

        with pytest.raises(ValueError, match="Tickers list cannot be empty"):
            use_case.execute(tickers=(), date_range=DATE_RANGE)

    def test_empty_tickers_does_not_call_provider(self):
        provider = FakeMarketDataProvider()
        use_case = FetchMarketData(provider)

        with pytest.raises(ValueError):
            use_case.execute(tickers=(), date_range=DATE_RANGE)

        assert provider.call_count == 0
