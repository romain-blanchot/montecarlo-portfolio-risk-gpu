from datetime import date

import pytest

from portfolio_risk_engine.application.use_cases.fetch_market_data import (
    FetchMarketData,
)
from portfolio_risk_engine.domain.value_objects.date_range import DateRange
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.infrastructure.market_data.yahoo_finance_market_data_provider import (
    YahooFinanceMarketDataProvider,
)

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")


@pytest.mark.integration
class TestFetchMarketDataIntegration:
    def setup_method(self):
        provider = YahooFinanceMarketDataProvider()
        self.use_case = FetchMarketData(provider)

    def test_single_ticker(self):
        result = self.use_case.execute(
            tickers=(AAPL,),
            date_range=DateRange(start=date(2024, 1, 1), end=date(2024, 3, 1)),
        )

        assert result.tickers == (AAPL,)
        assert len(result.dates) > 0
        assert all(p > 0 for p in result.prices_by_ticker[AAPL])

    def test_multiple_tickers(self):
        result = self.use_case.execute(
            tickers=(AAPL, MSFT),
            date_range=DateRange(start=date(2024, 1, 1), end=date(2024, 3, 1)),
        )

        assert result.tickers == (AAPL, MSFT)
        assert AAPL in result.prices_by_ticker
        assert MSFT in result.prices_by_ticker
        assert len(result.prices_by_ticker[AAPL]) == len(result.dates)
        assert len(result.prices_by_ticker[MSFT]) == len(result.dates)

    def test_dates_are_chronological(self):
        result = self.use_case.execute(
            tickers=(AAPL,),
            date_range=DateRange(start=date(2024, 1, 1), end=date(2024, 3, 1)),
        )

        assert result.dates == tuple(sorted(result.dates))

    def test_dates_within_requested_range(self):
        start = date(2024, 1, 1)
        end = date(2024, 3, 1)
        result = self.use_case.execute(
            tickers=(AAPL,),
            date_range=DateRange(start=start, end=end),
        )

        assert all(start <= d <= end for d in result.dates)

    def test_prices_are_strictly_positive(self):
        result = self.use_case.execute(
            tickers=(AAPL, MSFT),
            date_range=DateRange(start=date(2024, 1, 1), end=date(2024, 3, 1)),
        )

        for ticker in result.tickers:
            assert all(p > 0 for p in result.prices_by_ticker[ticker])
