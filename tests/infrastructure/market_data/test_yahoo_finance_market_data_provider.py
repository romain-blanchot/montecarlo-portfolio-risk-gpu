from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portfolio_risk_engine.domain.models.asset import Asset
from portfolio_risk_engine.domain.value_objects.currency import Currency
from portfolio_risk_engine.domain.value_objects.date_range import DateRange
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.infrastructure.market_data.yahoo_finance_market_data_provider import (
    YahooFinanceMarketDataProvider,
)

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")
DATE_RANGE = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 5))


# ---------------------------------------------------------------------------
# get_asset
# ---------------------------------------------------------------------------
class TestGetAsset:
    @patch("yfinance.Ticker")
    def test_returns_asset_with_currency_and_name(self, mock_ticker_cls: MagicMock):
        mock_ticker_cls.return_value.info = {
            "currency": "USD",
            "shortName": "Apple Inc.",
            "trailingPegRatio": 1.5,
        }
        provider = YahooFinanceMarketDataProvider()

        asset = provider.get_asset(AAPL)

        assert asset == Asset(ticker=AAPL, currency=Currency("USD"), name="Apple Inc.")

    @patch("yfinance.Ticker")
    def test_returns_asset_with_name_none(self, mock_ticker_cls: MagicMock):
        mock_ticker_cls.return_value.info = {
            "currency": "EUR",
            "trailingPegRatio": 1.0,
        }
        provider = YahooFinanceMarketDataProvider()

        asset = provider.get_asset(MSFT)

        assert asset.name is None
        assert asset.currency == Currency("EUR")

    @patch("yfinance.Ticker")
    def test_falls_back_to_long_name(self, mock_ticker_cls: MagicMock):
        mock_ticker_cls.return_value.info = {
            "currency": "USD",
            "longName": "Apple Inc. Long",
            "trailingPegRatio": 1.5,
        }
        provider = YahooFinanceMarketDataProvider()

        asset = provider.get_asset(AAPL)

        assert asset.name == "Apple Inc. Long"

    @patch("yfinance.Ticker")
    def test_raises_when_currency_missing(self, mock_ticker_cls: MagicMock):
        mock_ticker_cls.return_value.info = {"trailingPegRatio": 1.5}
        provider = YahooFinanceMarketDataProvider()

        with pytest.raises(ValueError, match="Currency not available"):
            provider.get_asset(AAPL)


# ---------------------------------------------------------------------------
# get_historical_prices
# ---------------------------------------------------------------------------
def _make_single_ticker_df() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        name="Date",
    )
    return pd.DataFrame(
        {"Close": [150.0, 151.0, 152.0], "Volume": [1000, 1100, 1200]},
        index=index,
    )


def _make_multi_ticker_df() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        name="Date",
    )
    arrays = [
        ["Close", "Close", "Volume", "Volume"],
        ["AAPL", "MSFT", "AAPL", "MSFT"],
    ]
    columns = pd.MultiIndex.from_arrays(arrays)
    data = [
        [150.0, 300.0, 1000, 2000],
        [151.0, 301.0, 1100, 2100],
        [152.0, 302.0, 1200, 2200],
    ]
    return pd.DataFrame(data, index=index, columns=columns)


def _make_df_with_nan() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        name="Date",
    )
    return pd.DataFrame(
        {"Close": [150.0, float("nan"), 152.0], "Volume": [1000, 1100, 1200]},
        index=index,
    )


class TestGetHistoricalPricesSingleTicker:
    @patch("yfinance.download")
    def test_returns_historical_prices(self, mock_download: MagicMock):
        mock_download.return_value = _make_single_ticker_df()
        provider = YahooFinanceMarketDataProvider()

        result = provider.get_historical_prices(tickers=(AAPL,), date_range=DATE_RANGE)

        assert result.tickers == (AAPL,)
        assert result.dates == (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
        assert result.prices_by_ticker[AAPL] == (150.0, 151.0, 152.0)

    @patch("yfinance.download")
    def test_dates_are_sorted(self, mock_download: MagicMock):
        mock_download.return_value = _make_single_ticker_df()
        provider = YahooFinanceMarketDataProvider()

        result = provider.get_historical_prices(tickers=(AAPL,), date_range=DATE_RANGE)

        assert result.dates == tuple(sorted(result.dates))


class TestGetHistoricalPricesMultipleTickers:
    @patch("yfinance.download")
    def test_returns_all_tickers(self, mock_download: MagicMock):
        mock_download.return_value = _make_multi_ticker_df()
        provider = YahooFinanceMarketDataProvider()

        result = provider.get_historical_prices(
            tickers=(AAPL, MSFT), date_range=DATE_RANGE
        )

        assert result.tickers == (AAPL, MSFT)
        assert result.prices_by_ticker[AAPL] == (150.0, 151.0, 152.0)
        assert result.prices_by_ticker[MSFT] == (300.0, 301.0, 302.0)


class TestGetHistoricalPricesNanHandling:
    @patch("yfinance.download")
    def test_drops_dates_with_nan(self, mock_download: MagicMock):
        mock_download.return_value = _make_df_with_nan()
        provider = YahooFinanceMarketDataProvider()

        result = provider.get_historical_prices(tickers=(AAPL,), date_range=DATE_RANGE)

        assert date(2024, 1, 3) not in result.dates
        assert len(result.dates) == 2
        assert result.prices_by_ticker[AAPL] == (150.0, 152.0)


class TestGetHistoricalPricesErrors:
    @patch("yfinance.download")
    def test_raises_when_empty_dataframe(self, mock_download: MagicMock):
        mock_download.return_value = pd.DataFrame()
        provider = YahooFinanceMarketDataProvider()

        with pytest.raises(ValueError, match="No price data returned"):
            provider.get_historical_prices(tickers=(AAPL,), date_range=DATE_RANGE)

    @patch("yfinance.download")
    def test_raises_when_all_nan(self, mock_download: MagicMock):
        index = pd.DatetimeIndex([date(2024, 1, 2)], name="Date")
        df = pd.DataFrame(
            {"Close": [float("nan")]},
            index=index,
        )
        mock_download.return_value = df
        provider = YahooFinanceMarketDataProvider()

        with pytest.raises(ValueError, match="No complete price data"):
            provider.get_historical_prices(tickers=(AAPL,), date_range=DATE_RANGE)


# ---------------------------------------------------------------------------
# Integration test (requires network — run manually)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestYahooFinanceIntegration:
    def test_get_asset_aapl(self):
        provider = YahooFinanceMarketDataProvider()
        asset = provider.get_asset(Ticker("AAPL"))

        assert asset.ticker == Ticker("AAPL")
        assert asset.currency == Currency("USD")
        assert asset.name is not None

    def test_get_historical_prices_aapl(self):
        provider = YahooFinanceMarketDataProvider()
        dr = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 10))

        result = provider.get_historical_prices(
            tickers=(Ticker("AAPL"),), date_range=dr
        )

        assert len(result.dates) > 0
        assert all(p > 0 for p in result.prices_by_ticker[Ticker("AAPL")])
