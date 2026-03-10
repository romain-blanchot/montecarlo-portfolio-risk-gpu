import pytest

from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.domain.value_objects.currency import Currency
from portfolio_risk_engine.infrastructure.market_data.yahoo_finance_market_data_provider import (
    YahooFinanceMarketDataProvider,
)


@pytest.mark.integration
class TestGetAssetIntegration:
    def setup_method(self):
        self.provider = YahooFinanceMarketDataProvider()

    def test_aapl_returns_usd_asset(self):
        asset = self.provider.get_asset(Ticker("AAPL"))

        assert asset.ticker == Ticker("AAPL")
        assert asset.currency == Currency("USD")
        assert asset.name is not None

    def test_msft_returns_usd_asset(self):
        asset = self.provider.get_asset(Ticker("MSFT"))

        assert asset.ticker == Ticker("MSFT")
        assert asset.currency == Currency("USD")

    def test_european_ticker_returns_eur(self):
        asset = self.provider.get_asset(Ticker("MC.PA"))

        assert asset.ticker == Ticker("MC.PA")
        assert asset.currency == Currency("EUR")

    def test_multiple_assets_have_distinct_names(self):
        aapl = self.provider.get_asset(Ticker("AAPL"))
        msft = self.provider.get_asset(Ticker("MSFT"))

        assert aapl.name != msft.name
