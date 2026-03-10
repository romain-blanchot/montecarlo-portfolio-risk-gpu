from dataclasses import FrozenInstanceError

import pytest

from portfolio_risk_engine.domain.models.asset import Asset
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.domain.value_objects.currency import Currency


class TestAssetCreation:
    def test_with_all_fields(self):
        asset = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"), name="Apple")
        assert asset.ticker == Ticker("AAPL")
        assert asset.currency == Currency("USD")
        assert asset.name == "Apple"

    def test_without_name(self):
        asset = Asset(ticker=Ticker("MSFT"), currency=Currency("EUR"))
        assert asset.ticker == Ticker("MSFT")
        assert asset.currency == Currency("EUR")
        assert asset.name is None

    def test_name_defaults_to_none(self):
        asset = Asset(ticker=Ticker("GOOG"), currency=Currency("GBP"))
        assert asset.name is None


class TestAssetImmutability:
    def test_frozen(self):
        asset = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"))
        with pytest.raises(FrozenInstanceError):
            asset.ticker = Ticker("MSFT") # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            asset.currency = Currency("EUR") # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            asset.name = "Changed" # type: ignore[misc]


class TestAssetEquality:
    def test_equal_assets(self):
        a1 = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"), name="Apple")
        a2 = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"), name="Apple")
        assert a1 == a2

    def test_different_ticker(self):
        a1 = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"))
        a2 = Asset(ticker=Ticker("MSFT"), currency=Currency("USD"))
        assert a1 != a2

    def test_different_currency(self):
        a1 = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"))
        a2 = Asset(ticker=Ticker("AAPL"), currency=Currency("EUR"))
        assert a1 != a2

    def test_different_name(self):
        a1 = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"), name="Apple")
        a2 = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"), name="Apple Inc.")
        assert a1 != a2


class TestAssetValidation:
    def test_invalid_ticker_propagates(self):
        with pytest.raises(ValueError):
            Asset(ticker=Ticker("!!!"), currency=Currency("USD"))

    def test_invalid_currency_propagates(self):
        with pytest.raises(ValueError):
            Asset(ticker=Ticker("AAPL"), currency=Currency("XX"))
