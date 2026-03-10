from dataclasses import FrozenInstanceError

import pytest

from portfolio_risk_engine.domain.models.position import Position
from portfolio_risk_engine.domain.models.asset import Asset
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.domain.value_objects.currency import Currency
from portfolio_risk_engine.domain.value_objects.weight import Weight


class TestPositionCreation:
    def test_valid_position(self):
        asset = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"))
        weight = Weight(0.5)
        pos = Position(asset=asset, weight=weight)
        assert pos.asset == asset
        assert pos.weight == weight

    def test_zero_weight(self):
        asset = Asset(ticker=Ticker("MSFT"), currency=Currency("USD"))
        pos = Position(asset=asset, weight=Weight(0.0))
        assert pos.weight.value == pytest.approx(0.0)

    def test_full_weight(self):
        asset = Asset(ticker=Ticker("GOOG"), currency=Currency("USD"))
        pos = Position(asset=asset, weight=Weight(1.0))
        assert pos.weight.value == pytest.approx(1.0)


class TestPositionImmutability:
    def test_frozen(self):
        asset = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"))
        pos = Position(asset=asset, weight=Weight(0.5))

        with pytest.raises(FrozenInstanceError):
            pos.asset = Asset(ticker=Ticker("MSFT"), currency=Currency("EUR"))  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            pos.weight = Weight(0.3)  # type: ignore[misc]


class TestPositionEquality:
    def test_equal_positions(self):
        asset = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"))
        p1 = Position(asset=asset, weight=Weight(0.5))
        p2 = Position(asset=asset, weight=Weight(0.5))
        assert p1 == p2

    def test_different_asset(self):
        a1 = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"))
        a2 = Asset(ticker=Ticker("MSFT"), currency=Currency("USD"))
        p1 = Position(asset=a1, weight=Weight(0.5))
        p2 = Position(asset=a2, weight=Weight(0.5))
        assert p1 != p2

    def test_different_weight(self):
        asset = Asset(ticker=Ticker("AAPL"), currency=Currency("USD"))
        p1 = Position(asset=asset, weight=Weight(0.3))
        p2 = Position(asset=asset, weight=Weight(0.7))
        assert p1 != p2
