import pytest
from portfolio_risk_engine.domain.entities.asset import Asset


def test_asset_creation():
    asset = Asset("AAPL", "USD", "Apple")

    assert asset.ticker == "AAPL"
    assert asset.currency == "USD"
    assert asset.name == "Apple"


def test_asset_strips_and_uppercases():
    asset = Asset(" aapl ", " usd ")

    assert asset.ticker == "AAPL"
    assert asset.currency == "USD"


def test_invalid_ticker():
    with pytest.raises(ValueError):
        Asset("!!!", "USD")


def test_invalid_currency():
    with pytest.raises(ValueError):
        Asset("AAPL", "US")  # trop court


def test_empty_ticker():
    with pytest.raises(ValueError):
        Asset("   ", "USD")