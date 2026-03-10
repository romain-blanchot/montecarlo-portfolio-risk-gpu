import pytest

from portfolio_risk_engine.domain.models.portfolio import Portfolio
from portfolio_risk_engine.domain.models.position import Position
from portfolio_risk_engine.domain.models.asset import Asset
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.domain.value_objects.currency import Currency
from portfolio_risk_engine.domain.value_objects.weight import Weight


def _make_position(ticker: str, weight: float) -> Position:
    return Position(
        asset=Asset(ticker=Ticker(ticker), currency=Currency("USD")),
        weight=Weight(weight),
    )


class TestPortfolioCreation:
    def test_single_position(self):
        pos = _make_position("AAPL", 1.0)
        portfolio = Portfolio(positions=(pos,))
        assert len(portfolio.positions) == 1

    def test_multiple_positions(self):
        positions = (
            _make_position("AAPL", 0.4),
            _make_position("MSFT", 0.3),
            _make_position("GOOG", 0.3),
        )
        portfolio = Portfolio(positions=positions)
        assert len(portfolio.positions) == 3


class TestPortfolioValidation:
    def test_empty_positions(self):
        with pytest.raises(ValueError, match="at least one position"):
            Portfolio(positions=())

    def test_duplicate_tickers(self):
        positions = (
            _make_position("AAPL", 0.5),
            _make_position("AAPL", 0.5),
        )
        with pytest.raises(ValueError, match="duplicate tickers"):
            Portfolio(positions=positions)

    def test_weights_not_summing_to_one(self):
        positions = (
            _make_position("AAPL", 0.3),
            _make_position("MSFT", 0.3),
        )
        with pytest.raises(ValueError, match="weights must sum to 1"):
            Portfolio(positions=positions)

    def test_weights_exceeding_one(self):
        positions = (
            _make_position("AAPL", 0.6),
            _make_position("MSFT", 0.6),
        )
        with pytest.raises(ValueError, match="weights must sum to 1"):
            Portfolio(positions=positions)

    def test_weights_within_tolerance(self):
        positions = (
            _make_position("AAPL", 1.0 / 3),
            _make_position("MSFT", 1.0 / 3),
            _make_position("GOOG", 1.0 / 3),
        )
        portfolio = Portfolio(positions=positions)
        assert len(portfolio.positions) == 3


class TestPortfolioProperties:
    def test_tickers(self):
        positions = (
            _make_position("AAPL", 0.6),
            _make_position("MSFT", 0.4),
        )
        portfolio = Portfolio(positions=positions)
        assert portfolio.tickers == [Ticker("AAPL"), Ticker("MSFT")]

    def test_weights(self):
        positions = (
            _make_position("AAPL", 0.6),
            _make_position("MSFT", 0.4),
        )
        portfolio = Portfolio(positions=positions)
        assert portfolio.weights == [0.6, 0.4]


class TestPortfolioImmutability:
    def test_frozen(self):
        pos = _make_position("AAPL", 1.0)
        portfolio = Portfolio(positions=(pos,))
        with pytest.raises(AttributeError):
            portfolio.positions = ()  # type: ignore[misc]
