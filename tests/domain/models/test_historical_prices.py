from datetime import date

import pytest

from portfolio_risk_engine.domain.models.historical_prices import HistoricalPrices
from portfolio_risk_engine.domain.value_objects.ticker import Ticker


AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")

DATES = (date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3))


def _make_prices(**kwargs: object) -> HistoricalPrices:
    defaults: dict[str, object] = {
        "tickers": (AAPL,),
        "dates": DATES,
        "prices_by_ticker": {AAPL: (150.0, 151.0, 152.0)},
    }
    defaults.update(kwargs)
    return HistoricalPrices(**defaults)  # type: ignore[arg-type]


class TestHistoricalPricesCreation:
    def test_single_ticker(self):
        hp = _make_prices()
        assert hp.tickers == (AAPL,)
        assert len(hp.dates) == 3

    def test_multiple_tickers(self):
        hp = _make_prices(
            tickers=(AAPL, MSFT),
            prices_by_ticker={
                AAPL: (150.0, 151.0, 152.0),
                MSFT: (300.0, 301.0, 302.0),
            },
        )
        assert len(hp.tickers) == 2


class TestHistoricalPricesValidation:
    def test_empty_tickers(self):
        with pytest.raises(ValueError, match="at least one ticker"):
            _make_prices(tickers=(), prices_by_ticker={})

    def test_empty_dates(self):
        with pytest.raises(ValueError, match="at least one date"):
            _make_prices(dates=(), prices_by_ticker={AAPL: ()})

    def test_duplicate_dates(self):
        with pytest.raises(ValueError, match="Dates must be unique"):
            _make_prices(
                dates=(date(2024, 1, 1), date(2024, 1, 1)),
                prices_by_ticker={AAPL: (150.0, 151.0)},
            )

    def test_dates_not_sorted(self):
        with pytest.raises(ValueError, match="chronological order"):
            _make_prices(
                dates=(date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)),
                prices_by_ticker={AAPL: (150.0, 151.0, 152.0)},
            )

    def test_keys_mismatch(self):
        with pytest.raises(ValueError, match="keys must match tickers"):
            _make_prices(
                tickers=(AAPL, MSFT),
                prices_by_ticker={AAPL: (150.0, 151.0, 152.0)},
            )

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            _make_prices(prices_by_ticker={AAPL: (150.0, 151.0)})

    def test_zero_price(self):
        with pytest.raises(ValueError, match="strictly positive"):
            _make_prices(prices_by_ticker={AAPL: (150.0, 0.0, 152.0)})

    def test_negative_price(self):
        with pytest.raises(ValueError, match="strictly positive"):
            _make_prices(prices_by_ticker={AAPL: (150.0, -1.0, 152.0)})


class TestHistoricalPricesImmutability:
    def test_frozen(self):
        hp = _make_prices()
        with pytest.raises(AttributeError):
            hp.tickers = ()  # type: ignore[misc]
