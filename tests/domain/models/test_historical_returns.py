from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from portfolio_risk_engine.domain.models.historical_returns import HistoricalReturns
from portfolio_risk_engine.domain.value_objects.ticker import Ticker

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")
DATES = (date(2024, 1, 2), date(2024, 1, 3))


class TestHistoricalReturnsCreation:
    def test_single_ticker(self):
        hr = HistoricalReturns(
            tickers=(AAPL,),
            dates=DATES,
            returns_by_ticker={AAPL: (0.05, -0.02)},
        )
        assert hr.tickers == (AAPL,)
        assert len(hr.dates) == 2

    def test_multiple_tickers(self):
        hr = HistoricalReturns(
            tickers=(AAPL, MSFT),
            dates=DATES,
            returns_by_ticker={
                AAPL: (0.05, -0.02),
                MSFT: (-0.01, 0.03),
            },
        )
        assert len(hr.tickers) == 2

    def test_negative_returns_allowed(self):
        hr = HistoricalReturns(
            tickers=(AAPL,),
            dates=(date(2024, 1, 2),),
            returns_by_ticker={AAPL: (-0.5,)},
        )
        assert hr.returns_by_ticker[AAPL] == (-0.5,)


class TestHistoricalReturnsValidation:
    def test_empty_tickers(self):
        with pytest.raises(ValueError, match="at least one ticker"):
            HistoricalReturns(tickers=(), dates=DATES, returns_by_ticker={})

    def test_empty_dates(self):
        with pytest.raises(ValueError, match="at least one date"):
            HistoricalReturns(tickers=(AAPL,), dates=(), returns_by_ticker={AAPL: ()})

    def test_duplicate_dates(self):
        with pytest.raises(ValueError, match="Dates must be unique"):
            HistoricalReturns(
                tickers=(AAPL,),
                dates=(date(2024, 1, 2), date(2024, 1, 2)),
                returns_by_ticker={AAPL: (0.01, 0.02)},
            )

    def test_unsorted_dates(self):
        with pytest.raises(ValueError, match="chronological order"):
            HistoricalReturns(
                tickers=(AAPL,),
                dates=(date(2024, 1, 3), date(2024, 1, 2)),
                returns_by_ticker={AAPL: (0.01, 0.02)},
            )

    def test_keys_mismatch(self):
        with pytest.raises(ValueError, match="keys must match tickers"):
            HistoricalReturns(
                tickers=(AAPL, MSFT),
                dates=DATES,
                returns_by_ticker={AAPL: (0.01, 0.02)},
            )

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            HistoricalReturns(
                tickers=(AAPL,),
                dates=DATES,
                returns_by_ticker={AAPL: (0.01,)},
            )


class TestHistoricalReturnsImmutability:
    def test_frozen(self):
        hr = HistoricalReturns(
            tickers=(AAPL,),
            dates=(date(2024, 1, 2),),
            returns_by_ticker={AAPL: (0.01,)},
        )
        with pytest.raises(FrozenInstanceError):
            hr.tickers = ()  # type: ignore[misc]
