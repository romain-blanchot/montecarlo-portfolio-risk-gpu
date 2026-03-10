import math
from datetime import date

import pytest

from portfolio_risk_engine.application.use_cases.compute_log_returns import (
    ComputeLogReturns,
)
from portfolio_risk_engine.domain.models.historical_prices import HistoricalPrices
from portfolio_risk_engine.domain.value_objects.ticker import Ticker

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")
DATES = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))


def _make_prices(**kwargs: object) -> HistoricalPrices:
    defaults: dict[str, object] = {
        "tickers": (AAPL,),
        "dates": DATES,
        "prices_by_ticker": {AAPL: (100.0, 110.0, 121.0)},
    }
    defaults.update(kwargs)
    return HistoricalPrices(**defaults)  # type: ignore[arg-type]


class TestComputeLogReturnsSingleTicker:
    def test_computes_log_returns(self):
        use_case = ComputeLogReturns()
        prices = _make_prices()

        result = use_case.execute(prices)

        expected = (math.log(110.0 / 100.0), math.log(121.0 / 110.0))
        assert result.returns_by_ticker[AAPL] == pytest.approx(expected)

    def test_dates_start_from_second(self):
        use_case = ComputeLogReturns()
        prices = _make_prices()

        result = use_case.execute(prices)

        assert result.dates == (date(2024, 1, 3), date(2024, 1, 4))

    def test_tickers_preserved(self):
        use_case = ComputeLogReturns()
        prices = _make_prices()

        result = use_case.execute(prices)

        assert result.tickers == (AAPL,)


class TestComputeLogReturnsMultipleTickers:
    def test_computes_all_tickers(self):
        use_case = ComputeLogReturns()
        prices = _make_prices(
            tickers=(AAPL, MSFT),
            prices_by_ticker={
                AAPL: (100.0, 110.0, 121.0),
                MSFT: (200.0, 210.0, 220.5),
            },
        )

        result = use_case.execute(prices)

        assert result.returns_by_ticker[AAPL] == pytest.approx(
            (math.log(1.1), math.log(1.1))
        )
        assert result.returns_by_ticker[MSFT] == pytest.approx(
            (math.log(210.0 / 200.0), math.log(220.5 / 210.0))
        )

    def test_ticker_order_preserved(self):
        use_case = ComputeLogReturns()
        prices = _make_prices(
            tickers=(MSFT, AAPL),
            prices_by_ticker={
                MSFT: (200.0, 210.0, 220.0),
                AAPL: (100.0, 110.0, 121.0),
            },
        )

        result = use_case.execute(prices)

        assert result.tickers == (MSFT, AAPL)


class TestComputeLogReturnsExactTwoPrices:
    def test_minimal_input(self):
        use_case = ComputeLogReturns()
        prices = _make_prices(
            dates=(date(2024, 1, 2), date(2024, 1, 3)),
            prices_by_ticker={AAPL: (100.0, 105.0)},
        )

        result = use_case.execute(prices)

        assert len(result.dates) == 1
        assert result.returns_by_ticker[AAPL] == pytest.approx((math.log(1.05),))


class TestComputeLogReturnsValidation:
    def test_single_price_raises(self):
        use_case = ComputeLogReturns()
        prices = _make_prices(
            dates=(date(2024, 1, 2),),
            prices_by_ticker={AAPL: (100.0,)},
        )

        with pytest.raises(ValueError, match="at least 2 prices"):
            use_case.execute(prices)
