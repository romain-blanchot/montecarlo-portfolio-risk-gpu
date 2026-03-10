from datetime import date, timedelta

import pytest

from portfolio_risk_engine.application.use_cases.estimate_market_parameters import (
    EstimateMarketParameters,
)
from portfolio_risk_engine.domain.models.historical_returns import HistoricalReturns
from portfolio_risk_engine.domain.value_objects.ticker import Ticker

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")


def _daily_dates(n: int) -> tuple[date, ...]:
    return tuple(date(2024, 1, 2) + timedelta(days=i) for i in range(n))


def _weekly_dates(n: int) -> tuple[date, ...]:
    return tuple(date(2024, 1, 1) + timedelta(weeks=i) for i in range(n))


def _monthly_dates(n: int) -> tuple[date, ...]:
    return tuple(date(2024, m, 1) for m in range(1, n + 1))


def _make_returns(
    tickers: tuple[Ticker, ...],
    dates: tuple[date, ...],
    returns_by_ticker: dict[Ticker, tuple[float, ...]],
) -> HistoricalReturns:
    return HistoricalReturns(
        tickers=tickers,
        dates=dates,
        returns_by_ticker=returns_by_ticker,
    )


class TestEstimateSingleTicker:
    def test_drift_is_annualized_mean(self):
        use_case = EstimateMarketParameters()
        returns = _make_returns(
            tickers=(AAPL,),
            dates=_daily_dates(3),
            returns_by_ticker={AAPL: (0.01, 0.03, 0.02)},
        )

        result = use_case.execute(returns)

        expected_mean = (0.01 + 0.03 + 0.02) / 3
        assert result.drift_vector[0] == pytest.approx(expected_mean * 252)

    def test_covariance_is_annualized_variance(self):
        use_case = EstimateMarketParameters()
        returns = _make_returns(
            tickers=(AAPL,),
            dates=_daily_dates(3),
            returns_by_ticker={AAPL: (0.01, 0.03, 0.02)},
        )

        result = use_case.execute(returns)

        # Variance with ddof=1
        mean = (0.01 + 0.03 + 0.02) / 3
        var = ((0.01 - mean) ** 2 + (0.03 - mean) ** 2 + (0.02 - mean) ** 2) / 2
        assert result.covariance_matrix[0][0] == pytest.approx(var * 252)

    def test_covariance_matrix_shape(self):
        use_case = EstimateMarketParameters()
        returns = _make_returns(
            tickers=(AAPL,),
            dates=_daily_dates(3),
            returns_by_ticker={AAPL: (0.01, 0.03, 0.02)},
        )

        result = use_case.execute(returns)

        assert len(result.covariance_matrix) == 1
        assert len(result.covariance_matrix[0]) == 1


class TestEstimateMultipleTickers:
    def test_drift_vector(self):
        use_case = EstimateMarketParameters()
        returns = _make_returns(
            tickers=(AAPL, MSFT),
            dates=_daily_dates(2),
            returns_by_ticker={
                AAPL: (0.01, 0.03),
                MSFT: (0.04, 0.02),
            },
        )

        result = use_case.execute(returns)

        assert result.drift_vector[0] == pytest.approx(0.02 * 252)
        assert result.drift_vector[1] == pytest.approx(0.03 * 252)

    def test_covariance_matrix_shape(self):
        use_case = EstimateMarketParameters()
        returns = _make_returns(
            tickers=(AAPL, MSFT),
            dates=_daily_dates(2),
            returns_by_ticker={
                AAPL: (0.01, 0.03),
                MSFT: (0.04, 0.02),
            },
        )

        result = use_case.execute(returns)

        assert len(result.covariance_matrix) == 2
        assert len(result.covariance_matrix[0]) == 2
        assert len(result.covariance_matrix[1]) == 2

    def test_covariance_values(self):
        use_case = EstimateMarketParameters()
        returns = _make_returns(
            tickers=(AAPL, MSFT),
            dates=_daily_dates(2),
            returns_by_ticker={
                AAPL: (0.01, 0.03),
                MSFT: (0.04, 0.02),
            },
        )

        result = use_case.execute(returns)

        # var(AAPL) ddof=1 = 0.0002
        assert result.covariance_matrix[0][0] == pytest.approx(0.0002 * 252)
        # cov(AAPL, MSFT) = -0.0002 (negatively correlated)
        assert result.covariance_matrix[0][1] == pytest.approx(-0.0002 * 252)
        assert result.covariance_matrix[1][0] == pytest.approx(-0.0002 * 252)
        # var(MSFT) ddof=1 = 0.0002
        assert result.covariance_matrix[1][1] == pytest.approx(0.0002 * 252)

    def test_ticker_order_preserved(self):
        use_case = EstimateMarketParameters()
        returns = _make_returns(
            tickers=(MSFT, AAPL),
            dates=_daily_dates(2),
            returns_by_ticker={
                MSFT: (0.04, 0.02),
                AAPL: (0.01, 0.03),
            },
        )

        result = use_case.execute(returns)

        assert result.tickers == (MSFT, AAPL)


class TestFrequencyEstimation:
    def test_daily_returns_252(self):
        use_case = EstimateMarketParameters()
        returns = _make_returns(
            tickers=(AAPL,),
            dates=_daily_dates(5),
            returns_by_ticker={AAPL: (0.01, 0.02, 0.01, 0.03, -0.01)},
        )

        result = use_case.execute(returns)

        assert result.annualization_factor == 252

    def test_weekly_returns_52(self):
        use_case = EstimateMarketParameters()
        returns = _make_returns(
            tickers=(AAPL,),
            dates=_weekly_dates(5),
            returns_by_ticker={AAPL: (0.01, 0.02, 0.01, 0.03, -0.01)},
        )

        result = use_case.execute(returns)

        assert result.annualization_factor == 52

    def test_monthly_returns_12(self):
        use_case = EstimateMarketParameters()
        returns = _make_returns(
            tickers=(AAPL,),
            dates=_monthly_dates(5),
            returns_by_ticker={AAPL: (0.01, 0.02, 0.01, 0.03, -0.01)},
        )

        result = use_case.execute(returns)

        assert result.annualization_factor == 12


class TestEstimateValidation:
    def test_single_observation_raises(self):
        use_case = EstimateMarketParameters()
        returns = _make_returns(
            tickers=(AAPL,),
            dates=(date(2024, 1, 2),),
            returns_by_ticker={AAPL: (0.01,)},
        )

        with pytest.raises(ValueError, match="at least 2 return observations"):
            use_case.execute(returns)
