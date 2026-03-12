from datetime import date, timedelta

import numpy as np
import pytest

from portfolio_risk_engine.application.use_cases.estimate_student_t_parameters import (
    EstimateStudentTParameters,
)
from portfolio_risk_engine.domain.models.historical_returns import HistoricalReturns
from portfolio_risk_engine.domain.value_objects.ticker import Ticker


def _make_returns(values: list[float], ticker: str = "AAPL") -> HistoricalReturns:
    n = len(values)
    dates = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(n))
    t = Ticker(ticker)
    return HistoricalReturns(
        tickers=(t,),
        dates=dates,
        returns_by_ticker={t: tuple(values)},
    )


class TestEstimateStudentTParameters:
    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            returns = _make_returns([0.01, 0.02, -0.01])
            EstimateStudentTParameters().execute(returns)

    def test_gaussian_returns_high_df(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.standard_normal(500).tolist()
        returns = _make_returns(values)
        nu = EstimateStudentTParameters().execute(returns)
        # Gaussian has zero excess kurtosis -> should return 30.0 or high value
        assert nu >= 10.0

    def test_fat_tailed_returns_low_df(self) -> None:
        rng = np.random.default_rng(42)
        # Student-t with df=4 has heavy tails (excess kurtosis = 6/(4-4) -> undefined,
        # but with df=5 excess_kurtosis = 6/(5-4) = 6)
        values = (rng.standard_t(df=5, size=2000) * 0.01).tolist()
        returns = _make_returns(values)
        nu = EstimateStudentTParameters().execute(returns)
        # Should detect fat tails and return relatively low df
        assert nu < 20.0
        assert nu > 2.5

    def test_result_is_clamped_above_2_5(self) -> None:
        # Extremely fat-tailed data
        rng = np.random.default_rng(42)
        values = (rng.standard_t(df=3, size=5000) * 0.01).tolist()
        returns = _make_returns(values)
        nu = EstimateStudentTParameters().execute(returns)
        assert nu >= 2.5

    def test_result_is_clamped_below_100(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.standard_normal(500).tolist()
        returns = _make_returns(values)
        nu = EstimateStudentTParameters().execute(returns)
        assert nu <= 100.0
