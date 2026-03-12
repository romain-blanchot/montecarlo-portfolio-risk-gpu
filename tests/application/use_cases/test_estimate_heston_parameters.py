from datetime import date, timedelta

import numpy as np

from portfolio_risk_engine.application.use_cases.estimate_heston_parameters import (
    EstimateHestonParameters,
)
from portfolio_risk_engine.domain.models.historical_returns import HistoricalReturns
from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.value_objects.ticker import Ticker


def _make_single_asset_data(
    n: int = 200, seed: int = 42
) -> tuple[HistoricalReturns, MarketParameters]:
    rng = np.random.default_rng(seed)
    r = (rng.standard_normal(n) * 0.015).tolist()
    t = Ticker("AAPL")
    dates = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(n))

    returns = HistoricalReturns(
        tickers=(t,),
        dates=dates,
        returns_by_ticker={t: tuple(r)},
    )
    market_params = MarketParameters(
        tickers=(t,),
        drift_vector=(0.10,),
        covariance_matrix=((0.04,),),
        annualization_factor=252,
    )
    return returns, market_params


def _make_two_asset_data(
    n: int = 200, seed: int = 42
) -> tuple[HistoricalReturns, MarketParameters]:
    rng = np.random.default_rng(seed)
    r1 = (rng.standard_normal(n) * 0.015).tolist()
    r2 = (rng.standard_normal(n) * 0.012).tolist()
    t1, t2 = Ticker("AAPL"), Ticker("MSFT")
    dates = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(n))

    returns = HistoricalReturns(
        tickers=(t1, t2),
        dates=dates,
        returns_by_ticker={t1: tuple(r1), t2: tuple(r2)},
    )
    market_params = MarketParameters(
        tickers=(t1, t2),
        drift_vector=(0.10, 0.12),
        covariance_matrix=((0.04, 0.01), (0.01, 0.05)),
        annualization_factor=252,
    )
    return returns, market_params


class TestEstimateHestonParameters:
    def test_single_asset_returns_valid_model(self) -> None:
        returns, market_params = _make_single_asset_data()
        model = EstimateHestonParameters().execute(returns, market_params)

        assert len(model.tickers) == 1
        assert len(model.asset_params) == 1

        p = model.asset_params[0]
        assert p.kappa > 0
        assert p.theta > 0
        assert p.xi > 0
        assert -1 < p.rho < 1
        assert p.v0 > 0

    def test_two_asset_returns_valid_model(self) -> None:
        returns, market_params = _make_two_asset_data()
        model = EstimateHestonParameters().execute(returns, market_params)

        assert len(model.tickers) == 2
        assert len(model.asset_params) == 2
        assert len(model.correlation_cholesky) == 2
        assert len(model.correlation_cholesky[0]) == 2

    def test_short_data_returns_defaults(self) -> None:
        rng = np.random.default_rng(42)
        # Use very few data points to trigger the short-data branch
        # window = min(21, max(n//3, 5)) = 5, need n < window + 5 = 10
        r = (rng.standard_normal(8) * 0.015).tolist()
        t = Ticker("AAPL")
        dates = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(8))

        returns = HistoricalReturns(
            tickers=(t,),
            dates=dates,
            returns_by_ticker={t: tuple(r)},
        )
        market_params = MarketParameters(
            tickers=(t,),
            drift_vector=(0.10,),
            covariance_matrix=((0.04,),),
            annualization_factor=252,
        )

        model = EstimateHestonParameters().execute(returns, market_params)
        p = model.asset_params[0]
        # With short data, should use defaults
        assert p.kappa == 1.0
        assert p.rho == -0.5

    def test_model_preserves_drift(self) -> None:
        returns, market_params = _make_single_asset_data()
        model = EstimateHestonParameters().execute(returns, market_params)
        assert model.drift_vector == market_params.drift_vector

    def test_model_preserves_annualization_factor(self) -> None:
        returns, market_params = _make_single_asset_data()
        model = EstimateHestonParameters().execute(returns, market_params)
        assert model.annualization_factor == 252

    def test_custom_variance_window(self) -> None:
        returns, market_params = _make_single_asset_data()
        model = EstimateHestonParameters(variance_window=10).execute(
            returns, market_params
        )
        assert model.asset_params[0].kappa > 0
