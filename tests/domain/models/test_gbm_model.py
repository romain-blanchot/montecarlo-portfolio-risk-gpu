from dataclasses import FrozenInstanceError

import pytest

from portfolio_risk_engine.domain.models.gbm_model import MultivariateGBM
from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.value_objects.ticker import Ticker

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")


def _make_params(n: int = 1) -> MarketParameters:
    if n == 1:
        return MarketParameters(
            tickers=(AAPL,),
            drift_vector=(0.10,),
            covariance_matrix=((0.04,),),
            annualization_factor=252,
        )
    return MarketParameters(
        tickers=(AAPL, MSFT),
        drift_vector=(0.10, 0.12),
        covariance_matrix=((0.04, 0.01), (0.01, 0.06)),
        annualization_factor=252,
    )


class TestMultivariateGBMCreation:
    def test_single_ticker(self):
        params = _make_params(1)
        model = MultivariateGBM(
            market_parameters=params,
            cholesky_factor=((0.2,),),
        )
        assert model.market_parameters == params
        assert model.cholesky_factor == ((0.2,),)

    def test_multi_ticker(self):
        params = _make_params(2)
        model = MultivariateGBM(
            market_parameters=params,
            cholesky_factor=((0.2, 0.0), (0.05, 0.24)),
        )
        assert len(model.cholesky_factor) == 2


class TestMultivariateGBMValidation:
    def test_cholesky_row_count_mismatch_raises(self):
        params = _make_params(2)
        with pytest.raises(ValueError, match="cholesky_factor rows"):
            MultivariateGBM(
                market_parameters=params,
                cholesky_factor=((0.2,),),
            )

    def test_cholesky_column_count_mismatch_raises(self):
        params = _make_params(2)
        with pytest.raises(ValueError, match="columns, expected 2"):
            MultivariateGBM(
                market_parameters=params,
                cholesky_factor=((0.2, 0.0, 0.0), (0.05, 0.24, 0.0)),
            )


class TestMultivariateGBMImmutability:
    def test_frozen(self):
        params = _make_params(1)
        model = MultivariateGBM(
            market_parameters=params,
            cholesky_factor=((0.2,),),
        )
        with pytest.raises(FrozenInstanceError):
            model.cholesky_factor = ()  # type: ignore[misc]
